from typing import Dict, Any, List

import pytorch_lightning as pl
import torch
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN
from torch.nn import TripletMarginWithDistanceLoss
from pytorch_metric_learning.utils.inference import FaissKNN
from torch import nn as nn
from torchmetrics import AveragePrecision, AUROC
import numpy as np
from time import time

from src.custom_metrics.accuracy_calculator import MotionAccuracyCalculator
from src.metrics import DatasetPurpose
from src.utils.kl_distance import KLDivergenceDistance, kl_divergence_two_matrix_embeddings, kl_divergence_variant_2


class RetrievalGaussianModule(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer_options: Dict = None, loss_options: Dict[str, Any] = None,
                 labels: List[str] = None, subject_tuple_length: int = 3, **kwargs):
        super().__init__()
        self.best_logged_metrics = {}
        self.labels = labels
        self.optimizer_options = {} if optimizer_options is None else optimizer_options
        self.batch_tuning_mode = False
        self.num_classes = len(labels)
        self.model = model
        self.variance_activation = nn.ELU()
        assert model.num_out_classes % 2 == 0, "Model has to have an even number of outputs"
        self.distance = KLDivergenceDistance()
        self._init_miner(loss_options)
        self.cuda_device = torch.device("cuda:0")
        # self.faiss_evaluator = AccuracyCalculator(k="max_bin_count", device=torch.device("cpu"))
        self.faiss_evaluator = MotionAccuracyCalculator(sequence_lengths_minutes=[5, 10, 15],
                                                        sliding_window_step_size_seconds=1,
                                                        k="max_bin_count",
                                                        device=torch.device("cpu"))
        self.kl_divergence_evaluator = AccuracyCalculator(
            # sequence_lengths_minutes=[5, 10, 15],
            # sliding_window_step_size_seconds=1,
            knn_func=CustomKNN(self.distance, batch_size=600),
            k="max_bin_count",
            device=torch.device("cpu")
        )
        self.subject_tuple_length = subject_tuple_length
        self.save_hyperparameters()

    def _init_miner(self, loss_options):
        try:
            margin = loss_options["margin"]
            type_of_triplets = loss_options["type_of_triplets"]
        except AttributeError:
            margin = 0.2
            type_of_triplets = "hard"
        self.loss_func = TripletMarginWithDistanceLoss(
            margin=margin,
            distance_function=kl_divergence_variant_2,
            reduction="mean"
        )
        self.miner = TripletMarginMiner(margin=margin,
                                        type_of_triplets=type_of_triplets,
                                        distance=self.distance)
        self.easy_miner = TripletMarginMiner(margin=margin,
                                             type_of_triplets="easy",
                                             distance=self.distance)

    def configure_optimizers(self) -> Any:
        optimizer_name = self.optimizer_options.pop("name", "Adam")
        optimizer = getattr(torch.optim, optimizer_name)(params=self.model.parameters(), **self.optimizer_options)

        return optimizer

    def forward(self, X):
        output = self.model.forward(X.float())
        cut = int(output.size(1) / 2)
        means = output[:, :cut]
        variance = self.variance_activation(output[:, cut:]) + 1
        return torch.cat([means, variance], 1)

    def training_step(self, batch, batch_idx):
        X = batch["data"]
        y = batch["targets"]
        embeddings = self.forward(X)

        anchor, positive, negative = self.miner(embeddings, y.long())
        total_loss = self.loss_func(
            embeddings[anchor],
            embeddings[positive],
            embeddings[negative]
        )

        # Following code can be used instead of previous two lines, which acts like a batch in a batch,
        # slows down speed of one epoch, but trains more during one epoch

        # possible_combinations = torch.combinations(torch.tensor(self.labels), r=self.subject_tuple_length)
        # if y.get_device() != -1:
        #     possible_combinations = possible_combinations.to(device=y.get_device())
        # loss_sum = 0
        # non_processed = 0
        # for idx, combination in enumerate(possible_combinations):
        #     index_filter = torch.isin(y, combination)
        #     y_comb = y[index_filter]
        #     embeddings_comb = embeddings[index_filter]
        #     anchor, positive, negative = self.miner(embeddings_comb, y_comb.long())
        #     a_e, p_e, n_e = self.easy_miner(embeddings_comb, y_comb.long())
        #     if len(anchor) == 0 or len(positive) == 0 or len(negative) == 0:
        #         non_processed += 1
        #         continue
        #     loss = self.loss_func(
        #         embeddings_comb[anchor],
        #         embeddings_comb[positive],
        #         embeddings_comb[negative]
        #     )
        #     loss_sum = loss_sum + loss
        #
        # if (len(possible_combinations) - non_processed) != 0:
        #     total_loss = loss_sum / (len(possible_combinations) - non_processed)
        # else:
        #     total_loss = 0

        self.log(f"loss_{DatasetPurpose.TRAIN.value}", total_loss, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, _batch_idx):
        X = batch["data"]
        y = batch["targets"]
        session_idxs = batch["session_idx"]

        h = self.forward(X)
        return h, y, session_idxs

    def predict_step(self, batch, batch_idx, **kwargs):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        if self.batch_tuning_mode:
            return
        embeddings = torch.cat([emb for emb, _, _ in validation_step_outputs])
        y = torch.cat([y for _, y, _ in validation_step_outputs])
        session_idxs = torch.cat([session_idxs for _, _, session_idxs in validation_step_outputs])

        self._compute_and_log_validation_metrics(embeddings, y, session_idxs)
        self._note_best_metric_values()

    def _compute_and_log_validation_metrics(self, embeddings: torch.Tensor, y: torch.Tensor, session_idxs: torch.Tensor):
        torch.set_num_threads(8)

        session_1_embeddings = embeddings[session_idxs == 0]
        session_1_y = y[session_idxs == 0]
        session_2_embeddings = embeddings[session_idxs == 1]
        session_2_y = y[session_idxs == 1]

        step_width_references = 150
        step_width_queries = 1
        reference_embeddings = session_1_embeddings[::step_width_references].contiguous()
        reference_y = session_1_y[::step_width_references].contiguous()
        query_embeddings = session_2_embeddings[::step_width_queries].contiguous()
        query_y = session_2_y[::step_width_queries].contiguous()
        cut = int(embeddings.size(1) / 2)

        if query_embeddings.any() and reference_embeddings.any():
            with torch.no_grad():
                # with torch.cuda.device(-1):
                q_e = query_embeddings[:, :cut]
                r_e = reference_embeddings[:, :cut]
                print("Performing FaissKNN evaluation")
                start_faiss = time()
                accuracy_faiss = self.faiss_evaluator.get_accuracy(
                    q_e,
                    r_e,
                    query_y,
                    reference_y,
                    embeddings_come_from_same_source=False
                )
                end_faiss = time()
                print(f"Done in <{end_faiss - start_faiss:.3f}> ms")
                # print("Performing KLDivKNN evaluation", (query_embeddings.size(0), reference_embeddings.size(0)))
                # start_kl_divergence = time()
                #
                # accuracy_kl_divergence = self.kl_divergence_evaluator.get_accuracy(
                #     query_embeddings,
                #     reference_embeddings,
                #     query_y,
                #     reference_y,
                #     embeddings_come_from_same_source=False
                # )
                # end_kl_divergence = time()
                # print(f"Done in <{end_kl_divergence - start_kl_divergence:.6f}> ms")

            self.log_metrics(accuracy_faiss, "validation")
            self.log_metrics(accuracy_faiss, "validation/mean")
            # self.log_metrics(accuracy_kl_divergence, "validation/kl_divergence")

    def _note_best_metric_values(self):
        for metric_name, value in self.trainer.logged_metrics.items():
            old_min_value = self.best_logged_metrics.get((metric_name, "min"), np.inf)
            old_max_value = self.best_logged_metrics.get((metric_name, "max"), -np.inf)

            self.best_logged_metrics[(metric_name, "min")] = min(old_min_value,
                                                                 self.trainer.logged_metrics[metric_name])
            self.best_logged_metrics[(metric_name, "max")] = max(old_max_value,
                                                                 self.trainer.logged_metrics[metric_name])

    def log_metrics(self, metrics: dict, stage: str):
        for key, item in metrics.items():
            if isinstance(item, dict):
                self.log_metrics(item, stage)
            else:
                name = f"{key}/{stage}"
                self.log(name, item)


    # def training_step(self, batch, batch_idx):


