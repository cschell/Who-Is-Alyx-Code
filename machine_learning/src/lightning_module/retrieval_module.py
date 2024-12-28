from typing import List, Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import pytorch_metric_learning.losses
import wandb
from torch import nn as nn

from src.custom_metrics.accuracy_calculator import MotionAccuracyCalculator
from src.metrics import DatasetPurpose
from src.utils import utils

logger = utils.get_logger(__name__)


class RetrievalModule(pl.LightningModule):
    def __init__(self, model: nn.Module, metrics=None, loss_weights=None, optimizer_options: Dict = None, loss_options: Dict[str, Any] = None,
                 labels: List[str] = None):
        super().__init__()
        self.best_logged_metrics = {}
        self.labels = labels
        self.num_classes = len(labels)
        self.optimizer_options = {} if optimizer_options is None else optimizer_options
        self.hyperparameters = model.hparams
        self.model = model
        self.evaluator = MotionAccuracyCalculator(sequence_lengths_minutes=[5, 10, 15], sliding_window_step_size_seconds=1, k="max_bin_count", device=torch.device("cpu"))
        self.batch_tuning_mode = False
        self.loss_func = self._initialize_loss_function(loss_options)
        # wandb.watch(model, criterion=self.loss_func)
        if loss_weights is not None:
            self.loss_weights = torch.from_numpy(loss_weights).float().to(self.device)
        else:
            self.loss_weights = None
        self.save_hyperparameters()

    def _initialize_loss_function(self, loss_options):
        loss_options = loss_options.copy()
        loss_options: Dict[str, Any] = {"name": "NormalizedSoftmaxLoss", "num_classes": "auto", "embedding_size": "auto"} if loss_options is None else loss_options

        if loss_options.get("num_classes") == "auto":
            loss_options["num_classes"] = self.num_classes
        if loss_options.get("embedding_size") == "auto":
            loss_options["embedding_size"] = self.model.num_out_classes

        loss_func = getattr(pytorch_metric_learning.losses, loss_options.pop("name"))(**loss_options)
        return loss_func

    def forward(self, X):
        embeddings = self.model.forward(X.float())
        return embeddings

    def training_step(self, batch, batch_idx):
        X = batch["data"]
        y = batch["targets"]

        embedding = self.forward(X)
        loss = self.loss_func(embedding, y.long())
        self.log(f"loss_{DatasetPurpose.TRAIN.value}", loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer_name = self.optimizer_options.pop("name", "Adam")
        optimizer = getattr(torch.optim, optimizer_name)(params=self.model.parameters(), **self.optimizer_options)

        return optimizer

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

    def _compute_and_log_validation_metrics(self, embeddings, y, session_idxs):
        # speeds up cpu calculations
        torch.set_num_threads(8)

        session_1_embeddings = embeddings[session_idxs == 0].cpu()
        session_1_y = y[session_idxs == 0].cpu()
        session_2_embeddings = embeddings[session_idxs == 1].cpu()
        session_2_y = y[session_idxs == 1].cpu()

        reference_embeddings = session_1_embeddings[::150].contiguous()
        reference_y = session_1_y[::150].contiguous()
        query_embeddings = session_2_embeddings.contiguous()
        query_y = session_2_y.contiguous()

        if query_embeddings.any() and reference_embeddings.any():
            with torch.no_grad():
                with torch.cuda.device(-1):
                    accuracy = self.evaluator.get_accuracy(query_embeddings, reference_embeddings, query_y, reference_y,
                                                           embeddings_come_from_same_source=False)
            self.log_metrics(accuracy, "validation/mean")
            self.log_metrics(accuracy, "validation")

    def _note_best_metric_values(self):
        for metric_name, value in self.trainer.logged_metrics.items():
            old_min_value = self.best_logged_metrics.get((metric_name, "min"), np.inf)
            old_max_value = self.best_logged_metrics.get((metric_name, "max"), -np.inf)

            self.best_logged_metrics[(metric_name, "min")] = min(old_min_value, self.trainer.logged_metrics[metric_name])
            self.best_logged_metrics[(metric_name, "max")] = max(old_max_value, self.trainer.logged_metrics[metric_name])

    def log_metrics(self, metrics: dict, stage: str):
        """ Logs the metrics recursively. """
        for key, item in metrics.items():
            if not isinstance(item, dict):
                name = f"{key}/{stage}"
                self.log(name, item)
            else:
                self.log_metrics(item, stage)

    def count_total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
