from collections import defaultdict
from time import time
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn as nn

from src.metrics import DatasetPurpose, Metrics, initialize_metric
from src.utils import utils

log = utils.get_logger(__name__)


class ClassificationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_weights=None, optimizer_options: dict = None, metrics: List[dict] = None, *, labels: List[str] = None):
        super().__init__()
        self.labels = labels
        self.optimizer_options = {} if optimizer_options is None else optimizer_options
        self.hyperparameters = model.hparams.__dict__
        self.model = model
        self.save_hyperparameters()
        self.best_logged_metrics = defaultdict(lambda: np.nan)

        # the Lightning Trainer#tune expects "batch_size" to be set, even though we don't need it here
        self.batch_size = 1

        self._initialize_metrics([] if metrics is None else metrics)

        if loss_weights is not None:
            self.loss_weights = torch.from_numpy(loss_weights).float().to(self.device)
        else:
            self.loss_weights = None

    def _initialize_metrics(self, metrics):
        # dictionaries for all metrics are initialized as train_metric and validation_metric
        self.metrics = nn.ModuleDict({
            f"{DatasetPurpose.TRAIN.value}_metrics": nn.ModuleDict({}),
            f"{DatasetPurpose.VALIDATION.value}_metrics": nn.ModuleDict({}),
        })
        for metric in metrics:
            self.metrics[f"{metric['dataset']}_metrics"][metric["name"]] = initialize_metric(metric["name"], num_out_classes=self.model.num_out_classes)

    def forward(self, X):
        return self.model.forward(X)

    def training_step(self, batch, _batch_idx):
        X = batch["data"].to(self.dtype)
        y = batch["targets"].long()

        h = self.forward(X)

        loss = F.cross_entropy(h, y, weight=self.loss_weights.to(self.device).to(self.dtype)).mean()
        self.log(f"loss/{DatasetPurpose.TRAIN.value}", loss, on_step=False, on_epoch=True)

        for metric_name, metric_fn in self.metrics[f"{DatasetPurpose.TRAIN.value}_metrics"].items():
            self.log(f"{metric_name}/{DatasetPurpose.TRAIN.value}", metric_fn.cpu()(h.cpu(), y.cpu()), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer_options = self.optimizer_options.copy()
        optimizer_name = optimizer_options.pop("name", "Adam")
        optimizer = getattr(torch.optim, optimizer_name)(params=self.parameters(), **optimizer_options)

        return optimizer

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        X = batch["data"].to(self.dtype)
        y = batch["targets"].long()

        h = self.forward(X)
        return h, y
    
    @torch.no_grad()
    def validation_epoch_end(self, validation_step_outputs):
        # concat h and y from all steps
        h = torch.cat([h_ for h_, y_ in validation_step_outputs])
        y = torch.cat([y_ for h_, y_ in validation_step_outputs])
        dataset = self.trainer.val_dataloaders[0].dataset
        if hasattr(dataset, "loss_weights"):
            val_loss_weights = torch.from_numpy(dataset.loss_weights).float().to(self.device).to(self.dtype)
        else:
            val_loss_weights = None
        loss = F.cross_entropy(h, y, weight=val_loss_weights).mean()
        self.log(f"loss/{DatasetPurpose.VALIDATION.value}", loss, on_step=False, on_epoch=True, )
        preds = h.softmax(axis=1)

        torch.use_deterministic_algorithms(False)
        for metric_name, metric_fn in self.metrics[f"{DatasetPurpose.VALIDATION.value}_metrics"].items():
            self.log(f"{metric_name}/{DatasetPurpose.VALIDATION.value}", metric_fn(preds, y), on_step=False, on_epoch=True, prog_bar=metric_name == Metrics.MIN_ACCURACY.value)

        torch.use_deterministic_algorithms(True)
        self._note_best_metric_values()
        return loss

    def _note_best_metric_values(self):
        for metric_name, value in self.trainer.logged_metrics.items():
            old_min_value = self.best_logged_metrics.get((metric_name, "min"), np.inf)
            old_max_value = self.best_logged_metrics.get((metric_name, "max"), -np.inf)

            self.best_logged_metrics[(metric_name, "min")] = min(old_min_value, self.trainer.logged_metrics[metric_name])
            self.best_logged_metrics[(metric_name, "max")] = max(old_max_value, self.trainer.logged_metrics[metric_name])
