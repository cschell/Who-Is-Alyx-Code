import torch
import torchmetrics


class MinAccuracy(torchmetrics.classification.MulticlassAccuracy):
    def __init__(self, **kwargs):
        kwargs["average"] = "none"

        super().__init__(**kwargs)

    def compute(self) -> torch.Tensor:
        accuracies = super().compute()

        return accuracies.min()
