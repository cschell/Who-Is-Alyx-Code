import enum

import torchmetrics

from src.custom_metrics.min_accuracy import MinAccuracy
from src.custom_metrics.num_correct_takes import NumCorrectTakes


class Metrics(str, enum.Enum):
    ACCURACY = ("accuracy", torchmetrics.Accuracy, lambda kwargs: dict(task="multiclass", num_classes=kwargs["num_out_classes"], average="macro"))
    MIN_ACCURACY = ("min_accuracy", MinAccuracy, lambda kwargs: dict(num_classes=kwargs["num_out_classes"]))
    F1 = ("f1", torchmetrics.F1Score, lambda kwargs: dict(task="multiclass", num_classes=kwargs["num_out_classes"]))
    PRECISION = ("precision", torchmetrics.Precision, lambda kwargs: dict(task="multiclass", num_classes=kwargs["num_out_classes"]))
    RECALL = ("recall", torchmetrics.Recall, lambda kwargs: dict(task="multiclass", num_classes=kwargs["num_out_classes"]))
    COHEN_KAPPA = ("cohen_kappa", torchmetrics.CohenKappa, lambda kwargs: dict(task="multiclass", num_classes=kwargs["num_out_classes"]))
    MATTHEWS = ("matthews_corrcoef", torchmetrics.MatthewsCorrCoef, lambda kwargs: dict(task="multiclass", num_classes=kwargs["num_out_classes"]))
    NUM_CORRECT_TAKES = ("num_correct_takes", NumCorrectTakes, lambda kwargs: dict(num_classes=kwargs["num_out_classes"]))

    def __new__(cls, name: str, metric_cls, kwargs_fn) -> "Metrics":
        obj = str.__new__(cls, name)
        obj._value_ = name

        obj.cls = metric_cls
        obj.kwargs_fn = kwargs_fn
        return obj

    def initialize(self, **kwargs):
        return self.cls(**self.kwargs_fn(kwargs))


class DatasetPurpose(enum.Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def get_metric_name(metric: Metrics, purpose: DatasetPurpose):
    return ".".join([metric.value, purpose.value])


def initialize_metric(metric_name: str, num_out_classes: int = None):
    return Metrics(metric_name).initialize(num_out_classes=num_out_classes)
