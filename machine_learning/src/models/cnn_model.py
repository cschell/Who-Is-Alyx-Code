import numpy as np
import torch
from torch import nn

from src.models._cnn_layer import CNNLayer
from src.hyperparameters.cnn_hyperparameters import CNNHyperparameters
from src.utils.normalization_helper import normalize_embedding

class CNNModel(nn.Module):
    def __init__(self, hyperparameters: CNNHyperparameters, num_features: int, window_size: int, num_out_classes: int,
                 **_kwargs):
        super().__init__()
        self.num_features = num_features
        self.num_out_classes = num_out_classes
        self.hparams = hyperparameters

        self.ops = nn.Sequential(
            *[CNNLayer(kernel_size=self.hparams.kernel_size,
                      out_channels=int(self.hparams.initial_channel_size * self.hparams.channels_factor**layer_idx),
                      conv_stride=self.hparams.conv_stride,
                      max_pool_kernel_size=self.hparams.max_pool_size,
                      dropout=self.hparams.dropout,
                      activation=self.hparams.activation,
                ) for layer_idx in range(self.hparams.num_layers)],
            nn.Flatten(),
            nn.LazyLinear(num_out_classes)
        )

        # required to init the lazy modules, otherwise PyTorch Lightning will raise an error during startup
        self.ops(torch.zeros((1, num_features, window_size)))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ops(x)

        if self.hparams.normalize_model_outputs:
            x = normalize_embedding(x)

        return x