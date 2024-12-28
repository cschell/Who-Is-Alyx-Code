from torch import nn

class CNNLayer(nn.Module):
    def __init__(self,
                 kernel_size: int, out_channels: int,
                 conv_stride: int, max_pool_kernel_size: int, dropout: float,
                 activation: str):
        super().__init__()

        activation_klass = getattr(nn, activation)

        self.ops = nn.Sequential(
            nn.LazyConv1d(out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=conv_stride,
                            padding=1),
            activation_klass(),
            nn.LazyBatchNorm1d(),
            nn.MaxPool1d(kernel_size=max_pool_kernel_size),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.ops(x)



