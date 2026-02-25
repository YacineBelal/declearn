from torch import nn


class ECGResNet(nn.Module):
    def __init__(
        self,
        n_features_in=1000,
        n_channels_in=12,
        num_classes=5,
        kernel_size=16,
        num_resBlk=4,
        n_channels_out_n=[64, 128, 192, 256, 320],
        downsample=4,
        drop_out=0.5,
    ):
        if not isinstance(n_channels_out_n, list):
            raise TypeError("n_channels_out must be a list")
        if len(n_channels_out_n) == 0:
            raise ValueError("n_channels_out cannot be empty")
        if len(n_channels_out_n) != num_resBlk + 1:
            raise ValueError(
                "n_channels_out size must match residual blocks number + 1"
            )
        if (
            downsample != 0
            and n_features_in // downsample ** (num_resBlk - 1) < kernel_size
        ):
            raise ValueError(
                "Kernel size cannot be greater than downsampled feature size"
            )

        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(
                n_channels_in,
                n_channels_out_n[0],
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(n_channels_out_n[0]),
            nn.ReLU(),
        )
        self.resBlk = nn.ModuleList(
            [
                ResidualBlock(
                    n_channels_out_n[i],
                    n_channels_out_n[i + 1],
                    kernel_size=kernel_size,
                    downsample=downsample,
                    drop_out_prob=drop_out,
                )
                for i in range(num_resBlk)
            ]
        )

        # TODO switch to none lazy once protoype is ready:)
        self.dense = nn.LazyLinear(out_features=num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.first_conv[0].weight, nonlinearity="relu")
        nn.init.ones_(self.first_conv[1].weight)
        nn.init.zeros_(self.first_conv[1].bias)

    def forward(self, x):
        x = self.first_conv(x)
        y = x
        for resblk in self.resBlk:
            x, y = resblk([x, y])
        x = x.flatten(start_dim=1)
        out = self.dense(x)
        return out


model = ECGResNet()