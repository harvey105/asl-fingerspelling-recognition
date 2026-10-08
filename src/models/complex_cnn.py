"""
Deep residual convolutional neural network built from scratch with bottleneck/residual blocks.
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual block with optional projection shortcut: F(x) + x.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ComplexCNN(nn.Module):
    """
    Deep residual CNN with modular stages, residual skip connections F(x) + x,
    and configurable stage depths and channel capacities.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 36,
        stage_channels: list[int] | None = None,
        stage_depths: list[int] | None = None,
        dropout: float = 0.3,
        fc_dim: int = 256,
    ) -> None:
        super().__init__()
        if stage_channels is None:
            stage_channels = [32, 64, 128, 256]
        if stage_depths is None:
            stage_depths = [2, 2, 2, 2]

        self.stage_channels = stage_channels
        self.stage_depths = stage_depths
        self.dropout_rate = dropout
        self.fc_dim = fc_dim

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stage_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        stages: list[nn.Module] = []
        c_in = stage_channels[0]
        for idx, (c_out, depth) in enumerate(zip(stage_channels, stage_depths)):
            for b in range(depth):
                stride = 2 if (idx > 0 and b == 0) else 1
                stages.append(ResidualBlock(c_in, c_out, stride=stride))
                c_in = c_out

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(stage_channels[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stem, residual stages, and classifier."""
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.classifier(x)
