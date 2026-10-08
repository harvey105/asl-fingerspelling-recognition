"""
Simple Convolutional Neural Network baseline built from Conv, Pool, and FC layers.
"""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """
    Standard sequential CNN architecture conforming to baseline rubric:
    Convolutional feature extractor followed by pooling and fully connected classifier.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 36,
        channel_list: list[int] | None = None,
        dropout: float = 0.3,
        fc_dim: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channel_list is None:
            channel_list = [32, 64, 128, 256]

        self.channel_list = channel_list
        self.fc_dim = fc_dim
        self.dropout_rate = dropout
        self.kernel_size = kernel_size

        layers: list[nn.Module] = []
        c_in = in_channels
        padding = kernel_size // 2
        for c_out in channel_list:
            layers.extend(
                [
                    nn.Conv2d(
                        c_in,
                        c_out,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            c_in = c_out

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channel_list[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and classifier head."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
