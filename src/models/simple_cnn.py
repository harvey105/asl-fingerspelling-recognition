"""
Simple Convolutional Neural Network baseline built from Conv, Pool, and FC layers.
"""

from typing import List
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Standard sequential CNN architecture conforming to baseline rubric:
    Convolutional feature extractor followed by pooling and fully connected classifier.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 36,
        channel_list: List[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if channel_list is None:
            channel_list = [32, 64, 128, 256]

        layers: List[nn.Module] = []
        c_in = in_channels
        for c_out in channel_list:
            layers.extend(
                [
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
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
            nn.Linear(channel_list[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and classifier head."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
