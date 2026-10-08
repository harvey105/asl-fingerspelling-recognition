"""
Transfer learning classifier utilizing torchvision pretrained vision backbones.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)


class TransferCNN(nn.Module):
    """
    Transfer learning network adapting pretrained models to 36-class fingerspelling recognition.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        num_classes: int = 36,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name in ("mobilenet", "mobilenet_v3_small"):
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = mobilenet_v3_small(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes),
            )
            self.model = base
        elif self.backbone_name in ("resnet", "resnet18"):
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freeze feature extraction layers."""
        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "fc"):
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
