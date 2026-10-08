"""
Transfer learning classifier utilizing torchvision pretrained vision backbones.
"""

from typing import Any, List, Optional
import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
)


class TransferCNN(nn.Module):
    """
    Transfer learning network adapting pretrained vision models to 36-class fingerspelling recognition.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 36,
        pretrained: bool = True,
        dropout: float = 0.3,
        fc_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower().replace("-", "_")
        self.num_classes = num_classes
        self.fc_dim = fc_dim

        if self.backbone_name in ("resnet18", "resnet"):
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            in_features = base.fc.in_features
            if fc_dim > 0:
                base.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, fc_dim),
                    nn.BatchNorm1d(fc_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(fc_dim, num_classes),
                )
            else:
                base.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, num_classes),
                )
            self.model = base

        elif self.backbone_name in ("mobilenet_v3_large", "mobilenet_large"):
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = mobilenet_v3_large(weights=weights)
            in_features = base.classifier[3].in_features
            base.classifier[3] = nn.Linear(in_features, num_classes)
            if dropout != 0.2:
                base.classifier[2] = nn.Dropout(p=dropout)
            self.model = base

        elif self.backbone_name in (
            "mobilenet_v3_small",
            "mobilenet_small",
            "mobilenet",
        ):
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = mobilenet_v3_small(weights=weights)
            in_features = base.classifier[3].in_features
            base.classifier[3] = nn.Linear(in_features, num_classes)
            if dropout != 0.2:
                base.classifier[2] = nn.Dropout(p=dropout)
            self.model = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freeze feature extraction layers, leaving only custom classifier head trainable."""
        for name, param in self.model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float
    ) -> List[dict[str, Any]]:
        """Return parameter groups with differential learning rates for backbone vs head."""
        head_params = []
        backbone_params = []
        head_keys = ("classifier", "fc")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in head_keys):
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
            {
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
            },
        ]
