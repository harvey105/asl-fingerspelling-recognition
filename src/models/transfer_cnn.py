from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class TransferCNN(nn.Module):
    """Transfer learning classifier adapting pretrained ResNet-18 (Model 3).
    Replaces the default ImageNet classification head with an MLP head for 36-class gesture recognition.

    Args:
        num_classes: Number of target gesture classes (default 36).
        pretrained: If True, loads official ImageNet-1K pretrained weights.
        dropout: Dropout probability applied before classifier layers.
        fc_dim: Intermediate linear layer dimension in classifier head.
    """

    def __init__(
        self,
        num_classes: int = 36,
        pretrained: bool = True,
        dropout: float = 0.3,
        fc_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc_dim = fc_dim

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18 backbone and classifier head."""
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freeze feature extractor parameters, training only classifier head."""
        for name, param in self.model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters for full end-to-end fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict[str, Any]]:
        """Group parameters with differential learning rates for backbone vs head.

        Args:
            lr_backbone: Learning rate for pretrained feature extraction stages.
            lr_head: Learning rate for newly initialized classifier head layers.
            weight_decay: L2 regularization penalty.

        Returns:
            List of optimizer parameter group dictionaries.
        """
        head_params = []
        backbone_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name:
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
