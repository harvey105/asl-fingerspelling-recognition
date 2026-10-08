"""
Neural network model architectures and factory interface.
"""

from typing import Any, Dict

from torch import nn

from .complex_cnn import ComplexCNN
from .simple_cnn import SimpleCNN
from .transfer_cnn import TransferCNN

__all__ = [
    "ComplexCNN",
    "SimpleCNN",
    "TransferCNN",
    "build_model",
]


def build_model(model_name: str, num_classes: int = 36, **kwargs: Any) -> nn.Module:
    """
    Factory function returning model instance given architecture identifier.

    Args:
        model_name: 'simple_cnn', 'complex_cnn', 'transfer_cnn' (or 'mobilenet', 'resnet18').
        num_classes: Number of output classification targets.
        **kwargs: Additional model kwargs (e.g. dropout, channel list, pretrained).
    """
    name = model_name.strip().lower()
    if name in ("simple_cnn", "simple", "model_1"):
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif name in ("complex_cnn", "complex", "model_2"):
        return ComplexCNN(num_classes=num_classes, **kwargs)
    elif name in ("transfer_cnn", "transfer", "mobilenet", "resnet18", "model_3"):
        backbone = kwargs.pop("backbone_name", name)
        if backbone in ("transfer_cnn", "transfer", "model_3"):
            backbone = "mobilenet_v3_small"
        return TransferCNN(backbone_name=backbone, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
