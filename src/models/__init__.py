from typing import Any

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


def build_model(
    model_name: str, num_classes: int = 36, **kwargs: Any
) -> nn.Module:
    """Instantiate a model architecture by name or assignment identifier.

    Args:
        model_name: Architecture identifier:
            - 'simple_cnn', 'simple', 'model_1': Section III.2 baseline CNN.
            - 'complex_cnn', 'complex', 'model_2': Section III.3 deep residual CNN.
            - 'transfer_cnn', 'transfer', 'model_3', 'resnet18': Section III.4 transfer learning model.
        num_classes: Number of target classification categories (default 36).
        **kwargs: Additional model parameters (dropout, channel_list, pretrained, etc.).

    Returns:
        Instantiated PyTorch neural network module.

    Raises:
        ValueError: If model_name does not match any supported architecture.
    """
    name = model_name.strip().lower()
    if name in ("simple_cnn", "simple", "model_1"):
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif name in ("complex_cnn", "complex", "model_2"):
        return ComplexCNN(num_classes=num_classes, **kwargs)
    elif name in ("transfer_cnn", "transfer", "model_3", "resnet", "resnet18"):
        return TransferCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
