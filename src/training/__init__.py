"""
Training infrastructure package exporting losses, metrics, and execution engines.
"""

from .engine import Trainer, TrainerConfig, get_default_device
from .losses import LabelSmoothingCrossEntropy
from .metrics import MetricTracker, compute_metrics

__all__ = [
    "Trainer",
    "TrainerConfig",
    "get_default_device",
    "LabelSmoothingCrossEntropy",
    "compute_metrics",
    "MetricTracker",
]
