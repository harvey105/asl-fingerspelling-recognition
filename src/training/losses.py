"""
Loss functions for model training and regularization.
"""

from typing import Optional
import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with optional label smoothing.

    When smoothing=0.0, standard cross-entropy is computed.
    When smoothing > 0.0, targets are smoothed uniformly across all classes.
    """

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in range [0.0, 1.0), got {smoothing}")
        self.smoothing = float(smoothing)
        self.weight = weight
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weight, label_smoothing=self.smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Predicted class logits of shape (N, C).
            targets: Ground-truth class indices of shape (N,).

        Returns:
            Scalar loss tensor.
        """
        return self.loss_fn(logits, targets)
