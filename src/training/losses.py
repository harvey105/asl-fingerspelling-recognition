import torch
from torch import nn


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with optional label smoothing.

    When smoothing is 0.0, computes standard categorical cross-entropy.
    When smoothing > 0.0, redistributes probability uniformly across classes.

    Args:
        smoothing: Label smoothing factor in [0.0, 1.0).
        weight: Optional class rebalancing weights tensor of shape (num_classes,).
    """

    def __init__(
        self, smoothing: float = 0.1, weight: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in range [0.0, 1.0), got {smoothing}")
        self.smoothing = float(smoothing)
        self.weight = weight
        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.weight, label_smoothing=self.smoothing
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss between unnormalized logits and ground-truth labels."""
        return self.loss_fn(logits, targets)
