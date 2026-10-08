"""
Multi-class evaluation metrics calculation across classification classes.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def compute_metrics(
    y_true: Union[torch.Tensor, np.ndarray, List[int]],
    y_pred: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: Optional[int] = 36,
) -> Dict[str, float]:
    """
    Compute multi-class evaluation metrics: Top-1 Accuracy, Macro-Precision,
    Macro-Recall, and Macro-F1 score.

    Args:
        y_true: Ground truth target labels (1D array or tensor).
        y_pred: Predicted class labels or logits (1D or 2D array or tensor).
        num_classes: Optional total number of classes.

    Returns:
        Dictionary containing:
            - accuracy: Top-1 classification accuracy [0.0, 1.0]
            - precision_macro: Macro-averaged precision [0.0, 1.0]
            - recall_macro: Macro-averaged recall [0.0, 1.0]
            - f1_macro: Macro-averaged F1 score [0.0, 1.0]
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = np.asarray(y_pred)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    labels = list(range(num_classes)) if num_classes is not None else None

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
    }


class MetricTracker:
    """
    Online accumulator for batch predictions and loss tracking during an epoch.
    """

    def __init__(self, num_classes: int = 36) -> None:
        self.num_classes = num_classes
        self.total_loss: float = 0.0
        self.total_samples: int = 0
        self.all_preds: List[int] = []
        self.all_targets: List[int] = []

    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate batch loss and prediction vectors."""
        batch_size = targets.size(0)
        self.total_loss += float(loss) * batch_size
        self.total_samples += batch_size

        if preds.ndim > 1:
            preds = torch.argmax(preds, dim=1)

        self.all_preds.extend(preds.detach().cpu().tolist())
        self.all_targets.extend(targets.detach().cpu().tolist())

    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics across all recorded samples."""
        avg_loss = (self.total_loss / self.total_samples) if self.total_samples > 0 else 0.0
        metrics = compute_metrics(self.all_targets, self.all_preds, num_classes=self.num_classes)
        metrics["loss"] = float(avg_loss)
        return metrics

    def reset(self) -> None:
        """Reset internal accumulator state."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.all_preds.clear()
        self.all_targets.clear()
