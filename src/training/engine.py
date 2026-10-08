"""
Unified training engine supporting hardware acceleration and graceful CPU fallback.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import warnings
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader

from .losses import LabelSmoothingCrossEntropy
from .metrics import MetricTracker

# Suppress repetitive DirectML CPU fallback notices for aten::lerp during optimizer steps
warnings.filterwarnings(
    "ignore",
    message=".*The operator 'aten::lerp.Scalar_out' is not currently supported on the DML backend.*",
)


def get_default_device(device_preference: Optional[str] = None) -> torch.device:
    """
    Select execution device with DirectML support on Windows/AMD and CPU fallback.
    """
    if device_preference is not None:
        pref = device_preference.strip().lower()
        if pref in ("dml", "directml"):
            try:
                import torch_directml

                return torch_directml.device()
            except Exception as exc:
                warnings.warn(f"DirectML requested but unavailable ({exc}); falling back to cpu.")
                return torch.device("cpu")
        return torch.device(pref)

    try:
        import torch_directml

        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


@dataclass
class TrainerConfig:
    """Trainer configuration hyper-parameters and runtime settings."""

    model_name: str
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    save_dir: str = "weights"
    log_dir: str = "logs/training"
    early_stopping_patience: int = 10


class Trainer:
    """
    Unified model training and evaluation engine.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config
        self.device = device or get_default_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.save_dir = Path(self.config.save_dir)
        self.log_dir = Path(self.config.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = LabelSmoothingCrossEntropy(smoothing=self.config.label_smoothing)

        # Enforce foreach=False for DirectML compatibility
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            foreach=False,
        )

        self.scheduler = self._build_scheduler()
        self.best_val_f1: float = -1.0
        self.best_val_loss: float = float("inf")
        self.history: list[dict[str, Any]] = []

    def _build_scheduler(self) -> _LRScheduler:
        """Construct Cosine Annealing learning rate scheduler with linear warmup."""
        warmup_epochs = max(1, self.config.warmup_epochs)
        main_epochs = max(1, self.config.epochs - warmup_epochs)

        warmup_sched = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=main_epochs,
            eta_min=self.config.min_lr,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )

    def train_epoch(self) -> Dict[str, float]:
        """Execute a single training epoch across train_loader batches."""
        self.model.train()
        tracker = MetricTracker(num_classes=36)
        start_time = time.perf_counter()

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            tracker.update(loss.item(), outputs, targets)

        epoch_time = time.perf_counter() - start_time
        metrics = tracker.compute()
        metrics["epoch_time_sec"] = epoch_time
        metrics["samples_per_sec"] = (
            (tracker.total_samples / epoch_time) if epoch_time > 0 else 0.0
        )
        return metrics

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Execute evaluation over a dataloader without computing gradients."""
        self.model.eval()
        tracker = MetricTracker(num_classes=36)
        start_time = time.perf_counter()

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            tracker.update(loss.item(), outputs, targets)

        eval_time = time.perf_counter() - start_time
        metrics = tracker.compute()
        metrics["eval_time_sec"] = eval_time
        return metrics

    def fit(self) -> list[dict[str, Any]]:
        """Run the full multi-epoch training pipeline."""
        log_file = self.log_dir / f"{self.config.model_name}_history.jsonl"
        best_ckpt_path = self.save_dir / f"best_{self.config.model_name}.pt"
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train": train_metrics,
                "val": val_metrics,
            }
            self.history.append(epoch_record)

            # Write event directly to JSONL log
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_record) + "\n")

            # Checkpoint on best validation macro F1
            val_f1 = val_metrics["f1_macro"]
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_f1": val_f1,
                        "config": asdict(self.config),
                    },
                    best_ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break

        return self.history
