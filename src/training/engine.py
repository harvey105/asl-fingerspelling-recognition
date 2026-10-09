import json
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader

from .losses import LabelSmoothingCrossEntropy
from .metrics import MetricTracker

warnings.filterwarnings(
    "ignore",
    message=".*The operator 'aten::lerp.Scalar_out' is not currently supported on the DML backend.*",
)


def get_default_device(device_preference: str | None = None) -> torch.device:
    """Select compute device supporting DirectML, CUDA, or CPU fallback.

    Args:
        device_preference: Optional device string ('dml', 'cuda', 'cpu').

    Returns:
        Configured torch.device.
    """
    if device_preference is not None:
        pref = device_preference.strip().lower()
        if pref in ("dml", "directml"):
            try:
                import torch_directml

                return torch_directml.device()
            except (ImportError, RuntimeError) as exc:
                warnings.warn(
                    f"DirectML requested but unavailable ({exc}); falling back to cpu."
                )
                return torch.device("cpu")
        return torch.device(pref)

    try:
        import torch_directml

        return torch_directml.device()
    except (ImportError, RuntimeError):
        return torch.device("cpu")


@dataclass
class TrainerConfig:
    """Hyperparameter and runtime configuration for Trainer."""

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
    checkpoint_name: str | None = None
    log_filename: str | None = None
    print_freq: int = 50
    lr_backbone: float | None = None
    freeze_epochs: int = 0


class Trainer:
    """Standardized neural network training, validation, and evaluation engine.

    Args:
        model: PyTorch model module.
        config: TrainerConfig settings.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Target torch compute device.
        test_loader: Optional test set DataLoader.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | None = None,
        test_loader: DataLoader | None = None,
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

        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=self.config.label_smoothing
        )

        if (
            self.config.lr_backbone is not None
            and hasattr(self.model, "get_parameter_groups")
        ):
            param_groups = self.model.get_parameter_groups(
                lr_backbone=self.config.lr_backbone,
                lr_head=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.optimizer = torch.optim.AdamW(param_groups, foreach=False)
        else:
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
        """Construct Cosine Annealing learning rate schedule with linear warmup."""
        warmup_epochs = max(1, self.config.warmup_epochs)
        main_epochs = max(1, self.config.epochs - warmup_epochs)
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=main_epochs, eta_min=self.config.min_lr
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    def train_epoch(
        self, epoch: int = 1, total_epochs: int = 1
    ) -> dict[str, float]:
        """Execute one complete training epoch over train_loader batches."""
        self.model.train()
        tracker = MetricTracker(num_classes=36)
        start_time = time.perf_counter()
        num_batches = len(self.train_loader)

        for step, (inputs, targets) in enumerate(self.train_loader, start=1):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            tracker.update(loss.item(), outputs, targets)

            if step % self.config.print_freq == 0 or step == num_batches:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch:02d}/{total_epochs:02d}] Step [{step:03d}/{num_batches:03d}] "
                    f"Loss: {tracker.loss:.4f} Acc: {tracker.accuracy:.4f} LR: {lr:.2e}",
                    flush=True,
                )

        elapsed = time.perf_counter() - start_time
        metrics = tracker.compute()
        metrics["epoch_time_sec"] = elapsed
        metrics["samples_per_sec"] = (
            tracker.total_samples / elapsed if elapsed > 0 else 0.0
        )
        return metrics

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        """Compute evaluation metrics over a DataLoader without tracking gradients."""
        self.model.eval()
        tracker = MetricTracker(num_classes=36)
        start_time = time.perf_counter()

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            tracker.update(loss.item(), outputs, targets)

        metrics = tracker.compute()
        metrics["eval_time_sec"] = time.perf_counter() - start_time
        return metrics

    def fit(self) -> list[dict[str, Any]]:
        """Run full training loop with early stopping, scheduling, and checkpointing."""
        log_name = (
            self.config.log_filename or f"{self.config.model_name}_history.jsonl"
        )
        log_file = self.log_dir / log_name
        ckpt_name = (
            self.config.checkpoint_name or f"best_{self.config.model_name}"
        )
        ckpt_path = self.save_dir / f"{ckpt_name}.pt"
        patience_counter = 0
        session_id = str(uuid.uuid4())

        print(
            f"Training session {session_id} started. Checkpoint target: {ckpt_path}",
            flush=True,
        )

        for epoch in range(1, self.config.epochs + 1):
            if self.config.freeze_epochs > 0:
                if epoch <= self.config.freeze_epochs:
                    if hasattr(self.model, "freeze_backbone"):
                        self.model.freeze_backbone()
                elif epoch == self.config.freeze_epochs + 1 and hasattr(
                    self.model, "unfreeze_backbone"
                ):
                    self.model.unfreeze_backbone()
                    print(
                        f"--> Epoch {epoch:02d}: Backbone unfrozen for fine-tuning.",
                        flush=True,
                    )

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            train_metrics = self.train_epoch(epoch, self.config.epochs)
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()

            record = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "lr": current_lr,
                "train": train_metrics,
                "val": val_metrics,
            }
            self.history.append(record)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            val_f1 = val_metrics["f1_macro"]
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_f1": val_f1,
                        "val_acc": val_metrics["accuracy"],
                        "config": asdict(self.config),
                    },
                    ckpt_path,
                )

            marker = " (*BEST*)" if is_best else ""
            print(
                f"--> Epoch {epoch:02d}/{self.config.epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1_macro']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1_macro']:.4f}"
                f"{marker} | LR: {current_lr:.2e} | Time: {train_metrics['epoch_time_sec']:.1f}s",
                flush=True,
            )

            if not is_best:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} (patience={self.config.early_stopping_patience}).",
                        flush=True,
                    )
                    break

        if ckpt_path.exists():
            print(
                f"Restoring best weights from {ckpt_path} (Val F1: {self.best_val_f1:.4f}).",
                flush=True,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)

        if self.test_loader is not None:
            print("Evaluating best model on test set...", flush=True)
            test_metrics = self.evaluate(self.test_loader)
            test_record = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "test_evaluation",
                "test": test_metrics,
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(test_record) + "\n")
            print(
                f"Test Results ({ckpt_path.name}):\n"
                f"  Accuracy:  {test_metrics['accuracy'] * 100:.2f}%\n"
                f"  Precision: {test_metrics['precision_macro']:.4f}\n"
                f"  Recall:    {test_metrics['recall_macro']:.4f}\n"
                f"  F1-score:  {test_metrics['f1_macro']:.4f}\n"
                f"  Loss:      {test_metrics['loss']:.4f}",
                flush=True,
            )

        return self.history
