"""
Unit tests for losses, metrics computation, and training engine.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.simple_cnn import SimpleCNN
from src.training.engine import Trainer, TrainerConfig, get_default_device
from src.training.losses import LabelSmoothingCrossEntropy
from src.training.metrics import MetricTracker, compute_metrics


def test_label_smoothing_cross_entropy():
    # When smoothing=0, output matches nn.CrossEntropyLoss
    criterion_smooth0 = LabelSmoothingCrossEntropy(smoothing=0.0)
    criterion_std = nn.CrossEntropyLoss()

    logits = torch.randn(8, 36)
    targets = torch.randint(0, 36, (8,))

    loss_smooth0 = criterion_smooth0(logits, targets)
    loss_std = criterion_std(logits, targets)

    assert torch.isclose(loss_smooth0, loss_std, atol=1e-5)

    # When smoothing > 0, loss is valid and finite
    criterion_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_smooth = criterion_smooth(logits, targets)
    assert loss_smooth.item() > 0.0
    assert torch.isfinite(loss_smooth)


def test_compute_metrics():
    # Perfect predictions across 36 classes
    y_true = np.arange(36)
    y_pred = np.arange(36)

    metrics = compute_metrics(y_true, y_pred, num_classes=36)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0

    # Test tensor input with logits
    logits = torch.zeros((10, 36))
    for i in range(10):
        logits[i, i] = 5.0
    targets = torch.arange(10)

    partial_metrics = compute_metrics(targets, logits, num_classes=36)
    assert partial_metrics["accuracy"] == 1.0


def test_metric_tracker():
    tracker = MetricTracker(num_classes=36)
    logits = torch.randn(16, 36)
    targets = torch.randint(0, 36, (16,))

    tracker.update(loss=0.5, preds=logits, targets=targets)
    res = tracker.compute()

    assert "accuracy" in res
    assert "precision_macro" in res
    assert "recall_macro" in res
    assert "f1_macro" in res
    assert res["loss"] == 0.5


def test_trainer_single_step_train_and_eval(tmp_path):
    device = torch.device("cpu")
    model = SimpleCNN(in_channels=3, num_classes=36)

    # Synthetic dataset
    inputs = torch.randn(8, 3, 224, 224)
    targets = torch.randint(0, 36, (8,))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    config = TrainerConfig(
        model_name="test_simple_cnn",
        epochs=1,
        learning_rate=1e-3,
        save_dir=str(tmp_path / "weights"),
        log_dir=str(tmp_path / "logs"),
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loader,
        val_loader=loader,
        device=device,
    )

    train_metrics = trainer.train_epoch()
    eval_metrics = trainer.evaluate(loader)

    assert "loss" in train_metrics
    assert "accuracy" in train_metrics
    assert "f1_macro" in train_metrics
    assert "loss" in eval_metrics
    assert "accuracy" in eval_metrics
