"""
Export training curve plots, test confusion matrices, and detailed evaluation metrics.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch

# Ensure project root is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSES, DOCS_DIR, NUM_CLASSES, SPLITS_DIR
from src.data.dataloaders import get_dataloaders
from src.models import build_model
from src.training.engine import get_default_device


def plot_training_curves(log_file: Path, output_file: Path) -> None:
    """Plot training and validation loss, accuracy, and macro F1 curves with academic styling."""
    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if "epoch" in json.loads(line)]

    if not records:
        print(f"No epoch records found in {log_file}")
        return

    # Select the most recent continuous session
    last_session_id = records[-1].get("session_id")
    if last_session_id:
        records = [r for r in records if r.get("session_id") == last_session_id]

    epochs = [r["epoch"] for r in records]
    train_loss = [r["train"]["loss"] for r in records]
    val_loss = [r["val"]["loss"] for r in records]
    train_acc = [r["train"]["accuracy"] for r in records]
    val_acc = [r["val"]["accuracy"] for r in records]
    val_f1 = [r["val"]["f1_macro"] for r in records]

    # Academic palette
    c_train = "#2563eb"    # Blue
    c_val = "#dc2626"      # Crimson
    c_acc_tr = "#059669"   # Emerald
    c_acc_val = "#2563eb"  # Blue
    c_f1_val = "#7c3aed"   # Purple

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Left: Cross-Entropy Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", color=c_train, lw=2.2)
    axes[0].plot(epochs, val_loss, label="Val Loss", color=c_val, lw=2.0, linestyle="--")
    axes[0].set_title("Cross-Entropy Loss (Label Smoothing = 0.1)", fontsize=12, fontweight="600", pad=12)
    axes[0].set_xlabel("Epoch", fontsize=11, labelpad=8)
    axes[0].set_ylabel("Loss", fontsize=11, labelpad=8)
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color="#94a3b8")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["left"].set_color("#64748b")
    axes[0].spines["bottom"].set_color("#64748b")
    axes[0].legend(frameon=True, framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=10)

    # Right: Accuracy & Macro-F1
    axes[1].plot(epochs, train_acc, label="Train Accuracy", color=c_acc_tr, lw=2.2)
    axes[1].plot(epochs, val_acc, label="Val Accuracy", color=c_acc_val, lw=2.0, linestyle="--")
    axes[1].plot(epochs, val_f1, label="Val Macro-F1", color=c_f1_val, lw=2.0, linestyle=":")
    axes[1].set_title("Validation Accuracy & Macro-F1", fontsize=12, fontweight="600", pad=12)
    axes[1].set_xlabel("Epoch", fontsize=11, labelpad=8)
    axes[1].set_ylabel("Score", fontsize=11, labelpad=8)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color="#94a3b8")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["left"].set_color("#64748b")
    axes[1].spines["bottom"].set_color("#64748b")
    axes[1].legend(frameon=True, framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=10)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {output_file}")


def evaluate_test_set(
    model_name: str,
    checkpoint_path: Path,
    test_manifest: Path,
    output_matrix_file: Path,
    output_metrics_file: Path,
    device_name: str = None,
) -> dict:
    """Evaluate checkpoint on unseen test set and save confusion matrix and per-class stats."""
    device = get_default_device(device_name)
    print(f"Evaluating checkpoint: {checkpoint_path} on {device}")

    # Build model and load weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fc_dim = 256
    if "classifier.2.weight" in ckpt["model_state_dict"]:
        fc_dim = ckpt["model_state_dict"]["classifier.2.weight"].shape[0]
    elif "model.fc.1.weight" in ckpt["model_state_dict"] and ckpt["model_state_dict"]["model.fc.1.weight"].shape[0] == NUM_CLASSES:
        fc_dim = 0
    model = build_model(model_name, num_classes=NUM_CLASSES, fc_dim=fc_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load test dataloader
    _, _, test_loader = get_dataloaders(
        train=test_manifest,
        val=test_manifest,
        test=test_manifest,
        batch_size=64,
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Test metrics JSON saved to: {output_metrics_file}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(15, 13), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        cbar=True,
        linewidths=0.5,
        linecolor="#f1f5f9",
        annot_kws={"size": 8.5},
    )
    model_display_names = {
        "simple_cnn": "Model 1 (Simple CNN)",
        "complex_cnn": "Model 2 (Complex Residual CNN)",
        "transfer_cnn": "Model 3 (Transfer Learning ResNet-18)",
    }
    disp_title = model_display_names.get(model_name, model_name)
    ax.set_title(
        f"Test Set Confusion Matrix: {disp_title}\nEvaluation on Unseen Signers P9 & P10",
        fontsize=13,
        fontweight="600",
        pad=15,
    )
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Ground Truth Class", fontsize=11, labelpad=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    output_matrix_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_matrix_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {output_matrix_file}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export report figures and evaluation data")
    parser.add_argument("--model", type=str, default="simple_cnn")
    parser.add_argument("--checkpoint", type=str, default="weights/best_model_1_simple_cnn.pt")
    parser.add_argument("--log-file", type=str, default="logs/training/simple_cnn_history.jsonl")
    parser.add_argument("--test-manifest", type=str, default="data/splits/subject_test.csv")
    args = parser.parse_args()

    curves_out = DOCS_DIR / "figures" / "evaluations" / f"{args.model}_training_curves.png"
    cm_out = DOCS_DIR / "figures" / "evaluations" / f"{args.model}_confusion_matrix.png"
    metrics_out = DOCS_DIR / "metrics" / f"{args.model}_test_evaluation.json"

    plot_training_curves(Path(args.log_file), curves_out)
    evaluate_test_set(
        model_name=args.model,
        checkpoint_path=Path(args.checkpoint),
        test_manifest=Path(args.test_manifest),
        output_matrix_file=cm_out,
        output_metrics_file=metrics_out,
    )


if __name__ == "__main__":
    main()
