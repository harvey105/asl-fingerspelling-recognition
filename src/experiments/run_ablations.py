"""
Controlled ablation experiment runner isolating single architectural and methodological variables:
  1. Data split leakage: Random split (burst leakage) vs Subject split (LOSO)
  2. Aspect ratio preservation: Letterbox padding vs Naive squashing
  3. Label smoothing regularization: Smoothing (alpha=0.10) vs Standard Cross-Entropy (alpha=0.0)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.config import NUM_CLASSES, SPLITS_DIR
from src.data.dataloaders import get_dataloaders
from src.models import build_model
from src.training.engine import Trainer, TrainerConfig, get_default_device

METRICS_OUT = PROJECT_ROOT / "docs" / "metrics" / "ablation_results.json"
REPORT_OUT = PROJECT_ROOT / "docs" / "reports" / "ablation_experiments.md"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_GRAY = (128, 128, 128)


# ==============================================================================
# Helper Datasets & Inference
# ==============================================================================


class SquashedTestDataset(Dataset):
    """
    Dataset that loads raw images, crops the hand bounding box from roi_crop_log,
    and directly resizes to 224x224 (naive squashing) without aspect-preserving padding.
    """

    def __init__(self, manifest: pd.DataFrame, crop_log: pd.DataFrame):
        self.samples = []
        log_lookup = crop_log.set_index("filepath").to_dict(orient="index")

        for r in manifest.itertuples(index=False):
            fp = str(r.filepath)
            cls_id = int(r.class_id)
            info = log_lookup.get(fp)
            if info is not None and not np.isnan(info["margin_x0"]):
                box = (
                    round(info["margin_x0"]),
                    round(info["margin_y0"]),
                    round(info["margin_x1"]),
                    round(info["margin_y1"]),
                )
            else:
                box = (0, 0, 299, 299)
            self.samples.append((PROJECT_ROOT / fp, box, cls_id))

        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, (x0, y0, x1, y1), label = self.samples[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Missing raw image: {path}")

        # Crop bounding box and squash directly to 224x224
        crop = bgr[y0 : y1 + 1, x0 : x1 + 1]
        squashed = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(squashed, cv2.COLOR_BGR2RGB)
        tensor = self.normalize(rgb)
        return tensor, label


def evaluate_dataloader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Runs evaluation over a DataLoader and computes classification metrics and confidence."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_max_probs: list[float] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_prob, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.tolist())
            all_max_probs.extend(max_prob.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float((y_true == y_pred).mean())
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    avg_conf = float(np.mean(all_max_probs))

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "mean_confidence": avg_conf,
        "total_samples": len(y_true),
        "correct_samples": int((y_true == y_pred).sum()),
    }


# ==============================================================================
# Ablation 2: Aspect Ratio Preservation
# ==============================================================================


def run_ablation_aspect_ratio(device: torch.device) -> dict[str, Any]:
    """Evaluates Model 3 on canonical letterbox padding vs naive spatial squashing."""
    print("\n" + "=" * 70)
    print("ABLATION 2: ASPECT RATIO PRESERVATION (LETTERBOX VS NAIVE SQUASHING)")
    print("=" * 70, flush=True)

    cache_path = PROJECT_ROOT / "docs" / "metrics" / "ablation_aspect_ratio_cached.json"
    if cache_path.exists():
        print(
            f"Loading measured aspect ratio degradation metrics from {cache_path}...",
            flush=True,
        )
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        m3_drop = cached["model_3"]["accuracy_degradation"]
        m1_drop = cached["model_1"]["accuracy_degradation"]
        print(
            f"Model 3 Letterbox: Acc {cached['model_3']['letterbox_padding']['accuracy']:.4f} | Squashed: Acc {cached['model_3']['naive_squashing']['accuracy']:.4f} (Drop: -{m3_drop * 100:.2f}%)"
        )
        print(
            f"Model 1 Letterbox: Acc {cached['model_1']['letterbox_padding']['accuracy']:.4f} | Squashed: Acc {cached['model_1']['naive_squashing']['accuracy']:.4f} (Drop: -{m1_drop * 100:.2f}%)",
            flush=True,
        )
        return cached

    # Load Model 3 checkpoint
    ckpt_path = PROJECT_ROOT / "weights" / "best_model_3_transfer_cnn.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(
        "transfer_cnn",
        backbone_name="resnet18",
        num_classes=NUM_CLASSES,
        fc_dim=0,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_csv = SPLITS_DIR / "subject_test.csv"
    crop_log_csv = PROJECT_ROOT / "data" / "processed" / "roi_crop_log.csv"
    manifest = pd.read_csv(test_csv)
    crop_log = pd.read_csv(crop_log_csv)

    # 1. Letterbox Test Loader (Standard Canonical Pipeline)
    _, _, letterbox_loader = get_dataloaders(
        train=SPLITS_DIR / "subject_train.csv",
        val=SPLITS_DIR / "subject_val.csv",
        test=test_csv,
        batch_size=64,
        num_workers=4,
    )
    print("Evaluating Model 3 with Letterbox Padding (Aspect Preserved)...", flush=True)
    letterbox_metrics_m3 = evaluate_dataloader(model, letterbox_loader, device)

    # 2. Squashed Test Loader
    squashed_dataset = SquashedTestDataset(manifest, crop_log)
    squashed_loader = DataLoader(
        squashed_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    print(
        "Evaluating Model 3 with Naive Spatial Squashing (Distorted Aspect)...",
        flush=True,
    )
    squashed_metrics_m3 = evaluate_dataloader(model, squashed_loader, device)

    # 3. Model 1 evaluation
    ckpt_m1_path = PROJECT_ROOT / "weights" / "best_model_1_simple_cnn.pt"
    ckpt_m1 = torch.load(ckpt_m1_path, map_location="cpu", weights_only=False)
    model_m1 = build_model("simple_cnn", num_classes=NUM_CLASSES, fc_dim=256)
    model_m1.load_state_dict(ckpt_m1["model_state_dict"])
    model_m1.to(device)
    model_m1.eval()

    print("Evaluating Model 1 with Letterbox Padding...", flush=True)
    letterbox_metrics_m1 = evaluate_dataloader(model_m1, letterbox_loader, device)
    print("Evaluating Model 1 with Naive Spatial Squashing...", flush=True)
    squashed_metrics_m1 = evaluate_dataloader(model_m1, squashed_loader, device)

    m3_acc_drop = letterbox_metrics_m3["accuracy"] - squashed_metrics_m3["accuracy"]
    m3_f1_drop = letterbox_metrics_m3["f1_macro"] - squashed_metrics_m3["f1_macro"]
    m1_acc_drop = letterbox_metrics_m1["accuracy"] - squashed_metrics_m1["accuracy"]
    m1_f1_drop = letterbox_metrics_m1["f1_macro"] - squashed_metrics_m1["f1_macro"]

    print(
        f"Model 3 Letterbox: Acc {letterbox_metrics_m3['accuracy']:.4f} | Squashed: Acc {squashed_metrics_m3['accuracy']:.4f} (Drop: -{m3_acc_drop * 100:.2f}%)"
    )
    print(
        f"Model 1 Letterbox: Acc {letterbox_metrics_m1['accuracy']:.4f} | Squashed: Acc {squashed_metrics_m1['accuracy']:.4f} (Drop: -{m1_acc_drop * 100:.2f}%)",
        flush=True,
    )

    return {
        "model_3": {
            "letterbox_padding": letterbox_metrics_m3,
            "naive_squashing": squashed_metrics_m3,
            "accuracy_degradation": m3_acc_drop,
            "f1_degradation": m3_f1_drop,
        },
        "model_1": {
            "letterbox_padding": letterbox_metrics_m1,
            "naive_squashing": squashed_metrics_m1,
            "accuracy_degradation": m1_acc_drop,
            "f1_degradation": m1_f1_drop,
        },
    }


# ==============================================================================
# Ablation 1: Data Split Leakage
# ==============================================================================


def run_ablation_split_leakage(device: torch.device) -> dict[str, Any]:
    """Trains and compares Model 1 and Model 3 on Random Split vs Subject Split."""
    print("\n" + "=" * 70)
    print("ABLATION 1: DATA SPLIT LEAKAGE (RANDOM SPLIT VS SUBJECT SPLIT)")
    print("=" * 70, flush=True)

    cache_path = (
        PROJECT_ROOT / "docs" / "metrics" / "ablation_split_leakage_cached.json"
    )
    if cache_path.exists():
        print(
            f"Loading measured split leakage metrics from {cache_path}...", flush=True
        )
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(
            f"Model 1: Subject Split {cached['model_1']['subject_split']['accuracy']:.4f} vs Random Split {cached['model_1']['random_split']['accuracy']:.4f} (+{cached['model_1']['leakage_inflation'] * 100:.2f}%)"
        )
        print(
            f"Model 3: Subject Split {cached['model_3']['subject_split']['accuracy']:.4f} vs Random Split {cached['model_3']['random_split']['accuracy']:.4f} (+{cached['model_3']['leakage_inflation'] * 100:.2f}%)",
            flush=True,
        )
        return cached

    # Subject split reference metrics already confirmed
    with open(
        PROJECT_ROOT / "docs" / "metrics" / "transfer_cnn_test_evaluation.json", "r"
    ) as f:
        m3_subj = json.load(f)
    with open(
        PROJECT_ROOT / "docs" / "metrics" / "simple_cnn_test_evaluation.json", "r"
    ) as f:
        m1_subj = json.load(f)

    # 1. Train Model 1 on Random Split
    print(
        "\n--> Training Model 1 on Random Split (up to 15 epochs, early stopping 8)...",
        flush=True,
    )
    m1_random_cfg = TrainerConfig(
        model_name="simple_cnn",
        epochs=15,
        early_stopping_patience=8,
        learning_rate=5e-4,
        weight_decay=5e-4,
        label_smoothing=0.1,
        warmup_epochs=2,
        save_dir=str(PROJECT_ROOT / "weights"),
        log_dir=str(PROJECT_ROOT / "logs" / "ablation"),
        checkpoint_name="ablation_m1_random_split",
        log_filename="ablation_m1_random_split.jsonl",
        print_freq=100,
    )
    m1_random = build_model("simple_cnn", num_classes=NUM_CLASSES, fc_dim=256)
    m1_train_loader, m1_val_loader, m1_test_loader = get_dataloaders(
        train=SPLITS_DIR / "random_train.csv",
        val=SPLITS_DIR / "random_val.csv",
        test=SPLITS_DIR / "random_test.csv",
        batch_size=64,
    )
    m1_trainer = Trainer(
        model=m1_random,
        config=m1_random_cfg,
        train_loader=m1_train_loader,
        val_loader=m1_val_loader,
        test_loader=m1_test_loader,
        device=device,
    )
    m1_trainer.fit()
    m1_random_test = m1_trainer.evaluate(m1_test_loader)

    # 2. Train Model 3 on Random Split
    print(
        "\n--> Training Model 3 on Random Split (up to 15 epochs, early stopping 8)...",
        flush=True,
    )
    m3_random_cfg = TrainerConfig(
        model_name="transfer_cnn",
        epochs=15,
        early_stopping_patience=8,
        learning_rate=5e-4,
        lr_backbone=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.1,
        warmup_epochs=2,
        save_dir=str(PROJECT_ROOT / "weights"),
        log_dir=str(PROJECT_ROOT / "logs" / "ablation"),
        checkpoint_name="ablation_m3_random_split",
        log_filename="ablation_m3_random_split.jsonl",
        print_freq=100,
    )
    m3_random = build_model(
        "transfer_cnn",
        backbone_name="resnet18",
        num_classes=NUM_CLASSES,
        fc_dim=0,
        pretrained=True,
    )
    m3_trainer = Trainer(
        model=m3_random,
        config=m3_random_cfg,
        train_loader=m1_train_loader,
        val_loader=m1_val_loader,
        test_loader=m1_test_loader,
        device=device,
    )
    m3_trainer.fit()
    m3_random_test = m3_trainer.evaluate(m1_test_loader)

    results = {
        "model_1": {
            "subject_split": {
                "accuracy": m1_subj["accuracy"],
                "f1_macro": m1_subj["macro avg"]["f1-score"],
            },
            "random_split": {
                "accuracy": m1_random_test["accuracy"],
                "f1_macro": m1_random_test["f1_macro"],
            },
            "leakage_inflation": m1_random_test["accuracy"] - m1_subj["accuracy"],
        },
        "model_3": {
            "subject_split": {
                "accuracy": m3_subj["accuracy"],
                "f1_macro": m3_subj["macro avg"]["f1-score"],
            },
            "random_split": {
                "accuracy": m3_random_test["accuracy"],
                "f1_macro": m3_random_test["f1_macro"],
            },
            "leakage_inflation": m3_random_test["accuracy"] - m3_subj["accuracy"],
        },
    }

    print(
        f"\nModel 1: Subject Split {results['model_1']['subject_split']['accuracy']:.4f} vs Random Split {results['model_1']['random_split']['accuracy']:.4f} (+{results['model_1']['leakage_inflation'] * 100:.2f}%)"
    )
    print(
        f"Model 3: Subject Split {results['model_3']['subject_split']['accuracy']:.4f} vs Random Split {results['model_3']['random_split']['accuracy']:.4f} (+{results['model_3']['leakage_inflation'] * 100:.2f}%)",
        flush=True,
    )

    return results


# ==============================================================================
# Ablation 3: Label Smoothing Regularization
# ==============================================================================


def run_ablation_label_smoothing(device: torch.device) -> dict[str, Any]:
    """Compares Model 3 trained with alpha=0.10 vs alpha=0.0 under identical conditions."""
    print("\n" + "=" * 70)
    print("ABLATION 3: LABEL SMOOTHING REGULARIZATION (ALPHA=0.10 VS ALPHA=0.0)")
    print("=" * 70, flush=True)

    # Baseline with alpha=0.10: evaluate best checkpoint
    ckpt_path = PROJECT_ROOT / "weights" / "best_model_3_transfer_cnn.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m3_smooth = build_model(
        "transfer_cnn",
        backbone_name="resnet18",
        num_classes=NUM_CLASSES,
        fc_dim=0,
        pretrained=False,
    )
    m3_smooth.load_state_dict(ckpt["model_state_dict"])
    m3_smooth.to(device)

    train_loader, val_loader, test_loader = get_dataloaders(
        train=SPLITS_DIR / "subject_train.csv",
        val=SPLITS_DIR / "subject_val.csv",
        test=SPLITS_DIR / "subject_test.csv",
        batch_size=64,
    )
    print("Evaluating Model 3 with Label Smoothing (alpha = 0.10)...", flush=True)
    smooth_metrics = evaluate_dataloader(m3_smooth, test_loader, device)

    # Train Model 3 with alpha=0.0 (Standard Cross-Entropy)
    print(
        "\n--> Training Model 3 with Standard Cross-Entropy (up to 15 epochs, early stopping 8)...",
        flush=True,
    )
    ce_cfg = TrainerConfig(
        model_name="transfer_cnn",
        epochs=15,
        early_stopping_patience=8,
        learning_rate=5e-4,
        lr_backbone=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.0,
        warmup_epochs=2,
        save_dir=str(PROJECT_ROOT / "weights"),
        log_dir=str(PROJECT_ROOT / "logs" / "ablation"),
        checkpoint_name="ablation_m3_ce_nosmooth",
        log_filename="ablation_m3_ce_nosmooth.jsonl",
        print_freq=100,
    )
    m3_ce = build_model(
        "transfer_cnn",
        backbone_name="resnet18",
        num_classes=NUM_CLASSES,
        fc_dim=0,
        pretrained=True,
    )
    ce_trainer = Trainer(
        model=m3_ce,
        config=ce_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )
    ce_trainer.fit()
    ce_metrics = evaluate_dataloader(ce_trainer.model, test_loader, device)

    print(
        f"\nLabel Smoothing (alpha=0.10): Acc: {smooth_metrics['accuracy']:.4f} | F1: {smooth_metrics['f1_macro']:.4f} | Mean Conf: {smooth_metrics['mean_confidence']:.4f}"
    )
    print(
        f"Standard CE     (alpha=0.00): Acc: {ce_metrics['accuracy']:.4f} | F1: {ce_metrics['f1_macro']:.4f} | Mean Conf: {ce_metrics['mean_confidence']:.4f}"
    )

    return {
        "label_smoothing_alpha_0_10": smooth_metrics,
        "standard_cross_entropy_alpha_0_0": ce_metrics,
        "accuracy_gain_from_smoothing": smooth_metrics["accuracy"]
        - ce_metrics["accuracy"],
        "confidence_calibration_delta": ce_metrics["mean_confidence"]
        - smooth_metrics["mean_confidence"],
    }


# ==============================================================================
# Main Orchestrator
# ==============================================================================


def main() -> None:
    device = get_default_device(None)
    print(f"Ablation Suite starting on device: {device}", flush=True)

    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    }

    # Step 1: Aspect Ratio Preservation Ablation
    results["ablation_2_aspect_ratio"] = run_ablation_aspect_ratio(device)

    # Step 2: Data Split Leakage Ablation
    results["ablation_1_split_leakage"] = run_ablation_split_leakage(device)

    # Step 3: Label Smoothing Ablation
    results["ablation_3_label_smoothing"] = run_ablation_label_smoothing(device)

    # Save machine-readable results
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all ablation metrics to {METRICS_OUT}", flush=True)


if __name__ == "__main__":
    main()
