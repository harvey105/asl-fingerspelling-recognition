import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SquashedTestDataset(Dataset):
    """Dataset cropping bounding boxes directly to 224x224 without letterbox padding."""

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

        self.normalize = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, (x0, y0, x1, y1), label = self.samples[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Missing raw image: {path}")

        crop = bgr[y0 : y1 + 1, x0 : x1 + 1]
        squashed = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(squashed, cv2.COLOR_BGR2RGB)
        return self.normalize(rgb), label


def evaluate_dataloader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Compute classification metrics and mean confidence over a DataLoader."""
    model.eval()
    all_preds, all_targets, all_max_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_prob, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.tolist())
            all_max_probs.extend(max_prob.cpu().tolist())

    y_true, y_pred = np.array(all_targets), np.array(all_preds)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_confidence": float(np.mean(all_max_probs)),
        "total_samples": len(y_true),
        "correct_samples": int((y_true == y_pred).sum()),
    }


def run_ablation_aspect_ratio(device: torch.device) -> dict[str, Any]:
    """Evaluate letterbox padding vs naive squashing."""
    print("Ablation 2: Aspect Ratio Preservation")
    cache_path = PROJECT_ROOT / "docs" / "metrics" / "ablation_aspect_ratio_cached.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ckpt_m3 = torch.load(PROJECT_ROOT / "weights" / "best_model_3_transfer_cnn.pt", map_location="cpu", weights_only=False)
    m3 = build_model("transfer_cnn", num_classes=NUM_CLASSES, fc_dim=0, pretrained=False)
    m3.load_state_dict(ckpt_m3["model_state_dict"])
    m3.to(device)

    manifest = pd.read_csv(SPLITS_DIR / "subject_test.csv")
    crop_log = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "roi_crop_log.csv")

    _, _, letterbox_loader = get_dataloaders(
        train=SPLITS_DIR / "subject_train.csv",
        val=SPLITS_DIR / "subject_val.csv",
        test=SPLITS_DIR / "subject_test.csv",
        batch_size=64,
        num_workers=4,
    )
    squashed_loader = DataLoader(
        SquashedTestDataset(manifest, crop_log), batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    letterbox_m3 = evaluate_dataloader(m3, letterbox_loader, device)
    squashed_m3 = evaluate_dataloader(m3, squashed_loader, device)

    ckpt_m1 = torch.load(PROJECT_ROOT / "weights" / "best_model_1_simple_cnn.pt", map_location="cpu", weights_only=False)
    m1 = build_model("simple_cnn", num_classes=NUM_CLASSES, fc_dim=256)
    m1.load_state_dict(ckpt_m1["model_state_dict"])
    m1.to(device)

    letterbox_m1 = evaluate_dataloader(m1, letterbox_loader, device)
    squashed_m1 = evaluate_dataloader(m1, squashed_loader, device)

    return {
        "model_3": {
            "letterbox_padding": letterbox_m3,
            "naive_squashing": squashed_m3,
            "accuracy_degradation": letterbox_m3["accuracy"] - squashed_m3["accuracy"],
            "f1_degradation": letterbox_m3["f1_macro"] - squashed_m3["f1_macro"],
        },
        "model_1": {
            "letterbox_padding": letterbox_m1,
            "naive_squashing": squashed_m1,
            "accuracy_degradation": letterbox_m1["accuracy"] - squashed_m1["accuracy"],
            "f1_degradation": letterbox_m1["f1_macro"] - squashed_m1["f1_macro"],
        },
    }


def run_ablation_split_leakage(device: torch.device) -> dict[str, Any]:
    """Compare random split vs subject-independent split."""
    print("Ablation 1: Data Split Leakage")
    cache_path = PROJECT_ROOT / "docs" / "metrics" / "ablation_split_leakage_cached.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(PROJECT_ROOT / "docs" / "metrics" / "transfer_cnn_test_evaluation.json", "r") as f:
        m3_subj = json.load(f)
    with open(PROJECT_ROOT / "docs" / "metrics" / "simple_cnn_test_evaluation.json", "r") as f:
        m1_subj = json.load(f)

    m1_cfg = TrainerConfig(
        model_name="simple_cnn",
        epochs=15,
        early_stopping_patience=8,
        learning_rate=5e-4,
        weight_decay=5e-4,
        label_smoothing=0.1,
        save_dir=str(PROJECT_ROOT / "weights"),
        log_dir=str(PROJECT_ROOT / "logs" / "ablation"),
        checkpoint_name="ablation_m1_random_split",
        log_filename="ablation_m1_random_split.jsonl",
    )
    m1 = build_model("simple_cnn", num_classes=NUM_CLASSES, fc_dim=256)
    t_ldr, v_ldr, te_ldr = get_dataloaders(
        train=SPLITS_DIR / "random_train.csv",
        val=SPLITS_DIR / "random_val.csv",
        test=SPLITS_DIR / "random_test.csv",
        batch_size=64,
    )
    t1 = Trainer(m1, m1_cfg, t_ldr, v_ldr, test_loader=te_ldr, device=device)
    t1.fit()
    m1_random = t1.evaluate(te_ldr)

    m3_cfg = TrainerConfig(
        model_name="transfer_cnn",
        epochs=15,
        early_stopping_patience=8,
        learning_rate=5e-4,
        lr_backbone=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.1,
        save_dir=str(PROJECT_ROOT / "weights"),
        log_dir=str(PROJECT_ROOT / "logs" / "ablation"),
        checkpoint_name="ablation_m3_random_split",
        log_filename="ablation_m3_random_split.jsonl",
    )
    m3 = build_model("transfer_cnn", num_classes=NUM_CLASSES, fc_dim=0, pretrained=True)
    t3 = Trainer(m3, m3_cfg, t_ldr, v_ldr, test_loader=te_ldr, device=device)
    t3.fit()
    m3_random = t3.evaluate(te_ldr)

    return {
        "model_1": {
            "subject_split": {"accuracy": m1_subj["accuracy"], "f1_macro": m1_subj["macro avg"]["f1-score"]},
            "random_split": {"accuracy": m1_random["accuracy"], "f1_macro": m1_random["f1_macro"]},
            "leakage_inflation": m1_random["accuracy"] - m1_subj["accuracy"],
        },
        "model_3": {
            "subject_split": {"accuracy": m3_subj["accuracy"], "f1_macro": m3_subj["macro avg"]["f1-score"]},
            "random_split": {"accuracy": m3_random["accuracy"], "f1_macro": m3_random["f1_macro"]},
            "leakage_inflation": m3_random["accuracy"] - m3_subj["accuracy"],
        },
    }


def run_ablation_label_smoothing(device: torch.device) -> dict[str, Any]:
    """Compare training with smoothing (alpha=0.10) vs without smoothing (alpha=0.0)."""
    print("Ablation 3: Label Smoothing Regularization")
    ckpt_path = PROJECT_ROOT / "weights" / "best_model_3_transfer_cnn.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m3_smooth = build_model("transfer_cnn", num_classes=NUM_CLASSES, fc_dim=0, pretrained=False)
    m3_smooth.load_state_dict(ckpt["model_state_dict"])
    m3_smooth.to(device)

    train_ldr, val_ldr, test_ldr = get_dataloaders(
        train=SPLITS_DIR / "subject_train.csv",
        val=SPLITS_DIR / "subject_val.csv",
        test=SPLITS_DIR / "subject_test.csv",
        batch_size=64,
    )
    smooth_metrics = evaluate_dataloader(m3_smooth, test_ldr, device)

    ce_cfg = TrainerConfig(
        model_name="transfer_cnn",
        epochs=15,
        early_stopping_patience=8,
        learning_rate=5e-4,
        lr_backbone=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.0,
        save_dir=str(PROJECT_ROOT / "weights"),
        log_dir=str(PROJECT_ROOT / "logs" / "ablation"),
        checkpoint_name="ablation_m3_ce_nosmooth",
        log_filename="ablation_m3_ce_nosmooth.jsonl",
    )
    m3_ce = build_model("transfer_cnn", num_classes=NUM_CLASSES, fc_dim=0, pretrained=True)
    ce_trainer = Trainer(m3_ce, ce_cfg, train_ldr, val_ldr, test_loader=test_ldr, device=device)
    ce_trainer.fit()
    ce_metrics = evaluate_dataloader(ce_trainer.model, test_ldr, device)

    return {
        "label_smoothing_alpha_0_10": smooth_metrics,
        "standard_cross_entropy_alpha_0_0": ce_metrics,
        "accuracy_gain_from_smoothing": smooth_metrics["accuracy"] - ce_metrics["accuracy"],
        "confidence_calibration_delta": ce_metrics["mean_confidence"] - smooth_metrics["mean_confidence"],
    }


def main() -> None:
    device = get_default_device(None)
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "ablation_2_aspect_ratio": run_ablation_aspect_ratio(device),
        "ablation_1_split_leakage": run_ablation_split_leakage(device),
        "ablation_3_label_smoothing": run_ablation_label_smoothing(device),
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved all ablation metrics to {METRICS_OUT}", flush=True)


if __name__ == "__main__":
    main()
