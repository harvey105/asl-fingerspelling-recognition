import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torchvision import transforms as T

from src.config import NUM_CLASSES
from src.data.roi_detector import detect, make_hands
from src.models import build_model
from src.training.engine import get_default_device

EXT_RAW_DIR = PROJECT_ROOT / "data" / "external_test"
EXT_CROPS_DIR = PROJECT_ROOT / "data" / "external_crops"
EXT_MANIFEST = EXT_RAW_DIR / "manifest.csv"
METRICS_OUT = PROJECT_ROOT / "docs" / "metrics" / "external_benchmark_results.json"

FRAME_SIZE = 300
CROP_SIZE = 224
PAD_GRAY = (128, 128, 128)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_to_frame(bgr: np.ndarray) -> np.ndarray:
    """Scale longest side to 300 px and symmetrically pad to 300x300."""
    h, w = bgr.shape[:2]
    scale = FRAME_SIZE / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
    canvas = np.full((FRAME_SIZE, FRAME_SIZE, 3), PAD_GRAY, dtype=np.uint8)
    top = (FRAME_SIZE - nh) // 2
    left = (FRAME_SIZE - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def letterbox_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Pad cropped hand to square 224x224 maintaining aspect ratio."""
    h, w = crop_bgr.shape[:2]
    scale = CROP_SIZE / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((CROP_SIZE, CROP_SIZE, 3), PAD_GRAY, dtype=np.uint8)
    top = (CROP_SIZE - nh) // 2
    left = (CROP_SIZE - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def preprocess_external_dataset() -> pd.DataFrame:
    """Process external photos into canonical letterbox crops."""
    print("Normalizing and cropping external test images...", flush=True)
    manifest = pd.read_csv(EXT_MANIFEST)
    EXT_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    hands = make_hands()

    crop_records: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        raw_path = PROJECT_ROOT / row.filepath
        class_name = str(row.class_name)
        out_path = EXT_CROPS_DIR / class_name / (raw_path.stem + ".png")

        if not out_path.exists():
            bgr = cv2.imread(str(raw_path))
            if bgr is None:
                raise FileNotFoundError(f"Failed to load photo: {raw_path}")

            norm_300 = normalize_to_frame(bgr)
            if class_name == "0":
                box = (0, 0, FRAME_SIZE - 1, FRAME_SIZE - 1)
            else:
                detected_box, _ = detect(norm_300, hands)
                box = (
                    detected_box
                    if detected_box is not None
                    else (0, 0, FRAME_SIZE - 1, FRAME_SIZE - 1)
                )

            x0, y0, x1, y1 = box
            side = max(x1 - x0, y1 - y0) + 1
            margin = round(side * 0.20)
            mx0, my0 = max(0, x0 - margin), max(0, y0 - margin)
            mx1, my1 = min(FRAME_SIZE - 1, x1 + margin), min(FRAME_SIZE - 1, y1 + margin)

            crop = norm_300[my0 : my1 + 1, mx0 : mx1 + 1]
            out_img = letterbox_crop(crop)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), out_img)

        crop_records.append(
            {
                "filepath": row.filepath,
                "crop_path": out_path.relative_to(PROJECT_ROOT).as_posix(),
                "class_name": class_name,
                "class_id": row.class_id,
                "volunteer_name": row.volunteer_name,
                "source": row.source,
                "is_mirrored": row.is_mirrored,
            }
        )

    return pd.DataFrame(crop_records)


def load_model_from_checkpoint(
    model_name: str, ckpt_path: Path, device: torch.device
) -> nn.Module:
    """Load model architecture and weights from saved checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if model_name == "simple_cnn":
        model = build_model("simple_cnn", num_classes=NUM_CLASSES)
    elif model_name == "complex_cnn":
        model = build_model("complex_cnn", num_classes=NUM_CLASSES, fc_dim=384)
    elif model_name == "transfer_cnn":
        model = build_model(
            "transfer_cnn",
            num_classes=NUM_CLASSES,
            fc_dim=0,
            pretrained=False,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_model_inference(
    model: nn.Module, df: pd.DataFrame, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference across all manifest rows and return predictions and labels."""
    transform = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for r in df.itertuples(index=False):
            img_path = PROJECT_ROOT / r.crop_path
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
            tensor = transform(rgb).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = int(logits.argmax(dim=1).item())
            predictions.append(pred)
            labels.append(int(r.class_id))

    return np.array(predictions), np.array(labels)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate accuracy, macro precision, recall, and F1."""
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    acc = float((y_true == y_pred).mean())
    p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "accuracy": acc,
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "total_samples": len(y_true),
        "correct_samples": int((y_true == y_pred).sum()),
    }


def execute_external_benchmark() -> dict[str, Any]:
    """Run end-to-end external evaluation for all 3 models."""
    device = get_default_device(None)
    df_crops = preprocess_external_dataset()

    models_meta = [
        ("Model 1: Baseline Simple CNN", "simple_cnn", PROJECT_ROOT / "weights" / "best_model_1_simple_cnn.pt"),
        ("Model 2: Deep Residual CNN", "complex_cnn", PROJECT_ROOT / "weights" / "best_model_2_complex_cnn.pt"),
        ("Model 3: Fine-Tuned ResNet-18", "transfer_cnn", PROJECT_ROOT / "weights" / "best_model_3_transfer_cnn.pt"),
    ]

    all_results: dict[str, Any] = {}
    for display_name, m_key, ckpt_path in models_meta:
        print(f"Evaluating {display_name}...", flush=True)
        model = load_model_from_checkpoint(m_key, ckpt_path, device)
        y_pred, y_true = run_model_inference(model, df_crops, device)

        overall_metrics = calculate_metrics(y_true, y_pred)
        mask_orig = df_crops["source"] == "original"
        orig_metrics = calculate_metrics(y_true[mask_orig], y_pred[mask_orig])

        volunteers = sorted(df_crops["volunteer_name"].unique())
        volunteer_metrics = {
            vol: calculate_metrics(y_true[df_crops["volunteer_name"] == vol], y_pred[df_crops["volunteer_name"] == vol])
            for vol in volunteers
        }

        mask_right = (df_crops["volunteer_name"] == "hck") & (~df_crops["is_mirrored"])
        mask_left = (df_crops["volunteer_name"] == "hck") & (df_crops["is_mirrored"])
        metrics_right = calculate_metrics(y_true[mask_right], y_pred[mask_right])
        metrics_left = calculate_metrics(y_true[mask_left], y_pred[mask_left])

        all_results[m_key] = {
            "display_name": display_name,
            "overall_180": overall_metrics,
            "headline_144_originals": orig_metrics,
            "volunteer_breakdown": volunteer_metrics,
            "handedness_parity": {
                "right_hand_accuracy": metrics_right["accuracy"],
                "left_hand_accuracy": metrics_left["accuracy"],
                "parity_delta": abs(metrics_left["accuracy"] - metrics_right["accuracy"]),
                "paired_sample_count": 36,
            },
        }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved benchmark metrics to {METRICS_OUT}", flush=True)
    return all_results


if __name__ == "__main__":
    execute_external_benchmark()
