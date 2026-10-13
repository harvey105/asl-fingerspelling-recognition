import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.models.complex_cnn import ComplexCNN
from src.models.simple_cnn import SimpleCNN
from src.models.transfer_cnn import TransferCNN

ID_TO_CLASS: list[str] = [str(i) for i in range(10)] + [
    chr(ord("A") + i) for i in range(26)
]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "model_1": {
        "title": "Model 1: Simple CNN",
        "checkpoint": Path("weights/best_model_1_simple_cnn.pt"),
    },
    "model_2": {
        "title": "Model 2: Complex Residual CNN",
        "checkpoint": Path("weights/best_model_2_complex_cnn.pt"),
    },
    "model_3": {
        "title": "Model 3: Transfer ResNet-18",
        "checkpoint": Path("weights/best_model_3_transfer_cnn.pt"),
    },
}

TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.paths = df["crop_path"].tolist()
        self.labels = df["class_id"].tolist()
        self.class_names = df["class_name"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return TRANSFORM(img), self.labels[idx], self.class_names[idx]


def normalize_model_name(name: str) -> str:
    cleaned = name.strip().lower()
    if cleaned in ("1", "model_1", "simple", "simple_cnn"):
        return "model_1"
    if cleaned in ("2", "model_2", "complex", "complex_cnn"):
        return "model_2"
    if cleaned in ("3", "model_3", "transfer", "transfer_cnn", "resnet", "resnet18"):
        return "model_3"
    return cleaned


def parse_predict_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASL Fingerspelling Recognition Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "--models",
        dest="models",
        nargs="+",
        default=["model_1", "model_2", "model_3"],
        help="Model(s) to evaluate (model_1, model_2, model_3, or all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate on the full test set (7,200 images). If omitted, uses 250 sample images.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250,
        help="Number of images to evaluate when running in sample mode",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image file path for immediate prediction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predict_results.txt",
        help="Path of text file to save prediction summary",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for test evaluation",
    )
    return parser.parse_args()


def load_model(model_key: str, device: torch.device) -> tuple[torch.nn.Module, str]:
    norm_key = normalize_model_name(model_key)
    if norm_key not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model identifier: {model_key}. Choose model_1, model_2, or model_3."
        )

    spec = MODEL_SPECS[norm_key]
    title = spec["title"]
    ckpt_path = spec["checkpoint"]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if norm_key == "model_1":
        model = SimpleCNN(num_classes=36)
    elif norm_key == "model_2":
        model = ComplexCNN(num_classes=36, fc_dim=384)
    else:
        model = TransferCNN(num_classes=36, fc_dim=0, pretrained=False)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, title


def prepare_test_loader(
    test_csv: Path, full: bool, sample_size: int, batch_size: int, seed: int = 105
) -> tuple[DataLoader, int]:
    if not test_csv.exists():
        raise FileNotFoundError(f"Split test file not found: {test_csv}")

    df = pd.read_csv(test_csv)
    if not full:
        df = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(
            drop=True
        )

    dataset = TestDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, len(df)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    preview_count: int = 10,
    seed: int = 42,
) -> tuple[float, int, int, list[str]]:
    total = 0
    correct = 0
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_imgs, batch_labels, batch_names in loader:
            batch_imgs = batch_imgs.to(device)
            logits = model(batch_imgs)
            probs, preds = torch.softmax(logits, dim=1).max(dim=1)

            for i in range(len(preds)):
                pred_label = ID_TO_CLASS[preds[i].item()]
                is_correct = preds[i].item() == batch_labels[i].item()
                if is_correct:
                    correct += 1
                total += 1
                records.append(
                    {
                        "true": batch_names[i],
                        "pred": pred_label,
                        "conf": probs[i].item() * 100,
                        "status": "OK" if is_correct else "FAIL",
                    }
                )

    accuracy = (correct / total * 100) if total > 0 else 0.0

    preview_lines = []
    if records:
        sampled_indices = random.Random(seed).sample(
            range(len(records)), k=min(preview_count, len(records))
        )
        for rank, idx in enumerate(sampled_indices, 1):
            r = records[idx]
            preview_lines.append(
                f"  [{rank:02d}] True: {r['true']:>2} | "
                f"Pred: {r['pred']:>2} | "
                f"Conf: {r['conf']:5.1f}% | "
                f"{r['status']}"
            )

    return accuracy, correct, total, preview_lines


def predict_single_image(
    model: torch.nn.Module, image_path: Path, device: torch.device
) -> tuple[str, float]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob, pred_id = torch.softmax(logits, dim=1).max(dim=1)
    return ID_TO_CLASS[pred_id.item()], prob.item() * 100


def save_and_display(lines: list[str], output_path: Path) -> None:
    text = "\n".join(lines) + "\n"
    print(text, end="")
    output_path.write_text(text, encoding="utf-8")
