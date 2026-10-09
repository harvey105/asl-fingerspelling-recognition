from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.config import NUM_CLASSES, PROJECT_ROOT

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
FILL: tuple[int, int, int] = (128, 128, 128)
ROTATION_DEG: int = 12
TRANSLATE: tuple[float, float] = (0.06, 0.06)
SCALE_RANGE: tuple[float, float] = (0.75, 1.35)
JITTER: float = 0.25


class KeepPadJitter:
    """ColorJitter transform that leaves neutral gray padding pixels unchanged."""

    def __init__(
        self, brightness: float, contrast: float, saturation: float
    ) -> None:
        self.jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img)
        pad_mask = np.all(arr == np.array(FILL, dtype=arr.dtype), axis=2)
        jittered = np.asarray(self.jitter(img)).copy()
        jittered[pad_mask] = FILL
        return Image.fromarray(jittered)


def build_transform(is_train: bool) -> T.Compose:
    """Construct data transformation pipeline for training or evaluation."""
    transforms_list = []
    if is_train:
        transforms_list.extend(
            [
                KeepPadJitter(
                    brightness=JITTER, contrast=JITTER, saturation=JITTER
                ),
                T.RandomAffine(
                    degrees=ROTATION_DEG,
                    translate=TRANSLATE,
                    scale=SCALE_RANGE,
                    fill=FILL,
                ),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
    transforms_list.extend(
        [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return T.Compose(transforms_list)


class ASLDataset(Dataset):
    """PyTorch Dataset loading preprocessed ASL hand crop images."""

    def __init__(self, manifest: pd.DataFrame, is_train: bool) -> None:
        self.paths: list[Path] = [
            PROJECT_ROOT / p for p in manifest["crop_path"]
        ]
        self.labels: np.ndarray = manifest["class_id"].to_numpy(dtype=np.int64)
        self.transform: T.Compose = build_transform(is_train)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as img_file:
            img = img_file.convert("RGB")
        return self.transform(img), int(self.labels[index])


def class_weights(manifest: pd.DataFrame) -> torch.Tensor:
    """Calculate inverse-frequency class weights for loss rebalancing."""
    counts = (
        manifest["class_id"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .astype(float)
    )
    if (counts == 0).any():
        raise ValueError("Manifest has missing classes with zero samples.")
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights.to_numpy(), dtype=torch.float32)
