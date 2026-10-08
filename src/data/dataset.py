"""
src/dataset.py
Augmentation pipeline and PyTorch Dataset over letterboxed crops.

Train transforms:
  1. KeepPadJitter: ColorJitter preserving neutral gray (128, 128, 128) letterbox padding.
  2. RandomAffine: rotation +-12 deg, translation 6%, scale zoom [0.75, 1.35], neutral gray fill.
  3. RandomHorizontalFlip(p=0.5): stochastic horizontal reflection for ambidextrous generalization.
  4. ToTensor, then ImageNet normalization.

Eval transforms:
  ToTensor and ImageNet normalization only.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.config import NUM_CLASSES, PROJECT_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FILL = (128, 128, 128)  # neutral gray matching letterbox padding
ROTATION_DEG = 12
TRANSLATE = (0.06, 0.06)
SCALE_RANGE = (0.75, 1.35)
JITTER = 0.25


class KeepPadJitter:
    """ColorJitter that leaves pixels equal to FILL unchanged."""

    def __init__(self, brightness, contrast, saturation):
        self.jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        )

    def __call__(self, img):
        arr = np.asarray(img)
        pad = np.all(arr == np.array(FILL, dtype=arr.dtype), axis=2)
        out = np.asarray(self.jitter(img)).copy()
        out[pad] = FILL
        return Image.fromarray(out)


def build_transform(is_train):
    ops = []
    if is_train:
        ops += [
            KeepPadJitter(brightness=JITTER, contrast=JITTER, saturation=JITTER),
            T.RandomAffine(
                degrees=ROTATION_DEG, translate=TRANSLATE, scale=SCALE_RANGE, fill=FILL
            ),
            T.RandomHorizontalFlip(p=0.5),
        ]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class ASLDataset(Dataset):
    """
    manifest: DataFrame with columns crop_path (relative to project root) and class_id.
    is_train: True applies the augmentation pipeline; False applies normalisation only.
    """

    def __init__(self, manifest, is_train):
        self.paths = [PROJECT_ROOT / p for p in manifest["crop_path"]]
        self.labels = manifest["class_id"].to_numpy(dtype=np.int64)
        self.transform = build_transform(is_train)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.labels[i])


def class_weights(manifest):
    """Inverse-frequency weights from the given (training) manifest, in class_id order."""
    counts = (
        manifest["class_id"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .astype(float)
    )
    if (counts == 0).any():
        raise ValueError("A class has no images in this manifest.")
    w = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(w.to_numpy(), dtype=torch.float32)
