"""
src/profile_transforms.py
Task 2.3 Part B diagnostic: per-stage cost of the training pipeline, single process.

Decides what to optimise. Reports the share of time spent in PNG decoding and in each
augmentation step, over a fixed seeded sample of training crops.

Writes nothing except the printed table.
Run (from repository root): uv run python src/profile_transforms.py
"""

import time

import numpy as np
from PIL import Image
from torchvision import transforms as T

from dataset import (FILL, JITTER, ROTATION_DEG, SCALE_RANGE, TRANSLATE,
                     IMAGENET_MEAN, IMAGENET_STD, KeepPadJitter)
from manifests import build_manifests
from config import PROJECT_ROOT

N = 300
SEED = 42


def main():
    train = build_manifests()["train"].sample(n=N, random_state=SEED)
    paths = [PROJECT_ROOT / p for p in train["crop_path"]]

    keep_pad = KeepPadJitter(JITTER, JITTER, JITTER)
    affine = T.RandomAffine(degrees=ROTATION_DEG, translate=TRANSLATE,
                            scale=SCALE_RANGE, fill=FILL)
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    times = {"png_decode": 0.0, "keep_pad_jitter": 0.0,
             "random_affine": 0.0, "to_tensor_normalise": 0.0}
    for p in paths:
        t = time.perf_counter()
        with Image.open(p) as im:
            img = im.convert("RGB")
        times["png_decode"] += time.perf_counter() - t

        t = time.perf_counter()
        img = keep_pad(img)
        times["keep_pad_jitter"] += time.perf_counter() - t

        t = time.perf_counter()
        img = affine(img)
        times["random_affine"] += time.perf_counter() - t

        t = time.perf_counter()
        to_tensor(img)
        times["to_tensor_normalise"] += time.perf_counter() - t

    total = sum(times.values())
    print("=" * 60)
    print(f"Per-image pipeline cost over {N} training crops (single process)")
    print("=" * 60)
    for k, v in times.items():
        print(f"  {k:<22} {1000 * v / N:7.2f} ms/img   {100 * v / total:5.1f} %")
    print(f"  {'total':<22} {1000 * total / N:7.2f} ms/img")
    print("=" * 60)
    print("Reading the table:")
    print("  png_decode dominant (>30%): a precomputed uint8 cache is the fix.")
    print("  keep_pad_jitter or random_affine dominant: cheapen those steps first.")


if __name__ == "__main__":
    main()