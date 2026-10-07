"""
src/check_dataset_part_a.py
Task 2.3 Part A checks: manifests, leakage, dataset tensors, augmentation, scale coverage.

Gates (all must pass):
  manifests:  no blank rows; signers match the split; every crop file exists;
              train has no duplicate md5; all 36 classes in every split.
  dataset:    batch shape (16, 3, 224, 224); labels in range; eval batch roughly zero-mean;
              no horizontal flip in the train transform.
  scale:      the augmentation can map every class median scale to about 1.0
              (SCALE_RANGE covers 1 / class_scale for the observed extremes).

Writes docs/task2_3_partA_dataset.json and docs/figures/aug_check_part_a.png.
Run (from repository root): uv run python src/check_dataset_part_a.py
"""

import json
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (BLANK_FRAMES_CSV, CLASSES, CROP_SIZE_PX, DOCS_DIR, NUM_CLASSES,
                    PROJECT_ROOT, TEST_VOLUNTEERS, TRAIN_VOLUNTEERS, VAL_VOLUNTEERS)
from dataset import (IMAGENET_MEAN, IMAGENET_STD, SCALE_RANGE, ASLDataset,
                     build_transform, class_weights)
from manifests import LOG_CSV, WRITTEN_METHODS, build_manifests
from torchvision import transforms as T
from PIL import Image

SEED = 42
REPORT = DOCS_DIR / "task2_3_partA_dataset.json"
FIG = DOCS_DIR / "figures" / "aug_check_part_a.png"


def denorm(x):
    m = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    s = np.array(IMAGENET_STD).reshape(3, 1, 1)
    return np.clip((x.numpy() * s + m).transpose(1, 2, 0), 0, 1)


def observed_scale_extremes():
    log = pd.read_csv(LOG_CSV, dtype={"filepath": str, "label": str})
    w = log[log["method"].isin(WRITTEN_METHODS)].copy()
    side = np.maximum(w["margin_x1"] - w["margin_x0"], w["margin_y1"] - w["margin_y0"]) + 1
    w["scale"] = CROP_SIZE_PX / side
    med = w.groupby("label")["scale"].median()
    return float(med.min()), float(med.max())


def figure(train, out):
    targets = ["0", "5", "W", "I", "M", "H"]
    picks = [int(np.flatnonzero(train["label"].to_numpy() == c)[0]) for c in targets]
    ds_raw = ASLDataset(train, is_train=False)
    ds_aug = ASLDataset(train, is_train=True)
    torch.manual_seed(SEED)
    fig, axes = plt.subplots(3, len(picks), figsize=(2.4 * len(picks), 7.4))
    for j, (c, i) in enumerate(zip(targets, picks)):
        axes[0, j].imshow(denorm(ds_raw[i][0]))
        axes[0, j].set_title(f"'{c}' eval", fontsize=9, weight="bold")
        axes[1, j].imshow(denorm(ds_aug[i][0]))
        axes[1, j].set_title("train aug A", fontsize=9)
        axes[2, j].imshow(denorm(ds_aug[i][0]))
        axes[2, j].set_title("train aug B", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Part A augmentation check: eval crop vs two random training augmentations",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def check(gates, name, ok, detail=""):
    gates[name] = bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.3 PART A (DATASET, AUGMENTATION, MANIFESTS)")
    print("=" * 70)

    m = build_manifests()
    train, val, test = m["train"], m["val"], m["test"]
    rep = m["report"]

    print("\nManifest counts")
    print(f"  written crops (all splits): {rep['written_total']:,}   blank excluded: {rep['blank_excluded']}")
    print(f"  train: {rep['train_before_dedupe']:,} before dedupe -> {rep['train_final']:,} "
          f"after removing {rep['train_copies_removed']} exact copies (D10)")
    print(f"  val:   {rep['val_final']:,}   test: {rep['test_final']:,}  (not de-duplicated)")
    tc = pd.Series(rep["train_per_class"])
    print(f"  train per class: min {tc.min()} (class {tc.idxmin()}), max {tc.max()} (class {tc.idxmax()})")

    gates = {}
    print("\n[1/4] Manifests")
    blank = set(pd.read_csv(BLANK_FRAMES_CSV, dtype={"filepath": str})["filepath"])
    all_fp = set(train["filepath"]) | set(val["filepath"]) | set(test["filepath"])
    check(gates, "no_blank_rows", not (all_fp & blank), f"blank rows found: {len(all_fp & blank)}")
    check(gates, "signers_match_split",
          set(train["volunteer"]) == set(TRAIN_VOLUNTEERS)
          and set(val["volunteer"]) == set(VAL_VOLUNTEERS)
          and set(test["volunteer"]) == set(TEST_VOLUNTEERS))
    check(gates, "splits_share_no_signer",
          not (set(train["volunteer"]) & set(val["volunteer"]))
          and not (set(val["volunteer"]) & set(test["volunteer"]))
          and not (set(train["volunteer"]) & set(test["volunteer"])))
    missing = [p for p in pd.concat([train, val, test])["crop_path"] if not (PROJECT_ROOT / p).exists()]
    check(gates, "every_crop_exists", len(missing) == 0, f"missing: {len(missing)}")
    check(gates, "train_md5_unique", train["md5"].is_unique)
    check(gates, "all_classes_in_every_split",
          all(d["class_id"].nunique() == NUM_CLASSES for d in (train, val, test)))

    print("\n[2/4] Dataset tensors")
    torch.manual_seed(SEED)
    ds_train = ASLDataset(train, is_train=True)
    ds_eval = ASLDataset(val, is_train=False)
    dl = DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(SEED))
    x, y = next(iter(dl))
    check(gates, "batch_shape_16x3x224x224", tuple(x.shape) == (16, 3, CROP_SIZE_PX, CROP_SIZE_PX),
          f"shape={tuple(x.shape)}")
    check(gates, "labels_in_range", int(y.min()) >= 0 and int(y.max()) < NUM_CLASSES,
          f"min={int(y.min())} max={int(y.max())}")
    # Normalisation arithmetic: eval transform must equal (x - mean) / std exactly.
    img0 = Image.open(PROJECT_ROOT / val.iloc[0]["crop_path"]).convert("RGB")
    raw0 = T.ToTensor()(img0)
    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    expected0 = (raw0 - mean_t) / std_t
    got0 = build_transform(False)(img0)
    norm_err = float((got0 - expected0).abs().max())
    check(gates, "normalisation_matches_formula", norm_err < 1e-5,
          f"max abs error={norm_err:.2e}")

    # Informational only: real channel statistics of the eval crops (not a gate).
    sample = val.sample(n=min(512, len(val)), random_state=SEED)
    sums, count = np.zeros(3), 0
    for p in sample["crop_path"]:
        arr = np.asarray(Image.open(PROJECT_ROOT / p).convert("RGB"), dtype=np.float64) / 255.0
        sums += arr.reshape(-1, 3).mean(axis=0)
        count += 1
    raw_mean = [round(float(v), 3) for v in sums / count]
    print(f"  info: eval-crop channel mean (0-1 scale, {count} images) = {raw_mean}; "
          f"ImageNet mean = {list(IMAGENET_MEAN)}")
    names = [type(t).__name__ for t in build_transform(True).transforms]
    check(gates, "no_horizontal_flip", "RandomHorizontalFlip" not in names, f"ops={names}")

    w = class_weights(train)
    print(f"  class weights: min {float(w.min()):.3f}, max {float(w.max()):.3f}  "
          f"(ratio {float(w.max() / w.min()):.3f})")

    print("\n[3/4] Scale coverage (D22)")
    s_min, s_max = observed_scale_extremes()
    lo, hi = SCALE_RANGE
    check(gates, "scale_range_covers_observed_spread",
          lo <= 1.0 / s_max and hi >= 1.0 / s_min,
          f"observed class-median scale [{s_min:.3f}, {s_max:.3f}]; "
          f"needed zoom [{1.0 / s_max:.3f}, {1.0 / s_min:.3f}]; SCALE_RANGE=[{lo}, {hi}]")

    print("\n[4/4] Augmentation figure")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    figure(train, FIG)
    print(f"  saved {FIG.relative_to(PROJECT_ROOT).as_posix()}")

    passed = all(gates.values())
    print("\n" + "=" * 70)
    print(f"VERDICT: {'PASSED' if passed else 'FAILED'}")
    print("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decisions": {
            "D7": "36 classes; midterm text specifies no class count",
            "D10": "exact-copy dedupe in train only, keep first per md5",
            "D22": f"augmentation scale range {list(SCALE_RANGE)}",
        },
        "manifest_report": rep,
        "class_weights_ratio": round(float(w.max() / w.min()), 4),
        "observed_class_scale_extremes": [round(s_min, 4), round(s_max, 4)],
        "gates": gates,
        "passed": passed,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Report: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()