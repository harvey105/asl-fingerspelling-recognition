"""
src/diagnose_external_scale.py
Diagnostic only. Lists external photos whose letterbox scale falls outside the training p1-p99
range, and says which direction each one is off.

Reads:
  data/processed/external_crop_log.csv   (written by prepare_external.py before its gates)
  data/processed/roi_crop_log.csv        (training reference, Task 2.2)
Writes nothing.

Run (from repository root): uv run python src/diagnose_external_scale.py
"""

import numpy as np
import pandas as pd

import preprocess_roi as pr
from config import PROCESSED_DIR

LOG_EXT = PROCESSED_DIR / "external_crop_log.csv"


def main():
    print("=" * 70)
    print("ASL-HG: EXTERNAL SCALE DIAGNOSTIC (DIAGNOSTIC ONLY, NOTHING IS CHANGED)")
    print("=" * 70)

    tl = pd.read_csv(pr.LOG_CSV, dtype={"label": str})
    tw = tl[tl["method"].isin(pr.WRITTEN_METHODS)]
    side = np.maximum(tw["margin_x1"] - tw["margin_x0"], tw["margin_y1"] - tw["margin_y0"]) + 1
    train_scale = pr.CROP_SIZE / side
    lo, hi = np.percentile(train_scale, [1, 99])
    print(f"Training scale p1 / p50 / p99: {lo:.4f} / {np.median(train_scale):.4f} / {hi:.4f}")

    ext = pd.read_csv(LOG_EXT, dtype={"label": str, "rel_path": str})
    ext = ext[ext["method"].isin(pr.WRITTEN_METHODS)].copy()
    print(f"External written crops: {len(ext)}   median scale: {ext['scale'].median():.4f}")

    below = ext[ext["scale"] < lo]
    above = ext[ext["scale"] > hi]
    inside = len(ext) - len(below) - len(above)
    print(f"\nInside range : {inside} ({inside / len(ext):.1%})")
    print(f"Below p1     : {len(below)}   -> hand fills MORE of the frame than training (too close)")
    print(f"Above p99    : {len(above)}   -> hand is SMALLER than training (too far)")

    out = ext[(ext["scale"] < lo) | (ext["scale"] > hi)].sort_values("scale")
    if out.empty:
        print("\nNo photo is outside the range.")
        return
    print("\nOutside range, sorted by scale:")
    for r in out.itertuples(index=False):
        side_txt = "too close" if r.scale < lo else "too far"
        print(f"  {r.scale:.4f}  {side_txt:<9}  {r.rel_path}")

    print("\nOutside range, by class:")
    print(out["label"].value_counts().sort_index().to_string())

    print("\nBoundary check (within 1e-3 of p1 or p99 counts as borderline):")
    border = ext[((ext["scale"] - lo).abs() < 1e-3) | ((ext["scale"] - hi).abs() < 1e-3)]
    print(f"  borderline photos: {len(border)}")
    print("=" * 70)


if __name__ == "__main__":
    main()