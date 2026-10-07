"""
src/hand_roi_audit.py
Task 2.2, Part A (revised): detection feasibility on a stratified 720-image sample.
Uses roi_detector.detect (single hand: MediaPipe -> skin -> none), the same code as Part B.

Read-only on data/raw/. Writes:
  data/processed/hand_roi_sample.csv
  docs/task2_2_partA_detection.json

Run (from repository root): uv run python src/hand_roi_audit.py
"""

import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CLASSES, DOCS_DIR, PROCESSED_DIR, PROJECT_ROOT
from roi_detector import (EDGE_PX, IMG_SIZE, MAX_NUM_HANDS, MAX_SKIN_AREA_FRAC,
                          MIN_SKIN_AREA_FRAC, MP_MIN_DETECTION_CONF,
                          describe_box, detect, make_hands)

SEED = 42
PER_VOLUNTEER_PER_CLASS = 2
INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
SAMPLE_CSV = PROCESSED_DIR / "hand_roi_sample.csv"
REPORT = DOCS_DIR / "task2_2_partA_detection.json"
METHODS = ["mediapipe", "skin", "none"]


def draw_sample(df):
    parts = [g.sample(PER_VOLUNTEER_PER_CLASS, random_state=SEED)
             for _, g in df.groupby(["label", "volunteer"])]
    return pd.concat(parts, ignore_index=True)


def detect_all(sample):
    rows = []
    with make_hands() as hands:
        for _, r in tqdm(sample.iterrows(), total=len(sample), desc="Detecting", unit="img"):
            bgr = cv2.imread(str(PROJECT_ROOT / r["filepath"]))
            if bgr is None:
                raise FileNotFoundError(f"cv2 could not read {r['filepath']}")
            box, method = detect(bgr, hands)
            row = {"filepath": r["filepath"], "label": r["label"],
                   "volunteer": int(r["volunteer"]), "std_L": float(r["std_L"]),
                   "method": method}
            if box is not None:
                side, area_frac, touches = describe_box(box)
                row.update(x0=box[0], y0=box[1], x1=box[2], y1=box[3], side=side,
                           area_frac=round(area_frac, 4), touches_border=touches)
            else:
                row.update(x0=None, y0=None, x1=None, y1=None, side=None,
                           area_frac=None, touches_border=None)
            rows.append(row)
    return pd.DataFrame(rows)


def q(series):
    s = series.dropna().astype(float)
    if s.empty:
        return {}
    return {"p5": float(s.quantile(0.05)), "median": float(s.median()),
            "p95": float(s.quantile(0.95))}


def summarise(res):
    n = len(res)
    counts = res["method"].value_counts().reindex(METHODS, fill_value=0)
    per_class = pd.crosstab(res["label"], res["method"]).reindex(
        index=CLASSES, columns=METHODS, fill_value=0)
    per_class["fallback_rate"] = (per_class["skin"] + per_class["none"]) / per_class[METHODS].sum(axis=1)
    per_vol = pd.crosstab(res["volunteer"], res["method"]).reindex(columns=METHODS, fill_value=0)
    det = res[res["method"] != "none"].copy()
    det["touches_border"] = det["touches_border"].astype(float)
    touch = det.groupby("method")["touches_border"].mean() if len(det) else pd.Series(dtype=float)
    none_rows = res[res["method"] == "none"][["filepath", "label", "volunteer", "std_L"]]
    low_std = res.nsmallest(10, "std_L")[["filepath", "label", "volunteer", "std_L", "method"]]
    return {
        "n": n,
        "method_counts": counts.to_dict(),
        "method_rates": {m: round(counts[m] / n, 4) for m in METHODS},
        "per_class": per_class.round(3).to_dict(orient="index"),
        "per_volunteer": per_vol.to_dict(orient="index"),
        "box_side_px": q(res["side"]),
        "box_area_frac": q(res["area_frac"]),
        "touches_border_rate_by_method": {k: round(float(v), 4) for k, v in touch.items()},
        "no_hand_frames": none_rows.to_dict(orient="records"),
        "lowest_std_L_images": low_std.to_dict(orient="records"),
    }


def print_summary(s):
    print(f"\nSample size: {s['n']}")
    print("Method counts:", s["method_counts"])
    print("Method rates :", s["method_rates"])
    pc = pd.DataFrame(s["per_class"]).T
    print("\nLowest MediaPipe rate by class:")
    print(pc.sort_values("mediapipe").head(8)[["mediapipe", "skin", "none"]].to_string())
    print("\nBy volunteer:")
    print(pd.DataFrame(s["per_volunteer"]).T.to_string())
    print("\nBox side (px):", {k: round(v, 1) for k, v in s["box_side_px"].items()})
    print("Box area fraction:", {k: round(v, 3) for k, v in s["box_area_frac"].items()})
    print("Touches frame edge, by method:", s["touches_border_rate_by_method"])
    print(f"\nNo-hand frames (method = none): {len(s['no_hand_frames'])}")
    for r in s["no_hand_frames"]:
        print(f"  {r['filepath']}  std_L={r['std_L']:.1f}")


def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.2 PART A (HAND DETECTION AUDIT, SINGLE HAND)")
    print("=" * 70)

    df = pd.read_csv(INDEX_CSV, dtype={"md5": str, "dhash": str})
    sample = draw_sample(df)
    print(f"Sample: {len(sample)} images ({PER_VOLUNTEER_PER_CLASS} per volunteer per class)")

    res = detect_all(sample)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(SAMPLE_CSV, index=False)
    s = summarise(res)
    print_summary(s)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "per_volunteer_per_class": PER_VOLUNTEER_PER_CLASS,
        "img_size": IMG_SIZE,
        "edge_px": EDGE_PX,
        "min_skin_area_frac": MIN_SKIN_AREA_FRAC,
        "max_skin_area_frac": MAX_SKIN_AREA_FRAC,
        "mediapipe": {"static_image_mode": True, "max_num_hands": MAX_NUM_HANDS,
                      "min_detection_confidence": MP_MIN_DETECTION_CONF},
        "summary": s,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                      encoding="utf-8")
    print(f"\nReport: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()