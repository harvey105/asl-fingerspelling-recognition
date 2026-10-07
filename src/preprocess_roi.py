"""
src/preprocess_roi.py
,  (revised): hand crop, 20% margin, letterbox to 224x224, full dataset.

Every image receives exactly one method label. The label records WHY it was or was not
hand-cropped, so the report cannot conflate a fixed rule with a detection failure:

  mediapipe         MediaPipe single-hand box, 20% margin                    
  skin              YCrCb skin-fallback box (rejected if touching border or oversized)
  class_rule_full   class 0 only: full frame by a fixed class rule            
  undetected_full   non-zero class where MediaPipe and skin both failed: full frame 
  blank             listed in blank_frames.csv: not written                  

Prerequisite: data/processed/blank_frames.csv, written by src/blank_scan.py --threshold T.
Refuses to run if data/processed_crops already holds files (stale-output guard).

Per image the log also records margin_side and scale (= 224 / margin_side), the factor by
which the raw pixels are resampled.  uses scale to measure the class-0 size confound.

Read-only on data/raw/. Writes:
  data/processed_crops/<class folders>/*.png
  data/processed/roi_crop_log.csv
  docs/task2_2_partB_crop.json
  docs/figures/roi_crop_inspection.png

Run (from repository root): uv run python src/preprocess_roi.py
"""

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import (BLANK_FRAMES_CSV, CLASSES, CROPS_DIR, DOCS_DIR, NUM_VOLUNTEERS,
                    PROCESSED_DIR, PROJECT_ROOT)
from roi_detector import (IMG_SIZE, MAX_NUM_HANDS, MP_MIN_DETECTION_CONF, describe_box,
                          detect, make_hands)

SEED = 42
MARGIN_FRAC = 0.20                       # 
CROP_SIZE = 224
PAD_VALUE = (128, 128, 128)              # neutral gray, identical in BGR and RGB
NUM_WORKERS = 8
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 3]
FULL_BOX = (0, 0, IMG_SIZE - 1, IMG_SIZE - 1)
FULL_FRAME_CLASSES = {"0"}               # : two-handed sign, single-hand box clips a hand
WRITTEN_METHODS = ["mediapipe", "skin", "class_rule_full", "undetected_full"]

INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
LOG_CSV = PROCESSED_DIR / "roi_crop_log.csv"
REPORT = DOCS_DIR / "task2_2_partB_crop.json"
INSPECT_FIG = DOCS_DIR / "figures" / "roi_crop_inspection.png"
INSPECT_CLASSES = ["0", "5", "W", "B", "C", "H", "Y", "K", "U", "3"]


def add_margin(box, frac, size):
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) + 1
    m = int(round(side * frac))
    return (max(0, x0 - m), max(0, y0 - m), min(size - 1, x1 + m), min(size - 1, y1 + m))


def letterbox(crop_bgr, out_size, pad_value):
    h, w = crop_bgr.shape[:2]
    scale = out_size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_size, out_size, 3), pad_value, dtype=np.uint8)
    top = (out_size - new_h) // 2
    left = (out_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


_HANDS = None


def _init_worker():
    global _HANDS
    _HANDS = make_hands()


def process_one(task):
    filepath, label, volunteer, is_blank, out_root = task
    base = {"filepath": filepath, "label": label, "volunteer": volunteer}
    if is_blank:
        return {**base, "method": "blank"}

    bgr = cv2.imread(str(PROJECT_ROOT / filepath))
    if bgr is None:
        return {**base, "method": "error", "error": "cv2.imread failed"}

    if label in FULL_FRAME_CLASSES:
        box, method = None, "class_rule_full"          # rule, not a detection result
    else:
        box, method = detect(bgr, _HANDS)
        if box is None:
            method = "undetected_full"                  # detection failed; full frame kept

    if box is None:
        box = FULL_BOX
        side = area_frac = touches = None
    else:
        side, area_frac, touches = describe_box(box)

    x0, y0, x1, y1 = add_margin(box, MARGIN_FRAC, IMG_SIZE)
    margin_side = max(x1 - x0, y1 - y0) + 1
    out_img = letterbox(bgr[y0:y1 + 1, x0:x1 + 1], CROP_SIZE, PAD_VALUE)

    out_path = out_root / label / (Path(filepath).stem + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), out_img, PNG_PARAMS):
        return {**base, "method": "error", "error": "cv2.imwrite failed"}

    return {**base, "method": method,
            "box_side": side,
            "box_area_frac": None if area_frac is None else round(area_frac, 4),
            "box_touches_border": touches,
            "margin_x0": x0, "margin_y0": y0, "margin_x1": x1, "margin_y1": y1,
            "margin_side": int(margin_side),
            "scale": round(CROP_SIZE / margin_side, 4)}


def summarise(res):
    n = len(res)
    counts = {k: int(v) for k, v in res["method"].value_counts().items()}
    written = res[res["method"].isin(WRITTEN_METHODS)].copy()
    written["scale"] = written["scale"].astype(float)

    det = written[written["method"].isin(["mediapipe", "skin"])]
    touch_by_method = {m: round(float(g["box_touches_border"].astype(float).mean()), 4)
                       for m, g in det.groupby("method")}
    scale_by_method = {m: round(float(g["scale"].median()), 4)
                       for m, g in written.groupby("method")}
    errors = (res.loc[res["method"] == "error", ["filepath", "error"]].to_dict("records")
              if "error" in res.columns else [])
    return {
        "n": n,
        "method_counts": counts,
        "method_rates": {k: round(v / n, 4) for k, v in counts.items()},
        "written": int(len(written)),
        "box_touches_border_rate_by_method": touch_by_method,
        "median_scale_by_method": scale_by_method,
        "errors": errors,
    }


def inspection_grid(res, out):
    ok = res[res["method"].isin(WRITTEN_METHODS)]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.4))
    for ax, cls, vol in zip(axes.flatten(), INSPECT_CLASSES, range(1, NUM_VOLUNTEERS + 1)):
        ax.axis("off")
        pool = ok[(ok["label"] == cls) & (ok["volunteer"] == vol)]
        if pool.empty:
            ax.set_title(f"'{cls}'  P{vol}: no written sample", fontsize=9)
            continue
        row = pool.sample(1, random_state=SEED).iloc[0]
        png = CROPS_DIR / cls / (Path(row["filepath"]).stem + ".png")
        with Image.open(png) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"'{cls}'  P{vol}   [{row['method']}]", fontsize=10, weight="bold")
    fig.suptitle(" 20% margin, letterboxed 224x224 PNG crops, one seeded "
                 "sample per volunteer (method in brackets)", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    print("=" * 70)
    print("ASL-HG:  (HAND CROP, 20% MARGIN, LETTERBOX, FULL DATASET)")
    print("=" * 70)

    if not BLANK_FRAMES_CSV.exists():
        raise FileNotFoundError(
            f"{BLANK_FRAMES_CSV} not found. Run: uv run python src/blank_scan.py --threshold 0.001")
    stale = [p for p in CROPS_DIR.rglob("*") if p.is_file()] if CROPS_DIR.exists() else []
    if stale:
        raise SystemExit(f"{CROPS_DIR} already holds {len(stale)} files from an earlier run. "
                         "Delete it first: Remove-Item -Recurse -Force data\\processed_crops")

    # Labels are read as strings so the digit '0' is never coerced to a number.
    df = pd.read_csv(INDEX_CSV, dtype={"filepath": str, "label": str})
    blank_set = set(pd.read_csv(BLANK_FRAMES_CSV, dtype={"filepath": str})["filepath"])
    print(f"Images: {len(df):,}   blank (excluded from crops): {len(blank_set):,}")
    print(f"Full-frame classes by rule : {sorted(FULL_FRAME_CLASSES)}")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(r.filepath, r.label, int(r.volunteer), r.filepath in blank_set, CROPS_DIR)
             for r in df.itertuples(index=False)]

    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_init_worker) as pool:
        for res in tqdm(pool.map(process_one, tasks, chunksize=50),
                        total=len(tasks), desc="Cropping", unit="img"):
            results.append(res)

    res_df = pd.DataFrame(results)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(LOG_CSV, index=False)

    s = summarise(res_df)
    print("\nMethod counts:", s["method_counts"])
    print("Method rates :", s["method_rates"])
    print("Border-touching rate by detector:", s["box_touches_border_rate_by_method"])
    print("Median scale by method:", s["median_scale_by_method"])
    print(f"Written to crops: {s['written']:,}   errors: {len(s['errors'])}")
    for e in s["errors"][:20]:
        print(f"  {e['filepath']}: {e['error']}")

    on_disk = sum(1 for _ in CROPS_DIR.rglob("*.png"))
    print(f"\nPNG files on disk: {on_disk:,} (log says {s['written']:,})")
    if on_disk != s["written"]:
        print("  [WARN] mismatch between log and disk; investigate before .")

    inspection_grid(res_df, INSPECT_FIG)
    print(f"Inspection figure: {INSPECT_FIG.relative_to(PROJECT_ROOT).as_posix()}")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "margin_frac": MARGIN_FRAC,
        "crop_size": CROP_SIZE,
        "pad_value": list(PAD_VALUE),
        "output_format": "png (lossless)",
        "resample": "bilinear (INTER_LINEAR)",
        "full_frame_classes": sorted(FULL_FRAME_CLASSES),
        "method_definitions": {
            "mediapipe": "single-hand MediaPipe box with margin",
            "skin": "YCrCb skin-fallback box with margin",
            "class_rule_full": "class-0 rule: full frame, no detection attempted",
            "undetected_full": "detector failed on a non-rule class: full frame kept",
            "blank": "excluded (blank_frames.csv)",
        },
        "mediapipe": {"static_image_mode": True, "max_num_hands": MAX_NUM_HANDS,
                      "min_detection_confidence": MP_MIN_DETECTION_CONF},
        "blank_frames_excluded": len(blank_set),
        "files_on_disk": on_disk,
        "summary": s,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                      encoding="utf-8")
    print(f"Report: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Per-image log: {LOG_CSV.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()