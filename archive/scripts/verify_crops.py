"""
src/verify_crops.py
Task 2.2, Part C (revised): verification, scale diagnostics, and review figures.

GATED (must pass):
  1. Set reconciliation: 36,000 log rows; written rows == PNG files on disk; no blank on
     disk; per-class PNG counts == 1,000 minus blanks; no error rows.
  2. Format: every PNG decodes as 224x224, uint8, 3 channels.
  3. Letterbox geometry: every pixel outside the placed content equals PAD_VALUE exactly,
     and the content is not uniform padding.
  4. Method rule: class_rule_full rows are exactly the full-frame-class rows.

REPORTED (not gated):
  5. Scale by class: the resampling factor (224 / margin side). Quantifies the class-0 size
     confound (D19) that Task 3.5 must test.
  6. Partial-entry flags: detected boxes touching the frame border, and boxes upscaled
     more than 2x.
  7. Review figures in docs/figures/ (each titled with what it actually shows).

Read-only on data/raw/ and data/processed_crops/. Writes:
  docs/task2_2_partC_verify.json
  docs/figures/crop_review_class0.png
  docs/figures/crop_review_wide_poses.png
  docs/figures/crop_review_full_frames.png
  docs/figures/crop_review_lowest_edge_nonblank.png

Run (from repository root): uv run python src/verify_crops.py
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (BLANK_FRAMES_CSV, CLASSES, CROPS_DIR, DOCS_DIR, IMAGES_PER_CLASS,
                    NUM_VOLUNTEERS, PROCESSED_DIR, PROJECT_ROOT, TOTAL_IMAGES)
from preprocess_roi import CROP_SIZE, FULL_FRAME_CLASSES, PAD_VALUE, WRITTEN_METHODS

SEED = 42
LOG_CSV = PROCESSED_DIR / "roi_crop_log.csv"
SCAN_CSV = PROCESSED_DIR / "blank_scan.csv"
REPORT = DOCS_DIR / "task2_2_partC_verify.json"
FIG_DIR = DOCS_DIR / "figures"
NUM_WORKERS = 8
PAD_BGR = np.array(PAD_VALUE, dtype=np.uint8)
WIDE_POSES = [("5", 2), ("5", 6), ("W", 3), ("W", 7), ("W", 9), ("0", 9)]
UPSCALE_FLAG = 2.0                       # a crop enlarged more than 2x is flagged as soft
N_LOWEST = 10
N_UNDETECTED_SHOWN = 16


def placement(w, h):
    """Same arithmetic as preprocess_roi.letterbox. Returns the placed content rectangle."""
    scale = CROP_SIZE / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    top, left = (CROP_SIZE - nh) // 2, (CROP_SIZE - nw) // 2
    return nw, nh, top, left


def check_png(task):
    rel, label, stem, x0, y0, x1, y1 = task
    out = {"filepath": rel, "file_ok": False, "pad_ok": False,
           "content_uniform": False, "content_frac": None, "error": None}
    img = cv2.imread(str(CROPS_DIR / label / (stem + ".png")), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint8 or img.shape != (CROP_SIZE, CROP_SIZE, 3):
        out["error"] = "missing or wrong shape/dtype"
        return out
    out["file_ok"] = True
    w, h = x1 - x0 + 1, y1 - y0 + 1
    nw, nh, top, left = placement(w, h)
    mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=bool)
    mask[top:top + nh, left:left + nw] = True
    out["pad_ok"] = bool(np.all(img[~mask] == PAD_BGR))
    out["content_uniform"] = bool(np.all(img[mask] == PAD_BGR))
    out["content_frac"] = float(nw * nh / CROP_SIZE ** 2)
    return out


def load_raw(rel):
    return cv2.cvtColor(cv2.imread(str(PROJECT_ROOT / rel)), cv2.COLOR_BGR2RGB)


def load_crop(label, rel):
    p = CROPS_DIR / label / (Path(rel).stem + ".png")
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def pair_grid(rows, out, title):
    if rows.empty:
        print(f"  [skip] no rows for {out.name}")
        return
    per_row = 2
    nrows = int(np.ceil(len(rows) / per_row))
    fig, axes = plt.subplots(nrows, per_row * 2, figsize=(per_row * 2 * 2.6, nrows * 2.9))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, r in enumerate(rows.itertuples(index=False)):
        i, j = divmod(k, per_row)
        axes[i, 2 * j].imshow(load_raw(r.filepath))
        axes[i, 2 * j].set_title(f"raw '{r.label}' P{int(r.volunteer)}", fontsize=8)
        axes[i, 2 * j + 1].imshow(load_crop(r.label, r.filepath))
        axes[i, 2 * j + 1].set_title(f"crop [{r.method}]", fontsize=8)
    fig.suptitle(title, fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def crop_grid(rows, out, title, ncols=5):
    if rows.empty:
        print(f"  [skip] no rows for {out.name}")
        return
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.9))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, r in enumerate(rows.itertuples(index=False)):
        axes.flat[k].imshow(load_crop(r.label, r.filepath))
        axes.flat[k].set_title(f"'{r.label}' P{int(r.volunteer)} [{r.method}]", fontsize=8)
    fig.suptitle(title, fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def sample_pairs(written, pairs):
    picked = []
    for label, vol in pairs:
        pool = written[(written["label"] == label) & (written["volunteer"] == vol)]
        if len(pool):
            picked.append(pool.sample(1, random_state=SEED).iloc[0])
    return pd.DataFrame(picked)


def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.2 PART C (CROP VERIFICATION, SCALE, AND REVIEW)")
    print("=" * 70)

    log = pd.read_csv(LOG_CSV, dtype={"filepath": str, "label": str})
    blank = pd.read_csv(BLANK_FRAMES_CSV, dtype={"filepath": str, "label": str})
    scan = pd.read_csv(SCAN_CSV, dtype={"filepath": str, "label": str})
    blank_set = set(blank["filepath"])

    written = log[log["method"].isin(WRITTEN_METHODS)].reset_index(drop=True).copy()
    mx0, my0 = written["margin_x0"].astype(int), written["margin_y0"].astype(int)
    mx1, my1 = written["margin_x1"].astype(int), written["margin_y1"].astype(int)
    written["margin_side"] = np.maximum(mx1 - mx0, my1 - my0) + 1
    written["scale"] = CROP_SIZE / written["margin_side"]
    is_c0 = written["label"].isin(FULL_FRAME_CLASSES)

    # 1. Set reconciliation
    print("\n[1/5] Set reconciliation")
    expected_keys = {(l, Path(f).stem) for f, l in zip(written["filepath"], written["label"])}
    disk = {(p.parent.name, p.stem) for p in CROPS_DIR.rglob("*.png")}
    missing = sorted(expected_keys - disk)
    extra = sorted(disk - expected_keys)
    blank_keys = {(l, Path(f).stem) for f, l in zip(blank["filepath"], blank["label"])}
    blank_on_disk = len(disk & blank_keys)
    per_cls_written = written["label"].value_counts().reindex(CLASSES, fill_value=0)
    per_cls_expected = (IMAGES_PER_CLASS
                        - blank["label"].value_counts().reindex(CLASSES, fill_value=0))
    class_mismatch = {c: [int(per_cls_written[c]), int(per_cls_expected[c])] for c in CLASSES
                      if per_cls_written[c] != per_cls_expected[c]}
    n_error = int((log["method"] == "error").sum())
    print(f"  log rows: {len(log):,}   written: {len(written):,}   blank: {len(blank_set):,}")
    print(f"  method counts: {log['method'].value_counts().to_dict()}")
    print(f"  PNG files on disk: {len(disk):,}   missing: {len(missing)}   extra: {len(extra)}")
    print(f"  blank frames found on disk: {blank_on_disk}   error rows: {n_error}")
    print(f"  per-class mismatches: {class_mismatch if class_mismatch else 'none'}")

    # 2-3. Per-file format and letterbox geometry (decodes every PNG)
    print("\n[2/5] Format and letterbox geometry per file (decodes every PNG)")
    tasks = [(r.filepath, r.label, Path(r.filepath).stem, int(r.margin_x0), int(r.margin_y0),
              int(r.margin_x1), int(r.margin_y1)) for r in written.itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        checks = pd.DataFrame(list(tqdm(pool.map(check_png, tasks), total=len(tasks),
                                        desc="Checking PNGs", unit="img")))
    n_file_bad = int((~checks["file_ok"]).sum())
    n_pad_bad = int((checks["file_ok"] & ~checks["pad_ok"]).sum())
    n_uniform = int((checks["file_ok"] & checks["content_uniform"]).sum())
    print(f"  files not 224x224 uint8 3-channel : {n_file_bad}")
    print(f"  files with non-exact padding      : {n_pad_bad}")
    print(f"  files with uniform-padding content: {n_uniform}")

    # 4. Method rule
    print("\n[3/5] Method rule (class 0 rule applied exactly)")
    c0_rows = written[is_c0]
    rule_rows = written[written["method"] == "class_rule_full"]
    rule_only_c0 = bool(rule_rows["label"].isin(FULL_FRAME_CLASSES).all())
    all_c0_rule = bool((c0_rows["method"] == "class_rule_full").all())
    print(f"  full-frame-class rows: {len(c0_rows):,}   class_rule_full rows: {len(rule_rows):,}")
    print(f"  rule applies only to full-frame classes: {rule_only_c0}")
    print(f"  every full-frame-class row uses the rule: {all_c0_rule}")
    undet = written[written["method"] == "undetected_full"]
    print(f"  undetected_full rows (detector failed on non-rule classes): {len(undet)}")
    if len(undet):
        print(undet[["filepath", "label", "volunteer"]].to_string(index=False))

    # 5. Scale by class: the size confound
    print("\n[4/5] Scale (resampling factor, 224 / margin side) by class")
    scale_by_class = written.groupby("label")["scale"].median().reindex(CLASSES)
    cells = [f"{c}:{scale_by_class[c]:.3f}" for c in CLASSES]
    for i in range(0, len(cells), 9):
        print("  " + "  ".join(f"{x:<11}" for x in cells[i:i + 9]))
    c0_scale = float(written.loc[is_c0, "scale"].median())
    rest_scale = float(written.loc[~is_c0, "scale"].median())
    print(f"  class 0 median scale: {c0_scale:.3f}   other classes: {rest_scale:.3f}   "
          f"ratio: {c0_scale / rest_scale:.3f}")
    print("  (ratio < 1: class-0 hands are drawn smaller than other signs; see D19)")

    # 6. Partial-entry flags
    print("\n[5/5] Partial-entry flags (detected crops only)")
    det = written[written["method"].isin(["mediapipe", "skin"])]
    touch = det["box_touches_border"].map(lambda v: str(v) == "True")
    n_touch = int(touch.sum())
    soft = det[det["scale"] > UPSCALE_FLAG]
    print(f"  detected boxes touching the frame border: {n_touch:,} of {len(det):,}")
    print(f"  detected crops enlarged more than {UPSCALE_FLAG}x: {len(soft):,}")
    if len(soft):
        print(f"  by class (top 8): {soft['label'].value_counts().head(8).to_dict()}")
    low_edge = scan[~scan["filepath"].isin(blank_set)].nsmallest(N_LOWEST, "edge_frac_40")
    print("  lowest edge_frac_40 among non-blank frames (kept):")
    print(low_edge[["filepath", "label", "volunteer", "edge_frac_40"]].to_string(index=False))

    # Review figures (titles state exactly what each figure shows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pair_grid(sample_pairs(written, [("0", v) for v in range(1, NUM_VOLUNTEERS + 1)]),
              FIG_DIR / "crop_review_class0.png",
              "Class 0 (rule: full frame, D19): raw | 224 letterbox crop, one seeded sample per volunteer")
    pair_grid(sample_pairs(written, WIDE_POSES),
              FIG_DIR / "crop_review_wide_poses.png",
              "Wide poses (5, W, 0): raw | crop, checks for clipped fingertips")

    undet_show = undet.sort_values(["label", "volunteer"]).head(N_UNDETECTED_SHOWN)
    c0_show = (written[written["method"] == "class_rule_full"]
               .groupby("volunteer").sample(1, random_state=SEED))
    full_show = pd.concat([undet_show, c0_show]).reset_index(drop=True)
    crop_grid(full_show, FIG_DIR / "crop_review_full_frames.png",
              f"Full-frame crops: {len(undet_show)} undetected (non-zero classes; "
              f"all {len(undet)} shown if fewer than 16) + {len(c0_show)} class-0 rule "
              f"crops (one per volunteer)", ncols=5)

    low_rows = (low_edge.merge(written[["filepath", "method"]], on="filepath", how="inner")
                .sort_values("edge_frac_40"))
    pair_grid(low_rows, FIG_DIR / "crop_review_lowest_edge_nonblank.png",
              "Lowest-edge non-blank frames: raw | crop (partial-entry check)")
    print(f"\n  figures written to {FIG_DIR.relative_to(PROJECT_ROOT).as_posix()}/")

    # Verdict and report
    gating = {
        "total_rows_is_36000": len(log) == TOTAL_IMAGES,
        "no_error_rows": n_error == 0,
        "no_missing_files": len(missing) == 0,
        "no_extra_files": len(extra) == 0,
        "no_blank_on_disk": blank_on_disk == 0,
        "per_class_counts_match": len(class_mismatch) == 0,
        "all_files_224x224_rgb": n_file_bad == 0,
        "all_padding_exact": n_pad_bad == 0 and n_uniform == 0,
        "rule_only_on_full_frame_classes": rule_only_c0,
        "every_full_frame_class_row_uses_rule": all_c0_rule,
    }
    passed = all(gating.values())
    print("\nGATES")
    for k, v in gating.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nVERDICT: {'PASSED' if passed else 'FAILED'}")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "crop_size": CROP_SIZE,
        "pad_value": list(PAD_VALUE),
        "full_frame_classes": sorted(FULL_FRAME_CLASSES),
        "gating": gating,
        "passed": passed,
        "counts": {"log_rows": len(log), "written": len(written), "blank": len(blank_set),
                   "png_on_disk": len(disk), "missing": len(missing), "extra": len(extra),
                   "blank_on_disk": blank_on_disk, "error_rows": n_error,
                   "class_mismatch": class_mismatch},
        "method_counts": log["method"].value_counts().to_dict(),
        "file_checks": {"not_224x224_rgb": n_file_bad, "non_exact_padding": n_pad_bad,
                        "uniform_content": n_uniform},
        "scale": {"by_class_median": {c: round(float(scale_by_class[c]), 4) for c in CLASSES},
                  "class0_median": round(c0_scale, 4),
                  "other_classes_median": round(rest_scale, 4),
                  "class0_to_other_ratio": round(c0_scale / rest_scale, 4)},
        "partial_entry": {"detected_touching_border": n_touch,
                          "detected_total": int(len(det)),
                          "detected_upscaled_over_2x": int(len(soft)),
                          "upscaled_over_2x_by_class": soft["label"].value_counts().to_dict()},
        "undetected_full": undet[["filepath", "label", "volunteer"]].to_dict(orient="records"),
        "lowest_edge_nonblank": low_edge.to_dict(orient="records"),
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                      encoding="utf-8")
    print(f"Report: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()