"""
src/blank_scan.py
Task 2.2, Part B-1: blank / no-hand frame scan over all 36,000 raw images.

Why edge density and not std_L alone: a blank pale background has low luminance spread,
but so does a frame with a fingertip at the edge, and a single-feature cutoff on std_L
flagged one frame in the Part A sample. Edge density is the fraction of pixels whose
Sobel gradient magnitude exceeds t. A plain background has almost no edges. A hand, even
partly in frame, has many (knuckles, nails, creases, outline).

Features per image (5x5 Gaussian blur, grayscale, Sobel ksize 3):
  edge_frac_20, edge_frac_40, edge_frac_80   fraction of pixels with |grad| > t
  std_L                                       from the index

Iterative workflow (threshold is chosen by review, not guessed):
  1. uv run python src/blank_scan.py
       Prints quantiles, counts at candidate thresholds, and natural breaks in the lowest
       500 values. Writes docs/figures/blank_scan_histogram.png and blank_scan_review.png
       (the 48 lowest-edge frames, ascending).
  2. Inspect the review grid. Find the rank where frames stop being empty.
  3. Refine the band: uv run python src/blank_scan.py --grid-start 40
  4. Apply: uv run python src/blank_scan.py --threshold T --mediapipe-check
       Writes data/processed/blank_frames.csv. preprocess_roi.py requires this file.
       --mediapipe-check runs MediaPipe on frames with edge_frac_40 <= 0.02, as a cross-check:
       blanks should show no hand; a hand found there means a partial-entry frame.

Read-only on data/raw/. Caches features in data/processed/blank_scan.csv.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import BLANK_FRAMES_CSV, CLASSES, DOCS_DIR, PROCESSED_DIR, PROJECT_ROOT
from roi_detector import make_hands, mediapipe_box

INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
SCAN_CSV = PROCESSED_DIR / "blank_scan.csv"
FIG_DIR = DOCS_DIR / "figures"
HIST_FIG = FIG_DIR / "blank_scan_histogram.png"
GRID_FIG = FIG_DIR / "blank_scan_review.png"
REPORT = DOCS_DIR / "task2_2_partB1_blank_scan.json"
EDGE_MAGS = [20, 40, 80]
PRIMARY_MAG = 40
CHECK_THRESHOLDS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
MP_CHECK_CEILING = 0.02
GRID_N, GRID_COLS = 48, 8
NUM_WORKERS = 8


def edge_features(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cv2 could not read {path}")
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return {f"edge_frac_{t}": float((mag > t).mean()) for t in EDGE_MAGS}


def compute_features(df):
    if SCAN_CSV.exists():
        cached = pd.read_csv(SCAN_CSV, usecols=["filepath"])
        if set(cached["filepath"]) == set(df["filepath"]):
            print(f"  [CACHE] Reusing {SCAN_CSV.name}")
            return pd.read_csv(SCAN_CSV)
        print("  [CACHE] Stale (file list changed); recomputing.")

    paths = [PROJECT_ROOT / f for f in df["filepath"]]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        feats = list(tqdm(pool.map(edge_features, paths), total=len(paths),
                          desc="Edge scan", unit="img"))
    out = pd.concat([df[["filepath", "label", "volunteer", "std_L"]].reset_index(drop=True),
                     pd.DataFrame(feats)], axis=1)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCAN_CSV, index=False)
    return out


def natural_breaks(sorted_vals, n=500, top=5):
    """Largest relative jumps between consecutive sorted values among the lowest n."""
    v = np.maximum(sorted_vals[:n], 1e-5)
    ratios = v[1:] / v[:-1]
    best = np.argsort(ratios)[::-1][:top]
    return [{"between_ranks": [int(i + 1), int(i + 2)],
             "values": [float(v[i]), float(v[i + 1])],
             "ratio": round(float(ratios[i]), 2)}
            for i in sorted(best)]


def histogram(scan, col, threshold, out):
    vals = np.maximum(scan[col].to_numpy(), 1e-5)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(np.log10(vals), bins=90, color="#2b5c8f", edgecolor="black", linewidth=0.3)
    if threshold is not None:
        ax.axvline(np.log10(max(threshold, 1e-5)), color="crimson", linestyle="--",
                   label=f"blank threshold {threshold}")
        ax.legend(frameon=True)
    ax.set_title(f"Fraction of edge pixels per frame ({col}), all 36,000 images",
                 fontsize=11, weight="bold")
    ax.set_xlabel("log10(fraction of edge pixels), floored at 1e-5")
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def review_grid(ranked, start, col, out):
    sub = ranked.iloc[start:start + GRID_N]
    if sub.empty:
        print(f"  [grid] no rows at --grid-start {start}")
        return
    rows = int(np.ceil(len(sub) / GRID_COLS))
    fig, axes = plt.subplots(rows, GRID_COLS, figsize=(GRID_COLS * 1.9, rows * 2.3))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, (_, r) in enumerate(sub.iterrows()):
        ax = axes.flat[k]
        bgr = cv2.imread(str(PROJECT_ROOT / r["filepath"]))
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"#{start + k + 1} '{r['label']}' P{int(r['volunteer'])}\n{col}={r[col]:.4f}",
                     fontsize=7.5)
    fig.suptitle(f"Lowest-edge frames, ranks {start + 1} to {start + len(sub)} "
                 f"(ascending {col}). Blank = no hand in frame.", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def mediapipe_check(low):
    found = []
    with make_hands() as hands:
        for _, r in tqdm(low.iterrows(), total=len(low), desc="MediaPipe check", unit="img"):
            bgr = cv2.imread(str(PROJECT_ROOT / r["filepath"]))
            found.append(mediapipe_box(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hands) is not None)
    return np.array(found)


def main():
    ap = argparse.ArgumentParser(description="Blank / no-hand frame scan (Task 2.2 Part B-1).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="edge_frac_40 at or below which a frame is treated as blank")
    ap.add_argument("--grid-start", type=int, default=0,
                    help="0-based rank of the first frame shown in the review grid")
    ap.add_argument("--mediapipe-check", action="store_true",
                    help=f"run MediaPipe on frames with edge_frac_40 <= {MP_CHECK_CEILING}")
    args = ap.parse_args()

    print("=" * 70)
    print("ASL-HG: TASK 2.2 PART B-1 (BLANK / NO-HAND FRAME SCAN, ALL 36,000)")
    print("=" * 70)

    df = pd.read_csv(INDEX_CSV, dtype={"md5": str, "dhash": str})
    scan = compute_features(df)
    col = f"edge_frac_{PRIMARY_MAG}"
    ranked = scan.sort_values(col, kind="stable").reset_index(drop=True)
    v = ranked[col].to_numpy()

    quant = {str(q): float(np.quantile(v, q / 100)) for q in [0, 0.1, 0.5, 1, 2, 5, 10, 25, 50]}
    counts = {str(t): int((v <= t).sum()) for t in CHECK_THRESHOLDS}
    breaks = natural_breaks(v)

    print(f"\nQuantiles of {col} (percent of images):")
    for k, val in quant.items():
        print(f"  {k:>5}%: {val:.5f}")
    print("\nImages at or below candidate thresholds:")
    for t, n in counts.items():
        print(f"  <= {float(t):<7}: {n:>6}")
    print("\nNatural breaks in the lowest 500 values (largest relative jumps):")
    for b in breaks:
        print(f"  between ranks {b['between_ranks'][0]} and {b['between_ranks'][1]}: "
              f"{b['values'][0]:.5f} -> {b['values'][1]:.5f}  (ratio {b['ratio']})")

    print("\nLowest 20 frames (ascending):")
    print(ranked.head(20)[["filepath", "label", "volunteer", col, "std_L"]].to_string())

    histogram(scan, col, args.threshold, HIST_FIG)
    review_grid(ranked, args.grid_start, col, GRID_FIG)
    print(f"\nFigures: {HIST_FIG.name}, {GRID_FIG.name} (ranks {args.grid_start + 1} onward)")

    mp_summary = None
    if args.mediapipe_check:
        low = ranked[ranked[col] <= MP_CHECK_CEILING].reset_index(drop=True)
        found = mediapipe_check(low)
        hits = low[found]
        mp_summary = {"checked": int(len(low)), "hand_found": int(found.sum())}
        print(f"\nMediaPipe check on {len(low)} frames with {col} <= {MP_CHECK_CEILING}:")
        print(f"  hand found in {int(found.sum())} of them")
        if len(hits):
            print("  first frames with a hand (partial-entry, or a misfire):")
            print(hits.head(20)[["filepath", "label", "volunteer", col]].to_string())

    blank_summary = None
    if args.threshold is not None:
        blank = ranked[ranked[col] <= args.threshold][
            ["filepath", "label", "volunteer", col, "std_L"]].reset_index(drop=True)
        blank.to_csv(BLANK_FRAMES_CSV, index=False)
        per_class = blank["label"].value_counts().reindex(CLASSES, fill_value=0)
        per_vol = blank["volunteer"].value_counts().sort_index()
        blank_summary = {"threshold": args.threshold, "blank_count": int(len(blank)),
                         "per_class_nonzero": {k: int(n) for k, n in per_class.items() if n},
                         "per_volunteer": {int(k): int(n) for k, n in per_vol.items()}}
        print(f"\nBlank threshold {col} <= {args.threshold}")
        print(f"  blank frames: {len(blank)}  -> {BLANK_FRAMES_CSV.name}")
        print(f"  per class (nonzero): {blank_summary['per_class_nonzero']}")
        print(f"  per volunteer      : {blank_summary['per_volunteer']}")
        print("\nFirst 10 frames just above the threshold (kept; review these):")
        above = ranked.iloc[len(blank):len(blank) + 10]
        print(above[["filepath", "label", "volunteer", col]].to_string())

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "primary_feature": col,
        "edge_magnitudes": EDGE_MAGS,
        "quantiles_pct": quant,
        "counts_at_thresholds": counts,
        "natural_breaks": breaks,
        "mediapipe_check": mp_summary,
        "blank": blank_summary,
        "lowest_48": ranked.head(GRID_N)[["filepath", "label", "volunteer", col, "std_L"]]
                           .to_dict(orient="records"),
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()