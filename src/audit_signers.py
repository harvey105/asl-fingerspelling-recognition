"""
src/audit_signers.py
Task 2.1, Part B: signer and background audit for ASL-HG.

Measures:
  1. Byte-identical copies: do any span volunteers? (would leak across the subject split)
  2. Per-signer brightness and colour statistics
  3. Burst structure: consecutive-index dHash distance within a signer, versus
     random same-class pairs within a signer and across signers
  4. Background proxy: luminance of the outer 10% border, per signer

Read-only on data/raw/. Writes:
  data/processed/asl_hg_image_index.csv   (cache of per-image stats and hashes)
  docs/task2_1_partB_signers.json
  docs/figures/eda_signer_luminance.png
  docs/figures/eda_signer_hamming.png

Run (from the repository root): uv run python src/audit_signers.py
"""

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import (
    CLASSES,
    DOCS_DIR,
    IMAGE_EXTENSIONS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    find_dataset_root,
    parse_filename,
)

INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
REPORT = DOCS_DIR / "task2_1_partB_signers.json"
FIG_DIR = DOCS_DIR / "figures"
SEED = 42
BORDER_FRAC = 0.10
NUM_WORKERS = 8
PAIRS_PER_CLASS = 2000
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------- per-image stats
def describe(path):
    """One read per file: MD5 of the bytes, a 64-bit dHash, and luminance/colour statistics."""
    data = path.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
        gray_im = im.convert("L")
        gray = np.asarray(gray_im, dtype=np.float32)
        small = np.asarray(gray_im.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)

    bits = (small[:, 1:] > small[:, :-1]).flatten()          # 8 rows x 8 comparisons = 64 bits
    h, w = gray.shape
    b = max(1, int(round(min(h, w) * BORDER_FRAC)))
    inner = np.zeros((h, w), dtype=bool)
    inner[b:h - b, b:w - b] = True
    border = ~inner

    return {
        "md5": md5,
        "dhash": np.packbits(bits).tobytes().hex(),
        "mean_L": float(gray.mean()),
        "std_L": float(gray.std()),
        "mean_R": float(rgb[..., 0].mean()),
        "mean_G": float(rgb[..., 1].mean()),
        "mean_B": float(rgb[..., 2].mean()),
        "border_mean_L": float(gray[border].mean()),
        "border_std_L": float(gray[border].std()),
    }


def list_images(root):
    rows = []
    for cls in CLASSES:
        for p in sorted((root / cls).iterdir()):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            vol, _, idx = parse_filename(p.name)
            rows.append({
                "filepath": p.relative_to(PROJECT_ROOT).as_posix(),
                "label": cls,
                "volunteer": vol,
                "index": idx,
            })
    return pd.DataFrame(rows)


def build_index(root):
    files = list_images(root)
    if INDEX_CSV.exists():
        cached = pd.read_csv(INDEX_CSV, dtype={"filepath": str, "md5": str, "dhash": str})
        if set(cached["filepath"]) == set(files["filepath"]):
            print(f"  [CACHE] Reusing {INDEX_CSV.name}")
            return cached
        print("  [CACHE] Stale (file list changed); recomputing.")

    paths = [PROJECT_ROOT / f for f in files["filepath"]]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        stats = list(tqdm(pool.map(describe, paths), total=len(paths), desc="Describing", unit="img"))
    df = pd.concat([files, pd.DataFrame(stats)], axis=1)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)
    return df


# ---------------------------------------------------------------- helpers
def to_mat(hexes):
    raw = b"".join(bytes.fromhex(h) for h in hexes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hexes), 8)


def hamming_rows(A, B):
    """Row-wise Hamming distance between two (n, 8) uint8 hash matrices."""
    return POP8[np.bitwise_xor(A, B)].sum(axis=1, dtype=np.int32)


def dist_summary(x):
    x = np.asarray(x)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "mean": round(float(x.mean()), 2),
        "min": int(x.min()),
        "p1": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
        "frac_le_1": round(float((x <= 1).mean()), 4),
        "frac_le_5": round(float((x <= 5).mean()), 4),
    }


# ---------------------------------------------------------------- analyses
def exact_duplicates(df):
    g = df.groupby("md5").agg(n=("md5", "size"), vols=("volunteer", "nunique"),
                              classes=("label", "nunique"))
    multi = g[g["n"] > 1]
    cross = multi[multi["vols"] > 1]

    cross_rows = df[df["md5"].isin(cross.index)].sort_values("md5")
    examples = []
    for _, grp in cross_rows.groupby("md5"):
        examples.append(grp["filepath"].tolist())
        if len(examples) == 20:
            break

    return {
        "groups_size_gt1": int(len(multi)),
        "files_in_groups": int(multi["n"].sum()),
        "groups_within_one_volunteer": int((multi["vols"] == 1).sum()),
        "groups_spanning_volunteers": int(len(cross)),
        "files_in_cross_volunteer_groups": int(cross["n"].sum()),
        "groups_spanning_classes": int((multi["classes"] > 1).sum()),
        "cross_volunteer_examples": examples,
    }


def per_volunteer_table(df):
    g = df.groupby("volunteer")
    t = pd.DataFrame({
        "images": g.size(),
        "mean_L": g["mean_L"].mean(),
        "std_of_image_mean_L": g["mean_L"].std(),
        "mean_R": g["mean_R"].mean(),
        "mean_G": g["mean_G"].mean(),
        "mean_B": g["mean_B"].mean(),
        "border_mean_L": g["border_mean_L"].mean(),
        "border_std_L": g["border_std_L"].mean(),
    }).round(2)
    return t


def burst_analysis(df, rng):
    # Consecutive-index distances within each (volunteer, class), ordered by numeric index
    consec, per_vol = [], {}
    for (vol, _), g in df.sort_values("index").groupby(["volunteer", "label"]):
        M = to_mat(g["dhash"].tolist())
        d = hamming_rows(M[:-1], M[1:])
        consec.append(d)
        per_vol.setdefault(int(vol), []).append(d)
    consec = np.concatenate(consec)

    # Random same-class pairs, split by whether the two images share a volunteer
    within, cross = [], []
    for _, g in df.groupby("label"):
        M = to_mat(g["dhash"].tolist())
        V = g["volunteer"].to_numpy()
        i = rng.integers(0, len(g), size=PAIRS_PER_CLASS)
        j = rng.integers(0, len(g), size=PAIRS_PER_CLASS)
        keep = i != j
        same = V[i] == V[j]
        d = hamming_rows(M[i], M[j])
        within.append(d[keep & same])
        cross.append(d[keep & ~same])
    within = np.concatenate(within)
    cross = np.concatenate(cross)

    stats = {
        "consecutive_within_volunteer": dist_summary(consec),
        "random_same_class_same_volunteer": dist_summary(within),
        "random_same_class_cross_volunteer": dist_summary(cross),
        "median_consecutive_by_volunteer": {v: float(np.median(np.concatenate(lst)))
                                            for v, lst in sorted(per_vol.items())},
    }
    return stats, consec, cross


# ---------------------------------------------------------------- figures
def plot_luminance(df, out):
    vols = sorted(df["volunteer"].unique())
    labels = [f"P{v}" for v in vols]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, title in [
        (axes[0], "mean_L", "Whole-image mean luminance per image"),
        (axes[1], "border_mean_L", "Border (outer 10%) mean luminance per image"),
    ]:
        data = [df.loc[df["volunteer"] == v, col].values for v in vols]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(vols) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Volunteer")
        ax.set_ylabel("Grayscale intensity (0-255)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_hamming(consec, cross, out):
    bins = np.arange(0, 66) - 0.5   # full 64-bit range; cross-signer p99 is 45
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(consec, bins=bins, alpha=0.7, density=True, label="Consecutive index, same signer",
            color="#2b5c8f", edgecolor="black")
    ax.hist(cross, bins=bins, alpha=0.5, density=True, label="Random same class, different signers",
            color="#d95f02", edgecolor="black")
    ax.set_title("64-bit dHash Hamming distance (ASL-HG)", fontsize=11, weight="bold")
    ax.set_xlabel("Hamming distance (0 = identical hash)")
    ax.set_ylabel("Share of pairs")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.1 PART B (SIGNER AND BACKGROUND AUDIT)")
    print("=" * 70)

    root = find_dataset_root(RAW_DIR)
    print("\n[1/4] Building the per-image index")
    df = build_index(root)
    print(f"  {len(df):,} images, {df['volunteer'].nunique()} volunteers, {df['label'].nunique()} classes")

    print("\n[2/4] Byte-identical copies across signers")
    dup = exact_duplicates(df)
    print(f"  groups (size > 1)           : {dup['groups_size_gt1']}  ({dup['files_in_groups']} files)")
    print(f"  within one volunteer        : {dup['groups_within_one_volunteer']}")
    print(f"  spanning volunteers         : {dup['groups_spanning_volunteers']}  "
          f"({dup['files_in_cross_volunteer_groups']} files)")
    print(f"  spanning classes            : {dup['groups_spanning_classes']}")

    print("\n[3/4] Per-signer statistics")
    vt = per_volunteer_table(df)
    print(vt.to_string())

    print("\n[4/4] Burst structure and cross-signer distances")
    rng = np.random.default_rng(SEED)
    burst, consec, cross = burst_analysis(df, rng)
    for k in ["consecutive_within_volunteer", "random_same_class_same_volunteer",
              "random_same_class_cross_volunteer"]:
        s = burst[k]
        if s["n"]:
            print(f"  {k:<36} n={s['n']:>6}  median={s['median']:<5} "
                  f"min={s['min']:<3} frac<=1={s['frac_le_1']}  frac<=5={s['frac_le_5']}")
    med = burst["median_consecutive_by_volunteer"]
    print(f"  median consecutive by signer: {med}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_luminance(df, FIG_DIR / "eda_signer_luminance.png")
    plot_hamming(consec, cross, FIG_DIR / "eda_signer_hamming.png")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "border_fraction": BORDER_FRAC,
        "pairs_per_class": PAIRS_PER_CLASS,
        "exact_duplicates": dup,
        "per_volunteer": vt.reset_index().to_dict("records"),
        "burst_analysis": burst,
        "limitations": [
            "Background count is not in the metadata; border luminance is a proxy only.",
            "Skin tone is not measured directly; mean RGB is confounded by background.",
            "dHash uses the same 9x8 definition as src/split_data.py, so its distances are comparable.",
            "Consecutive-index distances assume the index follows capture order; this is tested, not assumed.",
        ],
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Report: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    print("Figures: docs/figures/eda_signer_luminance.png, docs/figures/eda_signer_hamming.png")
    print("=" * 70)


if __name__ == "__main__":
    main()