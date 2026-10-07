"""
src/split_data.py
Task 2.4: group-aware, class-stratified split of data/raw/asl_alphabet_train.

Near-duplicate grouping uses the connected components of the graph
{(i, j) : Hamming(dHash_i, dHash_j) <= t} within each class. Every pair within
distance t lands in the same group, so no near-duplicate pair can cross splits.

Threshold choice: connected components chain (A~B and B~C put A, B, C together),
so a large t can merge most of a class into one group that cannot fit the
450-image val/test quota. The script sweeps t = 0..5, prints the largest group
at each t, and uses the largest t at which every group in every class fits.

Quota per class (70/15/15 of 3,000): train 2,100 / val 450 / test 450.
Whole groups cannot be split, so per-class counts are exact only if group sizes
allow it. Actual counts and deviations are reported, not hidden.

Run:     uv run python src/split_data.py
Outputs: data/splits/{train,val,test}.csv, class_mapping.json, split_report.json
Cache:   data/processed/image_dhash.csv
         (delete it if you change the hash function or its preprocessing)
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "asl_alphabet_train"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
HASH_CACHE = PROJECT_ROOT / "data" / "processed" / "image_dhash.csv"

RANDOM_SEED = 42
EXPECTED_CLASSES = sorted([chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"])
CLASS_TO_ID = {c: i for i, c in enumerate(EXPECTED_CLASSES)}
IMAGES_PER_CLASS = 3000
QUOTA = {"train": 2100, "val": 450, "test": 450}   # per class
MAX_GROUP = min(QUOTA["val"], QUOTA["test"])       # no group may exceed this
THRESHOLD_SWEEP = [0, 1, 2, 3, 4, 5]               # Hamming distances (of 64 bits)
NUM_WORKERS = 8
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

assert sum(QUOTA.values()) == IMAGES_PER_CLASS, "Quotas must sum to images per class"


# ---------------------------------------------------------------- indexing
def index_images():
    records = []
    for cls in EXPECTED_CLASSES:
        cls_dir = DATA_DIR / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        entries = sorted(cls_dir.iterdir())
        stray = [p.name for p in entries if not (p.is_file() and p.suffix.lower() == ".jpg")]
        if stray:
            raise ValueError(f"'{cls}' has non-.jpg entries {stray[:5]}. Run src/verify_dataset.py.")
        if len(entries) != IMAGES_PER_CLASS:
            raise ValueError(f"'{cls}' has {len(entries)} images, expected {IMAGES_PER_CLASS}.")
        for p in entries:
            records.append({
                "filepath": p.relative_to(PROJECT_ROOT).as_posix(),
                "label": cls,
                "class_id": CLASS_TO_ID[cls],
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------- hashing
def dhash_hex(path):
    """64-bit difference hash: 9x8 grayscale, compare each pixel with its right neighbour."""
    with Image.open(path) as im:
        small = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()        # 8 rows x 8 comparisons = 64 bits
    return np.packbits(bits).tobytes().hex()          # 8 bytes -> 16 hex characters


def load_or_compute_hashes(df):
    if HASH_CACHE.exists():
        cached = pd.read_csv(HASH_CACHE, dtype={"filepath": str, "dhash": str})
        if set(cached["filepath"]) == set(df["filepath"]):
            print(f"  [CACHE] Reusing {HASH_CACHE.name}")
            lookup = dict(zip(cached["filepath"], cached["dhash"]))
            return [lookup[f] for f in df["filepath"]]
        print("  [CACHE] Stale (file list changed); recomputing.")

    paths = [PROJECT_ROOT / f for f in df["filepath"]]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        hexes = list(tqdm(pool.map(dhash_hex, paths), total=len(paths), desc="dHash", unit="img"))
    HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filepath": df["filepath"], "dhash": hexes}).to_csv(HASH_CACHE, index=False)
    return hexes


def hex_to_matrix(hexes):
    raw = b"".join(bytes.fromhex(h) for h in hexes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hexes), 8)


def hamming_to_all(row, mat):
    """Hamming distance between one 8-byte hash and every row of an (n, 8) matrix."""
    return POPCOUNT8[np.bitwise_xor(mat, row)].sum(axis=1, dtype=np.int32)


def pairwise_hamming(Bc):
    n = len(Bc)
    D = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        D[i] = hamming_to_all(Bc[i], Bc)
    return D


# ---------------------------------------------------------------- graphs
def analyse_class(Bc):
    """
    Returns:
      nn:     nearest-neighbour distance of each image within its class
      labels: {t: component id per image} for every threshold in the sweep
    """
    D = pairwise_hamming(Bc)
    off = D.copy()
    np.fill_diagonal(off, 255)
    nn = off.min(axis=1).astype(np.int32)

    labels = {}
    for t in THRESHOLD_SWEEP:
        adj = csr_matrix((D <= t).astype(np.int8))
        _, lab = connected_components(adj, directed=False)
        labels[t] = lab
    return nn, labels


def analyse_all(df, B):
    analysis = {}
    for cls in tqdm(EXPECTED_CLASSES, desc="Near-duplicate graphs", unit="cls"):
        idx = np.nonzero(df["label"].to_numpy() == cls)[0]
        nn, labels = analyse_class(B[idx])
        analysis[cls] = (idx, nn, labels)
    return analysis


def choose_threshold(analysis):
    """Largest t for which every group in every class fits MAX_GROUP. Returns (t or None, sweep rows)."""
    rows = []
    for t in THRESHOLD_SWEEP:
        worst, worst_cls, n_groups = 0, "", 0
        for cls, (_, _, labels) in analysis.items():
            sizes = np.bincount(labels[t])
            n_groups += len(sizes)
            if sizes.max() > worst:
                worst, worst_cls = int(sizes.max()), cls
        rows.append({
            "threshold": t,
            "largest_group": worst,
            "largest_group_class": worst_cls,
            "total_groups": int(n_groups),
            "fits_quota": worst <= MAX_GROUP,
        })
    feasible = [r["threshold"] for r in rows if r["fits_quota"]]
    return (max(feasible) if feasible else None), rows


# ---------------------------------------------------------------- assignment
def assign_groups(group_sizes, rng):
    """Largest groups first (random tie-break from seeded RNG), each to the split with the largest remaining deficit."""
    gids = sorted(group_sizes)
    keys = rng.random(len(gids))
    order = sorted(range(len(gids)), key=lambda k: (-group_sizes[gids[k]], keys[k]))
    totals = {s: 0 for s in QUOTA}
    assignment = {}
    for k in order:
        g = gids[k]
        split = max(QUOTA, key=lambda s: QUOTA[s] - totals[s])
        assignment[g] = split
        totals[split] += group_sizes[g]
    return assignment, totals


def build_splits(df, analysis, t):
    rng = np.random.default_rng(RANDOM_SEED)
    df = df.copy()
    df["group_id"] = ""
    df["split"] = ""
    per_class = {}
    nn_hist = np.zeros(65, dtype=np.int64)

    for cls in EXPECTED_CLASSES:
        idx, nn, labels = analysis[cls]
        nn_hist += np.bincount(nn, minlength=65)
        gids = [f"{cls}_{int(k):04d}" for k in labels[t]]
        sizes = Counter(gids)
        largest = max(sizes.values())
        assert largest <= MAX_GROUP, f"{cls}: group of {largest} exceeds {MAX_GROUP} (sweep should prevent this)"

        assignment, totals = assign_groups(sizes, rng)
        df.loc[idx, "group_id"] = gids
        df.loc[idx, "split"] = [assignment[g] for g in gids]
        per_class[cls] = {
            "images": int(len(idx)),
            "groups": int(len(sizes)),
            "largest_group": int(largest),
            **{s: int(totals[s]) for s in QUOTA},
        }
    return df, per_class, nn_hist


# ---------------------------------------------------------------- verification
def verify_splits(df, B, t, per_class):
    # 1. Every group lies in exactly one split
    spans = df.groupby("group_id")["split"].nunique()
    if not (spans == 1).all():
        raise AssertionError("A duplicate group spans more than one split.")

    # 2. Complete, disjoint partition
    if len(df) != IMAGES_PER_CLASS * len(EXPECTED_CLASSES):
        raise AssertionError(f"Expected {IMAGES_PER_CLASS * len(EXPECTED_CLASSES)} rows, got {len(df)}.")
    if not df["filepath"].is_unique:
        raise AssertionError("Duplicate filepaths across splits.")

    # 3. Independent leakage test, computed directly from hashes (not from the grouping)
    min_cross = 64
    for cls in EXPECTED_CLASSES:
        in_class = (df["label"] == cls).to_numpy()
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            A = B[in_class & (df["split"] == a).to_numpy()]
            C = B[in_class & (df["split"] == b).to_numpy()]
            d = POPCOUNT8[np.bitwise_xor(A[:, None, :], C[None, :, :])].sum(axis=2, dtype=np.int32)
            min_cross = min(min_cross, int(d.min()))
    if min_cross <= t:
        raise AssertionError(f"Near-duplicate leakage: cross-split Hamming {min_cross} <= threshold {t}.")

    max_dev = max(abs(per_class[c][s] - QUOTA[s]) for c in per_class for s in QUOTA)
    return {"min_cross_split_hamming": min_cross, "max_count_deviation": int(max_dev)}


# ---------------------------------------------------------------- outputs
def write_outputs(df, per_class, nn_hist, leak, t, sweep):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["filepath", "label", "class_id", "group_id"]
    for s in QUOTA:
        df.loc[df["split"] == s, cols].reset_index(drop=True).to_csv(SPLITS_DIR / f"{s}.csv", index=False)

    mapping = {
        "class_to_id": CLASS_TO_ID,
        "id_to_class": EXPECTED_CLASSES,   # list: index == class_id
        "total_classes": len(EXPECTED_CLASSES),
    }
    (SPLITS_DIR / "class_mapping.json").write_text(json.dumps(mapping, indent=4), encoding="utf-8")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": RANDOM_SEED,
        "quota_per_class": QUOTA,
        "max_group_size": MAX_GROUP,
        "chosen_threshold_hamming": t,
        "threshold_sweep": sweep,
        "totals": {s: int((df["split"] == s).sum()) for s in QUOTA},
        "exact_per_class_counts": leak["max_count_deviation"] == 0,
        "max_count_deviation": leak["max_count_deviation"],
        "min_cross_split_hamming": leak["min_cross_split_hamming"],
        "images_with_near_duplicate_at_threshold": int(nn_hist[:t + 1].sum()),
        "nearest_neighbour_histogram_within_class": {str(d): int(nn_hist[d]) for d in range(17)},
        "per_class": per_class,
    }
    (SPLITS_DIR / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_sweep(sweep):
    print(f"  {'t':>2} | {'largest group':>13} | {'class':>7} | {'groups':>7} | fits {MAX_GROUP}")
    for r in sweep:
        mark = "yes" if r["fits_quota"] else "NO"
        print(f"  {r['threshold']:>2} | {r['largest_group']:>13} | {r['largest_group_class']:>7} | "
              f"{r['total_groups']:>7} | {mark}")


def main():
    print("=" * 70)
    print("TASK 2.4: GROUP-AWARE STRATIFIED SPLIT")
    print(f"seed={RANDOM_SEED}  quota/class={QUOTA}  threshold sweep={THRESHOLD_SWEEP}")
    print("=" * 70)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Source directory not found: {DATA_DIR}")

    print("\n[1/6] Indexing images")
    df = index_images()
    print(f"  {len(df):,} images across {df['label'].nunique()} classes")

    print("\n[2/6] Perceptual hashes")
    hexes = load_or_compute_hashes(df)
    B = hex_to_matrix(hexes)

    print("\n[3/6] Near-duplicate graphs per class")
    analysis = analyse_all(df, B)

    print("\n[4/6] Choosing the threshold")
    t, sweep = choose_threshold(analysis)
    print_sweep(sweep)
    if t is None:
        raise SystemExit(
            "\nNo threshold keeps every near-duplicate group within "
            f"{MAX_GROUP} images, even t=0. The 9x8 dHash is too coarse. Next step: "
            "switch to a 16x16 dHash (256 bits), delete data/processed/image_dhash.csv, and rerun."
        )
    print(f"  Chosen threshold: t={t} (largest feasible value)")

    print("\n[5/6] Assigning whole groups to splits")
    df, per_class, nn_hist = build_splits(df, analysis, t)

    print("\n[6/6] Verifying and writing outputs")
    leak = verify_splits(df, B, t, per_class)
    report = write_outputs(df, per_class, nn_hist, leak, t, sweep)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for s, n in report["totals"].items():
        print(f"  {s:<5}: {n:,} images (quota {QUOTA[s] * len(EXPECTED_CLASSES):,})")
    print(f"  Exact per-class counts       : {report['exact_per_class_counts']}")
    print(f"  Max per-class deviation      : {report['max_count_deviation']} images")
    print(f"  Images with near-duplicate   : {report['images_with_near_duplicate_at_threshold']:,} (d <= {t})")
    print(f"  Min cross-split Hamming      : {report['min_cross_split_hamming']} (must be > {t})")
    print("\n  Nearest-neighbour histogram (within class):")
    for d in range(0, 13):
        print(f"    d={d:>2}: {report['nearest_neighbour_histogram_within_class'][str(d)]:>7,}")
    print("\n  Largest group per class (top 5):")
    for cls, info in sorted(per_class.items(), key=lambda kv: -kv[1]["largest_group"])[:5]:
        print(f"    {cls:<8} largest={info['largest_group']:<4} groups={info['groups']}")
    print("\nOutputs written to data/splits/")
    print("=" * 70)


if __name__ == "__main__":
    main()