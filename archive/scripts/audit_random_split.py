"""
src/audit_random_split.py
Task 2.1, Part B-3: how much does a random stratified 70/15/15 split leak on ASL-HG?

Builds the random split in memory only (seed 42). Writes nothing to data/splits/.
Compares against the subject split from audit_cross_split.py.

Reads the cache written by audit_signers.py. No image decoding.
Writes docs/task2_1_partB3_random_split.json.
Run (from the repository root): uv run python src/audit_random_split.py
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLASSES, DOCS_DIR, PROCESSED_DIR

INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
REPORT = DOCS_DIR / "task2_1_partB3_random_split.json"
SEED = 42
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def to_mat(hexes):
    raw = b"".join(bytes.fromhex(h) for h in hexes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hexes), 8)


def random_split(df):
    tv, _ = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
    test_idx = df.index.difference(tv.index)
    train, val = train_test_split(tv, test_size=0.15 / 0.85, stratify=tv["label"], random_state=SEED)
    out = df.copy()
    out["split"] = ""
    out.loc[train.index, "split"] = "train"
    out.loc[val.index, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def cross_split_copies(df):
    multi = df[df.duplicated("md5", keep=False)]
    groups = multi.groupby("md5")
    crossing = groups["split"].nunique().gt(1)
    return {
        "groups_size_gt1": int(groups.ngroups),
        "groups_crossing_split_boundary": int(crossing.sum()),
        "files_in_crossing_groups": int(multi[multi["md5"].isin(crossing[crossing].index)].shape[0]),
    }


def dist_summary(x):
    x = np.asarray(x)
    return {
        "n": int(x.size),
        "min": int(x.min()),
        "p1": float(np.percentile(x, 1)),
        "median": float(np.median(x)),
        "frac_eq_0": round(float((x == 0).mean()), 4),
        "frac_le_1": round(float((x <= 1).mean()), 4),
        "frac_le_5": round(float((x <= 5).mean()), 4),
    }


def nearest_to_train(df):
    train = df[df["split"] == "train"]
    out = {}
    for held in ["val", "test"]:
        mins = []
        for cls in CLASSES:
            tr = train[train["label"] == cls]
            ho = df[(df["split"] == held) & (df["label"] == cls)]
            Mt = to_mat(tr["dhash"].tolist())
            Mh = to_mat(ho["dhash"].tolist())
            D = POP8[np.bitwise_xor(Mh[:, None, :], Mt[None, :, :])].sum(axis=2, dtype=np.int32)
            mins.append(D.min(axis=1))
        out[held] = dist_summary(np.concatenate(mins))
    return out


def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.1 PART B-3 (RANDOM SPLIT LEAKAGE, FOR ABLATION 1)")
    print("=" * 70)

    df = pd.read_csv(INDEX_CSV, dtype={"md5": str, "dhash": str})
    df = random_split(df)
    print(df["split"].value_counts().to_string())

    print("\n[1/2] Exact copies crossing random-split boundaries")
    copies = cross_split_copies(df)
    print(f"  groups (size > 1)             : {copies['groups_size_gt1']}")
    print(f"  groups crossing a split       : {copies['groups_crossing_split_boundary']}")
    print(f"  files in crossing groups      : {copies['files_in_crossing_groups']}")

    print("\n[2/2] Nearest same-class distance: held-out image to train (random split)")
    near = nearest_to_train(df)
    for held, s in near.items():
        print(f"  {held:<5} n={s['n']:>6}  min={s['min']}  p1={s['p1']}  median={s['median']}  "
              f"frac=0: {s['frac_eq_0']}  frac<=1: {s['frac_le_1']}  frac<=5: {s['frac_le_5']}")

    print("\nFor comparison, subject split (from task2_1_partB2_cross_split.json):")
    print("  val  frac<=1: 0.0   frac<=5: 0.0")
    print("  test frac<=1: 0.0   frac<=5: 0.0042   groups crossing a split: 0")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "split_method": "stratified 70/15/15, in memory, not written to data/splits/",
        "split_counts": df["split"].value_counts().to_dict(),
        "exact_copies": copies,
        "nearest_to_train": near,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {REPORT.relative_to(DOCS_DIR.parent).as_posix()}")


if __name__ == "__main__":
    main()