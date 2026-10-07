"""
src/audit_cross_split.py
Task 2.1, Part B-2: does the subject-independent split leak?

Uses the cached index written by audit_signers.py (no image decoding).
Checks the roadmap split: train P1-P7, val P8, test P9-P10.

  1. Byte-identical copies that cross a split boundary (expected 0)
  2. Volunteer-pair matrix of byte-identical copies
  3. Per-class: cross-volunteer copy groups and distinct byte contents
  4. Nearest same-class dHash distance from each val/test image to the train split

Read-only. Writes docs/task2_1_partB2_cross_split.json.
Run (from the repository root): uv run python src/audit_cross_split.py
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    CLASSES,
    DOCS_DIR,
    PROCESSED_DIR,
    TEST_VOLUNTEERS,
    TRAIN_VOLUNTEERS,
    VAL_VOLUNTEERS,
)

INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
REPORT = DOCS_DIR / "task2_1_partB2_cross_split.json"
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def split_of(v):
    if v in TRAIN_VOLUNTEERS:
        return "train"
    if v in VAL_VOLUNTEERS:
        return "val"
    if v in TEST_VOLUNTEERS:
        return "test"
    return "unassigned"


def to_mat(hexes):
    raw = b"".join(bytes.fromhex(h) for h in hexes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hexes), 8)


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


def exact_copy_checks(df):
    multi = df[df.duplicated("md5", keep=False)]
    cross_vol = multi.groupby("md5").filter(lambda g: g["volunteer"].nunique() > 1)
    cross_split = multi.groupby("md5").filter(lambda g: g["split"].nunique() > 1)

    # Volunteer-pair matrix: how many copy groups contain both volunteer a and volunteer b
    pair = pd.DataFrame(0, index=range(1, 11), columns=range(1, 11))
    for _, g in cross_vol.groupby("md5"):
        vols = sorted(set(g["volunteer"].astype(int)))
        for i, a in enumerate(vols):
            for b in vols[i + 1:]:
                pair.loc[a, b] += 1

    per_class = {}
    for cls in CLASSES:
        sub = cross_vol[cross_vol["label"] == cls]
        per_class[cls] = {
            "cross_volunteer_groups": int(sub["md5"].nunique()),
            "files_in_those_groups": int(len(sub)),
            "distinct_byte_contents": int(df.loc[df["label"] == cls, "md5"].nunique()),
        }

    return {
        "groups_cross_volunteer": int(cross_vol["md5"].nunique()),
        "groups_crossing_split_boundary": int(cross_split["md5"].nunique()),
        "volunteer_pair_matrix": pair.to_dict(orient="index"),
        "per_class": per_class,
        "cross_split_examples": [
            g["filepath"].tolist()
            for _, g in cross_split.groupby("md5")
        ][:20],
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
    print("ASL-HG: TASK 2.1 PART B-2 (SUBJECT SPLIT LEAKAGE CHECK)")
    print("=" * 70)

    df = pd.read_csv(INDEX_CSV, dtype={"md5": str, "dhash": str})
    df["split"] = df["volunteer"].map(split_of)
    if (df["split"] == "unassigned").any():
        raise ValueError("Some volunteers are not assigned to a split in config.py.")

    print("\n[1/3] Exact copies across volunteers and splits")
    copies = exact_copy_checks(df)
    print(f"  groups across volunteers      : {copies['groups_cross_volunteer']}")
    print(f"  groups crossing a split        : {copies['groups_crossing_split_boundary']}  (expected 0)")
    print("  volunteer-pair matrix (groups containing both):")
    pm = pd.DataFrame(copies["volunteer_pair_matrix"])
    print(pm.to_string())

    print("\n[2/3] Per-class cross-volunteer copies")
    pcls = pd.DataFrame(copies["per_class"]).T
    print(pcls.to_string())

    print("\n[3/3] Nearest same-class distance: held-out image to train split")
    near = nearest_to_train(df)
    for held, s in near.items():
        print(f"  {held:<5} n={s['n']:>6}  min={s['min']}  p1={s['p1']}  median={s['median']}  "
              f"frac=0: {s['frac_eq_0']}  frac<=1: {s['frac_le_1']}  frac<=5: {s['frac_le_5']}")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": {"train": TRAIN_VOLUNTEERS, "val": VAL_VOLUNTEERS, "test": TEST_VOLUNTEERS},
        "exact_copies": copies,
        "nearest_to_train": near,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {REPORT.relative_to(DOCS_DIR.parent).as_posix()}")


if __name__ == "__main__":
    main()