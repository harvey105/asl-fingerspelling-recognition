"""
src/audit_leakage.py
Task 2.4 diagnostic (read-only): residual near-duplicate leakage across splits,
and whether consecutive filenames are visually adjacent (recording order).
Writes docs/task2_4_leakage_report.json.

Run: uv run python src/audit_leakage.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = PROJECT_ROOT / "data" / "splits"
CACHE = PROJECT_ROOT / "data" / "processed" / "image_dhash.csv"
REPORT = PROJECT_ROOT / "docs" / "task2_4_leakage_report.json"
THRESHOLDS = [0, 1, 2, 3, 5]
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def to_mat(hexes):
    raw = b"".join(bytes.fromhex(h) for h in hexes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hexes), 8)


def hamming_matrix(A, C):
    return POP[np.bitwise_xor(A[:, None, :], C[None, :, :])].sum(axis=2, dtype=np.int32)


def main():
    cache = pd.read_csv(CACHE, dtype={"filepath": str, "dhash": str})
    lookup = dict(zip(cache["filepath"], cache["dhash"]))
    splits = {k: pd.read_csv(SPLITS / f"{k}.csv") for k in ["train", "val", "test"]}
    for df in splits.values():
        df["dhash"] = df["filepath"].map(lookup)
        assert df["dhash"].notna().all(), "Split file contains paths missing from the hash cache."

    report = {"thresholds": THRESHOLDS, "held_out_vs_train": {}}

    # 1. For each held-out image: distance to its nearest training image of the same class
    for held in ["val", "test"]:
        nn = []
        for cls, g in splits[held].groupby("label"):
            tr = splits["train"][splits["train"]["label"] == cls]
            nn.extend(hamming_matrix(to_mat(g["dhash"]), to_mat(tr["dhash"])).min(axis=1).tolist())
        nn = np.array(nn)
        report["held_out_vs_train"][held] = {
            "n": int(len(nn)),
            "median_nn_distance": float(np.median(nn)),
            **{f"frac_with_train_neighbour_d<={t}": round(float((nn <= t).mean()), 4) for t in THRESHOLDS},
        }

    # 2. Are consecutive filenames visually adjacent? (tests whether numbering follows recording order)
    rng = np.random.default_rng(42)
    adjacent, random_pairs = [], []
    cache["label"] = cache["filepath"].str.split("/").str[-2]
    for cls, g in cache.groupby("label"):
        g = g.assign(num=g["filepath"].str.extract(r"(\d+)\.jpg$")[0].astype(int)).sort_values("num")
        M = to_mat(g["dhash"])
        adjacent.extend(POP[np.bitwise_xor(M[:-1], M[1:])].sum(axis=1).tolist())
        i, j = rng.integers(0, len(M), size=(2, len(M)))
        random_pairs.extend(POP[np.bitwise_xor(M[i], M[j])].sum(axis=1).tolist())
    adjacent, random_pairs = np.array(adjacent), np.array(random_pairs)
    report["filename_adjacency"] = {
        "median_consecutive_distance": float(np.median(adjacent)),
        "median_random_pair_distance": float(np.median(random_pairs)),
        "frac_consecutive_d<=1": round(float((adjacent <= 1).mean()), 4),
        "frac_random_d<=1": round(float((random_pairs <= 1).mean()), 4),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {REPORT.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()