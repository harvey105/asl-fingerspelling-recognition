"""
src/freeze_random_splits.py
 random stratified 70/15/15 split (seed 42), for Ablation 1 only.

Universe: the 35,988 written crops (blank frames excluded, ). No de-duplication: this is the
naive protocol that Ablation 1 compares against the subject split. The random split is NOT the
primary evaluation .

Method: the same two-stage stratified split as src/audit_random_split.py (test first, then val
from the remainder), so the  leakage numbers remain comparable.

Writes (write-once; refuses to overwrite):
  data/splits/random_train.csv, random_val.csv, random_test.csv
  data/splits/SHA256SUMS_random.txt
  docs/task2_4_partB_random_csv.json

Columns match the subject CSVs: filepath, class_name, class_id, volunteer_id, crop_path, md5.
Run (from repository root): uv run python src/freeze_random_splits.py
"""

import json
import sys
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BLANK_FRAMES_CSV,
    CLASS_TO_ID,
    CLASSES,
    DOCS_DIR,
    PROJECT_ROOT,
    SPLITS_DIR,
)

from .manifests import load_written_table
from .splits import COLUMNS, gate, sha256, to_frame, write_and_verify

SEED = 42
EXPECTED_TOTAL = 35988
TARGETS = {
    "train": SPLITS_DIR / "random_train.csv",
    "val": SPLITS_DIR / "random_val.csv",
    "test": SPLITS_DIR / "random_test.csv",
}
CHECKSUMS = SPLITS_DIR / "SHA256SUMS_random.txt"
REPORT = DOCS_DIR / "task2_4_partB_random_csv.json"


def random_split(df):
    trainval, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=SEED
    )
    train, val = train_test_split(
        trainval, test_size=0.15 / 0.85, stratify=trainval["label"], random_state=SEED
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def per_class_table(df, train, val, test):
    t = (
        pd.DataFrame(
            {
                "total": df["label"].value_counts(),
                "train": train["label"].value_counts(),
                "val": val["label"].value_counts(),
                "test": test["label"].value_counts(),
            }
        )
        .reindex(CLASSES)
        .fillna(0)
        .astype(int)
    )
    return t


def leakage(train, val, test):
    """Informational, not gated: how much the random split shares MD5 content across splits."""
    train_md5 = set(train["md5"])
    out = {
        "val_rows_with_train_md5": int(val["md5"].isin(train_md5).sum()),
        "test_rows_with_train_md5": int(test["md5"].isin(train_md5).sum()),
        "val_rows": len(val),
        "test_rows": len(test),
    }
    tagged = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ]
    )
    crossing = tagged.groupby("md5")["split"].nunique()
    out["md5_groups_crossing_splits"] = int((crossing > 1).sum())
    return out


def main():
    print("=" * 70)
    print("ASL-HG:  (RANDOM STRATIFIED SPLIT, ABLATION 1 ONLY)")
    print("=" * 70)

    existing = [p for p in list(TARGETS.values()) + [CHECKSUMS] if p.exists()]
    if existing:
        print("Refusing to overwrite existing random split files (write-once):")
        for p in existing:
            print(f"  {p.relative_to(PROJECT_ROOT).as_posix()}")
        sys.exit(1)

    df = load_written_table()
    blank = set(pd.read_csv(BLANK_FRAMES_CSV, dtype={"filepath": str})["filepath"])
    train, val, test = random_split(df)
    frames = {"train": to_frame(train), "val": to_frame(val), "test": to_frame(test)}
    per = per_class_table(df, train, val, test)

    results = {}
    print("\nGates")
    gate(results, "universe_is_35988", len(df) == EXPECTED_TOTAL, f"{len(df)}")
    gate(results, "no_blank_rows", not df["filepath"].isin(blank).any())
    total = sum(len(frames[k]) for k in frames)
    gate(results, "splits_sum_to_universe", total == len(df), f"{total} of {len(df)}")

    fp = {k: set(frames[k]["filepath"]) for k in frames}
    gate(
        results,
        "no_filepath_in_two_splits",
        fp["train"].isdisjoint(fp["val"])
        and fp["val"].isdisjoint(fp["test"])
        and fp["train"].isdisjoint(fp["test"]),
    )

    for k in frames:
        n = frames[k]["class_id"].nunique()
        gate(results, f"all_classes_in_{k}", n == len(CLASSES), f"{n} classes")

    dev_val = (per["val"] - 0.15 * per["total"]).abs().max()
    dev_test = (per["test"] - 0.15 * per["total"]).abs().max()
    dev = float(max(dev_val, dev_test))
    gate(
        results,
        "per_class_within_one_of_15pct",
        dev <= 1.0,
        f"max deviation {dev:.2f} images",
    )

    name_ok = all(
        CLASS_TO_ID[n] == i
        for k in frames
        for n, i in zip(frames[k]["class_name"], frames[k]["class_id"])
    )
    gate(results, "class_name_matches_class_id", name_ok)

    missing = sum(
        not (PROJECT_ROOT / p).exists() for k in frames for p in frames[k]["crop_path"]
    )
    gate(results, "every_crop_exists", missing == 0, f"missing {missing}")

    if not all(v["pass"] for v in results.values()):
        print("\nVERDICT: FAILED. No files written.")
        sys.exit(1)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for k, path in TARGETS.items():
        ok, diffs = write_and_verify(frames[k], path, COLUMNS)
        gate(results, f"read_back_{k}", ok, "; ".join(diffs))
        all_ok &= ok
    if not all_ok:
        print("\nVERDICT: FAILED on read-back. Failed files were not kept.")
        sys.exit(1)

    CHECKSUMS.write_text(
        "\n".join(f"{sha256(p)}  {p.name}" for p in TARGETS.values()) + "\n",
        encoding="utf-8",
    )

    leak = leakage(train, val, test)
    print("\nInformational: random-split content sharing (not gated)")
    print(
        f"  val rows whose MD5 appears in train : {leak['val_rows_with_train_md5']:,} "
        f"of {leak['val_rows']:,}"
    )
    print(
        f"  test rows whose MD5 appears in train: {leak['test_rows_with_train_md5']:,} "
        f"of {leak['test_rows']:,}"
    )
    print(
        f"  MD5 groups crossing splits          : {leak['md5_groups_crossing_splits']}"
    )

    print("\nPer-class counts (total | train | val | test)")
    for c, row in per.iterrows():
        print(
            f"  {c:>2}: {row['total']:>4} | {row['train']:>4} | {row['val']:>4} | {row['test']:>4}"
        )

    print("\nWritten:")
    for p in list(TARGETS.values()) + [CHECKSUMS]:
        print(f"  {p.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"\nSizes: train {len(train):,}  val {len(val):,}  test {len(test):,}")
    print("VERDICT: PASSED")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "method": "two-stage stratified split: test 15%, then val 15% of the remainder",
        "universe": len(df),
        "counts": {k: len(frames[k]) for k in frames},
        "per_class": per.to_dict(orient="index"),
        "max_per_class_deviation": round(dev, 3),
        "leakage_informational": leak,
        "gates": results,
        "passed": True,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Report: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    sys.exit(0)


if __name__ == "__main__":
    main()
