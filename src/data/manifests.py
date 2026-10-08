"""
src/manifests.py
 subject-independent manifests over the verified crop output, built in memory.

Inputs:
  data/processed/asl_hg_image_index.csv   per-image md5, label, volunteer
  data/processed/roi_crop_log.csv         per-image method (what was actually written)

Rules:
  - Only images with a written crop are included. Blank frames are excluded .
  - Split by signer from config.py: train P1-P7, val P8, test P9-P10 .
  - Exact-copy de-duplication in TRAIN ONLY : from each byte-identical group, keep one
    file (first by filepath). Val and test are not de-duplicated, so the evaluation set is the
    full crop output. Within-signer repeats there are reported, not removed.
  - Writes nothing.  freezes the result to CSV.

Run as a module through check_dataset_part_a.py; no standalone run.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    BLANK_FRAMES_CSV,
    CLASS_TO_ID,
    CLASSES,
    CROPS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TEST_VOLUNTEERS,
    TRAIN_VOLUNTEERS,
    VAL_VOLUNTEERS,
)

INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
LOG_CSV = PROCESSED_DIR / "roi_crop_log.csv"
WRITTEN_METHODS = ["mediapipe", "skin", "class_rule_full", "undetected_full"]


def _split_of(volunteer):
    if volunteer in TRAIN_VOLUNTEERS:
        return "train"
    if volunteer in VAL_VOLUNTEERS:
        return "val"
    if volunteer in TEST_VOLUNTEERS:
        return "test"
    return None


def _crop_rel_path(label, filepath):
    return (
        (CROPS_DIR / label / (Path(filepath).stem + ".png"))
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )


def load_written_table():
    # Labels are read as strings so the digit '0' is never coerced to a number.
    idx = pd.read_csv(INDEX_CSV, dtype={"filepath": str, "label": str, "md5": str})
    log = pd.read_csv(LOG_CSV, dtype={"filepath": str, "label": str})
    df = idx[["filepath", "label", "volunteer", "md5"]].merge(
        log[["filepath", "method"]], on="filepath", how="inner", validate="one_to_one"
    )
    df = df[df["method"].isin(WRITTEN_METHODS)].copy()

    df["volunteer"] = df["volunteer"].astype(int)
    df["class_id"] = df["label"].map(CLASS_TO_ID)
    if df["class_id"].isna().any():
        raise ValueError(
            f"Unknown labels: {sorted(df.loc[df['class_id'].isna(), 'label'].unique())}"
        )
    df["class_id"] = df["class_id"].astype(int)
    df["crop_path"] = [
        _crop_rel_path(l, f) for l, f in zip(df["label"], df["filepath"])
    ]
    df["split"] = df["volunteer"].map(_split_of)
    if df["split"].isna().any():
        raise ValueError("Some volunteers are not assigned to a split in config.py.")
    return df.sort_values("filepath").reset_index(drop=True)


def build_manifests():
    df = load_written_table()
    blank = set(pd.read_csv(BLANK_FRAMES_CSV, dtype={"filepath": str})["filepath"])
    if df["filepath"].isin(blank).any():
        raise ValueError("A blank frame has a written crop. Check preprocess_roi.py.")

    train_all = df[df["split"] == "train"]
    keep_mask = ~train_all.duplicated(subset="md5", keep="first")
    train = train_all[keep_mask]
    excluded = train_all[~keep_mask].assign(
        kept_filepath=lambda d: d["md5"].map(train.set_index("md5")["filepath"])
    )
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    def per_class(d):
        return (
            d["label"]
            .value_counts()
            .reindex(CLASSES, fill_value=0)
            .astype(int)
            .to_dict()
        )

    report = {
        "written_total": len(df),
        "blank_excluded": len(blank),
        "train_before_dedupe": len(train_all),
        "train_copies_removed": int(len(train_all) - len(train)),
        "train_final": len(train),
        "val_final": len(val),
        "test_final": len(test),
        "train_per_class": per_class(train),
        "val_per_class": per_class(val),
        "test_per_class": per_class(test),
    }
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "excluded": excluded.reset_index(drop=True),
        "report": report,
    }
