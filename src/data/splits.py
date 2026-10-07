"""
src/freeze_splits.py
 freeze the subject-independent split to immutable CSV files.

Writes (write-once; the script refuses to overwrite any existing file):
  data/splits/subject_train.csv          24,777 rows, P1-P7, after train de-duplication 
  data/splits/subject_val.csv             3,588 rows, P8 (, blanks excluded)
  data/splits/subject_test.csv            7,200 rows, P9-P10
  data/splits/excluded_train_copies.csv     423 rows, the removed duplicates and the kept file
  data/splits/class_mapping.json          class_id <-> class_name, from config.py
  data/splits/SHA256SUMS.txt              checksums of every file above

Column order: filepath, class_name, class_id, volunteer_id, crop_path, md5.
Required by the leader: filepath, class_name, class_id, volunteer_id.
Extras: crop_path (used by the loader), md5 (audit).

Gates (any failure exits with code 1 and writes nothing):
  counts, no volunteer in two splits, all 36 classes in each split, no MD5 duplicate in
  train, every crop file exists, class_name agrees with class_id, excluded rows all point
  to a kept train file, and the CSVs read back identically.

Writes docs/task2_4_partA_subject_csv.json.
Run (from repository root): uv run python src/freeze_splits.py
"""

import hashlib
import json
import sys
from datetime import datetime
import os

import pandas as pd

from config import CLASS_TO_ID, CLASSES, DOCS_DIR, PROJECT_ROOT, SPLITS_DIR
from .manifests import build_manifests

EXPECTED = {"train": 24777, "val": 3588, "test": 7200, "excluded": 423}   # measured in 
COLUMNS = ["filepath", "class_name", "class_id", "volunteer_id", "crop_path", "md5"]
REPORT = DOCS_DIR / "task2_4_partA_subject_csv.json"

TARGETS = {
    "train": SPLITS_DIR / "subject_train.csv",
    "val": SPLITS_DIR / "subject_val.csv",
    "test": SPLITS_DIR / "subject_test.csv",
    "excluded": SPLITS_DIR / "excluded_train_copies.csv",
}
MAPPING = SPLITS_DIR / "class_mapping.json"
CHECKSUMS = SPLITS_DIR / "SHA256SUMS.txt"


def to_frame(df):
    out = pd.DataFrame({
        "filepath": df["filepath"].astype(str),
        "class_name": df["label"].astype(str),
        "class_id": df["class_id"].astype(int),
        "volunteer_id": df["volunteer"].astype(int),
        "crop_path": df["crop_path"].astype(str),
        "md5": df["md5"].astype(str),
    })
    return out.sort_values(["class_id", "volunteer_id", "filepath"]).reset_index(drop=True)


def excluded_frame(df):
    out = to_frame(df.rename(columns={"label": "label"}))
    out.insert(len(out.columns), "kept_filepath", df["kept_filepath"].astype(str).values)
    return out.sort_values(["md5", "filepath"]).reset_index(drop=True)


def gate(results, name, ok, detail=""):
    results[name] = {"pass": bool(ok), "detail": detail}
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

STR_COLS = ["filepath", "class_name", "crop_path", "md5"]


def write_and_verify(frame, path, cols):
    """Write to a temporary file, read it back, compare values column by column, then rename.
    Compares as text, so dtype differences (int vs int64, object vs string) cannot cause a
    false failure. Returns (ok, diffs). On failure the temporary file is deleted."""
    tmp = path.with_name(path.name + ".tmp")
    expected = frame[cols].reset_index(drop=True)
    expected.to_csv(tmp, index=False, lineterminator="\n")
    back = pd.read_csv(tmp, dtype={c: str for c in STR_COLS if c in cols},
                       keep_default_na=False)
    diffs = []
    if len(back) != len(expected):
        diffs.append(f"row count {len(back)} != {len(expected)}")
    for c in cols:
        a = expected[c].astype(str).tolist()
        b = back[c].astype(str).tolist()
        bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if bad:
            i = bad[0]
            diffs.append(f"{c}: {len(bad)} rows differ; first at row {i}: "
                         f"wrote {a[i]!r}, read back {b[i]!r}")
    if diffs:
        tmp.unlink()
        return False, diffs
    os.replace(tmp, path)
    return True, []


def main():
    print("=" * 70)
    print("ASL-HG:  (FREEZE SUBJECT SPLIT TO CSV)")
    print("=" * 70)

    existing = [p for p in list(TARGETS.values()) + [MAPPING, CHECKSUMS] if p.exists()]
    if existing:
        print("Refusing to overwrite existing split files (write-once):")
        for p in existing:
            print(f"  {p.relative_to(PROJECT_ROOT).as_posix()}")
        print("A change requires a logged decision in consolidation Section 8 and "
              "a deliberate move of these files to an archive folder.")
        sys.exit(1)

    m = build_manifests()
    frames = {
        "train": to_frame(m["train"]),
        "val": to_frame(m["val"]),
        "test": to_frame(m["test"]),
        "excluded": excluded_frame(m["excluded"]),
    }

    results = {}
    print("\nGates")
    for k, df in frames.items():
        gate(results, f"count_{k}", len(df) == EXPECTED[k], f"{len(df)} (expected {EXPECTED[k]})")

    sv = {k: set(frames[k]["volunteer_id"]) for k in ("train", "val", "test")}
    gate(results, "no_signer_in_two_splits",
         sv["train"].isdisjoint(sv["val"]) and sv["val"].isdisjoint(sv["test"])
         and sv["train"].isdisjoint(sv["test"]),
         f"train {sorted(sv['train'])}, val {sorted(sv['val'])}, test {sorted(sv['test'])}")

    for k in ("train", "val", "test"):
        n = frames[k]["class_id"].nunique()
        gate(results, f"all_classes_in_{k}", n == len(CLASSES), f"{n} classes")

    gate(results, "train_md5_unique", frames["train"]["md5"].is_unique)

    name_ok = all(CLASS_TO_ID[n] == i for k in ("train", "val", "test")
                  for n, i in zip(frames[k]["class_name"], frames[k]["class_id"]))
    gate(results, "class_name_matches_class_id", name_ok)

    missing = sum(not (PROJECT_ROOT / p).exists()
                  for k in ("train", "val", "test") for p in frames[k]["crop_path"])
    gate(results, "every_crop_exists", missing == 0, f"missing {missing}")

    train_md5 = set(frames["train"]["md5"])
    exc_ok = frames["excluded"]["kept_filepath"].isin(set(frames["train"]["filepath"])).all() \
        and frames["excluded"]["md5"].isin(train_md5).all()
    gate(results, "excluded_point_to_kept_train_files", bool(exc_ok))

    passed = all(v["pass"] for v in results.values())
    if not passed:
        print("\nVERDICT: FAILED. No files written.")
        sys.exit(1)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for k, path in TARGETS.items():
        cols = COLUMNS if k != "excluded" else COLUMNS + ["kept_filepath"]
        ok, diffs = write_and_verify(frames[k], path, cols)
        gate(results, f"read_back_{k}", ok, "; ".join(diffs))
        all_ok &= ok
    if not all_ok:
        print("\nVERDICT: FAILED on read-back. No split file was written.")
        sys.exit(1)

    mapping = {"class_to_id": CLASS_TO_ID, "id_to_class": CLASSES, "total_classes": len(CLASSES)}
    MAPPING.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    for path in list(TARGETS.values()) + [MAPPING]:
        lines.append(f"{sha256(path)}  {path.name}")
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nWritten:")
    for path in list(TARGETS.values()) + [MAPPING, CHECKSUMS]:
        print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"\nVERDICT: {'PASSED' if all_ok else 'FAILED (read-back mismatch)'}")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "counts": {k: len(v) for k, v in frames.items()},
        "per_class_train": frames["train"]["class_name"].value_counts().reindex(CLASSES).to_dict(),
        "gates": results,
        "passed": bool(all_ok),
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Report: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()