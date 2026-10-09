import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CLASS_TO_ID, CLASSES, DOCS_DIR, PROJECT_ROOT, SPLITS_DIR

from .manifests import build_manifests

EXPECTED_COUNTS: dict[str, int] = {
    "train": 24777,
    "val": 3588,
    "test": 7200,
    "excluded": 423,
}
COLUMNS: list[str] = [
    "filepath",
    "class_name",
    "class_id",
    "volunteer_id",
    "crop_path",
    "md5",
]
REPORT: Path = DOCS_DIR / "metrics" / "split_subject.json"

TARGETS: dict[str, Path] = {
    "train": SPLITS_DIR / "subject_train.csv",
    "val": SPLITS_DIR / "subject_val.csv",
    "test": SPLITS_DIR / "subject_test.csv",
    "excluded": SPLITS_DIR / "excluded_train_copies.csv",
}
MAPPING: Path = SPLITS_DIR / "class_mapping.json"
CHECKSUMS: Path = SPLITS_DIR / "SHA256SUMS.txt"
STR_COLS: list[str] = ["filepath", "class_name", "crop_path", "md5"]


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize split DataFrame columns and sorting."""
    out = pd.DataFrame(
        {
            "filepath": df["filepath"].astype(str),
            "class_name": df["label"].astype(str),
            "class_id": df["class_id"].astype(int),
            "volunteer_id": df["volunteer"].astype(int),
            "crop_path": df["crop_path"].astype(str),
            "md5": df["md5"].astype(str),
        }
    )
    return out.sort_values(["class_id", "volunteer_id", "filepath"]).reset_index(
        drop=True
    )


def excluded_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame for excluded duplicate training records."""
    out = to_frame(df.rename(columns={"label": "label"}))
    out.insert(
        len(out.columns),
        "kept_filepath",
        df["kept_filepath"].astype(str).values,
    )
    return out.sort_values(["md5", "filepath"]).reset_index(drop=True)


def gate(
    results: dict[str, dict[str, Any]], name: str, ok: bool, detail: str = ""
) -> None:
    """Record verification gate status and print formatted output."""
    results[name] = {"pass": bool(ok), "detail": detail}
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash digest of a given file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_and_verify(
    frame: pd.DataFrame, path: Path, cols: list[str]
) -> tuple[bool, list[str]]:
    """Write DataFrame to temporary file, read back to verify data integrity, then replace."""
    tmp = path.with_name(path.name + ".tmp")
    expected = frame[cols].reset_index(drop=True)
    expected.to_csv(tmp, index=False, lineterminator="\n")
    back = pd.read_csv(
        tmp, dtype={c: str for c in STR_COLS if c in cols}, keep_default_na=False
    )
    diffs = []
    if len(back) != len(expected):
        diffs.append(f"Row count mismatch: {len(back)} != {len(expected)}")
    for c in cols:
        a = expected[c].astype(str).tolist()
        b = back[c].astype(str).tolist()
        bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if bad:
            i = bad[0]
            diffs.append(
                f"{c}: {len(bad)} rows differ; row {i}: wrote {a[i]!r}, read back {b[i]!r}"
            )
    if diffs:
        tmp.unlink()
        return False, diffs
    os.replace(tmp, path)
    return True, []


def main() -> None:
    print("ASL-HG: Subject-Independent Dataset Split Generation")

    existing = [
        p for p in list(TARGETS.values()) + [MAPPING, CHECKSUMS] if p.exists()
    ]
    if existing:
        print("Existing split files detected. Archive existing splits before re-generating.")
        sys.exit(1)

    manifests = build_manifests()
    frames = {
        "train": to_frame(manifests["train"]),
        "val": to_frame(manifests["val"]),
        "test": to_frame(manifests["test"]),
        "excluded": excluded_frame(manifests["excluded"]),
    }

    results: dict[str, dict[str, Any]] = {}
    for k, df in frames.items():
        gate(
            results,
            f"count_{k}",
            len(df) == EXPECTED_COUNTS[k],
            f"{len(df)} (expected {EXPECTED_COUNTS[k]})",
        )

    signers = {
        k: set(frames[k]["volunteer_id"]) for k in ("train", "val", "test")
    }
    gate(
        results,
        "no_signer_overlap",
        signers["train"].isdisjoint(signers["val"])
        and signers["val"].isdisjoint(signers["test"])
        and signers["train"].isdisjoint(signers["test"]),
        f"train {sorted(signers['train'])}, val {sorted(signers['val'])}, test {sorted(signers['test'])}",
    )

    for k in ("train", "val", "test"):
        n_classes = frames[k]["class_id"].nunique()
        gate(
            results,
            f"all_classes_in_{k}",
            n_classes == len(CLASSES),
            f"{n_classes} classes",
        )

    gate(results, "train_md5_unique", frames["train"]["md5"].is_unique)

    name_ok = all(
        CLASS_TO_ID[n] == i
        for k in ("train", "val", "test")
        for n, i in zip(frames[k]["class_name"], frames[k]["class_id"])
    )
    gate(results, "class_name_matches_class_id", name_ok)

    missing = sum(
        not (PROJECT_ROOT / p).exists()
        for k in ("train", "val", "test")
        for p in frames[k]["crop_path"]
    )
    gate(results, "every_crop_exists", missing == 0, f"missing {missing}")

    train_md5 = set(frames["train"]["md5"])
    exc_ok = (
        frames["excluded"]["kept_filepath"]
        .isin(set(frames["train"]["filepath"]))
        .all()
        and frames["excluded"]["md5"].isin(train_md5).all()
    )
    gate(results, "excluded_point_to_kept_train_files", bool(exc_ok))

    passed = all(v["pass"] for v in results.values())
    if not passed:
        print("VERDICT: FAILED. Split creation aborted.")
        sys.exit(1)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for k, path in TARGETS.items():
        cols = COLUMNS if k != "excluded" else COLUMNS + ["kept_filepath"]
        ok, diffs = write_and_verify(frames[k], path, cols)
        gate(results, f"read_back_{k}", ok, "; ".join(diffs))
        all_ok &= ok
    if not all_ok:
        print("VERDICT: FAILED during read-back verification.")
        sys.exit(1)

    mapping = {
        "class_to_id": CLASS_TO_ID,
        "id_to_class": CLASSES,
        "total_classes": len(CLASSES),
    }
    MAPPING.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"{compute_sha256(path)}  {path.name}"
        for path in list(TARGETS.values()) + [MAPPING]
    ]
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": {k: len(v) for k, v in frames.items()},
        "per_class_train": frames["train"]["class_name"]
        .value_counts()
        .reindex(CLASSES)
        .to_dict(),
        "gates": results,
        "passed": bool(all_ok),
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
