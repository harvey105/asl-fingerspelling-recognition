"""
src/verify_dataset.py
Task 2.1, Part A: structure, class counts, filename metadata, full decode, exact duplicates.

Read-only. Never writes to data/raw/. Writes docs/task2_1_partA_inspection.json.
Exit code: 0 if every blocking check passes, 1 otherwise. The JSON report is written either way.

Run (from the repository root): uv run python src/verify_dataset.py
"""

import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import (
    CLASSES,
    DOCS_DIR,
    EXPECTED_MODE,
    EXPECTED_SIZE,
    IMAGE_EXTENSIONS,
    IMAGES_PER_CLASS,
    IMAGES_PER_VOLUNTEER_PER_CLASS,
    NUM_VOLUNTEERS,
    PROJECT_ROOT,
    RAW_DIR,
    TOTAL_IMAGES,
    find_dataset_root,
    parse_filename,
)

REPORT_PATH = DOCS_DIR / "task2_1_partA_inspection.json"
NUM_WORKERS = 8
LIST_LIMIT = 20


def rel(path, base=PROJECT_ROOT):
    return path.relative_to(base).as_posix()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


# ---------------------------------------------------------------- structure
def scan_layout(root):
    """Describe every class folder and return the image paths per class."""
    root_entries = list(root.iterdir())
    root_dirs = sorted(p.name for p in root_entries if p.is_dir())
    root_files = sorted(p.name for p in root_entries if p.is_file())

    per_class, image_paths = {}, {}
    for cls in CLASSES:
        cdir = root / cls
        if not cdir.is_dir():
            per_class[cls] = {"present": False, "images": 0, "non_image_entries": 0,
                              "subdirs": [], "non_image_examples": []}
            image_paths[cls] = []
            continue
        entries = list(cdir.iterdir())
        imgs = sorted(p for p in entries if is_image(p))
        others = sorted(p for p in entries if not is_image(p))
        per_class[cls] = {
            "present": True,
            "images": len(imgs),
            "non_image_entries": len(others),
            "subdirs": sorted(p.name for p in others if p.is_dir())[:10],
            "non_image_examples": [p.name for p in others[:5]],
        }
        image_paths[cls] = imgs

    layout = {
        "root_dirs": root_dirs,
        "root_files": root_files,
        "unexpected_root_dirs": sorted(set(root_dirs) - set(CLASSES)),
        "missing_classes": [c for c in CLASSES if c not in root_dirs],
        "per_class": per_class,
    }
    return layout, image_paths


def find_non_image_files(raw_dir):
    """Any file under raw/ that is not an image (zips, CSVs, JSON, notes, Thumbs.db, ...)."""
    return sorted(rel(p, raw_dir) for p in raw_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS)


# ---------------------------------------------------------------- decoding
def inspect_image(path):
    """Read the file once: MD5 of the raw bytes, then a full decode from memory."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"error": f"read: {exc}", "md5": None, "size": None, "mode": None}
    if not data:
        return {"error": "zero-byte", "md5": None, "size": None, "mode": None}
    md5 = hashlib.md5(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()                                   # full pixel decode, not header only
            return {"error": None, "md5": md5, "size": list(im.size), "mode": im.mode}
    except Exception as exc:
        return {"error": f"decode: {exc}", "md5": md5, "size": None, "mode": None}


# ---------------------------------------------------------------- audit
def audit(raw_dir):
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": rel(raw_dir) if raw_dir.exists() else str(raw_dir),
        "expected": {
            "classes": len(CLASSES),
            "images_per_class": IMAGES_PER_CLASS,
            "total_images": TOTAL_IMAGES,
            "volunteers": NUM_VOLUNTEERS,
            "images_per_volunteer_per_class": IMAGES_PER_VOLUNTEER_PER_CLASS,
            "size": list(EXPECTED_SIZE),
            "mode": EXPECTED_MODE,
        },
        "failures": [],
        "warnings": [],
        "passed": False,
    }
    failures, warnings = report["failures"], report["warnings"]

    try:
        root = find_dataset_root(raw_dir)
    except (FileNotFoundError, ValueError) as exc:
        failures.append(str(exc))
        return report

    report["dataset_root"] = rel(root)
    report["raw_top_level"] = sorted(p.name + ("/" if p.is_dir() else "") for p in raw_dir.iterdir())

    # 1. Layout
    layout, image_paths = scan_layout(root)
    report["layout"] = layout
    if layout["missing_classes"]:
        failures.append(f"missing class folders: {layout['missing_classes']}")
    if layout["unexpected_root_dirs"]:
        warnings.append(f"unexpected folders next to class folders: {layout['unexpected_root_dirs']}")

    counts = {c: layout["per_class"][c]["images"] for c in CLASSES}
    off_counts = {c: n for c, n in counts.items() if n != IMAGES_PER_CLASS}
    if off_counts:
        failures.append(f"per-class image count differs from {IMAGES_PER_CLASS}: {off_counts}")
    total = sum(counts.values())
    if total != TOTAL_IMAGES:
        failures.append(f"total images {total} != {TOTAL_IMAGES}")

    non_image = find_non_image_files(raw_dir)
    report["non_image_files_under_raw"] = {"count": len(non_image), "first_50": non_image[:50]}
    if non_image:
        warnings.append(f"{len(non_image)} non-image file(s) under raw/ (see report)")

    # 2. Decode every image and read its metadata
    entries = [(cls, p) for cls in CLASSES for p in image_paths[cls]]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        results = list(tqdm(pool.map(inspect_image, [p for _, p in entries]),
                            total=len(entries), desc="Decoding", unit="img"))

    size_counts, mode_counts = Counter(), Counter()
    decode_fail, size_bad, mode_bad = [], [], []
    md5_index = defaultdict(list)                        # md5 -> [(class, rel_path), ...]
    for (cls, path), res in zip(entries, results):
        rp = rel(path)
        if res["md5"]:
            md5_index[res["md5"]].append((cls, rp))
        if res["error"]:
            decode_fail.append({"file": rp, "error": res["error"]})
            continue
        w, h = res["size"]
        size_counts[f"{w}x{h}"] += 1
        mode_counts[res["mode"]] += 1
        if (w, h) != tuple(EXPECTED_SIZE):
            size_bad.append(rp)
        if res["mode"] != EXPECTED_MODE:
            mode_bad.append(rp)

    report["decoding"] = {
        "decoded_ok": len(entries) - len(decode_fail),
        "failed": len(decode_fail),
        "failures_first": decode_fail[:LIST_LIMIT],
        "size_counts": dict(size_counts),
        "mode_counts": dict(mode_counts),
        "size_mismatch_count": len(size_bad),
        "size_mismatch_examples": size_bad[:LIST_LIMIT],
        "mode_mismatch_count": len(mode_bad),
        "mode_mismatch_examples": mode_bad[:LIST_LIMIT],
    }
    if decode_fail:
        failures.append(f"{len(decode_fail)} file(s) failed to read or decode")
    if size_bad:
        failures.append(f"{len(size_bad)} image(s) are not {EXPECTED_SIZE}")
    if mode_bad:
        failures.append(f"{len(mode_bad)} image(s) are not {EXPECTED_MODE}")

    # 3. Filenames: volunteer IDs, class consistency, index uniqueness
    pair_counts = Counter()                              # (volunteer, folder_class) -> n
    key_counts = Counter()                               # (volunteer, file_class, index) -> n
    unparsed, class_mismatch = [], []
    for cls, path in entries:
        meta = parse_filename(path.name)
        if meta is None:
            unparsed.append(path.name)
            continue
        vol, file_cls, idx = meta
        if file_cls != cls:
            class_mismatch.append(rel(path))
        pair_counts[(vol, cls)] += 1
        key_counts[(vol, file_cls, idx)] += 1

    vols_seen = sorted({v for v, _ in pair_counts})
    expected_vols = list(range(1, NUM_VOLUNTEERS + 1))
    missing_vols = [v for v in expected_vols if v not in vols_seen]
    unexpected_vols = [v for v in vols_seen if v not in expected_vols]
    off_cells = [(v, c, pair_counts[(v, c)]) for v in vols_seen for c in CLASSES
                 if pair_counts[(v, c)] != IMAGES_PER_VOLUNTEER_PER_CLASS]
    dup_keys = sum(1 for n in key_counts.values() if n > 1)

    report["filenames"] = {
        "parsed": len(entries) - len(unparsed),
        "unparsed": len(unparsed),
        "unparsed_examples": unparsed[:LIST_LIMIT],
        "volunteers_seen": vols_seen,
        "missing_volunteers": missing_vols,
        "unexpected_volunteers": unexpected_vols,
        "volunteer_class_matrix": {str(v): {c: pair_counts[(v, c)] for c in CLASSES} for v in vols_seen},
        "cells_off_target_count": len(off_cells),
        "cells_off_target_first": off_cells[:LIST_LIMIT],
        "filename_class_vs_folder_mismatch": len(class_mismatch),
        "mismatch_examples": class_mismatch[:LIST_LIMIT],
        "duplicate_volunteer_class_index_keys": dup_keys,
    }
    if class_mismatch:
        failures.append(f"{len(class_mismatch)} file(s) whose filename class differs from its folder")
    if unparsed:
        warnings.append(f"{len(unparsed)} filename(s) do not match P<id>_<class>_<idx>.<ext>")
    if off_cells:
        warnings.append(f"{len(off_cells)} volunteer/class cell(s) differ from "
                        f"{IMAGES_PER_VOLUNTEER_PER_CLASS}")
    if missing_vols:
        warnings.append(f"volunteers not found in filenames: {missing_vols}")
    if unexpected_vols:
        warnings.append(f"unexpected volunteer IDs in filenames: {unexpected_vols}")
    if dup_keys:
        warnings.append(f"{dup_keys} duplicate (volunteer, class, index) key(s)")

    # 4. Exact duplicates (byte-identical files)
    groups = [g for g in md5_index.values() if len(g) > 1]
    cross = [g for g in groups if len({c for c, _ in g}) > 1]
    report["exact_duplicates"] = {
        "groups": len(groups),
        "files_in_groups": sum(len(g) for g in groups),
        "groups_spanning_classes": len(cross),
        "examples": [[rp for _, rp in g][:5] for g in groups[:LIST_LIMIT]],
        "cross_class_examples": [[rp for _, rp in g][:5] for g in cross[:LIST_LIMIT]],
    }
    if groups:
        warnings.append(f"{len(groups)} group(s) of byte-identical images "
                        f"({sum(len(g) for g in groups)} files)")
    if cross:
        warnings.append(f"{len(cross)} group(s) of identical images span different classes "
                        f"(possible label conflict)")

    report["passed"] = not failures
    return report


# ---------------------------------------------------------------- output
def print_summary(report):
    line = "=" * 70
    print("\n" + line)
    print("RESULTS")
    print(line)

    if "dataset_root" not in report:
        print("Dataset root could not be resolved.")
    else:
        print(f"Raw top level      : {report['raw_top_level']}")
        print(f"Dataset root       : {report['dataset_root']}")
        print(f"Root subfolders    : {len(report['layout']['root_dirs'])}  "
              f"(missing classes: {report['layout']['missing_classes'] or 'none'})")
        print(f"Non-image files    : {report['non_image_files_under_raw']['count']}")

        print("\nImages per class:")
        cells = [f"{c}:{report['layout']['per_class'][c]['images']}" for c in CLASSES]
        for i in range(0, len(cells), 9):
            print("  " + "  ".join(f"{x:<9}" for x in cells[i:i + 9]))

        dec = report["decoding"]
        print(f"\nDecoded OK         : {dec['decoded_ok']}   Failed: {dec['failed']}")
        print(f"Sizes seen         : {dec['size_counts']}")
        print(f"Modes seen         : {dec['mode_counts']}")

        fn = report["filenames"]
        print(f"\nFilenames parsed   : {fn['parsed']}   Unparsed: {fn['unparsed']}")
        print(f"Volunteers seen    : {fn['volunteers_seen']}")
        if fn["volunteer_class_matrix"]:
            totals = {v: sum(row.values()) for v, row in fn["volunteer_class_matrix"].items()}
            print(f"Images per volunteer: {totals}")
        print(f"Cells off target   : {fn['cells_off_target_count']}")
        print(f"Filename/folder mismatches: {fn['filename_class_vs_folder_mismatch']}")

        ed = report["exact_duplicates"]
        print(f"\nByte-identical groups: {ed['groups']} ({ed['files_in_groups']} files), "
              f"spanning classes: {ed['groups_spanning_classes']}")

    print("\n" + line)
    if report["passed"]:
        print("VERDICT: PASSED (structure and decoding)")
    else:
        print("VERDICT: FAILED")
    for f in report["failures"]:
        print(f"  FAIL: {f}")
    for w in report["warnings"]:
        print(f"  WARN: {w}")
    print(line)


def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.1 PART A (STRUCTURE, COUNTS, FILENAMES, DECODING)")
    print(f"Raw folder: {RAW_DIR}")
    print("=" * 70)

    report = audit(RAW_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)
    print(f"\nReport saved to: {rel(REPORT_PATH)}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()