"""
src/prepare_external.py
 Normalise, crop, cryptographically check external photos, and generate
data/external_test/manifest.csv without requiring any external CSV input.

Design Decisions (Academic & Engineering Rationale):
  - Direct File Discovery: Metadata (volunteer_name, class_name, class_id) is inferred
    directly from folder paths and filenames: <member>_<class>_<idx>.jpg.
  - Photo Set Allocation : Exactly 36 classes x 5 photos = 180 total.
    * Exactly 4 original photos per class (contributed by team members).
    * Exactly 1 horizontally mirrored photo per class (hck_<class>_02.jpg).
    * Headline metrics evaluate ORIGINALS ONLY (; 144 photos). All-photo metrics (180) reported second.
  - Pixel-Level Mirror Verification :
    * Mirroring is verified directly from pixel difference hashing (64-bit dHash, 9x8 grid).
    * Every detected mirror pair must consist of one original and one mirrored photo.
  - Frame Normalisation :
    * Longer side scaled to 300 px (Mendeley frame format), shorter side padded to 300x300
      with neutral gray RGB (128, 128, 128) using INTER_AREA downsampling.
  - Pipeline Identity (, , ):
    * Reuses preprocess_roi.process_one verbatim: class-0 full-frame rule, MediaPipe detection,
      20% dynamic margin padding, and letterbox padding to 224x224.
  - Domain-Shift Tolerance :
    * Real webcam hands naturally vary in distance compared to laboratory captures.
    * DOMAIN_MIN_FRAC is set to 85% (measured external overlap: ~87.8% inside training p1-p99).

Gates (All must pass for manifest creation):
  every photo decodes; folders match class vocabulary; filenames strictly match folder;
  filenames unique per class; total is exactly 180; exactly 5 per class; exactly 4 originals
  and 1 mirrored per class; mirror pairs consistent with pixel hashes; volunteer names are text;
  zero byte-identical matches with Mendeley dataset; every photo cropped successfully; >=85%
  letterbox scale within training p1-p99 range ; atomic manifest read-back verification.

Writes:
  data/external_test/manifest.csv            (tracked by git; raw photos untracked)
  data/processed/external_crop_log.csv       (per-crop operational metrics)
  docs/task2_4_partC_external.json           (audit report)

Run (from repository root): uv run python src/prepare_external.py
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import preprocess_roi as pr
from blank_scan import edge_features
from config import CLASS_TO_ID, DOCS_DIR, PROCESSED_DIR, PROJECT_ROOT
from freeze_splits import gate, write_and_verify

# Core paths
EXT_DIR = PROJECT_ROOT / "data" / "external_test"
NORM_DIR = PROJECT_ROOT / "data" / "external_processed"
CROP_DIR = PROJECT_ROOT / "data" / "external_crops"
MANIFEST = EXT_DIR / "manifest.csv"
INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
LOG_OUT = PROCESSED_DIR / "external_crop_log.csv"
REPORT = DOCS_DIR / "task2_4_partC_external.json"

# Constants & Decision Thresholds
FRAME = pr.IMG_SIZE                       # 300 px Mendeley capture frame
IMG_EXT = {".jpg", ".jpeg", ".png"}
PER_CLASS_ORIGINAL = 4
PER_CLASS_MIRRORED = 1
PER_CLASS_TOTAL = PER_CLASS_ORIGINAL + PER_CLASS_MIRRORED        # 5
TOTAL = len(CLASS_TO_ID) * PER_CLASS_TOTAL                        # 180
DOMAIN_MIN_FRAC = 0.85                    # : Real-world camera distance domain tolerance
TOLERANCE = 1e-3                          # Floating point boundary tolerance
BLANK_REVIEW = 0.001                      # Blank frame detection threshold
MIRROR_MAX_DIST = 5                       # dHash threshold for horizontal reflection
MIRROR_MIN_ORIG_DIST = 10                 # Rejects self-symmetric signs from false mirror matches
LIST_SHOW = 5


def normalise(bgr):
    """: Resizes longest side to 300 px (INTER_AREA) and pads symmetrically to 300x300."""
    h, w = bgr.shape[:2]
    s = FRAME / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
    canvas = np.full((FRAME, FRAME, 3), pr.PAD_VALUE, dtype=np.uint8)
    top, left = (FRAME - nh) // 2, (FRAME - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def dhash_bits(bgr):
    """64-bit difference hash on 9x8 grid matching  audit resolution."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).flatten()


def hamming(x, y):
    """Bit-level Hamming distance between two binary hashes."""
    return int(np.count_nonzero(x != y))


def scan_photos():
    """
    Directly scans data/external_test/ and extracts metadata from file paths:
      - Validates parent folder belongs to 36 valid class names.
      - Enforces <member>_<class>_<idx> naming convention.
      - Assigns 'source' tag: hck_*_02 is tagged 'mirrored' , all others 'original'.
    """
    rows, bad_folder, bad_name = [], [], []
    for p in sorted(EXT_DIR.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        rel = p.relative_to(EXT_DIR).as_posix()
        folder = p.parent.name
        if folder not in CLASS_TO_ID:
            bad_folder.append(rel)
            continue
        parts = p.stem.split("_")
        if len(parts) != 3 or parts[1] != folder:
            bad_name.append(rel)
            continue
        
        member = parts[0]
        idx = parts[2]
        
        #  convention: member hck's _02 image is the designated mirrored partner
        source = "mirrored" if member == "hck" and idx in {"02", "2"} else "original"

        rows.append({
            "rel_path": rel,
            "path": p,
            "class_name": folder,
            "class_id": CLASS_TO_ID[folder],
            "volunteer_name": member,
            "source": source
        })

    photos = pd.DataFrame(rows, columns=[
        "rel_path", "path", "class_name", "class_id", "volunteer_name", "source"
    ])
    return photos, bad_folder, bad_name


def find_mirror_pairs(photos):
    """Detects horizontal reflection pairs using perceptual 64-bit dHash comparison."""
    feats = {}
    for r in photos.itertuples(index=False):
        bgr = cv2.imread(str(r.path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        feats[r.rel_path] = (r.class_name, dhash_bits(bgr), dhash_bits(cv2.flip(bgr, 1)))

    keys = sorted(feats)
    pairs = []
    for i, a in enumerate(keys):
        cls_a, h_a, f_a = feats[a]
        for b in keys[i + 1:]:
            cls_b, h_b, f_b = feats[b]
            if cls_a != cls_b:
                continue
            d_flip = min(hamming(f_a, h_b), hamming(h_a, f_b))
            d_orig = hamming(h_a, h_b)
            if d_flip <= MIRROR_MAX_DIST and d_orig >= MIRROR_MIN_ORIG_DIST:
                pairs.append({
                    "a": a, "b": b, "class_name": cls_a,
                    "d_flip": d_flip, "d_orig": d_orig
                })
    return pairs


def short(items):
    items = list(items)
    tail = f" ... (+{len(items) - LIST_SHOW} more)" if len(items) > LIST_SHOW else ""
    return f"{items[:LIST_SHOW]}{tail}"


def main():
    print("=" * 70)
    print("ASL-HG:  (EXTERNAL CROP, AUDIT & MANIFEST GENERATOR)")
    print("=" * 70)

    # 1. Discover photos directly on disk
    photos, bad_folder, bad_name = scan_photos()
    n_total = len(photos)
    print(f"Photos discovered on disk: {n_total}")

    if n_total == 0:
        print("[FAIL] No valid photos found under data/external_test/. Check folder layout.")
        sys.exit(1)

    # Compute MD5 hashes for byte-duplicate checks
    photos["md5"] = [hashlib.md5(p.read_bytes()).hexdigest() for p in photos["path"]]
    name_keys = pd.DataFrame({"c": photos["class_name"], "s": [p.stem for p in photos["path"]]})
    dup_names = int(name_keys.duplicated().sum())

    index = pd.read_csv(INDEX_CSV, dtype={"md5": str})
    mendeley_md5 = set(index["md5"])

    # 2. Normalise , crop ( pipeline), and record metrics
    pr._init_worker()
    log_rows, decode_fail = [], []
    for r in photos.itertuples(index=False):
        bgr = cv2.imread(str(r.path), cv2.IMREAD_COLOR)  # Honors phone EXIF orientation
        if bgr is None:
            decode_fail.append(r.rel_path)
            continue
        
        # Save normalized 300x300 intermediate
        out_norm = NORM_DIR / r.class_name / (r.path.stem + ".png")
        out_norm.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_norm), normalise(bgr))

        # Pass through the exact  processing pipeline
        rel_norm = out_norm.relative_to(PROJECT_ROOT).as_posix()
        res = pr.process_one((rel_norm, r.class_name, 0, False, CROP_DIR))
        res["rel_path"] = r.rel_path
        res["edge_frac_40"] = edge_features(out_norm)["edge_frac_40"]
        log_rows.append(res)

    log = pd.DataFrame(log_rows)
    log.to_csv(LOG_OUT, index=False)

    if len(log) and "method" in log.columns:
        written = log[log["method"].isin(pr.WRITTEN_METHODS)].copy()
    else:
        written = pd.DataFrame(columns=["rel_path", "label", "scale", "method", "edge_frac_40"])

    # 3. Training scale distribution reference
    train_log = pd.read_csv(pr.LOG_CSV)
    tw = train_log[train_log["method"].isin(pr.WRITTEN_METHODS)]
    side = np.maximum(tw["margin_x1"] - tw["margin_x0"], tw["margin_y1"] - tw["margin_y0"]) + 1
    lo, hi = np.percentile(pr.CROP_SIZE / side, [1, 99])

    # Calculate scale overlap with boundary tolerance
    in_range = float(((written["scale"] >= (lo - TOLERANCE)) & (written["scale"] <= (hi + TOLERANCE))).mean()) \
        if len(written) else 0.0

    # 4. Pixel-level mirror analysis
    pairs = find_mirror_pairs(photos)
    src_map = dict(zip(photos["rel_path"], photos["source"]))
    paired_files, bad_pairs = set(), []
    for p in pairs:
        paired_files |= {p["a"], p["b"]}
        if {src_map.get(p["a"]), src_map.get(p["b"])} != {"original", "mirrored"}:
            bad_pairs.append(p)

    declared_mirrored = set(photos.loc[photos["source"] == "mirrored", "rel_path"])
    orphan_mirrors = sorted(declared_mirrored - paired_files)

    # 5. Class balance counters
    per_total = photos["class_name"].value_counts().reindex(list(CLASS_TO_ID), fill_value=0)
    per_orig = photos.loc[photos["source"] == "original", "class_name"] \
        .value_counts().reindex(list(CLASS_TO_ID), fill_value=0)
    per_mirr = photos.loc[photos["source"] == "mirrored", "class_name"] \
        .value_counts().reindex(list(CLASS_TO_ID), fill_value=0)

    n_orig = int((photos["source"] == "original").sum())
    n_mirr = int((photos["source"] == "mirrored").sum())
    dup = int(photos["md5"].isin(mendeley_md5).sum())

    # 6. Strict Quality Gates Execution
    results = {}
    print("\nGates")
    gate(results, "folders_are_classes", not bad_folder, f"bad: {short(bad_folder)}")
    gate(results, "filenames_match_folder", not bad_name, f"bad: {short(bad_name)}")
    gate(results, "every_photo_decodes", not decode_fail, f"failed: {short(decode_fail)}")
    gate(results, "filenames_unique_per_class", dup_names == 0, f"{dup_names} duplicates")
    gate(results, "total_is_180", n_total == TOTAL, f"{n_total} photos ({n_orig} orig, {n_mirr} mirrored)")
    
    bad_total = per_total[per_total != PER_CLASS_TOTAL]
    gate(results, "exactly_5_per_class", bad_total.empty,
         f"classes not at 5: {bad_total.to_dict()}" if not bad_total.empty else "all 36 at 5")
    
    bad_orig = per_orig[per_orig != PER_CLASS_ORIGINAL]
    bad_mirr = per_mirr[per_mirr != PER_CLASS_MIRRORED]
    gate(results, "exactly_4_original_per_class", bad_orig.empty,
         f"classes off target: {bad_orig.to_dict()}" if not bad_orig.empty else "all 36 at 4")
    gate(results, "exactly_1_mirrored_per_class", bad_mirr.empty,
         f"classes off target: {bad_mirr.to_dict()}" if not bad_mirr.empty else "all 36 at 1")
    
    gate(results, "mirror_pairs_are_original_plus_mirrored", not bad_pairs,
         f"{len(bad_pairs)} pairs with wrong source labels: {bad_pairs[:3]}")
    gate(results, "every_mirrored_has_partner", not orphan_mirrors,
         f"mirrored photos with no detected partner: {short(orphan_mirrors)}")
    gate(results, "volunteer_names_are_text", not any(str(v).isdigit() for v in photos["volunteer_name"]))
    gate(results, "no_mendeley_duplicates", dup == 0, f"{dup} byte-identical matches")
    gate(results, "every_photo_cropped", n_total > 0 and len(written) == n_total, f"{len(written)} of {n_total}")
    gate(results, "domain_scale_in_training_range", in_range >= DOMAIN_MIN_FRAC,
         f"{in_range:.1%} inside [{lo:.3f}, {hi:.3f}] (threshold {DOMAIN_MIN_FRAC:.0%}, )")

    flagged = log[log["edge_frac_40"] <= BLANK_REVIEW] if len(log) else log
    print(f"\nInformational: blank-review flags (edge_frac_40 <= {BLANK_REVIEW}): {len(flagged)}")
    print(f"Informational: mirror pairs verified: {len(pairs)}")

    if not all(v["pass"] for v in results.values()):
        print("\nVERDICT: FAILED. Manifest not written. Fix errors and re-run.")
        sys.exit(1)

    # 7. Write Manifest Directly (Excluding camera/lighting/environment columns)
    man = pd.DataFrame({
        "filepath": "data/external_test/" + photos["rel_path"],
        "class_name": photos["class_name"],
        "class_id": photos["class_id"],
        "volunteer_name": photos["volunteer_name"],
        "source": photos["source"],
        "is_mirrored": photos["source"] == "mirrored",
        "md5": photos["md5"],
    }).sort_values("filepath").reset_index(drop=True)

    ok, diffs = write_and_verify(man, MANIFEST, list(man.columns))
    gate(results, "manifest_read_back", ok, "; ".join(diffs))
    if not ok:
        print("\nVERDICT: FAILED on manifest read-back.")
        sys.exit(1)

    print("\nPer-class photos (originals / mirrored):")
    for c in CLASS_TO_ID:
        print(f"  {c:>2}: {per_orig[c]} / {per_mirr[c]}", end="   ")
    print()
    print(f"Headline benchmark set (originals only, ) : {n_orig} photos")
    print(f"Full benchmark set (originals + mirrored)     : {n_total} photos")
    print(f"Scale: {in_range:.1%} of external crops inside the training range ")
    print(f"Manifest: {MANIFEST.relative_to(PROJECT_ROOT).as_posix()} ({n_total} rows)")
    print("VERDICT: PASSED")

    # 8. Export JSON report
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decision": ": 5 per class (4 original, 1 mirrored); : headline on originals; : 85% scale threshold",
        "frame_px": FRAME,
        "normalisation": "longer side to 300 (INTER_AREA), neutral gray (128,128,128) pad",
        "photos_total": n_total,
        "photos_original": n_orig,
        "photos_mirrored": n_mirr,
        "headline_set": "originals only",
        "per_class_original": {c: int(per_orig[c]) for c in CLASS_TO_ID},
        "per_class_mirrored": {c: int(per_mirr[c]) for c in CLASS_TO_ID},
        "mirror_pairs": pairs,
        "training_scale_range_p1_p99": [round(float(lo), 4), round(float(hi), 4)],
        "domain_min_frac_gate": DOMAIN_MIN_FRAC,
        "external_fraction_in_range": round(in_range, 4),
        "external_scale_median_by_class": (
            written.groupby("label")["scale"].median().round(4).to_dict() if len(written) else {}),
        "blank_review_flags": flagged["rel_path"].tolist() if len(flagged) else [],
        "gates": results,
        "passed": True,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Report: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    sys.exit(0)


if __name__ == "__main__":
    main()