import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.config import CLASS_TO_ID, DOCS_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.data import preprocess_roi as pr
from src.data.splits import gate, write_and_verify

EXT_DIR = PROJECT_ROOT / "data" / "external_test"
NORM_DIR = PROJECT_ROOT / "data" / "external_processed"
CROP_DIR = PROJECT_ROOT / "data" / "external_crops"
MANIFEST = EXT_DIR / "manifest.csv"
INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
LOG_OUT = PROCESSED_DIR / "external_crop_log.csv"
REPORT = DOCS_DIR / "metrics" / "external_test.json"

FRAME = pr.IMG_SIZE
IMG_EXT = {".jpg", ".jpeg", ".png"}
PER_CLASS_ORIGINAL = 4
PER_CLASS_MIRRORED = 1
PER_CLASS_TOTAL = PER_CLASS_ORIGINAL + PER_CLASS_MIRRORED
TOTAL = len(CLASS_TO_ID) * PER_CLASS_TOTAL
DOMAIN_MIN_FRAC = 0.85
BLANK_REVIEW = 0.001
MIRROR_MAX_DIST = 5
MIRROR_MIN_ORIG_DIST = 10


def edge_features(path: Path) -> dict[str, float]:
    """Calculate Sobel gradient edge fraction matching blank frame detector."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cv2 could not read {path}")
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return {"edge_frac_40": float((mag > 40).mean())}


def normalise(bgr: np.ndarray) -> np.ndarray:
    """Resize longest side to 300 px and symmetrically pad to 300x300."""
    h, w = bgr.shape[:2]
    s = FRAME / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
    canvas = np.full((FRAME, FRAME, 3), pr.PAD_VALUE, dtype=np.uint8)
    top, left = (FRAME - nh) // 2, (FRAME - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def dhash_bits(bgr: np.ndarray) -> np.ndarray:
    """Compute 64-bit difference hash on 9x8 grid."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).flatten()


def scan_photos() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Scan external test photos and extract class metadata."""
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

        member, idx = parts[0], parts[2]
        source = "mirrored" if member == "hck" and idx in {"02", "2"} else "original"
        rows.append(
            {
                "rel_path": rel,
                "path": p,
                "class_name": folder,
                "class_id": CLASS_TO_ID[folder],
                "volunteer_name": member,
                "source": source,
            }
        )

    photos = pd.DataFrame(
        rows,
        columns=[
            "rel_path",
            "path",
            "class_name",
            "class_id",
            "volunteer_name",
            "source",
        ],
    )
    return photos, bad_folder, bad_name


def find_mirror_pairs(photos: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect horizontal reflection pairs using perceptual dHash comparison."""
    feats = {}
    for r in photos.itertuples(index=False):
        bgr = cv2.imread(str(r.path), cv2.IMREAD_COLOR)
        if bgr is not None:
            feats[r.rel_path] = (
                r.class_name,
                dhash_bits(bgr),
                dhash_bits(cv2.flip(bgr, 1)),
            )

    keys = sorted(feats)
    pairs = []
    for i, a in enumerate(keys):
        cls_a, h_a, f_a = feats[a]
        for b in keys[i + 1 :]:
            cls_b, h_b, f_b = feats[b]
            if cls_a != cls_b:
                continue
            d_flip = min(int(np.count_nonzero(f_a != h_b)), int(np.count_nonzero(h_a != f_b)))
            d_orig = int(np.count_nonzero(h_a != h_b))
            if d_flip <= MIRROR_MAX_DIST and d_orig >= MIRROR_MIN_ORIG_DIST:
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "class_name": cls_a,
                        "d_flip": d_flip,
                        "d_orig": d_orig,
                    }
                )
    return pairs


def main() -> None:
    print("ASL-HG: External Test Set Normalization and Verification")
    photos, bad_folder, bad_name = scan_photos()
    n_total = len(photos)

    if n_total == 0:
        print("[FAIL] No valid photos found under data/external_test/.")
        sys.exit(1)

    photos["md5"] = [hashlib.md5(p.read_bytes()).hexdigest() for p in photos["path"]]
    name_keys = pd.DataFrame(
        {"c": photos["class_name"], "s": [p.stem for p in photos["path"]]}
    )
    dup_names = int(name_keys.duplicated().sum())

    index = pd.read_csv(INDEX_CSV, dtype={"md5": str})
    mendeley_md5 = set(index["md5"])

    pr._init_worker()
    log_rows, decode_fail = [], []
    for r in photos.itertuples(index=False):
        bgr = cv2.imread(str(r.path), cv2.IMREAD_COLOR)
        if bgr is None:
            decode_fail.append(r.rel_path)
            continue

        out_norm = NORM_DIR / r.class_name / (r.path.stem + ".png")
        out_norm.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_norm), normalise(bgr))

        rel_norm = out_norm.relative_to(PROJECT_ROOT).as_posix()
        res = pr.process_one((rel_norm, r.class_name, 0, False, CROP_DIR))
        res["rel_path"] = r.rel_path
        res["edge_frac_40"] = edge_features(out_norm)["edge_frac_40"]
        log_rows.append(res)

    log = pd.DataFrame(log_rows)
    log.to_csv(LOG_OUT, index=False)

    written = (
        log[log["method"].isin(pr.WRITTEN_METHODS)].copy()
        if len(log) and "method" in log.columns
        else pd.DataFrame(columns=["rel_path", "label", "scale", "method", "edge_frac_40"])
    )

    train_log = pd.read_csv(pr.LOG_CSV)
    tw = train_log[train_log["method"].isin(pr.WRITTEN_METHODS)]
    side = (
        np.maximum(tw["margin_x1"] - tw["margin_x0"], tw["margin_y1"] - tw["margin_y0"])
        + 1
    )
    lo, hi = np.percentile(pr.CROP_SIZE / side, [1, 99])
    in_range = float(
        ((written["scale"] >= lo - 1e-3) & (written["scale"] <= hi + 1e-3)).mean()
    ) if len(written) else 0.0

    pairs = find_mirror_pairs(photos)
    per_orig = photos[photos["source"] == "original"]["class_name"].value_counts()
    per_mirr = photos[photos["source"] == "mirrored"]["class_name"].value_counts()
    n_orig = int((photos["source"] == "original").sum())
    n_mirr = int((photos["source"] == "mirrored").sum())
    dup = sum(m in mendeley_md5 for m in photos["md5"])

    results: dict[str, dict[str, Any]] = {}
    gate(results, "every_photo_decodes", len(decode_fail) == 0)
    gate(results, "folders_match_vocabulary", len(bad_folder) == 0)
    gate(results, "filenames_match_folder", len(bad_name) == 0)
    gate(results, "filenames_unique_per_class", dup_names == 0)
    gate(results, "total_count_exact", n_total == TOTAL, f"{n_total} photos (expected {TOTAL})")
    gate(results, "per_class_total_exact", (photos["class_name"].value_counts() == PER_CLASS_TOTAL).all())
    gate(results, "per_class_originals_exact", (per_orig == PER_CLASS_ORIGINAL).all())
    gate(results, "per_class_mirrored_exact", (per_mirr == PER_CLASS_MIRRORED).all())
    gate(results, "volunteer_names_are_text", not any(str(v).isdigit() for v in photos["volunteer_name"]))
    gate(results, "no_mendeley_duplicates", dup == 0)
    gate(results, "every_photo_cropped", n_total > 0 and len(written) == n_total)
    gate(results, "domain_scale_in_training_range", in_range >= DOMAIN_MIN_FRAC)

    flagged = log[log["edge_frac_40"] <= BLANK_REVIEW] if len(log) else log

    if not all(v["pass"] for v in results.values()):
        print("VERDICT: FAILED. Fix errors and re-run.")
        sys.exit(1)

    man = (
        pd.DataFrame(
            {
                "filepath": "data/external_test/" + photos["rel_path"],
                "class_name": photos["class_name"],
                "class_id": photos["class_id"],
                "volunteer_name": photos["volunteer_name"],
                "source": photos["source"],
                "is_mirrored": photos["source"] == "mirrored",
                "md5": photos["md5"],
            }
        )
        .sort_values("filepath")
        .reset_index(drop=True)
    )

    ok, diffs = write_and_verify(man, MANIFEST, list(man.columns))
    gate(results, "manifest_read_back", ok, "; ".join(diffs))
    if not ok:
        print("VERDICT: FAILED on manifest read-back.")
        sys.exit(1)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "photos_total": n_total,
        "photos_original": n_orig,
        "photos_mirrored": n_mirr,
        "mirror_pairs": pairs,
        "training_scale_range_p1_p99": [round(float(lo), 4), round(float(hi), 4)],
        "domain_min_frac_gate": DOMAIN_MIN_FRAC,
        "external_fraction_in_range": round(in_range, 4),
        "blank_review_flags": flagged["rel_path"].tolist() if len(flagged) else [],
        "gates": results,
        "passed": True,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("VERDICT: PASSED")


if __name__ == "__main__":
    main()
