import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    BLANK_FRAMES_CSV,
    CROPS_DIR,
    DOCS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from src.utils.inspection import generate_crop_inspection_grid

from .roi_detector import (
    IMG_SIZE,
    BoxCoords,
    describe_box,
    detect,
    make_hands,
)

SEED: int = 42
MARGIN_FRAC: float = 0.20
CROP_SIZE: int = 224
PAD_VALUE: tuple[int, int, int] = (128, 128, 128)
NUM_WORKERS: int = 8
PNG_PARAMS: list[int] = [cv2.IMWRITE_PNG_COMPRESSION, 3]
FULL_BOX: BoxCoords = (0, 0, IMG_SIZE - 1, IMG_SIZE - 1)
FULL_FRAME_CLASSES: set[str] = {"0"}
WRITTEN_METHODS: list[str] = [
    "mediapipe",
    "skin",
    "class_rule_full",
    "undetected_full",
]

INDEX_CSV: Path = PROCESSED_DIR / "asl_hg_image_index.csv"
LOG_CSV: Path = PROCESSED_DIR / "roi_crop_log.csv"
REPORT: Path = DOCS_DIR / "metrics" / "crop_log.json"
INSPECT_FIG: Path = DOCS_DIR / "figures" / "eda" / "roi_crop_inspection.png"


def add_margin(box: BoxCoords, frac: float, frame_size: int) -> BoxCoords:
    """Expand a bounding box symmetrically by a fraction of its longest side."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) + 1
    margin = round(side * frac)
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(frame_size - 1, x1 + margin),
        min(frame_size - 1, y1 + margin),
    )


def letterbox(
    crop_bgr: np.ndarray,
    out_size: int,
    pad_value: tuple[int, int, int],
) -> np.ndarray:
    """Resize image preserving aspect ratio and center on a padded square canvas."""
    h, w = crop_bgr.shape[:2]
    scale = out_size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_size, out_size, 3), pad_value, dtype=np.uint8)
    top = (out_size - new_h) // 2
    left = (out_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


_HANDS = None


def _init_worker() -> None:
    global _HANDS
    _HANDS = make_hands()


TaskTuple = tuple[str, str, int, bool, Path]


def process_one(task: TaskTuple) -> dict[str, Any]:
    """Process a single image record: detect hand, crop, letterbox, and save to disk."""
    filepath, label, volunteer, is_blank, out_root = task
    base_info = {
        "filepath": filepath,
        "label": label,
        "volunteer": volunteer,
    }
    if is_blank:
        return {**base_info, "method": "blank"}

    bgr = cv2.imread(str(PROJECT_ROOT / filepath))
    if bgr is None:
        return {**base_info, "method": "error", "error": "cv2.imread failed"}

    box: BoxCoords | None = None
    if label in FULL_FRAME_CLASSES:
        method = "class_rule_full"
    else:
        box, method = detect(bgr, _HANDS)
        if box is None:
            method = "undetected_full"

    if box is None:
        box = FULL_BOX
        side = area_frac = touches = None
    else:
        side, area_frac, touches = describe_box(box)

    x0, y0, x1, y1 = add_margin(box, MARGIN_FRAC, IMG_SIZE)
    margin_side = max(x1 - x0, y1 - y0) + 1
    out_img = letterbox(bgr[y0 : y1 + 1, x0 : x1 + 1], CROP_SIZE, PAD_VALUE)

    out_path = out_root / label / (Path(filepath).stem + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), out_img, PNG_PARAMS):
        return {**base_info, "method": "error", "error": "cv2.imwrite failed"}

    return {
        **base_info,
        "method": method,
        "box_side": side,
        "box_area_frac": None if area_frac is None else round(area_frac, 4),
        "box_touches_border": touches,
        "margin_x0": x0,
        "margin_y0": y0,
        "margin_x1": x1,
        "margin_y1": y1,
        "margin_side": int(margin_side),
        "scale": round(CROP_SIZE / margin_side, 4),
    }


def summarise(res: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate detector statistics over processed crops."""
    n = len(res)
    counts = {k: int(v) for k, v in res["method"].value_counts().items()}
    written = res[res["method"].isin(WRITTEN_METHODS)].copy()
    written["scale"] = written["scale"].astype(float)

    det = written[written["method"].isin(["mediapipe", "skin"])]
    touch_by_method = {
        m: round(float(g["box_touches_border"].astype(float).mean()), 4) for m, g in det.groupby("method")
    }
    scale_by_method = {m: round(float(g["scale"].median()), 4) for m, g in written.groupby("method")}
    errors = (
        res.loc[res["method"] == "error", ["filepath", "error"]].to_dict("records") if "error" in res.columns else []
    )
    return {
        "n": n,
        "method_counts": counts,
        "method_rates": {k: round(v / n, 4) for k, v in counts.items()},
        "written": len(written),
        "box_touches_border_rate_by_method": touch_by_method,
        "median_scale_by_method": scale_by_method,
        "errors": errors,
    }


def main() -> None:
    print("ASL-HG: Dataset ROI Preprocessing and Letterboxing")

    if not BLANK_FRAMES_CSV.exists():
        raise FileNotFoundError(f"{BLANK_FRAMES_CSV} not found.")
    stale = [p for p in CROPS_DIR.rglob("*") if p.is_file()] if CROPS_DIR.exists() else []
    if stale:
        raise SystemExit(
            f"{CROPS_DIR} already contains {len(stale)} existing files. "
            "Remove directory before re-running preprocessing."
        )

    df = pd.read_csv(INDEX_CSV, dtype={"filepath": str, "label": str})
    blank_set = set(pd.read_csv(BLANK_FRAMES_CSV, dtype={"filepath": str})["filepath"])
    print(f"Images: {len(df):,} | Blank excluded: {len(blank_set):,}")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            r.filepath,
            r.label,
            int(r.volunteer),
            r.filepath in blank_set,
            CROPS_DIR,
        )
        for r in df.itertuples(index=False)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_init_worker) as pool:
        for res in tqdm(
            pool.map(process_one, tasks, chunksize=50),
            total=len(tasks),
            desc="Cropping",
            unit="img",
        ):
            results.append(res)

    res_df = pd.DataFrame(results)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(LOG_CSV, index=False)

    summary_stats = summarise(res_df)
    print("Method counts:", summary_stats["method_counts"])
    print("Method rates :", summary_stats["method_rates"])

    on_disk = sum(1 for _ in CROPS_DIR.rglob("*.png"))
    print(f"PNG files on disk: {on_disk:,} (logged: {summary_stats['written']:,})")

    generate_crop_inspection_grid(res_df, INSPECT_FIG)

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "margin_frac": MARGIN_FRAC,
        "crop_size": CROP_SIZE,
        "pad_value": list(PAD_VALUE),
        "output_format": "png",
        "full_frame_classes": sorted(FULL_FRAME_CLASSES),
        "blank_frames_excluded": len(blank_set),
        "files_on_disk": on_disk,
        "summary": summary_stats,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
