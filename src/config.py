"""
src/config.py
Single source of truth for dataset constants and path discovery.
Other scripts import from here instead of hard-coding values.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"   # our outputs only
DOCS_DIR = PROJECT_ROOT / "docs"
CROPS_DIR = PROJECT_ROOT / "data" / "processed_crops"   #  output (PNG)
BLANK_FRAMES_CSV = PROCESSED_DIR / "blank_frames.csv"   #  output

# Class set: digits first, then letters. Digit 0 is the two-handed ASL zero sign (paper, Table 1).
CLASSES = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
NUM_CLASSES = len(CLASSES)                      # expected 36
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Expectations from the paper, dataset page, and roadmap.  checks these; it does not assume them.
NUM_VOLUNTEERS = 10
IMAGES_PER_VOLUNTEER_PER_CLASS = 100
IMAGES_PER_CLASS = 1000
TOTAL_IMAGES = IMAGES_PER_CLASS * NUM_CLASSES   # expected 36,000
EXPECTED_SIZE = (300, 300)
CROP_SIZE_PX = 224   #  letterbox output; must match preprocess_roi.CROP_SIZE
EXPECTED_MODE = "RGB"
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

# Subject-independent split (roadmap ). Used in  onward.
TRAIN_VOLUNTEERS = [1, 2, 3, 4, 5, 6, 7]
VAL_VOLUNTEERS = [8]
TEST_VOLUNTEERS = [9, 10]

# Filenames are expected as P<volunteer>_<class>_<index>.<ext>, e.g. P3_K_0042.jpg (roadmap).
#  verifies this pattern against the real files.
FILENAME_RE = re.compile(r"^P(\d+)_([0-9A-Za-z]+)_(\d+)\.[A-Za-z0-9]+$")


def parse_filename(name):
    """Return (volunteer_id, class_token, index) or None if the name does not match."""
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


def find_dataset_root(raw_dir=RAW_DIR, max_depth=3):
    """
    Return the folder that directly contains all class folders.
    Works whether the download was extracted as data/raw/<class folders>
    or data/raw/<some folder>/<class folders>. Raises if zero or several match.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data folder not found: {raw_dir}")

    wanted = set(CLASSES)
    hits = []
    for dirpath, dirnames, _ in os.walk(raw_dir):
        here = Path(dirpath)
        if wanted.issubset(dirnames):
            hits.append(here)
        depth = len(here.relative_to(raw_dir).parts)
        if depth >= max_depth:
            dirnames[:] = []                          # stop descending

    if not hits:
        raise ValueError(
            f"No folder under {raw_dir} (depth <= {max_depth}) contains all "
            f"{len(CLASSES)} class folders. Check the extraction."
        )
    if len(hits) > 1:
        listing = [h.relative_to(raw_dir).as_posix() or "." for h in hits]
        raise ValueError(f"Several folders contain all class folders: {listing}. Keep only one.")
    return hits[0]