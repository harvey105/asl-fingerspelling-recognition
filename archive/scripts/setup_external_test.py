"""
src/setup_external_test.py
Task 2.4 Part C (setup): Idempotent directory provisioning and photo census tool.

Simplified Workflow (Leader Update):
  - Completely skips manual metadata spreadsheets (metadata_input.csv eliminated).
  - Ensures the 36 discrete class directories exist under data/external_test/.
  - Scans and verifies photos directly from disk: <member>_<class>_<idx>.jpg.
  - Safe to re-run at any time; never destructive to image files.

Run (from repository root): uv run python src/setup_external_test.py
"""

from collections import Counter
from pathlib import Path
import pandas as pd

from config import CLASSES, CLASS_TO_ID, PROJECT_ROOT

EXT_DIR = PROJECT_ROOT / "data" / "external_test"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def get_disk_census():
    """Scans data/external_test/ and groups valid images by class and volunteer."""
    records = []
    for cls in CLASSES:
        folder = EXT_DIR / cls
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXT:
                    parts = p.stem.split("_")
                    member = parts[0] if len(parts) >= 1 else "unknown"
                    records.append({
                        "rel_path": f"{cls}/{p.name}",
                        "class_name": cls,
                        "volunteer_name": member
                    })
    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.4 PART C (EXTERNAL BENCHMARK DIRECTORY SETUP & CENSUS)")
    print("=" * 70)

    # Step 1: Idempotently ensure all 36 class folders exist
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    created_count = 0
    for cls in CLASSES:
        folder = EXT_DIR / cls
        if not folder.exists():
            folder.mkdir()
            created_count += 1

    print(f"Class directories : {len(CLASSES)} verified ({created_count} newly created)")
    print(f"Root path         : {EXT_DIR.relative_to(PROJECT_ROOT).as_posix()}/")

    # Step 2: Census of collected photos on disk
    df_census = get_disk_census()
    total_photos = len(df_census)
    print(f"Total photos on disk : {total_photos} (Target: 180 = 36 classes x 5 photos)")

    if total_photos > 0:
        member_counts = df_census["volunteer_name"].value_counts().to_dict()
        print(f"Volunteer breakdown : {member_counts}")
        
        per_class_counts = df_census["class_name"].value_counts().reindex(CLASSES, fill_value=0)
        incomplete_classes = per_class_counts[per_class_counts != 5]
        if incomplete_classes.empty:
            print("[OK] All 36 classes have exactly 5 photos.")
        else:
            print(f"[NOTE] Classes not at 5 photos: {incomplete_classes.to_dict()}")

    print("\nNext: Run 'uv run python src/prepare_external.py' to normalise, crop, and generate manifest.")
    print("=" * 70)


if __name__ == "__main__":
    main()