"""
src/setup_external_test.py
Initializes and audits the Out-of-Distribution Real-World Benchmark.
Task 2.4 Mandate:
  - Curate 60 to 100 original ASL photos taken by team members.
  - Initializes the 29 class folders inside `data/external_test/` - where to drop their webcam/phone photos.
  - Generates an automated manifest `data/external_test/external_manifest.csv`
    and validates image formats, sizes, and label integrity.
"""

import json
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_test"
MAPPING_FILE = PROJECT_ROOT / "data" / "splits" / "class_mapping.json"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def init_and_audit_external_set():
    print("=" * 70)
    print("TASK 2.4: OUT-OF-DISTRIBUTION EXTERNAL BENCHMARK MANAGER")
    print(f"Target Directory: {EXTERNAL_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 70)

    # Load deterministic class mapping
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Class mapping not found at {MAPPING_FILE}. Run src/split_data.py first.")

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_to_id = mapping["class_to_id"]
    classes = list(class_to_id.keys())

    # 1. Create subfolders for all 29 classes
    print("\n[1/3] Ensuring all 29 class folders exist under data/external_test/...")
    for cls_name in classes:
        cls_folder = EXTERNAL_DIR / cls_name
        cls_folder.mkdir(parents=True, exist_ok=True)
    print("  [OK] All 29 class subdirectories initialized.")

    # 2. Scan and audit collected images
    print("\n[2/3] Scanning for user-contributed external images...")
    valid_records = []
    corrupted_images = []

    # Robust discovery using suffix checking (avoids glob pattern errors)
    found_files = [p for p in EXTERNAL_DIR.glob("*/*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

    unknown_folder = []
    for img_p in found_files:
        cls_name = img_p.parent.name
        if cls_name not in class_to_id:
            unknown_folder.append(str(img_p))
            continue

        try:
            with Image.open(img_p) as img:
                img.load()
            
            # Record relative path with forward slashes
            rel_path = img_p.relative_to(PROJECT_ROOT).as_posix()
            valid_records.append({
                "filepath": rel_path,
                "label": cls_name,
                "class_id": class_to_id[cls_name]
            })
        except Exception as e:
            corrupted_images.append((str(img_p), str(e)))

    # 3. Report Status & Save Manifest
    print(f"\n[3/3] External Benchmark Status:")
    print(f"  - Valid images collected: {len(valid_records)}")
    print(f"  - Target quota          : 60 to 100 images (2 to 4 photos per class)")

    manifest_csv = EXTERNAL_DIR / "external_manifest.csv"
    if valid_records:
        df_ext = pd.DataFrame(valid_records)
        df_ext.to_csv(manifest_csv, index=False)
        print(f"  [OK] Saved manifest to: {manifest_csv.relative_to(PROJECT_ROOT)} ({len(df_ext)} rows)")
    else:
        print("  [INFO] Folders are currently empty. Team members can now add their photos.")

    if not 60 <= len(valid_records) <= 100: 
        print("  [WARN] Count is outside the 60 to 100 quota.")

    if corrupted_images:
        print(f"  [WARN] Found {len(corrupted_images)} corrupted image(s):")
        for bad_p, err in corrupted_images[:5]:
            print(f"    - {bad_p}: {err}")

    print("\n" + "=" * 70)
    print("TEAM CAPTURE PROTOCOL INSTRUCTIONS:")
    print("1. Take 2 to 4 photos for each ASL gesture using your phone/laptop webcam.")
    print("2. Vary conditions: different team members, room lighting, cluttered backgrounds.")
    print("3. Save photos directly into data/external_test/<CLASS_NAME>/")
    print("   (e.g., data/external_test/A/hck_webcam_01.jpg)")
    print("4. Re-run 'python src/setup_external_test.py' to generate external_manifest.csv.")
    print("=" * 70)


if __name__ == "__main__":
    init_and_audit_external_set()