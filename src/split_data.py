"""
src/split_data.py
Deterministic Stratified Splitting Engine for ASL Alphabet Dataset.
Rubric Mandates:
  - 70% Train (60,900 samples, 2,100/class)
  - 15% Validation (13,050 samples, 450/class)
  - 15% Test (13,050 samples, 450/class)
  - Random Seed = 42
Outputs:
  - data/splits/train.csv
  - data/splits/val.csv
  - data/splits/test.csv
  - data/splits/class_mapping.json
"""

import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Hyperparameters & Constants
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "asl_alphabet_train"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def generate_stratified_splits():
    print("=" * 70)
    print("TASK 2.4: DETERMINISTIC STRATIFIED DATASET PARTITIONING")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Target Ratios: Train={TRAIN_RATIO*100:.0f}%, Val={VAL_RATIO*100:.0f}%, Test={TEST_RATIO*100:.0f}%")
    print("=" * 70)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {DATA_DIR}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover classes and build sorted deterministic mapping
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    assert len(classes) == 29, f"Expected 29 classes, but found {len(classes)}"
    
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}
    id_to_class = {idx: cls_name for idx, cls_name in enumerate(classes)}

    print(f"\n[1/4] Discovered {len(classes)} classes.")
    
    # Save class mapping JSON for model inference & Gradio app
    mapping_payload = {
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "total_classes": len(classes)
    }
    mapping_path = SPLITS_DIR / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=4)
    print(f"  [OK] Saved class mapping to: {mapping_path.relative_to(PROJECT_ROOT)}")

    # 2. Gather all image paths with relative formatting
    print("\n[2/4] Indexing 87,000 image filepaths...")
    records = []
    for cls_name in classes:
        cls_folder = DATA_DIR / cls_name
        image_files = sorted(list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.jpeg")) + list(cls_folder.glob("*.png")))
        
        assert len(image_files) == 3000, f"Class '{cls_name}' has {len(image_files)} images, expected 3000."
        
        for img_p in image_files:
            # Store POSIX relative path (forward slashes) for Linux/Kaggle/Windows compatibility
            rel_path = img_p.relative_to(PROJECT_ROOT).as_posix()
            records.append({
                "filepath": rel_path,
                "label": cls_name,
                "class_id": class_to_id[cls_name]
            })

    df_all = pd.DataFrame(records)
    print(f"  [OK] Total indexed records: {len(df_all)}")

    # 3. Two-Stage Stratified Splitting
    print("\n[3/4] Performing two-stage stratified split (Seed=42)...")
    # Stage A: Split off Test set (15%) -> leaves 85% for (Train + Val)
    df_train_val, df_test = train_test_split(
        df_all,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df_all["class_id"],
        shuffle=True
    )

    # Stage B: Split remaining 85% into Train (70/85) and Val (15/85)
    # 0.15 / (0.70 + 0.15) = 0.15 / 0.85 approx 0.176470588
    val_relative_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative_ratio,
        random_state=RANDOM_SEED,
        stratify=df_train_val["class_id"],
        shuffle=True
    )

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 4. Serialize CSV splits
    print("\n[4/4] Serializing split CSVs to disk...")
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"  - Train Split CSV : {train_csv.relative_to(PROJECT_ROOT)} ({len(df_train):,} rows)")
    print(f"  - Val Split CSV   : {val_csv.relative_to(PROJECT_ROOT)} ({len(df_val):,} rows)")
    print(f"  - Test Split CSV  : {test_csv.relative_to(PROJECT_ROOT)} ({len(df_test):,} rows)")

    # 5. Strict Verification Check
    assert len(df_train) == 60900, f"Expected 60,900 train samples, got {len(df_train)}"
    assert len(df_val) == 13050, f"Expected 13,050 val samples, got {len(df_val)}"
    assert len(df_test) == 13050, f"Expected 13,050 test samples, got {len(df_test)}"
    assert set(df_train["filepath"]).isdisjoint(set(df_val["filepath"])), "DATA LEAKAGE: Overlap between Train and Val!"
    assert set(df_train["filepath"]).isdisjoint(set(df_test["filepath"])), "DATA LEAKAGE: Overlap between Train and Test!"
    assert set(df_val["filepath"]).isdisjoint(set(df_test["filepath"])), "DATA LEAKAGE: Overlap between Val and Test!"

    print("\n" + "=" * 70)
    print("VERIFICATION PASSED: EXACT RUBRIC PARTITION ACHIEVED")
    print("  - Train Samples : 60,900 (Exactly 2,100 / class)")
    print("  - Val Samples   : 13,050 (Exactly 450 / class)")
    print("  - Test Samples  : 13,050 (Exactly 450 / class)")
    print("  - Data Leakage  : 0 samples overlap across all sets")
    print("=" * 70)


if __name__ == "__main__":
    generate_stratified_splits()