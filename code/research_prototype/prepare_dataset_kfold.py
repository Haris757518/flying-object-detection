"""Dataset preparation script for Flying Object Detection.
Performs:
1. Verification of train/test images and labels.
2. Multilabel Stratified 10-Fold Cross-Validation partitioning on the 70% training pool.
3. Generation of fold text files (fold_{k}_train.txt, fold_{k}_val.txt) and data_fold_{k}.yaml files.
4. Generates data_70_30.yaml for baseline training and evaluation against the 30% holdout test set.
"""

import os
import json
import glob
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.abspath("DS/comb DSF")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(BASE_DIR, "train", "labels")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test", "images")
TEST_LBL_DIR = os.path.join(BASE_DIR, "test", "labels")

KFOLD_DIR = os.path.join(BASE_DIR, "kfold_splits")
os.makedirs(KFOLD_DIR, exist_ok=True)

CLASS_NAMES = ["airplane", "bird", "helicopter", "kite", "parachute", "UAV"]
NUM_CLASSES = len(CLASS_NAMES)
K_FOLDS = 10
RANDOM_SEED = 42

def inspect_and_build_dataset():
    print("=" * 60)
    print("STEP 1: AUDITING DATASET AND EXTRACTING MULTI-LABEL PROFILES")
    print("=" * 60)
    
    train_images = sorted(os.listdir(TRAIN_IMG_DIR))
    print(f"Found {len(train_images)} training images on disk.")
    
    data_records = []
    
    for img_name in train_images:
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        stem, _ = os.path.splitext(img_name)
        lbl_path = os.path.join(TRAIN_LBL_DIR, stem + ".txt")
        
        class_counts = [0] * NUM_CLASSES
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < NUM_CLASSES:
                            class_counts[cls_id] += 1
        
        # Primary label for stratified partitioning:
        # Prioritize rare classes: UAV (5) > Helicopter (2) > Bird (1) > Airplane (0)
        priority_order = [5, 2, 4, 3, 1, 0]
        primary_class = -1
        for p in priority_order:
            if class_counts[p] > 0:
                primary_class = p
                break
        if primary_class == -1:
            primary_class = NUM_CLASSES  # Empty/Background
            
        data_records.append({
            "image_name": img_name,
            "image_path": img_path.replace("\\", "/"),
            "label_path": lbl_path.replace("\\", "/"),
            "primary_class": primary_class,
            "boxes_total": sum(class_counts),
            **{f"class_{i}": class_counts[i] for i in range(NUM_CLASSES)}
        })
        
    df = pd.DataFrame(data_records)
    print(f"Primary class distribution across {len(df)} images:")
    for c_id, count in df["primary_class"].value_counts().sort_index().items():
        name = CLASS_NAMES[c_id] if c_id < NUM_CLASSES else "Background"
        print(f"  Class {c_id} ({name}): {count} images")
        
    print("\n" + "=" * 60)
    print("STEP 2: COMPUTING 10-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["primary_class"])):
        df.loc[val_idx, "fold"] = fold_idx
        
    manifest_csv = os.path.join(KFOLD_DIR, "kfold_manifest.csv")
    df.to_csv(manifest_csv, index=False)
    print(f"Saved K-Fold split manifest to: {manifest_csv}")
    
    # Generate list files for each fold
    test_images = [os.path.join(TEST_IMG_DIR, f).replace("\\", "/") for f in sorted(os.listdir(TEST_IMG_DIR))]
    test_list_file = os.path.join(KFOLD_DIR, "test_holdout_30.txt")
    with open(test_list_file, "w") as f:
        for p in test_images:
            f.write(p + "\n")
    print(f"Saved 30% Holdout Test list ({len(test_images)} images) to: {test_list_file}")

    print("\n" + "=" * 60)
    print("STEP 3: GENERATING YOLO DATA CONFIGS FOR ALL 10 FOLDS")
    print("=" * 60)

    norm_base_dir = BASE_DIR.replace('\\', '/')
    norm_test_file = test_list_file.replace('\\', '/')

    for fold in range(K_FOLDS):
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        
        train_txt = os.path.join(KFOLD_DIR, f"fold_{fold}_train.txt")
        val_txt = os.path.join(KFOLD_DIR, f"fold_{fold}_val.txt")
        
        with open(train_txt, "w") as f:
            for p in train_df["image_path"]:
                f.write(p + "\n")
                
        with open(val_txt, "w") as f:
            for p in val_df["image_path"]:
                f.write(p + "\n")

        norm_train_txt = train_txt.replace('\\', '/')
        norm_val_txt = val_txt.replace('\\', '/')
                
        yaml_content = f"""# 10-Fold Cross-Validation - Fold {fold}
path: "{norm_base_dir}"
train: "{norm_train_txt}"
val: "{norm_val_txt}"
test: "{norm_test_file}"

nc: {NUM_CLASSES}
names:
  0: airplane
  1: bird
  2: helicopter
  3: kite
  4: parachute
  5: UAV
"""
        yaml_path = os.path.join(KFOLD_DIR, f"data_fold_{fold}.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
            
        print(f"Fold {fold}: Train={len(train_df)} images (90%), Val={len(val_df)} images (10%) -> {os.path.basename(yaml_path)}")

    # Also generate main 70:30 baseline config (using Fold 0 val as standard val split)
    baseline_train = os.path.join(KFOLD_DIR, "baseline_train_70.txt")
    with open(baseline_train, "w") as f:
        for p in df["image_path"]:
            f.write(p + "\n")

    norm_f0_train = os.path.join(KFOLD_DIR, 'fold_0_train.txt').replace('\\', '/')
    norm_f0_val = os.path.join(KFOLD_DIR, 'fold_0_val.txt').replace('\\', '/')

    baseline_yaml = f"""# Flying Object Detection 70:30 Master Config
path: "{norm_base_dir}"
train: "{norm_f0_train}"
val: "{norm_f0_val}"
test: "{norm_test_file}"

nc: {NUM_CLASSES}
names:
  0: airplane
  1: bird
  2: helicopter
  3: kite
  4: parachute
  5: UAV
"""
    master_yaml_path = os.path.join(BASE_DIR, "data_70_30.yaml")
    with open(master_yaml_path, "w") as f:
        f.write(baseline_yaml)
    print(f"\nMaster 70:30 config written to: {master_yaml_path}")
    print("\nDataset partitioning and 10-Fold configuration completed successfully!")

if __name__ == "__main__":
    inspect_and_build_dataset()
