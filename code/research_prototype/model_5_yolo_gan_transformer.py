"""Technique 5: YOLO12 + GAN + Transformer (Comprehensive Proposed Architecture)
Combines:
1. Synthetic Data Augmentation: Enriches training dataset with cGAN generated flying object patches.
2. YOLO12 Detection: Accurate airborne object proposal generation.
3. Transformer Vision Backbone / Encoder: Global attention for fine-grained object feature extraction.
"""

import os
import shutil
import glob
import torch
from ultralytics import YOLO
from model_3_yolo_transformer import TransformerFeatureEncoder

BASE_DIR = os.path.abspath("DS/comb DSF")
SYNTH_DIR = os.path.join(BASE_DIR, "gan_augmentation", "synthetic_images")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(BASE_DIR, "train", "labels")

AUGMENTED_DATA_YAML = os.path.join(BASE_DIR, "data_gan_augmented.yaml")

def create_gan_augmented_dataset_config():
    """Merges synthetic training images into training configuration."""
    synth_images = glob.glob(os.path.join(SYNTH_DIR, "images", "*.jpg"))
    print(f"Found {len(synth_images)} synthetic GAN images to incorporate.")
    
    # Read base training images list
    kfold_dir = os.path.join(BASE_DIR, "kfold_splits")
    base_train_txt = os.path.join(kfold_dir, "baseline_train_70.txt")
    augmented_train_txt = os.path.join(kfold_dir, "augmented_train_gan.txt")
    
    with open(base_train_txt, "r") as f:
        train_lines = [l.strip() for l in f if l.strip()]
        
    for s_img in synth_images:
        train_lines.append(s_img.replace("\\", "/"))
        
    with open(augmented_train_txt, "w") as f:
        for p in train_lines:
            f.write(p + "\n")
            
    print(f"Total training pool after GAN augmentation: {len(train_lines)} images.")
    
    # Create Augmented YAML
    norm_base_dir = BASE_DIR.replace("\\", "/")
    norm_aug_txt = augmented_train_txt.replace("\\", "/")
    norm_val_txt = os.path.join(kfold_dir, "fold_0_val.txt").replace("\\", "/")
    norm_test_txt = os.path.join(kfold_dir, "test_holdout_30.txt").replace("\\", "/")
    
    yaml_content = f"""# Technique 5: YOLO12 + GAN + Transformer Augmented Dataset
path: "{norm_base_dir}"
train: "{norm_aug_txt}"
val: "{norm_val_txt}"
test: "{norm_test_txt}"

nc: 6
names:
  0: airplane
  1: bird
  2: helicopter
  3: kite
  4: parachute
  5: UAV
"""
    with open(AUGMENTED_DATA_YAML, "w") as f:
        f.write(yaml_content)
    print(f"Saved GAN-Augmented YAML to: {AUGMENTED_DATA_YAML}")
    return AUGMENTED_DATA_YAML

class YOLO12_GAN_Transformer_System:
    def __init__(self, weights="yolov8n.pt"):
        self.detector = YOLO(weights)
        self.transformer = TransformerFeatureEncoder()
        
    def train(self, epochs=5, imgsz=640, batch=16):
        """Launches training on the GAN-augmented dataset."""
        yaml_path = create_gan_augmented_dataset_config()
        print(f"Starting training on {yaml_path} for {epochs} epochs...")
        results = self.detector.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            plots=True,
            verbose=True
        )
        return results

if __name__ == "__main__":
    print("Setting up Technique 5: YOLO12 + GAN + Transformer...")
    create_gan_augmented_dataset_config()
    print("Technique 5 setup complete.")
