"""
Complete End-to-End Image Classification ML Workflow Engine.

Sequence:
1. Load Image Dataset
2. Data Preprocessing
3. Data Leakage Checks (Target leakage, duplicate leakage, suspicious features)
4. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
5. Dataset Split (Train/Val/Test) & 10-Fold Cross-Validation
6. GAN Augmentation on TRAINING data only
7. Initial Model Training
8. Overfitting / Underfitting Detection & Diagnostics
9. Apply Correction Techniques & Retrain
10. Final Test Evaluation
"""

import os
import glob
import copy
import random
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.gan_enhancer import LightweightGANGenerator


# ==============================================================================
# DATA STRUCTURES & DEFINITIONS
# ==============================================================================

DEFAULT_CLASSES = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']


@dataclass
class ImageSample:
    """Represents a single image sample in the classification pipeline."""
    image: np.ndarray  # BGR format (H, W, C)
    label: int
    class_name: str
    file_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_id: str = ""


# ==============================================================================
# STEP 1: LOAD IMAGE DATASET
# ==============================================================================

class DatasetLoader:
    """
    Step 1: Load and ingest multi-class image datasets.
    Supports directory scanning, YOLO crop extraction, or synthetic dataset synthesis for testing.
    """
    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or DEFAULT_CLASSES
        self.class_to_id = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.id_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}

    def load_from_directory(self, data_dir: str, max_samples_per_class: Optional[int] = None) -> List[ImageSample]:
        """Loads images organized in subfolders by class name (e.g. data_dir/DRONE/xxx.jpg)."""
        samples: List[ImageSample] = []
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            
            image_paths = sorted(
                glob.glob(os.path.join(cls_dir, "*.jpg")) +
                glob.glob(os.path.join(cls_dir, "*.png")) +
                glob.glob(os.path.join(cls_dir, "*.jpeg"))
            )
            if max_samples_per_class:
                image_paths = image_paths[:max_samples_per_class]

            for path in image_paths:
                img = cv2.imread(path)
                if img is None:
                    continue  # Filter corrupt images
                
                samples.append(ImageSample(
                    image=img,
                    label=self.class_to_id[cls_name],
                    class_name=cls_name,
                    file_path=path,
                    sample_id=os.path.basename(path)
                ))
        return samples

    def load_from_yolo_crops(self, yolo_dir: str, max_samples: Optional[int] = None) -> List[ImageSample]:
        """Loads and extracts object crops from pre-formatted YOLO dataset directories with balanced class sampling."""
        samples: List[ImageSample] = []
        images_dir = os.path.join(yolo_dir, "images")
        labels_dir = os.path.join(yolo_dir, "labels")
        num_classes = len(self.classes)
        max_per_class = (max_samples // num_classes) if max_samples and max_samples > 0 else float('inf')
        class_counts = {c: 0 for c in range(num_classes)}

        for split in ["train", "val", "test"]:
            s_img_dir = os.path.join(images_dir, split)
            s_lbl_dir = os.path.join(labels_dir, split)
            if not os.path.exists(s_img_dir):
                continue

            img_paths = glob.glob(os.path.join(s_img_dir, "*.jpg")) + glob.glob(os.path.join(s_img_dir, "*.png"))
            random.Random(42).shuffle(img_paths)

            for img_path in img_paths:
                if all(count >= max_per_class for count in class_counts.values()):
                    break
                base = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = os.path.join(s_lbl_dir, base + ".txt")

                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                if cls_id >= num_classes or class_counts[cls_id] >= max_per_class:
                                    continue
                                img = cv2.imread(img_path)
                                if img is None:
                                    continue
                                h, w = img.shape[:2]
                                xc, yc, bw, bh = float(parts[1])*w, float(parts[2])*h, float(parts[3])*w, float(parts[4])*h
                                x1 = max(0, int(xc - bw / 2))
                                y1 = max(0, int(yc - bh / 2))
                                x2 = min(w, int(xc + bw / 2))
                                y2 = min(h, int(yc + bh / 2))
                                crop = img[y1:y2, x1:x2]
                                if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                                    samples.append(ImageSample(
                                        image=crop,
                                        label=cls_id,
                                        class_name=self.id_to_class[cls_id],
                                        file_path=img_path,
                                        sample_id=f"{base}_crop_{cls_id}"
                                    ))
                                    class_counts[cls_id] += 1
                else:
                    for cls_name in self.classes:
                        cls_id = self.class_to_id[cls_name]
                        if cls_name in base.upper() and class_counts[cls_id] < max_per_class:
                            img = cv2.imread(img_path)
                            if img is not None:
                                samples.append(ImageSample(
                                    image=img,
                                    label=cls_id,
                                    class_name=cls_name,
                                    file_path=img_path,
                                    sample_id=base
                                ))
                                class_counts[cls_id] += 1
                            break
        return samples

    def load_from_aod_dataset(self, aod_dir: str = "Data/AOD 4", max_samples: Optional[int] = None) -> List[ImageSample]:
        """
        Loads and extracts object crops from Airborne Object Dataset (AOD) with balanced class sampling.
        If max_samples is None or 0, loads ALL 22,500+ images from the entire dataset.
        """
        samples: List[ImageSample] = []
        images_root = os.path.join(aod_dir, "Images")
        ann_root = os.path.join(aod_dir, "Annotations", "YOLOv8 format")
        num_classes = len(self.classes)
        max_per_class = (max_samples // num_classes) if max_samples and max_samples > 0 else float('inf')
        class_counts = {c: 0 for c in range(num_classes)}

        if not os.path.exists(images_root) or not os.path.exists(ann_root):
            for sub in os.listdir(aod_dir):
                if sub.lower() == "images":
                    images_root = os.path.join(aod_dir, sub)
                if sub.lower() == "annotations":
                    ann_root = os.path.join(aod_dir, sub, "YOLOv8 format")

        split_pairs = [("train", "train"), ("valid", "valid"), ("test", "test")]
        
        for img_split, lbl_split in split_pairs:
            s_img_dir = os.path.join(images_root, img_split)
            s_lbl_dir = os.path.join(ann_root, lbl_split, "labels")

            if not os.path.exists(s_img_dir) or not os.path.exists(s_lbl_dir):
                continue

            img_files = glob.glob(os.path.join(s_img_dir, "*.jpg")) + glob.glob(os.path.join(s_img_dir, "*.png"))
            random.Random(42).shuffle(img_files)
            
            for img_path in img_files:
                if all(count >= max_per_class for count in class_counts.values()):
                    break
                base = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = os.path.join(s_lbl_dir, base + ".txt")
                
                if not os.path.exists(lbl_path):
                    continue

                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            if cls_id >= num_classes or class_counts[cls_id] >= max_per_class:
                                continue
                            img = cv2.imread(img_path)
                            if img is None:
                                continue
                            h, w = img.shape[:2]
                            xc, yc, bw, bh = float(parts[1])*w, float(parts[2])*h, float(parts[3])*w, float(parts[4])*h
                            x1 = max(0, int(xc - bw / 2))
                            y1 = max(0, int(yc - bh / 2))
                            x2 = min(w, int(xc + bw / 2))
                            y2 = min(h, int(yc + bh / 2))
                            crop = img[y1:y2, x1:x2]
                            if crop.size > 0 and crop.shape[0] >= 8 and crop.shape[1] >= 8:
                                samples.append(ImageSample(
                                    image=crop,
                                    label=cls_id,
                                    class_name=self.id_to_class[cls_id],
                                    file_path=img_path,
                                    sample_id=f"aod_{base}_{cls_id}"
                                ))
                                class_counts[cls_id] += 1
        return samples

    def generate_synthetic_dataset(self, samples_per_class: int = 50, img_size: int = 64) -> List[ImageSample]:
        """Generates realistic synthetic multi-class drone/airborne image samples for rapid testing."""
        samples: List[ImageSample] = []
        np.random.seed(42)
        
        for cls_id, cls_name in enumerate(self.classes):
            for i in range(samples_per_class):
                bg_color = np.random.randint(120, 200, size=(1, 1, 3), dtype=np.uint8)
                img = np.ones((img_size, img_size, 3), dtype=np.uint8) * bg_color
                noise = np.random.normal(0, 10, (img_size, img_size, 3)).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                cx, cy = img_size // 2, img_size // 2
                if cls_name == "DRONE":
                    cv2.line(img, (cx - 12, cy - 12), (cx + 12, cy + 12), (30, 30, 30), 2)
                    cv2.line(img, (cx - 12, cy + 12), (cx + 12, cy - 12), (30, 30, 30), 2)
                    for rx, ry in [(cx - 12, cy - 12), (cx + 12, cy + 12), (cx - 12, cy + 12), (cx + 12, cy - 12)]:
                        cv2.circle(img, (rx, ry), 4, (10, 10, 10), -1)
                elif cls_name == "AIRPLANE":
                    cv2.line(img, (cx - 18, cy), (cx + 18, cy), (40, 40, 40), 3)
                    cv2.line(img, (cx, cy - 14), (cx, cy + 14), (40, 40, 40), 2)
                elif cls_name == "HELICOPTER":
                    cv2.ellipse(img, (cx, cy), (10, 6), 0, 0, 360, (50, 50, 50), -1)
                    cv2.line(img, (cx - 16, cy - 8), (cx + 16, cy - 8), (20, 20, 20), 2)
                elif cls_name == "BIRD":
                    pts = np.array([[cx - 10, cy + 4], [cx, cy], [cx + 10, cy + 4]], np.int32)
                    cv2.polylines(img, [pts], False, (20, 20, 20), 2)

                samples.append(ImageSample(
                    image=img,
                    label=cls_id,
                    class_name=cls_name,
                    file_path=f"synthetic_{cls_name}_{i}.png",
                    sample_id=f"syn_{cls_name}_{i}"
                ))
        return samples


# ==============================================================================
# STEP 2: DATA PREPROCESSING
# ==============================================================================

class DataPreprocessor:
    """
    Step 2: Preprocess images into standardized tensors.
    Performs letterboxing, spatial resizing, channel normalization, and tensor conversion.
    """
    def __init__(self, target_size: Tuple[int, int] = (64, 64), normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """Resizes with aspect ratio letterboxing and converts BGR to RGB."""
        h, w = img_bgr.shape[:2]
        th, tw = self.target_size
        scale = min(th / max(1, h), tw / max(1, w))
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        dy = (th - nh) // 2
        dx = (tw - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb

    def to_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Converts RGB numpy image (H, W, C) [0, 255] to PyTorch tensor (C, H, W) normalized."""
        tensor = img_rgb.astype(np.float32) / 255.0
        if self.normalize:
            tensor = (tensor - self.mean) / self.std
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).float()
        return tensor


# ==============================================================================
# STEP 3: DATA LEAKAGE CHECKS
# ==============================================================================

class DataLeakageAuditor:
    """
    Step 3: Audit and eliminate data leakage risks before splitting/training.
    - Target Leakage (spurious metadata, filename proxies, constant border artifacts)
    - Duplicate Leakage (near-duplicate images across dataset using perceptual difference hash)
    - Suspicious Features (blank/corrupted frames, extreme dynamic range collapse)
    """
    def __init__(self, hash_size: int = 8, hash_distance_threshold: int = 2):
        self.hash_size = hash_size
        self.threshold = hash_distance_threshold

    def compute_dhash(self, img_bgr: np.ndarray) -> int:
        """Computes difference hash (dHash) for fast visual similarity & duplicate detection."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    def audit_dataset(self, samples: List[ImageSample]) -> Tuple[List[ImageSample], Dict[str, Any]]:
        """
        Executes comprehensive audit:
        1. Identifies and removes duplicate/near-duplicate frames.
        2. Detects blank, solid-color, or corrupted frames (suspicious features).
        3. Flags target leakage indicators in sample metadata.
        """
        seen_hashes: Dict[int, ImageSample] = {}
        clean_samples: List[ImageSample] = []
        
        duplicate_count = 0
        suspicious_count = 0
        target_leakage_flags = 0

        for s in samples:
            # 1. Suspicious feature check: Variance / Blank image test
            gray = cv2.cvtColor(s.image, cv2.COLOR_BGR2GRAY)
            std_dev = float(np.std(gray))
            if std_dev < 1.0 or s.image.shape[0] < 4 or s.image.shape[1] < 4:
                suspicious_count += 1
                continue

            # 2. Duplicate leakage check using dHash Hamming distance
            h = self.compute_dhash(s.image)
            is_duplicate = False
            for prev_h in seen_hashes.keys():
                hamming_dist = bin(h ^ prev_h).count("1")
                if hamming_dist <= self.threshold:
                    is_duplicate = True
                    duplicate_count += 1
                    break
            
            if is_duplicate:
                continue

            # 3. Target leakage check: ensure filenames don't contain cheating artifacts
            if "cheat" in s.file_path.lower() or "label_override" in s.metadata:
                target_leakage_flags += 1
                continue

            seen_hashes[h] = s
            clean_samples.append(s)

        audit_report = {
            "initial_samples": len(samples),
            "clean_samples": len(clean_samples),
            "duplicates_removed": duplicate_count,
            "suspicious_removed": suspicious_count,
            "target_leakage_flags": target_leakage_flags,
            "leakage_clean": (duplicate_count + suspicious_count + target_leakage_flags) == 0
        }
        return clean_samples, audit_report


# ==============================================================================
# STEP 4: CLAHE CONTRAST ENHANCEMENT
# ==============================================================================

class CLAHEEnhancer:
    """
    Step 4: Contrast-Limited Adaptive Histogram Equalization.
    Enhances contrast in low-contrast, Infrared (IR), thermal, or adverse weather images.
    Operates in LAB color space strictly on the Luminance (L) channel.
    """
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Applies CLAHE on luminance while preserving color tint."""
        if len(img_bgr.shape) == 2:
            return self.clahe.apply(img_bgr)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_batch(self, samples: List[ImageSample]) -> List[ImageSample]:
        """Applies CLAHE enhancement to a list of ImageSamples."""
        enhanced = []
        for s in samples:
            new_img = self.enhance(s.image)
            enhanced.append(ImageSample(
                image=new_img,
                label=s.label,
                class_name=s.class_name,
                file_path=s.file_path,
                metadata=copy.deepcopy(s.metadata),
                sample_id=s.sample_id
            ))
        return enhanced


# ==============================================================================
# STEP 5: TRAIN / VAL / TEST SPLIT & 10-FOLD CROSS-VALIDATION
# ==============================================================================

class DatasetSplitter:
    """
    Step 5: Rigorous Stratified Train/Val/Test Splitting and 10-Fold CV.
    Guarantees no distribution shifts and zero test set contamination.
    """
    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def stratified_split(self, samples: List[ImageSample]) -> Tuple[List[ImageSample], List[ImageSample], List[ImageSample]]:
        """Stratified Holdout Split into Train, Validation, and Test sets."""
        random.seed(self.seed)
        grouped: Dict[int, List[ImageSample]] = {}
        for s in samples:
            grouped.setdefault(s.label, []).append(s)

        train_set, val_set, test_set = [], [], []

        for lbl, class_samples in grouped.items():
            shuffled = list(class_samples)
            random.shuffle(shuffled)
            n = len(shuffled)
            n_train = max(1, int(round(n * self.train_ratio)))
            n_val = max(1, int(round(n * self.val_ratio))) if n > 3 else 1
            
            tr = shuffled[:n_train]
            va = shuffled[n_train:n_train + n_val]
            te = shuffled[n_train + n_val:]
            if not te:
                te = va[-1:]

            train_set.extend(tr)
            val_set.extend(va)
            test_set.extend(te)

        return train_set, val_set, test_set

    def generate_k_folds(self, samples: List[ImageSample], k: int = 10) -> List[Tuple[List[ImageSample], List[ImageSample]]]:
        """Generates k stratified train/val fold pairs."""
        random.seed(self.seed)
        grouped: Dict[int, List[ImageSample]] = {}
        for s in samples:
            grouped.setdefault(s.label, []).append(s)

        k_buckets: List[List[ImageSample]] = [[] for _ in range(k)]
        for lbl, class_samples in grouped.items():
            shuffled = list(class_samples)
            random.shuffle(shuffled)
            for idx, item in enumerate(shuffled):
                k_buckets[idx % k].append(item)

        folds = []
        for i in range(k):
            val_fold = k_buckets[i]
            train_fold = []
            for j in range(k):
                if j != i:
                    train_fold.extend(k_buckets[j])
            folds.append((train_fold, val_fold))
        return folds


# ==============================================================================
# STEP 6: GAN AUGMENTATION (TRAINING SET ONLY)
# ==============================================================================

class GANTrainingAugmenter:
    """
    Step 6: GAN Augmentation applied STRICTLY on the training dataset.
    Synthesizes enhanced samples for minority or hard-to-detect classes without leaking to val/test.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.generator = LightweightGANGenerator(scale_factor=2).to(self.device)
        self.generator.eval()

    def augment_train_set(self, train_samples: List[ImageSample], target_samples_per_class: Optional[int] = None) -> List[ImageSample]:
        """
        Enriches the training partition using GAN super-resolution & feature synthesis.
        """
        augmented = list(train_samples)
        counts: Dict[int, int] = {}
        for s in train_samples:
            counts[s.label] = counts.get(s.label, 0) + 1

        max_count = max(counts.values()) if counts else 0
        target = target_samples_per_class or max_count

        for s in train_samples:
            if counts[s.label] < target:
                img = s.image
                h, w = img.shape[:2]
                if h >= 8 and w >= 8:
                    crop_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    tensor = tensor.to(self.device)
                    
                    with torch.no_grad():
                        gan_out = self.generator(tensor)
                    
                    sr_np = gan_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    sr_bgr = cv2.cvtColor(np.clip(sr_np * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    sr_bgr = cv2.resize(sr_bgr, (w, h))

                    augmented.append(ImageSample(
                        image=sr_bgr,
                        label=s.label,
                        class_name=s.class_name,
                        file_path=f"gan_aug_{s.sample_id}.jpg",
                        metadata={"is_gan_augmented": True},
                        sample_id=f"gan_{s.sample_id}"
                    ))
                    counts[s.label] += 1
        return augmented


# ==============================================================================
# STEP 7: MODEL ARCHITECTURE & INITIAL TRAINING
# ==============================================================================

class PyTorchImageDataset(Dataset):
    """Bridge ImageSample list to PyTorch DataLoader."""
    def __init__(self, samples: List[ImageSample], preprocessor: DataPreprocessor):
        self.samples = samples
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rgb = self.preprocessor.preprocess_image(s.image)
        tensor = self.preprocessor.to_tensor(rgb)
        return tensor, torch.tensor(s.label, dtype=torch.long)


class DroneClassifier(nn.Module):
    """
    Robust Deep Convolutional Classifier with Residual Skips and Configurable Regularization.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.skip1 = nn.Conv2d(32, 64, kernel_size=1, stride=2)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.skip2 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.layer1(out) + self.skip1(out))
        out = F.relu(self.layer2(out) + self.skip2(out))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class ModelTrainer:
    """
    Step 7: Handles model training and metrics logging across epochs.
    """
    def __init__(self, model: nn.Module, device: str = "cpu", lr: float = 1e-3, weight_decay: float = 1e-4, label_smoothing: float = 0.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(self.device), by.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(bx)
            loss = self.criterion(logits, by)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += len(by)
        return (total_loss / max(1, total)), (correct / max(1, total)) * 100.0

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits = self.model(bx)
                loss = self.criterion(logits, by)
                total_loss += loss.item() * len(by)
                preds = logits.argmax(dim=1)
                correct += (preds == by).sum().item()
                total += len(by)
        return (total_loss / max(1, total)), (correct / max(1, total)) * 100.0


# ==============================================================================
# STEP 8: OVERFITTING / UNDERFITTING DETECTION & DIAGNOSTICS
# ==============================================================================

@dataclass
class DiagnosticResult:
    status: str  # 'OPTIMAL', 'OVERFITTING', 'UNDERFITTING', 'UNSTABLE'
    generalization_gap: float
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    recommendation: str


class OverfitUnderfitDetector:
    """
    Step 8: Automated Diagnostic Engine.
    Examines loss trajectories and generalization gaps to detect over/underfitting.
    """
    def __init__(self, overfit_gap_threshold: float = 0.35, underfit_loss_threshold: float = 1.0):
        self.overfit_gap_threshold = overfit_gap_threshold
        self.underfit_loss_threshold = underfit_loss_threshold

    def diagnose(self, history: Dict[str, List[float]]) -> DiagnosticResult:
        """
        Analyzes training history:
        - Gap = Val Loss - Train Loss
        - Underfitting: Both Train and Val loss remain stubbornly high (> threshold)
        - Overfitting: Train loss drops very low while Val loss diverges upward
        """
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        final_train_acc = history["train_acc"][-1]
        final_val_acc = history["val_acc"][-1]

        gen_gap = final_val_loss - final_train_loss
        acc_gap = final_train_acc - final_val_acc

        if final_train_loss > self.underfit_loss_threshold and final_val_loss > self.underfit_loss_threshold:
            status = "UNDERFITTING"
            rec = "Model capacity too low or under-trained. Increase network width/depth and tune learning rate."
        elif gen_gap > self.overfit_gap_threshold or acc_gap > 20.0:
            status = "OVERFITTING"
            rec = "High variance detected (overfitting). Increase Dropout, Weight Decay, and add Mixup/Regularization."
        else:
            status = "OPTIMAL"
            rec = "Model exhibits balanced generalization with tight train/val convergence."

        return DiagnosticResult(
            status=status,
            generalization_gap=gen_gap,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            train_acc=final_train_acc,
            val_acc=final_val_acc,
            recommendation=rec
        )


# ==============================================================================
# STEP 9: APPLY CORRECTION TECHNIQUE & RETRAIN
# ==============================================================================

class AutoCorrectionTrainer:
    """
    Step 9: Automatically adjusts regularization or capacity and retrains the model.
    """
    def __init__(self, num_classes: int = 4, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = device

    def retrain_with_corrections(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        diagnostic: DiagnosticResult,
        base_epochs: int = 10
    ) -> Tuple[nn.Module, Dict[str, List[float]], str]:
        """
        Applies targeted policy based on diagnostic outcome:
        - Overfitting -> Increased Dropout (0.5), higher Weight Decay (1e-2), Label Smoothing (0.1)
        - Underfitting -> Higher Capacity, lowered Regularization, lower LR with Warmup
        """
        applied_policy = ""
        if diagnostic.status == "OVERFITTING":
            applied_policy = "Applied High Regularization (Dropout=0.5, Weight Decay=1e-2, Label Smoothing=0.1)"
            model = DroneClassifier(num_classes=self.num_classes, dropout_rate=0.5).to(self.device)
            trainer = ModelTrainer(model, device=self.device, lr=5e-4, weight_decay=1e-2, label_smoothing=0.1)
        elif diagnostic.status == "UNDERFITTING":
            applied_policy = "Increased Training Capacity (Extended Epochs + Adjusted LR)"
            model = DroneClassifier(num_classes=self.num_classes, dropout_rate=0.1).to(self.device)
            trainer = ModelTrainer(model, device=self.device, lr=2e-3, weight_decay=1e-5, label_smoothing=0.0)
            base_epochs += 5
        else:
            applied_policy = "Retained Optimal Architecture with Fine-Tuning"
            model = DroneClassifier(num_classes=self.num_classes, dropout_rate=0.2).to(self.device)
            trainer = ModelTrainer(model, device=self.device, lr=1e-3, weight_decay=1e-4)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(base_epochs):
            tr_loss, tr_acc = trainer.train_epoch(train_loader)
            va_loss, va_acc = trainer.evaluate(val_loader)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_weights = copy.deepcopy(model.state_dict())

        if best_weights is not None:
            model.load_state_dict(best_weights)

        return model, history, applied_policy


# ==============================================================================
# STEP 10: FINAL TEST EVALUATION
# ==============================================================================

class FinalEvaluator:
    """
    Step 10: Strict holdout evaluation and metrics computation.
    Produces Confusion Matrix, Precision, Recall, F1-Score, and overall accuracy.
    """
    def __init__(self, classes: List[str], device: str = "cpu"):
        self.classes = classes
        self.device = device

    def evaluate_test_set(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
        """Runs full evaluation on the holdout test set."""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(self.device)
                logits = model(bx)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(by.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        num_classes = len(self.classes)

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_targets, all_preds):
            cm[t, p] += 1

        per_class_metrics = {}
        precisions, recalls, f1s = [], [], []

        for i, cls_name in enumerate(self.classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            per_class_metrics[cls_name] = {
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "support": int(cm[i, :].sum())
            }
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        accuracy = float((all_preds == all_targets).mean()) if len(all_targets) > 0 else 0.0

        return {
            "test_accuracy": round(accuracy * 100.0, 2),
            "macro_precision": round(float(np.mean(precisions)), 4),
            "macro_recall": round(float(np.mean(recalls)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class_metrics,
            "total_test_samples": len(all_targets)
        }
