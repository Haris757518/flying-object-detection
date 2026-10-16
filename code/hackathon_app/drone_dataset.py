"""
PyTorch Dataset and DataLoader for Drone Detection Dataset.
Supports:
1. Loading pre-extracted YOLO/COCO formatted image directories.
2. Direct on-the-fly video decoding without requiring massive image extraction.
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from mcos_decoder import load_groundtruth

CLASSES = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}


class DroneImageDataset(Dataset):
    """
    PyTorch Dataset loading pre-extracted images and YOLO txt label files.
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root_dir, "images", split)
        self.labels_dir = os.path.join(root_dir, "labels", split)
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        label_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_file)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc = float(parts[1]) * w
                        yc = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h
                        
                        # Convert to [xmin, ymin, xmax, ymax]
                        xmin = max(0.0, xc - bw / 2.0)
                        ymin = max(0.0, yc - bh / 2.0)
                        xmax = min(float(w), xc + bw / 2.0)
                        ymax = min(float(h), yc + bh / 2.0)

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([h, w]),
            "image_path": img_path
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


class DroneDirectVideoDataset(Dataset):
    """
    On-the-fly video frame dataset (does not require pre-extracting images to disk).
    Directly opens MP4 files and reads frames indexed by frame stride.
    """
    def __init__(self, data_dir="Data", sensor="all", stride=10, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # list of (video_path, frame_idx, bbox, class_id, sensor)

        folders = []
        if sensor in ["all", "IR"]:
            folders.append(("IR", os.path.join(data_dir, "Video_IR")))
        if sensor in ["all", "V"]:
            folders.append(("V", os.path.join(data_dir, "Video_V")))

        print(f"Indexing videos with stride={stride}...")
        for sensor_type, folder in folders:
            if not os.path.exists(folder):
                continue
            mp4s = sorted(glob.glob(os.path.join(folder, "*.mp4")))
            for mp4 in mp4s:
                mat = mp4.replace(".mp4", "_LABELS.mat")
                if not os.path.exists(mat):
                    continue
                cls_name = os.path.basename(mp4).split("_")[1].upper()
                if cls_name not in CLASS_TO_ID:
                    continue
                cls_id = CLASS_TO_ID[cls_name]

                try:
                    bboxes = load_groundtruth(mat)
                except Exception:
                    continue

                for frame_idx in range(0, len(bboxes), stride):
                    bbox = bboxes[frame_idx]
                    self.samples.append({
                        "video_path": mp4,
                        "frame_idx": frame_idx,
                        "bbox": bbox,
                        "class_id": cls_id,
                        "sensor": sensor_type
                    })

        print(f"Ready: {len(self.samples)} frame samples indexed.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        cap = cv2.VideoCapture(item["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame_idx"])
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            # Fallback black image if frame read fails
            frame = np.zeros((256, 320, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        w, h = image.size

        bbox = item["bbox"]
        boxes = []
        labels = []
        if bbox is not None:
            bx, by, bw, bh = bbox
            xmin = max(0.0, float(bx))
            ymin = max(0.0, float(by))
            xmax = min(float(w), xmin + float(bw))
            ymax = min(float(h), ymin + float(bh))
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(item["class_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([h, w]),
            "video_path": item["video_path"],
            "frame_idx": item["frame_idx"]
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


def collate_fn(batch):
    """Standard detection collate function for variable number of objects."""
    return tuple(zip(*batch))
