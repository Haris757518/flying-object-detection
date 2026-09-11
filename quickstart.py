"""
Drone Detection Dataset - Quickstart & Demo Usage
Demonstrates how to use the dataset with:
1. PyTorch DataLoader (DroneImageDataset & DroneDirectVideoDataset)
2. PyTorch Audio DataLoader (DroneAudioDataset)
3. Ultralytics YOLO (one-line model training & inference)
"""

import os
import torch
from torch.utils.data import DataLoader
from drone_dataset import DroneImageDataset, DroneDirectVideoDataset, collate_fn
from drone_audio_dataset import DroneAudioDataset

def demo_pytorch_image_dataset():
    print("\n--- 1. Testing PyTorch DroneImageDataset (from dataset_yolo) ---")
    dataset_dir = "dataset_yolo"
    if not os.path.exists(dataset_dir):
        print("Run python prepare_dataset.py first!")
        return

    train_ds = DroneImageDataset(root_dir=dataset_dir, split="train")
    print(f"Loaded train dataset with {len(train_ds)} samples.")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for images, targets in loader:
        print(f"Batch loaded: {len(images)} images.")
        for i, target in enumerate(targets):
            print(f"  Sample {i}: boxes={target['boxes'].shape}, labels={target['labels']}")
        break


def demo_pytorch_audio_dataset():
    print("\n--- 2. Testing PyTorch DroneAudioDataset ---")
    audio_ds = DroneAudioDataset(data_dir="Data/Audio")
    print(f"Loaded audio dataset with {len(audio_ds)} audio clips.")

    loader = DataLoader(audio_ds, batch_size=4, shuffle=True)
    for waveforms, labels, meta in loader:
        print(f"Waveform batch shape: {waveforms.shape}, Labels: {labels}")
        break


def print_yolo_instructions():
    print("\n--- 3. How to Train with Ultralytics YOLO ---")
    print("To train YOLOv8/YOLOv11 on this dataset, run:")
    print("  pip install ultralytics")
    print("  yolo detect train data=dataset_yolo/data.yaml model=yolov8n.pt epochs=50 imgsz=640")


if __name__ == "__main__":
    demo_pytorch_image_dataset()
    demo_pytorch_audio_dataset()
    print_yolo_instructions()
