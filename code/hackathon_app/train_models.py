"""
Unified Training Script for Drone Detection System:
1. YOLO12 / YOLOv8 Object Detector on drone dataset
2. GAN Super-Resolution & Contrast Enhancer
3. Trajectory Transformer for kinematic behavior modeling
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.gan_enhancer import LightweightGANGenerator
from models.temporal_transformer import TrajectoryTransformer, BEHAVIORS


def parse_args():
    parser = argparse.ArgumentParser(description="Train Models for Drone Detection Pipeline")
    parser.add_argument("--model", type=str, default="transformer", choices=["yolo", "gan", "transformer", "all"], help="Which model to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def train_trajectory_transformer(epochs=10, batch_size=32, lr=1e-3, device="cpu"):
    """
    Trains the Trajectory Transformer on synthetic & sampled kinematic tracks.
    """
    print("\n" + "=" * 50)
    print("TRAINING SPATIO-TEMPORAL TRAJECTORY TRANSFORMER")
    print("=" * 50)

    # Generate synthetic kinematics representative of the 5 behavior classes
    num_samples_per_class = 200
    seq_len = 16
    in_features = 9

    X_list = []
    y_list = []

    np.random.seed(42)

    for cls_idx, beh in enumerate(BEHAVIORS):
        for _ in range(num_samples_per_class):
            seq = np.zeros((seq_len, in_features), dtype=np.float32)
            # base starting position
            x0 = np.random.uniform(0.2, 0.8)
            y0 = np.random.uniform(0.2, 0.8)
            w0 = np.random.uniform(0.02, 0.08)
            h0 = w0 * np.random.uniform(0.8, 1.2)

            if beh == "HOVERING":
                # minimal motion, slight jitter
                vx = np.random.normal(0, 0.001, seq_len)
                vy = np.random.normal(0, 0.001, seq_len)
                vscale = np.random.normal(0, 0.0001, seq_len)
            elif beh == "APPROACHING":
                # bounding box grows rapidly, moving towards center
                vx = np.random.normal(0.005, 0.002, seq_len)
                vy = np.random.normal(0.005, 0.002, seq_len)
                vscale = np.linspace(0.001, 0.005, seq_len)
            elif beh == "TRANSITING":
                # steady linear crossing
                vx = np.ones(seq_len) * np.random.choice([-0.015, 0.015])
                vy = np.random.normal(0, 0.002, seq_len)
                vscale = np.random.normal(0, 0.0001, seq_len)
            elif beh == "DIVING":
                # rapid downward velocity and accelerating
                vx = np.random.normal(0, 0.003, seq_len)
                vy = np.linspace(0.01, 0.04, seq_len)
                vscale = np.linspace(0.001, 0.003, seq_len)
            else:  # NATURAL
                # sinusoidal flap / erratic
                t = np.linspace(0, 4 * np.pi, seq_len)
                vx = 0.01 * np.cos(t) + np.random.normal(0, 0.005, seq_len)
                vy = 0.01 * np.sin(t) + np.random.normal(0, 0.005, seq_len)
                vscale = np.random.normal(0, 0.0002, seq_len)

            cur_x, cur_y, cur_w, cur_h = x0, y0, w0, h0
            for t_step in range(seq_len):
                cur_x = np.clip(cur_x + vx[t_step], 0.0, 1.0)
                cur_y = np.clip(cur_y + vy[t_step], 0.0, 1.0)
                cur_w = np.clip(cur_w + vscale[t_step], 0.01, 0.5)
                cur_h = np.clip(cur_h + vscale[t_step], 0.01, 0.5)
                ax = vx[t_step] - (vx[t_step-1] if t_step > 0 else 0)
                ay = vy[t_step] - (vy[t_step-1] if t_step > 0 else 0)
                seq[t_step] = [cur_x, cur_y, cur_w, cur_h, vx[t_step], vy[t_step], ax, ay, vscale[t_step]]

            X_list.append(seq)
            y_list.append(cls_idx)

    X_train = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_list), dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TrajectoryTransformer(in_features=in_features, num_classes=len(BEHAVIORS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        acc = (correct / total) * 100.0
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/total:.4f} - Accuracy: {acc:.1f}%")

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/trajectory_transformer.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved Trajectory Transformer checkpoint: {save_path}")


def train_gan_enhancer(epochs=5, batch_size=8, lr=2e-4, device="cpu"):
    """
    Trains the lightweight Super-Resolution GAN Generator.
    """
    print("\n" + "=" * 50)
    print("TRAINING LIGHTWEIGHT GAN ENHANCER")
    print("=" * 50)

    model = LightweightGANGenerator(scale_factor=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()

    # Synthetic training on downscaled patches
    dummy_hr = torch.rand(batch_size * 20, 3, 64, 64)
    dummy_lr = nn.functional.interpolate(dummy_hr, scale_factor=0.5, mode="bilinear")

    dataset = TensorDataset(dummy_lr, dummy_hr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for lr_batch, hr_batch in loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            optimizer.zero_grad()
            sr = model(lr_batch)
            loss = l1_loss(sr, hr_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - GAN L1 Loss: {total_loss/len(loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/gan_enhancer.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved GAN Enhancer checkpoint: {save_path}")


def train_yolo(epochs=30, imgsz=640, data_yaml="dataset_yolo/data.yaml"):
    """
    Trains Ultralytics YOLO model on dataset_yolo.
    """
    print("\n" + "=" * 50)
    print("TRAINING YOLO OBJECT DETECTOR")
    print("=" * 50)
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, project="runs/detect", name="drone_detector")
        print("YOLO Training finished successfully!")
    except Exception as e:
        print(f"YOLO training error: {e}")


def main():
    args = parse_args()
    if args.model in ["transformer", "all"]:
        train_trajectory_transformer(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    if args.model in ["gan", "all"]:
        train_gan_enhancer(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    if args.model in ["yolo", "all"]:
        train_yolo(epochs=args.epochs)


if __name__ == "__main__":
    main()
