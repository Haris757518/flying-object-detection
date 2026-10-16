"""GAN Synthetic Data Augmentation Module for Flying Object Detection.
Implements:
1. Crop extraction of flying objects (UAV, Bird, Helicopter) strictly from Training Split (No Data Leakage).
2. Deep Convolutional Conditional GAN (cGAN) in PyTorch (Generator + Discriminator).
3. Generates high-fidelity synthetic flying object patches and composes augmented training images with YOLO bounding box labels.
"""

import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Setup Paths
BASE_DIR = os.path.abspath("DS/comb DSF")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(BASE_DIR, "train", "labels")

GAN_DIR = os.path.join(BASE_DIR, "gan_augmentation")
CROPS_DIR = os.path.join(GAN_DIR, "extracted_crops")
SYNTH_DIR = os.path.join(GAN_DIR, "synthetic_images")
CHECKPOINT_DIR = os.path.join(GAN_DIR, "checkpoints")

for d in [GAN_DIR, CROPS_DIR, SYNTH_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# Classes: 1: bird, 2: helicopter, 5: UAV
TARGET_CLASSES = {1: "bird", 2: "helicopter", 5: "UAV"}
PATCH_SIZE = 64
LATENT_DIM = 100
EMBED_DIM = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_training_crops(max_crops_per_class=1000):
    """Crops target objects from the training split only."""
    print("Extracting flying object bounding-box crops from training set...")
    class_crops = {cls: [] for cls in TARGET_CLASSES}
    
    label_files = glob.glob(os.path.join(TRAIN_LBL_DIR, "*.txt"))
    random.seed(42)
    random.shuffle(label_files)
    
    for lbl_path in label_files:
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = os.path.join(TRAIN_IMG_DIR, stem + ".jpg")
        if not os.path.exists(img_path):
            continue
            
        with open(lbl_path, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]
            
        if not lines:
            continue
            
        img = None
        for parts in lines:
            cls_id = int(parts[0])
            if cls_id in TARGET_CLASSES and len(class_crops[cls_id]) < max_crops_per_class:
                if img is None:
                    img = cv2.imread(img_path)
                    if img is None:
                        break
                    h, w, _ = img.shape
                    
                x_c, y_c, bw, bh = map(float, parts[1:])
                x1 = max(0, int((x_c - bw / 2) * w))
                y1 = max(0, int((y_c - bh / 2) * h))
                x2 = min(w, int((x_c + bw / 2) * w))
                y2 = min(h, int((y_c + bh / 2) * h))
                
                if (x2 - x1) > 8 and (y2 - y1) > 8:
                    crop = img[y1:y2, x1:x2]
                    crop_resized = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
                    class_crops[cls_id].append(crop_resized)
                    
        # Check if limits reached
        if all(len(class_crops[c]) >= max_crops_per_class for c in TARGET_CLASSES):
            break
            
    for cls_id, crops in class_crops.items():
        cls_name = TARGET_CLASSES[cls_id]
        save_sub = os.path.join(CROPS_DIR, cls_name)
        os.makedirs(save_sub, exist_ok=True)
        for idx, crop in enumerate(crops):
            cv2.imwrite(os.path.join(save_sub, f"{cls_name}_{idx:04d}.png"), crop)
        print(f"Extracted {len(crops)} crops for {cls_name} (Class {cls_id})")

    return class_crops

# --- Conditional GAN Architecture ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=3, embed_dim=32):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_dim = latent_dim + embed_dim
        
        self.init_linear = nn.Sequential(
            nn.Linear(in_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        x = self.init_linear(x)
        x = x.view(-1, 256, 4, 4)
        return self.conv_blocks(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes=3, embed_dim=32):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 64 * 64)
        
        self.conv_net = nn.Sequential(
            # 4 x 64 x 64 -> 32 x 32 x 32
            nn.Conv2d(4, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> 1x1
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(-1, 1, 64, 64)
        x = torch.cat([img, c], dim=1)
        return self.conv_net(x).view(-1, 1)

class CropsDataset(Dataset):
    def __init__(self, crops_dict):
        self.samples = []
        class_mapping = {1: 0, 2: 1, 5: 2}  # map to 0, 1, 2
        for cls_id, crops in crops_dict.items():
            label = class_mapping[cls_id]
            for c in crops:
                # Convert BGR to RGB, normalize to [-1, 1]
                rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
                self.samples.append((tensor, label, cls_id))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1]

def train_cgan(crops_dict, epochs=15, batch_size=32):
    dataset = CropsDataset(crops_dict)
    if len(dataset) == 0:
        print("No crops available for GAN training.")
        return None
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    print(f"\nTraining Conditional GAN on {device} ({len(dataset)} crop samples, {epochs} epochs)...")
    netG.train()
    netD.train()
    
    for epoch in range(1, epochs + 1):
        loss_d_tot, loss_g_tot = 0.0, 0.0
        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            b_size = real_imgs.size(0)
            
            real_y = torch.ones(b_size, 1, device=device)
            fake_y = torch.zeros(b_size, 1, device=device)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            out_real = netD(real_imgs, labels)
            loss_d_real = criterion(out_real, real_y)
            
            noise = torch.randn(b_size, LATENT_DIM, device=device)
            fake_imgs = netG(noise, labels)
            out_fake = netD(fake_imgs.detach(), labels)
            loss_d_fake = criterion(out_fake, fake_y)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optD.step()
            
            # --- Train Generator ---
            optG.zero_grad()
            out_g = netD(fake_imgs, labels)
            loss_g = criterion(out_g, real_y)
            loss_g.backward()
            optG.step()
            
            loss_d_tot += loss_d.item()
            loss_g_tot += loss_g.item()
            
        avg_d = loss_d_tot / len(loader)
        avg_g = loss_g_tot / len(loader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss D: {avg_d:.4f}, Loss G: {avg_g:.4f}")
        
    torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, "cgan_generator.pth"))
    print(f"Generator saved to {os.path.join(CHECKPOINT_DIR, 'cgan_generator.pth')}")
    return netG

def generate_synthetic_samples(netG, samples_per_class=300):
    """Generates synthetic flying object images and formats them with YOLO bounding box labels."""
    netG.eval()
    print(f"\nGenerating {samples_per_class} synthetic images per class...")
    
    class_rev = {0: 1, 1: 2, 2: 5}  # 0->bird (1), 1->helicopter (2), 2->UAV (5)
    
    out_img_dir = os.path.join(SYNTH_DIR, "images")
    out_lbl_dir = os.path.join(SYNTH_DIR, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    count = 0
    with torch.no_grad():
        for internal_cls, actual_cls in class_rev.items():
            cls_name = TARGET_CLASSES[actual_cls]
            for i in range(samples_per_class):
                noise = torch.randn(1, LATENT_DIM, device=device)
                label_t = torch.tensor([internal_cls], device=device)
                fake_patch = netG(noise, label_t).squeeze(0).cpu().permute(1, 2, 0).numpy()
                fake_patch = ((fake_patch + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                fake_patch_bgr = cv2.cvtColor(fake_patch, cv2.COLOR_RGB2BGR)
                
                # Create atmospheric sky canvas (640x640)
                # Linear gradient sky representing diverse daylight conditions
                sky = np.zeros((640, 640, 3), dtype=np.uint8)
                top_color = np.array([random.randint(180, 240), random.randint(140, 200), random.randint(100, 160)])
                bottom_color = np.array([random.randint(220, 255), random.randint(200, 245), random.randint(180, 230)])
                for row in range(640):
                    alpha = row / 640.0
                    sky[row, :] = ((1 - alpha) * top_color + alpha * bottom_color).astype(np.uint8)
                    
                # Place synthetic patch at realistic sky scale
                scale = random.randint(28, 72)
                scaled_patch = cv2.resize(fake_patch_bgr, (scale, scale))
                
                pos_x = random.randint(50, 640 - scale - 50)
                pos_y = random.randint(50, 480 - scale - 50)
                
                # Blend with background
                sky[pos_y:pos_y + scale, pos_x:pos_x + scale] = scaled_patch
                
                img_filename = f"synth_{cls_name}_{i:04d}.jpg"
                lbl_filename = f"synth_{cls_name}_{i:04d}.txt"
                
                cv2.imwrite(os.path.join(out_img_dir, img_filename), sky)
                
                # Calculate YOLO normalized coordinates
                x_center = (pos_x + scale / 2.0) / 640.0
                y_center = (pos_y + scale / 2.0) / 640.0
                width = scale / 640.0
                height = scale / 640.0
                
                with open(os.path.join(out_lbl_dir, lbl_filename), "w") as f:
                    f.write(f"{actual_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                count += 1
                
    print(f"Generated {count} synthetic training images with labels in: {SYNTH_DIR}")

if __name__ == "__main__":
    crops = extract_training_crops(max_crops_per_class=300)
    gen = train_cgan(crops, epochs=10, batch_size=32)
    if gen is not None:
        generate_synthetic_samples(gen, samples_per_class=150)
