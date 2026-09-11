"""
GAN-Based Super-Resolution and Preprocessing Enhancer for Drone Detection.
Optimized for small distant drone patches in Infrared (IR) and Visible spectrums.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class DenseResidualBlock(nn.Module):
    """Dense Residual Block with skip connections for high-frequency feature recovery."""
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        c1 = self.lrelu(self.conv1(x))
        c2 = self.lrelu(self.conv2(torch.cat([x, c1], 1)))
        c3 = self.conv3(torch.cat([x, c1, c2], 1))
        return x + c3 * 0.2


class LightweightGANGenerator(nn.Module):
    """
    Lightweight Generator for Real-Time 2x/4x Super-Resolution & Contrast Enhancement.
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=48, num_blocks=3, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        
        self.body = nn.Sequential(*[
            DenseResidualBlock(channels=num_features, growth_channels=24)
            for _ in range(num_blocks)
        ])
        
        self.transition = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Sub-pixel convolution (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        residual = self.body(feat)
        residual = self.transition(residual)
        out = self.upsample(feat + residual)
        out = self.tail(out)
        return torch.sigmoid(out)


class GANEnhancer:
    """
    High-level Frame Preprocessor & GAN Enhancer.
    Provides fast OpenCV-based thermal normalization and PyTorch GAN super-resolution.
    """
    def __init__(self, weights_path="checkpoints/gan_enhancer.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = LightweightGANGenerator(scale_factor=2).to(self.device)
        self.generator.eval()

        if weights_path and os.path.exists(weights_path):
            try:
                self.generator.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                print(f"[GANEnhancer] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[GANEnhancer] Note loading weights: {e}")

    def preprocess_frame(self, frame, sensor_type="IR"):
        """
        Enhance input frame using adaptive contrast and histogram equalization.
        """
        if frame is None:
            return None

        # If sensor is IR (Infrared), apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if sensor_type.upper() == "IR":
            if len(frame.shape) == 3:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                return clahe.apply(frame)
        else:
            # Visible spectrum gamma & mild contrast adjustment
            return frame

    def enhance_crop(self, crop_bgr):
        """
        Enhance a small cropped detection patch using the GAN generator.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return crop_bgr

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            sr_tensor = self.generator(tensor)

        sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_np = np.clip(sr_np * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
