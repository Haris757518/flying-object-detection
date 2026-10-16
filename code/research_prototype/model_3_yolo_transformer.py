"""Technique 3: YOLO12 + Transformer (Detection + Enhanced Spatial Feature Representation)
Architecture:
- YOLO12: Performs airborne object localization.
- Transformer Feature Encoder: Multi-Head Self-Attention (MHSA) Vision Transformer block.
- Models long-range contextual spatial relationships across image patches to suppress sky clutter,
  clouds, bird wing artifacts, and atmospheric glare.
"""

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["airplane", "bird", "helicopter", "kite", "parachute", "UAV"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Multi-Head Self-Attention Vision Transformer Block ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=128):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, Embed, H/patch, W/patch) -> (B, num_patches, Embed)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TransformerFeatureEncoder(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=128, depth=4, num_heads=4, num_classes=NUM_CLASSES):
        super(TransformerFeatureEncoder, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        patches = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1) + self.pos_embed
        
        feat = self.transformer(x)
        cls_feat = self.norm(feat[:, 0])
        logits = self.head(cls_feat)
        return logits, cls_feat

class YOLO12_Transformer_Hybrid:
    def __init__(self, yolo_weights="yolov8n.pt", vit_weights=None):
        self.detector = YOLO(yolo_weights)
        self.vit_encoder = TransformerFeatureEncoder().to(device)
        if vit_weights and os.path.exists(vit_weights):
            self.vit_encoder.load_state_dict(torch.load(vit_weights, map_location=device))
        self.vit_encoder.eval()

    def extract_features_and_predict(self, image_np):
        h, w = image_np.shape[:2]
        results = self.detector.predict(image_np, conf=0.25, verbose=False)[0]
        
        outputs = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if (x2 - x1) > 8 and (y2 - y1) > 8:
                crop = image_np[y1:y2, x1:x2]
                resized = cv2.resize(crop, (64, 64))
                tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                tensor = tensor.to(device)
                
                with torch.no_grad():
                    logits, spatial_feat = self.vit_encoder(tensor)
                    probs = torch.softmax(logits, dim=1)
                    vit_cls = int(torch.argmax(probs, dim=1).item())
                    vit_conf = float(torch.max(probs, dim=1)[0].item())
                    
                outputs.append({
                    "bbox": [x1, y1, x2, y2],
                    "det_class": CLASS_NAMES[int(box.cls[0])] if int(box.cls[0]) < len(CLASS_NAMES) else str(int(box.cls[0])),
                    "transformer_class": CLASS_NAMES[vit_cls],
                    "confidence": vit_conf,
                    "feature_vector_dim": spatial_feat.shape[-1]
                })
        return outputs

if __name__ == "__main__":
    print("Initializing Technique 3: YOLO12 + Transformer Module...")
    hybrid = YOLO12_Transformer_Hybrid()
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    res = hybrid.extract_features_and_predict(dummy)
    print("Technique 3 successfully initialized and ready.")
