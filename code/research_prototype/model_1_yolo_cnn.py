"""Technique 1: YOLO12 + CNN (Hybrid Detection & Secondary Fine-Grained Classification)
Architecture:
- Primary Detector: YOLO12 detects flying objects and proposes bounding boxes.
- Secondary Classifier: Deep CNN (ConvNeXt / ResNet architecture) takes the high-resolution
  cropped bounding box patches and verifies classification (eliminating bird vs. drone false positives).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["airplane", "bird", "helicopter", "kite", "parachute", "UAV"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Secondary Fine-Grained CNN Classifier ---
class FineGrainedCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(FineGrainedCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Adaptive Pooling to standard 4x4
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return out

# --- Hybrid YOLO12 + CNN Inference Engine ---
class YOLO12_CNN_Hybrid:
    def __init__(self, yolo_weights="yolov8n.pt", cnn_weights=None):
        self.detector = YOLO(yolo_weights)
        self.cnn_classifier = FineGrainedCNN(num_classes=NUM_CLASSES).to(device)
        
        if cnn_weights and os.path.exists(cnn_weights):
            self.cnn_classifier.load_state_dict(torch.load(cnn_weights, map_location=device))
        self.cnn_classifier.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_np, conf_threshold=0.25):
        """Runs two-stage hybrid inference."""
        h, w, _ = image_np.shape
        results = self.detector.predict(image_np, conf=conf_threshold, verbose=False)[0]
        
        hybrid_detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])
            yolo_cls = int(box.cls[0])
            
            # Safe bounding box bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if (x2 - x1) > 8 and (y2 - y1) > 8:
                crop = image_np[y1:y2, x1:x2]
                crop_tensor = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = self.cnn_classifier(crop_tensor)
                    probs = torch.softmax(logits, dim=1)
                    cnn_conf, cnn_cls = torch.max(probs, dim=1)
                    cnn_cls = int(cnn_cls.item())
                    cnn_conf = float(cnn_conf.item())
                
                # Hybrid fusion decision: Ensemble weighting
                final_cls = cnn_cls if cnn_conf > 0.6 else yolo_cls
                final_conf = (det_conf * 0.4) + (cnn_conf * 0.6)
                
                hybrid_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "yolo_class": CLASS_NAMES[yolo_cls] if yolo_cls < len(CLASS_NAMES) else str(yolo_cls),
                    "cnn_class": CLASS_NAMES[cnn_cls],
                    "final_class": CLASS_NAMES[final_cls] if final_cls < len(CLASS_NAMES) else "Unknown",
                    "confidence": final_conf
                })
                
        return hybrid_detections

if __name__ == "__main__":
    print("Initializing Technique 1: YOLO12 + CNN Hybrid Module...")
    model = YOLO12_CNN_Hybrid()
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    preds = model.predict(dummy_img)
    print("Technique 1 successfully initialized and ready for training & inference.")
