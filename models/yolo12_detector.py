"""
YOLO Detector Interface for Drone Detection.
Wraps YOLO12 / YOLOv8 / YOLOv11 models with Ultralytics and custom trained checkpoints.
"""

import os
import torch
import numpy as np
import cv2

CLASSES = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}


class YOLO12Detector:
    """
    Detector supporting YOLO12 / Ultralytics models.
    """
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.25, iou_thresh=0.45, device=None):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = CLASSES

        # Load Ultralytics YOLO model
        try:
            from ultralytics import YOLO
            if os.path.exists(model_path) or model_path.endswith(".pt") or model_path.endswith(".yaml"):
                self.model = YOLO(model_path)
                print(f"[YOLO12Detector] Loaded model from {model_path}")
            else:
                self.model = YOLO("yolov8n.pt")
                print(f"[YOLO12Detector] Loaded baseline model yolov8n.pt")
        except Exception as e:
            print(f"[YOLO12Detector] Warning: Ultralytics initialization note: {e}")

    def detect_thermal_saliency(self, frame, min_area=15, max_area=8000):
        """
        Adaptive Infrared/Optical Saliency Detector for Small Airborne Thermal Targets.
        Extracts high-contrast heat signatures using morphological top-hat filtering.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Morphological Top-Hat filter to isolate bright small objects against sky background
        kernel_size = min(21, max(9, w // 25))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Adaptive thresholding
        mean_val, std_val = cv2.meanStdDev(tophat)
        thresh_val = float(mean_val[0][0] + 3.2 * std_val[0][0])
        _, binary = cv2.threshold(tophat, max(22.0, thresh_val), 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / float(bh) if bh > 0 else 1.0
                # Airborne objects are typically in upper 85% of frame and not thin screen borders
                if 0.2 < aspect < 5.0 and int(h * 0.08) < by < int(h * 0.88) and bw < int(w * 0.4):
                    pad = max(3, int(max(bw, bh) * 0.25))
                    x1 = max(0, bx - pad)
                    y1 = max(0, by - pad)
                    x2 = min(w, bx + bw + pad)
                    y2 = min(h, by + bh + pad)

                    patch = gray[y1:y2, x1:x2]
                    contrast = float(patch.max() - patch.min()) / 255.0 if patch.size > 0 else 0.5
                    conf = float(np.clip(0.60 + contrast * 0.35, 0.50, 0.95))

                    candidates.append({
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "confidence": conf,
                        "class_id": 2,          # DRONE / TARGET
                        "class_name": "DRONE"
                    })

        # Keep top 5 most salient targets to avoid clutter
        candidates = sorted(candidates, key=lambda d: d["confidence"], reverse=True)[:5]
        return candidates

    def detect(self, frame, sensor_type="IR"):
        """
        Run detection on a single BGR frame.
        Fuses YOLO neural network detections with thermal saliency extraction for IR targets.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        detections = []

        if self.model is not None:
            try:
                results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False, device=self.device)[0]
                boxes = results.boxes

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(float)
                    conf = float(box.conf[0].cpu().numpy())
                    raw_cls_id = int(box.cls[0].cpu().numpy())

                    # Robustly map Ultralytics class names to project internal constants
                    raw_name = str(results.names.get(raw_cls_id, "")).upper() if hasattr(results, 'names') else ""
                    
                    if "AIRPLANE" in raw_name or "PLANE" in raw_name:
                        cls_name = "AIRPLANE"
                        cls_id = 0
                    elif "BIRD" in raw_name:
                        cls_name = "BIRD"
                        cls_id = 1
                    elif "HELICOPTER" in raw_name:
                        cls_name = "HELICOPTER"
                        cls_id = 3
                    elif "DRONE" in raw_name or raw_name == "":
                        # Default to DRONE only if there's no names dictionary but we have a detection
                        cls_name = "DRONE"
                        cls_id = 2
                    else:
                        print(f"[YOLO12Detector] Warning: Unknown class detected: {raw_name} (ID: {raw_cls_id}). Skipping.")
                        continue

                    detections.append({
                        "bbox": (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    })
            except Exception:
                pass

        # If no YOLO detection found on thermal feed, activate thermal saliency detector
        if len(detections) == 0:
            thermal_dets = self.detect_thermal_saliency(frame)
            detections.extend(thermal_dets)

        return detections
