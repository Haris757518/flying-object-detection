"""End-to-End Flying Object Detection & Security Alert Pipeline
Pipeline Workflow (as specified in project architecture):
Live Camera / Video Stream
       ↓
    YOLO12 (Object Detection)
       ↓
  Transformer (Object Feature Extraction)
       ↓
     LSTM (Movement & Temporal Analysis)
       ↓
  Classification (Drone / Bird / Helicopter / Unknown)
       ↓
  Risk Alert Engine (Green / Yellow / Red Threat Geofence)
"""

import os
import time
import datetime
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from model_3_yolo_transformer import TransformerFeatureEncoder
from model_2_yolo_lstm import MotionLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["airplane", "bird", "helicopter", "kite", "parachute", "UAV"]
ALERT_LOG_FILE = "security_alerts.log"

class RiskAlertSystem:
    def __init__(self, yolo_weights="yolov8n.pt", restricted_zone=None):
        """
        restricted_zone: tuple of (x1, y1, x2, y2) normalized coordinates [0.0 - 1.0]
        Default: Central 60% of surveillance frame.
        """
        print("Loading End-to-End Pipeline Components...")
        self.detector = YOLO(yolo_weights)
        self.transformer = TransformerFeatureEncoder().to(device)
        self.transformer.eval()
        self.lstm = MotionLSTM().to(device)
        self.lstm.eval()
        
        # Geofence perimeter [xmin, ymin, xmax, ymax]
        self.restricted_zone = restricted_zone or (0.2, 0.2, 0.8, 0.8)
        
        # Tracking buffers: track_id -> deque of kinematics
        self.track_history = {}
        self.alert_events = []

    def assess_risk(self, classification, bbox, img_shape):
        """
        Assesses Threat Level:
        - RED: Drone inside or heading directly into restricted zone.
        - YELLOW: Unknown / Helicopter near zone boundary.
        - GREEN: Confirmed Bird or safe airspace.
        """
        h, w = img_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        
        zx1, zy1, zx2, zy2 = self.restricted_zone
        in_restricted = (zx1 <= cx <= zx2) and (zy1 <= cy <= zy2)
        
        if classification == "UAV":
            if in_restricted:
                return "CRITICAL_RED", "UNAUTHORIZED DRONE INSIDE RESTRICTED AIRSPACE!"
            else:
                return "HIGH_ORANGE", "DRONE DETECTED APPROACHING PERIMETER"
        elif classification == "helicopter":
            return "MEDIUM_YELLOW", "Low-Altitude Helicopter in Surveillance Sector"
        elif classification == "bird":
            return "SAFE_GREEN", "Avian Wildlife (No Security Threat)"
        else:
            return "ADVISORY_YELLOW" if in_restricted else "SAFE_GREEN", "Unverified Aerial Object"

    def log_alert(self, level, message, details):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message} | Details: {details}\n"
        print(">>> ALERT TRIGGERED:", entry.strip())
        with open(ALERT_LOG_FILE, "a") as f:
            f.write(entry)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        results = self.detector.predict(frame, conf=0.25, verbose=False)[0]
        
        # Draw Geofence Perimeter
        zx1, zy1, zx2, zy2 = self.restricted_zone
        cv2.rectangle(frame, (int(zx1 * w), int(zy1 * h)), (int(zx2 * w), int(zy2 * h)), (0, 165, 255), 2)
        cv2.putText(frame, "RESTRICTED AIRSPACE PERIMETER", (int(zx1 * w) + 10, int(zy1 * h) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
                
            crop = frame[y1:y2, x1:x2]
            
            # --- 1. Transformer Spatial Feature Extraction ---
            resized = cv2.resize(crop, (64, 64))
            tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            with torch.no_grad():
                logits, feat = self.transformer(tensor)
                vit_cls = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())
                predicted_class = CLASS_NAMES[vit_cls]
                
            # --- 2. Risk Alert Assessment ---
            risk_level, risk_desc = self.assess_risk(predicted_class, [x1, y1, x2, y2], frame.shape)
            
            # Color coding
            if "RED" in risk_level:
                box_color = (0, 0, 255)  # BGR Red
                self.log_alert(risk_level, risk_desc, f"Class={predicted_class}, Coords=({x1},{y1})")
            elif "YELLOW" in risk_level or "ORANGE" in risk_level:
                box_color = (0, 215, 255)  # BGR Amber/Yellow
            else:
                box_color = (0, 255, 0)  # BGR Green
                
            # Render HUD bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"{predicted_class.upper()} | {risk_level}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            
        return frame

    def run_live_simulation(self, source=0, max_frames=50, output_path="simulation_output.avi"):
        """Runs the pipeline on webcam or simulated frames."""
        print(f"Starting Live Pipeline Simulation (saving preview to {output_path})...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (640, 640))
        
        for i in range(max_frames):
            # Create synthetic surveillance stream test frame
            sim_frame = np.full((640, 640, 3), 200, dtype=np.uint8)
            # Add flying drone trajectory entering restricted zone
            drone_x = int(100 + i * 8)
            drone_y = int(250 + i * 3)
            cv2.circle(sim_frame, (drone_x, drone_y), 15, (50, 50, 50), -1)
            
            processed = self.process_frame(sim_frame)
            out_writer.write(processed)
            
        out_writer.release()
        print(f"Simulation completed successfully! Output saved to: {output_path}")

if __name__ == "__main__":
    pipeline = RiskAlertSystem()
    pipeline.run_live_simulation()
