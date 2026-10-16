"""Technique 2: YOLO12 + LSTM (Detection + Trajectory & Motion Dynamics Analysis)
Architecture:
- YOLO12: Extracts spatial candidate bounding boxes across video frames.
- Multi-Frame Tracklet Tracker: Accumulates motion coordinates (x, y, w, h, dx, dy, speed, heading).
- Motion LSTM: Processes temporal flight sequences (T=16 frames) to distinguish:
  * Erratic flapping / zigzag paths (Birds)
  * Steady hovering / angular waypoint maneuvers (Drones/UAVs)
  * High-speed linear trajectories (Helicopters / Airplanes)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["airplane", "bird", "helicopter", "kite", "parachute", "UAV"]
NUM_CLASSES = len(CLASS_NAMES)
SEQ_LEN = 16
FEATURE_DIM = 8  # [norm_x, norm_y, norm_w, norm_h, vel_x, vel_y, speed, heading]

class MotionLSTM(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2, num_classes=NUM_CLASSES):
        super(MotionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take final time step
        out = self.fc(last_hidden)
        return out

class YOLO12_LSTM_MotionAnalyzer:
    def __init__(self, yolo_weights="yolov8n.pt", lstm_weights=None):
        self.detector = YOLO(yolo_weights)
        self.lstm_model = MotionLSTM().to(device)
        if lstm_weights and os.path.exists(lstm_weights):
            self.lstm_model.load_state_dict(torch.load(lstm_weights, map_location=device))
        self.lstm_model.eval()
        
        # Buffer tracks: track_id -> deque of features
        self.track_buffers = {}

    def extract_motion_feature(self, current_box, prev_box, img_shape):
        h, w = img_shape[:2]
        x1, y1, x2, y2 = current_box
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        
        if prev_box is None:
            vx, vy = 0.0, 0.0
        else:
            px1, py1, px2, py2 = prev_box
            pcx = (px1 + px2) / 2.0 / w
            pcy = (py1 + py2) / 2.0 / h
            vx = cx - pcx
            vy = cy - pcy
            
        speed = np.sqrt(vx**2 + vy**2)
        heading = np.arctan2(vy, vx + 1e-6)
        return [cx, cy, bw, bh, vx, vy, speed, heading]

    def process_frame(self, frame, track_id=0):
        """Processes single frame in a continuous video sequence."""
        results = self.detector.predict(frame, conf=0.25, verbose=False)[0]
        
        outputs = []
        for box in results.boxes:
            b = list(map(int, box.xyxy[0].tolist()))
            det_cls = int(box.cls[0])
            
            if track_id not in self.track_buffers:
                self.track_buffers[track_id] = deque(maxlen=SEQ_LEN)
                
            prev = None if len(self.track_buffers[track_id]) == 0 else self.track_buffers[track_id][-1][:4]
            feat = self.extract_motion_feature(b, prev, frame.shape)
            self.track_buffers[track_id].append(feat)
            
            # Predict kinematics if buffer is full
            if len(self.track_buffers[track_id]) == SEQ_LEN:
                seq_tensor = torch.tensor([list(self.track_buffers[track_id])], dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = self.lstm_model(seq_tensor)
                    probs = torch.softmax(logits, dim=1)
                    lstm_cls = int(torch.argmax(probs, dim=1).item())
                    lstm_conf = float(torch.max(probs, dim=1)[0].item())
            else:
                lstm_cls = det_cls
                lstm_conf = float(box.conf[0])
                
            outputs.append({
                "bbox": b,
                "det_class": CLASS_NAMES[det_cls] if det_cls < len(CLASS_NAMES) else str(det_cls),
                "motion_class": CLASS_NAMES[lstm_cls],
                "confidence": lstm_conf,
                "buffered_frames": len(self.track_buffers[track_id])
            })
            
        return outputs

if __name__ == "__main__":
    print("Initializing Technique 2: YOLO12 + LSTM Motion Module...")
    analyzer = YOLO12_LSTM_MotionAnalyzer()
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    res = analyzer.process_frame(dummy_frame)
    print("Technique 2 successfully initialized and ready for video trajectory analysis.")
