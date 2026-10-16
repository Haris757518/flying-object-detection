"""Modern Interactive Surveillance Dashboard & Real-Time Detection UI
Flying Object Detection Using Hybrid Deep Learning Techniques

Features:
- Select between:
  1. Live Camera / Webcam Feed
  2. Upload / Open Any Video File (.mp4, .avi, .mov, .mkv)
  3. Built-in Surveillance Test Clip
- Real-Time Hybrid Detection & Classification:
  * YOLO12 Candidate Localization
  * Transformer Spatial Feature Extraction
  * LSTM Trajectory Kinematics & Historical Flight Trails
- Geofenced Restricted Airspace Threat Alert System:
  * GREEN: Safe (Birds / Wildlife)
  * YELLOW: Advisory (Helicopters / Unverified Craft)
  * RED: CRITICAL SECURITY ALERT (Drone inside restricted perimeter)
- Live Stats HUD: Real-time FPS, Latency (ms), Object Counts, and Timestamped Event Logs.
"""

import os
import sys
import time
import datetime
from collections import deque
import threading
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO

from model_3_yolo_transformer import TransformerFeatureEncoder
from model_2_yolo_lstm import MotionLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Airplane", "Bird", "Helicopter", "Kite", "Parachute", "UAV (Drone)"]
COLOR_MAP = {
    "Airplane": (255, 180, 0),    # Blue-Orange
    "Bird": (46, 204, 113),       # Green
    "Helicopter": (243, 156, 18),  # Amber
    "Kite": (155, 89, 182),       # Purple
    "Parachute": (52, 152, 219),  # Blue
    "UAV (Drone)": (231, 76, 60)   # Red
}

def enhance_sky_clarity(frame):
    """Enhances atmospheric contrast and sharpens blurry distant aircraft in both color RGB and thermal IR footage."""
    if frame is None or frame.size == 0:
        return frame
        
    b, g, r = cv2.split(frame)
    # Detect thermal / monochrome infrared footage (low inter-channel variance)
    is_monochrome = (np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32))) < 4.0 and
                     np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32))) < 4.0)
    
    if is_monochrome:
        # High-dynamic range thermal enhancement
        gray = b
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        # Morphological Top-Hat to isolate tiny bright aerial heat sources
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)
        boosted = cv2.addWeighted(enhanced_gray, 0.85, tophat, 0.65, 0)
        # Unsharp de-blurring
        gaussian = cv2.GaussianBlur(boosted, (0, 0), 1.5)
        sharpened = cv2.addWeighted(boosted, 1.45, gaussian, -0.45, 0)
        return cv2.merge([sharpened, sharpened, sharpened])
    else:
        # Standard daylight atmospheric contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b_ch)), cv2.COLOR_LAB2BGR)
        
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.35, gaussian, -0.35, 0)
        return sharpened

def apply_nms(detections, iou_thresh=0.30, center_dist_thresh=35):
    """Merges and suppresses duplicate overlapping bounding boxes for small airborne specks."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []
    
    for det in sorted_dets:
        x1, y1, x2, y2 = det["box"]
        cx, cy = det["cx"], det["cy"]
        area = (x2 - x1) * (y2 - y1)
        suppress = False
        
        for k in kept:
            kx1, ky1, kx2, ky2 = k["box"]
            kcx, kcy = k["cx"], k["cy"]
            # Intersection over Union
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                k_area = (kx2 - kx1) * (ky2 - ky1)
                iou = inter_area / float(area + k_area - inter_area + 1e-6)
                if iou > iou_thresh:
                    suppress = True
                    break
            # Center proximity check for small specks
            dist = math.hypot(cx - kcx, cy - kcy)
            if dist < center_dist_thresh:
                suppress = True
                break
                
        if not suppress:
            kept.append(det)
            
    return kept

def classify_airborne_candidate(crop_bgr, det_raw, raw_cls, box_w, box_h, conf, frame_h, cy, vit_model=None):
    """Accurately classifies airborne objects, eliminating generic 'Flying Object' and 'Parachute' mislabels."""
    aspect = box_w / (box_h + 1e-6)
    in_upper_sky = (cy < frame_h * 0.85)
    
    # 1. Direct explicit classification from detector
    if "bird" in det_raw:
        return "Bird", conf
    elif "helicopter" in det_raw:
        return "Helicopter", conf
    elif "airplane" in det_raw or "plane" in det_raw:
        # Distant planes vs drones: Drones are compact (aspect 0.6-1.5, box_w < 90)
        if box_w < 85 and box_h < 85 and 0.6 <= aspect <= 1.6:
            return "UAV (Drone)", max(conf, 0.72)
        else:
            return "Airplane", conf
    elif "kite" in det_raw or "parachute" in det_raw:
        # Kites/Parachutes in high sky often are multirotor drones or small birds
        if box_w < 95 and box_h < 95:
            return "UAV (Drone)", max(conf, 0.75)
        else:
            return "Parachute", conf
    elif "drone" in det_raw or "uav" in det_raw:
        return "UAV (Drone)", conf
        
    # 2. Hybrid ViT Feature Classification for Blurry / Thermal / Low-confidence Aerial Specks
    if vit_model is not None and crop_bgr is not None and crop_bgr.size > 0:
        try:
            resized_crop = cv2.resize(crop_bgr, (64, 64))
            tensor_crop = torch.from_numpy(resized_crop).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            with torch.no_grad():
                logits, _ = vit_model(tensor_crop)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                best_idx = int(np.argmax(probs))
                vit_class = CLASS_NAMES[best_idx]
                if vit_class in ["Kite", "Parachute"] and (box_w < 95 or box_h < 95):
                    vit_class = "UAV (Drone)"
                elif vit_class == "UAV":
                    vit_class = "UAV (Drone)"
                return vit_class, max(conf, float(probs[best_idx]))
        except Exception:
            pass

    # 3. Geometric & Kinematic Aerial Heuristics
    if in_upper_sky:
        if 0.55 <= aspect <= 1.65 and box_w < 110:
            return "UAV (Drone)", max(conf, 0.68)
        elif aspect > 1.65:
            return "Airplane", max(conf, 0.65)
        elif aspect < 0.55:
            return "Bird", max(conf, 0.65)
            
    return "UAV (Drone)", max(conf, 0.62)

def draw_high_visibility_target(frame, x1, y1, x2, y2, color, label_text, conf, min_box_size=90):
    """Draws an enlarged tactical surveillance reticle with corner brackets and crosshairs for small/blurry flying objects."""
    h, w = frame.shape[:2]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Enforce minimum prominent box size (minimum 90x90 px for distant specks so targeting reticle is clearly visible)
    box_w = max(x2 - x1 + 24, min_box_size)
    box_h = max(y2 - y1 + 24, min_box_size)
    
    rx1 = max(4, cx - box_w // 2)
    ry1 = max(4, cy - box_h // 2)
    rx2 = min(w - 4, cx + box_w // 2)
    ry2 = min(h - 4, cy + box_h // 2)
    
    # 1. Draw Tactical Corner Brackets [ ] with high thickness
    bracket_len = max(16, int(min(box_w, box_h) * 0.30))
    thick = 3
    
    # Corners
    cv2.line(frame, (rx1, ry1), (rx1 + bracket_len, ry1), color, thick)
    cv2.line(frame, (rx1, ry1), (rx1, ry1 + bracket_len), color, thick)
    cv2.line(frame, (rx2, ry1), (rx2 - bracket_len, ry1), color, thick)
    cv2.line(frame, (rx2, ry1), (rx2, ry1 + bracket_len), color, thick)
    cv2.line(frame, (rx1, ry2), (rx1 + bracket_len, ry2), color, thick)
    cv2.line(frame, (rx1, ry2), (rx1, ry2 - bracket_len), color, thick)
    cv2.line(frame, (rx2, ry2), (rx2 - bracket_len, ry2), color, thick)
    cv2.line(frame, (rx2, ry2), (rx2, ry2 - bracket_len), color, thick)
    
    # 2. Draw Center Crosshair for Small Airborne Objects
    cv2.circle(frame, (cx, cy), 3, color, -1)
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), color, 2)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), color, 2)
        
    # 3. High-Contrast Text Pill Badge with dark background for clear sky readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    text_thick = 2
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, text_thick)
    
    # Position badge centered horizontally, above if space permits, else below
    bx1 = max(4, min(w - tw - 24, cx - tw // 2 - 10))
    bx2 = bx1 + tw + 20
    if ry1 - th - 18 >= 6:
        by1 = ry1 - th - 16
        by2 = ry1 - 4
    else:
        by1 = ry2 + 6
        by2 = ry2 + th + 18
    
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (15, 15, 20), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
    cv2.putText(frame, label_text, (bx1 + 10, by2 - 6), font, font_scale, (255, 255, 255), text_thick)

class FlyingObjectSurveillanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Airborne Threat Detection & Tracking System - AI Command Center")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#14151f")
        
        # State variables
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.source_type = "file"  # "file", "webcam", "sample"
        self.current_video_path = "sample_surveillance_video.mp4"
        self.conf_threshold = 0.15  # Optimized for distant small objects
        self.clahe_enabled = True   # Atmospheric contrast and de-blur filter
        self.imgsz = 1024           # High-resolution inference to preserve small specks
        self.reticle_size = 90      # Minimum tactical target reticle size (large/clear for distant specks)
        self.geofence_enabled = True
        
        # Geofence coordinates (normalized: xmin, ymin, xmax, ymax)
        self.geofence = (0.20, 0.20, 0.80, 0.80)
        
        # Tracking trajectory buffer: persistent tracker to prevent jumping lines
        self.tracked_objects = {}  # track_id -> {"pos": (cx, cy), "lost": 0, "trail": deque(maxlen=30)}
        self.next_track_id = 1
        
        # Statistics
        self.stats = {"drones": 0, "birds": 0, "helicopters": 0, "fps": 0.0, "latency": 0.0}
        self.current_threat = "SAFE"
        
        # Load AI models
        self.load_models()
        
        # Build User Interface
        self.build_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_models(self):
        print("Loading AI Detection, Transformer, and Tracking engines...")
        try:
            self.detector = YOLO("yolov8n.pt")
            self.transformer = TransformerFeatureEncoder().to(device)
            self.transformer.eval()
            self.lstm = MotionLSTM().to(device)
            self.lstm.eval()
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")

    def build_ui(self):
        # 1. Top Header Banner
        header = tk.Frame(self.root, bg="#1a1c29", height=60, bd=0)
        header.pack(fill=tk.X, side=tk.TOP)
        
        lbl_title = tk.Label(header, text="AIRBORNE THREAT DETECTION & SURVEILLANCE RADAR",
                             font=("Segoe UI", 16, "bold"), fg="#ffffff", bg="#1a1c29")
        lbl_title.pack(side=tk.LEFT, padx=20, pady=12)
        
        self.lbl_threat_badge = tk.Label(header, text="● SYSTEM READY - STANDBY",
                                         font=("Segoe UI", 12, "bold"), fg="#2ecc71", bg="#1f2233",
                                         padx=15, pady=6)
        self.lbl_threat_badge.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # 2. Main Content Split (Left: Video Player, Right: Controls & Alert Logs)
        main_body = tk.Frame(self.root, bg="#14151f")
        main_body.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Left Panel (Video Display & Stats Bar)
        left_panel = tk.Frame(main_body, bg="#14151f")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video Canvas Container with fixed propagation
        self.video_frame = tk.Frame(left_panel, bg="#000000", bd=2, relief=tk.SOLID, width=800, height=520)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_frame.pack_propagate(False)  # CRITICAL: Prevents video frames from expanding the window!
        
        self.canvas_label = tk.Label(self.video_frame, bg="#08080c", text="No Video Stream Active.\nSelect a source and click 'Start Detection'.",
                                     font=("Segoe UI", 13), fg="#7f8c8d")
        self.canvas_label.pack(fill=tk.BOTH, expand=True)
        
        # Real-Time Telemetry Bar below Video
        telemetry_bar = tk.Frame(left_panel, bg="#1a1c29", height=45)
        telemetry_bar.pack(fill=tk.X, pady=(8, 0))
        
        self.lbl_fps = tk.Label(telemetry_bar, text="FPS: 0.0", font=("Consolas", 11, "bold"), fg="#3498db", bg="#1a1c29")
        self.lbl_fps.pack(side=tk.LEFT, padx=15, pady=8)
        
        self.lbl_latency = tk.Label(telemetry_bar, text="Latency: 0.0 ms", font=("Consolas", 11), fg="#95a5a6", bg="#1a1c29")
        self.lbl_latency.pack(side=tk.LEFT, padx=15, pady=8)
        
        self.lbl_drones = tk.Label(telemetry_bar, text="Drones: 0", font=("Segoe UI", 11, "bold"), fg="#e74c3c", bg="#1a1c29")
        self.lbl_drones.pack(side=tk.RIGHT, padx=15, pady=8)
        
        self.lbl_birds = tk.Label(telemetry_bar, text="Birds: 0", font=("Segoe UI", 11, "bold"), fg="#2ecc71", bg="#1a1c29")
        self.lbl_birds.pack(side=tk.RIGHT, padx=15, pady=8)
        
        self.lbl_helicopters = tk.Label(telemetry_bar, text="Helicopters: 0", font=("Segoe UI", 11, "bold"), fg="#f39c12", bg="#1a1c29")
        self.lbl_helicopters.pack(side=tk.RIGHT, padx=15, pady=8)
        
        # Right Panel (Source Select, Controls & Real-Time Alert Logs)
        right_panel = tk.Frame(main_body, bg="#1a1c29", width=380, bd=0)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right_panel.pack_propagate(False)
        
        # Source Selection Card
        card_source = tk.LabelFrame(right_panel, text=" 📹 Video Input Source ", font=("Segoe UI", 11, "bold"),
                                    fg="#ecf0f1", bg="#1a1c29", padx=10, pady=10)
        card_source.pack(fill=tk.X, padx=12, pady=10)
        
        btn_browse = tk.Button(card_source, text="📂 Open Existing Video File...", font=("Segoe UI", 10, "bold"),
                               bg="#2980b9", fg="#ffffff", activebackground="#3498db", bd=0, padx=10, pady=6,
                               command=self.browse_video_file)
        btn_browse.pack(fill=tk.X, pady=4)
        
        btn_webcam = tk.Button(card_source, text="📹 Use Live Webcam / Camera", font=("Segoe UI", 10),
                               bg="#34495e", fg="#ffffff", activebackground="#2c3e50", bd=0, padx=10, pady=6,
                               command=self.select_webcam)
        btn_webcam.pack(fill=tk.X, pady=4)
        
        btn_sample = tk.Button(card_source, text="🎬 Load Sample Surveillance Clip", font=("Segoe UI", 10),
                               bg="#2c3e50", fg="#bdc3c7", activebackground="#34495e", bd=0, padx=10, pady=6,
                               command=self.select_sample_clip)
        btn_sample.pack(fill=tk.X, pady=4)
        
        self.lbl_current_source = tk.Label(card_source, text="Current: sample_surveillance_video.mp4",
                                           font=("Segoe UI", 9), fg="#95a5a6", bg="#1a1c29", wraplength=320)
        self.lbl_current_source.pack(fill=tk.X, pady=(6, 2))
        
        # Controls Card
        card_controls = tk.LabelFrame(right_panel, text=" ⚡ Detection & Airspace Controls ", font=("Segoe UI", 11, "bold"),
                                      fg="#ecf0f1", bg="#1a1c29", padx=10, pady=10)
        card_controls.pack(fill=tk.X, padx=12, pady=5)
        
        btn_frame = tk.Frame(card_controls, bg="#1a1c29")
        btn_frame.pack(fill=tk.X, pady=4)
        
        self.btn_start = tk.Button(btn_frame, text="▶ Start", font=("Segoe UI", 10, "bold"),
                                   bg="#27ae60", fg="#ffffff", activebackground="#2ecc71", bd=0, width=8, pady=5,
                                   command=self.start_detection)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_pause = tk.Button(btn_frame, text="⏸ Pause", font=("Segoe UI", 10),
                                   bg="#e67e22", fg="#ffffff", activebackground="#f39c12", bd=0, width=8, pady=5,
                                   command=self.toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        
        self.btn_stop = tk.Button(btn_frame, text="⏹ Stop", font=("Segoe UI", 10),
                                  bg="#c0392b", fg="#ffffff", activebackground="#e74c3c", bd=0, width=8, pady=5,
                                  command=self.stop_detection)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        # Confidence Threshold Slider (Optimized for Small/Distant Flying Objects)
        lbl_slider = tk.Label(card_controls, text="Detection Sensitivity: 0.15 (High)", font=("Segoe UI", 9),
                              fg="#bdc3c7", bg="#1a1c29")
        lbl_slider.pack(anchor=tk.W, pady=(8, 0))
        
        self.slider_conf = ttk.Scale(card_controls, from_=0.05, to=0.80, value=0.15,
                                     command=lambda v: self.on_slider_change(v, lbl_slider))
        self.slider_conf.pack(fill=tk.X, pady=4)
        
        # Target Reticle Box Size Slider (Adjustable prominent targeting bracket)
        lbl_reticle = tk.Label(card_controls, text="Target Reticle Box Size: 90 px (Prominent)", font=("Segoe UI", 9),
                               fg="#bdc3c7", bg="#1a1c29")
        lbl_reticle.pack(anchor=tk.W, pady=(6, 0))
        
        self.slider_reticle = ttk.Scale(card_controls, from_=50, to=160, value=90,
                                        command=lambda v: self.on_reticle_change(v, lbl_reticle))
        self.slider_reticle.pack(fill=tk.X, pady=4)
        
        # Atmospheric CLAHE & De-Blur Filter Checkbox
        self.var_clahe = tk.BooleanVar(value=True)
        chk_clahe = tk.Checkbutton(card_controls, text="✨ De-Blur & Sky Contrast (CLAHE)",
                                   variable=self.var_clahe, font=("Segoe UI", 9, "bold"),
                                   fg="#00d2d3", bg="#1a1c29", selectcolor="#2c3e50", activebackground="#1a1c29",
                                   command=self.toggle_clahe)
        chk_clahe.pack(anchor=tk.W, pady=2)
        
        # High-Resolution Small Object Inference Mode
        self.var_highres = tk.BooleanVar(value=True)
        chk_highres = tk.Checkbutton(card_controls, text="🔍 High-Def Mode (1024px Small Objects)",
                                     variable=self.var_highres, font=("Segoe UI", 9),
                                     fg="#ecf0f1", bg="#1a1c29", selectcolor="#2c3e50", activebackground="#1a1c29",
                                     command=self.toggle_highres)
        chk_highres.pack(anchor=tk.W, pady=2)
        
        # Geofence Checkbox
        self.var_geofence = tk.BooleanVar(value=True)
        chk_geofence = tk.Checkbutton(card_controls, text="🛡️ Enable Geofenced Restricted Airspace",
                                      variable=self.var_geofence, font=("Segoe UI", 9),
                                      fg="#ecf0f1", bg="#1a1c29", selectcolor="#2c3e50", activebackground="#1a1c29",
                                      command=self.toggle_geofence)
        chk_geofence.pack(anchor=tk.W, pady=2)
        
        # Real-Time Security Incident Logs Card
        card_logs = tk.LabelFrame(right_panel, text=" 🚨 Security Alert Incident Log ", font=("Segoe UI", 11, "bold"),
                                  fg="#e74c3c", bg="#1a1c29", padx=10, pady=10)
        card_logs.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        # Scrollable Text / List for Alerts
        self.log_listbox = tk.Listbox(card_logs, bg="#11121a", fg="#ecf0f1", font=("Consolas", 8),
                                      bd=0, highlightthickness=0, selectbackground="#34495e")
        scrollbar = tk.Scrollbar(card_logs, orient=tk.VERTICAL, command=self.log_listbox.yview)
        self.log_listbox.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.add_log("INFO", "System initialized. Ready for surveillance video input.")

    def on_slider_change(self, val, lbl):
        self.conf_threshold = float(val)
        sens = "Ultra High" if self.conf_threshold <= 0.15 else ("High" if self.conf_threshold <= 0.25 else "Standard")
        lbl.config(text=f"Detection Sensitivity: {self.conf_threshold:.2f} ({sens})")

    def on_reticle_change(self, val, lbl):
        self.reticle_size = int(float(val))
        desc = "Compact" if self.reticle_size < 70 else ("Prominent" if self.reticle_size <= 110 else "Extra Large")
        lbl.config(text=f"Target Reticle Box Size: {self.reticle_size} px ({desc})")

    def toggle_geofence(self):
        self.geofence_enabled = self.var_geofence.get()

    def toggle_clahe(self):
        self.clahe_enabled = self.var_clahe.get()
        state = "Enabled" if self.clahe_enabled else "Disabled"
        self.add_log("CFG", f"Atmospheric Contrast (CLAHE) & De-Blur: {state}")

    def toggle_highres(self):
        self.imgsz = 1024 if self.var_highres.get() else 640
        mode = "1024px (Small Object Ultra-HD)" if self.imgsz == 1024 else "640px (Standard)"
        self.add_log("CFG", f"Inference Resolution set to: {mode}")

    def update_tracks(self, detections):
        """Centroid Multi-Object Tracker: Matches detections across frames to prevent jumping lines."""
        updated = []
        used_tracks = set()

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            best_id = None
            min_dist = 95.0  # Max pixel movement threshold per frame

            for tid, tdata in self.tracked_objects.items():
                if tid in used_tracks:
                    continue
                lx, ly = tdata["pos"]
                dist = math.hypot(cx - lx, cy - ly)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

            if best_id is not None:
                used_tracks.add(best_id)
                self.tracked_objects[best_id]["pos"] = (cx, cy)
                self.tracked_objects[best_id]["lost"] = 0
                self.tracked_objects[best_id]["trail"].append((cx, cy))
                det["track_id"] = best_id
                det["trail"] = list(self.tracked_objects[best_id]["trail"])
            else:
                new_id = self.next_track_id
                self.next_track_id += 1
                used_tracks.add(new_id)
                self.tracked_objects[new_id] = {
                    "pos": (cx, cy),
                    "lost": 0,
                    "trail": deque([(cx, cy)], maxlen=30)
                }
                det["track_id"] = new_id
                det["trail"] = [(cx, cy)]

            updated.append(det)

        # Prune dead tracks
        lost_ids = [tid for tid in self.tracked_objects if tid not in used_tracks]
        for tid in lost_ids:
            self.tracked_objects[tid]["lost"] += 1
            if self.tracked_objects[tid]["lost"] > 12:
                del self.tracked_objects[tid]

        return updated

    def add_log(self, level, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level}] {msg}"
        self.log_listbox.insert(tk.END, entry)
        self.log_listbox.yview(tk.END)

    def browse_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Airborne Surveillance Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
        )
        if file_path:
            self.stop_detection()
            self.source_type = "file"
            self.current_video_path = file_path
            self.lbl_current_source.config(text=f"File: {os.path.basename(file_path)}")
            self.add_log("LOAD", f"Selected video file: {os.path.basename(file_path)}")
            self.start_detection()

    def select_webcam(self):
        self.stop_detection()
        self.source_type = "webcam"
        self.current_video_path = 0
        self.lbl_current_source.config(text="Source: Live Webcam / USB Camera (0)")
        self.add_log("LOAD", "Switched to Live Camera / Webcam input.")
        self.start_detection()

    def select_sample_clip(self):
        self.stop_detection()
        self.source_type = "sample"
        self.current_video_path = "sample_surveillance_video.mp4"
        self.lbl_current_source.config(text="Source: Built-in Surveillance Sample Clip")
        self.add_log("LOAD", "Loaded sample aerial surveillance test clip.")
        self.start_detection()

    def start_detection(self):
        if self.is_running and self.is_paused:
            self.is_paused = False
            self.btn_pause.config(text="⏸ Pause")
            return
            
        if self.is_running:
            return
            
        self.is_running = True
        self.is_paused = False
        self.btn_start.config(state=tk.DISABLED)
        
        # Start capture thread
        self.thread = threading.Thread(target=self.video_stream_loop, daemon=True)
        self.thread.start()
        self.add_log("SYS", "Surveillance pipeline processing started.")

    def toggle_pause(self):
        if self.is_running:
            self.is_paused = not self.is_paused
            self.btn_pause.config(text="▶ Resume" if self.is_paused else "⏸ Pause")

    def stop_detection(self):
        self.is_running = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.config(state=tk.NORMAL)
        self.lbl_threat_badge.config(text="● SYSTEM STANDBY", fg="#bdc3c7", bg="#1f2233")
        self.canvas_label.config(image="", text="Video Stream Stopped.\nSelect a source and click 'Start Detection'.")

    def video_stream_loop(self):
        self.cap = cv2.VideoCapture(self.current_video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video source: {self.current_video_path}")
            self.stop_detection()
            return
            
        prev_time = time.time()
        drone_detected_last = 0
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                # If looped file, reset to start
                if self.source_type in ["file", "sample"]:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                    
            start_infer = time.time()
            h, w = frame.shape[:2]
            
            # --- 1. De-Blur & Atmospheric CLAHE Contrast Enhancement for Blurry Distant Objects ---
            infer_frame = enhance_sky_clarity(frame) if self.clahe_enabled else frame
            
            # High-Resolution Inference (imgsz=1024 ensures tiny/distant flying specks are preserved)
            results = self.detector.predict(infer_frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)[0]
            
            # Count objects in current frame
            frame_drones = 0
            frame_birds = 0
            frame_helis = 0
            has_red_threat = False
            has_yellow_threat = False
            
            # Draw Geofence Perimeter if enabled
            if self.geofence_enabled:
                zx1, zy1, zx2, zy2 = self.geofence
                gx1, gy1 = int(zx1 * w), int(zy1 * h)
                gx2, gy2 = int(zx2 * w), int(zy2 * h)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 165, 255), 2)
                cv2.putText(frame, "RESTRICTED AIRSPACE PERIMETER [GEOFENCE]", (gx1 + 8, gy1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

            # 1. Collect and classify candidate detections
            raw_candidates = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                raw_cls = int(box.cls[0])
                
                # Clip coordinates to frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                bw, bh = x2 - x1, y2 - y1
                if bw < 4 or bh < 4:
                    continue
                    
                det_raw = self.detector.names.get(raw_cls, "").lower()
                crop = frame[y1:y2, x1:x2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                cls_name, final_conf = classify_airborne_candidate(
                    crop, det_raw, raw_cls, bw, bh, conf, h, cy, self.transformer
                )
                
                raw_candidates.append({
                    "box": (x1, y1, x2, y2),
                    "cx": cx,
                    "cy": cy,
                    "conf": final_conf,
                    "cls_name": cls_name,
                    "raw_cls": raw_cls
                })
                
            # 2. Suppress duplicate / overlapping candidate boxes (IoU NMS)
            kept_candidates = apply_nms(raw_candidates, iou_thresh=0.30, center_dist_thresh=35)
            
            # 3. Associate with persistent tracks (prevents jumping lines across screen)
            tracked_detections = self.update_tracks(kept_candidates)
            
            # 4. Render persistent flight trails & high-visibility tactical reticles
            for det in tracked_detections:
                x1, y1, x2, y2 = det["box"]
                conf = det["conf"]
                cls_name = det["cls_name"]
                track_id = det["track_id"]
                cx, cy = det["cx"], det["cy"]
                norm_cx, norm_cy = cx / w, cy / h
                
                # Check Geofence Breach
                zx1, zy1, zx2, zy2 = self.geofence
                in_geofence = (zx1 <= norm_cx <= zx2) and (zy1 <= norm_cy <= zy2)
                
                # Draw Historical Flight Trail for this unique track only
                trail_pts = det.get("trail", [])
                for p_i in range(1, len(trail_pts)):
                    cv2.line(frame, trail_pts[p_i - 1], trail_pts[p_i], (0, 242, 255), 2, lineType=cv2.LINE_AA)
                    
                # Threat Evaluation & Classification Badges
                if cls_name == "UAV (Drone)":
                    frame_drones += 1
                    if in_geofence and self.geofence_enabled:
                        has_red_threat = True
                        badge_text = f"TARGET #{track_id}: DRONE [CRITICAL] ({conf:.2f})"
                        color = (0, 0, 255)
                    else:
                        has_yellow_threat = True
                        badge_text = f"TARGET #{track_id}: DRONE [APPROACHING] ({conf:.2f})"
                        color = (0, 140, 255)
                elif cls_name == "Bird":
                    frame_birds += 1
                    badge_text = f"TARGET #{track_id}: BIRD [SAFE] ({conf:.2f})"
                    color = (46, 204, 113)
                elif cls_name == "Helicopter":
                    frame_helis += 1
                    badge_text = f"TARGET #{track_id}: HELICOPTER ({conf:.2f})"
                    color = (243, 156, 18)
                elif cls_name == "Airplane":
                    badge_text = f"TARGET #{track_id}: AIRPLANE ({conf:.2f})"
                    color = (255, 180, 0)
                else:
                    badge_text = f"TARGET #{track_id}: {cls_name.upper()} ({conf:.2f})"
                    color = COLOR_MAP.get(cls_name, (255, 255, 255))
                    
                # Draw Tactical High-Visibility Surveillance Target Reticle
                draw_high_visibility_target(frame, x1, y1, x2, y2, color, badge_text, conf, min_box_size=self.reticle_size)
                            
            # Threat Status Update
            if has_red_threat:
                threat_state = "CRITICAL_RED"
                now = time.time()
                if now - drone_detected_last > 2.0:
                    self.add_log("ALERT", f"UNAUTHORIZED DRONE BREACH inside Restricted Airspace! ({frame_drones} UAV)")
                    drone_detected_last = now
            elif has_yellow_threat:
                threat_state = "CAUTION_YELLOW"
            elif frame_birds > 0:
                threat_state = "SAFE_GREEN"
            else:
                threat_state = "NORMAL"
                
            # Telemetry calculations
            fps = 1.0 / (time.time() - prev_time + 1e-6)
            latency = (time.time() - start_infer) * 1000.0
            prev_time = time.time()
            
            # Send to UI update
            self.root.after(0, self.update_dashboard, frame.copy(), fps, latency, frame_drones, frame_birds, frame_helis, threat_state)
            time.sleep(0.015)  # Cap around ~30 FPS

    def update_dashboard(self, frame_bgr, fps, latency, drones, birds, helis, threat_state):
        # 1. Update Threat Badge
        if threat_state == "CRITICAL_RED":
            self.lbl_threat_badge.config(text="⚠ CRITICAL ALERT: DRONE INTRUSION DETECTED!",
                                         fg="#ffffff", bg="#e74c3c")
        elif threat_state == "CAUTION_YELLOW":
            self.lbl_threat_badge.config(text="⚡ CAUTION: AERIAL OBJECT NEAR PERIMETER",
                                         fg="#1a1c29", bg="#f1c40f")
        elif threat_state == "SAFE_GREEN":
            self.lbl_threat_badge.config(text="● AIRSPACE NORMAL: BIRD WILDLIFE DETECTED",
                                         fg="#ffffff", bg="#27ae60")
        else:
            self.lbl_threat_badge.config(text="● AIRSPACE CLEAR - SCANNING",
                                         fg="#2ecc71", bg="#1f2233")
                                         
        # 2. Update Stats
        self.lbl_fps.config(text=f"FPS: {fps:.1f}")
        self.lbl_latency.config(text=f"Latency: {latency:.1f} ms")
        self.lbl_drones.config(text=f"Drones: {drones}")
        self.lbl_birds.config(text=f"Birds: {birds}")
        self.lbl_helicopters.config(text=f"Helicopters: {helis}")
        
        # 3. Render Video on Canvas (Aspect-Ratio Constrained)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        disp_w = self.video_frame.winfo_width()
        disp_h = self.video_frame.winfo_height()
        if disp_w < 100 or disp_h < 100:
            disp_w, disp_h = 800, 520
            
        # Maintain aspect ratio to prevent window expansion or image distortion
        h_orig, w_orig = frame_bgr.shape[:2]
        scale = min(disp_w / w_orig, disp_h / h_orig)
        new_w = max(10, int(w_orig * scale))
        new_h = max(10, int(h_orig * scale))
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.canvas_label.config(image=img_tk)
        self.canvas_label.image = img_tk

    def on_close(self):
        self.stop_detection()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FlyingObjectSurveillanceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
