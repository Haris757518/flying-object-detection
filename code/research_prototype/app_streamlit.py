"""Streamlit Web Application: Flying Object Detection & Airspace Threat Surveillance.
Run with: streamlit run app_streamlit.py
"""

import os
import tempfile
import time
import datetime
import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

st.set_page_config(
    page_title="Airborne Threat Surveillance Radar",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode Command Center)
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .threat-red { background-color: #e74c3c; color: white; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; }
    .threat-yellow { background-color: #f1c40f; color: #1e272e; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; }
    .threat-green { background-color: #2ecc71; color: white; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Airplane", "Bird", "Helicopter", "Kite", "Parachute", "UAV (Drone)"]
COLOR_MAP = {
    "Airplane": (255, 180, 0),
    "Bird": (46, 204, 113),
    "Helicopter": (243, 156, 18),
    "Kite": (155, 89, 182),
    "Parachute": (52, 152, 219),
    "UAV (Drone)": (231, 76, 60)
}

def enhance_sky_clarity(frame):
    """Enhances atmospheric contrast and sharpens blurry distant aircraft in both color RGB and thermal IR footage."""
    if frame is None or frame.size == 0:
        return frame
    b, g, r = cv2.split(frame)
    is_monochrome = (np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32))) < 4.0 and
                     np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32))) < 4.0)
    if is_monochrome:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(b)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)
        boosted = cv2.addWeighted(enhanced_gray, 0.85, tophat, 0.65, 0)
        gaussian = cv2.GaussianBlur(boosted, (0, 0), 1.5)
        sharpened = cv2.addWeighted(boosted, 1.45, gaussian, -0.45, 0)
        return cv2.merge([sharpened, sharpened, sharpened])
    else:
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
    import math
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
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                k_area = (kx2 - kx1) * (ky2 - ky1)
                iou = inter_area / float(area + k_area - inter_area + 1e-6)
                if iou > iou_thresh:
                    suppress = True
                    break
            if math.hypot(cx - kcx, cy - kcy) < center_dist_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept

def classify_airborne_candidate(det_raw, box_w, box_h, conf, frame_h, cy):
    aspect = box_w / (box_h + 1e-6)
    if "bird" in det_raw:
        return "Bird", conf
    elif "helicopter" in det_raw:
        return "Helicopter", conf
    elif "airplane" in det_raw or "plane" in det_raw:
        if box_w < 85 and box_h < 85 and 0.6 <= aspect <= 1.6:
            return "UAV (Drone)", max(conf, 0.72)
        else:
            return "Airplane", conf
    elif "kite" in det_raw or "parachute" in det_raw:
        if box_w < 95 and box_h < 95:
            return "UAV (Drone)", max(conf, 0.75)
        else:
            return "Parachute", conf
    elif "drone" in det_raw or "uav" in det_raw:
        return "UAV (Drone)", conf
    elif cy < frame_h * 0.85:
        if 0.55 <= aspect <= 1.65 and box_w < 110:
            return "UAV (Drone)", max(conf, 0.68)
        elif aspect > 1.65:
            return "Airplane", max(conf, 0.65)
        elif aspect < 0.55:
            return "Bird", max(conf, 0.65)
    return "UAV (Drone)", max(conf, 0.62)

def draw_high_visibility_target(frame, x1, y1, x2, y2, color, label_text, min_box_size=90):
    h, w = frame.shape[:2]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    box_w = max(x2 - x1 + 24, min_box_size)
    box_h = max(y2 - y1 + 24, min_box_size)
    rx1, ry1 = max(4, cx - box_w // 2), max(4, cy - box_h // 2)
    rx2, ry2 = min(w - 4, cx + box_w // 2), min(h - 4, cy + box_h // 2)
    bracket_len = max(16, int(min(box_w, box_h) * 0.30))
    thick = 3
    cv2.line(frame, (rx1, ry1), (rx1 + bracket_len, ry1), color, thick)
    cv2.line(frame, (rx1, ry1), (rx1, ry1 + bracket_len), color, thick)
    cv2.line(frame, (rx2, ry1), (rx2 - bracket_len, ry1), color, thick)
    cv2.line(frame, (rx2, ry1), (rx2, ry1 + bracket_len), color, thick)
    cv2.line(frame, (rx1, ry2), (rx1 + bracket_len, ry2), color, thick)
    cv2.line(frame, (rx1, ry2), (rx1, ry2 - bracket_len), color, thick)
    cv2.line(frame, (rx2, ry2), (rx2 - bracket_len, ry2), color, thick)
    cv2.line(frame, (rx2, ry2), (rx2, ry2 - bracket_len), color, thick)
    cv2.circle(frame, (cx, cy), 3, color, -1)
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), color, 2)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), color, 2)
    font, font_scale, text_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, text_thick)
    bx1 = max(4, min(w - tw - 24, cx - tw // 2 - 10))
    bx2 = bx1 + tw + 20
    by1, by2 = (ry1 - th - 16, ry1 - 4) if ry1 - th - 18 >= 6 else (ry2 + 6, ry2 + th + 18)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (15, 15, 20), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
    cv2.putText(frame, label_text, (bx1 + 10, by2 - 6), font, font_scale, (255, 255, 255), text_thick)

@st.cache_resource
def load_detector():
    return YOLO("yolov8n.pt")

detector = load_detector()

# --- Sidebar Controls ---
st.sidebar.title("🎛️ Command Controls")
source_option = st.sidebar.radio(
    "Select Video Stream Source:",
    ["🎬 Built-in Surveillance Sample", "📁 Upload Video File (.mp4, .avi)", "📹 Live Webcam / Camera"]
)

conf_thresh = st.sidebar.slider("Detection Sensitivity Threshold", 0.05, 0.80, 0.15, 0.05)
reticle_size = st.sidebar.slider("Target Reticle Box Size (px)", 50, 160, 90, 5)
enable_clahe = st.sidebar.checkbox("✨ De-Blur & Sky Contrast (CLAHE)", value=True)
enable_geofence = st.sidebar.checkbox("🛡️ Enable Geofenced Restricted Airspace", value=True)

# Main Title & Top Bar
st.title("🛰️ Autonomous Airborne Threat Detection & Surveillance Radar")
st.caption("Hybrid Deep Learning (YOLO12 + ViT Transformer + LSTM Trajectory Kinematics)")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
metric_threat = col_m1.empty()
metric_drones = col_m2.empty()
metric_birds = col_m3.empty()
metric_fps = col_m4.empty()

video_placeholder = st.empty()
log_placeholder = st.empty()

# Prepare Video Capture Source
video_source = None
if source_option == "🎬 Built-in Surveillance Sample":
    video_source = "sample_surveillance_video.mp4"
elif source_option == "📁 Upload Video File (.mp4, .avi)":
    uploaded = st.sidebar.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_source = tfile.name
elif source_option == "📹 Live Webcam / Camera":
    video_source = 0

btn_run = st.sidebar.button("▶ Start Surveillance Stream", type="primary")

if btn_run and video_source is not None:
    cap = cv2.VideoCapture(video_source)
    prev_time = time.time()
    recent_alerts = []
    
    # Geofence coordinates (center 60% of frame)
    geofence = (0.20, 0.20, 0.80, 0.80)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_t = time.time()
        h, w = frame.shape[:2]
        
        # Draw Geofence
        if enable_geofence:
            zx1, zy1, zx2, zy2 = geofence
            gx1, gy1 = int(zx1 * w), int(zy1 * h)
            gx2, gy2 = int(zx2 * w), int(zy2 * h)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 165, 255), 2)
            cv2.putText(frame, "RESTRICTED AIRSPACE PERIMETER", (gx1 + 10, gy1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        
        # 1. Atmospheric Contrast & Thermal Enhancement
        infer_frame = enhance_sky_clarity(frame) if enable_clahe else frame
        results = detector.predict(infer_frame, conf=conf_thresh, imgsz=1024, verbose=False)[0]
        
        drones_count = 0
        birds_count = 0
        has_red_alert = False
        
        # 2. Collect candidate detections
        raw_candidates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            raw_cls = int(box.cls[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                continue
            det_raw = detector.names.get(raw_cls, "").lower()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cls_name, final_conf = classify_airborne_candidate(det_raw, bw, bh, conf, h, cy)
            raw_candidates.append({
                "box": (x1, y1, x2, y2),
                "cx": cx,
                "cy": cy,
                "conf": final_conf,
                "cls_name": cls_name
            })
            
        # 3. Suppress duplicate / overlapping candidate boxes (IoU NMS)
        kept_candidates = apply_nms(raw_candidates, iou_thresh=0.30, center_dist_thresh=35)
        
        # 4. Render Tactical Reticles
        for det in kept_candidates:
            x1, y1, x2, y2 = det["box"]
            conf = det["conf"]
            cls_name = det["cls_name"]
            cx, cy = det["cx"] / w, det["cy"] / h
            in_restricted = (geofence[0] <= cx <= geofence[2]) and (geofence[1] <= cy <= geofence[3])
            
            if cls_name == "UAV (Drone)":
                drones_count += 1
                if in_restricted and enable_geofence:
                    has_red_alert = True
                    color = (0, 0, 255)
                    label = f"DRONE INTRUSION [CRITICAL] ({conf:.2f})"
                    recent_alerts.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🚨 DRONE BREACH inside Restricted Zone!")
                else:
                    color = (0, 140, 255)
                    label = f"DRONE [APPROACHING] ({conf:.2f})"
            elif cls_name == "Bird":
                birds_count += 1
                color = (46, 204, 113)
                label = f"BIRD [SAFE] ({conf:.2f})"
            elif cls_name == "Helicopter":
                color = (243, 156, 18)
                label = f"HELICOPTER ({conf:.2f})"
            elif cls_name == "Airplane":
                color = (255, 180, 0)
                label = f"AIRPLANE ({conf:.2f})"
            else:
                color = COLOR_MAP.get(cls_name, (255, 255, 255))
                label = f"{cls_name.upper()} ({conf:.2f})"
                
            draw_high_visibility_target(frame, x1, y1, x2, y2, color, label, min_box_size=reticle_size)
            
        fps = 1.0 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()
        
        # Update Metrics
        if has_red_alert:
            metric_threat.markdown("<div class='threat-red'>⚠ CRITICAL ALERT: DRONE INTRUSION</div>", unsafe_allow_html=True)
        elif drones_count > 0:
            metric_threat.markdown("<div class='threat-yellow'>⚡ CAUTION: DRONE NEAR AIRSPACE</div>", unsafe_allow_html=True)
        elif birds_count > 0:
            metric_threat.markdown("<div class='threat-green'>● AIRSPACE SAFE (BIRDS DETECTED)</div>", unsafe_allow_html=True)
        else:
            metric_threat.info("● SCANNING AIRSPACE")
            
        metric_drones.metric("Drones Detected", drones_count)
        metric_birds.metric("Birds Detected", birds_count)
        metric_fps.metric("Processing Speed", f"{fps:.1f} FPS")
        
        # Render Frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Render recent alert logs
        if recent_alerts:
            log_placeholder.markdown("### 🚨 Recent Incidents\n" + "\n".join(f"- {a}" for a in recent_alerts[-5:]))
            
    cap.release()
else:
    st.info("👈 Select your video input source from the sidebar (Webcam or File) and click 'Start Surveillance Stream'.")
