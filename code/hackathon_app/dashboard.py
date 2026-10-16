"""
Tactical Air Defense & Drone Detection Command Dashboard.
Interactive Streamlit application featuring YOLO12, GAN Super-Resolution,
ByteTrack Tracking, Trajectory Transformer Behavior Analysis, and 2D Polar Radar.
"""

import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from core.pipeline import DroneDefensePipeline
from core.risk_engine import ThreatLevel, REAL_WORLD_DIAMETERS
from drone_audio_dataset import AUDIO_CLASSES


# Page Configuration
st.set_page_config(
    page_title="Anti-Drone Tactical Defense Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Tactical CSS
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    .stApp { background: radial-gradient(circle at 50% 10%, #151f32 0%, #080c14 100%); }
    .tactical-card {
        background: rgba(18, 26, 43, 0.85);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .metric-val { font-size: 26px; font-weight: 700; color: #38bdf8; font-family: monospace; }
    .threat-low { color: #22c55e; font-weight: 700; }
    .threat-med { color: #eab308; font-weight: 700; }
    .threat-high { color: #f97316; font-weight: 700; }
    .threat-crit { color: #ef4444; font-weight: 700; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0.3; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(conf_thresh, iou_thresh, imgsz):
    return DroneDefensePipeline(conf_thresh=conf_thresh, iou_thresh=iou_thresh, imgsz=imgsz)


def render_radar_plot(targets, no_fly_r=30.0, warning_r=100.0, max_r=200.0):
    """
    Renders top-down 2D polar radar display with geofence zones and active targets.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#0d1322')
    ax.set_facecolor('#080d1a')

    # Draw Geofence Rings
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta, [no_fly_r] * len(theta), color='#ef4444', linestyle='--', linewidth=1.5, label='No-Fly Zone (30m)')
    ax.plot(theta, [warning_r] * len(theta), color='#f59e0b', linestyle=':', linewidth=1.2, label='Warning Zone (100m)')
    ax.set_ylim(0, max_r)

    # Styling polar grid
    ax.grid(True, color='#1e293b', linestyle='-', alpha=0.6)
    ax.tick_params(colors='#64748b', labelsize=8)
    ax.set_yticklabels([])

    # Plot Radar Center (Defense Asset / Camera)
    ax.scatter(0, 0, color='#38bdf8', s=120, marker='P', zorder=5, label='Defense Asset')

    # Plot Active Targets
    for t in targets:
        dist = min(max_r - 5, t["distance_m"])
        heading_rad = np.radians(t.get("heading", 45.0))
        score = t["threat_score"]

        if score >= 75:
            dot_color = '#ef4444'
        elif score >= 50:
            dot_color = '#f97316'
        elif score >= 25:
            dot_color = '#eab308'
        else:
            dot_color = '#22c55e'

        ax.scatter(heading_rad, dist, color=dot_color, s=150, edgecolor='#ffffff', linewidth=1.5, zorder=6)
        ax.text(heading_rad, dist + 12, f"ID:{t['track_id']}\n{t['class_name']}", color='#e2e8f0',
                fontsize=8, fontweight='bold', ha='center', zorder=7)

    plt.tight_layout()
    return fig


def main():
    # Sidebar: Control Panel
    st.sidebar.markdown("### 🎛️ DEFENSE CONTROL PANEL")

    st.sidebar.markdown("""
    <div style="background: rgba(18, 26, 43, 0.85); border: 1px solid #1e293b; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
        <span style="color: #64748b; font-size: 12px;">ACTIVE MODEL</span><br>
        <span style="color: #38bdf8; font-weight: bold; font-family: monospace;">YOLOv8s AOD-4</span><br>
        <span style="color: #64748b; font-size: 12px;">CLASSES</span><br>
        <span style="color: #e2e8f0; font-size: 14px;">4 (AIRPLANE, BIRD, DRONE, HELICOPTER)</span>
    </div>
    """, unsafe_allow_html=True)

    source_type = st.sidebar.selectbox(
        "Input Source",
        ["Dataset Video (IR)", "Dataset Video (Visible)", "Upload Video File", "Live Webcam", "4-Class Demo"]
    )

    video_path = None
    sensor_type = "IR"
    BASE_DIR = Path(__file__).resolve().parent

    if source_type == "Dataset Video (IR)":
        ir_files = sorted(glob.glob(str(BASE_DIR / "Data" / "Video_IR" / "*.mp4")))
        root_ir = sorted(glob.glob(str(BASE_DIR / "Demo_IR_*.mp4")))
        all_ir = root_ir + ir_files
        if all_ir:
            video_name = st.sidebar.selectbox("Select IR Video", [os.path.basename(f) for f in all_ir[:50]])
            video_path = next(f for f in all_ir if os.path.basename(f) == video_name)
            sensor_type = "IR"
        else:
            st.sidebar.warning(f"No video found. Searched:\n{BASE_DIR / 'Demo_IR_*.mp4'}\n{BASE_DIR / 'Data' / 'Video_IR' / '*.mp4'}")
    elif source_type == "Dataset Video (Visible)":
        v_files = sorted(glob.glob(str(BASE_DIR / "Data" / "Video_V" / "*.mp4")))
        root_v = sorted(glob.glob(str(BASE_DIR / "Demo_V_*.mp4")))
        all_v = root_v + v_files
        if all_v:
            video_name = st.sidebar.selectbox("Select Visible Video", [os.path.basename(f) for f in all_v[:50]])
            video_path = next(f for f in all_v if os.path.basename(f) == video_name)
            sensor_type = "V"
        else:
            st.sidebar.warning(f"No video found. Searched:\n{BASE_DIR / 'Demo_V_*.mp4'}\n{BASE_DIR / 'Data' / 'Video_V' / '*.mp4'}")
    elif source_type == "Upload Video File":
        uploaded = st.sidebar.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])
        if uploaded:
            suffix = Path(uploaded.name).suffix.lower() or ".mp4"
            temp_path = str(BASE_DIR / f"temp_uploaded{suffix}")
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            video_path = temp_path
            sensor_type = st.sidebar.radio("Sensor Type", ["Visible", "IR"], index=0)
    elif source_type == "4-Class Demo":
        demo_class = st.sidebar.selectbox("Demo Target", ["Airplane", "Bird", "Drone", "Helicopter"])
        demo_path = BASE_DIR / "Data" / "Demo" / demo_class / f"demo_{demo_class.lower()}.mp4"
        if demo_path.exists():
            video_path = str(demo_path)
            sensor_type = "V"
        else:
            st.sidebar.warning(f"Demo video missing: {demo_path}")
            video_path = None
    else:
        video_path = 0  # Webcam index

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Stream Info")
    
    if source_type == "4-Class Demo":
        st.sidebar.markdown(f"""
        <div style="background: rgba(15, 30, 45, 0.8); border: 1px solid #0ea5e9; padding: 10px; border-radius: 6px; margin-bottom: 10px;">
            <span style="color: #64748b; font-size: 11px;">DEMO TARGET</span><br>
            <span style="color: #e2e8f0; font-weight: bold; font-size: 16px;">{demo_class.upper()}</span><br>
            <span style="color: #64748b; font-size: 11px;">SENSOR</span><br>
            <span style="color: #e2e8f0; font-size: 14px;">VISIBLE</span><br>
            <span style="color: #64748b; font-size: 11px;">MODEL</span><br>
            <span style="color: #e2e8f0; font-size: 14px;">YOLOv8s AOD-4</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.text(f"Selected Input: {source_type}")
    if isinstance(video_path, str):
        st.sidebar.text(f"Video Path: ...{video_path[-30:]}")
    else:
        st.sidebar.text(f"Video Path: {video_path}")
    st.sidebar.text(f"Sensor Type: {sensor_type}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Detection & Tracking Settings")
    default_conf = 0.40 if source_type == "4-Class Demo" else 0.30
    conf_thresh = st.sidebar.slider("Detection Confidence", 0.10, 0.90, default_conf, 0.05)
    iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.20, 0.80, 0.45, 0.05)

    # Keep demo behavior at 640, but allow uploaded videos to use more pixels for small targets.
    size_options = [640, 960, 1280]
    default_size_idx = 0 if source_type == "4-Class Demo" else (1 if source_type == "Upload Video File" else 0)
    inference_imgsz = st.sidebar.select_slider(
        "Inference Resolution",
        options=size_options,
        value=size_options[default_size_idx],
        help="960/1280 can improve small distant-object detection but uses more GPU/CPU."
    )

    # This module currently enhances crops for experimentation only; YOLO remains the classifier.
    enable_gan = st.sidebar.checkbox(
        "Experimental GAN crop enhancement",
        value=False,
        help="Does not replace YOLO classification. Leave off for fastest and most repeatable detection."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Geofence Configuration")
    no_fly_radius = st.sidebar.slider("No-Fly Perimeter (m)", 10, 80, 30, 5)
    warning_radius = st.sidebar.slider("Warning Perimeter (m)", 50, 200, 100, 10)

    # Header
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1e293b; padding-bottom: 10px; margin-bottom: 20px;">
        <div>
            <h1 style="margin: 0; color: #38bdf8; font-size: 28px;">🛡️ ANTI-DRONE TACTICAL DEFENSE COMMAND</h1>
            <p style="margin: 0; color: #94a3b8; font-size: 14px;">Real-Time Multi-Sensor Drone Detection, GAN Enhancement, Trajectory Transformer & Threat Matrix</p>
        </div>
        <div style="text-align: right;">
            <span style="background: #1e293b; color: #38bdf8; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-family: monospace;">SYS: ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pipeline = get_pipeline(conf_thresh, iou_thresh, inference_imgsz)
    pipeline.risk_engine.no_fly_radius_m = float(no_fly_radius)
    pipeline.risk_engine.warning_radius_m = float(warning_radius)

    # Layout: Left column (Video + Telemetry), Right column (Radar + Threat Matrix)
    col_video, col_telemetry = st.columns([1.6, 1.0])

    with col_video:
        st.markdown("### 📹 Live Sensor Stream (Tactical HUD)")
        video_placeholder = st.empty()
        classification_placeholder = st.empty()
        playback_btn = st.button("▶️ Start / Resume Stream", type="primary")

    with col_telemetry:
        st.markdown("### 🎯 Airspace Radar & Threat Matrix")
        radar_placeholder = st.empty()
        threat_card_placeholder = st.empty()

    st.markdown("### 📋 Security Incident Audit Log")
    log_placeholder = st.empty()

    if playback_btn and video_path is not None:
        # Cached Streamlit resources must not leak tracks/classes from the previous video.
        pipeline.reset()
        cap = cv2.VideoCapture(video_path)
        incident_logs = []
        frame_idx = 0
        detected_frames = 0
        video_class_scores = {"AIRPLANE": 0.0, "BIRD": 0.0, "DRONE": 0.0, "HELICOPTER": 0.0}
        video_class_hits = {"AIRPLANE": 0, "BIRD": 0, "DRONE": 0, "HELICOPTER": 0}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated_frame, active_targets, highest_threat = pipeline.process_frame(
                frame,
                sensor_type=sensor_type,
                enhance_crops=enable_gan
            )

            # Video-level classification: use one stable primary target vote per frame.
            # Tracks need at least 3 observations before contributing, which filters one-frame noise.
            stable_targets = [t for t in active_targets if len(t.get("history", [])) >= 3]
            if stable_targets:
                primary = max(
                    stable_targets,
                    key=lambda t: float(t.get("class_confidence", t.get("confidence", 0.0))) * float(t.get("class_vote_share", 1.0))
                )
                cls = str(primary["class_name"]).upper()
                if cls in video_class_scores:
                    class_conf = float(primary.get("class_confidence", primary.get("confidence", 0.0)))
                    vote_share = float(primary.get("class_vote_share", 1.0))
                    video_class_scores[cls] += max(0.01, class_conf * vote_share)
                    video_class_hits[cls] += 1
                    detected_frames += 1

            # Update running video-class summary every few frames.
            if source_type == "Upload Video File" and frame_idx % 5 == 0:
                total_vote = sum(video_class_scores.values())
                if total_vote > 0:
                    dominant = max(video_class_scores, key=video_class_scores.get)
                    share = 100.0 * video_class_scores[dominant] / total_vote
                    classification_placeholder.markdown(
                        f"**Video classification (running): {dominant}**  \n"
                        f"Temporal vote share: **{share:.1f}%** · Stable detected frames: **{detected_frames}**"
                    )
                else:
                    classification_placeholder.info("Analyzing video… waiting for a stable tracked detection.")

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Update 2D Radar
            radar_fig = render_radar_plot(
                active_targets,
                no_fly_r=no_fly_radius,
                warning_r=warning_radius
            )
            radar_placeholder.pyplot(radar_fig)
            plt.close(radar_fig)

            # Update Threat Matrix Telemetry Card
            if highest_threat:
                score = highest_threat["threat_score"]
                badge = highest_threat["threat_badge"]
                track_id = highest_threat["track_id"]
                cls_name = highest_threat["class_name"]
                dist = highest_threat["distance_m"]
                beh = highest_threat["behavior"]
                zone = highest_threat["zone"]

                threat_card_placeholder.markdown(f"""
                <div class="tactical-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 18px; font-weight: bold;">CRITICAL TARGET #ID-{track_id}</span>
                        <span style="font-size: 16px;">{badge}</span>
                    </div>
                    <hr style="border-color: #1e293b; margin: 8px 0;" />
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <span style="color: #64748b; font-size: 12px;">OBJECT CLASS</span>
                            <div class="metric-val">{cls_name}</div>
                        </div>
                        <div>
                            <span style="color: #64748b; font-size: 12px;">THREAT SCORE</span>
                            <div class="metric-val">{score:.1f}/100</div>
                        </div>
                        <div>
                            <span style="color: #64748b; font-size: 12px;">EST. DISTANCE</span>
                            <div class="metric-val">{dist}m</div>
                        </div>
                        <div>
                            <span style="color: #64748b; font-size: 12px;">BEHAVIOR</span>
                            <div class="metric-val" style="font-size: 18px; color: #f59e0b;">{beh}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding: 6px 10px; background: rgba(0,0,0,0.3); border-radius: 6px;">
                        <span style="color: #94a3b8; font-size: 12px;">Zone: <strong>{zone}</strong> | Sensor: <strong>{sensor_type}</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if frame_idx % 15 == 0:
                    incident_logs.append({
                        "Timestamp": time.strftime("%H:%M:%S"),
                        "Track ID": track_id,
                        "Class": cls_name,
                        "Distance (m)": dist,
                        "Behavior": beh,
                        "Zone": zone,
                        "Threat Level": highest_threat["threat_level"],
                        "Threat Score": f"{score:.1f}"
                    })
                    if len(incident_logs) > 10:
                        incident_logs.pop(0)
                    log_placeholder.dataframe(pd.DataFrame(incident_logs), use_container_width=True)

            else:
                threat_card_placeholder.markdown("""
                <div class="tactical-card" style="text-align: center; padding: 30px;">
                    <div style="font-size: 36px;">🟢</div>
                    <div style="font-size: 20px; font-weight: bold; color: #22c55e; margin-top: 8px;">AIRSPACE CLEAR</div>
                    <p style="color: #64748b; font-size: 13px; margin: 4px 0 0 0;">No active threats detected inside defense perimeter.</p>
                </div>
                """, unsafe_allow_html=True)

            time.sleep(0.01)

        cap.release()

        if source_type == "Upload Video File":
            total_vote = sum(video_class_scores.values())
            if total_vote > 0:
                dominant = max(video_class_scores, key=video_class_scores.get)
                share = 100.0 * video_class_scores[dominant] / total_vote
                ranking = sorted(video_class_scores.items(), key=lambda kv: kv[1], reverse=True)
                breakdown = " · ".join(
                    f"{name}: {(100.0 * score / total_vote):.1f}%" for name, score in ranking if score > 0
                )
                classification_placeholder.success(
                    f"Final video result: {dominant} | temporal vote share {share:.1f}% | "
                    f"stable detected frames {detected_frames}\n\n{breakdown}"
                )
            else:
                classification_placeholder.warning(
                    "No stable AIRPLANE/BIRD/DRONE/HELICOPTER detection was found. "
                    "Try lowering confidence slightly or use 960/1280 inference resolution."
                )


if __name__ == "__main__":
    main()
