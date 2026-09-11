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
def get_pipeline(conf_thresh, iou_thresh):
    return DroneDefensePipeline(conf_thresh=conf_thresh, iou_thresh=iou_thresh)


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

    source_type = st.sidebar.selectbox(
        "Input Source",
        ["Dataset Video (IR)", "Dataset Video (Visible)", "Upload Video File", "Live Webcam"]
    )

    video_path = None
    sensor_type = "IR"

    if source_type == "Dataset Video (IR)":
        ir_files = sorted(glob.glob("Data/Video_IR/*.mp4"))
        if ir_files:
            video_name = st.sidebar.selectbox("Select IR Video", [os.path.basename(f) for f in ir_files[:50]])
            video_path = os.path.join("Data/Video_IR", video_name)
            sensor_type = "IR"
    elif source_type == "Dataset Video (Visible)":
        v_files = sorted(glob.glob("Data/Video_V/*.mp4"))
        if v_files:
            video_name = st.sidebar.selectbox("Select Visible Video", [os.path.basename(f) for f in v_files[:50]])
            video_path = os.path.join("Data/Video_V", video_name)
            sensor_type = "V"
    elif source_type == "Upload Video File":
        uploaded = st.sidebar.file_uploader("Upload MP4 / AVI", type=["mp4", "avi", "mov"])
        if uploaded:
            temp_path = os.path.join("temp_uploaded.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())
            video_path = temp_path
            sensor_type = st.sidebar.radio("Sensor Type", ["IR", "Visible"])
    else:
        video_path = 0  # Webcam index

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Detection & Tracking Settings")
    conf_thresh = st.sidebar.slider("Detection Confidence", 0.10, 0.90, 0.25, 0.05)
    iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.20, 0.80, 0.45, 0.05)
    enable_gan = st.sidebar.checkbox("Enable GAN Super-Resolution", value=True)

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

    pipeline = get_pipeline(conf_thresh, iou_thresh)
    pipeline.risk_engine.no_fly_radius_m = float(no_fly_radius)
    pipeline.risk_engine.warning_radius_m = float(warning_radius)

    # Layout: Left column (Video + Telemetry), Right column (Radar + Threat Matrix)
    col_video, col_telemetry = st.columns([1.6, 1.0])

    with col_video:
        st.markdown("### 📹 Live Sensor Stream (Tactical HUD)")
        video_placeholder = st.empty()
        playback_btn = st.button("▶️ Start / Resume Stream", type="primary")

    with col_telemetry:
        st.markdown("### 🎯 Airspace Radar & Threat Matrix")
        radar_placeholder = st.empty()
        threat_card_placeholder = st.empty()

    st.markdown("### 📋 Security Incident Audit Log")
    log_placeholder = st.empty()

    if playback_btn and video_path is not None:
        cap = cv2.VideoCapture(video_path)
        incident_logs = []
        frame_idx = 0

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


if __name__ == "__main__":
    main()
