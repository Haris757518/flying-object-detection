"""Generates a realistic test surveillance video with atmospheric clouds,
a flying bird crossing the frame, and a drone entering the geofenced perimeter.
"""

import cv2
import numpy as np

def create_sample_video(filename="sample_surveillance_video.mp4", num_frames=120, width=640, height=480):
    print(f"Creating sample test video: {filename} ({num_frames} frames)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    
    # Pre-generate atmospheric gradient sky
    sky_bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        # Daylight sky gradient
        alpha = y / height
        color = (1 - alpha) * np.array([230, 200, 160]) + alpha * np.array([255, 235, 210])
        sky_bg[y, :] = color.astype(np.uint8)
        
    for frame_idx in range(num_frames):
        frame = sky_bg.copy()
        
        # Add light atmospheric cloud textures
        cv2.ellipse(frame, (200 + frame_idx, 100), (90, 30), 0, 0, 360, (255, 250, 245), -1)
        cv2.ellipse(frame, (450 - frame_idx, 80), (120, 35), 0, 0, 360, (255, 250, 245), -1)
        
        # 1. Flying Bird (flapping across top-right: erratic zigzag)
        bird_x = int(580 - frame_idx * 4)
        bird_y = int(90 + 15 * np.sin(frame_idx * 0.4))
        # Draw bird silhouette (V-shape wings)
        wing_span = 8 + int(4 * np.sin(frame_idx * 0.8))
        cv2.line(frame, (bird_x - wing_span, bird_y - 4), (bird_x, bird_y), (40, 40, 40), 2)
        cv2.line(frame, (bird_x, bird_y), (bird_x + wing_span, bird_y - 4), (40, 40, 40), 2)
        cv2.circle(frame, (bird_x, bird_y), 3, (30, 30, 30), -1)
        
        # 2. Drone / UAV (Entering from left, hovering into center geofenced restricted zone)
        drone_x = int(80 + frame_idx * 3.5)
        drone_y = int(140 + frame_idx * 1.5)
        # Draw quadcopter body and rotors
        cv2.rectangle(frame, (drone_x - 12, drone_y - 6), (drone_x + 12, drone_y + 6), (50, 50, 50), -1)
        # Arms
        cv2.line(frame, (drone_x - 18, drone_y - 10), (drone_x + 18, drone_y + 10), (70, 70, 70), 2)
        cv2.line(frame, (drone_x - 18, drone_y + 10), (drone_x + 18, drone_y - 10), (70, 70, 70), 2)
        # Rotors
        rotor_spin = int(6 * np.sin(frame_idx * 1.5))
        cv2.ellipse(frame, (drone_x - 18, drone_y - 10), (8, 2), rotor_spin, 0, 360, (90, 90, 90), 1)
        cv2.ellipse(frame, (drone_x + 18, drone_y - 10), (8, 2), -rotor_spin, 0, 360, (90, 90, 90), 1)
        cv2.ellipse(frame, (drone_x - 18, drone_y + 10), (8, 2), rotor_spin, 0, 360, (90, 90, 90), 1)
        cv2.ellipse(frame, (drone_x + 18, drone_y + 10), (8, 2), -rotor_spin, 0, 360, (90, 90, 90), 1)
        
        out.write(frame)
        
    out.release()
    print(f"Sample video created successfully: {filename}")

if __name__ == "__main__":
    create_sample_video()
