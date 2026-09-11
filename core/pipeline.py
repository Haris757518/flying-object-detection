"""
Master End-to-End Drone Detection & Threat Defense Pipeline.
Connects: Preprocessing -> GAN Enhancer -> YOLO12 -> ByteTrack -> Trajectory Transformer -> Risk Engine -> Tactical HUD.
"""

import os
import cv2
import numpy as np
import time
from models.gan_enhancer import GANEnhancer
from models.temporal_transformer import SpatioTemporalBehaviorModel
from models.yolo12_detector import YOLO12Detector
from tracking.bytetrack import ByteTracker
from core.risk_engine import RiskEngine, ThreatLevel


class DroneDefensePipeline:
    """
    Complete real-time aerial threat detection pipeline.
    """
    def __init__(self, yolo_model="yolov8n.pt", conf_thresh=0.25, iou_thresh=0.45, device=None):
        self.gan_enhancer = GANEnhancer(device=device)
        self.detector = YOLO12Detector(model_path=yolo_model, conf_thresh=conf_thresh, iou_thresh=iou_thresh, device=device)
        self.tracker = ByteTracker(high_thresh=conf_thresh, low_thresh=max(0.1, conf_thresh - 0.15))
        self.behavior_model = SpatioTemporalBehaviorModel(device=device)
        self.risk_engine = RiskEngine()

        self.frame_count = 0
        self.fps = 0.0
        self._prev_time = time.time()

    def process_frame(self, frame, sensor_type="IR", enhance_crops=False):
        """
        Processes a single video frame through the full pipeline.
        Returns:
            annotated_frame: Frame with tactical HUD, boxes, trails, and threat badges.
            active_targets: List of active target metadata dicts.
            highest_threat: Summary dict of the most critical target detected.
        """
        if frame is None:
            return None, [], None

        self.frame_count += 1
        curr_time = time.time()
        dt = curr_time - self._prev_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self._prev_time = curr_time

        h, w = frame.shape[:2]

        # 1. Preprocessing & Thermal Normalization
        preprocessed = self.gan_enhancer.preprocess_frame(frame, sensor_type=sensor_type)

        # 2. YOLO12 & Thermal Detection
        detections = self.detector.detect(preprocessed, sensor_type=sensor_type)

        # Optional: GAN Super-Resolution on low-res detection crops
        if enhance_crops:
            for det in detections:
                bx = [int(v) for v in det["bbox"]]
                crop = preprocessed[max(0, bx[1]):min(h, bx[3]), max(0, bx[0]):min(w, bx[2])]
                if crop.size > 0:
                    _ = self.gan_enhancer.enhance_crop(crop)

        # 3. ByteTrack Multi-Object Tracking
        tracked_objects = self.tracker.update(detections)

        # 4. Trajectory Transformer & Risk Engine Evaluation
        active_targets = []
        highest_threat = None
        max_threat_score = -1.0

        for track in tracked_objects:
            # Analyze temporal kinematics & behavior with Transformer
            behavior, conf, kinematics = self.behavior_model.analyze_track_history(track.history, img_w=w, img_h=h)

            # Compute Multi-Factor Threat Score
            threat_info = self.risk_engine.calculate_threat(track, behavior, kinematics, img_w=w, img_h=h)
            threat_info["confidence"] = track.confidence
            threat_info["history"] = list(track.history)
            threat_info["bbox"] = track.tlbr
            active_targets.append(threat_info)

            if threat_info["threat_score"] > max_threat_score:
                max_threat_score = threat_info["threat_score"]
                highest_threat = threat_info

        # 5. Render Tactical HUD & Overlays
        annotated_frame = self.render_tactical_hud(preprocessed, active_targets, sensor_type=sensor_type)

        return annotated_frame, active_targets, highest_threat

    def render_tactical_hud(self, frame, targets, sensor_type="IR"):
        """
        Draws tactical defense bounding boxes, motion trails, telemetry, and threat alert banner.
        Dynamically scales with input video resolution for crisp display.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Resolution-adaptive scaling factor
        scale = max(0.35, min(1.2, w / 640.0))
        bar_h = max(26, int(h * 0.075))
        font_scale = 0.45 * scale
        thickness = max(1, int(1.5 * scale))

        # Draw Global HUD Header Bar with Semi-Transparent Overlay
        header_overlay = vis.copy()
        cv2.rectangle(header_overlay, (0, 0), (w, bar_h), (12, 16, 24), -1)
        cv2.addWeighted(header_overlay, 0.85, vis, 0.15, 0, vis)

        title_text = "ANTIDRONE TACTICAL DEFENSE"
        cv2.putText(vis, title_text, (8, int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, (0, 220, 255), thickness, cv2.LINE_AA)

        info_text = f"SENSOR: {sensor_type.upper()} | FPS: {self.fps:.1f} | TARGETS: {len(targets)}"
        (iw, ih), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, 1)
        cv2.putText(vis, info_text, (max(8, w - iw - 8), int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (200, 220, 240), max(1, thickness - 1), cv2.LINE_AA)

        for t in targets:
            bbox = [int(v) for v in t["bbox"]]
            xmin, ymin, xmax, ymax = bbox
            color = t["color_bgr"]
            track_id = t["track_id"]
            cls_name = t["class_name"]
            beh = t["behavior"]
            dist = t["distance_m"]
            score = t["threat_score"]

            # Draw Motion History Trail
            history = t["history"]
            if len(history) > 1:
                pts = np.array([(int(p[0]), int(p[1])) for p in history], dtype=np.int32)
                cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=max(1, int(2 * scale)), lineType=cv2.LINE_AA)

            # Draw Bounding Box
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, max(1, int(2 * scale)))
            
            # Corner accents
            corner_len = min(14, max(4, (xmax - xmin) // 4))
            cv2.line(vis, (xmin, ymin), (xmin + corner_len, ymin), (255, 255, 255), max(1, int(2 * scale)))
            cv2.line(vis, (xmin, ymin), (xmin, ymin + corner_len), (255, 255, 255), max(1, int(2 * scale)))
            cv2.line(vis, (xmax, ymax), (xmax - corner_len, ymax), (255, 255, 255), max(1, int(2 * scale)))
            cv2.line(vis, (xmax, ymax), (xmax, ymax - corner_len), (255, 255, 255), max(1, int(2 * scale)))

            # Telemetry Tag
            tag = f"#{track_id} {cls_name} {dist}m {beh}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, 1)
            tag_y = max(bar_h + th + 4, ymin - 4)
            cv2.rectangle(vis, (xmin, tag_y - th - 4), (xmin + tw + 6, tag_y + 2), (10, 14, 20), -1)
            cv2.putText(vis, tag, (xmin + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, color, max(1, thickness - 1), cv2.LINE_AA)

            # Threat Badge Tag below box
            badge_str = f"RISK: {score:.0f}/100"
            (bw_t, bh_t), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, 1)
            badge_y = min(h - 4, ymax + bh_t + 4)
            cv2.rectangle(vis, (xmin, badge_y - bh_t - 2), (xmin + bw_t + 4, badge_y + 2), (10, 14, 20), -1)
            cv2.putText(vis, badge_str, (xmin + 2, badge_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, max(1, thickness - 1), cv2.LINE_AA)

        return vis
