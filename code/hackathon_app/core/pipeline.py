"""
Master End-to-End Drone Detection & Threat Defense Pipeline.
Connects: Preprocessing -> GAN Enhancer -> YOLO12 -> ByteTrack -> Trajectory Transformer -> Risk Engine -> Tactical HUD.
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path
from models.gan_enhancer import GANEnhancer
from models.temporal_transformer import SpatioTemporalBehaviorModel
from models.yolo12_detector import YOLO12Detector
from tracking.bytetrack import ByteTracker
from core.risk_engine import RiskEngine, ThreatLevel


class DroneDefensePipeline:
    """
    Complete real-time aerial threat detection pipeline.
    """
    def __init__(self, yolo_model=None, conf_thresh=0.25, iou_thresh=0.45, device=None, imgsz=640):
        if yolo_model is None or yolo_model == "yolov8n.pt":
            yolo_model = str(Path(__file__).parent.parent / "models" / "best_aod4.pt")
            
        self.gan_enhancer = GANEnhancer(device=device)
        self.detector = YOLO12Detector(model_path=yolo_model, conf_thresh=conf_thresh, iou_thresh=iou_thresh, device=device, imgsz=imgsz)
        self._tracker_high_thresh = float(conf_thresh)
        self._tracker_low_thresh = max(0.1, float(conf_thresh) - 0.15)
        self.tracker = ByteTracker(high_thresh=self._tracker_high_thresh, low_thresh=self._tracker_low_thresh)
        self.behavior_model = SpatioTemporalBehaviorModel(device=device)
        self.risk_engine = RiskEngine()

        self.frame_count = 0
        self.fps = 0.0
        self._prev_time = time.time()

    def reset(self):
        """Reset state before starting a new video or camera session."""
        self.tracker.reset(reset_track_ids=True)
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
            threat_info["class_confidence"] = getattr(track, "class_confidence", track.confidence)
            threat_info["class_vote_share"] = getattr(track, "class_vote_share", 1.0)
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

        drawn_rects = []
        def does_overlap(r1, r2):
            return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

        for t in targets:
            bbox = [int(v) for v in t["bbox"]]
            xmin, ymin, xmax, ymax = bbox
            color = t["color_bgr"]
            track_id = t["track_id"]
            cls_name = t["class_name"]
            beh = t["behavior"]
            score = t["threat_score"]
            conf = t.get("class_confidence", t.get("confidence", 0.0)) * 100

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

            # Tags formatting
            top_tag = f"#{track_id} {cls_name} {conf:.0f}%"
            bot_tag = f"{beh} | RISK: {score:.0f}/100"
            
            (tw1, th1), _ = cv2.getTextSize(top_tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, 1)
            (tw2, th2), _ = cv2.getTextSize(bot_tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, 1)
            
            # Top tag initial position
            tag_y1 = max(bar_h + th1 + 4, ymin - 4)
            r1 = [xmin, tag_y1 - th1 - 4, xmin + tw1 + 6, tag_y1 + 2]
            
            # Collision resolution for top tag
            while any(does_overlap(r1, d_rect) for d_rect in drawn_rects) and r1[1] > bar_h + 10:
                shift = th1 + 6
                r1[1] -= shift
                r1[3] -= shift
                tag_y1 -= shift
            drawn_rects.append(r1)

            cv2.rectangle(vis, (r1[0], r1[1]), (r1[2], r1[3]), (10, 14, 20), -1)
            cv2.putText(vis, top_tag, (r1[0] + 3, tag_y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, color, max(1, thickness - 1), cv2.LINE_AA)

            # Bottom tag initial position
            tag_y2 = min(h - 4, ymax + th2 + 4)
            r2 = [xmin, tag_y2 - th2 - 2, xmin + tw2 + 4, tag_y2 + 2]
            
            # Collision resolution for bottom tag
            while any(does_overlap(r2, d_rect) for d_rect in drawn_rects) and r2[3] < h - 10:
                shift = th2 + 6
                r2[1] += shift
                r2[3] += shift
                tag_y2 += shift
            drawn_rects.append(r2)

            cv2.rectangle(vis, (r2[0], r2[1]), (r2[2], r2[3]), (10, 14, 20), -1)
            cv2.putText(vis, bot_tag, (r2[0] + 2, tag_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, max(1, thickness - 1), cv2.LINE_AA)

        return vis
