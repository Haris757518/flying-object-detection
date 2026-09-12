from enum import Enum
import numpy as np


class ThreatLevel(Enum):
    LOW = "LOW"            # 🟢 0-29
    WATCH = "WATCH"        # 🟡 30-59
    HIGH = "HIGH"          # 🟠 60-79
    CRITICAL = "CRITICAL"  # 🔴 80-100


CLASS_BASE_WEIGHTS = {
    "DRONE": 20.0,
    "HELICOPTER": 15.0,
    "AIRPLANE": 10.0,
    "BIRD": 2.0
}

BEHAVIOR_MULTIPLIERS = {
    "DIVING": 20.0,
    "HOVERING": 15.0,
    "APPROACHING": 10.0,
    "TRANSITING": 5.0,
    "NATURAL": 0.0
}

# Real-world baseline size (meters)
REAL_WORLD_DIAMETERS = {
    "DRONE": 0.5,        # 50cm quadcopter
    "HELICOPTER": 10.0,  # 10m fuselage
    "AIRPLANE": 35.0,    # 35m airliner / light aircraft
    "BIRD": 0.35         # 35cm wingspan
}


class RiskEngine:
    """
    Computes real-time threat scores (0-100) and assigns threat level categories.
    """
    def __init__(self, focal_length_px=800.0, no_fly_radius_m=30.0, warning_radius_m=100.0):
        self.focal_length_px = focal_length_px
        self.no_fly_radius_m = no_fly_radius_m
        self.warning_radius_m = warning_radius_m

    def estimate_distance(self, bbox, class_name, img_w=640, img_h=512):
        """
        Estimate 3D distance (meters) using pinhole optical camera model:
        Distance = (Real_Size * Focal_Length) / Pixel_Size
        """
        w_px = max(2.0, bbox[2] - bbox[0])
        h_px = max(2.0, bbox[3] - bbox[1])
        dim_px = max(w_px, h_px)

        real_size = REAL_WORLD_DIAMETERS.get(class_name.upper(), 0.5)
        dist_m = (real_size * self.focal_length_px) / dim_px
        return float(np.clip(dist_m, 1.0, 500.0))

    def evaluate_zone(self, distance_m):
        """
        Determine spatial risk zone based on estimated distance to asset/camera.
        """
        if distance_m <= self.no_fly_radius_m:
            return "NO_FLY_ZONE", 30.0
        elif distance_m <= self.warning_radius_m:
            return "WARNING_ZONE", 15.0
        else:
            return "SAFE_ZONE", 0.0

    def calculate_threat(self, track, behavior, kinematics, img_w=640, img_h=512):
        """
        Calculates multi-factor threat metrics for a tracked target.
        Returns: dict with threat_score, threat_level, color, distance, zone, breakdown.
        """
        class_name = track.class_name.upper()
        bbox = track.tlbr

        # 1. Distance & Zone
        distance_m = self.estimate_distance(bbox, class_name, img_w, img_h)
        zone_name, zone_weight = self.evaluate_zone(distance_m)

        # 2. Class Weight
        cls_weight = CLASS_BASE_WEIGHTS.get(class_name, 10.0)

        # 3. Behavior Additive Score
        beh_score = BEHAVIOR_MULTIPLIERS.get(behavior.upper(), 0.0)

        # 4. Distance Proximity Score (closer -> higher score, max 30 points)
        dist_score = max(0.0, (150.0 - min(150.0, distance_m)) / 150.0 * 30.0)

        # Multi-factor score formula (Additive to max 100)
        raw_score = dist_score + zone_weight + cls_weight + beh_score
        threat_score = float(np.clip(raw_score, 0.0, 100.0))

        # Assign Threat Level bands
        if threat_score >= 80.0:
            level = ThreatLevel.CRITICAL
            color_bgr = (0, 0, 255)      # Red
            badge = "🔴 CRITICAL"
        elif threat_score >= 60.0:
            level = ThreatLevel.HIGH
            color_bgr = (0, 128, 255)    # Orange
            badge = "🟠 HIGH"
        elif threat_score >= 30.0:
            level = ThreatLevel.WATCH
            color_bgr = (0, 255, 255)    # Yellow
            badge = "🟡 WATCH"
        else:
            level = ThreatLevel.LOW
            color_bgr = (0, 255, 0)      # Green
            badge = "🟢 LOW"

        return {
            "threat_score": threat_score,
            "threat_level": level.value,
            "threat_badge": badge,
            "color_bgr": color_bgr,
            "distance_m": round(distance_m, 1),
            "zone": zone_name,
            "speed": round(kinematics.get("speed", 0.0), 2),
            "heading": round(kinematics.get("heading", 0.0), 1),
            "behavior": behavior,
            "class_name": class_name,
            "track_id": track.track_id
        }
