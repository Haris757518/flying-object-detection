"""
ByteTrack Multi-Object Tracker (Pure Python Implementation).
Provides robust tracking with Kalman Filter state estimation, track persistence,
and historical trajectory recording for aerial drone monitoring.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_iou(box1, box2):
    """Compute IoU between box1 and box2 (xmin, ymin, xmax, ymax)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def iou_distance(tracks, detections):
    """Compute cost matrix based on 1.0 - IoU."""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        t_box = track.tlbr
        for j, det in enumerate(detections):
            d_box = det["bbox"]
            cost_matrix[i, j] = 1.0 - bbox_iou(t_box, d_box)
    return cost_matrix


class KalmanFilterState:
    """Kalman Filter for constant-velocity bounding box tracking."""
    def __init__(self, bbox):
        # State vector: [xc, yc, aspect_ratio, height, v_xc, v_yc, v_a, v_h]
        xmin, ymin, xmax, ymax = bbox
        w = max(1.0, xmax - xmin)
        h = max(1.0, ymax - ymin)
        xc = xmin + w / 2.0
        yc = ymin + h / 2.0
        a = w / h

        self.mean = np.array([xc, yc, a, h, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.diag([10, 10, 1, 10, 100, 100, 10, 100]).astype(np.float32)

        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = 1.0

        self._H = np.eye(4, 8, dtype=np.float32)
        self._Q = np.diag([1, 1, 0.01, 1, 10, 10, 0.1, 10]).astype(np.float32)
        self._R = np.diag([10, 10, 0.1, 10]).astype(np.float32)

    def predict(self):
        self.mean = np.dot(self._F, self.mean)
        self.covariance = np.dot(np.dot(self._F, self.covariance), self._F.T) + self._Q
        return self.tlbr

    def update(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = max(1.0, xmax - xmin)
        h = max(1.0, ymax - ymin)
        xc = xmin + w / 2.0
        yc = ymin + h / 2.0
        a = w / h
        measurement = np.array([xc, yc, a, h], dtype=np.float32)

        y = measurement - np.dot(self._H, self.mean)
        S = np.dot(np.dot(self._H, self.covariance), self._H.T) + self._R
        K = np.dot(np.dot(self.covariance, self._H.T), np.linalg.inv(S))

        self.mean = self.mean + np.dot(K, y)
        self.covariance = self.covariance - np.dot(np.dot(K, self._H), self.covariance)

    @property
    def tlbr(self):
        xc, yc, a, h = self.mean[:4]
        w = a * h
        return (xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0)


class STrack:
    _count = 0

    def __init__(self, detection):
        STrack._count += 1
        self.track_id = STrack._count
        self.kf = KalmanFilterState(detection["bbox"])
        self.confidence = detection["confidence"]
        self.class_id = detection["class_id"]
        self.class_name = detection["class_name"]
        self.state = "New"  # New, Tracked, Lost, Removed
        self.age = 0
        self.time_since_update = 0
        self.history = []  # [(xc, yc, w, h), ...]
        self._record_history()

    def _record_history(self):
        box = self.tlbr
        w = box[2] - box[0]
        h = box[3] - box[1]
        xc = box[0] + w / 2.0
        yc = box[1] + h / 2.0
        self.history.append((xc, yc, w, h))
        if len(self.history) > 60:
            self.history.pop(0)

    @property
    def tlbr(self):
        return self.kf.tlbr

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        self.kf.update(detection["bbox"])
        self.confidence = detection["confidence"]
        self.class_id = detection["class_id"]
        self.class_name = detection["class_name"]
        self.state = "Tracked"
        self.time_since_update = 0
        self._record_history()

    def mark_lost(self):
        self.state = "Lost"

    def mark_removed(self):
        self.state = "Removed"


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker algorithm.
    """
    def __init__(self, high_thresh=0.45, low_thresh=0.15, match_thresh=0.7, max_time_lost=30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0

    def update(self, detections):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Predict Kalman states
        for t in self.tracked_stracks:
            t.predict()
        for t in self.lost_stracks:
            t.predict()

        # Split detections by confidence
        high_dets = [d for d in detections if d["confidence"] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d["confidence"] < self.high_thresh]

        # 1st association: High confidence detections with tracked tracks
        pool = self.tracked_stracks + self.lost_stracks
        matched_tracks, unmatched_tracks, unmatched_dets = self._linear_assignment(pool, high_dets, self.match_thresh)

        for t_idx, d_idx in matched_tracks:
            track = pool[t_idx]
            det = high_dets[d_idx]
            track.update(det)
            if track in self.lost_stracks:
                self.lost_stracks.remove(track)
                refind_stracks.append(track)
            activated_stracks.append(track)

        # 2nd association: Unmatched tracked tracks with low confidence detections
        rem_tracked = [pool[i] for i in unmatched_tracks if pool[i] in self.tracked_stracks]
        matched_tracks2, unmatched_tracks2, _ = self._linear_assignment(rem_tracked, low_dets, self.match_thresh)

        for t_idx, d_idx in matched_tracks2:
            track = rem_tracked[t_idx]
            det = low_dets[d_idx]
            track.update(det)
            activated_stracks.append(track)

        # Handle unmatched tracks
        for t_idx in unmatched_tracks2:
            track = rem_tracked[t_idx]
            track.mark_lost()
            lost_stracks.append(track)

        # Initialize new tracks from unmatched high confidence detections
        for d_idx in unmatched_dets:
            det = high_dets[d_idx]
            new_track = STrack(det)
            new_track.state = "Tracked"
            activated_stracks.append(new_track)

        # Prune dead tracks
        for track in self.lost_stracks:
            if track.time_since_update > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in activated_stracks if t.state == "Tracked"]
        self.lost_stracks = [t for t in (self.lost_stracks + lost_stracks) if t.state == "Lost" and t not in removed_stracks]
        self.removed_stracks.extend(removed_stracks)

        return self.tracked_stracks

    def _linear_assignment(self, tracks, detections, thresh):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = iou_distance(tracks, detections)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < thresh:
                matched_indices.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched_indices, list(unmatched_tracks), list(unmatched_dets)
