import os
import math
import torch
import torch.nn as nn
import numpy as np

BEHAVIORS = ['APPROACHING', 'HOVERING', 'TRANSITING', 'DIVING', 'NATURAL']
BEHAVIOR_TO_ID = {b: i for i, b in enumerate(BEHAVIORS)}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TrajectoryTransformer(nn.Module):
    """
    Transformer model for kinematic track trajectory behavior classification.
    Input features per frame: (norm_x, norm_y, norm_w, norm_h, norm_vx, norm_vy, norm_ax, norm_ay, scale_change) = 9 dims
    """
    def __init__(self, in_features=9, d_model=64, nhead=4, num_layers=2, num_classes=len(BEHAVIORS), dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq_len, in_features)
        h = self.input_projection(x)
        h = self.pos_encoder(h)
        h_enc = self.transformer_encoder(h)
        # Global temporal pooling (mean over sequence)
        h_pool = h_enc.mean(dim=1)
        logits = self.classifier(h_pool)
        return logits


class SpatioTemporalBehaviorModel:
    """
    Wrapper for behavior classification with Transformer and rule-based kinematic fallback.
    """
    def __init__(self, weights_path="checkpoints/trajectory_transformer.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrajectoryTransformer().to(self.device)
        self.model.eval()

        if weights_path and os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                print(f"[TrajectoryTransformer] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[TrajectoryTransformer] Note: {e}")

    def analyze_track_history(self, track_history, img_w=640, img_h=512):
        """
        Analyze a track's historical states [(xc, yc, w, h), ...]
        Returns: predicted_behavior (str), confidence (float), kinematics (dict)
        """
        if len(track_history) < 2:
            return "TRANSITING", 0.5, {"speed": 0.0, "heading": 0.0, "scale_rate": 0.0}

        # Compute Kinematics
        coords = np.array(track_history, dtype=np.float32)  # shape (N, 4)
        xs = coords[:, 0] / float(img_w)
        ys = coords[:, 1] / float(img_h)
        ws = coords[:, 2] / float(img_w)
        hs = coords[:, 3] / float(img_h)
        areas = ws * hs

        # Velocities
        vxs = np.diff(xs)
        vys = np.diff(ys)
        vareas = np.diff(areas)

        mean_speed = float(np.mean(np.sqrt(vxs**2 + vys**2)))
        scale_rate = float(np.mean(vareas))
        heading_deg = float(np.degrees(np.arctan2(np.mean(vys), np.mean(vxs)))) if len(vxs) > 0 else 0.0

        kinematics = {
            "speed": mean_speed * 100.0,  # normalized percentage speed
            "heading": heading_deg,
            "scale_rate": scale_rate * 100.0,
            "altitude_change": float(ys[-1] - ys[0])
        }

        # If sequence is long enough, run Transformer inference
        if len(track_history) >= 8:
            features = []
            # Build sliding features
            for i in range(1, len(coords)):
                vx = xs[i] - xs[i-1]
                vy = ys[i] - ys[i-1]
                ax = (xs[i] - 2 * xs[i-1] + (xs[i-2] if i > 1 else xs[0]))
                ay = (ys[i] - 2 * ys[i-1] + (ys[i-2] if i > 1 else ys[0]))
                sc = (areas[i] - areas[i-1])
                feat = [xs[i], ys[i], ws[i], hs[i], vx, vy, ax, ay, sc]
                features.append(feat)

            # Pad or truncate to fixed length
            feat_arr = np.array(features, dtype=np.float32)
            tensor_in = torch.from_numpy(feat_arr).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            predicted_behavior = BEHAVIORS[pred_idx]
            confidence = float(probs[pred_idx])
            return predicted_behavior, confidence, kinematics

        # Fast kinematic rule-based classifier for short tracks
        if mean_speed < 0.008:
            return "HOVERING", 0.85, kinematics
        elif scale_rate > 0.0005:
            return "APPROACHING", 0.88, kinematics
        elif vys[-1] > 0.02 and mean_speed > 0.025:
            return "DIVING", 0.82, kinematics
        elif np.std(vys) > 0.015:
            return "NATURAL", 0.75, kinematics
        else:
            return "TRANSITING", 0.80, kinematics
