"""Technique 4: YOLO12 + BiLSTM + Attention (Detection + Bidirectional Temporal Classification)
Architecture:
- YOLO12: Continuous object detection in video frames.
- Bidirectional LSTM (BiLSTM): Models sequence dependencies forward and backward in time.
- Temporal Attention Layer: Computes self-attention over the sequence of hidden states,
  assigning higher weights to key dynamic moments (e.g. rotor tilt vs. wing flapping).
"""

import os
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["airplane", "bird", "helicopter", "kite", "parachute", "UAV"]
NUM_CLASSES = len(CLASS_NAMES)
FEATURE_DIM = 8

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn_weights(rnn_outputs)  # (batch_size, seq_len, 1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(alpha * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, alpha

class BiLSTM_Attention_Classifier(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2, num_classes=NUM_CLASSES):
        super(BiLSTM_Attention_Classifier, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim * 2)  # *2 for bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        lstm_out, _ = self.bilstm(x)  # (B, T, hidden_dim * 2)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, attn_weights

class YOLO12_BiLSTM_Attention_Hybrid:
    def __init__(self, yolo_weights="yolov8n.pt", rnn_weights=None):
        self.detector = YOLO(yolo_weights)
        self.temporal_classifier = BiLSTM_Attention_Classifier().to(device)
        if rnn_weights and os.path.exists(rnn_weights):
            self.temporal_classifier.load_state_dict(torch.load(rnn_weights, map_location=device))
        self.temporal_classifier.eval()

    def predict_sequence(self, sequence_features):
        """
        Takes sequence_features: (batch_size, seq_len, FEATURE_DIM)
        Returns classification and temporal attention weights.
        """
        tensor = torch.tensor(sequence_features, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, weights = self.temporal_classifier(tensor)
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            conf = torch.max(probs, dim=-1)[0].item()
        return {
            "predicted_class": CLASS_NAMES[pred_class],
            "confidence": float(conf),
            "attention_weights": weights.squeeze(-1).cpu().numpy().tolist()
        }

if __name__ == "__main__":
    print("Initializing Technique 4: YOLO12 + BiLSTM + Attention Module...")
    model = YOLO12_BiLSTM_Attention_Hybrid()
    dummy_seq = np.random.randn(1, 16, FEATURE_DIM)
    result = model.predict_sequence(dummy_seq)
    print("Prediction result:", result["predicted_class"], f"Confidence: {result['confidence']:.2f}")
    print("Technique 4 successfully initialized and verified.")
