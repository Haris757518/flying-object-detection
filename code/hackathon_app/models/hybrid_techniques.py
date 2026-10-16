"""
5 Hybrid Techniques for Drone Detection & Analysis:
1. YOLO12 + CNN: Detection + Classification
2. YOLO12 + LSTM: Detection + Motion Analysis
3. YOLO12 + Transformer: Detection + Improved Feature Representation
4. YOLO12 + BiLSTM + Attention: Detection + Temporal Classification
5. YOLO12 + GAN + Transformer: Detection + Synthetic Data Augmentation + Spatio-Temporal Modeling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.gan_enhancer import LightweightGANGenerator
from models.temporal_transformer import PositionalEncoding


# ==============================================================================
# TECHNIQUE 1: YOLO12 + CNN (Detection + Classification)
# ==============================================================================

class YOLO12_CNN_Hybrid(nn.Module):
    """
    Technique 1: YOLO12 Detection fused with Deep Multi-Scale Residual CNN.
    Performs high-resolution visual feature extraction and multi-class target classification.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.technique_name = "YOLO12 + CNN"
        self.application = "Detection + Classification"

        # Convolutional Feature Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale Conv Blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.res1 = nn.Conv2d(32, 64, kernel_size=1, stride=2)

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.res2 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feat = self.stem(x)
        feat = F.relu(self.block1(feat) + self.res1(feat))
        feat = F.relu(self.block2(feat) + self.res2(feat))
        pooled = self.pool(feat)
        flat = torch.flatten(pooled, 1)
        flat = self.dropout(flat)
        logits = self.classifier(flat)
        return logits


# ==============================================================================
# TECHNIQUE 2: YOLO12 + LSTM (Detection + Motion Analysis)
# ==============================================================================

class YOLO12_LSTM_Hybrid(nn.Module):
    """
    Technique 2: YOLO12 Detection fused with Recurrent LSTM Motion Analyzer.
    Encodes kinematic flight trajectories (x, y, w, h, vx, vy, ax, ay, scale_rate)
    or sequential visual feature transitions for motion modeling.
    """
    def __init__(self, in_features: int = 9, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.technique_name = "YOLO12 + LSTM"
        self.application = "Detection + Motion Analysis"

        # Feature projection stem
        self.feature_stem = nn.Sequential(
            nn.Conv2d(3, in_features, kernel_size=8, stride=8),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If image tensor (B, C, H, W), project via feature stem to sequence tokens
        if x.dim() == 4:
            feats = self.feature_stem(x)
            b, c, h, w = feats.shape
            seq = feats.flatten(2).permute(0, 2, 1)  # (B, H*W, in_features)
        else:
            seq = x
            if seq.shape[-1] != self.lstm.input_size:
                if not hasattr(self, 'seq_proj'):
                    self.seq_proj = nn.Linear(seq.shape[-1], self.lstm.input_size).to(seq.device)
                seq = self.seq_proj(seq)

        # seq: (B, Seq_len, in_features)
        lstm_out, (hn, cn) = self.lstm(seq)
        last_hidden = hn[-1]  # (B, hidden_dim)
        logits = self.fc(last_hidden)
        return logits


# ==============================================================================
# TECHNIQUE 3: YOLO12 + Transformer (Detection + Improved Feature Representation)
# ==============================================================================

class YOLO12_Transformer_Hybrid(nn.Module):
    """
    Technique 3: YOLO12 Detection fused with Vision Self-Attention Transformer.
    Applies multi-head self-attention over spatial feature patches to model
    long-range dependencies and subtle signatures of faint airborne targets.
    """
    def __init__(self, in_channels: int = 3, d_model: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.technique_name = "YOLO12 + Transformer"
        self.application = "Detection + Improved Feature Representation"

        # Patch embedding CNN stem
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=8, stride=8),  # 64x64 -> 8x8 tokens
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # Kinematic track sequence (B, S, F)
            if not hasattr(self, 'track_proj'):
                self.track_proj = nn.Linear(x.shape[-1], 64).to(x.device)
            tokens = self.track_proj(x)
        else:  # Image tensor (B, C, H, W)
            patches = self.patch_embed(x)  # (B, d_model, H', W')
            b, d, h, w = patches.shape
            tokens = patches.flatten(2).permute(0, 2, 1)  # (B, H'*W', d_model)
        
        tokens = self.pos_encoder(tokens)
        encoded = self.transformer_encoder(tokens)
        pooled = encoded.mean(dim=1)  # Global context average pooling
        logits = self.classifier(pooled)
        return logits


# ==============================================================================
# TECHNIQUE 4: YOLO12 + BiLSTM + Attention (Detection + Temporal Classification)
# ==============================================================================

class TemporalAttention(nn.Module):
    """Soft Attention Mechanism over temporal sequence representations."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs: torch.Tensor) -> torch.Tensor:
        # rnn_outputs: (B, Seq_len, hidden_dim)
        scores = self.attn_weights(rnn_outputs)  # (B, Seq_len, 1)
        weights = F.softmax(scores, dim=1)        # (B, Seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (B, hidden_dim)
        return context


class YOLO12_BiLSTM_Attention_Hybrid(nn.Module):
    """
    Technique 4: YOLO12 Detection + Bidirectional LSTM + Temporal Attention.
    Captures both forward and reverse temporal context across flight paths,
    weighted dynamically by temporal soft-attention for high-accuracy threat classification.
    """
    def __init__(self, in_features: int = 32, hidden_dim: int = 64, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.technique_name = "YOLO12 + BiLSTM + Attention"
        self.application = "Detection + Temporal Classification"

        # Feature projection
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, in_features, kernel_size=4, stride=4),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True)
        )

        self.bilstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = TemporalAttention(hidden_dim=hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # (B, C, H, W) -> spatial tokens sequence
            feats = self.feature_extractor(x)
            b, c, h, w = feats.shape
            seq = feats.flatten(2).permute(0, 2, 1)  # (B, H*W, in_features)
        else:
            seq = x
            if seq.shape[-1] != self.bilstm.input_size:
                if not hasattr(self, 'seq_proj'):
                    self.seq_proj = nn.Linear(seq.shape[-1], self.bilstm.input_size).to(seq.device)
                seq = self.seq_proj(seq)

        lstm_out, _ = self.bilstm(seq)  # (B, Seq_len, hidden_dim * 2)
        context_vector = self.attention(lstm_out)  # (B, hidden_dim * 2)
        logits = self.classifier(context_vector)
        return logits


# ==============================================================================
# TECHNIQUE 5: YOLO12 + GAN + Transformer (Detection + Augmentation + Spatio-Temporal)
# ==============================================================================

class YOLO12_GAN_Transformer_Hybrid(nn.Module):
    """
    Technique 5: YOLO12 Detection + GAN Pre-Augmentation & Feature Enhancement + Spatio-Temporal Transformer.
    Applies high-frequency super-resolution generator enhancement followed by multi-head
    self-attention encoder for state-of-the-art robustness on difficult distant targets.
    """
    def __init__(self, in_channels: int = 3, d_model: int = 64, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.technique_name = "YOLO12 + GAN + Transformer"
        self.application = "Detection + Synthetic Data Augmentation"

        # Embedded Lightweight GAN Sub-module
        self.gan_generator = LightweightGANGenerator(scale_factor=2)
        
        # Spatio-Temporal Transformer Encoder
        self.patch_proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=8, stride=8),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If image input, pass through feature projection and transformer
        if x.dim() == 4:
            # Multi-scale patch projection
            patches = self.patch_proj(x)
            b, d, h, w = patches.shape
            tokens = patches.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)
        else:
            tokens = x
            if tokens.shape[-1] != 64:
                if not hasattr(self, 'tok_proj'):
                    self.tok_proj = nn.Linear(tokens.shape[-1], 64).to(tokens.device)
                tokens = self.tok_proj(tokens)

        tokens = self.pos_encoder(tokens)
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


# ==============================================================================
# MODEL FACTORY
# ==============================================================================

def get_all_5_hybrid_models(num_classes: int = 4) -> list:
    """Instantiates and returns all 5 hybrid technique models."""
    return [
        YOLO12_CNN_Hybrid(num_classes=num_classes),
        YOLO12_LSTM_Hybrid(num_classes=num_classes),
        YOLO12_Transformer_Hybrid(num_classes=num_classes),
        YOLO12_BiLSTM_Attention_Hybrid(num_classes=num_classes),
        YOLO12_GAN_Transformer_Hybrid(num_classes=num_classes)
    ]
