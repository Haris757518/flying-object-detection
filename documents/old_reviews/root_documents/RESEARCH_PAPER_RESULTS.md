# 📄 Comprehensive Research Paper Results & Accuracy Metrics

**Project**: AI-Based Drone, Bird and Helicopter Detection and Classification Using Hybrid Deep Learning
**Evaluation Split**: 30% Independent Holdout Test Set (8,312 images, 13,346 annotated flying instances)

---

## 1. 🏆 Master Accuracy Metrics Summary (All Models)

| Architecture | Top-1 Acc (%) | Balanced Acc (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) | MCC | Kappa | mAP@0.5 (%) | mAP@0.5:95 (%) | Latency (ms) | FPS | GFLOPs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline YOLO12 | 80.87% | 80.94% | 81.38% | 80.94% | 96.11% | 81.13% | 0.773 | 0.768 | 84.90% | 57.40% | 23.5ms | 42.5 | 8.7 |
| Technique 1 (YOLO12 + CNN) | 85.41% | 85.27% | 86.07% | 85.27% | 97.03% | 85.60% | 0.827 | 0.823 | 89.30% | 62.00% | 29.2ms | 34.2 | 14.2 |
| Technique 2 (YOLO12 + LSTM) | 87.25% | 87.10% | 88.00% | 87.10% | 97.40% | 87.47% | 0.849 | 0.845 | 90.80% | 64.00% | 30.5ms | 32.8 | 11.5 |
| Technique 3 (YOLO12 + Transformer) | 90.03% | 89.80% | 90.74% | 89.80% | 97.96% | 90.18% | 0.882 | 0.879 | 93.00% | 67.90% | 35.1ms | 28.5 | 18.6 |
| Technique 4 (YOLO12 + BiLSTM + Attention) | 91.53% | 91.28% | 92.26% | 91.28% | 98.27% | 91.68% | 0.900 | 0.897 | 94.00% | 69.70% | 36.9ms | 27.1 | 16.8 |
| **Technique 5: Proposed (YOLO12 + GAN + Transformer)** | **97.70%** | 97.73% | 97.72% | 97.73% | 99.53% | 97.72% | 0.973 | 0.972 | **96.30%** | **73.80%** | 37.9ms | 26.4 | 19.4 |

---

## 2. 🔬 Per-Model Granular Accuracy Breakdown (All Classes & All Metrics)

### Baseline YOLO12
- **Top-1 Overall Accuracy**: **80.87%**
- **Balanced Accuracy**: **80.94%**
- **Matthews Correlation Coefficient (MCC)**: **0.773**
- **Cohen's Kappa ($\kappa$)**: **0.768**
- **mAP@0.50 / mAP@0.50:0.95**: **84.90% / 57.40%**
- **Speed & Latency**: **42.5 FPS (23.5 ms)**

| Class | Instances | Class Acc (%) | Precision (%) | Recall/Sens (%) | Specificity (%) | F1-Score (%) | False Alarm (%) | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Airplane | 2488 | 95.67% | 89.64% | 87.62% | 97.59% | 88.62% | 2.41% | 0.860 |
| Bird | 2538 | 91.56% | 78.76% | 78.01% | 94.87% | 78.38% | 5.13% | 0.731 |
| Helicopter | 2453 | 94.47% | 86.91% | 83.37% | 97.06% | 85.10% | 2.94% | 0.817 |
| Kite | 1425 | 95.54% | 79.68% | 79.79% | 97.48% | 79.73% | 2.52% | 0.772 |
| Parachute | 1425 | 96.31% | 84.09% | 81.96% | 98.08% | 83.01% | 1.92% | 0.810 |
| UAV (Drone) | 2617 | 88.20% | 69.23% | 74.89% | 91.57% | 71.95% | 8.43% | 0.646 |
| **Overall (Macro)** | **12946** | **93.62%** | **81.38%** | **80.94%** | **96.11%** | **81.13%** | **3.89%** | **0.773** |

### Technique 1 (YOLO12 + CNN)
- **Top-1 Overall Accuracy**: **85.41%**
- **Balanced Accuracy**: **85.27%**
- **Matthews Correlation Coefficient (MCC)**: **0.827**
- **Cohen's Kappa ($\kappa$)**: **0.823**
- **mAP@0.50 / mAP@0.50:0.95**: **89.30% / 62.00%**
- **Speed & Latency**: **34.2 FPS (29.2 ms)**

| Class | Instances | Class Acc (%) | Precision (%) | Recall/Sens (%) | Specificity (%) | F1-Score (%) | False Alarm (%) | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Airplane | 2488 | 96.69% | 92.70% | 89.83% | 98.32% | 91.24% | 1.68% | 0.892 |
| Bird | 2538 | 94.03% | 85.71% | 83.45% | 96.61% | 84.57% | 3.39% | 0.809 |
| Helicopter | 2453 | 95.71% | 90.48% | 86.42% | 97.87% | 88.41% | 2.13% | 0.858 |
| Kite | 1425 | 96.57% | 85.31% | 83.16% | 98.23% | 84.22% | 1.77% | 0.823 |
| Parachute | 1425 | 97.15% | 88.32% | 85.40% | 98.60% | 86.84% | 1.40% | 0.853 |
| UAV (Drone) | 2617 | 90.68% | 73.87% | 83.38% | 92.53% | 78.33% | 7.47% | 0.726 |
| **Overall (Macro)** | **12946** | **95.14%** | **86.07%** | **85.27%** | **97.03%** | **85.60%** | **2.97%** | **0.827** |

### Technique 2 (YOLO12 + LSTM)
- **Top-1 Overall Accuracy**: **87.25%**
- **Balanced Accuracy**: **87.10%**
- **Matthews Correlation Coefficient (MCC)**: **0.849**
- **Cohen's Kappa ($\kappa$)**: **0.845**
- **mAP@0.50 / mAP@0.50:0.95**: **90.80% / 64.00%**
- **Speed & Latency**: **32.8 FPS (30.5 ms)**

| Class | Instances | Class Acc (%) | Precision (%) | Recall/Sens (%) | Specificity (%) | F1-Score (%) | False Alarm (%) | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Airplane | 2488 | 97.04% | 93.76% | 90.64% | 98.57% | 92.17% | 1.43% | 0.904 |
| Bird | 2538 | 94.88% | 88.28% | 85.19% | 97.24% | 86.71% | 2.76% | 0.836 |
| Helicopter | 2453 | 96.32% | 92.08% | 88.18% | 98.23% | 90.09% | 1.77% | 0.879 |
| Kite | 1425 | 97.06% | 87.89% | 85.05% | 98.55% | 86.45% | 1.45% | 0.848 |
| Parachute | 1425 | 97.55% | 90.32% | 87.09% | 98.85% | 88.67% | 1.15% | 0.873 |
| UAV (Drone) | 2617 | 91.65% | 75.69% | 86.47% | 92.96% | 80.72% | 7.04% | 0.757 |
| **Overall (Macro)** | **12946** | **95.75%** | **88.00%** | **87.10%** | **97.40%** | **87.47%** | **2.60%** | **0.849** |

### Technique 3 (YOLO12 + Transformer)
- **Top-1 Overall Accuracy**: **90.03%**
- **Balanced Accuracy**: **89.80%**
- **Matthews Correlation Coefficient (MCC)**: **0.882**
- **Cohen's Kappa ($\kappa$)**: **0.879**e4
- **mAP@0.50 / mAP@0.50:0.95**: **93.00% / 67.90%**
- **Speed & Latency**: **28.5 FPS (35.1 ms)**

| Class | Instances | Class Acc (%) | Precision (%) | Recall/Sens (%) | Specificity (%) | F1-Score (%) | False Alarm (%) | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Airplane | 2488 | 97.80% | 95.57% | 92.85% | 98.98% | 94.19% | 1.02% | 0.928 |
| Bird | 2538 | 96.15% | 91.67% | 88.42% | 98.04% | 90.01% | 1.96% | 0.877 |
| Helicopter | 2453 | 97.15% | 94.23% | 90.50% | 98.70% | 92.33% | 1.30% | 0.906 |
| Kite | 1425 | 97.69% | 91.09% | 87.58% | 98.94% | 89.30% | 1.06% | 0.880 |
| Parachute | 1425 | 98.05% | 92.77% | 89.19% | 99.14% | 90.95% | 0.86% | 0.899 |
| UAV (Drone) | 2617 | 93.22% | 79.13% | 90.26% | 93.97% | 84.33% | 6.03% | 0.803 |
| **Overall (Macro)** | **12946** | **96.68%** | **90.74%** | **89.80%** | **97.96%** | **90.18%** | **2.04%** | **0.882** |

### Technique 4 (YOLO12 + BiLSTM + Attention)
- **Top-1 Overall Accuracy**: **91.53%**
- **Balanced Accuracy**: **91.28%**
- **Matthews Correlation Coefficient (MCC)**: **0.900**
- **Cohen's Kappa ($\kappa$)**: **0.897**
- **mAP@0.50 / mAP@0.50:0.95**: **94.00% / 69.70%**
- **Speed & Latency**: **27.1 FPS (36.9 ms)**

| Class | Instances | Class Acc (%) | Precision (%) | Recall/Sens (%) | Specificity (%) | F1-Score (%) | False Alarm (%) | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Airplane | 2488 | 98.20% | 96.49% | 94.05% | 99.19% | 95.26% | 0.81% | 0.942 |
| Bird | 2538 | 96.77% | 93.44% | 89.83% | 98.46% | 91.60% | 1.54% | 0.896 |
| Helicopter | 2453 | 97.63% | 95.50% | 91.81% | 98.99% | 93.62% | 1.01% | 0.922 |
| Kite | 1425 | 98.04% | 92.89% | 88.98% | 99.16% | 90.90% | 0.84% | 0.898 |
| Parachute | 1425 | 98.35% | 94.30% | 90.53% | 99.32% | 92.37% | 0.68% | 0.915 |
| UAV (Drone) | 2617 | 94.08% | 80.94% | 92.47% | 94.48% | 86.32% | 5.52% | 0.829 |
| **Overall (Macro)** | **12946** | **97.18%** | **92.26%** | **91.28%** | **98.27%** | **91.68%** | **1.73%** | **0.900** |

### Technique 5: Proposed (YOLO12 + GAN + Transformer)
- **Top-1 Overall Accuracy**: **97.70%**
- **Balanced Accuracy**: **97.73%**
- **Matthews Correlation Coefficient (MCC)**: **0.973**
- **Cohen's Kappa ($\kappa$)**: **0.972**
- **mAP@0.50 / mAP@0.50:0.95**: **96.30% / 73.80%**
- **Speed & Latency**: **26.4 FPS (37.9 ms)**

| Class | Instances | Class Acc (%) | Precision (%) | Recall/Sens (%) | Specificity (%) | F1-Score (%) | False Alarm (%) | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Airplane | 2488 | 99.00% | 97.89% | 96.86% | 99.50% | 97.37% | 0.50% | 0.968 |
| Bird | 2538 | 99.07% | 97.45% | 97.79% | 99.38% | 97.62% | 0.62% | 0.970 |
| Helicopter | 2453 | 99.17% | 97.57% | 98.04% | 99.43% | 97.80% | 0.57% | 0.973 |
| Kite | 1425 | 99.42% | 97.34% | 97.40% | 99.67% | 97.37% | 0.33% | 0.970 |
| Parachute | 1425 | 99.64% | 98.32% | 98.39% | 99.79% | 98.35% | 0.21% | 0.981 |
| UAV (Drone) | 2617 | 99.11% | 97.75% | 97.86% | 99.43% | 97.80% | 0.57% | 0.972 |
| **Overall (Macro)** | **12946** | **99.23%** | **97.72%** | **97.73%** | **99.53%** | **97.72%** | **0.47%** | **0.973** |

