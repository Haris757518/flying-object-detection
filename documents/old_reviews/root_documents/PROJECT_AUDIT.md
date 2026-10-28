# 1. Current Architecture
The current system implements a complex, multi-stage hybrid pipeline rather than a unified end-to-end object detector:
1. **Frame Ingestion**: Video frames are ingested and enhanced using a CLAHE (Contrast-Limited Adaptive Histogram Equalization) / GAN Enhancer.
2. **Localization**: A baseline YOLOv8 Nano (`yolov8n.pt`) is used to extract bounding boxes. If no detections occur in IR, a custom morphological thermal saliency detector is used as a fallback.
3. **Classification**: Bounding box crops are extracted, resized to 64x64, and classified using a custom Convolutional Neural Network (CNN) with residual skips (YOLO12 + CNN).
4. **Tracking**: ByteTrack associates detections across frames.
5. **Behavior Analysis**: A Kinematic Trajectory Transformer analyzes bounding box movement to classify behavior (HOVERING, APPROACHING, etc.).
6. **Threat Engine**: A rule-based risk engine scores the threat level.

# 2. Current Dataset
- **Dataset Name**: AOD 4 (Airborne Object Detection Benchmark)
- **Number of Images**: 22,516 total images
- **Classes**: 4 (`AIRPLANE`, `BIRD`, `DRONE`, `HELICOPTER`)
- **Annotations**: YOLOv8 Bounding Box format (`.txt`)
- **Split Sizes**: Train: 15,761 (70%), Valid: 4,514 (15%), Test: 2,241 (15%)
- **Class Distribution**: Internally balanced during the data ingestion pipeline using stratified sampling.

# 3. Current Training Configuration
*(For the Hybrid CNN Classifier, since YOLO is not fully trained in the workflow)*
- **Image Size**: 64x64 (for CNN classification crops); 640 (for YOLO baseline)
- **Batch Size**: 16 (default) / 32
- **Epochs**: 10
- **Learning Rate**: 1e-3 (initial), up to 2e-3 on underfit auto-correction
- **Optimizer**: AdamW
- **Scheduler**: None explicitly used (manual LR adjustments triggered by diagnostic engine)
- **Augmentation**: CLAHE luminance contrast enhancement, GAN super-resolution (training set only)
- **Pretrained Weights**: `yolov8n.pt` used for YOLO backbone
- **Loss Functions**: CrossEntropyLoss (with Label Smoothing optionally 0.1) for classifier, L1 Loss for GAN
- **Confidence Threshold**: 0.25 (YOLO)
- **IoU Threshold**: 0.45 (YOLO NMS)

# 4. Current Model
The system uses **YOLOv8 Nano** (`yolov8n.pt`) as the baseline detector. 
However, the actual classification is deferred to a **Custom Deep Convolutional Classifier** (3-layer CNN with residual skips, batch normalization, Adaptive Avg Pooling, and Dropout). The repository also includes a `TrajectoryTransformer` (Self-Attention) for motion and a `LightweightGANGenerator` for upscaling patches.

# 5. Current Results
*(Based on the AOD 4 Benchmark JSON for the best model: YOLO12 + CNN)*
- **Precision**: 0.7604
- **Recall**: 0.6821
- **mAP@50**: Not Reported (System measures crop classification accuracy, not mAP)
- **mAP@50:95**: Not Reported
- **F1**: 0.6415
- **Accuracy**: 67.80%
- **Per-Class Metrics**:
  - `AIRPLANE`: Precision 0.6875, Recall 0.7333, F1 0.7097
  - `BIRD`: Precision 0.8125, Recall 0.8667, F1 0.8387
  - `DRONE`: Precision 0.5417, Recall 0.9286, F1 0.6842
  - `HELICOPTER`: Precision 1.0000, Recall 0.2000, F1 0.3333

# 6. Main Accuracy Problems

**HIGH: Fragmented Multi-Stage Pipeline**
Instead of training a unified object detector (YOLO) to jointly predict bounding boxes and classes, the system relies on an untrained YOLOv8n to find generic objects, then passes small crops to a custom CNN. If the baseline YOLO misses the object, the CNN never sees it.

**HIGH: Extreme Downsampling of Crops (64x64)**
The CNN classifier resizes all crops to 64x64. Drones and distant aircraft are already small and lack features; downscaling further destroys crucial spatial information like rotors or wings, leading to poor classification (especially for Helicopters which have 20% recall).

**MEDIUM: Unnecessary GAN Complexity**
Applying a GAN to super-resolve 64x64 patches is a computationally expensive workaround for simply not extracting larger patches in the first place or using a higher resolution feature map.

# 7. Accuracy Improvement Opportunities

**A. Dataset improvements**
- *Expected Benefit*: High. 
- *Implementation*: Modify `prepare_dataset.py` to extract native resolution crops or train entirely on full-frame images. 
- *Risk*: Low.

**B. Preprocessing**
- *Expected Benefit*: Medium.
- *Implementation*: Remove 64x64 resizing constraint.
- *Risk*: Low.

**C. Augmentation**
- *Expected Benefit*: High.
- *Implementation*: Native Mosaic and Mixup augmentations inside Ultralytics.
- *Risk*: Low.

**D. Model architecture**
- *Expected Benefit*: VERY HIGH.
- *Implementation*: Abandon the 10-step hybrid crop-classifier workflow. Fine-tune a single YOLOv8/YOLO11 model end-to-end on the AOD 4 dataset.
- *Difficulty*: Low (Ultralytics is already installed).

**E. Hyperparameter tuning**
- *Expected Benefit*: High.
- *Implementation*: Train for 50 epochs instead of 10.
- *Risk*: Medium (Overfitting, but YOLO has strong native augmentations).

**F. Loss/function improvements**
- *Expected Benefit*: Medium.
- *Implementation*: Leverage YOLOv8 focal loss for hard-to-detect classes (like Helicopter).
- *Risk*: Low.

**G. Inference tuning**
- *Expected Benefit*: Medium.
- *Implementation*: Optimize NMS thresholds for small targets.
- *Risk*: Low.

**H. Temporal/video improvements**
- *Expected Benefit*: Low (for hackathon timeframe).
- *Implementation*: ByteTrack is fine, rely on it rather than adding new temporal models.
- *Risk*: High complexity for low short-term gain.

# 8. Hackathon Recommendation
**FASTEST realistic improvement plan (12 hours):**
Ditch the custom CNN, GAN, and Transformer modules for detection. We will fine-tune a native **YOLOv8s** (Small) model end-to-end directly on the AOD 4 bounding box dataset. This eliminates the multi-stage bottleneck, natively supports high-resolution inputs (640x640), inherently handles class imbalances via Focal Loss, and runs easily within a 12-hour Colab session.

# 9. Experiment Plan

| Experiment | Configuration | Unchanged |
| :--- | :--- | :--- |
| **Experiment 0** | Existing baseline (YOLO12 + CNN at 64x64, 10 epochs) | Data |
| **Experiment 1** | Native YOLOv8n end-to-end training (640 imgsz, 30 epochs) | Data |
| **Experiment 2** | Native YOLOv8s end-to-end training (640 imgsz, 30 epochs) | Data |
| **Experiment 3** | YOLOv8s + heavy augmentation (Mosaic, MixUp) | Data |
| **Experiment 4** | Final YOLOv8s (50 epochs, optimized hyperparameters) | Data |

# 10. Final Recommendation
**Implement a unified YOLOv8s (Small) model trained end-to-end.** 
Do not recommend adding the Transformer + GAN + LSTM. A well-trained YOLOv8s will vastly outperform a complex hybrid pipeline that forces crops into 64x64 resolution, and it can be easily trained on Google Colab in under 3 hours.
