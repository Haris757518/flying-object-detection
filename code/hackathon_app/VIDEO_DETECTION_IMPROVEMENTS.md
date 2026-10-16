# Video Detection / Classification Improvements

## What changed

1. **4-Class Demo is preserved.** The existing Airplane/Bird/Drone/Helicopter demo selector is unchanged.
2. **Uploaded videos now default to Visible sensor mode** and preserve the uploaded file extension.
3. **Visible-video false DRONE fallback fixed.** Thermal saliency is now used only when the sensor is explicitly IR.
4. **Temporal class voting added.** Each tracked target keeps recent class evidence and uses confidence-weighted voting, so a target does not flip class every frame.
5. **Tracker resets for every new video.** Cached Streamlit state no longer carries old tracks into a different video.
6. **Upload inference resolution defaults to 960.** 640 / 960 / 1280 are selectable; higher resolution helps small distant aerial targets at the cost of speed.
7. **Uploaded-video classification summary added.** The UI shows a running and final dominant class using stable tracked detections over time.
8. **GAN option is marked experimental and defaults off.** In the current architecture it does not feed its enhanced crop back into YOLO, so it should not be presented as an accuracy boost.

## Recommended settings for uploaded videos

- Ordinary RGB/phone/CCTV video: **Sensor = Visible**
- Thermal/infrared video: **Sensor = IR**
- Start with **Confidence = 0.30**, **IoU = 0.45**, **Resolution = 960**
- For a very small/distant object: try **1280** before lowering confidence.
- If detections are missed: lower confidence slowly to **0.25**, then **0.20**. Very low thresholds create false positives.

## Biggest remaining accuracy limitation

Inference changes improve stability, but they cannot make the model recognize visual conditions it did not learn during training. For reliable arbitrary-video classification, improve/retrain the detector itself.

The included demo report already shows that **BIRD is the weakest demo class** (much less consistent than Drone/Airplane/Helicopter). That is a model/data issue, not only a dashboard issue.

## Best retraining plan

Train one end-to-end detector on full frames and split the data **by source video**, not by extracted frame. Frames from the same video must not appear in both train and validation/test sets.

Recommended starting point:

```bash
yolo detect train model=yolov8s.pt data=PATH_TO_DATA_YAML epochs=80 imgsz=960 batch=-1 patience=20 mosaic=1.0 mixup=0.10 close_mosaic=10
```

Then fine-tune/evaluate with:

- More **Bird vs Drone hard examples**
- More small/far-away objects
- Sky-only / cloud / insect / tree / pole hard-negative images
- Motion blur and compression artifacts from real video
- Low-light, haze, scale and exposure variation
- Visible and IR kept as separate domains unless the training set clearly labels/represents both

Measure **per-class precision, recall, confusion matrix, mAP50 and mAP50-95**. Do not choose a model only from overall accuracy.

## Important next improvement if you want maximum video reliability

Add a held-out validation set made from entire videos and tune class-specific confidence thresholds from that set. Example: if BIRD has lower scores than DRONE, one global threshold can unfairly suppress birds. The threshold values should come from validation statistics rather than guesses.
