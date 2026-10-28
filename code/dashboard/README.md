# LAVEE detection dashboard

Run `./start.ps1` in PowerShell, then open http://127.0.0.1:8765.
If needed, first install dependencies with `python -m pip install -r requirements.txt`.

Uses the supplied six-class YOLOv8s checkpoint: airplane, bird, helicopter,
kite, parachute, UAV. `models/args.yaml` records training arguments; it is
not a dataset class configuration. Class names come from the checkpoint.

Video mode now streams annotated frames immediately over WebSocket while
inference runs. ByteTrack maintains IDs across consecutive frames and the
viewer draws the last 45 observed center points as movement trails. A new
tracker is created for every video, preventing ID state leaking between
uploads. Pause/resume preserves tracker state; Stop ends processing and
saves a recording of the processed portion. The live table shows current
boxes, class labels, confidences and IDs. Inference speed depends on the
hardware and may be slower than the source video's playback speed.
Tracking IDs can change after long occlusion or a missed detection.

Upload images (20 MB maximum) or videos (100 MB, 2 minutes maximum).
Video inference processes every decoded frame, resizes frames to at most
960 pixels on the longest side, and runs YOLO at 640 pixels. The output
is an annotated H.264 MP4 without audio. CPU processing may take longer
than playback. Video summaries count frames with each detected class,
not unique objects. Confidence is the peak score for each class.

Four demos are real video excerpts from the local project's demo pack.
Kite and parachute are explicitly labeled still-image clips because no
real videos for these classes were available. Replace `demos/3.mp4` and
`demos/4.mp4` with real videos and update their `video_kind` entries in
`demos/manifest.json` when available. Source provenance is in that manifest.
Curated demos are demonstrations, not an independent accuracy evaluation.

The demo gallery offers two clips under every class and two Mixed classes
clips (14 total). Extra real-motion class clips are distinct excerpts of
the same source recordings, not independent recording sessions. Kite and
parachute alternatives use different still dataset images. Mixed demos
are explicitly labeled four-camera mosaics combining airplane, bird,
helicopter and drone recordings simultaneously; these are not footage of
all four objects in one shared physical scene. Choose a clip using each
card's selector, then press Start live tracking.

All predictions come from the supplied model. No forced labels or fake
boxes are used. Detection errors are possible. This dashboard adds no
10-fold, cGAN, or classifier training to the checkpoint.

## Imported classification results

The supplied `classification_results.zip` is copied under
`research/classification_results/`. It contains final ResNet18 and
DeiT-Tiny extractor/head checkpoints, 30 fold classifier heads, 11 cGAN
generator checkpoints, generated sample grids, losses, fold assignments
and metrics. No LSTM or BiLSTM checkpoints are present.

Six-class LSTM and BiLSTM + Attention loaders are prepared but stay
disabled until compatible trained weights are installed. Use
`balanced_research/LSTM_BiLSTM_Six_Class_Colab.ipynb` with the same balanced
dataset. These models operate on a 4 x 4 spatial grid of frozen ResNet18
features (16 tokens, 512 features each), not a temporal motion sequence.
Expected files are `LSTM_final.pt` and `BiLSTM_Attention_final.pt`, copied
from the resulting recurrent_results archive into
`research/classification_results/`. Keep the recurrent metrics as separate
evidence and merge their method entries into metrics.json when importing.
Restart the server after importing weights. The selector availability is
read from the model metadata endpoint. Older four-class spatial checkpoint
formats are rejected; random fallback weights are never used.

Select YOLO only, YOLO + CNN, YOLO + Transformer, or YOLO + cGAN-trained
Transformer in Detection settings. YOLO continues to determine boxes
and track associations. The classifier predicts each detected crop's
label using the same 8% total context, 224-pixel aspect-preserving padding,
and ImageNet normalization as training. Both classifier and detector labels
are shown. Exported JSON preserves both confidences. The confidence slider
filters YOLO proposals, not classifier probabilities. Classifiers are
loaded lazily from local weights with no pretrained-weight download.

cGAN is training augmentation only; live frames are not synthesized or
enhanced by the GAN. Saved crop test accuracies are CNN 81.0%, Transformer
82.0%, and cGAN + Transformer 79.67%. The ten-fold macro-F1 means are
88.70%, 89.33%, and 88.32% respectively. These measure classification on
ground-truth crops, not the end-to-end detection/tracking pipeline.

Results and temporary uploaded video data are stored in `results/`.
Uploaded sources are removed after processing; output videos and JSON
remain locally. The server binds only to localhost. Image uploads stay
in memory. The interface works with system fonts when offline.


## Recurrent models installed

Imported recurrent_results.zip with matching dataset fingerprint, complete
ten-fold results, and six-class LSTM and BiLSTM + Attention final weights.
Both options are now active for image upload, video upload and live demos.
They process spatial feature tokens from individual crops, not temporal
flight sequences. Saved test accuracy: LSTM 77.83%, BiLSTM + Attention
81.50%; mean grouped CV macro-F1: 89.08% and 90.97% respectively.
Both checkpoint loading and all six dashboard options passed image/live
stream inference checks. Original recurrent metrics and folds are retained
under research/recurrent_results; their metrics are also merged into the
dashboard training-results panel.
# Two selectable detector families (27 October 2026)

The Detection model selector keeps both research runs available. It switches the actual detector, matching crop classifiers, preprocessing, inference size, and displayed research results. No weights are shared between the two runs. Existing uploads and all demo videos work with either selection.

| Selection | Detector | Matching classifiers | Runtime image size |
| --- | --- | --- | --- |
| YOLOv8s | Original `models/best.pt` | Original five final classifiers in `research/classification_results` | 640 |
| YOLO12n | `models/yolo12_fold1/yolo12_best.pt` | Matching frozen ResNet18 + five Fold 1 heads | 416 |

YOLO12 weights were imported from the user's Drive `deadline_9af342f18d34/fold1_demo_export`. They are Fold 1 demo weights, not a final model fitted on all development images. `run_config.json`, `cv_progress.json`, and `cv_summary.csv` accompany the package. `validated.json` records checksums and provenance. All five head definitions match the supplied deadline notebook; every head consumes 16 spatial tokens × 512 ResNet18 features. CNN is a Conv1d head, Transformer uses a two-layer token encoder, and LSTM/BiLSTM process spatial tokens. This differs from the older YOLOv8s classifiers, whose CNN/Transformer use different backbones. Crop resize and tight bounding boxes also follow the new notebook.

The recorded YOLO12 experiment has ten folds, detector epochs=5, classifier epochs=4, GAN epochs=2, and image-level fallback splits (`image_level_fallback_source_unknown`). Recording/session independence is not established. The dashboard shows YOLO detection mAP and ground-truth crop F1 separately, with actual fold values. It does not present these as ten-fold end-to-end metrics or a locked test result.

Live camera support uses browser camera permission and streams JPEGs to the local server with one outstanding frame at a time. Each connection gets its own ByteTrack state. Stop camera releases the camera and closes the connection. The automated test exercises the camera protocol with sample JPEGs; actual camera hardware/permission was not activated during verification.

Validation: `python verify_families.py` passed all 12 detector/technique combinations for actual image upload, live video frames and recording completion, plus camera-stream inference. All new checkpoint state dictionaries loaded strictly with the verified six-class order. This checks integration, not dataset-wide detection accuracy. `family-verification.json` stores the observed predictions. The original server command is unchanged.

