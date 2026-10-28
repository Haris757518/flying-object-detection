# LAVEE · Current flying-object detection project

This is the current six-class dashboard with selectable YOLOv8s and YOLO12n detectors. Each family uses its own matching CNN, LSTM, Transformer, BiLSTM + Attention and cGAN-trained Transformer classifiers. Detection and Demo modes support images, videos, live tracking and browser camera input.

| Folder | Contents |
| --- | --- |
| `code/dashboard` | Current application and local trained weights |
| `code/training` | Earlier balanced-crop training scripts used for the YOLOv8s classifiers |
| `notebooks` | Current YOLO12 training and saved-model inference notebooks, plus earlier crop training notebooks |
| `papers` | Thirty-reference bibliography, available PDFs, download status and supplemental references |
| `datasets/combined_dataset` | Active corrected six-class YOLO dataset |
| `datasets/source_material` | Original dataset provenance, including previous merge; not an active training dataset |
| `datasets/source_archives` | Original dataset recovery archive |
| `documents/current` | Current implementation notes and verification evidence |
| `documents/old_reviews` | Earlier reviews and reports, retained as historical documents |
| `results` | Actual recorded detector/crop metrics, without the obsolete simulated benchmark files |

## Run locally

```powershell
cd D:\ai
python -m pip install -r requirements.txt
cd code\dashboard
python -m uvicorn app:app --host 127.0.0.1 --port 8765
```

Or double-click `Start_Dashboard.cmd`. Open http://127.0.0.1:8765/. Close an existing server on that port before launching another copy.

## Colab

Upload either notebook in `notebooks/` to Colab. Notebook 01 trains/evaluates YOLO12; notebook 02 loads saved models and runs inference without training. The local `LAVEE_Dashboard_Colab_Bundle.zip` includes all runtime weights and demos. Trained weights are versioned through Git LFS. After cloning, run `git lfs install` and `git lfs pull`. Dataset images, PDF binaries and the large portable bundle stay local; see `ASSETS.md`.

## Research scope

YOLO12 is a Fold 1 demo export. Its recorded ten-fold detector mAP and ground-truth crop-classifier F1 are different measurements. Full ten-fold end-to-end evaluation and final all-data fitting were disabled. The run used image-level fallback groups; recording-session independence is not established. LSTM/BiLSTM consume spatial image tokens, not actual flight trajectories. The older review documents describe superseded systems and do not establish current performance.
