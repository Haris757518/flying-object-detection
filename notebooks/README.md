# LAVEE Colab notebooks

1. `01_YOLO12_Training_10Fold.ipynb`: clean training notebook from the user-supplied source. Original training code is unchanged; outputs are cleared and a complete Fold 1 packaging cell is added. Use only when training/resuming is needed.
2. `02_Detection_Tracking_Dashboard.ipynb`: installs dependencies, loads the supplied runtime bundle, exposes the actual source code, runs saved-model image detection, continuous video tracking, research tables, and the dashboard. Does not train.
3. `LAVEE_Dashboard_Colab_Bundle.zip`: upload once to Google Drive and edit the inference notebook's BUNDLE_PATH, or upload when prompted. Includes saved weights and demos for both detector families.

Open either .ipynb using Google Colab > File > Upload notebook. Training needs your own combined YOLO dataset; inference needs only the runtime bundle. YOLO12 is a Fold 1 demo package, not final all-data weights. GPU is recommended for training; the current crop-classifier runtime runs on CPU. Colab proxy streaming and physical webcam access may differ from the tested local app. Use the notebook's live video output if the proxy blocks WebSockets.

Validation: notebook Python code cells were syntax-checked with installation magics excluded, original training code cells compared exactly, ZIP contents/checksums verified, and packaged runtime used for inference checks locally. These notebooks have not been executed end to end in an actual Colab runtime.
