# Model and dataset assets

- Git LFS: `code/dashboard/models/best.pt`: six-class YOLOv8s detector.
- Git LFS: `code/dashboard/models/yolo12_fold1/*.pt`: matching YOLO12n detector, frozen ResNet18 and five classifier heads.
- Git LFS: `code/dashboard/research/classification_results/*_final.pt`: five older trained crop classifiers for the YOLOv8s family.
- `datasets/combined_dataset/`: active 55,409-image corrected dataset and labels.
- `notebooks/LAVEE_Dashboard_Colab_Bundle.zip`: portable, checksummed dashboard/model package. Extract its `dashboard/` into `code/dashboard/` when restoring a clone; the extracted bundle contains its own provenance manifest.
- `papers/pdf/` and `papers/supplemental/`: local supplied or officially downloaded PDFs. All thirty citations and download status remain in Git.

Do not substitute arbitrary pretrained weights for these six-class fine-tuned checkpoints. The class order and feature/head architectures must match. YOLO12 is Fold 1 demo weights. The assets remain present in the local D:\ai installation.

Restore the versioned model weights after cloning with `git lfs install` followed by `git lfs pull`. Dataset images and paper PDFs are local-only assets.
