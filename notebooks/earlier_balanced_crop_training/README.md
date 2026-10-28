# Balanced six-class Colab research kit

Open `Balanced_Six_Class_Training.ipynb` in Google Colab, select a GPU,
then upload `balanced_six_class.zip` when the notebook prompts you.
No access to the large original dataset or detector checkpoint is needed.

The dataset contains exactly 500 training, 100 validation and 100 test
ground-truth crops per class: airplane, bird, helicopter, kite, parachute,
drone. The original class 5 name UAV is displayed as drone. All crops are
224 x 224, aspect-preserving with padding. Small/invalid boxes and exact
duplicate crop pixels are excluded. One crop is selected per source image.
The original source splits and group IDs are retained in `manifest.csv`.

The notebook trains:

1. Frozen pretrained ResNet18 features + a trained linear classifier.
2. Frozen pretrained DeiT-Tiny features + a trained linear classifier.
3. A conditional GAN trained on real training crops, followed by the same
   frozen DeiT + linear classifier using real and generated crop features.

Feature extraction is cached. Ten grouped folds are created only within
the training split. Each fold initializes a fresh head; every cGAN fold
trains a fresh generator without its held-out fold. Folds are approximately
class-balanced because source groups must stay intact. The base dataset
splits themselves have exactly equal class counts. Final heads train on
the complete training split, with real validation/test scores reported
separately. Do not tune settings based on test scores. These are limited
budget experiments, not claims of convergence or state-of-the-art results.

Run the one-fold pilot first. It does not count as completed 10-fold CV.
Inspect `generated_samples.jpg` and GAN losses: a short cGAN run may produce
poor samples, so improved classifier performance is not guaranteed.
The complete run saves all fold assignments, per-fold scores, means and
standard deviations, confusion matrices and deployment weights.

These methods classify ground-truth crops. YOLO detection/crop integration
can be done once trained weights are available. They are not implemented
YOLO12 feature-fusion architectures and do not validate a previously trained
YOLO detector. LSTM and BiLSTM motion models require a separate ordered
sequence dataset and are not included in this still-crop notebook.

Some original recording IDs are missing, limiting claims of scene/session
independence even though retained source group IDs never cross splits.
A balanced subset also differs from natural deployment class frequencies.
See `dataset.json` for exact counts and limitations.

Visual spot checks include blurry crops and some ambiguous aircraft-like
kite examples inherited from source annotations. These labels have not
been manually verified for all 4,200 crops. Inspect the supplied preview
and review questionable source annotations before claiming a curated,
error-free dataset. No labels were silently changed using model guesses.

No Colab training has been run by the authoring assistant. The delivered
code was checked locally with random-weight feature extraction, classifier
fitting, and generator/discriminator forward/backward passes. Pretrained
weights download and complete GPU runs must be checked in Colab.
