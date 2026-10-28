# Detection and class demos

Run `python app_gui.py` for the desktop app or
`python -m streamlit run app_streamlit.py` for the web dashboard.

Load your trained six-class YOLO detection `.pt` checkpoint using the model
selector. You can also place it at `models/best.pt` or set the environment
variable `FLYING_OBJECT_WEIGHTS` to its absolute path. A single checkpoint at
`runs/**/weights/best.pt` is discovered automatically. Multiple checkpoints
require explicit selection. Both applications resolve default paths relative
to this project, regardless of the terminal's current directory.

The checkpoint must provide airplane, bird, helicopter, kite, parachute and
UAV/drone labels. Class IDs come from the checkpoint's own names, so differing
class orders work correctly. The supplied `yolov8n.pt` is a general COCO model;
it does not contain all six classes and is deliberately rejected.

Choose any of the six **Class demos** in the web app, or choose a class and
click **Show selected class demo** in the desktop app. Each uses a real sample
and its YOLO annotation from `Review_2_Evidence/08_Sample_Images`. The reference
panel works without model weights. Predictions require the trained model and
run across all classes, regardless of the demo selection. Missing detections
and wrong predictions remain visible. These existing dataset samples are not
an independent accuracy benchmark.

Inference preserves model labels and confidence values. It no longer assigns
classes using aspect ratio or sky position, invents drones from unrelated
COCO classes, boosts confidence, or runs a randomly initialized Transformer
as a classifier. Class-agnostic suppression removes overlapping competing
boxes; two nearby objects with separate boxes remain separate. Raw images
are the default input. Optional desktop contrast enhancement can change
predictions and should be compared with the raw input on your footage.

The default confidence threshold is 0.45. Raising it reduces weak predictions
but can miss small objects. A confidently incorrect model label still needs
evaluation and potentially improved training data or retraining. No setting
guarantees perfect recognition. The trained detector checkpoint was not found
in the supplied project during this fix, so six-class model accuracy could not
be measured.

Run regression checks with `python -m unittest discover -s tests -v`.
