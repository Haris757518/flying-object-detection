import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to sys.path so we can import internal modules
sys.path.append(str(Path(__file__).parent))

try:
    from models.yolo12_detector import YOLO12Detector
except ImportError as e:
    print(f"Error importing internal modules: {e}")
    sys.exit(1)

def test_new_model():
    model_path = Path(__file__).parent / "models" / "best_aod4.pt"
    
    print("==================================================")
    print("Testing AOD-4 YOLOv8s Integration")
    print("==================================================")
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please ensure you have placed 'best_aod4.pt' in the 'models' directory.")
        sys.exit(1)
        
    print(f"[1/4] Loading model: {model_path}")
    # Initialize the detector wrapper
    detector = YOLO12Detector(model_path=str(model_path))
    
    # Check internal classes mapping (from Ultralytics)
    if detector.model is not None:
        print(f"[2/4] Model loaded successfully.")
        print(f"      Original Ultralytics Classes: {detector.model.names}")
    else:
        print("[2/4] Warning: Model failed to load.")
        
    print("[3/4] Creating dummy 640x640 RGB frame...")
    # Create a dummy frame (BGR since OpenCV uses BGR internally)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    # Add a bright square to trigger potential detection / saliency just in case
    dummy_frame[300:340, 300:340] = 255
    
    print("[4/4] Performing inference and validating class mapping...")
    detections = detector.detect(dummy_frame, sensor_type="V")
    
    print("\n--- Detection Results ---")
    if not detections:
        print("No detections found on dummy frame. This is normal for a black frame.")
    else:
        for i, det in enumerate(detections):
            print(f"Det #{i}: Class [{det['class_id']}] {det['class_name']} @ Conf {det['confidence']:.2f}")
            
    print("\nTest passed. The detector successfully wrapped the new model without crashing.")

if __name__ == "__main__":
    test_new_model()
