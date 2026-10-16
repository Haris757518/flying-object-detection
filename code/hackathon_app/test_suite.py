"""
Automated Verification & Unit Test Suite for Drone Defense Pipeline.
Tests GAN Enhancer, Trajectory Transformer, YOLO Detector, ByteTrack, and Risk Engine.
"""

import unittest
import numpy as np
import torch
import cv2

from models.gan_enhancer import GANEnhancer, LightweightGANGenerator
from models.temporal_transformer import TrajectoryTransformer, SpatioTemporalBehaviorModel
from models.yolo12_detector import YOLO12Detector
from tracking.bytetrack import ByteTracker, STrack
from core.risk_engine import RiskEngine, ThreatLevel
from core.pipeline import DroneDefensePipeline


class TestDroneDefensePipeline(unittest.TestCase):

    def test_gan_generator(self):
        generator = LightweightGANGenerator(scale_factor=2)
        dummy_in = torch.rand(1, 3, 32, 32)
        out = generator(dummy_in)
        self.assertEqual(out.shape, (1, 3, 64, 64), "GAN output resolution mismatch.")
        print("[PASS] Test 1: GAN Super-Resolution shape validated.")

    def test_trajectory_transformer(self):
        model = TrajectoryTransformer(in_features=9, num_classes=5)
        dummy_track = torch.rand(2, 16, 9)
        logits = model(dummy_track)
        self.assertEqual(logits.shape, (2, 5), "Transformer logits shape mismatch.")
        print("[PASS] Test 2: Trajectory Transformer kinematics classification validated.")

    def test_bytetrack_association(self):
        tracker = ByteTracker()
        # Simulated 3 frames with 1 moving drone
        det1 = [{"bbox": (100.0, 100.0, 140.0, 140.0), "confidence": 0.85, "class_id": 2, "class_name": "DRONE"}]
        tracks1 = tracker.update(det1)
        self.assertEqual(len(tracks1), 1)
        track_id = tracks1[0].track_id

        # Next frame with displaced bbox
        det2 = [{"bbox": (105.0, 104.0, 145.0, 144.0), "confidence": 0.82, "class_id": 2, "class_name": "DRONE"}]
        tracks2 = tracker.update(det2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, track_id, "ByteTrack ID persistence failed.")
        print(f"[PASS] Test 3: ByteTrack ID persistence validated (Track ID #{track_id}).")

    def test_risk_engine(self):
        engine = RiskEngine(focal_length_px=800.0, no_fly_radius_m=30.0, warning_radius_m=100.0)
        
        # Test case 1: Distant bird -> LOW threat
        class DummyTrack:
            def __init__(self, class_name, bbox):
                self.class_name = class_name
                self.tlbr = bbox
                self.track_id = 1
                self.confidence = 0.9

        bird_track = DummyTrack("BIRD", (100.0, 100.0, 108.0, 108.0))
        kinematics_bird = {"speed": 0.5, "heading": 10.0}
        res_bird = engine.calculate_threat(bird_track, "NATURAL", kinematics_bird)
        self.assertEqual(res_bird["threat_level"], "LOW", "Bird was not classified as LOW threat.")

        # Test case 2: Close Drone Diving -> CRITICAL threat
        drone_track = DummyTrack("DRONE", (100.0, 100.0, 250.0, 250.0))  # Large close bounding box
        kinematics_drone = {"speed": 4.5, "heading": 90.0}
        res_drone = engine.calculate_threat(drone_track, "DIVING", kinematics_drone)
        self.assertIn(res_drone["threat_level"], ["HIGH", "CRITICAL"], "Close diving drone threat underestimated.")
        print(f"[PASS] Test 4: Risk Engine threat scoring validated (Bird: {res_bird['threat_level']}, Drone: {res_drone['threat_level']}).")

    def test_pipeline_end_to_end(self):
        pipeline = DroneDefensePipeline()
        dummy_frame = np.zeros((512, 640, 3), dtype=np.uint8)
        # Draw a synthetic drone spot
        cv2.circle(dummy_frame, (320, 256), 25, (255, 255, 255), -1)
        
        annotated, targets, highest = pipeline.process_frame(dummy_frame, sensor_type="IR")
        self.assertEqual(annotated.shape, (512, 640, 3), "Annotated frame dimensions mismatch.")
        print("[PASS] Test 5: Master DroneDefensePipeline frame loop validated.")

    def test_ml_workflow_steps_1_to_5(self):
        from core.ml_workflow import (
            DatasetLoader, DataPreprocessor, DataLeakageAuditor,
            CLAHEEnhancer, DatasetSplitter
        )
        # Step 1: Load synthetic dataset
        loader = DatasetLoader()
        samples = loader.generate_synthetic_dataset(samples_per_class=10, img_size=32)
        self.assertEqual(len(samples), 40)

        # Step 2: Preprocessing
        preprocessor = DataPreprocessor(target_size=(32, 32))
        tensor = preprocessor.to_tensor(preprocessor.preprocess_image(samples[0].image))
        self.assertEqual(tensor.shape, (3, 32, 32))

        # Step 3: Leakage Audit (inject 1 duplicate & 1 blank to verify detection)
        dup = samples[0]
        blank_img = np.zeros((32, 32, 3), dtype=np.uint8)
        from core.ml_workflow import ImageSample
        corrupted = ImageSample(image=blank_img, label=0, class_name="AIRPLANE")
        dirty_samples = list(samples) + [dup, corrupted]
        
        auditor = DataLeakageAuditor()
        cleaned, report = auditor.audit_dataset(dirty_samples)
        self.assertGreaterEqual(report["duplicates_removed"], 1)
        self.assertGreaterEqual(report["suspicious_removed"], 1)

        # Step 4: CLAHE
        clahe = CLAHEEnhancer()
        enhanced = clahe.apply_batch(cleaned[:5])
        self.assertEqual(len(enhanced), 5)

        # Step 5: Split & 10-Fold CV
        splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        tr, va, te = splitter.stratified_split(cleaned)
        self.assertGreater(len(tr), len(va))
        self.assertGreater(len(te), 0)

        folds = splitter.generate_k_folds(cleaned, k=5)
        self.assertEqual(len(folds), 5)
        print("[PASS] Test 6: 10-Step Workflow (Steps 1-5: Loader, Preprocess, Leakage, CLAHE, K-Fold) validated.")

    def test_ml_workflow_steps_6_to_10(self):
        from torch.utils.data import DataLoader
        from core.ml_workflow import (
            DatasetLoader, DataPreprocessor, GANTrainingAugmenter,
            PyTorchImageDataset, DroneClassifier, ModelTrainer,
            OverfitUnderfitDetector, AutoCorrectionTrainer, FinalEvaluator
        )
        loader = DatasetLoader()
        samples = loader.generate_synthetic_dataset(samples_per_class=12, img_size=32)
        preprocessor = DataPreprocessor(target_size=(32, 32))

        # Step 6: GAN Augmentation
        gan = GANTrainingAugmenter(device="cpu")
        augmented = gan.augment_train_set(samples)
        self.assertGreaterEqual(len(augmented), len(samples))

        # Step 7: Train classifier
        train_ds = PyTorchImageDataset(augmented[:30], preprocessor)
        val_ds = PyTorchImageDataset(samples[30:], preprocessor)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        model = DroneClassifier(num_classes=4)
        trainer = ModelTrainer(model, device="cpu", lr=1e-3)
        tr_loss, tr_acc = trainer.train_epoch(train_loader)
        va_loss, va_acc = trainer.evaluate(val_loader)
        self.assertIsInstance(tr_loss, float)

        # Step 8: Diagnostics
        detector = OverfitUnderfitDetector()
        dummy_hist = {"train_loss": [0.2], "val_loss": [0.9], "train_acc": [98.0], "val_acc": [65.0]}
        diag = detector.diagnose(dummy_hist)
        self.assertEqual(diag.status, "OVERFITTING")

        # Step 9: Auto-Correction
        corrector = AutoCorrectionTrainer(num_classes=4, device="cpu")
        rect_model, _, policy = corrector.retrain_with_corrections(train_loader, val_loader, diag, base_epochs=2)
        self.assertIn("Regularization", policy)

        # Step 10: Final Evaluation
        evaluator = FinalEvaluator(classes=['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER'])
        metrics = evaluator.evaluate_test_set(rect_model, val_loader)
        self.assertIn("test_accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)
        print("[PASS] Test 7: 10-Step Workflow (Steps 6-10: GAN, Train, Diagnose, Rectify, Final Eval) validated.")

    def test_5_hybrid_techniques(self):
        from models.hybrid_techniques import get_all_5_hybrid_models
        models = get_all_5_hybrid_models(num_classes=4)
        self.assertEqual(len(models), 5)
        
        dummy_img = torch.rand(2, 3, 64, 64)
        for m in models:
            out = m(dummy_img)
            self.assertEqual(out.shape, (2, 4), f"Failed output shape for {m.technique_name}")
            self.assertTrue(hasattr(m, "technique_name"))
            self.assertTrue(hasattr(m, "application"))
        print("[PASS] Test 8: All 5 Hybrid Techniques (YOLO12+CNN, YOLO12+LSTM, YOLO12+Transformer, YOLO12+BiLSTM+Attn, YOLO12+GAN+Transformer) validated.")


if __name__ == "__main__":
    unittest.main()


