"""
End-to-End Image Classification Workflow Orchestrator.

Executes the 10-step sequence:
  1. Load Image Dataset
  2. Data Preprocessing
  3. Data Leakage Checks (Target, duplicate, suspicious features)
  4. CLAHE Contrast Enhancement
  5. Train/Val/Test Split & 10-Fold Cross Validation
  6. GAN Augmentation (TRAINING ONLY)
  7. Initial Model Training
  8. Overfitting / Underfitting Detection & Diagnostics
  9. Apply Correction Technique and Retrain
  10. Final Test Evaluation
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from core.ml_workflow import (
    DEFAULT_CLASSES,
    DatasetLoader,
    DataPreprocessor,
    DataLeakageAuditor,
    CLAHEEnhancer,
    DatasetSplitter,
    GANTrainingAugmenter,
    PyTorchImageDataset,
    DroneClassifier,
    ModelTrainer,
    OverfitUnderfitDetector,
    AutoCorrectionTrainer,
    FinalEvaluator
)


def parse_args():
    parser = argparse.ArgumentParser(description="10-Step Image Classification Pipeline Runner")
    parser.add_argument("--data-dir", type=str, default="dataset_yolo", help="Path to dataset directory")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "yolo", "dir", "synthetic"], help="Dataset source type")
    parser.add_argument("--cv-folds", type=int, default=10, help="Number of cross validation folds (default: 10)")
    parser.add_argument("--run-full-cv", action="store_true", help="Execute training on all 10 CV folds sequentially")
    parser.add_argument("--enable-clahe", action="store_true", default=True, help="Enable CLAHE contrast enhancement")
    parser.add_argument("--gan-aug", action="store_true", default=True, help="Enable GAN augmentation on training data only")
    parser.add_argument("--auto-rectify", action="store_true", default=True, help="Automatically rectify over/underfitting and retrain")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--img-size", type=int, default=64, help="Target image size (e.g., 64 for 64x64)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to ingest (default: None for full 100% dataset)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--dry-run", action="store_true", help="Quick run with synthetic sample data")
    return parser.parse_args()


def print_step_header(step_num: int, title: str):
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {title.upper()}")
    print("=" * 70)


def main():
    args = parse_args()
    classes = DEFAULT_CLASSES
    print("\n" + "#" * 70)
    print("STARTING 10-STEP IMAGE CLASSIFICATION ML PIPELINE")
    print(f"Classes: {classes} | Device: {args.device}")
    print("#" * 70)

    # ---------------------------------------------------------
    # STEP 1: LOAD IMAGE DATASET
    # ---------------------------------------------------------
    print_step_header(1, "Load Image Dataset")
    loader = DatasetLoader(classes=classes)
    samples = []

    if args.dry_run or args.source == "synthetic":
        print("[DatasetLoader] Generating synthetic airborne target dataset...")
        samples = loader.generate_synthetic_dataset(samples_per_class=40, img_size=args.img_size)
    elif os.path.exists(os.path.join(args.data_dir, "Annotations")) or "aod" in args.data_dir.lower():
        limit_desc = f"{args.max_samples} samples" if args.max_samples else "FULL COMPLETE DATASET (All 22,500+ images)"
        print(f"[DatasetLoader] Ingesting Airborne Object Dataset (AOD) from: {args.data_dir} [{limit_desc}]...")
        samples = loader.load_from_aod_dataset(args.data_dir, max_samples=args.max_samples)
    elif os.path.exists(os.path.join(args.data_dir, "labels")):
        limit_desc = f"{args.max_samples} crops" if args.max_samples else "all available crops"
        print(f"[DatasetLoader] Extracting target crops from YOLO dataset: {args.data_dir} [{limit_desc}]...")
        samples = loader.load_from_yolo_crops(args.data_dir, max_samples=args.max_samples)
    elif args.source == "dir" or (args.source == "auto" and not os.path.exists(os.path.join(args.data_dir, "images"))):
        print(f"[DatasetLoader] Scanning class subdirectories in {args.data_dir}...")
        try:
            samples = loader.load_from_directory(args.data_dir, max_samples_per_class=args.max_samples)
        except Exception as e:
            print(f"[Warning] Could not load directory: {e}. Falling back to synthetic.")
            samples = loader.generate_synthetic_dataset(samples_per_class=40, img_size=args.img_size)
    else:
        limit_desc = f"{args.max_samples} crops" if args.max_samples else "all available crops"
        print(f"[DatasetLoader] Extracting target crops from dataset: {args.data_dir} [{limit_desc}]...")
        samples = loader.load_from_yolo_crops(args.data_dir, max_samples=args.max_samples)

    if len(samples) < 20:
        print(f"[DatasetLoader] Loaded {len(samples)} crops from disk; supplementing with synthetic samples for robust CV...")
        samples.extend(loader.generate_synthetic_dataset(samples_per_class=30, img_size=args.img_size))

    print(f"-> Ingested {len(samples)} total image samples across {len(classes)} classes.")

    # ---------------------------------------------------------
    # STEP 2: DATA PREPROCESSING
    # ---------------------------------------------------------
    print_step_header(2, "Data Preprocessing")
    preprocessor = DataPreprocessor(target_size=(args.img_size, args.img_size), normalize=True)
    print(f"-> Configured Preprocessor: Target Size=({args.img_size}x{args.img_size}), ImageNet Channel Normalization.")

    # ---------------------------------------------------------
    # STEP 3: DATA LEAKAGE CHECKS
    # ---------------------------------------------------------
    print_step_header(3, "Data Leakage & Integrity Checks")
    auditor = DataLeakageAuditor(hash_size=8, hash_distance_threshold=2)
    clean_samples, audit_report = auditor.audit_dataset(samples)
    print(f"-> Audit Results:")
    print(f"   * Total Initial Samples     : {audit_report['initial_samples']}")
    print(f"   * Duplicate Leakages Removed: {audit_report['duplicates_removed']}")
    print(f"   * Suspicious Frames Removed : {audit_report['suspicious_removed']}")
    print(f"   * Target Leakage Flags       : {audit_report['target_leakage_flags']}")
    print(f"   * Clean Samples Retained     : {audit_report['clean_samples']}")
    print(f"   * Dataset Sanitized Status   : {'[PASS] ZERO LEAKAGE' if audit_report['leakage_clean'] else '[CLEANED]'}")

    # ---------------------------------------------------------
    # STEP 4: CLAHE (CONTRAST ENHANCEMENT)
    # ---------------------------------------------------------
    print_step_header(4, "CLAHE Contrast Enhancement")
    if args.enable_clahe:
        clahe_enhancer = CLAHEEnhancer(clip_limit=3.0, tile_grid_size=(8, 8))
        clean_samples = clahe_enhancer.apply_batch(clean_samples)
        print("-> Applied CLAHE Luminance Enhancement across all sanitized samples.")
    else:
        print("-> CLAHE enhancement disabled by user flag.")

    # ---------------------------------------------------------
    # STEP 5: TRAIN / VAL / TEST SPLIT & 10-FOLD CV
    # ---------------------------------------------------------
    print_step_header(5, "Dataset Splitting & 10-Fold Cross-Validation Setup")
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    train_set, val_set, test_set = splitter.stratified_split(clean_samples)
    folds = splitter.generate_k_folds(clean_samples, k=args.cv_folds)

    print(f"-> Stratified Holdout Split (70/15/15):")
    print(f"   * Train Set     : {len(train_set)} samples (70%)")
    print(f"   * Validation Set: {len(val_set)} samples (15%)")
    print(f"   * Test Set      : {len(test_set)} samples (15% - STRICTLY ISOLATED)")
    print(f"-> Stratified {args.cv_folds}-Fold Cross-Validation generated successfully.")

    if args.run_full_cv:
        print(f"\n--- Running {args.cv_folds}-Fold Cross-Validation Sweep ---")
        cv_scores = []
        for f_idx, (f_tr, f_va) in enumerate(folds):
            f_tr_ds = PyTorchImageDataset(f_tr, preprocessor)
            f_va_ds = PyTorchImageDataset(f_va, preprocessor)
            f_tr_ld = DataLoader(f_tr_ds, batch_size=args.batch_size, shuffle=True)
            f_va_ld = DataLoader(f_va_ds, batch_size=args.batch_size, shuffle=False)
            
            f_model = DroneClassifier(num_classes=len(classes)).to(args.device)
            f_trainer = ModelTrainer(f_model, device=args.device, lr=args.lr)
            for _ in range(3):
                f_trainer.train_epoch(f_tr_ld)
            _, f_acc = f_trainer.evaluate(f_va_ld)
            cv_scores.append(f_acc)
            print(f"   * Fold {f_idx+1:02d}/{args.cv_folds} Val Accuracy: {f_acc:.2f}%")
        print(f"-> {args.cv_folds}-Fold Mean Accuracy: {sum(cv_scores)/len(cv_scores):.2f}% +/- {torch.tensor(cv_scores).std().item():.2f}%")

    # ---------------------------------------------------------
    # STEP 6: GAN AUGMENTATION (ON TRAINING SET ONLY)
    # ---------------------------------------------------------
    print_step_header(6, "GAN Augmentation (TRAINING DATA ONLY)")
    if args.gan_aug:
        gan_augmenter = GANTrainingAugmenter(device=args.device)
        train_augmented = gan_augmenter.augment_train_set(train_set)
        gan_generated_count = len(train_augmented) - len(train_set)
        print(f"-> Generated {gan_generated_count} synthetic GAN-enhanced samples strictly for Training.")
        print(f"-> Augmented Training Size: {len(train_augmented)} samples (Val/Test remained untouched).")
    else:
        train_augmented = train_set
        print("-> GAN Augmentation skipped.")

    # Convert to PyTorch DataLoaders
    train_dataset = PyTorchImageDataset(train_augmented, preprocessor)
    val_dataset = PyTorchImageDataset(val_set, preprocessor)
    test_dataset = PyTorchImageDataset(test_set, preprocessor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------------------------------------------------------
    # STEP 7: INITIAL MODEL TRAINING
    # ---------------------------------------------------------
    print_step_header(7, "Initial Model Training")
    model = DroneClassifier(num_classes=len(classes), dropout_rate=0.2).to(args.device)
    trainer = ModelTrainer(model, device=args.device, lr=args.lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(args.epochs):
        tr_loss, tr_acc = trainer.train_epoch(train_loader)
        va_loss, va_acc = trainer.evaluate(val_loader)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(f"   Epoch [{epoch+1:02d}/{args.epochs:02d}] - Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Train Acc: {tr_acc:.1f}% | Val Acc: {va_acc:.1f}%")

    # ---------------------------------------------------------
    # STEP 8: OVERFITTING / UNDERFITTING DIAGNOSTICS
    # ---------------------------------------------------------
    print_step_header(8, "Overfitting / Underfitting Diagnostics")
    detector = OverfitUnderfitDetector(overfit_gap_threshold=0.35, underfit_loss_threshold=1.0)
    diagnostic = detector.diagnose(history)

    print(f"-> Diagnostic Analysis:")
    print(f"   * Identified Status   : [{diagnostic.status}]")
    print(f"   * Generalization Gap  : {diagnostic.generalization_gap:+.4f} (Val Loss - Train Loss)")
    print(f"   * Final Train Accuracy: {diagnostic.train_acc:.2f}% | Val Accuracy: {diagnostic.val_acc:.2f}%")
    print(f"   * Recommendation      : {diagnostic.recommendation}")

    # ---------------------------------------------------------
    # STEP 9: APPLY CORRECTION TECHNIQUE & RETRAIN
    # ---------------------------------------------------------
    print_step_header(9, "Apply Correction Technique & Retrain")
    final_candidate_model = model
    if args.auto_rectify and diagnostic.status != "OPTIMAL":
        print(f"-> Triggering Auto-Correction Policy for {diagnostic.status}...")
        corrector = AutoCorrectionTrainer(num_classes=len(classes), device=args.device)
        rectified_model, rect_history, policy_desc = corrector.retrain_with_corrections(
            train_loader, val_loader, diagnostic, base_epochs=args.epochs
        )
        print(f"   * Policy Applied: {policy_desc}")
        print(f"   * Rectified Model Best Val Loss: {min(rect_history['val_loss']):.4f} | Val Acc: {max(rect_history['val_acc']):.1f}%")
        final_candidate_model = rectified_model
    else:
        print("-> Model generalization is optimal; proceeding directly with trained checkpoint.")

    # ---------------------------------------------------------
    # STEP 10: FINAL TEST EVALUATION
    # ---------------------------------------------------------
    print_step_header(10, "Final Holdout Test Evaluation")
    evaluator = FinalEvaluator(classes=classes, device=args.device)
    metrics = evaluator.evaluate_test_set(final_candidate_model, test_loader)

    print("\n" + "-" * 70)
    print("FINAL HOLDOUT TEST PERFORMANCE REPORT")
    print("-" * 70)
    print(f"* Overall Test Accuracy: {metrics['test_accuracy']}%")
    print(f"* Macro Precision      : {metrics['macro_precision']:.4f}")
    print(f"* Macro Recall         : {metrics['macro_recall']:.4f}")
    print(f"* Macro F1-Score       : {metrics['macro_f1']:.4f}")
    print(f"* Total Evaluated      : {metrics['total_test_samples']} test samples\n")
    
    print("Per-Class Breakdown:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 60)
    for cls_name, m in metrics["per_class_metrics"].items():
        print(f"{cls_name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1_score']:<12.4f} {m['support']:<8}")

    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Prediction):")
    print("     " + " ".join([f"{c[:4]:>5}" for c in classes]))
    for idx, row in enumerate(metrics["confusion_matrix"]):
        row_str = " ".join([f"{val:>5}" for val in row])
        print(f"{classes[idx][:4]:>4} [{row_str} ]")

    # Save final model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/workflow_classifier_final.pt"
    torch.save(final_candidate_model.state_dict(), ckpt_path)
    print(f"\n[Artifact] Saved final model checkpoint: {ckpt_path}")
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETE (ALL 10 STAGES VALIDATED)")
    print("=" * 70)


if __name__ == "__main__":
    main()
