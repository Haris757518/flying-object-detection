"""Master Runner & Evaluation Benchmark for Flying Object Detection
Using Hybrid Deep Learning Techniques (VIT Project Review 1)

Authors: Karan Chander S R, Haris K
Provides unified access to:
1. Dataset & 10-Fold CV Inspector
2. GAN Synthetic Data Augmentation
3. Technique 1: YOLO12 + CNN (Detection + Classification)
4. Technique 2: YOLO12 + LSTM (Detection + Motion Analysis)
5. Technique 3: YOLO12 + Transformer (Detection + Feature Representation)
6. Technique 4: YOLO12 + BiLSTM + Attention (Detection + Temporal Classification)
7. Technique 5: YOLO12 + GAN + Transformer (Complete Proposed Architecture)
8. End-to-End Live Camera & Risk Alert System (Green/Yellow/Red Geofence)
"""

import sys
import os
import subprocess
import argparse
import time

def print_banner():
    print("=" * 70)
    print("  FLYING OBJECT DETECTION USING HYBRID DEEP LEARNING TECHNIQUES")
    print("  AIRPORT & CRITICAL AIRSPACE SURVEILLANCE SYSTEM")
    print("=" * 70)

def show_menu():
    print("\nSelect a module to execute:")
    print("  [1] Dataset Status & 10-Fold Cross-Validation Summary")
    print("  [2] Run GAN Synthetic Data Augmentation on Training Split")
    print("  [3] Test Technique 1: YOLO12 + CNN (Fine-Grained Classification)")
    print("  [4] Test Technique 2: YOLO12 + LSTM (Trajectory Kinematics)")
    print("  [5] Test Technique 3: YOLO12 + Transformer (Feature Attention)")
    print("  [6] Test Technique 4: YOLO12 + BiLSTM + Attention (Temporal Dynamics)")
    print("  [7] Test Technique 5: YOLO12 + GAN + Transformer (Proposed Model)")
    print("  [8] LAUNCH DESKTOP UI APPLICATION (Webcam / Upload Video / Alerts HUD)")
    print("  [9] LAUNCH WEB DASHBOARD (Streamlit Web App)")
    print("  [10] Generate Research Paper Benchmark Tables & 300-DPI Plots")
    print("  [0] Exit")

def main():
    print_banner()
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        show_menu()
        choice = input("\nEnter choice [0-9]: ").strip()
        
    if choice == "1":
        print("\n--- DATASET & 10-FOLD STATUS ---")
        import pandas as pd
        kfold_csv = os.path.join("DS", "comb DSF", "kfold_splits", "kfold_manifest.csv")
        if os.path.exists(kfold_csv):
            df = pd.read_csv(kfold_csv)
            print(f"Total 70% Training Pool: {len(df)} images.")
            print("Distribution across 10 folds:")
            for f in range(10):
                cnt = (df["fold"] == f).sum()
                print(f"  Fold {f}: {cnt} images (10% Val / 90% Train)")
            test_file = os.path.join("DS", "comb DSF", "kfold_splits", "test_holdout_30.txt")
            if os.path.exists(test_file):
                with open(test_file) as f:
                    test_cnt = len(f.readlines())
                print(f"Total 30% Holdout Test Pool: {test_cnt} images.")
        else:
            print("K-Fold manifest not found. Run prepare_dataset_kfold.py first.")
            
    elif choice == "2":
        print("\n--- RUNNING GAN SYNTHETIC AUGMENTATION ---")
        subprocess.run([sys.executable, "gan_augmentation.py"])
        
    elif choice == "3":
        print("\n--- TECHNIQUE 1: YOLO12 + CNN ---")
        subprocess.run([sys.executable, "model_1_yolo_cnn.py"])
        
    elif choice == "4":
        print("\n--- TECHNIQUE 2: YOLO12 + LSTM ---")
        subprocess.run([sys.executable, "model_2_yolo_lstm.py"])
        
    elif choice == "5":
        print("\n--- TECHNIQUE 3: YOLO12 + TRANSFORMER ---")
        subprocess.run([sys.executable, "model_3_yolo_transformer.py"])
        
    elif choice == "6":
        print("\n--- TECHNIQUE 4: YOLO12 + BILSTM + ATTENTION ---")
        subprocess.run([sys.executable, "model_4_yolo_bilstm_attention.py"])
        
    elif choice == "7":
        print("\n--- TECHNIQUE 5: YOLO12 + GAN + TRANSFORMER ---")
        subprocess.run([sys.executable, "model_5_yolo_gan_transformer.py"])
        
    elif choice == "8":
        print("\n--- LAUNCHING DESKTOP UI APPLICATION ---")
        subprocess.run([sys.executable, "app_gui.py"])
        
    elif choice == "9":
        print("\n--- LAUNCHING STREAMLIT WEB DASHBOARD ---")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_streamlit.py"])
        
    elif choice == "10":
        print("\n--- GENERATING RESEARCH BENCHMARK ARTIFACTS ---")
        subprocess.run([sys.executable, "evaluate_detailed_results.py"])
        
    elif choice == "0":
        print("Exiting.")
        sys.exit(0)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
