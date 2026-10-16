"""
Drone Detection Dataset Preparation Script
Converts IR and Visible video datasets with MATLAB .mat groundTruth annotations
into standard machine learning formats: YOLO, COCO JSON, and CSV.
"""

import os
import sys
import glob
import json
import argparse
import random
import cv2
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from mcos_decoder import load_groundtruth

CLASSES = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Drone Detection Dataset for YOLO, COCO, and PyTorch")
    parser.add_argument("--data-dir", type=str, default="Data", help="Path to Data directory containing Video_IR, Video_V, Audio")
    parser.add_argument("--output-dir", type=str, default="dataset_yolo", help="Output directory for generated dataset")
    parser.add_argument("--format", type=str, default="yolo", choices=["yolo", "coco", "csv", "all"], help="Dataset export format")
    parser.add_argument("--sensor", type=str, default="all", choices=["all", "IR", "V"], help="Sensor modalities to include")
    parser.add_argument("--stride", type=int, default=5, help="Frame extraction stride (e.g., 1 for all frames, 5 for every 5th frame)")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--only-positives", action="store_true", help="Extract only frames containing a target object")
    parser.add_argument("--img-size", type=int, default=None, help="Resize extracted images to specific size (e.g., 640), optional")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for video-level split")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker processes for parallel frame extraction")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos processed (useful for testing)")
    return parser.parse_args()


def get_video_label_pairs(data_dir, sensor_filter="all", max_videos=None):
    pairs = []
    sensor_folders = []
    if sensor_filter in ["all", "IR"]:
        sensor_folders.append(("IR", os.path.join(data_dir, "Video_IR")))
    if sensor_filter in ["all", "V"]:
        sensor_folders.append(("V", os.path.join(data_dir, "Video_V")))

    for sensor, folder in sensor_folders:
        if not os.path.exists(folder):
            print(f"[Warning] Folder not found: {folder}")
            continue
        mp4_files = sorted(glob.glob(os.path.join(folder, "*.mp4")))
        for mp4 in mp4_files:
            mat = mp4.replace(".mp4", "_LABELS.mat")
            if not os.path.exists(mat):
                continue
            base = os.path.splitext(os.path.basename(mp4))[0]
            parts = base.split("_")
            cls_name = parts[1].upper()
            if cls_name not in CLASS_TO_ID:
                continue
            pairs.append({
                "video_path": mp4,
                "label_path": mat,
                "video_id": base,
                "sensor": sensor,
                "class_name": cls_name,
                "class_id": CLASS_TO_ID[cls_name]
            })

    if max_videos:
        pairs = pairs[:max_videos]

    return pairs


def split_videos(pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    
    # Stratified split by (sensor, class_name)
    grouped = {}
    for item in pairs:
        key = (item["sensor"], item["class_name"])
        grouped.setdefault(key, []).append(item)

    train_set, val_set, test_set = [], [], []

    for key, items in grouped.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        for item in train_items:
            item["split"] = "train"
        for item in val_items:
            item["split"] = "val"
        for item in test_items:
            item["split"] = "test"

        train_set.extend(train_items)
        val_set.extend(val_items)
        test_set.extend(test_items)

    return pairs


def process_single_video(args_tuple):
    item, output_dir, stride, only_positives, img_size, export_format = args_tuple
    video_path = item["video_path"]
    label_path = item["label_path"]
    video_id = item["video_id"]
    sensor = item["sensor"]
    class_name = item["class_name"]
    class_id = item["class_id"]
    split = item["split"]

    # Load annotations
    try:
        bboxes = load_groundtruth(label_path)
    except Exception as e:
        print(f"Error loading annotations for {label_path}: {e}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= len(bboxes):
            bbox = None
        else:
            bbox = bboxes[frame_idx]

        if frame_idx % stride == 0:
            has_target = (bbox is not None)
            if only_positives and not has_target:
                frame_idx += 1
                continue

            img_filename = f"{video_id}_f{frame_idx:05d}.jpg"
            img_save_dir = os.path.join(output_dir, "images", split)
            lbl_save_dir = os.path.join(output_dir, "labels", split)
            os.makedirs(img_save_dir, exist_ok=True)
            os.makedirs(lbl_save_dir, exist_ok=True)

            img_path = os.path.join(img_save_dir, img_filename)

            h, w = frame.shape[:2]
            if img_size is not None and (h != img_size or w != img_size):
                scale_x = img_size / w
                scale_y = img_size / h
                frame_to_save = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                cur_w, cur_h = img_size, img_size
            else:
                scale_x, scale_y = 1.0, 1.0
                frame_to_save = frame
                cur_w, cur_h = w, h

            cv2.imwrite(img_path, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Process Bounding Box
            yolo_box_str = ""
            abs_box = None
            if has_target:
                bx, by, bw, bh = bbox
                bx = max(0.0, float(bx) * scale_x)
                by = max(0.0, float(by) * scale_y)
                bw = min(float(cur_w - bx), float(bw) * scale_x)
                bh = min(float(cur_h - by), float(bh) * scale_y)

                if bw > 1 and bh > 1:
                    # YOLO normalized format: x_center, y_center, width, height (0-1)
                    x_center = (bx + bw / 2.0) / cur_w
                    y_center = (by + bh / 2.0) / cur_h
                    norm_w = bw / cur_w
                    norm_h = bh / cur_h

                    yolo_box_str = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                    abs_box = (bx, by, bx + bw, by + bh)

            # Write YOLO label file
            txt_filename = f"{video_id}_f{frame_idx:05d}.txt"
            txt_path = os.path.join(lbl_save_dir, txt_filename)
            with open(txt_path, "w") as f:
                f.write(yolo_box_str)

            record = {
                "image_path": img_path,
                "relative_path": os.path.join("images", split, img_filename),
                "label_path": txt_path,
                "split": split,
                "sensor": sensor,
                "video_id": video_id,
                "frame_index": frame_idx,
                "class_name": class_name if has_target else "BACKGROUND",
                "class_id": class_id if has_target else -1,
                "has_target": has_target,
                "image_width": cur_w,
                "image_height": cur_h,
                "bbox_xmin": abs_box[0] if abs_box else None,
                "bbox_ymin": abs_box[1] if abs_box else None,
                "bbox_xmax": abs_box[2] if abs_box else None,
                "bbox_ymax": abs_box[3] if abs_box else None,
                "bbox_width": abs_box[2] - abs_box[0] if abs_box else None,
                "bbox_height": abs_box[3] - abs_box[1] if abs_box else None,
            }
            frame_records.append(record)

        frame_idx += 1

    cap.release()
    return frame_records


def generate_coco_json(records, output_dir, split):
    split_records = [r for r in records if r["split"] == split]
    images = []
    annotations = []
    ann_id = 1

    for img_id, r in enumerate(split_records, start=1):
        images.append({
            "id": img_id,
            "file_name": os.path.basename(r["image_path"]),
            "width": r["image_width"],
            "height": r["image_height"],
            "video_id": r["video_id"],
            "frame_index": r["frame_index"],
            "sensor": r["sensor"]
        })
        if r["has_target"] and r["bbox_xmin"] is not None:
            bx = r["bbox_xmin"]
            by = r["bbox_ymin"]
            bw = r["bbox_width"]
            bh = r["bbox_height"]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": r["class_id"] + 1,  # 1-indexed for COCO
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0
            })
            ann_id += 1

    categories = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(CLASSES)
    ]

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    coco_path = os.path.join(output_dir, f"instances_{split}.json")
    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"Saved COCO annotations: {coco_path} ({len(images)} images, {len(annotations)} annotations)")


def create_yolo_yaml(output_dir):
    abs_out = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""# Drone Detection Dataset Config for Ultralytics YOLO
path: {abs_out}
train: images/train
val: images/val
test: images/test

# Classes (4 total)
names:
  0: AIRPLANE
  1: BIRD
  2: DRONE
  3: HELICOPTER
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Saved YOLO dataset config: {yaml_path}")


def main():
    args = parse_args()
    print("=" * 60)
    print("DRONE DETECTION DATASET PREPARATION")
    print("=" * 60)
    print(f"Data directory:   {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sensor mode:      {args.sensor}")
    print(f"Frame stride:     {args.stride} (1 = all frames, 5 = every 5th frame)")
    print(f"Format:           {args.format}")
    print(f"Workers:          {args.num_workers}")
    print("=" * 60)

    pairs = get_video_label_pairs(args.data_dir, sensor_filter=args.sensor, max_videos=args.max_videos)
    print(f"Found {len(pairs)} matching video & label pairs.")
    if not pairs:
        print("No videos found to process.")
        sys.exit(1)

    pairs = split_videos(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    split_counts = {"train": 0, "val": 0, "test": 0}
    for p in pairs:
        split_counts[p["split"]] += 1
    print(f"Video splits: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")

    # Prepare extraction jobs
    jobs = [
        (item, args.output_dir, args.stride, args.only_positives, args.img_size, args.format)
        for item in pairs
    ]

    all_records = []
    print("\nExtracting frames and generating annotations...")
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(tqdm(executor.map(process_single_video, jobs), total=len(jobs), desc="Videos"))
            for r in results:
                all_records.extend(r)
    else:
        for job in tqdm(jobs, desc="Videos"):
            records = process_single_video(job)
            all_records.extend(records)

    print(f"\nExtracted {len(all_records)} total frames.")

    # Save CSV metadata
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output_dir, "dataset_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset metadata table: {csv_path}")

    # Generate YOLO YAML
    if args.format in ["yolo", "all"]:
        create_yolo_yaml(args.output_dir)

    # Generate COCO JSONs
    if args.format in ["coco", "all"]:
        for split in ["train", "val", "test"]:
            generate_coco_json(all_records, args.output_dir, split)

    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    if not df.empty:
        print("\nSummary by Split:")
        print(df.groupby("split")["has_target"].value_counts().unstack(fill_value=0))
        print("\nSummary by Class:")
        print(df[df["has_target"]]["class_name"].value_counts())
        print("\nSummary by Sensor:")
        print(df["sensor"].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()
