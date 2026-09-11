"""
Visualize Dataset Annotations
Plots sample frames from the extracted dataset or directly from videos with bounding box overlays.
"""

import os
import glob
import random
import cv2
import matplotlib.pyplot as plt

CLASSES = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
COLORS = {
    'AIRPLANE': (255, 128, 0),     # Orange
    'BIRD': (0, 200, 255),         # Yellow
    'DRONE': (0, 255, 0),          # Green
    'HELICOPTER': (255, 0, 128)    # Magenta
}


def visualize_yolo_samples(dataset_dir="dataset_yolo", num_samples=6, output_image="samples_visualization.jpg"):
    image_paths = glob.glob(os.path.join(dataset_dir, "images", "*", "*.jpg"))
    if not image_paths:
        print(f"No images found in {dataset_dir}")
        return

    random.seed(42)
    selected = random.sample(image_paths, min(num_samples, len(image_paths)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, img_path in enumerate(selected):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Find matching label file
        rel_parts = img_path.split(os.sep)
        split = rel_parts[-2]
        base_txt = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        lbl_path = os.path.join(dataset_dir, "labels", split, base_txt)

        title = os.path.basename(img_path)
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc = float(parts[1]) * w
                        yc = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h

                        xmin = int(xc - bw / 2.0)
                        ymin = int(yc - bh / 2.0)
                        xmax = int(xc + bw / 2.0)
                        ymax = int(yc + bh / 2.0)

                        cls_name = CLASSES[cls_id]
                        color = (0, 255, 0)
                        cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img_rgb, f"{cls_name}", (xmin, max(15, ymin - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        title = f"{cls_name} ({split})"

        axes[idx].imshow(img_rgb)
        axes[idx].set_title(title, fontsize=11)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"Visualization saved to {output_image}")


if __name__ == "__main__":
    visualize_yolo_samples()
