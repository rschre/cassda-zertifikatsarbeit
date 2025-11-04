import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image


def get_idx(plot_id, num_images, imgs_per_plot):
    start_idx = plot_id * imgs_per_plot
    end_idx = min(start_idx + imgs_per_plot, num_images)
    return start_idx, end_idx


def plot_detection_subset(detections_subset, nrows, ncols, image_dir):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 30))
    axs = axs.flatten()

    for r, ax in zip(detections_subset, axs):
        img = cv2.imread(os.path.join(image_dir, r.get("image_path")))
        ax.imshow(img)
        ax.set_title(os.path.basename(r.get("image_path")), fontsize=10)

        polygon = json.loads(r.get("obb_xyxyxyxy"))

        cx, cy, w, h = r.get("obb_x"), r.get("obb_y"), r.get("obb_w"), r.get("obb_h")
        poly = plt.Polygon(
            np.array(polygon).reshape(-1, 2),
            fill=False,
            edgecolor="red",
            linewidth=0.5,
        )
        ax.add_patch(poly)

        # add confidence to the polygon
        conf = r.get("confidence")
        ax.text(
            cx + w / 2,
            cy - h / 2,
            f"{conf:.2f}",
            fontsize=12,
            color="red",
            ha="center",
            va="bottom",
        )
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_yolo_label(image_path, label_path, num_examples=3):
    """
    Visualize YOLO segmentation labels overlaid on images.

    Parameters:
    -----------
    image_path : str
        Path to the directory containing images
    label_path : str
        Path to the directory containing YOLO label files
    num_examples : int
        Number of examples to visualize
    """
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith(".png")])[
        :num_examples
    ]

    fig, axes = plt.subplots(1, num_examples, figsize=(5 * num_examples, 5))
    if num_examples == 1:
        axes = [axes]

    for idx, img_file in enumerate(image_files):
        # Load image
        img = Image.open(os.path.join(image_path, img_file))
        img_array = np.array(img)

        # Load corresponding label
        label_file = img_file.replace(".png", ".txt")
        label_file_path = os.path.join(label_path, label_file)

        ax = axes[idx]
        ax.imshow(img_array)

        # Read and parse YOLO label
        if os.path.exists(label_file_path):
            with open(label_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    coords = [float(x) for x in parts[1:]]

                    # Convert normalized coordinates to pixel coordinates
                    img_height, img_width = img_array.shape[:2]
                    pixel_coords = []
                    for i in range(0, len(coords), 2):
                        x_norm, y_norm = coords[i], coords[i + 1]
                        x_pixel = x_norm * img_width
                        y_pixel = y_norm * img_height
                        pixel_coords.append([x_pixel, y_pixel])

                    # Draw polygon
                    polygon = patches.Polygon(
                        pixel_coords,
                        closed=True,
                        edgecolor="red",
                        facecolor="red",
                        alpha=0.3,
                        linewidth=2,
                    )
                    ax.add_patch(polygon)

        ax.set_title(f"{img_file}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
