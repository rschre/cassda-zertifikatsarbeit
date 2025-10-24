import glob
import logging
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_dataset(
    input_dir, output_dir, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=42
):
    """
    Splits data exported from generic_data_download.ipynb into train, validation, and test sets.

    Parameters:
    -----------
    input_dir : str
        Path to the input directory containing data
    output_dir : str
        Path to the output directory to save splits
    train_size : float
        Proportion of data to use for training set
    val_size : float
        Proportion of data to use for validation set
    test_size : float
        Proportion of data to use for test set
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    tuple
        Three lists: (train_data, val_data, test_data)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1."

    images_folder = os.path.join(input_dir, "images")
    annotations_folder = os.path.join(input_dir, "labels")

    img_ids = glob.glob(os.path.join(images_folder, "*.png"))

    train_split = int(len(img_ids) * train_size)
    val_split = int(len(img_ids) * val_size)
    test_split = int(
        len(img_ids) - len(img_ids) * (train_size + val_size)
    )  # Remaining images

    logger.info(f"Total images: {len(img_ids)}")
    logger.info(f"Train: {train_split}, Val: {val_split}, Test: {test_split}")

    np.random.seed(random_seed)
    np.random.shuffle(img_ids)
    train_ids = img_ids[:train_split]
    val_ids = img_ids[train_split : train_split + val_split]
    test_ids = img_ids[train_split + val_split :]

    for split_name, split_ids in zip(
        ["train", "val", "test"], [train_ids, val_ids, test_ids]
    ):
        split_image_dir = os.path.join(output_dir, "images", split_name)
        split_label_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(split_image_dir, exist_ok=True)
        os.makedirs(split_label_dir, exist_ok=True)

        for img_path in split_ids:
            img_filename = os.path.basename(img_path)
            label_filename = img_filename.replace(".png", ".txt").replace(
                "images", "labels"
            )

            # Copy image
            os.rename(img_path, os.path.join(split_image_dir, img_filename))
            # Copy label
            os.rename(
                os.path.join(annotations_folder, label_filename),
                os.path.join(split_label_dir, label_filename),
            )

    logger.info(f"Data split completed. Wrote data to output directory: {output_dir}")

    return train_ids, val_ids, test_ids


def polygon_to_yolo_segmentation(polygon: Polygon, bbox: tuple, class_id=0):
    """
    Convert a geo polygon to YOLO segmentation format.

    YOLO segmentation format:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

    All coordinates are normalized (0-1 range) relative to the image dimensions.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        The polygon to convert
    bbox : tuple
        Bounding box (minx, miny, maxx, maxy) in the same CRS as polygon
    class_id : int
        Class ID for YOLO (default: 0)

    Returns:
    --------
    str
        YOLO format string for the polygon
    """
    minx, miny, maxx, maxy = bbox
    bbox_width = maxx - minx
    bbox_height = maxy - miny

    # Get exterior coordinates of the polygon
    if hasattr(polygon, "exterior"):
        coords = list(polygon.exterior.coords)
    else:
        # Handle Point geometries by creating a small square around them
        if polygon.geom_type == "Point":
            x, y = polygon.x, polygon.y
            # Create a small square (1% of bbox size)
            offset = min(bbox_width, bbox_height) * 0.01
            coords = [
                (x - offset, y - offset),
                (x + offset, y - offset),
                (x + offset, y + offset),
                (x - offset, y + offset),
                (x - offset, y - offset),
            ]
        else:
            return None

    # Convert coordinates to normalized YOLO format
    yolo_coords = []
    for x, y in coords[:-1]:  # Skip last point (same as first in closed polygon)
        # Normalize x coordinate (0 = left, 1 = right)
        norm_x = (x - minx) / bbox_width

        # Normalize y coordinate and FLIP for image coordinates
        # Geographic: y increases upward (miny at bottom, maxy at top)
        # Image: y increases downward (0 at top, 1 at bottom)
        # So we need: 1 - normalized_y
        norm_y = 1.0 - ((y - miny) / bbox_height)

        # Clamp to [0, 1] range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        yolo_coords.extend([norm_x, norm_y])

    # Format: class_id x1 y1 x2 y2 ... xn yn
    yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in yolo_coords])

    return yolo_line


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

                    class_id = int(parts[0])
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
