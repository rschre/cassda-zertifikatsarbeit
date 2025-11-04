import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Tuple

import geopandas as gpd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Thread-safe counter for progress tracking on the image tasks
class ProgressCounter:
    def __init__(self, total):
        self.lock = Lock()
        self.completed = 0
        self.total = total
        self.start_time = time.time()

    def increment(self):
        with self.lock:
            self.completed += 1
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = self.total - self.completed
            eta = remaining / rate if rate > 0 else 0
            print(
                f"Progress: {self.completed}/{self.total} ({self.completed / self.total * 100:.1f}%) - "
                f"Rate: {rate:.2f} img/s - ETA: {eta:.0f}s"
            )


def get_crop_bboxes(
    point_gdf: gpd.GeoDataFrame, buffer_distance: int
) -> list[tuple[float, float, float, float]]:
    """Get bounding boxes around points in a GeoDataFrame.

    Args:
        point_gdf (gpd.GeoDataFrame): GeoDataFrame containing Point geometries.
        buffer_distance (int): Buffer distance in meters to create bounding boxes.

    Returns:
        list[tuple[float, float, float, float]]: List of bounding boxes as (minx, miny, maxx, maxy) tuples.
    """

    if point_gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a defined CRS.")
    point_gdf = point_gdf.to_crs(epsg=3035)

    # keep only Point geometries print warning if any geometries were removed
    if len(point_gdf) != len(point_gdf[point_gdf.geometry.type == "Point"]):
        logger.warning(
            "Input GeoDataFrame contains non-Point geometries. These will be ignored."
        )

    point_gdf = point_gdf[point_gdf.geometry.type == "Point"]

    if point_gdf.empty:
        logger.warning("Input GeoDataFrame contains no Point geometries.")
        return []

    buffered = point_gdf.geometry.buffer(buffer_distance, cap_style="square")
    bboxes = list(
        zip(
            buffered.bounds["minx"],
            buffered.bounds["miny"],
            buffered.bounds["maxx"],
            buffered.bounds["maxy"],
        )
    )
    return bboxes


def download_image_task(args):
    """
    Download a single image (wrapper used in download_images_parallel).

    Parameters:
    -----------
    args : tuple
        (idx, geom, wms, layer_name, img_size, img_format, image_output_dir, progress)

    Returns:
    --------
    tuple : (idx, success, error_msg)
    """
    (
        idx,
        geom,
        wms,
        layer_name,
        img_size,
        img_format,
        image_output_dir,
        crs,
        progress,
    ) = args

    try:
        bbox = geom.bounds  # minx, miny, maxx, maxy
        tile_name = f"{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
        img_filename = os.path.join(image_output_dir, f"{tile_name}.jpg")

        if os.path.exists(img_filename):
            if progress:
                progress.increment()
            return (idx, True, None, img_filename)

        # Download image
        img = wms.getmap(
            layers=[layer_name],
            srs=crs,
            bbox=bbox,
            size=img_size,
            format=img_format,
        )

        # Save to file
        with open(img_filename, "wb") as f:
            f.write(img.read())

        # Update progress
        if progress:
            progress.increment()

        return (idx, True, None, img_filename)

    except Exception as e:
        if progress:
            progress.increment()
        return (idx, False, str(e), None)


def download_images_parallel(
    gdf,
    wms,
    layer_name,
    img_size,
    image_output_dir,
    img_format="image/jpeg",
    crs="EPSG:3035",
    max_workers=20,
    limit=None,
):
    """
    Download images in parallel using multiple threads.

    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing geometries
    wms : WebMapService
        WMS service instance
    layer_name : str
        Name of the WMS layer
    img_size : tuple
        (width, height) in pixels
    img_format : str
        Image format (e.g., 'image/jpeg')
    image_output_dir : str
        Directory to save images
    crs : str
        Coordinate reference system
    max_workers : int
        Number of parallel threads (default: 20)
    limit : int
        Maximum number of images to download (None for all)

    Returns:
    --------
    dict : Summary with 'success', 'failed', and 'errors' lists
    """
    # Prepare geometries to process
    geometries = gdf.geometry if limit is None else gdf.geometry.head(limit)
    total = len(geometries)

    print(
        f"Starting parallel download of {total} images using {max_workers} threads..."
    )

    # Progress counter
    progress = ProgressCounter(total)

    # Prepare arguments for each task
    tasks = [
        (
            idx,
            geom,
            wms,
            layer_name,
            img_size,
            img_format,
            image_output_dir,
            crs,
            progress,
        )
        for idx, geom in enumerate(geometries)
    ]

    # Execute downloads in parallel
    results = {"success": [], "failed": [], "errors": []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(download_image_task, task) for task in tasks]

        # Collect results as they complete
        for future in as_completed(futures):
            idx, success, error, filename = future.result()

            if success:
                results["success"].append((idx, filename))
            else:
                results["failed"].append(idx)
                results["errors"].append((idx, error))

    # Print summary
    print(f"\n{'=' * 60}")
    print("Download Complete!")
    print(
        f"Total: {total} | Success: {len(results['success'])} | Failed: {len(results['failed'])}"
    )

    if results["errors"]:
        print("\nErrors:")
        for idx, error in results["errors"][:10]:  # Show first 10 errors
            print(f"  Image {idx}: {error}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")

    return results


def get_image_size_px(
    target_resolution_m: int,
    bbox: Tuple[float, float, float, float],
    m_per_deg: float = 111320,
) -> Tuple[int, int]:
    """
    Calculate the pixel resolution (in meters per pixel) for a given bounding box and image size.

    Args:
        target_resolution_m (int): Desired target resolution in meters (e.g. 10 for 10m/pixel).
        bbox (Tuple[float, float, float, float]): Bounding box in the format (min_lon, min_lat, max_lon, max_lat) and WGS84.

    Returns:
        float: Pixel resolution in meters per pixel.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    center_lat = (min_lat + max_lat) / 2
    center_lat_rad = math.radians(center_lat)

    lon_correction_factor = math.cos(center_lat_rad)
    lat_span_degrees = max_lat - min_lat
    lon_span_degrees = max_lon - min_lon

    lat_span_meters = lat_span_degrees * m_per_deg
    lon_span_meters = lon_span_degrees * m_per_deg * lon_correction_factor

    required_width_pixels = int(lon_span_meters / target_resolution_m)
    required_height_pixels = int(lat_span_meters / target_resolution_m)

    # Return as (width, height)
    return (required_width_pixels, required_height_pixels)
