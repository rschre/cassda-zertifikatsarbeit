import math
from typing import Tuple


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
