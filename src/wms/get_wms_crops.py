import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    # Example usage
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=gpd.points_from_xy([10.0, 20.0], [50.0, 60.0]),
        crs="EPSG:4326",
    )
    print(get_crop_bboxes(gdf, 1000)[0])
