import os

import geopandas as gpd


def get_detection_center_coords(detection):
    xmin, ymin, _xmax, _ymax = (
        os.path.basename(detection["image_path"]).replace(".jpg", "").split("_")
    )
    xc = detection["obb_x"]
    yc = detection["obb_y"]

    xc_epsg3035 = int(xmin) - (int(xc) * 2)
    yc_epsg3035 = int(ymin) + (int(yc) * 2)

    return xc_epsg3035, yc_epsg3035


def filter_duplicate_detections(
    results_df: gpd.GeoDataFrame, min_distance: int = 15
) -> gpd.GeoDataFrame:
    distance_matrix = results_df.geometry.apply(
        lambda geom: results_df.geometry.distance(geom)
    )
    to_drop = set()
    for i in range(len(distance_matrix)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(distance_matrix)):
            if distance_matrix.iat[i, j] < min_distance:
                to_drop.add(j)
    return results_df.drop(index=to_drop)
