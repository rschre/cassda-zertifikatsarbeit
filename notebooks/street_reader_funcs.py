import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString


def tiled_reader(path, tile_size=1024, overlap=1):
    """
    Generator yielding (window, skeleton_tile, dist_tile, row_offset, col_offset)
    """
    with rasterio.open(path) as src:
        height, width = src.height, src.width

        # Read the distance transform once (you can also stream it similarly)
        # If you already have a separate distance raster, open it the same way.
        dist = src.read(1)  # whole distance band – optional
        # If dist is too big, you can also read it tile‑wise in the loop below.

        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                # Compute window extents with overlap
                win = Window(
                    col_off=max(col - overlap, 0),  # type: ignore
                    row_off=max(row - overlap, 0),  # type: ignore
                    width=min(tile_size + 2 * overlap, width - col + overlap),  # type: ignore
                    height=min(tile_size + 2 * overlap, height - row + overlap),  # type: ignore
                )
                # Clip to raster bounds
                win = win.intersection(Window(0, 0, width, height))  # type: ignore

                # Load the binary mask (skeleton) for this window
                # Assuming the skeleton is stored in a separate band/file.
                # Replace `'skeleton.tif'` with your actual skeleton raster.
                with rasterio.open("skeleton.tif") as sk_src:
                    sk_tile = sk_src.read(1, window=win)

                # Slice the distance array to the same window
                dist_tile = dist[
                    int(win.row_off) : int(win.row_off + win.height),
                    int(win.col_off) : int(win.col_off + win.width),
                ]

                yield (
                    win,
                    sk_tile.astype(bool),  # ensure boolean mask
                    dist_tile,
                    int(win.row_off),  # global row offset of the tile's origin
                    int(win.col_off),  # global col offset of the tile's origin
                )


def skeleton_to_weighted_graph(skel, dist):
    G = nx.Graph()
    rows, cols = np.where(skel)
    for r, c in zip(rows, cols):
        G.add_node((r, c), weight=dist[r, c])
        # 8‑neighbour offsets
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] and skel[nr, nc]:
                    # Edge weight = negative average distance (so Dijkstra prefers high distance)
                    w = -(dist[r, c] + dist[nr, nc]) / 2.0
                    G.add_edge((r, c), (nr, nc), weight=w)
    return G


def skeleton_to_positive_graph(skel, dist):
    """
    Returns a NetworkX Graph where each edge weight is:
        cost = max_dist - average_distance_of_the_two_pixels
    This way, paths that go through more central pixels (higher distance to edge)
    have lower cost and are preferred by shortest-path algorithms.
    8-connectivity is used (including diagonals).
    """
    G = nx.Graph()
    rows, cols = np.where(skel)
    max_dist = float(dist.max())  # biggest distance in the whole mask

    for r, c in zip(rows, cols):
        G.add_node((r, c), dist=dist[r, c])

        # 8‑connectivity (including diagonals)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] and skel[nr, nc]:
                    # Average distance of the two neighbouring pixels
                    avg = (dist[r, c] + dist[nr, nc]) / 2.0
                    # Larger avg → cheaper (more central)
                    cost = max_dist - avg
                    G.add_edge((r, c), (nr, nc), weight=cost)
    return G


def skeleton_to_positive_graph_tile(skel, dist, row_off=0, col_off=0):
    """
    Build a graph for a single tile.
    `row_off` / `col_off` are the global offsets of the tile's top‑left corner.
    """
    G = nx.Graph()
    rows, cols = np.where(skel)
    max_dist = float(dist.max())

    for r, c in zip(rows, cols):
        # Translate tile‑local (r,c) → global (R,C)
        R, C = r + row_off, c + col_off
        G.add_node((R, C), dist=dist[r, c])

        # 8‑connectivity (including diagonals)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                # Stay inside the tile (no need to check global bounds here)
                if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] and skel[nr, nc]:
                    avg = (dist[r, c] + dist[nr, nc]) / 2.0
                    cost = max_dist - avg
                    # Global neighbour coordinates:
                    N_R, N_C = nr + row_off, nc + col_off
                    G.add_edge((R, C), (N_R, N_C), weight=cost)
    return G


def graph_to_lines(g, pos):
    """Return a MultiLineString representing all edges."""
    lines = []
    for u, v in g.edges():
        # Grab the coordinates of the two endpoints
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        lines.append(LineString([(x1, y1), (x2, y2)]))
    return MultiLineString(lines)


def pixel_to_world(row, col, transform):
    """Apply the raster affine transform."""
    x, y = transform * (col, row)
    return x, y


def graph_to_geometries(G: nx.Graph, transform):
    """
    Returns a GeoDataFrame with a single MultiLineString geometry
    that represents every edge of the graph, correctly georeferenced.
    """
    lines = []
    for u, v, _ in G.edges(data=True):
        # u and v are (row, col) tuples
        x1, y1 = pixel_to_world(u[0], u[1], transform)
        x2, y2 = pixel_to_world(v[0], v[1], transform)
        lines.append(LineString([(x1, y1), (x2, y2)]))

    mls = MultiLineString(lines)

    # If you know the CRS (e.g., EPSG:32633), set it here.
    # Replace with the actual CRS of your raster.
    crs = "EPSG:3035"  # <-- adjust as needed
    gdf = gpd.GeoDataFrame({"geometry": [mls]}, crs=crs)
    return gdf
