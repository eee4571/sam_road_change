"""Continuous raw-width road surfaces; fine observations remain untouched."""
import geopandas as gpd
import numpy as np
import time
from shapely import make_valid, union_all
from shapely.geometry import Polygon


def _polygons(geometry):
    if geometry.geom_type == 'Polygon':
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from _polygons(part)


def _chain_polygons(geometries):
    """Join shared left/right sections, retaining exceptional valid cell pieces.

    Raw backend quads are ordered left-start, left-end, right-end, right-start.
    Discontinuities (including unresolved widths) terminate a boundary run.
    Already repaired/self-crossing cells are retained for the regional union.
    """
    left, right = [], []
    for geometry in geometries:
        coords = list(geometry.exterior.coords) if geometry.geom_type == 'Polygon' and not geometry.interiors else []
        quad = len(coords) == 5
        connected = quad and left and left[-1] == coords[0] and right[-1] == coords[3]
        if left and not connected:
            yield make_valid(Polygon(left + right[::-1]))
            left, right = [], []
        if quad:
            if not left:
                left, right = [coords[0]], [coords[3]]
            left.append(coords[1])
            right.append(coords[2])
        else:
            yield make_valid(geometry)
    if left:
        yield make_valid(Polygon(left + right[::-1]))


def image_pixel_size(image_path, metric_crs):
    """Physical pixel size without reading or resampling image pixels."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(image_path) as ds:
        xy = np.array([ds.transform * p for p in [(ds.width/2, ds.height/2),
                       (ds.width/2+1, ds.height/2), (ds.width/2, ds.height/2+1)]])
        x, y = Transformer.from_crs(ds.crs, metric_crs, always_xy=True).transform(xy[:, 0], xy[:, 1])
    points = np.column_stack([x, y])
    return float(min(np.linalg.norm(points[1:] - points[0], axis=1)))


def build_road_surfaces(corridors, *, pixel_size=None, image_path=None):
    """Build whole-chain outlines then dissolve to connected regional polygons.

    No width averaging, buffering, global closing or snapping: disconnected
    roads remain disconnected. Only tiny enclosed holes and boundary detail
    below 1/50 pixel are cleaned, independently inside each component.
    """
    if corridors.empty:
        return gpd.GeoDataFrame({'surface_id': [], 'area_m2': []}, geometry=[], crs=corridors.crs)
    started = time.perf_counter()
    metric_crs = (corridors.estimate_utm_crs() if corridors.crs.is_geographic or
                  corridors.crs.axis_info[0].unit_conversion_factor != 1 else corridors.crs)
    frame = corridors.to_crs(metric_crs)
    if pixel_size is None:
        if image_path is None:
            raise ValueError('Raw road surface cleanup requires an image pixel size')
        pixel_size = image_pixel_size(image_path, metric_crs)
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError('Invalid road surface pixel size')
    chains = []
    for _, group in frame.groupby('parent_id', sort=False):
        chains.extend(_chain_polygons(group.geometry))
    merged = make_valid(union_all(chains))
    rows = []
    for polygon in _polygons(merged):
        # Keep separate components, even small ones: they can be real roads.
        holes = [ring for ring in polygon.interiors if Polygon(ring).area > (pixel_size / 4)**2]
        cleaned = Polygon(polygon.exterior, holes).simplify(pixel_size / 50, preserve_topology=True)
        # Clipping removes any outward simplification drift: no closing of
        # medians or neighbouring components, and flat road ends stay flat.
        cleaned = make_valid(cleaned.intersection(Polygon(polygon.exterior, holes)))
        for part in _polygons(cleaned):
            rows.append({'surface_id': f'RRS{len(rows):07d}', 'area_m2': part.area, 'geometry': part})
    result = gpd.GeoDataFrame(rows, geometry='geometry', crs=metric_crs).to_crs(corridors.crs)
    print(f'[Raw width] surface dissolve: {len(corridors)} cells -> {len(result)} connected polygons; '
          f'raw_surface_dissolve={time.perf_counter()-started:.6f}s', flush=True)
    return result
