from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import transform as transform_coordinates
from shapely.geometry import LineString, Point, box


def _line_parts(geometry) -> list:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def _graph_from_world_lines(
    frame: gpd.GeoDataFrame, transform: rasterio.Affine,
    clip_geometry=None,
) -> tuple[np.ndarray, np.ndarray]:
    inverse = ~transform
    nodes: list[tuple[float, float]] = []
    edges: list[tuple[int, int]] = []
    tolerance = 0.05
    node_bins: dict[tuple[int, int], list[int]] = {}

    def node_id(point: tuple[float, float]) -> int:
        key = (int(math.floor(point[0] / tolerance)), int(math.floor(point[1] / tolerance)))
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                for match in node_bins.get((key[0] + row_offset, key[1] + col_offset), []):
                    existing = nodes[match]
                    if math.hypot(existing[0] - point[0], existing[1] - point[1]) <= tolerance:
                        return match
        nodes.append(point)
        node_bins.setdefault(key, []).append(len(nodes) - 1)
        return len(nodes) - 1

    for geometry in frame.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if clip_geometry is not None:
            if not geometry.intersects(clip_geometry):
                continue
            geometry = geometry.intersection(clip_geometry)
        for part in _line_parts(geometry):
            previous = None
            for x, y in part.coords:
                col, row = inverse * (x, y)
                current = node_id((float(row), float(col)))
                if previous is not None and previous != current:
                    edges.append((previous, current))
                previous = current
    unique_edges = sorted({tuple(sorted(edge)) for edge in edges})
    return (
        np.asarray(nodes, dtype=np.float32).reshape(-1, 2),
        np.asarray(unique_edges, dtype=np.int32).reshape(-1, 2),
    )


def _project_manual_widths(
    measurements: list[dict], source_transform: rasterio.Affine, source_crs,
    destination_transform: rasterio.Affine, destination_crs, destination_bounds,
) -> list[dict]:
    projected = []
    inverse = ~destination_transform
    destination_box = box(
        destination_bounds.left, destination_bounds.bottom,
        destination_bounds.right, destination_bounds.top,
    )
    for measurement in measurements:
        try:
            pixels = [
                (float(measurement[f"{name}_col"]), float(measurement[f"{name}_row"]))
                for name in ("start", "end", "target")
            ]
        except (KeyError, TypeError, ValueError):
            continue
        world = [source_transform * point for point in pixels]
        xs, ys = transform_coordinates(
            source_crs, destination_crs,
            [point[0] for point in world], [point[1] for point in world],
        )
        target = Point(xs[2], ys[2])
        interval_world = None
        if str(measurement.get("source", "")) == "manual_interval_width":
            try:
                interval_pixels = [
                    (float(measurement[f"range_{name}_col"]), float(measurement[f"range_{name}_row"]))
                    for name in ("start", "end")
                ]
                interval_source_world = [source_transform * point for point in interval_pixels]
                interval_xs, interval_ys = transform_coordinates(
                    source_crs, destination_crs,
                    [point[0] for point in interval_source_world],
                    [point[1] for point in interval_source_world],
                )
                interval_world = LineString(zip(interval_xs, interval_ys))
            except (KeyError, TypeError, ValueError):
                interval_world = None
        if not destination_box.covers(target) and (
            interval_world is None or not destination_box.intersects(interval_world)
        ):
            continue
        if interval_world is not None and not destination_box.covers(target):
            local_interval = interval_world.intersection(destination_box)
            representative = (
                local_interval.interpolate(0.5, normalized=True)
                if hasattr(local_interval, "interpolate") and not local_interval.is_empty
                else local_interval.centroid
            )
            xs[2], ys[2] = float(representative.x), float(representative.y)
        tile_pixels = [inverse * (x, y) for x, y in zip(xs, ys)]
        row_col = [(float(row), float(col)) for col, row in tile_pixels]
        start, end, target_pixel = row_col
        row = dict(measurement)
        row.update({
            "start_row": start[0], "start_col": start[1],
            "end_row": end[0], "end_col": end[1],
            "target_row": target_pixel[0], "target_col": target_pixel[1],
            "width_px": float(math.hypot(end[0] - start[0], end[1] - start[1])),
            "source": "global_manual_boundary_measurement",
        })
        if interval_world is not None:
            interval_pixels = [inverse * coordinate for coordinate in interval_world.coords]
            row.update({
                "range_start_row": float(interval_pixels[0][1]),
                "range_start_col": float(interval_pixels[0][0]),
                "range_end_row": float(interval_pixels[-1][1]),
                "range_end_col": float(interval_pixels[-1][0]),
                "source": "manual_interval_width",
            })
        projected.append(row)
    return projected
