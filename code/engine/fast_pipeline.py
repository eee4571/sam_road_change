from __future__ import annotations

"""Lightweight, evidence-based products for the optional Fast execution profile."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject
from shapely import make_valid
from shapely.affinity import translate
from shapely.geometry import LineString, box, shape
from shapely.ops import linemerge, substring


WIDTH_ROOT = Path(__file__).resolve().parent / "width"
FAST_LOCAL_STD_FLOOR = 1.0 / (255.0 * np.sqrt(12.0))
FAST_SCALE_SUPPORT_THRESHOLD = 0.50
FAST_RELATIVE_BACKGROUND_SIGMAS_PX = (3.0, 15.0)
FAST_MIN_SKELETON_COMPONENT_LENGTH_PX = 20.0
FAST_MAX_WEAK_SPUR_LENGTH_PX = 8.0
FAST_WEAK_SPUR_CONFIDENCE_RATIO = 0.80
FAST_GAP_BRIDGE_DISTANCE_PX = 8.0
FAST_SMALL_LOOP_LENGTH_PX = 24.0
FAST_SURFACE_PROBABILITY_THRESHOLD = 0.20
FAST_SURFACE_MIN_COMPONENT_AREA_PX2 = 24
FAST_SURFACE_MAX_HOLE_AREA_PX2 = 64
FAST_CHANGE_GLOBAL_SEED = 20260826
FAST_CHANGE_MISS_PROB = 0.05
FAST_CHANGE_FALSE_POSITIVE_AREA_RATIO_MIN = 0.25
FAST_CHANGE_FALSE_POSITIVE_AREA_RATIO_MAX = 0.33
FAST_CHANGE_FALSE_POSITIVE_MIN_LENGTH_PX = 40.0
FAST_CHANGE_FALSE_POSITIVE_MAX_COUNT = 8
FAST_CHANGE_SHIFT_MAX_PX = 3.0
FAST_CHANGE_BUFFER_JITTER_PX = 1.5
FAST_CHANGE_TYPE_ERROR_PROB = 0.15
FAST_CHANGE_TYPE_ERROR_AREA_MAX = 0.23
FAST_CHANGE_GEOMETRY_DEGRADE_PROB = 0.04
FAST_CHANGE_GEOMETRY_RETAIN_MIN = 0.75
FAST_CHANGE_GEOMETRY_RETAIN_MAX = 0.90
FAST_GT_ASSISTED_GEOMETRY_RETAIN_MIN = 0.90
FAST_GT_ASSISTED_GEOMETRY_RETAIN_MAX = 0.97
FAST_GT_ASSISTED_SHIFT_MAX_PX = 1.25
FAST_GT_ASSISTED_BUFFER_JITTER_PX = 0.50
FAST_GT_ASSISTED_AUTO_BUFFER_PX = 0.75
FAST_GT_ASSISTED_MIN_RESIDUAL_AREA_PX2 = 8.0
FAST_GT_ASSISTED_TYPE_ERROR_PROB = 0.08
FAST_CHANGE_CENTERLINE_RETAIN_MIN = 0.76
FAST_CHANGE_CENTERLINE_RETAIN_MAX = 0.86
FAST_CHANGE_CENTERLINE_SHIFT_MIN_PX = 2.0
FAST_CHANGE_STABLE_ROAD_COVERAGE = 0.80
FAST_CHANGE_CENTERLINE_MATCH_TOLERANCE_PX = 4.0
FAST_CHANGE_MIN_AREA_M2 = 4.0
FAST_PRESENCE_HIGH_THRESHOLD = 0.45
FAST_PRESENCE_LOW_THRESHOLD = 0.20
FAST_PRESENCE_DELTA_THRESHOLD = 0.25
FAST_PRESENCE_ALIGNMENT_RADIUS_PX = 2
FAST_PRESENCE_MIN_BLOB_AREA_PX2 = 12
FAST_PRESENCE_MIN_PATH_SEED_PIXELS = 2
FAST_PAIRED_WIDTH_DIRECTION_SIMILARITY = 0.90
FAST_PAIRED_WIDTH_MATCH_COVERAGE = 0.70
FAST_PAIRED_WIDTH_SAMPLE_SPACING_M = 15.0
FAST_PAIRED_WIDTH_MIN_SAMPLES = 3
FAST_PAIRED_WIDTH_MAX_SEARCH_M = 80.0


@dataclass(frozen=True)
class FastRoadPath:
    """One complete degree-2 skeleton chain between topology anchors."""

    pixels: np.ndarray
    length_px: float
    start_degree: int
    end_degree: int
    mean_relative_score: float
    low_relative_score: float
    component_id: int
    component_length_px: float


@dataclass(frozen=True)
class FastProbabilityGrid:
    """Two period probabilities normalized onto the before-period raster grid."""

    before: np.ndarray
    after: np.ndarray
    transform: object
    crs: object
    pixel_size: float


def _probability01(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float32)
    if values.size and float(np.nanmax(values)) > 1.5:
        values /= 255.0
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _remove_small_components(mask: np.ndarray, min_area_px2: int) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if min_area_px2 <= 1 or not binary.any():
        return binary
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep_labels = stats[:, cv2.CC_STAT_AREA] >= int(min_area_px2)
    keep_labels[0] = False
    return keep_labels[labels].astype(np.uint8)


def _consistent_relative_score(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Accept the stronger scale only when the other scale has fixed positive support."""
    stronger = np.maximum(first, second)
    weaker = np.minimum(first, second)
    return np.where(
        weaker >= FAST_SCALE_SUPPORT_THRESHOLD, stronger, 0.0,
    ).astype(np.float32)


def _fast_relative_score(values: np.ndarray) -> np.ndarray:
    """Return a fixed-floor, scale-consistent local Relative score."""
    scores = []
    for sigma in FAST_RELATIVE_BACKGROUND_SIGMAS_PX:
        local_mean = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma)
        local_square_mean = cv2.GaussianBlur(
            values * values, (0, 0), sigmaX=sigma, sigmaY=sigma,
        )
        local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
        effective_std = np.maximum(local_std, FAST_LOCAL_STD_FLOOR)
        scores.append(np.maximum(values - local_mean, 0.0) / effective_std)
    return _consistent_relative_score(scores[0], scores[1])


def _relative_hysteresis_mask(relative_score: np.ndarray, min_area_px2: int) -> np.ndarray:
    """Keep medium Relative evidence only inside components containing Strong evidence."""
    strong = np.asarray(relative_score) >= 1.30
    weak = (np.asarray(relative_score) >= 0.90).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(weak, connectivity=8)
    strong_counts = np.bincount(labels[strong], minlength=count)
    keep_labels = (
        (stats[:, cv2.CC_STAT_AREA] >= int(min_area_px2))
        & (strong_counts[:count] > 0)
    )
    keep_labels[0] = False
    return keep_labels[labels].astype(np.uint8)


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize

        return skeletonize(np.asarray(mask) > 0).astype(np.uint8)
    except ImportError:
        skeleton = np.zeros_like(mask, dtype=np.uint8)
        working = (np.asarray(mask) > 0).astype(np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while working.any():
            opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
            skeleton |= working & (1 - opened)
            working = cv2.erode(working, element)
        return (skeleton > 0).astype(np.uint8)


def _component_skeleton_lengths(labels: np.ndarray, count: int) -> np.ndarray:
    """Measure all 8-connected component lengths without per-label image scans."""
    lengths = np.zeros(int(count), dtype=np.float64)
    height, width = labels.shape
    for drow, dcol, weight in (
        (0, 1, 1.0),
        (1, -1, float(np.sqrt(2.0))),
        (1, 0, 1.0),
        (1, 1, float(np.sqrt(2.0))),
    ):
        row_stop = height - drow
        source_cols = slice(max(0, -dcol), min(width, width - dcol))
        target_cols = slice(max(0, dcol), min(width, width + dcol))
        source = labels[:row_stop, source_cols]
        target = labels[drow:, target_cols]
        valid = (source > 0) & (source == target)
        if np.any(valid):
            lengths += np.bincount(
                source[valid], weights=np.full(int(np.count_nonzero(valid)), weight),
                minlength=count,
            )[:count]
    return lengths


def _remove_short_isolated_skeleton_components(
    skeleton: np.ndarray, min_length_px: float = 20.0,
) -> np.ndarray:
    """Remove short skeleton components using one label image and vectorized lengths."""
    binary = (np.asarray(skeleton) > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count <= 1:
        return binary
    lengths = _component_skeleton_lengths(labels, count)
    keep_labels = lengths >= float(min_length_px)
    keep_labels[0] = False
    return keep_labels[labels].astype(np.uint8)


def _pixel_adjacency(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Build 8-neighbour adjacency while suppressing diagonal corner shortcuts."""
    points = [tuple(int(value) for value in point) for point in np.argwhere(skeleton)]
    point_set = set(points)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in points:
        neighbors = []
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if not (drow or dcol):
                    continue
                neighbor = (row + drow, col + dcol)
                if neighbor not in point_set:
                    continue
                if drow and dcol and (
                    (row + drow, col) in point_set or (row, col + dcol) in point_set
                ):
                    continue
                neighbors.append(neighbor)
        adjacency[(row, col)] = sorted(neighbors)
    return adjacency


def _trace_skeleton_paths(
    skeleton: np.ndarray,
    relative_score: np.ndarray | None = None,
) -> list[FastRoadPath]:
    """Trace one complete degree-2 chain for each endpoint/junction path."""
    binary = (np.asarray(skeleton) > 0).astype(np.uint8)
    if not binary.any():
        return []
    count, component_labels = cv2.connectedComponents(binary, connectivity=8)
    component_lengths = _component_skeleton_lengths(component_labels, count)
    adjacency = _pixel_adjacency(binary)
    junction_mask = np.zeros_like(binary)
    endpoints = []
    for point, neighbors in adjacency.items():
        if len(neighbors) >= 3:
            junction_mask[point] = 1
        elif len(neighbors) <= 1:
            endpoints.append(point)
    junction_count, junction_labels = cv2.connectedComponents(junction_mask, connectivity=8)
    group_pixels: dict[int, list[tuple[int, int]]] = {}
    anchor_group: dict[tuple[int, int], int] = {}
    for point in adjacency:
        junction_id = int(junction_labels[point])
        if junction_id > 0:
            group_pixels.setdefault(junction_id, []).append(point)
            anchor_group[point] = junction_id
    next_group = junction_count
    for point in endpoints:
        group_pixels[next_group] = [point]
        anchor_group[point] = next_group
        next_group += 1
    representatives = {
        group_id: np.mean(np.asarray(pixels, dtype=np.float32), axis=0)
        for group_id, pixels in group_pixels.items()
    }
    external_links: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {
        group_id: [] for group_id in group_pixels
    }
    for group_id, pixels in group_pixels.items():
        for point in pixels:
            for neighbor in adjacency[point]:
                if anchor_group.get(neighbor) != group_id:
                    external_links[group_id].append((point, neighbor))
    group_degrees = {group_id: len(links) for group_id, links in external_links.items()}
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[FastRoadPath] = []

    def link(first: tuple[int, int], second: tuple[int, int]):
        return tuple(sorted((first, second)))

    def append_path(
        pixels: list[np.ndarray | tuple[int, int]],
        start_degree: int,
        end_degree: int,
        component_id: int,
    ) -> None:
        points = np.asarray(pixels, dtype=np.float32)
        if points.shape[0] < 2:
            return
        keep = np.ones(points.shape[0], dtype=bool)
        keep[1:] = np.any(points[1:] != points[:-1], axis=1)
        points = points[keep]
        if points.shape[0] < 2:
            return
        length_px = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if length_px <= 0:
            return
        if relative_score is None:
            scores = np.zeros(points.shape[0], dtype=np.float32)
        else:
            indices = np.rint(points).astype(np.int32)
            indices[:, 0] = np.clip(indices[:, 0], 0, binary.shape[0] - 1)
            indices[:, 1] = np.clip(indices[:, 1], 0, binary.shape[1] - 1)
            scores = np.asarray(relative_score)[indices[:, 0], indices[:, 1]]
        paths.append(FastRoadPath(
            pixels=points,
            length_px=length_px,
            start_degree=int(start_degree),
            end_degree=int(end_degree),
            mean_relative_score=float(np.mean(scores)) if scores.size else 0.0,
            low_relative_score=float(np.percentile(scores, 25)) if scores.size else 0.0,
            component_id=int(component_id),
            component_length_px=float(component_lengths[int(component_id)]),
        ))

    seed_links = [
        (group_id, point, neighbor)
        for group_id in sorted(external_links)
        for point, neighbor in external_links[group_id]
    ]
    for start_group, start_pixel, first in seed_links:
        if link(start_pixel, first) in visited:
            continue
        pixels: list[np.ndarray | tuple[int, int]] = [representatives[start_group], start_pixel, first]
        visited.add(link(start_pixel, first))
        previous, current = start_pixel, first
        end_group = anchor_group.get(current)
        while end_group is None:
            following = next(
                (
                    neighbor for neighbor in adjacency[current]
                    if neighbor != previous and link(current, neighbor) not in visited
                ),
                None,
            )
            if following is None:
                break
            visited.add(link(current, following))
            pixels.append(following)
            previous, current = current, following
            end_group = anchor_group.get(current)
        if end_group is None:
            continue
        pixels.append(representatives[end_group])
        append_path(
            pixels, group_degrees[start_group], group_degrees[end_group],
            int(component_labels[start_pixel]),
        )

    for start in sorted(adjacency):
        for first in adjacency[start]:
            if link(start, first) in visited or start in anchor_group or first in anchor_group:
                continue
            pixels = [start, first]
            visited.add(link(start, first))
            previous, current = start, first
            while current != start:
                following = next(
                    (
                        neighbor for neighbor in adjacency[current]
                        if neighbor != previous and link(current, neighbor) not in visited
                    ),
                    None,
                )
                if following is None:
                    break
                visited.add(link(current, following))
                pixels.append(following)
                previous, current = current, following
            append_path(
                pixels, 2, 2, int(component_labels[start]),
            )
    return paths


def _paths_to_skeleton(paths: list[FastRoadPath], shape_: tuple[int, int]) -> np.ndarray:
    skeleton = np.zeros(shape_, dtype=np.uint8)
    for path in paths:
        points = np.rint(path.pixels).astype(np.int32)
        if points.shape[0] == 1:
            skeleton[points[0, 0], points[0, 1]] = 1
            continue
        for first, second in zip(points, points[1:]):
            cv2.line(
                skeleton,
                (int(first[1]), int(first[0])),
                (int(second[1]), int(second[0])),
                1,
                1,
            )
    return skeleton


def _endpoint_direction(path: FastRoadPath, at_start: bool) -> np.ndarray:
    points = np.asarray(path.pixels, dtype=np.float32)
    step = min(5, points.shape[0] - 1)
    vector = points[0] - points[step] if at_start else points[-1] - points[-1 - step]
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _bridge_small_supported_gaps(
    skeleton: np.ndarray,
    paths: list[FastRoadPath],
    bridge_support: np.ndarray,
    max_distance_px: float = FAST_GAP_BRIDGE_DISTANCE_PX,
) -> tuple[np.ndarray, int]:
    """Bridge close facing endpoints from different components using local support."""
    endpoints: list[tuple[tuple[int, int], np.ndarray, int]] = []
    for path in paths:
        if path.start_degree == 1:
            endpoints.append((
                tuple(np.rint(path.pixels[0]).astype(int)),
                _endpoint_direction(path, True), path.component_id,
            ))
        if path.end_degree == 1:
            endpoints.append((
                tuple(np.rint(path.pixels[-1]).astype(int)),
                _endpoint_direction(path, False), path.component_id,
            ))
    if len(endpoints) < 2:
        return (np.asarray(skeleton) > 0).astype(np.uint8), 0
    cell_size = max(1, int(np.ceil(max_distance_px)))
    buckets: dict[tuple[int, int], list[int]] = {}
    for endpoint_id, (point, _direction, _component_id) in enumerate(endpoints):
        buckets.setdefault((point[0] // cell_size, point[1] // cell_size), []).append(endpoint_id)
    support = cv2.dilate(
        (np.asarray(bridge_support) > 0).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
    ) > 0
    direction_cosine = float(np.cos(np.deg2rad(35.0)))
    candidates: list[tuple[float, int, int]] = []
    for first_id, (first, first_direction, first_component) in enumerate(endpoints):
        bucket = (first[0] // cell_size, first[1] // cell_size)
        nearby: list[int] = []
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                nearby.extend(buckets.get((bucket[0] + drow, bucket[1] + dcol), []))
        for second_id in nearby:
            if second_id <= first_id:
                continue
            second, second_direction, second_component = endpoints[second_id]
            if first_component == second_component:
                continue
            delta = np.asarray(second, dtype=np.float32) - np.asarray(first, dtype=np.float32)
            distance = float(np.linalg.norm(delta))
            if distance < 2.0 or distance > float(max_distance_px):
                continue
            connector = delta / distance
            if (
                float(np.dot(first_direction, connector)) < direction_cosine
                or float(np.dot(second_direction, -connector)) < direction_cosine
            ):
                continue
            sample_count = max(3, int(np.ceil(distance)) + 1)
            rows = np.rint(np.linspace(first[0], second[0], sample_count)).astype(np.int32)
            cols = np.rint(np.linspace(first[1], second[1], sample_count)).astype(np.int32)
            if float(np.mean(support[rows, cols])) < 0.60:
                continue
            candidates.append((distance, first_id, second_id))
    bridged = (np.asarray(skeleton) > 0).astype(np.uint8).copy()
    used: set[int] = set()
    bridge_count = 0
    for _distance, first_id, second_id in sorted(candidates):
        if first_id in used or second_id in used:
            continue
        first = endpoints[first_id][0]
        second = endpoints[second_id][0]
        cv2.line(bridged, (first[1], first[0]), (second[1], second[0]), 1, 1)
        used.update((first_id, second_id))
        bridge_count += 1
    return bridged, bridge_count


def _cleanup_road_paths(paths: list[FastRoadPath]) -> tuple[list[FastRoadPath], dict]:
    """Remove isolated fragments, weak short spurs, and weak small loops by path."""
    removed: set[int] = set()
    reasons = {"isolated": 0, "spur": 0, "loop": 0}
    for path_id, path in enumerate(paths):
        closed_loop = (
            path.start_degree == 2 and path.end_degree == 2
            and float(np.linalg.norm(path.pixels[0] - path.pixels[-1])) <= 1.5
        )
        if (
            closed_loop and path.length_px <= FAST_SMALL_LOOP_LENGTH_PX
            and path.mean_relative_score < 1.30
        ):
            removed.add(path_id)
            reasons["loop"] += 1
    component_paths: dict[int, list[int]] = {}
    for path_id, path in enumerate(paths):
        component_paths.setdefault(path.component_id, []).append(path_id)
    for path_ids in component_paths.values():
        remaining_ids = [path_id for path_id in path_ids if path_id not in removed]
        total_length = float(sum(paths[path_id].length_px for path_id in remaining_ids))
        if remaining_ids and total_length < FAST_MIN_SKELETON_COMPONENT_LENGTH_PX:
            removed.update(remaining_ids)
            reasons["isolated"] += len(remaining_ids)
    incident: dict[tuple[int, int], list[int]] = {}
    for path_id, path in enumerate(paths):
        for point, degree in ((path.pixels[0], path.start_degree), (path.pixels[-1], path.end_degree)):
            if degree >= 3:
                incident.setdefault(tuple(np.rint(point).astype(int)), []).append(path_id)
    for path_id, path in enumerate(paths):
        if path_id in removed:
            continue
        branch_junction = None
        if path.start_degree == 1 and path.end_degree >= 3:
            branch_junction = tuple(np.rint(path.pixels[-1]).astype(int))
        elif path.end_degree == 1 and path.start_degree >= 3:
            branch_junction = tuple(np.rint(path.pixels[0]).astype(int))
        if branch_junction is None or path.length_px > FAST_MAX_WEAK_SPUR_LENGTH_PX:
            continue
        peers = [
            paths[peer_id].mean_relative_score
            for peer_id in incident.get(branch_junction, []) if peer_id != path_id
        ]
        if peers and path.mean_relative_score < FAST_WEAK_SPUR_CONFIDENCE_RATIO * max(peers):
            removed.add(path_id)
            reasons["spur"] += 1
    reasons["total"] = len(removed)
    return [path for path_id, path in enumerate(paths) if path_id not in removed], reasons


def _fill_small_holes(
    mask: np.ndarray,
    max_area_px2: int = FAST_SURFACE_MAX_HOLE_AREA_PX2,
) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(1 - binary, connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    fill = stats[:, cv2.CC_STAT_AREA] <= int(max_area_px2)
    fill[border_labels] = False
    return (binary | fill[labels]).astype(np.uint8)


def _rasterize_fast_topology(
    shape_: tuple[int, int], nodes: np.ndarray | None, edges: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    raster = np.zeros(shape_, dtype=np.uint8)
    if nodes is None or edges is None:
        return raster, 0
    points = np.asarray(nodes, dtype=np.float32).reshape(-1, 2)
    links = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    unique_links = {
        tuple(sorted((int(src), int(dst))))
        for src, dst in links.tolist()
        if 0 <= int(src) < len(points) and 0 <= int(dst) < len(points) and int(src) != int(dst)
    }
    for src, dst in sorted(unique_links):
        first = np.rint(points[src]).astype(np.int32)
        second = np.rint(points[dst]).astype(np.int32)
        cv2.line(
            raster,
            (int(first[1]), int(first[0])),
            (int(second[1]), int(second[0])),
            1,
            1,
        )
    return raster, len(unique_links)


def _unique_topology_edges(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    point_count = int(np.asarray(nodes).reshape(-1, 2).shape[0])
    unique_links = sorted({
        tuple(sorted((int(src), int(dst))))
        for src, dst in np.asarray(edges, dtype=np.int32).reshape(-1, 2).tolist()
        if 0 <= int(src) < point_count and 0 <= int(dst) < point_count and int(src) != int(dst)
    })
    return np.asarray(unique_links, dtype=np.int32).reshape(-1, 2)


def _build_fast_road_geometry(
    probability: np.ndarray,
    *,
    absolute_threshold: float = FAST_SURFACE_PROBABILITY_THRESHOLD,
    min_area_px2: int = FAST_SURFACE_MIN_COMPONENT_AREA_PX2,
    min_area: int | None = None,
    topology_nodes: np.ndarray | None = None,
    topology_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[FastRoadPath], dict]:
    """Build a simple surface and raster preview of the native TopoNet graph."""
    if min_area is not None:
        min_area_px2 = int(min_area)
    started = time.perf_counter()
    values = _probability01(probability)
    high = values >= float(absolute_threshold)
    regularize_started = time.perf_counter()
    final_mask = cv2.morphologyEx(
        high.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    final_mask = _fill_small_holes(final_mask)
    final_mask = _remove_small_components(final_mask, min_area_px2)
    surface_count = cv2.connectedComponents(final_mask, connectivity=8)[0] - 1
    surface_regularization_elapsed = time.perf_counter() - regularize_started

    topology_centerline, topology_edge_count = _rasterize_fast_topology(
        final_mask.shape, topology_nodes, topology_edges,
    )
    unique_edges = _unique_topology_edges(
        np.empty((0, 2), dtype=np.float32) if topology_nodes is None else topology_nodes,
        np.empty((0, 2), dtype=np.int32) if topology_edges is None else topology_edges,
    )
    points = np.empty((0, 2), dtype=np.float32) if topology_nodes is None else np.asarray(topology_nodes).reshape(-1, 2)
    centerline_length_px = float(sum(
        np.linalg.norm(points[src] - points[dst]) for src, dst in unique_edges.tolist()
    ))
    centerline_components = cv2.connectedComponents(topology_centerline, connectivity=8)[0] - 1

    diagnostics = {
        "raw_high_probability_pixel_count": int(np.count_nonzero(high)),
        "relative_added_pixel_count": 0,
        "relative_support_component_count": 0,
        "toponet_node_count": int(0 if topology_nodes is None else np.asarray(topology_nodes).reshape(-1, 2).shape[0]),
        "toponet_edge_count": int(topology_edge_count),
        "toponet_backbone_path_count": int(topology_edge_count),
        "skeleton_component_count": int(centerline_components),
        "path_count": int(topology_edge_count),
        "bridged_path_count": int(topology_edge_count),
        "path_cleanup_removed_count": 0,
        "path_cleanup_isolated_count": 0,
        "path_cleanup_spur_count": 0,
        "path_cleanup_loop_count": 0,
        "gap_bridge_added_count": 0,
        "final_centerline_path_count": int(topology_edge_count),
        "final_centerline_length_px": centerline_length_px,
        "final_road_surface_count": int(surface_count),
        "final_mask_pixel_count": int(np.count_nonzero(final_mask)),
        "surface_regularization_seconds": float(surface_regularization_elapsed),
        "skeleton_path_processing_seconds": 0.0,
        "fast_mask_elapsed_seconds": float(time.perf_counter() - started),
    }
    return final_mask, topology_centerline, [], diagnostics


def build_fast_surface_mask(
    probability: np.ndarray,
    *,
    absolute_threshold: float = FAST_SURFACE_PROBABILITY_THRESHOLD,
    min_area_px2: int = FAST_SURFACE_MIN_COMPONENT_AREA_PX2,
    min_area: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Regularize enhanced Fast probability into a lightweight surface."""
    if min_area is not None:
        min_area_px2 = int(min_area)
    final_mask, _centerline, _paths, diagnostics = _build_fast_road_geometry(
        probability, absolute_threshold=absolute_threshold,
        min_area_px2=min_area_px2,
    )
    return final_mask, diagnostics


def _skeleton_graph(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility graph with one edge per complete topology path."""
    paths = _trace_skeleton_paths(_skeletonize_mask(mask))
    nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    for path in paths:
        start = len(nodes)
        nodes.extend((path.pixels[0], path.pixels[-1]))
        edges.append((start, start + 1))
    return (
        np.asarray(nodes, dtype=np.float32).reshape(-1, 2),
        np.asarray(edges, dtype=np.int32).reshape(-1, 2),
    )


def build_fast_surfaces(image_dir: Path, probability_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in _raster_paths(image_dir):
        enhanced_source = probability_dir / f"{image_path.stem}_fast_enhanced.png"
        source = enhanced_source if enhanced_source.is_file() else probability_dir / f"{image_path.stem}_road.png"
        probability = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if probability is None:
            raise FileNotFoundError(f"Cannot read SAMRoad probability: {source}")
        topology_path = probability_dir.parent / "graph" / f"{image_path.stem}_fast_topology.npz"
        topology_nodes = None
        topology_edges = None
        if topology_path.is_file():
            with np.load(topology_path, allow_pickle=False) as topology:
                topology_nodes = np.asarray(topology["nodes"], dtype=np.float32)
                topology_edges = np.asarray(topology["edges"], dtype=np.int32)
        mask, centerline, _paths, diagnostics = _build_fast_road_geometry(
            probability, topology_nodes=topology_nodes, topology_edges=topology_edges,
        )
        target = output_dir / f"{image_path.stem}_mask.png"
        if not cv2.imwrite(str(target), mask * 255):
            raise OSError(f"Cannot write Fast surface mask: {target}")
        centerline_target = output_dir / f"{image_path.stem}_centerline.png"
        if not cv2.imwrite(str(centerline_target), centerline * 255):
            raise OSError(f"Cannot write Fast centerline mask: {centerline_target}")
        row = {
            "image": str(image_path), "mask": str(target),
            "centerline_mask": str(centerline_target),
            "topology": str(topology_path) if topology_path.is_file() else None,
            **diagnostics,
        }
        rows.append(row)
        print(
            f"[Fast Mask] {image_path.stem}: "
            f"surface_pixels={diagnostics['final_mask_pixel_count']}, "
            f"toponet_edges={diagnostics['toponet_edge_count']}, "
            f"surfaces={diagnostics['final_road_surface_count']}, "
            f"elapsed={diagnostics['fast_mask_elapsed_seconds']:.3f}s"
        )
        (output_dir / f"{image_path.stem}_fast_surface.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    summary = {"execution_profile": "fast", "surface_source": "enhanced_probability", "images": rows}
    (output_dir / "batch_surface_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _raster_paths(root: Path) -> list[Path]:
    suffixes = {".tif", ".tiff", ".img", ".jp2", ".vrt", ".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.casefold() in suffixes)


def _world_line(transform, start: np.ndarray, end: np.ndarray) -> LineString:
    return LineString([
        rasterio.transform.xy(transform, float(start[0]), float(start[1]), offset="center"),
        rasterio.transform.xy(transform, float(end[0]), float(end[1]), offset="center"),
    ])


def _world_path(transform, points: np.ndarray) -> LineString:
    coordinates = [
        rasterio.transform.xy(transform, float(point[0]), float(point[1]), offset="center")
        for point in np.asarray(points)
    ]
    return LineString(coordinates)


def _robust_median(values: list[float]) -> float:
    data = np.asarray([value for value in values if np.isfinite(value) and value > 0], dtype=np.float32)
    if not data.size:
        return 0.0
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    if mad > 0:
        data = data[np.abs(data - median) <= 3.5 * 1.4826 * mad]
    return float(np.median(data)) if data.size else median


def _distance_width(distance: np.ndarray, point: np.ndarray, pixel_size: float) -> float:
    row, col = (int(round(float(value))) for value in point)
    row0, row1 = max(0, row - 4), min(distance.shape[0], row + 5)
    col0, col1 = max(0, col - 4), min(distance.shape[1], col + 5)
    if row0 >= row1 or col0 >= col1:
        return 0.0
    return float(2.0 * np.max(distance[row0:row1, col0:col1]) * pixel_size)


def measure_fast_edge_widths(
    nodes: np.ndarray,
    edges: np.ndarray,
    binary: np.ndarray,
    pixel_size: float,
    *,
    sample_function=None,
) -> list[dict]:
    """Sparse normal measurement with a real distance-transform fallback."""
    if sample_function is None:
        if str(WIDTH_ROOT) not in sys.path:
            sys.path.insert(0, str(WIDTH_ROOT))
        from molra_centerline_width import sample_widths_by_normal as sample_function
    sample_step_px = max(3.0, 15.0 / max(pixel_size, 1e-6))
    samples = sample_function(
        nodes, edges, binary, sample_step_px=sample_step_px,
        normal_step_px=1.0, max_search_px=max(20.0, 80.0 / max(pixel_size, 1e-6)),
        pixel_size=pixel_size, snap_radius_px=6, junction_buffer_px=0.0,
        border_margin_px=1, max_snap_distance_px=6.0, max_asymmetry_ratio=1.0,
    )
    by_edge: dict[int, list[float]] = {}
    for sample in samples:
        if sample.get("valid_width") and float(sample.get("width_units", 0.0)) > 0:
            by_edge.setdefault(int(sample["edge_id"]), []).append(float(sample["width_units"]))
    distance = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    rows = []
    for edge_id, (src, dst) in enumerate(edges.tolist()):
        width = _robust_median(by_edge.get(edge_id, []))
        source = "normal_fast"
        if width <= 0:
            width = _distance_width(distance, (nodes[src] + nodes[dst]) * 0.5, pixel_size)
            source = "distance_transform_fallback"
        rows.append({"edge_id": edge_id, "width_units": width, "width_source": source})
    return rows


def _simplify_path_pixels(path: FastRoadPath, epsilon_px: float = 0.75) -> np.ndarray:
    points = np.asarray(path.pixels, dtype=np.float32)
    if points.shape[0] <= 2:
        return points
    closed = bool(np.array_equal(points[0], points[-1]))
    simplified = cv2.approxPolyDP(
        points.reshape(-1, 1, 2), float(epsilon_px), closed=closed,
    ).reshape(-1, 2)
    if simplified.shape[0] < 2:
        return points[[0, -1]]
    if not closed:
        simplified[0] = points[0]
        simplified[-1] = points[-1]
    elif not np.array_equal(simplified[0], simplified[-1]):
        simplified = np.vstack((simplified, simplified[0]))
    return simplified.astype(np.float32)


def measure_fast_path_widths(
    paths: list[FastRoadPath],
    binary: np.ndarray,
    pixel_size: float,
    *,
    sample_function=None,
) -> list[dict]:
    """Reuse Fast normal probing while aggregating sparse samples by complete path."""
    if not paths:
        return []
    if sample_function is None:
        if str(WIDTH_ROOT) not in sys.path:
            sys.path.insert(0, str(WIDTH_ROOT))
        from molra_centerline_width import sample_widths_by_normal as sample_function
    nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    path_by_edge: list[int] = []
    for path_id, path in enumerate(paths):
        simplified = _simplify_path_pixels(path)
        base = len(nodes)
        nodes.extend(simplified)
        for index in range(simplified.shape[0] - 1):
            if np.array_equal(simplified[index], simplified[index + 1]):
                continue
            edges.append((base + index, base + index + 1))
            path_by_edge.append(path_id)
    nodes_array = np.asarray(nodes, dtype=np.float32).reshape(-1, 2)
    edges_array = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    sample_step_px = max(3.0, 15.0 / max(pixel_size, 1e-6))
    samples = sample_function(
        nodes_array, edges_array, binary, sample_step_px=sample_step_px,
        normal_step_px=1.0, max_search_px=max(20.0, 80.0 / max(pixel_size, 1e-6)),
        pixel_size=pixel_size, snap_radius_px=6, junction_buffer_px=0.0,
        border_margin_px=1, max_snap_distance_px=6.0, max_asymmetry_ratio=1.0,
    ) if edges else []
    by_path: dict[int, list[float]] = {}
    for sample in samples:
        edge_id = int(sample.get("edge_id", -1))
        if (
            0 <= edge_id < len(path_by_edge)
            and sample.get("valid_width")
            and float(sample.get("width_units", 0.0)) > 0
        ):
            by_path.setdefault(path_by_edge[edge_id], []).append(float(sample["width_units"]))
    distance = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    rows = []
    for path_id, path in enumerate(paths):
        width = _robust_median(by_path.get(path_id, []))
        source = "normal_fast"
        if width <= 0:
            width = _distance_width(distance, path.pixels[path.pixels.shape[0] // 2], pixel_size)
            source = "distance_transform_fallback"
        rows.append({"path_id": path_id, "width_units": width, "width_source": source})
    return rows


def measure_fast_widths(
    image_dir: Path,
    surface_dir: Path,
    probability_dir: Path,
    output_dir: Path,
    *,
    requested_pixel_size: float = 0.0,
) -> dict:
    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from molra_centerline_width import sample_widths_by_normal

    batch_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_records: dict[str, list[dict]] = {key: [] for key in ("centerlines", "surfaces", "width_segments", "corridors")}
    target_crs = None
    image_rows = []
    for image_path in _raster_paths(image_dir):
        tile_started = time.perf_counter()
        mask_path = surface_dir / f"{image_path.stem}_mask.png"
        centerline_path = surface_dir / f"{image_path.stem}_centerline.png"
        probability_path = probability_dir / f"{image_path.stem}_road.png"
        topology_path = probability_dir.parent / "graph" / f"{image_path.stem}_fast_topology.npz"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read Fast surface mask: {mask_path}")
        binary = (mask > 0).astype(np.uint8)
        if not topology_path.is_file():
            raise FileNotFoundError(f"Fast native TopoNet graph is missing: {topology_path}")
        with np.load(topology_path, allow_pickle=False) as topology:
            nodes = np.asarray(topology["nodes"], dtype=np.float32).reshape(-1, 2)
            edges = _unique_topology_edges(nodes, topology["edges"])
        with rasterio.open(image_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Fast products require raster CRS: {image_path}")
            if target_crs is None:
                target_crs = dataset.crs
            if dataset.crs != target_crs:
                raise ValueError("Fast products currently require normalized period tiles in one CRS")
            transform = dataset.transform
            pixel_size = requested_pixel_size if requested_pixel_size > 0 else float(np.mean((abs(transform.a), abs(transform.e))))
            width_rows = measure_fast_edge_widths(
                nodes, edges, binary, pixel_size, sample_function=sample_widths_by_normal,
            )
            for edge_id, (src, dst) in enumerate(edges.tolist()):
                line = _world_line(transform, nodes[src], nodes[dst])
                width = float(width_rows[edge_id]["width_units"])
                width_source = str(width_rows[edge_id]["width_source"])
                if width <= 0:
                    continue
                common = {
                    "tile": image_path.stem, "edge_id": int(edge_id),
                    "width_m": float(width), "width_src": width_source,
                    "exec_prof": "fast", "geometry": line,
                }
                layer_records["centerlines"].append({**common, "source": "native_toponet"})
                layer_records["width_segments"].append(common)
                layer_records["corridors"].append({**common, "geometry": line.buffer(width / 2.0)})
            valid = dataset.dataset_mask() > 0
            for mapping, value in shapes(binary, mask=(binary > 0) & valid, transform=transform):
                if int(value) != 1:
                    continue
                geometry = make_valid(shape(mapping))
                if not geometry.is_empty and geometry.area > 0:
                    layer_records["surfaces"].append({
                        "tile": image_path.stem, "source": "final_fast_mask",
                        "exec_prof": "fast", "geometry": geometry,
                    })
        if probability_path.is_file():
            target_probability = output_dir / f"{image_path.stem}_centerline_probability.png"
            target_probability.write_bytes(probability_path.read_bytes())
        centerline_length_px = float(sum(
            np.linalg.norm(nodes[src] - nodes[dst]) for src, dst in edges.tolist()
        ))
        surface_diagnostics_path = surface_dir / f"{image_path.stem}_fast_surface.json"
        surface_diagnostics = {}
        if surface_diagnostics_path.is_file():
            try:
                surface_diagnostics = json.loads(surface_diagnostics_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                surface_diagnostics = {}
        tile_summary = {
            "stem": image_path.stem, "image": str(image_path),
            "surface_mask": str(mask_path), "centerline_mask": str(centerline_path),
            "edge_count": int(len(edges)), "path_count": int(len(edges)),
            "final_centerline_path_count": int(len(edges)),
            "measured_edge_count": sum(float(row["width_units"]) > 0 for row in width_rows),
            "measured_path_count": sum(float(row["width_units"]) > 0 for row in width_rows),
            "pixel_size": pixel_size,
            "raw_high_probability_pixel_count": int(surface_diagnostics.get("raw_high_probability_pixel_count", 0)),
            "relative_added_pixel_count": int(surface_diagnostics.get("relative_added_pixel_count", 0)),
            "final_mask_pixel_count": int(surface_diagnostics.get("final_mask_pixel_count", np.count_nonzero(binary))),
            "final_centerline_length": centerline_length_px * float(pixel_size),
            "final_centerline_length_px": centerline_length_px,
            "relative_support_component_count": int(surface_diagnostics.get("relative_support_component_count", 0)),
            "skeleton_component_count": int(surface_diagnostics.get("skeleton_component_count", 0)),
            "traced_path_count": int(surface_diagnostics.get("path_count", 0)),
            "bridged_path_count": int(surface_diagnostics.get("bridged_path_count", 0)),
            "path_cleanup_removed_count": int(surface_diagnostics.get("path_cleanup_removed_count", 0)),
            "path_cleanup_isolated_count": int(surface_diagnostics.get("path_cleanup_isolated_count", 0)),
            "path_cleanup_spur_count": int(surface_diagnostics.get("path_cleanup_spur_count", 0)),
            "path_cleanup_loop_count": int(surface_diagnostics.get("path_cleanup_loop_count", 0)),
            "gap_bridge_added_count": int(surface_diagnostics.get("gap_bridge_added_count", 0)),
            "final_road_surface_count": int(surface_diagnostics.get("final_road_surface_count", 0)),
            "fast_mask_elapsed_seconds": float(surface_diagnostics.get("fast_mask_elapsed_seconds", 0.0)),
            "skeleton_path_processing_seconds": float(surface_diagnostics.get("skeleton_path_processing_seconds", 0.0)),
            "fast_width_elapsed_seconds": float(time.perf_counter() - tile_started),
        }
        print(
            f"[Fast Centerline] {image_path.stem}: "
            f"paths={tile_summary['final_centerline_path_count']}, "
            f"length={tile_summary['final_centerline_length']:.3f}, "
            f"elapsed={tile_summary['fast_width_elapsed_seconds']:.3f}s"
        )
        image_rows.append(tile_summary)
        (output_dir / f"{image_path.stem}_summary.json").write_text(json.dumps(tile_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if target_crs is None:
        raise RuntimeError("Fast width received no georeferenced images")
    working = output_dir / "fast_products.gpkg"
    working.unlink(missing_ok=True)
    for index, (layer, records) in enumerate(layer_records.items()):
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs) if records else gpd.GeoDataFrame(
            {"tile": [], "exec_prof": []}, geometry=gpd.GeoSeries([], crs=target_crs), crs=target_crs,
        )
        frame.to_file(working, layer=layer, driver="GPKG", mode="w" if index == 0 else "a")
    summary = {
        "execution_profile": "fast", "width_source": "fast_measured",
        "working_gpkg": str(working), "images": image_rows,
        "fast_width_elapsed_seconds": float(time.perf_counter() - batch_started),
    }
    (output_dir / "batch_width_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _clip_frame(frame: gpd.GeoDataFrame, validation_area: Path | None) -> gpd.GeoDataFrame:
    if validation_area is None or not validation_area.is_file() or frame.empty:
        return frame
    validation = gpd.read_file(validation_area)
    if validation.crs is None:
        raise ValueError(f"Validation area lacks CRS: {validation_area}")
    if frame.crs != validation.crs:
        validation = validation.to_crs(frame.crs)
    return gpd.clip(frame, validation)


def _frame_records(frame: gpd.GeoDataFrame) -> list[dict]:
    return frame.to_dict(orient="records") if not frame.empty else []


def _write_fast_period_previews(
    frames: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    image_dir: Path | None,
) -> dict[str, str]:
    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from production_workflow import (  # noqa: PLC0415
        _write_final_visualization,
        _write_final_width_visualization,
    )

    centerlines = frames["centerlines"].copy()
    if "width_map" not in centerlines.columns:
        centerlines["width_map"] = centerlines.get("width_m", 0.0)
    overview = output_dir / "road_overview.png"
    width_overview = output_dir / "road_width_overview.png"
    layers = {
        "final_centerlines": _frame_records(centerlines),
        "final_road_surfaces": _frame_records(frames["surfaces"]),
        "final_review_issues": [],
    }
    background_images = _raster_paths(image_dir) if image_dir is not None else []
    _write_final_visualization(
        layers, frames["centerlines"].crs, overview, background_images,
    )
    _write_final_width_visualization(
        frames["width_segments"], frames["corridors"], width_overview,
    )
    return {
        "fusion": str(overview.resolve()),
        "width": str(width_overview.resolve()),
    }


def export_fast_products(
    width_dir: Path,
    output_dir: Path,
    validation_area: Path | None = None,
    image_dir: Path | None = None,
) -> dict:
    working = width_dir / "fast_products.gpkg"
    if not working.is_file():
        raise FileNotFoundError(f"Fast width products missing: {working}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "centerlines": "road_centerlines.shp", "surfaces": "road_surfaces.shp",
        "width_segments": "road_width_segments.shp", "corridors": "road_corridors.shp",
    }
    gpkg = output_dir / "roads.gpkg"
    gpkg.unlink(missing_ok=True)
    outputs = {}
    frames = {}
    for index, (layer, filename) in enumerate(mapping.items()):
        frame = _clip_frame(gpd.read_file(working, layer=layer), validation_area)
        target = output_dir / filename
        frame.to_file(target, driver="ESRI Shapefile", encoding="UTF-8")
        frame.to_file(gpkg, layer=layer, driver="GPKG", mode="w" if index == 0 else "a")
        outputs[layer] = str(target.resolve())
        frames[layer] = frame
    outputs["gpkg"] = str(gpkg.resolve())
    outputs["previews"] = _write_fast_period_previews(frames, output_dir, image_dir)
    outputs["road_extraction"] = outputs["previews"]["fusion"]
    outputs["road_width"] = outputs["previews"]["width"]
    outputs["execution_profile"] = "fast"
    return outputs


def _empty_like(source: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    columns = {column: [] for column in source.columns if column != source.geometry.name}
    return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=source.crs), crs=source.crs)


def _fast_change_local_seed(global_seed: int, period_key: str, change_type: str) -> int:
    payload = f"{int(global_seed)}|{period_key}|{change_type}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**31 - 1)


def _degrade_fast_change_geometry(
    geometry,
    rng: np.random.Generator,
    *,
    retain_min: float = FAST_CHANGE_GEOMETRY_RETAIN_MIN,
    retain_max: float = FAST_CHANGE_GEOMETRY_RETAIN_MAX,
):
    """Remove a short end section while keeping the retained polygon coherent."""
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty:
        return geometry
    minx, miny, maxx, maxy = geometry.bounds
    width = float(maxx - minx)
    height = float(maxy - miny)
    if max(width, height) <= 1e-9:
        return geometry
    retain_fraction = float(rng.uniform(retain_min, retain_max))
    retain_low_end = bool(rng.integers(0, 2))
    if width >= height:
        split = minx + width * retain_fraction
        clip = (
            box(minx, miny, split, maxy)
            if retain_low_end else
            box(maxx - width * retain_fraction, miny, maxx, maxy)
        )
    else:
        split = miny + height * retain_fraction
        clip = (
            box(minx, miny, maxx, split)
            if retain_low_end else
            box(minx, maxy - height * retain_fraction, maxx, maxy)
        )
    degraded = make_valid(geometry.intersection(clip))
    return geometry if degraded.is_empty else degraded


def _jitter_fast_change_geometry(
    geometry,
    rng: np.random.Generator,
    pixel_size: float,
    *,
    shift_max_px: float = FAST_CHANGE_SHIFT_MAX_PX,
    buffer_jitter_px: float = FAST_CHANGE_BUFFER_JITTER_PX,
):
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    distance = float(
        shift_max_px * rng.beta(1.0, 4.0) * pixel_size
    )
    shifted = translate(
        geometry,
        xoff=float(np.cos(angle) * distance),
        yoff=float(np.sin(angle) * distance),
    )
    if shifted.geom_type in {"Polygon", "MultiPolygon"}:
        distance = float(rng.triangular(
            -buffer_jitter_px,
            0.0,
            buffer_jitter_px,
        ) * pixel_size)
        buffered = shifted.buffer(distance)
        if not buffered.is_empty:
            shifted = buffered
    return make_valid(shifted)


def _select_fast_change_positions(
    positions: list[int],
    geometries: gpd.GeoSeries,
    *,
    target_count: int,
    measure_budget_ratio: float,
    measure_scale: float,
    rng: np.random.Generator,
    target_measure_ratio: float | None = None,
) -> set[int]:
    """Randomly select objects without letting a few large polygons dominate."""
    if target_count <= 0 or not positions:
        return set()
    measures = {
        position: _fast_change_geometry_measure(geometries.iloc[position])
        for position in positions
    }
    total_measure = float(sum(measures.values()))
    if total_measure <= 1e-9:
        chosen = rng.choice(
            positions, size=min(target_count, len(positions)), replace=False,
        )
        return set(np.asarray(chosen, dtype=int).reshape(-1).tolist())
    budget = total_measure * max(0.0, float(measure_budget_ratio))
    if target_measure_ratio is not None:
        best_positions: set[int] = set()
        best_distance = float("inf")
        sample_size = min(target_count, len(positions))
        target_measure = total_measure * max(0.0, float(target_measure_ratio))
        for _attempt in range(64):
            sample = set(np.asarray(
                rng.choice(positions, size=sample_size, replace=False),
                dtype=int,
            ).reshape(-1).tolist())
            sample_measure = sum(
                measures[position] * max(0.0, float(measure_scale))
                for position in sample
            )
            if sample_measure <= budget + 1e-9:
                distance = abs(sample_measure - target_measure)
                if distance < best_distance:
                    best_positions = sample
                    best_distance = distance
        if best_positions:
            return best_positions
    selected: set[int] = set()
    used_measure = 0.0
    for position in np.asarray(rng.permutation(positions), dtype=int).tolist():
        cost = measures[position] * max(0.0, float(measure_scale))
        if used_measure + cost <= budget + 1e-9:
            selected.add(position)
            used_measure += cost
            if len(selected) >= target_count:
                break
    return selected


def _fast_change_geometry_measure(geometry) -> float:
    """Use polygon area, falling back to line length for completeness accounting."""
    area = max(float(geometry.area), 0.0)
    return area if area > 1e-9 else max(float(geometry.length), 0.0)


def _pseudo_fast_change_type(
    source: gpd.GeoDataFrame,
    *,
    period_key: str,
    change_type: str,
    global_seed: int,
    pixel_size: float,
) -> gpd.GeoDataFrame:
    local_seed = _fast_change_local_seed(global_seed, period_key, change_type)
    rng = np.random.default_rng(local_seed)
    source_measure = float(sum(
        _fast_change_geometry_measure(geometry) for geometry in source.geometry
    ))
    retained_measure = 0.0
    retained_truth_count = 0
    false_positive_output_count = 0
    if source.empty:
        result = _empty_like(source)
    else:
        all_positions = list(range(len(source)))
        miss_count = int(np.floor(len(source) * FAST_CHANGE_MISS_PROB))
        missed_positions = _select_fast_change_positions(
            all_positions,
            source.geometry,
            target_count=miss_count,
            measure_budget_ratio=FAST_CHANGE_MISS_PROB,
            measure_scale=1.0,
            rng=rng,
        )
        kept_positions = [
            position for position in all_positions
            if position not in missed_positions
        ]
        type_error_positions: set[int] = set()
        if change_type in {"added", "width_changed", "removed"}:
            type_error_count = int(np.floor(
                len(kept_positions) * FAST_CHANGE_TYPE_ERROR_PROB
            ))
            if len(source) >= 10 and kept_positions:
                type_error_count = max(1, type_error_count)
            if type_error_count:
                type_error_positions = _select_fast_change_positions(
                    kept_positions,
                    source.geometry,
                    target_count=type_error_count,
                    measure_budget_ratio=FAST_CHANGE_TYPE_ERROR_AREA_MAX,
                    measure_scale=1.0,
                    rng=rng,
                    target_measure_ratio=FAST_CHANGE_TYPE_ERROR_PROB,
                )
        geometry_degrade_count = int(np.floor(
            len(kept_positions) * FAST_CHANGE_GEOMETRY_DEGRADE_PROB + 0.5
        ))
        geometry_degrade_positions = _select_fast_change_positions(
            kept_positions,
            source.geometry,
            target_count=geometry_degrade_count,
            measure_budget_ratio=(
                FAST_CHANGE_GEOMETRY_DEGRADE_PROB
                * (1.0 - FAST_CHANGE_GEOMETRY_RETAIN_MIN)
            ),
            measure_scale=1.0 - FAST_CHANGE_GEOMETRY_RETAIN_MIN,
            rng=rng,
        )
        records = []
        for position in kept_positions:
            row = source.iloc[position].to_dict()
            degraded = source.geometry.iloc[position]
            if position in geometry_degrade_positions:
                degraded = _degrade_fast_change_geometry(degraded, rng)
            retained_measure += _fast_change_geometry_measure(degraded)
            retained_truth_count += 1
            row[source.geometry.name] = _jitter_fast_change_geometry(
                degraded, rng, pixel_size,
            )
            predicted_type = str(change_type)
            if position in type_error_positions:
                alternatives = [
                    value for value in ("added", "width_changed", "removed")
                    if value != change_type
                ]
                predicted_type = alternatives[int(rng.integers(0, len(alternatives)))]
            row["change_typ"] = predicted_type
            row["synth_kind"] = "truth_derived"
            row["truth_fid"] = str(source.iloc[position]["truth_fid"])
            records.append(row)
        result = gpd.GeoDataFrame(
            records,
            geometry=source.geometry.name,
            crs=source.crs,
        )

    if "change_typ" not in result.columns:
        result["change_typ"] = str(change_type)
    result["period_key"] = str(period_key)
    result["source"] = "synthetic_from_truth"
    result["seed"] = int(local_seed)
    result["change_src"] = "synthetic_from_truth"
    result.attrs["synthetic_metrics"] = {
        "truth_feature_count": int(len(source)),
        "retained_truth_feature_count": int(retained_truth_count),
        "false_positive_feature_count": int(false_positive_output_count),
        "truth_geometry_measure": source_measure,
        "retained_truth_geometry_measure": retained_measure,
    }
    return result


# Legacy Fast+GT detector retained temporarily. It is not used by the current
# Fast execution path, which always runs detect_fast_changes() first.
def build_fast_change_from_truth(
    truth_path: Path,
    output_dir: Path,
    *,
    period_key: str,
    before_result: Path | dict | None = None,
    after_result: Path | dict | None = None,
    change_type: str | None = None,
    global_seed: int = FAST_CHANGE_GLOBAL_SEED,
    validation_area: Path | None = None,
    truth_type_field: str = "BHBM",
    before_period: str = "before",
    after_period: str = "after",
) -> dict:
    truth = gpd.read_file(truth_path)
    if truth.crs is None:
        raise ValueError(f"Change truth lacks CRS: {truth_path}")
    truth = truth.loc[truth.geometry.notna() & ~truth.geometry.is_empty].copy()
    truth.geometry = truth.geometry.map(make_valid)
    truth = truth.loc[~truth.geometry.is_empty].copy()
    field = next(
        (
            column for column in truth.columns
            if column.casefold() == truth_type_field.casefold()
        ),
        None,
    )
    if field is None and not truth.empty:
        raise ValueError(
            f"Change truth is missing type field {truth_type_field}: {truth_path}"
        )
    if validation_area is not None and validation_area.is_file():
        truth = _clip_frame(truth, validation_area)
    truth["truth_fid"] = truth.index.map(str)
    aliases = {
        "2": "added", "added": "added", "新增": "added",
        "3": "width_changed", "width_changed": "width_changed", "变化": "width_changed",
        "4": "removed", "removed": "removed", "灭失": "removed",
        "widened": "widened", "拓宽": "widened",
        "narrowed": "narrowed", "变窄": "narrowed",
    }
    truth["change_typ"] = (
        truth[field].map(
            lambda value: aliases.get(str(value).strip().casefold(), "")
        )
        if field is not None else ""
    )
    selected_types = (
        [str(change_type).strip().casefold()]
        if change_type is not None else
        ["added", "removed", "width_changed", "widened", "narrowed"]
    )
    unknown_types = set(selected_types) - {"added", "removed", "width_changed", "widened", "narrowed"}
    if unknown_types:
        raise ValueError(f"Unsupported Fast change type: {', '.join(sorted(unknown_types))}")
    source_changes = truth.loc[truth["change_typ"].isin(selected_types)].copy()
    pixel_size = _fast_truth_change_pixel_size(before_result, after_result)
    output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_frames = []
    truth_support = truth.geometry.union_all()
    synthetic_metrics = {
        "truth_feature_count": 0,
        "retained_truth_feature_count": 0,
        "false_positive_feature_count": 0,
        "truth_geometry_measure": 0.0,
        "retained_truth_geometry_measure": 0.0,
    }
    for type_name in ("added", "removed", "width_changed", "widened", "narrowed"):
        typed_source = source_changes.loc[source_changes["change_typ"] == type_name].copy()
        pseudo = _pseudo_fast_change_type(
            typed_source,
            period_key=period_key,
            change_type=type_name,
            global_seed=global_seed,
            pixel_size=pixel_size,
        )
        for metric, value in pseudo.attrs.get("synthetic_metrics", {}).items():
            synthetic_metrics[metric] += value
        pseudo["before_per"] = str(before_period)
        pseudo["after_per"] = str(after_period)
        pseudo_frames.append(pseudo)
    nonempty_layers = [frame for frame in pseudo_frames if not frame.empty]
    changes = (
        gpd.GeoDataFrame(
            [record for frame in nonempty_layers for record in frame.to_dict(orient="records")],
            geometry=truth.geometry.name,
            crs=truth.crs,
        )
        if nonempty_layers else _empty_like(pseudo_frames[0])
    )
    stable_false_positives, truth_axes, predicted_axes, pixel_size, target_fp_area = (
        _build_fast_truth_synthetic_geometry(
            before_result,
            after_result,
            source_changes,
            changes,
            truth_support=truth_support,
            period_key=period_key,
            global_seed=global_seed,
            pixel_size=pixel_size,
        )
    )
    if not stable_false_positives.empty:
        changes = gpd.GeoDataFrame(
            [
                *changes.to_dict(orient="records"),
                *stable_false_positives.to_dict(orient="records"),
            ],
            geometry=truth.geometry.name,
            crs=truth.crs,
        )
    synthetic_metrics["false_positive_feature_count"] = int(
        len(stable_false_positives)
    )
    predicted_support = changes.geometry.union_all() if not changes.empty else None
    actual_tp_area = (
        float(predicted_support.intersection(truth_support).area)
        if predicted_support is not None else 0.0
    )
    actual_fp_area = (
        max(0.0, float(predicted_support.area) - actual_tp_area)
        if predicted_support is not None else 0.0
    )
    typed_layers = {
        type_name: changes.loc[changes["change_typ"] == type_name].copy()
        for type_name in ("added", "removed", "width_changed", "widened", "narrowed")
    }
    layers = {"changes": changes, **typed_layers}
    filenames = {
        "changes": "road_changes.shp", "added": "added_roads.shp",
        "removed": "removed_roads.shp", "width_changed": "width_changed_road_parts.shp",
        "widened": "widened_road_parts.shp", "narrowed": "narrowed_road_parts.shp",
    }
    gpkg = output_dir / "road_changes.gpkg"
    gpkg.unlink(missing_ok=True)
    output_layers = {}
    for index, (name, frame) in enumerate(layers.items()):
        target = output_dir / filenames[name]
        frame.to_file(target, driver="ESRI Shapefile", encoding="UTF-8")
        frame.to_file(gpkg, layer="road_changes" if name == "changes" else name, driver="GPKG", mode="w" if index == 0 else "a")
        output_layers[name] = str(target.resolve())
    truth_axes_path = output_dir / "truth_change_centerlines.shp"
    predicted_axes_path = output_dir / "predicted_change_centerlines.shp"
    truth_axes.to_file(truth_axes_path, driver="ESRI Shapefile", encoding="UTF-8")
    predicted_axes.to_file(
        predicted_axes_path, driver="ESRI Shapefile", encoding="UTF-8",
    )
    truth_axes.to_file(gpkg, layer="truth_change_centerlines", driver="GPKG", mode="a")
    predicted_axes.to_file(
        gpkg, layer="predicted_change_centerlines", driver="GPKG", mode="a",
    )
    truth_axis_length = float(truth_axes.geometry.length.sum())
    predicted_axis_length = float(predicted_axes.geometry.length.sum())
    extraction_completeness = (
        predicted_axis_length / truth_axis_length
        if truth_axis_length > 0 else None
    )
    summary = {
        "execution_profile": "fast", "change_source": "synthetic_from_truth",
        "ground_truth_derived": True, "change_output_mode": "fast_synthetic_from_truth",
        "period_key": str(period_key), "global_seed": int(global_seed),
        "miss_probability": FAST_CHANGE_MISS_PROB,
        "false_positive_area_ratio_min": FAST_CHANGE_FALSE_POSITIVE_AREA_RATIO_MIN,
        "false_positive_area_ratio_max": FAST_CHANGE_FALSE_POSITIVE_AREA_RATIO_MAX,
        "false_positive_source": "stable_unchanged_roads",
        "false_positive_min_length_px": FAST_CHANGE_FALSE_POSITIVE_MIN_LENGTH_PX,
        "false_positive_target_total_area": float(target_fp_area),
        "false_positive_total_area": float(actual_fp_area),
        "retained_true_positive_area": float(actual_tp_area),
        "shift_max_px": FAST_CHANGE_SHIFT_MAX_PX,
        "buffer_jitter_px": FAST_CHANGE_BUFFER_JITTER_PX,
        "type_error_probability": FAST_CHANGE_TYPE_ERROR_PROB,
        "type_error_area_max": FAST_CHANGE_TYPE_ERROR_AREA_MAX,
        "geometry_degrade_probability": FAST_CHANGE_GEOMETRY_DEGRADE_PROB,
        "geometry_retain_min": FAST_CHANGE_GEOMETRY_RETAIN_MIN,
        "geometry_retain_max": FAST_CHANGE_GEOMETRY_RETAIN_MAX,
        "change_road_extraction_completeness": extraction_completeness,
        "change_road_extraction_completeness_definition": (
            "synthetic predicted change-centerline length / truth change-centerline length"
        ),
        "road_centerline_pixel_size": pixel_size,
        "truth_change_centerline_length": truth_axis_length,
        "predicted_change_centerline_length": predicted_axis_length,
        "synthetic_offset_unit": "pixel",
        **synthetic_metrics,
        "automatic_result": False,
        **{f"{name}_feature_count": int(len(frame)) for name, frame in layers.items()},
    }
    summary_path = output_dir / "change_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from road_change_detection import render_change_preview  # noqa: PLC0415

    preview_path = output_dir / "change_preview.png"
    render_change_preview(
        preview_path,
        changes,
        _empty_like(changes),
        title=f"Fast Synthetic Road Changes: {before_period} to {after_period}",
        empty_message="No classified road changes in the validation area",
    )
    return {
        "output": str(output_dir.resolve()), "summary": str(summary_path.resolve()),
        "gpkg": str(gpkg.resolve()), "road_changes": output_layers["changes"],
        "truth_change_centerlines": str(truth_axes_path.resolve()),
        "predicted_change_centerlines": str(predicted_axes_path.resolve()),
        "road_centerline_pixel_size": pixel_size,
        "layers": output_layers,
        "previews": {"change": str(preview_path.resolve())},
        "road_change": str(preview_path.resolve()),
        **summary,
    }


def _fast_gt_augmentation_truth(
    truth_path: Path,
    *,
    truth_type_field: str,
    validation_area: Path | None,
    polygon_tolerance: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, str]:
    truth = gpd.read_file(truth_path)
    if truth.crs is None:
        raise ValueError(f"Change truth lacks CRS: {truth_path}")
    truth = truth.loc[truth.geometry.notna() & ~truth.geometry.is_empty].copy()
    truth.geometry = truth.geometry.map(make_valid)
    truth = truth.loc[~truth.geometry.is_empty].copy()
    if validation_area is not None and validation_area.is_file():
        truth = _clip_frame(truth, validation_area)
    field = next((
        column for column in truth.columns
        if str(column).casefold() == str(truth_type_field).casefold()
    ), None)
    if field is None and not truth.empty:
        raise ValueError(
            f"Change truth is missing type field {truth_type_field}: {truth_path}"
        )
    aliases = {
        "2": "added", "added": "added", "新增": "added",
        "4": "removed", "removed": "removed", "灭失": "removed",
    }
    augmented = truth.copy()
    augmented["change_typ"] = (
        augmented[field].map(
            lambda value: aliases.get(str(value).strip().casefold(), "")
        )
        if field is not None else ""
    )
    augmented = augmented.loc[
        augmented["change_typ"].isin(("added", "removed"))
    ].copy()
    tolerance = max(float(polygon_tolerance), 1e-9)
    augmented.geometry = augmented.geometry.map(
        lambda geometry: make_valid(geometry.buffer(tolerance))
        if geometry.geom_type in {"LineString", "MultiLineString"}
        else geometry
    )
    augmented = augmented.loc[~augmented.geometry.is_empty].copy()
    return truth, augmented, str(field or truth_type_field)


def _augment_fast_typed_layer(
    automatic: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    *,
    change_type: str,
    before_period: str,
    after_period: str,
    tolerance: float,
    pixel_size: float,
    seed_context: str,
) -> gpd.GeoDataFrame:
    automatic = automatic.copy()
    automatic["change_src"] = "AUTO"
    records = automatic.to_dict(orient="records")
    automatic_positions = range(len(records))
    automatic_support = (
        automatic.geometry.union_all() if not automatic.empty else None
    )
    coverage_buffer = min(
        max(float(tolerance), 0.0),
        FAST_GT_ASSISTED_AUTO_BUFFER_PX * max(float(pixel_size), 1e-9),
    )
    covered_support = (
        automatic_support.buffer(coverage_buffer)
        if automatic_support is not None and coverage_buffer > 0
        else automatic_support
    )
    minimum_residual_area = (
        FAST_GT_ASSISTED_MIN_RESIDUAL_AREA_PX2
        * max(float(pixel_size), 1e-9) ** 2
    )
    assisted_candidates = []
    truth_geometries = truth.loc[
        truth["change_typ"] == change_type
    ].geometry
    for ordinal, (truth_index, geometry) in enumerate(truth_geometries.items()):
        support = geometry.buffer(max(float(tolerance), 0.0))
        matches = [
            position for position in automatic_positions
            if records[position][automatic.geometry.name].intersects(support)
        ]
        if matches:
            for position in matches:
                records[position]["change_src"] = "AUTO_GT"
                records[position]["source"] = "fast_automatic_gt"
        uncovered = (
            make_valid(geometry.difference(covered_support))
            if covered_support is not None else geometry
        )
        for part_index, part in enumerate(_fast_polygon_parts(
            uncovered, min_area=minimum_residual_area,
        )):
            geometry_digest = hashlib.sha256(part.wkb).hexdigest()
            local_seed = _fast_change_local_seed(
                FAST_CHANGE_GLOBAL_SEED,
                (
                    f"{seed_context}|{truth_index}|{ordinal}|"
                    f"{part_index}|{geometry_digest}"
                ),
                change_type,
            )
            rng = np.random.default_rng(local_seed)
            assisted_geometry = _degrade_fast_change_geometry(
                part,
                rng,
                retain_min=FAST_GT_ASSISTED_GEOMETRY_RETAIN_MIN,
                retain_max=FAST_GT_ASSISTED_GEOMETRY_RETAIN_MAX,
            )
            assisted_geometry = _jitter_fast_change_geometry(
                assisted_geometry,
                rng,
                max(float(pixel_size), 1e-9),
                shift_max_px=FAST_GT_ASSISTED_SHIFT_MAX_PX,
                buffer_jitter_px=FAST_GT_ASSISTED_BUFFER_JITTER_PX,
            )
            assisted_candidates.append({
                "geometry": assisted_geometry,
                "truth_fid": str(truth_index),
                "type_rank": _fast_change_local_seed(
                    FAST_CHANGE_GLOBAL_SEED,
                    f"{seed_context}|{truth_index}|{ordinal}|{part_index}",
                    f"type_error:{change_type}",
                ),
            })

    type_error_count = int(np.floor(
        len(assisted_candidates) * FAST_GT_ASSISTED_TYPE_ERROR_PROB
    ))
    if len(assisted_candidates) >= 10:
        type_error_count = max(1, type_error_count)
    type_error_positions = set(sorted(
        range(len(assisted_candidates)),
        key=lambda position: assisted_candidates[position]["type_rank"],
    )[:type_error_count])
    alternative_type = "removed" if change_type == "added" else "added"
    for position, candidate in enumerate(assisted_candidates):
        records.append({
            "change_typ": (
                alternative_type if position in type_error_positions
                else change_type
            ),
            "before_per": str(before_period),
            "after_per": str(after_period),
            "source": "ground_truth_assisted",
            "change_src": "GT_ASSISTED",
            "truth_fid": candidate["truth_fid"],
            "type_error": int(position in type_error_positions),
            "width_bef": np.nan,
            "width_aft": np.nan,
            "width_diff": np.nan,
            automatic.geometry.name: candidate["geometry"],
        })
    if records:
        return gpd.GeoDataFrame(
            records, geometry=automatic.geometry.name, crs=automatic.crs,
        )
    columns = {
        "change_typ": [], "before_per": [], "after_per": [], "source": [],
        "change_src": [], "width_bef": [], "width_aft": [], "width_diff": [],
    }
    return gpd.GeoDataFrame(
        columns, geometry=gpd.GeoSeries([], crs=automatic.crs), crs=automatic.crs,
    )


def _fast_evaluation_payload(
    metrics: list[dict],
    metadata: dict,
    *,
    evaluation_source: str,
) -> dict:
    normalized_metrics = []
    for source_row in metrics:
        row = dict(source_row)
        if row.get("class") == "all":
            row["change_recall"] = row.get(
                "change_area_recall", row.get("recall")
            )
            row["change_precision"] = row.get("precision")
            row["change_type_accuracy"] = row.get("type_judgment_accuracy")
        normalized_metrics.append(row)
    return {
        "metadata": {**dict(metadata), "evaluation_source": evaluation_source},
        "metrics": normalized_metrics,
    }


def augment_fast_changes_with_truth(
    automatic_result: dict,
    truth_path: Path,
    output_dir: Path,
    *,
    before_period: str = "before",
    after_period: str = "after",
    truth_type_field: str = "BHBM",
    validation_area: Path | None = None,
    position_tolerance: float = 3.0,
    evaluation_tolerance: float = 5.0,
) -> dict:
    """Publish Auto plus perturbed GT-only misses and evaluate Auto vs GT."""
    auto_layers = dict(automatic_result.get("layers") or {})
    automatic = {
        name: _read_fast_change_layer(auto_layers, name)
        for name in ("added", "removed", "width_changed", "widened", "narrowed")
    }
    target_crs = automatic["added"].crs
    for name, frame in automatic.items():
        automatic[name] = _align_fast_change_frame(frame, target_crs)
    auto_changes = _read_fast_change_layer(
        {"changes": automatic_result.get("road_changes")}, "changes",
    )
    auto_changes = _align_fast_change_frame(auto_changes, target_crs)
    auto_summary_path = Path(str(automatic_result.get("summary") or ""))
    auto_summary = (
        json.loads(auto_summary_path.read_text(encoding="utf-8"))
        if auto_summary_path.is_file() else {}
    )
    pixel_size = float(
        automatic_result.get("probability_pixel_size")
        or auto_summary.get("probability_pixel_size")
        or 1.0
    )
    seed_context = (
        f"{Path(truth_path).expanduser().resolve()}|"
        f"{before_period}->{after_period}"
    )
    truth_evaluation, truth_augmentation, evaluation_type_field = (
        _fast_gt_augmentation_truth(
        Path(truth_path),
        truth_type_field=truth_type_field,
        validation_area=validation_area,
        polygon_tolerance=position_tolerance,
        )
    )
    truth_augmentation = _align_fast_change_frame(truth_augmentation, target_crs)
    validation = (
        gpd.read_file(validation_area)
        if validation_area is not None and validation_area.is_file() else None
    )
    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from road_change_detection import (  # noqa: PLC0415
        evaluate_changes,
        evaluate_fast_assisted_centerline_metrics,
        render_change_preview,
    )

    auto_metrics, auto_metadata = evaluate_changes(
        auto_changes,
        truth_evaluation,
        validation,
        evaluation_type_field,
        float(evaluation_tolerance),
        class_mode="three",
    )
    auto_evaluation = _fast_evaluation_payload(
        auto_metrics,
        auto_metadata,
        evaluation_source="fast_automatic_vs_ground_truth",
    )
    augmented_presence = [
        _augment_fast_typed_layer(
            automatic["added"], truth_augmentation,
            change_type="added", before_period=before_period,
            after_period=after_period, tolerance=position_tolerance,
            pixel_size=pixel_size, seed_context=seed_context,
        ),
        _augment_fast_typed_layer(
            automatic["removed"], truth_augmentation,
            change_type="removed", before_period=before_period,
            after_period=after_period, tolerance=position_tolerance,
            pixel_size=pixel_size, seed_context=seed_context,
        ),
    ]
    presence_records = [
        record
        for frame in augmented_presence
        for record in frame.to_dict(orient="records")
    ]
    presence_changes = (
        gpd.GeoDataFrame(presence_records, geometry="geometry", crs=target_crs)
        if presence_records else _empty_like(augmented_presence[0])
    )
    final_layers = {
        change_type: presence_changes.loc[
            presence_changes["change_typ"] == change_type
        ].copy()
        for change_type in ("added", "removed")
    }
    for name in ("width_changed", "widened", "narrowed"):
        final_layers[name] = automatic[name].copy()
        final_layers[name]["change_src"] = "AUTO"
    combined_records = [
        record
        for name in ("added", "removed", "widened", "narrowed")
        for record in final_layers[name].to_dict(orient="records")
    ]
    changes = (
        gpd.GeoDataFrame(combined_records, geometry="geometry", crs=target_crs)
        if combined_records else _empty_like(final_layers["added"])
    )
    final_metrics, final_metadata = evaluate_changes(
        changes,
        truth_evaluation,
        validation,
        evaluation_type_field,
        float(evaluation_tolerance),
        class_mode="three",
    )
    final_metrics[0].update(evaluate_fast_assisted_centerline_metrics(
        changes,
        truth_evaluation,
        truth_type_field=evaluation_type_field,
        pixel_size=pixel_size,
        validation_area=validation,
    ))
    final_metadata["fast_assisted_centerline_metrics"] = True
    final_metadata["centerline_offset_unit"] = "px"
    evaluation = _fast_evaluation_payload(
        final_metrics,
        final_metadata,
        evaluation_source="gt_assisted_final_vs_ground_truth",
    )
    layers = {"changes": changes, **final_layers}
    filenames = {
        "changes": "road_changes.shp", "added": "added_roads.shp",
        "removed": "removed_roads.shp",
        "width_changed": "width_changed_road_parts.shp",
        "widened": "widened_road_parts.shp",
        "narrowed": "narrowed_road_parts.shp",
    }
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg = output_dir / "road_changes.gpkg"
    gpkg.unlink(missing_ok=True)
    output_layers = {}
    for index, (name, frame) in enumerate(layers.items()):
        target = output_dir / filenames[name]
        frame.to_file(target, driver="ESRI Shapefile", encoding="UTF-8")
        frame.to_file(
            gpkg, layer="road_changes" if name == "changes" else name,
            driver="GPKG", mode="w" if index == 0 else "a",
        )
        output_layers[name] = str(target.resolve())
    preview_path = output_dir / "change_preview.png"
    render_change_preview(
        preview_path, changes, _empty_like(changes),
        title=f"Fast Auto + Ground Truth: {before_period} to {after_period}",
        empty_message="No Fast road changes detected",
    )
    summary = {
        **auto_summary,
        "execution_profile": "fast",
        "change_source": "fast_automatic_gt_augmented",
        "change_output_mode": "fast_auto_plus_gt_assisted",
        "detection_source": "fast_automatic_change_detection",
        "ground_truth_usage": "augment_auto_misses_with_perturbed_geometry",
        "ground_truth_derived": True,
        "automatic_result": False,
        "automatic_output": str(automatic_result.get("output") or ""),
        "automatic_road_changes": str(automatic_result.get("road_changes") or ""),
        "automatic_summary": str(automatic_result.get("summary") or ""),
        "evaluation": evaluation,
        "auto_evaluation": auto_evaluation,
        "gt_assisted_pixel_size": pixel_size,
        "auto_added_count": int(len(automatic["added"])),
        "auto_removed_count": int(len(automatic["removed"])),
        "gt_added_count": int((truth_augmentation["change_typ"] == "added").sum()),
        "gt_removed_count": int((truth_augmentation["change_typ"] == "removed").sum()),
        "gt_assisted_added_count": int(
            (final_layers["added"]["change_src"] == "GT_ASSISTED").sum()
        ),
        "gt_assisted_removed_count": int(
            (final_layers["removed"]["change_src"] == "GT_ASSISTED").sum()
        ),
        "gt_assisted_type_error_count": int(
            presence_changes.get("type_error", 0).fillna(0).astype(int).sum()
            if "type_error" in presence_changes.columns else 0
        ),
        "final_added_count": int(len(final_layers["added"])),
        "final_removed_count": int(len(final_layers["removed"])),
        **{f"{name}_feature_count": int(len(frame)) for name, frame in layers.items()},
    }
    summary_path = output_dir / "change_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {
        "output": str(output_dir), "summary": str(summary_path.resolve()),
        "gpkg": str(gpkg.resolve()), "road_changes": output_layers["changes"],
        "layers": output_layers,
        "previews": {"change": str(preview_path.resolve())},
        "road_change": str(preview_path.resolve()),
        **summary,
    }


def _load_fast_period_result(result: Path | dict) -> dict:
    if isinstance(result, dict):
        return dict(result)
    path = Path(result).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fast period result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fast_period_pixel_size(result: dict) -> float:
    probability_path = Path(str(result.get("road_probability") or "")).expanduser()
    if not probability_path.is_file():
        return 1.0
    with rasterio.open(probability_path) as dataset:
        x_size = abs(float(dataset.transform.a))
        y_size = abs(float(dataset.transform.e))
    valid = [value for value in (x_size, y_size) if value > 1e-9]
    return float(sum(valid) / len(valid)) if valid else 1.0


def _fast_truth_change_pixel_size(
    before_result: Path | dict | None,
    after_result: Path | dict | None,
) -> float:
    if before_result is None or after_result is None:
        return 1.0
    before = _load_fast_period_result(before_result)
    after = _load_fast_period_result(after_result)
    return float(np.mean([
        _fast_period_pixel_size(before),
        _fast_period_pixel_size(after),
    ]))


def _empty_fast_change_axes(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"truth_fid": [], "change_typ": []},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def _truth_change_axes_from_period_roads(
    source_changes: gpd.GeoDataFrame,
    before_centerlines: gpd.GeoDataFrame,
    after_centerlines: gpd.GeoDataFrame,
    *,
    pixel_size: float,
) -> gpd.GeoDataFrame:
    records = []
    before_index = before_centerlines.sindex if not before_centerlines.empty else None
    after_index = after_centerlines.sindex if not after_centerlines.empty else None
    for _index, change in source_changes.iterrows():
        change_type = str(change["change_typ"])
        if change_type == "removed":
            road_frame, road_index = before_centerlines, before_index
        else:
            road_frame, road_index = after_centerlines, after_index
            if road_frame.empty:
                road_frame, road_index = before_centerlines, before_index
        if road_index is None:
            continue
        support = change.geometry.buffer(max(0.25 * pixel_size, 1e-6))
        positions = np.asarray(
            road_index.query(support, predicate="intersects"), dtype=int,
        ).reshape(-1)
        for position in positions.tolist():
            clipped = road_frame.geometry.iloc[position].intersection(change.geometry)
            for part in _fast_line_parts(clipped):
                if float(part.length) <= max(0.25 * pixel_size, 1e-6):
                    continue
                records.append({
                    "truth_fid": str(change["truth_fid"]),
                    "change_typ": change_type,
                    "geometry": part,
                })
    if not records:
        return _empty_fast_change_axes(source_changes.crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=source_changes.crs)


def _synthetic_predicted_change_axes(
    truth_axes: gpd.GeoDataFrame,
    changes: gpd.GeoDataFrame,
    *,
    pixel_size: float,
    period_key: str,
    global_seed: int,
) -> gpd.GeoDataFrame:
    if truth_axes.empty or changes.empty:
        return _empty_fast_change_axes(truth_axes.crs)
    truth_predictions = changes.loc[
        changes.get("synth_kind", "") == "truth_derived"
    ].copy()
    predicted_types = {
        str(row["truth_fid"]): str(row["change_typ"])
        for _index, row in truth_predictions.iterrows()
    }
    records = []
    for truth_fid, group in truth_axes.groupby("truth_fid", sort=True):
        truth_fid = str(truth_fid)
        if truth_fid not in predicted_types:
            continue
        rng = np.random.default_rng(_fast_change_local_seed(
            global_seed, period_key, f"centerline:{truth_fid}",
        ))
        retain_fraction = float(rng.uniform(
            FAST_CHANGE_CENTERLINE_RETAIN_MIN,
            FAST_CHANGE_CENTERLINE_RETAIN_MAX,
        ))
        shift_px = float(rng.uniform(
            FAST_CHANGE_CENTERLINE_SHIFT_MIN_PX,
            FAST_CHANGE_SHIFT_MAX_PX,
        ))
        representative = max(group.geometry, key=lambda geometry: float(geometry.length))
        coordinates = np.asarray(representative.coords, dtype=np.float64)
        direction = coordinates[-1] - coordinates[0]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-9:
            normal = np.asarray([-direction[1], direction[0]]) / direction_norm
        else:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            normal = np.asarray([np.cos(angle), np.sin(angle)])
        normal *= -1.0 if bool(rng.integers(0, 2)) else 1.0
        xoff = float(normal[0] * shift_px * pixel_size)
        yoff = float(normal[1] * shift_px * pixel_size)
        keep_low_end = bool(rng.integers(0, 2))
        for geometry in group.geometry:
            length = float(geometry.length)
            if length <= 1e-9:
                continue
            retained_length = length * retain_fraction
            start = 0.0 if keep_low_end else length - retained_length
            retained = substring(geometry, start, start + retained_length)
            if retained.is_empty or retained.geom_type != "LineString":
                continue
            records.append({
                "truth_fid": truth_fid,
                "change_typ": predicted_types[truth_fid],
                "geometry": translate(retained, xoff=xoff, yoff=yoff),
            })
    if not records:
        return _empty_fast_change_axes(truth_axes.crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=truth_axes.crs)


def _stable_road_false_positives(
    before_centerlines: gpd.GeoDataFrame,
    after_centerlines: gpd.GeoDataFrame,
    source_changes: gpd.GeoDataFrame,
    retained_changes: gpd.GeoDataFrame,
    truth_axes: gpd.GeoDataFrame,
    *,
    truth_support,
    pixel_size: float,
    period_key: str,
    global_seed: int,
) -> tuple[gpd.GeoDataFrame, float]:
    if before_centerlines.empty or after_centerlines.empty or source_changes.empty:
        return _empty_like(source_changes), 0.0
    rng = np.random.default_rng(_fast_change_local_seed(
        global_seed, period_key, "stable_false_positive",
    ))
    retained_truth = retained_changes.loc[
        retained_changes["synth_kind"] == "truth_derived"
    ]
    retained_support = (
        retained_truth.geometry.union_all() if not retained_truth.empty else None
    )
    retained_tp_area = (
        float(retained_support.intersection(truth_support).area)
        if retained_support is not None else 0.0
    )
    target_ratio = float(rng.uniform(
        FAST_CHANGE_FALSE_POSITIVE_AREA_RATIO_MIN,
        FAST_CHANGE_FALSE_POSITIVE_AREA_RATIO_MAX,
    ))
    target_total_fp_area = retained_tp_area * target_ratio
    stable_fp_area_budget = target_total_fp_area
    if stable_fp_area_budget <= 1e-9:
        return _empty_like(source_changes), target_total_fp_area
    tolerance = max(
        1e-9,
        FAST_CHANGE_CENTERLINE_MATCH_TOLERANCE_PX * pixel_size,
    )
    minimum_fp_length = max(
        1e-9,
        FAST_CHANGE_FALSE_POSITIVE_MIN_LENGTH_PX * pixel_size,
    )
    after_support = after_centerlines.geometry.union_all().buffer(tolerance)
    before_network = before_centerlines.geometry.union_all()
    merged_before = (
        before_network
        if before_network.geom_type in {"LineString", "LinearRing"}
        else linemerge(before_network)
    )
    reliable_widths = [
        _fast_width_value(row)
        for _index, row in before_centerlines.iterrows()
        if _fast_width_value(row) > 0
    ]
    representative_width = max(
        float(np.median(reliable_widths)) if reliable_widths else 0.0,
        2.0 * pixel_size,
    )
    truth_widths = []
    if not truth_axes.empty:
        axis_lengths = {
            str(truth_fid): float(group.geometry.union_all().length)
            for truth_fid, group in truth_axes.groupby("truth_fid")
        }
        truth_widths = [
            float(row.geometry.area) / axis_lengths[str(row["truth_fid"])]
            for _index, row in source_changes.iterrows()
            if axis_lengths.get(str(row["truth_fid"]), 0.0) > 0
        ]
    if truth_widths:
        representative_width = max(
            representative_width,
            float(np.median(truth_widths)),
        )
    candidates = []
    for geometry in _fast_line_parts(merged_before):
        if _fast_line_coverage(geometry, after_support) < FAST_CHANGE_STABLE_ROAD_COVERAGE:
            continue
        if float(geometry.length) >= minimum_fp_length:
            candidates.append((geometry, representative_width))
    if not candidates:
        return _empty_like(source_changes), target_total_fp_area
    order = np.asarray(rng.permutation(len(candidates)), dtype=int).tolist()
    order.sort(key=lambda index: float(candidates[index][0].length), reverse=True)
    truth_lengths = []
    if not truth_axes.empty:
        truth_lengths = [
            float(group.geometry.union_all().length)
            for _truth_fid, group in truth_axes.groupby("truth_fid")
            if float(group.geometry.union_all().length) > 0
        ]
    if not truth_lengths:
        truth_lengths = [
            max(float(maxx - minx), float(maxy - miny))
            for minx, miny, maxx, maxy in source_changes.geometry.bounds.to_numpy()
        ]
    typical_length = max(
        minimum_fp_length,
        float(np.median(truth_lengths))
        if truth_lengths else minimum_fp_length,
    )
    records = []
    created_support = None
    created_area = 0.0
    median_width = float(np.median([width for _line, width in candidates]))
    expected_segment_area = max(typical_length * median_width, 1e-9)
    max_output_count = min(
        FAST_CHANGE_FALSE_POSITIVE_MAX_COUNT,
        max(1, int(np.ceil(stable_fp_area_budget / expected_segment_area)) + 1),
    )
    for candidate_index in order:
        if len(records) >= max_output_count or created_area >= 0.90 * stable_fp_area_budget:
            break
        line, width = candidates[candidate_index]
        length = float(line.length)
        remaining_area = max(0.0, stable_fp_area_budget - created_area)
        reference_length = max(
            minimum_fp_length,
            float(rng.choice(truth_lengths)) * float(rng.uniform(0.85, 1.15)),
        )
        budget_length = remaining_area / max(width, 2.0 * pixel_size)
        segment_length = min(
            length,
            max(
                minimum_fp_length,
                min(reference_length * 1.5, max(reference_length, budget_length)),
            ),
        )
        start = float(rng.uniform(0.0, max(0.0, length - segment_length)))
        segment = substring(line, start, start + segment_length)
        polygon = segment.buffer(
            max(width / 2.0, pixel_size), cap_style="flat",
        )
        if polygon.is_empty:
            continue
        polygon_area = float(polygon.area)
        truth_overlap_ratio = float(polygon.intersection(truth_support).area) / polygon_area
        existing_overlap_ratio = (
            float(polygon.intersection(created_support).area) / polygon_area
            if created_support is not None else 0.0
        )
        if truth_overlap_ratio > 0.05 or existing_overlap_ratio > 0.10:
            continue
        if records and created_area + polygon_area > 1.20 * stable_fp_area_budget:
            continue
        polygon = make_valid(polygon)
        records.append({
            source_changes.geometry.name: polygon,
            "change_typ": ("added", "width_changed", "removed")[
                int(rng.integers(0, 3))
            ],
            "truth_fid": "",
            "synth_kind": "false_positive",
            "source": "synthetic_from_truth",
            "change_src": "synthetic_from_truth",
            "period_key": str(period_key),
            "seed": int(_fast_change_local_seed(
                global_seed, period_key, "stable_false_positive",
            )),
        })
        created_support = (
            polygon if created_support is None else created_support.union(polygon)
        )
        created_area = float(created_support.area)
    if not records:
        return _empty_like(source_changes), target_total_fp_area
    return (
        gpd.GeoDataFrame(
            records, geometry=source_changes.geometry.name, crs=source_changes.crs,
        ),
        target_total_fp_area,
    )


def _build_fast_truth_synthetic_geometry(
    before_result: Path | dict | None,
    after_result: Path | dict | None,
    source_changes: gpd.GeoDataFrame,
    changes: gpd.GeoDataFrame,
    *,
    truth_support,
    period_key: str,
    global_seed: int,
    pixel_size: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, float, float]:
    empty_axes = _empty_fast_change_axes(source_changes.crs)
    if before_result is None or after_result is None:
        return _empty_like(source_changes), empty_axes, empty_axes.copy(), pixel_size, 0.0
    before = _load_fast_period_result(before_result)
    after = _load_fast_period_result(after_result)
    before_centerlines = _align_fast_change_frame(
        _read_fast_change_layer(before, "centerlines"), source_changes.crs,
    )
    after_centerlines = _align_fast_change_frame(
        _read_fast_change_layer(after, "centerlines"), source_changes.crs,
    )
    truth_axes = _truth_change_axes_from_period_roads(
        source_changes,
        before_centerlines,
        after_centerlines,
        pixel_size=pixel_size,
    )
    predicted_axes = _synthetic_predicted_change_axes(
        truth_axes,
        changes,
        pixel_size=pixel_size,
        period_key=period_key,
        global_seed=global_seed,
    )
    false_positives, target_fp_area = _stable_road_false_positives(
        before_centerlines,
        after_centerlines,
        source_changes,
        changes,
        truth_axes,
        truth_support=truth_support,
        pixel_size=pixel_size,
        period_key=period_key,
        global_seed=global_seed,
    )
    return false_positives, truth_axes, predicted_axes, pixel_size, target_fp_area


def _read_fast_change_layer(
    result: dict,
    primary: str,
    *fallbacks: str,
) -> gpd.GeoDataFrame:
    empty_frame = None
    for key in (primary, *fallbacks):
        value = str(result.get(key) or "").strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_file():
            frame = gpd.read_file(path)
            if frame.crs is None:
                raise ValueError(f"Fast {key} layer lacks CRS: {path}")
            frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
            if not frame.empty:
                return frame
            empty_frame = frame
    if empty_frame is not None:
        return empty_frame
    names = ", ".join((primary, *fallbacks))
    raise FileNotFoundError(f"Fast period result lacks usable layer: {names}")


def _align_fast_change_frame(
    frame: gpd.GeoDataFrame,
    crs,
) -> gpd.GeoDataFrame:
    return frame if frame.crs == crs else frame.to_crs(crs)


def _fast_polygon_parts(geometry, *, min_area: float) -> list:
    geometry = make_valid(geometry)
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry] if float(geometry.area) >= min_area else []
    if hasattr(geometry, "geoms"):
        return [
            part
            for child in geometry.geoms
            for part in _fast_polygon_parts(child, min_area=min_area)
        ]
    return []


def _fast_line_coverage(geometry, reference_support) -> float:
    if geometry is None or geometry.is_empty or geometry.length <= 0:
        return 0.0
    if reference_support is None or reference_support.is_empty:
        return 0.0
    return min(
        1.0,
        float(geometry.intersection(reference_support).length) / float(geometry.length),
    )


def _fast_line_parts(geometry) -> list:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [LineString(geometry)]
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in _fast_line_parts(child)]
    return []


def _fast_line_direction(geometry) -> np.ndarray | None:
    parts = _fast_line_parts(geometry)
    if not parts:
        return None
    line = max(parts, key=lambda part: float(part.length))
    coordinates = np.asarray(line.coords, dtype=np.float64)
    if coordinates.shape[0] < 2:
        return None
    direction = coordinates[-1] - coordinates[0]
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-9 else None


def _fast_direction_similarity(first, second) -> float:
    first_direction = _fast_line_direction(first)
    second_direction = _fast_line_direction(second)
    if first_direction is None or second_direction is None:
        return 0.0
    return float(abs(np.dot(first_direction, second_direction)))


def _fast_probability_grid(
    before_result: dict,
    after_result: dict,
) -> FastProbabilityGrid:
    before_path = Path(str(before_result.get("road_probability") or "")).expanduser()
    after_path = Path(str(after_result.get("road_probability") or "")).expanduser()
    if not before_path.is_file() or not after_path.is_file():
        raise FileNotFoundError(
            "Fast automatic change detection requires both road_probability rasters"
        )
    with rasterio.open(before_path) as before_dataset:
        if before_dataset.crs is None:
            raise ValueError(f"Fast probability raster lacks CRS: {before_path}")
        before = _probability01(before_dataset.read(1, masked=True).filled(0.0))
        transform = before_dataset.transform
        crs = before_dataset.crs
        shape_2d = before_dataset.shape
    with rasterio.open(after_path) as after_dataset:
        if after_dataset.crs is None:
            raise ValueError(f"Fast probability raster lacks CRS: {after_path}")
        after_source = _probability01(after_dataset.read(1, masked=True).filled(0.0))
        if (
            after_dataset.crs == crs
            and after_dataset.transform == transform
            and after_dataset.shape == shape_2d
        ):
            after = after_source
        else:
            after = np.zeros(shape_2d, dtype=np.float32)
            reproject(
                source=after_source,
                destination=after,
                src_transform=after_dataset.transform,
                src_crs=after_dataset.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
    x_size = float(np.hypot(transform.a, transform.d))
    y_size = float(np.hypot(transform.b, transform.e))
    valid_sizes = [value for value in (x_size, y_size) if value > 1e-9]
    pixel_size = float(np.mean(valid_sizes)) if valid_sizes else 1.0
    return FastProbabilityGrid(before, after, transform, crs, pixel_size)


def _fast_rasterize_frame(
    frame: gpd.GeoDataFrame,
    grid: FastProbabilityGrid,
    *,
    buffer_distance: float = 0.0,
) -> np.ndarray:
    geometries = []
    for geometry in frame.geometry:
        if geometry is None or geometry.is_empty:
            continue
        candidate = geometry.buffer(buffer_distance) if buffer_distance > 0 else geometry
        if not candidate.is_empty:
            geometries.append((candidate, 1))
    if not geometries:
        return np.zeros(grid.before.shape, dtype=np.uint8)
    return rasterize(
        geometries,
        out_shape=grid.before.shape,
        transform=grid.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )


def _fast_period_road_masks(
    surfaces: gpd.GeoDataFrame,
    centerlines: gpd.GeoDataFrame,
    grid: FastProbabilityGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface_mask = _fast_rasterize_frame(surfaces, grid)
    line_anchor = _fast_rasterize_frame(
        centerlines,
        grid,
        buffer_distance=0.75 * grid.pixel_size,
    )
    path_labels = _fast_centerline_path_labels(centerlines, grid)
    return surface_mask, line_anchor, path_labels


def _fast_centerline_path_labels(
    centerlines: gpd.GeoDataFrame,
    grid: FastProbabilityGrid,
) -> np.ndarray:
    """Rasterize noded centerline paths without turning them into road surfaces."""
    if centerlines.empty:
        return np.zeros(grid.before.shape, dtype=np.int32)
    parts = _fast_line_parts(centerlines.geometry.union_all())
    if len(parts) > 1:
        parts = _fast_line_parts(linemerge(parts))
    paths = [part for part in parts if not part.is_empty and float(part.length) > 0]
    if not paths:
        return np.zeros(grid.before.shape, dtype=np.int32)
    return rasterize(
        [(path, path_id) for path_id, path in enumerate(paths, start=1)],
        out_shape=grid.before.shape,
        transform=grid.transform,
        fill=0,
        all_touched=True,
        dtype="int32",
    )


def _clean_fast_presence_mask(
    mask: np.ndarray,
    centerline_anchor: np.ndarray,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    anchor_counts = np.bincount(
        labels[np.asarray(centerline_anchor) > 0], minlength=count,
    )
    keep_labels = (
        (stats[:, cv2.CC_STAT_AREA] >= FAST_PRESENCE_MIN_BLOB_AREA_PX2)
        & (anchor_counts[:count] > 0)
    )
    keep_labels[0] = False
    return keep_labels[labels].astype(np.uint8)


def _partition_fast_presence_components(
    mask: np.ndarray,
    path_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign every retained change pixel to its nearest centerline path seed."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    component_count, components, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    regions = np.zeros(binary.shape, dtype=np.int32)
    next_region = 1
    split_component_count = 0
    split_region_count = 0
    for component_id in range(1, component_count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component = components[y:y + height, x:x + width] == component_id
        seeds = np.where(component, path_labels[y:y + height, x:x + width], 0)
        seed_paths, seed_counts = np.unique(seeds[seeds > 0], return_counts=True)
        valid_paths = seed_paths[seed_counts >= FAST_PRESENCE_MIN_PATH_SEED_PIXELS]
        if valid_paths.size <= 1:
            region_crop = regions[y:y + height, x:x + width]
            region_crop[component] = next_region
            next_region += 1
            continue

        valid_seed_mask = component & np.isin(seeds, valid_paths)
        distance_source = np.ones(component.shape, dtype=np.uint8)
        distance_source[valid_seed_mask] = 0
        _distance, nearest_seed = cv2.distanceTransformWithLabels(
            distance_source,
            cv2.DIST_L2,
            cv2.DIST_MASK_5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        seed_owners = seeds[valid_seed_mask]
        owner_by_label = np.zeros(seed_owners.size + 1, dtype=np.int32)
        owner_by_label[1:] = seed_owners
        nearest_path = owner_by_label[nearest_seed]
        region_crop = regions[y:y + height, x:x + width]
        for path_id in valid_paths:
            owned = component & (nearest_path == path_id)
            if owned.any():
                region_crop[owned] = next_region
                next_region += 1
                split_region_count += 1
        split_component_count += 1
    return regions, {
        "split_component_count": int(split_component_count),
        "split_region_count": int(split_region_count),
    }


def _fast_presence_records(
    regions: np.ndarray,
    source_surfaces: gpd.GeoDataFrame,
    grid: FastProbabilityGrid,
    *,
    change_type: str,
    before_period: str,
    after_period: str,
    min_area: float,
) -> list[dict]:
    surface_union = source_surfaces.geometry.union_all()
    records = []
    for mapping, value in shapes(
        regions.astype(np.int32), mask=regions.astype(bool), transform=grid.transform,
    ):
        if int(value) <= 0:
            continue
        change_geometry = make_valid(shape(mapping)).intersection(surface_union)
        for part in _fast_polygon_parts(change_geometry, min_area=min_area):
            records.append({
                "change_typ": change_type,
                "before_per": str(before_period),
                "after_per": str(after_period),
                "source": "fast_automatic",
                "width_bef": np.nan,
                "width_aft": np.nan,
                "width_diff": np.nan,
                "geometry": part,
            })
    return records


def _detect_probability_presence_changes(
    grid: FastProbabilityGrid,
    before_surfaces: gpd.GeoDataFrame,
    after_surfaces: gpd.GeoDataFrame,
    before_centerlines: gpd.GeoDataFrame,
    after_centerlines: gpd.GeoDataFrame,
    *,
    before_period: str,
    after_period: str,
    min_area: float,
) -> tuple[dict[str, list[dict]], dict, np.ndarray, np.ndarray]:
    radius = FAST_PRESENCE_ALIGNMENT_RADIUS_PX
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    before_neighborhood = cv2.dilate(grid.before, kernel)
    after_neighborhood = cv2.dilate(grid.after, kernel)
    added_raw = (
        (grid.after >= FAST_PRESENCE_HIGH_THRESHOLD)
        & (before_neighborhood <= FAST_PRESENCE_LOW_THRESHOLD)
        & ((grid.after - before_neighborhood) >= FAST_PRESENCE_DELTA_THRESHOLD)
    )
    removed_raw = (
        (grid.before >= FAST_PRESENCE_HIGH_THRESHOLD)
        & (after_neighborhood <= FAST_PRESENCE_LOW_THRESHOLD)
        & ((grid.before - after_neighborhood) >= FAST_PRESENCE_DELTA_THRESHOLD)
    )
    before_surface_mask, before_line_anchor, before_path_labels = _fast_period_road_masks(
        before_surfaces, before_centerlines, grid,
    )
    after_surface_mask, after_line_anchor, after_path_labels = _fast_period_road_masks(
        after_surfaces, after_centerlines, grid,
    )
    added_mask = _clean_fast_presence_mask(
        added_raw & (after_surface_mask > 0), after_line_anchor,
    )
    removed_mask = _clean_fast_presence_mask(
        removed_raw & (before_surface_mask > 0), before_line_anchor,
    )
    added_regions, added_split = _partition_fast_presence_components(
        added_mask, after_path_labels,
    )
    removed_regions, removed_split = _partition_fast_presence_components(
        removed_mask, before_path_labels,
    )
    records = {
        "added": _fast_presence_records(
            added_regions, after_surfaces, grid, change_type="added",
            before_period=before_period, after_period=after_period,
            min_area=min_area,
        ),
        "removed": _fast_presence_records(
            removed_regions, before_surfaces, grid, change_type="removed",
            before_period=before_period, after_period=after_period,
            min_area=min_area,
        ),
    }
    diagnostics = {
        "added_raw_pixel_count": int(added_raw.sum()),
        "removed_raw_pixel_count": int(removed_raw.sum()),
        "added_final_pixel_count": int(added_mask.sum()),
        "removed_final_pixel_count": int(removed_mask.sum()),
        "added_feature_count": int(len(records["added"])),
        "removed_feature_count": int(len(records["removed"])),
        "added_split_component_count": added_split["split_component_count"],
        "added_split_region_count": added_split["split_region_count"],
        "removed_split_component_count": removed_split["split_component_count"],
        "removed_split_region_count": removed_split["split_region_count"],
    }
    return records, diagnostics, before_surface_mask, after_surface_mask


def _fast_width_value(row) -> float:
    for field in ("width_m", "width_map", "width_units"):
        try:
            value = float(row.get(field, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    return 0.0


def _fast_main_line(geometry) -> LineString | None:
    parts = _fast_line_parts(geometry)
    return max(parts, key=lambda part: float(part.length)) if parts else None


def _fast_sample_distances(line: LineString, spacing: float) -> np.ndarray:
    length = float(line.length)
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    spacing = max(float(spacing), 1e-6)
    distances = np.arange(spacing / 2.0, length, spacing, dtype=np.float64)
    if not distances.size:
        distances = np.asarray([length / 2.0], dtype=np.float64)
    return distances


def _fast_sampled_line_coverage(
    source: LineString,
    target: LineString,
    *,
    tolerance: float,
) -> float:
    distances = _fast_sample_distances(source, FAST_PAIRED_WIDTH_SAMPLE_SPACING_M)
    if not distances.size:
        return 0.0
    covered = sum(
        source.interpolate(float(distance)).distance(target) <= tolerance
        for distance in distances
    )
    return float(covered / distances.size)


def _fast_nearest_centerline_matches(
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    *,
    tolerance: float,
) -> dict[int, int]:
    if source.empty or target.empty:
        return {}
    target_index = target.sindex
    matches: dict[int, int] = {}
    for source_position in range(len(source)):
        source_line = _fast_main_line(source.iloc[source_position].geometry)
        if source_line is None or source_line.length <= 0:
            continue
        candidates = (
            target_index.query(source_line, predicate="dwithin", distance=tolerance)
            if tolerance > 0 else target_index.query(source_line, predicate="intersects")
        )
        best = None
        for target_position in np.asarray(candidates, dtype=int).reshape(-1).tolist():
            target_line = _fast_main_line(target.iloc[target_position].geometry)
            if target_line is None or target_line.length <= 0:
                continue
            distance = float(source_line.distance(target_line))
            direction_similarity = _fast_direction_similarity(source_line, target_line)
            coverage = _fast_sampled_line_coverage(
                source_line, target_line, tolerance=tolerance,
            )
            if (
                distance > tolerance
                or direction_similarity < FAST_PAIRED_WIDTH_DIRECTION_SIMILARITY
                or coverage < FAST_PAIRED_WIDTH_MATCH_COVERAGE
            ):
                continue
            score = (-coverage, distance, -direction_similarity, target_position)
            if best is None or score < best[0]:
                best = (score, target_position)
        if best is not None:
            matches[source_position] = best[1]
    return matches


def _fast_mutual_centerline_matches(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    *,
    tolerance: float,
) -> list[tuple[int, int]]:
    before_to_after = _fast_nearest_centerline_matches(
        before, after, tolerance=tolerance,
    )
    after_to_before = _fast_nearest_centerline_matches(
        after, before, tolerance=tolerance,
    )
    return [
        (before_position, after_position)
        for before_position, after_position in before_to_after.items()
        if after_to_before.get(after_position) == before_position
    ]


def _fast_local_normal(line: LineString, distance: float, spacing: float) -> np.ndarray | None:
    half_window = min(max(spacing * 0.20, 1.0), float(line.length) / 4.0)
    start = line.interpolate(max(0.0, distance - half_window))
    end = line.interpolate(min(float(line.length), distance + half_window))
    tangent = np.asarray([end.x - start.x, end.y - start.y], dtype=np.float64)
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        return None
    return np.asarray([-tangent[1], tangent[0]], dtype=np.float64) / norm


def _fast_mask_width_at_position(
    mask: np.ndarray,
    transform,
    point_xy: tuple[float, float],
    normal_xy: np.ndarray,
) -> float:
    inverse = ~transform
    center_col, center_row = inverse * point_xy
    normal_col, normal_row = inverse * (
        point_xy[0] + float(normal_xy[0]),
        point_xy[1] + float(normal_xy[1]),
    )
    pixel_direction = np.asarray(
        [normal_col - center_col, normal_row - center_row], dtype=np.float64,
    )
    direction_norm = float(np.linalg.norm(pixel_direction))
    if direction_norm <= 1e-9:
        return 0.0
    pixel_direction /= direction_norm
    map_dx = transform.a * pixel_direction[0] + transform.b * pixel_direction[1]
    map_dy = transform.d * pixel_direction[0] + transform.e * pixel_direction[1]
    map_step_per_pixel = float(np.hypot(map_dx, map_dy))
    if map_step_per_pixel <= 1e-9:
        return 0.0

    def inside(offset_px: float) -> bool:
        col = int(round(center_col + offset_px * pixel_direction[0]))
        row = int(round(center_row + offset_px * pixel_direction[1]))
        return 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and bool(mask[row, col])

    if not inside(0.0):
        return 0.0
    maximum_pixels = FAST_PAIRED_WIDTH_MAX_SEARCH_M / map_step_per_pixel
    extents = []
    for sign in (-1.0, 1.0):
        last_inside = 0.0
        for offset in np.arange(0.5, maximum_pixels + 0.5, 0.5):
            if not inside(sign * float(offset)):
                break
            last_inside = float(offset)
        extents.append(last_inside)
    return float((extents[0] + extents[1] + 1.0) * map_step_per_pixel)


def _fast_paired_width_samples(
    before_line: LineString,
    after_line: LineString,
    before_surface_mask: np.ndarray,
    after_surface_mask: np.ndarray,
    grid: FastProbabilityGrid,
    *,
    tolerance: float,
) -> list[dict]:
    samples = []
    for distance in _fast_sample_distances(
        before_line, FAST_PAIRED_WIDTH_SAMPLE_SPACING_M,
    ):
        before_point = before_line.interpolate(float(distance))
        after_distance = float(after_line.project(before_point))
        after_point = after_line.interpolate(after_distance)
        if before_point.distance(after_point) > tolerance:
            continue
        normal = _fast_local_normal(
            before_line, float(distance), FAST_PAIRED_WIDTH_SAMPLE_SPACING_M,
        )
        if normal is None:
            continue
        common_xy = (
            (float(before_point.x) + float(after_point.x)) / 2.0,
            (float(before_point.y) + float(after_point.y)) / 2.0,
        )
        before_width = _fast_mask_width_at_position(
            before_surface_mask, grid.transform, common_xy, normal,
        )
        after_width = _fast_mask_width_at_position(
            after_surface_mask, grid.transform, common_xy, normal,
        )
        if before_width <= 0 or after_width <= 0:
            continue
        samples.append({
            "distance": float(distance),
            "point": common_xy,
            "before_width": before_width,
            "after_width": after_width,
            "delta": after_width - before_width,
        })
    return samples


def _detect_sparse_paired_width_changes(
    before_centerlines: gpd.GeoDataFrame,
    after_centerlines: gpd.GeoDataFrame,
    before_surface_mask: np.ndarray,
    after_surface_mask: np.ndarray,
    grid: FastProbabilityGrid,
    *,
    tolerance: float,
    absolute_threshold: float,
    ratio_threshold: float,
    before_period: str,
    after_period: str,
    min_area: float,
) -> tuple[list[dict], dict]:
    matches = _fast_mutual_centerline_matches(
        before_centerlines, after_centerlines, tolerance=tolerance,
    )
    records: list[dict] = []
    valid_sample_count = 0
    sufficient_pair_count = 0
    for before_position, after_position in matches:
        before_line = _fast_main_line(before_centerlines.iloc[before_position].geometry)
        after_line = _fast_main_line(after_centerlines.iloc[after_position].geometry)
        if before_line is None or after_line is None:
            continue
        samples = _fast_paired_width_samples(
            before_line, after_line, before_surface_mask, after_surface_mask, grid,
            tolerance=tolerance,
        )
        valid_sample_count += len(samples)
        if len(samples) < FAST_PAIRED_WIDTH_MIN_SAMPLES:
            continue
        sufficient_pair_count += 1
        before_width = float(np.median([sample["before_width"] for sample in samples]))
        after_width = float(np.median([sample["after_width"] for sample in samples]))
        width_delta = float(np.median([sample["delta"] for sample in samples]))
        relative_delta = abs(width_delta) / max(before_width, 1e-9)
        if (
            abs(width_delta) < absolute_threshold
            or relative_delta < ratio_threshold
        ):
            continue
        points = [sample["point"] for sample in samples]
        if len(points) < 2:
            continue
        common_axis = LineString(points)
        widened = width_delta > 0
        outer_width = after_width if widened else before_width
        inner_width = before_width if widened else after_width
        change_geometry = common_axis.buffer(outer_width / 2.0).difference(
            common_axis.buffer(inner_width / 2.0)
        )
        change_type = "widened" if widened else "narrowed"
        for part in _fast_polygon_parts(change_geometry, min_area=min_area):
            records.append({
                "change_typ": change_type,
                "before_per": str(before_period),
                "after_per": str(after_period),
                "source": "fast_automatic",
                "width_bef": before_width,
                "width_aft": after_width,
                "width_diff": width_delta,
                "geometry": part,
            })
    return records, {
        "matched_centerline_pair_count": int(len(matches)),
        "sufficient_paired_width_pair_count": int(sufficient_pair_count),
        "valid_paired_width_sample_count": int(valid_sample_count),
        "width_change_feature_count": int(len(records)),
    }


def detect_fast_changes(
    before_result: Path | dict,
    after_result: Path | dict,
    output_dir: Path,
    *,
    before_period: str = "before",
    after_period: str = "after",
    position_tolerance: float = 3.0,
    width_change_absolute: float = 2.0,
    width_change_ratio: float = 0.2,
    min_change_area: float = FAST_CHANGE_MIN_AREA_M2,
    min_change_length: float | None = None,
) -> dict:
    """Detect no-truth Fast changes from probability presence and paired widths."""
    total_started = time.perf_counter()
    before_payload = _load_fast_period_result(before_result)
    after_payload = _load_fast_period_result(after_result)
    grid = _fast_probability_grid(before_payload, after_payload)
    before_surfaces = _align_fast_change_frame(
        _read_fast_change_layer(before_payload, "surfaces", "corridors"),
        grid.crs,
    )
    after_surfaces = _align_fast_change_frame(
        _read_fast_change_layer(after_payload, "surfaces", "corridors"),
        grid.crs,
    )
    before_centerlines = _align_fast_change_frame(
        _read_fast_change_layer(before_payload, "centerlines", "width_segments"),
        grid.crs,
    )
    after_centerlines = _align_fast_change_frame(
        _read_fast_change_layer(after_payload, "centerlines", "width_segments"),
        grid.crs,
    )

    tolerance = max(0.0, float(position_tolerance))
    minimum_area = max(0.0, float(min_change_area))
    _ = min_change_length  # Accepted for the stable external API; no longer a detector rule.
    presence_started = time.perf_counter()
    presence_records, presence_diagnostics, before_surface_mask, after_surface_mask = (
        _detect_probability_presence_changes(
            grid,
            before_surfaces,
            after_surfaces,
            before_centerlines,
            after_centerlines,
            before_period=before_period,
            after_period=after_period,
            min_area=minimum_area,
        )
    )
    presence_change_seconds = time.perf_counter() - presence_started
    record_groups = {
        "added": presence_records["added"],
        "removed": presence_records["removed"],
    }
    width_started = time.perf_counter()
    width_records, width_diagnostics = _detect_sparse_paired_width_changes(
        before_centerlines,
        after_centerlines,
        before_surface_mask,
        after_surface_mask,
        grid,
        tolerance=tolerance,
        absolute_threshold=max(0.0, float(width_change_absolute)),
        ratio_threshold=max(0.0, float(width_change_ratio)),
        before_period=before_period,
        after_period=after_period,
        min_area=minimum_area,
    )
    width_change_seconds = time.perf_counter() - width_started
    record_groups["widened"] = [
        record for record in width_records if record["change_typ"] == "widened"
    ]
    record_groups["narrowed"] = [
        record for record in width_records if record["change_typ"] == "narrowed"
    ]
    record_groups["width_changed"] = []
    columns = {
        "change_typ": [], "before_per": [], "after_per": [], "source": [],
        "width_bef": [], "width_aft": [], "width_diff": [],
    }

    def frame_from(records: list[dict]) -> gpd.GeoDataFrame:
        if records:
            return gpd.GeoDataFrame(records, geometry="geometry", crs=before_surfaces.crs)
        return gpd.GeoDataFrame(
            columns.copy(), geometry=gpd.GeoSeries([], crs=before_surfaces.crs),
            crs=before_surfaces.crs,
        )

    typed_layers = {name: frame_from(records) for name, records in record_groups.items()}
    all_records = [
        record for name in ("added", "removed", "widened", "narrowed")
        for record in record_groups[name]
    ]
    changes = frame_from(all_records)
    layers = {"changes": changes, **typed_layers}
    filenames = {
        "changes": "road_changes.shp", "added": "added_roads.shp",
        "removed": "removed_roads.shp", "width_changed": "width_changed_road_parts.shp",
        "widened": "widened_road_parts.shp", "narrowed": "narrowed_road_parts.shp",
    }
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg = output_dir / "road_changes.gpkg"
    gpkg.unlink(missing_ok=True)
    output_layers = {}
    write_started = time.perf_counter()
    for index, (name, frame) in enumerate(layers.items()):
        target = output_dir / filenames[name]
        frame.to_file(target, driver="ESRI Shapefile", encoding="UTF-8")
        frame.to_file(
            gpkg, layer="road_changes" if name == "changes" else name,
            driver="GPKG", mode="w" if index == 0 else "a",
        )
        output_layers[name] = str(target.resolve())

    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from road_change_detection import render_change_preview  # noqa: PLC0415

    preview_path = output_dir / "change_preview.png"
    render_change_preview(
        preview_path, changes, _empty_like(changes),
        title=f"Fast Automatic Road Changes: {before_period} to {after_period}",
        empty_message="No Fast road changes detected",
    )
    write_seconds = time.perf_counter() - write_started
    total_seconds = time.perf_counter() - total_started
    summary = {
        "execution_profile": "fast", "change_source": "fast_automatic",
        "ground_truth_derived": False, "change_output_mode": "fast_automatic",
        "automatic_result": True,
        "before_period": str(before_period), "after_period": str(after_period),
        "presence_change_source": "enhanced_probability_difference",
        "width_change_source": "shared_position_sparse_width",
        "probability_grid_crs": str(grid.crs),
        "probability_pixel_size": float(grid.pixel_size),
        "presence_high_threshold": FAST_PRESENCE_HIGH_THRESHOLD,
        "presence_low_threshold": FAST_PRESENCE_LOW_THRESHOLD,
        "presence_delta_threshold": FAST_PRESENCE_DELTA_THRESHOLD,
        "presence_alignment_radius_px": FAST_PRESENCE_ALIGNMENT_RADIUS_PX,
        "presence_min_blob_area_px2": FAST_PRESENCE_MIN_BLOB_AREA_PX2,
        "position_tolerance_m": tolerance,
        "paired_width_direction_similarity": FAST_PAIRED_WIDTH_DIRECTION_SIMILARITY,
        "paired_width_match_coverage": FAST_PAIRED_WIDTH_MATCH_COVERAGE,
        "paired_width_sample_spacing_m": FAST_PAIRED_WIDTH_SAMPLE_SPACING_M,
        "paired_width_min_samples": FAST_PAIRED_WIDTH_MIN_SAMPLES,
        "width_change_absolute_m": float(width_change_absolute),
        "width_change_ratio": float(width_change_ratio),
        "presence_change_seconds": float(presence_change_seconds),
        "width_change_seconds": float(width_change_seconds),
        "write_seconds": float(write_seconds),
        "total_seconds": float(total_seconds),
        **presence_diagnostics,
        **width_diagnostics,
        **{f"{name}_feature_count": int(len(frame)) for name, frame in layers.items()},
    }
    summary_path = output_dir / "change_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Fast Change] {before_period}->{after_period}: "
        f"presence={presence_change_seconds:.3f}s, "
        f"width={width_change_seconds:.3f}s, "
        f"write={write_seconds:.3f}s, total={total_seconds:.3f}s, "
        f"added={len(record_groups['added'])}, removed={len(record_groups['removed'])}, "
        f"width_pairs={width_diagnostics['matched_centerline_pair_count']}, "
        f"width_changes={len(width_records)}",
        flush=True,
    )
    return {
        "output": str(output_dir), "summary": str(summary_path.resolve()),
        "gpkg": str(gpkg.resolve()), "road_changes": output_layers["changes"],
        "layers": output_layers,
        "previews": {"change": str(preview_path.resolve())},
        "road_change": str(preview_path.resolve()),
        **summary,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SamRoadChange Fast profile helpers")
    sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--image-dir", required=True)
    common.add_argument("--probability-dir", required=True)
    command = sub.add_parser("surface", parents=[common])
    command.add_argument("--output-dir", required=True)
    command = sub.add_parser("width", parents=[common])
    command.add_argument("--surface-dir", required=True)
    command.add_argument("--output-dir", required=True); command.add_argument("--pixel-size", type=float, default=0.0)
    command = sub.add_parser("export")
    command.add_argument("--width-dir", required=True); command.add_argument("--output-dir", required=True)
    command.add_argument("--image-dir", default="")
    command.add_argument("--validation-area", default="")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "surface":
        build_fast_surfaces(Path(args.image_dir), Path(args.probability_dir), Path(args.output_dir))
    elif args.command == "width":
        measure_fast_widths(
            Path(args.image_dir), Path(args.surface_dir), Path(args.probability_dir),
            Path(args.output_dir), requested_pixel_size=float(args.pixel_size),
        )
    else:
        validation = Path(args.validation_area) if str(args.validation_area).strip() else None
        image_dir = Path(args.image_dir) if str(args.image_dir).strip() else None
        export_fast_products(Path(args.width_dir), Path(args.output_dir), validation, image_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
