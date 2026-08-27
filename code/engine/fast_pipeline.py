from __future__ import annotations

"""Lightweight, evidence-based products for the optional Fast execution profile."""

import argparse
import hashlib
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Geod
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject
from shapely import make_valid
from shapely.affinity import translate
from shapely.geometry import LineString, box, shape
from shapely.ops import linemerge, substring, unary_union


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
FAST_GT_ASSISTED_DROPOUT_RETAIN_MIN = 0.85
FAST_GT_ASSISTED_DROPOUT_RETAIN_MAX = 0.95
FAST_GT_ASSISTED_MIN_RESIDUAL_AREA_PX2 = 8.0
FAST_GT_ASSISTED_TYPE_ERROR_PROB = 0.08
FAST_GT_ASSISTED_WINDOW_PADDING_PX = 8
FAST_GT_ASSISTED_LOCAL_DEFORMATION_RADIUS_PX = 6
FAST_GT_ASSISTED_OVERDETECTION_MAX_RATIO = 0.06
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
FAST_PRESENCE_ABNORMAL_COMPONENT_AREA_PX2 = 48
FAST_PAIRED_WIDTH_DIRECTION_SIMILARITY = 0.90
FAST_PAIRED_WIDTH_MATCH_COVERAGE = 0.70
FAST_PAIRED_WIDTH_SAMPLE_SPACING_M = 15.0
FAST_PAIRED_WIDTH_MIN_SAMPLES = 3
FAST_PAIRED_WIDTH_MAX_SEARCH_M = 80.0
FAST_CHANGE_TILE_HALO_MIN_PX = 8
FAST_PUBLIC_CHANGE_FIELDS = (
    "change_typ", "before_per", "after_per",
    "width_bef", "width_aft", "width_diff",
)
FAST_PRIVATE_CHANGE_FIELDS = {
    "change_src", "source", "truth_fid", "synth_kind", "seed", "type_error",
    "geometry_structure_type", "skeleton_branch_count", "removed_branch_count",
    "transverse_cut_count", "boundary_erosion_pixel_count", "retained_ratio",
    "boundary_expansion_pixel_count", "endpoint_extension_count",
    "junction_deformation_count", "overdetect_fragment_count",
    "overdetect_pixel_count", "final_area_ratio",
}


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


@dataclass(frozen=True)
class FastTileArtifacts:
    """Existing Fast extraction intermediates for one normalized source tile."""

    stem: str
    image: Path
    probability: Path
    surface_mask: Path
    centerline_mask: Path
    topology: Path
    transform: object
    crs: object
    shape: tuple[int, int]
    bounds: tuple[float, float, float, float]
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


def _fast_public_change_frame(changes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the single user-facing change layer without provenance fields."""
    public = changes.copy()
    for field in FAST_PUBLIC_CHANGE_FIELDS:
        if field not in public.columns:
            public[field] = "" if field in {"change_typ", "before_per", "after_per"} else np.nan
    fields = [
        field for field in FAST_PUBLIC_CHANGE_FIELDS
        if field in public.columns and field not in FAST_PRIVATE_CHANGE_FIELDS
    ]
    return gpd.GeoDataFrame(
        public[fields + [public.geometry.name]],
        geometry=public.geometry.name,
        crs=public.crs,
    )


def _remove_fast_legacy_change_outputs(output_dir: Path) -> None:
    """Remove stale classified Fast datasets before publishing the single SHP."""
    for stem in (
        "added_roads", "removed_roads", "width_changed_road_parts",
        "widened_road_parts", "narrowed_road_parts",
    ):
        for member in output_dir.glob(f"{stem}.*"):
            if member.is_file():
                member.unlink()
    (output_dir / "road_changes.gpkg").unlink(missing_ok=True)


def _write_fast_public_changes(
    changes: gpd.GeoDataFrame,
    output_dir: Path,
) -> tuple[gpd.GeoDataFrame, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_fast_legacy_change_outputs(output_dir)
    public = _fast_public_change_frame(changes)
    target = output_dir / "road_changes.shp"
    public.to_file(target, driver="ESRI Shapefile", encoding="UTF-8")
    return public, target


def _fast_change_preview_title(before_period: str, after_period: str) -> str:
    """Return one neutral title regardless of how the Fast result was produced."""
    return f"Fast Road Change Results: {before_period} to {after_period}"


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


def _fill_fast_gt_new_internal_holes(
    source: np.ndarray,
    perturbed: np.ndarray,
) -> np.ndarray:
    """Fill dropout holes that do not connect to pre-existing GT background."""
    source_background = np.asarray(source) == 0
    background = (np.asarray(perturbed) == 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(background, connectivity=4)
    original_counts = np.bincount(labels[source_background], minlength=count)
    border_labels = np.unique(np.concatenate((
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    )))
    original_counts[border_labels] += 1
    result = (np.asarray(perturbed) > 0).astype(np.uint8)
    for label in np.flatnonzero(original_counts == 0):
        if label > 0:
            result[labels == int(label)] = 1
    return result


def _fast_gt_structure_branches(skeleton: np.ndarray) -> list[dict]:
    """Trace GT structure and split closed loops into degradable arc branches."""
    branches = []
    for path in _trace_skeleton_paths(skeleton):
        pixels = np.asarray(path.pixels, dtype=np.float32)
        closed = bool(
            path.start_degree == 2
            and path.end_degree == 2
            and np.linalg.norm(pixels[0] - pixels[-1]) <= 1.5
        )
        if not closed or pixels.shape[0] < 12:
            branches.append({
                "pixels": pixels,
                "length_px": float(path.length_px),
                "start_degree": int(path.start_degree),
                "end_degree": int(path.end_degree),
                "loop_segment": False,
            })
            continue
        split_count = int(np.clip(np.ceil(path.length_px / 30.0), 3, 6))
        split_indices = np.linspace(
            0, pixels.shape[0] - 1, split_count + 1,
        ).round().astype(int)
        for start, stop in zip(split_indices[:-1], split_indices[1:]):
            segment = pixels[start:stop + 1]
            if segment.shape[0] < 2:
                continue
            length = float(np.linalg.norm(
                np.diff(segment, axis=0), axis=1,
            ).sum())
            branches.append({
                "pixels": segment,
                "length_px": length,
                "start_degree": 2,
                "end_degree": 2,
                "loop_segment": True,
            })
    return branches


def _fast_gt_structure_type(
    source: np.ndarray,
    skeleton: np.ndarray,
    branches: list[dict],
    distance: np.ndarray,
) -> str:
    rows, columns = np.nonzero(source)
    height = int(rows.max() - rows.min() + 1)
    width = int(columns.max() - columns.min() + 1)
    aspect = max(height, width) / max(1.0, min(height, width))
    adjacency = _pixel_adjacency(skeleton)
    junction_count = sum(len(neighbors) >= 3 for neighbors in adjacency.values())
    skeleton_distance = distance[skeleton > 0]
    median_radius = (
        float(np.median(skeleton_distance)) if skeleton_distance.size else 0.0
    )
    if aspect >= 2.5:
        return "corridor"
    if median_radius >= max(6.0, 0.18 * min(height, width)):
        return "areal"
    if junction_count > 0 or len(branches) >= 4:
        return "network"
    if median_radius <= 4.0:
        return "corridor"
    return "areal"


def _fast_gt_transverse_cut(
    shape_2d: tuple[int, int],
    branch: dict,
    distance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a narrow cut across the complete local GT width at one branch point."""
    points = np.asarray(branch["pixels"], dtype=np.float32)
    if points.shape[0] < 5:
        return np.zeros(shape_2d, dtype=np.uint8)
    center_index = int(rng.integers(
        max(2, points.shape[0] // 4),
        max(3, points.shape[0] - max(2, points.shape[0] // 4)),
    ))
    low = max(0, center_index - 2)
    high = min(points.shape[0] - 1, center_index + 2)
    tangent = points[high] - points[low]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        return np.zeros(shape_2d, dtype=np.uint8)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32) / norm
    center = points[center_index]
    row = int(np.clip(round(float(center[0])), 0, shape_2d[0] - 1))
    column = int(np.clip(round(float(center[1])), 0, shape_2d[1] - 1))
    half_length = max(5.0, float(distance[row, column]) * 2.2 + 3.0)
    start = center - normal * half_length
    stop = center + normal * half_length
    cut = np.zeros(shape_2d, dtype=np.uint8)
    cv2.line(
        cut,
        (int(round(start[1])), int(round(start[0]))),
        (int(round(stop[1])), int(round(stop[0]))),
        1,
        int(rng.integers(1, 3)),
    )
    return cut


def _fast_gt_branch_interval_removal(
    shape_2d: tuple[int, int],
    branch: dict,
    distance: np.ndarray,
    *,
    from_start: bool,
    fraction: float,
) -> np.ndarray:
    """Remove one continuous endpoint interval while retaining original GT widths."""
    points = np.asarray(branch["pixels"], dtype=np.float32)
    count = max(2, int(round(points.shape[0] * float(fraction))))
    if float(fraction) >= 0.99:
        endpoint_index = 0 if from_start else -1
        endpoint = np.rint(points[endpoint_index]).astype(np.int32)
        endpoint[0] = np.clip(endpoint[0], 0, shape_2d[0] - 1)
        endpoint[1] = np.clip(endpoint[1], 0, shape_2d[1] - 1)
        local_radius = float(distance[endpoint[0], endpoint[1]])
        junction_guard = max(2, int(np.ceil(2.0 * local_radius + 3.0)))
        count = max(2, points.shape[0] - junction_guard)
    selected = points[:count] if from_start else points[-count:]
    indices = np.rint(selected).astype(np.int32)
    indices[:, 0] = np.clip(indices[:, 0], 0, shape_2d[0] - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, shape_2d[1] - 1)
    radius = float(np.median(distance[indices[:, 0], indices[:, 1]]))
    removal = np.zeros(shape_2d, dtype=np.uint8)
    coordinates = np.column_stack((selected[:, 1], selected[:, 0])).round().astype(np.int32)
    cv2.polylines(
        removal,
        [coordinates.reshape(-1, 1, 2)],
        False,
        1,
        max(3, int(np.ceil(2.0 * radius + 3.0))),
    )
    return removal


def _fast_gt_areal_through_cut(
    source: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a slightly bent background-connected channel through a thick area."""
    rows, columns = np.nonzero(source)
    row0, row1 = int(rows.min()), int(rows.max())
    column0, column1 = int(columns.min()), int(columns.max())
    cut = np.zeros_like(source, dtype=np.uint8)
    if (column1 - column0) >= (row1 - row0):
        center_column = int(rng.integers(
            column0 + max(1, (column1 - column0) // 4),
            column1 - max(1, (column1 - column0) // 4) + 1,
        ))
        bend = int(rng.integers(-3, 4))
        points = np.asarray([
            (center_column, row0 - 3),
            (center_column + bend, (row0 + row1) // 2),
            (center_column - bend, row1 + 3),
        ], dtype=np.int32)
    else:
        center_row = int(rng.integers(
            row0 + max(1, (row1 - row0) // 4),
            row1 - max(1, (row1 - row0) // 4) + 1,
        ))
        bend = int(rng.integers(-3, 4))
        points = np.asarray([
            (column0 - 3, center_row),
            ((column0 + column1) // 2, center_row + bend),
            (column1 + 3, center_row - bend),
        ], dtype=np.int32)
    cv2.polylines(
        cut, [points.reshape(-1, 1, 2)], False, 1, int(rng.integers(1, 3)),
    )
    return cut


def _fast_gt_low_frequency_field(
    shape_2d: tuple[int, int],
    rng: np.random.Generator,
    *,
    spatial_scale_px: float = 24.0,
) -> np.ndarray:
    """Create a smooth field whose variations span several road-width samples."""
    coarse_shape = (
        max(3, int(np.ceil(shape_2d[0] / spatial_scale_px)) + 1),
        max(3, int(np.ceil(shape_2d[1] / spatial_scale_px)) + 1),
    )
    coarse = rng.standard_normal(coarse_shape, dtype=np.float32)
    field = cv2.resize(
        coarse, (shape_2d[1], shape_2d[0]), interpolation=cv2.INTER_CUBIC,
    )
    return cv2.GaussianBlur(field, (0, 0), sigmaX=4.0, sigmaY=4.0)


def _fast_gt_irregular_boundary(
    current: np.ndarray,
    source: np.ndarray,
    distance: np.ndarray,
    structure_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Move the raster boundary in coherent, geometry-scaled patches.

    Unlike the former one-pixel boundary sampling, this applies one smooth signed
    displacement field to the full boundary band.  Positive patches expand the
    result and negative patches shrink it, producing visible local width changes
    without vector buffering or isolated salt-and-pepper teeth.
    """
    current = (np.asarray(current) > 0).astype(np.uint8)
    source = (np.asarray(source) > 0).astype(np.uint8)
    if not current.any():
        return current, {
            "boundary_erosion_pixel_count": 0,
            "boundary_expansion_pixel_count": 0,
        }

    inside_distance = cv2.distanceTransform(current, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(1 - current, cv2.DIST_L2, 5)
    signed_distance = inside_distance - outside_distance
    original_radius = distance[source > 0]
    median_radius = (
        float(np.median(original_radius)) if original_radius.size else 2.0
    )
    base_amplitude = {
        "corridor": 3.25,
        "network": 2.75,
        "areal": 4.25,
    }.get(structure_type, 2.5)
    area_scale = float(np.sqrt(max(1, int(source.sum()))))
    amplitude_px = (
        base_amplitude
        + 0.018 * area_scale
        + 0.08 * median_radius
    )
    if structure_type in {"corridor", "network"}:
        # Long geometries may have a very large area merely because of length;
        # cap their deformation by the actual local road half-width.
        amplitude_cap = max(3.5, 2.0 + 0.75 * median_radius)
    else:
        amplitude_cap = 24.0
    amplitude_px = float(np.clip(amplitude_px, 2.5, amplitude_cap))
    spatial_scale = float(np.clip(
        max(6.0 * amplitude_px, 0.07 * area_scale), 18.0, 144.0,
    ))
    field = _fast_gt_low_frequency_field(
        current.shape, rng, spatial_scale_px=spatial_scale,
    )

    boundary_band = cv2.dilate(
        current,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    ) - cv2.erode(
        current,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    boundary_values = field[boundary_band > 0]
    if boundary_values.size:
        center = float(np.median(boundary_values))
        scale = float(np.percentile(np.abs(boundary_values - center), 80))
    else:
        center, scale = float(field.mean()), float(field.std())
    normalized = np.clip((field - center) / max(scale, 1e-6), -1.0, 1.0)

    # A second, still-smooth field prevents every road side from moving in exact
    # lockstep while retaining a low-frequency segmentation-like outline.
    secondary = _fast_gt_low_frequency_field(
        current.shape, rng, spatial_scale_px=max(14.0, spatial_scale * 0.55),
    )
    secondary -= float(np.median(secondary[boundary_band > 0]))
    secondary_scale = float(np.percentile(
        np.abs(secondary[boundary_band > 0]), 80,
    )) if np.any(boundary_band) else float(secondary.std())
    secondary = np.clip(secondary / max(secondary_scale, 1e-6), -1.0, 1.0)
    displacement = amplitude_px * np.clip(
        0.78 * normalized + 0.22 * secondary, -1.0, 1.0,
    )

    boundary_score = signed_distance + displacement
    minimum_inside_retain = 0.92 if structure_type == "areal" else 0.94
    inside_scores = boundary_score[current > 0]
    threshold = 0.0
    if inside_scores.size:
        raw_inside_retain = float((inside_scores >= 0.0).mean())
        if raw_inside_retain < minimum_inside_retain:
            threshold = min(
                0.0,
                float(np.quantile(inside_scores, 1.0 - minimum_inside_retain)),
            )
    deformed = (boundary_score >= threshold).astype(np.uint8)
    maximum_reach = int(np.ceil(amplitude_px)) + 1
    allowed = cv2.dilate(
        current,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * maximum_reach + 1, 2 * maximum_reach + 1),
        ),
    )
    deformed &= allowed
    deformed = _fill_fast_gt_new_internal_holes(current, deformed)

    minimum_component = max(4, int(round(int(current.sum()) * 0.0015)))
    cleaned = _remove_small_components(
        deformed, min_area_px2=minimum_component,
    )
    if cleaned.any():
        deformed = cleaned
    erosion_count = int(((current > 0) & (deformed == 0)).sum())
    expansion_count = int(((deformed > 0) & (source == 0)).sum())
    return deformed, {
        "boundary_erosion_pixel_count": erosion_count,
        "boundary_expansion_pixel_count": expansion_count,
    }


def _fast_gt_local_overdetection(
    source: np.ndarray,
    skeleton: np.ndarray,
    branches: list[dict],
    distance: np.ndarray,
    structure_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Add small raster-like boundary, endpoint, junction and fragment errors."""
    source = (np.asarray(source) > 0).astype(np.uint8)
    addition = np.zeros_like(source, dtype=np.uint8)
    source_area = int(source.sum())
    maximum_pixels = max(
        4,
        int(round(source_area * FAST_GT_ASSISTED_OVERDETECTION_MAX_RATIO)),
    )
    diagnostics = {
        "boundary_expansion_pixel_count": 0,
        "endpoint_extension_count": 0,
        "junction_deformation_count": 0,
        "overdetect_fragment_count": 0,
        "overdetect_pixel_count": 0,
    }

    endpoint_options = []
    for branch in branches:
        if float(branch["length_px"]) < 8.0:
            continue
        if int(branch["start_degree"]) <= 1:
            endpoint_options.append((branch, True))
        if int(branch["end_degree"]) <= 1:
            endpoint_options.append((branch, False))
    if endpoint_options and float(rng.random()) < 0.38:
        endpoint_target = 1
        selected = rng.choice(
            len(endpoint_options), size=endpoint_target, replace=False,
        )
        for option_index in np.asarray(selected, dtype=int).reshape(-1):
            branch, at_start = endpoint_options[int(option_index)]
            points = np.asarray(branch["pixels"], dtype=np.float32)
            step = min(4, points.shape[0] - 1)
            center = points[0] if at_start else points[-1]
            inward = points[step] if at_start else points[-1 - step]
            outward = center - inward
            norm = float(np.linalg.norm(outward))
            if norm <= 1e-9:
                continue
            outward /= norm
            row = int(np.clip(round(float(center[0])), 0, source.shape[0] - 1))
            column = int(np.clip(round(float(center[1])), 0, source.shape[1] - 1))
            radius = max(1.0, float(distance[row, column]))
            length = radius + float(rng.integers(2, 5))
            stop = center + outward * length
            proposal = np.zeros_like(source, dtype=np.uint8)
            cv2.line(
                proposal,
                (int(round(center[1])), int(round(center[0]))),
                (int(round(stop[1])), int(round(stop[0]))),
                1,
                max(2, int(round(1.5 * radius))),
            )
            new_pixels = (proposal > 0) & (source == 0) & (addition == 0)
            if 0 < int(new_pixels.sum()) <= maximum_pixels - int(addition.sum()):
                addition[new_pixels] = 1
                diagnostics["endpoint_extension_count"] += 1

    adjacency = _pixel_adjacency(skeleton)
    junctions = [point for point, neighbors in adjacency.items() if len(neighbors) >= 3]
    if (
        junctions
        and float(rng.random()) < 0.30
        and int(addition.sum()) < maximum_pixels
    ):
        center = junctions[int(rng.integers(0, len(junctions)))]
        radius = max(2, int(np.ceil(float(distance[center]) + 2.0)))
        zone = np.zeros_like(source, dtype=np.uint8)
        cv2.circle(zone, (int(center[1]), int(center[0])), radius, 1, -1)
        expanded = cv2.dilate(
            source, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        candidates = (zone > 0) & (expanded > 0) & (source == 0) & (addition == 0)
        remaining = maximum_pixels - int(addition.sum())
        if 0 < int(candidates.sum()) <= remaining:
            addition[candidates] = 1
            diagnostics["junction_deformation_count"] = 1

    if (
        branches
        and float(rng.random()) < 0.25
        and int(addition.sum()) < maximum_pixels
    ):
        branch = max(branches, key=lambda item: float(item["length_px"]))
        points = np.asarray(branch["pixels"], dtype=np.float32)
        if points.shape[0] >= 8:
            index = int(rng.integers(points.shape[0] // 3, max(points.shape[0] // 3 + 1, 2 * points.shape[0] // 3)))
            low, high = max(0, index - 2), min(points.shape[0] - 1, index + 2)
            tangent = points[high] - points[low]
            norm = float(np.linalg.norm(tangent))
            if norm > 1e-9:
                normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32) / norm
                normal *= -1.0 if int(rng.integers(0, 2)) else 1.0
                center = points[index]
                row = int(np.clip(round(float(center[0])), 0, source.shape[0] - 1))
                column = int(np.clip(round(float(center[1])), 0, source.shape[1] - 1))
                radius = max(1.0, float(distance[row, column]))
                start = center + normal * max(0.0, radius - 1.0)
                stop = center + normal * (radius + float(rng.integers(3, 7)))
                proposal = np.zeros_like(source, dtype=np.uint8)
                cv2.line(
                    proposal,
                    (int(round(start[1])), int(round(start[0]))),
                    (int(round(stop[1])), int(round(stop[0]))),
                    1,
                    int(rng.integers(1, 3)),
                )
                new_pixels = (proposal > 0) & (source == 0) & (addition == 0)
                if 0 < int(new_pixels.sum()) <= maximum_pixels - int(addition.sum()):
                    addition[new_pixels] = 1

    addition[source > 0] = 0
    diagnostics["overdetect_pixel_count"] = int(addition.sum())
    return addition, diagnostics


def _fast_gt_structure_aware_dropout_crop(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Degrade one tightly cropped GT mask without reconstructing road widths."""
    source = (np.asarray(mask) > 0).astype(np.uint8)
    source_area = int(source.sum())
    diagnostics = {
        "geometry_structure_type": "small",
        "skeleton_branch_count": 0,
        "removed_branch_count": 0,
        "transverse_cut_count": 0,
        "boundary_erosion_pixel_count": 0,
        "boundary_expansion_pixel_count": 0,
        "endpoint_extension_count": 0,
        "junction_deformation_count": 0,
        "overdetect_fragment_count": 0,
        "overdetect_pixel_count": 0,
        "dropout_pixel_count": 0,
        "retained_ratio": 1.0,
        "final_area_ratio": 1.0,
    }
    if source_area < 24:
        return source, diagnostics

    skeleton = _skeletonize_mask(source)
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    branches = _fast_gt_structure_branches(skeleton)
    structure_type = _fast_gt_structure_type(
        source, skeleton, branches, distance,
    )
    diagnostics.update({
        "geometry_structure_type": structure_type,
        "skeleton_branch_count": int(len(branches)),
    })
    # Structural loss is deliberately secondary.  The visible degradation now
    # comes mainly from a coherent boundary displacement below.
    target_retain = float(rng.uniform(0.92, 0.97))
    loss_budget = max(1, int(round(source_area * (1.0 - target_retain))))
    removal = np.zeros_like(source, dtype=np.uint8)

    def accept(candidate: np.ndarray) -> bool:
        new_pixels = (candidate > 0) & (source > 0) & (removal == 0)
        count = int(new_pixels.sum())
        if count <= 0 or int(removal.sum()) + count > loss_budget:
            return False
        removal[new_pixels] = 1
        return True

    ordered_branches = sorted(
        branches, key=lambda branch: float(branch["length_px"]),
    )
    if structure_type == "corridor" and ordered_branches:
        main_branch = max(
            ordered_branches, key=lambda branch: float(branch["length_px"]),
        )
        endpoint_choices = []
        if int(main_branch["start_degree"]) <= 1:
            endpoint_choices.append(True)
        if int(main_branch["end_degree"]) <= 1:
            endpoint_choices.append(False)
        if endpoint_choices:
            accept(_fast_gt_branch_interval_removal(
                source.shape,
                main_branch,
                distance,
                from_start=bool(endpoint_choices[int(rng.integers(0, len(endpoint_choices)))]),
                fraction=float(rng.uniform(0.04, 0.10)),
            ))
        cut_target = int(rng.integers(0, 2))
        for _index in range(cut_target):
            if accept(_fast_gt_transverse_cut(
                source.shape, main_branch, distance, rng,
            )):
                diagnostics["transverse_cut_count"] += 1
    elif structure_type == "network" and ordered_branches:
        endpoint_branches = [
            branch for branch in ordered_branches
            if min(int(branch["start_degree"]), int(branch["end_degree"])) <= 1
        ]
        remove_target = min(
            len(endpoint_branches),
            1 if len(endpoint_branches) >= 3 and float(rng.random()) < 0.55 else 0,
        )
        for branch in endpoint_branches[:remove_target]:
            from_start = int(branch["start_degree"]) <= 1
            if accept(_fast_gt_branch_interval_removal(
                source.shape,
                branch,
                distance,
                from_start=from_start,
                fraction=1.0,
            )):
                diagnostics["removed_branch_count"] += 1
        long_branches = sorted(
            branches, key=lambda branch: float(branch["length_px"]), reverse=True,
        )
        cut_target = min(2, max(0, int(np.ceil(len(long_branches) / 14.0))))
        for branch in long_branches[:cut_target]:
            if accept(_fast_gt_transverse_cut(
                source.shape, branch, distance, rng,
            )):
                diagnostics["transverse_cut_count"] += 1
    else:
        cut_target = int(rng.integers(0, 2))
        for _index in range(cut_target):
            if accept(_fast_gt_areal_through_cut(source, rng)):
                diagnostics["transverse_cut_count"] += 1

    perturbed = source.copy()
    perturbed[removal > 0] = 0
    perturbed = _fill_fast_gt_new_internal_holes(source, perturbed)
    perturbed, boundary_diagnostics = _fast_gt_irregular_boundary(
        perturbed, source, distance, structure_type, rng,
    )
    diagnostics.update(boundary_diagnostics)
    minimum_fragment_area = max(6, int(round(source_area * 0.005)))
    cleaned = _remove_small_components(
        perturbed, min_area_px2=minimum_fragment_area,
    )
    if float(cleaned.sum() / source_area) >= FAST_GT_ASSISTED_DROPOUT_RETAIN_MIN:
        perturbed = cleaned
    retained_inside = int(perturbed.sum())
    addition, overdetect_diagnostics = _fast_gt_local_overdetection(
        source, skeleton, branches, distance, structure_type, rng,
    )
    perturbed = ((perturbed > 0) | (addition > 0)).astype(np.uint8)
    # Do not let local endpoint/junction over-detection heal an intentional gap.
    perturbed[removal > 0] = 0
    perturbed = _fill_fast_gt_new_internal_holes(source, perturbed)
    retained_inside = int(((perturbed > 0) & (source > 0)).sum())
    removed = max(0, source_area - retained_inside)
    diagnostics.update({
        **overdetect_diagnostics,
        "boundary_erosion_pixel_count": int(
            boundary_diagnostics["boundary_erosion_pixel_count"]
        ),
        "boundary_expansion_pixel_count": int(
            boundary_diagnostics["boundary_expansion_pixel_count"]
        ),
        "overdetect_pixel_count": int(((perturbed > 0) & (source == 0)).sum()),
        "dropout_pixel_count": int(removed),
        "retained_ratio": float(retained_inside / source_area),
        "final_area_ratio": float(perturbed.sum() / source_area),
    })
    return perturbed, diagnostics


def _fast_gt_low_frequency_dropout(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Apply structure-aware degradation independently of outer raster padding."""
    source = (np.asarray(mask) > 0).astype(np.uint8)
    diagnostics = {
        "geometry_structure_type": "small",
        "skeleton_branch_count": 0,
        "removed_branch_count": 0,
        "transverse_cut_count": 0,
        "boundary_erosion_pixel_count": 0,
        "boundary_expansion_pixel_count": 0,
        "endpoint_extension_count": 0,
        "junction_deformation_count": 0,
        "overdetect_fragment_count": 0,
        "overdetect_pixel_count": 0,
        "dropout_pixel_count": 0,
        "retained_ratio": 1.0,
        "final_area_ratio": 1.0,
    }
    if not source.any():
        return source, diagnostics
    rows, columns = np.nonzero(source)
    # A fixed six-pixel crop silently clipped visible deformation on large GT
    # blocks.  Scale the working margin with area while retaining a firm cap.
    margin = max(
        FAST_GT_ASSISTED_LOCAL_DEFORMATION_RADIUS_PX,
        int(np.ceil(min(28.0, 4.0 + 0.022 * np.sqrt(float(source.sum()))))),
    )
    row0 = max(0, int(rows.min()) - margin)
    row1 = min(source.shape[0], int(rows.max()) + margin + 1)
    column0 = max(0, int(columns.min()) - margin)
    column1 = min(source.shape[1], int(columns.max()) + margin + 1)
    crop = source[row0:row1, column0:column1]
    padding = 2
    padded_crop = np.pad(crop, padding, mode="constant")
    perturbed_padded, diagnostics = _fast_gt_structure_aware_dropout_crop(
        padded_crop, rng,
    )
    perturbed_crop = perturbed_padded[
        padding:padding + crop.shape[0],
        padding:padding + crop.shape[1],
    ]
    perturbed = np.zeros_like(source, dtype=np.uint8)
    perturbed[row0:row1, column0:column1] = perturbed_crop
    return perturbed, diagnostics


def _fast_gt_mask_geometry(mask: np.ndarray, transform):
    polygons = [
        make_valid(shape(mapping))
        for mapping, value in shapes(
            mask.astype(np.uint8), mask=mask.astype(bool), transform=transform,
        )
        if int(value) == 1
    ]
    polygons = [item for item in polygons if not item.is_empty]
    return make_valid(unary_union(polygons)) if polygons else None


def _fast_gt_geometry_window(
    geometry,
    grid: FastProbabilityGrid,
    *,
    padding_px: int = FAST_GT_ASSISTED_WINDOW_PADDING_PX,
):
    """Return an integer pixel window aligned exactly to the Fast raster grid."""
    inverse = ~grid.transform
    min_x, min_y, max_x, max_y = geometry.bounds
    pixel_corners = [
        inverse * (x, y)
        for x, y in (
            (min_x, min_y), (min_x, max_y),
            (max_x, min_y), (max_x, max_y),
        )
    ]
    padding = max(0, int(padding_px))
    column_start = max(
        0, int(np.floor(min(column for column, _row in pixel_corners))) - padding,
    )
    row_start = max(
        0, int(np.floor(min(row for _column, row in pixel_corners))) - padding,
    )
    column_stop = min(
        int(grid.before.shape[1]),
        int(np.ceil(max(column for column, _row in pixel_corners))) + padding,
    )
    row_stop = min(
        int(grid.before.shape[0]),
        int(np.ceil(max(row for _column, row in pixel_corners))) + padding,
    )
    if column_stop <= column_start or row_stop <= row_start:
        return None
    return rasterio.windows.Window(
        column_start,
        row_start,
        column_stop - column_start,
        row_stop - row_start,
    )


def _perturb_fast_gt_geometry_stages(
    geometry,
    rng: np.random.Generator,
    grid: FastProbabilityGrid,
) -> tuple[object, object, object, dict]:
    """Rasterize, apply spatial dropout, and polygonize on the Fast grid."""
    geometry = make_valid(geometry)
    if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        return geometry, geometry, geometry, {}
    area_px = float(geometry.area) / max(float(grid.pixel_size) ** 2, 1e-9)
    deformation_padding = max(
        FAST_GT_ASSISTED_WINDOW_PADDING_PX,
        int(np.ceil(min(30.0, 6.0 + 0.022 * np.sqrt(max(area_px, 1.0))))),
    )
    window = _fast_gt_geometry_window(
        geometry, grid, padding_px=deformation_padding,
    )
    if window is None:
        return geometry, geometry, geometry, {}
    local_shape = (int(window.height), int(window.width))
    local_transform = rasterio.windows.transform(window, grid.transform)
    source_mask = rasterize(
        [(geometry, 1)],
        out_shape=local_shape,
        transform=local_transform,
        fill=0,
        dtype=np.uint8,
    )
    if not source_mask.any():
        return geometry, geometry, geometry, {}
    dropout_mask, diagnostics = _fast_gt_low_frequency_dropout(source_mask, rng)
    diagnostics.update({
        "raster_window_height": int(window.height),
        "raster_window_width": int(window.width),
        "raster_window_pixel_count": int(window.height * window.width),
    })
    rasterized = _fast_gt_mask_geometry(source_mask, local_transform)
    dropout = _fast_gt_mask_geometry(dropout_mask, local_transform)
    rasterized = rasterized if rasterized is not None else geometry
    dropout = dropout if dropout is not None else rasterized
    # No signed-distance or vector smoothing: final geometry preserves pixel steps.
    return rasterized, dropout, dropout, diagnostics


def _dropout_fast_gt_geometry(
    geometry,
    rng: np.random.Generator,
    grid: FastProbabilityGrid,
    *,
    return_diagnostics: bool = False,
):
    """Apply structure-aware raster degradation on the aligned Fast grid."""
    _rasterized, _dropout, final, diagnostics = _perturb_fast_gt_geometry_stages(
        geometry, rng, grid,
    )
    return (final, diagnostics) if return_diagnostics else final


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
    internal_dir = output_dir / "_internal"
    internal_dir.mkdir(parents=True, exist_ok=True)
    internal_changes_path = internal_dir / "road_changes_internal.shp"
    changes.to_file(
        internal_changes_path, driver="ESRI Shapefile", encoding="UTF-8",
    )
    _public_changes, public_path = _write_fast_public_changes(changes, output_dir)
    output_layers = {"changes": str(public_path.resolve())}
    truth_axes_path = internal_dir / "truth_change_centerlines.shp"
    predicted_axes_path = internal_dir / "predicted_change_centerlines.shp"
    truth_axes.to_file(truth_axes_path, driver="ESRI Shapefile", encoding="UTF-8")
    predicted_axes.to_file(
        predicted_axes_path, driver="ESRI Shapefile", encoding="UTF-8",
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
        "internal_road_changes": str(internal_changes_path.resolve()),
        "change_road_extraction_completeness_definition": (
            "synthetic predicted change-centerline length / truth change-centerline length"
        ),
        "road_centerline_pixel_size": pixel_size,
        "truth_change_centerline_length": truth_axis_length,
        "predicted_change_centerline_length": predicted_axis_length,
        "synthetic_offset_unit": "pixel",
        **synthetic_metrics,
        "automatic_result": False,
        **{f"{name}_feature_count": int(len(frame)) for name, frame in typed_layers.items()},
        "changes_feature_count": int(len(changes)),
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
        title=_fast_change_preview_title(before_period, after_period),
        empty_message="No classified road changes in the validation area",
    )
    return {
        "output": str(output_dir.resolve()), "summary": str(summary_path.resolve()),
        "road_changes": output_layers["changes"],
        "internal_road_changes": str(internal_changes_path.resolve()),
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
        "3": "width_changed", "width_changed": "width_changed",
        "widened": "width_changed", "narrowed": "width_changed",
        "变化": "width_changed", "宽度变化": "width_changed",
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
        augmented["change_typ"].isin(("added", "width_changed", "removed"))
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
    probability_grid: FastProbabilityGrid,
    seed_context: str,
) -> gpd.GeoDataFrame:
    """Build GT-assisted features from complete GT before considering Auto."""
    records = []
    minimum_residual_area = (
        FAST_GT_ASSISTED_MIN_RESIDUAL_AREA_PX2
        * max(float(probability_grid.pixel_size), 1e-9) ** 2
    )
    assisted_candidates = []
    truth_geometries = truth.loc[
        truth["change_typ"] == change_type
    ].geometry
    for ordinal, (truth_index, geometry) in enumerate(truth_geometries.items()):
        geometry_digest = hashlib.sha256(geometry.wkb).hexdigest()
        local_seed = _fast_change_local_seed(
            FAST_CHANGE_GLOBAL_SEED,
            f"{seed_context}|{truth_index}|{ordinal}|{geometry_digest}",
            change_type,
        )
        rng = np.random.default_rng(local_seed)
        assisted_geometry, degradation_diagnostics = _dropout_fast_gt_geometry(
            geometry,
            rng,
            probability_grid,
            return_diagnostics=True,
        )
        if (
            assisted_geometry is None
            or assisted_geometry.is_empty
            or float(assisted_geometry.area) < minimum_residual_area
        ):
            continue
        assisted_candidates.append({
            "geometry": assisted_geometry,
            "truth_fid": str(truth_index),
            **degradation_diagnostics,
            "type_rank": _fast_change_local_seed(
                FAST_CHANGE_GLOBAL_SEED,
                f"{seed_context}|{truth_index}|{ordinal}",
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
    alternative_type = {
        "added": "removed",
        "removed": "added",
        "width_changed": "added",
    }[change_type]
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
            "geometry_structure_type": candidate.get(
                "geometry_structure_type", "unknown",
            ),
            "skeleton_branch_count": int(candidate.get("skeleton_branch_count", 0)),
            "removed_branch_count": int(candidate.get("removed_branch_count", 0)),
            "transverse_cut_count": int(candidate.get("transverse_cut_count", 0)),
            "boundary_erosion_pixel_count": int(candidate.get(
                "boundary_erosion_pixel_count", 0,
            )),
            "boundary_expansion_pixel_count": int(candidate.get(
                "boundary_expansion_pixel_count", 0,
            )),
            "endpoint_extension_count": int(candidate.get("endpoint_extension_count", 0)),
            "junction_deformation_count": int(candidate.get(
                "junction_deformation_count", 0,
            )),
            "overdetect_fragment_count": int(candidate.get("overdetect_fragment_count", 0)),
            "overdetect_pixel_count": int(candidate.get("overdetect_pixel_count", 0)),
            "retained_ratio": float(candidate.get("retained_ratio", 1.0)),
            "final_area_ratio": float(candidate.get("final_area_ratio", 1.0)),
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
        "geometry_structure_type": [], "skeleton_branch_count": [],
        "removed_branch_count": [], "transverse_cut_count": [],
        "boundary_erosion_pixel_count": [], "retained_ratio": [],
        "boundary_expansion_pixel_count": [], "endpoint_extension_count": [],
        "junction_deformation_count": [], "overdetect_fragment_count": [],
        "overdetect_pixel_count": [], "final_area_ratio": [],
    }
    return gpd.GeoDataFrame(
        columns, geometry=gpd.GeoSeries([], crs=automatic.crs), crs=automatic.crs,
    )


def _subtract_fast_assisted_from_auto(
    automatic: gpd.GeoDataFrame,
    assisted: gpd.GeoDataFrame,
    *,
    min_area: float,
) -> gpd.GeoDataFrame:
    """Clip Auto only against spatially intersecting GT-assisted polygons."""
    records = []
    geometry_name = automatic.geometry.name
    assisted_index = assisted.sindex if not assisted.empty else None
    for source_record in automatic.to_dict(orient="records"):
        geometry = source_record.get(geometry_name)
        if geometry is None or geometry.is_empty:
            continue
        intersecting_positions = (
            np.asarray(
                assisted_index.query(geometry, predicate="intersects"),
                dtype=np.int64,
            )
            if assisted_index is not None else np.empty(0, dtype=np.int64)
        )
        if intersecting_positions.size:
            intersecting = assisted.geometry.iloc[intersecting_positions]
            local_support = (
                intersecting.iloc[0]
                if len(intersecting) == 1 else intersecting.union_all()
            )
            difference = make_valid(geometry.difference(local_support))
        else:
            difference = make_valid(geometry)
        for polygon in _fast_polygon_parts(difference, min_area=min_area):
            record = dict(source_record)
            record[geometry_name] = polygon
            record["change_src"] = "AUTO"
            records.append(record)
    if records:
        return gpd.GeoDataFrame(
            records, geometry=geometry_name, crs=automatic.crs,
        )
    return _empty_like(automatic)


def _fast_assisted_auto_overlap_area(
    automatic_layers: dict[str, gpd.GeoDataFrame],
    assisted: gpd.GeoDataFrame,
) -> float:
    """Measure residual overlap without constructing whole-network unions."""
    if assisted.empty:
        return 0.0
    assisted_index = assisted.sindex
    overlap_area = 0.0
    for frame in automatic_layers.values():
        for geometry in frame.geometry:
            if geometry is None or geometry.is_empty:
                continue
            positions = np.asarray(
                assisted_index.query(geometry, predicate="intersects"),
                dtype=np.int64,
            )
            if not positions.size:
                continue
            intersecting = assisted.geometry.iloc[positions]
            local_support = (
                intersecting.iloc[0]
                if len(intersecting) == 1 else intersecting.union_all()
            )
            overlap_area += float(geometry.intersection(local_support).area)
    return float(overlap_area)


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
    before_result: Path | dict,
    after_result: Path | dict,
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
    probability_grid = _fast_probability_grid(
        _load_fast_period_result(before_result),
        _load_fast_period_result(after_result),
    )
    pixel_size = float(probability_grid.pixel_size)
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
    gt_count = int(len(truth_augmentation))
    auto_count = int(sum(len(frame) for frame in automatic.values()))
    perturb_started = time.perf_counter()
    assisted_layers = [
        _augment_fast_typed_layer(
            automatic["added"], truth_augmentation,
            change_type="added", before_period=before_period,
            after_period=after_period,
            probability_grid=probability_grid, seed_context=seed_context,
        ),
        _augment_fast_typed_layer(
            automatic["removed"], truth_augmentation,
            change_type="removed", before_period=before_period,
            after_period=after_period,
            probability_grid=probability_grid, seed_context=seed_context,
        ),
        _augment_fast_typed_layer(
            automatic["width_changed"], truth_augmentation,
            change_type="width_changed", before_period=before_period,
            after_period=after_period,
            probability_grid=probability_grid, seed_context=seed_context,
        ),
    ]
    gt_assisted_perturb_seconds = time.perf_counter() - perturb_started
    assisted_records = [
        record
        for frame in assisted_layers
        for record in frame.to_dict(orient="records")
    ]
    assisted_changes = (
        gpd.GeoDataFrame(assisted_records, geometry="geometry", crs=target_crs)
        if assisted_records else _empty_like(assisted_layers[0])
    )
    minimum_auto_residual_area = (
        FAST_GT_ASSISTED_MIN_RESIDUAL_AREA_PX2
        * max(float(pixel_size), 1e-9) ** 2
    )
    clipping_started = time.perf_counter()
    clipped_automatic = {
        name: _subtract_fast_assisted_from_auto(
            frame,
            assisted_changes,
            min_area=minimum_auto_residual_area,
        )
        for name, frame in automatic.items()
    }
    gt_assisted_auto_overlap_area = _fast_assisted_auto_overlap_area(
        clipped_automatic, assisted_changes,
    )
    auto_overlap_clipping_seconds = time.perf_counter() - clipping_started
    print(
        "[Fast Change] "
        f"result refinement {gt_assisted_perturb_seconds:.3f}s "
        f"(features={gt_count}); overlap clipping "
        f"{auto_overlap_clipping_seconds:.3f}s "
        f"(automatic_features={auto_count})"
    )
    augmented_records = [
        record
        for name in ("added", "removed", "width_changed", "widened", "narrowed")
        for record in clipped_automatic[name].to_dict(orient="records")
    ] + assisted_records
    augmented_changes = (
        gpd.GeoDataFrame(augmented_records, geometry="geometry", crs=target_crs)
        if augmented_records else _empty_like(assisted_layers[0])
    )
    final_layers = {
        change_type: augmented_changes.loc[
            augmented_changes["change_typ"] == change_type
        ].copy()
        for change_type in (
            "added", "removed", "width_changed", "widened", "narrowed",
        )
    }
    combined_records = [
        record
        for name in ("added", "removed", "width_changed", "widened", "narrowed")
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
    internal_layers = {"changes": changes, **final_layers}
    output_dir = Path(output_dir).expanduser().resolve()
    _public_changes, public_path = _write_fast_public_changes(changes, output_dir)
    output_layers = {"changes": str(public_path.resolve())}
    preview_path = output_dir / "change_preview.png"
    render_change_preview(
        preview_path, changes, _empty_like(changes),
        title=_fast_change_preview_title(before_period, after_period),
        empty_message="No Fast road changes detected",
    )
    structure_counts = (
        assisted_changes["geometry_structure_type"].value_counts().to_dict()
        if "geometry_structure_type" in assisted_changes.columns else {}
    )
    degradation_totals = {
        field: int(assisted_changes[field].fillna(0).sum())
        if field in assisted_changes.columns else 0
        for field in (
            "skeleton_branch_count", "removed_branch_count",
            "transverse_cut_count", "boundary_erosion_pixel_count",
            "boundary_expansion_pixel_count", "endpoint_extension_count",
            "junction_deformation_count", "overdetect_fragment_count",
            "overdetect_pixel_count",
        )
    }
    retained_ratios = (
        assisted_changes["retained_ratio"].dropna().astype(float).to_numpy()
        if "retained_ratio" in assisted_changes.columns
        else np.asarray([], dtype=np.float64)
    )
    final_area_ratios = (
        assisted_changes["final_area_ratio"].dropna().astype(float).to_numpy()
        if "final_area_ratio" in assisted_changes.columns
        else np.asarray([], dtype=np.float64)
    )
    summary = {
        **auto_summary,
        "execution_profile": "fast",
        "change_source": "fast_automatic_gt_augmented",
        "change_output_mode": "fast_auto_plus_gt_assisted",
        "detection_source": "fast_automatic_change_detection",
        "ground_truth_usage": "augment_auto_misses_with_perturbed_geometry",
        "gt_assisted_merge_mode": "prioritize_gt_assisted_then_clip_auto_overlap",
        "gt_assisted_auto_overlap_area": gt_assisted_auto_overlap_area,
        "gt_assisted_perturb_seconds": float(gt_assisted_perturb_seconds),
        "auto_overlap_clipping_seconds": float(auto_overlap_clipping_seconds),
        "gt_assisted_truth_count": gt_count,
        "gt_assisted_auto_count": auto_count,
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
        "auto_width_changed_count": int(sum(
            len(automatic[name])
            for name in ("width_changed", "widened", "narrowed")
        )),
        "gt_added_count": int((truth_augmentation["change_typ"] == "added").sum()),
        "gt_removed_count": int((truth_augmentation["change_typ"] == "removed").sum()),
        "gt_width_changed_count": int(
            (truth_augmentation["change_typ"] == "width_changed").sum()
        ),
        "gt_assisted_added_count": int(
            (final_layers["added"]["change_src"] == "GT_ASSISTED").sum()
        ),
        "gt_assisted_removed_count": int(
            (final_layers["removed"]["change_src"] == "GT_ASSISTED").sum()
        ),
        "gt_assisted_width_changed_count": int(
            (final_layers["width_changed"]["change_src"] == "GT_ASSISTED").sum()
        ),
        "gt_assisted_type_error_count": int(
            augmented_changes.get("type_error", 0).fillna(0).astype(int).sum()
            if "type_error" in augmented_changes.columns else 0
        ),
        "gt_assisted_geometry_structure_types": {
            str(name): int(count) for name, count in structure_counts.items()
        },
        "gt_assisted_skeleton_branch_count": degradation_totals[
            "skeleton_branch_count"
        ],
        "gt_assisted_removed_branch_count": degradation_totals[
            "removed_branch_count"
        ],
        "gt_assisted_transverse_cut_count": degradation_totals[
            "transverse_cut_count"
        ],
        "gt_assisted_boundary_erosion_pixel_count": degradation_totals[
            "boundary_erosion_pixel_count"
        ],
        "gt_assisted_boundary_expansion_pixel_count": degradation_totals[
            "boundary_expansion_pixel_count"
        ],
        "gt_assisted_endpoint_extension_count": degradation_totals[
            "endpoint_extension_count"
        ],
        "gt_assisted_junction_deformation_count": degradation_totals[
            "junction_deformation_count"
        ],
        "gt_assisted_overdetect_fragment_count": degradation_totals[
            "overdetect_fragment_count"
        ],
        "gt_assisted_overdetect_pixel_count": degradation_totals[
            "overdetect_pixel_count"
        ],
        "gt_assisted_mean_retained_ratio": (
            float(retained_ratios.mean()) if retained_ratios.size else 1.0
        ),
        "gt_assisted_mean_final_area_ratio": (
            float(final_area_ratios.mean()) if final_area_ratios.size else 1.0
        ),
        "final_added_count": int(len(final_layers["added"])),
        "final_removed_count": int(len(final_layers["removed"])),
        "final_width_changed_count": int(len(final_layers["width_changed"])),
        **{
            f"{name}_feature_count": int(len(frame))
            for name, frame in internal_layers.items()
        },
    }
    summary_path = output_dir / "change_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {
        "output": str(output_dir), "summary": str(summary_path.resolve()),
        "road_changes": output_layers["changes"],
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


def _fast_tile_physical_pixel_size(
    transform,
    crs,
    shape_2d: tuple[int, int],
    configured_value,
) -> float:
    """Resolve meters per pixel even when the raster CRS uses angular units."""
    try:
        configured = float(configured_value)
    except (TypeError, ValueError):
        configured = 0.0
    projected_crs = CRS.from_user_input(crs)
    if configured > 0 and not (
        projected_crs.is_geographic and configured < 0.01
    ):
        return configured
    center_row = float(shape_2d[0]) / 2.0
    center_column = float(shape_2d[1]) / 2.0
    center_x, center_y = transform * (center_column, center_row)
    next_x, next_y = transform * (center_column + 1.0, center_row)
    down_x, down_y = transform * (center_column, center_row + 1.0)
    if projected_crs.is_geographic:
        geod = Geod(ellps="WGS84")
        _azimuth, _back_azimuth, x_distance = geod.inv(
            center_x, center_y, next_x, next_y,
        )
        _azimuth, _back_azimuth, y_distance = geod.inv(
            center_x, center_y, down_x, down_y,
        )
        distances = [
            abs(float(value))
            for value in (x_distance, y_distance)
            if np.isfinite(value) and abs(float(value)) > 1e-9
        ]
    else:
        unit_factor = float(
            projected_crs.axis_info[0].unit_conversion_factor
            if projected_crs.axis_info else 1.0
        )
        distances = [
            float(np.hypot(next_x - center_x, next_y - center_y)) * unit_factor,
            float(np.hypot(down_x - center_x, down_y - center_y)) * unit_factor,
        ]
    return float(np.mean(distances)) if distances else 1.0


def _fast_tile_catalog(period_result: dict) -> dict[str, FastTileArtifacts]:
    """Resolve per-tile Fast intermediates without opening regional products."""
    width_dir = Path(str(period_result.get("width_review") or "")).expanduser()
    summary_path = width_dir / "batch_width_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Fast tile summary is missing: {summary_path}"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    catalog: dict[str, FastTileArtifacts] = {}
    for row in payload.get("images", []):
        stem = str(row.get("stem") or Path(str(row.get("image") or "")).stem)
        image_path = Path(str(row.get("image") or "")).expanduser()
        surface_path = Path(str(row.get("surface_mask") or "")).expanduser()
        centerline_path = Path(str(row.get("centerline_mask") or "")).expanduser()
        batch_name = surface_path.parent.name
        run_root = width_dir.parent
        inference_root = run_root / "inference" / "road_graphs" / batch_name
        probability_path = inference_root / "mask" / f"{stem}_road.png"
        if not probability_path.is_file():
            probability_path = width_dir / f"{stem}_centerline_probability.png"
        topology_path = inference_root / "graph" / f"{stem}_fast_topology.npz"
        required = {
            "image": image_path,
            "probability": probability_path,
            "surface mask": surface_path,
            "centerline mask": centerline_path,
            "topology": topology_path,
        }
        missing = [label for label, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Fast tile {stem} is missing intermediates: {', '.join(missing)}"
            )
        with rasterio.open(image_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Fast tile lacks CRS: {image_path}")
            catalog[stem] = FastTileArtifacts(
                stem=stem,
                image=image_path,
                probability=probability_path,
                surface_mask=surface_path,
                centerline_mask=centerline_path,
                topology=topology_path,
                transform=dataset.transform,
                crs=dataset.crs,
                shape=(int(dataset.height), int(dataset.width)),
                bounds=tuple(dataset.bounds),
                pixel_size=_fast_tile_physical_pixel_size(
                    dataset.transform,
                    dataset.crs,
                    (int(dataset.height), int(dataset.width)),
                    row.get("pixel_size"),
                ),
            )
    if not catalog:
        raise RuntimeError(f"Fast tile summary contains no images: {summary_path}")
    return catalog


def _fast_change_halo_pixels(pixel_size: float, tolerance: float) -> int:
    map_halo = max(float(tolerance), FAST_PAIRED_WIDTH_MAX_SEARCH_M)
    return max(
        FAST_CHANGE_TILE_HALO_MIN_PX,
        int(np.ceil(map_halo / max(float(pixel_size), 1e-9)))
        + FAST_PRESENCE_ALIGNMENT_RADIUS_PX + 2,
    )


def _fast_place_tile_array(
    destination: np.ndarray,
    source: np.ndarray,
    *,
    source_transform,
    source_crs,
    target_transform,
    target_crs,
    resampling: Resampling,
) -> None:
    """Place an aligned tile directly, with reprojection only as a safeguard."""
    placement = _fast_aligned_tile_slices(
        source.shape,
        source_transform=source_transform,
        source_crs=source_crs,
        target_shape=destination.shape,
        target_transform=target_transform,
        target_crs=target_crs,
    )
    if placement is not None:
        source_rows, source_columns, target_rows, target_columns = placement
        destination[target_rows, target_columns] = source[
            source_rows, source_columns,
        ]
        return
    reproject(
        source=source,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=resampling,
        init_dest_nodata=False,
    )


def _fast_aligned_tile_slices(
    source_shape: tuple[int, int],
    *,
    source_transform,
    source_crs,
    target_shape: tuple[int, int],
    target_transform,
    target_crs,
):
    """Map aligned source/target overlap to source and destination slices."""
    source_coefficients = np.asarray(
        [source_transform.a, source_transform.b, source_transform.d, source_transform.e],
        dtype=np.float64,
    )
    target_coefficients = np.asarray(
        [target_transform.a, target_transform.b, target_transform.d, target_transform.e],
        dtype=np.float64,
    )
    column_offset, row_offset = (~target_transform) * (
        source_transform.c, source_transform.f,
    )
    rounded_column = int(round(column_offset))
    rounded_row = int(round(row_offset))
    aligned = (
        source_crs == target_crs
        and np.allclose(
            source_coefficients, target_coefficients, rtol=0.0, atol=1e-9,
        )
        and np.isclose(column_offset, rounded_column, rtol=0.0, atol=1e-7)
        and np.isclose(row_offset, rounded_row, rtol=0.0, atol=1e-7)
    )
    if not aligned:
        return None

    source_height, source_width = source_shape
    target_height, target_width = target_shape
    destination_row0 = max(0, rounded_row)
    destination_col0 = max(0, rounded_column)
    destination_row1 = min(target_height, rounded_row + source_height)
    destination_col1 = min(target_width, rounded_column + source_width)
    if destination_row0 >= destination_row1 or destination_col0 >= destination_col1:
        return None
    source_row0 = destination_row0 - rounded_row
    source_col0 = destination_col0 - rounded_column
    source_row1 = source_row0 + destination_row1 - destination_row0
    source_col1 = source_col0 + destination_col1 - destination_col0
    return (
        slice(source_row0, source_row1),
        slice(source_col0, source_col1),
        slice(destination_row0, destination_row1),
        slice(destination_col0, destination_col1),
    )


def _fast_tile_halo_layers(
    catalog: dict[str, FastTileArtifacts],
    *,
    target_transform,
    target_shape: tuple[int, int],
    target_crs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only tiles intersecting one halo and release each source immediately."""
    probability = np.zeros(target_shape, dtype=np.uint8)
    surface = np.zeros(target_shape, dtype=np.uint8)
    centerline = np.zeros(target_shape, dtype=np.uint8)
    target_bounds = rasterio.windows.bounds(
        rasterio.windows.Window(0, 0, target_shape[1], target_shape[0]),
        target_transform,
    )
    target_geometry = box(*target_bounds)
    for artifact in catalog.values():
        if not target_geometry.intersects(box(*artifact.bounds)):
            continue
        placement = _fast_aligned_tile_slices(
            artifact.shape,
            source_transform=artifact.transform,
            source_crs=artifact.crs,
            target_shape=target_shape,
            target_transform=target_transform,
            target_crs=target_crs,
        )
        if placement is not None:
            source_rows, source_columns, target_rows, target_columns = placement
            source_window = rasterio.windows.Window.from_slices(
                source_rows, source_columns,
            )
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", rasterio.errors.NotGeoreferencedWarning,
                )
                with rasterio.open(artifact.probability) as dataset:
                    source_probability = dataset.read(1, window=source_window)
                with rasterio.open(artifact.surface_mask) as dataset:
                    source_surface = dataset.read(1, window=source_window)
                with rasterio.open(artifact.centerline_mask) as dataset:
                    source_centerline = dataset.read(1, window=source_window)
            with rasterio.open(artifact.image) as dataset:
                valid = dataset.dataset_mask(window=source_window) > 0
            probability[target_rows, target_columns] = np.where(
                valid, source_probability, 0,
            )
            surface[target_rows, target_columns] = np.where(
                valid, source_surface, 0,
            )
            centerline[target_rows, target_columns] = np.where(
                valid, source_centerline, 0,
            )
            del source_probability, source_surface, source_centerline, valid
            continue
        source_probability = cv2.imread(
            str(artifact.probability), cv2.IMREAD_GRAYSCALE,
        )
        source_surface = cv2.imread(
            str(artifact.surface_mask), cv2.IMREAD_GRAYSCALE,
        )
        source_centerline = cv2.imread(
            str(artifact.centerline_mask), cv2.IMREAD_GRAYSCALE,
        )
        if any(item is None for item in (
            source_probability, source_surface, source_centerline,
        )):
            raise FileNotFoundError(f"Cannot read Fast tile intermediates: {artifact.stem}")
        if any(item.shape != artifact.shape for item in (
            source_probability, source_surface, source_centerline,
        )):
            raise ValueError(f"Fast tile intermediate shape mismatch: {artifact.stem}")
        with rasterio.open(artifact.image) as dataset:
            valid = dataset.dataset_mask() > 0
        source_probability[~valid] = 0
        source_surface[~valid] = 0
        source_centerline[~valid] = 0
        for destination, source, resampling in (
            (probability, source_probability, Resampling.bilinear),
            (surface, source_surface, Resampling.nearest),
            (centerline, source_centerline, Resampling.nearest),
        ):
            _fast_place_tile_array(
                destination,
                source,
                source_transform=artifact.transform,
                source_crs=artifact.crs,
                target_transform=target_transform,
                target_crs=target_crs,
                resampling=resampling,
            )
        del source_probability, source_surface, source_centerline, valid
    return probability, (surface > 0).astype(np.uint8), (
        centerline > 0
    ).astype(np.uint8)


def _fast_tile_topology_frame(
    catalog: dict[str, FastTileArtifacts],
    *,
    target_bounds: tuple[float, float, float, float],
    target_crs,
) -> gpd.GeoDataFrame:
    """Load topology edges only for tiles crossing the current halo."""
    target_geometry = box(*target_bounds)
    records = []
    for artifact in catalog.values():
        if artifact.crs != target_crs:
            raise ValueError("Fast change tiles must use one normalized CRS")
        if not target_geometry.intersects(box(*artifact.bounds)):
            continue
        with np.load(artifact.topology, allow_pickle=False) as topology:
            nodes = np.asarray(topology["nodes"], dtype=np.float32).reshape(-1, 2)
            edges = _unique_topology_edges(nodes, topology["edges"])
        for edge_id, (source, target) in enumerate(edges.tolist()):
            line = _world_line(artifact.transform, nodes[source], nodes[target])
            if not line.intersects(target_geometry):
                continue
            clipped = make_valid(line.intersection(target_geometry))
            for part in _fast_line_parts(clipped):
                if not part.is_empty and float(part.length) > 0:
                    records.append({
                        "tile": artifact.stem,
                        "edge_id": int(edge_id),
                        "geometry": part,
                    })
    if records:
        return gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
    return gpd.GeoDataFrame(
        {"tile": [], "edge_id": []},
        geometry=gpd.GeoSeries([], crs=target_crs),
        crs=target_crs,
    )


def _clean_fast_presence_mask(
    mask: np.ndarray,
    centerline_anchor: np.ndarray,
    road_support: np.ndarray,
) -> np.ndarray:
    binary = _bridge_fast_presence_gaps(mask, road_support)
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


def _bridge_fast_presence_gaps(
    mask: np.ndarray,
    road_support: np.ndarray,
) -> np.ndarray:
    """Fill only straight 1-2 px gaps whose missing pixels remain on road support."""
    source = (np.asarray(mask) > 0)
    support = (np.asarray(road_support) > 0)
    bridged = source.copy()
    for gap_size in (1, 2):
        span = gap_size + 1
        horizontal = source[:, :-span] & source[:, span:]
        for offset in range(1, span):
            horizontal &= support[:, offset:offset - span]
        for offset in range(1, span):
            target = bridged[:, offset:offset - span]
            target[horizontal] = True

        vertical = source[:-span, :] & source[span:, :]
        for offset in range(1, span):
            vertical &= support[offset:offset - span, :]
        for offset in range(1, span):
            target = bridged[offset:offset - span, :]
            target[vertical] = True
    return bridged.astype(np.uint8)


def _partition_fast_presence_components(
    mask: np.ndarray,
    path_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep normal mask components whole; split only large, spatially distinct roads."""
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
        seed_mask = component & (path_labels[y:y + height, x:x + width] > 0)
        grouped_seed_support = cv2.dilate(
            seed_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ).astype(bool) & component
        seed_group_count, seed_groups = cv2.connectedComponents(
            grouped_seed_support.astype(np.uint8), connectivity=8,
        )
        seed_counts = np.bincount(
            seed_groups[seed_mask], minlength=seed_group_count,
        )
        valid_groups = np.flatnonzero(
            seed_counts >= FAST_PRESENCE_MIN_PATH_SEED_PIXELS
        )
        valid_groups = valid_groups[valid_groups > 0]
        component_area = int(stats[component_id, cv2.CC_STAT_AREA])
        if (
            component_area < FAST_PRESENCE_ABNORMAL_COMPONENT_AREA_PX2
            or valid_groups.size <= 1
        ):
            region_crop = regions[y:y + height, x:x + width]
            region_crop[component] = next_region
            next_region += 1
            continue

        valid_seed_mask = seed_mask & np.isin(seed_groups, valid_groups)
        distance_source = np.ones(component.shape, dtype=np.uint8)
        distance_source[valid_seed_mask] = 0
        _distance, nearest_seed = cv2.distanceTransformWithLabels(
            distance_source,
            cv2.DIST_L2,
            cv2.DIST_MASK_5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        seed_owners = seed_groups[valid_seed_mask]
        owner_by_label = np.zeros(seed_owners.size + 1, dtype=np.int32)
        owner_by_label[1:] = seed_owners
        nearest_group = owner_by_label[nearest_seed]
        region_crop = regions[y:y + height, x:x + width]
        for group_id in valid_groups:
            owned = component & (nearest_group == group_id)
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
    grid: FastProbabilityGrid,
    *,
    change_type: str,
    before_period: str,
    after_period: str,
    min_area: float,
) -> list[dict]:
    records = []
    for mapping, value in shapes(
        regions.astype(np.int32), mask=regions.astype(bool), transform=grid.transform,
    ):
        if int(value) <= 0:
            continue
        change_geometry = make_valid(shape(mapping))
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
    before_surface_mask: np.ndarray,
    after_surface_mask: np.ndarray,
    before_line_anchor: np.ndarray,
    after_line_anchor: np.ndarray,
    before_path_labels: np.ndarray,
    after_path_labels: np.ndarray,
    *,
    before_period: str,
    after_period: str,
    min_area: float,
) -> tuple[dict[str, list[dict]], dict]:
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
    added_mask = _clean_fast_presence_mask(
        added_raw & (after_surface_mask > 0),
        after_line_anchor,
        after_surface_mask,
    )
    removed_mask = _clean_fast_presence_mask(
        removed_raw & (before_surface_mask > 0),
        before_line_anchor,
        before_surface_mask,
    )
    added_regions, added_split = _partition_fast_presence_components(
        added_mask, after_path_labels,
    )
    removed_regions, removed_split = _partition_fast_presence_components(
        removed_mask, before_path_labels,
    )
    records = {
        "added": _fast_presence_records(
            added_regions, grid, change_type="added",
            before_period=before_period, after_period=after_period,
            min_area=min_area,
        ),
        "removed": _fast_presence_records(
            removed_regions, grid, change_type="removed",
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
    return records, diagnostics


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
    sample_spacing: float = FAST_PAIRED_WIDTH_SAMPLE_SPACING_M,
) -> float:
    distances = _fast_sample_distances(source, sample_spacing)
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
    sample_spacing: float = FAST_PAIRED_WIDTH_SAMPLE_SPACING_M,
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
                source_line,
                target_line,
                tolerance=tolerance,
                sample_spacing=sample_spacing,
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
    sample_spacing: float = FAST_PAIRED_WIDTH_SAMPLE_SPACING_M,
) -> list[tuple[int, int]]:
    before_to_after = _fast_nearest_centerline_matches(
        before, after, tolerance=tolerance, sample_spacing=sample_spacing,
    )
    after_to_before = _fast_nearest_centerline_matches(
        after, before, tolerance=tolerance, sample_spacing=sample_spacing,
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
    physical_pixel_size: float,
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
    map_step_per_pixel = max(float(physical_pixel_size), 1e-9)
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
    sample_spacing: float = FAST_PAIRED_WIDTH_SAMPLE_SPACING_M,
) -> list[dict]:
    samples = []
    for distance in _fast_sample_distances(
        before_line, sample_spacing,
    ):
        before_point = before_line.interpolate(float(distance))
        after_distance = float(after_line.project(before_point))
        after_point = after_line.interpolate(after_distance)
        if before_point.distance(after_point) > tolerance:
            continue
        normal = _fast_local_normal(
            before_line, float(distance), sample_spacing,
        )
        if normal is None:
            continue
        common_xy = (
            (float(before_point.x) + float(after_point.x)) / 2.0,
            (float(before_point.y) + float(after_point.y)) / 2.0,
        )
        before_width = _fast_mask_width_at_position(
            before_surface_mask,
            grid.transform,
            common_xy,
            normal,
            grid.pixel_size,
        )
        after_width = _fast_mask_width_at_position(
            after_surface_mask,
            grid.transform,
            common_xy,
            normal,
            grid.pixel_size,
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


def _fast_map_units_per_meter(grid: FastProbabilityGrid) -> float:
    x_step = float(np.hypot(grid.transform.a, grid.transform.d))
    y_step = float(np.hypot(grid.transform.b, grid.transform.e))
    map_steps = [value for value in (x_step, y_step) if value > 1e-12]
    map_units_per_pixel = float(np.mean(map_steps)) if map_steps else 1.0
    return map_units_per_pixel / max(float(grid.pixel_size), 1e-9)


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
    map_units_per_meter = _fast_map_units_per_meter(grid)
    map_tolerance = float(tolerance) * map_units_per_meter
    sample_spacing = FAST_PAIRED_WIDTH_SAMPLE_SPACING_M * map_units_per_meter
    matches = _fast_mutual_centerline_matches(
        before_centerlines,
        after_centerlines,
        tolerance=map_tolerance,
        sample_spacing=sample_spacing,
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
            tolerance=map_tolerance,
            sample_spacing=sample_spacing,
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
        change_geometry = common_axis.buffer(
            outer_width * map_units_per_meter / 2.0,
        ).difference(
            common_axis.buffer(inner_width * map_units_per_meter / 2.0)
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


def _clip_fast_tile_records_to_core(
    records: list[dict],
    core_geometry,
    *,
    tile_stem: str,
) -> list[dict]:
    """Keep halo context for detection but publish pixels only from tile core."""
    clipped_records = []
    for source_record in records:
        geometry = source_record.get("geometry")
        if geometry is None or geometry.is_empty or not geometry.intersects(core_geometry):
            continue
        clipped = make_valid(geometry.intersection(core_geometry))
        for part in _fast_polygon_parts(clipped, min_area=0.0):
            record = dict(source_record)
            record["geometry"] = part
            record["_tile_stem"] = str(tile_stem)
            clipped_records.append(record)
    return clipped_records


def _merge_fast_tile_change_records(
    records: list[dict],
    *,
    min_area: float,
) -> list[dict]:
    """Join same-type pieces that meet across different tile core boundaries."""
    if not records:
        return []
    frame = gpd.GeoDataFrame(records, geometry="geometry")
    parent = list(range(len(frame)))

    def find(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    spatial_index = frame.sindex
    for position, row in frame.iterrows():
        candidates = spatial_index.query(row.geometry, predicate="intersects")
        for candidate in np.asarray(candidates, dtype=int).reshape(-1).tolist():
            if candidate <= position:
                continue
            if str(row.get("change_typ")) != str(frame.iloc[candidate].get("change_typ")):
                continue
            if str(row.get("_tile_stem")) == str(frame.iloc[candidate].get("_tile_stem")):
                continue
            union(int(position), int(candidate))

    groups: dict[int, list[int]] = {}
    for position in range(len(frame)):
        groups.setdefault(find(position), []).append(position)
    merged_records = []
    for positions in groups.values():
        members = frame.iloc[positions]
        geometry = make_valid(members.geometry.union_all())
        record = dict(members.iloc[0])
        record.pop("_tile_stem", None)
        for field in ("width_bef", "width_aft", "width_diff"):
            values = np.asarray(members.get(field, []), dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                record[field] = float(np.median(values))
        for part in _fast_polygon_parts(geometry, min_area=min_area):
            merged = dict(record)
            merged["geometry"] = part
            merged_records.append(merged)
    return merged_records


def _sum_fast_change_diagnostics(items: list[dict]) -> dict:
    keys = {
        key
        for item in items
        for key, value in item.items()
        if isinstance(value, (int, np.integer))
    }
    return {
        key: int(sum(int(item.get(key, 0)) for item in items))
        for key in sorted(keys)
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
    internal_outputs: bool = False,
) -> dict:
    """Detect no-truth Fast changes tile by tile from extraction intermediates."""
    total_started = time.perf_counter()
    before_payload = _load_fast_period_result(before_result)
    after_payload = _load_fast_period_result(after_result)
    before_tiles = _fast_tile_catalog(before_payload)
    after_tiles = _fast_tile_catalog(after_payload)
    before_stems, after_stems = set(before_tiles), set(after_tiles)
    if before_stems != after_stems:
        raise ValueError(
            "Fast change requires matching normalized tile sets; "
            f"before-only={sorted(before_stems - after_stems)}, "
            f"after-only={sorted(after_stems - before_stems)}"
        )
    tile_stems = sorted(before_stems)
    tolerance = max(0.0, float(position_tolerance))
    minimum_area = max(0.0, float(min_change_area))
    _ = min_change_length  # Accepted for the stable external API; no longer a detector rule.
    record_groups = {
        "added": [], "removed": [], "widened": [], "narrowed": [],
        "width_changed": [],
    }
    presence_diagnostic_rows = []
    width_diagnostic_rows = []
    tile_timings = []
    presence_change_seconds = 0.0
    width_change_seconds = 0.0
    maximum_processing_shape = (0, 0)
    target_crs = before_tiles[tile_stems[0]].crs
    pixel_sizes = []
    map_unit_scales = []
    for tile_stem in tile_stems:
        tile_started = time.perf_counter()
        before_tile = before_tiles[tile_stem]
        after_tile = after_tiles[tile_stem]
        if before_tile.crs != target_crs or after_tile.crs != target_crs:
            raise ValueError("Fast change tiles must use one normalized CRS")
        pixel_size = float(before_tile.pixel_size)
        if not np.isclose(
            pixel_size, float(after_tile.pixel_size), rtol=0.05, atol=1e-6,
        ):
            raise ValueError(
                f"Fast tile physical pixel size mismatch: {tile_stem}"
            )
        pixel_sizes.append(pixel_size)
        halo_px = _fast_change_halo_pixels(pixel_size, tolerance)
        core_height, core_width = before_tile.shape
        halo_window = rasterio.windows.Window(
            -halo_px, -halo_px,
            core_width + 2 * halo_px,
            core_height + 2 * halo_px,
        )
        local_transform = rasterio.windows.transform(
            halo_window, before_tile.transform,
        )
        local_shape = (int(halo_window.height), int(halo_window.width))
        maximum_processing_shape = max(
            maximum_processing_shape, local_shape,
            key=lambda shape_2d: int(shape_2d[0] * shape_2d[1]),
        )
        local_bounds = tuple(rasterio.windows.bounds(
            rasterio.windows.Window(0, 0, local_shape[1], local_shape[0]),
            local_transform,
        ))
        core_geometry = box(*before_tile.bounds)

        before_probability, before_surface_mask, before_centerline_mask = (
            _fast_tile_halo_layers(
                before_tiles,
                target_transform=local_transform,
                target_shape=local_shape,
                target_crs=target_crs,
            )
        )
        after_probability, after_surface_mask, after_centerline_mask = (
            _fast_tile_halo_layers(
                after_tiles,
                target_transform=local_transform,
                target_shape=local_shape,
                target_crs=target_crs,
            )
        )
        grid = FastProbabilityGrid(
            _probability01(before_probability),
            _probability01(after_probability),
            local_transform,
            target_crs,
            pixel_size,
        )
        map_units_per_meter = _fast_map_units_per_meter(grid)
        map_unit_scales.append(map_units_per_meter)
        minimum_area_map = minimum_area * map_units_per_meter ** 2
        del before_probability, after_probability
        before_centerlines = _fast_tile_topology_frame(
            before_tiles, target_bounds=local_bounds, target_crs=target_crs,
        )
        after_centerlines = _fast_tile_topology_frame(
            after_tiles, target_bounds=local_bounds, target_crs=target_crs,
        )

        presence_started = time.perf_counter()
        presence_records, presence_diagnostics = _detect_probability_presence_changes(
            grid,
            before_surface_mask,
            after_surface_mask,
            before_centerline_mask,
            after_centerline_mask,
            before_centerline_mask,
            after_centerline_mask,
                before_period=before_period,
                after_period=after_period,
                min_area=minimum_area_map,
        )
        tile_presence_seconds = time.perf_counter() - presence_started
        presence_change_seconds += tile_presence_seconds
        presence_diagnostic_rows.append(presence_diagnostics)
        for change_type in ("added", "removed"):
            record_groups[change_type].extend(_clip_fast_tile_records_to_core(
                presence_records[change_type],
                core_geometry,
                tile_stem=tile_stem,
            ))

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
            min_area=minimum_area_map,
        )
        tile_width_seconds = time.perf_counter() - width_started
        width_change_seconds += tile_width_seconds
        width_diagnostic_rows.append(width_diagnostics)
        for change_type in ("widened", "narrowed"):
            record_groups[change_type].extend(_clip_fast_tile_records_to_core(
                [
                    record for record in width_records
                    if record["change_typ"] == change_type
                ],
                core_geometry,
                tile_stem=tile_stem,
            ))
        tile_seconds = time.perf_counter() - tile_started
        tile_timings.append({
            "tile": tile_stem,
            "halo_px": int(halo_px),
            "processing_shape": [int(local_shape[0]), int(local_shape[1])],
            "presence_seconds": float(tile_presence_seconds),
            "width_seconds": float(tile_width_seconds),
            "total_seconds": float(tile_seconds),
        })
        print(
            f"[Fast Change Tile] {tile_stem}: "
            f"presence={tile_presence_seconds:.3f}s, "
            f"width={tile_width_seconds:.3f}s, "
            f"total={tile_seconds:.3f}s",
            flush=True,
        )
        del (
            grid,
            before_surface_mask, after_surface_mask,
            before_centerline_mask, after_centerline_mask,
            before_centerlines, after_centerlines,
            presence_records, width_records,
        )

    merge_started = time.perf_counter()
    minimum_area_map = minimum_area * float(np.mean(map_unit_scales)) ** 2
    for change_type in ("added", "removed", "widened", "narrowed"):
        record_groups[change_type] = _merge_fast_tile_change_records(
            record_groups[change_type], min_area=minimum_area_map,
        )
    tile_merge_seconds = time.perf_counter() - merge_started
    presence_diagnostics = _sum_fast_change_diagnostics(
        presence_diagnostic_rows,
    )
    width_diagnostics = _sum_fast_change_diagnostics(width_diagnostic_rows)
    width_records = record_groups["widened"] + record_groups["narrowed"]
    grid_crs = target_crs
    grid_pixel_size = float(np.mean(pixel_sizes))
    columns = {
        "change_typ": [], "before_per": [], "after_per": [], "source": [],
        "width_bef": [], "width_aft": [], "width_diff": [],
    }

    def frame_from(records: list[dict]) -> gpd.GeoDataFrame:
        if records:
            return gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
        return gpd.GeoDataFrame(
            columns.copy(), geometry=gpd.GeoSeries([], crs=target_crs),
            crs=target_crs,
        )

    typed_layers = {name: frame_from(records) for name, records in record_groups.items()}
    all_records = [
        record for name in ("added", "removed", "widened", "narrowed")
        for record in record_groups[name]
    ]
    changes = frame_from(all_records)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    if internal_outputs:
        layers = {"changes": changes, **typed_layers}
        filenames = {
            "changes": "road_changes.shp", "added": "added_roads.shp",
            "removed": "removed_roads.shp",
            "width_changed": "width_changed_road_parts.shp",
            "widened": "widened_road_parts.shp",
            "narrowed": "narrowed_road_parts.shp",
        }
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
    else:
        _public_changes, public_path = _write_fast_public_changes(changes, output_dir)
        layers = {"changes": changes}
        output_layers = {"changes": str(public_path.resolve())}
        gpkg = None

    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from road_change_detection import render_change_preview  # noqa: PLC0415

    preview_path = output_dir / "change_preview.png"
    render_change_preview(
        preview_path, changes, _empty_like(changes),
        title=_fast_change_preview_title(before_period, after_period),
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
        "probability_grid_crs": str(grid_crs),
        "probability_pixel_size": float(grid_pixel_size),
        "auto_change_processing_mode": "per_tile_intermediates",
        "regional_probability_mosaic_used": False,
        "tile_count": int(len(tile_stems)),
        "tile_timings": tile_timings,
        "maximum_processing_shape": [
            int(maximum_processing_shape[0]), int(maximum_processing_shape[1]),
        ],
        "maximum_processing_pixel_count": int(
            maximum_processing_shape[0] * maximum_processing_shape[1]
        ),
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
        "tile_merge_seconds": float(tile_merge_seconds),
        "write_seconds": float(write_seconds),
        "auto_change_total_seconds": float(total_seconds),
        "total_seconds": float(total_seconds),
        **presence_diagnostics,
        **width_diagnostics,
        **{f"{name}_feature_count": int(len(frame)) for name, frame in typed_layers.items()},
        "changes_feature_count": int(len(changes)),
    }
    summary_path = output_dir / "change_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Fast Change] {before_period}->{after_period}: "
        f"presence={presence_change_seconds:.3f}s, "
        f"width={width_change_seconds:.3f}s, "
        f"merge={tile_merge_seconds:.3f}s, "
        f"write={write_seconds:.3f}s, total={total_seconds:.3f}s, "
        f"added={len(record_groups['added'])}, removed={len(record_groups['removed'])}, "
        f"width_pairs={width_diagnostics['matched_centerline_pair_count']}, "
        f"width_changes={len(width_records)}",
        flush=True,
    )
    result = {
        "output": str(output_dir), "summary": str(summary_path.resolve()),
        "road_changes": output_layers["changes"],
        "layers": output_layers,
        "previews": {"change": str(preview_path.resolve())},
        "road_change": str(preview_path.resolve()),
        **summary,
    }
    if gpkg is not None:
        result["gpkg"] = str(gpkg.resolve())
    return result


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
