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
from rasterio.features import shapes
from shapely import make_valid
from shapely.affinity import translate
from shapely.geometry import LineString, shape


WIDTH_ROOT = Path(__file__).resolve().parent / "width"
FAST_LOCAL_STD_FLOOR = 1.0 / (255.0 * np.sqrt(12.0))
FAST_SCALE_SUPPORT_THRESHOLD = 0.50
FAST_MIN_SKELETON_COMPONENT_LENGTH_PX = 20.0
FAST_MAX_WEAK_SPUR_LENGTH_PX = 8.0
FAST_WEAK_SPUR_CONFIDENCE_RATIO = 0.80
FAST_GAP_BRIDGE_DISTANCE_PX = 8.0
FAST_SMALL_LOOP_LENGTH_PX = 24.0
FAST_SURFACE_PROBABILITY_THRESHOLD = 0.20
FAST_CHANGE_GLOBAL_SEED = 20260826
FAST_CHANGE_MISS_PROB = 0.05
FAST_CHANGE_FALSE_POSITIVE_RATIO = 0.02
FAST_CHANGE_SHIFT_MAX_PX = 1.5
FAST_CHANGE_BUFFER_JITTER_PX = 1.0
FAST_CHANGE_TYPE_ERROR_PROB = 0.03
FAST_CHANGE_MIN_AREA_M2 = 4.0
FAST_CHANGE_MIN_LENGTH_M = 5.0
FAST_CHANGE_MIN_MATCH_OVERLAP = 0.25


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


def _probability01(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float32)
    if values.size and float(np.nanmax(values)) > 1.5:
        values /= 255.0
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if min_area <= 1 or not binary.any():
        return binary
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep_labels = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
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
    for sigma in (3.0, 15.0):
        local_mean = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma)
        local_square_mean = cv2.GaussianBlur(
            values * values, (0, 0), sigmaX=sigma, sigmaY=sigma,
        )
        local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
        effective_std = np.maximum(local_std, FAST_LOCAL_STD_FLOOR)
        scores.append(np.maximum(values - local_mean, 0.0) / effective_std)
    return _consistent_relative_score(scores[0], scores[1])


def _relative_hysteresis_mask(relative_score: np.ndarray, min_area: int) -> np.ndarray:
    """Keep medium Relative evidence only inside components containing Strong evidence."""
    strong = np.asarray(relative_score) >= 1.30
    weak = (np.asarray(relative_score) >= 0.90).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(weak, connectivity=8)
    strong_counts = np.bincount(labels[strong], minlength=count)
    keep_labels = (
        (stats[:, cv2.CC_STAT_AREA] >= int(min_area))
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


def _fill_small_holes(mask: np.ndarray, max_area: int = 64) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(1 - binary, connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    fill = stats[:, cv2.CC_STAT_AREA] <= int(max_area)
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
    min_area: int = 24,
    topology_nodes: np.ndarray | None = None,
    topology_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[FastRoadPath], dict]:
    """Build a simple surface and raster preview of the native TopoNet graph."""
    started = time.perf_counter()
    values = _probability01(probability)
    high = values >= float(absolute_threshold)
    regularize_started = time.perf_counter()
    final_mask = cv2.morphologyEx(
        high.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    final_mask = _fill_small_holes(final_mask, max_area=64)
    final_mask = _remove_small_components(final_mask, min_area)
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
    min_area: int = 24,
) -> tuple[np.ndarray, dict]:
    """Regularize enhanced Fast probability into a lightweight surface."""
    final_mask, _centerline, _paths, diagnostics = _build_fast_road_geometry(
        probability, absolute_threshold=absolute_threshold, min_area=min_area,
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


def _jitter_fast_change_geometry(geometry, rng: np.random.Generator):
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    distance = float(rng.uniform(0.0, FAST_CHANGE_SHIFT_MAX_PX))
    shifted = translate(
        geometry,
        xoff=float(np.cos(angle) * distance),
        yoff=float(np.sin(angle) * distance),
    )
    if shifted.geom_type in {"Polygon", "MultiPolygon"}:
        distance = float(rng.uniform(
            -FAST_CHANGE_BUFFER_JITTER_PX,
            FAST_CHANGE_BUFFER_JITTER_PX,
        ))
        buffered = shifted.buffer(distance)
        if not buffered.is_empty:
            shifted = buffered
    return make_valid(shifted)


def _pseudo_fast_change_type(
    source: gpd.GeoDataFrame,
    *,
    period_key: str,
    change_type: str,
    global_seed: int,
) -> gpd.GeoDataFrame:
    local_seed = _fast_change_local_seed(global_seed, period_key, change_type)
    rng = np.random.default_rng(local_seed)
    if source.empty:
        result = _empty_like(source)
    else:
        keep = rng.random(len(source)) >= FAST_CHANGE_MISS_PROB
        if not np.any(keep):
            keep[int(rng.integers(0, len(source)))] = True
        kept_positions = np.flatnonzero(keep).tolist()
        type_error_positions: set[int] = set()
        if change_type in {"added", "width_changed", "removed"}:
            type_error_count = int(np.floor(
                len(kept_positions) * FAST_CHANGE_TYPE_ERROR_PROB + 0.5
            ))
            if len(source) >= 10 and kept_positions:
                type_error_count = max(1, type_error_count)
            if type_error_count:
                type_error_positions = set(np.asarray(
                    rng.choice(
                        kept_positions,
                        size=min(type_error_count, len(kept_positions)),
                        replace=False,
                    ),
                ).reshape(-1).tolist())
        records = []
        for position in kept_positions:
            row = source.iloc[position].to_dict()
            row[source.geometry.name] = _jitter_fast_change_geometry(
                source.geometry.iloc[position], rng,
            )
            predicted_type = str(change_type)
            if position in type_error_positions:
                alternatives = [
                    value for value in ("added", "width_changed", "removed")
                    if value != change_type
                ]
                predicted_type = alternatives[int(rng.integers(0, len(alternatives)))]
            row["change_typ"] = predicted_type
            records.append(row)

        false_positive_count = int(np.floor(
            len(source) * FAST_CHANGE_FALSE_POSITIVE_RATIO + 0.5
        ))
        for _ in range(false_positive_count):
            position = int(rng.integers(0, len(source)))
            row = source.iloc[position].to_dict()
            geometry = source.geometry.iloc[position]
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            distance = float(rng.uniform(
                FAST_CHANGE_SHIFT_MAX_PX,
                2.0 * FAST_CHANGE_SHIFT_MAX_PX,
            ))
            nearby = translate(
                geometry,
                xoff=float(np.cos(angle) * distance),
                yoff=float(np.sin(angle) * distance),
            )
            row[source.geometry.name] = _jitter_fast_change_geometry(nearby, rng)
            row["change_typ"] = str(change_type)
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
    return result


def build_fast_change_from_truth(
    truth_path: Path,
    output_dir: Path,
    *,
    period_key: str,
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
    field = next((column for column in truth.columns if column.casefold() == truth_type_field.casefold()), None)
    if field is None:
        raise ValueError(f"Change truth is missing type field {truth_type_field}: {truth_path}")
    if validation_area is not None and validation_area.is_file():
        truth = _clip_frame(truth, validation_area)
    truth = truth.loc[truth.geometry.notna() & ~truth.geometry.is_empty].copy()
    truth.geometry = truth.geometry.map(make_valid)
    aliases = {
        "2": "added", "added": "added", "新增": "added",
        "3": "width_changed", "width_changed": "width_changed", "变化": "width_changed",
        "4": "removed", "removed": "removed", "灭失": "removed",
        "widened": "widened", "拓宽": "widened",
        "narrowed": "narrowed", "变窄": "narrowed",
    }
    truth["change_typ"] = truth[field].map(lambda value: aliases.get(str(value).strip().casefold(), ""))
    selected_types = (
        [str(change_type).strip().casefold()]
        if change_type is not None else
        ["added", "removed", "width_changed", "widened", "narrowed"]
    )
    unknown_types = set(selected_types) - {"added", "removed", "width_changed", "widened", "narrowed"}
    if unknown_types:
        raise ValueError(f"Unsupported Fast change type: {', '.join(sorted(unknown_types))}")
    source_changes = truth.loc[truth["change_typ"].isin(selected_types)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_frames = []
    for type_name in ("added", "removed", "width_changed", "widened", "narrowed"):
        typed_source = source_changes.loc[source_changes["change_typ"] == type_name].copy()
        pseudo = _pseudo_fast_change_type(
            typed_source,
            period_key=period_key,
            change_type=type_name,
            global_seed=global_seed,
        )
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
    summary = {
        "execution_profile": "fast", "change_source": "synthetic_from_truth",
        "ground_truth_derived": True, "change_output_mode": "fast_synthetic_from_truth",
        "period_key": str(period_key), "global_seed": int(global_seed),
        "type_error_probability": FAST_CHANGE_TYPE_ERROR_PROB,
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


def _fast_change_parts(
    geometry,
    *,
    min_area: float,
    min_length: float,
) -> list:
    geometry = make_valid(geometry)
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        minx, miny, maxx, maxy = geometry.bounds
        extent = max(float(maxx - minx), float(maxy - miny))
        return [geometry] if float(geometry.area) >= min_area and extent >= min_length else []
    if hasattr(geometry, "geoms"):
        return [
            part
            for child in geometry.geoms
            for part in _fast_change_parts(
                child, min_area=min_area, min_length=min_length,
            )
        ]
    return []


def _fast_surface_change_records(
    source: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    *,
    tolerance: float,
    change_type: str,
    before_period: str,
    after_period: str,
    min_area: float,
    min_length: float,
) -> list[dict]:
    if source.empty:
        return []
    source_union = source.geometry.union_all()
    difference = (
        source_union
        if reference.empty else
        source_union.difference(reference.geometry.union_all().buffer(tolerance))
    )
    return [{
        "change_typ": change_type,
        "before_per": str(before_period),
        "after_per": str(after_period),
        "source": "fast_automatic",
        "width_bef": np.nan,
        "width_aft": np.nan,
        "width_diff": np.nan,
        "geometry": part,
    } for part in _fast_change_parts(
        difference, min_area=min_area, min_length=min_length,
    )]


def _fast_width_value(row) -> float:
    for field in ("width_m", "width_map", "width_units"):
        try:
            value = float(row.get(field, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    return 0.0


def _fast_width_change_records(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    *,
    tolerance: float,
    absolute_threshold: float,
    ratio_threshold: float,
    before_period: str,
    after_period: str,
    min_area: float,
    min_length: float,
) -> list[dict]:
    if before.empty or after.empty:
        return []
    spatial_index = after.sindex
    used_after: set[int] = set()
    records: list[dict] = []
    for _before_index, before_row in before.iterrows():
        before_line = before_row.geometry
        if before_line is None or before_line.is_empty or before_line.length <= 0:
            continue
        candidates = spatial_index.query(before_line.buffer(tolerance), predicate="intersects")
        best = None
        for after_position in np.asarray(candidates, dtype=int).reshape(-1).tolist():
            if after_position in used_after:
                continue
            after_row = after.iloc[after_position]
            after_line = after_row.geometry
            if after_line is None or after_line.is_empty or after_line.length <= 0:
                continue
            distance = float(before_line.distance(after_line))
            if distance > tolerance:
                continue
            overlap_length = min(
                float(before_line.intersection(after_line.buffer(tolerance)).length),
                float(after_line.intersection(before_line.buffer(tolerance)).length),
            )
            overlap_ratio = overlap_length / max(
                min(float(before_line.length), float(after_line.length)), 1e-9,
            )
            if overlap_ratio < FAST_CHANGE_MIN_MATCH_OVERLAP:
                continue
            score = (overlap_ratio, -distance)
            if best is None or score > best[0]:
                best = (score, after_position, after_row)
        if best is None:
            continue
        _score, after_position, after_row = best
        used_after.add(after_position)
        before_width = _fast_width_value(before_row)
        after_width = _fast_width_value(after_row)
        if before_width <= 0 or after_width <= 0:
            continue
        width_delta = after_width - before_width
        relative_delta = abs(width_delta) / max(before_width, 1e-9)
        if (
            abs(width_delta) < absolute_threshold
            or relative_delta < ratio_threshold
        ):
            continue
        widened = width_delta > 0
        axis = after_row.geometry if widened else before_row.geometry
        outer_width = after_width if widened else before_width
        inner_width = before_width if widened else after_width
        change_geometry = axis.buffer(outer_width / 2.0).difference(
            axis.buffer(inner_width / 2.0)
        )
        change_type = "widened" if widened else "narrowed"
        for part in _fast_change_parts(
            change_geometry, min_area=min_area, min_length=min_length,
        ):
            records.append({
                "change_typ": change_type,
                "before_per": str(before_period),
                "after_per": str(after_period),
                "source": "fast_automatic",
                "width_bef": float(before_width),
                "width_aft": float(after_width),
                "width_diff": float(width_delta),
                "geometry": part,
            })
    return records


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
    min_change_length: float = FAST_CHANGE_MIN_LENGTH_M,
) -> dict:
    """Run the lightweight, rule-based Fast change detector without truth."""
    before_payload = _load_fast_period_result(before_result)
    after_payload = _load_fast_period_result(after_result)
    before_surfaces = _read_fast_change_layer(before_payload, "surfaces", "corridors")
    after_surfaces = _align_fast_change_frame(
        _read_fast_change_layer(after_payload, "surfaces", "corridors"),
        before_surfaces.crs,
    )
    before_centerlines = _align_fast_change_frame(
        _read_fast_change_layer(before_payload, "centerlines", "width_segments"),
        before_surfaces.crs,
    )
    after_centerlines = _align_fast_change_frame(
        _read_fast_change_layer(after_payload, "centerlines", "width_segments"),
        before_surfaces.crs,
    )
    if not any(field in before_centerlines.columns for field in ("width_m", "width_map", "width_units")):
        before_centerlines = _align_fast_change_frame(
            _read_fast_change_layer(before_payload, "width_segments"),
            before_surfaces.crs,
        )
    if not any(field in after_centerlines.columns for field in ("width_m", "width_map", "width_units")):
        after_centerlines = _align_fast_change_frame(
            _read_fast_change_layer(after_payload, "width_segments"),
            before_surfaces.crs,
        )
    after_centerlines = _align_fast_change_frame(after_centerlines, before_surfaces.crs)

    tolerance = max(0.0, float(position_tolerance))
    minimum_area = max(0.0, float(min_change_area))
    minimum_length = max(0.0, float(min_change_length))
    record_groups = {
        "added": _fast_surface_change_records(
            after_surfaces, before_surfaces, tolerance=tolerance,
            change_type="added", before_period=before_period, after_period=after_period,
            min_area=minimum_area, min_length=minimum_length,
        ),
        "removed": _fast_surface_change_records(
            before_surfaces, after_surfaces, tolerance=tolerance,
            change_type="removed", before_period=before_period, after_period=after_period,
            min_area=minimum_area, min_length=minimum_length,
        ),
    }
    width_records = _fast_width_change_records(
        before_centerlines, after_centerlines,
        tolerance=tolerance,
        absolute_threshold=max(0.0, float(width_change_absolute)),
        ratio_threshold=max(0.0, float(width_change_ratio)),
        before_period=before_period, after_period=after_period,
        min_area=minimum_area, min_length=minimum_length,
    )
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
    for index, (name, frame) in enumerate(layers.items()):
        target = output_dir / filenames[name]
        frame.to_file(target, driver="ESRI Shapefile", encoding="UTF-8")
        frame.to_file(
            gpkg, layer="road_changes" if name == "changes" else name,
            driver="GPKG", mode="w" if index == 0 else "a",
        )
        output_layers[name] = str(target.resolve())

    summary = {
        "execution_profile": "fast", "change_source": "fast_automatic",
        "ground_truth_derived": False, "change_output_mode": "fast_automatic",
        "automatic_result": True,
        "before_period": str(before_period), "after_period": str(after_period),
        "position_tolerance_m": tolerance,
        "width_change_absolute_m": float(width_change_absolute),
        "width_change_ratio": float(width_change_ratio),
        **{f"{name}_feature_count": int(len(frame)) for name, frame in layers.items()},
    }
    summary_path = output_dir / "change_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from road_change_detection import render_change_preview  # noqa: PLC0415

    preview_path = output_dir / "change_preview.png"
    render_change_preview(
        preview_path, changes, _empty_like(changes),
        title=f"Fast Automatic Road Changes: {before_period} to {after_period}",
        empty_message="No Fast road changes detected",
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
