from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import pickle
import sys
import time
from collections import deque
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import torch

from graph_spatial_context import (
    GraphSpatialContext,
    PointGridIndex,
    find_vertical_divided_anchor_pair as find_vertical_divided_anchor_pair_indexed,
)
from parallel_utils import resolve_worker_count, spawn_map
from topology_optimizer import candidate_path_for_graph, optimize_divided_road_junctions, read_topology_candidates


TOOL_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TOOL_DIR.parent
SAM_MOLRA_ROOT = PACKAGE_ROOT / "sam_molra"
RUNTIME_MODELS_ROOT = Path(
    os.environ.get("SAMROAD_MODELS_ROOT", TOOL_DIR.parents[2] / "runtime" / "models")
).expanduser()
DEFAULT_MOLRA_SAM = RUNTIME_MODELS_ROOT / "sam_molra" / "sam_vit_b_01ec64.pth"
DEFAULT_MOLRA_WEIGHT = RUNTIME_MODELS_ROOT / "sam_molra" / "adapter.th"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MASK_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _accumulate_profile(field: str):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            profiling = kwargs.get("profiling")
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                if profiling is not None:
                    profiling[field] = float(
                        profiling.get(field, 0.0) + time.perf_counter() - started
                    )
        return measured
    return decorate


def add_sam_molra_path() -> None:
    root = str(SAM_MOLRA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def read_rgb_for_viz(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        add_sam_molra_path()
        from infer_img import read_image

        img, _ = read_image(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as file:
        graph = pickle.load(file)

    node_to_idx: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    for node, neighbors in graph.items():
        node = tuple(int(round(v)) for v in node)
        node_to_idx.setdefault(node, len(node_to_idx))
        for neighbor in neighbors:
            neighbor = tuple(int(round(v)) for v in neighbor)
            node_to_idx.setdefault(neighbor, len(node_to_idx))
            edges.append((node_to_idx[node], node_to_idx[neighbor]))

    nodes = np.zeros((len(node_to_idx), 2), dtype=np.float32)
    for node, idx in node_to_idx.items():
        nodes[idx] = node
    unique_edges = sorted({tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]})
    return nodes, np.array(unique_edges, dtype=np.int32).reshape(-1, 2)


def save_graph(path: Path, nodes_rc: np.ndarray, edges: np.ndarray) -> None:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for src_idx, dst_idx in edges.tolist():
        src = tuple(int(round(value)) for value in nodes_rc[int(src_idx)])
        dst = tuple(int(round(value)) for value in nodes_rc[int(dst_idx)])
        if src == dst:
            continue
        graph.setdefault(src, [])
        graph.setdefault(dst, [])
        if dst not in graph[src]:
            graph[src].append(dst)
        if src not in graph[dst]:
            graph[dst].append(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(graph, file)


def compact_graph(nodes_rc: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.int32)
    unique_edges = sorted({tuple(sorted((int(src), int(dst)))) for src, dst in edges.tolist() if src != dst})
    used = sorted({node_idx for edge in unique_edges for node_idx in edge})
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used)}
    compact_edges = np.asarray([(remap[src], remap[dst]) for src, dst in unique_edges], dtype=np.int32).reshape(-1, 2)
    return nodes_rc[used].astype(np.float32), compact_edges


def sample_polyline_probability(probability: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    length = max(1, int(np.ceil(float(np.linalg.norm(end - start)))))
    rows = np.clip(np.rint(np.linspace(start[0], end[0], length + 1)).astype(np.int32), 0, probability.shape[0] - 1)
    cols = np.clip(np.rint(np.linspace(start[1], end[1], length + 1)).astype(np.int32), 0, probability.shape[1] - 1)
    return float(np.mean(probability[rows, cols]))


def prune_low_evidence_spurs(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    road_probability: np.ndarray,
    center_probability: np.ndarray,
    max_length_px: float,
    max_road_probability: float,
    max_center_probability: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if max_length_px <= 0 or edges.size == 0:
        return nodes_rc, edges, []
    degrees, node_edges = graph_degrees(edges.shape[0], nodes_rc.shape[0], edges)
    removed_edge_ids: set[int] = set()
    audit_rows: list[dict] = []
    for endpoint_idx in np.where(degrees == 1)[0].tolist():
        chain_edges: list[int] = []
        chain_nodes = [endpoint_idx]
        previous_idx = -1
        current_idx = endpoint_idx
        while True:
            available = [edge_id for edge_id in node_edges[current_idx] if edge_id not in chain_edges]
            if not available:
                break
            edge_id = available[0]
            src_idx, dst_idx = (int(value) for value in edges[edge_id])
            next_idx = dst_idx if src_idx == current_idx else src_idx
            if next_idx == previous_idx:
                break
            chain_edges.append(edge_id)
            chain_nodes.append(next_idx)
            previous_idx, current_idx = current_idx, next_idx
            if degrees[current_idx] != 2:
                break
        if not chain_edges or any(edge_id in removed_edge_ids for edge_id in chain_edges):
            continue
        length = float(sum(np.linalg.norm(nodes_rc[b] - nodes_rc[a]) for a, b in zip(chain_nodes[:-1], chain_nodes[1:])))
        if length > max_length_px:
            continue
        road_support = float(np.mean([
            sample_polyline_probability(road_probability, nodes_rc[a], nodes_rc[b])
            for a, b in zip(chain_nodes[:-1], chain_nodes[1:])
        ]))
        center_support = float(np.mean([
            sample_polyline_probability(center_probability, nodes_rc[a], nodes_rc[b])
            for a, b in zip(chain_nodes[:-1], chain_nodes[1:])
        ]))
        if road_support > max_road_probability or center_support > max_center_probability:
            continue
        removed_edge_ids.update(chain_edges)
        audit_rows.append(
            {
                "spur_id": len(audit_rows),
                "edge_ids": ";".join(str(edge_id) for edge_id in chain_edges),
                "length_px": length,
                "road_probability": road_support,
                "centerline_probability": center_support,
                "action": "auto_remove_low_evidence_spur",
            }
        )
    kept_edges = np.asarray(
        [edge for edge_id, edge in enumerate(edges.tolist()) if edge_id not in removed_edge_ids], dtype=np.int32
    ).reshape(-1, 2)
    compact_nodes, compact_edges = compact_graph(nodes_rc, kept_edges)
    return compact_nodes, compact_edges, audit_rows


def build_graph_adjacency(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
) -> list[list[tuple[int, int, float]]]:
    """Build stable neighbor/edge/length adjacency once for repeated queries."""
    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(nodes_rc.shape[0])]
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        length = float(np.linalg.norm(nodes_rc[int(dst_idx)] - nodes_rc[int(src_idx)]))
        adjacency[int(src_idx)].append((int(dst_idx), edge_id, length))
        adjacency[int(dst_idx)].append((int(src_idx), edge_id, length))
    return adjacency


def shortest_graph_path_length_excluding_edge(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    start_idx: int,
    end_idx: int,
    excluded_edge_id: int,
    max_length_px: float,
    adjacency: list[list[tuple[int, int, float]]] | None = None,
) -> tuple[float, list[int]]:
    if adjacency is None:
        adjacency = build_graph_adjacency(nodes_rc, edges)
    distances = {int(start_idx): 0.0}
    parents: dict[int, tuple[int, int]] = {}
    queue = [(0.0, int(start_idx))]
    while queue:
        distance, node_idx = heapq.heappop(queue)
        if distance > distances.get(node_idx, float("inf")) or distance > max_length_px:
            continue
        if node_idx == end_idx:
            path_edges = []
            current_idx = int(end_idx)
            while current_idx != int(start_idx):
                previous_idx, edge_id = parents[current_idx]
                path_edges.append(edge_id)
                current_idx = previous_idx
            return distance, path_edges
        for neighbor_idx, edge_id, length in adjacency[node_idx]:
            if edge_id == excluded_edge_id:
                continue
            new_distance = distance + length
            if new_distance < distances.get(neighbor_idx, float("inf")) and new_distance <= max_length_px:
                distances[neighbor_idx] = new_distance
                parents[neighbor_idx] = (node_idx, edge_id)
                heapq.heappush(queue, (new_distance, neighbor_idx))
    return float("inf"), []


def cleanup_junction_conflicts(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    topology_probabilities: np.ndarray,
    max_cycle_length_px: float = 120.0,
    min_cycle_length_px: float = 80.0,
    max_weak_probability: float = 0.75,
    min_probability_margin: float = 0.05,
    max_cycle_continuity_cosine: float = 0.60,
    fan_min_degree: int = 5,
    fan_max_edge_length_px: float = 30.0,
    fan_min_same_side_cosine: float = 0.75,
    fan_min_opposite_cosine: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Remove only locally redundant junction edges with strong geometric evidence."""
    if edges.size == 0:
        return nodes_rc, edges, []
    probabilities = np.asarray(topology_probabilities, dtype=np.float32)
    if probabilities.size != edges.shape[0]:
        probabilities = np.full(edges.shape[0], np.nan, dtype=np.float32)
    removed: set[int] = set()
    audit_rows: list[dict] = []

    # A short junction loop is usually several mutually exclusive TopoNet links.
    # Break it only when one edge is clearly weaker than every alternative.
    degrees, node_edges = graph_degrees(edges.shape[0], nodes_rc.shape[0], edges)
    adjacency = build_graph_adjacency(nodes_rc, edges)
    cycle_candidates = []
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        src_idx, dst_idx = int(src_idx), int(dst_idx)
        probability = float(probabilities[edge_id])
        if not np.isfinite(probability):
            continue
        if min(int(degrees[src_idx]), int(degrees[dst_idx])) < 3:
            continue
        edge_length = float(np.linalg.norm(nodes_rc[int(dst_idx)] - nodes_rc[int(src_idx)]))
        edge_direction = (nodes_rc[int(dst_idx)] - nodes_rc[int(src_idx)]) / max(edge_length, 1e-6)
        endpoint_continuities = []
        for node_idx, incoming_direction in ((int(src_idx), -edge_direction), (int(dst_idx), edge_direction)):
            continuities = []
            for incident_edge_id in node_edges[node_idx]:
                if incident_edge_id == edge_id:
                    continue
                edge_src, edge_dst = (int(value) for value in edges[incident_edge_id])
                neighbor_idx = edge_dst if edge_src == node_idx else edge_src
                vector = nodes_rc[neighbor_idx] - nodes_rc[node_idx]
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    continuities.append(float(np.dot(vector / norm, incoming_direction)))
            endpoint_continuities.append(max(continuities, default=-1.0))
        continuity = min(endpoint_continuities)
        if continuity > max_cycle_continuity_cosine:
            continue
        alternative, alternative_edge_ids = shortest_graph_path_length_excluding_edge(
            nodes_rc, edges, src_idx, dst_idx, edge_id,
            max_cycle_length_px - edge_length, adjacency=adjacency,
        )
        cycle_length = edge_length + alternative
        if not np.isfinite(alternative) or not min_cycle_length_px <= cycle_length <= max_cycle_length_px:
            continue
        cycle_candidates.append(
            (probability, edge_id, edge_length, cycle_length, int(src_idx), int(dst_idx), alternative_edge_ids, continuity)
        )
    for weakest_probability, weakest_edge_id, edge_length, cycle_length, src_idx, dst_idx, alternative_edge_ids, continuity in sorted(cycle_candidates):
        if weakest_edge_id in removed or any(edge_id in removed for edge_id in alternative_edge_ids):
            continue
        if weakest_probability <= max_weak_probability:
            removed.add(weakest_edge_id)
            audit_rows.append(
                {
                    "cleanup_id": len(audit_rows),
                    "action": "remove_weak_short_cycle_edge",
                    "edge_id": weakest_edge_id,
                    "src_row": float(nodes_rc[src_idx, 0]),
                    "src_col": float(nodes_rc[src_idx, 1]),
                    "dst_row": float(nodes_rc[dst_idx, 0]),
                    "dst_col": float(nodes_rc[dst_idx, 1]),
                    "edge_length_px": edge_length,
                    "cycle_length_px": cycle_length,
                    "topology_probability": weakest_probability,
                    "probability_margin": max_cycle_continuity_cosine - continuity,
                }
            )

    remaining_edges = np.asarray(
        [edge for edge_id, edge in enumerate(edges.tolist()) if edge_id not in removed], dtype=np.int32
    ).reshape(-1, 2)
    remaining_probabilities = np.asarray(
        [probability for edge_id, probability in enumerate(probabilities.tolist()) if edge_id not in removed],
        dtype=np.float32,
    )
    degrees, node_edges = graph_degrees(remaining_edges.shape[0], nodes_rc.shape[0], remaining_edges)
    remaining_adjacency = build_graph_adjacency(nodes_rc, remaining_edges)
    fan_removed: set[int] = set()
    for node_idx in np.where(degrees >= fan_min_degree)[0].tolist():
        incident = []
        for edge_id in node_edges[node_idx]:
            src_idx, dst_idx = (int(value) for value in remaining_edges[edge_id])
            neighbor_idx = dst_idx if src_idx == node_idx else src_idx
            vector = nodes_rc[neighbor_idx] - nodes_rc[node_idx]
            length = float(np.linalg.norm(vector))
            if length > 0:
                incident.append((edge_id, neighbor_idx, vector / length, length, float(remaining_probabilities[edge_id])))
        if sum(np.isfinite(row[4]) and row[4] <= max_weak_probability for row in incident) < 3:
            continue
        opposite_pairs = []
        for first_pos, first in enumerate(incident):
            for second in incident[first_pos + 1 :]:
                cosine = float(np.dot(first[2], second[2]))
                if cosine <= -fan_min_opposite_cosine:
                    opposite_pairs.append((cosine, first[0], second[0]))
        paired: set[int] = set()
        for _, first_edge_id, second_edge_id in sorted(opposite_pairs):
            if first_edge_id in paired or second_edge_id in paired:
                continue
            paired.update({first_edge_id, second_edge_id})
        unpaired = []
        for candidate in incident:
            if candidate[0] in paired or not np.isfinite(candidate[4]):
                continue
            if candidate[3] > fan_max_edge_length_px or candidate[4] > max_weak_probability:
                continue
            same_side_competitors = [
                other
                for other in incident
                if other[0] != candidate[0]
                and float(np.dot(candidate[2], other[2])) >= fan_min_same_side_cosine
                and np.isfinite(other[4])
                and other[4] <= max_weak_probability
            ]
            same_side = max(
                (float(np.dot(candidate[2], other[2])) for other in same_side_competitors),
                default=-1.0,
            )
            if same_side < fan_min_same_side_cosine:
                continue
            alternative, _ = shortest_graph_path_length_excluding_edge(
                nodes_rc,
                remaining_edges,
                node_idx,
                candidate[1],
                candidate[0],
                max_length_px=260.0,
                adjacency=remaining_adjacency,
            )
            if np.isfinite(alternative):
                unpaired.append((candidate[4], candidate, same_side))
        if not unpaired:
            continue
        _, candidate, same_side = min(unpaired)
        edge_id, neighbor_idx, _, edge_length, probability = candidate
        fan_removed.add(edge_id)
        audit_rows.append(
            {
                "cleanup_id": len(audit_rows),
                "action": "detach_redundant_fan_branch",
                "edge_id": edge_id,
                "src_row": float(nodes_rc[node_idx, 0]),
                "src_col": float(nodes_rc[node_idx, 1]),
                "dst_row": float(nodes_rc[neighbor_idx, 0]),
                "dst_col": float(nodes_rc[neighbor_idx, 1]),
                "edge_length_px": edge_length,
                "cycle_length_px": "",
                "topology_probability": probability,
                "probability_margin": same_side,
            }
        )
    kept_edges = np.asarray(
        [edge for edge_id, edge in enumerate(remaining_edges.tolist()) if edge_id not in fan_removed], dtype=np.int32
    ).reshape(-1, 2)
    return compact_graph(nodes_rc, kept_edges) + (audit_rows,)


def _find_vertical_divided_anchor_pair(
    nodes_rc: np.ndarray,
    center_probability: np.ndarray,
    center: np.ndarray,
    side: int,
    min_distance_px: float = 65.0,
    max_distance_px: float = 230.0,
    max_row_difference_px: float = 10.0,
    min_spacing_px: float = 8.0,
    max_spacing_px: float = 30.0,
    lateral_search_px: float = 45.0,
    node_index: PointGridIndex | None = None,
) -> tuple[int, int] | None:
    return find_vertical_divided_anchor_pair_indexed(
        nodes_rc,
        center_probability,
        center,
        side,
        min_distance_px=min_distance_px,
        max_distance_px=max_distance_px,
        max_row_difference_px=max_row_difference_px,
        min_spacing_px=min_spacing_px,
        max_spacing_px=max_spacing_px,
        lateral_search_px=lateral_search_px,
        node_index=node_index,
    )


def _probability_guided_vertical_path(
    center_probability: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    lateral_margin_px: int = 35,
    separation_penalty: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    if end[0] <= start[0]:
        return []
    start_row, end_row = int(round(start[0])), int(round(end[0]))
    min_col = max(0, int(np.floor(min(start[1], end[1]) - lateral_margin_px)))
    max_col = min(center_probability.shape[1] - 1, int(np.ceil(max(start[1], end[1]) + lateral_margin_px)))
    rows = np.arange(start_row, end_row + 1, dtype=np.int32)
    cols = np.arange(min_col, max_col + 1, dtype=np.int32)
    target_cols = np.linspace(float(start[1]), float(end[1]), rows.size)
    cost = np.full((rows.size, cols.size), np.inf, dtype=np.float32)
    parents = np.zeros((rows.size, cols.size), dtype=np.int16)
    penalty = separation_penalty if separation_penalty is not None else np.zeros_like(center_probability)
    cost[0] = 2.5 * np.abs(cols - float(start[1])) + 4.0 * (1.0 - center_probability[start_row, cols]) + penalty[start_row, cols]
    for row_pos, row in enumerate(rows[1:], start=1):
        for col_pos, col in enumerate(cols):
            previous_positions = np.arange(max(0, col_pos - 2), min(cols.size, col_pos + 3))
            values = (
                cost[row_pos - 1, previous_positions]
                + 0.45 * np.abs(previous_positions - col_pos)
                + 0.025 * abs(float(col) - target_cols[row_pos])
                + 4.0 * (1.0 - float(center_probability[row, col]))
                + float(penalty[row, col])
            )
            best = int(np.argmin(values))
            cost[row_pos, col_pos] = float(values[best])
            parents[row_pos, col_pos] = int(previous_positions[best])
    col_pos = int(np.argmin(cost[-1] + 2.5 * np.abs(cols - float(end[1]))))
    path = []
    for row_pos in range(rows.size - 1, -1, -1):
        path.append((float(rows[row_pos]), float(cols[col_pos])))
        if row_pos:
            col_pos = int(parents[row_pos, col_pos])
    path.reverse()
    path[0] = (float(start[0]), float(start[1]))
    path[-1] = (float(end[0]), float(end[1]))
    return path


def _probability_guided_vertical_path_pair(
    center_probability: np.ndarray,
    left_start: np.ndarray,
    left_end: np.ndarray,
    right_start: np.ndarray,
    right_end: np.ndarray,
    lateral_margin_px: int = 35,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Trace both carriageways together so the junction cannot collapse their spacing."""
    if min(left_end[0], right_end[0]) <= max(left_start[0], right_start[0]):
        return [], []
    left_path = _probability_guided_vertical_path(
        center_probability, left_start, left_end, lateral_margin_px=lateral_margin_px
    )
    right_path = _probability_guided_vertical_path(
        center_probability, right_start, right_end, lateral_margin_px=lateral_margin_px
    )
    if not left_path or not right_path:
        return [], []

    left = np.asarray(left_path, dtype=np.float32)
    right = np.asarray(right_path, dtype=np.float32)
    rows = np.arange(int(round(left_start[0])), int(round(left_end[0])) + 1, dtype=np.int32)
    target_spacing = np.linspace(
        float(right_start[1] - left_start[1]), float(right_end[1] - left_end[1]), rows.size
    )
    minimum_spacing = max(8.0, 0.82 * min(float(target_spacing[0]), float(target_spacing[-1])))
    midpoint_target = 0.5 * (
        np.linspace(float(left_start[1]), float(left_end[1]), rows.size)
        + np.linspace(float(right_start[1]), float(right_end[1]), rows.size)
    )

    corrected_left = left[:, 1].copy()
    corrected_right = right[:, 1].copy()
    for row_pos, row in enumerate(rows):
        spacing = float(corrected_right[row_pos] - corrected_left[row_pos])
        desired_spacing = max(minimum_spacing, float(target_spacing[row_pos]))
        if spacing < desired_spacing:
            midpoint = 0.5 * float(corrected_left[row_pos] + corrected_right[row_pos])
            midpoint = 0.75 * midpoint + 0.25 * float(midpoint_target[row_pos])
            corrected_left[row_pos] = midpoint - 0.5 * desired_spacing
            corrected_right[row_pos] = midpoint + 0.5 * desired_spacing

    # Light smoothing removes one-pixel stair steps introduced by the spacing projection.
    kernel = np.ones(5, dtype=np.float32) / 5.0
    for values, start_col, end_col in (
        (corrected_left, float(left_start[1]), float(left_end[1])),
        (corrected_right, float(right_start[1]), float(right_end[1])),
    ):
        padded = np.pad(values, (2, 2), mode="edge")
        values[:] = np.convolve(padded, kernel, mode="valid")
        values[0], values[-1] = start_col, end_col
    for row_pos in range(rows.size):
        spacing = float(corrected_right[row_pos] - corrected_left[row_pos])
        if spacing < minimum_spacing:
            midpoint = 0.5 * float(corrected_left[row_pos] + corrected_right[row_pos])
            corrected_left[row_pos] = midpoint - 0.5 * minimum_spacing
            corrected_right[row_pos] = midpoint + 0.5 * minimum_spacing

    return (
        [(float(row), float(col)) for row, col in zip(rows, corrected_left)],
        [(float(row), float(col)) for row, col in zip(rows, corrected_right)],
    )


def _find_quadrant_turn_endpoints(
    nodes_rc: np.ndarray,
    center: np.ndarray,
    quadrant_row_sign: int,
    quadrant_col_sign: int,
    edges: np.ndarray | None = None,
    min_offset_px: float = 28.0,
    max_offset_px: float = 90.0,
) -> tuple[int, int] | None:
    """Find the inner vertical/horizontal carriageway nodes for one junction corner."""
    used_nodes = set(edges.ravel().tolist()) if edges is not None and edges.size else set(range(nodes_rc.shape[0]))
    vertical = []
    horizontal = []
    for node_idx, point in enumerate(nodes_rc):
        if node_idx not in used_nodes:
            continue
        row_offset = quadrant_row_sign * float(point[0] - center[0])
        col_offset = quadrant_col_sign * float(point[1] - center[1])
        if min_offset_px <= row_offset <= max_offset_px and 1.0 <= col_offset <= 28.0:
            vertical.append((abs(row_offset - 50.0) + 0.25 * col_offset, node_idx))
        if min_offset_px <= col_offset <= max_offset_px and 1.0 <= row_offset <= 28.0:
            horizontal.append((abs(col_offset - 50.0) + 0.25 * row_offset, node_idx))
    if not vertical or not horizontal:
        return None
    return min(vertical)[1], min(horizontal)[1]


def _quadrant_has_turn_connection(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    center: np.ndarray,
    quadrant_row_sign: int,
    quadrant_col_sign: int,
    min_diagonal_degrees: float = 25.0,
    max_diagonal_degrees: float = 65.0,
) -> bool:
    for src_idx, dst_idx in edges.tolist():
        start, end = nodes_rc[int(src_idx)], nodes_rc[int(dst_idx)]
        midpoint = 0.5 * (start + end)
        row_offset = quadrant_row_sign * float(midpoint[0] - center[0])
        col_offset = quadrant_col_sign * float(midpoint[1] - center[1])
        if not 8.0 <= row_offset <= 75.0 or not 8.0 <= col_offset <= 75.0:
            continue
        vector = end - start
        angle = float(np.degrees(np.arctan2(abs(vector[0]), abs(vector[1]) + 1e-6)))
        if min_diagonal_degrees <= angle <= max_diagonal_degrees:
            return True
    # Resampled turn connectors can be split into short pieces whose individual
    # angles are less diagonal. Treat a connected path across the corner as present.
    candidates = np.where(
        (quadrant_row_sign * (nodes_rc[:, 0] - center[0]) >= 4.0)
        & (quadrant_row_sign * (nodes_rc[:, 0] - center[0]) <= 85.0)
        & (quadrant_col_sign * (nodes_rc[:, 1] - center[1]) >= 4.0)
        & (quadrant_col_sign * (nodes_rc[:, 1] - center[1]) <= 85.0)
    )[0]
    candidate_set = set(candidates.tolist())
    adjacency: dict[int, list[int]] = {node_idx: [] for node_idx in candidate_set}
    for src_idx, dst_idx in edges.tolist():
        src_idx, dst_idx = int(src_idx), int(dst_idx)
        if src_idx in candidate_set and dst_idx in candidate_set:
            adjacency[src_idx].append(dst_idx)
            adjacency[dst_idx].append(src_idx)
    vertical_nodes = [
        node_idx for node_idx in candidate_set
        if quadrant_col_sign * float(nodes_rc[node_idx, 1] - center[1]) <= 22.0
        and quadrant_row_sign * float(nodes_rc[node_idx, 0] - center[0]) >= 28.0
    ]
    horizontal_nodes = {
        node_idx for node_idx in candidate_set
        if quadrant_row_sign * float(nodes_rc[node_idx, 0] - center[0]) <= 22.0
        and quadrant_col_sign * float(nodes_rc[node_idx, 1] - center[1]) >= 28.0
    }
    queue = deque(vertical_nodes)
    visited = set(vertical_nodes)
    while queue:
        node_idx = queue.popleft()
        if node_idx in horizontal_nodes:
            return True
        for neighbor in adjacency.get(node_idx, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _add_missing_quadrant_turns(
    final_nodes: list[tuple[float, float]],
    final_edges: list[tuple[int, int]],
    center_probability: np.ndarray,
    center: np.ndarray,
    sample_step_px: float,
) -> list[str]:
    """Complete one missing corner only when the other three turn connectors are present."""
    nodes = np.asarray(final_nodes, dtype=np.float32)
    edges = np.asarray(final_edges, dtype=np.int32).reshape(-1, 2)
    quadrants = [(-1, -1, "northwest"), (-1, 1, "northeast"), (1, -1, "southwest"), (1, 1, "southeast")]
    present = {
        name: _quadrant_has_turn_connection(nodes, edges, center, row_sign, col_sign)
        for row_sign, col_sign, name in quadrants
    }
    missing = [(row_sign, col_sign, name) for row_sign, col_sign, name in quadrants if not present[name]]
    if len(missing) != 1 or sum(present.values()) != 3:
        return []
    row_sign, col_sign, name = missing[0]
    endpoints = _find_quadrant_turn_endpoints(nodes, center, row_sign, col_sign, edges=edges)
    if endpoints is None:
        return []
    vertical_idx, horizontal_idx = endpoints
    start, end = nodes[vertical_idx], nodes[horizontal_idx]
    corner = np.asarray([start[0], end[1]], dtype=np.float32)
    control = 0.7 * corner + 0.3 * (0.5 * (start + end))
    count = max(3, int(np.ceil(float(np.linalg.norm(end - start)))) + 1)
    ratios = np.linspace(0.0, 1.0, count, dtype=np.float32)
    dense = [
        ((1.0 - ratio) ** 2 * start + 2.0 * (1.0 - ratio) * ratio * control + ratio ** 2 * end).tolist()
        for ratio in ratios
    ]
    support = sample_polyline_probability(
        center_probability, np.asarray(dense[0], dtype=np.float32), np.asarray(dense[-1], dtype=np.float32)
    )
    # The missing corner is often the lowest-confidence turn, so require contextual symmetry
    # but only modest direct probability support.
    if support < 0.12:
        return []
    sampled = resample_polyline([tuple(point) for point in dense], sample_step_px)
    previous_idx = vertical_idx
    for point in sampled[1:-1]:
        point_idx = len(final_nodes)
        final_nodes.append((float(point[0]), float(point[1])))
        final_edges.append((previous_idx, point_idx))
        previous_idx = point_idx
    final_edges.append((previous_idx, horizontal_idx))
    return [name]


def _segment_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    first_vector = first_end - first_start
    second_vector = second_end - second_start
    denominator = float(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])
    if abs(denominator) <= 1e-6:
        return None
    offset = second_start - first_start
    first_ratio = float((offset[0] * second_vector[1] - offset[1] * second_vector[0]) / denominator)
    second_ratio = float((offset[0] * first_vector[1] - offset[1] * first_vector[0]) / denominator)
    if not 0.0 <= first_ratio <= 1.0 or not 0.0 <= second_ratio <= 1.0:
        return None
    return first_ratio, second_ratio, first_start + first_ratio * first_vector


def restore_divided_corridors_through_junctions(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    center_probability: np.ndarray,
    cleanup_rows: list[dict],
    sample_step_px: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Replace a vertical Y-merge with two probability-guided through lanes."""
    if edges.size == 0 or not cleanup_rows:
        return nodes_rc, edges, []
    anchor_node_index = PointGridIndex.build(nodes_rc, 32.0)
    for cleanup in cleanup_rows:
        if cleanup.get("action") != "remove_weak_short_cycle_edge":
            continue
        start = np.asarray([float(cleanup["src_row"]), float(cleanup["src_col"])], dtype=np.float32)
        end = np.asarray([float(cleanup["dst_row"]), float(cleanup["dst_col"])], dtype=np.float32)
        vector = end - start
        if abs(float(vector[1])) < 1.5 * abs(float(vector[0])):
            continue
        center = 0.5 * (start + end)
        north_pair = _find_vertical_divided_anchor_pair(
            nodes_rc, center_probability, center, -1, node_index=anchor_node_index,
        )
        south_pair = _find_vertical_divided_anchor_pair(
            nodes_rc, center_probability, center, 1, node_index=anchor_node_index,
        )
        if north_pair is None or south_pair is None:
            continue
        north_pair = tuple(sorted(north_pair, key=lambda node_idx: float(nodes_rc[node_idx, 1])))
        south_pair = tuple(sorted(south_pair, key=lambda node_idx: float(nodes_rc[node_idx, 1])))
        north_spacing = float(nodes_rc[north_pair[1], 1] - nodes_rc[north_pair[0], 1])
        south_spacing = float(nodes_rc[south_pair[1], 1] - nodes_rc[south_pair[0], 1])
        if abs(north_spacing - south_spacing) > 12.0:
            continue
        paths = list(_probability_guided_vertical_path_pair(
            center_probability,
            nodes_rc[north_pair[0]],
            nodes_rc[south_pair[0]],
            nodes_rc[north_pair[1]],
            nodes_rc[south_pair[1]],
        ))
        if len(paths) != 2:
            continue
        top_row = min(float(nodes_rc[idx, 0]) for idx in north_pair)
        bottom_row = max(float(nodes_rc[idx, 0]) for idx in south_pair)
        path_arrays = [np.asarray(path, dtype=np.float32) for path in paths]

        removed_edge_ids: set[int] = set()
        anchor_ids = set(north_pair + south_pair)
        for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
            src_idx, dst_idx = int(src_idx), int(dst_idx)
            edge_start, edge_end = nodes_rc[src_idx], nodes_rc[dst_idx]
            midpoint = 0.5 * (edge_start + edge_end)
            if not top_row <= midpoint[0] <= bottom_row:
                continue
            edge_vector = edge_end - edge_start
            edge_length = float(np.linalg.norm(edge_vector))
            if edge_length <= 0:
                continue
            from_vertical_degrees = float(np.degrees(np.arctan2(abs(edge_vector[1]), abs(edge_vector[0]))))
            if from_vertical_degrees > 25.0:
                continue
            distances = []
            for path in path_arrays:
                row_pos = int(np.clip(round(midpoint[0] - path[0, 0]), 0, path.shape[0] - 1))
                distances.append(abs(float(midpoint[1] - path[row_pos, 1])))
            if min(distances) <= 11.0:
                removed_edge_ids.add(edge_id)
        kept = [(edge_id, int(src), int(dst)) for edge_id, (src, dst) in enumerate(edges.tolist()) if edge_id not in removed_edge_ids]
        final_nodes = [tuple(float(value) for value in node) for node in nodes_rc.tolist()]
        final_edges: list[tuple[int, int]] = []
        split_points: dict[int, list[tuple[float, np.ndarray, int]]] = {}
        path_intersections: list[list[tuple[float, int]]] = [[] for _ in paths]

        def add_node(point: np.ndarray) -> int:
            final_nodes.append((float(point[0]), float(point[1])))
            return len(final_nodes) - 1

        for kept_edge_id, src_idx, dst_idx in kept:
            edge_start, edge_end = nodes_rc[src_idx], nodes_rc[dst_idx]
            edge_vector = edge_end - edge_start
            if abs(float(edge_vector[1])) < 1.2 * abs(float(edge_vector[0])):
                continue
            if not center[0] - 65.0 <= 0.5 * float(edge_start[0] + edge_end[0]) <= center[0] + 65.0:
                continue
            for path_id, path in enumerate(path_arrays):
                cumulative = 0.0
                for first, second in zip(path[:-1], path[1:]):
                    intersection = _segment_intersection(first, second, edge_start, edge_end)
                    segment_length = float(np.linalg.norm(second - first))
                    if intersection is not None:
                        path_ratio, edge_ratio, point = intersection
                        node_idx = add_node(point)
                        split_points.setdefault(kept_edge_id, []).append((edge_ratio, point, node_idx))
                        path_intersections[path_id].append((cumulative + path_ratio * segment_length, node_idx))
                        break
                    cumulative += segment_length
        for kept_edge_id, src_idx, dst_idx in kept:
            points = sorted(split_points.get(kept_edge_id, []), key=lambda row: row[0])
            node_ids = [src_idx] + [row[2] for row in points] + [dst_idx]
            final_edges.extend((first, second) for first, second in zip(node_ids[:-1], node_ids[1:]) if first != second)
        for path_id, (path, north_idx, south_idx) in enumerate(zip(path_arrays, north_pair, south_pair)):
            sampled = resample_polyline([tuple(point) for point in path.tolist()], sample_step_px)
            vertices: list[tuple[float, int | None, np.ndarray]] = [(0.0, north_idx, nodes_rc[north_idx])]
            cumulative = 0.0
            for first, second in zip(path[:-1], path[1:]):
                cumulative += float(np.linalg.norm(second - first))
            for point in sampled[1:-1]:
                point_array = np.asarray(point, dtype=np.float32)
                distance = float(point_array[0] - path[0, 0])
                vertices.append((distance, None, point_array))
            vertices.extend((distance, node_idx, np.asarray(final_nodes[node_idx], dtype=np.float32)) for distance, node_idx in path_intersections[path_id])
            vertices.append((cumulative, south_idx, nodes_rc[south_idx]))
            vertices.sort(key=lambda row: row[0])
            path_node_ids = []
            for _, node_idx, point in vertices:
                if node_idx is None:
                    node_idx = add_node(point)
                if not path_node_ids or path_node_ids[-1] != node_idx:
                    path_node_ids.append(node_idx)
            final_edges.extend((first, second) for first, second in zip(path_node_ids[:-1], path_node_ids[1:]) if first != second)
        restored_turn_quadrants = _add_missing_quadrant_turns(
            final_nodes, final_edges, center_probability, center, sample_step_px
        )
        compact_nodes, compact_edges = compact_graph(
            np.asarray(final_nodes, dtype=np.float32), np.asarray(final_edges, dtype=np.int32).reshape(-1, 2)
        )
        audit = [{
            "repair_id": 0,
            "action": "restore_divided_corridor_through_junction",
            "center_row": float(center[0]),
            "center_col": float(center[1]),
            "north_anchor_nodes": f"{north_pair[0]};{north_pair[1]}",
            "south_anchor_nodes": f"{south_pair[0]};{south_pair[1]}",
            "north_spacing_px": north_spacing,
            "south_spacing_px": south_spacing,
            "removed_edge_count": len(removed_edge_ids),
            "added_path_count": 2,
            "restored_turn_count": len(restored_turn_quadrants),
            "restored_turn_quadrants": ";".join(restored_turn_quadrants),
        }]
        return compact_nodes, compact_edges, audit
    return nodes_rc, edges, []


def recenter_graph_nodes(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    binary: np.ndarray,
    road_probability: np.ndarray,
    max_shift_px: int,
) -> tuple[np.ndarray, list[dict]]:
    if max_shift_px <= 0 or nodes_rc.size == 0:
        return nodes_rc.copy(), []
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    degrees, _ = graph_degrees(edges.shape[0], nodes_rc.shape[0], edges)
    corrected = nodes_rc.copy()
    audit_rows: list[dict] = []
    height, width = binary.shape
    for node_idx, (row, col) in enumerate(nodes_rc.tolist()):
        if degrees[node_idx] != 2:
            continue
        center_row, center_col = int(round(row)), int(round(col))
        y0, y1 = max(0, center_row - max_shift_px), min(height, center_row + max_shift_px + 1)
        x0, x1 = max(0, center_col - max_shift_px), min(width, center_col + max_shift_px + 1)
        patch_distance = distance[y0:y1, x0:x1]
        patch_probability = road_probability[y0:y1, x0:x1]
        if patch_distance.size == 0 or float(patch_distance.max()) <= 0:
            continue
        normalized_distance = patch_distance / max(float(patch_distance.max()), 1e-6)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        displacement = np.hypot(yy - row, xx - col) / max(float(max_shift_px), 1.0)
        score = 0.65 * normalized_distance + 0.35 * patch_probability - 0.20 * displacement
        score[(patch_distance <= 0) | (displacement > 1.0)] = -np.inf
        local_row, local_col = np.unravel_index(int(np.argmax(score)), score.shape)
        target = np.asarray([local_row + y0, local_col + x0], dtype=np.float32)
        shift = float(np.linalg.norm(target - nodes_rc[node_idx]))
        if shift < 0.75:
            continue
        corrected[node_idx] = target
        audit_rows.append(
            {
                "node_idx": node_idx,
                "old_row": row,
                "old_col": col,
                "new_row": float(target[0]),
                "new_col": float(target[1]),
                "shift_px": shift,
                "road_probability": float(road_probability[int(target[0]), int(target[1])]),
                "action": "recenter_to_surface_medial_ridge",
            }
        )
    return corrected, audit_rows


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def infer_molra_mask(
    image_path: Path,
    sam_pretrained_path: Path,
    weight_path: Path,
    device: torch.device,
    tile: int,
    overlap: int,
    threshold: float,
    image_size: int,
) -> np.ndarray:
    add_sam_molra_path()
    from infer_img import get_model_holder, infer_tile, load_checkpoint, read_image
    from networks.sam_multi_lora import build_sam_vit_b_adapter_linknet_multi_lora, resize_model_pos_embed

    if not sam_pretrained_path.is_file():
        raise FileNotFoundError(f"SAM pretrained weight not found: {sam_pretrained_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"SAM_MLoRA weight not found: {weight_path}")

    img_bgr, _ = read_image(image_path)
    model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_multi_lora(
        str(sam_pretrained_path),
        image_size=image_size,
    )
    model = model.to(device)
    load_checkpoint(model, str(weight_path))

    holder = get_model_holder(model)
    holder.enc = resize_model_pos_embed(
        holder.enc,
        img_size=tile,
        encoder_global_attn_indexes=encoder_global_attn_indexes,
    )
    model.eval()

    probability = infer_tile(
        img_bgr,
        model,
        device,
        tile=tile,
        overlap=overlap,
        threshold=threshold,
        return_probability=True,
    )
    return np.clip(probability * 255.0, 0, 255).astype(np.uint8)


def read_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return mask


def probability_from_u8(image: np.ndarray) -> tuple[np.ndarray, str]:
    probability = image.astype(np.float32) / 255.0
    unique = np.unique(image)
    source_type = "binary_mask_fallback" if unique.size <= 2 else "continuous_u8_probability"
    return probability, source_type


def read_probability(path: Path, expected_shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    image = read_mask(path)
    if image.shape != expected_shape:
        raise ValueError(f"Probability/image shape mismatch: probability={image.shape}, expected={expected_shape}")
    return probability_from_u8(image)


def auto_centerline_probability_path(graph_path: Path, stem: str) -> Path | None:
    run_dir = graph_path.parent.parent
    candidates = [
        run_dir / "mask" / f"{stem}_road.png",
        graph_path.parent / f"{stem}_road.png",
    ]
    return next((path for path in candidates if path.is_file()), None)


def auto_road_probability_path(mask_path: Path | None, stem: str) -> Path | None:
    if mask_path is None:
        return None
    candidates = [
        mask_path.with_name(f"{mask_path.stem}_probability.png"),
        mask_path.parent / f"{stem}_mask_probability.png",
        mask_path.parent / f"{stem}_probability.png",
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_edge_topology_probabilities(
    graph_path: Path, nodes_rc: np.ndarray, edges: np.ndarray, match_tolerance_px: float = 3.0
) -> tuple[np.ndarray, str]:
    score_path = graph_path.with_name(f"{graph_path.stem}_edge_scores.csv")
    if not score_path.is_file():
        return np.full(edges.shape[0], np.nan, dtype=np.float32), "unavailable"
    coordinate_scores: dict[tuple[tuple[int, int], tuple[int, int]], list[float]] = {}
    with open(score_path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            src = (int(round(float(row["src_row"]))), int(round(float(row["src_col"]))))
            dst = (int(round(float(row["dst_row"]))), int(round(float(row["dst_col"]))))
            key = tuple(sorted((src, dst)))
            coordinate_scores.setdefault(key, []).append(float(row["topology_probability"]))
    probabilities = np.full(edges.shape[0], np.nan, dtype=np.float32)
    score_segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    for key, values in coordinate_scores.items():
        score_segments.append(
            (np.asarray(key[0], dtype=np.float32), np.asarray(key[1], dtype=np.float32), float(np.mean(values)))
        )
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        src = tuple(int(round(value)) for value in nodes_rc[int(src_idx)])
        dst = tuple(int(round(value)) for value in nodes_rc[int(dst_idx)])
        values = coordinate_scores.get(tuple(sorted((src, dst))))
        if values:
            probabilities[edge_id] = float(np.mean(values))
            continue
        src_point = nodes_rc[int(src_idx)]
        dst_point = nodes_rc[int(dst_idx)]
        best_distance = float("inf")
        best_score = np.nan
        for score_src, score_dst, score in score_segments:
            direct = max(float(np.linalg.norm(src_point - score_src)), float(np.linalg.norm(dst_point - score_dst)))
            reverse = max(float(np.linalg.norm(src_point - score_dst)), float(np.linalg.norm(dst_point - score_src)))
            distance = min(direct, reverse)
            if distance < best_distance:
                best_distance = distance
                best_score = score
        if best_distance <= match_tolerance_px:
            probabilities[edge_id] = best_score
    matched = int(np.count_nonzero(np.isfinite(probabilities)))
    return probabilities, f"{score_path} ({matched}/{edges.shape[0]} matched)"


def load_edge_provenance(
    graph_path: Path,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    match_tolerance_px: float = 3.0,
) -> list[dict]:
    """Match SAMRoad edge provenance to the current (possibly recentered) graph."""
    score_path = graph_path.with_name(f"{graph_path.stem}_edge_scores.csv")
    defaults = [
        {
            "line_source": "samroad", "recovery_score": 0.0,
            "center_conf": 0.0, "surface_conf": 0.0,
            "recovery_reason": "", "qa_state": "auto", "recovery_id": "",
        }
        for _ in range(edges.shape[0])
    ]
    if not score_path.is_file():
        return defaults
    records = []
    with open(score_path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            try:
                src = np.asarray([float(row["src_row"]), float(row["src_col"])], dtype=np.float32)
                dst = np.asarray([float(row["dst_row"]), float(row["dst_col"])], dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                continue
            records.append((src, dst, row))
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        src = nodes_rc[int(src_idx)]
        dst = nodes_rc[int(dst_idx)]
        best_distance = float("inf")
        best_row = None
        for record_src, record_dst, row in records:
            direct = max(float(np.linalg.norm(src - record_src)), float(np.linalg.norm(dst - record_dst)))
            reverse = max(float(np.linalg.norm(src - record_dst)), float(np.linalg.norm(dst - record_src)))
            distance = min(direct, reverse)
            if distance < best_distance:
                best_distance, best_row = distance, row
        if best_row is None or best_distance > match_tolerance_px:
            continue
        defaults[edge_id] = {
            "line_source": str(best_row.get("line_source", "samroad") or "samroad"),
            "recovery_score": float(best_row.get("recovery_score", 0.0) or 0.0),
            "center_conf": float(best_row.get("center_conf", 0.0) or 0.0),
            "surface_conf": float(best_row.get("surface_conf", 0.0) or 0.0),
            "recovery_reason": str(best_row.get("recovery_reason", "") or ""),
            "qa_state": str(best_row.get("qa_state", "auto") or "auto"),
            "recovery_id": str(best_row.get("recovery_id", "") or ""),
        }
    return defaults


def resolve_pixel_size(image_path: Path, requested: float) -> tuple[float, str]:
    if requested > 0:
        return float(requested), "argument"
    try:
        import rasterio

        with rasterio.open(image_path) as dataset:
            transform = dataset.transform
            col_size = float(np.hypot(transform.a, transform.d))
            row_size = float(np.hypot(transform.b, transform.e))
        sizes = [value for value in (row_size, col_size) if value > 0]
        if sizes:
            return float(np.mean(np.asarray(sizes, dtype=np.float64))), "raster_transform_mean"
    except Exception as exc:
        print(f"Warning: could not read pixel size from {image_path}: {exc}")
    print(f"Warning: no usable geospatial pixel size for {image_path}; falling back to 1.0 map unit/pixel.")
    return 1.0, "fallback_1"


def clean_mask(mask: np.ndarray, threshold: int, close_kernel: int, open_kernel: int, min_area: int) -> np.ndarray:
    binary = (mask >= threshold).astype(np.uint8)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == label] = 1
        binary = keep
    return binary


def graph_buffer_mask(shape: tuple[int, int], nodes_rc: np.ndarray, edges: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    thickness = max(1, int(radius) * 2 + 1)
    for src_idx, dst_idx in edges.tolist():
        r0, c0 = nodes_rc[src_idx]
        r1, c1 = nodes_rc[dst_idx]
        cv2.line(
            mask,
            (int(round(c0)), int(round(r0))),
            (int(round(c1)), int(round(r1))),
            1,
            thickness,
        )
    return mask


def graph_degrees(edge_count: int, node_count: int, edges: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    degrees = np.zeros(node_count, dtype=np.int32)
    node_edges: list[list[int]] = [[] for _ in range(node_count)]
    for edge_id in range(edge_count):
        src_idx, dst_idx = edges[edge_id]
        degrees[src_idx] += 1
        degrees[dst_idx] += 1
        node_edges[src_idx].append(edge_id)
        node_edges[dst_idx].append(edge_id)
    return degrees, node_edges


def graph_component_ids(node_count: int, edges: np.ndarray) -> np.ndarray:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for src_idx, dst_idx in edges.tolist():
        adjacency[int(src_idx)].append(int(dst_idx))
        adjacency[int(dst_idx)].append(int(src_idx))
    component_ids = np.full(node_count, -1, dtype=np.int32)
    component_id = 0
    for start in range(node_count):
        if component_ids[start] >= 0:
            continue
        component_ids[start] = component_id
        queue = deque([start])
        while queue:
            node_idx = queue.popleft()
            for neighbor_idx in adjacency[node_idx]:
                if component_ids[neighbor_idx] < 0:
                    component_ids[neighbor_idx] = component_id
                    queue.append(neighbor_idx)
        component_id += 1
    return component_ids


def graph_bridge_edge_ids(node_count: int, edges: np.ndarray) -> set[int]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        adjacency[int(src_idx)].append((int(dst_idx), edge_id))
        adjacency[int(dst_idx)].append((int(src_idx), edge_id))

    discovery = np.full(node_count, -1, dtype=np.int64)
    low = np.full(node_count, -1, dtype=np.int64)
    parent_node = np.full(node_count, -1, dtype=np.int64)
    parent_edge = np.full(node_count, -1, dtype=np.int64)
    bridges: set[int] = set()
    clock = 0
    for root in range(node_count):
        if discovery[root] >= 0:
            continue
        discovery[root] = low[root] = clock
        clock += 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            node_idx, neighbor_position = stack[-1]
            if neighbor_position < len(adjacency[node_idx]):
                neighbor_idx, edge_id = adjacency[node_idx][neighbor_position]
                stack[-1] = (node_idx, neighbor_position + 1)
                if edge_id == parent_edge[node_idx]:
                    continue
                if discovery[neighbor_idx] < 0:
                    parent_node[neighbor_idx] = node_idx
                    parent_edge[neighbor_idx] = edge_id
                    discovery[neighbor_idx] = low[neighbor_idx] = clock
                    clock += 1
                    stack.append((neighbor_idx, 0))
                else:
                    low[node_idx] = min(low[node_idx], discovery[neighbor_idx])
                continue

            stack.pop()
            parent_idx = int(parent_node[node_idx])
            if parent_idx < 0:
                continue
            low[parent_idx] = min(low[parent_idx], low[node_idx])
            if low[node_idx] > discovery[parent_idx]:
                bridges.add(int(parent_edge[node_idx]))
    return bridges


def graph_topology_metrics(
    nodes_rc: np.ndarray, edges: np.ndarray,
    graph_context: GraphSpatialContext | None = None,
) -> dict:
    node_count = int(nodes_rc.shape[0])
    edge_count = int(edges.shape[0])
    if graph_context is None:
        degrees, _node_edges = graph_degrees(edge_count, node_count, edges)
        component_ids = graph_component_ids(node_count, edges)
        bridge_edge_count = len(graph_bridge_edge_ids(node_count, edges))
    else:
        degrees = graph_context.degrees
        component_ids = graph_context.component_ids
        bridge_edge_count = len(graph_context.bridge_edge_ids)
    component_sizes = np.bincount(component_ids, minlength=int(component_ids.max()) + 1) if node_count else np.asarray([])
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "component_count": int(component_sizes.size),
        "largest_component_node_count": int(component_sizes.max()) if component_sizes.size else 0,
        "endpoint_count": int(np.count_nonzero(degrees == 1)),
        "isolated_node_count": int(np.count_nonzero(degrees == 0)),
        "junction_node_count": int(np.count_nonzero(degrees >= 3)),
        "bridge_edge_count": bridge_edge_count,
    }


def build_road_chains(
    nodes_rc: np.ndarray, edges: np.ndarray,
    graph_context: GraphSpatialContext | None = None,
) -> list[dict]:
    context = graph_context or GraphSpatialContext.build(nodes_rc, edges)
    return context.build_road_chain_rows()


def annotate_edge_topology(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    edge_rows: list[dict],
    road_probability: np.ndarray | None = None,
    center_probability: np.ndarray | None = None,
    topology_probabilities: np.ndarray | None = None,
    auto_retain_center_probability: float = 0.6,
    graph_context: GraphSpatialContext | None = None,
) -> None:
    context = graph_context or GraphSpatialContext.build(nodes_rc, edges)
    degrees = context.degrees
    component_ids = context.component_ids
    component_sizes = np.bincount(component_ids, minlength=int(component_ids.max()) + 1) if component_ids.size else np.asarray([])
    bridge_ids = context.bridge_edge_ids
    for row in edge_rows:
        edge_id = int(row["edge_id"])
        src_idx, dst_idx = (int(value) for value in edges[edge_id])
        status = str(row.get("status", ""))
        is_bridge = edge_id in bridge_ids
        is_dangling = bool(degrees[src_idx] == 1 or degrees[dst_idx] == 1)
        component_id = int(component_ids[src_idx])
        component_node_count = int(component_sizes[component_id])
        edge_road_probability = sample_polyline_probability(road_probability, nodes_rc[src_idx], nodes_rc[dst_idx]) if road_probability is not None else 0.0
        edge_center_probability = sample_polyline_probability(center_probability, nodes_rc[src_idx], nodes_rc[dst_idx]) if center_probability is not None else 0.0
        topology_probability = float(topology_probabilities[edge_id]) if topology_probabilities is not None and edge_id < topology_probabilities.size and np.isfinite(topology_probabilities[edge_id]) else edge_center_probability
        if is_bridge and component_node_count >= 4:
            topology_impact = "bridge_connectivity"
        elif is_dangling:
            topology_impact = "dangling_branch"
        else:
            topology_impact = "network_edge"

        if status == "surface_missing":
            conflict_type = "line_without_surface"
        elif status in {"partial_surface", "interpolated_short_gap", "measurement_censored"}:
            conflict_type = "line_surface_uncertain"
        elif row.get("requires_review"):
            conflict_type = "width_quality"
        else:
            conflict_type = "line_surface_agree"

        score = 0
        if conflict_type == "line_without_surface":
            score += 55
        elif conflict_type == "line_surface_uncertain":
            score += 35
        elif conflict_type == "width_quality":
            score += 20
        if topology_impact == "bridge_connectivity":
            score += 35
        elif topology_impact == "dangling_branch":
            score += 15
        if float(row.get("length_px", 0.0)) > 80.0:
            score += 5
        score = min(score, 100)
        auto_retained = bool(
            status == "surface_missing"
            and topology_probability >= auto_retain_center_probability
            and topology_impact in {"bridge_connectivity", "network_edge"}
        )
        if auto_retained:
            row["requires_review"] = False
            row["review_reasons"] = "auto_retained_high_centerline_probability"
        row.update(
            {
                "src_degree": int(degrees[src_idx]),
                "dst_degree": int(degrees[dst_idx]),
                "component_id": component_id,
                "component_node_count": component_node_count,
                "is_bridge": is_bridge,
                "is_dangling_edge": is_dangling,
                "topology_impact": topology_impact,
                "conflict_type": conflict_type,
                "review_priority_score": score,
                "review_priority": "critical" if score >= 80 else ("high" if score >= 60 else ("medium" if score >= 30 else "low")),
                "centerline_policy": "retain_unless_explicitly_deleted",
                "mean_road_probability": edge_road_probability,
                "mean_centerline_probability": edge_center_probability,
                "topology_probability": topology_probability,
                "auto_retained": auto_retained,
            }
        )


def build_edge_surface_evidence(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    binary: np.ndarray,
    supported_ratio: float = 0.8,
    missing_ratio: float = 0.05,
    graph_context: GraphSpatialContext | None = None,
) -> list[dict]:
    """Describe line/surface agreement without estimating road width."""
    context = graph_context or GraphSpatialContext.build(nodes_rc, edges)
    lengths = context.edge_lengths
    rows: list[dict] = []
    surface = binary.astype(np.float32)
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        support = sample_polyline_probability(surface, nodes_rc[src_idx], nodes_rc[dst_idx])
        if support >= supported_ratio:
            status = "surface_supported"
            confidence = "high"
            requires_review = False
            reasons = ""
        elif support <= missing_ratio:
            status = "surface_missing"
            confidence = "low"
            requires_review = True
            reasons = "surface_missing"
        else:
            status = "partial_surface"
            confidence = "low"
            requires_review = True
            reasons = "partial_surface"
        rows.append(
            {
                "edge_id": edge_id,
                "src_idx": int(src_idx),
                "dst_idx": int(dst_idx),
                "length_px": float(lengths[edge_id]),
                "surface_support_ratio": float(support),
                "status": status,
                "confidence": confidence,
                "requires_review": requires_review,
                "review_reasons": reasons,
            }
        )
    return rows


def _candidate_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _candidate_polyline(row: dict) -> np.ndarray:
    fallback = np.asarray(
        [[_candidate_float(row, "start_row"), _candidate_float(row, "start_col")], [_candidate_float(row, "end_row"), _candidate_float(row, "end_col")]],
        dtype=np.float32,
    )
    try:
        points = np.asarray(json.loads(str(row.get("polyline_points_json", ""))), dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError):
        points = fallback
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return fallback
    return points[:, :2]


def _candidate_probability(points: np.ndarray, probability: np.ndarray | None) -> float:
    if probability is None or probability.size == 0:
        return 0.0
    values = []
    for row, col in points:
        rr, cc = int(round(float(row))), int(round(float(col)))
        if 0 <= rr < probability.shape[0] and 0 <= cc < probability.shape[1]:
            values.append(float(probability[rr, cc]))
    return float(np.mean(values)) if values else 0.0


def candidate_score_rule(auto_score: float, hard_veto_reasons: list[str]) -> tuple[str, str]:
    """Classify the published score bands before geometry-specific eligibility."""
    if hard_veto_reasons:
        return "hard_veto", "review"
    if auto_score >= 70.0:
        return "auto_accept", "accept"
    if auto_score >= 50.0:
        return "medium_confidence_auto", "accept"
    return "manual_review", "review"


def _network_attachment_vetoes(hard_veto_reasons: list[str]) -> list[str]:
    """Keep only vetoes that make a graph-attached skeleton unsafe to add.

    Surface skeletons that already meet the graph are valuable continuity
    evidence even when they are short, cross a tile boundary, or have an
    unusual width.  Those properties remain audited, but they no longer keep a
    real network attachment out of the final graph.  Duplicate carriageways,
    loops, block-like skeletons and protected divided-road endpoints still do.
    """
    unsafe = {
        "duplicate_or_near_parallel",
        "protected_divided_carriageway",
        "short_loop",
        "dense_branching_or_blocky",
        "invalid_endpoint_connection",
    }
    return sorted(reason for reason in hard_veto_reasons if reason in unsafe)


def _candidate_endpoint_edge_match(
    points: np.ndarray, nodes_rc: np.ndarray, edges: np.ndarray, max_distance_px: float,
    graph_context: GraphSpatialContext | None = None,
) -> dict:
    """Find a non-parallel endpoint-to-edge (T) attachment, if one exists."""
    context = graph_context or GraphSpatialContext.build(
        nodes_rc, edges, cell_size=max_distance_px,
    )
    best: dict | None = None
    for endpoint_position in (0, 1):
        endpoint = points[endpoint_position]
        inner = points[1] if endpoint_position == 0 else points[-2]
        local_direction = inner - endpoint
        local_norm = float(np.linalg.norm(local_direction))
        if local_norm <= 0:
            continue
        for edge_id in context.edge_index.query_point_radius(endpoint, max_distance_px):
            src_idx, dst_idx = edges[edge_id]
            start, end = nodes_rc[int(src_idx)], nodes_rc[int(dst_idx)]
            projection, ratio, distance = point_segment_projection(endpoint, start, end)
            if not 0.08 <= ratio <= 0.92 or distance > max_distance_px:
                continue
            connector = projection - endpoint
            connector_norm = float(np.linalg.norm(connector))
            tangent = end - start
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-6:
                continue
            if connector_norm <= 1e-6:
                # An endpoint already lying on the edge is an exact T contact.
                # It is safe to audit for later graph noding, unlike a remote
                # projection that would manufacture a connector.
                parallel = abs(float(np.dot(local_direction / local_norm, tangent / tangent_norm)))
                if parallel > 0.85:
                    continue
                proposal = {
                    "endpoint_position": endpoint_position,
                    "edge_id": edge_id,
                    "distance": distance,
                    "projection": projection,
                    "alignment": 1.0,
                    "parallel_cosine": parallel,
                }
                if best is None or proposal["distance"] < best["distance"]:
                    best = proposal
                continue
            # The candidate must run away from the target edge; near-parallel
            # candidates are a duplicate-carriageway risk, not a T junction.
            arrival = -float(np.dot(local_direction / local_norm, connector / connector_norm))
            parallel = abs(float(np.dot(connector / connector_norm, tangent / tangent_norm)))
            if arrival < 0.45 or parallel > 0.85:
                continue
            proposal = {
                "endpoint_position": endpoint_position,
                "edge_id": edge_id,
                "distance": distance,
                "projection": projection,
                "alignment": arrival,
                "parallel_cosine": parallel,
            }
            if best is None or proposal["distance"] < best["distance"]:
                best = proposal
    return best or {}


def _candidate_hard_vetoes(
    row: dict,
    points: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    protected_endpoints: set[int],
    region_candidate_count: int,
    image_shape: tuple[int, int] | None,
    graph_context: GraphSpatialContext | None = None,
) -> list[str]:
    context = graph_context or GraphSpatialContext.build(nodes_rc, edges)
    reasons: list[str] = []
    support = _candidate_float(row, "surface_support_ratio")
    half_width = _candidate_float(row, "median_half_width_px")
    width_cv = _candidate_float(row, "width_cv")
    length = _candidate_float(row, "length_px")
    direct = float(np.linalg.norm(points[-1] - points[0]))
    note = str(row.get("note", "")).lower()
    if support < 0.75:
        reasons.append("low_surface_support")
    if length < 40.0:
        reasons.append("length_too_short")
    if half_width < 2.0 or half_width > 14.0 or width_cv > 0.45:
        reasons.append("width_abnormal")
    if bool(row.get("boundary_truncated", False)) or "boundary_truncated" in note:
        reasons.append("boundary_truncated")
    elif image_shape is not None:
        margin = max(2.0, half_width)
        height, width = image_shape[:2]
        if np.any(points[:, 0] <= margin) or np.any(points[:, 1] <= margin) or np.any(points[:, 0] >= height - 1 - margin) or np.any(points[:, 1] >= width - 1 - margin):
            reasons.append("boundary_truncated")
    if (length > 0 and direct <= max(10.0, 0.20 * length)) or "short_loop" in note:
        reasons.append("short_loop")
    declared_count = int(round(_candidate_float(row, "surface_region_candidate_count", 0.0)))
    branch_count = max(region_candidate_count, declared_count)
    skeleton_density = _candidate_float(row, "skeleton_pixels") / max(length, 1.0)
    if branch_count > 4 or skeleton_density > 2.2 or "blocky" in note or "dense_branch" in note:
        reasons.append("dense_branching_or_blocky")
    if edges.size and points.shape[0] > 2:
        # Ignore the endpoints: a true connector legitimately touches the graph.
        for point in points[1:-1]:
            for edge_id in context.edge_index.query_point_radius(
                point, max(3.0, half_width),
            ):
                src_idx, dst_idx = edges[edge_id]
                projection, ratio, distance = point_segment_projection(point, nodes_rc[int(src_idx)], nodes_rc[int(dst_idx)])
                tangent = nodes_rc[int(dst_idx)] - nodes_rc[int(src_idx)]
                direction = points[-1] - points[0]
                tangent_norm, direction_norm = float(np.linalg.norm(tangent)), float(np.linalg.norm(direction))
                parallel = abs(float(np.dot(tangent, direction) / max(tangent_norm * direction_norm, 1e-6)))
                if 0.05 <= ratio <= 0.95 and distance <= max(3.0, half_width) and parallel >= 0.90:
                    reasons.append("duplicate_or_near_parallel")
                    break
            if "duplicate_or_near_parallel" in reasons:
                break
    # A candidate directly pairing protected divided-road dangling endpoints is
    # prohibited even when the geometry itself appears straight.
    for endpoint in (points[0], points[-1]):
        if nodes_rc.size:
            distance, nearest = context.nearest_node(endpoint)
            if nearest in protected_endpoints and distance <= 120.0:
                reasons.append("protected_divided_carriageway")
                break
    return sorted(set(reasons))


def annotate_candidate_graph_matches(
    candidate_rows: list[dict],
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    snap_px: float,
    auto_fuse_surface_components: bool = True,
    auto_extend_surface_skeletons: bool = True,
    surface_extension_min_alignment: float = 0.65,
    surface_extension_max_distance_px: float = 120.0,
    surface_extension_min_length_px: float = 40.0,
    surface_extension_max_length_px: float = 0.0,
    surface_extension_min_support_ratio: float = 0.85,
    surface_extension_max_half_width_px: float = 14.0,
    center_probability: np.ndarray | None = None,
    image_shape: tuple[int, int] | None = None,
    graph_context: GraphSpatialContext | None = None,
) -> None:
    context = graph_context or GraphSpatialContext.build(
        nodes_rc, edges, cell_size=max(snap_px, surface_extension_max_distance_px),
    )
    component_ids = context.component_ids
    protected_endpoints = divided_road_endpoint_ids(
        nodes_rc, edges, graph_context=context,
    )
    outward_vectors = context.outward_vectors
    surface_region_candidate_counts: dict[str, int] = {}
    for candidate in candidate_rows:
        if candidate.get("candidate_type") == "surface_skeleton":
            region_id = str(candidate.get("region_id", "") or "")
            if region_id:
                surface_region_candidate_counts[region_id] = surface_region_candidate_counts.get(region_id, 0) + 1
    for row in candidate_rows:
        candidate_points = _candidate_polyline(row)
        endpoints = np.asarray([candidate_points[0], candidate_points[-1]], dtype=np.float32)
        matched: list[int] = []
        distances: list[float] = []
        for endpoint in endpoints:
            distance, node_idx = context.nearest_node(endpoint)
            matched.append(node_idx if distance <= snap_px else -1)
            distances.append(distance)
        matched_components = {int(component_ids[node_idx]) for node_idx in matched if node_idx >= 0}
        if len(matched_components) >= 2:
            impact = "connect_components"
        elif sum(node_idx >= 0 for node_idx in matched) == 2:
            impact = "close_local_loop"
        elif any(node_idx >= 0 for node_idx in matched):
            impact = "extend_existing_network"
        else:
            impact = "isolated_surface_candidate"
        row.update(
            {
                "matched_start_node_idx": matched[0] if matched[0] >= 0 else "",
                "matched_end_node_idx": matched[1] if matched[1] >= 0 else "",
                "start_match_distance_px": distances[0],
                "end_match_distance_px": distances[1],
                "matched_component_count": len(matched_components),
                "topology_impact": impact,
            }
        )
        candidate_length_px = float(row.get("length_px", 0.0) or 0.0)
        if candidate_length_px <= 0:
            candidate_length_px = float(np.linalg.norm(np.diff(candidate_points, axis=0), axis=1).sum())
        relaxed_matches: list[tuple[int, int, float]] = []
        for endpoint_position, endpoint in enumerate(endpoints):
            distance, node_idx = context.nearest_endpoint(endpoint)
            if node_idx >= 0 and distance <= surface_extension_max_distance_px:
                relaxed_matches.append((endpoint_position, node_idx, distance))
        # A long candidate can have both ends inside the attachment radius of
        # the same dangling endpoint.  Collapse only that ambiguous case.  Two
        # different endpoint matches must be retained because they may bridge
        # two disconnected graph components.
        if len(relaxed_matches) > 1 and len({item[1] for item in relaxed_matches}) == 1:
            relaxed_matches = [min(relaxed_matches, key=lambda item: item[2])]
        extension_alignment = -1.0
        extension_node_idx = -1
        extension_distance = -1.0
        exact_matches = [(position, node_idx, distances[position]) for position, node_idx in enumerate(matched) if node_idx >= 0]
        preferred_matches = exact_matches or relaxed_matches
        if preferred_matches:
            endpoint_position, extension_node_idx, extension_distance = min(preferred_matches, key=lambda item: item[2])
            candidate_direction = candidate_points[1] - candidate_points[0]
            if endpoint_position == 1:
                candidate_direction = candidate_points[-2] - candidate_points[-1]
            candidate_length = float(np.linalg.norm(candidate_direction))
            outward = outward_vectors.get(extension_node_idx)
            if candidate_length > 0 and outward is not None:
                extension_alignment = float(np.dot(outward, candidate_direction / candidate_length))
        row["single_endpoint_extension_node_idx"] = extension_node_idx if extension_node_idx >= 0 else ""
        row["single_endpoint_extension_distance_px"] = extension_distance
        row["single_endpoint_extension_alignment"] = extension_alignment
        row["single_endpoint_extension_length_px"] = candidate_length_px
        region_id = str(row.get("region_id", "") or "")
        surface_region_candidate_count = max(surface_region_candidate_counts.get(region_id, 1), int(round(_candidate_float(row, "surface_region_candidate_count", 0.0))))
        row["surface_region_candidate_count"] = surface_region_candidate_count
        edge_match = _candidate_endpoint_edge_match(
            candidate_points, nodes_rc, edges, surface_extension_max_distance_px,
            graph_context=context,
        )
        center_support = _candidate_probability(candidate_points, center_probability)
        direct_length = float(np.linalg.norm(candidate_points[-1] - candidate_points[0]))
        tortuosity = candidate_length_px / max(direct_length, 1.0)
        half_width = _candidate_float(row, "median_half_width_px")
        width_cv = _candidate_float(row, "width_cv")
        hard_veto_reasons = _candidate_hard_vetoes(
            row, candidate_points, nodes_rc, edges, protected_endpoints,
            surface_region_candidate_count, image_shape, graph_context=context,
        )
        connection_type = "isolated_simple_road"
        connection_score = 8.0
        direction_score = 10.0 if tortuosity <= 1.15 else 0.0
        connection_alignment = 0.0
        relaxed_component_ids = {
            int(component_ids[node_idx]) for _, node_idx, _ in relaxed_matches if node_idx >= 0
        }
        if len(matched_components) >= 2 or len(relaxed_component_ids) >= 2:
            connection_type = "component_bridge"
            connection_score = 20.0
            direction_score = 15.0
        elif exact_matches:
            # Exact graph contact is already a valid attachment.  Do not make
            # it pass the looser remote-endpoint direction and width tests.
            connection_type = "endpoint_to_node"
            connection_score = 20.0
            direction_score = 15.0
            connection_alignment = max(extension_alignment, 0.0)
        elif (
            len(relaxed_matches) == 1
            and extension_node_idx >= 0
            and extension_node_idx not in protected_endpoints
            and extension_alignment >= surface_extension_min_alignment
            and _candidate_float(row, "surface_support_ratio") >= surface_extension_min_support_ratio
            and 2.0 <= half_width <= surface_extension_max_half_width_px
        ):
            connection_type = "endpoint_to_node"
            connection_score = 20.0
            direction_score = 15.0
            connection_alignment = extension_alignment
        elif edge_match and _candidate_float(row, "surface_support_ratio") >= surface_extension_min_support_ratio:
            connection_type = "endpoint_to_edge"
            connection_score = 18.0
            direction_score = 15.0
            connection_alignment = float(edge_match["alignment"])
        attachment_matches: dict[int, tuple[int, float]] = {}
        for position, node_idx, distance in relaxed_matches:
            attachment_matches[position] = (node_idx, distance)
        for position, node_idx, distance in exact_matches:
            attachment_matches[position] = (node_idx, distance)
        if relaxed_matches and connection_type == "isolated_simple_road" and min(item[2] for item in relaxed_matches) <= max(snap_px, 5.0):
            # A nearby graph endpoint that fails the outward-direction test is
            # neither an isolated road nor a safe connector.
            hard_veto_reasons = sorted(set(hard_veto_reasons + ["invalid_endpoint_connection"]))
        simple_isolated = (
            connection_type == "isolated_simple_road"
            and candidate_length_px >= 100.0
            and candidate_length_px / max(2.0 * half_width, 1.0) >= 5.0
            and tortuosity <= 1.25
        )
        if connection_type == "isolated_simple_road" and not simple_isolated:
            direction_score = 0.0
        surface_score = 25.0 if _candidate_float(row, "surface_support_ratio") >= 0.90 else 15.0
        shape_score = 20.0 if 2.0 <= half_width <= 14.0 and width_cv <= 0.45 and tortuosity <= 1.35 else 0.0
        center_score = 10.0 if center_support >= 0.35 else (5.0 if center_support >= 0.15 else 0.0)
        length_score = 10.0 if candidate_length_px >= 40.0 else 0.0
        auto_score = surface_score + shape_score + connection_score + direction_score + center_score + length_score
        long_road_evidence = bool(
            candidate_length_px > 600.0
            and _candidate_float(row, "surface_support_ratio") >= 0.90
            and 2.0 <= half_width <= 14.0
            and width_cv <= 0.45
            and tortuosity <= 1.10
        )
        long_road_evidence_reason = "long_continuous_linear_surface_supported" if long_road_evidence else ""
        effective_veto_reasons = hard_veto_reasons
        if connection_type != "isolated_simple_road":
            effective_veto_reasons = _network_attachment_vetoes(hard_veto_reasons)
        auto_rule, auto_decision = candidate_score_rule(auto_score, effective_veto_reasons)
        if connection_type == "isolated_simple_road":
            # Isolated surface-only roads (commonly field tracks) are not
            # inserted into the authoritative road network automatically.
            auto_rule, auto_decision = "isolated_excluded_from_network", "review"
        elif not effective_veto_reasons:
            # Network continuity is the primary policy: once the skeleton has a
            # real node/edge/component attachment, retain it by default.  The
            # score and non-safety warnings remain available for audit.
            auto_rule, auto_decision = "network_connected_default", "accept"
        row.update(
            {
                "auto_score": float(auto_score),
                "auto_rule": auto_rule,
                "auto_score_surface": surface_score,
                "auto_score_shape_width": shape_score,
                "auto_score_connection": connection_score,
                "auto_score_direction_junction": direction_score,
                "auto_score_centerline_weak": center_score,
                "auto_score_continuity_length": length_score,
                "hard_veto": bool(effective_veto_reasons),
                "hard_veto_reasons": ";".join(hard_veto_reasons),
                "effective_veto_reasons": ";".join(effective_veto_reasons),
                "connection_type": connection_type,
                "connection_alignment": float(connection_alignment),
                "candidate_tortuosity": float(tortuosity),
                "candidate_center_probability": float(center_support),
                "candidate_width_cv": float(width_cv),
                "long_road_evidence": long_road_evidence,
                "long_road_evidence_reason": long_road_evidence_reason,
                "connection_endpoint_position": edge_match.get("endpoint_position", "") if edge_match else (endpoint_position if extension_node_idx >= 0 else ""),
                "connection_edge_id": edge_match.get("edge_id", "") if edge_match else "",
                "connection_projection_row": float(edge_match["projection"][0]) if edge_match else "",
                "connection_projection_col": float(edge_match["projection"][1]) if edge_match else "",
                "connection_node_idx": extension_node_idx if connection_type == "endpoint_to_node" else "",
                "connection_node_row": float(nodes_rc[extension_node_idx][0]) if connection_type == "endpoint_to_node" else "",
                "connection_node_col": float(nodes_rc[extension_node_idx][1]) if connection_type == "endpoint_to_node" else "",
                "connection_start_node_row": float(nodes_rc[attachment_matches[0][0]][0]) if connection_type == "component_bridge" and 0 in attachment_matches else "",
                "connection_start_node_col": float(nodes_rc[attachment_matches[0][0]][1]) if connection_type == "component_bridge" and 0 in attachment_matches else "",
                "connection_end_node_row": float(nodes_rc[attachment_matches[1][0]][0]) if connection_type == "component_bridge" and 1 in attachment_matches else "",
                "connection_end_node_col": float(nodes_rc[attachment_matches[1][0]][1]) if connection_type == "component_bridge" and 1 in attachment_matches else "",
            }
        )
        if row.get("candidate_type") == "surface_skeleton":
            # Preserve only contacts that already coincide with the graph.  A
            # scorer's relaxed endpoint/edge match is audit information, not
            # permission for the finalizer to draw a remote straight link.
            endpoint_position = edge_match.get("endpoint_position", "") if edge_match else ""
            if (
                edge_match
                and float(edge_match.get("distance", float("inf"))) <= 1.5
                and _candidate_float(row, "surface_support_ratio") >= 0.95
                and endpoint_position in {0, 1}
            ):
                row.update({
                    "surface_attachment_audit": "exact_surface_contact",
                    "surface_attachment_kind": "surface_skeleton_to_graph_edge",
                    "surface_attachment_endpoint_position": endpoint_position,
                    "surface_attachment_projection_row": float(edge_match["projection"][0]),
                    "surface_attachment_projection_col": float(edge_match["projection"][1]),
                    "surface_attachment_surface_support_ratio": _candidate_float(row, "surface_support_ratio"),
                    "surface_attachment_direction_alignment": float(edge_match["alignment"]),
                    "surface_attachment_path_ratio": 1.0,
                    "surface_attachment_evidence_mode": "continuous_surface_mask",
                })
            elif (
                connection_type == "endpoint_to_node"
                and extension_node_idx >= 0
                and str(row.get("connection_endpoint_position", "")) in {"0", "1"}
                and extension_distance <= 1.5
                and _candidate_float(row, "surface_support_ratio") >= 0.95
            ):
                row.update({
                    "surface_attachment_audit": "exact_surface_contact",
                    "surface_attachment_kind": "surface_skeleton_to_graph_endpoint",
                    "surface_attachment_endpoint_position": int(float(row["connection_endpoint_position"])),
                    "surface_attachment_node_row": float(nodes_rc[extension_node_idx][0]),
                    "surface_attachment_node_col": float(nodes_rc[extension_node_idx][1]),
                    "surface_attachment_surface_support_ratio": _candidate_float(row, "surface_support_ratio"),
                    "surface_attachment_direction_alignment": max(float(connection_alignment), 0.0),
                    "surface_attachment_path_ratio": 1.0,
                    "surface_attachment_evidence_mode": "continuous_surface_mask",
                })
            # Product policy: every centerline produced by road-surface
            # skeletonization is retained exactly as traced.  Graph matching
            # remains audit-only and must never create a straight connector to
            # a guessed remote node or edge.
            row["auto_decision"] = "accept" if auto_extend_surface_skeletons else "review"
            row["auto_rule"] = "all_surface_skeletons_as_is"
            row["action"] = "auto_add_surface_skeleton_as_is" if row["auto_decision"] == "accept" else "propose_add_centerline"
            row["hard_veto"] = False
            row["effective_veto_reasons"] = ""
            if row["auto_decision"] == "accept":
                row["confidence"] = "medium"


def build_width_change_segments(
    sample_rows: list[dict], pixel_size: float, change_ratio: float, min_samples: int
) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in sample_rows:
        if float(row.get("width_px", 0.0)) > 0:
            grouped.setdefault(int(row["edge_id"]), []).append(row)
    segments: list[dict] = []
    for edge_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["sample_index"]))
        runs: list[list[dict]] = [[rows[0]]]
        for row in rows[1:]:
            baseline = float(np.median([float(item["width_px"]) for item in runs[-1]]))
            relative_change = abs(float(row["width_px"]) - baseline) / max(baseline, 1.0)
            if relative_change >= change_ratio and len(runs[-1]) >= min_samples:
                runs.append([row])
            else:
                runs[-1].append(row)
        for local_segment_id, run in enumerate(runs):
            widths = np.asarray([float(row["width_px"]) for row in run], dtype=np.float32)
            if widths.size >= 5:
                lower, upper = np.percentile(widths, [10, 90])
                trimmed = widths[(widths >= lower) & (widths <= upper)]
            else:
                trimmed = widths
            segments.append(
                {
                    "width_segment_id": len(segments),
                    "edge_id": edge_id,
                    "edge_segment_index": local_segment_id,
                    "start_sample_index": int(run[0]["sample_index"]),
                    "end_sample_index": int(run[-1]["sample_index"]),
                    "sample_count": len(run),
                    "start_row": run[0]["row_used"],
                    "start_col": run[0]["col_used"],
                    "end_row": run[-1]["row_used"],
                    "end_col": run[-1]["col_used"],
                    "median_width_px": float(np.median(widths)),
                    "trimmed_mean_width_px": float(np.mean(trimmed)),
                    "median_width_units": float(np.median(widths) * pixel_size),
                    "width_cv": float(np.std(widths) / max(float(np.median(widths)), 1.0)),
                }
            )
    return segments


def build_conflict_review_rows(
    edge_rows: list[dict], surface_rows: list[dict], candidate_rows: list[dict]
) -> list[dict]:
    records: list[dict] = []
    candidate_region_ids = {str(row.get("region_id", "")) for row in candidate_rows if row.get("region_id", "") != ""}
    for row in edge_rows:
        if not row.get("requires_review"):
            continue
        records.append(
            {
                "item_type": "edge_review",
                "item_id": row["edge_id"],
                "conflict_type": row.get("conflict_type", ""),
                "topology_impact": row.get("topology_impact", ""),
                "priority_score": row.get("review_priority_score", 0),
                "priority": row.get("review_priority", "low"),
                "auto_action": "retain_centerline",
                "requires_manual_review": False,
                "confidence": row.get("confidence", ""),
                "reason": row.get("review_reasons", ""),
                "region_id": "",
                "edge_id": row["edge_id"],
                "candidate_id": "",
            }
        )
    for row in candidate_rows:
        auto_accept = row.get("auto_decision") == "accept"
        score = int(round(float(row.get("topology_score", 0.0) or 0.0) * 100)) if auto_accept else (70 if row.get("confidence") == "medium" else 45)
        records.append(
            {
                "item_type": "candidate_centerline",
                "item_id": row["candidate_id"],
                "conflict_type": "surface_without_line" if row.get("candidate_type") == "surface_skeleton" else "centerline_gap",
                "topology_impact": row.get("topology_impact", ""),
                "priority_score": score,
                "priority": "auto" if auto_accept else ("high" if score >= 60 else "medium"),
                "auto_action": "add_centerline" if auto_accept else "propose_centerline",
                "requires_manual_review": not auto_accept,
                "confidence": row.get("confidence", ""),
                "reason": row.get("note", ""),
                "region_id": row.get("region_id", ""),
                "edge_id": "",
                "candidate_id": row["candidate_id"],
            }
        )
    for row in surface_rows:
        region_id = str(row.get("region_id", ""))
        if region_id in candidate_region_ids:
            continue
        status = str(row.get("status", ""))
        score = 65 if status == "surface_only_candidate" else (40 if status == "surface_only_review" else 20)
        records.append(
            {
                "item_type": "surface_only_region",
                "item_id": region_id,
                "conflict_type": "surface_without_line",
                "topology_impact": "unconnected_surface_region",
                "priority_score": score,
                "priority": "high" if score >= 60 else ("medium" if score >= 30 else "low"),
                "auto_action": "review_surface_region",
                "requires_manual_review": True,
                "confidence": "low",
                "reason": row.get("note", ""),
                "region_id": region_id,
                "edge_id": "",
                "candidate_id": "",
            }
        )
    records.sort(key=lambda row: (-int(row["priority_score"]), str(row["item_type"]), str(row["item_id"])))
    for rank, row in enumerate(records, start=1):
        row["review_rank"] = rank
    return records


def endpoint_outward_vectors(nodes_rc: np.ndarray, edges: np.ndarray) -> dict[int, np.ndarray]:
    degrees, node_edges = graph_degrees(edges.shape[0], nodes_rc.shape[0], edges)
    vectors: dict[int, np.ndarray] = {}
    for node_idx in np.where(degrees == 1)[0].tolist():
        src_idx, dst_idx = edges[node_edges[node_idx][0]]
        neighbor_idx = int(dst_idx if int(src_idx) == node_idx else src_idx)
        vector = nodes_rc[node_idx] - nodes_rc[neighbor_idx]
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vectors[node_idx] = vector / norm
    return vectors


def divided_road_endpoint_ids(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    min_spacing_px: float = 5.0,
    max_spacing_px: float = 40.0,
    min_parallel_cosine: float = 0.90,
    max_lateral_cosine: float = 0.55,
    graph_context: GraphSpatialContext | None = None,
) -> set[int]:
    """Return dangling endpoints that form a plausible parallel-road pair."""
    context = graph_context or GraphSpatialContext.build(nodes_rc, edges, cell_size=max_spacing_px)
    outward = context.outward_vectors
    protected: set[int] = set()
    endpoint_ids = sorted(outward)
    endpoint_set = set(endpoint_ids)
    for first_idx in endpoint_ids:
        for second_idx in context.node_index.query_radius_box(nodes_rc[first_idx], max_spacing_px):
            if second_idx <= first_idx or second_idx not in endpoint_set:
                continue
            connector = nodes_rc[second_idx] - nodes_rc[first_idx]
            spacing = float(np.linalg.norm(connector))
            if not min_spacing_px <= spacing <= max_spacing_px:
                continue
            first_outward, second_outward = outward[first_idx], outward[second_idx]
            if float(np.dot(first_outward, second_outward)) < min_parallel_cosine:
                continue
            lateral = connector / max(spacing, 1e-6)
            if max(abs(float(np.dot(lateral, first_outward))), abs(float(np.dot(lateral, second_outward)))) > max_lateral_cosine:
                continue
            protected.update({first_idx, second_idx})
    return protected


def endpoint_alignment_score(
    start: np.ndarray,
    end: np.ndarray,
    start_outward: np.ndarray,
    end_outward: np.ndarray,
) -> float:
    connector = end - start
    length = float(np.linalg.norm(connector))
    if length <= 0:
        return -1.0
    direction = connector / length
    return float(min(np.dot(start_outward, direction), np.dot(end_outward, -direction)))


def point_segment_projection(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float, float]:
    vector = end - start
    length2 = float(np.dot(vector, vector))
    ratio = 0.0 if length2 <= 0 else float(np.clip(np.dot(point - start, vector) / length2, 0.0, 1.0))
    projection = start + ratio * vector
    return projection, ratio, float(np.linalg.norm(point - projection))


def repair_endpoint_to_edge_junctions(
    binary: np.ndarray,
    road_probability: np.ndarray,
    center_probability: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    max_distance_px: float,
    min_alignment_cosine: float,
    min_surface_support: float,
    path_margin_px: int,
    outside_cost: float,
    sample_step_px: float,
    max_target_parallel_cosine: float = 0.75,
    graph_context: GraphSpatialContext | None = None,
    profiling: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if max_distance_px <= 0 or edges.size == 0:
        return nodes_rc, edges, []
    context = graph_context or GraphSpatialContext.build(
        nodes_rc, edges, cell_size=max_distance_px,
    )
    outward_vectors = context.outward_vectors
    protected_endpoints = divided_road_endpoint_ids(
        nodes_rc, edges, graph_context=context,
    )
    component_ids = context.component_ids
    proposals = []
    for endpoint_idx, outward in outward_vectors.items():
        if endpoint_idx in protected_endpoints:
            continue
        endpoint = nodes_rc[endpoint_idx]
        for edge_id in context.edge_index.query_point_radius(endpoint, max_distance_px):
            src_idx, dst_idx = edges[edge_id]
            src_idx, dst_idx = int(src_idx), int(dst_idx)
            if endpoint_idx in {src_idx, dst_idx} or component_ids[endpoint_idx] == component_ids[src_idx]:
                continue
            projection, ratio, distance = point_segment_projection(endpoint, nodes_rc[src_idx], nodes_rc[dst_idx])
            if not 0.15 <= ratio <= 0.85 or distance < 2.0 or distance > max_distance_px:
                continue
            direction = (projection - endpoint) / max(distance, 1e-6)
            alignment = float(np.dot(outward, direction))
            if alignment < min_alignment_cosine:
                continue
            target_neighbors = []
            for target_node_idx in (src_idx, dst_idx):
                for target_edge_id in context.node_edges[target_node_idx]:
                    if target_edge_id == edge_id:
                        continue
                    edge_src, edge_dst = (int(value) for value in edges[target_edge_id])
                    if edge_src == target_node_idx:
                        target_neighbors.append(nodes_rc[edge_dst])
                    elif edge_dst == target_node_idx:
                        target_neighbors.append(nodes_rc[edge_src])
            target_start = target_neighbors[0] if len(target_neighbors) >= 1 else nodes_rc[src_idx]
            target_end = target_neighbors[-1] if len(target_neighbors) >= 2 else nodes_rc[dst_idx]
            target_tangent = target_end - target_start
            target_tangent_norm = float(np.linalg.norm(target_tangent))
            if target_tangent_norm > 0:
                parallel_cosine = abs(float(np.dot(outward, target_tangent / target_tangent_norm)))
                if parallel_cosine > max_target_parallel_cosine:
                    continue
            path, support_ratio = shortest_mask_path(
                binary,
                tuple(endpoint),
                tuple(projection),
                path_margin_px,
                outside_cost,
                road_probability=road_probability,
                center_probability=center_probability,
                profiling=profiling,
            )
            if not path or support_ratio < min_surface_support:
                continue
            score = 0.5 * alignment + 0.35 * support_ratio + 0.15 * (1.0 - distance / max_distance_px)
            proposals.append((score, endpoint_idx, edge_id, projection, path, distance, alignment, support_ratio))

    final_nodes = [tuple(float(value) for value in node) for node in nodes_rc.tolist()]
    final_edges = [tuple(int(value) for value in edge) for edge in edges.tolist()]
    used_endpoints: set[int] = set()
    used_edges: set[int] = set()
    audit_rows = []
    for score, endpoint_idx, edge_id, projection, path, distance, alignment, support_ratio in sorted(proposals, reverse=True):
        if endpoint_idx in used_endpoints or edge_id in used_edges:
            continue
        original_edge = tuple(int(value) for value in edges[edge_id])
        try:
            final_edges.remove(original_edge)
        except ValueError:
            try:
                final_edges.remove((original_edge[1], original_edge[0]))
            except ValueError:
                continue
        junction_idx = len(final_nodes)
        final_nodes.append((float(projection[0]), float(projection[1])))
        final_edges.extend([(original_edge[0], junction_idx), (junction_idx, original_edge[1])])
        sampled = resample_polyline(path, sample_step_px)
        previous_idx = endpoint_idx
        for point in sampled[1:-1]:
            node_idx = len(final_nodes)
            final_nodes.append((float(point[0]), float(point[1])))
            final_edges.append((previous_idx, node_idx))
            previous_idx = node_idx
        final_edges.append((previous_idx, junction_idx))
        used_endpoints.add(endpoint_idx)
        used_edges.add(edge_id)
        audit_rows.append(
            {
                "junction_repair_id": len(audit_rows),
                "endpoint_node_idx": endpoint_idx,
                "target_edge_id": edge_id,
                "junction_row": float(projection[0]),
                "junction_col": float(projection[1]),
                "gap_distance_px": distance,
                "direction_alignment": alignment,
                "surface_support_ratio": support_ratio,
                "topology_score": score,
                "action": "split_edge_and_connect_endpoint",
            }
        )
    final_nodes_array = np.asarray(final_nodes, dtype=np.float32).reshape(-1, 2)
    final_edges_array = np.asarray(final_edges, dtype=np.int32).reshape(-1, 2)
    return compact_graph(final_nodes_array, final_edges_array) + (audit_rows,)


@_accumulate_profile("a_star_path_search_seconds")
def shortest_mask_path(
    binary: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    margin_px: int,
    outside_cost: float,
    road_probability: np.ndarray | None = None,
    center_probability: np.ndarray | None = None,
    road_weight: float = 0.7,
    center_weight: float = 0.3,
    max_expansions: int = 250000,
    profiling: dict | None = None,
) -> tuple[list[tuple[float, float]], float]:
    height, width = binary.shape
    start_rc = (int(round(start[0])), int(round(start[1])))
    end_rc = (int(round(end[0])), int(round(end[1])))
    if not (0 <= start_rc[0] < height and 0 <= start_rc[1] < width):
        return [], 0.0
    if not (0 <= end_rc[0] < height and 0 <= end_rc[1] < width):
        return [], 0.0

    y0 = max(0, min(start_rc[0], end_rc[0]) - margin_px)
    y1 = min(height, max(start_rc[0], end_rc[0]) + margin_px + 1)
    x0 = max(0, min(start_rc[1], end_rc[1]) - margin_px)
    x1 = min(width, max(start_rc[1], end_rc[1]) + margin_px + 1)
    local_start = (start_rc[0] - y0, start_rc[1] - x0)
    local_end = (end_rc[0] - y0, end_rc[1] - x0)
    local_mask = binary[y0:y1, x0:x1]
    local_road = road_probability[y0:y1, x0:x1] if road_probability is not None else local_mask.astype(np.float32)
    local_center = center_probability[y0:y1, x0:x1] if center_probability is not None else np.zeros_like(local_road)
    movements = [
        (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, np.sqrt(2.0)),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, np.sqrt(2.0)), (1, 0, 1.0), (1, 1, np.sqrt(2.0)),
    ]
    best_cost = {local_start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(float(np.hypot(local_end[0] - local_start[0], local_end[1] - local_start[1])), 0.0, local_start)]
    expansions = 0
    reached = False
    while queue and expansions < max_expansions:
        _, cost, current = heapq.heappop(queue)
        if cost > best_cost.get(current, float("inf")):
            continue
        if current == local_end:
            reached = True
            break
        expansions += 1
        for dr, dc, step_length in movements:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < local_mask.shape[0] and 0 <= neighbor[1] < local_mask.shape[1]):
                continue
            evidence_cost = road_weight * (1.0 - float(local_road[neighbor])) + center_weight * (1.0 - float(local_center[neighbor]))
            obstacle_cost = 0.0 if local_mask[neighbor] > 0 else max(1.0, outside_cost)
            pixel_cost = 1.0 + evidence_cost + obstacle_cost
            new_cost = cost + step_length * pixel_cost
            if new_cost >= best_cost.get(neighbor, float("inf")):
                continue
            best_cost[neighbor] = new_cost
            parents[neighbor] = current
            heuristic = float(np.hypot(local_end[0] - neighbor[0], local_end[1] - neighbor[1]))
            heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))
    if not reached:
        return [], 0.0

    local_path = [local_end]
    while local_path[-1] != local_start:
        local_path.append(parents[local_path[-1]])
    local_path.reverse()
    support_ratio = float(np.mean([local_mask[row, col] > 0 for row, col in local_path]))
    return [(float(row + y0), float(col + x0)) for row, col in local_path], support_ratio


def build_endpoint_gap_candidates(
    binary: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    max_gap_px: float,
    min_alignment_cosine: float,
    min_surface_support: float,
    path_margin_px: int,
    outside_cost: float,
    sample_step_px: float,
    road_probability: np.ndarray | None = None,
    center_probability: np.ndarray | None = None,
    max_path_ratio: float = 1.15,
    ambiguity_ratio: float = 1.20,
    excluded_endpoint_ids: set[int] | None = None,
    graph_context: GraphSpatialContext | None = None,
    profiling: dict | None = None,
) -> list[dict]:
    if max_gap_px <= 0:
        return []
    road_probability = road_probability if road_probability is not None else binary.astype(np.float32)
    center_probability = center_probability if center_probability is not None else np.zeros_like(road_probability)
    context = graph_context or GraphSpatialContext.build(
        nodes_rc, edges, cell_size=max_gap_px,
    )
    outward_vectors = context.outward_vectors
    protected_endpoints = divided_road_endpoint_ids(
        nodes_rc, edges, graph_context=context,
    )
    excluded_endpoint_ids = excluded_endpoint_ids or set()
    component_ids = context.component_ids
    geometric_pairs: list[tuple[int, int, float, float]] = []
    endpoint_neighbors: dict[int, list[tuple[float, int]]] = {node_idx: [] for node_idx in outward_vectors}
    endpoint_ids = sorted(outward_vectors)
    endpoint_set = set(endpoint_ids)
    for start_idx in endpoint_ids:
        if start_idx in protected_endpoints or start_idx in excluded_endpoint_ids:
            continue
        for end_idx in context.node_index.query_radius_box(nodes_rc[start_idx], max_gap_px):
            if end_idx <= start_idx or end_idx not in endpoint_set:
                continue
            if end_idx in protected_endpoints or end_idx in excluded_endpoint_ids:
                continue
            if component_ids[start_idx] == component_ids[end_idx]:
                continue
            distance = float(np.linalg.norm(nodes_rc[end_idx] - nodes_rc[start_idx]))
            if distance < 2.0 or distance > max_gap_px:
                continue
            alignment = endpoint_alignment_score(
                nodes_rc[start_idx], nodes_rc[end_idx], outward_vectors[start_idx], outward_vectors[end_idx]
            )
            if alignment < min_alignment_cosine:
                continue
            geometric_pairs.append((start_idx, end_idx, distance, alignment))
            endpoint_neighbors[start_idx].append((distance, end_idx))
            endpoint_neighbors[end_idx].append((distance, start_idx))

    proposals: list[tuple[float, int, int, list[tuple[float, float]], float, float, float]] = []
    for start_idx, end_idx, distance, alignment in geometric_pairs:
        start_neighbors = sorted(endpoint_neighbors[start_idx])
        end_neighbors = sorted(endpoint_neighbors[end_idx])
        if start_neighbors[0][1] != end_idx or end_neighbors[0][1] != start_idx:
            continue
        if len(start_neighbors) > 1 and start_neighbors[1][0] < distance * ambiguity_ratio:
            continue
        if len(end_neighbors) > 1 and end_neighbors[1][0] < distance * ambiguity_ratio:
            continue
        path, support_ratio = shortest_mask_path(
                binary,
                tuple(nodes_rc[start_idx]),
                tuple(nodes_rc[end_idx]),
                margin_px=path_margin_px,
                outside_cost=outside_cost,
                road_probability=road_probability,
                center_probability=center_probability,
                profiling=profiling,
        )
        if not path or support_ratio < min_surface_support:
            continue
        path_length = float(sum(
            np.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(path[:-1], path[1:])
        ))
        path_ratio = path_length / max(distance, 1e-6)
        if path_ratio > max_path_ratio:
            continue
        score = 0.45 * alignment + 0.35 * support_ratio + 0.20 * (1.0 - distance / max_gap_px)
        proposals.append((score, start_idx, end_idx, path, support_ratio, alignment, path_ratio))

    candidates = []
    used_endpoints: set[int] = set()
    for score, start_idx, end_idx, path, support_ratio, alignment, path_ratio in sorted(
        proposals, key=lambda item: item[0], reverse=True
    ):
        if start_idx in used_endpoints or end_idx in used_endpoints:
            continue
        sampled = resample_polyline(path, sample_step_px)
        if len(sampled) < 2:
            continue
        used_endpoints.update({start_idx, end_idx})
        candidates.append(
            {
                "candidate_id": -1,
                "region_id": "",
                "branch_index": 0,
                "candidate_type": "endpoint_gap",
                "review_status": "auto_accept",
                "action": "auto_add_centerline",
                "confidence": "high",
                "auto_decision": "accept",
                "topology_impact": "connect_components",
                "topology_score": score,
                "surface_support_ratio": support_ratio,
                "direction_alignment": alignment,
                "path_ratio": path_ratio,
                "start_node_idx": start_idx,
                "end_node_idx": end_idx,
                "start_row": sampled[0][0],
                "start_col": sampled[0][1],
                "end_row": sampled[-1][0],
                "end_col": sampled[-1][1],
                "length_px": float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(sampled[:-1], sampled[1:]))),
                "skeleton_pixels": len(path),
                "point_count": len(sampled),
                "polyline_points_json": json.dumps([[row, col] for row, col in sampled], separators=(",", ":")),
                "area_px": "",
                "note": "high_confidence_endpoint_gap_on_road_surface",
            }
        )
    return candidates


def connect_surface_skeletons_by_mask_path(
    candidate_rows: list[dict],
    binary: np.ndarray,
    road_probability: np.ndarray,
    center_probability: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    max_distance_px: float = 96.0,
    min_alignment_cosine: float = 0.85,
    min_surface_support: float = 0.95,
    max_path_ratio: float = 1.35,
    path_margin_px: int = 24,
    outside_cost: float = 30.0,
    sample_step_px: float = 6.0,
    fallback_min_alignment: float = 0.95,
    fallback_min_road_probability: float = 0.50,
    fallback_min_road_probability_q25: float = 0.10,
    fallback_max_path_ratio: float = 1.15,
    graph_context: GraphSpatialContext | None = None,
    profiling: dict | None = None,
) -> list[dict]:
    """Extend surface skeleton endpoints only along strongly supported road pixels.

    This deliberately targets degree-1 graph endpoints only.  It cannot draw a
    straight endpoint-to-edge shortcut, and the accepted A* path must agree
    with both endpoint tangents, remain almost entirely on the road surface,
    and be only modestly longer than the Euclidean gap.
    """
    if max_distance_px <= 0 or nodes_rc.size == 0 or edges.size == 0:
        return []
    context = graph_context or GraphSpatialContext.build(
        nodes_rc, edges, cell_size=max_distance_px,
    )
    outward_vectors = context.outward_vectors
    protected_endpoints = divided_road_endpoint_ids(
        nodes_rc, edges, graph_context=context,
    )
    proposals: list[tuple[float, int, int, int, list[tuple[float, float]], float, float, float]] = []
    for candidate_index, candidate in enumerate(candidate_rows):
        if candidate.get("candidate_type") != "surface_skeleton":
            continue
        points = _candidate_polyline(candidate)
        if points.shape[0] < 2:
            continue
        for endpoint_position in (0, 1):
            endpoint = points[0] if endpoint_position == 0 else points[-1]
            inner = points[1] if endpoint_position == 0 else points[-2]
            candidate_outward = endpoint - inner
            candidate_norm = float(np.linalg.norm(candidate_outward))
            if candidate_norm <= 1e-6:
                continue
            candidate_outward /= candidate_norm
            geometric_matches = []
            for node_idx in context.endpoint_ids_within(endpoint, max_distance_px):
                if node_idx in protected_endpoints:
                    continue
                graph_outward = outward_vectors.get(node_idx)
                if graph_outward is None:
                    continue
                delta = nodes_rc[node_idx] - endpoint
                distance = float(np.linalg.norm(delta))
                if distance < 2.0 or distance > max_distance_px:
                    continue
                candidate_to_graph = delta / distance
                graph_to_candidate = -candidate_to_graph
                candidate_alignment = float(np.dot(candidate_outward, candidate_to_graph))
                graph_alignment = float(np.dot(graph_outward, graph_to_candidate))
                alignment = min(candidate_alignment, graph_alignment)
                if alignment >= min_alignment_cosine:
                    geometric_matches.append((distance, node_idx, alignment))
            # Evaluate only the nearest few direction-compatible endpoints.
            for distance, node_idx, alignment in sorted(geometric_matches)[:3]:
                path, support_ratio = shortest_mask_path(
                    binary,
                    tuple(endpoint),
                    tuple(nodes_rc[node_idx]),
                    margin_px=path_margin_px,
                    outside_cost=outside_cost,
                    road_probability=road_probability,
                    center_probability=center_probability,
                    road_weight=0.85,
                    center_weight=0.15,
                    profiling=profiling,
                )
                if len(path) < 2:
                    continue
                path_length = float(sum(
                    np.hypot(second[0] - first[0], second[1] - first[1])
                    for first, second in zip(path[:-1], path[1:])
                ))
                path_ratio = path_length / max(distance, 1e-6)
                if path_ratio > max_path_ratio:
                    continue
                path_indices = np.rint(np.asarray(path)).astype(np.int32)
                path_road_values = road_probability[path_indices[:, 0], path_indices[:, 1]]
                road_probability_mean = float(np.mean(path_road_values))
                road_probability_q25 = float(np.quantile(path_road_values, 0.25))
                strong_surface_evidence = support_ratio >= min_surface_support
                probability_fallback = (
                    alignment >= fallback_min_alignment
                    and path_ratio <= fallback_max_path_ratio
                    and road_probability_mean >= fallback_min_road_probability
                    and road_probability_q25 >= fallback_min_road_probability_q25
                )
                if not strong_surface_evidence and not probability_fallback:
                    continue
                evidence_mode = "continuous_surface_mask" if strong_surface_evidence else "high_collinearity_road_probability"
                score = 0.50 * alignment + 0.35 * support_ratio + 0.15 * (1.0 - distance / max_distance_px)
                proposals.append((
                    score, candidate_index, endpoint_position, node_idx, path,
                    support_ratio, alignment, path_ratio, road_probability_mean,
                    road_probability_q25, evidence_mode,
                ))

    used_candidate_endpoints: set[tuple[int, int]] = set()
    used_graph_endpoints: set[int] = set()
    audit_rows: list[dict] = []
    for (
        score, candidate_index, endpoint_position, node_idx, path, support_ratio,
        alignment, path_ratio, road_probability_mean, road_probability_q25, evidence_mode,
    ) in sorted(
        proposals, key=lambda item: item[0], reverse=True
    ):
        candidate_key = (candidate_index, endpoint_position)
        if candidate_key in used_candidate_endpoints or node_idx in used_graph_endpoints:
            continue
        candidate = candidate_rows[candidate_index]
        points = [tuple(float(value) for value in point) for point in _candidate_polyline(candidate).tolist()]
        connector = resample_polyline(path, sample_step_px)
        if endpoint_position == 0:
            combined = list(reversed(connector)) + points[1:]
        else:
            combined = points[:-1] + connector
        if len(combined) < 2:
            continue
        candidate_length = float(sum(
            np.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(combined[:-1], combined[1:])
        ))
        candidate.update({
            "start_row": combined[0][0],
            "start_col": combined[0][1],
            "end_row": combined[-1][0],
            "end_col": combined[-1][1],
            "length_px": candidate_length,
            "point_count": len(combined),
            "polyline_points_json": json.dumps([[row, col] for row, col in combined], separators=(",", ":")),
            "surface_connector_count": int(candidate.get("surface_connector_count", 0) or 0) + 1,
            "surface_connector_mode": "a_star_on_road_surface_to_degree1_endpoint",
            "surface_attachment_audit": "accepted_surface_mask_path",
            "surface_attachment_kind": "surface_skeleton_to_graph_endpoint",
            "surface_attachment_endpoint_position": endpoint_position,
            "surface_attachment_node_row": float(nodes_rc[node_idx][0]),
            "surface_attachment_node_col": float(nodes_rc[node_idx][1]),
            "surface_attachment_surface_support_ratio": support_ratio,
            "surface_attachment_direction_alignment": alignment,
            "surface_attachment_path_ratio": path_ratio,
            "surface_attachment_road_probability_mean": road_probability_mean,
            "surface_attachment_road_probability_q25": road_probability_q25,
            "surface_attachment_evidence_mode": evidence_mode,
        })
        connected_positions = {
            int(value) for value in str(candidate.get("surface_connector_endpoint_positions", "")).split(";")
            if value.strip() in {"0", "1"}
        }
        connected_positions.add(endpoint_position)
        candidate["surface_connector_endpoint_positions"] = ";".join(str(value) for value in sorted(connected_positions))
        note = str(candidate.get("note", "") or "")
        candidate["note"] = ";".join(filter(None, (note, "mask_path_connected_to_graph_endpoint")))
        used_candidate_endpoints.add(candidate_key)
        used_graph_endpoints.add(node_idx)
        audit_rows.append({
            "connector_id": len(audit_rows),
            "connector_kind": "surface_skeleton_to_graph_endpoint",
            "candidate_id": candidate.get("candidate_id", candidate_index),
            "target_candidate_id": "",
            "candidate_endpoint_position": endpoint_position,
            "graph_endpoint_node_idx": node_idx,
            "gap_distance_px": float(np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])),
            "path_length_px": float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))),
            "path_ratio": path_ratio,
            "direction_alignment": alignment,
            "surface_support_ratio": support_ratio,
            "road_probability_mean": road_probability_mean,
            "road_probability_q25": road_probability_q25,
            "evidence_mode": evidence_mode,
            "topology_score": score,
            "action": "extend_surface_skeleton_along_mask_path",
        })
    return audit_rows


def build_surface_skeleton_pair_connectors(
    candidate_rows: list[dict],
    binary: np.ndarray,
    road_probability: np.ndarray,
    center_probability: np.ndarray,
    max_distance_px: float = 128.0,
    min_alignment_cosine: float = 0.95,
    min_surface_support: float = 0.70,
    max_path_ratio: float = 1.15,
    path_margin_px: int = 28,
    outside_cost: float = 30.0,
    sample_step_px: float = 6.0,
    audit_start_id: int = 0,
    min_road_probability: float = 0.30,
    min_road_probability_q25: float = 0.12,
    ambiguity_ratio: float = 1.20,
    profiling: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build A*-constrained links between facing endpoints of distinct skeleton regions."""
    endpoints: list[dict] = []
    for candidate_index, candidate in enumerate(candidate_rows):
        if candidate.get("candidate_type") != "surface_skeleton":
            continue
        points = _candidate_polyline(candidate)
        if points.shape[0] < 2:
            continue
        connected_positions = {
            int(value) for value in str(candidate.get("surface_connector_endpoint_positions", "")).split(";")
            if value.strip() in {"0", "1"}
        }
        for endpoint_position in (0, 1):
            if endpoint_position in connected_positions:
                continue
            endpoint = points[0] if endpoint_position == 0 else points[-1]
            inner = points[1] if endpoint_position == 0 else points[-2]
            outward = endpoint - inner
            norm = float(np.linalg.norm(outward))
            if norm <= 1e-6:
                continue
            endpoints.append({
                "candidate_index": candidate_index,
                "candidate_id": candidate.get("candidate_id", candidate_index),
                "region_id": str(candidate.get("region_id", "")),
                "endpoint_position": endpoint_position,
                "point": endpoint,
                "outward": outward / norm,
            })

    geometric_pairs: list[tuple[int, int, float, float]] = []
    endpoint_neighbors: dict[int, list[tuple[float, int]]] = {index: [] for index in range(len(endpoints))}
    endpoint_index = PointGridIndex.build(
        np.asarray([row["point"] for row in endpoints], dtype=np.float32).reshape(-1, 2),
        max_distance_px,
    )
    for first_index, first in enumerate(endpoints):
        for second_index in endpoint_index.query_radius_box(first["point"], max_distance_px):
            if second_index <= first_index:
                continue
            second = endpoints[second_index]
            if first["candidate_index"] == second["candidate_index"]:
                continue
            if first["region_id"] and first["region_id"] == second["region_id"]:
                continue
            delta = second["point"] - first["point"]
            distance = float(np.linalg.norm(delta))
            if distance < 2.0 or distance > max_distance_px:
                continue
            direction = delta / distance
            alignment = min(
                float(np.dot(first["outward"], direction)),
                float(np.dot(second["outward"], -direction)),
            )
            if alignment < min_alignment_cosine:
                continue
            geometric_pairs.append((first_index, second_index, distance, alignment))
            endpoint_neighbors[first_index].append((distance, second_index))
            endpoint_neighbors[second_index].append((distance, first_index))

    proposals: list[tuple] = []
    for first_index, second_index, distance, alignment in geometric_pairs:
        # A connector must be the unambiguous mutual-nearest continuation for
        # both endpoints. This is the main guard against diagonal cross-links
        # when several nearby roads or branches offer plausible targets.
        first_neighbors = sorted(endpoint_neighbors[first_index])
        second_neighbors = sorted(endpoint_neighbors[second_index])
        if first_neighbors[0][1] != second_index or second_neighbors[0][1] != first_index:
            continue
        if len(first_neighbors) > 1 and first_neighbors[1][0] < distance * ambiguity_ratio:
            continue
        if len(second_neighbors) > 1 and second_neighbors[1][0] < distance * ambiguity_ratio:
            continue
        first, second = endpoints[first_index], endpoints[second_index]
        path, support_ratio = shortest_mask_path(
            binary,
            tuple(first["point"]),
            tuple(second["point"]),
            margin_px=path_margin_px,
            outside_cost=outside_cost,
            road_probability=road_probability,
            center_probability=center_probability,
            road_weight=0.85,
            center_weight=0.15,
            profiling=profiling,
        )
        if len(path) < 2:
            continue
        path_length = float(sum(
            np.hypot(second_point[0] - first_point[0], second_point[1] - first_point[1])
            for first_point, second_point in zip(path[:-1], path[1:])
        ))
        path_ratio = path_length / max(distance, 1e-6)
        if path_ratio > max_path_ratio:
            continue
        path_indices = np.rint(np.asarray(path)).astype(np.int32)
        path_road_values = road_probability[path_indices[:, 0], path_indices[:, 1]]
        road_probability_mean = float(np.mean(path_road_values))
        road_probability_q25 = float(np.quantile(path_road_values, 0.25))
        strong_surface_evidence = support_ratio >= min_surface_support
        probability_fallback = (
            road_probability_mean >= min_road_probability
            and road_probability_q25 >= min_road_probability_q25
        )
        if not strong_surface_evidence and not probability_fallback:
            continue
        evidence_mode = "continuous_surface_mask" if strong_surface_evidence else "high_collinearity_road_probability"
        score = 0.50 * alignment + 0.35 * support_ratio + 0.15 * (1.0 - distance / max_distance_px)
        proposals.append((
            score, first_index, second_index, path, support_ratio, alignment,
            path_ratio, road_probability_mean, road_probability_q25, evidence_mode,
        ))

    used_endpoints: set[int] = set()
    connector_rows: list[dict] = []
    audit_rows: list[dict] = []
    for (
        score, first_index, second_index, path, support_ratio, alignment,
        path_ratio, road_probability_mean, road_probability_q25, evidence_mode,
    ) in sorted(
        proposals, key=lambda item: item[0], reverse=True
    ):
        if first_index in used_endpoints or second_index in used_endpoints:
            continue
        first, second = endpoints[first_index], endpoints[second_index]
        sampled = resample_polyline(path, sample_step_px)
        if len(sampled) < 2:
            continue
        connector_id = len(connector_rows)
        connector_rows.append({
            "candidate_id": len(candidate_rows) + connector_id,
            "region_id": "",
            "branch_index": 0,
            "candidate_type": "surface_skeleton_connector",
            "review_status": "auto_accept",
            "action": "auto_add_surface_gap_by_mask_path",
            "confidence": "high",
            "auto_decision": "accept",
            "auto_rule": "surface_skeleton_pair_mask_path",
            "topology_impact": "connect_surface_skeleton_fragments",
            "topology_score": score,
            "surface_support_ratio": support_ratio,
            "direction_alignment": alignment,
            "path_ratio": path_ratio,
            "start_row": sampled[0][0],
            "start_col": sampled[0][1],
            "end_row": sampled[-1][0],
            "end_col": sampled[-1][1],
            "length_px": float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(sampled[:-1], sampled[1:]))),
            "skeleton_pixels": len(path),
            "point_count": len(sampled),
            "polyline_points_json": json.dumps([[row, col] for row, col in sampled], separators=(",", ":")),
            "surface_connector_count": 1,
            "surface_connector_mode": "a_star_on_road_surface_between_skeletons",
            "hard_veto": False,
            "hard_veto_reasons": "",
            "effective_veto_reasons": "",
            "note": "mask_path_connected_surface_skeleton_fragments",
        })
        audit_rows.append({
            "connector_id": audit_start_id + len(audit_rows),
            "connector_kind": "surface_skeleton_to_surface_skeleton",
            "candidate_id": first["candidate_id"],
            "target_candidate_id": second["candidate_id"],
            "candidate_endpoint_position": first["endpoint_position"],
            "target_endpoint_position": second["endpoint_position"],
            "graph_endpoint_node_idx": "",
            "gap_distance_px": float(np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])),
            "path_length_px": float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))),
            "path_ratio": path_ratio,
            "direction_alignment": alignment,
            "surface_support_ratio": support_ratio,
            "road_probability_mean": road_probability_mean,
            "road_probability_q25": road_probability_q25,
            "evidence_mode": evidence_mode,
            "topology_score": score,
            "action": "connect_surface_skeletons_along_mask_path",
        })
        used_endpoints.update({first_index, second_index})
    return connector_rows, audit_rows


def edge_lengths(nodes_rc: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lengths = np.zeros(edges.shape[0], dtype=np.float32)
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        lengths[edge_id] = float(np.linalg.norm(nodes_rc[dst_idx] - nodes_rc[src_idx]))
    return lengths


def maybe_snap_to_mask(row: float, col: float, binary: np.ndarray, radius: int) -> tuple[float, float, bool, float]:
    r = int(round(row))
    c = int(round(col))
    if 0 <= r < binary.shape[0] and 0 <= c < binary.shape[1] and binary[r, c] > 0:
        return row, col, False, 0.0
    if radius <= 0:
        return row, col, False, 0.0

    y0, y1 = max(0, r - radius), min(binary.shape[0], r + radius + 1)
    x0, x1 = max(0, c - radius), min(binary.shape[1], c + radius + 1)
    patch = binary[y0:y1, x0:x1]
    ys, xs = np.nonzero(patch)
    if ys.size == 0:
        return row, col, False, 0.0
    abs_rows = ys + y0
    abs_cols = xs + x0
    dist2 = (abs_rows - row) ** 2 + (abs_cols - col) ** 2
    idx = int(np.argmin(dist2))
    snap_distance = float(np.sqrt(dist2[idx]))
    return float(abs_rows[idx]), float(abs_cols[idx]), True, snap_distance


def scan_one_side(
    binary: np.ndarray,
    row: float,
    col: float,
    normal_rc: np.ndarray,
    sign: float,
    max_search_px: float,
    step_px: float,
) -> tuple[float, str]:
    last_inside = 0.0
    distance = step_px
    while distance <= max_search_px:
        rr = int(round(row + sign * normal_rc[0] * distance))
        cc = int(round(col + sign * normal_rc[1] * distance))
        if rr < 0 or rr >= binary.shape[0] or cc < 0 or cc >= binary.shape[1]:
            return float(last_inside), "image_border"
        if binary[rr, cc] == 0:
            return float(last_inside), "mask_boundary"
        last_inside = distance
        distance += step_px
    return float(last_inside), "max_search"


def sample_widths_by_normal(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    binary: np.ndarray,
    sample_step_px: float,
    normal_step_px: float,
    max_search_px: float,
    pixel_size: float,
    snap_radius_px: int,
    junction_buffer_px: float = 30.0,
    border_margin_px: int = 2,
    max_snap_distance_px: float = 4.0,
    max_asymmetry_ratio: float = 0.65,
) -> list[dict]:
    rows: list[dict] = []
    sample_id = 0
    degrees, _ = graph_degrees(edges.shape[0], nodes_rc.shape[0], edges)
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        start = nodes_rc[src_idx]
        end = nodes_rc[dst_idx]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1.0:
            continue
        tangent = vec / length
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        # Midpoint sampling avoids graph nodes, whose intersection pavement is not a road width.
        sample_count = max(1, int(np.ceil(length / max(sample_step_px, 1.0))))
        for sample_index in range(sample_count):
            t = (sample_index + 0.5) / sample_count
            row, col = start + vec * t
            row_used, col_used, snapped, snap_distance = maybe_snap_to_mask(row, col, binary, snap_radius_px)
            r = int(round(row_used))
            c = int(round(col_used))
            inside = bool(0 <= r < binary.shape[0] and 0 <= c < binary.shape[1] and binary[r, c] > 0)
            distance_to_edge_start = t * length
            distance_to_edge_end = (1.0 - t) * length
            near_junction = bool(
                (degrees[src_idx] >= 3 and distance_to_edge_start < junction_buffer_px)
                or (degrees[dst_idx] >= 3 and distance_to_edge_end < junction_buffer_px)
            )
            near_image_border = bool(
                r < border_margin_px
                or c < border_margin_px
                or r >= binary.shape[0] - border_margin_px
                or c >= binary.shape[1] - border_margin_px
            )
            if inside:
                left_px, left_stop = scan_one_side(binary, row_used, col_used, normal, -1.0, max_search_px, normal_step_px)
                right_px, right_stop = scan_one_side(binary, row_used, col_used, normal, 1.0, max_search_px, normal_step_px)
                raw_width_px = left_px + right_px + normal_step_px
            else:
                left_px = right_px = raw_width_px = 0.0
                left_stop = right_stop = "outside_mask"
            asymmetry_ratio = abs(left_px - right_px) / max(left_px + right_px, normal_step_px)
            censored = left_stop != "mask_boundary" or right_stop != "mask_boundary"
            valid_width = bool(
                inside
                and not near_junction
                and not near_image_border
                and not censored
                and snap_distance <= max_snap_distance_px
            )
            width_px = raw_width_px if valid_width else 0.0
            quality_flags = []
            if not inside:
                quality_flags.append("outside_mask")
            if near_junction:
                quality_flags.append("junction_excluded")
            if near_image_border or "image_border" in {left_stop, right_stop}:
                quality_flags.append("image_border_censored")
            if "max_search" in {left_stop, right_stop}:
                quality_flags.append("max_search_censored")
            if snap_distance > max_snap_distance_px:
                quality_flags.append("large_snap")
            if asymmetry_ratio > max_asymmetry_ratio:
                quality_flags.append("asymmetric")
            rows.append(
                {
                    "sample_id": sample_id,
                    "edge_id": edge_id,
                    "sample_index": sample_index,
                    "row": float(row),
                    "col": float(col),
                    "row_used": float(row_used),
                    "col_used": float(col_used),
                    "snapped": snapped,
                    "snap_distance_px": snap_distance,
                    "inside_mask": inside,
                    "valid_width": valid_width,
                    "near_junction": near_junction,
                    "near_image_border": near_image_border,
                    "left_px": left_px,
                    "right_px": right_px,
                    "left_stop": left_stop,
                    "right_stop": right_stop,
                    "raw_width_px": raw_width_px,
                    "asymmetry_ratio": asymmetry_ratio,
                    "quality_flags": ";".join(quality_flags),
                    "width_px": width_px,
                    "width_units": width_px * pixel_size,
                }
            )
            sample_id += 1
    return rows


def edge_summary(rows: list[dict], pixel_size: float) -> list[dict]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row["width_px"] > 0:
            grouped.setdefault(int(row["edge_id"]), []).append(float(row["width_px"]))
    summaries = []
    for edge_id, widths in sorted(grouped.items()):
        values = np.array(widths, dtype=np.float32)
        summaries.append(
            {
                "edge_id": edge_id,
                "valid_samples": int(values.size),
                "median_width_px": float(np.median(values)),
                "mean_width_px": float(np.mean(values)),
                "median_width_units": float(np.median(values) * pixel_size),
                "mean_width_units": float(np.mean(values) * pixel_size),
            }
        )
    return summaries


def classify_and_interpolate_edges(
    rows: list[dict],
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    pixel_size: float,
    min_coverage: float,
    short_gap_px: float,
    max_width_cv: float,
    outlier_mad_scale: float = 6.0,
) -> list[dict]:
    edge_count = edges.shape[0]
    lengths = edge_lengths(nodes_rc, edges)
    _, node_edges = graph_degrees(edge_count, nodes_rc.shape[0], edges)
    rows_by_edge: list[list[dict]] = [[] for _ in range(edge_count)]
    for row in rows:
        edge_id = int(row["edge_id"])
        if 0 <= edge_id < edge_count:
            rows_by_edge[edge_id].append(row)

    summaries: list[dict] = []
    measured_widths: dict[int, float] = {}
    for edge_id, edge_rows in enumerate(rows_by_edge):
        widths = np.array([r["width_px"] for r in edge_rows if r["width_px"] > 0], dtype=np.float32)
        raw_widths = np.array([r.get("raw_width_px", 0.0) for r in edge_rows if r.get("raw_width_px", 0.0) > 0], dtype=np.float32)
        sample_count = len(edge_rows)
        valid_count = int(widths.size)
        coverage = valid_count / sample_count if sample_count else 0.0
        median_width = float(np.median(widths)) if valid_count else 0.0
        mean_width = float(np.mean(widths)) if valid_count else 0.0
        std_width = float(np.std(widths)) if valid_count else 0.0
        width_cv = std_width / median_width if median_width > 0 else 0.0
        censored_count = sum(
            1 for r in edge_rows if "censored" in str(r.get("quality_flags", ""))
        )
        junction_excluded_count = sum(1 for r in edge_rows if r.get("near_junction"))
        asymmetric_count = sum(1 for r in edge_rows if "asymmetric" in str(r.get("quality_flags", "")))
        large_snap_count = sum(1 for r in edge_rows if "large_snap" in str(r.get("quality_flags", "")))
        raw_median_width = float(np.median(raw_widths)) if raw_widths.size else 0.0

        if coverage >= min_coverage and median_width > 0:
            status = "measured"
            width_source = "measured"
            confidence = "medium" if width_cv > max_width_cv else "high"
            final_width = median_width
        elif coverage > 0 and lengths[edge_id] <= short_gap_px:
            status = "interpolated_short_gap"
            width_source = "partial_measured"
            confidence = "medium"
            final_width = median_width
        elif coverage > 0:
            status = "partial_surface"
            width_source = "partial_measured"
            confidence = "low"
            final_width = median_width
        elif raw_widths.size > 0:
            status = "measurement_censored"
            width_source = "unresolved"
            confidence = "low"
            final_width = 0.0
        else:
            status = "surface_missing"
            width_source = "unknown"
            confidence = "low"
            final_width = 0.0

        if final_width > 0:
            measured_widths[edge_id] = final_width

        src_idx, dst_idx = edges[edge_id]
        summaries.append(
            {
                "edge_id": edge_id,
                "src_idx": int(src_idx),
                "dst_idx": int(dst_idx),
                "length_px": float(lengths[edge_id]),
                "sample_count": sample_count,
                "valid_samples": valid_count,
                "coverage_ratio": coverage,
                "median_width_px": median_width,
                "mean_width_px": mean_width,
                "std_width_px": std_width,
                "width_cv": width_cv,
                "raw_median_width_px": raw_median_width,
                "censored_samples": censored_count,
                "junction_excluded_samples": junction_excluded_count,
                "asymmetric_samples": asymmetric_count,
                "large_snap_samples": large_snap_count,
                "final_width_px": final_width,
                "final_width_units": final_width * pixel_size,
                "width_source": width_source,
                "status": status,
                "confidence": confidence,
                "requires_review": status != "measured" or width_cv > max_width_cv or asymmetric_count > 0 or censored_count > 0 or large_snap_count > 0,
                "review_reasons": "",
            }
        )

    for summary in summaries:
        if summary["status"] not in {"surface_missing"}:
            continue
        if summary["length_px"] > short_gap_px:
            continue
        edge_id = int(summary["edge_id"])
        src_idx, dst_idx = edges[edge_id]
        neighbor_widths = []
        for node_idx in (int(src_idx), int(dst_idx)):
            for neighbor_edge_id in node_edges[node_idx]:
                if neighbor_edge_id != edge_id and neighbor_edge_id in measured_widths:
                    neighbor_widths.append(measured_widths[neighbor_edge_id])
        if neighbor_widths:
            final_width = float(np.median(np.array(neighbor_widths, dtype=np.float32)))
            summary["final_width_px"] = final_width
            summary["final_width_units"] = final_width * pixel_size
            summary["width_source"] = "interpolated_neighbor"
            summary["status"] = "interpolated_short_gap"
            summary["confidence"] = "medium"

    positive = np.asarray([s["final_width_px"] for s in summaries if s["final_width_px"] > 0], dtype=np.float32)
    global_median = float(np.median(positive)) if positive.size else 0.0
    mad = float(np.median(np.abs(positive - global_median))) if positive.size else 0.0
    robust_sigma = max(1.0, 1.4826 * mad)
    outlier_limit = global_median + outlier_mad_scale * robust_sigma
    for summary in summaries:
        reasons = []
        if summary["status"] != "measured":
            reasons.append(str(summary["status"]))
        if summary["width_cv"] > max_width_cv:
            reasons.append("high_width_variation")
        if summary["asymmetric_samples"] > 0:
            reasons.append("asymmetric_samples")
        if summary["censored_samples"] > 0:
            reasons.append("censored_samples")
        if summary["large_snap_samples"] > 0:
            reasons.append("large_snap")
        if summary["final_width_px"] > outlier_limit and positive.size >= 10:
            reasons.append("global_width_outlier")
        summary["requires_review"] = bool(reasons)
        summary["review_reasons"] = ";".join(reasons)
    return summaries


def analyze_surface_only_regions(
    binary: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    centerline_buffer_px: int,
    endpoint_radius_px: int,
    min_area: int,
    min_candidate_skeleton_ratio: float,
    max_suspect_extent_ratio: float,
    graph_context: GraphSpatialContext | None = None,
) -> tuple[np.ndarray, list[dict]]:
    road_buffer = graph_buffer_mask(binary.shape, nodes_rc, edges, centerline_buffer_px)
    surface_only = np.where((binary > 0) & (road_buffer == 0), 1, 0).astype(np.uint8)
    endpoint_mask = np.zeros(binary.shape, dtype=np.uint8)
    context = graph_context or GraphSpatialContext.build(nodes_rc, edges)
    degrees = context.degrees
    endpoint_indices = np.where(degrees <= 1)[0]
    for node_idx in endpoint_indices.tolist():
        row, col = nodes_rc[node_idx]
        cv2.circle(endpoint_mask, (int(round(col)), int(round(row))), endpoint_radius_px, 1, -1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_only, connectivity=8)
    records: list[dict] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        skeleton = cv2.ximgproc.thinning(component) if hasattr(cv2, "ximgproc") else component
        skeleton_len = int(np.count_nonzero(skeleton))
        extent_ratio = area / max(1, w * h)
        skeleton_ratio = skeleton_len / max(1, area)
        touches_endpoint = bool(np.any((labels == label) & (endpoint_mask > 0)))
        if touches_endpoint and skeleton_ratio >= min_candidate_skeleton_ratio:
            status = "surface_only_candidate"
            note = "near_graph_endpoint_and_linear"
        elif extent_ratio >= max_suspect_extent_ratio or skeleton_ratio < min_candidate_skeleton_ratio:
            status = "surface_only_suspect"
            note = "blocky_or_non_linear_surface"
        else:
            status = "surface_only_review"
            note = "needs_review"
        cx, cy = centroids[label]
        records.append(
            {
                "region_id": int(label),
                "status": status,
                "area_px": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "centroid_row": float(cy),
                "centroid_col": float(cx),
                "skeleton_len_px": skeleton_len,
                "skeleton_area_ratio": float(skeleton_ratio),
                "extent_ratio": float(extent_ratio),
                "touches_graph_endpoint": touches_endpoint,
                "note": note,
            }
        )
    return surface_only, records


def thinning(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    if hasattr(cv2, "ximgproc"):
        return (cv2.ximgproc.thinning(binary * 255) > 0).astype(np.uint8)

    # Zhang-Suen thinning preserves junction connectivity. Morphological
    # skeletonization can leave a gap in the center of wide T/X junctions.
    image = np.pad(binary, 1, mode="constant")
    changed = True
    while changed:
        changed = False
        for first_pass in (True, False):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            center = image[1:-1, 1:-1]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if first_pass:
                topology = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                topology = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (center == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & topology
            if np.any(remove):
                center[remove] = 0
                changed = True
    return image[1:-1, 1:-1]


def skeleton_neighbors(point: tuple[int, int], point_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, col = point
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not (dr or dc):
                continue
            neighbor = (row + dr, col + dc)
            if neighbor not in point_set:
                continue
            # A one-pixel staircase already has an orthogonal route between
            # diagonal pixels. Adding the diagonal edge creates triangles,
            # turns ordinary bends into false junctions, and fragments long
            # centerlines into branches that are then removed as short spurs.
            if dr and dc and ((row + dr, col) in point_set or (row, col + dc) in point_set):
                continue
            neighbors.append(neighbor)
    return neighbors


def farthest_skeleton_point(
    start: tuple[int, int], point_set: set[tuple[int, int]]
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    distance = {start: 0.0}
    queue = deque([start])
    while queue:
        point = queue.popleft()
        for neighbor in skeleton_neighbors(point, point_set):
            if neighbor in parent:
                continue
            parent[neighbor] = point
            distance[neighbor] = distance[point] + float(np.hypot(neighbor[0] - point[0], neighbor[1] - point[1]))
            queue.append(neighbor)
    return max(distance, key=distance.get), parent


def longest_skeleton_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(skeleton)
    point_set = {(int(row), int(col)) for row, col in zip(ys.tolist(), xs.tolist())}
    if len(point_set) < 2:
        return []
    # Work on the largest 8-connected skeleton component. Small fragments are
    # common around anti-aliased mask edges and must not become candidates.
    components = []
    remaining = set(point_set)
    while remaining:
        start = next(iter(remaining))
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            point = queue.popleft()
            for neighbor in skeleton_neighbors(point, point_set):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    point_set = max(components, key=len)
    if len(point_set) < 2:
        return []
    endpoints = [point for point in point_set if len(skeleton_neighbors(point, point_set)) <= 1]
    start = endpoints[0] if endpoints else next(iter(point_set))
    first, _ = farthest_skeleton_point(start, point_set)
    second, parent = farthest_skeleton_point(first, point_set)
    path = []
    current: tuple[int, int] | None = second
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def skeleton_branch_paths(skeleton: np.ndarray, min_length_px: float) -> list[list[tuple[int, int]]]:
    ys, xs = np.nonzero(skeleton)
    all_points = {(int(row), int(col)) for row, col in zip(ys.tolist(), xs.tolist())}
    if len(all_points) < 2:
        return []

    components = []
    remaining = set(all_points)
    while remaining:
        start = next(iter(remaining))
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            point = queue.popleft()
            for neighbor in skeleton_neighbors(point, all_points):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    paths: list[list[tuple[int, int]]] = []
    for point_set in components:
        vertices = {point for point in point_set if len(skeleton_neighbors(point, point_set)) != 2}
        if not vertices:
            component_mask = np.zeros_like(skeleton)
            for row, col in point_set:
                component_mask[row, col] = 1
            fallback = longest_skeleton_path(component_mask)
            if fallback:
                paths.append(fallback)
            continue

        visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
            return tuple(sorted((a, b)))

        for vertex in vertices:
            for neighbor in skeleton_neighbors(vertex, point_set):
                first_edge = edge_key(vertex, neighbor)
                if first_edge in visited_edges:
                    continue
                visited_edges.add(first_edge)
                path = [vertex, neighbor]
                previous, current = vertex, neighbor
                while current not in vertices:
                    next_points = [point for point in skeleton_neighbors(current, point_set) if point != previous]
                    if not next_points:
                        break
                    next_point = next_points[0]
                    next_edge = edge_key(current, next_point)
                    if next_edge in visited_edges:
                        break
                    visited_edges.add(next_edge)
                    path.append(next_point)
                    previous, current = current, next_point
                length = sum(
                    float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(path[:-1], path[1:])
                )
                if length >= min_length_px:
                    paths.append(path)
    return paths


def resample_polyline(points: list[tuple[float, float]], step_px: float) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    step_px = max(2.0, float(step_px))
    segment_lengths = np.asarray(
        [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:])], dtype=np.float64
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        return [points[0]]
    targets = list(np.arange(0.0, total, step_px))
    if not targets or total - targets[-1] > 1.0:
        targets.append(total)
    sampled = []
    for target in targets:
        segment_idx = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(points) - 2)
        length = segment_lengths[segment_idx]
        ratio = 0.0 if length <= 0 else (target - cumulative[segment_idx]) / length
        start, end = points[segment_idx], points[segment_idx + 1]
        sampled.append((start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio))
    return sampled


def candidate_record_from_path(
    path: list[tuple[int, int]],
    bbox_x: int,
    bbox_y: int,
    sample_step_px: float,
    skeleton_pixels: int,
    simplify_epsilon_px: float = 1.5,
    median_half_width_px: float = 0.0,
) -> dict | None:
    global_path = [(float(row + bbox_y), float(col + bbox_x)) for row, col in path]
    if simplify_epsilon_px > 0 and len(global_path) >= 3:
        points_xy = np.asarray([[col, row] for row, col in global_path], dtype=np.float32).reshape(-1, 1, 2)
        simplified_xy = cv2.approxPolyDP(points_xy, simplify_epsilon_px, False).reshape(-1, 2)
        global_path = [(float(row), float(col)) for col, row in simplified_xy.tolist()]
    sampled = resample_polyline(global_path, sample_step_px)
    length = float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(sampled[:-1], sampled[1:])))
    if length < 1.0:
        return None
    return {
        "start_row": sampled[0][0],
        "start_col": sampled[0][1],
        "end_row": sampled[-1][0],
        "end_col": sampled[-1][1],
        "length_px": length,
        "skeleton_pixels": skeleton_pixels,
        "median_half_width_px": median_half_width_px,
        "point_count": len(sampled),
        "polyline_points_json": json.dumps([[row, col] for row, col in sampled], separators=(",", ":")),
    }


def component_candidate_centerlines(
    component: np.ndarray,
    bbox_x: int,
    bbox_y: int,
    sample_step_px: float,
    min_branch_length_px: float,
    simplify_epsilon_px: float = 1.5,
    min_half_width_px: float = 1.0,
) -> list[dict]:
    skel = thinning(component)
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    paths = skeleton_branch_paths(skel, min_branch_length_px)
    if not paths:
        fallback = longest_skeleton_path(skel)
        paths = [fallback] if fallback else []
    records = []
    for path in paths:
        median_half_width = float(np.median([distance[row, col] for row, col in path]))
        if median_half_width < min_half_width_px:
            continue
        record = candidate_record_from_path(
            path,
            bbox_x,
            bbox_y,
            sample_step_px,
            int(np.count_nonzero(skel)),
            simplify_epsilon_px=simplify_epsilon_px,
            median_half_width_px=median_half_width,
        )
        if record is not None:
            records.append(record)
    return records


def component_candidate_centerline(
    component: np.ndarray, bbox_x: int, bbox_y: int, sample_step_px: float
) -> dict | None:
    skeleton = thinning(component)
    path = longest_skeleton_path(skeleton)
    return candidate_record_from_path(
        path, bbox_x, bbox_y, sample_step_px, int(np.count_nonzero(skeleton))
    ) if path else None


def build_candidate_centerlines(
    surface_only: np.ndarray,
    surface_only_rows: list[dict],
    min_length_px: float,
    sample_step_px: float = 10.0,
    simplify_epsilon_px: float = 1.5,
    min_half_width_px: float = 1.0,
) -> list[dict]:
    if not surface_only_rows:
        return []
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(surface_only.astype(np.uint8), connectivity=8)
    del num_labels
    candidates = []
    for row in surface_only_rows:
        if row["status"] not in {"surface_only_candidate", "surface_only_review"}:
            continue
        region_id = int(row["region_id"])
        x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        component = (labels[y:y + h, x:x + w] == region_id).astype(np.uint8)
        lines = component_candidate_centerlines(
            component,
            x,
            y,
            sample_step_px,
            min_length_px,
            simplify_epsilon_px=simplify_epsilon_px,
            min_half_width_px=min_half_width_px,
        )
        for branch_index, line in enumerate(lines):
            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "region_id": region_id,
                    "branch_index": branch_index,
                    "candidate_type": "surface_skeleton",
                    "review_status": row["status"],
                    "action": "propose_add_centerline",
                    "confidence": "medium" if row["status"] == "surface_only_candidate" else "low",
                    "auto_decision": "review",
                    "topology_impact": "extend_or_add_surface_road",
                    "topology_score": 0.0,
                    "surface_support_ratio": 1.0,
                    "direction_alignment": "",
                    "start_node_idx": "",
                    "end_node_idx": "",
                    **line,
                    "area_px": row["area_px"],
                    "note": row["note"],
                }
            )
    return candidates


def summarize(rows: list[dict], pixel_size: float) -> dict:
    widths = np.array([row["width_px"] for row in rows if row["width_px"] > 0], dtype=np.float32)
    if widths.size == 0:
        return {"valid_samples": 0}
    return {
        "valid_samples": int(widths.size),
        "median_width_px": float(np.median(widths)),
        "mean_width_px": float(np.mean(widths)),
        "p10_width_px": float(np.percentile(widths, 10)),
        "p90_width_px": float(np.percentile(widths, 90)),
        "median_width_units": float(np.median(widths) * pixel_size),
        "mean_width_units": float(np.mean(widths) * pixel_size),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def index_files_by_stem(root: Path, suffixes: set[str]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.stem, []).append(path)
    for paths in index.values():
        paths.sort()
    return index


def mask_priority(path: Path, stem: str) -> tuple[int, str]:
    path_stem = path.stem
    if path_stem == f"{stem}_mask":
        return 0, str(path)
    if path_stem == stem:
        return 1, str(path)
    if path_stem.startswith(stem) and path_stem.endswith("_mask"):
        return 2, str(path)
    if path_stem.startswith(stem) and "mask" in path_stem:
        return 3, str(path)
    return 9, str(path)


def find_mask_by_stem(mask_index: dict[str, list[Path]], stem: str) -> Path | None:
    matches: list[Path] = []
    for indexed_stem, paths in mask_index.items():
        if (indexed_stem == stem or indexed_stem.startswith(stem)) and ("mask" in indexed_stem.lower() or indexed_stem == stem):
            matches.extend(paths)
    if not matches:
        return None
    return sorted(matches, key=lambda path: mask_priority(path, stem))[0]


def build_batch_jobs(image_dir: Path, graph_dir: Path, mask_dir: Path | None) -> list[tuple[Path, Path, Path | None]]:
    images = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise FileNotFoundError(f"No input images found under: {image_dir}")

    graph_index = index_files_by_stem(graph_dir, {".p", ".pickle", ".pkl"})
    mask_index = index_files_by_stem(mask_dir, MASK_SUFFIXES) if mask_dir else {}

    jobs: list[tuple[Path, Path, Path | None]] = []
    missing_graphs: list[str] = []
    missing_masks: list[str] = []
    for image_path in images:
        stem = image_path.stem
        graph_matches = graph_index.get(stem, [])
        graph_path = graph_matches[0] if graph_matches else None
        if graph_path is None:
            missing_graphs.append(stem)
            continue
        mask_path = find_mask_by_stem(mask_index, stem) if mask_dir else None
        if mask_dir and mask_path is None:
            missing_masks.append(stem)
            continue
        jobs.append((image_path, graph_path, mask_path))

    if missing_graphs:
        print(f"Skipped {len(missing_graphs)} images without matched graph: {', '.join(missing_graphs[:10])}")
    if missing_masks:
        print(f"Skipped {len(missing_masks)} images without matched mask: {', '.join(missing_masks[:10])}")
    if not jobs:
        raise FileNotFoundError("No matched image/graph/mask jobs were found.")
    return jobs


def draw_viz(
    rgb: np.ndarray,
    binary: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    rows: list[dict],
    edge_rows: list[dict] | None = None,
    surface_only: np.ndarray | None = None,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    overlay = canvas.copy()
    overlay[binary > 0] = (40, 180, 255)
    if surface_only is not None:
        overlay[surface_only > 0] = (0, 120, 255)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    edge_status = {}
    if edge_rows is not None:
        edge_status = {int(row["edge_id"]): row["status"] for row in edge_rows}
    colors = {
        "measured": (0, 255, 0),
        "interpolated_short_gap": (0, 255, 255),
        "partial_surface": (255, 128, 0),
        "surface_missing": (0, 0, 255),
        "measurement_censored": (0, 0, 255),
    }
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        r0, c0 = nodes_rc[src_idx]
        r1, c1 = nodes_rc[dst_idx]
        color = colors.get(edge_status.get(edge_id, "measured"), (0, 255, 0))
        cv2.line(canvas, (int(round(c0)), int(round(r0))), (int(round(c1)), int(round(r1))), color, 1)
    for row in rows:
        color = (255, 0, 0) if row["width_px"] > 0 else (0, 0, 255)
        cv2.circle(canvas, (int(round(row["col_used"])), int(round(row["row_used"]))), 2, color, -1)
    return canvas


def draw_review_demo(
    rgb: np.ndarray,
    binary: np.ndarray,
    surface_only: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    edge_rows: list[dict],
    surface_only_rows: list[dict],
    candidate_lines: list[dict],
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    overlay = canvas.copy()
    overlay[binary > 0] = (60, 180, 255)
    overlay[surface_only > 0] = (0, 0, 255)
    canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0)

    status_by_edge = {int(row["edge_id"]): row["status"] for row in edge_rows}
    edge_colors = {
        "measured": (0, 255, 0),
        "interpolated_short_gap": (0, 255, 255),
        "partial_surface": (255, 0, 255),
        "surface_missing": (0, 0, 255),
        "measurement_censored": (0, 0, 255),
    }
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        r0, c0 = nodes_rc[src_idx]
        r1, c1 = nodes_rc[dst_idx]
        color = edge_colors.get(status_by_edge.get(edge_id, "measured"), (0, 255, 0))
        cv2.line(canvas, (int(round(c0)), int(round(r0))), (int(round(c1)), int(round(r1))), color, 1)

    # Candidate geometry uses the same visual stroke width as existing centerlines.
    for candidate in candidate_lines:
        try:
            points = json.loads(candidate.get("polyline_points_json", ""))
        except (TypeError, json.JSONDecodeError):
            points = []
        if len(points) < 2:
            points = [
                [candidate.get("start_row", 0), candidate.get("start_col", 0)],
                [candidate.get("end_row", 0), candidate.get("end_col", 0)],
            ]
        pts = np.asarray([[int(round(float(col))), int(round(float(row)))] for row, col in points], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (255, 255, 0), 1, lineType=cv2.LINE_8)

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call SAM_MLoRA for road-surface segmentation, then estimate widths on a SAMRoad centerline graph."
    )
    parser.add_argument("--image", default="", help="Input image path for single-tile mode.")
    parser.add_argument("--graph", default="", help="SAMRoad graph .p path for single-tile mode.")
    parser.add_argument("--mask", default="", help="Existing road-surface mask. If omitted, SAM_MLoRA is called.")
    parser.add_argument("--road-probability", default="", help="Optional road-surface probability image; auto-detected beside --mask when omitted.")
    parser.add_argument("--centerline-probability", default="", help="Optional SAMRoad road-probability image; auto-detected beside the graph when omitted.")
    parser.add_argument("--image-dir", default="", help="Input image folder for batch mode.")
    parser.add_argument("--graph-dir", default="", help="SAMRoad graph folder for batch mode.")
    parser.add_argument("--mask-dir", default="", help="Road-surface mask folder for batch mode. Matched by image stem.")
    parser.add_argument("--output-dir", default=str(TOOL_DIR.parent / "runs" / "width_review"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel slice workers. 0 uses a conservative automatic maximum of 2; 1 is serial.",
    )
    parser.add_argument("--sam-pretrained-path", default=str(DEFAULT_MOLRA_SAM))
    parser.add_argument("--weight-path", default=str(DEFAULT_MOLRA_WEIGHT))
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--molra-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--centerline-buffer-px", type=int, default=0, help="Optional mask restriction around graph; 0 disables it.")
    parser.add_argument("--sample-step-px", type=float, default=20.0)
    parser.add_argument("--normal-step-px", type=float, default=1.0)
    parser.add_argument("--max-search-px", type=float, default=120.0)
    parser.add_argument("--snap-radius-px", type=int, default=8)
    parser.add_argument("--max-snap-distance-px", type=float, default=4.0, help="Snaps beyond this distance are diagnostic only and are not measured.")
    parser.add_argument("--junction-buffer-px", type=float, default=30.0, help="Exclude samples this close to degree>=3 graph nodes.")
    parser.add_argument("--border-margin-px", type=int, default=2, help="Exclude samples near image borders.")
    parser.add_argument("--max-asymmetry-ratio", type=float, default=0.65, help="Flag strongly asymmetric left/right transects for review.")
    parser.add_argument("--pixel-size", type=float, default=0.0, help="Map units per pixel; <=0 reads the raster transform and falls back to 1.0.")
    parser.add_argument("--min-edge-coverage", type=float, default=0.6, help="Coverage ratio needed for measured edge status.")
    parser.add_argument("--short-gap-px", type=float, default=80.0, help="Max edge length for neighbor-width interpolation.")
    parser.add_argument("--max-width-cv", type=float, default=0.5, help="Width variation threshold for high confidence.")
    parser.add_argument("--width-change-ratio", type=float, default=0.35, help="Relative local width change used to split width-profile segments.")
    parser.add_argument("--width-change-min-samples", type=int, default=3)
    parser.add_argument("--outlier-mad-scale", type=float, default=6.0, help="Robust global width-outlier review threshold.")
    parser.add_argument("--surface-only-buffer-px", type=int, default=40, help="Subtract this graph buffer before MLoRA-only region analysis.")
    parser.add_argument("--surface-only-endpoint-radius-px", type=int, default=80)
    parser.add_argument("--surface-only-min-area", type=int, default=400)
    parser.add_argument("--surface-only-min-skeleton-ratio", type=float, default=0.02)
    parser.add_argument("--surface-only-max-suspect-extent-ratio", type=float, default=0.45)
    parser.add_argument("--candidate-min-length-px", type=float, default=40.0)
    parser.add_argument("--candidate-sample-step-px", type=float, default=16.0, help="Vertex spacing along surface-skeleton candidate polylines; matches ordinary SAMRoad node spacing by default.")
    parser.add_argument("--candidate-simplify-epsilon-px", type=float, default=1.5, help="Douglas-Peucker tolerance before candidate resampling.")
    parser.add_argument("--candidate-min-half-width-px", type=float, default=1.0, help="Distance-transform ridge width used to reject very thin skeleton noise.")
    parser.add_argument("--recenter-max-shift-px", type=int, default=2, help="Maximum Euclidean shift of degree-2 nodes toward the surface medial ridge; 0 disables it.")
    parser.add_argument("--prune-spurs", action=argparse.BooleanOptionalAction, default=True, help="Remove only short dangling chains with weak surface and centerline probability.")
    parser.add_argument("--spur-max-length-px", type=float, default=20.0)
    parser.add_argument("--spur-max-road-probability", type=float, default=0.2)
    parser.add_argument("--spur-max-center-probability", type=float, default=0.2)
    parser.add_argument("--auto-retain-center-probability", type=float, default=0.5, help="Keep a line-without-surface conflict out of review when centerline probability and topology are strong.")
    parser.add_argument("--candidate-graph-snap-px", type=float, default=20.0, help="Spatial matching radius between surface skeleton endpoints and graph nodes.")
    parser.add_argument("--auto-fuse-surface-components", action=argparse.BooleanOptionalAction, default=True, help="Auto-accept a medium-confidence surface skeleton only when both ends match different graph components.")
    parser.add_argument("--auto-extend-surface-skeletons", action=argparse.BooleanOptionalAction, default=True, help="Auto-accept only a narrow, linear, surface-supported medium-confidence skeleton extending one unprotected degree-1 endpoint.")
    parser.add_argument("--connect-surface-skeletons", action=argparse.BooleanOptionalAction, default=True, help="Close only short, direction-consistent gaps from a surface skeleton to a degree-1 graph endpoint using an A* path on the road mask.")
    parser.add_argument("--surface-connector-max-distance-px", type=float, default=96.0)
    parser.add_argument("--surface-connector-min-alignment", type=float, default=0.85)
    parser.add_argument("--surface-connector-min-support", type=float, default=0.95)
    parser.add_argument("--surface-connector-max-path-ratio", type=float, default=1.35)
    parser.add_argument("--surface-connector-path-margin-px", type=int, default=24)
    parser.add_argument("--surface-connector-outside-cost", type=float, default=30.0)
    parser.add_argument("--surface-connector-fallback-min-alignment", type=float, default=0.95)
    parser.add_argument("--surface-connector-fallback-min-road-probability", type=float, default=0.50)
    parser.add_argument("--surface-connector-fallback-min-road-probability-q25", type=float, default=0.10)
    parser.add_argument("--surface-connector-fallback-max-path-ratio", type=float, default=1.15)
    parser.add_argument("--connect-surface-skeleton-pairs", action=argparse.BooleanOptionalAction, default=True, help="Connect facing endpoints of different surface-skeleton regions only through a strongly supported A* road-mask path.")
    parser.add_argument("--surface-pair-max-distance-px", type=float, default=128.0)
    parser.add_argument("--surface-pair-min-alignment", type=float, default=0.95)
    parser.add_argument("--surface-pair-min-support", type=float, default=0.70)
    parser.add_argument("--surface-pair-max-path-ratio", type=float, default=1.15)
    parser.add_argument("--surface-pair-path-margin-px", type=int, default=28)
    parser.add_argument("--surface-pair-min-road-probability", type=float, default=0.30)
    parser.add_argument("--surface-pair-min-road-probability-q25", type=float, default=0.12)
    parser.add_argument("--surface-pair-ambiguity-ratio", type=float, default=1.20)
    parser.add_argument("--surface-extension-min-alignment", type=float, default=0.35, help="Minimum endpoint outward-direction cosine for automatic one-end surface-skeleton extension.")
    parser.add_argument("--surface-extension-max-distance-px", type=float, default=160.0, help="Maximum endpoint-to-node or endpoint-to-edge attachment distance.")
    parser.add_argument("--surface-extension-min-length-px", type=float, default=40.0, help="Minimum candidate length; shorter candidates receive a hard veto.")
    parser.add_argument("--surface-extension-max-length-px", type=float, default=0.0, help="Deprecated compatibility option. 0 means unlimited; no maximum length veto is applied.")
    parser.add_argument("--surface-extension-min-support-ratio", type=float, default=0.65, help="Minimum road-surface support ratio for automatic graph attachment.")
    parser.add_argument("--surface-extension-max-half-width-px", type=float, default=14.0, help="Maximum median half-width eligible for automatic endpoint attachment.")
    parser.add_argument("--restore-divided-roads", action=argparse.BooleanOptionalAction, default=True, help="Restore only high-confidence paired continuations from saved low-threshold topology candidates.")
    parser.add_argument("--divided-road-min-candidate-probability", type=float, default=0.20)
    parser.add_argument("--divided-road-min-pair-score", type=float, default=0.72)
    parser.add_argument("--divided-road-min-spacing-px", type=float, default=5.0)
    parser.add_argument("--divided-road-max-spacing-px", type=float, default=40.0)
    parser.add_argument("--cleanup-junction-conflicts", action=argparse.BooleanOptionalAction, default=True, help="Remove only clearly redundant short cycles and dense fan links at junctions.")
    parser.add_argument("--junction-cleanup-max-cycle-px", type=float, default=120.0)
    parser.add_argument("--junction-cleanup-max-weak-probability", type=float, default=0.75)
    parser.add_argument("--junction-cleanup-min-probability-margin", type=float, default=0.05)
    parser.add_argument("--restore-divided-junction-corridors", action=argparse.BooleanOptionalAction, default=True, help="Split Y-merged divided roads into two probability-guided lanes through a junction conflict.")
    parser.add_argument("--repair-junctions", action=argparse.BooleanOptionalAction, default=False, help="Optional endpoint-to-edge repair. Disabled by default to avoid guessed connector lines.")
    parser.add_argument("--junction-repair-max-distance-px", type=float, default=120.0)
    parser.add_argument("--junction-repair-max-parallel-cosine", type=float, default=0.75, help="Reject endpoint-to-edge repairs that run along, rather than into, the target road.")
    parser.add_argument("--auto-connect-gaps", action=argparse.BooleanOptionalAction, default=True, help="Close only unambiguous, mutually facing degree-1 graph gaps on a continuous road surface.")
    parser.add_argument("--auto-gap-max-distance-px", type=float, default=96.0, help="Maximum endpoint distance considered for strict graph-gap repair.")
    parser.add_argument("--auto-gap-min-alignment", type=float, default=0.95, help="Minimum cosine alignment at both endpoints.")
    parser.add_argument("--auto-gap-min-surface-support", type=float, default=0.95, help="Minimum fraction of the A* path covered by the road-surface mask.")
    parser.add_argument("--auto-gap-max-path-ratio", type=float, default=1.15, help="Reject winding graph-gap paths relative to endpoint distance.")
    parser.add_argument("--auto-gap-ambiguity-ratio", type=float, default=1.20, help="Require the next-best endpoint to be this much farther than the chosen mutual-nearest endpoint.")
    parser.add_argument("--auto-gap-path-margin-px", type=int, default=28, help="Search margin around each endpoint pair.")
    parser.add_argument("--auto-gap-outside-cost", type=float, default=30.0, help="A* penalty for pixels outside the road-surface mask.")
    args = parser.parse_args()
    if not -1.0 <= args.surface_extension_min_alignment <= 1.0:
        parser.error("--surface-extension-min-alignment must be between -1 and 1.")
    if args.surface_extension_max_distance_px <= 0 or args.surface_extension_min_length_px <= 0:
        parser.error("surface extension distance and minimum length must be greater than zero.")
    if args.surface_extension_max_length_px < 0:
        parser.error("--surface-extension-max-length-px must be >=0; 0 means unlimited.")
    if not 0.0 <= args.surface_extension_min_support_ratio <= 1.0 or args.surface_extension_max_half_width_px <= 0:
        parser.error("surface extension support must be 0..1 and maximum half-width must be greater than zero.")
    if args.surface_connector_max_distance_px <= 0 or not -1.0 <= args.surface_connector_min_alignment <= 1.0:
        parser.error("surface connector distance must be positive and alignment must be between -1 and 1.")
    if not 0.0 <= args.surface_connector_min_support <= 1.0 or args.surface_connector_max_path_ratio < 1.0:
        parser.error("surface connector support must be 0..1 and max path ratio must be >= 1.")
    if not -1.0 <= args.surface_connector_fallback_min_alignment <= 1.0:
        parser.error("surface connector fallback alignment must be between -1 and 1.")
    if not 0.0 <= args.surface_connector_fallback_min_road_probability <= 1.0 or not 0.0 <= args.surface_connector_fallback_min_road_probability_q25 <= 1.0:
        parser.error("surface connector fallback road-probability thresholds must be between 0 and 1.")
    if args.surface_connector_fallback_max_path_ratio < 1.0:
        parser.error("surface connector fallback max path ratio must be >= 1.")
    if args.surface_pair_max_distance_px <= 0 or not -1.0 <= args.surface_pair_min_alignment <= 1.0:
        parser.error("surface-pair connector distance must be positive and alignment must be between -1 and 1.")
    if not 0.0 <= args.surface_pair_min_support <= 1.0 or args.surface_pair_max_path_ratio < 1.0:
        parser.error("surface-pair connector support must be 0..1 and max path ratio must be >= 1.")
    if not 0.0 <= args.surface_pair_min_road_probability <= 1.0 or not 0.0 <= args.surface_pair_min_road_probability_q25 <= 1.0:
        parser.error("surface-pair road-probability thresholds must be between 0 and 1.")
    if args.surface_pair_ambiguity_ratio <= 1.0:
        parser.error("surface-pair ambiguity ratio must be greater than 1.")
    if args.auto_gap_max_path_ratio < 1.0 or args.auto_gap_ambiguity_ratio <= 1.0:
        parser.error("auto-gap max path ratio must be >=1 and ambiguity ratio must be >1.")
    single_mode = bool(args.image or args.graph)
    batch_mode = bool(args.image_dir or args.graph_dir)
    if single_mode and batch_mode:
        parser.error("Use either --image/--graph single mode or --image-dir/--graph-dir batch mode, not both.")
    if batch_mode:
        if not args.image_dir or not args.graph_dir:
            parser.error("Batch mode requires both --image-dir and --graph-dir.")
    elif not args.image or not args.graph:
        parser.error("Single mode requires both --image and --graph, or use --image-dir and --graph-dir.")
    return args


def process_one(
    args: argparse.Namespace,
    image_path: Path,
    graph_path: Path,
    mask_path: Path | None,
    output_dir: Path,
    device: torch.device,
) -> dict:
    process_started = time.perf_counter()
    profiling = {
        "graph_load_seconds": 0.0,
        "road_surface_read_cleanup_seconds": 0.0,
        "road_probability_read_seconds": 0.0,
        "centerline_probability_read_seconds": 0.0,
        "topology_candidate_read_seconds": 0.0,
        "divided_road_recovery_seconds": 0.0,
        "junction_conflict_cleanup_seconds": 0.0,
        "junction_endpoint_candidate_search_seconds": 0.0,
        "surface_skeleton_candidate_analysis_seconds": 0.0,
        "a_star_path_search_seconds": 0.0,
        "graph_context_build_seconds": 0.0,
        "road_chain_build_seconds": 0.0,
        "file_writing_seconds": 0.0,
    }
    step_started = time.perf_counter()
    original_nodes_rc, original_edges = load_graph(graph_path)
    original_topology = graph_topology_metrics(original_nodes_rc, original_edges)
    profiling["graph_load_seconds"] = float(time.perf_counter() - step_started)

    step_started = time.perf_counter()
    if mask_path is not None:
        raw_mask = read_mask(mask_path)
        mask_source = str(mask_path)
    else:
        raw_mask = infer_molra_mask(
            image_path=image_path,
            sam_pretrained_path=Path(args.sam_pretrained_path),
            weight_path=Path(args.weight_path),
            device=device,
            tile=args.tile,
            overlap=args.overlap,
            threshold=args.molra_threshold,
            image_size=args.image_size,
        )
        mask_source = "sam_molra"
    profiling["road_surface_read_cleanup_seconds"] += float(time.perf_counter() - step_started)

    rgb = read_rgb_for_viz(image_path)
    if raw_mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"Image/mask shape mismatch: image={rgb.shape[:2]}, mask={raw_mask.shape[:2]}")
    h, w = raw_mask.shape[:2]
    invalid_nodes = np.where(
        (original_nodes_rc[:, 0] < -0.5)
        | (original_nodes_rc[:, 0] > h - 0.5)
        | (original_nodes_rc[:, 1] < -0.5)
        | (original_nodes_rc[:, 1] > w - 0.5)
    )[0]
    if invalid_nodes.size:
        raise ValueError(f"Graph/image alignment failed: {invalid_nodes.size} graph nodes lie outside the {h}x{w} raster.")
    pixel_size, pixel_size_source = resolve_pixel_size(image_path, args.pixel_size)

    step_started = time.perf_counter()
    binary = clean_mask(raw_mask, args.mask_threshold, args.close_kernel, args.open_kernel, args.min_area)
    profiling["road_surface_read_cleanup_seconds"] += float(time.perf_counter() - step_started)
    step_started = time.perf_counter()
    road_probability_path = Path(args.road_probability) if args.road_probability else auto_road_probability_path(mask_path, image_path.stem)
    if road_probability_path and road_probability_path.is_file():
        road_probability, road_probability_type = read_probability(road_probability_path, binary.shape)
        road_probability_source = str(road_probability_path)
    else:
        road_probability, road_probability_type = probability_from_u8(raw_mask)
        road_probability_source = mask_source
    profiling["road_probability_read_seconds"] = float(time.perf_counter() - step_started)
    step_started = time.perf_counter()
    center_probability_path = Path(args.centerline_probability) if args.centerline_probability else auto_centerline_probability_path(graph_path, image_path.stem)
    if center_probability_path and center_probability_path.is_file():
        center_probability, center_probability_type = read_probability(center_probability_path, binary.shape)
        center_probability_source = str(center_probability_path)
    else:
        center_probability = graph_buffer_mask(binary.shape, original_nodes_rc, original_edges, 2).astype(np.float32)
        center_probability_type = "graph_raster_fallback"
        center_probability_source = "rasterized_input_graph"
    profiling["centerline_probability_read_seconds"] = float(
        time.perf_counter() - step_started
    )
    step_started = time.perf_counter()
    original_topology_probabilities, topology_probability_source = load_edge_topology_probabilities(
        graph_path, original_nodes_rc, original_edges
    )
    profiling["topology_candidate_read_seconds"] += float(time.perf_counter() - step_started)

    nodes_rc, edges = original_nodes_rc, original_edges
    divided_road_repair_rows: list[dict] = []
    topology_candidate_path = candidate_path_for_graph(graph_path)
    if args.restore_divided_roads:
        step_started = time.perf_counter()
        topology_candidate_rows = read_topology_candidates(topology_candidate_path)
        profiling["topology_candidate_read_seconds"] += float(time.perf_counter() - step_started)
        step_started = time.perf_counter()
        edges, divided_road_repair_rows = optimize_divided_road_junctions(
            nodes_rc,
            edges,
            center_probability,
            topology_candidate_rows,
            min_candidate_probability=args.divided_road_min_candidate_probability,
            min_pair_spacing_px=args.divided_road_min_spacing_px,
            max_pair_spacing_px=args.divided_road_max_spacing_px,
            min_pair_score=args.divided_road_min_pair_score,
        )
        profiling["divided_road_recovery_seconds"] += float(time.perf_counter() - step_started)
    spur_rows: list[dict] = []
    if args.prune_spurs:
        nodes_rc, edges, spur_rows = prune_low_evidence_spurs(
            nodes_rc,
            edges,
            road_probability,
            center_probability,
            max_length_px=args.spur_max_length_px,
            max_road_probability=args.spur_max_road_probability,
            max_center_probability=args.spur_max_center_probability,
        )
    nodes_rc, recenter_rows = recenter_graph_nodes(
        nodes_rc,
        edges,
        binary,
        road_probability,
        max_shift_px=args.recenter_max_shift_px,
    )
    junction_repair_rows: list[dict] = []
    if args.repair_junctions:
        step_started = time.perf_counter()
        repair_context = GraphSpatialContext.build(
            nodes_rc, edges, cell_size=args.junction_repair_max_distance_px,
        )
        nodes_rc, edges, junction_repair_rows = repair_endpoint_to_edge_junctions(
            binary,
            road_probability,
            center_probability,
            nodes_rc,
            edges,
            max_distance_px=args.junction_repair_max_distance_px,
            min_alignment_cosine=args.auto_gap_min_alignment,
            min_surface_support=args.auto_gap_min_surface_support,
            path_margin_px=args.auto_gap_path_margin_px,
            outside_cost=args.auto_gap_outside_cost,
            sample_step_px=args.candidate_sample_step_px,
            max_target_parallel_cosine=args.junction_repair_max_parallel_cosine,
            graph_context=repair_context,
            profiling=profiling,
        )
        profiling["junction_endpoint_candidate_search_seconds"] += float(
            time.perf_counter() - step_started
        )
    junction_cleanup_rows: list[dict] = []
    junction_cleanup_started = time.perf_counter()
    if args.cleanup_junction_conflicts:
        cleanup_probabilities, _ = load_edge_topology_probabilities(graph_path, nodes_rc, edges)
        nodes_rc, edges, junction_cleanup_rows = cleanup_junction_conflicts(
            nodes_rc,
            edges,
            cleanup_probabilities,
            max_cycle_length_px=args.junction_cleanup_max_cycle_px,
            max_weak_probability=args.junction_cleanup_max_weak_probability,
            min_probability_margin=args.junction_cleanup_min_probability_margin,
        )
    junction_cleanup_seconds = time.perf_counter() - junction_cleanup_started
    profiling["junction_conflict_cleanup_seconds"] = float(junction_cleanup_seconds)
    divided_junction_rows: list[dict] = []
    if args.restore_divided_junction_corridors:
        step_started = time.perf_counter()
        nodes_rc, edges, divided_junction_rows = restore_divided_corridors_through_junctions(
            nodes_rc,
            edges,
            center_probability,
            junction_cleanup_rows,
            sample_step_px=args.candidate_sample_step_px,
        )
        profiling["divided_road_recovery_seconds"] += float(time.perf_counter() - step_started)
    if args.centerline_buffer_px > 0:
        binary = binary & graph_buffer_mask(binary.shape, nodes_rc, edges, args.centerline_buffer_px)

    step_started = time.perf_counter()
    graph_context = GraphSpatialContext.build(
        nodes_rc,
        edges,
        cell_size=max(
            32.0,
            args.junction_repair_max_distance_px,
            args.auto_gap_max_distance_px,
            args.surface_connector_max_distance_px,
            args.surface_extension_max_distance_px,
            args.candidate_graph_snap_px,
        ),
    )
    profiling["graph_context_build_seconds"] = float(time.perf_counter() - step_started)
    edge_rows = build_edge_surface_evidence(
        nodes_rc, edges, binary, graph_context=graph_context,
    )
    edge_provenance = load_edge_provenance(graph_path, nodes_rc, edges)
    for row, provenance in zip(edge_rows, edge_provenance):
        row.update(provenance)
    prepared_topology_probabilities, prepared_topology_probability_source = load_edge_topology_probabilities(
        graph_path, nodes_rc, edges
    )
    annotate_edge_topology(
        nodes_rc,
        edges,
        edge_rows,
        road_probability=road_probability,
        center_probability=center_probability,
        topology_probabilities=prepared_topology_probabilities,
        auto_retain_center_probability=args.auto_retain_center_probability,
        graph_context=graph_context,
    )
    step_started = time.perf_counter()
    surface_only, surface_only_rows = analyze_surface_only_regions(
        binary=binary,
        nodes_rc=nodes_rc,
        edges=edges,
        centerline_buffer_px=args.surface_only_buffer_px,
        endpoint_radius_px=args.surface_only_endpoint_radius_px,
        min_area=args.surface_only_min_area,
        min_candidate_skeleton_ratio=args.surface_only_min_skeleton_ratio,
        max_suspect_extent_ratio=args.surface_only_max_suspect_extent_ratio,
        graph_context=graph_context,
    )
    candidate_lines = build_candidate_centerlines(
        surface_only=surface_only,
        surface_only_rows=surface_only_rows,
        min_length_px=args.candidate_min_length_px,
        sample_step_px=args.candidate_sample_step_px,
        simplify_epsilon_px=args.candidate_simplify_epsilon_px,
        min_half_width_px=args.candidate_min_half_width_px,
    )
    profiling["surface_skeleton_candidate_analysis_seconds"] = float(
        time.perf_counter() - step_started
    )
    candidate_search_started = time.perf_counter()
    surface_connector_rows: list[dict] = []
    if args.connect_surface_skeletons:
        surface_connector_rows = connect_surface_skeletons_by_mask_path(
            candidate_lines,
            binary,
            road_probability,
            center_probability,
            nodes_rc,
            edges,
            max_distance_px=args.surface_connector_max_distance_px,
            min_alignment_cosine=args.surface_connector_min_alignment,
            min_surface_support=args.surface_connector_min_support,
            max_path_ratio=args.surface_connector_max_path_ratio,
            path_margin_px=args.surface_connector_path_margin_px,
            outside_cost=args.surface_connector_outside_cost,
            sample_step_px=args.candidate_sample_step_px,
            fallback_min_alignment=args.surface_connector_fallback_min_alignment,
            fallback_min_road_probability=args.surface_connector_fallback_min_road_probability,
            fallback_min_road_probability_q25=args.surface_connector_fallback_min_road_probability_q25,
            fallback_max_path_ratio=args.surface_connector_fallback_max_path_ratio,
            graph_context=graph_context,
            profiling=profiling,
        )
    endpoint_gap_candidates = []
    if args.auto_connect_gaps:
        surface_attached_graph_endpoints = {
            int(row["graph_endpoint_node_idx"])
            for row in surface_connector_rows
            if str(row.get("graph_endpoint_node_idx", "")).strip()
        }
        endpoint_gap_candidates = build_endpoint_gap_candidates(
            binary=binary,
            road_probability=road_probability,
            center_probability=center_probability,
            max_path_ratio=args.auto_gap_max_path_ratio,
            ambiguity_ratio=args.auto_gap_ambiguity_ratio,
            excluded_endpoint_ids=surface_attached_graph_endpoints,
            nodes_rc=nodes_rc,
            edges=edges,
            max_gap_px=args.auto_gap_max_distance_px,
            min_alignment_cosine=args.auto_gap_min_alignment,
            min_surface_support=args.auto_gap_min_surface_support,
            path_margin_px=args.auto_gap_path_margin_px,
            outside_cost=args.auto_gap_outside_cost,
            sample_step_px=args.candidate_sample_step_px,
            graph_context=graph_context,
            profiling=profiling,
        )
    for candidate in endpoint_gap_candidates:
        candidate["candidate_id"] = len(candidate_lines)
        candidate_lines.append(candidate)
    annotate_candidate_graph_matches(
        candidate_lines,
        nodes_rc,
        edges,
        args.candidate_graph_snap_px,
        auto_fuse_surface_components=args.auto_fuse_surface_components,
        auto_extend_surface_skeletons=args.auto_extend_surface_skeletons,
        surface_extension_min_alignment=args.surface_extension_min_alignment,
        surface_extension_max_distance_px=args.surface_extension_max_distance_px,
        surface_extension_min_length_px=args.surface_extension_min_length_px,
        surface_extension_max_length_px=args.surface_extension_max_length_px,
        surface_extension_min_support_ratio=args.surface_extension_min_support_ratio,
        surface_extension_max_half_width_px=args.surface_extension_max_half_width_px,
        center_probability=center_probability,
        image_shape=binary.shape,
        graph_context=graph_context,
    )
    surface_pair_connector_count = 0
    if args.connect_surface_skeleton_pairs:
        pair_candidates, pair_audit_rows = build_surface_skeleton_pair_connectors(
            candidate_lines,
            binary,
            road_probability,
            center_probability,
            max_distance_px=args.surface_pair_max_distance_px,
            min_alignment_cosine=args.surface_pair_min_alignment,
            min_surface_support=args.surface_pair_min_support,
            max_path_ratio=args.surface_pair_max_path_ratio,
            path_margin_px=args.surface_pair_path_margin_px,
            outside_cost=args.surface_connector_outside_cost,
            sample_step_px=args.candidate_sample_step_px,
            audit_start_id=len(surface_connector_rows),
            min_road_probability=args.surface_pair_min_road_probability,
            min_road_probability_q25=args.surface_pair_min_road_probability_q25,
            ambiguity_ratio=args.surface_pair_ambiguity_ratio,
            profiling=profiling,
        )
        for candidate in pair_candidates:
            candidate["candidate_id"] = len(candidate_lines)
            candidate_lines.append(candidate)
        surface_pair_connector_count = len(pair_candidates)
        surface_connector_rows.extend(pair_audit_rows)
    profiling["junction_endpoint_candidate_search_seconds"] += float(
        time.perf_counter() - candidate_search_started
    )
    conflict_rows = build_conflict_review_rows(edge_rows, surface_only_rows, candidate_lines)
    step_started = time.perf_counter()
    road_chains = build_road_chains(nodes_rc, edges, graph_context=graph_context)
    profiling["road_chain_build_seconds"] = float(time.perf_counter() - step_started)

    stem = image_path.stem
    writing_started = time.perf_counter()
    prepared_graph_path = output_dir / f"{stem}_prepared_graph.p"
    save_graph(prepared_graph_path, nodes_rc, edges)
    cv2.imwrite(str(output_dir / f"{stem}_molra_raw_mask.png"), raw_mask)
    cv2.imwrite(str(output_dir / f"{stem}_road_probability.png"), np.clip(road_probability * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{stem}_centerline_probability.png"), np.clip(center_probability * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{stem}_molra_clean_mask.png"), binary.astype(np.uint8) * 255)
    cv2.imwrite(str(output_dir / f"{stem}_surface_only.png"), surface_only.astype(np.uint8) * 255)
    if rgb.shape[:2] == binary.shape:
        viz = draw_viz(rgb, binary, nodes_rc, edges, [], edge_rows=edge_rows, surface_only=surface_only)
        cv2.imwrite(str(output_dir / f"{stem}_molra_width_viz.png"), viz)
        review_demo = draw_review_demo(
            rgb=rgb,
            binary=binary,
            surface_only=surface_only,
            nodes_rc=nodes_rc,
            edges=edges,
            edge_rows=edge_rows,
            surface_only_rows=surface_only_rows,
            candidate_lines=candidate_lines,
        )
        cv2.imwrite(str(output_dir / f"{stem}_review_demo.png"), review_demo)

    edge_fields = [
        "edge_id",
        "src_idx",
        "dst_idx",
        "length_px",
        "surface_support_ratio",
        "status",
        "confidence",
        "requires_review",
        "review_reasons",
        "src_degree",
        "dst_degree",
        "component_id",
        "component_node_count",
        "is_bridge",
        "is_dangling_edge",
        "topology_impact",
        "conflict_type",
        "review_priority_score",
        "review_priority",
        "centerline_policy",
        "mean_road_probability",
        "mean_centerline_probability",
        "topology_probability",
        "auto_retained",
        "line_source",
        "recovery_score",
        "center_conf",
        "surface_conf",
        "recovery_reason",
        "qa_state",
        "recovery_id",
    ]
    surface_only_fields = [
        "region_id",
        "status",
        "area_px",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "centroid_row",
        "centroid_col",
        "skeleton_len_px",
        "skeleton_area_ratio",
        "extent_ratio",
        "touches_graph_endpoint",
        "note",
    ]
    candidate_fields = [
        "candidate_id",
        "region_id",
        "branch_index",
        "candidate_type",
        "review_status",
        "action",
        "confidence",
        "auto_decision",
        "topology_impact",
        "topology_score",
        "surface_support_ratio",
        "direction_alignment",
        "path_ratio",
        "start_node_idx",
        "end_node_idx",
        "matched_start_node_idx",
        "matched_end_node_idx",
        "start_match_distance_px",
        "end_match_distance_px",
        "matched_component_count",
        "single_endpoint_extension_node_idx",
        "single_endpoint_extension_distance_px",
        "single_endpoint_extension_alignment",
        "single_endpoint_extension_length_px",
        "surface_region_candidate_count",
        "auto_score",
        "auto_rule",
        "auto_score_surface",
        "auto_score_shape_width",
        "auto_score_connection",
        "auto_score_direction_junction",
        "auto_score_centerline_weak",
        "auto_score_continuity_length",
        "hard_veto",
        "hard_veto_reasons",
        "effective_veto_reasons",
        "connection_type",
        "connection_alignment",
        "connection_endpoint_position",
        "connection_node_idx",
        "connection_node_row",
        "connection_node_col",
        "connection_start_node_row",
        "connection_start_node_col",
        "connection_end_node_row",
        "connection_end_node_col",
        "connection_edge_id",
        "connection_projection_row",
        "connection_projection_col",
        "candidate_tortuosity",
        "candidate_center_probability",
        "candidate_width_cv",
        "long_road_evidence",
        "long_road_evidence_reason",
        "surface_connector_count",
        "surface_connector_mode",
        "surface_connector_endpoint_positions",
        "surface_attachment_audit",
        "surface_attachment_kind",
        "surface_attachment_endpoint_position",
        "surface_attachment_node_row",
        "surface_attachment_node_col",
        "surface_attachment_projection_row",
        "surface_attachment_projection_col",
        "surface_attachment_surface_support_ratio",
        "surface_attachment_direction_alignment",
        "surface_attachment_path_ratio",
        "surface_attachment_road_probability_mean",
        "surface_attachment_road_probability_q25",
        "surface_attachment_evidence_mode",
        "road_probability_mean",
        "road_probability_q25",
        "evidence_mode",
        "start_row",
        "start_col",
        "end_row",
        "end_col",
        "length_px",
        "skeleton_pixels",
        "median_half_width_px",
        "point_count",
        "polyline_points_json",
        "area_px",
        "note",
    ]
    # Keep the historical filename for old GUI/finalizer compatibility. New
    # rows contain only geometry, surface support, probability and topology.
    write_csv(output_dir / f"{stem}_edge_widths.csv", edge_rows, edge_fields)
    write_csv(output_dir / f"{stem}_surface_only_regions.csv", surface_only_rows, surface_only_fields)
    write_csv(output_dir / f"{stem}_candidate_centerlines.csv", candidate_lines, candidate_fields)
    write_csv(
        output_dir / f"{stem}_road_chains.csv",
        road_chains,
        [
            "road_chain_id", "component_id", "start_node_idx", "end_node_idx", "start_degree", "end_degree",
            "micro_edge_count", "length_px", "edge_ids", "polyline_points_json",
        ],
    )
    conflict_fields = [
        "review_rank",
        "item_type",
        "item_id",
        "conflict_type",
        "topology_impact",
        "priority_score",
        "priority",
        "auto_action",
        "requires_manual_review",
        "confidence",
        "reason",
        "region_id",
        "edge_id",
        "candidate_id",
    ]
    write_csv(output_dir / f"{stem}_conflict_review.csv", conflict_rows, conflict_fields)
    write_csv(
        output_dir / f"{stem}_pruned_spurs.csv",
        spur_rows,
        ["spur_id", "edge_ids", "length_px", "road_probability", "centerline_probability", "action"],
    )
    write_csv(
        output_dir / f"{stem}_recentered_nodes.csv",
        recenter_rows,
        ["node_idx", "old_row", "old_col", "new_row", "new_col", "shift_px", "road_probability", "action"],
    )
    write_csv(
        output_dir / f"{stem}_junction_cleanup.csv",
        junction_cleanup_rows,
        [
            "cleanup_id", "action", "edge_id", "src_row", "src_col", "dst_row", "dst_col",
            "edge_length_px", "cycle_length_px", "topology_probability", "probability_margin",
        ],
    )
    write_csv(
        output_dir / f"{stem}_divided_junction_repairs.csv",
        divided_junction_rows,
        [
            "repair_id", "action", "center_row", "center_col", "north_anchor_nodes", "south_anchor_nodes",
            "north_spacing_px", "south_spacing_px", "removed_edge_count", "added_path_count",
            "restored_turn_count", "restored_turn_quadrants",
        ],
    )
    write_csv(
        output_dir / f"{stem}_junction_repairs.csv",
        junction_repair_rows,
        [
            "junction_repair_id", "endpoint_node_idx", "target_edge_id", "junction_row", "junction_col",
            "gap_distance_px", "direction_alignment", "surface_support_ratio", "topology_score", "action",
        ],
    )
    write_csv(
        output_dir / f"{stem}_surface_skeleton_connectors.csv",
        surface_connector_rows,
        [
            "connector_id", "connector_kind", "candidate_id", "target_candidate_id",
            "candidate_endpoint_position", "target_endpoint_position", "graph_endpoint_node_idx",
            "gap_distance_px", "path_length_px", "path_ratio", "direction_alignment",
            "surface_support_ratio", "road_probability_mean", "road_probability_q25",
            "evidence_mode", "topology_score", "action",
        ],
    )
    write_csv(
        output_dir / f"{stem}_divided_road_repairs.csv",
        divided_road_repair_rows,
        [
            "repair_id", "action", "first_endpoint", "second_endpoint", "first_target", "second_target",
            "first_topology_probability", "second_topology_probability", "first_direction_alignment",
            "second_direction_alignment", "first_center_support", "second_center_support", "approach_spacing_px",
            "target_spacing_px", "pair_score", "added_edge_count",
        ],
    )
    profiling["file_writing_seconds"] = float(time.perf_counter() - writing_started)

    status_counts: dict[str, int] = {}
    for row in edge_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    surface_only_counts: dict[str, int] = {}
    for row in surface_only_rows:
        status = str(row["status"])
        surface_only_counts[status] = surface_only_counts.get(status, 0) + 1

    summary = {
        "image": str(image_path),
        "graph": str(prepared_graph_path),
        "original_graph": str(graph_path),
        "prepared_graph": str(prepared_graph_path),
        "mask_source": mask_source,
        "device": str(device),
        "node_count": int(nodes_rc.shape[0]),
        "edge_count": int(edges.shape[0]),
        "pre_review_width_estimation": False,
        "sample_count": 0,
        "edge_width_count": 0,
        "edge_evidence_count": int(len(edge_rows)),
        "edge_status_counts": status_counts,
        "surface_only_region_count": int(len(surface_only_rows)),
        "surface_only_status_counts": surface_only_counts,
        "candidate_centerline_count": int(len(candidate_lines)),
        "auto_endpoint_gap_count": int(len(endpoint_gap_candidates)),
        "surface_skeleton_connector_count": int(len(surface_connector_rows)),
        "surface_skeleton_pair_connector_count": int(surface_pair_connector_count),
        "auto_surface_component_fusion_count": int(sum(row.get("action") == "auto_fuse_surface_centerline" for row in candidate_lines)),
        "auto_surface_endpoint_extension_count": int(sum(row.get("action") == "auto_extend_surface_centerline" for row in candidate_lines)),
        "conflict_review_item_count": int(len(conflict_rows)),
        "manual_review_item_count": int(sum(bool(row["requires_manual_review"]) for row in conflict_rows)),
        "auto_handled_conflict_count": int(sum(not bool(row["requires_manual_review"]) for row in conflict_rows)),
        "auto_retained_line_without_surface_count": int(sum(bool(row.get("auto_retained")) for row in edge_rows)),
        "strong_edge_count": int(sum(row.get("line_source", "samroad") == "samroad" for row in edge_rows)),
        "weak_recovered_edge_count": int(sum(row.get("line_source") == "weak_recovered" for row in edge_rows)),
        "surface_supported_recovery_count": int(sum(
            row.get("recovery_reason") == "weak_probability_surface_supported" for row in edge_rows
        )),
        "width_segment_count": 0,
        "road_chain_count": len(road_chains),
        "original_topology": original_topology,
        "prepared_topology": graph_topology_metrics(
            nodes_rc, edges, graph_context=graph_context,
        ),
        "pruned_spur_count": len(spur_rows),
        "recentered_node_count": len(recenter_rows),
        "junction_cleanup_count": len(junction_cleanup_rows),
        "divided_junction_repair_count": len(divided_junction_rows),
        "junction_repair_count": len(junction_repair_rows),
        "divided_road_repair_count": len(divided_road_repair_rows),
        "topology_candidate_source": str(topology_candidate_path) if topology_candidate_path.is_file() else "not_available",
        "road_probability_type": road_probability_type,
        "road_probability_source": road_probability_source,
        "centerline_probability_source": center_probability_source,
        "centerline_probability_type": center_probability_type,
        "topology_probability_source": topology_probability_source,
        "prepared_topology_probability_source": prepared_topology_probability_source,
        "profiling": {
            **profiling,
            # Preserve the existing key consumed by older reports.
            "junction_cleanup_seconds": float(junction_cleanup_seconds),
            "total_seconds": float(time.perf_counter() - process_started),
        },
        "sample_step_px": args.sample_step_px,
        "normal_step_px": args.normal_step_px,
        "max_search_px": args.max_search_px,
        "snap_radius_px": args.snap_radius_px,
        "max_snap_distance_px": args.max_snap_distance_px,
        "junction_buffer_px": args.junction_buffer_px,
        "border_margin_px": args.border_margin_px,
        "max_asymmetry_ratio": args.max_asymmetry_ratio,
        "min_edge_coverage": args.min_edge_coverage,
        "short_gap_px": args.short_gap_px,
        "max_width_cv": args.max_width_cv,
        "width_change_ratio": args.width_change_ratio,
        "width_change_min_samples": args.width_change_min_samples,
        "outlier_mad_scale": args.outlier_mad_scale,
        "auto_connect_gaps": args.auto_connect_gaps,
        "auto_gap_max_distance_px": args.auto_gap_max_distance_px,
        "auto_gap_min_alignment": args.auto_gap_min_alignment,
        "auto_gap_min_surface_support": args.auto_gap_min_surface_support,
        "auto_gap_max_path_ratio": args.auto_gap_max_path_ratio,
        "auto_gap_ambiguity_ratio": args.auto_gap_ambiguity_ratio,
        "auto_gap_path_margin_px": args.auto_gap_path_margin_px,
        "auto_gap_outside_cost": args.auto_gap_outside_cost,
        "auto_extend_surface_skeletons": args.auto_extend_surface_skeletons,
        "connect_surface_skeletons": args.connect_surface_skeletons,
        "surface_connector_max_distance_px": args.surface_connector_max_distance_px,
        "surface_connector_min_alignment": args.surface_connector_min_alignment,
        "surface_connector_min_support": args.surface_connector_min_support,
        "surface_connector_max_path_ratio": args.surface_connector_max_path_ratio,
        "surface_connector_path_margin_px": args.surface_connector_path_margin_px,
        "surface_connector_outside_cost": args.surface_connector_outside_cost,
        "surface_connector_fallback_min_alignment": args.surface_connector_fallback_min_alignment,
        "surface_connector_fallback_min_road_probability": args.surface_connector_fallback_min_road_probability,
        "surface_connector_fallback_min_road_probability_q25": args.surface_connector_fallback_min_road_probability_q25,
        "surface_connector_fallback_max_path_ratio": args.surface_connector_fallback_max_path_ratio,
        "connect_surface_skeleton_pairs": args.connect_surface_skeleton_pairs,
        "surface_pair_max_distance_px": args.surface_pair_max_distance_px,
        "surface_pair_min_alignment": args.surface_pair_min_alignment,
        "surface_pair_min_support": args.surface_pair_min_support,
        "surface_pair_max_path_ratio": args.surface_pair_max_path_ratio,
        "surface_pair_path_margin_px": args.surface_pair_path_margin_px,
        "surface_pair_min_road_probability": args.surface_pair_min_road_probability,
        "surface_pair_min_road_probability_q25": args.surface_pair_min_road_probability_q25,
        "surface_pair_ambiguity_ratio": args.surface_pair_ambiguity_ratio,
        "surface_extension_min_alignment": args.surface_extension_min_alignment,
        "surface_extension_max_distance_px": args.surface_extension_max_distance_px,
        "surface_extension_min_length_px": args.surface_extension_min_length_px,
        "surface_extension_max_length_px": args.surface_extension_max_length_px,
        "surface_extension_min_support_ratio": args.surface_extension_min_support_ratio,
        "surface_extension_max_half_width_px": args.surface_extension_max_half_width_px,
        "pixel_size": pixel_size,
        "pixel_size_source": pixel_size_source,
    }
    with open(output_dir / f"{stem}_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def _process_one_worker(payload: tuple[dict, str, str, str, str]) -> dict:
    """Spawn-safe one-slice worker; every output name is stem-scoped."""
    args_values, image_value, graph_value, mask_value, output_value = payload
    image_path = Path(image_value)
    graph_path = Path(graph_value)
    mask_path = Path(mask_value) if mask_value else None
    try:
        summary = process_one(
            argparse.Namespace(**args_values),
            image_path,
            graph_path,
            mask_path,
            Path(output_value),
            resolve_device(str(args_values.get("device", "auto"))),
        )
        return {"ok": True, "summary": summary}
    except Exception as exc:
        return {
            "ok": False,
            "failure": {
                "image": str(image_path),
                "graph": str(graph_path),
                "mask": str(mask_path or ""),
                "error": str(exc),
            },
        }


def main() -> int:
    batch_started = time.perf_counter()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.image_dir:
        mask_dir = Path(args.mask_dir) if args.mask_dir else None
        jobs = build_batch_jobs(Path(args.image_dir), Path(args.graph_dir), mask_dir)
    else:
        jobs = [(Path(args.image), Path(args.graph), Path(args.mask) if args.mask else None)]

    summaries = []
    failures = []
    worker_count = resolve_worker_count(args.workers, len(jobs))
    if worker_count > 1 and any(mask_path is None or not mask_path.is_file() for _, _, mask_path in jobs):
        print("Parallel width processing disabled because at least one slice would invoke SAM_MLoRA inference.")
        worker_count = 1
    print(f"Width slice workers: {worker_count or 1}; slice count: {len(jobs)}")
    if worker_count <= 1:
        device = resolve_device(args.device)
        for index, (image_path, graph_path, mask_path) in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] Processing {image_path.name}")
            try:
                summaries.append(process_one(args, image_path, graph_path, mask_path, output_dir, device))
            except Exception as exc:
                failures.append({"image": str(image_path), "graph": str(graph_path), "mask": str(mask_path or ""), "error": str(exc)})
                print(f"Failed: {image_path} -> {exc}")
    else:
        payloads = [
            (
                dict(vars(args)), str(image_path), str(graph_path),
                str(mask_path) if mask_path is not None else "", str(output_dir),
            )
            for image_path, graph_path, mask_path in jobs
        ]
        results = spawn_map(_process_one_worker, payloads, worker_count)
        for result in results:
            if result["ok"]:
                summaries.append(result["summary"])
            else:
                failures.append(result["failure"])
                print(f"Failed: {result['failure']['image']} -> {result['failure']['error']}")

    batch_summary = {
        "output_dir": str(output_dir),
        "job_count": len(jobs),
        "success_count": len(summaries),
        "failure_count": len(failures),
        "workers": int(worker_count),
        "failures": failures,
        "profiling": {
            "slice_total_seconds": float(sum(
                (summary.get("profiling") or {}).get("total_seconds", 0.0)
                for summary in summaries
            )),
            "wall_seconds": float(time.perf_counter() - batch_started),
        },
    }
    # The production pipeline consumes this stable batch contract even when a
    # developer test contains exactly one image.
    with open(output_dir / "batch_width_summary.json", "w", encoding="utf-8") as file:
        json.dump(batch_summary, file, indent=2, ensure_ascii=False)
    if len(jobs) > 1:
        print(json.dumps(batch_summary, indent=2, ensure_ascii=False))
    print(f"Outputs written to: {output_dir}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
