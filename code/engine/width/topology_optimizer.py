from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_topology_candidates(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def candidate_path_for_graph(graph_path: Path) -> Path:
    return graph_path.with_name(f"{graph_path.stem}_edge_candidates.csv")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-6)


def _graph_adjacency(node_count: int, edges: np.ndarray) -> list[list[int]]:
    adjacency = [[] for _ in range(node_count)]
    for src, dst in edges.tolist():
        adjacency[int(src)].append(int(dst))
        adjacency[int(dst)].append(int(src))
    return adjacency


def _endpoint_outward_vectors(nodes_rc: np.ndarray, edges: np.ndarray) -> dict[int, np.ndarray]:
    adjacency = _graph_adjacency(nodes_rc.shape[0], edges)
    return {
        node_idx: _unit(nodes_rc[node_idx] - nodes_rc[neighbors[0]])
        for node_idx, neighbors in enumerate(adjacency)
        if len(neighbors) == 1
    }


def _sample_segment(probability: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    length = max(1, int(np.ceil(float(np.linalg.norm(end - start)))))
    rows = np.clip(np.rint(np.linspace(start[0], end[0], length + 1)).astype(np.int32), 0, probability.shape[0] - 1)
    cols = np.clip(np.rint(np.linspace(start[1], end[1], length + 1)).astype(np.int32), 0, probability.shape[1] - 1)
    return float(np.mean(probability[rows, cols]))


def _match_node(nodes_rc: np.ndarray, point: np.ndarray, tolerance: float) -> int:
    if nodes_rc.size == 0:
        return -1
    distances = np.linalg.norm(nodes_rc - point, axis=1)
    node_idx = int(np.argmin(distances))
    return node_idx if float(distances[node_idx]) <= tolerance else -1


def _indexed_candidates(
    nodes_rc: np.ndarray,
    rows: list[dict],
    match_tolerance_px: float,
    min_probability: float,
) -> tuple[dict[int, list[tuple[int, float]]], dict[tuple[int, int], float]]:
    by_node: dict[int, list[tuple[int, float]]] = {}
    score_by_edge: dict[tuple[int, int], float] = {}
    for row in rows:
        try:
            probability = float(row.get("topology_probability", 0.0) or 0.0)
            start = np.asarray([float(row["src_row"]), float(row["src_col"])], dtype=np.float32)
            end = np.asarray([float(row["dst_row"]), float(row["dst_col"])], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            continue
        if probability < min_probability:
            continue
        src_idx = _match_node(nodes_rc, start, match_tolerance_px)
        dst_idx = _match_node(nodes_rc, end, match_tolerance_px)
        if src_idx < 0 or dst_idx < 0 or src_idx == dst_idx:
            continue
        key = tuple(sorted((src_idx, dst_idx)))
        score_by_edge[key] = max(score_by_edge.get(key, 0.0), probability)
    for (src_idx, dst_idx), probability in score_by_edge.items():
        by_node.setdefault(src_idx, []).append((dst_idx, probability))
        by_node.setdefault(dst_idx, []).append((src_idx, probability))
    return by_node, score_by_edge


def optimize_divided_road_junctions(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    center_probability: np.ndarray,
    candidate_rows: list[dict],
    *,
    min_candidate_probability: float = 0.20,
    match_tolerance_px: float = 1.5,
    min_pair_spacing_px: float = 5.0,
    max_pair_spacing_px: float = 40.0,
    min_parallel_cosine: float = 0.90,
    max_lateral_cosine: float = 0.55,
    min_forward_cosine: float = 0.55,
    min_center_support: float = 0.35,
    min_pair_score: float = 0.72,
) -> tuple[np.ndarray, list[dict]]:
    """Recover two one-to-one continuations for a divided-road approach.

    Only paired dangling endpoints are considered. The chosen targets must stay
    separated laterally, so the optimizer cannot turn two carriageways into one
    shared centerline node.
    """
    if edges.size == 0 or not candidate_rows:
        return edges, []
    outward = _endpoint_outward_vectors(nodes_rc, edges)
    candidate_by_node, score_by_edge = _indexed_candidates(
        nodes_rc, candidate_rows, match_tolerance_px, min_candidate_probability
    )
    existing = {tuple(sorted((int(src), int(dst)))) for src, dst in edges.tolist()}
    endpoint_ids = sorted(outward)
    proposals = []
    for position, first_idx in enumerate(endpoint_ids):
        for second_idx in endpoint_ids[position + 1 :]:
            connector = nodes_rc[second_idx] - nodes_rc[first_idx]
            spacing = float(np.linalg.norm(connector))
            if not min_pair_spacing_px <= spacing <= max_pair_spacing_px:
                continue
            first_outward, second_outward = outward[first_idx], outward[second_idx]
            if float(np.dot(first_outward, second_outward)) < min_parallel_cosine:
                continue
            lateral_direction = connector / max(spacing, 1e-6)
            if max(abs(float(np.dot(lateral_direction, first_outward))), abs(float(np.dot(lateral_direction, second_outward)))) > max_lateral_cosine:
                continue

            first_options = []
            for target_idx, probability in candidate_by_node.get(first_idx, []):
                direction = _unit(nodes_rc[target_idx] - nodes_rc[first_idx])
                alignment = float(np.dot(first_outward, direction))
                support = _sample_segment(center_probability, nodes_rc[first_idx], nodes_rc[target_idx])
                if alignment >= min_forward_cosine and support >= min_center_support:
                    first_options.append((target_idx, probability, alignment, support))
            second_options = []
            for target_idx, probability in candidate_by_node.get(second_idx, []):
                direction = _unit(nodes_rc[target_idx] - nodes_rc[second_idx])
                alignment = float(np.dot(second_outward, direction))
                support = _sample_segment(center_probability, nodes_rc[second_idx], nodes_rc[target_idx])
                if alignment >= min_forward_cosine and support >= min_center_support:
                    second_options.append((target_idx, probability, alignment, support))

            for first_target, first_probability, first_alignment, first_support in first_options:
                for second_target, second_probability, second_alignment, second_support in second_options:
                    if first_target == second_target or first_target == second_idx or second_target == first_idx:
                        continue
                    target_connector = nodes_rc[second_target] - nodes_rc[first_target]
                    target_spacing = float(np.linalg.norm(target_connector))
                    if not min_pair_spacing_px <= target_spacing <= max_pair_spacing_px * 1.5:
                        continue
                    first_direction = _unit(nodes_rc[first_target] - nodes_rc[first_idx])
                    second_direction = _unit(nodes_rc[second_target] - nodes_rc[second_idx])
                    continuation_parallel = float(np.dot(first_direction, second_direction))
                    if continuation_parallel < min_parallel_cosine:
                        continue
                    target_lateral = target_connector / max(target_spacing, 1e-6)
                    if max(abs(float(np.dot(target_lateral, first_direction))), abs(float(np.dot(target_lateral, second_direction)))) > max_lateral_cosine:
                        continue
                    topology = 0.5 * (first_probability + second_probability)
                    alignment = 0.5 * (first_alignment + second_alignment)
                    support = 0.5 * (first_support + second_support)
                    spacing_consistency = 1.0 - min(1.0, abs(target_spacing - spacing) / max(spacing, 1.0))
                    score = 0.50 * topology + 0.25 * alignment + 0.15 * support + 0.10 * spacing_consistency
                    if score >= min_pair_score:
                        proposals.append(
                            (
                                score,
                                first_idx,
                                second_idx,
                                first_target,
                                second_target,
                                first_probability,
                                second_probability,
                                first_alignment,
                                second_alignment,
                                first_support,
                                second_support,
                                spacing,
                                target_spacing,
                            )
                        )

    used_endpoints: set[int] = set()
    added_edges: list[tuple[int, int]] = []
    audit_rows: list[dict] = []
    for proposal in sorted(proposals, reverse=True):
        (
            score,
            first_idx,
            second_idx,
            first_target,
            second_target,
            first_probability,
            second_probability,
            first_alignment,
            second_alignment,
            first_support,
            second_support,
            spacing,
            target_spacing,
        ) = proposal
        if first_idx in used_endpoints or second_idx in used_endpoints:
            continue
        pair_edges = [tuple(sorted((first_idx, first_target))), tuple(sorted((second_idx, second_target)))]
        new_pair_edges = [edge for edge in pair_edges if edge not in existing]
        if not new_pair_edges:
            continue
        added_edges.extend(new_pair_edges)
        existing.update(new_pair_edges)
        used_endpoints.update({first_idx, second_idx})
        audit_rows.append(
            {
                "repair_id": len(audit_rows),
                "action": "restore_divided_road_pair",
                "first_endpoint": first_idx,
                "second_endpoint": second_idx,
                "first_target": first_target,
                "second_target": second_target,
                "first_topology_probability": first_probability,
                "second_topology_probability": second_probability,
                "first_direction_alignment": first_alignment,
                "second_direction_alignment": second_alignment,
                "first_center_support": first_support,
                "second_center_support": second_support,
                "approach_spacing_px": spacing,
                "target_spacing_px": target_spacing,
                "pair_score": score,
                "added_edge_count": len(new_pair_edges),
            }
        )
    if not added_edges:
        return edges, audit_rows
    combined = np.asarray(edges.tolist() + added_edges, dtype=np.int32).reshape(-1, 2)
    return combined, audit_rows
