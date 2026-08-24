from __future__ import annotations

"""Deterministic graph caches and conservative spatial candidate indexes."""

from dataclasses import dataclass, field
import json
from math import floor

import numpy as np


def _cell(value: float, cell_size: float) -> int:
    return int(floor(float(value) / cell_size))


@dataclass(frozen=True)
class PointGridIndex:
    points_rc: np.ndarray
    cell_size: float
    cells: dict[tuple[int, int], tuple[int, ...]]
    row_cell_min: int
    row_cell_max: int
    col_cell_min: int
    col_cell_max: int

    @classmethod
    def build(cls, points_rc: np.ndarray, cell_size: float) -> "PointGridIndex":
        size = max(float(cell_size), 1.0)
        grouped: dict[tuple[int, int], list[int]] = {}
        for point_id, (row, col) in enumerate(np.asarray(points_rc).reshape(-1, 2)):
            grouped.setdefault((_cell(row, size), _cell(col, size)), []).append(point_id)
        cells = {key: tuple(sorted(value)) for key, value in grouped.items()}
        row_cells = [key[0] for key in cells]
        col_cells = [key[1] for key in cells]
        return cls(
            np.asarray(points_rc), size, cells,
            min(row_cells, default=0), max(row_cells, default=0),
            min(col_cells, default=0), max(col_cells, default=0),
        )

    def query_box(
        self, row_min: float, row_max: float, col_min: float, col_max: float,
    ) -> list[int]:
        if row_min > row_max or col_min > col_max:
            return []
        candidates: set[int] = set()
        for row_cell in range(_cell(row_min, self.cell_size), _cell(row_max, self.cell_size) + 1):
            for col_cell in range(_cell(col_min, self.cell_size), _cell(col_max, self.cell_size) + 1):
                candidates.update(self.cells.get((row_cell, col_cell), ()))
        return [
            point_id for point_id in sorted(candidates)
            if row_min <= float(self.points_rc[point_id, 0]) <= row_max
            and col_min <= float(self.points_rc[point_id, 1]) <= col_max
        ]

    def query_radius_box(self, point_rc: np.ndarray, radius: float) -> list[int]:
        row, col = (float(value) for value in point_rc)
        distance = max(float(radius), 0.0)
        return self.query_box(row - distance, row + distance, col - distance, col + distance)

    def nearest(self, point_rc: np.ndarray) -> tuple[float, int]:
        if not self.cells:
            return float("inf"), -1
        row, col = (float(value) for value in point_rc)
        row_cell, col_cell = _cell(row, self.cell_size), _cell(col, self.cell_size)
        max_ring = max(
            abs(row_cell - self.row_cell_min), abs(row_cell - self.row_cell_max),
            abs(col_cell - self.col_cell_min), abs(col_cell - self.col_cell_max),
        )
        best = (float("inf"), -1)
        visited: set[int] = set()
        for ring in range(max_ring + 1):
            for current_row in range(row_cell - ring, row_cell + ring + 1):
                for current_col in range(col_cell - ring, col_cell + ring + 1):
                    if ring and max(abs(current_row - row_cell), abs(current_col - col_cell)) != ring:
                        continue
                    for point_id in self.cells.get((current_row, current_col), ()):
                        if point_id in visited:
                            continue
                        visited.add(point_id)
                        distance = float(np.linalg.norm(self.points_rc[point_id] - point_rc))
                        best = min(best, (distance, point_id))
            covered_row_min = (row_cell - ring) * self.cell_size
            covered_row_max = (row_cell + ring + 1) * self.cell_size
            covered_col_min = (col_cell - ring) * self.cell_size
            covered_col_max = (col_cell + ring + 1) * self.cell_size
            nearest_unvisited_boundary = min(
                row - covered_row_min, covered_row_max - row,
                col - covered_col_min, covered_col_max - col,
            )
            if best[0] <= nearest_unvisited_boundary:
                return best
        return best


def find_vertical_divided_anchor_pair(
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
    """Return the exact legacy-scored pair from a conservative spatial window."""
    node_index = node_index or PointGridIndex.build(
        nodes_rc, max(max_row_difference_px, max_spacing_px, 1.0),
    )
    if side < 0:
        row_min = float(center[0]) - max_distance_px
        row_max = float(center[0]) - min_distance_px
    else:
        row_min = float(center[0]) + min_distance_px
        row_max = float(center[0]) + max_distance_px
    eligible_ids = node_index.query_box(
        row_min,
        row_max,
        float(center[1]) - lateral_search_px,
        float(center[1]) + lateral_search_px,
    )
    if len(eligible_ids) < 2:
        return None
    eligible_array = np.asarray(eligible_ids, dtype=np.int64)
    eligible_rows = np.asarray(nodes_rc[eligible_array, 0], dtype=np.float64)
    row_order = np.lexsort((eligible_array, eligible_rows))
    row_sorted_ids = eligible_array[row_order]
    row_sorted_values = eligible_rows[row_order]
    best: tuple[float, tuple[int, int]] | None = None
    for first_idx in eligible_ids:
        first = nodes_rc[first_idx]
        first_distance = side * float(first[0] - center[0])
        if not min_distance_px <= first_distance <= max_distance_px:
            continue
        if abs(float(first[1] - center[1])) > lateral_search_px:
            continue
        row_start = int(np.searchsorted(
            row_sorted_values, float(first[0]) - max_row_difference_px,
            side="left",
        ))
        row_stop = int(np.searchsorted(
            row_sorted_values, float(first[0]) + max_row_difference_px,
            side="right",
        ))
        second_ids = row_sorted_ids[row_start:row_stop]
        second_ids = second_ids[second_ids > first_idx]
        if not len(second_ids):
            continue
        second_points = nodes_rc[second_ids]
        spacings = np.abs(
            np.asarray(second_points[:, 1], dtype=np.float64) - float(first[1])
        )
        spacing_mask = (spacings >= min_spacing_px) & (spacings <= max_spacing_px)
        second_ids = second_ids[spacing_mask]
        second_points = second_points[spacing_mask]
        if not len(second_ids):
            continue

        first_rc = np.rint(first).astype(np.int32)
        second_rc = np.rint(second_points).astype(np.int32)
        # Keep the legacy float32 probability addition before applying its
        # float64 score weights. The row and lateral terms mirror the original
        # scalar float conversions.
        probabilities = np.asarray(
            center_probability[first_rc[0], first_rc[1]]
            + center_probability[second_rc[:, 0], second_rc[:, 1]],
            dtype=np.float32,
        )
        second_distances = side * (
            np.asarray(second_points[:, 0], dtype=np.float64) - float(center[0])
        )
        outward_distances = 0.5 * (first_distance + second_distances)
        lateral_centers = np.abs(
            0.5 * (float(first[1]) + np.asarray(second_points[:, 1], dtype=np.float64))
            - float(center[1])
        )
        scores = outward_distances + 0.7 * lateral_centers - 20.0 * probabilities
        minimum_score = float(np.min(scores))
        minimum_positions = np.flatnonzero(scores == minimum_score)
        minimum_second_ids = second_ids[minimum_positions]
        minimum_second_cols = second_points[minimum_positions, 1]
        ordered_first = np.where(
            float(first[1]) <= minimum_second_cols, first_idx, minimum_second_ids,
        )
        ordered_second = np.where(
            float(first[1]) <= minimum_second_cols, minimum_second_ids, first_idx,
        )
        ordered_position = int(np.lexsort((ordered_second, ordered_first))[0])
        candidate = (
            minimum_score,
            (
                int(ordered_first[ordered_position]),
                int(ordered_second[ordered_position]),
            ),
        )
        best = candidate if best is None else min(best, candidate)
    return best[1] if best is not None else None


def points_within_radius_of_references(
    points_rc: np.ndarray,
    references_rc: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Equivalent bounded-memory replacement for an all-pairs distance matrix."""
    points = np.asarray(points_rc, dtype=np.float32).reshape(-1, 2)
    references = np.asarray(references_rc, dtype=np.float32).reshape(-1, 2)
    result = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(references):
        return result
    distance_limit = max(float(radius), 0.0)
    index = PointGridIndex.build(references, max(distance_limit, 1.0))
    for point_id, point in enumerate(points):
        candidate_ids = index.query_radius_box(point, distance_limit)
        result[point_id] = any(
            float(np.linalg.norm(references[reference_id] - point)) <= distance_limit
            for reference_id in candidate_ids
        )
    return result


@dataclass(frozen=True)
class EdgeGridIndex:
    nodes_rc: np.ndarray
    edges: np.ndarray
    cell_size: float
    cells: dict[tuple[int, int], tuple[int, ...]]

    @classmethod
    def build(
        cls, nodes_rc: np.ndarray, edges: np.ndarray, cell_size: float,
    ) -> "EdgeGridIndex":
        nodes = np.asarray(nodes_rc).reshape(-1, 2)
        graph_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        size = max(float(cell_size), 1.0)
        grouped: dict[tuple[int, int], list[int]] = {}
        for edge_id, (src_idx, dst_idx) in enumerate(graph_edges.tolist()):
            start, end = nodes[int(src_idx)], nodes[int(dst_idx)]
            row_min, row_max = sorted((float(start[0]), float(end[0])))
            col_min, col_max = sorted((float(start[1]), float(end[1])))
            for row_cell in range(_cell(row_min, size), _cell(row_max, size) + 1):
                for col_cell in range(_cell(col_min, size), _cell(col_max, size) + 1):
                    grouped.setdefault((row_cell, col_cell), []).append(edge_id)
        return cls(
            nodes, graph_edges, size,
            {key: tuple(sorted(set(value))) for key, value in grouped.items()},
        )

    def query_point_radius(self, point_rc: np.ndarray, radius: float) -> list[int]:
        row, col = (float(value) for value in point_rc)
        distance = max(float(radius), 0.0)
        row_min, row_max = row - distance, row + distance
        col_min, col_max = col - distance, col + distance
        candidates: set[int] = set()
        for row_cell in range(_cell(row_min, self.cell_size), _cell(row_max, self.cell_size) + 1):
            for col_cell in range(_cell(col_min, self.cell_size), _cell(col_max, self.cell_size) + 1):
                candidates.update(self.cells.get((row_cell, col_cell), ()))
        result = []
        for edge_id in sorted(candidates):
            src_idx, dst_idx = (int(value) for value in self.edges[edge_id])
            start, end = self.nodes_rc[src_idx], self.nodes_rc[dst_idx]
            edge_row_min, edge_row_max = sorted((float(start[0]), float(end[0])))
            edge_col_min, edge_col_max = sorted((float(start[1]), float(end[1])))
            if (
                edge_row_max >= row_min and edge_row_min <= row_max
                and edge_col_max >= col_min and edge_col_min <= col_max
            ):
                result.append(edge_id)
        return result


@dataclass
class GraphSpatialContext:
    nodes_rc: np.ndarray
    edges: np.ndarray
    adjacency: tuple[tuple[tuple[int, int, float], ...], ...]
    node_edges: tuple[tuple[int, ...], ...]
    degrees: np.ndarray
    component_ids: np.ndarray
    bridge_edge_ids: frozenset[int]
    endpoints: tuple[int, ...]
    outward_vectors: dict[int, np.ndarray]
    edge_lengths: np.ndarray
    endpoint_index: PointGridIndex
    endpoint_node_ids: np.ndarray
    node_index: PointGridIndex
    edge_index: EdgeGridIndex
    road_chains: list[dict] | None = field(default=None, repr=False)

    def nearest_node(self, point_rc: np.ndarray) -> tuple[float, int]:
        return self.node_index.nearest(point_rc)

    def nearest_endpoint(self, point_rc: np.ndarray) -> tuple[float, int]:
        distance, position = self.endpoint_index.nearest(point_rc)
        return (
            (distance, int(self.endpoint_node_ids[position]))
            if position >= 0 else (float("inf"), -1)
        )

    def endpoint_ids_within(self, point_rc: np.ndarray, radius: float) -> list[int]:
        positions = self.endpoint_index.query_radius_box(point_rc, radius)
        return sorted(int(self.endpoint_node_ids[int(position)]) for position in positions)

    def build_road_chain_rows(self) -> list[dict]:
        if self.road_chains is not None:
            return self.road_chains
        visited: set[int] = set()
        chains: list[tuple[list[int], list[int]]] = []

        def trace(start_node: int, first_edge: int) -> tuple[list[int], list[int]]:
            node_path = [start_node]
            edge_path: list[int] = []
            current_node = start_node
            current_edge = first_edge
            while current_edge not in visited:
                visited.add(current_edge)
                edge_path.append(current_edge)
                src_idx, dst_idx = (
                    int(value) for value in self.edges[current_edge]
                )
                next_node = dst_idx if src_idx == current_node else src_idx
                node_path.append(next_node)
                if self.degrees[next_node] != 2:
                    break
                next_edges = [
                    edge_id for edge_id in self.node_edges[next_node]
                    if edge_id not in visited
                ]
                if not next_edges:
                    break
                current_node, current_edge = next_node, next_edges[0]
            return node_path, edge_path

        for node_idx in np.where(self.degrees != 2)[0].tolist():
            for edge_id in self.node_edges[node_idx]:
                if edge_id in visited:
                    continue
                node_path, edge_path = trace(node_idx, edge_id)
                if edge_path:
                    chains.append((node_path, edge_path))
        for edge_id in range(len(self.edges)):
            if edge_id in visited:
                continue
            node_path, edge_path = trace(int(self.edges[edge_id, 0]), edge_id)
            if edge_path:
                chains.append((node_path, edge_path))

        rows = []
        for chain_id, (node_path, edge_path) in enumerate(chains):
            points = [self.nodes_rc[node_idx] for node_idx in node_path]
            length = float(sum(
                np.linalg.norm(second - first)
                for first, second in zip(points[:-1], points[1:])
            ))
            start_idx, end_idx = node_path[0], node_path[-1]
            rows.append({
                "road_chain_id": chain_id,
                "component_id": int(self.component_ids[start_idx]),
                "start_node_idx": start_idx,
                "end_node_idx": end_idx,
                "start_degree": int(self.degrees[start_idx]),
                "end_degree": int(self.degrees[end_idx]),
                "micro_edge_count": len(edge_path),
                "length_px": length,
                "edge_ids": ";".join(str(value) for value in edge_path),
                "polyline_points_json": json.dumps(
                    [[float(point[0]), float(point[1])] for point in points],
                    separators=(",", ":"),
                ),
            })
        self.road_chains = rows
        return rows

    @classmethod
    def build(
        cls, nodes_rc: np.ndarray, edges: np.ndarray, *, cell_size: float = 64.0,
    ) -> "GraphSpatialContext":
        nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
        graph_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(len(nodes))]
        node_edges: list[list[int]] = [[] for _ in range(len(nodes))]
        edge_lengths = np.zeros(len(graph_edges), dtype=np.float32)
        for edge_id, (src_idx, dst_idx) in enumerate(graph_edges.tolist()):
            src_idx, dst_idx = int(src_idx), int(dst_idx)
            length = float(np.linalg.norm(nodes[dst_idx] - nodes[src_idx]))
            edge_lengths[edge_id] = length
            adjacency[src_idx].append((dst_idx, edge_id, length))
            adjacency[dst_idx].append((src_idx, edge_id, length))
            node_edges[src_idx].append(edge_id)
            node_edges[dst_idx].append(edge_id)
        for rows in adjacency:
            rows.sort(key=lambda row: (row[0], row[1]))
        for edge_ids in node_edges:
            edge_ids.sort()
        degrees = np.asarray([len(value) for value in node_edges], dtype=np.int32)

        component_ids = np.full(len(nodes), -1, dtype=np.int32)
        component_id = 0
        for start_idx in range(len(nodes)):
            if component_ids[start_idx] >= 0:
                continue
            component_ids[start_idx] = component_id
            pending = [start_idx]
            while pending:
                node_idx = pending.pop()
                for neighbor_idx, _edge_id, _length in adjacency[node_idx]:
                    if component_ids[neighbor_idx] < 0:
                        component_ids[neighbor_idx] = component_id
                        pending.append(neighbor_idx)
            component_id += 1

        discovery = np.full(len(nodes), -1, dtype=np.int64)
        low = np.full(len(nodes), -1, dtype=np.int64)
        parent_node = np.full(len(nodes), -1, dtype=np.int64)
        parent_edge = np.full(len(nodes), -1, dtype=np.int64)
        bridge_edge_ids: set[int] = set()
        clock = 0
        for root in range(len(nodes)):
            if discovery[root] >= 0:
                continue
            discovery[root] = low[root] = clock
            clock += 1
            stack: list[tuple[int, int]] = [(root, 0)]
            while stack:
                node_idx, neighbor_position = stack[-1]
                if neighbor_position < len(adjacency[node_idx]):
                    neighbor_idx, edge_id, _length = adjacency[node_idx][neighbor_position]
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
                    bridge_edge_ids.add(int(parent_edge[node_idx]))

        endpoints = tuple(int(value) for value in np.where(degrees == 1)[0].tolist())
        outward_vectors: dict[int, np.ndarray] = {}
        for node_idx in endpoints:
            neighbor_idx = adjacency[node_idx][0][0]
            vector = nodes[node_idx] - nodes[neighbor_idx]
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                outward_vectors[node_idx] = vector / norm
        outward_endpoint_ids = tuple(sorted(outward_vectors))
        return cls(
            nodes_rc=nodes,
            edges=graph_edges,
            adjacency=tuple(tuple(value) for value in adjacency),
            node_edges=tuple(tuple(value) for value in node_edges),
            degrees=degrees,
            component_ids=component_ids,
            bridge_edge_ids=frozenset(bridge_edge_ids),
            endpoints=endpoints,
            outward_vectors=outward_vectors,
            edge_lengths=edge_lengths,
            endpoint_index=PointGridIndex.build(
                nodes[list(outward_endpoint_ids)]
                if outward_endpoint_ids else np.empty((0, 2), dtype=np.float32),
                cell_size,
            ),
            endpoint_node_ids=np.asarray(outward_endpoint_ids, dtype=np.int32),
            node_index=PointGridIndex.build(nodes, cell_size),
            edge_index=EdgeGridIndex.build(nodes, graph_edges, cell_size),
        )
