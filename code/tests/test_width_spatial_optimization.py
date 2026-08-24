from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


WIDTH_ROOT = Path(__file__).resolve().parents[1] / "engine" / "width"
if str(WIDTH_ROOT) not in sys.path:
    sys.path.insert(0, str(WIDTH_ROOT))

from graph_spatial_context import (  # noqa: E402
    GraphSpatialContext,
    PointGridIndex,
    find_vertical_divided_anchor_pair,
    points_within_radius_of_references,
)
from parallel_utils import resolve_worker_count, spawn_map  # noqa: E402


def square_in_spawn_worker(value: int) -> int:
    return value * value


def naive_vertical_pair(
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
) -> tuple[int, int] | None:
    candidates = []
    for first_idx in range(nodes_rc.shape[0]):
        first = nodes_rc[first_idx]
        first_distance = side * float(first[0] - center[0])
        if not min_distance_px <= first_distance <= max_distance_px:
            continue
        if abs(float(first[1] - center[1])) > lateral_search_px:
            continue
        for second_idx in range(first_idx + 1, nodes_rc.shape[0]):
            second = nodes_rc[second_idx]
            second_distance = side * float(second[0] - center[0])
            spacing = abs(float(second[1] - first[1]))
            if not min_distance_px <= second_distance <= max_distance_px:
                continue
            if abs(float(second[1] - center[1])) > lateral_search_px:
                continue
            if abs(float(second[0] - first[0])) > max_row_difference_px:
                continue
            if not min_spacing_px <= spacing <= max_spacing_px:
                continue
            first_rc = np.rint(first).astype(np.int32)
            second_rc = np.rint(second).astype(np.int32)
            probability = float(
                center_probability[first_rc[0], first_rc[1]]
                + center_probability[second_rc[0], second_rc[1]]
            )
            outward_distance = 0.5 * (first_distance + second_distance)
            lateral_center = abs(float(0.5 * (first[1] + second[1]) - center[1]))
            score = outward_distance + 0.7 * lateral_center - 20.0 * probability
            ordered = (
                (first_idx, second_idx)
                if first[1] <= second[1]
                else (second_idx, first_idx)
            )
            candidates.append((score, ordered))
    return min(candidates)[1] if candidates else None


class WidthSpatialOptimizationTests(unittest.TestCase):
    def test_worker_count_is_conservative_and_explicitly_controllable(self) -> None:
        self.assertEqual(resolve_worker_count(0, 6, automatic_limit=2), 2)
        self.assertEqual(resolve_worker_count(1, 6), 1)
        self.assertEqual(resolve_worker_count(4, 2), 2)
        self.assertEqual(resolve_worker_count(0, 1), 1)
        with self.assertRaisesRegex(ValueError, "workers"):
            resolve_worker_count(-1, 6)

    def test_spawn_map_runs_top_level_workers_in_stable_order(self) -> None:
        self.assertEqual(spawn_map(square_in_spawn_worker, [3, 1, 2], 2), [9, 1, 4])

    def test_vertical_anchor_index_matches_naive_implementation(self) -> None:
        for seed in range(12):
            rng = np.random.default_rng(seed)
            nodes = rng.uniform(1.0, 510.0, size=(350, 2)).astype(np.float32)
            probability = rng.random((512, 512), dtype=np.float32)
            center = rng.uniform(240.0, 272.0, size=2).astype(np.float32)
            for side in (-1, 1):
                expected = naive_vertical_pair(nodes, probability, center, side)
                actual = find_vertical_divided_anchor_pair(
                    nodes, probability, center, side,
                )
                self.assertEqual(actual, expected, (seed, side))

    def test_vertical_anchor_tie_breaking_matches_naive(self) -> None:
        nodes = np.asarray([
            [170.0, 240.0], [170.0, 250.0],
            [170.0, 260.0], [170.0, 270.0],
            [400.0, 400.0],
        ], dtype=np.float32)
        probability = np.zeros((512, 512), dtype=np.float32)
        center = np.asarray([256.0, 255.0], dtype=np.float32)
        self.assertEqual(
            find_vertical_divided_anchor_pair(nodes, probability, center, -1),
            naive_vertical_pair(nodes, probability, center, -1),
        )

    def test_edge_index_is_a_superset_of_exact_distance_candidates(self) -> None:
        rng = np.random.default_rng(41)
        nodes = rng.uniform(0.0, 600.0, size=(220, 2)).astype(np.float32)
        edges = np.asarray(
            [(index, index + 1) for index in range(len(nodes) - 1)],
            dtype=np.int32,
        )
        context = GraphSpatialContext.build(nodes, edges, cell_size=48.0)
        for point in rng.uniform(0.0, 600.0, size=(30, 2)).astype(np.float32):
            radius = 35.0
            indexed = set(context.edge_index.query_point_radius(point, radius))
            exact = set()
            for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
                vector = nodes[dst_idx] - nodes[src_idx]
                length2 = float(np.dot(vector, vector))
                ratio = 0.0 if length2 <= 0 else float(np.clip(
                    np.dot(point - nodes[src_idx], vector) / length2, 0.0, 1.0,
                ))
                projection = nodes[src_idx] + ratio * vector
                distance = float(np.linalg.norm(point - projection))
                if distance <= radius:
                    exact.add(edge_id)
            self.assertTrue(exact <= indexed)

    def test_point_grid_nearest_matches_stable_brute_force(self) -> None:
        rng = np.random.default_rng(73)
        points = rng.uniform(-300.0, 700.0, size=(500, 2)).astype(np.float32)
        points[17] = points[9]
        index = PointGridIndex.build(points, 37.0)
        for query in rng.uniform(-400.0, 800.0, size=(80, 2)).astype(np.float32):
            distances = np.linalg.norm(points - query, axis=1)
            expected_id = int(np.argmin(distances))
            distance, point_id = index.nearest(query)
            self.assertEqual(point_id, expected_id)
            self.assertAlmostEqual(distance, float(distances[expected_id]), places=6)

    def test_radius_membership_matches_all_pairs_distance_matrix(self) -> None:
        rng = np.random.default_rng(99)
        points = rng.uniform(-50.0, 450.0, size=(400, 2)).astype(np.float32)
        references = rng.uniform(0.0, 400.0, size=(45, 2)).astype(np.float32)
        for radius in (0.0, 7.5, 35.0, 120.0):
            expected = np.min(
                np.linalg.norm(points[:, None, :] - references[None, :, :], axis=2),
                axis=1,
            ) <= radius
            np.testing.assert_array_equal(
                points_within_radius_of_references(points, references, radius),
                expected,
            )

    def test_graph_context_matches_existing_graph_helpers_and_caches_chains(self) -> None:
        nodes = np.asarray([
            [0.0, 0.0], [0.0, 10.0], [0.0, 20.0],
            [10.0, 10.0], [50.0, 50.0], [60.0, 50.0],
        ], dtype=np.float32)
        edges = np.asarray([(0, 1), (1, 2), (1, 3), (4, 5)], dtype=np.int32)
        context = GraphSpatialContext.build(nodes, edges)
        degrees = np.asarray([1, 3, 1, 1, 1, 1], dtype=np.int32)
        node_edges = ((0,), (0, 1, 2), (1,), (2,), (3,), (3,))

        np.testing.assert_array_equal(context.degrees, degrees)
        self.assertEqual(context.node_edges, node_edges)
        np.testing.assert_array_equal(context.component_ids, [0, 0, 0, 0, 1, 1])
        self.assertEqual(context.bridge_edge_ids, frozenset({0, 1, 2, 3}))
        first = context.build_road_chain_rows()
        second = context.build_road_chain_rows()
        self.assertIs(first, second)
        self.assertEqual(sum(row["micro_edge_count"] for row in first), len(edges))


if __name__ == "__main__":
    unittest.main()
