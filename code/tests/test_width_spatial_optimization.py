from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
import finalize_review_results  # noqa: E402
import production_workflow  # noqa: E402
import width_surface_reconstruction  # noqa: E402
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
    def test_finalize_memory_failure_detection_is_specific(self) -> None:
        self.assertTrue(finalize_review_results._is_memory_allocation_failure({
            "error": "Unable to allocate 16.0 MiB for an array",
        }))
        self.assertTrue(finalize_review_results._is_memory_allocation_failure({
            "error": "", "error_type": "MemoryError",
        }))
        self.assertFalse(finalize_review_results._is_memory_allocation_failure({
            "error": "invalid optimized graph",
            "error_type": "ValueError",
        }))

    def test_finalize_memory_failures_are_retried_serially_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            failures = [
                {"summary": str(root / "v0001_summary.json"), "error": "Unable to allocate 16 MiB"},
                {"summary": str(root / "v0002_summary.json"), "error": "out of memory"},
            ]
            calls: list[str] = []

            def fake_finalize(_args, _output, _final, summary_path, _decisions, _decisions_path):
                calls.append(summary_path.name)
                if summary_path.stem.startswith("v0002"):
                    raise MemoryError("serial allocation still failed")
                return {"tile": summary_path.stem, "profiling": {"total_seconds": 1.0}}

            with mock.patch.object(finalize_review_results, "_finalize_one_atomic", side_effect=fake_finalize):
                recovered, remaining = finalize_review_results._retry_memory_failures_serially(
                    argparse.Namespace(), root, root / "final", {}, root / "decisions.csv", failures,
                )

            self.assertEqual(calls, ["v0001_summary.json", "v0002_summary.json"])
            self.assertEqual(len(recovered), 1)
            self.assertEqual(len(remaining), 1)
            self.assertTrue(remaining[0]["serial_memory_retry"])
            self.assertEqual(remaining[0]["parallel_error"], "out of memory")

    def test_regular_surface_does_not_allocate_one_full_canvas_per_edge(self) -> None:
        shape = (160, 160)
        nodes = np.asarray(
            [[20.0 + index % 100, 10.0 + (index * 7) % 130] for index in range(121)],
            dtype=np.float32,
        )
        edges = np.asarray([(index, index + 1) for index in range(120)], dtype=np.int32)
        widths = [
            {"edge_id": index, "width_px": 8.0, "quality_grade": "A", "source": "remeasured"}
            for index in range(len(edges))
        ]
        original_zeros = np.zeros
        full_canvas_allocations = 0

        def tracked_zeros(requested_shape, *args, **kwargs):
            nonlocal full_canvas_allocations
            if isinstance(requested_shape, tuple) and requested_shape == shape:
                full_canvas_allocations += 1
            return original_zeros(requested_shape, *args, **kwargs)

        with mock.patch.object(width_surface_reconstruction.np, "zeros", side_effect=tracked_zeros):
            result = width_surface_reconstruction.reconstruct_surface_from_widths(
                shape, nodes, edges, widths, [],
                config=width_surface_reconstruction.WidthSurfaceConfig(
                    regular_surface=True, close_kernel=1, boundary_smooth_sigma_px=0.0,
                ),
            )
        self.assertLess(full_canvas_allocations, 20)
        self.assertEqual(0, result.metadata["uncovered_centerline_px"])
        self.assertIn("edge_surface_drawing_seconds", result.metadata)

    def test_regular_shared_canvas_matches_legacy_per_edge_union(self) -> None:
        shape = (120, 180)
        nodes = np.asarray(
            [[25.0, 15.0], [25.0, 70.0], [90.0, 105.0], [90.0, 165.0]],
            dtype=np.float32,
        )
        edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        widths = [
            {"edge_id": 0, "width_px": 8.0, "quality_grade": "A", "source": "remeasured"},
            {"edge_id": 1, "width_px": 14.0, "quality_grade": "A", "source": "remeasured"},
        ]
        result = width_surface_reconstruction.reconstruct_surface_from_widths(
            shape, nodes, edges, widths, [],
            config=width_surface_reconstruction.WidthSurfaceConfig(
                regular_surface=True, close_kernel=1, boundary_smooth_sigma_px=0.0,
            ),
        )
        legacy_union = np.zeros(shape, dtype=np.uint8)
        for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
            edge_canvas = np.zeros(shape, dtype=np.uint8)
            width_surface_reconstruction._draw_buffer_edge(
                edge_canvas, nodes[src_idx], nodes[dst_idx],
                result.metadata["resolved_widths_px"][edge_id], 1.0,
            )
            legacy_union |= edge_canvas
        np.testing.assert_array_equal(legacy_union, result.surface)

    def test_finalize_identity_adopts_reuses_and_invalidates_one_slice(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output, final = root / "review", root / "final"
            output.mkdir(); final.mkdir()
            source = output / "v0001_summary.json"
            source.write_text('{"graph": "v0001.p"}', encoding="utf-8")
            (output / "v0001.p").write_bytes(b"graph-a")
            optimized = {
                "outputs": {
                    "optimized_graph": "v0001_optimized_graph.p",
                    "optimized_edges": "v0001_optimized_edges.csv",
                    "optimized_road_surface": "v0001_optimized_road_surface.png",
                    "optimized_width_samples": "v0001_optimized_width_samples.csv",
                    "optimized_width_segments": "v0001_optimized_width_segments.csv",
                },
            }
            for name in optimized["outputs"].values():
                (final / name).write_bytes(b"complete")
            (final / "v0001_optimized_summary.json").write_text(json.dumps(optimized), encoding="utf-8")
            args = argparse.Namespace(edited_dir="", workers=0, only_stem=[], surface_width_scale=1.0)
            decisions = output / "review_decisions.csv"

            action, _, _ = finalize_review_results._inspect_finalized_slice(
                args, output, final, source, decisions,
            )
            self.assertEqual("adopt", action)
            action, _, _ = finalize_review_results._inspect_finalized_slice(
                args, output, final, source, decisions,
            )
            self.assertEqual("reuse", action)
            incomplete = final / ".v0001_finalize_incomplete"
            incomplete.write_text("interrupted", encoding="utf-8")
            action, _, _ = finalize_review_results._inspect_finalized_slice(
                args, output, final, source, decisions,
            )
            self.assertEqual("rebuild", action)
            incomplete.unlink()
            (output / "v0001.p").write_bytes(b"graph-b")
            action, _, _ = finalize_review_results._inspect_finalized_slice(
                args, output, final, source, decisions,
            )
            self.assertEqual("rebuild", action)

    def test_finalize_identity_ignores_other_slice_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            summary = output / "v0001_summary.json"
            summary.write_text("{}", encoding="utf-8")
            decisions = output / "review_decisions.csv"
            decisions.write_text(
                "stem,item_type,item_id,decision\n"
                "v0001,candidate_centerline,one,accept\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(edited_dir="", workers=0, only_stem=[])
            original = finalize_review_results._build_finalize_identity(
                args, output, summary, decisions,
            )["digest"]
            decisions.write_text(
                "stem,item_type,item_id,decision\n"
                "v0001,candidate_centerline,one,accept\n"
                "v0002,candidate_centerline,two,reject\n",
                encoding="utf-8",
            )
            unrelated = finalize_review_results._build_finalize_identity(
                args, output, summary, decisions,
            )["digest"]
            self.assertEqual(original, unrelated)
            decisions.write_text(
                "stem,item_type,item_id,decision\n"
                "v0001,candidate_centerline,one,reject\n",
                encoding="utf-8",
            )
            changed = finalize_review_results._build_finalize_identity(
                args, output, summary, decisions,
            )["digest"]
            self.assertNotEqual(original, changed)

    def test_surface_assembly_and_export_checkpoint_avoid_unconditional_union(self) -> None:
        from rasterio import Affine
        from shapely.geometry import box

        with mock.patch.object(production_workflow, "unary_union", side_effect=AssertionError("unexpected union")):
            geometry = production_workflow._assemble_surface_polygons(
                [box(0, 0, 10, 10), box(20, 0, 30, 10)], 1.0, 1.0,
            )
        self.assertEqual("MultiPolygon", geometry.geom_type)

        with tempfile.TemporaryDirectory() as raw:
            final = Path(raw)
            mask_source = final / "v0001_surface.png"
            mask_source.write_bytes(b"surface-a")
            summary = final / "v0001_optimized_summary.json"
            summary.write_text(json.dumps({
                "outputs": {"optimized_road_surface": mask_source.name},
            }), encoding="utf-8")
            identity = production_workflow._build_export_surface_identity(final, None)
            cache = final / "_export_cache"
            mask = np.zeros((12, 14), dtype=np.uint8); mask[2:8, 3:10] = 1
            transform = Affine(1, 0, 0, 0, -1, 12)
            production_workflow._save_export_mask_cache(cache, identity, mask, transform)
            production_workflow._save_export_geometry_cache(cache, geometry)
            production_workflow._save_export_gap_cache(
                cache,
                [{"global_id": np.int64(7), "width_map": 9.0, "geometry": geometry.boundary.geoms[0]}],
                1,
            )
            loaded_mask, loaded_transform, loaded_geometry = production_workflow._load_export_surface_cache(
                cache, identity,
            )
            np.testing.assert_array_equal(mask, loaded_mask)
            self.assertEqual(transform, loaded_transform)
            self.assertTrue(loaded_geometry.equals(geometry))
            loaded_lines, loaded_gap_count = production_workflow._load_export_gap_cache(cache)
            self.assertEqual(1, loaded_gap_count)
            self.assertEqual(7, loaded_lines[0]["global_id"])
            self.assertTrue(loaded_lines[0]["geometry"].equals(geometry.boundary.geoms[0]))
            self.assertEqual(
                production_workflow.EXPORT_SURFACE_PARAMETERS,
                identity["parameters"],
            )
            (cache / "final_surface_geometry.wkb").write_bytes(b"damaged")
            loaded_mask, _, loaded_geometry = production_workflow._load_export_surface_cache(
                cache, identity,
            )
            np.testing.assert_array_equal(mask, loaded_mask)
            self.assertIsNone(loaded_geometry)
            mask_source.write_bytes(b"surface-b")
            changed = production_workflow._build_export_surface_identity(final, None)
            self.assertIsNone(production_workflow._load_export_surface_cache(cache, changed)[0])

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
