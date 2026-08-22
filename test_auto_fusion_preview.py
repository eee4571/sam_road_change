from __future__ import annotations

import sys
import tempfile
import unittest
import csv
import json
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import cv2
import networkx as nx
import numpy as np
import rasterio
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, unary_union


ENGINE = Path(__file__).resolve().parent / "engine" / "width"
sys.path.insert(0, str(ENGINE))

import molra_centerline_width as fusion
import road_change_detection as change
import finalize_review_results as finalize
import production_workflow as workflow


def surface_candidate(start: tuple[float, float], end: tuple[float, float]) -> dict:
    return {
        "candidate_type": "surface_skeleton",
        "review_status": "surface_only_candidate",
        "confidence": "medium",
        "auto_decision": "review",
        "action": "propose_add_centerline",
        "start_row": start[0],
        "start_col": start[1],
        "end_row": end[0],
        "end_col": end[1],
        "topology_score": 0.0,
        "note": "near_graph_endpoint_and_linear",
        "length_px": float(np.linalg.norm(np.asarray(end) - np.asarray(start))),
        "median_half_width_px": 3.0,
        "surface_support_ratio": 1.0,
    }


class AutoFusionTests(unittest.TestCase):
    def test_unicode_image_reader_handles_user_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "中文概率图.png"
            expected = np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
            path.write_bytes(cv2.imencode(".png", expected)[1].tobytes())
            actual = workflow._read_image(path)
            self.assertTrue(np.array_equal(expected, actual))

    def _annotate(self, rows: list[dict], nodes: np.ndarray, edges: np.ndarray, **kwargs) -> None:
        fusion.annotate_candidate_graph_matches(rows, nodes, edges, 2.0, **kwargs)

    def test_one_endpoint_extension_auto_accepts_when_outward_aligned(self) -> None:
        rows = [surface_candidate((0.0, 0.0), (-100.0, 0.0))]
        self._annotate(rows, np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32), np.asarray([[0, 1]], dtype=np.int32))
        self.assertEqual("accept", rows[0]["auto_decision"])
        self.assertEqual("auto_add_surface_skeleton_as_is", rows[0]["action"])

    def test_exact_node_contacts_accept_without_legacy_direction_or_degree_filter(self) -> None:
        endpoint_contact = surface_candidate((0.0, 0.0), (0.0, 111.0))
        non_endpoint_contact = surface_candidate((10.0, 0.0), (10.0, 60.0))
        nodes = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
        edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
        self._annotate([endpoint_contact, non_endpoint_contact], nodes, edges)
        for row in (endpoint_contact, non_endpoint_contact):
            self.assertEqual("endpoint_to_node", row["connection_type"])
            self.assertEqual("all_surface_skeletons_as_is", row["auto_rule"])
            self.assertEqual("accept", row["auto_decision"])

    def test_connected_low_confidence_short_and_isolated_candidates_are_all_accepted(self) -> None:
        low_confidence = surface_candidate((0.0, 0.0), (-100.0, 0.0))
        low_confidence["confidence"] = "low"
        short = surface_candidate((10.0, 0.0), (35.0, 0.0))
        isolated = surface_candidate((300.0, 0.0), (450.0, 0.0))
        nodes = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        self._annotate([low_confidence, short, isolated], nodes, edges)
        self.assertEqual("accept", low_confidence["auto_decision"])
        self.assertEqual("all_surface_skeletons_as_is", low_confidence["auto_rule"])
        self.assertEqual("accept", short["auto_decision"])
        self.assertIn("length_too_short", short["hard_veto_reasons"])
        self.assertEqual("", short["effective_veto_reasons"])
        self.assertEqual("isolated_simple_road", isolated["connection_type"])
        self.assertEqual("all_surface_skeletons_as_is", isolated["auto_rule"])
        self.assertEqual("accept", isolated["auto_decision"])

    def test_surface_skeleton_gap_is_connected_by_mask_path_when_both_tangents_agree(self) -> None:
        binary = np.zeros((80, 40), dtype=np.uint8)
        binary[5:56, 8:13] = 1
        road_probability = binary.astype(np.float32)
        center_probability = np.zeros_like(road_probability)
        nodes = np.asarray([[10.0, 10.0], [20.0, 10.0]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        candidate = surface_candidate((30.0, 10.0), (50.0, 10.0))
        candidate["candidate_id"] = 7
        audit = fusion.connect_surface_skeletons_by_mask_path(
            [candidate], binary, road_probability, center_probability, nodes, edges,
            max_distance_px=32.0, min_alignment_cosine=0.9,
            min_surface_support=0.95, max_path_ratio=1.2,
        )
        points = np.asarray(json.loads(candidate["polyline_points_json"]), dtype=np.float32)
        self.assertEqual(1, len(audit))
        self.assertEqual(1, candidate["surface_connector_count"])
        self.assertTrue(np.allclose(points[0], nodes[1]))
        self.assertTrue(np.allclose(points[-1], (50.0, 10.0)))
        self.assertTrue(np.all(np.abs(points[:, 1] - 10.0) <= 0.5))
        self.assertEqual("accepted_surface_mask_path", candidate["surface_attachment_audit"])
        self.assertEqual("surface_skeleton_to_graph_endpoint", candidate["surface_attachment_kind"])
        self.assertEqual(0, candidate["surface_attachment_endpoint_position"])
        self.assertEqual(20.0, candidate["surface_attachment_node_row"])
        self.assertEqual(10.0, candidate["surface_attachment_node_col"])

    def test_surface_skeleton_gap_rejects_wrong_direction_or_missing_surface(self) -> None:
        nodes = np.asarray([[10.0, 10.0], [20.0, 10.0]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        wrong_direction = surface_candidate((30.0, 20.0), (30.0, 40.0))
        missing_surface = surface_candidate((30.0, 10.0), (50.0, 10.0))
        binary = np.zeros((80, 60), dtype=np.uint8)
        binary[8:23, 8:13] = 1
        road_probability = binary.astype(np.float32)
        audit = fusion.connect_surface_skeletons_by_mask_path(
            [wrong_direction, missing_surface], binary, road_probability,
            np.zeros_like(road_probability), nodes, edges,
            max_distance_px=40.0, min_alignment_cosine=0.85,
            min_surface_support=0.95, max_path_ratio=1.3,
        )
        self.assertEqual([], audit)
        self.assertNotIn("surface_connector_count", wrong_direction)
        self.assertNotIn("surface_connector_count", missing_surface)

    def test_facing_surface_skeleton_fragments_are_connected_only_through_road_mask(self) -> None:
        binary = np.zeros((90, 50), dtype=np.uint8)
        binary[5:76, 8:13] = 1
        road_probability = binary.astype(np.float32)
        first = surface_candidate((10.0, 10.0), (30.0, 10.0))
        second = surface_candidate((50.0, 10.0), (70.0, 10.0))
        first.update({"candidate_id": 3, "region_id": "left"})
        second.update({"candidate_id": 8, "region_id": "right"})

        connectors, audit = fusion.build_surface_skeleton_pair_connectors(
            [first, second], binary, road_probability, np.zeros_like(road_probability),
            max_distance_px=40.0, min_alignment_cosine=0.9,
            min_surface_support=0.95, max_path_ratio=1.2,
        )

        self.assertEqual(1, len(connectors))
        self.assertEqual(1, len(audit))
        self.assertEqual("surface_skeleton_connector", connectors[0]["candidate_type"])
        self.assertEqual("accept", connectors[0]["auto_decision"])
        self.assertEqual("surface_skeleton_to_surface_skeleton", audit[0]["connector_kind"])
        self.assertEqual({"3", "8"}, {str(audit[0]["candidate_id"]), str(audit[0]["target_candidate_id"])})
        points = np.asarray(json.loads(connectors[0]["polyline_points_json"]), dtype=np.float32)
        self.assertTrue(np.allclose(points[0], (30.0, 10.0)))
        self.assertTrue(np.allclose(points[-1], (50.0, 10.0)))
        self.assertTrue(np.all(binary[np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int)] > 0))

    def test_surface_skeleton_pair_connector_rejects_wrong_direction_same_region_and_missing_surface(self) -> None:
        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[5:86, 8:13] = 1
        probability = binary.astype(np.float32)
        first = surface_candidate((10.0, 10.0), (30.0, 10.0))
        wrong_direction = surface_candidate((50.0, 10.0), (50.0, 35.0))
        same_region = surface_candidate((50.0, 10.0), (70.0, 10.0))
        first["region_id"] = same_region["region_id"] = "one_surface_region"
        missing_surface = surface_candidate((50.0, 40.0), (70.0, 40.0))
        missing_surface["region_id"] = "other"
        connectors, audit = fusion.build_surface_skeleton_pair_connectors(
            [first, wrong_direction, same_region, missing_surface],
            binary, probability, np.zeros_like(probability),
            max_distance_px=50.0, min_alignment_cosine=0.85,
            min_surface_support=0.95, max_path_ratio=1.3,
        )
        self.assertEqual([], connectors)
        self.assertEqual([], audit)

    def test_strict_graph_endpoint_gap_connects_mutual_collinear_road_endpoints(self) -> None:
        binary = np.zeros((90, 50), dtype=np.uint8)
        binary[5:76, 8:13] = 1
        nodes = np.asarray([[10.0, 10.0], [30.0, 10.0], [50.0, 10.0], [70.0, 10.0]], dtype=np.float32)
        edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        candidates = fusion.build_endpoint_gap_candidates(
            binary, nodes, edges,
            max_gap_px=30.0, min_alignment_cosine=0.95,
            min_surface_support=0.95, path_margin_px=16,
            outside_cost=30.0, sample_step_px=6.0,
            road_probability=binary.astype(np.float32),
            center_probability=np.zeros_like(binary, dtype=np.float32),
            max_path_ratio=1.2, ambiguity_ratio=1.2,
        )
        self.assertEqual(1, len(candidates))
        self.assertEqual({1, 2}, {candidates[0]["start_node_idx"], candidates[0]["end_node_idx"]})
        self.assertGreaterEqual(candidates[0]["surface_support_ratio"], 0.95)
        self.assertLessEqual(candidates[0]["path_ratio"], 1.2)

    def test_strict_graph_endpoint_gap_rejects_ambiguous_parallel_targets(self) -> None:
        binary = np.ones((100, 100), dtype=np.uint8)
        nodes = np.asarray([
            [10.0, 10.0], [30.0, 10.0],
            [50.0, 10.0], [70.0, 10.0],
            [52.0, 12.0], [72.0, 12.0],
        ], dtype=np.float32)
        edges = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
        candidates = fusion.build_endpoint_gap_candidates(
            binary, nodes, edges,
            max_gap_px=30.0, min_alignment_cosine=0.95,
            min_surface_support=0.95, path_margin_px=16,
            outside_cost=30.0, sample_step_px=6.0,
            max_path_ratio=1.2, ambiguity_ratio=1.2,
        )
        self.assertEqual([], candidates)

    def test_protected_divided_endpoint_still_keeps_surface_skeleton_as_is(self) -> None:
        rows = [surface_candidate((0.0, 0.0), (-100.0, 0.0))]
        nodes = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 8.0], [10.0, 8.0]], dtype=np.float32)
        edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        self._annotate(rows, nodes, edges)
        self.assertEqual("accept", rows[0]["auto_decision"])
        self.assertEqual("all_surface_skeletons_as_is", rows[0]["auto_rule"])

    def test_blocky_and_long_surface_skeletons_are_both_retained_as_is(self) -> None:
        blocky = surface_candidate((0.0, 0.0), (-100.0, 0.0))
        blocky["note"] = "blocky_or_non_linear_surface"
        long_candidate = surface_candidate((0.0, 0.0), (-180.0, 0.0))
        nodes = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        self._annotate([blocky, long_candidate], nodes, edges)
        self.assertEqual("accept", blocky["auto_decision"])
        self.assertEqual("accept", long_candidate["auto_decision"])

    def test_small_acyclic_multi_skeleton_region_is_accepted(self) -> None:
        main_branch = surface_candidate((0.0, 0.0), (-100.0, 0.0))
        competing_branch = surface_candidate((0.0, 0.0), (-90.0, 0.0))
        main_branch["region_id"] = "parking_like_surface"
        competing_branch["region_id"] = "parking_like_surface"
        nodes = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        self._annotate([main_branch, competing_branch], nodes, edges)
        self.assertEqual("accept", main_branch["auto_decision"])
        self.assertEqual("accept", competing_branch["auto_decision"])
        self.assertEqual(2, main_branch["surface_region_candidate_count"])

    def test_real_2022_artifact_promotes_the_safe_surface_extension(self) -> None:
        roads = (
            Path(__file__).resolve().parent / "outputs" / "run_20260810_161904" / "grids"
            / "change_grid_01" / "periods" / "2022" / "runs" / "roads"
        )
        review = roads / "width_review"
        if not review.is_dir():
            self.skipTest("real 2022 runtime artifact is not available")
        with (review / "2022_candidate_centerlines.csv").open(encoding="utf-8-sig", newline="") as file:
            row = next(item for item in csv.DictReader(file) if item["candidate_id"] == "0")
        self.assertEqual("surface_only_candidate", row["review_status"])
        self.assertEqual("medium", row["confidence"])
        nodes, edges = fusion.load_graph(review / "2022_prepared_graph.p")
        self._annotate([row], nodes, edges)
        self.assertEqual("accept", row["auto_decision"])
        self.assertEqual("auto_add_surface_skeleton_as_is", row["action"])

    def test_long_straight_isolated_surface_skeleton_is_retained_as_is(self) -> None:
        row = surface_candidate((100.0, 100.0), (100.0, 250.0))
        row["candidate_id"] = "isolated"
        row["polyline_points_json"] = "[[100,100],[100,175],[100,250]]"
        center_probability = np.ones((300, 300), dtype=np.float32) * 0.5
        self._annotate([row], np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32), np.asarray([[0, 1]], dtype=np.int32), center_probability=center_probability)
        self.assertEqual("accept", row["auto_decision"])
        self.assertGreaterEqual(float(row["auto_score"]), 70.0)
        self.assertGreaterEqual(float(row["candidate_center_probability"]), 0.35)
        self.assertEqual("isolated_simple_road", row["connection_type"])
        self.assertEqual("all_surface_skeletons_as_is", row["auto_rule"])
        self.assertFalse(row["hard_veto"])

    def test_candidate_spanning_two_graph_components_is_accepted(self) -> None:
        bridge = surface_candidate((0.0, 0.0), (100.0, 100.0))
        bridge["polyline_points_json"] = "[[0,0],[50,50],[100,100]]"
        nodes = np.asarray(
            [[-20.0, 0.0], [0.0, 0.0], [100.0, 100.0], [100.0, 120.0]],
            dtype=np.float32,
        )
        edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        self._annotate([bridge], nodes, edges)
        self.assertEqual(2, bridge["matched_component_count"])
        self.assertEqual("connect_components", bridge["topology_impact"])
        self.assertEqual("component_bridge", bridge["connection_type"])
        self.assertEqual("all_surface_skeletons_as_is", bridge["auto_rule"])
        self.assertEqual("accept", bridge["auto_decision"])
        self.assertEqual("auto_add_surface_skeleton_as_is", bridge["action"])

    def test_scoring_rule_marks_medium_band_and_hard_vetoes(self) -> None:
        self.assertEqual(("medium_confidence_auto", "accept"), fusion.candidate_score_rule(60.0, []))
        veto_cases = {
            "low_surface": {"surface_support_ratio": 0.5},
            "width_abnormal": {"median_half_width_px": 20.0},
            "boundary_truncated": {"boundary_truncated": True},
            "short_loop": {"polyline_points_json": "[[20,20],[20,30],[30,30],[30,20],[20,20]]", "length_px": 40.0},
            "dense_branching": {"surface_region_candidate_count": 5},
        }
        rows = []
        for name, values in veto_cases.items():
            row = surface_candidate((20.0, 20.0), (20.0, 180.0))
            row["candidate_id"] = name
            row.update(values)
            rows.append(row)
        self._annotate(rows, np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32), np.asarray([[0, 1]], dtype=np.int32))
        for row in rows:
            self.assertFalse(row["hard_veto"])
            self.assertEqual("accept", row["auto_decision"])
            self.assertTrue(row["hard_veto_reasons"])
            self.assertEqual("", row["effective_veto_reasons"])

    def test_endpoint_to_edge_is_scored_for_t_connection(self) -> None:
        row = surface_candidate((30.0, 50.0), (140.0, 50.0))
        row["candidate_id"] = "t"
        row["polyline_points_json"] = "[[30,50],[85,50],[140,50]]"
        nodes = np.asarray([[0.0, 0.0], [0.0, 100.0]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        self._annotate([row], nodes, edges)
        self.assertEqual("endpoint_to_edge", row["connection_type"])
        self.assertEqual("accept", row["auto_decision"])
        self.assertGreaterEqual(float(row["auto_score"]), 70.0)

    def test_duplicate_parallel_and_short_surface_skeletons_are_retained_with_audit(self) -> None:
        parallel = surface_candidate((3.0, 0.0), (3.0, 160.0))
        parallel["polyline_points_json"] = "[[3,0],[3,80],[3,160]]"
        too_short = surface_candidate((300.0, 0.0), (300.0, 39.0))
        nodes = np.asarray([[0.0, -100.0], [0.0, 200.0]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        self._annotate([parallel, too_short], nodes, edges)
        self.assertIn("duplicate_or_near_parallel", parallel["hard_veto_reasons"])
        self.assertIn("length_too_short", too_short["hard_veto_reasons"])
        self.assertFalse(parallel["hard_veto"])
        self.assertFalse(too_short["hard_veto"])
        self.assertEqual("accept", parallel["auto_decision"])
        self.assertEqual("accept", too_short["auto_decision"])

    def test_clean_long_isolated_surface_skeleton_is_retained(self) -> None:
        long_road = surface_candidate((300.0, 100.0), (300.0, 1000.0))
        long_road["polyline_points_json"] = "[[300,100],[300,550],[300,1000]]"
        center_probability = np.ones((1200, 1200), dtype=np.float32) * 0.2
        self._annotate(
            [long_road],
            np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
            np.asarray([[0, 1]], dtype=np.int32),
            center_probability=center_probability,
            surface_extension_max_length_px=120.0,
        )
        self.assertEqual("accept", long_road["auto_decision"])
        self.assertEqual("isolated_simple_road", long_road["connection_type"])
        self.assertEqual("all_surface_skeletons_as_is", long_road["auto_rule"])
        self.assertTrue(long_road["long_road_evidence"])
        self.assertNotIn("length_out_of_range", long_road["hard_veto_reasons"])

    def test_global_fusion_does_not_create_connector_between_near_endpoints(self) -> None:
        first = LineString([(0.0, 0.0), (10.0, 0.0)])
        nearby = LineString([(10.4, 0.0), (20.0, 0.0)])
        rows = [
            {"tile_stem": "a", "geometry": first, "width_map": 4.0, "quality_grade": "A"},
            {"tile_stem": "b", "geometry": nearby, "width_map": 6.0, "quality_grade": "A"},
        ]
        fused = workflow._fuse_centerline_records(rows, {}, match_tolerance=0.5)
        geometry = linemerge(unary_union([row["geometry"] for row in fused]))
        self.assertEqual("MultiLineString", geometry.geom_type)
        self.assertAlmostEqual(first.length + nearby.length, geometry.length)

    def test_global_surface_gap_repairs_only_unique_facing_cross_component_seam(self) -> None:
        left = LineString([(0.0, 0.0), (10.0, 0.0)])
        right = LineString([(20.0, 0.0), (30.0, 0.0)])
        rows = [
            {"tile_stem": "left", "geometry": left, "width_map": 4.0, "quality_grade": "A"},
            {"tile_stem": "right", "geometry": right, "width_map": 4.0, "quality_grade": "A"},
        ]
        repaired, count = workflow._connect_surface_supported_global_gaps(
            rows, LineString([(0.0, 0.0), (30.0, 0.0)]).buffer(1.0), 1.0,
        )

        self.assertEqual(1, count)
        self.assertEqual("global_gap_repair", repaired[-1]["fusion_sta"])
        self.assertEqual(4.0, repaired[-1]["width_map"])
        self.assertEqual("attached_road", repaired[-1]["width_src"])
        self.assertEqual("C", repaired[-1]["quality_gr"])
        self.assertNotIn("quality_grade", repaired[-1])
        self.assertEqual("LineString", linemerge(unary_union([row["geometry"] for row in repaired])).geom_type)
        self.assertEqual(
            "fully_automatic_probability_and_surface_guided_unique_facing_cross_component_gap_repairs",
            workflow.FINAL_CENTERLINE_GEOMETRY_POLICY,
        )

        merged_surface, buffered_count, added_area = workflow._merge_global_gap_surfaces(
            unary_union([left.buffer(2.0), right.buffer(2.0)]), repaired, 1.0,
        )
        self.assertEqual(1, buffered_count)
        self.assertGreater(added_area, 0.0)
        self.assertTrue(merged_surface.covers(Point(15.0, 1.5)))

    def test_global_gap_width_uses_both_attached_roads_not_network_median(self) -> None:
        left = {"geometry": LineString([(0.0, 0.0), (10.0, 0.0)]), "width_map": 4.0, "quality_grade": "A"}
        right = {"geometry": LineString([(20.0, 0.0), (30.0, 0.0)]), "width_map": 8.0, "quality_grade": "A"}
        repaired, count = workflow._connect_surface_supported_global_gaps(
            [left, right],
            LineString([(0.0, 0.0), (30.0, 0.0)]).buffer(2.0),
            1.0,
        )
        self.assertEqual(1, count)
        connector = next(row for row in repaired if row.get("fusion_sta") == "global_gap_repair")
        self.assertEqual(6.0, connector["width_map"])

    def test_global_gap_uses_low_absolute_centerline_probability_without_review_state(self) -> None:
        left = {"tile_stem": "left", "geometry": LineString([(5.0, 20.0), (20.0, 20.0)]), "width_map": 4.0, "quality_grade": "A"}
        right = {"tile_stem": "right", "geometry": LineString([(35.0, 20.0), (50.0, 20.0)]), "width_map": 4.0, "quality_grade": "A"}
        probability = np.zeros((64, 64), dtype=np.float32)
        probability[20, 5:51] = 0.02

        repaired, count = workflow._connect_surface_supported_global_gaps(
            [left, right],
            None,
            1.0,
            centerline_probability=probability,
            centerline_transform=rasterio.Affine.identity(),
        )

        self.assertEqual(1, count)
        connector = repaired[-1]
        self.assertEqual("centerline_probability_astar", connector["evidence"])
        self.assertEqual("accepted", connector["auto_state"])
        self.assertNotIn("review_status", connector)
        self.assertNotIn("requires_manual_review", connector)
        self.assertGreater(float(connector["center_n"]), 0.9)
        self.assertEqual("LineString", linemerge(unary_union([row["geometry"] for row in repaired])).geom_type)

    def test_global_gap_auto_skips_when_probability_has_no_continuous_ridge(self) -> None:
        left = {"tile_stem": "left", "geometry": LineString([(5.0, 20.0), (20.0, 20.0)]), "width_map": 4.0, "quality_grade": "A"}
        right = {"tile_stem": "right", "geometry": LineString([(35.0, 20.0), (50.0, 20.0)]), "width_map": 4.0, "quality_grade": "A"}
        probability = np.zeros((64, 64), dtype=np.float32)

        repaired, count = workflow._connect_surface_supported_global_gaps(
            [left, right],
            None,
            1.0,
            centerline_probability=probability,
            centerline_transform=rasterio.Affine.identity(),
        )

        self.assertEqual(0, count)
        self.assertEqual(2, len(repaired))

    def test_global_probability_connects_branch_endpoint_to_noded_target_edge(self) -> None:
        branch = {"tile_stem": "branch", "geometry": LineString([(2.0, 20.0), (20.000000004, 20.0)]), "width_map": 4.0, "quality_grade": "A"}
        main = {"tile_stem": "main", "geometry": LineString([(35.000000004, 5.0), (35.000000004, 35.0)]), "width_map": 6.0, "quality_grade": "A"}
        probability = np.zeros((48, 48), dtype=np.float32)
        probability[20, 2:36] = 0.02
        probability[5:36, 35] = 0.02

        repaired, count = workflow._connect_surface_supported_global_gaps(
            [branch, main],
            None,
            1.0,
            centerline_probability=probability,
            centerline_transform=rasterio.Affine.identity(),
        )

        self.assertEqual(1, count)
        connector = next(row for row in repaired if row.get("fusion_sta") == "global_gap_repair")
        self.assertEqual("edge", connector["target"])
        self.assertEqual("centerline_probability_astar_to_edge", connector["evidence"])
        self.assertEqual("accepted", connector["auto_state"])
        main_parts = [row["geometry"] for row in repaired if row.get("tile_stem") == "main"]
        self.assertEqual(2, len(main_parts))
        contact = Point(35.000000004, 20.0)
        self.assertTrue(all(part.boundary.distance(contact) < 1e-8 for part in main_parts))
        noded = unary_union([row["geometry"] for row in repaired])
        graph = nx.Graph()
        for part in workflow._line_parts(noded):
            coordinates = list(part.coords)
            graph.add_edge(tuple(coordinates[0]), tuple(coordinates[-1]))
        self.assertEqual(1, nx.number_connected_components(graph))

    def test_continuous_probability_fusion_preserves_low_values_across_tile_seam(self) -> None:
        first = np.zeros((16, 16), dtype=np.float32)
        second = np.zeros((16, 16), dtype=np.float32)
        first[8, :] = 0.02
        second[8, :] = 0.02
        first_transform = rasterio.Affine(1, 0, 0, 0, -1, 16)
        second_transform = rasterio.Affine(1, 0, 16, 0, -1, 16)
        sources = [
            {"mask": first, "shape": first.shape, "transform": first_transform},
            {"mask": second, "shape": second.shape, "transform": second_transform},
        ]

        fused, transform, probability = workflow._fuse_surface_masks(
            sources, "EPSG:3857", feather_pixels=4.0, continuous=True,
        )

        self.assertEqual((16, 32), fused.shape)
        self.assertTrue(np.allclose(fused, probability))
        self.assertAlmostEqual(0.02, float(fused[8, 15]), places=4)
        self.assertAlmostEqual(0.02, float(fused[8, 16]), places=4)
        self.assertEqual(first_transform, transform)

    def test_global_surface_gap_rejects_missing_surface_wrong_direction_same_component_and_ambiguity(self) -> None:
        left = {"tile_stem": "left", "geometry": LineString([(0.0, 0.0), (10.0, 0.0)]), "width_map": 4.0, "quality_grade": "A"}
        right = {"tile_stem": "right", "geometry": LineString([(20.0, 0.0), (30.0, 0.0)]), "width_map": 4.0, "quality_grade": "A"}
        continuous = LineString([(0.0, 0.0), (30.0, 0.0)]).buffer(3.0)
        _, missing_count = workflow._connect_surface_supported_global_gaps([left, right], None, 1.0)
        self.assertEqual(0, missing_count)

        wrong = {"tile_stem": "wrong", "geometry": LineString([(20.0, 0.0), (30.0, 8.0)]), "width_map": 4.0, "quality_grade": "A"}
        _, wrong_count = workflow._connect_surface_supported_global_gaps([left, wrong], continuous, 1.0)
        self.assertEqual(0, wrong_count)

        same_component = {"tile_stem": "same", "geometry": LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (20.0, 10.0), (20.0, 0.0)]), "width_map": 4.0, "quality_grade": "A"}
        _, same_count = workflow._connect_surface_supported_global_gaps([same_component], continuous, 1.0)
        self.assertEqual(0, same_count)

        alternate = {"tile_stem": "alternate", "geometry": LineString([(20.0, 2.0), (30.0, 2.0)]), "width_map": 4.0, "quality_grade": "A"}
        _, ambiguous_count = workflow._connect_surface_supported_global_gaps([left, right, alternate], continuous, 1.0)
        self.assertEqual(0, ambiguous_count)

    def test_long_surface_skeleton_with_bad_audit_evidence_is_still_retained(self) -> None:
        bad_long_road = surface_candidate((300.0, 100.0), (300.0, 1000.0))
        bad_long_road["note"] = "blocky_or_non_linear_surface"
        center_probability = np.ones((1200, 1200), dtype=np.float32) * 0.2
        self._annotate(
            [bad_long_road],
            np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
            np.asarray([[0, 1]], dtype=np.int32),
            center_probability=center_probability,
        )
        self.assertEqual("accept", bad_long_road["auto_decision"])
        self.assertIn("dense_branching_or_blocky", bad_long_road["hard_veto_reasons"])
        self.assertEqual("", bad_long_road["effective_veto_reasons"])

    def test_finalizer_nodes_target_edge_for_scored_endpoint_to_edge(self) -> None:
        candidate = surface_candidate((30.0, 50.0), (140.0, 50.0))
        candidate.update({
            "candidate_type": "endpoint_gap",
            "connection_type": "endpoint_to_edge",
            "connection_endpoint_position": 0,
            "connection_projection_row": 0.0,
            "connection_projection_col": 50.0,
        })
        nodes = [(0, 0), (0, 100)]
        edges = [(0, 1)]
        keys = {(0, 1)}
        records = [{"src_idx": 0, "dst_idx": 1, "src_row": 0, "src_col": 0, "dst_row": 0, "dst_col": 100, "final_status": "geometry_kept", "source": "samroad"}]
        points, forced = finalize.prepare_candidate_topology_attachment(candidate, finalize.candidate_points(candidate), nodes, edges, keys, records)
        self.assertEqual((0, 50), nodes[2])
        self.assertIn((0, 2), edges)
        self.assertIn((2, 1), edges)
        self.assertEqual(2, forced[0])
        self.assertEqual((0.0, 50.0), points[0])

    def test_finalizer_resolves_endpoint_node_by_audited_coordinate(self) -> None:
        candidate = surface_candidate((30.0, 50.0), (140.0, 50.0))
        candidate.update({
            "candidate_type": "endpoint_gap",
            "connection_type": "endpoint_to_node",
            "connection_endpoint_position": 0,
            "connection_node_idx": 0,  # deliberately stale after graph reload
            "connection_node_row": 0.0,
            "connection_node_col": 100.0,
        })
        nodes = [(0, 0), (0, 100)]
        points, forced = finalize.prepare_candidate_topology_attachment(candidate, finalize.candidate_points(candidate), nodes, [(0, 1)], {(0, 1)}, [])
        self.assertEqual(1, forced[0])
        self.assertEqual((0.0, 100.0), points[0])

    def test_finalizer_nodes_audited_surface_edge_contact_by_splitting_target_edge(self) -> None:
        candidate = surface_candidate((0.0, 50.0), (80.0, 50.0))
        candidate.update({
            "surface_attachment_audit": "accepted_surface_mask_path",
            "surface_attachment_kind": "surface_skeleton_to_graph_edge",
            "surface_attachment_endpoint_position": 0,
            "surface_attachment_projection_row": 0.0,
            "surface_attachment_projection_col": 50.0,
            "surface_attachment_surface_support_ratio": 1.0,
            "surface_attachment_evidence_mode": "continuous_surface_mask",
        })
        nodes = [(0, 0), (0, 100)]
        edges = [(0, 1)]
        keys = {(0, 1)}
        records = [{"src_idx": 0, "dst_idx": 1, "src_row": 0, "src_col": 0, "dst_row": 0, "dst_col": 100, "final_status": "geometry_kept", "source": "samroad"}]

        points, forced = finalize.finalization_candidate_points(candidate, nodes, edges, keys, records)

        self.assertEqual((0, 50), nodes[2])
        self.assertEqual([(0, 2), (2, 1)], edges)
        self.assertEqual({0: 2}, forced)
        self.assertEqual((0.0, 50.0), points[0])

    def test_finalizer_reuses_audited_surface_mask_path_endpoint_node(self) -> None:
        candidate = surface_candidate((20.0, 10.0), (50.0, 10.0))
        candidate.update({
            "surface_attachment_audit": "accepted_surface_mask_path",
            "surface_attachment_kind": "surface_skeleton_to_graph_endpoint",
            "surface_attachment_endpoint_position": 0,
            "surface_attachment_node_row": 20.0,
            "surface_attachment_node_col": 10.0,
            "surface_attachment_surface_support_ratio": 1.0,
            "surface_attachment_evidence_mode": "continuous_surface_mask",
        })
        nodes = [(10, 10), (20, 10)]

        points, forced = finalize.finalization_candidate_points(candidate, nodes, [(0, 1)], {(0, 1)}, [])

        self.assertEqual({0: 1}, forced)
        self.assertEqual((20.0, 10.0), points[0])
        self.assertEqual([(10, 10), (20, 10)], nodes)

    def test_finalizer_keeps_surface_skeleton_points_without_remote_attachments(self) -> None:
        candidate = surface_candidate((20.0, 0.0), (80.0, 0.0))
        candidate.update({
            "connection_type": "component_bridge",
            "connection_start_node_row": 0.0,
            "connection_start_node_col": 0.0,
            "connection_end_node_row": 100.0,
            "connection_end_node_col": 0.0,
        })
        nodes = [(0, 0), (100, 0)]
        points, forced = finalize.finalization_candidate_points(
            candidate, nodes, [], set(), []
        )
        self.assertEqual([(20.0, 0.0), (80.0, 0.0)], points)
        self.assertEqual({}, forced)
        self.assertEqual([(0, 0), (100, 0)], nodes)

    def test_finalizer_keeps_surface_skeleton_connector_astar_geometry_exact(self) -> None:
        candidate = surface_candidate((20.0, 0.0), (40.0, 10.0))
        candidate.update({
            "candidate_type": "surface_skeleton_connector",
            "polyline_points_json": json.dumps([[20.0, 0.0], [30.0, 3.0], [40.0, 10.0]]),
            "connection_type": "component_bridge",
            "connection_start_node_row": 0.0,
            "connection_start_node_col": 0.0,
            "connection_end_node_row": 100.0,
            "connection_end_node_col": 0.0,
        })
        nodes = [(0, 0), (100, 0)]
        points, forced = finalize.finalization_candidate_points(candidate, nodes, [], set(), [])
        self.assertEqual([(20.0, 0.0), (30.0, 3.0), (40.0, 10.0)], points)
        self.assertEqual({}, forced)
        self.assertEqual([(0, 0), (100, 0)], nodes)


class ChangePreviewTests(unittest.TestCase):
    def test_empty_prediction_remains_evaluable_as_zero_recall(self) -> None:
        predicted = gpd.GeoDataFrame(
            {"change_typ": []}, geometry=gpd.GeoSeries([], crs="EPSG:3857"), crs="EPSG:3857",
        )
        truth = gpd.GeoDataFrame(
            {"BHBM": [2]},
            geometry=[Polygon([(0, 0), (20, 0), (20, 4), (0, 4)])], crs="EPSG:3857",
        )
        rows, _metadata = change.evaluate_changes(
            predicted, truth, None, "BHBM", 1.0, class_mode="three",
        )
        overall = rows[0]
        self.assertEqual(overall["change_area_recall"], 0.0)
        self.assertEqual(overall["centerline_offset_status"], "unavailable")

    def test_centerline_offset_excludes_width_change_truth(self) -> None:
        predicted = gpd.GeoDataFrame(
            {"change_typ": ["added", "width_changed", "removed"]},
            geometry=[
                Polygon([(0, 0), (20, 0), (20, 4), (0, 4)]),
                Polygon([(0, 10), (20, 10), (20, 12), (0, 12)]),
                Polygon([(0, 20), (20, 20), (20, 24), (0, 24)]),
            ], crs="EPSG:3857",
        )
        truth = gpd.GeoDataFrame(
            {"BHBM": [2, 3, 4]},
            geometry=[
                Polygon([(0, 1), (20, 1), (20, 5), (0, 5)]),
                Polygon([(0, 10), (20, 10), (20, 12), (0, 12)]),
                Polygon([(0, 21), (20, 21), (20, 25), (0, 25)]),
            ], crs="EPSG:3857",
        )
        validation = gpd.GeoDataFrame(
            geometry=[Polygon([(-5, -5), (30, -5), (30, 30), (-5, 30)])], crs="EPSG:3857",
        )
        rows, metadata = change.evaluate_changes(
            predicted, truth, validation, "BHBM", 1.0, class_mode="three",
        )
        by_class = {row["class"]: row for row in rows}
        self.assertEqual(by_class["width_changed"]["centerline_offset_status"], "excluded")
        self.assertEqual(by_class["all"]["excluded_truth_feature_count"], 1)
        self.assertEqual(by_class["all"]["centerline_offset_status"], "computed")
        self.assertGreater(float(by_class["all"]["centerline_avg_offset_m"]), 0)
        self.assertIn("BHBM=3", metadata["centerline_offset_definition"])

    def test_centerline_offset_merges_dual_carriageway_prediction(self) -> None:
        predicted = gpd.GeoDataFrame(
            {"change_typ": ["added", "added"]},
            geometry=[
                Polygon([(0, -5), (40, -5), (40, -1), (0, -1)]),
                Polygon([(0, 1), (40, 1), (40, 5), (0, 5)]),
            ], crs="EPSG:3857",
        )
        truth = gpd.GeoDataFrame(
            {"BHBM": [2]},
            geometry=[Polygon([(0, -5), (40, -5), (40, 5), (0, 5)])], crs="EPSG:3857",
        )
        rows, _metadata = change.evaluate_changes(
            predicted, truth, None, "BHBM", 2.0, class_mode="three",
        )
        added = next(row for row in rows if row["class"] == "added")
        self.assertEqual(added["centerline_offset_status"], "computed")
        self.assertLess(float(added["centerline_avg_offset_m"]), 2.0)
    def test_change_preview_is_nonempty_for_changes_and_empty_changes(self) -> None:
        crs = "EPSG:3857"
        unchanged = gpd.GeoDataFrame({"area_m2": [4.0]}, geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], crs=crs)
        added = gpd.GeoDataFrame({"change_typ": ["added"], "area_m2": [2.0]}, geometry=[Polygon([(3, 0), (4, 0), (4, 2), (3, 2)])], crs=crs)
        removed = gpd.GeoDataFrame({"change_typ": ["narrowed"], "area_m2": [1.0]}, geometry=[Polygon([(5, 0), (6, 0), (6, 1), (5, 1)])], crs=crs)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "change_preview.png"
            change.write_change_preview(output, added, removed, unchanged)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)
            empty = Path(tmp) / "empty_preview.png"
            change.write_change_preview(
                empty,
                added.iloc[0:0],
                removed.iloc[0:0],
                unchanged.iloc[0:0],
            )
            self.assertTrue(empty.is_file())
            self.assertGreater(empty.stat().st_size, 100)
            detected = Path(tmp) / "detected"
            change._write_outputs(detected, added, removed, unchanged, crs)
            self.assertTrue((detected / "change_preview.png").is_file())
            self.assertTrue((detected / "review_preview.png").is_file())
            detected_empty = Path(tmp) / "detected_empty"
            change._write_outputs(detected_empty, added.iloc[0:0], removed.iloc[0:0], unchanged.iloc[0:0], crs)
            self.assertTrue((detected_empty / "change_preview.png").is_file())
            self.assertTrue((detected_empty / "review_preview.png").is_file())
            self.assertGreater((detected_empty / "review_preview.png").stat().st_size, 100)

    def test_only_review_rows_are_sent_exclusively_to_review_preview(self) -> None:
        crs = "EPSG:3857"
        added = gpd.GeoDataFrame(
            {"change_typ": ["added"], "qa_state": ["review"]},
            geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], crs=crs,
        )
        removed = gpd.GeoDataFrame(
            {"change_typ": ["removed"], "qa_state": ["review"]},
            geometry=[Polygon([(3, 0), (5, 0), (5, 2), (3, 2)])], crs=crs,
        )
        unchanged = gpd.GeoDataFrame(geometry=[], crs=crs)
        captured: dict[str, list[str]] = {}

        def capture(path, rows, _reference, **_kwargs):
            captured[Path(path).name] = list(rows["change_typ"])

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(change, "render_change_preview", side_effect=capture), \
                patch.object(change.pyogrio, "write_dataframe"), \
                patch.object(gpd.GeoDataFrame, "to_file"):
            change._write_outputs(
                Path(tmp), added, removed, unchanged, crs, include_artifacts=False,
            )

        self.assertEqual(captured["change_preview.png"], [])
        self.assertCountEqual(captured["review_preview.png"], ["added", "removed"])

    def test_formal_and_review_rows_are_sent_to_separate_previews(self) -> None:
        crs = "EPSG:3857"
        added = gpd.GeoDataFrame(
            {"change_typ": ["added", "widened"], "qa_state": ["auto", "auto"]},
            geometry=[
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(3, 0), (5, 0), (5, 2), (3, 2)]),
            ], crs=crs,
        )
        removed = gpd.GeoDataFrame(
            {"change_typ": ["removed"], "qa_state": ["review"]},
            geometry=[Polygon([(6, 0), (8, 0), (8, 2), (6, 2)])], crs=crs,
        )
        unchanged = gpd.GeoDataFrame(geometry=[], crs=crs)
        captured: dict[str, list[str]] = {}

        def capture(path, rows, _reference, **_kwargs):
            captured[Path(path).name] = list(rows["change_typ"])

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(change, "render_change_preview", side_effect=capture), \
                patch.object(change.pyogrio, "write_dataframe"), \
                patch.object(gpd.GeoDataFrame, "to_file"):
            change._write_outputs(
                Path(tmp), added, removed, unchanged, crs, include_artifacts=False,
            )

        self.assertCountEqual(captured["change_preview.png"], ["added", "widened"])
        self.assertEqual(captured["review_preview.png"], ["removed"])


class FusionComparisonTests(unittest.TestCase):
    def test_source_preview_distinguishes_original_automatic_and_manual_lines(self) -> None:
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        records = []
        for row, source in enumerate(
            ("samroad", "auto_added_surface", "auto_added_gap", "manual_edited"), start=3
        ):
            records.append({
                "source": source,
                "final_status": "geometry_kept",
                "src_row": row * 25, "dst_row": row * 25,
                "src_col": 20, "dst_col": 290,
            })
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fusion_comparison.png"
            finalize.draw_viz(
                image, None, None, [], records, [], {}, "tile", output,
                color_mode="source",
            )
            rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(rendered)
            self.assertTrue(np.array_equal(rendered[75, 280], np.asarray([0, 210, 0])))
            self.assertTrue(np.array_equal(rendered[100, 280], np.asarray([255, 0, 255])))
            self.assertTrue(np.array_equal(rendered[125, 280], np.asarray([255, 220, 0])))
            self.assertTrue(np.array_equal(rendered[150, 280], np.asarray([0, 145, 255])))


if __name__ == "__main__":
    unittest.main()
