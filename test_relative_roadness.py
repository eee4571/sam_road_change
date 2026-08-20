from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


SAMROAD = Path(__file__).resolve().parent / "engine" / "samroad"
if str(SAMROAD) not in sys.path:
    sys.path.insert(0, str(SAMROAD))

import graph_extraction  # noqa: E402


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def relative_config():
    return Config(
        ROAD_THRESHOLD=0.364,
        ROAD_HIGH_THRESHOLD=0.364,
        ROAD_LOW_THRESHOLD=0.20,
        ROAD_THRESHOLD_PROFILE="default",
        RELATIVE_ROADNESS_ENABLED=True,
        RELATIVE_ROADNESS_BACKGROUND_SCALES_PX=[9, 21, 41],
        RELATIVE_ROADNESS_NORMAL_WEAK_PERCENTILE=45.0,
        RELATIVE_ROADNESS_NORMAL_STRONG_PERCENTILE=95.0,
        RELATIVE_ROADNESS_LOW_SCENE_WEAK_PERCENTILE=25.0,
        RELATIVE_ROADNESS_LOW_SCENE_STRONG_PERCENTILE=85.0,
        RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX=32.0,
        RELATIVE_ROADNESS_MIN_ELONGATION=3.0,
        RELATIVE_ROADNESS_MAX_TORTUOSITY=1.5,
        RELATIVE_ROADNESS_ABSOLUTE_SUPPRESSION_PX=3.0,
        RELATIVE_ROADNESS_CLOSE_KERNEL=3,
        WEAK_BOOTSTRAP_ENABLED=True,
        WEAK_BOOTSTRAP_CLOSE_KERNEL=3,
        WEAK_BOOTSTRAP_MIN_LENGTH_PX=32.0,
        WEAK_BOOTSTRAP_MIN_MEAN_PROBABILITY=0.16,
        WEAK_BOOTSTRAP_MIN_Q25_PROBABILITY=0.12,
        WEAK_BOOTSTRAP_MIN_BACKGROUND_CONTRAST=0.08,
        WEAK_BOOTSTRAP_MAX_TORTUOSITY=1.5,
        WEAK_BOOTSTRAP_MIN_WEAK_FRACTION=0.80,
        WEAK_BOOTSTRAP_STRONG_SUPPRESSION_PX=3.0,
        WEAK_BOOTSTRAP_STRONG_CONNECTION_PX=10.0,
        WEAK_BOOTSTRAP_SAMPLE_STEP_PX=12.0,
        WEAK_BOOTSTRAP_AUTO_SCORE=0.74,
        WEAK_BOOTSTRAP_INDEPENDENT_LENGTH_FACTOR=1.5,
        WEAK_RECOVERY_MAX_GAP_PX=64.0,
        WEAK_RECOVERY_BACKGROUND_OFFSET_PX=4.0,
        WEAK_RECOVERY_SURFACE_THRESHOLD=0.60,
        WEAK_RECOVERY_SURFACE_MIN_CENTER_PROBABILITY=0.10,
        WEAK_RECOVERY_SURFACE_MIN_MEAN=0.70,
        WEAK_RECOVERY_SURFACE_MIN_FRACTION=0.80,
    )


def straight_scene(center, background):
    probability = np.full((128, 192), background, dtype=np.float32)
    probability[62:65, 16:176] = center
    return probability


class RelativeRoadnessTests(unittest.TestCase):
    @staticmethod
    def _ladder_skeleton(with_branch=False):
        skeleton = np.zeros((140, 240), dtype=np.uint8)
        cv2.line(skeleton, (15, 52), (225, 52), 1, 1)
        cv2.line(skeleton, (15, 64), (225, 64), 1, 1)
        for col in range(40, 221, 16):
            cv2.line(skeleton, (col, 52), (col, 64), 1, 1)
        if with_branch:
            cv2.line(skeleton, (120, 64), (120, 100), 1, 1)
        return skeleton

    @staticmethod
    def _normalize_synthetic(skeleton, config=None):
        candidate = cv2.dilate(skeleton, np.ones((3, 3), dtype=np.uint8))
        evidence = candidate.astype(np.float32)
        return graph_extraction.normalize_relative_skeleton(
            skeleton,
            candidate,
            config or relative_config(),
            relative_score=evidence,
            scale_agreement=evidence,
        )

    def test_ladder_junction_zone_becomes_one_long_corridor(self):
        cfg = relative_config()
        cfg["RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX"] = 48.0
        result = self._normalize_synthetic(self._ladder_skeleton(), cfg)
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["raw_chain_count"], 38)
        self.assertEqual(diagnostics["raw_short_chain_count"], 38)
        self.assertEqual(diagnostics["normalized_chain_count"], 1)
        self.assertEqual(diagnostics["normalized_short_chain_count"], 0)
        self.assertGreater(diagnostics["normalized_max_chain_length"], 200.0)
        self.assertEqual(diagnostics["collapsed_zone_count"], 1)
        zone = diagnostics["junction_zones"][0]
        self.assertEqual(len(zone["branch_lengths"]), zone["incident_branch_count"])
        self.assertEqual(len(zone["branch_tangents"]), zone["incident_branch_count"])

    def test_dense_ladder_keeps_real_branch_that_leaves_zone(self):
        cfg = relative_config()
        cfg["RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX"] = 48.0
        result = self._normalize_synthetic(self._ladder_skeleton(with_branch=True), cfg)
        normalized = result["normalized_skeleton"] > 0
        self.assertGreater(np.count_nonzero(normalized[65:101, 118:123]), 30)
        self.assertEqual(result["diagnostics"]["junction_zones"][0]["preserved_branch_count"], 1)

    def test_compact_t_and_x_junctions_are_unchanged(self):
        for branch_start in (70, 20):
            skeleton = np.zeros((140, 180), dtype=np.uint8)
            cv2.line(skeleton, (15, 70), (165, 70), 1, 1)
            cv2.line(skeleton, (90, branch_start), (90, 125), 1, 1)
            result = self._normalize_synthetic(skeleton)
            self.assertTrue(np.array_equal(result["normalized_skeleton"] > 0, skeleton > 0))
            self.assertEqual(result["diagnostics"]["collapsed_zone_count"], 0)

    def test_roof_grid_is_not_normalized_into_a_road(self):
        skeleton = np.zeros((140, 200), dtype=np.uint8)
        for row in (30, 60, 90):
            cv2.line(skeleton, (30, row), (150, row), 1, 1)
        for col in (40, 80, 120):
            cv2.line(skeleton, (col, 20), (col, 100), 1, 1)
        cfg = relative_config()
        cfg["RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX"] = 48.0
        result = self._normalize_synthetic(skeleton, cfg)
        self.assertEqual(result["diagnostics"]["collapsed_zone_count"], 0)
        self.assertTrue(np.array_equal(result["normalized_skeleton"] > 0, skeleton > 0))
        retained, _rejected, _summary = graph_extraction.extract_relative_skeleton(
            cv2.dilate(skeleton, np.ones((3, 3), dtype=np.uint8)),
            cfg,
            input_skeleton=result["normalized_skeleton"],
        )
        self.assertEqual(np.count_nonzero(retained), 0)

    def test_parallel_roads_and_ring_are_unchanged(self):
        parallel = np.zeros((140, 240), dtype=np.uint8)
        cv2.line(parallel, (15, 52), (225, 52), 1, 1)
        cv2.line(parallel, (15, 64), (225, 64), 1, 1)
        ring = np.zeros((140, 240), dtype=np.uint8)
        cv2.ellipse(ring, (110, 70), (52, 35), 0, 0, 360, 1, 1)
        for skeleton in (parallel, ring):
            result = self._normalize_synthetic(skeleton)
            self.assertTrue(np.array_equal(result["normalized_skeleton"] > 0, skeleton > 0))
            self.assertEqual(result["diagnostics"]["collapsed_zone_count"], 0)

    def test_t_junction_is_filtered_chain_by_chain_not_by_component_elongation(self):
        candidate = np.zeros((128, 128), dtype=np.uint8)
        cv2.line(candidate, (12, 62), (116, 62), 1, 1)
        cv2.line(candidate, (64, 62), (64, 116), 1, 1)
        retained, _rejected, summary = graph_extraction.extract_relative_skeleton(
            candidate, relative_config()
        )
        self.assertGreater(np.count_nonzero(retained[60:65, 12:117]), 90)
        self.assertGreater(np.count_nonzero(retained[62:117, 62:67]), 45)
        self.assertGreaterEqual(summary["relative_chain_geometry_pass"], 3)

    def test_long_smooth_curve_is_not_rejected_by_endpoint_tortuosity(self):
        candidate = np.zeros((160, 160), dtype=np.uint8)
        cv2.ellipse(candidate, (80, 80), (52, 52), 0, 0, 210, 1, 1)
        retained, _rejected, summary = graph_extraction.extract_relative_skeleton(
            candidate, relative_config()
        )
        self.assertGreater(np.count_nonzero(retained), 120)
        self.assertEqual(summary["relative_structure_reject_reason_counts"].get("tortuosity", 0), 0)

    def test_calibration_invariance_across_four_probability_scales(self):
        scenes = [
            straight_scene(0.70, 0.05),
            straight_scene(0.28, 0.02),
            straight_scene(0.105, 0.0075),
            straight_scene(0.049, 0.0035),
        ]
        masks = [
            graph_extraction.compute_relative_roadness(
                scene, relative_config(), scene_state="normal"
            )["relative_skeleton"] > 0
            for scene in scenes
        ]
        self.assertGreater(np.count_nonzero(masks[0]), 100)
        for mask in masks[1:]:
            intersection = np.count_nonzero(mask & masks[0])
            union = np.count_nonzero(mask | masks[0])
            self.assertGreater(intersection / max(union, 1), 0.95)

        final_lengths = []
        for scene in scenes:
            context = graph_extraction.compute_relative_roadness(
                scene, relative_config(), scene_state="normal"
            )
            nodes, edges, _metadata, _summary = graph_extraction.bootstrap_weak_road_network(
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.int32),
                scene,
                relative_config(),
                relative_context=context,
                include_absolute_candidates=False,
            )
            final_lengths.append(sum(
                float(np.linalg.norm(nodes[dst] - nodes[src])) for src, dst in edges
            ))
        self.assertGreater(final_lengths[0], 100.0)
        for length in final_lengths[1:]:
            self.assertAlmostEqual(length, final_lengths[0], delta=1.0)

    def test_background_only_noise_does_not_form_long_roads(self):
        rng = np.random.default_rng(7)
        probability = np.clip(
            rng.normal(0.05, 0.004, size=(128, 192)), 0.0, 1.0
        ).astype(np.float32)
        result = graph_extraction.compute_relative_roadness(
            probability, relative_config(), scene_state="normal"
        )
        nodes, edges, _metadata, _summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            probability,
            relative_config(),
            relative_context=result,
            include_absolute_candidates=False,
        )
        false_length = sum(float(np.linalg.norm(nodes[dst] - nodes[src])) for src, dst in edges)
        self.assertEqual(false_length, 0.0)

    def test_compact_blobs_and_rectangles_are_rejected(self):
        probability = np.full((128, 192), 0.04, dtype=np.float32)
        cv2.circle(probability, (45, 45), 16, 0.30, -1)
        cv2.rectangle(probability, (115, 32), (150, 67), 0.25, -1)
        result = graph_extraction.compute_relative_roadness(
            probability, relative_config(), scene_state="low_confidence"
        )
        nodes, edges, _metadata, _summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            probability,
            relative_config(),
            relative_context=result,
            include_absolute_candidates=False,
        )
        self.assertEqual(len(edges), 0)

    def test_strong_and_weak_roads_coexist(self):
        probability = np.full((160, 192), 0.005, dtype=np.float32)
        probability[39:42, 12:180] = 0.75
        probability[119:122, 12:180] = 0.08
        result = graph_extraction.compute_relative_roadness(
            probability, relative_config(), scene_state="normal"
        )
        skeleton = result["relative_skeleton"] > 0
        self.assertGreater(np.count_nonzero(skeleton[35:46]), 100)
        self.assertGreater(np.count_nonzero(skeleton[115:126]), 100)

    def test_absolute_relative_merge_is_deduplicated(self):
        result = graph_extraction.compute_relative_roadness(
            straight_scene(0.70, 0.05), relative_config(), scene_state="normal"
        )
        absolute = result["absolute_skeleton"] > 0
        relative_only = result["relative_only_skeleton"] > 0
        combined = result["combined_skeleton"] > 0
        self.assertEqual(np.count_nonzero(absolute & relative_only), 0)
        self.assertEqual(
            np.count_nonzero(combined),
            np.count_nonzero(absolute) + np.count_nonzero(relative_only),
        )

    def test_relative_bootstrap_is_not_blocked_by_raw_probability_q25(self):
        cfg = relative_config()
        road = straight_scene(0.07, 0.01)
        context = graph_extraction.compute_relative_roadness(
            road, cfg, scene_state="low_confidence"
        )
        _nodes, _edges, metadata, summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            road,
            cfg,
            relative_context=context,
            include_absolute_candidates=False,
        )
        self.assertGreater(summary["relative_recovered_edge_count"], 0)
        self.assertIn("relative_roadness", {row["line_source"] for row in metadata})
        self.assertTrue(all(row["center_conf"] < 0.12 for row in metadata))

    def test_relative_review_is_preserved_without_being_called_rejected(self):
        cfg = relative_config()
        cfg["WEAK_BOOTSTRAP_AUTO_SCORE"] = 0.99
        road = straight_scene(0.07, 0.01)
        context = graph_extraction.compute_relative_roadness(
            road, cfg, scene_state="low_confidence"
        )
        context["scale_agreement_fraction"][:] = 0.0
        audit = []
        _nodes, edges, _metadata, summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            road,
            cfg,
            relative_context=context,
            include_absolute_candidates=False,
            candidate_audit=audit,
        )
        self.assertEqual(len(edges), 0)
        self.assertEqual(summary["relative_auto_count"], 0)
        self.assertEqual(summary["relative_review_count"], 1)
        self.assertEqual(summary["relative_recovered_edge_count"], 0)
        self.assertEqual(audit[0]["qa_state"], "review")
        self.assertEqual(audit[0]["decision"], "review")
        self.assertEqual(audit[0]["reject_reason"], "")
        self.assertEqual(audit[0]["review_reason"], "relative_evidence_requires_review")

    def test_low_topology_relative_candidate_is_promoted_by_combined_evidence(self):
        cfg = relative_config()
        cfg["WEAK_BOOTSTRAP_AUTO_SCORE"] = 0.99
        road = np.full((96, 128), 0.01, dtype=np.float32)
        road[47:50, 34:82] = 0.07
        context = graph_extraction.compute_relative_roadness(
            road, cfg, scene_state="low_confidence"
        )
        topology_nodes = np.asarray([[48.0, 34.0], [48.0, 81.0]], dtype=np.float32)
        topology_edges = np.asarray([[0, 1]], dtype=np.int32)
        nodes, edges, metadata, summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            road,
            cfg,
            relative_context=context,
            include_absolute_candidates=False,
            topology_candidate_nodes_rc=topology_nodes,
            topology_candidate_edges=topology_edges,
            topology_candidate_scores=np.asarray([0.30], dtype=np.float32),
        )
        self.assertGreater(len(edges), 0)
        self.assertGreater(sum(np.linalg.norm(nodes[d] - nodes[s]) for s, d in edges), 40.0)
        self.assertEqual(summary["relative_auto_count"], 1)
        self.assertTrue(all(row["topology_probability"] < 0.5 for row in metadata))
        self.assertTrue(all(row["relative_evidence_tier"] == "A" for row in metadata))


if __name__ == "__main__":
    unittest.main()
