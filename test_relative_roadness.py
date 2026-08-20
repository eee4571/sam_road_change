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

    def test_background_only_noise_does_not_form_long_roads(self):
        rng = np.random.default_rng(7)
        probability = np.clip(
            rng.normal(0.05, 0.004, size=(128, 192)), 0.0, 1.0
        ).astype(np.float32)
        result = graph_extraction.compute_relative_roadness(
            probability, relative_config(), scene_state="normal"
        )
        self.assertEqual(np.count_nonzero(result["relative_skeleton"]), 0)

    def test_compact_blobs_and_rectangles_are_rejected(self):
        probability = np.full((128, 192), 0.04, dtype=np.float32)
        cv2.circle(probability, (45, 45), 16, 0.30, -1)
        cv2.rectangle(probability, (115, 32), (150, 67), 0.25, -1)
        result = graph_extraction.compute_relative_roadness(
            probability, relative_config(), scene_state="low_confidence"
        )
        self.assertEqual(np.count_nonzero(result["relative_skeleton"]), 0)

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
        self.assertIn("relative_bootstrap", {row["line_source"] for row in metadata})
        self.assertTrue(all(row["center_conf"] < 0.12 for row in metadata))

    def test_relative_review_candidate_is_not_written_to_final_graph(self):
        cfg = relative_config()
        cfg["WEAK_BOOTSTRAP_AUTO_SCORE"] = 0.99
        road = straight_scene(0.07, 0.01)
        context = graph_extraction.compute_relative_roadness(
            road, cfg, scene_state="low_confidence"
        )
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
        self.assertEqual(audit[0]["reject_reason"], "manual_review_required")


if __name__ == "__main__":
    unittest.main()
