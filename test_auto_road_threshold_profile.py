from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

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

    def __setattr__(self, name, value):
        self[name] = value


def config(requested="auto"):
    return Config(
        ROAD_THRESHOLD=0.364,
        ROAD_HIGH_THRESHOLD=0.364,
        ROAD_LOW_THRESHOLD=0.20,
        ROAD_THRESHOLD_PROFILE=requested,
        ROAD_THRESHOLD_PROFILES={
            "default": {"ROAD_HIGH_THRESHOLD": 0.364, "ROAD_LOW_THRESHOLD": 0.20},
            "weak_sensor": {"ROAD_HIGH_THRESHOLD": 0.24, "ROAD_LOW_THRESHOLD": 0.10},
        },
        SCENE_DIAGNOSTIC_REFERENCE_PROFILE="default",
        WEAK_BOOTSTRAP_CLOSE_KERNEL=3,
        WEAK_BOOTSTRAP_MIN_LENGTH_PX=48.0,
        ITSC_THRESHOLD=0.248,
        ITSC_NMS_RADIUS=8,
        ROAD_NMS_RADIUS=16,
        ROAD_NMS_MIN_SEPARATION=4.0,
        ROAD_TANGENT_RADIUS=5.0,
        PARALLEL_BRANCH_COSINE=0.90,
        PARALLEL_BRANCH_LATERAL_COSINE=0.55,
        JUNCTION_NODE_MODE="sparse",
        JUNCTION_SPARSE_RADIUS=20.0,
        JUNCTION_POINT_MERGE_RADIUS=4.0,
    )


def scene(value):
    probability = np.full((96, 128), 0.03, dtype=np.float32)
    probability[47:50, 8:121] = value
    return probability


class AutoRoadThresholdProfileTests(unittest.TestCase):
    def test_normal_scene_auto_selects_default(self):
        decision = graph_extraction.resolve_effective_road_profile(scene(0.70), config())
        self.assertEqual(decision["scene_confidence_state"], "normal")
        self.assertEqual(decision["recommended_profile"], "default")
        self.assertEqual(decision["effective_profile"], "default")

    def test_weak_scene_auto_selects_weak_sensor_and_changes_point_extraction(self):
        road = scene(0.22)
        road[47:50, 50:70] = 0.30
        cfg = config()
        decision = graph_extraction.resolve_effective_road_profile(road, cfg)
        self.assertEqual(decision["scene_confidence_state"], "low_confidence")
        self.assertEqual(decision["effective_profile"], "weak_sensor")
        keypoints = np.zeros_like(road, dtype=np.uint8)
        road_u8 = np.rint(road * 255).astype(np.uint8)
        default_points = graph_extraction.extract_graph_points(
            keypoints, road_u8, config("default")
        )
        effective = config()
        effective.ROAD_THRESHOLD_PROFILE = decision["effective_profile"]
        auto_points = graph_extraction.extract_graph_points(keypoints, road_u8, effective)
        self.assertEqual(len(default_points), 0)
        self.assertGreater(len(auto_points), 0)

    def test_manual_default_overrides_weak_recommendation(self):
        decision = graph_extraction.resolve_effective_road_profile(scene(0.22), config("default"))
        self.assertEqual(decision["recommended_profile"], "weak_sensor")
        self.assertEqual(decision["effective_profile"], "default")

    def test_manual_weak_sensor_overrides_normal_recommendation(self):
        decision = graph_extraction.resolve_effective_road_profile(scene(0.70), config("weak_sensor"))
        self.assertEqual(decision["recommended_profile"], "default")
        self.assertEqual(decision["effective_profile"], "weak_sensor")

    def test_sequential_images_do_not_mutate_shared_config(self):
        shared = config("auto")
        first = graph_extraction.resolve_effective_road_profile(scene(0.22), copy.deepcopy(shared))
        second = graph_extraction.resolve_effective_road_profile(scene(0.70), copy.deepcopy(shared))
        self.assertEqual(first["effective_profile"], "weak_sensor")
        self.assertEqual(second["effective_profile"], "default")
        self.assertEqual(shared.ROAD_THRESHOLD_PROFILE, "auto")

    def test_no_usable_structure_does_not_lower_thresholds(self):
        decision = graph_extraction.resolve_effective_road_profile(
            np.zeros((96, 128), dtype=np.float32), config()
        )
        self.assertEqual(decision["scene_confidence_state"], "very_low_confidence")
        self.assertEqual(decision["effective_profile"], "default")

    def test_postprocess_qa_uses_effective_profile(self):
        cfg = config()
        decision = graph_extraction.resolve_effective_road_profile(scene(0.22), cfg)
        cfg.ROAD_THRESHOLD_PROFILE = decision["effective_profile"]
        diagnosis = graph_extraction.diagnose_scene_confidence(
            scene(0.22), np.empty((0, 2)), np.empty((0, 2), dtype=np.int32), cfg
        )
        self.assertEqual(diagnosis["active_profile"], decision["effective_profile"])

    def test_profile_decisions_report_is_per_image_and_marks_mixed(self):
        report = graph_extraction.summarize_profile_decisions("auto", "default", [
            {"image": "a.tif", "effective_profile": "weak_sensor"},
            {"image": "b.tif", "effective_profile": "default"},
        ])
        self.assertEqual(report["weak_sensor_image_count"], 1)
        self.assertEqual(report["default_image_count"], 1)
        self.assertTrue(report["mixed_profile"])
        self.assertEqual(len(report["decisions"]), 2)


if __name__ == "__main__":
    unittest.main()
