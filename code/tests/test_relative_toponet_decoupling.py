from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SAMROAD = ROOT / "engine" / "samroad"
CONFIG_PATH = ROOT.parent / "runtime" / "config" / "samroad_inference.yaml"
if str(SAMROAD) not in sys.path:
    sys.path.insert(0, str(SAMROAD))

import graph_extraction  # noqa: E402
from utils import load_config  # noqa: E402

_ORIGINAL_ARGV = sys.argv
try:
    sys.argv = ["inferencer.py"]
    import inferencer  # noqa: E402
finally:
    sys.argv = _ORIGINAL_ARGV


class _MaskOnlyNet:
    def infer_masks_and_img_features(self, batch):
        batch_size, height, width, _channels = batch.shape
        masks = torch.zeros((batch_size, height, width, 2), dtype=torch.float32)
        features = torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32)
        return masks, features


class _FastTopoNet(_MaskOnlyNet):
    def infer_masks_and_img_features(self, batch):
        masks, features = super().infer_masks_and_img_features(batch)
        masks[:, :, 15:17, 1] = 0.04
        masks[:, 4, 16, 0] = 0.95
        masks[:, 27, 16, 0] = 0.95
        return masks, features

    def infer_toponet(self, _features, _points, pairs, _valid, _mask):
        return torch.ones((*pairs.shape[:-1], 1), dtype=torch.float32)


class RelativeTopoNetDecouplingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG_PATH)

    def _relative_only_scene(self):
        probability = np.full((96, 160), 0.01, dtype=np.float32)
        cv2.line(probability, (12, 48), (147, 48), 0.22, 7)
        profile = graph_extraction.resolve_effective_road_profile(
            probability, self.config
        )
        self.config.ROAD_THRESHOLD_PROFILE = profile["effective_profile"]
        context = graph_extraction.compute_relative_roadness(
            probability,
            self.config,
            scene_state=profile["scene_confidence_state"],
        )
        return probability, context

    def test_default_excludes_relative_points_but_postprocess_recovers_road(self):
        probability, context = self._relative_only_scene()
        keypoints = np.zeros(probability.shape, dtype=np.uint8)
        roads = np.rint(probability * 255.0).astype(np.uint8)

        self.config.RELATIVE_INJECT_INTO_TOPONET = False
        native_points = graph_extraction.extract_graph_points(
            keypoints, roads, self.config
        )
        toponet_points = graph_extraction.extract_graph_points(
            keypoints, roads, self.config, relative_context=context
        )
        relative_point_count = int(np.count_nonzero(
            context["relative_only_skeleton"]
        ))

        self.assertTrue(np.array_equal(toponet_points, native_points))
        self.assertEqual(toponet_points.shape[0], 0)
        self.assertGreater(relative_point_count, 0)

        _nodes, edges, metadata, summary = (
            graph_extraction.postprocess_weak_road_network(
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.int32),
                probability,
                self.config,
                relative_context=context,
            )
        )
        self.assertGreater(edges.shape[0], 0)
        self.assertGreater(summary["relative_final_length_px"], 0.0)
        self.assertTrue(any(
            row.get("candidate_source") in {"relative", "absolute+relative"}
            or str(row.get("line_source", "")).startswith("relative")
            for row in metadata
        ))

    def test_legacy_switch_restores_relative_point_injection(self):
        probability, context = self._relative_only_scene()
        keypoints = np.zeros(probability.shape, dtype=np.uint8)
        roads = np.rint(probability * 255.0).astype(np.uint8)

        self.config.RELATIVE_INJECT_INTO_TOPONET = False
        native_points = graph_extraction.extract_graph_points(
            keypoints, roads, self.config, relative_context=context
        )
        self.config.RELATIVE_INJECT_INTO_TOPONET = True
        legacy_points = graph_extraction.extract_graph_points(
            keypoints, roads, self.config, relative_context=context
        )

        self.assertEqual(native_points.shape[0], 0)
        self.assertGreater(legacy_points.shape[0], native_points.shape[0])

    def test_relative_disabled_matches_native_pipeline(self):
        probability, context = self._relative_only_scene()
        keypoints = np.zeros(probability.shape, dtype=np.uint8)
        roads = np.rint(probability * 255.0).astype(np.uint8)
        self.config.RELATIVE_ROADNESS_ENABLED = False
        self.config.RELATIVE_INJECT_INTO_TOPONET = True

        expected = graph_extraction.extract_graph_points(
            keypoints, roads, self.config
        )
        actual = graph_extraction.extract_graph_points(
            keypoints, roads, self.config, relative_context=context
        )

        self.assertTrue(np.array_equal(actual, expected))

    def test_normal_production_sequence_computes_relative_once(self):
        config = copy.deepcopy(self.config)
        config.PATCH_SIZE = 32
        config.INFER_BATCH_SIZE = 1
        config.SAMPLE_MARGIN = 0
        config.RELATIVE_INJECT_INTO_TOPONET = False
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        original_device = inferencer.args.device
        inferencer.args.device = "cpu"
        original_compute = graph_extraction.compute_relative_roadness
        try:
            with mock.patch.object(
                graph_extraction,
                "compute_relative_roadness",
                wraps=original_compute,
            ) as compute:
                result = inferencer.infer_one_img(
                    _MaskOnlyNet(), image, config, diagnostic_shape=(32, 32)
                )
                self.assertIsNone(result[8])
                self.assertEqual(result[9]["relative_compute_call_count"], 0)
                self.assertIs(result[9]["relative_injected_into_toponet"], False)
                self.assertEqual(
                    result[9]["toponet_graph_point_count"],
                    result[9]["native_graph_point_count"],
                )
                for field in (
                    "mask_inference_seconds",
                    "native_graph_and_toponet_seconds",
                    "relative_roadness_seconds",
                    "weak_postprocess_seconds",
                    "total_image_seconds",
                    "toponet_candidate_edge_count",
                    "toponet_pred_edge_count",
                ):
                    self.assertIn(field, result[9])
                context, added_calls = inferencer.resolve_relative_context_for_postprocess(
                    result[6],
                    config,
                    scene_state=result[7]["scene_confidence_state"],
                )
                _same_context, reused_calls = (
                    inferencer.resolve_relative_context_for_postprocess(
                        result[6],
                        config,
                        scene_state=result[7]["scene_confidence_state"],
                        precomputed_context=context,
                    )
                )
        finally:
            inferencer.args.device = original_device

        self.assertEqual(compute.call_count, 1)
        self.assertEqual(added_calls, 1)
        self.assertEqual(reused_calls, 0)

    def test_fast_uses_enhanced_native_graph_and_one_toponet_pass(self):
        config = copy.deepcopy(self.config)
        config.PATCH_SIZE = 32
        config.INFER_BATCH_SIZE = 1
        config.SAMPLE_MARGIN = 0
        config.ROAD_THRESHOLD_PROFILE = "default"
        config.ROAD_THRESHOLD_PROFILES = {}
        config.ROAD_HIGH_THRESHOLD = 0.20
        config.ROAD_LOW_THRESHOLD = 0.10
        config.RELATIVE_ROADNESS_ENABLED = False
        config.RELATIVE_INJECT_INTO_TOPONET = False
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        original_device = inferencer.args.device
        original_profile = inferencer.args.execution_profile
        inferencer.args.device = "cpu"
        inferencer.args.execution_profile = "fast"
        net = _FastTopoNet()
        try:
            with mock.patch.object(net, "infer_toponet", wraps=net.infer_toponet) as toponet:
                result = inferencer.infer_one_img(net, image, config, diagnostic_shape=(32, 32))
        finally:
            inferencer.args.device = original_device
            inferencer.args.execution_profile = original_profile

        self.assertGreater(result[0].shape[0], 0)
        self.assertGreater(result[1].shape[0], 0)
        self.assertEqual(toponet.call_count, 1)
        self.assertGreaterEqual(result[9]["fast_graph_point_count"], result[9]["raw_graph_point_count"])
        self.assertGreater(result[9]["relative_boost_pixel_count"], 0)
        self.assertEqual(result[9]["relative_compute_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
