from __future__ import annotations

import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np


SAMROAD = Path(__file__).resolve().parent / "engine" / "samroad"
if str(SAMROAD) not in sys.path:
    sys.path.insert(0, str(SAMROAD))

import graph_extraction  # noqa: E402


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def recovery_config() -> Config:
    return Config(
        ROAD_THRESHOLD=0.50,
        ROAD_HIGH_THRESHOLD=0.50,
        ROAD_LOW_THRESHOLD=0.18,
        ROAD_THRESHOLD_PROFILE="default",
        WEAK_RECOVERY_ENABLED=True,
        WEAK_RECOVERY_MAX_GAP_PX=48.0,
        WEAK_RECOVERY_MAX_EXTENSION_PX=36.0,
        WEAK_RECOVERY_MIN_EXTENSION_PX=8.0,
        WEAK_RECOVERY_MIN_DIRECTION_COSINE=0.70,
        WEAK_RECOVERY_MAX_PATH_RATIO=1.25,
        WEAK_RECOVERY_MIN_MEAN_PROBABILITY=0.20,
        WEAK_RECOVERY_MIN_Q25_PROBABILITY=0.17,
        WEAK_RECOVERY_MIN_WEAK_FRACTION=0.75,
        WEAK_RECOVERY_MIN_BACKGROUND_CONTRAST=0.08,
        WEAK_RECOVERY_BACKGROUND_OFFSET_PX=4.0,
        WEAK_RECOVERY_PATH_MARGIN_PX=10.0,
        WEAK_RECOVERY_SAMPLE_STEP_PX=10.0,
        WEAK_RECOVERY_AUTO_SCORE=0.62,
        WEAK_RECOVERY_SURFACE_THRESHOLD=0.60,
        WEAK_RECOVERY_SURFACE_MIN_CENTER_PROBABILITY=0.10,
        WEAK_RECOVERY_SURFACE_MIN_MEAN=0.70,
        WEAK_RECOVERY_SURFACE_MIN_FRACTION=0.80,
        WEAK_BOOTSTRAP_ENABLED=True,
        WEAK_BOOTSTRAP_ONLY_IF_LOW_CONFIDENCE=True,
        WEAK_BOOTSTRAP_CLOSE_KERNEL=3,
        WEAK_BOOTSTRAP_MIN_LENGTH_PX=30.0,
        WEAK_BOOTSTRAP_MIN_MEAN_PROBABILITY=0.16,
        WEAK_BOOTSTRAP_MIN_Q25_PROBABILITY=0.12,
        WEAK_BOOTSTRAP_MIN_BACKGROUND_CONTRAST=0.08,
        WEAK_BOOTSTRAP_MAX_TORTUOSITY=1.35,
        WEAK_BOOTSTRAP_MIN_WEAK_FRACTION=0.80,
        WEAK_BOOTSTRAP_STRONG_SUPPRESSION_PX=3.0,
        WEAK_BOOTSTRAP_STRONG_CONNECTION_PX=10.0,
        WEAK_BOOTSTRAP_SAMPLE_STEP_PX=10.0,
        WEAK_BOOTSTRAP_AUTO_SCORE=0.70,
        WEAK_BOOTSTRAP_INDEPENDENT_LENGTH_FACTOR=1.5,
    )


def split_graph():
    nodes = np.asarray([[32, 8], [32, 32], [32, 64], [32, 88]], dtype=np.float32)
    edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
    return nodes, edges


def draw_probability(strong=True, weak_value=0.0):
    probability = np.zeros((96, 112), dtype=np.float32)
    if strong:
        probability[31:34, 8:33] = 0.70
        probability[31:34, 64:89] = 0.70
    if weak_value > 0:
        probability[31:34, 32:65] = weak_value
    return probability


def component_count(node_count, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges.tolist())
    return nx.number_connected_components(graph)


class WeakRoadRecoveryTests(unittest.TestCase):
    def test_case_a_strong_road_is_unchanged(self):
        nodes = np.asarray([[32, 8], [32, 88]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        probability = np.zeros((96, 112), dtype=np.float32)
        probability[31:34, 8:89] = 0.75

        result_nodes, result_edges, metadata, summary = graph_extraction.recover_weak_road_edges(
            nodes, edges, probability, recovery_config(), edge_scores=np.asarray([0.9])
        )

        np.testing.assert_array_equal(result_nodes, nodes)
        np.testing.assert_array_equal(result_edges, edges)
        self.assertEqual(metadata[0]["line_source"], "samroad")
        self.assertEqual(summary["weak_recovered_edge_count"], 0)

    def test_case_b_weak_middle_gap_is_recovered(self):
        nodes, edges = split_graph()
        probability = draw_probability(weak_value=0.22)

        result_nodes, result_edges, metadata, summary = graph_extraction.recover_weak_road_edges(
            nodes, edges, probability, recovery_config()
        )

        self.assertEqual(component_count(len(result_nodes), result_edges), 1)
        self.assertGreater(summary["weak_recovered_edge_count"], 0)
        recovered = metadata[len(edges):]
        self.assertTrue(recovered)
        self.assertEqual({row["line_source"] for row in recovered}, {"weak_recovered"})
        self.assertEqual(
            {row["recovery_reason"] for row in recovered},
            {"weak_probability_endpoint_bridge"},
        )

    def test_reciprocal_edges_do_not_hide_true_endpoints(self):
        nodes, _edges = split_graph()
        reciprocal_edges = np.asarray(
            [[0, 1], [1, 0], [2, 3], [3, 2]], dtype=np.int32
        )
        probability = draw_probability(weak_value=0.22)

        result_nodes, result_edges, _metadata, summary = (
            graph_extraction.recover_weak_road_edges(
                nodes, reciprocal_edges, probability, recovery_config()
            )
        )

        self.assertEqual(component_count(len(result_nodes), result_edges), 1)
        self.assertGreater(summary["weak_candidate_count"], 0)
        self.assertGreater(summary["weak_recovered_edge_count"], 0)

    def test_case_c_isolated_weak_noise_is_not_recovered(self):
        nodes = np.asarray([[16, 8], [16, 32]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        probability = np.zeros((96, 112), dtype=np.float32)
        probability[15:18, 8:33] = 0.70
        probability[70:75, 80:90] = 0.22

        result_nodes, result_edges, _metadata, summary = graph_extraction.recover_weak_road_edges(
            nodes, edges, probability, recovery_config()
        )

        np.testing.assert_array_equal(result_nodes, nodes)
        np.testing.assert_array_equal(result_edges, edges)
        self.assertEqual(summary["weak_recovered_edge_count"], 0)

    def test_case_d_near_endpoints_without_evidence_are_not_connected(self):
        nodes, edges = split_graph()
        probability = draw_probability(weak_value=0.0)
        candidate_audit = []

        result_nodes, result_edges, _metadata, summary = graph_extraction.recover_weak_road_edges(
            nodes, edges, probability, recovery_config(), candidate_audit=candidate_audit
        )

        self.assertEqual(component_count(len(result_nodes), result_edges), 2)
        self.assertEqual(summary["weak_recovered_edge_count"], 0)
        self.assertGreater(summary["rejected_weak_candidate_count"], 0)
        self.assertEqual(
            summary["rejected_weak_candidate_count"],
            sum(summary["weak_recovery_reject_reason_counts"].values()),
        )
        self.assertIn("no_astar_path", summary["weak_recovery_reject_reason_counts"])
        self.assertEqual(len(candidate_audit), summary["weak_candidate_count"])

    def test_case_e_weak_centerline_with_strong_surface_is_recovered(self):
        nodes, edges = split_graph()
        probability = draw_probability(weak_value=0.16)
        surface = np.zeros_like(probability)
        surface[29:36, 8:89] = 0.90

        result_nodes, result_edges, metadata, summary = graph_extraction.recover_weak_road_edges(
            nodes,
            edges,
            probability,
            recovery_config(),
            surface_probability=surface,
        )

        self.assertEqual(component_count(len(result_nodes), result_edges), 1)
        recovered = metadata[len(edges):]
        self.assertTrue(recovered)
        self.assertEqual(
            {row["recovery_reason"] for row in recovered},
            {"weak_probability_surface_supported"},
        )
        self.assertGreater(summary["surface_supported_recovery_count"], 0)

    def test_case_f_parallel_road_endpoints_are_not_cross_connected(self):
        nodes = np.asarray(
            [[28, 8], [28, 48], [38, 8], [38, 48]], dtype=np.float32
        )
        edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        probability = np.zeros((80, 80), dtype=np.float32)
        probability[27:30, 8:49] = 0.72
        probability[37:40, 8:49] = 0.72

        result_nodes, result_edges, _metadata, summary = graph_extraction.recover_weak_road_edges(
            nodes, edges, probability, recovery_config()
        )

        np.testing.assert_array_equal(result_nodes, nodes)
        np.testing.assert_array_equal(result_edges, edges)
        self.assertEqual(summary["weak_recovered_edge_count"], 0)

    def test_threshold_profile_can_override_default_thresholds(self):
        config = recovery_config()
        config["ROAD_THRESHOLD_PROFILE"] = "sensor_b"
        config["ROAD_THRESHOLD_PROFILES"] = {
            "sensor_b": {"ROAD_HIGH_THRESHOLD": 0.42, "ROAD_LOW_THRESHOLD": 0.16}
        }
        high, low, profile = graph_extraction.resolve_road_thresholds(config)
        self.assertEqual((high, low, profile), (0.42, 0.16, "sensor_b"))

    def test_scene_diagnosis_uses_fixed_reference_profile(self):
        config = recovery_config()
        config["ROAD_THRESHOLD_PROFILES"] = {
            "default": {"ROAD_HIGH_THRESHOLD": 0.50, "ROAD_LOW_THRESHOLD": 0.18},
            "weak_sensor": {"ROAD_HIGH_THRESHOLD": 0.24, "ROAD_LOW_THRESHOLD": 0.10},
        }
        config["SCENE_DIAGNOSTIC_REFERENCE_PROFILE"] = "default"
        probability = np.full((96, 128), 0.03, dtype=np.float32)
        probability[47:50, 8:121] = 0.22

        config["ROAD_THRESHOLD_PROFILE"] = "default"
        default_diagnosis = graph_extraction.diagnose_scene_confidence(
            probability,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            config,
        )
        config["ROAD_THRESHOLD_PROFILE"] = "weak_sensor"
        weak_diagnosis = graph_extraction.diagnose_scene_confidence(
            probability,
            np.asarray([[48, 8], [48, 120]], dtype=np.float32),
            np.asarray([[0, 1]], dtype=np.int32),
            config,
        )

        self.assertEqual(
            default_diagnosis["scene_confidence_state"],
            weak_diagnosis["scene_confidence_state"],
        )
        self.assertEqual(
            default_diagnosis["recommended_profile"],
            weak_diagnosis["recommended_profile"],
        )
        self.assertEqual(weak_diagnosis["scene_confidence_state"], "low_confidence")
        self.assertEqual(weak_diagnosis["recommended_profile"], "weak_sensor")
        self.assertEqual(weak_diagnosis["active_profile"], "weak_sensor")
        self.assertEqual(weak_diagnosis["diagnostic_reference_profile"], "default")
        self.assertNotEqual(
            default_diagnosis["active_low_threshold"],
            weak_diagnosis["active_low_threshold"],
        )
        self.assertNotEqual(
            default_diagnosis["strong_graph_edge_count"],
            weak_diagnosis["strong_graph_edge_count"],
        )

    def test_weak_sensor_bootstrap_gate_uses_reference_diagnosis(self):
        config = recovery_config()
        config["ROAD_THRESHOLD_PROFILES"] = {
            "default": {"ROAD_HIGH_THRESHOLD": 0.50, "ROAD_LOW_THRESHOLD": 0.18},
            "weak_sensor": {"ROAD_HIGH_THRESHOLD": 0.24, "ROAD_LOW_THRESHOLD": 0.10},
        }
        config["SCENE_DIAGNOSTIC_REFERENCE_PROFILE"] = "default"
        config["ROAD_THRESHOLD_PROFILE"] = "weak_sensor"
        probability = np.full((96, 128), 0.03, dtype=np.float32)
        probability[47:50, 8:121] = 0.22

        _nodes, _edges, _metadata, summary = graph_extraction.postprocess_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            probability,
            config,
        )

        self.assertEqual(summary["scene_confidence_state"], "low_confidence")
        self.assertEqual(summary["recommended_profile"], "weak_sensor")
        self.assertTrue(summary["bootstrap_ran"])

    def test_bootstrap_case_a_normal_strong_scene_is_not_changed(self):
        nodes = np.asarray([[48, 8], [48, 112]], dtype=np.float32)
        edges = np.asarray([[0, 1]], dtype=np.int32)
        probability = np.full((96, 128), 0.03, dtype=np.float32)
        probability[47:50, 8:113] = 0.70

        result_nodes, result_edges, metadata, summary = (
            graph_extraction.postprocess_weak_road_network(
                nodes, edges, probability, recovery_config(), edge_scores=np.asarray([0.9])
            )
        )

        np.testing.assert_array_equal(result_nodes, nodes)
        np.testing.assert_array_equal(result_edges, edges)
        self.assertEqual(summary["scene_confidence_state"], "normal")
        self.assertFalse(summary["bootstrap_ran"])
        self.assertEqual(summary["bootstrap_recovered_edge_count"], 0)
        self.assertEqual(metadata[0]["line_source"], "samroad")
        self.assertEqual(
            set(summary["timing"]),
            {
                "diagnosis_seconds", "relative_context_seconds", "bootstrap_seconds",
                "weak_endpoint_recovery_seconds", "endpoint_to_segment_recovery_seconds",
                "connectivity_statistics_seconds", "total_seconds",
            },
        )
        self.assertTrue(all(value >= 0.0 for value in summary["timing"].values()))

    def test_bootstrap_case_b_low_confidence_scene_recovers_long_weak_road(self):
        config = recovery_config()
        probability = np.full((96, 128), 0.03, dtype=np.float32)
        probability[47:50, 8:121] = 0.22

        nodes, edges, metadata, summary = graph_extraction.postprocess_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            probability,
            config,
        )

        self.assertEqual(summary["scene_confidence_state"], "low_confidence")
        self.assertEqual(summary["recommended_profile"], "weak_sensor")
        self.assertTrue(summary["bootstrap_ran"])
        self.assertGreater(summary["bootstrap_recovered_edge_count"], 0)
        self.assertGreater(len(nodes), 0)
        self.assertGreater(len(edges), 0)
        self.assertEqual({row["line_source"] for row in metadata}, {"weak_bootstrap"})

    def test_bootstrap_case_c_low_contrast_chain_is_rejected(self):
        config = recovery_config()
        config["ROAD_LOW_THRESHOLD"] = 0.15
        probability = np.full((96, 128), 0.14, dtype=np.float32)
        probability[47:50, 8:121] = 0.16

        _nodes, edges, _metadata, summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            probability,
            config,
        )

        self.assertEqual(len(edges), 0)
        self.assertEqual(summary["bootstrap_recovered_edge_count"], 0)
        self.assertGreater(summary["bootstrap_rejected_count"], 0)

    def test_bootstrap_case_d_short_isolated_weak_line_is_rejected(self):
        probability = np.full((96, 128), 0.03, dtype=np.float32)
        probability[47:50, 8:28] = 0.24
        candidate_audit = []

        _nodes, edges, _metadata, summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            probability,
            recovery_config(),
            candidate_audit=candidate_audit,
        )

        self.assertEqual(len(edges), 0)
        self.assertGreater(summary["bootstrap_candidate_count"], 0)
        self.assertEqual(
            summary["bootstrap_rejected_count"], summary["bootstrap_candidate_count"]
        )
        self.assertEqual(
            summary["bootstrap_rejected_count"],
            sum(summary["bootstrap_reject_reason_counts"].values()),
        )
        self.assertIn("too_short", summary["bootstrap_reject_reason_counts"])
        self.assertEqual(len(candidate_audit), summary["bootstrap_candidate_count"])

    def test_bootstrap_case_e_long_relative_response_is_accepted(self):
        config = recovery_config()
        config["ROAD_LOW_THRESHOLD"] = 0.10
        probability = np.full((112, 160), 0.04, dtype=np.float32)
        values = np.linspace(0.18, 0.24, 140, dtype=np.float32)
        probability[55:58, 10:150] = values[np.newaxis, :]

        _nodes, edges, metadata, summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            probability,
            config,
        )

        self.assertGreater(summary["bootstrap_recovered_edge_count"], 0)
        self.assertGreater(len(edges), 0)
        self.assertTrue(all(row["probability_contrast"] > 0.08 for row in metadata))

    def test_bootstrap_case_f_strong_road_gap_stays_weak_recovered(self):
        nodes, edges = split_graph()
        probability = draw_probability(weak_value=0.22)

        _nodes, _edges, metadata, summary = graph_extraction.postprocess_weak_road_network(
            nodes, edges, probability, recovery_config()
        )

        self.assertEqual(summary["bootstrap_recovered_edge_count"], 0)
        self.assertGreater(summary["weak_recovered_edge_count"], 0)
        added_sources = {row["line_source"] for row in metadata[len(edges):]}
        self.assertEqual(added_sources, {"weak_recovered"})

    def test_bootstrap_optional_surface_support_can_confirm_low_absolute_response(self):
        config = recovery_config()
        config["ROAD_LOW_THRESHOLD"] = 0.10
        probability = np.full((96, 128), 0.03, dtype=np.float32)
        probability[47:50, 8:121] = 0.11
        surface = np.zeros_like(probability)
        surface[44:53, 8:121] = 0.90

        _nodes, edges, metadata, summary = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            probability,
            config,
            surface_probability=surface,
        )

        self.assertGreater(len(edges), 0)
        self.assertGreater(summary["bootstrap_recovered_edge_count"], 0)
        self.assertTrue(all(row["surface_conf"] > 0.8 for row in metadata))


if __name__ == "__main__":
    unittest.main()
