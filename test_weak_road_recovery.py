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

        result_nodes, result_edges, _metadata, summary = graph_extraction.recover_weak_road_edges(
            nodes, edges, probability, recovery_config()
        )

        self.assertEqual(component_count(len(result_nodes), result_edges), 2)
        self.assertEqual(summary["weak_recovered_edge_count"], 0)
        self.assertGreater(summary["rejected_weak_candidate_count"], 0)

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


if __name__ == "__main__":
    unittest.main()
