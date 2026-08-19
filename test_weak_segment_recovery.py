from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SAMROAD_ROOT = REPO_ROOT / "engine" / "samroad"
DEV_TOOL_ROOT = REPO_ROOT / "dev_tools" / "weak_road_recovery_test"
for import_root in (SAMROAD_ROOT, DEV_TOOL_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import graph_extraction  # noqa: E402
import run_test  # noqa: E402


def test_config(**overrides):
    config = {
        "ROAD_THRESHOLD_PROFILE": "weak_sensor",
        "ROAD_THRESHOLD_PROFILES": {
            "weak_sensor": {"ROAD_HIGH_THRESHOLD": 0.24, "ROAD_LOW_THRESHOLD": 0.10},
        },
        "WEAK_SEGMENT_RECOVERY_ENABLED": True,
        "WEAK_SEGMENT_RECOVERY_MAX_DISTANCE_PX": 64.0,
        "WEAK_SEGMENT_RECOVERY_MIN_DIRECTION_COSINE": 0.50,
        "WEAK_ENDPOINT_DIRECTION_LOOKBACK_PX": 32.0,
        "WEAK_RECOVERY_MAX_PATH_RATIO": 1.35,
        "WEAK_RECOVERY_MIN_MEAN_PROBABILITY": 0.16,
        "WEAK_RECOVERY_MIN_Q25_PROBABILITY": 0.12,
        "WEAK_RECOVERY_MIN_WEAK_FRACTION": 0.80,
        "WEAK_RECOVERY_MIN_BACKGROUND_CONTRAST": 0.05,
        "WEAK_RECOVERY_BACKGROUND_OFFSET_PX": 4.0,
        "WEAK_RECOVERY_PATH_MARGIN_PX": 16.0,
        "WEAK_RECOVERY_SAMPLE_STEP_PX": 12.0,
        "WEAK_RECOVERY_AUTO_SCORE": 0.62,
    }
    config.update(overrides)
    return config


def t_graph(*, reciprocal=False):
    # Horizontal target A-B and a separate vertical branch whose upper endpoint P
    # should connect to the midpoint Q=(20, 30).
    nodes = np.asarray([
        [20.0, 10.0],  # A
        [20.0, 50.0],  # B
        [55.0, 30.0],  # branch interior
        [35.0, 30.0],  # dangling endpoint P
    ], dtype=np.float32)
    edges = [(0, 1), (2, 3)]
    if reciprocal:
        edges += [(1, 0), (3, 2)]
    return nodes, np.asarray(edges, dtype=np.int32)


def supported_probability(nodes, edges):
    probability = np.zeros((80, 80), dtype=np.float32)
    for src_idx, dst_idx in edges.tolist():
        src, dst = nodes[src_idx], nodes[dst_idx]
        cv2.line(
            probability,
            (int(dst[1]), int(dst[0])),
            (int(src[1]), int(src[0])),
            0.80,
            3,
            cv2.LINE_8,
        )
    cv2.line(probability, (30, 35), (30, 20), 0.80, 3, cv2.LINE_8)
    return probability


class WeakSegmentRecoveryTests(unittest.TestCase):
    def run_recovery(self, nodes, edges, probability, **config_overrides):
        audit = []
        result = graph_extraction.recover_endpoint_to_segment_connections(
            nodes,
            edges,
            probability,
            test_config(**config_overrides),
            candidate_audit=audit,
        )
        return (*result, audit)

    def test_1_continuous_probability_merges_components(self):
        nodes, edges = t_graph()
        probability = supported_probability(nodes, edges)
        final_nodes, final_edges, metadata, summary, audit = self.run_recovery(
            nodes, edges, probability
        )
        self.assertEqual(summary["endpoint_segment_accepted_count"], 1)
        self.assertEqual(summary["endpoint_segment_connectivity_gain"], 1)
        self.assertEqual(
            graph_extraction.graph_connectivity_stats(nodes, edges)["component_count"] - 1,
            graph_extraction.graph_connectivity_stats(final_nodes, final_edges)["component_count"],
        )
        self.assertTrue(any(row["accepted"] for row in audit))
        self.assertEqual(len(metadata), len(final_edges))

    def test_2_low_probability_path_is_rejected(self):
        nodes, edges = t_graph()
        probability = np.zeros((80, 80), dtype=np.float32)
        _nodes, _edges, _metadata, summary, audit = self.run_recovery(
            nodes, edges, probability
        )
        self.assertEqual(summary["endpoint_segment_accepted_count"], 0)
        self.assertTrue(any(row["reject_reason"] == "no_astar_path" for row in audit))

    def test_3_direction_mismatch_is_rejected(self):
        nodes = np.asarray([
            [20.0, 20.0], [20.0, 50.0],
            [35.0, 10.0], [35.0, 30.0],
        ], dtype=np.float32)
        edges = np.asarray([(0, 1), (2, 3)], dtype=np.int32)
        probability = supported_probability(nodes, edges)
        _nodes, _edges, _metadata, summary, audit = self.run_recovery(
            nodes, edges, probability
        )
        self.assertEqual(summary["endpoint_segment_accepted_count"], 0)
        self.assertTrue(any(row["reject_reason"] == "direction_mismatch" for row in audit))

    def test_4_same_component_is_rejected(self):
        nodes, edges = t_graph()
        edges = np.vstack([edges, np.asarray([[1, 2]], dtype=np.int32)])
        probability = supported_probability(nodes, edges)
        _nodes, _edges, _metadata, summary, audit = self.run_recovery(
            nodes, edges, probability
        )
        self.assertEqual(summary["endpoint_segment_accepted_count"], 0)
        self.assertTrue(any(row["reject_reason"] == "reject_same_component" for row in audit))

    def test_5_target_segment_is_split_at_projection(self):
        nodes, edges = t_graph()
        probability = supported_probability(nodes, edges)
        final_nodes, final_edges, metadata, summary, _audit = self.run_recovery(
            nodes, edges, probability
        )
        self.assertEqual(summary["endpoint_segment_accepted_count"], 1)
        junction_ids = np.where(np.all(np.isclose(final_nodes, [20.0, 30.0]), axis=1))[0]
        self.assertEqual(len(junction_ids), 1)
        junction_idx = int(junction_ids[0])
        undirected = {tuple(sorted(map(int, edge))) for edge in final_edges.tolist()}
        self.assertNotIn((0, 1), undirected)
        self.assertIn(tuple(sorted((0, junction_idx))), undirected)
        self.assertIn(tuple(sorted((1, junction_idx))), undirected)
        self.assertGreaterEqual(len(graph_extraction._undirected_adjacency(len(final_nodes), final_edges)[junction_idx]), 3)
        self.assertEqual(len(metadata), len(final_edges))

    def test_6_reciprocal_split_has_no_duplicates_or_metadata_drift(self):
        nodes, edges = t_graph(reciprocal=True)
        probability = supported_probability(nodes, edges)
        final_nodes, final_edges, metadata, summary, _audit = self.run_recovery(
            nodes, edges, probability
        )
        self.assertEqual(summary["endpoint_segment_accepted_count"], 1)
        directed_edges = [tuple(map(int, edge)) for edge in final_edges.tolist()]
        self.assertEqual(len(directed_edges), len(set(directed_edges)))
        self.assertEqual(len(metadata), len(directed_edges))
        junction_idx = int(np.where(np.all(np.isclose(final_nodes, [20.0, 30.0]), axis=1))[0][0])
        for edge in ((0, junction_idx), (junction_idx, 1), (1, junction_idx), (junction_idx, 0)):
            self.assertIn(edge, directed_edges)

    def test_7_accepted_candidate_visualization_and_montage_are_written(self):
        nodes, edges = t_graph()
        probability = supported_probability(nodes, edges)
        final_nodes, final_edges, metadata, summary, audit = self.run_recovery(
            nodes, edges, probability
        )
        self.assertEqual(summary["endpoint_segment_accepted_count"], 1)
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            run_test.write_endpoint_segment_candidate_visualizations(
                run_dir, image, final_nodes, final_edges, metadata, audit
            )
            candidate_dir = run_dir / "endpoint_segment_candidates"
            self.assertEqual(len(list(candidate_dir.glob("segment_*.png"))), 1)
            self.assertTrue(
                (candidate_dir / "endpoint_segment_candidates_montage.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
