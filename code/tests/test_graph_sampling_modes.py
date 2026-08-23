from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SAMROAD = Path(__file__).resolve().parents[1] / "engine" / "samroad"
if str(SAMROAD) not in sys.path:
    sys.path.insert(0, str(SAMROAD))

import graph_extraction  # noqa: E402


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def config(mode: str) -> Config:
    return Config(
        ITSC_THRESHOLD=0.2,
        ROAD_THRESHOLD=0.2,
        ITSC_NMS_RADIUS=8,
        ROAD_NMS_RADIUS=16,
        ROAD_NMS_MIN_SEPARATION=4.0,
        ROAD_TANGENT_RADIUS=5.0,
        PARALLEL_BRANCH_COSINE=0.9,
        PARALLEL_BRANCH_LATERAL_COSINE=0.55,
        JUNCTION_POINT_MERGE_RADIUS=4.0,
        JUNCTION_SPARSE_RADIUS=20.0,
        JUNCTION_NODE_MODE=mode,
    )


class GraphSamplingModeTests(unittest.TestCase):
    def test_sparse_mode_removes_regular_samples_around_junction_core(self) -> None:
        keypoints = np.zeros((65, 65), dtype=np.uint8)
        roads = np.zeros_like(keypoints)
        keypoints[32, 32] = 255
        roads[32, 4:61] = 255
        roads[4:61, 32] = 255

        sparse = graph_extraction.extract_graph_points(keypoints, roads, config("sparse"))
        dense = graph_extraction.extract_graph_points(keypoints, roads, config("dense_legacy"))
        center = np.asarray([32.0, 32.0])
        sparse_near = int(np.count_nonzero(np.linalg.norm(sparse - center, axis=1) <= 20.0))
        dense_near = int(np.count_nonzero(np.linalg.norm(dense - center, axis=1) <= 20.0))

        self.assertLess(sparse_near, dense_near)
        self.assertLess(sparse.shape[0], dense.shape[0])

    def test_regular_road_keeps_coarse_spacing_without_junctions(self) -> None:
        keypoints = np.zeros((33, 97), dtype=np.uint8)
        roads = np.zeros_like(keypoints)
        roads[16, 4:93] = 255
        points = graph_extraction.extract_graph_points(keypoints, roads, config("sparse"))
        ordered = points[np.argsort(points[:, 0])]
        distances = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
        self.assertTrue(np.all(distances >= 12.0), distances)

    def test_invalid_mode_is_rejected(self) -> None:
        keypoints = np.zeros((33, 33), dtype=np.uint8)
        roads = np.zeros_like(keypoints)
        keypoints[16, 16] = 255
        roads[16, 4:29] = 255
        with self.assertRaisesRegex(ValueError, "JUNCTION_NODE_MODE"):
            graph_extraction.extract_graph_points(keypoints, roads, config("invalid"))


if __name__ == "__main__":
    unittest.main()
