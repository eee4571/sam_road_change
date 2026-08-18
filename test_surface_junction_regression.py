from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ENGINE = Path(__file__).resolve().parent / "engine" / "width"
sys.path.insert(0, str(ENGINE))

from width_surface_reconstruction import (
    WidthSurfaceConfig,
    _fill_parallel_road_gaps,
    reconstruct_surface_from_widths,
)


class SurfaceJunctionRegressionTests(unittest.TestCase):
    def test_continuity_gap_inherits_the_attached_road_width(self) -> None:
        shape = (100, 180)
        nodes = np.asarray(
            [[50.0, 10.0], [50.0, 60.0], [50.0, 90.0], [50.0, 120.0], [50.0, 170.0]],
            dtype=np.float32,
        )
        edges = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
        widths = [
            {"edge_id": 0, "width_px": 12.0, "quality_grade": "A", "source": "remeasured"},
            {"edge_id": 1, "width_px": 3.0, "quality_grade": "A", "source": "remeasured"},
            {"edge_id": 2, "width_px": 3.0, "quality_grade": "A", "source": "remeasured"},
            {"edge_id": 3, "width_px": 12.0, "quality_grade": "A", "source": "remeasured"},
        ]
        metadata = [
            {"source": "samroad", "line_feature_id": "base:left"},
            {"source": "auto_added_gap", "line_feature_id": "candidate:gap-1"},
            {"source": "auto_added_gap", "line_feature_id": "candidate:gap-1"},
            {"source": "samroad", "line_feature_id": "base:right"},
        ]
        result = reconstruct_surface_from_widths(
            shape, nodes, edges, widths, [],
            config=WidthSurfaceConfig(regular_surface=True, close_kernel=1, boundary_smooth_sigma_px=0.0),
            edge_metadata=metadata,
        )
        self.assertEqual([12.0] * 4, result.metadata["resolved_widths_px"])
        self.assertEqual(2, result.metadata["supplemental_path_inherited_edge_count"])
        self.assertGreaterEqual(int(np.count_nonzero(result.surface[:, 85])), 12)

    def test_large_width_disagreement_does_not_split_a_straight_corridor(self) -> None:
        nodes = np.asarray([[50.0, 10.0], [50.0, 60.0], [50.0, 110.0]], dtype=np.float32)
        result = reconstruct_surface_from_widths(
            (120, 140), nodes, np.asarray([[0, 1], [1, 2]], dtype=np.int32),
            [
                {"edge_id": 0, "width_px": 8.0, "quality_grade": "A", "source": "remeasured"},
                {"edge_id": 1, "width_px": 24.0, "quality_grade": "A", "source": "remeasured"},
            ],
            [],
            config=WidthSurfaceConfig(regular_surface=True, close_kernel=1, boundary_smooth_sigma_px=0.0),
        )
        self.assertEqual(result.metadata["resolved_widths_px"][0], result.metadata["resolved_widths_px"][1])

    def test_bent_supplemental_path_keeps_one_width(self) -> None:
        nodes = np.asarray([[20.0, 20.0], [40.0, 60.0], [80.0, 80.0]], dtype=np.float32)
        result = reconstruct_surface_from_widths(
            (110, 110), nodes, np.asarray([[0, 1], [1, 2]], dtype=np.int32),
            [
                {"edge_id": 0, "width_px": 6.0, "quality_grade": "A", "source": "remeasured"},
                {"edge_id": 1, "width_px": 14.0, "quality_grade": "A", "source": "remeasured"},
            ],
            [],
            config=WidthSurfaceConfig(regular_surface=True, close_kernel=1, boundary_smooth_sigma_px=0.0),
            edge_metadata=[
                {"source": "auto_added_surface", "line_feature_id": "candidate:bent"},
                {"source": "auto_added_surface", "line_feature_id": "candidate:bent"},
            ],
        )
        self.assertEqual(result.metadata["resolved_widths_px"][0], result.metadata["resolved_widths_px"][1])

    def test_parallel_gap_fill_is_suppressed_inside_a_junction(self) -> None:
        nodes = np.asarray(
            [[20.0, 40.0], [60.0, 40.0], [100.0, 40.0], [20.0, 70.0], [100.0, 70.0], [60.0, 100.0]],
            dtype=np.float32,
        )
        edges = np.asarray([[0, 1], [1, 2], [3, 4], [1, 5]], dtype=np.int32)
        canvas = np.zeros((130, 130), dtype=np.uint8)
        _fill_parallel_road_gaps(
            canvas, nodes, edges, np.full(len(edges), 10.0, dtype=np.float32),
            WidthSurfaceConfig(parallel_min_overlap_px=12.0),
        )
        self.assertEqual(0, int(canvas[60, 55]))

    def test_grade_c_short_degree2_edge_cannot_turn_junction_into_a_disk(self) -> None:
        """A width sampled wholly inside a junction must not be rendered as a wide stub.

        The short eastbound edge is a dense degree-2 split immediately adjacent
        to a T junction.  Its grade-C width is the diameter of the source
        junction disk (78 px), whereas all evidence-backed branches are 12 px.
        A final user-facing surface may cover the centerline, but it must not
        create an 78 px-wide block around that split.
        """
        shape = (180, 180)
        nodes = np.asarray(
            [
                [90.0, 30.0],   # west
                [90.0, 90.0],   # T junction
                [90.0, 98.0],   # dense degree-2 split inside junction
                [90.0, 150.0],  # east
                [30.0, 90.0],   # north
            ],
            dtype=np.float32,
        )
        edges = np.asarray([[0, 1], [1, 2], [2, 3], [1, 4]], dtype=np.int32)
        widths = [
            {"edge_id": 0, "width_px": 78.0, "quality_grade": "C", "source": "unresolved_after_final_measurement"},
            {"edge_id": 1, "width_px": 78.0, "quality_grade": "C", "source": "unresolved_after_final_measurement"},
            {"edge_id": 2, "width_px": 12.0, "quality_grade": "A", "source": "remeasured"},
            {"edge_id": 3, "width_px": 78.0, "quality_grade": "C", "source": "unresolved_after_final_measurement"},
        ]
        reference = np.zeros(shape, dtype=np.uint8)
        for start, end in ((nodes[0], nodes[1]), (nodes[2], nodes[3]), (nodes[1], nodes[4])):
            cv2.line(reference, tuple(np.rint(start[::-1]).astype(int)), tuple(np.rint(end[::-1]).astype(int)), 1, 12)
        cv2.circle(reference, (90, 90), 39, 1, -1)

        result = reconstruct_surface_from_widths(
            shape, nodes, edges, widths, [], reference_surface=reference,
            config=WidthSurfaceConfig(regular_surface=True, boundary_smooth_sigma_px=0.0),
        )

        # A 32x32 window around the degree-2 split should retain branch-scale
        # area, not the nearly solid 78 px disk/block produced by the old code.
        local = result.surface[74:106, 82:114]
        self.assertLess(int(np.count_nonzero(local)), 650)
        self.assertEqual(0, result.metadata["uncovered_centerline_px"])


if __name__ == "__main__":
    unittest.main()
