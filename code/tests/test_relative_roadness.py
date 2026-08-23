from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


SAMROAD = Path(__file__).resolve().parents[1] / "engine" / "samroad"
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
        skeleton = self._ladder_skeleton()
        result = self._normalize_synthetic(skeleton, cfg)
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["raw_chain_count"], 38)
        self.assertEqual(diagnostics["raw_short_chain_count"], 38)
        self.assertTrue(np.array_equal(result["normalized_skeleton"] > 0, skeleton > 0))
        self.assertEqual(diagnostics["collapsed_zone_count"], 0)
        self.assertGreaterEqual(diagnostics["complex_zone_skipped_collapse_count"], 1)
        grouping = graph_extraction.build_relative_chain_corridors(
            skeleton,
            candidate_mask=cv2.dilate(skeleton, np.ones((3, 3), dtype=np.uint8)),
        )
        self.assertGreater(max(row["total_length"] for row in grouping["corridors"]), 200.0)
        zone = diagnostics["junction_zones"][0]
        self.assertEqual(len(zone["branch_lengths"]), zone["incident_branch_count"])
        self.assertEqual(len(zone["branch_tangents"]), zone["incident_branch_count"])

    def test_dense_ladder_keeps_real_branch_that_leaves_zone(self):
        cfg = relative_config()
        cfg["RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX"] = 48.0
        skeleton = self._ladder_skeleton(with_branch=True)
        result = self._normalize_synthetic(skeleton, cfg)
        normalized = result["normalized_skeleton"] > 0
        self.assertGreater(np.count_nonzero(normalized[65:101, 118:123]), 30)
        self.assertTrue(np.array_equal(normalized, skeleton > 0))
        self.assertEqual(result["diagnostics"]["collapsed_zone_count"], 0)

    def test_logical_corridor_rescues_split_sub48_chains(self):
        skeleton = np.zeros((96, 160), dtype=np.uint8)
        cv2.line(skeleton, (15, 48), (145, 48), 1, 1)
        for col in (41, 67, 93, 119):
            cv2.line(skeleton, (col, 48), (col, 43), 1, 1)
        cfg = relative_config()
        cfg["RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX"] = 48.0
        evidence = skeleton.astype(np.float32)
        retained, _rejected, summary = graph_extraction.extract_relative_skeleton(
            skeleton,
            cfg,
            input_skeleton=skeleton,
            relative_score=evidence,
            scene_rank=evidence,
            scale_agreement=evidence,
            relative_weak_threshold=0.5,
        )
        horizontal_records = [
            row for row in summary["relative_micro_chains"]
            if abs(row["start"][0] - row["end"][0]) <= 1
        ]
        self.assertTrue(horizontal_records)
        self.assertTrue(all(row["micro_chain_length"] < 48.0 for row in horizontal_records))
        self.assertTrue(all(row["rescued_by_corridor"] for row in horizontal_records))
        self.assertGreaterEqual(summary["corridor_rescued_chain_count"], len(horizontal_records))
        self.assertGreater(np.count_nonzero(retained[47:50, 15:146]), 120)

    def test_logical_corridor_preserves_t_three_branches(self):
        skeleton = np.zeros((180, 180), dtype=np.uint8)
        cv2.line(skeleton, (15, 80), (165, 80), 1, 1)
        cv2.line(skeleton, (90, 80), (90, 155), 1, 1)
        cfg = relative_config()
        cfg["RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX"] = 48.0
        evidence = skeleton.astype(np.float32)
        retained, _rejected, summary = graph_extraction.extract_relative_skeleton(
            skeleton,
            cfg,
            input_skeleton=skeleton,
            relative_score=evidence,
            scene_rank=evidence,
            scale_agreement=evidence,
            relative_weak_threshold=0.5,
        )
        self.assertGreater(np.count_nonzero(retained[78:83, 15:166]), 140)
        self.assertGreater(np.count_nonzero(retained[80:156, 88:93]), 65)
        self.assertEqual(summary["corridor_count"], 2)

    def test_logical_corridor_keeps_parallel_roads_separate(self):
        skeleton = np.zeros((100, 180), dtype=np.uint8)
        cv2.line(skeleton, (15, 35), (165, 35), 1, 1)
        cv2.line(skeleton, (15, 65), (165, 65), 1, 1)
        grouping = graph_extraction.build_relative_chain_corridors(
            skeleton, candidate_mask=skeleton
        )
        self.assertEqual(len(grouping["corridors"]), 2)
        self.assertEqual({row["chain_count"] for row in grouping["corridors"]}, {1})
        self.assertNotEqual(
            grouping["corridor_labels"][35, 90],
            grouping["corridor_labels"][65, 90],
        )

    def test_logical_corridor_crossing_has_no_triangle_or_shortcut(self):
        skeleton = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(skeleton, (10, 100), (190, 100), 1, 1)
        cv2.line(skeleton, (100, 10), (100, 190), 1, 1)
        cfg = relative_config()
        cfg["RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX"] = 48.0
        cfg["WEAK_BOOTSTRAP_MIN_LENGTH_PX"] = 48.0
        evidence = skeleton.astype(np.float32)
        retained, _rejected, summary = graph_extraction.extract_relative_skeleton(
            skeleton,
            cfg,
            input_skeleton=skeleton,
            relative_score=evidence,
            scene_rank=evidence,
            scale_agreement=evidence,
            relative_weak_threshold=0.5,
        )
        chain_labels = summary.pop("_relative_chain_labels")
        corridor_labels = summary.pop("_relative_corridor_labels")
        self.assertEqual(summary["corridor_count"], 2)
        context = {
            "relative_score": evidence,
            "scene_rank": evidence,
            "local_background": np.zeros_like(evidence),
            "local_contrast": evidence,
            "normalized_contrast": evidence,
            "scale_agreement_fraction": evidence,
            "relative_skeleton": retained,
            "relative_only_skeleton": retained,
            "relative_chain_labels": chain_labels,
            "relative_corridor_labels": corridor_labels,
            "diagnostics": {"relative_weak_threshold": 0.5, **summary},
        }
        nodes, edges, _metadata, _bootstrap = graph_extraction.bootstrap_weak_road_network(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            np.full(skeleton.shape, 0.07, dtype=np.float32),
            cfg,
            relative_context=context,
            include_absolute_candidates=False,
        )
        self.assertGreater(len(edges), 0)
        for src_idx, dst_idx in edges:
            delta = np.abs(nodes[int(dst_idx)] - nodes[int(src_idx)])
            self.assertTrue(delta[0] < 1e-6 or delta[1] < 1e-6)
        adjacency = {index: set() for index in range(len(nodes))}
        for src_idx, dst_idx in edges:
            adjacency[int(src_idx)].add(int(dst_idx))
            adjacency[int(dst_idx)].add(int(src_idx))
        triangle_count = sum(
            len(adjacency[first] & adjacency[second])
            for first, neighbors in adjacency.items()
            for second in neighbors if first < second
        ) // 3
        self.assertEqual(triangle_count, 0)

    def test_relative_ridge_wide_straight_line_rejects_edge_fishbones(self):
        rows, cols = np.indices((96, 176), dtype=np.float32)
        score = np.zeros((96, 176), dtype=np.float32)
        score[:, 12:164] = np.exp(-0.5 * ((rows[:, 12:164] - 48.0) / 3.0) ** 2)
        candidate = np.zeros_like(score, dtype=np.uint8)
        candidate[41:56, 12:164] = 1
        for col in range(28, 157, 16):
            candidate[34:42, col - 1:col + 2] = 1
        result = graph_extraction.extract_relative_ridge_centerline(score, candidate)
        ridge = result["relative_ridge_mask"] > 0
        diagnostics = result["diagnostics"]
        self.assertGreater(np.count_nonzero(ridge[47:50, 12:164]), 135)
        self.assertLess(np.count_nonzero(ridge[:44]), 8)
        self.assertLess(
            diagnostics["ridge_junction_pixel_count"],
            diagnostics["old_junction_pixel_count"],
        )

    def test_relative_ridge_smooth_curve_keeps_ramp_geometry(self):
        rows, cols = np.indices((112, 112), dtype=np.float32)
        centerline = 30.0 + 0.0065 * (cols - 56.0) ** 2
        active = (cols >= 10) & (cols <= 102)
        score = np.exp(-0.5 * ((rows - centerline) / 2.8) ** 2) * active
        candidate = ((np.abs(rows - centerline) <= 6.0) & active).astype(np.uint8)
        ridge = graph_extraction.extract_relative_ridge_centerline(
            score.astype(np.float32), candidate
        )["relative_ridge_mask"] > 0
        points = np.column_stack(np.where(ridge))
        expected_rows = 30.0 + 0.0065 * (points[:, 1] - 56.0) ** 2
        self.assertGreater(len(np.unique(points[:, 1])), 80)
        self.assertLess(float(np.median(np.abs(points[:, 0] - expected_rows))), 1.5)
        middle_row = float(np.median(points[np.abs(points[:, 1] - 56) <= 2, 0]))
        end_rows = points[(points[:, 1] <= 14) | (points[:, 1] >= 98), 0]
        self.assertGreater(float(np.median(end_rows)) - middle_row, 8.0)

    def test_relative_ridge_crossing_keeps_axes_without_diagonal_shortcut(self):
        rows, cols = np.indices((112, 112), dtype=np.float32)
        horizontal = np.exp(-0.5 * ((rows - 56.0) / 2.5) ** 2)
        vertical = np.exp(-0.5 * ((cols - 56.0) / 2.5) ** 2)
        active = (rows >= 10) & (rows <= 102) & (cols >= 10) & (cols <= 102)
        score = np.maximum(horizontal, vertical) * active
        candidate = (
            active & ((np.abs(rows - 56.0) <= 6.0) | (np.abs(cols - 56.0) <= 6.0))
        ).astype(np.uint8)
        ridge = graph_extraction.extract_relative_ridge_centerline(
            score.astype(np.float32), candidate
        )["relative_ridge_mask"] > 0
        self.assertGreater(np.count_nonzero(ridge[55:58, 10:103]), 80)
        self.assertGreater(np.count_nonzero(ridge[10:103, 55:58]), 80)
        off_axis = ridge & (np.abs(rows - 56.0) > 2) & (np.abs(cols - 56.0) > 2)
        self.assertEqual(np.count_nonzero(off_axis), 0)

    def test_relative_backbone_keeps_main_ridge_and_rejects_fishbone_spurs(self):
        binary = np.zeros((96, 180), dtype=np.uint8)
        cv2.line(binary, (15, 48), (165, 48), 1, 1)
        for col in range(30, 166, 18):
            cv2.line(binary, (col, 38), (col, 58), 1, 1)
        ridge = np.zeros_like(binary)
        cv2.line(ridge, (15, 48), (165, 48), 1, 1)
        evidence = binary.astype(np.float32)
        support = graph_extraction.build_relative_support_graph(
            binary,
            relative_score=evidence,
            scene_rank=evidence,
            ridge_mask=ridge,
            ridge_strength=ridge.astype(np.float32),
            ridge_orientation=np.zeros(binary.shape, dtype=np.float32),
            scale_agreement=evidence,
            candidate_mask=binary,
        )
        result = graph_extraction.trace_relative_backbone(
            support, min_chain_length=48.0, relative_weak_threshold=0.5
        )
        backbone = result["relative_backbone_mask"] > 0
        self.assertGreater(np.count_nonzero(backbone[47:50, 15:166]), 145)
        self.assertEqual(np.count_nonzero(backbone[:45]), 0)
        self.assertEqual(np.count_nonzero(backbone[52:]), 0)
        self.assertGreater(result["diagnostics"]["spur_rejected_count"], 0)

    def test_relative_backbone_bridges_a_ridge_gap_on_binary_support(self):
        binary = np.zeros((96, 180), dtype=np.uint8)
        cv2.line(binary, (15, 48), (165, 48), 1, 1)
        ridge = np.zeros_like(binary)
        cv2.line(ridge, (15, 48), (62, 48), 1, 1)
        cv2.line(ridge, (101, 48), (165, 48), 1, 1)
        evidence = binary.astype(np.float32)
        support = graph_extraction.build_relative_support_graph(
            binary,
            relative_score=evidence,
            scene_rank=evidence,
            ridge_mask=ridge,
            ridge_strength=ridge.astype(np.float32),
            ridge_orientation=np.zeros(binary.shape, dtype=np.float32),
            scale_agreement=evidence,
            candidate_mask=binary,
        )
        result = graph_extraction.trace_relative_backbone(
            support, min_chain_length=48.0, relative_weak_threshold=0.5
        )
        backbone = result["relative_backbone_mask"] > 0
        sources = result["relative_backbone_source_labels"]
        self.assertGreater(np.count_nonzero(backbone[47:50, 15:166]), 145)
        self.assertGreater(np.count_nonzero(sources[48, 65:99] == 2), 25)
        self.assertGreater(
            result["diagnostics"]["ridge_to_ridge_bridge_length"], 25.0
        )

    def test_relative_backbone_preserves_supported_t_junction_branch(self):
        binary = np.zeros((160, 180), dtype=np.uint8)
        cv2.line(binary, (15, 62), (165, 62), 1, 1)
        cv2.line(binary, (90, 62), (90, 145), 1, 1)
        ridge = binary.copy()
        evidence = binary.astype(np.float32)
        orientation = np.zeros(binary.shape, dtype=np.float32)
        orientation[62:146, 88:93] = np.float32(np.pi / 2.0)
        support = graph_extraction.build_relative_support_graph(
            binary,
            relative_score=evidence,
            scene_rank=evidence,
            ridge_mask=ridge,
            ridge_strength=ridge.astype(np.float32),
            ridge_orientation=orientation,
            scale_agreement=evidence,
            candidate_mask=binary,
        )
        result = graph_extraction.trace_relative_backbone(
            support, min_chain_length=48.0, relative_weak_threshold=0.5
        )
        backbone = result["relative_backbone_mask"] > 0
        self.assertGreater(np.count_nonzero(backbone[60:65, 15:166]), 145)
        self.assertGreater(np.count_nonzero(backbone[62:146, 88:93]), 75)
        self.assertEqual(result["diagnostics"]["spur_rejected_count"], 0)

    def test_relative_ribbon_varying_width_has_one_continuous_centerline(self):
        height, width = 120, 210
        candidate = np.zeros((height, width), dtype=np.uint8)
        for col in range(10, 200):
            half_width = (4, 8, 5, 10)[min(3, (col - 10) // 48)]
            candidate[60 - half_width:61 + half_width, col] = 1
        score = candidate.astype(np.float32) * 0.6
        result = graph_extraction.extract_relative_ribbon_centerline(
            score, candidate, np.zeros_like(score)
        )
        centerline = result["ribbon_centerline_mask"] > 0
        self.assertGreater(np.count_nonzero(centerline[58:63, 10:200]), 180)
        self.assertEqual(np.count_nonzero(centerline[:55]), 0)
        self.assertEqual(np.count_nonzero(centerline[66:]), 0)
        self.assertEqual(result["diagnostics"]["ribbon_junction_count"], 0)

    def test_relative_ribbon_flat_probability_top_remains_continuous(self):
        candidate = np.zeros((96, 180), dtype=np.uint8)
        candidate[44:50, 12:168] = 1
        score = np.zeros(candidate.shape, dtype=np.float32)
        profile = np.asarray([0.20, 0.30, 0.32, 0.32, 0.31, 0.20], dtype=np.float32)
        score[44:50, 12:168] = profile[:, None]
        result = graph_extraction.extract_relative_ribbon_centerline(
            score, candidate, np.zeros_like(score)
        )
        centerline = result["ribbon_centerline_mask"] > 0
        self.assertGreater(np.count_nonzero(centerline[45:49, 12:168]), 145)
        self.assertGreater(len(np.unique(np.where(centerline)[1])), 145)
        self.assertEqual(result["diagnostics"]["ribbon_component_count"], 1)

    def test_relative_ribbon_preserves_three_way_t_junction(self):
        candidate = np.zeros((160, 190), dtype=np.uint8)
        candidate[57:66, 15:176] = 1
        candidate[61:148, 91:100] = 1
        score = candidate.astype(np.float32) * 0.7
        orientation = np.zeros(candidate.shape, dtype=np.float32)
        orientation[64:148, 88:103] = np.float32(np.pi / 2.0)
        result = graph_extraction.extract_relative_ribbon_centerline(
            score, candidate, orientation
        )
        centerline = result["ribbon_centerline_mask"] > 0
        self.assertGreater(np.count_nonzero(centerline[59:64, 15:176]), 145)
        self.assertGreater(np.count_nonzero(centerline[61:148, 93:98]), 75)
        self.assertGreaterEqual(result["diagnostics"]["ribbon_junction_count"], 1)

    def test_relative_ribbon_keeps_close_parallel_ribbons_separate(self):
        candidate = np.zeros((112, 190), dtype=np.uint8)
        candidate[36:45, 12:178] = 1
        candidate[57:66, 12:178] = 1
        score = candidate.astype(np.float32) * 0.65
        result = graph_extraction.extract_relative_ribbon_centerline(
            score, candidate, np.zeros_like(score)
        )
        centerline = result["ribbon_centerline_mask"] > 0
        self.assertGreater(np.count_nonzero(centerline[38:43]), 155)
        self.assertGreater(np.count_nonzero(centerline[59:64]), 155)
        self.assertEqual(np.count_nonzero(centerline[46:56]), 0)
        self.assertEqual(result["diagnostics"]["ribbon_component_count"], 2)

    def test_continuous_trace_handles_repeated_width_changes_without_fishbones(self):
        candidate = np.zeros((128, 230), dtype=np.uint8)
        widths = (8, 18, 11, 22, 9)
        for col in range(12, 218):
            width = widths[min(4, (col - 12) // 42)]
            candidate[64 - width // 2:65 + width // 2, col] = 1
        score = candidate.astype(np.float32) * 0.65
        ridge = np.zeros(candidate.shape, dtype=np.uint8)
        cv2.line(ridge, (12, 64), (217, 64), 1, 1)
        result = graph_extraction.trace_relative_ribbon_centerlines(
            score, candidate, np.zeros_like(score),
            ridge_mask=ridge, ridge_strength=ridge.astype(np.float32),
        )
        centerline = result["continuous_centerline_mask"] > 0
        self.assertEqual(result["diagnostics"]["continuous_centerline_component_count"], 1)
        self.assertGreater(np.count_nonzero(centerline[61:68, 12:218]), 195)
        self.assertEqual(np.count_nonzero(centerline[:58]), 0)
        self.assertEqual(result["diagnostics"]["continuous_junction_count"], 0)

    def test_continuous_trace_survives_flat_top_and_local_orientation_noise(self):
        candidate = np.zeros((112, 212), dtype=np.uint8)
        candidate[51:59, 12:200] = 1
        score = np.zeros(candidate.shape, dtype=np.float32)
        score[51:59, 12:200] = np.asarray(
            [0.20, 0.30, 0.32, 0.32, 0.31, 0.28, 0.24, 0.20],
            dtype=np.float32,
        )[:, None]
        orientation = np.zeros(candidate.shape, dtype=np.float32)
        orientation[49:61, 82:118] = np.float32(np.pi / 2.0)
        ridge = np.zeros(candidate.shape, dtype=np.uint8)
        cv2.line(ridge, (12, 55), (199, 55), 1, 1)
        result = graph_extraction.trace_relative_ribbon_centerlines(
            score, candidate, orientation,
            ridge_mask=ridge, ridge_strength=ridge.astype(np.float32),
        )
        centerline = result["continuous_centerline_mask"] > 0
        self.assertEqual(result["diagnostics"]["continuous_centerline_component_count"], 1)
        self.assertGreater(len(np.unique(np.where(centerline)[1])), 180)

    def test_continuous_trace_follows_smooth_curved_ribbon(self):
        candidate = np.zeros((220, 220), dtype=np.uint8)
        ridge = np.zeros(candidate.shape, dtype=np.uint8)
        cv2.ellipse(candidate, (110, 110), (72, 72), 0, 20, 160, 1, 11)
        cv2.ellipse(ridge, (110, 110), (72, 72), 0, 20, 160, 1, 1)
        score = candidate.astype(np.float32) * 0.68
        rows, cols = np.indices(candidate.shape, dtype=np.float32)
        phase = np.arctan2(rows - 110.0, cols - 110.0)
        orientation = np.mod(phase + np.float32(np.pi / 2.0), np.float32(np.pi))
        result = graph_extraction.trace_relative_ribbon_centerlines(
            score, candidate, orientation,
            ridge_mask=ridge, ridge_strength=ridge.astype(np.float32),
        )
        centerline = result["continuous_centerline_mask"] > 0
        ridge_neighborhood = cv2.dilate(ridge, np.ones((5, 5), dtype=np.uint8)) > 0
        self.assertEqual(result["diagnostics"]["continuous_centerline_component_count"], 1)
        self.assertGreater(result["diagnostics"]["continuous_centerline_length"], 150.0)
        self.assertGreater(np.mean(ridge_neighborhood[centerline]), 0.85)

    def test_continuous_trace_keeps_t_branch_and_rejects_short_spur(self):
        candidate = np.zeros((190, 230), dtype=np.uint8)
        candidate[70:81, 15:215] = 1
        candidate[75:176, 105:116] = 1
        candidate[57:75, 105:116] = 1
        score = candidate.astype(np.float32) * 0.70
        orientation = np.zeros(candidate.shape, dtype=np.float32)
        orientation[55:178, 103:118] = np.float32(np.pi / 2.0)
        ridge = np.zeros(candidate.shape, dtype=np.uint8)
        cv2.line(ridge, (15, 75), (214, 75), 1, 1)
        cv2.line(ridge, (110, 57), (110, 175), 1, 1)
        result = graph_extraction.trace_relative_ribbon_centerlines(
            score, candidate, orientation,
            ridge_mask=ridge, ridge_strength=ridge.astype(np.float32),
        )
        centerline = result["continuous_centerline_mask"] > 0
        self.assertGreater(np.count_nonzero(centerline[72:79, 15:215]), 185)
        self.assertGreater(np.count_nonzero(centerline[75:176, 107:114]), 85)
        self.assertLess(np.count_nonzero(centerline[55:69, 107:114]), 5)
        self.assertGreaterEqual(result["diagnostics"]["confirmed_branch_count"], 1)
        self.assertGreaterEqual(result["diagnostics"]["rejected_spur_count"], 1)

    def test_continuous_trace_keeps_nearby_parallel_ribbons_independent(self):
        candidate = np.zeros((126, 220), dtype=np.uint8)
        candidate[35:46, 12:208] = 1
        candidate[58:69, 12:208] = 1
        score = candidate.astype(np.float32) * 0.66
        ridge = np.zeros(candidate.shape, dtype=np.uint8)
        cv2.line(ridge, (12, 40), (207, 40), 1, 1)
        cv2.line(ridge, (12, 63), (207, 63), 1, 1)
        result = graph_extraction.trace_relative_ribbon_centerlines(
            score, candidate, np.zeros_like(score),
            ridge_mask=ridge, ridge_strength=ridge.astype(np.float32),
        )
        centerline = result["continuous_centerline_mask"] > 0
        self.assertEqual(result["diagnostics"]["continuous_centerline_component_count"], 2)
        self.assertGreater(np.count_nonzero(centerline[37:44]), 185)
        self.assertGreater(np.count_nonzero(centerline[60:67]), 185)
        self.assertEqual(np.count_nonzero(centerline[47:57]), 0)

    def test_continuous_trace_suppresses_parallel_fragment_seeds_in_one_ribbon(self):
        candidate = np.zeros((130, 260), dtype=np.uint8)
        candidate[50:74, 10:250] = 1
        score = candidate.astype(np.float32) * 0.68
        ridge = np.zeros(candidate.shape, dtype=np.uint8)
        cv2.line(ridge, (10, 56), (95, 56), 1, 1)
        cv2.line(ridge, (75, 62), (180, 62), 1, 1)
        cv2.line(ridge, (150, 68), (249, 68), 1, 1)
        cv2.line(ridge, (20, 65), (120, 65), 1, 1)
        result = graph_extraction.trace_relative_ribbon_centerlines(
            score,
            candidate,
            np.zeros_like(score),
            ridge_mask=ridge,
            ridge_strength=ridge.astype(np.float32),
        )
        centerline = result["continuous_centerline_mask"] > 0
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["continuous_trace_count"], 1)
        self.assertGreaterEqual(
            diagnostics["seed_suppressed_existing_trace_count"], 3
        )
        self.assertEqual(diagnostics["confirmed_branch_count"], 0)
        self.assertEqual(diagnostics["continuous_junction_count"], 0)
        self.assertGreater(np.count_nonzero(centerline[58:66, 10:250]), 225)
        self.assertLessEqual(
            sum(np.count_nonzero(centerline[row]) > 80 for row in range(48, 76)),
            2,
        )

    def test_continuous_trace_wide_junction_keeps_branch_without_fishbone(self):
        candidate = np.zeros((230, 380), dtype=np.uint8)
        candidate[96:116, 15:365] = 1
        candidate[105:205, 188:202] = 1
        candidate[82:96, 310:321] = 1
        score = candidate.astype(np.float32) * 0.70
        orientation = np.zeros(candidate.shape, dtype=np.float32)
        orientation[103:208, 185:205] = np.float32(np.pi / 2.0)
        orientation[94:118, 10:370] = 0.0
        ridge = np.zeros(candidate.shape, dtype=np.uint8)
        cv2.line(ridge, (15, 105), (230, 105), 1, 1)
        cv2.line(ridge, (235, 101), (300, 101), 1, 1)
        cv2.line(ridge, (305, 110), (364, 110), 1, 1)
        cv2.line(ridge, (195, 105), (195, 204), 1, 1)
        cv2.line(ridge, (315, 82), (315, 95), 1, 1)
        result = graph_extraction.trace_relative_ribbon_centerlines(
            score,
            candidate,
            orientation,
            ridge_mask=ridge,
            ridge_strength=ridge.astype(np.float32),
        )
        centerline = result["continuous_centerline_mask"] > 0
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["continuous_centerline_component_count"], 1)
        self.assertLessEqual(diagnostics["continuous_trace_count"], 3)
        self.assertEqual(diagnostics["true_branch_count"], 1)
        self.assertEqual(diagnostics["confirmed_branch_count"], 1)
        self.assertEqual(diagnostics["continuous_junction_count"], 1)
        self.assertGreaterEqual(diagnostics["rejected_spur_count"], 1)
        self.assertGreaterEqual(
            diagnostics["seed_suppressed_existing_trace_count"], 2
        )
        self.assertGreater(np.count_nonzero(centerline[92:120, 15:365]), 330)
        self.assertGreater(np.count_nonzero(centerline[105:205, 184:206]), 85)
        self.assertLess(np.count_nonzero(centerline[80:95, 306:324]), 5)
        self.assertLessEqual(
            sum(np.count_nonzero(centerline[row]) > 80 for row in range(90, 121)),
            3,
        )

    def test_regularized_skeleton_a_jagged_straight_road_has_one_trunk(self):
        candidate = np.zeros((128, 270), dtype=np.uint8)
        candidate[52:76, 15:255] = 1
        for index, col in enumerate(range(28, 245, 24)):
            depth = 2 + index % 4
            candidate[52 - depth:52, col:col + 4] = 1
            candidate[76:76 + depth, col + 9:col + 13] = 1
            candidate[52:52 + min(depth, 4), col + 15:col + 18] = 0
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        skeleton = result["final_skeleton"] > 0
        adjacency = graph_extraction._skeleton_adjacency(skeleton)
        self.assertEqual(cv2.connectedComponents(skeleton.astype(np.uint8), 8)[0] - 1, 1)
        self.assertGreater(len(np.unique(np.where(skeleton)[1])), 225)
        self.assertLessEqual(sum(len(items) >= 3 for items in adjacency.values()), 2)

    def test_regularized_skeleton_b_small_holes_do_not_create_loops(self):
        candidate = np.zeros((130, 260), dtype=np.uint8)
        candidate[48:78, 12:248] = 1
        holes = [(70, 61, 2), (118, 64, 3), (175, 59, 2), (218, 66, 3)]
        for col, row, radius in holes:
            cv2.circle(candidate, (col, row), radius, 0, -1)
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        regularized = result["regularized_candidate"] > 0
        skeleton = result["final_skeleton"] > 0
        for col, row, _radius in holes:
            self.assertTrue(regularized[row, col])
        adjacency = graph_extraction._skeleton_adjacency(skeleton)
        self.assertEqual(sum(len(items) == 1 for items in adjacency.values()), 2)
        self.assertEqual(sum(len(items) >= 3 for items in adjacency.values()), 0)

    def test_regularized_skeleton_c_keeps_t_and_removes_boundary_spurs(self):
        candidate = np.zeros((210, 300), dtype=np.uint8)
        candidate[82:102, 18:282] = 1
        candidate[92:195, 140:160] = 1
        for col in (45, 82, 205, 244):
            candidate[77:82, col:col + 4] = 1
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        skeleton = result["final_skeleton"] > 0
        self.assertGreater(np.count_nonzero(skeleton[78:106, 18:282]), 230)
        self.assertGreater(np.count_nonzero(skeleton[92:195, 136:164]), 85)
        adjacency = graph_extraction._skeleton_adjacency(skeleton)
        self.assertEqual(sum(len(items) == 1 for items in adjacency.values()), 3)
        self.assertFalse(result["performance"]["junction_collapse_active"])

    def test_regularized_skeleton_d_preserves_y_junction(self):
        candidate = np.zeros((220, 260), dtype=np.uint8)
        cv2.line(candidate, (130, 190), (130, 105), 1, 18)
        cv2.line(candidate, (130, 105), (45, 28), 1, 18)
        cv2.line(candidate, (130, 105), (215, 28), 1, 18)
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        skeleton = result["final_skeleton"] > 0
        adjacency = graph_extraction._skeleton_adjacency(skeleton)
        self.assertEqual(sum(len(items) == 1 for items in adjacency.values()), 3)
        self.assertGreater(np.count_nonzero(skeleton[150:200, 122:138]), 35)
        self.assertGreater(np.count_nonzero(skeleton[20:75, 35:95]), 35)
        self.assertGreater(np.count_nonzero(skeleton[20:75, 165:225]), 35)

    def test_regularized_skeleton_e_keeps_close_parallel_roads_separate(self):
        candidate = np.zeros((130, 270), dtype=np.uint8)
        candidate[42:56, 12:258] = 1
        candidate[61:75, 12:258] = 1
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        regularized = result["regularized_candidate"] > 0
        skeleton = result["final_skeleton"] > 0
        self.assertEqual(cv2.connectedComponents(regularized.astype(np.uint8), 8)[0] - 1, 2)
        self.assertEqual(cv2.connectedComponents(skeleton.astype(np.uint8), 8)[0] - 1, 2)
        self.assertEqual(np.count_nonzero(regularized[57:60]), 0)

    def test_regularized_skeleton_f_collapses_complex_junction_cluster(self):
        candidate = np.zeros((240, 280), dtype=np.uint8)
        candidate[100:126, 18:262] = 1
        candidate[20:220, 126:152] = 1
        candidate[91:136, 112:166] = 1
        result = graph_extraction.extract_regularized_relative_centerline(
            candidate, junction_collapse=True
        )
        raw = result["raw_skeleton"] > 0
        final = result["final_skeleton"] > 0
        raw_adjacency = graph_extraction._skeleton_adjacency(raw)
        final_adjacency = graph_extraction._skeleton_adjacency(final)
        self.assertEqual(sum(len(items) == 1 for items in final_adjacency.values()), 4)
        self.assertLessEqual(
            sum(len(items) >= 3 for items in final_adjacency.values()),
            sum(len(items) >= 3 for items in raw_adjacency.values()),
        )
        self.assertGreaterEqual(result["performance"]["junction_cluster_count_after"], 1)

    def test_regularized_skeleton_g_does_not_bridge_real_gap(self):
        candidate = np.zeros((110, 280), dtype=np.uint8)
        candidate[46:66, 12:128] = 1
        candidate[46:66, 150:268] = 1
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        regularized = result["regularized_candidate"] > 0
        self.assertEqual(cv2.connectedComponents(regularized.astype(np.uint8), 8)[0] - 1, 2)
        self.assertEqual(np.count_nonzero(regularized[:, 132:146]), 0)

    def test_regularized_skeleton_h_fills_long_narrow_crack(self):
        candidate = np.zeros((130, 280), dtype=np.uint8)
        candidate[40:90, 10:270] = 1
        candidate[62:67, 50:230] = 0
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        audit = result["performance"]
        self.assertTrue(np.all(result["regularized_candidate"][62:67, 50:230] > 0))
        self.assertEqual(audit["narrow_hole_count"], 1)
        self.assertEqual(audit["hole_filled_by_narrow_width_count"], 1)
        self.assertEqual(audit["final_cycle_count"], 0)

    def test_regularized_skeleton_i_fills_repeated_narrow_holes(self):
        candidate = np.zeros((150, 340), dtype=np.uint8)
        candidate[42:102, 10:330] = 1
        for start in (45, 130, 215):
            candidate[67:72, start:start + 65] = 0
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        self.assertEqual(result["performance"]["narrow_hole_filled_count"], 3)
        self.assertEqual(result["performance"]["final_cycle_count"], 0)

    def test_regularized_skeleton_j_preserves_x_junction(self):
        candidate = np.zeros((190, 240), dtype=np.uint8)
        candidate[84:106, 15:225] = 1
        candidate[15:175, 109:131] = 1
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        skeleton = result["final_skeleton"] > 0
        adjacency = graph_extraction._skeleton_adjacency(skeleton)
        self.assertEqual(sum(len(items) == 1 for items in adjacency.values()), 4)
        self.assertEqual(result["performance"]["removed_cycle_count"], 0)

    def test_regularized_skeleton_k_preserves_true_central_median(self):
        candidate = np.zeros((140, 300), dtype=np.uint8)
        candidate[25:115, 10:290] = 1
        candidate[52:88, 40:260] = 0
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        regularized = result["regularized_candidate"] > 0
        self.assertFalse(regularized[70, 150])
        self.assertGreaterEqual(result["performance"]["hole_preserved_count"], 1)
        self.assertEqual(result["performance"]["narrow_hole_filled_count"], 0)

    def test_regularized_skeleton_l_removes_small_fake_cycle_locally(self):
        skeleton = np.zeros((120, 180), dtype=np.uint8)
        cv2.line(skeleton, (10, 55), (170, 55), 1, 1)
        cv2.rectangle(skeleton, (75, 45), (95, 65), 1, 1)
        result = graph_extraction.remove_width_scale_small_cycles(
            skeleton, np.full(skeleton.shape, 10.0, dtype=np.float32)
        )
        self.assertGreaterEqual(result["raw_cycle_count"], 1)
        self.assertGreaterEqual(result["removed_cycle_count"], 1)
        self.assertEqual(result["final_cycle_count"], 0)
        self.assertTrue(result["cleaned_skeleton"][55, 15])
        self.assertTrue(result["cleaned_skeleton"][55, 165])

    def test_regularized_skeleton_m_preserves_large_roundabout(self):
        skeleton = np.zeros((180, 180), dtype=np.uint8)
        cv2.circle(skeleton, (90, 90), 50, 1, 1)
        result = graph_extraction.remove_width_scale_small_cycles(
            skeleton, np.full(skeleton.shape, 5.0, dtype=np.float32)
        )
        self.assertEqual(result["raw_cycle_count"], 1)
        self.assertEqual(result["removed_cycle_count"], 0)
        self.assertEqual(result["final_cycle_count"], 1)

    def test_regularized_skeleton_n_preserves_curved_ramp_loop(self):
        skeleton = np.zeros((220, 260), dtype=np.uint8)
        cv2.ellipse(skeleton, (145, 105), (62, 48), 0, 0, 360, 1, 1)
        cv2.line(skeleton, (15, 105), (83, 105), 1, 1)
        result = graph_extraction.remove_width_scale_small_cycles(
            skeleton, np.full(skeleton.shape, 6.0, dtype=np.float32)
        )
        self.assertGreaterEqual(result["raw_cycle_count"], 1)
        self.assertEqual(result["removed_cycle_count"], 0)
        self.assertGreaterEqual(result["final_cycle_count"], 1)

    def test_regularized_skeleton_small_benchmark_has_complete_audit(self):
        candidate = np.zeros((512, 512), dtype=np.uint8)
        for row in range(70, 470, 80):
            candidate[row:row + 18, 20:492] = 1
        candidate[40:490, 245:265] = 1
        result = graph_extraction.extract_regularized_relative_centerline(candidate)
        audit = result["performance"]
        for key in (
            "candidate_regularization_seconds", "distance_transform_seconds",
            "hole_fill_seconds", "smoothing_seconds", "skeletonize_seconds",
            "hole_detection_seconds", "narrow_hole_analysis_seconds",
            "spur_pruning_seconds", "cycle_detection_seconds",
            "cycle_cleanup_seconds", "junction_collapse_seconds",
            "vector_simplification_seconds", "total_seconds",
            "candidate_pixel_count_before", "candidate_pixel_count_after",
            "skeleton_length_before_pruning", "skeleton_length_after_pruning",
            "spur_removed_count", "spur_removed_length",
            "raw_cycle_count", "small_cycle_count", "removed_cycle_count",
            "preserved_cycle_count", "hole_filled_by_size_count",
            "hole_filled_by_narrow_width_count",
            "cycle_near_detected_hole_fraction",
            "cycle_near_filled_hole_fraction",
            "junction_pixel_count_before", "junction_cluster_count_after",
        ):
            self.assertIn(key, audit)
        self.assertLess(audit["total_seconds"], 3.0)

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
        self.assertTrue(all(
            str(row["line_source"]).startswith("relative") for row in metadata
        ))
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
