from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely import union_all
from shapely.geometry import LineString, Point, box


WIDTH = Path(__file__).resolve().parents[1] / "engine" / "width"
if str(WIDTH) not in sys.path:
    sys.path.insert(0, str(WIDTH))

from road_change_detection import (  # noqa: E402
    DetectionConfig,
    _detect_changes_internal,
    _write_paired_width_debug,
    detect_changes,
)
from paired_width_profile import (  # noqa: E402
    PairedWidthConfig,
    PairedWidthProfile,
    PairedWidthSample,
    candidate_change_runs,
    evaluate_change_run,
    measure_paired_width_profile,
)
from road_existence_evidence import RoadProbabilityRaster  # noqa: E402


def roads(rows):
    return gpd.GeoDataFrame(
        {
            "width_map": [row[1] for row in rows],
            "quality_gr": [row[2] if len(row) > 2 else "A" for row in rows],
        },
        geometry=[row[0] for row in rows],
        crs="EPSG:3857",
    )


def surface(*polygons):
    return gpd.GeoDataFrame(geometry=list(polygons), crs="EPSG:3857")


def probability_raster(
    road_width: float,
    *,
    road_probability: float = 0.90,
    background_probability: float = 0.10,
) -> RoadProbabilityRaster:
    height, width = 160, 140
    transform = from_origin(-20, 80, 1, 1)
    y = 80 - (np.arange(height, dtype=np.float32) + 0.5)
    values = np.full((height, width), background_probability, dtype=np.float32)
    values[np.abs(y) <= road_width * 0.5, :] = road_probability
    return RoadProbabilityRaster(values, transform, "EPSG:3857")


def paired_sample(index: int, difference: float | None) -> PairedWidthSample:
    valid = difference is not None
    before_width = 5.0 if valid else None
    after_width = 5.0 + float(difference) if valid else None
    return PairedWidthSample(
        canonical_id="C00000001",
        sample_index=index,
        position_m=float(index * 2),
        t=index / 10,
        point=Point(index * 2, 0),
        before_width=before_width,
        after_width=after_width,
        width_diff=difference,
        valid=valid,
        reject_reason="" if valid else "synthetic_invalid",
    )


class PairedWidthProfileTests(unittest.TestCase):
    def test_same_width_with_small_centerline_offset_is_unchanged(self):
        before_line = LineString([(0, 0), (100, 0)])
        after_line = LineString([(0, 1.4), (100, 1.4)])
        positive, negative, summary = detect_changes(
            roads([(before_line, 6.0)]),
            roads([(after_line, 6.0)]),
            before_surfaces=surface(box(0, -3, 100, 3)),
            after_surfaces=surface(box(0, -1.6, 100, 4.4)),
        )
        self.assertNotIn("widened", set(positive["change_typ"]))
        self.assertNotIn("narrowed", set(negative["change_typ"]))
        self.assertEqual(summary["width_changed_centerline_count"], 0)

    def test_same_width_with_different_segmentation_is_unchanged(self):
        before_line = LineString([(0, 0), (100, 0)])
        after_lines = [LineString([(0, 0.6), (43, 0.6)]), LineString([(43, 0.6), (100, 0.6)])]
        positive, negative, summary = detect_changes(
            roads([(before_line, 6.0)]),
            roads([(after_lines[0], 6.0), (after_lines[1], 6.0)]),
            before_surfaces=surface(box(0, -3, 100, 3)),
            after_surfaces=surface(box(0, -2.4, 100, 3.6)),
        )
        self.assertNotIn("widened", set(positive["change_typ"]))
        self.assertNotIn("narrowed", set(negative["change_typ"]))
        self.assertEqual(summary["width_changed_centerline_count"], 0)

    def test_sparse_outlier_cross_sections_do_not_create_change(self):
        line = LineString([(0, 0), (100, 0)])
        positive, negative, summary = detect_changes(
            roads([(line, 6.0)]), roads([(line, 6.0)]),
            before_surfaces=surface(box(0, -3, 100, 3)),
            after_surfaces=surface(box(0, -3, 100, 3), box(48, -6, 52, 6)),
        )
        self.assertNotIn("widened", set(positive["change_typ"]))
        self.assertNotIn("narrowed", set(negative["change_typ"]))
        self.assertEqual(summary["width_changed_centerline_count"], 0)

    def test_local_continuous_widening_outputs_only_true_interval(self):
        line = LineString([(0, 0), (100, 0)])
        positive, negative, summary = detect_changes(
            roads([(line, 4.0)]), roads([(line, 8.0)]),
            before_surfaces=surface(box(0, -2, 100, 2)),
            after_surfaces=surface(
                box(0, -2, 30, 2), box(30, -4, 70, 4), box(70, -2, 100, 2),
            ),
        )
        widened = positive.loc[(positive["change_typ"] == "widened") & (positive["qa_state"] == "auto")]
        self.assertFalse(widened.empty)
        self.assertNotIn("narrowed", set(negative["change_typ"]))
        bounds = union_all(widened.geometry.values).bounds
        self.assertGreaterEqual(bounds[0], 28.0)
        self.assertLessEqual(bounds[2], 72.0)
        self.assertGreaterEqual(summary["width_changed_centerline_count"], 1)

    def test_local_continuous_narrowing_outputs_only_true_interval(self):
        line = LineString([(0, 0), (100, 0)])
        positive, negative, summary = detect_changes(
            roads([(line, 8.0)]), roads([(line, 4.0)]),
            before_surfaces=surface(
                box(0, -2, 30, 2), box(30, -4, 70, 4), box(70, -2, 100, 2),
            ),
            after_surfaces=surface(box(0, -2, 100, 2)),
        )
        narrowed = negative.loc[(negative["change_typ"] == "narrowed") & (negative["qa_state"] == "auto")]
        self.assertFalse(narrowed.empty)
        self.assertNotIn("widened", set(positive["change_typ"]))
        bounds = union_all(narrowed.geometry.values).bounds
        self.assertGreaterEqual(bounds[0], 28.0)
        self.assertLessEqual(bounds[2], 72.0)
        self.assertGreaterEqual(summary["width_changed_centerline_count"], 1)

    def test_insufficient_valid_samples_never_auto_detect_change(self):
        line = LineString([(0, 0), (100, 0)])
        positive, negative, summary = detect_changes(
            roads([(line, 4.0)]), roads([(line, 8.0)]),
            before_surfaces=surface(box(0, -2, 100, 2)),
            after_surfaces=surface(box(0, -4, 8, 4), box(92, -4, 100, 4)),
        )
        self.assertTrue(positive.loc[positive["change_typ"] == "widened"].empty)
        self.assertTrue(negative.loc[negative["change_typ"] == "narrowed"].empty)
        self.assertEqual(summary["width_changed_centerline_count"], 0)

    def test_stable_sustained_width_increase_is_detected(self):
        line = LineString([(0, 0), (100, 0)])
        positive, negative, summary = detect_changes(
            roads([(line, 5.0)]), roads([(line, 8.0)]),
            before_surfaces=surface(box(0, -2.5, 100, 2.5)),
            after_surfaces=surface(box(0, -4, 100, 4)),
        )
        widened = positive.loc[(positive["change_typ"] == "widened") & (positive["qa_state"] == "auto")]
        self.assertFalse(widened.empty)
        self.assertTrue(negative.empty)
        self.assertGreaterEqual(summary["width_changed_centerline_count"], 1)

    def test_probability_profile_measures_width_when_surface_is_missing(self):
        line = LineString([(0, 0), (40, 0)])
        probability = probability_raster(6.0)
        profile = measure_paired_width_profile(
            "C00000001", line, line, line, None, None,
            PairedWidthConfig(normal_half_length=20.0),
            before_probability=probability,
            after_probability=probability,
            geometry_crs="EPSG:3857",
        )
        self.assertGreater(profile.valid_ratio, 0.95)
        sample = profile.samples[len(profile.samples) // 2]
        self.assertEqual(sample.before_width_source, "probability")
        self.assertAlmostEqual(float(sample.before_probability_width), 6.0, delta=1.0)
        self.assertGreater(sample.before_width_confidence, 0.55)
        self.assertIsNotNone(sample.before_probability_left_distance)
        self.assertIsNotNone(sample.before_probability_right_distance)

    def test_probability_cross_section_marks_nodata_and_raster_bounds_invalid(self):
        values = np.full((10, 10), 0.5, dtype=np.float32)
        valid = np.ones((10, 10), dtype=bool)
        valid[5, :] = False
        probability = RoadProbabilityRaster(
            values, from_origin(0, 10, 1, 1), "EPSG:3857", valid,
        )
        profile = probability.sample_cross_section(
            Point(5, 5), (0.0, 1.0), "EPSG:3857", search_radius=8.0,
        )
        distances = np.asarray(profile["distance"])
        probabilities = np.asarray(profile["probability"])
        valid_mask = np.asarray(profile["valid_mask"])
        self.assertEqual(distances.shape, probabilities.shape)
        self.assertEqual(distances.shape, valid_mask.shape)
        self.assertFalse(valid_mask[0])
        self.assertFalse(valid_mask[-1])
        self.assertTrue(np.isnan(probabilities[~valid_mask]).all())

    def test_probability_recovers_widening_when_after_surface_is_missing(self):
        line = LineString([(0, 0), (100, 0)])
        positive, negative, summary = detect_changes(
            roads([(line, 4.0)]), roads([(line, 8.0)]),
            config=DetectionConfig(paired_width_normal_half_length=20.0),
            before_surfaces=surface(box(0, -2, 100, 2)),
            after_surfaces=None,
            before_probability=probability_raster(4.0),
            after_probability=probability_raster(8.0),
        )
        widened = positive.loc[
            (positive["change_typ"] == "widened") & (positive["qa_state"] == "auto")
        ]
        self.assertFalse(widened.empty)
        self.assertTrue(negative.empty)
        self.assertGreaterEqual(summary["width_changed_centerline_count"], 1)

    def test_consistent_surface_and_probability_are_fused_stably(self):
        line = LineString([(0, 0), (40, 0)])
        road_surface = box(0, -3, 40, 3)
        probability = probability_raster(6.0)
        profile = measure_paired_width_profile(
            "C00000001", line, line, line, road_surface, road_surface,
            PairedWidthConfig(normal_half_length=20.0),
            before_probability=probability,
            after_probability=probability,
            geometry_crs="EPSG:3857",
        )
        sample = profile.samples[len(profile.samples) // 2]
        self.assertTrue(sample.valid)
        self.assertEqual(sample.before_width_source, "fused")
        self.assertAlmostEqual(float(sample.before_width), 6.0, delta=0.75)
        self.assertGreater(sample.before_width_confidence, 0.90)

    def test_surface_probability_conflict_cannot_auto_confirm_width_change(self):
        line = LineString([(0, 0), (100, 0)])
        artifacts = {}
        positive, negative, _unchanged, summary = _detect_changes_internal(
            roads([(line, 4.0)]), roads([(line, 8.0)]),
            DetectionConfig(paired_width_normal_half_length=20.0), "before", "after",
            before_surfaces=surface(box(0, -2, 100, 2)),
            after_surfaces=surface(box(0, -4, 100, 4)),
            before_probability=probability_raster(4.0),
            after_probability=probability_raster(4.0),
            artifacts=artifacts,
        )
        self.assertTrue(positive.loc[positive["change_typ"] == "widened"].empty)
        self.assertTrue(negative.loc[negative["change_typ"] == "narrowed"].empty)
        self.assertEqual(summary["width_changed_centerline_count"], 0)
        samples = artifacts["paired_width_samples"]
        self.assertTrue(samples["surface_probability_disagreement"].any())
        self.assertTrue(samples["reject_reason"].str.contains("surface_probability_disagreement").any())
        self.assertTrue(samples["change_decision"].str.startswith("review:").any())

    def test_scene_probability_strength_change_does_not_create_false_width_change(self):
        line = LineString([(0, 0), (100, 0)])
        positive, negative, summary = detect_changes(
            roads([(line, 6.0)]), roads([(line, 6.0)]),
            config=DetectionConfig(paired_width_normal_half_length=20.0),
            before_probability=probability_raster(
                6.0, road_probability=0.90, background_probability=0.10,
            ),
            after_probability=probability_raster(
                6.0, road_probability=0.45, background_probability=0.15,
            ),
        )
        self.assertNotIn("widened", set(positive["change_typ"]))
        self.assertNotIn("narrowed", set(negative["change_typ"]))
        self.assertEqual(summary["width_changed_centerline_count"], 0)

    def test_single_invalid_sample_does_not_split_true_change_run(self):
        samples = tuple(
            paired_sample(index, None if index == 5 else 3.0)
            for index in range(11)
        )
        profile = PairedWidthProfile(
            "C00000001", LineString([(0, 0), (20, 0)]), samples, 10 / 11,
        )
        config = PairedWidthConfig(minimum_continuous_length=15.0)
        runs = candidate_change_runs(profile, config)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0].axis.length, 20.0, places=6)
        decision = evaluate_change_run(
            runs[0].samples,
            axis_length=runs[0].axis.length,
            valid_ratio=runs[0].valid_ratio,
            config=config,
        )
        self.assertTrue(decision["accepted"])

    def test_sign_reversal_is_never_bridged(self):
        differences = [3.0] * 5 + [None] + [-3.0] * 5
        samples = tuple(paired_sample(index, value) for index, value in enumerate(differences))
        profile = PairedWidthProfile(
            "C00000001", LineString([(0, 0), (20, 0)]), samples, 10 / 11,
        )
        runs = candidate_change_runs(profile, PairedWidthConfig())
        self.assertEqual([run.sign for run in runs], [1, -1])
        self.assertLess(runs[0].end_m, runs[1].start_m)

    def test_uncertainty_gate_rejects_noise_scale_change(self):
        samples = tuple(paired_sample(index, value) for index, value in enumerate((1.0, 2.5, 4.0)))
        decision = evaluate_change_run(
            samples,
            axis_length=30.0,
            valid_ratio=1.0,
            config=PairedWidthConfig(
                minimum_samples=3,
                minimum_continuous_length=20.0,
                maximum_diff_mad=10.0,
                uncertainty_scale=2.5,
            ),
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reject_reason"], "paired_width_uncertainty_too_large")

    def test_debug_tables_contain_paired_measurement_and_decision_fields(self):
        line = LineString([(0, 0), (40, 0)])
        artifacts = {}
        _positive, _negative, _unchanged, _summary = _detect_changes_internal(
            roads([(line, 4.0)]), roads([(line, 8.0)]), DetectionConfig(), "before", "after",
            before_surfaces=surface(box(0, -2, 40, 2)),
            after_surfaces=surface(box(0, -4, 40, 4)),
            artifacts=artifacts,
        )
        samples = artifacts["paired_width_samples"]
        decisions = artifacts["paired_width_decisions"]
        for field in (
            "canonical_id", "sample_position_m", "before_width", "after_width",
            "width_diff", "valid", "reject_reason", "mad", "uncertainty",
            "valid_ratio", "sample_count", "change_decision",
            "before_surface_width", "before_probability_width", "before_final_width",
            "before_width_source", "before_width_confidence",
            "before_probability_confidence",
            "after_surface_width", "after_probability_width", "after_final_width",
            "after_width_source", "after_width_confidence",
            "after_probability_confidence",
            "surface_probability_disagreement",
        ):
            self.assertIn(field, samples.columns)
        self.assertIn("change_decision", decisions.columns)
        with tempfile.TemporaryDirectory() as raw:
            outputs = _write_paired_width_debug(Path(raw), artifacts)
            self.assertTrue(Path(outputs["paired_width_samples"]).is_file())
            self.assertTrue(Path(outputs["paired_width_decisions"]).is_file())
            self.assertIn("_debug", Path(outputs["paired_width_samples"]).parts)


if __name__ == "__main__":
    unittest.main()
