from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely import union_all
from shapely.geometry import LineString, box


WIDTH = Path(__file__).resolve().parents[1] / "engine" / "width"
if str(WIDTH) not in sys.path:
    sys.path.insert(0, str(WIDTH))

from road_change_detection import (  # noqa: E402
    DetectionConfig,
    _detect_changes_internal,
    _write_paired_width_debug,
    detect_changes,
)


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
