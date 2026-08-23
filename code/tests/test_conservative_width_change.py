from __future__ import annotations

import sys
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, box


ENGINE = Path(__file__).resolve().parents[1] / "engine" / "width"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from road_change_detection import DetectionConfig, detect_changes  # noqa: E402


def lines(widths, geometries, qualities=None):
    data = {"width_map": widths}
    if qualities is not None:
        data["quality_gr"] = qualities
    return gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:3857")


class ConservativeWidthChangeTests(unittest.TestCase):
    def test_default_rejects_one_metre_boundary_jitter(self) -> None:
        before = lines([4.0], [LineString([(0, 0), (100, 0)])], ["A"])
        after = lines([5.0], [LineString([(0, 0.5), (100, 0.5)])], ["A"])
        before_surface = gpd.GeoDataFrame(geometry=[box(0, -2, 100, 2)], crs=before.crs)
        after_surface = gpd.GeoDataFrame(geometry=[box(0, -2, 100, 3)], crs=before.crs)

        positive, negative, summary = detect_changes(
            before, after, before_surfaces=before_surface, after_surfaces=after_surface,
        )

        self.assertNotIn("widened", set(positive["change_typ"]))
        self.assertNotIn("narrowed", set(negative["change_typ"]))
        self.assertEqual(summary["width_changed_centerline_count"], 0)

    def test_default_accepts_long_reciprocal_high_quality_change(self) -> None:
        before = lines([5.0], [LineString([(0, 0), (100, 0)])], ["A"])
        after = lines([8.0], [LineString([(0, 0.5), (100, 0.5)])], ["A"])
        before_surface = gpd.GeoDataFrame(geometry=[box(0, -2.5, 100, 2.5)], crs=before.crs)
        after_surface = gpd.GeoDataFrame(geometry=[box(0, -3.5, 100, 4.5)], crs=before.crs)

        positive, negative, summary = detect_changes(
            before, after, before_surfaces=before_surface, after_surfaces=after_surface,
        )

        self.assertIn("widened", set(positive["change_typ"]))
        self.assertTrue(negative.empty)
        self.assertEqual(summary["width_changed_centerline_count"], 1)

    def test_default_rejects_short_or_low_quality_widths(self) -> None:
        before = lines(
            [4.0, 4.0],
            [LineString([(0, 0), (15, 0)]), LineString([(0, 50), (100, 50)])],
            ["A", "C"],
        )
        after = lines(
            [8.0, 8.0],
            [LineString([(0, 0.5), (15, 0.5)]), LineString([(0, 50.5), (100, 50.5)])],
            ["A", "A"],
        )
        before_surface = gpd.GeoDataFrame(
            geometry=[box(0, -2, 15, 2), box(0, 48, 100, 52)], crs=before.crs,
        )
        after_surface = gpd.GeoDataFrame(
            geometry=[box(0, -3.5, 15, 4.5), box(0, 47, 100, 55)], crs=before.crs,
        )

        positive, negative, summary = detect_changes(
            before, after, before_surfaces=before_surface, after_surfaces=after_surface,
        )

        self.assertNotIn("widened", set(positive["change_typ"]))
        self.assertNotIn("narrowed", set(negative["change_typ"]))
        self.assertEqual(summary["width_rejected_short_count"], 1)
        self.assertEqual(summary["width_rejected_quality_count"], 1)

    def test_missing_widths_never_create_width_change(self) -> None:
        before = lines([0.0], [LineString([(0, 0), (100, 0)])], ["C"])
        after = lines([8.0], [LineString([(0, 0.5), (100, 0.5)])], ["A"])
        before_surface = gpd.GeoDataFrame(geometry=[box(0, -4, 100, 4)], crs=before.crs)
        after_surface = gpd.GeoDataFrame(geometry=[box(0, -4, 100, 4)], crs=before.crs)
        positive, negative, summary = detect_changes(
            before, after, before_surfaces=before_surface, after_surfaces=after_surface,
        )
        self.assertTrue(positive.empty)
        self.assertTrue(negative.empty)
        self.assertEqual(summary["width_changed_centerline_count"], 0)

    def test_centerline_presence_disagreement_is_retained_for_review(self) -> None:
        before = lines([6.0], [LineString([(0, 0), (100, 0)])], ["A"])
        after = lines([6.0], [LineString([(0, 30), (100, 30)])], ["A"])
        before_surface = gpd.GeoDataFrame(
            geometry=[box(0, -3, 100, 3), box(0, 27, 100, 33)], crs=before.crs,
        )
        after_surface = before_surface.copy()
        positive, negative, summary = detect_changes(
            before, after, before_surfaces=before_surface, after_surfaces=after_surface,
        )
        self.assertIn("added", set(positive["change_typ"]))
        self.assertIn("removed", set(negative["change_typ"]))
        self.assertEqual(set(positive["qa_state"]), {"review"})
        self.assertEqual(set(negative["qa_state"]), {"review"})
        self.assertGreaterEqual(summary["presence_review_count"], 2)
        self.assertEqual(summary["presence_suppressed_low_confidence_count"], 0)

    def test_implausibly_large_width_jump_is_not_width_change(self) -> None:
        before = lines([4.0], [LineString([(0, 0), (100, 0)])], ["A"])
        after = lines([20.0], [LineString([(0, 0.5), (100, 0.5)])], ["A"])
        before_surface = gpd.GeoDataFrame(geometry=[box(0, -2, 100, 2)], crs=before.crs)
        after_surface = gpd.GeoDataFrame(geometry=[box(0, -9.5, 100, 10.5)], crs=before.crs)

        positive, negative, summary = detect_changes(
            before, after, before_surfaces=before_surface, after_surfaces=after_surface,
        )

        self.assertNotIn("widened", set(positive["change_typ"]))
        self.assertNotIn("narrowed", set(negative["change_typ"]))
        self.assertEqual(summary["width_rejected_excessive_count"], 1)

    def test_whole_removed_road_is_counted_before_width_rules(self) -> None:
        before = lines(
            [6.0, 6.0],
            [LineString([(0, 0), (100, 0)]), LineString([(0, 30), (100, 30)])],
            ["A", "A"],
        )
        after = lines([6.0], [LineString([(0, 0.5), (100, 0.5)])], ["A"])
        before_surface = gpd.GeoDataFrame(
            geometry=[box(0, -3, 100, 3), box(0, 27, 100, 33)], crs=before.crs,
        )
        after_surface = gpd.GeoDataFrame(geometry=[box(0, -2.5, 100, 3.5)], crs=before.crs)

        positive, negative, summary = detect_changes(
            before, after, before_surfaces=before_surface, after_surfaces=after_surface,
            before_valid_area=gpd.GeoDataFrame(
                geometry=[box(-20, -20, 120, 60)], crs=before.crs
            ),
            after_valid_area=gpd.GeoDataFrame(
                geometry=[box(-20, -20, 120, 60)], crs=after.crs
            ),
        )

        self.assertIn("removed", set(negative["change_typ"]))
        self.assertGreaterEqual(summary["presence_confirmed_count"], 1)

    def test_dual_model_evidence_confirms_long_added_road(self) -> None:
        before = lines([6.0], [LineString([(0, 0), (100, 0)])], ["A"])
        after = lines(
            [6.0, 6.0],
            [LineString([(0, 0), (100, 0)]), LineString([(0, 30), (100, 30)])],
            ["A", "A"],
        )
        before_surface = gpd.GeoDataFrame(geometry=[box(0, -3, 100, 3)], crs=before.crs)
        after_surface = gpd.GeoDataFrame(
            geometry=[box(0, -3, 100, 3), box(0, 27, 100, 33)], crs=before.crs,
        )
        positive, negative, summary = detect_changes(
            before, after, before_surfaces=before_surface, after_surfaces=after_surface,
            before_valid_area=gpd.GeoDataFrame(
                geometry=[box(-20, -20, 120, 60)], crs=before.crs
            ),
            after_valid_area=gpd.GeoDataFrame(
                geometry=[box(-20, -20, 120, 60)], crs=after.crs
            ),
        )
        self.assertIn("added", set(positive["change_typ"]))
        self.assertTrue(negative.empty)
        self.assertGreaterEqual(summary["presence_confirmed_count"], 1)


if __name__ == "__main__":
    unittest.main()
