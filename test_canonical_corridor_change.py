from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely import union_all
from shapely.geometry import LineString, box


WIDTH = Path(__file__).resolve().parent / "engine" / "width"
if str(WIDTH) not in sys.path:
    sys.path.insert(0, str(WIDTH))

from production_workflow import _fuse_centerline_records  # noqa: E402
from road_change_detection import (  # noqa: E402
    DetectionConfig,
    _detect_changes_internal,
    _write_outputs,
    detect_changes,
)
import user_pipeline  # noqa: E402


def roads(rows):
    return gpd.GeoDataFrame(
        {
            "width_map": [row[1] for row in rows],
            "quality_gr": [row[2] if len(row) > 2 else "A" for row in rows],
            "line_source": [row[3] if len(row) > 3 else "samroad" for row in rows],
        },
        geometry=[row[0] for row in rows],
        crs="EPSG:3857",
    )


def surfaces(lines, widths):
    return gpd.GeoDataFrame(
        geometry=[line.buffer(width * 0.5, cap_style="flat") for line, width in zip(lines, widths)],
        crs="EPSG:3857",
    )


class CanonicalCorridorChangeTests(unittest.TestCase):
    def test_offset_same_width_has_no_width_change(self):
        before_line = LineString([(0, 0), (100, 0)])
        after_line = LineString([(0, 1.8), (100, 1.8)])
        positive, negative, _summary = detect_changes(
            roads([(before_line, 6, "A")]), roads([(after_line, 6, "A")]),
            before_surfaces=surfaces([before_line], [6]), after_surfaces=surfaces([after_line], [6]),
        )
        self.assertTrue(positive.empty)
        self.assertTrue(negative.empty)

    def test_irregular_actual_surface_boundaries_do_not_create_change(self):
        line = LineString([(0, 0), (100, 0)])
        before_surface = gpd.GeoDataFrame(geometry=[box(0, -3, 100, 3)], crs="EPSG:3857")
        after_surface = gpd.GeoDataFrame(
            geometry=[box(0, -4, 30, 3), box(30, -3, 70, 4), box(70, -3.5, 100, 3)],
            crs="EPSG:3857",
        )
        positive, negative, _summary = detect_changes(
            roads([(line, 6, "A")]), roads([(line, 6, "A")]),
            before_surfaces=before_surface, after_surfaces=after_surface,
        )
        self.assertTrue(positive.empty)
        self.assertTrue(negative.empty)

    def test_local_continuous_widening_outputs_only_local_band(self):
        before = roads([(LineString([(0, 0), (100, 0)]), 4, "A")])
        after = roads([
            (LineString([(0, 0), (30, 0)]), 4, "A"),
            (LineString([(30, 0), (70, 0)]), 8, "A"),
            (LineString([(70, 0), (100, 0)]), 4, "A"),
        ])
        positive, negative, _summary = detect_changes(before, after)
        widened = positive.loc[positive["change_typ"] == "widened"]
        self.assertFalse(widened.empty)
        self.assertTrue(negative.empty)
        bounds = union_all(widened.geometry.values).bounds
        self.assertGreaterEqual(bounds[0], 28.0)
        self.assertLessEqual(bounds[2], 72.0)

    def test_perpendicular_crossing_is_not_a_match(self):
        before = roads([(LineString([(-50, 0), (50, 0)]), 6, "A")])
        after = roads([(LineString([(0, -50), (0, 50)]), 6, "A")])
        positive, negative, summary = detect_changes(before, after)
        self.assertEqual(summary["valid_candidate_count"], 0)
        self.assertEqual(set(positive["change_typ"]), {"added"})
        self.assertEqual(set(negative["change_typ"]), {"removed"})

    def test_parallel_dual_roads_match_nearest_carriageway(self):
        before = roads([
            (LineString([(0, 0), (100, 0)]), 5, "A"),
            (LineString([(0, 2.5), (100, 2.5)]), 5, "A"),
        ])
        after = roads([
            (LineString([(0, 0.4), (100, 0.4)]), 5, "A"),
            (LineString([(0, 2.9), (100, 2.9)]), 5, "A"),
        ])
        positive, negative, summary = detect_changes(before, after)
        self.assertTrue(positive.empty)
        self.assertTrue(negative.empty)
        self.assertGreater(summary["rejected_parallel_competitor_count"], 0)

    def test_one_long_line_matches_many_short_lines(self):
        before = roads([(LineString([(0, 0), (90, 0)]), 6, "A")])
        after = roads([
            (LineString([(0, 0.5), (30, 0.5)]), 6, "A"),
            (LineString([(30, 0.5), (60, 0.5)]), 6, "A"),
            (LineString([(60, 0.5), (90, 0.5)]), 6, "A"),
        ])
        positive, negative, summary = detect_changes(before, after)
        self.assertTrue(positive.empty)
        self.assertTrue(negative.empty)
        self.assertGreater(summary["matched_centerline_count"], 1)

    def test_presence_case_a_different_segmentation_has_no_added_or_removed(self):
        before = roads([(LineString([(0, 0), (100, 0)]), 6, "A")])
        after = roads([
            (LineString([(0, 0), (49, 0)]), 6, "A"),
            (LineString([(51, 0), (100, 0)]), 6, "A"),
        ])
        positive, negative, _summary = detect_changes(before, after)
        self.assertNotIn("added", set(positive["change_typ"]))
        self.assertNotIn("removed", set(negative["change_typ"]))

    def test_presence_case_b_shifted_junction_split_has_no_added_or_removed(self):
        before = roads([
            (LineString([(-50, 0), (0, 0)]), 6, "A"),
            (LineString([(0, 0), (50, 0)]), 6, "A"),
            (LineString([(0, -50), (0, 50)]), 6, "A"),
        ])
        after = roads([
            (LineString([(-50, 1), (2, 1)]), 6, "A"),
            (LineString([(2, 1), (50, 1)]), 6, "A"),
            (LineString([(2, -50), (2, 50)]), 6, "A"),
        ])
        positive, negative, _summary = detect_changes(before, after)
        self.assertNotIn("added", set(positive["change_typ"]))
        self.assertNotIn("removed", set(negative["change_typ"]))

    def test_presence_case_c_small_centerline_offset_has_no_added_or_removed(self):
        before = roads([(LineString([(0, 0), (100, 0)]), 6, "A")])
        after = roads([(LineString([(0, 2.5), (100, 2.5)]), 6, "A")])
        positive, negative, _summary = detect_changes(before, after)
        self.assertNotIn("added", set(positive["change_typ"]))
        self.assertNotIn("removed", set(negative["change_typ"]))

    def test_presence_case_d_true_added_road_has_surface_confirmation(self):
        existing = LineString([(0, 0), (100, 0)])
        new_road = LineString([(0, 30), (60, 30)])
        before = roads([(existing, 6, "A")])
        after = roads([(existing, 6, "A"), (new_road, 6, "A")])
        positive, negative, summary = detect_changes(
            before, after,
            before_surfaces=surfaces([existing], [6]),
            after_surfaces=surfaces([existing, new_road], [6, 6]),
        )
        added = positive.loc[
            (positive["change_typ"] == "added") & (positive["qa_state"] == "auto")
        ]
        self.assertEqual(len(added), 1)
        self.assertTrue(negative.empty)
        self.assertEqual(summary["added_feature_count"], 1)

    def test_presence_case_e_surface_residual_is_review_only(self):
        existing = LineString([(0, 0), (100, 0)])
        omitted_centerline = LineString([(0, 30), (60, 30)])
        before = roads([(existing, 6, "A")])
        after = roads([(existing, 6, "A"), (omitted_centerline, 6, "A")])
        both_surfaces = surfaces([existing, omitted_centerline], [6, 6])
        artifacts = {}
        positive, negative, unchanged, summary = _detect_changes_internal(
            before, after, DetectionConfig(), "before", "after",
            before_surfaces=both_surfaces, after_surfaces=both_surfaces,
            artifacts=artifacts,
        )
        review_added = positive.loc[
            (positive["change_typ"] == "added") & (positive["qa_state"] == "review")
        ]
        self.assertFalse(review_added.empty)
        self.assertTrue(review_added["audit_reason"].str.contains("surface_present").all())
        self.assertEqual(summary["added_feature_count"], 0)
        self.assertGreater(summary["review_added_feature_count"], 0)

        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            formal = _write_outputs(output, positive, negative, unchanged, after.crs, artifacts)
            self.assertTrue(formal.empty)
            self.assertTrue(gpd.read_file(output / "road_changes.shp").empty)
            self.assertTrue(gpd.read_file(output / "added_roads.shp").empty)
            review = gpd.read_file(output / "review_changes.shp")
            self.assertFalse(review.empty)
            self.assertEqual(set(review["qa_state"]), {"review"})

    def test_presence_case_f_true_removed_road_has_surface_confirmation(self):
        existing = LineString([(0, 0), (100, 0)])
        removed_road = LineString([(0, 30), (60, 30)])
        before = roads([(existing, 6, "A"), (removed_road, 6, "A")])
        after = roads([(existing, 6, "A")])
        positive, negative, summary = detect_changes(
            before, after,
            before_surfaces=surfaces([existing, removed_road], [6, 6]),
            after_surfaces=surfaces([existing], [6]),
        )
        removed = negative.loc[
            (negative["change_typ"] == "removed") & (negative["qa_state"] == "auto")
        ]
        self.assertEqual(len(removed), 1)
        self.assertTrue(positive.empty)
        self.assertEqual(summary["removed_feature_count"], 1)

    def test_extension_outputs_only_extended_part(self):
        before = roads([(LineString([(0, 0), (100, 0)]), 6, "A")])
        after = roads([(LineString([(0, 0), (120, 0)]), 6, "A")])
        positive, negative, _summary = detect_changes(before, after)
        added = positive.loc[positive["change_typ"] == "added"]
        self.assertFalse(added.empty)
        self.assertTrue(negative.empty)
        self.assertGreaterEqual(union_all(added.geometry.values).bounds[0], 99.0)

    def test_large_position_change_is_complete_added_and_removed(self):
        before = roads([(LineString([(0, 0), (100, 0)]), 6, "A")])
        after = roads([(LineString([(0, 12), (100, 12)]), 6, "A")])
        positive, negative, summary = detect_changes(before, after)
        self.assertEqual(set(positive["change_typ"]), {"added"})
        self.assertEqual(set(negative["change_typ"]), {"removed"})
        self.assertAlmostEqual(float(positive["axis_len_m"].sum()), 100.0, places=4)
        self.assertAlmostEqual(float(negative["axis_len_m"].sum()), 100.0, places=4)
        self.assertEqual(summary["width_changed_centerline_count"], 0)

    def test_position_change_never_outputs_width_change(self):
        before = roads([(LineString([(0, 0), (100, 0)]), 4, "A")])
        after = roads([(LineString([(0, 12), (100, 12)]), 10, "A")])
        positive, negative, _summary = detect_changes(before, after)
        self.assertNotIn("widened", set(positive["change_typ"]))
        self.assertNotIn("narrowed", set(negative["change_typ"]))

    def test_nodata_only_road_does_not_become_removed(self):
        before = roads([(LineString([(60, 0), (100, 0)]), 6, "A")])
        after = roads([])
        before_valid = gpd.GeoDataFrame(geometry=[box(0, -20, 120, 20)], crs="EPSG:3857")
        after_valid = gpd.GeoDataFrame(geometry=[box(0, -20, 50, 20)], crs="EPSG:3857")
        _positive, negative, summary = detect_changes(
            before, after, before_valid_area=before_valid, after_valid_area=after_valid,
        )
        self.assertFalse(negative.empty)
        self.assertEqual(set(negative["qa_state"]), {"review"})
        self.assertEqual(set(negative["after_state"]), {"uncertain"})
        self.assertEqual(set(negative["audit_reason"]), {"invalid_or_nodata_reference"})
        self.assertEqual(summary["removed_feature_count"], 0)
        self.assertTrue(summary["valid_observation_intersection_applied"])

    def test_valid_pixel_mask_is_exported_on_unicode_path(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "中文有效范围"
            root.mkdir()
            raster_path = root / "影像.tif"
            data = np.ones((10, 10), dtype=np.uint8)
            data[:, 5:] = 0
            with rasterio.open(
                raster_path, "w", driver="GTiff", width=10, height=10, count=1,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 10, 1, 1),
                nodata=0,
            ) as dataset:
                dataset.write(data, 1)
            output = root / "有效观测.shp"
            result = user_pipeline._write_valid_observation_area(root, output)
            self.assertEqual(Path(result), output.resolve())
            footprint = gpd.read_file(output)
            self.assertAlmostEqual(float(footprint.geometry.area.sum()), 50.0, places=4)

    def test_surface_skeleton_provenance_survives_fusion(self):
        line = LineString([(0, 0), (30, 0)])
        result = _fuse_centerline_records([
            {
                "tile_stem": "tile", "width_map": 6.0, "quality_grade": "B",
                "line_source": "surface_skeleton", "surface_conf": 0.9,
                "qa_state": "auto", "geometry": line,
            }
        ], {"tile": {"footprint": box(-10, -10, 40, 10), "feather_distance": 10.0}})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["line_source"], "surface_skeleton")

    def test_empty_change_writes_complete_products(self):
        line = LineString([(0, 0), (100, 0)])
        before = roads([(line, 6, "A")])
        after = roads([(line, 6, "A")])
        artifacts = {}
        added, removed, unchanged, _summary = _detect_changes_internal(
            before, after, DetectionConfig(), "2021", "2022", artifacts=artifacts,
        )
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            _write_outputs(output, added, removed, unchanged, after.crs, artifacts)
            for name in (
                "added_roads", "removed_roads", "widened_road_parts", "narrowed_road_parts",
                "road_changes", "review_changes", "road_width_segments", "road_corridors",
                "road_matches", "canonical_roads",
            ):
                for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    self.assertTrue((output / f"{name}{suffix}").is_file(), f"missing {name}{suffix}")
            self.assertTrue((output / "road_changes.gpkg").is_file())
            self.assertTrue(gpd.read_file(output / "road_changes.shp").empty)


if __name__ == "__main__":
    unittest.main()
