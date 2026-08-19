from __future__ import annotations

import sys
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely import union_all
from shapely.geometry import LineString, box


WIDTH = Path(__file__).resolve().parent / "engine" / "width"
if str(WIDTH) not in sys.path:
    sys.path.insert(0, str(WIDTH))

from road_change_detection import detect_changes  # noqa: E402
from road_existence_evidence import RoadProbabilityRaster  # noqa: E402


CRS = "EPSG:3857"
EXISTING = LineString([(0, 0), (100, 0)])
CHANGED = LineString([(0, 30), (100, 30)])


def roads(lines):
    return gpd.GeoDataFrame(
        {"width_map": [6.0] * len(lines), "quality_gr": ["A"] * len(lines)},
        geometry=lines, crs=CRS,
    )


def surfaces(lines):
    return gpd.GeoDataFrame(
        geometry=[line.buffer(3.0, cap_style="flat") for line in lines], crs=CRS,
    )


def valid(geometry=box(-20, -20, 130, 60)):
    return gpd.GeoDataFrame(geometry=[geometry], crs=CRS)


def probability(lines=(), scale=1.0, noise=0.0):
    values = np.zeros((120, 160), dtype=np.float32)
    transform = from_origin(-20, 80, 1, 1)
    for line in lines:
        y = int(round((80 - float(line.coords[0][1]))))
        values[max(0, y - 2):min(values.shape[0], y + 3), 20:121] = scale
    if noise:
        rng = np.random.default_rng(7)
        values = np.clip(values + rng.normal(0, noise, values.shape), 0, 1)
    return RoadProbabilityRaster(values, transform, CRS)


class CrossSensorExistenceEvidenceTests(unittest.TestCase):
    def test_case_1_same_imagery_different_segmentation_has_zero_auto_change(self):
        before = roads([EXISTING])
        after = roads([LineString([(0, 0), (48, 0)]), LineString([(52, 0), (100, 0)])])
        positive, negative, summary = detect_changes(
            before, after, before_surfaces=surfaces([EXISTING]),
            after_surfaces=surfaces([EXISTING]),
        )
        self.assertEqual(summary["added_feature_count"], 0)
        self.assertEqual(summary["removed_feature_count"], 0)
        self.assertFalse(any(positive.get("qa_state", []) == "auto"))
        self.assertFalse(any(negative.get("qa_state", []) == "auto"))

    def test_case_2_missing_centerline_probability_present_is_not_removed(self):
        partial = LineString([(0, 0), (35, 0)])
        _positive, negative, summary = detect_changes(
            roads([EXISTING]), roads([partial]),
            before_surfaces=surfaces([EXISTING]), after_surfaces=surfaces([partial]),
            before_probability=probability([EXISTING]), after_probability=probability([EXISTING], 0.25),
        )
        self.assertEqual(summary["removed_feature_count"], 0)
        self.assertTrue((negative["qa_state"] == "review").all())
        self.assertTrue(negative["audit_reason"].str.contains("probability_present").all())

    def test_case_3_missing_centerline_surface_present_is_review(self):
        _positive, negative, summary = detect_changes(
            roads([EXISTING]), roads([]),
            before_surfaces=surfaces([EXISTING]), after_surfaces=surfaces([EXISTING]),
            before_valid_area=valid(), after_valid_area=valid(),
        )
        self.assertEqual(summary["removed_feature_count"], 0)
        self.assertEqual(set(negative["after_state"]), {"present"})
        self.assertTrue(negative["audit_reason"].str.contains("surface_present").all())

    def test_case_4_multiple_negative_evidence_confirms_removed(self):
        unrelated = LineString([(0, 55), (100, 55)])
        _positive, negative, summary = detect_changes(
            roads([EXISTING]), roads([]),
            before_surfaces=surfaces([EXISTING]), after_surfaces=surfaces([unrelated]),
            before_valid_area=valid(), after_valid_area=valid(),
            before_probability=probability([EXISTING]), after_probability=probability([]),
        )
        formal = negative.loc[negative["qa_state"] == "auto"]
        self.assertEqual(summary["removed_feature_count"], 1)
        self.assertEqual(set(formal["after_state"]), {"absent"})

    def test_case_5_geometry_probability_surface_support_added(self):
        unrelated = LineString([(0, 55), (100, 55)])
        positive, _negative, summary = detect_changes(
            roads([]), roads([CHANGED]),
            before_surfaces=surfaces([unrelated]), after_surfaces=surfaces([CHANGED]),
            before_valid_area=valid(), after_valid_area=valid(),
            before_probability=probability([]), after_probability=probability([CHANGED], 0.22),
        )
        formal = positive.loc[positive["qa_state"] == "auto"]
        self.assertEqual(summary["added_feature_count"], 1)
        self.assertEqual(set(formal["before_state"]), {"absent"})
        self.assertEqual(set(formal["after_state"]), {"present"})

    def test_case_6_nodata_reference_is_uncertain_not_removed(self):
        _positive, negative, summary = detect_changes(
            roads([EXISTING]), roads([]),
            before_surfaces=surfaces([EXISTING]), after_surfaces=surfaces([LineString([(0, 55), (100, 55)])]),
            before_valid_area=valid(), after_valid_area=valid(box(-20, 20, 130, 60)),
        )
        self.assertEqual(summary["removed_feature_count"], 0)
        self.assertEqual(set(negative["after_state"]), {"uncertain"})
        self.assertEqual(set(negative["audit_reason"]), {"invalid_or_nodata_reference"})

    def test_case_7_true_added_polygon_survives_sensor_degradation(self):
        before = roads([EXISTING])
        after = roads([EXISTING, CHANGED])
        positive, _negative, summary = detect_changes(
            before, after,
            before_surfaces=surfaces([EXISTING]), after_surfaces=surfaces([EXISTING, CHANGED]),
            before_valid_area=valid(), after_valid_area=valid(),
            before_probability=probability([EXISTING]),
            after_probability=probability([EXISTING, CHANGED], 0.20, 0.01),
        )
        truth = CHANGED.buffer(3.0, cap_style="flat")
        formal = positive.loc[(positive["change_typ"] == "added") & (positive["qa_state"] == "auto")]
        self.assertGreater(float(union_all(formal.geometry.values).intersection(truth).area), 0.0)
        self.assertEqual(summary["added_feature_count"], 1)

    def test_case_8_true_removed_polygon_survives_sensor_degradation(self):
        before = roads([EXISTING, CHANGED])
        after = roads([EXISTING])
        _positive, negative, summary = detect_changes(
            before, after,
            before_surfaces=surfaces([EXISTING, CHANGED]), after_surfaces=surfaces([EXISTING]),
            before_valid_area=valid(), after_valid_area=valid(),
            before_probability=probability([EXISTING, CHANGED]),
            after_probability=probability([EXISTING], 0.20, 0.01),
        )
        truth = CHANGED.buffer(3.0, cap_style="flat")
        formal = negative.loc[(negative["change_typ"] == "removed") & (negative["qa_state"] == "auto")]
        self.assertGreater(float(union_all(formal.geometry.values).intersection(truth).area), 0.0)
        self.assertEqual(summary["removed_feature_count"], 1)


if __name__ == "__main__":
    unittest.main()
