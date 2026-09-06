import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.fast_auto_change import RoadScene, WindowedProbability, analyze_scenes, _runs


class FastFinalAutoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.crs = "EPSG:32650"
        self.transform = from_origin(0, 250, .5, .5)
        self.counter = 0

    def scene(self, roads, *, surfaces=None, probability_roads=None, valid=None, low=.03, high=.8):
        self.counter += 1
        geometries = [line for line, width in roads]
        width_values = [width for line, width in roads]
        lines = gpd.GeoDataFrame({"width_m": width_values}, geometry=geometries, crs=self.crs)
        surfaces = [line.buffer(width/2, cap_style="flat") for line, width in roads] if surfaces is None else surfaces
        probabilities = surfaces if probability_roads is None else probability_roads
        mask = rasterize([(g, 1) for g in probabilities], out_shape=(500, 600), transform=self.transform) if probabilities else np.zeros((500, 600))
        array = np.where(mask, high, low).astype("float32")
        path = Path(self.tmp.name)/f"p{self.counter}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=600, height=500, count=1,
                           dtype="float32", transform=self.transform, crs=self.crs) as dataset:
            dataset.write(array, 1)
        probability = WindowedProbability(path, self.crs)
        self.addCleanup(probability.close)
        return RoadScene(lines, gpd.GeoDataFrame(geometry=surfaces, crs=self.crs), lines,
                         gpd.GeoDataFrame(geometry=[valid if valid is not None else box(0, 0, 300, 250)], crs=self.crs),
                         probability, self.crs)

    @staticmethod
    def road(y, width=8, start=30, end=260):
        return LineString([(start, y), (end, y)]), width

    def test_shift_and_segmentation_are_unchanged(self):
        before = self.scene([self.road(100)])
        after = self.scene([self.road(102, start=30, end=130), self.road(102, start=130, end=260)])
        changes, audit, _, _ = analyze_scenes(before, after)
        self.assertFalse(changes)
        self.assertTrue(any(row["matched"] for row in audit))

    def test_missing_axis_with_surface_or_probability_is_not_removed(self):
        before = self.scene([self.road(100), self.road(160)])
        target_surfaces = [line.buffer(width/2) for line, width in [self.road(100), self.road(160)]]
        after = self.scene([self.road(160)], surfaces=target_surfaces)
        changes, audit, _, _ = analyze_scenes(before, after)
        self.assertFalse(changes)
        self.assertTrue(any(row["after_reason"] == "surface_without_centerline" for row in audit))
        after = self.scene([self.road(160)], probability_roads=target_surfaces)
        changes, audit, _, _ = analyze_scenes(before, after)
        self.assertFalse(changes)
        self.assertTrue(any(row["after_reason"] == "probability_without_centerline" for row in audit))

    def test_added_removed_and_time_reversal(self):
        before = self.scene([self.road(60), self.road(120)])
        after = self.scene([self.road(120), self.road(180)], low=.15, high=.65)
        changes, _, _, _ = analyze_scenes(before, after)
        self.assertCountEqual([c["change_typ"] for c in changes], ["added", "removed"])
        reverse, _, _, _ = analyze_scenes(after, before)
        self.assertCountEqual([c["change_typ"] for c in reverse], ["added", "removed"])
        self.assertAlmostEqual(sum(c["geometry"].area for c in changes), sum(c["geometry"].area for c in reverse))

    def test_local_widening_and_narrowing(self):
        before = self.scene([self.road(70, 8), self.road(160, 14)])
        after = self.scene([self.road(70, 14), self.road(160, 8)])
        changes, _, _, _ = analyze_scenes(before, after)
        self.assertCountEqual([c["change_typ"] for c in changes], ["widened", "narrowed"])

    def test_partial_overlap_detects_only_extension(self):
        before = self.scene([self.road(100, end=140)])
        after = self.scene([self.road(100)])
        changes, _, _, _ = analyze_scenes(before, after)
        self.assertEqual([c["change_typ"] for c in changes], ["added"])
        self.assertGreater(changes[0]["geometry"].bounds[0], 140)

    def test_width_change_stays_on_local_run(self):
        before = self.scene([self.road(100)])
        after = self.scene([self.road(100)], surfaces=[box(30, 96, 150, 104), box(150, 93, 260, 107)])
        changes, _, _, _ = analyze_scenes(before, after)
        self.assertEqual([c["change_typ"] for c in changes], ["widened"])
        self.assertGreater(changes[0]["geometry"].bounds[0], 145)
        self.assertLess(changes[0]["length_m"], 125)

    def test_nodata_prevents_presence_and_width(self):
        before = self.scene([self.road(70, 8)], valid=box(0, 90, 300, 250))
        after = self.scene([self.road(70, 14), self.road(30)])
        changes, audit, _, _ = analyze_scenes(before, after)
        self.assertFalse(changes)
        self.assertTrue(any(row["before_state"] == "uncertain" for row in audit))

    def test_parallel_tracks_do_not_swap(self):
        before = self.scene([self.road(100, 5), self.road(108, 8)])
        after = self.scene([self.road(102, 5), self.road(110, 8)])
        changes, audit, _, _ = analyze_scenes(before, after)
        self.assertFalse(changes)
        self.assertTrue(all(row["offset_m"] < 2.01 for row in audit if row["matched"]))

    def test_noisy_short_changes_and_invalid_gap(self):
        before = self.scene([self.road(100)])
        surface = [box(30+10*i, 100-(8 if i%2 else 14)/2,
                       min(260, 40+10*i), 100+(8 if i%2 else 14)/2) for i in range(23)]
        after = self.scene([self.road(100)], surfaces=surface)
        changes, _, _, _ = analyze_scenes(before, after)
        self.assertFalse(changes)
        self.assertEqual(list(_runs(["added", "", "added"], [True, False, True], 4)),
                         [(0, 0, "added"), (2, 2, "added")])

    def test_junction_small_change_is_suppressed(self):
        roads = [self.road(100), (LineString([(150, 30), (150, 190)]), 8)]
        before = self.scene(roads)
        surfaces = [line.buffer(width/2, cap_style="flat") for line, width in roads] + [box(140, 90, 160, 110)]
        after = self.scene(roads, surfaces=surfaces)
        changes, _, _, _ = analyze_scenes(before, after)
        self.assertFalse(changes)

    def test_formal_entry_publishes_vectors_funnel_and_preview(self):
        from engine.fast_pipeline import detect_fast_changes
        scenes = [self.scene([self.road(100)]), self.scene([self.road(100), self.road(170)])]
        inputs = []
        for index, scene in enumerate(scenes):
            directory = Path(self.tmp.name)/f"period{index}"
            directory.mkdir()
            frames = {"centerlines": scene.widths, "surfaces": scene.surfaces,
                      "width_segments": scene.widths,
                      "valid_observation": gpd.GeoDataFrame(geometry=[scene.valid], crs=self.crs)}
            payload = {"road_probability": scene.probability.dataset.name}
            for key, frame in frames.items():
                path = directory/f"{key}.shp"
                frame.to_file(path)
                payload[key] = str(path)
            inputs.append(payload)
        output = Path(self.tmp.name)/"auto"
        result = detect_fast_changes(*inputs, output)
        self.assertFalse(result["ground_truth_used"])
        self.assertEqual(result["added_feature_count"], 1)
        self.assertEqual(result["removed_feature_count"], 0)
        for key in ("road_changes", "candidate_funnel", "summary", "road_change"):
            self.assertTrue(Path(result[key]).is_file(), key)
        self.assertEqual(len(gpd.read_file(result["diagnostics"], layer="changes")), 1)
        self.assertTrue(gpd.read_file(result["road_changes"]).is_valid.all())


if __name__ == "__main__":
    unittest.main()
