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
from engine.fast_auto_change import RoadScene, WindowedProbability, analyze_scenes


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

    def test_nodata_retains_probable_presence_but_prevents_width(self):
        before = self.scene([self.road(70, 8)], valid=box(0, 90, 300, 250))
        after = self.scene([self.road(70, 14), self.road(30)])
        changes, audit, _, _ = analyze_scenes(before, after)
        self.assertEqual([r["change_typ"] for r in changes], ["added"])
        self.assertEqual(changes[0]["qa_state"], "probable")
        self.assertIn("invalid_or_boundary", changes[0]["audit_reason"])
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

    def test_junction_small_change_is_suppressed(self):
        roads = [self.road(100), (LineString([(150, 30), (150, 190)]), 8)]
        before = self.scene(roads)
        surfaces = [line.buffer(width/2, cap_style="flat") for line, width in roads] + [box(140, 90, 160, 110)]
        after = self.scene(roads, surfaces=surfaces)
        changes, _, _, _ = analyze_scenes(before, after)
        self.assertFalse(changes)

    def test_formal_entry_publishes_vectors_funnel_and_preview(self):
        from engine.fast_pipeline import detect_fast_changes
        from engine import fast_auto_change as baseline
        from unittest.mock import patch
        # Guard the actual public entry and posterior GT chain, not just V2's
        # direct analyzer. Fast1 stays independently callable in baseline tests.
        for owner, name in ((baseline, 'analyze_scenes'), (RoadScene, 'match'),
                            (baseline, '_measure_period_width')):
            guard = patch.object(owner, name, side_effect=AssertionError('production called Fast1'))
            guard.start()
            self.addCleanup(guard.stop)
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
        self.assertTrue(result['performance']['v2_enabled'])
        self.assertEqual(result['performance']['v2_station_count'], 0)
        # This fixture has model rasters only. Raw-image-primary verification
        # must preserve the incoming Fast2 decision as uncertain, not reject it.
        self.assertEqual(result["added_feature_count"], 1)
        import json
        patch_audit=json.loads((output/'patch_verification.json').read_text(encoding='utf8'))
        self.assertTrue(patch_audit['candidates'])
        self.assertTrue(all(row['state']=='uncertain' and row['published'] and
                            'raw_image_unavailable_or_invalid' in row['reasons']
                            for row in patch_audit['candidates']))
        self.assertEqual(result["removed_feature_count"], 0)
        for key in ("road_changes", "summary", "road_change"):
            self.assertTrue(Path(result[key]).is_file(), key)
        self.assertEqual(result['diagnostics'], '')
        self.assertEqual(set(result['layers']), {'changes'})
        self.assertFalse(list(output.glob('*.gpkg')))
        self.assertFalse(list(output.glob('*.csv')))
        self.assertEqual([p.name for p in output.glob('*.shp')], ['road_changes.shp'])
        self.assertFalse((output/'candidate_funnel.json').exists())
        self.assertTrue(gpd.read_file(result["road_changes"]).is_valid.all())

        # Rerunning for GT retains only its two required internal layers.
        for name in ('auto_diagnostics.gpkg', 'existence_candidates.csv', 'assembly_summary.json'):
            (output/name).touch()
        result = detect_fast_changes(*inputs, output, internal_outputs=True)
        self.assertEqual(set(gpd.list_layers(output/'network_assembly.gpkg').name),
                         {'change_objects', 'object_axes'})
        self.assertFalse((output/'auto_diagnostics.gpkg').exists())
        self.assertFalse(list(output.glob('*.csv')))
        import user_pipeline
        from engine.fast_multitemporal import AUTO_REVISION
        truth = Path(self.tmp.name)/'truth.gpkg'
        gpd.GeoDataFrame({'BHBM': [2]}, geometry=[self.road(170)[0].buffer(4)],
                         crs=self.crs).to_file(truth)
        corrected = user_pipeline._run_fast_change_result(*inputs,
                            Path(self.tmp.name)/'corrected', truth_path=truth,
                            before_period='1', after_period='2', position_tolerance=3.,
                            width_change_absolute=2., width_change_ratio=.2, defer_finalization=True)
        self.assertTrue(corrected['ground_truth_used'])
        self.assertEqual(corrected['fast_auto_revision'], AUTO_REVISION)
        self.assertEqual(corrected['fast_finalization_state'], 'pending')
        auto_summary = json.loads((Path(self.tmp.name)/'corrected/_automatic/change_summary.json').read_text(encoding='utf8'))
        self.assertTrue(auto_summary['performance']['v2_enabled'])
        self.assertFalse(auto_summary['ground_truth_used'])
        self.assertIn('corrected_intervals', set(gpd.list_layers(corrected['correction_audit']).name))
        detect_fast_changes(*inputs, output, internal_outputs=False)
        self.assertFalse((output/'network_assembly.gpkg').exists())

        # The same upstream addition without independent source surface support
        # stays available internally as Candidate, even with diagnostics disabled.
        weak_after=dict(inputs[1],surfaces=inputs[0]['surfaces'])
        weak_output=Path(self.tmp.name)/'weak_auto'
        weak=detect_fast_changes(inputs[0],weak_after,weak_output)
        self.assertEqual(weak['added_feature_count'],0)
        weak_audit=json.loads((weak_output/'patch_verification.json').read_text(encoding='utf8'))
        candidates=[r for r in weak_audit['candidates'] if r['publication_level']=='Candidate']
        self.assertTrue(candidates)
        from shapely import from_wkb
        self.assertTrue(all(from_wkb(bytes.fromhex(r['candidate_geometry_wkb'])).is_valid for r in candidates))
        self.assertTrue(all(r['state']=='uncertain' and not r['published'] for r in candidates))

    def test_empty_formal_result_still_publishes_funnel_and_audits(self):
        from engine.fast_auto_change import finalize_auto_candidates
        scenes = {period: self.scene([self.road(100)]) for period in ("before", "after")}
        output = Path(self.tmp.name)/"empty_auto"
        result = finalize_auto_candidates([], [], [], {}, presence_audit=[], scenes=scenes,
                    centerlines=[scene.widths for scene in scenes.values()], output_dir=output,
                    before_period="before", after_period="after")
        self.assertEqual(result['changes_feature_count'], 0)
        self.assertEqual(result['candidate_funnel'], '')
        self.assertFalse(list(output.glob('*.gpkg')))


if __name__ == "__main__":
    unittest.main()
