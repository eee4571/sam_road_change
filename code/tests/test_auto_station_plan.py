"""Synthetic candidate planning versus full scalar station evaluation."""
import time
import unittest
from unittest.mock import patch
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box
import test_fast_auto_change as fixtures
from engine.fast_auto_change import analyze_scenes, RoadScene, _normal, _normals
from engine.auto_station_plan import BatchSections
from engine.auto_presence_candidates import qualify_presence_candidates


class StationPlanTests(unittest.TestCase):
    setUp = fixtures.FastFinalAutoTests.setUp
    scene = fixtures.FastFinalAutoTests.scene
    road = staticmethod(fixtures.FastFinalAutoTests.road)

    def reference(self, before, after):
        with patch('engine.auto_station_plan.BatchSections', side_effect=lambda sampler, *args: sampler), \
             patch.object(RoadScene, 'widths_at', lambda scene, points: np.array([scene.width(p) for p in points])), \
             patch('engine.fast_auto_change._normals', side_effect=lambda axis, stations: np.array([_normal(axis, s) for s in stations])):
            return analyze_scenes(before, after, candidate_driven=False)

    def assert_candidates_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for a, b in zip(expected, actual):
            self.assertEqual(set(a), set(b))
            for key in a:
                if key == 'geometry':
                    self.assertLess(a[key].symmetric_difference(b[key]).area, 1e-7)
                elif key == 'axis_wkt':
                    from shapely import from_wkt
                    self.assertLess(from_wkt(a[key]).hausdorff_distance(from_wkt(b[key])), 1e-8)
                elif isinstance(a[key], (float, np.floating)):
                    self.assertAlmostEqual(a[key], b[key], places=7, msg=key)
                else:
                    self.assertEqual(a[key], b[key], key)

    def test_scalar_and_candidate_results_and_presence_qualification_agree(self):
        cases = [
            (self.scene([self.road(70)]), self.scene([self.road(70), self.road(170)])),
            (self.scene([self.road(100, end=140)]), self.scene([self.road(100)])),
            (self.scene([self.road(70, 8), self.road(170, 14)]),
             self.scene([self.road(70, 14), self.road(170, 8)])),
            # Identical stored widths must not hide different measured surfaces.
            (self.scene([self.road(100)]), self.scene([self.road(100)],
                surfaces=[box(30, 96, 150, 104), box(150, 93, 260, 107)])),
            (self.scene([self.road(100, 5), self.road(108, 8)]),
             self.scene([self.road(102, 5), self.road(110, 8)])),
            (self.scene([self.road(70)], valid=box(0,90,300,250)),
             self.scene([self.road(70,14), self.road(30)])),
            (self.scene([self.road(100)]), self.scene([self.road(100)], low=.2, high=.65)),
        ]
        for before, after in cases:
            for source, target in ((before, after), (after, before)):
                with self.subTest(before=source.probability.dataset.name, after=target.probability.dataset.name):
                    old = self.reference(source, target)
                    new = analyze_scenes(source, target)
                    self.assert_candidates_equal(old[0], new[0])
                    if old[0]:
                        qualified = []
                        for records, audit, _, _ in (old, new):
                            frame = gpd.GeoDataFrame(records, crs=self.crs)
                            if 'qa_state' not in frame: frame['qa_state'] = 'confirmed'
                            else: frame['qa_state'] = frame.qa_state.fillna('confirmed')
                            _, details = qualify_presence_candidates(frame, {'before':source,'after':target},
                                gpd.GeoDataFrame(audit, crs=self.crs), minimum_length=24., minimum_area=4.)
                            qualified.append(details)
                        self.assertEqual(qualified[0].publication_state.tolist(), qualified[1].publication_state.tolist())
                        self.assertEqual(qualified[0].precision_reason.tolist(), qualified[1].precision_reason.tolist())

    def test_stable_scene_skips_exact_stations(self):
        roads = [self.road(y) for y in (50,100,150,200)]
        before, after = self.scene(roads), self.scene(roads)
        started = time.perf_counter(); old = self.reference(before, after); scalar = time.perf_counter()-started
        started = time.perf_counter(); new = analyze_scenes(before, after); planned = time.perf_counter()-started
        self.assert_candidates_equal(old[0], new[0])
        self.assertEqual(new[3]['candidate_station_count'], 0)
        self.assertGreater(new[3]['total_station_count'], 400)
        print(f'[Synthetic performance] scalar={scalar:.4f}s candidate={planned:.4f}s '
              f"stations={new[3]['candidate_station_count']}/{new[3]['total_station_count']}")

    def test_batch_cross_sections_and_normals_match_scalar(self):
        axis = LineString([(30,80),(120,80),(150,110),(260,130)])
        scene = self.scene([(axis,8)])
        stations = np.linspace(0,axis.length,600)
        points = [axis.interpolate(s) for s in stations]
        normals = _normals(axis, stations)
        np.testing.assert_allclose(normals, [_normal(axis,s) for s in stations], atol=1e-14)
        batched = BatchSections(scene.probability, points, normals, 60.)
        for point, normal in zip(points, normals):
            old = scene.probability.sample_cross_section(point, normal, self.crs, search_radius=60.)
            new = batched.sample_cross_section(point, normal, self.crs, search_radius=60.)
            np.testing.assert_equal(old['probability'],new['probability'])
        np.testing.assert_equal(scene.widths_at(np.array(points)), [scene.width(p) for p in points])

    def test_partial_width_blocks_preserve_change_boundaries(self):
        from rasterio.transform import from_origin
        self.transform = from_origin(0, 1000, 2, 2)
        roads = [(LineString([(50,y),(1100,y)]),w) for y,w in ((300,8),(700,16))]
        valid = box(0,0,1200,1000)
        before = self.scene(roads, valid=valid)
        surfaces = [box(50,y-w/2,600,y+w/2) for y,w in ((300,8),(700,16))]
        surfaces += [box(600,y-w/2,1100,y+w/2) for y,w in ((300,16),(700,8))]
        after = self.scene(roads, surfaces=surfaces, valid=valid)
        old = self.reference(before, after)
        new = analyze_scenes(before, after)
        self.assert_candidates_equal(old[0], new[0])
        self.assertTrue(old[0])
        self.assertGreater(new[3]['width_prefilter_skipped_station_count'], 0)
        self.assertLess(new[3]['candidate_station_count'],old[3]['candidate_station_count'])

    def test_missing_width_quality_falls_back(self):
        before, after = self.scene([self.road(100)]), self.scene([self.road(100)])
        before.width_values = np.full(len(before.width_values), np.nan)
        result = analyze_scenes(before, after)
        self.assertEqual(result[3]['width_prefilter_skipped_station_count'], 0)
        self.assertGreater(result[3]['candidate_station_count'], 0)
        self.assert_candidates_equal(self.reference(before,after)[0], result[0])


if __name__ == '__main__':
    unittest.main()
