"""Known-geometry tests for the isolated experiment; no models or production imports."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from boundary_width import Config, ImageReader, measure_road, optimize_chain, runs, states_for_row


class BoundaryWidthTests(unittest.TestCase):
    def test_surface_uses_asymmetric_boundaries_and_does_not_bridge_rejection(self):
        import pandas as pd
        from export_width_surfaces import build_surfaces
        rows=[]
        for i in range(4):
            left,right=2+i,6+i
            rows.append(dict(road_id='road_000',sample_id=i,s_m=i*3,accepted=i!=2,
                optimized_left_x=i*3,optimized_left_y=left,optimized_right_x=i*3,optimized_right_y=-right,
                optimized_left_distance=left,optimized_right_distance=right,
                optimized_width=left+right,optimized_confidence=.8))
        polygons,invalid=build_surfaces(pd.DataFrame(rows),'EPSG:32650')
        self.assertEqual(len(polygons),1)
        self.assertEqual(invalid,[])
        self.assertEqual(polygons.iloc[0].width,9)
        self.assertAlmostEqual(polygons.geometry.iloc[0].area,27)
        self.assertEqual(polygons.geometry.iloc[0].bounds,(0.,-7.,3.,3.))

    def test_joint_dp_matches_exhaustive_and_suppresses_transient_edge(self):
        c = Config(position_weight=1, width_weight=1)
        states = [(np.array([[0, 1], [2, 3]]), np.array([[3., 7.], [3., 17.]]),
                   np.array([1.4, 1.8 if i == 2 else .2])) for i in range(5)]
        selected = optimize_chain(states, 3, c)
        np.testing.assert_array_equal(selected, np.zeros(5, int))
        from boundary_width import huber
        def objective(path):
            cost = -sum(states[i][2][k] for i, k in enumerate(path))
            for i in range(1, len(path)):
                a, b = states[i-1][1][path[i-1]], states[i][1][path[i]]
                cost += huber((b-a)/3).sum()+huber((b.sum()-a.sum())/3)
            return cost
        expected = min(itertools.product(range(2), repeat=5), key=objective)
        self.assertEqual(tuple(selected), expected)

    def test_gradual_asymmetric_widening_is_preserved(self):
        c = Config()
        states = []
        for i in range(40):
            states.append((np.array([[0, 1], [2, 3]]), np.array([[2., 5.+i*.12], [2., 5.]]), np.array([1.8, 1.])))
        selected = optimize_chain(states, 3, c)
        np.testing.assert_array_equal(selected, np.zeros(40, int))
        self.assertGreater(states[-1][1][selected[-1]].sum()-states[0][1][selected[0]].sum(), 4)

    def test_crossing_pairs_are_impossible_and_width_is_sum(self):
        c = Config()
        offsets = np.arange(-20, 20.5, .5)
        score = np.sin(offsets)**2
        ids, distances, _ = states_for_row(offsets, score, c)
        self.assertTrue((offsets[ids[:, 0]] > 0).all())
        self.assertTrue((offsets[ids[:, 1]] < 0).all())
        self.assertTrue((distances.sum(1) <= c.max_width).all())

    def test_low_confidence_breaks_runs(self):
        self.assertEqual(list(runs(np.array([1, 1, 0, 1, 0, 1, 1], bool))), [(0, 2), (3, 4), (5, 7)])

    def test_image_known_width_offcenter_nodata_and_flat_region(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'synthetic.tif'
            rgb = np.full((3, 240, 300), 35, np.uint8)
            # Physical road y=40..50, width=10; centerline y=43 is 3/7 m offcenter.
            rgb[:, 140:160, :] = 165
            rgb[:, :, 115:125] = 0  # Nodata breaks the optimization chain.
            with rasterio.open(path, 'w', driver='GTiff', width=300, height=240, count=3,
                               dtype='uint8', crs='EPSG:32650', transform=from_origin(400000, 120, .5, .5), nodata=0) as ds:
                ds.write(rgb)
            reader = ImageReader(path, 'EPSG:32650')
            c = Config()
            data = measure_road(reader, LineString([(400020, 43), (400130, 43)]), None, c)
            good = np.array([not f for f in data['flags']])
            self.assertGreater(good.sum(), 20)
            np.testing.assert_allclose(np.median(data['optimized'][good], axis=0), [7, 3], atol=.6)
            self.assertTrue(any('image_or_nodata_boundary' in f for f in data['flags']))
            flat = measure_road(reader, LineString([(400020, 85), (400130, 85)]), None, c)
            self.assertTrue(all(flat['flags']))
            limit = measure_road(reader, LineString([(400020, 42), (400130, 42)]), None, Config(search_radius=8))
            self.assertTrue(any('search_limit' in f for f in limit['flags']))
            reader.ds.close()


if __name__ == '__main__':
    unittest.main()
