import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, box
from evaluate import line_comparison, surface_iou, sample_lines, clipped

class MetricTests(unittest.TestCase):
    def test_translation_is_measured_in_meters(self):
        a=gpd.GeoDataFrame(geometry=[LineString([(0,0),(100,0)])],crs=32650)
        b=gpd.GeoDataFrame(geometry=[LineString([(0,2),(100,2)])],crs=32650)
        r=line_comparison(a,b)
        self.assertEqual(r['before_to_after']['offset_m']['median'],2)
        self.assertEqual(r['before_to_after']['overlap_within_1m'],0)
        self.assertEqual(r['symmetric_overlap_3m'],1)
    def test_surface_union_not_feature_count(self):
        a=gpd.GeoDataFrame(geometry=[box(0,0,10,10),box(0,0,10,10)],crs=32650)
        b=gpd.GeoDataFrame(geometry=[box(5,0,15,10)],crs=32650)
        r=surface_iou(a,b)
        self.assertAlmostEqual(r['iou'],1/3)
        self.assertEqual(r['before_area_m2'],100)
    def test_uniform_length_sampling(self):
        points=sample_lines([LineString([(0,0),(100,0)])])
        self.assertEqual(len(points),50)
        np.testing.assert_allclose([p.x for p in points],np.arange(1,100,2))
    def test_clipping_keeps_interior_and_removes_nodata_hole(self):
        region=box(0,0,10,10).difference(box(4,4,6,6))
        frame=gpd.GeoDataFrame(geometry=[LineString([(0,5),(10,5)]),
            LineString([(1,1),(2,1)]),LineString([(20,20),(30,20)])],crs=32650)
        result=clipped(frame,region)
        self.assertEqual(len(result),2)
        np.testing.assert_allclose(result.length,[8,1])

if __name__=='__main__':unittest.main()
