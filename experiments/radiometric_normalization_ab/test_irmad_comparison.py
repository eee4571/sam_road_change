import unittest
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from compare_irmad_roads import stations, match, overlap_counts


class ComparisonTests(unittest.TestCase):
    def test_parallel_offsets_and_perpendicular_rejection(self):
        reference=gpd.GeoDataFrame(geometry=[LineString([(0,0),(100,0)])],crs='EPSG:32650')
        sample=stations(reference)
        self.assertAlmostEqual(sample['weights'].sum(),100)
        target=gpd.GeoDataFrame(geometry=[LineString([(0,2),(100,2)]),LineString([(49,-10),(49,10)])],crs=reference.crs)
        matched=match(sample,target)
        np.testing.assert_allclose(matched['distance'],2)
        np.testing.assert_allclose(matched['lateral'],2)
        self.assertTrue((matched['index']==0).all())

    def test_invalid_pixels_do_not_contribute_overlap(self):
        a=np.array([1,1,0],dtype=bool)
        b=np.array([1,0,1],dtype=bool)
        np.testing.assert_equal(overlap_counts(a,b,np.array([0,1,1],dtype=bool)),[1,1,0,2])


if __name__=='__main__': unittest.main()
