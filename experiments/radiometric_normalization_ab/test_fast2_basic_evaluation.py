import unittest
import geopandas as gpd
from shapely.geometry import box
from evaluate_fast2_basic import compare


class OfflineEvaluationTests(unittest.TestCase):
    def test_touching_is_zero_area_and_wrong_type_is_reported_separately(self):
        pred=gpd.GeoDataFrame(dict(interval_id=[0,1],change_typ=['removed','widened'],length_m=[2.,2.]),
            geometry=[box(0,0,2,2),box(3,0,5,2)],crs='EPSG:32650')
        gt=gpd.GeoDataFrame(dict(evaluation_type=['added']),geometry=[box(2,0,4,2)],crs=pred.crs)
        table,result,_=compare(pred,gt)
        self.assertFalse(table.iloc[0].has_GT_overlap)
        self.assertTrue(table.iloc[1].has_GT_overlap)
        self.assertFalse(table.iloc[1].has_same_type_GT_overlap)
        self.assertEqual(result['no_GT_intersection_count'],1)
        self.assertEqual(result['typed_GT_area_coverage'],0.)


if __name__=='__main__':unittest.main()
