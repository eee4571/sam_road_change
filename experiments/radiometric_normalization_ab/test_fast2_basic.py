import unittest
import geopandas as gpd
from shapely.geometry import LineString,box
from fast2_basic import RoadScene,analyze


def scene(line,width):
    roads=gpd.GeoDataFrame(geometry=[line],crs='EPSG:32650')
    surfaces=gpd.GeoDataFrame(geometry=[line.buffer(width/2)],crs=roads.crs)
    widths=gpd.GeoDataFrame({'width_m':[width]},geometry=[line],crs=roads.crs)
    valid=gpd.GeoDataFrame(geometry=[box(-100,-100,1000,1000)],crs=roads.crs)
    return RoadScene(roads,surfaces,widths,valid,None,roads.crs)


class BasicTests(unittest.TestCase):
    def test_identical_roads_stable_and_width_changes_are_not_bias_corrected(self):
        line=LineString([(0,0),(200,0)])
        a=scene(line,6)
        self.assertEqual(analyze(a,scene(line,6))[0],[])
        wider=analyze(a,scene(line,9))[0]
        self.assertEqual([r['change_typ'] for r in wider],['widened'])
        self.assertAlmostEqual(wider[0]['width_diff'],3)
        self.assertAlmostEqual(wider[0]['length_m'],200)
        narrow=analyze(scene(line,9),a)[0]
        self.assertEqual([r['change_typ'] for r in narrow],['narrowed'])

    def test_parallel_unmatched_intervals_remain_both_added_and_removed(self):
        records=analyze(scene(LineString([(0,0),(200,0)]),6),
                        scene(LineString([(0,20),(200,20)]),6))[0]
        self.assertEqual(sorted(r['change_typ'] for r in records),['added','removed'])
        self.assertTrue(all(abs(r['length_m']-200)<1e-5 for r in records))


if __name__=='__main__':unittest.main()
