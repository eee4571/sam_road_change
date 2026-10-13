"""Continuous variable-width surfaces must not create separate review events."""
import json
from pathlib import Path
import sys
import unittest

import geopandas as gpd
import numpy as np
from shapely.affinity import translate
from shapely.geometry import LineString, Polygon

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from engine.auto_change_geometry import corridor
from engine.continuous_road_geometry import network_surface,change_surfaces


class SurfaceTests(unittest.TestCase):
    def fixture(self):
        value=json.loads((ROOT/'tests/fixtures/variable_corridor_fragment.json').read_text())
        return LineString(value['axis']),np.array(value['stations']),np.array(value['widths'])

    def test_real_fold_does_not_export_detached_triangle(self):
        axis,ss,ww=self.fixture()
        for dx,dy in [(0,0),(757875.75,2461753.25)]:
            with self.subTest(origin=(dx,dy)):
                line=translate(axis,dx,dy)
                actual=corridor(line,ss,ww)
                self.assertEqual(actual.geom_type,'Polygon')
                self.assertTrue(actual.is_valid)
                self.assertLess(line.difference(actual.buffer(1e-5)).length,1e-5)

    def test_straight_taper_preserves_width_area_and_flat_ends(self):
        line=LineString([(0,0),(40,0)])
        actual=corridor(line,[0,40],[4,12])
        expected=Polygon([(0,-2),(40,-6),(40,6),(0,2)])
        self.assertLess(actual.symmetric_difference(expected).area,1e-7)

    def test_almost_constant_profile_matches_round_buffer(self):
        line=LineString([(0,0),(30,0),(30,30),(60,40)])
        actual=corridor(line,[0,line.length],[8,8.000001])
        expected=line.buffer(4,cap_style='flat',join_style='round')
        self.assertLess(actual.symmetric_difference(expected).area,.01)

    def test_translation_does_not_create_round_join_islands(self):
        line=LineString([(0,0),(20,0),(20.5,1),(50,15),(70,15)])
        original=corridor(line,[0,line.length],[12,18])
        moved=corridor(translate(line,760000,2460000),[0,line.length],[12,18])
        self.assertEqual(moved.geom_type,'Polygon')
        self.assertLess(original.symmetric_difference(translate(moved,-760000,-2460000)).area,.001)

    def test_variable_width_loop_keeps_real_hole(self):
        line=LineString([(0,0),(100,0),(100,100),(0,100),(0,0)])
        actual=corridor(line,[0,100,200,300,400],[6,8,10,8,6])
        self.assertEqual(actual.geom_type,'Polygon')
        self.assertEqual(len(actual.interiors),1)
        self.assertGreater(Polygon(actual.interiors[0]).area,8000)

    def test_independent_tiny_roads_are_not_area_filtered(self):
        profiles=[(LineString([(x,0),(x+1,0)]),[0,1],[.5,1]) for x in (0,10)]
        result=network_surface(profiles)
        self.assertEqual(result.geom_type,'MultiPolygon')
        self.assertEqual(len(result.geoms),2)
        self.assertTrue(all(0<p.area<4 for p in result.geoms))

    def test_small_width_change_strips_survive_without_axis_intersection(self):
        line=LineString([(0,0),(10,0)])
        frame=gpd.GeoDataFrame([dict(change_typ='widened',width_bef=4.,width_aft=4.2,
            confidence=.6,axis_wkt=line.wkt,geometry=line.buffer(3))],crs='EPSG:32649')
        result=change_surfaces(frame)
        self.assertEqual(len(result),2)
        self.assertAlmostEqual(result.area.sum(),2.)
        self.assertTrue(all(p.distance(line)>0 for p in result.geometry))


if __name__=='__main__':unittest.main()
