import unittest
import numpy as np
from shapely.geometry import LineString,box
from engine.continuous_road_geometry import network_surface


def profile(points,width=8):
    a=LineString(points);return a,[0,a.length],[width,width]


class ContinuousRoadGeometryTests(unittest.TestCase):
    def test_near_duplicate_endpoint_does_not_create_hook_cap(self):
        actual=network_surface([profile([(0,0),(60,0),(60-1e-10,0)],8)])
        self.assertLess(actual.symmetric_difference(box(0,-4,60,4)).area,.001)

    def test_right_angle_internal_joint_is_not_two_rectangles(self):
        actual=network_surface([profile([(0,0),(30,0)]),profile([(30,0),(30,30)])])
        expected=LineString([(0,0),(30,0),(30,30)]).buffer(4,cap_style='flat',join_style='round')
        self.assertLess(actual.symmetric_difference(expected).area,.01)

    def test_internal_bend_has_no_flat_cap_notch(self):
        parts=[profile([(0,0),(30,0)]),profile([(30,0),(60,15)])]
        actual=network_surface(parts)
        expected=LineString([(0,0),(30,0),(60,15)]).buffer(4,cap_style='flat',join_style='round')
        self.assertLess(actual.symmetric_difference(expected).area,.01)
        self.assertEqual(len(actual.interiors),0)

    def test_t_junction_has_one_boundary_and_flat_external_ends(self):
        actual=network_surface([profile([(0,0),(50,0)]),profile([(50,0),(100,0)]),profile([(50,0),(50,30)])])
        expected=box(0,-4,100,4).union(box(46,0,54,30))
        self.assertLess(actual.symmetric_difference(expected).area,.01)

    def test_parallel_tracks_are_not_connected(self):
        actual=network_surface([profile([(0,0),(100,0)],4),profile([(0,8),(100,8)],4)])
        self.assertEqual(actual.geom_type,'MultiPolygon')
        self.assertEqual(len(actual.geoms),2)

    def test_junction_short_terminal_patches_are_absorbed(self):
        actual=network_surface([profile([(0,0),(60,0)],10),
            profile([(60,0),(66,3)],10),profile([(60,0),(63,-6)],10)])
        self.assertEqual(actual.geom_type,'Polygon')
        self.assertGreaterEqual(actual.bounds[2],66.)
        self.assertLess(actual.bounds[2],71.)

    def test_isolated_short_road_is_retained(self):
        actual=network_surface([profile([(0,0),(8,0)],10)])
        self.assertAlmostEqual(actual.area,80.)

    def test_absorbing_patch_retains_reach_to_existing_crossroad(self):
        actual=network_surface([profile([(0,0),(60,0)],10),
            profile([(60,0),(66,3)],10),profile([(60,0),(63,-6)],10),
            profile([(70,-30),(70,30)],10)])
        self.assertEqual(actual.geom_type,'Polygon')

    def test_width_transition_is_continuous_across_feature_boundary(self):
        actual=network_surface([profile([(0,0),(30,0)],6),profile([(30,0),(60,0)],10)])
        self.assertTrue(actual.is_valid)
        self.assertEqual(actual.geom_type,'Polygon')
        self.assertEqual(len(actual.interiors),0)
        self.assertEqual(actual.bounds[0],0.)
        self.assertEqual(actual.bounds[2],60.)

    def test_small_junction_hole_is_filled_but_road_loop_is_retained(self):
        small=network_surface([profile([(0,0),(12,0),(6,10),(0,0)],6)])
        large=network_surface([profile([(0,0),(100,0),(100,100),(0,100),(0,0)],6)])
        self.assertEqual(len(small.interiors),0)
        self.assertEqual(len(large.interiors),1)


if __name__=='__main__':unittest.main()
