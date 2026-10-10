import unittest
import numpy as np
from shapely import union_all
from shapely.geometry import LineString, box, Point
from internal_axis import straight_runs, regularize, rgb_comparison, collapse_internal_aliases
from test_repair import frame
from test_block_graph import evidence
from graph import build
from engine.canonical_road_surface import _adjacency


class InternalAxisTests(unittest.TestCase):
    def test_same_pavement_alias_is_rebuilt_before_straightening(self):
        old = frame([[(-90,0),(-12,0)],[(-12,0),(0,2),(12,0)],
                     [(-12,0),(0,-2),(12,0)],[(12,0),(90,0)]])
        rgb = evidence(LineString([(-100,0),(100,0)]))
        result,audit = collapse_internal_aliases(old,rgb,box(-110,-110,110,110))
        self.assertTrue(any(r['applied'] for r in audit))
        self.assertTrue(union_all(result.geometry).equals(LineString([(-90,0),(90,0)])))

    def test_real_loop_with_distinct_interior_is_kept(self):
        old = frame([[(-70,0),(-20,0)],[(-20,0),(0,20),(20,0)],
                     [(-20,0),(0,-20),(20,0)],[(20,0),(70,0)]])
        result,audit = collapse_internal_aliases(old,evidence(union_all(old.geometry)),box(-110,-110,110,110))
        self.assertFalse(any(r['applied'] for r in audit))
        self.assertTrue(union_all(result.geometry).equals(union_all(old.geometry)))

    def test_close_parallel_roads_with_separator_are_kept(self):
        old = frame([[(-12,0),(0,3),(12,0)],[(-12,0),(0,-3),(12,0)],
                     [(-90,0),(-12,0)],[(12,0),(90,0)]])
        rgb = evidence(union_all(old.geometry))
        # A persistent grass separator inside an otherwise thin candidate loop.
        from rasterio.transform import rowcol
        for x in np.arange(-8,8,.5):
            for y in np.arange(-.9,1,.5):
                r,c=rowcol(rgb.image.transform,x,y)
                rgb.image.rgb[r,c]=[20,150,20]
                rgb.image.gray[r,c]=.3
        result,audit=collapse_internal_aliases(old,rgb,box(-110,-110,110,110))
        self.assertFalse(any(r['applied'] for r in audit))
        self.assertTrue(union_all(result.geometry).equals(union_all(old.geometry)))

    def test_straight_run_is_exact_and_endpoints_are_fixed(self):
        old = LineString([(-90,0),(-60,1),(-30,-1),(0,0),(30,1),(60,-1),(90,0)])
        result = straight_runs(old, 2)
        self.assertTrue(result.equals(LineString([(-90,0),(90,0)])))

    def test_real_corner_and_its_approaches_remain(self):
        old = LineString([(-70,0),(-40,1),(-10,0),(-3,0),(0,3),(0,10),(1,40),(0,70)])
        result = straight_runs(old, 2)
        self.assertTrue(result.buffer(.0001).covers(LineString([(-3,0),(0,3)])))
        self.assertLessEqual(result.hausdorff_distance(old), 2)

    def test_real_curve_cannot_be_replaced_by_a_chord(self):
        angles = np.linspace(0,np.pi/2,51)
        old = LineString(np.c_[40*np.cos(angles),40*np.sin(angles)])
        result = straight_runs(old, 2)
        self.assertGreater(result.length, old.length*.96)
        self.assertLess(result.hausdorff_distance(old), 2.01)

    def test_refit_keeps_junction_and_carriageway_geometry(self):
        old = frame([[(-90,0),(-60,1),(-30,-1),(0,0)],
                     [(0,0),(30,1),(60,-1),(90,0)],[(0,0),(0,70)],
                     [(-90,30),(-50,31),(-20,30)]])
        old['carriageway_id'] = ['', '', '', 'protected:0']
        target = union_all([LineString([(-100,0),(100,0)]),LineString([(0,0),(0,80)]),old.geometry.iloc[3]])
        result,audit = regularize(old, evidence(target), box(-110,-110,110,110))
        self.assertTrue(any(r['applied'] for r in audit))
        self.assertTrue(union_all(result[result.carriageway_id=='protected:0'].geometry).equals(old.geometry.iloc[3]))
        nodes = _adjacency(build(result),set(range(len(build(result)))))
        self.assertEqual(sum(len(v)==3 for v in nodes.values()),1)
        self.assertIn((0.,0.), [tuple(n[1:]) for n,v in nodes.items() if len(v)==3])

    def test_same_geometry_without_rgb_ribbon_is_not_enough(self):
        old = LineString([(-80,0),(0,1),(80,0)])
        rgb = evidence(LineString([(-80,40),(80,40)]))
        rgb.image.rgb[:] = 120
        rgb.image.gray[:] = 120/255
        self.assertFalse(rgb_comparison(old, straight_runs(old, 2), rgb)['passed'])


if __name__ == '__main__':
    unittest.main()
