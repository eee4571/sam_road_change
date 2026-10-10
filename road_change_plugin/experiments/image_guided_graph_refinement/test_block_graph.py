"""Block fitting must regularize connected streets, not destroy real loops."""
import unittest
from types import SimpleNamespace
import numpy as np
from rasterio.transform import from_origin
from shapely import union_all, distance, points
from shapely.geometry import LineString, Point, box
from evidence import ImageGrid
from block_graph import joint_fit, pavement_inside, complete_streets
from test_repair import frame
from graph import build
from engine.canonical_road_surface import _adjacency
from street_route import turning_route
from color_ribbon import color_features
from graph_transaction import validate


def evidence(roads):
    image = ImageGrid.__new__(ImageGrid)
    image.resolution = .5
    image.transform = from_origin(-120, 120, .5, .5)
    rr, cc = np.indices((480, 480))
    xy = np.c_[-120+(cc.ravel()+.5)*.5, 120-(rr.ravel()+.5)*.5]
    mask = distance(points(xy), roads).reshape(480, 480) < 4
    image.gray = np.where(mask, .7, .2)
    image.rgb = np.repeat((image.gray*255)[..., None], 3, axis=2)
    image.valid = np.ones_like(mask)
    image.valid_float = image.valid.astype(float)
    return SimpleNamespace(image=image, multichannel_ribbon=True, independent_flanks=True,
                           values=lambda xy: (np.zeros(len(xy)), np.zeros(len(xy)), np.zeros(len(xy))))


class BlockGraphTests(unittest.TestCase):
    def test_road_between_bright_building_and_dark_vegetation(self):
        fractions = np.linspace(-.5, 1.5, 41)
        rgb = np.ones((5, 41, 3))*.5
        rgb[:, fractions < 0] = .9
        rgb[:, fractions > 1] = .1
        self.assertFalse(color_features(rgb, fractions)[1].any())
        self.assertTrue(color_features(rgb, fractions, independent_flanks=True)[1].all())
        rgb[:, fractions > .5] = .9
        self.assertFalse(color_features(rgb, fractions, independent_flanks=True)[1].any())

    def test_joint_fit_regularizes_long_roads_with_shared_junction(self):
        old = frame([[(-90, 0), (-60, 2), (-30, -2), (0, 0)],
                     [(0, 0), (30, 2), (60, -2), (90, 0)],
                     [(0, 0), (2, 30), (-2, 60), (0, 90)]])
        target = union_all([LineString([(-100, 0), (100, 0)]), LineString([(0, 0), (0, 100)])])
        result, report = joint_fit(old, evidence(target), box(-110, -110, 110, 110))
        graph = build(result); nodes = _adjacency(graph, set(range(len(graph))))
        self.assertEqual(sum(len(v) == 3 for v in nodes.values()), 1)
        self.assertEqual(sum(len(v) == 1 for v in nodes.values()), 3)
        self.assertLess(union_all(result.geometry).hausdorff_distance(target.intersection(box(-90, -90, 90, 90))), 1.5)
        self.assertGreater(report['equations'], 100)

    def test_outside_curve_is_unchanged(self):
        old = frame([[(-110, 0), (-60, 3), (0, -3), (110, 0)]])
        result, _ = joint_fit(old, evidence(union_all(old.geometry)), box(-50, -50, 50, 50))
        self.assertTrue(union_all(old.geometry).equals(union_all(result.geometry)))

    def test_real_ring_has_distinct_interior(self):
        ring = Point(0, 0).buffer(15).boundary
        score = pavement_inside(Point(0, 0).buffer(15), evidence(ring))
        self.assertFalse(score['consistent'])

    def test_parallel_roads_remain_separate(self):
        old = frame([[(-90, -12), (0, -10), (90, -12)], [(-90, 12), (0, 10), (90, 12)]])
        result, _ = joint_fit(old, evidence(union_all(old.geometry)), box(-110, -110, 110, 110))
        self.assertEqual(len(result), 2)
        self.assertGreater(result.geometry.iloc[0].distance(result.geometry.iloc[1]), 15)

    def test_street_continuation_with_zero_model_support(self):
        old = frame([[(-90, 0), (-50, 1), (-10, 0)], [(10, 0), (50, -1), (90, 0)]])
        e = evidence(LineString([(-110, 0), (110, 0)]))
        after, audit = complete_streets(old, e, box(-110, -110, 110, 110))
        self.assertTrue(any(a['applied'] for a in audit))
        self.assertEqual(sum(len(v) == 1 for v in _adjacency(build(after), set(range(len(build(after))))).values()), 2)
        # Collinearity alone must not bridge an absent road.
        e = evidence(union_all(old.geometry))
        _, audit = complete_streets(old, e, box(-110, -110, 110, 110))
        self.assertFalse(any(a['applied'] for a in audit))

    def test_turning_route_follows_l_shaped_pavement(self):
        roads = LineString([(-80, 0), (0, 0), (0, 80)])
        line, stats = turning_route((-30, 0), (0, 30), np.array([1., 0.]), np.array([0., 1.]), evidence(roads))
        self.assertIsNotNone(line)
        self.assertLess(line.difference(roads.buffer(4)).length, .1)
        self.assertGreater(line.length, 50)

    def test_forecourt_uses_pavement_not_missing_curbs(self):
        old = frame([[(-90, 0), (-10, 0)], [(10, 0), (90, 0)]])
        paved = union_all([LineString([(-110, 0), (110, 0)]), box(-20, -20, 20, 20)])
        e = evidence(paved)
        _, audit = complete_streets(old, e, box(-110, -110, 110, 110))
        self.assertTrue(any(a['applied'] and a['evidence']['forecourt_hypothesis'] for a in audit))
        # A roof occupying that gap is a different material, not a forecourt.
        rr, cc = np.indices(e.image.gray.shape)
        roof = (abs(-120+(cc+.5)*.5)<8)&(abs(120-(rr+.5)*.5)<8)
        e.image.gray[roof] = .98; e.image.rgb[roof] = .98*255
        _, audit = complete_streets(old, e, box(-110, -110, 110, 110))
        self.assertFalse(any(a['applied'] for a in audit))

    def test_loop_guard_uses_common_precision_on_both_graphs(self):
        old = frame([[(-20, 0), (0, 0)], [(0, 0), (10, 2), (20, 0)],
                     [(0, .00002), (10, -2), (20, 0)], [(20, 0), (40, 0)]])
        exact = frame([[(-20, 0), (0, 0)], [(0, 0), (10, 2), (20, 0)],
                       [(0, 0), (10, -2), (20, 0)], [(20, 0), (40, 0)]])
        e = evidence(union_all(exact.geometry))
        check = validate(old, exact, box(-50, -50, 50, 50), e)
        self.assertNotIn('new_small_loop', check['failures'])
        # A genuinely new second route still creates an extra loop.
        simple = frame([[(-20, 0), (0, 0), (10, 2), (20, 0), (40, 0)]])
        check = validate(simple, exact, box(-50, -50, 50, 50), e)
        self.assertIn('new_small_loop', check['failures'])


if __name__ == '__main__':
    unittest.main()
