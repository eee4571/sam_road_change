"""Counterexamples for multi-scene refinement; tests never run models or width estimation."""
import unittest
from types import SimpleNamespace

import numpy as np
from shapely import union_all, points, distance
from shapely.geometry import LineString, box

from test_repair import frame, Evidence
from color_ribbon import color_features
from graph_transaction import validate
from local_graph_optimizer import cycle_steps, refit_path, choose, optimize, junction_steps, branch_steps


class MultiRoiTests(unittest.TestCase):
    def test_equal_luminance_chromatic_ribbon(self):
        fractions = np.linspace(-.5, 1.5, 41)
        rgb = np.tile([.2, .65, .2], (5, 41, 1))
        rgb[:, (fractions >= 0) & (fractions <= 1)] = [.7, .25, .5]
        _, supported, _, _ = color_features(rgb, fractions)
        self.assertTrue(supported.all())

    def test_single_building_edge_is_not_a_ribbon(self):
        fractions = np.linspace(-.5, 1.5, 41)
        rgb = np.tile([.2, .2, .2], (5, 41, 1))
        rgb[:, fractions > .5] = [.9, .9, .9]
        self.assertFalse(color_features(rgb, fractions)[1].any())

    def test_outside_window_and_connections_are_transactional(self):
        old = frame([[(-30, 0), (30, 0)]])
        broken = frame([[(-30, 0), (-2, 0)], [(2, 0), (30, 0)]])
        result = validate(old, broken, box(-10, -10, 10, 10), Evidence())
        self.assertFalse(result['passed'])
        self.assertIn('previous_external_connections_split', result['failures'])
        shifted = frame([[(-30, 1), (30, 1)]])
        self.assertIn('outside_window_changed', validate(old, shifted, box(-10, -10, 10, 10), Evidence())['failures'])

    def test_supported_short_branch_is_protected(self):
        old = frame([[(-30, 0), (30, 0)], [(0, 0), (0, 18)]])
        e = Evidence()
        roads = union_all(old.geometry)
        e.image = SimpleNamespace(gray=None, sample=lambda array, xy:
            np.where(distance(points(xy), roads) <= 4, .9, .1))
        after = frame([[(-30, 0), (30, 0)]])
        result = validate(old, after, box(-20, -20, 20, 20), e)
        self.assertIn('independently_supported_road_lost', result['failures'])

    def test_small_true_closed_ribbon_is_preserved(self):
        old = frame([[(-20, 0), (0, 0)], [(0, 0), (10, 3), (20, 0)],
                     [(0, 0), (10, -3), (20, 0)], [(20, 0), (40, 0)]])
        old.width_m = 2.
        roads = union_all(old.geometry)
        e = Evidence()
        e.image = SimpleNamespace(gray=None, resolution=.75, sample=lambda array, xy:
            np.where(distance(points(xy), roads) < 1, .9, .1))
        audit = []
        result, count = cycle_steps(old, e, box(-50, -50, 50, 50), audit, 1)
        self.assertEqual(count, 0)
        self.assertTrue(union_all(result.geometry).equals(roads))

    def test_alias_loop_is_rebuilt_without_deleting_arms(self):
        old = frame([[(-20, 0), (0, 0)], [(0, 0), (10, .5), (20, 0)],
                     [(0, 0), (10, -.5), (20, 0)], [(20, 0), (40, 0)]])
        audit = []
        result, count = cycle_steps(old, Evidence(), box(-50, -50, 50, 50), audit, 1)
        self.assertGreater(count, 0)
        self.assertTrue(union_all(result.geometry).covers(LineString([(-20, 0), (40, 0)])))

    def test_refit_keeps_junction_approaches_fixed(self):
        line = LineString([(0, 2), (80, 2)])
        fitted = refit_path(line, 8, Evidence())
        self.assertEqual(fitted.coords[0], line.coords[0])
        self.assertEqual(fitted.coords[-1], line.coords[-1])
        self.assertLess(abs(fitted.interpolate(.5, normalized=True).y), 2)

    def test_keep_beats_unsupported_modification(self):
        old = frame([[(-20, 0), (20, 0)]])
        alternative = frame([[(-20, 0), (0, 3), (20, 0)]])
        audit = []
        result, changed = choose(old, [(alternative, dict(action='refit', evidence_passed=False, objective_gain=1))],
                                 box(-30, -10, 30, 10), Evidence(), audit, 'test', 1)
        self.assertFalse(changed)
        self.assertTrue(result.equals(old))

    def test_normal_control_uses_same_optimizer_and_remains_stable(self):
        old = frame([[(-80, 0), (80, 0)]])
        config = dict(max_gap_m=100, direction_cosine=.94, max_lateral_m=12, max_width_ratio=2)
        result, audit, _ = optimize(old, Evidence(), box(-100, -100, 100, 100), config)
        self.assertTrue(union_all(result.geometry).equals(union_all(old.geometry)))
        self.assertFalse(any(r.get('applied') for r in audit))

    def test_directional_gap_can_join_interior_of_existing_road(self):
        old = frame([[(-30, 0), (0, 0)], [(25, -25), (25, 25)]])
        roads = union_all([LineString([(-40, 0), (25, 0)]), old.geometry.iloc[1]])
        e = Evidence()
        e.image = SimpleNamespace(gray=None, resolution=.75, sample=lambda array, xy:
            np.where(distance(points(xy), roads) < 4, .9, .1))
        audit = []
        result, count = junction_steps(old, e, box(-50, -50, 50, 50), audit, 1)
        self.assertEqual(count, 1)
        self.assertTrue(union_all(result.geometry).covers(LineString([(0, 0), (25, 0)])))
        separated = old.copy()
        separated.loc[1, 'layer'] = 1
        self.assertEqual(junction_steps(separated, e, box(-50, -50, 50, 50), [], 1)[1], 0)

    def test_short_spur_is_not_deleted_from_length_or_low_sam(self):
        old = frame([[(-30, 0), (30, 0)], [(0, 0), (0, 1)]])
        config = dict(max_gap_m=100, direction_cosine=.94, max_lateral_m=12, max_width_ratio=2)
        result, count = branch_steps(old, Evidence(), box(-50, -50, 50, 50), config, [], 1)
        self.assertEqual(count, 0)
        self.assertTrue(union_all(result.geometry).equals(union_all(old.geometry)))


if __name__ == '__main__':
    unittest.main()
