"""Visual failure mechanisms: missed ribbons, tree rows and broken cap contacts."""
import unittest
import numpy as np
from shapely import union_all
from shapely.geometry import LineString, Point, box

from graph import build
from engine.canonical_road_surface import _adjacency
from medial_repair import (straight_priors, propose_axes, same_pavement_between,
                           AxisProposal, replace_corridors)
from rgb_route import longitudinal_ribbon
from test_repair import frame, Evidence
from test_block_graph import evidence
from block_graph import consolidate_junctions


class CorridorEvidenceTests(unittest.TestCase):
    def test_long_duplicate_loop_becomes_one_axis_with_mid_feature_branch(self):
        old = frame([[(-110, 0), (-80, 0)],
                     [(-80, 0), (-60, 2), (60, 2), (80, 0)],
                     [(-80, 0), (-60, -2), (60, -2), (80, 0)],
                     [(80, 0), (110, 0)], [(20, 2), (20, 35)]])
        target = union_all([LineString([(-115, 0), (115, 0)]), LineString([(20, 0), (20, 40)])])
        result, audit = consolidate_junctions(old, evidence(target), box(-115, -60, 115, 60), corridor_mode=True)
        self.assertTrue(any(r['action'] == 'canonical_corridor' and r['applied'] for r in audit))
        graph = build(result); nodes = _adjacency(graph, set(range(len(graph))))
        self.assertEqual(sum(len(v) == 1 for v in nodes.values()), 3)
        self.assertEqual(sum(len(v) >= 3 for v in nodes.values()), 1)
        self.assertLess(union_all(result.geometry).distance(Point(20, 35)), .001)

    def test_long_true_parallel_ring_with_median_is_retained(self):
        old = frame([[(-80, 0), (-60, 9), (60, 9), (80, 0)],
                     [(-80, 0), (-60, -9), (60, -9), (80, 0)],
                     [(-110, 0), (-80, 0)], [(80, 0), (110, 0)]])
        result, audit = consolidate_junctions(old, evidence(union_all(old.geometry)),
                                              box(-115, -60, 115, 60), corridor_mode=True)
        self.assertFalse(any(r['applied'] for r in audit))
        self.assertTrue(union_all(result.geometry).equals(union_all(old.geometry)))

    def test_closed_ring_yields_direction_priors_not_zero_direction(self):
        ring = LineString([(-40, -20), (40, -20), (40, 20), (-40, 20), (-40, -20)])
        parts = straight_priors(ring)
        self.assertGreaterEqual(len(parts), 4)
        self.assertTrue(all(Point(p.coords[0]).distance(Point(p.coords[-1])) > 15 for p in parts))

    def test_wrong_cached_width_does_not_prevent_narrow_ribbon_search(self):
        old = frame([[(-90, 3), (90, 3)]])
        old.width_m = 24.
        target = LineString([(-110, 0), (110, 0)])
        e = evidence(target)
        proposals, _ = propose_axes(build(old), e, box(-100, -50, 100, 50))
        self.assertTrue(any(p.axis.length > 100 and p.axis.centroid.distance(target) < 1.5 for p in proposals))

    def test_transverse_tree_ribbon_is_not_a_road_axis(self):
        old = frame([[(-90, 10), (90, 10)]])
        old.width_m = 24.
        e = evidence(LineString([(-110, 0), (110, 0)]))
        rr, _ = np.indices(e.image.gray.shape)
        y = 120-(rr+.5)*.5
        trees = abs(y-12) < 4
        e.image.gray[trees] = .5
        e.image.rgb[trees] = [40, 160, 40]
        proposals, _ = propose_axes(build(old), e, box(-100, -50, 100, 50))
        self.assertTrue(proposals)
        self.assertTrue(all(abs(p.axis.centroid.y-12) > 4 for p in proposals))

    def test_longitudinal_repetition_rejects_isolated_bright_vehicle(self):
        fractions = np.linspace(-.5, 1.5, 41)
        profiles = np.full((31, 41), .2)
        profiles[:, (fractions >= 0) & (fractions <= 1)] = .7
        profiles[::5, 15:26] = .95
        probes = np.zeros((31, 41, 2))
        _, supported = longitudinal_ribbon(profiles, probes, fractions, Evidence())
        self.assertGreater(supported[4:-4].mean(), .95)

    def test_paired_curbs_support_dark_road_but_single_edge_does_not(self):
        fractions = np.linspace(-.5, 1.5, 41)
        profiles = np.full((31, 41), .2)
        profiles[:, abs(fractions) < .06] = .8
        profiles[:, abs(fractions-1) < .06] = .8
        probes = np.zeros((31, 41, 2))
        _, supported = longitudinal_ribbon(profiles, probes, fractions, Evidence())
        self.assertTrue(supported.all())
        profiles[:, fractions > .6] = .2
        _, supported = longitudinal_ribbon(profiles, probes, fractions, Evidence())
        self.assertFalse(supported.any())

    def test_same_pavement_equivalence_requires_no_separator(self):
        a = LineString([(-60, -2), (60, -2)])
        b = LineString([(-60, 2), (60, 2)])
        self.assertEqual(same_pavement_between(a, b, Evidence()), 1.)
        left = LineString([(-60, -12), (60, -12)])
        right = LineString([(-60, 12), (60, 12)])
        self.assertLess(same_pavement_between(left, right, evidence(union_all([left, right]))), .2)

    def test_cap_trim_and_noded_branch_preserve_actual_graph_contacts(self):
        old = frame([[(-70, 2), (70, 2)], [(0, 2), (0, 30)], [(0, 15), (30, 15)]])
        axis = LineString([(-50, 0), (50, 0)])
        p = AxisProposal(axis, axis.buffer(5, cap_style='flat'), 8., 1., {}, [])
        out, _, contacts = replace_corridors(old, [p], box(-90, -40, 90, 40))
        nodes = _adjacency(build(out), set(range(len(build(out)))))
        self.assertEqual(sum(len(v) == 1 for v in nodes.values()), 4)
        self.assertEqual(sum(len(v) == 3 for v in nodes.values()), 2)
        self.assertTrue(contacts)
        self.assertLess(p.axis.length, 100)


if __name__ == '__main__':
    unittest.main()
