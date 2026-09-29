"""Branch, layer and independent-evidence regressions for actual repair."""
import unittest
from types import SimpleNamespace

import numpy as np
import geopandas as gpd
from shapely import union_all
from shapely.geometry import LineString, box, Point

from graph import Candidate
from medial_repair import AxisProposal, replace_corridors, pin_axis_contacts, parallel_ribbon_veto
from graph import build
from engine.canonical_road_surface import _adjacency, _node
from local_graph_repair import clean_local, smooth_local
from rgb_route import constrained_route, ribbon_evidence


class Image:
    resolution = .75
    gray = None

    def sample(self, array, xy):
        return np.where(abs(xy[:, 1]) < 4, .9, .1)


class Evidence:
    image = Image()

    def values(self, xy):
        return np.zeros(len(xy)), np.full(len(xy), np.nan), np.ones(len(xy))


def frame(lines, layers=None):
    return gpd.GeoDataFrame(dict(source_id=range(len(lines)), width_m=[8.]*len(lines),
        layer=layers or [0]*len(lines)), geometry=[LineString(x) for x in lines], crs='EPSG:32650')


class RepairTests(unittest.TestCase):
    def test_low_sam_does_not_negate_rgb(self):
        line = LineString([(0, 0), (30, 0)])
        score = ribbon_evidence(line, 8., Evidence())
        self.assertEqual(score['raw_rgb_support'], 1.)
        self.assertEqual(score['continuous_support'], 1.)
        self.assertEqual(score['sam_positive_mean'], 0.)

    def test_direction_is_a_search_constraint(self):
        c = Candidate('gap', (0, 1), LineString([(0, 0), (30, 0)]), 8.,
                      dict(approach_directions=[[1, 0], [1, 0]]))
        self.assertIsNotNone(constrained_route(c, Evidence()))
        c.geometry['approach_directions'][0] = [-1, 0]
        self.assertIsNone(constrained_route(c, Evidence()))

    def test_surface_cannot_prove_a_uniform_non_ribbon(self):
        e = Evidence()
        e.image = SimpleNamespace(gray=None, sample=lambda array, xy: np.full(len(xy), .8))
        self.assertEqual(ribbon_evidence(LineString([(0, 0), (30, 0)]), 8., e)['continuous_support'], 0.)

    def test_branch_contacts_and_grade_separation_are_preserved(self):
        f = frame([[(-20, -3), (20, -3)], [(-20, 3), (20, 3)],
                   [(0, 3), (0, 20)], [(-20, 2), (20, 2)]], [0, 0, 0, 1])
        axis = LineString([(-15, 0), (15, 0)])
        p = AxisProposal(axis, axis.buffer(5, cap_style='flat'), 8., 1., {}, [])
        result, edits, contacts = replace_corridors(f, [p], box(-15, -15, 15, 15))
        bridge = union_all(result[result.layer == 1].geometry)
        self.assertTrue(bridge.equals(f.iloc[3].geometry))
        ground = union_all(result[result.layer == 0].geometry)
        self.assertTrue(ground.covers(LineString([(0, 0), (0, 20)])))
        for x in (-15, 15):
            for y in (-3, 3):
                self.assertTrue(ground.covers(LineString([(x, y), (x, 0)])))

    def test_thin_cycle_retains_both_external_arms(self):
        f = frame([[(-20, 0), (0, 0)], [(0, 0), (10, 1), (20, 0)],
                   [(0, 0), (10, -1), (20, 0)], [(20, 0), (40, 0)]])
        result, audit = clean_local(f, Evidence(), box(-50, -50, 50, 50))
        network = union_all(result.geometry)
        self.assertTrue(audit)
        self.assertTrue(network.covers(LineString([(-20, 0), (40, 0)])))
        self.assertTrue(all(r['action'] == 'collapse_thin_cycle' for r in audit))

    def test_large_real_loop_is_not_collapsed(self):
        f = frame([[(-20, 0), (0, 0)], [(0, 0), (20, 15), (40, 0)],
                   [(0, 0), (20, -15), (40, 0)], [(40, 0), (60, 0)]])
        result, audit = clean_local(f, Evidence(), box(-100, -100, 100, 100))
        self.assertFalse(audit)
        self.assertTrue(union_all(result.geometry).equals(union_all(f.geometry)))

    def test_parallel_road_with_separator_survives_replacement(self):
        f = frame([[(-30, 3), (30, 3)], [(-25, 8), (25, 8)]])
        axis = LineString([(-30, 0), (30, 0)])
        p = AxisProposal(axis, axis.buffer(4.64, cap_style='flat'), 8., 1., {}, [])
        e = Evidence()
        e.image = SimpleNamespace(gray=None, resolution=.75, sample=lambda array, xy:
            np.where((abs(xy[:, 1]) < 4) | (abs(xy[:, 1]-8) < 1), .9, .1))
        result, _, _ = replace_corridors(f, [p], box(-40, -20, 40, 20), e)
        self.assertTrue(union_all(result.geometry).covers(f.iloc[1].geometry))

    def test_smoothing_pins_endpoints_and_preserves_road_appearance(self):
        f = frame([[(0, 0), (5, .8), (10, -.8), (15, .8), (20, -.8), (25, 0), (40, 0)]])
        result, audit = smooth_local(f, Evidence(), box(-10, -20, 50, 20))
        self.assertTrue(audit)
        self.assertEqual(result.iloc[0].geometry.coords[0], (0., 0.))
        self.assertEqual(result.iloc[0].geometry.coords[-1], (40., 0.))
        self.assertLessEqual(audit[0]['evidence']['maximum_displacement_m'], 1.2)

    def test_large_coordinate_branch_is_a_real_graph_node(self):
        axis = LineString([(259001.12345, 2614000.4567), (259093.43234, 2614123.6763)])
        point = axis.interpolate(.317, normalized=True)
        pinned = pin_axis_contacts(axis, [point.coords[0]])
        self.assertIn(point.coords[0], list(pinned.coords))
        branch = LineString([point, (point.x+20, point.y-5)])
        f = frame([list(pinned.coords), list(branch.coords)])
        edges = build(f)
        nodes = _adjacency(edges, set(range(len(edges))))
        self.assertEqual(len(nodes[_node(('0', False, False), point.coords[0])]), 3)

    def test_adjacent_alias_faces_share_one_hub(self):
        f = frame([[(-20, 0), (0, 0)], [(0, 0), (10, 2), (20, 0)],
                   [(0, 0), (10, -2), (20, 0)], [(0, 0), (20, 0)],
                   [(20, 0), (40, 0)], [(10, 2), (10, 20)]])
        result, audit = clean_local(f, Evidence(), box(-50, -50, 50, 50))
        self.assertEqual(sum(a['action'] == 'collapse_thin_cycle' for a in audit), 1)
        edges = build(result)
        nodes = _adjacency(edges, set(range(len(edges))))
        self.assertEqual(sum(len(v) == 1 for v in nodes.values()), 3)

    def test_dark_median_is_not_rebuilt_as_road(self):
        f = frame([[(-40, -10), (40, -10)], [(-40, 10), (40, 10)]])
        e = Evidence()
        e.image = SimpleNamespace(gray=None, resolution=.75, sample=lambda array, xy:
            np.where(abs(abs(xy[:, 1])-10) < 4, .9, .1))
        axis = LineString([(-35, 0), (35, 0)])
        p = AxisProposal(axis, axis.buffer(12, cap_style='flat'), 20., 1., {}, [])
        veto = parallel_ribbon_veto(p, f, e)
        self.assertIsNotNone(veto)
        self.assertFalse(veto['applied'])


if __name__ == '__main__':
    unittest.main()
