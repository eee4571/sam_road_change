"""Structural positive cases and protected real-road counterexamples."""
import unittest
import numpy as np
from shapely import union_all
from shapely.geometry import LineString, Point, box
from test_block_graph import evidence
from test_repair import frame
from structural_refinement import short_chains, active_continuations, repair_domains, rebuild_domains, parallel_alias_contact
from graph_transaction import validate
from assemble_multi_roi import stitch_regions, graph_summary
from block_graph import complete_streets


class StructuralTests(unittest.TestCase):
    def test_incompatible_endpoint_materials_do_not_authorize_intervening_roof(self):
        old = frame([[(-90, 0), (-30, 0)], [(30, 0), (90, 0)]])
        e = evidence(LineString([(-110, 0), (110, 0)]))
        rr, cc = np.indices(e.image.gray.shape)
        x = -120+(cc+.5)*.5
        ribbon = e.image.gray > .5
        e.image.gray[ribbon & (x < -10)] = .25
        e.image.gray[ribbon & (x >= -10)] = .6
        e.image.rgb = np.repeat((e.image.gray*255)[..., None], 3, axis=2)
        _, audit = complete_streets(old, e, box(-115, -60, 115, 60))
        self.assertFalse(any(r.get('applied') for r in audit))

    def test_side_axis_hook_is_distinguished_from_collinear_gap_and_t_junction(self):
        approach = LineString([(-15, 4), (0, 4)])
        self.assertTrue(parallel_alias_contact(approach, LineString([(-40, 0), (40, 0)]), (10, 0), 8)[0])
        self.assertFalse(parallel_alias_contact(approach, LineString([(20, 4), (60, 4)]), (20, 4), 8)[0])
        self.assertFalse(parallel_alias_contact(approach, LineString([(20, -20), (20, 20)]), (20, 4), 8)[0])

    def test_short_curved_chain_moves_toward_ribbon_and_pins_ends(self):
        truth = LineString([(-22, 0), (-10, 4), (0, 6), (10, 4), (22, 0)])
        old = frame([[(-22, 0), (-10, 6), (0, 8), (10, 6), (22, 0)]])
        old.width_m = 8.
        audit = []
        result, count = short_chains(old, evidence(truth), box(-50, -50, 50, 50), audit)
        self.assertGreater(count, 0)
        self.assertEqual(result.geometry.iloc[0].coords[0], old.geometry.iloc[0].coords[0])
        self.assertEqual(result.geometry.iloc[0].coords[-1], old.geometry.iloc[0].coords[-1])
        self.assertLess(result.geometry.iloc[0].hausdorff_distance(truth), old.geometry.iloc[0].hausdorff_distance(truth))

    def test_active_trace_connects_road_interior_without_gap_pair(self):
        old = frame([[(-40, 0), (0, 0)], [(25, -30), (25, 30)]])
        truth = union_all([LineString([(-45, 0), (25, 0)]), old.geometry.iloc[1]])
        after, count = active_continuations(old, evidence(truth), box(-55, -45, 55, 45), [])
        self.assertGreater(count, 0)
        self.assertTrue(validate(old, after, box(-55, -45, 55, 45), evidence(truth))['passed'])

    def test_no_ribbon_and_different_level_do_not_connect(self):
        old = frame([[(-40, 0), (0, 0)], [(25, -30), (25, 30)]])
        self.assertEqual(active_continuations(old, evidence(union_all(old.geometry)), box(-55, -45, 55, 45), [])[1], 0)
        truth = union_all([LineString([(-45, 0), (25, 0)]), old.geometry.iloc[1]])
        old.loc[1, 'layer'] = 1
        self.assertEqual(active_continuations(old, evidence(truth), box(-55, -45, 55, 45), [])[1], 0)

    def test_centered_short_control_does_not_move(self):
        old = frame([[(-22, 0), (22, 0)]])
        result, count = short_chains(old, evidence(union_all(old.geometry)), box(-50, -50, 50, 50), [])
        self.assertEqual(count, 0)
        self.assertTrue(result.equals(old))

    def test_real_parallel_roads_and_true_loop_preserved(self):
        old = frame([[(-40, -12), (40, -12)], [(-40, 12), (40, 12)],
                     [(-12, -12), (-12, 12)], [(12, -12), (12, 12)]])
        truth = union_all(old.geometry)
        config = dict(max_gap_m=100, direction_cosine=.94, max_lateral_m=12, max_width_ratio=2)
        result, _ = rebuild_domains(old, evidence(truth), box(-80, -80, 80, 80), config, [], 1)
        self.assertTrue(union_all(result.geometry).equals(truth))

    def test_micron_roi_seam_is_pinned_to_unchanged_original_contact(self):
        old = frame([[(-30, .1234567), (30, .1234567)]])
        window = box(-10.123456, -10, 10.123456, 10)
        region = frame([[(-10.12344, .123451), (0, .1234), (10.12345, .123449)]])
        after = stitch_regions(old, [region], [window])
        self.assertEqual(graph_summary(after)['components'], 1)
        self.assertEqual(graph_summary(after)['degree_1'], 2)
        outside = union_all(old.geometry).difference(window)
        self.assertLess(outside.difference(union_all(after.geometry).buffer(1e-8)).length, 1e-6)

    def test_domain_compares_whole_alias_and_preserves_external_arms(self):
        old = frame([[(-50, 0), (-10, 0)], [(-10, 0), (0, .6), (10, 0)],
                     [(-10, 0), (0, -.6), (10, 0)], [(10, 0), (50, 0)], [(0, .6), (0, 8)]])
        truth = union_all([LineString([(-55, 0), (55, 0)]), LineString([(0, 0), (0, 8)])])
        config = dict(max_gap_m=100, direction_cosine=.94, max_lateral_m=12, max_width_ratio=2)
        audit = []
        after, count = rebuild_domains(old, evidence(truth), box(-80, -80, 80, 80), config, audit, 1)
        self.assertGreater(count, 0)
        self.assertTrue(any(r.get('alternatives_include_keep') for r in audit))
        self.assertLess(Point(0, 8).distance(union_all(after.geometry)), 1e-5)
        self.assertTrue(validate(old, after, box(-25, -25, 25, 25), evidence(truth))['passed'])


if __name__ == '__main__':
    unittest.main()
