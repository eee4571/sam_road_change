"""Axis interval optimization versus the original cell overlay, without models."""
import unittest
from unittest.mock import patch

import numpy as np
from shapely import length
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import substring, unary_union

import engine.fast_auto_change as auto
import test_fast_auto_change as fixtures


def scalar_coverage(axis, support, starts, ends, cells, cell_lengths):
    return np.asarray([cell.intersection(support).length/max(cell.length, 1e-9) for cell in cells])


class AxisCoverageTests(unittest.TestCase):
    def check_coverage(self, axis, support, selected=None):
        count = max(1, int(np.ceil(axis.length/4)))
        spacing = axis.length/count
        selected = np.arange(count) if selected is None else selected
        starts, ends = selected*spacing, (selected+1)*spacing
        cells = np.asarray([substring(axis, a, b) for a, b in zip(starts, ends)], dtype=object)
        args = axis, support, starts, ends, cells, length(cells)
        expected = scalar_coverage(*args)
        actual = auto._axis_surface_coverage(*args)
        np.testing.assert_array_equal(actual, expected)

    def test_curves_holes_disjoint_tangencies_and_large_coordinates(self):
        axes = [LineString([(0, 0), (20, 3), (45, -8), (61, 10), (100, 0)]),
                LineString([(0, 0), (100, 0)]),
                LineString([(0, 0), (40, 20), (0, 20), (40, 0)]),
                LineString([(0, 0), (40, 0), (10, 0), (70, 0)]),
                LineString([(0, 0), (40, 0), (40, 40), (0, 0)])]
        supports = [box(-1, -50, 101, 50), Polygon(),
                    box(0, 0, 100, 10), Point(45, -8).buffer(7),
                    unary_union([box(2.2, -20, 16.8, 25), box(39, -30, 88, 30)]),
                    box(-10, -50, 110, 60).difference(box(28, -10, 52, 20))]
        for axis in axes:
            for support in supports:
                for x, y in ((0, 0), (500000, 4200000)):
                    with self.subTest(axis=axis.wkt, support=support.wkt, shift=x):
                        self.check_coverage(translate(axis, x, y), translate(support, x, y))

    def test_random_curves_selected_cells(self):
        rng = np.random.default_rng(917)
        for _ in range(30):
            axis = LineString(np.column_stack([np.arange(40)*6., rng.normal(0, 3, 40)]))
            support = unary_union([box(x, -20, x+float(rng.uniform(1, 55)), 20)
                                   for x in rng.uniform(0, 230, 6)])
            count = int(np.ceil(axis.length/4))
            self.check_coverage(axis, support, np.arange(count)[::2])

    def test_coverage_thresholds_keep_exact_evidence_states_and_reasons(self):
        from types import SimpleNamespace
        axis = LineString([(0, 0), (4, 0)])
        scene = SimpleNamespace(valid=box(-100, -100, 100, 100))
        probability = {'scene_percentile_rank': .2, 'background_percentile_rank': .2,
                       'local_probability_contrast': 0., 'probability_valid_ratio': 1.}
        for threshold in (.1, .55):
            for fraction in (np.nextafter(threshold, 0.), threshold, np.nextafter(threshold, 1.)):
                support = box(-1, -1, 4*fraction, 1)
                cells = np.asarray([axis], dtype=object)
                coverage = auto._axis_surface_coverage(axis, support, np.array([0.]), np.array([4.]), cells, length(cells))[0]
                expected = auto.RoadScene.evidence(scene, axis, False, 8., 3., support, probability)
                actual = auto.RoadScene.evidence(scene, axis, False, 8., 3., support, probability,
                                                 geometry_evidence=(True, coverage))
                self.assertEqual(expected, actual)

    def test_interior_cells_do_not_execute_station_overlay(self):
        axis = LineString([(0, 0), (4000, 0)])
        import shapely
        original = shapely.intersection
        with patch.object(shapely, 'intersection', wraps=original) as calls:
            self.check_coverage(axis, box(-1, -1, 4001, 1), np.arange(1, 999))
        # check_coverage's scalar reference uses 998 calls; optimized code has
        # one axis overlay plus an empty boundary batch, never 998 cell overlays.
        optimized = calls.call_args_list[998:]
        self.assertEqual(len(optimized), 2)
        self.assertEqual(len(optimized[-1].args[0]), 0)


class AxisCandidateEquivalenceTests(unittest.TestCase):
    setUp = fixtures.FastFinalAutoTests.setUp
    scene = fixtures.FastFinalAutoTests.scene
    road = staticmethod(fixtures.FastFinalAutoTests.road)

    def test_states_reasons_candidates_geometries_attributes_and_order(self):
        before = self.scene([self.road(40, 8), self.road(95, 14), self.road(150)])
        after = self.scene([self.road(40, 14), self.road(95, 8), self.road(205)],
                           valid=box(25, 5, 275, 245))
        for driven in (True, False):
            with patch.object(auto, '_axis_surface_coverage', scalar_coverage):
                old = auto.analyze_scenes(before, after, candidate_driven=driven)
            new = auto.analyze_scenes(before, after, candidate_driven=driven)
            for expected, actual in zip(old[:3], new[:3]):
                self.assertEqual(expected, actual)
            def counts(result):
                return {k: v for k, v in result[3].items() if 'seconds' not in k}
            self.assertEqual(counts(old), counts(new))
            self.assertEqual({r['change_typ'] for r in new[0]}, {'added', 'removed', 'widened', 'narrowed'})


if __name__ == '__main__':
    unittest.main()
