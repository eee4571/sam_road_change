"""Explicit backend regression: synthetic geometry only, no models or projects.

Run with the plugin backend interpreter. Also runs the development repository's
existing road-network tests against the plugin copy when executed as a script.
Kept outside host unittest discovery because it imports backend dependencies.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import engine
engine.__path__ = [str(ROOT / 'code/engine')]
import numpy as np
from shapely.geometry import LineString
from engine import road_network_connection as network
from engine.road_geometry import _RegionalRoadSeed, _polyline_length


def seed(points, width=10., source=0):
    return _RegionalRoadSeed(np.asarray(points, dtype=float).reshape(-1, 2), width, (source,))


class DegenerateRoadTests(unittest.TestCase):
    def test_endpoint_snapping_collapse_preserves_crossing_road(self):
        # The 2-micrometre span has positive length before both endpoints snap
        # to the vertical road. Previously _join_chains raised ZeroDivisionError.
        roads = [seed([[0, 0], [2e-6, 0]], width=80., source=0),
                 seed([[1e-6, -1], [1e-6, 1]], width=10., source=1)]
        for order in (roads, roads[::-1]):
            with self.subTest(order=[r.source_ids for r in order]):
                noded = network._node_network(order)
                self.assertTrue(all(_polyline_length(r.points) > 0 for r in noded))
                joined = network._join_chains(noded)
                self.assertEqual(len(joined), 1)
                self.assertEqual(joined[0].source_ids, (1,))
                self.assertEqual(joined[0].width_m, 10.)
                self.assertTrue(LineString(joined[0].points).equals(LineString(roads[1].points)))
                # Covers the production _apply -> noding -> joining path too.
                applied = network._apply(order, [])
                self.assertEqual(len(applied), 1)
                self.assertTrue(LineString(applied[0].points).equals(LineString(roads[1].points)))

    def test_empty_point_and_repeated_points_have_no_network_edge(self):
        for roads in ([], [seed([])], [seed([[0, 0]])],
                      [seed([[0, 0], [0, 0], [0, 0]])]):
            with self.subTest(roads=roads):
                self.assertEqual(network._node_network(roads), [])
                self.assertEqual(network._join_chains(roads), [])
                self.assertEqual(network._apply(roads, []), [])

    def test_degenerate_neighbor_does_not_affect_width_or_sources(self):
        roads = [seed([[0, 0], [10, 0]], width=4., source=1),
                 seed([[10, 0], [10, 0]], width=999., source=2),
                 seed([[10, 0], [40, 0]], width=8., source=3)]
        for order in (roads, roads[::-1]):
            joined = network._join_chains(order)
            self.assertEqual(len(joined), 1)
            self.assertEqual(joined[0].width_m, 7.)  # (10*4 + 30*8) / 40
            self.assertEqual(joined[0].source_ids, (1, 3))
            self.assertAlmostEqual(_polyline_length(joined[0].points), 40.)

    def test_positive_closed_loop_is_not_removed(self):
        road = seed([[0, 0], [10, 0], [10, 10], [0, 0]])
        joined = network._join_chains(network._node_network([road]))
        self.assertEqual(len(joined), 1)
        np.testing.assert_array_equal(joined[0].points, road.points)
        self.assertEqual(joined[0].width_m, road.width_m)

    def test_short_positive_road_is_retained(self):
        for length in (2e-6, 1e-4):
            road = seed([[0, 0], [length, 0]])
            joined = network._join_chains(network._node_network([road]))
            self.assertEqual(len(joined), 1)
            np.testing.assert_array_equal(joined[0].points, road.points)

    def test_chain_collapsing_during_deduplication_emits_no_point(self):
        # Each step is below the existing deduplication tolerance, even though
        # the sum of steps is positive; a point is not a usable road output.
        road = seed([[0, 0], [5e-9, 0], [1e-8, 0], [1.5e-8, 0]])
        self.assertEqual(network._join_chains([road]), [])


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT.parent / 'code/tests'))
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(DegenerateRoadTests),
        unittest.defaultTestLoader.loadTestsFromName('test_road_network_connection'),
    ])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    for name, module in list(sys.modules.items()):
        if name.startswith('engine.') and getattr(module, '__file__', None):
            assert Path(module.__file__).resolve().is_relative_to(ROOT / 'code'), name
    raise SystemExit(not result.wasSuccessful())
