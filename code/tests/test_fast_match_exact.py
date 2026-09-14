"""Frozen pre-P3 match oracle: strict scores, order, and query geometry bytes."""
import struct
import sys
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring
from shapely.strtree import STRtree

from engine.fast_auto_change import RoadScene, _normal


def _original_match(self, axis, station, tolerance, source_width, *, point=None, normal=None):
    """Frozen production method before direction/corridor performance changes."""
    point = axis.interpolate(station) if point is None else point
    local = substring(axis, max(0, station-6), min(axis.length, station+6))
    normal = _normal(axis, station) if normal is None else normal
    ranked = []
    for target_id in self.tree.query(point, predicate="dwithin", distance=tolerance+1e-6):
        target = self.lines[int(target_id)]
        target_station = target.project(point)
        target_point = target.interpolate(target_station)
        distance = point.distance(target_point)
        cosine = abs(float(np.dot(normal, _normal(target, target_station))))
        target_local = substring(target, max(0, target_station-6), min(target.length, target_station+6))
        overlap = local.intersection(target_local.buffer(tolerance+0.1)).length / max(local.length, 1e-9)
        if cosine < .90 or overlap < .50:
            continue
        target_width = self.width(target_point)
        corridor_overlap = local.buffer(source_width/2).intersection(target_local.buffer(target_width/2)).area
        corridor_overlap /= max(min(local.length*source_width, target_local.length*target_width), 1e-9)
        compatibility = min(source_width, target_width) / max(source_width, target_width)
        # Distance dominates: width is supporting evidence, never a veto of real widening.
        score = distance + 2*(1-cosine) + (1-overlap) + .25*(1-corridor_overlap) + .15*(1-compatibility)
        ranked.append((score, int(target_id), target_station, distance, cosine, overlap, corridor_overlap))
    ranked.sort()
    if not ranked:
        return None
    best = ranked[0]
    ambiguous = False
    if len(ranked) > 1 and ranked[1][0]-best[0] < .5:
        other = self.lines[ranked[1][1]].interpolate(ranked[1][2])
        # Two features meeting at one node are segmentation, distinct nearby axes are ambiguous tracks.
        ambiguous = other.distance(self.lines[best[1]].interpolate(best[2])) > 1.0
    return {"target": best[1], "station": best[2], "distance": best[3],
            "direction": best[4], "coverage": best[5], "corridor": best[6],
            "reliable": not ambiguous}


def _bytes(value):
    """No tolerance, rounding, normalized geometry, or dictionary key sorting."""
    if isinstance(value, np.generic):
        return ('numpy', value.dtype.str, value.tobytes())
    if isinstance(value, float):
        return ('float', struct.pack('!d', value))
    if isinstance(value, dict):
        return tuple((key, _bytes(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_bytes(item) for item in value))
    if hasattr(value, 'wkb'):
        return ('geometry', value.wkb)
    return (type(value).__name__, value)


class _RecordedTree:
    def __init__(self, lines):
        self.tree = STRtree(lines)
        self.calls = []

    def query(self, point, **kwargs):
        ids = self.tree.query(point, **kwargs)
        self.calls.append((point.wkb, _bytes(kwargs), ids.dtype.str, ids.tobytes()))
        return ids


def _scene(lines, target_width):
    scene = RoadScene.__new__(RoadScene)
    scene.lines = lines
    scene.tree = _RecordedTree(lines)
    scene.width_calls = []

    def width(point):
        value = target_width(point) if callable(target_width) else target_width
        scene.width_calls.append((point.wkb, _bytes(value)))
        return value
    scene.width = width
    return scene


def _invoke(function, scene, axis, station, tolerance, width, options):
    """Observe local ranked on function return without changing its statements."""
    ranked = []

    def profile(frame, event, result):
        if frame.f_code is function.__code__ and event == 'return':
            ranked.append(_bytes(frame.f_locals['ranked']))
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = function(scene, axis, station, tolerance, width, **options)
    finally:
        sys.setprofile(previous)
    if len(ranked) != 1:
        raise AssertionError('expected exactly one ranked capture')
    return result, ranked[0]


class FastMatchExactTests(unittest.TestCase):
    def compare(self, axis, lines, stations, *, source_width=8., target_width=8., tolerance=3.,
                normal_override=None):
        results = []
        for station in stations:
            for explicit in (False, True):
                with self.subTest(station=station, explicit=explicit, source_width=source_width):
                    options = dict(point=axis.interpolate(station), normal=_normal(axis, station)) if explicit else {}
                    if normal_override is not None:
                        options['normal'] = normal_override
                    expected_scene, actual_scene = _scene(lines, target_width), _scene(lines, target_width)
                    expected, expected_ranked = _invoke(_original_match, expected_scene, axis, station,
                                                       tolerance, source_width, options)
                    actual, actual_ranked = _invoke(RoadScene.match, actual_scene, axis, station,
                                                   tolerance, source_width, options)
                    self.assertEqual(_bytes(expected), _bytes(actual))
                    self.assertEqual(expected_ranked, actual_ranked)
                    self.assertEqual(expected_scene.tree.calls, actual_scene.tree.calls)
                    self.assertEqual(expected_scene.width_calls, actual_scene.width_calls)
                    if actual is not None:
                        self.assertEqual(list(actual), ['target', 'station', 'distance', 'direction',
                                                        'coverage', 'corridor', 'reliable'])
                    results.append(actual)
        return results

    def test_straight_reversed_and_endpoint_stations(self):
        points = [(0, 0), (30, 0), (80, 0)]
        for reverse_axis in (False, True):
            axis = LineString(points[::-1] if reverse_axis else points)
            for reverse_target in (False, True):
                target_points = [(0, 1.25), (80, 1.25)]
                target = LineString(target_points[::-1] if reverse_target else target_points)
                self.compare(axis, [target], [0., .01, 6., 30., 74., axis.length])

    def test_curved_large_coordinates_and_variable_width(self):
        points = [(500000., 4000000.), (500018., 4000001.75), (500034., 4000008.),
                  (500061., 4000005.125), (500085., 4000011.5)]
        for reverse in (False, True):
            axis = LineString(points[::-1] if reverse else points)
            lines = [LineString([(x+.125, y+1.25) for x, y in points]),
                     LineString([(x-.25, y-2.125) for x, y in points[::-1]])]
            for source_width in (3.5, 8.125, 23.75):
                self.compare(axis, lines, [0., 2., 18., 37.125, 63., axis.length], source_width=source_width,
                             target_width=lambda point: 4.125 + (point.x-500000.)/31.)

    def test_direction_and_insufficient_overlap_rejections(self):
        axis = LineString([(0, 0), (80, 0)])
        lines = [LineString([(22, -20), (22, 20)]),
                 LineString([(21.9, 2.99), (22.1, 2.99)])]
        results = self.compare(axis, lines, [22.])
        self.assertTrue(all(result is None for result in results))
        # Add a valid road, ensuring rejected observations cannot alter ranking.
        lines.append(LineString([(0, 1), (80, 1)]))
        results = self.compare(axis, lines, [22.])
        self.assertTrue(all(result['target'] == 2 for result in results))

    def test_parallel_tracks_tie_and_ambiguity(self):
        axis = LineString([(0, 0), (80, 0)])
        lines = [LineString([(0, -1), (80, -1)]), LineString([(0, 1), (80, 1)])]
        results = self.compare(axis, lines, [20., 40., 60.])
        self.assertTrue(all(result is not None and not result['reliable'] for result in results))
        # Duplicate geometry makes score equality exact; target id breaks ties.
        results = self.compare(axis, [lines[0], lines[0]], [40.])
        self.assertTrue(all(result['target'] == 0 and result['reliable'] for result in results))

    def test_crossing_junction_and_same_node_segmentation(self):
        axis = LineString([(-40, 0), (40, 0)])
        lines = [LineString([(-40, 0), (0, 0)]), LineString([(0, 0), (40, 0)]),
                 LineString([(0, -30), (0, 30)])]
        results = self.compare(axis, lines, [34., 40., 46.])
        self.assertTrue(all(result is not None and result['reliable'] for result in results))

    def test_empty_and_outside_query(self):
        axis = LineString([(0, 0), (80, 0)])
        for lines in ([], [LineString([(0, 30), (80, 30)])]):
            results = self.compare(axis, lines, [0., 20., 80.])
            self.assertTrue(all(result is None for result in results))

    def test_width_differences_and_query_boundary(self):
        axis = LineString([(0, 0), (80, 0)])
        lines = [LineString([(0, 3.), (80, 3.)]),
                 LineString([(0, 3.0000005), (80, 3.0000005)]),
                 LineString([(0, 3.000002), (80, 3.000002)]),
                 LineString([(0, -.25), (80, -.25)])]
        for source_width in (2.125, 8., 30.):
            self.compare(axis, lines, [0., 4., 20., 76., 80.], source_width=source_width,
                         target_width=lambda point: 2.5 if point.y < 0 else 24.125)

    def test_direction_threshold_adjacent_float_bytes_and_nan(self):
        axis = LineString([(0, 0), (80, 0)])
        lines = [LineString([(0, 1), (80, 1)])]
        values = [np.nextafter(.90, 0.), .90, np.nextafter(.90, 1.)]
        for cosine in values:
            with self.subTest(cosine=cosine.hex()):
                normal = np.array([np.sqrt(1-cosine*cosine), cosine])
                # Horizontal target normal is exactly (0, 1), so these input
                # floats exercise the threshold and its adjacent IEEE values.
                self.assertEqual(struct.pack('!d', cosine),
                                 struct.pack('!d', abs(float(np.dot(normal, _normal(lines[0], 20.))))))
                results = self.compare(axis, lines, [20.], normal_override=normal)
                if cosine < .90:
                    self.assertTrue(all(result is None for result in results))
                else:
                    self.assertTrue(all(result is not None for result in results))
        # Preserve the original '<' behavior: NaN is not below the threshold.
        # Rewriting the gate as 'not cosine >= threshold' would reject it.
        results = self.compare(axis, lines, [20.], normal_override=np.array([0., np.nan]))
        self.assertTrue(all(result is not None and np.isnan(result['direction']) for result in results))

    def test_numpy_width_types_and_signed_zero(self):
        axis = LineString([(0, 0), (20, .25), (80, 0)])
        lines = [LineString([(0, 1), (80, 1)]), LineString([(0, -.5), (80, -.5)])]
        for scalar_type in (float, np.float32, np.float64):
            for width in (0., -0., 3.125, 17.25):
                with self.subTest(scalar_type=scalar_type.__name__, width=width.hex()):
                    self.compare(axis, lines, [0., 20., axis.length], source_width=scalar_type(width),
                                 target_width=scalar_type(8.125))

    def test_no_source_corridor_for_empty_or_all_rejected_queries(self):
        axis = LineString([(0, 0), (80, 0)])
        local_wkb = substring(axis, 14., 26.).wkb
        original_buffer = BaseGeometry.buffer
        cases = [[], [LineString([(0, 30), (80, 30)])],
                 [LineString([(20, -30), (20, 30)])],
                 [LineString([(19.9, 2.99), (20.1, 2.99)])]]
        for lines in cases:
            for method in (_original_match, RoadScene.match):
                with self.subTest(method=method.__name__, lines=len(lines)):
                    source_buffers = []

                    def buffer(geometry, *args, **kwargs):
                        if geometry.wkb == local_wkb:
                            source_buffers.append((_bytes(args), _bytes(kwargs)))
                        return original_buffer(geometry, *args, **kwargs)
                    scene = _scene(lines, 8.)
                    with patch.object(BaseGeometry, 'buffer', buffer):
                        result = method(scene, axis, 20., 3., 8.)
                    self.assertIsNone(result)
                    self.assertEqual(scene.width_calls, [])
                    self.assertEqual(source_buffers, [])

    def test_width_exception_precedes_lazy_source_corridor(self):
        axis = LineString([(0, 0), (80, 0)])
        local_wkb = substring(axis, 14., 26.).wkb
        lines = [LineString([(0, 1), (80, 1)]), LineString([(0, -1), (80, -1)]),
                 LineString([(20, -30), (20, 30)])]
        original_buffer = BaseGeometry.buffer
        observed = []
        for method in (_original_match, RoadScene.match):
            source_buffers, width_queries = [], []

            def buffer(geometry, *args, **kwargs):
                if geometry.wkb == local_wkb:
                    source_buffers.append((_bytes(args), _bytes(kwargs)))
                return original_buffer(geometry, *args, **kwargs)

            def failing_width(point):
                width_queries.append(point.wkb)
                raise RuntimeError('width lookup failed')
            scene = _scene(lines, 8.)
            scene.width = failing_width
            with patch.object(BaseGeometry, 'buffer', buffer):
                with self.assertRaisesRegex(RuntimeError, '^width lookup failed$') as caught:
                    method(scene, axis, 20., 3., 8.)
            self.assertEqual(source_buffers, [])
            self.assertEqual(len(width_queries), 1)
            observed.append((type(caught.exception), str(caught.exception), width_queries, scene.tree.calls))
        self.assertEqual(observed[0], observed[1])


if __name__ == '__main__':
    unittest.main()
