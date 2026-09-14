"""Byte-exact probe regressions, without raster or model inference."""
import inspect
import linecache
import struct
import textwrap
import types
import unittest

import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree

import engine.fast_auto_change as fast
from match_probe import install


def _exact(value):
    if isinstance(value, float):
        return ('float', struct.pack('!d', value))
    if isinstance(value, np.generic):
        return ('numpy', value.dtype.str, value.tobytes())
    if isinstance(value, dict):
        return tuple((key, _exact(item)) for key, item in value.items())
    if hasattr(value, 'wkb'):
        return value.wkb
    return value


def _module(function=fast.RoadScene.match):
    module = types.ModuleType('probe_experiment')
    module.__dict__.update(fast.__dict__)
    module.RoadScene = type('ProbeRoadScene', (fast.RoadScene,), {'match': function})
    return module


def _scene(module, lines):
    scene = module.RoadScene.__new__(module.RoadScene)
    scene.lines = lines
    scene.tree = STRtree(lines)
    scene.width_calls = []

    def width(point):
        scene.width_calls.append(point.wkb)
        return 7.5 + float(point.y)/10
    scene.width = width
    return scene


class MatchProbeTests(unittest.TestCase):
    def setUp(self):
        self.axis = LineString([(0, 0), (20, 0), (40, 3), (80, 3)])
        self.lines = [LineString([(0, 1), (80, 1)]),
                      LineString([(15, -30), (15, 30)]),
                      LineString([(22, 2.9), (23, 2.9)]),
                      LineString([(0, 2), (80, 2)]),
                      LineString([(0, 50), (80, 50)])]

    def compare(self, module, *, detailed):
        original = module.RoadScene.match
        expected_scene = _scene(module, self.lines)
        stations = [0., 1., 10., 15., 22., 38., 65., self.axis.length]
        calls = [(station, {}) for station in stations] + [
            (station, dict(point=self.axis.interpolate(station), normal=fast._normal(self.axis, station)))
            for station in stations]
        expected = [_exact(original(expected_scene, self.axis, station, 3., 8., **options))
                    for station, options in calls]
        report = install(module, detailed=detailed)
        actual_scene = _scene(module, self.lines)
        actual = [_exact(actual_scene.match(self.axis, station, 3., 8., **options))
                  for station, options in calls]
        self.assertEqual(expected, actual)
        self.assertEqual(expected_scene.width_calls, actual_scene.width_calls)
        result = report()
        self.assertEqual(result['counts']['match_calls'], len(calls))
        self.assertGreater(result['seconds']['match_total'], 0)
        if detailed:
            counts = result['counts']
            self.assertEqual(counts['query_candidates'], counts['candidate_visits'])
            self.assertEqual(counts['query_candidates'], counts.get('ranked_candidates', 0) +
                             counts.get('direction_rejected', 0) + counts.get('overlap_rejected', 0))
            self.assertGreater(counts['direction_rejected'], 0)
            self.assertGreater(counts['overlap_rejected'], 0)
            for stage in ('strtree_query', 'source_interpolate', 'source_substring', 'source_normal',
                          'target_project_interpolate_distance', 'target_normal_cosine',
                          'target_substring', 'overlap_buffer_intersection', 'width_lookup',
                          'corridor_buffer_intersection', 'score_rank', 'sort_ambiguity'):
                self.assertGreater(result['seconds'][stage], 0, stage)
        return result

    def test_trace_keeps_result_and_width_query_bytes(self):
        self.compare(_module(), detailed=False)

    def test_detailed_keeps_result_and_width_query_bytes(self):
        self.compare(_module(), detailed=True)

    def test_detailed_accepts_early_direction_and_source_corridor_cache(self):
        source = textwrap.dedent(inspect.getsource(fast.RoadScene.match))
        # Construct the independently requested prospective optimization for
        # probe compatibility only; never patch the production module.
        if 'if cosine < .90 or overlap < .50:' in source:
            source = source.replace('        target_local = ', '        if cosine < .90:\n            continue\n        target_local = ', 1)
            source = source.replace('if cosine < .90 or overlap < .50:', 'if overlap < .50:')
        if 'local.buffer(source_width/2).intersection' in source:
            source = source.replace('    ranked = []', '    local_length = local.length\n    source_corridor = local.buffer(source_width/2)\n    ranked = []', 1)
            source = source.replace('local.buffer(source_width/2).intersection', 'source_corridor.intersection')
            source = source.replace('max(local.length, 1e-9)', 'max(local_length, 1e-9)')
            source = source.replace('min(local.length*source_width,', 'min(local_length*source_width,')
        filename = '<match-probe-prospective-source>'
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = dict(fast.__dict__)
        exec(compile(source, filename, 'exec'), namespace)
        result = self.compare(_module(namespace['match']), detailed=True)
        self.assertGreater(result['seconds']['source_corridor_buffer'], 0)

    def test_empty_query_records_sort_and_returns_none(self):
        module = _module()
        report = install(module, detailed=True)
        scene = _scene(module, [])
        self.assertIsNone(scene.match(self.axis, 5., 3., 8.))
        result = report()
        self.assertEqual(result['counts']['query_candidates'], 0)
        self.assertGreater(result['seconds']['sort_ambiguity'], 0)

    def test_trace_propagates_exception_and_times_failed_call(self):
        def failing(*args, **kwargs):
            raise RuntimeError('original failure')
        module = _module(failing)
        report = install(module)
        with self.assertRaisesRegex(RuntimeError, 'original failure'):
            _scene(module, []).match(None, 0, 0, 0)
        self.assertEqual(report()['counts']['match_calls'], 1)
        self.assertGreater(report()['seconds']['match_total'], 0)


if __name__ == '__main__':
    unittest.main()
