"""Exact serial/threaded axis preparation and bounded worker failure regression."""
import unittest
from collections import Counter
from concurrent.futures import Future
from unittest.mock import patch

from engine import fast_auto_change as auto
import test_fast_auto_change as fixtures
from compare_cached_auto_profiles import encoded


class InlinePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


class AxisSurfaceParallelTests(unittest.TestCase):
    setUp = fixtures.FastFinalAutoTests.setUp
    scene = fixtures.FastFinalAutoTests.scene
    road = staticmethod(fixtures.FastFinalAutoTests.road)

    def test_complete_ordered_candidates_and_audits_match_serial(self):
        before = self.scene([self.road(40, 8), self.road(95, 14), self.road(150)])
        after = self.scene([self.road(40, 14), self.road(95, 8), self.road(205)])
        for driven in (True, False):
            before.surface.cache_clear()
            after.surface.cache_clear()
            serial_presence = []
            with patch('concurrent.futures.ThreadPoolExecutor', InlinePool):
                serial = auto.analyze_scenes(before, after, candidate_driven=driven, presence_audit=serial_presence)
            before.surface.cache_clear()
            after.surface.cache_clear()
            parallel_presence = []
            parallel = auto.analyze_scenes(before, after, candidate_driven=driven, presence_audit=parallel_presence)
            self.assertEqual(encoded(serial[:3]), encoded(parallel[:3]))
            self.assertEqual(encoded(serial_presence), encoded(parallel_presence))
            counts = lambda r: {k:v for k,v in r[3].items() if 'seconds' not in k}
            self.assertEqual(counts(serial), counts(parallel))

    def test_surface_failure_propagates_and_workers_are_joined(self):
        import threading
        before = self.scene([self.road(70)])
        after = self.scene([self.road(70, 14)])
        with patch.object(auto, '_prepare_axis_surface', side_effect=RuntimeError('surface failed')):
            with self.assertRaisesRegex(RuntimeError, 'surface failed'):
                auto.analyze_scenes(before, after)
        self.assertFalse(any(t.name.startswith('auto-axis-surface') for t in threading.enumerate()))

    def test_station_failure_joins_prefetched_workers(self):
        import threading
        before = self.scene([self.road(70), self.road(160)])
        after = self.scene([self.road(70, 14), self.road(160)])
        with patch.object(auto.RoadScene, 'evidence', side_effect=RuntimeError('station failed')):
            with self.assertRaisesRegex(RuntimeError, 'station failed'):
                auto.analyze_scenes(before, after)
        self.assertFalse(any(t.name.startswith('auto-axis-surface') for t in threading.enumerate()))

    def test_pipeline_prefetches_only_one_axis_and_consumes_in_order(self):
        calls = []
        def prepare(scene, axis, tolerance):
            calls.append((scene, axis))
            return axis, axis, 0., 0., 0., 0.
        plans = [(i, i) for i in range(4)]
        with patch.object(auto, '_prepare_axis_surface', prepare):
            stream = auto._axis_surface_pipeline(plans, 'before', 'after', 3., InlinePool(), Counter())
            for i in range(4):
                plan, results = next(stream)
                self.assertEqual(plan[0], i)
                self.assertEqual([v[0] for v in results], [i, i])
                self.assertEqual(len(calls), 2*min(i+2, 4))
                self.assertEqual(calls[-1][1], min(i+1, 3))
            with self.assertRaises(StopIteration):
                next(stream)

    def test_no_selected_stations_never_submit_surface_work(self):
        import numpy as np
        before = self.scene([self.road(70)])
        after = self.scene([self.road(70)])
        with patch('engine.auto_station_plan.presence_mask', side_effect=lambda count,*a:np.zeros(count,dtype=bool)), \
             patch('engine.auto_station_plan.width_mask', side_effect=lambda axis,source,target,stations,*a:np.zeros(len(stations),dtype=bool)), \
             patch.object(auto, '_prepare_axis_surface') as prepare:
            result = auto.analyze_scenes(before, after)
        prepare.assert_not_called()
        self.assertEqual(result[3]['candidate_station_count'], 0)


if __name__ == '__main__':
    unittest.main()
