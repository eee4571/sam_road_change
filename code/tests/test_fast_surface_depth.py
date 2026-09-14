"""Byte-exact output and resource bounds for the surface concurrency experiment."""
from collections import Counter
from concurrent.futures import Future
import threading
import unittest
from unittest.mock import patch

import numpy as np

from engine import fast_auto_change as auto
from compare_cached_auto_profiles import encoded
from surface_pipeline_experiment import install
from test_fast_axis_surface_parallel import InlinePool
import test_fast_auto_change as fixtures


class SurfaceDepthTests(unittest.TestCase):
    scene = fixtures.FastFinalAutoTests.scene
    road = staticmethod(fixtures.FastFinalAutoTests.road)

    def setUp(self):
        fixtures.FastFinalAutoTests.setUp(self)
        self.original_analyze = auto.analyze_scenes
        self.original_pipeline = auto._axis_surface_pipeline
        self.addCleanup(setattr, auto, 'analyze_scenes', self.original_analyze)
        self.addCleanup(setattr, auto, '_axis_surface_pipeline', self.original_pipeline)

    def test_exact_candidates_all_audits_and_statistics(self):
        before = self.scene([self.road(40, 8), self.road(95, 14), self.road(150)])
        after = self.scene([self.road(40, 14), self.road(95, 8), self.road(205)])
        for driven in (True, False):
            auto._axis_surface_pipeline = self.original_pipeline
            baseline_presence = []
            baseline = self.original_analyze(before, after, candidate_driven=driven,
                                             presence_audit=baseline_presence)
            for workers, depth in ((2, 1), (3, 2), (4, 2)):
                with self.subTest(workers=workers, depth=depth, driven=driven):
                    install(auto, workers, depth)
                    before.surface.cache_clear()
                    after.surface.cache_clear()
                    presence = []
                    result = auto.analyze_scenes(before, after, candidate_driven=driven,
                                                 presence_audit=presence)
                    self.assertEqual(encoded(baseline[:3]), encoded(result[:3]))
                    self.assertEqual(encoded(baseline_presence), encoded(presence))
                    counts = lambda r: {k: v for k, v in r[3].items() if 'seconds' not in k}
                    self.assertEqual(encoded(counts(baseline)), encoded(counts(result)))

    def test_at_most_two_future_effective_axes_and_order(self):
        for depth in (1, 2):
            install(auto, 4, depth)
            calls = []
            planned = []
            def plans():
                for index in (0, 3, 7, 8, 12):
                    planned.append(index)
                    yield (index, index)
            def prepare(scene, axis, tolerance):
                calls.append((scene, axis))
                return axis, axis, 0., 0., 0., 0.
            with patch.object(auto, '_prepare_axis_surface', prepare):
                stream = auto._axis_surface_pipeline(plans(), 'before', 'after', 3., InlinePool(), Counter())
                for count, expected in enumerate((0, 3, 7, 8, 12), 1):
                    plan, results = next(stream)
                    self.assertEqual(plan[0], expected)
                    self.assertEqual([r[0] for r in results], [expected, expected])
                    self.assertEqual(len(planned), min(count+depth, 5))
                    self.assertEqual(len(calls), 2*min(count+depth, 5))
                with self.assertRaises(StopIteration):
                    next(stream)

    def test_empty_selected_never_submits(self):
        install(auto, 4, 2)
        before = self.scene([self.road(70)])
        after = self.scene([self.road(70)])
        with patch('engine.auto_station_plan.presence_mask',
                   side_effect=lambda count, *a: np.zeros(count, dtype=bool)), \
             patch('engine.auto_station_plan.width_mask',
                   side_effect=lambda axis, source, target, stations, *a: np.zeros(len(stations), dtype=bool)), \
             patch.object(auto, '_prepare_axis_surface') as prepare:
            result = auto.analyze_scenes(before, after)
        prepare.assert_not_called()
        self.assertEqual(result[3]['candidate_station_count'], 0)

    def test_early_close_cancels_all_queued_work(self):
        install(auto, 4, 2)
        class DeferredPool:
            def __init__(self):
                self.jobs = []
            def submit(self, function, scene, axis, tolerance):
                future = Future()
                if axis == 0:
                    future.set_result((axis, axis, 0., 0., 0., 0.))
                self.jobs.append(future)
                return future
        pool = DeferredPool()
        stream = auto._axis_surface_pipeline([(i, i) for i in range(8)],
                                             'before', 'after', 3., pool, Counter())
        self.assertEqual(next(stream)[0][0], 0)
        self.assertEqual(len(pool.jobs), 6)
        stream.close()
        self.assertTrue(all(job.cancelled() for job in pool.jobs[2:]))

    def test_surface_and_station_failure_join_workers(self):
        before = self.scene([self.road(70), self.road(160)])
        after = self.scene([self.road(70, 14), self.road(160, 14)])
        for workers in (3, 4):
            install(auto, workers, 2)
            for owner, name in ((auto, '_prepare_axis_surface'), (auto.RoadScene, 'evidence')):
                before.surface.cache_clear()
                after.surface.cache_clear()
                with self.subTest(workers=workers, failure=name), \
                     patch.object(owner, name, side_effect=RuntimeError('controlled failure')):
                    with self.assertRaisesRegex(RuntimeError, 'controlled failure'):
                        auto.analyze_scenes(before, after)
                self.assertFalse(any(t.name.startswith('auto-axis-surface') for t in threading.enumerate()))


if __name__ == '__main__':
    unittest.main()
