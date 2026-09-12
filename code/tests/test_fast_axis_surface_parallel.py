"""Exact serial/threaded axis preparation and bounded worker failure regression."""
import unittest
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


if __name__ == '__main__':
    unittest.main()
