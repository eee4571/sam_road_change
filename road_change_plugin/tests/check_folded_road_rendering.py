"""Folded roads remain renderable; use cached profiles, never model inference."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.errors import GEOSException

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT.parent / 'code/tests')]
import engine
engine.__path__ = [str(ROOT / 'code/engine')]
import test_fast_gt_reconciliation as fixtures
from engine.auto_change_geometry import corridor
from engine.canonical_road_surface import _chain_surface, _junction
from engine.continuous_road_geometry import network_surface
from engine.geometry_recovery import Recovery, recovery_scope
from engine.road_axis_quality import axis_quality
from engine.fast_gt_reconciliation import reconcile_periods, _publish_assisted_changes, _frame
from temporal_road_analysis import build_temporal_grid


def folded_axis():
    xy = [(0, 0)]
    for x in range(0, 100, 10):
        xy.extend([(x + 4, 0), (x + 4, 6), (x + 8, 6), (x + 8, 0)])
    return LineString(xy + [(100, 0)])


class FoldedRoadTests(unittest.TestCase):
    def test_fold_crossing_and_backtracking_keep_complete_axis(self):
        axes = [folded_axis(), LineString([(0, 0), (10, 10), (0, 10), (10, 0)]),
                LineString([(0, 0), (20, 0), (5, 0), (30, 0)])]
        for axis in axes:
            original = axis.wkb
            for widths in ([8., 8.], [8., 12.]):
                for render in (corridor, _chain_surface):
                    with self.subTest(axis=axis.wkt, widths=widths, render=render.__name__):
                        area = render(axis, [0., axis.length], widths)
                        self.assertTrue(area.is_valid)
                        self.assertFalse(area.is_empty)
                        self.assertLess(axis.difference(area.buffer(1e-5)).length, 1e-5)
                        self.assertEqual(axis.wkb, original)

    def test_quality_notice_is_audited_once_and_not_skipped(self):
        axis = folded_axis(); recovery = Recovery()
        with recovery_scope(recovery):
            corridor(axis, [0., axis.length], [8., 12.])
            corridor(axis, [0., axis.length], [9., 13.])
        self.assertEqual(recovery.summary()['warning_count'], 1)
        self.assertEqual(recovery.summary()['skipped_count'], 0)
        self.assertEqual(recovery.failures[0]['final_action'], 'retain_valid_geometry')

    def test_network_keeps_folded_and_unrelated_road(self):
        axes = [folded_axis(), LineString([(0, 50), (100, 50)])]
        area = network_surface([(a, [0., a.length], [8., 12.]) for a in axes])
        self.assertTrue(area.is_valid)
        for axis in axes:
            self.assertLess(axis.difference(area.buffer(1e-5)).length, 1e-5)

    def test_junction_keeps_folded_stub_and_internal_connection(self):
        axis = folded_axis()
        portal = dict(point=np.array([100., 0.]), width=8.,
                      direction=np.array([1., 0.]), left=np.array([100., 4.]),
                      right=np.array([100., -4.]))
        for required in ([axis], [axis, LineString([(0, 0), (100, 0)])]):
            area, _ = _junction([portal], np.array([0., 0.]), required)
            self.assertTrue(area.is_valid)
            self.assertLess(axis.difference(area.buffer(1e-5)).length, 1e-5)

    def test_local_overlay_failure_uses_buffer_and_continues(self):
        axis = folded_axis(); recovery = Recovery()
        with recovery_scope(recovery), patch('engine.auto_change_geometry._swept_corridor',
                                             side_effect=GEOSException('local overlay failed')):
            area = corridor(axis, [0., axis.length], [8., 12.])
        self.assertTrue(area.equals(axis.buffer(5., cap_style='flat', join_style='round')))
        self.assertEqual(recovery.summary()['fallback_count'], 1)

    def test_shared_resource_errors_are_not_hidden(self):
        axis = folded_axis()
        for error in (MemoryError('memory'), PermissionError('file'), ValueError('out of memory')):
            with self.subTest(error=type(error).__name__), patch(
                    'engine.auto_change_geometry._swept_corridor', side_effect=error):
                with self.assertRaises(type(error)):
                    corridor(axis, [0., axis.length], [8., 12.])

    def test_profile_peak_fold_finishes_reconciliation_events_and_temporal(self):
        # Reconciliation checks scalar width 4, while the saved profile peaks at
        # 16. Previously corridor's stricter check stopped the entire grid here.
        axis = folded_axis(); other = LineString([(0, 70), (100, 70)])
        self.assertFalse(axis_quality(axis, 4.)['abnormal'])
        self.assertTrue(axis_quality(axis, 16.)['abnormal'])
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); fixture = fixtures.FastGTReconciliationTests()
            periods = [fixture.period(root, name, [(axis, width), (other, 6.)])
                       for name, width in [('T1', 2.), ('T2', 4.)]]
            profile = json.dumps(dict(axis=axis.wkt, stations=[0., .5, 1.],
                                      widths=[2., 8., 2.], reference_width=2.))
            row = dict(change_id='fold', change_typ='widened', change_src='AUTO',
                       width_bef=2., width_aft=4., axis_wkt=axis.wkt,
                       match_axis_wkt=axis.wkt, gt_width_profile=profile,
                       geometry=axis.buffer(8.).difference(axis.buffer(1.)))
            pairs = [dict(frame=_frame([row], 3857), before_period='T1', after_period='T2')]
            outputs = reconcile_periods(periods, pairs, root / 'final')
            self.assertEqual(pairs[0]['frame'].iloc[0].axis_wkt, axis.wkt)
            published = _publish_assisted_changes(pairs[0]['frame'], root / 'change', 'T1', 'T2')
            self.assertTrue((root / 'change/road_changes.shp').is_file())
            self.assertTrue(gpd.read_file(root / 'change/road_changes.shp').is_valid.all())
            result = build_temporal_grid('region', outputs,
                [dict(published, output=str(root / 'change'), before_period='T1', after_period='T2')],
                root / 'temporal')
            track = pairs[0]['frame'].iloc[0].track_id
            life = gpd.read_file(result['life_shp'])
            events = gpd.read_file(result['events_shp'])
            self.assertIn(track, set(life.road_id))
            self.assertIn(track, set(events.road_id))
            for period in outputs:
                roads = gpd.read_file(period['centerlines'])
                self.assertLess(other.difference(roads.geometry.union_all()).length, 1e-5)


if __name__ == '__main__':
    unittest.main()
