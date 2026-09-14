import unittest
import numpy as np
from shapely.geometry import LineString,box
from engine.fast_multitemporal import (RoadContext,reconcile_temporal,estimate_width_bias,
                                      neighbor_results,dependency_periods)


class TemporalTests(unittest.TestCase):
    def row(self,kind):
        axis=LineString([(0,0),(100,0)])
        return dict(change_typ=kind,axis_wkt=axis.wkt,geometry=axis.buffer(4,cap_style='flat'),
                    width_bef=8 if kind=='removed' else 0,width_aft=8 if kind=='added' else 0,
                    start_m=0.,end_m=100.,length_m=100.,v2_publish=True,confidence=.8)

    def context(self,present,valid=True):
        return RoadContext([LineString([(0,0),(100,0)])] if present else [],
                           box(-10,-10,110,10) if valid else box(0,0,1,1),3.)

    def test_transient_and_gap_both_transition_directions(self):
        for kind,side,present in [('added','previous',True),('added','next',False),
                                  ('removed','previous',False),('removed','next',True)]:
            with self.subTest(kind=kind,side=side):
                rows=[self.row(kind)];counts=reconcile_temporal(rows,{side:self.context(present)})
                self.assertFalse(rows[0]['v2_publish'])
                self.assertEqual(counts[f'v2_temporal_{kind}_suppressed'],1)

    def test_persistent_events_remain(self):
        for kind,present in [('added',True),('removed',False)]:
            rows=[self.row(kind)];original=rows[0]['geometry'].wkb
            reconcile_temporal(rows,{'next':self.context(present)})
            self.assertTrue(rows[0]['v2_publish']);self.assertEqual(rows[0]['geometry'].wkb,original)
            self.assertEqual(rows[0]['v2_temporal_state'],'persistent_change')

    def test_unknown_or_invalid_is_not_absence(self):
        for contexts in ({},{'next':self.context(False,False)}):
            rows=[self.row('added')];reconcile_temporal(rows,contexts)
            self.assertTrue(rows[0]['v2_publish'])
            self.assertEqual(rows[0]['v2_temporal_state'],'temporal_context_unknown')

    def test_partial_interval_keeps_only_persistent_regular_corridor(self):
        rows=[self.row('added')]
        context=RoadContext([LineString([(0,0),(50,0)])],box(-10,-10,110,10),3.)
        reconcile_temporal(rows,{'next':context})
        self.assertFalse(rows[0]['v2_publish']);self.assertEqual(len(rows),2)
        child=rows[1];self.assertTrue(child['v2_publish'])
        self.assertAlmostEqual(child['end_m'],53.)
        self.assertTrue(child['geometry'].is_valid)
        self.assertEqual(child['geometry'].bounds,(0.,-4.,53.,4.))

    def test_missing_planned_period_is_not_skipped(self):
        entries=[dict(grid='g',period=p) for p in ['1','3','4','6']]
        result=neighbor_results(entries,'g','3','4',['1','2','3','4','5','6'])
        self.assertEqual(result,{})
        self.assertEqual(dependency_periods(['1','2','3','4','5'],'2','3'),{'1','2','3','4'})

    def test_fast_rerun_refreshes_context_dependent_pairs(self):
        import user_pipeline as p
        manifest=dict(execution_profile='fast',period_results=[dict(grid='g',period=str(i)) for i in range(7)])
        self.assertEqual(p._affected_manifest_pairs(manifest,'g','3'),[('1','2'),('2','3'),('3','4'),('4','5')])
        manifest['execution_profile']='full'
        self.assertEqual(p._affected_manifest_pairs(manifest,'g','3'),[('2','3'),('3','4')])


class WidthBiasTests(unittest.TestCase):
    def test_robust_offset_ignores_minority_true_changes(self):
        result=estimate_width_bias([3.]*40+[12.]*4)
        self.assertEqual(result['bias'],3.);self.assertEqual(result['scatter'],0.)
        self.assertTrue(result['reliable'])

    def test_insufficient_controls_do_not_invent_bias(self):
        self.assertEqual(estimate_width_bias([5.]*10)['bias'],0.)

    def test_population_scatter_is_robust_and_time_symmetric(self):
        samples=np.arange(41)/10
        forward=estimate_width_bias(samples);backward=estimate_width_bias(-samples)
        self.assertEqual(forward['bias'],-backward['bias'])
        self.assertEqual(forward['scatter'],backward['scatter'])

    def test_analyzer_suppresses_population_bias_but_keeps_large_residual(self):
        from test_fast_auto_change import FastFinalAutoTests
        from engine.fast_auto_v2 import analyze_scenes
        from engine import fast_auto_change as legacy
        from unittest.mock import patch
        f=FastFinalAutoTests();f.setUp();self.addCleanup(f.doCleanups)
        # 31 independent narrow roads, one real large widening among controls.
        before=f.scene([f.road(5+7*i,2) for i in range(31)])
        after=f.scene([f.road(5+7*i,7 if i<30 else 14) for i in range(31)])
        with patch.object(legacy.RoadScene,'match',side_effect=AssertionError('point match')), \
             patch.object(legacy,'_measure_period_width',side_effect=AssertionError('remeasurement')):
            rows,_,_,counts=analyze_scenes(before,after)
        self.assertTrue(counts['v2_width_bias_reliable'])
        self.assertEqual(counts['v2_width_bias_m'],5.)
        self.assertGreater(counts['v2_width_bias_suppressed'],0)
        self.assertEqual(counts['v2_width_before_bias_published'],31)
        self.assertEqual(len([r for r in rows if r['change_typ'] in ('widened','narrowed')]),31)
        accepted=[r for r in rows if r['v2_publish'] and r['change_typ'] in ('widened','narrowed')]
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0]['change_typ'],'widened')


if __name__=='__main__':unittest.main()
