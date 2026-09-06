import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.auto_width_precision import review_width_profile, qualify_width_candidates, measure_candidate_profile


def profile(sign=1, count=20, start=0.):
    return [dict(valid=True,cross_track=False,junction=False,near_end=False,offset_bias=False,
                 surface_disagreement=False,before_width=8. if sign>0 else 12.,after_width=12. if sign>0 else 8.,
                 measurement_uncertainty_m=.3,pixel_uncertainty_m=.35,signed_offset_m=.5,
                 spacing_m=4.,before_axis=0,before_station_m=start+4*i)
            for i in range(count)]


class WidthPrecisionTests(unittest.TestCase):
    def test_stable_change_is_symmetric(self):
        for sign in (-1,1):
            reasons,stats=review_width_profile(profile(sign),sign)
            self.assertEqual(reasons,[])
            self.assertEqual(stats['width_sustained_length_m'],80.)

    def test_isolated_short_interval_is_review(self):
        reasons,_=review_width_profile(profile(count=6),1)
        self.assertIn('insufficient_sustained_width_change',reasons)

    def test_independent_width_reasons(self):
        for field,reason in [('cross_track','cross_track_width_match'),('junction','junction_width_instability'),
                             ('near_end','road_end_width_instability'),('offset_bias','centerline_offset_measurement_bias'),
                             ('surface_disagreement','surface_geometry_disagreement')]:
            samples=profile()
            for row in samples[:5]:row[field]=True
            reasons,_=review_width_profile(samples,1)
            self.assertIn(reason,reasons)

    def test_local_width_variation_is_not_divided_by_sample_count(self):
        samples=profile(count=80)
        for i,row in enumerate(samples):
            row['before_width']+=4*np.sin(i)
            row['after_width']+=4*np.sin(i)
        reasons,_=review_width_profile(samples,1)
        self.assertIn('uncertainty_not_clearly_exceeded',reasons)

    def test_sign_and_offset_instability_are_review(self):
        samples=profile()
        for row in samples[6:10]:row['after_width']=5.
        for row in samples[10:]:row['signed_offset_m']=2.5
        reasons,_=review_width_profile(samples,1)
        self.assertIn('unstable_width_profile',reasons)
        self.assertIn('centerline_offset_measurement_bias',reasons)

    def test_invalid_samples_break_continuity(self):
        samples=profile()
        for row in samples[4::5]:row['valid']=False
        reasons,stats=review_width_profile(samples,1)
        self.assertLess(stats['width_sustained_length_m'],32.)
        self.assertIn('insufficient_sustained_width_change',reasons)

    def test_opposing_neighbour_runs_go_to_review_without_geometry_loss(self):
        geometries=[LineString([(0,0),(80,0)]).buffer(4),LineString([(90,0),(170,0)]).buffer(4)]
        candidates=gpd.GeoDataFrame(dict(candidate_id=[0,1],change_typ=['widened','narrowed'],
            publication_state=['accepted']*2,audit_reason=['paired_width_accepted']*2),geometry=geometries,crs=32650)
        with patch('engine.auto_width_precision.measure_candidate_profile',side_effect=[profile(1),profile(-1,start=90)]):
            result,_=qualify_width_candidates(candidates,{})
        self.assertTrue(result.width_qa_state.eq('review').all())
        self.assertTrue(result.precision_reason.str.contains('alternating_width_change_signs').all())
        self.assertEqual(result.geometry.to_wkb().tolist(),candidates.geometry.to_wkb().tolist())

    def test_parallel_tracks_are_not_alternating_same_road_runs(self):
        candidates=gpd.GeoDataFrame(dict(candidate_id=[0,1],change_typ=['widened','narrowed'],
            publication_state=['accepted']*2,audit_reason=['paired_width_accepted']*2),
            geometry=[LineString([(0,y),(80,y)]).buffer(4) for y in (0,20)],crs=32650)
        second=profile(-1)
        for row in second:row['before_axis']=1
        with patch('engine.auto_width_precision.measure_candidate_profile',side_effect=[profile(1),second]):
            result,_=qualify_width_candidates(candidates,{})
        self.assertTrue(result.width_qa_state.eq('accepted').all())

    def test_actual_paired_measurement_accepts_stable_widening_and_narrowing(self):
        from test_fast_auto_change import FastFinalAutoTests
        fixture=FastFinalAutoTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        first=fixture.scene([fixture.road(100,8)])
        second=fixture.scene([fixture.road(101,14)])
        axis=LineString([(80,100.5),(180,100.5)])
        for sign,scenes in [(1,dict(before=first,after=second)),(-1,dict(before=second,after=first))]:
            row=__import__('pandas').Series(dict(candidate_id=0,axis_wkt=axis.wkt,width_bef=8 if sign>0 else 14))
            samples=measure_candidate_profile(row,scenes)
            reasons,_=review_width_profile(samples,sign)
            self.assertEqual(reasons,[])

    def test_actual_nearer_parallel_lane_defeats_reverse_width_match(self):
        from test_fast_auto_change import FastFinalAutoTests
        fixture=FastFinalAutoTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        before=fixture.scene([fixture.road(100,2),fixture.road(104,2)])
        after=fixture.scene([fixture.road(102.8,3),fixture.road(106.8,3)])
        row=__import__('pandas').Series(dict(candidate_id=0,width_bef=2,
                            axis_wkt=LineString([(80,101.4),(180,101.4)]).wkt))
        samples=measure_candidate_profile(row,dict(before=before,after=after))
        reasons,_=review_width_profile(samples,1)
        self.assertIn('cross_track_width_match',reasons)

    def test_shared_surface_with_centerline_shift_is_not_width_change(self):
        from test_fast_auto_change import FastFinalAutoTests
        fixture=FastFinalAutoTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        before=fixture.scene([fixture.road(100,8)])
        after=fixture.scene([fixture.road(101.5,8)],surfaces=list(before.surfaces.geometry))
        row=__import__('pandas').Series(dict(candidate_id=0,width_bef=8,
                            axis_wkt=LineString([(80,100.75),(180,100.75)]).wkt))
        samples=measure_candidate_profile(row,dict(before=before,after=after))
        reasons,_=review_width_profile(samples,1)
        self.assertIn('insufficient_sustained_width_change',reasons)


if __name__=='__main__':unittest.main()
