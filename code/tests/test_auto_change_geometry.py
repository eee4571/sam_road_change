import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.auto_change_geometry import corridor, FinalWidths, render_change_geometry, _clean_overlay
from engine.auto_change_assembly import assemble_change_objects


def frame(rows):
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=32650)


class GeometryTests(unittest.TestCase):
    def fixtures(self, kind='added', gap=True):
        axes = [LineString([(0, 0), (40, 0)])]
        if gap:
            axes.append(LineString([(60, 0), (100, 0)]))
        seeds = frame([dict(candidate_id=i, change_typ=kind, width_bef=8 if kind != 'added' else 0,
                            width_aft=12 if kind != 'removed' else 0, length_m=a.length,
                            axis_wkt=a.wkt, qa_state='confirmed', geometry=a.buffer(2).difference(box(10,-1,12,1)))
                       for i,a in enumerate(axes)])
        roads = frame([dict(geometry=LineString([(0,0),(50,0),(100,0)])),
                       dict(geometry=LineString([(50,0),(50,50)]))])
        widths = {side: frame([dict(width_m=w,geometry=roads.geometry.iloc[0])])
                  for side,w in [('before',8),('after',12)]}
        objects,assembly = assemble_change_objects(seeds,roads,roads)
        return seeds,objects,assembly,widths

    def test_raw_holes_and_fragments_never_enter_public_geometry(self):
        seeds,objects,assembly,widths = self.fixtures()
        before = assembly['decisions'].copy(deep=True)
        rendered, audit = render_change_geometry(objects,seeds,assembly,widths)
        self.assertEqual(len(rendered),1)
        self.assertAlmostEqual(rendered.geometry.iloc[0].symmetric_difference(box(0,-6,100,6)).area,0)
        pd.testing.assert_frame_equal(before,assembly['decisions'])
        pd.testing.assert_frame_equal(objects.drop(columns='geometry'),rendered.drop(columns='geometry'))
        self.assertEqual(len(audit['published_geometry_parts']),3)
        self.assertEqual(len(seeds.geometry.iloc[0].interiors),1)

    def test_presence_period_symmetry(self):
        for kind,half in [('added',6),('removed',4)]:
            s,o,a,w = self.fixtures(kind,gap=False)
            result,_ = render_change_geometry(o,s,a,w)
            self.assertAlmostEqual(result.geometry.iloc[0].symmetric_difference(box(0,-half,40,half)).area,0)

    def test_width_ribbons_time_reversal_and_no_end_bars(self):
        s,o,a,w = self.fixtures('widened',gap=False)
        result,_ = render_change_geometry(o,s,a,w)
        expected = box(0,-6,40,6).difference(box(0,-4,40,4))
        self.assertAlmostEqual(result.geometry.iloc[0].symmetric_difference(expected).area,0)
        s['change_typ']='narrowed'; s['width_bef']=12; s['width_aft']=8
        result2,_ = render_change_geometry(o,s,a,w)
        self.assertAlmostEqual(result2.geometry.iloc[0].symmetric_difference(expected).area,0)

    def test_variable_width_is_centered_and_continuous(self):
        axis = LineString([(0,0),(100,0)])
        polygon = corridor(axis,np.array([0,50,100]),np.array([8,12,16]))
        self.assertTrue(polygon.is_valid)
        self.assertEqual(len(polygon.interiors),0)
        for station,width in [(10,8.8),(50,12),(90,15.2)]:
            section = polygon.intersection(LineString([(station,-30),(station,30)]))
            self.assertAlmostEqual(section.length,width)
            self.assertAlmostEqual(section.centroid.y,0)

    def test_same_track_width_does_not_take_nearby_parallel_or_crossing(self):
        widths=frame([dict(width_m=8,geometry=LineString([(0,0),(100,0)])),
                      dict(width_m=30,geometry=LineString([(0,1),(100,1)])),
                      dict(width_m=40,geometry=LineString([(50,-20),(50,20)]))])
        _,values,coverage=FinalWidths(widths).profile(LineString([(0,0),(100,0)]),6)
        np.testing.assert_allclose(values,8); self.assertEqual(coverage,1)

    def test_missing_axis_does_not_publish_mask_fallback(self):
        s,o,a,w = self.fixtures(gap=False)
        a['object_axes'] = a['object_axes'].iloc[:0]
        with self.assertRaisesRegex(ValueError,'saved final/canonical axis'):
            render_change_geometry(o,s,a,w)

    def test_smooth_paired_profile_preserves_ribbon_sign(self):
        s,o,a,w = self.fixtures('widened',gap=False)
        samples=[dict(candidate_id=0,station_m=p,valid=True,before_width=8+p/20,after_width=12+p/20)
                 for p in range(0,41,4)]
        result,audit=render_change_geometry(o,s,a,w,samples)
        profile=audit['published_width_profiles']
        self.assertGreater(profile.width_aft.max()-profile.width_aft.min(),1)
        np.testing.assert_allclose(profile.width_aft-profile.width_bef,4)
        self.assertTrue(result.is_valid.all())
        self.assertFalse(result.geometry.iloc[0].contains(LineString([(1,0),(39,0)])))

    def test_empty_accepted_set(self):
        s,o,a,w=self.fixtures(gap=False)
        result,_=render_change_geometry(o.iloc[:0],s.iloc[:0],a,w)
        self.assertTrue(result.empty)

    def test_coincident_seed_endpoints_need_no_junction_square(self):
        s,_,_,w=self.fixtures()
        axis=LineString([(40,0),(80,0)])
        s.at[1,'axis_wkt']=axis.wkt;s.at[1,'geometry']=axis.buffer(2)
        roads=frame([dict(geometry=LineString([(0,0),(80,0)]))])
        o,a=assemble_change_objects(s,roads,roads)
        result,_=render_change_geometry(o,s,a,w)
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(result.geometry.iloc[0].symmetric_difference(box(0,-6,80,6)).area,0)

    def test_overlay_debris_removed_but_real_loop_hole_preserved(self):
        from shapely import union_all
        road=box(0,0,100,100).difference(box(10,10,90,90)).difference(box(2,2,2.01,2.01))
        dirty=union_all([road,box(101,0,101.01,.01)])
        cleaned=_clean_overlay(dirty)
        self.assertEqual(cleaned.geom_type,'Polygon')
        self.assertEqual(len(cleaned.interiors),1)
        self.assertAlmostEqual(cleaned.area,3600)

    def test_finalizer_keeps_qualification_geometry_but_publishes_corridor(self):
        from engine.fast_auto_change import finalize_auto_candidates
        s,_,_,w=self.fixtures(gap=False)
        s=s.assign(confidence=.9,audit_reason='cached_evidence',precision_reason='corroborated_source_and_sustained_absence',
                   publication_state='accepted')
        roads=frame([dict(geometry=LineString([(0,0),(40,0)]))])
        scenes={side:SimpleNamespace(crs=s.crs,widths=widths) for side,widths in w.items()}
        with tempfile.TemporaryDirectory() as temporary:
            with patch('engine.auto_presence_candidates.qualify_presence_candidates',return_value=(s,s)), \
                 patch('engine.auto_width_precision.qualify_width_candidates',return_value=(s,[])):
                finalize_auto_candidates(s.to_dict('records'),[],[],{},presence_audit=[],scenes=scenes,
                    centerlines=[roads,roads],output_dir=temporary,before_period='before',after_period='after')
            saved=gpd.read_file(Path(temporary)/'auto_diagnostics.gpkg',layer='local_seeds')
            public=gpd.read_file(Path(temporary)/'road_changes.shp')
            self.assertAlmostEqual(saved.geometry.iloc[0].symmetric_difference(s.geometry.iloc[0]).area,0)
            self.assertAlmostEqual(public.geometry.iloc[0].symmetric_difference(box(0,-6,40,6)).area,0)


if __name__ == '__main__':
    unittest.main()
