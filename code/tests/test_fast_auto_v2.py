"""Interval-only V2: no station engine, profile geometry and event validation."""
import unittest
from unittest.mock import patch
from shapely import from_wkt
from shapely.geometry import box

from test_fast_auto_change import FastFinalAutoTests
from engine.fast_auto_v2 import analyze_scenes, ChangeEvidence, V2Config
from engine import fast_auto_change as legacy


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.fixture=FastFinalAutoTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        for owner,name in ((legacy,'analyze_scenes'),(legacy.RoadScene,'match')):
            guard=patch.object(owner,name,side_effect=AssertionError('v2 entered station algorithm'))
            guard.start();self.addCleanup(guard.stop)

    def test_stable_segments_do_not_run_match(self):
        f=self.fixture
        before=f.scene([f.road(100)])
        after=f.scene([f.road(100)])
        with patch.object(legacy.RoadScene,'match',side_effect=AssertionError('stable road matched per station')):
            rows,audit,width,counts=analyze_scenes(before,after)
        self.assertFalse(rows)
        self.assertEqual(counts['v2_station_count'],0)
        self.assertGreater(counts['v2_stable_segments'],0)

    def test_presence_symmetry_and_flat_regular_geometry(self):
        f=self.fixture
        b=f.scene([f.road(60),f.road(120)])
        a=f.scene([f.road(120),f.road(180)])
        forward=analyze_scenes(b,a)
        backward=analyze_scenes(a,b)
        self.assertCountEqual([r['change_typ'] for r in forward[0]],['added','removed'])
        self.assertCountEqual([r['change_typ'] for r in backward[0]],['added','removed'])
        for row in forward[0]+backward[0]:
            axis=from_wkt(row['axis_wkt'])
            width=max(row['width_bef'],row['width_aft'])
            self.assertTrue(row['geometry'].equals(axis.buffer(width/2,cap_style='flat')))
            self.assertTrue(row['geometry'].is_valid)
            self.assertEqual(len(row['geometry'].interiors),0)
        self.assertEqual(forward[3]['v2_local_sample_count'],6)
        self.assertEqual(forward[3]['v2_station_count'],0)

    def test_width_profile_is_primary_without_surface_station_rescan(self):
        f=self.fixture
        b=f.scene([f.road(100)])
        a=f.scene([f.road(100)],surfaces=[box(30,96,150,104),box(150,93,260,107)])
        records,_,_,counts=analyze_scenes(b,a)
        self.assertEqual(counts['v2_exact_width_event_sections'],0)
        self.assertFalse(records)

    def test_width_changes_use_existing_profiles(self):
        f=self.fixture
        b=f.scene([f.road(70,8),f.road(160,14)])
        a=f.scene([f.road(70,14),f.road(160,8)])
        with patch.object(legacy,'_measure_period_width',side_effect=AssertionError('unnecessary remeasurement')):
            records,_,_,counts=analyze_scenes(b,a)
        self.assertEqual({r['change_typ'] for r in records},{'widened','narrowed'})
        self.assertGreater(counts['v2_width_profile_intervals'],0)

    def test_missing_centerline_with_existing_surface_not_presence_change(self):
        f=self.fixture
        b=f.scene([f.road(70),f.road(160)])
        surfaces=[line.buffer(width/2) for line,width in [f.road(70),f.road(160)]]
        a=f.scene([f.road(160)],surfaces=surfaces)
        records,_,_,_=analyze_scenes(b,a)
        self.assertFalse(any(r['change_typ']=='removed' for r in records))

    def test_qualification_does_not_enter_station_review(self):
        import geopandas as gpd
        from engine.fast_auto_v2 import qualify_candidates
        f=self.fixture
        b=f.scene([f.road(70)])
        a=f.scene([f.road(70),f.road(160)])
        records,audit,_,_=analyze_scenes(b,a)
        candidates=gpd.GeoDataFrame(records,geometry='geometry',crs=f.crs)
        accepted,_=qualify_candidates(candidates,minimum_length=24.,minimum_area=4.)
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted.iloc[0].publication_state,'accepted')
        self.assertGreater(accepted.iloc[0].length_m,200)

    def test_profile_conflict_does_not_remeasure(self):
        f=self.fixture
        b=f.scene([f.road(100,8)],surfaces=[box(30,93,260,107)])
        a=f.scene([f.road(100,14)],surfaces=[box(30,96,260,104)])
        with patch.object(legacy,'_measure_period_width',side_effect=AssertionError('remeasurement')):
            records,_,_,counts=analyze_scenes(b,a)
        self.assertEqual(counts['v2_exact_width_event_sections'],0)
        self.assertEqual(counts['v2_station_count'],0)
        self.assertTrue(all(r['geometry'].is_valid for r in records))

    def test_finalization_never_reenters_station_qualification(self):
        import geopandas as gpd
        from pathlib import Path
        f=self.fixture
        b=f.scene([f.road(70)]);a=f.scene([f.road(70),f.road(160)])
        presence=[];result=analyze_scenes(b,a,presence_audit=presence)
        centers=[gpd.GeoDataFrame(geometry=s.lines,crs=s.crs) for s in (b,a)]
        output=Path(f.tmp.name)/'v2_final'
        with patch('engine.auto_presence_candidates.qualify_presence_candidates',side_effect=AssertionError('legacy qualification')),\
             patch('engine.auto_width_precision.qualify_width_candidates',side_effect=AssertionError('station width review')):
            legacy.finalize_auto_candidates(*result,presence_audit=presence,scenes=dict(before=b,after=a),
                centerlines=centers,output_dir=output,before_period='1',after_period='2',internal_outputs=True)
        published=gpd.read_file(output/'road_changes.shp')
        self.assertTrue(published.is_valid.all())
        self.assertTrue((output/'network_assembly.gpkg').is_file())

    def published(self, records):
        import geopandas as gpd
        from engine.fast_auto_v2 import qualify_candidates
        if not records:return []
        frame,_=qualify_candidates(gpd.GeoDataFrame(records,geometry='geometry',crs=self.fixture.crs),minimum_length=24.,minimum_area=4.)
        return frame.loc[frame.publication_state=='accepted'].to_dict('records')

    def test_small_width_fluctuation_retained_internally(self):
        f=self.fixture
        rows,*_=analyze_scenes(f.scene([f.road(100,8)]),f.scene([f.road(100,10.5)]))
        self.assertTrue(rows)
        self.assertFalse(self.published(rows))

    def test_width_difference_must_exceed_own_profile_variation(self):
        f=self.fixture
        before=f.scene([f.road(100,4,start=30,end=145),f.road(100,14,start=145,end=260)])
        after=f.scene([f.road(100,14,start=30,end=145),f.road(100,24,start=145,end=260)])
        with patch.object(legacy,'_measure_period_width',side_effect=AssertionError('remeasurement')):
            rows,_,_,counts=analyze_scenes(before,after)
        self.assertTrue(any(r['change_typ']=='widened' for r in rows))
        self.assertFalse(self.published(rows))
        self.assertGreater(counts['v2_width_variability_suppressed'],0)

    def test_short_strong_width_change_not_published(self):
        f=self.fixture
        rows,*_=analyze_scenes(f.scene([f.road(100,8,start=50,end=85)]),f.scene([f.road(100,14,start=50,end=85)]))
        self.assertTrue(rows)
        self.assertFalse(self.published(rows))

    def test_confirmed_presence_publishes_symmetrically(self):
        f=self.fixture
        before=f.scene([f.road(60),f.road(120)])
        after=f.scene([f.road(120),f.road(180)])
        for b,a in ((before,after),(after,before)):
            rows,*_=analyze_scenes(b,a)
            self.assertCountEqual([r['change_typ'] for r in self.published(rows)],['added','removed'])

    def test_uncertain_opposite_evidence_never_published(self):
        f=self.fixture
        before=f.scene([f.road(60)])
        after=f.scene([f.road(60),f.road(180)])
        original=before.evidence
        def uncertain(*args,**kwargs):
            result=original(*args,**kwargs)
            result.update(state='uncertain',reason='conflicting_or_weak_evidence',surface=.3)
            return result
        with patch.object(before,'evidence',side_effect=uncertain):
            rows,*_=analyze_scenes(before,after)
        self.assertTrue(rows)
        self.assertFalse(self.published(rows))

    def test_clear_surface_loss_not_vetoed_by_weak_probability_rank(self):
        from engine.fast_auto_v2 import presence_publication
        for kind,source,target in [('added','after','before'),('removed','before','after')]:
            record=dict(change_typ=kind,length_m=100.,qa_state='probable',
                **{source+'_valid_ratio':1.,target+'_valid_ratio':1.,source+'_surface_ratio':.95,
                   target+'_surface_ratio':.05,target+'_probability_ratio':0.})
            self.assertTrue(presence_publication(record,V2Config())[0])
            record[target+'_probability_ratio']=1.
            self.assertFalse(presence_publication(record,V2Config())[0])

    def test_stable_position_margin_does_not_create_station_scan(self):
        f=self.fixture
        before=f.scene([f.road(100)],surfaces=[])
        after=f.scene([f.road(104)],surfaces=[])
        rows,_,_,counts=analyze_scenes(before,after)
        self.assertFalse(rows)
        self.assertEqual(counts['v2_station_count'],0)

    def test_auto_has_no_truth_file_dependency(self):
        import inspect
        f=self.fixture
        before=f.scene([f.road(100)])
        after=f.scene([f.road(100),f.road(180)])
        self.assertFalse(any('truth' in key or key=='gt' for key in inspect.signature(analyze_scenes).parameters))
        with patch('geopandas.read_file',side_effect=AssertionError('Auto attempted external vector read')):
            records,*_=analyze_scenes(before,after)
            self.assertEqual([r['change_typ'] for r in self.published(records)],['added'])


if __name__=='__main__':unittest.main()
