import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.fast_gt_reconciliation import (GTProfile,track_intervals,perturb_truth,correct_changes,
    reconcile_periods,augment_fast_changes_with_truth,_frame,_change_polygon)
from engine.auto_change_geometry import FinalWidths
from engine.auto_change_assembly import assemble_change_objects,write_assembly_audit


def line(y,start=0,end=100):return LineString([(start,y),(end,y)])
def frame(rows):return _frame(rows,3857)


class FastGTReconciliationTests(unittest.TestCase):
    def test_bhbm_three_area_deficit_is_completed_after_existing_perturbation(self):
        truth=frame([dict(BHBM=3,DLKD=14,geometry=box(0,-7,100,7))])
        truth['_gt_type']=truth.BHBM
        widths={side:FinalWidths(frame([dict(width_m=value,geometry=line(0))]))
                for side,value in (('before',8),('after',14))}
        gt,_=perturb_truth(truth,widths,'T1','T2',GTProfile(omission_probability=0,type_error_probability=0))
        self.assertTrue(gt.gt_type.eq('width_changed').all())
        auto=frame([dict(change_id='auto',change_src='AUTO',change_typ='widened',width_bef=8.,width_aft=9.,
            axis_wkt=line(0).wkt,geometry=_change_polygon(line(0),'widened',8,9))])
        result,audit=correct_changes(auto,gt)
        target=gt.geometry.union_all()
        before=target.intersection(auto.geometry.union_all()).area/target.area
        after=target.intersection(result.geometry.union_all()).area/target.area
        self.assertGreater(after,.9);self.assertGreater(after,before+.3)
        self.assertTrue(any(row['action']=='complete_same_type_area' for row in audit))

    def test_same_type_longitudinal_match_repairs_area_for_all_four_types(self):
        for kind,b,a,gb,ga in [('added',0,4,0,10),('removed',4,0,10,0),
                               ('widened',8,9,8,14),('narrowed',14,13,14,8)]:
            with self.subTest(kind=kind):
                axis=line(0);local=line(0,20,80)
                auto=frame([dict(change_id='auto',change_src='AUTO',change_typ=kind,width_bef=b,width_aft=a,
                    axis_wkt=axis.wkt,geometry=_change_polygon(axis,kind,b,a))])
                gt=frame([dict(change_id='gt',truth_id='gt',change_src='GT_ASSISTED',change_typ=kind,
                    gt_type='width_changed' if kind in ('widened','narrowed') else kind,
                    width_bef=gb,width_aft=ga,axis_wkt=local.wkt,match_axis_wkt=local.wkt,
                    geometry=_change_polygon(local,kind,gb,ga))])
                original=auto.geometry.iloc[0].wkb
                result,audit=correct_changes(auto,gt)
                self.assertLess(gt.geometry.iloc[0].difference(result.geometry.union_all()).area,1e-6)
                self.assertTrue(any(r['action']=='complete_same_type_area' for r in audit))
                retained=result[result.change_src.eq('AUTO')]
                self.assertAlmostEqual(retained.length_m.sum(),40.)
                self.assertTrue(retained.width_bef.eq(b).all());self.assertTrue(retained.width_aft.eq(a).all())
                self.assertEqual(auto.geometry.iloc[0].wkb,original)

    def test_sufficient_same_type_surface_remains_auto(self):
        axis=line(0)
        row=dict(change_id='auto',change_src='AUTO',change_typ='widened',width_bef=8.,width_aft=14.,
                 axis_wkt=axis.wkt,geometry=_change_polygon(axis,'widened',8,14))
        gt={**row,'change_id':'gt','truth_id':'gt','gt_type':'width_changed','match_axis_wkt':axis.wkt,'change_src':'GT_ASSISTED'}
        result,audit=correct_changes(frame([row]),frame([gt]))
        self.assertEqual(len(result),1);self.assertEqual(result.iloc[0].change_src,'AUTO')
        self.assertEqual(audit[0]['action'],'retain_correct_auto')

    def period(self,root,name,roads):
        directory=root/name;directory.mkdir()
        f=frame([dict(width_m=w,geometry=g) for g,w in roads]);result=dict(period=name,grid='region')
        for key in ('centerlines','width_segments'):
            path=directory/f'{key}.shp';f.to_file(path);result[key]=str(path)
        for key in ('surfaces','corridors'):
            path=directory/f'{key}.shp'
            frame([dict(geometry=g.buffer(w/2,cap_style='flat')) for g,w in roads]).to_file(path)
            result[key]=str(path)
        return result

    def test_track_edits_exclude_crossing_and_nearby_carriageway(self):
        axis=line(0,20,80)
        hits=track_intervals(axis,[line(0),line(2),LineString([(50,-30),(50,30)])])
        self.assertEqual({h['target'] for h in hits},{0})
        self.assertAlmostEqual(hits[0]['start'],20)
        self.assertAlmostEqual(hits[0]['end'],80)
        ambiguous=track_intervals(line(1),[line(0),line(2)])
        self.assertEqual({h['target'] for h in ambiguous},{0})

    def test_reconcile_added_removed_width_and_temporal_consistency(self):
        from temporal_road_analysis import build_temporal_grid
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw)
            periods=[self.period(root,'T1',[(line(0),6),(line(2),4),(line(60),6),(line(90),10)]),
                     self.period(root,'T2',[(line(2),4),(line(30),6),(line(60),6),(line(90),10)])]
            records=[]
            for i,(kind,y,b,a) in enumerate([('added',0,0,8),('removed',30,7,0),('widened',60,6,10),('narrowed',90,8,5)]):
                axis=line(y,20,80)
                records.append(dict(change_id=str(i),change_typ=kind,width_bef=b,width_aft=a,axis_wkt=axis.wkt,
                    match_axis_wkt=axis.wkt,geometry=_change_polygon(axis,kind,b,a)))
            pairs=[dict(frame=frame(records),before_period='T1',after_period='T2')]
            updated=reconcile_periods(periods,pairs,root/'reconciled')
            before,after=[gpd.read_file(p['centerlines']) for p in updated]
            self.assertAlmostEqual(before.loc[before.track_id.isna()].geometry.intersection(line(2)).length.sum(),100)
            for product in updated:
                surface=gpd.read_file(product['surfaces']);corridor=gpd.read_file(product['corridors'])
                self.assertTrue(surface.geometry.equals(corridor.geometry));self.assertTrue(surface.is_valid.all())
            changes=root/'changes';changes.mkdir();pairs[0]['frame'].drop(columns=['axis_wkt','match_axis_wkt']).to_file(changes/'road_changes.shp')
            result=build_temporal_grid('region',updated,[dict(output=str(changes),before_period='T1',after_period='T2')],root/'temporal')
            events=gpd.read_file(result['events_shp']);tracked=events.loc[events.road_id.str.startswith('RC')]
            self.assertEqual(set(tracked.event_typ),{'added','removed','widened','narrowed'})
            wide=tracked.loc[tracked.event_typ=='widened'].iloc[0]
            self.assertEqual((wide.before_st,wide.after_st),('present','present'))
            self.assertAlmostEqual(wide.after_w-wide.before_w,4)
            narrow=tracked.loc[tracked.event_typ=='narrowed'].iloc[0]
            self.assertEqual((narrow.before_st,narrow.after_st),('present','present'))
            self.assertAlmostEqual(narrow.after_w-narrow.before_w,-3)

    def test_regular_perturbation_reproducible_and_no_mask_holes(self):
        truth=frame([dict(_gt_type='2',DLKD=8,geometry=box(0,-4,100,4).difference(box(45,-1,46,1)))])
        widths={s:FinalWidths(frame([dict(width_m=8,geometry=line(0))])) for s in ('before','after')}
        p=GTProfile(omission_probability=0,type_error_probability=0)
        a,qa=perturb_truth(truth,widths,'T1','T2',p);b,qb=perturb_truth(truth,widths,'T1','T2',p)
        self.assertTrue(a.equals(b));self.assertEqual(qa,qb)
        self.assertTrue(a.is_valid.all());self.assertTrue(all(len(g.interiors)==0 for g in a.geometry))

    def test_conflicting_type_is_replaced_by_axis_interval_not_polygon_clip(self):
        def record(kind,y,start,end):
            axis=line(y,start,end);b,a=(8,0) if kind=='removed' else (0,8)
            return dict(change_id=kind,change_typ=kind,width_bef=b,width_aft=a,axis_wkt=axis.wkt,match_axis_wkt=axis.wkt,
                        truth_id='GT',gt_type=kind,geometry=_change_polygon(axis,kind,b,a))
        auto=frame([record('removed',0,0,100),record('added',3,0,100)])
        gt=frame([record('added',0,20,80)])
        result,audit=correct_changes(auto,gt)
        self.assertTrue(any(r['action']=='correct_conflicting_type' for r in audit))
        self.assertEqual(len(result.loc[result.change_typ=='removed']),2)
        self.assertTrue(any(g.equals(auto.geometry.iloc[1]) for g in result.geometry))

    def test_shared_period_conflict_is_automatically_resolved_and_audited(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);periods=[self.period(root,p,[(line(0),8)]) for p in ('1','2','3')]
            axis=line(0);row=dict(change_id='x',change_typ='added',width_bef=0,width_aft=8,axis_wkt=axis.wkt,
                match_axis_wkt=axis.wkt,geometry=axis.buffer(4))
            pairs=[dict(frame=frame([row]),before_period='1',after_period='2'),
                   dict(frame=frame([row]),before_period='2',after_period='3')]
            outputs=reconcile_periods(periods,pairs,root/'out')
            state=gpd.read_file(outputs[1]['road_state'])
            self.assertTrue(state.status.eq('present').all())
            self.assertTrue(pairs[1]['frame'].empty)
            self.assertIn('adjacent_state_conflict_resolved',(root/'out/change_track_assignments.csv').read_text())

    def test_end_to_end_keeps_auto_cache_and_builds_one_final_temporal(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);periods=[self.period(root,'T1',[(line(30),6)]),self.period(root,'T2',[(line(0),8),(line(30),6)])]
            seed=frame([dict(change_typ='added',width_bef=0.,width_aft=8.,length_m=40.,axis_wkt=line(0,30,70).wkt,geometry=line(0,30,70).buffer(4))])
            roads=[gpd.read_file(p['centerlines']) for p in periods]
            auto=root/'auto';objects,assembly=assemble_change_objects(seed,*roads)
            write_assembly_audit(auto,seed,assembly);objects.to_file(auto/'road_changes.shp')
            automatic=dict(output=str(auto),road_changes=str(auto/'road_changes.shp'))
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in auto.iterdir() if p.is_file()}
            truth=root/'truth.shp';frame([dict(BHBM='2',DLKD=8,geometry=box(0,-4,100,4))]).to_file(truth)
            result=augment_fast_changes_with_truth(automatic,truth,root/'assisted',before_result=periods[0],after_result=periods[1],
                before_period='T1',after_period='T2',profile=GTProfile(omission_probability=0,type_error_probability=0))
            self.assertTrue(result['ground_truth_used']);self.assertTrue(Path(result['final_temporal']['life_shp']).exists())
            self.assertEqual(len(list((root/'assisted').rglob('road_life.shp'))),1)
            self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in auto.iterdir() if p.is_file()})
            from engine.fast_gt_reconciliation import complete_auto_pair_temporal
            standalone=complete_auto_pair_temporal(automatic,periods[0],periods[1],'T1','T2')
            self.assertTrue(Path(standalone['final_temporal']['width_evolution']).is_file())
            self.assertFalse(standalone.get('ground_truth_used',False))
            self.assertEqual(hashes['road_changes.shp'],hashlib.sha256((auto/'road_changes.shp').read_bytes()).hexdigest())

    def test_partial_later_removal_shares_atomic_track_identity(self):
        from temporal_road_analysis import build_temporal_grid
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw)
            periods=[self.period(root,'1',[(line(30),6)]),self.period(root,'2',[(line(0),8),(line(30),6)]),
                     self.period(root,'3',[(line(0,0,40),8),(line(30),6)])]
            pairs=[]
            for kind,axis,b,a,first,last in [('added',line(0),0,8,'1','2'),('removed',line(0,40,100),8,0,'2','3')]:
                row=dict(change_id=kind,change_typ=kind,width_bef=b,width_aft=a,axis_wkt=axis.wkt,match_axis_wkt=axis.wkt,
                         geometry=_change_polygon(axis,kind,b,a))
                pairs.append(dict(frame=frame([row]),before_period=first,after_period=last))
            updated=reconcile_periods(periods,pairs,root/'out')
            self.assertEqual(len(pairs[0]['frame']),2)
            self.assertTrue(set(pairs[1]['frame'].track_id)<=set(pairs[0]['frame'].track_id))
            entries=[]
            for i,pair in enumerate(pairs):
                directory=root/f'change{i}';directory.mkdir()
                pair['frame'].drop(columns=['axis_wkt','match_axis_wkt']).to_file(directory/'road_changes.shp')
                entries.append(dict(output=str(directory),before_period=pair['before_period'],after_period=pair['after_period']))
            result=build_temporal_grid('region',updated,entries,root/'temporal')
            events=gpd.read_file(result['events_shp']);events=events.loc[events.road_id.str.startswith('RC')]
            self.assertEqual(events.event_typ.value_counts().to_dict(),{'added':2,'removed':1})

    def test_shared_period_width_is_reconciled_back_into_both_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);periods=[self.period(root,p,[(line(0),w)]) for p,w in [('1',6),('2',10),('3',14)]]
            pairs=[]
            for i,b,a in [(1,6,10),(2,10.6,14)]:
                axis=line(0);row=dict(change_id=str(i),change_typ='widened',width_bef=b,width_aft=a,
                    axis_wkt=axis.wkt,match_axis_wkt=axis.wkt,geometry=_change_polygon(axis,'widened',b,a))
                pairs.append(dict(frame=frame([row]),before_period=str(i),after_period=str(i+1)))
            reconcile_periods(periods,pairs,root/'out')
            self.assertAlmostEqual(pairs[0]['frame'].iloc[0].width_aft,10.3)
            self.assertAlmostEqual(pairs[1]['frame'].iloc[0].width_bef,10.3)

    def test_missing_track_does_not_delete_adjacent_persistent_carriageway(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);periods=[self.period(root,'1',[(line(2),6)]),self.period(root,'2',[(line(0),8),(line(2),6)])]
            axis=line(0);r=dict(change_id='gt',change_typ='added',width_bef=0,width_aft=8,
                axis_wkt=axis.wkt,match_axis_wkt=axis.wkt,geometry=axis.buffer(4))
            result=reconcile_periods(periods,[dict(frame=frame([r]),before_period='1',after_period='2')],root/'out')
            before=gpd.read_file(result[0]['centerlines'])
            self.assertAlmostEqual(before.geometry.intersection(line(2)).length.sum(),100)
            self.assertFalse(before.track_id.notna().any())

    def test_gt_junction_endpoints_are_not_trimmed_or_randomly_disconnected(self):
        geometry=MultiLineString([[(0,0),(50,0)],[(50,0),(100,0)],[(50,0),(50,50)]])
        truth=frame([dict(_gt_type='2',DLKD=8,geometry=geometry)])
        widths={s:FinalWidths(frame([dict(width_m=8,geometry=line(0))])) for s in ('before','after')}
        result,audit=perturb_truth(truth,widths,'1','2',GTProfile(omission_probability=0,type_error_probability=0))
        from shapely import from_wkt
        endpoints=[tuple(c) for wkt in result.axis_wkt for c in (from_wkt(wkt).coords[0],from_wkt(wkt).coords[-1])]
        self.assertEqual(endpoints.count((50.,0.)),3)

    def test_final_rebuild_preserves_curve_and_connects_supported_junction(self):
        from engine.road_network_products import rebuild_corrected_road_axes
        from shapely import union_all
        curve=LineString([(0,0),(0,30),(10,50),(30,60),(70,60)])
        main=line(0);branch=LineString([(50,7),(50,45)])
        evidence=union_all([main.buffer(5),LineString([(50,0),(50,45)]).buffer(5)])
        result=rebuild_corrected_road_axes([main,branch],[10,10],evidence)
        self.assertLess(result[0].distance(result[1]),1e-6)
        fitted=rebuild_corrected_road_axes([curve])[0]
        self.assertLess(fitted.hausdorff_distance(curve),3.1)
        self.assertGreater(fitted.length,90.)

    def test_flat_road_ends_and_no_raw_surface_noise_in_final_products(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);periods=[self.period(root,p,[(line(0),8)]) for p in ('1','2')]
            # The evidence surface deliberately contains a detached noisy patch.
            for p in periods:
                frame([dict(geometry=box(-2,-6,102,6)),dict(geometry=box(40,20,41,21))]).to_file(p['surfaces'])
            from engine.fast_gt_reconciliation import _write_final_period
            original=gpd.read_file(periods[0]['centerlines']);directory=root/'final';directory.mkdir()
            rows=[{**r,'track_id':'','_original_row':i} for i,r in enumerate(original.to_dict('records'))]
            result=_write_final_period(periods[0],original,{},rows,[],directory,3857,3857)
            geometry=gpd.read_file(result['surfaces']).geometry.union_all()
            self.assertTrue(geometry.equals(box(0,-4,100,4)))
            self.assertEqual(_change_polygon(line(0),'added',0,8).bounds,(0.,-4.,100.,4.))


if __name__=='__main__':unittest.main()
