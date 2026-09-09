"""Production orchestration only: temporary fixtures, no model or real project."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import user_pipeline as p
from app.result_publisher import ResultPublisher, result_index_from_manifest


class FastProductionTests(unittest.TestCase):
    def test_batch_change_failures_continue_and_complete_reports_manifest_status(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); path = root/'pipeline.json'
            p.write_json(path, {'job_root': raw, 'execution_profile': 'fast',
                'period_results': [{'grid':'area', 'period':str(i)} for i in range(3)],
                'change_results': []})
            with patch.object(p, '_rerun_change_entry', side_effect=[RuntimeError('pair failed'), {}]) as run, \
                 patch.object(p, '_refresh_manifest_downstream'), \
                 patch.object(p, '_persist_existing_pipeline'), \
                 patch.object(p, '_period_result_ready', return_value=True), \
                 patch.object(p, 'emit') as emit:
                result = p.rerun_all_pipeline_changes(argparse.Namespace(
                    pipeline_manifest=str(path), continue_on_error=True))
            self.assertEqual(run.call_count, 2)
            self.assertEqual(result['failure_count'], 1)
            self.assertEqual(result['change_count'], 1)
            emit.assert_any_call('complete', stage='rerun-all-changes',
                                 status='completed_with_errors', **result)

    def test_completed_gt_final_exports_existing_pixel_centerline_metrics(self):
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);gt=root/'truth.shp';final=root/'final.shp';raster=root/'probability.tif'
            gpd.GeoDataFrame({'BHBM':[2]},geometry=[box(5,20,50,28)],crs=32650).to_file(gt)
            gpd.GeoDataFrame({'change_typ':['added']},geometry=[box(5,20,50,28)],crs=32650).to_file(final)
            with rasterio.open(raster,'w',driver='GTiff',width=64,height=64,count=1,dtype='uint8',
                               crs=32650,transform=from_origin(0,64,1,1)) as dst:
                dst.write(np.ones((1,64,64),dtype=np.uint8))
            m=dict(execution_profile='fast',job_root=str(root),period_results=[
                dict(grid='g',period='1',road_probability=str(raster))],change_results=[dict(grid='g',before_period='1',
                after_period='2',output=str(root),road_changes=str(final),product_variant='final',execution_profile='fast',
                ground_truth_used=True,truth_path=str(gt),fast_finalization_state='completed')])
            args=argparse.Namespace(pipeline_manifest=str(root/'manifest.json'),_manifest=m,grid='g',before_period='1',
                after_period='2',truth=str(gt),validation_area='',truth_type_field='BHBM',evaluation_tolerance=0.,
                defer_publish=True,defer_aggregate=True)
            result=p._evaluate_existing_changes_impl(args)
            metrics=pd.read_csv(result['metrics'])
            for name in ('centerline_mean_offset_px','road_centerline_completeness'):
                self.assertIsNotNone(result[name]);self.assertTrue(np.isfinite(result[name]))
                self.assertTrue(np.isfinite(metrics.iloc[0][name]))
            summary=p.read_json(Path(result['summary']))
            self.assertTrue(summary['evaluation']['metadata']['fast_assisted_centerline_metrics'])
            self.assertNotIn('auto_evaluation',summary)

    def test_existing_gui_rerun_commands_still_parse(self):
        from app.task_manager import TaskManager
        commands=[TaskManager.build_rerun_period('manifest.json','g','2'),
            TaskManager.build_rerun_period('manifest.json','g','2',True),
            TaskManager.build_rerun_change('manifest.json','g','1','2'),
            TaskManager.build_rerun_change('manifest.json','g','1','2',True),
            TaskManager.build_rerun_all_periods('manifest.json'),
            TaskManager.build_rerun_all_changes('manifest.json')]
        for command in commands:
            parsed=p.parser().parse_args(command)
            self.assertEqual(parsed.pipeline_manifest,'manifest.json')

    def test_evaluator_reads_final_layer_only_even_with_legacy_source_fields(self):
        import geopandas as gpd
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);gt=root/'truth.shp';final=root/'final.shp'
            gpd.GeoDataFrame({'BHBM':[2]},geometry=[box(0,0,30,8)],crs=32650).to_file(gt)
            gpd.GeoDataFrame({'change_typ':['added']},geometry=[box(0,0,30,8)],crs=32650).to_file(final)
            m=dict(execution_profile='fast',job_root=str(root),change_results=[dict(grid='g',before_period='1',
                after_period='2',output=str(root),road_changes=str(final),product_variant='gt_assisted',
                automatic_road_changes=str(root/'missing_auto.shp'),fast_finalization_state='completed')])
            args=argparse.Namespace(pipeline_manifest=str(root/'manifest.json'),_manifest=m,grid='g',before_period='1',
                after_period='2',truth=str(gt),validation_area='',truth_type_field='BHBM',evaluation_tolerance=0.,
                defer_publish=True,defer_aggregate=True)
            with patch.object(p,'_persist_existing_pipeline') as persist:
                result=p._evaluate_existing_changes_impl(args)
            persist.assert_not_called()
            self.assertAlmostEqual(result['precision'],1.)
            self.assertAlmostEqual(result['recall'],1.)
            self.assertTrue(Path(m['change_results'][0]['evaluation_metrics']).is_file())

    def test_change_rerun_uses_auto_periods_and_drops_previous_gt_correction(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);m=self.manifest(root)
            m['auto_period_results']=[dict(grid='g',period=x,result=str(root/f'auto{x}.json')) for x in ('1','2','3')]
            m['change_results'][0].update(correction_audit='old.gpkg',final_temporal={'old':True})
            m['input_spec']={'truths':{'g\0'+'1\0'+'2':''}}
            with patch.object(p,'_run_fast_change_result',return_value={'output':str(root/'new')}) as run:
                result=p._rerun_change_entry(m,'g','1','2')
            self.assertEqual(run.call_args.args[:2],(root/'auto1.json',root/'auto2.json'))
            self.assertTrue(run.call_args.kwargs['defer_finalization'])
            self.assertIsNone(run.call_args.kwargs['truth_path'])
            self.assertNotIn('correction_audit',result);self.assertNotIn('final_temporal',result)

    def test_both_fast_batch_commands_finalize_once_after_all_pairs(self):
        for periods in (True,False):
            with self.subTest(periods=periods),tempfile.TemporaryDirectory() as raw:
                root=Path(raw);m=self.manifest(root);path=root/'manifest.json';p.write_json(path,m);order=[]
                with patch.object(p,'_rerun_period_entry',side_effect=lambda *a:order.append('period') or {}), \
                     patch.object(p,'_rerun_change_entry',side_effect=lambda *a:order.append('pair') or {}), \
                     patch.object(p,'_refresh_manifest_downstream',side_effect=lambda m:order.append('finalize')), \
                     patch.object(p,'_persist_existing_pipeline'):
                    fn=p.rerun_all_pipeline_periods if periods else p.rerun_all_pipeline_changes
                    fn(argparse.Namespace(pipeline_manifest=str(path),continue_on_error=False))
                self.assertEqual(order,(['period']*3 if periods else [])+['pair','pair','finalize'])

    def manifest(self, root):
        return dict(execution_profile='fast',job_root=str(root),period_results=[
            dict(grid='g',period=x) for x in ('1','2','3')],change_results=[
            dict(grid='g',before_period=b,after_period=a,output=str(root/f'{b}_{a}'),
                 truth='gt.shp' if b=='1' else '',evaluation_metrics='obsolete.csv')
            for b,a in (('1','2'),('2','3'))],final_period_results=['old'],temporal_results=['old'],
            evaluation_summary={'json':'old.json'},period_orders={'g':{'period_order':['1','2','3']}})

    def test_finalization_orders_geometry_then_evaluation_and_clears_stale_state(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);m=self.manifest(root);order=[]
            def build(manifest,job):
                self.assertNotIn('final_period_results',manifest)
                self.assertNotIn('temporal_results',manifest)
                for entry in manifest['change_results']:
                    self.assertNotIn('evaluation_metrics',entry)
                    entry['road_changes']=str(root/'final.shp')
                manifest['final_period_results']=manifest['period_results']
                order.append('final_roads_changes_temporal');return [{'grid':'g'}]
            def evaluate(args):
                self.assertTrue(args.defer_publish)
                self.assertTrue(args.defer_aggregate)
                self.assertEqual(args._manifest['change_results'][0]['road_changes'],str(root/'final.shp'))
                self.assertEqual(args._manifest['temporal_results'],[{'grid':'g'}])
                order.append('evaluate_final')
            with patch.object(p,'build_temporal_outputs',side_effect=build), \
                 patch.object(p,'_evaluate_existing_changes_impl',side_effect=evaluate), \
                 patch.object(p,'aggregate_change_evaluations',side_effect=lambda *_:order.append('aggregate')):
                p._finalize_fast_manifest(m,root)
            self.assertEqual(order,['final_roads_changes_temporal','evaluate_final','aggregate'])
            self.assertEqual(m['fast_finalization_state'],'completed')
            self.assertEqual(m['change_results'][1]['evaluation_state'],'skipped_no_truth')
            self.assertNotIn('evaluation_metrics',m['change_results'][1])

    def test_pending_results_never_publish_or_enter_gui_fallback(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);m=self.manifest(root);p._invalidate_fast_finalization(m)
            publisher=ResultPublisher(root/'results')
            with patch.object(publisher,'publish_period') as period,patch.object(publisher,'publish_change') as change:
                publisher.publish_manifest(m)
            period.assert_not_called();change.assert_not_called()
            self.assertEqual(result_index_from_manifest(m,root)['areas'],{})

    def test_fast_local_reruns_force_dependency_refresh_without_cli_flags(self):
        for period_mode in (True,False):
            with self.subTest(period_mode=period_mode),tempfile.TemporaryDirectory() as raw:
                root=Path(raw);m=self.manifest(root);path=root/'pipeline_result.json';p.write_json(path,m)
                order=[]
                with patch.object(p,'_rerun_period_entry',side_effect=lambda *_:order.append('period') or {}), \
                     patch.object(p,'_rerun_change_entry',side_effect=lambda _,g,b,a:order.append(f'{b}-{a}') or {}), \
                     patch.object(p,'_refresh_manifest_downstream',side_effect=lambda _:order.append('finalize')), \
                     patch.object(p,'_persist_existing_pipeline'):
                    if period_mode:
                        result=p.rerun_pipeline_period(argparse.Namespace(pipeline_manifest=str(path),grid='g',period='2',update_related=False))
                        self.assertTrue(result['updated_related'])
                    else:
                        result=p.rerun_pipeline_change(argparse.Namespace(pipeline_manifest=str(path),grid='g',before_period='1',after_period='2',update_temporal=False))
                        self.assertTrue(result['updated_temporal'])
                self.assertEqual(order,['period','1-2','2-3','finalize'] if period_mode else ['1-2','finalize'])

    def test_single_pair_with_and_without_gt_use_same_finalizer(self):
        for truth in (False,True):
            with self.subTest(truth=truth),tempfile.TemporaryDirectory() as raw:
                root=Path(raw);gt=root/'gt.shp';gt.touch();order=[]
                def finalize(m,output):
                    order.append('finalize');m['final_period_results']=m['period_results'];m['temporal_results']=[{}]
                with patch('engine.fast_pipeline.detect_fast_changes',side_effect=lambda *a,**k:order.append('auto') or {'output':str(root)}), \
                     patch('engine.fast_pipeline.augment_fast_changes_with_truth',side_effect=lambda *a,**k:order.append('correct') or {'output':str(root)}) as correct, \
                     patch('engine.fast_pipeline._load_fast_period_result',return_value={}), \
                     patch.object(p,'_finalize_fast_manifest',side_effect=finalize):
                    result=p._run_fast_change_result(root/'1.json',root/'2.json',root,before_period='1',after_period='2',
                        position_tolerance=3,width_change_absolute=2,width_change_ratio=.2,truth_path=gt if truth else None)
                self.assertEqual(order,['auto','correct','finalize'] if truth else ['auto','finalize'])
                self.assertIn('final_period_results',result);self.assertIn('final_temporal',result)
                if truth:self.assertTrue(correct.call_args.kwargs['defer_finalization'])

    def test_run_all_defers_all_pairs_until_one_finalization_and_publish(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);source=root/'input';source.mkdir();order=[]
            args=argparse.Namespace(source_root=str(source),output_root=str(root/'results'),run_id='run',
                checkpoint='model.ckpt',config='config.yaml',device='cpu',pixel_size='0',rescale='off',
                absolute='2',ratio='.2',tolerance='3',execution_profile='fast')
            def pair(*a,**kw):
                self.assertTrue(kw['defer_finalization']);order.append('pair')
                return {'output':str(a[2])}
            def finalize(m,job):
                self.assertEqual(len(m['change_results']),2);order.append('finalize')
                m['final_period_results']=m['period_results'];m['temporal_results']=[]
            with patch.object(p,'discover_grid_periods',return_value={'g':{x:source/f'{x}.txt' for x in ('1','2','3')}}), \
                 patch.object(p,'prepare'),patch.object(p,'extract',return_value={}), \
                 patch.object(p,'_run_fast_change_result',side_effect=pair), \
                 patch.object(p,'_finalize_fast_manifest',side_effect=finalize), \
                 patch.object(ResultPublisher,'publish_period') as periods,patch.object(ResultPublisher,'publish_change') as changes, \
                 patch.object(ResultPublisher,'publish_manifest',side_effect=lambda *a,**kw:order.append('publish')), \
                 patch.object(ResultPublisher,'publish_reports'),patch.object(p,'aggregate_change_evaluations'):
                p.run_all(args)
            periods.assert_not_called();changes.assert_not_called()
            self.assertEqual(order,['pair','pair','finalize','publish'])

    def test_road_state_is_internal_and_fast_evaluation_stays_under_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);state=root/'road_state.gpkg';state.write_text('private')
            publisher=ResultPublisher(root/'results');target=root/'results'/'g'/'01_单期道路'/'1'
            target.mkdir(parents=True);(target/'road_state.gpkg').write_text('obsolete')
            publisher.publish_period('g','1',{'road_state':str(state)})
            self.assertFalse((target/'road_state.gpkg').exists());self.assertTrue(state.exists())
            summary=root/'summary.json';p.write_json(summary,{'metrics':[]})
            publisher.publish_evaluation(['g'],{'json':str(summary)},within_changes=True)
            self.assertTrue((root/'results/g/02_变化检测/精度评价/evaluation_summary.json').exists())
            self.assertFalse((root/'results/g/04_精度评价').exists())


if __name__=='__main__':unittest.main()
