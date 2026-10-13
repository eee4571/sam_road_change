"""Stage selection and cached formal upgrades, no inference/model dependencies."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import user_pipeline as pipeline
from engine.product_cache import signature
from engine.formal_road_products import formal_metadata
import check_formal_period_products as fixtures


class RefreshTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.FormalPeriodTests();f.setUp();self.addCleanup(f.doCleanups)
        self.f=f;f.regional([fixtures.line(0,100)])
        images=f.root/'images';images.mkdir()
        pipeline.write_json(f.root/'input_manifest.json',dict(images=str(images)))
        self.path=f.root/'latest_result.json'
        self.payload=dict(workspace=str(f.root),execution_profile='fast',
            centerlines=str(f.output/'road_centerlines.shp'),width_review=str(f.width),
            width_method='sam_molra',road_probability=str(f.root/'probability.tif'),
            valid_observation=str(f.root/'valid.shp'))
        for key in ('road_probability','valid_observation'):Path(self.payload[key]).write_bytes(b'unchanged source')
        pipeline.write_json(self.path,self.payload)
        entry=dict(self.payload,grid='north',period='2020',result=str(self.path))
        self.manifest=dict(execution_profile='fast',auto_period_results=[entry],period_results=[copy.deepcopy(entry)])
        patches=patch.multiple(pipeline,extract=Mock(side_effect=AssertionError('no inference')),
            prepare=Mock(side_effect=AssertionError('no image preparation')),emit=Mock())
        patches.start();self.addCleanup(patches.stop)
        preview=patch('engine.fast_pipeline._write_fast_period_previews',side_effect=f.preview)
        preview.start();self.addCleanup(preview.stop)

    def test_stale_export_is_upgraded_once_and_sources_are_unchanged(self):
        original=signature([self.payload['road_probability'],self.payload['valid_observation']])
        pipeline._refresh_fast_period_entries(self.manifest,'north',{'2020'})
        self.assertEqual(signature([self.payload['road_probability'],self.payload['valid_observation']]),original)
        for key in ('auto_period_results','period_results'):
            self.assertEqual(self.manifest[key][0]['formal_road_revision'],formal_metadata()['formal_road_revision'])
        with patch('engine.fast_pipeline.export_fast_products',side_effect=AssertionError('duplicate export')):
            pipeline._refresh_fast_period_entries(self.manifest,'north',{'2020'})

    def test_missing_cache_reports_period_instead_of_falling_back_to_inference(self):
        (self.f.output/'regional_products.gpkg').unlink()
        with self.assertRaisesRegex(FileNotFoundError,'north/2020.*重跑该道路期次'):
            pipeline._refresh_fast_period_entries(self.manifest,'north')

    def test_unselected_area_or_period_does_not_touch_missing_inputs(self):
        with patch('engine.fast_pipeline.export_fast_products',side_effect=AssertionError('outside scope')):
            pipeline._refresh_fast_period_entries(self.manifest,'south')
            pipeline._refresh_fast_period_entries(self.manifest,'north',{'2021'})
        self.assertEqual(pipeline.read_json(self.path),self.payload)

    def test_retry_after_export_before_manifest_save_reconciles_metadata(self):
        original=copy.deepcopy(self.manifest)
        pipeline._refresh_fast_period_entries(self.manifest,'north')
        with patch('engine.fast_pipeline.export_fast_products',side_effect=AssertionError('repeat finished export')):
            pipeline._refresh_fast_period_entries(original,'north')
        self.assertEqual(original['auto_period_results'][0]['formal_road_revision'],formal_metadata()['formal_road_revision'])

    def test_change_entry_refreshes_both_inputs_before_detection(self):
        second=copy.deepcopy(self.manifest['auto_period_results'][0]);second['period']='2021'
        self.manifest['auto_period_results'].append(second)
        self.manifest.update(period_orders={'north':{'period_order':['2020','2021']}},
            job_root=str(self.f.root),change_results=[])
        def detect(before,after,output,**kwargs):
            self.assertEqual(pipeline.read_json(before)['formal_road_revision'],formal_metadata()['formal_road_revision'])
            self.assertEqual(pipeline.read_json(after)['formal_road_revision'],formal_metadata()['formal_road_revision'])
            return dict(output=str(output),execution_profile='fast')
        with patch.object(pipeline,'_run_fast_change_result',side_effect=detect) as detector:
            pipeline._rerun_change_entry(self.manifest,'north','2020','2021')
        self.assertEqual(detector.call_count,1)


class StageReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);images=self.root/'images';images.mkdir()
        self.image=images/'tile.tif';self.image.write_bytes(b'image')
        paths=self.root/'images.txt';paths.write_text(str(self.image))
        ckpt=self.root/'model.pt';ckpt.write_bytes(b'model')
        config=self.root/'config.yaml';config.write_text('model: a')
        self.args=SimpleNamespace(workspace=str(self.root),source='',checkpoint=str(ckpt),config=str(config),
            run_id='roads',device='cpu',grid='north',period='2020',execution_profile='fast',
            resume=True,pixel_size='0.0',rescale='off',junction_node_mode='sparse',width_method='raw_image')
        pipeline.write_json(self.root/'input_manifest.json',dict(images=str(images),image_txt=str(paths)))
        self.products=self.root/'runs/roads/products';self.products.mkdir(parents=True)
        (self.products/'road_probability.tif').write_bytes(b'probability')
        exported=dict(formal_metadata(),analysis_surfaces='surface',axis_quality_audit='axis',surface_quality_audit='surface',previews={})
        pipeline.write_json(self.products/'fast_export_cache.json',dict(result=exported))
        from engine.width.width_reliability import implementation_identity
        state=pipeline._period_state_template('north','2020','fast')
        state.update(width_method='raw_image',rgb_pipeline_revision=2,width_implementation=implementation_identity(),
            pixel_size='0.0',extraction_inputs=[signature([ckpt,config,self.image]),'fast','off','sparse'])
        state['stages']={key:'completed' for key in state['stages']}
        self.state=state;self.save_state()
        self.commands=[]
        def run(command,cwd,env,label,context):
            self.commands.append(context['stage_key']);return {}
        replacements=dict(run_command=run,_period_stage_output_complete=Mock(return_value=True),
            _shapefile_complete=Mock(return_value=True),build_fusion_metadata=Mock(return_value={}),
            _ensure_extract_manifest_fields=lambda x:x,emit=Mock(),
            _write_valid_observation_area=Mock(return_value='valid'),_write_probability_mosaic=Mock(return_value='probability'))
        patches=patch.multiple(pipeline,**replacements);patches.start();self.addCleanup(patches.stop)

    def save_state(self):pipeline.write_json(self.root/'period_state.json',self.state)

    def test_current_period_skips_all_stages_and_raster_mosaics(self):
        pipeline.extract(self.args)
        self.assertEqual(self.commands,[])
        pipeline._write_valid_observation_area.assert_not_called()
        pipeline._write_probability_mosaic.assert_not_called()

    def test_export_only_does_not_rebuild_probability_or_width(self):
        self.state['stages']['export']='pending';self.save_state()
        pipeline.extract(self.args)
        self.assertEqual(self.commands,['export'])
        pipeline._write_probability_mosaic.assert_not_called()

    def test_changed_width_method_starts_at_width(self):
        self.args.width_method='sam_molra'
        pipeline.extract(self.args)
        self.assertEqual(self.commands,['width','regional','export'])

    def test_changed_resolution_starts_at_width(self):
        self.args.pixel_size='0.5';pipeline.extract(self.args)
        self.assertEqual(self.commands,['width','regional','export'])

    def test_changed_image_or_model_config_cannot_reuse_stale_inference(self):
        self.image.write_bytes(b'changed image');pipeline.extract(self.args)
        self.assertEqual(self.commands,['centerline','surface','width','regional','export'])
        self.commands.clear();Path(self.args.config).write_text('model: changed')
        pipeline.extract(self.args)
        self.assertEqual(self.commands,['centerline','surface','width','regional','export'])

    def test_interrupted_export_retry_skips_finished_stages(self):
        self.state['stages']['export']='failed';self.save_state()
        pipeline.extract(self.args)
        self.assertEqual(self.commands,['export'])
        self.commands.clear();pipeline.extract(self.args)
        self.assertEqual(self.commands,[])

    def test_stop_between_stages_does_not_reuse_stale_downstream(self):
        self.image.write_bytes(b'new source')
        def stop(kind,**event):
            if kind=='stage' and event.get('stage_key')=='centerline' and event.get('status')=='complete':
                raise KeyboardInterrupt()
        with patch.object(pipeline,'emit',side_effect=stop):
            with self.assertRaises(KeyboardInterrupt):pipeline.extract(self.args)
        self.assertEqual(self.commands,['centerline'])
        self.commands.clear();pipeline.extract(self.args)
        self.assertEqual(self.commands,['surface','width','regional','export'])

    def test_period_rerun_reuses_saved_run_directory_and_enables_stage_resume(self):
        entry=dict(grid='north',period='2020',result=str(self.root/'latest_result.json'),run_id='roads',source=str(self.root/'images'))
        manifest=dict(execution_profile='fast',job_root=str(self.root),period_orders={'north':{'period_order':['2020']}},
            auto_period_results=[entry],period_results=[copy.deepcopy(entry)],
            input_spec=dict(checkpoint={'path':self.args.checkpoint},config={'path':self.args.config},width_method='raw_image'))
        with patch.object(pipeline,'prepare',return_value={}),patch.object(pipeline,'extract',return_value={}) as extraction:
            pipeline._rerun_period_entry(manifest,'north','2020')
        args=extraction.call_args.args[0]
        self.assertEqual(args.run_id,'roads');self.assertTrue(args.resume)
        self.assertEqual(args.workspace,str(self.root))
        self.assertEqual((args.grid,args.period),('north','2020'))


class DependencyTests(unittest.TestCase):
    def test_width_identity_ignores_export_edits_but_tracks_helpers_and_constants(self):
        from dependency_identity import local_code_identity
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stages.py'
            source='SCALE=2\ndef helper(x): return x*SCALE\ndef width(x): return helper(x)\ndef export(x): return x\n'
            path.write_text(source)
            original=local_code_identity(path,('width',))
            path.write_text(source.replace('def export(x): return x','def export(x): return x+1'))
            self.assertEqual(original,local_code_identity(path,('width',)))
            for changed in (source.replace('SCALE=2','SCALE=3'),source.replace('x*SCALE','x+SCALE')):
                path.write_text(changed)
                self.assertNotEqual(original,local_code_identity(path,('width',)))


if __name__=='__main__':unittest.main()
