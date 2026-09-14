"""Production IR-MAD integration: synthetic RGB, no model inference."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from engine import irmad_core as core
from engine import irmad_preprocessing as rrn


class IRMADTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        rng=np.random.default_rng(801)
        raw=rng.uniform(45,160,(3,64,80))
        self.sources={}
        for period,gain,offset in [('20250118',1.,0.),('20240106',.8,14.),('20260203',1.1,-6.)]:
            folder=self.root/period;folder.mkdir();self.sources[period]=folder
            values=np.clip(np.rint(raw*gain+offset+rng.normal(0,.3,raw.shape)),1,254).astype('uint8')
            values[:,0,0]=255;values[1,0,1]=255
            with rasterio.open(folder/'v0001.tif','w',driver='GTiff',width=80,height=64,count=3,
                               dtype='uint8',nodata=255,crs=32650,transform=from_origin(100,200,1,1)) as dst:
                dst.write(values);dst.update_tags(custom='keep');dst.set_band_description(1,'Red')

    def test_promoted_kernel_ast_is_identical_to_experiment(self):
        original=Path(__file__).resolve().parents[2]/'experiments/radiometric_normalization_ab/irmad_rrn.py'
        nodes=lambda s:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body
                        if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before=nodes(original.read_text(encoding='utf8'));after=nodes(Path(core.__file__).read_text(encoding='utf8'))
        self.assertEqual(set(after),{'windows','paired_blocks','Moments','cca','ncp','fit_irmad','tls','write_normalized'})
        for name in after:self.assertEqual(after[name],before[name],name)

    def test_independent_pairs_reference_identity_and_cache(self):
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for d in self.sources.values() for p in d.glob('*.tif')}
        with patch.object(core,'fit_irmad',wraps=core.fit_irmad) as fit,patch.object(core,'tls',wraps=core.tls) as tls:
            reference=rrn.prepare_period('20250118',self.sources,self.root/'cache',enabled=True)
            self.assertEqual(reference.source,self.sources['20250118']);fit.assert_not_called()
            outputs=[rrn.prepare_period(p,self.sources,self.root/'cache',enabled=True) for p in ('20240106','20260203')]
            self.assertEqual(fit.call_count,2);self.assertEqual(tls.call_count,2)
            repeated=rrn.prepare_period('20240106',self.sources,self.root/'cache',enabled=True)
            self.assertEqual(repeated.metadata['status'],'cache_hit');self.assertEqual(fit.call_count,2)
        audits=[json.loads(Path(o.metadata['audit']).read_text()) for o in outputs]
        self.assertNotEqual(audits[0]['gain'],audits[1]['gain'])
        self.assertTrue(all(a['pif_pixels']>10 for a in audits))
        for period,out in zip(('20240106','20260203'),outputs):
            paired=json.loads((Path(out.metadata['audit']).parent/'paired_tiles.json').read_text())
            self.assertEqual(Path(paired[0]['reference']).parent,self.sources['20250118'])
            self.assertEqual(Path(paired[0]['target']).parent,self.sources[period])
            with rasterio.open(self.sources[period]/'v0001.tif') as a,rasterio.open(out.source/'v0001.tif') as b:
                self.assertEqual((a.crs,a.transform,a.shape,a.nodata,a.dtypes,a.colorinterp),
                                 (b.crs,b.transform,b.shape,b.nodata,b.dtypes,b.colorinterp))
                np.testing.assert_array_equal(a.read_masks(),b.read_masks())
                invalid=a.read_masks()==0;np.testing.assert_array_equal(a.read()[invalid],b.read()[invalid])
                self.assertEqual(b.tags()['custom'],'keep')
        self.assertEqual(hashes,{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes})

    def test_disabled_never_opens_reference_or_creates_cache(self):
        with patch.object(core,'fit_irmad',side_effect=AssertionError('must not run')):
            result=rrn.prepare_period('20260203',{'20260203':self.sources['20260203']},self.root/'cache')
        self.assertEqual(result.source,self.sources['20260203']);self.assertFalse((self.root/'cache').exists())
        self.assertEqual(rrn.workspace_for(self.root/'workspace','raw'),self.root/'workspace')

    def test_grid_mismatch_missing_reference_and_failure_never_fall_back(self):
        with self.assertRaisesRegex(ValueError,'参考期'):
            rrn.prepare_period('20260203',{'20260203':self.sources['20260203']},self.root/'cache',enabled=True)
        with patch.object(core,'fit_irmad',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                rrn.prepare_period('20260203',self.sources,self.root/'cache',enabled=True)
        self.assertFalse(list((self.root/'cache').glob('*/complete.json')))
        with rasterio.open(self.sources['20260203']/'v0001.tif','r+') as dst:
            dst.transform=from_origin(101,200,1,1)
        with self.assertRaisesRegex(ValueError,'网格不一致'):
            rrn.prepare_period('20260203',self.sources,self.root/'cache',enabled=True)

    def test_cache_identity_depends_on_each_pair_and_fixed_parameters(self):
        source=rrn.fingerprint(self.sources['20260203']/'v0001.tif')
        ref=rrn.fingerprint(self.sources['20250118']/'v0001.tif')
        a=rrn.request_identity(True,'20260203',source,ref)
        self.assertNotEqual(a,rrn.request_identity(False,'20260203',source,ref))
        self.assertNotEqual(a,rrn.request_identity(True,'20260203',dict(source,size=1),ref))
        self.assertNotEqual(a,rrn.request_identity(True,'20260203',source,dict(ref,mtime_ns=0)))
        with patch.object(rrn,'VERSION','next'):
            self.assertNotEqual(a,rrn.request_identity(True,'20260203',source,ref))
        self.assertFalse(rrn.extraction_matches({'radiometric_identity':a},'raw'))
        self.assertNotEqual(rrn.workspace_for(self.root,a),rrn.workspace_for(self.root,'raw'))


class PipelineIRMADTests(unittest.TestCase):
    def test_single_period_interrupted_resume_and_reference_invalidation(self):
        import user_pipeline as p
        from app.result_publisher import ResultPublisher
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sources={}
            for period in ('20240106','20250118'):
                folder=root/period;folder.mkdir();(folder/'v0001.tif').write_bytes(b'raw')
                sources[period]=folder
            boundary=root/'area.shp';boundary.write_bytes(b'boundary')
            discovered=dict(project_root=str(root),output_root=str(root/'out'),areas=[dict(
                area_id='g',validation_area=str(boundary),periods=[dict(period=k,source=str(v)) for k,v in sources.items()])])
            args=argparse.Namespace(project_root=str(root),area_id='g',period='20240106',run_id='test',
                                    device='cpu',pixel_size='0',rescale='off',resume=False,irmad=True)
            identities=[]
            def prepare(call):
                identities.append(call.radiometric_identity)
                Path(call.workspace).mkdir(parents=True,exist_ok=True)
            def preprocess(period,raw,cache,**kwargs):
                self.assertEqual(raw,sources)
                return rrn.PreparedInput(sources[period],{})
            with patch.object(p,'discover_validation_project',return_value=discovered), \
                 patch.object(p,'validate_validation_inputs',return_value=sources), \
                 patch.object(p,'normalize_validation_sources',return_value=sources) as normalize, \
                 patch.object(p,'_normalized_sources_ready',return_value=sources) as ready, \
                 patch.object(p,'_period_result_ready',return_value=False), \
                 patch.object(p,'prepare',side_effect=prepare), \
                 patch.object(rrn,'prepare_period',side_effect=preprocess), \
                 patch.object(p,'extract',side_effect=RuntimeError('interrupted')) as extract, \
                 patch.object(ResultPublisher,'publish_period',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):p.extract_project_period(args)
                self.assertEqual(normalize.call_count,1)
                args.resume=True;args.irmad=None;extract.side_effect=None;extract.return_value={}
                p.extract_project_period(args)
                self.assertEqual(normalize.call_count,1);ready.assert_called_once()
                self.assertEqual(identities[0],identities[1])
                (sources['20250118']/'v0001.tif').write_bytes(b'changed reference')
                p.extract_project_period(args)
                self.assertEqual(normalize.call_count,2)
                self.assertNotEqual(identities[1],identities[2])

    def test_nested_irmad_workspace_resume_rebases_owned_cache_only(self):
        from app.project_relocation import repair_task_batch_lists
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'_work/tasks/runs/run'
            workspace=root/'grids/g/periods/20260203/radiometric/identity'
            batches=workspace/'batches';batches.mkdir(parents=True)
            image=Path(temp)/'_work/cache/irmad/pair/normalized_tiles/v0001.tif'
            image.parent.mkdir(parents=True);image.write_bytes(b'raster')
            listing=batches/'grid_tiles.txt'
            listing.write_text('C:/previous/project/_work/cache/irmad/pair/normalized_tiles/v0001.tif\n',encoding='utf-8-sig')
            repair_task_batch_lists(root)
            self.assertEqual(listing.read_text(encoding='utf-8-sig').strip(),str(image.resolve()))
            image.unlink()
            with self.assertRaises(FileNotFoundError):repair_task_batch_lists(root)

    def test_full_pipeline_switch_resume_and_prefetch_inputs(self):
        import user_pipeline as p
        from app.result_publisher import ResultPublisher
        from engine.fast_multitemporal import AUTO_REVISION
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';raw.mkdir();observed=[];callbacks=[]
            for period in ('20240106','20250118','20260203'):
                folder=raw/period;folder.mkdir()
                (folder/'v0001.tif').write_bytes(b'fixture')
            grids={'g':{period:raw/period for period in ('20240106','20250118','20260203')}}
            args=argparse.Namespace(source_root=str(raw),output_root=str(root/'results'),run_id='test',
                checkpoint='m',config='c',device='cpu',pixel_size='0',rescale='off',absolute='2',ratio='.2',
                tolerance='3',execution_profile='fast',irmad=False)
            def preprocess(period,sources,cache_root,*,enabled,origin):
                self.assertEqual(Path(sources['20250118']).parent.name,'20250118')
                folder=Path(sources[period]) if period=='20250118' else root/'normalized'/period
                folder.mkdir(parents=True,exist_ok=True);(folder/'v0001.tif').write_bytes(b'fixture')
                return rrn.PreparedInput(folder,dict(config=rrn.configuration(True),raw_analysis_source=str(sources[period])))
            def extract(args):
                manifest=p.read_json(Path(args.workspace)/'input_manifest.json');observed.append(manifest)
                return dict(radiometric_identity=manifest['radiometric_identity'])
            def finalize(manifest,job):
                manifest['final_period_results']=manifest['period_results'];manifest['temporal_results']=[{'grid':'g'}]
                manifest['fast_finalization_state']='completed'
            with patch.object(p,'discover_grid_periods',return_value=grids), \
                 patch.object(p,'import_raster',side_effect=lambda src,dst:dst.write_bytes(b'fixture')), \
                 patch.object(p,'extract',side_effect=extract) as extraction, \
                 patch.object(p,'_run_fast_change_result',return_value={'fast_auto_revision':AUTO_REVISION}), \
                 patch.object(p,'_finalize_fast_manifest',side_effect=finalize), \
                 patch.object(p,'_period_result_ready',return_value=True),patch.object(p,'_change_result_ready',return_value=True), \
                 patch.object(p,'_temporal_result_ready',return_value=True),patch.object(ResultPublisher,'publish_manifest'), \
                 patch.object(ResultPublisher,'publish_period'),patch.object(ResultPublisher,'publish_reports'), \
                 patch.object(p,'aggregate_change_evaluations'),patch.object(rrn,'prepare_period',side_effect=preprocess) as pre, \
                 patch.object(p,'plan_next_centerline',side_effect=lambda cb:callbacks.append(cb)):
                p.run_all(args);pre.assert_not_called();self.assertEqual(extraction.call_count,3)
                args.resume=True;args.irmad=True;observed.clear();extraction.reset_mock()
                changed=p.run_all(args)
                self.assertEqual(extraction.call_count,3);self.assertEqual(pre.call_count,3)
                self.assertTrue(all(m['radiometric_identity']!='raw' for m in observed))
                self.assertTrue(all('radiometric' in m['workspace'] for m in observed))
                self.assertTrue(all('normalized' in m['source'] for m in observed if '20250118' not in m['source']))
                # Prefetch receives normalized directory and the same isolated workspace.
                command=['python','inferencer.py','--input_txt_dir','x','--output_root','x','--output_dir','x']
                next_command=callbacks[-3](command)
                self.assertIn('radiometric',next_command[next_command.index('--input_txt_dir')+1])
                args.irmad=None;extraction.reset_mock();pre.reset_mock()
                resumed=p.run_all(args);extraction.assert_not_called();pre.assert_not_called()
                self.assertTrue(resumed['input_spec']['irmad']['enabled'])
                args.irmad=False;observed.clear();p.run_all(args)
                self.assertEqual(extraction.call_count,3)
                self.assertTrue(all(m['radiometric_identity']=='raw' for m in observed))

    def test_gui_command_and_cli_boolean_switch(self):
        from app.task_manager import build_pipeline_command, TaskManager
        import user_pipeline as p
        base=dict(mode='grid',source_root=str(Path.cwd()),output_root='out',checkpoint='m',config='c',device='cpu',
                  pixel_size='0',rescale='off',absolute='2',ratio='.2',tolerance='3',execution_profile='fast')
        for enabled in (True,False):
            command=build_pipeline_command(**base,irmad=enabled)
            self.assertEqual(p.parser().parse_args(command).irmad,enabled)
            stage=dict(output_root='out',device='cpu',pixel_size='0',rescale='off',
                       junction_node_mode='sparse',irmad=enabled)
            commands=[TaskManager.build_extract_all(Path.cwd(),'test',**stage),
                      TaskManager.build_extract_period(Path.cwd(),'g','20240106','test',**stage)]
            for command in commands:
                self.assertEqual(p.parser().parse_args(command).irmad,enabled)
        source=(Path(__file__).resolve().parents[1]/'gui/run_page.py').read_text(encoding='utf8')
        self.assertIn('跨时相辐射归一化（IR-MAD）',source);self.assertIn('参考期：20250118',source)


if __name__=='__main__':unittest.main()
