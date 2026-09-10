"""No checkpoints or real projects: batch lifecycle, cache and ordered execution."""
import io
import json
import tempfile
import threading
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
from engine.batch_runtime import ModelPool, Worker, command_key
from engine.bounded_pipeline import Prefetch
from engine.samroad.topology_aggregation import aggregate_votes


class BatchRuntimeTests(unittest.TestCase):
    def test_worker_dispatch_reuses_pool_and_preserves_width_subcommand(self):
        import types
        import engine.batch_runtime as runtime
        fake=types.ModuleType('inferencer')
        from unittest.mock import Mock
        fake.main=Mock()
        requests=[dict(command=['python','inferencer.py','--device','cpu'],cwd=str(Path.cwd())),
                  dict(command=['python','-m','engine.fast_pipeline','width','--device','cpu'],cwd=str(Path.cwd())),
                  dict(command=['python','inferencer.py','--device','cpu'],cwd=str(Path.cwd()))]
        with patch.dict('sys.modules',{'inferencer':fake}), \
             patch('sys.stdin',io.StringIO(''.join(json.dumps(r)+'\n' for r in requests))), \
             patch('sys.stdout',io.StringIO()) as stdout, \
             patch('engine.fast_pipeline.main',return_value=0) as width:
            runtime.main()
        self.assertEqual(stdout.getvalue().count(runtime.PREFIX),3)
        self.assertNotIn('error',stdout.getvalue())
        self.assertEqual(width.call_args.args[0],['width','--device','cpu'])
        self.assertIs(fake.main.call_args_list[0].kwargs['model_pool'],width.call_args.kwargs['model_pool'])
        self.assertIs(fake.main.call_args_list[1].kwargs['model_pool'],width.call_args.kwargs['model_pool'])

    def test_model_family_reuse_and_configuration_invalidation(self):
        class Model:
            def __init__(self): self.moves=[]
            def to(self, device): self.moves.append(str(device)); return self
        pool=ModelPool(); loaded=[]
        def factory():
            model=Model(); loaded.append(model); return model
        a=pool.acquire('samroad',('weights',1),factory,'cpu')
        self.assertIs(a,pool.acquire('samroad',('weights',1),factory,'cpu'))
        b=pool.acquire('molra',('weights',1),factory,'cpu')
        self.assertIs(b,pool.acquire('molra',('weights',1),factory,'cpu'))
        pool.acquire('samroad',('weights',2),factory,'cpu')
        self.assertEqual(len(loaded),3)
        self.assertEqual(len(pool.items),2)
        self.assertEqual(a.moves[-1],'cpu')

    def test_prefetched_period_consumed_once_and_failures_belong_to_next_period(self):
        for fail in (False,True):
            worker=Worker(); calls=[]
            first=['python','inferencer.py','--output_dir','T1']
            second=['python','inferencer.py','--output_dir','T2']
            width=['python','-m','engine.fast_pipeline','width']
            def execute(command,*args):
                calls.append(command)
                if fail and command==second: raise ValueError('T2 failed')
                return 0
            worker._execute=execute
            worker.planner=lambda _:second
            try:
                worker.run(first,Path('.'),{})
                worker.run(width,Path('.'),{})
                if fail:
                    with self.assertRaisesRegex(ValueError,'T2 failed'): worker.run(second,Path('.'),{})
                else: worker.run(second,Path('.'),{})
                self.assertEqual(calls,[first,second,width])
            finally: worker.close()
        self.assertEqual(command_key(['p','i','--a','1','--b']),command_key(['p','i','--b','--a','1']))

    def test_prefetch_is_bounded_ordered_and_exception_is_propagated(self):
        started=threading.Event(); release=threading.Event(); loaded=[]
        def load(i):
            loaded.append(i)
            if i==1:
                started.set()
                if not release.wait(2): raise RuntimeError('CPU did not release producer')
            return i
        with Prefetch(range(5),load) as jobs:
            iterator=iter(jobs)
            self.assertEqual(next(iterator),0)
            self.assertTrue(started.wait(2))
            self.assertEqual(loaded,[0,1])
            release.set()
            self.assertEqual(list(iterator),[1,2,3,4])
        with self.assertRaisesRegex(ValueError,'producer'):
            with Prefetch([1],lambda _: (_ for _ in ()).throw(ValueError('producer'))) as jobs:
                list(jobs)

    def test_topology_scores_counts_and_insertion_order_are_exact(self):
        rng=np.random.default_rng(15)
        old_scores,old_counts=defaultdict(float),defaultdict(float)
        new_scores,new_counts=defaultdict(float),defaultdict(float)
        for _ in range(5):
            pairs=rng.integers(0,8,(3,20,12,2))
            scores=rng.random((3,20,12),dtype=np.float32)
            valid=rng.random((3,20,12))>.25
            maps=[{i:i+b for i in range(8)} for b in range(3)]
            for b,s,p in np.ndindex(scores.shape):
                if valid[b,s,p]:
                    x,y=pairs[b,s,p]; key=(maps[b][x],maps[b][y])
                    old_scores[key]+=scores[b,s,p]; old_counts[key]+=1.
            aggregate_votes(scores,pairs,valid,maps,new_scores,new_counts)
            self.assertEqual(list(old_scores),list(new_scores))
            np.testing.assert_array_equal(list(old_scores.values()),list(new_scores.values()))
            self.assertEqual(old_counts,new_counts)

    def test_project_normalization_reused_without_workspace_copy(self):
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box
        import user_pipeline as p
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); source=root/'image.tif'; area=root/'area.gpkg'
            with rasterio.open(source,'w',driver='GTiff',height=8,width=8,count=3,dtype='uint8',
                               crs=3857,transform=from_origin(0,8,1,1)) as dst:
                dst.write(np.full((3,8,8),100,dtype='uint8'))
            gpd.GeoDataFrame(geometry=[box(0,0,8,8)],crs=3857).to_file(area)
            a=p.normalize_validation_sources({'T1':source},area,root/'_work/tasks/runs/a/normalized',tile_size=8)
            with patch.object(p,'_write_valid_observation_area',side_effect=AssertionError('cache miss')):
                b=p.normalize_validation_sources({'T1':source},area,root/'_work/tasks/runs/b/normalized',tile_size=8)
            self.assertEqual(a,b)
            self.assertTrue(a['T1'].is_relative_to(root/'_work/cache'))
            from argparse import Namespace
            prepared=p.prepare(Namespace(source=str(a['T1']),workspace=str(root/'workspace')))
            self.assertEqual(Path(prepared['images']),a['T1'])
            self.assertFalse((root/'workspace/images').exists())
            c=p.normalize_validation_sources({'T1':source},area,root/'_work/tasks/runs/c/normalized',tile_size=4)
            self.assertNotEqual(a,c)

    def test_corrected_period_cache_and_unchanged_axis_profiles(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        from engine.fast_gt_reconciliation import _write_final_period
        import test_fast_gt_reconciliation as fixtures
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);out=root/'final';out.mkdir()
            fixture=fixtures.FastGTReconciliationTests()
            period=fixture.period(root,'T1',[(LineString([(0,0),(100,0)]),8)])
            base=gpd.read_file(period['centerlines'])
            rows=[dict(geometry=base.geometry.iloc[0],width_m=8.,track_id='RC1')]
            first=_write_final_period(period,base,{},rows,[],out,3857,3857)
            with patch('engine.continuous_road_geometry.network_surface',side_effect=AssertionError('cache miss')):
                second=_write_final_period(period,base,{},rows,[],out,3857,3857)
            self.assertEqual(first,second)
            rows[0]['width_m']=10.
            _write_final_period(period,base,{},rows,[],out,3857,3857)
            self.assertAlmostEqual(gpd.read_file(second['surfaces']).area.sum(),1000.)

    def test_samroad_image_pipeline_writes_both_images_without_loading_models(self):
        import test_relative_toponet_decoupling as fixture
        m=fixture.inferencer
        class Config(dict):
            def __getattr__(self,key):
                if key not in self: raise AttributeError(key)
                return self[key]
            __setattr__=dict.__setitem__
        config=Config(PATCH_SIZE=32)
        def infer(*args,**kwargs):
            nodes=np.array([[5,16],[25,16]],dtype=np.float32)
            edges=np.array([[0,1]],dtype=np.int32); scores=np.array([.9],dtype=np.float32)
            mask=np.zeros((32,32),dtype=np.uint8)
            perf=dict(fast_graph_point_count=2,toponet_candidate_edge_count=1,toponet_final_edge_count=1)
            return nodes,edges,scores,edges,scores,mask,mask,{},None,perf
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw)
            with patch.object(m,'args',m.parser.parse_args(['--execution-profile','fast','--device','cpu'])), \
                 patch.object(m,'resolved_device_name','cpu',create=True), \
                 patch.object(m,'read_rgb_img',return_value=np.zeros((32,32,3),dtype=np.uint8)), \
                 patch.object(m,'rescale_image_to_model_gsd',side_effect=lambda img,*_: (img,1.,1.,1.)), \
                 patch.object(m,'infer_one_img',side_effect=infer) as prediction, \
                 patch.object(m.graph_extraction,'summarize_profile_decisions',return_value={}):
                m.run_inference_on_images(None,config,[root/'a.tif',root/'b.tif'],str(root),'synthetic')
                self.assertEqual(prediction.call_count,2)
                prediction.reset_mock()
                with patch.object(m,'ImageResumeManager') as manager, \
                     patch.object(m,'marker_summaries',return_value=({'tile':'b','total_image_seconds':0.},{})):
                    manager.return_value.inspect.side_effect=[{'action':'process'},
                        {'action':'skip','marker':{},'origin':'marker'}]
                    m.run_inference_on_images(None,config,[root/'a.tif',root/'b.tif'],str(root),'synthetic',
                                              resume_existing_images=True,resume_identity={})
                self.assertEqual(prediction.call_count,1)
            self.assertEqual([r['tile'] for r in json.loads((root/'weak_recovery_summary.json').read_text())['tiles']],['a','b'])
            self.assertTrue((root/'graph/a_fast_topology.npz').is_file())
            self.assertTrue((root/'graph/b_fast_topology.npz').is_file())

    def test_pipelined_width_products_equal_serial_products(self):
        import cv2
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin
        from engine.fast_pipeline import build_fast_surfaces, measure_fast_widths
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); images=root/'images'; images.mkdir()
            probs=root/'probabilities'; probs.mkdir(); (root/'graph').mkdir()
            probability=np.zeros((80,80),dtype=np.uint8); probability[:,30:40]=220
            molra=np.zeros((80,80),dtype=np.float32); molra[:,29:41]=1
            for index in range(2):
                with rasterio.open(images/f'{index}.tif','w',driver='GTiff',width=80,height=80,
                    count=3,dtype='uint8',crs=3857,transform=from_origin(index*80,80,1,1)) as dst:
                    dst.write(np.zeros((3,80,80),dtype=np.uint8))
                cv2.imwrite(str(probs/f'{index}_road.png'),probability)
                cv2.imwrite(str(probs/f'{index}_fast_enhanced.png'),probability)
                np.savez_compressed(root/'graph'/f'{index}_fast_topology.npz',
                    nodes=np.array([[10,35],[70,35]],dtype=np.float32),edges=np.array([[0,1]],dtype=np.int32))
            for name,pool in (('serial',None),('pipelined',ModelPool())):
                surfaces=root/(name+'_surfaces')
                build_fast_surfaces(images,probs,surfaces)
                measure_fast_widths(images,surfaces,probs,root/name,
                                   molra_surface_provider=lambda _: molra.copy(),model_pool=pool)
            for layer in ('centerlines','width_segments','surfaces','corridors'):
                a=gpd.read_file(root/'serial/fast_products.gpkg',layer=layer)
                b=gpd.read_file(root/'pipelined/fast_products.gpkg',layer=layer)
                self.assertEqual(a.geometry.to_wkb().tolist(),b.geometry.to_wkb().tolist())
                if 'width_m' in a: np.testing.assert_array_equal(a.width_m,b.width_m)


if __name__=='__main__': unittest.main()
