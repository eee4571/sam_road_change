"""Bounded geometry/cache smoke tests. Explicit backend Python; no models."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Barrier, Lock
import time
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT.parent/'code/tests')]
import engine
engine.__path__=[str(ROOT/'code/engine')]
import numpy as np
import cv2
from shapely.geometry import LineString,box
from shapely.ops import unary_union
from shapely.prepared import prep
from engine import road_network_connection as network
from engine.road_geometry import _RegionalRoadSeed
from engine.road_connection_evidence import ConnectionEvidence
from engine.road_axis_cleanup import correct_oscillating_chains,remove_exact_duplicates
from engine.width.raw_feature_cache import FeatureCache
import user_pipeline  # pin plugin entry before workbench fixtures alter sys.path
from test_road_network_products import NetworkProductTests


def seed(points, i=0):
    return _RegionalRoadSeed(np.asarray(points,dtype=float),8.,(i,))


class OptimizationTests(unittest.TestCase):
    def test_candidate_geometry_matches_original_and_avoids_support_work(self):
        # Exact legacy function is a test reference, never a production path.
        tree=ast.parse((ROOT.parent/'code/engine/road_network_connection.py').read_text(encoding='utf8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_continuations')
        namespace=dict(network.__dict__)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<reference>','exec'),namespace)
        roads=[seed([[x,y],[x+25,y]],i) for i,(x,y) in enumerate((
            (x,y) for x in range(0,400,60) for y in range(0,300,50)))]
        lines=[LineString(r.points) for r in roads]
        ports=network._ports(roads,lines,network.STRtree(lines),network._graph(roads))
        surface=prep(box(-10,-10,500,400))
        support=network._support
        with patch.dict(namespace,{'_support':unittest.mock.Mock(wraps=support)}):
            tick=time.perf_counter();before=namespace['_continuations'](ports,roads,150,surface);old=time.perf_counter()-tick
            old_calls=namespace['_support'].call_count
        with patch.object(network,'_support',wraps=support) as calls:
            tick=time.perf_counter();after=network._continuations(ports,roads,150,surface);new=time.perf_counter()-tick
        self.assertEqual([r.row for r in before],[r.row for r in after])
        for a,b in zip(before,after):np.testing.assert_array_equal(a.points,b.points)
        self.assertLess(calls.call_count,old_calls)
        print(f'SMOKE network_candidates old={old:.4f}s new={new:.4f}s support_calls={old_calls}->{calls.call_count}',flush=True)

    def test_incremental_apply_matches_full_noding_and_touches_local_edges(self):
        roads=[seed([[0,0],[20,0]],0),seed([[40,0],[60,0]],1),seed([[30,-10],[30,10]],2)]
        roads += [seed([[1000+i*40,0],[1020+i*40,0]],i+3) for i in range(300)]
        port=network._Port(0,False,np.array([20.,0.]),np.array([1.,0.]),np.array([20.,0.]),np.array([1.,0.]),0.,20.)
        end=network._Port(1,True,np.array([40.,0.]),np.array([-1.,0.]),np.array([40.,0.]),np.array([-1.,0.]),0.,20.)
        candidate=network._Candidate(port,end,1,np.array([[20.,0.],[40.,0.]]),20.,{})
        noding=network._node_network
        with patch.object(network,'_node_network',lambda roads,changed=None:noding(roads)):
            tick=time.perf_counter();expected=network._apply(roads,[candidate]);old=time.perf_counter()-tick
        with patch.object(network,'_contact_geometry',wraps=network._contact_geometry) as contacts:
            tick=time.perf_counter();actual=network._apply(roads,[candidate]);new=time.perf_counter()-tick
        self.assertEqual(len(expected),len(actual))
        for a,b in zip(expected,actual):
            np.testing.assert_allclose(a.points,b.points,atol=1e-10,rtol=0)
            self.assertEqual((a.width_m,a.source_ids),(b.width_m,b.source_ids))
        self.assertLess(contacts.call_count,20)
        print(f'SMOKE network_apply_noding old={old:.4f}s new={new:.4f}s local_contacts={contacts.call_count}',flush=True)

    def test_evidence_caps_search_before_generation(self):
        roads=[seed([[0,0],[40,0]]),seed([[200,0],[240,0]],1)]
        original=network._continuations
        with patch.object(network,'_continuations',wraps=original) as calls:
            result,stats,_=network.connect_clean_road_seeds(roads,evidence=ConnectionEvidence())
        self.assertTrue(calls.call_args_list)
        self.assertTrue(all(c.args[2]==150 for c in calls.call_args_list))
        self.assertEqual(stats['connection_added_count'],0)

    def test_wobble_corrects_internal_axis_and_preserves_sustained_curve(self):
        x=np.arange(0.,161.,1.)
        bend=.002*(x-80)**2
        for base in (np.zeros_like(x),bend):
            road=seed(np.column_stack([x,base+1.2*np.sin(x*np.pi/4)]))
            surface=LineString(np.column_stack([x,base])).buffer(5)
            corrected,count=correct_oscillating_chains([road],ConnectionEvidence(surface))
            self.assertEqual(count,1)
            self.assertLess(LineString(road.points).hausdorff_distance(LineString(corrected[0].points)),2.)
            np.testing.assert_array_equal(corrected[0].points[[0,-1]],road.points[[0,-1]])
            interior=corrected[0].points[(corrected[0].points[:,0]>25)&(corrected[0].points[:,0]<135)]
            expected=.002*(interior[:,0]-80)**2 if np.any(base) else np.zeros(len(interior))
            self.assertLess(np.std(interior[:,1]-expected),.35)
        smooth=seed(np.column_stack([x,bend]))
        corrected,count=correct_oscillating_chains([smooth],ConnectionEvidence(LineString(smooth.points).buffer(5)))
        self.assertEqual(count,0)
        self.assertIs(corrected[0],smooth)

    def test_axis_correction_requires_evidence_and_keeps_junction(self):
        x=np.arange(0.,161.,1.)
        points=np.column_stack([x,1.2*np.sin(x*np.pi/4)])
        roads=network._join_chains(network._node_network([
            seed(points[:81]),seed(points[80:],1),seed([[80,0],[80,40]],2)]))
        result,count=correct_oscillating_chains(roads,ConnectionEvidence())
        self.assertEqual(count,0)
        support=unary_union([LineString(r.points).buffer(5) for r in roads])
        result,count=correct_oscillating_chains(roads,ConnectionEvidence(support))
        self.assertGreater(count,0)
        for a,b in zip(roads,result):np.testing.assert_array_equal(a.points[[0,-1]],b.points[[0,-1]])

    def test_sustained_s_bend_preserved_but_classified_large_fault_repaired(self):
        x=np.arange(0.,241.)
        for abnormal,y in ((False,8*np.sin(x*np.pi/120)),(True,4*np.sin(x*np.pi/4))):
            road=seed(np.column_stack([x,y]))
            result,count=correct_oscillating_chains([road],ConnectionEvidence(LineString(road.points).buffer(10)))
            if abnormal:
                self.assertEqual(count,1)
                self.assertGreater(LineString(road.points).hausdorff_distance(LineString(result[0].points)),2.)
            else:
                self.assertEqual(count,0)
                np.testing.assert_array_equal(result[0].points,road.points)

    def test_duplicate_cleanup_retains_provenance_and_parallel_roads(self):
        first=seed([[0,0],[20,0]],0);reverse=seed([[20,0],[0,0]],1);parallel=seed([[0,1],[20,1]],2)
        result=remove_exact_duplicates([first,reverse,parallel])
        self.assertEqual(len(result),2)
        self.assertEqual(result[0].source_ids,(0,1))
        np.testing.assert_array_equal(result[1].points,parallel.points)

    def test_feature_filters_run_on_four_workers_with_exact_outputs(self):
        class Source:
            width=512;height=512
            def __init__(self):self.reads=0;self.data=np.random.default_rng(3).integers(0,256,(512,512,3),dtype=np.uint8)
            def read_window(self,w):
                self.reads+=1;x,y,width,height=map(int,(w.col_off,w.row_off,w.width,w.height))
                return self.data[y:y+height,x:x+width].copy(),np.ones((height,width),bool)
        source=Source();cache=FeatureCache(source)
        windows=[(np.array([10+i*40,10]),np.array([200+i*40,200])) for i in range(4)]
        serial=FeatureCache(Source())
        expected=[serial.features(*w) for w in windows]
        barrier=Barrier(4);blur=cv2.GaussianBlur
        def synchronized(*args,**kwargs):
            barrier.wait(timeout=5)
            return blur(*args,**kwargs)
        with patch.object(cv2,'GaussianBlur',synchronized),ThreadPoolExecutor(4) as pool:
            actual=list(pool.map(lambda w:cache.features(*w),windows))
        for a,b in zip(expected,actual):
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(source.reads,1)
        self.assertLessEqual(cache.bytes,cache.max_bytes)
        self.assertFalse(cache.pending)
        self.assertGreater(cache.hits,0)

    def test_lru_budget_and_failed_build_can_retry(self):
        cache=FeatureCache(None,max_bytes=16)
        for i in range(4):cache._get(i,lambda:(np.ones(8,np.uint8),))
        self.assertEqual((cache.bytes,cache.evictions),(16,2))
        with self.assertRaisesRegex(ValueError,'read failed'):
            cache._get('bad',lambda:(_ for _ in ()).throw(ValueError('read failed')))
        self.assertNotIn('bad',cache.pending)
        self.assertEqual(cache._get('bad',lambda:(np.ones(4,np.uint8),))[0].sum(),4)


class CurrentNetworkProductTests(NetworkProductTests):
    def test_fast_export_rebuilds_width_layers_and_failed_retry_has_no_marker(self):
        # The workbench fixture predates the separate prepare/export contract.
        # Exercise that current contract, rather than hiding a pipeline call in
        # export or changing the workbench's source/tests.
        import geopandas as gpd
        import os
        from engine.fast_pipeline import prepare_regional_products,export_fast_products
        from engine.road_network_products import network_products_current
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);width=root/'width';width.mkdir();output=root/'products'
            source=self.frame()
            surfaces=gpd.GeoDataFrame(geometry=[box(-10,-10,240,10)],crs=source.crs)
            working=width/'fast_products.gpkg'
            for name,frame in {'centerlines':source,'surfaces':surfaces,'width_segments':source,'corridors':surfaces}.items():
                frame.to_file(working,layer=name,driver='GPKG')
            with patch('engine.fast_pipeline._write_fast_period_previews',return_value={'fusion':'','width':''}):
                prepare_regional_products(width,output)
                export_fast_products(width,output)
                self.assertTrue(network_products_current(output))
                final=gpd.read_file(output/'roads.gpkg',layer='centerlines')
                measured=gpd.read_file(output/'roads.gpkg',layer='width_segments')
                self.assertLess(unary_union(final.geometry).symmetric_difference(unary_union(measured.geometry)).length,.001)
                self.assertAlmostEqual(final.length.sum(),250)
                with patch('engine.road_network_products.recover_centerline_frame',side_effect=AssertionError('repeated recovery')):
                    prepare_regional_products(width,output)
                    export_fast_products(width,output)
            info=working.stat();os.utime(working,ns=(info.st_atime_ns,info.st_mtime_ns+1000000000))
            with patch('engine.road_network_products.recover_centerline_frame',side_effect=RuntimeError('recovery failure')):
                with self.assertRaises(RuntimeError):prepare_regional_products(width,output)
            self.assertFalse(network_products_current(output))


if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(OptimizationTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(CurrentNetworkProductTests),
        unittest.defaultTestLoader.loadTestsFromName('test_rgb_production_chain'),
        unittest.defaultTestLoader.loadTestsFromName('test_raw_image_width_backend')])
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    for name,module in list(sys.modules.items()):
        if name.startswith('engine.') and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(ROOT/'code'),name
    raise SystemExit(not result.wasSuccessful())
