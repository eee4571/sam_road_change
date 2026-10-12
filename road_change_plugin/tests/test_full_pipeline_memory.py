"""Memory contracts and exact small-case equivalence without large allocation."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/engine/samroad'),str(ROOT/'code/engine/width')]
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString,box
from rasterio.transform import Affine
from engine.resource_policy import Resources,overlap_allowed,ResourceSession,try_speculation,GIB
from engine.disk_records import DiskRows,LazyFrames,write_gpkg_rows


class FullMemoryTests(unittest.TestCase):
    def test_host_budget_switches_to_serial(self):
        self.assertFalse(overlap_allowed(snapshot=Resources(2*GIB,8*GIB)))
        self.assertTrue(overlap_allowed(snapshot=Resources(24*GIB,32*GIB)))
        self.assertFalse(overlap_allowed(vram_bytes=GIB,snapshot=Resources(24*GIB,32*GIB,None)))

    def test_shared_ledger_prevents_overbooking_and_releases(self):
        with patch('engine.resource_policy.resources',return_value=Resources(20*GIB,32*GIB)),patch.dict(os.environ,{'ROAD_PIPELINE_RAM_MB':'6144'}):
            with ResourceSession():
                a=try_speculation(2*GIB);b=try_speculation(2*GIB)
                self.assertIsNotNone(a);self.assertIsNotNone(b)
                self.assertIsNone(try_speculation(2*GIB))
                a.close();c=try_speculation(2*GIB);self.assertIsNotNone(c)
                b.close();c.close()

    def test_shared_ledger_across_processes(self):
        import subprocess
        script="""import sys
sys.path.insert(0,sys.argv[1])
import engine.resource_policy as r
r.resources=lambda:r.Resources(20*r.GIB,32*r.GIB)
lease=r.try_speculation(2*r.GIB)
print(int(lease is not None),flush=True)
input()
if lease:lease.close()
"""
        with patch('engine.resource_policy.resources',return_value=Resources(20*GIB,32*GIB)),patch.dict(os.environ,{'ROAD_PIPELINE_RAM_MB':'6144'}):
            with ResourceSession():
                children=[]
                try:
                    for _ in range(3):
                        children.append(subprocess.Popen([sys.executable,'-c',script,str(ROOT/'code')],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
                    admitted=[int(child.stdout.readline()) for child in children]
                    self.assertEqual(sum(admitted),2)
                    for child in children:
                        _,error=child.communicate('\n',timeout=15)
                        self.assertEqual(child.returncode,0,error)
                finally:
                    for child in children:
                        if child.poll() is None:child.kill();child.wait()
                lease=try_speculation(2*GIB);self.assertIsNotNone(lease);lease.close()

    def test_prefetch_serial_parallel_order_and_exception(self):
        from engine.bounded_pipeline import Prefetch
        from engine.resource_policy import Lease
        for permit in (False,True):
            with patch('engine.resource_policy.try_speculation',side_effect=lambda:Lease(None,None) if permit else None):
                with Prefetch(range(20),lambda n:n*n) as jobs:
                    self.assertEqual(list(jobs),[n*n for n in range(20)])
                def fail(n):
                    if n==3:raise MemoryError('injected')
                    return n
                with self.assertRaises(MemoryError),Prefetch(range(8),fail) as jobs:list(jobs)

    def test_tensor_spool_exact_and_cleanup(self):
        import torch
        from inference_memory import TensorSpool
        tensors=[torch.arange(120,dtype=torch.float32).reshape(2,3,4,5),torch.ones(2,2,32,32)]
        with TensorSpool() as spool:
            directory=Path(spool.temp.name)
            for _ in range(10):spool.append(*tensors)
            for i in range(10):
                for k,t in enumerate(tensors):self.assertTrue(torch.equal(spool.read(i,k,'cpu'),t))
        self.assertFalse(directory.exists())
        with TensorSpool(ram_limit=0) as spool:
            spool.append(*tensors)
            self.assertFalse(spool.memory)
            self.assertTrue(torch.equal(spool.read(0,0,'cpu'),tensors[0]))

    def test_topology_queries_bounded_without_truncating_neighbours(self):
        import torch
        from inference_memory import topology_scores
        class Net:
            def infer_toponet(self,features,points,pairs,valid,masks):
                self.assertion(pairs.shape[0]*pairs.shape[1]<=7)
                return pairs.sum(-1,keepdim=True).float()
            assertion=self.assertTrue
        pairs=np.arange(3*29*16*2).reshape(3,29,16,2)
        output=np.empty((3,29,16))
        for b,q,scores in topology_scores(Net(),torch.zeros(3,1,1,1),np.zeros((3,200,2)),
                pairs,np.ones((3,29,16),bool),torch.zeros(3,2,4,4),query_budget=7):output[b,q]=scores
        np.testing.assert_array_equal(output,pairs.sum(-1))
        from inference_memory import topology_batches
        index=SimpleNamespace(count=lambda bounds:4000)
        patches=[(i,(0,0),(512,512)) for i in range(16)]
        plan=list(topology_batches(patches,16,index))
        self.assertEqual([(b,s.start,s.stop) for b,s in plan],[(0,i,i+2) for i in range(0,16,2)])

    def test_pair_frames_mutations_survive_spill(self):
        from engine.disk_records import FrameSpool,FramePair
        spool=FrameSpool();self.addCleanup(spool.close)
        pairs=[FramePair(spool,i,gpd.GeoDataFrame(dict(width=[i]),geometry=[box(i,0,i+1,1)])) for i in range(40)]
        for i,pair in enumerate(pairs):pair['frame'].loc[0,'width']=i+10
        for i,pair in enumerate(pairs):self.assertEqual(pair['frame'].width.iloc[0],i+10)

    def test_external_matching_preserves_ties_and_components(self):
        from engine.evaluation_memory import greedy_pairs,intersection_groups
        candidates=[(.5,0,0),(.5,0,1),(.5,1,0),(.8,2,2)]
        self.assertEqual(greedy_pairs(iter(candidates)),[(2,2),(1,0),(0,1)])
        p=[box(0,0,2,2),box(0,0,2,2),box(10,10,11,11)]
        self.assertEqual(intersection_groups(p,p),[({0,1},{0,1}),({2},{2})])

    def test_grouped_union_exact_and_fatal_allocation(self):
        from engine.geometry_recovery import geometry_union,recovery_scope,Recovery
        from shapely import union_all
        from shapely.errors import GEOSException
        shapes=[box(i,0,i+2,2) for i in range(600)]
        self.assertTrue(geometry_union(iter(shapes)).equals(union_all(shapes)))
        with recovery_scope(Recovery()),patch('shapely.union_all',side_effect=GEOSException('bad_alloc')):
            with self.assertRaises(GEOSException):geometry_union(shapes[:2])

    def test_streamed_corridor_and_chunk_index_match_baseline(self):
        import ast,copy,subprocess
        from engine import road_track_corridors as module,auto_presence_candidates as presence
        def previous(filename,name,namespace):
            source=subprocess.check_output(['git','show',f'9b6dc28:road_change_plugin/code/engine/{filename}'],cwd=ROOT,text=True,encoding='utf8')
            node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
            namespace=dict(namespace)
            exec(compile(ast.Module(body=[node],type_ignores=[]),'baseline','exec'),namespace)
            return namespace[name]
        x=np.arange(0.,400,4.)
        points=np.concatenate([np.column_stack((x,3*np.sin(x/80)+y)) for y in (-10.,10.)])
        tangents=np.tile([1.,0.],(len(points),1))
        before=module.TrackCorridor(np.zeros(2),np.array([1.,0.]),np.array([-10.,10.]),0.,400.,1.)
        after=copy.deepcopy(before)
        previous('road_track_corridors.py','_follow_paired_tracks',module.__dict__)(before,points,tangents)
        module._follow_paired_tracks(after,points,tangents)
        for field in ('stations','centers','spacings'):np.testing.assert_array_equal(getattr(before,field),getattr(after,field))
        lines=[LineString([(v,offset+np.sin(v/30)) for v in np.arange(0.,800.,2.)]) for offset in (0.,5.,100.)]
        old=previous('auto_presence_candidates.py','LongitudinalCoverage',presence.__dict__)(lines,3.)
        new=presence.LongitudinalCoverage(lines,3.)
        for y in (0.,4.,8.,99.):
            axis=LineString([(-10,y),(400,y),(820,y)])
            self.assertEqual(new.uncovered(axis),old.uncovered(axis))
        self.assertLessEqual(len(new.cache),32)

    def test_prefetch_recovers_parallel_admission_and_holds_pending_lease(self):
        from engine.bounded_pipeline import Prefetch
        from engine.resource_policy import Lease
        calls=[];lease=Lease(None,None)
        with patch('engine.resource_policy.try_speculation',side_effect=[None,lease,None]) as admit:
            with Prefetch(range(4),lambda i:calls.append(i) or i) as jobs:
                iterator=iter(jobs)
                self.assertEqual(next(iterator),0)
                self.assertEqual(next(iterator),1)
                self.assertIn(lease,jobs.leases)
                self.assertEqual(list(iterator),[2,3])
            self.assertEqual(admit.call_count,3)

    def test_connection_evidence_window_matches_full_surface(self):
        from engine.vector_evidence import WindowedConnectionEvidence
        from engine.road_connection_evidence import ConnectionEvidence
        line=LineString([(0,0),(100,0)]);surface=box(-100,-3,200,3)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'surface.gpkg'
            gpd.GeoDataFrame(geometry=[surface],crs=3857).to_file(path,driver='GPKG')
            local=WindowedConnectionEvidence([path],3857).local(line,6)
            self.assertEqual(local.measure(line),ConnectionEvidence(surface).measure(line))

    def test_input_and_rescale_dimensions_rejected_before_allocation(self):
        from inference_memory import check_image_shape
        check_image_shape(4096,4096)
        with self.assertRaises(ValueError):check_image_shape(81920,81920,0,stage='GSD')

    def test_irmad_windows_cover_pixels_in_population_order(self):
        from engine.irmad_core import windows
        for width,height in ((4096,260),(600_000,2)):
            result=list(windows(SimpleNamespace(width=width,height=height)))
            self.assertEqual(sum(int(w.width*w.height) for w in result),width*height)
            self.assertTrue(all(w.width*w.height<=128*4096 for w in result))
            starts=[int(w.row_off*width+w.col_off) for w in result]
            self.assertEqual(starts,sorted(starts))

    def test_disk_rows_index_rollback_and_stable_schema_export(self):
        rows=DiskRows();self.addCleanup(rows.close)
        line=LineString([(0,0),(10,0)])
        for i in range(5000):rows.append(dict(road_id=str(i),period='1',width_m=6,geometry=line))
        rows[4999]=dict(rows[4999],fallback_reason='estimate')
        self.assertEqual(rows['12','1']['width_m'],6)
        del rows[4998:]
        self.assertEqual(len(rows),4998)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.gpkg';write_gpkg_rows(path,'roads',rows,3857)
            result=gpd.read_file(path,layer='roads')
            self.assertEqual(len(result),4998);self.assertTrue((result.width_m==6).all())

    def test_lazy_frames_retains_two(self):
        frames=LazyFrames(range(100),lambda n:np.zeros(20)+n)
        for key,value in frames.items():
            self.assertEqual(value[0],key);self.assertLessEqual(len(frames.cache),2)

    def test_long_valid_axis_preserved_with_c_width(self):
        from engine.formal_road_products import reconstruct_frames
        line=LineString([(0,0),(1e9,0)])
        centers=gpd.GeoDataFrame(dict(width_m=[6.],quality_grade=['A']),geometry=[line],crs=3857)
        frames,axis,audit=reconstruct_frames(centers,centers)
        self.assertEqual(len(frames['centerlines']),1)
        self.assertTrue(frames['centerlines'].geometry.iloc[0].equals(line))
        self.assertTrue(frames['width_segments'].quality_grade.eq('C').all())
        self.assertTrue(np.isfinite(frames['width_segments'].width_m).all())
        from engine.road_network_products import recover_centerline_frame
        recovered,_,_=recover_centerline_frame(centers,None)
        self.assertEqual(len(recovered),1)
        self.assertTrue(recovered.geometry.iloc[0].equals(line))

    def test_evaluation_does_not_rasterize_unsafe_grid_or_fabricate_scores(self):
        import road_change_detection as evaluation
        frame=gpd.GeoDataFrame(dict(change_typ=['added']),geometry=[box(0,0,1e12,10)],crs=3857)
        with patch.object(evaluation,'rasterize',side_effect=AssertionError('must preflight')):
            result=evaluation._centerline_offset_metrics(frame,frame,5,matched_pairs=[(0,0)])
        self.assertEqual(result['centerline_offset_status'],'unavailable_resource_limit')
        self.assertIsNone(result['centerline_avg_offset_m'])
        frame.geometry=[box(10,10,80000,80000)]
        with patch.object(evaluation,'rasterize',side_effect=AssertionError('must preflight')):
            result=evaluation.evaluate_fast_assisted_centerline_metrics(frame,frame,truth_type_field='change_typ',
                image_crs=frame.crs,image_transform=Affine.identity(),image_shape=(80032,80032))
        self.assertIsNone(result['centerline_mean_offset_px'])

    def test_temporal_streamed_observations_and_c_width(self):
        import temporal_road_analysis as temporal
        line=LineString([(0,0),(80,0)])
        with tempfile.TemporaryDirectory() as tmp:
            entries=[]
            for i,width in enumerate((6.,12.,6.)):
                path=Path(tmp)/f'p{i}.gpkg'
                gpd.GeoDataFrame(dict(width_m=[width],quality_grade=['C']),geometry=[line],crs=3857).to_file(path,driver='GPKG')
                entries.append(dict(period=str(i),centerlines=str(path)))
            result=temporal.build_temporal_grid('test',entries,[],Path(tmp)/'out')
            observed=gpd.read_file(result['observations_shp'])
            events=gpd.read_file(result['events_shp'])
            self.assertEqual(len(observed),3)
            self.assertFalse(events.event_typ.isin(['widened','narrowed']).any())
            change=Path(tmp)/'change.shp'
            gpd.GeoDataFrame(dict(change_typ=['widened']),geometry=[line.buffer(6)],crs=3857).to_file(change,driver='ESRI Shapefile')
            result=temporal.build_temporal_grid('test',entries,[dict(before_period='0',after_period='1',road_changes=str(change))],Path(tmp)/'with_change')
            events=gpd.read_file(result['events_shp'])
            self.assertFalse(events.event_typ.isin(['widened','narrowed','width_changed']).any())

    def test_streamed_geometry_digest_and_disk_array_exact(self):
        import hashlib,json
        from engine.geometry_memory import json_geometry_digest
        from engine.disk_records import scratch_array
        geometries=gpd.GeoSeries([LineString([(0,0),(100,2)]),None])
        self.assertEqual(json_geometry_digest(geometries),hashlib.sha256(json.dumps(geometries.to_wkb(hex=True).tolist()).encode()).hexdigest())
        with scratch_array((4,8),'int32',ram_limit=0) as array:
            path=Path(array.filename);array[:]=np.arange(32).reshape(4,8)
            self.assertEqual(int(array.sum()),496)
        self.assertFalse(path.exists())

    def test_temporal_rejects_numpy_false_width_quality(self):
        import temporal_road_analysis as temporal
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'unreliable.gpkg'
            gpd.GeoDataFrame(dict(width_m=[6.],quality_grade=['A'],valid_ratio=[.01]),
                geometry=[LineString([(0,0),(80,0)])],crs=3857).to_file(path,driver='GPKG')
            self.assertFalse(temporal._read_period(path).width_reliable.iloc[0])


if __name__=='__main__':unittest.main()
