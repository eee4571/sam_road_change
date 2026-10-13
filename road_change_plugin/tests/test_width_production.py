"""Production width contracts; synthetic GIS only, no model inference."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'code/engine/width'))
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString,Point,box
from engine.width import raw_image_backend as backend
from engine.width.chain_width_calculator import resolve_chain_fallbacks,calculate_chain_hybrid_widths
from engine.width.width_reliability import InputContractError,clean_line,validate_mask,validate_graph
from engine.width.final_width_calculator import FinalWidthRequest,FinalWidthConfig
from engine import fast_pipeline as fast


class WidthProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.images=self.root/'images';self.images.mkdir()
        self.output=self.root/'output';self.image=self.images/'a.tif'
        self.tile(self.image)
        self.roads=gpd.GeoDataFrame(geometry=[LineString([(500020,940),(500100,940)]),
            LineString([(500030,900),(500070,900)])],crs=32650)

    def tile(self,path,nodata=False):
        data=np.full((3,128,128),40,np.uint8);data[:,56:64,:]=180;data[:,96:104,:]=180
        if nodata:data[:]=0
        with rasterio.open(path,'w',driver='GTiff',width=128,height=128,count=3,dtype='uint8',
            crs=32650,nodata=0,transform=from_origin(500000,1000,1,1)) as ds:ds.write(data)

    def summary(self):return json.loads((self.output/'summary.json').read_text(encoding='utf8'))

    def test_network_rebuild_keeps_local_widths_and_grades_on_a_mixed_chain(self):
        from engine.road_network_products import rebuild_network_width_products
        observed=backend.measure_region(self.roads,self.images,self.output)
        chain=observed.index[observed.parent_id.eq('road_00000')]
        self.assertGreater(len(chain),2)
        for index,grade,width in zip(chain[:3],['A','C','B'],[6.,.5,10.]):
            observed.loc[index,['quality_grade','width_quality']]=grade
            observed.loc[index,['final_width','width_m']]=width
        segments,corridors=rebuild_network_width_products(self.roads,observed)
        self.assertEqual(len(segments),len(observed))
        self.assertEqual(segments.quality_grade.tolist(),observed.quality_grade.tolist())
        self.assertEqual(segments.width_m.tolist(),observed.width_m.tolist())
        self.assertFalse(corridors.empty)

    def test_single_road_failure_retains_every_axis(self):
        real=backend.measure_road
        def fail(reader,line,*args):
            if line.coords[0][1]==900:raise IndexError('injected section index')
            return real(reader,line,*args)
        with patch.object(backend,'measure_road',side_effect=fail):
            result=backend.measure_region(self.roads,self.images,self.output)
        self.assertEqual(result.parent_id.nunique(),2)
        self.assertAlmostEqual(result.length_m.sum(),120.)
        self.assertTrue(np.isfinite(result.final_width).all() and result.final_width.gt(0).all())
        report=self.summary();self.assertEqual(report['status'],'completed_with_warnings')
        self.assertEqual(report['exception_count'],1)
        self.assertIn('IndexError',report['exceptions'][0]['traceback'])
        self.assertEqual(report['exceptions'][0]['tile_id'],'a')
        self.assertEqual(report['exceptions'][0]['fallback'],'region_median')
        self.assertTrue(result.loc[result.parent_id=='road_00001','quality_grade'].eq('C').all())

    def test_no_support_uses_configured_default_without_false_measurement(self):
        self.tile(self.image,nodata=True)
        result=backend.measure_region(self.roads,self.images,self.output,default_width_m=7.25)
        self.assertTrue(result.final_width.eq(7.25).all())
        self.assertTrue(result.width_source.eq('project_default').all())
        self.assertTrue(result.quality_grade.eq('C').all())
        self.assertTrue(result.qa_state.eq('review').all())

    def test_wholly_outside_does_not_snap_to_image_border(self):
        roads=self.roads.copy();roads.geometry=roads.translate(xoff=10000)
        result=backend.measure_region(roads,self.images,self.output)
        self.assertTrue(result.width_source.eq('project_default').all())
        self.assertAlmostEqual(result.length_m.sum(),roads.length.sum())

    def test_partial_coverage_short_loop_crossing_and_duplicates(self):
        geometry=[LineString([(499990,940),(500030,940)]),
            LineString([(500020,900),(500020,900),(500020.001,900)]),
            LineString([(500020,920),(500030,930),(500020,930),(500030,920)]),
            LineString([(500060,900),(500070,900),(500070,910),(500060,910),(500060,900)])]
        result=backend.measure_region(gpd.GeoDataFrame(geometry=geometry,crs=32650),self.images,self.output)
        self.assertTrue(result.geometry.is_valid.all())
        self.assertTrue(np.isfinite(result.final_width).all())
        self.assertGreaterEqual(result.length_m.sum(),sum(g.length for g in geometry)-1e-5)

    def test_invalid_geometry_isolated_without_inventing_roads(self):
        roads=gpd.GeoDataFrame(geometry=[self.roads.geometry[0],Point(0,0),
            LineString([(500010,940),(float('nan'),950)]),LineString([(0,0),(0,0)])],crs=32650)
        result=backend.measure_region(roads,self.images,self.output)
        self.assertAlmostEqual(result.length_m.sum(),80.)
        self.assertEqual(self.summary()['exception_count'],3)

    def test_all_invalid_geometry_fails(self):
        with self.assertRaises(InputContractError):
            backend.measure_region(gpd.GeoDataFrame(geometry=[Point(0,0)],crs=32650),self.images,self.output)
        self.assertFalse((self.output/'completed.json').exists())

    def test_system_failures_propagate_and_never_commit(self):
        for error in (PermissionError('denied'),MemoryError('exhausted'),RuntimeError('environment broken')):
            with self.subTest(type=type(error).__name__),patch.object(backend,'measure_road',side_effect=error):
                with self.assertRaises(type(error)):backend.measure_region(self.roads,self.images,self.output)
            self.assertFalse((self.output/'completed.json').exists())

    def test_interrupted_chain_resume_reuses_completed_work(self):
        real=backend.measure_road
        def fail(reader,line,*args):
            if line.coords[0][1]==900:raise MemoryError('injected interruption')
            return real(reader,line,*args)
        with patch('os.cpu_count',return_value=1),patch.object(backend,'measure_road',side_effect=fail):
            with self.assertRaises(MemoryError):backend.measure_region(self.roads,self.images,self.output)
        with patch.object(backend,'measure_road',wraps=real) as measured:
            actual=backend.measure_region(self.roads,self.images,self.output)
        self.assertEqual(measured.call_count,1)
        expected=backend.measure_region(self.roads,self.images,self.root/'fresh')
        np.testing.assert_allclose(actual.final_width,expected.final_width)

    def test_damaged_checkpoint_recomputed(self):
        backend.measure_region(self.roads,self.images,self.output)
        (self.output/'completed.json').unlink()
        checkpoint=next(p for p in (self.output/'chains').glob('*/*.json') if '.completed.' not in p.name)
        checkpoint.write_text('{}')
        with patch.object(backend,'measure_road',wraps=backend.measure_road) as measure:
            backend.measure_region(self.roads,self.images,self.output)
        self.assertEqual(measure.call_count,1)

    def test_default_parameter_invalidates_width_cache(self):
        self.tile(self.image,nodata=True)
        backend.measure_region(self.roads,self.images,self.output,default_width_m=6.)
        result=backend.measure_region(self.roads,self.images,self.output,default_width_m=9.)
        self.assertTrue(result.final_width.eq(9.).all())

    def test_invalid_defaults_fail_before_measurement(self):
        for value in (0.,-1.,np.nan,np.inf):
            with self.subTest(value=value),patch.object(backend,'measure_road') as measure:
                with self.assertRaises(InputContractError):backend.measure_region(self.roads,self.images,self.output,default_width_m=value)
                measure.assert_not_called()

    def test_smoother_failure_preserves_reliable_sections(self):
        with patch.object(backend,'reconstruct_chain',side_effect=ValueError('injected solver')):
            result=backend.measure_region(self.roads,self.images,self.output)
        self.assertTrue(result.width_source.eq('measured').any())
        self.assertTrue(np.isfinite(result.final_width).all())

    def test_fallback_priority_and_non_cascading_donors(self):
        roads={0:LineString([(0,0),(10,0)]),1:LineString([(10,0),(20,0)]),
               2:LineString([(10,0),(10,10)]),3:LineString([(30,0),(40,0)])}
        result=resolve_chain_fallbacks(roads,{0:[8.,np.nan,-1,np.inf]},local_estimates={1:[4.],2:[5.]})
        self.assertEqual(result[0]['width_source'],'same_road_median')
        self.assertEqual(result[1]['width_source'],'chain_neighbor')
        self.assertEqual(result[2]['width_source'],'local_mask_edt')
        self.assertEqual(result[3]['width_source'],'region_median')
        self.assertEqual(result[1]['quality_grade'],'C')

    def test_large_real_width_not_clamped(self):
        result=resolve_chain_fallbacks({0:LineString([(0,0),(10,0)])},{0:[75.,80.,85.]})
        self.assertEqual(result[0]['width_m'],80.)

    def test_many_roads_do_not_require_quadratic_adjacency(self):
        roads={i:LineString([(i*20,0),(i*20+10,0)]) for i in range(5000)}
        result=resolve_chain_fallbacks(roads,{0:[8.]})
        self.assertEqual(len(result),5000)
        self.assertEqual(result[4999]['width_m'],8.)

    def test_mask_shape_and_nonfinite_are_contract_errors(self):
        for mask,shape in ((np.zeros((3,4)),(4,3)),(np.zeros((0,0)),None),(np.full((2,2),np.nan),None)):
            with self.subTest(shape=shape),self.assertRaises(InputContractError):validate_mask(mask,shape)

    def test_graph_bad_indices_and_nonfinite_coordinates_isolated(self):
        errors=[]
        edges,ids=validate_graph(np.array([[0,0],[0,1],[np.inf,2]]),np.array([[0,1],[1,1],[1,2],[0,9],[-1,0]]),errors)
        self.assertEqual(ids,[0]);self.assertEqual(len(errors),4)

    def test_chain_empty_mask_short_and_duplicate_edges(self):
        nodes=np.array([[10,10],[10,10.1],[10,20],[10,20],[20,20]],float)
        request=FinalWidthRequest(nodes,np.array([[0,1],[1,2],[2,3],[3,4]]),np.zeros((32,32)),FinalWidthConfig())
        result=calculate_chain_hybrid_widths(request)
        self.assertTrue(all(r['width_units']>0 for r in result.edge_widths if r['status']!='invalid_geometry'))
        self.assertEqual(result.edge_widths[2]['status'],'invalid_geometry')
        self.assertTrue(all(r['quality_grade']=='C' for r in result.edge_widths))

    def test_cross_section_failure_does_not_abort_other_edge(self):
        from engine.width import molra_centerline_width as molra
        nodes=np.array([[16,2],[16,29],[20,2],[20,29]],float)
        mask=np.zeros((32,32),np.uint8);mask[13:23]=1
        real=molra.maybe_snap_to_mask
        def fail(row,col,*args):
            if row==16:raise IndexError('local')
            return real(row,col,*args)
        with patch.object(molra,'maybe_snap_to_mask',side_effect=fail):
            result=molra.sample_widths_by_normal(nodes,np.array([[0,1],[2,3]]),mask,5,1,20,1,0)
        self.assertTrue(result and all(r['edge_id']==1 for r in result))

    def test_low_quality_width_cannot_drive_auto_change(self):
        from engine.fast_auto_change import RoadScene
        centers=self.roads.iloc[:1].copy()
        widths=centers.assign(width_m=20.,quality_grade='C',valid_ratio=0.,width_std=0.)
        surfaces=gpd.GeoDataFrame(geometry=[box(500000,850,500128,1000)],crs=32650)
        scene=RoadScene(centers,surfaces,widths,surfaces,None,centers.crs)
        self.assertFalse(scene.reliable_width_at(Point(500060,940)))
        scene.widths['quality_grade']='A';scene.widths['valid_ratio']=1.
        self.assertTrue(scene.reliable_width_at(Point(500060,940)))

    def test_fast2_never_builds_profile_from_low_quality_or_missing_grade(self):
        from engine.fast_auto_change import RoadScene
        from engine.fast_auto_v2 import _paired_profiles
        centers=self.roads.iloc[:1].copy()
        surface=gpd.GeoDataFrame(geometry=[box(500000,850,500128,1000)],crs=32650)
        def scene(width,grade):
            frame=centers.assign(width_m=width)
            if grade:frame['quality_grade']=grade
            return RoadScene(centers,surface,frame,surface,None,centers.crs)
        before=scene(8.,'A');axis=before.lines[0]
        for grade in ('C',None):
            after=scene(20.,grade)
            self.assertEqual(_paired_profiles(axis,[(0.,axis.length,0)],before,after,2.),[])
        self.assertTrue(_paired_profiles(axis,[(0.,axis.length,0)],before,scene(20.,'A'),2.))

    def test_local_mask_estimate_precedes_region_and_default(self):
        self.tile(self.image,nodata=True)
        surfaces=gpd.GeoDataFrame(geometry=[box(500010,936,500110,944)],crs=32650)
        result=backend.measure_region(self.roads.iloc[:1],self.images,self.output,surface_evidence=surfaces)
        self.assertTrue(result.width_source.eq('local_mask_edt').all())
        self.assertTrue(result.final_width.eq(8.).all())
        self.assertTrue(result.quality_grade.eq('C').all())

    def test_surface_construction_failure_retains_road(self):
        from engine.width import raw_road_surfaces as surfaces
        result=backend.measure_region(self.roads,self.images,self.output)
        errors=[]
        with patch.object(surfaces,'_display_boundaries',side_effect=ValueError('bad offset ring')):
            polygons=surfaces.build_road_surfaces(result,pixel_size=1.,diagnostics=errors)
        self.assertFalse(polygons.empty)
        self.assertEqual(len(errors),2)
        self.assertTrue(all(e['fallback']=='symmetric_buffer' for e in errors))
        self.assertTrue(polygons.width_qual.eq('C').all())
        self.assertTrue(polygons.qa_state.eq('review').all())

    def test_final_profile_failure_cannot_delete_valid_axis(self):
        from engine.formal_road_products import reconstruct_frames
        centers=self.roads.iloc[:1].assign(width_m=8.)
        widths=centers.assign(quality_grade='A',valid_ratio=1.,width_std=0.)
        with patch('engine.auto_change_geometry.FinalWidths.profile_quality',side_effect=IndexError('profile map')):
            frames,_,audit=reconstruct_frames(centers,widths)
        self.assertFalse(frames['centerlines'].empty)
        self.assertTrue(frames['width_segments'].quality_grade.eq('C').all())
        self.assertTrue(audit['failures'])

    def test_null_geometry_is_isolated_before_cache_identity(self):
        roads=gpd.GeoDataFrame(geometry=[None,self.roads.geometry.iloc[0]],crs=self.roads.crs)
        result=backend.measure_region(roads,self.images,self.output)
        self.assertEqual(result.parent_id.nunique(),1)
        self.assertEqual(self.summary()['exception_count'],1)
        self.assertEqual(self.summary()['exceptions'][0]['fallback'],'invalid_geometry_excluded')

    def test_malformed_completed_manifest_is_not_reused(self):
        from engine.product_cache import read_completed
        path=self.root/'completed.json'
        for paths,result in [(None,{}),([],['invalid result']),([None],{})]:
            path.write_text(json.dumps(dict(inputs=[],outputs=[],paths=paths,result=result)),encoding='utf8')
            self.assertIsNone(read_completed(path,[]))

    def test_final_segment_failure_retains_axis_and_low_quality(self):
        from engine.formal_road_products import reconstruct_frames
        centers=self.roads.iloc[:1].assign(width_m=8.)
        widths=centers.assign(quality_grade='A',valid_ratio=1.,width_std=0.)
        with patch('shapely.ops.substring',side_effect=IndexError('segment mapping')):
            frames,_,audit=reconstruct_frames(centers,widths)
        self.assertAlmostEqual(frames['centerlines'].length.sum(),centers.length.sum())
        self.assertTrue(frames['width_segments'].quality_grade.eq('C').all())
        self.assertTrue(any(r['stage']=='formal_chain' for r in audit['failures']))

    def test_native_oom_is_systemic(self):
        with patch.object(backend,'measure_road',side_effect=ValueError('failed to allocate buffer')):
            with self.assertRaisesRegex(ValueError,'failed to allocate'):
                backend.measure_region(self.roads,self.images,self.output)
        self.assertFalse((self.output/'completed.json').exists())

    def test_gpkg_final_values_and_quality_survive_roundtrip(self):
        self.tile(self.image,nodata=True)
        result=backend.measure_region(self.roads,self.images,self.output)
        actual=gpd.read_file(self.output/'width_observations.gpkg',layer='width_observations')
        self.assertEqual(actual.crs,result.crs)
        self.assertTrue(actual.width_unit.eq('m').all())
        self.assertTrue(np.isfinite(actual.final_width).all())
        self.assertTrue(actual.quality_grade.eq('C').all())
        self.assertTrue(actual.segment_id.is_unique)

    def test_supported_measurements_are_unchanged_by_resilience_policy(self):
        result=backend.measure_region(self.roads,self.images,self.output)
        import pandas as pd
        frames=[pd.read_json(path,orient='table') for path in sorted((self.output/'chains').glob('*/road_*.json'))
                if '.completed.' not in path.name]
        # Checkpoint profiles are the original sampler/smoother output, before fallback policy.
        baseline=pd.concat(frames,ignore_index=True)
        good=np.isfinite(baseline.final_width)&baseline.final_width.gt(0)
        self.assertTrue(good.all())
        np.testing.assert_array_equal(result.final_width,baseline.final_width)
        np.testing.assert_array_equal(result.final_left_distance,baseline.final_left_distance)
        np.testing.assert_array_equal(result.final_right_distance,baseline.final_right_distance)

    def prepare_fast(self):
        surfaces=self.root/'surfaces';surfaces.mkdir()
        probability=self.root/'infer/mask';probability.mkdir(parents=True)
        graphs=probability.parent/'graph';graphs.mkdir()
        mask=np.zeros((128,128),np.uint8);mask[56:65]=255
        for stem in ('a','b'):
            if stem=='b':self.tile(self.images/'b.tif')
            cv2.imwrite(str(surfaces/f'{stem}_mask.png'),mask)
            cv2.imwrite(str(probability/f'{stem}_road.png'),mask)
            np.savez(graphs/f'{stem}_fast_topology.npz',nodes=np.array([[60,10],[60,110]],np.float32),edges=np.array([[0,1]]))
        return surfaces,probability

    def test_tile_failure_retains_source_and_other_tile(self):
        surfaces,probability=self.prepare_fast()
        real=fast.regularize_fast_road_network;count=0
        def fail(*args,**kwargs):
            nonlocal count
            count+=1
            if count==2:raise ValueError('bad tile topology')
            return real(*args,**kwargs)
        with patch.object(fast,'regularize_fast_road_network',side_effect=fail):
            summary=fast.measure_fast_widths(self.images,surfaces,probability,self.output,width_method='raw_image')
        frame=gpd.read_file(self.output/'fast_products.gpkg',layer='centerlines')
        self.assertEqual(set(frame.tile),{'a','b'})
        self.assertEqual(summary['status'],'completed_with_warnings')
        self.assertEqual(len(list((self.output/'tile_checkpoints').glob('*.json'))),2)
        with patch.object(fast,'regularize_fast_road_network',side_effect=AssertionError('repeated completed tile')):
            fast.measure_fast_widths(self.images,surfaces,probability,self.output,width_method='raw_image')

    def test_tile_system_interruption_preserves_checkpoint_and_masks(self):
        surfaces,probability=self.prepare_fast()
        original=(surfaces/'a_mask.png').read_bytes()
        real=fast.regularize_fast_road_network;count=0
        def fail(*args,**kwargs):
            nonlocal count
            count+=1
            if count==2:raise MemoryError('stop second tile')
            return real(*args,**kwargs)
        with patch.object(fast,'regularize_fast_road_network',side_effect=fail):
            with self.assertRaises(MemoryError):fast.measure_fast_widths(self.images,surfaces,probability,self.output,width_method='raw_image')
        self.assertTrue((self.output/'tile_checkpoints/a.json').exists())
        self.assertFalse((self.output/'width_completed.json').exists())
        self.assertEqual((surfaces/'a_mask.png').read_bytes(),original)
        with patch.object(fast,'regularize_fast_road_network',wraps=real) as called:
            fast.measure_fast_widths(self.images,surfaces,probability,self.output,width_method='raw_image')
        self.assertEqual(called.call_count,1)

    def test_cli_exit_matches_warning_and_contract_failure(self):
        import subprocess
        surfaces,probability=self.prepare_fast()
        graph=probability.parent/'graph/a_fast_topology.npz'
        np.savez(graph,nodes=np.array([[60,10],[60,110]],np.float32),edges=np.array([[0,1],[1,1]]))
        command=[sys.executable,'-m','engine.fast_pipeline','width','--image-dir',str(self.images),
            '--surface-dir',str(surfaces),'--probability-dir',str(probability),'--output-dir',str(self.output),
            '--width-method','raw_image']
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        result=subprocess.run(command,cwd=ROOT/'code',env=env,capture_output=True,text=True,encoding='utf8',timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        summary=json.loads((self.output/'batch_width_summary.json').read_text())
        self.assertEqual(summary['status'],'completed_with_warnings')
        cv2.imwrite(str(surfaces/'a_mask.png'),np.zeros((5,5),np.uint8))
        result=subprocess.run(command,cwd=ROOT/'code',env=env,capture_output=True,text=True,encoding='utf8',timeout=60)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Mask/image shape mismatch',result.stderr)
        self.assertFalse((self.output/'width_completed.json').exists())


if __name__=='__main__':unittest.main()
