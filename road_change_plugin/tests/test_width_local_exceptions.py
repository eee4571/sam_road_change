"""Fault injection at owned-road versus shared-state/publication boundaries."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from concurrent.futures import CancelledError

sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_width_production as fixtures
from test_width_production import backend,np,gpd,LineString
from shapely import union_all
from shapely.errors import GEOSException
import cv2
from engine.width import raw_boundary_core as core,raw_road_surfaces as surfaces,road_pair_matcher as matcher
from engine.width.width_reliability import SharedWidthStateError,InputContractError,require_road_local


class UnclassifiedRoadError(Exception):
    pass


class LocalExceptionTests(unittest.TestCase):
    setUp=fixtures.WidthProductionTests.setUp
    tile=fixtures.WidthProductionTests.tile
    summary=fixtures.WidthProductionTests.summary

    def bad(self,value):
        if isinstance(value,str):return value=='road_00001'
        if isinstance(value,gpd.GeoDataFrame) or hasattr(value,'columns'):
            field='road_id' if 'road_id' in value else 'parent_id'
            return value.iloc[0][field]=='road_00001'
        if hasattr(value,'get'):return value.get('road_id',value.get('parent_id'))=='road_00001'
        return value.coords[0][1]==900

    def assert_roads(self,result):
        self.assertEqual(set(result.parent_id),{'road_00000','road_00001'})
        for key,axis in zip(('road_00000','road_00001'),self.roads.geometry):
            self.assertTrue(union_all(result.loc[result.parent_id.eq(key)].geometry).equals(axis))
        bad=result[result.parent_id.eq('road_00001')]
        self.assertTrue(bad.quality_grade.eq('C').all())
        self.assertTrue(np.isfinite(bad.final_width).all() and bad.final_width.gt(0).all())
        self.assertTrue(bad.width_source.astype(str).str.len().gt(0).all())
        return result[result.parent_id.eq('road_00000')]

    def test_local_fault_matrix_keeps_other_road_values_and_geometry(self):
        baseline=backend.measure_region(self.roads,self.images,self.root/'baseline')
        expected=baseline[baseline.parent_id.eq('road_00000')]
        types=(KeyError,TypeError,AttributeError,ValueError,IndexError,GEOSException,cv2.error,UnclassifiedRoadError)
        stages=((backend,'measure_road',1),(backend,'rows_for_road',0),
                (backend,'reconstruct_chain',0),(backend,'quality',0),
                (backend,'_fill_road_profile',0),(backend,'_observation_segments',0))
        for module,name,arg in stages:
            original=getattr(module,name)
            for error_type in types:
                with self.subTest(stage=name,error=error_type.__name__):
                    def inject(*args,**kwargs):
                        if self.bad(args[arg]):raise error_type('injected local '+name)
                        return original(*args,**kwargs)
                    output=self.root/(name+error_type.__name__)
                    with patch.object(module,name,new=inject):
                        result=backend.measure_region(self.roads,self.images,output)
                    healthy=self.assert_roads(result)
                    for field in ('final_width','final_left_distance','final_right_distance','quality_grade'):
                        np.testing.assert_array_equal(healthy[field].to_numpy(),expected[field].to_numpy())
                    report=json.loads((output/'summary.json').read_text(encoding='utf8'))
                    self.assertEqual(report['status'],'completed_with_warnings')
                    error=report['exceptions'][0]
                    self.assertEqual(error['road_id'],'road_00001');self.assertEqual(error['tile_id'],'a')
                    self.assertIn(error_type.__name__,error['traceback'])
                    self.assertIn('injected local',error['fallback_reason'])
                    self.assertNotEqual(error['fallback'],'pending')

    def test_solver_mutation_is_discarded_and_profile_contract_is_checked(self):
        real=backend.reconstruct_chain
        def mutate(frame,config):
            if self.bad(frame):
                frame['optimized_width']=-100
                frame.drop(columns=['optimized_left_distance'],inplace=True)
                raise KeyError('mutated local frame')
            return real(frame,config)
        with patch.object(backend,'reconstruct_chain',side_effect=mutate):
            result=backend.measure_region(self.roads,self.images,self.output)
        self.assert_roads(result)
        self.assertTrue(result[result.parent_id.eq('road_00001')].width_source.eq('measured').any())
        def wrong_owner(frame,config):
            value=real(frame,config)
            if self.bad(frame):value['road_id']='wrong-owner'
            return value
        with patch.object(backend,'reconstruct_chain',side_effect=wrong_owner):
            result=backend.measure_region(self.roads,self.images,self.root/'wrong_owner')
        self.assert_roads(result)

    def test_representative_attribute_failure_preserves_segment_associations(self):
        observed=backend.measure_region(self.roads,self.images,self.output)
        original=backend._representative_observation
        def inject(group):
            if self.bad(group):raise AttributeError('local representative')
            return original(group)
        with patch.object(backend,'_representative_observation',side_effect=inject):
            result=backend.rebuild_observations(observed)
        self.assert_roads(result)
        bad=observed[observed.parent_id.eq('road_00001')]
        self.assertEqual(set(result[result.parent_id.eq('road_00001')].segment_id),set(bad.segment_id))
        self.assertEqual(result.attrs['width_diagnostics'][0]['stage'],'representative_width')

    def test_corridor_attribute_and_surface_profile_faults_are_local(self):
        observed=backend.measure_region(self.roads,self.images,self.output)
        for module,name,argument in ((matcher,'_number',0),(surfaces,'_profile_runs',0),(surfaces,'_display_boundaries',0)):
            original=getattr(module,name)
            with self.subTest(stage=name):
                def inject(*args,**kwargs):
                    if self.bad(args[argument]):raise TypeError('local presentation')
                    return original(*args,**kwargs)
                errors=[]
                with patch.object(module,name,side_effect=inject):
                    if module is matcher:
                        result=matcher.build_corridors(observed);errors=result.attrs['width_diagnostics']
                    else:result=surfaces.build_road_surfaces(observed,pixel_size=1.,diagnostics=errors)
                self.assertFalse(result.empty);self.assertTrue(errors)
                self.assertEqual(errors[0]['road_id'],'road_00001')
                self.assertTrue(union_all(result.geometry).covers(self.roads.geometry.iloc[1]))

    def test_formal_profile_width_and_surface_faults_keep_valid_axes(self):
        from engine.formal_road_products import reconstruct_frames
        from engine import canonical_road_surface as canonical,auto_change_geometry as geometry
        centers=self.roads.assign(width_m=[8.,12.],tile_id='a')
        widths=centers.assign(quality_grade='A',valid_ratio=1.,width_std=0.)
        for module,name,arg in ((geometry.FinalWidths,'profile_quality',1),
                                (canonical,'stable_width',1),(canonical,'_chain_surface',0)):
            original=getattr(module,name)
            with self.subTest(stage=name):
                def inject(*args,**kwargs):
                    is_bad=(np.median(args[arg])>10 if name=='stable_width' else self.bad(args[arg]))
                    if is_bad:raise AttributeError('formal local '+name)
                    return original(*args,**kwargs)
                with patch.object(module,name,autospec=True,side_effect=inject):
                    frames,_,audit=reconstruct_frames(centers,widths)
                self.assertTrue(union_all(frames['centerlines'].geometry).equals(union_all(centers.geometry)))
                affected=frames['width_segments'].geometry.distance(centers.geometry.iloc[1])<.01
                self.assertTrue(frames['width_segments'].loc[affected,'quality_grade'].eq('C').all())
                self.assertTrue(frames['width_segments'].loc[~affected,'width_m'].eq(8.).all())
                self.assertTrue(audit['failures'])
                self.assertEqual(audit['failures'][0]['tile_id'],'a')

    def test_formal_invalid_width_attribute_isolated(self):
        from engine.formal_road_products import reconstruct_frames
        centers=self.roads.assign(width_m=[8.,'bad-width'],tile_id='a')
        widths=self.roads.assign(width_m=[8.,12.],quality_grade='A',valid_ratio=1.,width_std=0.)
        frames,_,audit=reconstruct_frames(centers,widths)
        self.assertTrue(union_all(frames['centerlines'].geometry).equals(union_all(centers.geometry)))
        bad=frames['width_segments'].geometry.distance(centers.geometry.iloc[1])<.01
        self.assertTrue(frames['width_segments'].loc[bad,'quality_grade'].eq('C').all())
        self.assertTrue(any(r['stage']=='width_attribute' for r in audit['failures']))

    def test_system_faults_cannot_be_disguised_by_local_boundary(self):
        errors=(MemoryError('memory'),PermissionError('denied'),OSError('disk'),ImportError('dependency'),
            InputContractError('CRS mismatch'),RuntimeError('unknown shared state'),CancelledError(),
            SystemError('native state'),ValueError('resource temporarily unavailable'),
            ValueError('Unable to allocate 40.5 GiB'),
            cv2.error('out of memory'),KeyboardInterrupt())
        for error in errors:
            with self.subTest(error=type(error).__name__),patch.object(backend,'measure_road',side_effect=error):
                with self.assertRaises(type(error)):backend.measure_region(self.roads,self.images,self.output)
            self.assertFalse((self.output/'completed.json').exists())
        try:
            try:raise PermissionError('disk write')
            except PermissionError as cause:raise TypeError('wrapped type') from cause
        except TypeError as error:
            with self.assertRaises(TypeError):require_road_local(error)

    def test_shared_reader_unknown_error_is_fatal(self):
        from engine.width.raw_feature_cache import FeatureCache
        with patch.object(FeatureCache,'features',side_effect=KeyError('shared cache damaged')):
            with self.assertRaises(SharedWidthStateError):
                backend.measure_region(self.roads,self.images,self.output)
        self.assertFalse((self.output/'completed.json').exists())
        failure=json.loads((self.output/'failure.json').read_text(encoding='utf8'))
        self.assertIn('KeyError',failure['traceback'])

    def test_checkpoint_write_unknown_error_is_not_road_recovery(self):
        import pandas as pd
        original=pd.DataFrame.to_json
        def inject(frame,*args,**kwargs):
            if 'road_id' in frame and self.bad(frame):raise TypeError('serializer failure')
            return original(frame,*args,**kwargs)
        with patch.object(pd.DataFrame,'to_json',autospec=True,side_effect=inject):
            with self.assertRaises(TypeError):backend.measure_region(self.roads,self.images,self.output)
        self.assertFalse((self.output/'completed.json').exists())
        with patch.object(backend,'measure_road',wraps=backend.measure_road) as measured:
            backend.measure_region(self.roads,self.images,self.output)
        self.assertEqual(measured.call_count,1)

    def test_output_permission_error_preserves_completed_chain_checkpoints(self):
        original=gpd.GeoDataFrame.to_file
        def inject(frame,path,*args,**kwargs):
            if str(path).endswith('width_observations.partial.gpkg'):raise PermissionError('output denied')
            return original(frame,path,*args,**kwargs)
        with patch.object(gpd.GeoDataFrame,'to_file',autospec=True,side_effect=inject):
            with self.assertRaises(PermissionError):backend.measure_region(self.roads,self.images,self.output)
        self.assertFalse((self.output/'completed.json').exists())
        with patch.object(backend,'measure_road',wraps=backend.measure_road) as measured:
            result=backend.measure_region(self.roads,self.images,self.output)
        self.assertEqual(measured.call_count,0);self.assertEqual(result.parent_id.nunique(),2)

    def test_fast_tile_road_export_keeps_axis_on_unknown_attribute_error(self):
        from engine import fast_pipeline as fast
        from types import SimpleNamespace
        from rasterio.transform import from_origin
        path=SimpleNamespace(pixels=np.array([[10,10],[10,20],[10,30]],dtype=float))
        transform=from_origin(500000,1000,1,1)
        original=fast._world_path(transform,path.pixels)
        for error_type in (KeyError,TypeError,AttributeError):
            errors=[]
            with patch.object(fast,'_simplify_path_pixels',side_effect=error_type('local path attributes')):
                row,corridor,width=fast._fast_road_export(path,dict(width_units=8.,width_source='measured'),
                    transform,1.,1.,'a',0,errors)
            self.assertTrue(row['geometry'].equals(original));self.assertEqual(row['width_qual'],'C')
            self.assertEqual(width['width_units'],8.);self.assertTrue(corridor.covers(original))
            self.assertEqual(errors[0]['tile_id'],'a')
        with patch.object(fast,'_simplify_path_pixels',side_effect=PermissionError('read error')):
            with self.assertRaises(PermissionError):
                fast._fast_road_export(path,{},transform,1.,1.,'a',0,[])

    def test_system_failures_in_reconstruction_and_formal_surface_propagate(self):
        from engine.formal_road_products import reconstruct_frames
        from engine import canonical_road_surface as canonical
        for error in (MemoryError('allocation'),PermissionError('denied'),CancelledError()):
            with self.subTest(stage='reconstruction',error=type(error).__name__),patch.object(backend,'reconstruct_chain',side_effect=error):
                with self.assertRaises(type(error)):backend.measure_region(self.roads,self.images,self.output)
            centers=self.roads.assign(width_m=8.)
            widths=centers.assign(quality_grade='A',valid_ratio=1.,width_std=0.)
            with self.subTest(stage='formal_surface',error=type(error).__name__),patch.object(canonical,'_chain_surface',side_effect=error):
                with self.assertRaises(type(error)):reconstruct_frames(centers,widths)

    def test_local_mask_estimation_error_uses_next_evidence(self):
        from engine.width.width_reliability import surface_width_estimates
        from shapely.geometry import Point,box
        masks=gpd.GeoDataFrame(geometry=[box(500000,880,500128,960)],crs=self.roads.crs)
        original=Point.distance
        def inject(point,other):
            if point.y==900:raise KeyError('local distance')
            return original(point,other)
        errors=[]
        with patch.object(Point,'distance',new=inject):
            result=surface_width_estimates(dict(zip(('good','bad'),self.roads.geometry)),masks,errors,{'bad':'a'})
        self.assertIn('good',result);self.assertNotIn('bad',result)
        self.assertEqual(errors[0]['fallback'],'next_width_evidence')

    def test_subprocess_exit_distinguishes_unknown_local_from_system_fault(self):
        import os,subprocess
        source=self.root/'roads.gpkg';self.roads.to_file(source,driver='GPKG')
        script='''import sys
from unittest.mock import patch
import geopandas as gpd
from engine.width import raw_image_backend as backend
roads=gpd.read_file(sys.argv[1]);real=backend.measure_road
def inject(reader,line,*args):
    if line.coords[0][1]==900:
        if sys.argv[4]=='local':raise KeyError('local subprocess fault')
        raise PermissionError('system subprocess fault')
    return real(reader,line,*args)
with patch.object(backend,'measure_road',new=inject):
    backend.measure_region(roads,sys.argv[2],sys.argv[3])
'''
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        for mode in ('local','system'):
            output=self.root/mode
            result=subprocess.run([sys.executable,'-c',script,str(source),str(self.images),str(output),mode],
                cwd=fixtures.ROOT/'code',env=env,capture_output=True,text=True,encoding='utf8',timeout=60)
            if mode=='local':
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads((output/'summary.json').read_text(encoding='utf8'))
                self.assertEqual(report['status'],'completed_with_warnings')
            else:
                self.assertNotEqual(result.returncode,0)
                self.assertFalse((output/'completed.json').exists())

    def test_regional_postprocessing_warning_survives_export_and_resume(self):
        import check_width_resilience as resilience
        from engine.fast_pipeline import prepare_regional_products
        self.roads=self.roads.assign(width_m=8.)
        width=resilience.WidthResilienceTests.working(self)
        real=backend._representative_observation
        def inject(group):
            if self.bad(group):raise KeyError('regional representative')
            return real(group)
        with patch.object(backend,'_representative_observation',new=inject):
            result=prepare_regional_products(width,self.output,image_dir=self.images,width_method='raw_image')
        self.assertEqual(result['status'],'completed_with_warnings')
        report=json.loads((self.output/'width_quality_summary.json').read_text(encoding='utf8'))
        self.assertGreaterEqual(report['exception_count'],1)
        self.assertTrue(any(r['stage']=='representative_width' for r in report['exceptions']))
        self.assertTrue(all(r['road_id']=='road_00001' for r in report['exceptions']))
        frame=gpd.read_file(result['gpkg'],layer='width_segments')
        raw=gpd.read_file(self.output/'raw_width/width_observations.gpkg',layer='width_observations')
        self.assertEqual(frame.quality_grade.tolist(),raw.quality_grade.tolist())
        self.assertTrue(frame.loc[frame.parent_id.eq('road_00000'),'quality_grade'].isin(['A','B']).all())
        with patch.object(backend,'measure_road',side_effect=AssertionError('unexpected remeasurement')):
            resumed=prepare_regional_products(width,self.output,image_dir=self.images,width_method='raw_image')
        self.assertEqual(resumed['status'],'completed_with_warnings')


if __name__=='__main__':unittest.main()
