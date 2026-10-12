"""Dataset boundaries and interrupted RGB/regional work, without model inference."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from affine import Affine
from shapely.geometry import LineString, MultiLineString, box
from engine.width import raw_image_backend as backend
from engine.width.raw_boundary_core import Config, ImageReader, load_roads, measure_road, sample_line
from engine.width.raw_feature_cache import TileMosaic
from engine.fast_pipeline import prepare_regional_products
from engine.width.width_reliability import InputContractError


class WidthResilienceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.images=self.root/'images';self.images.mkdir()
        self.image=self.images/'a.tif';self.tile(self.image)
        self.output=self.root/'output'
        self.roads=gpd.GeoDataFrame(geometry=[
            LineString([(500020,940),(500100,940)]),
            LineString([(500030,900),(500070,900)])],crs=32650)

    def tile(self,path,count=3,transform=None,nodata=False):
        data=np.full((count,128,128),40,np.uint8)
        data[:,56:64,:]=180;data[:,96:104,:]=180
        if nodata:data[:]=0
        with rasterio.open(path,'w',driver='GTiff',width=128,height=128,count=count,
                           dtype='uint8',crs=32650,nodata=0,
                           transform=transform or from_origin(500000,1000,1,1)) as ds:ds.write(data)

    def working(self):
        width=self.root/'width';width.mkdir()
        surfaces=gpd.GeoDataFrame(geometry=[box(500000,850,500128,1000)],crs=32650)
        for layer,frame in dict(centerlines=self.roads,width_segments=self.roads,
                                surfaces=surfaces,corridors=surfaces).items():
            frame.to_file(width/'fast_products.gpkg',layer=layer,driver='GPKG')
        return width

    def test_short_line_avoids_unbounded_gaussian_kernel(self):
        for length in (1e-6,.01,1.,2.9):
            with self.subTest(length=length),patch('engine.width.raw_boundary_core.gaussian_filter1d',side_effect=AssertionError('unbounded kernel')):
                s,xy,normal=sample_line(LineString([(0,0),(length,0)]),Config())
            self.assertEqual(len(s),1)
            self.assertTrue(np.isfinite(xy).all() and np.isfinite(normal).all())

    def test_metric_crs_and_multiline_input(self):
        source=self.root/'roads.gpkg'
        gpd.GeoDataFrame(geometry=[MultiLineString([[(500000,900),(500010,900)],
            [(500020,900),(500030,900)]])],crs=32650).to_file(source,layer='roads')
        roads,_,crs,_=load_roads(source,'roads',Config(min_length=0,max_roads=0))
        self.assertEqual(len(roads),2);self.assertEqual(crs.to_epsg(),32650)

    def test_invalid_second_tile_rejected_before_network_work(self):
        width=self.working();self.tile(self.images/'b.tif',count=2)
        with patch('engine.road_network_products.recover_centerline_frame') as recover:
            with self.assertRaisesRegex(ValueError,'three uint8 bands'):
                prepare_regional_products(width,self.output,image_dir=self.images,width_method='raw_image')
        recover.assert_not_called()

    def test_all_tiles_checked_and_closed_on_grid_failure(self):
        second=self.images/'b.tif';self.tile(second,transform=Affine(1,0,500000,0,1,872))
        opened=[];real=rasterio.open
        def track(*a,**kw):
            ds=real(*a,**kw);opened.append(ds);return ds
        with patch('engine.width.raw_feature_cache.rasterio.open',side_effect=track):
            with self.assertRaisesRegex(ValueError,'north-up'):
                TileMosaic([self.image,second])
        self.assertTrue(opened and all(ds.closed for ds in opened))

    def test_fractional_grid_offset_rejected(self):
        second=self.images/'b.tif';self.tile(second,transform=from_origin(500128.5,1000,1,1))
        with self.assertRaisesRegex(ValueError,'one pixel grid'):TileMosaic([self.image,second])

    def test_adjacent_tiles_can_measure_across_the_seam(self):
        second=self.images/'b.tif';self.tile(second,transform=from_origin(500128,1000,1,1))
        road=gpd.GeoDataFrame(geometry=[LineString([(500080,940),(500180,940)])],crs=32650)
        result=backend.measure_region(road,self.images,self.output)
        self.assertTrue(np.isfinite(result.final_width).any())
        self.assertGreater(result.geometry.length.sum(),99.)

    def test_empty_and_invalid_roads_fail_before_measurement(self):
        from shapely.geometry import Point
        for geometries in ([],[None],[Point(500010,940)]):
            with self.subTest(geometries=geometries),patch.object(backend,'measure_road') as measure:
                with self.assertRaises((ValueError,TypeError,InputContractError)):
                    backend.measure_region(gpd.GeoDataFrame(geometry=geometries,crs=32650),self.images,self.output)
                measure.assert_not_called()

    def test_clip_point_contact_does_not_enter_recovery(self):
        self.roads.geometry=[LineString([(500020,940),(500100,940)]),
                            LineString([(499990,900),(500000,900)])]
        width=self.working();area=self.root/'area.gpkg'
        gpd.GeoDataFrame(geometry=[box(500000,850,500128,1000)],crs=32650).to_file(area)
        def recovery(frame,*args,**kwargs):
            self.assertTrue(frame.geom_type.eq('LineString').all())
            self.assertEqual(len(frame),1)
            raise RuntimeError('clipping checked')
        with patch('engine.road_network_products.recover_centerline_frame',side_effect=recovery):
            with self.assertRaisesRegex(RuntimeError,'clipping checked'):
                prepare_regional_products(width,self.output,area,self.images,'raw_image')

    def test_wholly_unsupported_region_has_marked_temporary_products(self):
        width=self.working();self.tile(self.image,nodata=True)
        result=prepare_regional_products(width,self.output,image_dir=self.images,width_method='raw_image')
        self.assertEqual(result['status'],'completed_with_warnings')
        self.assertTrue((self.output/'raw_width/completed.json').exists())
        self.assertTrue((self.output/'regional_cache.json').exists())
        roads=gpd.read_file(result['gpkg'],layer='width_segments')
        self.assertTrue(roads.width_m.gt(0).all())
        self.assertTrue(roads.quality_grade.eq('C').all())

    def test_no_coverage_and_nodata_remain_unresolved(self):
        for outside in (False,True):
            with self.subTest(outside=outside):
                self.tile(self.image,nodata=not outside)
                roads=self.roads.copy()
                if outside:roads.geometry=roads.translate(xoff=10000)
                result=backend.measure_region(roads,self.images,self.root/str(outside))
                self.assertTrue(np.isfinite(result.final_width).all())
                self.assertTrue(result.width_source.eq('project_default').all())
                self.assertTrue(result.quality_grade.eq('C').all())

    def test_short_closed_reversed_and_duplicate_vertex_roads(self):
        geometries=[LineString([(500010,940),(500010.01,940)]),
            LineString([(500100,940),(500080,940),(500080,940),(500060,940)]),
            LineString([(500020,900),(500030,900),(500030,910),(500020,910),(500020,900)]),
            LineString([(501000,900),(501020,900)])]
        roads=gpd.GeoDataFrame(geometry=geometries,crs=32650)
        result=backend.measure_region(roads,self.images,self.output)
        self.assertTrue(result.geometry.is_valid.all())
        self.assertTrue(result.geometry.length.gt(0).all())
        self.assertTrue(result.width_source.isin(['project_default','region_median','same_road_median']).any())
        self.assertTrue(np.isfinite(result.final_width).any())

    def test_geographic_and_metric_sources_measure_the_same_roads(self):
        metric=backend.measure_region(self.roads,self.images,self.output)
        geographic=backend.measure_region(self.roads.to_crs(4326),self.images,self.root/'geographic')
        self.assertEqual(geographic.crs.to_epsg(),4326)
        np.testing.assert_allclose(metric.final_width,geographic.final_width,atol=.05,equal_nan=True)

    def test_interrupted_measurement_reuses_completed_chains(self):
        original=backend.measure_road
        def fail_second(reader,line,*args):
            if line.coords[0][1]==900:raise MemoryError('synthetic system interruption')
            return original(reader,line,*args)
        with patch('os.cpu_count',return_value=1),patch.object(backend,'measure_road',side_effect=fail_second):
            with self.assertRaisesRegex(MemoryError,'synthetic system'):
                backend.measure_region(self.roads,self.images,self.output)
        self.assertFalse((self.output/'completed.json').exists())
        self.assertEqual(len(list((self.output/'chains').glob('*/*.completed.json'))),1)
        error=json.loads((self.output/'failure.json').read_text())
        self.assertEqual(error['road_id'],'road_00001')
        self.assertIn('MemoryError',error['traceback'])
        with patch.object(backend,'measure_road',wraps=original) as measure:
            actual=backend.measure_region(self.roads,self.images,self.output)
        self.assertEqual(measure.call_count,1)
        self.assertEqual(json.loads((self.output/'summary.json').read_text())['reused_chains'],1)
        self.assertFalse((self.output/'failure.json').exists())
        expected=backend.measure_region(self.roads,self.images,self.root/'fresh')
        np.testing.assert_allclose(actual.final_width,expected.final_width,equal_nan=True,atol=1e-12)
        self.assertEqual(actual.quality_grade.tolist(),expected.quality_grade.tolist())
        with patch.object(backend,'measure_road',side_effect=AssertionError('completed work repeated')):
            backend.measure_region(self.roads,self.images,self.output)

    def test_image_change_invalidates_chain_checkpoints(self):
        backend.measure_region(self.roads,self.images,self.output)
        stat=self.image.stat();os.utime(self.image,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000000))
        with patch.object(backend,'measure_road',wraps=backend.measure_road) as measure:
            backend.measure_region(self.roads,self.images,self.output)
        self.assertEqual(measure.call_count,2)

    def test_system_failure_is_not_reported_as_success(self):
        for error in (PermissionError('denied'),MemoryError('exhausted')):
            with self.subTest(error=type(error).__name__),patch.object(backend,'measure_road',side_effect=error):
                with self.assertRaises(type(error)):backend.measure_region(self.roads,self.images,self.output)
            self.assertFalse((self.output/'completed.json').exists())

    def test_unwritable_preflight_fails_before_measurement(self):
        with patch.object(backend,'measure_road',side_effect=ArithmeticError('original solver error')), \
             patch.object(Path,'write_text',side_effect=PermissionError('report denied')):
            with self.assertRaisesRegex(PermissionError,'report denied'):
                backend.measure_region(self.roads,self.images,self.output)
        self.assertFalse((self.output/'completed.json').exists())

    def test_width_failure_preserves_network_checkpoint(self):
        width=self.working()
        for attempt in range(2):
            with patch.object(backend,'measure_region',side_effect=RuntimeError('width interrupted')):
                if attempt==0:
                    with self.assertRaisesRegex(RuntimeError,'width interrupted'):
                        prepare_regional_products(width,self.output,image_dir=self.images,width_method='raw_image')
                else:
                    with patch('engine.road_network_products.recover_centerline_frame',side_effect=AssertionError('repeated network')):
                        with self.assertRaisesRegex(RuntimeError,'width interrupted'):
                            prepare_regional_products(width,self.output,image_dir=self.images,width_method='raw_image')
            self.assertTrue((self.output/'network_checkpoint/completed.json').exists())
            self.assertFalse((self.output/'regional_cache.json').exists())
        stat=(width/'fast_products.gpkg').stat()
        os.utime(width/'fast_products.gpkg',ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000000))
        with patch('engine.road_network_products.recover_centerline_frame',side_effect=RuntimeError('new input')):
            with self.assertRaisesRegex(RuntimeError,'new input'):
                prepare_regional_products(width,self.output,image_dir=self.images,width_method='raw_image')


if __name__=='__main__':unittest.main()
