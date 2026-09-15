"""Small synthetic production checks; no models or real project data."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import LineString
from engine.width.raw_boundary_core import ImageReader,Config,extract_profiles
from engine.width.raw_feature_cache import TileMosaic,FeatureCache
from engine.width.raw_image_backend import original_images,measure_region
from engine.irmad_preprocessing import reference_for,request_identity


class RGBProductionTests(unittest.TestCase):
    def fixture(self,root):
        paths=[]
        rng=np.random.default_rng(19)
        array=rng.integers(30,190,(3,96,128),dtype=np.uint8)
        array[:,44:53,:]=110
        for i in range(2):
            p=root/f'v{i}.tif';paths.append(p)
            with rasterio.open(p,'w',driver='GTiff',width=64,height=96,count=3,
                    dtype='uint8',crs=32650,transform=from_origin(500000+64*i,3000000,1,1),nodata=0) as dst:
                dst.write(array[:,:,64*i:64*(i+1)])
        return paths,array

    def test_tile_cache_grid_values_budget_and_parallel_profiles(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as tmp:
            paths,array=self.fixture(Path(tmp));reader=ImageReader(paths,32650)
            try:
                data,valid=reader.ds.read_window(Window(0,0,128,96))
                np.testing.assert_array_equal(data,array.transpose(1,2,0));self.assertTrue(valid.all())
                xy=np.array([[500030.,2999952.],[500063.,2999952.],[500090.,2999952.]])
                normals=np.tile([0.,1.],(3,1))
                before=extract_profiles(reader,xy,normals,Config())
                with ThreadPoolExecutor(2) as pool:
                    results=list(pool.map(lambda _:extract_profiles(reader,xy,normals,Config()),range(3)))
                for result in results:
                    for a,b in zip(before,result):np.testing.assert_array_equal(a,b)
                self.assertGreater(reader.feature_cache.hits,0)
                self.assertLessEqual(reader.feature_cache.bytes,reader.feature_cache.max_bytes)
            finally:reader.close()

    def test_normalized_tiles_are_not_unwrapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,_=self.fixture(root)
            (root/'normalized_cache.json').write_text('{"irmad_identity":"x","source":"missing-raw"}')
            self.assertEqual(original_images(root),paths)

    def test_rgb_tile_stage_never_calls_molra_or_old_width(self):
        import cv2
        from engine.fast_pipeline import build_fast_surfaces,measure_fast_widths
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'images';images.mkdir()
            with rasterio.open(images/'tile.tif','w',driver='GTiff',width=80,height=80,count=3,
                    dtype='uint8',crs=32650,transform=from_origin(500000,3000000,1,1)) as dst:
                dst.write(np.full((3,80,80),100,np.uint8))
            probability=root/'probability';probability.mkdir();(root/'graph').mkdir()
            values=np.zeros((80,80),np.uint8);values[:,30:40]=220
            for name in ('tile_road.png','tile_fast_enhanced.png'):cv2.imwrite(str(probability/name),values)
            np.savez(root/'graph/tile_fast_topology.npz',nodes=np.array([[10.,35.],[70.,35.]],np.float32),
                     edges=np.array([[0,1],[1,0]],np.int32),scores=np.array([.9,.9]))
            build_fast_surfaces(images,probability,root/'surface')
            with patch('engine.fast_pipeline.measure_fast_path_widths',side_effect=AssertionError('legacy width')), \
                 patch('engine.fast_pipeline._cached_fast_molra_probability',side_effect=AssertionError('MoLRA cache')):
                summary=measure_fast_widths(images,root/'surface',probability,root/'width',width_method='raw_image',
                    molra_surface_provider=lambda _:self.fail('MoLRA inference'))
            self.assertGreater(summary['images'][0]['final_centerline_length'],0)
            self.assertFalse(summary['images'][0]['molra_surface_available'])

    def test_cached_filters_equal_original_full_crop(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            paths,_=self.fixture(Path(tmp));reader=ImageReader(paths,32650)
            try:
                rgb,mask=reader.ds.read_window(Window(7,9,105,78));f=rgb.astype(np.float32)/255
                gray=cv2.cvtColor(f,cv2.COLOR_RGB2GRAY);blur=cv2.GaussianBlur(gray,(0,0),.8)
                edge=np.hypot(cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)/8)
                canny=cv2.dilate(cv2.Canny((blur*255).astype('uint8'),40,100),np.ones((3,3),'uint8'))/255
                texture=np.sqrt(np.maximum(cv2.boxFilter(gray*gray,-1,(7,7))-cv2.boxFilter(gray,-1,(7,7))**2,0))
                expected=(f,cv2.cvtColor(f,cv2.COLOR_RGB2LAB),edge,canny,texture,
                    cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8')).astype(float))
                actual=reader.feature_cache.features(np.array([7,9]),np.array([112,87]))
                for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)
            finally:reader.close()

    def test_region_measurement_representative_and_export_only(self):
        from engine.fast_pipeline import export_fast_products
        from engine.road_network_products import rebuild_network_width_products
        from engine.width.raw_road_surfaces import build_road_surfaces
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,_=self.fixture(root)
            roads=gpd.GeoDataFrame(geometry=[LineString([(500010,2999952),(500117,2999952)])],crs=32650)
            observations=measure_region(roads,root,root/'measure')
            self.assertGreater(len(observations),1)
            segments,corridors=rebuild_network_width_products(roads,observations)
            self.assertEqual(len(segments),1);self.assertFalse(corridors.empty)
            self.assertFalse((root/'measure/regional_rgb.tif').exists())
            frames=dict(centerlines=segments,width_segments=segments,corridors=corridors,
                surfaces=build_road_surfaces(segments,image_path=paths[0]))
            out=root/'products';out.mkdir()
            for key,frame in frames.items():frame.to_file(out/'regional_products.gpkg',layer=key,driver='GPKG')
            with patch('engine.width.raw_image_backend.measure_region',side_effect=AssertionError('export measured')), \
                 patch('engine.road_network_products.recover_centerline_frame',side_effect=AssertionError('export recovered')), \
                 patch('engine.fast_pipeline._write_fast_period_previews',return_value={'fusion':str(paths[0]),'width':str(paths[0])}):
                result=export_fast_products(root/'measure',out,image_dir=root,width_method='raw_image')
            self.assertTrue(Path(result['width_segments']).is_file())
            self.assertEqual(Path(result['width_segments']).suffix,'.gpkg')
            self.assertFalse((out/'road_width_segments.shp').exists())

    def test_independent_area_reference_identity(self):
        config={'a':'2021','b':'2022'}
        self.assertEqual(reference_for(config,'a'),'2021')
        self.assertEqual(reference_for('{"b":"2022"}','b'),'2022')
        with self.assertRaises(ValueError):reference_for(config,'missing')
        self.assertNotEqual(request_identity(True,'2023','target','r1','grid',reference_period='2021'),
                            request_identity(True,'2023','target','r2','grid',reference_period='2022'))


if __name__=='__main__':unittest.main()
