"""Non-model memory regression: publication, evidence, grid planning and Auto."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import rasterio
from rasterio.transform import from_origin,Affine
from rasterio.windows import Window
from rasterio.warp import reproject,Resampling
from PIL import Image
import user_pipeline
from engine.probability_mosaic import write_probability_mosaic
from engine.road_connection_evidence import RoadProbability
from engine.bounded_grid import bounded_grid_shape
from engine.raster_memory import check_native_blocks
from engine.width.width_reliability import InputContractError


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.images=self.root/'images';self.images.mkdir()
        self.probs=self.root/'probabilities';self.probs.mkdir()

    def tile(self,name,transform,size=32):
        path=self.images/(name+'.tif')
        values=np.arange(size*size,dtype=np.uint8).reshape(size,size)
        valid=np.full((size,size),255,np.uint8);valid[:3,:5]=0
        with rasterio.open(path,'w',driver='GTiff',count=3,width=size,height=size,dtype='uint8',
                           crs=32650,transform=transform) as ds:
            ds.write(np.stack([values]*3));ds.write_mask(valid)
        Image.fromarray(values).save(self.probs/(name+'_centerline_probability.png'))
        return path

    def baseline(self,paths):
        sources=[]
        for path in paths:
            with rasterio.open(path) as ds:
                sources.append((ds.transform,ds.bounds,ds.dataset_mask(),np.asarray(Image.open(self.probs/(path.stem+'_centerline_probability.png')))))
        xres=float(np.median([abs(s[0].a) for s in sources]));yres=float(np.median([abs(s[0].e) for s in sources]))
        left=min(s[1].left for s in sources);right=max(s[1].right for s in sources)
        bottom=min(s[1].bottom for s in sources);top=max(s[1].top for s in sources)
        transform=from_origin(left,top,xres,yres)
        shape=(int(np.ceil((top-bottom)/yres)),int(np.ceil((right-left)/xres)))
        result=np.zeros(shape,np.uint8);valid=np.zeros(shape,np.uint8)
        for trans,_,mask,data in sources:
            for source,dst,method in ((data,result,Resampling.bilinear),(mask,valid,Resampling.nearest)):
                reproject(source,dst,src_transform=trans,src_crs=32650,dst_transform=transform,dst_crs=32650,
                          resampling=method,init_dest_nodata=False)
        return result,valid,transform

    def test_streamed_mosaic_matches_full_array_overlap_mask_and_reprojection(self):
        for variant in ('aligned','fractional','rotated'):
            with self.subTest(variant=variant):
                a=self.tile('a',from_origin(500000,1000,1,1))
                transform=(from_origin(500016,996,1,1) if variant=='aligned' else
                           from_origin(500016.3,995.7,.8,.8))
                if variant=='rotated':transform=transform*Affine.rotation(8)
                b=self.tile('b',transform)
                expected,mask,trans=self.baseline([a,b])
                for direct in (True,False):
                    result=user_pipeline._write_probability_mosaic(self.images,self.probs,self.root/'mosaic.tif',allow_direct_placement=direct)
                    with rasterio.open(result) as ds:
                        np.testing.assert_array_equal(ds.read(1),expected)
                        np.testing.assert_array_equal(ds.dataset_mask(),mask)
                        self.assertEqual(ds.transform,trans)

    def test_sparse_far_tiles_do_not_allocate_the_gap(self):
        self.tile('a',from_origin(500000,1000,1,1))
        self.tile('b',from_origin(580000,-79000,1,1))
        real=np.zeros
        def bounded(shape,*args,**kwargs):
            if isinstance(shape,tuple):self.assertLessEqual(np.prod(shape),512*512)
            return real(shape,*args,**kwargs)
        with patch('engine.probability_mosaic.np.zeros',new=bounded):
            result=user_pipeline._write_probability_mosaic(self.images,self.probs,self.root/'sparse.tif')
        self.assertLess(Path(result).stat().st_size,10*1024*1024)
        with rasterio.open(result) as ds:
            self.assertEqual(ds.shape,(80032,80032))
            self.assertFalse(ds.dataset_mask(window=Window(40000,40000,4,4)).any())
            self.assertEqual(ds.read(1,window=Window(80005,80005,1,1))[0,0],165)

    def test_publication_failure_preserves_existing_output(self):
        self.tile('a',from_origin(500000,1000,1,1))
        output=self.root/'existing.tif';output.write_bytes(b'previous-completed-product')
        with patch('engine.probability_mosaic._warp_tile',side_effect=PermissionError('injected')):
            with self.assertRaises(PermissionError):
                user_pipeline._write_probability_mosaic(self.images,self.probs,output,allow_direct_placement=False)
        self.assertEqual(output.read_bytes(),b'previous-completed-product')
        self.assertEqual(list(self.root.glob('probability-*')),[])

    def test_output_block_directory_preflight_rejects_absurd_extent(self):
        self.tile('a',from_origin(500000,1000,1,1))
        self.tile('b',from_origin(1e10,-1e10,1,1))
        with self.assertRaisesRegex(ValueError,'safe block count'):
            user_pipeline._write_probability_mosaic(self.images,self.probs,self.root/'bad.tif')
        self.assertFalse((self.root/'bad.tif').exists())

    def test_sparse_evidence_reads_only_requested_blocks(self):
        path=self.root/'evidence.tif'
        with rasterio.open(path,'w',driver='GTiff',count=1,width=80000,height=80000,dtype='uint8',
                           crs=32650,transform=from_origin(0,80000,1,1),tiled=True,SPARSE_OK=True) as ds:
            ds.write(np.array([[127]],np.uint8),1,window=Window(0,0,1,1))
            ds.write(np.array([[255]],np.uint8),1,window=Window(79999,79999,1,1))
        evidence=RoadProbability([(path,None)],32650)
        real=rasterio.open;reads=[]
        class Reader:
            def __init__(self,ds):self.ds=ds
            def __getattr__(self,name):return getattr(self.ds,name)
            def __enter__(self):return self
            def __exit__(self,*args):self.ds.close()
            def read(self,*args,**kwargs):
                win=kwargs['window'];reads.append(win.width*win.height)
                if win.width*win.height>512*512:raise AssertionError('unbounded raster read')
                return self.ds.read(*args,**kwargs)
        with patch.object(rasterio,'open',new=lambda *a,**k:Reader(real(*a,**k))):
            values=evidence.values(np.array([[.5,79999.5],[79999.5,.5],[90000.,.5]]))
        np.testing.assert_allclose(values[:2],[127/255,1.]);self.assertTrue(np.isnan(values[2]))
        self.assertEqual(reads,[1,1])

    def test_png_evidence_never_decodes_full_image_with_opencv(self):
        image=self.tile('a',from_origin(500000,1000,1,1))
        path=self.probs/'a_centerline_probability.png'
        evidence=RoadProbability([(path,image)],32650)
        with patch('cv2.imdecode',side_effect=AssertionError('full image decode')):
            values=evidence.values(np.array([[500005.5,994.5],[500009.5,990.5]]))
        np.testing.assert_allclose(values,[165/255,41/255])

    def test_grid_budget_is_strict_for_extreme_aspect_ratios_and_rounding(self):
        for x,y in ((1e12,1.),(1.,1e12),(4500,4500),(100.,100.),(1e100,1e-100)):
            for endpoints in (False,True):
                step,w,h=bounded_grid_shape(x,y,1.,200000,endpoints=endpoints)
                self.assertLessEqual(w*h,200000);self.assertGreater(step,0)
        self.assertEqual(bounded_grid_shape(100,100,1,200000),(1,100,100))
        with self.assertRaises(ValueError):bounded_grid_shape(np.inf,1,1,200000)
        from engine.fast_auto_v2 import ChangeEvidence,V2Config
        from shapely.geometry import LineString
        scene=SimpleNamespace(lines=[LineString([(0,0),(1e12,0)])],surfaces=SimpleNamespace(geometry=[]))
        real=np.zeros
        def bounded(shape,*args,**kwargs):
            self.assertLessEqual(np.prod(shape),2000)
            return real(shape,*args,**kwargs)
        with patch('engine.fast_auto_v2.np.zeros',new=bounded):
            field=ChangeEvidence(scene,scene,V2Config(maximum_raster_cells=2000))
        self.assertLessEqual(field.width*field.height,2000)

    def test_native_decoder_preflight(self):
        ds=SimpleNamespace(block_shapes=[(1,100_000_000)],dtypes=['uint8'],count=1,name='giant-strip')
        with self.assertRaises(InputContractError):check_native_blocks(ds)

    def test_auto_native_decoder_failure_closes_file_and_restores_budget(self):
        from engine.fast_auto_change import WindowedProbability
        from rasterio.env import get_gdal_config
        closed=[];before=get_gdal_config('GDAL_CACHEMAX')
        ds=SimpleNamespace(block_shapes=[(1,100_000_000)],dtypes=['uint8'],count=1,name='giant-strip',
                           close=lambda:closed.append(True))
        with patch.object(rasterio,'open',return_value=ds):
            with self.assertRaises(InputContractError):WindowedProbability('giant-strip',32650)
        self.assertTrue(closed);self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),before)

    def test_auto_uses_subwindows_for_oversized_native_strip(self):
        from engine.fast_auto_change import WindowedProbability
        from rasterio.env import get_gdal_config
        path=self.root/'strip.tif'
        with rasterio.open(path,'w',driver='GTiff',count=1,width=600000,height=2,dtype='uint8',
                           crs=32650,transform=from_origin(500000,1000,1,1)) as ds:
            ds.write(np.full((2,600000),255,np.uint8),1)
        before=get_gdal_config('GDAL_CACHEMAX')
        reader=WindowedProbability(path,32650,ram_limit_bytes=0)
        try:
            self.assertLessEqual(get_gdal_config('GDAL_CACHEMAX'),32*1024*1024)
            self.assertLessEqual(reader._block_shape[1],512)
            np.testing.assert_array_equal(reader._values_at(np.array([500000.5,1099999.5]),np.array([999.5,998.5])),[1.,1.])
        finally:reader.close()
        self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),before)


if __name__=='__main__':unittest.main()
