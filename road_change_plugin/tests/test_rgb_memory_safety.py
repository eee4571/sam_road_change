"""Allocation guards and real RGB workers; no model inference or large rasters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_width_production as fixtures
from test_width_production import backend,np,gpd,LineString,rasterio,from_origin
from engine.width import raw_boundary_core as core
from engine.width import raw_width_reconstruction as reconstruction
from engine.width.raw_feature_cache import FeatureCache,TileMosaic
from engine.width.raw_memory import (MemoryPolicy,RoadMemoryLimit,WORKSPACES,
                                     CACHES,CACHE_LOCK,CACHE_BYTES,GDAL_CACHE_BYTES,MIB)


def tile(path,x=500000,y=1000,res=1,size=128):
    data=np.full((3,size,size),40,np.uint8)
    data[:,56:64,:]=180
    with rasterio.open(path,'w',driver='GTiff',width=size,height=size,count=3,
                       dtype='uint8',crs=32650,nodata=0,transform=from_origin(x,y,res,res)) as ds:
        ds.write(data)


def peak_rss():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024


def stress_probe():
    # Four callers request full-road work simultaneously. Only admitted workers
    # allocate; missing tiles are genuine unsupported evidence, never resized.
    with tempfile.TemporaryDirectory() as temp:
        paths=[Path(temp)/f'{i}.tif' for i in range(2)]
        tile(paths[0],y=3000);tile(paths[1],x=502000,y=1000)
        reader=core.ImageReader(paths,32650)
        before=peak_rss()
        lines=[LineString([(500020,2900-i*20),(501420,1500-i*20)]) for i in range(4)]
        try:
            with ThreadPoolExecutor(4) as pool:
                results=list(pool.map(lambda line:backend._solve_road(
                    str(line.coords[0]),line,reader,None,core.Config(),reconstruction.ReconstructionConfig(),'a,b'),lines))
            stats=dict(**WORKSPACES.snapshot(),baseline_peak_rss=before,peak_rss=peak_rss(),
                       max_window_pixels=reader.feature_cache.max_window_pixels,
                       warnings=[len(errors) for frame,errors in results],
                       grades=[frame.quality_grade.unique().tolist() for frame,errors in results])
        finally:reader.close()
        stats['rss_growth']=stats['peak_rss']-before
        return stats


class RGBMemoryTests(unittest.TestCase):
    setUp=fixtures.WidthProductionTests.setUp
    tile=fixtures.WidthProductionTests.tile

    def test_normal_measurements_exactly_match_previous_full_crop_and_nearest(self):
        baseline=backend.measure_region(self.roads,self.images,self.root/'new')
        def original_nearest(s,a):return np.abs(s[:,None]-a[None,:]).argmin(axis=1)
        def original_extract(reader,xy,normal,c):
            reader.thread.windowed=False
            return core._extract_window_profiles(reader,xy,normal,c)
        with patch.object(core,'extract_profiles',new=original_extract),patch.object(reconstruction,'nearest_anchor',new=original_nearest):
            original=backend.measure_region(self.roads,self.images,self.root/'previous')
        for field in ('final_width','final_left_distance','final_right_distance','quality_grade','width_source'):
            np.testing.assert_array_equal(baseline[field],original[field])
        self.assertTrue(baseline.geometry.equals(original.geometry))

    def test_nearest_anchor_preserves_ties_and_is_linear_for_large_inputs(self):
        s=np.arange(1001,dtype=float);a=s[::7]
        expected=np.abs(s[:,None]-a[None,:]).argmin(axis=1)
        np.testing.assert_array_equal(reconstruction.nearest_anchor(s,a),expected)
        np.testing.assert_array_equal(reconstruction.nearest_anchor(np.array([0.,2.,4.,6.]),np.array([1.,3.,5.])),[0,0,1,2])
        s=np.arange(100000,dtype=float)
        result=reconstruction.nearest_anchor(s,s[::2])
        self.assertEqual(len(result),100000)
        self.assertEqual(int(result[-1]),49999)

    def test_giant_bbox_and_ultralong_road_refused_before_sampling_keep_axes(self):
        # 120000² pixels would require 160.9 GiB for rgbf alone. Only two
        # 128² source tiles exist: this is the original giant-mosaic failure.
        tile(self.images/'far.tif',x=620000,y=-119000)
        giant=LineString([(500020,980),(620020,-119020)])
        roads=gpd.GeoDataFrame(geometry=[giant,self.roads.geometry[1]],crs=32650)
        original=core.sample_line
        lengths=[]
        def record(line,c):lengths.append(line.length);return original(line,c)
        with patch.object(core,'sample_line',new=record):
            result=backend.measure_region(roads,self.images,self.output)
        self.assertLess(max(lengths),100)
        bad=result[result.parent_id.eq('road_00000')]
        self.assertEqual(len(bad),1);self.assertTrue(bad.geometry.iloc[0].equals(giant))
        self.assertEqual(bad.quality_grade.iloc[0],'C');self.assertGreater(bad.final_width.iloc[0],0)
        report=json.loads((self.output/'summary.json').read_text(encoding='utf8'))
        self.assertEqual(report['exceptions'][0]['exception_type'],'RoadMemoryLimit')
        with patch.object(backend,'measure_road',side_effect=AssertionError('must reuse checkpoints')):
            (self.output/'completed.json').unlink()
            again=backend.measure_region(roads,self.images,self.output)
        self.assertEqual(len(result),len(again))
        self.assertEqual(json.loads((self.output/'summary.json').read_text(encoding='utf8'))['reused_chains'],2)

    def test_direct_huge_feature_crop_and_read_refused_before_allocating(self):
        reader=core.ImageReader([self.image],32650)
        self.addCleanup(reader.close)
        with patch('engine.width.raw_feature_cache.np.empty',side_effect=AssertionError('allocated')):
            with self.assertRaises(RoadMemoryLimit):
                reader.feature_cache.features(np.array([0,0]),np.array([100000,100000]))
        with patch('engine.width.raw_feature_cache.np.zeros',side_effect=AssertionError('allocated')):
            with self.assertRaises(RoadMemoryLimit):
                reader.ds.read_window(rasterio.windows.Window(0,0,100000,100000))

    def test_road_workspace_refused_before_arrays_and_default_is_c(self):
        roads=gpd.GeoDataFrame(geometry=[LineString([(500020,980),(520020,980)])],crs=32650)
        with patch.object(backend,'measure_road',side_effect=AssertionError('allocation forbidden')):
            result=backend.measure_region(roads,self.images,self.output,default_width_m=8.5)
        self.assertEqual(result.final_width.iloc[0],8.5)
        self.assertEqual(result.quality_grade.iloc[0],'C')
        self.assertIn('road_workspace_budget_exceeded',result.qa_reason.iloc[0])
        self.assertEqual(WORKSPACES.used,0)

    def test_native_gdal_budget_shared_and_restored_after_last_reader_and_failed_open(self):
        from rasterio.env import get_gdal_config
        before=get_gdal_config('GDAL_CACHEMAX')
        first=core.ImageReader([self.image],32650)
        second=core.ImageReader([self.image],32650)
        try:
            self.assertLessEqual(get_gdal_config('GDAL_CACHEMAX'),GDAL_CACHE_BYTES)
            first.close()
            self.assertLessEqual(get_gdal_config('GDAL_CACHEMAX'),GDAL_CACHE_BYTES)
        finally:
            first.close();second.close()
        self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),before)
        with self.assertRaises(rasterio.errors.RasterioIOError):
            core.ImageReader([self.root/'missing.tif'],32650)
        self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),before)

    def test_single_cross_section_too_large_degrades_before_feature_build(self):
        # High-resolution tiles 1 km apart: a single 44 m diagonal section
        # covers millions of pixels even though the road itself is short.
        tile(self.image,res=.01)
        tile(self.images/'far.tif',x=501000,y=0,res=.01)
        line=LineString([(500490,510),(500510,490)])
        roads=gpd.GeoDataFrame(geometry=[line],crs=32650)
        with patch.object(FeatureCache,'_features',side_effect=AssertionError('feature allocation forbidden')):
            result=backend.measure_region(roads,self.images,self.output,default_width_m=7.5)
        self.assertEqual(result.final_width.iloc[0],7.5)
        self.assertEqual(result.quality_grade.iloc[0],'C')
        self.assertTrue(result.geometry.iloc[0].equals(line))

    def test_bounded_windows_follow_road_and_keep_cross_tile_sections(self):
        tile(self.images/'b.tif',x=500128)
        roads=gpd.GeoDataFrame(geometry=[LineString([(500020,940),(500230,940)])],crs=32650)
        baseline=backend.measure_region(roads,self.images,self.root/'normal')
        real_reader=core.ImageReader
        def bounded(*args,**kwargs):return real_reader(*args,memory_policy=MemoryPolicy(max_window_pixels=4096),**kwargs)
        with patch.object(backend,'ImageReader',new=bounded):
            result=backend.measure_region(roads,self.images,self.output)
        report=json.loads((self.output/'summary.json').read_text(encoding='utf8'))
        self.assertLessEqual(report['memory']['max_observed_window_pixels'],4096)
        self.assertGreater(report['memory']['window_builds'],1)
        self.assertTrue(result.quality_grade.eq('C').all())
        self.assertTrue(result.tile_id.str.contains('a').all() and result.tile_id.str.contains('b').all())
        np.testing.assert_allclose(result.final_width,baseline.final_width,rtol=0,atol=1e-10)
        self.assertAlmostEqual(result.length_m.sum(),210)
        self.assertTrue(result.geometry.equals(baseline.geometry))

    def test_global_cache_limit_applies_across_reader_instances(self):
        caches=[FeatureCache(None) for _ in range(3)]
        for i,cache in enumerate(caches):
            cache._get(i,lambda:(np.ones(50*MIB,np.uint8),))
        with CACHE_LOCK:self.assertLessEqual(sum(c.bytes for c in CACHES),CACHE_BYTES)
        for cache in caches:cache.clear()

    def test_width_cache_identity_tracks_memory_policy(self):
        from engine.width.width_reliability import implementation_identity
        previous=implementation_identity()
        original=Path.read_bytes
        def changed(path):
            value=original(path)
            return value+b'\n# changed memory policy' if path.name=='raw_memory.py' else value
        with patch.object(Path,'read_bytes',new=changed):
            self.assertNotEqual(previous,implementation_identity())

    def test_four_worker_real_rgb_peak_memory_and_admission(self):
        process=subprocess.run([sys.executable,__file__,'--memory-probe'],text=True,capture_output=True,timeout=120)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        stats=json.loads(process.stdout.strip().splitlines()[-1])
        print('RGB_MEMORY_STRESS '+json.dumps(stats),flush=True)
        self.assertLessEqual(stats['peak_workspace_bytes'],stats['workspace_limit_bytes'])
        self.assertEqual(stats['active_workspace_bytes'],0)
        self.assertGreater(stats['admission_waits'],0)
        self.assertGreaterEqual(stats['peak_active_workers'],2)
        self.assertLessEqual(stats['max_window_pixels'],MemoryPolicy().max_window_pixels)
        # RSS includes native allocator/runtime overhead outside the algorithm
        # ledger. Test a separate real-process ceiling, not just the counters.
        self.assertLess(stats['rss_growth'],640*MIB)
        self.assertTrue(all(stats['warnings']))
        self.assertTrue(all(grades==['C'] for grades in stats['grades']))


if __name__=='__main__':
    if '--memory-probe' in sys.argv:print(json.dumps(stress_probe()))
    else:unittest.main()
