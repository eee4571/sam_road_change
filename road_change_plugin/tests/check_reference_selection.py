"""Bounded sampling/ranking with an injected predictor, plus proxy metric tests."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from app import reference_selection as selection


class FakeProbe:
    version = 'test-model-v1'
    def identity(self):
        return [self.version]
    def evaluate(self, rgb, valid):
        pixels = rgb[valid]
        score = float(pixels.mean()/255) if np.percentile(pixels,90)-np.percentile(pixels,10)>1 else 0.
        assert rgb.shape == (512,512,3)
        return dict(score=score, length_m=score*100, valid_area_m2=36864., edge_count=10 if score else 0)


class ReferenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.probe_patch = patch.object(selection, 'RoadModelProbe', FakeProbe)
        self.probe_patch.start()
        self.addCleanup(self.probe_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.geometry = box(0, 0, 2048, 2048)
        self.boundary = self.root / 'area.shp'
        gpd.GeoDataFrame(geometry=[self.geometry], crs='EPSG:3857').to_file(self.boundary)
        yy, xx = np.indices((2048, 2048))
        self.rgb = np.stack([60 + (xx + yy + offset) % 120 for offset in (0, 20, 40)]).astype('uint8')
        self.data = dict(project_root=str(self.root), areas=[['A', str(self.boundary)]], periods=[], area_irmad_references={})
        for period, offset in (('2020', -35), ('2021', 0), ('2022', 35)):
            self.add(period, np.clip(self.rgb.astype(int) + offset, 1, 254).astype('uint8'))

    def add(self, period, pixels, nodata=None, area='A'):
        image = self.root / f'{area}-{period}.tif'
        with rasterio.open(image, 'w', driver='GTiff', height=2048, width=2048, count=3,
                           dtype='uint8', crs='EPSG:3857', transform=from_origin(0, 2048, 1, 1),
                           nodata=nodata, tiled=True, blockxsize=128, blockysize=128) as ds:
            ds.write(pixels)
        listing = image.with_suffix('.txt')
        listing.write_text(image.name, encoding='utf8')
        self.data['periods'].append([area, period, str(listing)])
        return image

    def test_two_passes_are_disjoint_and_spatially_balanced(self):
        first, weights = selection.sampling_plan(self.geometry)
        second, extra_weights = selection.sampling_plan(self.geometry,supplemental=True)
        self.assertEqual(len(first),8)
        self.assertEqual(len(second),8)
        self.assertFalse(set(first)&set(second))
        self.assertAlmostEqual(sum(weights+extra_weights),1.)
        quadrants = [(x>=1024,y>=1024) for x,y in first]
        self.assertEqual([quadrants.count(q) for q in ((False,False),(False,True),(True,False),(True,True))],[2,2,2,2])

    def test_merge_weights_unequal_area_passes_correctly(self):
        first = dict(sampled_coverage=.5,patches=[],windows_read=8,cached=True)
        second = dict(sampled_coverage=1.,patches=[],windows_read=7,cached=True)
        merged = selection.merge_samples(first,second,.8,.2)
        self.assertAlmostEqual(merged['sampled_coverage'],.6)
        self.assertEqual(merged['windows_read'],15)

    def test_model_winner_not_spectral_middle_and_bounded_reads(self):
        original = rasterio.open
        reads = []
        class Reader:
            def __init__(self, source): self.source = source
            def __getattr__(self, name): return getattr(self.source, name)
            def __enter__(self): return self
            def __exit__(self, *args): self.source.close()
            def read(self, *args, **kwargs):
                window = kwargs['window']
                reads.append((window, kwargs.get('out_shape')))
                assert 0 < window.width <= 2048 and 0 < window.height <= 2048
                assert kwargs.get('out_shape') in (None, (3, 64, 64))
                assert kwargs['boundless']
                return self.source.read(*args, **kwargs)
        with patch.object(selection.rasterio, 'open', side_effect=lambda *a, **kw: Reader(original(*a, **kw))):
            result = selection.recommend(self.data)
        self.assertEqual(result['issues'], [])
        self.assertEqual(result['area_irmad_references']['A'], '2022')
        self.assertEqual(sum(size is None for _, size in reads), 24)
        self.assertEqual(sum(size is not None for _, size in reads), 24)
        self.assertTrue(all(p.width*p.height < 2048*2048 for p, _ in reads))

    def test_saved_selection_is_not_replaced(self):
        self.data['area_irmad_references']['A'] = '2020'
        with patch.object(selection, 'sample_period', side_effect=AssertionError('must not resample')):
            result = selection.recommend(self.data)
        self.assertEqual(result['area_irmad_references']['A'], '2020')
        self.assertEqual(result['reference_selections'], {})
        self.assertEqual(selection.recommend(self.data, force=True)['area_irmad_references']['A'], '2022')

    def test_close_candidates_trigger_only_one_bounded_supplement(self):
        self.data['periods'] = []
        for period in ('2020', '2021', '2022'):
            self.add(period, self.rgb)
        result = selection.recommend(self.data, use_cache=False)
        report = result['reference_selections']['A']
        self.assertTrue(report['supplemental_sampling'])
        self.assertTrue(report['close_candidates'])
        self.assertEqual(report['sampled_locations'], 16)
        self.assertTrue(all(s['windows_read'] == 16 for s in report['candidates'].values()))
        self.assertEqual(report['period'], '2020')

    def test_cache_skips_pixels_and_changed_source_invalidates_it(self):
        result = selection.recommend(self.data)
        with patch.object(selection, '_patch', side_effect=AssertionError('cached')):
            cached = selection.recommend(self.data)
        self.assertEqual(cached['area_irmad_references'], result['area_irmad_references'])
        self.assertTrue(all(s['cached'] for s in cached['reference_selections']['A']['candidates'].values()))
        path = self.root / 'A-2021.tif'
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000_000))
        with patch.object(selection, '_patch', wraps=selection._patch) as read:
            selection.recommend(self.data)
        self.assertEqual(read.call_count, 8)

    def test_no_road_evidence_needs_manual_selection(self):
        self.data['periods'] = []
        self.add('2020', np.full_like(self.rgb, 90))
        self.add('2021', np.full_like(self.rgb, 255))
        result = selection.recommend(self.data, use_cache=False)
        self.assertTrue(result['issues'])
        self.assertIsNone(result['reference_selections']['A']['period'])

    def test_shared_nodata_and_unique_missing_coverage_are_distinct(self):
        self.data['periods'] = []
        for period, offset in (('2020', -35), ('2021', 0), ('2022', 35)):
            pixels = np.clip(self.rgb.astype(int)+offset, 1, 254).astype('uint8')
            pixels[:, :, :640] = 0
            if period == '2022':
                pixels[:, :, 640:1280] = 0
            self.add(period, pixels, nodata=0)
        result = selection.recommend(self.data, use_cache=False)
        candidates = result['reference_selections']['A']['candidates']
        self.assertEqual(result['issues'], [])
        self.assertGreater(candidates['2020']['sampled_coverage'], .98)
        self.assertGreater(candidates['2020']['relative_valid_coverage'], .99)
        self.assertIn('抽样有效覆盖不足', candidates['2022']['excluded_reasons'])
        self.assertNotEqual(result['area_irmad_references']['A'], '2022')

    def test_white_padding_without_nodata_moves_shared_windows_inward(self):
        self.data['periods'] = []
        for period in ('2020', '2021', '2022'):
            pixels = self.rgb.copy()
            pixels[:, :, :500] = 255
            self.add(period, pixels)
        sources = [(p, source) for _, p, source in self.data['periods']]
        with patch.object(selection, '_preview', wraps=selection._preview) as read:
            points, weights, details = selection.content_sampling_plan(sources, self.geometry, 'EPSG:3857')
        self.assertEqual(len(points), 8)
        self.assertLessEqual(read.call_count, 8*len(selection.INWARD_STEPS)*3)
        self.assertTrue(all(d['valid_fraction'] >= .98 for d in details))
        self.assertTrue(any(list(p) != d['original'] for p,d in zip(points, details)))
        self.assertEqual(points, selection.content_sampling_plan(list(reversed(sources)), self.geometry, 'EPSG:3857')[0])
        for _, source in sources:
            result = selection.sample_period(source, self.geometry, 'EPSG:3857', points, weights, probe=FakeProbe())
            self.assertTrue(all(p['coverage'] >= .98 and p['model'] is not None for p in result['patches']))

    def test_exact_white_and_black_fill_invalid_but_bright_texture_remains(self):
        for fill, expected in ((255, 0.), (0, 0.), (250, 1.)):
            path = self.add(str(fill), np.full_like(self.rgb, fill))
            with rasterio.open(path) as ds:
                sample = selection._patch(ds, (1024,1024), self.geometry)
                self.assertEqual(sample['coverage'], expected)
                self.assertEqual(selection._preview(ds, (896,896,1152,1152), self.geometry), expected)

    def test_all_blank_search_is_bounded_and_never_calls_model(self):
        self.data['periods'] = []
        self.add('empty', np.full_like(self.rgb, 255))
        with patch.object(FakeProbe, 'evaluate', side_effect=AssertionError('blank model input')):
            result = selection.recommend(self.data, use_cache=False)
        self.assertTrue(result['issues'])
        report = result['reference_selections']['A']
        self.assertIsNone(report['period'])
        self.assertEqual(len(report['sampling_windows']), 16)
        self.assertTrue(all(d['attempts'] <= len(selection.INWARD_STEPS) for d in report['sampling_windows']))

    def test_each_area_has_its_own_choice_and_order_is_deterministic(self):
        self.data['areas'].append(['B', str(self.boundary)])
        self.add('2030', self.rgb, area='B')
        self.add('2031', np.full_like(self.rgb, 80), area='B')
        self.add('2032', np.full_like(self.rgb, 250), area='B')
        result = selection.recommend(self.data)
        self.assertEqual(result['area_irmad_references'], {'A': '2022', 'B': '2030'})
        shuffled = copy.deepcopy(self.data)
        shuffled['periods'].reverse()
        self.assertEqual(selection.recommend(shuffled)['area_irmad_references'], result['area_irmad_references'])

    def test_unreadable_image_returns_actionable_area_issue(self):
        (self.root / 'A-2020.tif').write_bytes(b'invalid')
        result = selection.recommend(self.data, use_cache=False)
        self.assertTrue(result['issues'][0].startswith('A：自动选择参考期失败'))


    def test_model_change_invalidates_all_cached_trials(self):
        selection.recommend(self.data)
        with patch.object(FakeProbe, 'version', 'changed-weights'), patch.object(selection, '_patch', wraps=selection._patch) as read:
            selection.recommend(self.data)
        self.assertEqual(read.call_count, 24)

    def test_trial_failure_never_falls_back_to_rgb_ranking(self):
        with patch.object(FakeProbe, 'evaluate', side_effect=RuntimeError('model unavailable')):
            result = selection.recommend(self.data)
        self.assertTrue(result['issues'])
        self.assertEqual(result['area_irmad_references'], {})

    def test_graph_length_ignores_duplicates_width_and_segmentation(self):
        from engine.samroad.reference_probe import graph_metrics
        valid = np.ones((512,512), dtype=bool)
        nodes = np.array([[100,100],[100,200],[100,300]])
        a = graph_metrics(nodes, [[0,2]], valid)
        b = graph_metrics(nodes, [[0,1],[1,2],[1,0],[2,1]], valid)
        self.assertEqual(a['length_m'],100.)
        self.assertEqual(a['length_m'],b['length_m'])
        self.assertEqual(a['score'],b['score'])
        self.assertEqual(graph_metrics(nodes,[],valid)['score'],0.)

    def test_graph_length_excludes_nodata_and_window_margin(self):
        from engine.samroad.reference_probe import graph_metrics
        valid = np.ones((512,512), dtype=bool)
        nodes = np.array([[100,0],[100,511]])
        complete = graph_metrics(nodes, [[0,1]], valid)
        self.assertAlmostEqual(complete['length_m'],192.,delta=.5)
        valid[:,128:256] = False
        partial = graph_metrics(nodes, [[0,1]], valid)
        self.assertAlmostEqual(complete['length_m']-partial['length_m'],64.,delta=.5)

    def test_same_ground_window_at_different_source_resolutions(self):
        outputs = []
        for size in (512,1024):
            resolution = 512/size
            yy,xx = np.indices((size,size))
            rgb = np.stack([60+np.floor(xx*resolution/4)%100]*3).astype('uint8')
            path = self.root/f'resolution-{size}.tif'
            with rasterio.open(path,'w',driver='GTiff',height=size,width=size,count=3,
                               dtype='uint8',crs='EPSG:3857',transform=from_origin(0,512,resolution,resolution)) as ds:
                ds.write(rgb)
            with rasterio.open(path) as ds:
                outputs.append(selection._patch(ds,(256,256),box(0,0,512,512)))
        self.assertEqual(outputs[0]['rgb'].shape,(512,512,3))
        self.assertLess(np.abs(outputs[0]['rgb'].astype(float)-outputs[1]['rgb']).mean(),1.)
        self.assertTrue(np.all(outputs[0]['valid']))
        self.assertTrue(np.all(outputs[1]['valid']))

    def test_reused_warp_matches_gdal_for_geographic_and_projected_axis_orders(self):
        from pyproj import Transformer
        original = selection.reproject
        def discovered_warp(*args, **kwargs):
            kwargs.pop('COORDINATE_OPERATION', None)
            return original(*args, **kwargs)
        pixels = self.rgb[:, :512, :512]
        for crs in ('EPSG:3857','EPSG:4326','EPSG:4490','EPSG:4526'):
            with self.subTest(crs=crs):
                x,y = Transformer.from_crs('EPSG:4326',crs,always_xy=True).transform(113.6,23.6)
                resolution = .0000046 if crs in ('EPSG:4326','EPSG:4490') else .5
                affine = from_origin(x-256*resolution,y+256*resolution,resolution,resolution)
                path = self.root/(crs.replace(':','-')+'.tif')
                with rasterio.open(path,'w',driver='GTiff',height=512,width=512,count=3,
                                   dtype='uint8',crs=crs,transform=affine) as ds:
                    ds.write(pixels)
                with rasterio.open(path) as ds:
                    geometry = box(*ds.bounds)
                    accelerated = selection._patch(ds,(x,y),geometry)
                    with patch.object(selection,'reproject',side_effect=discovered_warp):
                        baseline = selection._patch(ds,(x,y),geometry)
                np.testing.assert_array_equal(accelerated['rgb'],baseline['rgb'])
                np.testing.assert_array_equal(accelerated['valid'],baseline['valid'])

    def test_longer_extracted_roads_win(self):
        from engine.samroad.reference_probe import graph_metrics
        valid = np.ones((512,512), dtype=bool)
        nodes = np.array([[100,100],[100,200],[200,100],[200,200]])
        short = graph_metrics(nodes,[[0,1]],valid)
        longer = graph_metrics(nodes,[[0,1],[2,3]],valid)
        self.assertEqual(longer['length_m'],2*short['length_m'])
        self.assertEqual(longer['score'],2*short['score'])


if __name__ == '__main__':
    unittest.main()
