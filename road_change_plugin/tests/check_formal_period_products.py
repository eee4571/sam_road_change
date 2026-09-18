"""Synthetic formal-period/export/cache contract checks; no models or real data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import geopandas as gpd
from shapely.geometry import LineString, box
from shapely import union_all
import user_pipeline
from engine.formal_road_products import reconstruct_frames, formal_metadata, formal_products_current
from engine.fast_pipeline import export_fast_products, _read_fast_change_layer
from engine.fast_gt_reconciliation import build_fast_temporal_outputs, reconcile_periods


def line(a, b, y=0):
    return LineString([(a,y),(b,y)])


class FormalPeriodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'products'; self.output.mkdir()
        self.width = self.root/'width'; self.width.mkdir()

    def regional(self, axes=None):
        axes = axes or [line(0,50),line(50,100),LineString([(50,0),(50,7)])]
        centers = gpd.GeoDataFrame({'width_m':[8.]*len(axes)},geometry=axes,crs=3857)
        surface = gpd.GeoDataFrame(geometry=[box(0,-4,100,4)],crs=3857)
        frames = dict(centerlines=centers,width_segments=centers,surfaces=surface,corridors=surface)
        for name,frame in frames.items():
            frame.to_file(self.output/'regional_products.gpkg',layer=name,driver='GPKG')
        return frames

    def preview(self, frames, directory, image_dir):
        self.preview_frames = frames
        result = {}
        for key,name in [('fusion','road_overview.png'),('width','road_width_overview.png')]:
            path = directory/name; path.write_bytes(b'synthetic preview marker')
            result[key] = str(path)
        return result

    def export(self):
        with patch('engine.fast_pipeline._write_fast_period_previews',side_effect=self.preview):
            return export_fast_products(self.width,self.output)

    def test_spur_is_removed_from_axes_surface_widths_and_preview_together(self):
        self.regional(); result = self.export()
        axes = gpd.read_file(result['centerlines'])
        widths = gpd.read_file(result['width_segments'])
        surface = union_all(gpd.read_file(result['surfaces']).geometry)
        self.assertEqual(len(axes),1)
        self.assertAlmostEqual(axes.length.sum(),100.)
        self.assertTrue(surface.covers(union_all(axes.geometry)))
        self.assertLess(union_all(widths.geometry).symmetric_difference(union_all(axes.geometry)).length,1e-6)
        self.assertTrue(self.preview_frames['centerlines'].geometry.iloc[0].equals(axes.geometry.iloc[0]))
        self.assertEqual(json.loads(Path(result['surface_quality_audit']).read_text())['summary']['removed_short_spur_count'],1)

    def test_duplicate_collapse_is_published_as_one_axis(self):
        self.regional([line(0,100),line(0,100,1)])
        result = self.export()
        self.assertEqual(len(gpd.read_file(result['centerlines'])),1)
        self.assertAlmostEqual(gpd.read_file(result['centerlines']).length.sum(),100.)

    def test_fast2_reader_gets_formal_layers(self):
        self.regional(); result = self.export()
        for key in ('centerlines','surfaces','width_segments'):
            actual = _read_fast_change_layer(result,key)
            expected = self.preview_frames[key]
            self.assertLess(union_all(actual.geometry).symmetric_difference(union_all(expected.geometry)).length,1e-6)
        self.assertEqual(result['analysis_surfaces'],result['surfaces'])

    def test_real_overview_is_written_during_single_period_export(self):
        self.regional()
        result = export_fast_products(self.width,self.output)
        for name in ('road_overview.png','road_width_overview.png'):
            self.assertTrue((self.output/name).read_bytes().startswith(b'\x89PNG'))
        self.assertTrue(formal_products_current(self.output))

    def test_fast2_pipeline_refuses_legacy_inputs_before_detection(self):
        source = self.root/'old.json'; source.write_text('{}')
        with patch('engine.fast_pipeline.detect_fast_changes',side_effect=AssertionError('legacy input reached Auto')):
            with self.assertRaisesRegex(RuntimeError,'单期正式道路缓存已过期'):
                user_pipeline._run_fast_change_result(source,source,self.root/'changes',
                    before_period='1',after_period='2',position_tolerance=3.,
                    width_change_absolute=2.,width_change_ratio=.2)

    def test_empty_period_has_valid_formal_empty_products(self):
        empty = gpd.GeoDataFrame(geometry=[],crs=3857)
        for name in ('centerlines','surfaces','width_segments','corridors'):
            empty.to_file(self.output/'regional_products.gpkg',layer=name,driver='GPKG')
        result = self.export()
        self.assertTrue(gpd.read_file(result['centerlines']).empty)
        self.assertTrue(gpd.read_file(result['surfaces']).empty)
        self.assertTrue(formal_products_current(self.output))

    def test_cached_export_reuses_all_geometry_and_preview(self):
        self.regional(); result = self.export()
        self.assertTrue(formal_products_current(self.output))
        with patch('engine.formal_road_products.reconstruct_frames',side_effect=AssertionError('duplicate reconstruction')):
            self.assertEqual(export_fast_products(self.width,self.output),result)

    def test_legacy_or_changed_renderer_cannot_reuse_period(self):
        self.regional(); result = self.export()
        result_file = self.root/'period.json'; result_file.write_text(json.dumps(result))
        self.assertTrue(user_pipeline._period_result_ready({'result':str(result_file)}))
        result.pop('formal_road_revision'); result_file.write_text(json.dumps(result))
        self.assertFalse(user_pipeline._period_result_ready({'result':str(result_file)}))
        with patch('engine.formal_road_products.FORMAL_ROAD_REVISION',999):
            self.assertFalse(formal_products_current(self.output))
            self.export()
            self.assertTrue(formal_products_current(self.output))

    def test_source_change_and_missing_preview_invalidate_export(self):
        self.regional(); result = self.export()
        Path(result['previews']['fusion']).unlink()
        self.assertFalse(formal_products_current(self.output))
        self.export()
        with (self.output/'regional_products.gpkg').open('ab') as f: f.write(b'changed')
        self.assertFalse(formal_products_current(self.output))

    def test_only_export_stage_is_stale_with_old_period_products(self):
        self.regional(); result = self.export()
        (self.output/'regional_products.json').write_text('{}')
        (self.output/'fast_export_cache.json').unlink()
        context = dict(centerline=Path(result['centerlines']),surface=Path(result['surfaces']),
                       gpkg=Path(result['gpkg']),execution_profile='fast',image_stems=[])
        self.assertTrue(user_pipeline._period_stage_output_complete('regional',context))
        with patch.object(user_pipeline,'network_products_current',return_value=True):
            self.assertFalse(user_pipeline._period_stage_output_complete('export',context))

    def test_no_events_finalization_does_not_rebuild_or_rewrite_period(self):
        self.regional(); result = self.export()
        manifest = dict(period_results=[dict(result,period='2024',grid='test')],change_results=[])
        with patch('engine.fast_gt_reconciliation._write_final_period',side_effect=AssertionError('unnecessary rebuild')), \
             patch('engine.fast_pipeline._write_fast_period_previews',side_effect=AssertionError('unnecessary preview')), \
             patch('temporal_road_analysis.build_from_manifest',return_value=[]):
            build_fast_temporal_outputs(manifest,self.root/'final')
        self.assertEqual(manifest['period_results'][0]['centerlines'],result['centerlines'])
        self.assertTrue(formal_products_current(self.output))

    def test_finalization_rejects_legacy_first_canonicalization(self):
        self.regional(); result = self.export(); result.pop('formal_road_revision')
        manifest = dict(period_results=[dict(result,period='2024',grid='test')],change_results=[])
        with self.assertRaisesRegex(RuntimeError,'单期正式道路'):
            build_fast_temporal_outputs(manifest,self.root/'final')

    def test_lifecycle_absent_without_observation_needs_no_geometry_rebuild(self):
        self.regional([line(0,100,100)]); result = self.export()
        # An explicit added event only changes the after period. The earlier
        # absent track has no observation to remove; keep its formal roads.
        event = gpd.GeoDataFrame(dict(change_typ=['added'],width_m=[8.],width_bef=[0.],width_aft=[8.],change_id=['c1']),
                                 geometry=[line(0,100)],crs=3857)
        # Use canonical audit axis fields consumed by reconciliation.
        event['axis_wkt'] = event.geometry.to_wkt(); event['match_axis_wkt'] = event.geometry.to_wkt()
        from engine.fast_gt_reconciliation import _write_final_period
        with patch('engine.fast_gt_reconciliation._write_final_period',wraps=_write_final_period) as writer:
            results = reconcile_periods([dict(result,period='1'),dict(result,period='2')],
                [dict(frame=event,before_period='1',after_period='2')],self.root/'reconciled')
        self.assertEqual(writer.call_count,1)
        self.assertEqual(results[0]['centerlines'],result['centerlines'])
        self.assertNotEqual(results[1]['centerlines'],result['centerlines'])

    def test_stable_width_retains_bad_measurement_quality(self):
        axes = gpd.GeoDataFrame(dict(width_m=[8.]),geometry=[line(0,100)],crs=3857)
        widths = gpd.GeoDataFrame(dict(width_m=[8.,.5,8.],quality_grade=['A','C','B'],width_std=[.1,.2,.1]),
                                  geometry=[line(0,40),line(40,60),line(60,100)],crs=3857)
        frames,_,_ = reconstruct_frames(axes,widths)
        middle = frames['width_segments'].loc[frames['width_segments'].quality_grade.eq('C')]
        self.assertFalse(middle.empty)
        self.assertTrue(middle.width_m.eq(8.).all())
        self.assertTrue(middle.measured_w.eq(.5).all())

    def test_unobserved_interval_does_not_inherit_chain_owner_grade(self):
        axes = gpd.GeoDataFrame(dict(width_m=[8.],quality_grade=['A']),geometry=[line(0,100)],crs=3857)
        widths = gpd.GeoDataFrame(dict(width_m=[8.],quality_grade=['A']),geometry=[line(0,40)],crs=3857)
        frames,_,_ = reconstruct_frames(axes,widths)
        unsupported = frames['width_segments'].loc[frames['width_segments'].geometry.centroid.x.gt(50)]
        self.assertTrue(unsupported.quality_grade.isna().all())

    def test_internal_junction_connectors_remain_in_formal_axes(self):
        axes = [line(-100,0),line(0,3),line(3,100),
                LineString([(0,0),(0,-100)]),LineString([(3,0),(3,100)])]
        source = gpd.GeoDataFrame(dict(width_m=[8.]*len(axes)),geometry=axes,crs=3857)
        frames,_,audit = reconstruct_frames(source,source)
        self.assertGreater(audit['summary']['junction_internal_chain_count'],0)
        self.assertTrue(union_all(frames['centerlines'].geometry).buffer(1e-5).covers(axes[1]))
        self.assertTrue(union_all(frames['surfaces'].geometry).buffer(1e-5).covers(union_all(frames['centerlines'].geometry)))


if __name__ == '__main__':
    unittest.main()
