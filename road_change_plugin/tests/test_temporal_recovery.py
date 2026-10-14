"""Completed old products survive resume, publication and failed aggregation."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import geopandas as gpd
from shapely.geometry import MultiLineString
import user_pipeline as pipeline
from app.temporal_recovery import run_temporal, validate_cached_inputs


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = self.root / '_work/current'
        self.job.mkdir(parents=True)
        self.path = self.job / 'job_state.json'
        geometry = MultiLineString([[(0, 0), (100, 0)], [(150, 0), (200, 0)]])
        periods = []
        for period in ('1', '2'):
            path = self.job / f'road{period}.shp'
            gpd.GeoDataFrame([dict(width_m=4., geometry=geometry)], crs=32650).to_file(path)
            periods.append(dict(grid='g', period=period, centerlines=str(path),
                                formal_road_revision=-1, width_implementation='old-version'))
        change_path = self.job / 'road_changes.shp'
        gpd.GeoDataFrame({'change_typ': []}, geometry=[], crs=32650).to_file(change_path, geometry_type='Polygon')
        self.manifest = dict(execution_profile='fast', status='failed', run_id='current',
            job_root=str(self.job), project_root=str(self.root), output_root=str(self.root / 'published'),
            period_results=periods, final_period_results=copy.deepcopy(periods),
            period_orders={'g': {'period_order': ['1', '2']}}, input_spec={},
            change_results=[dict(grid='g', before_period='1', after_period='2', output=str(self.job),
                layers={'changes': str(change_path)}, fast_finalization_state='completed', evaluation_state='skipped_no_truth')],
            temporal_results=[], temporal_status='failed')
        pipeline.write_json(self.path, self.manifest)

    def run_recovery(self):
        return run_temporal(argparse.Namespace(pipeline_manifest=str(self.path), grid=None), pipeline)

    def test_resume_uses_old_products_without_loading_models_or_republishing_roads(self):
        with patch.object(pipeline, '_resolve_full_run_job_root', return_value=self.job), \
             patch.object(pipeline, 'validation_batch_inputs', side_effect=AssertionError('image inputs consulted')), \
             patch.object(pipeline, 'extract', side_effect=AssertionError('inference')), \
             patch.object(pipeline, 'build_temporal_outputs', side_effect=AssertionError('reconciliation')), \
             patch.object(pipeline.ResultPublisher, 'publish_period', side_effect=AssertionError('period republished')), \
             patch.object(pipeline.ResultPublisher, 'publish_change', side_effect=AssertionError('change republished')):
            result = pipeline.run_all(argparse.Namespace(resume=True, output_root=self.manifest['output_root'], run_id='current'))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['final_period_results'], self.manifest['final_period_results'])
        self.assertTrue(pipeline._temporal_result_ready(result['temporal_results'][0]))

    def test_missing_component_blocks_before_any_mutation(self):
        before = self.path.read_bytes()
        Path(self.manifest['period_results'][0]['centerlines']).with_suffix('.dbf').unlink()
        with self.assertRaisesRegex(ValueError, '单期道路不完整'):
            self.run_recovery()
        with patch.object(pipeline, '_resolve_full_run_job_root', return_value=self.job), \
             patch.object(pipeline, 'validation_batch_inputs', side_effect=AssertionError('must not restart extraction')):
            with self.assertRaisesRegex(ValueError, '单期道路不完整'):
                pipeline.run_all(argparse.Namespace(resume=True, output_root=self.manifest['output_root'], run_id='current'))
        self.assertEqual(self.path.read_bytes(), before)

    def test_incomplete_finalization_cannot_masquerade_as_temporal_only(self):
        self.manifest.pop('final_period_results')
        self.manifest['change_results'][0]['fast_finalization_state'] = 'pending'
        with self.assertRaisesRegex(ValueError, '尚未固化'):
            validate_cached_inputs(self.manifest)

    def test_failed_temporal_preserves_upstream_entries_and_allows_retry(self):
        with patch('temporal_road_analysis.build_from_manifest', side_effect=RuntimeError('write failed')):
            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                self.run_recovery()
        failed = pipeline.read_json(self.path)
        self.assertEqual(failed['period_results'], self.manifest['period_results'])
        self.assertEqual(failed['change_results'], self.manifest['change_results'])
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(self.run_recovery()['status'], 'completed')

    def test_area_recovery_does_not_claim_other_missing_temporal_outputs_are_complete(self):
        for key in ('period_results', 'final_period_results', 'change_results'):
            self.manifest[key] += [{**r, 'grid': 'other'} for r in copy.deepcopy(self.manifest[key])]
        self.manifest['period_orders']['other'] = {'period_order': ['1', '2']}
        pipeline.write_json(self.path, self.manifest)
        result = run_temporal(argparse.Namespace(pipeline_manifest=str(self.path), grid='g'), pipeline)
        self.assertEqual(result['status'], 'completed_with_errors')
        self.assertEqual(result['temporal_status'], 'pending')
        self.assertEqual([t['grid'] for t in result['temporal_results']], ['g'])
        self.assertEqual(result['final_period_results'], self.manifest['final_period_results'])


if __name__ == '__main__':
    unittest.main()
