"""Synthetic review/truth association and real evaluation checks; no models."""
import argparse
from pathlib import Path
import sys
import unittest

import check_change_review

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/engine/width'))
import geopandas as gpd
import pyogrio
from shapely.geometry import box, LineString
from engine.fast_gt_reconciliation import _id
from engine.reviewed_truth import build_truth_filter, apply_truth_filter
from app.change_review import ReviewSession, source_signature
from app.review_publication import read_json, write_json
from road_change_detection import evaluate_changes


class ReviewedTruthTests(unittest.TestCase):
    def setUp(self):
        fixture = check_change_review.ReviewTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.polygon_session()
        self.fixture = fixture
        self.original = gpd.read_file(fixture.source)
        self.truth = self.original.copy()
        self.truth['BHBM'] = 2
        self.truth_path = fixture.root / 'truth.shp'
        pyogrio.write_dataframe(self.truth, self.truth_path, encoding='UTF-8')
        self.truth = gpd.read_file(self.truth_path)
        self.audit_path = fixture.root / 'gt_assisted_audit.gpkg'
        self.audit = self.original.copy()
        self.audit['truth_id'] = self.truth.geometry.map(lambda g: _id(g, 'GT'))
        self.audit['change_src'] = ['GT_ASSISTED', 'AUTO', 'GT_ASSISTED']
        self.audit['gt_type'] = 'added'
        self.audit['axis_wkt'] = fixture.frame.geometry.map(lambda g: g.wkt)
        self.audit['width_bef'] = 0.
        self.audit['width_aft'] = 4.
        self.write_audit()
        manifest = read_json(fixture.manifest)
        manifest['change_results'][0].update(ground_truth_used=True, truth_path=str(self.truth_path))
        for period in manifest['period_results']:
            period['road_probability'] = str(fixture.root / 'before.tif')
        write_json(fixture.manifest, manifest)

    def write_audit(self):
        pyogrio.write_dataframe(self.audit, self.audit_path, layer='changes', driver='GPKG')

    def exclusions(self, decisions):
        return build_truth_filter(self.original, decisions, self.audit_path)

    def test_only_rejected_assisted_objects_are_excluded(self):
        review = self.exclusions({'0': 'reject', '1': 'reject', '2': 'pending'})
        effective, metadata = apply_truth_filter(self.truth, review, 'BHBM')
        self.assertEqual(list(effective.index), [1, 2])
        self.assertEqual(metadata['excluded_feature_count'], 1)
        self.assertEqual(review['objects'][0]['review_ids'], ['0'])
        self.assertEqual(len(self.truth), 3)

    def test_fragment_retained_pending_or_unmapped_protects_whole_truth(self):
        self.audit.loc[2, 'truth_id'] = self.audit.loc[0, 'truth_id']
        self.write_audit()
        for status in ('pending', 'keep'):
            review = self.exclusions({'0': 'reject', '1': 'reject', '2': status})
            self.assertEqual(review['objects'], [])
        review = self.exclusions({'0': 'reject', '1': 'reject', '2': 'reject'})
        self.assertEqual(review['objects'][0]['review_ids'], ['0', '2'])
        self.audit.loc[2, 'geometry'] = box(600000, 3400000, 600010, 3400010)
        self.write_audit()
        self.assertEqual(self.exclusions({'0': 'reject', '1': 'reject', '2': 'reject'})['objects'], [])

    def test_retained_auto_support_protects_a_shared_truth(self):
        self.audit.loc[1, 'truth_id'] = self.audit.loc[0, 'truth_id']
        self.write_audit()
        review = self.exclusions({'0': 'reject', '1': 'keep', '2': 'pending'})
        self.assertEqual(review['objects'], [])

    def test_dissolved_public_polygon_preserves_explicit_truth_associations(self):
        from engine.continuous_road_geometry import change_surfaces
        axes = [LineString([(500030, 3300030), (500030, 3300060)]),
                LineString([(500080, 3300030), (500080, 3300060)]),
                LineString([(500030, 3300060), (500030, 3300090)])]
        self.audit['axis_wkt'] = [axis.wkt for axis in axes]
        self.audit.geometry = [axis.buffer(2) for axis in axes]
        self.audit['truth_id'] = self.audit.geometry.map(lambda geometry: _id(geometry, 'GT'))
        self.write_audit()
        self.original = change_surfaces(self.audit)
        self.assertEqual(len(self.original), 2)
        decisions = {str(i): 'reject' if row.geometry.centroid.x < 500050 else 'keep'
                     for i, row in self.original.iterrows()}
        review = self.exclusions(decisions)
        self.assertEqual({item['truth_id'] for item in review['objects']},
                         {self.audit.loc[0, 'truth_id'], self.audit.loc[2, 'truth_id']})

    def test_boundary_contact_does_not_exclude_neighbour_or_block_rejection(self):
        self.original.loc[1, 'geometry'] = box(500032, 3300030, 500036, 3300060)
        self.audit.loc[0, 'geometry'] = box(500028, 3300030, 500032, 3300060)
        self.write_audit()
        review = self.exclusions({'0': 'reject', '1': 'keep', '2': 'pending'})
        self.assertEqual([item['truth_id'] for item in review['objects']], [self.audit.loc[0, 'truth_id']])

    def test_changed_class_or_geometry_does_not_inherit_an_exclusion(self):
        review = self.exclusions({'0': 'reject', '1': 'keep', '2': 'pending'})
        changed = self.truth.copy()
        changed.loc[0, 'BHBM'] = 4
        effective, metadata = apply_truth_filter(changed, review, 'BHBM')
        self.assertEqual(len(effective), 3)
        self.assertEqual(metadata['excluded_feature_count'], 0)
        changed = self.truth.copy()
        changed.loc[0, 'geometry'] = changed.geometry.iloc[0].buffer(1)
        self.assertEqual(len(apply_truth_filter(changed, review, 'BHBM')[0]), 3)

    def test_geographic_truth_and_custom_type_mapping(self):
        from user_pipeline import apply_truth_value_mapping
        geographic = self.truth.to_crs('EPSG:4326')
        # Generation identifies truth only after projecting to its metric CRS.
        self.audit['truth_id'] = geographic.to_crs(self.audit.crs).geometry.map(lambda g: _id(g, 'GT'))
        self.write_audit()
        geographic['category'] = 'new-road'
        mapped, field = apply_truth_value_mapping(geographic, 'category', {
            'added': 'new-road', 'removed': 'gone-road', 'width_changed': 'width'})
        effective, metadata = apply_truth_filter(mapped, self.exclusions({
            '0': 'reject', '1': 'keep', '2': 'pending'}), field)
        self.assertEqual(list(effective.index), [1, 2])
        self.assertEqual(metadata['excluded_feature_count'], 1)
        self.assertEqual(effective.crs, geographic.crs)

    def test_evaluation_keeps_original_extent_and_unrelated_false_positive(self):
        truth = gpd.GeoDataFrame({'BHBM': [2, 2]}, geometry=[box(0, 0, 4, 20), box(1000, 0, 1004, 20)], crs=self.truth.crs)
        predicted = gpd.GeoDataFrame({'change_typ': ['added', 'added']}, geometry=[truth.geometry.iloc[0], box(990, 0, 994, 20)], crs=truth.crs)
        original_rows, _ = evaluate_changes(predicted, truth, truth_type_field='BHBM', class_mode='three', evaluation_tolerance=0)
        reviewed_rows, _ = evaluate_changes(predicted, truth.iloc[[0]], truth_type_field='BHBM', class_mode='three', evaluation_tolerance=0, truth_extent=truth)
        self.assertEqual(reviewed_rows[0]['object_fp'], 1)
        self.assertEqual(reviewed_rows[0]['validation_area_m2'], original_rows[0]['validation_area_m2'])
        empty_rows, _ = evaluate_changes(predicted.iloc[:0], truth.iloc[:0], truth_type_field='BHBM', class_mode='three', truth_extent=truth)
        self.assertEqual(empty_rows[0]['object_fn'], 0)

    def test_save_reopen_evaluate_and_restore_preserve_original_truth(self):
        fixture = self.fixture
        before_signature = source_signature(self.truth_path, {})
        session = ReviewSession(fixture.manifest)
        session.open_pair(0)
        session.save({'0': 'reject', '1': 'reject', '2': 'keep'})
        manifest = read_json(fixture.manifest)
        entry = manifest['change_results'][0]
        evaluated = read_json(entry['summary'])['evaluation']
        self.assertEqual(evaluated['metadata']['manual_review_truth']['effective_feature_count'], 2)
        self.assertEqual(evaluated['metrics'][0]['object_fn'], 1)  # Auto rejection stays in truth.
        self.assertEqual(evaluated['metrics'][0]['object_tp'], 1)
        self.assertEqual(evaluated['metadata']['evaluation_source'], 'manual_reviewed_changes_vs_reviewed_truth')
        self.assertTrue(evaluated['metadata']['fast_assisted_centerline_metrics'])
        self.assertEqual(source_signature(self.truth_path, {}), before_signature)

        # A later evaluation reads the saved correction rather than reverting it.
        from user_pipeline import _evaluate_existing_changes_impl
        _evaluate_existing_changes_impl(argparse.Namespace(
            pipeline_manifest=str(fixture.manifest), grid='北区', before_period='before', after_period='after',
            truth=str(self.truth_path), validation_area='', truth_type_field='BHBM', evaluation_tolerance=5.,
            defer_publish=True, defer_aggregate=True))
        self.assertEqual(read_json(entry['summary'])['evaluation']['metrics'][0]['object_fn'], 1)
        reopened = ReviewSession(fixture.manifest)
        reopened.open_pair(0)
        reopened.save({'0': 'keep', '1': 'reject', '2': 'keep'})
        restored = read_json(fixture.manifest)['change_results'][0]
        self.assertEqual(restored['manual_review']['truth_filter']['objects'], [])
        metrics = read_json(restored['summary'])['evaluation']['metrics'][0]
        self.assertEqual(metrics['object_tp'], 2)
        self.assertEqual(metrics['object_fn'], 1)
        self.assertEqual(source_signature(self.truth_path, {}), before_signature)

    def test_all_assisted_changes_rejected_can_be_evaluated(self):
        self.audit['change_src'] = 'GT_ASSISTED'
        self.write_audit()
        session = ReviewSession(self.fixture.manifest)
        session.open_pair(0)
        result = session.save({str(i): 'reject' for i in range(3)})
        self.assertEqual(result['evaluation_state'], 'completed')
        entry = read_json(self.fixture.manifest)['change_results'][0]
        evaluation = read_json(entry['summary'])['evaluation']
        self.assertEqual(evaluation['metadata']['manual_review_truth']['effective_feature_count'], 0)
        self.assertEqual(evaluation['metrics'][0]['object_fn'], 0)
        self.assertEqual(evaluation['metrics'][0]['object_fp'], 0)
        self.assertGreater(evaluation['metrics'][0]['validation_area_m2'], 1)


if __name__ == '__main__':
    unittest.main()
