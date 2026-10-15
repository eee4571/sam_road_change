"""Pre-review assistance budgets, evaluated from actual rendered road products."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'code/engine/width')]
import geopandas as gpd
from affine import Affine
from shapely.geometry import LineString, box
from engine.continuous_road_geometry import change_surfaces
from engine.fast_gt_reconciliation import _changes
from engine.truth_assistance import limit_truth_assistance
from road_change_detection import evaluate_changes


GRID = (3857, Affine(.5, 0, 0, 0, -.5, 1000))


def roads(count=20):
    rows = []
    for i in range(count):
        axis = LineString([(0, i*30), (100, i*30)])
        rows.append(dict(change_id=f'GT{i}', truth_id=f'GT{i:04}', change_src='GT_ASSISTED',
            change_typ='added', gt_type='added', axis_wkt=axis.wkt, match_axis_wkt=axis.wkt,
            width_bef=0., width_aft=8., width_diff=8., length_m=axis.length,
            geometry=axis.buffer(4., cap_style='flat')))
    candidates = _changes(rows, 3857)
    truth = gpd.GeoDataFrame(dict(BHBM=[2]*count), geometry=candidates.geometry.copy(), crs=3857)
    return candidates, truth


def automatic(frame):
    result = frame.copy()
    result['change_src'] = 'AUTO'
    result['truth_id'] = ''
    return result


class AssistanceTests(unittest.TestCase):
    def assert_actual_score(self, result, truth, audit, **kwargs):
        metrics, _ = evaluate_changes(change_surfaces(result), truth, truth_type_field='BHBM',
                                      position_grid=GRID, **kwargs)
        self.assertAlmostEqual(metrics[0]['change_recall'], audit['candidate_recall'], places=8)

    def test_whole_roads_reach_target_and_keep_unrelated_auto(self):
        gt, truth = roads()
        auto = automatic(gt.iloc[:5])
        false_positive = automatic(gt.iloc[[0]]).copy()
        axis = LineString([(200, 20), (300, 20)])
        false_positive['axis_wkt'] = false_positive['match_axis_wkt'] = axis.wkt
        false_positive.geometry = [axis.buffer(4., cap_style='flat')]
        false_positive['change_id'] = 'unrelated'
        auto = _changes(auto.to_dict('records') + false_positive.to_dict('records'), 3857)
        result, _, audit = limit_truth_assistance(auto, gt, truth, GRID)
        self.assertAlmostEqual(audit['baseline_recall'], .25)
        self.assertAlmostEqual(audit['candidate_recall'], .95)
        self.assertEqual(audit['status'], 'target_reached')
        self.assertIn('unrelated', set(result.change_id))
        self.assertEqual(sum(o['action']=='accepted' for o in audit['objects']), 14)
        self.assert_actual_score(result, truth, audit)
        # Selected roads retain their full axes; no final interval is clipped to a score.
        self.assertTrue((result.length_m == 100.).all())

    def test_baseline_above_ceiling_is_never_deleted_or_capped(self):
        gt, truth = roads()
        auto = automatic(gt)
        result, changes, audit = limit_truth_assistance(auto, gt, truth, GRID)
        self.assertEqual(audit['candidate_recall'], 1.)
        self.assertEqual(audit['status'], 'baseline_at_or_above_target')
        self.assertEqual(changes, [])
        self.assertTrue(result.equals(auto))
        self.assert_actual_score(result, truth, audit)

    def test_large_supplement_is_shortened_without_changing_auto(self):
        gt, truth = roads(10)
        auto = automatic(gt.iloc[:9])
        result, _, audit = limit_truth_assistance(auto, gt, truth, GRID)
        self.assertAlmostEqual(audit['candidate_recall'], .95, delta=1e-6)
        retained = result.loc[result.change_src.eq('AUTO')]
        self.assertTrue(retained.geometry.union_all().equals(auto.geometry.union_all()))
        self.assertEqual(audit['status'], 'target_reached')
        self.assertTrue(any(o['action']=='accepted_shortened' for o in audit['objects']))
        self.assertLess(result.loc[result.change_src.eq('GT_ASSISTED')].length_m.sum(), 100.)
        self.assert_actual_score(result, truth, audit)

    def test_exact_ceiling_proposal_is_shortened(self):
        gt, truth = roads(25)
        result, _, audit = limit_truth_assistance(automatic(gt.iloc[:23]), gt, truth, GRID)
        self.assertAlmostEqual(audit['candidate_recall'], .95, delta=1e-6)
        self.assertEqual(len(result), 24)
        self.assert_actual_score(result, truth, audit)

    def test_multipart_proposal_is_selected_as_a_whole(self):
        gt, truth = roads()
        gt.loc[[17, 18], 'truth_id'] = 'GT_branch_group'
        gt.loc[[17, 18], 'change_id'] = 'multipart_proposal'
        # A two-part object fills 10%, and both spatially disconnected axes survive.
        auto = automatic(gt.iloc[:17])
        result, _, audit = limit_truth_assistance(auto, gt.iloc[17:], truth, GRID)
        self.assertEqual(audit['candidate_recall'], .95)
        self.assertEqual(len(result.loc[result.truth_id.eq('GT_branch_group')]), 2)
        self.assert_actual_score(result, truth, audit)

    def test_partial_source_network_cannot_remove_unseen_truth_on_review(self):
        gt, truth = roads()
        gt['truth_id'] = 'source_network'
        result, _, audit = limit_truth_assistance(automatic(gt.iloc[:0]), gt, truth, GRID)
        self.assertEqual(len(result), 19)
        self.assertFalse(result.truth_object_complete.any())
        self.assert_actual_score(result, truth, audit)
        from engine.reviewed_truth import build_truth_filter
        import tempfile
        public = change_surfaces(result)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)/'audit.gpkg'
            result.to_file(path, layer='changes', driver='GPKG')
            filtered = build_truth_filter(public, {str(i): 'reject' for i in range(len(public))}, path)
        self.assertEqual(filtered['objects'], [])

    def test_curved_supplement_keeps_original_axis_and_audits_missing_range(self):
        gt, _ = roads(1)
        axis = LineString([(0,0),(100,0),(150,40),(250,40)])
        gt['axis_wkt'] = gt['match_axis_wkt'] = axis.wkt
        gt['length_m'] = axis.length
        gt.geometry = [axis.buffer(4., cap_style='flat')]
        truth = gpd.GeoDataFrame(dict(BHBM=[2]), geometry=gt.geometry.copy(), crs=3857)
        original = gt.copy()
        result, _, audit = limit_truth_assistance(automatic(gt.iloc[:0]), gt, truth, GRID)
        self.assertAlmostEqual(audit['candidate_recall'], .95, delta=1e-6)
        self.assertTrue(gt.equals(original))
        self.assertFalse(result.truth_object_complete.any())
        from shapely import from_wkt
        local = from_wkt(result.iloc[0].axis_wkt)
        self.assertLess(local.length, axis.length)
        self.assertLess(local.difference(axis.buffer(1e-8)).length, 1e-6)
        self.assertTrue(any(o['included_axis_fraction'] < 1. for o in audit['objects']
                            if o['action']=='accepted_shortened'))
        self.assert_actual_score(result, truth, audit)

    def test_shortened_candidates_publish_for_review_with_full_provenance(self):
        gt, truth = roads(10)
        result, _, audit = limit_truth_assistance(automatic(gt.iloc[:9]), gt, truth, GRID)
        from engine.fast_gt_reconciliation import _publish_assisted_changes
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            published = _publish_assisted_changes(result, raw, 'before', 'after', require_review=True)
            self.assertTrue(gpd.read_file(published['road_changes']).empty)
            candidates = gpd.read_file(published['review_candidates'])
            provenance = gpd.read_file(Path(raw)/'gt_assisted_audit.gpkg', layer='review_candidates')
            metrics, _ = evaluate_changes(candidates, truth, truth_type_field='BHBM', position_grid=GRID)
            self.assertAlmostEqual(metrics[0]['change_recall'], audit['candidate_recall'], places=8)
            self.assertFalse(provenance.loc[provenance.change_src.eq('GT_ASSISTED')].truth_object_complete.any())

    def test_shortening_uses_original_variable_width_stations(self):
        import json
        from engine.truth_assistance import _shortened_proposal
        from engine.gt_road_geometry import record_polygon
        gt, _ = roads(1)
        profile = json.dumps(dict(axis=gt.iloc[0].axis_wkt, stations=[0., 1.],
                                  widths=[4., 12.], reference_width=8.))
        gt['gt_width_profile'] = profile
        gt.geometry = [record_polygon(gt.iloc[0])]
        shortened = _shortened_proposal(gt, .5)
        self.assertEqual(shortened.iloc[0].gt_width_profile, profile)
        self.assertAlmostEqual(shortened.iloc[0].length_m, 50.)
        # Half the length has widths 4..8 m, so its area is 300, not 400 m².
        self.assertAlmostEqual(shortened.iloc[0].geometry.area, 300., places=5)

    def test_candidate_order_does_not_select_different_roads(self):
        gt, truth = roads()
        auto = automatic(gt.iloc[:5])
        left, _, la = limit_truth_assistance(auto, gt, truth, GRID)
        right, _, ra = limit_truth_assistance(auto, gt.iloc[::-1], truth, GRID)
        self.assertEqual(la, ra)
        self.assertTrue(left.equals(right))

    def test_missing_grid_retains_auto_and_defers_assistance(self):
        gt, truth = roads()
        auto = automatic(gt.iloc[:5])
        result, changes, audit = limit_truth_assistance(auto, gt, truth, None)
        self.assertTrue(result.equals(auto))
        self.assertEqual(changes, [])
        self.assertIsNone(audit['candidate_recall'])
        self.assertEqual(audit['status'], 'missing_image_grid')

    def test_validation_clipping_matches_report(self):
        gt, truth = roads()
        area = gpd.GeoDataFrame(geometry=[box(0, -10, 50, 600)], crs=3857)
        result, _, audit = limit_truth_assistance(automatic(gt.iloc[:5]), gt, truth, GRID,
                                                 validation_area=area)
        self.assert_actual_score(result, truth, audit, validation_area=area)

    def test_real_pixel_grid_is_used_for_offset_already_within_tolerance(self):
        gt, truth = roads()
        auto = automatic(gt)
        from shapely.affinity import translate
        from shapely import from_wkt
        auto['axis_wkt'] = auto.axis_wkt.map(lambda wkt: translate(from_wkt(wkt), yoff=1.5).wkt)
        auto['match_axis_wkt'] = auto['axis_wkt']
        auto.geometry = auto.geometry.translate(yoff=1.5)
        result, _, audit = limit_truth_assistance(auto, gt, truth, GRID)
        self.assertTrue(result.equals(auto))
        self.assertAlmostEqual(audit['candidate_recall'], 1.)
        self.assert_actual_score(result, truth, audit)


if __name__ == '__main__':
    unittest.main()
