"""Road-segment position must not absorb false branches or censor large offsets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code/engine/width')]
import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiLineString, GeometryCollection, box
from engine.centerline_correspondence import axis_samples, corresponding_distances
from road_change_detection import evaluate_fast_assisted_centerline_metrics, evaluate_changes
import user_pipeline as pipeline


def road_frame(geometry, truth=False):
    return gpd.GeoDataFrame([{'BHBM': 2} if truth else {'change_typ': 'added'}],
                           geometry=[geometry], crs=3857)


def evaluate(predicted, truth):
    return evaluate_fast_assisted_centerline_metrics(
        road_frame(predicted), road_frame(truth, True), truth_type_field='BHBM',
        image_crs=3857, image_transform=from_origin(0, 500, 1., 1.), image_shape=(500, 500))


class CorrespondenceTests(unittest.TestCase):
    def test_large_matched_displacement_is_not_clipped_at_four_pixels(self):
        for offset in (0, 3, 8, 24, 48):
            with self.subTest(offset=offset):
                result = evaluate(LineString([(50, 100+offset), (250, 100+offset)]),
                                  LineString([(50, 100), (250, 100)]))
                self.assertAlmostEqual(result['centerline_mean_offset_px'], offset, places=6)
                self.assertGreater(result['centerline_matched_prediction_fraction'], .99)

    def test_longitudinal_extension_is_unmatched_not_lateral_displacement(self):
        result = evaluate(LineString([(30, 108), (350, 108)]), LineString([(80, 100), (280, 100)]))
        self.assertAlmostEqual(result['centerline_mean_offset_px'], 8., delta=.05)
        self.assertGreater(result['centerline_unmatched_prediction_length_px'], 110.)
        self.assertLess(result['centerline_matched_prediction_fraction'], .65)

    def test_attached_false_branch_still_reduces_area_precision(self):
        truth = box(50, 95, 250, 105)
        prediction = box(50, 103, 250, 113).union(box(145, 103, 155, 210))
        result = evaluate(prediction, truth)
        self.assertAlmostEqual(result['centerline_mean_offset_px'], 8., delta=.6)
        self.assertGreater(result['centerline_unmatched_prediction_length_px'], 80.)
        rows, _ = evaluate_changes(road_frame(prediction), road_frame(truth, True), None, 'BHBM', 5.,
                                  class_mode='three', position_grid=(3857, from_origin(0, 500, 1, 1)))
        base, _ = evaluate_changes(road_frame(box(50, 103, 250, 113)), road_frame(truth, True),
                                   None, 'BHBM', 5., class_mode='three',
                                   position_grid=(3857, from_origin(0, 500, 1, 1)))
        self.assertLess(rows[0]['change_precision'], base[0]['change_precision'])

    def test_multipart_and_collection_keep_all_corresponding_roads(self):
        truth = MultiLineString([[(30, 50), (130, 50)], [(280, 300), (380, 300)]])
        pred = GeometryCollection([LineString([(30, 58), (130, 58)]),
                                   LineString([(280, 312), (380, 312)])])
        result = evaluate(pred, truth)
        self.assertAlmostEqual(result['centerline_mean_offset_px'], 10., delta=.05)
        self.assertGreater(result['predicted_centerline_length_px'], 190.)
        self.assertGreater(result['centerline_matched_prediction_fraction'], .99)

    def test_distant_road_is_unmatched_not_zero_error(self):
        result = evaluate(LineString([(30, 300), (230, 300)]), LineString([(30, 100), (230, 100)]))
        self.assertIsNone(result['centerline_mean_offset_px'])
        self.assertEqual(result['centerline_matched_prediction_fraction'], 0.)

    def test_curved_parallel_road_retains_eight_pixel_error(self):
        angle = np.linspace(.2, 1.3, 200)
        def arc(radius):
            return LineString(np.column_stack((200 + radius*np.cos(angle), 200 + radius*np.sin(angle))))
        result = evaluate(arc(108), arc(100))
        self.assertAlmostEqual(result['centerline_mean_offset_px'], 8., delta=.6)
        self.assertGreater(result['centerline_matched_prediction_fraction'], .7)

    def test_perpendicular_crossing_is_not_corresponding_road(self):
        result = evaluate(LineString([(150, 30), (150, 230)]), LineString([(50, 100), (250, 100)]))
        self.assertIsNone(result['centerline_mean_offset_px'])

    def test_source_labels_and_feature_order_do_not_select_evaluation_geometry(self):
        truth = gpd.GeoDataFrame({'BHBM': [2, 2]}, geometry=[box(30, 90, 150, 100), box(250, 290, 370, 300)], crs=3857)
        prediction = gpd.GeoDataFrame({'change_typ': ['added', 'added'],
                                      'change_src': ['GT_ASSISTED', 'AUTO'], 'truth_id': ['known', '']},
                                     geometry=[box(30, 90, 150, 100), box(250, 298, 370, 308)], crs=3857)
        kwargs = dict(truth_type_field='BHBM', image_crs=3857,
                      image_transform=from_origin(0, 500, 1, 1), image_shape=(500, 500))
        first = evaluate_fast_assisted_centerline_metrics(prediction, truth, **kwargs)
        second = evaluate_fast_assisted_centerline_metrics(prediction.iloc[::-1].drop(columns=['change_src', 'truth_id']),
                                                          truth.iloc[::-1], **kwargs)
        self.assertAlmostEqual(first['centerline_mean_offset_px'], 4., delta=.1)
        self.assertAlmostEqual(first['centerline_mean_offset_px'], second['centerline_mean_offset_px'])

    def test_diagonal_samples_have_euclidean_length_weights(self):
        mask = np.eye(30, dtype=bool)
        samples = axis_samples(mask)
        self.assertAlmostEqual(samples[:, 4].sum(), 29 * np.sqrt(2))
        distances, weights = corresponding_distances(samples, samples)
        self.assertEqual(float(distances.max()), 0.)
        self.assertAlmostEqual(weights.sum(), samples[:, 4].sum())

    def test_closed_loop_keeps_all_edges(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[5, 5:25] = mask[24, 5:25] = True
        mask[5:25, 5] = mask[5:25, 24] = True
        samples = axis_samples(mask)
        self.assertAlmostEqual(samples[:, 4].sum(), 76.)
        distances, weights = corresponding_distances(samples, samples)
        self.assertEqual(float(distances.max()), 0.)
        self.assertAlmostEqual(weights.sum(), 76.)

    def test_ambiguous_parallel_roads_not_resolved_by_row_order(self):
        def horizontal(y):
            return np.column_stack((np.arange(30), np.full(30, y), np.ones(30), np.zeros(30), np.ones(30)))
        prediction = horizontal(10)
        truth = np.concatenate((horizontal(0), horizontal(20)))
        for order in (truth, truth[::-1]):
            distances, _ = corresponding_distances(prediction, order)
            self.assertEqual(len(distances), 0)

    def test_aggregation_rejects_mixed_offset_definitions(self):
        current = dict(evaluate(LineString([(30, 100), (230, 100)]),
                                LineString([(30, 100), (230, 100)])), **{'class': 'all'})
        older = dict(current)
        older.pop('centerline_offset_method')
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            entries = []
            for i, row in enumerate((current, older)):
                path = root / f'{i}.json'
                path.write_text(json.dumps({'evaluation': {'metrics': [row]}}))
                entries.append(dict(summary=str(path), evaluation_state='completed'))
            result = pipeline.aggregate_change_evaluations(dict(change_results=entries), root)['metrics'][0]
            self.assertIsNone(result['centerline_mean_offset_px'])
            self.assertEqual(result['road_centerline_completeness'], 1.)


if __name__ == '__main__':
    unittest.main()
