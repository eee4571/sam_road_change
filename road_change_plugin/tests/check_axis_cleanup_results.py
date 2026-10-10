"""Exercise network QA's omitted and split-axis results at the seed boundary."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

import numpy as np
from shapely.geometry import LineString, MultiLineString
from engine.road_geometry import _RegionalRoadSeed
from engine.road_connection_evidence import ConnectionEvidence
from engine.road_axis_cleanup import correct_oscillating_chains
from engine.road_axis_quality import AxisQualityError, repair_network_axes
from engine.road_network_connection import connect_clean_road_seeds


def seed(x, source):
    return _RegionalRoadSeed(np.array([[x, 0.], [x+20., 0.]]), 8., (source,), 'observed')


class AxisCleanupResultTests(unittest.TestCase):
    def test_omitted_and_split_axes_preserve_provenance_without_gap_bridge(self):
        roads = [seed(0., 10), seed(100., 20), seed(200., 30)]
        pieces = [LineString([(100., 0.), (105., 0.)]),
                  LineString([(115., 0.), (120., 0.)])]
        audit = [dict(feature=0, corrected=False, repair_status='skipped'),
                 dict(feature=1, corrected=True, repair_status='fallback')]
        evidence = ConnectionEvidence()
        with patch('engine.road_axis_quality.repair_network_axes',
                   return_value=([None, MultiLineString(pieces), LineString(roads[2].points)], audit)):
            result, count = correct_oscillating_chains(roads, evidence)
        self.assertEqual(count, 1)
        self.assertEqual(len(result), 3)
        self.assertIs(result[2], roads[2])
        for road, geometry in zip(result[:2], pieces):
            np.testing.assert_array_equal(road.points, geometry.coords)
            self.assertEqual((road.width_m, road.source_ids, road.geometry_kind),
                             (8., (20,), 'axis_quality_repaired'))
        self.assertEqual(evidence.axis_quality_audit, audit)

    def test_correction_count_uses_original_indices_after_omission(self):
        roads = [seed(0., 10), seed(100., 20)]
        repaired = LineString(roads[1].points)
        audit = [dict(feature=0, corrected=False), dict(feature=1, corrected=True)]
        candidate = LineString([(100., 0.), (110., .1), (120., 0.)])
        evidence = ConnectionEvidence()
        with patch('engine.road_axis_quality.repair_network_axes', return_value=([None, repaired], audit)), \
             patch('engine.road_axis_cleanup._axis_candidate', return_value=candidate), \
             patch.object(evidence, 'measure', return_value=dict(joint_support=1., unsupported_run_m=0.)):
            result, count = correct_oscillating_chains(roads, evidence)
        self.assertEqual(count, 1)
        self.assertEqual(result[0].source_ids, (20,))
        self.assertEqual(result[0].geometry_kind, 'low_frequency_axis')

    def test_all_omitted_axes_complete_network_without_restoring_input(self):
        evidence = ConnectionEvidence()
        audit = [dict(feature=0, corrected=False, repair_status='skipped')]
        with patch('engine.road_axis_quality.repair_network_axes', return_value=([None], audit)):
            result, stats, _ = connect_clean_road_seeds([seed(0., 10)], evidence=evidence)
        self.assertEqual(result, [])
        self.assertEqual(stats['connection_axis_corrected_count'], 0)
        self.assertEqual(stats['axis_quality_audit'], audit)

    def test_producer_failure_results_are_consumed_as_safe_parts(self):
        axis = LineString([(0., 0.), (10., 10.), (0., 10.), (10., 0.)])
        roads = [_RegionalRoadSeed(np.asarray(axis.coords), 8., (42,))]
        pieces = [(0., 2., LineString([(0., 0.), (1., 1.)])),
                  (axis.length-2., axis.length, LineString([(9., 1.), (10., 0.)]))]
        for safe_parts in ([], pieces):
            with self.subTest(parts=len(safe_parts)), \
                 patch('engine.road_axis_quality.repair_axis', side_effect=AxisQualityError('topology conflict')), \
                 patch('engine.road_axis_quality.conservative_axis_parts', return_value=(safe_parts, [])), \
                 patch('engine.road_axis_quality.repair_network_axes', wraps=repair_network_axes) as repair:
                result, count = correct_oscillating_chains(roads, ConnectionEvidence())
            self.assertEqual(repair.call_count, 1)
            self.assertEqual(len(result), len(safe_parts))
            self.assertEqual(count, int(bool(safe_parts)))
            for road, (_, _, geometry) in zip(result, safe_parts):
                np.testing.assert_allclose(road.points, geometry.coords)
                self.assertEqual(road.source_ids, (42,))


if __name__ == '__main__':
    unittest.main()
