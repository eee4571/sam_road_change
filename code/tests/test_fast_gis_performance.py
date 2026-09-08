"""Small deterministic GIS fixtures; never run models or project imagery."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, GeometryCollection, LineString, box

from engine.fast_auto_change import WindowedProbability
from engine.width.road_change_detection import (
    _clip_frame, _matched_object_pairs, _support_geometry, evaluate_changes,
)
from engine.product_cache import signature, read_completed, write_completed


class FastGISPerformanceTests(unittest.TestCase):
    def test_ram_blocks_and_original_window_have_identical_values(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)/'prob.tif'
            data = np.arange(64*64, dtype=np.uint16).reshape(64, 64) % 256
            data[3:8, 4:9] = 999
            with rasterio.open(path, 'w', driver='GTiff', width=64, height=64,
                               count=1, dtype=data.dtype, crs=32650,
                               transform=from_origin(0, 64, 1, 1), nodata=999,
                               tiled=True, blockxsize=16, blockysize=16) as dst:
                dst.write(data, 1)
            samplers = [WindowedProbability(path, 32650),
                        WindowedProbability(path, 32650, ram_limit_bytes=0, cache_limit_bytes=2048)]
            rng = np.random.default_rng(42)
            x, y = rng.uniform(-2, 66, (2, 300))
            try:
                expected = np.full(x.shape, np.nan)
                cols, rows = np.floor(x).astype(int), np.floor(64-y).astype(int)
                mask = (cols >= 0) & (rows >= 0) & (cols < 64) & (rows < 64)
                with rasterio.open(path) as ds:
                    expected[mask] = ds.read(1, masked=True)[rows[mask], cols[mask]].astype(float).filled(np.nan)/samplers[0].divisor
                for sampler in samplers:
                    for _ in range(2):
                        np.testing.assert_array_equal(sampler._values_at(x, y), expected)
                    axes = [LineString([(2, 10), (30, 20)]), LineString([(15, 45), (50, 48)])]
                    batched = sampler.sample_axes(axes, [6., 8.], 3.)
                    scalar = [sampler.sample_axis(axis, 32650, road_width=width, position_tolerance=3.)
                              for axis, width in zip(axes, [6., 8.])]
                    self.assertEqual(batched, scalar)
                self.assertLessEqual(samplers[1]._cache_bytes, 2048)
            finally:
                for sampler in samplers:
                    sampler.close()

    def test_invalid_reprojected_polygon_and_collapsed_components(self):
        invalid = Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)])
        valid = box(20, 0, 30, 10)
        predicted = gpd.GeoDataFrame({'change_typ': ['added', 'added']}, geometry=[valid, valid], crs=32650)
        truth = predicted.copy()
        # Simulate a polygon becoming invalid during _analysis_crs reprojection.
        projected = predicted.copy()
        projected.geometry = [invalid, GeometryCollection([valid, LineString([(50, 0), (60, 0)])])]
        with patch('engine.width.road_change_detection._analysis_crs',
                   return_value=(projected, truth, truth.crs, truth.crs)):
            rows, metadata = evaluate_changes(predicted, truth, evaluation_tolerance=0.)
        self.assertTrue(np.isfinite(rows[0]['iou']))
        self.assertIn('evaluation_clipping_seconds', metadata)
        result = _clip_frame(projected, box(-5, -5, 40, 20))
        self.assertEqual(len(result), 2)
        self.assertTrue(result.is_valid.all())
        self.assertTrue(result.geom_type.isin(['Polygon', 'MultiPolygon']).all())
        self.assertEqual(result.change_typ.tolist(), ['added', 'added'])
        boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=32650)
        self.assertTrue(_clip_frame(boundary, box(10, 0, 20, 10)).empty)

    def test_indexed_matching_preserves_exhaustive_ties(self):
        predicted = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(0, 0, 10, 10),
                                               box(90, 0, 100, 10)], crs=32650)
        truth = gpd.GeoDataFrame(geometry=[box(1, 0, 11, 10), box(1, 0, 11, 10),
                                           box(200, 0, 210, 10)], crs=32650)
        candidates = []
        for i, a in enumerate(predicted.geometry):
            for j, b in enumerate(truth.geometry):
                intersection = a.intersection(b).area
                if intersection > 0:
                    iou = intersection/a.union(b).area
                    if iou >= .1:
                        candidates.append((iou, i, j))
        expected, used_a, used_b = [], set(), set()
        for _, i, j in sorted(candidates, reverse=True):
            if i not in used_a and j not in used_b:
                expected.append((i, j)); used_a.add(i); used_b.add(j)
        self.assertEqual(_matched_object_pairs(predicted, truth, 0., .1), expected)

    def test_product_cache_invalidates_input_and_output_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, target, marker = root/'input', root/'output', root/'cache.json'
            source.write_text('a'); target.write_text('b')
            write_completed(marker, signature([source]), {'count': 1}, [target])
            self.assertEqual(read_completed(marker, signature([source])), {'count': 1})
            target.write_text('changed')
            self.assertIsNone(read_completed(marker, signature([source])))
            write_completed(marker, signature([source]), {'count': 1}, [target])
            source.write_text('new input')
            self.assertIsNone(read_completed(marker, signature([source])))

    def test_unchanged_final_road_reuses_geometry_and_attributes(self):
        from engine.fast_gt_reconciliation import _write_final_period
        from engine.continuous_road_geometry import network_surface
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = gpd.GeoDataFrame({'width_m': [8.]},
                                   geometry=[LineString([(0, 0), (100, 0)])], crs=32650)
            original = {}
            for key in ('centerlines', 'width_segments'):
                path = root/(key+'.shp')
                base.to_file(path)
                original[key] = str(path)
            output = root/'final'; output.mkdir()
            centers = [{**row, '_original_row': i, 'track_id': ''}
                       for i, row in enumerate(base.to_dict('records'))]
            with patch('engine.continuous_road_geometry.network_surface', wraps=network_surface) as build:
                first = _write_final_period(original, base, {}, centers, [], output, base.crs, base.crs)
                before = signature([first[key] for key in original]+[first['surfaces'], first['corridors']])
                second = _write_final_period(original, base, {}, centers, [], output, base.crs, base.crs)
                self.assertEqual(first, second)
                self.assertEqual(before, signature([second[key] for key in original]+[second['surfaces'], second['corridors']]))
                self.assertEqual(build.call_count, 1)
                # Metadata is part of the cache identity too.
                centers[0]['state_src'] = 'auto_retained'
                _write_final_period(original, base, {}, centers, [], output, base.crs, base.crs)
                self.assertEqual(build.call_count, 2)


if __name__ == '__main__':
    unittest.main()
