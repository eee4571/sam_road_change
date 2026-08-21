from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from dev_tools.cross_sensor_change_test import run_full_pipeline_pair_test as bench


def write_raster(path: Path, value: int = 20, transform=None) -> None:
    transform = transform or from_origin(100, 200, 0.5, 0.5)
    data = np.full((3, 16, 24), value, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=24, height=16, count=3,
        dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as dataset:
        dataset.write(data)


class CrossSensorFullPipelineBenchTests(unittest.TestCase):
    def test_discovery_prefers_no_change_pair_and_skips_truth_pair(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            changed = root / "newer_added"; changed.mkdir()
            safe = root / "sensor_no_change"; safe.mkdir()
            for directory in (changed, safe):
                write_raster(directory / "A_original.tif")
                write_raster(directory / "B_degraded.tif", 15)
            (changed / "truth_added.shp").touch()
            with patch.object(bench, "SEARCH_ROOTS", (root,)):
                before, after, searched = bench.discover_pair()
            self.assertEqual(before.parent, safe)
            self.assertEqual(after.parent, safe)
            self.assertEqual(searched, [str(root.resolve())])

    def test_pair_validation_and_input_difference(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            before = root / "A_original.tif"
            after = root / "B_degraded.tif"
            write_raster(before, 30)
            write_raster(after, 20)
            metadata = bench.validate_pair(before, after)
            difference = bench.input_difference(before, after)
            self.assertTrue(metadata["shape_equal"])
            self.assertTrue(metadata["crs_equal"])
            self.assertTrue(metadata["transform_equal"])
            self.assertTrue(metadata["bounds_equal"])
            self.assertAlmostEqual(difference["mean_absolute_pixel_difference"], 10.0)
            self.assertAlmostEqual(difference["rmse"], 10.0)

    def test_complete_cache_requires_all_formal_products(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_root = root / "run"; run_root.mkdir()
            payload = {"run_root": str(run_root)}
            for key in ("centerlines", "surfaces", "width_segments", "gpkg"):
                path = root / f"{key}.dat"; path.touch(); payload[key] = str(path)
            result = root / "latest_result.json"
            result.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(bench.result_complete(result))
            Path(payload["width_segments"]).unlink()
            self.assertFalse(bench.result_complete(result))

    def test_observed_surface_never_falls_back_to_product_surface(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_root = root / "run"
            clean = run_root / "width_review" / "A_original_molra_clean_mask.png"
            clean.parent.mkdir(parents=True)
            clean.touch()
            result = {"run_root": str(run_root), "surfaces": str(root / "road_surfaces.shp")}
            source, kind = bench.observed_surface_source(result, "A_original")
            self.assertEqual(source, clean)
            self.assertEqual(kind, "sam_molra_clean_mask")
            clean.unlink()
            source, kind = bench.observed_surface_source(result, "A_original")
            self.assertIsNone(source)
            self.assertIsNone(kind)

    def test_bench_invokes_only_production_pipeline_and_config(self):
        source = inspect.getsource(bench)
        self.assertIn('PIPELINE = ROOT / "user_pipeline.py"', source)
        self.assertIn('DEFAULT_CONFIG = ROOT / "config" / "samroad_inference.yaml"', source)
        self.assertIn('"prepare"', source)
        self.assertIn('"extract"', source)
        self.assertIn('"change"', source)
        self.assertNotIn("import graph_extraction", source)
        self.assertNotIn("road_change_detection", source)

    def test_width_audit_excludes_unmatched_placeholders(self):
        matches = gpd.GeoDataFrame({
            "relation": ["one_to_one", "unmatched", "partial"],
            "before_seg": ["A1", "A2", "A3"],
            "after_seg": ["B1", None, "B3"],
            "before_w": [4.0, 5.0, 0.0],
            "after_w": [4.5, 0.0, 3.0],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        }, crs="EPSG:3857")
        filtered = bench.formal_matched_width_rows(matches)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["before_seg"], "A1")


if __name__ == "__main__":
    unittest.main()
