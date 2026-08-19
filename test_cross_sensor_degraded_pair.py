from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from dev_tools.cross_sensor_change_test import generate_degraded_pair


class CrossSensorDegradedPairTests(unittest.TestCase):
    def _write_source(self, path: Path):
        values = np.full((3, 128, 160), 110, dtype=np.uint8)
        with rasterio.open(
            path, "w", driver="GTiff", width=160, height=128, count=3,
            dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 128, 1, 1),
        ) as dataset:
            dataset.write(values)

    def _run(self, root: Path, change: str):
        source = root / "source.tif"
        output = root / change
        self._write_source(source)
        with mock.patch.object(
            generate_degraded_pair, "degrade", side_effect=lambda rgb, **_kwargs: rgb
        ):
            result = generate_degraded_pair.main([
                "--input", str(source), "--output-dir", str(output),
                "--change", change,
            ])
        self.assertEqual(result, 0)
        with rasterio.open(output / "A_original.tif") as dataset:
            before = np.moveaxis(dataset.read(), 0, 2)
        with rasterio.open(output / "B_degraded.tif") as dataset:
            after = np.moveaxis(dataset.read(), 0, 2)
        truth = gpd.read_file(output / f"truth_{change}.shp")
        return before, after, truth

    def test_added_exists_only_after_and_truth_remains_polygon(self):
        with tempfile.TemporaryDirectory() as raw:
            before, after, truth = self._run(Path(raw), "added")
        row0, row1 = int(128 * 0.46), int(128 * 0.49)
        col0, col1 = int(160 * 0.18), int(160 * 0.82)
        self.assertEqual(len(np.unique(before[row0:row1, col0:col1], axis=0)), 1)
        self.assertGreater(np.ptp(after[row0:row1, col0:col1].astype(np.int16)), 0)
        self.assertTrue(all(kind in {"Polygon", "MultiPolygon"} for kind in truth.geom_type))

    def test_removed_exists_only_before_and_truth_remains_polygon(self):
        with tempfile.TemporaryDirectory() as raw:
            before, after, truth = self._run(Path(raw), "removed")
        row0, row1 = int(128 * 0.46), int(128 * 0.49)
        col0, col1 = int(160 * 0.18), int(160 * 0.82)
        self.assertGreater(np.ptp(before[row0:row1, col0:col1].astype(np.int16)), 0)
        self.assertEqual(len(np.unique(after[row0:row1, col0:col1], axis=0)), 1)
        self.assertTrue(all(kind in {"Polygon", "MultiPolygon"} for kind in truth.geom_type))


if __name__ == "__main__":
    unittest.main()
