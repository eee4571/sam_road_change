from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from input_catalog import period_order_manifest, read_path_list
import user_pipeline


class PathListEncodingTests(unittest.TestCase):
    def test_gbk_and_utf16_lists_resolve_chinese_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            image = root / "中文影像.tif"
            image.touch()
            for name, encoding in (("ansi.txt", "gbk"), ("utf16.txt", "utf-16")):
                listing = root / name
                listing.write_bytes(str(image).encode(encoding))
                parsed = read_path_list(listing)
                self.assertEqual([entry.path for entry in parsed.entries], [image.resolve()])

    def test_missing_path_reports_txt_line_number_and_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            listing = Path(raw) / "images.txt"
            listing.write_bytes("# comment\n不存在.tif\n".encode("gbk"))
            with self.assertRaisesRegex(FileNotFoundError, "第 2 行") as caught:
                read_path_list(listing)
            self.assertIn("检测编码", str(caught.exception))


class PeriodOrderingTests(unittest.TestCase):
    def test_full_dates_are_validated_and_frozen_chronologically(self) -> None:
        manifest = period_order_manifest(["20260402", "20251231", "20260321"])
        self.assertEqual(manifest["period_order"], ["20251231", "20260321", "20260402"])
        self.assertEqual(
            manifest["change_pairs"],
            [["20251231", "20260321"], ["20260321", "20260402"]],
        )

    def test_invalid_calendar_date_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            period_order_manifest(["20260321", "20261340"])


class PolygonWindowTests(unittest.TestCase):
    def test_l_shape_skips_bounding_box_empty_corner_and_resumes(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Polygon

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_path = root / "l_area.geojson"
            l_shape = Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
            gpd.GeoDataFrame(geometry=[l_shape], crs="EPSG:3857").to_file(
                area_path, driver="GeoJSON",
            )
            source = root / "source.tif"
            with rasterio.open(
                source, "w", driver="GTiff", width=4, height=4, count=1,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 4, 1, 1),
            ) as destination:
                destination.write(np.ones((1, 4, 4), dtype="uint8"))

            first = user_pipeline.normalize_validation_sources(
                {"20260321": source}, area_path, root / "normalized", tile_size=2,
            )
            outputs = user_pipeline.listed_rasters(first["20260321"])
            self.assertEqual(len(outputs), 3)
            mtimes = {path.name: path.stat().st_mtime_ns for path in outputs}

            second = user_pipeline.normalize_validation_sources(
                {"20260321": source}, area_path, root / "normalized", tile_size=2,
            )
            self.assertEqual(first, second)
            self.assertEqual(
                mtimes,
                {path.name: path.stat().st_mtime_ns for path in user_pipeline.listed_rasters(second["20260321"])},
            )

    def test_partial_l_shape_coverage_is_nodata_not_fatal(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Polygon

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_path = root / "l_area.geojson"
            l_shape = Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
            gpd.GeoDataFrame(geometry=[l_shape], crs="EPSG:3857").to_file(
                area_path, driver="GeoJSON",
            )
            source = root / "left_only.tif"
            with rasterio.open(
                source, "w", driver="GTiff", width=2, height=4, count=1,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 4, 1, 1),
            ) as destination:
                destination.write(np.ones((1, 4, 2), dtype="uint8"))

            normalized = user_pipeline.normalize_validation_sources(
                {"20260321": source}, area_path, root / "normalized", tile_size=2,
            )
            self.assertEqual(len(user_pipeline.listed_rasters(normalized["20260321"])), 2)
            marker = user_pipeline.read_json(root / "normalized" / "normalization_complete.json")
            self.assertEqual(marker["coverage"]["20260321"]["missing_pixels"], 4)


if __name__ == "__main__":
    unittest.main()
