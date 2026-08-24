from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock

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
            self.assertIn("已尝试编码", str(caught.exception))
            self.assertIn(str(listing.resolve()), str(caught.exception))

    def test_common_unicode_and_windows_encodings(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cases = (
                ("utf8.txt", "utf-8", "中文 UTF8.tif"),
                ("utf8_bom.txt", "utf-8-sig", "中文 BOM.tif"),
                ("gbk.txt", "gbk", "中文 GBK.tif"),
                ("gb18030.txt", "gb18030", "扩展字符𠀀.tif"),
                ("utf16.txt", "utf-16", "中文 UTF16.tif"),
                ("utf16le.txt", "utf-16-le", "中文 LE.tif"),
                ("utf16be.txt", "utf-16-be", "中文 BE.tif"),
            )
            for listing_name, encoding, image_name in cases:
                image = root / image_name
                image.touch()
                listing = root / listing_name
                listing.write_bytes((image.name + "\n").encode(encoding))
                self.assertEqual(read_path_list(listing).entries[0].path, image.resolve())

    def test_path_existence_disambiguates_single_byte_decoders(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            image = root / "道路影像.tif"
            image.touch()
            listing = root / "ambiguous.txt"
            listing.write_bytes((image.name + "\n").encode("gbk"))
            parsed = read_path_list(listing)
            self.assertEqual(parsed.entries[0].path, image.resolve())
            self.assertIn(parsed.encoding.casefold(), {"gbk", "gb18030"})
            self.assertNotEqual(parsed.encoding.casefold(), "cp1252")

    def test_system_preferred_encoding_candidate_can_win_by_path_hit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            image = root / "表画像.tif"
            image.touch()
            listing = root / "system.txt"
            listing.write_bytes((image.name + "\n").encode("cp932"))
            with mock.patch("input_catalog.locale.getpreferredencoding", return_value="cp932"):
                parsed = read_path_list(listing)
            self.assertEqual(parsed.entries[0].path, image.resolve())
            self.assertEqual(parsed.encoding.casefold(), "cp932")

    def test_relative_paths_prefer_txt_directory_then_search_roots(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            second = root / "subdir"
            second.mkdir()
            first_image = root / "image1.tif"
            second_image = second / "image2.tif"
            first_image.touch(); second_image.touch()
            listing = root / "relative.txt"
            listing.write_text('"image1.tif"\nsubdir\\image2.tif\n', encoding="utf-8")
            parsed = read_path_list(listing)
            self.assertEqual(
                [entry.path for entry in parsed.entries],
                [first_image.resolve(), second_image.resolve()],
            )

    def test_missing_absolute_paths_use_explicit_root_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            old = root / "old-source"
            new = root / "new-source"
            image = new / "tiles" / "image.tif"
            image.parent.mkdir(parents=True)
            image.touch()
            listing = root / "absolute.txt"
            listing.write_text(str(old / "tiles" / "image.tif"), encoding="utf-8")
            environment = json.dumps({str(old): str(new)}, ensure_ascii=False)

            with mock.patch.dict("os.environ", {"SAMROAD_PATH_RELOCATIONS": environment}):
                parsed = read_path_list(listing)

            self.assertEqual(parsed.entries[0].path, image.resolve())


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
