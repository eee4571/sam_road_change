from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent
WIDTH = ROOT / "engine" / "width"
if str(WIDTH) not in sys.path:
    sys.path.insert(0, str(WIDTH))

import geometry_editor as editor  # noqa: E402


class GeometryEditorCacheTests(unittest.TestCase):
    def _touch(self, path: Path) -> None:
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    def _case(self, root: Path) -> dict[str, Path]:
        roads = root / "roads"
        review = roads / "width_review"
        edited = roads / "centerline_edit"
        products = roads / "products"
        review.mkdir(parents=True)
        edited.mkdir()
        products.mkdir()
        for index, (stem, left) in enumerate((("tile_a", 0.0), ("tile_b", 16.0))):
            image = root / f"{stem}.tif"
            values = np.full((3, 16, 16), 60 + index * 40, dtype=np.uint8)
            with rasterio.open(
                image, "w", driver="GTiff", width=16, height=16, count=3,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(left, 16, 1, 1),
            ) as dataset:
                dataset.write(values)
            (review / f"{stem}_summary.json").write_text(
                json.dumps({"image": str(image)}), encoding="utf-8",
            )
        centerlines = products / "road_centerlines.shp"
        surfaces = products / "road_surfaces.shp"
        line = LineString([(2, 8), (30, 8)])
        gpd.GeoDataFrame(
            {"road_id": [1], "geometry": [line]}, crs="EPSG:3857",
        ).to_file(centerlines)
        gpd.GeoDataFrame(
            {"road_id": [1], "geometry": [line.buffer(2)]}, crs="EPSG:3857",
        ).to_file(surfaces)
        return {
            "roads": roads, "review": review, "edited": edited,
            "centerlines": centerlines, "surfaces": surfaces,
            "image": root / "tile_a.tif",
            "image_b": root / "tile_b.tif",
        }

    def _load(self, case: dict[str, Path], timings: dict | None = None):
        return editor._final_centerline_documents(
            case["review"], case["edited"], case["centerlines"], case["surfaces"],
            timings=timings,
        )[0]

    def test_file_fingerprint_is_stable_and_changes_with_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.bin"
            path.write_bytes(b"same content")
            first = editor.build_file_fingerprint(path)
            self.assertEqual(first, editor.build_file_fingerprint(path))
            self._touch(path)
            second = editor.build_file_fingerprint(path)
            self.assertNotEqual(first, second)
            self.assertEqual(first["size"], second["size"])

    def test_shapefile_sidecar_change_invalidates_only_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case(Path(temporary))
            self._load(case)
            self._touch(case["surfaces"].with_suffix(".dbf"))
            timings = {}
            with mock.patch(
                "geometry_editor._build_global_overview",
                wraps=editor._build_global_overview,
            ) as mosaic, mock.patch(
                "geometry_editor.rasterize", wraps=editor.rasterize,
            ) as surface_rasterize:
                self._load(case, timings)
            self.assertEqual(mosaic.call_count, 0)
            self.assertEqual(surface_rasterize.call_count, 1)
            self.assertEqual(timings["background_cache_used"], 1.0)
            self.assertEqual(timings["surface_cache_used"], 0.0)

    def test_hot_load_skips_mosaic_and_surface_rasterization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case(Path(temporary))
            cold = {}
            first = self._load(case, cold)
            hot = {}
            with mock.patch(
                "geometry_editor._build_global_overview",
                side_effect=AssertionError("hot load rebuilt mosaic"),
            ), mock.patch(
                "geometry_editor.rasterize",
                side_effect=AssertionError("hot load rasterized surface"),
            ):
                second = self._load(case, hot)
            np.testing.assert_array_equal(first.image, second.image)
            np.testing.assert_array_equal(first.mask, second.mask)
            self.assertEqual(hot["background_cache_used"], 1.0)
            self.assertEqual(hot["surface_cache_used"], 1.0)

    def test_image_change_rebuilds_background_but_reuses_same_grid_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case(Path(temporary))
            self._load(case)
            self._touch(case["image"])
            timings = {}
            with mock.patch(
                "geometry_editor._build_global_overview",
                wraps=editor._build_global_overview,
            ) as mosaic, mock.patch(
                "geometry_editor.rasterize", wraps=editor.rasterize,
            ) as surface_rasterize:
                self._load(case, timings)
            self.assertEqual(mosaic.call_count, 1)
            self.assertEqual(surface_rasterize.call_count, 0)
            self.assertEqual(timings["background_cache_used"], 0.0)
            self.assertEqual(timings["surface_cache_used"], 1.0)

    def test_image_grid_change_invalidates_surface_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case(Path(temporary))
            self._load(case)
            with rasterio.open(
                case["image_b"], "w", driver="GTiff", width=20, height=16, count=3,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(16, 16, 1, 1),
            ) as dataset:
                dataset.write(np.full((3, 16, 20), 120, dtype=np.uint8))
            with mock.patch(
                "geometry_editor._build_global_overview",
                wraps=editor._build_global_overview,
            ) as mosaic, mock.patch(
                "geometry_editor.rasterize", wraps=editor.rasterize,
            ) as surface_rasterize:
                self._load(case)
            self.assertEqual(mosaic.call_count, 1)
            self.assertEqual(surface_rasterize.call_count, 1)

    def test_missing_background_and_corrupt_surface_are_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case(Path(temporary))
            self._load(case)
            cache = editor.editor_cache_directory(case["review"])
            (cache / editor.BACKGROUND_CACHE_NAME).unlink()
            with mock.patch(
                "geometry_editor._build_global_overview",
                wraps=editor._build_global_overview,
            ) as mosaic:
                self._load(case)
            self.assertEqual(mosaic.call_count, 1)
            (cache / editor.SURFACE_CACHE_NAME).write_bytes(b"not-a-tiff")
            with mock.patch(
                "geometry_editor._build_global_overview",
                wraps=editor._build_global_overview,
            ) as mosaic, mock.patch(
                "geometry_editor.rasterize", wraps=editor.rasterize,
            ) as surface_rasterize:
                self._load(case)
            self.assertEqual(mosaic.call_count, 0)
            self.assertEqual(surface_rasterize.call_count, 1)

    def test_corrupt_metadata_and_cache_version_are_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case(Path(temporary))
            self._load(case)
            cache = editor.editor_cache_directory(case["review"])
            metadata_path = cache / editor.CACHE_METADATA_NAME
            metadata_path.write_text("{broken", encoding="utf-8")
            self._load(case)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["cache_version"], editor.EDITOR_CACHE_VERSION)
            metadata["cache_version"] = editor.EDITOR_CACHE_VERSION + 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            timings = {}
            self._load(case, timings)
            self.assertEqual(timings["background_cache_used"], 0.0)
            self.assertEqual(timings["surface_cache_used"], 0.0)

    def test_cache_creation_is_atomic_and_does_not_touch_formal_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case(Path(temporary))
            sentinels = [
                case["roads"] / "latest_result.json",
                case["roads"] / "pipeline_result.json",
                case["roads"] / "latest_pipeline.json",
                case["edited"] / "global_manual_widths.json",
                case["edited"] / "edited_manifest.json",
            ]
            for index, path in enumerate(sentinels):
                path.write_text(f"sentinel-{index}", encoding="utf-8")
            protected = list(case["centerlines"].parent.glob("*")) + sentinels
            before = {
                str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in protected if path.is_file()
            }
            self._load(case)
            cache = editor.editor_cache_directory(case["review"])
            self.assertTrue((cache / editor.BACKGROUND_CACHE_NAME).is_file())
            self.assertTrue((cache / editor.SURFACE_CACHE_NAME).is_file())
            self.assertTrue((cache / editor.CACHE_METADATA_NAME).is_file())
            self.assertEqual(list(cache.glob("*.tmp*")), [])
            after = {
                str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in protected if path.is_file()
            }
            self.assertEqual(before, after)

    def test_clear_cache_and_lock_contention_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self._case(Path(temporary))
            self._load(case)
            cache = editor.editor_cache_directory(case["review"])
            with editor.editor_cache_write_lock(cache) as first:
                with editor.editor_cache_write_lock(cache, timeout_seconds=0.0) as second:
                    self.assertTrue(first)
                    self.assertFalse(second)
            self.assertTrue(editor.clear_editor_cache(case["review"]))
            self.assertFalse(cache.exists())
            self.assertFalse(editor.clear_editor_cache(case["review"]))


if __name__ == "__main__":
    unittest.main()
