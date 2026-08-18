from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent
WIDTH = ROOT / "engine" / "width"
if str(WIDTH) not in sys.path:
    sys.path.insert(0, str(WIDTH))

from geometry_editor import (  # noqa: E402
    _final_centerline_documents,
    _global_change_geometry,
    _world_lines_from_document,
    save_global_document,
)
from finalize_review_results import load_graph  # noqa: E402


class GlobalGeometryEditorTests(unittest.TestCase):
    def _write_raster(self, path: Path, left: float) -> None:
        data = np.full((3, 32, 32), 80 + int(left), dtype=np.uint8)
        with rasterio.open(
            path, "w", driver="GTiff", width=32, height=32, count=3,
            dtype="uint8", crs="EPSG:3857", transform=from_origin(left, 32, 1, 1),
        ) as dataset:
            dataset.write(data)

    def test_final_products_open_as_one_global_document_and_save_back_to_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review = root / "review"
            edited = root / "edited"
            review.mkdir()
            edited.mkdir()
            images = []
            for stem, left in (("tile_a", 0.0), ("tile_b", 32.0)):
                image = root / f"{stem}.tif"
                self._write_raster(image, left)
                images.append(image)
                (review / f"{stem}_summary.json").write_text(
                    json.dumps({"image": str(image)}), encoding="utf-8",
                )
            centerlines = root / "centerlines.gpkg"
            surfaces = root / "surfaces.gpkg"
            line = LineString([(5, 16), (20, 16), (32, 16), (45, 16), (59, 16)])
            gpd.GeoDataFrame(
                {"road_id": [1], "geometry": [line]}, crs="EPSG:3857",
            ).to_file(centerlines, layer="centerlines", driver="GPKG")
            gpd.GeoDataFrame(
                {"road_id": [1], "geometry": [line.buffer(3)]}, crs="EPSG:3857",
            ).to_file(surfaces, layer="surfaces", driver="GPKG")

            documents = _final_centerline_documents(review, edited, centerlines, surfaces)
            self.assertEqual(len(documents), 1)
            document = documents[0]
            self.assertTrue(document.global_mode)
            self.assertEqual(document.stem, "全局最终中心线")
            unchanged = _global_change_geometry(document, _world_lines_from_document(document))
            self.assertTrue(unchanged is None or unchanged.is_empty)

            # Move the left endpoint and add one manual width measurement there.
            endpoint = int(np.argmin(document.nodes[:, 1]))
            document.nodes[endpoint, 0] += 2.0
            inverse = ~document.global_transform
            start_col, start_row = inverse * (10, 13)
            end_col, end_row = inverse * (10, 19)
            target_col, target_row = inverse * (10, 16)
            document.manual_widths.append({
                "measurement_id": "MW00001",
                "start_row": start_row, "start_col": start_col,
                "end_row": end_row, "end_col": end_col,
                "target_row": target_row, "target_col": target_col,
                "width_px": abs(end_row - start_row),
            })

            manifest = save_global_document(document, edited, centerlines, surfaces)
            self.assertEqual(manifest["editing_scope"], "period_final_fused_centerlines_global_once")
            self.assertIn("tile_a", manifest["affected_tiles"])
            self.assertNotIn("tile_b", manifest["affected_tiles"])
            self.assertTrue((edited / "global_edited_centerlines.gpkg").is_file())
            for stem in ("tile_a", "tile_b"):
                graph = edited / f"{stem}_edited_graph.p"
                self.assertTrue(graph.is_file())
                nodes, edges = load_graph(graph)
                self.assertGreater(len(nodes), 0)
                self.assertGreater(len(edges), 0)
            widths_a = json.loads((edited / "tile_a_manual_widths.json").read_text(encoding="utf-8"))
            widths_b = json.loads((edited / "tile_b_manual_widths.json").read_text(encoding="utf-8"))
            self.assertEqual(len(widths_a), 1)
            self.assertEqual(widths_a[0]["source"], "global_manual_boundary_measurement")
            self.assertEqual(widths_b, [])


if __name__ == "__main__":
    unittest.main()
