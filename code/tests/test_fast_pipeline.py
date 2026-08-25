from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from app.task_manager import build_pipeline_command
import user_pipeline
from engine.fast_pipeline import (
    build_fast_change_from_truth,
    build_fast_surface_mask,
    build_fast_surfaces,
    export_fast_products,
    measure_fast_edge_widths,
    measure_fast_widths,
)
from engine.samroad.image_resume import required_image_outputs


class FastCommandTests(unittest.TestCase):
    def test_full_keeps_legacy_cli_and_fast_adds_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            area = root / "area.shp"
            before = root / "2021.txt"
            after = root / "2022.txt"
            truth = root / "truth.shp"
            for path in (area, before, after, truth):
                path.touch()
            common = dict(
                mode="validation", output_root=str(root / "output"), checkpoint="model.pth",
                config="config.yml", device="cpu", pixel_size="1", rescale="off",
                absolute="2", ratio="0.2", tolerance="3", validation_area=str(area),
                periods=[("2021", str(before)), ("2022", str(after))],
                truths=[("2021", "2022", str(truth))], truth_type_field="BHBM",
                runtime_preflight=False,
            )
            full = build_pipeline_command(**common)
            fast = build_pipeline_command(**common, execution_profile="fast")
            self.assertNotIn("--execution-profile", full)
            self.assertEqual(fast[fast.index("--execution-profile") + 1], "fast")

    def test_fast_resume_requires_probability_but_no_graph(self) -> None:
        outputs = required_image_outputs(Path("output"), "tile", "fast")
        self.assertEqual([item["role"] for item in outputs], ["road_probability"])
        self.assertTrue(str(outputs[0]["path"]).endswith("tile_road.png"))

    def test_legacy_full_resume_treats_missing_profile_as_full(self) -> None:
        prior = {
            "pipeline_version": "v", "mode": "validation", "device": "cpu",
            "pixel_size": "1", "rescale": "off", "junction_node_mode": "sparse",
            "validation_area": None, "checkpoint": None, "config": None,
            "grids": {"area": {"2021": {"path": "same"}}}, "truths": {},
            "absolute": "2", "ratio": "0.2", "tolerance": "3",
            "truth_type_field": "BHBM", "evaluation_enabled": True,
        }
        current = copy.deepcopy(prior)
        current["execution_profile"] = "full"
        plan = user_pipeline.dependency_invalidation_plan(prior, current)
        self.assertEqual(plan["periods"], [])
        current["execution_profile"] = "fast"
        self.assertEqual(user_pipeline.dependency_invalidation_plan(prior, current)["periods"], [("area", "2021")])


class FastRelativeTests(unittest.TestCase):
    def test_high_probability_road_is_retained(self) -> None:
        probability = np.full((80, 80), 0.03, dtype=np.float32)
        probability[:, 36:44] = 0.85
        mask, metadata = build_fast_surface_mask(probability)
        self.assertEqual(metadata["raw_high_probability_pixel_count"], 80 * 8)
        self.assertGreater(float(mask[:, 38].mean()), 0.9)

    def test_weak_road_is_recovered_from_local_relative_contrast(self) -> None:
        probability = np.full((100, 100), 0.04, dtype=np.float32)
        probability[:, 47:53] = 0.24
        mask, metadata = build_fast_surface_mask(probability)
        self.assertEqual(metadata["raw_high_probability_pixel_count"], 0)
        self.assertGreater(metadata["relative_added_pixel_count"], 0)
        self.assertGreater(float(mask[:, 50].mean()), 0.75)
        self.assertLess(float(mask.mean()), 0.2)

    def test_weak_road_is_recovered_beside_a_strong_road(self) -> None:
        probability = np.full((120, 120), 0.04, dtype=np.float32)
        probability[:, 20:27] = 0.82
        probability[:, 86:92] = 0.26
        mask, metadata = build_fast_surface_mask(probability)
        self.assertGreater(float(mask[:, 23].mean()), 0.9)
        self.assertGreater(float(mask[:, 89].mean()), 0.75)
        self.assertGreater(metadata["relative_added_pixel_count"], 0)

    def test_background_noise_does_not_become_foreground(self) -> None:
        rng = np.random.default_rng(7)
        probability = np.clip(rng.normal(0.04, 0.006, (100, 100)), 0, 1).astype(np.float32)
        mask, metadata = build_fast_surface_mask(probability)
        self.assertEqual(int(mask.sum()), 0)
        self.assertEqual(metadata["relative_added_pixel_count"], 0)


class FastWidthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((80, 80), dtype=np.uint8)
        self.mask[:, 30:40] = 1
        self.nodes = np.asarray([[10.0, 35.0], [70.0, 35.0]], dtype=np.float32)
        self.edges = np.asarray([[0, 1]], dtype=np.int32)

    def test_sparse_normal_width_is_measured_from_surface(self) -> None:
        rows = measure_fast_edge_widths(self.nodes, self.edges, self.mask, 1.0)
        self.assertEqual(rows[0]["width_source"], "normal_fast")
        self.assertAlmostEqual(rows[0]["width_units"], 10.0, delta=2.0)

    def test_lightweight_period_phases_export_compatible_products(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"; images.mkdir()
            probabilities = root / "probabilities"; probabilities.mkdir()
            surfaces = root / "surfaces"
            widths = root / "widths"
            products = root / "products"
            image_path = images / "tile.tif"
            with rasterio.open(
                image_path, "w", driver="GTiff", width=80, height=80, count=3,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 80, 1, 1),
            ) as dataset:
                dataset.write(np.zeros((3, 80, 80), dtype=np.uint8))
            probability = np.zeros((80, 80), dtype=np.uint8)
            probability[:, 30:40] = 220
            cv2.imwrite(str(probabilities / "tile_road.png"), probability)
            build_fast_surfaces(images, probabilities, surfaces)
            summary = measure_fast_widths(images, surfaces, probabilities, widths)
            exported = export_fast_products(widths, products, image_dir=images)
            for key in ("centerlines", "surfaces", "width_segments", "corridors", "gpkg"):
                self.assertTrue(Path(exported[key]).is_file(), key)
            self.assertTrue((products / "road_overview.png").is_file())
            self.assertTrue((products / "road_width_overview.png").is_file())
            self.assertEqual(Path(exported["previews"]["fusion"]), products / "road_overview.png")
            self.assertEqual(Path(exported["previews"]["width"]), products / "road_width_overview.png")
            self.assertFalse((root / "graphs").exists())
            self.assertGreater(summary["images"][0]["final_centerline_length"], 0)
            self.assertGreater(summary["images"][0]["measured_edge_count"], 0)

    def test_distance_transform_is_used_when_normal_probe_fails(self) -> None:
        rows = measure_fast_edge_widths(
            self.nodes, self.edges, self.mask, 1.0,
            sample_function=lambda *_args, **_kwargs: [],
        )
        self.assertEqual(rows[0]["width_source"], "distance_transform_fallback")
        self.assertAlmostEqual(rows[0]["width_units"], 10.0, delta=2.0)


class FastTruthChangeTests(unittest.TestCase):
    def test_truth_codes_generate_three_semantic_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth_path = root / "truth.shp"
            truth = gpd.GeoDataFrame(
                {"BHBM": [2, 3, 4]},
                geometry=[box(0, 0, 2, 2), box(3, 0, 5, 2), box(6, 0, 8, 2)],
                crs="EPSG:3857",
            )
            truth.to_file(truth_path)
            result = build_fast_change_from_truth(truth_path, root / "result")
            self.assertEqual(len(gpd.read_file(result["layers"]["added"])), 1)
            self.assertEqual(len(gpd.read_file(result["layers"]["width_changed"])), 1)
            self.assertEqual(len(gpd.read_file(result["layers"]["removed"])), 1)
            self.assertTrue(result["ground_truth_derived"])
            self.assertTrue((root / "result" / "change_preview.png").is_file())
            self.assertEqual(Path(result["previews"]["change"]), root / "result" / "change_preview.png")

    def test_fast_truth_result_uses_existing_evaluation_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth_path = root / "truth.shp"
            gpd.GeoDataFrame(
                {"BHBM": [2, 3, 4]},
                geometry=[box(0, 0, 2, 2), box(3, 0, 5, 2), box(6, 0, 8, 2)],
                crs="EPSG:3857",
            ).to_file(truth_path)
            change = build_fast_change_from_truth(truth_path, root / "change")
            job_root = root / "job"; job_root.mkdir()
            manifest_path = job_root / "pipeline_result.json"
            manifest = {
                "execution_profile": "fast", "project_root": str(root),
                "output_root": str(root / "results"), "job_root": str(job_root),
                "change_results": [{
                    "grid": "area", "before_period": "2021", "after_period": "2022",
                    "truth": str(truth_path), "truth_type_field": "BHBM", **change,
                }],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            evaluated = user_pipeline.evaluate_existing_changes(argparse.Namespace(
                pipeline_manifest=str(manifest_path), grid="area",
                before_period="2021", after_period="2022", truth=str(truth_path),
                validation_area="", truth_type_field="BHBM", truth_added_value="",
                truth_width_changed_value="", truth_removed_value="", evaluation_tolerance=5.0,
            ))
            self.assertTrue(Path(evaluated["metrics"]).is_file())
            self.assertIn("evaluation", json.loads(Path(change["summary"]).read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
