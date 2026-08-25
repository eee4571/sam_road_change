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
    measure_fast_path_widths,
    measure_fast_widths,
    _build_fast_road_geometry,
    _cleanup_road_paths,
    _consistent_relative_score,
    _remove_short_isolated_skeleton_components,
    _relative_hysteresis_mask,
    _trace_skeleton_paths,
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

    def test_very_low_probability_road_needs_no_absolute_floor(self) -> None:
        probability = np.full((100, 100), 0.0003, dtype=np.float32)
        probability[:, 47:53] = 0.0025
        mask, metadata = build_fast_surface_mask(probability)
        self.assertEqual(metadata["raw_high_probability_pixel_count"], 0)
        self.assertGreater(metadata["relative_added_pixel_count"], 0)
        self.assertGreater(float(mask[:, 50].mean()), 0.75)

    def test_relative_weak_component_must_connect_to_strong(self) -> None:
        score = np.zeros((60, 80), dtype=np.float32)
        score[20, 10:60] = 1.0
        score[20, 10] = 1.5
        score[40, 10:60] = 1.0
        mask = _relative_hysteresis_mask(score, min_area=24)
        self.assertEqual(int(mask[20, 10:60].sum()), 50)
        self.assertEqual(int(mask[40, 10:60].sum()), 0)

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

    def test_tiny_fluctuations_in_low_variance_region_are_suppressed(self) -> None:
        rng = np.random.default_rng(11)
        probability = np.clip(
            rng.normal(0.002, 0.0000001, (160, 160)), 0, 1,
        ).astype(np.float32)
        mask, _metadata = build_fast_surface_mask(probability)
        self.assertLess(float(mask.mean()), 0.001)

    def test_single_scale_response_needs_other_scale_support(self) -> None:
        first = np.asarray([[2.0, 2.0]], dtype=np.float32)
        second = np.asarray([[0.2, 0.6]], dtype=np.float32)
        combined = _consistent_relative_score(first, second)
        self.assertEqual(float(combined[0, 0]), 0.0)
        self.assertEqual(float(combined[0, 1]), 2.0)


class FastSkeletonCleanupTests(unittest.TestCase):
    def test_isolated_short_fragment_is_removed(self) -> None:
        skeleton = np.zeros((60, 80), dtype=np.uint8)
        skeleton[20, 10:21] = 1
        cleaned = _remove_short_isolated_skeleton_components(
            skeleton, min_length_px=20.0,
        )
        self.assertEqual(int(cleaned.sum()), 0)

    def test_long_road_and_connected_branch_are_retained(self) -> None:
        skeleton = np.zeros((80, 100), dtype=np.uint8)
        skeleton[50, 10:91] = 1
        skeleton[30:51, 50] = 1
        cleaned = _remove_short_isolated_skeleton_components(
            skeleton, min_length_px=20.0,
        )
        self.assertEqual(int(cleaned[50, 10:91].sum()), 81)
        self.assertEqual(int(cleaned[30:51, 50].sum()), 21)

    def test_straight_degree_two_chain_is_one_complete_path(self) -> None:
        skeleton = np.zeros((60, 80), dtype=np.uint8)
        skeleton[30, 10:71] = 1
        paths = _trace_skeleton_paths(skeleton)
        self.assertEqual(len(paths), 1)
        self.assertAlmostEqual(paths[0].length_px, 60.0, delta=0.01)
        self.assertEqual(paths[0].pixels.shape[0], 61)

    def test_t_junction_is_split_only_at_the_junction(self) -> None:
        skeleton = np.zeros((80, 100), dtype=np.uint8)
        skeleton[50, 10:91] = 1
        skeleton[20:51, 50] = 1
        paths = _trace_skeleton_paths(skeleton)
        self.assertEqual(len(paths), 3)
        self.assertEqual(sorted(round(path.length_px) for path in paths), [30, 40, 40])

    def test_short_weak_endpoint_branch_is_removed_but_strong_branch_stays(self) -> None:
        skeleton = np.zeros((80, 100), dtype=np.uint8)
        skeleton[50, 10:91] = 1
        skeleton[43:51, 50] = 1
        score = np.zeros_like(skeleton, dtype=np.float32)
        score[50, 10:91] = 2.0
        score[43:50, 50] = 0.6
        paths = _trace_skeleton_paths(skeleton, score)
        kept, removed = _cleanup_road_paths(paths)
        self.assertEqual(removed, 1)
        self.assertEqual(len(kept), 2)

        score[43:50, 50] = 1.8
        kept, removed = _cleanup_road_paths(_trace_skeleton_paths(skeleton, score))
        self.assertEqual(removed, 0)
        self.assertEqual(len(kept), 3)

    def test_centerline_is_derived_from_regularized_surface(self) -> None:
        probability = np.full((100, 100), 0.04, dtype=np.float32)
        probability[45:55, 10:47] = 0.24
        probability[45:55, 49:90] = 0.24
        surface, centerline, paths, diagnostics = _build_fast_road_geometry(probability)
        self.assertGreater(int(surface[45:55, 47:50].sum()), 0)
        self.assertTrue(np.all(surface[centerline > 0] > 0))
        self.assertGreater(len(paths), 0)
        self.assertEqual(diagnostics["gap_bridge_added_count"], 0)


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
            centerlines = gpd.read_file(widths / "fast_products.gpkg", layer="centerlines")
            self.assertEqual(len(centerlines), 1)
            self.assertGreater(len(centerlines.geometry.iloc[0].coords), 2)

    def test_distance_transform_is_used_when_normal_probe_fails(self) -> None:
        rows = measure_fast_edge_widths(
            self.nodes, self.edges, self.mask, 1.0,
            sample_function=lambda *_args, **_kwargs: [],
        )
        self.assertEqual(rows[0]["width_source"], "distance_transform_fallback")
        self.assertAlmostEqual(rows[0]["width_units"], 10.0, delta=2.0)

    def test_path_width_is_aggregated_for_one_complete_polyline(self) -> None:
        skeleton = np.zeros_like(self.mask)
        skeleton[10:71, 35] = 1
        paths = _trace_skeleton_paths(skeleton)
        rows = measure_fast_path_widths(paths, self.mask, 1.0)
        self.assertEqual(len(rows), 1)
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
