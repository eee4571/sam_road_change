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
    _bridge_small_supported_gaps,
    _cleanup_road_paths,
    _consistent_relative_score,
    _remove_short_isolated_skeleton_components,
    _relative_hysteresis_mask,
    _trace_skeleton_paths,
)
from engine.samroad.image_resume import required_image_outputs
from engine.samroad.fast_probability import build_fast_enhanced_road_probability


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

    def test_fast_resume_requires_probability_and_native_topology(self) -> None:
        outputs = required_image_outputs(Path("output"), "tile", "fast")
        self.assertEqual(
            [item["role"] for item in outputs],
            [
                "road_probability", "fast_enhanced_probability",
                "fast_probability_boost", "fast_topology",
            ],
        )
        self.assertTrue(str(outputs[0]["path"]).endswith("tile_road.png"))
        self.assertTrue(str(outputs[1]["path"]).endswith("tile_fast_enhanced.png"))
        self.assertTrue(str(outputs[2]["path"]).endswith("tile_fast_boost.png"))
        self.assertTrue(str(outputs[3]["path"]).endswith("tile_fast_topology.npz"))

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
    def test_weak_road_crosses_native_graph_threshold(self) -> None:
        probability = np.full((100, 100), 0.003, dtype=np.float32)
        probability[:, 50] = 0.04
        graph_probability, diagnostics = build_fast_enhanced_road_probability(
            probability, high_threshold=0.36,
        )
        self.assertAlmostEqual(float(graph_probability[:, 50].mean()), 0.50, places=6)
        self.assertGreater(diagnostics["relative_candidate_pixel_count"], 0)

    def test_extremely_low_noise_is_not_boosted(self) -> None:
        probability = np.full((100, 100), 0.0001, dtype=np.float32)
        probability[50, 50] = 0.001
        graph_probability, diagnostics = build_fast_enhanced_road_probability(
            probability, high_threshold=0.36,
        )
        self.assertTrue(np.allclose(graph_probability, probability))
        self.assertEqual(diagnostics["relative_candidate_pixel_count"], 0)


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
        self.assertEqual(removed["spur"], 1)
        self.assertEqual(len(kept), 2)

        score[43:50, 50] = 1.8
        kept, removed = _cleanup_road_paths(_trace_skeleton_paths(skeleton, score))
        self.assertEqual(removed["total"], 0)
        self.assertEqual(len(kept), 3)

    def test_gap_bridge_happens_before_short_fragment_cleanup(self) -> None:
        skeleton = np.zeros((50, 70), dtype=np.uint8)
        skeleton[25, 8:22] = 1
        skeleton[25, 26:42] = 1
        paths = _trace_skeleton_paths(skeleton)
        support = np.zeros_like(skeleton)
        support[25, 8:42] = 1
        bridged, bridge_count = _bridge_small_supported_gaps(skeleton, paths, support)
        self.assertEqual(bridge_count, 1)
        bridged_paths = _trace_skeleton_paths(bridged)
        kept, removed = _cleanup_road_paths(bridged_paths)
        self.assertEqual(removed["isolated"], 0)
        self.assertEqual(len(kept), 1)
        self.assertGreater(kept[0].length_px, 30.0)

    def test_small_weak_loop_is_removed_at_path_level(self) -> None:
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(skeleton, (25, 25), 3, 1, 1)
        score = np.full_like(skeleton, 0.7, dtype=np.float32)
        paths = _trace_skeleton_paths(skeleton, score)
        kept, removed = _cleanup_road_paths(paths)
        self.assertEqual(removed["loop"], 1)
        self.assertEqual(len(kept), 0)

    def test_gap_bridge_rejects_missing_support(self) -> None:
        skeleton = np.zeros((50, 70), dtype=np.uint8)
        skeleton[25, 8:22] = 1
        skeleton[25, 26:42] = 1
        paths = _trace_skeleton_paths(skeleton)
        _bridged, bridge_count = _bridge_small_supported_gaps(
            skeleton, paths, np.zeros_like(skeleton),
        )
        self.assertEqual(bridge_count, 0)

    def test_centerline_is_derived_only_from_native_toponet(self) -> None:
        probability = np.full((100, 100), 0.24, dtype=np.float32)
        nodes = np.asarray([[50, 10], [50, 90]], dtype=np.float32)
        edges = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
        surface, centerline, paths, diagnostics = _build_fast_road_geometry(
            probability, topology_nodes=nodes, topology_edges=edges,
        )
        self.assertGreater(int(surface.sum()), 0)
        self.assertGreater(int(centerline.sum()), 0)
        self.assertEqual(paths, [])
        self.assertEqual(diagnostics["toponet_edge_count"], 1)


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
            cv2.imwrite(str(probabilities / "tile_fast_enhanced.png"), probability)
            graph_dir = probabilities.parent / "graph"
            graph_dir.mkdir()
            np.savez_compressed(
                graph_dir / "tile_fast_topology.npz",
                nodes=np.asarray([[10.0, 35.0], [70.0, 35.0]], dtype=np.float32),
                edges=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
                scores=np.asarray([0.9, 0.9], dtype=np.float32),
            )
            build_fast_surfaces(images, probabilities, surfaces)
            summary = measure_fast_widths(images, surfaces, probabilities, widths)
            exported = export_fast_products(widths, products, image_dir=images)
            for key in ("centerlines", "surfaces", "width_segments", "corridors", "gpkg"):
                self.assertTrue(Path(exported[key]).is_file(), key)
            self.assertTrue((products / "road_overview.png").is_file())
            self.assertTrue((products / "road_width_overview.png").is_file())
            self.assertEqual(Path(exported["previews"]["fusion"]), products / "road_overview.png")
            self.assertEqual(Path(exported["previews"]["width"]), products / "road_width_overview.png")
            self.assertGreater(summary["images"][0]["final_centerline_length"], 0)
            self.assertGreater(summary["images"][0]["measured_edge_count"], 0)
            centerlines = gpd.read_file(widths / "fast_products.gpkg", layer="centerlines")
            self.assertEqual(len(centerlines), 1)
            self.assertEqual(centerlines.iloc[0]["source"], "native_toponet")

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
            result = build_fast_change_from_truth(
                truth_path, root / "result", period_key="area:2021->2022",
            )
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
            change = build_fast_change_from_truth(
                truth_path, root / "change", period_key="area:2021->2022",
            )
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

    def test_pseudo_change_is_reproducible_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth_path = root / "truth.shp"
            truth = gpd.GeoDataFrame(
                {"BHBM": [2] * 40},
                geometry=[box(index * 5, 0, index * 5 + 2, 2) for index in range(40)],
                crs="EPSG:3857",
            )
            truth.to_file(truth_path)
            first = build_fast_change_from_truth(
                truth_path, root / "first", period_key="area:2021->2022",
                change_type="added", global_seed=20260826,
            )
            second = build_fast_change_from_truth(
                truth_path, root / "second", period_key="area:2021->2022",
                change_type="added", global_seed=20260826,
            )
            first_frame = gpd.read_file(first["layers"]["added"])
            second_frame = gpd.read_file(second["layers"]["added"])
            self.assertEqual(len(first_frame), len(second_frame))
            self.assertEqual(
                first_frame.geometry.to_wkb().tolist(),
                second_frame.geometry.to_wkb().tolist(),
            )


if __name__ == "__main__":
    unittest.main()
