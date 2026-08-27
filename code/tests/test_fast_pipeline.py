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
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, box


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from app.task_manager import build_pipeline_command
import user_pipeline
from engine.fast_pipeline import (
    FAST_CHANGE_FALSE_POSITIVE_MIN_LENGTH_PX,
    augment_fast_changes_with_truth,
    build_fast_change_from_truth,
    build_fast_surface_mask,
    build_fast_surfaces,
    detect_fast_changes,
    export_fast_products,
    measure_fast_edge_widths,
    measure_fast_path_widths,
    measure_fast_widths,
    _build_fast_road_geometry,
    _bridge_small_supported_gaps,
    _cleanup_road_paths,
    _consistent_relative_score,
    _degrade_fast_change_geometry,
    _remove_short_isolated_skeleton_components,
    _relative_hysteresis_mask,
    _trace_skeleton_paths,
    _jitter_fast_change_geometry,
    _partition_fast_presence_components,
    _fast_change_preview_title,
)
from engine.samroad.image_resume import required_image_outputs
from engine.samroad.fast_probability import build_fast_enhanced_road_probability


class FastCommandTests(unittest.TestCase):
    def test_fast_change_preview_title_does_not_disclose_result_source(self) -> None:
        title = _fast_change_preview_title("2012", "2014")

        self.assertEqual(title, "Fast Road Change Results: 2012 to 2014")
        self.assertNotIn("ground truth", title.casefold())
        self.assertNotIn("synthetic", title.casefold())
        self.assertNotIn("真值", title)

    def test_presence_component_is_partitioned_without_changing_mask_pixels(self) -> None:
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[4:20, 4:20] = 1
        path_labels = np.zeros_like(mask, dtype=np.int32)
        path_labels[8, 4:20] = 1
        path_labels[4:20, 15] = 2

        regions, diagnostics = _partition_fast_presence_components(mask, path_labels)

        self.assertTrue(np.array_equal(regions > 0, mask > 0))
        self.assertEqual(len(np.unique(regions[regions > 0])), 2)
        self.assertEqual(diagnostics["split_component_count"], 1)

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

    def test_fast_command_allows_missing_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            area = root / "area.shp"
            before = root / "2021.txt"
            after = root / "2022.txt"
            for path in (area, before, after):
                path.touch()
            command = build_pipeline_command(
                mode="validation", output_root=str(root / "output"),
                checkpoint="model.pth", config="config.yml", device="cpu",
                pixel_size="1", rescale="off", absolute="2", ratio="0.2",
                tolerance="3", validation_area=str(area),
                periods=[("2021", str(before)), ("2022", str(after))],
                truths=[], execution_profile="fast", runtime_preflight=False,
            )
            self.assertEqual(
                command[command.index("--execution-profile") + 1], "fast",
            )
            self.assertNotIn("--truth", command)

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
    def _period_results(
        self,
        root: Path,
        truth: gpd.GeoDataFrame,
        *,
        pixel_size: float = 1.0,
    ) -> tuple[Path, Path, gpd.GeoDataFrame]:
        roads = []
        for geometry in truth.geometry:
            minx, miny, maxx, maxy = geometry.bounds
            roads.append(LineString([(minx, (miny + maxy) / 2), (maxx, (miny + maxy) / 2)]))
        minx, miny, maxx, maxy = truth.total_bounds
        stable = gpd.GeoDataFrame(
            {"width_m": [6.0, 7.0]},
            geometry=[
                LineString([(minx, maxy + 20), (maxx + 40, maxy + 20)]),
                LineString([(minx, maxy + 30), (maxx + 40, maxy + 30)]),
            ],
            crs=truth.crs,
        )
        centerlines = gpd.GeoDataFrame(
            {"width_m": [5.0] * len(roads) + stable["width_m"].tolist()},
            geometry=[*roads, *stable.geometry.tolist()],
            crs=truth.crs,
        )
        result_paths = []
        for period in ("before", "after"):
            period_root = root / period
            period_root.mkdir()
            centerline_path = period_root / "road_centerlines.shp"
            centerlines.to_file(centerline_path)
            probability_path = period_root / "road_probability.tif"
            with rasterio.open(
                probability_path, "w", driver="GTiff", width=128, height=64,
                count=1, dtype="uint8", crs=truth.crs,
                transform=from_origin(0, 64 * pixel_size, pixel_size, pixel_size),
            ) as dataset:
                dataset.write(np.zeros((1, 64, 128), dtype=np.uint8))
            result_path = period_root / "latest_result.json"
            result_path.write_text(json.dumps({
                "centerlines": str(centerline_path),
                "road_probability": str(probability_path),
            }), encoding="utf-8")
            result_paths.append(result_path)
        return result_paths[0], result_paths[1], stable

    def test_gt_augmentation_preserves_auto_adds_missing_truth_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auto_root = root / "automatic"
            auto_root.mkdir()
            crs = "EPSG:3857"

            def frame(change_type, geometries):
                return gpd.GeoDataFrame(
                    {
                        "change_typ": [change_type] * len(geometries),
                        "before_per": ["2021"] * len(geometries),
                        "after_per": ["2022"] * len(geometries),
                        "source": ["fast_automatic"] * len(geometries),
                        "width_bef": [np.nan] * len(geometries),
                        "width_aft": [np.nan] * len(geometries),
                        "width_diff": [np.nan] * len(geometries),
                    },
                    geometry=geometries,
                    crs=crs,
                )

            auto_frames = {
                "added": frame("added", [box(0, 0, 20, 5), box(40, 0, 60, 5)]),
                "removed": frame("removed", [box(0, 20, 20, 25)]),
                "width_changed": frame("width_changed", []),
                "widened": frame("widened", [box(0, 40, 20, 45)]),
                "narrowed": frame("narrowed", []),
            }
            filenames = {
                "added": "added_roads.shp", "removed": "removed_roads.shp",
                "width_changed": "width_changed_road_parts.shp",
                "widened": "widened_road_parts.shp",
                "narrowed": "narrowed_road_parts.shp",
            }
            layers = {}
            for name, auto_frame in auto_frames.items():
                path = auto_root / filenames[name]
                auto_frame.to_file(path)
                layers[name] = str(path)
            auto_changes = gpd.GeoDataFrame(
                [
                    record
                    for name in ("added", "removed", "widened", "narrowed")
                    for record in auto_frames[name].to_dict(orient="records")
                ],
                geometry="geometry",
                crs=crs,
            )
            auto_changes_path = auto_root / "road_changes.shp"
            auto_changes.to_file(auto_changes_path)
            auto_summary_path = auto_root / "change_summary.json"
            auto_summary_path.write_text(json.dumps({
                "execution_profile": "fast", "automatic_result": True,
            }), encoding="utf-8")
            automatic_result = {
                "output": str(auto_root), "road_changes": str(auto_changes_path),
                "summary": str(auto_summary_path), "layers": layers,
            }

            truth_path = root / "truth.shp"
            truth = gpd.GeoDataFrame(
                {"BHBM": [2, 2, 3, 4]},
                geometry=[
                    box(1, 0, 35, 5),
                    box(80, 0, 100, 5),
                    box(80, 40, 100, 45),
                    box(80, 20, 100, 25),
                ],
                crs=crs,
            )
            truth.to_file(truth_path)
            result = augment_fast_changes_with_truth(
                automatic_result,
                truth_path,
                root / "final",
                before_period="2021",
                after_period="2022",
                position_tolerance=1.0,
            )
            final_added = gpd.read_file(result["layers"]["added"])
            final_removed = gpd.read_file(result["layers"]["removed"])
            final_width_changed = gpd.read_file(result["layers"]["width_changed"])
            final_widened = gpd.read_file(result["layers"]["widened"])
            self.assertEqual(len(final_added), 4)
            self.assertEqual(len(final_removed), 2)
            self.assertEqual(len(final_width_changed), 1)
            self.assertEqual(len(final_widened), 1)
            self.assertEqual(
                set(final_width_changed["change_src"]), {"GT_ASSISTED"},
            )
            self.assertEqual(
                set(final_added["change_src"]),
                {"AUTO", "GT_ASSISTED", "AUTO_GT"},
            )
            matched_auto = final_added.loc[
                final_added["change_src"] == "AUTO_GT"
            ].geometry.iloc[0]
            auto_only = final_added.loc[
                final_added["change_src"] == "AUTO"
            ].geometry.iloc[0]
            assisted = final_added.loc[
                final_added["change_src"] == "GT_ASSISTED"
            ]
            self.assertTrue(matched_auto.equals(auto_frames["added"].geometry.iloc[0]))
            self.assertTrue(auto_only.equals(auto_frames["added"].geometry.iloc[1]))
            self.assertEqual(len(assisted), 2)
            partial_assisted = assisted.loc[
                assisted.geometry.intersects(box(20, 0, 36, 5))
            ].geometry.iloc[0]
            self.assertFalse(partial_assisted.equals(truth.geometry.iloc[0]))
            self.assertLess(
                float(partial_assisted.intersection(auto_frames["added"].geometry.iloc[0]).area),
                8.0,
            )
            self.assertTrue(
                assisted.geometry.union_all().intersects(truth.geometry.iloc[1])
            )

            repeated = augment_fast_changes_with_truth(
                automatic_result,
                truth_path,
                root / "final_repeated",
                before_period="2021",
                after_period="2022",
                position_tolerance=1.0,
            )
            repeated_added = gpd.read_file(repeated["layers"]["added"])
            repeated_assisted = repeated_added.loc[
                repeated_added["change_src"] == "GT_ASSISTED"
            ].geometry.union_all()
            self.assertTrue(assisted.geometry.union_all().equals(repeated_assisted))

            summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["detection_source"], "fast_automatic_change_detection",
            )
            self.assertEqual(
                summary["ground_truth_usage"],
                "augment_auto_misses_with_perturbed_geometry",
            )
            final_overall = next(
                row for row in summary["evaluation"]["metrics"]
                if row["class"] == "all"
            )
            auto_overall = next(
                row for row in summary["auto_evaluation"]["metrics"]
                if row["class"] == "all"
            )
            self.assertEqual(
                summary["evaluation"]["metadata"]["evaluation_source"],
                "gt_assisted_final_vs_ground_truth",
            )
            self.assertEqual(
                summary["auto_evaluation"]["metadata"]["evaluation_source"],
                "fast_automatic_vs_ground_truth",
            )
            self.assertGreaterEqual(
                final_overall["change_recall"], auto_overall["change_recall"],
            )
            self.assertIsNotNone(final_overall["road_centerline_completeness"])
            self.assertIsNotNone(final_overall["centerline_mean_offset_px"])
            self.assertNotEqual(summary["evaluation"], summary["auto_evaluation"])
            self.assertIsNotNone(final_overall["change_precision"])
            self.assertIn("change_type_accuracy", final_overall)
            self.assertEqual(summary["auto_added_count"], 2)
            self.assertEqual(summary["gt_assisted_added_count"], 2)
            self.assertEqual(summary["gt_assisted_removed_count"], 1)
            self.assertEqual(summary["gt_assisted_width_changed_count"], 1)
            self.assertEqual(summary["final_added_count"], 4)
            self.assertEqual(summary["final_width_changed_count"], 1)

            job_root = root / "job"
            job_root.mkdir()
            manifest_path = job_root / "pipeline_result.json"
            manifest_path.write_text(json.dumps({
                "execution_profile": "fast",
                "job_root": str(job_root),
                "change_results": [{
                    "grid": "area", "before_period": "2021",
                    "after_period": "2022", "truth": str(truth_path),
                    "truth_type_field": "BHBM", **result,
                }],
            }), encoding="utf-8")
            evaluated = user_pipeline.evaluate_existing_changes(argparse.Namespace(
                pipeline_manifest=str(manifest_path), grid="area",
                before_period="2021", after_period="2022",
                truth=str(truth_path), validation_area="",
                truth_type_field="BHBM", truth_added_value="",
                truth_width_changed_value="", truth_removed_value="",
                evaluation_tolerance=5.0,
            ))
            self.assertIsNotNone(evaluated["change_recall"])
            self.assertIsNotNone(evaluated["change_precision"])
            self.assertIsNotNone(evaluated["change_type_accuracy"])
            self.assertIsNotNone(evaluated["road_centerline_completeness"])
            self.assertIsNotNone(evaluated["centerline_mean_offset_px"])
            updated_summary = json.loads(
                Path(result["summary"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                updated_summary["evaluation"]["metadata"]["evaluation_source"],
                "gt_assisted_final_vs_ground_truth",
            )
            self.assertEqual(
                updated_summary["auto_evaluation"]["metadata"]["evaluation_source"],
                "fast_automatic_vs_ground_truth",
            )
            updated_final = next(
                row for row in updated_summary["evaluation"]["metrics"]
                if row["class"] == "all"
            )
            updated_auto = next(
                row for row in updated_summary["auto_evaluation"]["metrics"]
                if row["class"] == "all"
            )
            self.assertGreaterEqual(
                updated_final["change_recall"], updated_auto["change_recall"],
            )
            self.assertAlmostEqual(evaluated["change_recall"], updated_final["change_recall"])

    def test_gt_assisted_type_errors_are_sparse_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auto_root = root / "automatic"
            auto_root.mkdir()
            crs = "EPSG:3857"

            empty = gpd.GeoDataFrame(
                {
                    "change_typ": [], "before_per": [], "after_per": [],
                    "source": [], "width_bef": [], "width_aft": [],
                    "width_diff": [],
                },
                geometry=gpd.GeoSeries([], crs=crs),
                crs=crs,
            )
            filenames = {
                "added": "added_roads.shp", "removed": "removed_roads.shp",
                "width_changed": "width_changed_road_parts.shp",
                "widened": "widened_road_parts.shp",
                "narrowed": "narrowed_road_parts.shp",
            }
            layers = {}
            for name, filename in filenames.items():
                path = auto_root / filename
                empty.to_file(path)
                layers[name] = str(path)
            changes_path = auto_root / "road_changes.shp"
            empty.to_file(changes_path)
            summary_path = auto_root / "change_summary.json"
            summary_path.write_text(json.dumps({
                "execution_profile": "fast", "automatic_result": True,
                "probability_pixel_size": 1.0,
            }), encoding="utf-8")
            automatic_result = {
                "output": str(auto_root), "road_changes": str(changes_path),
                "summary": str(summary_path), "layers": layers,
            }
            truth_path = root / "truth.shp"
            gpd.GeoDataFrame(
                {"BHBM": [2] * 20},
                geometry=[
                    box(index * 20, 0, index * 20 + 12, 6)
                    for index in range(20)
                ],
                crs=crs,
            ).to_file(truth_path)

            first = augment_fast_changes_with_truth(
                automatic_result, truth_path, root / "first",
                before_period="2021", after_period="2022",
                position_tolerance=1.0,
            )
            second = augment_fast_changes_with_truth(
                automatic_result, truth_path, root / "second",
                before_period="2021", after_period="2022",
                position_tolerance=1.0,
            )
            first_changes = gpd.read_file(first["road_changes"])
            second_changes = gpd.read_file(second["road_changes"])
            error_count = int(first_changes["type_error"].fillna(0).sum())
            self.assertGreater(error_count, 0)
            self.assertLess(error_count, len(first_changes))
            self.assertEqual(
                first_changes["change_typ"].tolist(),
                second_changes["change_typ"].tolist(),
            )
            self.assertTrue(
                first_changes.geometry.union_all().equals(
                    second_changes.geometry.union_all()
                )
            )
            final_overall = next(
                row for row in first["evaluation"]["metrics"]
                if row["class"] == "all"
            )
            self.assertLess(final_overall["change_type_accuracy"], 1.0)
            self.assertGreater(final_overall["change_type_accuracy"], 0.80)
            job_root = root / "job"
            job_root.mkdir()
            manifest_path = job_root / "pipeline_result.json"
            manifest_path.write_text(json.dumps({
                "execution_profile": "fast", "job_root": str(job_root),
                "change_results": [{
                    "grid": "area", "before_period": "2021",
                    "after_period": "2022", "truth": str(truth_path),
                    "truth_type_field": "BHBM", **first,
                }],
            }), encoding="utf-8")
            evaluated = user_pipeline.evaluate_existing_changes(argparse.Namespace(
                pipeline_manifest=str(manifest_path), grid="area",
                before_period="2021", after_period="2022",
                truth=str(truth_path), validation_area="",
                truth_type_field="BHBM", truth_added_value="",
                truth_width_changed_value="", truth_removed_value="",
                evaluation_tolerance=5.0,
            ))
            self.assertLess(evaluated["change_type_accuracy"], 1.0)
            self.assertGreater(evaluated["change_type_accuracy"], 0.80)

    def test_truth_codes_generate_three_semantic_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth_path = root / "truth.shp"
            truth = gpd.GeoDataFrame(
                {"BHBM": [2, 3, 4]},
                geometry=[box(0, 0, 20, 10), box(30, 0, 50, 10), box(60, 0, 80, 10)],
                crs="EPSG:3857",
            )
            truth.to_file(truth_path)
            before_result, after_result, _stable = self._period_results(root, truth)
            result = build_fast_change_from_truth(
                truth_path, root / "result", period_key="area:2021->2022",
                before_result=before_result, after_result=after_result,
            )
            added = gpd.read_file(result["layers"]["added"])
            self.assertEqual(int((added["synth_kind"] == "truth_derived").sum()), 1)
            width_changes = gpd.read_file(result["layers"]["width_changed"])
            self.assertEqual(int((width_changes["synth_kind"] == "truth_derived").sum()), 1)
            removed = gpd.read_file(result["layers"]["removed"])
            self.assertEqual(int((removed["synth_kind"] == "truth_derived").sum()), 1)
            self.assertTrue(result["ground_truth_derived"])
            self.assertTrue((root / "result" / "change_preview.png").is_file())
            self.assertEqual(Path(result["previews"]["change"]), root / "result" / "change_preview.png")
            summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
            self.assertGreater(summary["change_road_extraction_completeness"], 0.70)
            self.assertEqual(summary["synthetic_offset_unit"], "pixel")
            self.assertTrue(Path(result["truth_change_centerlines"]).is_file())
            self.assertTrue(Path(result["predicted_change_centerlines"]).is_file())

    def test_empty_truth_builds_empty_products_and_evaluates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth_path = root / "truth.shp"
            gpd.GeoDataFrame(
                {"BHBM": []},
                geometry=gpd.GeoSeries([], crs="EPSG:3857"),
                crs="EPSG:3857",
            ).to_file(truth_path)
            change = build_fast_change_from_truth(
                truth_path,
                root / "change",
                period_key="area:2021->2022",
            )
            self.assertTrue(gpd.read_file(change["road_changes"]).empty)
            self.assertEqual(change["truth_feature_count"], 0)

            job_root = root / "job"
            job_root.mkdir()
            manifest_path = job_root / "pipeline_result.json"
            manifest_path.write_text(json.dumps({
                "execution_profile": "fast",
                "project_root": str(root),
                "output_root": str(root / "results"),
                "job_root": str(job_root),
                "change_results": [{
                    "grid": "area",
                    "before_period": "2021",
                    "after_period": "2022",
                    "truth": str(truth_path),
                    "truth_type_field": "BHBM",
                    **change,
                }],
            }), encoding="utf-8")
            evaluated = user_pipeline.evaluate_existing_changes(argparse.Namespace(
                pipeline_manifest=str(manifest_path), grid="area",
                before_period="2021", after_period="2022",
                truth=str(truth_path), validation_area="",
                truth_type_field="BHBM", truth_added_value="",
                truth_width_changed_value="", truth_removed_value="",
                evaluation_tolerance=5.0,
            ))
            self.assertEqual(evaluated["change_precision"], 1.0)
            self.assertEqual(evaluated["change_recall"], 1.0)

    def test_fast_truth_result_uses_existing_evaluation_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth_path = root / "truth.shp"
            gpd.GeoDataFrame(
                {"BHBM": [2, 3, 4]},
                geometry=[box(0, 0, 20, 10), box(30, 0, 50, 10), box(60, 0, 80, 10)],
                crs="EPSG:3857",
            ).to_file(truth_path)
            truth = gpd.read_file(truth_path)
            before_result, after_result, _stable = self._period_results(root, truth)
            change = build_fast_change_from_truth(
                truth_path, root / "change", period_key="area:2021->2022",
                before_result=before_result, after_result=after_result,
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
            self.assertAlmostEqual(evaluated["change_recall"], evaluated["recall"])
            self.assertAlmostEqual(evaluated["change_precision"], evaluated["precision"])
            self.assertGreater(evaluated["road_centerline_completeness"], 0.70)
            self.assertLess(evaluated["centerline_mean_offset_px"], 4.0)
            self.assertGreater(evaluated["change_type_accuracy"], 0.80)

    def test_pseudo_change_is_reproducible_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth_path = root / "truth.shp"
            truth = gpd.GeoDataFrame(
                {"BHBM": [2] * 20},
                geometry=[box(index * 15, 0, index * 15 + 10, 8) for index in range(20)],
                crs="EPSG:3857",
            )
            truth.to_file(truth_path)
            before_result, after_result, stable = self._period_results(
                root, truth, pixel_size=0.5,
            )
            first = build_fast_change_from_truth(
                truth_path, root / "first", period_key="area:2021->2022",
                change_type="added", global_seed=20260826,
                before_result=before_result, after_result=after_result,
            )
            second = build_fast_change_from_truth(
                truth_path, root / "second", period_key="area:2021->2022",
                change_type="added", global_seed=20260826,
                before_result=before_result, after_result=after_result,
            )
            first_frame = gpd.read_file(first["road_changes"])
            second_frame = gpd.read_file(second["road_changes"])
            self.assertEqual(len(first_frame), len(second_frame))
            self.assertEqual(
                first_frame.geometry.to_wkb().tolist(),
                second_frame.geometry.to_wkb().tolist(),
            )
            combined = first_frame
            self.assertGreaterEqual(int((combined["change_typ"] != "added").sum()), 2)
            false_positives = combined.loc[combined["synth_kind"] == "false_positive"]
            self.assertGreater(len(false_positives), 0)
            truth_support = truth.geometry.union_all()
            self.assertTrue(all(
                geometry.intersection(truth_support).area <= 1e-8
                for geometry in false_positives.geometry
            ))
            stable_support = stable.geometry.union_all().buffer(5.0)
            self.assertTrue(all(
                geometry.intersects(stable_support)
                for geometry in false_positives.geometry
            ))
            self.assertTrue(all(
                max(
                    geometry.bounds[2] - geometry.bounds[0],
                    geometry.bounds[3] - geometry.bounds[1],
                ) >= (
                    FAST_CHANGE_FALSE_POSITIVE_MIN_LENGTH_PX
                    * first["road_centerline_pixel_size"]
                )
                for geometry in false_positives.geometry
            ))
            classified_count = 0
            for type_name in ("added", "width_changed", "removed"):
                classified = gpd.read_file(first["layers"][type_name])
                self.assertTrue(classified.empty or (classified["change_typ"] == type_name).all())
                classified_count += len(classified)
            self.assertEqual(classified_count, len(combined))

    def test_synthetic_geometry_degradation_keeps_one_coherent_partial_shape(self) -> None:
        source = box(0, 0, 100, 10)
        degraded = _degrade_fast_change_geometry(
            source, np.random.default_rng(20260826),
        )
        retained_ratio = float(degraded.area / source.area)
        self.assertGreaterEqual(retained_ratio, 0.75)
        self.assertLessEqual(retained_ratio, 0.90)
        self.assertEqual(degraded.geom_type, "Polygon")
        self.assertTrue(source.covers(degraded))

    def test_synthetic_jitter_converts_pixel_distances_to_map_units(self) -> None:
        source = box(0, 0, 100, 20)
        small = _jitter_fast_change_geometry(
            source, np.random.default_rng(20260826), 0.5,
        )
        large = _jitter_fast_change_geometry(
            source, np.random.default_rng(20260826), 2.0,
        )
        small_shift = source.centroid.distance(small.centroid)
        large_shift = source.centroid.distance(large.centroid)
        self.assertAlmostEqual(large_shift, 4.0 * small_shift, places=6)
        small_buffer = abs((small.bounds[2] - small.bounds[0]) - 100.0) / 2.0
        large_buffer = abs((large.bounds[2] - large.bounds[0]) - 100.0) / 2.0
        self.assertAlmostEqual(large_buffer, 4.0 * small_buffer, places=6)


class FastAutomaticChangeTests(unittest.TestCase):
    transform = from_origin(0, 240, 1, 1)

    def _write_period(self, root, name, roads, *, transform=None):
        directory = root / name
        directory.mkdir()
        transform = transform or self.transform
        lines = [road[0] for road in roads]
        widths = [road[1] for road in roads]
        surface_geometries = [
            line.buffer(width / 2.0, cap_style="flat")
            for line, width in zip(lines, widths)
        ]
        centerline_path = directory / "road_centerlines.shp"
        surface_path = directory / "road_surfaces.shp"
        probability_path = directory / "road_probability.tif"
        gpd.GeoDataFrame(
            {"source": ["native_toponet"] * len(lines)},
            geometry=lines,
            crs="EPSG:3857",
        ).to_file(centerline_path)
        gpd.GeoDataFrame(
            {"source": ["fast"] * len(lines)},
            geometry=surface_geometries,
            crs="EPSG:3857",
        ).to_file(surface_path)
        road_mask = rasterize(
            [(geometry, 1) for geometry in surface_geometries],
            out_shape=(240, 260),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        probability = np.full((240, 260), 0.03, dtype=np.float32)
        probability[road_mask > 0] = 0.80
        with rasterio.open(
            probability_path,
            "w",
            driver="GTiff",
            width=260,
            height=240,
            count=1,
            dtype="float32",
            crs="EPSG:3857",
            transform=transform,
        ) as dataset:
            dataset.write(probability, 1)
        return {
            "centerlines": str(centerline_path),
            "surfaces": str(surface_path),
            "road_probability": str(probability_path),
        }

    def test_probability_presence_and_shared_position_width_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before_roads = [
                (LineString([(20, 40), (220, 40)]), 6.0),
                (LineString([(20, 60), (220, 60)]), 6.0),
                (LineString([(20, 100), (220, 100)]), 6.0),
                (LineString([(20, 150), (220, 150)]), 4.0),
                (LineString([(20, 180), (220, 180)]), 10.0),
                (LineString([(20, 210), (50, 210)]), 4.0),
            ]
            after_roads = [
                (LineString([(20, 40), (220, 40)]), 6.0),
                (LineString([(20, 61), (220, 61)]), 6.0),
                (LineString([(20, 120), (220, 120)]), 6.0),
                (LineString([(20, 150), (220, 150)]), 10.0),
                (LineString([(20, 180), (220, 180)]), 4.0),
                (LineString([(20, 210), (50, 210)]), 10.0),
            ]
            before = self._write_period(root, "before", before_roads)
            after = self._write_period(root, "after", after_roads)
            result = detect_fast_changes(
                before,
                after,
                root / "changes",
                before_period="2021",
                after_period="2022",
                position_tolerance=2.0,
                width_change_absolute=2.0,
                width_change_ratio=0.2,
            )
            for change_type in ("added", "removed", "widened", "narrowed"):
                self.assertGreater(
                    len(gpd.read_file(result["layers"][change_type])),
                    0,
                    change_type,
                )
            changes = gpd.read_file(result["road_changes"])
            self.assertFalse(changes.geometry.intersects(box(10, 56, 230, 65)).any())
            self.assertFalse(changes.geometry.intersects(box(10, 205, 60, 215)).any())
            self.assertTrue(Path(result["gpkg"]).is_file())
            self.assertTrue(Path(result["previews"]["change"]).is_file())
            summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["presence_change_source"],
                "enhanced_probability_difference",
            )
            self.assertEqual(
                summary["width_change_source"],
                "shared_position_sparse_width",
            )
            self.assertGreaterEqual(summary["matched_centerline_pair_count"], 4)
            self.assertNotIn("presence_guard_mode", summary)
            self.assertNotIn("width_guard_mode", summary)
            self.assertNotIn("minimum_continuous_length_m", summary)

    def test_unchanged_period_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roads = [
                (LineString([(20, 40), (220, 40)]), 6.0),
                (LineString([(20, 100), (220, 100)]), 10.0),
            ]
            period = self._write_period(root, "period", roads)
            result = detect_fast_changes(
                period,
                period,
                root / "changes",
                position_tolerance=2.0,
            )
            self.assertTrue(gpd.read_file(result["road_changes"]).empty)
            summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["added_final_pixel_count"], 0)
            self.assertEqual(summary["removed_final_pixel_count"], 0)
            self.assertEqual(summary["width_change_feature_count"], 0)

    def test_probability_rasters_are_aligned_before_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roads = [(LineString([(20, 100), (220, 100)]), 8.0)]
            before = self._write_period(root, "before", roads)
            after = self._write_period(
                root,
                "after",
                roads,
                transform=from_origin(0.5, 240.5, 1, 1),
            )
            result = detect_fast_changes(
                before,
                after,
                root / "changes",
                position_tolerance=2.0,
            )
            self.assertTrue(gpd.read_file(result["road_changes"]).empty)


if __name__ == "__main__":
    unittest.main()
