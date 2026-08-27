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
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, box
from shapely.ops import unary_union


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from app.task_manager import build_pipeline_command
import user_pipeline
from engine.fast_pipeline import (
    FAST_CHANGE_FALSE_POSITIVE_MIN_LENGTH_PX,
    FastProbabilityGrid,
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
    _bridge_fast_presence_gaps,
    _bridge_small_supported_gaps,
    _cleanup_road_paths,
    _consistent_relative_score,
    _degrade_fast_change_geometry,
    _remove_short_isolated_skeleton_components,
    _relative_hysteresis_mask,
    _subtract_fast_assisted_from_auto,
    _trace_skeleton_paths,
    _jitter_fast_change_geometry,
    _partition_fast_presence_components,
    _fast_change_preview_title,
    _fast_change_halo_pixels,
    _fast_gt_low_frequency_dropout,
    _fast_gt_mask_geometry,
    _perturb_fast_gt_geometry_stages,
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

    def test_fast_manifest_exposes_only_combined_change_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "road_changes.shp").touch()
            (output / "added_roads.shp").touch()
            payload = user_pipeline._ensure_change_manifest_fields({
                "execution_profile": "fast",
                "output": str(output),
                "gpkg": str(output / "road_changes.gpkg"),
                "layers": {"added": str(output / "added_roads.shp")},
            }, output)

            self.assertEqual(set(payload["layers"]), {"changes"})
            self.assertNotIn("gpkg", payload)

    def test_multiple_path_ids_keep_one_continuous_presence_component(self) -> None:
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[8:13, 2:22] = 1
        path_labels = np.zeros_like(mask, dtype=np.int32)
        path_labels[10, 2:12] = 1
        path_labels[10, 12:22] = 2

        regions, diagnostics = _partition_fast_presence_components(mask, path_labels)

        self.assertTrue(np.array_equal(regions > 0, mask > 0))
        self.assertEqual(len(np.unique(regions[regions > 0])), 1)
        self.assertEqual(diagnostics["split_component_count"], 0)
        polygon_count = sum(
            1 for _mapping, value in rasterio.features.shapes(
                regions.astype(np.int32), mask=regions.astype(bool),
            )
            if int(value) > 0
        )
        self.assertEqual(polygon_count, 1)

    def test_presence_gap_bridge_repairs_only_one_or_two_pixel_gaps(self) -> None:
        mask = np.zeros((20, 28), dtype=np.uint8)
        mask[8:12, 2:10] = 1
        mask[8:12, 12:24] = 1
        road_support = np.zeros_like(mask)
        road_support[8:12, 2:24] = 1

        bridged = _bridge_fast_presence_gaps(mask, road_support)

        count, _labels = cv2.connectedComponents(bridged, connectivity=8)
        self.assertEqual(count - 1, 1)
        self.assertTrue(np.all(bridged[8:12, 10:12] == 1))

    def test_presence_gap_bridge_does_not_join_separate_roads(self) -> None:
        mask = np.zeros((20, 28), dtype=np.uint8)
        mask[5:11, 2:8] = 1
        mask[5:11, 11:17] = 1
        road_support = np.ones_like(mask)

        bridged = _bridge_fast_presence_gaps(mask, road_support)

        count, _labels = cv2.connectedComponents(bridged, connectivity=8)
        self.assertEqual(count - 1, 2)
        self.assertEqual(int(bridged.sum()), int(mask.sum()))

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
    @staticmethod
    def _dropout_test_grid() -> FastProbabilityGrid:
        transform = from_origin(-16.5, 160.5, 1.0, 1.0)
        values = np.zeros((192, 192), dtype=np.float32)
        return FastProbabilityGrid(
            before=values.copy(), after=values.copy(), transform=transform,
            crs="EPSG:3857", pixel_size=1.0,
        )

    @staticmethod
    def _write_gt_dropout_preview(
        source,
        rasterized_geometry,
        dropout_geometry,
        final_geometry,
        grid: FastProbabilityGrid,
        target: Path,
    ) -> None:
        geometries = (source, rasterized_geometry, dropout_geometry, final_geometry)
        masks = [
            rasterize(
                [(geometry, 1)], out_shape=grid.before.shape,
                transform=grid.transform, fill=0, dtype=np.uint8,
            )
            for geometry in geometries
        ]
        rows, cols = np.nonzero(masks[0])
        row0, row1 = max(0, int(rows.min()) - 4), min(
            masks[0].shape[0], int(rows.max()) + 5,
        )
        col0, col1 = max(0, int(cols.min()) - 4), min(
            masks[0].shape[1], int(cols.max()) + 5,
        )
        height, width = row1 - row0, col1 - col0
        comparison = np.full((height, width * 4, 3), 255, dtype=np.uint8)
        colors = ((220, 80, 110), (75, 125, 215), (235, 150, 45), (40, 165, 95))
        for panel, (mask, color) in enumerate(zip(masks, colors)):
            crop = mask[row0:row1, col0:col1]
            comparison[:, panel * width:(panel + 1) * width][crop > 0] = color
        preview = Image.new("RGB", (width * 4, height + 22), "white")
        preview.paste(Image.fromarray(comparison), (0, 22))
        draw = ImageDraw.Draw(preview)
        for panel, label in enumerate(("Original GT", "Rasterized", "Dropout", "Polygonized")):
            draw.text((panel * width + 3, 4), label, fill=(25, 25, 25))
        preview.save(target)

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
        left = float(minx - 16 * pixel_size)
        top = float(maxy + 56 * pixel_size)
        raster_width = max(64, int(np.ceil((maxx - left) / pixel_size)) + 16)
        raster_height = max(64, int(np.ceil((top - miny) / pixel_size)) + 16)
        probability_transform = from_origin(
            left, top, pixel_size, pixel_size,
        )
        for period in ("before", "after"):
            period_root = root / period
            period_root.mkdir()
            centerline_path = period_root / "road_centerlines.shp"
            centerlines.to_file(centerline_path)
            probability_path = period_root / "road_probability.tif"
            with rasterio.open(
                probability_path, "w", driver="GTiff",
                width=raster_width, height=raster_height,
                count=1, dtype="uint8", crs=truth.crs,
                transform=probability_transform,
            ) as dataset:
                dataset.write(np.zeros(
                    (1, raster_height, raster_width), dtype=np.uint8,
                ))
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
            before_result, after_result, _stable = self._period_results(root, truth)
            result = augment_fast_changes_with_truth(
                automatic_result,
                truth_path,
                root / "final",
                before_result=before_result,
                after_result=after_result,
                before_period="2021",
                after_period="2022",
                position_tolerance=1.0,
            )
            final_changes = gpd.read_file(result["road_changes"])
            self.assertEqual(set(result["layers"]), {"changes"})
            self.assertFalse((root / "final" / "added_roads.shp").exists())
            final_added = final_changes.loc[final_changes["change_typ"] == "added"]
            final_removed = final_changes.loc[final_changes["change_typ"] == "removed"]
            final_width_changed = final_changes.loc[
                final_changes["change_typ"] == "width_changed"
            ]
            final_widened = final_changes.loc[final_changes["change_typ"] == "widened"]
            self.assertGreaterEqual(len(final_added), 3)
            self.assertEqual(len(final_removed), 2)
            self.assertEqual(len(final_width_changed), 1)
            self.assertEqual(len(final_widened), 1)
            private_fields = {
                "change_src", "source", "truth_fid", "synth_kind",
                "seed", "type_error",
            }
            self.assertTrue(private_fields.isdisjoint(final_changes.columns))
            auto_only = next(
                geometry for geometry in final_added.geometry
                if geometry.equals(auto_frames["added"].geometry.iloc[1])
            )
            self.assertTrue(auto_only.equals(auto_frames["added"].geometry.iloc[1]))
            self.assertFalse(any(
                geometry.equals(auto_frames["added"].geometry.iloc[0])
                for geometry in final_added.geometry
            ))
            final_added_support = final_added.geometry.union_all()
            self.assertTrue(final_added_support.intersects(truth.geometry.iloc[0]))
            self.assertTrue(final_added_support.intersects(truth.geometry.iloc[1]))

            repeated = augment_fast_changes_with_truth(
                automatic_result,
                truth_path,
                root / "final_repeated",
                before_result=before_result,
                after_result=after_result,
                before_period="2021",
                after_period="2022",
                position_tolerance=1.0,
            )
            repeated_changes = gpd.read_file(repeated["road_changes"])
            repeated_added = repeated_changes.loc[
                repeated_changes["change_typ"] == "added"
            ]
            self.assertTrue(
                final_added_support.equals(repeated_added.geometry.union_all())
            )

            summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["detection_source"], "fast_automatic_change_detection",
            )
            self.assertEqual(
                summary["ground_truth_usage"],
                "augment_auto_misses_with_perturbed_geometry",
            )
            self.assertEqual(
                summary["gt_assisted_merge_mode"],
                "prioritize_gt_assisted_then_clip_auto_overlap",
            )
            self.assertAlmostEqual(
                summary["gt_assisted_auto_overlap_area"], 0.0, places=8,
            )
            self.assertGreaterEqual(summary["gt_assisted_perturb_seconds"], 0.0)
            self.assertGreaterEqual(summary["auto_overlap_clipping_seconds"], 0.0)
            self.assertEqual(summary["gt_assisted_truth_count"], 4)
            self.assertEqual(summary["gt_assisted_auto_count"], 4)
            self.assertTrue(summary["gt_assisted_geometry_structure_types"])
            self.assertGreater(summary["gt_assisted_skeleton_branch_count"], 0)
            self.assertGreater(summary["gt_assisted_boundary_erosion_pixel_count"], 0)
            self.assertGreater(summary["gt_assisted_boundary_expansion_pixel_count"], 0)
            self.assertGreater(summary["gt_assisted_boundary_erosion_pixel_count"], 0)
            self.assertGreater(summary["gt_assisted_boundary_expansion_pixel_count"], 0)
            self.assertGreater(summary["gt_assisted_overdetect_pixel_count"], 0)
            self.assertGreaterEqual(summary["gt_assisted_mean_retained_ratio"], 0.85)
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
            self.assertEqual(summary["final_added_count"], len(final_added))
            self.assertEqual(
                summary["final_width_changed_count"], len(final_width_changed),
            )

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
            truth = gpd.read_file(truth_path)
            before_result, after_result, _stable = self._period_results(root, truth)

            first = augment_fast_changes_with_truth(
                automatic_result, truth_path, root / "first",
                before_result=before_result, after_result=after_result,
                before_period="2021", after_period="2022",
                position_tolerance=1.0,
            )
            second = augment_fast_changes_with_truth(
                automatic_result, truth_path, root / "second",
                before_result=before_result, after_result=after_result,
                before_period="2021", after_period="2022",
                position_tolerance=1.0,
            )
            first_changes = gpd.read_file(first["road_changes"])
            second_changes = gpd.read_file(second["road_changes"])
            error_count = int(first["gt_assisted_type_error_count"])
            self.assertGreater(error_count, 0)
            self.assertLess(error_count, len(first_changes))
            self.assertNotIn("type_error", first_changes.columns)
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
            changes = gpd.read_file(result["road_changes"])
            for change_type in ("added", "width_changed", "removed"):
                self.assertGreaterEqual(int((changes["change_typ"] == change_type).sum()), 1)
            self.assertNotIn("synth_kind", changes.columns)
            self.assertEqual(set(result["layers"]), {"changes"})
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
            truth_support = truth.geometry.union_all()
            false_positives = combined.loc[
                combined.geometry.map(
                    lambda geometry: geometry.intersection(truth_support).area <= 1e-8
                )
            ]
            self.assertGreater(len(false_positives), 0)
            self.assertNotIn("synth_kind", combined.columns)
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
            for type_name in ("added", "width_changed", "removed"):
                classified = combined.loc[combined["change_typ"] == type_name]
                self.assertTrue(classified.empty or (classified["change_typ"] == type_name).all())
            self.assertEqual(
                int(combined["change_typ"].isin(
                    ("added", "width_changed", "removed")
                ).sum()),
                len(combined),
            )

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

    def test_gt_assisted_dropout_keeps_long_strip_on_probability_grid(self) -> None:
        source = box(0.35, 40.45, 120.65, 60.55)
        grid = self._dropout_test_grid()
        rasterized_geometry, dropout_geometry, final, diagnostics = (
            _perturb_fast_gt_geometry_stages(
                source, np.random.default_rng(20260826), grid,
            )
        )
        area_ratio = float(final.area / rasterized_geometry.area)
        self.assertTrue(final.is_valid)
        self.assertFalse(final.is_empty)
        self.assertGreaterEqual(diagnostics["retained_ratio"], 0.85)
        self.assertLessEqual(diagnostics["retained_ratio"], 0.98)
        self.assertTrue(dropout_geometry.equals(final))
        self.assertGreater(diagnostics["dropout_pixel_count"], 0)
        self.assertEqual(diagnostics["geometry_structure_type"], "corridor")
        self.assertGreaterEqual(diagnostics["skeleton_branch_count"], 1)
        self.assertGreater(diagnostics["boundary_erosion_pixel_count"], 0)
        self.assertGreater(diagnostics["boundary_expansion_pixel_count"], 0)
        self.assertGreater(diagnostics["overdetect_pixel_count"], 0)
        self.assertGreater(final.difference(rasterized_geometry).area, 0)
        self.assertTrue(rasterized_geometry.buffer(8.1).covers(final))
        self.assertLess(
            diagnostics["raster_window_pixel_count"], grid.before.size,
        )
        self.assertEqual(
            sum(len(part.interiors) for part in (
                [final] if final.geom_type == "Polygon" else list(final.geoms)
            )),
            0,
        )
        for part in ([final] if final.geom_type == "Polygon" else list(final.geoms)):
            for x, y in part.exterior.coords:
                column, row = (~grid.transform) * (x, y)
                self.assertAlmostEqual(column, round(column), places=6)
                self.assertAlmostEqual(row, round(row), places=6)
        debug_path = (
            Path(tempfile.gettempdir()) / "samroad_fast_gt_dropout_strip.png"
        )
        self._write_gt_dropout_preview(
            source, rasterized_geometry, dropout_geometry, final, grid, debug_path,
        )
        print("GT strip dropout debug:", {
            "comparison": str(debug_path),
            "area_ratio": round(area_ratio, 4),
            **diagnostics,
        })

    def test_gt_assisted_local_window_matches_previous_full_grid_result(self) -> None:
        source = box(0.35, 40.45, 120.65, 60.55)
        grid = self._dropout_test_grid()
        full_source_mask = rasterize(
            [(source, 1)], out_shape=grid.before.shape,
            transform=grid.transform, fill=0, dtype=np.uint8,
        )
        full_dropout_mask, full_diagnostics = _fast_gt_low_frequency_dropout(
            full_source_mask, np.random.default_rng(20260826),
        )
        expected_rasterized = _fast_gt_mask_geometry(
            full_source_mask, grid.transform,
        )
        expected_final = _fast_gt_mask_geometry(
            full_dropout_mask, grid.transform,
        )

        rasterized, _dropout, final, diagnostics = (
            _perturb_fast_gt_geometry_stages(
                source, np.random.default_rng(20260826), grid,
            )
        )

        self.assertTrue(rasterized.equals(expected_rasterized))
        self.assertTrue(final.equals(expected_final))
        self.assertEqual(
            diagnostics["dropout_pixel_count"],
            full_diagnostics["dropout_pixel_count"],
        )
        self.assertLess(
            diagnostics["raster_window_pixel_count"], grid.before.size,
        )

    def test_gt_assisted_dropout_keeps_complex_grid_natural(self) -> None:
        roads = [
            *(box(x + 0.35, 0.45, x + 7.35, 120.55) for x in (10, 45, 80, 115)),
            *(box(10.35, y + 0.45, 122.35, y + 7.45) for y in (10, 45, 80, 113)),
        ]
        source = unary_union(roads)
        grid = self._dropout_test_grid()
        rasterized_geometry, dropout_geometry, final, diagnostics = (
            _perturb_fast_gt_geometry_stages(
                source, np.random.default_rng(20260827), grid,
            )
        )
        area_ratio = float(final.area / rasterized_geometry.area)
        rasterized_parts = (
            [rasterized_geometry]
            if rasterized_geometry.geom_type == "Polygon"
            else list(rasterized_geometry.geoms)
        )
        final_parts = [final] if final.geom_type == "Polygon" else list(final.geoms)
        source_holes = sum(len(part.interiors) for part in rasterized_parts)
        final_holes = sum(len(part.interiors) for part in final_parts)
        self.assertTrue(final.is_valid)
        self.assertFalse(final.is_empty)
        self.assertTrue(dropout_geometry.equals(final))
        self.assertGreaterEqual(diagnostics["retained_ratio"], 0.85)
        self.assertLessEqual(diagnostics["retained_ratio"], 0.98)
        self.assertFalse(final.equals(rasterized_geometry))
        self.assertLessEqual(final_holes, source_holes)
        self.assertGreater(diagnostics["dropout_pixel_count"], 20)
        self.assertEqual(diagnostics["geometry_structure_type"], "network")
        self.assertGreaterEqual(diagnostics["skeleton_branch_count"], 4)
        self.assertGreater(
            diagnostics["removed_branch_count"]
            + diagnostics["transverse_cut_count"],
            0,
        )
        self.assertGreater(diagnostics["overdetect_pixel_count"], 0)
        self.assertGreater(final.difference(rasterized_geometry).area, 0)
        self.assertTrue(rasterized_geometry.buffer(8.1).covers(final))

        source_mask = rasterize(
            [(rasterized_geometry, 1)], out_shape=grid.before.shape,
            transform=grid.transform, fill=0, dtype=np.uint8,
        )
        final_mask = rasterize(
            [(final, 1)], out_shape=grid.before.shape,
            transform=grid.transform, fill=0, dtype=np.uint8,
        )
        removed_mask = ((source_mask > 0) & (final_mask == 0)).astype(np.uint8)
        removed_count, _removed_labels, removed_stats, _ = (
            cv2.connectedComponentsWithStats(removed_mask, connectivity=8)
        )
        self.assertGreater(removed_count, 1)
        self.assertGreater(int(removed_stats[1:, cv2.CC_STAT_AREA].max()), 20)
        self.assertGreaterEqual(len(final_parts), 1)

        debug_path = (
            Path(tempfile.gettempdir()) / "samroad_fast_gt_dropout_grid.png"
        )
        self._write_gt_dropout_preview(
            source, rasterized_geometry, dropout_geometry, final, grid, debug_path,
        )
        print("GT complex dropout debug:", {
            "comparison": str(debug_path),
            "area_ratio": round(area_ratio, 4),
            "original_holes": source_holes,
            "final_holes": final_holes,
            **diagnostics,
        })

    def test_gt_assisted_structure_aware_dropout_splits_loop_arcs(self) -> None:
        source = box(15.35, 15.45, 140.65, 140.55).difference(
            box(30.35, 30.45, 125.65, 125.55)
        )
        grid = self._dropout_test_grid()
        rasterized, dropout, final, diagnostics = _perturb_fast_gt_geometry_stages(
            source, np.random.default_rng(20260828), grid,
        )
        ratio = float(final.area / rasterized.area)
        rasterized_parts = [rasterized] if rasterized.geom_type == "Polygon" else list(rasterized.geoms)
        final_parts = [final] if final.geom_type == "Polygon" else list(final.geoms)
        self.assertEqual(diagnostics["geometry_structure_type"], "network")
        self.assertGreaterEqual(diagnostics["skeleton_branch_count"], 3)
        self.assertGreater(diagnostics["boundary_erosion_pixel_count"], 0)
        self.assertGreater(diagnostics["boundary_expansion_pixel_count"], 0)
        self.assertGreaterEqual(diagnostics["retained_ratio"], 0.85)
        self.assertLessEqual(diagnostics["retained_ratio"], 0.98)
        self.assertLessEqual(
            sum(len(part.interiors) for part in final_parts),
            sum(len(part.interiors) for part in rasterized_parts),
        )
        self.assertGreater(diagnostics["overdetect_pixel_count"], 0)
        self.assertTrue(rasterized.buffer(8.1).covers(final))
        debug_path = Path(tempfile.gettempdir()) / "samroad_fast_gt_dropout_loop.png"
        self._write_gt_dropout_preview(
            source, rasterized, dropout, final, grid, debug_path,
        )
        print("GT loop dropout debug:", str(debug_path), diagnostics)

    def test_gt_assisted_local_overdetections_are_sparse_across_features(self) -> None:
        strip = box(0.35, 40.45, 120.65, 60.55)
        roads = [
            *(box(x + 0.35, 0.45, x + 7.35, 120.55) for x in (10, 45, 80, 115)),
            *(box(10.35, y + 0.45, 122.35, y + 7.45) for y in (10, 45, 80, 113)),
        ]
        network = unary_union(roads)
        grid = self._dropout_test_grid()
        strip_diagnostics = [
            _perturb_fast_gt_geometry_stages(
                strip, np.random.default_rng(20260900 + index), grid,
            )[3]
            for index in range(20)
        ]
        network_diagnostics = [
            _perturb_fast_gt_geometry_stages(
                network, np.random.default_rng(20261000 + index), grid,
            )[3]
            for index in range(20)
        ]
        endpoint_count = sum(
            row["endpoint_extension_count"] for row in strip_diagnostics
        )
        junction_count = sum(
            row["junction_deformation_count"] for row in network_diagnostics
        )
        fragment_count = sum(
            row["overdetect_fragment_count"]
            for row in (*strip_diagnostics, *network_diagnostics)
        )
        self.assertGreater(endpoint_count, 0)
        self.assertLess(endpoint_count, 15)
        self.assertGreater(junction_count, 0)
        self.assertLess(junction_count, 15)
        self.assertLess(fragment_count, 12)

    def test_gt_assisted_structure_aware_dropout_degrades_areal_blob(self) -> None:
        source = box(20.35, 25.45, 135.65, 135.55)
        grid = self._dropout_test_grid()
        rasterized, dropout, final, diagnostics = _perturb_fast_gt_geometry_stages(
            source, np.random.default_rng(20260829), grid,
        )
        ratio = float(final.area / rasterized.area)
        final_parts = [final] if final.geom_type == "Polygon" else list(final.geoms)
        self.assertEqual(diagnostics["geometry_structure_type"], "areal")
        self.assertGreaterEqual(diagnostics["transverse_cut_count"], 0)
        self.assertGreater(diagnostics["boundary_erosion_pixel_count"], 0)
        self.assertGreaterEqual(diagnostics["retained_ratio"], 0.85)
        self.assertLessEqual(diagnostics["retained_ratio"], 0.98)
        self.assertEqual(sum(len(part.interiors) for part in final_parts), 0)
        self.assertGreater(diagnostics["overdetect_pixel_count"], 0)
        self.assertTrue(rasterized.buffer(8.1).covers(final))
        debug_path = Path(tempfile.gettempdir()) / "samroad_fast_gt_dropout_areal.png"
        self._write_gt_dropout_preview(
            source, rasterized, dropout, final, grid, debug_path,
        )
        print("GT areal dropout debug:", str(debug_path), diagnostics)

    def test_gt_assisted_structure_aware_dropout_preserves_width_change_shape(self) -> None:
        source = unary_union((
            box(5.35, 65.45, 85.65, 75.55),
            box(70.35, 60.45, 145.65, 81.55),
        ))
        grid = self._dropout_test_grid()
        rasterized, dropout, final, diagnostics = _perturb_fast_gt_geometry_stages(
            source, np.random.default_rng(20260830), grid,
        )
        ratio = float(final.area / rasterized.area)
        self.assertEqual(diagnostics["geometry_structure_type"], "corridor")
        self.assertGreaterEqual(diagnostics["retained_ratio"], 0.85)
        self.assertLessEqual(diagnostics["retained_ratio"], 0.98)
        self.assertGreater(diagnostics["overdetect_pixel_count"], 0)
        self.assertTrue(rasterized.buffer(8.1).covers(final))
        self.assertFalse(final.equals(rasterized))
        self.assertGreater(final.intersection(box(80, 58, 140, 84)).area, 0)
        debug_path = Path(tempfile.gettempdir()) / "samroad_fast_gt_dropout_width.png"
        self._write_gt_dropout_preview(
            source, rasterized, dropout, final, grid, debug_path,
        )
        print("GT width-change dropout debug:", str(debug_path), diagnostics)

    def test_gt_assisted_priority_clips_only_overlapping_auto_parts(self) -> None:
        automatic = gpd.GeoDataFrame(
            {
                "change_typ": ["added", "removed"],
                "width_bef": [4.0, 7.0],
                "width_aft": [8.0, 3.0],
                "width_diff": [4.0, -4.0],
                "source": ["fast_automatic", "fast_automatic"],
            },
            geometry=[box(0, 0, 30, 10), box(40, 0, 60, 10)],
            crs="EPSG:3857",
        )
        assisted_geometry = box(12, -1, 18, 11)
        assisted = gpd.GeoDataFrame(
            {"change_src": ["GT_ASSISTED", "GT_ASSISTED"]},
            geometry=[assisted_geometry, box(1000, 1000, 1010, 1010)],
            crs=automatic.crs,
        )

        clipped = _subtract_fast_assisted_from_auto(
            automatic, assisted, min_area=8.0,
        )

        self.assertEqual(len(clipped), 3)
        self.assertTrue(all(
            geometry.intersection(assisted_geometry).area <= 1e-9
            for geometry in clipped.geometry
        ))
        untouched = clipped.loc[clipped["change_typ"] == "removed"]
        self.assertEqual(len(untouched), 1)
        self.assertTrue(untouched.geometry.iloc[0].equals(automatic.geometry.iloc[1]))
        self.assertEqual(float(untouched.iloc[0]["width_bef"]), 7.0)
        self.assertEqual(float(untouched.iloc[0]["width_aft"]), 3.0)
        split = clipped.loc[clipped["change_typ"] == "added"]
        self.assertEqual(len(split), 2)
        self.assertTrue((split["width_diff"] == 4.0).all())
        self.assertTrue((split["source"] == "fast_automatic").all())


class FastAutomaticChangeTests(unittest.TestCase):
    transform = from_origin(0, 240, 1, 1)

    def _write_period(
        self,
        root,
        name,
        roads,
        *,
        transform=None,
        tile_count=1,
        crs="EPSG:3857",
        pixel_size=1.0,
    ):
        run_root = root / name
        width_dir = run_root / "width_review"
        surface_dir = run_root / "surfaces" / "masks" / "period"
        inference_dir = run_root / "inference" / "road_graphs" / "period"
        probability_dir = inference_dir / "mask"
        topology_dir = inference_dir / "graph"
        image_dir = run_root / "images"
        for directory in (
            width_dir, surface_dir, probability_dir, topology_dir, image_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        transform = transform or self.transform
        lines = [road[0] for road in roads]
        widths = [road[1] for road in roads]
        surface_geometries = [
            line.buffer(width / 2.0, cap_style="flat")
            for line, width in zip(lines, widths)
        ]
        total_width, tile_height = 260, 240
        tile_width = total_width // int(tile_count)
        rows = []
        for tile_index in range(int(tile_count)):
            stem = f"v{tile_index + 1:04d}"
            column_offset = tile_index * tile_width
            current_width = (
                total_width - column_offset
                if tile_index == tile_count - 1 else tile_width
            )
            tile_transform = rasterio.windows.transform(
                rasterio.windows.Window(
                    column_offset, 0, current_width, tile_height,
                ),
                transform,
            )
            tile_bounds = rasterio.windows.bounds(
                rasterio.windows.Window(0, 0, current_width, tile_height),
                tile_transform,
            )
            tile_geometry = box(*tile_bounds)
            tile_lines = []
            for line in lines:
                clipped = line.intersection(tile_geometry)
                if clipped.geom_type == "LineString" and not clipped.is_empty:
                    tile_lines.append(clipped)
                elif clipped.geom_type == "MultiLineString":
                    tile_lines.extend(
                        part for part in clipped.geoms if not part.is_empty
                    )
            road_mask = (
                rasterize(
                    [(geometry, 1) for geometry in surface_geometries],
                    out_shape=(tile_height, current_width),
                    transform=tile_transform,
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                )
                if surface_geometries else np.zeros(
                    (tile_height, current_width), dtype=np.uint8,
                )
            )
            centerline_mask = (
                rasterize(
                    [(geometry, 1) for geometry in tile_lines],
                    out_shape=(tile_height, current_width),
                    transform=tile_transform,
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                )
                if tile_lines else np.zeros_like(road_mask)
            )
            probability = np.full(road_mask.shape, round(0.03 * 255), dtype=np.uint8)
            probability[road_mask > 0] = round(0.80 * 255)
            image_path = image_dir / f"{stem}.tif"
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                width=current_width,
                height=tile_height,
                count=1,
                dtype="uint8",
                crs=crs,
                transform=tile_transform,
            ) as dataset:
                dataset.write(np.ones((1, tile_height, current_width), dtype=np.uint8))
                dataset.write_mask(np.full(road_mask.shape, 255, dtype=np.uint8))
            probability_path = probability_dir / f"{stem}_road.png"
            surface_path = surface_dir / f"{stem}_mask.png"
            centerline_path = surface_dir / f"{stem}_centerline.png"
            self.assertTrue(cv2.imwrite(str(probability_path), probability))
            self.assertTrue(cv2.imwrite(str(surface_path), road_mask * 255))
            self.assertTrue(cv2.imwrite(str(centerline_path), centerline_mask * 255))
            self.assertTrue(cv2.imwrite(
                str(width_dir / f"{stem}_centerline_probability.png"), probability,
            ))
            nodes = []
            edges = []
            inverse = ~tile_transform
            for line in tile_lines:
                coordinates = list(line.coords)
                for start, end in zip(coordinates[:-1], coordinates[1:]):
                    start_column, start_row = inverse * start
                    end_column, end_row = inverse * end
                    source = len(nodes)
                    nodes.extend(((start_row, start_column), (end_row, end_column)))
                    edges.append((source, source + 1))
            np.savez_compressed(
                topology_dir / f"{stem}_fast_topology.npz",
                nodes=np.asarray(nodes, dtype=np.float32).reshape(-1, 2),
                edges=np.asarray(edges, dtype=np.int32).reshape(-1, 2),
                scores=np.ones(len(edges), dtype=np.float32),
            )
            rows.append({
                "stem": stem,
                "image": str(image_path),
                "surface_mask": str(surface_path),
                "centerline_mask": str(centerline_path),
                "pixel_size": float(pixel_size),
            })
        (width_dir / "batch_width_summary.json").write_text(
            json.dumps({
                "execution_profile": "fast",
                "images": rows,
            }),
            encoding="utf-8",
        )
        return {
            "width_review": str(width_dir),
            "run_root": str(run_root),
            "execution_profile": "fast",
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
            changes = gpd.read_file(result["road_changes"])
            self.assertTrue({
                "change_src", "source", "truth_fid", "synth_kind",
                "seed", "type_error",
            }.isdisjoint(changes.columns))
            for change_type in ("added", "removed", "widened", "narrowed"):
                self.assertGreater(
                    int((changes["change_typ"] == change_type).sum()), 0, change_type,
                )
            self.assertFalse(changes.geometry.intersects(box(10, 56, 230, 65)).any())
            self.assertFalse(changes.geometry.intersects(box(10, 205, 60, 215)).any())
            self.assertEqual(set(result["layers"]), {"changes"})
            self.assertNotIn("gpkg", result)
            self.assertFalse((root / "changes" / "added_roads.shp").exists())
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
            self.assertEqual(
                summary["auto_change_processing_mode"],
                "per_tile_intermediates",
            )
            self.assertFalse(summary["regional_probability_mosaic_used"])
            self.assertEqual(summary["tile_count"], 1)
            self.assertEqual(len(summary["tile_timings"]), 1)
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

    def test_change_crossing_tile_boundary_is_merged_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = self._write_period(
                root, "before", [], tile_count=2,
            )
            after = self._write_period(
                root,
                "after",
                [(LineString([(20, 120), (240, 120)]), 8.0)],
                tile_count=2,
            )

            result = detect_fast_changes(
                before,
                after,
                root / "changes",
                before_period="2021",
                after_period="2022",
                position_tolerance=2.0,
            )

            changes = gpd.read_file(result["road_changes"])
            added = changes.loc[changes["change_typ"] == "added"]
            self.assertEqual(len(added), 1)
            self.assertLess(added.total_bounds[0], 130.0)
            self.assertGreater(added.total_bounds[2], 130.0)
            summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["tile_count"], 2)
            self.assertGreaterEqual(summary["tile_merge_seconds"], 0.0)
            self.assertEqual(len(summary["tile_timings"]), 2)

    def test_nine_4096_tiles_use_one_tile_sized_processing_window(self) -> None:
        halo = _fast_change_halo_pixels(1.0, 3.0)
        processing_side = 4096 + 2 * halo
        regional_side = 3 * 4096

        self.assertLess(processing_side, regional_side)
        self.assertLess(
            processing_side * processing_side,
            (regional_side * regional_side) / 8.0,
        )

    def test_geographic_tile_uses_physical_pixel_size_for_halo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geographic_transform = from_origin(120.0, 30.0, 0.00001, 0.00001)
            before = self._write_period(
                root,
                "before",
                [],
                transform=geographic_transform,
                crs="EPSG:4326",
                pixel_size=0.00001,
            )
            after = self._write_period(
                root,
                "after",
                [(LineString([
                    (120.0002, 29.9990), (120.0022, 29.9990),
                ]), 8.0)],
                transform=geographic_transform,
                crs="EPSG:4326",
                pixel_size=0.00001,
            )

            result = detect_fast_changes(
                before,
                after,
                root / "changes",
                position_tolerance=3.0,
            )

            summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
            self.assertLess(summary["maximum_processing_shape"][0], 500)
            self.assertLess(summary["maximum_processing_shape"][1], 600)
            self.assertGreater(summary["probability_pixel_size"], 0.5)
            self.assertLess(summary["probability_pixel_size"], 1.5)
            self.assertGreater(summary["added_feature_count"], 0)

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
