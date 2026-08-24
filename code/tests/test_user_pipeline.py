from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import user_pipeline
from dev_tools import generate_mock_change_truth


class GridDiscoveryTests(unittest.TestCase):
    def test_discovers_txt_direct_raster_and_period_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            grid = root / "grid_01"
            grid.mkdir()
            source = root / "source.tif"
            source.touch()
            (grid / "2021.txt").write_bytes(str(source).encode("gb18030"))
            (grid / "2022.tif").touch()
            period_dir = grid / "2023"
            period_dir.mkdir()
            (period_dir / "tile.tif").touch()
            (root / "province_shp").mkdir()

            found = user_pipeline.discover_grid_periods(root)

            self.assertEqual(list(found), ["grid_01"])
            self.assertEqual(list(found["grid_01"]), ["2021", "2022", "2023"])

    def test_rejects_grid_with_only_one_period(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            grid = Path(raw) / "grid_01"
            grid.mkdir()
            (grid / "2024.tif").touch()
            with self.assertRaisesRegex(RuntimeError, "少于两个期次"):
                user_pipeline.discover_grid_periods(Path(raw))


class ProjectPeriodExtractionTests(unittest.TestCase):
    @staticmethod
    def _checkpoint_workspace(root: Path) -> tuple[Path, argparse.Namespace]:
        workspace = root / "workspace"
        images = workspace / "images"; images.mkdir(parents=True)
        (images / "tile.tif").touch()
        image_txt = workspace / "tiles.txt"; image_txt.write_text(str(images / "tile.tif"), encoding="utf-8")
        user_pipeline.write_json(
            workspace / "input_manifest.json",
            {"images": str(images), "image_txt": str(image_txt)},
        )
        checkpoint = root / "model.ckpt"; checkpoint.touch()
        config = root / "config.yaml"; config.touch()
        return workspace, argparse.Namespace(
            workspace=str(workspace), source="", checkpoint=str(checkpoint), config=str(config),
            device="cpu", pixel_size="0", rescale="off", run_id="roads",
            junction_node_mode="sparse", grid="area_03", period="2021",
            resume=False, pipeline_state="",
        )

    @staticmethod
    def _complete_fake_stage(workspace: Path, label: str) -> dict:
        run = workspace / "runs" / "roads"
        if label == "道路提取":
            path = run / "inference" / "road_graphs" / "tiles" / "graph" / "tile.p"
            path.parent.mkdir(parents=True, exist_ok=True); path.touch()
        elif label == "道路面提取":
            path = run / "surfaces" / "masks" / "tiles" / "tile_mask.png"
            path.parent.mkdir(parents=True, exist_ok=True); path.touch()
        elif label == "道路宽度计算":
            root = run / "width_review"; root.mkdir(parents=True, exist_ok=True)
            (root / "batch_width_summary.json").write_text("{}", encoding="utf-8")
            (root / "tile_summary.json").write_text("{}", encoding="utf-8")
        elif label == "结果固化":
            root = run / "finalized"; root.mkdir(parents=True, exist_ok=True)
            (root / "batch_optimized_summary.json").write_text("{}", encoding="utf-8")
            (root / "tile_optimized_summary.json").write_text("{}", encoding="utf-8")
        elif label == "道路产品导出":
            root = run / "products"; root.mkdir(parents=True, exist_ok=True)
            for stem in ("road_centerlines", "road_surfaces"):
                for suffix in (".shp", ".shx", ".dbf"):
                    (root / f"{stem}{suffix}").touch()
            (root / "roads.gpkg").touch()
        return {"stage": label, "elapsed_seconds": 0.01}

    def test_resume_after_centerline_does_not_run_centerline_again(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace, args = self._checkpoint_workspace(root)
            pipeline_state = root / "job_state.json"
            user_pipeline.write_json(pipeline_state, {"run_id": "run_a", "status": "running"})
            args.pipeline_state = str(pipeline_state)
            first_calls = []

            def interrupted(_command, _cwd, _env, label, _context=None):
                first_calls.append(label)
                if label == "道路面提取":
                    raise RuntimeError("interrupted")
                return self._complete_fake_stage(workspace, label)

            with patch.object(user_pipeline, "run_command", side_effect=interrupted), patch.object(user_pipeline, "_write_valid_observation_area", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    user_pipeline.extract(args)
            self.assertEqual(first_calls, ["道路提取", "道路面提取"])
            state = user_pipeline.read_json(workspace / "period_state.json")
            self.assertEqual(state["stages"]["centerline"], "completed")
            job_state = user_pipeline.read_json(pipeline_state)
            self.assertEqual(
                (job_state["current_grid"], job_state["current_period"], job_state["current_stage"]),
                ("area_03", "2021", "surface"),
            )
            self.assertEqual(job_state["last_completed_stage"], "centerline")

            resumed_calls = []
            args.resume = True

            def resumed(_command, _cwd, _env, label, _context=None):
                resumed_calls.append(label)
                return self._complete_fake_stage(workspace, label)

            with patch.object(user_pipeline, "run_command", side_effect=resumed), patch.object(user_pipeline, "_write_valid_observation_area", return_value=None):
                result = user_pipeline.extract(args)

            self.assertNotIn("道路提取", resumed_calls)
            self.assertEqual(resumed_calls[0], "道路面提取")
            self.assertEqual(user_pipeline.read_json(Path(result["period_state"]))["status"], "completed")

    def test_running_stage_is_reexecuted_from_that_stage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace, args = self._checkpoint_workspace(Path(raw))
            self._complete_fake_stage(workspace, "道路提取")
            state = user_pipeline._period_state_template("area_03", "2021")
            state.update({"status": "running", "current_stage": "surface", "current_stage_label": "道路面提取"})
            state["stages"].update({"prepare": "completed", "centerline": "completed", "surface": "running"})
            user_pipeline.write_json(workspace / "period_state.json", state)
            args.resume = True
            calls = []

            def fake_run(_command, _cwd, _env, label, _context=None):
                calls.append(label)
                return self._complete_fake_stage(workspace, label)

            with patch.object(user_pipeline, "run_command", side_effect=fake_run), patch.object(user_pipeline, "_write_valid_observation_area", return_value=None):
                user_pipeline.extract(args)

            self.assertEqual(calls[0], "道路面提取")
            self.assertNotIn("道路提取", calls)

    def test_completed_stage_with_missing_output_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace, args = self._checkpoint_workspace(Path(raw))
            state = user_pipeline._period_state_template("area_03", "2021")
            state["stages"].update({"prepare": "completed", "centerline": "completed"})
            state.update({"status": "running", "last_completed_stage": "centerline"})
            user_pipeline.write_json(workspace / "period_state.json", state)
            args.resume = True
            calls = []

            def fake_run(_command, _cwd, _env, label, _context=None):
                calls.append(label)
                return self._complete_fake_stage(workspace, label)

            with patch.object(user_pipeline, "run_command", side_effect=fake_run), patch.object(user_pipeline, "_write_valid_observation_area", return_value=None):
                user_pipeline.extract(args)

            self.assertEqual(calls[0], "道路提取")

    def test_single_period_extract_and_resume_reuse_complete_products(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "2024.txt"; source.write_text("tile.tif\n", encoding="utf-8")
            boundary = root / "area.shp"; boundary.touch()
            output = root / "04_outputs"
            discovered = {
                "project_root": str(root), "output_root": str(output),
                "areas": [{"area_id": "area-a", "validation_area": str(boundary), "periods": [{"period": "2024", "source": str(source)}]}],
            }
            args = argparse.Namespace(project_root=str(root), area_id="area-a", period="2024", run_id="roads", device="cpu", pixel_size="0.0", rescale="off", resume=False)

            def fake_normalize(_periods, _area, normalized_root):
                period_dir = normalized_root / "2024"; period_dir.mkdir(parents=True)
                (period_dir / "v00001.tif").touch()
                user_pipeline.write_json(normalized_root / "normalization_complete.json", {"periods": {"2024": {}}})
                return {"2024": period_dir}

            def fake_prepare(call_args):
                workspace = Path(call_args.workspace); (workspace / "images").mkdir(parents=True)
                user_pipeline.write_json(workspace / "input_manifest.json", {"images": str(workspace / "images"), "image_txt": str(workspace / "tiles.txt")})

            def fake_extract(call_args):
                workspace = Path(call_args.workspace); products = workspace / "products"; products.mkdir()
                result = {"run_root": str(workspace / "runs" / call_args.run_id), "centerlines": str(products / "center.shp"), "surfaces": str(products / "surface.shp"), "gpkg": str(products / "roads.gpkg"), "final_dir": str(workspace / "final")}
                for key in ("centerlines", "surfaces", "gpkg"): Path(result[key]).touch()
                user_pipeline.write_json(workspace / "latest_result.json", result)
                return result

            with patch.object(user_pipeline, "discover_validation_project", return_value=discovered), patch.object(user_pipeline, "validate_validation_inputs", return_value={"2024": source}) as validate_mock, patch.object(user_pipeline, "normalize_validation_sources", side_effect=fake_normalize), patch.object(user_pipeline, "prepare", side_effect=fake_prepare), patch.object(user_pipeline, "extract", side_effect=fake_extract) as extract_mock:
                first = user_pipeline.extract_project_period(args)
                args.resume = True
                resumed = user_pipeline.extract_project_period(args)

            self.assertEqual(first["status"], "completed")
            self.assertEqual(resumed["attempt"], 2)
            self.assertEqual(extract_mock.call_count, 1)
            self.assertEqual(validate_mock.call_args.kwargs["minimum_periods"], 1)
            self.assertTrue(Path(resumed["result"]).is_file())


class ProjectBatchExtractionTests(unittest.TestCase):
    def test_extracts_all_periods_without_change_detection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); output = root / "04_成果输出"
            discovered = {"project_root": str(root), "output_root": str(output), "areas": [
                {"area_id": "north", "validation_area": "north.shp", "periods": [{"period": "2021", "source": "a.txt"}, {"period": "2022", "source": "b.txt"}]},
                {"area_id": "south", "validation_area": "south.shp", "periods": [{"period": "2020", "source": "c.txt"}]},
            ]}

            def complete(period_args):
                state_path = output / "period_extractions" / period_args.area_id / period_args.period / period_args.run_id / "period_task.json"
                state = {"status": "completed", "result": str(state_path.parent / "latest_result.json"), "result_manifest": {"centerlines": "center.shp", "surfaces": "surface.shp", "gpkg": "roads.gpkg"}}
                user_pipeline.write_json(state_path, state)
                return state

            args = argparse.Namespace(project_root=str(root), area_id=[], run_id="batch_roads", device="cpu", pixel_size="0", rescale="off", resume=False, continue_on_error=True)
            with patch.object(user_pipeline, "discover_validation_project", return_value=discovered), patch.object(user_pipeline, "extract_project_period", side_effect=complete) as extract_mock, patch.object(user_pipeline, "change") as change_mock:
                result = user_pipeline.extract_project_all(args)
            self.assertEqual(extract_mock.call_count, 3)
            change_mock.assert_not_called()
            self.assertEqual((result["succeeded"], result["failed"], result["status"]), (3, 0, "completed"))
            self.assertTrue(Path(result["batch_root"], "batch_extract_task.json").is_file())

    def test_scope_continue_and_resume_completed_units(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); output = root / "04_成果输出"
            discovered = {"project_root": str(root), "output_root": str(output), "areas": [
                {"area_id": "north", "validation_area": "north.shp", "periods": [{"period": "2021", "source": "a.txt"}, {"period": "2022", "source": "b.txt"}]},
                {"area_id": "south", "validation_area": "south.shp", "periods": [{"period": "2020", "source": "c.txt"}]},
            ]}
            args = argparse.Namespace(project_root=str(root), area_id=["north"], run_id="resume_batch", device="cpu", pixel_size="0", rescale="off", resume=False, continue_on_error=True)

            def flaky(period_args):
                if period_args.period == "2022": raise RuntimeError("synthetic failure")
                state_path = output / "period_extractions" / period_args.area_id / period_args.period / period_args.run_id / "period_task.json"
                product_dir = state_path.parent / "products"; product_dir.mkdir(parents=True, exist_ok=True)
                result = {"centerlines": str(product_dir / "center.shp"), "surfaces": str(product_dir / "surface.shp"), "gpkg": str(product_dir / "roads.gpkg")}
                for value in result.values(): Path(value).touch()
                result_path = state_path.parent / "latest_result.json"; user_pipeline.write_json(result_path, result)
                state = {"status": "completed", "result": str(result_path), "result_manifest": result}; user_pipeline.write_json(state_path, state)
                return state

            with patch.object(user_pipeline, "discover_validation_project", return_value=discovered), patch.object(user_pipeline, "extract_project_period", side_effect=flaky):
                first = user_pipeline.extract_project_all(args)
            self.assertEqual((first["total"], first["succeeded"], first["failed"]), (2, 1, 1))
            args.resume = True
            with patch.object(user_pipeline, "discover_validation_project", return_value=discovered), patch.object(user_pipeline, "extract_project_period", side_effect=flaky) as resumed_mock:
                second = user_pipeline.extract_project_all(args)
            self.assertEqual(resumed_mock.call_count, 1)
            self.assertTrue(second["units"][0]["skipped"])


class ProjectPeriodChangeTests(unittest.TestCase):
    def test_accepts_completed_historical_latest_result_as_period_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            result_path = root / "latest_result.json"
            products = root / "products"; products.mkdir()
            result = {"centerlines": str(products / "center.shp"), "surfaces": str(products / "surface.shp"), "gpkg": str(products / "roads.gpkg")}
            for value in result.values(): Path(value).touch()
            user_pipeline.write_json(result_path, result)

            resolved_path, resolved = user_pipeline._period_result_from_state(str(result_path), str(root), "area-a", "2024")

            self.assertEqual(resolved_path, result_path)
            self.assertEqual(resolved["gpkg"], result["gpkg"])

    def test_adjacent_completed_periods_change_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(); output = root / "04_outputs"
            states = []
            for period in ("2021", "2022"):
                task = output / "period_extractions" / "area-a" / period / "roads"
                products = task / "workspace" / "products"; products.mkdir(parents=True)
                result_path = task / "workspace" / "latest_result.json"
                result = {"centerlines": str(products / "center.shp"), "surfaces": str(products / "surface.shp"), "gpkg": str(products / "roads.gpkg")}
                for value in result.values(): Path(value).touch()
                user_pipeline.write_json(result_path, result)
                state_path = task / "period_task.json"
                user_pipeline.write_json(state_path, {"status": "completed", "result": str(result_path), "input_spec": {"project_root": str(root), "area_id": "area-a", "period": period}})
                states.append(state_path)
            discovered = {"project_root": str(root), "output_root": str(output), "areas": [{"area_id": "area-a", "validation_area": str(root / "area.shp"), "periods": [{"period": "2021"}, {"period": "2022"}], "truths": []}]}
            args = argparse.Namespace(project_root=str(root), area_id="area-a", before_period="2021", after_period="2022", before_state=str(states[0]), after_state=str(states[1]), run_id="change", absolute="2.0", ratio="0.2", tolerance="3.0", resume=False)

            def fake_change(call_args):
                products = Path(call_args.output); products.mkdir(parents=True)
                summary = products / "change_summary.json"
                user_pipeline.write_json(summary, {"added_feature_count": 2, "added_length_m": 1500, "added_area_m2": 300})
                for name in ("added_roads.shp", "removed_roads.shp", "widened_road_parts.shp", "narrowed_road_parts.shp"): (products / name).touch()
                return {"output": str(products), "summary": str(summary), "gpkg": str(products / "road_changes.gpkg")}

            with patch.object(user_pipeline, "discover_validation_project", return_value=discovered), patch.object(user_pipeline, "change", side_effect=fake_change) as change_mock:
                first = user_pipeline.change_project_periods(args)
                args.resume = True
                resumed = user_pipeline.change_project_periods(args)

            self.assertEqual(first["status"], "completed")
            self.assertEqual(resumed["attempt"], 2)
            self.assertEqual(change_mock.call_count, 1)
            self.assertEqual(resumed["result_manifest"]["statistics"]["added"]["feature_count"], 2)
            self.assertEqual(resumed["result_manifest"]["statistics"]["added"]["length_m"], 1500)

    def test_rejects_non_adjacent_period_pair_before_change(self) -> None:
        discovered = {"project_root": str(Path.cwd()), "output_root": str(Path.cwd() / "out"), "areas": [{"area_id": "area-a", "validation_area": "area.shp", "periods": [{"period": "2020"}, {"period": "2021"}, {"period": "2022"}], "truths": []}]}
        args = argparse.Namespace(project_root=str(Path.cwd()), area_id="area-a", before_period="2020", after_period="2022", before_state="missing", after_state="missing", run_id="change", absolute="2", ratio="0.2", tolerance="3", resume=False)
        with patch.object(user_pipeline, "discover_validation_project", return_value=discovered), patch.object(user_pipeline, "change") as change_mock:
            with self.assertRaisesRegex(ValueError, "相邻"):
                user_pipeline.change_project_periods(args)
        change_mock.assert_not_called()


class MapSceneTests(unittest.TestCase):
    def test_accepts_historical_pipeline_result_and_change_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            result = root / "latest_result.json"
            summary = root / "change_summary.json"
            user_pipeline.write_json(result, {"centerlines": str(root / "center.shp"), "surfaces": str(root / "surface.shp")})
            user_pipeline.write_json(summary, {"added_feature_count": 0})

            extraction = user_pipeline._scene_state_result(str(result), root, "period_task.json")
            change = user_pipeline._scene_state_result(str(summary), root, "change_task.json")

            self.assertEqual(extraction["centerlines"], str(root / "center.shp"))
            self.assertEqual(Path(change["layers"]["added"]).name, "added_roads.shp")

    def test_builds_bounded_raster_and_vector_scene_without_models(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from shapely.geometry import LineString, box
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(); imagery = root / "02_images"; imagery.mkdir()
            raster = imagery / "tile.tif"
            data = np.zeros((3, 64, 64), dtype=np.uint8); data[0] = np.arange(64, dtype=np.uint8)[None, :]; data[1] = 110; data[2] = 70
            with rasterio.open(raster, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8", crs="EPSG:3857", transform=from_bounds(0, 0, 640, 640, 64, 64)) as dataset: dataset.write(data)
            txt = imagery / "2024.txt"; txt.write_text(str(raster), encoding="utf-8")
            boundary = root / "area.shp"; gpd.GeoDataFrame(geometry=[box(100, 100, 540, 540)], crs="EPSG:3857").to_file(boundary)
            state_root = root / "04_outputs" / "task"; products = state_root / "products"; products.mkdir(parents=True)
            centerlines = products / "center.shp"; gpd.GeoDataFrame(geometry=[LineString([(120, 120), (520, 520)])], crs="EPSG:3857").to_file(centerlines)
            result = {"centerlines": str(centerlines)}
            state = state_root / "period_task.json"; user_pipeline.write_json(state, {"status": "completed", "result_manifest": result})
            discovered = {"project_root": str(root), "output_root": str(root / "04_outputs"), "areas": [{"area_id": "area-a", "validation_area": str(boundary), "periods": [{"period": "2024", "source": str(txt)}], "truths": []}]}
            args = argparse.Namespace(project_root=str(root), area_id="area-a", period="2024", extraction_state=str(state), change_state="", width=500, height=300)
            with patch.object(user_pipeline, "discover_validation_project", return_value=discovered):
                scene = user_pipeline.map_scene(args)
            self.assertTrue(scene["raster"].startswith("data:image/jpeg;base64,"))
            self.assertLessEqual(scene["width"], 500); self.assertLessEqual(scene["height"], 300)
            centerline_layer = next(item for item in scene["vectors"] if item["kind"] == "centerlines")
            self.assertEqual(centerline_layer["feature_count"], 1)
            self.assertTrue(centerline_layer["paths"][0].startswith("M"))


class ValidationInputTests(unittest.TestCase):
    def test_mock_truth_defaults_to_exact_detection_copy(self) -> None:
        import geopandas as gpd
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "changes.gpkg"
            output = root / "truth.shp"
            changes = gpd.GeoDataFrame(
                {"change_typ": ["added", "narrowed", "removed"]},
                geometry=[box(0, 0, 4, 2), box(10, 0, 14, 2), box(20, 0, 24, 2)],
                crs="EPSG:3857",
            )
            changes.to_file(source, layer="road_changes", driver="GPKG")
            metadata = generate_mock_change_truth.build_mock_truth(source, output)
            truth = gpd.read_file(output)
            self.assertEqual(metadata["generation_mode"], "exact_copy_of_detection")
            self.assertEqual(metadata["omitted_prediction_count"], 0)
            self.assertEqual(metadata["synthetic_missed_count"], 0)
            self.assertEqual(list(truth["BHBM"]), [2, 3, 4])
            self.assertTrue(all(left.equals(right) for left, right in zip(changes.geometry, truth.geometry)))

    def test_multi_area_cli_normalizes_area_period_and_truth_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            areas = []
            for name in ("north", "south"):
                area = root / f"{name}.shp"; area.touch(); areas.append([name, str(area)])
            truth = root / "truth.shp"; truth.touch()
            args = argparse.Namespace(
                validation_area=areas,
                period=[
                    ["north", "2021", "n21.txt"], ["north", "2022", "n22.txt"],
                    ["south", "2021", "s21.txt"], ["south", "2022", "s22.txt"],
                ],
                truth=[["north", "2021", "2022", str(truth)]],
            )
            with patch.object(user_pipeline, "validate_validation_inputs", side_effect=lambda entries, _area: {
                period: Path(source) for period, source in entries
            }):
                grids, area_map, truths = user_pipeline.validation_batch_inputs(args)
            self.assertEqual(set(grids), {"north", "south"})
            self.assertEqual(list(grids["north"]), ["2021", "2022"])
            self.assertEqual(set(area_map), {"north", "south"})
            self.assertEqual(truths[("north", "2021", "2022")], truth.resolve())

    def test_total_evaluation_uses_additive_area_and_centerline_weights(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            entries = []
            for index, values in enumerate(((8, 2, 10, 20, 5, 30), (12, 8, 30, 10, 15, 40))):
                tp, fn, truth_len, pred_len, truth_integral, pred_integral = values
                output = root / f"task_{index}"; output.mkdir()
                summary = output / "change_summary.json"
                summary.write_text(json.dumps({"evaluation": {"metrics": [{
                    "class": "all", "tp_m2": tp, "fp_m2": 0, "fn_m2": fn,
                    "tn_m2": 0, "truth_axis_length_m": truth_len,
                    "predicted_axis_length_m": pred_len,
                    "truth_distance_integral_m2": truth_integral,
                    "predicted_distance_integral_m2": pred_integral,
                }]}}), encoding="utf-8")
                entries.append({"summary": str(summary)})
            manifest = {"change_results": entries}
            result = user_pipeline.aggregate_change_evaluations(manifest, root)
            total = result["metrics"][0]
            self.assertAlmostEqual(total["recall"], 20 / 30)
            self.assertAlmostEqual(total["truth_to_pred_avg_m"], 20 / 40)
            self.assertAlmostEqual(total["pred_to_truth_avg_m"], 70 / 30)
            self.assertAlmostEqual(total["centerline_avg_offset_m"], 90 / 70)
            self.assertEqual(result["evaluated_task_count"], 2)
            self.assertTrue((root / "evaluation_summary.csv").is_file())

    def test_different_image_boundaries_are_allowed_when_each_covers_validation(self) -> None:
        import geopandas as gpd
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_path = root / "area.shp"; area_path.touch()
            first = root / "2021.tif"; second = root / "2022.tif"
            first.touch(); second.touch()
            first_txt = root / "2021.txt"; first_txt.write_text(str(first), encoding="utf-8")
            second_txt = root / "2022.txt"; second_txt.write_text(str(second), encoding="utf-8")
            area = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3857")
            footprints = {first: box(-4, -2, 11, 13), second: box(-2, -4, 15, 12)}
            with patch.object(user_pipeline, "_read_validation_area", return_value=area), \
                    patch.object(user_pipeline, "listed_rasters", side_effect=lambda path: [first if path == first_txt.resolve() else second]), \
                    patch.object(user_pipeline, "_valid_raster_footprint", side_effect=lambda path, target_crs=None: (footprints[path], "EPSG:3857")):
                periods = user_pipeline.validate_validation_inputs(
                    [["2021", str(first_txt)], ["2022", str(second_txt)]], area_path
                )
            self.assertEqual(list(periods), ["2021", "2022"])

    def test_preflight_allows_partial_coverage_for_window_normalization(self) -> None:
        import geopandas as gpd
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_path = root / "area.shp"; area_path.touch()
            first = root / "2021.tif"; second = root / "2022.tif"
            first.touch(); second.touch()
            first_txt = root / "2021.txt"; first_txt.write_text(str(first), encoding="utf-8")
            second_txt = root / "2022.txt"; second_txt.write_text(str(second), encoding="utf-8")
            area = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3857")
            with patch.object(user_pipeline, "_read_validation_area", return_value=area), \
                    patch.object(user_pipeline, "listed_rasters", side_effect=lambda path: [path]), \
                    patch.object(user_pipeline, "_valid_raster_footprint", return_value=(box(0, 0, 8, 10), "EPSG:3857")):
                periods = user_pipeline.validate_validation_inputs(
                    [["2021", str(first_txt)], ["2022", str(second_txt)]], area_path
                )
            self.assertEqual(list(periods), ["2021", "2022"])

    def test_validation_run_normalizes_file_txt_and_directory_to_one_analysis_extent(self) -> None:
        """Regression: raw sources with wider/different bounds must never reach prepare."""
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Polygon
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); output = root / "out"; output.mkdir()
            area_path = root / "area.shp"
            gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (0, 10)])], crs="EPSG:3857").to_file(area_path)
            def raster(path: Path, left: float, crs: str = "EPSG:3857", nodata: int | None = None) -> None:
                data = np.full((1, 12, 16), 7, dtype="uint8")
                if nodata is not None:
                    data[:, 0:2, 4:6] = nodata
                with rasterio.open(path, "w", driver="GTiff", width=16, height=12, count=1, dtype="uint8", crs=crs, transform=from_origin(left, 12, 1, 1), nodata=nodata) as dst:
                    dst.write(data)
            direct = root / "direct.tif"; raster(direct, -3)
            direct_listing = root / "2021.txt"; direct_listing.write_text(str(direct), encoding="utf-8")
            txt_raster = root / "txt.tif"; raster(txt_raster, -2, nodata=0)
            listing = root / "2022.txt"; listing.write_text(str(txt_raster), encoding="utf-8")
            directory_raster = root / "tile_2023.tif"; raster(directory_raster, -1, "EPSG:4326")
            third_listing = root / "2023.txt"; third_listing.write_text(str(directory_raster), encoding="utf-8")
            args = argparse.Namespace(mode="validation", output_root=str(output), run_id="normal", checkpoint="m", config="c", device="cpu", pixel_size="0", rescale="off", absolute="1", ratio="0.1", tolerance="3", validation_area=str(area_path), period=[["2021", str(direct_listing)], ["2022", str(listing)], ["2023", str(third_listing)]], truth=[["2021", "2022", str(area_path)], ["2022", "2023", str(area_path)]], truth_type_field="")
            with patch.object(user_pipeline, "extract", return_value={"centerlines": "c.shp", "surfaces": "s.shp"}), patch.object(user_pipeline, "change", return_value={"output": "change"}), patch.object(user_pipeline, "prepare", wraps=user_pipeline.prepare) as prepare_mock:
                user_pipeline.run_all(args)
            prepared_sources = [Path(call.args[0].source) for call in prepare_mock.call_args_list]
            self.assertNotIn(direct, prepared_sources)
            specs = []
            for source in prepared_sources:
                normalized = user_pipeline.listed_rasters(source)
                self.assertTrue(normalized)
                with rasterio.open(normalized[0]) as dataset:
                    specs.append((dataset.crs.to_string(), dataset.width, dataset.height, tuple(dataset.transform)))
                    self.assertEqual(dataset.dataset_mask()[1, 8], 0, "验证区外像元必须由掩膜标为无效")
            self.assertEqual(len(set(specs)), 1)

    def test_integer_without_nodata_preserves_valid_zero_pixels(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Polygon
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_path = root / "area.geojson"
            gpd.GeoDataFrame(
                geometry=[Polygon([(0, 0), (4, 0), (0, 4)])], crs="EPSG:3857",
            ).to_file(area_path, driver="GeoJSON")
            source = root / "source.tif"
            data = np.full((1, 4, 4), 9, dtype="uint8")
            data[0, 1, 1] = 0  # valid measurement, not NoData
            with rasterio.open(
                source, "w", driver="GTiff", width=4, height=4, count=1,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 4, 1, 1),
            ) as destination:
                destination.write(data)

            normalized = user_pipeline.normalize_validation_sources(
                {"2021": source}, area_path, root / "normalized",
            )
            target = user_pipeline.listed_rasters(normalized["2021"])[0]
            with rasterio.open(target) as dataset:
                self.assertIsNone(dataset.nodata)
                self.assertEqual(dataset.read(1)[1, 1], 0)
                self.assertEqual(dataset.dataset_mask()[1, 1], 255)
                self.assertEqual(dataset.dataset_mask()[0, 3], 0)

    def test_multiple_source_tiles_are_mosaicked_then_split_for_inference(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_path = root / "area.geojson"
            gpd.GeoDataFrame(geometry=[box(0, 0, 4, 2)], crs="EPSG:3857").to_file(area_path, driver="GeoJSON")
            tiles = root / "tiles"; tiles.mkdir()
            for name, left, value in (("left.tif", 0, 1), ("right.tif", 2, 2)):
                with rasterio.open(
                    tiles / name, "w", driver="GTiff", width=2, height=2, count=1,
                    dtype="uint8", crs="EPSG:3857", transform=from_origin(left, 2, 1, 1),
                ) as destination:
                    destination.write(np.full((1, 2, 2), value, dtype="uint8"))

            normalized = user_pipeline.normalize_validation_sources(
                {"2021": tiles}, area_path, root / "normalized", tile_size=2,
            )
            outputs = user_pipeline.listed_rasters(normalized["2021"])
            self.assertEqual(len(outputs), 2)
            self.assertEqual([path.name for path in outputs], ["v0001.tif", "v0002.tif"])
            values = []
            for output in outputs:
                with rasterio.open(output) as dataset:
                    self.assertLessEqual(dataset.width * dataset.height, 4)
                    self.assertTrue((dataset.dataset_mask() == 255).all())
                    values.append(dataset.read(1).tolist())
            self.assertEqual(values, [[[1, 1], [1, 1]], [[2, 2], [2, 2]]])

    def test_existing_alpha_band_is_reused_instead_of_adding_another(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.enums import ColorInterp
        from rasterio.transform import from_origin
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_path = root / "area.geojson"
            gpd.GeoDataFrame(geometry=[box(0, 0, 4, 4)], crs="EPSG:3857").to_file(area_path, driver="GeoJSON")
            source = root / "rgba.tif"
            rgba = np.zeros((4, 4, 4), dtype="uint8")
            rgba[:3] = 25
            rgba[3] = 255
            with rasterio.open(
                source, "w", driver="GTiff", width=4, height=4, count=4,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 4, 1, 1),
            ) as destination:
                destination.write(rgba)
                destination.colorinterp = (
                    ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha,
                )

            normalized = user_pipeline.normalize_validation_sources(
                {"2021": source}, area_path, root / "normalized", tile_size=4,
            )
            target = user_pipeline.listed_rasters(normalized["2021"])[0]
            with rasterio.open(target) as dataset:
                self.assertEqual(dataset.count, 3)
                self.assertTrue((dataset.dataset_mask() == 255).all())

    def test_geographic_validation_area_keeps_projected_image_analysis_crs(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_path = root / "area.geojson"
            image_path = root / "image.tif"
            projected_area = gpd.GeoDataFrame(geometry=[box(0, 0, 4, 4)], crs="EPSG:3857")
            projected_area.to_crs("EPSG:4326").to_file(area_path, driver="GeoJSON")
            with rasterio.open(
                image_path, "w", driver="GTiff", width=4, height=4, count=1,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 4, 1, 1),
            ) as destination:
                destination.write(np.ones((1, 4, 4), dtype="uint8"))

            normalized = user_pipeline.normalize_validation_sources(
                {"2021": image_path, "2022": image_path}, area_path, root / "normalized",
            )

            with rasterio.open(next(normalized["2021"].glob("*.tif"))) as dataset:
                self.assertEqual(dataset.crs.to_epsg(), 3857)
                self.assertAlmostEqual(abs(dataset.transform.a), 1.0, places=5)

    def test_real_nodata_hole_inside_validation_is_preserved_as_mask(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); area_path = root / "area.shp"
            gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3857").to_file(area_path)
            def write(path: Path, hole: bool) -> None:
                data = np.ones((1, 10, 10), dtype="uint8")
                if hole:
                    data[:, 4:6, 4:6] = 0
                with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=1, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 10, 1, 1), nodata=0) as dst:
                    dst.write(data)
            good = root / "good.tif"; hole = root / "hole.tif"; write(good, False); write(hole, True)
            good_txt = root / "2021.txt"; good_txt.write_text(str(good), encoding="utf-8")
            hole_txt = root / "2022.txt"; hole_txt.write_text(str(hole), encoding="utf-8")
            periods = user_pipeline.validate_validation_inputs(
                [["2021", str(good_txt)], ["2022", str(hole_txt)]], area_path,
            )
            normalized = user_pipeline.normalize_validation_sources(
                periods, area_path, root / "normalized", tile_size=10,
            )
            target = user_pipeline.listed_rasters(normalized["2022"])[0]
            with rasterio.open(target) as dataset:
                self.assertEqual(int((dataset.dataset_mask() == 0).sum()), 4)
            marker = user_pipeline.read_json(root / "normalized" / "normalization_complete.json")
            self.assertEqual(marker["coverage"]["2022"]["missing_pixels"], 4)

    def test_probability_mosaic_direct_placement_matches_reproject_pixel_for_pixel(self) -> None:
        import numpy as np
        import rasterio
        from PIL import Image
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images, probabilities = root / "images", root / "probabilities"
            images.mkdir(); probabilities.mkdir()
            rng = np.random.default_rng(20260822)
            for index, left in enumerate((0, 6)):
                image_path = images / f"tile_{index}.tif"
                with rasterio.open(
                    image_path, "w", driver="GTiff", width=6, height=5, count=1,
                    dtype="uint8", crs="EPSG:3857", transform=from_origin(left, 5, 1, 1),
                ) as dataset:
                    dataset.write(np.ones((1, 5, 6), dtype=np.uint8))
                    mask = np.full((5, 6), 255, dtype=np.uint8)
                    mask[2, 2] = 0
                    dataset.write_mask(mask)
                probability = rng.integers(0, 256, size=(5, 6), dtype=np.uint8)
                Image.fromarray(probability).save(
                    probabilities / f"tile_{index}_centerline_probability.png"
                )

            fast_path = root / "fast.tif"
            baseline_path = root / "baseline.tif"
            user_pipeline._write_probability_mosaic(images, probabilities, fast_path)
            user_pipeline._write_probability_mosaic(
                images, probabilities, baseline_path, allow_direct_placement=False,
            )

            with rasterio.open(fast_path) as fast, rasterio.open(baseline_path) as baseline:
                self.assertEqual(fast.transform, baseline.transform)
                np.testing.assert_array_equal(fast.read(1), baseline.read(1))
                np.testing.assert_array_equal(fast.dataset_mask(), baseline.dataset_mask())

    def test_valid_observation_cache_preserves_nodata_hole_and_skips_raster_scan(self) -> None:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images = root / "images"; images.mkdir()
            image_path = images / "tile.tif"
            with rasterio.open(
                image_path, "w", driver="GTiff", width=6, height=6, count=1,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 6, 1, 1),
            ) as dataset:
                dataset.write(np.ones((1, 6, 6), dtype=np.uint8))
                mask = np.full((6, 6), 255, dtype=np.uint8)
                mask[2:4, 2:4] = 0
                dataset.write_mask(mask)

            cache = root / "cache" / "valid_observation.shp"
            copied = root / "products" / "valid_observation.shp"
            user_pipeline._write_valid_observation_area(images, cache)
            with patch.object(
                user_pipeline, "listed_rasters",
                side_effect=AssertionError("cache hit must not scan rasters"),
            ):
                result = user_pipeline._write_valid_observation_area(
                    root / "unused", copied, cached_observation=cache,
                )

            self.assertEqual(result, str(copied.resolve()))
            cached_frame = gpd.read_file(cache)
            copied_frame = gpd.read_file(copied)
            self.assertAlmostEqual(float(cached_frame.geometry.area.sum()), 32.0)
            self.assertAlmostEqual(
                float(cached_frame.geometry.union_all().symmetric_difference(
                    copied_frame.geometry.union_all()
                ).area),
                0.0,
            )


class OneClickPipelineTests(unittest.TestCase):
    def test_extracts_each_period_and_compares_adjacent_periods(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            layout = {
                "grid_a": {"2021": source / "a21.txt", "2022": source / "a22.txt"},
                "grid_b": {"2020": source / "b20.txt", "2022": source / "b22.txt", "2023": source / "b23.txt"},
            }
            args = argparse.Namespace(
                source_root=str(source), output_root=str(output), run_id="test_run",
                checkpoint="model.ckpt", config="config.yaml", device="cpu",
                pixel_size="0.0", rescale="off", absolute="1.0", ratio="0.1", tolerance="3.0",
            )

            with patch.object(user_pipeline, "discover_grid_periods", return_value=layout), \
                    patch.object(user_pipeline, "prepare") as prepare_mock, \
                    patch.object(user_pipeline, "extract", return_value={"centerlines": "c.shp", "surfaces": "s.shp"}) as extract_mock, \
                    patch.object(user_pipeline, "change", return_value={"output": "changes"}) as change_mock:
                result = user_pipeline.run_all(args)

            self.assertEqual(prepare_mock.call_count, 5)
            self.assertEqual(extract_mock.call_count, 5)
            self.assertEqual(change_mock.call_count, 3)
            self.assertEqual(result["grid_count"], 2)
            self.assertEqual(result["period_count"], 5)
            self.assertEqual(result["change_count"], 3)
            for period_result in result["period_results"]:
                self.assertEqual(
                    set(period_result["previews"]),
                    {"centerline", "surface", "fusion", "width"},
                )
                self.assertEqual(period_result["review"]["manual_item_count"], 0)
            for change_result in result["change_results"]:
                self.assertEqual(change_result["previews"], {})
            latest_path = root / "_work" / "tasks" / "latest_pipeline.json"
            self.assertTrue(latest_path.is_file())
            latest = user_pipeline.read_json(latest_path)
            self.assertEqual(len(latest["period_results"]), 5)
            self.assertIn("previews", latest["period_results"][0])
            self.assertIn("review", latest["period_results"][0])
            self.assertEqual(
                [(entry["before_period"], entry["after_period"]) for entry in result["change_results"] if entry["grid"] == "grid_b"],
                [("2020", "2022"), ("2022", "2023")],
            )

    def test_cli_defaults_to_validation_mode_and_keeps_grid_mode(self) -> None:
        parsed = user_pipeline.parser().parse_args([
            "all", "--output-root", "out", "--validation-area", "area.shp",
            "--period", "2021", "one.txt", "--period", "2022", "two.txt",
            "--truth", "2021", "2022", "truth.shp", "--checkpoint", "model", "--config", "config",
        ])
        self.assertEqual(parsed.mode, "validation")
        grid = user_pipeline.parser().parse_args([
            "all", "--mode", "grid", "--source-root", "legacy", "--output-root", "out", "--checkpoint", "model", "--config", "config",
        ])
        self.assertEqual(grid.mode, "grid")

    def test_validation_run_forwards_area_and_pair_truth_to_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "images"
            output = root / "output"
            source.mkdir()
            validation = root / "area.shp"
            truth = root / "truth.shp"
            validation.touch(); truth.touch()
            args = argparse.Namespace(
                mode="validation", output_root=str(output), run_id="validation_run",
                checkpoint="model.ckpt", config="config.yaml", device="cpu",
                pixel_size="0.0", rescale="off", absolute="1.0", ratio="0.1", tolerance="3.0",
                validation_area=str(validation),
                period=[["2021", str(source / "2021.tif")], ["2022", str(source / "2022.tif")]],
                truth=[["2021", "2022", str(truth)]], truth_type_field="change_kind",
            )
            with patch.object(user_pipeline, "validate_validation_inputs", return_value={
                "2021": source / "2021.tif", "2022": source / "2022.tif",
            }), patch.object(user_pipeline, "normalize_validation_sources", return_value={
                "2021": source / "analysis_2021", "2022": source / "analysis_2022",
            }), patch.object(user_pipeline, "prepare"), \
                    patch.object(user_pipeline, "extract", return_value={"centerlines": "c.shp", "surfaces": "s.shp"}), \
                    patch.object(user_pipeline, "change", return_value={"output": "changes"}) as change_mock:
                user_pipeline.run_all(args)
            change_args = change_mock.call_args.args[0]
            self.assertEqual(change_args.truth, str(truth.resolve()))
            self.assertEqual(change_args.validation_area, str(validation.resolve()))
            self.assertEqual(change_args.truth_type_field, "change_kind")

    def test_production_validation_run_does_not_require_truth(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); source = root / "images"; source.mkdir(); output = root / "out"
            area = root / "area.shp"; area.touch()
            args = argparse.Namespace(
                mode="validation", output_root=str(output), run_id="production",
                checkpoint="m", config="c", device="cpu", pixel_size="0", rescale="off",
                absolute="1", ratio="0.1", tolerance="3", validation_area=str(area),
                period=[["2021", str(source / "2021.tif")], ["2022", str(source / "2022.tif")]],
                truth=[], truth_type_field="", no_evaluation=True,
            )
            with patch.object(user_pipeline, "validate_validation_inputs", return_value={
                "2021": source / "2021.tif", "2022": source / "2022.tif",
            }), patch.object(user_pipeline, "normalize_validation_sources", return_value={
                "2021": source / "n1", "2022": source / "n2",
            }), patch.object(user_pipeline, "prepare"), \
                    patch.object(user_pipeline, "extract", return_value={"centerlines": "c", "surfaces": "s"}), \
                    patch.object(user_pipeline, "change", return_value={"output": "changes"}) as change_mock:
                result = user_pipeline.run_all(args)
            self.assertFalse(result["evaluation_enabled"])
            self.assertEqual(change_mock.call_args.args[0].truth, "")

    def test_resume_skips_complete_periods_and_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); source = root / "source"; source.mkdir(); output = root / "output"
            layout = {"g": {"2021": source / "2021.tif", "2022": source / "2022.tif"}}
            common = dict(
                mode="grid", source_root=str(source), output_root=str(output), run_id="resume_run",
                checkpoint="m", config="c", device="cpu", pixel_size="0", rescale="off",
                absolute="1", ratio="0.1", tolerance="3", truth_type_field="",
                continue_on_error=True,
            )

            def fake_extract(extract_args):
                workspace = Path(extract_args.workspace); products = workspace / "products"; products.mkdir(parents=True)
                center = products / "center.shp"; surface = products / "surface.shp"; gpkg = products / "roads.gpkg"
                for path in (center, surface, gpkg): path.touch()
                result = {"workspace": str(workspace), "run_root": str(workspace / "run"), "centerlines": str(center), "surfaces": str(surface), "gpkg": str(gpkg)}
                user_pipeline.write_json(workspace / "latest_result.json", result)
                return result

            def fake_change(change_args):
                directory = Path(change_args.output); directory.mkdir(parents=True)
                summary = directory / "change_summary.json"; user_pipeline.write_json(summary, {})
                return {"output": str(directory), "summary": str(summary)}

            first_args = argparse.Namespace(**common, resume=False)
            with patch.object(user_pipeline, "discover_grid_periods", return_value=layout), \
                    patch.object(user_pipeline, "prepare"), \
                    patch.object(user_pipeline, "extract", side_effect=fake_extract), \
                    patch.object(user_pipeline, "change", side_effect=fake_change):
                first = user_pipeline.run_all(first_args)
            self.assertEqual(first["status"], "completed")

            second_args = argparse.Namespace(**common, resume=True)
            with patch.object(user_pipeline, "discover_grid_periods", return_value=layout), \
                    patch.object(user_pipeline, "prepare") as prepare_mock, \
                    patch.object(user_pipeline, "extract") as extract_mock, \
                    patch.object(user_pipeline, "change") as change_mock:
                resumed = user_pipeline.run_all(second_args)
            prepare_mock.assert_not_called(); extract_mock.assert_not_called(); change_mock.assert_not_called()
            self.assertEqual(resumed["attempt"], 2)
            self.assertEqual(resumed["period_count"], 2)
            self.assertEqual(resumed["change_count"], 1)

            changed_args = argparse.Namespace(**{**common, "absolute": "2"}, resume=True)
            with patch.object(user_pipeline, "discover_grid_periods", return_value=layout), \
                    patch.object(user_pipeline, "prepare") as prepare_mock, \
                    patch.object(user_pipeline, "extract") as extract_mock, \
                    patch.object(user_pipeline, "change", side_effect=fake_change) as change_mock:
                changed = user_pipeline.run_all(changed_args)
            prepare_mock.assert_not_called(); extract_mock.assert_not_called()
            self.assertEqual(change_mock.call_count, 1)
            self.assertTrue(changed["invalidation_plan"]["threshold_changed"])

    def test_continue_on_error_records_failure_and_dependency_skip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); source = root / "source"; source.mkdir(); output = root / "output"
            layout = {"g": {"2021": source / "2021.tif", "2022": source / "2022.tif"}}
            args = argparse.Namespace(
                mode="grid", source_root=str(source), output_root=str(output), run_id="failure_run",
                checkpoint="m", config="c", device="cpu", pixel_size="0", rescale="off",
                absolute="1", ratio="0.1", tolerance="3", truth_type_field="",
                continue_on_error=True, resume=False,
            )

            def extract_once(extract_args):
                if "2021" in str(extract_args.workspace):
                    raise RuntimeError("synthetic period failure")
                workspace = Path(extract_args.workspace); products = workspace / "products"; products.mkdir(parents=True)
                center = products / "center.shp"; surface = products / "surface.shp"; gpkg = products / "roads.gpkg"
                for path in (center, surface, gpkg): path.touch()
                result = {"workspace": str(workspace), "run_root": str(workspace / "run"), "centerlines": str(center), "surfaces": str(surface), "gpkg": str(gpkg)}
                user_pipeline.write_json(workspace / "latest_result.json", result)
                return result

            with patch.object(user_pipeline, "discover_grid_periods", return_value=layout), \
                    patch.object(user_pipeline, "prepare"), \
                    patch.object(user_pipeline, "extract", side_effect=extract_once), \
                    patch.object(user_pipeline, "change") as change_mock:
                result = user_pipeline.run_all(args)
            change_mock.assert_not_called()
            self.assertEqual(result["status"], "completed_with_errors")
            self.assertEqual({item["status"] for item in result["failures"]}, {"failed", "skipped_dependency"})
            self.assertTrue((Path(result["job_root"]) / "task_report.csv").is_file())


class ManifestMetadataTests(unittest.TestCase):
    def test_change_indexes_evaluation_metrics_when_truth_evaluation_creates_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); output = root / "change"; output.mkdir()
            before = root / "before.json"; after = root / "after.json"
            user_pipeline.write_json(before, {"centerlines": "before.shp", "surfaces": "before_surface.shp"})
            user_pipeline.write_json(after, {"centerlines": "after.shp", "surfaces": "after_surface.shp"})
            metrics = output / "evaluation_metrics.csv"; metrics.touch()
            args = argparse.Namespace(before_result=str(before), after_result=str(after), output=str(output), before_period="2021", after_period="2022", absolute="1", ratio="0.1", tolerance="3", truth="truth.geojson", validation_area="area.geojson", truth_type_field="kind")
            with patch.object(user_pipeline, "run_command") as command_mock:
                result = user_pipeline.change(args)
            self.assertEqual(result["evaluation_metrics"], str(metrics))
            command = command_mock.call_args.args[0]
            self.assertEqual(command[-6:], ["--truth", "truth.geojson", "--validation-area", "area.geojson", "--truth-type-field", "kind"])

    def test_preview_helper_uses_priority_and_expresses_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_root = Path(raw)
            preferred_centerline = run_root / "inference" / "road_graphs" / "tile" / "viz" / "center.png"
            fallback_centerline = run_root / "products" / "road_centerlines.png"
            preferred_surface = run_root / "surfaces" / "masks" / "tile_mask.png"
            fallback_surface = run_root / "width_review" / "tile_molra_clean_mask.png"
            fusion = run_root / "width_review" / "tile_review_demo.png"
            final_fusion = run_root / "products" / "road_overview.png"
            width = run_root / "finalized" / "tile_optimized_viz.png"
            for path in (preferred_centerline, fallback_centerline, preferred_surface, fallback_surface, fusion, final_fusion, width):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            previews = user_pipeline.discover_preview_paths(run_root)

            self.assertEqual(previews["centerline"], str(preferred_centerline.resolve()))
            self.assertEqual(previews["surface"], str(preferred_surface.resolve()))
            self.assertEqual(previews["fusion"], str(final_fusion.resolve()))
            self.assertEqual(previews["width"], str(width.resolve()))

            preferred_surface.unlink()
            width.unlink()
            fallback_previews = user_pipeline.discover_preview_paths(run_root)
            self.assertEqual(fallback_previews["surface"], str(fallback_surface.resolve()))
            self.assertEqual(fallback_previews["width"], str(final_fusion.resolve()))

            missing = user_pipeline.discover_preview_paths(run_root / "missing")
            self.assertIsNone(missing["centerline"])
            self.assertIsNone(missing["surface"])
            self.assertIsNone(missing["fusion"])
            self.assertIsNone(missing["width"])

    def test_review_helper_aggregates_manual_items_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_root = Path(raw)
            review_dir = run_root / "width_review"
            review_dir.mkdir(parents=True)
            (review_dir / "review_decisions.csv").touch()
            (review_dir / "a_summary.json").write_text(
                '{"manual_review_item_count": 2}', encoding="utf-8"
            )
            (review_dir / "b_summary.json").write_text(
                '{"manual_review_item_count": 3}', encoding="utf-8"
            )

            review = user_pipeline.build_review_metadata(run_root)

            self.assertTrue(review["available"])
            self.assertEqual(review["directory"], str(review_dir.resolve()))
            self.assertEqual(review["decisions"], str((review_dir / "review_decisions.csv").resolve()))
            self.assertEqual(review["manual_item_count"], 5)

    def test_change_preview_helper_returns_only_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            self.assertIsNone(user_pipeline.discover_change_preview(output))
            preview = output / "change_preview.png"
            preview.touch()
            self.assertEqual(user_pipeline.discover_change_preview(output), str(preview.resolve()))

    def test_change_manifest_records_formal_and_review_previews(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            formal = output / "change_preview.png"
            review = output / "review_preview.png"
            formal.touch()
            review.touch()

            payload = user_pipeline._ensure_change_manifest_fields(
                {"output": str(output), "previews": {"legacy": "kept"}}, output,
            )

            self.assertEqual(payload["previews"]["change"], str(formal.resolve()))
            self.assertEqual(payload["previews"]["review_change"], str(review.resolve()))
            self.assertEqual(payload["previews"]["legacy"], "kept")

            review.unlink()
            legacy_payload = user_pipeline._ensure_change_manifest_fields(payload, output)
            self.assertEqual(legacy_payload["previews"]["change"], str(formal.resolve()))
            self.assertNotIn("review_change", legacy_payload["previews"])

    def test_fusion_metadata_reports_automatic_and_manual_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_root = Path(raw)
            final_dir = run_root / "finalized"
            products = run_root / "products"
            final_dir.mkdir()
            products.mkdir()
            (final_dir / "a_optimized_summary.json").write_text(
                '{"original_edge_count": 10, "optimized_edge_count": 14, '
                '"auto_accepted_gap_count": 1, "auto_accepted_surface_count": 2, '
                '"geometry_edited": true}',
                encoding="utf-8",
            )
            (products / "final_quality_report.json").write_text(
                '{"fusion": {"global_surface_gap_count": 3, '
                '"global_endpoint_gap_count": 1, "global_edge_attachment_count": 2}}',
                encoding="utf-8",
            )
            fusion = user_pipeline.build_fusion_metadata(final_dir)
            self.assertEqual(fusion["added_edge_count"], 4)
            self.assertEqual(fusion["local_gap_count"], 1)
            self.assertEqual(fusion["global_gap_count"], 3)
            self.assertEqual(fusion["global_endpoint_gap_count"], 1)
            self.assertEqual(fusion["global_edge_attachment_count"], 2)
            self.assertEqual(fusion["auto_gap_count"], 4)
            self.assertEqual(fusion["auto_surface_count"], 2)
            self.assertEqual(fusion["geometry_edited_tile_count"], 1)


class ApplyCenterlineEditsTests(unittest.TestCase):
    def test_global_edit_uses_authoritative_gpkg_without_stitching(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = root / "workspace"
            run_root = workspace / "runs" / "roads"
            width_dir = run_root / "width_review"
            final_dir = run_root / "finalized"
            products = run_root / "products"
            images = workspace / "images"
            edited_dir = run_root / "centerline_edit"
            for directory in (width_dir, final_dir, products, images, edited_dir):
                directory.mkdir(parents=True, exist_ok=True)
            global_centerlines = edited_dir / "global_edited_centerlines.gpkg"
            global_centerlines.touch()
            user_pipeline.write_json(edited_dir / "edited_manifest.json", {
                "editing_scope": "period_final_fused_centerlines_global_once",
                "global_centerlines": str(global_centerlines),
                "affected_tiles": ["tile_a"],
            })
            user_pipeline.write_json(workspace / "input_manifest.json", {"images": str(images)})
            result_path = workspace / "latest_result.json"
            result = {
                "workspace": str(workspace), "run_root": str(run_root),
                "width_review": str(width_dir), "final_dir": str(final_dir),
                "gpkg": str(products / "roads.gpkg"),
                "centerlines": str(products / "road_centerlines.shp"),
                "surfaces": str(products / "road_surfaces.shp"),
                "review": {"edited_directory": str(edited_dir)},
            }
            user_pipeline.write_json(result_path, result)
            args = argparse.Namespace(result=str(result_path), edited_dir="", pipeline_manifest="")

            with patch.object(user_pipeline, "run_command") as run_mock:
                updated = user_pipeline.apply_centerline_edits(args)

            self.assertEqual(run_mock.call_count, 3)
            commands = [call.args[0] for call in run_mock.call_args_list]
            self.assertIn("apply-global-edit", commands[0])
            self.assertNotIn("stitch-edited", commands[0])
            self.assertIn("--only-stem", commands[0])
            self.assertIn("tile_a", commands[0])
            self.assertIn("--only-stem", commands[1])
            stitched_index = commands[2].index("--stitched-centerlines") + 1
            self.assertEqual(commands[2][stitched_index], str(global_centerlines.resolve()))
            self.assertEqual(updated["manual_edit"]["canonical_centerlines"], str(global_centerlines.resolve()))
            self.assertTrue(updated["manual_edit"]["canonical_centerlines_authoritative"])

    def test_applies_saved_graph_then_remeasures_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = root / "workspace"
            run_root = workspace / "runs" / "roads"
            width_dir = run_root / "width_review"
            final_dir = run_root / "finalized"
            products = run_root / "products"
            images = workspace / "images"
            edited_dir = run_root / "centerline_edit"
            for directory in (width_dir, final_dir, products, images, edited_dir):
                directory.mkdir(parents=True, exist_ok=True)
            (edited_dir / "tile_edited_graph.p").touch()
            user_pipeline.write_json(workspace / "input_manifest.json", {"images": str(images)})
            result_path = workspace / "latest_result.json"
            result = {
                "workspace": str(workspace), "run_root": str(run_root),
                "width_review": str(width_dir), "final_dir": str(final_dir),
                "gpkg": str(products / "roads.gpkg"),
                "centerlines": str(products / "road_centerlines.shp"),
                "surfaces": str(products / "road_surfaces.shp"),
                "review": {"edited_directory": str(edited_dir)},
            }
            user_pipeline.write_json(result_path, result)
            pipeline_path = root / "latest_pipeline.json"
            user_pipeline.write_json(pipeline_path, {
                "period_results": [{"grid": "g", "period": "2026", "result": str(result_path)}]
            })
            args = argparse.Namespace(
                result=str(result_path), edited_dir="", pipeline_manifest=str(pipeline_path)
            )

            with patch.object(user_pipeline, "run_command") as run_mock, \
                    patch.object(user_pipeline, "extract") as extract_mock:
                updated = user_pipeline.apply_centerline_edits(args)

            self.assertEqual(run_mock.call_count, 3)
            commands = [call.args[0] for call in run_mock.call_args_list]
            self.assertIn("stitch-edited", commands[0])
            self.assertIn("--edited-dir", commands[1])
            self.assertIn("--stitched-centerlines", commands[2])
            self.assertTrue(updated["manual_edit"]["applied"])
            extract_mock.assert_not_called()
            pipeline = user_pipeline.read_json(pipeline_path)
            self.assertTrue(pipeline["period_results"][0]["manual_edit"]["applied"])

    def test_middle_period_edit_reruns_only_its_two_adjacent_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "output"; job = output / "run"
            result_paths = []
            periods = []
            for period in ("2021", "2022", "2023", "2024"):
                workspace = job / "periods" / period
                run_root = workspace / "runs" / "roads"
                edited_dir = run_root / "centerline_edit"
                for directory in (workspace / "images", run_root / "width_review", run_root / "finalized", run_root / "products", edited_dir):
                    directory.mkdir(parents=True, exist_ok=True)
                (edited_dir / "tile_edited_graph.p").touch()
                user_pipeline.write_json(workspace / "input_manifest.json", {"images": str(workspace / "images")})
                result_path = workspace / "latest_result.json"
                result = {"workspace": str(workspace), "run_root": str(run_root), "width_review": str(run_root / "width_review"), "final_dir": str(run_root / "finalized"), "gpkg": str(run_root / "products" / "roads.gpkg"), "centerlines": str(run_root / "products" / "center.shp"), "surfaces": str(run_root / "products" / "surface.shp"), "review": {"edited_directory": str(edited_dir)}}
                user_pipeline.write_json(result_path, result)
                result_paths.append(result_path)
                periods.append({"grid": "validation", "period": period, "result": str(result_path), **result})
            pipeline = {"job_root": str(job), "period_results": periods, "change_results": [
                {"grid": "validation", "before_period": "2021", "after_period": "2022", "output": str(job / "changes" / "21_22")},
                {"grid": "validation", "before_period": "2022", "after_period": "2023", "output": str(job / "changes" / "22_23")},
                {"grid": "validation", "before_period": "2023", "after_period": "2024", "output": str(job / "changes" / "23_24")},
            ]}
            job_manifest = job / "pipeline_result.json"; latest = output / "latest_pipeline.json"
            user_pipeline.write_json(job_manifest, pipeline); user_pipeline.write_json(latest, pipeline)
            untouched = dict(pipeline["change_results"][2])
            with patch.object(user_pipeline, "run_command"), patch.object(user_pipeline, "change", return_value={"output": "rerun"}) as change_mock:
                updated_period = user_pipeline.apply_centerline_edits(argparse.Namespace(result=str(result_paths[1]), edited_dir="", pipeline_manifest=str(latest)))
            self.assertEqual(change_mock.call_count, 2)
            self.assertEqual(updated_period["change_rerun_count"], 2)
            self.assertEqual(
                [(entry["before_period"], entry["after_period"]) for entry in updated_period["change_reruns"]],
                [("2021", "2022"), ("2022", "2023")],
            )
            self.assertEqual(
                [(call.args[0].before_period, call.args[0].after_period) for call in change_mock.call_args_list],
                [("2021", "2022"), ("2022", "2023")],
            )
            updated_job = user_pipeline.read_json(job_manifest)
            updated_latest = user_pipeline.read_json(latest)
            self.assertEqual(updated_job, updated_latest)
            self.assertEqual(updated_latest["period_results"][1]["change_rerun_count"], 2)
            self.assertEqual(updated_latest["change_results"][2], untouched)
            with patch.object(user_pipeline, "run_command"), patch.object(user_pipeline, "change", return_value={"output": "rerun-start"}) as start_mock:
                user_pipeline.apply_centerline_edits(argparse.Namespace(result=str(result_paths[0]), edited_dir="", pipeline_manifest=str(latest)))
            self.assertEqual(
                [(call.args[0].before_period, call.args[0].after_period) for call in start_mock.call_args_list],
                [("2021", "2022")],
            )
            with patch.object(user_pipeline, "run_command"), patch.object(user_pipeline, "change", return_value={"output": "rerun-end"}) as end_mock:
                user_pipeline.apply_centerline_edits(argparse.Namespace(result=str(result_paths[3]), edited_dir="", pipeline_manifest=str(latest)))
            self.assertEqual(
                [(call.args[0].before_period, call.args[0].after_period) for call in end_mock.call_args_list],
                [("2023", "2024")],
            )


class DependencyInvalidationTests(unittest.TestCase):
    @staticmethod
    def spec() -> dict:
        return {
            "pipeline_version": user_pipeline.PIPELINE_VERSION,
            "mode": "validation", "validation_area": {"north": {"path": "area.shp"}},
            "grids": {"north": {
                "2021": {"path": "2021.txt", "size": 1},
                "2022": {"path": "2022.txt", "size": 2},
                "2024": {"path": "2024.txt", "size": 4},
            }},
            "checkpoint": {"path": "model.ckpt"}, "config": {"path": "config.yaml"},
            "device": "cpu", "pixel_size": "0", "rescale": "off", "junction_node_mode": "sparse",
            "absolute": "2", "ratio": "0.2", "tolerance": "3",
            "truths": {}, "truth_type_field": "", "evaluation_enabled": False,
        }

    def test_threshold_change_invalidates_changes_but_not_extraction(self) -> None:
        prior = self.spec(); current = json.loads(json.dumps(prior)); current["absolute"] = "3"
        plan = user_pipeline.dependency_invalidation_plan(prior, current)
        self.assertEqual(plan["periods"], [])
        self.assertEqual(plan["changes"], [
            ("north", "2021", "2022"), ("north", "2022", "2024"),
        ])
        self.assertTrue(plan["threshold_changed"])

    def test_one_period_input_invalidates_only_it_and_adjacent_changes(self) -> None:
        prior = self.spec(); current = json.loads(json.dumps(prior)); current["grids"]["north"]["2022"]["size"] = 99
        plan = user_pipeline.dependency_invalidation_plan(prior, current)
        self.assertEqual(plan["periods"], [("north", "2022")])
        self.assertEqual(plan["changes"], [
            ("north", "2021", "2022"), ("north", "2022", "2024"),
        ])

    def test_period_rerun_without_cascade_marks_pairs_stale(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); job = root / "run"; job.mkdir()
            manifest_path = root / "latest_pipeline.json"
            manifest = {
                "job_root": str(job),
                "period_results": [
                    {"grid": "north", "period": year, "result": str(job / year / "latest_result.json")}
                    for year in ("2021", "2022", "2024")
                ],
                "change_results": [
                    {"grid": "north", "before_period": "2021", "after_period": "2022"},
                    {"grid": "north", "before_period": "2022", "after_period": "2024"},
                ],
            }
            user_pipeline.write_json(manifest_path, manifest)
            with patch.object(user_pipeline, "_rerun_period_entry", return_value={"result": "updated"}), \
                    patch.object(user_pipeline, "_rerun_change_entry") as change_mock, \
                    patch.object(user_pipeline, "_write_task_report"):
                result = user_pipeline.rerun_pipeline_period(argparse.Namespace(
                    pipeline_manifest=str(manifest_path), grid="north", period="2022", update_related=False,
                ))
            updated = user_pipeline.read_json(manifest_path)
            self.assertFalse(result["updated_related"])
            change_mock.assert_not_called()
            self.assertEqual([entry["status"] for entry in updated["change_results"]], ["stale", "stale"])
            self.assertEqual(updated["temporal_status"], "stale")

    def test_period_rerun_with_cascade_is_serial_and_refreshes_downstream_last(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); job = root / "run"; job.mkdir(); manifest_path = root / "latest_pipeline.json"
            user_pipeline.write_json(manifest_path, {
                "job_root": str(job),
                "period_results": [{"grid": "north", "period": year} for year in ("2021", "2022", "2024")],
                "change_results": [],
            })
            order = []
            with patch.object(user_pipeline, "_rerun_period_entry", side_effect=lambda *_args: order.append("period") or {}), \
                    patch.object(user_pipeline, "_rerun_change_entry", side_effect=lambda _m, _g, b, a: order.append(f"change:{b}:{a}") or {}), \
                    patch.object(user_pipeline, "_refresh_manifest_downstream", side_effect=lambda _m: order.append("downstream")), \
                    patch.object(user_pipeline, "_write_task_report"):
                user_pipeline.rerun_pipeline_period(argparse.Namespace(
                    pipeline_manifest=str(manifest_path), grid="north", period="2022", update_related=True,
                ))
            self.assertEqual(order, ["period", "change:2021:2022", "change:2022:2024", "downstream"])


if __name__ == "__main__":
    unittest.main()
