from __future__ import annotations

import io
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.backend_client import BackendClient, BackendEvent, parse_backend_line
from app.project_manager import ProjectManager
from app.task_manager import TaskManager

CODE_ROOT = Path(__file__).resolve().parents[1]


class _BackendStub:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: (args, kwargs)


class BackendClientArchitectureTests(unittest.TestCase):
    def test_structured_stdout_is_parsed_into_backend_event(self) -> None:
        line = '__SAMROAD_USER__{"kind":"stage","stage":"道路提取","grid":"area1","period":"2021"}'
        event = parse_backend_line(line)
        self.assertIsInstance(event, BackendEvent)
        self.assertEqual((event.kind, event.stage, event.area_id, event.period), (
            "stage", "道路提取", "area1", "2021",
        ))
        self.assertIsNone(parse_backend_line("ordinary backend log"))

    def test_command_line_owns_python_backend_and_cli_prefix(self) -> None:
        client = BackendClient(
            app_root=CODE_ROOT, backend_script=CODE_ROOT / "user_pipeline.py",
            python_executable="portable-python.exe",
        )
        self.assertEqual(
            client.command_line(["doctor"]),
            ["portable-python.exe", str((CODE_ROOT / "user_pipeline.py").resolve()), "doctor"],
        )

    def test_worker_separates_structured_events_and_plain_logs(self) -> None:
        normal, priority = queue.Queue(), queue.Queue()
        client = BackendClient(
            app_root=CODE_ROOT, event_queue=normal, priority_queue=priority,
        )
        process = mock.Mock()
        process.stdout = io.StringIO(
            '__SAMROAD_USER__{"kind":"progress","stage":"道路提取","completed":1,"total":2}\n'
            "plain line\n"
        )
        process.wait.return_value = 0
        process.poll.return_value = None
        with mock.patch("app.backend_client.subprocess.Popen", return_value=process):
            thread = client.start(["doctor"])
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        event_kind, event = priority.get_nowait()
        self.assertEqual(event_kind, "backend_event")
        self.assertEqual(event.kind, "progress")
        self.assertEqual(normal.get_nowait(), ("backend_log", "plain line"))
        self.assertEqual(priority.get_nowait(), ("done", "0"))


class ProjectManagerArchitectureTests(unittest.TestCase):
    def test_project_config_round_trip_remains_optional_and_compatible(self) -> None:
        manager = ProjectManager()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.assertEqual(manager.read_config(root), {})
            payload = {
                "version": 3, "project_root": str(root),
                "external_data_sources": ["D:/data"],
                "validation_areas": [["area1", "D:/area1.shp"]],
            }
            path = manager.save_config(root, payload)
            self.assertEqual(path.name, "project_config.json")
            restored = manager.read_config(root)
            self.assertEqual(restored["version"], payload["version"])
            self.assertEqual(Path(restored["project_root"]), root.resolve())
            self.assertEqual(Path(restored["external_data_sources"][0]), Path("D:/data"))
            self.assertEqual(Path(restored["validation_areas"][0][1]), Path("D:/area1.shp"))

    def test_project_owned_paths_are_stored_relative_and_rebased_after_move(self) -> None:
        manager = ProjectManager()
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            old_root = workspace / "old" / "project"
            moved_root = workspace / "moved" / "project"
            moved_root.mkdir(parents=True)
            internal_area = moved_root / "data" / "area.shp"
            internal_area.parent.mkdir()
            internal_area.touch()
            payload = {
                "version": 3,
                "project_root": str(old_root),
                "output_root": str(old_root / "成果输出"),
                "external_data_sources": [],
                "validation_areas": [["area", str(old_root / "data" / "area.shp")]],
            }
            (moved_root / "project_config.json").write_text(
                json.dumps(payload), encoding="utf-8",
            )

            restored = manager.read_config(moved_root)

            self.assertEqual(restored["project_root"], str(moved_root.resolve()))
            self.assertEqual(restored["output_root"], str((moved_root / "成果输出").resolve()))
            self.assertEqual(restored["validation_areas"][0][1], str(internal_area.resolve()))
            manager.save_config(moved_root, restored)
            stored = json.loads((moved_root / "project_config.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["project_root"], ".")
            self.assertEqual(stored["output_root"], "成果输出")
            self.assertEqual(stored["validation_areas"][0][1], str(Path("data") / "area.shp"))

    def test_external_root_relocation_updates_all_configured_inputs(self) -> None:
        manager = ProjectManager()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            old = root / "old-data"
            new = root / "new-data"
            image_dir = new / "area"
            image_dir.mkdir(parents=True)
            area = image_dir / "boundary.shp"
            listing = image_dir / "2022.txt"
            truth = image_dir / "2021_to_2022.shp"
            image = new / "tiles" / "image.tif"
            image.parent.mkdir()
            for path in (area, truth, image):
                path.touch()
            listing.write_text(str(old / "tiles" / image.name), encoding="utf-8")
            payload = {
                "external_data_sources": [str(old)],
                "validation_areas": [["area", str(old / "area" / area.name)]],
                "area_periods": {"area": [["2022", str(old / "area" / listing.name)]]},
                "area_truths": [["area", "2021", "2022", str(old / "area" / truth.name)]],
                "external_scan_cache": {
                    str(old): {"root": str(old), "candidates": {"shp": [str(old / "area" / area.name)], "txt": []}},
                },
            }

            relocated = manager.relocate_paths(payload, old, new)

            self.assertEqual(manager.path_issues(relocated), [])
            self.assertEqual(relocated["external_data_sources"], [str(new.resolve())])
            self.assertEqual(relocated["path_relocations"][str(old.resolve())], str(new.resolve()))
            self.assertIn(str(new.resolve()), relocated["external_scan_cache"])

    def test_legacy_manifest_paths_are_rebased_to_the_moved_project(self) -> None:
        manager = ProjectManager()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            old_project = root / "old-project"
            moved_project = root / "moved-project"
            output = moved_project / "成果输出"
            result = moved_project / "_work" / "tasks" / "runs" / "run1" / "result.json"
            result.parent.mkdir(parents=True)
            output.mkdir(parents=True)
            result.write_text("{}", encoding="utf-8")
            latest = moved_project / "_work" / "tasks" / "latest_pipeline.json"
            latest.parent.mkdir(parents=True, exist_ok=True)
            latest.write_text(json.dumps({
                "project_root": str(old_project),
                "output_root": str(old_project / "成果输出"),
                "period_results": [{
                    "grid": "area", "period": "2022", "status": "completed",
                    "result": str(old_project / result.relative_to(moved_project)),
                }],
                "change_results": [],
            }), encoding="utf-8")

            manifest, manifest_path = manager.result_context(moved_project, output)

            self.assertEqual(manifest_path, latest)
            self.assertEqual(manifest["project_root"], str(moved_project.resolve()))
            self.assertEqual(Path(manifest["period_results"][0]["result"]), result.resolve())

    def test_manifest_result_models_are_built_outside_tk(self) -> None:
        manager = ProjectManager()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            extraction = root / "road_extraction.png"
            width = root / "road_width.png"
            extraction.touch(); width.touch()
            manifest = {
                "period_results": [{
                    "grid": "area1", "period": "2021", "status": "completed",
                    "centerlines": str(root / "center.shp"),
                    "previews": {"fusion": str(extraction), "width": str(width)},
                }],
                "change_results": [],
            }
            items = manager.result_items(manifest, root)
            labels = {item["label"] for item in items}
            self.assertEqual(labels, {"area1", "单期结果", "2021", "道路提取图", "道路宽度图"})
            extraction_item = next(item for item in items if item["label"] == "道路提取图")
            self.assertEqual(Path(extraction_item["path"]), extraction.resolve())


class TaskManagerArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = TaskManager(_BackendStub())

    def test_resume_state_and_affected_pairs_are_ui_independent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            state = output / "run_a" / "job_state.json"
            state.parent.mkdir()
            state.write_text('{"status":"cancelled"}', encoding="utf-8")
            run_id, resume, resolved = self.manager.resolve_run(
                output, "", {"run_id": "run_a"},
            )
            self.assertEqual((run_id, resume, resolved), ("run_a", True, state))
        self.assertEqual(
            self.manager.affected_change_pairs(["2024", "2021", "2022"], "2022"),
            [("2021", "2022"), ("2022", "2024")],
        )

    def test_new_run_never_reuses_an_existing_task_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "成果输出"
            first_id, first_state = self.manager.create_new_run(
                output, generated_run_id="run_fixed",
            )
            first_state.parent.mkdir(parents=True)
            second_id, second_state = self.manager.create_new_run(
                output, generated_run_id="run_fixed",
            )
            self.assertEqual(first_id, "run_fixed")
            self.assertEqual(second_id, "run_fixed_01")
            self.assertNotEqual(first_state.parent, second_state.parent)

    def test_active_manifest_is_exact_task_not_merged_latest_context(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "成果输出"
            run_id, state = self.manager.create_new_run(
                output, generated_run_id="run_active",
            )
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({
                "run_id": run_id, "execution_profile": "fast",
            }), encoding="utf-8")
            pipeline = state.parent / "pipeline_result.json"
            pipeline.write_text(json.dumps({
                "run_id": run_id, "execution_profile": "fast",
            }), encoding="utf-8")
            merged = state.parents[2] / "latest_pipeline.json"
            merged.parent.mkdir(parents=True, exist_ok=True)
            merged.write_text(json.dumps({"run_id": "another_run"}), encoding="utf-8")

            resolved = self.manager.active_pipeline_manifest(
                output, {"run_id": run_id, "state": str(state)},
            )

            self.assertEqual(resolved, pipeline.resolve())
            self.assertEqual(self.manager.task_execution_profile(resolved), "fast")

    def test_local_rerun_commands_are_built_by_task_manager(self) -> None:
        manifest = Path("pipeline_result.json")
        period = self.manager.build_rerun_period(manifest, "area1", "2021", True)
        change = self.manager.build_rerun_change(
            manifest, "area1", "2021", "2022", True,
        )
        self.assertEqual(period[0], "rerun-period")
        self.assertIn("--update-related", period)
        self.assertEqual(change[0], "rerun-change")
        self.assertIn("--update-temporal", change)
        all_periods = self.manager.build_rerun_all_periods(manifest, True)
        all_changes = self.manager.build_rerun_all_changes(manifest, True)
        self.assertEqual(all_periods[0], "rerun-all-periods")
        self.assertEqual(all_changes[0], "rerun-all-changes")
        self.assertIn("--continue-on-error", all_periods)
        self.assertIn("--continue-on-error", all_changes)


class LayerBoundaryTests(unittest.TestCase):
    def test_app_layer_has_no_tk_dependency_and_gui_has_no_popen(self) -> None:
        root = CODE_ROOT
        app_source = "\n".join(path.read_text(encoding="utf-8") for path in (root / "app").glob("*.py"))
        gui_source = "\n".join(path.read_text(encoding="utf-8") for path in (root / "gui").glob("*.py"))
        self.assertNotIn("tkinter", app_source)
        self.assertNotIn("StringVar", app_source)
        self.assertNotIn("subprocess.Popen", gui_source)
        self.assertNotIn("__SAMROAD_USER__", gui_source)


if __name__ == "__main__":
    unittest.main()
