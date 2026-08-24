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
            self.assertEqual(manager.read_config(root), payload)

    def test_manifest_result_models_are_built_outside_tk(self) -> None:
        manager = ProjectManager()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            center = root / "center.shp"
            center.touch()
            manifest = {
                "period_results": [{
                    "grid": "area1", "period": "2021", "status": "completed",
                    "centerlines": str(center), "surfaces": str(root / "surface.shp"),
                }],
                "change_results": [],
            }
            items = manager.result_items(manifest, root)
            center_item = next(item for item in items if item["label"] == "中心线")
            self.assertEqual(center_item["status"], "已生成")
            self.assertEqual(Path(center_item["path"]), center.resolve())


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
