import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
from PySide6.QtCore import QProcess
from plugin import create_plugin
from plugin.runner import Runner, parse_line, PREFIX
from plugin.result_parser import read_results, RESULT_TYPES

APP = QApplication.instance() or QApplication([])


class FakeRunner(Runner):
    """Only a synthetic text-emitting child, never user_pipeline or models."""
    script = "print('ordinary log')"

    def missing_runtime(self, profile="full"):
        return []

    def command(self, args):
        return sys.executable, ["-u", "-c", self.script]

    def environment(self):
        env = super().environment()
        # Do not activate backend sitecustomize in the synthetic test interpreter.
        env.remove("PYTHONPATH")
        return env


def wait_done(runner):
    deadline = time.monotonic() + 8
    while runner.running and time.monotonic() < deadline:
        APP.processEvents()
        time.sleep(0.01)
    if runner.running:
        runner.shutdown()
        raise AssertionError("Synthetic child timed out")
    APP.processEvents()


class PluginTests(unittest.TestCase):
    def test_api_pages_and_narrow_layout(self):
        plugin = create_plugin()
        parent = QWidget()
        widget = plugin.create_widget(parent)
        self.assertIsInstance(widget, QWidget)
        self.assertNotIsInstance(widget, (QMainWindow, QDialog))
        self.assertIs(widget.parent(), parent)
        self.assertIs(QApplication.instance(), APP)
        widget.resize(300, 700)
        widget.show()
        for index in range(4):
            widget.pages.setCurrentIndex(index)
            APP.processEvents()
            self.assertEqual(widget.pages.currentIndex(), index)
            self.assertLessEqual(widget.minimumSizeHint().width(), 300)
        self.assertFalse(widget.local.toggle.isChecked())
        self.assertFalse(widget.advanced.toggle.isChecked())
        self.assertEqual(set(widget.groups), set(RESULT_TYPES))
        widget.close()
        plugin.shutdown()

    def test_standalone_lifecycle(self):
        script = "from PySide6.QtCore import QTimer; import standalone; original=standalone.create_plugin\ndef factory():\n p=original(); QTimer.singleShot(100, standalone.QApplication.instance().quit); return p\nstandalone.create_plugin=factory\nraise SystemExit(standalone.main())"
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_no_heavy_imports_or_machine_paths(self):
        forbidden = {"torch", "rasterio", "osgeo", "geopandas", "numpy", "cv2", "tensorflow", "tkinter", "user_pipeline", "engine", "app"}
        for source in (ROOT / "plugin").rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                modules = [n.name.split('.')[0] for n in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else []
                self.assertFalse(forbidden.intersection(modules), source)
            self.assertNotIn("setFixedSize(", text)
            self.assertNotIn("setStyleSheet(", text)
            self.assertNotIn("setFont(", text)
        for folder in ("plugin", "code", "runtime/config"):
            for source in (ROOT / folder).rglob("*"):
                if source.suffix not in {".py", ".yaml", ".json"}:
                    continue
                text = source.read_text(encoding="utf-8")
                self.assertNotRegex(text, r"[A-Za-z]:[\\/](?:Users|tmp|projects|home)[\\/]")
        self.assertFalse(forbidden.intersection(sys.modules))

    def test_snapshot_and_placeholders(self):
        snapshot = json.loads((ROOT / "resources/backend_snapshot.json").read_text())
        for path, digest in snapshot.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        for name in ("env", "model"):
            self.assertEqual([p.name for p in (ROOT / "runtime" / name).iterdir()], ["PLACEHOLDER"])

    def test_commands(self):
        plugin = create_plugin()
        controller = plugin._controller
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("area.shp", "a.txt", "b.txt", "truth.shp"):
                (root/name).touch()
            manifest = root/"pipeline_result.json"
            manifest.write_text('{}')
            data = dict(areas=[["区1", str(root/"area.shp")]], periods=[["区1", "2020", str(root/"a.txt")], ["区1", "2021", str(root/"b.txt")]], output=str(root/"成果"), profile="fast", run_id="run_1", resume=True,
                        manifest=str(manifest), grid="区1", period="2020", before_period="2020", after_period="2021")
            args = controller.build_command("all", data)
            self.assertIn("--resume", args)
            self.assertIn("fast", args)
            self.assertIn("--no-evaluation", args)
            self.assertIn("--update-related", controller.build_command("rerun-period", data))
            self.assertIn("--update-temporal", controller.build_command("rerun-change", data))
            self.assertEqual(controller.build_command("evaluate-all-existing", data)[0], "evaluate-all-existing")
            program, command = controller.runner.command(args)
            self.assertTrue(Path(program).is_relative_to(ROOT/"runtime/env"))
            self.assertEqual(command[1], str(ROOT/"code/user_pipeline.py"))
        plugin.shutdown()

    def test_missing_runtime_signal(self):
        runner = Runner()
        failures = []
        runner.task_failed.connect(failures.append)
        runner.start(["all"])
        self.assertEqual(runner.state, "failed")
        self.assertIn("runtime", failures[0]["message"])
        self.assertEqual(runner.process.state(), QProcess.ProcessState.NotRunning)
        runner.shutdown()

    def test_protocol(self):
        self.assertIsNone(parse_line("all completed failed"))
        self.assertEqual(parse_line(PREFIX+'{"kind":"stage"}')["kind"], "stage")
        with self.assertRaises(ValueError):
            parse_line(PREFIX+'[]')
        runner = Runner()
        progress = []
        runner.task_progress.connect(progress.append)
        runner.consume_line(PREFIX+'{"kind":"pipeline","completed":2,"total":4,"stage":"道路"}')
        self.assertEqual(progress[-1]["progress"], 0.5)
        runner.consume_line(PREFIX+'{"kind":"stage","stage_index":9,"stage_total":4}')
        self.assertEqual(progress[-1]["progress"], 1.0)

    def test_qprocess_success_and_fragmented_unicode(self):
        runner = FakeRunner()
        runner.script = "import os,time; data='__SAMROAD_USER__{\"kind\":\"complete\",\"stage\":\"all\",\"message\":\"道路\"}'.encode(); os.write(1,data[:-3]); time.sleep(.03); os.write(1,data[-3:])"
        finished, started = [], []
        runner.task_finished.connect(finished.append)
        runner.task_started.connect(started.append)
        runner.start([])
        wait_done(runner)
        self.assertEqual(len(started), 1)
        self.assertEqual(finished[0]["event"]["message"], "道路")
        self.assertEqual(runner.state, "completed")

    def test_failure_exit_and_partial_failure(self):
        for script in ("raise SystemExit(3)", "print('__SAMROAD_USER__{\"kind\":\"complete\",\"status\":\"completed_with_errors\"}')", "print('__SAMROAD_USER__broken')"):
            runner = FakeRunner()
            runner.script = script
            failures = []
            runner.task_failed.connect(failures.append)
            runner.start([])
            wait_done(runner)
            self.assertEqual(runner.state, "failed")
            self.assertEqual(len(failures), 1)

    def test_cancel_and_shutdown(self):
        for shutdown in (False, True):
            runner = FakeRunner()
            runner.script = "import time; time.sleep(20)"
            runner.start([])
            self.assertTrue(runner.process.waitForStarted(3000))
            runner.shutdown() if shutdown else runner.cancel()
            wait_done(runner)
            self.assertEqual(runner.state, "cancelled")

    def test_results_and_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("center.shp", "surface.shp", "width.shp", "change.shp", "life.shp", "score.csv"):
                (root/name).touch()
            data = dict(final_period_results=[dict(grid="a", period="1", published=dict(centerlines="center.shp", surfaces="surface.shp", width_segments="width.shp"))], change_results=[dict(published=dict(changes="change.shp"))], temporal_results=[dict(life_shp="life.shp")], evaluation_summary=dict(csv="score.csv"))
            path = root/"pipeline_result.json"
            path.write_text(json.dumps(data))
            self.assertEqual({r['result_type'] for r in read_results(path)}, set(RESULT_TYPES))
            data['fast_finalization_state'] = 'pending'
            path.write_text(json.dumps(data))
            self.assertEqual(read_results(path), [])

    def test_relocation_and_foreign_cwd(self):
        with tempfile.TemporaryDirectory(prefix="road plugin ") as tmp:
            moved = Path(tmp)/"moved"
            shutil.copytree(ROOT/"plugin", moved/"plugin", ignore=shutil.ignore_patterns("__pycache__"))
            script = "import sys; sys.path.insert(0, sys.argv[1]); from PySide6.QtWidgets import QApplication; from plugin import create_plugin; from plugin.runner import ROOT,Runner; from pathlib import Path; a=QApplication([]); p=create_plugin(); w=p.create_widget(); assert ROOT==Path(sys.argv[1]); assert Runner().python_path().is_relative_to(ROOT); p.shutdown()"
            result = subprocess.run([sys.executable, "-c", script, str(moved)], cwd=tmp, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
