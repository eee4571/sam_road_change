"""Lightweight UI acceptance; synthetic files only, never launch a backend."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication, QScrollArea, QTabBar, QStackedWidget
from PySide6.QtCore import QThreadPool
from plugin import create_plugin
from plugin.ui.project_browser import scan_project, check_files, pairs

APP = QApplication.instance() or QApplication([])


def fixture(root):
    """A conventional project with empty placeholder GIS files for file checks."""
    for area in ("北区", "南区"):
        base = root / area
        for folder in ("01_验证区", "02_影像", "03_变化真值"):
            (base / folder).mkdir(parents=True)
        for suffix in (".shp", ".shx", ".dbf", ".prj"):
            (base / "01_验证区" / (area + suffix)).touch()
            (base / "03_变化真值" / ("2020_to_2022" + suffix)).touch()
        for year in ("2020", "2022", "2024"):
            (base / "02_影像" / (year + ".tif")).touch()
            (base / "02_影像" / (year + ".txt")).write_text(year + ".tif\n", encoding="utf-8")
    output = root / "成果输出"
    output.mkdir()
    products = {}
    for key in ("centerlines", "surfaces", "width_segments", "changes", "life_shp", "csv"):
        p = output / (key + (".csv" if key == "csv" else ".shp"))
        p.touch()
        products[key] = str(p)
    manifest = {"run_id": "示例任务", "status": "completed", "period_results": [
        {"grid": a, "period": y, "published": {k: products[k] for k in ("centerlines", "surfaces", "width_segments")}}
        for a in ("北区", "南区") for y in ("2020", "2022", "2024")],
        "change_results": [{"grid": "北区", "before_period": "2020", "after_period": "2022", "published": {"changes": products["changes"]}}],
        "temporal_results": [{"grid": "北区", "life_shp": products["life_shp"]}],
        "evaluation_summary": {"csv": products["csv"]}}
    target = root / "_work/tasks/latest_pipeline.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(manifest), encoding="utf-8")


class ProcessingUiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture(self.root)
        self.plugin = create_plugin()
        self.widget = self.plugin.create_widget()
        self.widget.project.edit.setText(str(self.root))
        self.widget._scanned(scan_project(self.root))

    def tearDown(self):
        self.widget.close()
        self.plugin.shutdown()
        QThreadPool.globalInstance().waitForDone(3000)
        APP.processEvents()
        self.tmp.cleanup()

    def test_scan_and_check(self):
        self.assertEqual(len(self.widget.model["areas"]), 2)
        self.assertEqual(len(self.widget.model["periods"]), 6)
        self.assertEqual(len(pairs(self.widget.model["periods"])), 4)
        self.assertEqual(len(self.widget.model["truths"]), 2)
        self.assertEqual(check_files(self.widget.model), [])
        self.widget._checked([])
        self.assertTrue(self.widget.run_button.isEnabled())
        self.assertIn("已就绪", self.widget.check_note.text())
        self.assertIn("GT 2/4", self.widget.summary.text())

    def test_background_scan(self):
        self.widget.scan()
        deadline = time.monotonic() + 5
        while self.widget.browsing and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(.01)
        self.assertFalse(self.widget.browsing)
        self.assertEqual(self.widget.data_tree.topLevelItemCount(), 2)

    def test_corrections_invalidate_and_apply(self):
        self.widget._checked([])
        self.widget.periods.table.item(0, 1).setText("2019")
        self.assertFalse(self.widget.checked)
        self.widget._apply_corrections()
        self.assertIn("2019", [r[1] for r in self.widget.model["periods"]])
        self.assertTrue(check_files(self.widget.model))  # stale GT pair detected

    def test_dropdowns_and_unchanged_controller_contract(self):
        self.assertEqual(self.widget.grid.count(), 2)
        self.assertEqual(self.widget.period.count(), 3)
        self.assertEqual(self.widget.pair.count(), 2)
        data = self.widget._data()
        self.assertTrue(data["manifest"].endswith("latest_pipeline.json"))
        self.assertEqual((data["before_period"], data["after_period"]), ("2020", "2022"))
        for action in ("all", "rerun-period", "rerun-change"):
            command = self.widget.controller.build_command(action, data)
            self.assertEqual(command[0], action)

    def test_file_check_missing_image(self):
        (self.root / "北区/02_影像/2020.tif").unlink()
        self.assertTrue(any("影像路径不存在" in e for e in check_files(self.widget.model)))

    def test_status_uses_events_not_log_text(self):
        self.widget._started({"task_id": "ui-test"})
        self.widget._progress({"stage": "道路提取", "progress": .8, "event": {"kind": "stage", "grid": "北区", "period": "2022"}})
        self.assertEqual(self.widget.progress.maximum(), 0)
        self.assertEqual(self.widget.area_text.text(), "北区")
        self.widget._progress({"stage": "变化检测", "event": {"kind": "pipeline", "grid": "南区", "before_period": "2020", "after_period": "2022", "progress": .4, "completed": 4, "total": 10}})
        self.assertEqual(self.widget.progress.value(), 400)
        self.assertEqual(self.widget.scope_text.text(), "2020 → 2022")
        self.widget._log({"message": "completed failed 100%"})
        self.assertEqual(self.widget.state_text.text(), "运行中")
        self.widget._failed({"task_id": "ui-test", "message": "示例错误", "detail": "详细原因"})
        self.assertIn("详细原因", self.widget.error_detail.toPlainText())

    def test_result_open_signal_only(self):
        received = []
        self.plugin.result_ready.connect(received.append)
        self.widget._refresh_results()
        self.assertFalse(received)
        action = self.widget.result_menus["单期道路"].actions()[0]
        action.trigger()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["result_type"], "road_centerline")
        self.assertNotIn("road_centerline", action.text())

    def test_narrow_dock_and_default_visibility(self):
        self.widget.show()
        for width in (300, 380, 450):
            self.widget.resize(width, 760)
            APP.processEvents()
            self.assertLessEqual(self.widget.minimumSizeHint().width(), 300)
            self.assertEqual(self.widget.scroll.horizontalScrollBar().maximum(), 0)
        self.assertEqual(len(self.widget.findChildren(QScrollArea)), 1)
        self.assertFalse(self.widget.findChildren(QTabBar))
        self.assertFalse(self.widget.findChildren(QStackedWidget))
        self.assertFalse(self.widget.local.toggle.isChecked())
        self.assertFalse(self.widget.advanced.toggle.isChecked())
        self.assertFalse(self.widget.corrections.toggle.isChecked())
        self.assertFalse(self.widget.records_fold.toggle.isChecked())

    def test_group_open_and_automatic_completion_results(self):
        received = []
        self.plugin.result_ready.connect(received.append)
        self.widget._reset_results()
        self.widget._finished({"status": "completed", "task_id": "demo"})
        self.assertTrue(self.widget.group_buttons["单期道路"].isEnabled())
        self.assertFalse(received)
        self.widget.group_buttons["单期道路"].click()
        self.assertEqual({r["result_type"] for r in received}, {"road_centerline", "road_surface", "road_width"})

    def test_hidden_widget_keeps_task_updates(self):
        with patch.object(self.widget.controller, "cancel") as cancel:
            self.widget._started({"task_id": "hidden-test"})
            self.widget.hide()
            for fold in (self.widget.corrections, self.widget.advanced, self.widget.local, self.widget.records_fold):
                fold.toggle.setChecked(True)
                fold.toggle.setChecked(False)
            self.widget._progress({"stage": "道路提取", "event": {"kind": "pipeline", "progress": .25}})
            self.assertEqual(self.widget.progress.value(), 250)
            self.assertTrue(self.widget.timer.isActive())
            cancel.assert_not_called()

    def test_evaluation_summary(self):
        report = self.root / "metrics.json"
        report.write_text(json.dumps({"metrics": [{"class": "all", "precision": .9, "recall": .8, "f1": .847}]}))
        self.widget._result({"result_type": "road_evaluation", "path": str(report), "name": "评价报告", "metadata": {}})
        self.assertIn("P 90%", self.widget.metrics.text())
        self.assertIn("F1 85%", self.widget.metrics.text())

    def test_visual_hierarchy_and_native_boundaries(self):
        from PySide6.QtWidgets import QTreeWidget
        self.assertFalse(self.widget.results.findChildren(QTreeWidget))
        self.assertTrue(self.widget.scan_button.icon().isNull())
        self.assertTrue(self.widget.check_button.icon().isNull())
        self.assertTrue(self.widget.run_button.icon().isNull())
        self.assertEqual(self.widget.run_button.property("role"), "primary")
        self.assertIn("#roadChangeDock", self.widget.styleSheet())
        self.assertEqual(self.widget.scan_button.styleSheet(), "")
        for fold in (self.widget.corrections, self.widget.advanced, self.widget.local, self.widget.records_fold):
            self.assertTrue(fold.toggle.autoRaise())

    def test_host_can_override_local_appearance(self):
        application_style = APP.styleSheet()
        self.widget.setProperty("hostStyled", True)
        self.assertEqual(self.widget.styleSheet(), "")
        self.widget.setProperty("hostStyled", False)
        self.assertTrue(self.widget.styleSheet())
        self.assertEqual(APP.styleSheet(), application_style)


if __name__ == "__main__":
    unittest.main()
