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
        for year in ("2020", "2022", "20250118"):
            (base / "02_影像" / (year + ".tif")).touch()
            (base / "02_影像" / (year + ".txt")).write_text(year + ".tif\n", encoding="utf-8")
    output = root / "成果输出"
    output.mkdir()
    products = {}
    for key in ("centerlines", "surfaces", "width_segments", "changes", "life_shp", "csv"):
        p = output / (key + (".csv" if key == "csv" else ".shp"))
        p.touch()
        products[key] = str(p)
    manifest = {"run_id": "示例任务", "execution_profile": "fast", "status": "completed", "input_spec": {"width_method": "raw_image", "irmad": {"enabled": True, "reference_period": "20250118"}}, "period_results": [
        {"grid": a, "period": y, "published": {k: products[k] for k in ("centerlines", "surfaces", "width_segments")}}
        for a in ("北区", "南区") for y in ("2020", "2022", "20250118")],
        "change_results": [{"grid": "北区", "before_period": "2020", "after_period": "2022", "published": {"changes": products["changes"]}}],
        "temporal_results": [{"grid": "北区", "life_shp": products["life_shp"]}],
        "evaluation_summary": {"csv": products["csv"]}}
    target = root / "_work/tasks/latest_pipeline.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "project_config.json").write_text(json.dumps({"area_irmad_references": {"北区": "2020", "南区": "2022"}, "custom_key": "preserve"}), encoding="utf-8")


class ProcessingUiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture(self.root)
        check=patch('plugin.controller.Controller.inspect_data',return_value={'periods':[]});check.start();self.addCleanup(check.stop)
        self.plugin = create_plugin()
        self.widget = self.plugin.create_widget()
        self.widget.project.edit.setText(str(self.root))
        model=scan_project(self.root);model['area_irmad_references']={'北区':'2020','南区':'2022'}
        self.widget._scanned(model)
        self.wait_ready()

    def wait_ready(self):
        deadline = time.monotonic() + 5
        while (self.widget.browsing or self.widget.open_timer.isActive()) and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(.01)
        self.assertFalse(self.widget.browsing)

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
        self.assertEqual(self.widget.check_note.text(), "")
        self.assertIn("2 个区域", self.widget.summary.text())

    def test_fixed_processing_and_automatic_evaluation(self):
        self.assertFalse(hasattr(self.widget, "profile"))
        self.assertFalse(hasattr(self.widget, "evaluate"))
        data = self.widget._data()
        self.assertEqual(data["profile"], "fast")
        self.assertTrue(data["evaluate"])  # partial truth coverage is sufficient
        command = self.widget.controller.build_command("all", data)
        self.assertNotIn("--no-evaluation", command)
        self.widget.model["truths"] = []
        data = self.widget._data()
        self.assertFalse(data["evaluate"])
        self.assertIn("--no-evaluation", self.widget.controller.build_command("all", data))
        task = self.widget._current
        task["data"]["execution_profile"] = "full"
        self.widget._update_controls()
        self.assertFalse(self.widget.rerun_period.isEnabled())
        self.assertFalse(hasattr(self.widget, "task"))
        self.assertFalse(hasattr(self.widget, "run_id"))
        self.assertFalse(hasattr(self.widget, "recent"))

    def test_background_scan(self):
        self.widget.scan()
        deadline = time.monotonic() + 5
        while self.widget.browsing and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(.01)
        self.assertFalse(self.widget.browsing)
        self.assertEqual(self.widget.areas.table.rowCount(), 2)

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
        self.assertEqual(len(self.widget.findChildren(QScrollArea)), 3)
        self.assertFalse(self.widget.findChildren(QTabBar))
        self.assertEqual(self.widget.pages.count(), 3)
        self.assertFalse(self.widget.local.toggle.isChecked())
        self.assertFalse(self.widget.advanced.toggle.isChecked())
        self.assertFalse(self.widget.records_fold.toggle.isChecked())

    def test_group_open_and_automatic_completion_results(self):
        received = []
        self.plugin.result_ready.connect(received.append)
        self.widget._reset_results()
        self.widget._finished({"status": "completed", "task_id": "demo"})
        self.assertTrue(self.widget.group_buttons["单期道路"].isEnabled())
        self.assertFalse(received)
        self.widget.group_buttons["单期道路"].click()
        self.assertFalse(received)
        self.assertFalse(self.widget.chooser.isHidden())
        self.widget.chooser.open_button.click()
        self.assertEqual(len(received), 1)

    def test_hidden_widget_keeps_task_updates(self):
        with patch.object(self.widget.controller, "cancel") as cancel:
            self.widget._started({"task_id": "hidden-test"})
            self.widget.hide()
            for fold in (self.widget.advanced, self.widget.local, self.widget.records_fold):
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
        for fold in (self.widget.advanced, self.widget.local, self.widget.records_fold):
            self.assertTrue(fold.toggle.autoRaise())

    def test_footer_stays_visible_with_expanded_tools(self):
        self.widget.show()
        for width in (300, 360, 680):
            self.widget.resize(width, 600)
            for fold in (self.widget.advanced, self.widget.local, self.widget.records_fold):
                self.widget._reveal(fold)
                APP.processEvents()
                APP.processEvents()
                self.assertEqual(self.widget.scroll.horizontalScrollBar().maximum(), 0)
                bottom = self.widget.run_button.mapTo(self.widget, self.widget.run_button.rect().bottomRight())
                self.assertLess(bottom.y(), self.widget.height())
                self.widget._reveal(fold)
        self.widget._started({"task_id": "state-test"})
        self.assertFalse(self.widget.running_status.isHidden())
        self.widget._finished({"status": "completed", "task_id": "state-test"})
        self.assertTrue(self.widget.running_status.isHidden())
        self.assertEqual(self.widget.task_line.text().count("已完成"), 1)
        self.widget._failed({"message": "测试错误", "detail": "错误详情"})
        self.assertIn("测试错误", self.widget.task_line.text())
        self.assertFalse(self.widget.failure_details.isHidden())

    def test_area_reference_selection_persists_independently(self):
        self.widget._show_configuration()
        self.widget.configuration.reference_boxes['北区'].setCurrentText('2022')
        self.assertFalse(self.widget.checked)
        self.assertEqual(scan_project(self.root)['area_irmad_references']['北区'], '2020')
        self.widget._save_and_check()
        self.wait_ready()
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.main_page)
        restored = scan_project(self.root)
        self.assertEqual(restored['area_irmad_references'], {'北区': '2022', '南区': '2022'})
        self.assertEqual(restored['config']['custom_key'], 'preserve')
        self.assertEqual(restored['periods'], self.widget.model['periods'])

    def test_auto_open_missing_reference_and_save(self):
        path = self.root / 'project_config.json'
        path.write_text('{}')
        self.widget.scan()
        self.wait_ready()
        self.assertFalse(self.widget.checked)
        self.assertEqual(self.widget.check_note.text(), '北区：请选择 IR-MAD 参考期')
        self.assertTrue(self.widget.group_buttons['单期道路'].isEnabled())
        self.widget._show_configuration()
        self.widget._save_and_check()
        self.wait_ready()
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.configuration)
        self.widget.configuration.reference_boxes['北区'].setCurrentText('2020')
        self.widget.configuration.reference_boxes['南区'].setCurrentText('2022')
        self.widget._save_and_check()
        self.wait_ready()
        self.assertTrue(self.widget.run_button.isEnabled())
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.main_page)

    def test_auto_open_and_continue_task(self):
        manifest = self.root / '_work/tasks/latest_pipeline.json'
        data = json.loads(manifest.read_text())
        data['status'] = 'cancelled'
        manifest.write_text(json.dumps(data))
        (manifest.parent / 'job_state.json').write_text(json.dumps(data))
        self.widget.project.edit.setText('')
        self.widget.project.edit.setText(str(self.root))
        self.wait_ready()
        self.assertTrue(self.widget.checked)
        self.assertFalse(self.widget.resume_button.isHidden())
        with patch.object(self.widget.controller, 'run') as run:
            self.widget.resume_button.click()
            action, payload = run.call_args.args
            self.assertEqual(action, 'all')
            self.assertTrue(payload['resume'])
            self.assertNotIn('run_id', payload)

    def test_configuration_scroll_positions_and_wheel(self):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent
        self.widget.show()
        for width in (300, 360, 680):
            self.widget.resize(width, 600)
            self.widget._show_configuration()
            APP.processEvents()
            APP.processEvents()
            panel = self.widget.configuration
            self.assertEqual(panel.scroll.horizontalScrollBar().maximum(), 0)
            bar = panel.scroll.verticalScrollBar()
            bar.setValue(min(120, bar.maximum()))
            value = bar.value()
            self.widget._back_to_main()
            self.widget._show_configuration()
            APP.processEvents()
            self.assertEqual(bar.value(), value)
            bottom = panel.save_button.mapTo(self.widget, panel.save_button.rect().bottomRight())
            self.assertLess(bottom.y(), self.widget.height())
            viewport = panel.periods.table.viewport()
            before = panel.periods.table.verticalScrollBar().value()
            event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, -120),
                                __import__('PySide6.QtCore', fromlist=['Qt']).Qt.MouseButton.NoButton,
                                __import__('PySide6.QtCore', fromlist=['Qt']).Qt.KeyboardModifier.NoModifier,
                                __import__('PySide6.QtCore', fromlist=['Qt']).Qt.ScrollPhase.NoScrollPhase, False)
            APP.sendEvent(viewport, event)
            self.assertGreaterEqual(bar.value(), value)
            self.assertEqual(panel.periods.table.verticalScrollBar().value(), before)

    def test_stale_check_cannot_overwrite_changed_project(self):
        from threading import Event
        entered, release = Event(), Event()
        def delayed(data):
            entered.set()
            release.wait(3)
            return {'periods': []}
        with patch.object(self.widget.controller, 'inspect_data', side_effect=delayed):
            self.widget.check_data()
            self.assertTrue(entered.wait(1))
            self.widget.project.edit.setText(str(self.root / 'missing-project'))
            release.set()
            self.wait_ready()
        self.assertFalse(self.widget.checked)
        self.assertFalse(self.widget.group_buttons['单期道路'].isEnabled())
        self.assertIn('存在的项目', self.widget.check_note.text())

    def test_save_error_stays_in_configuration_and_preserves_file(self):
        self.widget._show_configuration()
        target = self.root / 'project_config.json'
        before = target.read_bytes()
        with patch('plugin.ui.project_browser.os.replace', side_effect=OSError('文件被占用')):
            self.widget._save_and_check()
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.configuration)
        self.assertIn('文件被占用', self.widget.check_note.text())

    def test_missing_image_targets_its_table_row(self):
        self.widget._show_configuration()
        self.widget.periods.table.item(0, 2).setText('missing-images.txt')
        self.widget._save_and_check()
        self.wait_ready()
        message = self.widget.check_note.text()
        self.assertIn('北区 / 2020', message)
        self.assertIn('文件不存在', message)
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.configuration)
        self.widget.configuration.locate(message)
        self.assertEqual(self.widget.periods.table.currentRow(), 0)
        self.assertEqual(self.widget.periods.table.item(0, 2).toolTip(), message)

    def create_from_inputs(self, name, inputs):
        self.widget._show_creation()
        page = self.widget.creation
        page.project_name.setText(name)
        page.project_location.edit.setText(str(self.root))
        page.load(inputs)
        page.save_button.click()
        self.wait_ready()
        return self.root / name

    def test_create_empty_project_then_configure(self):
        source = self.widget.configuration.inputs()
        root = self.create_from_inputs('空项目', dict(areas=[], periods=[], truths=[], area_irmad_references={}))
        for folder in ('01_验证区', '02_影像', '03_变化真值', '成果输出', '_work'):
            self.assertTrue((root / folder).is_dir())
        self.assertTrue((root / 'project_config.json').is_file())
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.configuration)
        self.assertFalse(self.widget.checked)
        self.assertTrue(self.widget.configuration.save_button.isEnabled())
        self.widget.configuration.load(source)
        self.widget.configuration.save_button.click()
        self.wait_ready()
        self.assertTrue(self.widget.checked)
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.main_page)
        self.widget.project.edit.setText(str(self.root))
        self.wait_ready()
        self.widget.project.edit.setText(str(root))
        self.wait_ready()
        self.assertTrue(self.widget.checked)
        self.assertEqual(len(self.widget.model['periods']), 6)

    def test_create_complete_project_with_external_images(self):
        source = self.widget.configuration.inputs()
        source['periods'][0][-1] = str(self.root / '北区/02_影像/2020.tif')
        source['periods'][1][-1] = '\n'.join([str(self.root / '北区/02_影像/2022.tif'), str(self.root / '北区/02_影像/20250118.tif')])
        root = self.create_from_inputs('完整项目', source)
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.main_page)
        self.assertTrue(self.widget.run_button.isEnabled())
        model = scan_project(root)
        self.assertEqual(model['areas'], source['areas'])
        self.assertEqual(model['area_irmad_references'], source['area_irmad_references'])
        self.assertEqual(model['output'], str(root / '成果输出'))
        self.assertFalse(list((root / '02_影像').rglob('*.tif')))
        first = Path(model['periods'][0][-1])
        self.assertTrue(first.is_relative_to(root))
        self.assertEqual(first.read_text(encoding='utf-8').strip(), source['periods'][0][-1])
        self.assertEqual(len(Path(model['periods'][1][-1]).read_text(encoding='utf-8').splitlines()), 2)
        self.assertEqual(check_files(model), [])
        self.widget._show_configuration()
        self.widget.configuration.reference_boxes['北区'].setCurrentText('2022')
        self.widget.configuration.save_button.click()
        self.wait_ready()
        with patch('plugin.widget.QFileDialog.getExistingDirectory', return_value=str(self.root)):
            self.widget.open_button.click()
        self.wait_ready()
        self.assertTrue(self.widget.group_buttons['单期道路'].isEnabled())
        with patch('plugin.widget.QFileDialog.getExistingDirectory', return_value=str(root)):
            self.widget.open_button.click()
        self.wait_ready()
        self.assertEqual(self.widget.model['area_irmad_references']['北区'], '2022')

    def test_creation_collision_and_failed_write_preserve_existing_project(self):
        from plugin.ui.project_browser import create_project
        source = self.widget.configuration.inputs()
        config = (self.root / 'project_config.json').read_bytes()
        with self.assertRaises(ValueError):
            create_project(self.root.parent, self.root.name, source)
        self.assertEqual((self.root / 'project_config.json').read_bytes(), config)
        for name in ('../outside', 'bad/name', 'CON', ''):
            with self.assertRaises(ValueError):
                create_project(self.root, name, source)
        with patch('plugin.ui.project_browser.os.replace', side_effect=OSError('写入失败')):
            with self.assertRaises(OSError):
                create_project(self.root, '失败项目', source)
        self.assertFalse((self.root / '失败项目').exists())
        self.assertFalse(list(self.root.glob('.road-change-new-*')))

    def test_creation_draft_and_narrow_page(self):
        self.widget.show()
        self.widget._show_creation()
        page = self.widget.creation
        page.project_name.setText('草稿')
        for width in (300, 360, 450, 680):
            self.widget.resize(width, 600)
            APP.processEvents()
            APP.processEvents()
            self.assertEqual(page.scroll.horizontalScrollBar().maximum(), 0)
            page.scroll.verticalScrollBar().setValue(100)
            value = page.scroll.verticalScrollBar().value()
            self.widget._back_to_main()
            self.widget._show_creation()
            APP.processEvents()
            self.assertEqual(page.project_name.text(), '草稿')
            self.assertEqual(page.scroll.verticalScrollBar().value(), value)
            self.assertLess(page.save_button.mapTo(self.widget, page.save_button.rect().bottomRight()).y(), self.widget.height())

    def test_host_can_override_local_appearance(self):
        application_style = APP.styleSheet()
        self.widget.setProperty("hostStyled", True)
        self.assertEqual(self.widget.styleSheet(), "")
        self.widget.setProperty("hostStyled", False)
        self.assertTrue(self.widget.styleSheet())
        self.assertEqual(APP.styleSheet(), application_style)


if __name__ == "__main__":
    unittest.main()
