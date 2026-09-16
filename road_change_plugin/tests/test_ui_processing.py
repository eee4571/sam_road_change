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
        check = patch('plugin.controller.Controller.inspect_data', return_value={'issues': []})
        check.start()
        self.addCleanup(check.stop)
        self.plugin = create_plugin()
        self.widget = self.plugin.create_widget()
        self.widget._load_project(self.root)
        self.wait_ready()

    def wait_ready(self):
        deadline = time.monotonic() + 5
        while (self.widget.browsing or self.widget.open_timer.isActive()) and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(.01)
        self.assertFalse(self.widget.browsing)
        APP.processEvents()

    def tearDown(self):
        self.widget.close()
        self.plugin.shutdown()
        QThreadPool.globalInstance().waitForDone(3000)
        APP.processEvents()
        self.tmp.cleanup()

    def test_project_entry_and_current_state(self):
        self.assertTrue(self.widget.run_button.isHidden())
        self.assertTrue(self.widget.locate_button.isEnabled())
        self.assertTrue(self.widget.project_entry.isHidden())
        self.assertFalse(self.widget.project_details.isHidden())
        self.assertEqual(self.widget.project_path.text(), str(self.root))
        self.assertFalse(hasattr(self.widget, 'project'))  # no editable project-path control
        self.assertFalse(hasattr(self.widget, 'scan_button'))
        self.assertIn('2 个区域', self.widget.summary.text())
        self.assertEqual(self.widget.data_state.text(), '数据已就绪')
        self.assertEqual(self.widget.logs_button.text(), '日志')
        self.widget._reveal(self.widget.records_fold)
        self.widget._load_project('')
        self.wait_ready()
        self.assertFalse(self.widget.project_entry.isHidden())
        for item in (self.widget.project_details, self.widget.processing_group, self.widget.result_group,
                     self.widget.footer, self.widget.logs_button, self.widget.more_button, self.widget.records_fold):
            self.assertTrue(item.isHidden())

    def test_scan_check_and_fixed_processing(self):
        self.assertEqual(len(self.widget.model['areas']), 2)
        self.assertEqual(len(self.widget.model['periods']), 6)
        self.assertEqual(len(pairs(self.widget.model['periods'])), 4)
        self.assertEqual(check_files(self.widget.model), [])
        self.assertEqual(self.widget.check_note.text(), '')
        data = self.widget._data()
        self.assertEqual(data['profile'], 'fast')
        self.assertTrue(data['evaluate'])
        self.assertNotIn('--no-evaluation', self.widget.controller.build_command('all', data))
        self.widget.model['truths'] = []
        self.assertIn('--no-evaluation', self.widget.controller.build_command('all', self.widget._data()))
        for name in ('profile', 'evaluate', 'task', 'run_id', 'recent', 'resume_button'):
            self.assertFalse(hasattr(self.widget, name))
        self.widget._current['data']['execution_profile'] = 'full'
        self.widget._update_controls()
        self.assertFalse(self.widget.update_button.isEnabled())

    def test_background_rescan(self):
        self.widget.rescan_action.trigger()
        self.wait_ready()
        self.assertEqual(self.widget.configuration.area.count(), 2)
        self.assertEqual(self.widget.configuration.periods.rowCount(), 3)

    def test_removing_all_regions_is_not_undone_by_automatic_scan(self):
        self.widget._show_configuration()
        page = self.widget.configuration
        page.remove_area()
        page.remove_area()
        page.save_button.click()
        self.wait_ready()
        self.assertEqual(scan_project(self.root)['areas'], [])
        self.assertEqual(self.widget.run_button.text(), '配置数据')
        self.assertTrue((self.root / '北区/01_验证区/北区.shp').is_file())
        self.assertEqual(self.widget.pages.currentWidget(), page)

    def test_switch_cancel_preserves_unsaved_area_changes(self):
        from PySide6.QtWidgets import QMessageBox
        page = self.widget.configuration
        page.add_area('未保存区域')
        with patch('plugin.widget.QMessageBox.question', return_value=QMessageBox.StandardButton.No), patch('plugin.widget.QFileDialog.getExistingDirectory') as choose:
            self.widget.switch_button.click()
            choose.assert_not_called()
        self.assertEqual(self.widget.model['root'], str(self.root))
        self.assertIn(['未保存区域', ''], page.inputs()['areas'])

    def test_pending_image_editor_cannot_be_silently_saved(self):
        page = self.widget.configuration
        before = (self.root / 'project_config.json').read_bytes()
        page.edit_period('2020')
        self.widget._save_and_check()
        self.assertEqual((self.root / 'project_config.json').read_bytes(), before)
        self.assertFalse(page.save_button.isEnabled())
        self.assertFalse(page.set_period(None, '2022', ''))
        page._cancel_period()
        self.assertTrue(page.area.isEnabled())

    def test_batch_import_confirms_once_saves_and_stays_on_configuration_page(self):
        from plugin.ui.project_browser import save_configuration
        page = self.widget.configuration
        self.widget._show_configuration()
        images = []
        for name in ('A_20300118_01.tif', 'B_20300118_02.tif', 'C_20310203.tif'):
            path = self.root / name
            path.touch()
            images.append(path)
        before = (self.root / 'project_config.json').read_bytes()
        page.begin_import(images)
        self.widget._save_and_check()
        self.assertEqual((self.root / 'project_config.json').read_bytes(), before)
        with patch('plugin.widget.save_configuration', wraps=save_configuration) as save:
            page.import_confirmation.apply_button.click()
            self.wait_ready()
            save.assert_called_once()
        self.assertEqual(self.widget.pages.currentWidget(), page)
        periods = scan_project(self.root)['periods']
        imported = {p: source for area, p, source in periods if area == '北区' and p.startswith('203')}
        self.assertEqual(set(imported), {'20300118', '20310203'})
        self.assertEqual(len(Path(imported['20300118']).read_text(encoding='utf8').splitlines()), 2)
        self.assertTrue(self.widget.checked)
        self.widget._load_project(self.root)
        self.wait_ready()
        self.assertEqual(len([p for a, p, _ in self.widget.model['periods'] if a == '北区']), 5)

    def test_reference_selection_is_scoped_and_persists(self):
        page = self.widget.configuration
        self.widget._show_configuration()
        page.area.setCurrentText('北区')
        page.reference.setCurrentText('2022')
        page.area.setCurrentText('南区')
        self.assertEqual(page.reference.currentText(), '2022')
        page.reference.setCurrentText('20250118')
        page.area.setCurrentText('北区')
        self.assertEqual(page.reference.currentText(), '2022')
        self.assertFalse(self.widget.checked)
        self.assertEqual(self.widget.run_button.text(), '修正数据')
        page.save_button.click()
        self.wait_ready()
        restored = scan_project(self.root)
        self.assertEqual(restored['area_irmad_references'], {'北区': '2022', '南区': '20250118'})
        self.assertEqual(restored['config']['custom_key'], 'preserve')
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.main_page)

    def test_area_add_rename_delete_preserves_other_area(self):
        page = self.widget.configuration
        original = page.inputs()
        self.assertTrue(page.add_area('新区', str(self.root / 'new.shp')))
        self.assertEqual(page.area.currentText(), '新区')
        self.assertEqual(page.periods.rowCount(), 0)
        page.set_period(None, '2020', str(self.root / 'new.txt'))
        page.reference.setCurrentText('2020')
        page.rename_area('东区')
        inputs = page.inputs()
        self.assertIn(['东区', '2020', str(self.root / 'new.txt')], inputs['periods'])
        self.assertEqual(inputs['area_irmad_references']['东区'], '2020')
        page.remove_area()
        self.assertEqual(page.inputs(), original)
        self.assertFalse(page.add_area('北区'))

    def test_image_editor_supports_txt_and_multiple_rasters_without_internal_paths(self):
        page = self.widget.configuration
        self.widget._show_configuration()
        page.area.setCurrentText('北区')
        self.assertEqual(page.periods.item(0, 1).text(), '1 幅影像')
        page.edit_period('2022')
        images = [str(self.root / '北区/02_影像' / (p + '.tif')) for p in ('2020', '2022', '20250118')]
        with patch('plugin.ui.data_configuration.QFileDialog.getOpenFileNames', return_value=(images, '')):
            page._choose_images()
        self.assertEqual(page.source_summary.text(), '3 幅影像')
        self.assertFalse(self.widget.checked)
        self.assertFalse(page.save_button.isEnabled())
        self.assertFalse(page.area.isEnabled())
        page.apply_period_button.click()
        page.save_button.click()
        self.wait_ready()
        rows = scan_project(self.root)['periods']
        source = next(p for a, year, p in rows if (a, year) == ('北区', '2022'))
        self.assertTrue(Path(source).is_file())
        self.assertEqual(len(Path(source).read_text(encoding='utf-8').splitlines()), 3)
        page.area.setCurrentText('北区')
        self.assertEqual(page.periods.item(1, 1).text(), '3 幅影像')
        for row in range(page.periods.rowCount()):
            for col in (0, 1):
                self.assertNotIn('_lists', page.periods.item(row, col).text())
        page.edit_period('2020')
        txt = str(self.root / '北区/02_影像/2022.txt')
        with patch('plugin.ui.data_configuration.QFileDialog.getOpenFileName', return_value=(txt, '')):
            page._choose_txt()
        page.apply_period_button.click()
        self.assertIn(['北区', '2020', txt], page.inputs()['periods'])

    def test_pairs_are_derived_and_truth_is_scoped(self):
        page = self.widget.configuration
        page.area.setCurrentText('北区')
        self.assertEqual(page.truths.rowCount(), 2)
        self.assertEqual(page.truths.item(0, 0).text(), '2020 → 2022')
        self.assertEqual(page.truths.item(0, 1).text(), '已提供')
        page.set_truth('2022', '20250118', str(self.root / 'truth.shp'))
        self.assertEqual(page.truths.item(1, 1).text(), '已提供')
        page.remove_period('2022')
        self.assertEqual(page.truths.rowCount(), 1)
        self.assertEqual(page.truths.item(0, 0).text(), '2020 → 20250118')
        self.assertEqual(page.truths.item(0, 1).text(), '未提供')
        self.assertTrue(any(r[0] == '南区' for r in page.inputs()['truths']))

    def test_missing_reference_locates_area_and_blocks_processing(self):
        (self.root / 'project_config.json').write_text('{}')
        self.widget.scan()
        self.wait_ready()
        self.assertEqual(self.widget.run_button.text(), '修正数据')
        self.assertEqual(self.widget.check_note.text(), '北区：请选择 IR-MAD 参考期')
        self.widget.run_button.click()
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.configuration)
        self.assertEqual(self.widget.configuration.area.currentText(), '北区')
        self.assertTrue(self.widget.configuration.reference.property('inputProblem'))
        for area, year in (('北区', '2020'), ('南区', '2022')):
            self.widget.configuration.area.setCurrentText(area)
            self.widget.configuration.reference.setCurrentText(year)
        self.widget._save_and_check()
        self.wait_ready()
        self.assertTrue(self.widget.checked)
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.main_page)

    def test_missing_image_targets_specific_period(self):
        self.widget._show_configuration()
        page = self.widget.configuration
        page.area.setCurrentText('南区')
        page.set_period('2022', '2022', 'missing.txt')
        self.widget._save_and_check()
        self.wait_ready()
        message = self.widget.check_note.text()
        self.assertIn('南区 / 2022', message)
        self.assertIn('文件不存在', message)
        self.assertEqual(self.widget.pages.currentWidget(), page)
        page.locate(message)
        self.assertEqual(page.area.currentText(), '南区')
        self.assertEqual(page.periods.item(page.periods.currentRow(), 0).text(), '2022')
        self.assertEqual(page.periods.item(page.periods.currentRow(), 1).toolTip(), message)

    def test_missing_truth_targets_pair_instead_of_its_first_period(self):
        self.widget._show_configuration()
        page = self.widget.configuration
        page.area.setCurrentText('北区')
        page.set_truth('2020', '2022', str(self.root / 'missing-truth.shp'))
        page.save_button.click()
        self.wait_ready()
        message = self.widget.check_note.text()
        self.assertIn('北区 / 2020 / 2022', message)
        self.assertEqual(page.truths.currentRow(), 0)
        self.assertEqual(page.truths.item(0, 1).toolTip(), message)
        self.assertEqual(page.periods.item(0, 1).toolTip(), '')

    def test_full_restart_requires_confirmation_and_cancel_does_nothing(self):
        from PySide6.QtWidgets import QMessageBox
        self.assertTrue(self.widget.run_button.isHidden())
        self.assertTrue(self.widget.locate_button.isEnabled())
        with patch('plugin.widget.QMessageBox.question', return_value=QMessageBox.StandardButton.No) as question, patch.object(self.widget.controller, 'run') as run:
            self.widget.restart_action.trigger()
            run.assert_not_called()
            self.assertIn('项目数据和可复用缓存不会删除', question.call_args.args[2])
        with patch('plugin.widget.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes), patch.object(self.widget.controller, 'run') as run:
            self.widget.restart_action.trigger()
            self.assertEqual(run.call_args.args[0], 'all')
            self.assertFalse(run.call_args.args[1]['resume'])

    def test_ready_primary_starts_first_run(self):
        self.widget._current = None
        self.widget._project_operation = {}
        self.widget._reset_results()
        self.widget._update_controls()
        self.assertEqual(self.widget.run_button.text(), '开始处理')
        with patch.object(self.widget.controller, 'run') as run, patch('plugin.widget.QMessageBox.question') as confirm:
            self.widget.run_button.click()
            self.assertEqual(run.call_args.args[0], 'all')
            confirm.assert_not_called()

    def test_continue_full_run_uses_project_checkpoint(self):
        manifest = self.root / '_work/tasks/latest_pipeline.json'
        data = json.loads(manifest.read_text())
        data['status'] = 'cancelled'
        manifest.write_text(json.dumps(data))
        (manifest.parent / 'job_state.json').write_text(json.dumps(data))
        self.widget.scan()
        self.wait_ready()
        self.assertEqual(self.widget.run_button.text(), '继续处理')
        with patch.object(self.widget.controller, 'run') as run:
            self.widget.run_button.click()
            action, payload = run.call_args.args
            self.assertEqual(action, 'all')
            self.assertTrue(payload['resume'])
            self.assertNotIn('run_id', payload)

    def test_continue_local_update_uses_saved_action_scope(self):
        from plugin.project_state import ProjectState, write_json
        store = ProjectState(self.root)
        state = store.state()
        state.update(status='cancelled', action='rerun-period',
                     scope={'grid': '北区', 'periods': ['2022'], 'changes': ['2020_to_2022', '2022_to_20250118']},
                     parameters={'grid': '北区', 'period': '2022'})
        write_json(store.path, state)
        self.widget._refresh_results()
        self.assertEqual(self.widget.run_button.text(), '继续更新')
        self.assertIn('上次更新尚未完成', self.widget.processing_hint.text())
        self.assertIn('北区 / 2022', self.widget.processing_hint.text())
        self.assertFalse(self.widget.update_button.isEnabled())
        with patch.object(self.widget.controller, 'run') as run:
            self.widget.run_button.click()
            self.assertEqual(run.call_args.args[0], 'rerun-period')
            self.assertTrue(run.call_args.args[1]['resume'])

    def test_update_preview_uses_exact_shared_scope_and_starts_single_command(self):
        from plugin.project_state import ProjectState
        self.widget.update_button.click()
        page = self.widget.update_page
        self.assertEqual(self.widget.pages.currentWidget(), page)
        self.assertFalse(page.save_button.isEnabled())
        page.area.setCurrentText('北区')
        button = next(b for b in page.buttons.buttons() if b.selection.get('period') == '2022')
        before = (self.root / '_work/tasks/latest_pipeline.json').read_bytes()
        button.click()
        expected = ProjectState(self.root).local_scope('rerun-period', button.selection)
        self.assertEqual(page.scope, expected)
        self.assertEqual((self.root / '_work/tasks/latest_pipeline.json').read_bytes(), before)
        self.assertFalse((self.root / '_work/project_state.json').exists())
        self.assertIn('2020 → 2022', page.impact.text())
        self.assertIn('长时序成果', page.impact.text())
        self.assertIn('精度评价', page.impact.text())
        with patch.object(self.widget.controller, 'run') as run:
            page.save_button.click()
            action, data = run.call_args.args
            self.assertEqual((action, data['grid'], data['period']), ('rerun-period', '北区', '2022'))
            self.assertIn('--update-related', self.widget.controller.build_command(action, data))
            self.assertEqual(self.widget.pages.currentWidget(), self.widget.main_page)

    def test_change_update_is_mutually_exclusive_and_automatic_downstream(self):
        self.widget._show_update()
        page = self.widget.update_page
        page.buttons.buttons()[0].click()
        pair = next(b for b in page.buttons.buttons() if b.selection['action'] == 'rerun-change')
        pair.click()
        self.assertEqual(sum(b.isChecked() for b in page.buttons.buttons()), 1)
        self.assertEqual(page.scope['periods'], [])
        with patch.object(self.widget.controller, 'run') as run:
            page.save_button.click()
            action, data = run.call_args.args
            self.assertEqual(action, 'rerun-change')
            self.assertIn('--update-temporal', self.widget.controller.build_command(action, data))

    def test_summary_opens_local_output_without_reemitting_results(self):
        received = []
        self.plugin.result_ready.connect(received.append)
        self.widget._refresh_results()
        with patch('plugin.widget.QDesktopServices.openUrl', return_value=True) as open_url:
            self.widget.locate_button.click()
        self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()), Path(self.widget.output.text()))
        self.assertFalse(received)
        self.assertFalse(hasattr(self.widget, 'results_page'))
        self.assertFalse(hasattr(self.widget, 'group_buttons'))
        self.assertIn('1 个期次', self.widget.result_summary.text())

    def test_saved_configuration_marks_results_outdated_after_successful_check(self):
        self.widget.configuration.area.setCurrentText('北区')
        self.widget.configuration.reference.setCurrentText('2022')
        self.widget._save_and_check()
        self.wait_ready()
        self.assertTrue(self.widget.checked)
        self.assertIn('当前成果尚未更新', self.widget.configuration_note.text())

    def test_checked_parameter_change_is_saved_and_does_not_refresh_results(self):
        self.widget.parameters['absolute'].setText('4.0')
        self.widget.check_data()
        self.wait_ready()
        self.assertTrue(self.widget.checked)
        self.assertEqual(json.loads((self.root / 'project_config.json').read_text(encoding='utf8'))['plugin_processing_parameters']['absolute'], '4.0')
        self.assertIn('当前成果尚未更新', self.widget.configuration_note.text())
        self.widget._show_update()
        self.assertIn('上次已处理的数据', self.widget.update_page.configuration_note.text())
        self.widget._load_project(self.root)
        self.wait_ready()
        self.assertIn('当前成果尚未更新', self.widget.configuration_note.text())

    def test_evaluation_summary(self):
        report = self.root / 'metrics.json'
        report.write_text(json.dumps({'metrics': [{'class': 'all', 'precision': .9, 'recall': .8, 'f1': .847}]}))
        manifest = self.root / '_work/tasks/latest_pipeline.json'
        data = json.loads(manifest.read_text())
        data['evaluation_summary']['json'] = str(report)
        manifest.write_text(json.dumps(data))
        self.widget._refresh_results()
        self.assertIn('P 90%', self.widget.metrics.text())
        self.assertIn('F1 85%', self.widget.metrics.text())

    def test_status_is_structured_and_hidden_widget_keeps_processing(self):
        with patch.object(self.widget.controller, 'cancel') as cancel:
            self.widget._started({'task_id': 'ui-test'})
            self.assertTrue(self.widget.run_button.isHidden())
            self.assertFalse(self.widget.cancel_button.isHidden())
            self.widget._progress({'stage': '道路提取', 'progress': .8, 'event': {'kind': 'stage', 'grid': '北区', 'period': '2022'}})
            self.assertEqual(self.widget.progress.maximum(), 0)
            self.widget._show_configuration()
            self.widget.hide()
            self.widget._progress({'stage': '变化检测', 'event': {'kind': 'pipeline', 'grid': '南区', 'before_period': '2020', 'after_period': '2022', 'progress': .4}})
            self.assertEqual(self.widget.progress.value(), 400)
            self.assertEqual(self.widget.scope_text.text(), '2020 → 2022')
            self.widget._log({'message': 'completed failed 100%'})
            self.assertEqual(self.widget.state_text.text(), '运行中')
            self.assertTrue(self.widget.timer.isActive())
            cancel.assert_not_called()
        self.widget._failed({'message': '示例错误', 'detail': '错误详情'})
        self.assertIn('错误详情', self.widget.error_detail.toPlainText())
        self.assertFalse(self.widget.failure_details.isHidden())

    def test_narrow_pages_and_fixed_actions(self):
        self.widget.show()
        self.widget._show_configuration()
        self.widget._show_update()
        for width in (300, 360, 450, 680):
            self.widget.resize(width, 600)
            for page in (self.widget.main_page, self.widget.configuration, self.widget.creation, self.widget.update_page):
                self.widget.pages.setCurrentWidget(page)
                APP.processEvents()
                APP.processEvents()
                self.assertLessEqual(self.widget.minimumSizeHint().width(), 300)
                scroll = getattr(page, 'scroll', self.widget.scroll if page is self.widget.main_page else None)
                if isinstance(scroll, QScrollArea):
                    self.assertEqual(scroll.horizontalScrollBar().maximum(), 0, (width, type(page)))
                button = self.widget.run_button if page is self.widget.main_page else getattr(page, 'save_button', None)
                if button and not button.isHidden():
                    self.assertLess(button.mapTo(self.widget, button.rect().bottomRight()).y(), self.widget.height())
        self.assertFalse(self.widget.findChildren(QTabBar))
        self.assertEqual(self.widget.pages.count(), 4)

    def test_configuration_scroll_wheel_and_page_position(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        self.widget.show()
        self.widget.resize(300, 550)
        self.widget._show_configuration()
        APP.processEvents()
        panel = self.widget.configuration
        bar = panel.scroll.verticalScrollBar()
        bar.setValue(min(100, bar.maximum()))
        value = bar.value()
        self.widget._back_to_main()
        self.widget._show_configuration()
        APP.processEvents()
        self.assertEqual(bar.value(), value)
        old = panel.periods.verticalScrollBar().value()
        event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        APP.sendEvent(panel.periods.viewport(), event)
        self.assertGreaterEqual(bar.value(), value)
        self.assertEqual(panel.periods.verticalScrollBar().value(), old)

    def test_stale_check_cannot_replace_current_project(self):
        from threading import Event
        entered, release = Event(), Event()
        def delayed(data):
            entered.set()
            release.wait(3)
            return {'issues': []}
        with patch.object(self.widget.controller, 'inspect_data', side_effect=delayed):
            self.widget.check_data()
            self.assertTrue(entered.wait(1))
            empty = self.root / 'empty-project'
            empty.mkdir()
            (empty / 'project_config.json').write_text('{"validation_areas": []}')
            self.widget._load_project(empty)
            release.set()
            self.wait_ready()
        self.assertFalse(self.widget.checked)
        self.assertFalse(self.widget.locate_button.isEnabled())
        self.assertIn('请添加验证区', self.widget.check_note.text())

    def test_save_error_keeps_configuration_and_existing_config(self):
        self.widget._show_configuration()
        target = self.root / 'project_config.json'
        before = target.read_bytes()
        with patch('plugin.ui.project_browser.os.replace', side_effect=OSError('文件被占用')):
            self.widget._save_and_check()
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.configuration)
        self.assertIn('文件被占用', self.widget.check_note.text())

    def test_create_empty_then_configure_and_reopen(self):
        source = self.widget.configuration.inputs()
        self.widget._show_creation()
        page = self.widget.creation
        self.assertFalse(hasattr(page, 'areas'))
        self.assertFalse(hasattr(page, 'periods'))
        from PySide6.QtWidgets import QLineEdit
        self.assertEqual(len(page.findChildren(QLineEdit)), 2)
        page.project_name.setText('新项目')
        page.project_location.edit.setText(str(self.root))
        page.save_button.click()
        self.wait_ready()
        target = self.root / '新项目'
        for folder in ('01_验证区', '02_影像', '03_变化真值', '成果输出', '_work'):
            self.assertTrue((target / folder).is_dir())
        self.assertEqual(scan_project(target)['areas'], [])
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.configuration)
        self.assertEqual(self.widget.run_button.text(), '配置数据')
        source['periods'][0][-1] = str(self.root / '北区/02_影像/2020.tif')
        self.widget.configuration.load(source)
        self.widget.configuration.save_button.click()
        self.wait_ready()
        self.assertEqual(self.widget.pages.currentWidget(), self.widget.main_page)
        self.assertEqual(self.widget.run_button.text(), '开始处理')
        model = scan_project(target)
        self.assertEqual(model['area_irmad_references'], source['area_irmad_references'])
        self.assertFalse(list((target / '02_影像').rglob('*.tif')))
        with patch('plugin.widget.QFileDialog.getExistingDirectory', return_value=str(self.root)):
            self.widget.switch_button.click()
        self.wait_ready()
        self.assertTrue(self.widget.run_button.isHidden())
        self.assertTrue(self.widget.locate_button.isEnabled())
        with patch('plugin.widget.QFileDialog.getExistingDirectory', return_value=str(target)):
            self.widget.switch_button.click()
        self.wait_ready()
        self.assertEqual(self.widget.run_button.text(), '开始处理')
        self.assertEqual(len(self.widget.model['periods']), 6)

    def test_creation_collision_and_failed_write_preserve_project(self):
        from plugin.ui.project_browser import create_project
        empty = dict(areas=[], periods=[], truths=[], area_irmad_references={})
        before = (self.root / 'project_config.json').read_bytes()
        with self.assertRaises(ValueError):
            create_project(self.root.parent, self.root.name, empty)
        self.assertEqual((self.root / 'project_config.json').read_bytes(), before)
        for name in ('../outside', 'bad/name', 'CON', ''):
            with self.assertRaises(ValueError):
                create_project(self.root, name, empty)
        with patch('plugin.ui.project_browser.os.replace', side_effect=OSError('写入失败')):
            with self.assertRaises(OSError):
                create_project(self.root, '失败项目', empty)
        self.assertFalse((self.root / '失败项目').exists())

    def test_host_style_and_auxiliary_visibility(self):
        app_style = APP.styleSheet()
        self.assertFalse(self.widget.advanced.toggle.isChecked())
        self.assertFalse(self.widget.records_fold.toggle.isChecked())
        self.widget.setProperty('hostStyled', True)
        self.assertEqual(self.widget.styleSheet(), '')
        self.widget.setProperty('hostStyled', False)
        self.assertTrue(self.widget.styleSheet())
        self.assertEqual(APP.styleSheet(), app_style)
        self.assertTrue(self.widget.run_button.icon().isNull())


if __name__ == '__main__':
    unittest.main()
