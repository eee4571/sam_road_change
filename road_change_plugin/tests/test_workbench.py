"""Workbench layout and diagnostics, synthetic data, no inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_ui_processing as helpers
from PySide6.QtCore import QTimer, QRect
from PySide6.QtWidgets import QScrollArea, QMessageBox, QPlainTextEdit, QPushButton, QLabel
from plugin.runner import Runner
from plugin.diagnostics import failure_summary
from plugin.ui.run_log import log_entry
from plugin.ui.task_overview import stage_states, product_inventory
from plugin.ui.evaluation_summary import evaluation_context

APP = helpers.APP


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.case = helpers.ProcessingUiTests()
        self.case.setUp()
        self.w = self.case.widget
        self.w.show()
        self.w.resize(680, 700)
        APP.processEvents()

    def tearDown(self):
        self.case.tearDown()
        self.case.doCleanups()

    def test_log_is_outside_business_scroll_and_keeps_position(self):
        self.w.scroll.verticalScrollBar().setValue(0)
        self.w._show_logs()
        APP.processEvents()
        self.assertFalse(self.w.scroll.isAncestorOf(self.w.log_panel))
        self.assertEqual(self.w.scroll.verticalScrollBar().value(), 0)
        self.assertEqual(self.w.workspace.count(), 2)
        self.assertFalse(self.w.log_panel.details.isVisible())
        self.assertFalse(self.w.log_panel.findChildren(QScrollArea))
        for width in (300, 360, 680):
            self.w.resize(width, 650)
            APP.processEvents()
            APP.processEvents()
            self.assertEqual(self.w.scroll.horizontalScrollBar().maximum(), 0)
            self.assertLessEqual(self.w.minimumSizeHint().width(), 300)
        self.w.workspace.setSizes([300, 210])
        APP.processEvents()
        height = self.w.workspace.sizes()[1]
        self.w._show_logs()
        self.w._show_logs()
        APP.processEvents()
        self.assertAlmostEqual(self.w.workspace.sizes()[1], height, delta=8)

    def test_model_parameter_dump_is_hidden_but_full_log_is_preserved(self):
        messages = ["'image_encoder.blocks.1.attn.qkv.bias',", '任务开始', 'WARNING: 影像覆盖不足']
        for message in messages:
            self.w._log({'message': message})
        self.assertNotIn('image_encoder', self.w.log_panel.text.toPlainText())
        self.assertIn('任务开始', self.w.log_panel.text.toPlainText())
        self.assertIn('1 警告', self.w.log_counts.text())
        self.assertIn('image_encoder', '\n'.join(self.w._log_lines))
        self.w.log_panel.filter.setCurrentIndex(2)
        self.assertIn('image_encoder', self.w.log_panel.text.toPlainText())
        self.w.log_panel.filter.setCurrentIndex(1)
        self.assertNotIn('任务开始', self.w.log_panel.text.toPlainText())
        self.assertIn('影像覆盖不足', self.w.log_panel.text.toPlainText())

    def test_error_details_are_lazy_and_summary_does_not_guess_success(self):
        self.w._failed({'message': '后端退出，代码 1', 'detail': 'RuntimeError: Error(s) in loading state_dict: size mismatch'})
        self.assertIn('模型加载失败', self.w.task_line.text())
        self.assertEqual(self.w.process_badge.text(), '失败')
        self.assertNotIn('已完成', [label.text() for label in self.w.stages.values])
        self.assertTrue(self.w.log_panel.details.isVisible())
        self.assertTrue(self.w.log_toggle.isChecked())
        self.assertFalse(self.w.log_panel.findChildren(QPlainTextEdit))
        captured = []
        def inspect():
            dialog = APP.activeModalWidget()
            captured.append(dialog.findChild(QPlainTextEdit).toPlainText())
            dialog.accept()
        QTimer.singleShot(0, inspect)
        self.w.log_panel.show_details()
        self.assertIn('size mismatch', captured[0])

    def test_missing_evaluation_and_zero_are_distinct(self):
        self.assertEqual(self.w.metric_cards.values['recall'].text(), '未评价')
        report = self.case.root / 'zero.json'
        report.write_text(json.dumps({'metrics': [{'class': 'all', 'change_recall': 0}]}))
        self.w.metric_cards.refresh(report)
        self.assertEqual(self.w.metric_cards.values['recall'].text(), '0.0')
        self.assertEqual(self.w.metric_cards.units['recall'].text(), '%')
        self.assertEqual(self.w.metric_cards.values['precision'].text(), '未评价')

    def test_warning_focus_keeps_completed_state_and_business_position(self):
        self.w.log_panel.reset()
        self.w._log({'level': 'WARNING', 'message': '部分区域缺少评价数据'})
        for i in range(80):
            self.w._log({'message': f'后续处理记录 {i}'})
        self.w.scroll.verticalScrollBar().setValue(0)
        self.w._show_logs()
        APP.processEvents()
        self.assertEqual(self.w.process_badge.text(), '已完成')
        self.assertEqual(self.w.log_counts.property('status'), 'warning')
        self.assertIn('部分区域缺少评价数据', self.w.log_panel.text.textCursor().block().text())
        self.assertEqual(self.w.scroll.verticalScrollBar().value(), 0)
        self.assertFalse(self.w.log_panel.details.isVisible())
        self.w._show_logs()
        self.w._log({'level': 'ERROR', 'message': '写入失败'})
        self.w._log({'level': 'WARNING', 'message': '后续警告'})
        self.w._show_logs()
        self.assertIn('写入失败', self.w.log_panel.text.textCursor().block().text())
        self.assertEqual(self.w.log_counts.property('status'), 'error')

    def test_recent_project_reopens_and_missing_path_does_not_switch(self):
        self.w._load_project('')
        self.case.wait_ready()
        self.assertIn('尚未打开项目', [item.text() for item in self.w.project_entry.findChildren(QLabel)])
        buttons = self.w.recent_projects.findChildren(QPushButton)
        self.assertEqual(len(buttons), 1)
        self.assertEqual(buttons[0].toolTip(), str(self.case.root))
        buttons[0].click()
        self.case.wait_ready()
        self.assertEqual(self.w.model['root'], str(self.case.root))
        self.assertEqual(self.w._recent_paths(), [str(self.case.root)])
        with patch.object(QMessageBox, 'warning') as warning:
            self.w._open_recent(str(self.case.root / 'missing'))
        warning.assert_called_once()
        self.assertEqual(self.w.model['root'], str(self.case.root))
        self.case.settings.setValue('recentProjects', [])
        self.w._load_project('')
        self.case.wait_ready()
        self.assertTrue(self.w.recent_projects.isHidden())

    def test_result_actions_share_row_and_wrap_without_clipping(self):
        self.assertTrue(self.w.processing_hint.isHidden())
        self.assertTrue(self.w.evaluation_note.isHidden())
        buttons = [self.w.review_button, self.w.report_button, self.w.locate_button]
        self.w.resize(780, 750)
        APP.processEvents()
        self.assertEqual(len({button.y() for button in buttons}), 1)
        for width in (300, 360, 680):
            self.w.resize(width, 650)
            APP.processEvents()
            APP.processEvents()
            self.assertEqual(self.w.scroll.horizontalScrollBar().maximum(), 0)
            for button in buttons:
                self.assertTrue(self.w.result_actions.rect().contains(button.geometry()))

    def test_default_screen_fit_and_home_overview(self):
        from standalone import initial_window_geometry
        for available in (QRect(0, 0, 1920, 1040), QRect(0, 0, 1366, 728), QRect(-1280, 0, 1280, 680)):
            geometry = initial_window_geometry(available)
            self.assertTrue(available.contains(geometry))
            self.w.setGeometry(geometry)
            APP.processEvents()
            APP.processEvents()
            self.assertEqual(self.w.scroll.verticalScrollBar().maximum(), 0)
            self.assertEqual(self.w.scroll.horizontalScrollBar().maximum(), 0)
            self.assertTrue(self.w.scroll.viewport().rect().contains(
                self.w.locate_button.mapTo(self.w.scroll.viewport(), self.w.locate_button.rect().bottomRight())))
        self.assertEqual(initial_window_geometry(QRect(0, 0, 1920, 1040)).size().toTuple(), (820, 600))

    def test_metrics_switch_rows_and_keep_units(self):
        for width, columns in ((1280, 5), (360, 3), (1280, 5)):
            self.w.resize(width, 700)
            APP.processEvents()
            APP.processEvents()
            self.assertEqual(self.w.metric_cards.columns, columns)
            self.assertEqual(self.w.scroll.horizontalScrollBar().maximum(), 0)
        report = self.case.root / 'offset.json'
        report.write_text(json.dumps({'metrics': [{'class': 'all', 'centerline_avg_offset_m': 2.48}]}))
        self.w.metric_cards.refresh(report)
        self.assertEqual(self.w.metric_cards.values['offset'].text(), '2.48')
        self.assertEqual(self.w.metric_cards.units['offset'].text(), 'm')
        self.assertEqual(self.w.metric_cards.values['offset'].accessibleName(), '2.48 m')

    def test_stage_failure_missing_file_and_changed_configuration(self):
        descriptor = self.w._current
        self.assertEqual(stage_states(None, {}), ['未运行'] * 3)
        failure = stage_states(descriptor, {'status': 'failed'}, failed=True, active_stage='道路面提取')
        self.assertEqual(failure[1], '失败')
        self.assertEqual(stage_states(descriptor, {'status': 'completed'}, outdated=True), ['成果待更新'] * 3)
        Path(descriptor['data']['period_results'][0]['published']['width_segments']).unlink()
        self.assertEqual(stage_states(descriptor, {'status': 'completed'})[1], '成果失效')

    def test_failed_stage_restores_from_explicit_event_only(self):
        self.w._last_error = '处理失败'
        self.w._log_lines = ['[INFO] ' + json.dumps({'kind': 'stage', 'stage': '道路测宽'}),
                            '[ERROR] 处理失败']
        self.w._show_record()
        self.w._update_controls()
        self.assertEqual(self.w.stages.values[1].text(), '失败')
        self.w._log_lines = ['[INFO] 变化检测失败 道路提取完成']
        self.w._show_record()
        self.w._update_controls()
        self.assertNotIn('失败', [item.text() for item in self.w.stages.values])

    def test_rerun_cancel_does_not_start_or_modify_results(self):
        self.w._show_update()
        page = self.w.update_page
        page.buttons.buttons()[0].click()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No), patch.object(self.w.controller, 'run') as run:
            page.save_button.click()
        run.assert_not_called()
        self.assertFalse(self.w.busy)
        self.assertEqual(self.w.pages.currentWidget(), page)

    def test_stages_do_not_complete_from_one_period_or_old_products(self):
        descriptor = self.w._current
        states = stage_states(descriptor, {'status': 'completed'})
        self.assertEqual(states, ['已完成'] * 3)
        active = stage_states(descriptor, {'status': 'running'}, busy=True, active_stage='道路提取')
        self.assertEqual(active[0], '进行中')
        self.assertNotIn('已完成', active)
        data = dict(descriptor['data'])
        data['period_orders'] = {'北区': ['2020', '2022', '20250118'], '南区': ['2020', '2022', '20250118']}
        incomplete = stage_states({**descriptor, 'data': data}, {'status': 'completed'})
        self.assertEqual(incomplete[2], '已有 1/4')

    def test_saved_order_metadata_and_internal_width_products(self):
        descriptor = self.w._current
        data = json.loads(json.dumps(descriptor['data']))
        data['period_results'] = data['period_results'][:2]
        data['period_orders'] = {'北区': {
            'period_order': ['2020', '2022'], 'change_pairs': [['2020', '2022']],
            'periods': [{'value': '2020'}, {'value': '2022'}], 'custom_order_warning': False,
        }}
        for entry in data['period_results']:
            entry['width_segments'] = entry['published'].pop('width_segments')
        descriptor = {**descriptor, 'data': data}
        self.assertEqual(product_inventory(descriptor), ([2, 2, 1], [2, 2, 1], [False]*3))
        self.assertEqual(stage_states(descriptor, {'status': 'completed'}), ['已完成']*3)


class DiagnosticTests(unittest.TestCase):
    def test_model_error_header_survives_long_parameter_tail(self):
        runner = Runner()
        runner._terminal = False
        runner.consume_line('RuntimeError: Error(s) in loading state_dict for Model:')
        for i in range(250):
            runner.consume_line(f"'image_encoder.blocks.{i}.weight',")
        failures = []
        runner.task_failed.connect(failures.append)
        runner._fail('后端退出，代码 1')
        self.assertIn('模型加载失败', failure_summary(failures[0]['message'], failures[0]['detail']))

    def test_console_hints_do_not_change_task_state(self):
        runner = Runner()
        runner.state = 'running'
        runner.consume_line('RuntimeError: size mismatch')
        self.assertEqual(runner.state, 'running')
        self.assertEqual(runner._failure, '')

    def test_event_progress_is_readable_and_warning_retains_level(self):
        level, message = log_entry({'event': {'kind': 'stage', 'stage': '道路提取', 'grid': '北区', 'status': 'complete'}})
        self.assertEqual(level, 'PROGRESS')
        self.assertIn('北区 · 道路提取 · 已完成', message)
        self.assertEqual(log_entry({'event': {'kind': 'warning', 'message': '缺少覆盖'}}), ('WARNING', '缺少覆盖'))

    def test_evaluation_context_does_not_invent_version_or_evaluation_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report.json'
            report.write_text('{}')
            text = evaluation_context({'data': {'input_spec': {'truths': {'北区': {'path': 'truth.shp'}}}}}, report)
            self.assertIn('版本未记录', text)
            self.assertIn('报告文件更新时间', text)
            self.assertNotIn('评价时间：', text)


if __name__ == '__main__':
    unittest.main()
