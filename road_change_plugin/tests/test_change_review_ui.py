"""Review UI state/unsaved-edit guards; synthetic image replies, no GIS imports."""
import base64
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QObject, Signal, QByteArray, QBuffer, QIODevice, QTimer
from PySide6.QtGui import QImage, QColor
from PySide6.QtWidgets import QApplication, QMessageBox, QTableWidget, QPushButton
from plugin.ui.change_review import ChangeReviewDialog, change_text, PAIR_GRID_ROLE

APP = QApplication.instance() or QApplication([])


class FakeService(QObject):
    replied = Signal(str, object)
    failed = Signal(str, str)
    stopped = Signal(str)
    progress = Signal(str)

    def __init__(self):
        super().__init__()
        self.active = True
        self.pending = ""
        self.calls = []
        self.auto_record = True

    def request(self, op, **data):
        if self.pending:
            raise ValueError("已有请求")
        self.pending = op
        self.calls.append((op, data))
        if op == 'record' and self.auto_record:
            self.reply({'recorded': True})

    def reply(self, data):
        op, self.pending = self.pending, ""
        if op == 'save':
            data.setdefault('published_signature', 'published')
        self.replied.emit(op, data)

    def close(self):
        self.active = False


def image_reply(index):
    image = QImage(64, 64, QImage.Format.Format_RGB32)
    image.fill(QColor("#405060"))
    output = QByteArray()
    buffer = QBuffer(output)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    data = base64.b64encode(bytes(output)).decode("ascii")
    return {"id": str(index), "images": [data, data], "size": 64, "span_m": 60, "warnings": [],
            "has_images": True, "geometry": {"type": "LineString", "coordinates": [[20, 20], [45, 45]]}}


class ReviewUiTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.dialog = ChangeReviewDialog(self.service, None)
        self.dialog.show()
        self.service.pending = "init"
        self.service.reply({"pairs": [{"grid": "北区", "before_period": "2020", "after_period": "2022"},
                                     {"grid": "南区", "before_period": "2020", "after_period": "2022"}]})
        self.assertEqual(self.service.pending, "open")
        self.data = {"pair": {"before_period": "2020", "after_period": "2022"}, "saved_path": "",
                     'source_signature': 'original', 'published_signature': 'published',
                     "warning": "", "image_warnings": [], "features": [
                         {"id": str(i), "change_type": "added", "status": "pending", "attributes": {"road_id": str(i)}}
                         for i in range(3)]}
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()

    def finish_preview(self):
        APP.processEvents()
        if self.service.pending == "preview":
            self.service.reply(image_reply(self.service.calls[-1][1]["id"]))
        APP.processEvents()

    def tearDown(self):
        self.dialog.dirty = False
        self.dialog.drafts.clear()
        self.dialog._save_queue = None
        self.dialog.record_unsaved = False
        self.service.pending = ""
        self.dialog.close()
        self.dialog.dispose()
        self.dialog.deleteLater()
        APP.processEvents()

    def test_decisions_undo_filter_and_save(self):
        self.assertTrue(self.dialog.keep.isEnabled())
        self.dialog.reject_button.click()
        self.assertEqual(self.dialog.features[0]["status"], "reject")
        self.assertTrue(self.dialog.dirty)
        self.dialog.undo.click()
        self.finish_preview()
        self.assertEqual(self.dialog.features[0]["status"], "pending")
        self.assertFalse(self.dialog.dirty)
        self.dialog.keep.click()
        self.dialog.filter.setCurrentIndex(1)
        self.finish_preview()
        self.assertEqual(self.dialog.items.currentRow(), 1)
        self.dialog.reject_button.click()
        self.finish_preview()
        self.assertEqual(self.dialog.items.currentRow(), 2)
        self.dialog.save_button.click()
        self.assertEqual(self.service.calls[-1], ("save", {"decisions": {"0": "keep", "1": "reject", "2": "pending"}}))
        self.assertFalse(self.dialog.items.isEnabled())
        self.service.reply({"path": "C:/review/road_changes_reviewed.shp", "counts": {"keep": 1, "reject": 1, "pending": 1}})
        self.assertFalse(self.dialog.dirty)
        self.assertTrue(self.dialog.open_button.isEnabled())

    def test_pair_switch_preserves_unsaved_decisions_and_save_failure(self):
        self.dialog.keep.click()
        with patch.object(QMessageBox, "exec") as prompt:
            self.dialog.pairs.setCurrentIndex(1)
            prompt.assert_not_called()
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.assertEqual(self.dialog.pairs.currentIndex(), 1)
        self.assertTrue(self.dialog.dirty)
        self.dialog.pairs.setCurrentIndex(0)
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.assertEqual(self.dialog.features[0]['status'], 'keep')
        self.dialog.save_button.click()
        self.assertEqual(self.service.pending, "save")
        self.service.pending = ""
        self.service.failed.emit("save", "磁盘空间不足")
        self.assertTrue(self.dialog.dirty)
        self.assertEqual(self.dialog.pairs.currentIndex(), 0)

    def test_autosaved_judgements_close_without_applying(self):
        self.dialog.keep.click()
        with patch.object(QMessageBox, "exec") as prompt:
            self.dialog.reject()
            prompt.assert_not_called()
        self.assertFalse(self.dialog.isVisible())
        self.assertTrue(any(op == 'record' for op, _ in self.service.calls))
        self.assertFalse(any(op == 'save' for op, _ in self.service.calls))
        QTimer.singleShot(0, self.dialog.close)
        self.dialog.exec()
        self.assertFalse(self.dialog.isVisible())

    def test_missing_preview_disables_decisions_and_clears_stale_image(self):
        self.dialog.items.setCurrentRow(1)
        APP.processEvents()
        self.assertFalse(self.dialog.keep.isEnabled())
        self.service.pending = ""
        self.service.failed.emit("preview", "缺少影像")
        self.assertFalse(self.dialog.reject_button.isEnabled())
        self.assertEqual(len(self.dialog.views[0].scene().items()), 0)

    def test_overlay_and_zoom_are_linked(self):
        self.dialog.overlay_check.setChecked(False)
        self.assertTrue(all(not v.overlay.isVisible() for v in self.dialog.views))
        self.dialog.views[0].scale(2, 2)
        self.dialog.views[0].moved.emit()
        self.assertAlmostEqual(self.dialog.views[0].transform().m11(), self.dialog.views[1].transform().m11())

    def test_empty_pair_shows_completion_without_redundant_apply(self):
        self.dialog.pairs.setCurrentIndex(1)
        empty = copy.deepcopy(self.data)
        empty['features'] = []
        self.service.reply(empty)
        self.assertFalse(self.dialog.keep.isEnabled())
        self.assertFalse(self.dialog.save_button.isEnabled())
        self.assertTrue(self.dialog.completion.isVisible())

    def test_customer_labels_hide_raw_fields_and_unknown_types(self):
        for value, expected in (("added", "道路新增"), ("removed", "道路消失"),
                                ("widened", "道路变宽"), ("narrowed", "道路变窄"),
                                ("internal_unknown", "道路变化")):
            self.assertEqual(change_text(value), expected)
        self.assertIn("道路新增", self.dialog.items.item(0).text())
        self.assertNotIn("added", self.dialog.items.item(0).text())
        self.assertNotIn("road_id", self.dialog.details.text())
        self.assertIn('候选 0001', self.dialog.details.text())

    def test_decision_advances_immediately_and_does_not_double_confirm(self):
        self.dialog.pairs.removeItem(1)
        self.dialog.keep.click()
        self.assertEqual(self.dialog.items.currentRow(), 1)
        self.assertFalse(self.dialog.keep.isEnabled())
        self.dialog.keep.click()
        self.assertEqual(self.dialog.features[1]['status'], 'pending')
        self.finish_preview()
        self.dialog.reject_button.click()
        self.assertEqual(self.dialog.items.currentRow(), 2)
        self.finish_preview()
        self.dialog.keep.click()
        self.assertEqual(self.dialog.items.currentRow(), 2)
        self.assertIn('完成复核', self.dialog.message.text())
        self.assertIn('保存', self.dialog.message.text())
        self.assertFalse(self.dialog.next_pending.isEnabled())

    def test_skipped_entries_are_revisited_and_reviewed_entries_are_skipped(self):
        self.dialog.next_pending.trigger()
        self.assertEqual(self.dialog.items.currentRow(), 1)
        self.assertEqual(self.dialog.features[0]['status'], 'pending')
        self.assertFalse(self.dialog.dirty)
        self.finish_preview()
        self.dialog.keep.click()
        self.finish_preview()
        self.dialog.reject_button.click()
        self.assertEqual(self.dialog.items.currentRow(), 0)
        self.finish_preview()
        self.dialog.next_pending.trigger()
        self.assertEqual(self.dialog.items.currentRow(), 0)
        self.assertIn('只剩这一条', self.dialog.message.text())

    def test_pending_filter_finishes_and_undo_returns_to_last_decision(self):
        self.dialog.pairs.removeItem(1)
        self.dialog.filter.setCurrentIndex(1)
        for index in range(3):
            self.assertEqual(self.dialog.items.currentRow(), index)
            self.dialog.keep.click()
            self.finish_preview()
        self.assertIn('完成复核', self.dialog.message.text())
        self.dialog.undo.click()
        self.assertEqual(self.dialog.items.currentRow(), 2)
        self.assertEqual(self.dialog.features[2]['status'], 'pending')
        self.assertIn('撤销', self.dialog.message.text())
        self.assertIn('待复核', self.dialog.record_label.text())

    def test_resume_starts_at_first_unreviewed_and_reset_stays_on_item(self):
        resumed = copy.deepcopy(self.data)
        resumed['features'][0]['status'] = 'keep'
        resumed['features'][1]['status'] = 'reject'
        self.service.pending = 'open'
        self.service.reply(resumed)
        self.finish_preview()
        self.assertEqual(self.dialog.items.currentRow(), 2)
        self.dialog.items.setCurrentRow(0)
        self.finish_preview()
        self.dialog.reset.trigger()
        self.assertEqual(self.dialog.items.currentRow(), 0)
        self.assertEqual(self.dialog.features[0]['status'], 'pending')

    def test_decision_from_filtered_reviewed_list_continues_pending_work(self):
        self.dialog.keep.click()
        self.finish_preview()
        self.dialog.filter.setCurrentIndex(2)
        self.finish_preview()
        self.assertEqual(self.dialog.items.currentRow(), 0)
        self.dialog.reject_button.click()
        self.assertEqual(self.dialog.items.currentRow(), 1)
        self.assertFalse(self.dialog.items.item(1).isHidden())

    def test_save_waits_for_downstream_before_closing_and_reports_evaluation(self):
        self.dialog.keep.click()
        self.finish_preview()
        self.dialog.save_button.click()
        self.service.progress.emit('正在重新计算精度')
        self.assertIn('重新计算精度', self.dialog.message.text())
        self.assertFalse(self.dialog.save_button.isEnabled())
        for close in (self.dialog.close, self.dialog.reject):
            close()
            prompt = next(p for p in self.dialog.findChildren(QMessageBox) if p.isVisible())
            self.assertEqual(prompt.icon(), QMessageBox.Icon.Warning)
            self.assertEqual(prompt.windowTitle(), '正在应用复核结果')
            self.assertIn('等待完成后再关闭', prompt.text())
            button = prompt.button(QMessageBox.StandardButton.Ok)
            self.assertEqual(button.text(), '继续等待')
            self.assertEqual(prompt.defaultButton(), button)
            self.assertEqual(prompt.escapeButton(), button)
            button.click()
            APP.processEvents()
            self.assertTrue(self.dialog.isVisible())
            self.assertTrue(self.dialog.dirty)
            self.assertTrue(self.service.active)
            self.assertEqual(self.service.pending, 'save')
        self.service.reply({'path': 'C:/review/road_changes.shp', 'counts': {'keep': 1, 'reject': 0, 'pending': 2},
                            'evaluation_state': 'completed', 'temporal_updated': True})
        self.assertFalse(self.dialog.dirty)
        self.assertIn('精度评价已重新计算', self.dialog.message.text())
        self.assertIn('长时序和统计已更新', self.dialog.message.text())
        self.dialog.close()
        self.assertFalse(self.dialog.isVisible())

    def test_exit_during_save_closes_without_claiming_saved(self):
        self.dialog.keep.click()
        self.finish_preview()
        self.dialog.save_button.click()
        self.dialog.close()
        prompt = next(p for p in self.dialog.findChildren(QMessageBox) if p.isVisible())
        self.assertIn('判断记录已保存', prompt.informativeText())
        button = prompt.button(QMessageBox.StandardButton.Abort)
        self.assertEqual(button.text(), '仍然退出')
        button.click()
        APP.processEvents()
        self.assertFalse(self.dialog.isVisible())
        self.assertTrue(self.dialog.dirty)
        self.assertEqual(self.dialog.result(), self.dialog.DialogCode.Rejected)

    def complete_current_pair(self):
        for _ in range(3):
            self.dialog.keep.click()
            self.finish_preview()

    def saved_reply(self):
        counts = {state: sum(f['status'] == state for f in self.dialog.features)
                  for state in ('keep', 'reject', 'pending')}
        self.service.reply({'path': 'C:/review/changes.shp', 'counts': counts,
                            'evaluation_state': 'completed', 'published_signature': 'published'})

    def test_auto_next_pair_keeps_drafts_and_undo_returns_across_pairs(self):
        self.complete_current_pair()
        self.assertEqual(self.service.calls[-1], ('open', {'index': 1}))
        self.assertFalse(any(op == 'save' for op, _ in self.service.calls))
        self.assertTrue(self.dialog.dirty)
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.assertEqual(self.dialog.pair_index, 1)
        self.assertEqual(self.dialog.features[0]['status'], 'pending')
        self.dialog.undo.click()
        self.assertEqual(self.service.calls[-1], ('open', {'index': 0}))
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.assertEqual([f['status'] for f in self.dialog.features], ['keep', 'keep', 'pending'])
        self.assertEqual(self.dialog.items.currentRow(), 2)

    def test_auto_next_orders_periods_within_region_then_changes_region(self):
        self.dialog.close()
        self.dialog.dispose()
        self.dialog.deleteLater()
        self.service = FakeService()
        self.dialog = ChangeReviewDialog(self.service, None)
        self.dialog.show()
        self.service.pending = 'init'
        self.service.reply({'pairs': [
            {'grid': '南区', 'before_period': '2022', 'after_period': '2024'},
            {'grid': '北区', 'before_period': '2022', 'after_period': '2024'},
            {'grid': '北区', 'before_period': '2020', 'after_period': '2022'},
            {'grid': '南区', 'before_period': '2020', 'after_period': '2022'}]})
        self.assertEqual(self.service.calls[-1], ('open', {'index': 2}))
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.complete_current_pair()
        self.assertEqual(self.service.calls[-1], ('open', {'index': 1}))
        self.assertIn('北区', self.dialog.pairs.currentText())
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.complete_current_pair()
        self.assertEqual(self.service.calls[-1], ('open', {'index': 3}))
        self.assertIn('南区', self.dialog.pairs.currentText())

    def test_auto_next_skips_saved_and_empty_pairs_and_stops_at_end(self):
        self.dialog.pairs.addItem('南区 / 2022 → 2024', 2)
        self.complete_current_pair()
        completed = copy.deepcopy(self.data)
        for feature in completed['features']:
            feature['status'] = 'keep'
        self.service.reply(completed)
        self.assertEqual(self.service.calls[-1], ('open', {'index': 2}))
        empty = copy.deepcopy(self.data)
        empty['features'] = []
        self.service.reply(empty)
        APP.processEvents()
        self.assertEqual(self.service.pending, '')
        self.assertIn('所有区域与期次已完成复核', self.dialog.message.text())
        self.assertTrue(self.dialog.dirty)

    def test_auto_next_finishes_earlier_pairs_in_same_region_first(self):
        self.dialog.pairs.addItem('北区 / 2022 → 2024', 2)
        self.dialog.pairs.setItemData(2, '北区', PAIR_GRID_ROLE)
        self.dialog.pairs.setCurrentIndex(2)
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.complete_current_pair()
        self.assertEqual(self.service.calls[-1], ('open', {'index': 0}))
        self.assertIn('北区', self.dialog.pairs.currentText())

    def test_close_during_record_waits_for_ack_not_application(self):
        self.service.auto_record = False
        self.dialog.keep.click()
        self.assertEqual(self.service.pending, 'record')
        self.assertEqual(self.dialog.items.currentRow(), 0)
        self.dialog.close()
        self.assertTrue(self.dialog.isVisible())
        self.service.reply({'recorded': True})
        self.assertFalse(self.dialog.isVisible())
        self.assertFalse(any(op == 'save' for op, _ in self.service.calls))

    def test_failed_autosave_blocks_navigation_and_retries_on_close(self):
        self.service.auto_record = False
        self.dialog.keep.click()
        self.service.pending = ''
        self.service.failed.emit('record', '磁盘已满')
        self.assertTrue(self.dialog.record_unsaved)
        self.assertTrue(self.dialog.retry_record.isVisible())
        self.assertFalse(self.dialog.pairs.isEnabled())
        self.assertFalse(self.dialog.save_button.isEnabled())
        with patch.object(QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Cancel):
            self.dialog.close()
        self.assertTrue(self.dialog.isVisible())
        with patch.object(QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Save):
            self.dialog.close()
        self.assertEqual(self.service.pending, 'record')
        self.service.reply({'recorded': True})
        self.assertFalse(self.dialog.isVisible())

    def test_apply_includes_persisted_unvisited_pairs(self):
        self.dialog.pair_states[1] = {'known': True, 'modified': True,
            'counts': {'keep': 1, 'reject': 1, 'pending': 1}}
        self.dialog._remember_pair()
        self.dialog._counts()
        self.dialog._controls()
        self.dialog.save_button.click()
        self.assertEqual(self.service.calls[-1], ('open', {'index': 1}))
        data = copy.deepcopy(self.data)
        data['features'][0]['status'] = 'keep'
        data['features'][1]['status'] = 'reject'
        data['applied_decisions'] = {str(i): 'pending' for i in range(3)}
        self.service.reply(data)
        self.assertEqual(self.service.calls[-1], ('save', {'decisions': {'0': 'keep', '1': 'reject', '2': 'pending'}}))
        self.saved_reply()
        self.assertFalse(self.dialog.dirty)

    def test_completed_group_emphasizes_summary_and_allows_revision(self):
        self.dialog.pairs.removeItem(1)
        self.complete_current_pair()
        self.assertTrue(self.dialog.completion.isVisible())
        self.assertFalse(self.dialog.decision_bar.isVisible())
        self.assertFalse(self.dialog.keep.isEnabled())
        self.dialog._edit_completed()
        self.assertTrue(self.dialog.decision_bar.isVisible())
        self.dialog.reject_button.click()
        self.assertEqual(self.dialog.features[2]['status'], 'reject')
        self.assertFalse(self.dialog.record_unsaved)

    def test_batch_failure_keeps_remaining_drafts_and_does_not_close(self):
        self.dialog.keep.click()
        self.dialog.pairs.setCurrentIndex(1)
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.dialog.keep.click()
        self.finish_preview()
        self.dialog.save_button.click()
        self.service.reply(copy.deepcopy(self.data))
        self.saved_reply()
        self.service.reply(copy.deepcopy(self.data))
        self.service.pending = ''
        self.service.failed.emit('save', '磁盘空间不足')
        self.assertTrue(self.dialog.isVisible())
        self.assertTrue(self.dialog.dirty)
        self.assertEqual(self.dialog.drafts[0]['decisions'], self.dialog.drafts[0]['saved'])
        self.assertNotEqual(self.dialog.drafts[1]['decisions'], self.dialog.drafts[1]['saved'])
        self.dialog.save_button.click()
        self.assertEqual(self.service.pending, 'save')
        self.saved_reply()
        self.assertFalse(self.dialog.dirty)

    def test_changed_data_does_not_receive_a_stashed_decision(self):
        self.dialog.keep.click()
        self.dialog.pairs.setCurrentIndex(1)
        self.service.reply(copy.deepcopy(self.data))
        self.finish_preview()
        self.dialog.save_button.click()
        changed = copy.deepcopy(self.data)
        changed['source_signature'] = 'new-detection'
        self.service.reply(changed)
        self.assertFalse(any(op == 'save' for op, _ in self.service.calls))
        self.assertTrue(self.dialog.dirty)
        self.assertIn('检测结果已改变', self.dialog.message.text())

    def test_summary_exposes_pending_application_and_returns_to_review(self):
        self.dialog.keep.click()
        self.finish_preview()
        captured = []
        def inspect():
            summary = APP.activeModalWidget()
            table = summary.findChild(QTableWidget)
            captured.append([table.item(0, col).text() for col in range(5)])
            button = next(b for b in summary.findChildren(QPushButton) if b.text() == '返回所选组复核')
            button.click()
        QTimer.singleShot(0, inspect)
        self.dialog._summary()
        self.assertEqual(captured[0][1:], ['2', '1', '0', '待应用'])
        self.assertTrue(self.dialog._editing_complete)
        self.assertFalse(any(op == 'save' for op, _ in self.service.calls))

    def test_unvisited_stale_record_aborts_apply(self):
        self.dialog.pair_states[1] = {'known': True, 'modified': True}
        self.dialog._save()
        stale = copy.deepcopy(self.data)
        stale['warning'] = '检测结果已改变，请重新复核'
        self.service.reply(stale)
        self.assertFalse(any(op == 'save' for op, _ in self.service.calls))
        self.assertIn('检测结果已改变', self.dialog.message.text())


if __name__ == "__main__":
    unittest.main()
