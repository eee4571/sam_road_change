"""Batch import acceptance with synthetic lists and empty raster placeholders."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from plugin.ui.image_import import dates, import_periods, folder_files
from plugin.ui.data_configuration import DataConfiguration

APP = QApplication.instance() or QApplication([])


class ImageImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, name, content=''):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf8')
        return path

    def panel(self):
        self.changes = []
        page = DataConfiguration(lambda: self.changes.append(True), lambda: None, lambda: None)
        page.load(dict(root=str(self.root), areas=[['北区', 'boundary.shp']], periods=[], truths=[], area_irmad_references={}))
        self.addCleanup(page.close)
        return page

    def test_date_formats_calendar_validity_and_ambiguity(self):
        for text in ('A_20250118_01', 'A_2025-01-18', 'A_2025_01_18'):
            self.assertEqual(dates(text), {'20250118'})
        self.assertEqual(dates('20250230_20251301_1202501189'), set())
        self.assertEqual(dates('2024-02-29'), {'20240229'})
        rows, _ = import_periods([self.file('20250118_to_20260203.tif')])
        self.assertEqual(rows[0]['name'], '')
        self.assertIn('多个日期', rows[0]['reason'])

    def test_multiple_txts_each_create_one_period_with_filename_priority(self):
        a = self.file('first/A_20250118.tif')
        b = self.file('second/B_20260203.tif')
        x = self.file('wrong_19990101.txt', str(a))
        y = self.file('second.txt', str(b))
        rows, notices = import_periods([x, y])
        self.assertEqual([r['name'] for r in rows], ['20250118', '20260203'])
        self.assertEqual([r['count'] for r in rows], [1, 1])
        self.assertEqual({r['source'] for r in rows}, {str(x), str(y)})
        self.assertFalse(notices)

    def test_txt_filename_then_directory_fallback_and_relative_paths(self):
        self.file('20250118/no-date.tif')
        txt = self.file('20250118/list_2026-02-03.txt', '# comment\n"no-date.tif"\n')
        rows, _ = import_periods([txt])
        self.assertEqual(rows[0]['name'], '20260203')
        txt = self.file('20250118/list.txt', 'no-date.tif')
        rows, _ = import_periods([txt])
        self.assertEqual(rows[0]['name'], '20250118')

    def test_same_date_rasters_group_together_and_multiple_dates_split(self):
        files = [self.file(name) for name in ('A_20250118_01.tif', 'B_2025-01-18_02.tiff', 'C_20260203.img')]
        rows, _ = import_periods(files + [files[0]])
        self.assertEqual([(r['name'], r['count']) for r in rows], [('20250118', 2), ('20260203', 1)])

    def test_folder_only_scans_selected_and_first_level_and_avoids_txt_duplicates(self):
        a = self.file('data/A_20250118.tif')
        self.file('data/list.txt', a.name)
        self.file('data/20260203/no-date.jp2')
        self.file('data/20260203/second.vrt')
        self.file('data/deep/ignored/A_20270304.tif')
        self.file('data/ignored.csv')
        selected = folder_files(self.root / 'data')
        self.assertEqual(len(selected), 4)
        rows, notices = import_periods(selected)
        self.assertEqual([(r['name'], r['count']) for r in rows], [('20250118', 1), ('20260203', 2)])
        self.assertIn('未重复添加', notices[0])

    def test_txt_with_multiple_dates_requires_one_manual_name_not_silent_split(self):
        txt = self.file('fallback_20240101.txt', 'A_20250118.tif\nB_20260203.tif')
        rows, _ = import_periods([txt])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['name'], '')
        self.assertEqual(rows[0]['count'], 2)

    def test_unknown_batch_can_be_corrected_once_and_committed_atomically(self):
        page = self.panel()
        files = [self.file(f'unknown/tile_{i}.tif') for i in range(5)]
        page.begin_import(files)
        panel = page.import_confirmation
        self.assertEqual(panel.table.rowCount(), 1)
        self.assertEqual(panel.table.item(0, 1).text(), '5 幅影像')
        self.assertFalse(panel.apply_button.isEnabled())
        self.assertFalse(page.inputs()['periods'])
        self.assertTrue(page.has_pending_edit)
        panel.table.item(0, 0).setText('春季影像')
        self.assertTrue(panel.apply_button.isEnabled())
        panel.apply_button.click()
        self.assertEqual(page.inputs()['periods'][0][:2], ['北区', '春季影像'])
        self.assertEqual(len(self.changes), 1)
        self.assertFalse(page.has_pending_edit)

    def test_duplicate_names_require_correction_without_partial_import(self):
        page = self.panel()
        page.set_period(None, '20250118', 'old.txt')
        first = self.file('one.txt', 'A_20250118.tif')
        second = self.file('two.txt', 'B_20250118.tif')
        page.begin_import([first, second])
        panel = page.import_confirmation
        self.assertFalse(panel.apply_button.isEnabled())
        self.assertIn('已有期次重复', panel.message.text())
        panel.table.item(0, 0).setText('新的期次')
        panel.table.item(1, 0).setText('新的期次')
        self.assertIn('本批次重复', panel.message.text())
        page._confirm_import()
        self.assertEqual(len(page.inputs()['periods']), 1)
        panel.table.item(1, 0).setText('另一期次')
        panel.apply_button.click()
        self.assertEqual(len(page.inputs()['periods']), 3)

    def test_typing_missing_name_enables_confirmation_without_extra_enter(self):
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QLineEdit
        page = self.panel()
        page.begin_import([self.file('no-date.tif')])
        page.show()
        APP.processEvents()
        panel = page.import_confirmation
        panel.table.editItem(panel.table.item(0, 0))
        editor = panel.table.findChild(QLineEdit)
        self.assertIsNotNone(editor)
        QTest.keyClicks(editor, '20280205')
        self.assertTrue(panel.apply_button.isEnabled())
        panel.apply_button.click()
        self.assertEqual(page.inputs()['periods'][0][1], '20280205')
        page._confirm_import()
        self.assertEqual(len(page.inputs()['periods']), 1)

    def test_cancel_and_remove_unreadable_list_leave_existing_data_intact(self):
        page = self.panel()
        good = self.file('20250118.tif')
        empty = self.file('empty.txt')
        before = page.inputs()
        page.begin_import([good, empty])
        panel = page.import_confirmation
        self.assertFalse(panel.apply_button.isEnabled())
        self.assertIn('影像清单为空', panel.message.text())
        bad = next(i for i, r in enumerate(panel.drafts) if r['error'])
        panel.table.setCurrentCell(bad, 0)
        panel.remove_button.click()
        self.assertTrue(panel.apply_button.isEnabled())
        panel.cancel_button.click()
        self.assertEqual(page.inputs(), before)
        self.assertFalse(self.changes)
        self.assertTrue(page.save_button.isEnabled())

    def test_multiple_txt_file_dialog_and_folder_entry_use_same_confirmation(self):
        page = self.panel()
        txts = [str(self.file(f'{p}.txt', f'A_{p}.tif')) for p in ('20250118', '20260203')]
        with patch('plugin.ui.data_configuration.QFileDialog.getOpenFileNames', return_value=(txts, '')) as choose:
            page.add_period_button.click()
        self.assertEqual(page.import_confirmation.table.rowCount(), 2)
        self.assertIn('*.txt', choose.call_args.args[-1])
        page._cancel_import()
        with patch('plugin.ui.data_configuration.QFileDialog.getExistingDirectory', return_value=str(self.root)):
            page.add_folder_button.click()
        self.assertEqual(page.import_confirmation.table.rowCount(), 2)

    def test_single_period_rename_and_delete_preserve_reference_and_valid_truths(self):
        page = self.panel()
        for name in ('20250118', '20260203', '20270304'):
            page.set_period(None, name, 'source.txt')
        page.reference.setCurrentText('20260203')
        page.set_truth('20250118', '20260203', 'truth.shp')
        page.edit_period('20260203')
        page.period_name.setText('20260204')
        with patch('plugin.ui.data_configuration.QFileDialog.getOpenFileNames', return_value=([str(self.file('20280101.tif'))], '')):
            page._choose_images()
        self.assertEqual(page.period_name.text(), '20260204')
        page.apply_period_button.click()
        self.assertEqual(page.inputs()['area_irmad_references']['北区'], '20260204')
        self.assertEqual(page.inputs()['truths'][0][1:3], ['20250118', '20260204'])
        self.assertFalse(page.set_period('20260204', '20250118', 'different.txt'))
        page.remove_period('20260204')
        self.assertEqual(page.inputs()['area_irmad_references']['北区'], '')
        self.assertEqual(page.inputs()['truths'], [])
        self.assertEqual(page.truths.item(0, 0).text(), '20250118 → 20270304')

    def test_batch_controls_fit_300px_and_wheel_does_not_scroll_nested_table(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        page = self.panel()
        page.begin_import([self.file(f'{year}0101.tif') for year in range(2010, 2030)])
        page.resize(300, 600)
        page.show()
        APP.processEvents()
        APP.processEvents()
        self.assertEqual(page.scroll.horizontalScrollBar().maximum(), 0)
        self.assertLessEqual(page.minimumSizeHint().width(), 300)
        table = page.import_confirmation.table
        before = table.verticalScrollBar().value()
        event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        APP.sendEvent(table.viewport(), event)
        self.assertEqual(table.verticalScrollBar().value(), before)


if __name__ == '__main__':
    unittest.main()
