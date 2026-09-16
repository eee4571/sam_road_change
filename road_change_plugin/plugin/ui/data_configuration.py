"""Area-scoped input editor; retains the project's existing plain data contract."""
import copy
from pathlib import Path
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QAbstractItemView, QToolButton, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QInputDialog, QMessageBox)
from .forms import PathField, form_layout
from .presentation import inline
from .project_pages import ProjectPage, note
from .project_browser import natural, pairs, resolve
from .image_import import FILE_FILTER, folder_files, import_periods
from .import_confirmation import ImportConfirmation, ImportActions


def problem_for(message, prefix):
    return message.startswith(prefix + '：') or (message.startswith(prefix + ' ') and not message.startswith(prefix + ' / '))


def image_count(source, root):
    values = [v.strip() for v in source.splitlines() if v.strip()]
    if not values:
        return '未选择影像'
    if len(values) == 1 and Path(values[0]).suffix.lower() == '.txt':
        path = resolve(values[0], root)
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                return '影像清单过大'
            raw = path.read_bytes()
            try:
                text = raw.decode('utf-8-sig')
            except UnicodeDecodeError:
                text = raw.decode('gb18030')
            values = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        except (OSError, UnicodeError):
            return '影像清单不可读取'
    return f'{len(values)} 幅影像'


class DataConfiguration(ProjectPage):
    def __init__(self, changed, back, save, imported=None):
        super().__init__('数据配置', back, '保存并检查', save)
        self.changed = changed
        self.imported = imported
        self._loading = False
        self._model = dict(areas=[], periods=[], truths=[], area_irmad_references={})
        self._issues = []
        self._editing = None
        self.issues = QListWidget()
        self.issues.setWordWrap(True)
        self.issues.setMaximumHeight(96)
        self.issues.itemClicked.connect(lambda item: self.locate(item.text()))
        self.form.addRow(self.issues)
        self.area = QComboBox()
        self.area.setMinimumWidth(0)
        self.area.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.area.currentIndexChanged.connect(self._show_area)
        self.form.addRow('当前区域', self.area)
        self.add_area_button = QPushButton('添加区域')
        self.rename_area_button = QPushButton('重命名')
        self.delete_area_button = QPushButton('删除区域')
        self.add_area_button.clicked.connect(self._ask_add_area)
        self.rename_area_button.clicked.connect(self._ask_rename_area)
        self.delete_area_button.clicked.connect(self._ask_delete_area)
        self.form.addRow(inline(self.add_area_button, self.rename_area_button, self.delete_area_button, stretch_first=False))
        self.area_body = QWidget()
        form = form_layout(self.area_body)
        form.setContentsMargins(0, 0, 0, 0)
        self.boundary = PathField(file_filter='Shapefile (*.shp)')
        self.boundary.edit.textChanged.connect(self._boundary_changed)
        form.addRow('验证范围', self.boundary)
        self.reference = QComboBox()
        self.reference.setMinimumWidth(0)
        self.reference.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.reference.currentIndexChanged.connect(self._reference_changed)
        form.addRow('IR-MAD 参考期', self.reference)
        self.periods = self._table(['期次', '影像', ''])
        self.periods_label = QLabel('影像期次')
        form.addRow(self.periods_label, self.periods)
        self.add_period_button = QPushButton('添加文件')
        self.add_period_button.setToolTip('添加影像期次：批量选择 TXT 或影像')
        self.add_period_button.setAccessibleName('添加影像期次：添加文件')
        self.add_folder_button = QPushButton('添加文件夹')
        self.delete_period_button = QPushButton('删除所选期次')
        self.add_period_button.clicked.connect(self._import_files)
        self.add_folder_button.clicked.connect(self._import_folder)
        self.delete_period_button.clicked.connect(self._ask_delete_period)
        form.addRow(ImportActions(self.add_period_button, self.add_folder_button, self.delete_period_button))
        self.period_editor = QWidget()
        editor = form_layout(self.period_editor)
        editor.setContentsMargins(0, 8, 0, 8)
        self.period_name = QLineEdit()
        self.period_name.textEdited.connect(lambda: self.changed())
        self.source_summary = note()
        editor.addRow('期次名称', self.period_name)
        editor.addRow('影像', self.source_summary)
        txt = QPushButton('选择 TXT')
        images = QPushButton('选择多幅影像')
        txt.clicked.connect(self._choose_txt)
        images.clicked.connect(self._choose_images)
        editor.addRow(inline(txt, images, stretch_first=False))
        self.apply_period_button = QPushButton('应用')
        cancel = QPushButton('取消')
        self.apply_period_button.clicked.connect(self._apply_period)
        cancel.clicked.connect(self._cancel_period)
        editor.addRow(inline(note(), cancel, self.apply_period_button))
        form.addRow(self.period_editor)
        self.period_editor.hide()
        self.import_confirmation = ImportConfirmation(self._confirm_import, self._cancel_import, self)
        form.addRow(self.import_confirmation)
        self.import_confirmation.hide()
        self.truths = self._table(['变化对', '真值数据', ''])
        form.addRow('变化真值（可选）', self.truths)
        self.form.addRow(self.area_body)
        self.issues.viewport().installEventFilter(self)
        self.area.installEventFilter(self)
        self.reference.installEventFilter(self)
        self.load(self._model)

    def _table(self, headers):
        table = QTableWidget(0, len(headers))
        table.setMinimumWidth(0)
        table.setMaximumHeight(180)
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setMinimumSectionSize(35)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(2, 94 if headers[0] == '变化对' else 56)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.viewport().installEventFilter(self)
        return table

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            delta = event.pixelDelta().y() or event.angleDelta().y() / 120 * 48
            bar = self.scroll.verticalScrollBar()
            bar.setValue(bar.value() - int(delta))
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def load(self, model):
        self._model = copy.deepcopy(model)
        self._model.setdefault('area_irmad_references', {})
        self._reload_areas(self.area.currentText())

    def inputs(self):
        return {key: copy.deepcopy(self._model[key]) for key in ('areas', 'periods', 'truths', 'area_irmad_references')}

    def _reload_areas(self, selected=''):
        self.area.blockSignals(True)
        self.area.clear()
        self.area.addItems([name for name, _ in self._model['areas']])
        if self.area.findText(selected) >= 0:
            self.area.setCurrentText(selected)
        self.area.blockSignals(False)
        self._show_area()

    def _show_area(self):
        self._loading = True
        area = self.area.currentText()
        self.area_body.setVisible(bool(self.area.count()))
        self.rename_area_button.setEnabled(bool(self.area.count()))
        self.delete_area_button.setEnabled(bool(self.area.count()))
        self.boundary.edit.setText(next((p for a, p in self._model['areas'] if a == area), ''))
        self.reference.clear()
        self.reference.addItem('请选择参考期', '')
        rows = sorted([r for r in self._model['periods'] if r[0] == area], key=lambda r: natural(r[1]))
        for row in rows:
            self.reference.addItem(row[1], row[1])
        self.reference.setCurrentIndex(max(0, self.reference.findData(self._model['area_irmad_references'].get(area, ''))))
        self.periods.setRowCount(len(rows))
        for i, (_, period, source) in enumerate(rows):
            self.periods.setItem(i, 0, QTableWidgetItem(period))
            self.periods.setItem(i, 1, QTableWidgetItem(image_count(source, Path(self._model.get('root', '.')))))
            button = QPushButton('编辑')
            button.setProperty('role', 'resultAction')
            button.clicked.connect(lambda _checked=False, value=period: self.edit_period(value))
            self.periods.setCellWidget(i, 2, button)
        self._pairs = [(b, a) for g, b, a in pairs(self._model['periods']) if g == area]
        self.truths.setRowCount(len(self._pairs))
        for i, (before, after) in enumerate(self._pairs):
            source = next((p for g, b, a, p in self._model['truths'] if (g, b, a) == (area, before, after)), '')
            self.truths.setItem(i, 0, QTableWidgetItem(f'{before} → {after}'))
            self.truths.setItem(i, 1, QTableWidgetItem('已提供' if source else '未提供'))
            choose, clear = QToolButton(), QToolButton()
            choose.setText('选择')
            clear.setText('×')
            clear.setToolTip('清除真值配置')
            clear.setEnabled(bool(source))
            choose.clicked.connect(lambda _checked=False, b=before, a=after: self._choose_truth(b, a))
            clear.clicked.connect(lambda _checked=False, b=before, a=after: self.set_truth(b, a, ''))
            self.truths.setCellWidget(i, 2, inline(choose, clear, stretch_first=False))
        for table in (self.periods, self.truths):
            for row in range(table.rowCount()):
                table.setRowHeight(row, 32)
            table.setMinimumHeight(min(180, table.horizontalHeader().height() + max(1, min(4, table.rowCount())) * 32 + 4))
        self._editing = None
        self._cancel_import()
        self._cancel_period()
        self._loading = False
        self.show_issues(self._issues)

    def _boundary_changed(self):
        if self._loading:
            return
        for row in self._model['areas']:
            if row[0] == self.area.currentText():
                row[-1] = self.boundary.text()
                break
        self.changed()

    def _reference_changed(self):
        if not self._loading:
            self._model['area_irmad_references'][self.area.currentText()] = self.reference.currentData() or ''
            self.changed()

    def add_area(self, name, boundary=''):
        name = name.strip()
        if not name or name in [a for a, _ in self._model['areas']]:
            self.show_issues(['区域名称不能为空或重复'])
            return False
        self._model['areas'].append([name, boundary])
        self._reload_areas(name)
        self.changed()
        return True

    def _ask_add_area(self):
        name, ok = QInputDialog.getText(self, '添加区域', '区域名称')
        if ok:
            self.add_area(name)

    def _ask_rename_area(self):
        old = self.area.currentText()
        name, ok = QInputDialog.getText(self, '重命名区域', '区域名称', text=old)
        if not ok or name.strip() == old:
            return
        self.rename_area(name)

    def rename_area(self, name):
        old, name = self.area.currentText(), name.strip()
        if not name or name in [a for a, _ in self._model['areas']]:
            self.show_issues(['区域名称不能为空或重复'])
            return False
        for key in ('areas', 'periods', 'truths'):
            for row in self._model[key]:
                if row[0] == old:
                    row[0] = name
        refs = self._model['area_irmad_references']
        refs[name] = refs.pop(old, '')
        self._reload_areas(name)
        self.changed()
        return True

    def _ask_delete_area(self):
        area = self.area.currentText()
        if QMessageBox.question(self, '删除区域', f'移除 {area} 的数据配置？原始文件不会删除。',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.remove_area()

    def remove_area(self):
        area = self.area.currentText()
        for key in ('areas', 'periods', 'truths'):
            self._model[key] = [r for r in self._model[key] if r[0] != area]
        self._model['area_irmad_references'].pop(area, None)
        self._reload_areas()
        self.changed()

    def edit_period(self, period):
        self._editing = period
        self._source = next((s for a, p, s in self._model['periods'] if (a, p) == (self.area.currentText(), period)), '')
        self.period_name.setText(period or '')
        self._source_caption()
        self.period_editor.show()
        for field in self._other_inputs():
            field.setEnabled(False)
        self.period_name.setFocus()
        self.scroll.ensureWidgetVisible(self.period_editor)
        self.save_button.setEnabled(False)

    @property
    def has_pending_edit(self):
        return not self.period_editor.isHidden() or not self.import_confirmation.isHidden()

    def _import_files(self):
        paths = QFileDialog.getOpenFileNames(self, '添加影像期次', '', FILE_FILTER)[0]
        if paths:
            self.begin_import(paths)

    def _import_folder(self):
        directory = QFileDialog.getExistingDirectory(self, '添加文件夹（含一级子目录）')
        if directory:
            try:
                self.begin_import(folder_files(directory))
            except OSError as exc:
                self.show_issues([f'影像目录无法读取：{exc}'])
            except ValueError as exc:
                self.show_issues([str(exc)])

    def begin_import(self, paths):
        if not self.area.currentText():
            return
        drafts, notices = import_periods(paths)
        if not drafts:
            self.show_issues(notices or ['未找到支持的影像或 TXT 清单（仅扫描所选目录及一级子目录）'])
            return
        self._cancel_period()
        area = self.area.currentText()
        self.import_confirmation.heading.setText(f'添加影像期次 · {area}')
        self.import_confirmation.load(drafts, [p for a, p, _ in self._model['periods'] if a == area], notices)
        self.import_confirmation.show()
        self.periods.hide()
        self.periods_label.hide()
        for field in self._other_inputs():
            field.setEnabled(False)
        self.save_button.setEnabled(False)
        self.scroll.ensureWidgetVisible(self.import_confirmation)

    def _cancel_import(self):
        self.import_confirmation.hide()
        self.periods.show()
        self.periods_label.show()
        self.save_button.setEnabled(True)
        for field in self._other_inputs():
            field.setEnabled(True)
        self.rename_area_button.setEnabled(bool(self.area.count()))
        self.delete_area_button.setEnabled(bool(self.area.count()))

    def _confirm_import(self):
        panel = self.import_confirmation
        if panel.isHidden() or not panel.validate():
            return
        area = self.area.currentText()
        rows = [[area, row['name'], row['source']] for row in panel.values()]
        self._model['periods'].extend(rows)
        self._prune_truths()
        self._show_area()
        self.changed()
        if self.imported:
            self.imported()

    def _cancel_period(self):
        self.period_editor.hide()
        self.save_button.setEnabled(True)
        for field in self._other_inputs():
            field.setEnabled(True)
        self.rename_area_button.setEnabled(bool(self.area.count()))
        self.delete_area_button.setEnabled(bool(self.area.count()))

    def _other_inputs(self):
        return (self.area, self.add_area_button, self.rename_area_button, self.delete_area_button,
                self.boundary, self.reference, self.periods, self.truths, self.add_period_button,
                self.add_folder_button, self.delete_period_button)

    def _source_caption(self):
        self.source_summary.setText(image_count(self._source, Path(self._model.get('root', '.'))))

    def _choose_txt(self):
        value = QFileDialog.getOpenFileName(self, '选择影像 TXT', '', '影像清单 (*.txt)')[0]
        if value:
            self._source = value
            self._source_caption()
            self.changed()

    def _choose_images(self):
        values = QFileDialog.getOpenFileNames(self, '选择多幅影像', '', '影像 (*.tif *.tiff *.img *.jp2 *.vrt)')[0]
        if values:
            self._source = '\n'.join(values)
            self._source_caption()
            self.changed()

    def _apply_period(self):
        self.set_period(self._editing, self.period_name.text(), self._source)

    def set_period(self, old, name, source):
        area, name = self.area.currentText(), name.strip()
        if not name or any(a == area and p == name and p != old for a, p, _ in self._model['periods']):
            self.show_issues([f'{area}：期次名称不能为空或重复'])
            return False
        self._model['periods'] = [r for r in self._model['periods'] if (r[0], r[1]) != (area, old)]
        self._model['periods'].append([area, name, source])
        if self._model['area_irmad_references'].get(area) == old:
            self._model['area_irmad_references'][area] = name
        if old and old != name:
            for row in self._model['truths']:
                if row[0] == area:
                    row[1:3] = [name if value == old else value for value in row[1:3]]
        self._prune_truths()
        self._show_area()
        self.save_button.setEnabled(True)
        self.changed()
        return True

    def _ask_delete_period(self):
        row = self.periods.currentRow()
        if row < 0:
            return
        period = self.periods.item(row, 0).text()
        if QMessageBox.question(self, '删除期次', f'移除 {period} 及不再对应变化对的真值配置？原始文件不会删除。',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.remove_period(period)

    def remove_period(self, period):
        area = self.area.currentText()
        self._model['periods'] = [r for r in self._model['periods'] if (r[0], r[1]) != (area, period)]
        if self._model['area_irmad_references'].get(area) == period:
            self._model['area_irmad_references'][area] = ''
        self._prune_truths()
        self._show_area()
        self.changed()

    def _prune_truths(self):
        expected = set(pairs(self._model['periods']))
        area = self.area.currentText()
        self._model['truths'] = [r for r in self._model['truths'] if r[0] != area or tuple(r[:3]) in expected]

    def _choose_truth(self, before, after):
        value = QFileDialog.getOpenFileName(self, '选择变化真值', '', 'Shapefile (*.shp)')[0]
        if value:
            self.set_truth(before, after, value)

    def set_truth(self, before, after, source):
        area = self.area.currentText()
        self._model['truths'] = [r for r in self._model['truths'] if tuple(r[:3]) != (area, before, after)]
        if source:
            self._model['truths'].append([area, before, after, source])
        self._show_area()
        self.changed()

    def show_issues(self, issues):
        self._issues = list(issues)
        self.issues.clear()
        self.issues.addItems(issues)
        self.issues.setVisible(bool(issues))
        area = self.area.currentText()
        for field, word in ((self.reference, '参考期'), (self.boundary.edit, '验证区')):
            message = next((m for m in issues if problem_for(m, area) and word in m), '') if area else ''
            field.setToolTip(message)
            field.setProperty('inputProblem', bool(message))
            field.style().unpolish(field)
            field.style().polish(field)
        for table in (self.periods, self.truths):
            for row in range(table.rowCount()):
                scope = table.item(row, 0).text().replace(' → ', ' / ')
                prefix = area + ' / ' + scope
                message = next((m for m in issues if problem_for(m, prefix)), '')
                for col in (0, 1):
                    item = table.item(row, col)
                    item.setToolTip(message)
                    item.setBackground(QBrush(table.palette().alternateBase()) if message else QBrush())

    def locate(self, message):
        names = sorted([a for a, _ in self._model['areas']], key=len, reverse=True)
        for area in names:
            if message.startswith((area + '：', area + ' / ', area + ' ')):
                self.area.setCurrentText(area)
                if '参考期' in message:
                    self.reference.setFocus()
                    self.scroll.ensureWidgetVisible(self.reference)
                    return
                for table in (self.truths, self.periods):
                    for row in range(table.rowCount()):
                        prefix = area + ' / ' + table.item(row, 0).text().replace(' → ', ' / ')
                        if problem_for(message, prefix):
                            table.selectRow(row)
                            table.scrollToItem(table.item(row, 0))
                            self.scroll.ensureWidgetVisible(table)
                            return
                field = self.periods if '期次' in message else self.boundary.edit
                field.setFocus()
                self.scroll.ensureWidgetVisible(field)
                return
