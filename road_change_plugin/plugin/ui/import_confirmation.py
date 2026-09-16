"""Inline batch confirmation, hosted by the existing data configuration page."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QPushButton, QStyledItemDelegate, QSizePolicy)
from .project_pages import note
from .presentation import inline


class ImportActions(QWidget):
    def __init__(self, *buttons):
        super().__init__()
        self.buttons = buttons
        self.row = QGridLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(6)
        self._narrow = None
        self.arrange(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.arrange(self.width() < sum(b.sizeHint().width() for b in self.buttons) + 12)

    def arrange(self, narrow):
        if narrow == self._narrow:
            return
        self._narrow = narrow
        for button in self.buttons:
            self.row.removeWidget(button)
        for i, button in enumerate(self.buttons):
            self.row.addWidget(button, 1 if narrow and i == 2 else 0, 0 if narrow and i == 2 else i,
                               1, 2 if narrow and i == 2 else 1, Qt.AlignmentFlag.AlignLeft)
        self.row.setColumnStretch(3, 1)


class PeriodNameDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        return str(value) if value else '待填写'

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        editor.setPlaceholderText('待填写')
        # Keep validation in sync while typing, so confirmation needs no extra Enter.
        editor.textEdited.connect(lambda: self.commitData.emit(editor))
        return editor


class ImportConfirmation(QWidget):
    def __init__(self, apply, cancel, wheel_owner):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(6)
        self.heading = note('添加影像期次 · 确认识别结果')
        layout.addWidget(self.heading)
        layout.addWidget(note('可直接修改期次名称；同一区域的名称不能重复。'))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['期次', '数据'])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(50)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed | QAbstractItemView.EditTrigger.SelectedClicked)
        self.table.setItemDelegateForColumn(0, PeriodNameDelegate(self.table))
        self.table.setMinimumWidth(0)
        self.table.setMaximumHeight(220)
        self.table.viewport().installEventFilter(wheel_owner)
        layout.addWidget(self.table)
        self.detail = note()
        self.message = note()
        layout.addWidget(self.detail)
        layout.addWidget(self.message)
        self.remove_button = QPushButton('移除所选行')
        self.remove_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.remove_button.clicked.connect(self.remove_selected)
        layout.addWidget(inline(self.remove_button, note(), stretch_first=False))
        self.cancel_button = QPushButton('取消')
        self.apply_button = QPushButton('确认导入')
        self.cancel_button.clicked.connect(cancel)
        self.apply_button.clicked.connect(apply)
        layout.addWidget(inline(note(), self.cancel_button, self.apply_button))
        self.table.currentCellChanged.connect(self._detail)
        self.table.itemChanged.connect(self.validate)
        self.drafts, self.existing, self.notices = [], set(), []

    def load(self, drafts, existing, notices):
        self.drafts, self.existing, self.notices = drafts, set(existing), notices
        self.table.blockSignals(True)
        self.table.setRowCount(len(drafts))
        for i, row in enumerate(drafts):
            self.table.setItem(i, 0, QTableWidgetItem(row['name']))
            data = QTableWidgetItem(f"{row['count']} 幅影像")
            data.setFlags(data.flags() & ~Qt.ItemFlag.ItemIsEditable)
            data.setToolTip(row['origin'])
            self.table.setItem(i, 1, data)
            self.table.setRowHeight(i, 32)
        height = min(220, self.table.horizontalHeader().height() + max(1, len(drafts)) * 32 + 4)
        self.table.setMinimumHeight(height)
        self.table.setMaximumHeight(height)
        self.table.blockSignals(False)
        self.validate()
        self.table.setCurrentCell(-1, -1)
        self._detail()

    def values(self):
        return [{**row, 'name': self.table.item(i, 0).text().strip()} for i, row in enumerate(self.drafts)]

    def validate(self, *args):
        if self.table.signalsBlocked():
            return False
        values = self.values()
        names = [row['name'] for row in values]
        issues = []
        self.table.blockSignals(True)
        for i, row in enumerate(values):
            name = row['name']
            error = row['error'] or ('请填写期次名称' if not name else
                    f'期次 {name} 与已有期次重复' if name in self.existing else
                    f'期次 {name} 在本批次重复' if names.count(name) > 1 else '')
            item = self.table.item(i, 0)
            item.setToolTip(error or row['reason'])
            item.setBackground(QBrush(self.table.palette().alternateBase()) if error else QBrush())
            if error:
                issues.append(f'第 {i + 1} 行：{error}')
        self.table.blockSignals(False)
        visible = issues[:4] + ([f'另有 {len(issues) - 4} 行待修正，请检查标记行'] if len(issues) > 4 else [])
        self.message.setText('\n'.join(visible + self.notices))
        self.message.setVisible(bool(issues or self.notices))
        self.apply_button.setEnabled(bool(values) and not issues)
        return bool(values) and not issues

    def _detail(self, *args):
        i = self.table.currentRow()
        origin = self.drafts[i]['origin'] if 0 <= i < len(self.drafts) else ''
        self.detail.setText(origin[:140] + ('…' if len(origin) > 140 else ''))
        self.detail.setToolTip(origin)
        self.detail.setVisible(bool(self.detail.text()))

    def remove_selected(self):
        i = self.table.currentRow()
        if 0 <= i < len(self.drafts):
            values = self.values()
            values.pop(i)
            self.load(values, self.existing, self.notices)
