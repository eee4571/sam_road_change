from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QLineEdit,
    QToolButton, QFormLayout, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from .presentation import AccordionButton


def form_layout(widget):
    form = QFormLayout(widget)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setSpacing(8)
    return form


class PathField(QWidget):
    def __init__(self, directory=False, parent=None):
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setMinimumWidth(0)
        button = QToolButton()
        button.setText("…")
        button.setToolTip("选择目录" if directory else "选择文件")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)
        button.clicked.connect(lambda: self.browse(directory))

    def browse(self, directory):
        value = QFileDialog.getExistingDirectory(self, "选择目录") if directory else QFileDialog.getOpenFileName(self, "选择文件")[0]
        if value:
            self.edit.setText(value)

    def text(self):
        return self.edit.text().strip()


class Rows(QWidget):
    def __init__(self, headers, file_filter, parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumWidth(0)
        self.table.setMaximumHeight(160)
        layout.addWidget(self.table)
        bar = QHBoxLayout()
        for title, callback in (("添加", self.add), ("删除", self.remove), ("选择文件", self.browse)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            bar.addWidget(button)
        layout.addLayout(bar)

    def add(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column in range(self.table.columnCount()):
            self.table.setItem(row, column, QTableWidgetItem(""))
        self.table.setCurrentCell(row, 0)

    def remove(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def browse(self):
        if self.table.currentRow() < 0:
            self.add()
        path = QFileDialog.getOpenFileName(self, "选择数据", "", self.file_filter)[0]
        if path:
            self.table.setItem(self.table.currentRow(), self.table.columnCount() - 1, QTableWidgetItem(path))

    def values(self):
        return [[self.table.item(r, c).text().strip() if self.table.item(r, c) else ""
                 for c in range(self.table.columnCount())] for r in range(self.table.rowCount())]

    def set_values(self, values):
        self.table.blockSignals(True)
        self.table.setRowCount(len(values))
        for r, row in enumerate(values):
            for c, value in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(str(value)))
        self.table.blockSignals(False)


class Fold(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.toggle = AccordionButton()
        self.toggle.setAutoRaise(True)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.body = QWidget()
        self.form = form_layout(self.body)
        self.body.hide()
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self._toggle)

    def _toggle(self, checked):
        self.body.setVisible(checked)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
