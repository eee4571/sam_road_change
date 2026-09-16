from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLineEdit,
    QToolButton, QFormLayout)
from PySide6.QtCore import Qt
from .presentation import AccordionButton


def form_layout(widget):
    form = QFormLayout(widget)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setSpacing(8)
    return form


class PathField(QWidget):
    def __init__(self, directory=False, parent=None, file_filter='所有文件 (*)'):
        super().__init__(parent)
        self.file_filter = file_filter
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
        value = QFileDialog.getExistingDirectory(self, "选择目录") if directory else QFileDialog.getOpenFileName(self, "选择文件", '', self.file_filter)[0]
        if value:
            self.edit.setText(value)

    def text(self):
        return self.edit.text().strip()


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
        self.form.setContentsMargins(0, 8, 0, 0)
        self.body.hide()
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self._toggle)

    def _toggle(self, checked):
        self.body.setVisible(checked)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
