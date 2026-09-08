"""Responsive desktop rows and explicit single-file result selection."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QComboBox, QPushButton, QFormLayout


class ResponsiveRow(QWidget):
    def __init__(self, title, field, action=None):
        super().__init__()
        self.caption = QLabel(title)
        self.field, self.action = field, action
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self.narrow = None
        self._arrange(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange(self.width() < 480)

    def _arrange(self, narrow):
        if narrow == self.narrow:
            return
        self.narrow = narrow
        for widget in (self.caption, self.field, self.action):
            if widget:
                self.grid.removeWidget(widget)
        for column in range(3):
            self.grid.setColumnStretch(column, 0)
            self.grid.setColumnMinimumWidth(column, 0)
        self.caption.setVisible(bool(self.caption.text()))
        if narrow:
            offset = int(bool(self.caption.text()))
            self.grid.addWidget(self.caption, 0, 0, 1, 2)
            self.grid.addWidget(self.field, offset, 0)
            if self.action:
                self.grid.addWidget(self.action, offset + int(not self.caption.text()), 1 if self.caption.text() else 0)
                if not self.caption.text():
                    self.grid.setAlignment(self.action, Qt.AlignmentFlag.AlignRight)
            self.grid.setColumnStretch(0, 1)
        else:
            self.grid.addWidget(self.caption, 0, 0)
            self.grid.setColumnMinimumWidth(0, 80)
            self.grid.addWidget(self.field, 0, 1)
            if self.action:
                self.grid.addWidget(self.action, 0, 2)
            self.grid.setColumnStretch(1, 1)


class ResultChooser(QWidget):
    def __init__(self, open_result):
        super().__init__()
        self.entries = []
        form = QFormLayout(self)
        form.setContentsMargins(0, 8, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.title = QLabel()
        self.area, self.scope, self.file = QComboBox(), QComboBox(), QComboBox()
        for field in (self.area, self.scope, self.file):
            field.setMinimumWidth(0)
            field.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        form.addRow(self.title)
        form.addRow("区域", self.area)
        form.addRow("期次 / 变化对", self.scope)
        form.addRow("文件", self.file)
        self.open_button = QPushButton("打开所选成果")
        self.open_button.clicked.connect(lambda: open_result(self.file.currentData()) if self.file.currentData() else None)
        close = QPushButton("收起")
        close.clicked.connect(self.hide)
        form.addRow(close, self.open_button)
        self.area.currentIndexChanged.connect(self._scopes)
        self.scope.currentIndexChanged.connect(self._files)

    @staticmethod
    def context(payload):
        meta = payload.get("metadata", {})
        scope = str(meta.get("period") or "汇总")
        if meta.get("before_period") and meta.get("after_period"):
            scope = f"{meta['before_period']} → {meta['after_period']}"
        return str(meta.get("grid") or "全项目"), scope

    def populate(self, group, entries):
        self.entries = entries
        self.title.setText(group + " · 选择成果")
        self.area.clear()
        self.area.addItems(sorted({self.context(p)[0] for _, p in entries}))
        self._scopes()

    def _scopes(self):
        self.scope.clear()
        self.scope.addItems(sorted({self.context(p)[1] for _, p in self.entries if self.context(p)[0] == self.area.currentText()}))
        self._files()

    def _files(self):
        self.file.clear()
        for caption, payload in self.entries:
            if self.context(payload) == (self.area.currentText(), self.scope.currentText()):
                self.file.addItem(caption + " · " + payload['path'].replace('\\', '/').rsplit('/', 1)[-1], payload)
        self.open_button.setEnabled(self.file.count() > 0)
