"""Responsive desktop form rows."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel


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
