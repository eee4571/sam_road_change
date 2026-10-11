"""Responsive desktop form rows."""
from PySide6.QtCore import Qt, QSize, QRect, QPoint
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QLayout


class ActionRow(QLayout):
    """Left-aligned buttons that wrap only when the host is too narrow."""
    def __init__(self, parent, *buttons):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(8)
        for button in buttons:
            self.addWidget(button)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), False)

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        return size

    def sizeHint(self):
        return self.minimumSize()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, True)

    def _arrange(self, rect, apply):
        x, y, height = rect.x(), rect.y(), 0
        for item in self.items:
            size = item.sizeHint()
            if x > rect.x() and x + size.width() > rect.right() + 1:
                x, y, height = rect.x(), y + height + self.spacing(), 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), size))
            x += size.width() + self.spacing()
            height = max(height, size.height())
        return y + height - rect.y()


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
