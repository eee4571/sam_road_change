"""Small layout primitives; colors and typography come from the Qt host."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QPainter, QPen
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QMenu, QToolButton, QSizePolicy


class SectionHeader(QWidget):
    def __init__(self, title, first=False):
        super().__init__()
        self.setProperty("role", "sectionHeading")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0 if first else 10, 0, 0)
        row.setSpacing(12)
        heading = QLabel(title)
        heading.setProperty("role", "sectionHeading")
        row.addWidget(heading)
        row.addStretch(1)


def inline(*widgets, stretch_first=True):
    body = QWidget()
    row = QHBoxLayout(body)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    for index, widget in enumerate(widgets):
        row.addWidget(widget, 1 if index == 0 and stretch_first else 0)
    return body


def primary_button(button):
    """Style just the primary action with live host palette roles, never the app."""
    button.setObjectName("roadChangeRun")
    button.setProperty("role", "primary")
    button.setDefault(True)


class AccordionButton(QToolButton):
    """Native checkable control with a full-row hit target and trailing chevron."""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setAutoRaise(True)
        self.setProperty("role", "accordion")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        foreground = self.palette().color(QPalette.ColorRole.Text)
        subtle = foreground
        subtle.setAlpha(22)
        if self.underMouse():
            painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.AlternateBase))
        painter.setPen(QPen(subtle, 1))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        painter.drawText(self.rect().adjusted(0, 0, -26, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.PlaceholderText), 1.4))
        x, y = self.width() - 12, self.height() // 2
        if self.isChecked():
            painter.drawLine(x - 4, y - 2, x, y + 2)
            painter.drawLine(x, y + 2, x + 4, y - 2)
        else:
            painter.drawLine(x - 2, y - 4, x + 2, y)
            painter.drawLine(x + 2, y, x - 2, y + 4)
        if self.hasFocus():
            painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), 1, Qt.PenStyle.DotLine))
            painter.drawRect(self.rect().adjusted(1, 1, -2, -2))


def result_menu(button, title, payloads, open_result):
    """Keep individual-file access available without a permanent results tree."""
    menu = QMenu(button)
    for caption, payload in payloads:
        action = menu.addAction(caption)
        action.setToolTip(payload["path"])
        action.triggered.connect(lambda _checked=False, value=payload: open_result(value))
    button.setToolTip(f"打开全部{title}成果；右键可选择单项")
    return menu
