"""Data-configuration page; file selection only, no backend dependencies."""
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QLabel, QPushButton,
    QComboBox, QListWidget, QAbstractItemView, QToolButton)
from .forms import Rows, form_layout
from .presentation import inline
from .project_browser import natural


class DataConfiguration(QWidget):
    def __init__(self, changed, back, save):
        super().__init__()
        self.changed = changed
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        back_button = QToolButton()
        back_button.setText("返回")
        back_button.setProperty("role", "toolbarAction")
        back_button.clicked.connect(back)
        layout.addWidget(inline(QLabel("数据配置"), back_button))
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.body = QWidget()
        form = form_layout(self.body)
        form.setContentsMargins(8, 0, 8, 8)
        self.scroll.setWidget(self.body)
        layout.addWidget(self.scroll, 1)
        self.issues = QListWidget()
        self.issues.setMaximumHeight(100)
        self.issues.setWordWrap(True)
        self.issues.itemClicked.connect(lambda item: self.locate(item.text()))
        form.addRow(self.issues)
        self.areas = Rows(["验证区", "边界文件"], "Shapefile (*.shp)")
        self.periods = Rows(["验证区", "期次", "影像清单"], "影像清单 (*.txt)")
        self.truths = Rows(["验证区", "前期", "后期", "真值文件"], "Shapefile (*.shp)")
        form.addRow("验证区", self.areas)
        self.references = QWidget()
        self.reference_form = form_layout(self.references)
        self.reference_form.setRowWrapPolicy(self.reference_form.RowWrapPolicy.WrapLongRows)
        self.reference_form.setContentsMargins(0, 0, 0, 0)
        self.reference_boxes = {}
        form.addRow("各验证区 IR-MAD 参考期", self.references)
        form.addRow("影像期次", self.periods)
        form.addRow("可选变化真值", self.truths)
        for rows in (self.areas, self.periods, self.truths):
            rows.table.setMinimumHeight(90)
            rows.table.setMaximumHeight(170)
            rows.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            rows.table.viewport().installEventFilter(self)
            rows.table.itemChanged.connect(self._edited)
            rows.table.model().rowsRemoved.connect(self._edited)
        self.issues.viewport().installEventFilter(self)
        self.save_button = QPushButton("保存并检查")
        self.save_button.clicked.connect(save)
        layout.addWidget(inline(QLabel(""), self.save_button))

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            # Wheel always moves the page. Table scrollbars remain draggable.
            delta = event.pixelDelta().y() or event.angleDelta().y() / 120 * 48
            bar = self.scroll.verticalScrollBar()
            bar.setValue(bar.value() - int(delta))
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def load(self, model):
        self._loading = True
        for rows, key in ((self.areas, "areas"), (self.periods, "periods"), (self.truths, "truths")):
            rows.set_values(model[key])
        self._rebuild_references(model.get("area_irmad_references", {}))
        self._loading = False
        self._size_tables()

    def _size_tables(self):
        for rows in (self.areas, self.periods, self.truths):
            height = rows.table.horizontalHeader().height() + sum(rows.table.rowHeight(i) for i in range(min(6, rows.table.rowCount()))) + 4
            rows.table.setMaximumHeight(min(180, max(90, height)))

    def _rebuild_references(self, values):
        while self.reference_form.rowCount():
            self.reference_form.removeRow(0)
        self.reference_boxes = {}
        for area, _ in self.areas.values():
            if not area or area in self.reference_boxes:
                continue
            box = QComboBox()
            box.setMinimumWidth(0)
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.addItem("请选择参考期", "")
            for period in sorted({p for a, p, _ in self.periods.values() if a == area and p}, key=natural):
                box.addItem(period, period)
            index = box.findData(values.get(area, ""))
            box.setCurrentIndex(max(0, index))
            box.currentIndexChanged.connect(lambda _index: self.changed() if not self._loading else None)
            box.installEventFilter(self)
            self.reference_boxes[area] = box
            caption = QLabel(area)
            caption.setWordWrap(True)
            caption.setMinimumWidth(0)
            self.reference_form.addRow(caption, box)

    def _edited(self, *args):
        if self._loading:
            return
        self._rebuild_references(self.reference_values())
        self._size_tables()
        self.changed()

    def reference_values(self):
        return {area: box.currentData() or "" for area, box in self.reference_boxes.items()}

    def show_issues(self, issues):
        self.issues.clear()
        self.issues.addItems(issues)
        self.issues.setMaximumHeight(min(100, max(36, len(issues) * self.fontMetrics().height() + 14)))
        self.issues.setVisible(bool(issues))
        for area, box in self.reference_boxes.items():
            box.setToolTip(next((m for m in issues if m.startswith(area + "：") and "参考期" in m), ""))
            box.setProperty("inputProblem", bool(box.toolTip()))
            box.style().unpolish(box)
            box.style().polish(box)
        for rows in (self.areas, self.periods, self.truths):
            rows.table.blockSignals(True)
            for index, values in enumerate(rows.values()):
                prefix = " / ".join(values[:-1])
                message = next((m for m in issues if m.startswith(prefix + "：") or m.startswith(prefix + " ")), "")
                for column in range(rows.table.columnCount()):
                    item = rows.table.item(index, column)
                    if item:
                        item.setToolTip(message)
                        item.setBackground(QBrush(rows.palette().alternateBase()) if message else QBrush())
            rows.table.blockSignals(False)

    def locate(self, message):
        for area, box in self.reference_boxes.items():
            if message.startswith(area + "：") and "参考期" in message:
                self.scroll.ensureWidgetVisible(box)
                box.setFocus()
                return
        for rows in (self.periods, self.truths, self.areas):
            for index, values in enumerate(rows.values()):
                prefix = " / ".join(values[:-1])
                if message.startswith(prefix + "：") or message.startswith(prefix + " "):
                    rows.table.selectRow(index)
                    self.scroll.ensureWidgetVisible(rows)
                    rows.table.scrollToItem(rows.table.item(index, 0))
                    return
