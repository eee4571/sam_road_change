"""Five evaluation metrics; report context stays in the read-only dialog."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from .evaluation_summary import report_metrics


class MetricSummary(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 2, 0, 2)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(6)
        self.values, self.units = {}, {}
        self.cards, self.captions = [], []
        self.columns = None
        metrics = (
            ('recall', '变化图斑查全率'),
            ('precision', '变化图斑准确率'),
            ('completeness', '变化道路提取完整度'),
            ('offset', '中心线平均偏移距离'),
            ('type_accuracy', '变化类型正确率'),
        )
        for i, (key, title) in enumerate(metrics):
            card = QWidget()
            box = QVBoxLayout(card)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(2)
            value, unit, caption = QLabel('未评价'), QLabel(''), QLabel(title)
            value.setProperty('role', 'metricValue')
            value.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            unit.setProperty('role', 'metricUnit')
            unit.hide()
            caption.setProperty('role', 'secondary')
            caption.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            for item in (value, caption):
                item.setWordWrap(True)
                item.setMinimumWidth(0)
            number = QHBoxLayout()
            number.setSpacing(2)
            number.addWidget(value, 0, Qt.AlignmentFlag.AlignBaseline)
            number.addWidget(unit, 0, Qt.AlignmentFlag.AlignBaseline)
            number.addStretch(1)
            box.addLayout(number)
            box.addWidget(caption, 1)
            self.values[key] = value
            self.units[key] = unit
            self.cards.append(card)
            self.captions.append(caption)
        self._arrange(3)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange(self._columns_for_width(self.width()))

    def _columns_for_width(self, width):
        item_width = max(label.fontMetrics().horizontalAdvance(label.text()) for label in self.captions)
        return 5 if width >= item_width * 5 + self.grid.horizontalSpacing() * 4 else 3

    def heightForWidth(self, width):
        # Report the new row count before resizeEvent so surrounding actions
        # receive enough space in the same layout pass.
        columns = self._columns_for_width(width)
        card_width = max(1, (width - self.grid.horizontalSpacing() * (columns - 1)) // columns)
        height = max(card.layout().totalHeightForWidth(card_width) for card in self.cards)
        rows = (len(self.cards) + columns - 1) // columns
        return height * rows + (rows - 1) * self.grid.verticalSpacing() + 4

    def _arrange(self, columns):
        if self.columns == columns:
            return
        self.columns = columns
        for card in self.cards:
            self.grid.removeWidget(card)
        for column in range(5):
            self.grid.setColumnStretch(column, int(column < columns))
        for row in range(2):
            self.grid.setRowStretch(row, int(row * columns < len(self.cards)))
        for i, card in enumerate(self.cards):
            self.grid.addWidget(card, i // columns, i % columns)
        self.updateGeometry()

    def refresh(self, path=None):
        data = report_metrics(path) if path else {}
        for key, label in self.values.items():
            value = data.get(key, '—')
            number, unit = ('未评价', '') if value == '—' else (value[:-1], '%') if value.endswith('%') else value.split(' ', 1)
            label.setText(number)
            self.units[key].setText(unit)
            self.units[key].setVisible(bool(unit))
            label.setAccessibleName(value if value != '—' else '未评价')
