"""Business result browser; each explicit Open reports exactly one payload."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QToolButton,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QPushButton, QAbstractItemView)
from .presentation import inline
from .project_pages import note
from .project_browser import natural

RESULT_LABELS = {'road_centerline': '道路中心线', 'road_surface': '道路面', 'road_width': '道路宽度',
                 'road_change': '变化检测', 'road_temporal': '长时序', 'road_evaluation': '精度评价'}
GROUPS = {'road_centerline': '单期道路', 'road_surface': '单期道路', 'road_width': '单期道路',
          'road_change': '变化检测', 'road_temporal': '长时序', 'road_evaluation': '精度评价'}
DETAIL_LABELS = {'changes': '全部变化', 'added': '新增道路', 'removed': '消失道路',
                 'widened': '道路拓宽', 'narrowed': '道路收窄', 'gpkg': '变化数据集',
                 'life_shp': '道路生命周期', 'observations_shp': '逐期观测',
                 'events_shp': '变化事件', 'event_parts_shp': '事件分段', 'lineage_shp': '道路沿革',
                 'csv': '评价表格', 'json': '评价报告'}


class ResultsPage(QWidget):
    def __init__(self, back, open_result):
        super().__init__()
        self.open_result = open_result
        self.payloads = []
        self.open_buttons = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        back_button = QToolButton()
        self.back_button = back_button
        back_button.setText('返回')
        back_button.setProperty('role', 'toolbarAction')
        back_button.clicked.connect(back)
        layout.addWidget(inline(note('当前成果'), back_button))
        self.area = QComboBox()
        self.area.setMinimumWidth(0)
        self.area.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.area.currentIndexChanged.connect(self._render)
        area_row = inline(note('区域'), self.area)
        area_row.layout().setStretch(0, 0)
        area_row.layout().setStretch(1, 1)
        layout.addWidget(area_row)
        self.message = note()
        self.message.hide()
        layout.addWidget(self.message)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(12)
        self.tree.setMinimumWidth(0)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(1, 64)
        layout.addWidget(self.tree, 1)

    def populate(self, payloads, group=None):
        previous, position = self.area.currentText(), self.tree.verticalScrollBar().value()
        self.payloads = list(payloads)
        areas = {str(p.get('metadata', {}).get('grid')) for p in self.payloads if p.get('metadata', {}).get('grid')}
        self.area.blockSignals(True)
        self.area.clear()
        self.area.addItems(sorted(areas, key=natural) or ['全项目'])
        if previous in areas:
            self.area.setCurrentText(previous)
        self.area.blockSignals(False)
        self._render()
        self.tree.verticalScrollBar().setValue(position)
        if group in self.groups:
            self.tree.scrollToItem(self.groups[group])

    def _render(self):
        self.tree.clear()
        self.open_buttons = []
        self.groups = {}
        nodes = {}
        selected = self.area.currentText()
        order = list(dict.fromkeys(GROUPS.values()))
        for payload in sorted(self.payloads, key=lambda p: (order.index(GROUPS[p['result_type']]),
                              natural(str(p.get('metadata', {}).get('period') or p.get('metadata', {}).get('before_period') or '')))):
            meta = payload.get('metadata', {})
            if meta.get('grid') and str(meta['grid']) != selected:
                continue
            group = GROUPS[payload['result_type']]
            if group not in self.groups:
                self.groups[group] = QTreeWidgetItem(self.tree, [group])
            parent = self.groups[group]
            scope = str(meta.get('period') or '')
            if meta.get('before_period') and meta.get('after_period'):
                scope = f"{meta['before_period']} → {meta['after_period']}"
            if scope:
                key = (group, scope)
                if key not in nodes:
                    nodes[key] = QTreeWidgetItem(parent, [scope])
                parent = nodes[key]
            leaf = str(payload.get('name', '')).split(' / ')[-1]
            caption = DETAIL_LABELS.get(leaf, RESULT_LABELS[payload['result_type']])
            item = QTreeWidgetItem(parent, [caption])
            item.setToolTip(0, payload['path'])
            button = QPushButton('打开')
            button.setProperty('role', 'resultAction')
            button.setMinimumHeight(30)
            button.payload = dict(payload)
            button.clicked.connect(lambda _checked=False, value=button.payload: self.open_result(value))
            self.tree.setItemWidget(item, 1, button)
            self.open_buttons.append(button)
        self.tree.expandAll()
        self.message.setText('当前区域暂无成果' if not self.open_buttons else '')
        self.message.setVisible(not bool(self.open_buttons))
