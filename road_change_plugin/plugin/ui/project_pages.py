"""Small project creation and update pages. No processing code."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QToolButton, QLabel, QScrollArea,
    QPushButton, QLineEdit, QComboBox, QButtonGroup, QRadioButton)
from .forms import PathField, form_layout
from .presentation import inline, primary_button
from .project_browser import natural
from ..project_state import ProjectState


def note(text=''):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    return label


class ProjectPage(QWidget):
    def __init__(self, title, back, action, callback):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.back_button = QToolButton()
        self.back_button.setText('返回')
        self.back_button.setProperty('role', 'toolbarAction')
        self.back_button.clicked.connect(back)
        layout.addWidget(inline(note(title), self.back_button))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.body = QWidget()
        self.form = form_layout(self.body)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.scroll.setWidget(self.body)
        layout.addWidget(self.scroll, 1)
        self.message = note()
        self.message.hide()
        self.form.addRow(self.message)
        self.save_button = QPushButton(action)
        self.save_button.clicked.connect(callback)
        primary_button(self.save_button)
        layout.addWidget(inline(note(), self.save_button))

    def show_issues(self, issues):
        self.message.setText('\n'.join(issues))
        self.message.setVisible(bool(issues))


class ProjectCreation(ProjectPage):
    def __init__(self, back, create):
        super().__init__('新建项目', back, '创建项目', create)
        self.project_name = QLineEdit()
        self.project_location = PathField(directory=True)
        self.form.addRow('项目名称', self.project_name)
        self.form.addRow('保存位置', self.project_location)


class UpdatePage(ProjectPage):
    def __init__(self, back, start):
        super().__init__('更新部分成果', back, '开始更新', start)
        self.store = None
        self.selection = None
        self.scope = None
        self.area = QComboBox()
        self.area.setMinimumWidth(0)
        self.area.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.form.addRow('区域', self.area)
        self.configuration_note = note('更新针对上次已处理的数据；开始后受影响的旧成果将失效。')
        self.form.addRow(self.configuration_note)
        self.options = QWidget()
        self.option_layout = QVBoxLayout(self.options)
        self.option_layout.setContentsMargins(0, 0, 0, 0)
        self.form.addRow(self.options)
        self.impact = note('请选择需要更新的期次或变化对')
        self.form.addRow(self.impact)
        self.buttons = QButtonGroup(self)
        self.buttons.buttonClicked.connect(self._selected)
        self.area.currentIndexChanged.connect(self._area_changed)
        self.save_button.setEnabled(False)

    def load(self, store):
        self.store = store
        self.configuration_note.setText('数据配置已修改。本次局部更新仍使用上次已处理的数据；要应用新配置，请完整重算。'
            if store.configuration_status() == 'changed' else '更新针对上次已处理的数据；开始后受影响的旧成果将失效。')
        descriptor = store.descriptor()
        self.manifest = descriptor['data'] if descriptor else {}
        areas = set(self.manifest.get('period_orders', {}))
        areas.update(self.manifest.get('input_spec', {}).get('grids', {}))
        areas.update(str(e.get('grid', '')) for e in self.manifest.get('period_results', []))
        previous = self.area.currentText()
        self.area.blockSignals(True)
        self.area.clear()
        self.area.addItems(sorted(areas - {''}, key=natural))
        if previous in areas:
            self.area.setCurrentText(previous)
        self.area.blockSignals(False)
        self._area_changed()

    def _area_changed(self):
        for button in self.buttons.buttons():
            self.buttons.removeButton(button)
        while self.option_layout.count():
            self.option_layout.takeAt(0).widget().deleteLater()
        self.selection, self.scope = None, None
        self.save_button.setEnabled(False)
        self.impact.setText('请选择需要更新的期次或变化对')
        grid = self.area.currentText()
        names = ProjectState.period_order(getattr(self, 'manifest', {}), grid)
        for heading, values in (
            ('更新期次', [(p, dict(action='rerun-period', grid=grid, period=p)) for p in names]),
            ('更新变化对', [(f'{a} → {b}', dict(action='rerun-change', grid=grid, before_period=a, after_period=b))
                           for a, b in zip(names, names[1:])])):
            self.option_layout.addWidget(note(heading))
            for text, selection in values:
                button = QRadioButton(text)
                button.selection = selection
                self.buttons.addButton(button)
                self.option_layout.addWidget(button)

    def _selected(self, button):
        self.selection = dict(button.selection)
        try:
            self.scope = self.store.local_scope(self.selection['action'], self.selection)
            selected = self.selection.get('period') or f"{self.selection['before_period']} → {self.selection['after_period']}"
            changes = [pair.replace('_to_', ' → ') for pair in self.scope['changes']]
            if self.selection['action'] == 'rerun-change':
                changes = []  # already listed under the directly selected target
            downstream = changes + ['长时序成果']
            if self.manifest.get('input_spec', {}).get('truths') or any(p['result_type'] == 'road_evaluation' for p in self.store.results()):
                downstream.append('精度评价')
            self.impact.setText(f"重新处理：\n{self.scope['grid']} / {selected}\n\n系统将自动更新：\n" + '\n'.join(downstream))
            self.save_button.setEnabled(True)
        except (OSError, ValueError, KeyError) as exc:
            self.scope = None
            self.impact.setText(str(exc))
            self.save_button.setEnabled(False)
