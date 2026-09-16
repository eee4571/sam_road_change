"""Compact processing dock; only UI input selection and presentation live here."""
import copy
import time
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QComboBox, QScrollArea, QLabel, QPushButton, QLineEdit,
    QProgressBar, QPlainTextEdit,
    QGridLayout, QToolButton, QMenu, QGroupBox, QStackedWidget, QFileDialog, QMessageBox)
from .ui.forms import PathField, Fold, form_layout
from .ui.presentation import inline, primary_button
from .ui.appearance import dock_style
from .ui.data_configuration import DataConfiguration
from .ui.desktop import ResponsiveRow
from .ui.project_pages import ProjectCreation, UpdatePage
from .ui.project_browser import ROOT, scan_project, check_files, resolve, save_configuration, create_project
from .ui.background import BrowseJob
from .ui.evaluation_summary import metrics_text
from .project_state import ProjectState
from .result_parser import formal_results

def label(text, role=None):
    value = QLabel(text)
    if role:
        value.setProperty("role", role)
    value.setWordWrap(True)
    value.setMinimumWidth(0)
    return value


class RoadChangeWidget(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.model = dict(areas=[], periods=[], truths=[], issues=[])
        self._project_directory = ''
        self.checked = self.busy = self.browsing = False
        self._started_at = None
        self._action = 'all'
        self._job = self._task_log_path = self._current = None
        self._log_lines, self._last_error = [], ''
        self._active_record = 'session'
        self._result_keys, self._jobs = set(), {}
        self._project_revision = 0
        self._return_after_check = self._restoring = self._draft_dirty = False
        self._processing_dirty = False
        self._resume_task = self._created_root = None
        self._project_operation = {}
        self._configuration_status = 'unknown'
        self.result_timer = QTimer(self)
        self.result_timer.setSingleShot(True)
        self.result_timer.timeout.connect(self._refresh_results)
        self.open_timer = QTimer(self)
        self.open_timer.setSingleShot(True)
        self.open_timer.setInterval(200)
        self.open_timer.timeout.connect(self.scan)
        self.setObjectName('roadChangeDock')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages)
        self.main_page = QWidget()
        self.pages.addWidget(self.main_page)
        layout = QVBoxLayout(self.main_page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.logs_button = self._tool_button('日志', self._show_logs)
        self.more_button = self._tool_button('…', lambda: None)
        self.more_button.setToolTip('更多操作')
        for button in (self.logs_button, self.more_button):
            button.setProperty('role', 'toolbarAction')
        menu = QMenu(self.more_button)
        self.restart_action = menu.addAction('重新运行完整流程', self._restart)
        menu.addSeparator()
        self.new_project_action = menu.addAction('新建项目', self._show_creation)
        self.rescan_action = menu.addAction('刷新并检查', self.scan)
        self.runtime_action = menu.addAction('运行环境检查', self._runtime)
        self.more_button.setMenu(menu)
        layout.addWidget(inline(label('道路变化检测', 'title'), self.logs_button, self.more_button))
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName('roadChangeBody')
        form = form_layout(body)
        form.setContentsMargins(0, 0, 0, 0)
        self.scroll.setWidget(body)
        layout.addWidget(self.scroll, 1)
        project_form = self._section(form, '项目')
        self.project_group = project_form.parentWidget()
        self._data_section(project_form)
        process_form = self._section(form, '处理')
        self.processing_group = process_form.parentWidget()
        self._processing_section(process_form)
        result_form = self._section(form, '成果')
        self.result_group = result_form.parentWidget()
        self._results_section(result_form)
        self._history_section()
        form.addRow(self.records_fold)
        for fold in (self.advanced, self.records_fold):
            fold.toggle.hide()
            fold.hide()
        self.status = label('', 'secondary')
        form.addRow(self.status)
        self.status.hide()
        self._footer(layout)
        self.update_page = UpdatePage(self._back_to_main, self._start_update)
        for page in (self.configuration, self.creation, self.update_page):
            self.pages.addWidget(page)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        for signal, slot in (('task_started', self._started), ('task_progress', self._progress),
                             ('task_log', self._log), ('task_failed', self._failed),
                             ('task_finished', self._finished), ('result_ready', self._result)):
            getattr(controller, signal).connect(slot)
        for entry in controller.history:
            self._log(entry)
        self._ui_ready = True
        self._update_controls()
        for control in self.findChildren(QWidget):
            if isinstance(control, (QPushButton, QLineEdit, QComboBox, QToolButton)):
                control.setMinimumHeight(max(control.minimumHeight(), 28))
        self._refresh_appearance()

    def _section(self, parent_form, title, expand=False):
        group = QGroupBox(title)
        group.setProperty("role", "processingGroup")
        content = form_layout(group)
        content.setContentsMargins(8, 14, 8, 8)
        content.setSpacing(8)
        if expand:
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        parent_form.addRow(group)
        return content

    def _reveal(self, fold):
        visible = fold.isHidden()
        fold.setVisible(visible)
        fold.toggle.setChecked(visible)
        if visible:
            QTimer.singleShot(0, lambda: self.scroll.ensureWidgetVisible(fold) if fold.isVisible() else None)

    def _footer(self, layout):
        self.footer = QWidget()
        box = QVBoxLayout(self.footer)
        box.setContentsMargins(8, 14, 8, 8)
        box.setSpacing(8)
        self.task_line = label("选择项目目录，开始扫描数据")
        self.failure_details = self._button("详情", lambda: self._reveal(self.records_fold))
        self.task_fields = QWidget()
        fields = QGridLayout(self.task_fields)
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setHorizontalSpacing(8)
        fields.setVerticalSpacing(6)
        for row, (title, value) in enumerate((("当前区域", self.area_text), ("当前期次", self.scope_text),
                                             ("当前阶段", self.stage_text), ("已用时间", self.elapsed_text))):
            fields.addWidget(label(title, "secondary"), row, 0)
            fields.addWidget(value, row, 1)
        fields.setColumnMinimumWidth(0, 80)
        fields.setColumnStretch(1, 1)
        box.addWidget(self.task_fields)
        self.running_status = inline(self.progress, self.progress_text)
        box.addWidget(self.running_status)
        box.addWidget(inline(self.task_line, self.failure_details, self.cancel_button, self.run_button))
        layout.addWidget(self.footer)
        for value in (self.state_text, self.progress_note):
            value.setParent(self.footer)
            value.hide()

    def _render_task_state(self):
        if not hasattr(self, 'footer'):
            return
        self.running_status.setVisible(self.busy)
        self.task_fields.setVisible(self.busy)
        self.failure_details.setVisible(bool(self._last_error) and not self.busy)
        if self.busy:
            text = '正在处理'
        elif self._last_error:
            text = '上次处理未完成，请查看日志'
        elif self.state_text.text() == '已完成' and self._started_at is not None:
            text = '用时 ' + self.elapsed_text.text()
        else:
            text = ''
        self.task_line.setText(text)

    def _refresh_appearance(self):
        if getattr(self, "_styling", False):
            return
        self._styling = True
        try:
            # Hosts can take full visual ownership through a standard Qt property.
            palette = self.parentWidget().palette() if self.parentWidget() else QApplication.palette()
            self.setStyleSheet("" if self.property("hostStyled") else dock_style(palette, self.font()))
        finally:
            self._styling = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.FontChange) and hasattr(self, "run_button"):
            self._refresh_appearance()

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.Type.DynamicPropertyChange and bytes(event.propertyName()) == b"hostStyled" and hasattr(self, "run_button"):
            self._refresh_appearance()
        return result

    def _tool_button(self, text, callback):
        button = QToolButton()
        button.setText(text)
        button.setAutoRaise(True)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.clicked.connect(callback)
        return button

    def _button(self, text, callback):
        button = QPushButton(text)
        button.setMinimumWidth(0)
        button.clicked.connect(callback)
        return button

    def _data_section(self, form):
        self.new_button = self._button('新建项目', self._show_creation)
        self.open_button = self._button('打开项目', self._open_project)
        self.project_entry = inline(self.new_button, self.open_button, stretch_first=False)
        self.project_entry.layout().addStretch(1)
        form.addRow(self.project_entry)
        self.project_details = QWidget()
        fields = form_layout(self.project_details)
        fields.setContentsMargins(0, 0, 0, 0)
        self.summary = label('')
        self.project_path = label('', 'secondary')
        self.project_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.project_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.switch_button = self._button('切换项目', self._open_project)
        fields.addRow(inline(self.summary, self.switch_button))
        fields.addRow(self.project_path)
        self.data_state = label('', 'secondary')
        self.configure_button = self._button('配置数据', self._show_configuration)
        fields.addRow(inline(self.data_state, self.configure_button))
        form.addRow(self.project_details)
        self.check_note = label('')
        form.addRow(self.check_note)
        self.configuration = DataConfiguration(self._edited, self._back_to_main, self._save_and_check,
                                                lambda: self._save_and_check(return_to_main=False))
        self.creation = ProjectCreation(self._back_to_main, self._create_project)
        self.check_button = self.configuration.save_button

    def _show_creation(self):
        if self.busy or self.browsing:
            return
        if not self.creation.project_location.text():
            base = Path(self.model["root"]).parent if self.model.get("root") else ROOT.parent
            self.creation.project_location.edit.setText(str(base))
        self.pages.setCurrentWidget(self.creation)

    def _open_project(self):
        if self.busy or self.browsing:
            return
        if not self._allow_project_switch():
            return
        directory = QFileDialog.getExistingDirectory(self, '打开项目', self._project_directory)
        if directory:
            self._load_project(directory)

    def _allow_project_switch(self):
        if not self._draft_dirty:
            return True
        return QMessageBox.question(self, '切换项目', '当前数据配置尚未保存，切换项目会放弃这些修改。是否继续？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def _load_project(self, directory):
        if self.busy:
            return
        self._project_directory = str(directory)
        self._directory_changed()
        self._back_to_main()

    def _create_project(self):
        if self.busy or self.browsing:
            return
        if not self._allow_project_switch():
            return
        try:
            root = create_project(self.creation.project_location.text(), self.creation.project_name.text(), dict(areas=[], periods=[], truths=[], area_irmad_references={}))
        except (OSError, ValueError, TypeError) as exc:
            self.creation.show_issues([str(exc)])
            return
        self.creation.show_issues([])
        self._load_project(root)
        self._created_root = str(root)
        self.scan()

    def _show_configuration(self):
        self.pages.setCurrentWidget(self.configuration)

    def _back_to_main(self):
        self.pages.setCurrentWidget(self.main_page)

    def _continue_task(self):
        if self._resume_task and not self.busy and not self.browsing:
            self._run(self._resume_task.get('action', 'all'), resume=True)

    def _primary(self):
        if self.ui_state in {'unconfigured', 'problem'}:
            self._show_configuration()
            if self.check_note.text():
                self.configuration.locate(self.check_note.text())
        elif self.ui_state == 'resume':
            self._continue_task()
        elif self.ui_state == 'completed':
            self._open_output()
        elif self.ui_state == 'ready':
            if self._current or self._result_keys:
                self._restart()
            else:
                self._run('all')

    def _restart(self):
        if self.busy or self.browsing or not self.checked:
            return
        answer = QMessageBox.question(self, '重新运行完整流程',
            '当前正式成果将重新计算并替换，项目数据和可复用缓存不会删除。',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self._run('all')

    def _show_update(self):
        if self.update_button.isEnabled():
            self.update_page.load(ProjectState(self.model['root'], self.output.text()))
            self.pages.setCurrentWidget(self.update_page)

    def _start_update(self):
        if self.update_page.selection and not self.busy and self.checked and not self.browsing:
            selection = dict(self.update_page.selection)
            self._run(selection.pop('action'), selection=selection)

    def _processing_section(self, form):
        self.run_button = self._button('配置数据', self._primary)
        primary_button(self.run_button)
        self.state_text, self.area_text, self.scope_text = label(''), label('—'), label('—')
        self.stage_text, self.elapsed_text = label('—'), label('00:00:00')
        self.processing_hint = label('')
        form.addRow(self.processing_hint)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setAccessibleName('总体进度')
        self.progress.setTextVisible(False)
        self.progress_text = label('0%', 'secondary')
        self.progress.valueChanged.connect(self._display_progress)
        self.progress_note = label('', 'secondary')
        self.cancel_button = self._button('取消', self.controller.cancel)
        self.advanced = Fold('高级设置')
        self.output = PathField(directory=True)
        self.output.edit.textChanged.connect(self._invalidate_check)
        self.output.edit.editingFinished.connect(self.check_data)
        form.addRow(ResponsiveRow('成果目录', self.output))
        self.advanced_button = self._button('高级设置', lambda: self._reveal(self.advanced))
        self.update_button = self._button('更新部分成果', self._show_update)
        actions = inline(self.update_button, self.advanced_button, stretch_first=False)
        actions.layout().insertStretch(1, 1)
        form.addRow(actions)
        form.addRow(self.advanced)
        self.parameters = {}
        for key, title, value in (('pixel-size', '像元大小（0 自动）', '0.0'), ('absolute', '宽度变化阈值', '2.0'),
                                  ('ratio', '宽度变化比例', '0.2'), ('tolerance', '匹配容差', '3.0')):
            field = QLineEdit(value)
            self.parameters[key] = field
            field.textChanged.connect(self._invalidate_check)
            field.editingFinished.connect(self.check_data)
            self.advanced.form.addRow(title, field)

    def _results_section(self, form):
        self.result_summary = label('尚无当前成果', 'secondary')
        self.metrics = label('', 'secondary')
        self.configuration_note = label('', 'secondary')
        self.locate_button = self._button('打开成果目录', self._open_output)
        form.addRow(self.result_summary)
        form.addRow(self.metrics)
        form.addRow(self.configuration_note)
        form.addRow(inline(label(''), self.locate_button))
        self._reset_results()

    def _show_logs(self):
        self._reveal(self.records_fold)

    def _history_section(self):
        self.records_fold = Fold("日志")
        form = self.records_fold.form
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(2000)
        form.addRow(self.logs)
        self.error_detail = QPlainTextEdit()
        self.error_detail.setReadOnly(True)
        self.error_detail.setMaximumHeight(120)
        form.addRow("错误详情", self.error_detail)
        form.addRow(self._button("复制日志", lambda: QApplication.clipboard().setText(self.logs.toPlainText())))
        form.addRow(self._button("打开日志目录", self._open_logs))

    def _directory_changed(self):
        self._created_root = None
        self._project_revision += 1
        self.checked = False
        self._draft_dirty = False
        self._processing_dirty = False
        self._return_after_check = False
        if not self.busy and hasattr(self, "run_button"):
            self.model = {"areas": [], "periods": [], "truths": [], "issues": []}
            self._resume_task = None
            self._project_operation = {}
            self.state_text.setText("待就绪")
            self._reset_results()
            self._set_current(None)
            self._log_lines.clear()
            self._last_error = ""
            self._task_log_path = None
            for fold in (self.advanced, self.records_fold):
                fold.hide()
                fold.toggle.setChecked(False)
            self.status.hide()
            self._show_record()
            self.configuration.load(self.model)
            self._show_problems([])
            self._update_controls()
            self.open_timer.start()

    def _invalidate_check(self):
        if self._restoring:
            return
        self._project_revision += 1
        self.checked = False
        if not self._draft_dirty:
            self._processing_dirty = True
        if hasattr(self, "status"):
            self._update_controls()

    def _edited(self, *args):
        if self._restoring:
            return
        self._draft_dirty = True
        self._return_after_check = False
        self._invalidate_check()
        self._show_problems(["数据配置已修改，请保存并检查"])

    def _background(self, function, args, callback):
        self.browsing = True
        job = BrowseJob(function, *args)
        self._job = job
        self._jobs[id(job)] = job
        revision = self._project_revision
        self._update_controls()
        def complete(value=None, error=None):
            self._jobs.pop(id(job), None)
            if self._job is not job:
                return
            self.browsing = False
            if revision == self._project_revision:
                if error is None:
                    callback(value)
                else:
                    self._return_after_check = False
                    self._show_problems([error])
            self._update_controls()
        job.signals.completed.connect(lambda value: complete(value=value))
        job.signals.failed.connect(lambda message: complete(error=message))
        QThreadPool.globalInstance().start(job)

    def scan(self):
        self.open_timer.stop()
        if self.busy:
            return
        if not self._project_directory:
            self._show_problems([])
            return
        self._project_revision += 1
        self.checked = False
        self._return_after_check = False
        self._background(scan_project, (self._project_directory,), self._scanned)

    def _scanned(self, model):
        self.open_timer.stop()
        self._restoring = True
        self.model = model
        self.output.edit.setText(model["output"])
        saved = model.get("config", {}).get("plugin_processing_parameters", {})
        for key, field in self.parameters.items():
            field.setText(str(saved.get(key, {"pixel-size": "0.0", "absolute": "2.0", "ratio": "0.2", "tolerance": "3.0"}[key])))
        self.configuration.load(model)
        self._draft_dirty = False
        self._processing_dirty = False
        self._set_current(model.get("current"))
        log = Path(model['root']) / '_logs/plugin_ui/current.log'
        self._task_log_path = None
        try:
            self._log_lines = log.read_text(encoding='utf-8').splitlines()[-2000:] if log.is_file() else []
        except OSError:
            self._log_lines = []
        self._refresh_results()
        self._restoring = False
        if self._created_root == model["root"]:
            self._show_configuration()
            self._created_root = None
        self.check_data()

    def _apply_corrections(self):
        self.model.update(self.configuration.inputs(), issues=[], output=self.output.text())
        for key in ('areas', 'periods', 'truths'):
            for row in self.model[key]:
                if row[-1]:
                    row[-1] = '\n'.join(str(resolve(value, Path(self.model.get('root', ROOT))))
                                          for value in row[-1].splitlines() if value.strip())
        self.checked = False

    def _save_and_check(self, *, return_to_main=True):
        if self.busy or self.browsing or not self.model.get("root") or self.configuration.has_pending_edit:
            return
        self._project_revision += 1
        self._apply_corrections()
        try:
            ProjectState(self.model['root'], self.output.text()).remember_configuration()
            save_configuration(self.model, {key: field.text() for key, field in self.parameters.items()})
            self.model = scan_project(self.model["root"])
            self.configuration.load(self.model)
            self._refresh_results()
        except (OSError, ValueError, TypeError) as exc:
            self._show_problems([f"项目配置保存失败：{exc}"])
            return
        self._draft_dirty = False
        self._processing_dirty = False
        self._return_after_check = return_to_main
        self.check_data()

    def _show_problems(self, issues):
        self.check_note.setText(issues[0] if issues else "")
        self.check_note.setVisible(bool(issues))
        self.configuration.show_issues(issues)
        self._update_controls()

    def check_data(self):
        if self.busy or not self.model.get("root"):
            return
        if self._draft_dirty:
            self._show_problems(["数据配置已修改，请保存并检查"])
            return
        self.checked = False
        model = copy.deepcopy(self.model)
        data = self._data()
        def inspect():
            try:
                local_issues = model.get("issues", []) + check_files(model)
                if local_issues:
                    return local_issues
                self.controller.build_command("all", data)
                report = self.controller.inspect_data(data)
                return report.get("issues", [])
            except Exception as exc:
                # The existing check adapter can return stderr; expose its reason,
                # not an entire traceback in the compact project area.
                lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
                reason = lines[-1] if lines else "数据检查未返回报告"
                for prefix in ("ValueError: ", "RuntimeError: ", "FileNotFoundError: "):
                    reason = reason.removeprefix(prefix)
                return [reason]
        self._background(inspect, (), self._checked)

    def _checked(self, issues):
        if isinstance(issues, dict):
            issues = issues.get("issues", [])
        issues = list(dict.fromkeys(issues))
        if not issues and self._processing_dirty:
            try:
                ProjectState(self.model['root'], self.output.text()).remember_configuration()
                self.model['output'] = self.output.text()
                save_configuration(self.model, {key: field.text() for key, field in self.parameters.items()})
                self._processing_dirty = False
            except (OSError, ValueError, TypeError) as exc:
                issues = [f'处理参数保存失败：{exc}']
        self.checked = not issues
        self._show_problems(issues)
        self._refresh_results()
        self.state_text.setText("待配置" if issues else "")
        self._update_controls()
        if self._return_after_check:
            self._return_after_check = False
            if self.checked:
                self._back_to_main()
            else:
                self._show_configuration()
                if issues:
                    self.configuration.locate(issues[0])

    def _set_current(self, current):
        self._current = current

    def _data(self):
        return dict(project_root=self.model.get('root', ''),
                    area_irmad_references=self.model.get('area_irmad_references', {}),
                    areas=self.model['areas'], periods=self.model['periods'], truths=self.model['truths'],
                    output=self.output.text(), profile='fast',
                    evaluate=any(Path(row[-1]).is_file() for row in self.model['truths']),
                    truth_type_field='', resume=False, manifest=self._current['path'] if self._current else '',
                    grid='', period='', before_period='', after_period='', device='auto',
                    **{key: field.text() for key, field in self.parameters.items()})

    def _update_controls(self, *args):
        if not getattr(self, '_ui_ready', False):
            return
        available = not self.busy and not self.browsing
        opened = bool(self.model.get('root'))
        self.project_entry.setVisible(not opened)
        self.project_details.setVisible(opened)
        self.project_group.setTitle('项目' if opened else '')
        for section in (self.processing_group, self.result_group, self.footer, self.logs_button, self.more_button):
            section.setVisible(opened)
        self.new_button.setEnabled(available)
        self.new_project_action.setEnabled(available)
        self.open_button.setEnabled(available)
        self.switch_button.setEnabled(available)
        self.creation.body.setEnabled(available)
        self.creation.save_button.setEnabled(available)
        self.configuration.body.setEnabled(available)
        self.check_button.setEnabled(available and opened and not self.configuration.has_pending_edit)
        self.configure_button.setEnabled(available and opened)
        self.advanced.setEnabled(available)
        self.output.setEnabled(available)
        self.rescan_action.setEnabled(available and opened)
        self.restart_action.setEnabled(available and opened and self.checked)
        compatible = bool(self._current) and self._current['data'].get('execution_profile') == 'fast'
        self.update_button.setVisible(bool(self._result_keys))
        self.update_button.setEnabled(available and self.checked and compatible and bool(self._result_keys) and not self._resume_task)
        self.update_page.body.setEnabled(available)
        self.update_page.save_button.setEnabled(self.update_button.isEnabled() and bool(self.update_page.scope))
        self.cancel_button.setVisible(self.busy)
        self.cancel_button.setEnabled(self.busy)
        if opened:
            project_name = self.model.get('config', {}).get('project_name') or Path(self.model['root']).name
            self.summary.setText(f"{project_name} · {len(self.model['areas'])} 个区域 · {len(self.model['periods'])} 个期次")
            self.project_path.setText(self.model['root'])
        if not opened:
            state, title, description = 'closed', '', ''
        elif self.busy:
            state, title, description = 'running', '', '道路变化检测正在处理'
        elif self.browsing:
            state, title, description = 'checking', '正在检查', '正在读取项目并检查数据'
        elif self._resume_task:
            local = self._resume_task.get('action', 'all') != 'all'
            state, title = 'resume', '继续更新' if local else '继续处理'
            description = '上次更新尚未完成' if local else '上次处理尚未完成'
            if local:
                scope = self._resume_task.get('scope') or {}
                target = (scope.get('periods') or scope.get('changes') or [])
                description += '，继续完成上次选定范围\n' + scope.get('grid', '') + ' / ' + '、'.join(target).replace('_to_', ' → ')
        elif not self.model['areas']:
            state, title, description = 'unconfigured', '配置数据', '项目尚未配置数据'
        elif not self.checked:
            state, title, description = 'problem', '修正数据', '请修正数据配置后再处理'
        elif self._result_keys and self._project_operation.get('status') == 'completed':
            state, title, description = 'completed', '', '处理已完成'
            if self._configuration_status == 'changed':
                description = '数据配置已修改，当前成果尚未更新；完整重算可应用新配置'
        else:
            state, title, description = 'ready', '开始处理', '项目数据已就绪'
        self.ui_state = state
        self.run_button.setText(title)
        self.run_button.setVisible(opened and not self.busy and state != 'completed')
        self.run_button.setEnabled(available and opened)
        self.processing_hint.setText(description)
        self.data_state.setText('尚未配置' if not self.model['areas'] else '检查中' if self.browsing else '数据已就绪' if self.checked else '数据有问题')
        self.check_note.setVisible(bool(self.check_note.text()))
        if self._result_keys and (self._draft_dirty or self._processing_dirty):
            self.configuration_note.setText('配置已修改，当前成果尚未更新。')
        self.configuration_note.setVisible(bool(self.configuration_note.text()))
        self.metrics.setVisible(bool(self.metrics.text()))
        self._render_task_state()

    def _run(self, action, resume=False, selection=None):
        if self.busy or self.browsing or (not self.checked and not resume):
            return
        self._action = action
        self.busy = True
        self.state_text.setText('正在启动')
        self._last_error = ''
        self._back_to_main()
        self._update_controls()
        data = self._data()
        data.update(selection or {})
        data['resume'] = resume
        if action == 'all' and not resume:
            try:
                ProjectState(self.model['root'], self.output.text()).remember_configuration()
                save_configuration(self.model, {key: field.text() for key, field in self.parameters.items()})
            except (OSError, ValueError) as exc:
                self._failed({'message': str(exc), 'detail': '配置保存失败'})
                return
        self.controller.run(action, data)

    def _runtime(self):
        self.status.setText(self.controller.runtime_message())
        self.status.show()
        self.scroll.ensureWidgetVisible(self.status)

    def _started(self, payload):
        self.busy = True
        self._log_lines = [f"[{e.get('level', 'INFO')}] {e.get('message', '')}" for e in self.controller.history]
        self._last_error = ""
        self._show_record()
        self._refresh_results()
        self._started_at = time.monotonic()
        self._active_record = payload["task_id"]
        self._task_log_path = None
        root = self.model.get("root")
        if root:
            folder = Path(root) / "_logs/plugin_ui"
            try:
                folder.mkdir(parents=True, exist_ok=True)
                self._task_log_path = folder / "current.log"
                self._task_log_path.write_text("\n".join(self._log_lines) + ("\n" if self._log_lines else ""), encoding="utf-8")
            except OSError:
                self.status.setText("日志目录不可写，日志仅保留在本次会话")
        self.state_text.setText("运行中")
        self.area_text.setText("—")
        self.scope_text.setText("—")
        self.stage_text.setText("准备输入")
        self.progress.setRange(0, 0)
        self.progress_note.setText("等待后端总体进度")
        self._display_progress()
        self.timer.start()
        self._log({"task_id": self._active_record, "message": "任务开始", "level": "INFO"})
        self._update_controls()

    def _display_progress(self, *args):
        self.progress_text.setText("—" if self.progress.maximum() == 0 else f"{max(0, self.progress.value()) / 10:.0f}%")

    def _tick(self):
        self._display_progress()
        seconds = int(time.monotonic() - self._started_at) if self._started_at is not None else 0
        self.elapsed_text.setText(f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}")

        self._render_task_state()

    def _progress(self, payload):
        event = payload.get("event", {})
        self.stage_text.setText(payload.get("stage") or "处理中")
        area = event.get("grid") or event.get("area_id")
        if area:
            self.area_text.setText(str(area))
        if event.get("before_period") and event.get("after_period"):
            self.scope_text.setText(f"{event['before_period']} → {event['after_period']}")
        elif event.get("period"):
            self.scope_text.setText(str(event["period"]))
        # Only the explicit aggregate backend progress drives the overall bar.
        # Stage-local fractions must never jump the whole task to 100%.
        if event.get("kind") == "pipeline" and "progress" in event:
            try:
                value = max(0, min(1000, round(float(event["progress"]) * 1000)))
                self.progress.setRange(0, 1000)
                self.progress.setValue(value)
                self.progress_note.setText(f"已处理 {event.get('completed', '—')} / {event.get('total', '—')} 个工作单元")
            except (TypeError, ValueError):
                pass
        elif self._action != "all" and event.get("kind") == "pipeline" and event.get("total"):
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(payload.get("progress", 0) * 1000))
            self.progress_note.setText("当前重跑任务进度")
        if event.get("kind") == "pipeline" and event.get("stage") == "长时序道路汇总":
            self.progress.setRange(0, 0)
            self.progress_note.setText("处理单元已完成，正在汇总正式成果")

        self._render_task_state()

    def _log(self, payload):
        text = f"[{payload.get('level', 'INFO')}] {payload.get('message', '')}"
        self._log_lines.append(text)
        del self._log_lines[:-2000]
        self.logs.appendPlainText(text)
        if self._task_log_path:
            try:
                with self._task_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(text + "\n")
            except OSError:
                self._task_log_path = None

    def _show_record(self, *args):
        self.logs.setPlainText("\n".join(self._log_lines))
        self.error_detail.setPlainText(self._last_error)

    def _failed(self, payload):
        self.busy = False
        self.timer.stop()
        self.state_text.setText("失败")
        self._failure_reason = payload.get("message", "任务失败").splitlines()[0]
        self.status.setText("任务未完成，请查看日志中的错误详情")
        self.progress.setRange(0, 1000)
        self._log({**payload, "level": "ERROR"})
        self._refresh_results()
        self._last_error = payload["message"] + "\n" + payload.get("detail", "")
        self._show_record()
        self._update_controls()

    def _finished(self, payload):
        self.busy = False
        self._tick()
        self.timer.stop()
        finished = payload["status"] == "completed"
        self.state_text.setText("已完成" if finished else "已取消")
        self.progress.setRange(0, 1000)
        if finished:
            self.progress.setValue(1000)
        self.progress_note.setText("成果已生成" if finished else "已停止，已完成成果可供续跑")
        self.status.setText("处理完成，当前成果已生成" if finished else "任务已取消")
        self._log({**payload, "message": self.state_text.text()})
        self._refresh_results(latest=True)
        self._update_controls()

    def _reset_results(self):
        self._configuration_status = 'unknown'
        self._result_keys.clear()
        self._result_payloads = []
        self.result_summary.setText('尚无当前成果')
        self.metrics.clear()
        self.configuration_note.clear()
        self.locate_button.setEnabled(False)

    def _refresh_results(self, latest=False):
        self.result_timer.stop()
        root = self.model.get('root')
        if not root:
            return
        try:
            store = ProjectState(root, self.output.text())
            self._project_operation = store.state()
            self._resume_task = self._project_operation if store.resumable() else None
            self._set_current(store.descriptor())
            self._reset_results()
            self._result_payloads = formal_results(store)
            self._result_keys = {(p['result_type'], p['path']) for p in self._result_payloads}
            periods, changes = set(), set()
            for product in self._result_payloads:
                meta = product.get('metadata', {})
                if product['result_type'] in {'road_centerline', 'road_surface'}:
                    periods.add((meta.get('grid'), meta.get('period')))
                elif product['result_type'] == 'road_change':
                    changes.add((meta.get('grid'), meta.get('before_period'), meta.get('after_period')))
                elif product['result_type'] == 'road_evaluation':
                    self.metrics.setText(metrics_text(Path(product['path'])) if Path(product['path']).suffix.lower() == '.json' else self.metrics.text() or '评价报告已生成')
            if self._result_keys:
                self.result_summary.setText(f'当前成果：{len(periods)} 个期次道路成果、{len(changes)} 个变化对成果')
                if not self.metrics.text() and not self.model['truths']:
                    self.metrics.setText('未配置真值，未执行评价')
            self.locate_button.setEnabled(bool(self._result_keys))
            freshness = store.configuration_status()
            self._configuration_status = freshness
            self.configuration_note.setText(
                '数据配置已修改，当前成果尚未更新。局部更新仍针对上次已处理的数据。' if freshness == 'changed'
                else '旧成果缺少配置版本记录，尚不能确认与当前配置一致。' if freshness == 'unknown' and self._result_keys else '')
            self._last_error = store.state().get('last_error', self._last_error)
            self._show_record()
        except (OSError, ValueError, TypeError) as exc:
            self._resume_task = None
            self.result_summary.setText(f'项目状态读取失败：{exc}')
        self._update_controls()

    def _result(self, payload):
        # Completion can report many paths in one batch; read the index once.
        self.result_timer.start(0)

    def _open_output(self):
        if not self.model.get('root'):
            return
        folder = Path(self._project_operation.get('output') or self.output.text())
        if not folder.is_dir():
            self.status.setText('成果目录不存在，请刷新并检查项目')
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self.status.setText(f'无法打开成果目录：{folder}')
        else:
            return
        self.status.show()

    def _open_logs(self):
        root = self.model.get("root")
        if not root:
            self.status.setText("请先选择并扫描项目")
            return
        folder = Path(root) / "_logs"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
                self.status.setText(f"无法打开日志目录：{folder}")
        except OSError as exc:
            self.status.setText(str(exc))
