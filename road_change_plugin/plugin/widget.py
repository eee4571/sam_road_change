"""Compact processing dock; only UI input selection and presentation live here."""
import copy
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QComboBox, QScrollArea, QLabel, QPushButton, QLineEdit,
    QCheckBox, QProgressBar, QPlainTextEdit,
    QGridLayout, QToolButton, QMenu, QGroupBox, QStackedWidget)
from .ui.forms import PathField, Fold, form_layout
from .ui.presentation import inline, primary_button, result_menu
from .ui.appearance import dock_style
from .ui.data_configuration import DataConfiguration
from .ui.desktop import ResponsiveRow, ResultChooser
from .ui.project_browser import ROOT, scan_project, check_files, pairs, discover_tasks, natural, resolve, save_configuration
from .ui.background import BrowseJob
from .ui.evaluation_summary import metrics_text
from .result_parser import read_results

RESULT_LABELS = {"road_centerline": "道路中心线", "road_surface": "道路面",
                 "road_width": "道路宽度", "road_change": "变化检测",
                 "road_temporal": "长时序", "road_evaluation": "精度评价"}
GROUPS = {"road_centerline": "单期道路", "road_surface": "单期道路", "road_width": "单期道路",
          "road_change": "变化检测", "road_temporal": "长时序", "road_evaluation": "精度评价"}
DETAIL_LABELS = {"changes": "全部变化", "added": "新增道路", "removed": "消失道路",
                 "widened": "道路拓宽", "narrowed": "道路收窄", "gpkg": "变化数据集",
                 "life_shp": "道路生命周期", "observations_shp": "逐期观测",
                 "events_shp": "变化事件", "event_parts_shp": "事件分段",
                 "lineage_shp": "道路沿革", "csv": "评价表格", "json": "评价报告"}
STATUS = {"completed": "已完成", "running": "运行中", "failed": "失败",
          "cancelled": "已取消", "completed_with_errors": "部分失败", "unknown": "历史任务"}


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
        self.model = {"areas": [], "periods": [], "truths": [], "tasks": [], "issues": []}
        self.checked = False
        self.busy = False
        self.browsing = False
        self._started_at = None
        self._action = "all"
        self._job = None
        self._task_log_path = None
        self._records = {}
        self._active_record = "session"
        self._result_keys = set()
        self._project_revision = 0
        self._jobs = {}
        self._return_after_check = False
        self._restoring = False
        self._draft_dirty = False
        self._resume_task = None
        self.open_timer = QTimer(self)
        self.open_timer.setSingleShot(True)
        self.open_timer.setInterval(350)
        self.open_timer.timeout.connect(self.scan)
        self.setObjectName("roadChangeDock")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages)
        self.main_page = QWidget()
        self.pages.addWidget(self.main_page)
        layout = QVBoxLayout(self.main_page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        logs_button = self._tool_button("记录", lambda: self._reveal(self.records_fold))
        more = self._tool_button("…", lambda: None)
        logs_button.setToolTip("运行记录")
        more.setToolTip("更多操作")
        logs_button.setProperty("role", "toolbarAction")
        more.setProperty("role", "toolbarAction")
        menu = QMenu(more)
        menu.addAction("局部重跑", lambda: self._reveal(self.local))
        menu.addAction("检查运行资源", self._runtime)
        self.rescan_action = menu.addAction("重新扫描", self.scan)
        self.recheck_action = menu.addAction("重新检查", self.check_data)
        more.setMenu(menu)
        layout.addWidget(inline(label("道路变化检测", "title"), logs_button, more))
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName("roadChangeBody")
        form = form_layout(body)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self.scroll.setWidget(body)
        layout.addWidget(self.scroll, 1)
        self._data_section(self._section(form, "项目与数据"))
        self._processing_section(self._section(form, "处理参数"))
        self._results_section(self._section(form, "成果与评价", expand=True))
        self._history_section()
        for fold in (self.local, self.records_fold):
            form.addRow(fold)
        for fold in (self.advanced, self.local, self.records_fold):
            fold.toggle.hide()
            fold.setVisible(False)
        self.status = label("选择项目目录，开始扫描数据", "secondary")
        form.addRow(self.status)
        self.status.hide()
        self.status.setWordWrap(True)

        self._footer(layout)
        self.pages.addWidget(self.configuration)
        self.pages.setCurrentWidget(self.main_page)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        controller.task_started.connect(self._started)
        controller.task_progress.connect(self._progress)
        controller.task_log.connect(self._log)
        controller.task_failed.connect(self._failed)
        controller.task_finished.connect(self._finished)
        controller.result_ready.connect(self._result)
        for entry in controller.history:
            self._log(entry)
        self._update_controls()
        # Minimums follow the inherited font; the host may make controls larger.
        for control in self.findChildren(QWidget):
            if not isinstance(control, (QPushButton, QLineEdit, QComboBox, QToolButton)):
                continue
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
            self.scroll.ensureWidgetVisible(fold)

    def _footer(self, layout):
        self.footer = QGroupBox("任务状态")
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
        if not hasattr(self, "footer"):
            return
        self.running_status.setVisible(self.busy)
        self.task_fields.setVisible(self.busy)
        state = self.state_text.text()
        self.failure_details.setVisible(state == "失败")
        if self.busy:
            self.task_line.setText("运行中")
        elif state in ("已完成", "已取消"):
            self.task_line.setText(f"{state} · {self.area_text.text()} · {self.elapsed_text.text()}")
        elif state == "失败":
            self.task_line.setText("失败 · " + getattr(self, "_failure_reason", "请查看详情")[:120])
        else:
            self.task_line.setText("正在准备项目…" if self.browsing else "" if self.checked or self.model["areas"] else "选择项目目录开始")

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
        self.project = PathField(directory=True)
        self.project.edit.setPlaceholderText("选择项目目录，自动读取和检查数据")
        self.project.edit.textChanged.connect(self._directory_changed)
        self.summary = label("尚未打开项目")
        self.check_note = label("")
        self.configure_button = self._button("配置数据", self._show_configuration)
        self.scan_button = self._button("重新扫描", self.scan)
        form.addRow(self.project)
        form.addRow(self.summary)
        actions = inline(self.configure_button, self.scan_button, stretch_first=False)
        actions.layout().addStretch(1)
        form.addRow(actions)
        form.addRow(self.check_note)
        self.configuration = DataConfiguration(self._edited, self._back_to_main, self._save_and_check)
        self.areas, self.periods, self.truths = self.configuration.areas, self.configuration.periods, self.configuration.truths
        self.check_button = self.configuration.save_button
        self.resume_button = self._button("继续任务", self._continue_task)
        form.addRow(self.resume_button)

    def _show_configuration(self):
        self.pages.setCurrentWidget(self.configuration)

    def _back_to_main(self):
        self.pages.setCurrentWidget(self.main_page)

    def _continue_task(self):
        if not self._resume_task or not self.checked or self.busy or self.browsing:
            return
        for index in range(self.task.count()):
            if self.task.itemData(index) and self.task.itemData(index)["id"] == self._resume_task["id"]:
                self.task.setCurrentIndex(index)
                break
        self.resume.setChecked(True)
        self._run("all")

    def _processing_section(self, form):
        self.run_button = self._button("运行完整流程", lambda: self._run("all"))
        primary_button(self.run_button)
        self.state_text = label("待就绪")
        self.area_text = label("—")
        self.scope_text = label("—")
        self.stage_text = label("—")
        self.elapsed_text = label("00:00:00")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setAccessibleName("总体进度")
        self.progress.setTextVisible(False)
        self.progress_text = label("0%", "secondary")
        self.progress.valueChanged.connect(self._display_progress)
        self.progress_note = label("等待任务开始", "secondary")
        self.cancel_button = self._button("取消", self.controller.cancel)
        self.local = Fold("局部重跑")
        self.task = QComboBox()
        self.task.setMinimumWidth(0)
        self.task.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.task.currentIndexChanged.connect(self._task_changed)
        self.grid, self.period, self.pair = QComboBox(), QComboBox(), QComboBox()
        self.grid.currentIndexChanged.connect(self._grid_changed)
        self.local.form.addRow("当前项目任务", self.task)
        self.local.form.addRow("区域", self.grid)
        self.local.form.addRow("期次", self.period)
        self.rerun_period = self._button("重跑该期并更新相关成果", lambda: self._run("rerun-period"))
        self.local.form.addRow(self.rerun_period)
        self.local.form.addRow("变化对", self.pair)
        self.rerun_pair = self._button("重跑该变化对", lambda: self._run("rerun-change"))
        self.local.form.addRow(self.rerun_pair)
        self.advanced = Fold("高级设置")
        self.output = PathField(directory=True)
        self.output.edit.textChanged.connect(self._invalidate_check)
        self.output.edit.editingFinished.connect(self.check_data)
        form.addRow(ResponsiveRow("成果目录", self.output))
        advanced_button = self._button("高级设置", lambda: self._reveal(self.advanced))
        form.addRow(inline(label("道路变化检测"), advanced_button))
        form.addRow(self.advanced)
        self.run_id = QLineEdit()
        self.run_id.setPlaceholderText("留空自动命名")
        self.advanced.form.addRow("任务名称", self.run_id)
        self.resume = QCheckBox("继续所选的未完成任务")
        self.advanced.form.addRow(self.resume)
        self.parameters = {}
        for key, title, value in (("pixel-size", "像元大小（0 自动）", "0.0"), ("absolute", "宽度变化阈值", "2.0"),
                                  ("ratio", "宽度变化比例", "0.2"), ("tolerance", "匹配容差", "3.0")):
            field = QLineEdit(value)
            self.parameters[key] = field
            field.textChanged.connect(self._invalidate_check)
            field.editingFinished.connect(self.check_data)
            self.advanced.form.addRow(title, field)
        self.advanced.form.addRow(self._button("检查运行资源", self._runtime))

    def _results_section(self, form):
        self.result_summary = label("扫描项目后自动发现已有成果", "secondary")
        form.addRow(self.result_summary)
        self.results = QWidget()
        grid = QVBoxLayout(self.results)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        self.group_buttons = {}
        self.result_labels = {}
        self.result_menus = {}
        for row, name in enumerate(("单期道路", "变化检测", "长时序", "精度评价")):
            heading = label(name)
            summary = label("暂无成果", "secondary")
            button = self._button("查看", lambda _checked=False, group=name: self._open_group(group))
            button.setProperty("role", "resultAction")
            button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            button.customContextMenuRequested.connect(
                lambda point, group=name, target=button: self.result_menus[group].exec(target.mapToGlobal(point))
                if group in self.result_menus else None)
            line = QWidget()
            line.setProperty("role", "resultRow")
            line.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            line.setMinimumHeight(32)
            columns = QHBoxLayout(line)
            columns.setContentsMargins(0, 0, 0, 0)
            columns.setSpacing(8)
            heading.setMinimumWidth(heading.fontMetrics().horizontalAdvance("单期道路"))
            columns.addWidget(heading)
            columns.addWidget(summary, 1)
            columns.addWidget(button)
            grid.addWidget(line)
            self.result_labels[name] = summary
            self.group_buttons[name] = button
        self.metrics = self.result_labels["精度评价"]
        self._reset_results()
        form.addRow(self.results)
        self.chooser = ResultChooser(self._open_result)
        form.addRow(self.chooser)
        self.chooser.hide()
        self._ui_ready = True

    def _history_section(self):
        self.records_fold = Fold("运行记录")
        form = self.records_fold.form
        self.recent = QComboBox()
        self.recent.setMinimumWidth(0)
        self.recent.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.recent.currentIndexChanged.connect(self._show_record)
        form.addRow("当前 / 最近任务", self.recent)
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
        self._project_revision += 1
        self.checked = False
        self._draft_dirty = False
        self._return_after_check = False
        if not self.busy and hasattr(self, "run_button"):
            self.model = {"areas": [], "periods": [], "truths": [], "tasks": [], "issues": []}
            self._resume_task = None
            self.state_text.setText("待就绪")
            self._reset_results()
            self._set_tasks([])
            self.configuration.load(self.model)
            self._show_problems([])
            self._update_controls()
            self.open_timer.start()

    def _invalidate_check(self):
        if self._restoring:
            return
        self._project_revision += 1
        self.checked = False
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
        if not self.project.text():
            self._show_problems([])
            return
        self._project_revision += 1
        self.checked = False
        self._return_after_check = False
        self._background(scan_project, (self.project.text(),), self._scanned)

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
        self.resume.setChecked(False)
        self.run_id.clear()
        self._set_tasks(model["tasks"])
        self._refresh_results()
        self._resume_task = next((task for task in model["tasks"] if task["status"] in {"running", "failed", "cancelled", "completed_with_errors"}
                                  and task["data"].get("execution_profile") == "fast"), None)
        self._restoring = False
        self.check_data()

    def _apply_corrections(self):
        self.model.update(areas=self.areas.values(), periods=self.periods.values(), truths=self.truths.values(),
                          area_irmad_references=self.configuration.reference_values(), issues=[], output=self.output.text())
        for key in ("areas", "periods", "truths"):
            for row in self.model[key]:
                if row[-1]:
                    row[-1] = str(resolve(row[-1], Path(self.model.get("root", ROOT))))
        self.checked = False
        self._task_changed()

    def _save_and_check(self):
        if self.busy or self.browsing or not self.model.get("root"):
            return
        self._project_revision += 1
        self._apply_corrections()
        try:
            save_configuration(self.model, {key: field.text() for key, field in self.parameters.items()})
        except (OSError, ValueError, TypeError) as exc:
            self._show_problems([f"项目配置保存失败：{exc}"])
            return
        self._draft_dirty = False
        self._return_after_check = True
        self.check_data()

    def _show_problems(self, issues):
        self.check_note.setText(issues[0] if issues else "")
        self.check_note.setVisible(bool(issues))
        self.configuration.show_issues(issues)

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
        self.checked = not issues
        self._show_problems(issues)
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

    def _set_tasks(self, tasks):
        selected = self.task.currentData()
        selected_id = selected["id"] if selected else ""
        self.task.blockSignals(True)
        self.task.clear()
        for task in tasks:
            self.task.addItem(f"{task['id']} · {STATUS.get(task['status'], task['status'])}", task)
        if not tasks:
            self.task.addItem("暂无可重跑任务", None)
        else:
            for index, task in enumerate(tasks):
                if task["id"] == selected_id:
                    self.task.setCurrentIndex(index)
        self.task.blockSignals(False)
        self._task_changed()
        for task in tasks:
            task_id = task["id"]
            if task_id not in self._records:
                data = task["data"]
                self._records[task_id] = {"lines": [f"任务：{task_id}", f"状态：{STATUS.get(task['status'], task['status'])}"],
                                          "error": str(data.get("last_error") or "")}
                self.recent.addItem(f"{task_id} · {STATUS.get(task['status'], task['status'])}", task_id)

    def _task_changed(self, *args):
        task = self.task.currentData()
        data = task["data"] if task else {}
        self._period_rows = [[e.get("grid", ""), e.get("period", "")] for e in data.get("period_results", [])]
        if not self._period_rows:
            self._period_rows = [r[:2] for r in self.model["periods"]]
        self.grid.clear()
        self.grid.addItems(sorted({r[0] for r in self._period_rows}, key=natural))
        self._grid_changed()
        if getattr(self, "_ui_ready", False):
            self._update_controls()

    def _grid_changed(self, *args):
        area = self.grid.currentText()
        self.period.clear()
        self.period.addItems(sorted({r[1] for r in getattr(self, "_period_rows", []) if r[0] == area}, key=natural))
        self.pair.clear()
        for a, before, after in pairs(getattr(self, "_period_rows", [])):
            if a == area:
                self.pair.addItem(f"{before} → {after}", (before, after))

    def _data(self):
        task = self.task.currentData()
        pair = self.pair.currentData() or ("", "")
        run_id = task["id"] if self.resume.isChecked() and task else self.run_id.text()
        return dict(area_irmad_references=self.model.get("area_irmad_references",{}),areas=self.model["areas"], periods=self.model["periods"], truths=self.model["truths"],
                    output=self.output.text(), profile="fast", evaluate=any(Path(row[-1]).is_file() for row in self.model["truths"]),
                    truth_type_field="", run_id=run_id, resume=self.resume.isChecked(),
                    manifest=task["path"] if task else "", grid=self.grid.currentText(), period=self.period.currentText(),
                    before_period=pair[0], after_period=pair[1], device="auto",
                    **{key: field.text() for key, field in self.parameters.items()})

    def _update_controls(self, *args):
        if not getattr(self, "_ui_ready", False):
            return
        available = not self.busy and not self.browsing
        same_root = bool(self.model.get("root")) and self.controller.path(self.project.text()) == Path(self.model["root"])
        has_data = bool(self.model["areas"]) and same_root
        self.run_button.setEnabled(available and self.checked)
        self.scan_button.setEnabled(available)
        self.check_button.setEnabled(available and has_data)
        self.project.setEnabled(not self.busy)
        self.rescan_action.setEnabled(available)
        self.recheck_action.setEnabled(available and has_data)
        self.configuration.body.setEnabled(available)
        self.configure_button.setEnabled(bool(self.model.get("root")))
        self.advanced.setEnabled(available)
        self.output.setEnabled(available)
        self.cancel_button.setEnabled(self.busy)
        self.cancel_button.setVisible(self.busy)
        has_task = bool(self.task.currentData()) and same_root and self.task.currentData()["data"].get("execution_profile") == "fast"
        self.resume.setEnabled(available and has_task)
        if not has_task:
            self.resume.setChecked(False)
        self.rerun_period.setEnabled(available and has_task and self.period.count() > 0)
        self.rerun_pair.setEnabled(available and has_task and self.pair.count() > 0)
        project_name = Path(self.model["root"]).name if same_root else "尚未打开项目"
        self.summary.setText(f"项目：{project_name}  {len(self.model['areas'])} 个区域  {len(self.model['periods'])} 个期次" if same_root else project_name)
        self.resume_button.setVisible(bool(self._resume_task))
        self.resume_button.setEnabled(available and self.checked and bool(self._resume_task))
        if self._resume_task:
            self.resume_button.setText("继续任务：" + self._resume_task["id"])
        self.check_note.setVisible(bool(self.check_note.text()))
        self._render_task_state()

    def _run(self, action):
        self._action = action
        self.busy = True
        self.state_text.setText("正在启动")
        self._update_controls()
        self.controller.run(action, self._data())

    def _runtime(self):
        self.status.setText(self.controller.runtime_message())
        self.status.show()
        self.scroll.ensureWidgetVisible(self.status)

    def _started(self, payload):
        self.busy = True
        self._started_at = time.monotonic()
        self._active_record = payload["task_id"]
        self._task_log_path = None
        root = self.model.get("root")
        if root:
            folder = Path(root) / "_logs/plugin_ui"
            try:
                folder.mkdir(parents=True, exist_ok=True)
                self._task_log_path = folder / f"{payload['task_id']}.log"
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
        task_id = payload.get("task_id") or self._active_record
        if task_id not in self._records:
            self._records[task_id] = {"lines": [], "error": ""}
            title = "当前会话" if task_id == "session" else "任务 · " + datetime.now().strftime("%m-%d %H:%M:%S")
            self.recent.addItem(title, task_id)
            self.recent.setCurrentIndex(self.recent.count() - 1)
        text = f"[{payload.get('level', 'INFO')}] {payload.get('message', '')}"
        record = self._records[task_id]
        record["lines"].append(text)
        del record["lines"][:-2000]
        if self.recent.currentData() == task_id:
            self.logs.appendPlainText(text)
        if self._task_log_path and task_id == self._active_record:
            try:
                with self._task_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(text + "\n")
            except OSError:
                self._task_log_path = None

    def _show_record(self, *args):
        record = self._records.get(self.recent.currentData(), {})
        self.logs.setPlainText("\n".join(record.get("lines", [])))
        self.error_detail.setPlainText(record.get("error", ""))

    def _failed(self, payload):
        self.busy = False
        self.timer.stop()
        self.state_text.setText("失败")
        self._failure_reason = payload.get("message", "任务失败").splitlines()[0]
        self.status.setText("任务未完成，请查看运行记录中的错误详情")
        self.progress.setRange(0, 1000)
        self._log({**payload, "level": "ERROR"})
        task_id = payload.get("task_id") or self._active_record
        self._records[task_id]["error"] = payload["message"] + "\n" + payload.get("detail", "")
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
        self.status.setText("处理完成，可直接打开下方成果" if finished else "任务已取消")
        self._log({**payload, "message": self.state_text.text()})
        self._refresh_results(latest=True)
        self._update_controls()

    def _reset_results(self):
        self._result_keys.clear()
        if hasattr(self, "chooser"):
            self.chooser.hide()
        self.results.hide()
        self.result_summary.show()
        self._group_payloads = {name: [] for name in self.group_buttons}
        self._result_captions = {name: [] for name in self.group_buttons}
        for name, button in self.group_buttons.items():
            button.setEnabled(False)
            self.result_labels[name].setText("暂无成果")
        for menu in self.result_menus.values():
            menu.deleteLater()
        self.result_menus.clear()

    def _refresh_results(self, latest=False):
        root = self.model.get("root")
        if not root:
            return
        tasks = discover_tasks(Path(root), self.output.text())
        self._resume_task = next((task for task in tasks if task["status"] in {"running", "failed", "cancelled", "completed_with_errors"}
                                  and task["data"].get("execution_profile") == "fast"), None)
        self._set_tasks(tasks)
        if latest and tasks:
            self.task.setCurrentIndex(0)
        self._reset_results()
        task = self.task.currentData()
        if task:
            try:
                for payload in read_results(Path(task["path"])):
                    self._result({"plugin_id": "road_change", "task_id": task["id"], **payload})
            except (OSError, ValueError, TypeError) as exc:
                self.result_summary.setText(f"成果读取失败：{exc}")
                return
        self.result_summary.setText(f"{len(self._result_keys)} 项可用成果" if self._result_keys else "暂无成果，运行完成后在此查看")

    def _result(self, payload):
        kind = payload.get("result_type")
        if kind not in GROUPS:
            return
        key = (kind, payload["path"])
        if key in self._result_keys:
            return
        self._result_keys.add(key)
        self.results.show()
        self.result_summary.hide()
        group = GROUPS[kind]
        self._group_payloads[group].append(dict(payload))
        self.group_buttons[group].setEnabled(True)
        meta = payload.get("metadata", {})
        scope = str(meta.get("period") or "")
        if meta.get("before_period") and meta.get("after_period"):
            scope = f"{meta['before_period']} → {meta['after_period']}"
        leaf = str(payload.get("name", "")).split(" / ")[-1]
        title = DETAIL_LABELS.get(leaf, RESULT_LABELS[kind])
        context = " / ".join(str(v) for v in (meta.get("grid"), scope) if v)
        caption = title + (f" · {context}" if context else "")
        self._result_captions[group].append((caption, dict(payload)))
        previous = self.result_menus.get(group)
        if previous:
            previous.deleteLater()
        self.result_menus[group] = result_menu(self.group_buttons[group], group,
                                               self._result_captions[group], self._open_result)
        members = self._group_payloads[group]
        if group == "单期道路":
            scopes = {(p.get("metadata", {}).get("grid"), p.get("metadata", {}).get("period"))
                      for p in members if p.get("metadata", {}).get("period")}
            self.result_labels[group].setText(f"{len(scopes)} 个期次" if scopes else "已生成")
        elif group == "变化检测":
            scopes = {(p.get("metadata", {}).get("grid"), p.get("metadata", {}).get("before_period"),
                       p.get("metadata", {}).get("after_period")) for p in members
                      if p.get("metadata", {}).get("before_period") and p.get("metadata", {}).get("after_period")}
            self.result_labels[group].setText(f"{len(scopes)} 个变化对" if scopes else "已生成")
        elif group == "长时序":
            self.result_labels[group].setText("已生成")
        elif Path(payload["path"]).suffix.lower() == ".json":
            self.metrics.setText(metrics_text(Path(payload["path"])))
        elif self.metrics.text() == "暂无成果":
            self.metrics.setText("报告已生成")
        self.result_summary.setText(f"{len(self._result_keys)} 个成果文件可用")

    def _open_group(self, group):
        self.chooser.populate(group, self._result_captions[group])
        self.chooser.show()
        self.scroll.ensureWidgetVisible(self.chooser)

    def _open_result(self, payload):
        if not Path(payload["path"]).is_file():
            self.status.setText("成果文件已不存在，请重新扫描项目")
            self.status.show()
            self.scroll.ensureWidgetVisible(self.status)
            return
        self.controller.result_ready.emit(dict(payload))
        self.status.setText("已将成果路径报告给宿主")
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
