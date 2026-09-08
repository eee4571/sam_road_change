"""Compact processing dock; only UI input selection and presentation live here."""
import copy
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThreadPool, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
    QComboBox, QScrollArea, QLabel, QPushButton, QLineEdit,
    QCheckBox, QProgressBar, QPlainTextEdit, QTreeWidget, QTreeWidgetItem,
    QStyle, QHeaderView)
from .ui.forms import PathField, Rows, Fold, form_layout
from .ui.project_browser import ROOT, scan_project, check_files, pairs, discover_tasks, natural, resolve
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


def label(text):
    value = QLabel(text)
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
        self.setObjectName("roadChangeDock")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        body = QWidget()
        form = form_layout(body)
        form.setContentsMargins(0, 0, 4, 0)
        form.setSpacing(6)
        self.scroll.setWidget(body)
        layout.addWidget(self.scroll)
        self._data_section(form)
        self._processing_section(form)
        self._results_section(form)
        self._history_section()
        for fold in (self.corrections, self.advanced, self.local, self.records_fold):
            form.addRow(fold)
        self.status = label("选择项目目录，开始扫描数据")
        form.addRow(self.status)
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

    def _button(self, text, callback, icon=None):
        button = QPushButton(text)
        if icon is not None:
            button.setIcon(self.style().standardIcon(icon))
        button.clicked.connect(callback)
        return button

    def _data_section(self, form):
        self.project = PathField(directory=True)
        self.project.edit.setPlaceholderText("选择项目 / 数据目录")
        self.project.edit.textChanged.connect(self._directory_changed)
        form.addRow("项目 / 数据目录", self.project)
        self.scan_button = self._button("扫描数据", self.scan, QStyle.StandardPixmap.SP_BrowserReload)
        form.addRow(self.scan_button)
        self.summary = label("尚未扫描")
        form.addRow(self.summary)
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderLabels(["区域 / 期次", "识别结果"])
        self.data_tree.setMinimumWidth(0)
        self.data_tree.setMinimumHeight(190)
        self.data_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.data_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.corrections = Fold("数据详情与修正")
        self.corrections.form.addRow(self.data_tree)
        self.check_details = label("暂无检查详情")
        self.corrections.form.addRow(self.check_details)
        self.check_note = label("扫描后检查数据；无需手动填写区域与期次。")
        form.addRow(self.check_note)
        self.check_button = self._button("检查数据", self.check_data, QStyle.StandardPixmap.SP_DialogApplyButton)
        form.addRow(self.check_button)
        self.areas = Rows(["区域", "验证区 SHP"], "Shapefile (*.shp)")
        self.periods = Rows(["区域", "期次", "影像 TXT"], "影像清单 (*.txt)")
        self.truths = Rows(["区域", "前期", "后期", "GT SHP"], "Shapefile (*.shp)")
        for title, rows in (("验证区", self.areas), ("影像期次", self.periods), ("可选 GT", self.truths)):
            self.corrections.form.addRow(title, rows)
            rows.table.itemChanged.connect(self._edited)
            rows.table.model().rowsRemoved.connect(self._edited)
        self.corrections.form.addRow(self._button("应用修正", self._apply_corrections))
        self.corrections.form.addRow(label("目录结构：01_验证区 / 02_影像 / 03_变化真值（可选）。也支持 project_config.json。"))

    def _processing_section(self, form):
        form.addRow(label("处理设置"))
        self.profile = QComboBox()
        self.profile.addItem("标准", "full")
        self.profile.addItem("Fast", "fast")
        self.profile.currentIndexChanged.connect(self._update_controls)
        form.addRow("处理模式", self.profile)
        self.run_button = self._button("运行完整流程", lambda: self._run("all"), QStyle.StandardPixmap.SP_MediaPlay)
        self.run_button.setProperty("role", "primary")
        self.run_button.setDefault(True)
        form.addRow(self.run_button)
        form.addRow(label("当前任务"))
        self.state_text = label("待就绪")
        self.area_text = label("—")
        self.scope_text = label("—")
        self.stage_text = label("—")
        self.elapsed_text = label("00:00:00")
        state = QWidget()
        state_form = form_layout(state)
        state_form.setContentsMargins(0, 4, 0, 4)
        state_form.setRowWrapPolicy(state_form.RowWrapPolicy.WrapLongRows)
        for title, value in (("状态", self.state_text), ("当前区域", self.area_text),
                             ("期次 / 变化对", self.scope_text), ("当前阶段", self.stage_text), ("已用时间", self.elapsed_text)):
            state_form.addRow(title, value)
        form.addRow(state)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        form.addRow("总体进度", self.progress)
        self.progress_note = label("等待任务开始")
        form.addRow(self.progress_note)
        self.cancel_button = self._button("取消任务", self.controller.cancel)
        form.addRow(self.cancel_button)
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
        self.advanced.form.addRow("成果目录", self.output)
        self.evaluate = QCheckBox("使用可用 GT 评价成果")
        self.evaluate.toggled.connect(self._update_controls)
        self.advanced.form.addRow(self.evaluate)
        self.truth_field = QLineEdit()
        self.advanced.form.addRow("GT 类型字段（可选）", self.truth_field)
        self.run_id = QLineEdit()
        self.run_id.setPlaceholderText("留空自动命名")
        self.advanced.form.addRow("任务名称", self.run_id)
        self.resume = QCheckBox("继续所选的未完成任务")
        self.advanced.form.addRow(self.resume)
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.advanced.form.addRow("设备", self.device)
        self.parameters = {}
        for key, title, value in (("pixel-size", "像元大小（0 自动）", "0.0"), ("absolute", "宽度变化阈值", "2.0"),
                                  ("ratio", "宽度变化比例", "0.2"), ("tolerance", "匹配容差", "3.0")):
            field = QLineEdit(value)
            self.parameters[key] = field
            self.advanced.form.addRow(title, field)
        self.advanced.form.addRow(self._button("检查运行资源", self._runtime))

    def _results_section(self, form):
        form.addRow(label("成果与评价"))
        self.result_summary = label("扫描项目后自动发现已有成果")
        form.addRow(self.result_summary)
        self.results = QTreeWidget()
        self.results.setHeaderLabels(["成果", "操作"])
        self.results.setMinimumWidth(0)
        self.results.setMinimumHeight(140)
        self.results.setMaximumHeight(144)
        self.results.header().setStretchLastSection(False)
        self.results.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        form.addRow(self.results)
        self._reset_results()
        form.addRow(label("点击“打开”将成果路径交给宿主使用。"))
        self.metrics = label("Precision —   Recall —   F1 —")
        form.addRow(self.metrics)
        self.evaluate_button = self._button("评价已有成果", lambda: self._run("evaluate-all-existing"))
        self.advanced.form.addRow(self.evaluate_button)

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
        if hasattr(self, "run_button"):
            self._update_controls()
            self.check_note.setText("目录已改变，请重新扫描")

    def _invalidate_check(self):
        self.checked = False
        if hasattr(self, "status"):
            self._update_controls()

    def _edited(self, *args):
        self._invalidate_check()
        self.check_note.setText("修正尚未应用，请点击“应用修正”后重新检查")

    def _background(self, function, args, callback):
        self.browsing = True
        self._update_controls()
        job = BrowseJob(function, *args)
        self._job = job
        revision = self._project_revision

        def done(value):
            self.browsing = False
            if revision == self._project_revision:
                callback(value)
            self._update_controls()

        def failed(message):
            self.browsing = False
            self.status.setText(message)
            self.check_note.setText(message)
            self._update_controls()

        job.signals.completed.connect(done)
        job.signals.failed.connect(failed)
        QThreadPool.globalInstance().start(job)

    def scan(self):
        if not self.project.text():
            self.status.setText("请先选择项目 / 数据目录")
            return
        self.checked = False
        self.status.setText("正在识别项目文件…")
        self._background(scan_project, (self.project.text(),), self._scanned)

    def _scanned(self, model):
        self.model = model
        self.output.edit.setText(model["output"])
        for rows, key in ((self.areas, "areas"), (self.periods, "periods"), (self.truths, "truths")):
            rows.set_values(model[key])
        self._draw_catalog()
        self._set_tasks(model["tasks"])
        self.evaluate.setChecked(bool(model["truths"]) and len(model["truths"]) == len(pairs(model["periods"])))
        self.check_note.setText("\n".join(model["issues"]) or "识别完成，请检查数据文件")
        self.status.setText("扫描完成")
        self._refresh_results()

    def _apply_corrections(self):
        self.model.update(areas=self.areas.values(), periods=self.periods.values(), truths=self.truths.values(), issues=[])
        for key in ("areas", "periods", "truths"):
            for row in self.model[key]:
                if row[-1]:
                    row[-1] = str(resolve(row[-1], Path(self.model.get("root", ROOT))))
        self.checked = False
        self._draw_catalog()
        self._task_changed()
        self.check_note.setText("修正已应用，请重新检查数据")
        self._update_controls()

    def _draw_catalog(self):
        self.data_tree.clear()
        change_pairs = pairs(self.model["periods"])
        gt = {tuple(r[:3]) for r in self.model["truths"] if Path(r[-1]).is_file()}
        for area, path in self.model["areas"]:
            node = QTreeWidgetItem([area, "验证区" if Path(path).is_file() else "缺少 SHP"])
            node.setToolTip(0, path)
            self.data_tree.addTopLevelItem(node)
            for _, period, source in sorted((r for r in self.model["periods"] if r[0] == area), key=lambda r: natural(r[1])):
                child = QTreeWidgetItem([period, "影像清单" if Path(source).is_file() else "缺少 TXT"])
                child.setToolTip(0, source)
                node.addChild(child)
            for a, before, after in change_pairs:
                if a == area:
                    node.addChild(QTreeWidgetItem([f"{before} → {after}", "有 GT" if (a, before, after) in gt else "无 GT（可选）"]))
            node.setExpanded(True)
        text = f"{len(self.model['areas'])} 个区域 · {len(self.model['periods'])} 个期次\n{len(change_pairs)} 个变化对 · {len(gt)} 个 GT"
        self.summary.setText(text)

    def check_data(self):
        self.status.setText("正在检查输入文件…")
        self._background(check_files, (copy.deepcopy(self.model),), self._checked)

    def _checked(self, issues):
        issues = self.model.get("issues", []) + issues
        try:
            self.controller.build_command("all", self._data())
        except (ValueError, OSError, TypeError) as exc:
            issues.append(str(exc))
        self.checked = not issues
        self.check_details.setText("\n".join(issues) if issues else "输入文件与清单检查通过。空间参考与影像内容在正式运行前检查。")
        self.check_note.setText(f"检查失败 · {len(issues)} 项待修正" if issues else "数据已就绪")
        self.status.setText("检查失败，请展开数据详情修正" if issues else "数据已就绪，可以运行")
        self.state_text.setText("待修正" if issues else "数据已就绪")
        self._update_controls()

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
        if hasattr(self, "evaluate_button"):
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
        return dict(areas=self.model["areas"], periods=self.model["periods"], truths=self.model["truths"],
                    output=self.output.text(), profile=self.profile.currentData(), evaluate=self.evaluate.isChecked(),
                    truth_type_field=self.truth_field.text(), run_id=run_id, resume=self.resume.isChecked(),
                    manifest=task["path"] if task else "", grid=self.grid.currentText(), period=self.period.currentText(),
                    before_period=pair[0], after_period=pair[1], device=self.device.currentText(),
                    **{key: field.text() for key, field in self.parameters.items()})

    def _update_controls(self, *args):
        if not hasattr(self, "evaluate_button"):
            return
        available = not self.busy and not self.browsing
        same_root = bool(self.model.get("root")) and self.controller.path(self.project.text()) == Path(self.model["root"])
        has_data = bool(self.model["areas"]) and same_root
        gt_complete = len(self.model["truths"]) == len(pairs(self.model["periods"]))
        evaluation_ok = not self.evaluate.isChecked() or self.profile.currentData() == "fast" or gt_complete
        self.run_button.setEnabled(available and self.checked and evaluation_ok)
        self.scan_button.setEnabled(available)
        self.check_button.setEnabled(available and has_data)
        self.project.setEnabled(available)
        self.corrections.setEnabled(available)
        self.advanced.setEnabled(available)
        self.profile.setEnabled(available)
        self.cancel_button.setEnabled(self.busy)
        has_task = bool(self.task.currentData()) and same_root
        self.rerun_period.setEnabled(available and has_task and self.period.count() > 0)
        self.rerun_pair.setEnabled(available and has_task and self.pair.count() > 0)
        self.evaluate_button.setEnabled(available and has_task)
        summary = f"{len(self.model['areas'])} 个区域 · {len(self.model['periods'])} 个期次\n{len(pairs(self.model['periods']))} 个变化对 · {len(self.model['truths'])} 个 GT"
        self.summary.setText(summary + (" · 已就绪" if self.checked else " · 待检查"))
        self.check_note.setVisible(not self.checked)
        if not evaluation_ok and not self.busy:
            self.status.setText("标准模式评价需要每个变化对的 GT；请修正或在高级设置关闭评价")

    def _run(self, action):
        self._action = action
        self.busy = True
        self.state_text.setText("正在启动")
        self._update_controls()
        self.controller.run(action, self._data())

    def _runtime(self):
        self.status.setText(self.controller.runtime_message())

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
        self.timer.start()
        self._log({"task_id": self._active_record, "message": "任务开始", "level": "INFO"})
        self._update_controls()

    def _tick(self):
        seconds = int(time.monotonic() - self._started_at) if self._started_at is not None else 0
        self.elapsed_text.setText(f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}")

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
        self.results.clear()
        self._result_keys.clear()
        nodes = {}
        self._group_payloads = {}
        self.group_buttons = {}
        for name in ("单期道路", "变化检测", "长时序", "精度评价"):
            nodes[name] = QTreeWidgetItem([name, ""])
            self.results.addTopLevelItem(nodes[name])
            self._group_payloads[name] = []
            button = self._button("打开", lambda _checked=False, group=name: self._open_group(group))
            button.setEnabled(False)
            button.setToolTip("报告本类全部成果路径；展开分类可单独打开")
            self.results.setItemWidget(nodes[name], 1, button)
            self.group_buttons[name] = button
        self.groups = {kind: nodes[group] for kind, group in GROUPS.items()}
        if hasattr(self, "metrics"):
            self.metrics.setText("Precision —   Recall —   F1 —")

    def _refresh_results(self, latest=False):
        root = self.model.get("root")
        if not root:
            return
        tasks = discover_tasks(Path(root), self.output.text())
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
        self.result_summary.setText(f"{len(self._result_keys)} 项可用成果" + (f" · {task['id']}" if task else " · 尚无已完成任务"))

    def _result(self, payload):
        kind = payload.get("result_type")
        if kind not in self.groups:
            return
        key = (kind, payload["path"])
        if key in self._result_keys:
            return
        self._result_keys.add(key)
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
        item = QTreeWidgetItem([title + (f" · {context}" if context else ""), ""])
        item.setToolTip(0, payload["path"])
        self.groups[kind].addChild(item)
        self.groups[kind].setText(0, f"{group}（{len(self._group_payloads[group])}）")
        button = self._button("打开", lambda: self._open_result(payload))
        self.results.setItemWidget(item, 1, button)
        self.result_summary.setText(f"{len(self._result_keys)} 项可用成果 · 展开分类可单独打开")
        if kind == "road_evaluation" and Path(payload["path"]).suffix.lower() == ".json":
            self.metrics.setText(metrics_text(Path(payload["path"])))

    def _open_group(self, group):
        for payload in list(self._group_payloads[group]):
            self._open_result(payload)

    def _open_result(self, payload):
        if not Path(payload["path"]).is_file():
            self.status.setText("成果文件已不存在，请重新扫描项目")
            return
        self.controller.result_ready.emit(dict(payload))
        self.status.setText("已将成果路径报告给宿主")

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
