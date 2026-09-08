"""Embeddable UI; inherits all palette, font and style choices from its host."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QStackedWidget,
    QScrollArea, QLabel, QPushButton, QLineEdit, QCheckBox, QProgressBar,
    QPlainTextEdit, QTreeWidget, QTreeWidgetItem, QFileDialog)
from .ui.forms import PathField, Rows, Fold, form_layout
from .result_parser import RESULT_TYPES


class RoadChangeWidget(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.navigation = QComboBox()
        self.navigation.addItems(["数据准备", "运行处理", "成果与评价", "运行记录"])
        layout.addWidget(self.navigation)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.navigation.currentIndexChanged.connect(self.pages.setCurrentIndex)
        self._data_page()
        self._run_page()
        self._results_page()
        self._history_page()
        self.status = QLabel("就绪")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        controller.task_started.connect(self._started)
        controller.task_progress.connect(self._progress)
        controller.task_log.connect(self._log)
        controller.task_failed.connect(self._failed)
        controller.task_finished.connect(self._finished)
        controller.result_ready.connect(self._result)
        for entry in controller.history:
            self._log(entry)

    def _page(self):
        body = QWidget()
        form = form_layout(body)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        self.pages.addWidget(scroll)
        return form

    def _button(self, form, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        form.addRow(button)
        return button

    def _data_page(self):
        form = self._page()
        note = QLabel("添加区域与期次；同一区域至少两期。影像 TXT 每行一个路径。表格可横向滚动。")
        note.setWordWrap(True)
        form.addRow(note)
        self.areas = Rows(["区域名称", "验证区 SHP"], "Shapefile (*.shp)")
        self.periods = Rows(["区域名称", "期次", "影像 TXT"], "影像清单 (*.txt)")
        self.truths = Rows(["区域名称", "前期", "后期", "真值 SHP"], "Shapefile (*.shp)")
        form.addRow("验证区域", self.areas)
        form.addRow("多期影像", self.periods)
        gt = Fold("可选 GT 与评价")
        gt.form.addRow(self.truths)
        self.evaluate = QCheckBox("启用精度评价（标准模式需完整相邻期真值）")
        gt.form.addRow(self.evaluate)
        self.truth_field = QLineEdit()
        gt.form.addRow("变化类型字段（可选）", self.truth_field)
        form.addRow(gt)
        self.output = PathField(directory=True)
        form.addRow("成果输出目录", self.output)

    def _run_page(self):
        form = self._page()
        self.profile = QComboBox()
        self.profile.addItem("标准", "full")
        self.profile.addItem("Fast", "fast")
        form.addRow("处理模式", self.profile)
        self.run_id = QLineEdit()
        form.addRow("任务名称（留空自动生成）", self.run_id)
        self.resume = QCheckBox("继续同名未完成任务")
        form.addRow(self.resume)
        self._button(form, "运行完整流程", lambda: self._run("all"))
        self.local = Fold("局部运行与重跑")
        self.manifest = PathField()
        self.local.form.addRow("正式 pipeline_result.json", self.manifest)
        self.grid = QLineEdit()
        self.period = QLineEdit()
        self.before = QLineEdit()
        self.after = QLineEdit()
        for label, field in (("区域名称", self.grid), ("重跑期次", self.period)):
            self.local.form.addRow(label, field)
        self._button(self.local.form, "重跑某期并更新相关结果", lambda: self._run("rerun-period"))
        self.local.form.addRow("变化对前期", self.before)
        self.local.form.addRow("变化对后期", self.after)
        self._button(self.local.form, "重跑某变化对", lambda: self._run("rerun-change"))
        form.addRow(self.local)
        self.advanced = Fold("高级设置")
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.advanced.form.addRow("设备", self.device)
        self.parameters = {}
        for key, label, default in (("pixel-size", "像元大小（0 自动）", "0.0"), ("absolute", "宽度变化绝对阈值", "2.0"), ("ratio", "宽度变化比例阈值", "0.2"), ("tolerance", "匹配容差", "3.0")):
            field = QLineEdit(default)
            self.parameters[key] = field
            self.advanced.form.addRow(label, field)
        form.addRow(self.advanced)
        self._button(form, "检查环境与模型位置", self._runtime)
        self._button(form, "取消当前任务", self.controller.cancel)

    def _results_page(self):
        form = self._page()
        self.result_manifest = PathField()
        form.addRow("正式 pipeline_result.json", self.result_manifest)
        self._button(form, "读取并报告成果路径", lambda: self.controller.load_results(self.result_manifest.text()))
        self.results = QTreeWidget()
        self.results.setHeaderLabels(["成果", "路径"])
        self.results.setMinimumWidth(0)
        self.groups = {}
        self.result_keys = set()
        for kind in RESULT_TYPES:
            item = QTreeWidgetItem([kind])
            self.results.addTopLevelItem(item)
            self.groups[kind] = item
        form.addRow(self.results)
        note = QLabel("成果以 result_ready 信号报告，宿主决定是否加载地图。评价使用数据准备页的可选真值。")
        note.setWordWrap(True)
        form.addRow(note)
        self._button(form, "评价已有成果", lambda: self._run("evaluate-all-existing"))

    def _history_page(self):
        form = self._page()
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(2000)
        form.addRow(self.logs)
        self._button(form, "导出运行记录", self._export_logs)

    def _data(self):
        return dict(areas=self.areas.values(), periods=self.periods.values(), truths=self.truths.values(),
                    output=self.output.text(), profile=self.profile.currentData(), evaluate=self.evaluate.isChecked(),
                    truth_type_field=self.truth_field.text(), run_id=self.run_id.text(), resume=self.resume.isChecked(),
                    manifest=self.manifest.text(), grid=self.grid.text(), period=self.period.text(),
                    before_period=self.before.text(), after_period=self.after.text(), device=self.device.currentText(),
                    **{key: field.text() for key, field in self.parameters.items()})

    def _run(self, action):
        data = self._data()
        if action == "evaluate-all-existing":
            data["manifest"] = self.result_manifest.text()
        self.controller.run(action, data)

    def _runtime(self):
        self.status.setText(self.controller.runtime_message())

    def _started(self, payload):
        self.status.setText("正在运行 · " + payload["task_id"])
        self.progress.setValue(0)

    def _progress(self, payload):
        self.progress.setValue(round(payload["progress"] * 1000))
        self.status.setText(payload["stage"] + " " + payload["message"])

    def _log(self, payload):
        self.logs.appendPlainText(f"[{payload.get('level', 'INFO')}] {payload.get('message', '')}")

    def _failed(self, payload):
        self.status.setText(payload["message"])
        self._log({"level": "ERROR", "message": payload["message"]})

    def _finished(self, payload):
        self.status.setText("已取消" if payload["status"] == "cancelled" else "处理完成")
        if payload["status"] == "completed":
            self.progress.setValue(1000)

    def _result(self, payload):
        key = (payload["result_type"], payload["path"])
        if key in self.result_keys:
            return
        self.result_keys.add(key)
        item = QTreeWidgetItem([payload["name"], payload["path"]])
        item.setToolTip(1, payload["path"])
        self.groups[payload["result_type"]].addChild(item)

    def _export_logs(self):
        path = QFileDialog.getSaveFileName(self, "导出记录", "run.log", "日志 (*.log)")[0]
        if path:
            try:
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write(self.logs.toPlainText())
            except OSError as exc:
                self.status.setText(str(exc))
