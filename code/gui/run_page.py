from __future__ import annotations
import re
import time
from pathlib import Path
from tkinter import messagebox

from tkinter import BOTH, LEFT, RIGHT, X, StringVar
from tkinter import ttk

from .common_widgets import LAYOUT_METRICS

class RunPage:
    def _build_run_page(self, page: ttk.Frame) -> None:
        self.run_body = page
        run_card = ttk.LabelFrame(page, text="运行任务", padding=LAYOUT_METRICS["card_padding"])
        run_card.pack(fill=X)
        self.preflight_summary = StringVar(value="开始处理时会自动检查以下内容；发现问题将立即停止并说明原因。")
        ttk.Label(run_card, textvariable=self.preflight_summary, wraplength=620).pack(anchor="w", fill=X, pady=(0, 5))
        checklist = ttk.Frame(run_card)
        checklist.pack(fill=X, pady=(0, 6))
        self.preflight_check_labels = []
        for row_index, label in enumerate(("项目结构与验证区", "影像期次与覆盖范围", "模型、参数与运行环境", "输出位置与磁盘空间")):
            ttk.Label(checklist, text=label).grid(row=row_index, column=0, sticky="w", pady=1)
            state_label = ttk.Label(checklist, text="开始前检查", style="CardMuted.TLabel")
            state_label.grid(row=row_index, column=1, sticky="e", pady=1)
            self.preflight_check_labels.append(state_label)
        checklist.grid_columnconfigure(0, weight=1)
        actions = ttk.Frame(run_card)
        actions.pack(fill=X)
        self.preflight_button = ttk.Button(actions, text="检查数据", command=self.preflight_inputs)
        self.preflight_button.pack(side=LEFT)
        self.run_button = ttk.Button(
            actions, text="运行完整流程", style="Hero.TButton", command=self.run_all,
        )
        self.run_button.pack(side=RIGHT)
        self.cancel_button = ttk.Button(actions, text="取消任务", command=self.cancel_task)
        self.cancel_button.pack(side=RIGHT, padx=5)
        self.cancel_button.state(["disabled"])

        progress_card = ttk.LabelFrame(page, text="运行进度", padding=LAYOUT_METRICS["card_padding"])
        progress_card.pack(fill=X, pady=(LAYOUT_METRICS["section_gap"], 0))
        ttk.Label(progress_card, text="当前阶段：").pack(anchor="w")
        self.run_status = StringVar(value="等待开始任务。")
        ttk.Label(progress_card, textvariable=self.run_status, wraplength=620).pack(fill=X, pady=(2, 5))
        ttk.Label(progress_card, text="总体进度").pack(anchor="w", pady=(0, 3))
        self.progress = ttk.Progressbar(progress_card, mode="determinate", maximum=1, value=0, style="Modern.Horizontal.TProgressbar")
        self.progress.pack(fill=X)
        self.progress_text = StringVar(value="0 / 0 · 已用时 00:00:00 · 剩余 --")
        ttk.Label(progress_card, textvariable=self.progress_text).pack(anchor="w", pady=(5, 0))

        stage_card = ttk.LabelFrame(page, text="局部重跑", padding=LAYOUT_METRICS["card_padding"])
        stage_card.pack(fill=X, pady=(LAYOUT_METRICS["section_gap"], 0))
        selectors = ttk.Frame(stage_card)
        selectors.pack(fill=X, pady=(0, 5))
        ttk.Label(selectors, text="区域：").grid(row=0, column=0, sticky="w")
        self.stage_region_combo = ttk.Combobox(selectors, textvariable=self.stage_region, state="readonly", width=14)
        self.stage_region_combo.grid(row=0, column=1, sticky="ew", padx=(3, 7))
        self.stage_region_combo.bind("<<ComboboxSelected>>", self._stage_region_changed)
        ttk.Label(selectors, text="期次：").grid(row=0, column=2, sticky="w")
        self.stage_period_combo = ttk.Combobox(selectors, textvariable=self.project_period, state="readonly", width=12)
        self.stage_period_combo.grid(row=0, column=3, sticky="ew", padx=(3, 7))
        self.stage_period_combo.bind("<<ComboboxSelected>>", self._stage_period_changed)
        ttk.Label(selectors, text="相邻变化对：").grid(row=0, column=4, sticky="w")
        self.stage_pair_combo = ttk.Combobox(selectors, textvariable=self.project_change_pair, state="readonly", width=16)
        self.stage_pair_combo.grid(row=0, column=5, sticky="ew", padx=(3, 0))
        for column in (1, 3, 5):
            selectors.grid_columnconfigure(column, weight=1)
        self.affected_pairs_summary = StringVar(value="请选择期次以查看受影响的相邻变化对。")
        ttk.Label(stage_card, textvariable=self.affected_pairs_summary, wraplength=620).pack(anchor="w", fill=X, pady=(0, 5))
        stage_actions = ttk.Frame(stage_card)
        stage_actions.pack(fill=X)
        road_actions = ttk.LabelFrame(stage_actions, text="道路提取", padding=(7, 5))
        road_actions.pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        change_actions = ttk.LabelFrame(stage_actions, text="变化检测", padding=(7, 5))
        change_actions.pack(side=LEFT, fill=X, expand=True, padx=(3, 0))
        self.stage_buttons = []
        for parent, text, command in (
            (road_actions, "重跑该期", lambda: self.rerun_selected_period(False)),
            (road_actions, "重跑并更新相关结果", lambda: self.rerun_selected_period(True)),
            (change_actions, "重跑该变化对", lambda: self.rerun_selected_change(False)),
            (change_actions, "重跑并更新长时序成果", lambda: self.rerun_selected_change(True)),
        ):
            button = ttk.Button(parent, text=text, command=command)
            button.pack(side=LEFT, padx=(0, 4))
            self.stage_buttons.append(button)
        advanced_stage = ttk.Frame(stage_card)
        advanced_stage.pack(fill=X, pady=(5, 0))
        ttk.Label(advanced_stage, text="高级操作：").pack(side=LEFT)
        batch_button = ttk.Button(advanced_stage, text="批量重跑全部道路提取", command=self.run_extract_all)
        batch_button.pack(side=LEFT)
        self.stage_buttons.append(batch_button)

        summary_card = ttk.LabelFrame(self.step_summaries[1], text="任务摘要", padding=LAYOUT_METRICS["card_padding"])
        summary_card.pack(fill=BOTH, expand=True)
        self.run_destination_summary = StringVar(value=self.vars["output_root"].get())
        for row, (label, variable) in enumerate((
            ("输入数据：", self.input_summary),
            ("成果位置：", self.run_destination_summary),
            ("当前任务：", self.vars["run_id"]),
            ("任务状态：", self.run_status),
        )):
            ttk.Label(summary_card, text=label, width=12).grid(row=row, column=0, sticky="nw", pady=2)
            ttk.Label(summary_card, textvariable=variable, wraplength=400).grid(row=row, column=1, sticky="nw", pady=2)
        summary_card.grid_columnconfigure(1, weight=1)
        run_settings_shell = ttk.Frame(summary_card)
        run_settings_shell.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.run_settings_toggle = ttk.Button(run_settings_shell, text="输出位置与高级设置...", command=self._toggle_run_settings)
        self.run_settings_toggle.pack(anchor="w")
        self.run_settings_frame = ttk.Frame(run_settings_shell)
        self._field(self.run_settings_frame, "成果输出目录", "output_root", "dir")
        self._field(self.run_settings_frame, "手工任务名称", "run_id")
        run_options = ttk.Frame(self.run_settings_frame)
        run_options.pack(fill=X, pady=(3, 0))
        ttk.Checkbutton(
            run_options, text="单项失败后继续", variable=self.vars["continue_on_error"], onvalue="1", offvalue="0",
        ).pack(side=LEFT)
        ttk.Button(
            run_options, text="检查软件环境", style="Compact.TButton", command=lambda: self._command(["doctor"]),
        ).pack(side=RIGHT)
        self.advanced_toggle = ttk.Button(
            self.run_settings_frame, text="高级参数...", command=self._toggle_advanced,
        )
        self.advanced_toggle.pack(anchor="w", pady=(10, 2))
        self.advanced_frame = ttk.Frame(self.run_settings_frame)
        self._field(self.advanced_frame, "道路模型", "checkpoint", "file")
        self._field(self.advanced_frame, "推理配置", "config", "file")
        advanced_row = ttk.Frame(self.advanced_frame)
        advanced_row.pack(fill=X, pady=(4, 2))
        ttk.Label(advanced_row, text="计算设备", width=18).pack(side=LEFT)
        ttk.Combobox(
            advanced_row, textvariable=self.vars["device"], values=("auto", "cuda", "cpu"),
            state="readonly", width=10,
        ).pack(side=LEFT)
        ttk.Label(advanced_row, text="像元大小", width=10).pack(side=LEFT, padx=(18, 0))
        ttk.Entry(advanced_row, textvariable=self.vars["pixel_size"], width=9).pack(side=LEFT)
        thresholds_row = ttk.Frame(self.advanced_frame)
        thresholds_row.pack(fill=X, pady=(2, 4))
        ttk.Label(thresholds_row, text="宽变绝对阈值", width=18).pack(side=LEFT)
        ttk.Entry(thresholds_row, textvariable=self.vars["absolute"], width=9).pack(side=LEFT)
        ttk.Label(thresholds_row, text="宽变比例", width=10).pack(side=LEFT, padx=(18, 0))
        ttk.Entry(thresholds_row, textvariable=self.vars["ratio"], width=9).pack(side=LEFT)
        ttk.Label(thresholds_row, text="位置容差", width=10).pack(side=LEFT, padx=(18, 0))
        ttk.Entry(thresholds_row, textvariable=self.vars["tolerance"], width=9).pack(side=LEFT)
        junction_row = ttk.Frame(self.advanced_frame)
        junction_row.pack(fill=X, pady=(2, 4))
        ttk.Label(junction_row, text="路口节点模式", width=18).pack(side=LEFT)
        ttk.Combobox(
            junction_row, textvariable=self.vars["junction_node_mode"],
            values=("sparse", "dense_legacy"), state="readonly", width=16,
        ).pack(side=LEFT)
        ttk.Label(
            junction_row, text="sparse=稀疏路口；dense_legacy=旧版密集路口",
            style="Hint.TLabel",
        ).pack(side=LEFT, padx=(12, 0))

        self._refresh_stage_selectors()

    def _build_current_command(self, *, preflight_only: bool = False, data_check_only: bool = False) -> list[str]:
        return self.task_manager.build_pipeline(
            mode=self.vars["mode"].get(), output_root=self.vars["output_root"].get(),
            checkpoint=self.vars["checkpoint"].get(), config=self.vars["config"].get(),
            device=self.vars["device"].get(), pixel_size=self.vars["pixel_size"].get(),
            rescale=self.vars["rescale"].get(), absolute=self.vars["absolute"].get(),
            ratio=self.vars["ratio"].get(), tolerance=self.vars["tolerance"].get(),
            run_id=self.vars["run_id"].get(), validation_area=self.vars["validation_area"].get(),
            periods=self._period_values(), truths=self._truth_values(),
            truth_type_field=self.vars["truth_type_field"].get(),
            source_root=self.vars["source_root"].get(),
            evaluate=False,
            resume=(self.vars["resume"].get() == "1" and not preflight_only),
            continue_on_error=self.vars["continue_on_error"].get() == "1",
            preflight_only=preflight_only,
            data_check_only=data_check_only,
            junction_node_mode=self.vars["junction_node_mode"].get(),
            validation_areas=(self.project_validation_areas or None),
            area_truths=(self.project_area_truths or None),
            area_periods=(self.project_area_periods or None),
        )

    def preflight_inputs(self) -> None:
        try:
            args = self._build_current_command(data_check_only=True)
        except ValueError as exc:
            messagebox.showerror("输入不完整", str(exc), parent=self.root)
            if not self.vars["output_root"].get().strip() or (
                self.vars["resume"].get() == "1" and not self.vars["run_id"].get().strip()
            ):
                if not self.run_settings_visible:
                    self._toggle_run_settings()
                self._scroll_to_module(self.run_body)
            else:
                self._show_manual_inputs()
                self._scroll_to_module(self.data_body)
            return
        self.preflight_summary.set("正在检查验证区、影像清单、期次顺序、CRS、波段、数据类型、覆盖和真值映射…")
        self.data_status.set("正在检查数据")
        self._command(args)

    def run_all(self) -> None:
        output = Path(self.vars["output_root"].get().strip()).expanduser()
        run_id, should_resume, state_path = self.task_manager.resolve_run(
            output, self.vars["run_id"].get(), self.project_config.get("active_task"),
        )
        self.vars["run_id"].set(run_id)
        self.vars["resume"].set("1" if should_resume else "0")
        self.project_config["active_task"] = {
            "run_id": run_id, "state": str(state_path), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_project_config()
        try:
            args = self._build_current_command()
        except ValueError as exc:
            messagebox.showerror("输入不完整", str(exc), parent=self.root)
            if not self.vars["output_root"].get().strip() or (
                self.vars["resume"].get() == "1" and not self.vars["run_id"].get().strip()
            ):
                if not self.run_settings_visible:
                    self._toggle_run_settings()
                self._scroll_to_module(self.run_body)
            else:
                self._show_manual_inputs()
                self._scroll_to_module(self.data_body)
            return
        order_text = self._period_order_confirmation()
        if order_text and not messagebox.askokcancel(
            "确认影像期次顺序",
            "程序将按以下顺序提取并检测相邻期次变化：\n\n" + order_text,
            parent=self.root,
        ):
            return
        self._show_step(1, force=True)
        self._command(args)

    def resume_all(self) -> None:
        """Compatibility alias; normal users always get automatic continuation."""
        self.run_all()

    def _project_stage_context(self) -> tuple[Path, str, str]:
        root = Path(self.project_root_path).expanduser() if self.project_root_path else None
        if root is None or not root.is_dir():
            raise ValueError("分步执行需要先在首页打开规范项目文件夹。")
        region = self._selected_project_region("stage")
        if not region:
            raise ValueError("请选择需要处理的区域。")
        run_id = self.vars["run_id"].get().strip()
        if not run_id:
            run_id = time.strftime("stage_%Y%m%d_%H%M%S")
            self.vars["run_id"].set(run_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("分步任务名称只能包含字母、数字、连字符和下划线。")
        return root.resolve(), region, run_id

    def _stage_period_changed(self, _event=None) -> None:
        region = self._selected_project_region("stage")
        periods = [period for period, _source in self.project_area_periods.get(region, [])]
        pairs = self.task_manager.affected_change_pairs(
            periods, self.project_period.get().strip(),
        )
        if pairs:
            self.affected_pairs_summary.set(
                "该操作会影响：" + "；".join(f"{before} → {after}" for before, after in pairs)
            )
        else:
            self.affected_pairs_summary.set("所选期次没有可识别的相邻变化对。")

    def _current_pipeline_manifest_path(self) -> Path:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None or not latest.is_file():
            raise ValueError("请先完成或载入一个完整任务，局部重跑需要已有 pipeline_result.json。")
        return latest.resolve()

    def rerun_selected_period(self, update_related: bool) -> None:
        try:
            manifest = self._current_pipeline_manifest_path()
            region = self._selected_project_region("stage")
            period = self.project_period.get().strip()
            if not region or not period:
                raise ValueError("请选择需要重跑的区域和影像期次。")
            args = self.task_manager.build_rerun_period(
                manifest, region, period, update_related,
            )
        except ValueError as exc:
            messagebox.showerror("无法局部重跑", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(args)

    def rerun_selected_change(self, update_temporal: bool) -> None:
        try:
            manifest = self._current_pipeline_manifest_path()
            region = self._selected_project_region("stage")
            pair = self.project_change_pair.get().strip()
            if not region or "→" not in pair:
                raise ValueError("请选择需要重跑的区域和相邻变化对。")
            before, after = (value.strip() for value in pair.split("→", 1))
            args = self.task_manager.build_rerun_change(
                manifest, region, before, after, update_temporal,
            )
        except ValueError as exc:
            messagebox.showerror("无法重跑变化对", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(args)

    def run_extract_all(self) -> None:
        try:
            manifest = self._current_pipeline_manifest_path()
        except ValueError as exc:
            messagebox.showerror("无法批量重跑", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(["rerun-all-periods", "--pipeline-manifest", str(manifest)])

    def run_extract_all_legacy(self) -> None:
        """Retained for callers of the historical standalone stage commands."""
        try:
            root, _region, run_id = self._project_stage_context()
            args = self.task_manager.build_extract_all(
                root, run_id, output_root=self.vars["output_root"].get(),
                device=self.vars["device"].get(),
                pixel_size=self.vars["pixel_size"].get(),
                rescale=self.vars["rescale"].get(),
                junction_node_mode=self.vars["junction_node_mode"].get(),
                continue_on_error=self.vars["continue_on_error"].get() == "1",
            )
        except ValueError as exc:
            messagebox.showerror("无法分步提取", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(args)

    def run_extract_selected(self) -> None:
        try:
            root, region, run_id = self._project_stage_context()
            period = self.project_period.get().strip()
            if not period:
                raise ValueError("请选择需要提取的影像期次。")
            args = self.task_manager.build_extract_period(
                root, region, period, run_id,
                output_root=self.vars["output_root"].get(),
                device=self.vars["device"].get(),
                pixel_size=self.vars["pixel_size"].get(),
                rescale=self.vars["rescale"].get(),
                junction_node_mode=self.vars["junction_node_mode"].get(),
            )
        except ValueError as exc:
            messagebox.showerror("无法分步提取", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(args)

    def run_change_selected(self) -> None:
        try:
            root, region, run_id = self._project_stage_context()
            pair = self.project_change_pair.get().strip()
            if "→" not in pair:
                raise ValueError("请选择需要检测的相邻变化对。")
            before, after = (value.strip() for value in pair.split("→", 1))
            manifest, _latest = self._latest_manifest()
            args = self.task_manager.build_change_pair(
                root, region, before, after, run_id,
                output_root=self.vars["output_root"].get(), manifest=manifest,
                area_truths=self.project_area_truths,
                validation_areas=self.project_validation_areas,
                absolute=self.vars["absolute"].get(), ratio=self.vars["ratio"].get(),
                tolerance=self.vars["tolerance"].get(),
                truth_type_field=self.vars["truth_type_field"].get(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("无法分步检测", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(args)
