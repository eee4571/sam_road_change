from __future__ import annotations
import re
import time
from pathlib import Path
from tkinter import messagebox

from tkinter import BOTH, LEFT, RIGHT, X, StringVar
from tkinter import ttk

from .common_widgets import LAYOUT_METRICS, bind_dynamic_wrap

class RunPage:
    def _build_run_page(self, page: ttk.Frame) -> None:
        self.run_body = page
        run_card = ttk.LabelFrame(page, text="运行任务", padding=LAYOUT_METRICS["card_padding"])
        run_card.pack(fill=X)
        self.preflight_summary = StringVar(value="开始处理时会自动检查以下内容；发现问题将立即停止并说明原因。")
        preflight_label = ttk.Label(run_card, textvariable=self.preflight_summary)
        preflight_label.pack(anchor="w", fill=X, pady=(0, 5))
        bind_dynamic_wrap(preflight_label, run_card, minimum=220, padding=20)
        profile_frame = ttk.LabelFrame(run_card, text="处理模式", padding=(8, 5))
        profile_frame.pack(fill=X, pady=(0, 7))
        ttk.Radiobutton(
            profile_frame, text="标准模式", variable=self.vars["execution_profile"],
            value="full",
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            profile_frame, text="快速模式", variable=self.vars["execution_profile"],
            value="fast",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        checklist = ttk.Frame(run_card)
        checklist.pack(fill=X, pady=(0, 6))
        self.preflight_check_labels = []
        for row_index, label in enumerate(("项目结构与验证区", "影像期次与覆盖范围", "模型、参数与运行环境", "输出位置与磁盘空间")):
            ttk.Label(checklist, text=label).grid(row=row_index, column=0, sticky="w", pady=2)
            state_label = ttk.Label(checklist, text="开始前检查", style="CardMuted.TLabel")
            state_label.grid(row=row_index, column=1, sticky="e", pady=2)
            self.preflight_check_labels.append(state_label)
        checklist.grid_columnconfigure(0, weight=1)
        full_task = ttk.LabelFrame(run_card, text="完整任务", padding=(8, 6))
        full_task.pack(fill=X)
        self.continue_task_button = ttk.Button(
            full_task, text="继续当前任务", style="Hero.TButton", command=self.run_all,
        )
        self.continue_task_button.grid(row=0, column=0, sticky="ew", padx=(0, LAYOUT_METRICS["module_gap"]))
        self.fresh_task_button = ttk.Button(
            full_task, text="从头重新运行完整流程", command=self.run_fresh_all,
        )
        self.fresh_task_button.grid(row=0, column=1, sticky="ew")
        full_task.grid_columnconfigure(0, weight=1, uniform="full_task")
        full_task.grid_columnconfigure(1, weight=1, uniform="full_task")
        self.task_start_buttons = [self.continue_task_button, self.fresh_task_button]
        self.run_button = self.continue_task_button
        utility_actions = ttk.Frame(run_card)
        utility_actions.pack(fill=X, pady=(6, 0))
        self.preflight_button = ttk.Button(utility_actions, text="检查数据", command=self.preflight_inputs)
        self.preflight_button.pack(side=LEFT)
        self.cancel_button = ttk.Button(utility_actions, text="取消任务", command=self.cancel_task)
        self.cancel_button.pack(side=RIGHT)
        self.cancel_button.state(["disabled"])

        progress_card = ttk.LabelFrame(page, text="运行进度", padding=LAYOUT_METRICS["card_padding"])
        progress_card.pack(fill=X, pady=(LAYOUT_METRICS["section_gap"], 0))
        ttk.Label(progress_card, text="当前阶段：").pack(anchor="w")
        self.run_status = StringVar(value="等待开始任务。")
        run_status_label = ttk.Label(progress_card, textvariable=self.run_status)
        run_status_label.pack(fill=X, pady=(2, 5))
        bind_dynamic_wrap(run_status_label, progress_card, minimum=220, padding=20)
        ttk.Label(progress_card, text="总体进度").pack(anchor="w", pady=(0, 3))
        self.progress = ttk.Progressbar(progress_card, mode="determinate", maximum=1, value=0, style="Modern.Horizontal.TProgressbar")
        self.progress.pack(fill=X)
        self.progress_text = StringVar(value="0 / 0 · 已用时 00:00:00 · 剩余 --")
        ttk.Label(progress_card, textvariable=self.progress_text).pack(anchor="w", pady=(5, 0))

        stage_card = ttk.LabelFrame(page, text="分步重跑", padding=LAYOUT_METRICS["card_padding"])
        stage_card.pack(fill=BOTH, expand=True, pady=(LAYOUT_METRICS["section_gap"], 0))
        road_actions = ttk.LabelFrame(stage_card, text="道路提取", padding=LAYOUT_METRICS["card_padding"])
        road_actions.pack(fill=X)
        ttk.Label(road_actions, text="区域：", width=LAYOUT_METRICS["form_label_width"]).grid(row=0, column=0, sticky="w", pady=(0, LAYOUT_METRICS["form_gap"]))
        self.stage_region_combo = ttk.Combobox(road_actions, textvariable=self.stage_region, state="readonly", width=14)
        self.stage_region_combo.grid(row=0, column=1, sticky="ew", padx=(0, LAYOUT_METRICS["module_gap"]), pady=(0, LAYOUT_METRICS["form_gap"]))
        self.stage_region_combo.bind("<<ComboboxSelected>>", self._stage_region_changed)
        ttk.Label(road_actions, text="期次：", width=6).grid(row=0, column=2, sticky="w", pady=(0, LAYOUT_METRICS["form_gap"]))
        self.stage_period_combo = ttk.Combobox(road_actions, textvariable=self.project_period, state="readonly", width=12)
        self.stage_period_combo.grid(row=0, column=3, sticky="ew", pady=(0, LAYOUT_METRICS["form_gap"]))
        self.stage_period_combo.bind("<<ComboboxSelected>>", self._stage_period_changed)
        road_actions.grid_columnconfigure(1, weight=1)
        road_actions.grid_columnconfigure(3, weight=1)
        self.affected_pairs_summary = StringVar(value="请选择期次以查看受影响的相邻变化对。")
        affected_label = ttk.Label(road_actions, textvariable=self.affected_pairs_summary, style="Hint.TLabel")
        affected_label.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, LAYOUT_METRICS["form_gap"]))
        bind_dynamic_wrap(affected_label, road_actions, minimum=180, padding=20)
        self.stage_buttons = []
        road_button_row = ttk.Frame(road_actions)
        road_button_row.grid(row=2, column=0, columnspan=4, sticky="ew")
        for column, (text, command) in enumerate((
            ("重跑该期", lambda: self.rerun_selected_period(False)),
            ("重跑并更新相关结果", lambda: self.rerun_selected_period(True)),
            ("重跑全部道路提取", self.run_extract_all),
        )):
            button = ttk.Button(road_button_row, text=text, command=command)
            button.grid(row=0, column=column, sticky="ew", padx=(0, LAYOUT_METRICS["module_gap"] if column < 2 else 0))
            road_button_row.grid_columnconfigure(column, weight=1, uniform="road_actions")
            self.stage_buttons.append(button)

        change_actions = ttk.LabelFrame(stage_card, text="变化检测", padding=LAYOUT_METRICS["card_padding"])
        change_actions.pack(fill=X, pady=(LAYOUT_METRICS["module_gap"], 0))
        ttk.Label(change_actions, text="区域：", width=LAYOUT_METRICS["form_label_width"]).grid(row=0, column=0, sticky="w", pady=(0, LAYOUT_METRICS["form_gap"]))
        self.stage_change_region_combo = ttk.Combobox(change_actions, textvariable=self.stage_region, state="readonly", width=14)
        self.stage_change_region_combo.grid(row=0, column=1, sticky="ew", padx=(0, LAYOUT_METRICS["module_gap"]), pady=(0, LAYOUT_METRICS["form_gap"]))
        self.stage_change_region_combo.bind("<<ComboboxSelected>>", self._stage_region_changed)
        ttk.Label(change_actions, text="变化对：", width=8).grid(row=0, column=2, sticky="w", pady=(0, LAYOUT_METRICS["form_gap"]))
        self.stage_pair_combo = ttk.Combobox(change_actions, textvariable=self.project_change_pair, state="readonly", width=16)
        self.stage_pair_combo.grid(row=0, column=3, sticky="ew", pady=(0, LAYOUT_METRICS["form_gap"]))
        change_actions.grid_columnconfigure(1, weight=1)
        change_actions.grid_columnconfigure(3, weight=1)
        change_button_row = ttk.Frame(change_actions)
        change_button_row.grid(row=1, column=0, columnspan=4, sticky="ew")
        for column, (text, command) in enumerate((
            ("重跑该变化对", lambda: self.rerun_selected_change(False)),
            ("重跑并更新长时序成果", lambda: self.rerun_selected_change(True)),
            ("重跑全部变化检测", self.run_change_all),
        )):
            button = ttk.Button(change_button_row, text=text, command=command)
            button.grid(row=0, column=column, sticky="ew", padx=(0, LAYOUT_METRICS["module_gap"] if column < 2 else 0))
            change_button_row.grid_columnconfigure(column, weight=1, uniform="change_actions")
            self.stage_buttons.append(button)

        summary_card = ttk.LabelFrame(self.step_summaries[1], text="任务摘要", padding=LAYOUT_METRICS["card_padding"])
        summary_card.pack(fill=BOTH, expand=True)
        self.run_destination_summary = StringVar(value=self.vars["output_root"].get())
        for row, (label, variable) in enumerate((
            ("当前任务：", self.vars["run_id"]),
            ("成果位置：", self.run_destination_summary),
        )):
            ttk.Label(summary_card, text=label, width=12, style="Metric.TLabel").grid(row=row, column=0, sticky="nw", pady=LAYOUT_METRICS["form_gap"] // 2)
            value_label = ttk.Label(summary_card, textvariable=variable, style="Metric.TLabel")
            value_label.grid(row=row, column=1, sticky="nw", pady=LAYOUT_METRICS["form_gap"] // 2)
            bind_dynamic_wrap(value_label, summary_card, minimum=160, padding=130)
        summary_card.grid_columnconfigure(1, weight=1)
        run_settings_shell = ttk.Frame(summary_card)
        run_settings_shell.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
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
            evaluate=(self.vars["execution_profile"].get() == "fast"),
            resume=(self.vars["resume"].get() == "1" and not preflight_only),
            continue_on_error=self.vars["continue_on_error"].get() == "1",
            preflight_only=preflight_only,
            data_check_only=data_check_only,
            junction_node_mode=self.vars["junction_node_mode"].get(),
            validation_areas=(self.project_validation_areas or None),
            area_truths=(self.project_area_truths or None),
            area_periods=(self.project_area_periods or None),
            execution_profile=self.vars["execution_profile"].get(),
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
        self._run_complete_task(fresh=False)

    def run_fresh_all(self) -> None:
        if self.project_config.get("active_task"):
            if not messagebox.askokcancel(
                "确认从头运行",
                "将创建新的任务并从头运行，当前已有任务结果不会删除。",
                parent=self.root,
            ):
                return
        self._run_complete_task(fresh=True)

    def _run_complete_task(self, *, fresh: bool) -> None:
        output_value = self.vars["output_root"].get().strip()
        if not output_value:
            messagebox.showerror("输入不完整", "请选择成果输出根目录。", parent=self.root)
            return
        output = Path(output_value).expanduser()
        previous_run_id = self.vars["run_id"].get()
        previous_resume = self.vars["resume"].get()
        if fresh:
            run_id, state_path = self.task_manager.create_new_run(output)
            should_resume = False
        else:
            run_id, should_resume, state_path = self.task_manager.resolve_run(
                output, previous_run_id, self.project_config.get("active_task"),
            )
        self.vars["run_id"].set(run_id)
        self.vars["resume"].set("1" if should_resume else "0")
        try:
            args = self._build_current_command()
        except ValueError as exc:
            self.vars["run_id"].set(previous_run_id)
            self.vars["resume"].set(previous_resume)
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
        if should_resume and self.project_root_path:
            try:
                relocation = self.task_manager.relocation_preview(
                    output, run_id, self.project_root_path,
                )
            except ValueError as exc:
                messagebox.showerror("无法安全重定位任务", str(exc), parent=self.root)
                return
            if relocation and not messagebox.askokcancel(
                "确认项目重定位",
                self.task_manager.relocation_message(relocation),
                parent=self.root,
            ):
                return
        order_text = self._period_order_confirmation()
        if order_text and not messagebox.askokcancel(
            "确认影像期次顺序",
            "程序将按以下顺序提取并检测相邻期次变化：\n\n" + order_text,
            parent=self.root,
        ):
            self.vars["run_id"].set(previous_run_id)
            self.vars["resume"].set(previous_resume)
            return
        self.project_config["active_task"] = {
            "run_id": run_id, "state": str(state_path), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_project_config()
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
        self._command(self.task_manager.build_rerun_all_periods(
            manifest, self.vars["continue_on_error"].get() == "1",
        ))

    def run_change_all(self) -> None:
        try:
            manifest = self._current_pipeline_manifest_path()
        except ValueError as exc:
            messagebox.showerror("无法批量重跑", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(self.task_manager.build_rerun_all_changes(
            manifest, self.vars["continue_on_error"].get() == "1",
        ))

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
