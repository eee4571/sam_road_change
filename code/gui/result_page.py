from __future__ import annotations
import json
import platform
import re
import sys
import time
import zipfile
from pathlib import Path
from tkinter import END, Toplevel, filedialog, messagebox

from app.project_manager import TemporalAttributePager, _manifest_path
from app.result_publisher import ProjectLayout
from .common_widgets import (
    Tooltip, bind_dynamic_wrap,
    configure_window_geometry, format_percentage,
)

ROOT = Path(__file__).resolve().parents[1]
from tkinter import BOTH, LEFT, RIGHT, X, StringVar
from tkinter import ttk

from .common_widgets import LAYOUT_METRICS, UI


def _evaluation_period_plan(manifest: dict) -> dict[str, list[str]]:
    plan: dict[str, list[str]] = {}
    for area, value in (manifest.get("period_orders") or {}).items():
        if not isinstance(value, dict):
            continue
        periods = [str(item).strip() for item in value.get("period_order", [])]
        periods = [item for item in periods if item]
        if periods:
            plan[str(area)] = list(dict.fromkeys(periods))
    spec = manifest.get("input_spec") if isinstance(manifest.get("input_spec"), dict) else {}
    for area, values in (spec.get("grids") or {}).items():
        if str(area) in plan or not isinstance(values, dict):
            continue
        plan[str(area)] = [str(period) for period in values]
    for entry in manifest.get("period_results", []) or []:
        if not isinstance(entry, dict):
            continue
        area = str(entry.get("grid") or "").strip()
        period = str(entry.get("period") or "").strip()
        if area and period and area not in plan:
            plan.setdefault(area, []).append(period)
    return plan


def _evaluation_overall(entry: dict) -> dict | None:
    if entry.get("evaluation_stale") or str(entry.get("evaluation_state") or "") in {"stale", "running", "failed"}:
        return None
    metrics_path = Path(str(entry.get("evaluation_metrics") or "")).expanduser()
    summary_path = Path(str(entry.get("summary") or "")).expanduser()
    if not metrics_path.is_file() or not summary_path.is_file():
        return None
    try:
        evaluation = json.loads(summary_path.read_text(encoding="utf-8")).get("evaluation", {})
        return next(
            dict(row) for row in evaluation.get("metrics", [])
            if isinstance(row, dict) and row.get("class") == "all"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, StopIteration, TypeError):
        return None


def build_area_evaluation_rows(manifest: dict) -> list[dict[str, object]]:
    """Build active-run area rows without consulting the cross-run result index."""
    plan = _evaluation_period_plan(manifest)
    entries = {
        (
            str(entry.get("grid") or ""),
            str(entry.get("before_period") or ""),
            str(entry.get("after_period") or ""),
        ): entry
        for entry in (manifest.get("change_results", []) or [])
        if isinstance(entry, dict)
    }
    for area, before, after in entries:
        if area and area not in plan:
            plan[area] = list(dict.fromkeys(
                value for key in entries if key[0] == area for value in key[1:]
            ))

    rows: list[dict[str, object]] = []
    for area, periods in plan.items():
        pairs = [(before, after) for before, after in zip(periods, periods[1:])]
        area_entries = [entries.get((area, before, after)) for before, after in pairs]
        total = len(pairs)
        completed = sum(
            bool(entry) and str(entry.get("status") or "completed") == "completed"
            for entry in area_entries
        )
        failed = sum(
            bool(entry) and str(entry.get("status") or "") == "failed"
            for entry in area_entries
        )
        running = any(
            bool(entry) and str(entry.get("status") or "") == "running"
            for entry in area_entries
        )
        stale = any(
            bool(entry) and (
                bool(entry.get("evaluation_stale"))
                or str(entry.get("evaluation_state") or "") == "stale"
            )
            for entry in area_entries
        )
        evaluating = any(
            bool(entry) and str(entry.get("evaluation_state") or "") == "running"
            for entry in area_entries
        )
        evaluation_failed = any(
            bool(entry) and str(entry.get("evaluation_state") or "") == "failed"
            for entry in area_entries
        )
        missing_truth = any(
            not entry or not str(entry.get("truth") or "").strip()
            for entry in area_entries
        ) if total else False
        overall_rows = [
            _evaluation_overall(entry) if entry else None for entry in area_entries
        ]
        evaluated = sum(row is not None for row in overall_rows)

        aggregate: dict[str, object] = {}
        if total and evaluated == total:
            valid = [row for row in overall_rows if row is not None]
            sums = {
                key: sum(float(row.get(key, 0) or 0) for row in valid)
                for key in (
                    "tp_m2", "fp_m2", "fn_m2",
                    "truth_centerline_length_px", "covered_truth_centerline_length_px",
                    "predicted_centerline_length_px", "centerline_offset_integral_px2",
                    "type_correct_tp_count", "type_matched_tp_count",
                    "correctly_classified_m2", "detected_truth_m2",
                )
            }
            aggregate = {
                "change_recall": (
                    sums["tp_m2"] / (sums["tp_m2"] + sums["fn_m2"])
                    if sums["tp_m2"] + sums["fn_m2"] else 0.0
                ),
                "change_precision": (
                    sums["tp_m2"] / (sums["tp_m2"] + sums["fp_m2"])
                    if sums["tp_m2"] + sums["fp_m2"] else 0.0
                ),
                "road_centerline_completeness": (
                    sums["covered_truth_centerline_length_px"] / sums["truth_centerline_length_px"]
                    if sums["truth_centerline_length_px"] else None
                ),
                "centerline_mean_offset_px": (
                    sums["centerline_offset_integral_px2"] / sums["predicted_centerline_length_px"]
                    if sums["predicted_centerline_length_px"] else None
                ),
                "change_type_accuracy": (
                    sums["type_correct_tp_count"] / sums["type_matched_tp_count"]
                    if sums["type_matched_tp_count"]
                    else (
                        sums["correctly_classified_m2"] / sums["detected_truth_m2"]
                        if sums["detected_truth_m2"] else None
                    )
                ),
            }

        if failed or evaluation_failed:
            status = "部分失败"
        elif running:
            status = "检测中"
        elif completed < total:
            status = "检测中" if completed else "未开始"
        elif missing_truth:
            status = "缺少真值"
        elif evaluating:
            status = "评价中"
        elif stale:
            status = "需重新评价"
        elif evaluated == total and total:
            status = "已完成"
        elif evaluated:
            status = "评价中"
        else:
            status = "待评价"
        rows.append({
            "area": area,
            "change_progress": f"{completed} / {total}",
            "evaluation_progress": f"{evaluated} / {total}",
            "status": status,
            **aggregate,
        })
    return rows


class ResultPage:
    def _build_result_page(self, page: ttk.Frame) -> None:
        self.result_body = page
        result_card = ttk.LabelFrame(page, text="处理结果", padding=LAYOUT_METRICS["card_padding"])
        result_card.pack(fill=X)
        self.result_status = StringVar(value="打开项目或完成任务后，可在此查看处理结果。")
        self.result_period_count = StringVar(value="0 期")
        self.result_change_count = StringVar(value="0 组")
        self.result_area_count = StringVar(value="0 区")
        self.result_review_count = StringVar(value="0 处")
        result_status_label = ttk.Label(result_card, textvariable=self.result_status)
        result_status_label.pack(fill=X, pady=(0, 5))
        bind_dynamic_wrap(result_status_label, result_card, minimum=260, padding=20)
        browser = ttk.Frame(result_card)
        browser.pack(fill=X, pady=(0, 5))
        self.result_tree = ttk.Treeview(browser, columns=("status",), show="tree headings", height=6, style="Data.Treeview")
        self.result_tree.heading("#0", text="成果")
        self.result_tree.heading("status", text="状态")
        self.result_tree.column("#0", width=620, minwidth=240, stretch=True)
        self.result_tree.column("status", width=90, minwidth=80, stretch=False)
        tree_scroll = ttk.Scrollbar(browser, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=tree_scroll.set)
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        browser.grid_columnconfigure(0, weight=1)
        self.result_tree.tag_configure("area", font=("Microsoft YaHei UI", 9, "bold"))
        self.result_tree.tag_configure("category", foreground=UI["green"], font=("Microsoft YaHei UI", 9, "bold"))
        self.result_tree_tooltip = Tooltip(self.result_tree, self._result_tree_tooltip_text)
        self.result_tree.bind("<Double-1>", lambda _event: self.open_selected_result())
        result_actions = ttk.Frame(result_card)
        result_actions.pack(fill=X)
        ttk.Button(result_actions, text="打开所选成果", command=self.open_selected_result).pack(side=LEFT)
        ttk.Button(result_actions, text="打开所在目录", command=self.open_selected_result_folder).pack(side=LEFT, padx=(5, 0))
        ttk.Button(result_actions, text="查看长时序属性表", command=self.open_temporal_attribute_table).pack(side=LEFT, padx=(5, 0))
        ttk.Button(result_actions, text="刷新成果", command=self.refresh_project_results).pack(side=LEFT, padx=(5, 0))

        evaluation_card = ttk.LabelFrame(page, text="精度评价", padding=LAYOUT_METRICS["card_padding"])
        evaluation_card.pack(fill=X, pady=(LAYOUT_METRICS["section_gap"], 0))
        evaluation_hint = ttk.Label(
            evaluation_card,
            text="根据真值数据中的变化类型，对新增、变化和灭失道路进行精度评价，并汇总各验证区及相邻影像期次的评价结果。中心线平均偏差仅统计新增和灭失道路；宽度变化道路不纳入该指标。",
            style="Hint.TLabel",
        )
        evaluation_hint.pack(anchor="w", fill=X, pady=(0, 6))
        bind_dynamic_wrap(evaluation_hint, evaluation_card, minimum=260, padding=20)

        evaluation_table_frame = ttk.Frame(evaluation_card)
        evaluation_table_frame.pack(fill=X, pady=(0, 7))
        evaluation_columns = (
            "area", "change_progress", "evaluation_progress", "recall", "precision",
            "completeness", "offset", "type_accuracy", "status",
        )
        self.evaluation_tree = ttk.Treeview(
            evaluation_table_frame, columns=evaluation_columns, show="headings",
            height=5, style="Data.Treeview",
        )
        headings = {
            "area": "验证区域", "change_progress": "变化检测", "evaluation_progress": "精度评价",
            "recall": "变化图斑查全率", "precision": "变化图斑准确率",
            "completeness": "变化道路提取完整度", "offset": "中心线平均偏移",
            "type_accuracy": "动态过程检测正确率", "status": "状态",
        }
        widths = {
            "area": 130, "change_progress": 78, "evaluation_progress": 78,
            "recall": 115, "precision": 115, "completeness": 135,
            "offset": 110, "type_accuracy": 135, "status": 90,
        }
        for name in evaluation_columns:
            anchor = "w" if name == "area" else "center"
            self.evaluation_tree.heading(name, text=headings[name], anchor=anchor)
            self.evaluation_tree.column(
                name, width=widths[name], minwidth=65, stretch=(name == "area"), anchor=anchor,
            )
        evaluation_ybar = ttk.Scrollbar(
            evaluation_table_frame, orient="vertical", command=self.evaluation_tree.yview,
        )
        evaluation_xbar = ttk.Scrollbar(
            evaluation_table_frame, orient="horizontal", command=self.evaluation_tree.xview,
        )
        self.evaluation_tree.configure(
            yscrollcommand=evaluation_ybar.set, xscrollcommand=evaluation_xbar.set,
        )
        self.evaluation_tree.grid(row=0, column=0, sticky="nsew")
        evaluation_ybar.grid(row=0, column=1, sticky="ns")
        evaluation_xbar.grid(row=1, column=0, sticky="ew")
        evaluation_table_frame.grid_columnconfigure(0, weight=1)
        self.evaluation_tree.bind("<<TreeviewSelect>>", self._evaluation_area_selected)

        pair_row = ttk.Frame(evaluation_card)
        pair_row.pack(fill=X, pady=LAYOUT_METRICS["form_gap"])
        ttk.Label(pair_row, text="单项评价（可选）", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.evaluation_pair = StringVar(value="")
        self.evaluation_pair_combo = ttk.Combobox(pair_row, textvariable=self.evaluation_pair, state="readonly")
        self.evaluation_pair_combo.pack(side=LEFT, fill=X, expand=True)
        self.evaluation_pair_combo.bind("<<ComboboxSelected>>", self._evaluation_pair_changed)
        truth_row = ttk.Frame(evaluation_card)
        truth_row.pack(fill=X, pady=LAYOUT_METRICS["form_gap"])
        ttk.Label(truth_row, text="项目真值", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.evaluation_truth = StringVar(value="")
        ttk.Label(truth_row, textvariable=self.evaluation_truth, style="CardMuted.TLabel", anchor="w").pack(side=LEFT, fill=X, expand=True)
        self.supplement_evaluation_truth_button = ttk.Button(
            truth_row, text="补充真值", style="Compact.TButton",
            command=self.supplement_evaluation_truth,
        )
        self.supplement_evaluation_truth_button.pack(side=LEFT, padx=(8, 0))
        self.supplement_evaluation_truth_button.state(["disabled"])
        self.evaluation_truth_summary = StringVar(value="总体评价会按区域和相邻期次分别匹配真值，不使用上方单个路径代替全部真值。")
        truth_summary_label = ttk.Label(
            evaluation_card, textvariable=self.evaluation_truth_summary, style="CardMuted.TLabel",
        )
        truth_summary_label.pack(anchor="w", fill=X, pady=(0, 5))
        bind_dynamic_wrap(truth_summary_label, evaluation_card, minimum=260, padding=20)
        self.evaluation_advanced_toggle = ttk.Button(evaluation_card, text="评价高级设置...", command=self._toggle_evaluation_advanced)
        self.evaluation_advanced_toggle.pack(anchor="w", pady=(4, 4))
        options_row = ttk.Frame(evaluation_card)
        self.evaluation_advanced_frame = options_row
        field_row = ttk.Frame(options_row)
        field_row.pack(fill=X)
        ttk.Label(field_row, text="变化类型字段", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.evaluation_type_field = StringVar(value="BHBM")
        self.evaluation_type_field_combo = ttk.Combobox(
            field_row, textvariable=self.evaluation_type_field, width=18, state="normal",
        )
        self.evaluation_type_field_combo.pack(side=LEFT)
        self.evaluation_type_field_combo.bind(
            "<<ComboboxSelected>>", self._evaluation_type_field_changed,
        )
        self.evaluation_type_field_combo.bind(
            "<FocusOut>", self._evaluation_type_field_changed,
        )
        ttk.Button(
            field_row, text="检查当前真值字段", style="Compact.TButton",
            command=lambda: self._refresh_evaluation_truth_fields(show_error=True),
        ).pack(side=LEFT, padx=(6, 0))
        tolerance_row = ttk.Frame(options_row)
        tolerance_row.pack(fill=X, pady=(4, 0))
        ttk.Label(tolerance_row, text="中心线匹配容差（米）", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.evaluation_tolerance = StringVar(value="5.0")
        ttk.Entry(tolerance_row, textvariable=self.evaluation_tolerance, width=9).pack(side=LEFT)
        self.evaluation_type_field_status = StringVar(
            value="选择变化对并补充真值后，可检查 SHP 中实际存在的字段和样例值。",
        )
        self._evaluation_truth_field_summary: dict[str, object] = {}
        type_field_status_label = ttk.Label(
            options_row, textvariable=self.evaluation_type_field_status, style="CardMuted.TLabel",
        )
        type_field_status_label.pack(anchor="w", fill=X, pady=(4, 0))
        bind_dynamic_wrap(type_field_status_label, options_row, minimum=260, padding=20)
        value_row = ttk.Frame(options_row)
        value_row.pack(fill=X, pady=(5, 0))
        ttk.Label(value_row, text="字段值对应关系", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.evaluation_added_value = StringVar(value="2")
        self.evaluation_width_changed_value = StringVar(value="3")
        self.evaluation_removed_value = StringVar(value="4")
        self.evaluation_value_combos = {}
        for key, label, variable in (
            ("added", "新增", self.evaluation_added_value),
            ("width_changed", "宽度变化", self.evaluation_width_changed_value),
            ("removed", "灭失", self.evaluation_removed_value),
        ):
            ttk.Label(value_row, text=f"{label}=").pack(side=LEFT, padx=((8 if key != "added" else 0), 3))
            combo = ttk.Combobox(value_row, textvariable=variable, width=9, state="normal")
            combo.pack(side=LEFT)
            self.evaluation_value_combos[key] = combo
        action_row = ttk.Frame(evaluation_card)
        action_row.pack(fill=X)
        self.evaluation_status = StringVar(value="精度表仅显示当前 active task；任务事件会自动刷新进度。")
        evaluation_status_label = ttk.Label(action_row, textvariable=self.evaluation_status, style="Hint.TLabel")
        evaluation_status_label.pack(side=LEFT, fill=X, expand=True)
        bind_dynamic_wrap(evaluation_status_label, action_row, minimum=220, padding=280)
        self.run_evaluation_button = ttk.Button(
            action_row, text="评价所选结果", command=self.run_result_evaluation,
        )
        self.run_evaluation_button.pack(side=RIGHT)
        self.run_evaluation_button.state(["disabled"])
        self.run_total_evaluation_button = ttk.Button(
            action_row, text="评价全部结果", style="Hero.TButton", command=self.run_total_evaluation,
        )
        self.run_total_evaluation_button.pack(side=RIGHT, padx=(0, 8))
        self.run_total_evaluation_button.state(["disabled"])

        summary = ttk.LabelFrame(self.step_summaries[3], text="成果摘要", padding=LAYOUT_METRICS["card_padding"])
        summary.pack(fill=BOTH, expand=True)
        self.result_temporal_summary = StringVar(value="未生成")
        self.result_evaluable_count = StringVar(value="0 组")
        for row, (label, variable) in enumerate((
            ("验证区：", self.result_area_count),
            ("影像期次：", self.result_period_count),
            ("变化检测：", self.result_change_count),
            ("长时序成果：", self.result_temporal_summary),
            ("可人工编辑：", self.result_review_count),
            ("可评价变化对：", self.result_evaluable_count),
        )):
            ttk.Label(summary, text=label, width=15, style="Metric.TLabel").grid(row=row, column=0, sticky="nw", pady=LAYOUT_METRICS["form_gap"] // 2)
            summary_value = ttk.Label(summary, textvariable=variable, style="Metric.TLabel")
            summary_value.grid(row=row, column=1, sticky="nw", pady=LAYOUT_METRICS["form_gap"] // 2)
            bind_dynamic_wrap(summary_value, summary, minimum=160, padding=150)
        summary.grid_columnconfigure(1, weight=1)

    def _latest_manifest(self) -> tuple[dict | None, Path | None]:
        if not self.project_root_path:
            return None, None
        return self.project_manager.result_context(
            self.project_root_path, self.vars["output_root"].get().strip(), persist=True,
        )

    def _active_evaluation_manifest(self) -> tuple[dict | None, Path | None]:
        """Read only the exact active task; never merge metrics from historical runs."""
        if not self.project_root_path:
            return None, None
        try:
            path = self.task_manager.active_pipeline_manifest(
                self.vars["output_root"].get().strip(),
                self.project_config.get("active_task"),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeError, json.JSONDecodeError):
            return None, None
        if isinstance(payload.get("result_manifest"), dict):
            payload = payload["result_manifest"]
        return (payload, path) if isinstance(payload, dict) else (None, None)

    def _refresh_result_availability(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            self.results_available = False
            self.review_items = []
            self.temporal_items = []
            self.result_status.set("尚无可显示的任务成果；完成任务或打开已有项目后即可查看。")
            if hasattr(self, "result_period_count"):
                self.result_period_count.set("0 期")
                self.result_change_count.set("0 组")
                self.result_area_count.set("0 区")
                self.result_review_count.set("0 处")
                self.result_temporal_summary.set("未生成")
                self.result_evaluable_count.set("0 组")
            if hasattr(self, "review_status"):
                self.review_status.set("尚未完成自动处理，暂无可复核数据。")
                self._populate_review_step()
            if hasattr(self, "evaluation_pair_combo"):
                self.result_change_items = []
                self.evaluation_pair_combo.configure(values=())
                self.evaluation_pair.set("")
                self.evaluation_status.set("请先载入或生成包含变化检测的任务结果。")
                self._set_evaluation_actions_enabled(False)
                self._populate_evaluation_tree([])
            self._populate_result_tree([], None)
            return
        base_dir = latest.parent
        self.review_items = self.project_manager.review_items(manifest, base_dir)
        self.temporal_items = self.project_manager.temporal_items(manifest, base_dir)
        index = self.project_manager.result_index(self.vars["output_root"].get().strip())
        indexed_areas = (index or {}).get("areas") or {}
        self.results_available = bool(
            manifest.get("period_results") or manifest.get("change_results")
            or self.review_items or self.temporal_items or indexed_areas
        )
        area_names = {
            str(entry.get("grid") or "validation")
            for entry in (manifest.get("period_results", []) or []) if isinstance(entry, dict)
        }
        if indexed_areas:
            area_names.update(str(name) for name in indexed_areas)
        period_count = len(manifest.get("period_results", []) or []) or sum(
            len(value.get("periods") or {}) for value in indexed_areas.values()
            if isinstance(value, dict)
        )
        change_count = len(manifest.get("change_results", []) or []) or sum(
            len(value.get("changes") or {}) for value in indexed_areas.values()
            if isinstance(value, dict)
        )
        message = (
            f"本次任务包含 {len(area_names)} 个验证区、{period_count} 个影像期次，"
            f"已生成 {change_count} 组变化检测成果。"
            if area_names else "本次任务暂未生成可查看的正式成果。"
        )
        temporal_results = [
            entry for entry in (manifest.get("temporal_results", []) or [])
            if isinstance(entry, dict)
        ]
        if temporal_results:
            temporal_roads = sum(int(entry.get("road_count", 0) or 0) for entry in temporal_results)
            message += f" 长时序道路成果包含 {temporal_roads} 条稳定道路段。"
        self.result_status.set(message)
        self.result_period_count.set(f"{period_count} 期")
        self.result_change_count.set(f"{change_count} 组")
        self.result_area_count.set(f"{len(area_names)} 区")
        review_count = sum(
            max(0, int(item.get("manual_item_count", 0) or 0))
            for item in self.review_items
            if str(item.get("manual_item_count", 0) or 0).isdigit()
        )
        self.result_review_count.set(f"{review_count} 处")
        self.result_temporal_summary.set("已生成" if temporal_results else "未生成")
        self._populate_result_tree(self.project_manager.result_items(manifest, base_dir), base_dir)
        self._refresh_active_evaluation_results()
        self._populate_review_step()

    def _refresh_active_evaluation_results(self) -> None:
        manifest, path = self._active_evaluation_manifest()
        self.active_evaluation_manifest_path = path
        self._refresh_evaluation_results(manifest or {})

    def handle_evaluation_backend_event(self, payload: dict) -> None:
        """Refresh the active-run table only on relevant structured events."""
        stage = str(payload.get("stage") or "")
        if (
            "变化" in stage or "评价" in stage
            or stage in {"rerun-change", "rerun-all-changes", "rerun-period", "rerun-all-periods"}
        ):
            self._refresh_active_evaluation_results()

    def _populate_result_tree(self, items: list[dict[str, str]], _base_dir: Path | None) -> None:
        if not hasattr(self, "result_tree"):
            return
        fingerprint = tuple(
            (item.get("id", ""), item.get("parent", ""), item.get("label", ""), item.get("path", ""), item.get("status", ""))
            for item in items
        )
        if fingerprint == self._result_tree_fingerprint:
            return
        self._result_tree_fingerprint = fingerprint
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_tree_paths = {}
        pending = list(items)
        inserted = set()
        while pending:
            next_pending = []
            for item in pending:
                parent = item.get("parent", "")
                if parent and parent not in inserted:
                    next_pending.append(item)
                    continue
                is_area = not bool(parent)
                is_category = bool(parent) and not item.get("path")
                tag = "area" if is_area else "category" if is_category else ""
                status = "" if is_category else item.get("status", "")
                node = self.result_tree.insert(
                    parent, END, iid=item["id"], text=item["label"], values=(status,),
                    open=is_area, tags=(tag,) if tag else (),
                )
                inserted.add(node)
                path_text = item.get("path", "")
                if path_text:
                    self.result_tree_paths[node] = Path(path_text)
            if len(next_pending) == len(pending):
                break
            pending = next_pending
        visible_rows = sum(
            1 + len(self.result_tree.get_children(root))
            for root in self.result_tree.get_children("")
        )
        self.result_tree.configure(height=max(6, min(visible_rows, 10)))

    def _result_tree_tooltip_text(self, _event=None) -> str:
        if not hasattr(self, "result_tree"):
            return ""
        y = self.result_tree.winfo_pointery() - self.result_tree.winfo_rooty()
        row = self.result_tree.identify_row(y)
        path = self.result_tree_paths.get(row)
        return str(path) if path is not None else ""

    def open_selected_result(self) -> None:
        if not hasattr(self, "result_tree"):
            return
        selected = self.result_tree.selection()
        if not selected:
            messagebox.showinfo("请选择成果", "请先在成果树中选择一个已生成成果。", parent=self.root)
            return
        path = self.result_tree_paths.get(selected[0])
        if path is None or not path.exists():
            messagebox.showinfo("未生成", "所选成果尚未生成，不能打开。", parent=self.root)
            return
        self._open(path)

    def open_selected_result_folder(self) -> None:
        if not hasattr(self, "result_tree"):
            return
        selected = self.result_tree.selection()
        path = self.result_tree_paths.get(selected[0]) if selected else None
        if path is None:
            manifest, latest = self._latest_manifest()
            if manifest is None or latest is None:
                messagebox.showinfo("暂无目录", "尚未载入任务成果。", parent=self.root)
                return
            path = Path(str(manifest.get("job_root") or latest.parent))
        target = path if path.is_dir() else path.parent
        if not target.exists():
            messagebox.showinfo("目录不存在", f"成果目录尚不存在：\n{target}", parent=self.root)
            return
        self._open(target)

    def _set_evaluation_actions_enabled(self, enabled: bool) -> None:
        state = ["!disabled"] if enabled else ["disabled"]
        for name in (
            "supplement_evaluation_truth_button",
            "run_evaluation_button",
            "run_total_evaluation_button",
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.state(state)

    def _refresh_evaluation_results(self, manifest: dict) -> None:
        if not hasattr(self, "evaluation_pair_combo"):
            return
        self.result_change_items = [
            entry for entry in (manifest.get("change_results", []) or [])
            if isinstance(entry, dict) and (
                (entry.get("gpkg") and Path(str(entry["gpkg"])).is_file())
                or (entry.get("summary") and Path(str(entry["summary"])).is_file())
                or (
                    isinstance(entry.get("layers"), dict)
                    and entry["layers"].get("changes")
                    and Path(str(entry["layers"]["changes"])).is_file()
                )
            )
        ]
        labels = [
            f"{entry.get('grid', '项目')} / {entry.get('before_period', '前期')} - {entry.get('after_period', '后期')}"
            for entry in self.result_change_items
        ]
        if hasattr(self, "result_evaluable_count"):
            self.result_evaluable_count.set(f"{len(labels)} 组")
        area_rows = build_area_evaluation_rows(manifest)
        self._populate_evaluation_tree(area_rows)
        self.evaluation_pair_combo.configure(values=labels)
        if labels and self.evaluation_pair.get() not in labels:
            self.evaluation_pair.set(labels[0])
        if not labels:
            self.evaluation_pair.set("")
            self.evaluation_status.set(
                "当前任务没有可评价的变化检测成果。"
                if manifest else "当前项目没有可读取的 active task。"
            )
            self._set_evaluation_actions_enabled(False)
            return
        evaluated = sum(row.get("status") == "已完成" for row in area_rows)
        self.evaluation_status.set(
            f"当前 active task 包含 {len(area_rows)} 个验证区域；"
            f"{evaluated} 个区域已完成全部变化对评价。"
        )
        configured = len(self.project_area_truths)
        embedded = sum(bool(str(entry.get("truth") or "").strip()) for entry in self.result_change_items)
        self.evaluation_truth_summary.set(
            f"总体评价将自动匹配 {max(configured, embedded)} 个区域/相邻期次真值；"
            "上方路径只用于“评价当前结果”。"
        )
        self._evaluation_pair_changed()
        self._set_evaluation_actions_enabled(True)

    @staticmethod
    def _evaluation_metric_text(value: object, *, offset: bool = False) -> str:
        if value in {None, ""}:
            return "--"
        try:
            return f"{float(value):.2f} px" if offset else format_percentage(float(value))
        except (TypeError, ValueError):
            return "--"

    def _populate_evaluation_tree(self, rows: list[dict[str, object]]) -> None:
        tree = getattr(self, "evaluation_tree", None)
        if tree is None:
            return
        selected_area = ""
        selected = tree.selection()
        if selected:
            values = tree.item(selected[0], "values")
            selected_area = str(values[0]) if values else ""
        tree.delete(*tree.get_children())
        selected_id = ""
        for index, row in enumerate(rows):
            area = str(row.get("area") or "")
            item = tree.insert("", END, values=(
                area,
                row.get("change_progress", "0 / 0"),
                row.get("evaluation_progress", "0 / 0"),
                self._evaluation_metric_text(row.get("change_recall")),
                self._evaluation_metric_text(row.get("change_precision")),
                self._evaluation_metric_text(row.get("road_centerline_completeness")),
                self._evaluation_metric_text(row.get("centerline_mean_offset_px"), offset=True),
                self._evaluation_metric_text(row.get("change_type_accuracy")),
                row.get("status", "未开始"),
            ))
            if area == selected_area or (not selected_area and index == 0):
                selected_id = item
        if selected_id:
            tree.selection_set(selected_id)

    def _evaluation_area_selected(self, _event=None) -> None:
        tree = getattr(self, "evaluation_tree", None)
        if tree is None or not tree.selection():
            return
        values = tree.item(tree.selection()[0], "values")
        area = str(values[0]) if values else ""
        labels = list(self.evaluation_pair_combo["values"])
        label = next((item for item in labels if item.startswith(f"{area} / ")), "")
        if label:
            self.evaluation_pair.set(label)
            self._evaluation_pair_changed()

    def _evaluation_pair_changed(self, _event=None) -> None:
        if not self.result_change_items or not hasattr(self, "evaluation_pair_combo"):
            return
        try:
            index = list(self.evaluation_pair_combo["values"]).index(self.evaluation_pair.get())
            item = self.result_change_items[index]
        except (ValueError, IndexError):
            return
        key = (str(item.get("grid")), str(item.get("before_period")), str(item.get("after_period")))
        truth = next(
            (path for area, before, after, path in self.project_area_truths if (area, before, after) == key),
            str(item.get("truth") or ""),
        )
        if truth:
            self.evaluation_truth.set(truth)
            self.evaluation_truth_summary.set(f"已匹配项目真值：{truth}")
        else:
            self.evaluation_truth.set("缺少真值")
            self.evaluation_truth_summary.set("缺少真值；点击“补充真值”后会自动保存到项目配置。")
        self._refresh_evaluation_truth_fields()

    def _refresh_evaluation_truth_fields(self, *, show_error: bool = False) -> None:
        if not hasattr(self, "evaluation_type_field_combo"):
            return
        truth = str(self.evaluation_truth.get() or "").strip()
        if not truth or truth == "缺少真值":
            self._evaluation_truth_field_summary = {}
            self.evaluation_type_field_combo.configure(values=())
            self.evaluation_type_field_status.set(
                "选择变化对并补充真值后，可检查 SHP 中实际存在的字段和样例值。",
            )
            return
        try:
            summary = self.project_manager.truth_field_summary(truth)
        except (OSError, UnicodeError, ValueError) as exc:
            self._evaluation_truth_field_summary = {}
            self.evaluation_type_field_combo.configure(values=())
            self.evaluation_type_field_status.set(f"无法读取当前真值字段：{exc}")
            if show_error:
                messagebox.showerror("无法检查真值字段", str(exc), parent=self.root)
            return
        self._evaluation_truth_field_summary = summary
        fields = [str(name) for name in summary.get("fields", [])]
        self.evaluation_type_field_combo.configure(values=fields)
        selected = self.evaluation_type_field.get().strip()
        canonical = next(
            (name for name in fields if name.casefold() == selected.casefold()), None,
        )
        if canonical is not None and canonical != selected:
            self.evaluation_type_field.set(canonical)
        self._evaluation_type_field_changed()

    def _evaluation_type_field_changed(self, _event=None) -> None:
        if not hasattr(self, "evaluation_type_field_status"):
            return
        field = self.evaluation_type_field.get().strip()
        summary = getattr(self, "_evaluation_truth_field_summary", {})
        fields = [str(name) for name in summary.get("fields", [])]
        matched = next((name for name in fields if name.casefold() == field.casefold()), None)
        if not fields:
            return
        if matched is None:
            self.evaluation_type_field_status.set(
                f"当前真值不含字段“{field or '（空）'}”；请选择实际字段：{'、'.join(fields)}。",
            )
            return
        values_by_field = summary.get("values") if isinstance(summary.get("values"), dict) else {}
        values = [str(value) for value in values_by_field.get(matched, [])]
        samples = "、".join(values) if values else "未读取到非空值"
        for combo in getattr(self, "evaluation_value_combos", {}).values():
            combo.configure(values=values)
        if matched.casefold() == "bhbm":
            defaults = ("2", "3", "4")
            for variable, default in zip((
                self.evaluation_added_value,
                self.evaluation_width_changed_value,
                self.evaluation_removed_value,
            ), defaults):
                if default in values or not values:
                    variable.set(default)
            absent = [value for value in defaults if values and value not in values]
            absent_note = (
                f"；未出现的类别值：{'、'.join(absent)}（按当前真值无对应类别处理）"
                if absent else ""
            )
            detail = (
                f"编码规则：2=新增，3=宽度变化，4=灭失；当前样例值：{samples}"
                f"{absent_note}"
            )
        else:
            detail = f"样例值：{samples}"
        self.evaluation_type_field_status.set(f"已确认字段“{matched}”；{detail}。")

    def _evaluation_truth_value_map(self) -> dict[str, str]:
        mapping = {
            "added": self.evaluation_added_value.get().strip(),
            "width_changed": self.evaluation_width_changed_value.get().strip(),
            "removed": self.evaluation_removed_value.get().strip(),
        }
        if any(not value for value in mapping.values()):
            raise ValueError("请分别选择新增、宽度变化和灭失对应的真值字段值。")
        if len({value.casefold() for value in mapping.values()}) != 3:
            raise ValueError("新增、宽度变化和灭失必须对应三个不同的真值字段值。")
        return mapping

    def supplement_evaluation_truth(self) -> None:
        if not self.result_change_items or not hasattr(self, "evaluation_pair_combo"):
            messagebox.showinfo(
                "暂无可评价成果", "当前没有可补充真值的变化检测成果。", parent=self.root,
            )
            return
        try:
            index = list(self.evaluation_pair_combo["values"]).index(self.evaluation_pair.get())
            item = self.result_change_items[index]
        except (ValueError, IndexError):
            messagebox.showinfo("未选择变化对", "请先选择一组待评价的变化检测成果。", parent=self.root)
            return
        path = self._select_path("shp")
        if not path:
            return
        key = (str(item.get("grid")), str(item.get("before_period")), str(item.get("after_period")))
        self.project_area_truths = [
            row for row in self.project_area_truths if (row[0], row[1], row[2]) != key
        ]
        self.project_area_truths.append((*key, str(Path(path).resolve())))
        self._save_project_config()
        self._evaluation_pair_changed()
        self.status.set("真值映射已补充并自动保存到项目配置。")

    def reload_project_truths(self) -> None:
        if not self.project_root_path:
            messagebox.showinfo("尚未打开项目", "请先在首页打开包含各区域真值的项目文件夹。", parent=self.root)
            return
        try:
            project = self.project_manager.discover_project(self.project_root_path)
        except ValueError as exc:
            messagebox.showerror("无法读取项目真值", str(exc), parent=self.root)
            return
        self.project_area_truths = [
            (area, before, after, path)
            for (area, before, after), path in (project.get("area_truths") or {}).items()
        ]
        self._evaluation_pair_changed()
        self.evaluation_truth_summary.set(
            f"已从项目读取 {len(self.project_area_truths)} 个区域/相邻期次真值；总体评价会逐项自动匹配。"
        )
        self.status.set(self.evaluation_truth_summary.get())

    def run_result_evaluation(self) -> None:
        manifest, latest = self._active_evaluation_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无成果", "请先载入或生成任务结果。", parent=self.root)
            return
        if not self.result_change_items:
            messagebox.showinfo("暂无可评价成果", "当前任务没有可评价的变化检测成果。", parent=self.root)
            return
        if self.evaluation_truth.get().strip() in {"", "缺少真值"}:
            messagebox.showinfo("缺少真值", "请先为所选变化对补充真值。", parent=self.root)
            return
        try:
            index = list(self.evaluation_pair_combo["values"]).index(self.evaluation_pair.get())
            item = self.result_change_items[index]
        except (ValueError, IndexError):
            messagebox.showinfo("未选择变化对", "请先选择一组待评价的变化检测成果。", parent=self.root)
            return
        try:
            value_map = self._evaluation_truth_value_map()
        except ValueError as exc:
            messagebox.showinfo("真值类型映射不完整", str(exc), parent=self.root)
            return
        try:
            args = self.task_manager.build_evaluate_existing(
                item, latest, self.evaluation_truth.get(),
                truth_type_field=self.evaluation_type_field.get(),
                validation_area=str(item.get("validation_area") or manifest.get("validation_area") or ""),
                evaluation_tolerance=self.evaluation_tolerance.get(),
                truth_value_map=value_map,
            )
        except ValueError as exc:
            messagebox.showerror("无法运行精度评价", str(exc), parent=self.root)
            return
        self.evaluation_status.set("正在读取检测结果并计算区域、类型和中心线精度指标…")
        self._command(args)

    def run_total_evaluation(self) -> None:
        manifest, latest = self._active_evaluation_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无成果", "请先载入或生成任务结果。", parent=self.root)
            return
        if not self.result_change_items:
            messagebox.showinfo("暂无可评价成果", "当前任务没有可评价的变化检测成果。", parent=self.root)
            return
        try:
            value_map = self._evaluation_truth_value_map()
            args = self.task_manager.build_evaluate_all(
                manifest, latest, self.project_area_truths,
                truth_type_field=self.evaluation_type_field.get(),
                evaluation_tolerance=self.evaluation_tolerance.get(),
                truth_value_map=value_map,
            )
        except ValueError as exc:
            messagebox.showerror("无法运行总精度评价", str(exc), parent=self.root)
            return
        self.evaluation_status.set("正在评价全部验证区和相邻期次，并汇总精度与中心线平均偏差…")
        self._command(args)

    def open_latest(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无成果", f"尚未找到最新成果索引：\n{latest}", parent=self.root)
            return
        self._refresh_result_availability()
        output_root = Path(str(manifest.get("output_root") or self.vars["output_root"].get())).expanduser()
        self._open(output_root)

    def refresh_project_results(self, *, automatic: bool = False) -> None:
        """Reload the current project's index and manifests without a file picker."""
        self._result_tree_fingerprint = None
        self._refresh_result_availability()
        if not automatic:
            self.status.set(
                "成果已刷新；已重新读取当前项目的正式索引、任务状态和人工编辑资料。"
                if self.results_available else "成果已刷新；当前项目尚无正式成果。"
            )

    def open_task_report(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无报告", "尚未找到最新任务索引。", parent=self.root)
            return
        job_root = Path(str(manifest.get("job_root") or latest.parent)).expanduser()
        report = Path(str(manifest.get("output_root") or self.vars["output_root"].get())) / "task_report.csv"
        if not report.is_file():
            report = job_root / "task_report.csv"
        if not report.is_file():
            messagebox.showinfo(
                "暂无报告", f"该任务尚未生成报告，可能仍在运行或来自旧版本：\n{report}", parent=self.root,
            )
            return
        self._open(report)

    def open_temporal_roads(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无成果", "尚未找到最新任务索引。", parent=self.root)
            return
        results = manifest.get("temporal_results", []) or []
        life_paths = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            path = _manifest_path(entry.get("life_shp"), latest.parent)
            if path is not None and path.is_file():
                life_paths.append(path)
        if not life_paths:
            messagebox.showinfo(
                "暂无长时序成果",
                "最新任务尚未生成 road_life.shp；请完成至少两个有效期次的完整处理。",
                parent=self.root,
            )
            return
        target = life_paths[0] if len(life_paths) == 1 else Path(str(manifest.get("job_root") or latest.parent)) / "grids"
        self._open(target)

    def _choose_temporal_item(self) -> dict[str, str] | None:
        if not self.temporal_items:
            return None
        selected_path = None
        selected_area = ""
        if hasattr(self, "result_tree"):
            selected = self.result_tree.selection()
            if selected:
                node = selected[0]
                selected_path = self.result_tree_paths.get(node)
                while node:
                    if str(node).startswith("area:"):
                        selected_area = str(node).removeprefix("area:")
                        break
                    node = self.result_tree.parent(node)
        if selected_path is not None:
            matched = next((
                item for item in self.temporal_items
                if Path(item["path"]).resolve() == selected_path.resolve()
            ), None)
            if matched is not None:
                return matched
        region_candidates = [
            selected_area, self.data_region.get().strip(), self.stage_region.get().strip(),
        ]
        for region in region_candidates:
            matches = [item for item in self.temporal_items if item.get("grid") == region]
            if len(matches) == 1:
                return matches[0]
        if len(self.temporal_items) == 1:
            return self.temporal_items[0]

        chooser = Toplevel(self.root)
        chooser.title("选择长时序成果区域")
        configure_window_geometry(chooser, base_width=560, base_height=190, min_width=480, min_height=160)
        chooser.transient(self.root)
        chooser.grab_set()
        ttk.Label(chooser, text="存在多个区域，请选择要浏览的 road_life：", padding=(18, 18, 18, 8)).pack(anchor="w")
        labels = [item["label"] for item in self.temporal_items]
        choice = StringVar(value=labels[0])
        combo = ttk.Combobox(chooser, textvariable=choice, values=labels, state="readonly")
        combo.pack(fill=X, padx=18)
        result: list[dict[str, str]] = []
        actions = ttk.Frame(chooser, padding=(18, 14))
        actions.pack(fill=X)
        def accept() -> None:
            result.append(self.temporal_items[labels.index(choice.get())])
            chooser.destroy()
        ttk.Button(actions, text="打开", style="Primary.TButton", command=accept).pack(side=RIGHT)
        ttk.Button(actions, text="取消", command=chooser.destroy).pack(side=RIGHT, padx=(0, 8))
        chooser.protocol("WM_DELETE_WINDOW", chooser.destroy)
        self.root.wait_window(chooser)
        return result[0] if result else None

    def open_temporal_attribute_table(self) -> None:
        self._refresh_result_availability()
        if not self.temporal_items:
            messagebox.showinfo("暂无长时序属性表", "已有任务中没有可读取的 road_life.shp。", parent=self.root)
            return
        item = self._choose_temporal_item()
        if item is None:
            return
        try:
            frame = self.project_manager.read_temporal_attributes(item["path"])
        except Exception as exc:
            messagebox.showerror("属性表读取失败", str(exc), parent=self.root)
            return
        window = Toplevel(self.root)
        window.title("长时序道路属性表 · road_life.shp")
        configure_window_geometry(window, base_width=1400, base_height=760, min_width=980, min_height=560)
        window.configure(background=UI["page"])
        header = ttk.Frame(window, padding=(12, 8))
        header.pack(fill=X)
        title_area = ttk.Frame(header)
        title_area.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(title_area, text="长时序道路属性表", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_area, text=f"{item['label']}  ·  road_life.shp",
            style="HeaderMeta.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(header, text=f"共 {len(frame)} 条道路").pack(side=RIGHT)

        filters = ttk.Frame(window, padding=(12, 8))
        filters.pack(fill=X, padx=14, pady=(14, 10))
        search_var = StringVar()
        ttk.Label(filters, text="道路筛选", style="CardTitle.TLabel").pack(side=LEFT, padx=(0, 18))
        search_entry = ttk.Entry(filters, textvariable=search_var, width=34)
        search_entry.pack(side=LEFT)
        ttk.Label(
            filters, text="输入 road_id、状态或任意字段值", style="CardMuted.TLabel",
        ).pack(side=LEFT, padx=(12, 0))
        match_var = StringVar(value=f"显示 {len(frame)} 条")
        ttk.Label(filters, textvariable=match_var, style="CardMuted.TLabel").pack(side=RIGHT, padx=(12, 0))
        ttk.Button(filters, text="清除筛选", style="Secondary.TButton", command=lambda: search_var.set("")).pack(side=RIGHT)
        pager = TemporalAttributePager(frame)
        columns = pager.columns
        table_frame = ttk.Frame(window, padding=(14, 0, 14, 0), style="Page.TFrame")
        table_frame.pack(fill=BOTH, expand=True)
        table = ttk.Treeview(table_frame, columns=columns, show="headings", style="Data.Treeview")
        ybar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
        xbar = ttk.Scrollbar(table_frame, orient="horizontal", command=table.xview)
        table.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        table.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        compact_fields = {"present", "event", "max_conf", "min_conf", "period_count"}
        wide_fields = {"road_id", "grid_id", "from_node", "to_node"}
        for name in columns:
            table.heading(name, text=name, anchor="w")
            if name in wide_fields:
                width = 150
            elif name in compact_fields or re.match(r"^[SWCE]\d", name):
                width = 86
            else:
                width = max(105, min(145, len(name) * 12 + 28))
            table.column(name, width=width, minwidth=70, stretch=False, anchor="w")
        table.tag_configure("odd", background="#F7F4ED")
        table.tag_configure("review", foreground=UI["amber"])

        page_var = StringVar()
        debounce_after_id = None

        def fill() -> None:
            table.delete(*table.get_children())
            shown = 0
            for row in pager.page_frame().itertuples(index=False, name=None):
                values = ["" if value is None else str(value) for value in row]
                state_column = "review_state" if "review_state" in columns else "life_state"
                state = values[columns.index(state_column)].casefold() if state_column in columns else ""
                tags = ("review",) if "review" in state or "uncertain" in state else (("odd",) if shown % 2 else ())
                table.insert("", END, values=values, tags=tags)
                shown += 1
            start = pager.page_index * pager.page_size + (1 if pager.match_count else 0)
            stop = pager.page_index * pager.page_size + shown
            match_var.set(f"显示 {start}–{stop} / {pager.match_count} 条")
            page_var.set(f"第 {pager.page_index + 1} / {pager.page_count} 页")
            previous_button.state(["!disabled"] if pager.page_index > 0 else ["disabled"])
            next_button.state(["!disabled"] if pager.page_index + 1 < pager.page_count else ["disabled"])

        def apply_search() -> None:
            nonlocal debounce_after_id
            debounce_after_id = None
            pager.set_query(search_var.get())
            fill()

        def schedule_search(*_args) -> None:
            nonlocal debounce_after_id
            if debounce_after_id is not None:
                window.after_cancel(debounce_after_id)
            debounce_after_id = window.after(300, apply_search)

        def change_page(delta: int) -> None:
            pager.set_page(pager.page_index + delta)
            fill()

        search_var.trace_add("write", schedule_search)
        footer = ttk.Frame(window, padding=(16, 8), style="Footer.TFrame")
        footer.pack(fill=X)
        ttk.Label(
            footer, text="横向滚动查看全部时序字段  ·  搜索停止输入 300 ms 后执行",
            style="FooterStatus.TLabel",
        ).pack(side=LEFT)
        next_button = ttk.Button(footer, text="下一页", command=lambda: change_page(1))
        next_button.pack(side=RIGHT)
        ttk.Label(footer, textvariable=page_var, style="FooterStatus.TLabel").pack(side=RIGHT, padx=12)
        previous_button = ttk.Button(footer, text="上一页", command=lambda: change_page(-1))
        previous_button.pack(side=RIGHT)
        fill()
        search_entry.focus_set()

    def export_diagnostics(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无任务", "尚未找到可导出的最新任务索引。", parent=self.root)
            return
        destination = filedialog.asksaveasfilename(
            parent=self.root, title="导出诊断包", defaultextension=".zip",
            initialfile=f"SAMRoad_诊断包_{manifest.get('run_id', 'latest')}.zip",
            filetypes=(("ZIP 压缩包", "*.zip"),),
        )
        if not destination:
            return
        job_root = Path(str(manifest.get("job_root") or latest.parent)).expanduser()
        candidates = [
            latest,
            job_root / "job_state.json",
            job_root / "pipeline_result.json",
            job_root / "task_report.json",
            job_root / "task_report.csv",
        ]
        layout = (
            ProjectLayout.from_project(self.project_root_path, self.vars["output_root"].get().strip())
            if self.project_root_path else ProjectLayout.from_output(self.vars["output_root"].get().strip())
        )
        log_path = layout.logs_root / f"{manifest.get('run_id')}.log"
        candidates.append(log_path)
        diagnostics = {
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version,
            "platform": platform.platform(),
            "project_root": str(ROOT),
            "run_id": manifest.get("run_id"),
            "status": manifest.get("status"),
            "checkpoint": self.vars["checkpoint"].get(),
            "config": self.vars["config"].get(),
            "note": "诊断包包含任务索引、报告、日志和本机路径，不包含原始影像或模型。",
        }
        try:
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("diagnostics.json", json.dumps(diagnostics, ensure_ascii=False, indent=2))
                seen = set()
                for path in candidates:
                    try:
                        resolved = path.resolve()
                    except OSError:
                        continue
                    if not resolved.is_file() or resolved in seen:
                        continue
                    seen.add(resolved)
                    archive.write(resolved, arcname=f"task/{resolved.name}")
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)
            return
        self.status.set(f"诊断包已导出：{destination}")
