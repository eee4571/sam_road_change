from __future__ import annotations
import json
import threading
import time
from pathlib import Path
from tkinter import END, filedialog, messagebox, simpledialog

from input_catalog import period_order_manifest, period_sort_key
from app.project_manager import natural_key
from app.result_publisher import RESULT_DIRECTORY_NAME
from tkinter import BOTH, LEFT, RIGHT, X, StringVar
from tkinter import ttk

from .common_widgets import (
    LAYOUT_METRICS, PathDisplay, Tooltip, attach_treeview_tooltip,
    bind_dynamic_wrap,
)


DEFAULT_TRUTH_FIELD_CONFIG = {
    "truth_type_field": "BHBM",
    "truth_value_map": {"added": "2", "width_changed": "3", "removed": "4"},
}


def _fit_tree_height(tree: ttk.Treeview, item_count: int, minimum: int, maximum: int) -> None:
    tree.configure(height=max(minimum, min(item_count, maximum)))


class DataPage:
    def _build_data_page(self, page: ttk.Frame) -> None:
        self.data_body = page
        project_card = ttk.LabelFrame(page, text="项目与数据管理", padding=LAYOUT_METRICS["card_padding"])
        project_card.pack(fill=X)
        project_actions = ttk.Frame(project_card)
        project_actions.pack(fill=X, pady=(0, LAYOUT_METRICS["module_gap"]))
        ttk.Button(project_actions, text="新建项目", command=self.create_project_folder).pack(side=LEFT)
        ttk.Button(project_actions, text="打开项目", command=self.import_project_folder).pack(side=LEFT, padx=(5, 0))
        ttk.Button(project_actions, text="打开项目文件夹", command=self.open_project_folder).pack(side=LEFT, padx=(5, 0))
        project_meta = ttk.Frame(project_card)
        project_meta.pack(fill=X)
        self.project_name_display = StringVar(value="尚未打开项目")
        self.project_path_display = StringVar(value="尚未选择项目目录")
        ttk.Label(project_meta, text="项目路径：", width=LAYOUT_METRICS["form_label_width"]).grid(row=0, column=0, sticky="w")
        self.project_path_field = PathDisplay(project_meta, textvariable=self.project_path_display, width=1)
        self.project_path_field.grid(row=0, column=1, sticky="ew")
        ttk.Label(project_meta, text="外部数据源：", width=LAYOUT_METRICS["form_label_width"]).grid(row=1, column=0, sticky="w", pady=(LAYOUT_METRICS["form_gap"], 0))
        self.data_source_field = PathDisplay(project_meta, textvariable=self.data_source_display, width=1)
        self.data_source_field.grid(row=1, column=1, sticky="ew", pady=(LAYOUT_METRICS["form_gap"], 0))
        ttk.Label(project_meta, text="扫描状态：", width=LAYOUT_METRICS["form_label_width"]).grid(row=2, column=0, sticky="w", pady=(LAYOUT_METRICS["form_gap"], 0))
        self.project_scan_label = ttk.Label(project_meta, textvariable=self.project_scan_summary, width=1)
        self.project_scan_label.grid(row=2, column=1, sticky="ew", pady=(LAYOUT_METRICS["form_gap"], 0))
        bind_dynamic_wrap(self.project_scan_label, project_meta, minimum=220, padding=120)
        project_meta.grid_columnconfigure(1, weight=1)
        quick_actions = ttk.Frame(project_card)
        quick_actions.pack(fill=X, pady=(LAYOUT_METRICS["module_gap"], 0))
        ttk.Button(quick_actions, text="连接数据源", command=self.connect_data_source).pack(side=LEFT)
        ttk.Button(quick_actions, text="重新定位", command=self.relocate_data_source).pack(side=LEFT, padx=(5, 0))
        self.scan_data_button = ttk.Button(
            quick_actions, text="重新扫描",
            command=self.scan_data_sources,
        )
        self.scan_data_button.pack(side=LEFT, padx=(5, 0))
        self.cancel_scan_button = ttk.Button(
            quick_actions, text="取消扫描",
            command=self.cancel_data_source_scan,
        )
        self.cancel_scan_button.pack(side=LEFT, padx=(5, 0))
        self.cancel_scan_button.state(["disabled"])
        self.input_summary = StringVar(value="请选择项目目录；如需手工指定数据，可展开高级设置。")
        config_card = ttk.LabelFrame(page, text="区域数据配置", padding=LAYOUT_METRICS["card_padding"])
        config_card.pack(fill=X, pady=(LAYOUT_METRICS["section_gap"], 0))
        config_card.grid_columnconfigure(0, weight=1)
        region_row = ttk.Frame(config_card)
        region_row.grid(row=0, column=0, sticky="ew", pady=LAYOUT_METRICS["form_gap"])
        ttk.Label(region_row, text="区域：", width=LAYOUT_METRICS["form_label_width"]).pack(side=LEFT)
        self.project_region_combo = ttk.Combobox(region_row, textvariable=self.data_region, state="readonly", width=26)
        self.project_region_combo.pack(side=LEFT)
        self.project_region_combo.bind("<<ComboboxSelected>>", self._project_region_changed)
        ttk.Button(region_row, text="添加区域", command=self.add_project_region).pack(side=LEFT, padx=(5, 0))
        ttk.Button(region_row, text="移除区域", command=self.remove_project_region).pack(side=LEFT, padx=(5, 0))
        area_row = ttk.Frame(config_card)
        area_row.grid(row=1, column=0, sticky="ew", pady=LAYOUT_METRICS["form_gap"])
        ttk.Label(area_row, text="验证区 SHP：", width=LAYOUT_METRICS["form_label_width"]).pack(side=LEFT)
        self.project_validation_field = PathDisplay(area_row, textvariable=self.project_validation_path)
        self.project_validation_field.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(area_row, text="选择...", command=self.replace_project_validation_area).pack(side=LEFT, padx=(5, 0))
        self.project_config_container = ttk.Frame(config_card)
        self.project_config_container.grid(
            row=2, column=0, sticky="ew",
            pady=(LAYOUT_METRICS["module_gap"], LAYOUT_METRICS["form_gap"]),
        )
        config_actions = ttk.Frame(config_card)
        config_actions.grid(row=3, column=0, sticky="ew")
        ttk.Button(config_actions, text="导入配置...", command=self.load_task_config).pack(side=LEFT)
        ttk.Button(config_actions, text="导出配置...", command=self.save_task_config).pack(side=LEFT, padx=(5, 0))
        ttk.Button(config_actions, text="检查数据", style="Primary.TButton", command=self.preflight_inputs).pack(side=RIGHT)
        self._refresh_project_config_panel()

        self.manual_toggle = ttk.Button(
            page, text="高级设置...", command=self._toggle_manual_inputs,
        )
        self.manual_toggle.pack(anchor="w", pady=(6, 3))
        self.manual_frame = ttk.LabelFrame(page, text="高级设置", padding=LAYOUT_METRICS["card_padding"])
        ttk.Radiobutton(
            self.manual_frame, text="验证区项目", variable=self.vars["mode"], value="validation",
            command=self._refresh_input_summary,
        ).pack(anchor="w")
        self._field(self.manual_frame, "验证区 SHP", "validation_area", "shp")
        ttk.Label(self.manual_frame, text="影像期次（每期选择一个内含影像路径的 TXT）", style="Hint.TLabel").pack(anchor="w", pady=(7, 2))
        self.period_container = ttk.Frame(self.manual_frame)
        self.period_container.pack(fill=X)
        period_actions = ttk.Frame(self.manual_frame)
        period_actions.pack(fill=X, pady=(4, 5))
        ttk.Button(period_actions, text="添加期次", command=self._add_period_row).pack(side=LEFT)
        self.truth_container = ttk.Frame(self.manual_frame)
        self._add_period_row("2021")
        self._add_period_row("2022")
        self.grid_toggle = ttk.Button(
            self.manual_frame, text="兼容旧版多格网目录...", command=self._toggle_grid_options,
        )
        self.grid_toggle.pack(anchor="w", pady=(9, 2))
        self.grid_options = ttk.Frame(self.manual_frame)
        ttk.Radiobutton(
            self.grid_options, text="使用多格网目录", variable=self.vars["mode"], value="grid",
            command=self._refresh_input_summary,
        ).pack(anchor="w")
        self._field(self.grid_options, "格网数据根目录", "source_root", "dir")
        config_actions = ttk.Frame(self.manual_frame)
        config_actions.pack(fill=X, pady=(12, 0))
        ttk.Button(config_actions, text="加载配置…", style="Compact.TButton", command=self.load_task_config).pack(side=LEFT)
        ttk.Button(config_actions, text="导出配置…", style="Compact.TButton", command=self.save_task_config).pack(side=LEFT, padx=8)

        summary = self.step_summaries[0]
        summary_box = ttk.LabelFrame(summary, text="输入摘要", padding=LAYOUT_METRICS["card_padding"])
        summary_box.pack(fill=BOTH, expand=True)
        self.data_summary_sources = StringVar(value="0 个")
        self.data_summary_areas = StringVar(value="0 个")
        self.data_summary_periods = StringVar(value="0 个")
        self.data_summary_truths = StringVar(value="0 组")
        self.data_summary_encoding = StringVar(value="自动检测")
        for index, (label, variable) in enumerate((
            ("已连接数据源：", self.data_summary_sources),
            ("验证区：", self.data_summary_areas),
            ("影像期次：", self.data_summary_periods),
            ("变化真值：", self.data_summary_truths),
            ("TXT 编码：", self.data_summary_encoding),
        )):
            row, column = index % 3, (index // 3) * 2
            ttk.Label(summary_box, text=label, width=11, style="Metric.TLabel").grid(row=row, column=column, sticky="nw", pady=1)
            ttk.Label(summary_box, textvariable=variable, style="Metric.TLabel").grid(row=row, column=column + 1, sticky="nw", pady=1)
        summary_box.grid_columnconfigure(1, weight=1)
        summary_box.grid_columnconfigure(3, weight=1)
        ttk.Separator(summary_box).grid(row=3, column=0, columnspan=4, sticky="ew", pady=3)
        input_hint = ttk.Label(
            summary_box,
            text="提示：开始处理前会再次检查影像范围、CRS、波段及数据有效性。",
            style="CardMuted.TLabel",
        )
        input_hint.grid(row=4, column=0, columnspan=4, sticky="nw")
        bind_dynamic_wrap(input_hint, summary_box, minimum=220, padding=20)
        self._refresh_data_summary()

    def _refresh_data_summary(self) -> None:
        if not hasattr(self, "data_summary_sources"):
            return
        self.data_summary_sources.set(f"{len(self.project_data_sources)} 个")
        self.data_summary_areas.set(f"{len(self.project_validation_areas)} 个")
        self.data_summary_periods.set(
            f"{sum(len(rows) for rows in self.project_area_periods.values())} 个"
        )
        self.data_summary_truths.set(f"{len(self.project_area_truths)} 组")
        encodings = sorted({value for value in self.project_txt_encodings.values() if value})
        self.data_summary_encoding.set("、".join(encodings) if encodings else "自动检测")

    def _selected_project_region(self, scope: str = "data") -> str:
        variable = self.stage_region if scope == "stage" else self.data_region
        selected = variable.get().strip()
        names = [name for name, _path in self.project_validation_areas]
        if selected in names:
            return selected
        return names[0] if names else ""

    @staticmethod
    def _normalize_truth_field_config(config: object) -> dict[str, object]:
        source = config if isinstance(config, dict) else {}
        raw_map = source.get("truth_value_map") if isinstance(source.get("truth_value_map"), dict) else {}
        defaults = DEFAULT_TRUTH_FIELD_CONFIG["truth_value_map"]
        return {
            "truth_type_field": str(source.get("truth_type_field") or "BHBM").strip() or "BHBM",
            "truth_value_map": {
                key: str(raw_map.get(key) or defaults[key]).strip()
                for key in ("added", "width_changed", "removed")
            },
        }

    def _truth_field_config_for_area(self, area: str) -> dict[str, object]:
        return self._normalize_truth_field_config(
            self.project_area_truth_field_configs.get(str(area), {}),
        )

    def _store_truth_field_controls(self, *, save: bool = False) -> None:
        area = str(getattr(self, "_truth_field_config_area", "") or "").strip()
        if not area:
            return
        self.project_area_truth_field_configs[area] = {
            "truth_type_field": self.evaluation_type_field.get().strip() or "BHBM",
            "truth_value_map": {
                "added": self.evaluation_added_value.get().strip(),
                "width_changed": self.evaluation_width_changed_value.get().strip(),
                "removed": self.evaluation_removed_value.get().strip(),
            },
        }
        if save:
            self._save_project_config()

    def _load_truth_field_controls(self, area: str) -> None:
        config = self._truth_field_config_for_area(area)
        value_map = config["truth_value_map"]
        self._truth_field_config_area = area
        self.evaluation_type_field.set(str(config["truth_type_field"]))
        self.evaluation_added_value.set(str(value_map["added"]))
        self.evaluation_width_changed_value.set(str(value_map["width_changed"]))
        self.evaluation_removed_value.set(str(value_map["removed"]))
        self._evaluation_truth_field_summary = {}
        if hasattr(self, "evaluation_type_field_combo"):
            self.evaluation_type_field_combo.configure(values=())
        for combo in getattr(self, "evaluation_value_combos", {}).values():
            combo.configure(values=())
        self.evaluation_type_field_status.set(
            f"区域“{area}”使用字段“{config['truth_type_field']}”；"
            f"新增={value_map['added']}，宽度变化={value_map['width_changed']}，灭失={value_map['removed']}。"
            if area else "请选择区域后配置真值字段。"
        )

    def _project_truth_type_field_changed(self, event=None) -> None:
        self._evaluation_type_field_changed(event)
        self._store_truth_field_controls(save=True)

    def _project_truth_value_mapping_changed(self, _event=None) -> None:
        self._store_truth_field_controls(save=True)

    def _current_region_truth_path(self) -> str:
        area = self._selected_project_region()
        pair = self._selected_truth_pair() if hasattr(self, "project_truth_tree") else None
        if pair is not None:
            selected = next((
                path for region, before, after, path in self.project_area_truths
                if (region, before, after) == (area, pair[0], pair[1])
            ), "")
            if selected:
                return selected
        return next((path for region, _before, _after, path in self.project_area_truths if region == area), "")

    def _inspect_current_region_truth_fields(self) -> None:
        truth = self._current_region_truth_path()
        self._refresh_evaluation_truth_fields(show_error=True, truth_path=truth)
        self._store_truth_field_controls(save=True)

    def _ensure_project_config_tables(self) -> None:
        if hasattr(self, "project_period_tree"):
            return
        self.project_config_container.grid_columnconfigure(0, weight=1)
        ttk.Label(self.project_config_container, text="多时相影像").grid(row=0, column=0, sticky="w", pady=(0, 3))
        period_frame = ttk.Frame(self.project_config_container)
        period_frame.grid(row=1, column=0, sticky="ew")
        self.project_period_tree = ttk.Treeview(
            period_frame, columns=("period", "path", "encoding", "status"),
            show="headings", height=2, style="Data.Treeview",
        )
        for column, title, width, stretch in (
            ("period", "期次", 80, False), ("path", "影像路径 TXT", 520, True),
            ("encoding", "编码", 75, False), ("status", "状态", 85, False),
        ):
            self.project_period_tree.heading(column, text=title, anchor="w")
            self.project_period_tree.column(column, width=width, minwidth=70, stretch=stretch, anchor="w")
        period_scroll = ttk.Scrollbar(period_frame, orient="vertical", command=self.project_period_tree.yview)
        period_xscroll = ttk.Scrollbar(period_frame, orient="horizontal", command=self.project_period_tree.xview)
        self.project_period_tree.configure(yscrollcommand=period_scroll.set, xscrollcommand=period_xscroll.set)
        self.project_period_tree.grid(row=0, column=0, sticky="nsew")
        period_scroll.grid(row=0, column=1, sticky="ns")
        period_xscroll.grid(row=1, column=0, sticky="ew")
        period_frame.grid_columnconfigure(0, weight=1)
        attach_treeview_tooltip(self.project_period_tree)
        self.project_period_tree.bind("<Double-1>", lambda _event: self.replace_selected_project_period())
        period_actions = ttk.Frame(self.project_config_container)
        period_actions.grid(row=2, column=0, sticky="ew", pady=(3, LAYOUT_METRICS["module_gap"]))
        self.add_project_period_button = ttk.Button(period_actions, text="添加期次", style="Compact.TButton", command=self.add_project_period)
        self.add_project_period_button.pack(side=LEFT)
        ttk.Button(period_actions, text="更换路径", style="Compact.TButton", command=self.replace_selected_project_period).pack(side=LEFT, padx=(4, 0))
        ttk.Button(period_actions, text="移除期次", style="Compact.TButton", command=self.remove_selected_project_period).pack(side=LEFT, padx=(4, 0))
        ttk.Button(period_actions, text="指定 TXT 编码", style="Compact.TButton", command=self.set_selected_txt_encoding).pack(side=LEFT, padx=(4, 0))

        ttk.Label(self.project_config_container, text="变化真值（可选）").grid(row=3, column=0, sticky="w", pady=(0, 3))
        truth_frame = ttk.Frame(self.project_config_container)
        truth_frame.grid(row=4, column=0, sticky="ew")
        self.project_truth_tree = ttk.Treeview(
            truth_frame, columns=("pair", "path", "status"), show="headings",
            height=2, style="Data.Treeview",
        )
        self._project_truth_pair_by_iid: dict[str, tuple[str, str]] = {}
        for column, title, width, stretch in (
            ("pair", "变化对", 125, False), ("path", "真值 SHP", 560, True),
            ("status", "状态", 85, False),
        ):
            self.project_truth_tree.heading(column, text=title, anchor="w")
            self.project_truth_tree.column(column, width=width, minwidth=70, stretch=stretch, anchor="w")
        truth_scroll = ttk.Scrollbar(truth_frame, orient="vertical", command=self.project_truth_tree.yview)
        truth_xscroll = ttk.Scrollbar(truth_frame, orient="horizontal", command=self.project_truth_tree.xview)
        self.project_truth_tree.configure(yscrollcommand=truth_scroll.set, xscrollcommand=truth_xscroll.set)
        self.project_truth_tree.grid(row=0, column=0, sticky="nsew")
        truth_scroll.grid(row=0, column=1, sticky="ns")
        truth_xscroll.grid(row=1, column=0, sticky="ew")
        truth_frame.grid_columnconfigure(0, weight=1)
        attach_treeview_tooltip(self.project_truth_tree)
        self.project_truth_tree.bind("<Double-1>", lambda _event: self.set_selected_project_truth())
        truth_actions = ttk.Frame(self.project_config_container)
        truth_actions.grid(row=5, column=0, sticky="ew", pady=(3, LAYOUT_METRICS["module_gap"]))
        ttk.Button(truth_actions, text="选择真值", style="Compact.TButton", command=self.set_selected_project_truth).pack(side=LEFT)
        ttk.Button(truth_actions, text="移除真值", style="Compact.TButton", command=self.remove_selected_project_truth).pack(side=LEFT, padx=(4, 0))

        ttk.Label(self.project_config_container, text="真值字段配置（当前区域）").grid(
            row=6, column=0, sticky="w", pady=(0, 3),
        )
        field_row = ttk.Frame(self.project_config_container)
        field_row.grid(row=7, column=0, sticky="ew")
        ttk.Label(field_row, text="变化类型字段", width=LAYOUT_METRICS["form_label_width"]).pack(side=LEFT)
        self.evaluation_type_field_combo = ttk.Combobox(
            field_row, textvariable=self.evaluation_type_field, width=18, state="normal",
        )
        self.evaluation_type_field_combo.pack(side=LEFT)
        self.evaluation_type_field_combo.bind(
            "<<ComboboxSelected>>", self._project_truth_type_field_changed,
        )
        self.evaluation_type_field_combo.bind(
            "<FocusOut>", self._project_truth_type_field_changed,
        )
        ttk.Button(
            field_row, text="检查当前真值字段", style="Compact.TButton",
            command=self._inspect_current_region_truth_fields,
        ).pack(side=LEFT, padx=(6, 0))
        status_label = ttk.Label(
            self.project_config_container, textvariable=self.evaluation_type_field_status,
            style="CardMuted.TLabel",
        )
        status_label.grid(row=8, column=0, sticky="ew", pady=(4, 0))
        bind_dynamic_wrap(status_label, self.project_config_container, minimum=260, padding=20)
        value_row = ttk.Frame(self.project_config_container)
        value_row.grid(row=9, column=0, sticky="ew", pady=(5, 0))
        ttk.Label(value_row, text="字段值对应关系", width=LAYOUT_METRICS["form_label_width"]).pack(side=LEFT)
        self.evaluation_value_combos = {}
        for key, label, variable in (
            ("added", "新增", self.evaluation_added_value),
            ("width_changed", "宽度变化", self.evaluation_width_changed_value),
            ("removed", "灭失", self.evaluation_removed_value),
        ):
            ttk.Label(value_row, text=f"{label}=").pack(side=LEFT, padx=((8 if key != "added" else 0), 3))
            combo = ttk.Combobox(value_row, textvariable=variable, width=9, state="normal")
            combo.pack(side=LEFT)
            combo.bind("<<ComboboxSelected>>", self._project_truth_value_mapping_changed)
            combo.bind("<FocusOut>", self._project_truth_value_mapping_changed)
            self.evaluation_value_combos[key] = combo

    @staticmethod
    def _selected_tree_iid(tree) -> str:
        selected = tree.selection()
        return str(selected[0]) if selected else ""

    def replace_selected_project_period(self) -> None:
        period = self._selected_tree_iid(self.project_period_tree)
        if period:
            self.replace_project_period_source(period)

    def remove_selected_project_period(self) -> None:
        period = self._selected_tree_iid(self.project_period_tree)
        if period:
            self.remove_project_period(period)

    def _selected_truth_pair(self) -> tuple[str, str] | None:
        iid = self._selected_tree_iid(self.project_truth_tree)
        if not iid:
            return None
        pair = getattr(self, "_project_truth_pair_by_iid", {}).get(iid)
        if pair is not None:
            return pair
        if not iid.startswith("truth:"):
            return None
        parts = iid.split(":", 2)
        return (parts[1], parts[2]) if len(parts) == 3 and all(parts[1:]) else None

    def set_selected_project_truth(self) -> None:
        pair = self._selected_truth_pair()
        if pair is None:
            messagebox.showinfo("未选择变化对", "请先在变化真值列表中选择一个变化对。", parent=self.root)
            return
        self.set_project_truth(*pair)

    def remove_selected_project_truth(self) -> None:
        pair = self._selected_truth_pair()
        if pair is None:
            messagebox.showinfo("未选择变化对", "请先在变化真值列表中选择一个变化对。", parent=self.root)
            return
        self.remove_project_truth(*pair)

    def set_selected_txt_encoding(self) -> None:
        period = self._selected_tree_iid(self.project_period_tree)
        region = self._selected_project_region()
        source = next((path for name, path in self.project_area_periods.get(region, []) if name == period), "")
        if not source:
            return
        current = self.project_txt_encodings.get(str(Path(source).resolve()), "auto")
        value = simpledialog.askstring(
            "指定 TXT 编码", "输入编码名称（auto、utf-8、gbk、gb18030、utf-16、cp932、cp950）：",
            initialvalue=current, parent=self.root,
        )
        if value is None:
            return
        normalized = value.strip().casefold()
        key = str(Path(source).resolve())
        if normalized in {"", "auto"}:
            self.project_txt_encodings.pop(key, None)
        else:
            try:
                "".encode(normalized)
            except LookupError:
                messagebox.showerror("编码不可用", f"Python 不支持编码：{normalized}", parent=self.root)
                return
            self.project_txt_encodings[key] = normalized
        self._save_project_config()
        self._refresh_project_config_panel()

    def _refresh_project_config_panel(self) -> None:
        if not hasattr(self, "project_config_container"):
            return
        self._ensure_project_config_tables()
        names = [name for name, _path in self.project_validation_areas]
        self.project_region_combo.configure(values=names)
        if self.data_region.get() not in names:
            self.data_region.set(names[0] if names else "")
        region = self._selected_project_region()
        area_path = next((path for name, path in self.project_validation_areas if name == region), "")
        self.project_validation_path.set(area_path or "尚未选择验证区。")
        for tree in (self.project_period_tree, self.project_truth_tree):
            children = tree.get_children()
            if children:
                tree.delete(*children)
        self._project_truth_pair_by_iid = {}
        if not region:
            if hasattr(self, "add_project_period_button"):
                self.add_project_period_button.state(["disabled"])
            _fit_tree_height(self.project_period_tree, 0, 2, 5)
            _fit_tree_height(self.project_truth_tree, 0, 2, 4)
            self._load_truth_field_controls("")
            self._refresh_stage_selectors()
            self._refresh_data_summary()
            self._schedule_content_layout()
            return
        if hasattr(self, "add_project_period_button"):
            self.add_project_period_button.state(["!disabled"])
        rows = sorted(self.project_area_periods.get(region, []), key=lambda row: period_sort_key(row[0]))
        for period, source in rows:
            resolved = str(Path(source).expanduser().resolve())
            encoding = self.project_txt_encodings.get(resolved, "自动")
            status = "已配置" if Path(source).is_file() else "文件缺失"
            self.project_period_tree.insert("", END, iid=period, values=(period, source, encoding, status))
        truth_map = {
            (area, before, after): path
            for area, before, after, path in self.project_area_truths
        }
        pairs = [(before[0], after[0]) for before, after in zip(rows, rows[1:])]
        for before, after in pairs:
            truth = truth_map.get((region, before, after), "")
            iid = f"truth:{before}:{after}"
            self._project_truth_pair_by_iid[iid] = (before, after)
            self.project_truth_tree.insert(
                "", END, iid=iid, values=(f"{before} - {after}", truth or "", "已配置" if truth else "未配置"),
            )
        _fit_tree_height(self.project_period_tree, len(rows), 2, 5)
        _fit_tree_height(self.project_truth_tree, len(pairs), 2, 4)
        self._load_truth_field_controls(region)
        self._refresh_stage_selectors()
        self._refresh_data_summary()
        self._schedule_content_layout()

    def _refresh_stage_selectors(self) -> None:
        if not hasattr(self, "stage_region_combo"):
            return
        names = [name for name, _path in self.project_validation_areas]
        self.stage_region_combo.configure(values=names)
        if hasattr(self, "stage_change_region_combo"):
            self.stage_change_region_combo.configure(values=names)
        if self.stage_region.get() not in names:
            self.stage_region.set(names[0] if names else "")
        region = self._selected_project_region("stage")
        rows = sorted(self.project_area_periods.get(region, []), key=lambda row: period_sort_key(row[0]))
        periods = [period for period, _source in rows]
        pairs = [f"{before} → {after}" for before, after in zip(periods, periods[1:])]
        self.stage_period_combo.configure(values=periods)
        self.stage_pair_combo.configure(values=pairs)
        if self.project_period.get() not in periods:
            self.project_period.set(periods[0] if periods else "")
        if self.project_change_pair.get() not in pairs:
            self.project_change_pair.set(pairs[0] if pairs else "")
        self._stage_period_changed()

    def _project_region_changed(self, _event=None) -> None:
        self._store_truth_field_controls(save=True)
        self._refresh_project_config_panel()

    def _stage_region_changed(self, _event=None) -> None:
        self._refresh_stage_selectors()

    def _project_payload(self) -> dict:
        self._store_truth_field_controls()
        payload = dict(self.project_config)
        payload.update({
            "version": 3,
            "project_root": self.project_root_path,
            "external_data_sources": list(dict.fromkeys(self.project_data_sources)),
            "external_scan_cache": self.project_scan_cache,
            "txt_encodings": self.project_txt_encodings,
            "path_relocations": self.project_path_relocations,
            "validation_areas": [list(row) for row in self.project_validation_areas],
            "area_periods": {
                area: [list(row) for row in rows]
                for area, rows in self.project_area_periods.items()
            },
            "area_truths": [list(row) for row in self.project_area_truths],
            "area_truth_field_configs": self.project_area_truth_field_configs,
            "unmapped_candidates": self.project_candidates,
            "output_root": self.vars["output_root"].get().strip(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return payload

    def _save_project_config(self, *, notify: bool = False) -> bool:
        if not self.project_root_path:
            return False
        try:
            path = self.project_manager.save_config(self.project_root_path, self._project_payload())
            self.project_config = self.project_manager.read_config(self.project_root_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            messagebox.showerror("项目配置保存失败", str(exc), parent=self.root)
            return False
        if notify:
            self.status.set(f"项目配置已保存：{path}")
        return True

    def _consume_candidate(self, path: str) -> None:
        resolved = str(Path(path).expanduser().resolve())
        for kind in self.project_candidates:
            self.project_candidates[kind] = [value for value in self.project_candidates[kind] if str(Path(value).expanduser().resolve()) != resolved]

    def _apply_discovered_project(self, project: dict, *, merge: bool = False) -> None:
        discovered_areas = list(project.get("validation_areas") or [])
        raw_truths = project.get("area_truths") or {}
        if isinstance(raw_truths, dict):
            discovered_truths = [
                (area, before, after, path)
                for (area, before, after), path in raw_truths.items()
            ]
        else:
            discovered_truths = [tuple(row) for row in raw_truths if len(row) == 4]
        discovered_periods = {
            str(area): [(str(period), str(source)) for period, source in rows]
            for area, rows in (project.get("area_periods") or {}).items()
        }
        if not merge:
            self.project_validation_areas = discovered_areas
            self.project_area_truths = discovered_truths
            self.project_area_periods = discovered_periods
            return
        areas = {name: path for name, path in self.project_validation_areas}
        areas.update({str(name): str(path) for name, path in discovered_areas})
        self.project_validation_areas = sorted(areas.items(), key=lambda row: natural_key(row[0]))
        self.project_area_periods.update(discovered_periods)
        truths = {(area, before, after): path for area, before, after, path in self.project_area_truths}
        truths.update({(area, before, after): path for area, before, after, path in discovered_truths})
        self.project_area_truths = [(*key, path) for key, path in truths.items()]

    def _apply_project_config(self, payload: dict) -> None:
        self._truth_field_config_area = ""
        self.project_config = dict(payload)
        self.project_data_sources = [str(Path(value).expanduser().resolve()) for value in payload.get("external_data_sources", []) if str(value).strip()]
        self.project_scan_cache = {
            str(Path(source).expanduser().resolve()): dict(record)
            for source, record in (payload.get("external_scan_cache") or {}).items()
            if isinstance(record, dict)
        }
        self.project_txt_encodings = {
            str(Path(source).expanduser().resolve()): str(encoding)
            for source, encoding in (payload.get("txt_encodings") or {}).items()
            if str(source).strip() and str(encoding).strip()
        }
        self.project_path_relocations = {
            str(old): str(new)
            for old, new in (payload.get("path_relocations") or {}).items()
            if str(old).strip() and str(new).strip()
        }
        self.project_validation_areas = [
            (str(name), str(path)) for name, path in payload.get("validation_areas", [])
        ]
        self.project_area_periods = {
            str(area): [(str(period), str(source)) for period, source in rows]
            for area, rows in (payload.get("area_periods") or {}).items()
        }
        self.project_area_truths = [
            (str(area), str(before), str(after), str(path))
            for area, before, after, path in payload.get("area_truths", [])
        ]
        self.project_area_truth_field_configs = {
            str(area): self._normalize_truth_field_config(config)
            for area, config in (payload.get("area_truth_field_configs") or {}).items()
        }
        candidates = payload.get("unmapped_candidates") or {}
        self.project_candidates = {
            "shp": [str(value) for value in candidates.get("shp", [])],
            "txt": [str(value) for value in candidates.get("txt", [])],
        }
        output_root = self.project_manager.preferred_output_root(
            self.project_root_path or payload.get("project_root") or ".",
            str(payload.get("output_root") or "").strip() or None,
        )
        self.vars["output_root"].set(str(output_root))
        active = payload.get("active_task") or {}
        if isinstance(active, dict) and str(active.get("run_id") or "").strip():
            self.vars["run_id"].set(str(active["run_id"]))
            try:
                active_manifest = self.task_manager.active_pipeline_manifest(output_root, active)
            except ValueError:
                active_manifest = None
            if active_manifest is not None:
                profile = self.task_manager.task_execution_profile(active_manifest)
                if profile:
                    self.vars["execution_profile"].set(profile)
        self.data_source_display.set("；".join(self.project_data_sources) if self.project_data_sources else "尚未连接外部数据源")
        if self.project_scan_cache:
            cached_files = sum(int(record.get("visited_files", 0) or 0) for record in self.project_scan_cache.values())
            self.project_scan_summary.set(
                f"已恢复 {len(self.project_scan_cache)} 个数据源的扫描索引（上次遍历 {cached_files} 个文件），未重新递归扫描。"
            )
            self.data_status.set("已恢复扫描缓存")

        issues = self.project_manager.path_issues(payload)
        if issues:
            self.data_status.set("存在失效路径")
            self.project_scan_summary.set(
                f"检测到 {len(issues)} 个失效输入路径；请点击“重新定位”选择新的数据目录。"
            )

    def connect_data_source(self) -> None:
        if not self.project_root_path:
            messagebox.showinfo("请先打开项目", "请先新建或打开项目文件夹，再连接外部原始数据源。", parent=self.root)
            return
        directory = filedialog.askdirectory(parent=self.root, title="连接外部原始数据源")
        if not directory:
            return
        resolved = str(Path(directory).resolve())
        is_new = resolved not in self.project_data_sources
        if is_new:
            self.project_data_sources.append(resolved)
        self.data_source_display.set("；".join(self.project_data_sources))
        self.data_status.set("已连接，正在扫描新增数据源" if is_new else "数据源已连接")
        self._save_project_config()
        if is_new or resolved not in self.project_scan_cache:
            self.scan_data_sources(sources=[resolved], force=True)
        else:
            self.status.set("该数据源已有完整扫描缓存；点击“重新扫描”可显式刷新。")

    def relocate_data_source(self) -> None:
        if not self.project_root_path:
            messagebox.showinfo("请先打开项目", "请先打开包含失效路径的项目。", parent=self.root)
            return
        payload = self._project_payload()
        issues = self.project_manager.path_issues(payload)
        missing_sources = [
            str(source) for source in self.project_data_sources
            if str(source).strip() and not Path(source).expanduser().is_dir()
        ]
        if not issues and not missing_sources:
            messagebox.showinfo("路径有效", "当前配置的外部数据路径均有效。", parent=self.root)
            return
        suggested = missing_sources[0] if missing_sources else str(Path(issues[0]["path"]).parent)
        old_value = simpledialog.askstring(
            "原数据根目录",
            "请输入需要替换的原数据根目录。目录下的验证区、期次 TXT、真值和缓存路径会一起更新：",
            initialvalue=suggested,
            parent=self.root,
        )
        if old_value is None or not old_value.strip():
            return
        new_value = filedialog.askdirectory(parent=self.root, title="选择重新定位后的数据根目录")
        if not new_value:
            return
        try:
            updated = self.project_manager.relocate_paths(payload, old_value, new_value)
        except (OSError, ValueError) as exc:
            messagebox.showerror("重新定位失败", str(exc), parent=self.root)
            return
        self._apply_project_config(updated)
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        if not self._save_project_config():
            return
        remaining = self.project_manager.path_issues(self._project_payload())
        self.data_status.set("重新定位完成" if not remaining else "仍有失效路径")
        self.project_scan_summary.set(
            "数据路径重新定位完成。" if not remaining
            else f"本次映射已保存，仍有 {len(remaining)} 个失效路径；可再次点击“重新定位”。"
        )
        resolved_new = str(Path(new_value).expanduser().resolve())
        if resolved_new in self.project_data_sources:
            self.scan_data_sources(sources=[resolved_new], force=True)

    def scan_data_sources(self, *, sources: list[str] | None = None, force: bool = True) -> None:
        if not self.project_data_sources:
            self.data_status.set("未连接数据源")
            messagebox.showinfo("未连接数据源", "请先连接一个或多个外部原始数据目录。", parent=self.root)
            return
        if self.scan_thread is not None and self.scan_thread.is_alive():
            self.status.set("数据源扫描正在进行；如需重启，请先取消当前扫描。")
            return
        requested = list(dict.fromkeys(sources or self.project_data_sources))
        event_queue = self.priority_queue
        cancel_event = threading.Event()
        cache_snapshot = dict(self.project_scan_cache)
        self.scan_cancel_event = cancel_event
        self.data_status.set("正在后台扫描")
        self.project_scan_summary.set(f"准备扫描 {len(requested)} 个数据源；窗口可继续操作。")
        if hasattr(self, "scan_data_button"):
            self.scan_data_button.state(["disabled"])
            self.cancel_scan_button.state(["!disabled"])

        def worker() -> None:
            started = time.perf_counter()
            completed: list[dict] = []
            try:
                for index, source in enumerate(requested, start=1):
                    if cancel_event.is_set():
                        event_queue.put(("scan_cancelled", {"elapsed_seconds": time.perf_counter() - started}))
                        return
                    if not force:
                        cached = cache_snapshot.get(str(Path(source).expanduser().resolve()))
                        try:
                            unchanged = cached and cached.get("signature") == self.project_manager.source_signature(source)
                        except (OSError, ValueError):
                            unchanged = False
                        if unchanged:
                            completed.append(dict(cached))
                            continue
                    def report(payload: dict, source_index=index) -> None:
                        event_queue.put(("scan_progress", {**payload, "source_index": source_index, "source_total": len(requested)}))
                    scan = self.project_manager.scan_source(
                        source, cancel_event=cancel_event, progress=report,
                    )
                    if scan.get("cancelled") or cancel_event.is_set():
                        event_queue.put(("scan_cancelled", {"elapsed_seconds": time.perf_counter() - started}))
                        return
                    completed.append(self.project_manager.cache_scan(scan))
                event_queue.put(("scan_done", {
                    "results": completed, "requested": requested,
                    "elapsed_seconds": time.perf_counter() - started,
                }))
            except Exception as exc:
                event_queue.put(("scan_error", str(exc)))

        self.scan_thread = threading.Thread(target=worker, name="samroad-data-source-scan", daemon=True)
        self.scan_thread.start()

    def cancel_data_source_scan(self) -> None:
        event = self.scan_cancel_event
        if event is not None:
            event.set()
            self.data_status.set("正在取消扫描")
            self.project_scan_summary.set("正在安全停止目录遍历；上一次完整扫描结果将保留。")

    def _finish_scan_ui(self) -> None:
        self.scan_thread = None
        self.scan_cancel_event = None
        if hasattr(self, "scan_data_button"):
            self.scan_data_button.state(["!disabled"])
            self.cancel_scan_button.state(["disabled"])

    def _apply_scan_results(self, payload: dict) -> None:
        results = payload.get("results") or []
        for record in results:
            source = str(Path(record.get("root", "")).expanduser().resolve())
            self.project_scan_cache[source] = dict(record)
            if record.get("discovered"):
                self._apply_discovered_project(record["discovered"], merge=True)
        candidates = {"shp": [], "txt": []}
        for source in self.project_data_sources:
            record = self.project_scan_cache.get(str(Path(source).expanduser().resolve()), {})
            for kind in candidates:
                candidates[kind].extend((record.get("candidates") or {}).get(kind, []))
        mapped = {
            path for _area, path in self.project_validation_areas
        } | {
            path for rows in self.project_area_periods.values() for _period, path in rows
        } | {
            path for _area, _before, _after, path in self.project_area_truths
        }
        self.project_candidates = {}
        for kind, paths in candidates.items():
            seen = set()
            self.project_candidates[kind] = [
                path for path in paths
                if path not in mapped and not (path in seen or seen.add(path))
            ]
        self.data_status.set("已扫描，等待数据检查")
        self.project_scan_summary.set(
            f"已索引 {len(self.project_scan_cache)} 个外部目录；自动识别 {len(self.project_validation_areas)} 个区域、"
            f"{sum(len(rows) for rows in self.project_area_periods.values())} 个期次；"
            f"本轮用时 {float(payload.get('elapsed_seconds', 0.0)):.2f} 秒。"
        )
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()
        self.status.set(self.project_scan_summary.get())
        self._finish_scan_ui()

    def add_project_region(self) -> None:
        path = self._select_path("shp")
        if not path:
            return
        default = Path(path).stem
        name = simpledialog.askstring("区域名称", "请输入区域名称：", initialvalue=default, parent=self.root)
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showerror("区域名称为空", "区域名称不能为空。", parent=self.root)
            return
        if any(existing == name for existing, _value in self.project_validation_areas):
            messagebox.showerror("区域名称重复", f"区域“{name}”已经存在。", parent=self.root)
            return
        self.project_validation_areas.append((name, str(Path(path).resolve())))
        self._consume_candidate(path)
        self.project_area_periods[name] = []
        self.data_region.set(name)
        if not self.stage_region.get():
            self.stage_region.set(name)
        self.vars["mode"].set("validation")
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()

    def replace_project_validation_area(self) -> None:
        region = self._selected_project_region()
        if not region:
            self.add_project_region()
            return
        path = self._select_path("shp")
        if not path:
            return
        self.project_validation_areas = [
            (name, str(Path(path).resolve()) if name == region else value)
            for name, value in self.project_validation_areas
        ]
        self._consume_candidate(path)
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()

    def remove_project_region(self) -> None:
        region = self._selected_project_region()
        if not region:
            return
        if not messagebox.askyesno("移除区域映射", f"仅从项目配置移除区域“{region}”；不会删除任何原始文件。是否继续？", parent=self.root):
            return
        self.project_validation_areas = [row for row in self.project_validation_areas if row[0] != region]
        self.project_area_periods.pop(region, None)
        self.project_area_truths = [row for row in self.project_area_truths if row[0] != region]
        self.project_area_truth_field_configs.pop(region, None)
        self.data_region.set("")
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()

    def add_project_period(self) -> None:
        region = self._selected_project_region()
        if not region:
            messagebox.showinfo("请先添加区域", "请先添加或选择一个区域。", parent=self.root)
            return
        source = self._select_path("txt")
        if not source:
            return
        period = simpledialog.askstring("影像期次", "请输入影像期次名称：", initialvalue=Path(source).stem, parent=self.root)
        if period is None:
            return
        period = period.strip()
        rows = list(self.project_area_periods.get(region, []))
        if not period or any(existing == period for existing, _path in rows):
            messagebox.showerror("期次不可用", "期次名称不能为空或与现有期次重复。", parent=self.root)
            return
        rows.append((period, str(Path(source).resolve())))
        self._consume_candidate(source)
        self.project_area_periods[region] = rows
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()

    def replace_project_period_source(self, period: str) -> None:
        region = self._selected_project_region()
        source = self._select_path("txt")
        if not region or not source:
            return
        self.project_area_periods[region] = [
            (name, str(Path(source).resolve()) if name == period else value)
            for name, value in self.project_area_periods.get(region, [])
        ]
        self._consume_candidate(source)
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()

    def remove_project_period(self, period: str) -> None:
        region = self._selected_project_region()
        self.project_area_periods[region] = [
            row for row in self.project_area_periods.get(region, []) if row[0] != period
        ]
        self.project_area_truths = [
            row for row in self.project_area_truths
            if not (row[0] == region and period in {row[1], row[2]})
        ]
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()

    def set_project_truth(self, before: str, after: str) -> None:
        region = self._selected_project_region()
        path = self._select_path("shp")
        if not region or not path:
            return
        self.project_area_truths = [
            row for row in self.project_area_truths
            if (row[0], row[1], row[2]) != (region, before, after)
        ]
        self.project_area_truths.append((region, before, after, str(Path(path).resolve())))
        self._consume_candidate(path)
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()

    def remove_project_truth(self, before: str, after: str) -> None:
        region = self._selected_project_region()
        self.project_area_truths = [
            row for row in self.project_area_truths
            if (row[0], row[1], row[2]) != (region, before, after)
        ]
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()

    def create_project_folder(self) -> None:
        parent = filedialog.askdirectory(parent=self.root, title="选择新项目保存位置")
        if not parent:
            return
        name = simpledialog.askstring("新建项目", "请输入项目名称：", parent=self.root)
        if name is None:
            return
        safe_name = name.strip()
        if not safe_name or any(character in safe_name for character in '<>:"/\\|?*'):
            messagebox.showerror("项目名称不可用", "项目名称不能为空，且不能包含文件名非法字符。", parent=self.root)
            return
        root = Path(parent) / safe_name
        if root.exists() and any(root.iterdir()):
            messagebox.showerror("项目已存在", f"目标文件夹不是空文件夹：\n{root}", parent=self.root)
            return
        try:
            for child in (
                RESULT_DIRECTORY_NAME, "_work/tasks", "_work/cache",
                "_work/editor_cache", "_logs",
            ):
                (root / child).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("无法新建项目", str(exc), parent=self.root)
            return
        self.project_root_path = str(root.resolve())
        self.project_path_display.set(self.project_root_path)
        self.project_name_display.set(safe_name)
        self.current_project.set(f"当前项目：{safe_name}")
        self.project_validation_areas = []
        self.project_area_periods = {}
        self.project_area_truths = []
        self.project_area_truth_field_configs = {}
        self._truth_field_config_area = ""
        self.project_data_sources = []
        self.project_scan_cache = {}
        self.project_txt_encodings = {}
        self.project_path_relocations = {}
        self.project_candidates = {"shp": [], "txt": []}
        self.vars["output_root"].set(str((root / RESULT_DIRECTORY_NAME).resolve()))
        self.data_source_display.set("尚未连接外部数据源")
        self.data_status.set("未连接数据源")
        self.project_scan_summary.set("项目已创建；请连接外部原始数据源。")
        self._save_project_config()
        self.status.set(self.project_scan_summary.get())

    def open_project_folder(self) -> None:
        root = Path(self.project_root_path).expanduser() if self.project_root_path else None
        if root is None or not root.is_dir():
            messagebox.showinfo("尚未打开项目", "请先打开或新建项目。", parent=self.root)
            return
        self._open(root)

    def rescan_project_folder(self) -> None:
        self.scan_data_sources()

    def import_project_folder(self) -> None:
        directory = filedialog.askdirectory(parent=self.root, title="选择规范项目文件夹")
        if not directory:
            return
        self._load_project_directory(directory)

    def _load_project_directory(self, directory: str) -> None:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            messagebox.showerror("无法打开项目", f"项目文件夹不存在：{root}", parent=self.root)
            return
        try:
            payload = self.project_manager.read_config(root)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            messagebox.showerror("项目配置不可用", str(exc), parent=self.root)
            return
        legacy_project = None
        if not payload:
            try:
                legacy_project = self.project_manager.discover_project(root)
            except ValueError:
                legacy_project = None
        for state in self.period_rows:
            state["frame"].destroy()
        self.period_rows = []
        self.vars["mode"].set("validation")
        self.project_root_path = str(root)
        if payload:
            self._apply_project_config(payload)
        elif legacy_project:
            self.project_data_sources = [str(root)]
            self.project_area_truth_field_configs = {}
            self._truth_field_config_area = ""
            self._apply_discovered_project(legacy_project)
            self.vars["output_root"].set(str(legacy_project["output_root"]))
            self.data_source_display.set(str(root))
        else:
            self.project_config = {}
            self.project_data_sources = []
            self.project_scan_cache = {}
            self.project_txt_encodings = {}
            self.project_path_relocations = {}
            self.project_validation_areas = []
            self.project_area_truths = []
            self.project_area_periods = {}
            self.project_area_truth_field_configs = {}
            self._truth_field_config_area = ""
            self.project_candidates = {"shp": [], "txt": []}
            output = root / RESULT_DIRECTORY_NAME
            self.vars["output_root"].set(str(output))
            self.data_source_display.set("尚未连接外部数据源")
        flat_periods = next(iter(self.project_area_periods.values()), [])
        for period, source in flat_periods:
            self._add_period_row(period, source)
        if not self.period_rows:
            self._add_period_row("2021")
            self._add_period_row("2022")
        self.vars["evaluate"].set("0")
        self.project_path_display.set(str(root))
        self.project_name_display.set(root.name)
        self.current_project.set(f"当前项目：{root.name}")
        self.preflight_passed = False
        self.run_button.state(["!disabled"])
        self._refresh_input_summary()
        self._refresh_project_config_panel()
        self.data_status.set("未连接数据源" if not self.project_data_sources else "已扫描，等待数据检查")
        self.project_scan_summary.set(
            f"项目已打开：{len(self.project_validation_areas)} 个验证区、"
            f"{sum(len(rows) for rows in self.project_area_periods.values())} 个影像期次、"
            f"{len(self.project_area_truths)} 个变化真值；"
            f"已恢复 {len(self.project_scan_cache)} 个数据源扫描索引，未递归重扫。"
        )
        unfinished = self.task_manager.unfinished_state(
            self.vars["output_root"].get(), self.project_config.get("active_task"),
        )
        if unfinished is not None:
            notice = self.task_manager.unfinished_message(unfinished)
            self.status.set(notice.replace("\n", " "))
            self.run_status.set(notice)
            self.preflight_summary.set("检测到未完成任务；点击“继续当前任务”将从未完成位置续跑。")
        else:
            self.status.set(self.project_scan_summary.get())
        self._save_project_config()
        self.refresh_project_results(automatic=True, refresh_evaluation=True)

    def save_task_config(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root, title="导出兼容任务配置", defaultextension=".json",
            filetypes=(("JSON 配置", "*.json"),),
        )
        if not path:
            return
        payload = {
            "version": 2,
            "project_root": self.project_root_path,
            "settings": {key: variable.get() for key, variable in self.vars.items()},
            "periods": self._period_values(),
            "truths": self._truth_values(),
            "validation_areas": self.project_validation_areas,
            "area_truths": self.project_area_truths,
            "area_periods": self.project_area_periods,
        }
        try:
            self.project_manager.write_json(path, payload)
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)
            return
        self.status.set(f"兼容任务配置已导出：{path}")

    def load_task_config(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="加载任务配置", filetypes=(("JSON 配置", "*.json"),),
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("配置根节点必须是 JSON 对象。")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            messagebox.showerror("配置不可用", str(exc), parent=self.root)
            return
        settings = payload.get("settings") or {}
        for key, value in settings.items():
            if key in self.vars:
                self.vars[key].set(str(value))
        for state in self.period_rows:
            state["frame"].destroy()
        self.period_rows = []
        for period, source in payload.get("periods", []):
            self._add_period_row(str(period), str(source))
        if not self.period_rows:
            self._add_period_row("2021")
            self._add_period_row("2022")
        self._sync_truth_rows()
        truth_map = {
            (str(before), str(after)): str(value)
            for before, after, value in payload.get("truths", [])
        }
        for state in self.truth_rows:
            state["path"].set(truth_map.get((str(state["before"]), str(state["after"])), ""))
        self.project_validation_areas = [
            (str(name), str(area)) for name, area in (payload.get("validation_areas") or [])
        ]
        self.project_area_truths = [
            (str(area), str(before), str(after), str(truth))
            for area, before, after, truth in (payload.get("area_truths") or [])
        ]
        self.project_area_periods = {
            str(area): [(str(period), str(source)) for period, source in rows]
            for area, rows in (payload.get("area_periods") or {}).items()
        }
        self.project_root_path = str(payload.get("project_root") or "")
        if self.project_root_path:
            self.project_path_display.set(self.project_root_path)
            self.project_name_display.set(Path(self.project_root_path).name)
            self.current_project.set(f"当前项目：{Path(self.project_root_path).name}")
        self._refresh_project_config_panel()
        self._show_manual_inputs()
        self._refresh_input_summary()
        self.preflight_passed = False
        self.run_button.state(["!disabled"])
        self._save_project_config()
        self.status.set(f"任务配置已加载：{path}")

    def _add_period_row(self, period: str = "", source: str = "") -> None:
        frame = ttk.Frame(self.period_container)
        frame.pack(fill=X, pady=2)
        period_var = StringVar(value=period)
        source_var = StringVar(value=source)
        state: dict[str, object] = {"frame": frame, "period": period_var, "source": source_var}
        ttk.Label(frame, text="期次", width=8).pack(side=LEFT)
        period_entry = ttk.Entry(frame, textvariable=period_var, width=14)
        period_entry.pack(side=LEFT)
        period_entry.bind("<FocusOut>", lambda _event: self._refresh_input_summary())
        ttk.Label(frame, text="影像 TXT", width=9).pack(side=LEFT, padx=(12, 0))
        source_entry = ttk.Entry(frame, textvariable=source_var)
        source_entry.pack(side=LEFT, fill=X, expand=True)
        source_entry.bind("<FocusOut>", lambda _event: self._refresh_input_summary())
        ttk.Button(frame, text="选择 TXT…", style="Compact.TButton", command=lambda: self._browse_variable(source_var, "txt")).pack(side=LEFT, padx=(6, 0))
        ttk.Button(frame, text="移除", style="Compact.TButton", command=lambda: self._remove_period_row(state)).pack(side=LEFT, padx=(4, 0))
        self.period_rows.append(state)
        self._refresh_input_summary()
        self._schedule_content_layout()

    def _remove_period_row(self, state: dict[str, object]) -> None:
        if state not in self.period_rows:
            return
        frame = state.get("frame")
        if frame is not None:
            frame.destroy()
        self.period_rows.remove(state)
        self._sync_truth_rows()
        self._refresh_input_summary()
        self._schedule_content_layout()

    def _period_values(self) -> list[tuple[str, str]]:
        return [
            (state["period"].get().strip(), state["source"].get().strip())
            for state in self.period_rows
        ]

    def _truth_values(self) -> list[tuple[str, str, str]]:
        return [
            (str(state["before"]), str(state["after"]), state["path"].get().strip())
            for state in self.truth_rows
        ]

    def _period_order_confirmation(self) -> str:
        """Describe the exact frozen validation-period order before execution."""
        if self.vars["mode"].get().strip().casefold() != "validation":
            return ""
        by_area = self.project_area_periods or {
            (Path(self.vars["validation_area"].get()).stem or "validation"): self._period_values()
        }
        sections = []
        custom_warning = False
        for area, rows in by_area.items():
            names = [period for period, source in rows if period and source]
            if len(names) < 2:
                continue
            manifest = period_order_manifest(names)
            custom_warning |= bool(manifest["custom_order_warning"])
            pairs = "、".join(f"{before} - {after}" for before, after in manifest["change_pairs"])
            sections.append(
                f"{area}\n期次顺序：{'、'.join(manifest['period_order'])}\n相邻变化：{pairs}"
            )
        if not sections:
            return ""
        warning = "\n\n注意：存在自定义期次名，将按自然顺序排列。" if custom_warning else ""
        return "\n\n".join(sections) + warning

    def _sync_truth_rows(self) -> None:
        preserved = {
            (str(state["before"]), str(state["after"])): state["path"].get()
            for state in self.truth_rows
        }
        for child in self.truth_container.winfo_children():
            child.destroy()
        self.truth_rows = []
        names = sorted(
            {state["period"].get().strip() for state in self.period_rows if state["period"].get().strip()},
            key=natural_key,
        )
        if len(names) < 2:
            ttk.Label(self.truth_container, text="填写至少两个期次后点击“更新相邻变化对”。", style="Hint.TLabel").pack(anchor="w")
            self._schedule_content_layout()
            return
        for before, after in zip(names, names[1:]):
            frame = ttk.Frame(self.truth_container)
            frame.pack(fill=X, pady=2)
            path_var = StringVar(value=preserved.get((before, after), ""))
            ttk.Label(frame, text=f"{before} - {after}", width=23).pack(side=LEFT)
            truth_entry = ttk.Entry(frame, textvariable=path_var)
            truth_entry.pack(side=LEFT, fill=X, expand=True)
            truth_entry.bind("<FocusOut>", lambda _event: self._refresh_input_summary())
            ttk.Button(frame, text="选择真值 SHP…", style="Compact.TButton", command=lambda value=path_var: self._browse_variable(value, "shp")).pack(side=LEFT, padx=(6, 0))
            self.truth_rows.append({"frame": frame, "before": before, "after": after, "path": path_var})
        self._refresh_input_summary()
        self._schedule_content_layout()

    def _toggle_grid_options(self) -> None:
        self.grid_options_visible = not self.grid_options_visible
        if self.grid_options_visible:
            self.grid_options.pack(fill=X, pady=(2, 4), after=self.grid_toggle)
            self.grid_toggle.configure(text="收起旧版多格网目录")
        else:
            self.grid_options.pack_forget()
            self.grid_toggle.configure(text="兼容旧版多格网目录...")
        self._refresh_input_summary()
        self._schedule_content_layout()
