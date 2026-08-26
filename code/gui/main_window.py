from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
import ctypes
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Canvas, Menu, StringVar, TclError, Text, Tk, Toplevel, filedialog, font as tkfont, messagebox, simpledialog
from tkinter import ttk

from app.backend_client import BackendClient, BackendEvent
from app.editor_manager import EditorManager
from app.project_manager import ProjectManager
from app.result_publisher import ProjectLayout
from app.task_manager import TaskManager, _safe_task_name, structured_task_status
from gui.common_widgets import (CONTROL_METRICS, LAYOUT_METRICS, PREVIEW_LABELS, UI, WORKFLOW_STEPS, bind_dynamic_wrap, configure_window_geometry, format_percentage, treeview_metrics)
from gui.data_page import DataPage
from gui.run_page import RunPage
from gui.edit_page import EditPage
from gui.result_page import ResultPage



ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
RUNTIME_ROOT = WORKSPACE_ROOT / "runtime"
PROJECT_ROOT = WORKSPACE_ROOT / "project"
BACKEND = ROOT / "user_pipeline.py"
DEFAULT_CONFIG = RUNTIME_ROOT / "config" / "samroad_inference.yaml"
DEFAULT_CKPT = RUNTIME_ROOT / "models" / "samroad" / "samroad.ckpt"
DEFAULT_TEST_DATA = PROJECT_ROOT / "data"
MAX_QUEUE_EVENTS_PER_POLL = 200
MAX_PRIORITY_EVENTS_PER_POLL = 50
QUEUE_POLL_TIME_BUDGET_SECONDS = 0.012

# Interactive controls use three deliberate size tiers. Semantic aliases may
# change colour and weight, but must not introduce another control height.

# Shared spacing keeps the existing restrained desktop layout consistent without
# introducing a second theme or scattering per-widget magic numbers.


def enable_windows_high_dpi() -> None:
    """Render Tk at the monitor's native DPI instead of Windows bitmap scaling."""
    if sys.platform != "win32":
        return
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def configure_tk_scaling(root: Tk) -> float:
    """Match Tk points to the active Windows DPI and return the UI scale."""
    dpi = float(root.winfo_fpixels("1i"))
    root.tk.call("tk", "scaling", max(1.0, min(dpi / 72.0, 3.0)))
    return max(1.0, min(dpi / 96.0, 2.5))


def configure_ui_fonts(root: Tk) -> None:
    """Use one ClearType-friendly CJK family for default and ttk controls."""
    definitions = {
        "TkDefaultFont": ("Microsoft YaHei UI", 10, "normal"),
        "TkTextFont": ("Microsoft YaHei UI", 10, "normal"),
        "TkMenuFont": ("Microsoft YaHei UI", 10, "normal"),
        "TkHeadingFont": ("Microsoft YaHei UI", 10, "bold"),
        "TkCaptionFont": ("Microsoft YaHei UI", 10, "bold"),
        "TkSmallCaptionFont": ("Microsoft YaHei UI", 9, "normal"),
        "TkFixedFont": ("Consolas", 10, "normal"),
    }
    for name, (family, size, weight) in definitions.items():
        try:
            tkfont.nametofont(name, root=root).configure(family=family, size=size, weight=weight)
        except TclError:
            continue



























def format_duration(value: object) -> str:
    try:
        seconds = max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return "--"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"










def format_bytes(value: object) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "未知"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return "未知"















































class UserApp(DataPage, RunPage, EditPage, ResultPage):
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("道路实体变化智能检测与人工编辑")
        self.display_scale = configure_window_geometry(
            self.root, base_width=1280, base_height=820, min_width=1000, min_height=680,
        )
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.priority_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.backend_client = BackendClient(
            app_root=ROOT, backend_script=BACKEND,
            runtime_root=RUNTIME_ROOT,
            event_queue=self.queue, priority_queue=self.priority_queue,
        )
        self.project_manager = ProjectManager()
        self.task_manager = TaskManager(self.backend_client)
        self.editor_manager = EditorManager(self.queue)
        self.editor_ready_file: Path | None = None
        self.editor_started_monotonic: float | None = None
        self.editor_launch_state = "idle"
        self.editor_timeout_reported = False
        self.editor_stdout_lines: list[str] = []
        self.editor_stderr_lines: list[str] = []
        self.results_available = False
        self.review_items: list[dict[str, str]] = []
        self.temporal_items: list[dict[str, str]] = []
        self.period_rows: list[dict[str, object]] = []
        self.truth_rows: list[dict[str, object]] = []
        self.result_change_items: list[dict] = []
        self.project_validation_areas: list[tuple[str, str]] = []
        self.project_area_truths: list[tuple[str, str, str, str]] = []
        self.project_area_periods: dict[str, list[tuple[str, str]]] = {}
        self.project_data_sources: list[str] = []
        self.project_scan_cache: dict[str, dict] = {}
        self.project_txt_encodings: dict[str, str] = {}
        self.project_path_relocations: dict[str, str] = {}
        self.project_candidates: dict[str, list[str]] = {"shp": [], "txt": []}
        self.project_config: dict = {}
        self.project_root_path = ""
        self.data_region = StringVar(value="")
        self.stage_region = StringVar(value="")
        self.project_period = StringVar(value="")
        self.project_change_pair = StringVar(value="")
        self.project_scan_summary = StringVar(value="尚未扫描数据源。")
        self.data_status = StringVar(value="未连接数据源")
        self.data_source_display = StringVar(value="尚未连接外部数据源")
        self.project_validation_path = StringVar(value="尚未选择验证区。")
        self.review_edit_directory = StringVar(value="")
        self.grid_options_visible = False
        self.manual_inputs_visible = False
        self.run_settings_visible = False
        self.advanced_visible = False
        self.review_advanced_visible = False
        self.evaluation_advanced_visible = False
        self.log_visible = True
        self.current_step = 0
        self.preflight_passed = False
        self.step_pages: list[ttk.Frame] = []
        self.step_summaries: list[ttk.Frame] = []
        self.step_buttons: list[ttk.Button] = []
        self.active_command = ""
        self.cancel_requested = False
        self.task_started_monotonic: float | None = None
        self.last_elapsed_seconds = 0.0
        self.progress_completed = 0
        self.progress_total = 0
        self.progress_eta: float | None = None
        self.last_complete_payload: dict | None = None
        self.current_stage_payload: dict | None = None
        self.active_log_path: Path | None = None
        self.recent_log_lines: list[str] = []
        self._pending_log_insert_lines: list[str] = []
        self._pending_log_delete_lines = 0
        self.scan_thread: threading.Thread | None = None
        self.scan_cancel_event: threading.Event | None = None
        self._result_tree_fingerprint: tuple | None = None
        self.result_tree_paths: dict[str, Path] = {}
        self.vars = {
            key: StringVar(value=value)
            for key, value in {
                "mode": "validation",
                "execution_profile": "fast",
                "validation_area": "",
                "truth_type_field": "",
                "evaluate": "0",
                "resume": "0",
                "continue_on_error": "1",
                "source_root": str(DEFAULT_TEST_DATA),
                "output_root": str(PROJECT_ROOT / "成果输出"),
                "run_id": "",
                "checkpoint": str(DEFAULT_CKPT),
                "config": str(DEFAULT_CONFIG),
                "device": "auto",
                "pixel_size": "0.0",
                "rescale": "off",
                "absolute": "2.0",
                "ratio": "0.20",
                "tolerance": "3.0",
                "junction_node_mode": "sparse",
            }.items()
        }
        self._style()
        self._build()
        self.root.after(100, self._poll)

    @property
    def process(self):
        client = getattr(self, "backend_client", None)
        return client.process if client is not None else getattr(self, "_compat_process", None)

    @process.setter
    def process(self, value) -> None:
        client = getattr(self, "backend_client", None)
        if client is not None:
            client.process = value
        else:
            self._compat_process = value

    @property
    def editor_process(self):
        manager = getattr(self, "editor_manager", None)
        return manager.process if manager is not None else getattr(self, "_compat_editor_process", None)

    @editor_process.setter
    def editor_process(self, value) -> None:
        manager = getattr(self, "editor_manager", None)
        if manager is not None:
            manager.process = value
        else:
            self._compat_editor_process = value

    def _style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self.root.configure(background=UI["page"])
        style.configure("TFrame", background=UI["card"])
        style.configure("Page.TFrame", background=UI["page"])
        style.configure("Header.TFrame", background=UI["card"])
        style.configure("Footer.TFrame", background=UI["card"])
        style.configure("TLabel", background=UI["card"], foreground=UI["ink"], font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background=UI["card"], font=("Microsoft YaHei UI", 10, "bold"), foreground=UI["ink"])
        style.configure("Subtitle.TLabel", background=UI["card"], font=("Microsoft YaHei UI", 9), foreground=UI["muted"])
        style.configure("Brand.TLabel", background=UI["header"], foreground="#79C3AD", font=("Segoe UI", 9, "bold"))
        style.configure("HeaderTitle.TLabel", background=UI["card"], foreground=UI["ink"], font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("HeaderMeta.TLabel", background=UI["card"], foreground=UI["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("HeaderProject.TLabel", background=UI["card"], foreground=UI["ink"], font=("Microsoft YaHei UI", 9))
        style.configure("Card.TFrame", background=UI["card"], relief="solid", borderwidth=1, bordercolor=UI["line"])
        style.configure("Soft.TFrame", background=UI["slate_soft"])
        style.configure("BlueSoft.TFrame", background=UI["blue_soft"])
        style.configure("GreenSoft.TFrame", background=UI["green_soft"])
        style.configure("SelectedBorder.TFrame", background=UI["blue"])
        style.configure("IdleBorder.TFrame", background=UI["line"])
        style.configure("SelectedCard.TFrame", background=UI["blue_soft"])
        style.configure("Primary.TButton", **CONTROL_METRICS["primary"], foreground="#FFFFFF", background=UI["blue"], borderwidth=1, bordercolor=UI["blue"], focusthickness=0)
        style.map("Primary.TButton", background=[("active", UI["blue_hover"]), ("pressed", UI["header_deep"]), ("disabled", "#B6B8B1")])
        style.configure("TButton", **CONTROL_METRICS["regular"], foreground=UI["ink"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("TButton", background=[("active", UI["blue_soft"]), ("disabled", "#ECE9E1")], foreground=[("active", UI["blue"]), ("disabled", UI["subtle"])])
        style.configure("Hero.TButton", **CONTROL_METRICS["primary"], foreground="#FFFFFF", background=UI["blue"], borderwidth=1, bordercolor=UI["blue"], focusthickness=0)
        style.map("Hero.TButton", background=[("active", UI["blue_hover"]), ("pressed", UI["header_deep"]), ("disabled", "#B6B8B1")])
        style.configure("Secondary.TButton", **CONTROL_METRICS["regular"], foreground=UI["ink"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("Secondary.TButton", background=[("active", UI["blue_soft"]), ("disabled", "#ECE9E1")], foreground=[("disabled", UI["subtle"])])
        style.configure("Compact.TButton", **CONTROL_METRICS["compact"], foreground=UI["ink"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("Compact.TButton", background=[("active", UI["blue_soft"]), ("disabled", "#ECE9E1")], foreground=[("active", UI["blue"]), ("disabled", UI["subtle"])])
        style.configure("Quiet.TButton", **CONTROL_METRICS["compact"], foreground=UI["ink"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("Quiet.TButton", foreground=[("disabled", UI["subtle"])], background=[("active", UI["blue_soft"])])
        style.configure("Danger.TButton", **CONTROL_METRICS["compact"], foreground=UI["ink"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("Danger.TButton", background=[("active", UI["blue_soft"]), ("disabled", "#ECE9E1")], foreground=[("disabled", UI["subtle"])])
        style.configure("ResultPrimary.TButton", **CONTROL_METRICS["regular"], foreground="#FFFFFF", background=UI["blue"], borderwidth=1, bordercolor=UI["blue"], focusthickness=0)
        style.map("ResultPrimary.TButton", background=[("active", UI["blue_hover"]), ("pressed", UI["header_deep"]), ("disabled", "#B6B8B1")])
        style.configure("ResultSecondary.TButton", **CONTROL_METRICS["regular"], foreground=UI["ink"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("ResultSecondary.TButton", background=[("active", UI["blue_soft"]), ("disabled", "#ECE9E1")], foreground=[("disabled", UI["subtle"])])
        style.configure("Section.TLabel", font=("Microsoft YaHei UI", 10, "bold"), foreground=UI["ink"], background=UI["page"])
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", 9, "bold"), foreground=UI["ink"], background=UI["card"])
        style.configure("CardTitleSelected.TLabel", font=("Microsoft YaHei UI", 12, "bold"), foreground=UI["blue"], background=UI["blue_soft"])
        style.configure("Metric.TLabel", font=("Microsoft YaHei UI", 9), foreground=UI["ink"], background=UI["card"])
        style.configure("Muted.TLabel", foreground=UI["muted"], background=UI["page"], font=("Microsoft YaHei UI", 9))
        style.configure("CardMuted.TLabel", foreground=UI["muted"], background=UI["card"], font=("Microsoft YaHei UI", 9))
        style.configure("CardMutedSelected.TLabel", foreground=UI["muted"], background=UI["blue_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("Eyebrow.TLabel", foreground=UI["blue"], background=UI["card"], font=("Microsoft YaHei UI", 8, "bold"))
        style.configure("PathTitle.TLabel", foreground=UI["ink"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("PathText.TLabel", foreground=UI["muted"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("MetricName.TLabel", foreground=UI["muted"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("MetricValue.TLabel", foreground=UI["ink"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("FormLabel.TLabel", foreground=UI["ink"], background=UI["card"], font=("Microsoft YaHei UI", 10))
        style.configure("SelectedMark.TLabel", foreground=UI["blue"], background=UI["blue_soft"], font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("IdleMark.TLabel", foreground=UI["subtle"], background=UI["card"], font=("Microsoft YaHei UI", 15))
        style.configure("SuccessNote.TLabel", foreground=UI["green"], background=UI["blue_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("WarningNote.TLabel", foreground=UI["amber"], background=UI["card"], font=("Microsoft YaHei UI", 9))
        style.configure("TNotebook.Tab", font=("Microsoft YaHei UI", 10, "bold"), padding=(18, 8))
        style.configure("Hint.TLabel", foreground=UI["muted"], background=UI["card"], font=("Microsoft YaHei UI", 9))
        style.configure("Success.TLabel", foreground=UI["green"], background=UI["card"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("HeaderReady.TLabel", foreground=UI["green"], background=UI["card"], font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("HeaderIdle.TLabel", foreground=UI["muted"], background=UI["card"], font=("Microsoft YaHei UI", 9))
        style.configure("FooterStatus.TLabel", background=UI["card"], foreground=UI["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("TEntry", padding=(6, 5), fieldbackground=UI["card"], bordercolor=UI["line_strong"], lightcolor=UI["line_strong"], darkcolor=UI["line_strong"])
        style.configure("TCombobox", padding=(6, 5), fieldbackground=UI["card"], bordercolor=UI["line_strong"])
        style.map("TCombobox", fieldbackground=[("readonly", UI["card"])], background=[("readonly", UI["card"])], foreground=[("readonly", UI["ink"])])
        style.configure("TCheckbutton", background=UI["card"], foreground=UI["ink"])
        style.configure("TRadiobutton", background=UI["card"], foreground=UI["ink"])
        style.configure("Modern.Horizontal.TProgressbar", background=UI["blue"], troughcolor="#DEDAD0", borderwidth=1, thickness=9)
        style.configure("TLabelframe", background=UI["card"], bordercolor=UI["line_strong"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=UI["card"], foreground=UI["ink"], font=("Microsoft YaHei UI", 10, "bold"))
        tree_metrics = treeview_metrics(self.root, ("Microsoft YaHei UI", 9))
        style.configure(
            "Data.Treeview", background=UI["card"], fieldbackground=UI["card"],
            foreground=UI["ink"], rowheight=tree_metrics["rowheight"], borderwidth=1,
            font=tree_metrics["font"],
        )
        style.map(
            "Data.Treeview", background=[("selected", UI["blue"])],
            foreground=[("selected", "#FFFFFF")],
        )
        style.configure(
            "Data.Treeview.Heading", background="#E3DFD5", foreground=UI["ink"],
            relief="raised", padding=tree_metrics["heading_padding"], font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map("Data.Treeview.Heading", background=[("active", UI["blue_soft"])])

    def _build(self) -> None:
        self.status = StringVar(value="请选择 SAMRoad 项目目录。")
        self.current_project = StringVar(value="当前项目：未选择")
        self.header_state = StringVar(value="等待选择数据")
        menu = Menu(self.root)
        project_menu = Menu(menu, tearoff=False)
        project_menu.add_command(label="新建项目", command=self.create_project_folder)
        project_menu.add_command(label="打开项目", command=self.import_project_folder)
        project_menu.add_separator()
        project_menu.add_command(label="打开项目文件夹", command=self.open_project_folder)
        menu.add_cascade(label="项目", menu=project_menu)
        tools_menu = Menu(menu, tearoff=False)
        tools_menu.add_command(label="运行诊断 / 导出诊断包", command=self.export_diagnostics)
        menu.add_cascade(label="工具", menu=tools_menu)
        help_menu = Menu(menu, tearoff=False)
        help_menu.add_command(
            label="关于",
            command=lambda: messagebox.showinfo(
                "关于", "道路实体变化智能检测与人工编辑", parent=self.root,
            ),
        )
        menu.add_cascade(label="帮助", menu=help_menu)
        self.root.configure(menu=menu)

        header = ttk.Frame(self.root, padding=(12, 6), style="Header.TFrame")
        header.pack(fill=X)
        ttk.Label(header, textvariable=self.current_project, style="HeaderProject.TLabel").pack(side=LEFT, fill=X, expand=True)
        ttk.Label(header, text="状态：", style="HeaderMeta.TLabel").pack(side=LEFT)
        self.header_state_label = ttk.Label(header, textvariable=self.header_state, style="HeaderIdle.TLabel")
        self.header_state_label.pack(side=LEFT)

        self.stepper_canvas = Canvas(
            self.root, height=round(34 * self.display_scale), background=UI["page"], highlightthickness=0, borderwidth=0,
        )
        self.stepper_canvas.pack(fill=X)
        self.stepper_canvas.bind("<Configure>", lambda _event: self._draw_stepper())
        self.stepper_canvas.bind("<Button-1>", self._on_stepper_click)

        page_padding = LAYOUT_METRICS["page_padding"]
        self.content_shell = ttk.Frame(
            self.root, style="Page.TFrame",
            padding=(page_padding[0], 0, page_padding[2], 0),
        )
        self.content_shell.pack(fill=BOTH, expand=True)
        self.content_shell.grid_propagate(False)
        self.content_shell.grid_columnconfigure(0, weight=64, uniform="workflow")
        self.content_shell.grid_columnconfigure(1, weight=36, uniform="workflow")
        self.content_shell.grid_rowconfigure(0, weight=1)
        left_shell = ttk.Frame(self.content_shell, style="Page.TFrame")
        left_shell.grid(row=0, column=0, sticky="nsew", padx=(0, LAYOUT_METRICS["module_gap"] // 2))
        self.content_canvas = Canvas(
            left_shell, background=UI["page"], highlightthickness=0, borderwidth=0,
        )
        self.content_scrollbar = ttk.Scrollbar(
            left_shell, orient="vertical", command=self.content_canvas.yview,
        )
        self.content_scrollbar.pack(side=RIGHT, fill="y")
        self.content_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.content_canvas.configure(yscrollcommand=self.content_scrollbar.set)
        self.page_host = ttk.Frame(
            self.content_canvas,
            padding=(0, page_padding[1], LAYOUT_METRICS["module_gap"] // 2, page_padding[3]),
            style="Page.TFrame",
        )
        self.page_window = self.content_canvas.create_window((0, 0), window=self.page_host, anchor="nw")
        self.max_page_width = round(1600 * self.display_scale)
        self.content_canvas.bind("<Configure>", self._resize_content_canvas)
        self.page_host.bind("<Configure>", lambda _event: self._sync_content_scrollregion())
        self.root.bind_all("<MouseWheel>", self._on_content_mousewheel, add="+")
        self.sidebar = ttk.Frame(self.content_shell, style="Page.TFrame")
        self.sidebar.grid(
            row=0, column=1, sticky="nsew",
            padx=(LAYOUT_METRICS["module_gap"] // 2, 0),
            pady=(page_padding[1], page_padding[3]),
        )
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(0, weight=1)
        self.sidebar_paned = ttk.Panedwindow(self.sidebar, orient="vertical")
        self.sidebar_paned.grid(row=0, column=0, sticky="nsew")
        self.summary_host = ttk.Frame(self.sidebar_paned, style="Page.TFrame")
        self.sidebar_paned.add(self.summary_host, weight=35)
        for _ in WORKFLOW_STEPS:
            page = ttk.Frame(self.page_host, style="Page.TFrame")
            self.step_pages.append(page)
            summary = ttk.Frame(self.summary_host, style="Page.TFrame")
            self.step_summaries.append(summary)

        self._build_data_page(self.step_pages[0])
        self._build_run_page(self.step_pages[1])
        self._build_review_page(self.step_pages[2])
        self._build_result_page(self.step_pages[3])
        self._build_shared_log_panel()

        ttk.Separator(self.root).pack(fill=X)
        footer = ttk.Frame(self.root, padding=(12, 2), style="Footer.TFrame")
        footer.pack(fill=X)
        footer_status = ttk.Label(footer, textvariable=self.status, style="FooterStatus.TLabel")
        footer_status.pack(side=LEFT, fill=X, expand=True)
        bind_dynamic_wrap(footer_status, footer, minimum=240, padding=180)
        self.footer_next = ttk.Button(footer, text="下一步", command=self._go_next)
        self.footer_next.pack(side=RIGHT)
        self.footer_back = ttk.Button(footer, text="上一步", command=self._go_back)
        self.footer_back.pack(side=RIGHT, padx=(0, 6))

        self._show_step(0, force=True)
        self._refresh_input_summary()
        self._refresh_result_availability()
        self.root.after_idle(self._initialize_sidebar_sash)

    def _build_shared_log_panel(self) -> None:
        """Build the window-level log panel shared by every workflow step."""
        self.shared_log_shell = ttk.LabelFrame(
            self.sidebar_paned, text="全流程日志", padding=LAYOUT_METRICS["card_padding"],
        )
        self.sidebar_paned.add(self.shared_log_shell, weight=65)
        self.shared_log_shell.grid_rowconfigure(1, weight=1)
        self.shared_log_shell.grid_columnconfigure(0, weight=1)
        log_header = ttk.Frame(self.shared_log_shell)
        log_header.pack(fill=X)
        self.shared_log_status = StringVar(value="日志尚未开始")
        log_status = ttk.Label(
            log_header, textvariable=self.shared_log_status, style="CardMuted.TLabel",
        )
        log_status.pack(side=LEFT, fill=X, expand=True, padx=(LAYOUT_METRICS["module_gap"], 10))
        bind_dynamic_wrap(log_status, log_header, minimum=180, padding=220)
        ttk.Button(
            log_header, text="复制全部", style="Compact.TButton", command=self.copy_all_logs,
        ).pack(side=RIGHT, padx=(0, 5))
        ttk.Button(
            log_header, text="打开日志文件", style="Compact.TButton", command=self.open_active_log,
        ).pack(side=RIGHT, padx=(0, 5))

        self.log_frame = ttk.Frame(self.shared_log_shell)
        self.log_frame.pack(fill=BOTH, expand=True, pady=(LAYOUT_METRICS["module_gap"], 0))
        log_body = ttk.Frame(self.log_frame)
        log_body.pack(fill=BOTH, expand=True)
        self.log = Text(
            log_body, height=12, wrap="none", undo=False, exportselection=True,
            font=("Consolas", 9), foreground=UI["ink"], background="#FFFFFF",
            selectbackground=UI["blue"], selectforeground="#FFFFFF", padx=5, pady=4,
        )
        log_y = ttk.Scrollbar(log_body, orient="vertical", command=self.log.yview)
        log_x = ttk.Scrollbar(log_body, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=log_y.set, xscrollcommand=log_x.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_y.grid(row=0, column=1, sticky="ns")
        log_x.grid(row=1, column=0, sticky="ew")
        log_body.grid_rowconfigure(0, weight=1)
        log_body.grid_columnconfigure(0, weight=1)
        self.log.bind("<Control-a>", self._select_all_logs)
        self.log.bind("<Control-A>", self._select_all_logs)
        self.log.insert("1.0", "这里统一显示数据检查、自动处理、人工编辑和精度评价日志。\n")

    def _initialize_sidebar_sash(self) -> None:
        """Set a useful initial split once; later page changes preserve the user's sash."""
        if getattr(self, "_sidebar_sash_initialized", False):
            return
        self.sidebar_paned.update_idletasks()
        total_height = self.sidebar_paned.winfo_height()
        if total_height <= 1:
            self.root.after(50, self._initialize_sidebar_sash)
            return
        summary_height = self.summary_host.winfo_reqheight() + LAYOUT_METRICS["module_gap"]
        lower = round(total_height * 0.30)
        upper = round(total_height * 0.45)
        desired = max(lower, min(summary_height, upper))
        desired = min(desired, max(lower, total_height - 180))
        try:
            self.sidebar_paned.sashpos(0, desired)
        except TclError:
            return
        self._sidebar_sash_initialized = True

    def _resize_content_canvas(self, event=None) -> None:
        if not hasattr(self, "content_canvas"):
            return
        width = max(1, int(event.width if event is not None else self.content_canvas.winfo_width()))
        content_width = min(width, self.max_page_width)
        left = max(0, (width - content_width) // 2)
        self.content_canvas.coords(self.page_window, left, 0)
        self.content_canvas.itemconfigure(self.page_window, width=content_width)
        self._sync_content_scrollregion()

    def _sync_content_scrollregion(self) -> None:
        if not hasattr(self, "content_canvas"):
            return
        self.page_host.update_idletasks()
        required_height = max(self.page_host.winfo_reqheight(), self.content_canvas.winfo_height())
        self.content_canvas.itemconfigure(self.page_window, height=required_height)
        self.content_canvas.configure(scrollregion=(0, 0, self.content_canvas.winfo_width(), required_height))

    def _schedule_content_layout(self) -> None:
        if hasattr(self, "content_canvas"):
            self.root.after_idle(self._sync_content_scrollregion)

    def _on_content_mousewheel(self, event) -> str | None:
        if not hasattr(self, "content_canvas"):
            return None
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        inside_content = False
        while widget is not None:
            if widget in (self.content_canvas, self.page_host):
                inside_content = True
                break
            widget = getattr(widget, "master", None)
        if not inside_content:
            return None
        delta = int(-event.delta / 120) if event.delta else 0
        if delta:
            self.content_canvas.yview_scroll(delta * 3, "units")
            return "break"
        return None

    def _draw_stepper(self) -> None:
        if not hasattr(self, "stepper_canvas"):
            return
        canvas = self.stepper_canvas
        canvas.delete("all")
        width = max(800, canvas.winfo_width())
        height = max(round(30 * self.display_scale), canvas.winfo_height())
        margin_x = round(10 * self.display_scale)
        edge = max(3, round(3 * self.display_scale))
        top = edge
        bottom = height - edge
        tab_width = (width - margin_x * 2) / len(WORKFLOW_STEPS)
        centers = [margin_x + tab_width * (index + 0.5) for index in range(len(WORKFLOW_STEPS))]
        has_results = self.results_available
        for index, (center, label) in enumerate(zip(centers, WORKFLOW_STEPS)):
            left = margin_x + tab_width * index
            right = left + tab_width
            if index == self.current_step:
                fill, outline = UI["blue"], UI["blue_hover"]
                label_color, weight = "#FFFFFF", "bold"
            else:
                fill, outline = "#E8E4DA", UI["line_strong"]
                label_color, weight = UI["ink"], "normal"
            canvas.create_rectangle(left, top, right, bottom, fill=fill, outline=outline, width=1)
            canvas.create_text(
                center, (top + bottom) / 2, text=f"{index + 1}. {label}", fill=label_color,
                font=("Microsoft YaHei UI", 9, weight),
            )
        self._step_centers = centers
        self._step_tab_width = tab_width
        self._step_has_results = has_results

    def _on_stepper_click(self, event) -> None:
        centers = getattr(self, "_step_centers", [])
        if not centers:
            return
        index = min(range(len(centers)), key=lambda value: abs(centers[value] - event.x))
        if abs(centers[index] - event.x) > getattr(self, "_step_tab_width", 220) / 2:
            return
        self._show_step(index)

    @staticmethod
    def _bind_card(widget, callback) -> None:
        widget.configure(cursor="hand2")
        widget.bind("<Button-1>", callback)
        for child in widget.winfo_children():
            UserApp._bind_card(child, callback)

    def _set_task_mode(self, value: str) -> None:
        self.vars["evaluate"].set(value)
        self.preflight_passed = False
        self._refresh_task_mode_cards()
        self._refresh_input_summary()

    def _refresh_task_mode_cards(self) -> None:
        if not hasattr(self, "task_mode_cards"):
            return
        selected_value = self.vars["evaluate"].get()
        for value, card in self.task_mode_cards.items():
            selected = value == selected_value
            card["outer"].configure(style="SelectedBorder.TFrame" if selected else "IdleBorder.TFrame")
            card["inner"].configure(style="SelectedCard.TFrame" if selected else "TFrame")
            card["top"].configure(style="SelectedCard.TFrame" if selected else "TFrame")
            card["mark"].configure(text="选中" if selected else "未选", style="SelectedMark.TLabel" if selected else "IdleMark.TLabel")
            card["title"].configure(style="CardTitleSelected.TLabel" if selected else "CardTitle.TLabel")
            card["description"].configure(style="CardMutedSelected.TLabel" if selected else "CardMuted.TLabel")
            if value == "0":
                card["note"].configure(style="SuccessNote.TLabel" if selected else "Success.TLabel")
            else:
                card["note"].configure(style="CardMutedSelected.TLabel" if selected else "WarningNote.TLabel")





    def _scroll_to_module(self, widget) -> None:
        """Compatibility shim for callers from the former scrolling layout."""
        mapping = {
            getattr(self, "data_body", None): 0,
            getattr(self, "run_body", None): 1,
            getattr(self, "review_body", None): 2,
            getattr(self, "result_body", None): 3,
        }
        self._show_step(mapping.get(widget, self.current_step), force=True)

    def _show_step(self, index: int, force: bool = False) -> None:
        if not self.step_pages:
            return
        index = max(0, min(int(index), len(self.step_pages) - 1))
        if self.process is not None and index != self.current_step and index in {0, 2} and not force:
            self.status.set("任务运行期间不能修改数据配置或人工编辑；仍可查看日志和已有成果。")
            return
        has_results = self.results_available
        if index == 3 and not has_results and not force:
            self.status.set("请先完成自动处理或打开包含已有成果的项目，再进入成果与评价步骤。")
            return
        for page in self.step_pages:
            page.pack_forget()
        for summary in getattr(self, "step_summaries", []):
            summary.pack_forget()
        self.current_step = index
        self.step_pages[index].pack(fill=BOTH, expand=True)
        if getattr(self, "step_summaries", None):
            self.step_summaries[index].pack(fill=BOTH, expand=True)
        if hasattr(self, "content_canvas"):
            self.content_canvas.yview_moveto(0.0)
            self.root.after_idle(self._sync_content_scrollregion)
        completed_to = 0
        if self.preflight_passed:
            completed_to = 1
        if has_results:
            completed_to = 3
        self.completed_to = completed_to
        self.root.after_idle(self._draw_stepper)
        self.footer_back.state(["disabled"] if index == 0 else ["!disabled"])
        self.footer_next.configure(text="完成" if index == 3 else "下一步")
        if index == 3:
            self.footer_next.state(["disabled"])
        elif index == 2 and not has_results:
            self.footer_next.state(["disabled"])
        else:
            self.footer_next.state(["!disabled"])
        if index == 2:
            self._populate_review_step()
        elif index == 3:
            self._refresh_result_availability()

    def _go_back(self) -> None:
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _go_next(self) -> None:
        if self.current_step == 0:
            try:
                self._build_current_command(preflight_only=True)
            except ValueError as exc:
                messagebox.showerror("输入不完整", str(exc), parent=self.root)
                self._show_manual_inputs()
                return
            self._show_step(1)
            return
        if self.current_step == 1:
            if not self.results_available:
                self.status.set("请先完成自动处理，再进入人工编辑步骤。")
                return
            self._show_step(2)
            return
        if self.current_step == 2:
            self.status.set("已跳过人工编辑，采用自动处理结果。")
            self._show_step(3)

    def _toggle_manual_inputs(self) -> None:
        self.manual_inputs_visible = not self.manual_inputs_visible
        if self.manual_inputs_visible:
            self.manual_frame.pack(fill=X, pady=(0, 6), after=self.manual_toggle)
            self.manual_toggle.configure(text="收起高级设置")
        else:
            self.manual_frame.pack_forget()
            self.manual_toggle.configure(text="高级设置...")
        self._schedule_content_layout()

    def _show_manual_inputs(self) -> None:
        if not self.manual_inputs_visible:
            self._toggle_manual_inputs()

    def _toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill=X, pady=(2, 5), after=self.advanced_toggle)
            self.advanced_toggle.configure(text="收起高级参数")
        else:
            self.advanced_frame.pack_forget()
            self.advanced_toggle.configure(text="高级参数...")
        self._schedule_content_layout()

    def _toggle_review_advanced(self) -> None:
        self.review_advanced_visible = not self.review_advanced_visible
        if self.review_advanced_visible:
            self.review_advanced_frame.pack(fill=X, pady=(4, 0), after=self.review_advanced_toggle)
            self.review_advanced_toggle.configure(text="收起高级操作")
        else:
            self.review_advanced_frame.pack_forget()
            self.review_advanced_toggle.configure(text="高级操作...")
        self._schedule_content_layout()

    def _toggle_evaluation_advanced(self) -> None:
        self.evaluation_advanced_visible = not self.evaluation_advanced_visible
        if self.evaluation_advanced_visible:
            self.evaluation_advanced_frame.pack(fill=X, pady=(2, 10), after=self.evaluation_advanced_toggle)
            self.evaluation_advanced_toggle.configure(text="收起评价高级设置")
        else:
            self.evaluation_advanced_frame.pack_forget()
            self.evaluation_advanced_toggle.configure(text="评价高级设置...")
        self._schedule_content_layout()

    def _toggle_run_settings(self) -> None:
        self.run_settings_visible = not self.run_settings_visible
        if self.run_settings_visible:
            self.run_settings_frame.pack(fill=X, pady=(4, 2), after=self.run_settings_toggle)
            self.run_settings_toggle.configure(text="收起输出位置与高级设置")
        else:
            self.run_settings_frame.pack_forget()
            self.run_settings_toggle.configure(text="输出位置与高级设置...")
        self._schedule_content_layout()

    def _toggle_log(self) -> None:
        """Compatibility hook: the desktop layout keeps the shared log visible."""
        self.log_visible = True
        if not self.log_frame.winfo_manager():
            self.log_frame.pack(fill=BOTH, expand=True, pady=(LAYOUT_METRICS["module_gap"], 0))

    def _show_log(self) -> None:
        self._toggle_log()

    def _refresh_input_summary(self) -> None:
        if not hasattr(self, "input_summary"):
            return
        if hasattr(self, "run_destination_summary"):
            self.run_destination_summary.set(
                self.vars["output_root"].get().strip() or "尚未选择"
            )
        mode = self.vars["mode"].get()
        if mode == "grid":
            source = self.vars["source_root"].get().strip()
            self.input_summary.set(f"多格网模式 · 数据目录：{source or '尚未选择'}")
            if hasattr(self, "project_path_display"):
                self.project_path_display.set(source or "尚未选择格网数据目录")
            ready = bool(source and Path(source).is_dir())
            if hasattr(self, "header_state"):
                self.header_state.set("项目数据已就绪" if ready else "等待选择数据")
                self.header_state_label.configure(style="HeaderReady.TLabel" if ready else "HeaderIdle.TLabel")
            if hasattr(self, "_refresh_data_summary"):
                self._refresh_data_summary()
            self.root.after_idle(self._draw_stepper)
            return
        area = self.vars["validation_area"].get().strip()
        periods = [(period, source) for period, source in self._period_values() if period and source]
        if self.project_validation_areas and self.project_area_periods:
            area_count = len(self.project_validation_areas)
            task_count = sum(len(rows) for rows in self.project_area_periods.values())
            self.input_summary.set(f"{area_count} 个验证区 · {task_count} 个区内影像期次 · 生产检测")
            ready = all(len(self.project_area_periods.get(name, [])) >= 2 for name, _path in self.project_validation_areas)
            if hasattr(self, "header_state"):
                self.header_state.set("项目数据已就绪" if ready else "等待选择数据")
                self.header_state_label.configure(style="HeaderReady.TLabel" if ready else "HeaderIdle.TLabel")
            if hasattr(self, "_refresh_data_summary"):
                self._refresh_data_summary()
            self.root.after_idle(self._draw_stepper)
            return
        area_text = Path(area).name if area else "未选择验证区"
        self.input_summary.set(f"{area_text} · {len(periods)} 个影像期次 · 生产检测")
        ready = bool(area and len(periods) >= 2)
        if hasattr(self, "header_state"):
            self.header_state.set("项目数据已就绪" if ready else "等待选择数据")
            self.header_state_label.configure(style="HeaderReady.TLabel" if ready else "HeaderIdle.TLabel")
        if hasattr(self, "_refresh_data_summary"):
            self._refresh_data_summary()
        self.root.after_idle(self._draw_stepper)

    def _field(self, parent, label: str, key: str, browse: str | None = None) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=X, pady=4)
        ttk.Label(row, text=label, width=18).pack(side=LEFT)
        entry = ttk.Entry(row, textvariable=self.vars[key])
        entry.pack(side=LEFT, fill=X, expand=True)
        entry.bind("<FocusOut>", lambda _event: self._refresh_input_summary())
        if browse:
            ttk.Button(row, text="选择…", style="Compact.TButton", command=lambda: self._browse(key, browse)).pack(side=LEFT, padx=(8, 0))

    def _browse(self, key: str, kind: str) -> None:
        value = self._select_path(kind)
        if value:
            self.vars[key].set(value)
            self._refresh_input_summary()

    def _browse_variable(self, variable: StringVar, kind: str) -> None:
        value = self._select_path(kind)
        if value:
            variable.set(value)
            self._refresh_input_summary()

    def _select_path(self, kind: str) -> str:
        if kind == "dir":
            return filedialog.askdirectory(parent=self.root)
        if kind == "shp":
            return filedialog.askopenfilename(
                parent=self.root, title="选择 SHP 文件", filetypes=(("Shapefile", "*.shp"),),
            )
        if kind == "txt":
            return filedialog.askopenfilename(
                parent=self.root, title="选择影像路径 TXT", filetypes=(("TXT 影像路径清单", "*.txt"),),
            )
        return filedialog.askopenfilename(parent=self.root)




























































    def _command(self, args: list[str]) -> None:
        if self.process is not None:
            messagebox.showwarning("已有任务", "当前已有任务在运行，请等待完成。", parent=self.root)
            return
        self.log.delete("1.0", END)
        self.shared_log_status.set("任务启动中…")
        self.recent_log_lines = []
        self.status.set("任务启动中…")
        self.run_status.set("任务启动中，请稍候…")
        self._set_task_start_buttons_enabled(False)
        self.preflight_button.state(["disabled"])
        self._set_stage_buttons_enabled(False)
        self._set_cancel_enabled(True)
        self.active_command = (
            "data-check" if "--data-check-only" in args else
            ("preflight" if "--preflight-only" in args else (args[0] if args else ""))
        )
        if self.active_command == "apply-edits" and hasattr(self, "apply_review_button"):
            self.apply_review_button.state(["disabled"])
        self.cancel_requested = False
        self.task_started_monotonic = time.monotonic()
        self.last_elapsed_seconds = 0.0
        self.progress_completed = 0
        self.progress_total = 0
        self.progress_eta = None
        self.last_complete_payload = None
        self.current_stage_payload = None
        self.active_log_path = None
        if args and args[0] == "all" and "--preflight-only" not in args:
            try:
                output_value = args[args.index("--output-root") + 1]
                run_name = args[args.index("--run-id") + 1] if "--run-id" in args else time.strftime("run_%Y%m%d_%H%M%S")
                layout = (
                    ProjectLayout.from_project(self.project_root_path, output_value)
                    if self.project_root_path else ProjectLayout.from_output(output_value)
                )
                log_dir = layout.logs_root
                log_dir.mkdir(parents=True, exist_ok=True)
                self.active_log_path = log_dir / f"{run_name}.log"
            except (ValueError, IndexError, OSError):
                self.active_log_path = None
        elif args and args[0] == "apply-edits":
            try:
                log_dir = (
                    ProjectLayout.from_project(self.project_root_path, self.vars["output_root"].get()).logs_root
                    if self.project_root_path else
                    ProjectLayout.from_output(self.vars["output_root"].get()).logs_root
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                self.active_log_path = log_dir / f"人工编辑重建_{time.strftime('%Y%m%d_%H%M%S')}.log"
            except (ValueError, IndexError, OSError):
                self.active_log_path = None
        elif args and args[0] in {"evaluate-existing", "evaluate-all-existing"}:
            try:
                log_dir = (
                    ProjectLayout.from_project(self.project_root_path, self.vars["output_root"].get()).logs_root
                    if self.project_root_path else
                    ProjectLayout.from_output(self.vars["output_root"].get()).logs_root
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                self.active_log_path = log_dir / f"精度评价_{time.strftime('%Y%m%d_%H%M%S')}.log"
            except (ValueError, IndexError, OSError):
                self.active_log_path = None
        elif args:
            try:
                log_dir = (
                    ProjectLayout.from_project(self.project_root_path, self.vars["output_root"].get()).logs_root
                    if self.project_root_path else
                    ProjectLayout.from_output(self.vars["output_root"].get()).logs_root
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                self.active_log_path = log_dir / f"{_safe_task_name(args[0])}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            except OSError:
                self.active_log_path = None
        self.progress.configure(maximum=1, value=0)
        if hasattr(self, "review_progress"):
            self.review_progress.configure(maximum=1, value=0)
        self.progress_text.set("0 / 0 · 已用时 00:00:00 · 剩余 --")

        environment = {}
        if self.project_txt_encodings:
            environment["SAMROAD_TXT_ENCODINGS"] = json.dumps(
                self.project_txt_encodings, ensure_ascii=False,
            )
        path_relocations = getattr(self, "project_path_relocations", {})
        if path_relocations:
            environment["SAMROAD_PATH_RELOCATIONS"] = json.dumps(
                path_relocations, ensure_ascii=False,
            )
        try:
            self.task_manager.submit(
                args, log_path=self.active_log_path, environment=environment,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.priority_queue.put(("error", str(exc)))

    def cancel_task(self) -> None:
        process = self.process
        if process is None:
            return
        prompt = (
            "确定停止人工编辑后的增量重建吗？不会重新运行模型；停止时已完成的阶段会保留，"
            "之后可以再次点击“应用编辑并重新生成结果”。"
            if self.active_command == "apply-edits" else
            "确定取消当前任务吗？已经完成的格网和阶段会保留，可使用同一任务名称断点续跑。"
        )
        if not messagebox.askyesno(
            "取消任务",
            prompt,
            parent=self.root,
        ):
            return
        self.cancel_requested = True
        self.status.set("正在取消任务并停止相关子进程…")
        self._set_cancel_enabled(False)
        try:
            self.task_manager.cancel()
        except OSError as exc:
            self.status.set(f"取消失败：{exc}")

    def _append_log(self, stage: str, message: str) -> None:
        line = f"[{stage}] {message}" if stage else message
        self.recent_log_lines.append(line)
        if len(self.recent_log_lines) > 2000:
            remove_count = max(200, len(self.recent_log_lines) - 2000)
            del self.recent_log_lines[:remove_count]
            self._pending_log_delete_lines += remove_count
        self._pending_log_insert_lines.append(line + "\n")
        summary = " ".join(line.split())
        if len(summary) > 110:
            summary = summary[:107] + "…"
        self.shared_log_status.set(summary or "日志已更新")

    def _flush_log_batch(self) -> None:
        if not hasattr(self, "log"):
            self._pending_log_insert_lines.clear()
            self._pending_log_delete_lines = 0
            return
        if self._pending_log_delete_lines:
            self.log.delete("1.0", f"{self._pending_log_delete_lines + 1}.0")
            self._pending_log_delete_lines = 0
        if self._pending_log_insert_lines:
            self.log.insert(END, "".join(self._pending_log_insert_lines))
            self._pending_log_insert_lines.clear()
            self.log.see(END)

    def _set_cancel_enabled(self, enabled: bool) -> None:
        state = ["!disabled"] if enabled else ["disabled"]
        self.cancel_button.state(state)
        if hasattr(self, "review_cancel_button"):
            self.review_cancel_button.state(state)

    def _set_stage_buttons_enabled(self, enabled: bool) -> None:
        for button in getattr(self, "stage_buttons", []):
            button.state(["!disabled"] if enabled else ["disabled"])

    def _set_task_start_buttons_enabled(self, enabled: bool) -> None:
        for button in getattr(self, "task_start_buttons", [self.run_button]):
            button.state(["!disabled"] if enabled else ["disabled"])

    @staticmethod
    def _select_all_text(widget: Text):
        widget.tag_add("sel", "1.0", "end-1c")
        widget.mark_set("insert", "1.0")
        widget.see("insert")
        return "break"

    def _select_all_logs(self, _event=None):
        return self._select_all_text(self.log)

    def copy_all_logs(self) -> None:
        value = self.log.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self.status.set("详细日志已复制到剪贴板。")

    def open_active_log(self) -> None:
        path = self.active_log_path
        if path is None or not path.is_file():
            messagebox.showinfo("暂无日志文件", "当前任务尚未生成日志文件。", parent=self.root)
            return
        try:
            self._open(path)
        except OSError as exc:
            messagebox.showerror("无法打开日志", str(exc), parent=self.root)

    def _last_error_summary(self) -> str:
        for line in reversed(self.recent_log_lines):
            text = line.strip()
            if text and ("Error" in text or "Exception" in text or "错误" in text or "失败" in text):
                return text[-500:]
        return self.recent_log_lines[-1][-500:] if self.recent_log_lines else "请查看详细日志。"

    def _update_progress(self, payload: dict) -> None:
        completed = payload.get("completed")
        total = payload.get("total")
        if completed is not None and total is not None:
            try:
                self.progress_completed = max(0, int(completed))
                self.progress_total = max(0, int(total))
                self.progress.configure(maximum=max(1, self.progress_total), value=self.progress_completed)
                if hasattr(self, "review_progress"):
                    self.review_progress.configure(maximum=max(1, self.progress_total), value=self.progress_completed)
            except (TypeError, ValueError):
                pass
        if payload.get("elapsed_seconds") is not None:
            try:
                self.last_elapsed_seconds = max(self.last_elapsed_seconds, float(payload["elapsed_seconds"]))
            except (TypeError, ValueError):
                pass
        eta = payload.get("eta_seconds")
        try:
            self.progress_eta = float(eta) if eta is not None else None
        except (TypeError, ValueError):
            self.progress_eta = None
        self._refresh_progress_text()

    def _refresh_progress_text(self) -> None:
        elapsed = self.last_elapsed_seconds
        if self.task_started_monotonic is not None and self.process is not None:
            elapsed = max(elapsed, time.monotonic() - self.task_started_monotonic)
        eta_text = format_duration(self.progress_eta) if self.progress_eta is not None else "--"
        self.progress_text.set(
            f"{self.progress_completed} / {self.progress_total} · "
            f"已用时 {format_duration(elapsed)} · 剩余 {eta_text}"
        )

    def _mark_cancelled_state(self) -> None:
        output = Path(self.vars["output_root"].get().strip()).expanduser()
        run_id = self.vars["run_id"].get().strip()
        self.task_manager.mark_cancelled(output, run_id)

    def _poll(self) -> None:
        poll_started = time.perf_counter()
        handled = 0
        priority_handled = 0
        try:
            while True:
                if priority_handled < MAX_PRIORITY_EVENTS_PER_POLL:
                    try:
                        kind, value = self.priority_queue.get_nowait()
                        priority_handled += 1
                    except queue.Empty:
                        if handled >= MAX_QUEUE_EVENTS_PER_POLL or (
                            handled and time.perf_counter() - poll_started >= QUEUE_POLL_TIME_BUDGET_SECONDS
                        ):
                            break
                        kind, value = self.queue.get_nowait()
                        handled += 1
                else:
                    if handled >= MAX_QUEUE_EVENTS_PER_POLL or (
                        handled and time.perf_counter() - poll_started >= QUEUE_POLL_TIME_BUDGET_SECONDS
                    ):
                        break
                    kind, value = self.queue.get_nowait()
                    handled += 1
                if kind == "scan_progress":
                    progress = dict(value)
                    self.data_status.set("正在后台扫描")
                    self.project_scan_summary.set(
                        f"数据源 {progress.get('source_index', 1)}/{progress.get('source_total', 1)}；"
                        f"正在扫描：{progress.get('directory', progress.get('root', ''))}；"
                        f"已遍历 {progress.get('visited_files', 0)} 个文件，发现 "
                        f"SHP {progress.get('shp_count', 0)}、TXT {progress.get('txt_count', 0)}。"
                    )
                    continue
                if kind == "scan_done":
                    self._apply_scan_results(dict(value))
                    continue
                if kind == "scan_cancelled":
                    self.data_status.set("扫描已取消")
                    self.project_scan_summary.set("扫描已取消；上一次完整扫描索引保持不变。")
                    self.status.set(self.project_scan_summary.get())
                    self._finish_scan_ui()
                    continue
                if kind == "scan_error":
                    self.data_status.set("扫描失败")
                    self.project_scan_summary.set(f"数据源扫描失败：{value}")
                    self._finish_scan_ui()
                    messagebox.showerror("数据源不可用", str(value), parent=self.root)
                    continue
                if kind == "backend_event":
                    event = value
                    if not isinstance(event, BackendEvent):
                        self._append_log("后端协议", f"收到未知事件：{event!r}")
                        continue
                    payload = event.payload
                    friendly = self._friendly(payload)
                    self.status.set(friendly)
                    primary_status = structured_task_status(payload)
                    if primary_status is not None:
                        self.current_stage_payload = payload
                        self.run_status.set(primary_status)
                    elif event.kind == "pipeline" or (
                        event.kind == "complete" and event.stage == "all"
                    ):
                        self.current_stage_payload = None
                        self.run_status.set(friendly)
                    elif self.current_stage_payload is None:
                        self.run_status.set(friendly)
                    self._update_progress(payload)
                    if event.kind == "complete":
                        self.last_complete_payload = payload
                    self._append_log(event.stage or event.kind, friendly)
                    if hasattr(self, "handle_evaluation_backend_event"):
                        self.handle_evaluation_backend_event(payload)
                elif kind in {"backend_log", "log"}:
                    self._append_log("日志", str(value))
                elif kind == "backend_protocol_error":
                    detail = dict(value)
                    self._append_log(
                        "后端协议",
                        f"结构化消息解析失败：{detail.get('error', '未知错误')} | {detail.get('line', '')}",
                    )
                elif kind == "editor_stdout":
                    value = str(value)
                    self.editor_stdout_lines.append(value)
                    if len(self.editor_stdout_lines) > 200:
                        del self.editor_stdout_lines[:-200]
                    if value:
                        self._append_log("人工编辑", value)
                elif kind == "editor_stderr":
                    value = str(value)
                    self.editor_stderr_lines.append(value)
                    if len(self.editor_stderr_lines) > 200:
                        del self.editor_stderr_lines[:-200]
                    if value:
                        self._append_log("人工编辑错误", value)
                elif kind == "done":
                    value = str(value)
                    if self.cancel_requested:
                        if self.active_command == "all":
                            self._mark_cancelled_state()
                        if self.active_command == "apply-edits":
                            self.status.set("人工编辑增量重建已停止；已有正式结果未被删除，可再次应用编辑。")
                            self.run_status.set("增量重建已停止；查看上方日志后可再次点击“应用编辑并重新生成结果”。")
                        else:
                            self.status.set("任务已取消；已完成结果和当前位置已保留。")
                            self.run_status.set("任务已取消；再次点击“继续当前任务”将自动从未完成步骤继续。")
                    elif value == "0" and self.active_command in {"preflight", "data-check"}:
                        payload = self.last_complete_payload or {"kind": "complete", "stage": self.active_command}
                        self.preflight_passed = True
                        pending_candidates = sum(len(values) for values in self.project_candidates.values())
                        self.data_status.set("已扫描，存在待确认项" if pending_candidates else "数据已就绪")
                        self._set_task_start_buttons_enabled(True)
                        self.status.set(self._friendly(payload))
                        self.run_status.set(self._friendly(payload))
                        self.preflight_summary.set("所有阻断性检查均已通过，可以开始处理。")
                        for label in getattr(self, "preflight_check_labels", []):
                            label.configure(text="通过")
                        warnings = payload.get("warnings") or []
                        detail = self._friendly(payload)
                        if warnings:
                            detail += "\n\n风险提示：\n" + "\n".join(f"• {item}" for item in warnings[:20])
                            messagebox.showwarning("数据检查完成", detail, parent=self.root)
                        else:
                            messagebox.showinfo("数据检查完成", detail + "\n\n数据已就绪；自动处理开始时仍会独立检查模型、设备、输出目录和磁盘空间。", parent=self.root)
                    elif value == "0" and self.active_command == "apply-edits":
                        self.status.set("人工编辑已应用：道路面和宽度已重建，受影响的相邻期变化检测已重新运行。")
                        self.run_status.set("人工编辑成果已重新生成。")
                        self._show_step(3, force=True)
                    elif value == "0" and self.active_command in {"evaluate-existing", "evaluate-all-existing"}:
                        payload = self.last_complete_payload or {}
                        self.status.set(self._friendly(payload))
                        self.run_status.set("已有变化成果的精度评价已完成。")
                        if hasattr(self, "evaluation_status"):
                            self.evaluation_status.set("精度评价已完成；区域表已自动刷新。")
                        self._show_step(3, force=True)
                    elif value == "0" and self.active_command in {
                        "extract-project-period", "extract-project-all", "change-project-periods", "change",
                        "rerun-period", "rerun-change", "rerun-all-periods", "rerun-all-changes",
                    }:
                        labels = {
                            "extract-project-period": "所选期次道路提取已完成。",
                            "extract-project-all": "全部期次道路提取已完成。",
                            "change-project-periods": "所选变化对检测已完成。",
                            "change": "所选已有变化对已重新检测完成。",
                            "rerun-period": "所选期次已按指定范围重跑完成。",
                            "rerun-change": "所选变化对已重跑完成。",
                            "rerun-all-periods": "全部道路提取已按依赖顺序批量重跑完成。",
                            "rerun-all-changes": "全部相邻变化对已批量重跑完成。",
                        }
                        failure_count = int((self.last_complete_payload or {}).get("failure_count", 0) or 0)
                        if self.active_command == "rerun-all-changes" and failure_count:
                            message = f"批量变化检测已完成，其中 {failure_count} 项失败；其他结果已更新。"
                        elif self.active_command == "rerun-all-periods" and failure_count:
                            message = f"批量道路提取及相关更新已完成，其中 {failure_count} 项失败；其他结果已保留。"
                        else:
                            message = labels[self.active_command]
                        self.status.set(message)
                        self.run_status.set(message)
                        self._show_step(1, force=True)
                    elif value == "0":
                        failure_count = int((self.last_complete_payload or {}).get("failure_count", 0) or 0)
                        if failure_count:
                            self.status.set(f"批量任务已结束，其中 {failure_count} 项失败或因依赖被跳过；可查看任务报告后续跑。")
                            self.run_status.set(f"任务结束，但有 {failure_count} 项失败或跳过。请查看详细日志和任务报告。")
                            self._show_log()
                        else:
                            self.status.set("自动处理已完成；可进行人工编辑，或跳过并直接查看成果。")
                            self.run_status.set("自动处理完成，可进入“人工编辑（可选）”。")
                            self._show_step(2, force=True)
                    else:
                        summary = self._last_error_summary()
                        if self.active_command == "data-check":
                            self.data_status.set("数据检查失败")
                        self.status.set(f"任务失败：{summary}")
                        self.run_status.set(f"任务失败：{summary}")
                        self._show_log()
                    self.process = None
                    self.task_started_monotonic = None
                    self.active_command = ""
                    self._set_task_start_buttons_enabled(True)
                    self.preflight_button.state(["!disabled"])
                    self._set_stage_buttons_enabled(True)
                    if hasattr(self, "apply_review_button"):
                        self.apply_review_button.state(["!disabled"])
                    self._set_cancel_enabled(False)
                    self._refresh_result_availability()
                    self._show_step(self.current_step, force=True)
                else:
                    self.status.set(f"任务异常：{value}")
                    self.run_status.set(f"任务异常：{value}")
                    self._show_log()
                    self.process = None
                    self.task_started_monotonic = None
                    self.active_command = ""
                    self._set_task_start_buttons_enabled(True)
                    self.preflight_button.state(["!disabled"])
                    self._set_stage_buttons_enabled(True)
                    if hasattr(self, "apply_review_button"):
                        self.apply_review_button.state(["!disabled"])
                    self._set_cancel_enabled(False)
        except queue.Empty:
            pass
        self._poll_geometry_editor()
        self._flush_log_batch()
        self._refresh_progress_text()
        delay = 1 if (not self.priority_queue.empty() or not self.queue.empty()) else 100
        self.root.after(delay, self._poll)

    @staticmethod
    def _friendly(payload: dict) -> str:
        kind = payload.get("kind")
        status = payload.get("status")
        elapsed = payload.get("elapsed_seconds")
        elapsed_text = f"，用时 {format_duration(elapsed)}" if elapsed is not None else ""
        if kind == "normalize":
            return (
                f"规范化 {payload.get('period')} 期影像：窗口 "
                f"{payload.get('index')}/{payload.get('total')}，已生成 {payload.get('tile_index')} 个切片。"
            )
        if kind == "preflight":
            return (
                f"检查有效像元覆盖：{Path(str(payload.get('path', '影像'))).name}，"
                f"数据块 {payload.get('index')}/{payload.get('total')}。"
            )
        if kind == "input-list":
            return (
                f"影像路径清单 {Path(str(payload.get('path', ''))).name}："
                f"TXT 编码 {payload.get('encoding', '未知')}，影像 {payload.get('image_count', 0)} 景。"
            )
        if kind == "pipeline" and payload.get("stage") == "数据扫描":
            return f"已识别 {payload.get('grid_count')} 个项目/格网、{payload.get('period_count')} 个影像期次。"
        if kind == "pipeline" and payload.get("stage") == "道路提取":
            if status == "skipped":
                return f"已跳过 {payload.get('grid')} / {payload.get('period')}：{payload.get('reason', '已有完整成果')}。"
            if status == "failed":
                return f"提取失败 {payload.get('grid')} / {payload.get('scope')}：{payload.get('error')}"
            return (
                f"提取格网 {payload.get('grid')} 的 {payload.get('period')} 期道路 "
                f"（格网 {payload.get('grid_index')}/{payload.get('grid_total')}）{elapsed_text}。"
            )
        if kind == "pipeline" and payload.get("stage") == "变化检测":
            if status == "skipped":
                return (
                    f"已跳过 {payload.get('grid')}：{payload.get('before_period')} → "
                    f"{payload.get('after_period')}；{payload.get('reason', '已有完整成果')}"
                )
            if status == "failed":
                return f"变化检测失败 {payload.get('grid')} / {payload.get('scope')}：{payload.get('error')}"
            return f"检测格网 {payload.get('grid')}：{payload.get('before_period')} → {payload.get('after_period')}。"
        if kind == "complete" and payload.get("stage") == "all":
            change_label = "次变化成果" if payload.get("execution_profile") == "fast" else "次变化检测"
            return (
                f"全部完成：{payload.get('grid_count')} 个格网、{payload.get('period_count')} 次提取、"
                f"{payload.get('change_count')} {change_label}、{payload.get('failure_count', 0)} 项失败"
                f"{elapsed_text}。"
            )
        if kind == "complete" and payload.get("stage") == "preflight":
            grid = payload.get("estimated_analysis_grid") or {}
            grid_text = (
                f"；分析网格约 {grid.get('width')} × {grid.get('height')}"
                if grid else ""
            )
            warnings = payload.get("warnings") or []
            return (
                f"输入检查完成：{payload.get('grid_count', 0)} 个项目/格网、"
                f"{payload.get('period_count', 0)} 个期次、{payload.get('image_count', 0)} 张影像，"
                f"输入 {format_bytes(payload.get('input_bytes'))}{grid_text}；"
                f"{len(warnings)} 条风险提示。"
            )
        if kind == "complete" and payload.get("stage") == "apply-edits":
            reruns = payload.get("change_rerun_count", len(payload.get("change_reruns", []) or []))
            return f"人工编辑成果已重建并重新测宽；已重跑 {reruns} 个受影响的相邻期变化对。"
        if kind == "complete" and payload.get("stage") == "evaluate-existing":
            completeness = payload.get("road_centerline_completeness")
            completeness_text = (
                f"，变化道路提取完整度 {format_percentage(completeness)}"
                if completeness not in {None, ""} else ""
            )
            offset = payload.get("centerline_mean_offset_px")
            offset_text = f"，中心线平均偏移距离 {float(offset):.2f} px" if offset not in {None, ""} else ""
            return (
                f"精度评价完成：变化图斑查全率 {format_percentage(payload.get('change_recall', payload.get('change_area_recall', 0)))}，"
                f"变化图斑准确率 {format_percentage(payload.get('change_precision', payload.get('precision', 0)))}"
                f"{completeness_text}{offset_text}，动态过程检测正确率 "
                f"{format_percentage(payload.get('change_type_accuracy', payload.get('type_judgment_accuracy', 0)))}。"
            )
        if kind == "complete" and payload.get("stage") == "evaluate-all-existing":
            completeness = payload.get("road_centerline_completeness")
            completeness_text = (
                f"，变化道路提取完整度 {format_percentage(completeness)}"
                if completeness not in {None, ""} else ""
            )
            offset = payload.get("centerline_mean_offset_px")
            offset_text = f"，中心线平均偏移距离 {float(offset):.2f} px" if offset not in {None, ""} else ""
            return (
                f"总精度评价完成：{payload.get('evaluated_task_count', 0)} 个区域/变化对，"
                f"变化图斑查全率 {format_percentage(payload.get('change_recall', payload.get('change_area_recall', 0)))}，"
                f"变化图斑准确率 {format_percentage(payload.get('change_precision', payload.get('precision', 0)))}"
                f"{completeness_text}{offset_text}，动态过程检测正确率 "
                f"{format_percentage(payload.get('change_type_accuracy', payload.get('type_judgment_accuracy', 0)))}。"
            )
        if kind == "stage" and status == "complete":
            return f"{payload.get('stage', '阶段')}完成{elapsed_text}。"
        if kind == "complete":
            return f"{payload.get('stage', '阶段')}完成{elapsed_text}。"
        return f"{payload.get('stage', kind or '任务')}：{payload.get('status', '处理中')}"































    @staticmethod
    def _open(path: Path) -> None:
        ProjectManager.open_path(path)


def main() -> int:
    enable_windows_high_dpi()
    root = Tk()
    configure_tk_scaling(root)
    configure_ui_fonts(root)
    UserApp(root)
    if os.environ.get("SAMROAD_GUI_SMOKE_TEST") == "1":
        root.update_idletasks()
        print("GUI startup smoke test passed.")
        root.destroy()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
