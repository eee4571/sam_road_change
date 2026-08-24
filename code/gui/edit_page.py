from __future__ import annotations
import json
import os
import queue
import tempfile
import time
from pathlib import Path
from tkinter import filedialog, messagebox

from app.editor_manager import EditorManager

ROOT = Path(__file__).resolve().parents[1]
from tkinter import BOTH, LEFT, RIGHT, X, StringVar
from tkinter import ttk

from .common_widgets import LAYOUT_METRICS

class EditPage:
    def _editor_service(self) -> EditorManager:
        manager = getattr(self, "editor_manager", None)
        if manager is None:
            manager = EditorManager(getattr(self, "queue", queue.Queue()))
            self.editor_manager = manager
            compat_process = getattr(self, "_compat_editor_process", None)
            if compat_process is not None:
                manager.process = compat_process
        return manager

    def _build_review_page(self, page: ttk.Frame) -> None:
        self.review_body = page
        review_card = ttk.LabelFrame(page, text="人工编辑", padding=LAYOUT_METRICS["card_padding"])
        review_card.pack(fill=X)
        self.review_status = StringVar(value="完成自动处理后，可在此选择需要编辑的期次。")
        selector = ttk.Frame(review_card)
        selector.pack(fill=X, pady=(0, 5))
        ttk.Label(selector, text="区域 / 期次：", width=14).pack(side=LEFT)
        self.review_selection = StringVar()
        self.review_combo = ttk.Combobox(selector, textvariable=self.review_selection, state="readonly")
        self.review_combo.pack(side=LEFT, fill=X, expand=True)
        self.review_combo.bind("<<ComboboxSelected>>", self._review_selection_changed)
        self.review_detail = StringVar(value="暂无可复核数据。")
        status_row = ttk.Frame(review_card)
        status_row.pack(fill=X, pady=3)
        ttk.Label(status_row, text="当前状态：", width=14).pack(side=LEFT)
        ttk.Label(status_row, textvariable=self.review_status, anchor="w", wraplength=520).pack(side=LEFT, fill=X, expand=True)
        edit_row = ttk.Frame(review_card)
        edit_row.pack(fill=X, pady=3)
        ttk.Label(edit_row, text="项目编辑目录：", width=14).pack(side=LEFT)
        ttk.Label(edit_row, textvariable=self.review_edit_directory, anchor="w").pack(side=LEFT, fill=X, expand=True)
        ttk.Label(review_card, textvariable=self.review_detail, wraplength=620).pack(anchor="w", fill=X, pady=(3, 6))
        actions = ttk.Frame(review_card)
        actions.pack(fill=X)
        self.launch_review_button = ttk.Button(actions, text="打开编辑工作台", style="Hero.TButton", command=self.launch_selected_review_editor)
        self.launch_review_button.pack(side=LEFT)
        self.apply_review_button = ttk.Button(actions, text="应用编辑并更新相关结果", style="Primary.TButton", command=self.apply_selected_review)
        self.apply_review_button.pack(side=LEFT, padx=5)
        self.review_advanced_toggle = ttk.Button(review_card, text="高级操作...", command=self._toggle_review_advanced)
        self.review_advanced_toggle.pack(anchor="w", pady=(6, 0))
        self.review_advanced_frame = ttk.Frame(review_card)
        ttk.Button(self.review_advanced_frame, text="导入外部编辑成果", command=self.select_review_edit_directory).pack(side=LEFT)
        ttk.Button(self.review_advanced_frame, text="打开编辑资料目录", command=self.open_selected_review_folder).pack(side=LEFT, padx=(5, 0))

        self.review_task_frame = ttk.LabelFrame(page, text="编辑后增量重建", padding=LAYOUT_METRICS["card_padding"])
        self.review_task_frame.pack(fill=BOTH, expand=True, pady=(LAYOUT_METRICS["section_gap"], 0))
        review_task_header = ttk.Frame(self.review_task_frame)
        review_task_header.pack(fill=X)
        ttk.Label(review_task_header, text="当前状态：").pack(side=LEFT)
        self.review_cancel_button = ttk.Button(
            review_task_header, text="停止重建", command=self.cancel_task,
        )
        self.review_cancel_button.pack(side=RIGHT)
        self.review_cancel_button.state(["disabled"])
        ttk.Label(
            self.review_task_frame, textvariable=self.run_status, wraplength=620,
        ).pack(anchor="w", fill=X, pady=(5, 6))
        self.review_progress = ttk.Progressbar(
            self.review_task_frame, mode="determinate", maximum=1, value=0,
            style="Modern.Horizontal.TProgressbar",
        )
        self.review_progress.pack(fill=X)
        ttk.Label(
            self.review_task_frame, textvariable=self.progress_text,
        ).pack(anchor="w", pady=(5, 7))
        ttk.Label(
            self.review_task_frame,
            text="应用编辑后将重新读取编辑成果，执行测宽、道路面更新、受影响变化检测及长时序成果更新。\n详细输出显示在右侧“全流程日志”。",
            style="CardMuted.TLabel",
        ).pack(anchor="w")
        ttk.Label(page, text="人工编辑为可选步骤；跳过时将直接采用自动处理结果。", style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

        summary = ttk.LabelFrame(self.step_summaries[2], text="编辑任务信息", padding=LAYOUT_METRICS["card_padding"])
        summary.pack(fill=BOTH, expand=True)
        ttk.Label(summary, text="当前对象：", width=12).grid(row=0, column=0, sticky="nw", pady=2)
        ttk.Label(summary, textvariable=self.review_selection, wraplength=400).grid(row=0, column=1, sticky="nw", pady=2)
        ttk.Label(summary, text="对象状态：", width=12).grid(row=1, column=0, sticky="nw", pady=2)
        ttk.Label(summary, textvariable=self.review_status, wraplength=400).grid(row=1, column=1, sticky="nw", pady=2)
        ttk.Label(summary, text="编辑目录：", width=12).grid(row=2, column=0, sticky="nw", pady=2)
        ttk.Label(summary, textvariable=self.review_edit_directory, wraplength=400).grid(row=2, column=1, sticky="nw", pady=2)
        ttk.Label(summary, text="说明：", width=12).grid(row=3, column=0, sticky="nw", pady=2)
        ttk.Label(summary, textvariable=self.review_detail, wraplength=400).grid(row=3, column=1, sticky="nw", pady=2)
        summary.grid_columnconfigure(1, weight=1)

    def open_optional_review(self) -> None:
        self._refresh_result_availability()
        if not self.review_items:
            messagebox.showinfo(
                "暂无可选复核资料",
                "最新任务没有可用的人工编辑目录。自动流程不会因人工编辑暂停；如需编辑，请重新运行支持编辑资料输出的任务。",
                parent=self.root,
            )
            return
        item = self.review_items[0]
        directory = Path(item["directory"])
        decisions = item.get("decisions", "")
        try:
            self._open(directory)
        except OSError as exc:
            messagebox.showerror("无法打开复核资料", str(exc), parent=self.root)
            return
        self.status.set(f"已打开人工编辑资料目录：{item['label']}。")
        detail = f"已打开：\n{directory}\n\n这里只提供第 3 步人工编辑所需资料，不存在另一套逐项人工处理流程。"
        if decisions:
            detail += f"\n复核决定文件：\n{decisions}"
        messagebox.showinfo("可选人工编辑", detail, parent=self.root)

    def _geometry_editor_script(self) -> Path | None:
        # The user-facing package must remain independently distributable.
        # Never fall back to a development-tree editor outside this folder.
        packaged_editor = ROOT / "engine" / "width" / "geometry_editor.py"
        return packaged_editor if packaged_editor.is_file() else None

    def open_review_workflow(self) -> None:
        """Enter the optional review step directly, without an intermediate launcher."""
        self._refresh_result_availability()
        self._show_step(2, force=True)

    def _populate_review_step(self) -> None:
        if not hasattr(self, "review_combo"):
            return
        labels = [item["label"] for item in self.review_items]
        self.review_combo.configure(values=labels)
        if not labels:
            self.review_selection.set("")
            self.review_combo.configure(state="disabled")
            self.review_detail.set(
                "当前没有可编辑成果。完成至少一个期次的道路处理或载入已有成果后，"
                "可在此进行人工编辑。"
            )
            self.review_edit_directory.set("暂无可用编辑目录")
            self.review_status.set("当前没有可编辑成果；人工编辑是可选步骤。")
            self.launch_review_button.state(["disabled"])
            self.apply_review_button.state(["disabled"])
            return
        self.review_combo.configure(state="readonly")
        if self.review_combo.current() < 0:
            self.review_combo.current(0)
        pending = 0
        for item in self.review_items:
            try:
                pending += max(0, int(item.get("manual_item_count", 0) or 0))
            except (TypeError, ValueError):
                pass
        pending_text = f"，共有 {pending} 处结果可供核查" if pending else ""
        self.review_status.set(
            f"本次任务有 {len(self.review_items)} 个影像期次可进行人工编辑{pending_text}。"
            "如有修改，请保存后重新生成相关结果。"
        )
        self.launch_review_button.state(["!disabled"])
        self.apply_review_button.state(["!disabled"])
        self._review_selection_changed()

    def _selected_review_item(self) -> dict[str, str] | None:
        if not self.review_items:
            return None
        index = self.review_combo.current() if hasattr(self, "review_combo") else 0
        return self.review_items[index if 0 <= index < len(self.review_items) else 0]

    def _review_selection_changed(self, _event=None) -> None:
        item = self._selected_review_item()
        if item is None:
            self.review_detail.set("暂无可复核数据。")
            return
        saved, _checked = self.task_manager.find_saved_edits(item)
        state = "已人工编辑，待更新" if saved else "可编辑"
        result_path = self.task_manager.find_period_result(item)
        if result_path is not None:
            try:
                period_result = json.loads(result_path.read_text(encoding="utf-8"))
                if (period_result.get("manual_edit") or {}).get("applied"):
                    state = "已更新"
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        region = item.get("grid", "")
        periods = [period for period, _source in self.project_area_periods.get(region, [])]
        pairs = self.task_manager.affected_change_pairs(periods, item.get("scope", ""))
        pair_text = "；".join(f"{before} → {after}：需要重新计算" for before, after in pairs) or "无相邻变化对"
        self.review_detail.set(
            f"期次状态：{state}    ·    受影响变化对：{pair_text}"
        )
        self.review_edit_directory.set(str(saved or item.get("edited_directory") or "尚未保存编辑成果"))

    def select_review_edit_directory(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="选择已保存的人工编辑成果目录")
        if selected:
            self.review_edit_directory.set(str(Path(selected).resolve()))

    def open_selected_review_folder(self) -> None:
        item = self._selected_review_item()
        if item is not None:
            self._open(Path(item["directory"]))

    def _clear_geometry_editor_state(self) -> None:
        ready_file = self.editor_ready_file
        self.editor_process = None
        self.editor_ready_file = None
        self.editor_started_monotonic = None
        self.editor_launch_state = "idle"
        self.editor_timeout_reported = False
        if ready_file is not None:
            try:
                ready_file.unlink(missing_ok=True)
            except OSError:
                pass
        if hasattr(self, "launch_review_button") and self.review_items:
            self.launch_review_button.state(["!disabled"])

    def _poll_geometry_editor(self) -> None:
        process = self.editor_process
        if process is None:
            return
        if self.editor_launch_state == "ready":
            returncode = process.poll()
            if returncode is not None:
                self._append_log("人工编辑", f"人工编辑器已关闭，退出码 {returncode}。")
                self.status.set("人工编辑器已关闭；如已保存编辑，可应用并更新相关结果。")
                self._clear_geometry_editor_state()
            return
        started = self.editor_started_monotonic
        if started is None:
            started = time.monotonic()
            self.editor_started_monotonic = started
        state, detail = self._editor_service().process_state(
            process, self.editor_ready_file, started,
        )
        if state == "starting":
            return
        if state == "loading":
            if not self.editor_timeout_reported:
                self.editor_timeout_reported = True
                message = "人工编辑器仍在加载，但尚未完成窗口初始化。"
                self.status.set(message)
                if hasattr(self, "review_status"):
                    self.review_status.set(message)
                self._append_log("人工编辑", message + "进程仍在运行，将继续等待 ready 信号。")
            return
        if state == "ready":
            self.editor_launch_state = "ready"
            message = "人工编辑器已打开。完成编辑并保存后，可返回此处更新相关结果。"
            self.status.set(message)
            if hasattr(self, "review_status"):
                self.review_status.set(message)
            self._append_log(
                "人工编辑",
                f"编辑器窗口已就绪（PID {detail.get('pid', process.pid)}）。",
            )
            if self.editor_ready_file is not None:
                try:
                    self.editor_ready_file.unlink(missing_ok=True)
                except OSError:
                    pass
                self.editor_ready_file = None
            return
        raw_returncode = detail.get("returncode")
        returncode = int(raw_returncode) if raw_returncode is not None else None
        error_lines = self.editor_stderr_lines[-20:] or self.editor_stdout_lines[-20:]
        error_text = str(detail.get("error") or "\n".join(error_lines)).strip() or "编辑器未输出错误详情。"
        exit_text = f"退出码：{returncode}" if returncode is not None else "编辑器窗口报告加载失败"
        message = f"人工编辑器未能进入可编辑状态。{exit_text}\n\n{error_text}"
        self.status.set(
            f"人工编辑器启动失败（退出码 {returncode}）。"
            if returncode is not None else "人工编辑器数据加载失败。"
        )
        if hasattr(self, "review_status"):
            self.review_status.set("人工编辑器启动失败；请查看全流程日志。")
        self._append_log("人工编辑", message.replace("\n", " | "))
        messagebox.showerror("人工编辑器启动失败", message, parent=self.root)
        self._show_log()
        self._clear_geometry_editor_state()

    def launch_selected_review_editor(self) -> None:
        item = self._selected_review_item()
        if item is None:
            return
        if self.editor_process is not None:
            if self.editor_process.poll() is None:
                messagebox.showinfo(
                    "人工编辑器正在运行",
                    "当前人工编辑器仍在启动或运行，请先使用现有窗口。",
                    parent=self.root,
                )
                return
            self._clear_geometry_editor_state()
        script = self._geometry_editor_script()
        if script is None:
            messagebox.showerror("缺少中心线编辑器", "未找到 geometry_editor.py。", parent=self.root)
            return
        review_dir = Path(item["directory"]).expanduser().resolve()
        diagnostics = self._editor_service().diagnostics(review_dir)
        final_centerlines = Path(item.get("final_centerlines") or "")
        final_surfaces = Path(item.get("final_surfaces") or "")
        if not final_centerlines.is_file():
            diagnostics.append(f"最终融合中心线不存在：{final_centerlines}")
        if not final_surfaces.is_file():
            diagnostics.append(f"最终道路面不存在：{final_surfaces}")
        if diagnostics:
            messagebox.showerror(
                "中心线编辑器输入不可用",
                "启动前检查失败，未修改任何结果文件：\n\n" + "\n".join(diagnostics),
                parent=self.root,
            )
            return
        ready_file = Path(tempfile.gettempdir()) / (
            f"samroad_geometry_editor_ready_{os.getpid()}_{time.time_ns()}.json"
        )
        try:
            edited_dir = Path(
                item.get("edited_directory") or review_dir.parent / "centerline_edit"
            ).expanduser().resolve()
            edited_dir.mkdir(parents=True, exist_ok=True)
            self.review_edit_directory.set(str(edited_dir))
            ready_file.unlink(missing_ok=True)
            process = self._editor_service().launch(script, item, ready_file)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法启动中心线编辑器", str(exc), parent=self.root)
            return
        self.editor_process = process
        self.editor_ready_file = ready_file
        self.editor_started_monotonic = time.monotonic()
        self.editor_launch_state = "starting"
        self.editor_timeout_reported = False
        self.editor_stdout_lines = []
        self.editor_stderr_lines = []
        self.launch_review_button.state(["disabled"])
        self.status.set("正在启动人工编辑器，请稍候…")
        if hasattr(self, "review_status"):
            self.review_status.set("正在启动人工编辑器，请稍候…")
        self._append_log("人工编辑", f"正在启动人工编辑器：{review_dir}")

    def apply_selected_review(self) -> None:
        item = self._selected_review_item()
        if item is None:
            return
        result_path = self.task_manager.find_period_result(item)
        if result_path is None:
            messagebox.showerror(
                "缺少期次结果索引",
                "无法定位该期次的 latest_result.json。请载入与本次人工编辑对应的任务结果索引。",
                parent=self.root,
            )
            return
        edited_dir, checked = self.task_manager.find_saved_edits(
            item, self.review_edit_directory.get(),
        )
        if edited_dir is None:
            checked_text = "\n".join(f"• {path}" for path in checked[:8]) or "• 未提供编辑目录"
            messagebox.showerror(
                "尚未找到已保存的编辑成果",
                "请先在编辑工作台执行“保存全部”。系统已自动检查当前项目与期次的编辑目录：\n"
                + checked_text + "\n\n如需兼容外部编辑成果，可展开“高级操作”后导入。",
                parent=self.root,
            )
            return
        self.review_edit_directory.set(str(edited_dir))
        item = {**item, "result": str(result_path), "edited_directory": str(edited_dir)}
        _manifest, latest = self._latest_manifest()
        candidates: list[Path] = []
        if latest is not None:
            candidates.append(Path(latest))
        for parent in result_path.parents:
            candidates.extend((parent / "pipeline_result.json", parent / "latest_pipeline.json"))
        matching_manifest = next(
            (path for path in candidates if self.task_manager.manifest_contains_period(path, result_path)), None,
        )
        try:
            args = self.task_manager.build_apply_edits(item, matching_manifest)
        except ValueError as exc:
            messagebox.showerror("缺少结果索引", str(exc), parent=self.root)
            return
        if matching_manifest is None:
            self.status.set("已找到编辑成果，但任务总索引与该期次不匹配；本次将重新测宽和生成面，不自动重跑相邻变化对。")
        self._command(args)
