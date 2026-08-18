from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT.parents[1]
PROJECT_PYTHON = REPO_ROOT / "env" / "samroad_env" / "python.exe"
PROJECT_PYTHONW = REPO_ROOT / "env" / "samroad_env" / "pythonw.exe"


def _ensure_project_environment() -> None:
    """Relaunch a double-clicked .pyw with the repository environment."""
    current = Path(sys.executable).resolve()
    project_interpreters = {
        path.resolve() for path in (PROJECT_PYTHON, PROJECT_PYTHONW) if path.is_file()
    }
    if current in project_interpreters or not PROJECT_PYTHONW.is_file():
        return
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen(
        [str(PROJECT_PYTHONW), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(REPO_ROOT),
        creationflags=flags,
    )
    raise SystemExit(0)


_ensure_project_environment()

import yaml


RUN_TEST = TOOL_ROOT / "run_test.py"
DEFAULT_CONFIG = REPO_ROOT / "config" / "samroad_inference.yaml"
DEFAULT_OUTPUTS = TOOL_ROOT / "outputs"
SETTINGS_PATH = TOOL_ROOT / "launcher_settings.json"
CACHE_FILES = (
    "test_config.json",
    "road_probability.png",
    "original_graph.p",
    "original_edge_scores.csv",
)
PARAMETER_SPECS = (
    ("road_low_threshold", "Road Low Threshold", "ROAD_LOW_THRESHOLD", "--road-low-threshold", 0.20),
    ("max_gap", "Max Gap", "WEAK_RECOVERY_MAX_GAP_PX", "--max-gap", 64.0),
    ("max_extension", "Max Extension", "WEAK_RECOVERY_MAX_EXTENSION_PX", "--max-extension", 48.0),
    ("min_direction_cosine", "Min Direction Cosine", "WEAK_RECOVERY_MIN_DIRECTION_COSINE", "--min-direction-cosine", 0.65),
    ("min_mean_probability", "Min Mean Probability", "WEAK_RECOVERY_MIN_MEAN_PROBABILITY", "--min-mean-probability", 0.20),
    ("min_q25_probability", "Min Q25 Probability", "WEAK_RECOVERY_MIN_Q25_PROBABILITY", "--min-q25-probability", 0.17),
    ("min_weak_fraction", "Min Weak Fraction", "WEAK_RECOVERY_MIN_WEAK_FRACTION", "--min-weak-fraction", 0.80),
    ("min_background_contrast", "Min Background Contrast", "WEAK_RECOVERY_MIN_BACKGROUND_CONTRAST", "--min-background-contrast", 0.08),
    ("auto_score", "Auto Score", "WEAK_RECOVERY_AUTO_SCORE", "--auto-score", 0.62),
)


def load_default_parameters(config_path: Path = DEFAULT_CONFIG) -> dict[str, float]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    profile_name = str(data.get("ROAD_THRESHOLD_PROFILE", "default"))
    profiles = data.get("ROAD_THRESHOLD_PROFILES", {}) or {}
    profile = profiles.get(profile_name, {}) or {}
    defaults = {}
    for name, _label, config_key, _cli, fallback in PARAMETER_SPECS:
        if config_key == "ROAD_LOW_THRESHOLD":
            value = profile.get(config_key, data.get(config_key, fallback))
        else:
            value = data.get(config_key, fallback)
        defaults[name] = float(value)
    return defaults


def missing_cache_files(run_dir: Path) -> list[str]:
    return [name for name in CACHE_FILES if not (run_dir / name).is_file()]


def build_command(
    python_executable: str,
    run_test_path: Path,
    image_path: str,
    run_dir: str,
    device: str,
    batch_size: int,
    parameters: dict[str, float],
    *,
    recovery_only: bool,
) -> list[str]:
    command = [python_executable, str(run_test_path)]
    if recovery_only:
        command.extend(["--recovery-only", "--run-dir", run_dir])
    else:
        command.extend([
            "--image", image_path,
            "--run-dir", run_dir,
            "--device", device.casefold(),
            "--batch-size", str(batch_size),
        ])
    cli_by_name = {name: cli for name, _label, _key, cli, _fallback in PARAMETER_SPECS}
    for name, value in parameters.items():
        if name in cli_by_name:
            command.extend([cli_by_name[name], f"{float(value):g}"])
    return command


def read_result_summary(run_dir: Path) -> dict:
    recovery = json.loads((run_dir / "weak_recovery.json").read_text(encoding="utf-8"))
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    summary = recovery.get("summary", recovery)
    return {
        "strong_edge_count": int(summary.get("strong_edge_count", 0)),
        "weak_candidate_count": int(summary.get("weak_candidate_count", 0)),
        "weak_recovered_edge_count": int(summary.get("weak_recovered_edge_count", 0)),
        "rejected_weak_candidate_count": int(summary.get("rejected_weak_candidate_count", 0)),
        "weak_recovery_seconds": float(timing.get("weak_recovery_seconds", 0.0)),
        "total_seconds": float(timing.get("total_seconds", 0.0)),
    }


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class WeakRecoveryLauncher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.current_process: subprocess.Popen | None = None
        self.running = False
        self.advanced_visible = False
        self.defaults = load_default_parameters()
        self.settings = self._load_settings()

        self.image_var = tk.StringVar(value=str(self.settings.get("image", "")))
        self.output_var = tk.StringVar(value=str(self.settings.get("run_dir", "")))
        available = cuda_available()
        saved_device = str(self.settings.get("device", "CUDA" if available else "CPU")).upper()
        self.device_var = tk.StringVar(value="CPU" if saved_device == "CUDA" and not available else saved_device)
        self.batch_var = tk.StringVar(value=str(self.settings.get("batch_size", 16)))
        self.status_var = tk.StringVar(
            value="CUDA 不可用，已切换到 CPU。" if not available else "等待运行"
        )
        self.summary_var = tk.StringVar(value="尚无运行摘要")
        saved_parameters = self.settings.get("parameters", {})
        self.parameter_vars = {
            name: tk.StringVar(value=f"{float(saved_parameters.get(name, self.defaults[name])):g}")
            for name, _label, _key, _cli, _fallback in PARAMETER_SPECS
        }
        self._build_ui()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.title("Weak Road Recovery Test")
        self.root.geometry("960x720")
        self.root.minsize(820, 620)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=14)
        main.grid(sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(9, weight=1)

        ttk.Label(main, text="Weak Road Recovery Test", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        ttk.Label(main, text="输入影像").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self.image_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(main, text="选择", command=self._choose_image).grid(row=1, column=2)

        ttk.Label(main, text="输出目录").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(main, text="选择", command=self._choose_output).grid(row=2, column=2)

        options = ttk.Frame(main)
        options.grid(row=3, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Label(options, text="设备").grid(row=0, column=0, padx=(0, 6))
        ttk.Combobox(
            options, textvariable=self.device_var, values=("CUDA", "CPU"), state="readonly", width=9
        ).grid(row=0, column=1, padx=(0, 22))
        ttk.Label(options, text="Batch Size").grid(row=0, column=2, padx=(0, 6))
        ttk.Combobox(
            options, textvariable=self.batch_var, values=("8", "16", "32"), width=8
        ).grid(row=0, column=3)

        actions = ttk.Frame(main)
        actions.grid(row=4, column=0, columnspan=3, sticky="w", pady=6)
        self.full_button = ttk.Button(actions, text="完整运行", command=lambda: self._start(False))
        self.full_button.grid(row=0, column=0, padx=(0, 8))
        self.recovery_button = ttk.Button(
            actions, text="只重新弱恢复", command=lambda: self._start(True)
        )
        self.recovery_button.grid(row=0, column=1, padx=(0, 20))
        ttk.Button(actions, text="打开结果目录", command=self._open_run_dir).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(actions, text="打开对比图", command=self._open_compare).grid(row=0, column=3)

        self.advanced_button = ttk.Button(main, text="高级参数 ▼", command=self._toggle_advanced)
        self.advanced_button.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 2))
        self.advanced_frame = ttk.LabelFrame(main, text="弱恢复参数", padding=10)
        self.advanced_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.advanced_frame.columnconfigure(1, weight=1)
        for row, (name, label, _key, _cli, _fallback) in enumerate(PARAMETER_SPECS):
            ttk.Label(self.advanced_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(self.advanced_frame, textvariable=self.parameter_vars[name], width=18).grid(
                row=row, column=1, sticky="w", padx=8, pady=2
            )
        ttk.Button(
            self.advanced_frame, text="恢复默认值", command=self._restore_defaults
        ).grid(row=len(PARAMETER_SPECS), column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.advanced_frame.grid_remove()

        status = ttk.Frame(main)
        status.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 8))
        ttk.Label(status, text="状态：", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        ttk.Label(main, textvariable=self.summary_var, font=("Segoe UI", 10, "bold")).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        self.log = scrolledtext.ScrolledText(main, height=18, wrap="word", state="disabled")
        self.log.grid(row=9, column=0, columnspan=3, sticky="nsew")

    def _load_settings(self) -> dict:
        if not SETTINGS_PATH.is_file():
            return {}
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_settings(self) -> None:
        payload = {
            "image": self.image_var.get().strip(),
            "run_dir": self.output_var.get().strip(),
            "device": self.device_var.get(),
            "batch_size": self.batch_var.get().strip(),
            "parameters": {name: variable.get().strip() for name, variable in self.parameter_vars.items()},
        }
        SETTINGS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择遥感影像",
            filetypes=[
                ("Remote sensing images", "*.tif *.tiff *.png *.jpg *.jpeg *.webp"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.image_var.set(selected)
            self.output_var.set(str((DEFAULT_OUTPUTS / Path(selected).stem).resolve()))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(
            title="选择输出目录", initialdir=self.output_var.get() or str(DEFAULT_OUTPUTS)
        )
        if selected:
            self.output_var.set(selected)

    def _toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.grid()
            self.advanced_button.configure(text="高级参数 ▲")
        else:
            self.advanced_frame.grid_remove()
            self.advanced_button.configure(text="高级参数 ▼")

    def _restore_defaults(self) -> None:
        try:
            self.defaults = load_default_parameters()
            for name, value in self.defaults.items():
                self.parameter_vars[name].set(f"{value:g}")
            self.status_var.set("已重新读取正式 YAML 默认值")
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc), parent=self.root)

    def _parameters(self) -> dict[str, float]:
        values = {}
        probability_names = {
            "road_low_threshold", "min_direction_cosine", "min_mean_probability",
            "min_q25_probability", "min_weak_fraction", "min_background_contrast", "auto_score",
        }
        for name, variable in self.parameter_vars.items():
            try:
                value = float(variable.get())
            except ValueError as exc:
                raise ValueError(f"{name} 必须是数字") from exc
            if name in probability_names and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
            if name in {"max_gap", "max_extension"} and value <= 0:
                raise ValueError(f"{name} 必须大于 0")
            values[name] = value
        return values

    def _start(self, recovery_only: bool) -> None:
        if self.running:
            return
        image = self.image_var.get().strip()
        run_dir_text = self.output_var.get().strip()
        if not run_dir_text:
            messagebox.showwarning("缺少输出目录", "请选择或输入输出目录。", parent=self.root)
            return
        run_dir = Path(run_dir_text).expanduser().resolve()
        if not recovery_only and (not image or not Path(image).expanduser().is_file()):
            messagebox.showwarning("缺少输入影像", "请选择有效的输入遥感影像。", parent=self.root)
            return
        if recovery_only:
            missing = missing_cache_files(run_dir)
            if missing:
                messagebox.showwarning(
                    "缺少测试缓存",
                    "当前目录没有完整测试缓存，请先执行一次完整运行。\n\n缺少：" + ", ".join(missing),
                    parent=self.root,
                )
                return
        try:
            batch_size = int(self.batch_var.get())
            if batch_size <= 0:
                raise ValueError
            parameters = self._parameters()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc) or "Batch Size 必须是正整数。", parent=self.root)
            return
        command = build_command(
            sys.executable, RUN_TEST, image, str(run_dir), self.device_var.get(), batch_size,
            parameters, recovery_only=recovery_only,
        )
        self._save_settings()
        self._set_running(True)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.status_var.set("正在运行" if recovery_only else "正在加载模型")
        self._append_log("Starting recovery-only" if recovery_only else "Starting full inference")
        self._append_log(f"Run directory: {run_dir}")
        if not recovery_only:
            self._append_log(f"Image: {image}")
            self._append_log(f"Device: {self.device_var.get().lower()}")
            self._append_log(f"Batch size: {batch_size}")
        threading.Thread(
            target=self._run_process, args=(command, run_dir), daemon=True
        ).start()

    def _run_process(self, command: list[str], run_dir: Path) -> None:
        try:
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.current_process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=flags,
            )
            assert self.current_process.stdout is not None
            for line in self.current_process.stdout:
                self.events.put(("log", line.rstrip()))
            return_code = self.current_process.wait()
            self.events.put(("done", return_code, run_dir, ""))
        except Exception as exc:
            self.events.put(("done", -1, run_dir, str(exc)))
        finally:
            self.current_process = None

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "log":
                    line = event[1]
                    self._append_log(line)
                    if "Loading model" in line:
                        self.status_var.set("正在加载模型")
                    elif "Processing patches" in line or "Weak recovery" in line:
                        self.status_var.set("正在运行")
                elif event[0] == "done":
                    self._finish(event[1], event[2], event[3])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _finish(self, return_code: int, run_dir: Path, error: str) -> None:
        self._set_running(False)
        if return_code != 0:
            self.status_var.set("运行失败")
            detail = error or f"run_test.py exited with code {return_code}"
            self._append_log(detail)
            messagebox.showerror("运行失败", detail, parent=self.root)
            return
        try:
            summary = read_result_summary(run_dir)
            self.summary_var.set(
                "原始边数：{strong_edge_count}    弱候选：{weak_candidate_count}    "
                "成功恢复边：{weak_recovered_edge_count}    拒绝候选：{rejected_weak_candidate_count}    "
                "Weak Recovery：{weak_recovery_seconds:.2f}s    总耗时：{total_seconds:.2f}s".format(**summary)
            )
        except Exception as exc:
            self.summary_var.set(f"结果已生成，但摘要读取失败：{exc}")
        self.status_var.set("运行完成")
        messagebox.showinfo("测试完成", f"输出目录：\n{run_dir}", parent=self.root)

    def _set_running(self, running: bool) -> None:
        self.running = running
        state = "disabled" if running else "normal"
        self.full_button.configure(state=state)
        self.recovery_button.configure(state=state)

    def _append_log(self, message: str) -> None:
        if not message:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{stamp}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open_run_dir(self) -> None:
        path = Path(self.output_var.get().strip()).expanduser()
        if not path.is_dir():
            messagebox.showinfo("尚无测试结果", "尚无测试结果。", parent=self.root)
            return
        os.startfile(path.resolve())

    def _open_compare(self) -> None:
        path = Path(self.output_var.get().strip()).expanduser() / "recovery_compare.png"
        if not path.is_file():
            messagebox.showinfo("尚无对比图", "请先运行测试。", parent=self.root)
            return
        os.startfile(path.resolve())

    def _on_close(self) -> None:
        if self.running and not messagebox.askyesno(
            "任务正在运行", "测试仍在运行，确定终止并关闭吗？", parent=self.root
        ):
            return
        self._save_settings()
        if self.current_process is not None and self.current_process.poll() is None:
            self.current_process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    WeakRecoveryLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
