from __future__ import annotations

"""Subprocess boundary between the Tkinter application and ``user_pipeline``.

This module deliberately has no Tk imports.  Worker threads publish plain Python
events to queues; the Tk main thread decides how those events are displayed.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


STRUCTURED_PREFIX = "__SAMROAD_USER__"
_HARMLESS_TIFF_WARNING = re.compile(
    r"TIFFReadDirectory:\s*Unknown field with tag\s+(?:33550|33922|34735|34737)\b"
)


@dataclass(frozen=True)
class BackendEvent:
    """One structured message emitted by ``user_pipeline.py``."""

    kind: str
    message: str = ""
    stage: str | None = None
    area_id: str | None = None
    period: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BackendEvent":
        return cls(
            kind=str(payload.get("kind") or "event"),
            message=str(payload.get("message") or ""),
            stage=str(payload["stage"]) if payload.get("stage") is not None else None,
            area_id=str(payload.get("grid") or payload.get("area_id") or "") or None,
            period=str(payload["period"]) if payload.get("period") is not None else None,
            payload=dict(payload),
        )


def parse_backend_line(line: str) -> BackendEvent | None:
    """Parse one structured stdout line, returning ``None`` for ordinary logs."""
    value = str(line).rstrip("\r\n")
    if not value.startswith(STRUCTURED_PREFIX):
        return None
    payload = json.loads(value[len(STRUCTURED_PREFIX):])
    if not isinstance(payload, dict):
        raise ValueError("后端结构化消息必须是 JSON 对象")
    return BackendEvent.from_payload(payload)


def is_harmless_gui_log(line: str) -> bool:
    return bool(_HARMLESS_TIFF_WARNING.search(str(line)))


def write_and_filter_gui_log(line: str, log_file: TextIO | None = None) -> str | None:
    """Always persist stdout while suppressing harmless TIFF noise in the widget."""
    if log_file is not None:
        log_file.write(line)
        log_file.flush()
    visible = str(line).rstrip("\r\n")
    return None if is_harmless_gui_log(visible) else visible


class BackendClient:
    """Own the backend process, stdout protocol, logging and cancellation."""

    def __init__(
        self,
        *,
        app_root: Path | str,
        backend_script: Path | str | None = None,
        python_executable: Path | str | None = None,
        runtime_root: Path | str | None = None,
        event_queue: queue.Queue | None = None,
        priority_queue: queue.Queue | None = None,
    ) -> None:
        self.app_root = Path(app_root).expanduser().resolve()
        self.backend_script = Path(
            backend_script or self.app_root / "user_pipeline.py"
        ).expanduser().resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.runtime_root = Path(
            runtime_root or self.app_root.parent / "runtime"
        ).expanduser().resolve()
        self.event_queue = event_queue if event_queue is not None else queue.Queue()
        self.priority_queue = priority_queue if priority_queue is not None else queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self._starting = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        process = self.process
        return self._starting or (process is not None and process.poll() is None)

    def command_line(self, args: list[str] | tuple[str, ...]) -> list[str]:
        return [self.python_executable, str(self.backend_script), *map(str, args)]

    def environment(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        site_packages = self.runtime_root / "env" / "samroad_env" / "Lib" / "site-packages"
        proj_data = site_packages / "rasterio" / "proj_data"
        gdal_data = site_packages / "rasterio" / "gdal_data"
        if (proj_data / "proj.db").is_file():
            env["PROJ_DATA"] = str(proj_data)
            env["PROJ_LIB"] = str(proj_data)
        if gdal_data.is_dir():
            env["GDAL_DATA"] = str(gdal_data)
        if overrides:
            env.update({str(key): str(value) for key, value in overrides.items()})
        return env

    def start(
        self,
        args: list[str],
        *,
        log_path: Path | str | None = None,
        environment: dict[str, str] | None = None,
    ) -> threading.Thread:
        """Start asynchronously and publish ``backend_event/log/done/error`` tuples."""
        with self._lock:
            if self.running:
                raise RuntimeError("当前已有后端任务在运行")
            self._starting = True
        resolved_log = Path(log_path).expanduser().resolve() if log_path else None

        def worker() -> None:
            log_file: TextIO | None = None
            try:
                process = subprocess.Popen(
                    self.command_line(args),
                    cwd=str(self.app_root),
                    env=self.environment(environment),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
                    ),
                )
                self.process = process
                if resolved_log is not None:
                    resolved_log.parent.mkdir(parents=True, exist_ok=True)
                    log_file = resolved_log.open("a", encoding="utf-8", newline="")
                    log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                assert process.stdout is not None
                for line in process.stdout:
                    visible = write_and_filter_gui_log(line, log_file)
                    if visible is None:
                        continue
                    try:
                        event = parse_backend_line(visible)
                    except (json.JSONDecodeError, ValueError) as exc:
                        self.priority_queue.put(("backend_protocol_error", {
                            "line": visible, "error": str(exc),
                        }))
                        continue
                    if event is None:
                        self.event_queue.put(("backend_log", visible))
                    else:
                        self.priority_queue.put(("backend_event", event))
                self.priority_queue.put(("done", str(process.wait())))
            except Exception as exc:  # worker errors must cross the queue boundary
                self.priority_queue.put(("error", str(exc)))
            finally:
                if log_file is not None:
                    log_file.close()
                with self._lock:
                    self._starting = False

        thread = threading.Thread(target=worker, name="samroad-backend", daemon=True)
        thread.start()
        return thread

    # Semantic entry points keep pages independent of CLI executable details.
    def run_preflight(self, args: list[str], **kwargs) -> threading.Thread:
        return self.start(args, **kwargs)

    def run_full_project(self, args: list[str], **kwargs) -> threading.Thread:
        return self.start(args, **kwargs)

    def run_period(self, args: list[str], **kwargs) -> threading.Thread:
        return self.start(args, **kwargs)

    def run_all_periods(self, args: list[str], **kwargs) -> threading.Thread:
        return self.start(args, **kwargs)

    def run_change_pair(self, args: list[str], **kwargs) -> threading.Thread:
        return self.start(args, **kwargs)

    def apply_edits(self, args: list[str], **kwargs) -> threading.Thread:
        return self.start(args, **kwargs)

    def evaluate(self, args: list[str], **kwargs) -> threading.Thread:
        return self.start(args, **kwargs)

    def cancel(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            process.terminate()
