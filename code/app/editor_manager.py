from __future__ import annotations

"""Geometry-editor process boundary and input diagnostics."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def _rebase_editor_summary_path(path: Path, summary_path: Path) -> Path | None:
    """Map a stale path from an old result root into the current run tree."""
    for current_run in summary_path.parents:
        if not current_run.name.casefold().startswith("run_"):
            continue
        for index, part in enumerate(path.parts):
            if part.casefold() != current_run.name.casefold():
                continue
            candidate = current_run.joinpath(*path.parts[index + 1:])
            if candidate.is_file():
                return candidate.resolve()
    sibling = summary_path.parent / path.name
    if sibling.is_file():
        return sibling.resolve()
    return None


def _resolve_editor_summary_path(value: object, summary_path: Path) -> Path:
    """Resolve a geometry-editor summary path using the editor's path rules.

    Current summaries use absolute paths. Older summaries may retain absolute
    paths under the former ``04_成果输出/run_*`` tree after the run was moved to
    ``_work/tasks/runs/run_*``. Resolve those paths read-only against the run
    containing the summary before falling back to the frozen value.
    """
    raw = str(value or "").strip()
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        summary_relative = (summary_path.parent / path).resolve()
        if summary_relative.is_file():
            return summary_relative
    rebased = _rebase_editor_summary_path(path, summary_path)
    if rebased is not None:
        return rebased
    return path.resolve()

def collect_geometry_editor_inputs(review_dir: Path | str) -> list[dict[str, object]]:
    """Collect image/graph inputs that the root geometry editor will open.

    This is deliberately read-only: it parses review summaries and reports the
    exact paths without copying or rewriting the user's result files.  Keeping
    this resolver in the GUI makes stale/malformed manifests diagnosable before
    launching the optional editor.
    """
    review_path = Path(review_dir).expanduser().resolve()
    if not review_path.is_dir():
        return []
    inputs: list[dict[str, object]] = []
    for summary_path in sorted(review_path.glob("*_summary.json")):
        if summary_path.name.startswith("batch_") or summary_path.name.endswith("_optimized_summary.json"):
            continue
        stem = summary_path.name.removesuffix("_summary.json")
        entry: dict[str, object] = {"stem": stem, "summary": summary_path}
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            entry["error"] = f"无法读取摘要 {summary_path}: {exc}"
            inputs.append(entry)
            continue
        if not isinstance(summary, dict):
            entry["error"] = f"摘要不是 JSON 对象: {summary_path}"
            inputs.append(entry)
            continue
        image_path = _resolve_editor_summary_path(summary.get("image"), summary_path)
        graph_path = _resolve_editor_summary_path(
            summary.get("prepared_graph") or summary.get("graph"), summary_path
        )
        entry.update({
            "image": image_path,
            "prepared_graph": graph_path,
            "image_exists": image_path.is_file(),
            "prepared_graph_exists": graph_path.is_file(),
        })
        inputs.append(entry)
    return inputs

def geometry_editor_diagnostics(review_dir: Path | str) -> list[str]:
    """Return actionable preflight diagnostics for optional geometry editing."""
    review_path = Path(review_dir).expanduser().resolve()
    if not review_path.is_dir():
        return [f"复核目录不存在：{review_path}"]
    entries = collect_geometry_editor_inputs(review_path)
    if not entries:
        return [f"复核目录中没有 *_summary.json：{review_path}"]
    diagnostics: list[str] = []
    for entry in entries:
        summary = entry.get("summary", review_path)
        error = entry.get("error")
        if error:
            diagnostics.append(str(error))
            continue
        image_path = entry.get("image")
        graph_path = entry.get("prepared_graph")
        if not entry.get("image_exists"):
            diagnostics.append(f"影像不存在（{summary}）：{image_path}")
        if not entry.get("prepared_graph_exists"):
            diagnostics.append(f"prepared graph 不存在（{summary}）：{graph_path}")
    return diagnostics

def build_geometry_editor_command(
    script: Path | str, item: dict[str, str], ready_file: Path | str | None = None,
) -> list[str]:
    """Build the exact optional-editor command from one manifest review item."""
    script_path = Path(script).expanduser().resolve()
    review_path = Path(item["directory"]).expanduser().resolve()
    edited_value = item.get("edited_directory") or str(review_path.parent / "centerline_edit")
    edited_path = Path(edited_value).expanduser().resolve()
    final_centerlines = str(item.get("final_centerlines") or "").strip()
    final_surfaces = str(item.get("final_surfaces") or "").strip()
    if not final_centerlines:
        raise ValueError("该期次缺少最终融合中心线 SHP，无法按正式成果进行人工编辑。")
    if not final_surfaces:
        final_surfaces = str(Path(final_centerlines).with_name("road_surfaces.shp"))
    command = [
        sys.executable,
        str(script_path),
        "--review-dir",
        str(review_path),
        "--edited-dir",
        str(edited_path),
        "--final-centerlines",
        str(Path(final_centerlines).expanduser().resolve()),
        "--final-surfaces",
        str(Path(final_surfaces).expanduser().resolve()),
    ]
    if ready_file:
        command.extend(("--ready-file", str(Path(ready_file).expanduser().resolve())))
    return command

def geometry_editor_process_state(
    process: subprocess.Popen[str], ready_file: Path | None,
    started_monotonic: float, now: float | None = None, timeout: float = 60.0,
) -> tuple[str, dict[str, object]]:
    """Return the non-blocking geometry-editor launcher state."""
    if ready_file is not None and ready_file.is_file():
        try:
            payload = json.loads(ready_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict) and payload.get("status") == "ready":
            return "ready", payload
        if isinstance(payload, dict) and payload.get("status") == "failed":
            return "failed", payload
    returncode = process.poll()
    if returncode is not None:
        return "failed", {"returncode": int(returncode)}
    elapsed = (time.monotonic() if now is None else now) - started_monotonic
    if elapsed >= timeout:
        return "loading", {"elapsed_seconds": max(0.0, elapsed)}
    return "starting", {"elapsed_seconds": max(0.0, elapsed)}

class EditorManager:
    """Own the optional editor process without depending on Tk widgets."""

    def __init__(self, event_queue) -> None:
        self.event_queue = event_queue
        self.process: subprocess.Popen[str] | None = None

    def launch(self, script: Path | str, item: dict[str, str], ready_file: Path | str):
        script_path = Path(script).expanduser().resolve()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            build_geometry_editor_command(script_path, item, Path(ready_file)),
            cwd=str(script_path.parent), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        self.process = process
        assert process.stdout is not None and process.stderr is not None
        for kind, stream in (("editor_stdout", process.stdout), ("editor_stderr", process.stderr)):
            threading.Thread(target=self._read_stream, args=(kind, stream), daemon=True).start()
        return process

    def _read_stream(self, kind: str, stream) -> None:
        try:
            for line in stream:
                self.event_queue.put((kind, line.rstrip("\r\n")))
        finally:
            stream.close()

    def clear(self) -> None:
        self.process = None

    @staticmethod
    def diagnostics(review_dir: Path | str) -> list[str]:
        return geometry_editor_diagnostics(review_dir)

    @staticmethod
    def process_state(process, ready_file, started_monotonic, **kwargs):
        return geometry_editor_process_state(
            process, ready_file, started_monotonic, **kwargs,
        )
