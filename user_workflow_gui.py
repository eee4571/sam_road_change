from __future__ import annotations

import json
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
import ctypes
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Canvas, Menu, StringVar, TclError, Text, Tk, Toplevel, filedialog, font as tkfont, messagebox, simpledialog
from tkinter import ttk

from input_catalog import period_order_manifest, period_sort_key



ROOT = Path(__file__).resolve().parent
BACKEND = Path(__file__).with_name("user_pipeline.py")
DEFAULT_CONFIG = ROOT / "config" / "samroad_inference.yaml"
DEFAULT_CKPT = ROOT / "models" / "samroad" / "samroad.ckpt"
DEFAULT_TEST_DATA = ROOT / "功能测试数据"
USER_VECTOR_SUFFIX = ".shp"
USER_IMAGE_LIST_SUFFIX = ".txt"
PROJECT_CONFIG_NAME = "project_config.json"

_HARMLESS_TIFF_WARNING = re.compile(
    r"TIFFReadDirectory:\s*Unknown field with tag\s+(?:33550|33922|34735|34737)\b"
)

PREVIEW_LABELS = {
    "centerline": "中心线提取",
    "surface": "路面提取",
    "fusion": "融合",
    "width": "重新测宽",
    "change": "最终变化结果",
    "review_change": "待复核变化",
}

WORKFLOW_STEPS = (
    "数据准备",
    "自动处理",
    "人工编辑（可选）",
    "成果与评价",
)

UI = {
    # Warm ivory + forest green, matching the restrained GIS/industrial
    # application language used by the reference interface.
    "ink": "#18332A",
    "muted": "#68746E",
    "subtle": "#929A94",
    "page": "#F3F0E8",
    "card": "#FCFBF7",
    "line": "#D5D0C5",
    "line_strong": "#AAA79F",
    "blue": "#0B755C",
    "blue_hover": "#075C48",
    "blue_soft": "#E7F1EC",
    "green": "#0B755C",
    "green_soft": "#E7F1EC",
    "amber": "#A66A1F",
    "amber_soft": "#F5EBDD",
    "slate_soft": "#F5F2EA",
    "header": "#1D4034",
    "header_deep": "#17362C",
    "header_text": "#F8F5EC",
    "header_muted": "#AFC3B9",
    "viewer": "#162A23",
}

# Interactive controls use three deliberate size tiers. Semantic aliases may
# change colour and weight, but must not introduce another control height.
CONTROL_METRICS = {
    "primary": {"font": ("Microsoft YaHei UI", 10, "bold"), "padding": (16, 9)},
    "regular": {"font": ("Microsoft YaHei UI", 10), "padding": (12, 7)},
    "compact": {"font": ("Microsoft YaHei UI", 9), "padding": (9, 6)},
}

# Shared spacing keeps the existing restrained desktop layout consistent without
# introducing a second theme or scattering per-widget magic numbers.
LAYOUT_METRICS = {
    "page_padding": (20, 16, 20, 18),
    "card_padding": (18, 15),
    "section_gap": 14,
    "module_gap": 12,
    "form_gap": 5,
    "form_label_width": 16,
    "content_wrap": 1040,
}


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


def configure_window_geometry(
    root: Tk, *, base_width: int, base_height: int,
    min_width: int, min_height: int,
) -> float:
    """Size and center a window while preserving usable logical proportions."""
    scale = max(1.0, min(float(root.winfo_fpixels("1i")) / 96.0, 2.5))
    screen_width = max(1, root.winfo_screenwidth())
    screen_height = max(1, root.winfo_screenheight())
    width = min(round(screen_width * 0.94), round(base_width * scale))
    height = min(round(screen_height * 0.90), round(base_height * scale))
    minimum_width = min(round(screen_width * 0.88), round(min_width * scale))
    minimum_height = min(round(screen_height * 0.82), round(min_height * scale))
    root.minsize(max(860, minimum_width), max(580, minimum_height))
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    return scale


def _manifest_path(value: object, base_dir: Path | None = None) -> Path | None:
    """Return a manifest path, accepting the frozen string or a small path mapping."""
    if isinstance(value, dict):
        value = value.get("path") or value.get("file") or value.get("image")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip()).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _manifest_true(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def _display_scope(entry: dict, change: bool = False) -> tuple[str, str]:
    grid = str(entry.get("grid") or "未命名格网")
    if grid == "validation":
        grid = "验证区项目"
    if change:
        before = str(entry.get("before_period") or "前期")
        after = str(entry.get("after_period") or "后期")
        return grid, f"{before} → {after}"
    return grid, str(entry.get("period") or "未命名期次")


def collect_preview_items(manifest: dict, base_dir: Path | None = None) -> list[dict[str, str]]:
    """Collect final-result and clearly separated review previews.

    Old manifests simply yield an empty list; callers can then explain that the
    task must be rerun to populate the new ``previews`` fields.
    """
    items: list[dict[str, str]] = []
    for raw_entry in manifest.get("period_results", []) or []:
        if not isinstance(raw_entry, dict):
            continue
        grid, scope = _display_scope(raw_entry)
        previews = raw_entry.get("previews")
        if not isinstance(previews, dict):
            continue
        # The review/result pages deliberately omit extraction intermediates.
        # Fusion is the authoritative per-period centerline result and width is
        # the corresponding finalized measurement view.
        for key in ("fusion", "width"):
            path = _manifest_path(previews.get(key), base_dir)
            if path is None or not path.is_file():
                continue
            items.append({
                "label": f"{grid} · {scope} · {PREVIEW_LABELS[key]}",
                "category": PREVIEW_LABELS[key],
                "grid": grid,
                "scope": scope,
                "path": str(path),
                "detail": (
                    "原中心线 {original_edge_count} 条边；融合后 {optimized_edge_count} 条边；"
                    "自动断连 {auto_gap_count} 条；道路面骨架新增 {auto_surface_count} 条；"
                    "人工编辑切片 {geometry_edited_tile_count} 个。"
                ).format(**{
                    name: int((raw_entry.get("fusion") or {}).get(name, 0) or 0)
                    for name in (
                        "original_edge_count", "optimized_edge_count", "auto_gap_count",
                        "auto_surface_count", "geometry_edited_tile_count",
                    )
                }) if key == "fusion" and isinstance(raw_entry.get("fusion"), dict) else "",
            })
    for raw_entry in manifest.get("change_results", []) or []:
        if not isinstance(raw_entry, dict):
            continue
        grid, scope = _display_scope(raw_entry, change=True)
        previews = raw_entry.get("previews")
        if not isinstance(previews, dict):
            continue
        for key in ("change", "review_change"):
            path = _manifest_path(previews.get(key), base_dir)
            if path is None or not path.is_file():
                continue
            items.append({
                "label": f"{grid} · {scope} · {PREVIEW_LABELS[key]}",
                "category": PREVIEW_LABELS[key],
                "grid": grid,
                "scope": scope,
                "path": str(path),
            })
    return items


def collect_temporal_items(manifest: dict, base_dir: Path | None = None) -> list[dict[str, str]]:
    """Collect long-sequence SHP products for direct GUI attribute browsing."""
    items: list[dict[str, str]] = []
    for entry in manifest.get("temporal_results", []) or []:
        if not isinstance(entry, dict):
            continue
        life = _manifest_path(entry.get("life_shp"), base_dir)
        if life is None or not life.is_file():
            continue
        items.append({
            "label": f"{entry.get('grid', 'validation')} · {entry.get('period_count', 0)} 期 · road_life",
            "grid": str(entry.get("grid", "validation")),
            "path": str(life),
            "road_count": str(entry.get("road_count", 0)),
            "period_count": str(entry.get("period_count", 0)),
        })
    return items


def collect_review_items(manifest: dict, base_dir: Path | None = None) -> list[dict[str, str]]:
    """Collect optional review directories without making review a pipeline gate."""
    items: list[dict[str, str]] = []
    for raw_entry in manifest.get("period_results", []) or []:
        if not isinstance(raw_entry, dict):
            continue
        review = raw_entry.get("review")
        if not isinstance(review, dict) or not _manifest_true(review.get("available")):
            continue
        directory = _manifest_path(review.get("directory"), base_dir)
        if directory is None or not directory.is_dir():
            continue
        grid, scope = _display_scope(raw_entry)
        count = review.get("manual_item_count", 0)
        try:
            count_text = str(max(0, int(count)))
        except (TypeError, ValueError):
            count_text = str(count or "未知")
        decisions = _manifest_path(review.get("decisions"), base_dir)
        edited_directory = _manifest_path(
            review.get("edited_directory") or str(directory.parent / "centerline_edit"), base_dir
        )
        result_path = _manifest_path(raw_entry.get("result"), base_dir)
        final_centerlines = _manifest_path(raw_entry.get("centerlines"), base_dir)
        final_surfaces = _manifest_path(raw_entry.get("surfaces"), base_dir)
        if final_centerlines is None or not final_centerlines.is_file():
            continue
        items.append({
            "label": f"{grid} · {scope}",
            "grid": grid,
            "scope": scope,
            "directory": str(directory),
            "decisions": str(decisions) if decisions is not None else "",
            "edited_directory": str(edited_directory) if edited_directory is not None else "",
            "result": str(result_path) if result_path is not None else "",
            "final_centerlines": str(final_centerlines),
            "final_surfaces": str(final_surfaces) if final_surfaces is not None else "",
            "manual_item_count": count_text,
        })
    return items


def _resolve_editor_summary_path(value: object, summary_path: Path) -> Path:
    """Resolve a geometry-editor summary path using the editor's path rules.

    Current summaries use absolute paths.  The fallback for old summaries keeps
    the useful ``summary_dir``-relative form before resolving from the process
    working directory, matching how the bundled editor consumes these fields.
    """
    raw = str(value or "").strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    summary_relative = (summary_path.parent / path).resolve()
    if summary_relative.is_file():
        return summary_relative
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


def natural_key(value: str) -> tuple:
    return period_sort_key(value)


def format_duration(value: object) -> str:
    try:
        seconds = max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return "--"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def is_harmless_gui_log(line: str) -> bool:
    """Filter only known TIFF metadata chatter from the GUI, never from disk logs."""
    return bool(_HARMLESS_TIFF_WARNING.search(str(line)))


def write_and_filter_gui_log(line: str, log_file=None) -> str | None:
    """Persist the original subprocess line and return its optional GUI form."""
    if log_file is not None:
        log_file.write(line)
        log_file.flush()
    visible = line.rstrip("\r\n")
    return None if is_harmless_gui_log(visible) else visible


def structured_task_status(payload: dict) -> str | None:
    """Build the primary period-stage display without parsing plain log text."""
    if payload.get("kind") != "stage":
        return None
    grid = str(payload.get("grid") or "").strip()
    period = str(payload.get("period") or "").strip()
    stage = str(payload.get("stage") or "").strip()
    if not grid or not period or not stage:
        return None
    try:
        index = int(payload.get("stage_index"))
        total = int(payload.get("stage_total"))
    except (TypeError, ValueError):
        return None
    return (
        f"验证区：{grid}\n"
        f"期次：{period}\n"
        f"当前步骤：{stage}\n"
        f"步骤进度：{index} / {total}"
    )


def unfinished_task_state(output_root: Path | str, active_task: dict | None) -> dict | None:
    active = active_task if isinstance(active_task, dict) else {}
    run_id = str(active.get("run_id") or "").strip()
    if not run_id:
        return None
    state_value = str(active.get("state") or "").strip()
    state_path = Path(state_value).expanduser() if state_value else Path(output_root).expanduser() / _safe_task_name(run_id) / "job_state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or state.get("status") in {"completed", "completed_with_errors"}:
        return None
    period_state_value = str(state.get("period_state") or "").strip()
    if period_state_value and Path(period_state_value).expanduser().is_file():
        try:
            period_state = json.loads(Path(period_state_value).expanduser().read_text(encoding="utf-8"))
            for source_key, target_key in (
                ("grid", "current_grid"), ("period", "current_period"),
                ("current_stage", "current_stage"),
                ("current_stage_label", "current_stage_label"),
                ("last_completed_stage", "last_completed_stage"),
                ("last_completed_stage_label", "last_completed_stage_label"),
            ):
                if period_state.get(source_key) not in (None, ""):
                    state[target_key] = period_state[source_key]
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            pass
    state["state_path"] = str(state_path)
    return state


def unfinished_task_message(state: dict) -> str:
    grid = str(state.get("current_grid") or "未知验证区")
    period = str(state.get("current_period") or "未知期次")
    stage = str(state.get("current_stage_label") or state.get("current_stage") or "尚未记录")
    return (
        "检测到未完成任务\n\n"
        f"上次停止：{grid} · {period} · {stage}\n\n"
        "继续后将复用已完成结果，从未完成步骤继续。"
    )


def mark_task_cancelled(output_root: Path | str, run_id: str) -> dict | None:
    """Atomically mark a task cancelled while leaving every produced artifact intact."""
    run_id = str(run_id or "").strip()
    if not run_id:
        return None
    output = Path(output_root).expanduser()
    job_root = output / run_id
    state_path = job_root / "job_state.json"
    if not state_path.is_file():
        return None
    try:
        manifest = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
            return None
        manifest["status"] = "cancelled"
        manifest["cancelled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        for target in (state_path, job_root / "pipeline_result.json", output / "latest_pipeline.json"):
            atomic_write_json(target, manifest)
        period_state_value = str(manifest.get("period_state") or "").strip()
        if period_state_value:
            period_state_path = Path(period_state_value).expanduser()
            if period_state_path.is_file():
                period_state = json.loads(period_state_path.read_text(encoding="utf-8"))
                period_state["status"] = "cancelled"
                period_state["cancelled_at"] = manifest["cancelled_at"]
                atomic_write_json(period_state_path, period_state)
        return manifest
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None


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


def format_percentage(value: object, digits: int = 1) -> str:
    """Format a stored 0-1 metric for display without changing its value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number * 100:.{digits}f}%"


def atomic_write_json(path: Path | str, value: dict) -> Path:
    """Write a project-side JSON file without exposing a partial document."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def project_config_path(project_root: Path | str) -> Path:
    return Path(project_root).expanduser().resolve() / PROJECT_CONFIG_NAME


def read_project_config(project_root: Path | str) -> dict:
    """Read the new project config while accepting an absent legacy config."""
    path = project_config_path(project_root)
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"项目配置根节点必须是 JSON 对象：{path}")
    return value


def affected_change_pairs(periods: list[str] | tuple[str, ...], selected: str) -> list[tuple[str, str]]:
    """Return the adjacent changes invalidated by replacing one period result."""
    ordered = sorted({str(value).strip() for value in periods if str(value).strip()}, key=period_sort_key)
    return [
        (before, after)
        for before, after in zip(ordered, ordered[1:])
        if selected in {before, after}
    ]


def scan_external_data_source(source_dir: Path | str) -> dict:
    """Scan an external source without moving or rewriting any source file."""
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"外部数据源不存在：{root}")
    try:
        discovered = discover_validation_project(root)
    except ValueError:
        discovered = None
    candidates = {
        "shp": [str(path.resolve()) for path in sorted(root.rglob("*.shp"), key=lambda item: natural_key(str(item)))],
        "txt": [str(path.resolve()) for path in sorted(root.rglob("*.txt"), key=lambda item: natural_key(str(item)))],
    }
    return {"root": str(root), "discovered": discovered, "candidates": candidates}


def collect_result_tree_items(manifest: dict, base_dir: Path | None = None) -> list[dict[str, str]]:
    """Build one stable result-browser model for new and historical manifests."""
    items: list[dict[str, str]] = []
    area_nodes: set[str] = set()

    def add(node_id: str, parent: str, label: str, path_value: object = "") -> None:
        path = _manifest_path(path_value, base_dir)
        exists = bool(path and path.exists())
        items.append({
            "id": node_id, "parent": parent, "label": label,
            "path": str(path) if path is not None else "",
            "status": "已生成" if exists else "未生成",
        })

    def ensure_area(area: str) -> str:
        node = f"area:{area}"
        if node not in area_nodes:
            area_nodes.add(node)
            items.append({"id": node, "parent": "", "label": area, "path": "", "status": ""})
        return node

    for index, entry in enumerate(manifest.get("period_results", []) or []):
        if not isinstance(entry, dict):
            continue
        area = str(entry.get("grid") or "validation")
        parent = ensure_area(area)
        period = str(entry.get("period") or f"期次 {index + 1}")
        period_id = f"{parent}:period:{period}"
        items.append({"id": period_id, "parent": parent, "label": period, "path": "", "status": str(entry.get("status") or "")})
        add(f"{period_id}:centerline", period_id, "道路中心线", entry.get("centerlines"))
        add(f"{period_id}:surface", period_id, "道路面", entry.get("surfaces"))
        add(f"{period_id}:width", period_id, "道路宽度", entry.get("gpkg"))
    for index, entry in enumerate(manifest.get("change_results", []) or []):
        if not isinstance(entry, dict):
            continue
        area = str(entry.get("grid") or "validation")
        parent = ensure_area(area)
        before, after = str(entry.get("before_period") or "前期"), str(entry.get("after_period") or "后期")
        pair_id = f"{parent}:change:{before}:{after}:{index}"
        items.append({"id": pair_id, "parent": parent, "label": f"{before} → {after}", "path": "", "status": str(entry.get("status") or "")})
        add(f"{pair_id}:result", pair_id, "变化检测数据", entry.get("gpkg") or entry.get("summary") or entry.get("output"))
        previews = entry.get("previews") if isinstance(entry.get("previews"), dict) else {}
        add(f"{pair_id}:formal-preview", pair_id, "最终变化结果", previews.get("change"))
        add(f"{pair_id}:review-preview", pair_id, "待复核变化", previews.get("review_change"))
    temporal_by_area = {
        str(entry.get("grid") or "validation"): entry
        for entry in (manifest.get("temporal_results", []) or []) if isinstance(entry, dict)
    }
    for area_node in sorted(area_nodes):
        area = area_node.removeprefix("area:")
        temporal = temporal_by_area.get(area, {})
        add(f"{area_node}:temporal", area_node, "长时序道路成果", temporal.get("life_shp"))
    job_root = _manifest_path(manifest.get("job_root"), base_dir)
    report = job_root / "task_report.csv" if job_root is not None else None
    items.append({
        "id": "task-report", "parent": "", "label": "任务报告",
        "path": str(report) if report is not None else "",
        "status": "已生成" if report is not None and report.is_file() else "未生成",
    })
    return items


def discover_validation_project(project_dir: Path | str) -> dict:
    """Discover nested per-area projects, with the former flat layout as fallback."""
    root = Path(project_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"项目文件夹不存在：{root}")

    def named_directory(parent: Path, prefix: str, fallbacks: tuple[str, ...]) -> Path | None:
        candidates = [
            child for child in parent.iterdir()
            if child.is_dir() and (child.name.startswith(prefix) or child.name.casefold() in fallbacks)
        ]
        return sorted(candidates, key=lambda path: natural_key(path.name))[0] if candidates else None

    def period_truth_rows(area_root: Path) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
        imagery_dir = named_directory(area_root, "02_", ("images", "imagery", "影像"))
        if imagery_dir is None:
            raise ValueError(f"验证区文件夹中未找到 02_影像：{area_root}")
        periods = [
            (path.stem, str(path))
            for path in sorted(imagery_dir.iterdir(), key=lambda item: natural_key(item.name))
            if path.is_file() and path.suffix.lower() == USER_IMAGE_LIST_SUFFIX
        ]
        if len(periods) < 2:
            raise ValueError(f"影像文件夹中少于两个期次 TXT：{imagery_dir}")
        truth_dir = named_directory(area_root, "03_", ("truth", "ground_truth", "变化真值"))
        truths: dict[tuple[str, str], str] = {}
        if truth_dir is not None:
            for path in truth_dir.iterdir():
                if not path.is_file() or path.suffix.lower() != USER_VECTOR_SUFFIX:
                    continue
                match = re.match(r"^(.+?)_to_(.+)$", path.stem, flags=re.IGNORECASE)
                if match:
                    truths[(match.group(1), match.group(2))] = str(path)
        return sorted(periods, key=lambda item: natural_key(item[0])), truths

    output_dir = named_directory(root, "04_", ("outputs", "output", "成果输出")) or root / "04_成果输出"
    flat_area_dir = named_directory(root, "01_", ("validation", "validation_area", "验证区"))
    flat_imagery_dir = named_directory(root, "02_", ("images", "imagery", "影像"))
    areas: list[tuple[str, str]] = []
    area_periods: dict[str, list[tuple[str, str]]] = {}
    area_truths: dict[tuple[str, str, str], str] = {}
    truth_files: dict[tuple[str, str], str] = {}
    if flat_area_dir is not None and flat_imagery_dir is not None:
        area_files = sorted(flat_area_dir.glob("*.shp"), key=lambda path: natural_key(path.name))
        if not area_files:
            raise ValueError(f"验证区文件夹中没有 SHP 文件：{flat_area_dir}")
        periods, truth_files = period_truth_rows(root)
        for path in area_files:
            area_id = path.stem
            areas.append((area_id, str(path)))
            area_periods[area_id] = list(periods)
        truth_dir = named_directory(root, "03_", ("truth", "ground_truth", "变化真值"))
        for area_id, _path in areas:
            for (before, after), truth in truth_files.items():
                area_truths[(area_id, before, after)] = truth
            if truth_dir is not None:
                child = truth_dir / area_id
                for path in child.glob("*.shp") if child.is_dir() else []:
                    match = re.match(r"^(.+?)_to_(.+)$", path.stem, flags=re.IGNORECASE)
                    if match:
                        area_truths[(area_id, match.group(1), match.group(2))] = str(path)
    else:
        candidates = [
            child for child in sorted(root.iterdir(), key=lambda path: natural_key(path.name))
            if child.is_dir()
            and named_directory(child, "01_", ("validation", "validation_area", "验证区")) is not None
            and named_directory(child, "02_", ("images", "imagery", "影像")) is not None
        ]
        if not candidates:
            raise ValueError("未找到验证区子文件夹；每个验证区下应包含 01_验证区 和 02_影像。")
        for area_root in candidates:
            boundary_dir = named_directory(area_root, "01_", ("validation", "validation_area", "验证区"))
            boundaries = sorted(boundary_dir.glob("*.shp"), key=lambda path: natural_key(path.name))
            if len(boundaries) != 1:
                raise ValueError(f"每个验证区文件夹的 01_验证区 必须且只能有一个 SHP：{boundary_dir}")
            area_id = area_root.name
            periods, truths = period_truth_rows(area_root)
            areas.append((area_id, str(boundaries[0])))
            area_periods[area_id] = periods
            for (before, after), truth in truths.items():
                area_truths[(area_id, before, after)] = truth
        first_area = areas[0][0]
        truth_files = {
            (before, after): truth for (area, before, after), truth in area_truths.items()
            if area == first_area
        }
    first_area = areas[0][0]
    return {
        "project_root": str(root),
        "validation_area": areas[0][1],
        "validation_areas": areas,
        "periods": area_periods[first_area],
        "area_periods": area_periods,
        "truths": truth_files,
        "area_truths": area_truths,
        "output_root": str(output_dir),
    }


def ordered_period_pairs(period_rows: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Validate, naturally order periods and derive exactly the adjacent pairs."""
    periods: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_period, raw_source in period_rows:
        period = str(raw_period).strip()
        source = str(raw_source).strip()
        if not period or not source:
            raise ValueError("每个期次都必须填写期次名称和影像路径 TXT。")
        if period in seen:
            raise ValueError(f"期次名称重复：{period}")
        seen.add(period)
        periods.append((period, source))
    if len(periods) < 2:
        raise ValueError("验证区模式至少需要两个影像期次。")
    periods.sort(key=lambda item: period_sort_key(item[0]))
    pairs = [(before[0], after[0]) for before, after in zip(periods, periods[1:])]
    return periods, pairs


def build_pipeline_command(
    *, mode: str, output_root: str, checkpoint: str, config: str, device: str,
    pixel_size: str, rescale: str, absolute: str, ratio: str, tolerance: str,
    run_id: str = "", validation_area: str = "", periods: list[tuple[str, str]] | None = None,
    truths: list[tuple[str, str, str]] | None = None, truth_type_field: str = "",
    source_root: str = "", evaluate: bool = True, resume: bool = False,
    continue_on_error: bool = True, preflight_only: bool = False,
    data_check_only: bool = False, runtime_preflight: bool = True,
    junction_node_mode: str = "sparse",
    validation_areas: list[tuple[str, str]] | None = None,
    area_truths: list[tuple[str, str, str, str]] | None = None,
    area_periods: dict[str, list[tuple[str, str]]] | None = None,
) -> list[str]:
    """Build the backend command for the default validation or backup grid mode."""
    mode = str(mode or "validation").strip().casefold()
    if not str(output_root).strip():
        raise ValueError("请选择成果输出根目录。")
    args = ["all", "--mode", mode]
    if mode == "validation":
        legacy_single_area = validation_areas is None
        area_rows = validation_areas or [((Path(validation_area).stem or "validation"), validation_area)]
        normalized_areas = []
        seen_areas = set()
        for raw_name, raw_path in area_rows:
            name = str(raw_name).strip()
            area = Path(str(raw_path).strip()).expanduser()
            if not name or name in seen_areas:
                raise ValueError(f"验证区名称为空或重复：{name}")
            if not area.is_file():
                raise ValueError(f"请选择存在的验证区 SHP 文件：{area}")
            if area.suffix.lower() != USER_VECTOR_SUFFIX:
                raise ValueError("验证区必须使用 SHP 文件（.shp）。")
            seen_areas.add(name)
            normalized_areas.append((name, area))
        ordered_by_area: dict[str, list[tuple[str, str]]] = {}
        pairs_by_area: dict[str, list[tuple[str, str]]] = {}
        for name, _area in normalized_areas:
            ordered, area_pairs = ordered_period_pairs(
                (area_periods or {}).get(name, periods or [])
            )
            ordered_by_area[name] = ordered
            pairs_by_area[name] = area_pairs
            for period, source_value in ordered:
                source = Path(source_value).expanduser()
                if not source.is_file():
                    raise ValueError(f"找不到 {name} / {period} 期影像路径 TXT：{source}")
                if source.suffix.lower() != USER_IMAGE_LIST_SUFFIX:
                    raise ValueError(f"{name} / {period} 期影像必须使用内含影像路径的 TXT 文件：{source}")
        truth_map: dict[tuple[str, str, str], str] = {}
        supplied_area_truths = area_truths or [
            (normalized_areas[0][0], before, after, path) for before, after, path in (truths or [])
        ]
        for area_name, before, after, raw_path in supplied_area_truths:
            key = (str(area_name).strip(), str(before).strip(), str(after).strip())
            if key in truth_map:
                raise ValueError(f"变化真值重复：{key[0]} / {key[1]} → {key[2]}")
            truth_map[key] = str(raw_path).strip()
        expected_truths = {
            (name, before, after) for name, _area in normalized_areas
            for before, after in pairs_by_area[name]
        }
        if evaluate and (set(truth_map) != expected_truths or any(not truth_map.get(key) for key in expected_truths)):
            missing = [f"{name} / {before} → {after}" for name, before, after in sorted(expected_truths) if not truth_map.get((name, before, after))]
            raise ValueError("请为每个相邻期次选择变化真值：" + "、".join(missing or ["期次对应关系不一致"]))
        for name, area in normalized_areas:
            if legacy_single_area:
                args.extend(("--validation-area", str(area)))
            else:
                args.extend(("--validation-area", name, str(area)))
            for period, source in ordered_by_area[name]:
                args.extend(("--period", period, source) if legacy_single_area else ("--period", name, period, source))
        if truth_map:
            for name, _area in normalized_areas:
                for before, after in pairs_by_area[name]:
                    raw_truth = truth_map.get((name, before, after), "")
                    if not raw_truth:
                        continue
                    truth = Path(raw_truth).expanduser()
                    if not truth.is_file():
                        raise ValueError(f"找不到 {name} / {before} → {after} 的变化真值：{truth}")
                    if truth.suffix.lower() != USER_VECTOR_SUFFIX:
                        raise ValueError(f"{name} / {before} → {after} 的变化真值必须使用 SHP 文件（.shp）：{truth}")
                    args.extend(
                        ("--truth", before, after, str(truth)) if legacy_single_area
                        else ("--truth", name, before, after, str(truth))
                    )
        if not evaluate:
            args.append("--no-evaluation")
        if truth_map and str(truth_type_field).strip():
            args.extend(("--truth-type-field", str(truth_type_field).strip()))
    elif mode == "grid":
        source = Path(str(source_root).strip()).expanduser()
        if not source.is_dir():
            raise ValueError("请选择存在的格网数据根目录。")
        args.extend(("--source-root", str(source)))
    else:
        raise ValueError("输入模式必须是 validation 或 grid。")
    args.extend((
        "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--config", str(config),
        "--device", str(device), "--pixel-size", str(pixel_size), "--rescale", str(rescale),
        "--junction-node-mode", str(junction_node_mode or "sparse"),
        "--absolute", str(absolute), "--ratio", str(ratio), "--tolerance", str(tolerance),
    ))
    if str(run_id).strip():
        args.extend(("--run-id", str(run_id).strip()))
    if resume:
        if not str(run_id).strip():
            raise ValueError("断点续跑必须填写原任务名称。")
        args.append("--resume")
    if continue_on_error:
        args.append("--continue-on-error")
    if preflight_only:
        args.append("--preflight-only")
    if data_check_only:
        args.append("--data-check-only")
    elif runtime_preflight and not preflight_only:
        args.append("--runtime-preflight")
    return args


def build_apply_edits_command(item: dict[str, str], pipeline_manifest: Path | str | None = None) -> list[str]:
    result = str(item.get("result") or "").strip()
    if not result:
        raise ValueError("该期次缺少 result 索引，请重新运行任务。")
    args = ["apply-edits", "--result", result, "--edited-dir", str(item.get("edited_directory") or "")]
    if pipeline_manifest:
        args.extend(("--pipeline-manifest", str(pipeline_manifest)))
    return args


def _safe_task_name(value: str) -> str:
    """Return the path component used by the existing project-stage backend."""
    normalized = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).strip())
    return normalized.strip("._-") or "workspace"


def resolve_automatic_run(
    output_root: Path | str, requested_run_id: str = "", active_task: dict | None = None,
    *, generated_run_id: str | None = None,
) -> tuple[str, bool, Path]:
    """Choose a new or resumable task without exposing resume as a task type."""
    active = active_task if isinstance(active_task, dict) else {}
    run_id = str(requested_run_id or active.get("run_id") or generated_run_id or time.strftime("run_%Y%m%d_%H%M%S")).strip()
    state = Path(output_root).expanduser() / _safe_task_name(run_id) / "job_state.json"
    return run_id, state.is_file(), state


def find_period_result(item: dict[str, str]) -> Path | None:
    """Resolve a period result even when an older GUI manifest omitted ``result``."""
    explicit = Path(str(item.get("result") or "")).expanduser()
    if explicit.is_file():
        return explicit.resolve()
    anchors = [item.get("directory"), item.get("final_centerlines"), item.get("final_surfaces")]
    for value in anchors:
        path = Path(str(value or "")).expanduser()
        start = path if path.is_dir() else path.parent
        for parent in (start, *start.parents):
            candidate = parent / "latest_result.json"
            if candidate.is_file():
                return candidate.resolve()
    return None


def find_saved_edit_directory(item: dict[str, str], preferred: str = "") -> tuple[Path | None, list[Path]]:
    """Find the editor output belonging to one period and report checked locations."""
    candidates: list[Path] = []
    for value in (preferred, item.get("edited_directory")):
        if str(value or "").strip():
            candidates.append(Path(str(value)).expanduser())
    review_value = str(item.get("directory") or "").strip()
    if review_value:
        review = Path(review_value).expanduser()
        candidates.append(review.parent / "centerline_edit")
    centerline_value = str(item.get("final_centerlines") or "").strip()
    if centerline_value:
        centerlines = Path(centerline_value).expanduser()
        candidates.append(centerlines.parent.parent / "centerline_edit")
    result = find_period_result(item)
    if result is not None:
        runs = result.parent / "runs"
        if runs.is_dir():
            candidates.extend(runs.glob("*/centerline_edit"))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    for directory in unique:
        if not directory.is_dir() or not (directory / "edited_manifest.json").is_file():
            continue
        if any(directory.glob("*_edited_graph.p")):
            return directory, unique
    return None, unique


def manifest_contains_period_result(manifest_path: Path | str | None, result_path: Path) -> bool:
    """Check that applying edits through a pipeline index will update the same task."""
    if not manifest_path:
        return False
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    target = os.path.normcase(str(result_path.resolve()))
    for entry in manifest.get("period_results", []) or []:
        try:
            current = os.path.normcase(str(Path(str(entry.get("result") or "")).expanduser().resolve()))
        except (OSError, TypeError, ValueError):
            continue
        if current == target:
            return True
    return False


def build_evaluate_existing_command(
    item: dict[str, str], pipeline_manifest: Path | str, truth: str,
    *, truth_type_field: str = "", validation_area: str = "",
    evaluation_tolerance: str = "5.0",
) -> list[str]:
    """Build a result-stage evaluation command for one existing change pair."""
    truth_path = Path(str(truth).strip()).expanduser()
    if not truth_path.is_file() or truth_path.suffix.lower() != USER_VECTOR_SUFFIX:
        raise ValueError("请选择存在的变化真值 SHP 文件。")
    required = ("grid", "before_period", "after_period")
    if any(not str(item.get(key) or "").strip() for key in required):
        raise ValueError("该变化结果缺少格网或期次信息，无法运行精度评价。")
    args = [
        "evaluate-existing", "--pipeline-manifest", str(pipeline_manifest),
        "--grid", str(item["grid"]),
        "--before-period", str(item["before_period"]),
        "--after-period", str(item["after_period"]),
        "--truth", str(truth_path),
        "--evaluation-tolerance", str(evaluation_tolerance),
    ]
    if str(validation_area).strip():
        args.extend(("--validation-area", str(validation_area).strip()))
    if str(truth_type_field).strip():
        args.extend(("--truth-type-field", str(truth_type_field).strip()))
    return args


def build_evaluate_all_command(
    manifest: dict, pipeline_manifest: Path | str,
    truths: list[tuple[str, str, str, str]], *, truth_type_field: str = "BHBM",
    evaluation_tolerance: str = "5.0",
) -> list[str]:
    """Build one result-stage command covering every area and adjacent pair."""
    truth_map = {
        (str(area), str(before), str(after)): str(path)
        for area, before, after, path in truths
    }
    changes = [entry for entry in manifest.get("change_results", []) or [] if isinstance(entry, dict)]
    args = [
        "evaluate-all-existing", "--pipeline-manifest", str(pipeline_manifest),
        "--truth-type-field", str(truth_type_field or "BHBM"),
        "--evaluation-tolerance", str(evaluation_tolerance),
    ]
    missing = []
    for entry in changes:
        key = (str(entry.get("grid")), str(entry.get("before_period")), str(entry.get("after_period")))
        source_value = truth_map.get(key) or str(entry.get("truth") or "")
        source = Path(source_value).expanduser()
        if not source.is_file() or source.suffix.lower() != USER_VECTOR_SUFFIX:
            missing.append(f"{key[0]} / {key[1]} → {key[2]}")
            continue
        args.extend(("--truth", *key, str(source)))
    if missing:
        raise ValueError("以下变化结果缺少真值 SHP：" + "、".join(missing))
    if not changes:
        raise ValueError("当前任务没有可评价的变化结果。")
    return args


class UserApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("道路实体变化智能检测与人工编辑")
        self.display_scale = configure_window_geometry(
            self.root, base_width=1280, base_height=820, min_width=1000, min_height=680,
        )
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.editor_process: subprocess.Popen[str] | None = None
        self.editor_ready_file: Path | None = None
        self.editor_started_monotonic: float | None = None
        self.editor_launch_state = "idle"
        self.editor_timeout_reported = False
        self.editor_stdout_lines: list[str] = []
        self.editor_stderr_lines: list[str] = []
        self.results_available = False
        self.review_items: list[dict[str, str]] = []
        self.temporal_items: list[dict[str, str]] = []
        self.loaded_manifest_path: Path | None = None
        self.period_rows: list[dict[str, object]] = []
        self.truth_rows: list[dict[str, object]] = []
        self.result_change_items: list[dict] = []
        self.project_validation_areas: list[tuple[str, str]] = []
        self.project_area_truths: list[tuple[str, str, str, str]] = []
        self.project_area_periods: dict[str, list[tuple[str, str]]] = {}
        self.project_data_sources: list[str] = []
        self.project_candidates: dict[str, list[str]] = {"shp": [], "txt": []}
        self.project_config: dict = {}
        self.project_root_path = ""
        self.project_region = StringVar(value="")
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
        self.log_visible = False
        self.current_step = 0
        self.preflight_passed = False
        self.step_pages: list[ttk.Frame] = []
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
        self.result_tree_paths: dict[str, Path] = {}
        self.vars = {
            key: StringVar(value=value)
            for key, value in {
                "mode": "validation",
                "validation_area": "",
                "truth_type_field": "",
                "evaluate": "0",
                "resume": "0",
                "continue_on_error": "1",
                "source_root": str(DEFAULT_TEST_DATA),
                "output_root": str(ROOT / "outputs"),
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

    def _style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self.root.configure(background=UI["page"])
        style.configure("TFrame", background=UI["card"])
        style.configure("Page.TFrame", background=UI["page"])
        style.configure("Header.TFrame", background=UI["header"])
        style.configure("Footer.TFrame", background=UI["card"])
        style.configure("TLabel", background=UI["card"], foreground=UI["ink"], font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background=UI["card"], font=("Microsoft YaHei UI", 18, "bold"), foreground=UI["ink"])
        style.configure("Subtitle.TLabel", background=UI["card"], font=("Microsoft YaHei UI", 9), foreground=UI["muted"])
        style.configure("Brand.TLabel", background=UI["header"], foreground="#79C3AD", font=("Segoe UI", 9, "bold"))
        style.configure("HeaderTitle.TLabel", background=UI["header"], foreground=UI["header_text"], font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("HeaderMeta.TLabel", background=UI["header"], foreground=UI["header_muted"], font=("Microsoft YaHei UI", 9))
        style.configure("HeaderProject.TLabel", background=UI["header"], foreground=UI["header_text"], font=("Microsoft YaHei UI", 9))
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
        style.configure("Secondary.TButton", **CONTROL_METRICS["regular"], foreground=UI["blue"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("Secondary.TButton", background=[("active", UI["blue_soft"]), ("disabled", "#ECE9E1")], foreground=[("disabled", UI["subtle"])])
        style.configure("Compact.TButton", **CONTROL_METRICS["compact"], foreground=UI["ink"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("Compact.TButton", background=[("active", UI["blue_soft"]), ("disabled", "#ECE9E1")], foreground=[("active", UI["blue"]), ("disabled", UI["subtle"])])
        style.configure("Quiet.TButton", **CONTROL_METRICS["compact"], foreground=UI["muted"], background=UI["card"], borderwidth=0, focusthickness=0)
        style.map("Quiet.TButton", foreground=[("active", UI["blue"])], background=[("active", UI["blue_soft"])])
        style.configure("Danger.TButton", **CONTROL_METRICS["compact"], foreground="#B42318", background="#FFF1F2", borderwidth=0, focusthickness=0)
        style.map("Danger.TButton", background=[("active", "#FFE4E6"), ("disabled", "#F8FAFC")], foreground=[("disabled", UI["subtle"])])
        style.configure("ResultPrimary.TButton", **CONTROL_METRICS["regular"], foreground="#FFFFFF", background=UI["blue"], borderwidth=1, bordercolor=UI["blue"], focusthickness=0)
        style.map("ResultPrimary.TButton", background=[("active", UI["blue_hover"]), ("pressed", UI["header_deep"]), ("disabled", "#B6B8B1")])
        style.configure("ResultSecondary.TButton", **CONTROL_METRICS["regular"], foreground=UI["blue"], background=UI["card"], borderwidth=1, bordercolor=UI["line_strong"], focusthickness=0)
        style.map("ResultSecondary.TButton", background=[("active", UI["blue_soft"]), ("disabled", "#ECE9E1")], foreground=[("disabled", UI["subtle"])])
        style.configure("Section.TLabel", font=("Microsoft YaHei UI", 15, "bold"), foreground=UI["ink"], background=UI["page"])
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", 12, "bold"), foreground=UI["ink"], background=UI["card"])
        style.configure("CardTitleSelected.TLabel", font=("Microsoft YaHei UI", 12, "bold"), foreground=UI["blue"], background=UI["blue_soft"])
        style.configure("Metric.TLabel", font=("Microsoft YaHei UI", 18, "bold"), foreground=UI["ink"], background=UI["card"])
        style.configure("Muted.TLabel", foreground=UI["muted"], background=UI["page"], font=("Microsoft YaHei UI", 9))
        style.configure("CardMuted.TLabel", foreground=UI["muted"], background=UI["card"], font=("Microsoft YaHei UI", 9))
        style.configure("CardMutedSelected.TLabel", foreground=UI["muted"], background=UI["blue_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("Eyebrow.TLabel", foreground=UI["blue"], background=UI["card"], font=("Microsoft YaHei UI", 8, "bold"))
        style.configure("PathTitle.TLabel", foreground=UI["ink"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("PathText.TLabel", foreground=UI["muted"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("MetricName.TLabel", foreground=UI["muted"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("MetricValue.TLabel", foreground=UI["ink"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("FormLabel.TLabel", foreground=UI["ink"], background=UI["card"], font=("Microsoft YaHei UI", 10))
        style.configure("SelectedMark.TLabel", foreground=UI["blue"], background=UI["blue_soft"], font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("IdleMark.TLabel", foreground=UI["subtle"], background=UI["card"], font=("Microsoft YaHei UI", 15))
        style.configure("SuccessNote.TLabel", foreground=UI["green"], background=UI["blue_soft"], font=("Microsoft YaHei UI", 9))
        style.configure("WarningNote.TLabel", foreground=UI["amber"], background=UI["card"], font=("Microsoft YaHei UI", 9))
        style.configure("TNotebook.Tab", font=("Microsoft YaHei UI", 10, "bold"), padding=(18, 8))
        style.configure("Hint.TLabel", foreground=UI["muted"], background=UI["card"], wraplength=920)
        style.configure("Success.TLabel", foreground=UI["green"], background=UI["card"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("HeaderReady.TLabel", foreground="#8ED3B8", background=UI["header"], font=("Microsoft YaHei UI", 9, "bold"), padding=(8, 4))
        style.configure("HeaderIdle.TLabel", foreground=UI["header_muted"], background=UI["header"], font=("Microsoft YaHei UI", 9), padding=(8, 4))
        style.configure("FooterStatus.TLabel", background=UI["card"], foreground=UI["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("TEntry", padding=(9, 6), fieldbackground=UI["card"], bordercolor=UI["line_strong"], lightcolor=UI["line_strong"], darkcolor=UI["line_strong"])
        style.configure("TCombobox", padding=(9, 6), fieldbackground=UI["card"], bordercolor=UI["line_strong"])
        style.map("TCombobox", fieldbackground=[("readonly", UI["card"])], background=[("readonly", UI["card"])], foreground=[("readonly", UI["ink"])])
        style.configure("TCheckbutton", background=UI["card"], foreground=UI["ink"])
        style.configure("TRadiobutton", background=UI["card"], foreground=UI["ink"])
        style.configure("Modern.Horizontal.TProgressbar", background=UI["blue"], troughcolor="#DEDAD0", borderwidth=0, thickness=10)
        style.configure(
            "Data.Treeview", background=UI["card"], fieldbackground=UI["card"],
            foreground=UI["ink"], rowheight=30, borderwidth=0,
            font=("Microsoft YaHei UI", 9),
        )
        style.map(
            "Data.Treeview", background=[("selected", UI["blue"])],
            foreground=[("selected", "#FFFFFF")],
        )
        style.configure(
            "Data.Treeview.Heading", background="#E3DFD5", foreground=UI["ink"],
            relief="flat", padding=(8, 8), font=("Microsoft YaHei UI", 9, "bold"),
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
        project_menu.add_command(label="载入已有任务结果", command=self.load_existing_results)
        project_menu.add_separator()
        project_menu.add_command(label="打开项目文件夹", command=self.open_project_folder)
        menu.add_cascade(label="项目", menu=project_menu)
        tools_menu = Menu(menu, tearoff=False)
        tools_menu.add_command(label="运行诊断 / 导出诊断包", command=self.export_diagnostics)
        menu.add_cascade(label="工具", menu=tools_menu)
        self.root.configure(menu=menu)

        header = ttk.Frame(self.root, padding=(22, 14, 22, 14), style="Header.TFrame")
        header.pack(fill=X)
        ttk.Label(header, text="ROAD CHANGE", style="Brand.TLabel").pack(side=LEFT, padx=(0, 22))
        title_area = ttk.Frame(header, style="Header.TFrame")
        title_area.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(title_area, text="道路实体变化智能检测与人工编辑", style="HeaderTitle.TLabel").pack(anchor="w")
        project_area = ttk.Frame(header, style="Header.TFrame")
        project_area.pack(side=RIGHT)
        ttk.Label(project_area, textvariable=self.current_project, style="HeaderProject.TLabel").pack(side=LEFT, padx=(0, 20))
        self.header_state_label = ttk.Label(project_area, textvariable=self.header_state, style="HeaderIdle.TLabel")
        self.header_state_label.pack(side=LEFT, padx=(0, 20))
        ttk.Label(project_area, text="道路提取  ·  变化检测  ·  人工编辑  ·  成果交付", style="HeaderMeta.TLabel").pack(side=LEFT)

        self.stepper_canvas = Canvas(
            self.root, height=56, background=UI["page"], highlightthickness=0, borderwidth=0,
        )
        self.stepper_canvas.pack(fill=X)
        self.stepper_canvas.bind("<Configure>", lambda _event: self._draw_stepper())
        self.stepper_canvas.bind("<Button-1>", self._on_stepper_click)

        self.content_shell = ttk.Frame(self.root, style="Page.TFrame")
        self.content_shell.pack(fill=BOTH, expand=True)
        self.content_canvas = Canvas(
            self.content_shell, background=UI["page"], highlightthickness=0, borderwidth=0,
        )
        self.content_scrollbar = ttk.Scrollbar(
            self.content_shell, orient="vertical", command=self.content_canvas.yview,
        )
        self.content_scrollbar.pack(side=RIGHT, fill="y")
        self.content_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.content_canvas.configure(yscrollcommand=self.content_scrollbar.set)
        self.page_host = ttk.Frame(self.content_canvas, padding=LAYOUT_METRICS["page_padding"], style="Page.TFrame")
        self.page_window = self.content_canvas.create_window((0, 0), window=self.page_host, anchor="nw")
        self.max_page_width = round(1320 * self.display_scale)
        self.content_canvas.bind("<Configure>", self._resize_content_canvas)
        self.page_host.bind("<Configure>", lambda _event: self._sync_content_scrollregion())
        self.root.bind_all("<MouseWheel>", self._on_content_mousewheel, add="+")
        for _ in WORKFLOW_STEPS:
            page = ttk.Frame(self.page_host, style="Page.TFrame")
            self.step_pages.append(page)

        self._build_data_page(self.step_pages[0])
        self._build_run_page(self.step_pages[1])
        self._build_review_page(self.step_pages[2])
        self._build_result_page(self.step_pages[3])
        self._build_shared_log_panel()

        ttk.Separator(self.root).pack(fill=X)
        footer = ttk.Frame(self.root, padding=(18, 10, 18, 11), style="Footer.TFrame")
        footer.pack(fill=X)
        ttk.Label(footer, text="●", foreground=UI["blue"], background=UI["card"]).pack(side=LEFT, padx=(0, 8))
        ttk.Label(footer, textvariable=self.status, style="FooterStatus.TLabel", wraplength=760).pack(side=LEFT, fill=X, expand=True)
        self.footer_next = ttk.Button(footer, style="Primary.TButton", command=self._go_next)
        self.footer_next.pack(side=RIGHT)
        self.footer_back = ttk.Button(footer, text="上一步", style="Secondary.TButton", command=self._go_back)
        self.footer_back.pack(side=RIGHT, padx=(0, 10))

        self._show_step(0, force=True)
        self._refresh_input_summary()
        self._refresh_result_availability()

    def _build_shared_log_panel(self) -> None:
        """Build the window-level log panel shared by every workflow step."""
        self.shared_log_shell = ttk.Frame(
            self.root, style="Card.TFrame", padding=(18, 7, 18, 7),
        )
        self.shared_log_shell.pack(fill=X)
        log_header = ttk.Frame(self.shared_log_shell, style="Card.TFrame")
        log_header.pack(fill=X)
        ttk.Label(log_header, text="全流程日志", style="CardTitle.TLabel").pack(side=LEFT)
        self.shared_log_status = StringVar(value="日志尚未开始")
        ttk.Label(
            log_header, textvariable=self.shared_log_status, style="CardMuted.TLabel",
            wraplength=720,
        ).pack(side=LEFT, fill=X, expand=True, padx=(16, 12))
        self.log_toggle = ttk.Button(
            log_header, text="展开日志", style="Quiet.TButton", command=self._toggle_log,
        )
        self.log_toggle.pack(side=RIGHT)
        ttk.Button(
            log_header, text="复制全部", style="Compact.TButton", command=self.copy_all_logs,
        ).pack(side=RIGHT, padx=(0, 7))
        ttk.Button(
            log_header, text="打开日志文件", style="Compact.TButton", command=self.open_active_log,
        ).pack(side=RIGHT, padx=(0, 7))

        self.log_frame = ttk.Frame(self.shared_log_shell, style="Card.TFrame")
        log_body = ttk.Frame(self.log_frame, style="Card.TFrame")
        log_body.pack(fill=BOTH, expand=True)
        self.log = Text(
            log_body, height=9, wrap="none", undo=False, exportselection=True,
            font=("Consolas", 10), foreground=UI["ink"], background="#FFFFFF",
            selectbackground=UI["blue"], selectforeground="#FFFFFF", padx=8, pady=7,
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
        height = max(48, canvas.winfo_height())
        margin_x = 18
        top = 9
        bottom = height - 2
        tab_width = (width - margin_x * 2) / len(WORKFLOW_STEPS)
        centers = [margin_x + tab_width * (index + 0.5) for index in range(len(WORKFLOW_STEPS))]
        completed_to = getattr(self, "completed_to", 0)
        has_results = self.results_available
        for index, (center, label) in enumerate(zip(centers, WORKFLOW_STEPS)):
            left = margin_x + tab_width * index
            right = left + tab_width
            if index == self.current_step:
                fill, outline = UI["card"], "#383A36"
                label_color, weight = UI["blue"], "bold"
            elif index <= completed_to:
                fill, outline = UI["green_soft"], UI["line_strong"]
                label_color, weight = UI["green"], "bold"
            else:
                fill, outline = "#E8E4DA", UI["line_strong"]
                label_color, weight = UI["muted"], "normal"
            canvas.create_rectangle(left, top, right, bottom, fill=fill, outline=outline, width=2 if index == self.current_step else 1)
            marker = "✓" if index <= completed_to and index != self.current_step else str(index + 1)
            canvas.create_text(
                center, (top + bottom) / 2, text=f"{marker}  {label}", fill=label_color,
                font=("Microsoft YaHei UI", 10, weight),
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
            card["mark"].configure(text="●" if selected else "○", style="SelectedMark.TLabel" if selected else "IdleMark.TLabel")
            card["title"].configure(style="CardTitleSelected.TLabel" if selected else "CardTitle.TLabel")
            card["description"].configure(style="CardMutedSelected.TLabel" if selected else "CardMuted.TLabel")
            if value == "0":
                card["note"].configure(style="SuccessNote.TLabel" if selected else "Success.TLabel")
            else:
                card["note"].configure(style="CardMutedSelected.TLabel" if selected else "WarningNote.TLabel")

    def _build_data_page(self, page: ttk.Frame) -> None:
        self.data_body = page
        heading = ttk.Frame(page, style="Page.TFrame")
        heading.pack(fill=X, pady=(0, 13))
        ttk.Label(heading, text="项目与数据管理", style="Section.TLabel").pack(anchor="w")
        ttk.Label(heading, text="按项目统一管理多个验证区、各期影像和变化真值。", style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

        project_card = ttk.Frame(page, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        project_card.pack(fill=X)
        project_head = ttk.Frame(project_card)
        project_head.pack(fill=X)
        ttk.Label(project_head, text="▣  项目与数据管理", style="CardTitle.TLabel").pack(side=LEFT)
        ttk.Button(project_head, text="＋ 新建项目", style="Primary.TButton", command=self.create_project_folder).pack(side=RIGHT)
        ttk.Button(project_head, text="打开项目", style="Secondary.TButton", command=self.import_project_folder).pack(side=RIGHT, padx=(0, 8))
        ttk.Button(project_head, text="打开项目文件夹", style="Secondary.TButton", command=self.open_project_folder).pack(side=RIGHT, padx=(0, 8))
        project_meta = ttk.Frame(project_card, style="Soft.TFrame", padding=(14, 10))
        project_meta.pack(fill=X, pady=(12, 0))
        self.project_name_display = StringVar(value="尚未打开项目")
        ttk.Label(project_meta, text="项目名称", width=14, style="PathTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(project_meta, textvariable=self.project_name_display, style="PathText.TLabel").grid(row=0, column=1, sticky="ew")
        ttk.Label(project_meta, text="项目路径", width=14, style="PathTitle.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.project_path_display = StringVar(value="尚未选择项目目录")
        ttk.Label(project_meta, textvariable=self.project_path_display, style="PathText.TLabel", width=1).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        project_meta.grid_columnconfigure(1, weight=1)

        quick = ttk.Frame(page, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        quick.pack(fill=X, pady=(LAYOUT_METRICS["section_gap"], 0))
        ttk.Label(quick, text="▤  外部原始数据源", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(quick, text="只保存外部路径和映射；不会移动、复制或重命名原始数据。", style="CardMuted.TLabel").pack(anchor="w", pady=(4, LAYOUT_METRICS["module_gap"]))
        quick_actions = ttk.Frame(quick, style="Soft.TFrame", padding=(14, 12))
        quick_actions.pack(fill=X)
        ttk.Label(quick_actions, text="01", foreground=UI["blue"], background=UI["blue_soft"], font=("Microsoft YaHei UI", 9, "bold"), padding=(9, 7)).grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 12))
        path_area = ttk.Frame(quick_actions, style="Soft.TFrame")
        path_area.grid(row=0, column=1, rowspan=2, sticky="ew")
        quick_actions.grid_columnconfigure(1, weight=1)
        ttk.Label(path_area, text="已连接数据源", style="PathTitle.TLabel").pack(anchor="w")
        ttk.Label(path_area, textvariable=self.data_source_display, style="PathText.TLabel", width=1).pack(anchor="w", fill=X, pady=(3, 0))
        ttk.Button(
            quick_actions, text="连接数据源", style="Primary.TButton",
            command=self.connect_data_source,
        ).grid(row=0, column=3, rowspan=2, sticky="e")
        ttk.Button(
            quick_actions, text="扫描数据", style="Secondary.TButton",
            command=self.scan_data_sources,
        ).grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 8))
        self.input_summary = StringVar(value="请选择项目目录；如需手工指定数据，可展开高级设置。")
        summary_row = ttk.Frame(quick)
        summary_row.pack(fill=X, pady=(13, 0))
        self.input_summary_dot = ttk.Label(summary_row, text="●", foreground=UI["subtle"], background=UI["card"])
        self.input_summary_dot.pack(side=LEFT, padx=(0, 7))
        self.input_summary_label = ttk.Label(summary_row, textvariable=self.input_summary, style="CardMuted.TLabel", wraplength=1050)
        self.input_summary_label.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(quick, textvariable=self.data_status, style="Success.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(quick, textvariable=self.project_scan_summary, style="CardMuted.TLabel", wraplength=1040).pack(anchor="w", fill=X, pady=(3, 0))

        config_card = ttk.Frame(page, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        config_card.pack(fill=X, pady=(LAYOUT_METRICS["section_gap"], 0))
        ttk.Label(config_card, text="⚙  数据配置", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(config_card, text="按区域查看和补充验证区、多期影像与相邻期变化真值。", style="CardMuted.TLabel").pack(anchor="w", pady=(4, 10))
        region_row = ttk.Frame(config_card)
        region_row.pack(fill=X, pady=LAYOUT_METRICS["form_gap"])
        ttk.Label(region_row, text="区域", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.project_region_combo = ttk.Combobox(region_row, textvariable=self.project_region, state="readonly", width=35)
        self.project_region_combo.pack(side=LEFT, fill=X, expand=True)
        self.project_region_combo.bind("<<ComboboxSelected>>", self._project_region_changed)
        ttk.Button(region_row, text="＋ 添加区域", style="Compact.TButton", command=self.add_project_region).pack(side=LEFT, padx=(8, 0))
        area_row = ttk.Frame(config_card)
        area_row.pack(fill=X, pady=LAYOUT_METRICS["form_gap"])
        ttk.Label(area_row, text="验证区", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        ttk.Label(area_row, textvariable=self.project_validation_path, style="CardMuted.TLabel", anchor="w").pack(side=LEFT, fill=X, expand=True)
        ttk.Button(area_row, text="选择…", style="Compact.TButton", command=self.replace_project_validation_area).pack(side=LEFT, padx=(8, 0))
        ttk.Button(area_row, text="移除区域", style="Compact.TButton", command=self.remove_project_region).pack(side=LEFT, padx=(4, 0))
        self.project_config_container = ttk.Frame(config_card, style="Soft.TFrame", padding=(14, 10))
        self.project_config_container.pack(fill=X, pady=(8, 8))
        config_actions = ttk.Frame(config_card)
        config_actions.pack(fill=X)
        self.add_project_period_button = ttk.Button(config_actions, text="＋ 添加期次", style="Compact.TButton", command=self.add_project_period)
        self.add_project_period_button.pack(side=LEFT)
        ttk.Button(config_actions, text="检查数据", style="Primary.TButton", command=self.preflight_inputs).pack(side=RIGHT)
        self._refresh_project_config_panel()

        self.manual_toggle = ttk.Button(
            page, text="›  高级设置  ·  旧版单验证区、参数配置与兼容模式", style="Quiet.TButton", command=self._toggle_manual_inputs,
        )
        self.manual_toggle.pack(anchor="w", pady=(12, 3))
        self.manual_frame = ttk.Frame(page, style="Card.TFrame", padding=(18, 15))
        ttk.Label(self.manual_frame, text="高级设置", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))
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
        ttk.Button(period_actions, text="＋ 添加期次", style="Compact.TButton", command=self._add_period_row).pack(side=LEFT)
        self.truth_container = ttk.Frame(self.manual_frame)
        self._add_period_row("2021")
        self._add_period_row("2022")
        self.grid_toggle = ttk.Button(
            self.manual_frame, text="▶ 兼容旧版多格网目录", command=self._toggle_grid_options,
        )
        self.grid_toggle.pack(anchor="w", pady=(9, 2))
        self.grid_options = ttk.Frame(self.manual_frame)
        ttk.Radiobutton(
            self.grid_options, text="使用多格网目录", variable=self.vars["mode"], value="grid",
            command=self._refresh_input_summary,
        ).pack(anchor="w")
        self._field(self.grid_options, "格网数据根目录", "source_root", "dir")
        config_actions = ttk.Frame(self.manual_frame, style="Card.TFrame")
        config_actions.pack(fill=X, pady=(12, 0))
        ttk.Button(config_actions, text="加载配置…", style="Compact.TButton", command=self.load_task_config).pack(side=LEFT)
        ttk.Button(config_actions, text="导出配置…", style="Compact.TButton", command=self.save_task_config).pack(side=LEFT, padx=8)

    def _build_run_page(self, page: ttk.Frame) -> None:
        self.run_body = page
        ttk.Label(page, text="自动处理", style="Section.TLabel").pack(anchor="w")
        ttk.Label(page, text="系统将在开始处理前检查项目数据和运行环境。", style="Muted.TLabel").pack(anchor="w", pady=(4, LAYOUT_METRICS["section_gap"]))
        content = ttk.Frame(page, style="Page.TFrame")
        content.pack(fill=BOTH, expand=True)
        run_card = ttk.Frame(content, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        run_card.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))
        ttk.Label(run_card, text="开始任务", style="CardTitle.TLabel").pack(anchor="w")
        self.preflight_summary = StringVar(value="开始处理时会自动检查以下内容；发现问题将立即停止并说明原因。")
        ttk.Label(run_card, textvariable=self.preflight_summary, style="Hint.TLabel", wraplength=480).pack(anchor="w", fill=X, pady=(5, 13))
        checklist = ttk.Frame(run_card, style="Soft.TFrame", padding=(14, 8))
        checklist.pack(fill=X, pady=(0, 14))
        for label in ("项目结构与验证区", "影像期次与覆盖范围", "模型、参数与运行环境", "输出位置与磁盘空间"):
            row = ttk.Frame(checklist, style="Soft.TFrame")
            row.pack(fill=X, pady=4)
            ttk.Label(row, text="○", foreground=UI["subtle"], background=UI["slate_soft"]).pack(side=LEFT, padx=(0, 9))
            ttk.Label(row, text=label, foreground=UI["ink"], background=UI["slate_soft"]).pack(side=LEFT)
            ttk.Label(row, text="开始前检查", foreground=UI["subtle"], background=UI["slate_soft"], font=("Microsoft YaHei UI", 8)).pack(side=RIGHT)
        actions = ttk.Frame(run_card)
        actions.pack(fill=X, pady=(2, 12))
        self.preflight_button = ttk.Button(actions, text="检查数据", command=self.preflight_inputs)
        self.run_button = ttk.Button(
            actions, text="运行完整流程", style="Hero.TButton", command=self.run_all,
        )
        self.run_button.pack(side=RIGHT)
        self.cancel_button = ttk.Button(actions, text="取消任务", style="Danger.TButton", command=self.cancel_task)
        self.cancel_button.pack(side=RIGHT, padx=10)
        self.cancel_button.state(["disabled"])

        progress_card = ttk.Frame(run_card, style="Soft.TFrame", padding=(14, 11))
        progress_card.pack(fill=X, pady=(8, 0))
        ttk.Label(progress_card, text="正在处理", style="PathTitle.TLabel").pack(anchor="w", pady=(0, 8))
        self.run_status = StringVar(value="等待开始任务。")
        ttk.Label(progress_card, textvariable=self.run_status, style="PathText.TLabel", wraplength=470).pack(fill=X, pady=(0, 10))
        ttk.Label(progress_card, text="总体进度", style="PathTitle.TLabel").pack(anchor="w", pady=(0, 6))
        self.progress = ttk.Progressbar(progress_card, mode="determinate", maximum=1, value=0, style="Modern.Horizontal.TProgressbar")
        self.progress.pack(fill=X)
        self.progress_text = StringVar(value="0 / 0 · 已用时 00:00:00 · 剩余 --")
        ttk.Label(progress_card, textvariable=self.progress_text, style="PathText.TLabel").pack(anchor="w", pady=(8, 0))

        summary_card = ttk.Frame(content, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        summary_card.pack(side=LEFT, fill=BOTH, expand=True, padx=(8, 0))
        ttk.Label(summary_card, text="任务摘要", style="CardTitle.TLabel").pack(anchor="w")
        self.run_destination_summary = StringVar(value=f"成果保存到：{self.vars['output_root'].get()}")
        summary_body = ttk.Frame(summary_card, style="Soft.TFrame", padding=(14, 10))
        summary_body.pack(fill=X, pady=(12, 12))
        ttk.Label(summary_body, text="输入数据", style="PathTitle.TLabel").pack(anchor="w")
        ttk.Label(summary_body, textvariable=self.input_summary, style="PathText.TLabel", wraplength=480).pack(anchor="w", fill=X, pady=(3, 10))
        ttk.Label(summary_body, text="成果位置", style="PathTitle.TLabel").pack(anchor="w")
        ttk.Label(summary_body, textvariable=self.run_destination_summary, style="PathText.TLabel", wraplength=480).pack(anchor="w", fill=X, pady=(3, 0))
        self.run_settings_toggle = ttk.Button(
            summary_card, text="›  输出位置与高级设置", style="Quiet.TButton", command=self._toggle_run_settings,
        )
        self.run_settings_toggle.pack(anchor="w")
        self.run_settings_frame = ttk.Frame(summary_card, style="Card.TFrame")
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
            self.run_settings_frame, text="▶ 高级参数（通常不需要修改）", command=self._toggle_advanced,
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

        stage_card = ttk.Frame(page, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        stage_card.pack(fill=X, pady=(LAYOUT_METRICS["section_gap"], 0))
        ttk.Label(stage_card, text="局部重跑", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            stage_card,
            text="重跑会主动废弃所选阶段的旧结果；选择更新相关结果时，系统按依赖顺序串行更新变化、评价、长时序成果和报告。",
            style="Hint.TLabel", wraplength=1040,
        ).pack(anchor="w", pady=(4, 10))
        selectors = ttk.Frame(stage_card, style="Soft.TFrame", padding=(14, 10))
        selectors.pack(fill=X, pady=(0, 10))
        ttk.Label(selectors, text="区域", style="PathTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.stage_region_combo = ttk.Combobox(selectors, textvariable=self.project_region, state="readonly", width=24)
        self.stage_region_combo.grid(row=0, column=1, sticky="ew", padx=(8, 18))
        self.stage_region_combo.bind("<<ComboboxSelected>>", self._project_region_changed)
        ttk.Label(selectors, text="期次", style="PathTitle.TLabel").grid(row=0, column=2, sticky="w")
        self.stage_period_combo = ttk.Combobox(selectors, textvariable=self.project_period, state="readonly", width=22)
        self.stage_period_combo.grid(row=0, column=3, sticky="ew", padx=(8, 18))
        self.stage_period_combo.bind("<<ComboboxSelected>>", self._stage_period_changed)
        ttk.Label(selectors, text="相邻变化对", style="PathTitle.TLabel").grid(row=0, column=4, sticky="w")
        self.stage_pair_combo = ttk.Combobox(selectors, textvariable=self.project_change_pair, state="readonly", width=25)
        self.stage_pair_combo.grid(row=0, column=5, sticky="ew", padx=(8, 0))
        for column in (1, 3, 5):
            selectors.grid_columnconfigure(column, weight=1)
        self.affected_pairs_summary = StringVar(value="请选择期次以查看受影响的相邻变化对。")
        ttk.Label(stage_card, textvariable=self.affected_pairs_summary, style="WarningNote.TLabel", wraplength=1040).pack(anchor="w", fill=X, pady=(0, 10))
        stage_actions = ttk.Frame(stage_card)
        stage_actions.pack(fill=X)
        self.stage_buttons = []
        for text, command, style in (
            ("重跑所选期次", lambda: self.rerun_selected_period(False), "Secondary.TButton"),
            ("重跑并更新相关结果", lambda: self.rerun_selected_period(True), "Primary.TButton"),
            ("重跑所选变化对", lambda: self.rerun_selected_change(False), "Secondary.TButton"),
            ("重跑并更新长时序成果", lambda: self.rerun_selected_change(True), "Secondary.TButton"),
        ):
            button = ttk.Button(stage_actions, text=text, style=style, command=command)
            button.pack(side=LEFT, padx=(0, 8))
            self.stage_buttons.append(button)
        advanced_stage = ttk.Frame(stage_card)
        advanced_stage.pack(fill=X, pady=(10, 0))
        ttk.Label(advanced_stage, text="高级操作", style="CardMuted.TLabel").pack(side=LEFT)
        batch_button = ttk.Button(advanced_stage, text="批量重跑全部道路提取", style="Compact.TButton", command=self.run_extract_all)
        batch_button.pack(side=LEFT, padx=(10, 0))
        self.stage_buttons.append(batch_button)
        self._refresh_stage_selectors()

    def _build_review_page(self, page: ttk.Frame) -> None:
        self.review_body = page
        ttk.Label(page, text="人工编辑（可选）", style="Section.TLabel").pack(anchor="w")
        ttk.Label(page, text="如需提高成果质量，可修正中心线并重新生成受影响的结果。", style="Muted.TLabel").pack(anchor="w", pady=(4, LAYOUT_METRICS["section_gap"]))
        review_card = ttk.Frame(page, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        review_card.pack(fill=X)
        ttk.Label(review_card, text="编辑与自动更新", style="CardTitle.TLabel").pack(anchor="w")
        self.review_status = StringVar(value="完成自动处理后，可在此选择需要编辑的期次。")
        ttk.Label(review_card, textvariable=self.review_status, style="Hint.TLabel", wraplength=1040).pack(anchor="w", fill=X, pady=(4, 0))
        selector = ttk.Frame(review_card, style="Soft.TFrame", padding=(14, 11))
        selector.pack(fill=X, pady=(16, 10))
        ttk.Label(selector, text="项目 / 格网 / 期次", style="PathTitle.TLabel").pack(side=LEFT)
        self.review_selection = StringVar()
        self.review_combo = ttk.Combobox(selector, textvariable=self.review_selection, state="readonly", width=58)
        self.review_combo.pack(side=LEFT, fill=X, expand=True, padx=(16, 0))
        self.review_combo.bind("<<ComboboxSelected>>", self._review_selection_changed)
        self.review_detail = StringVar(value="暂无可复核数据。")
        ttk.Label(review_card, textvariable=self.review_detail, style="Hint.TLabel", wraplength=1040).pack(anchor="w", fill=X, pady=(0, 14))
        edit_row = ttk.Frame(review_card, style="Soft.TFrame", padding=(14, 9))
        edit_row.pack(fill=X, pady=(0, 12))
        ttk.Label(edit_row, text="项目编辑目录", style="PathTitle.TLabel").pack(side=LEFT)
        ttk.Label(edit_row, textvariable=self.review_edit_directory, style="PathText.TLabel", anchor="w").pack(side=LEFT, fill=X, expand=True, padx=(12, 8))
        actions = ttk.Frame(review_card)
        actions.pack(fill=X)
        self.launch_review_button = ttk.Button(actions, text="打开编辑工作台", style="Hero.TButton", command=self.launch_selected_review_editor)
        self.launch_review_button.pack(side=LEFT)
        self.apply_review_button = ttk.Button(actions, text="应用编辑并更新相关结果", style="Primary.TButton", command=self.apply_selected_review)
        self.apply_review_button.pack(side=LEFT, padx=10)
        self.review_advanced_toggle = ttk.Button(review_card, text="›  高级操作", style="Quiet.TButton", command=self._toggle_review_advanced)
        self.review_advanced_toggle.pack(anchor="w", pady=(10, 0))
        self.review_advanced_frame = ttk.Frame(review_card)
        ttk.Button(self.review_advanced_frame, text="导入外部编辑成果", style="Compact.TButton", command=self.select_review_edit_directory).pack(side=LEFT)
        ttk.Button(self.review_advanced_frame, text="打开编辑资料目录", style="Compact.TButton", command=self.open_selected_review_folder).pack(side=LEFT, padx=(8, 0))

        self.review_task_frame = ttk.Frame(page, style="Card.TFrame", padding=(18, 15))
        self.review_task_frame.pack(fill=BOTH, expand=True, pady=(12, 0))
        review_task_header = ttk.Frame(self.review_task_frame)
        review_task_header.pack(fill=X)
        ttk.Label(review_task_header, text="编辑后增量重建", style="CardTitle.TLabel").pack(side=LEFT)
        self.review_cancel_button = ttk.Button(
            review_task_header, text="停止重建", style="Danger.TButton", command=self.cancel_task,
        )
        self.review_cancel_button.pack(side=RIGHT)
        self.review_cancel_button.state(["disabled"])
        ttk.Label(
            self.review_task_frame, textvariable=self.run_status, style="Hint.TLabel", wraplength=1040,
        ).pack(anchor="w", fill=X, pady=(7, 8))
        self.review_progress = ttk.Progressbar(
            self.review_task_frame, mode="determinate", maximum=1, value=0,
            style="Modern.Horizontal.TProgressbar",
        )
        self.review_progress.pack(fill=X)
        ttk.Label(
            self.review_task_frame, textvariable=self.progress_text, style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(6, 8))
        ttk.Label(
            self.review_task_frame, text="详细输出统一显示在窗口底部的“全流程日志”。",
            style="CardMuted.TLabel",
        ).pack(anchor="w")
        ttk.Label(page, text="人工编辑不是必需步骤；跳过时将直接采用自动处理结果。", style="Muted.TLabel").pack(anchor="w", pady=(12, 0))

    def _build_result_page(self, page: ttk.Frame) -> None:
        self.result_body = page
        ttk.Label(page, text="成果与评价", style="Section.TLabel").pack(anchor="w")
        ttk.Label(page, text="查看、评价并导出本次任务的正式成果。", style="Muted.TLabel").pack(anchor="w", pady=(4, LAYOUT_METRICS["section_gap"]))
        result_card = ttk.Frame(page, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        result_card.pack(fill=X)
        ttk.Label(result_card, text="处理结果", style="CardTitle.TLabel").pack(anchor="w")
        self.result_status = StringVar(value="完成任务或载入已有成果后，可在此查看处理结果。")
        ttk.Label(result_card, textvariable=self.result_status, style="Hint.TLabel", wraplength=980).pack(fill=X, pady=(4, 0))
        metrics = ttk.Frame(result_card, style="Card.TFrame")
        metrics.pack(fill=X, pady=(14, 8))
        self.result_period_count = StringVar(value="0 期")
        self.result_change_count = StringVar(value="0 组")
        self.result_area_count = StringVar(value="0 区")
        self.result_review_count = StringVar(value="0 处")
        for label, variable in (
            ("影像期次", self.result_period_count), ("变化检测任务", self.result_change_count),
            ("验证区", self.result_area_count), ("可人工编辑", self.result_review_count),
        ):
            box = ttk.Frame(metrics, style="Soft.TFrame", padding=(16, 12))
            box.pack(side=LEFT, fill=X, expand=True, padx=3)
            ttk.Label(box, text=label, style="MetricName.TLabel").pack()
            ttk.Label(box, textvariable=variable, style="MetricValue.TLabel").pack(pady=(4, 0))
        browser = ttk.Frame(result_card, style="Soft.TFrame", padding=(10, 8))
        browser.pack(fill=BOTH, expand=True, pady=(14, 7))
        self.result_tree = ttk.Treeview(browser, columns=("status",), show="tree headings", height=11, style="Data.Treeview")
        self.result_tree.heading("#0", text="成果")
        self.result_tree.heading("status", text="状态")
        self.result_tree.column("#0", width=620, minwidth=320, stretch=True)
        self.result_tree.column("status", width=140, minwidth=100, stretch=False)
        tree_scroll = ttk.Scrollbar(browser, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=tree_scroll.set)
        self.result_tree.pack(side=LEFT, fill=BOTH, expand=True)
        tree_scroll.pack(side=RIGHT, fill="y")
        self.result_tree.bind("<Double-1>", lambda _event: self.open_selected_result())
        result_actions = ttk.Frame(result_card)
        result_actions.pack(fill=X)
        ttk.Button(result_actions, text="打开所选成果", style="ResultPrimary.TButton", command=self.open_selected_result).pack(side=LEFT)
        ttk.Button(result_actions, text="打开所在目录", style="ResultSecondary.TButton", command=self.open_selected_result_folder).pack(side=LEFT, padx=(10, 0))
        ttk.Button(result_actions, text="查看长时序属性表", style="ResultSecondary.TButton", command=self.open_temporal_attribute_table).pack(side=LEFT, padx=(10, 0))
        ttk.Label(
            page,
            text="成果目录中包含道路中心线、道路面、道路宽度、变化检测结果、长时序道路成果、精度评价结果及任务报告。",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=12)

        evaluation_card = ttk.Frame(page, style="Card.TFrame", padding=LAYOUT_METRICS["card_padding"])
        evaluation_card.pack(fill=X, pady=(4, 12))
        ttk.Label(evaluation_card, text="精度评价", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            evaluation_card,
            text="根据真值数据中的变化类型，对新增、变化和灭失道路进行精度评价，并汇总各验证区及相邻影像期次的评价结果。中心线位置偏差仅统计新增和灭失道路；宽度变化道路不纳入中心线偏差统计。",
            style="Hint.TLabel", wraplength=980,
        ).pack(anchor="w", pady=(4, 12))
        pair_row = ttk.Frame(evaluation_card)
        pair_row.pack(fill=X, pady=LAYOUT_METRICS["form_gap"])
        ttk.Label(pair_row, text="待评价结果", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.evaluation_pair = StringVar(value="")
        self.evaluation_pair_combo = ttk.Combobox(pair_row, textvariable=self.evaluation_pair, state="readonly")
        self.evaluation_pair_combo.pack(side=LEFT, fill=X, expand=True)
        self.evaluation_pair_combo.bind("<<ComboboxSelected>>", self._evaluation_pair_changed)
        truth_row = ttk.Frame(evaluation_card)
        truth_row.pack(fill=X, pady=LAYOUT_METRICS["form_gap"])
        ttk.Label(truth_row, text="项目真值", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.evaluation_truth = StringVar(value="")
        ttk.Label(truth_row, textvariable=self.evaluation_truth, style="CardMuted.TLabel", anchor="w").pack(side=LEFT, fill=X, expand=True)
        ttk.Button(
            truth_row, text="补充真值", style="Compact.TButton",
            command=self.supplement_evaluation_truth,
        ).pack(side=LEFT, padx=(8, 0))
        self.evaluation_truth_summary = StringVar(value="总体评价会按区域和相邻期次分别匹配真值，不使用上方单个路径代替全部真值。")
        ttk.Label(
            evaluation_card, textvariable=self.evaluation_truth_summary, style="CardMuted.TLabel", wraplength=980,
        ).pack(anchor="w", fill=X, pady=(0, 5))
        self.evaluation_advanced_toggle = ttk.Button(evaluation_card, text="›  评价高级设置", style="Quiet.TButton", command=self._toggle_evaluation_advanced)
        self.evaluation_advanced_toggle.pack(anchor="w", pady=(4, 4))
        options_row = ttk.Frame(evaluation_card)
        self.evaluation_advanced_frame = options_row
        ttk.Label(options_row, text="变化类型字段", width=LAYOUT_METRICS["form_label_width"], style="FormLabel.TLabel").pack(side=LEFT)
        self.evaluation_type_field = StringVar(value="BHBM")
        ttk.Entry(options_row, textvariable=self.evaluation_type_field, width=18).pack(side=LEFT)
        ttk.Label(options_row, text="中心线匹配容差（米）").pack(side=LEFT, padx=(20, 8))
        self.evaluation_tolerance = StringVar(value="5.0")
        ttk.Entry(options_row, textvariable=self.evaluation_tolerance, width=9).pack(side=LEFT)
        action_row = ttk.Frame(evaluation_card)
        action_row.pack(fill=X)
        self.evaluation_status = StringVar(value="请先载入或生成包含变化检测的任务结果。")
        ttk.Label(action_row, textvariable=self.evaluation_status, style="Hint.TLabel", wraplength=760).pack(side=LEFT, fill=X, expand=True)
        self.run_evaluation_button = ttk.Button(
            action_row, text="评价当前结果", style="Primary.TButton", command=self.run_result_evaluation,
        )
        self.run_evaluation_button.pack(side=RIGHT)
        self.run_evaluation_button.state(["disabled"])
        self.run_total_evaluation_button = ttk.Button(
            action_row, text="评价全部结果", style="Hero.TButton", command=self.run_total_evaluation,
        )
        self.run_total_evaluation_button.pack(side=RIGHT, padx=(0, 8))
        self.run_total_evaluation_button.state(["disabled"])

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
            self.status.set("请先完成自动处理或载入已有成果，再进入成果与评价步骤。")
            return
        for page in self.step_pages:
            page.pack_forget()
        self.current_step = index
        self.step_pages[index].pack(fill=BOTH, expand=True)
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
        labels = (
            "下一步：自动处理  →",
            "下一步：人工编辑（可选）  →",
            "跳过人工编辑，查看成果  →",
            "已到最后一步",
        )
        self.footer_next.configure(text=labels[index])
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
            self.manual_toggle.configure(text="⌄  收起高级设置")
        else:
            self.manual_frame.pack_forget()
            self.manual_toggle.configure(text="›  高级设置  ·  旧版单验证区、参数配置与兼容模式")
        self._schedule_content_layout()

    def _show_manual_inputs(self) -> None:
        if not self.manual_inputs_visible:
            self._toggle_manual_inputs()

    def _toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill=X, pady=(2, 5), after=self.advanced_toggle)
            self.advanced_toggle.configure(text="▼ 高级参数")
        else:
            self.advanced_frame.pack_forget()
            self.advanced_toggle.configure(text="▶ 高级参数（通常不需要修改）")
        self._schedule_content_layout()

    def _toggle_review_advanced(self) -> None:
        self.review_advanced_visible = not self.review_advanced_visible
        if self.review_advanced_visible:
            self.review_advanced_frame.pack(fill=X, pady=(4, 0), after=self.review_advanced_toggle)
            self.review_advanced_toggle.configure(text="⌄  收起高级操作")
        else:
            self.review_advanced_frame.pack_forget()
            self.review_advanced_toggle.configure(text="›  高级操作")
        self._schedule_content_layout()

    def _toggle_evaluation_advanced(self) -> None:
        self.evaluation_advanced_visible = not self.evaluation_advanced_visible
        if self.evaluation_advanced_visible:
            self.evaluation_advanced_frame.pack(fill=X, pady=(2, 10), after=self.evaluation_advanced_toggle)
            self.evaluation_advanced_toggle.configure(text="⌄  收起评价高级设置")
        else:
            self.evaluation_advanced_frame.pack_forget()
            self.evaluation_advanced_toggle.configure(text="›  评价高级设置")
        self._schedule_content_layout()

    def _toggle_run_settings(self) -> None:
        self.run_settings_visible = not self.run_settings_visible
        if self.run_settings_visible:
            self.run_settings_frame.pack(fill=X, pady=(4, 2), after=self.run_settings_toggle)
            self.run_settings_toggle.configure(text="⌄  收起输出位置与高级设置")
        else:
            self.run_settings_frame.pack_forget()
            self.run_settings_toggle.configure(text="›  输出位置与高级设置")
        self._schedule_content_layout()

    def _toggle_log(self) -> None:
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log_frame.pack(fill=X, pady=(7, 0))
            self.log_toggle.configure(text="收起日志")
        else:
            self.log_frame.pack_forget()
            self.log_toggle.configure(text="展开日志")

    def _show_log(self) -> None:
        if not self.log_visible:
            self._toggle_log()

    def _refresh_input_summary(self) -> None:
        if not hasattr(self, "input_summary"):
            return
        if hasattr(self, "run_destination_summary"):
            self.run_destination_summary.set(
                f"成果保存到：{self.vars['output_root'].get().strip() or '尚未选择'}"
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
            if hasattr(self, "input_summary_label"):
                self.input_summary_label.configure(style="Success.TLabel" if ready else "CardMuted.TLabel")
                self.input_summary_dot.configure(foreground=UI["green"] if ready else UI["subtle"])
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
            if hasattr(self, "input_summary_label"):
                self.input_summary_label.configure(style="Success.TLabel" if ready else "CardMuted.TLabel")
                self.input_summary_dot.configure(foreground=UI["green"] if ready else UI["subtle"])
            self.root.after_idle(self._draw_stepper)
            return
        area_text = Path(area).name if area else "未选择验证区"
        self.input_summary.set(f"{area_text} · {len(periods)} 个影像期次 · 生产检测")
        ready = bool(area and len(periods) >= 2)
        if hasattr(self, "header_state"):
            self.header_state.set("项目数据已就绪" if ready else "等待选择数据")
            self.header_state_label.configure(style="HeaderReady.TLabel" if ready else "HeaderIdle.TLabel")
        if hasattr(self, "input_summary_label"):
            self.input_summary_label.configure(style="Success.TLabel" if ready else "CardMuted.TLabel")
            self.input_summary_dot.configure(foreground=UI["green"] if ready else UI["subtle"])
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

    def _selected_project_region(self) -> str:
        selected = self.project_region.get().strip()
        names = [name for name, _path in self.project_validation_areas]
        if selected in names:
            return selected
        return names[0] if names else ""

    def _refresh_project_config_panel(self) -> None:
        if not hasattr(self, "project_config_container"):
            return
        def render_candidates() -> None:
            pending = [(kind.upper(), path) for kind, paths in self.project_candidates.items() for path in paths]
            if not pending:
                return
            ttk.Label(self.project_config_container, text="待确认候选", style="PathTitle.TLabel").pack(anchor="w", pady=(10, 5))
            for kind, path in pending[:10]:
                row = ttk.Frame(self.project_config_container, style="Soft.TFrame")
                row.pack(fill=X, pady=1)
                ttk.Label(row, text=kind, width=6, style="PathText.TLabel").pack(side=LEFT)
                ttk.Label(row, text=path, style="PathText.TLabel", anchor="w").pack(side=LEFT, fill=X, expand=True)
            if len(pending) > 10:
                ttk.Label(self.project_config_container, text=f"另有 {len(pending) - 10} 项候选保存在项目配置中。", style="PathText.TLabel").pack(anchor="w")
        names = [name for name, _path in self.project_validation_areas]
        self.project_region_combo.configure(values=names)
        if self.project_region.get() not in names:
            self.project_region.set(names[0] if names else "")
        region = self._selected_project_region()
        area_path = next((path for name, path in self.project_validation_areas if name == region), "")
        self.project_validation_path.set(area_path or "尚未选择验证区。")
        for child in self.project_config_container.winfo_children():
            child.destroy()
        if not region:
            ttk.Label(
                self.project_config_container,
                text="连接项目文件夹后，这里会按区域列出多期影像和变化真值；也可先添加区域。",
                style="PathText.TLabel",
            ).pack(anchor="w")
            if hasattr(self, "add_project_period_button"):
                self.add_project_period_button.state(["disabled"])
            render_candidates()
            self._refresh_stage_selectors()
            return
        if hasattr(self, "add_project_period_button"):
            self.add_project_period_button.state(["!disabled"])
        rows = sorted(self.project_area_periods.get(region, []), key=lambda row: period_sort_key(row[0]))
        ttk.Label(self.project_config_container, text="多时间影像", style="PathTitle.TLabel").pack(anchor="w", pady=(0, 5))
        if not rows:
            ttk.Label(self.project_config_container, text="尚未添加影像期次。", style="PathText.TLabel").pack(anchor="w")
        for period, source in rows:
            row = ttk.Frame(self.project_config_container, style="Soft.TFrame")
            row.pack(fill=X, pady=2)
            ttk.Label(row, text=period, width=18, style="PathText.TLabel").pack(side=LEFT)
            ttk.Label(row, text=source, style="PathText.TLabel", anchor="w").pack(side=LEFT, fill=X, expand=True)
            ttk.Button(row, text="选择", style="Compact.TButton", command=lambda p=period: self.replace_project_period_source(p)).pack(side=LEFT, padx=(6, 0))
            ttk.Button(row, text="移除", style="Compact.TButton", command=lambda p=period: self.remove_project_period(p)).pack(side=LEFT, padx=(4, 0))
        ttk.Label(self.project_config_container, text="变化真值（可选）", style="PathTitle.TLabel").pack(anchor="w", pady=(10, 5))
        truth_map = {
            (area, before, after): path
            for area, before, after, path in self.project_area_truths
        }
        pairs = [(before[0], after[0]) for before, after in zip(rows, rows[1:])]
        if not pairs:
            ttk.Label(self.project_config_container, text="至少添加两个期次后才会生成相邻变化对。", style="PathText.TLabel").pack(anchor="w")
        for before, after in pairs:
            row = ttk.Frame(self.project_config_container, style="Soft.TFrame")
            row.pack(fill=X, pady=2)
            ttk.Label(row, text=f"{before} → {after}", width=18, style="PathText.TLabel").pack(side=LEFT)
            truth = truth_map.get((region, before, after), "")
            ttk.Label(row, text=truth or "未设置", style="PathText.TLabel", anchor="w").pack(side=LEFT, fill=X, expand=True)
            ttk.Button(row, text="选择", style="Compact.TButton", command=lambda b=before, a=after: self.set_project_truth(b, a)).pack(side=LEFT, padx=(6, 0))
            if truth:
                ttk.Button(row, text="移除", style="Compact.TButton", command=lambda b=before, a=after: self.remove_project_truth(b, a)).pack(side=LEFT, padx=(4, 0))
        render_candidates()
        self._refresh_stage_selectors()
        self._schedule_content_layout()

    def _refresh_stage_selectors(self) -> None:
        if not hasattr(self, "stage_region_combo"):
            return
        names = [name for name, _path in self.project_validation_areas]
        self.stage_region_combo.configure(values=names)
        if self.project_region.get() not in names:
            self.project_region.set(names[0] if names else "")
        region = self._selected_project_region()
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
        self._refresh_project_config_panel()

    def _project_payload(self) -> dict:
        payload = dict(self.project_config)
        payload.update({
            "version": 3,
            "project_root": self.project_root_path,
            "external_data_sources": list(dict.fromkeys(self.project_data_sources)),
            "validation_areas": [list(row) for row in self.project_validation_areas],
            "area_periods": {
                area: [list(row) for row in rows]
                for area, rows in self.project_area_periods.items()
            },
            "area_truths": [list(row) for row in self.project_area_truths],
            "unmapped_candidates": self.project_candidates,
            "output_root": self.vars["output_root"].get().strip(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return payload

    def _save_project_config(self, *, notify: bool = False) -> bool:
        if not self.project_root_path:
            return False
        try:
            path = atomic_write_json(project_config_path(self.project_root_path), self._project_payload())
            self.project_config = read_project_config(self.project_root_path)
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
        discovered_truths = [
            (area, before, after, path)
            for (area, before, after), path in (project.get("area_truths") or {}).items()
        ]
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
        self.project_config = dict(payload)
        self.project_data_sources = [str(Path(value).expanduser().resolve()) for value in payload.get("external_data_sources", []) if str(value).strip()]
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
        candidates = payload.get("unmapped_candidates") or {}
        self.project_candidates = {
            "shp": [str(value) for value in candidates.get("shp", [])],
            "txt": [str(value) for value in candidates.get("txt", [])],
        }
        output_root = str(payload.get("output_root") or "").strip()
        if output_root:
            self.vars["output_root"].set(output_root)
        active = payload.get("active_task") or {}
        if isinstance(active, dict) and str(active.get("run_id") or "").strip():
            self.vars["run_id"].set(str(active["run_id"]))
        self.data_source_display.set("；".join(self.project_data_sources) if self.project_data_sources else "尚未连接外部数据源")

    def connect_data_source(self) -> None:
        if not self.project_root_path:
            messagebox.showinfo("请先打开项目", "请先新建或打开项目文件夹，再连接外部原始数据源。", parent=self.root)
            return
        directory = filedialog.askdirectory(parent=self.root, title="连接外部原始数据源")
        if not directory:
            return
        resolved = str(Path(directory).resolve())
        if resolved not in self.project_data_sources:
            self.project_data_sources.append(resolved)
        self.data_source_display.set("；".join(self.project_data_sources))
        self.data_status.set("已连接，尚未扫描")
        self._save_project_config()
        self.scan_data_sources()

    def scan_data_sources(self) -> None:
        if not self.project_data_sources:
            self.data_status.set("未连接数据源")
            messagebox.showinfo("未连接数据源", "请先连接一个或多个外部原始数据目录。", parent=self.root)
            return
        discovered_count = 0
        candidates = {"shp": [], "txt": []}
        for source in self.project_data_sources:
            try:
                scan = scan_external_data_source(source)
            except ValueError as exc:
                messagebox.showerror("数据源不可用", str(exc), parent=self.root)
                self.data_status.set("数据检查失败")
                return
            for kind in candidates:
                candidates[kind].extend(scan["candidates"][kind])
            if scan["discovered"]:
                self._apply_discovered_project(scan["discovered"], merge=True)
                discovered_count += 1
        mapped = {
            path for _area, path in self.project_validation_areas
        } | {
            path for rows in self.project_area_periods.values() for _period, path in rows
        } | {
            path for _area, _before, _after, path in self.project_area_truths
        }
        self.project_candidates = {
            kind: sorted({path for path in paths if path not in mapped}, key=natural_key)
            for kind, paths in candidates.items()
        }
        pending = sum(len(values) for values in self.project_candidates.values())
        self.data_status.set("已扫描，存在待确认项" if pending else "已扫描，等待数据检查")
        self.project_scan_summary.set(
            f"已扫描 {len(self.project_data_sources)} 个外部目录；自动识别 {len(self.project_validation_areas)} 个区域、"
            f"{sum(len(rows) for rows in self.project_area_periods.values())} 个期次；待确认候选 {pending} 项。"
        )
        self._refresh_project_config_panel()
        self._refresh_input_summary()
        self._save_project_config()
        self.status.set(self.project_scan_summary.get())

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
        self.project_region.set(name)
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
        self.project_region.set("")
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
            for child in ("04_成果输出", "_logs"):
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
        self.project_data_sources = []
        self.project_candidates = {"shp": [], "txt": []}
        self.vars["output_root"].set(str((root / "04_成果输出").resolve()))
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
            payload = read_project_config(root)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            messagebox.showerror("项目配置不可用", str(exc), parent=self.root)
            return
        legacy_project = None
        if not payload:
            try:
                legacy_project = discover_validation_project(root)
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
            self._apply_discovered_project(legacy_project)
            self.vars["output_root"].set(str(legacy_project["output_root"]))
            self.data_source_display.set(str(root))
        else:
            self.project_config = {}
            self.project_data_sources = []
            self.project_validation_areas = []
            self.project_area_truths = []
            self.project_area_periods = {}
            self.project_candidates = {"shp": [], "txt": []}
            output = root / "04_成果输出"
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
        pending = sum(len(values) for values in self.project_candidates.values())
        self.data_status.set(
            "未连接数据源" if not self.project_data_sources else
            ("已扫描，存在待确认项" if pending else "已扫描，等待数据检查")
        )
        self.project_scan_summary.set(
            f"项目已打开：{len(self.project_validation_areas)} 个验证区、"
            f"{sum(len(rows) for rows in self.project_area_periods.values())} 个影像期次、"
            f"{len(self.project_area_truths)} 个变化真值、{pending} 个待确认候选。"
        )
        unfinished = unfinished_task_state(
            self.vars["output_root"].get(), self.project_config.get("active_task"),
        )
        if unfinished is not None:
            notice = unfinished_task_message(unfinished)
            self.status.set(notice.replace("\n", " "))
            self.run_status.set(notice)
            self.preflight_summary.set("检测到同名未完成任务；点击“运行完整流程”将自动续跑。")
        else:
            self.status.set(self.project_scan_summary.get())
        self._save_project_config()

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
            atomic_write_json(path, payload)
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
            pairs = "、".join(f"{before} → {after}" for before, after in manifest["change_pairs"])
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
            ttk.Label(frame, text=f"{before} → {after}", width=23).pack(side=LEFT)
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
            self.grid_toggle.configure(text="▼ 兼容旧版多格网目录")
        else:
            self.grid_options.pack_forget()
            self.grid_toggle.configure(text="▶ 兼容旧版多格网目录")
        self._refresh_input_summary()
        self._schedule_content_layout()

    def _build_current_command(self, *, preflight_only: bool = False, data_check_only: bool = False) -> list[str]:
        return build_pipeline_command(
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
        run_id, should_resume, state_path = resolve_automatic_run(
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
        region = self._selected_project_region()
        if not region:
            raise ValueError("请选择需要处理的区域。")
        run_id = self.vars["run_id"].get().strip()
        if not run_id:
            run_id = time.strftime("stage_%Y%m%d_%H%M%S")
            self.vars["run_id"].set(run_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("分步任务名称只能包含字母、数字、连字符和下划线。")
        return root.resolve(), region, run_id

    def _project_stage_common_args(self) -> list[str]:
        return [
            "--device", self.vars["device"].get(),
            "--pixel-size", self.vars["pixel_size"].get(),
            "--rescale", self.vars["rescale"].get(),
            "--junction-node-mode", self.vars["junction_node_mode"].get(),
        ]

    def _stage_period_changed(self, _event=None) -> None:
        region = self._selected_project_region()
        periods = [period for period, _source in self.project_area_periods.get(region, [])]
        pairs = affected_change_pairs(periods, self.project_period.get().strip())
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
            region = self._selected_project_region()
            period = self.project_period.get().strip()
            if not region or not period:
                raise ValueError("请选择需要重跑的区域和影像期次。")
            args = ["rerun-period", "--pipeline-manifest", str(manifest), "--grid", region, "--period", period]
            if update_related:
                args.append("--update-related")
        except ValueError as exc:
            messagebox.showerror("无法局部重跑", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(args)

    def rerun_selected_change(self, update_temporal: bool) -> None:
        try:
            manifest = self._current_pipeline_manifest_path()
            region = self._selected_project_region()
            pair = self.project_change_pair.get().strip()
            if not region or "→" not in pair:
                raise ValueError("请选择需要重跑的区域和相邻变化对。")
            before, after = (value.strip() for value in pair.split("→", 1))
            args = [
                "rerun-change", "--pipeline-manifest", str(manifest), "--grid", region,
                "--before-period", before, "--after-period", after,
            ]
            if update_temporal:
                args.append("--update-temporal")
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
            args = [
                "extract-project-all", "--project-root", str(root), "--run-id", run_id,
                *self._project_stage_common_args(),
            ]
            output_root = Path(self.vars["output_root"].get()).expanduser()
            state = output_root / "batch_extractions" / _safe_task_name(run_id) / "batch_extract_task.json"
            if state.is_file():
                args.append("--resume")
            if self.vars["continue_on_error"].get() == "1":
                args.append("--continue-on-error")
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
            args = [
                "extract-project-period", "--project-root", str(root), "--area-id", region,
                "--period", period, "--run-id", run_id, *self._project_stage_common_args(),
            ]
            output_root = Path(self.vars["output_root"].get()).expanduser()
            state = (
                output_root / "period_extractions" / _safe_task_name(region) /
                _safe_task_name(period) / _safe_task_name(run_id) / "period_task.json"
            )
            if state.is_file():
                args.append("--resume")
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
            if isinstance(manifest, dict):
                periods = {
                    (str(entry.get("grid")), str(entry.get("period"))): entry
                    for entry in manifest.get("period_results", []) or [] if isinstance(entry, dict)
                }
                change_entry = next((
                    entry for entry in manifest.get("change_results", []) or []
                    if isinstance(entry, dict) and str(entry.get("grid")) == region
                    and str(entry.get("before_period")) == before and str(entry.get("after_period")) == after
                ), None)
                before_entry, after_entry = periods.get((region, before)), periods.get((region, after))
                if change_entry and before_entry and after_entry:
                    output = str(change_entry.get("output") or Path(str(change_entry.get("gpkg") or "")).parent)
                    args = [
                        "change", "--before-result", str(before_entry["result"]),
                        "--after-result", str(after_entry["result"]), "--output", output,
                        "--before-period", before, "--after-period", after,
                        "--absolute", self.vars["absolute"].get(), "--ratio", self.vars["ratio"].get(),
                        "--tolerance", self.vars["tolerance"].get(),
                    ]
                    truth = next((path for area, one, two, path in self.project_area_truths if (area, one, two) == (region, before, after)), "")
                    validation = next((path for area, path in self.project_validation_areas if area == region), "")
                    if truth:
                        args.extend(("--truth", truth))
                    if validation:
                        args.extend(("--validation-area", validation))
                    if self.vars["truth_type_field"].get().strip():
                        args.extend(("--truth-type-field", self.vars["truth_type_field"].get().strip()))
                    self._show_step(1, force=True)
                    self._command(args)
                    return
            output_root = Path(self.vars["output_root"].get()).expanduser()
            base = output_root / "period_extractions" / _safe_task_name(region)
            before_state = base / _safe_task_name(before) / _safe_task_name(run_id) / "period_task.json"
            after_state = base / _safe_task_name(after) / _safe_task_name(run_id) / "period_task.json"
            if not before_state.is_file() or not after_state.is_file():
                raise ValueError("所选前后期尚无分步提取状态。请先重跑并完成两个期次。")
            args = [
                "change-project-periods", "--project-root", str(root), "--area-id", region,
                "--before-period", before, "--after-period", after,
                "--before-state", str(before_state), "--after-state", str(after_state),
                "--run-id", run_id, "--absolute", self.vars["absolute"].get(),
                "--ratio", self.vars["ratio"].get(), "--tolerance", self.vars["tolerance"].get(),
            ]
            change_state = (
                output_root / "period_changes" / _safe_task_name(region) /
                f"{_safe_task_name(before)}_to_{_safe_task_name(after)}" / run_id / "change_task.json"
            )
            if change_state.is_file():
                args.append("--resume")
        except (KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("无法分步检测", str(exc), parent=self.root)
            return
        self._show_step(1, force=True)
        self._command(args)

    def _command(self, args: list[str]) -> None:
        if self.process is not None:
            messagebox.showwarning("已有任务", "当前已有任务在运行，请等待完成。", parent=self.root)
            return
        self.log.delete("1.0", END)
        self.shared_log_status.set("任务启动中…")
        self.recent_log_lines = []
        self.status.set("任务启动中…")
        self.run_status.set("任务启动中，请稍候…")
        self.run_button.state(["disabled"])
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
                log_dir = Path(output_value).expanduser() / "_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self.active_log_path = log_dir / f"{run_name}.log"
            except (ValueError, IndexError, OSError):
                self.active_log_path = None
        elif args and args[0] == "apply-edits":
            try:
                if "--pipeline-manifest" in args:
                    log_root = Path(args[args.index("--pipeline-manifest") + 1]).expanduser().resolve().parent
                else:
                    log_root = Path(args[args.index("--result") + 1]).expanduser().resolve().parent
                log_dir = log_root / "_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self.active_log_path = log_dir / f"人工编辑重建_{time.strftime('%Y%m%d_%H%M%S')}.log"
            except (ValueError, IndexError, OSError):
                self.active_log_path = None
        elif args and args[0] in {"evaluate-existing", "evaluate-all-existing"}:
            try:
                log_root = Path(args[args.index("--pipeline-manifest") + 1]).expanduser().resolve().parent
                log_dir = log_root / "_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self.active_log_path = log_dir / f"精度评价_{time.strftime('%Y%m%d_%H%M%S')}.log"
            except (ValueError, IndexError, OSError):
                self.active_log_path = None
        elif args:
            try:
                log_root = Path(self.vars["output_root"].get()).expanduser().resolve()
                log_dir = log_root / "_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self.active_log_path = log_dir / f"{_safe_task_name(args[0])}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            except OSError:
                self.active_log_path = None
        self.progress.configure(maximum=1, value=0)
        if hasattr(self, "review_progress"):
            self.review_progress.configure(maximum=1, value=0)
        self.progress_text.set("0 / 0 · 已用时 00:00:00 · 剩余 --")

        def worker() -> None:
            try:
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                site_packages = ROOT / "env" / "samroad_env" / "Lib" / "site-packages"
                proj_data = site_packages / "rasterio" / "proj_data"
                gdal_data = site_packages / "rasterio" / "gdal_data"
                if (proj_data / "proj.db").is_file():
                    env["PROJ_DATA"] = str(proj_data)
                    env["PROJ_LIB"] = str(proj_data)
                if gdal_data.is_dir():
                    env["GDAL_DATA"] = str(gdal_data)
                self.process = subprocess.Popen(
                    [sys.executable, str(BACKEND), *args],
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0),
                )
                assert self.process.stdout is not None
                log_file = None
                try:
                    if self.active_log_path is not None:
                        log_file = self.active_log_path.open("a", encoding="utf-8", newline="")
                        log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                    for line in self.process.stdout:
                        visible_line = write_and_filter_gui_log(line, log_file)
                        if visible_line is not None:
                            self.queue.put(("log", visible_line))
                finally:
                    if log_file is not None:
                        log_file.close()
                self.queue.put(("done", str(self.process.wait())))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

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
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=False,
                )
            else:
                process.terminate()
        except OSError as exc:
            self.status.set(f"取消失败：{exc}")

    def _append_log(self, stage: str, message: str) -> None:
        line = f"[{stage}] {message}" if stage else message
        self.recent_log_lines.append(line)
        if len(self.recent_log_lines) > 2000:
            del self.recent_log_lines[:200]
            self.log.delete("1.0", "201.0")
        self.log.insert(END, line + "\n")
        self.log.see(END)
        summary = " ".join(line.split())
        if len(summary) > 110:
            summary = summary[:107] + "…"
        self.shared_log_status.set(summary or "日志已更新")

    def _set_cancel_enabled(self, enabled: bool) -> None:
        state = ["!disabled"] if enabled else ["disabled"]
        self.cancel_button.state(state)
        if hasattr(self, "review_cancel_button"):
            self.review_cancel_button.state(state)

    def _set_stage_buttons_enabled(self, enabled: bool) -> None:
        for button in getattr(self, "stage_buttons", []):
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
        mark_task_cancelled(output, run_id)

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.queue.get_nowait()
                if kind == "log":
                    if value.startswith("__SAMROAD_USER__"):
                        try:
                            payload = json.loads(value[len("__SAMROAD_USER__"):])
                            friendly = self._friendly(payload)
                            self.status.set(friendly)
                            primary_status = structured_task_status(payload)
                            if primary_status is not None:
                                self.current_stage_payload = payload
                                self.run_status.set(primary_status)
                            elif payload.get("kind") == "pipeline" or (
                                payload.get("kind") == "complete" and payload.get("stage") == "all"
                            ):
                                self.current_stage_payload = None
                                self.run_status.set(friendly)
                            elif self.current_stage_payload is None:
                                self.run_status.set(friendly)
                            self._update_progress(payload)
                            if payload.get("kind") == "complete":
                                self.last_complete_payload = payload
                            self._append_log(payload.get("stage", payload.get("kind", "")), friendly)
                        except json.JSONDecodeError:
                            self._append_log("日志", value)
                    else:
                        self._append_log("日志", value)
                elif kind == "editor_stdout":
                    self.editor_stdout_lines.append(value)
                    if len(self.editor_stdout_lines) > 200:
                        del self.editor_stdout_lines[:-200]
                    if value:
                        self._append_log("人工编辑", value)
                elif kind == "editor_stderr":
                    self.editor_stderr_lines.append(value)
                    if len(self.editor_stderr_lines) > 200:
                        del self.editor_stderr_lines[:-200]
                    if value:
                        self._append_log("人工编辑错误", value)
                elif kind == "done":
                    if self.cancel_requested:
                        if self.active_command == "all":
                            self._mark_cancelled_state()
                        if self.active_command == "apply-edits":
                            self.status.set("人工编辑增量重建已停止；已有正式结果未被删除，可再次应用编辑。")
                            self.run_status.set("增量重建已停止；查看上方日志后可再次点击“应用编辑并重新生成结果”。")
                        else:
                            self.status.set("任务已取消；已完成结果和当前位置已保留。")
                            self.run_status.set("任务已取消；再次点击“运行完整流程”将自动从未完成步骤继续。")
                    elif value == "0" and self.active_command in {"preflight", "data-check"}:
                        payload = self.last_complete_payload or {"kind": "complete", "stage": self.active_command}
                        self.preflight_passed = True
                        pending_candidates = sum(len(values) for values in self.project_candidates.values())
                        self.data_status.set("已扫描，存在待确认项" if pending_candidates else "数据已就绪")
                        self.run_button.state(["!disabled"])
                        self.status.set(self._friendly(payload))
                        self.run_status.set(self._friendly(payload))
                        self.preflight_summary.set("✓ 所有阻断性检查均已通过，可以开始处理。")
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
                            self.evaluation_status.set(self.status.get())
                        self._show_step(3, force=True)
                    elif value == "0" and self.active_command in {
                        "extract-project-period", "extract-project-all", "change-project-periods", "change",
                        "rerun-period", "rerun-change", "rerun-all-periods",
                    }:
                        labels = {
                            "extract-project-period": "所选期次道路提取已完成。",
                            "extract-project-all": "全部期次道路提取已完成。",
                            "change-project-periods": "所选变化对检测已完成。",
                            "change": "所选已有变化对已重新检测完成。",
                            "rerun-period": "所选期次已按指定范围重跑完成。",
                            "rerun-change": "所选变化对已重跑完成。",
                            "rerun-all-periods": "全部道路提取已按依赖顺序批量重跑完成。",
                        }
                        self.status.set(labels[self.active_command])
                        self.run_status.set(labels[self.active_command])
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
                    self.run_button.state(["!disabled"])
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
                    self.run_button.state(["!disabled"])
                    self.preflight_button.state(["!disabled"])
                    self._set_stage_buttons_enabled(True)
                    if hasattr(self, "apply_review_button"):
                        self.apply_review_button.state(["!disabled"])
                    self._set_cancel_enabled(False)
        except queue.Empty:
            pass
        self._poll_geometry_editor()
        self._refresh_progress_text()
        self.root.after(100, self._poll)

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
            return (
                f"全部完成：{payload.get('grid_count')} 个格网、{payload.get('period_count')} 次提取、"
                f"{payload.get('change_count')} 次变化检测、{payload.get('failure_count', 0)} 项失败"
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
            offset = payload.get("centerline_avg_offset_m")
            offset_text = f"，新增/灭失中心线平均偏移 {float(offset):.3f} 米" if offset not in {None, ""} else ""
            return (
                f"精度评价完成：全部变化区域查全率 "
                f"{float(payload.get('change_area_recall', 0)):.3f}，"
                f"新增/变化/灭失判断正确率 "
                f"{float(payload.get('type_judgment_accuracy', 0)):.3f}{offset_text}。"
            )
        if kind == "complete" and payload.get("stage") == "evaluate-all-existing":
            offset = payload.get("centerline_avg_offset_m")
            offset_text = f"，新增/灭失中心线平均偏移 {float(offset):.3f} 米" if offset not in {None, ""} else ""
            return (
                f"总精度评价完成：{payload.get('evaluated_task_count', 0)} 个区域/变化对，"
                f"全部变化区域查全率 {float(payload.get('change_area_recall', 0)):.3f}，"
                f"类型判断正确率 {float(payload.get('type_judgment_accuracy', 0)):.3f}{offset_text}。"
            )
        if kind == "stage" and status == "complete":
            return f"{payload.get('stage', '阶段')}完成{elapsed_text}。"
        if kind == "complete":
            return f"{payload.get('stage', '阶段')}完成{elapsed_text}。"
        return f"{payload.get('stage', kind or '任务')}：{payload.get('status', '处理中')}"

    def _latest_manifest(self) -> tuple[dict | None, Path | None]:
        if self.loaded_manifest_path is not None and self.loaded_manifest_path.is_file():
            latest = self.loaded_manifest_path
        else:
            latest = Path(self.vars["output_root"].get().strip()) / "latest_pipeline.json"
        if not latest.is_file():
            return None, latest
        try:
            value = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None, latest
        return (value, latest) if isinstance(value, dict) else (None, latest)

    def _refresh_result_availability(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            self.results_available = False
            self.review_items = []
            self.temporal_items = []
            self.result_status.set("尚无可显示的任务成果；完成任务或载入已有成果后即可查看。")
            if hasattr(self, "result_period_count"):
                self.result_period_count.set("0 期")
                self.result_change_count.set("0 组")
                self.result_area_count.set("0 区")
                self.result_review_count.set("0 处")
            if hasattr(self, "review_status"):
                self.review_status.set("尚未完成自动处理，暂无可复核数据。")
                self._populate_review_step()
            if hasattr(self, "evaluation_pair_combo"):
                self.result_change_items = []
                self.evaluation_pair_combo.configure(values=())
                self.evaluation_pair.set("")
                self.evaluation_status.set("请先载入或生成包含变化检测的任务结果。")
                self.run_evaluation_button.state(["disabled"])
            self._populate_result_tree([], None)
            return
        base_dir = latest.parent
        self.review_items = collect_review_items(manifest, base_dir)
        self.temporal_items = collect_temporal_items(manifest, base_dir)
        self.results_available = bool(
            manifest.get("period_results") or manifest.get("change_results")
            or self.review_items or self.temporal_items
        )
        area_names = {
            str(entry.get("grid") or "validation")
            for entry in (manifest.get("period_results", []) or []) if isinstance(entry, dict)
        }
        period_count = len(manifest.get("period_results", []) or [])
        change_count = len(manifest.get("change_results", []) or [])
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
        self._populate_result_tree(collect_result_tree_items(manifest, base_dir), base_dir)
        self._refresh_evaluation_results(manifest)
        self._populate_review_step()

    def _populate_result_tree(self, items: list[dict[str, str]], _base_dir: Path | None) -> None:
        if not hasattr(self, "result_tree"):
            return
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
                node = self.result_tree.insert(
                    parent, END, iid=item["id"], text=item["label"], values=(item.get("status", ""),),
                    open=not bool(parent),
                )
                inserted.add(node)
                path_text = item.get("path", "")
                if path_text:
                    self.result_tree_paths[node] = Path(path_text)
            if len(next_pending) == len(pending):
                break
            pending = next_pending

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

    def _refresh_evaluation_results(self, manifest: dict) -> None:
        if not hasattr(self, "evaluation_pair_combo"):
            return
        self.result_change_items = [
            entry for entry in (manifest.get("change_results", []) or [])
            if isinstance(entry, dict) and (
                (entry.get("gpkg") and Path(str(entry["gpkg"])).is_file())
                or (entry.get("summary") and Path(str(entry["summary"])).is_file())
            )
        ]
        labels = [
            f"{entry.get('grid', '项目')} · {entry.get('before_period', '前期')} → {entry.get('after_period', '后期')}"
            for entry in self.result_change_items
        ]
        self.evaluation_pair_combo.configure(values=labels)
        if labels and self.evaluation_pair.get() not in labels:
            self.evaluation_pair.set(labels[0])
        if not labels:
            self.evaluation_pair.set("")
            self.evaluation_status.set("当前任务没有可评价的变化检测成果。")
            self.run_evaluation_button.state(["disabled"])
            self.run_total_evaluation_button.state(["disabled"])
            return
        evaluated = sum(bool(entry.get("evaluation_metrics")) for entry in self.result_change_items)
        if evaluated:
            status = f"当前共有 {len(labels)} 组检测结果可进行精度评价，已完成 {evaluated} 组。"
        else:
            status = f"当前共有 {len(labels)} 组检测结果可进行精度评价。"
        selected = self.result_change_items[labels.index(self.evaluation_pair.get())]
        summary_path = Path(str(selected.get("summary") or ""))
        if selected.get("evaluation_metrics") and summary_path.is_file():
            try:
                evaluation = json.loads(summary_path.read_text(encoding="utf-8")).get("evaluation", {})
                overall = next(row for row in evaluation.get("metrics", []) if row.get("class") == "all")
                status += (
                    f" 当前评价结果：变化区域查全率 "
                    f"{format_percentage(overall.get('change_area_recall', overall['recall']))}，"
                    f"变化类型判断准确率 "
                    f"{format_percentage(overall.get('type_judgment_accuracy', 0))}。"
                )
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, StopIteration, TypeError, ValueError):
                pass
        aggregate = manifest.get("evaluation_summary") or {}
        aggregate_overall = next((
            row for row in aggregate.get("metrics", []) if isinstance(row, dict) and row.get("class") == "all"
        ), None)
        if aggregate_overall:
            offset = aggregate_overall.get("centerline_avg_offset_m")
            status += (
                f" 汇总结果：变化区域查全率 {format_percentage(aggregate_overall.get('change_area_recall', aggregate_overall.get('recall', 0)))}，"
                f"变化类型判断准确率 {format_percentage(aggregate_overall.get('type_judgment_accuracy', 0) or 0)}"
                + (f"，新增/灭失道路中心线平均偏差 {float(offset):.2f} 米。" if offset not in {None, ""} else "；中心线偏差暂无可用值。")
            )
        self.evaluation_status.set(status)
        configured = len(self.project_area_truths)
        embedded = sum(bool(str(entry.get("truth") or "").strip()) for entry in self.result_change_items)
        self.evaluation_truth_summary.set(
            f"总体评价将自动匹配 {max(configured, embedded)} 个区域/相邻期次真值；"
            "上方路径只用于“评价当前结果”。"
        )
        self._evaluation_pair_changed()
        self.run_evaluation_button.state(["!disabled"])
        self.run_total_evaluation_button.state(["!disabled"])

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

    def supplement_evaluation_truth(self) -> None:
        if not self.result_change_items or not hasattr(self, "evaluation_pair_combo"):
            return
        try:
            index = list(self.evaluation_pair_combo["values"]).index(self.evaluation_pair.get())
            item = self.result_change_items[index]
        except (ValueError, IndexError):
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
            project = discover_validation_project(self.project_root_path)
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
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无成果", "请先载入或生成任务结果。", parent=self.root)
            return
        if self.evaluation_truth.get().strip() in {"", "缺少真值"}:
            messagebox.showinfo("缺少真值", "请先为所选变化对补充真值。", parent=self.root)
            return
        try:
            index = list(self.evaluation_pair_combo["values"]).index(self.evaluation_pair.get())
            item = self.result_change_items[index]
            args = build_evaluate_existing_command(
                item, latest, self.evaluation_truth.get(),
                truth_type_field=self.evaluation_type_field.get(),
                validation_area=str(item.get("validation_area") or manifest.get("validation_area") or ""),
                evaluation_tolerance=self.evaluation_tolerance.get(),
            )
        except (ValueError, IndexError) as exc:
            messagebox.showerror("无法运行精度评价", str(exc), parent=self.root)
            return
        self.evaluation_status.set("正在读取检测结果并计算区域、类型和中心线精度指标…")
        self._command(args)

    def run_total_evaluation(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无成果", "请先载入或生成任务结果。", parent=self.root)
            return
        try:
            args = build_evaluate_all_command(
                manifest, latest, self.project_area_truths,
                truth_type_field=self.evaluation_type_field.get(),
                evaluation_tolerance=self.evaluation_tolerance.get(),
            )
        except ValueError as exc:
            messagebox.showerror("无法运行总精度评价", str(exc), parent=self.root)
            return
        self.evaluation_status.set("正在评价全部验证区和相邻期次，并汇总精度与中心线偏差…")
        self._command(args)

    def open_latest(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无成果", f"尚未找到最新成果索引：\n{latest}", parent=self.root)
            return
        self._refresh_result_availability()
        try:
            job_root = Path(str(manifest["job_root"]))
        except (KeyError, TypeError, ValueError):
            job_root = latest
        self._open(job_root)

    def load_existing_results(self) -> None:
        """Attach the GUI to an existing saved run without executing inference."""
        selected = filedialog.askopenfilename(
            parent=self.root, title="选择已有任务结果索引",
            filetypes=(("SAMRoad 任务索引", "*.json"),),
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            messagebox.showerror("无法读取已有结果", str(exc), parent=self.root)
            return
        if not isinstance(manifest, dict) or not isinstance(manifest.get("period_results"), list):
            messagebox.showerror("不是任务结果索引", "请选择 latest_pipeline.json 或 pipeline_result.json。", parent=self.root)
            return
        self.loaded_manifest_path = path
        self.vars["output_root"].set(str(path.parent if path.name == "latest_pipeline.json" else Path(manifest.get("job_root", path.parent)).parent))
        self.current_project.set(f"已有结果：{manifest.get('run_id', path.parent.name)}")
        self.preflight_passed = True
        self._refresh_result_availability()
        self._show_step(3, force=True)
        self.status.set("已载入已有成果；可直接查看成果或进入人工编辑，不会运行推理。")

    def open_task_report(self) -> None:
        manifest, latest = self._latest_manifest()
        if manifest is None or latest is None:
            messagebox.showinfo("暂无报告", "尚未找到最新任务索引。", parent=self.root)
            return
        job_root = Path(str(manifest.get("job_root") or latest.parent)).expanduser()
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

    def open_temporal_attribute_table(self) -> None:
        self._refresh_result_availability()
        if not self.temporal_items:
            messagebox.showinfo("暂无长时序属性表", "已有任务中没有可读取的 road_life.shp。", parent=self.root)
            return
        try:
            import geopandas as gpd
            frame = gpd.read_file(self.temporal_items[0]["path"])
        except Exception as exc:
            messagebox.showerror("属性表读取失败", str(exc), parent=self.root)
            return
        window = Toplevel(self.root)
        window.title("长时序道路属性表 · road_life.shp")
        configure_window_geometry(window, base_width=1400, base_height=760, min_width=980, min_height=560)
        window.configure(background=UI["page"])
        header = ttk.Frame(window, padding=(20, 13), style="Header.TFrame")
        header.pack(fill=X)
        ttk.Label(header, text="ROAD LIFE", style="Brand.TLabel").pack(side=LEFT, padx=(0, 20))
        title_area = ttk.Frame(header, style="Header.TFrame")
        title_area.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(title_area, text="长时序道路属性表", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_area, text=f"{self.temporal_items[0]['label']}  ·  road_life.shp",
            style="HeaderMeta.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(header, text=f"共 {len(frame)} 条道路", style="HeaderProject.TLabel").pack(side=RIGHT)

        filters = ttk.Frame(window, padding=(18, 12), style="Card.TFrame")
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
        columns = [name for name in frame.columns if name != frame.geometry.name]
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

        def fill(*_args) -> None:
            table.delete(*table.get_children())
            needle = search_var.get().strip().casefold()
            shown = 0
            for _, row in frame.iterrows():
                values = ["" if row[name] is None else str(row[name]) for name in columns]
                if needle and needle not in " ".join(values).casefold():
                    continue
                state = str(row.get("review_state", row.get("life_state", ""))).casefold()
                tags = ("review",) if "review" in state or "uncertain" in state else (("odd",) if shown % 2 else ())
                table.insert("", END, values=values, tags=tags)
                shown += 1
            match_var.set(f"显示 {shown} / {len(frame)} 条")

        search_var.trace_add("write", fill)
        fill()
        footer = ttk.Frame(window, padding=(16, 8), style="Footer.TFrame")
        footer.pack(fill=X)
        ttk.Label(
            footer, text="横向滚动查看全部时序字段  ·  单击表头可清晰对应各列",
            style="FooterStatus.TLabel",
        ).pack(side=LEFT)
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
        log_path = Path(self.vars["output_root"].get().strip()) / "_logs" / f"{manifest.get('run_id')}.log"
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
        saved, _checked = find_saved_edit_directory(item)
        state = "已人工编辑，待更新" if saved else "可编辑"
        result_path = find_period_result(item)
        if result_path is not None:
            try:
                period_result = json.loads(result_path.read_text(encoding="utf-8"))
                if (period_result.get("manual_edit") or {}).get("applied"):
                    state = "已更新"
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        region = item.get("grid", "")
        periods = [period for period, _source in self.project_area_periods.get(region, [])]
        pairs = affected_change_pairs(periods, item.get("scope", ""))
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

    def _read_geometry_editor_stream(self, kind: str, stream) -> None:
        try:
            for line in stream:
                self.queue.put((kind, line.rstrip("\r\n")))
        finally:
            stream.close()

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
        state, detail = geometry_editor_process_state(
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
        diagnostics = geometry_editor_diagnostics(review_dir)
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
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen(
                build_geometry_editor_command(script, item, ready_file),
                cwd=str(script.parent), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
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
        assert process.stdout is not None and process.stderr is not None
        threading.Thread(
            target=self._read_geometry_editor_stream,
            args=("editor_stdout", process.stdout), daemon=True,
        ).start()
        threading.Thread(
            target=self._read_geometry_editor_stream,
            args=("editor_stderr", process.stderr), daemon=True,
        ).start()

    def apply_selected_review(self) -> None:
        item = self._selected_review_item()
        if item is None:
            return
        result_path = find_period_result(item)
        if result_path is None:
            messagebox.showerror(
                "缺少期次结果索引",
                "无法定位该期次的 latest_result.json。请载入与本次人工编辑对应的任务结果索引。",
                parent=self.root,
            )
            return
        edited_dir, checked = find_saved_edit_directory(item, self.review_edit_directory.get())
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
            (path for path in candidates if manifest_contains_period_result(path, result_path)), None,
        )
        try:
            args = build_apply_edits_command(item, matching_manifest)
        except ValueError as exc:
            messagebox.showerror("缺少结果索引", str(exc), parent=self.root)
            return
        if matching_manifest is None:
            self.status.set("已找到编辑成果，但任务总索引与该期次不匹配；本次将重新测宽和生成面，不自动重跑相邻变化对。")
        self._command(args)

    @staticmethod
    def _open(path: Path) -> None:
        if sys.platform == "win32":
            os.startfile(str(path))
        else:
            subprocess.Popen(["xdg-open", str(path)])


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
