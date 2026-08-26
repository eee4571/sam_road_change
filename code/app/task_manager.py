from __future__ import annotations

"""Task commands, resume state and dependency calculations."""

import json
import os
import re
import time
from pathlib import Path

from input_catalog import period_order_manifest, period_sort_key
from app.project_manager import USER_IMAGE_LIST_SUFFIX, USER_VECTOR_SUFFIX, atomic_write_json
from app.result_publisher import ProjectLayout
from app.project_relocation import build_relocation_plan

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
    if state_value:
        state_path = Path(state_value).expanduser()
    else:
        output = Path(output_root).expanduser()
        layout = ProjectLayout.from_output(output)
        job_root = layout.existing_full_run_root(run_id) or layout.full_run_root(run_id)
        state_path = job_root / "job_state.json"
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
    layout = ProjectLayout.from_output(output)
    job_root = layout.existing_full_run_root(run_id) or layout.full_run_root(run_id)
    state_path = job_root / "job_state.json"
    if not state_path.is_file():
        return None
    try:
        manifest = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
            return None
        manifest["status"] = "cancelled"
        manifest["cancelled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        latest = (
            layout.legacy_latest_pipeline_path
            if job_root == layout.legacy_full_run_root(run_id)
            else layout.latest_pipeline_path
        )
        for target in (state_path, job_root / "pipeline_result.json", latest):
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

def affected_change_pairs(periods: list[str] | tuple[str, ...], selected: str) -> list[tuple[str, str]]:
    """Return the adjacent changes invalidated by replacing one period result."""
    ordered = sorted({str(value).strip() for value in periods if str(value).strip()}, key=period_sort_key)
    return [
        (before, after)
        for before, after in zip(ordered, ordered[1:])
        if selected in {before, after}
    ]

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
    execution_profile: str = "full",
) -> list[str]:
    """Build the backend command for the default validation or backup grid mode."""
    mode = str(mode or "validation").strip().casefold()
    if not str(output_root).strip():
        raise ValueError("请选择成果输出根目录。")
    args = ["all", "--mode", mode]
    execution_profile = str(execution_profile or "full").strip().casefold()
    if execution_profile not in {"full", "fast"}:
        raise ValueError("处理模式必须是 full 或 fast。")
    if execution_profile == "fast":
        args.extend(("--execution-profile", "fast"))
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
        if (
            evaluate and execution_profile != "fast"
            and (set(truth_map) != expected_truths or any(not truth_map.get(key) for key in expected_truths))
        ):
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
    output = Path(output_root).expanduser()
    layout = ProjectLayout.from_output(output)
    job_root = layout.existing_full_run_root(run_id) or layout.full_run_root(run_id)
    state = job_root / "job_state.json"
    return run_id, state.is_file(), state


def create_new_run(
    output_root: Path | str, *, generated_run_id: str | None = None,
) -> tuple[str, Path]:
    """Reserve a unique run name without reusing any existing task directory."""
    output = Path(output_root).expanduser()
    layout = ProjectLayout.from_output(output)
    base = _safe_task_name(generated_run_id or time.strftime("run_%Y%m%d_%H%M%S"))
    for suffix in range(10000):
        run_id = base if suffix == 0 else f"{base}_{suffix:02d}"
        job_root = layout.full_run_root(run_id)
        if layout.existing_full_run_root(run_id) is None and not job_root.exists():
            return run_id, job_root / "job_state.json"
    raise ValueError("无法生成唯一的新任务名称，请稍后重试。")


def active_pipeline_manifest(
    output_root: Path | str, active_task: dict | None,
) -> Path:
    """Resolve the exact active task manifest without merging other runs."""
    active = active_task if isinstance(active_task, dict) else {}
    run_id = str(active.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("当前项目没有 active task，请先继续或创建一个完整任务。")
    layout = ProjectLayout.from_output(Path(output_root).expanduser())
    candidates: list[Path] = []
    current_root = layout.existing_full_run_root(run_id)
    if current_root is not None:
        candidates.extend((current_root / "pipeline_result.json", current_root / "job_state.json"))
    state_value = str(active.get("state") or "").strip()
    if state_value:
        state = Path(state_value).expanduser()
        candidates.extend((state.parent / "pipeline_result.json", state))
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and str(payload.get("run_id") or run_id) == run_id:
            return resolved
    raise ValueError(f"找不到当前 active task 的任务索引：{run_id}")


def task_execution_profile(manifest_path: Path | str) -> str | None:
    """Read the frozen Fast/Full profile from one exact task manifest."""
    try:
        payload = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    spec = payload.get("input_spec") if isinstance(payload.get("input_spec"), dict) else {}
    profile = str(
        payload.get("execution_profile") or spec.get("execution_profile") or "full"
    ).strip().casefold()
    return profile if profile in {"fast", "full"} else None


def project_relocation_preview(
    output_root: Path | str, run_id: str, project_root: Path | str,
) -> dict | None:
    """Build a read-only pre-run summary; the backend remains the only writer."""
    output = Path(output_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    layout = ProjectLayout.from_project(project, output)
    job_root = layout.existing_full_run_root(run_id)
    if job_root is None:
        return None
    state_path = job_root / "job_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取待续跑任务状态：{state_path}：{exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"待续跑任务状态格式错误：{state_path}")
    plan = build_relocation_plan(
        state, state_path, run_id=run_id,
        current_project_root=project, current_output_root=output,
        current_job_root=job_root,
    )
    if plan is None:
        return None
    relocated = plan.relocate_tree(state)
    completed: list[str] = []
    for entry in relocated.get("period_results", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            result_path = Path(str(entry.get("result") or "")).expanduser()
            result = plan.relocate_tree(json.loads(result_path.read_text(encoding="utf-8")))
            ready = all(
                Path(str(result.get(key) or "")).expanduser().is_file()
                for key in ("centerlines", "surfaces", "gpkg")
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            ready = False
        if ready:
            completed.append(f"{entry.get('grid')} / {entry.get('period')}")
    all_periods = {
        f"{path.parents[1].name} / {path.name}"
        for path in job_root.glob("grids/*/periods/*") if path.is_dir()
    }
    incomplete = sorted(all_periods - set(completed))
    legacy_candidates = 0
    for label in incomplete:
        grid, period = (part.strip() for part in label.split("/", 1))
        graph_roots = job_root.glob(
            f"grids/{grid}/periods/{period}/runs/*/inference/road_graphs/**/graph"
        )
        for graph_root in graph_roots:
            for graph in graph_root.glob("*.p"):
                stem = graph.stem
                root = graph_root.parent
                companions = (
                    graph_root / f"{stem}_edge_scores.csv",
                    graph_root / f"{stem}_weak_recovery.json",
                    graph_root / f"{stem}_edge_candidates.csv",
                    root / "mask" / f"{stem}_road.png",
                    root / "mask" / f"{stem}_itsc.png",
                    root / "mask" / f"{stem}_centerline_probability.png",
                    root / "viz" / f"{stem}.png",
                )
                if graph.stat().st_size > 0 and all(
                    path.is_file() and path.stat().st_size > 0 for path in companions
                ):
                    legacy_candidates += 1
    return {
        "old_project_root": str(plan.old_project_root),
        "new_project_root": str(plan.new_project_root),
        "completed_periods": sorted(completed),
        "incomplete_periods": incomplete,
        "legacy_slice_candidates": legacy_candidates,
        "will_reinfer": bool(incomplete),
    }


def project_relocation_message(preview: dict) -> str:
    completed = "、".join(preview.get("completed_periods") or []) or "无"
    incomplete = "、".join(preview.get("incomplete_periods") or []) or "无"
    inference = "会；仅对严格校验失败或尚未处理的切片推理" if preview.get("will_reinfer") else "不会"
    return (
        "检测到任务来自其他目录。\n"
        "程序将把任务状态重定位到当前项目，并只使用当前项目中的成果。\n"
        "迁移完成后不再依赖原目录。\n\n"
        f"旧项目根目录：{preview.get('old_project_root')}\n"
        f"当前项目根目录：{preview.get('new_project_root')}\n"
        f"将复用的完整期次：{completed}\n"
        f"将续跑的未完成期次：{incomplete}\n"
        f"可接纳的旧切片候选：{preview.get('legacy_slice_candidates', 0)}（运行时严格验证）\n"
        "缺失的外部输入：无（当前输入配置已通过检查）\n"
        f"是否会发生重新推理：{inference}"
    )

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
    evaluation_tolerance: str = "5.0", truth_value_map: dict[str, str] | None = None,
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
    for key, option in (
        ("added", "--truth-added-value"),
        ("width_changed", "--truth-width-changed-value"),
        ("removed", "--truth-removed-value"),
    ):
        value = str((truth_value_map or {}).get(key) or "").strip()
        if value:
            args.extend((option, value))
    return args

def build_evaluate_all_command(
    manifest: dict, pipeline_manifest: Path | str,
    truths: list[tuple[str, str, str, str]], *, truth_type_field: str = "BHBM",
    evaluation_tolerance: str = "5.0", truth_value_map: dict[str, str] | None = None,
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
    for key, option in (
        ("added", "--truth-added-value"),
        ("width_changed", "--truth-width-changed-value"),
        ("removed", "--truth-removed-value"),
    ):
        value = str((truth_value_map or {}).get(key) or "").strip()
        if value:
            args.extend((option, value))
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

class TaskManager:
    """Express user task intent without exposing subprocess details to Tk pages."""

    def __init__(self, backend_client) -> None:
        self.backend = backend_client

    @staticmethod
    def build_pipeline(**options) -> list[str]:
        return build_pipeline_command(**options)

    @staticmethod
    def affected_change_pairs(periods, selected):
        return affected_change_pairs(periods, selected)

    @staticmethod
    def unfinished_state(output_root, active_task):
        return unfinished_task_state(output_root, active_task)

    @staticmethod
    def resolve_run(
        output_root, requested_run_id="", active_task=None, *, generated_run_id=None,
    ):
        return resolve_automatic_run(
            output_root, requested_run_id, active_task,
            generated_run_id=generated_run_id,
        )

    @staticmethod
    def create_new_run(output_root, *, generated_run_id=None):
        return create_new_run(output_root, generated_run_id=generated_run_id)

    @staticmethod
    def active_pipeline_manifest(output_root, active_task):
        return active_pipeline_manifest(output_root, active_task)

    @staticmethod
    def task_execution_profile(manifest_path):
        return task_execution_profile(manifest_path)

    @staticmethod
    def relocation_preview(output_root, run_id, project_root):
        return project_relocation_preview(output_root, run_id, project_root)

    @staticmethod
    def relocation_message(preview):
        return project_relocation_message(preview)

    @staticmethod
    def unfinished_message(state):
        return unfinished_task_message(state)

    @staticmethod
    def mark_cancelled(output_root, run_id):
        return mark_task_cancelled(output_root, run_id)

    @staticmethod
    def build_rerun_period(manifest, area_id, period, update_related=False) -> list[str]:
        args = [
            "rerun-period", "--pipeline-manifest", str(manifest),
            "--grid", str(area_id), "--period", str(period),
        ]
        if update_related:
            args.append("--update-related")
        return args

    @staticmethod
    def build_rerun_change(manifest, area_id, before, after, update_temporal=False) -> list[str]:
        args = [
            "rerun-change", "--pipeline-manifest", str(manifest),
            "--grid", str(area_id), "--before-period", str(before),
            "--after-period", str(after),
        ]
        if update_temporal:
            args.append("--update-temporal")
        return args

    @staticmethod
    def build_rerun_all_periods(manifest, continue_on_error=False) -> list[str]:
        args = ["rerun-all-periods", "--pipeline-manifest", str(manifest)]
        if continue_on_error:
            args.append("--continue-on-error")
        return args

    @staticmethod
    def build_rerun_all_changes(manifest, continue_on_error=False) -> list[str]:
        args = ["rerun-all-changes", "--pipeline-manifest", str(manifest)]
        if continue_on_error:
            args.append("--continue-on-error")
        return args

    @staticmethod
    def build_extract_all(
        project_root, run_id, *, output_root, device, pixel_size, rescale,
        junction_node_mode, continue_on_error=False,
    ) -> list[str]:
        args = [
            "extract-project-all", "--project-root", str(project_root),
            "--run-id", str(run_id), "--device", str(device),
            "--pixel-size", str(pixel_size), "--rescale", str(rescale),
            "--junction-node-mode", str(junction_node_mode),
        ]
        state = ProjectLayout.from_output(output_root).batch_task_root(run_id) / "batch_extract_task.json"
        if state.is_file():
            args.append("--resume")
        if continue_on_error:
            args.append("--continue-on-error")
        return args

    @staticmethod
    def build_extract_period(
        project_root, area_id, period, run_id, *, output_root, device,
        pixel_size, rescale, junction_node_mode,
    ) -> list[str]:
        args = [
            "extract-project-period", "--project-root", str(project_root),
            "--area-id", str(area_id), "--period", str(period),
            "--run-id", str(run_id), "--device", str(device),
            "--pixel-size", str(pixel_size), "--rescale", str(rescale),
            "--junction-node-mode", str(junction_node_mode),
        ]
        state = ProjectLayout.from_output(output_root).period_task_root(
            area_id, period, run_id,
        ) / "period_task.json"
        if state.is_file():
            args.append("--resume")
        return args

    @staticmethod
    def build_change_pair(
        project_root, area_id, before, after, run_id, *, output_root,
        manifest=None, area_truths=(), validation_areas=(), absolute="2.0",
        ratio="0.20", tolerance="3.0", truth_type_field="",
    ) -> list[str]:
        if isinstance(manifest, dict):
            periods = {
                (str(entry.get("grid")), str(entry.get("period"))): entry
                for entry in manifest.get("period_results", []) or []
                if isinstance(entry, dict)
            }
            change_entry = next((
                entry for entry in manifest.get("change_results", []) or []
                if isinstance(entry, dict) and str(entry.get("grid")) == str(area_id)
                and str(entry.get("before_period")) == str(before)
                and str(entry.get("after_period")) == str(after)
            ), None)
            before_entry = periods.get((str(area_id), str(before)))
            after_entry = periods.get((str(area_id), str(after)))
            if change_entry and before_entry and after_entry:
                output = str(
                    change_entry.get("output") or
                    Path(str(change_entry.get("gpkg") or "")).parent
                )
                args = [
                    "change", "--before-result", str(before_entry["result"]),
                    "--after-result", str(after_entry["result"]), "--output", output,
                    "--before-period", str(before), "--after-period", str(after),
                    "--absolute", str(absolute), "--ratio", str(ratio),
                    "--tolerance", str(tolerance),
                ]
                truth = next((
                    path for area, one, two, path in area_truths
                    if (str(area), str(one), str(two)) ==
                    (str(area_id), str(before), str(after))
                ), "")
                validation = next((
                    path for area, path in validation_areas
                    if str(area) == str(area_id)
                ), "")
                if truth:
                    args.extend(("--truth", str(truth)))
                if validation:
                    args.extend(("--validation-area", str(validation)))
                if str(truth_type_field).strip():
                    args.extend(("--truth-type-field", str(truth_type_field).strip()))
                return args
        layout = ProjectLayout.from_output(output_root)
        before_state = layout.period_task_root(area_id, before, run_id) / "period_task.json"
        after_state = layout.period_task_root(area_id, after, run_id) / "period_task.json"
        if not before_state.is_file() or not after_state.is_file():
            raise ValueError("所选前后期尚无分步提取状态。请先重跑并完成两个期次。")
        args = [
            "change-project-periods", "--project-root", str(project_root),
            "--area-id", str(area_id), "--before-period", str(before),
            "--after-period", str(after), "--before-state", str(before_state),
            "--after-state", str(after_state), "--run-id", str(run_id),
            "--absolute", str(absolute), "--ratio", str(ratio),
            "--tolerance", str(tolerance),
        ]
        change_state = layout.change_task_root(area_id, before, after, run_id) / "change_task.json"
        if change_state.is_file():
            args.append("--resume")
        return args

    @staticmethod
    def build_evaluate_existing(*args, **kwargs) -> list[str]:
        return build_evaluate_existing_command(*args, **kwargs)

    @staticmethod
    def build_evaluate_all(*args, **kwargs) -> list[str]:
        return build_evaluate_all_command(*args, **kwargs)

    @staticmethod
    def build_apply_edits(*args, **kwargs) -> list[str]:
        return build_apply_edits_command(*args, **kwargs)

    @staticmethod
    def find_period_result(item):
        return find_period_result(item)

    @staticmethod
    def find_saved_edits(item, preferred=""):
        return find_saved_edit_directory(item, preferred)

    @staticmethod
    def manifest_contains_period(manifest_path, result_path) -> bool:
        return manifest_contains_period_result(manifest_path, result_path)

    def submit(self, args: list[str], **kwargs):
        command = args[0] if args else ""
        if command == "all":
            return self.backend.run_full_project(args, **kwargs)
        if command in {"extract-project-period", "rerun-period"}:
            return self.backend.run_period(args, **kwargs)
        if command in {"extract-project-all", "rerun-all-periods"}:
            return self.backend.run_all_periods(args, **kwargs)
        if command in {"change", "change-project-periods", "rerun-change", "rerun-all-changes"}:
            return self.backend.run_change_pair(args, **kwargs)
        if command == "apply-edits":
            return self.backend.apply_edits(args, **kwargs)
        if command in {"evaluate-existing", "evaluate-all-existing"}:
            return self.backend.evaluate(args, **kwargs)
        return self.backend.run_preflight(args, **kwargs)

    def cancel(self) -> None:
        self.backend.cancel()
