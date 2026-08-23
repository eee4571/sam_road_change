from __future__ import annotations

"""Task commands, resume state and dependency calculations."""

import json
import os
import re
import time
from pathlib import Path

from input_catalog import period_order_manifest, period_sort_key
from app.project_manager import USER_IMAGE_LIST_SUFFIX, USER_VECTOR_SUFFIX, atomic_write_json

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
        state = (
            Path(output_root).expanduser() / "batch_extractions" /
            _safe_task_name(run_id) / "batch_extract_task.json"
        )
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
        state = (
            Path(output_root).expanduser() / "period_extractions" /
            _safe_task_name(area_id) / _safe_task_name(period) /
            _safe_task_name(run_id) / "period_task.json"
        )
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
        base = Path(output_root).expanduser() / "period_extractions" / _safe_task_name(area_id)
        before_state = base / _safe_task_name(before) / _safe_task_name(run_id) / "period_task.json"
        after_state = base / _safe_task_name(after) / _safe_task_name(run_id) / "period_task.json"
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
        change_state = (
            Path(output_root).expanduser() / "period_changes" / _safe_task_name(area_id) /
            f"{_safe_task_name(before)}_to_{_safe_task_name(after)}" /
            str(run_id) / "change_task.json"
        )
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
        if command in {"change", "change-project-periods", "rerun-change"}:
            return self.backend.run_change_pair(args, **kwargs)
        if command == "apply-edits":
            return self.backend.apply_edits(args, **kwargs)
        if command in {"evaluate-existing", "evaluate-all-existing"}:
            return self.backend.evaluate(args, **kwargs)
        return self.backend.run_preflight(args, **kwargs)

    def cancel(self) -> None:
        self.backend.cancel()
