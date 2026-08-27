from __future__ import annotations

"""面向用户的无训练批处理后端。

界面只需要调用本文件的三个子命令：prepare、extract、change。
所有路径都可以是绝对路径；输出目录按期次隔离，便于同时处理多个格网任务。
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
from pathlib import Path

from input_catalog import (
    decode_text_auto,
    period_order_manifest,
    period_sort_key,
    read_path_list,
)
from app.result_publisher import (
    LEGACY_RESULT_DIRECTORY_NAME, RESULT_DIRECTORY_NAME,
    ProjectLayout, ResultPublisher, shapefile_has_records,
)
from app.project_relocation import (
    active_old_path_references,
    build_relocation_plan,
    repair_task_batch_lists,
    relocate_state_files,
)
from engine.samroad.image_resume import relocate_task_image_markers
from dependency_identity import (
    dependency_identity_equal,
    effective_config_identity,
    ordinary_file_identity_equal,
    stable_file_identity,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IMAGE_SUFFIXES = {".tif", ".tiff", ".img", ".jp2", ".vrt", ".png", ".jpg", ".jpeg", ".bmp"}
DIRECT_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_REPORTED_PATH_LISTS: set[Path] = set()
PIPELINE_VERSION = "2026.08-user-batch-6-cross-period-existence-evidence"

PERIOD_STAGE_DEFINITIONS = (
    ("centerline", "道路提取"),
    ("surface", "道路面提取"),
    ("width", "道路宽度计算"),
    ("finalize", "结果固化"),
    ("export", "道路产品导出"),
)
FAST_PERIOD_STAGE_DEFINITIONS = (
    ("centerline", "道路概率推理"),
    ("surface", "Final Fast Mask"),
    ("width", "快速道路宽度"),
    ("export", "道路产品导出"),
)
ATOMIC_REPLACE_ATTEMPTS = 12
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.05
ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.5


def project_root() -> Path:
    """Return the source-code root containing this backend module."""
    return Path(__file__).resolve().parent


ROOT = project_root()
WORKSPACE_ROOT = ROOT.parent
RUNTIME_ROOT = WORKSPACE_ROOT / "runtime"
SAMROAD = ROOT / "engine" / "samroad"
MOLRA = ROOT / "engine" / "sam_molra"
WIDTH = ROOT / "engine" / "width"
MODELS = Path(os.environ.get("SAMROAD_MODELS_ROOT", RUNTIME_ROOT / "models"))
PYTHON = Path(sys.executable)
for candidate in (
    RUNTIME_ROOT / "env" / "samroad_env" / "python.exe",
    RUNTIME_ROOT / "python.exe",
):
    if candidate.is_file():
        PYTHON = candidate
        break


def emit(kind: str, **payload) -> None:
    payload["kind"] = kind
    print("__SAMROAD_USER__" + json.dumps(payload, ensure_ascii=False), flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_with_retry(source: Path, target: Path) -> None:
    """Replace a file atomically while tolerating brief Windows reader locks."""
    delay = ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(ATOMIC_REPLACE_MAX_DELAY_SECONDS, delay * 1.8)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def elapsed_seconds(started: float) -> float:
    return round(max(0.0, time.monotonic() - started), 3)


PREVIEW_KEYS = ("centerline", "surface", "fusion", "width")


def _first_existing_path(root: Path, patterns: tuple[str, ...]) -> Path | None:
    """Return the first deterministic matching file without creating anything."""
    root = Path(root).expanduser()
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if matches:
            return matches[0].resolve()
    return None


def discover_preview_paths(run_root: Path) -> dict[str, str | None]:
    """Discover user-facing previews using the stable run directory layout.

    Values are absolute paths for existing files and ``None`` when a stage has
    not produced a preview yet.  The helper is intentionally read-only so a
    GUI can consume the manifest without reproducing directory heuristics.
    """
    root = Path(run_root).expanduser()
    paths = {
        "centerline": _first_existing_path(root, ("inference/road_graphs/**/viz/*.png",)),
        "surface": _first_existing_path(
            root,
            ("surfaces/**/*_mask.png", "width_review/*_molra_clean_mask.png"),
        ),
        "fusion": _first_existing_path(
            root,
            ("products/road_overview.png",),
        ),
        "width": _first_existing_path(
            root,
            ("products/road_width_overview.png",),
        ),
    }
    return {key: str(path) if path is not None else None for key, path in paths.items()}


def discover_change_preview(output: Path) -> str | None:
    """Return ``change_preview.png`` only when the change stage created it."""
    return discover_change_previews(output)["change"]


def discover_change_previews(output: Path) -> dict[str, str | None]:
    """Discover separate formal and review previews with legacy-safe keys."""
    root = Path(output).expanduser()
    paths = {
        "change": root / "change_preview.png",
        "review_change": (
            root / "review_preview.png"
            if shapefile_has_records(root / "review_changes.shp") else None
        ),
    }
    return {
        key: str(path.resolve()) if path is not None and path.is_file() else None
        for key, path in paths.items()
    }


def _manual_review_item_count(review_dir: Path) -> int:
    total = 0
    if not review_dir.is_dir():
        return total
    for path in sorted(review_dir.glob("*_summary.json")):
        try:
            value = read_json(path).get("manual_review_item_count", 0)
            count = int(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        total += max(count, 0)
    return total


def build_review_metadata(run_root: Path, review_dir: Path | None = None) -> dict:
    """Build optional review metadata without making review a pipeline gate."""
    root = Path(run_root).expanduser()
    directory = Path(review_dir).expanduser() if review_dir is not None else root / "width_review"
    decisions = directory / "review_decisions.csv"
    return {
        "available": directory.is_dir(),
        "directory": str(directory.resolve()),
        "decisions": str(decisions.resolve()),
        "edited_directory": str((root / "centerline_edit").resolve()),
        "manual_item_count": _manual_review_item_count(directory),
    }


def build_fusion_metadata(final_dir: Path) -> dict:
    """Summarize what automatic/manual fusion actually changed."""
    totals = {
        "original_edge_count": 0,
        "optimized_edge_count": 0,
        "auto_gap_count": 0,
        "local_gap_count": 0,
        "global_gap_count": 0,
        "global_endpoint_gap_count": 0,
        "global_edge_attachment_count": 0,
        "auto_surface_count": 0,
        "geometry_edited_tile_count": 0,
    }
    for path in sorted(Path(final_dir).glob("*_optimized_summary.json")):
        try:
            summary = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        totals["original_edge_count"] += int(summary.get("original_edge_count", 0) or 0)
        totals["optimized_edge_count"] += int(summary.get("optimized_edge_count", 0) or 0)
        local_gap_count = int(summary.get("auto_accepted_gap_count", 0) or 0)
        totals["local_gap_count"] += local_gap_count
        totals["auto_gap_count"] += local_gap_count
        totals["auto_surface_count"] += int(summary.get("auto_accepted_surface_count", 0) or 0)
        totals["geometry_edited_tile_count"] += int(bool(summary.get("geometry_edited")))
    quality_report_path = Path(final_dir).parent / "products" / "final_quality_report.json"
    if quality_report_path.is_file():
        try:
            fusion = read_json(quality_report_path).get("fusion", {})
            totals["global_gap_count"] = int(fusion.get("global_surface_gap_count", 0) or 0)
            totals["global_endpoint_gap_count"] = int(fusion.get("global_endpoint_gap_count", 0) or 0)
            totals["global_edge_attachment_count"] = int(fusion.get("global_edge_attachment_count", 0) or 0)
            totals["auto_gap_count"] += totals["global_gap_count"]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    totals["added_edge_count"] = max(
        0, totals["optimized_edge_count"] - totals["original_edge_count"]
    )
    return totals


def _empty_review_metadata() -> dict:
    return {
        "available": False,
        "directory": "width_review",
        "decisions": str(Path("width_review") / "review_decisions.csv"),
        "edited_directory": "centerline_edit",
        "manual_item_count": 0,
    }


def _ensure_extract_manifest_fields(result: dict | None) -> dict:
    """Normalize real and legacy mocked extract results to the manifest contract."""
    payload = dict(result or {})
    run_root_value = payload.get("run_root")
    run_root = Path(run_root_value).expanduser() if run_root_value else None
    discovered = discover_preview_paths(run_root) if run_root is not None else {key: None for key in PREVIEW_KEYS}
    existing_previews = payload.get("previews") if isinstance(payload.get("previews"), dict) else {}
    payload["previews"] = {
        key: (
            discovered[key] or existing_previews.get(key)
            if key in {"fusion", "width"}
            else existing_previews.get(key, discovered[key])
        )
        for key in PREVIEW_KEYS
    }
    if isinstance(payload.get("review"), dict):
        review = dict(payload["review"])
        defaults = build_review_metadata(run_root) if run_root is not None else _empty_review_metadata()
        for key, value in defaults.items():
            review.setdefault(key, value)
        payload["review"] = review
    else:
        payload["review"] = build_review_metadata(run_root) if run_root is not None else _empty_review_metadata()
    products = run_root / "products" if run_root is not None else None
    for key, name in (
        ("width_segments", "road_width_segments.shp"),
        ("corridors", "road_corridors.shp"),
        ("valid_observation", "valid_observation.shp"),
        ("road_probability", "road_probability.tif"),
    ):
        if not payload.get(key) and products is not None and (products / name).is_file():
            payload[key] = str((products / name).resolve())
    inference_metadata = (
        run_root / "inference" / "road_graphs" / "inference_metadata.json"
        if run_root is not None else None
    )
    if inference_metadata is not None and inference_metadata.is_file():
        metadata = read_json(inference_metadata)
        for key in (
            "relative_roadness_enabled",
            "relative_injected_into_toponet",
            "relative_centerline_method",
            "regularized_skeleton_active",
            "continuous_tracing_active",
            "junction_collapse_active",
            "endpoint_segment_recovery_active",
        ):
            if key in metadata:
                payload[key] = metadata[key]
        payload["inference_metadata"] = str(inference_metadata.resolve())
    return payload


def _ensure_change_manifest_fields(result: dict | None, output: Path | None = None) -> dict:
    payload = dict(result or {})
    target = output or (Path(payload["output"]).expanduser() if payload.get("output") else None)
    previews = dict(payload.get("previews")) if isinstance(payload.get("previews"), dict) else {}
    if target is not None:
        for key, preview in discover_change_previews(target).items():
            if preview is not None:
                previews[key] = preview
            else:
                previews.pop(key, None)
    payload["previews"] = previews
    if target is not None:
        fast_profile = str(payload.get("execution_profile") or "").casefold() == "fast"
        layers = {}
        if not fast_profile and isinstance(payload.get("layers"), dict):
            layers = dict(payload["layers"])
        layer_names = (
            (("changes", "road_changes.shp"),)
            if fast_profile else
            (
                ("changes", "road_changes.shp"), ("review", "review_changes.shp"),
                ("added", "added_roads.shp"), ("removed", "removed_roads.shp"),
                ("widened", "widened_road_parts.shp"),
                ("narrowed", "narrowed_road_parts.shp"),
                ("width_changed", "width_changed_road_parts.shp"),
                ("width_segments", "road_width_segments.shp"),
                ("corridors", "road_corridors.shp"),
                ("matches", "road_matches.shp"),
                ("canonical_roads", "canonical_roads.shp"),
            )
        )
        for key, name in layer_names:
            path = target / name
            if path.is_file():
                layers[key] = str(path.resolve())
        payload["layers"] = layers
        if fast_profile:
            payload.pop("gpkg", None)
    return payload


def _run_fast_change_result(
    before_result: Path,
    after_result: Path,
    output: Path,
    *,
    before_period: str,
    after_period: str,
    position_tolerance: float,
    width_change_absolute: float,
    width_change_ratio: float,
    truth_path: Path | None = None,
    validation_area: Path | None = None,
    truth_type_field: str = "BHBM",
    evaluation_tolerance: float = 5.0,
) -> dict:
    """Run the sole Fast auto detector, then optionally publish Auto∪GT."""
    from engine.fast_pipeline import (
        augment_fast_changes_with_truth,
        detect_fast_changes,
    )

    automatic_output = output / "_automatic" if truth_path is not None else output
    automatic = detect_fast_changes(
        before_result,
        after_result,
        automatic_output,
        before_period=before_period,
        after_period=after_period,
        position_tolerance=float(position_tolerance),
        width_change_absolute=float(width_change_absolute),
        width_change_ratio=float(width_change_ratio),
        internal_outputs=truth_path is not None,
    )
    if truth_path is None:
        return automatic
    truth_path = Path(truth_path).expanduser()
    if not truth_path.is_file():
        raise FileNotFoundError(f"Fast 变化真值不存在：{truth_path}")
    return augment_fast_changes_with_truth(
        automatic,
        truth_path,
        output,
        before_period=before_period,
        after_period=after_period,
        truth_type_field=truth_type_field,
        validation_area=validation_area,
        position_tolerance=float(position_tolerance),
        evaluation_tolerance=float(evaluation_tolerance),
    )


def clean_name(value: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return value.strip("._-") or "workspace"


def project_layout(output_root: Path | str, project_root_value: Path | str | None = None) -> ProjectLayout:
    """Resolve all user/output/work paths through one project-level policy."""
    return (
        ProjectLayout.from_project(project_root_value, output_root)
        if project_root_value is not None else ProjectLayout.from_output(output_root)
    )


def _resolve_full_run_job_root(
    layout: ProjectLayout, run_id: str, *, resume: bool,
) -> Path:
    """Choose a full-run workspace without creating a directory before resume validation."""
    current = layout.full_run_root(run_id)
    if not resume:
        return current
    existing = layout.existing_full_run_root(run_id)
    if existing is not None:
        return existing
    legacy = layout.legacy_full_run_root(run_id)
    raise FileNotFoundError(
        "找不到可续跑的任务状态。已依次检查：\n"
        f"- {current / 'job_state.json'}\n"
        f"- {legacy / 'job_state.json'}"
    )


def _read_full_run_resume_state(
    state_path: Path, run_id: str, job_root: Path, *,
    layout: ProjectLayout | None = None, output_root: Path | None = None,
) -> dict:
    """Read and validate the selected state before any run directory is created."""
    try:
        prior = read_json(state_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"任务状态 JSON 损坏或无法读取：{state_path}：{exc}") from exc
    if not isinstance(prior, dict):
        raise ValueError(f"任务状态 JSON 根节点必须是对象：{state_path}")
    state_run_id = str(prior.get("run_id") or "").strip()
    if state_run_id != run_id:
        raise ValueError(
            f"任务状态 run_id 不一致：请求 {run_id}，状态记录 {state_run_id or '（缺失）'}：{state_path}"
        )
    recorded_root_value = str(prior.get("job_root") or "").strip()
    if recorded_root_value:
        recorded_root = Path(recorded_root_value).expanduser()
        if not recorded_root.is_absolute():
            recorded_root = state_path.parent / recorded_root
        if recorded_root.resolve() != job_root.resolve():
            if layout is None or output_root is None:
                raise ValueError(
                    "任务状态中的 job_root 与找到的任务目录不一致，且没有当前项目上下文，"
                    "不能安全重定位或从头运行："
                    f"记录为 {recorded_root.resolve()}，实际为 {job_root.resolve()}"
                )
            plan = build_relocation_plan(
                prior, state_path, run_id=run_id,
                current_project_root=layout.project_root,
                current_output_root=output_root,
                current_job_root=job_root,
            )
            if plan is None:
                raise ValueError(
                    "任务状态路径不一致，但无法建立经过边界校验的项目重定位方案："
                    f"记录为 {recorded_root.resolve()}，实际为 {job_root.resolve()}"
                )
            prior, written = relocate_state_files(prior, plan)
            unresolved = active_old_path_references(
                prior, (plan.old_project_root, plan.old_output_root, plan.old_job_root),
            )
            if unresolved:
                raise ValueError(
                    "项目重定位后仍有活动状态引用旧项目路径，已停止续跑：\n"
                    + "\n".join(f"- {value}" for value in unresolved[:20])
                )
            print(
                f"[续跑] 已将复制项目的任务状态重定位到当前目录；"
                f"备份并更新 {len(written)} 个 JSON 文件。",
                flush=True,
            )
    return prior


def natural_key(value: str) -> tuple:
    """Sort validated calendar periods first and custom names naturally."""
    return period_sort_key(value)


def read_text_auto(path: Path) -> str:
    """Backward-compatible wrapper around the shared path-list decoder."""
    return decode_text_auto(path)[0]


def listed_rasters(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".txt":
        parsed = read_path_list(path)
        if parsed.source not in _REPORTED_PATH_LISTS:
            _REPORTED_PATH_LISTS.add(parsed.source)
            print("__SAMROAD_USER__" + json.dumps({
                "kind": "input-list", "stage": "数据检查", "status": "complete",
                "path": str(parsed.source), "encoding": parsed.encoding,
                "image_count": len(parsed.entries),
            }, ensure_ascii=False), flush=True)
        return [entry.path for entry in parsed.entries]
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        return sorted(item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    return []


def discover_grid_periods(source_root: Path) -> dict[str, dict[str, Path]]:
    """Discover <grid>/<period>.txt and two convenient raster variants."""
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"格网根目录不存在：{source_root}")
    grids: dict[str, dict[str, Path]] = {}
    ignored = {"province_shp", "shp", "vector", "vectors"}
    for grid_dir in sorted(source_root.iterdir(), key=lambda item: natural_key(item.name)):
        if not grid_dir.is_dir() or grid_dir.name.startswith((".", "_")) or grid_dir.name.casefold() in ignored:
            continue
        periods: dict[str, Path] = {}
        # The established GIS layout takes priority: area_01/2021.txt.
        for item in sorted(grid_dir.glob("*.txt"), key=lambda path: natural_key(path.stem)):
            if listed_rasters(item):
                periods[item.stem] = item.resolve()
        # Also accept area_01/2021.tif.
        for item in sorted(grid_dir.iterdir(), key=lambda path: natural_key(path.name)):
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
                periods.setdefault(item.stem, item.resolve())
        # And area_01/2021/*.tif.
        for item in sorted(grid_dir.iterdir(), key=lambda path: natural_key(path.name)):
            if item.is_dir() and listed_rasters(item):
                periods.setdefault(item.name, item.resolve())
        if periods:
            grids[grid_dir.name] = dict(sorted(periods.items(), key=lambda pair: natural_key(pair[0])))
    if not grids:
        raise RuntimeError(
            "未发现格网期次数据。推荐结构为：格网根目录/格网编号/期次.txt，"
            "TXT 每行填写一张影像路径。"
        )
    invalid = [name for name, periods in grids.items() if len(periods) < 2]
    if invalid:
        preview = "、".join(invalid[:10])
        raise RuntimeError(f"以下格网少于两个期次，无法检测变化：{preview}")
    return grids


def _read_validation_area(path: Path):
    """Load one usable polygon validation area with explicit GIS errors."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("影像覆盖预检需要本项目内的 geopandas 环境") from exc
    if not path.is_file():
        raise FileNotFoundError(f"找不到验证区矢量：{path}")
    area = gpd.read_file(path)
    if area.crs is None:
        raise ValueError(f"验证区缺少 CRS：{path}")
    area = area.loc[area.geometry.notna() & ~area.geometry.is_empty].copy()
    if area.empty:
        raise ValueError(f"验证区没有有效几何：{path}")
    invalid = [geometry.geom_type for geometry in area.geometry if geometry.geom_type not in {"Polygon", "MultiPolygon"}]
    if invalid:
        raise ValueError(f"验证区必须仅包含面要素：{path}")
    return area


def _require_shapefile_components(path: Path, label: str) -> None:
    """Reject incomplete user Shapefiles before a long task starts."""
    source = Path(path).expanduser().resolve()
    missing = [str(source.with_suffix(suffix)) for suffix in (".shp", ".shx", ".dbf", ".prj") if not source.with_suffix(suffix).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} Shapefile 附件不完整：\n" + "\n".join(missing))


def _validate_truth_shapefile(path: Path) -> None:
    _require_shapefile_components(path, "变化真值")
    import geopandas as gpd
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"变化真值缺少 CRS：{path}")
    usable = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty]
    if usable.empty:
        return
    invalid = sorted({geometry.geom_type for geometry in usable.geometry if geometry.geom_type not in {"LineString", "MultiLineString", "Polygon", "MultiPolygon"}})
    if invalid:
        raise ValueError(f"变化真值必须是线或面：{path}（发现 {', '.join(invalid)}）")


def _valid_raster_footprint(path: Path, target_crs=None):
    """Return raster bounds after a minimal readability/CRS check.

    The legacy implementation scanned every GDAL block and polygonized all
    valid pixels.  Window normalization now computes the real per-pixel mask,
    so preflight deliberately avoids that duplicate network-drive I/O.
    """
    try:
        from pyproj import Transformer
        from pyproj.exceptions import ProjError
        import rasterio
        from shapely.geometry import box
    except ImportError as exc:
        raise RuntimeError("影像覆盖预检需要本项目内的 rasterio/shapely 环境") from exc
    try:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"影像缺少 CRS：{path}")
            if target_crs is not None:
                try:
                    Transformer.from_crs(dataset.crs, target_crs, always_xy=True)
                except ProjError as exc:
                    raise ValueError(
                        "无法建立影像到验证区的坐标转换。"
                        f"\n影像：{path}\n影像 CRS：{dataset.crs}\n验证区 CRS：{target_crs}"
                        "\n请确认影像和验证区的坐标系定义正确，并检查离线环境的 PROJ 数据库。"
                    ) from exc
            if dataset.width <= 0 or dataset.height <= 0 or dataset.count <= 0:
                raise ValueError(f"影像尺寸或波段数无效：{path}")
            dataset.read(
                1,
                window=((0, min(1, dataset.height)), (0, min(1, dataset.width))),
            )
            return box(*dataset.bounds), dataset.crs
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"无法读取影像以完成覆盖预检：{path}: {exc}") from exc


def validate_validation_inputs(period_entries, validation_area: str | Path, *, minimum_periods: int = 2) -> dict[str, Path]:
    """Validate repeated ``--period PERIOD PATH`` inputs before any model starts.

    Each user-facing period uses one TXT path list.  The listed rasters' valid-data
    footprints may differ, but their union must cover the complete validation
    polygon; raster bounds alone are intentionally insufficient because NoData
    holes are not valid imagery.
    """
    area_path = Path(validation_area).expanduser().resolve()
    if area_path.suffix.lower() != ".shp":
        raise ValueError(f"验证区必须使用 SHP 文件（.shp）：{area_path}")
    area = _read_validation_area(area_path)
    periods: dict[str, Path] = {}
    for entry in period_entries or []:
        if len(entry) != 2:
            raise ValueError("--period 必须重复使用为：--period 期次 影像文件或目录")
        name, source_value = str(entry[0]).strip(), str(entry[1]).strip()
        if not name:
            raise ValueError("--period 的期次名称不能为空")
        if name in periods:
            raise ValueError(f"期次重复：{name}")
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到期次 {name} 的影像路径 TXT：{source}")
        if source.suffix.lower() != ".txt":
            raise ValueError(f"期次 {name} 的影像必须使用内含影像路径的 TXT 文件：{source}")
        rasters = listed_rasters(source)
        if not rasters:
            raise RuntimeError(f"期次 {name} 未发现有效影像文件：{source}")
        raster_errors: list[str] = []
        for raster in rasters:
            try:
                _valid_raster_footprint(raster, target_crs=area.crs)
            except Exception as exc:
                raster_errors.append(f"{raster}：{type(exc).__name__}: {exc}")
        if raster_errors:
            raise RuntimeError(
                f"期次 {name} 有 {len(raster_errors)} 张影像无法读取（TXT：{source}）：\n"
                + "\n".join(raster_errors[:20])
            )
        periods[name] = source
    if len(periods) < minimum_periods:
        if minimum_periods == 1:
            raise ValueError("至少需要一个 --period 期次")
        raise ValueError("验证模式至少需要两个 --period 期次")
    return dict(sorted(periods.items(), key=lambda pair: period_sort_key(pair[0])))


def validation_truths(
    truth_entries,
    periods: dict[str, Path],
    *,
    required: bool = True,
) -> dict[tuple[str, str], Path]:
    """Normalize adjacent truth vectors, optionally allowing production-only runs."""
    truths: dict[tuple[str, str], Path] = {}
    for entry in truth_entries or []:
        if len(entry) != 3:
            raise ValueError("--truth 必须重复使用为：--truth 前期 后期 真值矢量")
        before, after, source_value = (str(value).strip() for value in entry)
        key = (before, after)
        if key in truths:
            raise ValueError(f"变化对真值重复：{before} -> {after}")
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到变化对 {before} -> {after} 的真值：{source}")
        if source.suffix.lower() != ".shp":
            raise ValueError(f"变化对 {before} -> {after} 的真值必须使用 SHP 文件（.shp）：{source}")
        truths[key] = source
    expected = set(zip(periods, list(periods)[1:]))
    unexpected = set(truths) - expected
    missing = expected - set(truths)
    if unexpected:
        pairs = "、".join(f"{before}->{after}" for before, after in sorted(unexpected, key=lambda pair: (natural_key(pair[0]), natural_key(pair[1]))))
        raise ValueError(f"--truth 只能指定相邻期次：{pairs}")
    if required and missing:
        pairs = "、".join(f"{before}->{after}" for before, after in sorted(missing, key=lambda pair: (natural_key(pair[0]), natural_key(pair[1]))))
        raise ValueError(f"缺少相邻变化对真值：{pairs}")
    return truths


def validation_batch_inputs(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, Path]], dict[str, str], dict[tuple[str, str, str], Path]
]:
    """Normalize legacy single-area and new named multi-area CLI inputs."""
    raw_areas = getattr(args, "validation_area", [])
    if isinstance(raw_areas, (str, Path)):
        area_entries = [["validation", str(raw_areas)]]
    else:
        area_entries = list(raw_areas or [])
    areas: dict[str, str] = {}
    for entry in area_entries:
        values = [str(value).strip() for value in (entry if isinstance(entry, (list, tuple)) else [entry])]
        if len(values) == 1:
            area_id, source_value = "validation", values[0]
        elif len(values) == 2:
            area_id, source_value = values
        else:
            raise ValueError("--validation-area 使用：区域名 验证区.shp；旧版单路径仍兼容")
        if not area_id or area_id in areas:
            raise ValueError(f"验证区名称为空或重复：{area_id}")
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".shp":
            raise FileNotFoundError(f"找不到验证区 SHP：{source}")
        areas[area_id] = str(source)
    if not areas:
        raise ValueError("验证模式必须至少提供一个 --validation-area 区域名 验证区.shp")

    period_entries: dict[str, list[list[str]]] = {area_id: [] for area_id in areas}
    for entry in getattr(args, "period", []) or []:
        values = [str(value).strip() for value in entry]
        if len(values) == 2:
            targets, period, source = list(areas), values[0], values[1]
        elif len(values) == 3:
            area_id, period, source = values
            if area_id not in areas:
                raise ValueError(f"期次引用了未知验证区：{area_id}")
            targets = [area_id]
        else:
            raise ValueError("--period 使用：区域名 期次 影像TXT；两参数形式会应用于全部验证区")
        for area_id in targets:
            period_entries[area_id].append([period, source])
    grids = {
        area_id: validate_validation_inputs(entries, areas[area_id])
        for area_id, entries in period_entries.items()
    }

    truth_by_task: dict[tuple[str, str, str], Path] = {}
    for entry in getattr(args, "truth", []) or []:
        values = [str(value).strip() for value in entry]
        if len(values) == 3:
            if len(areas) != 1:
                raise ValueError("多验证区真值必须使用：--truth 区域名 前期 后期 真值SHP")
            area_id, before, after, source_value = next(iter(areas)), *values
        elif len(values) == 4:
            area_id, before, after, source_value = values
        else:
            raise ValueError("--truth 使用：区域名 前期 后期 真值SHP")
        if area_id not in areas:
            raise ValueError(f"真值引用了未知验证区：{area_id}")
        key = (area_id, before, after)
        if key in truth_by_task:
            raise ValueError(f"变化真值重复：{area_id} / {before} -> {after}")
        source = Path(source_value).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".shp":
            raise FileNotFoundError(f"找不到变化真值 SHP：{source}")
        truth_by_task[key] = source
    expected = {
        (area_id, before, after)
        for area_id, periods in grids.items()
        for before, after in zip(periods, list(periods)[1:])
    }
    unexpected = set(truth_by_task) - expected
    if unexpected:
        raise ValueError("真值只能对应各验证区的相邻期次：" + "、".join("/".join(item) for item in sorted(unexpected)))
    return grids, areas, truth_by_task


def discover_validation_project(project_dir: str | Path) -> dict:
    """Discover the documented project-folder layout without importing any UI code."""
    root = Path(project_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"项目文件夹不存在：{root}")

    def named_directory(parent: Path, prefix: str, fallbacks: tuple[str, ...]) -> Path | None:
        candidates = [
            child for child in parent.iterdir()
            if child.is_dir() and (child.name.startswith(prefix) or child.name.casefold() in fallbacks)
        ]
        return sorted(candidates, key=lambda path: natural_key(path.name))[0] if candidates else None

    def discover_area(area_root: Path, area_id: str) -> tuple[str, list[list[str]], list[list[str]]]:
        boundary_dir = named_directory(area_root, "01_", ("validation", "validation_area", "验证区"))
        imagery_dir = named_directory(area_root, "02_", ("images", "imagery", "影像"))
        if boundary_dir is None or imagery_dir is None:
            raise ValueError(f"区域文件夹必须包含 01_验证区 和 02_影像：{area_root}")
        boundaries = sorted(boundary_dir.glob("*.shp"), key=lambda path: natural_key(path.name))
        if len(boundaries) != 1:
            raise ValueError(f"每个区域的 01_验证区 必须且只能有一个 SHP：{boundary_dir}")
        periods = [
            [area_id, path.stem, str(path)]
            for path in sorted(imagery_dir.iterdir(), key=lambda item: natural_key(item.name))
            if path.is_file() and path.suffix.lower() == ".txt"
        ]
        if len(periods) < 2:
            raise ValueError(f"每个区域的 02_影像 至少需要两个期次 TXT：{imagery_dir}")
        period_names = [row[1] for row in periods]
        if len(period_names) != len(set(period_names)):
            raise ValueError(f"区域内期次名称重复：{area_root}")
        truth_dir = named_directory(area_root, "03_", ("truth", "ground_truth", "变化真值"))
        truths = []
        if truth_dir is not None:
            adjacent = set(zip(period_names, period_names[1:]))
            for path in sorted(truth_dir.glob("*.shp"), key=lambda item: natural_key(item.name)):
                match = re.match(r"^(.+?)_to_(.+)$", path.stem, flags=re.IGNORECASE)
                if match and (match.group(1), match.group(2)) in adjacent:
                    truths.append([area_id, match.group(1), match.group(2), str(path)])
        return str(boundaries[0]), periods, truths

    flat_boundary = named_directory(root, "01_", ("validation", "validation_area", "验证区"))
    flat_imagery = named_directory(root, "02_", ("images", "imagery", "影像"))
    area_roots: list[tuple[str, Path]] = []
    if flat_boundary is not None and flat_imagery is not None:
        area_roots = [(root.name or "validation", root)]
    else:
        for child in sorted(root.iterdir(), key=lambda path: natural_key(path.name)):
            if not child.is_dir():
                continue
            if named_directory(child, "01_", ("validation", "validation_area", "验证区")) and named_directory(child, "02_", ("images", "imagery", "影像")):
                area_roots.append((child.name, child))
    if not area_roots:
        raise ValueError("未找到规范项目结构；项目根目录或区域子目录必须包含 01_验证区 和 02_影像")

    validation_areas, periods, truths = [], [], []
    area_summaries = []
    for area_id, area_root in area_roots:
        boundary, area_periods, area_truths = discover_area(area_root, area_id)
        validation_areas.append([area_id, boundary])
        periods.extend(area_periods)
        truths.extend(area_truths)
        area_summaries.append({
            "area_id": area_id, "root": str(area_root), "validation_area": boundary,
            "periods": [{"period": row[1], "source": row[2]} for row in area_periods],
            "truths": [{"before": row[1], "after": row[2], "source": row[3]} for row in area_truths],
        })
    output_dir = (
        root / RESULT_DIRECTORY_NAME
        if (root / RESULT_DIRECTORY_NAME).is_dir()
        else root / LEGACY_RESULT_DIRECTORY_NAME
        if (root / LEGACY_RESULT_DIRECTORY_NAME).is_dir()
        else root / RESULT_DIRECTORY_NAME
    )
    return {
        "project_root": str(root), "project_name": root.name, "output_root": str(output_dir),
        "validation_areas": validation_areas, "periods": periods, "truths": truths, "areas": area_summaries,
    }


def preflight_project(args: argparse.Namespace) -> dict:
    """Discover and inspect a documented project folder; never create outputs or run models."""
    discovered = discover_validation_project(args.project_root)
    validation_args = argparse.Namespace(
        validation_area=discovered["validation_areas"], period=discovered["periods"], truth=discovered["truths"],
    )
    grids, validation_areas, truth_by_task = validation_batch_inputs(validation_args)
    report = build_preflight_report(
        "validation", grids, validation_areas, truth_by_task, Path(discovered["output_root"]),
    )
    report["project"] = discovered
    report["completed"] = 1
    report["total"] = 1
    emit("complete", stage="preflight", **report)
    return report


def _scene_path(geometry, bounds: tuple[float, float, float, float], width: int, height: int) -> str:
    min_x, min_y, max_x, max_y = bounds
    scale_x = width / max(max_x - min_x, 1e-9)
    scale_y = height / max(max_y - min_y, 1e-9)

    def point(value) -> str:
        return f"{(value[0] - min_x) * scale_x:.2f},{height - (value[1] - min_y) * scale_y:.2f}"

    def line(coords, close: bool = False) -> str:
        values = list(coords)
        if not values:
            return ""
        return "M" + " L".join(point(value) for value in values) + (" Z" if close else "")

    kind = geometry.geom_type
    if kind == "LineString":
        return line(geometry.coords)
    if kind == "Polygon":
        return " ".join([line(geometry.exterior.coords, True), *(line(ring.coords, True) for ring in geometry.interiors)])
    if kind.startswith("Multi") or kind == "GeometryCollection":
        return " ".join(_scene_path(part, bounds, width, height) for part in geometry.geoms)
    if kind == "Point":
        x, y = point(geometry.coords[0]).split(",")
        return f"M{x},{y} m-2,0 a2,2 0 1,0 4,0 a2,2 0 1,0 -4,0"
    return ""


def _scene_state_result(state_value: str, project_root: Path, expected_name: str) -> dict:
    if not state_value:
        return {}
    state_path = Path(state_value).expanduser().resolve()
    try:
        state_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"地图状态文件不在当前项目内：{state_path}") from exc
    compatible_names = {
        "period_task.json": {"period_task.json", "latest_result.json"},
        "change_task.json": {"change_task.json", "change_summary.json"},
    }.get(expected_name, {expected_name})
    if state_path.name not in compatible_names or not state_path.is_file():
        raise FileNotFoundError(f"地图状态文件无效：{state_path}")
    state = read_json(state_path)
    if state_path.name == "latest_result.json":
        return state
    if state_path.name == "change_summary.json":
        return {
            "summary": str(state_path),
            "layers": {
                "changes": str(state_path.parent / "road_changes.shp"),
            },
        }
    if state.get("status") != "completed":
        return {}
    return state.get("result_manifest") if isinstance(state.get("result_manifest"), dict) else {}


def map_scene(args: argparse.Namespace) -> dict:
    """Build a bounded browser scene from real raster/vector sources without model execution."""
    import numpy as np
    import geopandas as gpd
    import rasterio
    from PIL import Image
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling, reproject, transform_bounds

    discovered = discover_validation_project(args.project_root)
    project_root_path = Path(discovered["project_root"]).resolve()
    area = next((item for item in discovered["areas"] if item["area_id"] == args.area_id), None)
    if area is None:
        raise ValueError(f"项目中不存在区域：{args.area_id}")
    period_entry = next((item for item in area["periods"] if item["period"] == args.period), None)
    if period_entry is None:
        raise ValueError(f"区域中不存在期次：{args.period}")
    rasters = listed_rasters(Path(period_entry["source"]))
    if not rasters:
        raise FileNotFoundError(f"期次没有可显示的栅格：{period_entry['source']}")

    with rasterio.open(rasters[0]) as first:
        if first.crs is None:
            raise ValueError(f"影像缺少 CRS：{rasters[0]}")
        target_crs = first.crs
    validation = gpd.read_file(area["validation_area"]).to_crs(target_crs)
    min_x, min_y, max_x, max_y = (float(value) for value in validation.total_bounds)
    span_x, span_y = max_x - min_x, max_y - min_y
    padding = max(span_x, span_y) * 0.025
    bounds = (min_x - padding, min_y - padding, max_x + padding, max_y + padding)
    max_width, max_height = max(200, min(int(args.width), 1400)), max(150, min(int(args.height), 1000))
    ratio = max((bounds[2] - bounds[0]) / max(bounds[3] - bounds[1], 1e-9), 1e-3)
    width, height = (max_width, max(150, round(max_width / ratio))) if ratio >= max_width / max_height else (max(200, round(max_height * ratio)), max_height)
    transform = from_bounds(*bounds, width, height)
    rgb = np.zeros((3, height, width), dtype=np.uint8)
    covered = np.zeros((height, width), dtype=bool)
    source_count = 0
    for raster_path in rasters:
        with rasterio.open(raster_path) as source:
            if source.crs is None:
                continue
            source_bounds = transform_bounds(source.crs, target_crs, *source.bounds, densify_pts=8)
            if source_bounds[2] <= bounds[0] or source_bounds[0] >= bounds[2] or source_bounds[3] <= bounds[1] or source_bounds[1] >= bounds[3]:
                continue
            with WarpedVRT(source, crs=target_crs) as warped:
                indexes = [1, min(2, warped.count), min(3, warped.count)]
                tile = np.full((3, height, width), np.nan, dtype=np.float32)
                mask = np.zeros((height, width), dtype=np.uint8)
                for band, source_index in enumerate(indexes):
                    reproject(rasterio.band(warped, source_index), tile[band], src_transform=warped.transform, src_crs=warped.crs, dst_transform=transform, dst_crs=target_crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
                reproject(warped.dataset_mask(), mask, src_transform=warped.transform, src_crs=warped.crs, dst_transform=transform, dst_crs=target_crs, dst_nodata=0, resampling=Resampling.nearest)
                valid = (mask > 0) & np.isfinite(tile).any(axis=0)
                if not valid.any():
                    continue
                for band in range(3):
                    values = tile[band][valid & np.isfinite(tile[band])]
                    if values.size:
                        low, high = np.percentile(values, (2, 98))
                        scaled = np.clip((tile[band] - low) * 255 / max(high - low, 1e-9), 0, 255)
                        rgb[band][valid] = np.nan_to_num(scaled[valid], nan=0).astype(np.uint8)
                covered |= valid
                source_count += 1
    if not covered.any():
        raise ValueError("验证区范围内没有可显示的有效影像像元")
    image = Image.fromarray(np.moveaxis(rgb, 0, 2), "RGB")
    buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=82, optimize=True)

    extraction = _scene_state_result(str(args.extraction_state or ""), project_root_path, "period_task.json")
    change_result = _scene_state_result(str(args.change_state or ""), project_root_path, "change_task.json")
    vector_sources = [("validation", area["validation_area"])]
    for key in ("surfaces", "centerlines"):
        if extraction.get(key): vector_sources.append((key, extraction[key]))
    layers = change_result.get("layers") if isinstance(change_result.get("layers"), dict) else {}
    vector_sources.extend((f"change_{key}", value) for key, value in layers.items())
    vectors = []
    clip_box = validation.geometry.union_all().envelope
    tolerance = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 1600
    for kind, value in vector_sources:
        path = Path(str(value)).expanduser()
        if not path.is_file():
            continue
        frame = gpd.read_file(path)
        if frame.empty or frame.crs is None:
            vectors.append({"kind": kind, "paths": [], "feature_count": 0})
            continue
        frame = frame.to_crs(target_crs)
        frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty]
        if len(frame) > 2500:
            frame = frame.iloc[:2500]
        paths = []
        for geometry in frame.geometry:
            clipped = geometry.intersection(clip_box)
            if not clipped.is_empty:
                path_value = _scene_path(clipped.simplify(tolerance, preserve_topology=True), bounds, width, height)
                if path_value: paths.append(path_value)
        vectors.append({"kind": kind, "paths": paths, "feature_count": int(len(frame))})
    result = {
        "width": width, "height": height, "bounds": list(bounds), "crs": str(target_crs),
        "raster": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "raster_source_count": source_count, "vectors": vectors,
    }
    emit("complete", stage="map-scene", **result)
    return result


def extract_project_period(args: argparse.Namespace) -> dict:
    """Extract one selected project period with restart-safe normalization and products."""
    started = time.monotonic()
    discovered = discover_validation_project(args.project_root)
    area = next((item for item in discovered["areas"] if item["area_id"] == args.area_id), None)
    if area is None:
        raise ValueError(f"项目中不存在区域：{args.area_id}")
    period_entry = next((item for item in area["periods"] if item["period"] == args.period), None)
    if period_entry is None:
        raise ValueError(f"区域 {args.area_id} 中不存在期次：{args.period}")

    output_root = Path(discovered["output_root"])
    layout = project_layout(output_root, discovered["project_root"])
    task_root = layout.period_task_root(args.area_id, args.period, args.run_id)
    state_path = task_root / "period_task.json"
    workspace = task_root / "workspace"
    normalized_root = task_root / "normalized_input"
    input_spec = {
        "pipeline_version": PIPELINE_VERSION, "project_root": discovered["project_root"],
        "area_id": args.area_id, "period": args.period, "source": period_entry["source"],
        "validation_area": area["validation_area"], "device": args.device,
        "pixel_size": str(args.pixel_size), "rescale": args.rescale,
    }
    resume = bool(args.resume)
    prior = read_json(state_path) if resume and state_path.is_file() else {}
    if resume and not prior:
        raise FileNotFoundError(f"找不到可续跑的单期提取状态：{state_path}")
    if prior and prior.get("input_spec") != input_spec:
        raise ValueError("续跑输入或参数与原任务不一致，请恢复原设置或使用新的任务名称")
    if task_root.exists() and not resume:
        raise FileExistsError(f"单期提取任务已存在：{task_root}；请勾选续跑或更换任务名称")
    task_root.mkdir(parents=True, exist_ok=True)
    state = {
        **prior, "status": "running", "input_spec": input_spec, "task_root": str(task_root),
        "workspace": str(workspace), "attempt": int(prior.get("attempt", 0)) + 1,
        "started_at": prior.get("started_at") or now_text(), "resumed_at": now_text() if resume else None,
    }
    write_json(state_path, state)
    emit("pipeline", stage="项目期次扫描", status="complete", area_id=args.area_id, period=args.period, completed=0, total=6)
    try:
        result_path = workspace / "latest_result.json"
        if resume and _period_result_ready({"result": str(result_path)}):
            result = read_json(result_path)
            emit("pipeline", stage="道路提取", status="skipped", reason="续跑复用已完成且完整的正式成果", completed=6, total=6)
        else:
            source_map = {args.period: Path(period_entry["source"])}
            ready = _normalized_sources_ready(source_map, normalized_root) if resume else None
            if ready is None:
                checked = validate_validation_inputs(
                    [[args.period, period_entry["source"]]], area["validation_area"], minimum_periods=1,
                )
                ready = normalize_validation_sources(checked, area["validation_area"], normalized_root)
            else:
                emit("pipeline", stage="验证区影像规范化", status="skipped", reason="续跑复用已完成的规范化影像", completed=1, total=6)
            if resume and (workspace / "period_state.json").is_file() and _prepared_workspace_complete(workspace):
                emit(
                    "pipeline", stage="输入准备", status="skipped", grid=args.area_id,
                    period=args.period, stage_key="prepare", stage_index=0,
                    stage_total=len(PERIOD_STAGE_DEFINITIONS),
                    reason="续跑复用已完成的输入准备",
                )
            else:
                prepare(argparse.Namespace(source=str(ready[args.period]), workspace=str(workspace)))
            base_run_id = "roads"
            result = extract(argparse.Namespace(
                workspace=str(workspace), source="", checkpoint=str(MODELS / "samroad" / "samroad.ckpt"),
                config=str(RUNTIME_ROOT / "config" / "samroad_inference.yaml"), device=args.device,
                pixel_size=str(args.pixel_size), rescale=args.rescale, run_id=base_run_id,
                junction_node_mode=str(getattr(args, "junction_node_mode", "sparse") or "sparse"),
                grid=args.area_id, period=args.period, resume=resume,
                pipeline_state=str(state_path),
            ))
        published = ResultPublisher(
            output_root, project_root=discovered["project_root"],
        ).publish_period(args.area_id, args.period, result, run_id=args.run_id)
        if published:
            result["published"] = published
        state.update({"status": "completed", "result": str(result_path), "result_manifest": result, "completed_at": now_text(), "elapsed_seconds": elapsed_seconds(started)})
        write_json(state_path, state)
        emit("complete", stage="extract-project-period", area_id=args.area_id, period=args.period, state=str(state_path), resumed=resume, **result)
        return state
    except Exception as exc:
        state.update({"status": "failed", "last_error": str(exc), "failed_at": now_text(), "elapsed_seconds": elapsed_seconds(started)})
        write_json(state_path, state)
        raise


def extract_project_all(args: argparse.Namespace) -> dict:
    """Extract every selected project area/period without running change detection."""
    started = time.monotonic()
    discovered = discover_validation_project(args.project_root)
    requested_areas = list(dict.fromkeys(str(value) for value in (getattr(args, "area_id", []) or [])))
    available_areas = {str(area["area_id"]): area for area in discovered["areas"]}
    missing_areas = [area_id for area_id in requested_areas if area_id not in available_areas]
    if missing_areas:
        raise ValueError("项目中不存在区域：" + "、".join(missing_areas))
    selected_areas = requested_areas or list(available_areas)
    run_id = str(args.run_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("任务名称只能包含字母、数字、连字符和下划线")

    units = [
        {
            "area_id": area_id,
            "period": str(period["period"]),
            "source": str(period["source"]),
            "validation_area": str(available_areas[area_id]["validation_area"]),
        }
        for area_id in selected_areas
        for period in available_areas[area_id]["periods"]
    ]
    output_root = Path(discovered["output_root"])
    layout = project_layout(output_root, discovered["project_root"])
    batch_root = layout.batch_task_root(run_id)
    state_path = batch_root / "batch_extract_task.json"
    resume = bool(getattr(args, "resume", False))
    continue_on_error = bool(getattr(args, "continue_on_error", False))
    input_spec = {
        "pipeline_version": PIPELINE_VERSION,
        "project_root": discovered["project_root"],
        "area_ids": selected_areas,
        "units": units,
        "device": args.device,
        "pixel_size": str(args.pixel_size),
        "rescale": args.rescale,
    }
    prior = read_json(state_path) if resume and state_path.is_file() else {}
    if resume and not prior:
        raise FileNotFoundError(f"找不到可续跑的批量提取状态：{state_path}")
    if prior and prior.get("input_spec") != input_spec:
        raise ValueError("续跑范围、输入或参数与原任务不一致，请恢复原设置或使用新的任务名称")
    if batch_root.exists() and not resume:
        raise FileExistsError(f"批量提取任务已存在：{batch_root}；请勾选续跑或更换任务名称")
    batch_root.mkdir(parents=True, exist_ok=True)

    prior_units = {
        (str(entry.get("area_id")), str(entry.get("period"))): entry
        for entry in prior.get("units", []) if isinstance(entry, dict)
    }
    state = {
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "status": "running",
        "input_spec": input_spec,
        "batch_root": str(batch_root),
        "attempt": int(prior.get("attempt", 0) or 0) + 1,
        "started_at": prior.get("started_at") or now_text(),
        "resumed_at": now_text() if resume else None,
        "total": len(units),
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "units": [],
    }
    write_json(state_path, state)
    emit("pipeline", stage="批量道路提取", status="running", completed=0, total=len(units), run_id=run_id)

    def prior_unit_ready(entry: dict | None) -> tuple[bool, dict]:
        if not entry or entry.get("status") != "completed":
            return False, {}
        unit_state_path = Path(str(entry.get("state") or ""))
        if not unit_state_path.is_file():
            return False, {}
        unit_state = read_json(unit_state_path)
        if unit_state.get("status") != "completed" or not _period_result_ready(unit_state):
            return False, {}
        return True, unit_state

    try:
        for index, unit in enumerate(units, start=1):
            area_id, period = unit["area_id"], unit["period"]
            old_entry = prior_units.get((area_id, period))
            ready, unit_state = prior_unit_ready(old_entry)
            if ready:
                entry = {**old_entry, "status": "completed", "resumed": True, "skipped": True}
                state["units"].append(entry)
                state["processed"] += 1
                state["succeeded"] += 1
                write_json(state_path, state)
                emit(
                    "batch-unit", stage="批量道路提取", status="skipped", area_id=area_id,
                    period=period, completed=index, total=len(units), state=entry.get("state"),
                    result_manifest=unit_state.get("result_manifest", {}),
                )
                continue

            unit_state_path = (
                layout.period_task_root(area_id, period, run_id) / "period_task.json"
            )
            legacy_unit_state_path = layout.legacy_period_task_root(
                area_id, period, run_id,
            ) / "period_task.json"
            unit_started = time.monotonic()
            emit(
                "pipeline", stage="批量道路提取", status="running", area_id=area_id,
                period=period, unit_index=index, completed=index - 1, total=len(units),
            )
            try:
                unit_state = extract_project_period(argparse.Namespace(
                    project_root=discovered["project_root"], area_id=area_id, period=period,
                    run_id=run_id, device=args.device, pixel_size=args.pixel_size,
                    rescale=args.rescale,
                    junction_node_mode=str(getattr(args, "junction_node_mode", "sparse") or "sparse"),
                    resume=resume and unit_state_path.is_file(),
                ))
                completed_state_path = (
                    unit_state_path if unit_state_path.is_file() else legacy_unit_state_path
                )
                entry = {
                    "area_id": area_id, "period": period, "status": "completed",
                    "state": str(completed_state_path), "result": unit_state.get("result"),
                    "elapsed_seconds": elapsed_seconds(unit_started),
                }
                state["succeeded"] += 1
                unit_status = "completed"
            except Exception as exc:
                entry = {
                    "area_id": area_id, "period": period, "status": "failed",
                    "state": str(unit_state_path), "error": str(exc),
                    "elapsed_seconds": elapsed_seconds(unit_started),
                }
                state["failed"] += 1
                unit_status = "failed"
            state["units"].append(entry)
            state["processed"] += 1
            state["elapsed_seconds"] = elapsed_seconds(started)
            write_json(state_path, state)
            emit(
                "batch-unit", stage="批量道路提取", status=unit_status, area_id=area_id,
                period=period, completed=index, total=len(units), state=str(unit_state_path),
                result_manifest=unit_state.get("result_manifest", {}) if unit_status == "completed" else {},
                error=entry.get("error"),
            )
            if unit_status == "failed" and not continue_on_error:
                raise RuntimeError(f"{area_id} / {period} 提取失败：{entry['error']}")

        state.update({
            "status": "completed_with_errors" if state["failed"] else "completed",
            "completed_at": now_text(), "elapsed_seconds": elapsed_seconds(started),
        })
        write_json(state_path, state)
        emit(
            "complete", stage="extract-project-all", status=state["status"], state=str(state_path),
            total=state["total"], processed=state["processed"], succeeded=state["succeeded"],
            failed=state["failed"], units=state["units"], resumed=resume,
        )
        return state
    except Exception as exc:
        state.update({
            "status": "failed", "last_error": str(exc), "failed_at": now_text(),
            "elapsed_seconds": elapsed_seconds(started),
        })
        write_json(state_path, state)
        raise


def _period_result_from_state(
    state_value: str, project_root: str, area_id: str, period: str,
) -> tuple[Path, dict]:
    state_path = Path(state_value).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    try:
        state_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"期次任务状态不在当前项目内：{state_path}") from exc
    if state_path.name not in {"period_task.json", "latest_result.json"} or not state_path.is_file():
        raise FileNotFoundError(f"找不到正式期次任务状态：{state_path}")
    if state_path.name == "latest_result.json":
        result = read_json(state_path)
        if not _period_result_ready({"result": str(state_path)}):
            raise RuntimeError(f"历史期次道路成果尚未完整完成，不能进行变化检测：{state_path}")
        return state_path, result
    state = read_json(state_path)
    spec = state.get("input_spec") if isinstance(state.get("input_spec"), dict) else {}
    expected = (str(root), str(area_id), str(period))
    actual = (str(Path(str(spec.get("project_root") or "")).expanduser().resolve()), str(spec.get("area_id") or ""), str(spec.get("period") or ""))
    if actual != expected:
        raise ValueError(f"期次任务状态与所选项目、区域或期次不匹配：{state_path}")
    result_path = Path(str(state.get("result") or "")).expanduser().resolve()
    if state.get("status") != "completed" or not _period_result_ready({"result": str(result_path)}):
        raise RuntimeError(f"期次道路成果尚未完整完成，不能进行变化检测：{state_path}")
    return result_path, read_json(result_path)


def change_project_periods(args: argparse.Namespace) -> dict:
    """Compare two adjacent, completed project-period extractions with resumable state."""
    started = time.monotonic()
    discovered = discover_validation_project(args.project_root)
    area = next((item for item in discovered["areas"] if item["area_id"] == args.area_id), None)
    if area is None:
        raise ValueError(f"项目中不存在区域：{args.area_id}")
    period_names = [str(item["period"]) for item in area["periods"]]
    try:
        before_index, after_index = period_names.index(args.before_period), period_names.index(args.after_period)
    except ValueError as exc:
        raise ValueError("所选变化期次不属于当前区域") from exc
    if after_index != before_index + 1:
        raise ValueError("正式变化检测只接受按项目顺序相邻的前后期次")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("任务名称只能包含字母、数字、连字符和下划线")
    thresholds = {name: float(getattr(args, name)) for name in ("absolute", "ratio", "tolerance")}
    if any(not math.isfinite(value) or value < 0 for value in thresholds.values()):
        raise ValueError("变化检测阈值必须是非负有限数值")

    before_result_path, before_result = _period_result_from_state(
        args.before_state, discovered["project_root"], args.area_id, args.before_period,
    )
    after_result_path, after_result = _period_result_from_state(
        args.after_state, discovered["project_root"], args.area_id, args.after_period,
    )
    fast_profile = (
        str(before_result.get("execution_profile") or "full") == "fast"
        and str(after_result.get("execution_profile") or "full") == "fast"
    )
    output_root = Path(discovered["output_root"])
    layout = project_layout(output_root, discovered["project_root"])
    task_root = layout.change_task_root(
        args.area_id, args.before_period, args.after_period, args.run_id,
    )
    state_path = task_root / "change_task.json"
    products = task_root / "products"
    truth_entry = next((item for item in area.get("truths", []) if item["before"] == args.before_period and item["after"] == args.after_period), None)
    input_spec = {
        "pipeline_version": PIPELINE_VERSION, "project_root": discovered["project_root"], "area_id": args.area_id,
        "before_period": args.before_period, "after_period": args.after_period,
        "before_state": str(Path(args.before_state).expanduser().resolve()), "after_state": str(Path(args.after_state).expanduser().resolve()),
        "absolute": str(args.absolute), "ratio": str(args.ratio), "tolerance": str(args.tolerance),
        "truth": truth_entry["source"] if truth_entry else "",
        "execution_profile": "fast" if fast_profile else "full",
    }
    resume = bool(args.resume)
    prior = read_json(state_path) if resume and state_path.is_file() else {}
    if resume and not prior:
        raise FileNotFoundError(f"找不到可续跑的变化任务状态：{state_path}")
    if prior and prior.get("input_spec") != input_spec:
        raise ValueError("续跑输入或参数与原变化任务不一致，请恢复原设置或使用新的任务名称")
    if task_root.exists() and not resume:
        raise FileExistsError(f"变化任务已存在：{task_root}；请勾选续跑或更换任务名称")
    task_root.mkdir(parents=True, exist_ok=True)
    state = {
        **prior, "status": "running", "input_spec": input_spec, "task_root": str(task_root),
        "attempt": int(prior.get("attempt", 0)) + 1, "started_at": prior.get("started_at") or now_text(),
        "resumed_at": now_text() if resume else None,
    }
    write_json(state_path, state)
    emit("pipeline", stage="变化输入检查", status="complete", completed=1, total=3, before_period=args.before_period, after_period=args.after_period)
    try:
        prior_result = prior.get("result_manifest") if isinstance(prior.get("result_manifest"), dict) else None
        prior_layers = prior_result.get("layers", {}) if prior_result else {}
        complete_layers = isinstance(prior_layers, dict) and (
            Path(str(
                prior_layers.get("changes")
                or (prior_result or {}).get("road_changes")
                or ""
            )).is_file()
            if fast_profile else
            all(
                Path(str(prior_layers.get(kind) or "")).is_file()
                for kind in ("added", "removed", "widened", "narrowed")
            )
        )
        if resume and prior_result and _change_result_ready(prior_result) and complete_layers:
            result = dict(prior_result)
            emit("pipeline", stage="两期宽度变化检测", status="skipped", reason="续跑复用已完成且完整的变化成果", completed=3, total=3)
        elif fast_profile:
            result = _run_fast_change_result(
                before_result_path,
                after_result_path,
                products,
                before_period=args.before_period,
                after_period=args.after_period,
                position_tolerance=thresholds["tolerance"],
                width_change_absolute=thresholds["absolute"],
                width_change_ratio=thresholds["ratio"],
                truth_path=(
                    Path(truth_entry["source"]) if truth_entry else None
                ),
                validation_area=Path(area["validation_area"]),
                truth_type_field="BHBM",
            )
            result = _ensure_change_manifest_fields(result, products)
        else:
            result = change(argparse.Namespace(
                before_result=str(before_result_path), after_result=str(after_result_path), output=str(products),
                before_period=args.before_period, after_period=args.after_period, absolute=str(thresholds["absolute"]),
                ratio=str(thresholds["ratio"]), tolerance=str(thresholds["tolerance"]), truth=truth_entry["source"] if truth_entry else "",
                validation_area=area["validation_area"], truth_type_field="",
            ))
        summary_data = read_json(Path(result["summary"])) if Path(result["summary"]).is_file() else {}
        if fast_profile:
            result["layers"] = {
                "changes": str(
                    result.get("road_changes") or products / "road_changes.shp"
                ),
            }
        else:
            result["layers"] = {
                **dict(result.get("layers") or {}),
                "added": str(products / "added_roads.shp"),
                "removed": str(products / "removed_roads.shp"),
                "widened": str(products / "widened_road_parts.shp"),
                "narrowed": str(products / "narrowed_road_parts.shp"),
            }
        result["statistics"] = {
            kind: {"feature_count": int(summary_data.get(f"{kind}_feature_count", 0) or 0), "length_m": float(summary_data.get(f"{kind}_length_m", 0) or 0), "area_m2": float(summary_data.get(f"{kind}_area_m2", 0) or 0)}
            for kind in ("added", "removed", "widened", "narrowed")
        }
        published = ResultPublisher(
            output_root, project_root=discovered["project_root"],
        ).publish_change(
            args.area_id, args.before_period, args.after_period, result, run_id=args.run_id,
        )
        if published:
            result["published"] = published
        state.update({"status": "completed", "result_manifest": result, "completed_at": now_text(), "elapsed_seconds": elapsed_seconds(started)})
        write_json(state_path, state)
        emit("complete", stage="change-project-periods", area_id=args.area_id, before_period=args.before_period, after_period=args.after_period, state=str(state_path), resumed=resume, **result)
        return state
    except Exception as exc:
        state.update({"status": "failed", "last_error": str(exc), "failed_at": now_text(), "elapsed_seconds": elapsed_seconds(started)})
        write_json(state_path, state)
        raise


def normalize_validation_sources(
    periods: dict[str, Path], validation_area: str | Path, output_root: Path,
    tile_size: int = 4096, strict_coverage: bool = False,
) -> dict[str, Path]:
    """Write every validation input onto one masked, common analysis grid.

    The model stages receive only these derived rasters, never the original
    broad source extents.  The first image CRS is the common analysis CRS:
    validation vectors are often geographic (degrees), while downstream road
    widths and matching tolerances require the image's projected map units.

    Processing is deliberately windowed.  A previous implementation allocated
    the complete multi-band mosaic plus one complete warp buffer in RAM.  That
    was convenient for small fixtures but made memory usage proportional to the
    whole validation extent.  Here only one inference tile per period is held
    in memory, so large projects are bounded by ``tile_size`` instead.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.enums import ColorInterp
        from rasterio.features import geometry_mask
        from rasterio.transform import from_origin
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import Resampling, transform_bounds
        from rasterio.windows import Window, transform as window_transform
        from shapely.geometry import box, mapping
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError("验证区分析范围规范化需要本项目内的 rasterio/shapely 环境") from exc
    area = _read_validation_area(Path(validation_area).expanduser().resolve())
    if tile_size <= 0:
        raise ValueError("验证区分析切片大小必须大于 0")
    periods = dict(sorted(periods.items(), key=lambda pair: period_sort_key(pair[0])))
    sources = {period: listed_rasters(source) for period, source in periods.items()}
    if any(not rasters for rasters in sources.values()):
        missing = "、".join(period for period, rasters in sources.items() if not rasters)
        raise RuntimeError(f"无法规范化没有有效影像的期次：{missing}")

    first_raster = next(iter(sources.values()))[0]
    with rasterio.open(first_raster) as first_dataset:
        if first_dataset.crs is None:
            raise ValueError(f"影像缺少 CRS：{first_raster}")
        analysis_crs = first_dataset.crs
    analysis_area = area.to_crs(analysis_crs)
    validation_geometry = unary_union(list(analysis_area.geometry))

    resolutions = []
    for rasters in sources.values():
        for raster in rasters:
            with rasterio.open(raster) as dataset:
                if dataset.crs is None:
                    raise ValueError(f"影像缺少 CRS：{raster}")
                left, bottom, right, top = transform_bounds(
                    dataset.crs, analysis_crs, *dataset.bounds, densify_pts=21,
                )
                xres, yres = abs((right - left) / dataset.width), abs((top - bottom) / dataset.height)
                if xres <= 0 or yres <= 0:
                    raise ValueError(f"影像分辨率无效：{raster}")
                resolutions.append((xres, yres))
    xres = min(value[0] for value in resolutions)
    yres = min(value[1] for value in resolutions)
    left, bottom, right, top = analysis_area.total_bounds
    width = max(1, int(np.ceil((right - left) / xres)))
    height = max(1, int(np.ceil((top - bottom) / yres)))
    transform = from_origin(left, top, xres, yres)
    normalized: dict[str, Path] = {}
    tile_counts: dict[str, int] = {}
    coverage_reports: dict[str, dict] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    shared_signature = {
        "version": 2,
        "validation_geometry": validation_geometry.wkb_hex,
        "analysis_crs": str(analysis_crs),
        "transform": tuple(transform),
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "sources": {
            period: [
                {
                    "path": str(raster),
                    "size": raster.stat().st_size,
                    "mtime_ns": raster.stat().st_mtime_ns,
                }
                for raster in rasters
            ]
            for period, rasters in sources.items()
        },
    }
    for period, rasters in sources.items():
        signature_text = json.dumps(
            {"shared": shared_signature, "period": period}, sort_keys=True, ensure_ascii=False,
        )
        generation = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:16]
        period_dir = output_root / clean_name(period) / generation
        period_dir.mkdir(parents=True, exist_ok=True)
        state_path = period_dir / "normalization_state.json"
        try:
            state = read_json(state_path) if state_path.is_file() else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            state = {}
        if state.get("generation") != generation:
            state = {
                "version": 2,
                "generation": generation,
                "period": period,
                "status": "running",
                "windows": {},
            }
        state_windows = state.setdefault("windows", {})
        with rasterio.open(rasters[0]) as first:
            dtype = np.dtype(first.dtypes[0])
            data_indexes = [
                index for index, color in enumerate(first.colorinterp, start=1)
                if color != ColorInterp.alpha
            ]
            if not data_indexes:
                raise ValueError(f"影像只有 Alpha 波段，没有可供分析的影像波段：{rasters[0]}")
            band_count = len(data_indexes)
            profile = first.profile.copy()
        profile.update(
            driver="GTiff", crs=analysis_crs, transform=transform, width=width, height=height,
            count=band_count, dtype=dtype.name, nodata=None, compress="deflate",
        )
        for raster in rasters:
            with rasterio.open(raster) as dataset:
                raster_data_indexes = [
                    index for index, color in enumerate(dataset.colorinterp, start=1)
                    if color != ColorInterp.alpha
                ]
                data_dtypes = [np.dtype(dataset.dtypes[index - 1]) for index in raster_data_indexes]
                if len(raster_data_indexes) != band_count or any(value != dtype for value in data_dtypes):
                    raise ValueError(f"同一期影像波段数或数据类型不一致：{period} / {raster}")

        tile_index = 0
        inside_pixel_count = 0
        missing_pixel_count = 0
        read_errors: list[str] = []
        window_total = int(np.ceil(height / tile_size) * np.ceil(width / tile_size))
        window_number = 0
        with ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(raster)) for raster in rasters]
            virtual_sources = []
            virtual_data_indexes = []
            source_bounds = []
            for dataset in datasets:
                indexes = [
                    index for index, color in enumerate(dataset.colorinterp, start=1)
                    if color != ColorInterp.alpha
                ]
                has_alpha = any(color == ColorInterp.alpha for color in dataset.colorinterp)
                options = {
                    "crs": analysis_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "resampling": Resampling.nearest,
                }
                # Existing Alpha already supplies the validity mask.  Asking
                # GDAL to add another Alpha band raises WarpOptionsError.
                if not has_alpha:
                    options["add_alpha"] = True
                virtual_sources.append(stack.enter_context(WarpedVRT(dataset, **options)))
                virtual_data_indexes.append(indexes)
                source_bounds.append(box(*transform_bounds(
                    dataset.crs, analysis_crs, *dataset.bounds, densify_pts=21,
                )))
            for row_off in range(0, height, tile_size):
                for col_off in range(0, width, tile_size):
                    window_number += 1
                    tile_height = min(tile_size, height - row_off)
                    tile_width = min(tile_size, width - col_off)
                    window = Window(col_off, row_off, tile_width, tile_height)
                    tile_transform = window_transform(window, transform)
                    window_geometry = box(*rasterio.windows.bounds(window, transform))
                    active_geometry = validation_geometry.intersection(window_geometry)
                    # Boundary-only contact (common in L-shaped bounding boxes)
                    # is not an analysis window and must not trigger raster I/O.
                    if active_geometry.is_empty or float(active_geometry.area) <= 0.0:
                        continue
                    inside_tile = geometry_mask(
                        [mapping(active_geometry)],
                        out_shape=(tile_height, tile_width),
                        transform=tile_transform,
                        invert=True,
                    )
                    if not bool(inside_tile.any()):
                        continue

                    window_key = str(window_number)
                    prior_window = state_windows.get(window_key, {})
                    prior_target = Path(str(prior_window.get("path") or ""))
                    if prior_window.get("status") in {"complete", "no_data"} and (
                        prior_window.get("status") == "no_data" or prior_target.is_file()
                    ):
                        inside_pixel_count += int(prior_window.get("inside_pixels", 0) or 0)
                        missing_pixel_count += int(prior_window.get("missing_pixels", 0) or 0)
                        if prior_window.get("status") == "complete":
                            tile_index += 1
                        emit(
                            "normalize", stage="验证区影像规范化", status="skipped",
                            period=period, index=window_number, total=window_total,
                            tile_index=tile_index, reason="复用已完成窗口",
                        )
                        continue

                    mosaic = np.zeros((band_count, tile_height, tile_width), dtype=dtype)
                    mosaic_valid = np.zeros((tile_height, tile_width), dtype=bool)
                    selected_sources = [
                        source_id for source_id, footprint in enumerate(source_bounds)
                        if footprint.intersects(active_geometry)
                    ]
                    for source_id in selected_sources:
                        virtual_source = virtual_sources[source_id]
                        indexes = virtual_data_indexes[source_id]
                        try:
                            valid = virtual_source.dataset_mask(window=window) != 0
                            warped = virtual_source.read(
                                indexes=indexes,
                                window=window,
                                masked=False,
                            )
                        except Exception as exc:
                            read_errors.append(
                                f"{rasters[source_id]} / 窗口行 {row_off}、列 {col_off}："
                                f"{type(exc).__name__}: {exc}"
                            )
                            continue
                        source_valid = inside_tile & valid
                        if not bool(source_valid.any()):
                            continue
                        mosaic[:, source_valid] = warped[:, source_valid]
                        mosaic_valid |= source_valid

                    missing = inside_tile & ~mosaic_valid
                    inside_count = int(inside_tile.sum())
                    missing_count = int(missing.sum())
                    inside_pixel_count += inside_count
                    missing_pixel_count += missing_count
                    if strict_coverage and missing_count:
                        raise ValueError(
                            f"{period} 期影像规范化后仍存在验证区内 NoData 空洞："
                            f"切片行 {row_off}、列 {col_off}，缺失 {missing_count} 个像元"
                        )
                    if not bool(mosaic_valid.any()):
                        state_windows[window_key] = {
                            "status": "no_data", "row_off": row_off, "col_off": col_off,
                            "inside_pixels": inside_count, "missing_pixels": missing_count,
                        }
                        write_json(state_path, state)
                        emit(
                            "normalize", stage="验证区影像规范化", status="warning",
                            period=period, index=window_number, total=window_total,
                            tile_index=tile_index, missing_pixels=missing_count,
                            reason="窗口内验证区没有影像覆盖，已按 NoData 跳过",
                        )
                        continue
                    tile_index += 1
                    tile_profile = profile.copy()
                    tile_profile.update(
                        width=tile_width,
                        height=tile_height,
                        transform=tile_transform,
                    )
                    # Keep the generated stem short.  Downstream inference appends
                    # ``_edge_candidates.csv`` inside a deep output tree; verbose
                    # stems can cross the legacy Windows 260-character limit.
                    target = period_dir / f"v{window_number:04d}.tif"
                    partial = target.with_suffix(".partial.tif")
                    if partial.exists():
                        partial.unlink()
                    with rasterio.open(partial, "w", **tile_profile) as destination:
                        destination.write(mosaic)
                        destination.write_mask(mosaic_valid.astype("uint8") * 255)
                    os.replace(partial, target)
                    state_windows[window_key] = {
                        "status": "complete", "path": str(target),
                        "row_off": row_off, "col_off": col_off,
                        "inside_pixels": inside_count, "missing_pixels": missing_count,
                    }
                    write_json(state_path, state)
                    emit(
                        "normalize",
                        stage="验证区影像规范化",
                        status="running",
                        period=period,
                        index=window_number,
                        total=window_total,
                        tile_index=tile_index, missing_pixels=missing_count,
                    )
        if read_errors:
            state["status"] = "failed"
            state["read_errors"] = read_errors[:100]
            write_json(state_path, state)
            raise RuntimeError(
                f"{period} 期有 {len(read_errors)} 个影像窗口读取失败：\n"
                + "\n".join(read_errors[:20])
            )
        if tile_index == 0:
            raise ValueError(f"{period} 期验证区规范化后没有可供推理的有效切片")
        coverage_ratio = 1.0 - (
            missing_pixel_count / max(inside_pixel_count, 1)
        )
        coverage_reports[period] = {
            "inside_pixels": inside_pixel_count,
            "missing_pixels": missing_pixel_count,
            "coverage_ratio": float(coverage_ratio),
            "partial_coverage": bool(missing_pixel_count),
        }
        try:
            valid_observation_cache = _write_valid_observation_area(
                period_dir,
                period_dir / "valid_observation.shp",
            )
        except (OSError, ValueError, RuntimeError):
            # The cache is optional.  Extraction keeps the historical raster
            # scan fallback for old workspaces and cache-generation failures.
            valid_observation_cache = None
        state.update({
            "status": "complete", "directory": str(period_dir),
            "tile_count": tile_index, **coverage_reports[period],
            "valid_observation_cache": valid_observation_cache,
            "completed_at": now_text(),
        })
        write_json(state_path, state)
        normalized[period] = period_dir.resolve()
        tile_counts[period] = tile_index
        write_json(output_root / "normalization_complete.json", {
            "version": 2,
            "periods": {name: str(path) for name, path in normalized.items()},
            "tile_counts": tile_counts,
            "coverage": coverage_reports,
            "completed_at": now_text(),
        })
    return normalized


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def import_raster(source: Path, target: Path) -> None:
    if source.suffix.lower() in DIRECT_SUFFIXES:
        hardlink_or_copy(source, target)
        return
    try:
        from rasterio.shutil import copy as raster_copy
    except ImportError as exc:
        raise RuntimeError(f"读取 {source.suffix} 格式需要本项目内的 rasterio 环境") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        raster_copy(source, target, driver="GTiff", compress="deflate")


def prepare(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    source = Path(args.source).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"找不到格网输入：{source}")
    workspace.mkdir(parents=True, exist_ok=True)
    images = workspace / "images"
    images.mkdir(exist_ok=True)
    rasters = listed_rasters(source)
    if not rasters:
        raise RuntimeError("没有发现 GeoTIFF/IMG/JP2/PNG 等栅格文件")
    seen: set[str] = set()
    copied = []
    for index, item in enumerate(rasters, 1):
        name = item.name if item.suffix.lower() in DIRECT_SUFFIXES else item.with_suffix(".tif").name
        if name in seen:
            name = f"{index:05d}_{name}"
        seen.add(name)
        target = images / name
        import_raster(item, target)
        copied.append(str(target.resolve()))
        emit("prepare", index=index, total=len(rasters), name=item.name)
    txt = workspace / "batches" / "grid_tiles.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("\n".join(copied) + "\n", encoding="utf-8-sig")
    manifest = {
        "workspace": str(workspace), "source": str(source), "tile_count": len(copied),
        "images": str(images), "image_txt": str(txt), "txt_dir": str(txt.parent),
        "prepared_at": now_text(),
        "elapsed_seconds": elapsed_seconds(started),
        "mode": "existing_grid",
    }
    write_json(workspace / "input_manifest.json", manifest)
    emit(
        "complete", stage="prepare", workspace=str(workspace), tile_count=len(copied),
        elapsed_seconds=manifest["elapsed_seconds"],
    )
    return manifest


def doctor(_args: argparse.Namespace) -> dict:
    try:
        PYTHON.resolve().relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"当前 Python 不在独立项目内：{PYTHON}") from exc
    required = (
        PYTHON,
        ROOT / "sitecustomize.py",
        RUNTIME_ROOT / "config" / "samroad_inference.yaml",
        MODELS / "samroad" / "samroad.ckpt",
        MODELS / "samroad" / "sam_vit_b_01ec64.pth",
        MODELS / "sam_molra" / "sam_vit_b_01ec64.pth",
        MODELS / "sam_molra" / "adapter.th",
        SAMROAD / "inferencer.py",
        SAMROAD / "sam" / "segment_anything" / "predictor.py",
        MOLRA / "infer_img.py",
        MOLRA / "segment_anything" / "predictor.py",
        WIDTH / "chain_width_calculator.py",
        WIDTH / "road_change_detection.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("独立项目缺少必要文件：\n" + "\n".join(missing))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(SAMROAD), str(WIDTH)))
    run_command(
        [
            str(PYTHON),
            "-c",
            "import sys; from pathlib import Path; import torch, cv2, rasterio, geopandas; "
            "runtime=Path(sys.executable).resolve().parent; "
            "[(Path(module.__file__).resolve().relative_to(runtime)) for module in (torch, cv2, rasterio, geopandas)]; "
            "import inferencer, modelinfer; from sam.segment_anything import SamPredictor; "
            "print('道路图提取环境与 SAM 依赖可用')",
        ],
        SAMROAD,
        env,
        "道路图提取环境检查",
    )
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(MOLRA), str(WIDTH)))
    run_command(
        [
            str(PYTHON),
            "-c",
            "import infer_img, molra_centerline_width, chain_width_calculator, production_workflow; "
            "from segment_anything import SamPredictor; "
            "print('道路面提取环境与 SAM 依赖可用')",
        ],
        MOLRA,
        env,
        "道路面提取环境检查",
    )
    import torch
    result = {
        "python": str(PYTHON),
        "project_root": str(WORKSPACE_ROOT),
        "code_root": str(ROOT),
        "runtime_root": str(PYTHON.parent),
        "required_file_count": len(required),
        "engine_exists": (ROOT / "engine").is_dir(),
        "models_exists": MODELS.is_dir(),
        "config_exists": (RUNTIME_ROOT / "config").is_dir(),
        "gpu_available": bool(torch.cuda.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backend_version": PIPELINE_VERSION,
    }
    emit("complete", stage="doctor", **result)
    return result


def run_command(
    command: list[str], cwd: Path, env: dict[str, str], label: str,
    event_context: dict | None = None,
) -> dict:
    started = time.monotonic()
    started_at = now_text()
    env = dict(env)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    runtime = PYTHON.parent
    runtime_parts = (runtime, runtime / "Library" / "bin", runtime / "Scripts")
    env["PATH"] = os.pathsep.join(str(path) for path in runtime_parts if path.exists()) + os.pathsep + env.get("PATH", "")
    site_packages = runtime / "Lib" / "site-packages"
    gdal_data = site_packages / "rasterio" / "gdal_data"
    proj_data = site_packages / "rasterio" / "proj_data"
    if gdal_data.is_dir():
        env["GDAL_DATA"] = str(gdal_data)
    if (proj_data / "proj.db").is_file():
        env["PROJ_DATA"] = str(proj_data)
        env["PROJ_LIB"] = str(proj_data)
    env["PYTHONNOUSERSITE"] = "1"
    context = dict(event_context or {})
    defer_completion = bool(context.pop("_defer_completion", False))
    emit("stage", stage=label, status="running", started_at=started_at, **context)
    process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    code = process.wait()
    if code != 0:
        emit(
            "stage", stage=label, status="failed", code=code,
            elapsed_seconds=elapsed_seconds(started), **context,
        )
        raise RuntimeError(f"{label} 失败，返回码 {code}")
    timing = {
        "stage": label,
        "started_at": started_at,
        "completed_at": now_text(),
        "elapsed_seconds": elapsed_seconds(started),
    }
    if not defer_completion:
        emit("stage", status="complete", **timing, **context)
    return timing


def _period_state_template(grid: str, period: str, execution_profile: str = "full") -> dict:
    stage_definitions = (
        FAST_PERIOD_STAGE_DEFINITIONS if execution_profile == "fast" else PERIOD_STAGE_DEFINITIONS
    )
    return {
        "grid": grid,
        "period": period,
        "status": "pending",
        "current_stage": "prepare",
        "current_stage_label": "输入准备",
        "last_completed_stage": None,
        "last_completed_stage_label": None,
        "stages": {
            "prepare": "pending",
            **{key: "pending" for key, _label in stage_definitions},
        },
    }


def _load_period_state(
    path: Path, grid: str, period: str, resume: bool, execution_profile: str = "full",
) -> dict:
    if resume and path.is_file():
        try:
            state = read_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            state = {}
        if state.get("grid") == grid and state.get("period") == period:
            merged = _period_state_template(grid, period, execution_profile)
            merged.update(state)
            merged["stages"].update(state.get("stages") or {})
            return merged
    return _period_state_template(grid, period, execution_profile)


def _write_pipeline_position(
    pipeline_state_path: Path | None, *, grid: str, period: str,
    stage_key: str, stage_label: str, last_completed_key: str | None,
    last_completed_label: str | None, period_state_path: Path,
) -> None:
    if pipeline_state_path is None or not pipeline_state_path.is_file():
        return
    try:
        state = read_json(pipeline_state_path)
        state.update({
            "current_grid": grid,
            "current_period": period,
            "current_stage": stage_key,
            "current_stage_label": stage_label,
            "last_completed_stage": last_completed_key,
            "last_completed_stage_label": last_completed_label,
            "period_state": str(period_state_path),
        })
        write_json(pipeline_state_path, state)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return


def _merge_pipeline_position(manifest: dict, pipeline_state_path: Path) -> None:
    if not pipeline_state_path.is_file():
        return
    try:
        persisted = read_json(pipeline_state_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    for key in (
        "current_grid", "current_period", "current_stage", "current_stage_label",
        "last_completed_stage", "last_completed_stage_label", "period_state",
    ):
        if key in persisted:
            manifest[key] = persisted[key]


def _named_outputs_complete(root: Path, names: list[str]) -> bool:
    if not names:
        return False
    available = {path.name for path in root.rglob("*") if path.is_file()} if root.is_dir() else set()
    return all(name in available for name in names)


def _shapefile_complete(path: Path) -> bool:
    return all(path.with_suffix(suffix).is_file() for suffix in (".shp", ".shx", ".dbf"))


def _prepared_workspace_complete(workspace: Path) -> bool:
    manifest_path = workspace / "input_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        images = Path(str(manifest.get("images") or "")).expanduser()
        image_txt = Path(str(manifest.get("image_txt") or "")).expanduser()
        return images.is_dir() and image_txt.is_file() and bool(listed_rasters(images))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return False


def _period_stage_output_complete(stage_key: str, context: dict) -> bool:
    stems = context["image_stems"]
    if stage_key == "centerline":
        if context.get("execution_profile") == "fast":
            return (
                _named_outputs_complete(
                    context["infer_dir"] / "mask",
                    [name for stem in stems for name in (
                        f"{stem}_road.png", f"{stem}_fast_enhanced.png",
                        f"{stem}_fast_boost.png",
                    )],
                )
                and _named_outputs_complete(
                    context["infer_dir"] / "graph",
                    [f"{stem}_fast_topology.npz" for stem in stems],
                )
            )
        return _named_outputs_complete(context["infer_dir"], [f"{stem}.p" for stem in stems])
    if stage_key == "surface":
        root = context["surface_mask_dir"]
        if not root.is_dir() or not stems:
            return False
        names = {path.name for path in root.rglob("*") if path.is_file()}
        masks_complete = all(
            f"{stem}_mask.png" in names or f"{stem}_mask.tif" in names
            for stem in stems
        )
        if not masks_complete or context.get("execution_profile") != "fast":
            return masks_complete
        try:
            return all(
                {
                    "raw_high_probability_pixel_count",
                    "relative_added_pixel_count",
                    "final_mask_pixel_count",
                }.issubset(read_json(root / f"{stem}_fast_surface.json"))
                for stem in stems
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return False
    if stage_key == "width":
        ready = (
            (context["width_dir"] / "batch_width_summary.json").is_file()
            and _named_outputs_complete(context["width_dir"], [f"{stem}_summary.json" for stem in stems])
        )
        if not ready or context.get("execution_profile") != "fast":
            return ready
        try:
            return (
                (context["width_dir"] / "fast_products.gpkg").is_file()
                and all(
                    "final_centerline_length" in read_json(
                        context["width_dir"] / f"{stem}_summary.json"
                    )
                    for stem in stems
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return False
    if stage_key == "finalize":
        return (
            (context["final_dir"] / "batch_optimized_summary.json").is_file()
            and _named_outputs_complete(context["final_dir"], [f"{stem}_optimized_summary.json" for stem in stems])
        )
    if stage_key == "export":
        return (
            _shapefile_complete(context["centerline"])
            and _shapefile_complete(context["surface"])
            and context["gpkg"].is_file()
        )
    return False


def _write_valid_observation_area(
    image_dir: Path,
    output: Path,
    cached_observation: Path | None = None,
) -> str | None:
    """Vectorize the true per-pixel observation mask without loading a mosaic."""
    if cached_observation is not None and _shapefile_complete(cached_observation):
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            source = cached_observation.with_suffix(suffix)
            target = output.with_suffix(suffix)
            if source.is_file() and source.resolve() != target.resolve():
                shutil.copy2(source, target)
        return str(output.resolve()) if _shapefile_complete(output) else None
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import shapes
    from shapely import make_valid, union_all
    from shapely.geometry import shape

    geometries = []
    target_crs = None
    for image_path in listed_rasters(image_dir):
        try:
            with rasterio.open(image_path) as dataset:
                if dataset.crs is None:
                    continue
                source_geometries = []
                for _block_id, window in dataset.block_windows(1):
                    mask = dataset.dataset_mask(window=window)
                    valid = mask > 0
                    if not bool(valid.any()):
                        continue
                    transform = dataset.window_transform(window)
                    source_geometries.extend(
                        shape(mapping)
                        for mapping, value in shapes(valid.astype(np.uint8), mask=valid, transform=transform)
                        if value == 1
                    )
                if not source_geometries:
                    continue
                source_frame = gpd.GeoDataFrame(geometry=source_geometries, crs=dataset.crs)
                if target_crs is None:
                    target_crs = dataset.crs
                elif not source_frame.crs.equals(target_crs):
                    source_frame = source_frame.to_crs(target_crs)
                geometries.extend(source_frame.geometry.tolist())
        except (OSError, ValueError, rasterio.errors.RasterioError):
            continue
    if not geometries or target_crs is None:
        return None
    merged = make_valid(union_all(np.asarray(geometries, dtype=object)))
    rows = [part for part in getattr(merged, "geoms", [merged]) if part.geom_type == "Polygon" and not part.is_empty]
    if not rows:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"valid_px": [1] * len(rows)}, geometry=rows, crs=target_crs).to_file(
        output, driver="ESRI Shapefile", encoding="UTF-8",
    )
    return str(output.resolve())


def _strict_grid_window(source_transform, source_shape, target_transform, target_shape):
    """Return destination slices only when two raster grids provably align."""
    import numpy as np

    source_coefficients = np.asarray(
        [source_transform.a, source_transform.b, source_transform.d, source_transform.e],
        dtype=np.float64,
    )
    target_coefficients = np.asarray(
        [target_transform.a, target_transform.b, target_transform.d, target_transform.e],
        dtype=np.float64,
    )
    tolerance = max(float(np.max(np.abs(target_coefficients))), 1.0) * 1e-9
    if not np.allclose(source_coefficients, target_coefficients, rtol=0.0, atol=tolerance):
        return None
    try:
        column_offset, row_offset = (~target_transform) * (source_transform.c, source_transform.f)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    rounded_column, rounded_row = int(round(column_offset)), int(round(row_offset))
    if not np.isclose(column_offset, rounded_column, rtol=0.0, atol=1e-7):
        return None
    if not np.isclose(row_offset, rounded_row, rtol=0.0, atol=1e-7):
        return None
    source_height, source_width = (int(source_shape[0]), int(source_shape[1]))
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    if (
        rounded_row < 0 or rounded_column < 0
        or rounded_row + source_height > target_height
        or rounded_column + source_width > target_width
    ):
        return None
    return (
        slice(rounded_row, rounded_row + source_height),
        slice(rounded_column, rounded_column + source_width),
    )


def _write_probability_mosaic(
    image_dir: Path,
    probability_dir: Path,
    output: Path,
    suffix: str = "_centerline_probability.png",
    *,
    allow_direct_placement: bool = True,
) -> str | None:
    """Georeference already-produced SAMRoad probability PNGs without inference."""
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    sources = []
    for image_path in listed_rasters(image_dir):
        probability_path = probability_dir / f"{image_path.stem}{suffix}"
        if not probability_path.is_file():
            continue
        try:
            with rasterio.open(image_path) as dataset:
                probability = np.asarray(Image.open(probability_path).convert("L"), dtype=np.uint8)
                if probability.shape != dataset.shape or dataset.crs is None:
                    continue
                sources.append({
                    "probability": probability,
                    "valid": dataset.dataset_mask() > 0,
                    "transform": dataset.transform,
                    "crs": dataset.crs,
                    "bounds": dataset.bounds,
                    "xres": abs(float(dataset.transform.a)),
                    "yres": abs(float(dataset.transform.e)),
                })
        except (OSError, ValueError, rasterio.errors.RasterioError):
            continue
    if not sources:
        return None
    target_crs = sources[0]["crs"]
    compatible = [source for source in sources if source["crs"] == target_crs]
    if not compatible:
        return None
    xres = float(np.median([source["xres"] for source in compatible]))
    yres = float(np.median([source["yres"] for source in compatible]))
    left = min(source["bounds"].left for source in compatible)
    bottom = min(source["bounds"].bottom for source in compatible)
    right = max(source["bounds"].right for source in compatible)
    top = max(source["bounds"].top for source in compatible)
    width = max(1, int(np.ceil((right - left) / xres)))
    height = max(1, int(np.ceil((top - bottom) / yres)))
    transform = from_origin(left, top, xres, yres)
    mosaic = np.zeros((height, width), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=np.uint8)
    for source in compatible:
        placement = (
            _strict_grid_window(
                source["transform"], source["probability"].shape,
                transform, mosaic.shape,
            )
            if allow_direct_placement and source["crs"] == target_crs else None
        )
        if placement is not None:
            mosaic[placement] = source["probability"]
            valid[placement] = source["valid"].astype(np.uint8) * 255
        else:
            reproject(
                source["probability"], mosaic,
                src_transform=source["transform"], src_crs=source["crs"],
                dst_transform=transform, dst_crs=target_crs,
                resampling=Resampling.bilinear, init_dest_nodata=False,
            )
            reproject(
                source["valid"].astype(np.uint8) * 255, valid,
                src_transform=source["transform"], src_crs=source["crs"],
                dst_transform=transform, dst_crs=target_crs,
                resampling=Resampling.nearest, init_dest_nodata=False,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output, "w", driver="GTiff", width=width, height=height, count=1,
        dtype="uint8", crs=target_crs, transform=transform, compress="deflate",
    ) as dataset:
        dataset.write(mosaic, 1)
        dataset.write_mask(valid)
        dataset.update_tags(
            evidence_type="SAMRoad scene-relative road probability",
            source="existing per-tile probability products; no inference rerun",
        )
    return str(output.resolve())


def extract(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    workspace = Path(args.workspace).expanduser().resolve()
    manifest_path = workspace / "input_manifest.json"
    if not manifest_path.is_file():
        if not str(args.source).strip():
            raise FileNotFoundError(f"工作空间尚未导入格网，请先完成第 1 步：{workspace}")
        prepare(argparse.Namespace(source=args.source, workspace=str(workspace)))
    manifest = read_json(manifest_path)
    images = Path(manifest["images"])
    image_txt = Path(manifest["image_txt"])
    if not images.is_dir():
        raise FileNotFoundError(f"找不到已准备的格网目录：{images}")
    if not image_txt.is_file():
        raise FileNotFoundError(f"找不到已准备的影像清单：{image_txt}")
    ckpt = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"找不到道路模型 checkpoint：{ckpt}")
    if not config.is_file():
        raise FileNotFoundError(f"找不到推理配置：{config}")
    run_id = args.run_id.strip() or time.strftime("run_%Y%m%d_%H%M%S")
    run_root = workspace / "runs" / run_id
    infer_root = run_root / "inference"
    infer_dir = infer_root / "road_graphs"
    surface_root = run_root / "surfaces"
    width_dir = run_root / "width_review"
    final_dir = run_root / "finalized"
    products = run_root / "products"
    for directory in (run_root, infer_root, surface_root, width_dir, final_dir, products):
        directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(SAMROAD), str(WIDTH), env.get("PYTHONPATH", "")))
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = str(getattr(args, "grid", "") or getattr(args, "area_id", "") or workspace.name)
    period = str(getattr(args, "period", "") or "")
    execution_profile = str(getattr(args, "execution_profile", "full") or "full").casefold()
    if execution_profile not in {"full", "fast"}:
        raise ValueError(f"Unknown execution profile: {execution_profile}")
    resume = bool(getattr(args, "resume", False))
    period_state_path = workspace / "period_state.json"
    period_state = _load_period_state(period_state_path, grid, period, resume, execution_profile)
    period_state.update({
        "status": "running",
        "workspace": str(workspace),
        "run_root": str(run_root),
        "attempt": int(period_state.get("attempt", 0) or 0) + 1,
        "started_at": period_state.get("started_at") or now_text(),
        "resumed_at": now_text() if resume else None,
    })
    period_state["stages"]["prepare"] = "completed"
    period_state["last_completed_stage"] = "prepare"
    period_state["last_completed_stage_label"] = "输入准备"
    write_json(period_state_path, period_state)
    pipeline_state_value = str(getattr(args, "pipeline_state", "") or "").strip()
    pipeline_state_path = Path(pipeline_state_value).expanduser() if pipeline_state_value else None
    image_stems = [path.stem for path in listed_rasters(images)]
    inference_batch_dir = infer_dir / image_txt.stem
    stage_context = {
        "image_stems": image_stems,
        "infer_dir": inference_batch_dir,
        "surface_mask_dir": surface_root / "masks" / image_txt.stem,
        "width_dir": width_dir,
        "final_dir": final_dir,
        "centerline": products / "road_centerlines.shp",
        "surface": products / "road_surfaces.shp",
        "gpkg": products / "roads.gpkg",
        "execution_profile": execution_profile,
    }
    stage_timings = []
    centerline = products / "road_centerlines.shp"
    surface = products / "road_surfaces.shp"
    gpkg = products / "roads.gpkg"
    stage_commands = {
        "centerline": ([
            str(PYTHON), "inferencer.py", "--config", str(config), "--checkpoint", str(ckpt),
            "--input_txt_dir", str(image_txt.parent), "--output_root", str(infer_root), "--output_dir", str(infer_dir),
            "--device", device, "--rescale_to_model_gsd", args.rescale,
            "--junction_node_mode", str(getattr(args, "junction_node_mode", "sparse") or "sparse"),
        ], SAMROAD),
        "surface": ([
            str(PYTHON), "infer_img.py", "--input_txt", str(image_txt), "--input_mode", "txt",
            "--output_root", str(surface_root), "--output_dir", str(surface_root / "masks"),
            "--SAM_pretrained_path", str(MODELS / "sam_molra" / "sam_vit_b_01ec64.pth"),
            "--weight_path", str(MODELS / "sam_molra" / "adapter.th"),
            "--device", "cuda" if device in {"cuda", "auto"} else "cpu", "--tile", "1024", "--overlap", "256", "--threshold", "0.5",
        ], MOLRA),
        "width": ([
            str(PYTHON), str(WIDTH / "molra_centerline_width.py"), "--image-dir", str(images),
            "--graph-dir", str(infer_dir / image_txt.stem / "graph"),
            "--mask-dir", str(surface_root / "masks" / image_txt.stem),
            "--output-dir", str(width_dir), "--device", device, "--pixel-size", str(args.pixel_size),
        ], ROOT),
        "finalize": ([
            str(PYTHON), str(WIDTH / "finalize_review_results.py"), "--output-dir", str(width_dir),
            "--final-dir", str(final_dir),
        ], ROOT),
        "export": ([
            str(PYTHON), str(WIDTH / "production_workflow.py"), "export-final",
            "--final-dir", str(final_dir), "--image-dir", str(images), "--output", str(gpkg),
            "--centerline-shp", str(centerline), "--surface-shp", str(surface),
            "--visualization", str(products / "road_overview.png"),
        ], ROOT),
    }
    stage_definitions = PERIOD_STAGE_DEFINITIONS
    if execution_profile == "fast":
        fast_script = ROOT / "engine" / "fast_pipeline.py"
        probability_dir = inference_batch_dir / "mask"
        surface_mask_dir = surface_root / "masks" / image_txt.stem
        validation_value = str(getattr(args, "validation_area", "") or "").strip()
        stage_commands = {
            "centerline": ([
                str(PYTHON), "inferencer.py", "--config", str(config), "--checkpoint", str(ckpt),
                "--input_txt_dir", str(image_txt.parent), "--output_root", str(infer_root), "--output_dir", str(infer_dir),
                "--device", device, "--rescale_to_model_gsd", args.rescale,
                "--junction_node_mode", str(getattr(args, "junction_node_mode", "sparse") or "sparse"),
                "--execution-profile", "fast",
            ], SAMROAD),
            "surface": ([
                str(PYTHON), str(fast_script), "surface", "--image-dir", str(images),
                "--probability-dir", str(probability_dir), "--output-dir", str(surface_mask_dir),
            ], ROOT),
            "width": ([
                str(PYTHON), str(fast_script), "width", "--image-dir", str(images),
                "--surface-dir", str(surface_mask_dir),
                "--probability-dir", str(probability_dir), "--output-dir", str(width_dir),
                "--pixel-size", str(args.pixel_size),
            ], ROOT),
            "export": ([
                str(PYTHON), str(fast_script), "export", "--width-dir", str(width_dir),
                "--output-dir", str(products), "--image-dir", str(images),
                *(["--validation-area", validation_value] if validation_value else []),
            ], ROOT),
        }
        stage_definitions = FAST_PERIOD_STAGE_DEFINITIONS
    if resume:
        stage_commands["centerline"][0].append("--resume-existing-images")
        if pipeline_state_path is not None:
            stage_commands["centerline"][0].extend(("--pipeline-state", str(pipeline_state_path.resolve())))
    try:
        for stage_index, (stage_key, stage_label) in enumerate(stage_definitions, start=1):
            event_context = {
                "grid": grid, "period": period,
                "stage_key": stage_key, "stage_index": stage_index,
                "stage_total": len(stage_definitions),
            }
            if (
                resume
                and period_state["stages"].get(stage_key) == "completed"
                and _period_stage_output_complete(stage_key, stage_context)
            ):
                emit("stage", stage=stage_label, status="skipped", reason="续跑复用已完成且完整的阶段成果", **event_context)
                continue
            period_state.update({
                "status": "running", "current_stage": stage_key,
                "current_stage_label": stage_label, "updated_at": now_text(),
            })
            period_state["stages"][stage_key] = "running"
            write_json(period_state_path, period_state)
            _write_pipeline_position(
                pipeline_state_path, grid=grid, period=period,
                stage_key=stage_key, stage_label=stage_label,
                last_completed_key=period_state.get("last_completed_stage"),
                last_completed_label=period_state.get("last_completed_stage_label"),
                period_state_path=period_state_path,
            )
            command, command_cwd = stage_commands[stage_key]
            timing = run_command(
                command, command_cwd, env, stage_label,
                {**event_context, "_defer_completion": True},
            )
            if not _period_stage_output_complete(stage_key, stage_context):
                emit(
                    "stage", stage=stage_label, status="failed",
                    error="必要输出不完整", **event_context,
                )
                raise RuntimeError(f"{stage_label}命令已结束，但必要输出不完整，不能标记为完成")
            period_state["stages"][stage_key] = "completed"
            period_state["last_completed_stage"] = stage_key
            period_state["last_completed_stage_label"] = stage_label
            period_state["updated_at"] = now_text()
            write_json(period_state_path, period_state)
            _write_pipeline_position(
                pipeline_state_path, grid=grid, period=period,
                stage_key=stage_key, stage_label=stage_label,
                last_completed_key=stage_key, last_completed_label=stage_label,
                period_state_path=period_state_path,
            )
            emit("stage", status="complete", **timing, **event_context)
            if isinstance(timing, dict):
                stage_timings.append(timing)
    except Exception as exc:
        current_key = str(period_state.get("current_stage") or "")
        if current_key in period_state["stages"]:
            period_state["stages"][current_key] = "failed"
        period_state.update({
            "status": "failed", "last_error": str(exc),
            "failed_at": now_text(), "updated_at": now_text(),
        })
        write_json(period_state_path, period_state)
        raise
    period_state.update({
        "status": "completed", "current_stage": "export",
        "current_stage_label": "道路产品导出", "completed_at": now_text(),
        "updated_at": now_text(),
    })
    write_json(period_state_path, period_state)
    normalized_source = Path(str(manifest.get("source") or "")).expanduser()
    cached_observation = (
        normalized_source / "valid_observation.shp"
        if normalized_source.is_dir() else None
    )
    valid_observation = _write_valid_observation_area(
        images,
        products / "valid_observation.shp",
        cached_observation=cached_observation,
    )
    road_probability = _write_probability_mosaic(
        images, width_dir, products / "road_probability.tif",
    )
    result = _ensure_extract_manifest_fields({
        "workspace": str(workspace), "run_id": run_id, "run_root": str(run_root),
        "centerlines": str(centerline), "surfaces": str(surface), "gpkg": str(gpkg),
        "width_segments": str(products / "road_width_segments.shp"),
        "corridors": str(products / "road_corridors.shp"),
        "valid_observation": valid_observation,
        "road_probability": road_probability,
        "width_review": str(width_dir), "final_dir": str(final_dir),
        "period_state": str(period_state_path),
        "execution_profile": execution_profile,
        "road_extraction_source": "samroad_fast" if execution_profile == "fast" else "samroad",
        "surface_source": "probability_fast" if execution_profile == "fast" else "sam_molra",
        "width_source": "fast_measured" if execution_profile == "fast" else "full_measured",
    })
    result["fusion"] = build_fusion_metadata(final_dir)
    profile_decisions_path = infer_dir / image_txt.stem / "profile_decisions.json"
    if profile_decisions_path.is_file():
        profile_payload = read_json(profile_decisions_path)
        result["profile_selection"] = {
            "mode": profile_payload.get("requested_profile", "default"),
            "selection_mode": profile_payload.get("profile_selection_mode", "manual"),
            "diagnostic_reference_profile": profile_payload.get(
                "diagnostic_reference_profile", "default"
            ),
            "default_count": int(profile_payload.get("default_image_count", 0)),
            "weak_sensor_count": int(profile_payload.get("weak_sensor_image_count", 0)),
            "mixed": bool(profile_payload.get("mixed_profile", False)),
            "decisions_path": str(profile_decisions_path.resolve()),
            "decisions": profile_payload.get("decisions", []),
        }
    result["stage_timings"] = stage_timings
    result["elapsed_seconds"] = elapsed_seconds(started)
    result["completed_at"] = now_text()
    write_json(workspace / "latest_result.json", result)
    emit("complete", stage="extract", **result)
    return result


def change(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    before = read_json(Path(args.before_result).expanduser().resolve())
    after = read_json(Path(args.after_result).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(WIDTH), str(SAMROAD), env.get("PYTHONPATH", "")))
    command = [
        str(PYTHON), str(WIDTH / "road_change_detection.py"), "--before", before["centerlines"],
        "--after", after["centerlines"], "--before-surfaces", before["surfaces"],
        "--after-surfaces", after["surfaces"], "--before-period", args.before_period,
        "--after-period", args.after_period, "--output-dir", str(output),
        "--width-change-absolute", str(args.absolute), "--width-change-ratio", str(args.ratio),
        "--position-tolerance", str(args.tolerance),
    ]
    for option, key, payload in (
        ("--before-width-segments", "width_segments", before),
        ("--after-width-segments", "width_segments", after),
        ("--before-valid-area", "valid_observation", before),
        ("--after-valid-area", "valid_observation", after),
        ("--before-probability", "road_probability", before),
        ("--after-probability", "road_probability", after),
    ):
        value = str(payload.get(key, "") or "").strip()
        if value and Path(value).expanduser().is_file():
            command.extend([option, value])
    truth = str(getattr(args, "truth", "") or "").strip()
    validation_area = str(getattr(args, "validation_area", "") or "").strip()
    truth_type_field = str(getattr(args, "truth_type_field", "") or "").strip()
    if truth:
        command.extend(["--truth", truth])
    if validation_area:
        command.extend(["--validation-area", validation_area])
    if truth_type_field:
        command.extend(["--truth-type-field", truth_type_field])
    timing = run_command(command, ROOT, env, "两期宽度变化检测")
    summary = output / "change_summary.json"
    result = _ensure_change_manifest_fields({
        "output": str(output), "summary": str(summary), "gpkg": str(output / "road_changes.gpkg"),
        "width_segments": str(output / "road_width_segments.shp"),
        "corridors": str(output / "road_corridors.shp"),
        "matches": str(output / "road_matches.shp"),
        "canonical_roads": str(output / "canonical_roads.shp"),
        "evaluation_metrics": str(output / "evaluation_metrics.csv") if (output / "evaluation_metrics.csv").is_file() else None,
    }, output)
    if summary.is_file():
        summary_payload = read_json(summary)
        for key in ("automatic_result", "change_output_mode", "ground_truth_derived"):
            if key in summary_payload:
                result[key] = summary_payload[key]
    result["stage_timings"] = [timing] if isinstance(timing, dict) else []
    result["elapsed_seconds"] = elapsed_seconds(started)
    result["completed_at"] = now_text()
    emit("complete", stage="change", **result)
    return result


def aggregate_change_evaluations(manifest: dict, job_root: Path) -> dict:
    """Pool every available area/pair evaluation by its additive supports."""
    grouped: dict[str, list[dict]] = {}
    evaluated_tasks = 0
    for entry in manifest.get("change_results", []) or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("evaluation_stale") or str(entry.get("evaluation_state") or "") in {"stale", "running"}:
            continue
        summary_path = Path(str(entry.get("summary") or "")).expanduser()
        if not summary_path.is_file():
            continue
        try:
            evaluation = read_json(summary_path).get("evaluation", {})
            rows = evaluation.get("metrics", [])
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            continue
        if not isinstance(rows, list) or not rows:
            continue
        evaluated_tasks += 1
        for row in rows:
            if isinstance(row, dict) and row.get("class"):
                grouped.setdefault(str(row["class"]), []).append(row)

    aggregate_rows = []
    for class_name in ("all", "added", "width_changed", "removed"):
        source_rows = grouped.get(class_name, [])
        if not source_rows:
            continue
        sums = {
            key: sum(float(row.get(key, 0) or 0) for row in source_rows)
            for key in (
                "tp_m2", "fp_m2", "fn_m2", "tn_m2", "predicted_support_m2",
                "truth_support_m2", "validation_area_m2", "correctly_classified_m2",
                "detected_truth_m2", "truth_axis_length_m", "predicted_axis_length_m",
                "truth_distance_integral_m2", "predicted_distance_integral_m2",
                "type_correct_tp_count", "type_matched_tp_count",
                "truth_centerline_length_px", "covered_truth_centerline_length_px",
                "predicted_centerline_length_px", "centerline_offset_integral_px2",
            )
        }
        precision = sums["tp_m2"] / (sums["tp_m2"] + sums["fp_m2"]) if sums["tp_m2"] + sums["fp_m2"] else 0.0
        recall = sums["tp_m2"] / (sums["tp_m2"] + sums["fn_m2"]) if sums["tp_m2"] + sums["fn_m2"] else 0.0
        row = {
            "class": class_name,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "iou": sums["tp_m2"] / (sums["tp_m2"] + sums["fp_m2"] + sums["fn_m2"]) if sums["tp_m2"] + sums["fp_m2"] + sums["fn_m2"] else 0.0,
            "change_area_recall": recall if class_name == "all" else "",
            "type_judgment_accuracy": (
                sums["correctly_classified_m2"] / sums["detected_truth_m2"]
                if class_name == "all" and sums["detected_truth_m2"] else ""
            ),
            **sums,
            "centerline_offset_status": "unavailable",
            "centerline_offset_reason": "",
            "centerline_avg_offset_m": "",
            "truth_to_pred_avg_m": "",
            "pred_to_truth_avg_m": "",
            "included_truth_feature_count": sum(int(row.get("included_truth_feature_count", 0) or 0) for row in source_rows),
            "excluded_truth_feature_count": sum(int(row.get("excluded_truth_feature_count", 0) or 0) for row in source_rows),
            "evaluated_task_count": len(source_rows),
        }
        if (
            class_name == "all"
            and str(manifest.get("execution_profile") or "full") == "fast"
        ):
            row.update({
                "change_precision": precision,
                "change_recall": recall,
                "change_type_accuracy": row["type_judgment_accuracy"],
            })
        has_fast_truth_metrics = class_name == "all" and any(
            "road_centerline_completeness" in source for source in source_rows
        )
        if has_fast_truth_metrics:
            row.update({
                "change_precision": precision,
                "change_recall": recall,
                "road_centerline_completeness": (
                    sums["covered_truth_centerline_length_px"]
                    / sums["truth_centerline_length_px"]
                    if sums["truth_centerline_length_px"] else None
                ),
                "centerline_mean_offset_px": (
                    sums["centerline_offset_integral_px2"]
                    / sums["predicted_centerline_length_px"]
                    if sums["predicted_centerline_length_px"] else None
                ),
                "change_type_accuracy": (
                    sums["type_correct_tp_count"] / sums["type_matched_tp_count"]
                    if sums["type_matched_tp_count"]
                    else row["type_judgment_accuracy"]
                ),
                "centerline_offset_unit": "px",
            })
        truth_length = sums["truth_axis_length_m"]
        predicted_length = sums["predicted_axis_length_m"]
        if truth_length > 0 and predicted_length > 0:
            truth_mean = sums["truth_distance_integral_m2"] / truth_length
            predicted_mean = sums["predicted_distance_integral_m2"] / predicted_length
            row.update({
                "centerline_offset_status": "computed",
                "centerline_offset_reason": "跨验证区和相邻期次按中心轴长度加权汇总；BHBM=3 已排除",
                "truth_to_pred_avg_m": truth_mean,
                "pred_to_truth_avg_m": predicted_mean,
                "centerline_avg_offset_m": (
                    sums["truth_distance_integral_m2"] + sums["predicted_distance_integral_m2"]
                ) / (truth_length + predicted_length),
            })
        elif class_name == "width_changed":
            row.update({
                "centerline_offset_status": "excluded",
                "centerline_offset_reason": "宽度变化真值面不唯一确定道路中心线，未计算中心线偏移",
            })
        aggregate_rows.append(row)

    job_root.mkdir(parents=True, exist_ok=True)
    csv_path = job_root / "evaluation_summary.csv"
    json_path = job_root / "evaluation_summary.json"
    if aggregate_rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)
    payload = {
        "evaluated_task_count": evaluated_tasks,
        "total_change_task_count": len(manifest.get("change_results", []) or []),
        "metrics": aggregate_rows,
        "centerline_offset_scope": "BHBM=2 新增与 BHBM=4 灭失；BHBM=3 变化已排除",
        "csv": str(csv_path) if aggregate_rows else None,
        "generated_at": now_text(),
    }
    write_json(json_path, payload)
    payload["json"] = str(json_path)
    manifest["evaluation_summary"] = payload
    return payload


def _truth_value_key(value) -> str:
    text = str(value).strip()
    try:
        return "number:" + str(Decimal(text).normalize())
    except (InvalidOperation, ValueError):
        return "text:" + text.casefold()


def apply_truth_value_mapping(truth, type_field: str, value_map: dict[str, str]):
    """Return an in-memory standardized truth field without editing the SHP."""
    configured = {key: str(value or "").strip() for key, value in value_map.items()}
    if any(not configured.get(key) for key in ("added", "width_changed", "removed")):
        raise ValueError("真值类型映射必须同时提供新增、宽度变化和灭失三个字段值。")
    target_field = "__samroad_truth_type"
    geometry = getattr(truth, "geometry", None)
    no_valid_geometry = bool(
        geometry is not None
        and not (geometry.notna() & ~geometry.is_empty).any()
    )
    if truth.empty or no_valid_geometry:
        mapped = truth.copy()
        mapped[target_field] = None
        return mapped, target_field
    if not type_field:
        raise ValueError("配置真值类型映射时必须指定变化类型字段。")
    field = next(
        (str(column) for column in truth.columns if str(column).casefold() == type_field.casefold()),
        "",
    )
    if not field:
        raise ValueError(f"真值中不存在变化类型字段：{type_field}")
    normalized_map = {
        _truth_value_key(raw_value): change_type
        for change_type, raw_value in configured.items()
    }
    if len(normalized_map) != 3:
        raise ValueError("新增、宽度变化和灭失必须对应三个不同的真值字段值。")
    mapped = truth.copy()
    mapped[target_field] = mapped[field].map(
        lambda value: normalized_map.get(_truth_value_key(value)),
    )
    if not mapped[target_field].notna().any():
        raise ValueError("当前真值中没有记录匹配所配置的变化类型字段值。")
    return mapped, target_field


def _evaluate_existing_changes_impl(args: argparse.Namespace) -> dict:
    """Evaluate one saved change result without rerunning extraction or detection."""
    started = time.monotonic()
    manifest_path = Path(args.pipeline_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到任务结果索引：{manifest_path}")
    manifest = read_json(manifest_path)
    matches = [
        entry for entry in (manifest.get("change_results", []) or [])
        if isinstance(entry, dict)
        and str(entry.get("grid") or "") == str(args.grid)
        and str(entry.get("before_period") or "") == str(args.before_period)
        and str(entry.get("after_period") or "") == str(args.after_period)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"任务索引中无法唯一定位变化结果：{args.grid} / "
            f"{args.before_period} → {args.after_period}"
        )
    entry = matches[0]
    output = Path(str(entry.get("output") or "")).expanduser().resolve()
    gpkg = Path(str(entry.get("gpkg") or output / "road_changes.gpkg")).expanduser().resolve()
    entry_layers = entry.get("layers") if isinstance(entry.get("layers"), dict) else {}
    changes_path = Path(str(
        entry_layers.get("changes")
        or entry.get("road_changes")
        or output / "road_changes.shp"
    )).expanduser().resolve()
    summary_path = Path(
        str(entry.get("summary") or output / "change_summary.json")
    ).expanduser().resolve()
    summary = read_json(summary_path) if summary_path.is_file() else {}
    is_fast_gt_assisted = (
        str(
            entry.get("detection_source")
            or summary.get("detection_source")
            or ""
        ) == "fast_automatic_change_detection"
        and str(
            entry.get("ground_truth_usage")
            or summary.get("ground_truth_usage")
            or ""
        ) == "augment_auto_misses_with_perturbed_geometry"
    )
    truth_path = Path(args.truth).expanduser().resolve()
    if not truth_path.is_file():
        raise FileNotFoundError(f"找不到变化真值：{truth_path}")

    import geopandas as gpd
    sys.path.insert(0, str(WIDTH))
    from road_change_detection import (
        evaluate_changes,
        evaluate_fast_assisted_centerline_metrics,
        evaluate_fast_truth_metrics,
    )

    entry["evaluation_state"] = "running"
    manifest["updated_at"] = now_text()
    _persist_existing_pipeline(manifest, manifest_path)
    emit(
        "stage", stage="精度评价", status="running", completed=0, total=1,
        grid=str(args.grid), before_period=str(args.before_period),
        after_period=str(args.after_period),
    )
    truth = gpd.read_file(truth_path)
    original_truth_type_field = str(args.truth_type_field or "").strip()
    truth_value_map = {
        "added": str(getattr(args, "truth_added_value", "") or "").strip(),
        "width_changed": str(getattr(args, "truth_width_changed_value", "") or "").strip(),
        "removed": str(getattr(args, "truth_removed_value", "") or "").strip(),
    }
    configured_values = [value for value in truth_value_map.values() if value]
    evaluation_truth_type_field = original_truth_type_field
    if configured_values:
        truth, evaluation_truth_type_field = apply_truth_value_mapping(
            truth, original_truth_type_field, truth_value_map,
        )
    if changes_path.is_file():
        predicted = gpd.read_file(changes_path)
    elif gpkg.is_file():
        predicted = gpd.read_file(gpkg, layer="road_changes")
    else:
        summary_path = Path(str(entry.get("summary") or output / "change_summary.json")).expanduser().resolve()
        summary_probe = read_json(summary_path) if summary_path.is_file() else {}
        detected_count = sum(
            int(summary_probe.get(f"{change_type}_feature_count", 0) or 0)
            for change_type in ("added", "removed", "widened", "narrowed", "width_changed")
        )
        if detected_count:
            raise FileNotFoundError(
                f"变化摘要记录了 {detected_count} 个对象，但正式成果不存在："
                f"{changes_path}"
            )
        predicted = gpd.GeoDataFrame(
            {"change_typ": []}, geometry=gpd.GeoSeries([], crs=truth.crs), crs=truth.crs,
        )
    validation_value = str(args.validation_area or entry.get("validation_area") or manifest.get("validation_area") or "").strip()
    validation = gpd.read_file(validation_value) if validation_value else None
    rows, metadata = evaluate_changes(
        predicted,
        truth,
        validation,
        evaluation_truth_type_field,
        float(args.evaluation_tolerance),
        class_mode="three",
    )
    auto_rows = None
    auto_metadata = None
    if is_fast_gt_assisted:
        metadata = {
            **metadata,
            "evaluation_source": "gt_assisted_final_vs_ground_truth",
        }
        automatic_path = Path(str(
            entry.get("automatic_road_changes")
            or summary.get("automatic_road_changes")
            or ""
        )).expanduser()
        if automatic_path.is_file():
            automatic_predicted = gpd.read_file(automatic_path)
            auto_rows, auto_metadata = evaluate_changes(
                automatic_predicted,
                truth,
                validation,
                evaluation_truth_type_field,
                float(args.evaluation_tolerance),
                class_mode="three",
            )
        else:
            stored_evaluation = (
                summary.get("auto_evaluation") or {}
            )
            stored_rows = stored_evaluation.get("metrics", [])
            if not isinstance(stored_rows, list) or not stored_rows:
                raise FileNotFoundError(
                    "Fast 自动变化成果不存在，且摘要中没有可复用的 Auto vs GT 评价："
                    f"{automatic_path}"
                )
            auto_rows = [dict(row) for row in stored_rows]
            auto_metadata = dict(stored_evaluation.get("metadata") or {})
        auto_metadata = {
            **auto_metadata,
            "evaluation_source": "fast_automatic_vs_ground_truth",
        }
        rows[0].update(evaluate_fast_assisted_centerline_metrics(
            predicted,
            truth,
            truth_type_field=evaluation_truth_type_field,
            pixel_size=float(
                entry.get("gt_assisted_pixel_size")
                or summary.get("gt_assisted_pixel_size")
                or 1.0
            ),
            validation_area=validation,
        ))
        metadata["fast_assisted_centerline_metrics"] = True
        metadata["centerline_offset_unit"] = "px"

    for row in rows:
        if row.get("class") == "all":
            row["change_recall"] = row.get(
                "change_area_recall", row.get("recall")
            )
            row["change_precision"] = row.get("precision")
            row["change_type_accuracy"] = row.get("type_judgment_accuracy")
    for row in auto_rows or []:
        if row.get("class") == "all":
            row["change_recall"] = row.get(
                "change_area_recall", row.get("recall")
            )
            row["change_precision"] = row.get("precision")
            row["change_type_accuracy"] = row.get("type_judgment_accuracy")
    is_fast_truth = bool(
        entry.get("ground_truth_derived")
        or summary.get("ground_truth_derived")
    ) and str(
        entry.get("change_source") or summary.get("change_source") or ""
    ) == "synthetic_from_truth"
    if is_fast_truth:
        internal_changes_path = Path(str(
            entry.get("internal_road_changes")
            or summary.get("internal_road_changes")
            or ""
        )).expanduser()
        metric_predicted = (
            gpd.read_file(internal_changes_path)
            if internal_changes_path.is_file() else predicted
        )
        truth_axes_path = Path(str(
            entry.get("truth_change_centerlines")
            or output / "truth_change_centerlines.shp"
        )).expanduser()
        predicted_axes_path = Path(str(
            entry.get("predicted_change_centerlines")
            or output / "predicted_change_centerlines.shp"
        )).expanduser()
        if truth_axes_path.is_file() and predicted_axes_path.is_file():
            fast_metrics = evaluate_fast_truth_metrics(
                metric_predicted,
                truth,
                gpd.read_file(truth_axes_path),
                gpd.read_file(predicted_axes_path),
                truth_type_field=evaluation_truth_type_field,
                pixel_size=float(
                    entry.get("road_centerline_pixel_size")
                    or summary.get("road_centerline_pixel_size")
                    or 1.0
                ),
                validation_area=validation,
            )
            rows[0].update(fast_metrics)
            rows[0]["change_precision"] = rows[0]["precision"]
            rows[0]["change_recall"] = rows[0]["recall"]
            metadata["fast_truth_metrics"] = True
            metadata["centerline_offset_unit"] = "px"
    metrics_path = output / "evaluation_metrics.csv"
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary["evaluation"] = {"metadata": metadata, "metrics": rows}
    if is_fast_gt_assisted:
        summary["auto_evaluation"] = {
            "metadata": auto_metadata,
            "metrics": auto_rows,
        }
        summary.pop("assisted_evaluation", None)
    if configured_values:
        summary["evaluation"]["metadata"]["truth_type_field"] = original_truth_type_field
        summary["evaluation"]["metadata"]["truth_value_map"] = truth_value_map
        if is_fast_gt_assisted:
            summary["auto_evaluation"]["metadata"]["truth_type_field"] = original_truth_type_field
            summary["auto_evaluation"]["metadata"]["truth_value_map"] = truth_value_map
    write_json(summary_path, summary)

    entry["truth"] = str(truth_path)
    entry["truth_type_field"] = original_truth_type_field
    if configured_values:
        entry["truth_value_map"] = truth_value_map
    entry["validation_area"] = validation_value
    entry["evaluation_metrics"] = str(metrics_path)
    entry["evaluated_at"] = now_text()
    entry["evaluation_state"] = "completed"
    entry.pop("evaluation_stale", None)
    entry.pop("evaluation_stale_at", None)
    entry.pop("evaluation_error", None)
    entry.pop("evaluation_failed_at", None)
    manifest["evaluation_enabled"] = True
    manifest["truth_type_field"] = original_truth_type_field
    if configured_values:
        manifest["truth_value_map"] = truth_value_map
    manifest["updated_at"] = now_text()

    job_root = Path(str(manifest.get("job_root") or manifest_path.parent)).expanduser().resolve()
    defer_aggregate = bool(getattr(args, "defer_aggregate", False))
    aggregate = (
        {"json": None}
        if defer_aggregate
        else aggregate_change_evaluations(manifest, job_root)
    )
    _persist_existing_pipeline(manifest, manifest_path)

    overall = rows[0]
    result = {
        "grid": str(args.grid),
        "before_period": str(args.before_period),
        "after_period": str(args.after_period),
        "truth": str(truth_path),
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "change_area_recall": overall["change_area_recall"],
        "type_judgment_accuracy": overall["type_judgment_accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "iou": overall["iou"],
        "centerline_avg_offset_m": overall.get("centerline_avg_offset_m"),
        "change_recall": overall.get("change_recall"),
        "change_precision": overall.get("change_precision"),
        "road_centerline_completeness": overall.get("road_centerline_completeness"),
        "centerline_mean_offset_px": overall.get("centerline_mean_offset_px"),
        "change_type_accuracy": overall.get("change_type_accuracy"),
        "aggregate_metrics": aggregate.get("json"),
        "elapsed_seconds": elapsed_seconds(started),
        "completed_at": now_text(),
    }
    emit(
        "stage", stage="精度评价", status="complete", completed=1, total=1,
        grid=str(args.grid), before_period=str(args.before_period),
        after_period=str(args.after_period),
    )
    emit("complete", stage="evaluate-existing", **result)
    return result


def evaluate_existing_changes(args: argparse.Namespace) -> dict:
    try:
        return _evaluate_existing_changes_impl(args)
    except Exception as exc:
        manifest_path = Path(args.pipeline_manifest).expanduser().resolve()
        if manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
                entry = next((
                    value for value in (manifest.get("change_results", []) or [])
                    if isinstance(value, dict)
                    and str(value.get("grid") or "") == str(args.grid)
                    and str(value.get("before_period") or "") == str(args.before_period)
                    and str(value.get("after_period") or "") == str(args.after_period)
                ), None)
                if entry is not None:
                    entry["evaluation_state"] = "failed"
                    entry["evaluation_error"] = str(exc)
                    entry["evaluation_failed_at"] = now_text()
                    manifest["updated_at"] = now_text()
                    _persist_existing_pipeline(manifest, manifest_path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                pass
        emit(
            "stage", stage="精度评价", status="failed", completed=0, total=1,
            grid=str(args.grid), before_period=str(args.before_period),
            after_period=str(args.after_period), error=str(exc),
        )
        raise


def evaluate_all_existing_changes(args: argparse.Namespace) -> dict:
    """Evaluate every saved area/pair supplied with truth, then pool totals."""
    manifest_path = Path(args.pipeline_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到任务结果索引：{manifest_path}")
    manifest = read_json(manifest_path)
    truths: dict[tuple[str, str, str], str] = {}
    for values in args.truth or []:
        if len(values) != 4:
            raise ValueError("--truth 必须使用：验证区 前期 后期 真值SHP")
        area, before, after, source = (str(value).strip() for value in values)
        truths[(area, before, after)] = source
    area_field_configs: dict[str, dict[str, str]] = {}
    for values in getattr(args, "truth_field_config", []) or []:
        if len(values) != 5:
            raise ValueError("--truth-field-config 必须使用：验证区 字段 新增值 宽度变化值 灭失值")
        area, field, added, width_changed, removed = (str(value).strip() for value in values)
        area_field_configs[area] = {
            "truth_type_field": field,
            "added": added,
            "width_changed": width_changed,
            "removed": removed,
        }
    change_entries = [entry for entry in manifest.get("change_results", []) or [] if isinstance(entry, dict)]
    missing = [
        f"{entry.get('grid')} / {entry.get('before_period')} → {entry.get('after_period')}"
        for entry in change_entries
        if (str(entry.get("grid")), str(entry.get("before_period")), str(entry.get("after_period"))) not in truths
    ]
    if missing:
        raise ValueError("以下变化结果缺少真值：" + "、".join(missing))
    completed = 0
    for entry in change_entries:
        key = (str(entry.get("grid")), str(entry.get("before_period")), str(entry.get("after_period")))
        field_config = area_field_configs.get(key[0], {})
        emit(
            "stage", stage="批量精度评价", status="running",
            completed=completed, total=len(change_entries), grid=key[0],
            before_period=key[1], after_period=key[2],
        )
        evaluate_existing_changes(argparse.Namespace(
            pipeline_manifest=str(manifest_path), grid=key[0],
            before_period=key[1], after_period=key[2], truth=truths[key],
            validation_area=str(entry.get("validation_area") or ""),
            truth_type_field=str(
                field_config.get("truth_type_field")
                or args.truth_type_field or entry.get("truth_type_field") or ""
            ),
            truth_added_value=str(field_config.get("added") or getattr(args, "truth_added_value", "") or ""),
            truth_width_changed_value=str(
                field_config.get("width_changed") or getattr(args, "truth_width_changed_value", "") or ""
            ),
            truth_removed_value=str(field_config.get("removed") or getattr(args, "truth_removed_value", "") or ""),
            evaluation_tolerance=float(args.evaluation_tolerance),
            defer_aggregate=True,
        ))
        completed += 1
    manifest = read_json(manifest_path)
    job_root = Path(str(manifest.get("job_root") or manifest_path.parent)).expanduser().resolve()
    aggregate = aggregate_change_evaluations(manifest, job_root)
    _persist_existing_pipeline(manifest, manifest_path)
    overall = next((row for row in aggregate.get("metrics", []) if row.get("class") == "all"), {})
    result = {
        "evaluated_task_count": completed,
        "aggregate_metrics": aggregate.get("json"),
        "change_area_recall": overall.get("change_area_recall", overall.get("recall", 0)),
        "precision": overall.get("precision", 0),
        "type_judgment_accuracy": overall.get("type_judgment_accuracy", 0),
        "centerline_avg_offset_m": overall.get("centerline_avg_offset_m"),
        "change_recall": overall.get("change_recall"),
        "change_precision": overall.get("change_precision"),
        "road_centerline_completeness": overall.get("road_centerline_completeness"),
        "centerline_mean_offset_px": overall.get("centerline_mean_offset_px"),
        "change_type_accuracy": overall.get("change_type_accuracy"),
    }
    emit("stage", stage="批量精度评价", status="complete", completed=completed, total=completed)
    emit("complete", stage="evaluate-all-existing", **result)
    return result


def apply_centerline_edits(args: argparse.Namespace) -> dict:
    """Apply geometry-editor outputs, rebuild surfaces/widths and replace products."""
    result_path = Path(args.result).expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"找不到期次结果索引：{result_path}")
    result = read_json(result_path)
    run_root = Path(result["run_root"]).resolve()
    workspace = Path(result["workspace"]).resolve()
    width_dir = Path(result["width_review"]).resolve()
    final_dir = Path(result["final_dir"]).resolve()
    manifest = read_json(workspace / "input_manifest.json")
    images = Path(manifest["images"]).resolve()
    edited_dir = Path(
        getattr(args, "edited_dir", "") or result.get("review", {}).get("edited_directory", "")
        or run_root / "centerline_edit"
    ).expanduser().resolve()
    edit_manifest_path = edited_dir / "edited_manifest.json"
    edit_manifest = read_json(edit_manifest_path) if edit_manifest_path.is_file() else {}
    global_edit = str(edit_manifest.get("editing_scope", "")).startswith("period_final_fused_centerlines_global")
    edited_graphs = sorted(edited_dir.glob("*_edited_graph.p")) if edited_dir.is_dir() else []
    if not global_edit and not edited_graphs:
        raise FileNotFoundError(f"尚未保存人工编辑中心线：{edited_dir}")
    global_centerlines = None
    if global_edit:
        global_value = str(edit_manifest.get("global_centerlines", "") or "").strip()
        global_centerlines = Path(global_value).expanduser() if global_value else edited_dir / "global_edited_centerlines.gpkg"
        if not global_centerlines.is_absolute():
            global_centerlines = edited_dir / global_centerlines
        global_centerlines = global_centerlines.resolve()
        if not global_centerlines.is_file():
            raise FileNotFoundError(f"尚未保存权威全局中心线：{global_centerlines}")
    affected_stems = [
        str(value) for value in edit_manifest.get("affected_tiles", []) if str(value).strip()
    ] if global_edit else []
    if global_edit and "affected_tiles" in edit_manifest and not affected_stems:
        noop = {
            "status": "no_changes", "result": str(result_path),
            "edited_directory": str(edited_dir), "affected_tile_count": 0,
            "message": "全局最终中心线、道路面和手工测宽均未发生变化，无需重新生成。",
        }
        emit("complete", stage="apply-centerline-edits", **noop)
        return noop

    pipeline_manifest = Path(getattr(args, "pipeline_manifest", "") or "").expanduser()
    pipeline = read_json(pipeline_manifest.resolve()) if str(pipeline_manifest) and pipeline_manifest.is_file() else None
    affected_changes = 0
    if isinstance(pipeline, dict):
        changed_entry = next((
            entry for entry in pipeline.get("period_results", [])
            if isinstance(entry, dict) and Path(str(entry.get("result") or "")).expanduser().resolve() == result_path
        ), None)
        if changed_entry is not None:
            affected_changes = sum(
                1 for entry in pipeline.get("change_results", [])
                if isinstance(entry, dict)
                and entry.get("grid") == changed_entry.get("grid")
                and changed_entry.get("period") in {entry.get("before_period"), entry.get("after_period")}
            )
    progress_total = 4 + affected_changes
    progress_completed = 0

    def report_progress(stage: str) -> None:
        emit(
            "apply-edits", stage=stage, status="running",
            completed=progress_completed, total=progress_total,
        )

    products = Path(result["gpkg"]).resolve().parent
    products.mkdir(parents=True, exist_ok=True)
    canonical_centerlines = global_centerlines if global_edit else products / "edited_centerlines_stitched.gpkg"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(WIDTH), str(SAMROAD), env.get("PYTHONPATH", "")))
    if global_edit:
        report_progress("从权威全局中心线生成受影响切片并重建路面")
        prepare_command = [
            str(PYTHON), str(WIDTH / "production_workflow.py"), "apply-global-edit",
            "--review-dir", str(width_dir), "--edited-dir", str(edited_dir),
        ]
        for stem in affected_stems:
            prepare_command.extend(("--only-stem", stem))
        run_command(prepare_command, ROOT, env, "受影响切片中心线、测宽与道路面增量重建")
    else:
        report_progress("人工中心线全局拼接与受影响路面重建")
        run_command([
            str(PYTHON), str(WIDTH / "production_workflow.py"), "stitch-edited",
            "--review-dir", str(width_dir), "--edited-dir", str(edited_dir),
            "--output", str(canonical_centerlines), "--snap-tolerance", "1.5",
        ], ROOT, env, "人工中心线全局拼接与受影响路面重建")
    progress_completed += 1
    report_progress("编辑后受影响窗口重新测宽")
    finalize_command = [
        str(PYTHON), str(WIDTH / "finalize_review_results.py"),
        "--output-dir", str(width_dir), "--edited-dir", str(edited_dir),
        "--final-dir", str(final_dir),
    ]
    if global_edit and affected_stems:
        for stem in affected_stems:
            finalize_command.extend(("--only-stem", stem))
    run_command(finalize_command, ROOT, env, "编辑后受影响窗口重新测宽")
    progress_completed += 1
    report_progress("人工编辑成果重新导出")
    run_command([
        str(PYTHON), str(WIDTH / "production_workflow.py"), "export-final",
        "--final-dir", str(final_dir), "--image-dir", str(images),
        "--output", result["gpkg"], "--centerline-shp", result["centerlines"],
        "--surface-shp", result["surfaces"],
        "--visualization", str(products / "road_overview.png"),
        "--stitched-centerlines", str(canonical_centerlines),
    ], ROOT, env, "人工编辑成果重新导出")
    progress_completed += 1

    result = _ensure_extract_manifest_fields(result)
    result["fusion"] = build_fusion_metadata(final_dir)
    result["manual_edit"] = {
        "applied": True,
        "edited_directory": str(edited_dir),
        "stitched_centerlines": str(canonical_centerlines),
        "canonical_centerlines": str(canonical_centerlines),
        "canonical_centerlines_authoritative": bool(global_edit),
        "editing_scope": edit_manifest.get("editing_scope", "tile_local"),
        "affected_tiles": affected_stems,
        "affected_tile_count": len(affected_stems),
        "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result["change_reruns"] = []
    result["change_rerun_count"] = 0
    write_json(result_path, result)

    if isinstance(pipeline, dict):
        pipeline_manifest = pipeline_manifest.resolve()
        changed_entry = None
        for entry in pipeline.get("period_results", []):
            try:
                same_result = Path(entry.get("result", "")).resolve() == result_path
            except (OSError, TypeError, ValueError):
                same_result = False
            if same_result:
                preserved = {key: entry.get(key) for key in ("grid", "period", "source", "analysis_source", "result")}
                entry.update(result)
                entry.update({key: value for key, value in preserved.items() if value is not None})
                changed_entry = entry
                break
        if changed_entry is not None:
            result_by_period = {
                (entry.get("grid"), entry.get("period")): entry
                for entry in pipeline.get("period_results", [])
            }
            changed_grid, changed_period = changed_entry.get("grid"), changed_entry.get("period")
            for change_entry in pipeline.get("change_results", []):
                if change_entry.get("grid") != changed_grid or changed_period not in {
                    change_entry.get("before_period"), change_entry.get("after_period"),
                }:
                    continue
                before_period, after_period = change_entry.get("before_period"), change_entry.get("after_period")
                before_entry = result_by_period.get((changed_grid, before_period))
                after_entry = result_by_period.get((changed_grid, after_period))
                if not before_entry or not after_entry:
                    raise RuntimeError(f"变化对缺少期次结果，无法在编辑后重跑：{before_period} -> {after_period}")
                _invalidate_change_evaluation(
                    pipeline, str(changed_grid), str(before_period), str(after_period),
                )
                _persist_existing_pipeline(pipeline, pipeline_manifest)
                emit(
                    "pipeline", stage="编辑后变化检测重跑", status="running",
                    grid=str(changed_grid), before_period=str(before_period),
                    after_period=str(after_period), completed=0, total=1,
                    area_completed=0, area_total=1,
                )
                report_progress(f"重跑相邻变化：{before_period} → {after_period}")
                output_value = str(change_entry.get("output") or "").strip()
                if output_value:
                    change_output = Path(output_value).expanduser()
                else:
                    change_output = Path(pipeline["job_root"]) / "grids" / clean_name(str(changed_grid)) / "changes" / (
                        f"{clean_name(str(before_period))}_to_{clean_name(str(after_period))}"
                    )
                rerun = _ensure_change_manifest_fields(change(argparse.Namespace(
                    before_result=before_entry["result"], after_result=after_entry["result"],
                    output=str(change_output), before_period=before_period, after_period=after_period,
                    absolute=change_entry.get("absolute", pipeline.get("absolute", "2.0")),
                    ratio=change_entry.get("ratio", pipeline.get("ratio", "0.2")),
                    tolerance=change_entry.get("tolerance", pipeline.get("tolerance", "3.0")),
                    truth=change_entry.get("truth", ""),
                    validation_area=change_entry.get("validation_area", pipeline.get("validation_area", "")),
                    truth_type_field=change_entry.get("truth_type_field", pipeline.get("truth_type_field", "")),
                )), change_output)
                preserved_change = {key: change_entry.get(key) for key in ("grid", "before_period", "after_period", "truth", "validation_area", "truth_type_field")}
                change_entry.update(rerun)
                change_entry.update({key: value for key, value in preserved_change.items() if value not in (None, "")})
                change_entry.pop("evaluation_metrics", None)
                change_entry.pop("evaluated_at", None)
                change_entry["evaluation_stale"] = True
                change_entry["evaluation_state"] = "stale"
                change_entry["evaluation_stale_at"] = now_text()
                result["change_reruns"].append({
                    "before_period": before_period,
                    "after_period": after_period,
                    "output": change_entry.get("output", str(change_output)),
                    "evaluation_metrics": change_entry.get("evaluation_metrics"),
                })
                progress_completed += 1
                _persist_existing_pipeline(pipeline, pipeline_manifest)
                emit(
                    "pipeline", stage="编辑后变化检测重跑", status="complete",
                    grid=str(changed_grid), before_period=str(before_period),
                    after_period=str(after_period), completed=1, total=1,
                    area_completed=1, area_total=1,
                )

        result["change_rerun_count"] = len(result["change_reruns"])
        if changed_entry is not None:
            report_progress("更新长时序道路属性表")
            changed_entry["change_reruns"] = list(result["change_reruns"])
            changed_entry["change_rerun_count"] = result["change_rerun_count"]
            pipeline["temporal_results"] = build_temporal_outputs(
                pipeline, Path(pipeline.get("job_root") or pipeline_manifest.parent),
            )
        write_json(result_path, result)

        _persist_existing_pipeline(pipeline, pipeline_manifest)
    progress_completed = progress_total
    emit(
        "apply-edits", stage="人工编辑增量重建", status="complete",
        completed=progress_completed, total=progress_total,
    )
    emit("complete", stage="apply-edits", **result)
    return result


def _period_result_ready(entry: dict) -> bool:
    """Return whether a prior period result is complete enough to resume from."""
    status = str(entry.get("status") or "").casefold()
    if status and status != "completed":
        return False
    try:
        result_path = Path(str(entry.get("result") or "")).expanduser()
        if not result_path.is_file():
            return False
        result = read_json(result_path)
        return all(
            Path(str(result.get(key) or "")).expanduser().is_file()
            for key in ("centerlines", "surfaces", "gpkg")
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _change_result_ready(entry: dict) -> bool:
    """Require the summary and, for Fast, its single formal change Shapefile."""
    status = str(entry.get("status") or "").casefold()
    if status and status != "completed":
        return False
    try:
        if not Path(str(entry.get("summary") or "")).expanduser().is_file():
            return False
        if str(entry.get("execution_profile") or "").casefold() != "fast":
            return True
        layers = entry.get("layers") if isinstance(entry.get("layers"), dict) else {}
        changes = Path(str(
            layers.get("changes")
            or entry.get("road_changes")
            or Path(str(entry.get("output") or "")) / "road_changes.shp"
        )).expanduser()
        return changes.is_file()
    except (OSError, ValueError, TypeError):
        return False


def _temporal_result_ready(entry: dict) -> bool:
    """Return whether every formal long-term output is still available."""
    status = str(entry.get("status") or "").casefold()
    if status and status != "completed":
        return False
    try:
        return all(
            Path(str(entry.get(key) or "")).expanduser().is_file()
            for key in (
                "life_shp", "observations_shp", "events_shp",
                "event_parts_shp", "lineage_shp", "review_shp",
            )
        )
    except (OSError, ValueError, TypeError):
        return False


def _evaluation_summary_ready(summary: object) -> bool:
    if not isinstance(summary, dict):
        return False
    try:
        return all(
            Path(str(summary.get(key) or "")).expanduser().is_file()
            for key in ("csv", "json")
        )
    except (OSError, ValueError, TypeError):
        return False


def _normalized_sources_ready(periods: dict[str, Path], output_root: Path) -> dict[str, Path] | None:
    marker = output_root / "normalization_complete.json"
    if not marker.is_file():
        return None
    try:
        marker_value = read_json(marker)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    stored_periods = marker_value.get("periods", {})
    if set(stored_periods) != set(periods):
        return None
    ready: dict[str, Path] = {}
    for period in periods:
        directory = Path(str(stored_periods.get(period) or "")).expanduser()
        if not directory.is_absolute():
            directory = output_root / directory
        rasters = [
            path for path in listed_rasters(directory)
            if path.name.startswith("v") and ".partial." not in path.name
        ]
        if not rasters:
            return None
        ready[period] = directory.resolve()
    return ready


def _task_input_spec(
    mode: str,
    grids: dict[str, dict[str, Path]],
    validation_area: str | dict[str, str],
    truth_by_pair: dict[tuple, Path],
    args: argparse.Namespace,
) -> dict:
    def fingerprint(source: Path) -> dict:
        source = Path(source).expanduser()
        files = listed_rasters(source)
        stats = []
        for path in files:
            try:
                value = path.stat()
            except OSError:
                continue
            stats.append((value.st_size, value.st_mtime_ns))
        try:
            source_mtime = source.stat().st_mtime_ns
        except OSError:
            source_mtime = None
        return {
            "path": str(source),
            "file_count": len(files),
            "total_bytes": sum(value[0] for value in stats),
            "latest_mtime_ns": max((value[1] for value in stats), default=None),
            "source_mtime_ns": source_mtime,
        }

    hash_cache = project_layout(Path(args.output_root)).cache_root / "dependency_hashes.json"

    def file_fingerprint(value: str | Path) -> dict:
        return stable_file_identity(value)

    return {
        "pipeline_version": PIPELINE_VERSION,
        "mode": mode,
        "execution_profile": str(getattr(args, "execution_profile", "full") or "full"),
        "validation_area": (
            {key: file_fingerprint(value) for key, value in validation_area.items()}
            if isinstance(validation_area, dict) else
            (file_fingerprint(validation_area) if validation_area else None)
        ),
        "grids": {
            grid: {period: fingerprint(source) for period, source in periods.items()}
            for grid, periods in grids.items()
        },
        "truths": {
            "\u0000".join(str(value) for value in key): file_fingerprint(path)
            for key, path in truth_by_pair.items()
        },
        "checkpoint": stable_file_identity(
            args.checkpoint, content_hash=True, cache_path=hash_cache,
        ),
        "config": (
            effective_config_identity(
                args.config, hash_resources=True, cache_path=hash_cache,
            )
            if Path(args.config).expanduser().is_file()
            else file_fingerprint(args.config)
        ),
        "device": str(args.device),
        "pixel_size": str(args.pixel_size),
        "rescale": str(args.rescale),
        "junction_node_mode": str(getattr(args, "junction_node_mode", "sparse") or "sparse"),
        "absolute": str(args.absolute),
        "ratio": str(args.ratio),
        "tolerance": str(args.tolerance),
        "truth_type_field": str(getattr(args, "truth_type_field", "") or ""),
        "evaluation_enabled": (
            not bool(getattr(args, "no_evaluation", False))
            and (
                str(getattr(args, "execution_profile", "full") or "full") != "fast"
                or bool(truth_by_pair)
            )
        ),
    }


def dependency_invalidation_plan(prior: dict, current: dict) -> dict:
    """Describe the minimum stages invalidated by a changed frozen input spec."""
    current_grids = current.get("grids") or {}
    all_periods = {
        (str(grid), str(period))
        for grid, periods in current_grids.items()
        for period in (periods or {})
    }
    scalar_extraction_keys = (
        "pipeline_version", "mode", "device", "pixel_size", "rescale",
        "junction_node_mode", "execution_profile",
    )

    def identity_tree_equal(previous, latest) -> bool:
        if isinstance(previous, dict) and isinstance(latest, dict):
            if "path" in previous or "path" in latest:
                return ordinary_file_identity_equal(previous, latest)
            if set(previous) != set(latest):
                return False
            return all(identity_tree_equal(previous[key], latest[key]) for key in previous)
        if isinstance(previous, list) and isinstance(latest, list):
            return len(previous) == len(latest) and all(
                identity_tree_equal(first, second)
                for first, second in zip(previous, latest)
            )
        return previous == latest

    def scalar_value(payload: dict, key: str):
        if key == "execution_profile":
            return str(payload.get(key, "full") or "full")
        return payload.get(key)

    invalidate_all_extraction = (
        any(scalar_value(prior, key) != scalar_value(current, key) for key in scalar_extraction_keys)
        or not identity_tree_equal(prior.get("validation_area"), current.get("validation_area"))
        or not dependency_identity_equal(
            prior.get("checkpoint"), current.get("checkpoint"), kind="checkpoint",
        )
        or not dependency_identity_equal(
            prior.get("config"), current.get("config"), kind="config",
        )
    )
    changed_periods = set(all_periods if invalidate_all_extraction else ())
    if not invalidate_all_extraction:
        prior_grids = prior.get("grids") or {}
        for grid, periods in current_grids.items():
            previous = prior_grids.get(grid) or {}
            for period, fingerprint in (periods or {}).items():
                if not identity_tree_equal(previous.get(period), fingerprint):
                    changed_periods.add((str(grid), str(period)))
        for grid, periods in prior_grids.items():
            for period in (periods or {}):
                if period not in (current_grids.get(grid) or {}):
                    changed_periods.add((str(grid), str(period)))
    threshold_changed = any(prior.get(key) != current.get(key) for key in ("absolute", "ratio", "tolerance"))
    truth_changed = (
        not identity_tree_equal(prior.get("truths"), current.get("truths"))
        or any(
            prior.get(key) != current.get(key)
            for key in ("truth_type_field", "evaluation_enabled")
        )
    )
    changed_pairs = set()
    for grid, periods in current_grids.items():
        names = sorted((periods or {}).keys(), key=period_sort_key)
        for before, after in zip(names, names[1:]):
            if threshold_changed or truth_changed or (str(grid), str(before)) in changed_periods or (str(grid), str(after)) in changed_periods:
                changed_pairs.add((str(grid), str(before), str(after)))
    return {
        "periods": sorted(changed_periods),
        "changes": sorted(changed_pairs),
        "threshold_changed": threshold_changed,
        "truth_changed": truth_changed,
        "reuse_all": not changed_periods and not changed_pairs,
    }


def _validate_resume_execution_profile(prior: dict, requested_profile: str) -> None:
    spec = prior.get("input_spec") if isinstance(prior.get("input_spec"), dict) else {}
    frozen = str(
        prior.get("execution_profile") or spec.get("execution_profile") or "full"
    ).strip().casefold()
    requested = str(requested_profile or "full").strip().casefold()
    if frozen != requested:
        raise ValueError(
            "不能使用不同的快速/标准模式继续同一任务。"
            "请恢复该任务原模式，或选择“从头重新运行完整流程”创建新任务。"
        )


def _enrich_relocated_input_spec(input_spec: dict, job_root: Path) -> dict:
    """Add stable identities obtainable from a copied task's frozen runtime files."""
    enriched = dict(input_spec)
    config = dict(enriched.get("config") or {})
    if not config.get("effective_sha256"):
        candidates = sorted(
            job_root.glob("grids/*/periods/*/runs/*/inference/road_graphs/config.yaml"),
            key=str,
        )
        for candidate in candidates:
            try:
                saved_identity = effective_config_identity(
                    candidate, inspect_resources=False,
                )
            except (OSError, UnicodeError, ValueError):
                continue
            config.update({
                "effective_sha256": saved_identity["effective_sha256"],
                "effective_key_count": saved_identity["effective_key_count"],
                "identity_source": str(candidate),
            })
            break
    if config:
        enriched["config"] = config
    return enriched


def _refresh_resume_external_paths(
    prior: dict, *, grids: dict[str, dict[str, Path]],
    validation_area: str | dict[str, str], truth_by_pair: dict[tuple, Path],
    source_root: Path | None, input_spec: dict, historical_input_spec: dict,
) -> None:
    """Keep old external paths only as provenance; never as active resume inputs."""
    provenance = dict(prior.get("provenance") or {})
    provenance.setdefault("historical_input_spec", historical_input_spec)
    provenance.setdefault("historical_external_paths", {
        "source_root": prior.get("source_root"),
        "validation_area": prior.get("validation_area"),
        "validation_areas": prior.get("validation_areas"),
    })
    provenance["historical_only"] = True
    prior["provenance"] = provenance
    prior["input_spec"] = input_spec
    prior["source_root"] = str(source_root) if source_root is not None else ""
    prior["validation_area"] = (
        next(iter(validation_area.values()))
        if isinstance(validation_area, dict) and len(validation_area) == 1
        else validation_area
    )
    prior["validation_areas"] = validation_area if isinstance(validation_area, dict) else {}
    for entry in prior.get("period_results", []) or []:
        if not isinstance(entry, dict):
            continue
        source = (grids.get(str(entry.get("grid"))) or {}).get(str(entry.get("period")))
        if source is None:
            continue
        entry["source"] = str(source)
        analysis_text = str(entry.get("analysis_source") or "").strip()
        if not analysis_text or not Path(analysis_text).expanduser().exists():
            entry["analysis_source"] = str(source)
    for entry in prior.get("change_results", []) or []:
        if not isinstance(entry, dict):
            continue
        key = (
            str(entry.get("grid")), str(entry.get("before_period")),
            str(entry.get("after_period")),
        )
        entry["truth"] = str(truth_by_pair.get(key, ""))


def _refresh_relocated_runtime_metadata(job_root: Path, args: argparse.Namespace) -> None:
    """Remove active references to old runtime locations from copied metadata."""
    current = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
    }
    for path in job_root.glob(
        "grids/*/periods/*/runs/*/inference/road_graphs/inference_metadata.json"
    ):
        try:
            metadata = read_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        historical = {
            key: metadata.get(key) for key in current
            if metadata.get(key) and metadata.get(key) != current[key]
        }
        if historical:
            provenance = dict(metadata.get("provenance") or {})
            provenance.setdefault("historical_runtime_paths", historical)
            provenance["historical_only"] = True
            metadata["provenance"] = provenance
        metadata.update(current)
        write_json(path, metadata)


def check_runtime_environment(args: argparse.Namespace, output_root: Path) -> dict:
    """Check run-only concerns separately from project data validation."""
    checkpoint = Path(str(args.checkpoint)).expanduser().resolve()
    config = Path(str(args.config)).expanduser().resolve()
    missing = [str(path) for path in (checkpoint, config) if not path.is_file()]
    if missing:
        raise FileNotFoundError("运行所需模型或配置不存在：\n" + "\n".join(missing))
    required = (SAMROAD / "inferencer.py", MOLRA / "infer_img.py", WIDTH / "road_change_detection.py")
    engine_missing = [str(path) for path in required if not path.is_file()]
    if engine_missing:
        raise FileNotFoundError("运行引擎文件不完整：\n" + "\n".join(engine_missing))
    import torch
    requested = str(getattr(args, "device", "auto") or "auto").casefold()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已选择 CUDA，但当前运行环境没有可用 CUDA 设备。")
    existing = output_root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.exists():
        raise FileNotFoundError(f"无法定位成果输出目录所在磁盘：{output_root}")
    return {
        "checkpoint": str(checkpoint), "config": str(config),
        "device": requested, "cuda_available": bool(torch.cuda.is_available()),
        "output_root": str(output_root), "output_free_bytes": shutil.disk_usage(existing).free,
    }


def build_preflight_report(
    mode: str,
    grids: dict[str, dict[str, Path]],
    validation_area: str | dict[str, str],
    truth_by_pair: dict[tuple, Path],
    output_root: Path,
    *, include_runtime_checks: bool = True,
) -> dict:
    """Inspect the complete task without creating outputs or starting models."""
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as exc:
        raise RuntimeError("输入预检需要本项目内的 rasterio/numpy 环境") from exc

    area_paths = validation_area.values() if isinstance(validation_area, dict) else ([validation_area] if validation_area else [])
    for area_path in area_paths:
        _require_shapefile_components(Path(area_path), "验证区")
    for truth_path in truth_by_pair.values():
        _validate_truth_shapefile(Path(truth_path))

    image_count = 0
    input_bytes = 0
    details = []
    warnings = []
    all_metadata = []
    for grid, periods in grids.items():
        for period, source in periods.items():
            rasters = listed_rasters(source)
            period_bytes = sum(path.stat().st_size for path in rasters)
            image_count += len(rasters)
            input_bytes += period_bytes
            crs_values = set()
            band_values = set()
            dtype_values = set()
            resolutions = []
            for raster in rasters:
                with rasterio.open(raster) as dataset:
                    if dataset.crs is None:
                        warnings.append(f"影像缺少 CRS：{raster}")
                        crs_text = ""
                    else:
                        crs_text = dataset.crs.to_string()
                        crs_values.add(crs_text)
                    band_values.add(dataset.count)
                    dtype_values.update(dataset.dtypes)
                    resolutions.append((abs(dataset.res[0]), abs(dataset.res[1])))
                    all_metadata.append({
                        "grid": grid,
                        "path": raster,
                        "crs": dataset.crs,
                        "width": dataset.width,
                        "height": dataset.height,
                        "count": dataset.count,
                        "dtype": dataset.dtypes[0],
                        "bounds": dataset.bounds,
                    })
            if len(band_values) > 1 or len(dtype_values) > 1:
                warnings.append(f"{grid}/{period} 的波段数或数据类型不一致")
            details.append({
                "grid": grid,
                "period": period,
                "source": str(source),
                "image_count": len(rasters),
                "input_bytes": period_bytes,
                "crs": sorted(crs_values),
                "band_counts": sorted(band_values),
                "dtypes": sorted(dtype_values),
                "resolutions": resolutions[:10],
            })

    if mode == "validation":
        from shapely.geometry import box
        from shapely.ops import unary_union
        area_map = validation_area if isinstance(validation_area, dict) else {next(iter(grids)): validation_area}
        for area_id, area_path in area_map.items():
            area = _read_validation_area(Path(area_path))
            area_geometry = area.geometry.unary_union
            for period in grids.get(area_id, {}):
                footprints = []
                for item in all_metadata:
                    if item["grid"] != area_id or Path(item["path"]) not in listed_rasters(grids[area_id][period]):
                        continue
                    bounds = transform_bounds(item["crs"], area.crs, *item["bounds"], densify_pts=21)
                    footprints.append(box(*bounds))
                covered = unary_union(footprints) if footprints else None
                if covered is None or not covered.intersects(area_geometry):
                    raise ValueError(f"{area_id} / {period} 期影像与验证区没有空间交集。")
                missing_area = max(0.0, float(area_geometry.area - covered.intersection(area_geometry).area))
                if area_geometry.area > 0 and missing_area / float(area_geometry.area) > 0.01:
                    warnings.append(f"{area_id}/{period} 的影像边界未覆盖约 {missing_area / float(area_geometry.area):.1%} 验证区；请确认 NoData 和实际覆盖。")

    estimated_grid = None
    estimated_grids = []
    estimated_normalized_bytes = None
    estimated_streaming_memory_bytes = None
    if mode == "validation" and all_metadata:
        area_map = validation_area if isinstance(validation_area, dict) else {next(iter(grids)): validation_area}
        estimated_normalized_bytes = 0
        estimated_streaming_memory_bytes = 0
        for area_id, area_path in area_map.items():
            area_metadata = [item for item in all_metadata if item["grid"] == area_id]
            if not area_metadata:
                continue
            area = _read_validation_area(Path(area_path))
            first = area_metadata[0]
            if first["crs"] is None:
                continue
            analysis_crs = first["crs"]
            analysis_area = area.to_crs(analysis_crs)
            transformed_resolutions = []
            for item in area_metadata:
                if item["crs"] is None:
                    continue
                left, bottom, right, top = transform_bounds(
                    item["crs"], analysis_crs, *item["bounds"], densify_pts=21,
                )
                transformed_resolutions.append((
                    abs((right - left) / item["width"]),
                    abs((top - bottom) / item["height"]),
                ))
            if transformed_resolutions:
                xres = min(value[0] for value in transformed_resolutions)
                yres = min(value[1] for value in transformed_resolutions)
                left, bottom, right, top = analysis_area.total_bounds
                width = max(1, int(np.ceil((right - left) / xres)))
                height = max(1, int(np.ceil((top - bottom) / yres)))
                bands = int(first["count"])
                item_size = int(np.dtype(first["dtype"]).itemsize)
                period_count = len(grids.get(area_id, {}))
                grid_estimate = {
                    "grid": area_id,
                    "crs": analysis_crs.to_string(), "width": width, "height": height,
                    "x_resolution": xres, "y_resolution": yres,
                }
                estimated_grids.append(grid_estimate)
                estimated_normalized_bytes += width * height * bands * item_size * period_count
                tile_edge = min(4096, max(width, height))
                estimated_streaming_memory_bytes = max(
                    estimated_streaming_memory_bytes,
                    tile_edge * tile_edge * (bands * item_size * 2 + 4),
                )
        estimated_grid = estimated_grids[0] if estimated_grids else None

    free_bytes = None
    if include_runtime_checks:
        existing = output_root
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        free_bytes = shutil.disk_usage(existing).free if existing.exists() else None
        if estimated_normalized_bytes and free_bytes is not None and free_bytes < estimated_normalized_bytes * 2:
            warnings.append("输出盘剩余空间低于规范化影像估算量的 2 倍，建议更换磁盘或拆分任务")
    report = {
        "mode": mode,
        "grid_count": len(grids),
        "period_count": sum(len(periods) for periods in grids.values()),
        "change_count": sum(max(0, len(periods) - 1) for periods in grids.values()),
        "image_count": image_count,
        "truth_count": len(truth_by_pair),
        "input_bytes": input_bytes,
        "output_free_bytes": free_bytes,
        "estimated_analysis_grid": estimated_grid,
        "estimated_analysis_grids": estimated_grids,
        "estimated_normalized_bytes": estimated_normalized_bytes,
        "estimated_streaming_memory_bytes": estimated_streaming_memory_bytes,
        "warnings": warnings,
        "periods": details,
        "period_orders": {
            grid: period_order_manifest(periods.keys()) for grid, periods in grids.items()
        },
        "checked_at": now_text(),
    }
    return report


def _persist_pipeline(manifest: dict, job_root: Path, output_root: Path) -> None:
    manifest["updated_at"] = now_text()
    write_json(job_root / "job_state.json", manifest)
    write_json(job_root / "pipeline_result.json", manifest)
    layout = project_layout(output_root, manifest.get("project_root"))
    write_json(layout.latest_pipeline_path, manifest)


def _persist_existing_pipeline(manifest: dict, manifest_path: Path) -> None:
    job_value = str(manifest.get("job_root") or "").strip()
    job_root = Path(job_value or manifest_path.parent).expanduser().resolve()
    output_value = str(manifest.get("output_root") or "").strip()
    output_root = Path(
        output_value or (job_root.parent if job_value else manifest_path.parent)
    ).expanduser().resolve()
    project_value = str(manifest.get("project_root") or "").strip()
    if not project_value:
        project_value = str(
            output_root.parent
            if output_root.name.startswith("04_") or job_value
            else output_root
        )
        manifest["project_root"] = project_value
    manifest.setdefault("output_root", str(output_root))
    publisher = ResultPublisher(output_root, project_root=project_value)
    publisher.publish_manifest(manifest, source_manifest=manifest_path)
    _write_task_report(manifest, job_root)
    publisher.publish_reports(job_root)
    _persist_pipeline(manifest, job_root, output_root)
    latest_path = project_layout(output_root, manifest.get("project_root")).latest_pipeline_path
    if manifest_path.resolve() not in {
        (job_root / "job_state.json").resolve(),
        (job_root / "pipeline_result.json").resolve(),
        latest_path.resolve(),
    }:
        write_json(manifest_path, manifest)


def _manifest_period_plan(manifest: dict) -> dict[str, list[str]]:
    """Return the frozen task plan, never an order inferred across failed periods."""
    plan: dict[str, list[str]] = {}
    for raw_grid, raw_order in (manifest.get("period_orders") or {}).items():
        if not isinstance(raw_order, dict):
            continue
        values = [str(value).strip() for value in raw_order.get("period_order", [])]
        values = [value for value in values if value]
        if values:
            plan[str(raw_grid)] = list(dict.fromkeys(values))
    spec = manifest.get("input_spec") if isinstance(manifest.get("input_spec"), dict) else {}
    for raw_grid, periods in (spec.get("grids") or {}).items():
        if str(raw_grid) in plan or not isinstance(periods, dict):
            continue
        plan[str(raw_grid)] = sorted(
            (str(period) for period in periods), key=period_sort_key,
        )
    if plan:
        return plan
    for entry in manifest.get("period_results", []) or []:
        if not isinstance(entry, dict):
            continue
        grid = str(entry.get("grid") or "").strip()
        period = str(entry.get("period") or "").strip()
        if grid and period:
            plan.setdefault(grid, []).append(period)
    return {
        grid: sorted(set(periods), key=period_sort_key)
        for grid, periods in plan.items()
    }


def _manifest_identity_path(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("path") or "").strip()
    return str(value or "").strip()


def _manifest_period_source(manifest: dict, grid: str, period: str) -> str:
    spec = manifest.get("input_spec") if isinstance(manifest.get("input_spec"), dict) else {}
    grids = spec.get("grids") if isinstance(spec.get("grids"), dict) else {}
    periods = grids.get(grid) if isinstance(grids.get(grid), dict) else {}
    return _manifest_identity_path(periods.get(period))


def _manifest_analysis_source(
    manifest: dict, grid: str, period: str, old: dict,
) -> Path:
    for value in (old.get("analysis_source"), old.get("source")):
        source = Path(str(value or "")).expanduser()
        if str(value or "").strip() and source.exists():
            return source.resolve()
    job_root = Path(str(manifest.get("job_root") or ".")).expanduser().resolve()
    marker = job_root / "validation_inputs" / clean_name(grid) / "normalization_complete.json"
    if marker.is_file():
        try:
            stored = str((read_json(marker).get("periods") or {}).get(period) or "").strip()
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            stored = ""
        if stored:
            source = Path(stored).expanduser()
            if not source.is_absolute():
                source = marker.parent / source
            if source.exists():
                return source.resolve()
    source_value = _manifest_period_source(manifest, grid, period)
    source = Path(source_value).expanduser()
    if source_value and source.exists():
        return source.resolve()
    raise FileNotFoundError(f"找不到任务计划中的期次输入：{grid} / {period}")


def _rerun_period_entry(manifest: dict, grid: str, period: str) -> dict:
    plan = _manifest_period_plan(manifest)
    if period not in plan.get(grid, []):
        raise ValueError(f"任务计划中不存在期次：{grid} / {period}")
    entries = manifest.get("period_results", []) or []
    index = next((
        number for number, entry in enumerate(entries)
        if isinstance(entry, dict) and str(entry.get("grid")) == grid
        and str(entry.get("period")) == period
    ), None)
    old = entries[index] if index is not None else {}
    job_root = Path(str(manifest.get("job_root") or ".")).expanduser().resolve()
    workspace = Path(str(old.get("result") or (
        job_root / "grids" / clean_name(grid) / "periods" / clean_name(period) / "latest_result.json"
    ))).expanduser().resolve().parent
    analysis_source = _manifest_analysis_source(manifest, grid, period, old)
    input_spec = manifest.get("input_spec") or {}
    checkpoint = str((input_spec.get("checkpoint") or {}).get("path") or "")
    config = str((input_spec.get("config") or {}).get("path") or "")
    if not checkpoint or not config:
        raise ValueError("旧任务索引缺少模型或推理配置路径，无法局部重跑。")
    started = time.monotonic()
    prepare(argparse.Namespace(source=str(analysis_source), workspace=str(workspace)))
    result = _ensure_extract_manifest_fields(extract(argparse.Namespace(
        workspace=str(workspace), source="", checkpoint=checkpoint, config=config,
        device=str(input_spec.get("device") or "auto"),
        pixel_size=str(input_spec.get("pixel_size") or "0.0"),
        rescale=str(input_spec.get("rescale") or "off"),
        run_id=f"roads_rerun_{int(time.time())}",
        junction_node_mode=str(input_spec.get("junction_node_mode") or "sparse"),
        execution_profile=str(manifest.get("execution_profile") or input_spec.get("execution_profile") or "full"),
        validation_area=str((manifest.get("validation_areas") or {}).get(grid) or manifest.get("validation_area") or ""),
    )))
    updated = {
        **old, **result, "grid": grid, "period": period,
        "source": _manifest_period_source(manifest, grid, period) or str(old.get("source") or analysis_source),
        "analysis_source": str(analysis_source),
        "result": str(workspace / "latest_result.json"), "status": "completed",
        "rerun_at": now_text(), "elapsed_seconds": elapsed_seconds(started),
    }
    if index is None:
        entries.append(updated)
    else:
        entries[index] = updated
    manifest["period_results"] = entries
    return updated


def _manifest_change_context(
    manifest: dict, grid: str, before: str, after: str, old: dict,
) -> tuple[str, str, str]:
    spec = manifest.get("input_spec") if isinstance(manifest.get("input_spec"), dict) else {}
    truths = spec.get("truths") if isinstance(spec.get("truths"), dict) else {}
    truth = ""
    for key in (f"{grid}\0{before}\0{after}", f"{before}\0{after}"):
        if key in truths:
            truth = _manifest_identity_path(truths[key])
            break
    if not truth:
        truth = str(old.get("truth") or "").strip()
    validation_areas = manifest.get("validation_areas")
    validation = str(
        (validation_areas.get(grid) or "") if isinstance(validation_areas, dict) else ""
    ).strip()
    if not validation:
        configured = spec.get("validation_area")
        if isinstance(configured, dict) and grid in configured:
            validation = _manifest_identity_path(configured[grid])
        elif not isinstance(configured, dict):
            validation = _manifest_identity_path(configured)
    if not validation:
        validation = str(old.get("validation_area") or manifest.get("validation_area") or "").strip()
    truth_type_field = str(
        manifest.get("truth_type_field") or spec.get("truth_type_field")
        or old.get("truth_type_field") or ""
    ).strip()
    return truth, validation, truth_type_field


def _invalidate_change_evaluation(
    manifest: dict, grid: str, before: str, after: str, *, status: str = "running",
) -> dict | None:
    """Invalidate metrics before replacing a change result in the active task."""
    entry = next((
        value for value in (manifest.get("change_results", []) or [])
        if isinstance(value, dict)
        and str(value.get("grid")) == grid
        and str(value.get("before_period")) == before
        and str(value.get("after_period")) == after
    ), None)
    if entry is None:
        return None
    entry.pop("evaluation_metrics", None)
    entry.pop("evaluated_at", None)
    entry["evaluation_stale"] = True
    entry["evaluation_state"] = "stale"
    entry["evaluation_stale_at"] = now_text()
    entry["status"] = status
    manifest.pop("evaluation_summary", None)
    manifest["updated_at"] = now_text()
    return entry


def _mark_change_rerun_failed(
    manifest: dict, grid: str, before: str, after: str, error: Exception,
) -> None:
    entry = _invalidate_change_evaluation(
        manifest, grid, before, after, status="failed",
    )
    if entry is not None:
        entry["error"] = str(error)
        entry["failed_at"] = now_text()


def _rerun_change_entry(manifest: dict, grid: str, before: str, after: str) -> dict:
    periods = {
        (str(entry.get("grid")), str(entry.get("period"))): entry
        for entry in (manifest.get("period_results", []) or []) if isinstance(entry, dict)
    }
    before_entry, after_entry = periods.get((grid, before)), periods.get((grid, after))
    if before_entry is None or after_entry is None:
        raise ValueError(f"变化对缺少道路提取结果：{grid} / {before} → {after}")
    entries = manifest.get("change_results", []) or []
    index = next((
        number for number, entry in enumerate(entries)
        if isinstance(entry, dict) and str(entry.get("grid")) == grid
        and str(entry.get("before_period")) == before and str(entry.get("after_period")) == after
    ), None)
    old = entries[index] if index is not None else {}
    job_root = Path(str(manifest.get("job_root") or ".")).expanduser().resolve()
    output = Path(str(old.get("output") or job_root / "grids" / clean_name(grid) / "changes" / f"{clean_name(before)}_to_{clean_name(after)}"))
    spec = manifest.get("input_spec") or {}
    truth_value, validation_value, truth_type_field = _manifest_change_context(
        manifest, grid, before, after, old,
    )
    started = time.monotonic()
    if str(manifest.get("execution_profile") or "full") == "fast":
        result = _run_fast_change_result(
            Path(str(before_entry["result"])),
            Path(str(after_entry["result"])),
            output,
            before_period=before,
            after_period=after,
            position_tolerance=float(
                spec.get("tolerance") or old.get("tolerance") or 3.0
            ),
            width_change_absolute=float(
                spec.get("absolute") or old.get("absolute") or 2.0
            ),
            width_change_ratio=float(
                spec.get("ratio") or old.get("ratio") or 0.2
            ),
            truth_path=Path(truth_value).expanduser() if truth_value else None,
            validation_area=(
                Path(validation_value) if validation_value else None
            ),
            truth_type_field=str(
                truth_type_field
                or "BHBM"
            ),
        )
        result = _ensure_change_manifest_fields(result, output)
    else:
        result = _ensure_change_manifest_fields(change(argparse.Namespace(
            before_result=str(before_entry["result"]), after_result=str(after_entry["result"]),
            output=str(output), before_period=before, after_period=after,
            absolute=str(spec.get("absolute") or old.get("absolute") or "2.0"),
            ratio=str(spec.get("ratio") or old.get("ratio") or "0.2"),
            tolerance=str(spec.get("tolerance") or old.get("tolerance") or "3.0"),
            truth=truth_value, validation_area=validation_value,
            truth_type_field=truth_type_field,
        )), output)
    updated = {
        **old, **result, "grid": grid, "before_period": before, "after_period": after,
        "truth": truth_value, "validation_area": validation_value,
        "truth_type_field": truth_type_field,
        "absolute": str(spec.get("absolute") or old.get("absolute") or "2.0"),
        "ratio": str(spec.get("ratio") or old.get("ratio") or "0.2"),
        "tolerance": str(spec.get("tolerance") or old.get("tolerance") or "3.0"),
        "status": "completed", "rerun_at": now_text(), "elapsed_seconds": elapsed_seconds(started),
    }
    updated.pop("stale_reason", None)
    updated.pop("evaluation_metrics", None)
    updated.pop("evaluated_at", None)
    updated["evaluation_stale"] = True
    updated["evaluation_state"] = "stale"
    updated["evaluation_stale_at"] = now_text()
    if index is None:
        entries.append(updated)
    else:
        entries[index] = updated
    manifest["change_results"] = entries
    return updated


def _affected_manifest_pairs(manifest: dict, grid: str, period: str) -> list[tuple[str, str]]:
    names = _manifest_period_plan(manifest).get(grid, [])
    return [(before, after) for before, after in zip(names, names[1:]) if period in {before, after}]


def _refresh_manifest_downstream(manifest: dict) -> None:
    job_root = Path(str(manifest.get("job_root") or ".")).expanduser().resolve()
    if str(manifest.get("execution_profile") or "full") == "fast":
        manifest["temporal_results"] = []
        manifest["temporal_status"] = "skipped_fast_profile"
    else:
        manifest["temporal_results"] = build_temporal_outputs(manifest, job_root)
    aggregate_change_evaluations(manifest, job_root)
    manifest["downstream_updated_at"] = now_text()


def _evaluate_fast_manifest_pairs(
    manifest_path: Path, pairs: list[tuple[str, str, str]],
) -> None:
    manifest = read_json(manifest_path)
    if str(manifest.get("execution_profile") or "full") != "fast":
        return
    for grid, before, after in pairs:
        entry = next(
            item for item in manifest.get("change_results", [])
            if str(item.get("grid")) == grid
            and str(item.get("before_period")) == before
            and str(item.get("after_period")) == after
        )
        if not str(entry.get("truth") or "").strip():
            continue
        evaluate_existing_changes(argparse.Namespace(
            pipeline_manifest=str(manifest_path), grid=grid, before_period=before,
            after_period=after, truth=str(entry.get("truth") or ""),
            validation_area=str(entry.get("validation_area") or ""),
            truth_type_field=str(entry.get("truth_type_field") or manifest.get("truth_type_field") or "BHBM"),
            truth_added_value="", truth_width_changed_value="", truth_removed_value="",
            evaluation_tolerance=5.0,
        ))


def rerun_pipeline_period(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.pipeline_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    grid, period = str(args.grid), str(args.period)
    pairs = _affected_manifest_pairs(manifest, grid, period)
    emit("pipeline", stage="道路提取局部重跑", status="running", completed=0, total=1 + (len(pairs) if args.update_related else 0))
    updated = _rerun_period_entry(manifest, grid, period)
    if args.update_related:
        for index, (before, after) in enumerate(pairs, start=1):
            _invalidate_change_evaluation(manifest, grid, before, after)
            _persist_existing_pipeline(manifest, manifest_path)
            emit(
                "pipeline", stage="相关变化对更新", status="running",
                completed=index - 1, total=len(pairs), grid=grid,
                before_period=before, after_period=after,
                area_completed=index - 1, area_total=len(pairs),
            )
            try:
                _rerun_change_entry(manifest, grid, before, after)
            except Exception as exc:
                _mark_change_rerun_failed(manifest, grid, before, after, exc)
                _persist_existing_pipeline(manifest, manifest_path)
                raise
            _persist_existing_pipeline(manifest, manifest_path)
            emit(
                "pipeline", stage="相关变化对更新", status="complete",
                completed=index, total=len(pairs), grid=grid,
                before_period=before, after_period=after,
                area_completed=index, area_total=len(pairs),
            )
        _refresh_manifest_downstream(manifest)
    else:
        for before, after in pairs:
            entry = _invalidate_change_evaluation(
                manifest, grid, before, after, status="stale",
            )
            if entry is not None:
                entry["stale_reason"] = f"{period} 期道路成果已重跑"
        manifest["temporal_status"] = "stale"
    manifest["status"] = "completed"
    manifest["updated_at"] = now_text()
    _persist_existing_pipeline(manifest, manifest_path)
    if args.update_related:
        _evaluate_fast_manifest_pairs(
            manifest_path, [(grid, before, after) for before, after in pairs],
        )
    result = {"grid": grid, "period": period, "affected_pairs": pairs, "updated_related": bool(args.update_related), "result": updated.get("result")}
    emit("complete", stage="rerun-period", **result)
    return result


def rerun_pipeline_change(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.pipeline_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    grid, before, after = str(args.grid), str(args.before_period), str(args.after_period)
    _invalidate_change_evaluation(manifest, grid, before, after)
    _persist_existing_pipeline(manifest, manifest_path)
    emit(
        "pipeline", stage="变化检测重跑", status="running", completed=0, total=1,
        grid=grid, before_period=before, after_period=after,
        area_completed=0, area_total=1,
    )
    try:
        updated = _rerun_change_entry(manifest, grid, before, after)
    except Exception as exc:
        _mark_change_rerun_failed(manifest, grid, before, after, exc)
        _persist_existing_pipeline(manifest, manifest_path)
        emit(
            "pipeline", stage="变化检测重跑", status="failed", completed=0, total=1,
            grid=grid, before_period=before, after_period=after,
            area_completed=0, area_total=1, error=str(exc),
        )
        raise
    if args.update_temporal:
        _refresh_manifest_downstream(manifest)
    else:
        manifest["temporal_status"] = "stale"
    manifest["status"] = "completed"
    _persist_existing_pipeline(manifest, manifest_path)
    emit(
        "pipeline", stage="变化检测重跑", status="complete", completed=1, total=1,
        grid=grid, before_period=before, after_period=after,
        area_completed=1, area_total=1,
    )
    _evaluate_fast_manifest_pairs(manifest_path, [(grid, before, after)])
    result = {"grid": args.grid, "before_period": args.before_period, "after_period": args.after_period, "updated_temporal": bool(args.update_temporal), "output": updated.get("output")}
    emit("complete", stage="rerun-change", **result)
    return result


def rerun_all_pipeline_periods(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.pipeline_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    plan = _manifest_period_plan(manifest)
    period_keys = [
        (grid, period) for grid, periods in plan.items() for period in periods
    ]
    continue_on_error = bool(getattr(args, "continue_on_error", False))
    failures: list[dict] = []
    for index, (grid, period) in enumerate(period_keys, start=1):
        try:
            _rerun_period_entry(manifest, grid, period)
        except Exception as exc:
            failure = {
                "type": "period", "grid": grid, "scope": period,
                "status": "failed", "error": str(exc), "failed_at": now_text(),
            }
            failures.append(failure)
            emit("failure", **failure)
            if not continue_on_error:
                manifest["rerun_all_failures"] = failures
                manifest["status"] = "failed"
                _persist_existing_pipeline(manifest, manifest_path)
                raise
        emit("pipeline", stage="批量道路提取重跑", status="running", completed=index, total=len(period_keys))
    change_keys = _all_manifest_change_pairs(manifest)
    completed_changes: list[tuple[str, str, str]] = []
    area_totals = {
        area: sum(1 for key in change_keys if key[0] == area)
        for area in {key[0] for key in change_keys}
    }
    area_completed: dict[str, int] = {area: 0 for area in area_totals}
    for change_index, (grid, before, after) in enumerate(change_keys, start=1):
        pair_status = "complete"
        _invalidate_change_evaluation(manifest, grid, before, after)
        _persist_existing_pipeline(manifest, manifest_path)
        emit(
            "pipeline", stage="道路提取后的变化更新", status="running",
            completed=change_index - 1, total=len(change_keys), grid=grid,
            before_period=before, after_period=after,
            area_completed=area_completed[grid], area_total=area_totals[grid],
        )
        try:
            _rerun_change_entry(manifest, grid, before, after)
            completed_changes.append((grid, before, after))
            area_completed[grid] += 1
            _persist_existing_pipeline(manifest, manifest_path)
        except Exception as exc:
            pair_status = "failed"
            _mark_change_rerun_failed(manifest, grid, before, after, exc)
            failure = {
                "type": "change", "grid": grid, "scope": f"{before} -> {after}",
                "status": "failed", "error": str(exc), "failed_at": now_text(),
            }
            failures.append(failure)
            _persist_existing_pipeline(manifest, manifest_path)
            emit("failure", **failure)
            if not continue_on_error:
                manifest["rerun_all_failures"] = failures
                manifest["status"] = "failed"
                _persist_existing_pipeline(manifest, manifest_path)
                raise
        emit(
            "pipeline", stage="道路提取后的变化更新", status=pair_status,
            completed=change_index, total=len(change_keys), grid=grid,
            before_period=before, after_period=after,
            area_completed=area_completed[grid], area_total=area_totals[grid],
        )
    try:
        _refresh_manifest_downstream(manifest)
    except Exception as exc:
        failure = {
            "type": "downstream", "grid": "", "scope": "长时序/汇总",
            "status": "failed", "error": str(exc), "failed_at": now_text(),
        }
        failures.append(failure)
        emit("failure", **failure)
        if not continue_on_error:
            manifest["rerun_all_failures"] = failures
            manifest["status"] = "failed"
            _persist_existing_pipeline(manifest, manifest_path)
            raise
    manifest["failure_history"] = list(manifest.get("failure_history", [])) + list(manifest.get("failures", []))
    manifest["failures"] = failures
    manifest["rerun_all_failures"] = failures
    manifest["status"] = "completed_with_errors" if failures else "completed"
    manifest["period_count"] = len(manifest.get("period_results", []) or [])
    manifest["change_count"] = len(manifest.get("change_results", []) or [])
    _persist_existing_pipeline(manifest, manifest_path)
    _evaluate_fast_manifest_pairs(manifest_path, completed_changes)
    result = {
        "period_count": len(period_keys), "change_count": len(completed_changes),
        "failure_count": len(failures), "total_change_count": len(change_keys),
    }
    emit("complete", stage="rerun-all-periods", **result)
    return result


def _all_manifest_change_pairs(manifest: dict) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for grid, periods in _manifest_period_plan(manifest).items():
        pairs.extend((grid, before, after) for before, after in zip(periods, periods[1:]))
    return pairs


def rerun_all_pipeline_changes(args: argparse.Namespace) -> dict:
    """Rerun every adjacent change pair while reusing completed period results."""
    manifest_path = Path(args.pipeline_manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    change_keys = _all_manifest_change_pairs(manifest)
    continue_on_error = bool(getattr(args, "continue_on_error", False))
    failures: list[dict] = []
    completed_keys: list[tuple[str, str, str]] = []
    area_totals = {
        area: sum(1 for key in change_keys if key[0] == area)
        for area in {key[0] for key in change_keys}
    }
    area_completed: dict[str, int] = {area: 0 for area in area_totals}
    for index, (grid, before, after) in enumerate(change_keys, start=1):
        pair_status = "complete"
        _invalidate_change_evaluation(manifest, grid, before, after)
        _persist_existing_pipeline(manifest, manifest_path)
        emit(
            "pipeline", stage="批量变化检测重跑", status="running",
            completed=index - 1, total=len(change_keys), grid=grid,
            before_period=before, after_period=after,
            area_completed=area_completed[grid], area_total=area_totals[grid],
        )
        try:
            _rerun_change_entry(manifest, grid, before, after)
            completed_keys.append((grid, before, after))
            area_completed[grid] += 1
            _persist_existing_pipeline(manifest, manifest_path)
        except Exception as exc:
            pair_status = "failed"
            _mark_change_rerun_failed(manifest, grid, before, after, exc)
            failure = {
                "type": "change", "grid": grid, "scope": f"{before} -> {after}",
                "before_period": before, "after_period": after,
                "stage": "rerun-change", "status": "failed",
                "error": str(exc), "failed_at": now_text(),
            }
            failures.append(failure)
            _persist_existing_pipeline(manifest, manifest_path)
            emit("failure", **failure)
            if not continue_on_error:
                manifest["rerun_change_failures"] = failures
                manifest["status"] = "failed"
                manifest["updated_at"] = now_text()
                _persist_existing_pipeline(manifest, manifest_path)
                raise
        emit(
            "pipeline", stage="批量变化检测重跑", status=pair_status,
            completed=index, total=len(change_keys), grid=grid,
            before_period=before, after_period=after,
            area_completed=area_completed[grid], area_total=area_totals[grid],
        )
    try:
        _refresh_manifest_downstream(manifest)
    except Exception as exc:
        failure = {
            "type": "downstream", "grid": "", "scope": "长时序/汇总",
            "before_period": "", "after_period": "",
            "stage": "rerun-downstream", "status": "failed",
            "error": str(exc), "failed_at": now_text(),
        }
        failures.append(failure)
        emit("failure", **failure)
        if not continue_on_error:
            manifest["rerun_change_failures"] = failures
            manifest["status"] = "failed"
            _persist_existing_pipeline(manifest, manifest_path)
            raise
    manifest["failure_history"] = list(manifest.get("failure_history", [])) + list(manifest.get("failures", []))
    manifest["failures"] = failures
    manifest["rerun_change_failures"] = failures
    available_periods = {
        (str(entry.get("grid")), str(entry.get("period")))
        for entry in (manifest.get("period_results", []) or [])
        if isinstance(entry, dict) and _period_result_ready(entry)
    }
    missing_dependencies = {
        (grid, period)
        for grid, periods in _manifest_period_plan(manifest).items()
        for period in periods
        if (grid, period) not in available_periods
    }
    manifest["rerun_missing_period_dependencies"] = [
        {"grid": grid, "period": period}
        for grid, period in sorted(missing_dependencies)
    ]
    manifest["status"] = (
        "completed_with_errors" if failures or missing_dependencies else "completed"
    )
    manifest["updated_at"] = now_text()
    _persist_existing_pipeline(manifest, manifest_path)
    _evaluate_fast_manifest_pairs(manifest_path, completed_keys)
    result = {
        "change_count": len(completed_keys),
        "failure_count": len(failures),
        "total_change_count": len(change_keys),
    }
    emit("complete", stage="rerun-all-changes", **result)
    return result


def build_temporal_outputs(manifest: dict, job_root: Path | None = None) -> list[dict]:
    """Build the all-Shapefile temporal products without loading GIS at GUI startup."""
    from temporal_road_analysis import build_from_manifest

    target = Path(job_root or manifest.get("job_root") or ".").expanduser().resolve()
    return build_from_manifest(manifest, target)


def _write_task_report(manifest: dict, job_root: Path) -> None:
    rows = []
    for entry in manifest.get("period_results", []):
        published = entry.get("published") if isinstance(entry.get("published"), dict) else {}
        rows.append({
            "type": "period",
            "grid": entry.get("grid", ""),
            "scope": entry.get("period", ""),
            "status": "complete",
            "elapsed_seconds": entry.get("elapsed_seconds", ""),
            "output": published.get("centerlines") or entry.get("centerlines", ""),
            "message": "",
        })
    for entry in manifest.get("change_results", []):
        published = entry.get("published") if isinstance(entry.get("published"), dict) else {}
        rows.append({
            "type": "change",
            "grid": entry.get("grid", ""),
            "scope": f"{entry.get('before_period', '')} -> {entry.get('after_period', '')}",
            "status": "complete",
            "elapsed_seconds": entry.get("elapsed_seconds", ""),
            "output": published.get("changes") or entry.get("output", ""),
            "message": "",
        })
    for entry in manifest.get("temporal_results", []):
        published = entry.get("published") if isinstance(entry.get("published"), dict) else {}
        rows.append({
            "type": "temporal",
            "grid": entry.get("grid", ""),
            "scope": f"{entry.get('period_count', 0)} 期长时序道路",
            "status": "complete",
            "elapsed_seconds": entry.get("elapsed_seconds", ""),
            "output": published.get("life_shp") or entry.get("life_shp", ""),
            "message": "",
        })
    for entry in manifest.get("failures", []):
        rows.append({
            "type": entry.get("type", "failure"),
            "grid": entry.get("grid", ""),
            "scope": entry.get("scope", ""),
            "status": entry.get("status", "failed"),
            "elapsed_seconds": entry.get("elapsed_seconds", ""),
            "output": "",
            "message": entry.get("error", ""),
        })
    report = {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "planned_period_count": manifest.get("planned_period_count"),
        "planned_change_count": manifest.get("planned_change_count"),
        "period_count": manifest.get("period_count"),
        "change_count": manifest.get("change_count"),
        "failure_count": len(manifest.get("failures", [])),
        "rows": rows,
    }
    write_json(job_root / "task_report.json", report)
    csv_path = job_root / "task_report.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("type", "grid", "scope", "status", "elapsed_seconds", "output", "message"),
        )
        writer.writeheader()
        writer.writerows(rows)


def run_all(args: argparse.Namespace) -> dict:
    """Extract every grid/period and detect every adjacent-period change.

    The job manifest is written before expensive work starts and after every
    period/change unit.  A rerun with ``--resume`` validates the frozen input
    specification and skips only outputs that still pass completeness checks.
    """
    current_started = time.monotonic()
    mode = str(getattr(args, "mode", "grid") or "grid").strip().casefold()
    if mode not in {"validation", "grid"}:
        raise ValueError("--mode 必须是 validation 或 grid")
    execution_profile = str(getattr(args, "execution_profile", "full") or "full").casefold()
    if execution_profile not in {"full", "fast"}:
        raise ValueError("--execution-profile 必须是 full 或 fast")
    validation_area: str | dict[str, str] = ""
    truth_by_pair: dict[tuple, Path] = {}
    source_root = None
    if mode == "validation":
        grids, validation_areas, truth_by_task = validation_batch_inputs(args)
        truth_by_pair = truth_by_task
        validation_area = validation_areas
    else:
        source_value = str(getattr(args, "source_root", "") or "").strip()
        if not source_value:
            raise ValueError("格网模式必须提供 --source-root PATH")
        source_root = Path(source_value).expanduser().resolve()
        grids = discover_grid_periods(source_root)
    grids = {
        grid: dict(sorted(periods.items(), key=lambda pair: period_sort_key(pair[0])))
        for grid, periods in grids.items()
    }
    output_root = Path(args.output_root).expanduser().resolve()
    data_check_only = bool(getattr(args, "data_check_only", False))
    if bool(getattr(args, "preflight_only", False)) or data_check_only:
        report = build_preflight_report(
            mode, grids, validation_area, truth_by_pair, output_root,
            include_runtime_checks=not data_check_only,
        )
        report["elapsed_seconds"] = elapsed_seconds(current_started)
        report["completed"] = 1
        report["total"] = 1
        emit("complete", stage="data-check" if data_check_only else "preflight", **report)
        return report
    if bool(getattr(args, "runtime_preflight", False)):
        emit("pipeline", stage="运行前检查", status="running", completed=0, total=1)
        data_report = build_preflight_report(
            mode, grids, validation_area, truth_by_pair, output_root,
            include_runtime_checks=True,
        )
        runtime_report = check_runtime_environment(args, output_root)
        emit(
            "pipeline", stage="运行前检查", status="complete", completed=1, total=1,
            warning_count=len(data_report.get("warnings") or []),
            cuda_available=runtime_report["cuda_available"],
            output_free_bytes=runtime_report["output_free_bytes"],
        )
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = clean_name(args.run_id.strip() or time.strftime("run_%Y%m%d_%H%M%S"))
    layout = project_layout(output_root)
    resume = bool(getattr(args, "resume", False))
    job_root = _resolve_full_run_job_root(layout, run_id, resume=resume)
    continue_on_error = bool(getattr(args, "continue_on_error", False))
    if job_root.exists() and not resume:
        raise FileExistsError(f"任务目录已存在：{job_root}。如需继续，请使用 --resume。")

    input_spec = _task_input_spec(mode, grids, validation_area, truth_by_pair, args)
    prior = {}
    state_path = job_root / "job_state.json"
    invalidation = {"periods": [], "changes": [], "threshold_changed": False, "truth_changed": False, "reuse_all": False}
    if resume:
        prior = _read_full_run_resume_state(
            state_path, run_id, job_root, layout=layout, output_root=output_root,
        )
        _validate_resume_execution_profile(prior, execution_profile)
        batch_repair = repair_task_batch_lists(job_root)
        marker_repair = relocate_task_image_markers(job_root)
        if batch_repair.modified_lists:
            print(
                f"[项目路径检查] 已自动修正 {batch_repair.modified_lists} 个影像清单，"
                f"共 {batch_repair.modified_paths} 条路径。",
                flush=True,
            )
            print("[项目路径检查] 当前任务只使用当前项目中的文件。", flush=True)
        elif marker_repair.updated_markers:
            print(
                f"[项目路径检查] 已自动更新 {marker_repair.updated_markers} 个切片完成标记。",
                flush=True,
            )
            print("[项目路径检查] 当前任务只使用当前项目中的文件。", flush=True)
        else:
            print("[项目路径检查] 当前任务路径完整，无需修正。", flush=True)
        prior_input_spec = _enrich_relocated_input_spec(
            dict(prior.get("input_spec") or {}), job_root,
        )
        comparison_input_spec = dict(prior_input_spec)
        if job_root == layout.legacy_full_run_root(run_id):
            # A storage-layout upgrade alone must not invalidate completed inference.
            comparison_input_spec["pipeline_version"] = input_spec.get("pipeline_version")
        invalidation = dependency_invalidation_plan(comparison_input_spec, input_spec)
        _refresh_resume_external_paths(
            prior, grids=grids, validation_area=validation_area,
            truth_by_pair=truth_by_pair, source_root=source_root,
            input_spec=input_spec, historical_input_spec=prior_input_spec,
        )
        _refresh_relocated_runtime_metadata(job_root, args)
    layout.ensure_project_directories()
    if not resume:
        job_root.mkdir(parents=True, exist_ok=False)

    raw_prior_periods = {
        (str(entry.get("grid")), str(entry.get("period"))): entry
        for entry in prior.get("period_results", []) if isinstance(entry, dict)
    }
    frozen_periods = {
        key for key, entry in raw_prior_periods.items() if _period_result_ready(entry)
    }
    invalid_periods = {
        tuple(value) for value in invalidation["periods"]
        if tuple(value) not in frozen_periods
    }
    invalid_changes = {
        tuple(value) for value in invalidation["changes"]
        if invalidation["threshold_changed"] or invalidation["truth_changed"]
        or (tuple(value)[0], tuple(value)[1]) in invalid_periods
        or (tuple(value)[0], tuple(value)[2]) in invalid_periods
    }
    invalidation["periods"] = sorted(invalid_periods)
    invalidation["changes"] = sorted(invalid_changes)
    invalidation["frozen_periods"] = sorted(frozen_periods)
    invalidation["reuse_all"] = not invalid_periods and not invalid_changes

    planned_period_count = sum(len(periods) for periods in grids.values())
    planned_change_count = sum(max(0, len(periods) - 1) for periods in grids.values())
    total_work = planned_period_count + planned_change_count
    previous_elapsed = float(prior.get("elapsed_seconds", 0) or 0)
    prior_periods = {
        key: entry for key, entry in raw_prior_periods.items()
        if key not in invalid_periods
    }
    prior_changes = {
        (entry.get("grid"), entry.get("before_period"), entry.get("after_period")): entry
        for entry in prior.get("change_results", []) if isinstance(entry, dict)
        and (str(entry.get("grid")), str(entry.get("before_period")), str(entry.get("after_period"))) not in invalid_changes
    }
    expected_periods = {
        (str(grid), str(period))
        for grid, periods in grids.items() for period in periods
    }
    expected_changes = {
        (str(grid), str(before), str(after))
        for grid, periods in grids.items()
        for before, after in zip(list(periods), list(periods)[1:])
    }
    expected_temporal_grids = {str(grid) for grid in grids}
    prior_temporal = {
        str(entry.get("grid")): entry
        for entry in prior.get("temporal_results", []) if isinstance(entry, dict)
    }
    periods_ready = (
        set(prior_periods) == expected_periods
        and all(_period_result_ready(entry) for entry in prior_periods.values())
    )
    changes_ready = (
        set(prior_changes) == expected_changes
        and all(_change_result_ready(entry) for entry in prior_changes.values())
    )
    temporal_ready = (
        set(prior_temporal) == expected_temporal_grids
        and all(_temporal_result_ready(entry) for entry in prior_temporal.values())
    )
    reuse_completed_downstream = bool(
        resume and invalidation["reuse_all"]
        and periods_ready and changes_ready and temporal_ready
    )
    if resume and invalidation["reuse_all"] and prior.get("status") in {"completed", "completed_with_errors"}:
        if periods_ready and changes_ready and temporal_ready:
            prior["attempt"] = int(prior.get("attempt", 0) or 0) + 1
            prior["resumed_at"] = now_text()
            prior["last_reused_at"] = now_text()
            prior["reuse_count"] = int(prior.get("reuse_count", 0) or 0) + 1
            prior["job_root"] = str(job_root)
            prior["project_root"] = str(layout.project_root)
            prior["output_root"] = str(output_root)
            provenance = dict(prior.get("provenance") or {})
            provenance.setdefault("historical_input_spec", prior_input_spec)
            provenance["historical_only"] = True
            prior["provenance"] = provenance
            prior["input_spec"] = input_spec
            prior["invalidation_plan"] = invalidation
            publisher = ResultPublisher(output_root, project_root=layout.project_root)
            publisher.publish_manifest(
                prior, source_manifest=job_root / "pipeline_result.json",
            )
            _write_task_report(prior, job_root)
            publisher.publish_reports(job_root)
            _persist_pipeline(prior, job_root, output_root)
            emit(
                "complete", stage="all", manifest=str(job_root / "pipeline_result.json"),
                job_root=str(job_root), grid_count=len(grids),
                period_count=len(prior_periods), change_count=len(prior_changes),
                failure_count=len(prior.get("failures", [])), status=prior.get("status"),
                reused=True, completed=1, total=1,
            )
            return prior
    fast_change_pair_count = sum(
        max(0, len(periods) - 1) for periods in grids.values()
    )
    fast_change_source = (
        "fast_automatic" if not truth_by_pair else
        "synthetic_from_truth" if len(truth_by_pair) == fast_change_pair_count else
        "fast_truth_or_automatic"
    )
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "project_root": str(layout.project_root),
        "output_root": str(output_root),
        "result_index": str(layout.result_index_path),
        "mode": mode,
        "execution_profile": execution_profile,
        "road_extraction_source": "samroad_fast" if execution_profile == "fast" else "samroad",
        "surface_source": "probability_fast" if execution_profile == "fast" else "sam_molra",
        "width_source": "fast_measured" if execution_profile == "fast" else "full_measured",
        "change_source": fast_change_source if execution_profile == "fast" else "automatic_detection",
        "source_root": str(source_root) if source_root is not None else "",
        "validation_area": (
            next(iter(validation_area.values())) if isinstance(validation_area, dict) and len(validation_area) == 1
            else validation_area
        ),
        "validation_areas": validation_area if isinstance(validation_area, dict) else {},
        "truth_type_field": str(getattr(args, "truth_type_field", "") or ""),
        "evaluation_enabled": (
            not bool(getattr(args, "no_evaluation", False))
            and (execution_profile != "fast" or bool(truth_by_pair))
        ),
        "job_root": str(job_root),
        "input_spec": input_spec,
        "provenance": {
            **dict(prior.get("provenance") or {}),
            "historical_input_spec": prior_input_spec if resume else {},
            "historical_only": bool(resume),
        },
        "relocation_history": list(prior.get("relocation_history") or []),
        "invalidation_plan": invalidation,
        "status": "running",
        "attempt": int(prior.get("attempt", 0) or 0) + 1,
        "started_at": prior.get("started_at") or now_text(),
        "resumed_at": now_text() if resume else None,
        "grid_count": len(grids),
        "period_orders": {
            grid: period_order_manifest(periods.keys()) for grid, periods in grids.items()
        },
        "planned_period_count": planned_period_count,
        "planned_change_count": planned_change_count,
        "total_work": total_work,
        "processed_work": 0,
        "period_count": 0,
        "change_count": 0,
        "period_results": [],
        "change_results": [],
        "failures": [],
        "failure_history": list(prior.get("failure_history", [])) + list(prior.get("failures", [])),
    }
    _persist_pipeline(manifest, job_root, output_root)

    def update_elapsed() -> float:
        manifest["elapsed_seconds"] = round(previous_elapsed + elapsed_seconds(current_started), 3)
        return float(manifest["elapsed_seconds"])

    def progress(stage: str, **payload) -> None:
        processed = int(manifest["processed_work"])
        elapsed = update_elapsed()
        eta = None
        if processed > 0 and total_work > processed:
            attempt_elapsed = elapsed_seconds(current_started)
            eta = round((attempt_elapsed / processed) * (total_work - processed), 1)
        emit(
            "pipeline", stage=stage, status=payload.pop("status", "running"),
            completed=processed, total=total_work, progress=(processed / total_work if total_work else 1.0),
            elapsed_seconds=elapsed, eta_seconds=eta, **payload,
        )

    analysis_sources: dict[str, Path] = {}
    emit(
        "pipeline",
        stage="数据扫描",
        status="complete",
        grid_count=len(grids),
        period_count=sum(len(periods) for periods in grids.values()),
        run_id=run_id,
        completed=0,
        total=total_work,
    )
    try:
        if mode == "validation":
            for area_id, area_periods in grids.items():
                normalized_root = job_root / "validation_inputs" / clean_name(area_id)
                area_invalidated = any((str(area_id), str(period)) in invalid_periods for period in area_periods)
                ready = _normalized_sources_ready(area_periods, normalized_root) if resume and not area_invalidated else None
                if ready is None:
                    ready = normalize_validation_sources(
                        area_periods, validation_area[area_id], normalized_root,
                    )
                else:
                    emit(
                        "pipeline", stage="验证区影像规范化", status="skipped",
                        grid=area_id, reason="续跑复用已完成的规范化影像",
                        completed=0, total=total_work,
                    )
                for period, source in ready.items():
                    analysis_sources[f"{area_id}\0{period}"] = source

        result_by_period: dict[tuple[str, str], dict] = {}
        for grid_index, (grid_name, periods) in enumerate(grids.items(), start=1):
            safe_grid = clean_name(grid_name)
            for period_index, (period, source) in enumerate(periods.items(), start=1):
                unit_started = time.monotonic()
                safe_period = clean_name(period)
                workspace = job_root / "grids" / safe_grid / "periods" / safe_period
                prior_entry = prior_periods.get((grid_name, period))
                if resume and prior_entry and _period_result_ready(prior_entry):
                    entry = dict(prior_entry)
                    entry_provenance = dict(entry.get("provenance") or {})
                    entry_provenance.update({
                        "frozen_completed_result": True,
                        "historical_input_spec": prior_input_spec,
                        "historical_only": True,
                    })
                    entry["provenance"] = entry_provenance
                    manifest["period_results"].append(entry)
                    result_by_period[(grid_name, period)] = entry
                    published = ResultPublisher(output_root).publish_period(
                        grid_name, period, entry, run_id=run_id,
                    )
                    if published:
                        entry["published"] = published
                    manifest["processed_work"] += 1
                    message = f"[续跑] {period} 期已有完整成果，已验证并复用。"
                    print(message, flush=True)
                    progress(
                        "道路提取", status="skipped", grid=grid_name, period=period,
                        grid_index=grid_index, grid_total=len(grids),
                        period_index=period_index, period_total=len(periods),
                        reason=message,
                    )
                    _persist_pipeline(manifest, job_root, output_root)
                    continue

                progress(
                    "道路提取", grid=grid_name, period=period,
                    grid_index=grid_index, grid_total=len(grids),
                    period_index=period_index, period_total=len(periods),
                )
                analysis_source = analysis_sources.get(f"{grid_name}\0{period}", source)
                try:
                    internal_resume = resume and (grid_name, period) not in invalid_periods
                    if internal_resume and (workspace / "period_state.json").is_file() and _prepared_workspace_complete(workspace):
                        emit(
                            "pipeline", stage="输入准备", status="skipped", grid=grid_name,
                            period=period, stage_key="prepare", stage_index=0,
                            stage_total=len(PERIOD_STAGE_DEFINITIONS),
                            reason="续跑复用已完成的输入准备",
                        )
                    else:
                        prepare(argparse.Namespace(source=str(analysis_source), workspace=str(workspace)))
                    base_run_id = "roads"
                    result = _ensure_extract_manifest_fields(extract(
                        argparse.Namespace(
                            workspace=str(workspace), source="", checkpoint=args.checkpoint,
                            config=args.config, device=args.device, pixel_size=args.pixel_size,
                            rescale=args.rescale, run_id=base_run_id,
                            junction_node_mode=str(getattr(args, "junction_node_mode", "sparse") or "sparse"),
                            execution_profile=execution_profile,
                            validation_area=(validation_area.get(grid_name, "") if isinstance(validation_area, dict) else validation_area),
                            grid=grid_name, period=period,
                            resume=internal_resume,
                            pipeline_state=str(state_path),
                        )
                    ))
                    result_path = workspace / "latest_result.json"
                    entry = {
                        "grid": grid_name, "period": period, "source": str(source),
                        "analysis_source": str(analysis_source), "result": str(result_path), **result,
                    }
                    entry["extract_elapsed_seconds"] = result.get("elapsed_seconds")
                    entry["elapsed_seconds"] = elapsed_seconds(unit_started)
                    manifest["period_results"].append(entry)
                    result_by_period[(grid_name, period)] = entry
                except Exception as exc:
                    failure = {
                        "type": "period", "grid": grid_name, "scope": period,
                        "status": "failed", "error": str(exc),
                        "elapsed_seconds": elapsed_seconds(unit_started), "failed_at": now_text(),
                    }
                    manifest["failures"].append(failure)
                    emit("pipeline", stage="道路提取", **failure)
                    if not continue_on_error:
                        raise
                finally:
                    _merge_pipeline_position(manifest, state_path)
                    manifest["processed_work"] += 1
                    manifest["period_count"] = len(manifest["period_results"])
                    update_elapsed()
                    _persist_pipeline(manifest, job_root, output_root)

            period_names = list(periods)
            for before_period, after_period in zip(period_names, period_names[1:]):
                unit_started = time.monotonic()
                prior_entry = prior_changes.get((grid_name, before_period, after_period))
                if resume and prior_entry and _change_result_ready(prior_entry):
                    manifest["change_results"].append(prior_entry)
                    manifest["processed_work"] += 1
                    progress(
                        "变化成果生成" if execution_profile == "fast" else "变化检测", status="skipped", grid=grid_name,
                        before_period=before_period, after_period=after_period,
                        reason="已完成且成果完整",
                    )
                    _persist_pipeline(manifest, job_root, output_root)
                    continue
                before_entry = result_by_period.get((grid_name, before_period))
                after_entry = result_by_period.get((grid_name, after_period))
                if before_entry is None or after_entry is None:
                    failure = {
                        "type": "change", "grid": grid_name,
                        "scope": f"{before_period} -> {after_period}",
                        "status": "skipped_dependency",
                        "error": "前期或后期道路提取失败，已跳过该变化对。",
                        "elapsed_seconds": 0.0, "failed_at": now_text(),
                    }
                    manifest["failures"].append(failure)
                    manifest["processed_work"] += 1
                    progress(
                        "变化成果生成" if execution_profile == "fast" else "变化检测", status="skipped", grid=grid_name,
                        before_period=before_period, after_period=after_period,
                        reason=failure["error"],
                    )
                    _persist_pipeline(manifest, job_root, output_root)
                    continue

                change_output = job_root / "grids" / safe_grid / "changes" / (
                    f"{clean_name(before_period)}_to_{clean_name(after_period)}"
                )
                progress(
                    "变化成果生成" if execution_profile == "fast" else "变化检测", grid=grid_name,
                    before_period=before_period, after_period=after_period,
                )
                manifest.update({
                    "current_grid": grid_name,
                    "current_period": f"{before_period} → {after_period}",
                    "current_stage": "change",
                    "current_stage_label": "变化成果生成" if execution_profile == "fast" else "变化检测",
                    "period_state": None,
                })
                _persist_pipeline(manifest, job_root, output_root)
                try:
                    truth_value = str(truth_by_pair.get((grid_name, before_period, after_period), ""))
                    validation_value = (
                        validation_area.get(grid_name, "") if isinstance(validation_area, dict) else validation_area
                    )
                    if execution_profile == "fast":
                        result = _run_fast_change_result(
                            Path(str(before_entry["result"])),
                            Path(str(after_entry["result"])),
                            change_output,
                            before_period=before_period,
                            after_period=after_period,
                            position_tolerance=float(args.tolerance),
                            width_change_absolute=float(args.absolute),
                            width_change_ratio=float(args.ratio),
                            truth_path=(
                                Path(truth_value) if truth_value else None
                            ),
                            validation_area=(
                                Path(validation_value) if validation_value else None
                            ),
                            truth_type_field=str(
                                getattr(args, "truth_type_field", "") or "BHBM"
                            ),
                        )
                        result = _ensure_change_manifest_fields(result, change_output)
                    else:
                        result = _ensure_change_manifest_fields(change(
                            argparse.Namespace(
                                before_result=before_entry["result"], after_result=after_entry["result"],
                                output=str(change_output), before_period=before_period,
                                after_period=after_period, absolute=args.absolute, ratio=args.ratio,
                                tolerance=args.tolerance,
                                truth=("" if bool(getattr(args, "no_evaluation", False)) else truth_value),
                                validation_area=validation_value,
                                truth_type_field=str(getattr(args, "truth_type_field", "") or ""),
                            )
                        ), change_output)
                    entry = {
                        "grid": grid_name, "before_period": before_period,
                        "after_period": after_period,
                        "truth": str(truth_by_pair.get((grid_name, before_period, after_period), "")),
                        "validation_area": (validation_area.get(grid_name, "") if isinstance(validation_area, dict) else validation_area),
                        "truth_type_field": str(getattr(args, "truth_type_field", "") or ""),
                        "absolute": str(args.absolute), "ratio": str(args.ratio),
                        "tolerance": str(args.tolerance), **result,
                    }
                    entry["elapsed_seconds"] = elapsed_seconds(unit_started)
                    manifest["change_results"].append(entry)
                    published = ResultPublisher(output_root).publish_change(
                        grid_name, before_period, after_period, entry, run_id=run_id,
                    )
                    if published:
                        entry["published"] = published
                    if execution_profile == "fast" and truth_value:
                        _persist_pipeline(manifest, job_root, output_root)
                        evaluate_existing_changes(argparse.Namespace(
                            pipeline_manifest=str(job_root / "pipeline_result.json"), grid=grid_name,
                            before_period=before_period, after_period=after_period, truth=truth_value,
                            validation_area=validation_value,
                            truth_type_field=str(getattr(args, "truth_type_field", "") or "BHBM"),
                            truth_added_value="", truth_width_changed_value="", truth_removed_value="",
                            evaluation_tolerance=5.0,
                        ))
                        evaluated_manifest = read_json(job_root / "pipeline_result.json")
                        evaluated_entry = next(
                            candidate for candidate in evaluated_manifest.get("change_results", [])
                            if str(candidate.get("grid")) == str(grid_name)
                            and str(candidate.get("before_period")) == str(before_period)
                            and str(candidate.get("after_period")) == str(after_period)
                        )
                        entry.update(evaluated_entry)
                        manifest["evaluation_summary"] = evaluated_manifest.get("evaluation_summary", {})
                except Exception as exc:
                    failure = {
                        "type": "change", "grid": grid_name,
                        "scope": f"{before_period} -> {after_period}",
                        "status": "failed", "error": str(exc),
                        "elapsed_seconds": elapsed_seconds(unit_started), "failed_at": now_text(),
                    }
                    manifest["failures"].append(failure)
                    emit("pipeline", stage="变化成果生成" if execution_profile == "fast" else "变化检测", **failure)
                    if not continue_on_error:
                        raise
                finally:
                    manifest["processed_work"] += 1
                    manifest["change_count"] = len(manifest["change_results"])
                    update_elapsed()
                    _persist_pipeline(manifest, job_root, output_root)

        manifest["period_count"] = len(manifest["period_results"])
        manifest["change_count"] = len(manifest["change_results"])
        if execution_profile == "fast":
            manifest["temporal_results"] = []
            manifest["temporal_status"] = "skipped_fast_profile"
        else:
            temporal_started = time.monotonic()
            emit(
                "pipeline", stage="长时序道路汇总", status="running",
                completed=total_work, total=total_work,
            )
            if reuse_completed_downstream:
                manifest["temporal_results"] = [prior_temporal[str(grid)] for grid in grids]
                emit(
                    "pipeline", stage="长时序道路汇总", status="skipped",
                    reason="续跑复用已完成且完整的长时序成果",
                    temporal_grid_count=len(manifest["temporal_results"]),
                    completed=total_work, total=total_work,
                )
            else:
                manifest["temporal_results"] = build_temporal_outputs(manifest, job_root)
                for entry in manifest["temporal_results"]:
                    entry["elapsed_seconds"] = elapsed_seconds(temporal_started)
            emit(
                "pipeline", stage="长时序道路汇总", status="complete",
                temporal_grid_count=len(manifest["temporal_results"]),
                completed=total_work, total=total_work,
            )
        manifest["completed_at"] = now_text()
        manifest["status"] = "completed_with_errors" if manifest["failures"] else "completed"
        update_elapsed()
        if reuse_completed_downstream and _evaluation_summary_ready(prior.get("evaluation_summary")):
            manifest["evaluation_summary"] = prior["evaluation_summary"]
        else:
            aggregate_change_evaluations(manifest, job_root)
        publisher = ResultPublisher(output_root)
        publisher.publish_manifest(
            manifest, source_manifest=job_root / "pipeline_result.json",
        )
        _write_task_report(manifest, job_root)
        publisher.publish_reports(job_root)
        _persist_pipeline(manifest, job_root, output_root)
        emit(
            "complete", stage="all", manifest=str(job_root / "pipeline_result.json"),
            job_root=str(job_root), grid_count=len(grids),
            period_count=manifest["period_count"], change_count=manifest["change_count"],
            failure_count=len(manifest["failures"]), status=manifest["status"],
            execution_profile=execution_profile,
            elapsed_seconds=manifest["elapsed_seconds"], completed=total_work, total=total_work,
        )
        return manifest
    except Exception as exc:
        _merge_pipeline_position(manifest, state_path)
        manifest["status"] = "failed"
        manifest["last_error"] = str(exc)
        manifest["failed_at"] = now_text()
        update_elapsed()
        _write_task_report(manifest, job_root)
        ResultPublisher(output_root).publish_reports(job_root)
        _persist_pipeline(manifest, job_root, output_root)
        raise


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAMRoad 面向用户批处理后端")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查独立运行环境、模型和算法核心")
    a = sub.add_parser("preflight-project", help="发现规范项目文件夹并仅执行输入预检")
    a.add_argument("--project-root", required=True)
    a = sub.add_parser("extract-project-period", help="提取规范项目中的一个区域期次，支持完整成果续跑")
    a.add_argument("--project-root", required=True); a.add_argument("--area-id", required=True); a.add_argument("--period", required=True)
    a.add_argument("--run-id", required=True); a.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a.add_argument("--pixel-size", default="0.0"); a.add_argument("--rescale", choices=["on", "off"], default="off")
    a.add_argument("--junction-node-mode", choices=["sparse", "dense_legacy"], default="sparse")
    a.add_argument("--resume", action="store_true")
    a = sub.add_parser("extract-project-all", help="仅提取规范项目中的全部或指定区域期次，不运行变化检测")
    a.add_argument("--project-root", required=True); a.add_argument("--area-id", action="append", default=[])
    a.add_argument("--run-id", required=True); a.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a.add_argument("--pixel-size", default="0.0"); a.add_argument("--rescale", choices=["on", "off"], default="off")
    a.add_argument("--junction-node-mode", choices=["sparse", "dense_legacy"], default="sparse")
    a.add_argument("--resume", action="store_true"); a.add_argument("--continue-on-error", action="store_true")
    a = sub.add_parser("change-project-periods", help="对规范项目中两个相邻且已完成提取的期次执行变化检测")
    a.add_argument("--project-root", required=True); a.add_argument("--area-id", required=True)
    a.add_argument("--before-period", required=True); a.add_argument("--after-period", required=True)
    a.add_argument("--before-state", required=True); a.add_argument("--after-state", required=True); a.add_argument("--run-id", required=True)
    a.add_argument("--absolute", default="2.0"); a.add_argument("--ratio", default="0.2"); a.add_argument("--tolerance", default="3.0")
    a.add_argument("--resume", action="store_true")
    a = sub.add_parser("map-scene", help="只读生成真实栅格与矢量地图场景，不运行模型")
    a.add_argument("--project-root", required=True); a.add_argument("--area-id", required=True); a.add_argument("--period", required=True)
    a.add_argument("--extraction-state", default=""); a.add_argument("--change-state", default="")
    a.add_argument("--width", type=int, default=1000); a.add_argument("--height", type=int, default=700)
    a = sub.add_parser("prepare", help="导入格网目录或 TXT 清单")
    a.add_argument("--source", required=True); a.add_argument("--workspace", required=True)
    a = sub.add_parser("extract", help="批量道路提取和宽度计算")
    a.add_argument("--workspace", required=True); a.add_argument("--source", default="")
    a.add_argument("--checkpoint", required=True); a.add_argument("--config", required=True)
    a.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a.add_argument("--pixel-size", default="0.0"); a.add_argument("--rescale", choices=["on", "off"], default="off")
    a.add_argument("--junction-node-mode", choices=["sparse", "dense_legacy"], default="sparse")
    a.add_argument("--run-id", default="")
    a.add_argument("--resume", action="store_true")
    a = sub.add_parser("change", help="两期道路宽度变化检测")
    a.add_argument("--before-result", required=True); a.add_argument("--after-result", required=True); a.add_argument("--output", required=True)
    a.add_argument("--before-period", default="before"); a.add_argument("--after-period", default="after")
    a.add_argument("--absolute", default="2.0"); a.add_argument("--ratio", default="0.2"); a.add_argument("--tolerance", default="3.0")
    a.add_argument("--truth", default="", help="可选变化真值矢量")
    a.add_argument("--validation-area", default="", help="可选评价验证区面矢量")
    a.add_argument("--truth-type-field", default="", help="可选真值变化类型字段")
    a = sub.add_parser("apply-edits", help="应用人工编辑中心线并重新生成路面、宽度和正式成果")
    a.add_argument("--result", required=True, help="期次 latest_result.json")
    a.add_argument("--edited-dir", default="")
    a.add_argument("--pipeline-manifest", default="")
    a = sub.add_parser("rerun-period", help="主动重跑任务索引中的一个道路提取期次")
    a.add_argument("--pipeline-manifest", required=True); a.add_argument("--grid", required=True); a.add_argument("--period", required=True)
    a.add_argument("--update-related", action="store_true")
    a = sub.add_parser("rerun-change", help="主动重跑任务索引中的一个相邻变化对")
    a.add_argument("--pipeline-manifest", required=True); a.add_argument("--grid", required=True)
    a.add_argument("--before-period", required=True); a.add_argument("--after-period", required=True)
    a.add_argument("--update-temporal", action="store_true")
    a = sub.add_parser("rerun-all-periods", help="批量重跑全部道路提取并按依赖更新下游成果")
    a.add_argument("--pipeline-manifest", required=True)
    a.add_argument("--continue-on-error", action="store_true")
    a = sub.add_parser("rerun-all-changes", help="复用道路提取成果并批量重跑全部相邻变化对")
    a.add_argument("--pipeline-manifest", required=True)
    a.add_argument("--continue-on-error", action="store_true")
    a = sub.add_parser("evaluate-existing", help="使用真值评价已有变化检测结果，不重跑提取和检测")
    a.add_argument("--pipeline-manifest", required=True)
    a.add_argument("--grid", required=True)
    a.add_argument("--before-period", required=True)
    a.add_argument("--after-period", required=True)
    a.add_argument("--truth", required=True)
    a.add_argument("--validation-area", default="")
    a.add_argument("--truth-type-field", default="")
    a.add_argument("--truth-added-value", default="")
    a.add_argument("--truth-width-changed-value", default="")
    a.add_argument("--truth-removed-value", default="")
    a.add_argument("--evaluation-tolerance", type=float, default=5.0)
    a = sub.add_parser("evaluate-all-existing", help="评价已有任务的全部验证区和相邻期变化，并汇总总精度")
    a.add_argument("--pipeline-manifest", required=True)
    a.add_argument("--truth", action="append", nargs=4, metavar=("AREA", "BEFORE", "AFTER", "SHP"), default=[])
    a.add_argument(
        "--truth-field-config", action="append", nargs=5,
        metavar=("AREA", "FIELD", "ADDED", "WIDTH_CHANGED", "REMOVED"), default=[],
    )
    a.add_argument("--truth-type-field", default="BHBM")
    a.add_argument("--truth-added-value", default="")
    a.add_argument("--truth-width-changed-value", default="")
    a.add_argument("--truth-removed-value", default="")
    a.add_argument("--evaluation-tolerance", type=float, default=5.0)
    a = sub.add_parser("all", help="验证模式默认按显式期次运行；格网模式保留旧布局扫描")
    a.add_argument("--mode", choices=["validation", "grid"], default="validation")
    a.add_argument("--execution-profile", choices=["full", "fast"], default="full")
    a.add_argument(
        "--validation-area", action="append", nargs="+", default=[], metavar="AREA_OR_SHP",
        help="可重复：区域名 验证区SHP；单路径形式兼容旧任务",
    )
    a.add_argument(
        "--period", action="append", nargs="+", default=[], metavar="AREA_PERIOD_TXT",
        help="可重复：区域名 期次 影像TXT；两参数形式应用于全部验证区",
    )
    a.add_argument(
        "--truth", action="append", nargs="+", default=[], metavar="AREA_PAIR_SHP",
        help="可重复：区域名 前期 后期 真值SHP；单验证区兼容三参数形式",
    )
    a.add_argument("--truth-type-field", default="", help="可选真值变化类型字段")
    a.add_argument("--no-evaluation", action="store_true", help="生产模式：不要求变化真值，也不生成真值评价")
    a.add_argument("--source-root", default="", metavar="PATH", help="仅 --mode grid：旧格网根目录")
    a.add_argument("--output-root", required=True)
    a.add_argument("--checkpoint", required=True); a.add_argument("--config", required=True)
    a.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a.add_argument("--pixel-size", default="0.0"); a.add_argument("--rescale", choices=["on", "off"], default="off")
    a.add_argument("--junction-node-mode", choices=["sparse", "dense_legacy"], default="sparse")
    a.add_argument("--run-id", default="")
    a.add_argument("--resume", action="store_true", help="继续同名未完成任务并跳过完整成果")
    a.add_argument("--continue-on-error", action="store_true", help="单个格网/期次失败后继续处理其他任务")
    a.add_argument("--preflight-only", action="store_true", help="只检查输入规模、空间参考和磁盘风险，不创建任务")
    a.add_argument("--data-check-only", action="store_true", help="只检查项目数据，不检查模型、设备、输出空间")
    a.add_argument("--runtime-preflight", action="store_true", help="完整流程开始前同时检查运行环境和输出空间")
    a.add_argument("--absolute", default="2.0"); a.add_argument("--ratio", default="0.2"); a.add_argument("--tolerance", default="3.0")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.command == "doctor": doctor(args)
    elif args.command == "preflight-project": preflight_project(args)
    elif args.command == "extract-project-period": extract_project_period(args)
    elif args.command == "extract-project-all": extract_project_all(args)
    elif args.command == "change-project-periods": change_project_periods(args)
    elif args.command == "map-scene": map_scene(args)
    elif args.command == "prepare": prepare(args)
    elif args.command == "extract": extract(args)
    elif args.command == "change": change(args)
    elif args.command == "apply-edits": apply_centerline_edits(args)
    elif args.command == "rerun-period": rerun_pipeline_period(args)
    elif args.command == "rerun-change": rerun_pipeline_change(args)
    elif args.command == "rerun-all-periods": rerun_all_pipeline_periods(args)
    elif args.command == "rerun-all-changes": rerun_all_pipeline_changes(args)
    elif args.command == "evaluate-existing": evaluate_existing_changes(args)
    elif args.command == "evaluate-all-existing": evaluate_all_existing_changes(args)
    else: run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
