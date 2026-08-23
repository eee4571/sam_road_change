from __future__ import annotations

"""Project configuration, source discovery and result indexing."""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from input_catalog import period_sort_key
USER_VECTOR_SUFFIX = ".shp"

USER_IMAGE_LIST_SUFFIX = ".txt"

PROJECT_CONFIG_NAME = "project_config.json"

TEMPORAL_ATTRIBUTE_PAGE_SIZE = 500

SCAN_PROGRESS_FILE_INTERVAL = 250

SCAN_PROGRESS_TIME_INTERVAL = 0.3

SCAN_EXCLUDED_DIRECTORY_NAMES = frozenset({
    ".git", ".github", "env", ".venv", "venv", "__pycache__",
    "04_成果输出", "_logs", ".editor_cache", "models", ".runtime",
    "runtime", "node_modules", "tmp", "temp", ".cache",
    "cache", "caches", "_cache", "_tmp", "temporary",
})

_SCAN_EXCLUDED_DIRECTORY_NAMES_CASEFOLD = frozenset(
    value.casefold() for value in SCAN_EXCLUDED_DIRECTORY_NAMES
)

PREVIEW_LABELS = {
    "centerline": "中心线提取",
    "surface": "路面提取",
    "fusion": "融合",
    "width": "重新测宽",
    "change": "最终变化结果",
    "review_change": "待复核变化",
}

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

def natural_key(value: str) -> tuple:
    return period_sort_key(value)

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

def external_source_signature(source_dir: Path | str) -> dict[str, int]:
    """Return a cheap source fingerprint; explicit rescans remain authoritative."""
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"外部数据源不存在：{root}")
    stat = root.stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}

def _scan_directory_is_excluded(name: str) -> bool:
    folded = name.casefold()
    return (
        name.startswith(".")
        or folded in _SCAN_EXCLUDED_DIRECTORY_NAMES_CASEFOLD
    )

def scan_external_data_source(
    source_dir: Path | str, *, cancel_event: threading.Event | None = None,
    progress=None,
) -> dict:
    """Discover only SHP/TXT candidates, pruning irrelevant trees during traversal."""
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"外部数据源不存在：{root}")
    try:
        discovered = discover_validation_project(root)
    except ValueError:
        discovered = None
    candidates: dict[str, list[str]] = {"shp": [], "txt": []}
    visited_files = 0
    visited_directories = 0
    last_progress = time.monotonic()
    last_progress_files = 0
    for current, directories, filenames in os.walk(root, topdown=True):
        if cancel_event is not None and cancel_event.is_set():
            return {"root": str(root), "cancelled": True}
        directories[:] = [name for name in directories if not _scan_directory_is_excluded(name)]
        visited_directories += 1
        current_path = Path(current)
        for filename in filenames:
            visited_files += 1
            suffix = Path(filename).suffix.casefold()
            if suffix == ".shp":
                candidates["shp"].append(str((current_path / filename).resolve()))
            elif suffix == ".txt":
                candidates["txt"].append(str((current_path / filename).resolve()))
            if cancel_event is not None and cancel_event.is_set():
                return {"root": str(root), "cancelled": True}
        now = time.monotonic()
        if progress is not None and (
            visited_files - last_progress_files >= SCAN_PROGRESS_FILE_INTERVAL
            or now - last_progress >= SCAN_PROGRESS_TIME_INTERVAL
        ):
            progress({
                "root": str(root), "directory": str(current_path),
                "visited_files": visited_files,
                "visited_directories": visited_directories,
                "shp_count": len(candidates["shp"]), "txt_count": len(candidates["txt"]),
            })
            last_progress = now
            last_progress_files = visited_files
    return {
        "root": str(root), "discovered": discovered, "candidates": candidates,
        "signature": external_source_signature(root),
        "visited_files": visited_files, "visited_directories": visited_directories,
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cancelled": False,
    }

def scan_result_for_cache(scan: dict) -> dict:
    """Convert a completed scan to a project-config-compatible lightweight record."""
    discovered = scan.get("discovered")
    cached_discovered = None
    if isinstance(discovered, dict):
        area_truths = discovered.get("area_truths") or {}
        if isinstance(area_truths, dict):
            truth_rows = [[*key, value] for key, value in area_truths.items()]
        else:
            truth_rows = [list(row) for row in area_truths]
        cached_discovered = {
            "validation_areas": [list(row) for row in discovered.get("validation_areas", [])],
            "area_periods": {
                str(area): [list(row) for row in rows]
                for area, rows in (discovered.get("area_periods") or {}).items()
            },
            "area_truths": truth_rows,
        }
    return {
        "root": str(scan.get("root") or ""),
        "signature": dict(scan.get("signature") or {}),
        "scanned_at": str(scan.get("scanned_at") or ""),
        "visited_files": int(scan.get("visited_files", 0) or 0),
        "visited_directories": int(scan.get("visited_directories", 0) or 0),
        "candidates": {
            kind: [str(value) for value in (scan.get("candidates") or {}).get(kind, [])]
            for kind in ("shp", "txt")
        },
        "discovered": cached_discovered,
    }

def read_temporal_attributes(path: Path | str):
    """Read only DBF attributes; road geometry is intentionally excluded."""
    source = Path(path).expanduser().resolve()
    try:
        import pyogrio
        return pyogrio.read_dataframe(source, read_geometry=False)
    except (ImportError, TypeError):
        import geopandas as gpd
        return gpd.read_file(source, ignore_geometry=True)

class TemporalAttributePager:
    """In-memory attribute-only filtering and fixed-size Treeview pages."""
    def __init__(self, frame, page_size: int = TEMPORAL_ATTRIBUTE_PAGE_SIZE) -> None:
        self.frame = frame
        self.columns = [str(name) for name in frame.columns]
        self.page_size = max(1, int(page_size))
        self.page_index = 0
        self._filtered_positions: list[int] | None = None
        self._search_cache = None

    @property
    def match_count(self) -> int:
        return len(self.frame) if self._filtered_positions is None else len(self._filtered_positions)

    @property
    def page_count(self) -> int:
        return max(1, (self.match_count + self.page_size - 1) // self.page_size)

    def set_query(self, query: str) -> None:
        needle = str(query).strip().casefold()
        if not needle:
            self._filtered_positions = None
            self.page_index = 0
            return
        if self._search_cache is None:
            values = self.frame.fillna("").astype(str)
            self._search_cache = values.agg(" ".join, axis=1).str.casefold()
        mask = self._search_cache.str.contains(needle, regex=False, na=False)
        self._filtered_positions = [index for index, matched in enumerate(mask.tolist()) if matched]
        self.page_index = 0

    def set_page(self, page_index: int) -> None:
        self.page_index = max(0, min(int(page_index), self.page_count - 1))

    def page_frame(self):
        start = self.page_index * self.page_size
        stop = min(self.match_count, start + self.page_size)
        if self._filtered_positions is None:
            return self.frame.iloc[start:stop]
        return self.frame.iloc[self._filtered_positions[start:stop]]

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

class ProjectManager:
    """GUI-independent facade for project configuration and result discovery."""

    def __init__(self) -> None:
        self.project_root: Path | None = None
        self.config: dict = {}

    def open_project(self, project_root: Path | str) -> dict:
        root = Path(project_root).expanduser().resolve()
        discovered = discover_validation_project(root)
        config = read_project_config(root)
        self.project_root = root
        self.config = config
        return {"root": root, "discovered": discovered, "config": config}

    def read_config(self, project_root: Path | str) -> dict:
        return read_project_config(project_root)

    @staticmethod
    def discover_project(project_root: Path | str) -> dict:
        return discover_validation_project(project_root)

    def save_config(self, project_root: Path | str, payload: dict) -> Path:
        path = atomic_write_json(project_config_path(project_root), payload)
        self.project_root = Path(project_root).expanduser().resolve()
        self.config = dict(payload)
        return path

    def scan_source(self, source: Path | str, **kwargs) -> dict:
        return scan_external_data_source(source, **kwargs)

    @staticmethod
    def source_signature(source: Path | str) -> dict[str, int]:
        return external_source_signature(source)

    @staticmethod
    def cache_scan(scan: dict) -> dict:
        return scan_result_for_cache(scan)

    @staticmethod
    def write_json(path: Path | str, payload: dict) -> Path:
        return atomic_write_json(path, payload)

    @staticmethod
    def read_temporal_attributes(path: Path | str):
        return read_temporal_attributes(path)

    @staticmethod
    def result_items(manifest: dict, base_dir: Path | None = None) -> list[dict[str, str]]:
        return collect_result_tree_items(manifest, base_dir)

    @staticmethod
    def review_items(manifest: dict, base_dir: Path | None = None) -> list[dict[str, str]]:
        return collect_review_items(manifest, base_dir)

    @staticmethod
    def temporal_items(manifest: dict, base_dir: Path | None = None) -> list[dict[str, str]]:
        return collect_temporal_items(manifest, base_dir)

    @staticmethod
    def open_path(path: Path | str) -> None:
        target = Path(path).expanduser().resolve()
        if sys.platform == "win32":
            os.startfile(str(target))
        else:
            subprocess.Popen(["xdg-open", str(target)])
