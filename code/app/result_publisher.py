from __future__ import annotations

"""Project workspace paths and the single user-facing result publication layer."""

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RESULT_DIRECTORY_NAME = "成果输出"
LEGACY_RESULT_DIRECTORY_NAME = "04_成果输出"
WORK_DIRECTORY_NAME = "_work"
LOG_DIRECTORY_NAME = "_logs"
RESULT_INDEX_NAME = "result_index.json"
RESULT_INDEX_VERSION = 1
SHAPEFILE_SIDECAR_SUFFIXES = (
    ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".shp.xml",
)


def safe_name(value: object) -> str:
    text = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value or "").strip()
    )
    return text.strip("._-") or "未命名"


def atomic_write_json(path: Path | str, value: dict) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class ProjectLayout:
    """All project-side paths, kept out of algorithm implementations."""

    project_root: Path
    results_root: Path

    @classmethod
    def from_project(
        cls, project_root: Path | str, results_root: Path | str | None = None,
    ) -> "ProjectLayout":
        project = _resolved(project_root)
        results = _resolved(results_root) if results_root is not None else project / RESULT_DIRECTORY_NAME
        return cls(project, results)

    @classmethod
    def from_output(cls, output_root: Path | str) -> "ProjectLayout":
        results = _resolved(output_root)
        # The GUI normally supplies 成果输出.  For custom CLI output names the
        # containing directory is still the project root and receives _work/_logs.
        return cls(results.parent, results)

    @property
    def work_root(self) -> Path:
        return self.project_root / WORK_DIRECTORY_NAME

    @property
    def tasks_root(self) -> Path:
        return self.work_root / "tasks"

    @property
    def cache_root(self) -> Path:
        return self.work_root / "cache"

    @property
    def editor_cache_root(self) -> Path:
        return self.work_root / "editor_cache"

    @property
    def logs_root(self) -> Path:
        return self.project_root / LOG_DIRECTORY_NAME

    @property
    def result_index_path(self) -> Path:
        return self.results_root / RESULT_INDEX_NAME

    @property
    def legacy_results_root(self) -> Path:
        if self.results_root.name == RESULT_DIRECTORY_NAME:
            return self.project_root / LEGACY_RESULT_DIRECTORY_NAME
        return self.results_root

    @property
    def latest_pipeline_path(self) -> Path:
        return self.tasks_root / "latest_pipeline.json"

    def full_run_root(self, run_id: object) -> Path:
        return self.tasks_root / "runs" / safe_name(run_id)

    def legacy_full_run_root(self, run_id: object) -> Path:
        return self.legacy_results_root / safe_name(run_id)

    @property
    def legacy_latest_pipeline_path(self) -> Path:
        return self.legacy_results_root / "latest_pipeline.json"

    def period_task_root(self, area: object, period: object, run_id: object) -> Path:
        return (
            self.tasks_root / "period_extractions" / safe_name(area) /
            safe_name(period) / safe_name(run_id)
        )

    def legacy_period_task_root(self, area: object, period: object, run_id: object) -> Path:
        return (
            self.legacy_results_root / "period_extractions" / safe_name(area) /
            safe_name(period) / safe_name(run_id)
        )

    def batch_task_root(self, run_id: object) -> Path:
        return self.tasks_root / "batch_extractions" / safe_name(run_id)

    def change_task_root(
        self, area: object, before: object, after: object, run_id: object,
    ) -> Path:
        return (
            self.tasks_root / "period_changes" / safe_name(area) /
            f"{safe_name(before)}_to_{safe_name(after)}" / safe_name(run_id)
        )

    def ensure_project_directories(self) -> None:
        for directory in (
            self.results_root, self.tasks_root, self.cache_root,
            self.editor_cache_root, self.logs_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _manifest_path(value: object, base_dir: Path | None = None) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path") or value.get("file")
    if isinstance(value, os.PathLike):
        path = Path(value).expanduser()
    elif isinstance(value, str) and value.strip():
        path = Path(value).expanduser()
    else:
        return None
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _dataset_members(source: Path) -> list[Path]:
    if source.suffix.casefold() != ".shp":
        return [source] if source.is_file() else []
    stem_text = source.stem.casefold()
    return sorted(
        member for member in source.parent.iterdir()
        if member.is_file()
        and member.name.casefold().startswith(stem_text + ".")
        and any(member.name.casefold().endswith(suffix) for suffix in SHAPEFILE_SIDECAR_SUFFIXES)
    )


def copy_dataset(source: Path | str, destination: Path | str) -> Path:
    """Copy one file or every component of a Shapefile dataset."""
    origin = _resolved(source)
    target = _resolved(destination)
    if origin == target:
        return target
    members = _dataset_members(origin)
    if not members or not origin.is_file():
        raise FileNotFoundError(f"找不到待发布成果：{origin}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if origin.suffix.casefold() != ".shp":
        shutil.copy2(origin, target)
        return target
    for member in members:
        suffix = member.name[len(origin.stem):]
        shutil.copy2(member, target.with_name(target.stem + suffix))
    return target


def shapefile_has_records(path: Path | str | None) -> bool:
    """Read the DBF header count without loading a geospatial dataframe."""
    source = _manifest_path(path)
    if source is None or source.suffix.casefold() != ".shp":
        return False
    dbf = source.with_suffix(".dbf")
    try:
        with dbf.open("rb") as stream:
            header = stream.read(8)
    except OSError:
        return False
    return len(header) == 8 and int.from_bytes(header[4:8], "little") > 0


def empty_result_index(layout: ProjectLayout) -> dict:
    return {
        "schema_version": RESULT_INDEX_VERSION,
        "project_root": str(layout.project_root),
        "results_root": str(layout.results_root),
        "updated_at": "",
        "areas": {},
        "task_report": {},
    }


def read_result_index(path: Path | str) -> dict | None:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / RESULT_INDEX_NAME
    if not source.is_file():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("areas"), dict) else None


class ResultPublisher:
    """Publish current formal products and maintain their authoritative index."""

    def __init__(
        self, output_root: Path | str, *, project_root: Path | str | None = None,
    ) -> None:
        self.layout = (
            ProjectLayout.from_project(project_root, output_root)
            if project_root is not None else ProjectLayout.from_output(output_root)
        )
        self.layout.results_root.mkdir(parents=True, exist_ok=True)
        self.index = read_result_index(self.layout.result_index_path) or empty_result_index(self.layout)

    def _area(self, area: object) -> dict:
        name = str(area or "validation")
        return self.index["areas"].setdefault(name, {
            "periods": {}, "changes": {}, "temporal": {}, "evaluation": {},
        })

    def _save(self, source_manifest: Path | str | None = None) -> Path:
        # source_manifest is intentionally not serialized: task/run history is
        # internal state and must not leak into the user-facing result index.
        self.index["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return atomic_write_json(self.layout.result_index_path, self.index)

    @staticmethod
    def _copy_fields(
        sources: dict[str, object], target_dir: Path,
        mapping: Iterable[tuple[str, str]], *, base_dir: Path | None = None,
    ) -> dict[str, str]:
        published: dict[str, str] = {}
        for key, filename in mapping:
            source = _manifest_path(sources.get(key), base_dir)
            if source is None or not source.is_file():
                continue
            published[key] = str(copy_dataset(source, target_dir / filename))
        return published

    def publish_period(
        self, area: object, period: object, result: dict,
        *, run_id: object = "", base_dir: Path | None = None, save: bool = True,
    ) -> dict[str, str]:
        target = (
            self.layout.results_root / safe_name(area) / "01_单期道路" / safe_name(period)
        )
        published = self._copy_fields(result, target, (
            ("centerlines", "road_centerlines.shp"),
            ("surfaces", "road_surfaces.shp"),
            ("width_segments", "road_width_segments.shp"),
            ("corridors", "road_corridors.shp"),
        ), base_dir=base_dir)
        previews = result.get("previews") if isinstance(result.get("previews"), dict) else {}
        extraction_preview = result.get("road_extraction")
        if not extraction_preview:
            candidate = _manifest_path(previews.get("fusion"), base_dir)
            if candidate is not None and candidate.name.casefold() in {
                "road_overview.png", "road_extraction.png",
            }:
                extraction_preview = str(candidate)
        width_preview = result.get("road_width")
        if not width_preview:
            candidate = _manifest_path(previews.get("width"), base_dir)
            if candidate is not None and candidate.name.casefold() in {
                "road_width_overview.png", "road_width.png",
            }:
                width_preview = str(candidate)
        published.update(self._copy_fields(
            {"road_extraction": extraction_preview, "road_width": width_preview},
            target,
            (("road_extraction", "road_extraction.png"), ("road_width", "road_width.png")),
            base_dir=base_dir,
        ))
        if isinstance(result.get("previews"), dict):
            for key, filename in (
                ("road_extraction", "road_extraction.png"),
                ("road_width", "road_width.png"),
            ):
                if key not in published:
                    (target / filename).unlink(missing_ok=True)
        if published:
            self._area(area)["periods"][str(period)] = {
                **published, "status": "已生成",
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if save:
                self._save()
        return published

    def publish_change(
        self, area: object, before: object, after: object, result: dict,
        *, run_id: object = "", base_dir: Path | None = None, save: bool = True,
    ) -> dict[str, str]:
        pair = f"{safe_name(before)}_to_{safe_name(after)}"
        target = self.layout.results_root / safe_name(area) / "02_变化检测" / pair
        layers = dict(result.get("layers") or {})
        layers.setdefault("changes", result.get("road_changes"))
        published = self._copy_fields(layers, target, (
            ("changes", "road_changes.shp"),
            ("added", "added_roads.shp"),
            ("removed", "removed_roads.shp"),
            ("widened", "widened_road_parts.shp"),
            ("narrowed", "narrowed_road_parts.shp"),
        ), base_dir=base_dir)
        previews = result.get("previews") if isinstance(result.get("previews"), dict) else {}
        review_preview = result.get("review_change")
        if not review_preview and shapefile_has_records(layers.get("review")):
            review_preview = previews.get("review_change")
        published.update(self._copy_fields(
            {
                "road_change": result.get("road_change") or previews.get("change"),
                "review_change": review_preview,
            },
            target,
            (("road_change", "road_change.png"), ("review_change", "review_change.png")),
            base_dir=base_dir,
        ))
        if "review_change" not in published:
            (target / "review_change.png").unlink(missing_ok=True)
        if published:
            self._area(area)["changes"][f"{before}_to_{after}"] = {
                **published, "before_period": str(before), "after_period": str(after),
                "status": "已生成",
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if save:
                self._save()
        return published

    def publish_temporal(
        self, area: object, result: dict, *, base_dir: Path | None = None,
        save: bool = True,
    ) -> dict[str, str]:
        target = self.layout.results_root / safe_name(area) / "03_长时序"
        published = self._copy_fields(result, target, (
            ("life_shp", "road_life.shp"),
            ("observations_shp", "road_obs.shp"),
            ("events_shp", "road_event.shp"),
            ("event_parts_shp", "event_parts.shp"),
            ("lineage_shp", "road_lineage.shp"),
            ("review_shp", "road_review.shp"),
        ), base_dir=base_dir)
        if published:
            self._area(area)["temporal"] = {
                **published, "status": "已生成",
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if save:
                self._save()
        return published

    def publish_evaluation(
        self, areas: Iterable[object], summary: dict, *, base_dir: Path | None = None,
        save: bool = True,
    ) -> None:
        sources = {"csv": summary.get("csv"), "json": summary.get("json")}
        for area in areas:
            target = self.layout.results_root / safe_name(area) / "04_精度评价"
            published = self._copy_fields(sources, target, (
                ("csv", "evaluation_summary.csv"),
                ("json", "evaluation_summary.json"),
            ), base_dir=base_dir)
            if published:
                self._area(area)["evaluation"] = {
                    **published, "status": "已生成",
                    "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
        if save:
            self._save()

    def publish_reports(self, job_root: Path | str, *, save: bool = True) -> None:
        root = _resolved(job_root)
        published = self._copy_fields(
            {"csv": str(root / "task_report.csv"), "json": str(root / "task_report.json")},
            self.layout.results_root,
            (("csv", "task_report.csv"), ("json", "task_report.json")),
        )
        if published:
            self.index["task_report"] = published
        if save:
            self._save()

    def publish_manifest(
        self, manifest: dict, *, source_manifest: Path | str | None = None,
    ) -> dict:
        run_id = manifest.get("run_id", "")
        for entry in manifest.get("period_results", []) or []:
            if isinstance(entry, dict) and entry.get("status") not in {"failed", "stale"}:
                published = self.publish_period(
                    entry.get("grid"), entry.get("period"), entry,
                    run_id=run_id, save=False,
                )
                if published:
                    entry["published"] = published
        for entry in manifest.get("change_results", []) or []:
            if isinstance(entry, dict) and entry.get("status") not in {"failed", "stale"}:
                published = self.publish_change(
                    entry.get("grid"), entry.get("before_period"), entry.get("after_period"),
                    entry, run_id=run_id, save=False,
                )
                if published:
                    entry["published"] = published
        for entry in manifest.get("temporal_results", []) or []:
            if isinstance(entry, dict):
                published = self.publish_temporal(entry.get("grid"), entry, save=False)
                if published:
                    entry["published"] = published
        areas = list(self.index.get("areas", {}))
        summary = manifest.get("evaluation_summary")
        if isinstance(summary, dict) and areas:
            self.publish_evaluation(areas, summary, save=False)
        job_root = _manifest_path(manifest.get("job_root"))
        if job_root is not None:
            self.publish_reports(job_root, save=False)
        self._save(source_manifest)
        manifest["result_index"] = str(self.layout.result_index_path)
        manifest["output_root"] = str(self.layout.results_root)
        manifest["project_root"] = str(self.layout.project_root)
        return self.index


def result_index_from_manifest(
    manifest: dict, base_dir: Path | None = None,
) -> dict:
    """Build a read-only business index for a legacy manifest without migration."""
    project_value = manifest.get("project_root") or (base_dir.parent if base_dir else Path.cwd())
    output_value = manifest.get("output_root") or (Path(project_value) / RESULT_DIRECTORY_NAME)
    layout = ProjectLayout.from_project(project_value, output_value)
    index = empty_result_index(layout)

    def path_text(value: object) -> str:
        path = _manifest_path(value, base_dir)
        return str(path) if path is not None else ""

    for entry in manifest.get("period_results", []) or []:
        if not isinstance(entry, dict):
            continue
        area, period = str(entry.get("grid") or "validation"), str(entry.get("period") or "未命名期次")
        area_node = index["areas"].setdefault(area, {"periods": {}, "changes": {}, "temporal": {}, "evaluation": {}})
        published = entry.get("published") if isinstance(entry.get("published"), dict) else {}
        previews = entry.get("previews") if isinstance(entry.get("previews"), dict) else {}
        area_node["periods"][period] = {
            key: path_text(
                published.get(key)
                or entry.get(key)
                or (previews.get("fusion") if key == "road_extraction" else None)
                or (previews.get("width") if key == "road_width" else None)
            )
            for key in (
                "centerlines", "surfaces", "width_segments", "corridors",
                "road_extraction", "road_width",
            )
            if published.get(key) or entry.get(key)
            or (previews and (
                (key == "road_extraction" and previews.get("fusion"))
                or (key == "road_width" and previews.get("width"))
            ))
        }
    for entry in manifest.get("change_results", []) or []:
        if not isinstance(entry, dict):
            continue
        area = str(entry.get("grid") or "validation")
        before, after = str(entry.get("before_period") or "前期"), str(entry.get("after_period") or "后期")
        area_node = index["areas"].setdefault(area, {"periods": {}, "changes": {}, "temporal": {}, "evaluation": {}})
        published = entry.get("published") if isinstance(entry.get("published"), dict) else {}
        layers = entry.get("layers") if isinstance(entry.get("layers"), dict) else {}
        previews = entry.get("previews") if isinstance(entry.get("previews"), dict) else {}
        changes_value = published.get("changes") or layers.get("changes") or entry.get("gpkg") or entry.get("summary")
        area_node["changes"][f"{before}_to_{after}"] = {
            "before_period": before, "after_period": after,
            **({"changes": path_text(changes_value)} if changes_value else {}),
            **{
                key: path_text(
                    published.get(key) or layers.get(key)
                    or (previews.get("change") if key == "road_change" else None)
                    or (previews.get("review_change") if key == "review_change" else None)
                )
                for key in (
                    "added", "removed", "widened", "narrowed",
                    "road_change", "review_change",
                )
                if published.get(key) or layers.get(key)
                or (previews and (
                    (key == "road_change" and previews.get("change"))
                    or (key == "review_change" and previews.get("review_change"))
                ))
            },
        }
    for entry in manifest.get("temporal_results", []) or []:
        if not isinstance(entry, dict):
            continue
        area = str(entry.get("grid") or "validation")
        area_node = index["areas"].setdefault(area, {"periods": {}, "changes": {}, "temporal": {}, "evaluation": {}})
        published = entry.get("published") if isinstance(entry.get("published"), dict) else {}
        area_node["temporal"] = {
            key: path_text(published.get(key) or entry.get(key))
            for key in ("life_shp", "observations_shp", "events_shp", "event_parts_shp", "lineage_shp", "review_shp")
            if published.get(key) or entry.get(key)
        }
    summary = manifest.get("evaluation_summary")
    if isinstance(summary, dict):
        for area_node in index["areas"].values():
            area_node["evaluation"] = {
                key: path_text(summary.get(key)) for key in ("csv", "json") if summary.get(key)
            }
    job_root = _manifest_path(manifest.get("job_root"), base_dir)
    if job_root is not None:
        index["task_report"] = {
            key: str(job_root / f"task_report.{key}") for key in ("csv", "json")
        }
    return index
