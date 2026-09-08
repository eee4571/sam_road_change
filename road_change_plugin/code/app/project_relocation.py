from __future__ import annotations

"""Safe, centralized relocation of copied project-owned task paths."""

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


BACKUP_SUFFIX = ".pre_relocation.bak"
RELOCATION_SCHEMA_VERSION = 1
_HISTORICAL_KEYS = {"provenance", "relocation_history"}


def _resolved(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        common = os.path.commonpath((os.path.normcase(str(path)), os.path.normcase(str(root))))
    except ValueError:
        return None
    if common != os.path.normcase(str(root)):
        return None
    try:
        return Path(os.path.relpath(path, root))
    except ValueError:
        return None


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(_relative_to(path, root) is not None for root in roots)


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _backup_once(path: Path) -> Path:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        temporary = backup.with_name(f".{backup.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(path.read_bytes())
            os.replace(temporary, backup)
        finally:
            temporary.unlink(missing_ok=True)
    return backup


@dataclass(frozen=True)
class ProjectRelocationPlan:
    run_id: str
    state_path: Path
    old_project_root: Path
    new_project_root: Path
    old_output_root: Path
    new_output_root: Path
    old_job_root: Path
    new_job_root: Path

    @property
    def mappings(self) -> tuple[tuple[Path, Path], ...]:
        pairs = (
            (self.old_job_root, self.new_job_root),
            (self.old_output_root, self.new_output_root),
            (self.old_project_root, self.new_project_root),
        )
        return tuple(sorted(pairs, key=lambda item: len(str(item[0])), reverse=True))

    @property
    def allowed_targets(self) -> tuple[Path, ...]:
        return (self.new_project_root, self.new_output_root, self.new_job_root)

    def map_path(self, value: Path | str) -> Path | None:
        path = Path(value).expanduser()
        if not path.is_absolute():
            return None
        source = path.resolve()
        for old_root, new_root in self.mappings:
            relative = _relative_to(source, old_root)
            if relative is None:
                continue
            target = (new_root / relative).resolve()
            if not _inside(target, self.allowed_targets):
                raise ValueError(f"项目重定位目标逃出当前项目目录：{source} -> {target}")
            return target
        return None

    def relocate_tree(self, value, *, key: str = ""):
        if key in _HISTORICAL_KEYS:
            return copy.deepcopy(value)
        if isinstance(value, dict):
            return {
                child_key: self.relocate_tree(child, key=str(child_key))
                for child_key, child in value.items()
            }
        if isinstance(value, list):
            return [self.relocate_tree(child, key=key) for child in value]
        if isinstance(value, str) and value.strip():
            mapped = self.map_path(value)
            return str(mapped) if mapped is not None else value
        return copy.deepcopy(value)

    def audit_record(self) -> dict:
        return {
            "schema_version": RELOCATION_SCHEMA_VERSION,
            "kind": "project_relocation",
            "old_project_root": str(self.old_project_root),
            "new_project_root": str(self.new_project_root),
            "old_output_root": str(self.old_output_root),
            "new_output_root": str(self.new_output_root),
            "old_job_root": str(self.old_job_root),
            "new_job_root": str(self.new_job_root),
            "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "verified": True,
        }


def _legacy_project_root(recorded_job: Path, recorded_output: Path | None) -> Path:
    if recorded_output is not None:
        return recorded_output.parent
    parts = [part.casefold() for part in recorded_job.parts]
    for marker in ("04_成果输出", "成果输出", "_work"):
        if marker.casefold() in parts:
            index = parts.index(marker.casefold())
            return Path(*recorded_job.parts[:index])
    raise ValueError(f"无法从旧任务目录确定旧项目根目录：{recorded_job}")


def build_relocation_plan(
    state: dict, state_path: Path | str, *, run_id: str,
    current_project_root: Path | str, current_output_root: Path | str,
    current_job_root: Path | str,
) -> ProjectRelocationPlan | None:
    actual_state = _resolved(state_path)
    new_project = _resolved(current_project_root)
    new_output = _resolved(current_output_root)
    new_job = _resolved(current_job_root)
    if not actual_state.is_file() or actual_state.parent != new_job:
        raise ValueError(f"任务状态不位于实际任务目录中：{actual_state}")
    if str(state.get("run_id") or "").strip() != str(run_id).strip():
        raise ValueError("任务状态 run_id 与请求不一致，不能执行项目重定位。")
    if new_job.name.casefold() != str(run_id).casefold():
        raise ValueError(f"实际任务目录名与 run_id 不一致：{new_job.name} != {run_id}")
    recorded_text = str(state.get("job_root") or "").strip()
    if not recorded_text:
        return None
    old_job = Path(recorded_text).expanduser()
    if not old_job.is_absolute():
        return None
    old_job = old_job.resolve()
    if old_job == new_job:
        return None
    if old_job.name.casefold() != str(run_id).casefold():
        raise ValueError(f"旧任务目录名与 run_id 不一致：{old_job.name} != {run_id}")
    if not _inside(new_job, (new_project, new_output)):
        raise ValueError(f"实际任务目录不在当前项目或成果目录内：{new_job}")
    output_text = str(state.get("output_root") or "").strip()
    old_output = Path(output_text).expanduser().resolve() if output_text and Path(output_text).expanduser().is_absolute() else old_job.parent
    project_text = str(state.get("project_root") or "").strip()
    old_project = (
        Path(project_text).expanduser().resolve()
        if project_text and Path(project_text).expanduser().is_absolute()
        else _legacy_project_root(old_job, old_output)
    )
    plan = ProjectRelocationPlan(
        str(run_id), actual_state, old_project, new_project,
        old_output, new_output, old_job, new_job,
    )
    mapped_state = plan.map_path(old_job / "job_state.json")
    if mapped_state != actual_state:
        raise ValueError(
            "旧任务目录与当前目录没有相同的任务内结构："
            f"{old_job / 'job_state.json'} -> {mapped_state}，实际为 {actual_state}"
        )
    return plan


def _json_paths(value, plan: ProjectRelocationPlan):
    found: set[Path] = set()
    def visit(item, key=""):
        if key in _HISTORICAL_KEYS:
            return
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif isinstance(item, str) and item.strip().casefold().endswith(".json"):
            mapped = plan.map_path(item)
            path = mapped or Path(item).expanduser()
            if path.is_absolute() and path.is_file() and _inside(path.resolve(), plan.allowed_targets):
                found.add(path.resolve())
    visit(value)
    return found


def _required_json_paths(state: dict, plan: ProjectRelocationPlan) -> set[Path]:
    values = [state.get("period_state")]
    values.extend(
        entry.get("result") for entry in state.get("period_results", []) or []
        if isinstance(entry, dict)
    )
    values.extend(
        entry.get("summary") for entry in state.get("change_results", []) or []
        if isinstance(entry, dict)
    )
    evaluation = state.get("evaluation_summary")
    if isinstance(evaluation, dict):
        values.append(evaluation.get("json"))
    required: set[Path] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or not text.casefold().endswith(".json"):
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = plan.new_job_root / path
        path = path.resolve()
        if not _inside(path, plan.allowed_targets):
            raise ValueError(f"任务必要 JSON 不在当前项目目录内：{path}")
        if not path.is_file():
            raise ValueError(f"项目重定位所需文件不存在：{path}")
        required.add(path)
    return required


def relocate_state_files(state: dict, plan: ProjectRelocationPlan) -> tuple[dict, list[Path]]:
    """Relocate the selected task and referenced internal JSON files atomically."""
    relocated = plan.relocate_tree(state)
    history = list(relocated.get("relocation_history") or [])
    signature = (str(plan.old_job_root), str(plan.new_job_root))
    if not any(
        isinstance(row, dict)
        and (str(row.get("old_job_root")), str(row.get("new_job_root"))) == signature
        for row in history
    ):
        history.append(plan.audit_record())
    relocated["relocation_history"] = history
    candidates = {
        plan.state_path,
        plan.new_job_root / "pipeline_result.json",
        plan.new_output_root / "result_index.json",
    }
    candidates.update(
        path.resolve() for path in plan.new_job_root.rglob("*.json")
        if BACKUP_SUFFIX not in path.name
    )
    candidates.update(_json_paths(relocated, plan))
    required = _required_json_paths(relocated, plan)
    candidates.update(required)
    for path in required:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"项目重定位所需文件 JSON 损坏或无法读取：{path}：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"项目重定位所需文件 JSON 根节点必须是对象：{path}")
    written: list[Path] = []
    for path in sorted(candidates, key=lambda item: (item == plan.state_path, str(item))):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            if path == plan.state_path or path in required:
                label = "任务状态" if path == plan.state_path else "项目重定位所需文件"
                raise ValueError(f"{label} JSON 损坏或无法读取：{path}：{exc}") from exc
            continue
        if not isinstance(payload, dict):
            continue
        mapped = relocated if path == plan.state_path else plan.relocate_tree(payload)
        if path == plan.state_path:
            _backup_once(path)
        _atomic_write_json(path, mapped)
        written.append(path)
    return relocated, written


def active_old_path_references(value, old_roots: tuple[Path, ...]) -> list[str]:
    references: list[str] = []
    def visit(item, key=""):
        if key in _HISTORICAL_KEYS:
            return
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif isinstance(item, str) and item.strip():
            path = Path(item).expanduser()
            if path.is_absolute() and _inside(path.resolve(), old_roots):
                references.append(item)
    visit(value)
    return sorted(set(references))


def rebase_project_owned_tree(value, old_root: Path | str, new_root: Path | str):
    """Return a read-only, boundary-safe rebase for project discovery views."""
    old = _resolved(old_root)
    new = _resolved(new_root)

    def visit(item, key=""):
        if key in _HISTORICAL_KEYS:
            return copy.deepcopy(item)
        if isinstance(item, dict):
            return {child_key: visit(child, str(child_key)) for child_key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, key) for child in item]
        if not isinstance(item, str) or not item.strip():
            return copy.deepcopy(item)
        path = Path(item).expanduser()
        if not path.is_absolute():
            return item
        relative = _relative_to(path.resolve(), old)
        if relative is None:
            return item
        target = (new / relative).resolve()
        if not _inside(target, (new,)):
            raise ValueError(f"项目浏览路径重定位逃出当前项目目录：{path} -> {target}")
        return str(target)

    return visit(value)


@dataclass(frozen=True)
class BatchListRepairResult:
    checked_lists: int
    modified_lists: int
    modified_paths: int
    checked_json: int


def _path_basename(value: str) -> str:
    """Return a filename for either Windows or POSIX absolute path text."""
    windows_name = PureWindowsPath(value).name
    posix_name = Path(value.replace("\\", "/")).name
    return windows_name or posix_name


def _validate_active_task_json(job_root: Path) -> int:
    candidates = {job_root / "job_state.json", job_root / "pipeline_result.json"}
    for name in ("period_state.json", "latest_result.json", "input_manifest.json"):
        candidates.update(job_root.glob(f"grids/*/periods/*/{name}"))
    checked = 0
    for path in sorted(candidates, key=str):
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"当前任务活动 JSON 损坏或无法读取：{path}：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"当前任务活动 JSON 根节点必须是对象：{path}")
        checked += 1
    return checked


def repair_task_batch_lists(job_root: Path | str) -> BatchListRepairResult:
    """Rebase current-run batch lists to their sibling ``images`` directory.

    Every list is preflighted before any file is changed, so a missing copied
    image cannot leave the task half migrated.  Lists remain UTF-8 with BOM and
    keep their original line order and count.
    """
    root = _resolved(job_root)
    if not root.is_dir():
        raise ValueError(f"当前任务目录不存在：{root}")
    checked_json = _validate_active_task_json(root)
    plans: list[tuple[Path, str, str, int]] = []
    missing: list[Path] = []
    list_paths = sorted(root.glob("grids/*/periods/*/batches/*.txt"), key=str)
    for list_path in list_paths:
        try:
            original = list_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"影像清单无法按 utf-8-sig 读取：{list_path}：{exc}") from exc
        period_root = list_path.parent.parent.resolve()
        images_root = (period_root / "images").resolve()
        if _relative_to(images_root, root) is None:
            raise ValueError(f"影像清单映射目录逃出当前任务：{images_root}")
        lines = original.splitlines()
        rewritten: list[str] = []
        changed_paths = 0
        for line in lines:
            text = line.strip()
            if not text:
                rewritten.append(line)
                continue
            filename = _path_basename(text)
            if not filename or filename in {".", ".."}:
                raise ValueError(f"影像清单包含无效路径：{list_path}：{text}")
            target = (images_root / filename).resolve()
            if _relative_to(target, images_root) is None or _relative_to(target, root) is None:
                raise ValueError(f"影像清单映射目标逃出当前任务：{text} -> {target}")
            try:
                valid = target.is_file() and target.stat().st_size > 0
            except OSError:
                valid = False
            if not valid:
                missing.append(target)
            target_text = str(target)
            rewritten.append(target_text)
            if text != target_text:
                changed_paths += 1
        trailing_newline = original.endswith(("\n", "\r"))
        replacement = "\n".join(rewritten) + ("\n" if trailing_newline else "")
        plans.append((list_path, original, replacement, changed_paths))
    if missing:
        details = "\n".join(f"- {path}" for path in sorted(set(missing), key=str))
        raise FileNotFoundError(
            "复制后的当前项目缺少影像清单所需文件；不会回退读取其他项目，也不会重新切片：\n"
            + details
        )
    modified_lists = 0
    modified_paths = 0
    for list_path, original, replacement, changed_paths in plans:
        if replacement == original:
            continue
        _backup_once(list_path)
        temporary = list_path.with_name(f".{list_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(replacement, encoding="utf-8-sig")
            os.replace(temporary, list_path)
        finally:
            temporary.unlink(missing_ok=True)
        modified_lists += 1
        modified_paths += changed_paths
    return BatchListRepairResult(
        checked_lists=len(plans), modified_lists=modified_lists,
        modified_paths=modified_paths, checked_json=checked_json,
    )
