"""Read-only file selection model for the UI. No GIS or processing imports."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"runtime", "env", "model", "models", ".git", "__pycache__", "_work",
            "_logs", "成果输出", "04_成果输出", "outputs", "code", "plugin"}


def natural(value):
    return tuple((0, int(x)) if x.isdigit() else (1, x.casefold())
                 for x in re.split(r"(\d+)", str(value)))


def resolve(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_object(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"不是有效项目配置：{path.name}")
    return value


def pairs(periods):
    result = []
    for area in sorted({row[0] for row in periods}, key=natural):
        names = sorted({r[1] for r in periods if r[0] == area}, key=natural)
        result.extend((area, a, b) for a, b in zip(names, names[1:]))
    return result


def discover_tasks(root, output=None):
    """Only inspect formal task-index locations, never recurse into products."""
    root = Path(root)
    candidates = [root / "_work/tasks/latest_pipeline.json"]
    candidates += list((root / "_work/tasks/runs").glob("*/pipeline_result.json"))
    for out in {root / "成果输出", root / "04_成果输出", Path(output) if output else root / "成果输出"}:
        candidates += [out / "latest_pipeline.json"]
        candidates += list((out / "runs").glob("*/pipeline_result.json"))
    tasks, seen = [], set()
    for path in sorted((p for p in candidates if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = read_object(path)
            task_id = str(data.get("run_id") or path.parent.name)
            if task_id in seen:
                continue
            seen.add(task_id)
            tasks.append({"path": str(path.resolve()), "id": task_id, "data": data,
                          "status": data.get("status", "unknown")})
        except (OSError, ValueError):
            continue
    return tasks


def scan_project(directory):
    root = resolve(directory, ROOT)
    if not root.is_dir():
        raise ValueError("请选择存在的项目或数据目录")
    model = dict(root=str(root), areas=[], periods=[], truths=[], issues=[], output=str(root / "成果输出"))
    config_path = root / "project_config.json"
    if config_path.is_file():
        config = read_object(config_path)
        old_root = Path(str(config.get("project_root", ".")))

        def configured(value):
            path = Path(str(value))
            if path.is_absolute() and old_root.is_absolute() and path.is_relative_to(old_root):
                return str((root / path.relative_to(old_root)).resolve())
            return str(resolve(str(value), root))

        model["areas"] = [[str(a), configured(p)] for a, p in config.get("validation_areas", [])]
        model["periods"] = [[str(a), str(t), configured(p)] for a, rows in config.get("area_periods", {}).items() for t, p in rows]
        model["truths"] = [[str(a), str(b), str(c), configured(p)] for a, b, c, p in config.get("area_truths", [])]
        if config.get("output_root"):
            model["output"] = configured(config["output_root"])
    if not model["areas"]:
        def named(parent, prefix, aliases):
            found = sorted((p for p in parent.iterdir() if p.is_dir() and
                            (p.name.startswith(prefix) or p.name.casefold() in aliases)), key=lambda p: natural(p.name))
            if len(found) > 1:
                model["issues"].append(f"{parent.name} 有多个 {prefix} 目录，请修正识别结果")
            return found[0] if found else None

        def collect(parent, nested):
            boundary = named(parent, "01_", {"validation", "validation_area", "验证区"})
            imagery = named(parent, "02_", {"images", "imagery", "影像"})
            if boundary is None or imagery is None:
                return False
            bounds = sorted((p for p in boundary.iterdir() if p.suffix.lower() == ".shp"), key=lambda p: natural(p.stem))
            if nested and len(bounds) != 1:
                model["issues"].append(f"{parent.name} 的验证区 SHP 数量不是 1，请修正")
            truth = named(parent, "03_", {"truth", "ground_truth", "变化真值", "gt"})
            for shp in bounds:
                area = parent.name if nested and len(bounds) == 1 else shp.stem
                model["areas"].append([area, str(shp)])
                rows = sorted((p for p in imagery.iterdir() if p.suffix.lower() == ".txt"), key=lambda p: natural(p.stem))
                model["periods"] += [[area, p.stem, str(p)] for p in rows]
                truths = {}
                if truth:
                    for folder in (truth, truth / area):
                        if folder.is_dir():
                            for p in folder.iterdir():
                                match = re.fullmatch(r"(.+?)_to_(.+)", p.stem, re.I)
                                if p.suffix.lower() == ".shp" and match:
                                    truths[match.groups()] = str(p)
                model["truths"] += [[area, b, a, p] for (b, a), p in truths.items()]
            return True

        if not collect(root, False):
            for child in sorted(root.iterdir(), key=lambda p: natural(p.name)):
                if child.is_dir() and child.name not in EXCLUDED and not child.name.startswith(".") and not child.is_symlink():
                    collect(child, True)
    if not model["areas"]:
        model["issues"].append("未识别到验证区。支持项目配置，或 01_验证区 / 02_影像 / 03_变化真值 目录。")
    model["tasks"] = discover_tasks(root, model["output"])
    return model


def check_files(model):
    """Shallow input-file checks only. Spatial validation belongs to backend."""
    issues = []
    expected = set(pairs(model["periods"]))
    names = [r[0] for r in model["areas"]]
    if not names or len(set(names)) != len(names) or any(not n for n in names):
        issues.append("区域名称为空或重复，请修正")
    for name in names:
        rows = [r for r in model["periods"] if r[0] == name]
        if len(rows) < 2 or len({r[1] for r in rows}) != len(rows):
            issues.append(f"{name} 需要至少两个不同期次")
    for kind, rows in (("验证区", model["areas"]), ("影像清单", model["periods"]), ("真值数据", model["truths"])):
        for row in rows:
            path = Path(row[-1])
            label = " / ".join(row[:-1])
            if row[0] not in names:
                issues.append(f"{label} 引用了不存在的区域")
            if not path.is_file():
                issues.append(f"{label}：{kind} 文件不存在")
                continue
            if kind in {"验证区", "真值数据"}:
                for suffix in (".shp", ".shx", ".dbf", ".prj"):
                    if not path.with_suffix(suffix).is_file():
                        issues.append(f"{label} 缺少 {suffix} 文件")
            if kind == "影像清单":
                if path.stat().st_size > 8 * 1024 * 1024:
                    issues.append(f"{label} 清单过大，请拆分后再检查")
                    continue
                raw = path.read_bytes()
                try:
                    content = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    content = raw.decode("gb18030")
                lines = [line.strip().strip('"') for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#")]
                if not lines:
                    issues.append(f"{label} 影像清单为空")
                missing = sum(not resolve(line, path.parent).is_file() for line in lines)
                if missing:
                    issues.append(f"{label} 有 {missing} 个影像路径不存在")
    gt_keys = [tuple(r[:3]) for r in model["truths"]]
    if len(set(gt_keys)) != len(gt_keys):
        issues.append("同一变化对存在重复真值数据")
    if any(k not in expected for k in gt_keys):
        issues.append("真值数据的期次未对应相邻变化对")
    return issues
