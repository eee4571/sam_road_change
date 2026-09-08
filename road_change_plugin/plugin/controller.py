"""User intent and JSON data only; the backend stays in another interpreter."""
import json
import re
from pathlib import Path
from .runner import ROOT, Runner
from .signals import TaskSignals, forward
from .result_parser import read_results


class Controller(TaskSignals):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runner = Runner(self)
        forward(self.runner, self)
        self.runner.task_finished.connect(self._completed)
        self.manifest = ""
        self.history = []
        self.task_log.connect(self._record)

    def _record(self, event):
        self.history.append(event)
        del self.history[:-2000]

    def path(self, value):
        path = Path(str(value).strip())
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    def input_file(self, value, suffix=None):
        if not str(value).strip():
            raise ValueError("输入路径不能为空")
        path = self.path(value)
        if not path.is_file() or (suffix and path.suffix.lower() != suffix):
            raise ValueError(f"请选择存在的 {suffix or ''} 文件：{path}")
        return str(path)

    def build_command(self, action, data):
        if action != "all":
            manifest = self.input_file(data.get("manifest", ""), ".json")
            if action == "evaluate-all-existing":
                args = [action, "--pipeline-manifest", manifest]
                for row in data.get("truths", []):
                    args += ["--truth", *row[:3], self.input_file(row[3], ".shp")]
                if data.get("truth_type_field", "").strip():
                    args += ["--truth-type-field", data["truth_type_field"].strip()]
                return args
            if action not in {"rerun-period", "rerun-change"}:
                raise ValueError("不支持的任务")
            grid = data.get("grid", "").strip()
            if not grid:
                raise ValueError("请选择区域")
            args = [action, "--pipeline-manifest", manifest, "--grid", grid]
            keys = ["period"] if action == "rerun-period" else ["before_period", "after_period"]
            for key in keys:
                if not data.get(key, "").strip():
                    raise ValueError("请选择期次或变化对")
                args += ["--" + key.replace("_", "-"), data[key]]
            args += ["--update-related" if action == "rerun-period" else "--update-temporal"]
            return args
        areas, periods = data.get("areas", []), data.get("periods", [])
        if not areas:
            raise ValueError("请添加至少一个验证区 SHP")
        names = [r[0].strip() for r in areas]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("区域名称不能为空或重复")
        args = ["all", "--mode", "validation", "--execution-profile", data.get("profile", "full")]
        for name, area in areas:
            args += ["--validation-area", name, self.input_file(area, ".shp")]
            rows = [r for r in periods if r[0] == name]
            if len(rows) < 2 or len({r[1] for r in rows}) != len(rows) or any(not r[1].strip() for r in rows):
                raise ValueError(f"{name} 至少需要两个不重复期次及影像 TXT")
            for _, period, source in rows:
                args += ["--period", name, period, self.input_file(source, ".txt")]
        if any(r[0] not in names for r in periods):
            raise ValueError("影像期次引用了不存在的区域")
        for row in data.get("truths", []):
            if row[0] not in names:
                raise ValueError("真值引用了不存在的区域")
            args += ["--truth", *row[:3], self.input_file(row[3], ".shp")]
        if not data.get("evaluate", False):
            args.append("--no-evaluation")
        if not data.get("output", "").strip():
            raise ValueError("请选择成果输出目录")
        args += ["--output-root", str(self.path(data["output"])),
                 "--checkpoint", str(ROOT / "runtime/model/samroad/samroad.ckpt"),
                 "--config", str(ROOT / "runtime/config/samroad_inference.yaml"),
                 "--device", data.get("device", "auto"), "--runtime-preflight"]
        for key, default in (("pixel-size", "0.0"), ("absolute", "2.0"), ("ratio", "0.2"), ("tolerance", "3.0")):
            value = str(data.get(key, default))
            if not 0 <= float(value) < float("inf"):
                raise ValueError(f"{key} 必须是非负有限数")
            args += ["--" + key, value]
        run_id = data.get("run_id", "").strip()
        if run_id:
            if not re.fullmatch(r"[\w-]+", run_id):
                raise ValueError("任务名称仅使用文字、数字、下划线和连字符")
            args += ["--run-id", run_id]
        if data.get("resume"):
            if not run_id:
                raise ValueError("续跑需填写原任务名称")
            args += ["--resume"]
        if data.get("truth_type_field", "").strip():
            args += ["--truth-type-field", data["truth_type_field"].strip()]
        return args

    def run(self, action, data):
        try:
            if self.runner.running:
                raise ValueError("当前任务尚未结束")
            args = self.build_command(action, data)
            self.manifest = data.get("manifest", "")
            profile = data.get("profile", "full")
            if action != "all":
                profile = json.loads(self.path(self.manifest).read_text(encoding="utf-8")).get("execution_profile", "full")
            self.runner.start(args, profile)
        except (OSError, ValueError, TypeError) as exc:
            self.task_failed.emit({"plugin_id": "road_change", "task_id": "", "message": str(exc), "detail": "输入检查", "status": "failed"})

    def _completed(self, payload):
        if payload.get("status") != "completed":
            return
        event = payload.get("event", {})
        manifest = event.get("manifest") or event.get("pipeline_manifest") or self.manifest
        if manifest:
            self.load_results(manifest, payload.get("task_id", ""))

    def load_results(self, manifest, task_id=""):
        try:
            self.manifest = str(self.path(manifest))
            for result in read_results(self.path(manifest)):
                self.result_ready.emit({"plugin_id": "road_change", "task_id": task_id, **result})
        except (OSError, ValueError, TypeError) as exc:
            self.task_log.emit({"task_id": task_id, "level": "ERROR", "message": f"成果索引读取失败：{exc}"})

    def shutdown(self):
        self.runner.shutdown()

    def cancel(self):
        self.runner.cancel()

    def runtime_message(self):
        missing = self.runner.missing_runtime()
        return "缺少运行资源：\n" + "\n".join(missing) if missing else "运行资源已就绪"
