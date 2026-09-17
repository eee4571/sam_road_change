"""User intent and JSON data only; the backend stays in another interpreter."""
import json
import re
from pathlib import Path
from PySide6.QtCore import QLockFile
from .runner import ROOT, Runner
from .signals import TaskSignals
from .result_parser import read_results, formal_results
from .project_state import ProjectState
from .production_policy import task_arguments, configuration, check_existing_task


class Controller(TaskSignals):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runner = Runner(self)
        self.project_state = None
        self._project_lock = None
        self.runner.task_finished.connect(self._completed)
        self.runner.task_failed.connect(self._operation_failed)
        for name in ('task_started', 'task_progress', 'task_log', 'result_ready'):
            getattr(self.runner, name).connect(getattr(self, name).emit)
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
            if action == 'rerun-selection':
                store = ProjectState(data['project_root'], data['output'])
                scope = store.local_scope(action, data)
                check_existing_task(manifest)
                selection = {k: data.get(k, []) for k in ('selected_periods', 'selected_pairs')}
                return task_arguments([action, '--pipeline-manifest', manifest, '--grid', scope['grid'],
                                       '--selection', json.dumps(selection, ensure_ascii=False)])
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
            check_existing_task(manifest)
            args += ["--update-related" if action == "rerun-period" else "--update-temporal"]
            return task_arguments(args)
        areas, periods = data.get("areas", []), data.get("periods", [])
        if not areas:
            raise ValueError("请添加至少一个验证区 SHP")
        names = [r[0].strip() for r in areas]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("区域名称不能为空或重复")
        args = ["all", "--mode", "validation", "--execution-profile", "fast"]
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
                raise ValueError("内部运行标识无效，请重新运行完整流程")
            args += ["--run-id", run_id]
        if data.get("resume"):
            if not run_id:
                raise ValueError("未找到当前处理断点，请重新运行完整流程")
            args += ["--resume"]
        if data.get("truth_type_field", "").strip():
            args += ["--truth-type-field", data["truth_type_field"].strip()]
        references=data.get('area_irmad_references',{})
        for name in names:
            if references.get(name) not in [r[1] for r in periods if r[0]==name]:
                raise ValueError(f'{name} 请选择已有期次作为 IR-MAD 参考期')
        args += ['--irmad-reference',json.dumps(references,ensure_ascii=False)]
        return task_arguments(args)

    def run(self, action, data):
        if self.runner.running:
            self.task_log.emit({'level': 'WARNING', 'message': '当前处理尚未结束'})
            return
        prepared = False
        store = None
        try:
            data = dict(data)
            root = self.path(data.get("project_root") or self.path(data["output"]).parent)
            store = ProjectState(root, self.path(data["output"]))
            resuming = bool(data.get("resume"))
            if resuming:
                if not store.resumable():
                    raise ValueError("当前项目没有可继续的处理断点")
                state = store.state()
                action = state.get("action", "all")
                # Continue the saved processing inputs, even if the current configuration changed.
                data.update(state.get("parameters", {}))
                store.output = Path(state.get("output") or data["output"]).resolve()
                data["output"] = str(store.output)
                if action != "all":
                    data.update({k: v for k, v in state.get("parameters", {}).items()
                                 if k in {"grid", "period", "before_period", "after_period"}})
                data.update(manifest=str(store.manifest()), run_id=state.get("run_id", ""))
                check_existing_task(store.manifest())
            elif action != "all":
                if store.resumable():
                    raise ValueError("请先继续未完成处理，或重新运行完整流程")
                previous_output = store.state().get('output')
                if not previous_output and store.descriptor():
                    previous_output = store.descriptor()['data'].get('output_root')
                if previous_output:
                    store.output = Path(previous_output).resolve()
                    data['output'] = str(store.output)
                data["manifest"] = str(store.manifest())
            else:
                data.update(run_id="current", resume=False)
            # Validate all inputs and runtime before any destructive preparation.
            self.build_command(action, data)
            missing = self.runner.missing_runtime("fast")
            if missing:
                raise ValueError("缺少运行资源：" + "、".join(missing))
            configuration()
            store.work.mkdir(parents=True, exist_ok=True)
            lock = QLockFile(str(store.work / 'current.lock'))
            lock.setStaleLockTime(0)  # long processing is not a stale lock
            if not lock.tryLock(0):
                self.task_failed.emit({'plugin_id': 'road_change', 'task_id': '', 'status': 'failed',
                                       'message': '当前项目正在其他窗口处理中，请等待处理结束', 'detail': '项目运行锁'})
                return
            self._project_lock = lock
            self.project_state = store
            self.runner.project_root = root
            state = store.resume() if resuming else store.fresh(data) if action == "all" else store.local(action, data)
            prepared = True
            data.update(manifest=str(store.manifest()), run_id=state['run_id'], resume=resuming and action == "all")
            self.manifest = str(store.manifest())
            args = self.build_command(action, data)
            self.history.clear()
            self.runner.start(args, "fast")
        except (OSError, ValueError, TypeError) as exc:
            try:
                if prepared and store:
                    store.finish('failed', str(exc))
                elif store and store.root.is_dir():
                    store.note_error(str(exc))
            except (OSError, ValueError) as save_error:
                self.task_log.emit({'level': 'ERROR', 'message': f'运行状态保存失败：{save_error}'})
            self._unlock_project()
            self.task_failed.emit({"plugin_id": "road_change", "task_id": "", "message": str(exc), "detail": "项目运行准备", "status": "failed"})

    def _unlock_project(self):
        if self._project_lock:
            self._project_lock.unlock()
            self._project_lock = None

    def _operation_failed(self, payload):
        if self.project_state:
            try:
                self.project_state.finish('failed', '\n'.join(str(payload[k]) for k in ('message', 'detail') if payload.get(k)))
            except (OSError, ValueError) as exc:
                self.task_log.emit({"message": f"运行状态保存失败：{exc}", "level": "ERROR"})
        self._unlock_project()
        self.task_failed.emit(payload)

    def _completed(self, payload):
        if self.project_state:
            try:
                self.project_state.finish(payload.get('status', 'failed'))
                self.manifest = str(self.project_state.manifest())
                if payload.get('status') == 'completed':
                    for result in formal_results(self.project_state):
                        self.result_ready.emit({"plugin_id": "road_change", "task_id": payload.get('task_id', ''), **result})
            except (OSError, ValueError) as exc:
                self._operation_failed({**payload, "message": f"当前成果更新失败：{exc}", "detail": "项目成果发布", "status": "failed"})
                return
        self._unlock_project()
        self.task_finished.emit(payload)

    def load_results(self, manifest, task_id=""):
        try:
            self.manifest = str(self.path(manifest))
            for result in read_results(self.path(manifest)):
                self.result_ready.emit({"plugin_id": "road_change", "task_id": task_id, **result})
        except (OSError, ValueError, TypeError) as exc:
            self.task_log.emit({"task_id": task_id, "level": "ERROR", "message": f"成果索引读取失败：{exc}"})

    def shutdown(self):
        self.runner.shutdown()
        self._unlock_project()

    def cancel(self):
        self.runner.cancel()

    def runtime_message(self):
        missing = self.runner.missing_runtime()
        return "缺少运行资源：\n" + "\n".join(missing) if missing else "运行资源已就绪"

    def inspect_data(self, data):
        """Read-only GIS inspection in the independent backend interpreter."""
        import subprocess
        args=self.build_command('all',data)+['--data-check-only']
        executable,command=self.runner.command(args)
        environment=self.runner.environment()
        if data.get('project_root'):
            environment.insert('SAMROAD_PROJECT_ROOT', str(self.path(data['project_root'])))
        process=subprocess.run([executable,*command],cwd=self.runner.root/'code',
            env={k:environment.value(k) for k in environment.keys()},capture_output=True,
            text=True,encoding='utf8',errors='replace',timeout=180,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if process.returncode:
            raise ValueError((process.stderr or process.stdout)[-2500:])
        for line in process.stdout.splitlines():
            if line.startswith('__SAMROAD_USER__'):
                value=json.loads(line[len('__SAMROAD_USER__'):])
                if value.get('stage')=='data-check':return value
        raise ValueError('数据检查未返回报告')
