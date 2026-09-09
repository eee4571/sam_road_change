"""Qt-only process boundary. Never import the backend into the host."""
import codecs
import json
import os
from pathlib import Path
import uuid

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from .signals import TaskSignals

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "__SAMROAD_USER__"


def parse_line(line):
    if not line.startswith(PREFIX):
        return None
    payload = json.loads(line[len(PREFIX):])
    if not isinstance(payload, dict):
        raise ValueError("结构化事件必须是 JSON 对象")
    return payload


class Runner(TaskSignals):
    def __init__(self, parent=None, root=None):
        super().__init__(parent)
        self.root = Path(root or ROOT).resolve()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.state = "idle"
        self.task_id = ""
        self._terminal = True
        self._cancelled = False
        self._completion = {}
        self._failure = ""
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self.process.kill)

    @property
    def running(self):
        return self.state in {"queued", "running"}

    def python_path(self):
        env = self.root / "runtime" / "env" / "samroad_env"
        return env / ("python.exe" if os.name == "nt" else "bin/python")

    def missing_runtime(self, profile="full"):
        paths = [self.python_path(), self.root / "runtime/config/samroad_inference.yaml",
                 self.root / "runtime/model/samroad/samroad.ckpt",
                 self.root / "runtime/model/samroad/sam_vit_b_01ec64.pth"]
        if profile != "fast":
            paths += [self.root / "runtime/model/sam_molra/sam_vit_b_01ec64.pth",
                      self.root / "runtime/model/sam_molra/adapter.th"]
        return [str(p.relative_to(self.root)) for p in paths if not p.is_file()]

    def command(self, args):
        return str(self.python_path()), ["-u", str(self.root / "code/user_pipeline.py"), *map(str, args)]

    def environment(self):
        env = QProcessEnvironment.systemEnvironment()
        env.remove("PYTHONHOME")
        env.insert("PYTHONPATH", str(self.root / "code"))
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("SAMROAD_MODELS_ROOT", str(self.root / "runtime/model"))
        runtime = self.python_path().parent
        entries = [runtime, runtime / "Library/bin", runtime / "Scripts"]
        env.insert("PATH", os.pathsep.join(str(p) for p in entries if p.is_dir()) + os.pathsep + env.value("PATH"))
        site = self.python_path().parent / "Lib/site-packages/rasterio"
        for key, sub in (("PROJ_DATA", "proj_data"), ("PROJ_LIB", "proj_data"), ("GDAL_DATA", "gdal_data")):
            env.remove(key)
            if (site / sub).is_dir():
                env.insert(key, str(site / sub))
        return env

    def start(self, args, profile="full"):
        if self.running:
            raise ValueError("当前任务尚未结束")
        self.task_id = uuid.uuid4().hex
        self._terminal = False
        self._cancelled = False
        self._completion = {}
        self._failure = ""
        self._buffer = ""
        self._decoder.reset()
        self.state = "queued"
        missing = self.missing_runtime(profile)
        if missing:
            self._fail("缺少运行资源，请手动复制：\n" + "\n".join(missing))
            return
        program, arguments = self.command(args)
        self.process.setWorkingDirectory(str(self.root / "code"))
        self.process.setProcessEnvironment(self.environment())
        self.process.start(program, arguments)

    def _started(self):
        self.state = "running"
        self.task_started.emit(self._payload(name="道路变化检测"))

    def _payload(self, **values):
        return {"task_id": self.task_id, "plugin_id": "road_change", **values}

    def _read(self):
        self._buffer += self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.consume_line(line.rstrip("\r"))

    def consume_line(self, line):
        try:
            event = parse_line(line)
        except (ValueError, TypeError) as exc:
            self._failure = "后端事件协议错误：" + str(exc)
            self.task_log.emit(self._payload(level="ERROR", message=self._failure))
            return
        if event is None:
            self.task_log.emit(self._payload(level="INFO", message=line))
            return
        kind = event.get("kind")
        self.task_log.emit(self._payload(level="INFO", message=json.dumps(event, ensure_ascii=False), event=event))
        if kind in {"pipeline", "stage", "prepare"}:
            done = event.get("completed", event.get("stage_index", event.get("index", 0)))
            total = event.get("total", event.get("stage_total", 0))
            try:
                progress = max(0.0, min(1.0, float(done) / float(total))) if float(total) > 0 else 0.0
            except (TypeError, ValueError):
                progress = 0.0
            self.task_progress.emit(self._payload(progress=progress, stage=str(event.get("stage", "")),
                                                  message=str(event.get("message", "")), event=event))
        if kind == "failure" or event.get("status") in {"failed", "completed_with_errors"}:
            self._failure = str(event.get("error") or event.get("message") or "后端报告处理失败")
        if kind == "complete":
            self._completion = event
            if event.get("failure_count", 0):
                self._failure = f"任务结束，其中 {event['failure_count']} 项失败，请查看运行记录"

    def _fail(self, message):
        if self._terminal:
            return
        self._terminal = True
        self.state = "failed"
        self.task_failed.emit(self._payload(message=message, detail=self.process.errorString(), status="failed"))

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._fail("无法启动插件环境：" + self.process.errorString())

    def _finished(self, exit_code, exit_status):
        self._kill_timer.stop()
        self._read()
        self._buffer += self._decoder.decode(b"", final=True)
        if self._buffer:
            self.consume_line(self._buffer.rstrip("\r"))
            self._buffer = ""
        if self._terminal:
            return
        if self._cancelled:
            self.state = "cancelled"
        elif exit_code or exit_status != QProcess.ExitStatus.NormalExit or self._failure:
            self._fail(self._failure or f"后端退出，代码 {exit_code}")
            return
        else:
            self.state = "completed"
        self._terminal = True
        self.task_finished.emit(self._payload(status=self.state, exit_code=exit_code, event=self._completion))

    def cancel(self):
        if not self.running:
            return
        self._cancelled = True
        # Backend launches model children; terminate the owned process tree on Windows.
        if os.name == "nt" and self.process.processId():
            killer = QProcess(self)
            killer.start("taskkill", ["/PID", str(self.process.processId()), "/T", "/F"])
            killer.waitForFinished(3000)
            killer.deleteLater()
        else:
            self.process.terminate()
        self._kill_timer.start(2000)

    def shutdown(self):
        if self.running:
            self.cancel()
            if not self.process.waitForFinished(3500):
                self.process.kill()
                self.process.waitForFinished(1500)
