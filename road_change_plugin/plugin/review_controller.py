"""Application boundary for the manual review worker and project lock."""
import codecs
import json

from PySide6.QtCore import QObject, Signal, QProcess, QLockFile

from .runner import ROOT


class ReviewController(QObject):
    replied = Signal(str, object)
    failed = Signal(str, str)
    stopped = Signal(str)
    progress = Signal(str)
    updated = Signal(object)

    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.readyReadStandardError.connect(self._read_error)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._lock = None
        self.pending = ""
        self._closing = False

    @property
    def active(self):
        return self._lock is not None

    def start(self, store):
        if self.active or self.runner.running:
            raise ValueError("当前任务尚未结束，请稍后打开人工复核")
        if store.resumable():
            raise ValueError("请先完成未结束的处理，再复核变化成果")
        if not self.runner.python_path().is_file():
            raise ValueError("缺少插件数据读取环境 runtime/env/samroad_env")
        if not store.manifest().is_file():
            raise ValueError("没有可复核的变化成果")
        store.work.mkdir(parents=True, exist_ok=True)
        lock = QLockFile(str(store.work / "current.lock"))
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            raise ValueError("当前项目正在其他窗口处理或复核，请等待结束")
        self._lock = lock
        self._closing = False
        self._buffer = self._errors = self.pending = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._manifest = str(store.manifest())
        self._store = store
        env = self.runner.environment()
        env.insert("SAMROAD_PROJECT_ROOT", str(store.root))
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(ROOT / "code"))
        self.process.start(str(self.runner.python_path()), ["-u", "-m", "app.change_review"])

    def _started(self):
        self.request("init", manifest=self._manifest, project_root=str(self._store.root), output_root=str(self._store.output))

    def request(self, op, **data):
        if self.pending:
            raise ValueError("请等待当前复核操作完成")
        if not self.active or self.process.state() != QProcess.ProcessState.Running:
            raise ValueError("复核数据服务尚未启动或已经退出")
        self.pending = op
        self.process.write((json.dumps({"op": op, **data}, ensure_ascii=False) + "\n").encode("utf-8"))

    def _read(self):
        self._buffer += self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line.startswith("__ROAD_REVIEW__"):
                continue
            op = self.pending
            try:
                reply = json.loads(line[len("__ROAD_REVIEW__"):])
                if reply.get("op") != op:
                    raise ValueError("复核服务返回了不匹配的操作")
                if 'progress' in reply:
                    self.progress.emit(reply['progress'])
                    continue
                self.pending = ""
                if reply["ok"]:
                    if op == 'save':
                        from .project_state import write_json
                        from .result_parser import read_results
                        results = read_results(self._store.manifest())
                        write_json(self._store.results_path, results)
                        self.updated.emit(results)
                    self.replied.emit(op, reply["data"])
                else:
                    self.failed.emit(op, reply["error"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.pending = ""
                self.failed.emit(op, f"复核数据读取失败：{exc}")

    def _read_error(self):
        self._errors = (self._errors + bytes(self.process.readAllStandardError()).decode("utf-8", "replace"))[-6000:]

    def _release(self):
        if self._lock:
            self._lock.unlock()
            self._lock = None

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._release()
            self.pending = ""
            self.stopped.emit("无法启动复核服务：" + self.process.errorString())

    def _finished(self, *_):
        self._read_error()
        self._release()
        self.pending = ""
        if not self._closing:
            self.stopped.emit("复核服务已退出，请关闭并重新打开。\n" + self._errors)

    def close(self):
        self._closing = True
        if self.process.state() != QProcess.ProcessState.NotRunning:
            # EOF lets an in-flight save finish its atomic publication first.
            self.process.closeWriteChannel()
            if not self.process.waitForFinished(3000):
                self.process.kill()
                self.process.waitForFinished(1500)
        self.pending = ""
        self._release()
