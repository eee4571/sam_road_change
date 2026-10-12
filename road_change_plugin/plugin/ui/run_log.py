"""Bounded, filtered diagnostics; raw logs stay available for support."""
import json
import re
from html import escape

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QTextEdit, QDialog, QPlainTextEdit, QApplication, QLabel)


from ..diagnostics import failure_summary, known_failure


def log_entry(payload):
    """Classify presentation only; never infer process state from console text."""
    message = str(payload.get('message', ''))
    if message.startswith('[Width warning]'):
        return 'WARNING', message
    level = str(payload.get('level', 'INFO')).upper()
    event = payload.get('event')
    if not event:
        try:
            candidate = json.loads(message)
            event = candidate if isinstance(candidate, dict) and candidate.get('kind') else None
        except (ValueError, TypeError):
            pass
    if event:
        if event.get('kind') == 'failure' or event.get('status') in {'failed', 'completed_with_errors'}:
            return 'ERROR', failure_summary(event.get('error') or event.get('message') or '任务未完成')
        if level in {'WARNING', 'WARN'} or event.get('kind') == 'warning':
            return 'WARNING', str(event.get('message') or '处理警告，请查看技术详情')
        if event.get('kind') in {'pipeline', 'stage', 'prepare', 'complete'}:
            state = {'running': '进行中', 'complete': '已完成', 'completed': '已完成', 'skipped': '已跳过'}.get(event.get('status'), '')
            scope = ' / '.join(str(event[k]) for k in ('grid', 'period', 'before_period', 'after_period') if event.get(k))
            return 'PROGRESS', ' · '.join(filter(None, (scope, str(event.get('stage') or '任务进度'), state, str(event.get('message') or ''))))
        return 'DEBUG', message
    if known_failure(message):
        return 'ERROR', known_failure(message)
    if level == 'ERROR' or re.search(r'(^|\s)(?:ERROR|RuntimeError|ValueError|FileNotFoundError|PermissionError)[:\s]', message):
        return 'ERROR', failure_summary(message)
    if level in {'WARNING', 'WARN'} or re.search(r'\b(?:WARNING|UserWarning|FutureWarning)\b', message):
        return 'WARNING', message
    if level == 'DEBUG' or re.search(r"(?:image_encoder\.|mask_decoder\.|prompt_encoder\.|state_dict|\[Fast timing\]|\[Plugin configuration\]|^\s*File \"|^Traceback|^\s*['\"][\w.]+(?:weight|bias)['\"])", message):
        return 'DEBUG', message
    # Plain informational text remains readable; tensor/parameter dumps are debug.
    if re.match(r'^\s*(?:tensor\(|OrderedDict\(|[\[\]{}(),]+\s*$)', message):
        return 'DEBUG', message
    return level if level in {'INFO', 'DEBUG'} else 'INFO', message


class RunLogPanel(QWidget):
    countsChanged = Signal(int, int)

    def __init__(self, open_folder, parent=None):
        super().__init__(parent)
        self.entries = []
        self.errors = self.warnings = 0
        self.error = ''
        self.width_reports = set()
        self.setMinimumSize(0, 100)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        tools = QHBoxLayout()
        self.filter = QComboBox()
        for title, value in (('关键日志', 'normal'), ('警告与错误', 'issues'), ('全部（含调试）', 'all')):
            self.filter.addItem(title, value)
        self.filter.setAccessibleName('日志级别')
        self.filter.currentIndexChanged.connect(self.render)
        tools.addWidget(self.filter, 1)
        copy = QPushButton('复制')
        copy.setToolTip('复制当前筛选显示的日志')
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        tools.addWidget(copy)
        folder = QPushButton('目录')
        folder.setToolTip('打开日志目录')
        folder.clicked.connect(open_folder)
        tools.addWidget(folder)
        layout.addLayout(tools)
        self.details = QPushButton('查看技术详情 / 复制诊断信息')
        self.details.clicked.connect(self.show_details)
        self.details.hide()
        layout.addWidget(self.details)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumSize(0, 40)
        self.text.document().setMaximumBlockCount(2000)
        layout.addWidget(self.text, 1)

    def set_error(self, text):
        self.error = text
        self.details.setVisible(bool(text))

    def reset(self, payloads=()):
        self.entries.clear()
        self.width_reports.clear()
        self.errors = self.warnings = 0
        for payload in payloads:
            self._remember(payload)
        self.render()
        self.countsChanged.emit(self.errors, self.warnings)

    def _remember(self, payload):
        event=payload.get('event') or {}
        if event.get('status')=='completed_with_warnings' and event.get('details'):
            self.width_reports.add(str(event['details']))
            self.error=event.get('message','测宽质量报告')
            self.details.show()
        level, text = log_entry(payload)
        if level == 'PROGRESS' and self.entries and self.entries[-1] == (level, text):
            return None
        self.entries.append((level, text))
        del self.entries[:-2000]
        self.errors += level == 'ERROR'
        self.warnings += level == 'WARNING'
        return level, text

    def append(self, payload):
        entry = self._remember(payload)
        if entry is None:
            return
        level, text = entry
        if self._visible(level):
            self._append(level, text)
        self.countsChanged.emit(self.errors, self.warnings)

    def _visible(self, level):
        selection = self.filter.currentData()
        return selection == 'all' or (level in {'WARNING', 'ERROR'} if selection == 'issues' else level != 'DEBUG')

    def _append(self, level, text):
        names = {'INFO': '信息', 'PROGRESS': '进度', 'WARNING': '警告', 'ERROR': '错误', 'DEBUG': '调试'}
        color = {'WARNING': '#b56b00', 'ERROR': '#c74343'}.get(level, self.palette().text().color().name())
        bar = self.text.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 2
        cursor = QTextCursor(self.text.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.text.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(f'<span style="color:{color}">[{names[level]}] {escape(text).replace(chr(10), "<br>")}</span>')
        cursor.block().setUserState({'ERROR': 2, 'WARNING': 1}.get(level, 0))
        if follow:
            bar.setValue(bar.maximum())

    def focus_issue(self):
        """Show the latest error, otherwise latest warning, within its context."""
        warning = None
        block = self.text.document().lastBlock()
        while block.isValid():
            if block.userState() == 2:
                break
            if block.userState() == 1 and warning is None:
                warning = block
            block = block.previous()
        target = block if block.isValid() else warning
        if target is not None:
            self.text.setTextCursor(QTextCursor(target))
            self.text.ensureCursorVisible()

    def render(self, *_):
        self.text.clear()
        for level, text in self.entries:
            if self._visible(level):
                self._append(level, text)

    def show_details(self):
        if not self.error:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('测宽质量与诊断信息' if self.width_reports else '错误诊断信息')
        dialog.resize(680, 420)
        box = QVBoxLayout(dialog)
        summary = QLabel(failure_summary(self.error))
        summary.setWordWrap(True)
        box.addWidget(summary)
        content=self.error
        from pathlib import Path
        for path in sorted(self.width_reports):
            try:content+='\n\n'+path+'\n'+Path(path).read_text(encoding='utf8')
            except OSError as exc:content+='\n'+path+'\n无法读取质量报告：'+str(exc)
        text = QPlainTextEdit(content)
        text.setReadOnly(True)
        box.addWidget(text, 1)
        copy = QPushButton('复制诊断信息')
        copy.clicked.connect(lambda: QApplication.clipboard().setText(content))
        box.addWidget(copy)
        dialog.exec()
