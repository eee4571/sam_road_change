"""Background file browsing without blocking the dock."""
from PySide6.QtCore import QObject, QRunnable, Signal


class BrowseSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class BrowseJob(QRunnable):
    def __init__(self, function, *args):
        super().__init__()
        self.signals = BrowseSignals()
        self.function, self.args = function, args

    def run(self):
        try:
            self.signals.completed.emit(self.function(*self.args))
        except Exception as exc:
            self.signals.failed.emit(str(exc))
