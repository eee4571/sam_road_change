from PySide6.QtCore import QObject, Signal


class TaskSignals(QObject):
    task_started = Signal(dict)
    task_progress = Signal(dict)
    task_log = Signal(dict)
    task_finished = Signal(dict)
    task_failed = Signal(dict)
    result_ready = Signal(dict)


SIGNAL_NAMES = ("task_started", "task_progress", "task_log", "task_finished",
                "task_failed", "result_ready")


def forward(source, target):
    for name in SIGNAL_NAMES:
        getattr(source, name).connect(getattr(target, name).emit)
