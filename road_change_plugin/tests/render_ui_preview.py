"""Render actual QWidgets with synthetic inputs and status; no backend execution."""
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ui_processing import APP, ROOT, fixture
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from plugin import create_plugin
from plugin.ui.project_browser import scan_project


def main():
    # This preview host supplies a CJK font, as a real host application would.
    # The plugin itself never sets application fonts or themes.
    font = Path(os.environ.get("SystemRoot", "")) / "Fonts/msyh.ttc"
    if font.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            APP.setFont(QFont(families[0], 9))
    preview = QWidget()
    layout = QVBoxLayout(preview)
    layout.addWidget(QLabel("实际 QWidget 渲染 · 示例项目 / 模拟运行状态 · 未运行模型"))
    row = QHBoxLayout()
    layout.addLayout(row)
    plugins = []
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "示例道路项目"
        project.mkdir()
        fixture(project)
        for index in range(3):
            plugin = create_plugin()
            plugins.append(plugin)
            widget = plugin.create_widget()
            widget.project.edit.setText(str(project))
            widget._scanned(scan_project(project))
            widget._checked([])
            widget.navigation.setCurrentIndex(index)
            if index == 1:
                widget._started({"task_id": "preview-only"})
                widget._started_at -= 146
                widget._tick()
                widget._progress({"stage": "道路变化检测", "event": {
                    "kind": "pipeline", "grid": "南区", "before_period": "2020", "after_period": "2022",
                    "progress": .4, "completed": 4, "total": 10}})
                widget.status.setText("运行中 · 示例状态")
            row.addWidget(widget)
        preview.resize(1180, 790)
        preview.show()
        APP.processEvents()
        target = ROOT / "resources/ui_preview.png"
        preview.grab().save(str(target))
        print(target)
        for plugin in plugins:
            plugin.shutdown()
        preview.close()


if __name__ == "__main__":
    main()
