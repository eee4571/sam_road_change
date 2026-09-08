"""Run the unchanged standalone entry with synthetic UI-only preview inputs."""
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
import standalone


def main():
    original = standalone.create_plugin
    with tempfile.TemporaryDirectory() as tmp:
        def factory():
            # Preview-only host font. The plugin still inherits the host's style.
            font = Path(os.environ.get("SystemRoot", "")) / "Fonts/msyh.ttc"
            if font.is_file():
                ident = QFontDatabase.addApplicationFont(str(font))
                families = QFontDatabase.applicationFontFamilies(ident)
                if families:
                    QApplication.instance().setFont(QFont(families[0], 9))
            from test_ui_processing import fixture
            from plugin.ui.project_browser import scan_project
            project = Path(tmp) / "示例道路项目"
            project.mkdir()
            fixture(project)
            report = project / "成果输出/evaluation.json"
            report.write_text(json.dumps({"metrics": [{"class": "all", "precision": .912, "recall": .876, "f1": .894}]}))
            manifest = project / "_work/tasks/latest_pipeline.json"
            data = json.loads(manifest.read_text())
            data["evaluation_summary"]["json"] = str(report)
            manifest.write_text(json.dumps(data))
            plugin = original()
            create = plugin.create_widget

            def widget_factory(parent=None):
                widget = create(parent)
                widget.project.edit.setText(str(project))
                widget._scanned(scan_project(project))
                widget._checked([])
                widget._started({"task_id": "preview-only"})
                widget._started_at -= 146
                widget._finished({"status": "completed", "task_id": "preview-only"})
                widget.area_text.setText("南区")
                widget.scope_text.setText("2022 → 2024")
                widget.stage_text.setText("正式成果已生成")
                widget.status.setText("示例数据与模拟完成状态 · 未运行模型")

                def capture():
                    widget.setWindowTitle("道路变化检测 · 单页 standalone 预览")
                    widget.resize(400, 1020)
                    QApplication.processEvents()
                    target = ROOT / "resources/ui_preview.png"
                    widget.grab().save(str(target))
                    widget.resize(300, 760)
                    QApplication.processEvents()
                    assert widget.scroll.horizontalScrollBar().maximum() == 0
                    widget.grab().save(str(ROOT / "resources/ui_preview_300.png"))
                    print(target)
                    QApplication.instance().quit()

                QTimer.singleShot(100, capture)
                return widget

            plugin.create_widget = widget_factory
            return plugin

        standalone.create_plugin = factory
        try:
            return standalone.main()
        finally:
            standalone.create_plugin = original


if __name__ == "__main__":
    raise SystemExit(main())
