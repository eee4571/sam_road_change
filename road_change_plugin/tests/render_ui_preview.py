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
            # Distinct placeholder files make the displayed scope counts real.
            for entry in data["period_results"]:
                folder = project / "成果输出" / entry["grid"] / entry["period"]
                folder.mkdir(parents=True, exist_ok=True)
                for kind in entry["published"]:
                    target = folder / (kind + ".shp")
                    target.touch()
                    entry["published"][kind] = str(target)
            data["change_results"] = []
            for area in ("北区", "南区"):
                for before, after in (("2020", "2022"), ("2022", "2024")):
                    target = project / "成果输出" / area / (before + "_to_" + after + ".shp")
                    target.touch()
                    data["change_results"].append({"grid": area, "before_period": before, "after_period": after,
                                                   "published": {"changes": str(target)}})
            data["evaluation_summary"]["json"] = str(report)
            manifest.write_text(json.dumps(data))
            plugin = original()
            create = plugin.create_widget

            def widget_factory(parent=None):
                widget = create(parent)
                widget.project.edit.setText(str(project))
                widget._scanned(scan_project(project))
                widget._checked([])
                def capture():
                    widget.resize(680, 600)
                    def save(name):
                        QApplication.processEvents()
                        QApplication.processEvents()
                        assert widget.scroll.horizontalScrollBar().maximum() == 0
                        assert widget.run_button.isVisible()
                        assert widget.run_button.mapTo(widget, widget.run_button.rect().bottomRight()).y() < widget.height()
                        widget.grab().save(str(ROOT / ("resources/" + name + ".png")))
                    widget._reset_results()
                    widget.result_summary.setText("暂无成果，运行完成后在此查看")
                    save("ui_idle_680")
                    widget._started({"task_id": "preview-only"})
                    widget._started_at -= 98
                    widget._progress({"stage": "道路提取", "event": {"kind": "pipeline", "grid": "南区", "period": "2022", "progress": .68}})
                    widget._tick()
                    save("ui_running_680")
                    widget._started_at -= 48
                    widget._finished({"status": "completed", "task_id": "preview-only"})
                    save("ui_completed_680")
                    widget.resize(360, 600)
                    save("ui_narrow_360")
                    print("Captured idle, running, completed (680 × 600), narrow (360 × 600)")
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
