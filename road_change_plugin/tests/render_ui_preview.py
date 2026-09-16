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
                for before, after in (("2020", "2022"), ("2022", "20250118")):
                    target = project / "成果输出" / area / (before + "_to_" + after + ".shp")
                    target.touch()
                    data["change_results"].append({"grid": area, "before_period": before, "after_period": after,
                                                   "published": {"changes": str(target)}})
            data["evaluation_summary"]["json"] = str(report)
            manifest = project / '_work/current/pipeline_result.json'
            manifest.parent.mkdir(parents=True)
            data['job_root'] = str(manifest.parent)
            manifest.write_text(json.dumps(data))
            (manifest.parent / 'job_state.json').write_text(json.dumps(data))
            plugin = original()
            plugin._controller.inspect_data = lambda data: {"periods": []}  # UI-only fixture
            create = plugin.create_widget

            def widget_factory(parent=None):
                widget = create(parent)
                widget._project_directory = str(project)
                widget._scanned(scan_project(project))
                def capture():
                    if widget.browsing:
                        QTimer.singleShot(20, capture)
                        return
                    widget.resize(680, 600)
                    def save(name):
                        QApplication.processEvents()
                        QApplication.processEvents()
                        assert widget.scroll.horizontalScrollBar().maximum() == 0
                        if widget.pages.currentWidget() is widget.main_page and widget.model.get('root'):
                            action = widget.cancel_button if widget.busy else widget.run_button
                            assert action.isVisible()
                            assert action.mapTo(widget, action.rect().bottomRight()).y() < widget.height()
                        widget.grab().save(str(ROOT / ("resources/" + name + ".png")))
                    widget._reset_results()
                    widget.result_summary.setText("暂无成果，运行完成后在此查看")
                    widget._update_controls()
                    save("ui_idle_680")
                    widget._started({"task_id": "preview-only"})
                    widget._started_at -= 98
                    widget._progress({"stage": "道路提取", "event": {"kind": "pipeline", "grid": "南区", "period": "2022", "progress": .68}})
                    widget._tick()
                    save("ui_running_680")
                    widget._started_at -= 48
                    widget._finished({"status": "completed", "task_id": "preview-only"})
                    save("ui_completed_680")
                    from plugin.project_state import ProjectState
                    current = ProjectState(project)
                    current.finish('cancelled')
                    widget._finished({'status': 'cancelled', 'task_id': 'preview-only'})
                    save('ui_resume_680')
                    from plugin.project_state import write_json
                    state = current.state()
                    state.update(action='rerun-period', scope={'grid': '北区', 'periods': ['2022'], 'changes': ['2020_to_2022', '2022_to_20250118']})
                    write_json(current.path, state)
                    widget._refresh_results()
                    save('ui_resume_update_680')
                    state.update(action='all', scope=None)
                    write_json(current.path, state)
                    current.finish('completed')
                    widget._finished({'status': 'completed', 'task_id': 'preview-only'})
                    widget._show_update()
                    next(b for b in widget.update_page.buttons.buttons() if b.selection.get('period') == '2022').click()
                    save('ui_local_rerun_680')
                    widget.resize(300, 600)
                    save('ui_update_300')
                    widget.resize(680, 600)
                    widget._show_results()
                    save('ui_results_680')
                    widget.resize(300, 600)
                    save('ui_results_300')
                    widget._back_to_main()
                    widget.resize(360, 600)
                    save("ui_narrow_360")
                    widget.resize(680, 600)
                    widget._show_configuration()
                    QApplication.processEvents()
                    widget.grab().save(str(ROOT / "resources/ui_configuration_680.png"))
                    widget.configuration.area.setCurrentText("北区")
                    widget.configuration.reference.setCurrentIndex(0)
                    widget.configuration.show_issues(["北区：请选择 IR-MAD 参考期"])
                    widget.resize(360, 600)
                    QApplication.processEvents()
                    QApplication.processEvents()
                    widget.grab().save(str(ROOT / "resources/ui_configuration_360.png"))
                    widget.resize(680, 600)
                    widget._show_creation()
                    widget.creation.project_name.setText("道路变化项目")
                    widget.creation.project_location.edit.setText(str(project.parent))
                    QApplication.processEvents()
                    QApplication.processEvents()
                    widget.grab().save(str(ROOT / "resources/ui_create_project_680.png"))
                    widget.resize(360, 600)
                    QApplication.processEvents()
                    QApplication.processEvents()
                    assert widget.creation.scroll.horizontalScrollBar().maximum() == 0
                    widget.grab().save(str(ROOT / "resources/ui_create_project_360.png"))
                    widget._load_project('')
                    widget.open_timer.stop()
                    widget.resize(680, 600)
                    save('ui_welcome_680')
                    print("Captured main, configuration and creation pages with simulated states")
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
