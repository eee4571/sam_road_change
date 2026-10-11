"""Real Qt <-> runtime round trip with synthetic data only; no model inference."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QLockFile
from PySide6.QtWidgets import QApplication
from plugin.controller import Controller
from plugin.project_state import ProjectState, write_json, read_json
from plugin.result_parser import formal_results
from plugin.ui.change_review import ChangeReviewDialog

app = QApplication.instance() or QApplication([])
controller = Controller()
service = controller.review
events, errors = [], []
published, progress = [], []
controller.result_ready.connect(published.append)
service.progress.connect(progress.append)
service.replied.connect(lambda op, data: events.append((op, data)))
service.failed.connect(lambda op, message: errors.append((op, message)))
service.stopped.connect(lambda message: errors.append(("stopped", message)))


def wait(op):
    deadline = time.monotonic() + 45
    while not errors and (not events or events[-1][0] != op) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not errors, errors
    assert events and events[-1][0] == op, (op, events)
    return events[-1][1]


def until(condition):
    deadline = time.monotonic() + 45
    while not errors and not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not errors, errors
    assert condition(), 'Timed out waiting for review dialog'


try:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "中文复核项目"
        script = """
import sys, shutil, copy
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from check_change_review import ReviewTests
from app.review_publication import read_json, write_json, remap
from app.result_publisher import copy_dataset
t = ReviewTests()
t.setUp()
manifest = read_json(t.manifest)
south = copy_dataset(t.source, t.output / '南区/02_变化检测/before_to_after/road_changes.shp')
entry = copy.deepcopy(manifest['change_results'][0])
entry.update(grid='南区', layers={'changes': str(south)}, published={'changes': str(south)})
manifest['change_results'].append(entry)
manifest['period_results'] += [{**p, 'grid': '南区'} for p in manifest['period_results']]
manifest['input_spec']['grids']['南区'] = copy.deepcopy(manifest['input_spec']['grids']['北区'])
write_json(t.manifest, manifest)
shutil.copytree(t.root, sys.argv[2])
p = Path(sys.argv[2]) / 'pipeline_result.json'
write_json(p, remap(read_json(p), t.root, Path(sys.argv[2])))
t.doCleanups()
"""
        subprocess.run([str(controller.runner.python_path()), "-c", script, str(ROOT / "tests"), str(root)],
                       check=True, timeout=45)
        store = ProjectState(root)
        write_json(store.path, {"status": "completed", "manifest": str(root / "pipeline_result.json")})
        service.start(store)
        assert len(wait("init")["pairs"]) == 2
        other = QLockFile(str(store.work / "current.lock"))
        assert not other.tryLock(0), "Review must hold the same lock as processing"
        service.request("open", index=0)
        assert len(wait("open")["features"]) == 3
        service.request("preview", id="1")
        assert wait("preview")["has_images"]
        service.request("save", decisions={"0": "keep", "1": "reject", "2": "pending"})
        assert Path(wait("save")["path"]).is_file()
        assert progress, 'Downstream progress must reach the UI'
        assert {'road_change', 'road_temporal', 'road_evaluation'} <= {r['result_type'] for r in published}
        assert len(read_json(store.results_path)) == len(formal_results(store))
        service.close()
        assert other.tryLock(0), "Closing must release the project lock"
        other.unlock()
        # A host closed between backend commit and snapshot refresh must still
        # discover the authoritative completed manifest on next open.
        write_json(store.results_path, [])
        assert formal_results(store)
        events.clear()
        service.start(store)
        wait('init')
        service.request('open', index=0)
        reopened = wait('open')
        assert [f['status'] for f in reopened['features']] == ['keep', 'reject', 'pending']
        assert Path(reopened['saved_path']).is_file()
        service.close()
        dialog = ChangeReviewDialog(service, store)
        try:
            dialog.show()
            dialog.start()
            until(lambda: dialog.pair_index == 0 and dialog.preview_valid)
            assert dialog.items.currentRow() == 2
            dialog.keep.click()
            until(lambda: dialog.pair_index == 1 and dialog.preview_valid)
            for index in range(3):
                assert dialog.items.currentRow() == index
                dialog.reject_button.click()
                if index < 2:
                    until(lambda: dialog.preview_valid and not dialog.waiting)
            assert dialog.dirty
            until(lambda: not dialog.waiting and not dialog.record_unsaved)
            # Close/reopen before application: judgements survive, products do not change.
            before_apply = read_json(root / 'pipeline_result.json')
            dialog.close()
            dialog.dispose()
            dialog.deleteLater()
            dialog = ChangeReviewDialog(service, store)
            dialog.show()
            dialog.start()
            until(lambda: dialog.pair_index == 0 and dialog.preview_valid and not dialog.waiting)
            assert dialog.dirty
            assert dialog.pair_states[1]['modified'], 'Unvisited persisted decisions must be included'
            assert read_json(root / 'pipeline_result.json') == before_apply
            dialog.save_button.click()
            until(lambda: not dialog.dirty and not dialog.waiting and dialog._save_queue is None)
            assert dialog.isVisible()
            until(lambda: dialog.preview_valid and not dialog.waiting)
            assert dialog.items.currentRow() >= 0
            dialog.close()
        finally:
            dialog.dispose()
            dialog.deleteLater()
        events.clear()
        service.start(store)
        wait('init')
        try:
            for index, expected in ((0, ['keep', 'reject', 'keep']), (1, ['reject'] * 3)):
                events.clear()
                service.request('open', index=index)
                actual = [f['status'] for f in wait('open')['features']]
                assert actual == expected, (index, actual, expected)
        finally:
            service.close()
        assert not any(name in sys.modules for name in ("geopandas", "rasterio", "torch")), "GIS leaked into host"
        print("PASS: preview, automatic region switching, batch save/reopen, result signals, Chinese paths, lock, host isolation")
finally:
    controller.shutdown()
