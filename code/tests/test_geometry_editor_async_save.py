from __future__ import annotations

import inspect
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

WIDTH = Path(__file__).resolve().parents[1] / "engine" / "width"
if str(WIDTH) not in sys.path:
    sys.path.insert(0, str(WIDTH))

import geometry_editor as editor  # noqa: E402


class _Var:
    def __init__(self, value=""):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class _Button:
    def __init__(self):
        self.options = {"text": "保存编辑", "state": "normal"}
    def configure(self, **kwargs):
        self.options.update(kwargs)


class _Root:
    def __init__(self):
        self.callbacks = []
        self.destroyed = False
    def after(self, delay, callback):
        self.callbacks.append((time.monotonic() + delay / 1000.0, callback))
        return f"after-{len(self.callbacks)}"
    def destroy(self):
        self.destroyed = True
    def pump(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not predicate():
            now = time.monotonic()
            ready = [item for item in self.callbacks if item[0] <= now]
            self.callbacks = [item for item in self.callbacks if item[0] > now]
            for _due, callback in ready:
                callback()
            time.sleep(0.005)
        return predicate()


class AsyncSaveTests(unittest.TestCase):
    def _document(self, shape=(64, 64)):
        nodes = np.asarray([[20, 5], [20, 30], [20, 55]], dtype=np.float32)
        edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
        return editor.GeometryDocument(
            "tile", np.zeros((*shape, 3), dtype=np.uint8), nodes, edges,
            np.zeros(shape, dtype=np.uint8),
        )

    def _app(self, edited: Path, document=None):
        app = editor.GeometryEditorApp.__new__(editor.GeometryEditorApp)
        app.root = _Root()
        app.edited_dir = edited
        app.review_dir = edited.parent / "review"
        app.documents = [document or self._document()]
        app.document_index = 0
        app.final_centerlines = None
        app.final_surfaces = None
        app.saved_var = _Var("尚未保存")
        app.status_var = _Var("")
        app.save_button = _Button()
        app._save_queue = None
        app._save_thread = None
        app._active_save_snapshot = None
        app._save_in_progress = False
        app._save_requested_again = False
        app._save_show_message = False
        app._close_after_save = False
        app.last_save_result = None
        app.last_save_submit_seconds = 0.0
        return app

    def _pump_save(self, app, timeout=3.0):
        self.assertTrue(app.root.pump(lambda: not app._save_in_progress, timeout))

    def test_worker_has_no_tk_access(self):
        source = inspect.getsource(editor.save_snapshot_worker)
        for forbidden in ("tk.", "ttk.", "StringVar", "messagebox", "root.after", "canvas"):
            self.assertNotIn(forbidden, source)

    def test_save_all_returns_before_worker_io_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("geometry")
            release = threading.Event()
            def blocked(messages, snapshot):
                release.wait(1.0)
                messages.put(("saved", editor.SaveResult({}, [{
                    "document_index": 0,
                    "dirty_revisions": snapshot.documents[0].dirty_revisions,
                    "saved_nodes": snapshot.documents[0].nodes,
                    "saved_edges": snapshot.documents[0].edges,
                }], {"total": 1.0})))
            with mock.patch("geometry_editor.save_snapshot_worker", side_effect=blocked):
                started = time.perf_counter()
                self.assertTrue(app.save_all(show_message=False))
                returned = time.perf_counter() - started
                self.assertLess(returned, 0.1)
                self.assertTrue(app._save_in_progress)
                release.set()
                self._pump_save(app)

    def test_no_dirty_does_not_start_worker_or_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            with mock.patch("geometry_editor.threading.Thread") as thread, mock.patch(
                "geometry_editor.save_graph",
            ) as graph, mock.patch("geometry_editor.cv2.imencode") as encode:
                self.assertFalse(app.save_all(show_message=False))
            thread.assert_not_called()
            graph.assert_not_called()
            encode.assert_not_called()
            self.assertEqual(app.saved_var.get(), "✓ 已保存")

    def test_geometry_only_does_not_snapshot_or_write_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("geometry")
            snapshot = app._build_save_snapshot()
            self.assertIsNone(snapshot.documents[0].surface_additions)
            with mock.patch(
                "geometry_editor._atomic_surface_png_write",
                side_effect=AssertionError("geometry save wrote surface"),
            ):
                editor.persist_save_snapshot(snapshot)

    def test_width_only_does_not_snapshot_or_write_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("widths")
            snapshot = app._build_save_snapshot()
            self.assertIsNone(snapshot.documents[0].surface_additions)
            with mock.patch(
                "geometry_editor._atomic_surface_png_write",
                side_effect=AssertionError("width save wrote surface"),
            ), mock.patch(
                "geometry_editor._atomic_graph_write",
                side_effect=AssertionError("width save wrote graph"),
            ):
                editor.persist_save_snapshot(snapshot)

    def test_surface_only_does_not_snapshot_or_write_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("surface")
            snapshot = app._build_save_snapshot()
            self.assertIsNone(snapshot.documents[0].nodes)
            with mock.patch(
                "geometry_editor._atomic_graph_write",
                side_effect=AssertionError("surface save wrote graph"),
            ):
                editor.persist_save_snapshot(snapshot)

    def test_success_clears_matching_dirty_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("widths")
            self.assertTrue(app.save_all(show_message=False))
            self._pump_save(app)
            self.assertFalse(app.doc.dirty_widths)
            self.assertEqual(app.saved_var.get(), "✓ 已保存")

    def test_edit_during_save_keeps_new_revision_dirty(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("widths")
            release = threading.Event()
            original = editor.save_snapshot_worker
            def delayed(messages, snapshot):
                release.wait(1.0)
                original(messages, snapshot)
            with mock.patch("geometry_editor.save_snapshot_worker", side_effect=delayed):
                app.save_all(show_message=False)
                saved_revision = app.doc.edit_revision
                app.doc.checkpoint("widths")
                self.assertGreater(app.doc.edit_revision, saved_revision)
                release.set()
                self._pump_save(app)
            self.assertTrue(app.doc.dirty_widths)
            self.assertIn("还有未保存修改", app.saved_var.get())

    def test_repeated_save_uses_one_worker_and_then_saves_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("geometry")
            release = threading.Event()
            calls = []
            original = editor.save_snapshot_worker
            def delayed(messages, snapshot):
                calls.append(snapshot)
                if len(calls) == 1:
                    release.wait(1.0)
                original(messages, snapshot)
            with mock.patch("geometry_editor.save_snapshot_worker", side_effect=delayed):
                app.save_all(show_message=False)
                first_thread = app._save_thread
                app.doc.checkpoint("geometry")
                self.assertFalse(app.save_all(show_message=False))
                self.assertIs(app._save_thread, first_thread)
                release.set()
                self._pump_save(app, timeout=5.0)
            self.assertEqual(len(calls), 2)
            self.assertFalse(app.doc.dirty_geometry)

    def test_failure_keeps_dirty_and_window_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("surface")
            def failed(messages, _snapshot):
                messages.put(("error", OSError("disk full"), "traceback detail"))
            with mock.patch("geometry_editor.save_snapshot_worker", side_effect=failed), mock.patch(
                "geometry_editor.messagebox.showerror",
            ):
                app.save_all(show_message=False, close_after=True)
                self._pump_save(app)
            self.assertTrue(app.doc.dirty_surface)
            self.assertFalse(app.root.destroyed)

    def test_save_and_close_destroys_only_after_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("widths")
            release = threading.Event()
            original = editor.save_snapshot_worker
            def delayed(messages, snapshot):
                release.wait(1.0)
                original(messages, snapshot)
            with mock.patch("geometry_editor.save_snapshot_worker", side_effect=delayed):
                app.save_all(show_message=False, close_after=True)
                self.assertFalse(app.root.destroyed)
                release.set()
                self._pump_save(app)
            self.assertTrue(app.root.destroyed)

    def test_close_during_active_save_waits_for_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("geometry")
            release = threading.Event()
            original = editor.save_snapshot_worker
            def delayed(messages, snapshot):
                release.wait(1.0)
                original(messages, snapshot)
            with mock.patch("geometry_editor.save_snapshot_worker", side_effect=delayed), mock.patch(
                "geometry_editor.messagebox.showinfo",
            ):
                app.save_all(show_message=False)
                app.close()
                self.assertFalse(app.root.destroyed)
                release.set()
                self._pump_save(app)
            self.assertTrue(app.root.destroyed)

    def test_undo_redo_increment_revision_and_keep_kind_dirty(self):
        document = self._document()
        document.checkpoint("widths")
        document.manual_widths.append({"measurement_id": "one"})
        first_revision = document.edit_revision
        self.assertTrue(document.undo())
        self.assertGreater(document.edit_revision, first_revision)
        self.assertTrue(document.dirty_widths)
        undo_revision = document.edit_revision
        self.assertTrue(document.redo())
        self.assertGreater(document.edit_revision, undo_revision)
        self.assertTrue(document.dirty_widths)

    def test_save_does_not_reload_mosaic_or_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("geometry")
            snapshot = app._build_save_snapshot()
            with mock.patch(
                "geometry_editor._build_global_overview",
                side_effect=AssertionError("save reloaded mosaic"),
            ), mock.patch(
                "geometry_editor.read_background_cache",
                side_effect=AssertionError("save reloaded cache"),
            ):
                editor.persist_save_snapshot(snapshot)

    def test_heartbeat_runs_while_worker_is_busy(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary) / "edited")
            app.doc.checkpoint("geometry")
            release = threading.Event()
            heartbeats = []
            def heartbeat():
                heartbeats.append(time.monotonic())
                if app._save_in_progress:
                    app.root.after(20, heartbeat)
            app.root.after(0, heartbeat)
            original = editor.save_snapshot_worker
            def delayed(messages, snapshot):
                release.wait(0.25)
                original(messages, snapshot)
            with mock.patch("geometry_editor.save_snapshot_worker", side_effect=delayed):
                app.save_all(show_message=False)
                app.root.after(250, release.set)
                self._pump_save(app, timeout=3.0)
            self.assertGreaterEqual(len(heartbeats), 5)


if __name__ == "__main__":
    unittest.main()
