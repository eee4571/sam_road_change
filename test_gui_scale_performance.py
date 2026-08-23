from __future__ import annotations

import json
import inspect
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import user_workflow_gui as gui


class _Var:
    def __init__(self, value=""): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class _Button:
    def __init__(self): self.states = []
    def state(self, value): self.states.append(value)


class _Combo:
    def __init__(self): self.values = ()
    def configure(self, **kwargs): self.values = tuple(kwargs.get("values", self.values))


class _Tree:
    def __init__(self):
        self.rows = {}
        self.insert_calls = 0
        self._selection = ()
    def get_children(self): return tuple(self.rows)
    def delete(self, *items):
        for item in items: self.rows.pop(item, None)
    def insert(self, parent, _where, iid=None, text="", values=(), **_kwargs):
        iid = iid or f"row:{len(self.rows)}"
        self.rows[iid] = {"parent": parent, "text": text, "values": tuple(values)}
        self.insert_calls += 1
        return iid
    def selection(self): return self._selection
    def item(self, iid, option=None):
        row = self.rows.get(iid, {})
        return row.get(option, ()) if option else row


class _Log:
    def __init__(self): self.inserts = []; self.deletes = []; self.sees = 0
    def insert(self, _where, value): self.inserts.append(value)
    def delete(self, start, stop): self.deletes.append((start, stop))
    def see(self, _where): self.sees += 1


class _Root:
    def __init__(self): self.after_calls = []
    def after(self, delay, callback): self.after_calls.append((delay, callback))


class GuiScalePerformanceTests(unittest.TestCase):
    def test_scan_prunes_excluded_directories_and_collects_only_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for area_index in range(100):
                area = root / f"area_{area_index:03d}"
                area.mkdir()
                for period_index in range(20):
                    (area / f"{2000 + period_index}.txt").touch()
                (area / "validation.shp").touch()
                for unrelated in range(10):
                    (area / f"noise_{unrelated}.bin").touch()
            for excluded in (".git", "env", "04_成果输出", ".editor_cache"):
                directory = root / excluded
                directory.mkdir()
                (directory / "must_not_be_seen.txt").touch()
                (directory / "must_not_be_seen.shp").touch()
            started = time.perf_counter()
            result = gui.scan_external_data_source(root)
            elapsed = time.perf_counter() - started
            self.assertEqual(len(result["candidates"]["txt"]), 2000)
            self.assertEqual(len(result["candidates"]["shp"]), 100)
            self.assertFalse(any("must_not_be_seen" in path for paths in result["candidates"].values() for path in paths))
            cache = gui.scan_result_for_cache(result)
            cache_started = time.perf_counter()
            gui.atomic_write_json(root / "project_config.json", {"external_scan_cache": {str(root.resolve()): cache}})
            restored = json.loads((root / "project_config.json").read_text(encoding="utf-8"))
            cache_elapsed = time.perf_counter() - cache_started
            self.assertEqual(restored["external_scan_cache"][str(root.resolve())]["visited_files"], 3100)
            second = root / "_tmp"
            second.mkdir()
            for index in range(100): (second / f"new_{index}.txt").touch()
            second_started = time.perf_counter()
            second_result = gui.scan_external_data_source(second)
            second_elapsed = time.perf_counter() - second_started
            rescan_started = time.perf_counter()
            rescan = gui.scan_external_data_source(root)
            rescan_elapsed = time.perf_counter() - rescan_started
            self.assertEqual(len(second_result["candidates"]["txt"]), 100)
            self.assertEqual(len(rescan["candidates"]["txt"]), 2000)
            print(json.dumps({
                "first_scan_seconds": round(elapsed, 4),
                "cached_reopen_seconds": round(cache_elapsed, 6),
                "new_second_source_seconds": round(second_elapsed, 4),
                "explicit_rescan_seconds": round(rescan_elapsed, 4),
                "visited_files": result["visited_files"],
                "txt": len(result["candidates"]["txt"]),
                "shp": len(result["candidates"]["shp"]),
            }))

    def test_connecting_second_source_scans_only_the_new_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            old_source = root / "old"; new_source = root / "new"
            old_source.mkdir(); new_source.mkdir()
            app = gui.UserApp.__new__(gui.UserApp)
            app.root = object()
            app.project_root_path = str(root)
            app.project_data_sources = [str(old_source.resolve())]
            app.project_scan_cache = {str(old_source.resolve()): {"visited_files": 10}}
            app.data_source_display, app.data_status, app.status = _Var(), _Var(), _Var()
            app._save_project_config = mock.Mock(return_value=True)
            app.scan_data_sources = mock.Mock()
            with mock.patch.object(gui.filedialog, "askdirectory", return_value=str(new_source)):
                app.connect_data_source()
            app.scan_data_sources.assert_called_once_with(
                sources=[str(new_source.resolve())], force=True,
            )

    def test_log_poll_has_a_hard_budget_and_batches_tree_text_updates(self) -> None:
        app = gui.UserApp.__new__(gui.UserApp)
        app.queue, app.priority_queue = queue.Queue(), queue.Queue()
        for index in range(5000): app.queue.put(("log", f"line {index}"))
        app.recent_log_lines = []
        app._pending_log_insert_lines = []
        app._pending_log_delete_lines = 0
        app.log, app.shared_log_status, app.root = _Log(), _Var(), _Root()
        app._poll_geometry_editor = lambda: None
        app._refresh_progress_text = lambda: None
        app.data_status, app.project_scan_summary = _Var(), _Var()
        app.priority_queue.put(("scan_progress", {
            "source_index": 1, "source_total": 1, "directory": "D:/current",
            "visited_files": 123, "shp_count": 4, "txt_count": 5,
        }))
        app._poll()
        consumed = 5000 - app.queue.qsize()
        self.assertLessEqual(consumed, gui.MAX_QUEUE_EVENTS_PER_POLL)
        self.assertGreater(app.queue.qsize(), 0)
        self.assertEqual(len(app.log.inserts), 1)
        self.assertEqual(app.log.inserts[0].count("\n"), consumed)
        self.assertTrue(app.root.after_calls)
        self.assertIn("123", app.project_scan_summary.get())

    def _configuration_app(self):
        app = gui.UserApp.__new__(gui.UserApp)
        app.project_validation_areas = [(f"area_{index:03d}", f"C:/areas/{index}.shp") for index in range(100)]
        app.project_area_periods = {
            area: [(str(2000 + index), f"C:/{area}/{2000 + index}.txt") for index in range(20)]
            for area, _path in app.project_validation_areas
        }
        app.project_area_truths = [
            (area, str(2000 + index), str(2001 + index), f"C:/{area}/truth_{index}.shp")
            for area, _path in app.project_validation_areas for index in range(19)
        ]
        app.project_candidates = {"shp": [], "txt": []}
        app.project_txt_encodings = {}
        app.data_region, app.stage_region = _Var("area_000"), _Var("area_050")
        app.project_validation_path = _Var()
        app.project_region_combo = _Combo()
        app.project_config_container = object()
        app.project_period_tree, app.project_truth_tree, app.project_candidate_tree = _Tree(), _Tree(), _Tree()
        app.add_project_period_button = _Button()
        app._ensure_project_config_tables = lambda: None
        app._refresh_stage_selectors = mock.Mock()
        app._schedule_content_layout = lambda: None
        return app

    def test_100_area_configuration_refresh_keeps_only_current_region_rows(self) -> None:
        app = self._configuration_app()
        started = time.perf_counter()
        for area, _path in app.project_validation_areas:
            app.data_region.set(area)
            app._refresh_project_config_panel()
        elapsed = time.perf_counter() - started
        self.assertEqual(len(app.project_period_tree.rows), 20)
        self.assertEqual(len(app.project_truth_tree.rows), 19)
        self.assertEqual(app.stage_region.get(), "area_050")
        print(json.dumps({
            "configuration_regions": 100, "periods": 2000, "change_pairs": 1900,
            "switch_all_regions_seconds": round(elapsed, 4),
            "visible_period_rows": len(app.project_period_tree.rows),
        }))

    def test_result_tree_fingerprint_skips_unchanged_repopulation(self) -> None:
        app = gui.UserApp.__new__(gui.UserApp)
        app.result_tree, app.result_tree_paths = _Tree(), {}
        app._result_tree_fingerprint = None
        items = []
        for area_index in range(100):
            parent = f"area:{area_index}"
            items.append({"id": parent, "parent": "", "label": f"area_{area_index}", "status": "", "path": ""})
            items.extend({
                "id": f"{parent}:period:{period_index}", "parent": parent,
                "label": str(2000 + period_index), "status": "已生成", "path": "",
            } for period_index in range(20))
        started = time.perf_counter()
        app._populate_result_tree(items, None)
        first_elapsed = time.perf_counter() - started
        first_calls = app.result_tree.insert_calls
        app._populate_result_tree(items, None)
        self.assertEqual(app.result_tree.insert_calls, first_calls)
        self.assertEqual(first_calls, 2100)
        print(json.dumps({
            "result_tree_nodes": first_calls,
            "first_population_seconds": round(first_elapsed, 4),
            "unchanged_second_insertions": app.result_tree.insert_calls - first_calls,
        }))

    def test_temporal_pager_never_returns_more_than_page_size_for_100k_rows(self) -> None:
        frame = pd.DataFrame({
            "road_id": [f"road_{index}" for index in range(100_000)],
            "life_state": ["stable" if index % 2 else "review" for index in range(100_000)],
            "Y2021": list(range(100_000)),
        })
        started = time.perf_counter()
        pager = gui.TemporalAttributePager(frame, page_size=500)
        first = pager.page_frame()
        first_elapsed = time.perf_counter() - started
        self.assertEqual(len(first), 500)
        page_started = time.perf_counter()
        pager.set_page(1)
        self.assertEqual(len(pager.page_frame()), 500)
        page_elapsed = time.perf_counter() - page_started
        search_started = time.perf_counter()
        pager.set_query("road_99999")
        search_elapsed = time.perf_counter() - search_started
        self.assertEqual(pager.match_count, 1)
        self.assertLessEqual(len(pager.page_frame()), 500)
        print(json.dumps({
            "attribute_rows": len(frame), "page_size": pager.page_size,
            "first_page_seconds": round(first_elapsed, 6),
            "next_page_seconds": round(page_elapsed, 6),
            "search_100k_seconds": round(search_elapsed, 4),
        }))

    def test_temporal_search_ui_uses_300ms_debounce(self) -> None:
        source = inspect.getsource(gui.UserApp.open_temporal_attribute_table)
        self.assertIn("window.after(300, apply_search)", source)

    def test_scan_cache_record_is_json_compatible_and_cancel_discards_partial_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "one.txt").touch()
            complete = gui.scan_external_data_source(root)
            cached = gui.scan_result_for_cache(complete)
            json.dumps(cached, ensure_ascii=False)
            import threading
            event = threading.Event(); event.set()
            cancelled = gui.scan_external_data_source(root, cancel_event=event)
            self.assertTrue(cancelled["cancelled"])
            self.assertNotIn("candidates", cancelled)

    def test_temporal_reader_explicitly_disables_geometry(self) -> None:
        fake = mock.Mock()
        fake.read_dataframe.return_value = pd.DataFrame({"road_id": [1]})
        with mock.patch.dict("sys.modules", {"pyogrio": fake}):
            result = gui.read_temporal_attributes("roads.shp")
        fake.read_dataframe.assert_called_once_with(Path("roads.shp").resolve(), read_geometry=False)
        self.assertEqual(list(result.columns), ["road_id"])

    def test_temporal_item_uses_current_region_instead_of_first_result(self) -> None:
        app = gui.UserApp.__new__(gui.UserApp)
        app.temporal_items = [
            {"label": "area1", "grid": "area1", "path": "C:/area1/road_life.shp"},
            {"label": "area2", "grid": "area2", "path": "C:/area2/road_life.shp"},
        ]
        app.data_region, app.stage_region = _Var("area2"), _Var("area1")
        selected = app._choose_temporal_item()
        self.assertEqual(selected["grid"], "area2")


if __name__ == "__main__":
    unittest.main()
