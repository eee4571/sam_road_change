from __future__ import annotations

import inspect
import queue
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ENGINE = Path(__file__).resolve().parent / "engine" / "width"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from geometry_editor import (  # noqa: E402
    GeometryDocument,
    GeometryEditorApp,
    create_normal_width_measurement,
    load_documents_worker,
    main,
    point_segment_projection,
)


class _Var:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class _Event:
    x = 100
    y = 80
    delta = 120
    state = 0
    widget = None

    def __init__(self, rc=(45.0, 50.0)): self.rc = rc


class _Canvas:
    def __init__(self):
        self.coords_calls = []
        self.xview_value = (0.0, 1.0)
        self.yview_value = (0.0, 1.0)
    def canvasx(self, value): return value
    def canvasy(self, value): return value
    def winfo_width(self): return 400
    def winfo_height(self): return 300
    def xview(self, *args): return self.xview_value
    def yview(self, *args): return self.yview_value
    def scale(self, *args): pass
    def configure(self, **kwargs): pass
    def xview_moveto(self, value): pass
    def yview_moveto(self, value): pass
    def coords(self, *args): self.coords_calls.append(args)


class _Root:
    def __init__(self):
        self.after_calls = []
        self.cancelled = []

    def after(self, delay, callback):
        after_id = f"after-{len(self.after_calls) + 1}"
        self.after_calls.append((after_id, delay, callback))
        return after_id

    def after_cancel(self, after_id):
        self.cancelled.append(after_id)


class _Scrollbar:
    def __init__(self): self.values = []
    def set(self, first, last): self.values.append((first, last))


class GeometryEditorPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = np.asarray([[50, col] for col in range(0, 101, 10)], dtype=np.float32)
        self.edges = np.asarray([[index, index + 1] for index in range(10)], dtype=np.int32)
        self.doc = GeometryDocument(
            "cache", np.zeros((120, 120, 3), dtype=np.uint8), self.nodes, self.edges,
            np.zeros((120, 120), dtype=np.uint8),
        )

    def test_road_chain_cache_is_reused_and_invalidated_by_topology_change(self) -> None:
        first = self.doc.road_chains()
        second = self.doc.road_chains()
        self.assertIs(first, second)
        self.assertEqual(self.doc.cache_build_counts["chains"], 1)
        self.doc.add_polyline([(50, 100), (70, 110)], snap_tolerance=1.0)
        self.doc.road_chains()
        self.assertEqual(self.doc.cache_build_counts["chains"], 2)

    def test_loading_shell_is_built_before_worker_thread_starts(self) -> None:
        source = inspect.getsource(main)
        self.assertLess(source.index("build_loading_shell(root)"), source.index("threading.Thread("))
        self.assertLess(source.index("root.update()"), source.index("threading.Thread("))

    def test_loading_worker_has_no_tk_widget_access_and_completes(self) -> None:
        source = inspect.getsource(load_documents_worker)
        self.assertNotIn("tk.", source)
        self.assertNotIn("ttk.", source)
        messages = queue.Queue()

        def fake_load(*_args, progress=None, timings=None, **_kwargs):
            progress("正在准备遥感影像…")
            timings["fake"] = 1.0
            return [self.doc]

        timings = {}
        with mock.patch("geometry_editor.load_documents", side_effect=fake_load):
            load_documents_worker(
                messages, Path("review"), Path("edited"), None, None, timings,
            )
        self.assertEqual(messages.get_nowait(), ("status", "正在准备遥感影像…"))
        kind, documents = messages.get_nowait()
        self.assertEqual(kind, "loaded")
        self.assertIs(documents[0], self.doc)
        self.assertEqual(timings["fake"], 1.0)

    def test_startup_metrics_are_lazy(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.doc], 0
        app.last_metrics = {}
        app.topology_dirty = True
        app.update_status = lambda: None
        with mock.patch.object(self.doc, "topology_metrics", wraps=self.doc.topology_metrics) as metrics:
            app.refresh_metrics()
        self.assertEqual(metrics.call_count, 0)
        self.assertEqual(app.last_metrics["node_count"], len(self.doc.nodes))
        self.assertEqual(app.last_metrics["edge_count"], len(self.doc.edges))

    def test_global_loader_can_defer_legacy_width_normalization_until_save(self) -> None:
        legacy = [{
            "measurement_id": "MW00001", "target_row": 50, "target_col": 50,
            "start_row": 44, "start_col": 50, "end_row": 56, "end_col": 50,
            "width_px": 12, "source": "manual_boundary_measurement",
        }]
        with mock.patch(
            "geometry_editor.normalize_manual_width_measurements",
            wraps=__import__("geometry_editor").normalize_manual_width_measurements,
        ) as normalize:
            document = GeometryDocument(
                "lazy", self.doc.image, self.nodes, self.edges, self.doc.mask,
                manual_widths=legacy, defer_manual_width_normalization=True,
            )
            self.assertEqual(normalize.call_count, 0)
            self.assertNotIn("target_chain_id", document.manual_widths[0])
            document.compact()
            self.assertEqual(normalize.call_count, 1)
            self.assertIn("target_chain_id", document.manual_widths[0])

    def test_worker_owned_binary_arrays_are_adopted_without_copy(self) -> None:
        mask = np.zeros((120, 120), dtype=np.uint8)
        additions = np.zeros_like(mask)
        removals = np.zeros_like(mask)
        document = GeometryDocument(
            "adopt", self.doc.image, self.nodes, self.edges, mask,
            surface_additions=additions, surface_removals=removals,
            adopt_large_arrays=True,
        )
        self.assertIs(document.mask, mask)
        self.assertIs(document.surface_additions, additions)
        self.assertIs(document.surface_removals, removals)

    def test_spatial_index_nearest_edge_matches_full_scan(self) -> None:
        queries = [(50, 5), (47, 23), (54, 88), (80, 50)]
        for row, col in queries:
            indexed = self.doc.nearest_edge(row, col, 40.0)
            brute = None
            point = np.asarray([row, col], dtype=np.float32)
            for edge_id, (src, dst) in enumerate(self.edges.tolist()):
                projection, distance = point_segment_projection(point, self.nodes[src], self.nodes[dst])
                if distance <= 40.0 and (brute is None or distance < brute[2]):
                    brute = (edge_id, projection, distance)
            self.assertEqual(None if indexed is None else indexed[0], None if brute is None else brute[0])
            if indexed is not None:
                np.testing.assert_allclose(indexed[1], brute[1])
                self.assertAlmostEqual(indexed[2], brute[2])

    def test_many_width_previews_build_road_chains_once(self) -> None:
        for offset in range(20):
            measurement, error = create_normal_width_measurement(
                self.doc, (43.0, 40.0 + offset * 0.1), (57.0, 42.0 + offset * 0.1),
            )
            self.assertIsNotNone(measurement, error)
        self.assertEqual(self.doc.cache_build_counts["chains"], 1)
        self.assertEqual(self.doc.cache_build_counts["spatial"], 1)

    def test_width_mouse_move_does_not_call_topology_metrics(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.doc], 0
        app.mode, app.zoom = _Var("measure_width"), 1.0
        app.width_interaction_mode, app.active_pointer_action = "measure", "width_measure"
        app.width_drag_start, app.width_drag_current, app.width_preview = (43.0, 50.0), None, None
        app.space_panning = False
        app.source_point = lambda event: event.rc
        app.context_text = _Var("")
        app.update_context_panel = lambda: None
        app.refresh_dynamic_overlay = lambda: None
        with mock.patch.object(self.doc, "topology_metrics", wraps=self.doc.topology_metrics) as metrics:
            app.mouse_drag(_Event((57.0, 52.0)))
        self.assertEqual(metrics.call_count, 0)
        self.assertIsNotNone(app.width_preview)

    def test_zoom_uses_layers_without_metrics_or_document_mutation(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.doc], 0
        app.canvas = _Canvas()
        app.zoom = 1.0
        app.min_zoom, app.max_zoom = 0.02, 64.0
        app._background_scrollregion = None
        app.schedule_background_refresh = lambda force=False: None
        app.refresh_static_geometry = lambda: None
        app.update_status = lambda: None
        before_nodes, before_edges = self.doc.nodes.copy(), self.doc.edges.copy()
        with mock.patch.object(self.doc, "topology_metrics", wraps=self.doc.topology_metrics) as metrics:
            app.apply_zoom(1.5, _Event())
        self.assertEqual(metrics.call_count, 0)
        np.testing.assert_array_equal(self.doc.nodes, before_nodes)
        np.testing.assert_array_equal(self.doc.edges, before_edges)

    def test_zoom_is_continuous_and_supports_deep_inspection(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.min_zoom, app.max_zoom = 0.02, 64.0
        self.assertAlmostEqual(app._continuous_zoom(3.14159), 3.14159)
        self.assertEqual(app._continuous_zoom(100.0), 64.0)
        self.assertEqual(app._continuous_zoom(0.001), 0.02)

    def test_background_refresh_schedule_is_debounced(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.doc], 0
        app.canvas, app.root, app.zoom = _Canvas(), _Root(), 1.0
        app.surface_versions = [0]
        app._background_refresh_after_id = None
        app._refreshing_background = False
        app._last_background_view = None

        app.schedule_background_refresh()
        app.schedule_background_refresh()

        self.assertEqual(len(app.root.after_calls), 1)
        self.assertEqual(app.root.after_calls[0][1], 20)

    def test_internal_scrollbar_callback_does_not_self_schedule(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.doc], 0
        app.canvas, app.root, app.zoom = _Canvas(), _Root(), 1.0
        app.surface_versions = [0]
        app._background_refresh_after_id = None
        app._refreshing_background = True
        app._last_background_view = None
        app._last_scrollbar_views = {"x": None, "y": None}
        scrollbar = _Scrollbar()

        app._scrollbar_changed("x", scrollbar, "0.0", "1.0")

        self.assertEqual(scrollbar.values, [("0.0", "1.0")])
        self.assertEqual(app.root.after_calls, [])

    def test_node_drag_updates_only_node_and_incident_edge_items(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.doc], 0
        app.canvas, app.zoom = _Canvas(), 1.0
        app.node_items = {5: 500}
        app.edge_items = {edge_id: 100 + edge_id for edge_id in range(len(self.edges))}
        self.doc.nodes[5] = (55, 50)
        app.update_dragged_node_items(5)
        touched_items = {call[0] for call in app.canvas.coords_calls}
        self.assertEqual(touched_items, {500, 104, 105})


if __name__ == "__main__":
    unittest.main()
