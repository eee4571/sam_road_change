from __future__ import annotations

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
    def __init__(self): self.coords_calls = []
    def canvasx(self, value): return value
    def canvasy(self, value): return value
    def winfo_width(self): return 400
    def winfo_height(self): return 300
    def scale(self, *args): pass
    def configure(self, **kwargs): pass
    def xview_moveto(self, value): pass
    def yview_moveto(self, value): pass
    def coords(self, *args): self.coords_calls.append(args)


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
        app.refresh_background = lambda: None
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
