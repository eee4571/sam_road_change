from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ENGINE = Path(__file__).resolve().parent / "engine" / "width"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from chain_width_calculator import (  # noqa: E402
    apply_manual_width_constraints,
    normalize_manual_width_measurements,
)
from finalize_review_results import apply_manual_width_overrides  # noqa: E402
from geometry_editor import (  # noqa: E402
    GeometryDocument,
    GeometryEditorApp,
    create_default_interval_width_measurement,
    create_interval_width_measurement,
    create_normal_width_measurement,
    load_manual_widths,
    manual_width_interval_overlaps,
    manual_width_preview_geometry,
    update_manual_width_interval_endpoint,
)
from width_surface_reconstruction import WidthSurfaceConfig, reconstruct_surface_from_widths  # noqa: E402


class _Variable:
    def __init__(self, value=None): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class _Event:
    def __init__(self, rc): self.rc = rc


class ManualWidthOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = np.asarray([[50, col] for col in range(0, 110, 10)], dtype=np.float32)
        self.edges = np.asarray([[index, index + 1] for index in range(10)], dtype=np.int32)
        self.document = GeometryDocument(
            "tile", np.zeros((120, 120, 3), dtype=np.uint8), self.nodes, self.edges,
            np.zeros((120, 120), dtype=np.uint8),
        )
        self.document.pixel_size = 0.5

    def _measurement(self, width=12.0, start=(44.0, 50.0), end=(58.0, 54.0)) -> dict:
        measurement, error = create_normal_width_measurement(self.document, start, end)
        self.assertEqual(error, "")
        self.assertIsNotNone(measurement)
        measurement["width_px"] = width
        measurement["width_units"] = width * self.document.pixel_size
        return measurement

    def _rows(self, width=6.0, grade="B"):
        records = [{"optimized_width_px": width, "optimized_quality_grade": grade} for _ in self.edges]
        widths = [
            {"edge_id": edge_id, "width_px": width, "width_units": width * 0.5,
             "source": "chain_hybrid_v2", "quality_grade": grade}
            for edge_id in range(len(self.edges))
        ]
        return records, widths

    def _default_interval(self, width=12.0, max_length=30.0) -> dict:
        interval, error = create_default_interval_width_measurement(
            self.document, self._measurement(width=width),
            max_default_length_units=max_length,
        )
        self.assertEqual(error, "")
        self.assertIsNotNone(interval)
        return interval

    def test_drag_events_create_one_measurement(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.document], 0
        app.mode, app.zoom = _Variable("measure_width"), 1.0
        app.interval_measurement_id, app.interval_draft = None, []
        app.width_draft, app.width_drag_start, app.width_drag_current = [], None, None
        app.width_preview = None
        app.surface_stroke_active, app.drag_node, app.lasso = False, None, []
        app.space_pressed, app.space_panning, app.lasso_active = False, False, False
        app.saved_var, app.context_text = _Variable(), _Variable()
        app.source_point = lambda event: event.rc
        app.refresh = lambda: None
        app.refresh_dynamic_overlay = lambda: None
        app.refresh_manual_widths = lambda: None
        app.update_context_panel = lambda: None
        app._mode_changed = lambda: None
        with mock.patch("geometry_editor.messagebox.showwarning"):
            app.mouse_press(_Event((44.0, 48.0)))
            app.mouse_drag(_Event((58.0, 54.0)))
            app.mouse_release(_Event((58.0, 54.0)))
        self.assertEqual(len(self.document.manual_widths), 1)
        self.assertIsNone(app.width_drag_start)

    def test_diagonal_drag_is_saved_on_chain_normal(self) -> None:
        measurement, error = create_normal_width_measurement(
            self.document, (44.0, 47.0), (58.0, 55.0),
        )
        self.assertEqual(error, "")
        self.assertAlmostEqual(measurement["start_col"], 51.0, places=5)
        self.assertAlmostEqual(measurement["end_col"], 51.0, places=5)
        self.assertAlmostEqual(measurement["width_px"], 14.0, places=5)
        self.assertLess(measurement["width_px"], math.hypot(14.0, 8.0))
        self.assertEqual(measurement["target_chain_id"], 0)
        self.assertAlmostEqual(measurement["chain_position"], 51.0, places=5)
        self.assertAlmostEqual(measurement["width_units"], 7.0, places=5)

    def test_along_road_drag_is_rejected(self) -> None:
        measurement, error = create_normal_width_measurement(
            self.document, (50.0, 20.0), (51.0, 80.0),
        )
        self.assertIsNone(measurement)
        self.assertIn("跨道路", error)

    def test_one_anchor_does_not_replace_complete_long_chain(self) -> None:
        records, widths = self._rows()
        measurement = self._measurement(width=12.0)
        count = apply_manual_width_constraints(
            self.nodes, self.edges, records, widths, [measurement], 0.5,
        )
        self.assertGreater(count, 0)
        self.assertLess(count, len(self.edges))
        self.assertEqual(widths[measurement["target_edge_id"]]["width_px"], 12.0)
        self.assertEqual(widths[0]["width_px"], 6.0)
        self.assertEqual(widths[-1]["width_px"], 6.0)

    def test_multiple_anchors_keep_different_local_widths(self) -> None:
        first = self._measurement(width=8.0, start=(46.0, 20.0), end=(54.0, 20.0))
        self.document.manual_widths.append(first)
        second = self._measurement(width=14.0, start=(43.0, 90.0), end=(57.0, 90.0))
        records, widths = self._rows()
        apply_manual_width_constraints(self.nodes, self.edges, records, widths, [first, second], 0.5)
        self.assertEqual(widths[first["target_edge_id"]]["width_px"], 8.0)
        self.assertEqual(widths[second["target_edge_id"]]["width_px"], 14.0)

    def test_interval_spans_edges_but_leaves_outside_automatic(self) -> None:
        anchor = self._measurement(width=11.0)
        interval, error = create_interval_width_measurement(
            self.document, anchor, (50.0, 20.0), (50.0, 80.0),
        )
        self.assertEqual(error, "")
        records, widths = self._rows()
        count = apply_manual_width_constraints(self.nodes, self.edges, records, widths, [interval], 0.5)
        self.assertEqual(count, 6)
        self.assertEqual([row["width_px"] for row in widths[2:8]], [11.0] * 6)
        self.assertEqual(widths[0]["width_px"], 6.0)
        self.assertEqual(widths[9]["width_px"], 6.0)
        self.assertTrue(all(row["source"] == "manual_interval_width" for row in widths[2:8]))

    def test_interval_cannot_cross_junction_or_another_chain(self) -> None:
        nodes = np.asarray([[50, 0], [50, 20], [50, 40], [30, 20]], dtype=np.float32)
        edges = np.asarray([[0, 1], [1, 2], [1, 3]], dtype=np.int32)
        document = GeometryDocument(
            "junction", np.zeros((80, 80, 3), dtype=np.uint8), nodes, edges,
            np.zeros((80, 80), dtype=np.uint8),
        )
        anchor, _ = create_normal_width_measurement(document, (44, 10), (56, 10))
        interval, error = create_interval_width_measurement(document, anchor, (50, 5), (35, 20))
        self.assertIsNone(interval)
        self.assertIn("同一连续道路链", error)

    def test_measurement_is_promoted_to_bounded_interval_on_target_chain(self) -> None:
        interval = self._default_interval(max_length=30.0)
        self.assertEqual(interval["source"], "manual_interval_width")
        self.assertEqual(interval["target_chain_id"], 0)
        self.assertLess(interval["range_start_position"], interval["chain_position"])
        self.assertGreater(interval["range_end_position"], interval["chain_position"])
        self.assertAlmostEqual(
            (interval["range_end_position"] - interval["range_start_position"])
            * self.document.pixel_size,
            30.0,
        )

    def test_dragging_start_and_end_updates_ordered_interval(self) -> None:
        interval = self._default_interval(max_length=40.0)
        moved_start, error = update_manual_width_interval_endpoint(
            self.document, interval, "start", (50.0, 25.0),
        )
        self.assertEqual(error, "")
        self.assertAlmostEqual(moved_start["range_start_position"], 25.0)
        moved_end, error = update_manual_width_interval_endpoint(
            self.document, moved_start, "end", (50.0, 85.0),
        )
        self.assertEqual(error, "")
        self.assertAlmostEqual(moved_end["range_end_position"], 85.0)
        self.assertLess(moved_end["range_start_position"], moved_end["range_end_position"])

    def test_dragged_handle_is_projected_to_original_chain(self) -> None:
        interval = self._default_interval(max_length=30.0)
        moved, error = update_manual_width_interval_endpoint(
            self.document, interval, "start", (65.0, 20.0),
        )
        self.assertEqual(error, "")
        self.assertEqual(moved["target_chain_id"], interval["target_chain_id"])
        self.assertAlmostEqual(moved["range_start_row"], 50.0)

    def test_endpoint_drag_is_one_undo_redo_step(self) -> None:
        interval = self._default_interval(max_length=30.0)
        self.document.manual_widths = [interval]
        original_start = interval["range_start_position"]
        self.document.checkpoint()
        moved, error = update_manual_width_interval_endpoint(
            self.document, interval, "start", (50.0, 15.0),
        )
        self.assertEqual(error, "")
        self.document.manual_widths[0] = moved
        moved_start = moved["range_start_position"]
        self.assertNotEqual(moved_start, original_start)
        self.assertTrue(self.document.undo())
        self.assertAlmostEqual(self.document.manual_widths[0]["range_start_position"], original_start)
        self.assertTrue(self.document.redo())
        self.assertAlmostEqual(self.document.manual_widths[0]["range_start_position"], moved_start)

    def test_overlapping_intervals_are_detected_but_touching_is_allowed(self) -> None:
        first = self._default_interval(max_length=30.0)
        overlapping = dict(
            first, measurement_id="MW99998",
            range_start_position=first["range_end_position"] - 5.0,
            range_end_position=first["range_end_position"] + 5.0,
        )
        touching = dict(
            first, measurement_id="MW99999",
            range_start_position=first["range_end_position"],
            range_end_position=first["range_end_position"] + 5.0,
        )
        self.assertTrue(manual_width_interval_overlaps([first], overlapping))
        self.assertFalse(manual_width_interval_overlaps([first], touching))

    def test_preview_area_increases_with_width_and_range(self) -> None:
        narrow = self._default_interval(width=8.0, max_length=20.0)
        wide = dict(narrow, width_px=16.0, width_units=8.0)
        long = self._default_interval(width=8.0, max_length=40.0)
        narrow_preview = manual_width_preview_geometry(self.document, narrow)
        wide_preview = manual_width_preview_geometry(self.document, wide)
        long_preview = manual_width_preview_geometry(self.document, long)
        self.assertGreater(wide_preview.area, narrow_preview.area)
        self.assertGreater(long_preview.area, narrow_preview.area)

    def test_preview_does_not_write_formal_surface(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            surface = Path(raw) / "road_surfaces.shp"
            surface.write_bytes(b"formal-surface-sentinel")
            before = surface.read_bytes()
            preview = manual_width_preview_geometry(self.document, self._default_interval())
            self.assertFalse(preview.is_empty)
            self.assertEqual(surface.read_bytes(), before)

    def test_later_edit_has_predictable_precedence(self) -> None:
        anchor = self._measurement(width=8.0)
        interval, _ = create_interval_width_measurement(self.document, anchor, (50, 20), (50, 80))
        later = dict(anchor, width_px=15.0, width_units=7.5, edit_order=2)
        records, widths = self._rows()
        apply_manual_width_constraints(self.nodes, self.edges, records, widths, [interval, later], 0.5)
        self.assertEqual(widths[later["target_edge_id"]]["width_px"], 15.0)

    def test_delete_measurement_restores_automatic_width(self) -> None:
        records, widths = self._rows()
        apply_manual_width_constraints(
            self.nodes, self.edges, records, widths, [self._measurement(width=12.0)], 0.5,
        )
        restored_records, restored_widths = self._rows()
        count = apply_manual_width_constraints(
            self.nodes, self.edges, restored_records, restored_widths, [], 0.5,
        )
        self.assertEqual(count, 0)
        self.assertTrue(all(row["width_px"] == 6.0 for row in restored_widths))

    def test_old_json_is_normalized_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path = directory / "tile_manual_widths.json"
            path.write_text(json.dumps([{
                "target_row": 50, "target_col": 45, "width_px": 12,
                "start_row": 44, "start_col": 45, "end_row": 56, "end_col": 45,
            }]), encoding="utf-8")
            normalized = normalize_manual_width_measurements(
                self.nodes, self.edges, load_manual_widths(directory, "tile"),
            )
            path.write_text(json.dumps(normalized), encoding="utf-8")
            reopened = GeometryDocument(
                "tile", self.document.image, self.nodes, self.edges, self.document.mask,
                manual_widths=load_manual_widths(directory, "tile"),
            )
        self.assertEqual(len(reopened.manual_widths), 1)
        self.assertIn("target_chain_id", reopened.manual_widths[0])
        self.assertIn("chain_position", reopened.manual_widths[0])

    def test_interval_json_round_trip_preserves_range(self) -> None:
        interval = self._default_interval(max_length=30.0)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "tile_manual_widths.json").write_text(
                json.dumps([interval], ensure_ascii=False), encoding="utf-8",
            )
            reopened = GeometryDocument(
                "tile", self.document.image, self.nodes, self.edges, self.document.mask,
                manual_widths=load_manual_widths(directory, "tile"),
            )
        saved = reopened.manual_widths[0]
        self.assertEqual(saved["source"], "manual_interval_width")
        self.assertEqual(saved["target_chain_id"], interval["target_chain_id"])
        self.assertAlmostEqual(saved["range_start_position"], interval["range_start_position"])
        self.assertAlmostEqual(saved["range_end_position"], interval["range_end_position"])

    def test_file_wrapper_preserves_automatic_edges_outside_anchor(self) -> None:
        records, widths = self._rows()
        measurement = self._measurement(width=12.0)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tile_manual_widths.json"
            path.write_text(json.dumps([measurement]), encoding="utf-8")
            count = apply_manual_width_overrides(self.nodes, self.edges, records, widths, path, 0.5)
        self.assertGreater(count, 0)
        self.assertLess(count, len(self.edges))
        self.assertEqual(widths[0]["source"], "chain_hybrid_v2")

    def test_surface_regularization_preserves_manual_interval_widths(self) -> None:
        records, widths = self._rows()
        anchor = self._measurement(width=12.0)
        interval, _ = create_interval_width_measurement(
            self.document, anchor, (50, 20), (50, 80),
        )
        apply_manual_width_constraints(
            self.nodes, self.edges, records, widths, [interval], 0.5,
        )
        result = reconstruct_surface_from_widths(
            (120, 120), self.nodes, self.edges, widths, [],
            edge_metadata=records, config=WidthSurfaceConfig(regular_surface=True),
        )
        resolved = result.metadata["resolved_widths_px"]
        self.assertEqual(resolved[2:8], [12.0] * 6)
        self.assertTrue(np.any(result.surface))


if __name__ == "__main__":
    unittest.main()
