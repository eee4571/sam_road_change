from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import rasterio

ENGINE = Path(__file__).resolve().parent / "engine" / "width"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from chain_width_calculator import (  # noqa: E402
    apply_manual_width_constraints,
    normalize_manual_width_measurements,
)
from finalize_review_results import apply_manual_width_overrides  # noqa: E402
from global_edit_utils import _project_manual_widths  # noqa: E402
from geometry_editor import (  # noqa: E402
    GeometryDocument,
    GeometryEditorApp,
    create_default_interval_width_measurement,
    create_interval_width_measurement,
    create_normal_width_measurement,
    delete_manual_width_interval,
    effective_width_surface_preview,
    load_documents,
    load_manual_widths,
    manual_width_interval_overlaps,
    manual_width_preview_geometry,
    normalize_manual_width_intervals,
    query_effective_width,
    replace_manual_width_interval,
    update_manual_width_interval_endpoint,
)
from width_surface_reconstruction import WidthSurfaceConfig, reconstruct_surface_from_widths  # noqa: E402
from production_workflow import apply_global_edit_directory  # noqa: E402


class _Variable:
    def __init__(self, value=None): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class _Event:
    def __init__(self, rc): self.rc = rc


class _Canvas:
    def __init__(self): self.options = {}
    def configure(self, **kwargs): self.options.update(kwargs)


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

    def _interval(self, lo: float, hi: float, width: float, measurement_id: str) -> dict:
        anchor = self._measurement(width=width, start=(45.0, 0.5 * (lo + hi)), end=(55.0, 0.5 * (lo + hi)))
        interval, error = create_interval_width_measurement(
            self.document, anchor, (50.0, lo), (50.0, hi),
        )
        self.assertEqual(error, "")
        interval["measurement_id"] = measurement_id
        interval["width_px"] = width
        interval["width_units"] = width * self.document.pixel_size
        return interval

    @staticmethod
    def _profile(rows: list[dict]) -> list[tuple[float, float, float]]:
        return sorted(
            (
                round(float(row["range_start_position"]), 5),
                round(float(row["range_end_position"]), 5),
                round(float(row["width_px"]), 5),
            )
            for row in rows if row.get("source") == "manual_interval_width"
        )

    def _interaction_app(self, document: GeometryDocument | None = None) -> GeometryEditorApp:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [document or self.document], 0
        app.mode, app.zoom = _Variable("measure_width"), 1.0
        app.width_interaction_mode, app.active_pointer_action = "select", None
        app.active_width_measurement_id = None
        app.width_range_drag_handle = None
        app.width_range_drag_original = None
        app.width_range_drag_preview = None
        app.width_drag_start = app.width_drag_current = app.width_preview = None
        app.interval_measurement_id, app.interval_draft = None, []
        app.remeasure_interval_id = None
        app.width_draft, app.draft = [], []
        app.space_pressed = app.space_panning = False
        app.surface_stroke_active, app.drag_node = False, None
        app.lasso_active, app.lasso = False, []
        app.selected_edge_ids = set()
        app.context_text, app.saved_var = _Variable(), _Variable()
        app.canvas = _Canvas()
        app.source_point = lambda event: event.rc
        app.refresh_manual_widths = lambda: None
        app.refresh_dynamic_overlay = lambda: None
        app.update_context_panel = lambda: None
        app.update_status = lambda: None
        app._show_unsaved = lambda: None
        app._remove_lasso_canvas = lambda: None
        return app

    def test_drag_events_create_one_measurement(self) -> None:
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.document], 0
        app.mode, app.zoom = _Variable("measure_width"), 1.0
        app.interval_measurement_id, app.interval_draft = None, []
        app.width_draft, app.width_drag_start, app.width_drag_current = [], None, None
        app.width_preview = None
        app.width_interaction_mode, app.active_pointer_action = "measure", None
        app.remeasure_interval_id = None
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

    def test_automatic_profile_is_base_and_manual_interval_wins(self) -> None:
        interval = self._interval(30, 70, 18, "MW00001")
        self.assertEqual(query_effective_width(7.6, [interval], 0, 10), 7.6)
        self.assertEqual(query_effective_width(7.6, [interval], 0, 50), 9.0)
        self.assertEqual(query_effective_width(7.6, [interval], 0, 90), 7.6)

    def test_replacement_splits_old_interval_on_both_sides(self) -> None:
        old = self._interval(0, 100, 16, "MW00001")
        new = self._interval(40, 70, 20, "MW00002")
        result = replace_manual_width_interval([old], new, self.document)
        self.assertEqual(self._profile(result), [(0, 40, 16), (40, 70, 20), (70, 100, 16)])
        self.assertEqual(len({row["measurement_id"] for row in result}), 3)

    def test_replacement_fully_removes_covered_old_interval(self) -> None:
        old = self._interval(20, 60, 16, "MW00001")
        new = self._interval(0, 100, 20, "MW00002")
        self.assertEqual(self._profile(replace_manual_width_interval([old], new, self.document)), [(0, 100, 20)])

    def test_replacement_trims_old_interval_right_side(self) -> None:
        old = self._interval(20, 80, 16, "MW00001")
        new = self._interval(50, 100, 20, "MW00002")
        result = replace_manual_width_interval([old], new, self.document)
        self.assertEqual(self._profile(result), [(20, 50, 16), (50, 100, 20)])

    def test_replacement_covers_multiple_old_intervals(self) -> None:
        rows = [
            self._interval(0, 20, 12, "MW00001"),
            self._interval(25, 45, 14, "MW00002"),
            self._interval(50, 100, 16, "MW00003"),
        ]
        new = self._interval(10, 80, 22, "MW00004")
        self.assertEqual(
            self._profile(replace_manual_width_interval(rows, new, self.document)),
            [(0, 10, 12), (10, 80, 22), (80, 100, 16)],
        )

    def test_delete_interval_restores_automatic_query(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.assertEqual(query_effective_width(7.5, [interval], 0, 50), 8.0)
        deleted = delete_manual_width_interval([interval], "MW00001")
        self.assertEqual(query_effective_width(7.5, deleted, 0, 50), 7.5)

    def test_delete_interval_undo_restores_manual_override(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.document.manual_widths = [interval]
        self.document.checkpoint("widths")
        self.document.manual_widths = delete_manual_width_interval(self.document.manual_widths, "MW00001")
        self.assertTrue(self.document.undo())
        self.assertEqual(self._profile(self.document.manual_widths), [(20, 80, 16)])

    def test_replacement_undo_restores_unsplit_interval(self) -> None:
        old = self._interval(0, 100, 16, "MW00001")
        new = self._interval(40, 70, 20, "MW00002")
        self.document.manual_widths = [old]
        self.document.checkpoint("widths")
        self.document.manual_widths = replace_manual_width_interval(self.document.manual_widths, new, self.document)
        self.assertTrue(self.document.undo())
        self.assertEqual(self._profile(self.document.manual_widths), [(0, 100, 16)])

    def test_replacement_redo_restores_complete_split_structure(self) -> None:
        old = self._interval(0, 100, 16, "MW00001")
        new = self._interval(40, 70, 20, "MW00002")
        self.document.manual_widths = [old]
        self.document.checkpoint("widths")
        self.document.manual_widths = replace_manual_width_interval(self.document.manual_widths, new, self.document)
        split = self._profile(self.document.manual_widths)
        self.assertTrue(self.document.undo())
        self.assertTrue(self.document.redo())
        self.assertEqual(self._profile(self.document.manual_widths), split)

    def test_dragged_interval_overwrites_and_splits_neighbor(self) -> None:
        current = self._interval(20, 40, 16, "MW00001")
        neighbor = self._interval(50, 100, 12, "MW00002")
        moved, error = update_manual_width_interval_endpoint(
            self.document, current, "end", (50, 80), minimum_range_units=1,
        )
        self.assertEqual(error, "")
        result = replace_manual_width_interval([current, neighbor], moved, self.document)
        self.assertEqual(self._profile(result), [(20, 80, 16), (80, 100, 12)])

    def test_ui_handle_drag_is_preview_until_release_and_one_transaction(self) -> None:
        current = self._interval(20, 40, 16, "MW00001")
        neighbor = self._interval(50, 100, 12, "MW00002")
        self.document.manual_widths = [current, neighbor]
        app = GeometryEditorApp.__new__(GeometryEditorApp)
        app.documents, app.document_index = [self.document], 0
        app.mode, app.zoom = _Variable("measure_width"), 1.0
        app.active_width_measurement_id = "MW00001"
        app.width_range_drag_handle = None
        app.width_range_drag_original = None
        app.width_range_drag_preview = None
        app.width_interaction_mode, app.active_pointer_action = "adjust_range", None
        app.width_drag_start = app.width_drag_current = app.width_preview = None
        app.interval_measurement_id, app.interval_draft = None, []
        app.remeasure_interval_id = None
        app.space_pressed = app.space_panning = False
        app.surface_stroke_active, app.drag_node = False, None
        app.lasso_active, app.lasso = False, []
        app.context_text, app.saved_var = _Variable(), _Variable()
        app.source_point = lambda event: event.rc
        app.refresh_manual_widths = lambda: None
        app.refresh_dynamic_overlay = lambda: None
        app.update_context_panel = lambda: None
        app._show_unsaved = lambda: None
        app.mouse_press(_Event((50, 40)))
        app.mouse_drag(_Event((50, 80)))
        self.assertEqual(self._profile(self.document.manual_widths), [(20, 40, 16), (50, 100, 12)])
        self.assertEqual(len(self.document.undo_stack), 0)
        self.assertIsNotNone(app.width_range_drag_preview)
        app.mouse_release(_Event((50, 80)))
        self.assertEqual(self._profile(self.document.manual_widths), [(20, 80, 16), (80, 100, 12)])
        self.assertEqual(len(self.document.undo_stack), 1)

    def test_adjust_range_click_away_from_handle_never_starts_measurement(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.document.manual_widths = [interval]
        app = self._interaction_app()
        app.active_width_measurement_id = "MW00001"
        app.width_interaction_mode = "adjust_range"
        app.mouse_press(_Event((10, 10)))
        self.assertIsNone(app.active_pointer_action)
        self.assertIsNone(app.width_drag_start)
        self.assertIsNone(app.width_drag_current)
        self.assertIsNone(app.width_preview)

    def test_adjust_range_handle_routes_only_to_range_drag(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.document.manual_widths = [interval]
        app = self._interaction_app()
        app.active_width_measurement_id = "MW00001"
        app.width_interaction_mode = "adjust_range"
        app.mouse_press(_Event((50, 80)))
        self.assertEqual(app.active_pointer_action, "range_end_drag")
        self.assertEqual(app.width_range_drag_handle, "end")
        self.assertIsNone(app.width_drag_start)
        app.mouse_drag(_Event((50, 90)))
        self.assertIsNotNone(app.width_range_drag_preview)
        self.assertEqual(len(self.document.manual_widths), 1)

    def test_adjust_range_near_miss_never_becomes_width_measurement(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.document.manual_widths = [interval]
        app = self._interaction_app()
        app.active_width_measurement_id = "MW00001"
        app.width_interaction_mode = "adjust_range"
        app.mouse_press(_Event((50, 95)))
        app.mouse_drag(_Event((35, 95)))
        app.mouse_release(_Event((35, 95)))
        self.assertNotEqual(app.active_pointer_action, "width_measure")
        self.assertIsNone(app.width_drag_start)
        self.assertEqual(len(self.document.manual_widths), 1)

    def test_measure_state_drag_creates_new_width(self) -> None:
        app = self._interaction_app()
        app.width_interaction_mode = "measure"
        app.mouse_press(_Event((42, 50)))
        self.assertEqual(app.active_pointer_action, "width_measure")
        app.mouse_drag(_Event((58, 50)))
        app.mouse_release(_Event((58, 50)))
        self.assertEqual(len(self.document.manual_widths), 1)

    def test_remeasure_updates_width_but_preserves_range(self) -> None:
        interval = self._interval(20, 80, 12, "MW00001")
        self.document.manual_widths = [interval]
        app = self._interaction_app()
        app.active_width_measurement_id = "MW00001"
        app.begin_remeasure_width()
        original_range = (
            interval["range_start_position"], interval["range_end_position"],
        )
        app.mouse_press(_Event((40, 50)))
        app.mouse_drag(_Event((60, 50)))
        app.mouse_release(_Event((60, 50)))
        updated = app.active_width_measurement()
        self.assertEqual(
            (updated["range_start_position"], updated["range_end_position"]),
            original_range,
        )
        self.assertAlmostEqual(updated["width_px"], 20.0)

    def test_select_state_click_interval_only_selects(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.document.manual_widths = [interval]
        app = self._interaction_app()
        app.mouse_press(_Event((50, 50)))
        self.assertEqual(app.active_width_measurement_id, "MW00001")
        self.assertEqual(app.active_pointer_action, "interval_select")
        self.assertIsNone(app.width_drag_start)
        app.mouse_release(_Event((50, 50)))
        self.assertIsNone(app.active_pointer_action)
        self.assertEqual(len(self.document.manual_widths), 1)

    def test_new_measurement_finishes_in_adjust_range_state(self) -> None:
        app = self._interaction_app()
        app.begin_new_width_measurement()
        app.mouse_press(_Event((42, 50)))
        app.mouse_drag(_Event((58, 50)))
        app.mouse_release(_Event((58, 50)))
        self.assertEqual(app.width_interaction_mode, "adjust_range")
        self.assertIsNotNone(app.active_width_measurement())

    def test_range_drag_finishes_in_adjust_range_state(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.document.manual_widths = [interval]
        app = self._interaction_app()
        app.active_width_measurement_id = "MW00001"
        app.width_interaction_mode = "adjust_range"
        app.mouse_press(_Event((50, 80)))
        app.mouse_drag(_Event((50, 90)))
        app.mouse_release(_Event((50, 90)))
        self.assertEqual(app.width_interaction_mode, "adjust_range")
        self.assertIsNone(app.active_pointer_action)

    def test_width_cursor_reflects_interaction_state(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.document.manual_widths = [interval]
        app = self._interaction_app()
        app.active_width_measurement_id = "MW00001"
        app.width_interaction_mode = "measure"
        app.mouse_motion(_Event((10, 10)))
        self.assertEqual(app.canvas.options["cursor"], "crosshair")
        app.width_interaction_mode = "remeasure"
        app.mouse_motion(_Event((10, 10)))
        self.assertEqual(app.canvas.options["cursor"], "crosshair")
        app.width_interaction_mode = "adjust_range"
        app.mouse_motion(_Event((50, 80)))
        self.assertEqual(app.canvas.options["cursor"], "hand2")
        app.mouse_motion(_Event((10, 10)))
        self.assertEqual(app.canvas.options["cursor"], "arrow")

    def test_escape_clears_width_gesture_and_returns_to_safe_state(self) -> None:
        interval = self._interval(20, 80, 16, "MW00001")
        self.document.manual_widths = [interval]
        app = self._interaction_app()
        app.active_width_measurement_id = "MW00001"
        app.width_interaction_mode = "adjust_range"
        app.mouse_press(_Event((50, 80)))
        app.mouse_drag(_Event((50, 90)))
        self.assertIsNotNone(app.width_range_drag_preview)
        app.cancel_drawing()
        self.assertEqual(app.width_interaction_mode, "adjust_range")
        self.assertIsNone(app.active_pointer_action)
        self.assertIsNone(app.width_drag_start)
        self.assertIsNone(app.width_range_drag_handle)
        self.assertIsNone(app.width_range_drag_preview)

    def test_preview_shrinks_when_manual_width_changes_12_to_6(self) -> None:
        auto = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(auto, (0, 44), (110, 56), 1, -1)
        wide = self._interval(20, 80, 24, "MW00001")
        narrow = dict(wide, width_px=12, width_units=6)
        wide_area = int(effective_width_surface_preview(self.document, auto, [wide]).sum())
        narrow_area = int(effective_width_surface_preview(self.document, auto, [narrow]).sum())
        self.assertLess(narrow_area, wide_area)

    def test_preview_grows_when_manual_width_changes_6_to_12(self) -> None:
        auto = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(auto, (0, 47), (110, 53), 1, -1)
        narrow = self._interval(20, 80, 12, "MW00001")
        wide = dict(narrow, width_px=24, width_units=12)
        self.assertGreater(
            int(effective_width_surface_preview(self.document, auto, [wide]).sum()),
            int(effective_width_surface_preview(self.document, auto, [narrow]).sum()),
        )

    def test_preview_after_delete_matches_automatic_surface(self) -> None:
        auto = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(auto, (0, 44), (110, 56), 1, -1)
        interval = self._interval(20, 80, 8, "MW00001")
        changed = effective_width_surface_preview(self.document, auto, [interval])
        restored = effective_width_surface_preview(
            self.document, auto, delete_manual_width_interval([interval], "MW00001"),
        )
        self.assertFalse(np.array_equal(changed, auto))
        self.assertTrue(np.array_equal(restored, auto))

    def test_split_result_round_trips_through_manual_width_json(self) -> None:
        rows = replace_manual_width_interval(
            [self._interval(0, 100, 16, "MW00001")],
            self._interval(40, 70, 20, "MW00002"), self.document,
        )
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "tile_manual_widths.json").write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8",
            )
            reopened = GeometryDocument(
                "tile", self.document.image, self.nodes, self.edges, self.document.mask,
                manual_widths=load_manual_widths(directory, "tile"),
            )
        self.assertEqual(self._profile(reopened.manual_widths), self._profile(rows))

    def test_apply_pipeline_rebuilds_narrower_and_wider_surfaces(self) -> None:
        def rebuilt_area(width_px: float | None) -> int:
            records, widths = self._rows(width=24)
            manual = [] if width_px is None else [self._interval(0, 100, width_px, "MW00001")]
            apply_manual_width_constraints(
                self.nodes, self.edges, records, widths, manual, 0.5,
            )
            result = reconstruct_surface_from_widths(
                (120, 120), self.nodes, self.edges, widths, [],
                edge_metadata=records,
                config=WidthSurfaceConfig(regular_surface=True, preserve_reference_surface=False),
            )
            return int(result.surface.sum())

        automatic_area = rebuilt_area(None)
        narrow_area = rebuilt_area(12)
        wide_area = rebuilt_area(30)
        self.assertLess(narrow_area, automatic_area)
        self.assertGreater(wide_area, automatic_area)

    def test_normalize_legacy_overlaps_uses_later_edit_precedence(self) -> None:
        old = self._interval(0, 100, 16, "MW00001")
        new = self._interval(40, 70, 20, "MW00002")
        old["edit_order"], new["edit_order"] = 0, 1
        self.assertEqual(
            self._profile(normalize_manual_width_intervals([old, new], self.document)),
            [(0, 40, 16), (40, 70, 20), (70, 100, 16)],
        )

    def test_real_area1_2021_apply_edits_rebuilds_manual_width(self) -> None:
        if os.environ.get("SAMROAD_REAL_WIDTH_VERIFY") != "1":
            self.skipTest("set SAMROAD_REAL_WIDTH_VERIFY=1 for the isolated real-result verification")
        period = Path(
            "project/04_成果输出/run_20260818_231625/grids/area1/periods/2021"
        ).resolve()
        review = period / "runs/roads/width_review"
        source_edit = period / "runs/roads/centerline_edit"
        if not (review.is_dir() and source_edit.is_dir()):
            self.skipTest("run_20260818_231625 area1/2021 is not available")
        manifest = json.loads((source_edit / "edited_manifest.json").read_text(encoding="utf-8"))
        widths = json.loads((source_edit / "global_manual_widths.json").read_text(encoding="utf-8"))
        intervals = [row for row in widths if row.get("source") == "manual_interval_width"]
        self.assertTrue(intervals)
        global_transform = rasterio.Affine(*manifest["global_transform"][:6])
        selected_stem = None
        selected_interval = None
        for stem in manifest.get("affected_tiles", []):
            summary = json.loads((review / f"{stem}_summary.json").read_text(encoding="utf-8"))
            with rasterio.open(Path(summary["image"])) as dataset:
                for interval in intervals:
                    projected = _project_manual_widths(
                        [interval], global_transform, manifest["global_crs"],
                        dataset.transform, dataset.crs, dataset.bounds,
                    )
                    if any(row.get("source") == "manual_interval_width" for row in projected):
                        selected_stem, selected_interval = str(stem), interval
                        break
            if selected_interval is not None:
                break
        self.assertIsNotNone(selected_interval)

        def resize_interval(width_units: float) -> dict:
            row = dict(selected_interval)
            target = np.asarray([row["target_row"], row["target_col"]], dtype=np.float64)
            vector = np.asarray([
                row["end_row"] - row["start_row"],
                row["end_col"] - row["start_col"],
            ], dtype=np.float64)
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            half_pixels = 0.5 * width_units / max(
                abs(float(global_transform.a)), abs(float(global_transform.e)), 1e-9,
            )
            start, end = target - vector * half_pixels, target + vector * half_pixels
            row.update({
                "start_row": float(start[0]), "start_col": float(start[1]),
                "end_row": float(end[0]), "end_col": float(end[1]),
                "width_px": float(2.0 * half_pixels), "width_units": float(width_units),
            })
            return row

        def rebuilt_area(root: Path, variant: str, replacement: dict | None) -> int:
            edited = root / variant / "edited"
            final = root / variant / "final"
            edited.mkdir(parents=True)
            local_manifest = copy.deepcopy(manifest)
            for key in ("global_centerlines", "global_manual_surface_add", "global_manual_surface_remove"):
                source = Path(manifest[key])
                destination = edited / source.name
                shutil.copy2(source, destination)
                local_manifest[key] = str(destination)
            local_widths = [
                row for row in widths
                if row.get("measurement_id") != selected_interval.get("measurement_id")
            ]
            if replacement is not None:
                local_widths.append(replacement)
            width_path = edited / "global_manual_widths.json"
            width_path.write_text(json.dumps(local_widths, ensure_ascii=False), encoding="utf-8")
            local_manifest["global_manual_widths"] = str(width_path)
            local_manifest["affected_tiles"] = [selected_stem]
            (edited / "edited_manifest.json").write_text(
                json.dumps(local_manifest, ensure_ascii=False), encoding="utf-8",
            )
            apply_global_edit_directory(review, edited, {selected_stem})
            completed = subprocess.run(
                [
                    sys.executable, str(ENGINE / "finalize_review_results.py"),
                    "--output-dir", str(review), "--edited-dir", str(edited),
                    "--final-dir", str(final), "--only-stem", selected_stem,
                ],
                cwd=Path(__file__).resolve().parent,
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            surface = cv2.imread(str(final / f"{selected_stem}_optimized_road_surface.png"), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(surface)
            return int(np.count_nonzero(surface))

        with tempfile.TemporaryDirectory(prefix="samroad-real-width-") as raw:
            root = Path(raw)
            narrow_area = rebuilt_area(root, "manual_6m", resize_interval(6.0))
            wide_area = rebuilt_area(root, "manual_12m", resize_interval(12.0))
            automatic_area = rebuilt_area(root, "automatic", None)
        self.assertLess(narrow_area, wide_area)
        self.assertNotEqual(automatic_area, narrow_area)
        self.assertNotEqual(automatic_area, wide_area)
        print(json.dumps({
            "real_period": str(period), "stem": selected_stem,
            "manual_6m_surface_px": narrow_area,
            "manual_12m_surface_px": wide_area,
            "deleted_override_surface_px": automatic_area,
            "real_outputs_modified": False,
        }, ensure_ascii=False))

    def test_real_area1_2021_width_pointer_routing(self) -> None:
        if os.environ.get("SAMROAD_REAL_WIDTH_UI_VERIFY") != "1":
            self.skipTest("set SAMROAD_REAL_WIDTH_UI_VERIFY=1 for the real pointer-routing verification")
        period = Path(
            "project/04_成果输出/run_20260818_231625/grids/area1/periods/2021"
        ).resolve()
        run_root = period / "runs/roads"
        review = run_root / "width_review"
        edited = run_root / "centerline_edit"
        centerlines = run_root / "products/road_centerlines.shp"
        surfaces = run_root / "products/road_surfaces.shp"
        if not all(path.exists() for path in (review, edited, centerlines, surfaces)):
            self.skipTest("run_20260818_231625 area1/2021 is not available")

        document = load_documents(review, edited, centerlines, surfaces)[0]
        interval = next(
            row for row in document.manual_widths
            if row.get("source") == "manual_interval_width"
        )
        app = self._interaction_app(document)
        app.active_width_measurement_id = str(interval["measurement_id"])
        app.width_interaction_mode = "adjust_range"
        start = (
            float(interval["range_start_row"]),
            float(interval["range_start_col"]),
        )

        # Repeated clicks on both sides of the 14-screen-pixel hit boundary
        # may drag/select/do nothing, but must never become a width measure.
        routed_actions = []
        for offset in (0.0, 13.0, 15.0, -13.0, -15.0):
            point = (start[0] + offset, start[1])
            app.mouse_press(_Event(point))
            routed_actions.append(app.active_pointer_action)
            self.assertNotEqual(app.active_pointer_action, "width_measure")
            self.assertIsNone(app.width_drag_start)
            app.mouse_drag(_Event((point[0] + 6.0, point[1] + 6.0)))
            self.assertIsNone(app.width_preview)
            app.cancel_drawing()

        app.begin_new_width_measurement()
        measure_start = (float(interval["start_row"]), float(interval["start_col"]))
        measure_end = (float(interval["end_row"]), float(interval["end_col"]))
        with mock.patch("geometry_editor.messagebox.showwarning") as warning:
            app.mouse_press(_Event(measure_start))
            self.assertEqual(app.active_pointer_action, "width_measure")
            app.mouse_drag(_Event(measure_end))
            app.mouse_release(_Event(measure_end))
        warning.assert_not_called()
        self.assertEqual(app.width_interaction_mode, "adjust_range")
        self.assertIsNotNone(app.active_width_measurement())
        print(json.dumps({
            "real_period": str(period),
            "measurement_id": str(interval["measurement_id"]),
            "near_handle_actions": routed_actions,
            "new_measurement_state": app.width_interaction_mode,
            "formal_outputs_modified": False,
        }, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
