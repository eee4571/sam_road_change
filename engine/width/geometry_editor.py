from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import json
import math
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont, messagebox, ttk

import cv2
import geopandas as gpd
import networkx as nx
import numpy as np
from PIL import Image, ImageTk
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from finalize_review_results import load_graph, save_graph  # noqa: E402
from chain_width_calculator import (  # noqa: E402
    ChainProjection,
    _chain_geometry,
    _point_at,
    _tangent_at,
    build_road_chains,
    normalize_manual_width_measurements,
    project_point_to_road_chain,
)
from global_edit_utils import _graph_from_world_lines, _project_manual_widths  # noqa: E402
from review_geometry import accepted_surface_region_polylines  # noqa: E402


EDITOR_CACHE_VERSION = 1
EDITOR_CACHE_DIRECTORY_NAME = ".editor_cache"
BACKGROUND_CACHE_NAME = "background_mosaic.tif"
SURFACE_CACHE_NAME = "surface_mask.tif"
CACHE_METADATA_NAME = "cache_meta.json"
CACHE_LOCK_NAME = "cache.lock"
SHAPEFILE_SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Read an image through Unicode-safe bytes while preserving OpenCV flags.

    On Windows, bundled OpenCV builds may fail to open non-ASCII paths via
    ``cv2.imread``.  ``np.fromfile`` delegates path handling to Python's wide
    character APIs and ``cv2.imdecode`` retains the requested color/gray mode.
    Missing or undecodable files follow ``cv2.imread``'s ``None`` contract.
    """
    try:
        encoded = np.fromfile(str(Path(path)), dtype=np.uint8)
    except (OSError, TypeError, ValueError):
        return None
    if encoded.size == 0:
        return None
    try:
        return cv2.imdecode(encoded, flags)
    except cv2.error:
        return None


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def candidate_points(row: dict) -> list[tuple[float, float]]:
    try:
        points = json.loads(row.get("polyline_points_json", ""))
        if len(points) >= 2:
            return [(float(point[0]), float(point[1])) for point in points]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return [
        (float(row.get("start_row", 0)), float(row.get("start_col", 0))),
        (float(row.get("end_row", 0)), float(row.get("end_col", 0))),
    ]


def candidate_is_accepted(
    stem: str, row: dict, decisions: dict[tuple[str, str, str], str]
) -> bool:
    candidate_id = str(row.get("candidate_id", ""))
    decision = decisions.get((stem, "candidate_centerline", candidate_id), "")
    return decision == "accept" or (not decision and row.get("auto_decision") == "accept")


def polyline_is_covered(
    document: "GeometryDocument", points: list[tuple[float, float]], tolerance: float = 2.5
) -> bool:
    """Avoid re-adding a candidate already present in an imported/edit graph."""
    if len(points) < 2 or len(document.edges) == 0:
        return False
    segments = document.nodes[document.edges]
    starts = segments[:, 0]
    vectors = segments[:, 1] - starts
    denominators = np.sum(vectors * vectors, axis=1)
    lower = np.minimum(segments[:, 0], segments[:, 1]) - tolerance
    upper = np.maximum(segments[:, 0], segments[:, 1]) + tolerance

    sampled_points: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        start_array = np.asarray(start, dtype=np.float32)
        end_array = np.asarray(end, dtype=np.float32)
        length = float(np.linalg.norm(end_array - start_array))
        steps = max(1, int(math.ceil(length / max(1.0, tolerance))))
        sampled_points.extend(
            start_array + (end_array - start_array) * (index / steps)
            for index in range(steps + 1)
        )
    query_points = np.asarray(sampled_points, dtype=np.float32)
    for chunk in np.array_split(query_points, max(1, int(math.ceil(len(query_points) / 256)))):
        in_box = (
            (chunk[:, None, 0] >= lower[None, :, 0])
            & (chunk[:, None, 0] <= upper[None, :, 0])
            & (chunk[:, None, 1] >= lower[None, :, 1])
            & (chunk[:, None, 1] <= upper[None, :, 1])
        )
        safe_denominators = np.where(denominators > 1e-9, denominators, 1.0)
        ratios = np.sum((chunk[:, None, :] - starts[None, :, :]) * vectors[None, :, :], axis=2)
        ratios = np.clip(ratios / safe_denominators[None, :], 0.0, 1.0)
        projections = starts[None, :, :] + ratios[:, :, None] * vectors[None, :, :]
        distances = np.linalg.norm(chunk[:, None, :] - projections, axis=2)
        distances[~in_box] = np.inf
        if np.any(np.min(distances, axis=1) > tolerance):
            return False
    return True


def apply_accepted_candidates(
    stem: str,
    document: "GeometryDocument",
    candidates: list[dict],
    decisions: dict[tuple[str, str, str], str],
) -> None:
    for row in candidates:
        if not candidate_is_accepted(stem, row, decisions):
            continue
        points = candidate_points(row)
        if not polyline_is_covered(document, points):
            document.add_polyline(points)


def current_edited_graph(summary: dict, edited_graph: Path) -> bool:
    """Return true only when the edit was saved after the current prepared graph."""
    if not edited_graph.is_file():
        return False
    prepared_graph = Path(summary.get("prepared_graph", summary.get("graph", "")))
    if not prepared_graph.is_absolute():
        prepared_graph = Path.cwd() / prepared_graph
    if not prepared_graph.is_file():
        return True
    return edited_graph.stat().st_mtime_ns >= prepared_graph.stat().st_mtime_ns


def point_segment_projection(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    ratio = 0.0 if denominator <= 1e-9 else float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
    projection = start + vector * ratio
    return projection, float(np.linalg.norm(point - projection))


def _next_measurement_id(measurements: list[dict]) -> str:
    used = {str(row.get("measurement_id", "")) for row in measurements}
    index = len(measurements) + 1
    while f"MW{index:05d}" in used:
        index += 1
    return f"MW{index:05d}"


def create_normal_width_measurement(
    document: "GeometryDocument",
    start_rc: tuple[float, float],
    end_rc: tuple[float, float],
    *,
    max_centerline_distance: float = 24.0,
    minimum_normal_cosine: float = 0.5,
) -> tuple[dict | None, str]:
    """Turn a rough mouse drag into a chain-normal boundary measurement."""
    start = np.asarray(start_rc, dtype=np.float32)
    end = np.asarray(end_rc, dtype=np.float32)
    drag = end - start
    drag_length = float(np.linalg.norm(drag))
    if drag_length < 1.0:
        return None, "测宽线过短，请跨道路两侧拖动。"
    midpoint = 0.5 * (start + end)
    if hasattr(document, "project_to_road_chain"):
        projection = document.project_to_road_chain(
            midpoint, max_distance=max_centerline_distance,
        )
    else:
        projection = project_point_to_road_chain(document.nodes, document.edges, midpoint)
    if projection is None or projection.distance > max_centerline_distance:
        return None, "测宽线中点没有落在中心线附近，请跨道路两侧拖动。"
    tangent = projection.tangent / max(float(np.linalg.norm(projection.tangent)), 1e-6)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    normal_cosine = abs(float(np.dot(drag / drag_length, normal)))
    if normal_cosine < minimum_normal_cosine:
        return None, "请跨道路两侧拖动测宽线。"
    start_offset = float(np.dot(start - projection.point, normal))
    end_offset = float(np.dot(end - projection.point, normal))
    if start_offset * end_offset > 0:
        return None, "请从道路一侧拖到另一侧。"
    corrected_start = projection.point + normal * start_offset
    corrected_end = projection.point + normal * end_offset
    width_px = abs(end_offset - start_offset)
    if width_px < 1.0:
        return None, "测宽线过短，请跨道路两侧拖动。"
    transform = getattr(document, "global_transform", None)
    if transform is None:
        transform = getattr(document, "raster_transform", None)
    if transform is not None:
        start_xy = transform * (float(corrected_start[1]), float(corrected_start[0]))
        end_xy = transform * (float(corrected_end[1]), float(corrected_end[0]))
        width_units = float(math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
    else:
        width_units = width_px * float(getattr(document, "pixel_size", 1.0))
    measurement = {
        "measurement_id": _next_measurement_id(document.manual_widths),
        "target_chain_id": int(projection.chain_id),
        "chain_position": float(projection.chain_position),
        "target_edge_id": int(projection.edge_id),
        "target_row": float(projection.point[0]), "target_col": float(projection.point[1]),
        "start_row": float(corrected_start[0]), "start_col": float(corrected_start[1]),
        "end_row": float(corrected_end[0]), "end_col": float(corrected_end[1]),
        "width_px": float(width_px), "width_units": float(width_units),
        "target_distance_px": float(projection.distance),
        "source": "manual_boundary_measurement", "quality_grade": "A",
        "edit_order": len(document.manual_widths),
    }
    return measurement, ""


def create_interval_width_measurement(
    document: "GeometryDocument",
    measurement: dict,
    start_rc: tuple[float, float],
    end_rc: tuple[float, float],
    *,
    max_centerline_distance: float = 24.0,
) -> tuple[dict | None, str]:
    """Convert one point anchor to a strict interval on the same road chain."""
    try:
        chain_id = int(measurement["target_chain_id"])
    except (KeyError, TypeError, ValueError):
        return None, "请先完成一次人工测宽。"
    if hasattr(document, "project_to_road_chain"):
        nearest_start = document.project_to_road_chain(
            start_rc, max_distance=max_centerline_distance,
        )
        nearest_end = document.project_to_road_chain(
            end_rc, max_distance=max_centerline_distance,
        )
        start = document.project_to_road_chain(
            start_rc, chain_id=chain_id, max_distance=max_centerline_distance,
        )
        end = document.project_to_road_chain(
            end_rc, chain_id=chain_id, max_distance=max_centerline_distance,
        )
    else:
        nearest_start = project_point_to_road_chain(document.nodes, document.edges, start_rc)
        nearest_end = project_point_to_road_chain(document.nodes, document.edges, end_rc)
        start = project_point_to_road_chain(document.nodes, document.edges, start_rc, chain_id=chain_id)
        end = project_point_to_road_chain(document.nodes, document.edges, end_rc, chain_id=chain_id)
    if (
        start is None or end is None or nearest_start is None or nearest_end is None
        or start.distance > max_centerline_distance or end.distance > max_centerline_distance
    ):
        return None, "区间端点必须落在道路中心线上。"
    if (
        start.distance > nearest_start.distance + 2.0
        or end.distance > nearest_end.distance + 2.0
    ):
        return None, "区间终点必须位于同一连续道路链上。"
    result = dict(measurement)
    result.update({
        "source": "manual_interval_width", "quality_grade": "A",
        "range_start_position": float(min(start.chain_position, end.chain_position)),
        "range_end_position": float(max(start.chain_position, end.chain_position)),
        "range_start_row": float(start.point[0]), "range_start_col": float(start.point[1]),
        "range_end_row": float(end.point[0]), "range_end_col": float(end.point[1]),
    })
    return result, ""


def chain_interval_points(
    document: "GeometryDocument", chain_id: int, start_position: float, end_position: float,
) -> np.ndarray:
    """Return a clipped chain polyline for one finite chain-position interval."""
    chains = {int(chain.chain_id): chain for chain in document.road_chains()}
    chain = chains.get(int(chain_id))
    if chain is None:
        return np.empty((0, 2), dtype=np.float32)
    points, cumulative = document._chain_geometry_cache[int(chain_id)]
    total = float(cumulative[-1])
    lo = float(np.clip(min(start_position, end_position), 0.0, total))
    hi = float(np.clip(max(start_position, end_position), 0.0, total))
    if hi - lo <= 1e-6:
        return np.empty((0, 2), dtype=np.float32)
    start, _ = _point_at(points, cumulative, lo)
    end, _ = _point_at(points, cumulative, hi)
    interior = points[(cumulative > lo + 1e-6) & (cumulative < hi - 1e-6)]
    return np.vstack((start, interior, end)).astype(np.float32)


def create_default_interval_width_measurement(
    document: "GeometryDocument", measurement: dict,
    *, max_default_length_units: float = 100.0,
) -> tuple[dict | None, str]:
    """Promote one boundary measurement to a bounded natural-chain interval."""
    try:
        chain_id = int(measurement["target_chain_id"])
        center = float(measurement["chain_position"])
    except (KeyError, TypeError, ValueError):
        return None, "测宽结果没有匹配到道路链，请重新测量。"
    chains = {int(chain.chain_id): chain for chain in document.road_chains()}
    chain = chains.get(chain_id)
    if chain is None:
        return None, "测宽结果对应的道路链不存在，请重新测量。"
    points, cumulative = document._chain_geometry_cache[chain_id]
    total = float(cumulative[-1])
    scale = max(float(getattr(document, "pixel_size", 1.0) or 1.0), 1e-6)
    maximum_pixels = max(2.0, float(max_default_length_units) / scale)
    if total <= maximum_pixels:
        lo, hi = 0.0, total
    else:
        half = maximum_pixels * 0.5
        lo, hi = max(0.0, center - half), min(total, center + half)
        if hi - lo < maximum_pixels:
            if lo <= 0.0:
                hi = min(total, maximum_pixels)
            else:
                lo = max(0.0, total - maximum_pixels)
    start, _ = _point_at(points, cumulative, lo)
    end, _ = _point_at(points, cumulative, hi)
    result = dict(measurement)
    result.update({
        "source": "manual_interval_width",
        "range_start_position": float(lo),
        "range_end_position": float(hi),
        "range_start_row": float(start[0]),
        "range_start_col": float(start[1]),
        "range_end_row": float(end[0]),
        "range_end_col": float(end[1]),
    })
    return result, ""


def manual_width_interval_overlaps(
    measurements: list[dict], candidate: dict, *, exclude_measurement_id: str = "",
) -> bool:
    try:
        chain_id = int(candidate["target_chain_id"])
        lo = float(candidate["range_start_position"])
        hi = float(candidate["range_end_position"])
    except (KeyError, TypeError, ValueError):
        return False
    for row in measurements:
        if str(row.get("source", "")) != "manual_interval_width":
            continue
        if str(row.get("measurement_id", "")) == exclude_measurement_id:
            continue
        try:
            if int(row["target_chain_id"]) != chain_id:
                continue
            other_lo = float(row["range_start_position"])
            other_hi = float(row["range_end_position"])
        except (KeyError, TypeError, ValueError):
            continue
        if max(lo, other_lo) < min(hi, other_hi) - 1e-6:
            return True
    return False


def update_manual_width_interval_endpoint(
    document: "GeometryDocument", measurement: dict, handle: str,
    point_rc: tuple[float, float], *, max_centerline_distance: float = 48.0,
    minimum_range_units: float = 5.0,
) -> tuple[dict | None, str]:
    """Project a dragged range handle onto its existing chain and clamp its order."""
    if handle not in {"start", "end"}:
        return None, "未知的宽度区间控制点。"
    try:
        chain_id = int(measurement["target_chain_id"])
        lo = float(measurement["range_start_position"])
        hi = float(measurement["range_end_position"])
    except (KeyError, TypeError, ValueError):
        return None, "人工宽度记录缺少有效区间。"
    projection = document.project_to_road_chain(
        point_rc, chain_id=chain_id, max_distance=max_centerline_distance,
    )
    if projection is None:
        return None, "控制点只能沿当前道路链移动。"
    scale = max(float(getattr(document, "pixel_size", 1.0) or 1.0), 1e-6)
    minimum_pixels = max(1.0, float(minimum_range_units) / scale)
    position = float(projection.chain_position)
    if handle == "start":
        position = min(position, hi - minimum_pixels)
        if position < 0.0 or hi - position < minimum_pixels - 1e-6:
            return None, "宽度作用区间不能短于最小范围。"
        lo = position
    else:
        position = max(position, lo + minimum_pixels)
        chains = {int(chain.chain_id): chain for chain in document.road_chains()}
        chain = chains.get(chain_id)
        total = float(document._chain_geometry_cache[chain_id][1][-1]) if chain is not None else hi
        if position > total or position - lo < minimum_pixels - 1e-6:
            return None, "宽度作用区间不能短于最小范围。"
        hi = position
    points = chain_interval_points(document, chain_id, lo, hi)
    if len(points) < 2:
        return None, "无法生成有效的宽度作用区间。"
    result = dict(measurement)
    result.update({
        "range_start_position": lo,
        "range_end_position": hi,
        "range_start_row": float(points[0, 0]),
        "range_start_col": float(points[0, 1]),
        "range_end_row": float(points[-1, 0]),
        "range_end_col": float(points[-1, 1]),
    })
    return result, ""


def manual_width_preview_geometry(document: "GeometryDocument", measurement: dict):
    """Build a local pixel-space corridor preview without touching formal surfaces."""
    try:
        points = chain_interval_points(
            document, int(measurement["target_chain_id"]),
            float(measurement["range_start_position"]),
            float(measurement["range_end_position"]),
        )
        scale = max(float(getattr(document, "pixel_size", 1.0) or 1.0), 1e-6)
        width_px = float(measurement.get("width_units", 0.0)) / scale
        if width_px <= 0:
            width_px = float(measurement["width_px"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(points) < 2 or width_px <= 0:
        return None
    line = LineString([(float(col), float(row)) for row, col in points])
    return line.buffer(0.5 * width_px, cap_style=2, join_style=2)


@dataclass
class EditorSnapshot:
    nodes: np.ndarray
    edges: np.ndarray
    mask: np.ndarray
    candidates: dict[str, list[tuple[float, float]]]
    applied_surface_region_ids: set[str]
    manual_widths: list[dict]
    surface_additions: np.ndarray
    surface_removals: np.ndarray


class GeometryDocument:
    def __init__(
        self,
        stem: str,
        image: np.ndarray,
        nodes: np.ndarray,
        edges: np.ndarray,
        mask: np.ndarray,
        candidates: dict[str, list[tuple[float, float]]] | None = None,
        applied_surface_region_ids: set[str] | None = None,
        manual_widths: list[dict] | None = None,
        surface_additions: np.ndarray | None = None,
        surface_removals: np.ndarray | None = None,
        defer_manual_width_normalization: bool = False,
        adopt_large_arrays: bool = False,
    ) -> None:
        self.stem = stem
        self.image = image
        self.nodes = np.asarray(nodes, dtype=np.float32).reshape(-1, 2)
        self.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        self.mask = (
            np.asarray(mask, dtype=np.uint8)
            if adopt_large_arrays else (np.asarray(mask) > 0).astype(np.uint8)
        )
        self.candidates = copy.deepcopy(candidates or {})
        self.applied_surface_region_ids = set(applied_surface_region_ids or set())
        raw_manual_widths = copy.deepcopy(manual_widths or [])
        self.manual_widths = (
            raw_manual_widths if defer_manual_width_normalization
            else normalize_manual_width_measurements(self.nodes, self.edges, raw_manual_widths)
        )
        self.surface_additions = (
            np.asarray(surface_additions, dtype=np.uint8)
            if adopt_large_arrays and surface_additions is not None
            else (np.asarray(surface_additions) > 0).astype(np.uint8)
            if surface_additions is not None else np.zeros_like(self.mask)
        )
        self.surface_removals = (
            np.asarray(surface_removals, dtype=np.uint8)
            if adopt_large_arrays and surface_removals is not None
            else (np.asarray(surface_removals) > 0).astype(np.uint8)
            if surface_removals is not None else np.zeros_like(self.mask)
        )
        self.undo_stack: list[EditorSnapshot] = []
        self.redo_stack: list[EditorSnapshot] = []
        self._cache_revision = 0
        self._chain_cache = None
        self._edge_to_chain: dict[int, int] = {}
        self._chain_geometry_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._edge_grid: dict[tuple[int, int], list[int]] | None = None
        self._incident_edges: list[list[int]] | None = None
        self._topology_metrics_cache: dict | None = None
        self._grid_size = 128.0
        self.cache_build_counts = {"chains": 0, "spatial": 0}

    def invalidate_geometry_cache(self) -> None:
        self._cache_revision += 1
        self._chain_cache = None
        self._edge_to_chain = {}
        self._chain_geometry_cache = {}
        self._edge_grid = None
        self._incident_edges = None
        self._topology_metrics_cache = None

    def road_chains(self):
        if self._chain_cache is None:
            self._chain_cache = build_road_chains(self.nodes, self.edges)
            self._edge_to_chain = {}
            self._chain_geometry_cache = {}
            for chain in self._chain_cache:
                self._chain_geometry_cache[int(chain.chain_id)] = _chain_geometry(chain, self.nodes)
                for edge_id in chain.edge_ids:
                    self._edge_to_chain[int(edge_id)] = int(chain.chain_id)
            self.cache_build_counts["chains"] += 1
        return self._chain_cache

    def incident_edges(self, node_id: int) -> list[int]:
        if self._incident_edges is None:
            self._incident_edges = [[] for _ in range(len(self.nodes))]
            for edge_id, (src, dst) in enumerate(self.edges.tolist()):
                self._incident_edges[int(src)].append(edge_id)
                self._incident_edges[int(dst)].append(edge_id)
        return self._incident_edges[node_id] if 0 <= node_id < len(self._incident_edges) else []

    def _ensure_edge_grid(self) -> None:
        if self._edge_grid is not None:
            return
        grid: dict[tuple[int, int], list[int]] = {}
        size = self._grid_size
        for edge_id, (src, dst) in enumerate(self.edges.tolist()):
            points = self.nodes[[int(src), int(dst)]]
            row_min, col_min = np.min(points, axis=0)
            row_max, col_max = np.max(points, axis=0)
            for grid_row in range(int(math.floor(row_min / size)), int(math.floor(row_max / size)) + 1):
                for grid_col in range(int(math.floor(col_min / size)), int(math.floor(col_max / size)) + 1):
                    grid.setdefault((grid_row, grid_col), []).append(edge_id)
        self._edge_grid = grid
        self.cache_build_counts["spatial"] += 1

    def spatial_edge_candidates(self, row: float, col: float, tolerance: float) -> list[int]:
        self._ensure_edge_grid()
        size = self._grid_size
        row_min, row_max = row - tolerance, row + tolerance
        col_min, col_max = col - tolerance, col + tolerance
        candidates: set[int] = set()
        for grid_row in range(int(math.floor(row_min / size)), int(math.floor(row_max / size)) + 1):
            for grid_col in range(int(math.floor(col_min / size)), int(math.floor(col_max / size)) + 1):
                candidates.update(self._edge_grid.get((grid_row, grid_col), ()))
        return sorted(candidates)

    def project_to_road_chain(
        self,
        point_rc: tuple[float, float] | np.ndarray,
        *,
        chain_id: int | None = None,
        max_distance: float = 24.0,
        tangent_radius: float = 12.0,
    ) -> ChainProjection | None:
        point = np.asarray(point_rc, dtype=np.float32).reshape(2)
        chains = self.road_chains()
        if chain_id is None:
            candidate_ids = self.spatial_edge_candidates(
                float(point[0]), float(point[1]), max(1.0, max_distance),
            )
        else:
            chain = next((item for item in chains if int(item.chain_id) == int(chain_id)), None)
            candidate_ids = [] if chain is None else [int(value) for value in chain.edge_ids]
        best = None
        for edge_id in candidate_ids:
            src, dst = (int(value) for value in self.edges[edge_id])
            projection, distance = point_segment_projection(point, self.nodes[src], self.nodes[dst])
            if best is None or distance < best[2]:
                best = (edge_id, projection, distance)
        if best is None or best[2] > max_distance:
            return None
        edge_id, projection, distance = best
        resolved_chain_id = self._edge_to_chain.get(int(edge_id))
        if resolved_chain_id is None:
            return None
        chain = next(item for item in chains if int(item.chain_id) == resolved_chain_id)
        points, cumulative = self._chain_geometry_cache[resolved_chain_id]
        offset = chain.edge_ids.index(int(edge_id))
        start, end = points[offset], points[offset + 1]
        vector = end - start
        denominator = float(np.dot(vector, vector))
        ratio = 0.0 if denominator <= 1e-9 else float(
            np.clip(np.dot(projection - start, vector) / denominator, 0.0, 1.0)
        )
        position = float(cumulative[offset] + ratio * (cumulative[offset + 1] - cumulative[offset]))
        return ChainProjection(
            chain_id=resolved_chain_id, edge_id=int(edge_id), chain_position=position,
            point=np.asarray(projection, dtype=np.float32), distance=float(distance),
            tangent=_tangent_at(points, cumulative, position, max(1.0, tangent_radius)),
        )

    def snapshot(self) -> EditorSnapshot:
        return EditorSnapshot(
            self.nodes.copy(), self.edges.copy(), self.mask.copy(),
            copy.deepcopy(self.candidates), set(self.applied_surface_region_ids),
            copy.deepcopy(self.manual_widths),
            self.surface_additions.copy(), self.surface_removals.copy(),
        )

    def restore(self, state: EditorSnapshot) -> None:
        self.nodes = state.nodes.copy()
        self.edges = state.edges.copy()
        self.mask = state.mask.copy()
        self.candidates = copy.deepcopy(state.candidates)
        self.applied_surface_region_ids = set(state.applied_surface_region_ids)
        self.manual_widths = copy.deepcopy(state.manual_widths)
        self.surface_additions = state.surface_additions.copy()
        self.surface_removals = state.surface_removals.copy()
        self.invalidate_geometry_cache()

    def editable_surface(self) -> np.ndarray:
        return (((self.mask > 0) | (self.surface_additions > 0)) & (self.surface_removals == 0)).astype(np.uint8)

    def paint_surface(self, row: float, col: float, radius: int, add: bool) -> None:
        point = (int(round(col)), int(round(row)))
        radius = max(1, int(radius))
        if add:
            cv2.circle(self.surface_additions, point, radius, 1, -1)
            cv2.circle(self.surface_removals, point, radius, 0, -1)
        else:
            cv2.circle(self.surface_removals, point, radius, 1, -1)
            cv2.circle(self.surface_additions, point, radius, 0, -1)

    def checkpoint(self) -> None:
        self.undo_stack.append(self.snapshot())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.snapshot())
        self.restore(self.undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.snapshot())
        self.restore(self.redo_stack.pop())
        return True

    def nearest_node(self, row: float, col: float, tolerance: float) -> int | None:
        if len(self.nodes) == 0:
            return None
        distances = np.linalg.norm(self.nodes - np.asarray([row, col], dtype=np.float32), axis=1)
        index = int(np.argmin(distances))
        return index if float(distances[index]) <= tolerance else None

    def nearest_edge(self, row: float, col: float, tolerance: float) -> tuple[int, np.ndarray, float] | None:
        point = np.asarray([row, col], dtype=np.float32)
        best = None
        for edge_id in self.spatial_edge_candidates(row, col, tolerance):
            src, dst = (int(value) for value in self.edges[edge_id])
            projection, distance = point_segment_projection(point, self.nodes[src], self.nodes[dst])
            if distance <= tolerance and (best is None or distance < best[2]):
                best = (edge_id, projection, distance)
        return best

    def add_node(self, point: tuple[float, float]) -> int:
        self.nodes = np.vstack([self.nodes, np.asarray(point, dtype=np.float32)]) if len(self.nodes) else np.asarray([point], dtype=np.float32)
        self.invalidate_geometry_cache()
        return len(self.nodes) - 1

    def split_edge(self, edge_id: int, point: tuple[float, float]) -> int:
        src, dst = (int(value) for value in self.edges[edge_id])
        node_id = self.add_node(point)
        kept = np.delete(self.edges, edge_id, axis=0)
        self.edges = np.vstack([kept, np.asarray([[src, node_id], [node_id, dst]], dtype=np.int32)])
        self.invalidate_geometry_cache()
        return node_id

    def merge_nodes(self, source: int, target: int) -> None:
        if source == target:
            return
        updated = self.edges.copy()
        updated[updated == source] = target
        self.edges = updated
        self.compact()

    def snap_moved_node(self, node_id: int, tolerance: float) -> str:
        """Snap a dragged node to another node or split a nearby non-incident edge."""
        if not 0 <= node_id < len(self.nodes):
            return "none"
        point = self.nodes[node_id].copy()
        if len(self.nodes) > 1:
            distances = np.linalg.norm(self.nodes - point, axis=1)
            distances[node_id] = np.inf
            target = int(np.argmin(distances))
            if float(distances[target]) <= tolerance:
                self.merge_nodes(node_id, target)
                return "node"
        incident = set(self.incident_edges(node_id))
        best = None
        for edge_id in self.spatial_edge_candidates(float(point[0]), float(point[1]), tolerance):
            if edge_id in incident:
                continue
            src, dst = (int(value) for value in self.edges[edge_id])
            projection, distance = point_segment_projection(point, self.nodes[src], self.nodes[dst])
            if distance <= tolerance and (best is None or distance < best[2]):
                best = (edge_id, projection, distance)
        if best is not None:
            split_node = self.split_edge(best[0], (float(best[1][0]), float(best[1][1])))
            # split_edge appends the target, so source is still a valid old index.
            self.merge_nodes(node_id, split_node)
            return "edge"
        return "none"

    def snap_or_add_endpoint(self, point: tuple[float, float], tolerance: float) -> int:
        node_id = self.nearest_node(*point, tolerance)
        if node_id is not None:
            return node_id
        nearest = self.nearest_edge(*point, tolerance)
        if nearest is not None:
            edge_id, projection, _ = nearest
            return self.split_edge(edge_id, (float(projection[0]), float(projection[1])))
        return self.add_node(point)

    def add_polyline(self, points: list[tuple[float, float]], snap_tolerance: float = 8.0) -> bool:
        if len(points) < 2:
            return False
        node_ids = [self.snap_or_add_endpoint(points[0], snap_tolerance)]
        node_ids.extend(self.add_node(point) for point in points[1:-1])
        node_ids.append(self.snap_or_add_endpoint(points[-1], snap_tolerance))
        existing = {tuple(sorted((int(src), int(dst)))) for src, dst in self.edges.tolist()}
        additions = []
        for src, dst in zip(node_ids[:-1], node_ids[1:]):
            key = tuple(sorted((src, dst)))
            if src != dst and key not in existing:
                additions.append((src, dst))
                existing.add(key)
        if additions:
            array = np.asarray(additions, dtype=np.int32)
            self.edges = np.vstack([self.edges, array]) if len(self.edges) else array
            self.invalidate_geometry_cache()
        return bool(additions)

    def add_candidate(self, candidate_id: str, snap_tolerance: float = 8.0) -> bool:
        points = self.candidates.get(candidate_id)
        if not points:
            return False
        changed = self.add_polyline(points, snap_tolerance)
        if changed:
            self.candidates.pop(candidate_id, None)
        return changed

    def delete_edge_at(self, row: float, col: float, tolerance: float) -> bool:
        nearest = self.nearest_edge(row, col, tolerance)
        if nearest is None:
            return False
        self.edges = np.delete(self.edges, nearest[0], axis=0)
        self.compact()
        return True

    def edges_in_polygon(self, points: list[tuple[float, float]]) -> set[int]:
        if len(points) < 3:
            return set()
        polygon = Polygon([(col, row) for row, col in points])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return set()
        selected = set()
        for edge_id, (src, dst) in enumerate(self.edges.tolist()):
            line = LineString([
                (float(self.nodes[src, 1]), float(self.nodes[src, 0])),
                (float(self.nodes[dst, 1]), float(self.nodes[dst, 0])),
            ])
            if polygon.intersects(line):
                selected.add(edge_id)
        return selected

    def delete_edges(self, edge_ids: set[int]) -> int:
        valid = {edge_id for edge_id in edge_ids if 0 <= edge_id < len(self.edges)}
        if not valid:
            return 0
        self.edges = np.asarray(
            [edge for edge_id, edge in enumerate(self.edges.tolist()) if edge_id not in valid],
            dtype=np.int32,
        ).reshape(-1, 2)
        self.compact()
        return len(valid)

    def fill_polygon(self, points: list[tuple[float, float]], value: int) -> bool:
        if len(points) < 3:
            return False
        polygon = np.asarray(
            [[int(round(col)), int(round(row))] for row, col in points], dtype=np.int32
        )
        cv2.fillPoly(self.mask, [polygon], int(bool(value)))
        return True

    def compact(self) -> None:
        if not len(self.edges):
            self.nodes = np.empty((0, 2), dtype=np.float32)
            self.invalidate_geometry_cache()
            return
        unique_edges = []
        seen = set()
        for src, dst in self.edges.tolist():
            key = tuple(sorted((int(src), int(dst))))
            if src != dst and key not in seen:
                unique_edges.append((int(src), int(dst)))
                seen.add(key)
        used = sorted({node for edge in unique_edges for node in edge})
        mapping = {old: new for new, old in enumerate(used)}
        self.nodes = self.nodes[np.asarray(used, dtype=np.int32)]
        self.edges = np.asarray([(mapping[src], mapping[dst]) for src, dst in unique_edges], dtype=np.int32).reshape(-1, 2)
        self.manual_widths = normalize_manual_width_measurements(
            self.nodes, self.edges, self.manual_widths,
        )
        self.invalidate_geometry_cache()

    def node_intersections(self) -> None:
        """Split exact line crossings into shared graph nodes before saving."""
        lines = [
            LineString([
                (float(self.nodes[src, 1]), float(self.nodes[src, 0])),
                (float(self.nodes[dst, 1]), float(self.nodes[dst, 0])),
            ])
            for src, dst in self.edges.tolist()
            if src != dst
        ]
        if not lines:
            self.compact()
            return
        noded = unary_union(lines)
        parts = [noded] if noded.geom_type == "LineString" else list(noded.geoms) if noded.geom_type == "MultiLineString" else []
        nodes: list[tuple[float, float]] = []
        node_lookup: dict[tuple[float, float], int] = {}
        edges: list[tuple[int, int]] = []

        def node_id(row: float, col: float) -> int:
            key = (round(row, 4), round(col, 4))
            if key not in node_lookup:
                node_lookup[key] = len(nodes)
                nodes.append((row, col))
            return node_lookup[key]

        for part in parts:
            previous = None
            for col, row in part.coords:
                current = node_id(float(row), float(col))
                if previous is not None and previous != current:
                    edges.append((previous, current))
                previous = current
        self.nodes = np.asarray(nodes, dtype=np.float32).reshape(-1, 2)
        self.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        self.compact()

    def paint(self, row: float, col: float, radius: int, value: int) -> None:
        cv2.circle(self.mask, (int(round(col)), int(round(row))), max(1, int(radius)), int(bool(value)), -1)

    def topology_metrics(self, short_edge_px: float = 2.0) -> dict:
        if self._topology_metrics_cache is not None:
            return dict(self._topology_metrics_cache)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.nodes)))
        graph.add_edges_from((int(src), int(dst)) for src, dst in self.edges.tolist())
        lengths = [float(np.linalg.norm(self.nodes[dst] - self.nodes[src])) for src, dst in self.edges.tolist()]
        self._topology_metrics_cache = {
            "node_count": len(self.nodes), "edge_count": len(self.edges),
            "component_count": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
            "dangling_endpoint_count": sum(degree == 1 for _, degree in graph.degree()),
            "junction_count": sum(degree >= 3 for _, degree in graph.degree()),
            "isolated_node_count": sum(degree == 0 for _, degree in graph.degree()),
            "short_edge_count": sum(length < short_edge_px for length in lengths),
            "pending_candidate_count": len(self.candidates),
        }
        return dict(self._topology_metrics_cache)


def load_decisions(review_dir: Path) -> dict[tuple[str, str, str], str]:
    return {
        (row.get("stem", ""), row.get("item_type", ""), str(row.get("item_id", ""))): row.get("decision", "")
        for row in read_csv(review_dir / "review_decisions.csv")
    }


def apply_review_state(
    stem: str,
    nodes: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    surface_only: np.ndarray | None,
    candidates: list[dict],
    decisions: dict[tuple[str, str, str], str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[tuple[float, float]]]]:
    # Existing centerlines are now handled only by authoritative geometry
    # editing; legacy quick-review edge decisions must not remove them.
    document = GeometryDocument(stem, np.zeros((*mask.shape, 3), dtype=np.uint8), nodes, edges, mask)

    pending: dict[str, list[tuple[float, float]]] = {}
    candidate_decisions_by_region: dict[str, list[str]] = {}
    for row in candidates:
        candidate_id = str(row.get("candidate_id", ""))
        region_id = str(row.get("region_id", ""))
        decision = decisions.get((stem, "candidate_centerline", candidate_id), "")
        if decision:
            candidate_decisions_by_region.setdefault(region_id, []).append(decision)
        points = candidate_points(row)
        if decision == "accept" or (not decision and row.get("auto_decision") == "accept"):
            document.add_polyline(points)
        elif decision not in {"reject", "mark_nonroad"}:
            pending[candidate_id] = points

    if surface_only is not None:
        _, labels = cv2.connectedComponents((surface_only > 0).astype(np.uint8), connectivity=8)
        for region_id in (str(value) for value in np.unique(labels) if value):
            region_decision = decisions.get((stem, "surface_only_region", region_id), "")
            candidate_decisions = candidate_decisions_by_region.get(region_id, [])
            rejected = region_decision in {"reject", "mark_nonroad"} or (
                candidate_decisions and all(value in {"reject", "mark_nonroad"} for value in candidate_decisions)
            )
            if rejected:
                document.mask[labels == int(region_id)] = 0
    for region_id, polylines in accepted_surface_region_polylines(
        stem, surface_only, candidates, decisions
    ).items():
        for points in polylines:
            document.add_polyline(points)
        document.applied_surface_region_ids.add(region_id)
    document.candidates = pending
    document.compact()
    return document.nodes, document.edges, document.mask, pending


def _line_parts(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def load_manual_widths(edited_dir: Path, stem: str) -> list[dict]:
    path = edited_dir / f"{stem}_manual_widths.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    except (OSError, json.JSONDecodeError):
        return []
    return [dict(row) for row in payload if isinstance(row, dict)]


def load_surface_edits(edited_dir: Path, stem: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    additions = imread_unicode(edited_dir / f"{stem}_manual_surface_add.png", cv2.IMREAD_GRAYSCALE)
    removals = imread_unicode(edited_dir / f"{stem}_manual_surface_remove.png", cv2.IMREAD_GRAYSCALE)
    empty = np.zeros(shape, dtype=np.uint8)
    return (
        (additions > 0).astype(np.uint8) if additions is not None and additions.shape == shape else empty.copy(),
        (removals > 0).astype(np.uint8) if removals is not None and removals.shape == shape else empty.copy(),
    )


def editor_cache_directory(review_dir: Path) -> Path:
    """Return the disposable period-level editor cache beside road outputs."""
    return review_dir.expanduser().resolve().parent / EDITOR_CACHE_DIRECTORY_NAME


def editor_cache_identity(review_dir: Path) -> dict[str, str]:
    resolved = review_dir.expanduser().resolve()
    parts = list(resolved.parts)
    grid = ""
    period = ""
    if "grids" in parts and parts.index("grids") + 1 < len(parts):
        grid = parts[parts.index("grids") + 1]
    if "periods" in parts and parts.index("periods") + 1 < len(parts):
        period = parts[parts.index("periods") + 1]
    return {"grid": grid, "period": period, "review_dir": str(resolved)}


def build_file_fingerprint(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved), "exists": True,
        "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
    }


def build_background_fingerprint(image_paths: list[Path], target_crs) -> dict:
    return {
        "target_crs": _canonical_crs(target_crs),
        "max_dimension": 8192,
        "sources": [build_file_fingerprint(path) for path in image_paths],
    }


def build_surface_fingerprint(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() == ".shp":
        components = [
            build_file_fingerprint(resolved.with_suffix(suffix))
            for suffix in SHAPEFILE_SIDECARS
        ]
    else:
        components = [build_file_fingerprint(resolved)]
    return {"source": str(resolved), "components": components}


def load_cache_metadata(cache_dir: Path) -> dict:
    path = cache_dir / CACHE_METADATA_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("cache_version") != EDITOR_CACHE_VERSION:
        return {}
    return payload


def _canonical_crs(value) -> str:
    return rasterio.crs.CRS.from_user_input(value).to_wkt()


def _grid_metadata(
    width: int, height: int, crs, transform: rasterio.Affine,
    *, count: int, dtype: str,
) -> dict:
    return {
        "width": int(width), "height": int(height),
        "crs": _canonical_crs(crs),
        "transform": [float(value) for value in list(transform)[:6]],
        "count": int(count), "dtype": str(dtype),
    }


def _same_grid(left: dict | None, right: dict | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if (
        int(left.get("width", -1)) != int(right.get("width", -2))
        or int(left.get("height", -1)) != int(right.get("height", -2))
    ):
        return False
    try:
        same_crs = (
            rasterio.crs.CRS.from_user_input(left["crs"])
            == rasterio.crs.CRS.from_user_input(right["crs"])
        )
        return same_crs and bool(np.allclose(
            np.asarray(left["transform"], dtype=float),
            np.asarray(right["transform"], dtype=float),
            rtol=0.0, atol=1e-9,
        ))
    except (KeyError, TypeError, ValueError):
        return False


def _cached_raster_matches(path: Path, grid: dict, *, count: int, dtype: str) -> bool:
    if not path.is_file():
        return False
    try:
        with rasterio.open(path) as dataset:
            actual = _grid_metadata(
                dataset.width, dataset.height, dataset.crs, dataset.transform,
                count=dataset.count, dtype=dataset.dtypes[0],
            )
            if dataset.driver != "GTiff" or dataset.count != count:
                return False
            if any(str(value) != dtype for value in dataset.dtypes):
                return False
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return False
    return _same_grid(actual, grid) and actual["count"] == count and actual["dtype"] == dtype


def background_cache_is_valid(cache_dir: Path, metadata: dict, fingerprint: dict) -> bool:
    entry = metadata.get("background") if isinstance(metadata, dict) else None
    return bool(
        isinstance(entry, dict)
        and entry.get("fingerprint") == fingerprint
        and _cached_raster_matches(
            cache_dir / BACKGROUND_CACHE_NAME, entry.get("grid", {}),
            count=3, dtype="uint8",
        )
    )


def surface_cache_is_valid(
    cache_dir: Path, metadata: dict, fingerprint: dict, background_grid: dict,
) -> bool:
    entry = metadata.get("surface") if isinstance(metadata, dict) else None
    return bool(
        isinstance(entry, dict)
        and entry.get("fingerprint") == fingerprint
        and _same_grid(entry.get("grid"), background_grid)
        and _cached_raster_matches(
            cache_dir / SURFACE_CACHE_NAME, entry.get("grid", {}),
            count=1, dtype="uint8",
        )
    )


def read_background_cache(cache_dir: Path) -> tuple[np.ndarray, rasterio.Affine, object]:
    with rasterio.open(cache_dir / BACKGROUND_CACHE_NAME) as dataset:
        values = dataset.read()
        if values.shape[0] != 3:
            raise ValueError("影像编辑缓存必须包含三个波段")
        return np.moveaxis(values, 0, 2), dataset.transform, dataset.crs


def read_surface_cache(cache_dir: Path) -> np.ndarray:
    with rasterio.open(cache_dir / SURFACE_CACHE_NAME) as dataset:
        return (dataset.read(1) > 0).astype(np.uint8)


def _tiff_block_options(width: int, height: int) -> dict:
    if width < 16 or height < 16:
        return {"tiled": False}
    block_width = min(512, max(16, (int(width) // 16) * 16))
    block_height = min(512, max(16, (int(height) // 16) * 16))
    return {"tiled": True, "blockxsize": block_width, "blockysize": block_height}


def _atomic_raster_write(
    path: Path, values: np.ndarray, crs, transform: rasterio.Affine,
    *, nodata: int = 0, predictor: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.{time.time_ns()}.tmp.tif"
    bands = values[np.newaxis, ...] if values.ndim == 2 else np.moveaxis(values, 2, 0)
    profile = {
        "driver": "GTiff", "width": int(bands.shape[2]), "height": int(bands.shape[1]),
        "count": int(bands.shape[0]), "dtype": str(bands.dtype),
        "crs": crs, "transform": transform, "nodata": nodata,
        "compress": "DEFLATE", "predictor": predictor, "zlevel": 1,
        "BIGTIFF": "IF_SAFER", **_tiff_block_options(bands.shape[2], bands.shape[1]),
    }
    try:
        with rasterio.open(temporary, "w", **profile) as dataset:
            dataset.write(bands)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_background_cache(
    cache_dir: Path, image: np.ndarray, crs, transform: rasterio.Affine,
) -> None:
    _atomic_raster_write(
        cache_dir / BACKGROUND_CACHE_NAME, image.astype(np.uint8, copy=False),
        crs, transform, predictor=2,
    )


def write_surface_cache(
    cache_dir: Path, mask: np.ndarray, crs, transform: rasterio.Affine,
) -> None:
    _atomic_raster_write(
        cache_dir / SURFACE_CACHE_NAME, (mask > 0).astype(np.uint8, copy=False),
        crs, transform,
    )


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _update_cache_metadata(
    cache_dir: Path, section: str, entry: dict, *, background_grid: dict | None = None,
    identity: dict[str, str] | None = None,
) -> None:
    metadata = load_cache_metadata(cache_dir)
    if not metadata:
        metadata = {"cache_version": EDITOR_CACHE_VERSION}
    metadata["cache_version"] = EDITOR_CACHE_VERSION
    if identity:
        metadata.update(identity)
    metadata[section] = entry
    if section == "background" and not _same_grid(
        (metadata.get("surface") or {}).get("grid"), background_grid,
    ):
        metadata.pop("surface", None)
    _atomic_json_write(cache_dir / CACHE_METADATA_NAME, metadata)


@contextmanager
def editor_cache_write_lock(cache_dir: Path, timeout_seconds: float = 5.0):
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / CACHE_LOCK_NAME
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    acquired = False
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(descriptor, f"{os.getpid()} {time.time_ns()}".encode("ascii"))
            finally:
                os.close(descriptor)
            acquired = True
            break
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 300.0
                if stale:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        except OSError:
            break
    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


def clear_editor_cache(review_dir: Path) -> bool:
    cache_dir = editor_cache_directory(review_dir)
    if cache_dir.name != EDITOR_CACHE_DIRECTORY_NAME:
        raise ValueError("编辑缓存目录不安全")
    if not cache_dir.exists():
        return False
    shutil.rmtree(cache_dir)
    return True


def editor_cache_sizes(cache_dir: Path) -> dict[str, int]:
    sizes = {}
    for name in (BACKGROUND_CACHE_NAME, SURFACE_CACHE_NAME, CACHE_METADATA_NAME):
        try:
            sizes[name] = int((cache_dir / name).stat().st_size)
        except OSError:
            sizes[name] = 0
    sizes["total"] = sum(sizes.values())
    return sizes


def rasterize_final_surface(final_surfaces: gpd.GeoDataFrame, image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"原始影像缺少 CRS：{image_path}")
        frame = final_surfaces if final_surfaces.crs == dataset.crs else final_surfaces.to_crs(dataset.crs)
        footprint = box(dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top)
        geometries = [
            geometry.intersection(footprint) for geometry in frame.geometry
            if geometry is not None and not geometry.is_empty and geometry.intersects(footprint)
        ]
        if not geometries:
            return np.zeros((dataset.height, dataset.width), dtype=np.uint8)
        return rasterize(
            [(geometry, 1) for geometry in geometries], out_shape=(dataset.height, dataset.width),
            transform=dataset.transform, fill=0, all_touched=True, dtype="uint8",
        )


def _review_summary_rows(review_dir: Path) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for summary_path in sorted(review_dir.glob("*_summary.json")):
        if summary_path.name.startswith("batch_") or summary_path.name.endswith("_optimized_summary.json"):
            continue
        rows.append((
            summary_path.name.removesuffix("_summary.json"),
            json.loads(summary_path.read_text(encoding="utf-8")),
        ))
    return rows


def _frame_bounds(*frames: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    bounds = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        values = np.asarray(frame.total_bounds, dtype=float)
        if values.shape == (4,) and np.all(np.isfinite(values)):
            bounds.append(values)
    if not bounds:
        raise ValueError("最终中心线和道路面均为空，无法建立全局编辑视图")
    values = np.vstack(bounds)
    minx, miny = np.min(values[:, :2], axis=0)
    maxx, maxy = np.max(values[:, 2:], axis=0)
    span = max(maxx - minx, maxy - miny, 1.0)
    padding = span * 0.025
    return minx - padding, miny - padding, maxx + padding, maxy + padding


def _image_union_bounds(image_paths: list[Path], target_crs) -> tuple[float, float, float, float]:
    transformed = []
    for image_path in image_paths:
        with rasterio.open(image_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"影像缺少 CRS：{image_path}")
            transformed.append(transform_bounds(
                dataset.crs, target_crs,
                dataset.bounds.left, dataset.bounds.bottom,
                dataset.bounds.right, dataset.bounds.top,
                densify_pts=21,
            ))
    values = np.asarray(transformed, dtype=float)
    return (
        float(np.min(values[:, 0])), float(np.min(values[:, 1])),
        float(np.max(values[:, 2])), float(np.max(values[:, 3])),
    )


def _to_display_uint8(mosaic: np.ndarray, valid: np.ndarray) -> np.ndarray:
    bands = np.asarray(mosaic)
    if bands.ndim == 2:
        bands = bands[np.newaxis, ...]
    if bands.shape[0] == 1:
        bands = np.repeat(bands, 3, axis=0)
    elif bands.shape[0] == 2:
        bands = np.concatenate((bands, bands[-1:]), axis=0)
    else:
        bands = bands[:3]
    rgb = np.zeros(bands.shape, dtype=np.uint8)
    for index, band in enumerate(bands):
        values = np.asarray(band, dtype=np.float32)
        samples = values[valid & np.isfinite(values)]
        if samples.size == 0:
            continue
        low, high = np.percentile(samples, (2.0, 98.0))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(samples)), float(np.max(samples))
        if high <= low:
            rgb[index][valid] = np.clip(samples[0], 0, 255).astype(np.uint8)
        else:
            rgb[index] = np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            rgb[index][~valid] = 0
    return np.moveaxis(rgb, 0, 2)[:, :, ::-1].copy()


def _build_global_overview(
    image_paths: list[Path], target_crs, bounds: tuple[float, float, float, float],
    max_dimension: int = 8192,
) -> tuple[np.ndarray, rasterio.Affine]:
    if not image_paths:
        raise FileNotFoundError("复核目录中没有可用于全局编辑背景的影像")
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in image_paths]
        if any(source.crs is None for source in sources):
            missing = next(path for path, source in zip(image_paths, sources) if source.crs is None)
            raise ValueError(f"影像缺少 CRS：{missing}")
        virtual = [stack.enter_context(WarpedVRT(source, crs=target_crs)) for source in sources]
        xres = min(abs(float(source.res[0])) for source in virtual)
        yres = min(abs(float(source.res[1])) for source in virtual)
        width = max(1.0, (bounds[2] - bounds[0]) / xres)
        height = max(1.0, (bounds[3] - bounds[1]) / yres)
        scale = max(1.0, width / max_dimension, height / max_dimension)
        xres, yres = xres * scale, yres * scale
        band_count = min(3, min(source.count for source in virtual))
        indexes = list(range(1, band_count + 1))
        mosaic, transform = merge(
            virtual, bounds=bounds, res=(xres, yres), indexes=indexes,
            masked=True, method="first",
        )
        if np.ma.isMaskedArray(mosaic):
            mask = np.ma.getmaskarray(mosaic)
            valid = ~np.all(mask, axis=0)
            values = mosaic.filled(0)
        else:
            values = np.asarray(mosaic)
            valid = np.any(np.isfinite(values) & (values != 0), axis=0)
    return _to_display_uint8(values, valid), transform


def _world_lines_from_document(document: GeometryDocument) -> list[LineString]:
    transform = getattr(document, "global_transform", None)
    if transform is None:
        raise ValueError("全局编辑文档缺少地图变换参数")
    lines: list[LineString] = []
    for src, dst in document.edges.tolist():
        start_row, start_col = document.nodes[src]
        end_row, end_col = document.nodes[dst]
        start = transform * (float(start_col), float(start_row))
        end = transform * (float(end_col), float(end_row))
        if start != end:
            lines.append(LineString((start, end)))
    return lines


def _load_valid_background_cache(
    cache_dir: Path, metadata: dict, fingerprint: dict, progress=None,
) -> tuple[np.ndarray, rasterio.Affine, object] | None:
    if not background_cache_is_valid(cache_dir, metadata, fingerprint):
        return None
    if progress:
        progress("正在读取影像缓存…")
    try:
        return read_background_cache(cache_dir)
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return None


def _load_or_build_background_cache(
    review_dir: Path, image_paths: list[Path], target_crs,
    progress=None, timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, rasterio.Affine, object, tuple[float, float, float, float], dict, bool]:
    timings = timings if timings is not None else {}
    cache_dir = editor_cache_directory(review_dir)
    fingerprint = build_background_fingerprint(image_paths, target_crs)
    metadata = load_cache_metadata(cache_dir)
    cache_read_started = time.perf_counter()
    cached = _load_valid_background_cache(cache_dir, metadata, fingerprint, progress)
    if cached is not None:
        image, transform, crs = cached
        timings["background_cache_load"] = time.perf_counter() - cache_read_started
        timings["background_cache_used"] = 1.0
        bounds = tuple(float(value) for value in array_bounds(image.shape[0], image.shape[1], transform))
        grid = _grid_metadata(image.shape[1], image.shape[0], crs, transform, count=3, dtype="uint8")
        return image, transform, crs, bounds, grid, True

    if progress:
        period = editor_cache_identity(review_dir).get("period", "")
        period_label = f"{period} 期" if period else "当前期"
        progress(f"未找到可用缓存，正在准备 {period_label}编辑数据…")
    with editor_cache_write_lock(cache_dir) as may_write:
        # Another instance may have completed while this worker waited.
        metadata = load_cache_metadata(cache_dir)
        cache_read_started = time.perf_counter()
        cached = _load_valid_background_cache(cache_dir, metadata, fingerprint, progress)
        if cached is not None:
            image, transform, crs = cached
            timings["background_cache_load"] = time.perf_counter() - cache_read_started
            timings["background_cache_used"] = 1.0
            bounds = tuple(float(value) for value in array_bounds(image.shape[0], image.shape[1], transform))
            grid = _grid_metadata(image.shape[1], image.shape[0], crs, transform, count=3, dtype="uint8")
            return image, transform, crs, bounds, grid, True
        if progress:
            progress("正在拼接遥感影像…")
        stage_started = time.perf_counter()
        bounds = _image_union_bounds(image_paths, target_crs)
        timings["raster_metadata_read"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        image, transform = _build_global_overview(image_paths, target_crs, bounds)
        timings["global_image_mosaic"] = time.perf_counter() - stage_started
        bounds = tuple(float(value) for value in array_bounds(
            image.shape[0], image.shape[1], transform,
        ))
        crs = rasterio.crs.CRS.from_user_input(target_crs)
        grid = _grid_metadata(image.shape[1], image.shape[0], crs, transform, count=3, dtype="uint8")
        if may_write:
            if progress:
                progress("正在保存影像编辑缓存…")
            stage_started = time.perf_counter()
            try:
                write_background_cache(cache_dir, image, crs, transform)
                _update_cache_metadata(
                    cache_dir, "background", {
                        "fingerprint": fingerprint, "grid": grid,
                        "file": BACKGROUND_CACHE_NAME,
                        "file_size": int((cache_dir / BACKGROUND_CACHE_NAME).stat().st_size),
                    },
                    background_grid=grid, identity=editor_cache_identity(review_dir),
                )
                timings["background_cache_write"] = time.perf_counter() - stage_started
            except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
                timings["background_cache_write_failed"] = 1.0
                print(f"Unable to persist background editor cache: {exc}", file=sys.stderr, flush=True)
        else:
            timings["background_cache_write_skipped_locked"] = 1.0
    timings["background_cache_used"] = 0.0
    return image, transform, crs, bounds, grid, False


def _load_valid_surface_cache(
    cache_dir: Path, metadata: dict, fingerprint: dict, background_grid: dict,
    progress=None,
) -> np.ndarray | None:
    if not surface_cache_is_valid(cache_dir, metadata, fingerprint, background_grid):
        return None
    if progress:
        progress("正在读取道路面缓存…")
    try:
        return read_surface_cache(cache_dir)
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return None


def _load_or_build_surface_cache(
    review_dir: Path, final_surfaces: Path, image_shape: tuple[int, int],
    crs, transform: rasterio.Affine, background_grid: dict,
    *, background_was_cached: bool, progress=None,
    timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, gpd.GeoDataFrame | None, bool]:
    timings = timings if timings is not None else {}
    cache_dir = editor_cache_directory(review_dir)
    fingerprint = build_surface_fingerprint(final_surfaces)
    metadata = load_cache_metadata(cache_dir)
    cache_read_started = time.perf_counter()
    cached = _load_valid_surface_cache(
        cache_dir, metadata, fingerprint, background_grid, progress,
    )
    if cached is not None:
        mask = cached
        timings["surface_cache_load"] = time.perf_counter() - cache_read_started
        timings["surface_cache_used"] = 1.0
        return mask, None, True

    if progress:
        progress(
            "影像缓存有效。检测到道路面成果已更新，正在重新生成道路面缓存…"
            if background_was_cached else "正在生成道路面缓存…"
        )
    with editor_cache_write_lock(cache_dir) as may_write:
        metadata = load_cache_metadata(cache_dir)
        cache_read_started = time.perf_counter()
        cached = _load_valid_surface_cache(
            cache_dir, metadata, fingerprint, background_grid, progress,
        )
        if cached is not None:
            mask = cached
            timings["surface_cache_load"] = time.perf_counter() - cache_read_started
            timings["surface_cache_used"] = 1.0
            return mask, None, True
        stage_started = time.perf_counter()
        surfaces = gpd.read_file(final_surfaces)
        timings["surface_shp_read"] = time.perf_counter() - stage_started
        if surfaces.crs is None:
            raise ValueError(f"最终道路面缺少 CRS：{final_surfaces}")
        surfaces = surfaces if _canonical_crs(surfaces.crs) == _canonical_crs(crs) else surfaces.to_crs(crs)
        surface_shapes = [
            (geometry, 1) for geometry in surfaces.geometry
            if geometry is not None and not geometry.is_empty
        ]
        stage_started = time.perf_counter()
        mask = rasterize(
            surface_shapes, out_shape=image_shape, transform=transform,
            fill=0, all_touched=True, dtype="uint8",
        ) if surface_shapes else np.zeros(image_shape, dtype=np.uint8)
        timings["surface_rasterize"] = time.perf_counter() - stage_started
        if may_write:
            stage_started = time.perf_counter()
            try:
                write_surface_cache(cache_dir, mask, crs, transform)
                surface_grid = dict(background_grid, count=1, dtype="uint8")
                _update_cache_metadata(
                    cache_dir, "surface", {
                        "fingerprint": fingerprint, "grid": surface_grid,
                        "file": SURFACE_CACHE_NAME,
                        "file_size": int((cache_dir / SURFACE_CACHE_NAME).stat().st_size),
                    }, identity=editor_cache_identity(review_dir),
                )
                timings["surface_cache_write"] = time.perf_counter() - stage_started
            except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
                timings["surface_cache_write_failed"] = 1.0
                print(f"Unable to persist surface editor cache: {exc}", file=sys.stderr, flush=True)
        else:
            timings["surface_cache_write_skipped_locked"] = 1.0
    timings["surface_cache_used"] = 0.0
    return mask, surfaces, False


def _final_centerline_documents(
    review_dir: Path, edited_dir: Path, final_centerlines: Path, final_surfaces: Path,
    progress=None, timings: dict[str, float] | None = None,
) -> list[GeometryDocument]:
    """Create one global editor document from the authoritative period product."""
    timings = timings if timings is not None else {}
    stage_started = time.perf_counter()
    if progress:
        progress("正在加载道路中心线…")
    lines = gpd.read_file(final_centerlines)
    timings["centerline_shp_read"] = time.perf_counter() - stage_started
    if lines.crs is None:
        raise ValueError(f"最终中心线缺少 CRS：{final_centerlines}")
    manifest_path = edited_dir / "edited_manifest.json"
    try:
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        saved_manifest = {}
    source_matches = (
        str(saved_manifest.get("base_final_centerlines", "")) == str(final_centerlines.resolve())
        and int(saved_manifest.get("base_final_centerlines_mtime_ns", -1)) == final_centerlines.stat().st_mtime_ns
        and str(saved_manifest.get("base_final_surfaces", "")) == str(final_surfaces.resolve())
        and int(saved_manifest.get("base_final_surfaces_mtime_ns", -1)) == final_surfaces.stat().st_mtime_ns
    )
    stage_started = time.perf_counter()
    summary_rows = _review_summary_rows(review_dir)
    timings["summary_json_read"] = time.perf_counter() - stage_started
    if not summary_rows:
        raise FileNotFoundError(f"No review summaries under {review_dir}")
    edited_global_path = edited_dir / "global_edited_centerlines.gpkg"
    active_lines = lines
    if source_matches and edited_global_path.is_file():
        active_lines = gpd.read_file(edited_global_path, layer="edited_centerlines")
        if active_lines.crs is None:
            raise ValueError(f"已保存的全局中心线缺少 CRS：{edited_global_path}")
        active_lines = active_lines if active_lines.crs == lines.crs else active_lines.to_crs(lines.crs)
    image_paths = [Path(summary["image"]).expanduser().resolve() for _, summary in summary_rows]
    if progress:
        progress("正在检查本地编辑缓存…")
    image, transform, cache_crs, bounds, background_grid, background_cached = (
        _load_or_build_background_cache(
            review_dir, image_paths, lines.crs, progress=progress, timings=timings,
        )
    )
    mask, surfaces, surface_cached = _load_or_build_surface_cache(
        review_dir, final_surfaces, image.shape[:2], cache_crs, transform,
        background_grid, background_was_cached=background_cached,
        progress=progress, timings=timings,
    )
    if progress:
        if background_cached and surface_cached:
            progress("已使用本地编辑缓存。")
        else:
            progress("编辑缓存已建立，之后再次打开该期次将直接读取缓存。")
    cache_sizes = editor_cache_sizes(editor_cache_directory(review_dir))
    print(
        "GEOMETRY_EDITOR_CACHE=" + json.dumps({
            "directory": str(editor_cache_directory(review_dir)),
            "background_used": background_cached, "surface_used": surface_cached,
            "background_bytes": cache_sizes[BACKGROUND_CACHE_NAME],
            "surface_bytes": cache_sizes[SURFACE_CACHE_NAME],
            "total_bytes": cache_sizes["total"],
        }, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    stage_started = time.perf_counter()
    nodes, edges = _graph_from_world_lines(active_lines, transform, box(*bounds))
    timings["centerline_graph_build"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    surface_additions, surface_removals = load_surface_edits(edited_dir, "global", image.shape[:2])
    document = GeometryDocument(
        "全局最终中心线", image, nodes, edges, mask, candidates={},
        manual_widths=load_manual_widths(edited_dir, "global"),
        surface_additions=surface_additions, surface_removals=surface_removals,
        defer_manual_width_normalization=True, adopt_large_arrays=True,
    )
    document.global_mode = True
    document.global_transform = transform
    document.global_crs = cache_crs
    document.global_bounds = bounds
    document.global_summary_rows = summary_rows
    document.global_base_lines = lines.copy()
    document.global_base_surfaces = surfaces.copy() if surfaces is not None else None
    document.global_overview_scale = max(abs(float(transform.a)), abs(float(transform.e)))
    document.pixel_size = document.global_overview_scale
    timings["document_construct"] = time.perf_counter() - stage_started
    return [document]


def load_documents(
    review_dir: Path, edited_dir: Path, final_centerlines: Path | None = None,
    final_surfaces: Path | None = None, progress=None,
    timings: dict[str, float] | None = None,
) -> list[GeometryDocument]:
    load_started = time.perf_counter()
    if progress:
        progress("正在读取人工编辑资料…")
    if final_centerlines is not None:
        final_centerlines = final_centerlines.expanduser().resolve()
        if not final_centerlines.is_file():
            raise FileNotFoundError(f"找不到每期最终融合中心线：{final_centerlines}")
        if final_surfaces is None or not final_surfaces.is_file():
            raise FileNotFoundError(f"找不到每期最终道路面：{final_surfaces}")
        documents = _final_centerline_documents(
            review_dir, edited_dir, final_centerlines, final_surfaces,
            progress=progress, timings=timings,
        )
        if timings is not None:
            timings["load_documents_total"] = time.perf_counter() - load_started
        return documents
    decisions = load_decisions(review_dir)
    manifest_path = edited_dir / "edited_manifest.json"
    try:
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        saved_manifest = {}
    documents = []
    for summary_path in sorted(review_dir.glob("*_summary.json")):
        if summary_path.name.startswith("batch_") or summary_path.name.endswith("_optimized_summary.json"):
            continue
        stem = summary_path.name.removesuffix("_summary.json")
        if progress:
            progress(f"正在读取切片 {stem}…")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        image = imread_unicode(Path(summary["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image for {stem}: {summary.get('image')}")
        edited_graph = edited_dir / f"{stem}_edited_graph.p"
        if current_edited_graph(summary, edited_graph):
            nodes, edges = load_graph(edited_graph)
            mask = imread_unicode(review_dir / f"{stem}_molra_clean_mask.png", cv2.IMREAD_GRAYSCALE)
            rows = read_csv(review_dir / f"{stem}_candidate_centerlines.csv")
            # Quick-review decisions may have changed after the last editor
            # save. Merge accepted candidates into that saved graph without
            # duplicating geometry already imported from QGIS or the editor.
            document = GeometryDocument(stem, image, nodes, edges, mask)
            apply_accepted_candidates(stem, document, rows, decisions)
            nodes, edges = document.nodes, document.edges
            saved_ids = saved_manifest.get("tiles", {}).get(stem, {}).get("pending_candidate_ids")
            if saved_ids is None:
                saved_ids = [
                    str(row.get("candidate_id", "")) for row in rows
                    if not decisions.get((stem, "candidate_centerline", str(row.get("candidate_id", ""))))
                    and row.get("auto_decision") != "accept"
                ]
            saved_ids = {str(value) for value in saved_ids}
            pending = {
                str(row.get("candidate_id", "")): candidate_points(row)
                for row in rows
                if str(row.get("candidate_id", "")) in saved_ids
                and not candidate_is_accepted(stem, row, decisions)
                and decisions.get((stem, "candidate_centerline", str(row.get("candidate_id", ""))), "")
                not in {"reject", "delete_centerline", "mark_nonroad"}
            }
            applied_regions = {
                str(value)
                for value in saved_manifest.get("tiles", {}).get(stem, {}).get("applied_surface_region_ids", [])
            }
            document = GeometryDocument(
                stem, image, nodes, edges, mask, pending, applied_regions,
                load_manual_widths(edited_dir, stem),
            )
            surface_only = imread_unicode(review_dir / f"{stem}_surface_only.png", cv2.IMREAD_GRAYSCALE)
            for region_id, polylines in accepted_surface_region_polylines(
                stem, surface_only, rows, decisions
            ).items():
                if region_id in document.applied_surface_region_ids:
                    continue
                for points in polylines:
                    document.add_polyline(points)
                document.applied_surface_region_ids.add(region_id)
        else:
            if edited_graph.is_file():
                print(f"Ignoring stale edited graph for {stem}: {edited_graph}")
            nodes, edges = load_graph(Path(summary.get("prepared_graph", summary["graph"])))
            mask = imread_unicode(review_dir / f"{stem}_molra_clean_mask.png", cv2.IMREAD_GRAYSCALE)
            surface_only = imread_unicode(review_dir / f"{stem}_surface_only.png", cv2.IMREAD_GRAYSCALE)
            candidates = read_csv(review_dir / f"{stem}_candidate_centerlines.csv")
            nodes, edges, mask, pending = apply_review_state(stem, nodes, edges, mask, surface_only, candidates, decisions)
            document = GeometryDocument(
                stem, image, nodes, edges, mask, pending,
                manual_widths=load_manual_widths(edited_dir, stem),
            )
            for region_id in accepted_surface_region_polylines(stem, surface_only, candidates, decisions):
                document.applied_surface_region_ids.add(region_id)
        try:
            with rasterio.open(Path(summary["image"])) as dataset:
                document.raster_transform = dataset.transform
                document.pixel_size = max(abs(float(dataset.transform.a)), abs(float(dataset.transform.e)))
        except (OSError, rasterio.errors.RasterioError, KeyError):
            document.pixel_size = float(summary.get("pixel_size", 1.0) or 1.0)
        documents.append(document)
    if not documents:
        raise FileNotFoundError(f"No review summaries under {review_dir}")
    if timings is not None:
        timings["load_documents_total"] = time.perf_counter() - load_started
    return documents


def _mask_world_bounds(mask: np.ndarray, transform: rasterio.Affine):
    rows, cols = np.nonzero(mask > 0)
    if rows.size == 0:
        return None
    left, top = transform * (float(cols.min()), float(rows.min()))
    right, bottom = transform * (float(cols.max() + 1), float(rows.max() + 1))
    return box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def _global_change_geometry(document: GeometryDocument, edited_lines: list[LineString]):
    base_parts = [
        part for geometry in document.global_base_lines.geometry
        for part in _line_parts(geometry)
    ]
    base_geometry = unary_union(base_parts) if base_parts else None
    edited_geometry = unary_union(edited_lines) if edited_lines else None
    if base_geometry is None:
        changed = edited_geometry
    elif edited_geometry is None:
        changed = base_geometry
    else:
        tolerance = max(float(document.global_overview_scale) * 0.1, 0.01)
        # The editor round-trips map coordinates through overview pixels.  A
        # tolerant two-sided difference ignores that numerical jitter while
        # retaining genuinely added, removed, or moved linework.
        added = edited_geometry.difference(base_geometry.buffer(tolerance))
        removed = base_geometry.difference(edited_geometry.buffer(tolerance))
        centerline_changes = [
            geometry for geometry in (added, removed)
            if geometry is not None and not geometry.is_empty
        ]
        changed = unary_union(centerline_changes) if centerline_changes else None
    additions = _mask_world_bounds(document.surface_additions, document.global_transform)
    removals = _mask_world_bounds(document.surface_removals, document.global_transform)
    parts = [item for item in (changed, additions, removals) if item is not None and not item.is_empty]
    for measurement in document.manual_widths:
        try:
            x, y = document.global_transform * (
                float(measurement["target_col"]), float(measurement["target_row"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        parts.append(Point(x, y).buffer(max(document.global_overview_scale * 3.0, 0.01)))
        if str(measurement.get("source", "")) == "manual_interval_width":
            try:
                interval = LineString([
                    document.global_transform * (
                        float(measurement["range_start_col"]), float(measurement["range_start_row"]),
                    ),
                    document.global_transform * (
                        float(measurement["range_end_col"]), float(measurement["range_end_row"]),
                    ),
                ])
            except (KeyError, TypeError, ValueError):
                continue
            parts.append(interval.buffer(max(document.global_overview_scale * 3.0, 0.01)))
    if not parts:
        return None
    return unary_union(parts).buffer(max(document.global_overview_scale * 2.0, 0.01))


def save_global_document(
    document: GeometryDocument, edited_dir: Path,
    final_centerlines: Path, final_surfaces: Path,
) -> dict:
    """Persist the authoritative global edit without materializing tile inputs."""
    document.compact()
    edited_lines = _world_lines_from_document(document)
    if not edited_lines:
        raise ValueError("全局最终中心线已为空；请至少保留一条道路后再保存")
    global_path = (edited_dir / "global_edited_centerlines.gpkg").resolve()
    if global_path.exists():
        global_path.unlink()
    edited_frame = gpd.GeoDataFrame(
        {"edge_id": list(range(len(edited_lines))), "geometry": edited_lines},
        geometry="geometry", crs=document.global_crs,
    )
    edited_frame.to_file(global_path, layer="edited_centerlines", driver="GPKG")
    global_width_path = (edited_dir / "global_manual_widths.json").resolve()
    global_width_path.write_text(
        json.dumps(document.manual_widths, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    global_add_path = (edited_dir / "global_manual_surface_add.png").resolve()
    global_remove_path = (edited_dir / "global_manual_surface_remove.png").resolve()
    cv2.imencode(".png", document.surface_additions.astype(np.uint8) * 255)[1].tofile(global_add_path)
    cv2.imencode(".png", document.surface_removals.astype(np.uint8) * 255)[1].tofile(global_remove_path)

    changed = _global_change_geometry(document, edited_lines)
    affected_tiles: list[str] = []
    tile_manifest = {}
    for stem, summary in document.global_summary_rows:
        image_path = Path(summary["image"]).expanduser().resolve()
        with rasterio.open(image_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"影像缺少 CRS：{image_path}")
            raster_crs = dataset.crs
            raster_bounds = dataset.bounds
        affected = False
        if changed is not None and not changed.is_empty:
            left, bottom, right, top = transform_bounds(
                raster_crs, document.global_crs,
                raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top,
                densify_pts=21,
            )
            affected = changed.intersects(box(left, bottom, right, top))
        if affected:
            affected_tiles.append(stem)
        tile_manifest[stem] = {
            "image": str(image_path),
            "affected": bool(affected),
            "tile_inputs_materialized": False,
        }
    report = {
        "editing_scope": "period_final_fused_centerlines_global_once",
        "global_centerlines": str(global_path),
        "global_edge_count": len(edited_lines),
        "affected_tiles": sorted(affected_tiles),
        "affected_tile_count": len(affected_tiles),
        "tile_count": len(tile_manifest),
        "overview_pixel_size": float(document.global_overview_scale),
        "global_transform": list(document.global_transform)[:6],
        "global_crs": document.global_crs.to_wkt(),
        "global_raster_shape": list(document.surface_additions.shape),
    }
    (edited_dir / "global_edit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return {
        "editor": "builtin_geometry_editor_v3_global_final",
        "base_final_centerlines": str(final_centerlines),
        "base_final_centerlines_mtime_ns": final_centerlines.stat().st_mtime_ns,
        "base_final_surfaces": str(final_surfaces),
        "base_final_surfaces_mtime_ns": final_surfaces.stat().st_mtime_ns,
        **report,
        "tiles": tile_manifest,
        "global_manual_widths": str(global_width_path),
        "global_manual_surface_add": str(global_add_path),
        "global_manual_surface_remove": str(global_remove_path),
    }


class GeometryEditorApp:
    MODES = {
        "select": "V 选择/编辑",
        "draw": "L 绘制中心线",
        "measure_width": "W 人工测宽",
        "surface_add": "B 路面高级编辑",
    }
    PRIMARY_MODES = ("select", "draw", "measure_width")

    def __init__(
        self, root: tk.Tk, review_dir: Path, edited_dir: Path,
        final_centerlines: Path | None = None,
        final_surfaces: Path | None = None,
        documents: list[GeometryDocument] | None = None,
        startup_timings: dict[str, float] | None = None,
    ) -> None:
        self.root = root
        self.review_dir = review_dir
        self.edited_dir = edited_dir
        self.final_centerlines = final_centerlines.expanduser().resolve() if final_centerlines is not None else None
        self.final_surfaces = final_surfaces.expanduser().resolve() if final_surfaces is not None else None
        self.documents = documents if documents is not None else load_documents(
            review_dir, edited_dir, self.final_centerlines, self.final_surfaces,
        )
        if not self.documents:
            raise ValueError("人工编辑资料中没有可加载的道路文档。")
        self.document_index = 0
        self.mode = tk.StringVar(value="select")
        self.tile_var = tk.StringVar(value=self.documents[0].stem)
        self.status_var = tk.StringVar()
        self.zoom = 0.5
        self.photo = None
        self.background_item = None
        self.photo_cache: OrderedDict[tuple, ImageTk.PhotoImage] = OrderedDict()
        self.background_source_cache: dict[int, tuple[int, np.ndarray]] = {}
        self.surface_versions = [0 for _ in self.documents]
        self.min_zoom = 0.02
        self.max_zoom = 64.0
        self._background_refresh_after_id: str | None = None
        self._refreshing_background = False
        self._last_background_view: tuple | None = None
        self._last_scrollbar_views: dict[str, tuple[float, float] | None] = {
            "x": None, "y": None,
        }
        self._background_scrollregion: tuple[float, float, float, float] | None = None
        self.edge_items: dict[int, int] = {}
        self.node_items: dict[int, int] = {}
        self.width_preview: dict | None = None
        self.last_metrics: dict = {}
        self.mouse_status = (0.0, 0.0)
        self.geometry_dirty = True
        self.surface_dirty = True
        self.topology_dirty = True
        self.drag_node: int | None = None
        self.drag_checkpoint = False
        self.draft: list[tuple[float, float]] = []
        self.lasso: list[tuple[float, float]] = []
        self.lasso_active = False
        self.selected_edge_ids: set[int] = set()
        self.width_draft: list[tuple[float, float]] = []
        self.width_drag_start: tuple[float, float] | None = None
        self.width_drag_current: tuple[float, float] | None = None
        self.interval_draft: list[tuple[float, float]] = []
        self.interval_measurement_id: str | None = None
        self.active_width_measurement_id: str | None = None
        self.width_range_drag_handle: str | None = None
        self.width_range_drag_checkpoint = False
        self.surface_radius = tk.IntVar(value=12)
        self.surface_action = tk.StringVar(value="add")
        self.surface_stroke_active = False
        self.surface_right_active = False
        self.surface_stroke_points: list[tuple[float, float]] = []
        self.surface_preview_item = None
        self.space_pressed = False
        self.space_panning = False
        self.lasso_canvas_item = None

        root.title("道路实体变化智能检测与人工核验 · 中心线编辑")
        self.display_scale = max(1.0, min(float(root.winfo_fpixels("1i")) / 96.0, 2.5))
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        width = min(round(1500 * self.display_scale), int(screen_w * 0.94))
        height = min(round(950 * self.display_scale), int(screen_h * 0.90))
        min_width = min(round(1050 * self.display_scale), int(screen_w * 0.88))
        min_height = min(round(680 * self.display_scale), int(screen_h * 0.82))
        root.minsize(max(900, min_width), max(580, min_height))
        root.geometry(f"{width}x{height}+{max(0, (screen_w - width) // 2)}+{max(0, (screen_h - height) // 2)}")
        root.protocol("WM_DELETE_WINDOW", self.close)
        stage_started = time.perf_counter()
        self.build_ui()
        if startup_timings is not None:
            startup_timings["editor_ui_build"] = time.perf_counter() - stage_started
        self.full_refresh(startup_timings=startup_timings)
        # Wait until the three-column workspace has its final size. Calling
        # fit_image before Tk finishes layout makes the image open as a small
        # thumbnail in the upper-left corner on high-DPI displays.
        self.root.after(100, self.fit_image)

    @property
    def doc(self) -> GeometryDocument:
        return self.documents[self.document_index]

    def build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        palette = {
            "page": "#F3F0E8", "card": "#FCFBF7", "soft": "#F5F2EA",
            "header": "#1D4034", "viewer": "#162A23", "ink": "#18332A",
            "muted": "#68746E", "line": "#D5D0C5", "line_strong": "#AAA79F",
            "green": "#0B755C", "green_hover": "#075C48", "green_soft": "#E7F1EC",
            "amber": "#A66A1F",
        }
        self.root.configure(background=palette["page"])
        style.configure("Editor.TFrame", background=palette["page"])
        style.configure("EditorHeader.TFrame", background=palette["header"])
        style.configure("Toolbar.TFrame", background=palette["card"])
        style.configure("Panel.TFrame", background=palette["card"])
        style.configure("EditorTitle.TLabel", background=palette["header"], foreground="#F8F5EC", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("EditorSub.TLabel", background=palette["header"], foreground="#AFC3B9", font=("Microsoft YaHei UI", 9))
        style.configure("ToolbarHint.TLabel", background=palette["card"], foreground=palette["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("Tool.TButton", background=palette["card"], foreground=palette["ink"], padding=(12, 7), font=("Microsoft YaHei UI", 9), borderwidth=1, bordercolor=palette["line_strong"], focusthickness=0)
        style.map("Tool.TButton", background=[("active", palette["green_soft"])], foreground=[("active", palette["green"])])
        style.configure("Primary.TButton", background=palette["green"], foreground="#FFFFFF", padding=(16, 8), font=("Microsoft YaHei UI", 10, "bold"), borderwidth=1, bordercolor=palette["green"], focusthickness=0)
        style.map("Primary.TButton", background=[("active", palette["green_hover"]), ("pressed", "#17362C")])
        style.configure("Tool.TRadiobutton", background=palette["card"], foreground=palette["ink"], padding=(11, 10), font=("Microsoft YaHei UI", 9))
        style.map("Tool.TRadiobutton", background=[("selected", palette["green_soft"])], foreground=[("selected", palette["green"])])
        style.configure("Panel.TLabelframe", background=palette["card"], bordercolor=palette["line"], padding=12)
        style.configure("Panel.TLabelframe.Label", background=palette["card"], foreground=palette["ink"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Panel.TLabel", background=palette["card"], foreground=palette["muted"])
        style.configure("PanelTitle.TLabel", background=palette["card"], foreground=palette["ink"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Status.TLabel", background=palette["card"], foreground=palette["muted"], padding=(14, 8))
        style.configure("TCombobox", padding=(8, 6), fieldbackground=palette["card"], bordercolor=palette["line_strong"])

        header = ttk.Frame(self.root, padding=(18, 12), style="EditorHeader.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="ROAD EDIT", foreground="#79C3AD", background=palette["header"], font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 20))
        title = ttk.Frame(header, style="EditorHeader.TFrame")
        title.pack(side="left", fill="x", expand=True)
        ttk.Label(title, text="最终成果中心线编辑", style="EditorTitle.TLabel").pack(anchor="w")
        ttk.Label(title, text="直接修正正式融合成果；保存后仅重建本期与相关时序成果", style="EditorSub.TLabel").pack(anchor="w", pady=(2, 0))
        self.saved_var = tk.StringVar(value="尚未保存")
        ttk.Label(header, textvariable=self.saved_var, style="EditorSub.TLabel").pack(side="right", padx=(12, 0))
        ttk.Button(header, text="保存编辑", style="Primary.TButton", command=self.save_all).pack(side="right", padx=(12, 0))
        ttk.Label(header, text="当前切片", style="EditorSub.TLabel").pack(side="right", padx=(18, 6))
        tile = ttk.Combobox(header, textvariable=self.tile_var, values=[doc.stem for doc in self.documents], state="readonly", width=25)
        tile.pack(side="right")
        tile.bind("<<ComboboxSelected>>", self.change_tile)

        toolbar = ttk.Frame(self.root, padding=(14, 8), style="Toolbar.TFrame")
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="← 返回复核", style="Tool.TButton", command=self.close).pack(side="left", padx=(0, 10))
        for value in self.PRIMARY_MODES:
            label = self.MODES[value]
            ttk.Radiobutton(
                toolbar, text=label, variable=self.mode, value=value,
                style="Tool.TRadiobutton", command=self._mode_changed,
            ).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="↶  撤销", style="Tool.TButton", command=self.undo).pack(side="left")
        ttk.Button(toolbar, text="↷  重做", style="Tool.TButton", command=self.redo).pack(side="left", padx=(4, 10))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="适应窗口", style="Tool.TButton", command=self.fit_image).pack(side="left")
        ttk.Button(toolbar, text="1:1", style="Tool.TButton", command=self.one_to_one).pack(side="left", padx=4)
        ttk.Button(toolbar, text="−", style="Tool.TButton", command=lambda: self.zoom_by(1 / 1.2)).pack(side="left")
        ttk.Button(toolbar, text="＋", style="Tool.TButton", command=lambda: self.zoom_by(1.2)).pack(side="left", padx=(4, 14))
        ttk.Button(toolbar, text="拓扑检查", style="Tool.TButton", command=self.show_topology).pack(side="left")
        ttk.Button(toolbar, text="交叉线建路口", style="Tool.TButton", command=self.node_all_crossings).pack(side="left", padx=4)

        self.context_frame = ttk.Frame(self.root, padding=(18, 6), style="Toolbar.TFrame")
        self.context_text = tk.StringVar()
        self.context_finish = ttk.Button(self.context_frame, text="完成  Enter", style="Primary.TButton", command=self.finish_drawing)
        self.context_cancel = ttk.Button(self.context_frame, text="取消  Esc", style="Tool.TButton", command=self.cancel_drawing)
        self.context_delete = ttk.Button(self.context_frame, text="删除所选中心线", style="Primary.TButton", command=self.delete_selected_edges)
        self.context_clear = ttk.Button(self.context_frame, text="取消选择  Esc", style="Tool.TButton", command=self.clear_selection)
        self.context_interval = ttk.Button(
            self.context_frame, text="应用到道路区间", style="Primary.TButton",
            command=self.begin_width_interval,
        )

        workspace = ttk.Frame(self.root, padding=(12, 0, 12, 0), style="Editor.TFrame")
        workspace.pack(fill="both", expand=True)
        canvas_frame = ttk.Frame(workspace, style="Editor.TFrame")
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, background=palette["viewer"], highlightthickness=1, highlightbackground=palette["line_strong"])
        xscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self._canvas_xview)
        yscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._canvas_yview)
        self.canvas.configure(
            xscrollcommand=lambda first, last: self._scrollbar_changed("x", xscroll, first, last),
            yscrollcommand=lambda first, last: self._scrollbar_changed("y", yscroll, first, last),
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        right_panel = ttk.Frame(
            workspace, padding=(8, 0, 0, 0), style="Editor.TFrame",
            width=round(250 * self.display_scale),
        )
        right_panel.pack(side="left", fill="y")
        right_panel.pack_propagate(False)
        tool_panel = ttk.LabelFrame(right_panel, text="当前工具", style="Panel.TLabelframe")
        tool_panel.pack(fill="x")
        self.tool_title_var = tk.StringVar()
        self.tool_hint_var = tk.StringVar()
        self.tool_value_var = tk.StringVar()
        ttk.Label(tool_panel, textvariable=self.tool_title_var, style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(tool_panel, textvariable=self.tool_hint_var, style="Panel.TLabel", wraplength=220).pack(anchor="w", pady=(6, 8))
        ttk.Label(tool_panel, textvariable=self.tool_value_var, style="PanelTitle.TLabel", wraplength=220).pack(anchor="w", pady=(0, 8))
        self.tool_select_frame = ttk.Frame(tool_panel, style="Panel.TFrame")
        ttk.Button(self.tool_select_frame, text="自由圈选", command=self.begin_lasso).pack(fill="x", pady=2)
        ttk.Button(self.tool_select_frame, text="删除所选  Delete", command=self.delete_selected_edges).pack(fill="x", pady=2)
        ttk.Button(self.tool_select_frame, text="取消选择", command=self.clear_selection).pack(fill="x", pady=2)
        self.tool_draw_frame = ttk.Frame(tool_panel, style="Panel.TFrame")
        ttk.Button(self.tool_draw_frame, text="完成绘线  Enter", command=self.finish_drawing).pack(fill="x", pady=2)
        ttk.Button(self.tool_draw_frame, text="取消  Esc", command=self.cancel_drawing).pack(fill="x", pady=2)
        self.tool_width_frame = ttk.Frame(tool_panel, style="Panel.TFrame")
        ttk.Button(self.tool_width_frame, text="确认当前区间", command=self.confirm_active_width).pack(fill="x", pady=2)
        ttk.Button(self.tool_width_frame, text="取消当前区间", command=self.cancel_active_width).pack(fill="x", pady=2)
        ttk.Button(self.tool_width_frame, text="删除最近测量", command=self.delete_latest_width).pack(fill="x", pady=2)
        self.tool_surface_frame = ttk.Frame(tool_panel, style="Panel.TFrame")
        ttk.Radiobutton(
            self.tool_surface_frame, text="补画", variable=self.surface_action, value="add",
        ).pack(anchor="w")
        ttk.Radiobutton(
            self.tool_surface_frame, text="擦除", variable=self.surface_action, value="remove",
        ).pack(anchor="w")
        ttk.Label(self.tool_surface_frame, text="笔刷大小 (px)", style="Panel.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Scale(
            self.tool_surface_frame, from_=2, to=100, variable=self.surface_radius,
            orient="horizontal",
        ).pack(fill="x")
        self.metric_vars = {
            key: tk.StringVar(value="--")
            for key in ("node_count", "edge_count", "component_count", "dangling_endpoint_count")
        }
        more_panel = ttk.LabelFrame(right_panel, text="更多", style="Panel.TLabelframe")
        more_panel.pack(fill="x", pady=(10, 0))
        ttk.Button(more_panel, text="拓扑检查", command=self.show_topology).pack(fill="x")
        ttk.Button(more_panel, text="交叉线建路口", command=self.node_all_crossings).pack(fill="x", pady=(4, 0))
        advanced = ttk.Menubutton(more_panel, text="高级工具  ▸")
        advanced_menu = tk.Menu(advanced, tearoff=False)
        advanced_menu.add_command(label="路面局部修补", command=lambda: self.activate_surface_tool("add"))
        advanced_menu.add_command(label="路面局部擦除", command=lambda: self.activate_surface_tool("remove"))
        advanced_menu.add_separator()
        advanced_menu.add_command(label="清除本期编辑缓存", command=self.clear_current_editor_cache)
        advanced.configure(menu=advanced_menu)
        advanced.pack(fill="x", pady=(4, 0))
        ttk.Label(
            more_panel,
            text=("仅用于路口、匝道、广场式道路等特殊区域的局部修补。"
                  "通常修改中心线或人工宽度即可自动重建道路面。"),
            style="Panel.TLabel", wraplength=220,
        ).pack(anchor="w", pady=(6, 0))

        legend_panel = ttk.LabelFrame(right_panel, text="图层与图例", style="Panel.TLabelframe")
        legend_panel.pack(fill="x", pady=(10, 0))
        for text, color in (
            ("━━  最终中心线", "#22C55E"), ("━━  已选中心线", "#F97316"),
            ("●━●  人工法向测宽", "#38BDF8"), ("━━━━  人工定宽区间", "#FACC15"),
        ):
            ttk.Label(legend_panel, text=text, foreground=color, background=palette["card"]).pack(anchor="w", pady=3)

        status = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel")
        status.pack(fill="x")
        self.canvas.bind("<ButtonPress-1>", self.mouse_press)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)
        self.canvas.bind("<ButtonPress-2>", lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind("<B2-Motion>", self.middle_mouse_pan)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.finish_drawing() if self.mode.get() == "draw" else None)
        self.canvas.bind("<ButtonPress-3>", self.mouse_right_press)
        self.canvas.bind("<B3-Motion>", self.mouse_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.mouse_right_release)
        self.canvas.bind("<Motion>", self.mouse_motion)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Configure>", lambda _event: self.schedule_background_refresh())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-y>", lambda _event: self.redo())
        self.root.bind("<Control-s>", lambda _event: self.save_all())
        self.root.bind("<Return>", lambda _event: self.finish_drawing())
        self.root.bind("<Escape>", lambda _event: self.cancel_drawing())
        self.root.bind("<Delete>", lambda event: self.delete_selected_edges() if not self._text_input_focused(event) else None)
        for key, mode in (("v", "select"), ("l", "draw"), ("w", "measure_width"), ("b", "surface_add")):
            self.root.bind(key, lambda event, value=mode: self.set_mode_shortcut(event, value))
            self.root.bind(key.upper(), lambda event, value=mode: self.set_mode_shortcut(event, value))
        self.root.bind("f", lambda event: self.fit_image() if not self._text_input_focused(event) else None)
        self.root.bind("F", lambda event: self.fit_image() if not self._text_input_focused(event) else None)
        self.root.bind("1", lambda event: self.one_to_one() if not self._text_input_focused(event) else None)
        self.root.bind("<KeyPress-space>", self.space_press)
        self.root.bind("<KeyRelease-space>", self.space_release)
        self._mode_changed()

    def _mode_changed(self) -> None:
        mode = self.mode.get()
        if mode != "measure_width":
            self.width_drag_start = None
            self.width_drag_current = None
            self.width_preview = None
            self.interval_draft = []
            self.interval_measurement_id = None
            self.width_range_drag_handle = None
            self.width_range_drag_checkpoint = False
        self.update_context_panel()
        self.refresh_dynamic_overlay()

    def activate_surface_tool(self, action: str) -> None:
        self.surface_action.set("remove" if action == "remove" else "add")
        self.mode.set("surface_add")
        self._mode_changed()

    def clear_current_editor_cache(self) -> None:
        if not messagebox.askyesno(
            "清除本期编辑缓存",
            "确定清除当前期次的本地编辑缓存吗？\n\n"
            "这不会删除道路成果或人工编辑成果，下次打开时会自动重新生成。",
            parent=self.root,
        ):
            return
        try:
            removed = clear_editor_cache(self.review_dir)
        except OSError as exc:
            messagebox.showerror("无法清除缓存", str(exc), parent=self.root)
            return
        message = "本期编辑缓存已清除。" if removed else "当前期次尚无本地编辑缓存。"
        self.status_var.set(message)

    def active_width_measurement(self) -> dict | None:
        measurement_id = getattr(self, "active_width_measurement_id", None)
        if not measurement_id:
            return None
        return next((
            row for row in self.doc.manual_widths
            if str(row.get("measurement_id", "")) == measurement_id
        ), None)

    def update_context_panel(self) -> None:
        for frame in (
            self.tool_select_frame, self.tool_draw_frame,
            self.tool_width_frame, self.tool_surface_frame,
        ):
            frame.pack_forget()
        mode = self.mode.get()
        if mode == "draw":
            self.tool_title_var.set("绘制中心线  L")
            self.tool_hint_var.set("沿道路依次点击添加节点，双击或 Enter 完成；Esc 取消。")
            self.tool_value_var.set(f"当前节点：{len(self.draft)}")
            self.tool_draw_frame.pack(fill="x")
        elif mode == "measure_width":
            self.tool_title_var.set("人工测宽  W")
            active = self.active_width_measurement()
            if self.width_preview is not None:
                value = float(self.width_preview.get("width_units", 0.0))
                self.tool_value_var.set(f"人工宽度：{value:.2f} m")
                self.tool_hint_var.set("跨道路两侧拖动鼠标测量宽度。")
            elif active is not None:
                value = float(active.get("width_units", active.get("width_px", 0.0)))
                lo = float(active.get("range_start_position", 0.0)) * float(getattr(self.doc, "pixel_size", 1.0))
                hi = float(active.get("range_end_position", lo)) * float(getattr(self.doc, "pixel_size", 1.0))
                self.tool_hint_var.set("拖动地图中的两个端点，可调整该宽度的作用范围。")
                self.tool_value_var.set(
                    f"宽度\n{value:.2f} m\n\n应用范围\n当前道路链\n\n"
                    f"区间长度\n{max(0.0, hi - lo):.1f} m\n\n起点 {lo:.1f} m\n终点 {hi:.1f} m"
                )
            else:
                self.tool_hint_var.set(
                    "跨道路两侧拖动鼠标测量宽度。"
                    "系统会自动吸附到最近道路中心线，并将宽度应用到该道路的一段范围。"
                )
                self.tool_value_var.set("尚无人工测宽记录")
            self.tool_width_frame.pack(fill="x")
        elif mode == "surface_add":
            self.tool_title_var.set("路面高级编辑  B")
            self.tool_hint_var.set("仅用于特殊道路区域的局部修补。左键涂画，右键可临时擦除。")
            action = "补画" if self.surface_action.get() == "add" else "擦除"
            self.tool_value_var.set(f"模式：{action}\n笔刷：{self.surface_radius.get()} px")
            self.tool_surface_frame.pack(fill="x")
        else:
            self.tool_title_var.set("选择/编辑  V")
            self.tool_hint_var.set("拖动节点调整中心线；单击道路选择；Shift 可多选；Delete 删除。")
            self.tool_value_var.set(f"已选择：{len(self.selected_edge_ids)} 条中心线")
            self.tool_select_frame.pack(fill="x")

    def zoom_by(self, factor: float) -> None:
        self.apply_zoom(self.zoom * factor)

    def one_to_one(self) -> None:
        self.apply_zoom(1.0)

    def source_point(self, event) -> tuple[float, float]:
        col = self.canvas.canvasx(event.x) / self.zoom
        row = self.canvas.canvasy(event.y) / self.zoom
        return float(np.clip(row, 0, self.doc.image.shape[0] - 1)), float(np.clip(col, 0, self.doc.image.shape[1] - 1))

    def fit_image(self) -> None:
        self.root.update_idletasks()
        width = max(300, self.canvas.winfo_width())
        height = max(300, self.canvas.winfo_height())
        self.apply_zoom(min((width - 4) / self.doc.image.shape[1], (height - 4) / self.doc.image.shape[0]))

    def mouse_wheel(self, event) -> None:
        if event.delta == 0:
            return
        steps = event.delta / 120.0 if abs(event.delta) >= 120 else math.copysign(1.0, event.delta)
        base = 1.08 if event.state & 0x0004 else 1.25
        factor = base ** steps
        self.apply_zoom(self.zoom * factor, event)

    def _continuous_zoom(self, requested: float) -> float:
        return max(self.min_zoom, min(self.max_zoom, float(requested)))

    def apply_zoom(self, requested: float, event=None) -> None:
        new_zoom = self._continuous_zoom(requested)
        old_zoom = float(self.zoom)
        if abs(new_zoom - old_zoom) <= 1e-9:
            return
        if event is not None:
            source_col = self.canvas.canvasx(event.x) / old_zoom
            source_row = self.canvas.canvasy(event.y) / old_zoom
            anchor_x, anchor_y = float(event.x), float(event.y)
        else:
            anchor_x = max(0.0, self.canvas.winfo_width() * 0.5)
            anchor_y = max(0.0, self.canvas.winfo_height() * 0.5)
            source_col = self.canvas.canvasx(anchor_x) / old_zoom
            source_row = self.canvas.canvasy(anchor_y) / old_zoom
        self.zoom = new_zoom
        ratio = new_zoom / old_zoom
        for tag in ("edge", "node", "manual_width", "dynamic"):
            self.canvas.scale(tag, 0, 0, ratio, ratio)
        if (old_zoom < 0.8) != (new_zoom < 0.8):
            self.refresh_static_geometry()
        width = max(1.0, self.doc.image.shape[1] * new_zoom)
        height = max(1.0, self.doc.image.shape[0] * new_zoom)
        scrollregion = (0.0, 0.0, float(width), float(height))
        if self._background_scrollregion != scrollregion:
            self.canvas.configure(scrollregion=scrollregion)
            self._background_scrollregion = scrollregion
        self.canvas.xview_moveto(max(0.0, min(1.0, (source_col * new_zoom - anchor_x) / width)))
        self.canvas.yview_moveto(max(0.0, min(1.0, (source_row * new_zoom - anchor_y) / height)))
        self.schedule_background_refresh(force=True)
        self.update_status()

    def render_source(self) -> np.ndarray:
        canvas = self.doc.image.copy()
        overlay = canvas.copy()
        editable_surface = self.doc.editable_surface()
        overlay[editable_surface > 0] = (40, 180, 40)
        overlay[self.doc.surface_additions > 0] = (255, 120, 0)
        overlay[self.doc.surface_removals > 0] = (0, 0, 255)
        canvas = cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0)
        for edge_id, (src, dst) in enumerate(self.doc.edges.tolist()):
            r0, c0 = self.doc.nodes[src]
            r1, c1 = self.doc.nodes[dst]
            selected = edge_id in self.selected_edge_ids
            color = (0, 80, 255) if selected else (0, 255, 0)
            thickness = 4 if selected else 2
            cv2.line(canvas, (int(round(c0)), int(round(r0))), (int(round(c1)), int(round(r1))), color, thickness, cv2.LINE_AA)
        if self.zoom >= 0.55:
            for row, col in self.doc.nodes.tolist():
                cv2.circle(canvas, (int(round(col)), int(round(row))), 2, (0, 80, 255), -1)
        if len(self.draft) >= 1:
            pts = np.asarray([[int(round(col)), int(round(row))] for row, col in self.draft], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, (255, 0, 255), 2, cv2.LINE_AA)
            for point in pts:
                cv2.circle(canvas, tuple(point), 4, (255, 0, 255), -1)
        if len(self.lasso) >= 3:
            pts = np.asarray([[int(round(col)), int(round(row))] for row, col in self.lasso], dtype=np.int32)
            shade = canvas.copy()
            cv2.fillPoly(shade, [pts], (0, 180, 255))
            canvas = cv2.addWeighted(shade, 0.16, canvas, 0.84, 0)
            cv2.polylines(canvas, [pts], True, (0, 220, 255), 2, cv2.LINE_AA)
        for measurement in self.doc.manual_widths:
            if str(measurement.get("source", "")) == "manual_interval_width":
                try:
                    chain_id = int(measurement["target_chain_id"])
                    lo = float(measurement["range_start_position"])
                    hi = float(measurement["range_end_position"])
                except (KeyError, TypeError, ValueError):
                    chain_id = -1
                    lo = hi = -1.0
                for chain in self.doc.road_chains():
                    if chain.chain_id != chain_id:
                        continue
                    points = self.doc.nodes[np.asarray(chain.node_ids, dtype=np.int32)]
                    cumulative = np.concatenate((
                        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
                    ))
                    for offset, edge_id in enumerate(chain.edge_ids):
                        midpoint = 0.5 * float(cumulative[offset] + cumulative[offset + 1])
                        if lo - 1e-6 <= midpoint <= hi + 1e-6:
                            src, dst = self.doc.edges[int(edge_id)]
                            cv2.line(
                                canvas,
                                (int(round(self.doc.nodes[src, 1])), int(round(self.doc.nodes[src, 0]))),
                                (int(round(self.doc.nodes[dst, 1])), int(round(self.doc.nodes[dst, 0]))),
                                (0, 210, 255), 4, cv2.LINE_AA,
                            )
            try:
                p0 = (int(round(float(measurement["start_col"]))), int(round(float(measurement["start_row"]))))
                p1 = (int(round(float(measurement["end_col"]))), int(round(float(measurement["end_row"]))))
                target = (
                    int(round(float(measurement["target_col"]))),
                    int(round(float(measurement["target_row"]))),
                )
            except (KeyError, TypeError, ValueError):
                continue
            cv2.line(canvas, p0, p1, (255, 180, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, p0, 4, (255, 220, 80), -1)
            cv2.circle(canvas, p1, 4, (255, 220, 80), -1)
            cv2.circle(canvas, target, 3, (0, 255, 255), -1)
            width_units = float(measurement.get("width_units", measurement.get("width_px", 0.0)))
            cv2.putText(
                canvas, f"{width_units:.2f} m", (target[0] + 6, target[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 240, 120), 1, cv2.LINE_AA,
            )
        if self.width_drag_start is not None and self.width_drag_current is not None:
            raw0 = (int(round(self.width_drag_start[1])), int(round(self.width_drag_start[0])))
            raw1 = (int(round(self.width_drag_current[1])), int(round(self.width_drag_current[0])))
            cv2.line(canvas, raw0, raw1, (255, 0, 255), 1, cv2.LINE_AA)
            preview = self.width_preview
            if preview is not None:
                p0 = (int(round(preview["start_col"])), int(round(preview["start_row"])))
                p1 = (int(round(preview["end_col"])), int(round(preview["end_row"])))
                cv2.line(canvas, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    canvas, f"{float(preview['width_units']):.2f} m", p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
                )
        for row, col in self.interval_draft:
            cv2.circle(canvas, (int(round(col)), int(round(row))), 5, (0, 210, 255), -1)
        return canvas

    def _background_source(self) -> np.ndarray:
        version = self.surface_versions[self.document_index]
        cached = self.background_source_cache.get(self.document_index)
        if cached is not None and cached[0] == version:
            return cached[1]
        source = self.doc.image.copy()
        overlay = source.copy()
        editable_surface = self.doc.editable_surface()
        overlay[editable_surface > 0] = (40, 180, 40)
        overlay[self.doc.surface_additions > 0] = (255, 120, 0)
        overlay[self.doc.surface_removals > 0] = (0, 0, 255)
        source = cv2.addWeighted(overlay, 0.20, source, 0.80, 0)
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        self.background_source_cache[self.document_index] = (version, rgb)
        return rgb

    def _background_photo(self) -> tuple[ImageTk.PhotoImage, float, float]:
        source = self._background_source()
        canvas_left = max(0.0, self.canvas.canvasx(0))
        canvas_top = max(0.0, self.canvas.canvasy(0))
        canvas_right = canvas_left + max(1, self.canvas.winfo_width())
        canvas_bottom = canvas_top + max(1, self.canvas.winfo_height())
        # Keep a small screen-space margin so rapid panning does not expose the
        # viewer background before the next idle redraw.
        margin = 96.0
        left = max(0, int(math.floor((canvas_left - margin) / self.zoom)))
        top = max(0, int(math.floor((canvas_top - margin) / self.zoom)))
        right = min(source.shape[1], int(math.ceil((canvas_right + margin) / self.zoom)))
        bottom = min(source.shape[0], int(math.ceil((canvas_bottom + margin) / self.zoom)))
        right = max(left + 1, right)
        bottom = max(top + 1, bottom)
        key = (
            self.document_index, self.surface_versions[self.document_index],
            round(float(self.zoom), 5), left, top, right, bottom,
        )
        if key in self.photo_cache:
            photo = self.photo_cache.pop(key)
            self.photo_cache[key] = photo
            return photo, left * self.zoom, top * self.zoom
        crop = source[top:bottom, left:right]
        width = max(1, int(round((right - left) * self.zoom)))
        height = max(1, int(round((bottom - top) * self.zoom)))
        if self.zoom < 1:
            interpolation = cv2.INTER_AREA
        elif self.zoom <= 8:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_NEAREST
        resized = cv2.resize(crop, (width, height), interpolation=interpolation)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.photo_cache[key] = photo
        def cached_pixels() -> int:
            return sum(item.width() * item.height() for item in self.photo_cache.values())
        while len(self.photo_cache) > 5 or (len(self.photo_cache) > 1 and cached_pixels() > 32_000_000):
            self.photo_cache.popitem(last=False)
        return photo, left * self.zoom, top * self.zoom

    def _background_view_state(self) -> tuple:
        """Return the viewport inputs that materially affect the cached crop."""
        xview = tuple(round(float(value), 7) for value in self.canvas.xview())
        yview = tuple(round(float(value), 7) for value in self.canvas.yview())
        return (
            self.document_index,
            round(float(self.zoom), 7),
            xview,
            yview,
            int(self.canvas.winfo_width()),
            int(self.canvas.winfo_height()),
            int(self.surface_versions[self.document_index]),
        )

    def _cancel_background_refresh(self) -> None:
        after_id = self._background_refresh_after_id
        self._background_refresh_after_id = None
        if after_id is None:
            return
        try:
            self.root.after_cancel(after_id)
        except tk.TclError:
            pass

    def refresh_background(self) -> None:
        self._cancel_background_refresh()
        if self._refreshing_background:
            return
        self._refreshing_background = True
        try:
            self.photo, image_x, image_y = self._background_photo()
            if self.background_item is None or not self.canvas.find_withtag("background"):
                self.background_item = self.canvas.create_image(
                    image_x, image_y, image=self.photo, anchor="nw", tags=("background",),
                )
                self.canvas.tag_lower(self.background_item)
            else:
                self.canvas.itemconfigure(self.background_item, image=self.photo)
            self.canvas.coords(self.background_item, image_x, image_y)
            scrollregion = (
                0.0, 0.0,
                float(self.doc.image.shape[1] * self.zoom),
                float(self.doc.image.shape[0] * self.zoom),
            )
            if self._background_scrollregion != scrollregion:
                self.canvas.configure(scrollregion=scrollregion)
                self._background_scrollregion = scrollregion
            self.surface_dirty = False
        finally:
            self._refreshing_background = False
            self._last_background_view = self._background_view_state()

    def _run_scheduled_background_refresh(self) -> None:
        self._background_refresh_after_id = None
        if self._refreshing_background:
            return
        if self._last_background_view == self._background_view_state():
            return
        self.refresh_background()

    def schedule_background_refresh(self, force: bool = False) -> None:
        if self._refreshing_background:
            return
        if force:
            self._last_background_view = None
        elif self._last_background_view == self._background_view_state():
            return
        if self._background_refresh_after_id is not None:
            return
        self._background_refresh_after_id = self.root.after(
            20, self._run_scheduled_background_refresh,
        )

    def _scrollbar_changed(self, axis: str, scrollbar, first: str, last: str) -> None:
        scrollbar.set(first, last)
        current = (float(first), float(last))
        previous = self._last_scrollbar_views.get(axis)
        self._last_scrollbar_views[axis] = current
        if self._refreshing_background:
            return
        if previous is not None and all(
            abs(before - after) <= 1e-7 for before, after in zip(previous, current)
        ):
            return
        self.schedule_background_refresh()

    def _canvas_xview(self, *args) -> None:
        self.canvas.xview(*args)
        self.schedule_background_refresh()

    def _canvas_yview(self, *args) -> None:
        self.canvas.yview(*args)
        self.schedule_background_refresh()

    def middle_mouse_pan(self, event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.schedule_background_refresh()

    def refresh_static_geometry(self) -> None:
        self.canvas.delete("edge")
        self.canvas.delete("node")
        self.edge_items = {}
        self.node_items = {}
        z = self.zoom
        for edge_id, (src, dst) in enumerate(self.doc.edges.tolist()):
            r0, c0 = self.doc.nodes[int(src)]
            r1, c1 = self.doc.nodes[int(dst)]
            selected = edge_id in self.selected_edge_ids
            self.edge_items[edge_id] = self.canvas.create_line(
                c0 * z, r0 * z, c1 * z, r1 * z,
                fill="#F97316" if selected else "#22C55E",
                width=4 if selected else 2, tags=("edge", f"edge_{edge_id}"),
            )
        if z >= 0.8:
            for node_id, (row, col) in enumerate(self.doc.nodes.tolist()):
                radius = 3
                self.node_items[node_id] = self.canvas.create_oval(
                    col * z - radius, row * z - radius, col * z + radius, row * z + radius,
                    fill="#F97316", outline="", tags=("node", f"node_{node_id}"),
                )
        self.geometry_dirty = False

    def refresh_manual_widths(self) -> None:
        self.canvas.delete("manual_width")
        z = self.zoom
        interval_rows = [
            row for row in self.doc.manual_widths
            if str(row.get("source", "")) == "manual_interval_width"
        ]
        # Keep chain construction lazy: an untouched document needs no chain
        # cache merely to paint its first frame.
        chains = {
            int(chain.chain_id): chain for chain in self.doc.road_chains()
        } if interval_rows else {}
        for measurement in self.doc.manual_widths:
            if str(measurement.get("source", "")) == "manual_interval_width":
                try:
                    chain = chains[int(measurement["target_chain_id"])]
                    lo = float(measurement["range_start_position"])
                    hi = float(measurement["range_end_position"])
                    points, cumulative = self.doc._chain_geometry_cache[int(chain.chain_id)]
                except (KeyError, TypeError, ValueError):
                    chain = None
                if chain is not None:
                    preview = manual_width_preview_geometry(self.doc, measurement)
                    if preview is not None and not preview.is_empty:
                        polygon_coords = [
                            value
                            for col, row in preview.exterior.coords
                            for value in (float(col) * z, float(row) * z)
                        ]
                        if len(polygon_coords) >= 6:
                            self.canvas.create_polygon(
                                *polygon_coords, fill="#0EA5E9", outline="#38BDF8",
                                stipple="gray50", width=1, tags=("manual_width", "width_preview"),
                            )
                    interval_points = chain_interval_points(self.doc, int(chain.chain_id), lo, hi)
                    if len(interval_points) >= 2:
                        interval_coords = [
                            value for row, col in interval_points
                            for value in (float(col) * z, float(row) * z)
                        ]
                        self.canvas.create_line(
                            *interval_coords, fill="#FACC15", width=5,
                            capstyle="round", joinstyle="round",
                            tags=("manual_width", "width_interval"),
                        )
                    if str(measurement.get("measurement_id", "")) == getattr(self, "active_width_measurement_id", None):
                        chain_coords = [
                            value for row, col in points
                            for value in (float(col) * z, float(row) * z)
                        ]
                        self.canvas.create_line(
                            *chain_coords, fill="#FDE68A", width=2, dash=(6, 4),
                            tags=("manual_width", "active_width_chain"),
                        )
                        for handle, fill in (("start", "#F97316"), ("end", "#EF4444")):
                            row = float(measurement[f"range_{handle}_row"])
                            col = float(measurement[f"range_{handle}_col"])
                            radius = max(6.0, 7.0 * float(getattr(self, "display_scale", 1.0)))
                            self.canvas.create_oval(
                                col * z - radius, row * z - radius,
                                col * z + radius, row * z + radius,
                                fill=fill, outline="#FFFFFF", width=2,
                                tags=("manual_width", f"width_range_{handle}_handle"),
                            )
            try:
                p0 = (float(measurement["start_col"]) * z, float(measurement["start_row"]) * z)
                p1 = (float(measurement["end_col"]) * z, float(measurement["end_row"]) * z)
                target = (float(measurement["target_col"]) * z, float(measurement["target_row"]) * z)
            except (KeyError, TypeError, ValueError):
                continue
            self.canvas.create_line(*p0, *p1, fill="#38BDF8", width=2, tags=("manual_width",))
            for x, y in (p0, p1):
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#7DD3FC", outline="", tags=("manual_width",))
            self.canvas.create_oval(
                target[0] - 3, target[1] - 3, target[0] + 3, target[1] + 3,
                fill="#FACC15", outline="", tags=("manual_width",),
            )
            width_units = float(measurement.get("width_units", measurement.get("width_px", 0.0)))
            self.canvas.create_text(
                target[0] + 7, target[1] - 7, text=f"{width_units:.2f} m",
                fill="#E0F2FE", anchor="sw", tags=("manual_width",),
            )

    def width_range_handle_at(self, row: float, col: float) -> tuple[str, str] | None:
        """Return the nearest visible interval handle in source-pixel space."""
        tolerance = max(5.0, 10.0 / max(self.zoom, 1e-6))
        best: tuple[float, str, str] | None = None
        for measurement in self.doc.manual_widths:
            if str(measurement.get("source", "")) != "manual_interval_width":
                continue
            measurement_id = str(measurement.get("measurement_id", ""))
            for handle in ("start", "end"):
                try:
                    handle_row = float(measurement[f"range_{handle}_row"])
                    handle_col = float(measurement[f"range_{handle}_col"])
                except (KeyError, TypeError, ValueError):
                    continue
                distance = math.hypot(row - handle_row, col - handle_col)
                if distance <= tolerance and (best is None or distance < best[0]):
                    best = (distance, measurement_id, handle)
        return None if best is None else (best[1], best[2])

    def refresh_dynamic_overlay(self) -> None:
        self.canvas.delete("dynamic")
        z = self.zoom
        if self.draft:
            coords = [value for row, col in self.draft for value in (col * z, row * z)]
            if len(coords) >= 4:
                self.canvas.create_line(*coords, fill="#D946EF", width=2, tags=("dynamic",))
            for row, col in self.draft:
                self.canvas.create_oval(
                    col * z - 4, row * z - 4, col * z + 4, row * z + 4,
                    fill="#D946EF", outline="", tags=("dynamic",),
                )
        for row, col in self.interval_draft:
            self.canvas.create_oval(
                col * z - 5, row * z - 5, col * z + 5, row * z + 5,
                fill="#FACC15", outline="", tags=("dynamic",),
            )
        if self.width_drag_start is not None and self.width_drag_current is not None:
            self.canvas.create_line(
                self.width_drag_start[1] * z, self.width_drag_start[0] * z,
                self.width_drag_current[1] * z, self.width_drag_current[0] * z,
                fill="#D946EF", width=1, tags=("dynamic",),
            )
        if self.width_preview is not None:
            preview = self.width_preview
            p0 = (float(preview["start_col"]) * z, float(preview["start_row"]) * z)
            p1 = (float(preview["end_col"]) * z, float(preview["end_row"]) * z)
            target = (float(preview["target_col"]) * z, float(preview["target_row"]) * z)
            self.canvas.create_line(*p0, *p1, fill="#22D3EE", width=2, tags=("dynamic",))
            for x, y in (p0, p1, target):
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#22D3EE", outline="", tags=("dynamic",))
            self.canvas.create_text(
                p1[0] + 6, p1[1] - 6,
                text=f"人工测宽：{float(preview['width_units']):.2f} m",
                fill="#A5F3FC", anchor="sw", tags=("dynamic",),
            )
        if self.surface_stroke_points:
            coords = [value for row, col in self.surface_stroke_points for value in (col * z, row * z)]
            if len(coords) >= 4:
                self.canvas.create_line(
                    *coords, fill="#38BDF8" if self.surface_action.get() == "add" else "#FB7185",
                    width=max(2, self.surface_radius.get() * 2 * z), smooth=True,
                    capstyle="round", tags=("dynamic",),
                )
        self.update_status()

    def refresh_metrics(self) -> None:
        # Full connected-component/topology analysis is intentionally lazy.
        # It is computed by show_topology(), not while the first frame loads.
        self.last_metrics = {
            "node_count": int(len(self.doc.nodes)),
            "edge_count": int(len(self.doc.edges)),
            **{
                key: value for key, value in self.last_metrics.items()
                if key not in {"node_count", "edge_count"}
            },
        }
        if hasattr(self, "metric_vars"):
            for key, variable in self.metric_vars.items():
                variable.set(str(self.last_metrics.get(key, "--")))
        self.topology_dirty = False
        self.update_status()

    def update_status(self) -> None:
        metrics = self.last_metrics
        self.status_var.set(
            f"{self.doc.stem}  │  {self.MODES.get(self.mode.get(), self.mode.get())}  │  "
            f"Zoom {int(round(self.zoom * 100))}%  │  "
            f"鼠标 {self.mouse_status[1]:.1f}, {self.mouse_status[0]:.1f}  │  "
            f"节点 {metrics.get('node_count', len(self.doc.nodes))}  │  边 {metrics.get('edge_count', len(self.doc.edges))}"
        )

    def full_refresh(self, startup_timings: dict[str, float] | None = None) -> None:
        stage_started = time.perf_counter()
        self.refresh_background()
        if startup_timings is not None:
            startup_timings["first_background_render"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        self.refresh_static_geometry()
        if startup_timings is not None:
            startup_timings["first_vector_overlay"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        self.refresh_manual_widths()
        if startup_timings is not None:
            startup_timings["manual_width_overlay_and_lazy_chains"] = time.perf_counter() - stage_started
        self.refresh_dynamic_overlay()
        self.refresh_metrics()
        self.update_context_panel()

    def refresh(self) -> None:
        """Compatibility alias for infrequent document-level rebuilds."""
        self.full_refresh()

    def mouse_press(self, event) -> None:
        if self.space_pressed:
            self.space_panning = True
            self.canvas.scan_mark(event.x, event.y)
            return
        row, col = self.source_point(event)
        mode = self.mode.get()
        tolerance = max(3.0, 8.0 / self.zoom)
        if mode == "select":
            if self.lasso_active:
                self.lasso = [(row, col)]
                self.selected_edge_ids.clear()
                self._start_lasso_canvas(event)
                return
            node_id = self.doc.nearest_node(row, col, tolerance)
            if node_id is not None:
                self.doc.checkpoint()
                self.drag_checkpoint = True
                self.drag_node = node_id
            else:
                nearest = self.doc.nearest_edge(row, col, tolerance)
                if nearest is not None:
                    edge_id = int(nearest[0])
                    if event.state & 0x0001:
                        if edge_id in self.selected_edge_ids:
                            self.selected_edge_ids.remove(edge_id)
                        else:
                            self.selected_edge_ids.add(edge_id)
                    else:
                        self.selected_edge_ids = {edge_id}
                    self.refresh_edge_selection()
                    self.update_context_panel()
        elif mode == "draw":
            self.draft.append((row, col))
            self.update_context_panel()
            self.refresh_dynamic_overlay()
        elif mode == "measure_width":
            handle_hit = self.width_range_handle_at(row, col)
            if handle_hit is not None:
                self.active_width_measurement_id, self.width_range_drag_handle = handle_hit
                self.width_range_drag_checkpoint = False
                self.width_drag_start = None
                self.width_drag_current = None
                self.refresh_manual_widths()
                self.update_context_panel()
            elif self.interval_measurement_id is not None:
                self.interval_draft.append((row, col))
                if len(self.interval_draft) == 2:
                    self.finish_width_interval()
                else:
                    self.context_text.set("区间定宽：已选择起点，请在同一连续道路链上选择终点。")
                    self.refresh_dynamic_overlay()
            else:
                self.width_drag_start = (row, col)
                self.width_drag_current = (row, col)
                self.width_preview = None
                self.refresh_dynamic_overlay()
        elif mode == "surface_add":
            self.doc.checkpoint()
            self.surface_stroke_active = True
            self.surface_stroke_points = [(row, col)]
            self.doc.paint_surface(row, col, self.surface_radius.get(), self.surface_action.get() == "add")
            self.start_surface_preview(row, col, self.surface_action.get() == "add")

    def mouse_drag(self, event) -> None:
        if self.space_panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.schedule_background_refresh()
            return
        row, col = self.source_point(event)
        if self.mode.get() == "select" and self.drag_node is not None:
            self.doc.nodes[self.drag_node] = (row, col)
            self.update_dragged_node_items(self.drag_node)
        elif self.mode.get() == "select" and self.lasso_active and self.lasso:
            last = self.lasso[-1]
            if math.hypot(row - last[0], col - last[1]) >= max(1.0, 3.0 / self.zoom):
                self.lasso.append((row, col))
                self._extend_lasso_canvas(event)
        elif self.mode.get() == "measure_width" and getattr(self, "width_range_drag_handle", None) is not None:
            active = self.active_width_measurement()
            if active is None:
                self.width_range_drag_handle = None
                return
            updated, error = update_manual_width_interval_endpoint(
                self.doc, active, self.width_range_drag_handle, (row, col),
                max_centerline_distance=max(12.0, 24.0 / self.zoom),
            )
            if updated is None:
                self.context_text.set(error)
                return
            if manual_width_interval_overlaps(
                self.doc.manual_widths, updated,
                exclude_measurement_id=str(updated.get("measurement_id", "")),
            ):
                self.context_text.set("该范围与已有人工宽度区间重叠，已保持原范围。")
                return
            if not getattr(self, "width_range_drag_checkpoint", False):
                self.doc.checkpoint()
                self.width_range_drag_checkpoint = True
            index = self.doc.manual_widths.index(active)
            self.doc.manual_widths[index] = updated
            self.saved_var.set("尚未保存")
            self.context_text.set("拖动端点调整人工宽度作用范围。")
            self.refresh_manual_widths()
            self.update_context_panel()
        elif self.mode.get() == "measure_width" and self.width_drag_start is not None:
            self.width_drag_current = (row, col)
            preview, _error = create_normal_width_measurement(
                self.doc, self.width_drag_start, self.width_drag_current,
                max_centerline_distance=max(12.0, 24.0 / self.zoom),
            )
            if preview is not None:
                self.context_text.set(f"人工测宽：{float(preview['width_units']):.2f} m")
            self.width_preview = preview
            self.update_context_panel()
            self.refresh_dynamic_overlay()
        elif self.mode.get() == "surface_add" and self.surface_stroke_active:
            self.doc.paint_surface(row, col, self.surface_radius.get(), self.surface_action.get() == "add")
            if not self.surface_stroke_points or math.hypot(
                row - self.surface_stroke_points[-1][0], col - self.surface_stroke_points[-1][1]
            ) >= 1.0:
                self.surface_stroke_points.append((row, col))
                self.extend_surface_preview(row, col)

    def mouse_release(self, event) -> None:
        if self.space_panning:
            self.space_panning = False
            return
        surface_was_active = self.surface_stroke_active
        self.surface_stroke_active = False
        if self.mode.get() == "measure_width" and getattr(self, "width_range_drag_handle", None) is not None:
            self.width_range_drag_handle = None
            self.width_range_drag_checkpoint = False
            self.refresh_manual_widths()
            self.update_context_panel()
        elif self.mode.get() == "measure_width" and self.width_drag_start is not None:
            self.width_drag_current = self.source_point(event)
            self.finish_width_measurement()
        if self.drag_node is not None:
            self.doc.invalidate_geometry_cache()
            self.doc.snap_moved_node(self.drag_node, max(3.0, 8.0 / self.zoom))
            self.drag_node = None
            self.drag_checkpoint = False
            self.geometry_dirty = self.topology_dirty = True
            self.refresh_static_geometry()
            self.refresh_manual_widths()
            self.refresh_metrics()
        if self.mode.get() == "select" and self.lasso_active and self.lasso:
            if len(self.lasso) >= 3:
                self.selected_edge_ids = self.doc.edges_in_polygon(self.lasso)
            self._remove_lasso_canvas()
            self.lasso_active = False
            self.refresh_edge_selection()
            self.update_context_panel()
        if surface_was_active:
            self.surface_versions[self.document_index] += 1
            self.surface_stroke_points = []
            self.surface_preview_item = None
            self.surface_dirty = True
            self.refresh_background()
            self.refresh_dynamic_overlay()

    def finish_drawing(self) -> None:
        if len(self.draft) >= 2:
            self.doc.checkpoint()
            self.doc.add_polyline(self.draft, max(4.0, 10.0 / self.zoom))
        self.draft = []
        self.refresh_static_geometry()
        self.refresh_manual_widths()
        self.refresh_dynamic_overlay()
        self.refresh_metrics()

    def _start_lasso_canvas(self, event) -> None:
        self._remove_lasso_canvas()
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.lasso_canvas_item = self.canvas.create_line(x, y, x, y, fill="#facc15", width=2, tags="lasso_live")

    def begin_lasso(self) -> None:
        self.mode.set("select")
        self.lasso_active = True
        self.lasso = []
        self.tool_hint_var.set("按住左键圈住需要选择的中心线。")

    def refresh_edge_selection(self) -> None:
        for edge_id, item in self.edge_items.items():
            selected = edge_id in self.selected_edge_ids
            self.canvas.itemconfigure(
                item, fill="#F97316" if selected else "#22C55E",
                width=4 if selected else 2,
            )

    def update_dragged_node_items(self, node_id: int) -> None:
        z = self.zoom
        row, col = self.doc.nodes[node_id]
        item = self.node_items.get(node_id)
        if item is not None:
            self.canvas.coords(item, col * z - 3, row * z - 3, col * z + 3, row * z + 3)
        for edge_id in self.doc.incident_edges(node_id):
            edge_item = self.edge_items.get(edge_id)
            if edge_item is None:
                continue
            src, dst = (int(value) for value in self.doc.edges[edge_id])
            start, end = self.doc.nodes[src], self.doc.nodes[dst]
            self.canvas.coords(
                edge_item, start[1] * z, start[0] * z, end[1] * z, end[0] * z,
            )

    def keep_width_anchor(self) -> None:
        self.interval_measurement_id = None
        self.interval_draft = []
        self.tool_hint_var.set("最近一次测量保留为单点宽度锚点。")
        self.refresh_dynamic_overlay()

    def delete_latest_width(self) -> None:
        if not self.doc.manual_widths:
            return
        self.doc.checkpoint()
        deleted = self.doc.manual_widths.pop()
        if str(deleted.get("measurement_id", "")) == self.active_width_measurement_id:
            self.active_width_measurement_id = None
        self.saved_var.set("尚未保存")
        self.refresh_manual_widths()
        self.update_context_panel()

    def _text_input_focused(self, event=None) -> bool:
        widget = getattr(event, "widget", None) or self.root.focus_get()
        return isinstance(widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox, ttk.Spinbox))

    def set_mode_shortcut(self, event, mode: str) -> None:
        if self._text_input_focused(event):
            return
        self.mode.set(mode)
        self._mode_changed()

    def space_press(self, event) -> None:
        if not self._text_input_focused(event):
            self.space_pressed = True

    def space_release(self, _event) -> None:
        self.space_pressed = False
        self.space_panning = False

    def mouse_motion(self, event) -> None:
        self.mouse_status = self.source_point(event)
        if self.mode.get() == "measure_width":
            cursor = "hand2" if self.width_range_handle_at(*self.mouse_status) is not None else "crosshair"
            self.canvas.configure(cursor=cursor)
        elif self.mode.get() == "surface_add":
            self.canvas.configure(cursor="pencil")
        else:
            self.canvas.configure(cursor="")
        self.update_status()

    def start_surface_preview(self, row: float, col: float, add: bool) -> None:
        z = self.zoom
        self.surface_preview_item = self.canvas.create_line(
            col * z, row * z, col * z, row * z,
            fill="#38BDF8" if add else "#FB7185",
            width=max(2, self.surface_radius.get() * 2 * z), smooth=True,
            capstyle="round", tags=("dynamic",),
        )

    def extend_surface_preview(self, row: float, col: float) -> None:
        if self.surface_preview_item is None:
            return
        coords = list(self.canvas.coords(self.surface_preview_item))
        coords.extend((col * self.zoom, row * self.zoom))
        self.canvas.coords(self.surface_preview_item, *coords)

    def mouse_right_press(self, event) -> None:
        if self.mode.get() == "surface_add":
            row, col = self.source_point(event)
            self.doc.checkpoint()
            self.surface_right_active = True
            self.surface_stroke_points = [(row, col)]
            self.doc.paint_surface(row, col, self.surface_radius.get(), False)
            self.start_surface_preview(row, col, False)
        elif self.mode.get() == "draw":
            self.finish_drawing()

    def mouse_right_drag(self, event) -> None:
        if not self.surface_right_active:
            return
        row, col = self.source_point(event)
        self.doc.paint_surface(row, col, self.surface_radius.get(), False)
        self.surface_stroke_points.append((row, col))
        self.extend_surface_preview(row, col)

    def mouse_right_release(self, _event) -> None:
        if not self.surface_right_active:
            return
        self.surface_right_active = False
        self.surface_versions[self.document_index] += 1
        self.surface_stroke_points = []
        self.surface_preview_item = None
        self.refresh_background()
        self.refresh_dynamic_overlay()

    def _extend_lasso_canvas(self, event) -> None:
        if self.lasso_canvas_item is None:
            return
        coords = list(self.canvas.coords(self.lasso_canvas_item))
        coords.extend([self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)])
        self.canvas.coords(self.lasso_canvas_item, *coords)

    def _remove_lasso_canvas(self) -> None:
        if self.lasso_canvas_item is not None:
            self.canvas.delete(self.lasso_canvas_item)
            self.lasso_canvas_item = None

    def clear_selection(self) -> None:
        self.lasso = []
        self.selected_edge_ids.clear()
        self._remove_lasso_canvas()
        self.lasso_active = False
        self.refresh_edge_selection()
        self.update_context_panel()

    def delete_selected_edges(self) -> None:
        if not self.selected_edge_ids:
            messagebox.showinfo("没有选中线", "先切换到“自由圈选”，按住左键圈住需要删除的中心线。")
            return
        count = len(self.selected_edge_ids)
        if not messagebox.askyesno("批量删除中心线", f"确认删除圈选命中的 {count} 条中心线？"):
            return
        self.doc.checkpoint()
        self.doc.delete_edges(self.selected_edge_ids)
        self.selected_edge_ids.clear()
        self.refresh_static_geometry()
        self.refresh_manual_widths()
        self.refresh_metrics()
        self.update_context_panel()

    def cancel_drawing(self) -> None:
        self.draft = []
        self.width_draft = []
        self.width_drag_start = None
        self.width_drag_current = None
        self.interval_draft = []
        self.interval_measurement_id = None
        self.lasso = []
        self.lasso_active = False
        self._remove_lasso_canvas()
        self.width_preview = None
        self.refresh_dynamic_overlay()
        self.update_context_panel()

    def finish_width_measurement(self) -> None:
        if self.width_drag_start is None or self.width_drag_current is None:
            return
        start, end = self.width_drag_start, self.width_drag_current
        self.width_drag_start = None
        self.width_drag_current = None
        measurement = self.width_preview
        error = ""
        if measurement is None:
            measurement, error = create_normal_width_measurement(
                self.doc, start, end,
                max_centerline_distance=max(12.0, 24.0 / self.zoom),
            )
        self.width_preview = None
        if measurement is None:
            messagebox.showwarning("测宽未保存", error)
            self._mode_changed()
            return
        measurement, error = create_default_interval_width_measurement(self.doc, measurement)
        if measurement is None:
            messagebox.showwarning("测宽未保存", error)
            self._mode_changed()
            return
        if manual_width_interval_overlaps(self.doc.manual_widths, measurement):
            messagebox.showwarning(
                "人工宽度区间重叠",
                "新建范围与当前道路链上已有的人工宽度区间重叠。\n\n"
                "本次测宽已取消，请先调整已有区间的两端控制点。",
            )
            self.refresh_manual_widths()
            self.update_context_panel()
            return
        self.doc.checkpoint()
        self.doc.manual_widths.append(measurement)
        self.active_width_measurement_id = str(measurement.get("measurement_id", ""))
        self.saved_var.set("尚未保存")
        self.context_text.set(
            f"已测得 {float(measurement['width_units']):.2f} m。"
            "宽度已应用到当前道路段，拖动两端控制点可调整作用范围。"
        )
        self.refresh_manual_widths()
        self.refresh_dynamic_overlay()
        self.update_context_panel()

    def confirm_active_width(self) -> None:
        if self.active_width_measurement() is None:
            return
        self.active_width_measurement_id = None
        self.width_range_drag_handle = None
        self.width_range_drag_checkpoint = False
        self.context_text.set("人工宽度区间已保留在当前编辑状态。")
        self.refresh_manual_widths()
        self.update_context_panel()

    def cancel_active_width(self) -> None:
        active = self.active_width_measurement()
        if active is None:
            return
        self.doc.checkpoint()
        self.doc.manual_widths.remove(active)
        self.active_width_measurement_id = None
        self.width_range_drag_handle = None
        self.width_range_drag_checkpoint = False
        self.saved_var.set("尚未保存")
        self.context_text.set("已取消当前人工宽度区间。")
        self.refresh_manual_widths()
        self.update_context_panel()

    def begin_width_interval(self) -> None:
        measurement = next((
            row for row in reversed(self.doc.manual_widths)
            if str(row.get("source", "")) in {"manual_boundary_measurement", "manual_interval_width"}
        ), None)
        if measurement is None:
            messagebox.showinfo("请先测宽", "请先跨道路拖动一次，得到需要应用的人工宽度。")
            return
        self.interval_measurement_id = str(measurement.get("measurement_id", ""))
        self.interval_draft = []
        self.width_drag_start = None
        self.width_drag_current = None
        self.context_text.set("区间定宽：请在当前道路链中心线上选择区间起点。")
        self.refresh_dynamic_overlay()
        self.update_context_panel()

    def finish_width_interval(self) -> None:
        if self.interval_measurement_id is None or len(self.interval_draft) != 2:
            return
        index = next((
            item for item, row in enumerate(self.doc.manual_widths)
            if str(row.get("measurement_id", "")) == self.interval_measurement_id
        ), None)
        if index is None:
            self.interval_draft = []
            self.interval_measurement_id = None
            return
        interval, error = create_interval_width_measurement(
            self.doc, self.doc.manual_widths[index],
            self.interval_draft[0], self.interval_draft[1],
            max_centerline_distance=max(12.0, 24.0 / self.zoom),
        )
        if interval is None:
            messagebox.showwarning("区间未保存", error)
            self.interval_draft = []
            self.context_text.set("区间定宽：请重新选择同一连续道路链上的起点和终点。")
            self.refresh_dynamic_overlay()
            return
        self.doc.checkpoint()
        self.doc.manual_widths[index] = interval
        self.interval_draft = []
        self.interval_measurement_id = None
        self.saved_var.set("尚未保存")
        self.refresh_manual_widths()
        self.refresh_dynamic_overlay()
        self.update_context_panel()

    def add_candidate(self) -> None:
        candidate_id = self.candidate_var.get()
        if not candidate_id:
            return
        self.doc.checkpoint()
        if not self.doc.add_candidate(candidate_id, max(4.0, 10.0 / self.zoom)):
            self.doc.undo()
        self.refresh_static_geometry()
        self.refresh_manual_widths()
        self.refresh_metrics()

    def undo(self) -> None:
        if self.doc.undo():
            self.surface_versions[self.document_index] += 1
            self.draft = []
            self.lasso = []
            self.selected_edge_ids.clear()
            self.width_drag_start = None
            self.width_drag_current = None
            self.interval_draft = []
            self.interval_measurement_id = None
            self.width_range_drag_handle = None
            self.width_range_drag_checkpoint = False
            self.full_refresh()

    def redo(self) -> None:
        if self.doc.redo():
            self.surface_versions[self.document_index] += 1
            self.draft = []
            self.lasso = []
            self.selected_edge_ids.clear()
            self.width_drag_start = None
            self.width_drag_current = None
            self.interval_draft = []
            self.interval_measurement_id = None
            self.width_range_drag_handle = None
            self.width_range_drag_checkpoint = False
            self.full_refresh()

    def change_tile(self, _event=None) -> None:
        target = self.tile_var.get()
        self.document_index = next(index for index, doc in enumerate(self.documents) if doc.stem == target)
        self.draft = []
        self.lasso = []
        self.selected_edge_ids.clear()
        self.width_draft = []
        self.width_drag_start = None
        self.width_drag_current = None
        self.interval_draft = []
        self.interval_measurement_id = None
        self.active_width_measurement_id = None
        self.width_range_drag_handle = None
        self.width_range_drag_checkpoint = False
        self._last_background_view = None
        self._background_scrollregion = None
        self.full_refresh()
        self.fit_image()

    def show_topology(self) -> None:
        metrics = self.doc.topology_metrics()
        self.last_metrics = metrics
        if hasattr(self, "metric_vars"):
            for key, variable in self.metric_vars.items():
                variable.set(str(metrics.get(key, "--")))
        self.topology_dirty = False
        self.update_status()
        text = "\n".join(f"{key}: {value}" for key, value in metrics.items())
        warnings = []
        if metrics["isolated_node_count"]: warnings.append("存在孤立节点")
        if metrics["short_edge_count"]: warnings.append("存在小于2px的短边")
        if metrics["pending_candidate_count"]: warnings.append("仍有未加入候选；可保留为未采用")
        messagebox.showinfo("拓扑检查", text + ("\n\n提示：" + "；".join(warnings) if warnings else "\n\n未发现结构性错误。"))

    def node_all_crossings(self) -> None:
        if not messagebox.askyesno(
            "交叉线建路口",
            "这会把当前切片中所有几何交叉线切分为共享路口节点。\n\n"
            "如果影像中存在高架、下穿或不同层道路，请不要执行；应只拖动需要连接的端点到目标线。是否继续？",
        ):
            return
        self.doc.checkpoint()
        self.doc.node_intersections()
        self.refresh_static_geometry()
        self.refresh_manual_widths()
        self.refresh_metrics()

    def save_all(self, show_message: bool = True) -> None:
        self.edited_dir.mkdir(parents=True, exist_ok=True)
        if self.final_centerlines is not None and getattr(self.documents[0], "global_mode", False):
            manifest = save_global_document(
                self.documents[0], self.edited_dir,
                self.final_centerlines, self.final_surfaces,
            )
            (self.edited_dir / "edited_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            if hasattr(self, "saved_var"):
                self.saved_var.set("✓ 已保存全局最终中心线")
            if show_message:
                messagebox.showinfo(
                    "保存完成",
                    f"最终中心线已作为一个全局网络保存到：\n{self.edited_dir}\n\n"
                    f"受影响切片：{manifest['affected_tile_count']} / {manifest['tile_count']}。\n"
                    "返回主程序点击“应用编辑并重新生成结果”后，只会重新测算受影响窗口，"
                    "最终中心线直接采用当前全局编辑成果，并更新道路面和相邻期次变化。",
                )
            return
        manifest = {"editor": "builtin_geometry_editor_v2_final_fused", "tiles": {}}
        if self.final_centerlines is not None:
            manifest.update({
                "base_final_centerlines": str(self.final_centerlines),
                "base_final_centerlines_mtime_ns": self.final_centerlines.stat().st_mtime_ns,
                "base_final_surfaces": str(self.final_surfaces) if self.final_surfaces is not None else "",
                "base_final_surfaces_mtime_ns": self.final_surfaces.stat().st_mtime_ns if self.final_surfaces is not None else -1,
                "editing_scope": "period_final_fused_centerlines",
            })
        for doc in self.documents:
            doc.compact()
            graph_path = self.edited_dir / f"{doc.stem}_edited_graph.p"
            report_path = self.edited_dir / f"{doc.stem}_topology_report.json"
            save_graph(graph_path, [tuple(float(value) for value in point) for point in doc.nodes.tolist()], doc.edges.tolist())
            metrics = doc.topology_metrics()
            report_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            manual_width_path = self.edited_dir / f"{doc.stem}_manual_widths.json"
            manual_width_path.write_text(json.dumps(doc.manual_widths, indent=2, ensure_ascii=False), encoding="utf-8")
            manual_surface_add = self.edited_dir / f"{doc.stem}_manual_surface_add.png"
            manual_surface_remove = self.edited_dir / f"{doc.stem}_manual_surface_remove.png"
            cv2.imwrite(str(manual_surface_add), doc.surface_additions.astype(np.uint8) * 255)
            cv2.imwrite(str(manual_surface_remove), doc.surface_removals.astype(np.uint8) * 255)
            manifest["tiles"][doc.stem] = {
                "graph": str(graph_path),
                "topology_report": str(report_path), **metrics,
                "pending_candidate_ids": sorted(doc.candidates),
                "applied_surface_region_ids": sorted(doc.applied_surface_region_ids),
                "manual_widths": str(manual_width_path),
                "manual_width_count": len(doc.manual_widths),
                "manual_surface_add": str(manual_surface_add),
                "manual_surface_remove": str(manual_surface_remove),
                "manual_surface_added_px": int(np.count_nonzero(doc.surface_additions)),
                "manual_surface_removed_px": int(np.count_nonzero(doc.surface_removals)),
            }
        (self.edited_dir / "edited_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if hasattr(self, "saved_var"):
            self.saved_var.set("✓ 已保存")
        if show_message:
            messagebox.showinfo(
                "保存完成",
                f"已保存 {len(self.documents)} 个切片到：\n{self.edited_dir}\n\n"
                "请返回 SAMRoad 的“人工复核”步骤，点击“应用编辑并重新生成结果”。\n"
                "此操作只重建本期道路面、宽度、相邻变化和长时序成果，不会重跑模型推理。",
            )

    def close(self) -> None:
        if messagebox.askyesno("关闭编辑器", "关闭前是否保存所有修改？"):
            self.save_all(show_message=False)
        self.root.destroy()


def enable_windows_high_dpi() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass


def configure_tk_scaling(root: tk.Tk) -> None:
    dpi = float(root.winfo_fpixels("1i"))
    root.tk.call("tk", "scaling", max(1.0, min(dpi / 72.0, 3.0)))


def configure_ui_fonts(root: tk.Tk) -> None:
    for name, size, weight in (
        ("TkDefaultFont", 10, "normal"), ("TkTextFont", 10, "normal"),
        ("TkMenuFont", 10, "normal"), ("TkHeadingFont", 10, "bold"),
        ("TkCaptionFont", 10, "bold"), ("TkSmallCaptionFont", 9, "normal"),
    ):
        try:
            tkfont.nametofont(name, root=root).configure(
                family="Microsoft YaHei UI", size=size, weight=weight,
            )
        except tk.TclError:
            continue


def write_editor_signal(
    ready_file: Path | None, status: str, review_dir: Path, error: str = "",
) -> None:
    if ready_file is None:
        return
    ready_path = ready_file.expanduser().resolve()
    payload = {
        "status": status,
        "pid": os.getpid(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "review_dir": str(review_dir.expanduser().resolve()),
    }
    if error:
        payload["error"] = error
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ready_path.with_name(f".{ready_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    temporary.replace(ready_path)


def schedule_ready_signal(root: tk.Tk, ready_file: Path | None, review_dir: Path) -> None:
    """Write the launcher handshake only after Tk has mapped the editor window."""
    if ready_file is None:
        return

    def signal_when_viewable() -> None:
        try:
            if not root.winfo_exists():
                return
            if not root.winfo_viewable():
                root.after(50, signal_when_viewable)
                return
            write_editor_signal(ready_file, "ready", review_dir)
        except (OSError, tk.TclError) as exc:
            print(f"Unable to write geometry editor ready signal: {exc}", file=sys.stderr, flush=True)

    root.after(50, signal_when_viewable)


def build_loading_shell(root: tk.Tk) -> tuple[ttk.Frame, tk.StringVar]:
    """Build a lightweight editor-shaped shell before any GIS data is loaded."""
    root.title("道路实体变化智能检测与人工核验 · 中心线编辑")
    root.geometry("1100x720+80+60")
    root.minsize(900, 580)
    root.configure(background="#F3F0E8")
    shell = ttk.Frame(root, padding=0)
    shell.pack(fill="both", expand=True)
    header = tk.Frame(shell, background="#1D4034", height=72)
    header.pack(fill="x")
    header.pack_propagate(False)
    tk.Label(
        header, text="最终成果中心线编辑", background="#1D4034", foreground="#F8F5EC",
        font=("Microsoft YaHei UI", 15, "bold"),
    ).pack(side="left", padx=22, pady=18)
    body = ttk.Frame(shell, padding=24)
    body.pack(fill="both", expand=True)
    map_shell = tk.Frame(body, background="#162A23")
    map_shell.pack(side="left", fill="both", expand=True)
    status = tk.StringVar(value="正在读取人工编辑资料…")
    loading = tk.Frame(map_shell, background="#162A23")
    loading.place(relx=0.5, rely=0.5, anchor="center")
    tk.Label(
        loading, text="正在加载当前期次影像与道路数据…", background="#162A23",
        foreground="#F8F5EC", font=("Microsoft YaHei UI", 14, "bold"),
    ).pack(pady=(0, 10))
    tk.Label(
        loading, textvariable=status, background="#162A23", foreground="#AFC3B9",
        font=("Microsoft YaHei UI", 10),
    ).pack()
    side = ttk.Frame(body, width=250, padding=(18, 8))
    side.pack(side="left", fill="y")
    side.pack_propagate(False)
    ttk.Label(side, text="当前状态", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
    ttk.Label(
        side, text="数据加载完成后将自动解锁编辑工具。", wraplength=210,
    ).pack(anchor="w", pady=(10, 0))
    return shell, status


def load_documents_worker(
    messages: queue.Queue[tuple[str, object]], review_dir: Path, edited_dir: Path,
    final_centerlines: Path | None, final_surfaces: Path | None,
    timings: dict[str, float] | None = None,
) -> None:
    """Read/compute editor data in a worker without touching any Tk object."""
    try:
        documents = load_documents(
            review_dir, edited_dir, final_centerlines, final_surfaces,
            progress=lambda value: messages.put(("status", value)), timings=timings,
        )
        messages.put(("loaded", documents))
    except Exception:
        messages.put(("error", traceback.format_exc()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Built-in SAMRoad centerline and road-surface geometry editor.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--edited-dir", required=True)
    parser.add_argument("--final-centerlines", default="", help="Authoritative per-period fused centerline SHP.")
    parser.add_argument("--final-surfaces", default="", help="Authoritative per-period road-surface SHP.")
    parser.add_argument("--ready-file", default="", help="Launcher handshake written after the Tk window is viewable.")
    args = parser.parse_args()
    startup_started = time.perf_counter()
    startup_timings: dict[str, float] = {}
    enable_windows_high_dpi()
    stage_started = time.perf_counter()
    root = tk.Tk()
    startup_timings["tk_root_create"] = time.perf_counter() - stage_started
    configure_tk_scaling(root)
    configure_ui_fonts(root)
    review_dir = Path(args.review_dir)
    edited_dir = Path(args.edited_dir)
    final_centerlines = Path(args.final_centerlines) if args.final_centerlines else None
    final_surfaces = Path(args.final_surfaces) if args.final_surfaces else None
    ready_file = Path(args.ready_file) if args.ready_file else None
    stage_started = time.perf_counter()
    shell, loading_status = build_loading_shell(root)
    root.update()
    startup_timings["loading_shell_build_and_map"] = time.perf_counter() - stage_started
    startup_timings["window_first_visible"] = time.perf_counter() - startup_started
    messages: queue.Queue[tuple[str, object]] = queue.Queue()

    def show_loading_error(error: str) -> None:
        print(error, file=sys.stderr, flush=True)
        for child in root.winfo_children():
            child.destroy()
        panel = ttk.Frame(root, padding=30)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="人工编辑资料加载失败", font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w")
        ttk.Label(panel, text=error[-3000:], wraplength=900, justify="left").pack(anchor="w", pady=16)
        ttk.Button(panel, text="关闭编辑器", command=root.destroy).pack(anchor="w")
        try:
            write_editor_signal(ready_file, "failed", review_dir, error=error[-3000:])
        except OSError:
            pass

    def poll_loading() -> None:
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == "status":
                    loading_status.set(str(value))
                elif kind == "loaded":
                    try:
                        shell.destroy()
                        app = GeometryEditorApp(
                            root, review_dir, edited_dir, final_centerlines, final_surfaces,
                            documents=value, startup_timings=startup_timings,
                        )
                        root.update_idletasks()
                        stage_started = time.perf_counter()
                        app.fit_image()
                        root.update_idletasks()
                        startup_timings["initial_fit_and_render"] = time.perf_counter() - stage_started
                        period = review_dir.name
                        parts = list(review_dir.parts)
                        if "periods" in parts and parts.index("periods") + 1 < len(parts):
                            period = parts[parts.index("periods") + 1]
                        app.status_var.set(
                            f"数据已加载 · 中心线 {len(app.doc.edges)} 条 · 当前期次 {period}"
                        )
                        startup_timings["editor_usable"] = time.perf_counter() - startup_started
                        print(
                            "GEOMETRY_EDITOR_STARTUP_TIMINGS="
                            + json.dumps(startup_timings, ensure_ascii=False, sort_keys=True),
                            flush=True,
                        )
                        schedule_ready_signal(root, ready_file, review_dir)
                    except Exception:
                        show_loading_error(traceback.format_exc())
                    return
                elif kind == "error":
                    error = str(value)
                    show_loading_error(error)
                    return
        except queue.Empty:
            pass
        root.after(50, poll_loading)

    threading.Thread(
        target=load_documents_worker,
        args=(messages, review_dir, edited_dir, final_centerlines, final_surfaces, startup_timings),
        daemon=True,
    ).start()
    root.after(50, poll_loading)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
