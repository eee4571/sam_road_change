from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import json
import math
import sys
from contextlib import ExitStack
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
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from finalize_review_results import load_graph, save_graph  # noqa: E402
from chain_width_calculator import (  # noqa: E402
    build_road_chains,
    normalize_manual_width_measurements,
    project_point_to_road_chain,
)
from global_edit_utils import _graph_from_world_lines, _project_manual_widths  # noqa: E402
from review_geometry import accepted_surface_region_polylines  # noqa: E402


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
    ) -> None:
        self.stem = stem
        self.image = image
        self.nodes = np.asarray(nodes, dtype=np.float32).reshape(-1, 2)
        self.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        self.mask = (np.asarray(mask) > 0).astype(np.uint8)
        self.candidates = copy.deepcopy(candidates or {})
        self.applied_surface_region_ids = set(applied_surface_region_ids or set())
        self.manual_widths = normalize_manual_width_measurements(
            self.nodes, self.edges, copy.deepcopy(manual_widths or []),
        )
        self.surface_additions = (
            (np.asarray(surface_additions) > 0).astype(np.uint8)
            if surface_additions is not None else np.zeros_like(self.mask)
        )
        self.surface_removals = (
            (np.asarray(surface_removals) > 0).astype(np.uint8)
            if surface_removals is not None else np.zeros_like(self.mask)
        )
        self.undo_stack: list[EditorSnapshot] = []
        self.redo_stack: list[EditorSnapshot] = []

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
        for edge_id, (src, dst) in enumerate(self.edges.tolist()):
            projection, distance = point_segment_projection(point, self.nodes[src], self.nodes[dst])
            if distance <= tolerance and (best is None or distance < best[2]):
                best = (edge_id, projection, distance)
        return best

    def add_node(self, point: tuple[float, float]) -> int:
        self.nodes = np.vstack([self.nodes, np.asarray(point, dtype=np.float32)]) if len(self.nodes) else np.asarray([point], dtype=np.float32)
        return len(self.nodes) - 1

    def split_edge(self, edge_id: int, point: tuple[float, float]) -> int:
        src, dst = (int(value) for value in self.edges[edge_id])
        node_id = self.add_node(point)
        kept = np.delete(self.edges, edge_id, axis=0)
        self.edges = np.vstack([kept, np.asarray([[src, node_id], [node_id, dst]], dtype=np.int32)])
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
        incident = {edge_id for edge_id, edge in enumerate(self.edges.tolist()) if node_id in edge}
        best = None
        for edge_id, (src, dst) in enumerate(self.edges.tolist()):
            if edge_id in incident:
                continue
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
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.nodes)))
        graph.add_edges_from((int(src), int(dst)) for src, dst in self.edges.tolist())
        lengths = [float(np.linalg.norm(self.nodes[dst] - self.nodes[src])) for src, dst in self.edges.tolist()]
        return {
            "node_count": len(self.nodes), "edge_count": len(self.edges),
            "component_count": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
            "dangling_endpoint_count": sum(degree == 1 for _, degree in graph.degree()),
            "junction_count": sum(degree >= 3 for _, degree in graph.degree()),
            "isolated_node_count": sum(degree == 0 for _, degree in graph.degree()),
            "short_edge_count": sum(length < short_edge_px for length in lengths),
            "pending_candidate_count": len(self.candidates),
        }


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


def _final_centerline_documents(
    review_dir: Path, edited_dir: Path, final_centerlines: Path, final_surfaces: Path,
) -> list[GeometryDocument]:
    """Create one global editor document from the authoritative period product."""
    lines = gpd.read_file(final_centerlines)
    surfaces = gpd.read_file(final_surfaces)
    if lines.crs is None:
        raise ValueError(f"最终中心线缺少 CRS：{final_centerlines}")
    if surfaces.crs is None:
        raise ValueError(f"最终道路面缺少 CRS：{final_surfaces}")
    surfaces = surfaces if surfaces.crs == lines.crs else surfaces.to_crs(lines.crs)
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
    summary_rows = _review_summary_rows(review_dir)
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
    bounds = _image_union_bounds(image_paths, lines.crs)
    image, transform = _build_global_overview(image_paths, lines.crs, bounds)
    surface_shapes = [
        (geometry, 1) for geometry in surfaces.geometry
        if geometry is not None and not geometry.is_empty
    ]
    mask = rasterize(
        surface_shapes, out_shape=image.shape[:2], transform=transform,
        fill=0, all_touched=True, dtype="uint8",
    ) if surface_shapes else np.zeros(image.shape[:2], dtype=np.uint8)
    nodes, edges = _graph_from_world_lines(active_lines, transform, box(*bounds))
    surface_additions, surface_removals = load_surface_edits(edited_dir, "global", image.shape[:2])
    document = GeometryDocument(
        "全局最终中心线", image, nodes, edges, mask, candidates={},
        manual_widths=load_manual_widths(edited_dir, "global"),
        surface_additions=surface_additions, surface_removals=surface_removals,
    )
    document.global_mode = True
    document.global_transform = transform
    document.global_crs = lines.crs
    document.global_bounds = bounds
    document.global_summary_rows = summary_rows
    document.global_base_lines = lines.copy()
    document.global_base_surfaces = surfaces.copy()
    document.global_overview_scale = max(abs(float(transform.a)), abs(float(transform.e)))
    return [document]


def load_documents(
    review_dir: Path, edited_dir: Path, final_centerlines: Path | None = None,
    final_surfaces: Path | None = None,
) -> list[GeometryDocument]:
    if final_centerlines is not None:
        final_centerlines = final_centerlines.expanduser().resolve()
        if not final_centerlines.is_file():
            raise FileNotFoundError(f"找不到每期最终融合中心线：{final_centerlines}")
        if final_surfaces is None or not final_surfaces.is_file():
            raise FileNotFoundError(f"找不到每期最终道路面：{final_surfaces}")
        return _final_centerline_documents(review_dir, edited_dir, final_centerlines, final_surfaces)
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
        "select": "选择/拖动节点",
        "lasso": "自由圈选",
        "draw": "绘制中心线",
        "delete": "删除中心线",
        "measure_width": "手动测宽",
        "surface_add": "补画道路面",
        "surface_remove": "擦除道路面",
    }

    def __init__(
        self, root: tk.Tk, review_dir: Path, edited_dir: Path,
        final_centerlines: Path | None = None,
        final_surfaces: Path | None = None,
    ) -> None:
        self.root = root
        self.review_dir = review_dir
        self.edited_dir = edited_dir
        self.final_centerlines = final_centerlines.expanduser().resolve() if final_centerlines is not None else None
        self.final_surfaces = final_surfaces.expanduser().resolve() if final_surfaces is not None else None
        self.documents = load_documents(review_dir, edited_dir, self.final_centerlines, self.final_surfaces)
        self.document_index = 0
        self.mode = tk.StringVar(value="select")
        self.tile_var = tk.StringVar(value=self.documents[0].stem)
        self.status_var = tk.StringVar()
        self.zoom = 0.5
        self.photo = None
        self.drag_node: int | None = None
        self.drag_checkpoint = False
        self.draft: list[tuple[float, float]] = []
        self.lasso: list[tuple[float, float]] = []
        self.selected_edge_ids: set[int] = set()
        self.width_draft: list[tuple[float, float]] = []
        self.width_drag_start: tuple[float, float] | None = None
        self.width_drag_current: tuple[float, float] | None = None
        self.interval_draft: list[tuple[float, float]] = []
        self.interval_measurement_id: str | None = None
        self.surface_radius = tk.IntVar(value=12)
        self.surface_stroke_active = False
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
        self.build_ui()
        self.refresh()
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
        self.context_frame.pack(fill="x")
        self.context_text = tk.StringVar()
        ttk.Label(self.context_frame, textvariable=self.context_text, style="ToolbarHint.TLabel").pack(side="left")
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
        left_tools = ttk.Frame(
            workspace, padding=(7, 9), style="Panel.TFrame",
            width=round(150 * self.display_scale),
        )
        left_tools.pack(side="left", fill="y", padx=(0, 8))
        left_tools.pack_propagate(False)
        for value, label in self.MODES.items():
            ttk.Radiobutton(
                left_tools, text=label.replace("/拖动节点", "/移动"), variable=self.mode,
                value=value, style="Tool.TRadiobutton", command=self._mode_changed,
            ).pack(fill="x", pady=3)
        ttk.Label(left_tools, text="路面画笔半径(px)", style="Panel.TLabel").pack(fill="x", pady=(14, 3))
        ttk.Spinbox(left_tools, from_=2, to=100, textvariable=self.surface_radius, width=8).pack(fill="x")

        canvas_frame = ttk.Frame(workspace, style="Editor.TFrame")
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, background=palette["viewer"], highlightthickness=1, highlightbackground=palette["line_strong"])
        xscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        yscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
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
        topology_panel = ttk.LabelFrame(right_panel, text="拓扑概况", style="Panel.TLabelframe")
        topology_panel.pack(fill="x")
        self.metric_vars = {
            key: tk.StringVar(value="--")
            for key in ("node_count", "edge_count", "component_count", "dangling_endpoint_count")
        }
        for label, key in (
            ("节点", "node_count"), ("边", "edge_count"),
            ("连通分量", "component_count"), ("悬挂端点", "dangling_endpoint_count"),
        ):
            row = ttk.Frame(topology_panel, style="Panel.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, style="Panel.TLabel").pack(side="left")
            ttk.Label(row, textvariable=self.metric_vars[key], style="PanelTitle.TLabel").pack(side="right")
        ttk.Button(topology_panel, text="查看拓扑问题", command=self.show_topology).pack(fill="x", pady=(8, 0))

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
        self.canvas.bind("<B2-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))
        self.canvas.bind("<Button-3>", lambda _event: self.finish_drawing())
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-y>", lambda _event: self.redo())
        self.root.bind("<Control-s>", lambda _event: self.save_all())
        self.root.bind("<Return>", lambda _event: self.finish_drawing())
        self.root.bind("<Escape>", lambda _event: self.cancel_drawing())
        self._mode_changed()

    def _mode_changed(self) -> None:
        for widget in (
            self.context_finish, self.context_cancel, self.context_delete,
            self.context_clear, self.context_interval,
        ):
            widget.pack_forget()
        mode = self.mode.get()
        if mode != "measure_width":
            self.width_drag_start = None
            self.width_drag_current = None
            self.interval_draft = []
            self.interval_measurement_id = None
        if mode == "draw":
            self.context_text.set("绘制中心线：在影像上依次单击添加节点。")
            self.context_cancel.pack(side="right", padx=4)
            self.context_finish.pack(side="right")
        elif mode == "lasso":
            self.context_text.set("自由圈选：按住鼠标左键绘制选区，松开后完成选择。")
            self.context_clear.pack(side="right", padx=4)
            self.context_delete.pack(side="right")
        elif mode == "delete":
            self.context_text.set("删除中心线：单击需要删除的线段。")
        elif mode == "measure_width":
            self.context_text.set("手动测宽：按住左键从道路一侧拖到另一侧；松开后自动校正为道路法线。")
            self.context_cancel.pack(side="right", padx=4)
            self.context_interval.pack(side="right")
        elif mode == "surface_add":
            self.context_text.set("补画道路面：按住左键涂画遗漏路面；绿色为最终面，蓝色为本次补画。")
        elif mode == "surface_remove":
            self.context_text.set("擦除道路面：按住左键擦除错误路面；红色为本次删除范围。")
        else:
            self.context_text.set("选择/移动：单击节点并拖动；滚轮缩放，中键平移。")
        self.refresh()

    def zoom_by(self, factor: float) -> None:
        self.zoom = max(0.1, min(3.0, self.zoom * factor))
        self.refresh()

    def one_to_one(self) -> None:
        self.zoom = 1.0
        self.refresh()

    def source_point(self, event) -> tuple[float, float]:
        col = self.canvas.canvasx(event.x) / self.zoom
        row = self.canvas.canvasy(event.y) / self.zoom
        return float(np.clip(row, 0, self.doc.image.shape[0] - 1)), float(np.clip(col, 0, self.doc.image.shape[1] - 1))

    def fit_image(self) -> None:
        self.root.update_idletasks()
        width = max(300, self.canvas.winfo_width())
        height = max(300, self.canvas.winfo_height())
        self.zoom = max(0.1, min(3.0, min((width - 4) / self.doc.image.shape[1], (height - 4) / self.doc.image.shape[0])))
        self.refresh()

    def mouse_wheel(self, event) -> None:
        old = self.zoom
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self.zoom = max(0.1, min(3.0, self.zoom * factor))
        if abs(self.zoom - old) > 1e-6:
            self.refresh()

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
                for chain in build_road_chains(self.doc.nodes, self.doc.edges):
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
            preview, _error = create_normal_width_measurement(
                self.doc, self.width_drag_start, self.width_drag_current,
                max_centerline_distance=max(12.0, 24.0 / self.zoom),
            )
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

    def refresh(self) -> None:
        source = self.render_source()
        width = max(1, int(round(source.shape[1] * self.zoom)))
        height = max(1, int(round(source.shape[0] * self.zoom)))
        interpolation = cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(source, (width, height), interpolation=interpolation)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, width, height))
        metrics = self.doc.topology_metrics()
        if hasattr(self, "metric_vars"):
            for key, variable in self.metric_vars.items():
                variable.set(str(metrics.get(key, "--")))
        self.status_var.set(
            f"{self.doc.stem}   │   节点 {metrics['node_count']}   │   边 {metrics['edge_count']}   │   "
            f"连通分量 {metrics['component_count']}   │   ⚠ 悬挂端点 {metrics['dangling_endpoint_count']}"
            f"                                      当前工具：{self.MODES[self.mode.get()]}   │   {int(round(self.zoom * 100))}%"
        )

    def mouse_press(self, event) -> None:
        row, col = self.source_point(event)
        mode = self.mode.get()
        tolerance = max(3.0, 8.0 / self.zoom)
        if mode == "select":
            node_id = self.doc.nearest_node(row, col, tolerance)
            if node_id is not None:
                self.doc.checkpoint()
                self.drag_checkpoint = True
                self.drag_node = node_id
        elif mode == "lasso":
            self.lasso = [(row, col)]
            self.selected_edge_ids.clear()
            self._start_lasso_canvas(event)
        elif mode == "draw":
            self.draft.append((row, col))
            self.refresh()
        elif mode == "delete":
            nearest = self.doc.nearest_edge(row, col, tolerance)
            if nearest is not None:
                self.doc.checkpoint()
                self.doc.delete_edge_at(row, col, tolerance)
                self.refresh()
        elif mode == "measure_width":
            if self.interval_measurement_id is not None:
                self.interval_draft.append((row, col))
                if len(self.interval_draft) == 2:
                    self.finish_width_interval()
                else:
                    self.context_text.set("区间定宽：已选择起点，请在同一连续道路链上选择终点。")
                    self.refresh()
            else:
                self.width_drag_start = (row, col)
                self.width_drag_current = (row, col)
                self.refresh()
        elif mode in {"surface_add", "surface_remove"}:
            self.doc.checkpoint()
            self.surface_stroke_active = True
            self.doc.paint_surface(row, col, self.surface_radius.get(), mode == "surface_add")
            self.refresh()

    def mouse_drag(self, event) -> None:
        row, col = self.source_point(event)
        if self.mode.get() == "select" and self.drag_node is not None:
            self.doc.nodes[self.drag_node] = (row, col)
            self.refresh()
        elif self.mode.get() == "lasso" and self.lasso:
            last = self.lasso[-1]
            if math.hypot(row - last[0], col - last[1]) >= max(1.0, 3.0 / self.zoom):
                self.lasso.append((row, col))
                self._extend_lasso_canvas(event)
        elif self.mode.get() == "measure_width" and self.width_drag_start is not None:
            self.width_drag_current = (row, col)
            preview, _error = create_normal_width_measurement(
                self.doc, self.width_drag_start, self.width_drag_current,
                max_centerline_distance=max(12.0, 24.0 / self.zoom),
            )
            if preview is not None:
                self.context_text.set(f"人工测宽：{float(preview['width_units']):.2f} m")
            self.refresh()
        elif self.mode.get() in {"surface_add", "surface_remove"} and self.surface_stroke_active:
            self.doc.paint_surface(row, col, self.surface_radius.get(), self.mode.get() == "surface_add")
            self.refresh()

    def mouse_release(self, event) -> None:
        self.surface_stroke_active = False
        if self.mode.get() == "measure_width" and self.width_drag_start is not None:
            self.width_drag_current = self.source_point(event)
            self.finish_width_measurement()
        if self.drag_node is not None:
            self.doc.snap_moved_node(self.drag_node, max(3.0, 8.0 / self.zoom))
            self.drag_node = None
            self.drag_checkpoint = False
            self.refresh()
        if self.mode.get() == "lasso" and self.lasso:
            if len(self.lasso) >= 3:
                self.selected_edge_ids = self.doc.edges_in_polygon(self.lasso)
            self._remove_lasso_canvas()
            self.refresh()

    def finish_drawing(self) -> None:
        if len(self.draft) >= 2:
            self.doc.checkpoint()
            self.doc.add_polyline(self.draft, max(4.0, 10.0 / self.zoom))
        self.draft = []
        self.refresh()

    def _start_lasso_canvas(self, event) -> None:
        self._remove_lasso_canvas()
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.lasso_canvas_item = self.canvas.create_line(x, y, x, y, fill="#facc15", width=2, tags="lasso_live")

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
        self.refresh()

    def delete_selected_edges(self) -> None:
        if not self.selected_edge_ids:
            messagebox.showinfo("没有选中线", "先切换到“自由圈选”，按住左键圈住需要删除的中心线。")
            return
        count = len(self.selected_edge_ids)
        if not messagebox.askyesno("批量删除中心线", f"确认删除圈选命中的 {count} 条中心线？"):
            return
        self.doc.checkpoint()
        self.doc.delete_edges(self.selected_edge_ids)
        self.clear_selection()

    def cancel_drawing(self) -> None:
        self.draft = []
        self.width_draft = []
        self.width_drag_start = None
        self.width_drag_current = None
        self.interval_draft = []
        self.interval_measurement_id = None
        if self.mode.get() == "lasso":
            self.lasso = []
            self.selected_edge_ids.clear()
            self._remove_lasso_canvas()
        self.refresh()

    def finish_width_measurement(self) -> None:
        if self.width_drag_start is None or self.width_drag_current is None:
            return
        start, end = self.width_drag_start, self.width_drag_current
        self.width_drag_start = None
        self.width_drag_current = None
        measurement, error = create_normal_width_measurement(
            self.doc, start, end,
            max_centerline_distance=max(12.0, 24.0 / self.zoom),
        )
        if measurement is None:
            messagebox.showwarning("测宽未保存", error)
            self._mode_changed()
            return
        self.doc.checkpoint()
        self.doc.manual_widths.append(measurement)
        self.saved_var.set("尚未保存")
        self._mode_changed()

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
        self.refresh()

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
            self.refresh()
            return
        self.doc.checkpoint()
        self.doc.manual_widths[index] = interval
        self.interval_draft = []
        self.interval_measurement_id = None
        self.saved_var.set("尚未保存")
        self._mode_changed()

    def add_candidate(self) -> None:
        candidate_id = self.candidate_var.get()
        if not candidate_id:
            return
        self.doc.checkpoint()
        if not self.doc.add_candidate(candidate_id, max(4.0, 10.0 / self.zoom)):
            self.doc.undo()
        self.refresh()

    def undo(self) -> None:
        if self.doc.undo():
            self.draft = []
            self.lasso = []
            self.selected_edge_ids.clear()
            self.width_drag_start = None
            self.width_drag_current = None
            self.interval_draft = []
            self.interval_measurement_id = None
            self.refresh()

    def redo(self) -> None:
        if self.doc.redo():
            self.draft = []
            self.lasso = []
            self.selected_edge_ids.clear()
            self.width_drag_start = None
            self.width_drag_current = None
            self.interval_draft = []
            self.interval_measurement_id = None
            self.refresh()

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
        self.fit_image()

    def show_topology(self) -> None:
        metrics = self.doc.topology_metrics()
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
        self.refresh()

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Built-in SAMRoad centerline and road-surface geometry editor.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--edited-dir", required=True)
    parser.add_argument("--final-centerlines", default="", help="Authoritative per-period fused centerline SHP.")
    parser.add_argument("--final-surfaces", default="", help="Authoritative per-period road-surface SHP.")
    args = parser.parse_args()
    enable_windows_high_dpi()
    root = tk.Tk()
    configure_tk_scaling(root)
    configure_ui_fonts(root)
    try:
        GeometryEditorApp(
            root, Path(args.review_dir), Path(args.edited_dir),
            Path(args.final_centerlines) if args.final_centerlines else None,
            Path(args.final_surfaces) if args.final_surfaces else None,
        )
    except Exception as exc:
        messagebox.showerror("几何编辑器启动失败", str(exc))
        raise
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
