from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from chain_width_calculator import apply_manual_width_constraints
from final_width_calculator import (
    FinalWidthConfig,
    FinalWidthRequest,
    load_width_calculator,
    validate_width_result,
)
from molra_centerline_width import graph_topology_metrics
from width_surface_reconstruction import (
    WidthSurfaceConfig,
    _regular_corridor_widths,
    reconstruct_surface_from_widths,
)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def apply_manual_width_overrides(
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_records: list[dict],
    edge_widths: list[dict],
    manual_width_path: Path | None,
    pixel_size: float,
    samples: list[dict] | None = None,
    segments: list[dict] | None = None,
) -> int:
    """Apply local point anchors and strict intervals on road-chain positions."""
    if manual_width_path is None or not manual_width_path.is_file() or not len(edges):
        return 0
    try:
        payload = json.loads(manual_width_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    measurements = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    if not measurements:
        return 0
    return apply_manual_width_constraints(
        nodes, edges, edge_records, edge_widths, measurements, pixel_size, samples, segments,
    )


def candidate_points(candidate: dict) -> list[tuple[float, float]]:
    try:
        raw = json.loads(candidate.get("polyline_points_json", ""))
        points = [(float(point[0]), float(point[1])) for point in raw if len(point) >= 2]
    except (TypeError, ValueError, json.JSONDecodeError):
        points = []
    if len(points) >= 2:
        return points
    return [
        (as_float(candidate, "start_row"), as_float(candidate, "start_col")),
        (as_float(candidate, "end_row"), as_float(candidate, "end_col")),
    ]


def find_one(output_dir: Path, pattern: str) -> Path:
    matches = sorted(output_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {output_dir}")
    return matches[0]


def load_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as file:
        graph = pickle.load(file)

    node_to_idx: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    for node, neighbors in graph.items():
        node_key = tuple(int(round(v)) for v in node)
        node_to_idx.setdefault(node_key, len(node_to_idx))
        for neighbor in neighbors:
            neighbor_key = tuple(int(round(v)) for v in neighbor)
            node_to_idx.setdefault(neighbor_key, len(node_to_idx))
            edges.append((node_to_idx[node_key], node_to_idx[neighbor_key]))

    nodes = np.zeros((len(node_to_idx), 2), dtype=np.float32)
    for node_key, idx in node_to_idx.items():
        nodes[idx] = node_key
    unique_edges = sorted({tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]})
    return nodes, np.array(unique_edges, dtype=np.int32).reshape(-1, 2)


def save_graph(path: Path, nodes_rc: list[tuple[int, int]], edges: list[tuple[int, int]]) -> None:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {
        node: [] for node in nodes_rc
    }
    for src_idx, dst_idx in edges:
        src = nodes_rc[src_idx]
        dst = nodes_rc[dst_idx]
        if src == dst:
            continue
        graph.setdefault(src, [])
        graph.setdefault(dst, [])
        if dst not in graph[src]:
            graph[src].append(dst)
        if src not in graph[dst]:
            graph[dst].append(src)
    with open(path, "wb") as file:
        pickle.dump(graph, file)


def decisions_by_key(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {(row.get("stem", ""), row.get("item_type", ""), row.get("item_id", "")): row for row in rows}


def decision_for(decisions: dict[tuple[str, str, str], dict], stem: str, item_type: str, item_id: str) -> dict:
    row = decisions.get((stem, item_type, item_id))
    if row is not None:
        return row
    return decisions.get(("", item_type, item_id), {})


def exact_file(output_dir: Path, stem: str, suffix: str) -> Path:
    path = output_dir / f"{stem}_{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path


def nearest_node_idx(
    nodes: list[tuple[int, int]],
    row: float,
    col: float,
    max_dist: float,
    allowed_indices: list[int] | None = None,
) -> int | None:
    indices = allowed_indices if allowed_indices is not None else list(range(len(nodes)))
    if not indices:
        return None
    arr = np.asarray([nodes[idx] for idx in indices], dtype=np.float32)
    dist2 = (arr[:, 0] - row) ** 2 + (arr[:, 1] - col) ** 2
    idx = int(np.argmin(dist2))
    if float(np.sqrt(dist2[idx])) <= max_dist:
        return indices[idx]
    return None


def add_or_snap_node(
    nodes: list[tuple[int, int]],
    row: float,
    col: float,
    snap_px: float,
    allowed_snap_indices: list[int] | None = None,
) -> int:
    idx = nearest_node_idx(nodes, row, col, snap_px, allowed_snap_indices)
    if idx is not None:
        return idx
    key = (int(round(row)), int(round(col)))
    try:
        return nodes.index(key)
    except ValueError:
        nodes.append(key)
        return len(nodes) - 1


def add_exact_node(nodes: list[tuple[int, int]], row: float, col: float) -> int:
    key = (int(round(row)), int(round(col)))
    try:
        return nodes.index(key)
    except ValueError:
        nodes.append(key)
        return len(nodes) - 1


def prepare_candidate_topology_attachment(
    candidate: dict,
    points: list[tuple[float, float]],
    final_nodes: list[tuple[int, int]],
    final_edges: list[tuple[int, int]],
    final_edge_keys: set[tuple[int, int]],
    edge_records: list[dict],
) -> tuple[list[tuple[float, float]], dict[int, int]]:
    """Return a candidate path with a real graph attachment at its endpoint.

    The scorer records endpoint-to-edge and endpoint-to-node evidence.  This
    finalizer step makes that evidence topological: edge attachment creates a
    junction and splits the target edge, rather than merely drawing an almost
    touching candidate line.
    """
    connection_type = str(candidate.get("connection_type", ""))
    try:
        endpoint_position = int(float(candidate.get("connection_endpoint_position", "")))
    except (TypeError, ValueError):
        return points, {}
    if endpoint_position not in {0, 1} or len(points) < 2:
        return points, {}
    target_idx: int | None = None
    target_point: tuple[float, float] | None = None
    if connection_type == "endpoint_to_node":
        # Graph pickle iteration order is not a stable cross-process node ID.
        # Resolve the scorer's audited coordinate in the final graph instead.
        try:
            node_row = float(candidate.get("connection_node_row", ""))
            node_col = float(candidate.get("connection_node_col", ""))
            target_idx = nearest_node_idx(final_nodes, node_row, node_col, 2.0)
        except (TypeError, ValueError):
            target_idx = None
        # Backward compatibility for pre-audit CSVs: only use the legacy index
        # when a coordinate is unavailable, never when it contradicts one.
        if target_idx is None and not str(candidate.get("connection_node_row", "")).strip():
            try:
                legacy_idx = int(float(candidate.get("connection_node_idx", "")))
            except (TypeError, ValueError):
                legacy_idx = -1
            target_idx = legacy_idx if 0 <= legacy_idx < len(final_nodes) else None
        if target_idx is not None:
            target_point = tuple(float(value) for value in final_nodes[target_idx])
    elif connection_type == "endpoint_to_edge":
        try:
            projection = (
                float(candidate.get("connection_projection_row", "")),
                float(candidate.get("connection_projection_col", "")),
            )
        except (TypeError, ValueError):
            return points, {}
        best_edge_index, best_distance = -1, float("inf")
        for edge_index, (src_idx, dst_idx) in enumerate(final_edges):
            start = np.asarray(final_nodes[src_idx], dtype=np.float32)
            end = np.asarray(final_nodes[dst_idx], dtype=np.float32)
            vector = end - start
            length2 = float(np.dot(vector, vector))
            ratio = 0.0 if length2 <= 0 else float(np.clip(np.dot(np.asarray(projection) - start, vector) / length2, 0.0, 1.0))
            projected = start + ratio * vector
            distance = float(np.linalg.norm(projected - np.asarray(projection)))
            if 0.02 <= ratio <= 0.98 and distance < best_distance:
                best_edge_index, best_distance = edge_index, distance
        if best_edge_index < 0 or best_distance > 2.0:
            return points, {}
        src_idx, dst_idx = final_edges[best_edge_index]
        target_idx = add_exact_node(final_nodes, projection[0], projection[1])
        if target_idx not in {src_idx, dst_idx}:
            old_key = tuple(sorted((src_idx, dst_idx)))
            final_edge_keys.discard(old_key)
            final_edges[best_edge_index] = (src_idx, target_idx)
            final_edges.append((target_idx, dst_idx))
            final_edge_keys.update({tuple(sorted((src_idx, target_idx))), tuple(sorted((target_idx, dst_idx)))})
            # Keep the original provenance on both noded portions.  Records are
            # ordered against final_edges before width measurement below.
            original = next(
                (record for record in edge_records if record.get("final_status") != "removed_by_review" and tuple(sorted((int(record["src_idx"]), int(record["dst_idx"])))) == old_key),
                None,
            )
            if original is not None:
                original["dst_idx"] = target_idx
                original["dst_row"], original["dst_col"] = final_nodes[target_idx]
                original["length_px"] = float(np.hypot(final_nodes[src_idx][0] - final_nodes[target_idx][0], final_nodes[src_idx][1] - final_nodes[target_idx][1]))
                split_record = dict(original)
                split_record["src_idx"], split_record["dst_idx"] = target_idx, dst_idx
                split_record["src_row"], split_record["src_col"] = final_nodes[target_idx]
                split_record["dst_row"], split_record["dst_col"] = final_nodes[dst_idx]
                split_record["length_px"] = float(np.hypot(final_nodes[dst_idx][0] - final_nodes[target_idx][0], final_nodes[dst_idx][1] - final_nodes[target_idx][1]))
                split_record["topology_noding"] = "split_for_endpoint_to_edge"
                edge_records.append(split_record)
        target_point = tuple(float(value) for value in final_nodes[target_idx])
    if target_idx is None or target_point is None:
        return points, {}
    attached = list(points)
    if endpoint_position == 0:
        attached.insert(0, target_point)
        return attached, {0: target_idx}
    attached.append(target_point)
    return attached, {len(attached) - 1: target_idx}


def finalization_candidate_points(
    candidate: dict,
    final_nodes: list[tuple[int, int]],
    final_edges: list[tuple[int, int]],
    final_edge_keys: set[tuple[int, int]],
    edge_records: list[dict],
) -> tuple[list[tuple[float, float]], dict[int, int]]:
    """Materialize only audited, already-contacting surface attachments.

    Surface skeletons otherwise retain their traced geometry exactly: a stale
    score, remote coordinate, or legacy graph index is never permission to
    pull an endpoint across empty space.  The producer stamps the audit fields
    only after it has verified a road-surface path or an exact surface contact.
    """
    points = candidate_points(candidate)
    if str(candidate.get("candidate_type", "")).startswith("surface_skeleton"):
        audit = str(candidate.get("surface_attachment_audit", ""))
        kind = str(candidate.get("surface_attachment_kind", ""))
        try:
            endpoint_position = int(float(candidate.get("surface_attachment_endpoint_position", "")))
        except (TypeError, ValueError):
            return points, {}
        if audit not in {"accepted_surface_mask_path", "exact_surface_contact"} or endpoint_position not in {0, 1}:
            return points, {}
        evidence_mode = str(candidate.get("surface_attachment_evidence_mode", ""))
        support = as_float(candidate, "surface_attachment_surface_support_ratio", -1.0)
        probability_supported = (
            evidence_mode == "high_collinearity_road_probability"
            and as_float(candidate, "surface_attachment_road_probability_mean", 0.0) >= 0.50
            and as_float(candidate, "surface_attachment_road_probability_q25", 0.0) >= 0.10
        )
        if support < 0.95 and not probability_supported:
            return points, {}
        normalized = dict(candidate)
        if kind == "surface_skeleton_to_graph_endpoint":
            row = as_float(candidate, "surface_attachment_node_row", float("nan"))
            col = as_float(candidate, "surface_attachment_node_col", float("nan"))
            if not np.isfinite(row) or not np.isfinite(col):
                return points, {}
            if float(np.hypot(points[endpoint_position][0] - row, points[endpoint_position][1] - col)) > 2.0:
                return points, {}
            normalized.update({
                "connection_type": "endpoint_to_node",
                "connection_endpoint_position": endpoint_position,
                "connection_node_row": row,
                "connection_node_col": col,
            })
        elif kind == "surface_skeleton_to_graph_edge":
            row = as_float(candidate, "surface_attachment_projection_row", float("nan"))
            col = as_float(candidate, "surface_attachment_projection_col", float("nan"))
            if not np.isfinite(row) or not np.isfinite(col):
                return points, {}
            if float(np.hypot(points[endpoint_position][0] - row, points[endpoint_position][1] - col)) > 2.0:
                return points, {}
            normalized.update({
                "connection_type": "endpoint_to_edge",
                "connection_endpoint_position": endpoint_position,
                "connection_projection_row": row,
                "connection_projection_col": col,
            })
        else:
            return points, {}
        return prepare_candidate_topology_attachment(
            normalized, points, final_nodes, final_edges, final_edge_keys, edge_records
        )
    return prepare_candidate_topology_attachment(
        candidate, points, final_nodes, final_edges, final_edge_keys, edge_records
    )


def load_image(summary: dict, fallback: Path) -> np.ndarray | None:
    candidates = []
    image_value = summary.get("image", "")
    if image_value:
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        candidates.append(image_path)
    candidates.append(fallback)
    for path in candidates:
        image = read_image(path, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return None


def read_image(path: Path, flags: int) -> np.ndarray | None:
    """Read through bytes because OpenCV imread is unreliable on Unicode paths."""
    path = Path(path)
    if not path.is_file():
        return None
    return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flags)


def write_image(path: Path, image: np.ndarray) -> bool:
    """Write through bytes for the same Windows Unicode-path guarantee."""
    suffix = Path(path).suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    Path(path).write_bytes(encoded.tobytes())
    return True


def load_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    mask = read_image(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = (mask > 0).astype(np.uint8)
    if shape is not None and mask.shape != shape:
        return None
    return mask


def thinning(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    if binary.size == 0 or np.count_nonzero(binary) == 0:
        return np.zeros_like(binary)
    if hasattr(cv2, "ximgproc"):
        return (cv2.ximgproc.thinning(binary * 255) > 0).astype(np.uint8)

    image = binary.copy()
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.count_nonzero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(image, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        image = cv2.erode(image, element)
    return skeleton


def skeleton_neighbors(point: tuple[int, int], point_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, col = point
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            candidate = (row + dr, col + dc)
            if candidate in point_set:
                neighbors.append(candidate)
    return neighbors


def farthest_path_from(
    start: tuple[int, int],
    point_set: set[tuple[int, int]],
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None], dict[tuple[int, int], float]]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    distance = {start: 0.0}
    queue = deque([start])
    while queue:
        point = queue.popleft()
        for neighbor in skeleton_neighbors(point, point_set):
            if neighbor in parent:
                continue
            parent[neighbor] = point
            distance[neighbor] = distance[point] + float(np.hypot(neighbor[0] - point[0], neighbor[1] - point[1]))
            queue.append(neighbor)
    farthest = max(distance, key=distance.get)
    return farthest, parent, distance


def trace_path(
    end: tuple[int, int],
    parent: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]]:
    path = []
    current: tuple[int, int] | None = end
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def longest_skeleton_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(skeleton)
    if ys.size < 2:
        return []
    point_set = {(int(r), int(c)) for r, c in zip(ys.tolist(), xs.tolist())}
    endpoints = [point for point in point_set if len(skeleton_neighbors(point, point_set)) <= 1]
    start = endpoints[0] if endpoints else next(iter(point_set))
    farthest, _, _ = farthest_path_from(start, point_set)
    other, parent, _ = farthest_path_from(farthest, point_set)
    return trace_path(other, parent)


def resample_path(path: list[tuple[int, int]], step_px: float) -> list[tuple[float, float]]:
    if len(path) < 2:
        return [(float(path[0][0]), float(path[0][1]))] if path else []
    step_px = max(2.0, float(step_px))
    points = [(float(row), float(col)) for row, col in path]
    sampled = [points[0]]
    carry = 0.0
    prev = points[0]
    for current in points[1:]:
        segment_len = float(np.hypot(current[0] - prev[0], current[1] - prev[1]))
        if segment_len <= 0:
            prev = current
            continue
        remaining = segment_len
        start = prev
        while carry + remaining >= step_px:
            ratio = (step_px - carry) / remaining
            new_point = (
                start[0] + (current[0] - start[0]) * ratio,
                start[1] + (current[1] - start[1]) * ratio,
            )
            sampled.append(new_point)
            start = new_point
            remaining = float(np.hypot(current[0] - start[0], current[1] - start[1]))
            carry = 0.0
        carry += remaining
        prev = current
    if float(np.hypot(sampled[-1][0] - points[-1][0], sampled[-1][1] - points[-1][1])) > 1.0:
        sampled.append(points[-1])
    return sampled


def component_polyline(
    labels: np.ndarray,
    region: dict,
    step_px: float,
) -> list[tuple[float, float]]:
    region_id = as_int(region, "region_id", -1)
    x = as_int(region, "bbox_x")
    y = as_int(region, "bbox_y")
    w = max(1, as_int(region, "bbox_w"))
    h = max(1, as_int(region, "bbox_h"))
    component = (labels[y:y + h, x:x + w] == region_id).astype(np.uint8)
    skeleton = thinning(component)
    path = longest_skeleton_path(skeleton)
    if len(path) < 2:
        path = longest_skeleton_path(component)
    if len(path) < 2:
        return []
    sampled = resample_path(path, step_px)
    return [(row + y, col + x) for row, col in sampled]


def component_skeleton_edges(
    labels: np.ndarray,
    region: dict,
    step_px: float,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    region_id = as_int(region, "region_id", -1)
    x = as_int(region, "bbox_x")
    y = as_int(region, "bbox_y")
    w = max(1, as_int(region, "bbox_w"))
    h = max(1, as_int(region, "bbox_h"))
    component = (labels[y:y + h, x:x + w] == region_id).astype(np.uint8)
    skeleton = thinning(component)
    if np.count_nonzero(skeleton) < 2:
        skeleton = component

    ys, xs = np.nonzero(skeleton)
    point_set = {(int(r), int(c)) for r, c in zip(ys.tolist(), xs.tolist())}
    if len(point_set) < 2:
        return [], []

    vertices = {
        point
        for point in point_set
        if len(skeleton_neighbors(point, point_set)) != 2
    }
    if not vertices:
        vertices.add(next(iter(point_set)))

    paths: list[list[tuple[int, int]]] = []
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(sorted((a, b)))

    for vertex in list(vertices):
        for neighbor in skeleton_neighbors(vertex, point_set):
            key = edge_key(vertex, neighbor)
            if key in visited_edges:
                continue
            path = [vertex, neighbor]
            visited_edges.add(key)
            previous = vertex
            current = neighbor
            while current not in vertices:
                next_points = [p for p in skeleton_neighbors(current, point_set) if p != previous]
                if not next_points:
                    break
                next_point = next_points[0]
                next_key = edge_key(current, next_point)
                if next_key in visited_edges:
                    break
                path.append(next_point)
                visited_edges.add(next_key)
                previous, current = current, next_point
            if len(path) >= 2:
                paths.append(path)

    coord_to_idx: dict[tuple[int, int], int] = {}
    nodes: list[tuple[float, float]] = []
    edges: list[tuple[int, int]] = []

    def add_local_node(point: tuple[float, float]) -> int:
        key = (int(round(point[0])), int(round(point[1])))
        if key not in coord_to_idx:
            coord_to_idx[key] = len(nodes)
            nodes.append((float(key[0] + y), float(key[1] + x)))
        return coord_to_idx[key]

    for path in paths:
        sampled = resample_path(path, step_px)
        if len(sampled) < 2:
            continue
        local_indices = [add_local_node(point) for point in sampled]
        for src_idx, dst_idx in zip(local_indices[:-1], local_indices[1:]):
            if src_idx != dst_idx:
                edges.append((src_idx, dst_idx))

    unique_edges = sorted({tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]})
    return nodes, unique_edges


def draw_viz(
    image: np.ndarray,
    road_mask: np.ndarray | None,
    surface_only_mask: np.ndarray | None,
    nodes: list[tuple[int, int]],
    edge_records: list[dict],
    candidate_rows: list[dict],
    decisions: dict[tuple[str, str, str], dict],
    stem: str,
    out_path: Path,
    color_mode: str = "width",
) -> None:
    canvas = image.copy()
    overlay = canvas.copy()
    if road_mask is not None:
        overlay[road_mask > 0] = (40, 180, 255)
    if surface_only_mask is not None:
        overlay[surface_only_mask > 0] = (255, 110, 0)
    if road_mask is not None or surface_only_mask is not None:
        canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)

    removed = [row for row in edge_records if row["final_status"] == "removed_by_review"]
    kept = [
        row for row in edge_records
        if row["final_status"] != "removed_by_review"
        and row["source"] in {"samroad", "manual_edited"}
    ]
    added = [row for row in edge_records if row["source"] in {"review_added_candidate", "auto_added_gap", "auto_added_surface"}]

    for row in removed:
        p0 = (int(row["src_col"]), int(row["src_row"]))
        p1 = (int(row["dst_col"]), int(row["dst_row"]))
        cv2.line(canvas, p0, p1, (0, 0, 255), 1, lineType=cv2.LINE_AA)

    for row in kept:
        if color_mode == "source":
            color = (0, 145, 255) if row.get("source") == "manual_edited" else (0, 210, 0)
        else:
            width_source = row.get("optimized_width_source", "")
            status = row.get("final_status", "")
            if row.get("source") == "manual_edited":
                color = (0, 145, 255)
            elif width_source.startswith("review_neighbor"):
                color = (0, 255, 255)
            elif status in {"surface_missing", "partial_surface"}:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
        p0 = (int(row["src_col"]), int(row["src_row"]))
        p1 = (int(row["dst_col"]), int(row["dst_row"]))
        cv2.line(canvas, p0, p1, color, 2, lineType=cv2.LINE_AA)

    for row in added:
        if color_mode == "source":
            color = {
                "auto_added_gap": (255, 220, 0),
                "auto_added_surface": (255, 0, 255),
                "review_added_candidate": (255, 120, 0),
            }.get(row.get("source", ""), (255, 0, 0))
        else:
            color = (255, 0, 0)
        p0 = (int(row["src_col"]), int(row["src_row"]))
        p1 = (int(row["dst_col"]), int(row["dst_row"]))
        cv2.line(canvas, p0, p1, color, 2, lineType=cv2.LINE_AA)

    if color_mode != "source":
        for candidate in candidate_rows:
            decision = decision_for(decisions, stem, "candidate_centerline", str(candidate.get("candidate_id", ""))).get("decision", "")
            effective_decision = decision or candidate.get("auto_decision", "")
            if effective_decision != "accept":
                continue
            points = candidate_points(candidate)
            pts = np.asarray([[int(round(col)), int(round(row))] for row, col in points], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    if color_mode == "source":
        legend = [
            ("SAMRoad original", (0, 210, 0)),
            ("surface skeleton added", (255, 0, 255)),
            ("gap connection added", (255, 220, 0)),
            ("manual centerline", (0, 145, 255)),
            ("manual accepted candidate", (255, 120, 0)),
        ]
        panel_width = min(canvas.shape[1] - 12, 255)
        panel_height = 12 + 22 * len(legend)
        cv2.rectangle(canvas, (6, 6), (6 + panel_width, 6 + panel_height), (25, 25, 25), -1)
        for index, (label, color) in enumerate(legend):
            y = 23 + index * 22
            cv2.line(canvas, (15, y), (45, y), color, 3, lineType=cv2.LINE_AA)
            cv2.putText(canvas, label, (52, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(out_path, canvas)


def finalize_one(
    args: argparse.Namespace,
    output_dir: Path,
    final_dir: Path,
    summary_path: Path,
    decisions: dict[tuple[str, str, str], dict],
    decisions_path: Path,
) -> dict:
    finalize_started = time.perf_counter()
    with open(summary_path, "r", encoding="utf-8") as file:
        summary = json.load(file)
    stem = summary_path.name.removesuffix("_summary.json")

    edited_dir_value = str(getattr(args, "edited_dir", "") or "").strip()
    edited_dir = Path(edited_dir_value) if edited_dir_value else None
    edited_graph_path = edited_dir / f"{stem}_edited_graph.p" if edited_dir else None
    reconstructed_mask_path = edited_dir / f"{stem}_reconstructed_road_surface.png" if edited_dir else None
    prepared_graph_path = Path(summary.get("prepared_graph", summary.get("graph", "")))
    if not prepared_graph_path.is_absolute():
        prepared_graph_path = Path.cwd() / prepared_graph_path
    geometry_edited = bool(
        edited_graph_path
        and edited_graph_path.is_file()
        and (
            not prepared_graph_path.is_file()
            or edited_graph_path.stat().st_mtime_ns >= prepared_graph_path.stat().st_mtime_ns
        )
    )
    if geometry_edited and not (reconstructed_mask_path and reconstructed_mask_path.is_file()):
        raise RuntimeError(
            f"Edited centerlines exist for {stem}, but the reconstructed road surface is missing. "
            "Run GUI step 3 (global stitching and centerline-constrained surface reconstruction) first."
        )
    graph_path = edited_graph_path if geometry_edited else Path(summary.get("graph", ""))
    if not graph_path.is_absolute():
        graph_path = Path.cwd() / graph_path
    nodes_np, edges_np = load_graph(graph_path)
    original_nodes = [(int(round(r)), int(round(c))) for r, c in nodes_np.tolist()]
    final_nodes = list(original_nodes)

    candidate_analysis_started = time.perf_counter()
    edge_width_rows = read_csv(exact_file(output_dir, stem, "edge_widths.csv"))
    candidate_rows = [] if geometry_edited else read_csv(exact_file(output_dir, stem, "candidate_centerlines.csv"))
    surface_rows = read_csv(exact_file(output_dir, stem, "surface_only_regions.csv"))
    surface_by_id = {str(row.get("region_id", "")): row for row in surface_rows}
    surface_only_mask = load_mask(exact_file(output_dir, stem, "surface_only.png"))
    if reconstructed_mask_path and reconstructed_mask_path.is_file():
        clean_mask = load_mask(reconstructed_mask_path)
    else:
        clean_mask = load_mask(exact_file(output_dir, stem, "molra_clean_mask.png"))
    road_probability = None
    probability_path = output_dir / f"{stem}_road_probability.png"
    if probability_path.is_file():
        road_probability = read_image(probability_path, cv2.IMREAD_GRAYSCALE)
        if road_probability is not None:
            road_probability = road_probability.astype(np.float32) / 255.0
    surface_labels = None
    if surface_only_mask is not None:
        _, surface_labels = cv2.connectedComponents(surface_only_mask.astype(np.uint8), connectivity=8)

    candidate_decisions_by_region: dict[str, list[str]] = {}
    for candidate in candidate_rows:
        region_id = str(candidate.get("region_id", ""))
        if not region_id:
            continue
        decision = decision_for(
            decisions, stem, "candidate_centerline", str(candidate.get("candidate_id", ""))
        ).get("decision", "")
        if decision:
            candidate_decisions_by_region.setdefault(region_id, []).append(decision)
    rejected_surface_region_ids: set[str] = set()
    for region_id in surface_by_id:
        region_decision = decision_for(decisions, stem, "surface_only_region", region_id).get("decision", "")
        candidate_decisions = candidate_decisions_by_region.get(region_id, [])
        if region_decision in {"reject", "mark_nonroad"} or (
            candidate_decisions
            and all(value in {"reject", "mark_nonroad"} for value in candidate_decisions)
        ):
            rejected_surface_region_ids.add(region_id)
    # Once centerlines are authoritatively edited, step 3 reconstruction supersedes
    # earlier item-review surface decisions. Those decisions only prepare unedited runs.
    if not geometry_edited and clean_mask is not None and surface_labels is not None:
        clean_mask = clean_mask.copy()
        for region_id in rejected_surface_region_ids:
            try:
                clean_mask[surface_labels == int(region_id)] = 0
            except ValueError:
                continue

    final_edges: list[tuple[int, int]] = []
    final_edge_keys: set[tuple[int, int]] = set()
    edge_records: list[dict] = []
    original_edge_to_record_idx: dict[int, int] = {}
    for edge_id, (src_idx, dst_idx) in enumerate(edges_np.tolist()):
        width_row = edge_width_rows[edge_id] if not geometry_edited and edge_id < len(edge_width_rows) else {}
        # Existing centerlines are retained unless authoritative geometry
        # editing removed them. Legacy quick-review edge decisions are ignored.
        decision = ""
        removed = False
        r0, c0 = final_nodes[int(src_idx)]
        r1, c1 = final_nodes[int(dst_idx)]
        # Width is intentionally initialized only after final topology exists.
        width = 0.0
        width_source = "pending_final_measurement"
        final_status = "geometry_kept"
        record = {
            **width_row,
            "final_edge_id": len(edge_records),
            "source": (
                "manual_edited" if geometry_edited
                else str(width_row.get("line_source", "samroad") or "samroad")
            ),
            "line_feature_id": (
                f"edited:{edge_id}" if geometry_edited
                else f"{str(width_row.get('line_source', 'samroad') or 'samroad')}:{edge_id}"
            ),
            "original_edge_id": edge_id,
            "candidate_id": "",
            "src_idx": int(src_idx),
            "dst_idx": int(dst_idx),
            "src_row": r0,
            "src_col": c0,
            "dst_row": r1,
            "dst_col": c1,
            "review_decision": decision or "not_required",
            "final_status": "removed_by_review" if removed else final_status,
            "optimized_width_px": width,
            "optimized_width_source": width_source,
        }
        original_edge_to_record_idx[edge_id] = len(edge_records)
        edge_records.append(record)
        if not removed:
            edge_key = tuple(sorted((int(src_idx), int(dst_idx))))
            final_edges.append((int(src_idx), int(dst_idx)))
            final_edge_keys.add(edge_key)

    accepted_candidate_count = 0
    accepted_candidate_edge_count = 0
    auto_candidate_count = 0
    auto_candidate_edge_count = 0
    auto_surface_count = 0
    auto_surface_edge_count = 0
    rejected_candidate_count = 0
    retained_original_node_ids = sorted({node_idx for edge in final_edges for node_idx in edge})
    candidate_region_node_ids: dict[str, list[int]] = {}
    for candidate in candidate_rows:
        candidate_id = str(candidate.get("candidate_id", ""))
        region_id = str(candidate.get("region_id", ""))
        candidate_decision = decision_for(decisions, stem, "candidate_centerline", candidate_id).get("decision", "")
        region_decision = decision_for(decisions, stem, "surface_only_region", region_id).get("decision", "")
        auto_decision = candidate.get("auto_decision", "")
        final_decision = candidate_decision or region_decision or auto_decision
        if final_decision in {"reject", "keep_unknown", "mark_nonroad"}:
            rejected_candidate_count += 1
            continue
        if final_decision != "accept":
            continue

        is_auto_candidate = auto_decision == "accept" and not candidate_decision and not region_decision
        is_surface_candidate = str(candidate.get("candidate_type", "")).startswith("surface_skeleton")
        is_auto_gap = is_auto_candidate and candidate.get("candidate_type") == "endpoint_gap"
        is_auto_surface = is_auto_candidate and is_surface_candidate
        # Materialize a scored graph attachment before sampling the candidate.
        # In particular, endpoint-to-edge candidates node the target edge and
        # include the audited connector segment to the new junction.
        attached_points, forced_endpoint_nodes = finalization_candidate_points(
            candidate,
            final_nodes,
            final_edges,
            final_edge_keys,
            edge_records,
        )
        skeleton_nodes = resample_path(attached_points, args.added_centerline_step_px)
        skeleton_edges = list(zip(range(len(skeleton_nodes) - 1), range(1, len(skeleton_nodes))))
        if len(skeleton_nodes) < 2 or not skeleton_edges:
            rejected_candidate_count += 1
            continue
        degrees = [0 for _ in skeleton_nodes]
        for src_local, dst_local in skeleton_edges:
            degrees[src_local] += 1
            degrees[dst_local] += 1

        node_indices = []
        region_snap_ids = candidate_region_node_ids.setdefault(region_id, [])
        for point_idx, (row, col) in enumerate(skeleton_nodes):
            if is_surface_candidate:
                # add_exact_node reuses only the identical rounded pixel.  It
                # does not pull a skeleton endpoint across empty space toward
                # a nearby road, which is what caused the visible random links.
                node_idx = add_exact_node(final_nodes, row, col)
                node_indices.append(node_idx)
                if node_idx not in retained_original_node_ids and node_idx not in region_snap_ids:
                    region_snap_ids.append(node_idx)
            elif degrees[point_idx] == 1:
                node_idx = forced_endpoint_nodes.get(point_idx)
                if node_idx is None:
                    node_idx = add_or_snap_node(
                        final_nodes,
                        row,
                        col,
                        args.snap_additions_px,
                        allowed_snap_indices=retained_original_node_ids + region_snap_ids,
                    )
                node_indices.append(node_idx)
                if node_idx not in retained_original_node_ids and node_idx not in region_snap_ids:
                    region_snap_ids.append(node_idx)
            else:
                node_indices.append(add_exact_node(final_nodes, row, col))
        if len(set(node_indices)) < 2:
            continue

        added_any = False
        for segment_idx, (src_local, dst_local) in enumerate(skeleton_edges):
            src_idx = node_indices[src_local]
            dst_idx = node_indices[dst_local]
            if src_idx == dst_idx:
                continue
            edge_key = tuple(sorted((src_idx, dst_idx)))
            if edge_key in final_edge_keys:
                continue
            final_edges.append((src_idx, dst_idx))
            final_edge_keys.add(edge_key)
            r0, c0 = final_nodes[src_idx]
            r1, c1 = final_nodes[dst_idx]
            edge_records.append(
                {
                    "final_edge_id": len(edge_records),
                    "source": "auto_added_gap" if is_auto_gap else ("auto_added_surface" if is_auto_surface else "review_added_candidate"),
                    "line_feature_id": f"candidate:{candidate_id}",
                    "original_edge_id": "",
                    "candidate_id": candidate_id,
                    "candidate_segment_id": segment_idx,
                    "src_idx": src_idx,
                    "dst_idx": dst_idx,
                    "src_row": r0,
                    "src_col": c0,
                    "dst_row": r1,
                    "dst_col": c1,
                    "length_px": float(np.hypot(r1 - r0, c1 - c0)),
                    "review_decision": "auto_accept" if is_auto_candidate else "accept",
                    "final_status": "auto_added_gap" if is_auto_gap else ("auto_added_surface" if is_auto_surface else "added_by_review"),
                    "confidence": candidate.get("confidence", ""),
                    "auto_score": candidate.get("auto_score", ""),
                    "auto_rule": candidate.get("auto_rule", ""),
                    "hard_veto": candidate.get("hard_veto", ""),
                    "hard_veto_reasons": candidate.get("hard_veto_reasons", ""),
                    "connection_type": candidate.get("connection_type", ""),
                    "long_road_evidence": candidate.get("long_road_evidence", ""),
                    "long_road_evidence_reason": candidate.get("long_road_evidence_reason", ""),
                    "optimized_width_px": 0.0,
                    "optimized_width_source": "pending_final_measurement",
                }
            )
            added_any = True
            accepted_candidate_edge_count += 1
            if is_auto_gap:
                auto_candidate_edge_count += 1
            if is_auto_surface:
                auto_surface_edge_count += 1
        if not added_any:
            continue
        accepted_candidate_count += 1
        if is_auto_gap:
            auto_candidate_count += 1
        if is_auto_surface:
            auto_surface_count += 1

    estimated_count = 0

    # The original graph can be noded while candidates are accepted.  Reorder
    # active provenance records to the actual graph order before final IDs and
    # width measurement are assigned.
    active_by_key = {
        tuple(sorted((int(row["src_idx"]), int(row["dst_idx"])))): row
        for row in edge_records
        if row["final_status"] != "removed_by_review"
    }
    if all(tuple(sorted(edge)) in active_by_key for edge in final_edges):
        removed_records = [row for row in edge_records if row["final_status"] == "removed_by_review"]
        edge_records = [active_by_key[tuple(sorted(edge))] for edge in final_edges] + removed_records

    # Remove orphan nodes and make final_edge_id match the actual optimized graph.
    used_node_ids = sorted({node_idx for edge in final_edges for node_idx in edge})
    node_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_node_ids)}
    final_nodes = [final_nodes[old_idx] for old_idx in used_node_ids]
    final_edges = [(node_remap[src], node_remap[dst]) for src, dst in final_edges]
    active_edge_id = 0
    for record_id, row in enumerate(edge_records):
        row["record_id"] = record_id
        if row["final_status"] == "removed_by_review":
            row["final_edge_id"] = ""
            continue
        row["src_idx"] = node_remap[int(row["src_idx"])]
        row["dst_idx"] = node_remap[int(row["dst_idx"])]
        row["final_edge_id"] = active_edge_id
        active_edge_id += 1

    if active_edge_id != len(final_edges):
        raise RuntimeError(f"Final edge-table mismatch: active_rows={active_edge_id}, graph_edges={len(final_edges)}")
    candidate_analysis_seconds = time.perf_counter() - candidate_analysis_started
    final_width_measurement_started = time.perf_counter()
    pixel_size = as_float(summary, "pixel_size", 1.0)
    for row in edge_records:
        row["optimized_width_units"] = as_float(row, "optimized_width_px") * pixel_size

    final_nodes_np = np.asarray(final_nodes, dtype=np.float32).reshape(-1, 2)
    final_edges_np = np.asarray(final_edges, dtype=np.int32).reshape(-1, 2)
    active_records = [row for row in edge_records if row["final_status"] != "removed_by_review"]
    width_request = FinalWidthRequest(
        nodes_rc=final_nodes_np,
        edges=final_edges_np,
        road_surface=clean_mask,
        config=FinalWidthConfig(
            pixel_size=pixel_size,
            sample_step_px=as_float(summary, "sample_step_px", 20.0),
            normal_step_px=as_float(summary, "normal_step_px", 1.0),
            max_search_px=as_float(summary, "max_search_px", 120.0),
            snap_radius_px=int(as_float(summary, "snap_radius_px", 8.0)),
            junction_buffer_px=as_float(summary, "junction_buffer_px", 30.0),
            border_margin_px=int(as_float(summary, "border_margin_px", 2.0)),
            max_snap_distance_px=as_float(summary, "max_snap_distance_px", 4.0),
            max_asymmetry_ratio=as_float(summary, "max_asymmetry_ratio", 0.65),
            width_change_ratio=as_float(summary, "width_change_ratio", 0.35),
            width_change_min_samples=max(1, int(as_float(summary, "width_change_min_samples", 3.0))),
            min_edge_coverage=as_float(summary, "min_edge_coverage", 0.6),
            short_gap_px=as_float(summary, "short_gap_px", 80.0),
            max_width_cv=as_float(summary, "max_width_cv", 0.5),
            outlier_mad_scale=as_float(summary, "outlier_mad_scale", 3.5),
            hybrid_agreement_ratio=as_float(summary, "hybrid_agreement_ratio", 0.35),
        ),
        edge_metadata=tuple(dict(row) for row in active_records),
    )
    width_calculator = load_width_calculator(getattr(args, "width_calculator", ""))
    width_result = width_calculator(width_request)
    validate_width_result(width_result, len(active_records))
    for row, measured in zip(active_records, width_result.edge_widths):
        row["optimized_width_px"] = float(measured.get("width_px", 0.0) or 0.0)
        row["optimized_width_units"] = float(
            measured.get("width_units", row["optimized_width_px"] * pixel_size) or 0.0
        )
        row["optimized_width_source"] = str(measured.get("source", width_result.algorithm))
        row["optimized_quality_grade"] = str(measured.get("quality_grade", "C"))
        row["final_status"] = str(measured.get("status", "width_unresolved"))
    manual_width_path = edited_dir / f"{stem}_manual_widths.json" if edited_dir else None
    manual_width_edge_count = 0
    # The production pipeline now has one authoritative surface policy:
    # fixed-width Buffer geometry derived from regularized centerline widths.
    # Keep the exported centerline attributes consistent with the regular
    # Buffer surface: every edge in one uninterrupted chain shares a
    # single robust median width.
    corridor_reliable = np.asarray(
        [
            as_float(row, "optimized_width_px") > 0
            and str(row.get("optimized_quality_grade", "")).upper() != "C"
            and str(row.get("source", "")) not in {"auto_added_gap", "review_added_candidate"}
            for row in active_records
        ],
        dtype=bool,
    )
    corridor_widths, _ = _regular_corridor_widths(
            final_nodes_np,
            final_edges_np,
            np.asarray([as_float(row, "optimized_width_px") for row in active_records], dtype=np.float32),
            WidthSurfaceConfig(
                regular_corridor_cosine=max(0.0, min(1.0, as_float(vars(args), "surface_regular_corridor_cosine", 0.94))),
                regular_corridor_width_ratio=max(0.0, as_float(vars(args), "surface_regular_corridor_width_ratio", 0.35)),
            ),
            corridor_reliable,
            active_records,
    )
    for edge_id, chain_width in enumerate(corridor_widths.tolist()):
        if chain_width <= 0:
            continue
        active_records[int(edge_id)]["optimized_width_px"] = float(chain_width)
        active_records[int(edge_id)]["optimized_width_units"] = float(chain_width) * pixel_size
        width_result.edge_widths[int(edge_id)]["width_px"] = float(chain_width)
        width_result.edge_widths[int(edge_id)]["width_units"] = float(chain_width) * pixel_size
    # Manual boundary measurements are authoritative and therefore applied
    # after automatic corridor regularisation, immediately before surface rebuild.
    manual_width_edge_count = apply_manual_width_overrides(
        final_nodes_np, final_edges_np, active_records, width_result.edge_widths,
        manual_width_path, pixel_size, width_result.samples, width_result.segments,
    )
    final_width_samples = width_result.samples
    final_width_segments = width_result.segments
    final_width_measurement_seconds = time.perf_counter() - final_width_measurement_started

    surface_reconstruction_started = time.perf_counter()
    width_surface = None
    if clean_mask is not None and len(final_edges_np):
        width_surface = reconstruct_surface_from_widths(
            clean_mask.shape,
            final_nodes_np,
            final_edges_np,
            width_result.edge_widths,
            final_width_samples,
            reference_surface=clean_mask,
            road_probability=road_probability,
            edge_metadata=active_records,
            config=WidthSurfaceConfig(
                min_width_px=as_float(vars(args), "surface_min_width_px", 1.0),
                width_scale=as_float(vars(args), "surface_width_scale", 1.0),
                close_kernel=max(1, as_int(vars(args), "surface_close_kernel", 3)),
                preserve_reference_surface=False,
                centerline_support_margin_px=max(0, as_int(vars(args), "surface_support_margin_px", 4)),
                min_unsupported_area_px=max(1, as_int(vars(args), "surface_min_unsupported_area_px", 12)),
                chain_width_max_deviation_ratio=max(0.0, as_float(vars(args), "surface_chain_width_deviation_ratio", 0.25)),
                continuity_close_kernel=max(1, as_int(vars(args), "surface_continuity_close_kernel", 7)),
                continuity_max_gap_px=max(1, as_int(vars(args), "surface_continuity_max_gap_px", 8)),
                boundary_smooth_sigma_px=max(0.0, as_float(vars(args), "surface_boundary_smooth_sigma_px", 1.5)),
                regular_surface=True,
                regular_corridor_cosine=max(0.0, min(1.0, as_float(vars(args), "surface_regular_corridor_cosine", 0.94))),
                regular_corridor_width_ratio=max(0.0, as_float(vars(args), "surface_regular_corridor_width_ratio", 0.35)),
            ),
        )
        if np.any(width_surface.surface):
            clean_mask = width_surface.surface
        # The rendered surface can resolve grade-C/missing widths from trusted
        # neighbouring branches.  Export the same resolved width on the
        # centerline attributes, otherwise a user sees a narrow line attribute
        # next to a different final buffer polygon.
        rendered_widths = width_surface.metadata.get("resolved_widths_px", [])
        if len(rendered_widths) == len(active_records):
            for edge_id, rendered_width in enumerate(rendered_widths):
                width_value = float(rendered_width)
                active_records[edge_id]["optimized_width_px"] = width_value
                active_records[edge_id]["optimized_width_units"] = width_value * pixel_size
                width_result.edge_widths[edge_id]["width_px"] = width_value
                width_result.edge_widths[edge_id]["width_units"] = width_value * pixel_size
                if (
                    str(active_records[edge_id].get("optimized_quality_grade", "")).upper() == "C"
                    and not str(active_records[edge_id].get("optimized_width_source", "")).startswith("manual_")
                ):
                    active_records[edge_id]["optimized_width_source"] = "junction_aware_surface_fallback"
        write_image(final_dir / f"{stem}_width_surface_added.png", width_surface.added.astype(np.uint8) * 255)
        write_image(final_dir / f"{stem}_width_surface_removed.png", width_surface.removed.astype(np.uint8) * 255)

    # Surface painting/erasing is the last authoritative edit. Apply it after
    # automatic width-buffer reconstruction so later stages cannot undo what
    # the user saw and saved in the editor.
    manual_surface_add_path = edited_dir / f"{stem}_manual_surface_add.png" if edited_dir else None
    manual_surface_remove_path = edited_dir / f"{stem}_manual_surface_remove.png" if edited_dir else None
    manual_surface_add = load_mask(manual_surface_add_path) if manual_surface_add_path and manual_surface_add_path.is_file() else None
    manual_surface_remove = load_mask(manual_surface_remove_path) if manual_surface_remove_path and manual_surface_remove_path.is_file() else None
    if clean_mask is not None:
        clean_mask = (clean_mask > 0).astype(np.uint8)
        if manual_surface_add is not None and manual_surface_add.shape == clean_mask.shape:
            clean_mask[manual_surface_add > 0] = 1
        if manual_surface_remove is not None and manual_surface_remove.shape == clean_mask.shape:
            clean_mask[manual_surface_remove > 0] = 0
    surface_reconstruction_seconds = time.perf_counter() - surface_reconstruction_started

    sample_fields = list(final_width_samples[0].keys()) if final_width_samples else []
    write_csv(final_dir / f"{stem}_optimized_width_samples.csv", final_width_samples, sample_fields)
    segment_fields = list(final_width_segments[0].keys()) if final_width_segments else []
    write_csv(final_dir / f"{stem}_optimized_width_segments.csv", final_width_segments, segment_fields)

    edge_fields = [
        "record_id",
        "final_edge_id",
        "source",
        "line_feature_id",
        "original_edge_id",
        "candidate_id",
        "candidate_segment_id",
        "src_idx",
        "dst_idx",
        "src_row",
        "src_col",
        "dst_row",
        "dst_col",
        "length_px",
        "review_decision",
        "final_status",
        "confidence",
        "auto_score",
        "auto_rule",
        "hard_veto",
        "hard_veto_reasons",
        "connection_type",
        "long_road_evidence",
        "long_road_evidence_reason",
        "coverage_ratio",
        "median_width_px",
        "final_width_px",
        "width_source",
        "optimized_width_px",
        "optimized_width_units",
        "optimized_width_source",
        "optimized_quality_grade",
        "topology_impact",
        "conflict_type",
        "review_priority_score",
        "review_priority",
        "mean_road_probability",
        "mean_centerline_probability",
        "auto_retained",
        "line_source",
        "recovery_score",
        "center_conf",
        "surface_conf",
        "recovery_reason",
        "qa_state",
        "recovery_id",
    ]
    write_csv(final_dir / f"{stem}_optimized_edges.csv", edge_records, edge_fields)

    candidate_out = []
    for candidate in candidate_rows:
        candidate_id = str(candidate.get("candidate_id", ""))
        region_id = str(candidate.get("region_id", ""))
        candidate_decision = decision_for(decisions, stem, "candidate_centerline", candidate_id).get("decision", "")
        region_decision = decision_for(decisions, stem, "surface_only_region", region_id).get("decision", "")
        auto_decision = candidate.get("auto_decision", "")
        final_candidate_decision = candidate_decision or region_decision or auto_decision
        candidate_out.append(
            {
                **candidate,
                "candidate_review_decision": candidate_decision,
                "region_review_decision": region_decision,
                "final_action": "auto_add_centerline" if not candidate_decision and not region_decision and auto_decision == "accept" else ("add_centerline" if final_candidate_decision == "accept" else ("defer" if final_candidate_decision in {"defer", "skip", "review"} else "discard")),
            }
        )
    write_csv(final_dir / f"{stem}_optimized_candidate_centerlines.csv", candidate_out, list(candidate_out[0].keys()) if candidate_out else [])

    surface_out = []
    for row in surface_rows:
        region_id = str(row.get("region_id", ""))
        region_decision = decision_for(decisions, stem, "surface_only_region", region_id).get("decision", "")
        surface_out.append(
            {
                **row,
                "review_decision": region_decision,
                "final_surface_status": "removed_as_nonroad" if region_id in rejected_surface_region_ids else "retained_as_road",
            }
        )
    write_csv(final_dir / f"{stem}_optimized_surface_regions.csv", surface_out, list(surface_out[0].keys()) if surface_out else [])
    if clean_mask is not None:
        write_image(final_dir / f"{stem}_optimized_road_surface.png", clean_mask.astype(np.uint8) * 255)

    save_graph(final_dir / f"{stem}_optimized_graph.p", final_nodes, final_edges)

    review_demo = exact_file(output_dir, stem, "review_demo.png")
    image = load_image(summary, review_demo)
    if image is not None:
        draw_viz(
            image=image,
            road_mask=clean_mask,
            surface_only_mask=surface_only_mask,
            nodes=final_nodes,
            edge_records=edge_records,
            candidate_rows=candidate_rows,
            decisions=decisions,
            stem=stem,
            out_path=final_dir / f"{stem}_optimized_viz.png",
        )
        draw_viz(
            image=image,
            road_mask=clean_mask,
            surface_only_mask=surface_only_mask,
            nodes=final_nodes,
            edge_records=edge_records,
            candidate_rows=candidate_rows,
            decisions=decisions,
            stem=stem,
            out_path=final_dir / f"{stem}_fusion_comparison.png",
            color_mode="source",
        )

    base_sources = {"samroad", "weak_recovered", "manual_edited"}
    removed_original_count = sum(1 for row in edge_records if row["source"] in base_sources and row["final_status"] == "removed_by_review")
    kept_original_count = sum(1 for row in edge_records if row["source"] in base_sources and row["final_status"] != "removed_by_review")
    unresolved_added_count = sum(
        1
        for row in edge_records
        if row["source"] in {"review_added_candidate", "auto_added_gap", "auto_added_surface"}
        and as_float(row, "optimized_width_px") <= 0
    )
    auto_surface_rule_counts: dict[str, int] = {}
    auto_surface_veto_count = 0
    auto_long_surface_count = 0
    for candidate in candidate_rows:
        if candidate.get("candidate_type") != "surface_skeleton":
            continue
        if str(candidate.get("hard_veto", "")).strip().lower() in {"1", "true", "yes"}:
            auto_surface_veto_count += 1
        if candidate.get("auto_decision") == "accept":
            rule = str(candidate.get("auto_rule", "unclassified"))
            auto_surface_rule_counts[rule] = auto_surface_rule_counts.get(rule, 0) + 1
            if str(candidate.get("long_road_evidence", "")).strip().lower() in {"1", "true", "yes"}:
                auto_long_surface_count += 1
    optimized_summary = {
        "source_output_dir": str(output_dir),
        "image": summary.get("image", ""),
        "geometry_edited": geometry_edited,
        "edited_graph": str(edited_graph_path.resolve()) if geometry_edited else "",
        "reconstructed_road_surface": str(reconstructed_mask_path.resolve()) if reconstructed_mask_path and reconstructed_mask_path.is_file() else "",
        "surface_added": str((edited_dir / f"{stem}_surface_added.png").resolve()) if edited_dir and (edited_dir / f"{stem}_surface_added.png").is_file() else "",
        "surface_removed": str((edited_dir / f"{stem}_surface_removed.png").resolve()) if edited_dir and (edited_dir / f"{stem}_surface_removed.png").is_file() else "",
        "surface_uncertain": str((edited_dir / f"{stem}_surface_uncertain.png").resolve()) if edited_dir and (edited_dir / f"{stem}_surface_uncertain.png").is_file() else "",
        "review_decisions": str(decisions_path),
        "original_node_count": int(nodes_np.shape[0]),
        "original_edge_count": int(edges_np.shape[0]),
        "optimized_node_count": len(final_nodes),
        "optimized_edge_count": len(final_edges),
        "kept_original_edge_count": kept_original_count,
        "removed_original_edge_count": removed_original_count,
        "accepted_candidate_centerline_count": accepted_candidate_count,
        "accepted_candidate_edge_count": accepted_candidate_edge_count,
        "auto_accepted_gap_count": auto_candidate_count,
        "auto_accepted_gap_edge_count": auto_candidate_edge_count,
        "auto_accepted_surface_count": auto_surface_count,
        "auto_accepted_surface_edge_count": auto_surface_edge_count,
        "auto_accepted_surface_rule_counts": auto_surface_rule_counts,
        "auto_surface_hard_veto_count": auto_surface_veto_count,
        "auto_accepted_long_surface_count": auto_long_surface_count,
        "rejected_candidate_centerline_count": rejected_candidate_count,
        "review_estimated_width_count": estimated_count,
        "width_measurement_stage": "after_final_topology_only",
        "manual_width_measurement_file": str(manual_width_path.resolve()) if manual_width_path and manual_width_path.is_file() else "",
        "manual_width_overridden_edge_count": manual_width_edge_count,
        "manual_surface_add_file": str(manual_surface_add_path.resolve()) if manual_surface_add_path and manual_surface_add_path.is_file() else "",
        "manual_surface_remove_file": str(manual_surface_remove_path.resolve()) if manual_surface_remove_path and manual_surface_remove_path.is_file() else "",
        "manual_surface_added_px": int(np.count_nonzero(manual_surface_add)) if manual_surface_add is not None else 0,
        "manual_surface_removed_px": int(np.count_nonzero(manual_surface_remove)) if manual_surface_remove is not None else 0,
        "width_calculator": width_result.algorithm,
        "width_calculator_metadata": width_result.metadata,
        "profiling": {
            "junction_cleanup_seconds": float(
                (summary.get("profiling") or {}).get("junction_cleanup_seconds", 0.0)
            ),
            "candidate_analysis_seconds": float(candidate_analysis_seconds),
            "final_width_measurement_seconds": float(final_width_measurement_seconds),
            "surface_reconstruction_seconds": float(surface_reconstruction_seconds),
            "total_seconds": float(time.perf_counter() - finalize_started),
        },
        "final_surface_stage": "regular_buffer_surface",
        "width_surface_metadata": width_surface.metadata if width_surface is not None else {"status": "not_built"},
        "unresolved_review_width_count": 0,
        "unresolved_added_width_edge_count": unresolved_added_count,
        "global_median_width_fallback_count": 0,
        "input_topology": summary.get("original_topology", graph_topology_metrics(nodes_np, edges_np)),
        "prepared_topology": summary.get("prepared_topology", graph_topology_metrics(nodes_np, edges_np)),
        "optimized_topology": graph_topology_metrics(final_nodes_np, final_edges_np),
        "optimized_width_sample_count": len(final_width_samples),
        "optimized_width_segment_count": len(final_width_segments),
        "pixel_size": pixel_size,
        "snap_additions_px": args.snap_additions_px,
        "added_centerline_step_px": args.added_centerline_step_px,
        "outputs": {
            "optimized_graph": f"{stem}_optimized_graph.p",
            "optimized_edges": f"{stem}_optimized_edges.csv",
            "optimized_candidate_centerlines": f"{stem}_optimized_candidate_centerlines.csv",
            "optimized_surface_regions": f"{stem}_optimized_surface_regions.csv",
            "optimized_road_surface": f"{stem}_optimized_road_surface.png",
            "optimized_viz": f"{stem}_optimized_viz.png",
            "fusion_comparison": f"{stem}_fusion_comparison.png",
            "optimized_width_samples": f"{stem}_optimized_width_samples.csv",
            "optimized_width_segments": f"{stem}_optimized_width_segments.csv",
            "width_surface_added": f"{stem}_width_surface_added.png" if width_surface is not None else "",
            "width_surface_removed": f"{stem}_width_surface_removed.png" if width_surface is not None else "",
        },
    }
    with open(final_dir / f"{stem}_optimized_summary.json", "w", encoding="utf-8") as file:
        json.dump(optimized_summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(optimized_summary, ensure_ascii=False, indent=2))
    return optimized_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply review decisions to SAMRoad/SAM_MLoRA width outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory containing review_decisions.csv and *_summary.json.")
    parser.add_argument("--final-dir", default="", help="Output directory. Default: <output-dir>/finalized_review.")
    parser.add_argument(
        "--edited-dir", default="",
        help="Optional directory containing *_edited_graph.p and step-3 *_reconstructed_road_surface.png files.",
    )
    parser.add_argument("--require-complete-review", action="store_true", help="Stop when any manual review item is unfinished.")
    parser.add_argument("--snap-additions-px", type=float, default=20.0, help="Conservative endpoint snap radius for accepted candidates.")
    parser.add_argument("--added-centerline-step-px", type=float, default=10.0, help="Node spacing along the exact centerline shown during review.")
    parser.add_argument(
        "--width-calculator",
        default="",
        help="Optional final-width implementation as module:function. It must accept FinalWidthRequest and return FinalWidthResult.",
    )
    parser.add_argument("--surface-width-scale", type=float, default=1.0, help="Scale applied to measured widths when rebuilding the final road surface.")
    parser.add_argument("--surface-min-width-px", type=float, default=1.0, help="Minimum usable width for final surface reconstruction.")
    parser.add_argument("--surface-close-kernel", type=int, default=3, help="Odd closing kernel for the width-constrained final surface.")
    parser.add_argument("--surface-support-margin-px", type=int, default=4, help="Natural boundary allowance beyond chain-regularized centerline corridors.")
    parser.add_argument("--surface-min-unsupported-area-px", type=int, default=12, help="Minimum unsupported connected lobe removed from the final surface.")
    parser.add_argument("--surface-chain-width-deviation-ratio", type=float, default=0.25, help="Maximum edge support-width deviation from its road-chain median.")
    parser.add_argument("--surface-continuity-close-kernel", type=int, default=7, help="Closing kernel used only inside chain support corridors.")
    parser.add_argument("--surface-continuity-max-gap-px", type=int, default=8, help="Maximum centerline gap repaired from nearby SAM-MLoRA surface.")
    parser.add_argument("--surface-boundary-smooth-sigma-px", type=float, default=1.5, help="Pixel-scale Gaussian smoothing applied to the final road boundary; 0 disables it.")
    parser.add_argument("--surface-regular-corridor-cosine", type=float, default=0.94, help="方向连续道路走廊的最小方向余弦。")
    parser.add_argument("--surface-regular-corridor-width-ratio", type=float, default=0.35, help="方向连续道路走廊允许的最大相对宽度差。")
    parser.add_argument(
        "--only-stem", action="append", default=[],
        help="只重算指定切片 stem；可重复传入。未提供时保持原来的全量重算。",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    final_dir = Path(args.final_dir) if args.final_dir else output_dir / "finalized_review"
    final_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(
        path
        for path in output_dir.glob("*_summary.json")
        if path.name != "batch_width_summary.json" and not path.name.endswith("_optimized_summary.json")
    )
    requested_stems = {str(value) for value in args.only_stem if str(value).strip()}
    if requested_stems:
        summary_paths = [
            path for path in summary_paths
            if path.name.removesuffix("_summary.json") in requested_stems
        ]
    if not summary_paths:
        detail = f" for stems {sorted(requested_stems)}" if requested_stems else ""
        raise FileNotFoundError(f"No *_summary.json files under {output_dir}{detail}")

    decisions_path = output_dir / "review_decisions.csv"
    decisions = decisions_by_key(read_csv(decisions_path))
    if args.require_complete_review:
        required = set()
        for path in output_dir.glob("*_conflict_review.csv"):
            item_stem = path.name.removesuffix("_conflict_review.csv")
            for row in read_csv(path):
                if row.get("item_type") != "edge_review" and str(row.get("requires_manual_review", "")).strip().lower() in {"1", "true", "yes"}:
                    required.add((item_stem, str(row.get("item_type", "")), str(row.get("item_id", ""))))
        completed = {
            key for key, row in decisions.items()
            if str(row.get("decision", "")) not in {"", "defer", "skip"}
        }
        remaining = required - completed
        if remaining:
            raise RuntimeError(f"Manual review is incomplete: {len(remaining)} of {len(required)} required items remain.")
    success_count = 0
    failures = []
    for index, summary_path in enumerate(summary_paths, start=1):
        print(f"[{index}/{len(summary_paths)}] Finalizing {summary_path.name}")
        try:
            finalize_one(args, output_dir, final_dir, summary_path, decisions, decisions_path)
            success_count += 1
        except Exception as exc:
            failures.append({"summary": str(summary_path), "error": str(exc)})
            print(f"Failed: {summary_path} -> {exc}")
    batch_summary = {
        "source_output_dir": str(output_dir),
        "final_dir": str(final_dir),
        "slice_count": len(summary_paths),
        "success_count": success_count,
        "failure_count": len(failures),
        "failures": failures,
        "slices": [summary_path.name.removesuffix("_summary.json") for summary_path in summary_paths],
        "partial_rebuild": bool(requested_stems),
        "requested_stems": sorted(requested_stems),
    }
    with open(final_dir / "batch_optimized_summary.json", "w", encoding="utf-8") as file:
        json.dump(batch_summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
