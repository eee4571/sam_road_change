from __future__ import annotations

import argparse
import csv
import heapq
import json
import pickle
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import linemerge, snap, substring, unary_union


FINAL_CENTERLINE_GEOMETRY_POLICY = (
    "fully_automatic_probability_and_surface_guided_unique_facing_cross_component_gap_repairs"
)

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from finalize_review_results import load_graph, save_graph  # noqa: E402
from review_geometry import accepted_surface_region_polylines  # noqa: E402
from surface_reconstruction import SurfaceReconstructionConfig, reconstruct_surface  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _read_image(path: Path, flags: int = cv2.IMREAD_GRAYSCALE):
    """Read images through bytes so Windows Unicode paths remain reliable."""
    path = Path(path)
    if not path.is_file():
        return None
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(encoded, flags)


def summaries(directory: Path, optimized: bool = False) -> list[Path]:
    suffix = "*_optimized_summary.json" if optimized else "*_summary.json"
    return sorted(
        path for path in directory.glob(suffix)
        if not path.name.startswith("batch_")
    )


def stem_from_summary(path: Path) -> str:
    suffix = "_optimized_summary" if path.stem.endswith("_optimized_summary") else "_summary"
    return path.stem[: -len(suffix)]


def review_status(review_dir: Path) -> dict:
    decisions = read_csv(review_dir / "review_decisions.csv")
    decided = {(row.get("stem", ""), row.get("item_type", ""), str(row.get("item_id", ""))) for row in decisions if row.get("decision") not in {"", "defer", "skip"}}
    required: set[tuple[str, str, str]] = set()
    for summary_path in summaries(review_dir):
        stem = stem_from_summary(summary_path)
        conflict_path = review_dir / f"{stem}_conflict_review.csv"
        rows = read_csv(conflict_path)
        if rows:
            for row in rows:
                if row.get("item_type") != "edge_review" and str(row.get("requires_manual_review", "")).lower() in {"1", "true", "yes"}:
                    required.add((stem, row.get("item_type", ""), str(row.get("item_id", ""))))
        else:
            for row in read_csv(review_dir / f"{stem}_candidate_centerlines.csv"):
                if row.get("auto_decision") != "accept":
                    required.add((stem, "candidate_centerline", str(row.get("candidate_id", ""))))
    completed = required & decided
    remaining = sorted(required - decided)
    return {
        "required_count": len(required), "completed_count": len(completed),
        "remaining_count": len(remaining),
        "completion_percent": round(100.0 * len(completed) / max(1, len(required)), 1),
        "complete": not remaining, "remaining_preview": ["|".join(item) for item in remaining[:50]],
        "remaining_preview_truncated": len(remaining) > 50,
    }


def _raster_context(summary: dict) -> tuple[Path, rasterio.Affine, object, tuple[int, int]]:
    image_path = Path(summary["image"])
    with rasterio.open(image_path) as dataset:
        return image_path, dataset.transform, dataset.crs, (dataset.height, dataset.width)


def _world_line(points_rc: list[tuple[float, float]], transform) -> LineString:
    return LineString([rasterio.transform.xy(transform, row, col, offset="center") for row, col in points_rc])


def export_edit_package(review_dir: Path, output: Path) -> dict:
    line_rows: list[dict] = []
    surface_rows: list[dict] = []
    crs = None
    decisions = {
        (row.get("stem", ""), row.get("item_type", ""), str(row.get("item_id", ""))): row.get("decision", "")
        for row in read_csv(review_dir / "review_decisions.csv")
    }
    for summary_path in summaries(review_dir):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stem = stem_from_summary(summary_path)
        _, transform, tile_crs, _ = _raster_context(summary)
        if crs is None:
            crs = tile_crs
        elif tile_crs != crs:
            raise ValueError("All edit-package rasters must use the same CRS")
        nodes, edges = load_graph(Path(summary.get("prepared_graph", summary["graph"])))
        for edge_id, (src, dst) in enumerate(edges.tolist()):
            line_rows.append({
                "tile_stem": stem, "feature_id": f"samroad:{edge_id}", "source": "samroad",
                "active": 1, "review": "existing",
                "geometry": _world_line([tuple(nodes[src]), tuple(nodes[dst])], transform),
            })
        for row in read_csv(review_dir / f"{stem}_candidate_centerlines.csv"):
            try:
                points = json.loads(row.get("polyline_points_json", ""))
            except json.JSONDecodeError:
                points = []
            if len(points) < 2:
                points = [[float(row["start_row"]), float(row["start_col"])], [float(row["end_row"]), float(row["end_col"])]]
            candidate_id = str(row.get("candidate_id", ""))
            decision = decisions.get((stem, "candidate_centerline", candidate_id), "")
            line_rows.append({
                "tile_stem": stem, "feature_id": f"candidate:{candidate_id}",
                "source": row.get("candidate_type", "candidate"),
                "active": 1 if decision == "accept" or (not decision and row.get("auto_decision") == "accept") else 0,
                "review": decision or row.get("confidence", ""), "geometry": _world_line(points, transform),
            })
        mask = _read_image(review_dir / f"{stem}_molra_clean_mask.png")
        if mask is not None:
            surface_only = _read_image(review_dir / f"{stem}_surface_only.png")
            candidate_rows = read_csv(review_dir / f"{stem}_candidate_centerlines.csv")
            for region_id, polylines in accepted_surface_region_polylines(
                stem, surface_only, candidate_rows, decisions
            ).items():
                for branch_index, points in enumerate(polylines):
                    line_rows.append({
                        "tile_stem": stem,
                        "feature_id": f"surface_region:{region_id}:{branch_index}",
                        "source": "accepted_surface_skeleton",
                        "active": 1,
                        "review": "accept",
                        "geometry": _world_line(points, transform),
                    })
            if surface_only is not None:
                _, labels = cv2.connectedComponents((surface_only > 0).astype(np.uint8), connectivity=8)
                candidates_by_region: dict[str, list[str]] = {}
                for candidate in read_csv(review_dir / f"{stem}_candidate_centerlines.csv"):
                    region_id = str(candidate.get("region_id", ""))
                    if region_id:
                        candidates_by_region.setdefault(region_id, []).append(
                            decisions.get((stem, "candidate_centerline", str(candidate.get("candidate_id", ""))), "")
                        )
                for region_id in set(str(value) for value in np.unique(labels) if value):
                    region_decision = decisions.get((stem, "surface_only_region", region_id), "")
                    candidate_decisions = [value for value in candidates_by_region.get(region_id, []) if value]
                    rejected = region_decision in {"reject", "mark_nonroad"} or (
                        candidate_decisions and all(value in {"reject", "mark_nonroad"} for value in candidate_decisions)
                    )
                    if rejected:
                        mask[labels == int(region_id)] = 0
            for geom, value in shapes((mask > 0).astype(np.uint8), mask=mask > 0, transform=transform):
                if value:
                    surface_rows.append({
                        "tile_stem": stem, "source": "sam_molra_reference",
                        "editable": 0, "geometry": shape(geom),
                    })

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    gpd.GeoDataFrame(line_rows, geometry="geometry", crs=crs).to_file(output, layer="editable_centerlines", driver="GPKG")
    if surface_rows:
        gpd.GeoDataFrame(surface_rows, geometry="geometry", crs=crs).to_file(output, layer="road_surface_reference", driver="GPKG")
    status = review_status(review_dir)
    status["edit_package"] = str(output)
    return status


def _active(row) -> bool:
    value = str(row.get("active", 1)).strip().lower()
    return value not in {"0", "false", "no", "delete", "deleted"}


def _line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def _pixel_line(geometry, inverse) -> LineString:
    return LineString([(float((inverse * (x, y))[0]), float((inverse * (x, y))[1])) for x, y in geometry.coords])


def _node_pixel_lines(lines: list[LineString], tolerance: float):
    """Project dangling endpoints to nearby lines, then split all intersections."""
    adjusted: list[LineString] = []
    for line_id, line in enumerate(lines):
        coords = list(line.coords)
        for endpoint_index in (0, -1):
            endpoint = Point(coords[endpoint_index])
            best_point = None
            best_distance = float("inf")
            for other_id, other in enumerate(lines):
                if other_id == line_id:
                    continue
                distance = endpoint.distance(other)
                if distance < best_distance:
                    best_distance = distance
                    best_point = other.interpolate(other.project(endpoint))
            if best_point is not None and 1e-6 < best_distance <= tolerance:
                coords[endpoint_index] = tuple(best_point.coords[0])
        adjusted.append(LineString(coords))
    return unary_union(adjusted)


def _snap_node(nodes: list[tuple[float, float]], point: tuple[float, float], tolerance: float) -> int:
    if nodes:
        array = np.asarray(nodes, dtype=np.float32)
        distances = np.linalg.norm(array - np.asarray(point, dtype=np.float32), axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) <= tolerance:
            return index
    nodes.append(point)
    return len(nodes) - 1


def import_edit_package(gpkg: Path, review_dir: Path, edited_dir: Path, snap_tolerance_px: float) -> dict:
    lines = gpd.read_file(gpkg, layer="editable_centerlines")
    edited_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(gpkg), "tiles": {}}
    for summary_path in summaries(review_dir):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stem = stem_from_summary(summary_path)
        _, transform, crs, _ = _raster_context(summary)
        inverse = ~transform
        tile_lines = lines[(lines["tile_stem"] == stem) & lines.apply(_active, axis=1)]
        if tile_lines.crs != crs:
            tile_lines = tile_lines.to_crs(crs)
        pixel_lines = []
        for geometry in tile_lines.geometry:
            for part in _line_parts(geometry):
                pixel_lines.append(_pixel_line(part, inverse))
        noded = _node_pixel_lines(pixel_lines, snap_tolerance_px) if pixel_lines else None
        nodes: list[tuple[float, float]] = []
        edges: list[tuple[int, int]] = []
        for part in _line_parts(noded):
            previous = None
            for col, r in part.coords:
                node_id = _snap_node(nodes, (float(r), float(col)), 0.05)
                if previous is not None and previous != node_id:
                    edges.append((previous, node_id))
                previous = node_id
        graph_path = edited_dir / f"{stem}_edited_graph.p"
        save_graph(graph_path, nodes, edges)

        manifest["tiles"][stem] = {
            "graph": str(graph_path), "line_count": len(edges),
            "pending_candidate_ids": [], "editor": "external_gpkg",
            "surface_policy": "reconstruct_after_centerline_stitching",
        }
    (edited_dir / "edited_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _chain_segments(nodes: np.ndarray, edges: np.ndarray):
    from chain_width_calculator import build_road_chains
    for chain in build_road_chains(nodes, edges):
        yield chain, [tuple(nodes[node_id]) for node_id in chain.node_ids]


def _clean_surface(geometry, min_area: float, min_hole_area: float):
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else []
    cleaned = []
    for polygon in polygons:
        if polygon.area < min_area:
            continue
        holes = [ring.coords for ring in polygon.interiors if Polygon(ring).area >= min_hole_area]
        candidate = Polygon(polygon.exterior.coords, holes).buffer(0)
        if not candidate.is_empty:
            cleaned.append(candidate)
    if not cleaned:
        return None
    result = unary_union(cleaned).buffer(0)
    return result if result.is_valid else result.buffer(0)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = 0.5 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _line_direction(geometry: LineString) -> np.ndarray:
    coords = np.asarray(geometry.coords, dtype=np.float64)
    direction = coords[-1] - coords[0]
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-9 else np.asarray([1.0, 0.0], dtype=np.float64)


def _snap_network_endpoints(geometries: list[LineString], tolerance: float):
    """Globally close small endpoint seams before line conflation.

    Per-tile graphs can be internally connected yet miss one another by a
    fraction of a pixel at tile boundaries.  Cluster nearby endpoints, snap
    them to one representative, and also node endpoints that stop just short
    of another line.  This is deliberately limited to the fusion tolerance;
    longer, surface-supported gaps are repaired in raster space earlier.
    """
    parts = [part for geometry in geometries for part in _line_parts(geometry)]
    if not parts:
        return unary_union([])
    tolerance = max(float(tolerance), 1e-9)
    reference = unary_union(parts)
    representatives: list[Point] = []
    for part in parts:
        for coordinate in (part.coords[0], part.coords[-1]):
            endpoint = Point(coordinate)
            if not any(endpoint.distance(item) <= tolerance for item in representatives):
                representatives.append(endpoint)
    endpoint_reference = unary_union(representatives)
    snapped_parts = [
        snap(snap(part, endpoint_reference, tolerance), reference, tolerance)
        for part in parts
    ]
    return unary_union(snapped_parts)


def _probability_guided_gap_path(
    start: tuple[float, float],
    end: tuple[float, float],
    probability: np.ndarray,
    transform,
    road_surface,
    pixel_size: float,
    lateral_margin_px: int = 28,
) -> tuple[LineString | None, dict]:
    """Trace one automatic endpoint connector along a local probability ridge."""
    if probability is None or transform is None or probability.size == 0:
        return None, {}
    inverse = ~transform
    start_col, start_row = inverse * start
    end_col, end_row = inverse * end
    height, width = probability.shape
    if not (
        -0.5 <= start_row < height + 0.5 and -0.5 <= start_col < width + 0.5
        and -0.5 <= end_row < height + 0.5 and -0.5 <= end_col < width + 0.5
    ):
        return None, {}

    margin = max(8, int(lateral_margin_px))
    row0 = max(0, int(np.floor(min(start_row, end_row))) - margin)
    row1 = min(height, int(np.ceil(max(start_row, end_row))) + margin + 1)
    col0 = max(0, int(np.floor(min(start_col, end_col))) - margin)
    col1 = min(width, int(np.ceil(max(start_col, end_col))) + margin + 1)
    local = np.asarray(probability[row0:row1, col0:col1], dtype=np.float32)
    if local.size == 0:
        return None, {}
    positive = local[local > 0]
    local_reference = max(
        1.0 / 255.0,
        float(np.percentile(positive, 90)) if positive.size else 1.0 / 255.0,
    )
    signal = np.clip(local / local_reference, 0.0, 1.0)
    signal = np.sqrt(signal, dtype=np.float32)
    local_transform = transform * rasterio.Affine.translation(col0, row0)
    if road_surface is not None and not road_surface.is_empty:
        surface = rasterize(
            [(road_surface, 1)],
            out_shape=local.shape,
            transform=local_transform,
            fill=0,
            dtype=np.uint8,
        )
    else:
        surface = np.zeros(local.shape, dtype=np.uint8)

    source = (
        int(np.clip(round(start_row) - row0, 0, local.shape[0] - 1)),
        int(np.clip(round(start_col) - col0, 0, local.shape[1] - 1)),
    )
    target = (
        int(np.clip(round(end_row) - row0, 0, local.shape[0] - 1)),
        int(np.clip(round(end_col) - col0, 0, local.shape[1] - 1)),
    )
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    segment = target_array - source_array
    segment_length_sq = max(float(np.dot(segment, segment)), 1e-9)

    def lateral_penalty(row: int, col: int) -> float:
        point = np.asarray([row, col], dtype=np.float64)
        fraction = float(np.clip(np.dot(point - source_array, segment) / segment_length_sq, 0.0, 1.0))
        distance = float(np.linalg.norm(point - (source_array + fraction * segment)))
        return min(1.0, distance / max(float(margin), 1.0))

    neighbors = (
        (-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, 2 ** 0.5), (1, 0, 1.0), (1, 1, 2 ** 0.5),
    )
    queue = [(0.0, 0.0, source)]
    costs = {source: 0.0}
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    visited: set[tuple[int, int]] = set()
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == target:
            break
        for delta_row, delta_col, step in neighbors:
            row, col = current[0] + delta_row, current[1] + delta_col
            if not (0 <= row < local.shape[0] and 0 <= col < local.shape[1]):
                continue
            probability_cost = (1.0 - float(signal[row, col])) ** 2
            surface_cost = 0.0 if surface[row, col] else 1.0
            step_cost = step * (
                0.08 + 0.72 * probability_cost + 0.12 * surface_cost
                + 0.08 * lateral_penalty(row, col)
            )
            candidate_cost = current_cost + step_cost
            node = (row, col)
            if candidate_cost >= costs.get(node, float("inf")):
                continue
            costs[node] = candidate_cost
            parents[node] = current
            heuristic = 0.08 * float(np.hypot(target[0] - row, target[1] - col))
            heapq.heappush(queue, (candidate_cost + heuristic, candidate_cost, node))
    if target not in costs:
        return None, {}

    pixels = [target]
    while pixels[-1] != source:
        pixels.append(parents[pixels[-1]])
    pixels.reverse()
    path_values = np.asarray([local[row, col] for row, col in pixels], dtype=np.float32)
    normalized_values = np.asarray([signal[row, col] for row, col in pixels], dtype=np.float32)
    surface_support = float(np.mean([surface[row, col] > 0 for row, col in pixels]))
    path_length_px = float(sum(
        np.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(pixels[:-1], pixels[1:])
    ))
    direct_length_px = max(float(np.hypot(end_row - start_row, end_col - start_col)), 1e-6)
    high_support_ratio = float(np.mean(normalized_values >= 0.25))
    evidence = {
        "center_probability_mean": float(np.mean(path_values)),
        "center_probability_q25": float(np.quantile(path_values, 0.25)),
        "center_probability_normalized_mean": float(np.mean(normalized_values)),
        "center_probability_high_support_ratio": high_support_ratio,
        "surface_support_ratio": surface_support,
        "path_ratio": path_length_px / direct_length_px,
        "local_probability_reference": local_reference,
    }
    accepted = (
        evidence["path_ratio"] <= 1.45
        and evidence["center_probability_normalized_mean"] >= 0.32
        and high_support_ratio >= 0.55
    )
    if not accepted:
        return None, evidence

    sampled_pixels = pixels[::3]
    if sampled_pixels[-1] != pixels[-1]:
        sampled_pixels.append(pixels[-1])
    coordinates = [start]
    coordinates.extend(
        rasterio.transform.xy(transform, row + row0, col + col0, offset="center")
        for row, col in sampled_pixels[1:-1]
    )
    coordinates.append(end)
    connector = LineString(coordinates).simplify(0.35 * pixel_size, preserve_topology=False)
    return connector, evidence


def _connect_surface_supported_global_gaps(
    centerlines: list[dict],
    road_surface,
    pixel_size: float,
    centerline_probability: np.ndarray | None = None,
    centerline_transform=None,
    max_gap_px: float = 192.0,
    min_alignment: float = 0.50,
    min_surface_support: float = 0.95,
    ambiguity_ratio: float = 1.20,
) -> tuple[list[dict], int]:
    """Automatically repair endpoint-to-endpoint and endpoint-to-edge gaps."""
    has_surface = road_surface is not None and not road_surface.is_empty
    has_probability = centerline_probability is not None and centerline_transform is not None
    if not centerlines or (not has_surface and not has_probability) or max_gap_px <= 0:
        return centerlines, 0
    pixel_size = max(float(pixel_size), 1e-9)
    network = unary_union([row["geometry"] for row in centerlines])
    parts = _line_parts(network)
    graph = nx.Graph()
    endpoint_directions: dict[tuple[float, float], np.ndarray] = {}
    endpoint_coordinates: dict[tuple[float, float], tuple[float, float]] = {}

    def node_key(coordinate) -> tuple[float, float]:
        return round(float(coordinate[0]), 8), round(float(coordinate[1]), 8)

    part_endpoints: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for part_index, part in enumerate(parts):
        coordinates = list(part.coords)
        if len(coordinates) < 2:
            continue
        start, end = node_key(coordinates[0]), node_key(coordinates[-1])
        endpoint_coordinates.setdefault(start, tuple(float(value) for value in coordinates[0]))
        endpoint_coordinates.setdefault(end, tuple(float(value) for value in coordinates[-1]))
        part_endpoints[part_index] = (start, end)
        graph.add_edge(start, end)
        start_vector = np.asarray(coordinates[0], dtype=np.float64) - np.asarray(coordinates[1], dtype=np.float64)
        end_vector = np.asarray(coordinates[-1], dtype=np.float64) - np.asarray(coordinates[-2], dtype=np.float64)
        for key, vector in ((start, start_vector), (end, end_vector)):
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                endpoint_directions[key] = vector / norm
    if graph.number_of_nodes() == 0:
        return centerlines, 0

    component_id: dict[tuple[float, float], int] = {}
    for index, component in enumerate(nx.connected_components(graph)):
        for node in component:
            component_id[node] = index
    part_component_id = {
        part_index: component_id[start]
        for part_index, (start, _) in part_endpoints.items()
    }
    dangling = [node for node, degree in graph.degree() if degree == 1 and node in endpoint_directions]
    max_gap = max_gap_px * pixel_size
    proposals: list[tuple] = []
    surface_with_tolerance = road_surface.buffer(0.75 * pixel_size) if has_surface else None
    spatial_cells: dict[tuple[int, int], list[int]] = {}
    for endpoint_index, endpoint in enumerate(dangling):
        cell = (int(np.floor(endpoint[0] / max_gap)), int(np.floor(endpoint[1] / max_gap)))
        spatial_cells.setdefault(cell, []).append(endpoint_index)
    for position, start in enumerate(dangling):
        start_coordinate = endpoint_coordinates[start]
        start_point = np.asarray(start_coordinate, dtype=np.float64)
        start_cell = (int(np.floor(start[0] / max_gap)), int(np.floor(start[1] / max_gap)))
        nearby_indices = [
            endpoint_index
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for endpoint_index in spatial_cells.get((start_cell[0] + dx, start_cell[1] + dy), [])
            if endpoint_index > position
        ]
        for endpoint_index in nearby_indices:
            end = dangling[endpoint_index]
            if component_id[start] == component_id[end]:
                continue
            end_coordinate = endpoint_coordinates[end]
            end_point = np.asarray(end_coordinate, dtype=np.float64)
            delta = end_point - start_point
            distance = float(np.linalg.norm(delta))
            if distance <= 1.5 * pixel_size or distance > max_gap:
                continue
            direction = delta / distance
            start_alignment = float(np.dot(endpoint_directions[start], direction))
            end_alignment = float(np.dot(endpoint_directions[end], -direction))
            alignment = min(start_alignment, end_alignment)
            if alignment < min_alignment:
                continue
            straight_connector = LineString([start_coordinate, end_coordinate])
            sample_count = max(3, int(np.ceil(distance / pixel_size)) + 1)
            support = float(np.mean([
                surface_with_tolerance.covers(straight_connector.interpolate(index / (sample_count - 1), normalized=True))
                for index in range(sample_count)
            ])) if surface_with_tolerance is not None else 0.0
            connector, probability_evidence = _probability_guided_gap_path(
                start_coordinate,
                end_coordinate,
                centerline_probability,
                centerline_transform,
                road_surface,
                pixel_size,
            ) if has_probability else (None, {})
            probability_supported = connector is not None
            if support < min_surface_support and not probability_supported:
                continue
            if not probability_supported and alignment < 0.85:
                continue
            connector = connector if probability_supported else straight_connector
            probability_score = float(probability_evidence.get("center_probability_normalized_mean", 0.0))
            score = (
                0.35 * max(support, probability_score)
                + 0.35 * alignment
                + 0.20 * (1.0 - distance / max_gap)
                + 0.10 * (1.0 - float(probability_evidence.get("path_ratio", 1.0)))
            )
            evidence = {
                **probability_evidence,
                "straight_surface_support_ratio": support,
                "direction_alignment": alignment,
                "target_kind": "endpoint",
                "evidence_mode": (
                    "centerline_probability_astar_and_surface"
                    if probability_supported and support >= min_surface_support
                    else "centerline_probability_astar"
                    if probability_supported
                    else "continuous_surface"
                ),
            }
            proposals.append((
                score, distance, start, end, connector, evidence,
                component_id[start], component_id[end],
            ))

    # A large share of real omissions are T-connections: a branch endpoint
    # reaches the interior of a main-road line, so there is no opposite
    # dangling endpoint to pair with.  Search the nearest line in every other
    # component, trace the probability ridge to its projection, and later
    # split that target line at the exact contact coordinate.
    edge_proposals: list[tuple] = []
    for start in dangling:
        start_coordinate = endpoint_coordinates[start]
        start_point = Point(start_coordinate)
        valid_by_component: dict[int, tuple] = {}
        for part_index, part in enumerate(parts):
            target_component = part_component_id.get(part_index)
            if target_component is None or target_component == component_id[start]:
                continue
            projected_distance = float(part.project(start_point))
            if projected_distance <= 2.0 * pixel_size or part.length - projected_distance <= 2.0 * pixel_size:
                continue
            projection = part.interpolate(projected_distance)
            end = tuple(float(value) for value in projection.coords[0])
            delta = np.asarray(end, dtype=np.float64) - np.asarray(start_coordinate, dtype=np.float64)
            distance = float(np.linalg.norm(delta))
            if distance <= 1.5 * pixel_size or distance > max_gap:
                continue
            alignment = float(np.dot(endpoint_directions[start], delta / distance))
            if alignment < min_alignment:
                continue
            straight_connector = LineString([start_coordinate, end])
            sample_count = max(3, int(np.ceil(distance / pixel_size)) + 1)
            support = float(np.mean([
                surface_with_tolerance.covers(straight_connector.interpolate(index / (sample_count - 1), normalized=True))
                for index in range(sample_count)
            ])) if surface_with_tolerance is not None else 0.0
            connector, probability_evidence = _probability_guided_gap_path(
                start_coordinate,
                end,
                centerline_probability,
                centerline_transform,
                road_surface,
                pixel_size,
                lateral_margin_px=max(28, min(64, int(np.ceil(distance / pixel_size * 0.40)))),
            ) if has_probability else (None, {})
            probability_supported = connector is not None
            if support < min_surface_support and not probability_supported:
                continue
            if not probability_supported and alignment < 0.85:
                continue
            connector = connector if probability_supported else straight_connector
            probability_score = float(probability_evidence.get("center_probability_normalized_mean", 0.0))
            score = (
                0.35 * max(support, probability_score)
                + 0.35 * alignment
                + 0.20 * (1.0 - distance / max_gap)
                + 0.10 * (1.0 - float(probability_evidence.get("path_ratio", 1.0)))
            )
            evidence = {
                **probability_evidence,
                "straight_surface_support_ratio": support,
                "direction_alignment": alignment,
                "target_kind": "edge",
                "target_part_index": part_index,
                "evidence_mode": (
                    "centerline_probability_astar_to_edge_and_surface"
                    if probability_supported and support >= min_surface_support
                    else "centerline_probability_astar_to_edge"
                    if probability_supported
                    else "continuous_surface_to_edge"
                ),
            }
            proposal = (
                score, distance, start, end, connector, evidence,
                component_id[start], target_component,
            )
            previous = valid_by_component.get(target_component)
            if previous is None or distance < previous[1]:
                valid_by_component[target_component] = proposal
        choices = sorted(valid_by_component.values(), key=lambda item: item[1])
        if not choices:
            continue
        if len(choices) > 1 and choices[1][1] < choices[0][1] * ambiguity_ratio:
            continue
        edge_proposals.append(choices[0])

    parent = {index: index for index in set(component_id.values())}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> bool:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return False
        parent[second_root] = first_root
        return True

    # A mere nearby pair is insufficient at global scale: retain only mutually
    # unique choices, rejecting a target when another plausible endpoint is
    # nearly as close.  This prevents one road surface from tying together a
    # junction fan or parallel carriageways.
    proposals_by_endpoint: dict[tuple[float, float], list[tuple]] = {}
    for proposal in proposals:
        proposals_by_endpoint.setdefault(proposal[2], []).append(proposal)
        proposals_by_endpoint.setdefault(proposal[3], []).append(proposal)
    mutually_unique: list[tuple] = []
    for proposal in proposals:
        _, distance, start, end, _, _, _, _ = proposal
        start_choices = sorted(proposals_by_endpoint[start], key=lambda item: item[1])
        end_choices = sorted(proposals_by_endpoint[end], key=lambda item: item[1])
        if not start_choices or not end_choices or start_choices[0][2:4] != (start, end) and start_choices[0][2:4] != (end, start):
            continue
        if not end_choices or end_choices[0][2:4] != (start, end) and end_choices[0][2:4] != (end, start):
            continue
        if len(start_choices) > 1 and start_choices[1][1] < distance * ambiguity_ratio:
            continue
        if len(end_choices) > 1 and end_choices[1][1] < distance * ambiguity_ratio:
            continue
        mutually_unique.append(proposal)

    used_endpoints: set[tuple[float, float]] = set()
    additions: list[dict] = []
    split_points: list[Point] = []
    valid_widths = [float(row.get("width_map", 0.0) or 0.0) for row in centerlines if float(row.get("width_map", 0.0) or 0.0) > 0]
    fallback_width = float(np.median(valid_widths)) if valid_widths else 0.0

    def attached_endpoint_width(coordinate, direction: np.ndarray) -> float:
        """Return the aligned road width at a connector endpoint."""
        point = Point(coordinate)
        candidates: list[float] = []
        tolerance = max(1e-8, 1.5 * pixel_size)
        for row in centerlines:
            try:
                width_map = float(row.get("width_map", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(width_map) or width_map <= 0:
                continue
            for part in _line_parts(row.get("geometry")):
                if part.boundary.distance(point) > tolerance:
                    continue
                if abs(float(np.dot(_line_direction(part), direction))) < min_alignment:
                    continue
                candidates.append(width_map)
        return float(np.median(np.asarray(candidates, dtype=np.float64))) if candidates else 0.0

    accepted_proposals = sorted([*mutually_unique, *edge_proposals], key=lambda item: (-item[0], item[1]))
    for score, distance, start, end, connector, evidence, start_component, end_component in accepted_proposals:
        if start in used_endpoints or (evidence["target_kind"] == "endpoint" and end in used_endpoints):
            continue
        if not union(start_component, end_component):
            continue
        connector_coordinates = list(connector.coords)
        start_tangent = np.asarray(connector_coordinates[1], dtype=np.float64) - np.asarray(connector_coordinates[0], dtype=np.float64)
        start_tangent /= max(float(np.linalg.norm(start_tangent)), 1e-9)
        start_width = attached_endpoint_width(start, start_tangent)
        end_width = 0.0
        if evidence["target_kind"] == "endpoint":
            end_tangent = np.asarray(connector_coordinates[-2], dtype=np.float64) - np.asarray(connector_coordinates[-1], dtype=np.float64)
            end_tangent /= max(float(np.linalg.norm(end_tangent)), 1e-9)
            end_width = attached_endpoint_width(end, end_tangent)
        connected_widths = [value for value in (start_width, end_width) if value > 0]
        connector_width = float(np.median(np.asarray(connected_widths, dtype=np.float64))) if connected_widths else fallback_width
        additions.append({
            "global_id": len(centerlines) + len(additions),
            "width_map": connector_width,
            "width_std": 0.0,
            "src_count": 0,
            "src_tiles": "",
            "quality_gr": "C",
            "fusion_sta": "global_gap_repair",
            "conflict": 0,
            "gap_length": distance,
            "gap_score": score,
            "evidence": evidence["evidence_mode"],
            "center_p": evidence.get("center_probability_mean", 0.0),
            "center_q25": evidence.get("center_probability_q25", 0.0),
            "center_n": evidence.get("center_probability_normalized_mean", 0.0),
            "center_cov": evidence.get("center_probability_high_support_ratio", 0.0),
            "surface_cov": evidence.get("surface_support_ratio", evidence.get("straight_surface_support_ratio", 0.0)),
            "path_ratio": evidence.get("path_ratio", 1.0),
            "alignment": evidence.get("direction_alignment", 0.0),
            "width_src": "attached_road" if connected_widths else "network_median_fallback",
            "auto_state": "accepted",
            "target": evidence["target_kind"],
            "geometry": connector,
        })
        used_endpoints.add(start)
        if evidence["target_kind"] == "endpoint":
            used_endpoints.add(end)
        else:
            split_points.append(Point(end))

    # Materialize every endpoint-to-edge attachment as shared topology.  The
    # exported target feature is split at the connector endpoint instead of
    # relying on a merely visual line intersection.
    noded_centerlines: list[dict] = []
    split_tolerance = max(1e-8, 0.05 * pixel_size)
    for row in centerlines:
        geometry = row.get("geometry")
        if geometry is None or geometry.is_empty or geometry.geom_type != "LineString":
            noded_centerlines.append(row)
            continue
        positions = sorted({
            float(geometry.project(point))
            for point in split_points
            if point.distance(geometry) <= split_tolerance
            and split_tolerance < geometry.project(point) < geometry.length - split_tolerance
        })
        if not positions:
            noded_centerlines.append(row)
            continue
        boundaries = [0.0, *positions, float(geometry.length)]
        for first, second in zip(boundaries[:-1], boundaries[1:]):
            segment = substring(geometry, first, second)
            if segment.geom_type == "LineString" and segment.length > split_tolerance:
                noded_centerlines.append({**row, "geometry": segment})
    result = [*noded_centerlines, *additions]
    for global_id, row in enumerate(result):
        row["global_id"] = global_id
    return result, len(additions)


def _merge_global_gap_surfaces(road_surface, centerlines: list[dict], pixel_size: float):
    """Buffer export-time gap connectors and merge them into the final pavement."""
    connectors = []
    pixel_size = max(float(pixel_size), 1e-9)
    for row in centerlines:
        if row.get("fusion_sta") != "global_gap_repair":
            continue
        geometry = row.get("geometry")
        try:
            width_map = float(row.get("width_map", 0.0) or 0.0)
        except (TypeError, ValueError):
            width_map = 0.0
        if geometry is None or geometry.is_empty or width_map <= 0:
            continue
        connectors.append(geometry.buffer(
            max(0.5 * width_map, 1.5 * pixel_size), cap_style=2, join_style=2
        ))
    if not connectors:
        return road_surface, 0, 0.0
    original_area = float(road_surface.area) if road_surface is not None and not road_surface.is_empty else 0.0
    parts = [*connectors]
    if road_surface is not None and not road_surface.is_empty:
        parts.insert(0, road_surface)
    merged = unary_union(parts).buffer(0)
    return merged, len(connectors), max(0.0, float(merged.area) - original_area)


def _fuse_centerline_records(
    centerlines: list[dict],
    source_contexts: dict[str, dict],
    canonical_network=None,
    match_tolerance: float = 1.5,
) -> list[dict]:
    """Conflate duplicate tile lines onto one network and arbitrate width attributes."""
    source_parts = [
        {**row, "geometry": part}
        for row in centerlines
        for part in _line_parts(row.get("geometry"))
    ]
    if not source_parts:
        return []
    # Preserve every source centerline geometry.  Automatic endpoint snapping
    # can manufacture short or long connector segments that do not follow a
    # surface skeleton, so only exact geometric intersections are noded here.
    network = canonical_network if canonical_network is not None and not canonical_network.is_empty else unary_union(
        [row["geometry"] for row in source_parts]
    )
    try:
        network = linemerge(network)
    except ValueError:
        pass
    fused_parts = _line_parts(network)
    quality_weight = {"A": 1.0, "B": 0.65, "C": 0.35}
    quality_score = {"A": 3.0, "B": 2.0, "C": 1.0}
    manual_sources = {"manual_boundary_measurement"}
    fused: list[dict] = []
    tolerance = max(float(match_tolerance), 1e-6)

    for global_id, part in enumerate(fused_parts):
        if part.length <= 1e-6:
            continue
        corridor = part.buffer(tolerance, cap_style=2, join_style=2)
        midpoint = part.interpolate(0.5, normalized=True)
        part_direction = _line_direction(part)
        observations = []
        for row in source_parts:
            geometry = row["geometry"]
            if geometry.distance(part) > tolerance:
                continue
            if abs(float(np.dot(part_direction, _line_direction(geometry)))) < 0.70:
                continue
            support_length = float(geometry.intersection(corridor).length)
            support_ratio = min(1.0, support_length / max(float(part.length), tolerance))
            if support_ratio <= 0.01:
                continue
            try:
                width_map = float(row.get("width_map", 0.0) or 0.0)
            except (TypeError, ValueError):
                width_map = 0.0
            if not np.isfinite(width_map) or width_map <= 0:
                continue
            grade = str(row.get("quality_grade", "C") or "C").upper()
            context = source_contexts.get(str(row.get("tile_stem", "")), {})
            footprint = context.get("footprint")
            feather_distance = max(float(context.get("feather_distance", tolerance) or tolerance), tolerance)
            edge_weight = 1.0
            if footprint is not None and not footprint.is_empty:
                edge_distance = float(midpoint.distance(footprint.boundary)) if footprint.covers(midpoint) else 0.0
                edge_weight = float(np.clip(edge_distance / feather_distance, 0.05, 1.0))
            weight = max(1e-6, support_ratio * edge_weight * quality_weight.get(grade, 0.35))
            observations.append((
                width_map, weight, grade, str(row.get("tile_stem", "")),
                str(row.get("width_source", row.get("optimized_width_source", "")) or ""),
            ))

        if observations:
            observations_by_tile: dict[str, list[tuple[float, float, str, str]]] = {}
            for value, weight, grade, tile_stem, width_source in observations:
                observations_by_tile.setdefault(tile_stem, []).append((value, weight, grade, width_source))
            tile_observations = []
            for tile_stem, items in observations_by_tile.items():
                item_values = [item[0] for item in items]
                item_weights = [item[1] for item in items]
                item_total = max(sum(item_weights), 1e-9)
                tile_width = _weighted_median(item_values, item_weights)
                tile_grade_score = sum(quality_score.get(item[2], 1.0) * item[1] for item in items) / item_total
                tile_grade = "A" if tile_grade_score >= 2.5 else "B" if tile_grade_score >= 1.5 else "C"
                tile_source = "manual_boundary_measurement" if any(item[3] in manual_sources for item in items) else "automatic"
                tile_observations.append((tile_width, item_total, tile_grade, tile_stem, tile_source))
            manual_tiles = [item for item in tile_observations if item[4] == "manual_boundary_measurement"]
            if manual_tiles:
                # A saved boundary measurement is the authoritative width for
                # that fused road, even when overlapping automatic tiles disagree.
                tile_observations = manual_tiles
            values = [item[0] for item in tile_observations]
            weights = [item[1] for item in tile_observations]
            width_map = _weighted_median(values, weights)
            total_weight = max(sum(weights), 1e-9)
            mean_width = sum(value * weight for value, weight in zip(values, weights)) / total_weight
            width_std = float(np.sqrt(sum(weight * (value - mean_width) ** 2 for value, weight in zip(values, weights)) / total_weight))
            grade_value = sum(quality_score.get(grade, 1.0) * weight for (_, weight, grade, _, _) in tile_observations) / total_weight
            quality_grade = "A" if grade_value >= 2.5 else "B" if grade_value >= 1.5 else "C"
            conflict = int((max(values) - min(values)) / max(width_map, 1e-6) > 0.25) if len(values) > 1 else 0
            source_tiles = sorted({item[3] for item in tile_observations if item[3]})
            width_source = "manual_boundary_measurement" if manual_tiles else "automatic_fusion"
        else:
            width_map = 0.0
            width_std = 0.0
            quality_grade = "C"
            conflict = 1
            source_tiles = []
            width_source = "unresolved"
        fused.append({
            "global_id": global_id,
            "width_map": width_map,
            "width_std": width_std,
            "src_count": len(source_tiles),
            "src_tiles": ",".join(source_tiles),
            "quality_gr": quality_grade,
            "fusion_sta": "conflict" if conflict else "fused",
            "conflict": conflict,
            "width_src": width_source,
            "geometry": part,
        })
    return fused


def _fuse_surface_masks(
    surface_sources: list[dict],
    crs,
    feather_pixels: float = 256.0,
    continuous: bool = False,
):
    """Blend georeferenced binary masks or continuous probabilities."""
    if not surface_sources:
        return None, None, None
    x_resolution = float(np.median([abs(source["transform"].a) for source in surface_sources]))
    y_resolution = float(np.median([abs(source["transform"].e) for source in surface_sources]))
    bounds = [array_bounds(source["shape"][0], source["shape"][1], source["transform"]) for source in surface_sources]
    origin_x = float(surface_sources[0]["transform"].c)
    origin_y = float(surface_sources[0]["transform"].f)
    left = origin_x + np.floor((min(item[0] for item in bounds) - origin_x) / x_resolution) * x_resolution
    right = origin_x + np.ceil((max(item[2] for item in bounds) - origin_x) / x_resolution) * x_resolution
    bottom = origin_y + np.floor((min(item[1] for item in bounds) - origin_y) / y_resolution) * y_resolution
    top = origin_y + np.ceil((max(item[3] for item in bounds) - origin_y) / y_resolution) * y_resolution
    width = max(1, int(round((right - left) / x_resolution)))
    height = max(1, int(round((top - bottom) / y_resolution)))
    transform = rasterio.Affine(x_resolution, 0.0, left, 0.0, -y_resolution, top)
    numerator = np.zeros((height, width), dtype=np.float32)
    denominator = np.zeros((height, width), dtype=np.float32)

    for source in surface_sources:
        mask = np.asarray(source["mask"], dtype=np.float32)
        if continuous:
            if mask.size and float(np.nanmax(mask)) > 1.0:
                mask = mask / 255.0
            mask = np.clip(mask, 0.0, 1.0)
        else:
            mask = (mask > 0).astype(np.float32)
        coverage = np.ones(mask.shape, dtype=np.float32)
        edge_seed = np.ones(mask.shape, dtype=np.uint8)
        edge_seed[[0, -1], :] = 0
        edge_seed[:, [0, -1]] = 0
        edge_distance = cv2.distanceTransform(edge_seed, cv2.DIST_L2, 5)
        weight = np.clip(edge_distance / max(float(feather_pixels), 1.0), 0.05, 1.0).astype(np.float32)
        warped_mask = np.zeros((height, width), dtype=np.float32)
        warped_weight = np.zeros((height, width), dtype=np.float32)
        warped_coverage = np.zeros((height, width), dtype=np.float32)
        for source_array, destination, resampling in (
            (mask, warped_mask, Resampling.bilinear),
            (weight, warped_weight, Resampling.bilinear),
            (coverage, warped_coverage, Resampling.nearest),
        ):
            reproject(
                source=source_array,
                destination=destination,
                src_transform=source["transform"],
                src_crs=crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=resampling,
                src_nodata=None,
                dst_nodata=0,
            )
        effective_weight = warped_weight * (warped_coverage > 0.5)
        numerator += warped_mask * effective_weight
        denominator += effective_weight
    probability = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-6)
    if continuous:
        fused = probability
    else:
        fused = (probability >= 0.5).astype(np.uint8)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    return fused, transform, probability


def _write_final_visualization(
    layers: dict[str, list[dict]],
    crs,
    output: Path,
    background_images: list[Path] | None = None,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    legend_items = []

    background_images = background_images or []
    display_bounds = []
    # Keep the complete image mosaic while bounding total display memory.
    max_tile_dimension = min(1024, max(256, int(np.sqrt(16_000_000 / max(len(background_images), 1)))))
    for image_path in background_images:
        with rasterio.open(image_path) as dataset:
            scale = min(1.0, max_tile_dimension / max(dataset.width, dataset.height))
            out_height = max(1, int(round(dataset.height * scale)))
            out_width = max(1, int(round(dataset.width * scale)))
            indexes = list(range(1, min(dataset.count, 3) + 1))
            raster = dataset.read(indexes, out_shape=(len(indexes), out_height, out_width))
            if len(indexes) == 1:
                raster = np.repeat(raster, 3, axis=0)
            raster = np.moveaxis(raster[:3], 0, -1).astype(np.float32)
            valid_pixels = raster[np.isfinite(raster)]
            if valid_pixels.size:
                low, high = np.percentile(valid_pixels, [2, 98])
                raster = np.clip((raster - low) / max(high - low, 1e-6), 0, 1)
            bounds = dataset.bounds
            display_bounds.append(bounds)
        ax.imshow(raster, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin="upper")

    if display_bounds:
        span_x = max(item.right for item in display_bounds) - min(item.left for item in display_bounds)
        span_y = max(item.top for item in display_bounds) - min(item.bottom for item in display_bounds)
        aspect = span_x / max(span_y, 1e-9)
        fig.set_size_inches(15, float(np.clip(15 / max(aspect, 0.35) + 1.2, 6.5, 12.0)))

    surfaces = layers["final_road_surfaces"]
    if surfaces:
        gpd.GeoDataFrame(surfaces, geometry="geometry", crs=crs).plot(
            ax=ax, color="#00d084", edgecolor="#004d40", linewidth=0.8, alpha=0.34,
        )
        legend_items.append(Patch(facecolor="#00d084", edgecolor="#004d40", alpha=0.55, label="Final road surface"))

    centerlines = layers["final_centerlines"]
    if centerlines:
        centerline_frame = gpd.GeoDataFrame(centerlines, geometry="geometry", crs=crs)
        # At 180 dpi, 0.8 pt renders at approximately 2 screen pixels.
        centerline_frame.plot(ax=ax, color="#ffd400", linewidth=0.8, alpha=0.98)
        legend_items.append(Line2D([0], [0], color="#ffd400", linewidth=0.8, label="Final centerline"))
        unresolved = [row for row in centerlines if float(row.get("width_map", 0) or 0) <= 0]
        if unresolved:
            gpd.GeoDataFrame(unresolved, geometry="geometry", crs=crs).plot(
                ax=ax, color="#ff2d55", linewidth=1.5, linestyle="--",
            )
            legend_items.append(Line2D([0], [0], color="#ff2d55", linestyle="--", label="Width fallback"))

    issues = layers["final_review_issues"]
    if issues:
        issue_frame = gpd.GeoDataFrame(issues, geometry="geometry", crs=crs)
        if len(issue_frame) > 500:
            issue_frame = issue_frame.iloc[np.linspace(0, len(issue_frame) - 1, 500, dtype=int)]
        issue_frame.plot(
            ax=ax, color="#d00000", marker="x", markersize=7, linewidth=0.55, alpha=0.55,
        )
        legend_items.append(Line2D([0], [0], color="#d00000", marker="x", linestyle="None", label="Review issue"))

    ax.set_title("Original Imagery + Final Road Surface + Centerline", fontsize=16, fontweight="bold", pad=14)
    subtitle = (
        f"Images: {len(background_images):,}   Road surfaces: {len(surfaces):,}   "
        f"Centerlines: {len(centerlines):,}   "
        f"Review issues: {len(issues):,}"
    )
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#52606d")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    if legend_items:
        ax.legend(handles=legend_items, loc="lower left", frameon=True, framealpha=0.92)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(output),
        "display_mode": "imagery_surface_centerline_overlay",
        "background_image_count": len(background_images),
        "road_surface_count": len(surfaces),
        "centerline_count": len(centerlines),
        "width_colored_feature_count": 0,
    }


def export_final_products(
    final_dir: Path,
    image_dir: Path,
    output: Path | None = None,
    shp_dir: Path | None = None,
    visualization: Path | None = None,
    centerline_shp: Path | None = None,
    surface_shp: Path | None = None,
    stitched_centerlines: Path | None = None,
) -> dict:
    centerlines: list[dict] = []
    road_surfaces: list[dict] = []
    surface_sources: list[dict] = []
    centerline_probability_sources: list[dict] = []
    width_samples: list[dict] = []
    width_segments: list[dict] = []
    issues: list[dict] = []
    surface_added: list[dict] = []
    surface_removed: list[dict] = []
    surface_uncertain: list[dict] = []
    width_surface_added: list[dict] = []
    width_surface_removed: list[dict] = []
    background_images: list[Path] = []
    source_contexts: dict[str, dict] = {}
    crs = None
    for summary_path in summaries(final_dir, optimized=True):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stem = stem_from_summary(summary_path)
        image_path = Path(summary.get("image", ""))
        if not image_path.is_file():
            matches = list(image_dir.glob(f"{stem}.*"))
            if not matches:
                continue
            image_path = matches[0]
        with rasterio.open(image_path) as dataset:
            transform, tile_crs, bounds = dataset.transform, dataset.crs, dataset.bounds
            raster_shape = (dataset.height, dataset.width)
        if image_path not in background_images:
            background_images.append(image_path)
        if crs is None: crs = tile_crs
        if tile_crs != crs: raise ValueError("All final-product rasters must use the same CRS")
        pixel_size = float(summary.get("pixel_size", 1.0))
        source_contexts[stem] = {
            "footprint": box(bounds.left, bounds.bottom, bounds.right, bounds.top),
            "feather_distance": max(pixel_size * 256.0, pixel_size),
        }
        nodes, edges = load_graph(final_dir / summary["outputs"]["optimized_graph"])
        edge_rows = read_csv(final_dir / summary["outputs"]["optimized_edges"])
        edge_by_id = {int(row.get("final_edge_id", -1)): row for row in edge_rows if row.get("final_status") != "removed_by_review"}
        tile_centerline_geometries = []
        for chain, points in _chain_segments(nodes, edges):
            widths = [float(edge_by_id.get(edge_id, {}).get("optimized_width_px", 0) or 0) for edge_id in chain.edge_ids]
            valid = [value for value in widths if value > 0]
            width_px = float(np.median(valid)) if valid else 0.0
            grades = [str(edge_by_id.get(edge_id, {}).get("optimized_quality_grade", "C") or "C") for edge_id in chain.edge_ids]
            grade_counts = {grade: grades.count(grade) for grade in {"A", "B", "C"}}
            direct_ratio = grade_counts["A"] / max(len(grades), 1)
            automatic_ratio = grade_counts["C"] / max(len(grades), 1)
            if direct_ratio >= 0.8:
                quality_grade = "A"
            elif automatic_ratio < 0.5 and grade_counts["A"] + grade_counts["B"] > 0:
                quality_grade = "B"
            else:
                quality_grade = "C"
            line_geometry = _world_line(points, transform)
            tile_centerline_geometries.append(line_geometry)
            centerlines.append({
                "tile_stem": stem, "road_id": f"{stem}:{chain.chain_id}", "width_px": width_px,
                "width_map": width_px * float(summary.get("pixel_size", 1.0)),
                "edge_count": len(chain.edge_ids), "quality_grade": quality_grade,
                "direct_measurement_ratio": direct_ratio, "automatic_estimate_ratio": automatic_ratio,
                "quality": {"A": "direct_measurement", "B": "chain_interpolation", "C": "automatic_estimate"}.get(quality_grade, "automatic_estimate"),
                "width_source": (
                    "manual_boundary_measurement"
                    if any(str(edge_by_id.get(edge_id, {}).get("optimized_width_source", "")) == "manual_boundary_measurement" for edge_id in chain.edge_ids)
                    else "automatic"
                ),
                "geometry": line_geometry,
            })
            if not valid:
                midpoint = points[len(points) // 2]
                x, y = rasterio.transform.xy(transform, midpoint[0], midpoint[1], offset="center")
                issues.append({"tile_stem": stem, "issue": "width_unresolved", "severity": "high", "feature_id": f"chain:{chain.chain_id}", "geometry": Point(x, y)})
        mask_path = final_dir / summary["outputs"].get("optimized_road_surface", "")
        mask = _read_image(mask_path)
        tile_surface_geometries = []
        if mask is not None:
            for geom, value in shapes((mask > 0).astype(np.uint8), mask=mask > 0, transform=transform):
                if value:
                    tile_surface_geometries.append(shape(geom))
        combined_surface = unary_union(tile_surface_geometries) if tile_surface_geometries else None
        if combined_surface is not None and not combined_surface.is_empty:
            combined_surface = _clean_surface(combined_surface.buffer(0), 50.0 * pixel_size * pixel_size, 25.0 * pixel_size * pixel_size)
            if combined_surface is not None:
                road_surfaces.append({"tile_stem": stem, "source": "sammolra_structural_repair", "geometry": combined_surface})
            surface_sources.append({
                "tile_stem": stem,
                "mask": mask,
                "transform": transform,
                "shape": mask.shape,
                "source": str(mask_path),
            })
        probability_candidates = [final_dir / f"{stem}_centerline_probability.png"]
        source_output_dir = str(summary.get("source_output_dir", "") or "").strip()
        if source_output_dir:
            probability_candidates.append(Path(source_output_dir) / f"{stem}_centerline_probability.png")
        probability_path = next((path for path in probability_candidates if path.is_file()), None)
        if probability_path is not None:
            probability_bytes = np.frombuffer(probability_path.read_bytes(), dtype=np.uint8)
            centerline_probability = cv2.imdecode(probability_bytes, cv2.IMREAD_GRAYSCALE)
            if centerline_probability is not None and centerline_probability.shape == raster_shape:
                centerline_probability_sources.append({
                    "tile_stem": stem,
                    "mask": centerline_probability.astype(np.float32) / 255.0,
                    "transform": transform,
                    "shape": centerline_probability.shape,
                    "source": str(probability_path),
                })

        audit_layers = {
            "surface_added": surface_added,
            "surface_removed": surface_removed,
            "surface_uncertain": surface_uncertain,
        }
        for audit_key, records in {
            **audit_layers,
            "width_surface_added": width_surface_added,
            "width_surface_removed": width_surface_removed,
        }.items():
            audit_value = str(summary.get(audit_key, "") or summary.get("outputs", {}).get(audit_key, "") or "")
            if not audit_value:
                continue
            audit_path = Path(audit_value)
            if not audit_path.is_absolute():
                audit_path = final_dir / audit_path
            audit_mask = _read_image(audit_path)
            if audit_mask is None:
                continue
            for geom, value in shapes((audit_mask > 0).astype(np.uint8), mask=audit_mask > 0, transform=transform):
                if value:
                    records.append({"tile_stem": stem, "source_raster": str(audit_path), "geometry": shape(geom)})
        for row in read_csv(final_dir / summary["outputs"].get("optimized_width_samples", "")):
            x, y = rasterio.transform.xy(transform, float(row.get("row_used", row.get("row", 0))), float(row.get("col_used", row.get("col", 0))), offset="center")
            record = {"tile_stem": stem, **{key: value for key, value in row.items() if key not in {"geometry"}}, "geometry": Point(x, y)}
            width_samples.append(record)
            # Sampling flags remain in final_width_samples for traceability. They are
            # automatic rejection reasons, not tasks requiring another manual review.
        for row in read_csv(final_dir / summary["outputs"].get("optimized_width_segments", "")):
            points = [(float(row["start_row"]), float(row["start_col"])), (float(row["end_row"]), float(row["end_col"]))]
            width_segments.append({"tile_stem": stem, **row, "geometry": _world_line(points, transform)})

    canonical_network = None
    if stitched_centerlines is not None and stitched_centerlines.is_file():
        stitched_layers = set(gpd.list_layers(stitched_centerlines)["name"].tolist())
        stitched_layer = "stitched_centerlines" if "stitched_centerlines" in stitched_layers else next(iter(stitched_layers), "")
        if stitched_layer:
            stitched_frame = gpd.read_file(stitched_centerlines, layer=stitched_layer)
            if not stitched_frame.empty:
                if stitched_frame.crs != crs:
                    stitched_frame = stitched_frame.to_crs(crs)
                canonical_network = unary_union([geometry for geometry in stitched_frame.geometry if geometry is not None and not geometry.is_empty])
    pixel_sizes = [context["feather_distance"] / 256.0 for context in source_contexts.values()]
    match_tolerance = max(1.5 * float(np.median(pixel_sizes)), 1e-6) if pixel_sizes else 1.5
    fused_centerlines = _fuse_centerline_records(centerlines, source_contexts, canonical_network, match_tolerance)
    fused_surface_mask, fused_surface_transform, _ = _fuse_surface_masks(surface_sources, crs)
    fused_centerline_probability, fused_centerline_transform, _ = _fuse_surface_masks(
        centerline_probability_sources,
        crs,
        continuous=True,
    )
    fused_surface_geometries = []
    if fused_surface_mask is not None and np.any(fused_surface_mask):
        fused_surface_geometries = [
            shape(geometry)
            for geometry, value in shapes(
                fused_surface_mask,
                mask=fused_surface_mask > 0,
                transform=fused_surface_transform,
            )
            if value
        ]
    fused_surface_geometry = unary_union(fused_surface_geometries).buffer(0) if fused_surface_geometries else None
    if fused_surface_geometry is not None:
        pixel_area = float(np.median(pixel_sizes)) ** 2 if pixel_sizes else 1.0
        fused_surface_geometry = _clean_surface(fused_surface_geometry, 50.0 * pixel_area, 25.0 * pixel_area)
    fused_centerlines, global_gap_count = _connect_surface_supported_global_gaps(
        fused_centerlines,
        fused_surface_geometry,
        float(np.median(pixel_sizes)) if pixel_sizes else 1.0,
        centerline_probability=fused_centerline_probability,
        centerline_transform=fused_centerline_transform,
    )
    fused_surface_geometry, global_gap_surface_count, global_gap_surface_added_area = _merge_global_gap_surfaces(
        fused_surface_geometry,
        fused_centerlines,
        float(np.median(pixel_sizes)) if pixel_sizes else 1.0,
    )
    fused_surfaces = ([{"source": "sammolra_feathered_surface_fusion_with_global_gap_buffers", "geometry": fused_surface_geometry}]
                      if fused_surface_geometry is not None and not fused_surface_geometry.is_empty else [])

    layers = {
        "final_centerlines": fused_centerlines, "final_road_surfaces": fused_surfaces,
        "tile_centerlines": centerlines, "tile_road_surfaces": road_surfaces,
        "final_width_samples": width_samples, "final_width_segments": width_segments,
        "final_review_issues": issues,
        "surface_added": surface_added, "surface_removed": surface_removed,
        "surface_uncertain": surface_uncertain,
        "width_surface_added": width_surface_added,
        "width_surface_removed": width_surface_removed,
    }

    if shp_dir is not None:
        centerline_shp = centerline_shp or shp_dir / "road_centerlines.shp"
        surface_shp = surface_shp or shp_dir / "road_surfaces.shp"
    if centerline_shp is None and surface_shp is None and output is None:
        raise ValueError("At least one SHP output or GeoPackage output is required")

    if centerline_shp is not None:
        centerline_shp.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(fused_centerlines, geometry="geometry", crs=crs).to_file(centerline_shp, driver="ESRI Shapefile")
    if surface_shp is not None:
        surface_shp.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(fused_surfaces, geometry="geometry", crs=crs).to_file(surface_shp, driver="ESRI Shapefile")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        for layer, rows in layers.items():
            if rows:
                gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_file(output, layer=layer, driver="GPKG")

    report_dir = (
        centerline_shp.parent if centerline_shp is not None
        else surface_shp.parent if surface_shp is not None
        else output.parent
    )
    visualization = visualization or report_dir / "road_width_overview.png"
    visualization_report = _write_final_visualization(layers, crs, visualization, background_images)
    report = {
        "geopackage": str(output) if output is not None else "",
        "centerline_shp": str(centerline_shp) if centerline_shp is not None else "",
        "surface_shp": str(surface_shp) if surface_shp is not None else "",
        "visualization": visualization_report,
        "fusion": {
            "canonical_network": str(stitched_centerlines) if stitched_centerlines is not None else "",
            "match_tolerance": match_tolerance,
            "global_endpoint_snapping": False,
            "global_surface_gap_count": global_gap_count,
            "global_gap_surface_count": global_gap_surface_count,
            "global_gap_surface_added_area": global_gap_surface_added_area,
            "global_probability_gap_count": sum(
                row.get("fusion_sta") == "global_gap_repair"
                and str(row.get("evidence", "")).startswith("centerline_probability")
                for row in fused_centerlines
            ),
            "global_endpoint_gap_count": sum(
                row.get("fusion_sta") == "global_gap_repair" and row.get("target") == "endpoint"
                for row in fused_centerlines
            ),
            "global_edge_attachment_count": sum(
                row.get("fusion_sta") == "global_gap_repair" and row.get("target") == "edge"
                for row in fused_centerlines
            ),
            "centerline_probability_source_count": len(centerline_probability_sources),
            "gap_decision_policy": "fully_automatic_accept_or_skip_without_pending_review",
            "centerline_geometry_policy": FINAL_CENTERLINE_GEOMETRY_POLICY,
            "tile_centerline_count": len(centerlines),
            "fused_centerline_count": len(fused_centerlines),
            "conflict_count": sum(int(row.get("conflict", 0)) for row in fused_centerlines),
            "surface_policy": "sammolra_feathered_surface_fusion_with_global_gap_buffers",
            "surface_source_count": len(surface_sources),
        },
        **{f"{key}_count": len(value) for key, value in layers.items()},
    }
    (report_dir / "final_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def stitch_final_products(input_gpkg: Path, output: Path, snap_tolerance: float) -> dict:
    layer_names = set(gpd.list_layers(input_gpkg)["name"].tolist())
    layer = "final_centerlines" if "final_centerlines" in layer_names else "editable_centerlines"
    lines = gpd.read_file(input_gpkg, layer=layer)
    if "active" in lines.columns:
        lines = lines[lines.apply(_active, axis=1)]
    if lines.empty:
        raise ValueError("No final centerlines to stitch")
    geometries = [geom for geom in lines.geometry if geom is not None and not geom.is_empty]
    reference = unary_union(geometries)
    # First cluster nearby endpoints, then snap every line to the shared endpoint set.
    endpoints = [Point(coord) for geometry in geometries for coord in (geometry.coords[0], geometry.coords[-1])]
    representatives: list[Point] = []
    for endpoint in endpoints:
        match = next((item for item in representatives if endpoint.distance(item) <= snap_tolerance), None)
        if match is None:
            representatives.append(endpoint)
    endpoint_reference = unary_union(representatives)
    snapped = unary_union([snap(snap(geometry, endpoint_reference, snap_tolerance), reference, snap_tolerance) for geometry in geometries])
    try:
        snapped = linemerge(snapped)
    except ValueError:
        pass
    merged = gpd.GeoDataFrame([{"network_id": 0, "source_count": len(geometries), "snap_tolerance": snap_tolerance, "geometry": snapped}], geometry="geometry", crs=lines.crs)
    if output.exists(): output.unlink()
    merged.to_file(output, layer="stitched_centerlines", driver="GPKG")
    graph = nx.Graph()
    for part in _line_parts(snapped):
        coords = list(part.coords)
        if len(coords) >= 2:
            start = (round(coords[0][0], 6), round(coords[0][1], 6))
            end = (round(coords[-1][0], 6), round(coords[-1][1], 6))
            graph.add_edge(start, end)
    report = {
        "source_feature_count": len(geometries), "output": str(output),
        "component_count": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "dangling_endpoint_count": sum(degree == 1 for _, degree in graph.degree()),
        "junction_count": sum(degree >= 3 for _, degree in graph.degree()),
        "note": "Overlaps are dissolved and nearby endpoints are snapped in map coordinates.",
    }
    (output.parent / "stitch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def stitch_edit_package(input_gpkg: Path, review_dir: Path, edited_dir: Path, output: Path, snap_tolerance: float) -> dict:
    """Snap all edited tiles globally, then clip the noded network back to each raster."""
    lines = gpd.read_file(input_gpkg, layer="editable_centerlines")
    if "active" in lines.columns:
        lines = lines[lines.apply(_active, axis=1)]
    if lines.empty:
        raise ValueError("No active edited centerlines to stitch")
    geometries = [part for geometry in lines.geometry for part in _line_parts(geometry)]
    reference = unary_union(geometries)
    endpoints = [Point(coord) for geometry in geometries for coord in (geometry.coords[0], geometry.coords[-1])]
    representatives: list[Point] = []
    for endpoint in endpoints:
        if not any(endpoint.distance(item) <= snap_tolerance for item in representatives):
            representatives.append(endpoint)
    endpoint_reference = unary_union(representatives)
    network = unary_union([snap(snap(geometry, endpoint_reference, snap_tolerance), reference, snap_tolerance) for geometry in geometries])

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    gpd.GeoDataFrame(
        [{"network_id": 0, "source_count": len(geometries), "geometry": network}],
        geometry="geometry", crs=lines.crs,
    ).to_file(output, layer="stitched_centerlines", driver="GPKG")

    # Import only edited centerlines; road surfaces are reconstructed after global noding.
    manifest = import_edit_package(input_gpkg, review_dir, edited_dir, snap_tolerance_px=2.0)
    total_edges = 0
    for summary_path in summaries(review_dir):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stem = stem_from_summary(summary_path)
        image_path = Path(summary["image"])
        with rasterio.open(image_path) as dataset:
            transform, raster_crs, bounds = dataset.transform, dataset.crs, dataset.bounds
        tile_network = gpd.GeoSeries([network], crs=lines.crs).to_crs(raster_crs).iloc[0]
        clipped = tile_network.intersection(box(bounds.left, bounds.bottom, bounds.right, bounds.top))
        inverse = ~transform
        pixel_lines = [_pixel_line(part, inverse) for part in _line_parts(clipped)]
        noded = _node_pixel_lines(pixel_lines, 2.0) if pixel_lines else None
        nodes: list[tuple[float, float]] = []
        edges: list[tuple[int, int]] = []
        for part in _line_parts(noded):
            previous = None
            for col, row in part.coords:
                node_id = _snap_node(nodes, (float(row), float(col)), 0.05)
                if previous is not None and previous != node_id:
                    edges.append((previous, node_id))
                previous = node_id
        graph_path = edited_dir / f"{stem}_edited_graph.p"
        save_graph(graph_path, nodes, edges)
        manifest["tiles"][stem]["graph"] = str(graph_path)
        manifest["tiles"][stem]["line_count"] = len(edges)
        total_edges += len(edges)
    manifest.update({"stitched_network": str(output), "global_source_count": len(geometries), "stitched_tile_edge_count": total_edges})
    (edited_dir / "edited_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def stitch_edited_directory(
    review_dir: Path,
    edited_dir: Path,
    output: Path,
    snap_tolerance: float,
    surface_low_probability: float = 0.30,
    surface_high_probability: float = 0.55,
    surface_max_corridor_px: float = 60.0,
    only_stems: set[str] | None = None,
) -> dict:
    """Globally stitch built-in/QGIS-imported edited graphs and write tile graphs back."""
    geometries = []
    common_crs = None
    contexts = {}
    for summary_path in summaries(review_dir):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stem = stem_from_summary(summary_path)
        graph_path = edited_dir / f"{stem}_edited_graph.p"
        if not graph_path.is_file():
            raise FileNotFoundError(f"Missing edited graph for {stem}; save/import step 2 first")
        nodes, edges = load_graph(graph_path)
        image_path = Path(summary["image"])
        with rasterio.open(image_path) as dataset:
            transform, raster_crs, bounds = dataset.transform, dataset.crs, dataset.bounds
        if common_crs is None:
            common_crs = raster_crs
        if raster_crs != common_crs:
            raise ValueError("All edited rasters must use the same CRS for global stitching")
        contexts[stem] = (transform, raster_crs, bounds, summary)
        for src, dst in edges.tolist():
            geometries.append(_world_line([tuple(nodes[src]), tuple(nodes[dst])], transform))
    if not geometries:
        raise ValueError("No edited centerlines to stitch")

    reference = unary_union(geometries)
    endpoints = [Point(coord) for geometry in geometries for coord in (geometry.coords[0], geometry.coords[-1])]
    representatives: list[Point] = []
    for endpoint in endpoints:
        if not any(endpoint.distance(item) <= snap_tolerance for item in representatives):
            representatives.append(endpoint)
    endpoint_reference = unary_union(representatives)
    network = unary_union([snap(snap(geometry, endpoint_reference, snap_tolerance), reference, snap_tolerance) for geometry in geometries])

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    gpd.GeoDataFrame(
        [{"network_id": 0, "source_count": len(geometries), "geometry": network}],
        geometry="geometry", crs=common_crs,
    ).to_file(output, layer="stitched_centerlines", driver="GPKG")

    manifest_path = edited_dir / "edited_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"tiles": {}}
    except (OSError, json.JSONDecodeError):
        manifest = {"tiles": {}}
    total_edges = 0
    reconstruction_totals = {"added_surface_px": 0, "removed_surface_px": 0, "uncertain_surface_px": 0}
    for stem, (transform, raster_crs, bounds, summary) in contexts.items():
        clipped = network.intersection(box(bounds.left, bounds.bottom, bounds.right, bounds.top))
        inverse = ~transform
        pixel_lines = [_pixel_line(part, inverse) for part in _line_parts(clipped)]
        noded = _node_pixel_lines(pixel_lines, 2.0) if pixel_lines else None
        nodes: list[tuple[float, float]] = []
        edges: list[tuple[int, int]] = []
        for part in _line_parts(noded):
            previous = None
            for col, row in part.coords:
                node_id = _snap_node(nodes, (float(row), float(col)), 0.05)
                if previous is not None and previous != node_id:
                    edges.append((previous, node_id))
                previous = node_id
        graph_path = edited_dir / f"{stem}_edited_graph.p"
        save_graph(graph_path, nodes, edges)
        tile = manifest.setdefault("tiles", {}).setdefault(stem, {})
        tile.update({"graph": str(graph_path), "line_count": len(edges)})
        total_edges += len(edges)
        if only_stems is not None and stem not in only_stems:
            tile["surface_rebuild_skipped"] = True
            continue
        probability_path = review_dir / f"{stem}_road_probability.png"
        original_surface_path = review_dir / f"{stem}_molra_clean_mask.png"
        probability = _read_image(probability_path)
        original_surface = _read_image(original_surface_path)
        if probability is None or original_surface is None:
            raise FileNotFoundError(f"Missing road probability or original surface for {stem}")
        reconstruction = reconstruct_surface(
            probability,
            original_surface,
            np.asarray(nodes, dtype=np.float32),
            np.asarray(edges, dtype=np.int32),
            SurfaceReconstructionConfig(
                low_probability=surface_low_probability,
                high_probability=surface_high_probability,
                max_corridor_px=surface_max_corridor_px,
            ),
        )
        manual_add_path = edited_dir / f"{stem}_manual_surface_add.png"
        manual_remove_path = edited_dir / f"{stem}_manual_surface_remove.png"
        manual_add = _read_image(manual_add_path)
        manual_remove = _read_image(manual_remove_path)
        final_surface = reconstruction.surface.copy()
        if manual_add is not None and manual_add.shape == final_surface.shape:
            final_surface[manual_add > 0] = 1
        if manual_remove is not None and manual_remove.shape == final_surface.shape:
            final_surface[manual_remove > 0] = 0
        reconstruction.surface = final_surface.astype(np.uint8)
        reconstruction.added = (reconstruction.surface > (original_surface > 0)).astype(np.uint8)
        reconstruction.removed = ((original_surface > 0) > reconstruction.surface).astype(np.uint8)
        reconstruction.metadata.update({
            "manual_surface_add_px": int(np.count_nonzero(manual_add)) if manual_add is not None else 0,
            "manual_surface_remove_px": int(np.count_nonzero(manual_remove)) if manual_remove is not None else 0,
            "manual_surface_override_applied": bool(
                (manual_add is not None and np.any(manual_add))
                or (manual_remove is not None and np.any(manual_remove))
            ),
        })
        reconstructed_path = edited_dir / f"{stem}_reconstructed_road_surface.png"
        added_path = edited_dir / f"{stem}_surface_added.png"
        removed_path = edited_dir / f"{stem}_surface_removed.png"
        uncertain_path = edited_dir / f"{stem}_surface_uncertain.png"
        cv2.imwrite(str(reconstructed_path), reconstruction.surface * 255)
        cv2.imwrite(str(added_path), reconstruction.added * 255)
        cv2.imwrite(str(removed_path), reconstruction.removed * 255)
        cv2.imwrite(str(uncertain_path), reconstruction.uncertain * 255)
        image = _read_image(Path(summary["image"]), cv2.IMREAD_COLOR)
        viz_path = edited_dir / f"{stem}_surface_reconstruction_viz.png"
        if image is not None:
            overlay = image.copy()
            overlay[reconstruction.surface > 0] = (40, 190, 40)
            overlay[reconstruction.added > 0] = (255, 120, 0)
            overlay[reconstruction.removed > 0] = (0, 0, 255)
            overlay[reconstruction.uncertain > 0] = (0, 220, 255)
            viz = cv2.addWeighted(overlay, 0.38, image, 0.62, 0)
            cv2.imwrite(str(viz_path), viz)
        report_path = edited_dir / f"{stem}_surface_reconstruction.json"
        report_path.write_text(json.dumps(reconstruction.metadata, indent=2), encoding="utf-8")
        for key in reconstruction_totals:
            reconstruction_totals[key] += int(reconstruction.metadata[key])
        tile.update({
            "graph": str(graph_path), "road_surface": str(reconstructed_path),
            "reconstructed_road_surface": str(reconstructed_path),
            "surface_added": str(added_path), "surface_removed": str(removed_path),
            "surface_uncertain": str(uncertain_path), "surface_reconstruction_viz": str(viz_path),
            "surface_reconstruction_report": str(report_path), "line_count": len(edges),
        })
    manifest.update({
        "stitched_network": str(output), "global_source_count": len(geometries),
        "stitched_tile_edge_count": total_edges, "stitch_source": "edited_directory",
        "surface_reconstruction": reconstruction_totals,
        "partial_surface_rebuild": only_stems is not None,
        "requested_surface_rebuild_stems": sorted(only_stems or set()),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Production stages for SAMRoad road editing, QA and vector export.")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("review-status")
    status.add_argument("--review-dir", required=True)
    status.add_argument("--require-complete", action="store_true")
    export_edit = sub.add_parser("export-edit")
    export_edit.add_argument("--review-dir", required=True)
    export_edit.add_argument("--output", required=True)
    import_edit = sub.add_parser("import-edit")
    import_edit.add_argument("--gpkg", required=True)
    import_edit.add_argument("--review-dir", required=True)
    import_edit.add_argument("--edited-dir", required=True)
    import_edit.add_argument("--snap-tolerance-px", type=float, default=2.0)
    export_final = sub.add_parser("export-final")
    export_final.add_argument("--final-dir", required=True)
    export_final.add_argument("--image-dir", required=True)
    export_final.add_argument("--output", default="", help="Optional GeoPackage output.")
    export_final.add_argument("--shp-dir", default="")
    export_final.add_argument("--centerline-shp", default="")
    export_final.add_argument("--surface-shp", default="")
    export_final.add_argument("--visualization", default="")
    export_final.add_argument("--stitched-centerlines", default="", help="Optional authoritative global centerline network from step 3.")
    stitch = sub.add_parser("stitch")
    stitch.add_argument("--input-gpkg", required=True)
    stitch.add_argument("--output", required=True)
    stitch.add_argument("--snap-tolerance", type=float, default=1.5)
    stitch_edit = sub.add_parser("stitch-edit")
    stitch_edit.add_argument("--input-gpkg", required=True)
    stitch_edit.add_argument("--review-dir", required=True)
    stitch_edit.add_argument("--edited-dir", required=True)
    stitch_edit.add_argument("--output", required=True)
    stitch_edit.add_argument("--snap-tolerance", type=float, default=1.5)
    stitch_edited = sub.add_parser("stitch-edited")
    stitch_edited.add_argument("--review-dir", required=True)
    stitch_edited.add_argument("--edited-dir", required=True)
    stitch_edited.add_argument("--output", required=True)
    stitch_edited.add_argument("--snap-tolerance", type=float, default=1.5)
    stitch_edited.add_argument("--surface-low-probability", type=float, default=0.30)
    stitch_edited.add_argument("--surface-high-probability", type=float, default=0.55)
    stitch_edited.add_argument("--surface-max-corridor-px", type=float, default=60.0)
    stitch_edited.add_argument(
        "--only-stem", action="append", default=[],
        help="只为指定切片重建道路面；全局中心线仍会统一拼接并回写所有切片。",
    )
    args = parser.parse_args()
    if args.command == "review-status":
        result = review_status(Path(args.review_dir))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2 if args.require_complete and not result["complete"] else 0
    if args.command == "export-edit": result = export_edit_package(Path(args.review_dir), Path(args.output))
    elif args.command == "import-edit": result = import_edit_package(Path(args.gpkg), Path(args.review_dir), Path(args.edited_dir), args.snap_tolerance_px)
    elif args.command == "export-final": result = export_final_products(
        Path(args.final_dir), Path(args.image_dir), Path(args.output) if args.output else None,
        Path(args.shp_dir) if args.shp_dir else None,
        Path(args.visualization) if args.visualization else None,
        Path(args.centerline_shp) if args.centerline_shp else None,
        Path(args.surface_shp) if args.surface_shp else None,
        Path(args.stitched_centerlines) if args.stitched_centerlines else None,
    )
    elif args.command == "stitch-edit": result = stitch_edit_package(Path(args.input_gpkg), Path(args.review_dir), Path(args.edited_dir), Path(args.output), args.snap_tolerance)
    elif args.command == "stitch-edited": result = stitch_edited_directory(
        Path(args.review_dir), Path(args.edited_dir), Path(args.output), args.snap_tolerance,
        args.surface_low_probability, args.surface_high_probability, args.surface_max_corridor_px,
        set(args.only_stem) if args.only_stem else None,
    )
    else: result = stitch_final_products(Path(args.input_gpkg), Path(args.output), args.snap_tolerance)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
