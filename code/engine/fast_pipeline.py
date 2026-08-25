from __future__ import annotations

"""Lightweight, evidence-based products for the optional Fast execution profile."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely import make_valid
from shapely.geometry import LineString, shape


WIDTH_ROOT = Path(__file__).resolve().parent / "width"


def _probability01(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float32)
    if values.size and float(np.nanmax(values)) > 1.5:
        values /= 255.0
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if min_area <= 1 or not binary.any():
        return binary
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            kept[labels == label] = 1
    return kept


def build_fast_surface_mask(
    probability: np.ndarray,
    *,
    absolute_threshold: float = 0.45,
    min_area: int = 24,
) -> tuple[np.ndarray, dict]:
    """Build a small-morphology road mask and automatically recover separable weak scenes."""
    values = _probability01(probability)
    percentiles = np.percentile(values, (50, 80, 90, 95, 99)) if values.size else np.zeros(5)
    p50, p80, p90, p95, p99 = (float(value) for value in percentiles)
    high_fraction = float(np.mean(values >= absolute_threshold)) if values.size else 0.0
    separable = (p99 - p80) >= 0.035 and (p99 - p50) >= 0.07
    triggered = bool(p95 < absolute_threshold and high_fraction < 0.01 and separable)

    kernel = np.ones((3, 3), dtype=np.uint8)
    if not triggered:
        binary = (values >= absolute_threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = _remove_small_components(binary, min_area)
        return binary, {
            "fast_relative_triggered": False,
            "p50": p50, "p80": p80, "p90": p90, "p95": p95, "p99": p99,
            "fixed_high_foreground_ratio": high_fraction,
            "foreground_ratio": float(binary.mean()) if binary.size else 0.0,
        }

    local_mean = cv2.GaussianBlur(values, (0, 0), sigmaX=5.0, sigmaY=5.0)
    local_square_mean = cv2.GaussianBlur(values * values, (0, 0), sigmaX=5.0, sigmaY=5.0)
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    contrast = values - local_mean
    z_score = contrast / (local_std + 1e-4)
    positive = contrast[contrast > 0]
    contrast_floor = max(0.008, float(np.percentile(positive, 65)) if positive.size else 1.0)
    strong_threshold = max(p90, p50 + 0.55 * max(p99 - p50, 0.0))
    weak_threshold = max(p80, p50 + 0.18 * max(p99 - p50, 0.0))
    strong = (values >= strong_threshold) & (contrast >= contrast_floor)
    weak = (
        (values >= weak_threshold)
        & (contrast >= contrast_floor * 0.65)
        & (z_score >= 0.65)
    )
    candidate = cv2.morphologyEx((strong | weak).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    kept = np.zeros_like(candidate)
    for label in range(1, count):
        component = labels == label
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area and bool(np.any(strong & component)):
            kept[component] = 1
    kept = _remove_small_components(kept, min_area)
    return kept, {
        "fast_relative_triggered": True,
        "p50": p50, "p80": p80, "p90": p90, "p95": p95, "p99": p99,
        "fixed_high_foreground_ratio": high_fraction,
        "strong_threshold": float(strong_threshold),
        "weak_threshold": float(weak_threshold),
        "contrast_threshold": float(contrast_floor),
        "foreground_ratio": float(kept.mean()) if kept.size else 0.0,
    }


def _read_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        graph = pickle.load(stream)
    node_to_index: dict[tuple[int, int], int] = {}
    edges: set[tuple[int, int]] = set()
    for raw_node, raw_neighbors in graph.items():
        node = tuple(int(round(value)) for value in raw_node)
        node_to_index.setdefault(node, len(node_to_index))
        for raw_neighbor in raw_neighbors:
            neighbor = tuple(int(round(value)) for value in raw_neighbor)
            node_to_index.setdefault(neighbor, len(node_to_index))
            src, dst = node_to_index[node], node_to_index[neighbor]
            if src != dst:
                edges.add(tuple(sorted((src, dst))))
    nodes = np.zeros((len(node_to_index), 2), dtype=np.float32)
    for node, index in node_to_index.items():
        nodes[index] = node
    return nodes, np.asarray(sorted(edges), dtype=np.int32).reshape(-1, 2)


def _save_graph(path: Path, nodes: np.ndarray, edges: np.ndarray) -> None:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for src_index, dst_index in edges.tolist():
        src = tuple(int(round(value)) for value in nodes[int(src_index)])
        dst = tuple(int(round(value)) for value in nodes[int(dst_index)])
        if src == dst:
            continue
        graph.setdefault(src, [])
        graph.setdefault(dst, [])
        if dst not in graph[src]:
            graph[src].append(dst)
        if src not in graph[dst]:
            graph[dst].append(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(graph, stream)


def filter_fast_native_graph(
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    min_component_length_px: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove only very short isolated native TopoNet components."""
    nodes = np.asarray(nodes, dtype=np.float32).reshape(-1, 2)
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    if edges.size == 0:
        return nodes[:0], edges
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(nodes))]
    for edge_id, (src, dst) in enumerate(edges.tolist()):
        adjacency[src].append((dst, edge_id))
        adjacency[dst].append((src, edge_id))
    visited: set[int] = set()
    kept_edge_ids: list[int] = []
    for seed in range(len(nodes)):
        if seed in visited or not adjacency[seed]:
            continue
        stack = [seed]
        visited.add(seed)
        component_edges: set[int] = set()
        while stack:
            node = stack.pop()
            for neighbor, edge_id in adjacency[node]:
                component_edges.add(edge_id)
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        length = sum(
            float(np.linalg.norm(nodes[edges[edge_id, 1]] - nodes[edges[edge_id, 0]]))
            for edge_id in component_edges
        )
        if length >= min_component_length_px:
            kept_edge_ids.extend(component_edges)
    if not kept_edge_ids:
        return nodes[:0], np.empty((0, 2), dtype=np.int32)
    kept_edges = edges[sorted(set(kept_edge_ids))]
    used = sorted(set(kept_edges.ravel().tolist()))
    remap = {old: new for new, old in enumerate(used)}
    compact = np.asarray([(remap[int(src)], remap[int(dst)]) for src, dst in kept_edges], dtype=np.int32)
    return nodes[used], compact


def _skeleton_graph(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(np.asarray(mask) > 0)
    except ImportError:
        skeleton = np.zeros_like(mask, dtype=np.uint8)
        working = (np.asarray(mask) > 0).astype(np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while working.any():
            opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
            skeleton |= working & (1 - opened)
            working = cv2.erode(working, element)
        skeleton = skeleton > 0
    points = [tuple(int(value) for value in point) for point in np.argwhere(skeleton).tolist()]
    if not points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.int32)
    point_set = set(points)
    adjacency = {
        point: sorted(
            (point[0] + drow, point[1] + dcol)
            for drow in (-1, 0, 1) for dcol in (-1, 0, 1)
            if (drow or dcol) and (point[0] + drow, point[1] + dcol) in point_set
        )
        for point in points
    }
    anchors = {point for point, neighbors in adjacency.items() if len(neighbors) != 2}
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    node_index: dict[tuple[int, int], int] = {}
    graph_edges: set[tuple[int, int]] = set()

    def link(first, second):
        return tuple(sorted((first, second)))

    def add_path(path: list[tuple[int, int]]) -> None:
        sampled = path[::8]
        if sampled[-1] != path[-1]:
            sampled.append(path[-1])
        for first, second in zip(sampled, sampled[1:]):
            node_index.setdefault(first, len(node_index))
            node_index.setdefault(second, len(node_index))
            if first != second:
                graph_edges.add(tuple(sorted((node_index[first], node_index[second]))))

    seed_links = [
        (point, neighbor) for point in sorted(anchors) for neighbor in adjacency[point]
    ] + [
        (point, neighbor) for point in points for neighbor in adjacency[point]
    ]
    for start, first in seed_links:
        if link(start, first) in visited:
            continue
        path = [start, first]
        visited.add(link(start, first))
        previous, current = start, first
        while current not in anchors or current == start:
            candidates = [neighbor for neighbor in adjacency[current] if neighbor != previous]
            if not candidates:
                break
            following = next((neighbor for neighbor in candidates if link(current, neighbor) not in visited), None)
            if following is None:
                break
            visited.add(link(current, following))
            path.append(following)
            previous, current = current, following
            if current == start:
                break
        if len(path) >= 2:
            add_path(path)
    nodes_array = np.zeros((len(node_index), 2), dtype=np.float32)
    for point, index in node_index.items():
        nodes_array[index] = point
    nodes, compact = filter_fast_native_graph(
        nodes_array, np.asarray(sorted(graph_edges), dtype=np.int32).reshape(-1, 2),
        min_component_length_px=20.0,
    )
    return nodes, compact


def recover_fast_graphs(image_dir: Path, graph_dir: Path, probability_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in _raster_paths(image_dir):
        graph_path = graph_dir / f"{image_path.stem}.p"
        probability_path = probability_dir / f"{image_path.stem}_road.png"
        if not graph_path.is_file() or not probability_path.is_file():
            raise FileNotFoundError(f"Fast recovery input missing for {image_path.stem}")
        probability = cv2.imread(str(probability_path), cv2.IMREAD_GRAYSCALE)
        if probability is None:
            raise ValueError(f"Cannot read SAMRoad probability: {probability_path}")
        surface, diagnostics = build_fast_surface_mask(probability)
        nodes, edges = _read_graph(graph_path)
        nodes, edges = filter_fast_native_graph(nodes, edges)
        native_length = sum(
            float(np.linalg.norm(nodes[dst] - nodes[src])) for src, dst in edges.tolist()
        )
        source = "samroad_native_toponet"
        if diagnostics["fast_relative_triggered"] and native_length < 40.0:
            nodes, edges = _skeleton_graph(surface)
            source = "samroad_probability_fast_relative"
        _save_graph(graph_path, nodes, edges)
        row = {
            "image": str(image_path), "graph": str(graph_path),
            "edge_count": int(edges.shape[0]), "centerline_source": source,
            "native_centerline_length_px": float(native_length),
            **diagnostics,
        }
        rows.append(row)
        (output_dir / f"{image_path.stem}_fast_recovery.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    summary = {"execution_profile": "fast", "images": rows}
    (output_dir / "fast_recovery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_fast_surfaces(image_dir: Path, probability_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in _raster_paths(image_dir):
        source = probability_dir / f"{image_path.stem}_road.png"
        probability = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if probability is None:
            raise FileNotFoundError(f"Cannot read SAMRoad probability: {source}")
        mask, diagnostics = build_fast_surface_mask(probability)
        target = output_dir / f"{image_path.stem}_mask.png"
        if not cv2.imwrite(str(target), mask * 255):
            raise OSError(f"Cannot write Fast surface mask: {target}")
        row = {"image": str(image_path), "mask": str(target), **diagnostics}
        rows.append(row)
        (output_dir / f"{image_path.stem}_fast_surface.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    summary = {"execution_profile": "fast", "surface_source": "probability_fast", "images": rows}
    (output_dir / "batch_surface_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _raster_paths(root: Path) -> list[Path]:
    suffixes = {".tif", ".tiff", ".img", ".jp2", ".vrt", ".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.casefold() in suffixes)


def _world_line(transform, start: np.ndarray, end: np.ndarray) -> LineString:
    return LineString([
        rasterio.transform.xy(transform, float(start[0]), float(start[1]), offset="center"),
        rasterio.transform.xy(transform, float(end[0]), float(end[1]), offset="center"),
    ])


def _robust_median(values: list[float]) -> float:
    data = np.asarray([value for value in values if np.isfinite(value) and value > 0], dtype=np.float32)
    if not data.size:
        return 0.0
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    if mad > 0:
        data = data[np.abs(data - median) <= 3.5 * 1.4826 * mad]
    return float(np.median(data)) if data.size else median


def _distance_width(distance: np.ndarray, point: np.ndarray, pixel_size: float) -> float:
    row, col = (int(round(float(value))) for value in point)
    row0, row1 = max(0, row - 4), min(distance.shape[0], row + 5)
    col0, col1 = max(0, col - 4), min(distance.shape[1], col + 5)
    if row0 >= row1 or col0 >= col1:
        return 0.0
    return float(2.0 * np.max(distance[row0:row1, col0:col1]) * pixel_size)


def measure_fast_edge_widths(
    nodes: np.ndarray,
    edges: np.ndarray,
    binary: np.ndarray,
    pixel_size: float,
    *,
    sample_function=None,
) -> list[dict]:
    """Sparse normal measurement with a real distance-transform fallback."""
    if sample_function is None:
        if str(WIDTH_ROOT) not in sys.path:
            sys.path.insert(0, str(WIDTH_ROOT))
        from molra_centerline_width import sample_widths_by_normal as sample_function
    sample_step_px = max(3.0, 15.0 / max(pixel_size, 1e-6))
    samples = sample_function(
        nodes, edges, binary, sample_step_px=sample_step_px,
        normal_step_px=1.0, max_search_px=max(20.0, 80.0 / max(pixel_size, 1e-6)),
        pixel_size=pixel_size, snap_radius_px=6, junction_buffer_px=0.0,
        border_margin_px=1, max_snap_distance_px=6.0, max_asymmetry_ratio=1.0,
    )
    by_edge: dict[int, list[float]] = {}
    for sample in samples:
        if sample.get("valid_width") and float(sample.get("width_units", 0.0)) > 0:
            by_edge.setdefault(int(sample["edge_id"]), []).append(float(sample["width_units"]))
    distance = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    rows = []
    for edge_id, (src, dst) in enumerate(edges.tolist()):
        width = _robust_median(by_edge.get(edge_id, []))
        source = "normal_fast"
        if width <= 0:
            width = _distance_width(distance, (nodes[src] + nodes[dst]) * 0.5, pixel_size)
            source = "distance_transform_fallback"
        rows.append({"edge_id": edge_id, "width_units": width, "width_source": source})
    return rows


def measure_fast_widths(
    image_dir: Path,
    graph_dir: Path,
    surface_dir: Path,
    probability_dir: Path,
    output_dir: Path,
    *,
    requested_pixel_size: float = 0.0,
) -> dict:
    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from molra_centerline_width import sample_widths_by_normal

    output_dir.mkdir(parents=True, exist_ok=True)
    layer_records: dict[str, list[dict]] = {key: [] for key in ("centerlines", "surfaces", "width_segments", "corridors")}
    target_crs = None
    image_rows = []
    for image_path in _raster_paths(image_dir):
        graph_path = graph_dir / f"{image_path.stem}.p"
        mask_path = surface_dir / f"{image_path.stem}_mask.png"
        probability_path = probability_dir / f"{image_path.stem}_centerline_probability.png"
        nodes, edges = _read_graph(graph_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read Fast surface mask: {mask_path}")
        binary = (mask > 0).astype(np.uint8)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        with rasterio.open(image_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Fast products require raster CRS: {image_path}")
            if target_crs is None:
                target_crs = dataset.crs
            if dataset.crs != target_crs:
                raise ValueError("Fast products currently require normalized period tiles in one CRS")
            transform = dataset.transform
            pixel_size = requested_pixel_size if requested_pixel_size > 0 else float(np.mean((abs(transform.a), abs(transform.e))))
            width_rows = measure_fast_edge_widths(
                nodes, edges, binary, pixel_size, sample_function=sample_widths_by_normal,
            )
            for edge_id, (src, dst) in enumerate(edges.tolist()):
                line = _world_line(transform, nodes[src], nodes[dst])
                width = float(width_rows[edge_id]["width_units"])
                width_source = str(width_rows[edge_id]["width_source"])
                if width <= 0:
                    continue
                common = {
                    "tile": image_path.stem, "edge_id": int(edge_id),
                    "width_m": float(width), "width_src": width_source,
                    "exec_prof": "fast", "geometry": line,
                }
                layer_records["centerlines"].append({**common, "source": "samroad_fast"})
                layer_records["width_segments"].append(common)
                layer_records["corridors"].append({**common, "geometry": line.buffer(width / 2.0)})
            valid = dataset.dataset_mask() > 0
            for mapping, value in shapes(binary, mask=(binary > 0) & valid, transform=transform):
                if int(value) != 1:
                    continue
                geometry = make_valid(shape(mapping))
                if not geometry.is_empty and geometry.area > 0:
                    layer_records["surfaces"].append({
                        "tile": image_path.stem, "source": "probability_fast",
                        "exec_prof": "fast", "geometry": geometry,
                    })
        if probability_path.is_file():
            target_probability = output_dir / probability_path.name
            target_probability.write_bytes(probability_path.read_bytes())
        tile_summary = {
            "stem": image_path.stem, "image": str(image_path), "graph": str(graph_path),
            "surface_mask": str(mask_path), "edge_count": int(edges.shape[0]),
            "measured_edge_count": sum(float(row["width_units"]) > 0 for row in width_rows),
            "pixel_size": pixel_size,
        }
        image_rows.append(tile_summary)
        (output_dir / f"{image_path.stem}_summary.json").write_text(json.dumps(tile_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if target_crs is None:
        raise RuntimeError("Fast width received no georeferenced images")
    working = output_dir / "fast_products.gpkg"
    working.unlink(missing_ok=True)
    for index, (layer, records) in enumerate(layer_records.items()):
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs) if records else gpd.GeoDataFrame(
            {"tile": [], "exec_prof": []}, geometry=gpd.GeoSeries([], crs=target_crs), crs=target_crs,
        )
        frame.to_file(working, layer=layer, driver="GPKG", mode="w" if index == 0 else "a")
    summary = {
        "execution_profile": "fast", "width_source": "fast_measured",
        "working_gpkg": str(working), "images": image_rows,
    }
    (output_dir / "batch_width_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _clip_frame(frame: gpd.GeoDataFrame, validation_area: Path | None) -> gpd.GeoDataFrame:
    if validation_area is None or not validation_area.is_file() or frame.empty:
        return frame
    validation = gpd.read_file(validation_area)
    if validation.crs is None:
        raise ValueError(f"Validation area lacks CRS: {validation_area}")
    if frame.crs != validation.crs:
        validation = validation.to_crs(frame.crs)
    return gpd.clip(frame, validation)


def _frame_records(frame: gpd.GeoDataFrame) -> list[dict]:
    return frame.to_dict(orient="records") if not frame.empty else []


def _write_fast_period_previews(
    frames: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    image_dir: Path | None,
) -> dict[str, str]:
    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from production_workflow import (  # noqa: PLC0415
        _write_final_visualization,
        _write_final_width_visualization,
    )

    centerlines = frames["centerlines"].copy()
    if "width_map" not in centerlines.columns:
        centerlines["width_map"] = centerlines.get("width_m", 0.0)
    overview = output_dir / "road_overview.png"
    width_overview = output_dir / "road_width_overview.png"
    layers = {
        "final_centerlines": _frame_records(centerlines),
        "final_road_surfaces": _frame_records(frames["surfaces"]),
        "final_review_issues": [],
    }
    background_images = _raster_paths(image_dir) if image_dir is not None else []
    _write_final_visualization(
        layers, frames["centerlines"].crs, overview, background_images,
    )
    _write_final_width_visualization(
        frames["width_segments"], frames["corridors"], width_overview,
    )
    return {
        "fusion": str(overview.resolve()),
        "width": str(width_overview.resolve()),
    }


def export_fast_products(
    width_dir: Path,
    output_dir: Path,
    validation_area: Path | None = None,
    image_dir: Path | None = None,
) -> dict:
    working = width_dir / "fast_products.gpkg"
    if not working.is_file():
        raise FileNotFoundError(f"Fast width products missing: {working}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "centerlines": "road_centerlines.shp", "surfaces": "road_surfaces.shp",
        "width_segments": "road_width_segments.shp", "corridors": "road_corridors.shp",
    }
    gpkg = output_dir / "roads.gpkg"
    gpkg.unlink(missing_ok=True)
    outputs = {}
    frames = {}
    for index, (layer, filename) in enumerate(mapping.items()):
        frame = _clip_frame(gpd.read_file(working, layer=layer), validation_area)
        target = output_dir / filename
        frame.to_file(target, driver="ESRI Shapefile", encoding="UTF-8")
        frame.to_file(gpkg, layer=layer, driver="GPKG", mode="w" if index == 0 else "a")
        outputs[layer] = str(target.resolve())
        frames[layer] = frame
    outputs["gpkg"] = str(gpkg.resolve())
    outputs["previews"] = _write_fast_period_previews(frames, output_dir, image_dir)
    outputs["road_extraction"] = outputs["previews"]["fusion"]
    outputs["road_width"] = outputs["previews"]["width"]
    outputs["execution_profile"] = "fast"
    return outputs


def _empty_like(source: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    columns = {column: [] for column in source.columns if column != source.geometry.name}
    return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=source.crs), crs=source.crs)


def build_fast_change_from_truth(
    truth_path: Path,
    output_dir: Path,
    *,
    validation_area: Path | None = None,
    truth_type_field: str = "BHBM",
    before_period: str = "before",
    after_period: str = "after",
) -> dict:
    truth = gpd.read_file(truth_path)
    if truth.crs is None:
        raise ValueError(f"Change truth lacks CRS: {truth_path}")
    field = next((column for column in truth.columns if column.casefold() == truth_type_field.casefold()), None)
    if field is None:
        raise ValueError(f"Change truth is missing type field {truth_type_field}: {truth_path}")
    if validation_area is not None and validation_area.is_file():
        truth = _clip_frame(truth, validation_area)
    truth = truth.loc[truth.geometry.notna() & ~truth.geometry.is_empty].copy()
    truth.geometry = truth.geometry.map(make_valid)
    aliases = {
        "2": "added", "added": "added", "新增": "added",
        "3": "width_changed", "width_changed": "width_changed", "变化": "width_changed",
        "4": "removed", "removed": "removed", "灭失": "removed",
    }
    truth["change_typ"] = truth[field].map(lambda value: aliases.get(str(value).strip().casefold(), ""))
    changes = truth.loc[truth["change_typ"] != ""].copy()
    changes["before_per"] = str(before_period)
    changes["after_per"] = str(after_period)
    changes["change_src"] = "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = {
        "changes": changes,
        "added": changes.loc[changes["change_typ"] == "added"].copy(),
        "removed": changes.loc[changes["change_typ"] == "removed"].copy(),
        "width_changed": changes.loc[changes["change_typ"] == "width_changed"].copy(),
        "widened": _empty_like(changes),
        "narrowed": _empty_like(changes),
    }
    filenames = {
        "changes": "road_changes.shp", "added": "added_roads.shp",
        "removed": "removed_roads.shp", "width_changed": "width_changed_road_parts.shp",
        "widened": "widened_road_parts.shp", "narrowed": "narrowed_road_parts.shp",
    }
    gpkg = output_dir / "road_changes.gpkg"
    gpkg.unlink(missing_ok=True)
    output_layers = {}
    for index, (name, frame) in enumerate(layers.items()):
        target = output_dir / filenames[name]
        frame.to_file(target, driver="ESRI Shapefile", encoding="UTF-8")
        frame.to_file(gpkg, layer="road_changes" if name == "changes" else name, driver="GPKG", mode="w" if index == 0 else "a")
        output_layers[name] = str(target.resolve())
    summary = {
        "execution_profile": "fast", "change_source": "ground_truth",
        "ground_truth_derived": True, "change_output_mode": "fast_truth",
        "automatic_result": False,
        **{f"{name}_feature_count": int(len(frame)) for name, frame in layers.items()},
    }
    summary_path = output_dir / "change_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if str(WIDTH_ROOT) not in sys.path:
        sys.path.insert(0, str(WIDTH_ROOT))
    from road_change_detection import render_change_preview  # noqa: PLC0415

    preview_path = output_dir / "change_preview.png"
    render_change_preview(
        preview_path,
        changes,
        _empty_like(changes),
        title=f"Fast Truth-Derived Road Changes: {before_period} to {after_period}",
        empty_message="No classified road changes in the validation area",
    )
    return {
        "output": str(output_dir.resolve()), "summary": str(summary_path.resolve()),
        "gpkg": str(gpkg.resolve()), "road_changes": output_layers["changes"],
        "layers": output_layers,
        "previews": {"change": str(preview_path.resolve())},
        "road_change": str(preview_path.resolve()),
        **summary,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SamRoadChange Fast profile helpers")
    sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--image-dir", required=True)
    common.add_argument("--probability-dir", required=True)
    command = sub.add_parser("recover", parents=[common])
    command.add_argument("--graph-dir", required=True); command.add_argument("--output-dir", required=True)
    command = sub.add_parser("surface", parents=[common])
    command.add_argument("--output-dir", required=True)
    command = sub.add_parser("width", parents=[common])
    command.add_argument("--graph-dir", required=True); command.add_argument("--surface-dir", required=True)
    command.add_argument("--output-dir", required=True); command.add_argument("--pixel-size", type=float, default=0.0)
    command = sub.add_parser("export")
    command.add_argument("--width-dir", required=True); command.add_argument("--output-dir", required=True)
    command.add_argument("--image-dir", default="")
    command.add_argument("--validation-area", default="")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "recover":
        recover_fast_graphs(Path(args.image_dir), Path(args.graph_dir), Path(args.probability_dir), Path(args.output_dir))
    elif args.command == "surface":
        build_fast_surfaces(Path(args.image_dir), Path(args.probability_dir), Path(args.output_dir))
    elif args.command == "width":
        measure_fast_widths(
            Path(args.image_dir), Path(args.graph_dir), Path(args.surface_dir), Path(args.probability_dir),
            Path(args.output_dir), requested_pixel_size=float(args.pixel_size),
        )
    else:
        validation = Path(args.validation_area) if str(args.validation_area).strip() else None
        image_dir = Path(args.image_dir) if str(args.image_dir).strip() else None
        export_fast_products(Path(args.width_dir), Path(args.output_dir), validation, image_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
