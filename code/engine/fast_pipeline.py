from __future__ import annotations

"""Lightweight, evidence-based products for the optional Fast execution profile."""

import argparse
import json
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
FAST_LOCAL_STD_FLOOR = 1.0 / (255.0 * np.sqrt(12.0))
FAST_SCALE_SUPPORT_THRESHOLD = 0.50


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


def _consistent_relative_score(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Accept the stronger scale only when the other scale has fixed positive support."""
    stronger = np.maximum(first, second)
    weaker = np.minimum(first, second)
    return np.where(
        weaker >= FAST_SCALE_SUPPORT_THRESHOLD, stronger, 0.0,
    ).astype(np.float32)


def _fast_relative_score(values: np.ndarray) -> np.ndarray:
    """Return a fixed-floor, scale-consistent local Relative score."""
    scores = []
    for sigma in (3.0, 15.0):
        local_mean = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma)
        local_square_mean = cv2.GaussianBlur(
            values * values, (0, 0), sigmaX=sigma, sigmaY=sigma,
        )
        local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
        effective_std = np.maximum(local_std, FAST_LOCAL_STD_FLOOR)
        scores.append(np.maximum(values - local_mean, 0.0) / effective_std)
    return _consistent_relative_score(scores[0], scores[1])


def _relative_hysteresis_mask(relative_score: np.ndarray, min_area: int) -> np.ndarray:
    """Keep medium Relative evidence only inside components containing Strong evidence."""
    strong = np.asarray(relative_score) >= 1.30
    weak = (np.asarray(relative_score) >= 0.90).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(weak, connectivity=8)
    kept = np.zeros_like(weak)
    for label in range(1, count):
        component = labels == label
        if (
            int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area)
            and bool(np.any(strong & component))
        ):
            kept[component] = 1
    return kept


def build_fast_surface_mask(
    probability: np.ndarray,
    *,
    absolute_threshold: float = 0.45,
    min_area: int = 24,
) -> tuple[np.ndarray, dict]:
    """Build the single Fast road mask from absolute and local-relative evidence."""
    values = _probability01(probability)
    high = values >= float(absolute_threshold)
    relative_score = _fast_relative_score(values)
    relative = _relative_hysteresis_mask(relative_score, min_area)
    relative_added = relative & ~high
    kernel = np.ones((3, 3), dtype=np.uint8)
    final_mask = cv2.morphologyEx((high | relative).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    final_mask = _remove_small_components(final_mask, min_area)
    return final_mask, {
        "raw_high_probability_pixel_count": int(np.count_nonzero(high)),
        "relative_added_pixel_count": int(np.count_nonzero(relative_added)),
        "final_mask_pixel_count": int(np.count_nonzero(final_mask)),
    }


def _prune_short_skeleton_spurs(skeleton: np.ndarray, max_length_px: int = 6) -> np.ndarray:
    """Delete short endpoint-to-junction branches from a one-pixel skeleton."""
    working = np.asarray(skeleton, dtype=bool).copy()
    for _pass in range(max(1, int(max_length_px))):
        points = {tuple(int(value) for value in point) for point in np.argwhere(working)}
        adjacency = {
            point: [
                (point[0] + drow, point[1] + dcol)
                for drow in (-1, 0, 1) for dcol in (-1, 0, 1)
                if (drow or dcol) and (point[0] + drow, point[1] + dcol) in points
            ]
            for point in points
        }
        removed: set[tuple[int, int]] = set()
        for endpoint in sorted(point for point, neighbors in adjacency.items() if len(neighbors) == 1):
            if endpoint in removed:
                continue
            path = [endpoint]
            previous = None
            current = endpoint
            while len(path) <= int(max_length_px) + 1:
                candidates = [point for point in adjacency.get(current, []) if point != previous]
                if not candidates:
                    break
                following = candidates[0]
                path.append(following)
                previous, current = current, following
                if len(adjacency.get(current, [])) != 2:
                    break
            if len(adjacency.get(current, [])) >= 3 and len(path) - 1 <= int(max_length_px):
                removed.update(path[:-1])
        if not removed:
            break
        rows, cols = zip(*removed)
        working[np.asarray(rows), np.asarray(cols)] = False
    return working.astype(np.uint8)


def _remove_short_isolated_skeleton_components(
    skeleton: np.ndarray, min_length_px: float = 20.0,
) -> np.ndarray:
    """Remove isolated skeleton components with short total centerline length."""
    binary = (np.asarray(skeleton) > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    kept = np.zeros_like(binary)
    forward_neighbors = (
        (0, 1, 1.0),
        (1, -1, float(np.sqrt(2.0))),
        (1, 0, 1.0),
        (1, 1, float(np.sqrt(2.0))),
    )
    for label in range(1, count):
        rows, cols = np.nonzero(labels == label)
        length = 0.0
        for row, col in zip(rows.tolist(), cols.tolist()):
            for drow, dcol, weight in forward_neighbors:
                neighbor_row, neighbor_col = row + drow, col + dcol
                if (
                    0 <= neighbor_row < labels.shape[0]
                    and 0 <= neighbor_col < labels.shape[1]
                    and labels[neighbor_row, neighbor_col] == label
                ):
                    length += weight
        if length >= float(min_length_px):
            kept[labels == label] = 1
    return kept


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
    skeleton = _prune_short_skeleton_spurs(skeleton, max_length_px=6)
    skeleton = _remove_short_isolated_skeleton_components(
        skeleton, min_length_px=20.0,
    ) > 0
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
    return nodes_array, np.asarray(sorted(graph_edges), dtype=np.int32).reshape(-1, 2)


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
        print(
            f"[Fast Mask] {image_path.stem}: "
            f"high={diagnostics['raw_high_probability_pixel_count']}, "
            f"relative_added={diagnostics['relative_added_pixel_count']}, "
            f"final={diagnostics['final_mask_pixel_count']}"
        )
        (output_dir / f"{image_path.stem}_fast_surface.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    summary = {"execution_profile": "fast", "surface_source": "final_fast_mask", "images": rows}
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
        mask_path = surface_dir / f"{image_path.stem}_mask.png"
        probability_path = probability_dir / f"{image_path.stem}_road.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read Fast surface mask: {mask_path}")
        binary = (mask > 0).astype(np.uint8)
        nodes, edges = _skeleton_graph(binary)
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
                layer_records["centerlines"].append({**common, "source": "final_mask_skeleton"})
                layer_records["width_segments"].append(common)
                layer_records["corridors"].append({**common, "geometry": line.buffer(width / 2.0)})
            valid = dataset.dataset_mask() > 0
            for mapping, value in shapes(binary, mask=(binary > 0) & valid, transform=transform):
                if int(value) != 1:
                    continue
                geometry = make_valid(shape(mapping))
                if not geometry.is_empty and geometry.area > 0:
                    layer_records["surfaces"].append({
                        "tile": image_path.stem, "source": "final_fast_mask",
                        "exec_prof": "fast", "geometry": geometry,
                    })
        if probability_path.is_file():
            target_probability = output_dir / f"{image_path.stem}_centerline_probability.png"
            target_probability.write_bytes(probability_path.read_bytes())
        centerline_length_px = float(sum(
            np.linalg.norm(nodes[int(dst)] - nodes[int(src)])
            for src, dst in edges.tolist()
        ))
        surface_diagnostics_path = surface_dir / f"{image_path.stem}_fast_surface.json"
        surface_diagnostics = {}
        if surface_diagnostics_path.is_file():
            try:
                surface_diagnostics = json.loads(surface_diagnostics_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                surface_diagnostics = {}
        tile_summary = {
            "stem": image_path.stem, "image": str(image_path),
            "surface_mask": str(mask_path), "edge_count": int(edges.shape[0]),
            "measured_edge_count": sum(float(row["width_units"]) > 0 for row in width_rows),
            "pixel_size": pixel_size,
            "raw_high_probability_pixel_count": int(surface_diagnostics.get("raw_high_probability_pixel_count", 0)),
            "relative_added_pixel_count": int(surface_diagnostics.get("relative_added_pixel_count", 0)),
            "final_mask_pixel_count": int(surface_diagnostics.get("final_mask_pixel_count", np.count_nonzero(binary))),
            "final_centerline_length": centerline_length_px * float(pixel_size),
            "final_centerline_length_px": centerline_length_px,
        }
        print(
            f"[Fast Centerline] {image_path.stem}: "
            f"length={tile_summary['final_centerline_length']:.3f}"
        )
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
    command = sub.add_parser("surface", parents=[common])
    command.add_argument("--output-dir", required=True)
    command = sub.add_parser("width", parents=[common])
    command.add_argument("--surface-dir", required=True)
    command.add_argument("--output-dir", required=True); command.add_argument("--pixel-size", type=float, default=0.0)
    command = sub.add_parser("export")
    command.add_argument("--width-dir", required=True); command.add_argument("--output-dir", required=True)
    command.add_argument("--image-dir", default="")
    command.add_argument("--validation-area", default="")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "surface":
        build_fast_surfaces(Path(args.image_dir), Path(args.probability_dir), Path(args.output_dir))
    elif args.command == "width":
        measure_fast_widths(
            Path(args.image_dir), Path(args.surface_dir), Path(args.probability_dir),
            Path(args.output_dir), requested_pixel_size=float(args.pixel_size),
        )
    else:
        validation = Path(args.validation_area) if str(args.validation_area).strip() else None
        image_dir = Path(args.image_dir) if str(args.image_dir).strip() else None
        export_fast_products(Path(args.width_dir), Path(args.output_dir), validation, image_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
