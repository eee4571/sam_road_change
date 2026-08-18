from __future__ import annotations

"""Local road-width segmentation and cross-period centreline matching.

The module deliberately has no dependency on the extraction pipeline.  It accepts
ordinary GeoDataFrames, so old jobs that only contain ``road_centerlines.shp`` can
be upgraded at read time while new jobs may also supply the measured width-segment
layer exported by :mod:`production_workflow`.
"""

from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree, make_valid
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring


WIDTH_FIELDS = (
    "width_m", "width_map", "median_width_units", "width_units",
    "final_width", "width",
)
QUALITY_FIELDS = ("width_quality", "width_qual", "quality_gr", "quality_grade")


@dataclass(frozen=True)
class MatchConfig:
    position_tolerance: float = 3.0
    minimum_direction_cosine: float = 0.85
    minimum_candidate_coverage: float = 0.35
    minimum_overlap_length: float = 3.0
    distance_best_slack: float = 0.75
    sample_spacing: float = 2.0


def _line_parts(geometry: BaseGeometry | None) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    geometry = make_valid(geometry)
    if geometry.geom_type == "LineString":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _line_parts(part)


def _number(row: pd.Series | dict, fields: tuple[str, ...], default: float = 0.0) -> float:
    for field in fields:
        try:
            value = float(row.get(field, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _quality(row: pd.Series | dict) -> str:
    for field in QUALITY_FIELDS:
        value = str(row.get(field, "") or "").strip().upper()
        if value in {"A", "B", "C"}:
            return value
    return "C"


def normalize_line_source(row: pd.Series | dict) -> str:
    explicit = str(row.get("line_source", row.get("line_sourc", "")) or "").strip().lower()
    if explicit in {"samroad", "surface_skeleton", "connector", "manual"}:
        return explicit
    source = str(row.get("source", row.get("fusion_sta", "")) or "").strip().lower()
    if "manual" in source or "review_added" in source:
        return "manual"
    if "connector" in source or "gap" in source:
        return "connector"
    if "surface" in source or "skeleton" in source:
        return "surface_skeleton"
    return "samroad"


def _direction(line: LineString) -> np.ndarray:
    coordinates = np.asarray(line.coords, dtype=np.float64)
    if len(coordinates) < 2:
        return np.zeros(2, dtype=np.float64)
    centered = coordinates - coordinates.mean(axis=0)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        vector = vh[0]
    except np.linalg.LinAlgError:
        vector = coordinates[-1] - coordinates[0]
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(2, dtype=np.float64)


def direction_similarity(first: LineString, second: LineString) -> float:
    return abs(float(np.dot(_direction(first), _direction(second))))


def _sample_points(line: LineString, spacing: float) -> list[Point]:
    count = max(3, min(101, int(np.ceil(float(line.length) / max(spacing, 0.25))) + 1))
    return [line.interpolate(index / (count - 1), normalized=True) for index in range(count)]


def _median_distance(first: LineString, second: LineString, spacing: float) -> float:
    distances = [point.distance(second) for point in _sample_points(first, spacing)]
    distances.extend(point.distance(first) for point in _sample_points(second, spacing))
    return float(np.median(np.asarray(distances, dtype=np.float64))) if distances else float("inf")


def _endpoint_score(first: LineString, second: LineString, tolerance: float) -> float:
    first_ends = (Point(first.coords[0]), Point(first.coords[-1]))
    second_ends = (Point(second.coords[0]), Point(second.coords[-1]))
    distances = [min(point.distance(other) for other in second_ends) for point in first_ends]
    distances.extend(min(point.distance(other) for other in first_ends) for point in second_ends)
    return float(np.mean([max(0.0, 1.0 - value / max(3.0 * tolerance, 1e-6)) for value in distances]))


def _node_degrees(frame: gpd.GeoDataFrame, tolerance: float) -> list[tuple[int, int]]:
    endpoints: list[tuple[float, float]] = []
    owners: list[tuple[int, int]] = []
    for index, geometry in enumerate(frame.geometry):
        line = next(_line_parts(geometry), None)
        if line is None:
            owners.append((0, 0))
            continue
        start_index = len(endpoints)
        endpoints.extend([tuple(line.coords[0]), tuple(line.coords[-1])])
        owners.append((start_index, start_index + 1))
    if not endpoints:
        return [(0, 0)] * len(frame)
    points = np.asarray([Point(value) for value in endpoints], dtype=object)
    tree = STRtree(points)
    degrees = []
    snap = max(0.1, min(1.0, tolerance * 0.25))
    for start, end in owners:
        degrees.append((
            len(tree.query(points[start], predicate="dwithin", distance=snap)),
            len(tree.query(points[end], predicate="dwithin", distance=snap)),
        ))
    return degrees


def _robust(values: list[float]) -> tuple[float, float, int]:
    data = np.asarray([value for value in values if np.isfinite(value) and value > 0], dtype=np.float64)
    if data.size == 0:
        return 0.0, 0.0, 0
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    if mad > 0:
        data = data[np.abs(data - median) <= 3.5 * 1.4826 * mad]
    if data.size == 0:
        return median, 0.0, 0
    return float(np.median(data)), float(np.std(data)), int(data.size)


def build_width_segments(
    centerlines: gpd.GeoDataFrame,
    source_width_segments: gpd.GeoDataFrame | None = None,
    target_length: float = 15.0,
    source_tolerance: float = 2.0,
) -> gpd.GeoDataFrame:
    """Split centreline chains and attach robust local width observations.

    Existing endpoints (therefore junction nodes) are always retained. Long
    residual pieces are split to approximately ``target_length`` metres.  When
    measured source segments are supplied their A/B observations are combined by
    median/MAD; rejected, asymmetric, junction and boundary probes never enter the
    robust estimate because those flags have already assigned grade C upstream.
    """
    rows: list[dict] = []
    measured = None
    measured_geometries: np.ndarray | None = None
    measured_tree = None
    if source_width_segments is not None and not source_width_segments.empty:
        measured = source_width_segments
        if measured.crs is not None and centerlines.crs is not None and not measured.crs.equals(centerlines.crs):
            measured = measured.to_crs(centerlines.crs)
        measured = measured.loc[measured.geometry.notna() & ~measured.geometry.is_empty].reset_index(drop=True)
        measured_geometries = np.asarray(measured.geometry.values, dtype=object)
        measured_tree = STRtree(measured_geometries) if len(measured) else None

    for feature_index, source_row in centerlines.reset_index(drop=True).iterrows():
        parent_id = str(source_row.get("parent_id", source_row.get("global_id", source_row.get("road_id", feature_index))))
        source_width = _number(source_row, WIDTH_FIELDS)
        source_grade = _quality(source_row)
        line_source = normalize_line_source(source_row)
        source_valid_ratio = _number(
            source_row, ("valid_ratio", "valid_rati", "direct_measurement_ratio", "direct_mea"),
            1.0 if source_width > 0 and source_grade != "C" else 0.0,
        )
        for part_index, line in enumerate(_line_parts(source_row.geometry)):
            length = float(line.length)
            if length <= 1e-6:
                continue
            count = max(1, int(np.ceil(length / max(10.0, min(20.0, target_length)))))
            for local_index in range(count):
                start = length * local_index / count
                end = length * (local_index + 1) / count
                geometry = substring(line, start, end)
                if geometry.is_empty or float(geometry.length) <= 1e-6:
                    continue
                width_values: list[float] = []
                measured_grades: list[str] = []
                measured_overlap = 0.0
                if measured_tree is not None and measured is not None and measured_geometries is not None:
                    candidate_ids = measured_tree.query(
                        geometry, predicate="dwithin", distance=max(source_tolerance, 0.1),
                    )
                    for candidate_id in candidate_ids:
                        candidate = measured.iloc[int(candidate_id)]
                        candidate_line = next(_line_parts(candidate.geometry), None)
                        if candidate_line is None or direction_similarity(geometry, candidate_line) < 0.8:
                            continue
                        overlap = float(geometry.intersection(candidate_line.buffer(max(source_tolerance, 0.1))).length)
                        if overlap <= 0:
                            continue
                        grade = _quality(candidate)
                        flags = str(candidate.get("quality_flags", "") or "").lower()
                        rejected = any(token in flags for token in ("junction", "border", "boundary", "asym", "outlier", "outside"))
                        value = _number(candidate, WIDTH_FIELDS)
                        if value > 0 and grade != "C" and not rejected:
                            width_values.extend([value] * max(1, int(round(overlap))))
                            measured_grades.append(grade)
                            measured_overlap += min(overlap, float(geometry.length))
                local_width, local_std, sample_count = _robust(width_values)
                valid_ratio = min(1.0, measured_overlap / max(float(geometry.length), 1e-6)) if sample_count else source_valid_ratio
                if sample_count:
                    grade = "A" if valid_ratio >= 0.8 and measured_grades and set(measured_grades) == {"A"} else "B" if valid_ratio >= 0.6 else "C"
                    width = local_width
                    width_std = local_std
                else:
                    grade = source_grade
                    width = source_width
                    width_std = _number(source_row, ("width_std",), 0.0)
                qa_state = str(source_row.get("qa_state", "") or "").lower()
                if qa_state not in {"auto", "review"}:
                    qa_state = "review" if grade == "C" or line_source in {"connector"} else "auto"
                qa_reason = str(source_row.get("qa_reason", "") or "")
                if not qa_reason and qa_state == "review":
                    qa_reason = "low_width_quality" if grade == "C" else "connector_source"
                rows.append({
                    "segment_id": f"S{len(rows) + 1:08d}",
                    "parent_id": parent_id,
                    "part_id": part_index,
                    "length_m": float(geometry.length),
                    "width_m": float(width),
                    "width_std": float(width_std),
                    "valid_ratio": float(np.clip(valid_ratio, 0.0, 1.0)),
                    "width_quality": grade,
                    "line_source": line_source,
                    "quality_grade": grade,
                    "center_conf": _number(source_row, ("center_conf", "center_con", "center_p", "confidence"), 0.0),
                    "surface_conf": _number(source_row, ("surface_conf", "surface_co", "surface_cov", "mean_road_probability"), 0.0),
                    "qa_state": qa_state,
                    "qa_reason": qa_reason,
                    "geometry": geometry,
                })
    columns = {
        "segment_id": pd.Series(dtype="object"), "parent_id": pd.Series(dtype="object"),
        "length_m": pd.Series(dtype="float64"), "width_m": pd.Series(dtype="float64"),
        "width_std": pd.Series(dtype="float64"), "valid_ratio": pd.Series(dtype="float64"),
        "width_quality": pd.Series(dtype="object"), "line_source": pd.Series(dtype="object"),
    }
    return (
        gpd.GeoDataFrame(rows, geometry="geometry", crs=centerlines.crs)
        if rows else gpd.GeoDataFrame(columns, geometry=[], crs=centerlines.crs)
    )


def build_corridors(segments: gpd.GeoDataFrame, period: str = "") -> gpd.GeoDataFrame:
    rows = []
    for _, row in segments.iterrows():
        width = _number(row, ("width_m",))
        if width <= 0:
            continue
        geometry = row.geometry.buffer(width * 0.5, cap_style="flat", join_style="round")
        if geometry.is_empty:
            continue
        rows.append({
            **{key: row.get(key) for key in segments.columns if key != "geometry"},
            "period": period,
            "area_m2": float(geometry.area),
            "geometry": geometry,
        })
    return (
        gpd.GeoDataFrame(rows, geometry="geometry", crs=segments.crs)
        if rows else gpd.GeoDataFrame(
            {"segment_id": pd.Series(dtype="object"), "period": pd.Series(dtype="object"), "area_m2": pd.Series(dtype="float64")},
            geometry=[], crs=segments.crs,
        )
    )


def match_road_segments(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    config: MatchConfig = MatchConfig(),
) -> tuple[gpd.GeoDataFrame, dict[int, list[int]], dict[int, list[int]], dict]:
    """Return validated local pairs and coverage maps.

    ``STRtree`` results are only initial spatial candidates.  Direction,
    bidirectional coverage and distance checks are applied before a candidate is
    allowed into either coverage map.  A local-best distance guard prevents a
    neighbouring parallel carriageway from covering the true unmatched road.
    """
    before_geometries = np.asarray(before.geometry.values, dtype=object)
    after_geometries = np.asarray(after.geometry.values, dtype=object)
    after_tree = STRtree(after_geometries) if len(after_geometries) else None
    before_degrees = _node_degrees(before, config.position_tolerance)
    after_degrees = _node_degrees(after, config.position_tolerance)
    raw: list[dict] = []
    rejected_direction = rejected_overlap = rejected_distance = 0
    if after_tree is not None:
        for before_id, before_line in enumerate(before_geometries):
            candidate_ids = after_tree.query(
                before_line, predicate="dwithin", distance=max(config.position_tolerance, 0.1),
            )
            for after_id_value in candidate_ids:
                after_id = int(after_id_value)
                after_line = after_geometries[after_id]
                direction = direction_similarity(before_line, after_line)
                if direction < config.minimum_direction_cosine:
                    rejected_direction += 1
                    continue
                before_cover = float(before_line.intersection(after_line.buffer(config.position_tolerance)).length) / max(float(before_line.length), 1e-9)
                after_cover = float(after_line.intersection(before_line.buffer(config.position_tolerance)).length) / max(float(after_line.length), 1e-9)
                overlap_length = min(before_cover * float(before_line.length), after_cover * float(after_line.length))
                if max(before_cover, after_cover) < config.minimum_candidate_coverage or overlap_length < config.minimum_overlap_length:
                    rejected_overlap += 1
                    continue
                median_distance = _median_distance(before_line, after_line, config.sample_spacing)
                if median_distance > config.position_tolerance:
                    rejected_distance += 1
                    continue
                length_ratio = min(float(before_line.length), float(after_line.length)) / max(float(before_line.length), float(after_line.length), 1e-9)
                endpoint = _endpoint_score(before_line, after_line, config.position_tolerance)
                topology = 1.0 - min(
                    1.0,
                    (abs(sum(before_degrees[before_id]) - sum(after_degrees[after_id]))) / 4.0,
                )
                distance_score = max(0.0, 1.0 - median_distance / max(config.position_tolerance, 1e-9))
                score = (
                    0.30 * max(before_cover, after_cover)
                    + 0.10 * min(before_cover, after_cover)
                    + 0.25 * distance_score
                    + 0.20 * direction
                    + 0.05 * endpoint
                    + 0.05 * length_ratio
                    + 0.05 * topology
                )
                raw.append({
                    "before_id": before_id, "after_id": after_id,
                    "before_seg": str(before.iloc[before_id].get("segment_id", before_id)),
                    "after_seg": str(after.iloc[after_id].get("segment_id", after_id)),
                    "before_cov": before_cover, "after_cov": after_cover,
                    "median_dist": median_distance, "dir_score": direction,
                    "end_score": endpoint, "length_rat": length_ratio,
                    "topo_score": topology, "match_score": score,
                })

    best_before: dict[int, float] = {}
    best_after: dict[int, float] = {}
    for row in raw:
        best_before[row["before_id"]] = min(best_before.get(row["before_id"], float("inf")), row["median_dist"])
        best_after[row["after_id"]] = min(best_after.get(row["after_id"], float("inf")), row["median_dist"])
    valid = [
        row for row in raw
        if row["median_dist"] <= best_before[row["before_id"]] + config.distance_best_slack
        and row["median_dist"] <= best_after[row["after_id"]] + config.distance_best_slack
    ]
    # Stable IDs for connected one-to-many/many-to-one components.
    before_links: dict[int, list[int]] = {}
    after_links: dict[int, list[int]] = {}
    for row in valid:
        before_links.setdefault(row["before_id"], []).append(row["after_id"])
        after_links.setdefault(row["after_id"], []).append(row["before_id"])
    pair_by_key = {(row["before_id"], row["after_id"]): row for row in valid}
    component = 0
    visited_before: set[int] = set()
    visited_after: set[int] = set()
    for start in sorted(before_links):
        if start in visited_before:
            continue
        component += 1
        stack_before = [start]
        component_before: set[int] = set()
        component_after: set[int] = set()
        while stack_before:
            before_id = stack_before.pop()
            if before_id in component_before:
                continue
            component_before.add(before_id)
            visited_before.add(before_id)
            for after_id in before_links.get(before_id, []):
                if after_id not in component_after:
                    component_after.add(after_id)
                    visited_after.add(after_id)
                    stack_before.extend(after_links.get(after_id, []))
        if len(component_before) == 1 and len(component_after) == 1:
            relation = "one_to_one"
        elif len(component_before) == 1:
            relation = "one_to_many"
        elif len(component_after) == 1:
            relation = "many_to_one"
        else:
            relation = "partial"
        for before_id in component_before:
            for after_id in before_links.get(before_id, []):
                row = pair_by_key[(before_id, after_id)]
                row["match_id"] = f"M{component:08d}"
                row["relation"] = relation

    match_rows = []
    for row in valid:
        before_row = before.iloc[row["before_id"]]
        after_row = after.iloc[row["after_id"]]
        match_rows.append({
            **row,
            "before_w": _number(before_row, ("width_m",)),
            "after_w": _number(after_row, ("width_m",)),
            "qa_state": "review" if "review" in {str(before_row.get("qa_state", "")), str(after_row.get("qa_state", ""))} else "auto",
            "geometry": before_row.geometry,
        })
    columns = {
        "match_id": pd.Series(dtype="object"), "relation": pd.Series(dtype="object"),
        "before_id": pd.Series(dtype="int64"), "after_id": pd.Series(dtype="int64"),
        "match_score": pd.Series(dtype="float64"),
    }
    matches = (
        gpd.GeoDataFrame(match_rows, geometry="geometry", crs=before.crs)
        if match_rows else gpd.GeoDataFrame(columns, geometry=[], crs=before.crs)
    )
    metadata = {
        "spatial_candidate_count": len(raw) + rejected_direction + rejected_overlap + rejected_distance,
        "valid_candidate_count": len(valid),
        "rejected_direction_count": rejected_direction,
        "rejected_overlap_count": rejected_overlap,
        "rejected_distance_count": rejected_distance,
        "rejected_parallel_competitor_count": len(raw) - len(valid),
        "matched_before_segment_count": len(before_links),
        "matched_after_segment_count": len(after_links),
        "unmatched_before_segment_count": len(before) - len(before_links),
        "unmatched_after_segment_count": len(after) - len(after_links),
    }
    return matches, before_links, after_links, metadata
