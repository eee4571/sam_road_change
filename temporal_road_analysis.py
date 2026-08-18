from __future__ import annotations

"""Build cross-period road lifecycle Shapefiles from a pipeline manifest.

The module deliberately writes only ESRI Shapefile products.  ``road_life``
is the one-row-per-road wide table intended for direct GIS inspection, while
``road_obs`` and ``road_event`` retain normalized temporal detail.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely import STRtree, make_valid
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry


LINE_TYPES = {"LineString", "MultiLineString"}
WIDTH_FIELDS = ("width_map", "width_m", "final_width", "width")
CONF_FIELDS = ("confidence", "conf", "center_p", "gap_score")
STATUS_CODES = {"present": "P", "absent": "A", "uncertain": "U", "no_data": "N"}
EVENT_CODES = {
    "added": "A", "removed": "R", "widened": "W", "narrowed": "N",
    "rerouted": "D", "uncertain": "U",
}


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value)))


def clean_name(value: str) -> str:
    result = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).strip())
    return result.strip("._-") or "grid"


def _line_parts(geometry: BaseGeometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    geometry = make_valid(geometry)
    if geometry.geom_type == "LineString":
        if geometry.length > 0:
            yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _line_parts(part)


def _polygon_parts(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    if geometry is None or geometry.is_empty:
        return
    geometry = make_valid(geometry)
    if geometry.geom_type == "Polygon":
        if geometry.area > 0:
            yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def _metric_crs(frame: gpd.GeoDataFrame):
    crs = frame.crs
    if crs is None:
        raise ValueError("道路中心线缺少 CRS，无法进行长时序匹配")
    metric = (crs.is_projected or getattr(crs, "is_engineering", False)) and all(
        axis.unit_name and axis.unit_name.casefold() in {"metre", "meter"}
        for axis in crs.axis_info
    )
    if metric:
        return crs
    estimated = frame.estimate_utm_crs()
    if estimated is None:
        raise ValueError("无法为道路中心线估算米制分析 CRS")
    return estimated


def _numeric(row: pd.Series, fields: tuple[str, ...], default: float = 0.0) -> float:
    for field in fields:
        if field not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
        if pd.notna(value) and math.isfinite(float(value)):
            return float(value)
    return float(default)


def _read_period(path: Path, target_crs=None) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"道路中心线缺少 CRS：{path}")
    if target_crs is not None and frame.crs != target_crs:
        frame = frame.to_crs(target_crs)
    rows = []
    for source_index, row in frame.iterrows():
        for part_index, geometry in enumerate(_line_parts(row.geometry)):
            rows.append({
                "source_fid": str(row.get("global_id", source_index)),
                "part_idx": int(part_index),
                "width_m": _numeric(row, WIDTH_FIELDS, 0.0),
                "extract_cf": _numeric(row, CONF_FIELDS, 0.0),
                "geometry": geometry,
            })
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=frame.crs)
    if result.empty:
        return gpd.GeoDataFrame(
            {"source_fid": pd.Series(dtype="str"), "part_idx": pd.Series(dtype="int64"),
             "width_m": pd.Series(dtype="float64"), "extract_cf": pd.Series(dtype="float64")},
            geometry=gpd.GeoSeries([], crs=frame.crs), crs=frame.crs,
        )
    result["_sort"] = result.geometry.map(
        lambda geometry: tuple(round(value, 3) for value in geometry.bounds)
        + (round(float(geometry.length), 3), geometry.wkb_hex)
    )
    return result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _line_score(source: BaseGeometry, target: BaseGeometry, tolerance: float) -> tuple[float, float, float]:
    if source.is_empty or target.is_empty or source.length <= 0 or target.length <= 0:
        return 0.0, 0.0, float("inf")
    distance = float(source.distance(target))
    buffer_distance = max(float(tolerance), 0.1)
    source_cover = float(source.intersection(target.buffer(buffer_distance)).length) / float(source.length)
    target_cover = float(target.intersection(source.buffer(buffer_distance)).length) / float(target.length)
    overlap = min(max(source_cover, 0.0), max(target_cover, 0.0), 1.0)
    proximity = max(0.0, 1.0 - distance / buffer_distance)
    direction = _direction_similarity(source, target)
    # Direction is deliberately part of the score: at an intersection a
    # perpendicular road can be spatially close, but it is not the same road.
    return 0.65 * overlap + 0.20 * direction + 0.15 * proximity, overlap, distance


def _direction_vector(geometry: BaseGeometry) -> np.ndarray | None:
    """Return an orientation vector (sign-free) using sampled-point PCA."""
    if geometry is None or geometry.is_empty or geometry.length <= 0:
        return None
    distances = np.linspace(0.0, float(geometry.length), 9)
    coordinates = np.asarray([
        geometry.interpolate(float(distance)).coords[0][:2] for distance in distances
    ], dtype="float64")
    centered = coordinates - coordinates.mean(axis=0)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    if float(values[-1]) <= 1e-12:
        return None
    vector = vectors[:, -1]
    return vector / np.linalg.norm(vector)


def _direction_similarity(source: BaseGeometry, target: BaseGeometry) -> float:
    source_vector = _direction_vector(source)
    target_vector = _direction_vector(target)
    if source_vector is None or target_vector is None:
        return 0.5
    return float(np.clip(abs(np.dot(source_vector, target_vector)), 0.0, 1.0))


def _best_match(
    geometry: BaseGeometry,
    targets: list[BaseGeometry],
    tree: STRtree | None,
    tolerance: float,
) -> tuple[int | None, float, float, float, bool]:
    if tree is None or not targets:
        return None, 0.0, 0.0, float("inf"), False
    indices = tree.query(geometry, predicate="dwithin", distance=max(tolerance, 0.1))
    ranked = []
    for raw_index in indices:
        index = int(raw_index)
        score, overlap, distance = _line_score(geometry, targets[index], tolerance)
        ranked.append((score, overlap, -distance, index))
    if not ranked:
        return None, 0.0, 0.0, float("inf"), False
    ranked.sort(reverse=True)
    score, overlap, negative_distance, index = ranked[0]
    ambiguous = len(ranked) > 1 and ranked[1][0] >= score - 0.05 and ranked[1][0] >= 0.35
    return index, float(score), float(overlap), float(-negative_distance), ambiguous


def _build_reference(period_frames: dict[str, gpd.GeoDataFrame], tolerance: float) -> list[dict]:
    references: list[dict] = []
    for period, frame in period_frames.items():
        existing_geometries = [item["geometry"] for item in references]
        tree = STRtree(np.asarray(existing_geometries, dtype=object)) if existing_geometries else None
        new_references: list[dict] = []
        new_geometries: list[BaseGeometry] = []
        for _, row in frame.iterrows():
            geometry = row.geometry
            index, score, overlap, _distance, _ambiguous = _best_match(
                geometry, existing_geometries, tree, tolerance,
            )
            matched = index is not None and score >= 0.48 and overlap >= 0.35
            if not matched and new_geometries:
                local_scores = [(_line_score(geometry, candidate, tolerance), idx) for idx, candidate in enumerate(new_geometries)]
                local_scores.sort(key=lambda item: item[0][0], reverse=True)
                if local_scores and local_scores[0][0][0] >= 0.48 and local_scores[0][0][1] >= 0.35:
                    matched = True
            if matched:
                continue
            item = {"geometry": geometry, "born_period": period}
            new_references.append(item)
            new_geometries.append(geometry)
        references.extend(new_references)
    return references


def _numeric_road_id(value: str) -> int | None:
    match = re.fullmatch(r"RD(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def _assign_road_ids(references: list[dict], prior_path: Path, analysis_crs, tolerance: float) -> None:
    prior = None
    if prior_path.is_file():
        try:
            prior = gpd.read_file(prior_path)
            if prior.crs is not None and prior.crs != analysis_crs:
                prior = prior.to_crs(analysis_crs)
            if "road_id" not in prior.columns:
                prior = None
        except Exception:
            prior = None
    used_ids: set[str] = set()
    maximum = 0
    if prior is not None:
        prior = prior.loc[prior.geometry.notna() & ~prior.geometry.is_empty].copy()
        prior_geometries = list(prior.geometry)
        prior_tree = STRtree(np.asarray(prior_geometries, dtype=object)) if prior_geometries else None
        candidates = []
        for ref_index, reference in enumerate(references):
            prior_index, score, overlap, _distance, _ambiguous = _best_match(
                reference["geometry"], prior_geometries, prior_tree, tolerance,
            )
            if prior_index is not None and score >= 0.65 and overlap >= 0.55:
                candidates.append((score, overlap, ref_index, prior_index))
        assigned_refs: set[int] = set()
        assigned_prior: set[int] = set()
        for _score, _overlap, ref_index, prior_index in sorted(candidates, reverse=True):
            if ref_index in assigned_refs or prior_index in assigned_prior:
                continue
            road_id = str(prior.iloc[prior_index]["road_id"])
            if not road_id:
                continue
            references[ref_index]["road_id"] = road_id
            used_ids.add(road_id)
            assigned_refs.add(ref_index)
            assigned_prior.add(prior_index)
        prior_numbers = [_numeric_road_id(value) for value in prior["road_id"]]
        maximum = max((value for value in prior_numbers if value is not None), default=0)
    for reference in references:
        if reference.get("road_id"):
            continue
        maximum += 1
        road_id = f"RD{maximum:08d}"
        while road_id in used_ids:
            maximum += 1
            road_id = f"RD{maximum:08d}"
        reference["road_id"] = road_id
        used_ids.add(road_id)


def _node_id(coordinate: tuple[float, float], tolerance: float) -> str:
    scale = max(tolerance * 0.5, 0.1)
    snapped = (round(float(coordinate[0]) / scale), round(float(coordinate[1]) / scale))
    digest = hashlib.sha1(f"{snapped[0]}:{snapped[1]}".encode("ascii")).hexdigest()[:9].upper()
    return f"N{digest}"


def _observations(
    references: list[dict],
    period_frames: dict[str, gpd.GeoDataFrame],
    tolerance: float,
) -> list[dict]:
    observations = []
    for period, frame in period_frames.items():
        source_geometries = list(frame.geometry)
        source_tree = STRtree(np.asarray(source_geometries, dtype=object)) if source_geometries else None
        period_rows = []
        for reference_index, reference in enumerate(references):
            geometry = reference["geometry"]
            index, score, overlap, distance, ambiguous = _best_match(
                geometry, source_geometries, source_tree, tolerance,
            )
            if index is None:
                status = "absent"
                row = None
            else:
                row = frame.iloc[index]
                if overlap >= 0.35 and score >= 0.48 and not ambiguous:
                    status = "present"
                elif overlap >= 0.10 or distance <= tolerance:
                    status = "uncertain"
                else:
                    status = "absent"
            direction = _direction_similarity(geometry, source_geometries[index]) if index is not None else 0.0
            item = {
                "road_id": reference["road_id"], "period": str(period), "status": status,
                "width_m": float(row["width_m"]) if row is not None and status == "present" else np.nan,
                "length_m": float(geometry.length) if status == "present" else 0.0,
                "coverage": float(overlap), "match_sc": float(score),
                "extract_cf": float(row["extract_cf"]) if row is not None else 0.0,
                "geom_dev_m": float(distance) if math.isfinite(distance) else np.nan,
                "source_fid": str(row["source_fid"]) if row is not None else "",
                "qa_state": "review" if status == "uncertain" else "auto",
                "qa_reason": "ambiguous_or_weak_match" if status == "uncertain" else "",
                "dir_sim": direction,
                "_source_idx": int(index) if index is not None else -1,
                "_reference_idx": reference_index,
                "geometry": geometry,
            }
            period_rows.append(item)

        # A segmented source line may legitimately support several disjoint
        # references.  Reuse is only rejected when the references themselves
        # overlap strongly, which signals duplicate stable-road ownership.
        by_source: dict[int, list[dict]] = {}
        for item in period_rows:
            if item["status"] == "present" and item["_source_idx"] >= 0:
                by_source.setdefault(item["_source_idx"], []).append(item)
        for claimed_rows in by_source.values():
            claimed_rows.sort(key=lambda item: item["match_sc"], reverse=True)
            accepted: list[dict] = []
            for item in claimed_rows:
                duplicate = any(
                    _line_score(item["geometry"], other["geometry"], tolerance)[1] >= 0.35
                    for other in accepted
                )
                if duplicate:
                    item["status"] = "uncertain"
                    item["qa_state"] = "review"
                    item["qa_reason"] = "duplicate_source_ownership"
                else:
                    accepted.append(item)
        observations.extend(period_rows)
    by_road: dict[str, list[dict]] = {}
    for row in observations:
        by_road.setdefault(row["road_id"], []).append(row)
    period_order = {period: index for index, period in enumerate(period_frames)}
    for rows in by_road.values():
        rows.sort(key=lambda row: period_order[row["period"]])
        for index in range(1, len(rows) - 1):
            if rows[index - 1]["status"] == "present" and rows[index]["status"] == "absent" and rows[index + 1]["status"] == "present":
                rows[index]["status"] = "uncertain"
                rows[index]["qa_state"] = "review"
                rows[index]["qa_reason"] = "single_period_dropout"
    for row in observations:
        row.pop("_source_idx", None)
        row.pop("_reference_idx", None)
    return observations


def _events(observations: list[dict], periods: list[str], absolute: float, ratio: float) -> list[dict]:
    by_road_period = {(row["road_id"], row["period"]): row for row in observations}
    road_ids = sorted({row["road_id"] for row in observations})
    events = []
    counter = 0
    for road_id in road_ids:
        for before_period, after_period in zip(periods, periods[1:]):
            before = by_road_period[(road_id, before_period)]
            after = by_road_period[(road_id, after_period)]
            event_type = None
            if before["status"] == "absent" and after["status"] == "present":
                event_type = "added"
            elif before["status"] == "present" and after["status"] == "absent":
                event_type = "removed"
            elif before["status"] == "present" and after["status"] == "present":
                before_width, after_width = before["width_m"], after["width_m"]
                if pd.notna(before_width) and pd.notna(after_width):
                    width_diff = float(after_width - before_width)
                    relative = abs(width_diff) / max(float(before_width), float(after_width), 0.1)
                    if abs(width_diff) >= absolute and relative >= ratio:
                        event_type = "widened" if width_diff > 0 else "narrowed"
            elif before["status"] != after["status"] and "uncertain" in {before["status"], after["status"]}:
                event_type = "uncertain"
            if event_type is None:
                continue
            counter += 1
            before_width = before["width_m"]
            after_width = after["width_m"]
            width_diff = float(after_width - before_width) if pd.notna(before_width) and pd.notna(after_width) else np.nan
            events.append({
                "event_id": f"EV{counter:08d}", "road_id": road_id,
                "from_per": before_period, "to_per": after_period, "event_typ": event_type,
                "before_st": before["status"], "after_st": after["status"],
                "before_w": before_width, "after_w": after_width, "width_diff": width_diff,
                "event_cf": min(float(before["match_sc"]), float(after["match_sc"])),
                "evidence": "temporal_state_and_centerline_match",
                "qa_state": "review" if event_type == "uncertain" else "auto",
                "geometry": after["geometry"] if after["status"] == "present" else before["geometry"],
            })
    return events


def _field_name(prefix: str, period: str, used: set[str]) -> str:
    token = "".join(ch for ch in str(period) if ch.isalnum()).upper() or "P"
    base = (prefix + token)[-10:]
    candidate = base
    counter = 1
    while candidate.casefold() in used:
        suffix = str(counter)
        candidate = base[: 10 - len(suffix)] + suffix
        counter += 1
    used.add(candidate.casefold())
    return candidate


def _life_rows(references: list[dict], observations: list[dict], events: list[dict], periods: list[str], tolerance: float) -> list[dict]:
    by_road_period = {(row["road_id"], row["period"]): row for row in observations}
    event_by_key = {(row["road_id"], row["from_per"], row["to_per"]): row for row in events}
    event_counts: dict[str, int] = {}
    event_reviews: set[str] = set()
    for event in events:
        if event["event_typ"] != "uncertain":
            event_counts[event["road_id"]] = event_counts.get(event["road_id"], 0) + 1
        if event.get("qa_state") == "review":
            event_reviews.add(event["road_id"])
    used = {name.casefold() for name in ("road_id", "grid_id", "from_node", "to_node", "first_obs", "last_obs", "life_state", "present_n", "event_n", "max_conf", "min_conf", "review_st")}
    state_fields = {period: _field_name("S", period, used) for period in periods}
    width_fields = {period: _field_name("W", period, used) for period in periods}
    conf_fields = {period: _field_name("C", period, used) for period in periods}
    event_fields = {
        (before, after): _field_name("E", f"{before[-4:]}{after[-4:]}", used)
        for before, after in zip(periods, periods[1:])
    }
    rows = []
    for reference in references:
        road_id = reference["road_id"]
        road_observations = [by_road_period[(road_id, period)] for period in periods]
        present_periods = [row["period"] for row in road_observations if row["status"] == "present"]
        confidences = [float(row["match_sc"]) for row in road_observations]
        last_status = road_observations[-1]["status"]
        row = {
            "road_id": road_id, "grid_id": "", "from_node": _node_id(reference["geometry"].coords[0], tolerance),
            "to_node": _node_id(reference["geometry"].coords[-1], tolerance),
            "first_obs": present_periods[0] if present_periods else "",
            "last_obs": present_periods[-1] if present_periods else "",
            "life_state": "active" if last_status == "present" else ("removed" if last_status == "absent" else "uncertain"),
            "present_n": len(present_periods),
            "event_n": event_counts.get(road_id, 0),
            "max_conf": max(confidences, default=0.0), "min_conf": min(confidences, default=0.0),
            "review_st": "review" if (
                any(item["status"] == "uncertain" for item in road_observations)
                or road_id in event_reviews
            ) else "auto",
            "geometry": reference["geometry"],
        }
        for period in periods:
            observation = by_road_period[(road_id, period)]
            row[state_fields[period]] = STATUS_CODES[observation["status"]]
            row[width_fields[period]] = observation["width_m"]
            row[conf_fields[period]] = observation["match_sc"]
        for pair, field in event_fields.items():
            event = event_by_key.get((road_id, *pair))
            row[field] = EVENT_CODES.get(event["event_typ"], "0") if event else "0"
        rows.append(row)
    return rows


def _event_parts(
    change_entries: list[dict],
    references: list[dict],
    events: list[dict],
    analysis_crs,
    tolerance: float,
) -> list[dict]:
    reference_geometries = [item["geometry"] for item in references]
    tree = STRtree(np.asarray(reference_geometries, dtype=object)) if reference_geometries else None
    event_lookup = {(item["road_id"], item["from_per"], item["to_per"], item["event_typ"]): item["event_id"] for item in events}
    rows = []
    for entry in change_entries:
        path = Path(str(entry.get("output", ""))) / "road_changes.shp"
        required = [path, path.with_suffix(".dbf"), path.with_suffix(".shx")]
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError("变化检测成果不完整，缺少：" + "；".join(missing))
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"变化检测成果缺少 CRS：{path}")
        if frame.crs != analysis_crs:
            frame = frame.to_crs(analysis_crs)
        before_period, after_period = str(entry.get("before_period", "")), str(entry.get("after_period", ""))
        for source_index, source in frame.iterrows():
            geometry = source.geometry
            if geometry is None or geometry.is_empty:
                continue
            probe = geometry.boundary if hasattr(geometry, "boundary") else geometry
            indices = tree.query(geometry.buffer(tolerance), predicate="intersects") if tree is not None else []
            ranked = []
            for raw_index in indices:
                index = int(raw_index)
                reference = reference_geometries[index]
                support = float(reference.intersection(geometry.buffer(tolerance)).length)
                ranked.append((support, -float(reference.distance(probe)), index))
            road_id = ""
            if ranked:
                _support, _distance, ref_index = max(ranked)
                road_id = references[ref_index]["road_id"]
            event_type = str(source.get("change_typ", ""))
            for part_index, part in enumerate(_polygon_parts(geometry)):
                rows.append({
                    "event_id": event_lookup.get((road_id, before_period, after_period, event_type), ""),
                    "road_id": road_id, "from_per": before_period, "to_per": after_period,
                    "event_typ": event_type,
                    "source_id": f"{source.get('change_id', source_index)}:{part_index}",
                    "area_m2": float(part.area), "qa_state": "auto" if road_id else "review",
                    "geometry": part,
                })
    return rows


def _apply_event_evidence(events: list[dict], event_parts: list[dict], has_pair_results: bool) -> None:
    supported = {str(row.get("event_id", "")) for row in event_parts if row.get("event_id")}
    for event in events:
        if event["event_typ"] == "uncertain":
            continue
        if event["event_id"] in supported:
            event["evidence"] = "temporal_state_and_pair_change"
            continue
        event["evidence"] = "state_only_no_pair_part" if has_pair_results else "state_only_no_pair_input"
        event["qa_state"] = "review"


def _write_shp(path: Path, rows: list[dict], columns: dict[str, str], crs, geometry_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        for name, dtype in columns.items():
            if name not in frame.columns:
                frame[name] = "" if dtype == "str" else np.nan
        frame = frame[[*columns, "geometry"]]
    else:
        data = {name: pd.Series(dtype=dtype) for name, dtype in columns.items()}
        frame = gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    pyogrio.write_dataframe(
        frame, path, driver="ESRI Shapefile", encoding="UTF-8", geometry_type=geometry_type,
    )


def build_temporal_grid(
    grid_name: str,
    period_entries: list[dict],
    change_entries: list[dict],
    output_dir: Path,
    tolerance: float = 3.0,
    width_absolute: float = 1.0,
    width_ratio: float = 0.1,
) -> dict:
    period_entries = sorted(period_entries, key=lambda item: natural_key(str(item["period"])))
    if len(period_entries) < 2:
        raise ValueError(f"{grid_name} 至少需要两个完整期次才能生成长时序成果")
    paths = [Path(str(item.get("centerlines", ""))).expanduser() for item in period_entries]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少单期中心线：" + "；".join(missing))
    first_raw = gpd.read_file(paths[0])
    output_crs = first_raw.crs
    analysis_crs = _metric_crs(first_raw)
    period_frames = {
        str(entry["period"]): _read_period(path, analysis_crs)
        for entry, path in zip(period_entries, paths)
    }
    periods = list(period_frames)
    references = _build_reference(period_frames, tolerance)
    if not references:
        raise ValueError(f"{grid_name} 的所有期次均没有有效道路中心线")
    output_dir = Path(output_dir).resolve()
    _assign_road_ids(references, output_dir / "road_life.shp", analysis_crs, tolerance)
    observations = _observations(references, period_frames, tolerance)
    events = _events(observations, periods, width_absolute, width_ratio)
    event_parts = _event_parts(change_entries, references, events, analysis_crs, tolerance)
    _apply_event_evidence(events, event_parts, bool(change_entries))
    life = _life_rows(references, observations, events, periods, tolerance)
    for row in life:
        row["grid_id"] = str(grid_name)[:80]
    reviews = [{
        "road_id": row["road_id"], "period": row["period"],
        "reason": row.get("qa_reason") or "ambiguous_or_weak_match",
        "match_sc": row["match_sc"], "coverage": row["coverage"], "qa_state": "review",
        "geometry": row["geometry"],
    } for row in observations if row["status"] == "uncertain"]
    reviews.extend({
        "road_id": row["road_id"], "period": row["to_per"], "reason": row["evidence"],
        "match_sc": row["event_cf"], "coverage": 0.0, "qa_state": "review",
        "geometry": row["geometry"],
    } for row in events if row["qa_state"] == "review" and row["event_typ"] != "uncertain")

    _write_shp(output_dir / "road_life.shp", life,
               {name: ("str" if name not in {"present_n", "event_n", "max_conf", "min_conf"} and not name.startswith(("W", "C")) else "float64")
                for name in life[0] if name != "geometry"}, analysis_crs, "LineString")
    _write_shp(output_dir / "road_obs.shp", observations, {
        "road_id": "str", "period": "str", "status": "str", "width_m": "float64",
        "length_m": "float64", "coverage": "float64", "match_sc": "float64",
        "extract_cf": "float64", "geom_dev_m": "float64", "source_fid": "str", "qa_state": "str",
        "qa_reason": "str", "dir_sim": "float64",
    }, analysis_crs, "LineString")
    _write_shp(output_dir / "road_event.shp", events, {
        "event_id": "str", "road_id": "str", "from_per": "str", "to_per": "str",
        "event_typ": "str", "before_st": "str", "after_st": "str", "before_w": "float64",
        "after_w": "float64", "width_diff": "float64", "event_cf": "float64",
        "evidence": "str", "qa_state": "str",
    }, analysis_crs, "LineString")
    _write_shp(output_dir / "event_parts.shp", event_parts, {
        "event_id": "str", "road_id": "str", "from_per": "str", "to_per": "str",
        "event_typ": "str", "source_id": "str", "area_m2": "float64", "qa_state": "str",
    }, analysis_crs, "Polygon")
    _write_shp(output_dir / "road_lineage.shp", [], {
        "parent_id": "str", "child_id": "str", "period": "str", "relation": "str",
        "confidence": "float64", "qa_state": "str",
    }, analysis_crs, "LineString")
    _write_shp(output_dir / "road_review.shp", reviews, {
        "road_id": "str", "period": "str", "reason": "str", "match_sc": "float64",
        "coverage": "float64", "qa_state": "str",
    }, analysis_crs, "LineString")

    if output_crs is not None and output_crs != analysis_crs:
        for name in ("road_life.shp", "road_obs.shp", "road_event.shp", "event_parts.shp", "road_lineage.shp", "road_review.shp"):
            path = output_dir / name
            frame = gpd.read_file(path).to_crs(output_crs)
            geometry_type = "Polygon" if name == "event_parts.shp" else "LineString"
            pyogrio.write_dataframe(frame, path, driver="ESRI Shapefile", encoding="UTF-8", geometry_type=geometry_type)

    result = {
        "grid": grid_name, "output": str(output_dir),
        "life_shp": str(output_dir / "road_life.shp"),
        "observations_shp": str(output_dir / "road_obs.shp"),
        "events_shp": str(output_dir / "road_event.shp"),
        "event_parts_shp": str(output_dir / "event_parts.shp"),
        "lineage_shp": str(output_dir / "road_lineage.shp"),
        "review_shp": str(output_dir / "road_review.shp"),
        "period_count": len(periods), "road_count": len(references),
        "observation_count": len(observations), "event_count": len(events),
        "review_count": len(reviews),
    }
    return result


def build_from_manifest(manifest: dict, job_root: Path | None = None) -> list[dict]:
    root = Path(job_root or manifest.get("job_root") or ".").expanduser().resolve()
    period_by_grid: dict[str, list[dict]] = {}
    for entry in manifest.get("period_results", []):
        if isinstance(entry, dict):
            period_by_grid.setdefault(str(entry.get("grid", "validation")), []).append(entry)
    change_by_grid: dict[str, list[dict]] = {}
    for entry in manifest.get("change_results", []):
        if isinstance(entry, dict):
            change_by_grid.setdefault(str(entry.get("grid", "validation")), []).append(entry)
    tolerance = float(manifest.get("input_spec", {}).get("tolerance", manifest.get("tolerance", 3.0)) or 3.0)
    absolute = float(manifest.get("input_spec", {}).get("absolute", manifest.get("absolute", 1.0)) or 1.0)
    ratio = float(manifest.get("input_spec", {}).get("ratio", manifest.get("ratio", 0.1)) or 0.1)
    results = []
    for grid_name, entries in sorted(period_by_grid.items(), key=lambda item: natural_key(item[0])):
        if len(entries) < 2:
            continue
        centerline_paths = [Path(str(entry.get("centerlines", ""))) for entry in entries]
        missing = [
            str(required)
            for path in centerline_paths
            for required in (path, path.with_suffix(".dbf"), path.with_suffix(".shx"))
            if not required.is_file()
        ]
        if missing:
            # Mocked dry-runs historically use zero-byte placeholder .shp files.
            # Preserve that test/preview mode, but never hide damaged real data.
            placeholders = centerline_paths and all(path.is_file() and path.stat().st_size == 0 for path in centerline_paths)
            no_outputs_yet = not any(path.is_file() for path in centerline_paths)
            if placeholders or no_outputs_yet:
                continue
            raise FileNotFoundError(
                f"格网 {grid_name} 的单期中心线成果不完整，缺少：" + "；".join(missing)
            )
        output_dir = root / "grids" / clean_name(grid_name) / "05_长时序成果"
        results.append(build_temporal_grid(
            grid_name, entries, change_by_grid.get(grid_name, []), output_dir,
            tolerance=tolerance, width_absolute=absolute, width_ratio=ratio,
        ))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从多期道路成果生成纯 SHP 长时序道路成果")
    parser.add_argument("--pipeline-manifest", required=True)
    args = parser.parse_args(argv)
    path = Path(args.pipeline_manifest).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    results = build_from_manifest(manifest)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
