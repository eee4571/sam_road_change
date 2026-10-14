from __future__ import annotations

"""Build cross-period road lifecycle Shapefiles from a pipeline manifest.

The module deliberately writes only ESRI Shapefile products.  ``road_life``
is the one-row-per-road wide table intended for direct GIS inspection, while
``road_obs`` and ``road_event`` retain normalized temporal detail.
"""

import argparse
import hashlib
import json
from engine.fast_timing import timed_stage
import math
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from engine.disk_records import DiskRows, LazyFrames, observation_index, record_scope, batches
from shapely import STRtree, make_valid, line_merge
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry


LINE_TYPES = {"LineString", "MultiLineString"}
WIDTH_FIELDS = ("width_map", "width_m", "final_width", "width")
CONF_FIELDS = ("confidence", "conf", "center_p", "gap_score")
STATUS_CODES = {"present": "P", "absent": "A", "uncertain": "U", "no_data": "N"}
EVENT_CODES = {
    "added": "A", "removed": "R", "widened": "W", "narrowed": "N",
    "width_changed": "C",
    "uncertain": "U",
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


def _road_geometry(geometry):
    """Keep every linear component as one road, without bridging gaps.

    SHP cannot store GeometryCollection. Points/polygons in a collection are
    not road axes; recursively retain its complete linear geometry instead.
    """
    parts = list(_line_parts(geometry))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiLineString(parts)


def _road_nodes(geometry, tolerance):
    # Merge only exactly connected, non-branching components for the purpose
    # of finding endpoints. Never replace the stored geometry with this view.
    line = line_merge(geometry)
    if line.geom_type != 'LineString' or not line.is_simple or line.is_ring:
        return '', ''
    return _node_id(line.coords[0], tolerance), _node_id(line.coords[-1], tolerance)


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
    from engine.road_surface_quality import reliable_width
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"道路中心线缺少 CRS：{path}")
    if target_crs is not None and frame.crs != target_crs:
        frame = frame.to_crs(target_crs)
    rows = []
    for source_index, row in frame.iterrows():
        geometry = _road_geometry(row.geometry)
        if geometry is not None:
            rows.append({
                "source_fid": str(row.get("global_id", source_index)),
                "part_idx": 0,
                "width_m": _numeric(row, WIDTH_FIELDS, 0.0),
                "width_reliable": (True if (quality := reliable_width(row)) is None else bool(quality)),
                "extract_cf": _numeric(row, CONF_FIELDS, 0.0),
                "track_id": str(row.get("track_id") or "") if pd.notna(row.get("track_id")) else "",
                "geometry": geometry,
            })
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=frame.crs) if rows else gpd.GeoDataFrame(geometry=[], crs=frame.crs)
    if result.empty:
        return gpd.GeoDataFrame(
            {"source_fid": pd.Series(dtype="str"), "part_idx": pd.Series(dtype="int64"),
             "width_m": pd.Series(dtype="float64"), "extract_cf": pd.Series(dtype="float64"), "track_id": pd.Series(dtype="str")},
            geometry=gpd.GeoSeries([], crs=frame.crs), crs=frame.crs,
        )
    result["_sort"] = result.geometry.map(
        lambda geometry: tuple(round(value, 3) for value in geometry.bounds)
        + (round(float(geometry.length), 3), geometry.wkb_hex)
    )
    return result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


from functools import lru_cache, wraps


@lru_cache(maxsize=16384)
def _geometry_facts(geometry):
    ends = tuple(Point(part.coords[i]) for part in _line_parts(geometry) for i in (0, -1))
    return geometry.length, geometry.bounds, ends


@lru_cache(maxsize=16384)
def _geometry_buffer(geometry, distance):
    return geometry.buffer(distance)


def _geometry_scope(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        finally:
            _geometry_facts.cache_clear(); _geometry_buffer.cache_clear(); _direction_vector.cache_clear()
    return wrapped


def _line_score(source: BaseGeometry, target: BaseGeometry, tolerance: float) -> tuple[float, float, float]:
    if source.is_empty or target.is_empty or source.length <= 0 or target.length <= 0:
        return 0.0, 0.0, float("inf")
    distance = float(source.distance(target))
    buffer_distance = max(float(tolerance), 0.1)
    source_cover = float(source.intersection(_geometry_buffer(target,buffer_distance)).length) / _geometry_facts(source)[0]
    target_cover = float(target.intersection(_geometry_buffer(source,buffer_distance)).length) / _geometry_facts(target)[0]
    overlap = min(max(source_cover, 0.0), max(target_cover, 0.0), 1.0)
    proximity = max(0.0, 1.0 - distance / buffer_distance)
    direction = _direction_similarity(source, target)
    # Direction is deliberately part of the score: at an intersection a
    # perpendicular road can be spatially close, but it is not the same road.
    return 0.65 * overlap + 0.20 * direction + 0.15 * proximity, overlap, distance


@lru_cache(maxsize=16384)
def _direction_vector(geometry: BaseGeometry) -> np.ndarray | None:
    """Return an orientation vector (sign-free) using sampled-point PCA."""
    if geometry is None or geometry.is_empty or geometry.length <= 0:
        return None
    # Sum length-weighted, locally centered covariance. Empty space between
    # disconnected pieces must not invent an orientation across that gap.
    covariance = np.zeros((2, 2), dtype='float64')
    for part in _line_parts(geometry):
        coordinates = np.asarray([part.interpolate(float(d)).coords[0][:2]
                                  for d in np.linspace(0., part.length, 9)])
        centered = coordinates - coordinates.mean(axis=0)
        local = centered.T @ centered
        trace = np.trace(local)
        if trace > 0:
            covariance += local / trace * part.length
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
    indices=None,
) -> tuple[int | None, float, float, float, bool]:
    if tree is None or not targets:
        return None, 0.0, 0.0, float("inf"), False
    if indices is None:
        indices = tree.query(geometry, predicate="dwithin", distance=max(tolerance, 0.1))
    ranked = []
    for raw_index in indices:
        index = int(raw_index)
        score, overlap, distance = _line_score(geometry, targets[index], tolerance)
        direction = _direction_similarity(geometry, targets[index])
        if direction < .5:
            continue
        # A continuous counterpart must explain the longitudinal axis, including
        # its endpoints. Direction alone cannot select an adjacent branch.
        ends = [point.distance(targets[index]) for point in _geometry_facts(geometry)[2]]
        continuity = float(np.mean(np.exp(-np.asarray(ends)/max(tolerance, .1)))) if ends else 0.
        total = .85*score + .15*continuity
        ranked.append((total, overlap, -distance, index))
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
        period_geometries = list(frame.geometry)
        period_tree = STRtree(period_geometries)
        accepted_indices = set()
        for row_index, (_, row) in enumerate(frame.iterrows()):
            geometry = row.geometry
            index, score, overlap, _distance, _ambiguous = _best_match(
                geometry, existing_geometries, tree, tolerance,
            )
            matched = index is not None and score >= 0.48 and overlap >= 0.35
            if not matched and accepted_indices:
                # Outside tolerance the overlap is zero and the maximum score
                # is .20, below .48. Spatial pruning preserves the old decision
                # exactly while avoiding a quadratic all-road buffer workload.
                indices = [int(i) for i in period_tree.query(geometry, predicate="dwithin", distance=max(tolerance, .1)) if int(i) in accepted_indices]
                local_index,local_score,local_overlap,_,_=_best_match(geometry,period_geometries,period_tree,tolerance,indices=indices)
                matched=local_index is not None and local_score>=.48 and local_overlap>=.35
            if matched:
                continue
            item = {"geometry": geometry, "born_period": period}
            new_references.append(item)
            accepted_indices.add(row_index)
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
            if reference.get('road_id'):
                continue
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
    observations = DiskRows()
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
                if overlap >= 0.10 and score >= 0.25:
                    status = "present"
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
                "qa_state": "low_confidence" if ambiguous or (index is not None and score < .48) else "auto",
                "qa_reason": "best_continuous_match_selected" if ambiguous else "low_match_score" if index is not None and score < .48 else "",
                "dir_sim": direction,
                "_source_idx": int(index) if index is not None else -1,
                "_reference_idx": reference_index,
                "geometry": geometry,
            }
            period_rows.append(item)
            if row is not None and not row.get('width_reliable',True):
                item['qa_state']='low_confidence'
                item['qa_reason']='unreliable_width'

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
                    item["status"] = "absent"
                    item["qa_state"] = "low_confidence"
                    item["qa_reason"] = ';'.join(filter(None, (item.get('qa_reason'), 'duplicate_source_ownership')))
                else:
                    accepted.append(item)
        observations.extend(period_rows)
    period_order = {period: index for index, period in enumerate(period_frames)}
    for road_id in observations.road_ids():
        indexed=observations.road_rows(road_id)
        indexed.sort(key=lambda pair: period_order[pair[1]['period']])
        rows=[r for _,r in indexed]
        for index in range(1, len(rows) - 1):
            if rows[index - 1]["status"] == "present" and rows[index]["status"] == "absent" and rows[index + 1]["status"] == "present":
                rows[index]["qa_state"] = "low_confidence"
                rows[index]["qa_reason"] = ';'.join(filter(None, (rows[index].get('qa_reason'), 'single_period_dropout')))
        for (index,row) in indexed:
            row.pop("_source_idx", None)
            row.pop("_reference_idx", None)
            observations[index]=row
    return observations


def _events(observations: list[dict], periods: list[str], absolute: float, ratio: float) -> list[dict]:
    by_road_period = observation_index(observations)
    road_ids = by_road_period.road_ids()
    events = DiskRows(mutable=True)
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
                        event_type = ('uncertain' if any('unreliable_width' in str(r.get('qa_reason','')) for r in (before,after))
                                      else "widened" if width_diff > 0 else "narrowed")
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
                "qa_state": "low_confidence" if before.get('qa_state')=='low_confidence' or after.get('qa_state')=='low_confidence' else "auto",
                "geometry": after["geometry"] if after["status"] == "present" else before["geometry"],
            })
    return events


def _reconciled_tracks(period_entries, period_frames, analysis_crs):
    """Read explicit reconciled identities, verifying them against period roads.

    A state table cannot invent presence or disappearance: matching track IDs
    must physically exist (or be absent) in that period's centerline product.
    """
    if not any(entry.get('road_state') for entry in period_entries):
        return [], []
    if not all(entry.get('road_state') for entry in period_entries):
        raise ValueError('部分期次缺少道路状态表，无法保留跨期道路身份')
    references={};observations=DiskRows();expected=None
    for entry in period_entries:
        period=str(entry['period'])
        state=gpd.read_file(entry['road_state'],layer='road_state').to_crs(analysis_crs)
        state_groups={str(k):v for k,v in state.groupby('track_id',sort=False)}
        if expected is None:expected=set(state_groups)
        if set(state_groups)!=expected:raise ValueError(f'Reconciled road state incomplete: {period}')
        roads=period_frames[period]
        road_groups={str(k):v for k,v in roads.groupby('track_id',sort=False)}
        for track_id in sorted(expected):
            rows=state_groups[track_id]
            if len(rows)!=1:raise ValueError(f'Reconciled road state incomplete: {track_id}/{period}')
            row=rows.iloc[0].copy()
            row.geometry = _road_geometry(row.geometry)
            if row.geometry is None:
                raise ValueError(f'Reconciled road state has no line geometry: {track_id}/{period}')
            references.setdefault(track_id,dict(road_id=track_id,born_period=str(period_entries[0]['period']),geometry=row.geometry))
            actual=road_groups.get(track_id,roads.iloc[:0])
            present=row.status=='present'
            if present != (not actual.empty):raise ValueError(f'Road state contradicts period centerline: {track_id}/{period}')
            width=float(np.average(actual.width_m,weights=actual.length)) if present else np.nan
            if present and abs(width-float(row.width_m))>.05:raise ValueError(f'Road state width contradicts final width: {track_id}/{period}')
            observations.append(dict(road_id=track_id,period=period,status=row.status,width_m=width,
                length_m=float(row.geometry.length) if present else 0.,coverage=1. if present else 0.,match_sc=1.,extract_cf=.95,
                geom_dev_m=0.,source_fid=','.join(actual.source_fid),qa_state='event_constrained',
                qa_reason=';'.join(str(v) for v in (row.get('state_source','reconciled'),
                    row.get('observation_conflict',''),
                    'unreliable_width' if present and not actual.width_reliable.all() else '') if v and str(v)!='nan'),
                dir_sim=1.,geometry=row.geometry))
    # Original externally visible order: track, then configured period order.
    ordered=DiskRows()
    order={str(e['period']):i for i,e in enumerate(period_entries)}
    for track_id in sorted(expected or ()):
        ordered.extend(r for _,r in sorted(observations.road_rows(track_id),key=lambda pair:order[pair[1]['period']]))
    observations.close()
    return [references[k] for k in sorted(references)],ordered


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
    by_road_period = observation_index(observations)
    event_by_key = (events.index(('road_id','from_per','to_per')) if isinstance(events,DiskRows) else
                    {(row['road_id'],row['from_per'],row['to_per']):row for row in events})
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
    rows = DiskRows(mutable=True)
    for reference in references:
        road_id = reference["road_id"]
        road_observations = [by_road_period[(road_id, period)] for period in periods]
        present_periods = [row["period"] for row in road_observations if row["status"] == "present"]
        confidences = [float(row["match_sc"]) for row in road_observations]
        last_status = road_observations[-1]["status"]
        geometry = _road_geometry(reference['geometry'])
        from_node, to_node = _road_nodes(geometry, tolerance)
        row = {
            "road_id": road_id, "grid_id": "", "from_node": from_node,
            "to_node": to_node,
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
            "geometry": geometry,
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
    observations: list[dict] | None = None,
) -> list[dict]:
    reference_geometries = [item["geometry"] for item in references]
    reference_indices = {item['road_id']: i for i, item in enumerate(references)}
    tree = STRtree(np.asarray(reference_geometries, dtype=object)) if reference_geometries else None
    event_lookup = events.index(('road_id','from_per','to_per','event_typ')) if isinstance(events,DiskRows) else {
        (item["road_id"], item["from_per"], item["to_per"], item["event_typ"]): item
        for item in events
    }
    rows = DiskRows()
    observed=observation_index(observations or [])
    for entry in change_entries:
        path = Path(str((entry.get('layers') or {}).get('changes') or entry.get('road_changes')
                        or Path(str(entry.get("output", ""))) / "road_changes.shp"))
        required = [path, path.with_suffix(".dbf"), path.with_suffix(".shx")]
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError("变化检测成果不完整，缺少：" + "；".join(missing))
        frame = (gpd.read_file(entry['event_geometry_audit'],layer='changes')
                 if entry.get('event_geometry_audit') else gpd.read_file(path))
        if frame.crs is None:
            raise ValueError(f"变化检测成果缺少 CRS：{path}")
        if frame.crs != analysis_crs:
            frame = frame.to_crs(analysis_crs)
        before_period, after_period = str(entry.get("before_period", "")), str(entry.get("after_period", ""))
        for source_index, source in frame.iterrows():
            geometry = source.geometry
            if geometry is None or geometry.is_empty:
                continue
            probe = geometry.boundary
            if probe is None or probe.is_empty:
                probe = geometry
            buffered = geometry.buffer(tolerance)
            indices = tree.query(buffered, predicate="intersects") if tree is not None else []
            ranked = []
            for raw_index in indices:
                index = int(raw_index)
                reference = reference_geometries[index]
                support = float(reference.intersection(buffered).length)
                ranked.append((support, -float(reference.distance(probe)), index))
            road_id = ""
            explicit_track=str(source.get('track_id') or '')
            explicit_index = reference_indices.get(explicit_track)
            if explicit_index is not None:
                ref_index=explicit_index
                road_id=explicit_track
            elif ranked:
                maximum_support=max(r[0] for r in ranked)
                consistent=[r for r in ranked if r[0]>=.5*maximum_support and
                    (references[r[2]]['road_id'],before_period,after_period,str(source.get('change_typ',''))) in event_lookup]
                _support, _distance, ref_index = max(consistent or ranked)
                road_id = references[ref_index]["road_id"]
            event_type = str(source.get("change_typ", ""))
            unreliable = event_type in {'widened','narrowed','width_changed'} and any(
                'unreliable_width' in str(observed.get((road_id,period),{}).get('qa_reason',''))
                for period in (before_period,after_period))
            if unreliable:event_type='uncertain'
            event_key = (road_id, before_period, after_period, event_type)
            if event_type == "width_changed" and event_key not in event_lookup and road_id:
                # The pair result intentionally carries no widened/narrowed direction.
                # If temporal widths produced a directional event, replace that label
                # with the authoritative generic class instead of inferring a direction.
                directional = next((
                    item for item in events
                    if item["road_id"] == road_id
                    and item["from_per"] == before_period
                    and item["to_per"] == after_period
                    and item["event_typ"] in {"widened", "narrowed"}
                ), None)
                if directional is not None:
                    old_key = (road_id, before_period, after_period, directional["event_typ"])
                    event_lookup.pop(old_key, None)
                    directional["event_typ"] = "width_changed"
                    event_lookup[event_key] = directional
                else:
                    event = {
                        "event_id": f"EV{len(events) + 1:08d}", "road_id": road_id,
                        "from_per": before_period, "to_per": after_period,
                        "event_typ": "width_changed", "before_st": "present", "after_st": "present",
                        "before_w": np.nan, "after_w": np.nan, "width_diff": np.nan,
                        "event_cf": 0.0, "evidence": "pair_change_only",
                        "qa_state": "auto", "geometry": references[ref_index]["geometry"],
                    }
                    events.append(event)
                    event_lookup[event_key] = event
            if event_key not in event_lookup and road_id:
                # A published pair change remains an event even when a coarse
                # whole-road observation disagrees. Keep the actual observed
                # states, automatically select the best road, and audit the
                # disagreement internally instead of requesting human review.
                before=observed.get((road_id,before_period),{})
                after=observed.get((road_id,after_period),{})
                b=before.get('width_m',np.nan);a=after.get('width_m',np.nan)
                event=dict(event_id=f'EV{len(events)+1:08d}',road_id=road_id,from_per=before_period,to_per=after_period,
                    event_typ=event_type,before_st=before.get('status','absent'),after_st=after.get('status','absent'),
                    before_w=b,after_w=a,width_diff=a-b,event_cf=min(before.get('match_sc',0.),after.get('match_sc',0.)),
                    evidence='unreliable_width' if unreliable else 'pair_class_observation_disagreement',qa_state='low_confidence',geometry=references[ref_index]['geometry'])
                events.append(event);event_lookup[event_key]=event
            for part_index, part in enumerate(_polygon_parts(geometry)):
                rows.append({
                    "event_id": event_lookup.get(event_key, {}).get("event_id", ""),
                    "road_id": road_id, "from_per": before_period, "to_per": after_period,
                    "event_typ": event_type,
                    "source_id": f"{source.get('change_id', source_index)}:{part_index}",
                    "area_m2": float(part.area), "qa_state": "auto" if road_id and not unreliable else "review",
                    "geometry": part,
                })
    return rows


def _apply_event_evidence(events: list[dict], event_parts: list[dict], has_pair_results: bool) -> None:
    supported = (event_parts.index(('event_id',)) if isinstance(event_parts,DiskRows) else
                 {(str(row.get('event_id','')),) for row in event_parts if row.get('event_id')})
    for event in events:
        if event["event_typ"] == "uncertain":
            continue
        if (event["event_id"],) in supported:
            if event.get('evidence')!='pair_class_observation_disagreement':event["evidence"] = "temporal_state_and_pair_change"
            continue
        event["evidence"] = "state_only_no_pair_part" if has_pair_results else "state_only_no_pair_input"
        # Reliable observed transitions do not require GT or a polygon fragment.
        # Ambiguous observations and lifecycle conflicts retain their own QA.


def _write_shp(path: Path, rows: list[dict], columns: dict[str, str], crs, geometry_type: str, output_crs=None) -> None:
    first=True
    for batch in batches(rows):
        _write_shp_batch(path,batch,columns,crs,geometry_type,output_crs,append=not first)
        first=False
    if first:_write_shp_batch(path,[],columns,crs,geometry_type,output_crs)


def _write_shp_batch(path, rows, columns, crs, geometry_type, output_crs=None, *, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        if geometry_type == 'LineString':
            frame.geometry = frame.geometry.map(_road_geometry)
            if frame.geometry.isna().any():
                raise ValueError(f'道路成果包含无有效线性部件的几何：{path}')
        for name, dtype in columns.items():
            if name not in frame.columns:
                frame[name] = "" if dtype == "str" else np.nan
        frame = frame[[*columns, "geometry"]]
    else:
        data = {name: pd.Series(dtype=dtype) for name, dtype in columns.items()}
        frame = gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    if output_crs is not None and frame.crs != output_crs:
        frame = frame.to_crs(output_crs)
    pyogrio.write_dataframe(
        frame, path, driver="ESRI Shapefile", encoding="UTF-8", geometry_type=geometry_type,
        append=append,
    )



@timed_stage("temporal")
@_geometry_scope
@record_scope
def build_temporal_grid(
    grid_name: str,
    period_entries: list[dict],
    change_entries: list[dict],
    output_dir: Path,
    tolerance: float = 3.0,
    width_absolute: float = 1.0,
    width_ratio: float = 0.1,
    *,
    reuse_observations: dict | None = None,
    progress=None,
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
    output_dir = Path(output_dir).resolve()
    periods = [str(entry['period']) for entry in period_entries]
    report = progress or (lambda message: None)
    if reuse_observations is not None:
        # Review changes only pair decisions. Existing road identities and
        # per-period observations remain authoritative and need no rematching.
        report('正在读取已有道路匹配结果…')
        life_frame = gpd.read_file(reuse_observations['life_shp']).to_crs(analysis_crs)
        references = life_frame[['road_id', 'geometry']].to_dict('records')
        observations = DiskRows()
        offset=0
        while True:
            batch=pyogrio.read_dataframe(reuse_observations['observations_shp'],skip_features=offset,max_features=2048).to_crs(analysis_crs)
            if batch.empty:break
            observations.extend(batch.to_dict('records'));offset+=len(batch)
        if not references or not observations.validate_observations([r['road_id'] for r in references],periods):
            raise ValueError('已有长时序道路记录与当前期次不一致，请先完成道路处理')
    else:
        report('正在建立各期道路匹配结果…')
        period_paths={str(entry['period']):path for entry,path in zip(period_entries,paths)}
        period_frames = LazyFrames(period_paths, lambda period:_read_period(period_paths[period], analysis_crs))
        reconciled_refs, reconciled_obs = _reconciled_tracks(period_entries, period_frames, analysis_crs)
        untracked_frames = (LazyFrames(period_frames, lambda p:period_frames[p].loc[period_frames[p].track_id.eq('')].copy())
                            if reconciled_refs else period_frames)
        references = _build_reference(untracked_frames, tolerance)
        references.extend(reconciled_refs)
        if not references:
            raise ValueError(f'{grid_name} 的所有期次均没有有效道路中心线')
        _assign_road_ids(references, output_dir / 'road_life.shp', analysis_crs, tolerance)
        reconciled_ids = {r['road_id'] for r in reconciled_refs}
        observations = _observations([r for r in references if r['road_id'] not in reconciled_ids],
                                     untracked_frames, tolerance)
        observations.extend(reconciled_obs)
    report('正在更新变化事件…')
    events = _events(observations, periods, width_absolute, width_ratio)
    event_parts = _event_parts(change_entries, references, events, analysis_crs, tolerance, observations)
    reviewed_pairs = {(str(e['before_period']), str(e['after_period']))
                      for e in change_entries if e.get('manual_review')}
    supported_events = (event_parts.index(('event_id',)) if isinstance(event_parts,DiskRows)
                        else {(part['event_id'],) for part in event_parts})
    # A reviewed pair is authoritative: do not recreate rejected changes from
    # the unchanged per-period road observations during temporal rebuilding.
    events = DiskRows((e for e in events if (e['from_per'], e['to_per']) not in reviewed_pairs
              or (e['event_id'],) in supported_events or e['event_typ'] == 'uncertain'),mutable=True)
    _apply_event_evidence(events, event_parts, bool(change_entries))
    report('正在更新道路变化统计…')
    life = _life_rows(references, observations, events, periods, tolerance)
    for row in life:
        row["grid_id"] = str(grid_name)[:80]
    reviews = DiskRows({
        "road_id": row["road_id"], "period": row["period"],
        "reason": row.get("qa_reason") or "ambiguous_or_weak_match",
        "match_sc": row["match_sc"], "coverage": row["coverage"], "qa_state": "review",
        "geometry": row["geometry"],
    } for row in observations if row["status"] == "uncertain")
    reviews.extend({
        "road_id": row["road_id"], "period": row["to_per"], "reason": row["evidence"],
        "match_sc": row["event_cf"], "coverage": 0.0, "qa_state": "review",
        "geometry": row["geometry"],
    } for row in events if row["qa_state"] == "review" and row["event_typ"] != "uncertain")

    report('正在写入长时序成果…')
    _write_shp(output_dir / "road_life.shp", life,
               {name: ("str" if name not in {"present_n", "event_n", "max_conf", "min_conf"} and not name.startswith(("W", "C")) else "float64")
                for name in life[0] if name != "geometry"}, analysis_crs, "LineString", output_crs)
    if reuse_observations is not None:
        from app.result_publisher import copy_dataset
        import shutil
        copy_dataset(reuse_observations['observations_shp'], output_dir / 'road_obs.shp')
        shutil.copy2(reuse_observations['width_evolution'], output_dir / 'width_evolution.csv')
    else:
        _write_shp(output_dir / "road_obs.shp", observations, {
            "road_id": "str", "period": "str", "status": "str", "width_m": "float64",
            "length_m": "float64", "coverage": "float64", "match_sc": "float64",
            "extract_cf": "float64", "geom_dev_m": "float64", "source_fid": "str", "qa_state": "str",
            "qa_reason": "str", "dir_sim": "float64",
        }, analysis_crs, "LineString", output_crs)
        with (output_dir/'width_evolution.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            first=True
            for batch in batches(observations):
                pd.DataFrame([{k:v for k,v in row.items() if k!='geometry'} for row in batch]).to_csv(
                    stream,index=False,header=first)
                first=False
    _write_shp(output_dir / "road_event.shp", events, {
        "event_id": "str", "road_id": "str", "from_per": "str", "to_per": "str",
        "event_typ": "str", "before_st": "str", "after_st": "str", "before_w": "float64",
        "after_w": "float64", "width_diff": "float64", "event_cf": "float64",
        "evidence": "str", "qa_state": "str",
    }, analysis_crs, "LineString", output_crs)
    _write_shp(output_dir / "event_parts.shp", event_parts, {
        "event_id": "str", "road_id": "str", "from_per": "str", "to_per": "str",
        "event_typ": "str", "source_id": "str", "area_m2": "float64", "qa_state": "str",
    }, analysis_crs, "Polygon", output_crs)
    _write_shp(output_dir / "road_lineage.shp", [], {
        "parent_id": "str", "child_id": "str", "period": "str", "relation": "str",
        "confidence": "float64", "qa_state": "str",
    }, analysis_crs, "LineString", output_crs)
    _write_shp(output_dir / "road_review.shp", reviews, {
        "road_id": "str", "period": "str", "reason": "str", "match_sc": "float64",
        "coverage": "float64", "qa_state": "str",
    }, analysis_crs, "LineString", output_crs)
    result = {
        "grid": grid_name, "output": str(output_dir),
        "life_shp": str(output_dir / "road_life.shp"),
        "observations_shp": str(output_dir / "road_obs.shp"),
        "events_shp": str(output_dir / "road_event.shp"),
        "event_parts_shp": str(output_dir / "event_parts.shp"),
        "lineage_shp": str(output_dir / "road_lineage.shp"),
        "review_shp": str(output_dir / "road_review.shp"),
        "width_evolution": str(output_dir / "width_evolution.csv"),
        "period_count": len(periods), "road_count": len(references),
        "observation_count": len(observations), "event_count": len(events),
        "review_count": 0,
        "qa_count": sum(row.get('qa_state')=='low_confidence' for row in observations),
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
