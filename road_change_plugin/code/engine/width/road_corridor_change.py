from __future__ import annotations

"""Canonical-axis, local-width road change classification."""

from collections import defaultdict
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree, line_merge, make_valid, union_all
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from road_pair_matcher import (
    MatchConfig,
    build_corridors,
    build_network_cover,
    build_width_segments,
    direction_similarity,
    match_road_segments,
)
from road_existence_evidence import (
    RoadProbabilityRaster,
    cross_period_change_decision,
    evidence_audit_fields,
    evaluate_road_existence_evidence,
)
from paired_width_profile import (
    PairedWidthConfig,
    PairedWidthProfile,
    candidate_change_runs,
    evaluate_change_run,
    measure_paired_width_profile,
    profile_debug_rows,
    robust_change_statistics,
)


def _parts(geometry: BaseGeometry, family: str) -> Iterable[BaseGeometry]:
    if geometry is None or geometry.is_empty:
        return
    geometry = make_valid(geometry)
    if (family == "line" and geometry.geom_type == "LineString") or (
        family == "polygon" and geometry.geom_type == "Polygon"
    ):
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _parts(part, family)


def _quality_score(row: pd.Series) -> float:
    grade = str(row.get("width_quality", row.get("quality_gr", "C")) or "C").upper()
    base = {"A": 3.0, "B": 2.0, "C": 1.0}.get(grade, 1.0)
    confidence = max(float(row.get("center_conf", 0.0) or 0.0), float(row.get("surface_conf", 0.0) or 0.0))
    source_bonus = {"manual": 0.5, "samroad": 0.25, "surface_skeleton": 0.0, "connector": -0.25}.get(
        str(row.get("line_source", "samroad")), 0.0,
    )
    return base + 0.25 * float(np.clip(confidence, 0.0, 1.0)) + source_bonus


def _interval_on(source: LineString, target: LineString, tolerance: float) -> tuple[float, float] | None:
    coordinates = list(target.coords)
    if not coordinates:
        return None
    measures = [float(source.project(Point(coordinate))) for coordinate in coordinates]
    start, end = max(0.0, min(measures)), min(float(source.length), max(measures))
    if end - start <= 1e-6:
        return None
    midpoint = source.interpolate((start + end) * 0.5)
    if midpoint.distance(target) > tolerance:
        return None
    return start, end


def _merge_intervals(intervals: list[tuple[float, float]], length: float, gap: float = 0.05) -> list[tuple[float, float]]:
    cleaned = sorted((max(0.0, start), min(length, end)) for start, end in intervals if end - start > 1e-6)
    merged: list[list[float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1] + gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _complement(intervals: list[tuple[float, float]], length: float) -> list[tuple[float, float]]:
    result = []
    cursor = 0.0
    for start, end in _merge_intervals(intervals, length):
        if start > cursor + 1e-6:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length - 1e-6:
        result.append((cursor, length))
    return result


def _oriented_subline(line: LineString, start_point: Point, end_point: Point) -> LineString | None:
    start = float(line.project(start_point))
    end = float(line.project(end_point))
    if abs(end - start) <= 1e-6:
        return None
    geometry = substring(line, min(start, end), max(start, end))
    if geometry.geom_type != "LineString" or geometry.is_empty:
        return None
    if start > end:
        geometry = LineString(list(geometry.coords)[::-1])
    return geometry


def _canonical_axis(
    before_line: LineString,
    after_line: LineString,
    before_row: pd.Series,
    after_row: pd.Series,
    tolerance: float,
) -> tuple[LineString, LineString, LineString] | None:
    before_interval = _interval_on(before_line, after_line, tolerance)
    after_interval = _interval_on(after_line, before_line, tolerance)
    if before_interval is None or after_interval is None:
        return None
    before_part = substring(before_line, *before_interval)
    after_part = _oriented_subline(after_line, Point(before_part.coords[0]), Point(before_part.coords[-1]))
    if after_part is None:
        return None
    # Reverse the second line when the opposite endpoint pairing is shorter.
    same = Point(before_part.coords[0]).distance(Point(after_part.coords[0])) + Point(before_part.coords[-1]).distance(Point(after_part.coords[-1]))
    reverse = Point(before_part.coords[0]).distance(Point(after_part.coords[-1])) + Point(before_part.coords[-1]).distance(Point(after_part.coords[0]))
    if reverse < same:
        after_part = LineString(list(after_part.coords)[::-1])
    sample_count = max(3, int(np.ceil(max(float(before_part.length), float(after_part.length)) / 2.0)) + 1)
    before_points = [before_part.interpolate(index / (sample_count - 1), normalized=True) for index in range(sample_count)]
    after_points = [after_part.interpolate(index / (sample_count - 1), normalized=True) for index in range(sample_count)]
    maximum_offset = max(first.distance(second) for first, second in zip(before_points, after_points))
    if maximum_offset > tolerance:
        return None
    before_quality = _quality_score(before_row)
    after_quality = _quality_score(after_row)
    if before_quality >= after_quality + 0.75:
        canonical = before_part
    elif after_quality >= before_quality + 0.75:
        canonical = after_part
    else:
        total = max(before_quality + after_quality, 1e-9)
        before_weight = before_quality / total
        coordinates = [
            (
                before_weight * first.x + (1.0 - before_weight) * second.x,
                before_weight * first.y + (1.0 - before_weight) * second.y,
            )
            for first, second in zip(before_points, after_points)
        ]
        canonical = LineString(coordinates)
    return canonical, before_part, after_part


def _surface_coverage(line: LineString, surface: BaseGeometry | None, tolerance: float) -> float:
    if surface is None or surface.is_empty or line.is_empty or float(line.length) <= 0:
        return 0.0
    support = surface.buffer(max(0.25, tolerance * 0.25))
    return float(np.clip(line.intersection(support).length / max(float(line.length), 1e-9), 0.0, 1.0))


def _empty_change(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({
        "change_typ": pd.Series(dtype="object"), "source_fid": pd.Series(dtype="object"),
        "src_period": pd.Series(dtype="object"), "length_m": pd.Series(dtype="float64"),
        "area_m2": pd.Series(dtype="float64"), "axis_len_m": pd.Series(dtype="float64"),
        "class_rule": pd.Series(dtype="object"), "before_w": pd.Series(dtype="float64"),
        "after_w": pd.Series(dtype="float64"), "width_diff": pd.Series(dtype="float64"),
        "qa_state": pd.Series(dtype="object"), "audit_reason": pd.Series(dtype="object"),
    }, geometry=[], crs=crs)


def _append_change(rows: list[dict], geometry: BaseGeometry, minimum_area: float, values: dict) -> None:
    for part_index, part in enumerate(_parts(geometry, "polygon")):
        if float(part.area) < minimum_area:
            continue
        rows.append({
            **values,
            "source_fid": f"{values.get('source_fid', 'feature')}:{part_index}",
            "area_m2": float(part.area),
            "geometry": part,
        })


def _line_components(rows: list[dict], tolerance: float = 0.5) -> list[list[int]]:
    if not rows:
        return []
    geometries = np.asarray([row["canonical"] for row in rows], dtype=object)
    tree = STRtree(geometries)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for index, geometry in enumerate(geometries):
        for other_value in tree.query(geometry, predicate="dwithin", distance=tolerance):
            other = int(other_value)
            if other == index or rows[index]["sign"] != rows[other]["sign"]:
                continue
            if direction_similarity(geometry, geometries[other]) < 0.75:
                continue
            adjacency[index].add(other)
            adjacency[other].add(index)
    result: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(rows)):
        if start in seen:
            continue
        stack = [start]
        component = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(adjacency.get(current, set()) - seen)
        result.append(component)
    return result


def detect_corridor_changes(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    config,
    before_period: str,
    after_period: str,
    before_surfaces: gpd.GeoDataFrame | None = None,
    after_surfaces: gpd.GeoDataFrame | None = None,
    before_width_segments: gpd.GeoDataFrame | None = None,
    after_width_segments: gpd.GeoDataFrame | None = None,
    before_valid: BaseGeometry | None = None,
    after_valid: BaseGeometry | None = None,
    before_probability: RoadProbabilityRaster | None = None,
    after_probability: RoadProbabilityRaster | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict, dict[str, gpd.GeoDataFrame]]:
    crs = before.crs
    before_segments = build_width_segments(before, before_width_segments)
    after_segments = build_width_segments(after, after_width_segments)
    before_corridors = build_corridors(before_segments, before_period)
    after_corridors = build_corridors(after_segments, after_period)
    matches, _before_links, _after_links, match_metadata = match_road_segments(
        before_segments,
        after_segments,
        MatchConfig(
            position_tolerance=config.position_tolerance,
            minimum_candidate_coverage=config.line_match_ratio,
            minimum_overlap_length=min(config.min_line_length, 3.0),
        ),
    )
    before_surface = (
        union_all(np.asarray(before_surfaces.geometry.values, dtype=object))
        if before_surfaces is not None and not before_surfaces.empty else None
    )
    after_surface = (
        union_all(np.asarray(after_surfaces.geometry.values, dtype=object))
        if after_surfaces is not None and not after_surfaces.empty else None
    )
    common_valid = None
    if before_valid is not None and after_valid is not None:
        common_valid = make_valid(before_valid.intersection(after_valid))
    elif before_valid is not None:
        common_valid = make_valid(before_valid)
    elif after_valid is not None:
        common_valid = make_valid(after_valid)
    before_network_cover = build_network_cover(before_segments, config.position_tolerance)
    after_network_cover = build_network_cover(after_segments, config.position_tolerance)

    reliable_pairs: list[dict] = []
    before_intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    after_intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for _, match in matches.sort_values("match_score", ascending=False).iterrows():
        before_id, after_id = int(match["before_id"]), int(match["after_id"])
        before_row, after_row = before_segments.iloc[before_id], after_segments.iloc[after_id]
        built = _canonical_axis(before_row.geometry, after_row.geometry, before_row, after_row, config.position_tolerance)
        if built is None:
            continue
        canonical, before_part, after_part = built
        before_interval = _interval_on(before_row.geometry, before_part, config.position_tolerance)
        after_interval = _interval_on(after_row.geometry, after_part, config.position_tolerance)
        if before_interval is None or after_interval is None:
            continue
        # Ignore overlap already claimed by a clearly better parallel/local pair.
        midpoint = (before_interval[0] + before_interval[1]) * 0.5
        already = any(start - 0.05 <= midpoint <= end + 0.05 for start, end in before_intervals[before_id])
        if already:
            continue
        before_intervals[before_id].append(before_interval)
        after_intervals[after_id].append(after_interval)
        reliable_pairs.append({
            "match_id": match.get("match_id", ""), "relation": match.get("relation", "partial"),
            "match_score": float(match.get("match_score", 0.0)),
            "before_id": before_id, "after_id": after_id,
            "before": before_row, "after": after_row,
            "before_part": before_part, "after_part": after_part, "canonical": canonical,
        })

    paired_config = PairedWidthConfig(
        sample_spacing=float(getattr(config, "paired_width_sample_spacing", 2.0)),
        normal_half_length=float(getattr(config, "paired_width_normal_half_length", 60.0)),
        absolute_change=float(config.width_change_absolute),
        relative_change=float(config.width_change_ratio),
        maximum_relative_change=float(config.width_change_max_ratio),
        minimum_valid_ratio=float(config.width_min_valid_ratio),
        minimum_continuous_length=float(config.width_min_overlap_length),
        minimum_samples=int(getattr(config, "paired_width_min_samples", 5)),
        minimum_direction_ratio=float(config.width_same_direction_ratio),
        maximum_diff_mad=float(getattr(config, "paired_width_max_mad", 1.0)),
        uncertainty_scale=float(getattr(config, "paired_width_uncertainty_scale", 2.5)),
        maximum_gap_samples=int(getattr(config, "paired_width_max_gap_samples", 1)),
        maximum_gap_length=float(getattr(config, "paired_width_max_gap_length", 4.0)),
        surface_probability_max_relative_difference=float(getattr(
            config, "paired_width_surface_probability_max_relative_difference", 0.30,
        )),
        probability_minimum_contrast=float(getattr(
            config, "paired_width_probability_minimum_contrast", 0.08,
        )),
        probability_minimum_confidence=float(getattr(
            config, "paired_width_probability_minimum_confidence", 0.55,
        )),
    )
    canonical_rows = []
    match_rows = []
    width_candidates: list[dict] = []
    paired_profiles: dict[str, PairedWidthProfile] = {}
    profile_decisions: dict[str, str] = {}
    paired_decision_rows: list[dict] = []
    rejected_quality: set[tuple[str, str]] = set()
    rejected_missing: set[tuple[str, str]] = set()
    rejected_excessive: set[tuple[str, str]] = set()
    rejected_low_valid: set[tuple[str, str]] = set()
    rejected_samples: set[tuple[str, str]] = set()
    rejected_mad: set[tuple[str, str]] = set()
    rejected_uncertainty: set[tuple[str, str]] = set()
    review_missing_surface: set[tuple[str, str]] = set()
    review_surface_probability_disagreement: set[tuple[str, str]] = set()
    for pair_index, pair in enumerate(reliable_pairs):
        before_row, after_row = pair["before"], pair["after"]
        legacy_before_width = float(before_row.get("width_m", 0.0) or 0.0)
        legacy_after_width = float(after_row.get("width_m", 0.0) or 0.0)
        canonical_id = f"C{pair_index + 1:08d}"
        pair["canonical_id"] = canonical_id
        qa_state = "review" if "review" in {
            str(before_row.get("qa_state", "")), str(after_row.get("qa_state", "")),
        } else "auto"
        rejection_key = (
            str(before_row.get("parent_id", pair["before_id"])),
            str(after_row.get("parent_id", pair["after_id"])),
        )
        profile = measure_paired_width_profile(
            canonical_id,
            pair["canonical"],
            pair["before_part"],
            pair["after_part"],
            before_surface,
            after_surface,
            paired_config,
            before_probability=before_probability,
            after_probability=after_probability,
            geometry_crs=crs,
        )
        paired_profiles[canonical_id] = profile
        profile_stats = robust_change_statistics(profile.valid_samples)
        measured_before = float(profile_stats["before_width"])
        measured_after = float(profile_stats["after_width"])
        before_width = measured_before if int(profile_stats["sample_count"]) else legacy_before_width
        after_width = measured_after if int(profile_stats["sample_count"]) else legacy_after_width
        width_delta = after_width - before_width
        common = {
            "canonical_id": canonical_id, "match_id": pair["match_id"], "relation": pair["relation"],
            "before_seg": str(before_row.get("segment_id", pair["before_id"])),
            "after_seg": str(after_row.get("segment_id", pair["after_id"])),
            "before_w": before_width, "after_w": after_width, "width_diff": width_delta,
            "match_score": pair["match_score"], "qa_state": qa_state,
            "valid_ratio": profile.valid_ratio,
            "sample_count": int(profile_stats["sample_count"]),
            "width_mad": float(profile_stats["diff_mad"]),
            "uncertainty": float(profile_stats["uncertainty"]),
        }
        canonical_rows.append({**common, "length_m": float(pair["canonical"].length), "geometry": pair["canonical"]})
        match_rows.append({
            **common,
            "before_cov": float(pair["before_part"].length) / max(float(before_row.geometry.length), 1e-9),
            "after_cov": float(pair["after_part"].length) / max(float(after_row.geometry.length), 1e-9),
            "geometry": pair["canonical"],
        })
        grade_ok = (
            str(before_row.get("width_quality", "C")) != "C"
            and str(after_row.get("width_quality", "C")) != "C"
        )
        runs = candidate_change_runs(profile, paired_config)
        if (before_surface is None or after_surface is None) and not profile.valid_samples:
            legacy_delta = legacy_after_width - legacy_before_width
            legacy_relative = abs(legacy_delta) / max(legacy_before_width, legacy_after_width, 0.1)
            legacy_threshold = (
                legacy_before_width > 0 and legacy_after_width > 0
                and abs(legacy_delta) >= paired_config.absolute_change
                and legacy_relative >= paired_config.relative_change
            )
            if legacy_threshold and legacy_relative > paired_config.maximum_relative_change:
                rejected_excessive.add(rejection_key)
                profile_decisions[canonical_id] = "no_change:paired_width_relative_change_excessive"
            elif legacy_threshold and not grade_ok:
                rejected_quality.add(rejection_key)
                profile_decisions[canonical_id] = "no_change:paired_width_quality_rejected"
            elif legacy_threshold:
                review_missing_surface.add(rejection_key)
                profile_decisions[canonical_id] = "review:paired_width_surface_missing"
                width_candidates.append({
                    **common,
                    "canonical": pair["canonical"],
                    "sign": 1 if legacy_delta > 0 else -1,
                    "before_w": legacy_before_width,
                    "after_w": legacy_after_width,
                    "width_diff": legacy_delta,
                    "samples": tuple(),
                    "valid_ratio": 0.0,
                    "fallback_review": True,
                    "rejection_key": rejection_key,
                    "qa_state": "review",
                })
            else:
                if legacy_before_width <= 0 or legacy_after_width <= 0:
                    rejected_missing.add(rejection_key)
                profile_decisions[canonical_id] = "no_change:paired_width_surface_missing"
            continue
        if runs and not grade_ok:
            rejected_quality.add(rejection_key)
            profile_decisions[canonical_id] = "no_change:paired_width_quality_rejected"
            continue
        if not runs:
            if any(sample.surface_probability_disagreement for sample in profile.samples):
                review_surface_probability_disagreement.add(rejection_key)
                profile_decisions[canonical_id] = "review:surface_probability_disagreement"
            elif profile.valid_ratio < paired_config.minimum_valid_ratio:
                rejected_low_valid.add(rejection_key)
                profile_decisions[canonical_id] = "no_change:paired_width_valid_ratio_too_low"
            else:
                profile_decisions[canonical_id] = "no_change"
            continue
        profile_decisions[canonical_id] = "candidate"
        for run in runs:
            width_candidates.append({
                **common,
                "canonical": run.axis,
                "sign": run.sign,
                "before_w": run.before_width,
                "after_w": run.after_width,
                "width_diff": run.width_diff,
                "samples": run.samples,
                "valid_ratio": run.valid_ratio,
                "fallback_review": False,
                "rejection_key": rejection_key,
                "qa_state": qa_state,
            })

    positive_rows: list[dict] = []
    negative_rows: list[dict] = []
    width_changed_count = 0
    width_rejected_short = 0
    for component_index, component in enumerate(
        _line_components(width_candidates, tolerance=max(0.5, config.position_tolerance * 0.25)),
        start=1,
    ):
        selected = [width_candidates[index] for index in component]
        component_axis = union_all(np.asarray([row["canonical"] for row in selected], dtype=object))
        axis_length = float(component_axis.length)
        fallback_review = any(bool(row.get("fallback_review")) for row in selected)
        samples = tuple(sample for row in selected for sample in row.get("samples", tuple()))
        valid_ratio = (
            sum(sample.valid for sample in samples) / len(samples)
            if samples else 0.0
        )
        if fallback_review:
            before_width = float(np.median([row["before_w"] for row in selected]))
            after_width = float(np.median([row["after_w"] for row in selected]))
            width_diff = after_width - before_width
            stats = {
                "accepted": axis_length >= paired_config.minimum_continuous_length,
                "decision": "review" if axis_length >= paired_config.minimum_continuous_length else "no_change",
                "reject_reason": "" if axis_length >= paired_config.minimum_continuous_length else "paired_width_continuous_length_too_short",
                "before_width": before_width, "after_width": after_width, "width_diff": width_diff,
                "diff_mad": 0.0, "uncertainty": 0.0, "direction_ratio": 1.0,
                "sample_count": 0, "valid_ratio": 0.0, "relative_change": abs(width_diff) / max(before_width, after_width, 0.1),
            }
        else:
            stats = evaluate_change_run(
                samples,
                axis_length=axis_length,
                valid_ratio=valid_ratio,
                config=paired_config,
            )
        accepted = bool(stats["accepted"])
        reject_reason = str(stats["reject_reason"])
        if not accepted:
            if reject_reason == "paired_width_continuous_length_too_short":
                width_rejected_short += 1
            elif reject_reason == "paired_width_sample_count_too_low":
                rejected_samples.update(row["rejection_key"] for row in selected)
            elif reject_reason == "paired_width_relative_change_excessive":
                rejected_excessive.update(row["rejection_key"] for row in selected)
            elif reject_reason == "paired_width_valid_ratio_too_low":
                rejected_low_valid.update(row["rejection_key"] for row in selected)
            elif reject_reason == "paired_width_mad_too_large":
                rejected_mad.update(row["rejection_key"] for row in selected)
            elif reject_reason == "paired_width_uncertainty_too_large":
                rejected_uncertainty.update(row["rejection_key"] for row in selected)
            for row in selected:
                profile_decisions[row["canonical_id"]] = f"no_change:{reject_reason}"
        qa_value = "review" if fallback_review or any(row["qa_state"] == "review" for row in selected) else "auto"
        evidence_fields = {}
        if accepted and not fallback_review:
            before_evidence = evaluate_road_existence_evidence(
                component_axis, centerline_cover=before_network_cover,
                road_surface=before_surface, valid_area=before_valid,
                probability=before_probability, crs=crs, road_width=float(stats["before_width"]),
                position_tolerance=config.position_tolerance,
                allow_legacy_absence_without_valid_mask=bool(getattr(
                    config, "allow_legacy_absence_without_valid_mask", False
                )),
            )
            after_evidence = evaluate_road_existence_evidence(
                component_axis, centerline_cover=after_network_cover,
                road_surface=after_surface, valid_area=after_valid,
                probability=after_probability, crs=crs, road_width=float(stats["after_width"]),
                position_tolerance=config.position_tolerance,
                allow_legacy_absence_without_valid_mask=bool(getattr(
                    config, "allow_legacy_absence_without_valid_mask", False
                )),
            )
            if before_evidence.existence_state != "present" or after_evidence.existence_state != "present":
                qa_value = "review"
            evidence_fields = evidence_audit_fields(before_evidence, after_evidence)
        paired_decision_rows.append({
            "decision_id": f"PW{component_index:08d}",
            "canonical_id": ",".join(sorted({row["canonical_id"] for row in selected})),
            "before_width": float(stats["before_width"]),
            "after_width": float(stats["after_width"]),
            "width_diff": float(stats["width_diff"]),
            "mad": float(stats["diff_mad"]),
            "uncertainty": float(stats["uncertainty"]),
            "valid_ratio": float(stats["valid_ratio"]),
            "sample_count": int(stats["sample_count"]),
            "direction_ratio": float(stats["direction_ratio"]),
            "axis_length_m": axis_length,
            "change_decision": str(stats["decision"]),
            "reject_reason": reject_reason,
            "qa_state": qa_value if accepted else "",
            "geometry": component_axis,
        })
        if not accepted:
            continue
        before_buffers = [
            row["canonical"].buffer(row["before_w"] * 0.5, cap_style="flat", join_style="round")
            for row in selected
        ]
        after_buffers = [
            row["canonical"].buffer(row["after_w"] * 0.5, cap_style="flat", join_style="round")
            for row in selected
        ]
        before_union = union_all(np.asarray(before_buffers, dtype=object))
        after_union = union_all(np.asarray(after_buffers, dtype=object))
        widened = float(stats["width_diff"]) > 0
        geometry = after_union.difference(before_union) if widened else before_union.difference(after_union)
        target_rows = positive_rows if widened else negative_rows
        change_type = "widened" if widened else "narrowed"
        audit_reason = (
            "paired_width_surface_missing_review" if fallback_review
            else "paired_width_change_thresholds_met"
        )
        values = {
            "change_typ": change_type, "source_fid": selected[0]["canonical_id"],
            "src_period": after_period if widened else before_period,
            "length_m": 0.0, "axis_len_m": axis_length,
            "class_rule": "canonical_axis_paired_normal_width_buffer_difference",
            "before_w": float(stats["before_width"]),
            "after_w": float(stats["after_width"]),
            "width_diff": float(stats["width_diff"]),
            "width_mad": float(stats["diff_mad"]),
            "uncertainty": float(stats["uncertainty"]),
            "valid_ratio": float(stats["valid_ratio"]),
            "sample_count": int(stats["sample_count"]),
            "same_dir_ratio": float(stats["direction_ratio"]),
            "qa_state": qa_value,
            "audit_reason": audit_reason,
            **evidence_fields,
        }
        rows_before = len(target_rows)
        _append_change(
            target_rows,
            geometry,
            max(config.min_polygon_area, config.width_min_polygon_area),
            values,
        )
        if len(target_rows) > rows_before and qa_value == "auto":
            width_changed_count += 1
        for row in selected:
            profile_decisions[row["canonical_id"]] = f"{qa_value}:{change_type}"

    presence_confirmed = presence_review = 0
    audit_counts = defaultdict(int)
    raw_unmatched_length = 0.0

    def append_uncovered(
        source_segments: gpd.GeoDataFrame,
        source_network_cover: BaseGeometry | None,
        reference_network_cover: BaseGeometry | None,
        source_surface: BaseGeometry | None,
        reference_surface: BaseGeometry | None,
        source_valid: BaseGeometry | None,
        reference_valid: BaseGeometry | None,
        source_probability: RoadProbabilityRaster | None,
        reference_probability: RoadProbabilityRaster | None,
        change_type: str,
        period: str,
        target: list[dict],
    ) -> None:
        nonlocal presence_confirmed, presence_review, raw_unmatched_length
        source_lines = [geometry for geometry in source_segments.geometry if not geometry.is_empty]
        if not source_lines:
            return
        source_network = line_merge(union_all(np.asarray(source_lines, dtype=object)))
        uncovered = (
            source_network if reference_network_cover is None
            else source_network.difference(reference_network_cover)
        )
        uncovered = line_merge(uncovered)
        for part_index, line_part in enumerate(_parts(uncovered, "line")):
            if float(line_part.length) < config.min_line_length:
                continue
            raw_unmatched_length += float(line_part.length)
            contributors = source_segments.loc[
                source_segments.geometry.map(
                    lambda geometry: float(geometry.intersection(line_part.buffer(1e-6)).length) > 1e-6
                )
            ]
            if contributors.empty:
                continue
            widths = [
                float(value) for value in contributors.get("width_m", pd.Series(dtype="float64"))
                if pd.notna(value) and float(value) > 0
            ]
            width = float(np.median(widths)) if widths else max(4.0, 2.0 * config.position_tolerance)
            qa_values = {str(value) for value in contributors.get("qa_state", pd.Series(dtype="object"))}
            quality_values = {
                str(value) for value in contributors.get("width_quality", pd.Series(dtype="object"))
            }
            source_values = {
                str(value) for value in contributors.get("line_source", pd.Series(dtype="object"))
            }
            corridor = line_part.buffer(width * 0.5, cap_style="flat", join_style="round")
            source_evidence = evaluate_road_existence_evidence(
                line_part, centerline_cover=source_network_cover, road_surface=source_surface,
                valid_area=source_valid, probability=source_probability, crs=crs,
                road_width=width, position_tolerance=config.position_tolerance,
                allow_legacy_absence_without_valid_mask=bool(getattr(
                    config, "allow_legacy_absence_without_valid_mask", False
                )),
            )
            reference_evidence = evaluate_road_existence_evidence(
                line_part, centerline_cover=reference_network_cover, road_surface=reference_surface,
                valid_area=reference_valid, probability=reference_probability, crs=crs,
                road_width=width, position_tolerance=config.position_tolerance,
                allow_legacy_absence_without_valid_mask=bool(getattr(
                    config, "allow_legacy_absence_without_valid_mask", False
                )),
            )
            before_evidence, after_evidence = (
                (reference_evidence, source_evidence)
                if change_type == "added" else (source_evidence, reference_evidence)
            )
            qa_state, audit_reason = cross_period_change_decision(
                before_evidence, after_evidence, change_type,
            )
            reliable_source = "review" not in qa_values and "connector" not in source_values
            if not reliable_source:
                qa_state = "review"
                audit_reason = "geometry_mismatch_only"
            confirmed = qa_state == "auto"
            presence_confirmed += int(confirmed)
            presence_review += int(not confirmed)
            audit_counts["geometry_only_added_candidates" if change_type == "added" else "geometry_only_removed_candidates"] += 1
            audit_counts["confirmed_absent_count"] += int(
                before_evidence.existence_state == "absent" or after_evidence.existence_state == "absent"
            )
            audit_counts["uncertain_count"] += int(
                "uncertain" in {before_evidence.existence_state, after_evidence.existence_state}
            )
            audit_counts["present_by_probability_count"] += int(
                "probability_present" in before_evidence.existence_reason
                or "probability_present" in after_evidence.existence_reason
            )
            audit_counts["present_by_surface_count"] += int(
                "surface_present" in before_evidence.existence_reason
                or "surface_present" in after_evidence.existence_reason
            )
            audit_counts["suppressed_extraction_disagreement_count"] += int(
                not confirmed and (
                    reference_evidence.existence_state == "present"
                    or reference_evidence.existence_state == "uncertain"
                )
            )
            source_ids = sorted(str(value) for value in contributors.get("segment_id", contributors.index))
            _append_change(target, corridor, config.min_polygon_area, {
                "change_typ": change_type,
                "source_fid": f"{','.join(source_ids)}:{part_index}",
                "src_period": period, "length_m": float(line_part.length),
                "axis_len_m": float(line_part.length),
                "class_rule": "whole_reference_network_uncovered_segment_buffer",
                "before_w": width if change_type == "removed" else 0.0,
                "after_w": width if change_type == "added" else 0.0,
                "width_diff": width if change_type == "added" else -width,
                "qa_state": qa_state,
                "audit_reason": audit_reason,
                "line_source": "connector" if "connector" in source_values else "samroad",
                "quality_gr": max(quality_values) if quality_values else "C",
                **evidence_audit_fields(before_evidence, after_evidence),
            })

    append_uncovered(
        after_segments, after_network_cover, before_network_cover, after_surface, before_surface,
        after_valid, before_valid, after_probability, before_probability,
        "added", after_period, positive_rows,
    )
    append_uncovered(
        before_segments, before_network_cover, after_network_cover, before_surface, after_surface,
        before_valid, after_valid, before_probability, after_probability,
        "removed", before_period, negative_rows,
    )

    positive = gpd.GeoDataFrame(positive_rows, geometry="geometry", crs=crs) if positive_rows else _empty_change(crs)
    negative = gpd.GeoDataFrame(negative_rows, geometry="geometry", crs=crs) if negative_rows else _empty_change(crs)
    unchanged_parts = []
    for pair in reliable_pairs:
        before_width = float(pair["before"].get("width_m", 0.0) or 0.0)
        after_width = float(pair["after"].get("width_m", 0.0) or 0.0)
        if before_width <= 0 or after_width <= 0:
            continue
        before_buffer = pair["canonical"].buffer(before_width * 0.5, cap_style="flat", join_style="round")
        after_buffer = pair["canonical"].buffer(after_width * 0.5, cap_style="flat", join_style="round")
        unchanged_parts.extend(_parts(before_buffer.intersection(after_buffer), "polygon"))
    unchanged = (
        gpd.GeoDataFrame(
            [{"area_m2": float(part.area), "geometry": part} for part in unchanged_parts],
            geometry="geometry", crs=crs,
        )
        if unchanged_parts else gpd.GeoDataFrame({"area_m2": pd.Series(dtype="float64")}, geometry=[], crs=crs)
    )
    canonical = (
        gpd.GeoDataFrame(canonical_rows, geometry="geometry", crs=crs)
        if canonical_rows else gpd.GeoDataFrame({"canonical_id": pd.Series(dtype="object")}, geometry=[], crs=crs)
    )
    unmatched_counter = 0
    for side, segments, covered in (
        ("before", before_segments, before_intervals),
        ("after", after_segments, after_intervals),
    ):
        for segment_index, row in segments.iterrows():
            for start, end in _complement(covered.get(int(segment_index), []), float(row.geometry.length)):
                geometry = substring(row.geometry, start, end)
                if geometry.is_empty or float(geometry.length) < config.min_line_length:
                    continue
                unmatched_counter += 1
                match_rows.append({
                    "match_id": f"U{unmatched_counter:08d}", "relation": "unmatched",
                    "match_side": side, "match_score": 0.0,
                    "before_seg": str(row.get("segment_id", segment_index)) if side == "before" else "",
                    "after_seg": str(row.get("segment_id", segment_index)) if side == "after" else "",
                    "before_w": float(row.get("width_m", 0.0) or 0.0) if side == "before" else 0.0,
                    "after_w": float(row.get("width_m", 0.0) or 0.0) if side == "after" else 0.0,
                    "qa_state": row.get("qa_state", "review"),
                    "geometry": geometry,
                })
    match_output = (
        gpd.GeoDataFrame(match_rows, geometry="geometry", crs=crs)
        if match_rows else gpd.GeoDataFrame({"match_id": pd.Series(dtype="object")}, geometry=[], crs=crs)
    )
    period_segments = gpd.GeoDataFrame(pd.concat([
        before_segments.assign(period=before_period), after_segments.assign(period=after_period),
    ], ignore_index=True), geometry="geometry", crs=crs)
    period_corridors = gpd.GeoDataFrame(pd.concat([before_corridors, after_corridors], ignore_index=True), geometry="geometry", crs=crs)
    paired_sample_rows = []
    for canonical_id, profile in paired_profiles.items():
        paired_sample_rows.extend(profile_debug_rows(
            profile,
            profile_decisions.get(canonical_id, "no_change"),
        ))
    paired_samples = (
        gpd.GeoDataFrame(paired_sample_rows, geometry="geometry", crs=crs)
        if paired_sample_rows else gpd.GeoDataFrame({
            "canonical_id": pd.Series(dtype="object"),
            "sample_position_m": pd.Series(dtype="float64"),
            "valid": pd.Series(dtype="bool"),
            "reject_reason": pd.Series(dtype="object"),
            "change_decision": pd.Series(dtype="object"),
        }, geometry=[], crs=crs)
    )
    paired_decisions = (
        gpd.GeoDataFrame(paired_decision_rows, geometry="geometry", crs=crs)
        if paired_decision_rows else gpd.GeoDataFrame({
            "canonical_id": pd.Series(dtype="object"),
            "change_decision": pd.Series(dtype="object"),
            "reject_reason": pd.Series(dtype="object"),
        }, geometry=[], crs=crs)
    )
    summary = {
        "classification_method": "two_stage_geometry_candidate_then_symmetric_existence_evidence",
        "width_measurement_method": "canonical_axis_cross_period_paired_normals",
        "matched_centerline_count": len(reliable_pairs),
        "width_changed_centerline_count": width_changed_count,
        "width_change_absolute_m": config.width_change_absolute,
        "width_change_ratio": config.width_change_ratio,
        "width_change_max_ratio": config.width_change_max_ratio,
        "width_min_overlap_length_m": config.width_min_overlap_length,
        "width_min_valid_ratio": config.width_min_valid_ratio,
        "width_same_direction_ratio": config.width_same_direction_ratio,
        "paired_width_sample_spacing_m": paired_config.sample_spacing,
        "paired_width_min_samples": paired_config.minimum_samples,
        "paired_width_max_mad_m": paired_config.maximum_diff_mad,
        "paired_width_uncertainty_scale": paired_config.uncertainty_scale,
        "paired_width_max_gap_samples": paired_config.maximum_gap_samples,
        "paired_width_max_gap_length_m": paired_config.maximum_gap_length,
        "paired_width_surface_probability_max_relative_difference": (
            paired_config.surface_probability_max_relative_difference
        ),
        "paired_width_sample_count": len(paired_samples),
        "paired_width_valid_sample_count": int(paired_samples["valid"].sum()) if not paired_samples.empty else 0,
        "width_rejected_match_count": max(0, len(matches) - len(reliable_pairs)),
        "width_rejected_short_count": width_rejected_short,
        "width_rejected_quality_count": len(rejected_quality),
        "width_rejected_missing_count": len(rejected_missing),
        "width_rejected_excessive_count": len(rejected_excessive),
        "width_rejected_low_valid_ratio_count": len(rejected_low_valid),
        "width_rejected_min_samples_count": len(rejected_samples),
        "width_rejected_mad_count": len(rejected_mad),
        "width_rejected_uncertainty_count": len(rejected_uncertainty),
        "width_review_missing_surface_count": len(review_missing_surface),
        "width_review_surface_probability_disagreement_count": len(
            review_surface_probability_disagreement
        ),
        "presence_confirmed_count": presence_confirmed,
        "presence_review_count": presence_review,
        "presence_suppressed_low_confidence_count": 0,
        "raw_unmatched_length": raw_unmatched_length,
        **dict(audit_counts),
        "valid_observation_intersection_applied": common_valid is not None,
        "actual_surface_source": before_surface is not None and after_surface is not None,
        **match_metadata,
    }
    artifacts = {
        "road_width_segments": period_segments,
        "road_corridors": period_corridors,
        "road_matches": match_output,
        "canonical_roads": canonical,
        "paired_width_samples": paired_samples,
        "paired_width_decisions": paired_decisions,
    }
    return positive, negative, unchanged, summary, artifacts
