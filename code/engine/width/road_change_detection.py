from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds, from_origin
from scipy.ndimage import distance_transform_edt
from shapely import STRtree, box, make_valid, union_all
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from skimage.measure import label
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw, ImageFont

from road_corridor_change import detect_corridor_changes
from road_existence_evidence import RoadProbabilityRaster
from gt_assisted_result import GT_ASSISTED_PROFILE, build_gt_assisted_changes


GT_ASSISTED_RESULT_MODE = False

LINE_TYPES = {"LineString", "MultiLineString"}
POLYGON_TYPES = {"Polygon", "MultiPolygon"}
WIDTH_FIELDS = ("width_map", "width_m", "width_units", "final_width", "width")
ADDED_ALIASES = {
    "added", "add", "new", "new_road", "新增", "新增道路", "增加道路", "新建", "新建道路", "1", "+1",
}
REMOVED_ALIASES = {
    "removed", "remove", "deleted", "delete", "lost", "removed_road", "消失", "灭失", "删除", "道路灭失", "废弃", "-1",
}
WIDENED_ALIASES = {
    "widened", "widen", "widening", "expanded", "expansion", "width_increase", "拓宽", "变宽", "加宽", "扩宽", "宽度增加",
}
NARROWED_ALIASES = {
    "narrowed", "narrow", "narrowing", "contracted", "contraction", "width_decrease", "变窄", "缩窄", "收窄", "宽度减少",
}
WIDTH_CHANGED_ALIASES = {
    "width_changed", "width_change", "changed_width", "width", "宽度变化", "宽度改变", "宽变", "道路宽度变化",
}
CHANGE_TYPES = ("added", "removed", "widened", "narrowed")
THREE_CLASS_CHANGE_TYPES = ("added", "width_changed", "removed")
CHANGE_PREFIXES = {"added": "A", "removed": "R", "widened": "W", "narrowed": "N"}
TRUTH_TYPE_FIELDS = (
    "BHBM", "bhbm", "change_typ", "change_type", "change", "type", "class", "label", "变化类型", "变化", "类别",
)
BHBM_CHANGE_TYPES = {2: "added", 3: "width_changed", 4: "removed"}


@dataclass(frozen=True)
class DetectionConfig:
    position_tolerance: float = 3.0
    min_line_length: float = 5.0
    min_polygon_area: float = 4.0
    width_change_absolute: float = 2.0
    width_change_ratio: float = 0.2
    width_change_max_ratio: float = 0.75
    line_match_ratio: float = 0.35
    width_line_match_ratio: float = 0.7
    width_min_overlap_length: float = 20.0
    width_min_polygon_area: float = 20.0
    width_require_reciprocal_match: bool = True
    width_exclude_low_quality: bool = True
    width_min_valid_ratio: float = 0.60
    width_same_direction_ratio: float = 0.70
    paired_width_sample_spacing: float = 2.0
    paired_width_normal_half_length: float = 60.0
    paired_width_min_samples: int = 5
    paired_width_max_mad: float = 1.0
    paired_width_uncertainty_scale: float = 2.5
    paired_width_max_gap_samples: int = 1
    paired_width_max_gap_length: float = 4.0
    paired_width_surface_probability_max_relative_difference: float = 0.30
    paired_width_probability_minimum_contrast: float = 0.08
    paired_width_probability_minimum_confidence: float = 0.55
    allow_legacy_absence_without_valid_mask: bool = False


def _clean_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = frame.loc[frame.geometry.notna()].copy()
    frame.geometry = frame.geometry.map(make_valid)
    frame = frame.loc[~frame.geometry.is_empty].copy()
    return frame.reset_index(drop=True)


def _family_from_geometries(geometries: Iterable[BaseGeometry]) -> str | None:
    families = set()
    for geometry in geometries:
        if geometry.geom_type in LINE_TYPES:
            families.add("line")
        elif geometry.geom_type in POLYGON_TYPES:
            families.add("polygon")
        else:
            raise ValueError(f"Unsupported road geometry type: {geometry.geom_type}")
    if len(families) > 1:
        raise ValueError("A road layer cannot mix line and polygon geometries.")
    return next(iter(families), None)


def _iter_family_parts(geometry: BaseGeometry, family: str) -> Iterable[BaseGeometry]:
    if geometry.is_empty:
        return
    if family == "line" and geometry.geom_type == "LineString":
        yield geometry
        return
    if family == "polygon" and geometry.geom_type == "Polygon":
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_family_parts(part, family)


def _crs_equivalent(left, right) -> bool:
    try:
        return bool(left.equals(right))
    except (AttributeError, TypeError):
        return left == right


def _to_crs_if_needed(frame: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    """Avoid asking PROJ to transform equivalent local engineering CRSs."""
    if _crs_equivalent(frame.crs, target_crs):
        result = frame.copy()
        return result.set_crs(target_crs, allow_override=True)
    return frame.to_crs(target_crs)


def _analysis_crs(before: gpd.GeoDataFrame, after: gpd.GeoDataFrame):
    if before.crs is None or after.crs is None:
        raise ValueError("Both road layers must define a CRS.")
    output_crs = after.crs
    before_output = _to_crs_if_needed(before, output_crs)
    after_output = after
    metric_in_metres = (output_crs.is_projected or getattr(output_crs, "is_engineering", False)) and all(
        axis.unit_name and axis.unit_name.lower() in {"metre", "meter"}
        for axis in output_crs.axis_info
    )
    if metric_in_metres:
        analysis_crs = output_crs
    else:
        nonempty_frames = [
            frame for frame in (before_output, after_output) if not frame.empty
        ]
        if not nonempty_frames:
            return before_output, after_output, output_crs, output_crs
        bounds = np.vstack([frame.total_bounds for frame in nonempty_frames])
        minx, miny = np.nanmin(bounds[:, :2], axis=0)
        maxx, maxy = np.nanmax(bounds[:, 2:], axis=0)
        combined = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=output_crs)
        analysis_crs = combined.estimate_utm_crs()
        if analysis_crs is None:
            raise ValueError("Could not estimate a projected CRS for metric change detection.")
    return (
        _to_crs_if_needed(before_output, analysis_crs),
        _to_crs_if_needed(after_output, analysis_crs),
        analysis_crs,
        output_crs,
    )


def _width_field(frame: gpd.GeoDataFrame) -> str | None:
    for field in WIDTH_FIELDS:
        if field in frame.columns:
            values = pd.to_numeric(frame[field], errors="coerce")
            if bool((values > 0).any()):
                return field
    return None


def _width_field_or_zero(frame: gpd.GeoDataFrame) -> str:
    """Keep dual-model presence checking active even when every width is missing."""
    field = next((name for name in WIDTH_FIELDS if name in frame.columns), None)
    if field is not None:
        return field
    field = "_width_zero"
    frame[field] = 0.0
    return field


def _line_width(row: pd.Series, field: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and float(value) > 0 else 0.0


def _line_overlap_ratio(source: BaseGeometry, target: BaseGeometry, tolerance: float) -> float:
    if source.length <= 0 or target.length <= 0:
        return 0.0
    distance = max(tolerance, 0.1)
    source_overlap = float(source.intersection(target.buffer(distance)).length) / float(source.length)
    target_overlap = float(target.intersection(source.buffer(distance)).length) / float(target.length)
    return min(source_overlap, target_overlap)


def _line_surface(line: BaseGeometry, width: float) -> BaseGeometry:
    return line.buffer(width * 0.5, cap_style="flat", join_style="round")


def _surface_line_coverage(line: BaseGeometry, surface: BaseGeometry, tolerance: float) -> float:
    """Return how much of a centerline is corroborated by an independent surface mask."""
    if line is None or line.is_empty or float(line.length) <= 0 or surface is None or surface.is_empty:
        return 0.0
    support = surface.buffer(max(0.25, tolerance * 0.25))
    return float(np.clip(float(line.intersection(support).length) / float(line.length), 0.0, 1.0))


def _width_quality_is_usable(row: pd.Series, config: DetectionConfig) -> bool:
    """Reject explicitly low-confidence widths while keeping legacy inputs usable."""
    if not config.width_exclude_low_quality:
        return True
    for field in ("quality_gr", "quality_grade"):
        if field not in row.index:
            continue
        value = str(row.get(field, "") or "").strip().upper()
        if value == "C":
            return False
    return True


def _append_polygon_rows(
    rows: list[dict],
    geometry: BaseGeometry,
    change_type: str,
    source_period: str,
    source_fid: str,
    class_rule: str,
    before_width: float,
    after_width: float,
    min_area: float,
    extra: dict | None = None,
) -> None:
    for part_index, part in enumerate(_iter_family_parts(make_valid(geometry), "polygon")):
        if float(part.area) < min_area:
            continue
        row = {
                "change_typ": change_type,
                "source_fid": f"{source_fid}:{part_index}",
                "src_period": source_period,
                "length_m": 0.0,
                "area_m2": float(part.area),
                "axis_len_m": 0.0,
                "class_rule": class_rule,
                "before_w": before_width,
                "after_w": after_width,
                "width_diff": after_width - before_width,
                "geometry": part,
            }
        row.update(extra or {})
        rows.append(row)


def _detect_centerline_width_changes(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    before_width_field: str,
    after_width_field: str,
    before_period: str,
    after_period: str,
    config: DetectionConfig,
    before_surface_union: BaseGeometry,
    after_surface_union: BaseGeometry,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """Deprecated legacy detector retained only for import compatibility.

    Formal change detection is routed through ``detect_corridor_changes`` and
    ``road_existence_evidence`` in ``_detect_changes_internal``.  This fixed
    surface-threshold implementation must not be used for formal products.
    """
    before_geometries = np.asarray(before.geometry.values, dtype=object)
    after_geometries = np.asarray(after.geometry.values, dtype=object)
    before_tree = STRtree(before_geometries) if len(before_geometries) else None
    after_tree = STRtree(after_geometries) if len(after_geometries) else None
    line_tolerance = max(config.position_tolerance, 0.1)
    positive_rows: list[dict] = []
    negative_rows: list[dict] = []
    matched_count = 0
    width_changed_count = 0
    width_rejected_match_count = 0
    width_rejected_short_count = 0
    width_rejected_quality_count = 0
    width_rejected_missing_count = 0
    width_rejected_excessive_count = 0
    presence_review_count = 0
    presence_confirmed_count = 0

    def append_presence(
        rows: list[dict], part: BaseGeometry, change_type: str, source_period: str,
        source_fid: str, source_width: float, source_surface: BaseGeometry,
        reference_surface: BaseGeometry,
    ) -> None:
        """Classify road presence before width and retain disagreements for review.

        SAMRoad supplies the centerline candidate and SAM-MLoRA supplies road-surface
        support.  Dual-model agreement remains the high-confidence path, but an
        unmatched whole road is no longer silently discarded merely because the
        surface extractor disagrees.
        """
        nonlocal presence_review_count, presence_confirmed_count
        length = float(part.length)
        source_coverage = _surface_line_coverage(part, source_surface, config.position_tolerance)
        reference_coverage = _surface_line_coverage(part, reference_surface, config.position_tolerance)
        confirmed = (
            length >= config.min_line_length
            and source_coverage >= 0.60
            and reference_coverage <= 0.20
        )
        if length < config.min_line_length:
            return
        reasons = []
        if source_coverage < 0.60:
            reasons.append("source_surface_support_low")
        if reference_coverage > 0.20:
            reasons.append("reference_surface_residual")
        if confirmed:
            presence_confirmed_count += 1
        else:
            presence_review_count += 1
        # Width may be unresolved on an otherwise well-corroborated new/lost road.
        # A non-zero guide buffer is needed only to associate the actual surface
        # difference; it is not exported as a measured width.
        guide_width = source_width if source_width > 0 else max(4.0, 2.0 * config.position_tolerance)
        _append_polygon_rows(
            rows, _line_surface(part, guide_width), change_type, source_period,
            source_fid,
            "dual_model_presence_confirmation" if confirmed else "centerline_presence_review",
            0.0 if change_type == "added" else source_width,
            source_width if change_type == "added" else 0.0, config.min_polygon_area,
            {
                "qa_state": "auto" if confirmed else "review",
                "audit_reason": ";".join(reasons) if reasons else "dual_model_confirmed",
                "source_cov": source_coverage,
                "refer_cov": reference_coverage,
            },
        )

    # Width confirmation is deliberately stricter than road-presence matching.
    # Presence matching remains permissive so a conservative width rule cannot
    # turn an unchanged road into a false added/removed road.  Reciprocal best
    # matches avoid comparing one long road with several differently segmented
    # roads from the other period.
    reciprocal_after_by_before: dict[int, int] = {}
    if config.width_require_reciprocal_match and after_tree is not None:
        for before_index, before_line in enumerate(before_geometries):
            candidate_ids = after_tree.query(
                before_line, predicate="dwithin", distance=config.position_tolerance,
            )
            candidates = [
                (
                    _line_overlap_ratio(before_line, after_geometries[int(after_index)], config.position_tolerance),
                    int(after_index),
                )
                for after_index in candidate_ids
            ]
            candidates = [item for item in candidates if item[0] >= config.line_match_ratio]
            if candidates:
                reciprocal_after_by_before[before_index] = max(candidates)[1]

    for after_index, after_row in after.iterrows():
        after_line = after_row.geometry
        candidate_ids = (
            before_tree.query(after_line, predicate="dwithin", distance=config.position_tolerance)
            if before_tree is not None else []
        )
        candidates = []
        for before_index in candidate_ids:
            before_index = int(before_index)
            ratio = _line_overlap_ratio(after_line, before_geometries[before_index], config.position_tolerance)
            if ratio >= config.line_match_ratio:
                candidates.append((ratio, before_index))

        after_width = _line_width(after_row, after_width_field)
        local_before_cover = (
            union_all(before_geometries.take(candidate_ids)).buffer(line_tolerance)
            if len(candidate_ids) else box(0, 0, 0, 0)
        )
        uncovered_after = after_line.difference(local_before_cover)
        for uncovered_index, uncovered_part in enumerate(_iter_family_parts(make_valid(uncovered_after), "line")):
            if float(uncovered_part.length) < config.min_line_length:
                continue
            append_presence(
                positive_rows, uncovered_part, "added", after_period,
                f"{after_index}:new:{uncovered_index}", after_width,
                after_surface_union, before_surface_union,
            )
        if not candidates:
            if uncovered_after.is_empty and float(after_line.length) >= config.min_line_length:
                append_presence(
                    positive_rows, after_line, "added", after_period, str(after_index), after_width,
                    after_surface_union, before_surface_union,
                )
            continue

        _ratio, before_index = max(candidates)
        matched_count += 1
        before_row = before.iloc[before_index]
        before_line = before_row.geometry
        before_width = _line_width(before_row, before_width_field)
        overlap_zone = before_line.buffer(line_tolerance)
        matched_after_line = after_line.intersection(overlap_zone)
        matched_before_line = before_line.intersection(after_line.buffer(line_tolerance))
        matched_length = min(float(matched_after_line.length), float(matched_before_line.length))
        before_surface = _line_surface(matched_before_line, before_width)
        after_surface = _line_surface(matched_after_line, after_width)
        width_delta = after_width - before_width
        relative_delta = abs(width_delta) / max(before_width, after_width, 0.1)
        strict_match = (
            _ratio >= config.width_line_match_ratio
            and (
                not config.width_require_reciprocal_match
                or reciprocal_after_by_before.get(before_index) == int(after_index)
            )
        )
        usable_quality = (
            _width_quality_is_usable(before_row, config)
            and _width_quality_is_usable(after_row, config)
        )
        width_changed = (
            strict_match
            and matched_length >= config.width_min_overlap_length
            and usable_quality
            and before_width > 0
            and after_width > 0
            and abs(width_delta) >= config.width_change_absolute
            and relative_delta >= config.width_change_ratio
            and relative_delta <= config.width_change_max_ratio
        )
        if not width_changed:
            if (
                abs(width_delta) >= config.width_change_absolute
                and relative_delta > config.width_change_max_ratio
            ):
                width_rejected_excessive_count += 1
            elif abs(width_delta) >= config.width_change_absolute and relative_delta >= config.width_change_ratio:
                if before_width <= 0 or after_width <= 0:
                    width_rejected_missing_count += 1
                elif not strict_match:
                    width_rejected_match_count += 1
                elif matched_length < config.width_min_overlap_length:
                    width_rejected_short_count += 1
                elif not usable_quality:
                    width_rejected_quality_count += 1
            continue
        width_changed_count += 1
        width_min_area = max(config.min_polygon_area, config.width_min_polygon_area)
        if width_delta > 0:
            _append_polygon_rows(
                positive_rows, after_surface.difference(before_surface), "widened", after_period,
                str(after_index), "matched_centerline_width", before_width, after_width,
                width_min_area,
            )
        else:
            _append_polygon_rows(
                negative_rows, before_surface.difference(after_surface), "narrowed", before_period,
                str(before_index), "matched_centerline_width", before_width, after_width,
                width_min_area,
            )

    for before_index, before_row in before.iterrows():
        before_line = before_row.geometry
        before_width = _line_width(before_row, before_width_field)
        candidate_ids = (
            after_tree.query(before_line, predicate="dwithin", distance=config.position_tolerance)
            if after_tree is not None else []
        )
        local_after_cover = (
            union_all(after_geometries.take(candidate_ids)).buffer(line_tolerance)
            if len(candidate_ids) else box(0, 0, 0, 0)
        )
        uncovered_before = before_line.difference(local_after_cover)
        for uncovered_index, uncovered_part in enumerate(_iter_family_parts(make_valid(uncovered_before), "line")):
            if float(uncovered_part.length) < config.min_line_length:
                continue
            append_presence(
                negative_rows, uncovered_part, "removed", before_period,
                f"{before_index}:lost:{uncovered_index}", before_width,
                before_surface_union, after_surface_union,
            )

    columns = {
        "change_typ": [], "source_fid": [], "src_period": [], "length_m": [],
        "area_m2": [], "axis_len_m": [], "class_rule": [], "before_w": [],
        "after_w": [], "width_diff": [],
    }
    positive = gpd.GeoDataFrame(positive_rows, geometry="geometry", crs=before.crs) if positive_rows else gpd.GeoDataFrame(columns, geometry=[], crs=before.crs)
    negative = gpd.GeoDataFrame(negative_rows, geometry="geometry", crs=before.crs) if negative_rows else gpd.GeoDataFrame(columns, geometry=[], crs=before.crs)
    metadata = {
        "classification_method": "actual_surface_difference_guided_by_centerline_width",
        "before_width_field": before_width_field,
        "after_width_field": after_width_field,
        "matched_centerline_count": matched_count,
        "width_changed_centerline_count": width_changed_count,
        "width_change_absolute_m": config.width_change_absolute,
        "width_change_ratio": config.width_change_ratio,
        "width_change_max_ratio": config.width_change_max_ratio,
        "line_match_ratio": config.line_match_ratio,
        "width_line_match_ratio": config.width_line_match_ratio,
        "width_min_overlap_length_m": config.width_min_overlap_length,
        "width_min_polygon_area_m2": config.width_min_polygon_area,
        "width_require_reciprocal_match": config.width_require_reciprocal_match,
        "width_exclude_low_quality": config.width_exclude_low_quality,
        "width_rejected_match_count": width_rejected_match_count,
        "width_rejected_short_count": width_rejected_short_count,
        "width_rejected_quality_count": width_rejected_quality_count,
        "width_rejected_missing_count": width_rejected_missing_count,
        "width_rejected_excessive_count": width_rejected_excessive_count,
        "presence_confirmed_count": presence_confirmed_count,
        "presence_review_count": presence_review_count,
        "presence_suppressed_low_confidence_count": 0,
        "presence_confirmation_rule": "presence_first; source_surface>=0.60_and_reference_surface<=0.20_is_auto_else_review",
    }
    return positive, negative, metadata


def _partition_surface_part_by_guides(
    part: BaseGeometry,
    guide_geometries: dict[str, BaseGeometry],
    tolerance: float,
) -> dict[str, list[BaseGeometry]]:
    if len(guide_geometries) == 1:
        change_type = next(iter(guide_geometries))
        return {change_type: [part]}

    minx, miny, maxx, maxy = part.bounds
    padding = max(2.0, tolerance)
    minx, miny, maxx, maxy = minx - padding, miny - padding, maxx + padding, maxy + padding
    resolution = min(1.0, max(0.25, tolerance / 2.0 if tolerance > 0 else 0.5))
    width = max(3, int(math.ceil((maxx - minx) / resolution)))
    height = max(3, int(math.ceil((maxy - miny) / resolution)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    part_mask = rasterize(
        [(part, 1)], out_shape=(height, width), transform=transform,
        fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    change_types = list(guide_geometries)
    distance_layers = []
    for change_type in change_types:
        guide_mask = rasterize(
            [(guide_geometries[change_type], 1)], out_shape=(height, width), transform=transform,
            fill=0, dtype="uint8", all_touched=True,
        ).astype(bool)
        distance_layers.append(
            distance_transform_edt(~guide_mask, sampling=(pixel_height, pixel_width))
        )
    assignments = np.argmin(np.stack(distance_layers, axis=0), axis=0)
    result: dict[str, list[BaseGeometry]] = {change_type: [] for change_type in change_types}
    for index, change_type in enumerate(change_types):
        mask = part_mask & (assignments == index)
        for mapping, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
            if value != 1:
                continue
            clipped = make_valid(shape(mapping).intersection(part))
            result[change_type].extend(_iter_family_parts(clipped, "polygon"))
    return result


def _classify_actual_surface_difference(
    difference: BaseGeometry,
    guides: gpd.GeoDataFrame,
    source_period: str,
    crs,
    config: DetectionConfig,
) -> gpd.GeoDataFrame:
    rows: list[dict] = []
    if guides.empty:
        return gpd.GeoDataFrame(
            {
                "change_typ": [], "source_fid": [], "src_period": [], "length_m": [],
                "area_m2": [], "axis_len_m": [], "class_rule": [], "before_w": [],
                "after_w": [], "width_diff": [],
            },
            geometry=[], crs=crs,
        )
    guide_geometries = np.asarray(guides.geometry.values, dtype=object)
    tree = STRtree(guide_geometries)
    guide_widths = pd.concat(
        [pd.to_numeric(guides.get("before_w"), errors="coerce"), pd.to_numeric(guides.get("after_w"), errors="coerce")],
        ignore_index=True,
    )
    max_guide_width = float(guide_widths.max()) if bool(guide_widths.notna().any()) else 0.0
    guide_search_distance = max(config.position_tolerance * 2.0, max_guide_width, 5.0)
    for part_index, part in enumerate(_iter_family_parts(make_valid(difference), "polygon")):
        if float(part.area) < config.min_polygon_area:
            continue
        candidate_ids = tree.query(
            part, predicate="dwithin", distance=guide_search_distance,
        )
        if len(candidate_ids) == 0:
            continue
        candidate_guides = guides.iloc[np.asarray(candidate_ids, dtype=int)]
        class_guides = {
            change_type: union_all(np.asarray(group.geometry.values, dtype=object))
            for change_type, group in candidate_guides.groupby("change_typ")
        }
        classified = _partition_surface_part_by_guides(part, class_guides, config.position_tolerance)
        for change_type, regions in classified.items():
            class_rows = candidate_guides.loc[candidate_guides["change_typ"] == change_type]
            before_width = float(pd.to_numeric(class_rows.get("before_w"), errors="coerce").median())
            after_width = float(pd.to_numeric(class_rows.get("after_w"), errors="coerce").median())
            min_area = (
                max(config.min_polygon_area, config.width_min_polygon_area)
                if change_type in {"widened", "narrowed"}
                else config.min_polygon_area
            )
            for region_index, region in enumerate(regions):
                if float(region.area) < min_area:
                    continue
                rows.append(
                    {
                        "change_typ": change_type,
                        "source_fid": f"surface:{part_index}:{region_index}",
                        "src_period": source_period,
                        "length_m": 0.0,
                        "area_m2": float(region.area),
                        "axis_len_m": 0.0,
                        "class_rule": "actual_surface_by_centerline_guide",
                        "before_w": before_width,
                        "after_w": after_width,
                        "width_diff": after_width - before_width,
                        "qa_state": (
                            "review" if "review" in set(class_rows.get("qa_state", [])) else "auto"
                        ),
                        "audit_reason": ";".join(
                            sorted({str(value) for value in class_rows.get("audit_reason", []) if str(value)})
                        ),
                        "geometry": region,
                    }
                )
    if rows:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(
        {
            "change_typ": [], "source_fid": [], "src_period": [], "length_m": [],
            "area_m2": [], "axis_len_m": [], "class_rule": [], "before_w": [],
            "after_w": [], "width_diff": [],
        },
        geometry=[], crs=crs,
    )


def _retain_presence_fallbacks(
    classified: gpd.GeoDataFrame,
    guides: gpd.GeoDataFrame,
    change_type: str,
    config: DetectionConfig,
) -> gpd.GeoDataFrame:
    """Keep unmatched whole-road presence evidence even when surfaces disagree."""
    presence = guides.loc[guides.get("change_typ") == change_type].copy()
    if presence.empty:
        return classified
    same_type = classified.loc[classified.get("change_typ") == change_type]
    covered = (
        union_all(np.asarray(same_type.geometry.values, dtype=object))
        if not same_type.empty else box(0, 0, 0, 0)
    )
    rows: list[dict] = []
    for guide_index, guide in presence.iterrows():
        remainder = make_valid(guide.geometry.difference(covered))
        for part_index, part in enumerate(_iter_family_parts(remainder, "polygon")):
            if float(part.area) < config.min_polygon_area:
                continue
            row = {name: guide.get(name) for name in presence.columns if name != "geometry"}
            row.update({
                "source_fid": f"presence_fallback:{guide_index}:{part_index}",
                "area_m2": float(part.area),
                "class_rule": "presence_first_centerline_fallback",
                "geometry": part,
            })
            rows.append(row)
    if not rows:
        return classified
    fallback = gpd.GeoDataFrame(rows, geometry="geometry", crs=guides.crs)
    return gpd.GeoDataFrame(
        pd.concat([classified, fallback], ignore_index=True),
        geometry="geometry", crs=classified.crs,
    )


def _surface_change_regions(
    source_surface: BaseGeometry,
    reference_surface: BaseGeometry,
    changed_part: BaseGeometry,
    tolerance: float,
) -> tuple[list[BaseGeometry], list[BaseGeometry], float]:
    """Split one difference polygon into complete-road and lateral-width regions."""
    minx, miny, maxx, maxy = changed_part.bounds
    part_scale = math.sqrt(max(float(changed_part.area), 1.0))
    padding = max(20.0, tolerance * 4.0, min(100.0, part_scale * 2.0))
    minx, miny, maxx, maxy = minx - padding, miny - padding, maxx + padding, maxy + padding

    resolution = min(1.0, max(0.25, tolerance / 2.0 if tolerance > 0 else 0.5))
    width = max(3, int(math.ceil((maxx - minx) / resolution)))
    height = max(3, int(math.ceil((maxy - miny) / resolution)))
    max_pixels = 25_000_000
    if width * height > max_pixels:
        scale = math.sqrt((width * height) / max_pixels)
        width = max(3, int(math.ceil(width / scale)))
        height = max(3, int(math.ceil(height / scale)))

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    crop_box = box(minx, miny, maxx, maxy)
    source_crop = source_surface.intersection(crop_box)
    reference_crop = reference_surface.intersection(crop_box)
    source_mask = rasterize(
        [(source_crop, 1)], out_shape=(height, width), transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)
    part_mask = rasterize(
        [(changed_part, 1)], out_shape=(height, width), transform=transform,
        fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    reference_mask = rasterize(
        [(reference_crop, 1)], out_shape=(height, width), transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)
    source_axis = skeletonize(source_mask)
    reference_axis = skeletonize(reference_mask)
    axis_in_change = source_axis & part_mask
    if not axis_in_change.any():
        return [], [changed_part], 0.0

    components = label(axis_in_change, connectivity=2)
    counts = np.bincount(components.ravel())
    largest_component = int(counts[1:].max()) if len(counts) > 1 else 0
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    pixel_size = max(pixel_width, pixel_height)
    axis_length = largest_component * pixel_size

    distance_to_change_axis = distance_transform_edt(
        ~axis_in_change, sampling=(pixel_height, pixel_width),
    )
    if reference_axis.any():
        distance_to_reference_axis = distance_transform_edt(
            ~reference_axis, sampling=(pixel_height, pixel_width),
        )
        complete_road_mask = part_mask & (distance_to_change_axis <= distance_to_reference_axis)
    else:
        complete_road_mask = part_mask.copy()
    width_change_mask = part_mask & ~complete_road_mask

    def vectorize(mask: np.ndarray) -> list[BaseGeometry]:
        regions = []
        for mapping, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
            if value != 1:
                continue
            clipped = make_valid(shape(mapping).intersection(changed_part))
            regions.extend(_iter_family_parts(clipped, "polygon"))
        return regions

    return vectorize(complete_road_mask), vectorize(width_change_mask), axis_length


def _difference_features(
    source_union: BaseGeometry,
    reference_union: BaseGeometry,
    family: str,
    change_type: str,
    width_change_type: str | None,
    source_period: str,
    crs,
    config: DetectionConfig,
) -> gpd.GeoDataFrame:
    rows: list[dict] = []
    tolerance = max(0.0, config.position_tolerance)
    comparison_cover = reference_union.buffer(tolerance) if family == "line" and tolerance > 0 else reference_union
    difference = source_union.difference(comparison_cover)

    for part_index, part in enumerate(_iter_family_parts(make_valid(difference), family)):
        if family == "line":
            if float(part.length) < config.min_line_length:
                continue
            classified_parts = [(part, change_type, float(part.length), "centerline_difference")]
        else:
            significant_part = part.difference(reference_union.buffer(tolerance)) if tolerance > 0 else part
            if float(significant_part.area) < config.min_polygon_area:
                continue
            complete_regions, width_regions, axis_length = _surface_change_regions(
                source_union, reference_union, part, tolerance,
            )
            minimum_axis_length = max(2.0, math.sqrt(max(config.min_polygon_area, 0.0)))
            if axis_length < minimum_axis_length:
                classified_parts = [(part, width_change_type or change_type, 0.0, "lateral_width_band")]
            else:
                classified_parts = [
                    (region, change_type, axis_length, "surface_axis_change")
                    for region in complete_regions
                    if float(region.area) >= config.min_polygon_area
                ]
                classified_parts.extend(
                    (region, width_change_type or change_type, 0.0, "lateral_width_band")
                    for region in width_regions
                    if float(region.area) >= config.min_polygon_area
                )

        for sub_index, (region, part_change_type, axis_length, class_rule) in enumerate(classified_parts):
            rows.append(
                {
                    "change_typ": part_change_type,
                    "source_fid": f"{part_index}:{sub_index}",
                    "src_period": source_period,
                    "length_m": float(region.length) if family == "line" else 0.0,
                    "area_m2": float(region.area) if family == "polygon" else 0.0,
                    "axis_len_m": axis_length,
                    "class_rule": class_rule,
                    "geometry": region,
                }
            )

    if rows:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(
        {
            "change_typ": [], "source_fid": [], "src_period": [], "length_m": [],
            "area_m2": [], "axis_len_m": [], "class_rule": [],
        },
        geometry=[], crs=crs,
    )


def _unchanged_features(
    before_union: BaseGeometry,
    after_union: BaseGeometry,
    family: str,
    crs,
) -> gpd.GeoDataFrame:
    rows = [
        {"area_m2": float(part.area), "geometry": part}
        for part in _iter_family_parts(make_valid(before_union.intersection(after_union)), family)
    ]
    if rows:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame({"area_m2": []}, geometry=[], crs=crs)


def _detect_changes_internal(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    config: DetectionConfig,
    before_period: str,
    after_period: str,
    before_surfaces: gpd.GeoDataFrame | None = None,
    after_surfaces: gpd.GeoDataFrame | None = None,
    before_width_segments: gpd.GeoDataFrame | None = None,
    after_width_segments: gpd.GeoDataFrame | None = None,
    before_valid_area: gpd.GeoDataFrame | None = None,
    after_valid_area: gpd.GeoDataFrame | None = None,
    before_probability: RoadProbabilityRaster | None = None,
    after_probability: RoadProbabilityRaster | None = None,
    artifacts: dict[str, gpd.GeoDataFrame] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    before = _clean_geometries(before)
    after = _clean_geometries(after)
    before_analysis, after_analysis, analysis_crs, output_crs = _analysis_crs(before, after)
    before_family = _family_from_geometries(before_analysis.geometry)
    after_family = _family_from_geometries(after_analysis.geometry)
    family = before_family or after_family
    if family is None:
        raise ValueError("At least one road layer must contain a geometry.")
    if before_family and after_family and before_family != after_family:
        raise ValueError("The two road layers must both be centerlines or both be road surfaces.")

    before_width_field = _width_field(before_analysis) if family == "line" else None
    after_width_field = _width_field(after_analysis) if family == "line" else None
    classification_metadata = {}
    if family == "line":
        before_width_field = before_width_field or _width_field_or_zero(before_analysis)
        after_width_field = after_width_field or _width_field_or_zero(after_analysis)
        if before_surfaces is not None:
            before_surfaces = _to_crs_if_needed(_clean_geometries(before_surfaces), analysis_crs)
            if not before_surfaces.empty and _family_from_geometries(before_surfaces.geometry) != "polygon":
                raise ValueError("The actual before road surface layer must contain polygons.")
        if after_surfaces is not None:
            after_surfaces = _to_crs_if_needed(_clean_geometries(after_surfaces), analysis_crs)
            if not after_surfaces.empty and _family_from_geometries(after_surfaces.geometry) != "polygon":
                raise ValueError("The actual after road surface layer must contain polygons.")
        if before_width_segments is not None:
            before_width_segments = _to_crs_if_needed(_clean_geometries(before_width_segments), analysis_crs)
        if after_width_segments is not None:
            after_width_segments = _to_crs_if_needed(_clean_geometries(after_width_segments), analysis_crs)

        def valid_union(frame: gpd.GeoDataFrame | None):
            if frame is None:
                return None
            frame = _to_crs_if_needed(_clean_geometries(frame), analysis_crs)
            if frame.empty:
                return box(0, 0, 0, 0)
            if _family_from_geometries(frame.geometry) != "polygon":
                raise ValueError("Valid-observation layers must contain polygons.")
            return union_all(np.asarray(frame.geometry.values, dtype=object))

        added, removed, unchanged, classification_metadata, corridor_artifacts = detect_corridor_changes(
            before_analysis, after_analysis, config, before_period, after_period,
            before_surfaces=before_surfaces, after_surfaces=after_surfaces,
            before_width_segments=before_width_segments, after_width_segments=after_width_segments,
            before_valid=valid_union(before_valid_area), after_valid=valid_union(after_valid_area),
            before_probability=before_probability, after_probability=after_probability,
        )
        if artifacts is not None:
            artifacts.update(corridor_artifacts)
    else:
        before_union = union_all(np.asarray(before_analysis.geometry.values, dtype=object))
        after_union = union_all(np.asarray(after_analysis.geometry.values, dtype=object))
        unchanged = _unchanged_features(before_union, after_union, family, analysis_crs)
        added = _difference_features(
            after_union, before_union, family, "added", "widened" if family == "polygon" else None,
            after_period, analysis_crs, config,
        )
        removed = _difference_features(
            before_union, after_union, family, "removed", "narrowed" if family == "polygon" else None,
            before_period, analysis_crs, config,
        )
        classification_metadata = {
            "classification_method": "surface_intersection_difference_and_medial_axis"
            if family == "polygon" else "centerline_geometry_difference",
        }

    counters = {change_type: 0 for change_type in CHANGE_TYPES}
    for frame in (added, removed):
        identifiers = []
        for value in frame["change_typ"]:
            counters[value] += 1
            identifiers.append(f"{CHANGE_PREFIXES[value]}{counters[value]:07d}")
        frame.insert(0, "change_id", identifiers)

    summary = {
        "before_period": before_period,
        "after_period": after_period,
        "geometry_family": family,
        "analysis_crs": str(analysis_crs),
        "output_crs": str(output_crs),
        "position_tolerance_m": config.position_tolerance,
        "min_line_length_m": config.min_line_length,
        "min_polygon_area_m2": config.min_polygon_area,
        "unchanged_feature_count": len(unchanged),
        "unchanged_area_m2": float(unchanged["area_m2"].sum()),
        **classification_metadata,
    }
    combined = pd.concat([added, removed], ignore_index=True)
    qa_state = (
        combined["qa_state"].fillna("auto")
        if "qa_state" in combined.columns
        else pd.Series(["auto"] * len(combined), index=combined.index, dtype="object")
    )
    formal = combined.loc[qa_state == "auto"]
    for value in CHANGE_TYPES:
        selected = formal.loc[formal["change_typ"] == value]
        summary[f"{value}_feature_count"] = len(selected)
        summary[f"{value}_length_m"] = float(selected["length_m"].sum())
        summary[f"{value}_area_m2"] = float(selected["area_m2"].sum())
        summary[f"review_{value}_feature_count"] = int(
            ((combined["change_typ"] == value) & (qa_state == "review")).sum()
        )
    summary["review_feature_count"] = int((qa_state == "review").sum())
    return added, removed, unchanged, summary


def detect_changes(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    config: DetectionConfig = DetectionConfig(),
    before_period: str = "before",
    after_period: str = "after",
    before_surfaces: gpd.GeoDataFrame | None = None,
    after_surfaces: gpd.GeoDataFrame | None = None,
    before_width_segments: gpd.GeoDataFrame | None = None,
    after_width_segments: gpd.GeoDataFrame | None = None,
    before_valid_area: gpd.GeoDataFrame | None = None,
    after_valid_area: gpd.GeoDataFrame | None = None,
    before_probability: RoadProbabilityRaster | None = None,
    after_probability: RoadProbabilityRaster | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    added, removed, _unchanged, summary = _detect_changes_internal(
        before, after, config, before_period, after_period, before_surfaces, after_surfaces,
        before_width_segments, after_width_segments, before_valid_area, after_valid_area,
        before_probability, after_probability,
    )
    return added, removed, summary


def _normalized_change_type(value, class_mode: str = "four") -> str | None:
    if class_mode not in {"three", "four"}:
        raise ValueError("Change evaluation class mode must be 'three' or 'four'.")
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)) or float(value) == 0:
            return None
        return "added" if float(value) > 0 else "removed"
    normalized = str(value).strip().lower().replace(" ", "_")
    if normalized in ADDED_ALIASES:
        return "added"
    if normalized in REMOVED_ALIASES:
        return "removed"
    if normalized in WIDENED_ALIASES:
        return "width_changed" if class_mode == "three" else "widened"
    if normalized in NARROWED_ALIASES:
        return "width_changed" if class_mode == "three" else "narrowed"
    if normalized in WIDTH_CHANGED_ALIASES:
        return "width_changed" if class_mode == "three" else None
    return None


def normalized_change_series(frame: gpd.GeoDataFrame, field: str, class_mode: str) -> pd.Series:
    if field not in frame.columns:
        raise ValueError(f"Change type field does not exist: {field}")
    return frame[field].map(lambda value: _normalized_change_type(value, class_mode))


def _truth_type_field(truth: gpd.GeoDataFrame, requested: str) -> str | None:
    if requested:
        if requested not in truth.columns:
            if truth.empty:
                return None
            raise ValueError(f"Truth change type field does not exist: {requested}")
        return requested
    return next((field for field in TRUTH_TYPE_FIELDS if field in truth.columns), None)


def _normalized_truth_change_type(value, field: str | None, class_mode: str) -> str | None:
    """Normalize truth types, including the production BHBM 2/3/4 convention."""
    if field and field.casefold() == "bhbm":
        try:
            code = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None
        mapped = BHBM_CHANGE_TYPES.get(code)
        if mapped == "width_changed" and class_mode == "four":
            return None
        return mapped
    return _normalized_change_type(value, class_mode)


def _polygon_union(frame: gpd.GeoDataFrame) -> BaseGeometry:
    if frame.empty:
        return box(0, 0, 0, 0)
    polygons = [geometry for geometry in frame.geometry if geometry.geom_type in POLYGON_TYPES]
    if len(polygons) != len(frame):
        raise ValueError("The validation area layer must contain only polygons.")
    return union_all(np.asarray(polygons, dtype=object))


def _clip_frame(frame: gpd.GeoDataFrame, area: BaseGeometry) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.copy()
    clipped = frame.loc[frame.geometry.intersects(area)].copy()
    clipped.geometry = clipped.geometry.intersection(area)
    return clipped.loc[~clipped.geometry.is_empty].copy()


def _support_geometry(frame: gpd.GeoDataFrame, line_tolerance: float) -> BaseGeometry:
    if frame.empty:
        return box(0, 0, 0, 0)
    supports = []
    for geometry in frame.geometry:
        if geometry.geom_type in LINE_TYPES:
            if line_tolerance <= 0:
                raise ValueError("Evaluation tolerance must be greater than zero for line changes.")
            supports.append(geometry.buffer(line_tolerance))
        elif geometry.geom_type in POLYGON_TYPES:
            supports.append(geometry)
        elif hasattr(geometry, "geoms"):
            nested = gpd.GeoDataFrame(geometry=list(geometry.geoms), crs=frame.crs)
            supports.append(_support_geometry(nested, line_tolerance))
    return union_all(np.asarray(supports, dtype=object)) if supports else box(0, 0, 0, 0)


def _metric_row(name: str, predicted: BaseGeometry, truth: BaseGeometry, validation: BaseGeometry) -> dict:
    tp = float(predicted.intersection(truth).area)
    pred_area = float(predicted.area)
    truth_area = float(truth.area)
    fp = max(0.0, pred_area - tp)
    fn = max(0.0, truth_area - tp)
    occupied = float(predicted.union(truth).area)
    validation_area = float(validation.area)
    tn = max(0.0, validation_area - occupied)
    precision = tp / (tp + fp) if tp + fp else (1.0 if truth_area == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if pred_area == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    accuracy = (tp + tn) / validation_area if validation_area else 0.0
    return {
        "class": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "support_accuracy": accuracy,
        "tp_m2": tp,
        "fp_m2": fp,
        "fn_m2": fn,
        "tn_m2": tn,
        "predicted_support_m2": pred_area,
        "truth_support_m2": truth_area,
        "validation_area_m2": validation_area,
        "centerline_offset_status": "not_applicable",
        "centerline_offset_reason": "",
        "centerline_avg_offset_m": None,
        "truth_to_pred_avg_m": None,
        "pred_to_truth_avg_m": None,
        "truth_axis_length_m": 0.0,
        "predicted_axis_length_m": 0.0,
        "truth_distance_integral_m2": 0.0,
        "predicted_distance_integral_m2": 0.0,
        "included_truth_feature_count": 0,
        "excluded_truth_feature_count": 0,
    }


def _matched_object_pairs(
    predicted: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    line_tolerance: float,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    predicted_support = [
        _support_geometry(predicted.iloc[[index]], line_tolerance)
        for index in range(len(predicted))
    ]
    truth_support = [
        _support_geometry(truth.iloc[[index]], line_tolerance)
        for index in range(len(truth))
    ]
    candidates = []
    for predicted_index, predicted_geometry in enumerate(predicted_support):
        for truth_index, truth_geometry in enumerate(truth_support):
            intersection = float(predicted_geometry.intersection(truth_geometry).area)
            if intersection <= 0:
                continue
            union = float(predicted_geometry.union(truth_geometry).area)
            iou = intersection / union if union else 0.0
            if iou >= iou_threshold:
                candidates.append((iou, predicted_index, truth_index))
    matched_predicted = set()
    matched_truth = set()
    pairs = []
    for _iou, predicted_index, truth_index in sorted(candidates, reverse=True):
        if predicted_index in matched_predicted or truth_index in matched_truth:
            continue
        matched_predicted.add(predicted_index)
        matched_truth.add(truth_index)
        pairs.append((predicted_index, truth_index))
    return pairs


def _centerline_offset_metrics(
    predicted: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    evaluation_tolerance: float,
    object_iou_threshold: float = 0.1,
) -> dict:
    """Measure axes only for same-class object true-positive pairs."""
    total_truth_count = int(len(truth))
    result = {
        "centerline_offset_status": "unavailable",
        "centerline_offset_reason": "",
        "centerline_avg_offset_m": None,
        "truth_to_pred_avg_m": None,
        "pred_to_truth_avg_m": None,
        "truth_axis_length_m": 0.0,
        "predicted_axis_length_m": 0.0,
        "truth_distance_integral_m2": 0.0,
        "predicted_distance_integral_m2": 0.0,
        "included_truth_feature_count": 0,
        "excluded_truth_feature_count": total_truth_count,
    }
    if predicted.empty or truth.empty:
        result["centerline_offset_reason"] = "预测或真值中没有可配对的变化面"
        return result
    matched_pairs = _matched_object_pairs(
        predicted, truth, evaluation_tolerance, object_iou_threshold,
    )
    if not matched_pairs:
        result["centerline_offset_reason"] = "同类型预测与真值中没有对象级真阳性配对"
        return result
    matched_predicted = sorted({predicted_index for predicted_index, _truth_index in matched_pairs})
    matched_truth = sorted({truth_index for _predicted_index, truth_index in matched_pairs})
    predicted = predicted.iloc[matched_predicted].reset_index(drop=True)
    truth = truth.iloc[matched_truth].reset_index(drop=True)
    predicted_supports = [
        _support_geometry(predicted.iloc[[index]], evaluation_tolerance)
        for index in range(len(predicted))
    ]
    truth_supports = [
        _support_geometry(truth.iloc[[index]], evaluation_tolerance)
        for index in range(len(truth))
    ]
    truth_match_supports = [geometry.buffer(evaluation_tolerance) for geometry in truth_supports]
    predicted_links = [set() for _ in predicted_supports]
    truth_links = [set() for _ in truth_supports]
    for predicted_index, predicted_geometry in enumerate(predicted_supports):
        for truth_index, truth_geometry in enumerate(truth_match_supports):
            if predicted_geometry.intersects(truth_geometry):
                predicted_links[predicted_index].add(truth_index)
                truth_links[truth_index].add(predicted_index)

    # Connected bipartite groups handle one truth polygon corresponding to two
    # carriageway polygons (and the inverse) without forcing one-to-one matching.
    groups: list[tuple[set[int], set[int]]] = []
    visited_truth: set[int] = set()
    for start in range(len(truth_supports)):
        if start in visited_truth or not truth_links[start]:
            continue
        group_truth: set[int] = set()
        group_predicted: set[int] = set()
        truth_stack = [start]
        while truth_stack:
            truth_index = truth_stack.pop()
            if truth_index in group_truth:
                continue
            group_truth.add(truth_index)
            visited_truth.add(truth_index)
            for predicted_index in truth_links[truth_index]:
                if predicted_index in group_predicted:
                    continue
                group_predicted.add(predicted_index)
                truth_stack.extend(predicted_links[predicted_index] - group_truth)
        groups.append((group_predicted, group_truth))

    if not groups:
        result["centerline_offset_reason"] = "没有空间对应的预测与真值变化对象"
        return result

    truth_length = predicted_length = truth_integral = predicted_integral = 0.0
    included_truth: set[int] = set()
    close_distance = max(0.5, float(evaluation_tolerance))
    for predicted_indices, truth_indices in groups:
        predicted_geometry = union_all(np.asarray([predicted_supports[index] for index in predicted_indices], dtype=object))
        truth_geometry = union_all(np.asarray([truth_supports[index] for index in truth_indices], dtype=object))
        # A dual carriageway may be represented by two prediction polygons while
        # the truth is one road-surface polygon. Morphological closing creates one
        # corridor before skeletonization, yielding its representative centre axis.
        if len(predicted_indices) > 1 or len(truth_indices) > 1:
            predicted_closed = predicted_geometry.buffer(close_distance).buffer(-close_distance)
            truth_closed = truth_geometry.buffer(close_distance).buffer(-close_distance)
            if not predicted_closed.is_empty:
                predicted_geometry = predicted_closed
            if not truth_closed.is_empty:
                truth_geometry = truth_closed
        minx = min(predicted_geometry.bounds[0], truth_geometry.bounds[0])
        miny = min(predicted_geometry.bounds[1], truth_geometry.bounds[1])
        maxx = max(predicted_geometry.bounds[2], truth_geometry.bounds[2])
        maxy = max(predicted_geometry.bounds[3], truth_geometry.bounds[3])
        resolution = max(0.25, min(2.0, float(evaluation_tolerance) / 4.0))
        width = max(1, int(math.ceil((maxx - minx) / resolution)) + 4)
        height = max(1, int(math.ceil((maxy - miny) / resolution)) + 4)
        max_pixels = 8_000_000
        if width * height > max_pixels:
            scale = math.sqrt((width * height) / max_pixels)
            resolution *= scale
            width = max(1, int(math.ceil((maxx - minx) / resolution)) + 4)
            height = max(1, int(math.ceil((maxy - miny) / resolution)) + 4)
        transform = from_origin(minx - 2 * resolution, maxy + 2 * resolution, resolution, resolution)

        def axis(geometry: BaseGeometry) -> np.ndarray:
            mask = rasterize(
                [(geometry, 1)], out_shape=(height, width), transform=transform,
                fill=0, dtype="uint8", all_touched=True,
            ).astype(bool)
            return skeletonize(mask)

        predicted_axis = axis(predicted_geometry)
        truth_axis = axis(truth_geometry)
        predicted_count = int(predicted_axis.sum())
        truth_count = int(truth_axis.sum())
        if predicted_count == 0 or truth_count == 0:
            continue
        distance_to_predicted = distance_transform_edt(~predicted_axis, sampling=(resolution, resolution))
        distance_to_truth = distance_transform_edt(~truth_axis, sampling=(resolution, resolution))
        group_truth_length = float(truth_count * resolution)
        group_predicted_length = float(predicted_count * resolution)
        truth_integral += float(distance_to_predicted[truth_axis].mean()) * group_truth_length
        predicted_integral += float(distance_to_truth[predicted_axis].mean()) * group_predicted_length
        truth_length += group_truth_length
        predicted_length += group_predicted_length
        included_truth.update(truth_indices)
    if truth_length <= 0 or predicted_length <= 0:
        result["centerline_offset_reason"] = "已匹配变化面无法提取稳定中心轴"
        return result
    truth_to_predicted = truth_integral / truth_length
    predicted_to_truth = predicted_integral / predicted_length
    result.update({
        "centerline_offset_status": "computed",
        "centerline_offset_reason": "仅统计同类型对象级真阳性配对；双线道路先合并为走廊，再计算双向长度加权骨架偏移",
        "centerline_avg_offset_m": (
            (truth_integral + predicted_integral) / (truth_length + predicted_length)
        ),
        "truth_to_pred_avg_m": truth_to_predicted,
        "pred_to_truth_avg_m": predicted_to_truth,
        "truth_axis_length_m": truth_length,
        "predicted_axis_length_m": predicted_length,
        "truth_distance_integral_m2": truth_integral,
        "predicted_distance_integral_m2": predicted_integral,
        "included_truth_feature_count": len(included_truth),
        "excluded_truth_feature_count": int(total_truth_count - len(included_truth)),
    })
    return result


def _object_metric_values(
    predicted: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    line_tolerance: float,
    iou_threshold: float,
) -> dict:
    matched_pairs = _matched_object_pairs(
        predicted, truth, line_tolerance, iou_threshold,
    )
    matched_predicted = {predicted_index for predicted_index, _truth_index in matched_pairs}
    matched_truth = {truth_index for _predicted_index, truth_index in matched_pairs}
    true_positive = len(matched_predicted)
    false_positive = len(predicted) - true_positive
    false_negative = len(truth) - len(matched_truth)
    precision = true_positive / len(predicted) if len(predicted) else (1.0 if not len(truth) else 0.0)
    recall = true_positive / len(truth) if len(truth) else (1.0 if not len(predicted) else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
        "object_tp": true_positive,
        "object_fp": false_positive,
        "object_fn": false_negative,
    }


def evaluate_changes(
    predicted: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    validation_area: gpd.GeoDataFrame | None = None,
    truth_type_field: str = "",
    evaluation_tolerance: float = 5.0,
    class_mode: str = "four",
    object_iou_threshold: float = 0.1,
) -> tuple[list[dict], dict]:
    if predicted.crs is None or truth.crs is None:
        raise ValueError("Predicted changes and truth must define a CRS.")
    predicted = _clean_geometries(predicted)
    truth = _clean_geometries(truth)
    predicted, truth, metric_crs, _output_crs = _analysis_crs(predicted, truth)
    field = _truth_type_field(truth, truth_type_field)
    truth = truth.copy()
    if class_mode not in {"three", "four"}:
        raise ValueError("Change evaluation class mode must be 'three' or 'four'.")
    if not 0 < object_iou_threshold <= 1:
        raise ValueError("Object IoU threshold must be greater than zero and at most one.")
    predicted = predicted.copy()
    predicted["_eval_type"] = normalized_change_series(predicted, "change_typ", class_mode)
    truth["_eval_type"] = (
        truth[field].map(lambda value: _normalized_truth_change_type(value, field, class_mode)) if field else None
    )

    if validation_area is not None:
        if validation_area.crs is None:
            raise ValueError("The validation area layer must define a CRS.")
        validation = _polygon_union(_clean_geometries(validation_area).to_crs(predicted.crs))
        validation_extent_source = "validation_area"
    else:
        extent_source = truth if not truth.empty else predicted
        if extent_source.empty:
            validation = box(0.0, 0.0, 1.0, 1.0)
            validation_extent_source = "empty_default"
        else:
            minx, miny, maxx, maxy = extent_source.total_bounds
            pad = max(
                evaluation_tolerance * 4.0,
                max(maxx - minx, maxy - miny) * 0.02,
                1.0,
            )
            validation = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
            validation_extent_source = (
                "truth_bounds" if not truth.empty else "prediction_bounds"
            )

    predicted = _clip_frame(predicted, validation)
    truth = _clip_frame(truth, validation)
    overall = _metric_row(
            "all",
            _support_geometry(predicted, evaluation_tolerance).intersection(validation),
            _support_geometry(truth, evaluation_tolerance).intersection(validation),
            validation,
        )
    overall.update(_object_metric_values(predicted, truth, evaluation_tolerance, object_iou_threshold))
    rows = [overall]
    classified = int(truth["_eval_type"].notna().sum()) if field else 0
    if field:
        evaluation_types = THREE_CLASS_CHANGE_TYPES if class_mode == "three" else CHANGE_TYPES
        for change_type in evaluation_types:
            pred_part = predicted.loc[predicted["_eval_type"] == change_type]
            truth_part = truth.loc[truth["_eval_type"] == change_type]
            row = _metric_row(
                    change_type,
                    _support_geometry(pred_part, evaluation_tolerance).intersection(validation),
                    _support_geometry(truth_part, evaluation_tolerance).intersection(validation),
                    validation,
                )
            row.update(_object_metric_values(pred_part, truth_part, evaluation_tolerance, object_iou_threshold))
            if change_type in {"added", "removed"}:
                row.update(_centerline_offset_metrics(
                    pred_part,
                    truth_part,
                    evaluation_tolerance,
                    object_iou_threshold,
                ))
            elif change_type == "width_changed":
                row.update({
                    "centerline_offset_status": "excluded",
                    "centerline_offset_reason": "宽度变化真值面不唯一确定道路中心线，未计算中心线偏移",
                    "excluded_truth_feature_count": int(len(truth_part)),
                })
            rows.append(row)
        # Pool the two eligible classes from their additive distance integrals.
        # Computing one union here would allow a newly-added road to match a nearby
        # removed road, which is not a valid homologous centerline comparison.
        offset_rows = [row for row in rows if row["class"] in {"added", "removed"}]
        truth_length = sum(float(row["truth_axis_length_m"]) for row in offset_rows)
        predicted_length = sum(float(row["predicted_axis_length_m"]) for row in offset_rows)
        truth_integral = sum(float(row["truth_distance_integral_m2"]) for row in offset_rows)
        predicted_integral = sum(float(row["predicted_distance_integral_m2"]) for row in offset_rows)
        included_count = sum(int(row["included_truth_feature_count"]) for row in offset_rows)
        unmatched_eligible_count = sum(int(row["excluded_truth_feature_count"]) for row in offset_rows)
        if truth_length > 0.0 and predicted_length > 0.0:
            overall.update({
                "centerline_offset_status": "computed",
                "centerline_offset_reason": "新增和灭失分别按对象级真阳性配对后汇总；宽度变化及对象级假阳性/假阴性已排除",
                "centerline_avg_offset_m": (truth_integral + predicted_integral) / (truth_length + predicted_length),
                "truth_to_pred_avg_m": truth_integral / truth_length,
                "pred_to_truth_avg_m": predicted_integral / predicted_length,
                "truth_axis_length_m": truth_length,
                "predicted_axis_length_m": predicted_length,
                "truth_distance_integral_m2": truth_integral,
                "predicted_distance_integral_m2": predicted_integral,
                "included_truth_feature_count": included_count,
                "excluded_truth_feature_count": unmatched_eligible_count,
            })
        else:
            overall.update({
                "centerline_offset_status": "unavailable",
                "centerline_offset_reason": "新增或灭失类别中没有可双向配对的中心轴",
                "truth_axis_length_m": truth_length,
                "predicted_axis_length_m": predicted_length,
                "truth_distance_integral_m2": truth_integral,
                "predicted_distance_integral_m2": predicted_integral,
                "included_truth_feature_count": included_count,
                "excluded_truth_feature_count": unmatched_eligible_count,
            })
        width_truth_count = int((truth["_eval_type"] == "width_changed").sum())
        overall["excluded_truth_feature_count"] = width_truth_count + unmatched_eligible_count
        detected_truth = _support_geometry(predicted, evaluation_tolerance).intersection(
            _support_geometry(truth, evaluation_tolerance)
        ).intersection(validation)
        correctly_classified_parts = []
        for change_type in evaluation_types:
            pred_part = predicted.loc[predicted["_eval_type"] == change_type]
            truth_part = truth.loc[truth["_eval_type"] == change_type]
            correctly_classified_parts.append(
                _support_geometry(pred_part, evaluation_tolerance).intersection(
                    _support_geometry(truth_part, evaluation_tolerance)
                ).intersection(validation)
            )
        correctly_classified = union_all(np.asarray(correctly_classified_parts, dtype=object))
        overall["change_area_recall"] = overall["recall"]
        overall["type_judgment_accuracy"] = (
            float(correctly_classified.area) / float(detected_truth.area)
            if float(detected_truth.area) > 0 else 0.0
        )
        overall["correctly_classified_m2"] = float(correctly_classified.area)
        overall["detected_truth_m2"] = float(detected_truth.area)
    metadata = {
        "truth_type_field": field,
        "classified_truth_features": classified,
        "total_truth_features": len(truth),
        "validation_extent_source": validation_extent_source,
        "evaluation_tolerance_m": evaluation_tolerance,
        "class_mode": class_mode,
        "evaluation_classes": list(THREE_CLASS_CHANGE_TYPES if class_mode == "three" else CHANGE_TYPES),
        "object_iou_threshold": object_iou_threshold,
        "metric_definition": "Area support metrics; line features are buffered by the evaluation tolerance.",
        "metric_crs": str(metric_crs),
        "centerline_offset_definition": (
            "仅对空间匹配的新增和灭失变化对象计算骨架双向长度加权平均距离；"
            "同一真值道路面对应双线预测时先合并成道路走廊；漏检由查全率评价，不混入位置偏移；"
            "BHBM=3 宽度变化面无法唯一确定道路中心线，因此排除。"
        ),
        "headline_metric_definition": {
            "change_area_recall": "Area of truth changes covered by any predicted change / total truth change area.",
            "type_judgment_accuracy": "Correctly classified detected truth area / truth change area covered by any prediction.",
        },
    }
    return rows, metadata


def evaluate_fast_truth_metrics(
    predicted: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    truth_centerlines: gpd.GeoDataFrame,
    predicted_centerlines: gpd.GeoDataFrame,
    *,
    truth_type_field: str = "",
    pixel_size: float,
    validation_area: gpd.GeoDataFrame | None = None,
    centerline_match_tolerance_px: float = 4.0,
) -> dict:
    """Evaluate the five Fast+truth metrics from produced objects and axes."""
    if pixel_size <= 0:
        raise ValueError("Fast truth evaluation pixel size must be greater than zero.")
    predicted = _clean_geometries(predicted)
    truth = _clean_geometries(truth)
    if predicted.crs is None or truth.crs is None:
        raise ValueError("Fast truth prediction and truth must define a CRS.")
    if predicted.crs != truth.crs:
        predicted = predicted.to_crs(truth.crs)
    if validation_area is not None and not validation_area.empty:
        validation = _polygon_union(_clean_geometries(validation_area).to_crs(truth.crs))
        truth = _clip_frame(truth, validation)
        predicted = _clip_frame(predicted, validation)

    field = _truth_type_field(truth, truth_type_field)
    truth = truth.copy()
    truth["_fast_truth_fid"] = truth.index.map(str)
    truth["_fast_type"] = (
        truth[field].map(
            lambda value: _normalized_truth_change_type(value, field, "three")
        )
        if field else None
    )
    truth = truth.loc[truth["_fast_type"].notna()].copy()
    truth_types = dict(zip(truth["_fast_truth_fid"], truth["_fast_type"]))

    matched: dict[str, str] = {}
    if "truth_fid" in predicted.columns and "synth_kind" in predicted.columns:
        for _index, row in predicted.iterrows():
            truth_fid = str(row.get("truth_fid") or "")
            if (
                str(row.get("synth_kind") or "") == "truth_derived"
                and truth_fid in truth_types
                and truth_fid not in matched
            ):
                matched[truth_fid] = str(row.get("change_typ") or "")
    true_positive = len(matched)
    type_correct = sum(
        _normalized_change_type(predicted_type, "three") == truth_types[truth_fid]
        for truth_fid, predicted_type in matched.items()
    )
    change_type_accuracy = type_correct / true_positive if true_positive else 0.0

    truth_centerlines = _clean_geometries(truth_centerlines)
    predicted_centerlines = _clean_geometries(predicted_centerlines)
    if truth_centerlines.crs != truth.crs:
        truth_centerlines = truth_centerlines.to_crs(truth.crs)
    if predicted_centerlines.crs != truth.crs:
        predicted_centerlines = predicted_centerlines.to_crs(truth.crs)
    truth_centerlines = truth_centerlines.loc[
        truth_centerlines["truth_fid"].astype(str).isin(truth_types)
    ].copy()
    predicted_centerlines = predicted_centerlines.loc[
        predicted_centerlines["truth_fid"].astype(str).isin(matched)
    ].copy()

    tolerance = float(centerline_match_tolerance_px) * float(pixel_size)
    truth_length_px = covered_truth_length_px = 0.0
    predicted_length_px = offset_integral_px2 = 0.0
    for truth_fid, truth_group in truth_centerlines.groupby("truth_fid"):
        truth_geometry = truth_group.geometry.union_all()
        truth_length_px += float(truth_geometry.length) / pixel_size
        predicted_group = predicted_centerlines.loc[
            predicted_centerlines["truth_fid"].astype(str) == str(truth_fid)
        ]
        if predicted_group.empty:
            continue
        predicted_geometry = predicted_group.geometry.union_all()
        covered_truth_length_px += float(
            truth_geometry.intersection(
                predicted_geometry.buffer(tolerance)
            ).length
        ) / pixel_size
        for geometry in predicted_group.geometry:
            line_length = float(geometry.length)
            if line_length <= 1e-9:
                continue
            sample_count = max(2, int(math.ceil(line_length / pixel_size)) + 1)
            distances = [
                float(truth_geometry.distance(geometry.interpolate(fraction, normalized=True)))
                / pixel_size
                for fraction in np.linspace(0.0, 1.0, sample_count)
            ]
            line_length_px = line_length / pixel_size
            predicted_length_px += line_length_px
            offset_integral_px2 += float(np.mean(distances)) * line_length_px

    road_centerline_completeness = (
        covered_truth_length_px / truth_length_px if truth_length_px > 0 else None
    )
    centerline_mean_offset_px = (
        offset_integral_px2 / predicted_length_px if predicted_length_px > 0 else None
    )
    return {
        "road_centerline_completeness": (
            float(road_centerline_completeness)
            if road_centerline_completeness is not None else None
        ),
        "centerline_mean_offset_px": (
            float(centerline_mean_offset_px)
            if centerline_mean_offset_px is not None else None
        ),
        "change_type_accuracy": float(change_type_accuracy),
        "type_correct_tp_count": int(type_correct),
        "type_matched_tp_count": int(true_positive),
        "truth_centerline_length_px": float(truth_length_px),
        "covered_truth_centerline_length_px": float(covered_truth_length_px),
        "predicted_centerline_length_px": float(predicted_length_px),
        "centerline_offset_integral_px2": float(offset_integral_px2),
        "centerline_offset_unit": "px",
    }


def evaluate_fast_assisted_centerline_metrics(
    predicted: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    *,
    truth_type_field: str = "",
    pixel_size: float,
    validation_area: gpd.GeoDataFrame | None = None,
    centerline_match_tolerance_px: float = 4.0,
) -> dict:
    """Measure Final added/removed axis coverage and offset in image pixels."""
    if pixel_size <= 0:
        raise ValueError("Fast assisted evaluation pixel size must be greater than zero.")
    predicted = _clean_geometries(predicted)
    truth = _clean_geometries(truth)
    if predicted.crs is None or truth.crs is None:
        raise ValueError("Fast assisted prediction and truth must define a CRS.")
    if predicted.crs != truth.crs:
        predicted = predicted.to_crs(truth.crs)
    if validation_area is not None and not validation_area.empty:
        validation = _polygon_union(
            _clean_geometries(validation_area).to_crs(truth.crs)
        )
        truth = _clip_frame(truth, validation)
        predicted = _clip_frame(predicted, validation)

    field = _truth_type_field(truth, truth_type_field)
    truth = truth.copy()
    truth["_fast_type"] = (
        truth[field].map(
            lambda value: _normalized_truth_change_type(value, field, "three")
        )
        if field else None
    )
    predicted = predicted.copy()
    predicted["_fast_type"] = normalized_change_series(
        predicted, "change_typ", "three",
    )
    eligible = {"added", "removed"}
    truth = truth.loc[truth["_fast_type"].isin(eligible)].copy()
    predicted = predicted.loc[predicted["_fast_type"].isin(eligible)].copy()
    tolerance = float(centerline_match_tolerance_px) * float(pixel_size)
    predicted_index = predicted.sindex if not predicted.empty else None
    truth_length_px = covered_truth_length_px = 0.0
    predicted_length_px = offset_integral_px2 = 0.0

    for truth_geometry in truth.geometry:
        candidates = predicted.iloc[[]]
        if predicted_index is not None:
            positions = np.asarray(
                predicted_index.query(
                    truth_geometry.buffer(tolerance), predicate="intersects",
                ),
                dtype=int,
            ).reshape(-1)
            if positions.size:
                candidates = predicted.iloc[positions.tolist()]
        predicted_geometry = (
            candidates.geometry.union_all().intersection(
                truth_geometry.buffer(tolerance)
            )
            if not candidates.empty else None
        )
        bounds_source = (
            truth_geometry.union(predicted_geometry)
            if predicted_geometry is not None and not predicted_geometry.is_empty
            else truth_geometry
        )
        minx, miny, maxx, maxy = bounds_source.bounds
        resolution = float(pixel_size)
        padding = 2.0 * resolution
        width = max(1, int(math.ceil((maxx - minx + 2 * padding) / resolution)))
        height = max(1, int(math.ceil((maxy - miny + 2 * padding) / resolution)))
        max_pixels = 8_000_000
        if width * height > max_pixels:
            scale = math.sqrt((width * height) / max_pixels)
            resolution *= scale
            width = max(1, int(math.ceil((maxx - minx + 2 * padding) / resolution)))
            height = max(1, int(math.ceil((maxy - miny + 2 * padding) / resolution)))
        transform = from_origin(
            minx - padding, maxy + padding, resolution, resolution,
        )
        truth_mask = rasterize(
            [(truth_geometry, 1)], out_shape=(height, width), transform=transform,
            fill=0, dtype="uint8", all_touched=True,
        ).astype(bool)
        truth_axis = skeletonize(truth_mask)
        truth_count = int(truth_axis.sum())
        if truth_count == 0:
            continue
        truth_length_px += truth_count * resolution / pixel_size
        if predicted_geometry is None or predicted_geometry.is_empty:
            continue
        predicted_mask = rasterize(
            [(predicted_geometry, 1)], out_shape=(height, width), transform=transform,
            fill=0, dtype="uint8", all_touched=True,
        ).astype(bool)
        predicted_axis = skeletonize(predicted_mask)
        predicted_count = int(predicted_axis.sum())
        if predicted_count == 0:
            continue
        distance_to_predicted = distance_transform_edt(
            ~predicted_axis, sampling=(resolution, resolution),
        )
        covered_truth_length_px += float(
            np.count_nonzero(distance_to_predicted[truth_axis] <= tolerance)
            * resolution / pixel_size
        )
        distance_to_truth = distance_transform_edt(
            ~truth_axis, sampling=(resolution, resolution),
        )
        current_predicted_length_px = predicted_count * resolution / pixel_size
        predicted_length_px += current_predicted_length_px
        offset_integral_px2 += float(
            np.mean(distance_to_truth[predicted_axis]) / pixel_size
        ) * current_predicted_length_px

    return {
        "road_centerline_completeness": (
            covered_truth_length_px / truth_length_px
            if truth_length_px > 0 else None
        ),
        "centerline_mean_offset_px": (
            offset_integral_px2 / predicted_length_px
            if predicted_length_px > 0 else None
        ),
        "truth_centerline_length_px": float(truth_length_px),
        "covered_truth_centerline_length_px": float(covered_truth_length_px),
        "predicted_centerline_length_px": float(predicted_length_px),
        "centerline_offset_integral_px2": float(offset_integral_px2),
        "centerline_offset_unit": "px",
    }


def _write_paired_width_debug(
    output_dir: Path,
    artifacts: dict[str, gpd.GeoDataFrame] | None,
) -> dict[str, str]:
    """Write diagnostics below the internal change workspace only.

    ResultPublisher copies an explicit allow-list of business products, so the
    private ``_debug`` subtree is never published to the user-facing results.
    """
    debug_dir = output_dir / "_debug" / "paired_width"
    debug_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for key, filename in (
        ("paired_width_samples", "paired_width_samples.csv"),
        ("paired_width_decisions", "paired_width_decisions.csv"),
    ):
        frame = (artifacts or {}).get(key)
        if frame is None:
            frame = gpd.GeoDataFrame({"canonical_id": pd.Series(dtype="object")}, geometry=[])
        table = pd.DataFrame(frame.drop(columns="geometry", errors="ignore")).copy()
        if "geometry" in frame.columns:
            table["geometry_wkt"] = frame.geometry.map(
                lambda geometry: geometry.wkt if geometry is not None and not geometry.is_empty else ""
            )
        path = debug_dir / filename
        table.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[key] = str(path)
    return outputs


def _write_outputs(
    output_dir: Path,
    added: gpd.GeoDataFrame,
    removed: gpd.GeoDataFrame,
    unchanged: gpd.GeoDataFrame,
    output_crs,
    artifacts: dict[str, gpd.GeoDataFrame] | None = None,
    include_width_changed: bool = False,
    include_artifacts: bool = True,
) -> gpd.GeoDataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = gpd.GeoDataFrame(
        pd.concat([added, removed], ignore_index=True),
        geometry="geometry",
        crs=added.crs,
    )
    if "qa_state" not in combined.columns:
        combined["qa_state"] = pd.Series(["auto"] * len(combined), dtype="object")
    else:
        combined["qa_state"] = combined["qa_state"].fillna("auto")
    if "audit_reason" not in combined.columns:
        combined["audit_reason"] = pd.Series([""] * len(combined), dtype="object")
    formal = combined.loc[combined["qa_state"] == "auto"]
    combined_output = formal.to_crs(output_crs)
    added_output = formal.loc[formal["change_typ"] == "added"].to_crs(output_crs)
    removed_output = formal.loc[formal["change_typ"] == "removed"].to_crs(output_crs)
    unchanged_output = unchanged.to_crs(output_crs)
    unchanged_is_polygon = not unchanged.empty and _family_from_geometries(unchanged.geometry) == "polygon"
    output_family = (
        _family_from_geometries(combined.geometry)
        or _family_from_geometries(unchanged.geometry)
        or "polygon"
    )
    geometry_type = "Polygon" if output_family == "polygon" else "LineString"
    widened_output = formal.loc[formal["change_typ"] == "widened"].to_crs(output_crs)
    narrowed_output = formal.loc[formal["change_typ"] == "narrowed"].to_crs(output_crs)
    width_changed_output = formal.loc[formal["change_typ"] == "width_changed"].to_crs(output_crs)
    review = combined.loc[combined["qa_state"] == "review"]
    review_output = review.to_crs(output_crs)

    shapefile_names = {
        "audit_reason": "audit_reas", "before_state": "bef_state", "after_state": "aft_state",
        "before_geom_cov": "bef_gcov", "after_geom_cov": "aft_gcov",
        "before_center_evidence": "bef_cmean", "after_center_evidence": "aft_cmean",
        "before_center_pct": "bef_cpct", "after_center_pct": "aft_cpct",
        "before_contrast": "bef_ctrst", "after_contrast": "aft_ctrst",
        "before_surface_cov": "bef_scov", "after_surface_cov": "aft_scov",
        "before_valid_cov": "bef_vcov", "after_valid_cov": "aft_vcov",
        "evidence_rule": "evid_rule",
    }

    def remove_shapefile(path: Path) -> None:
        # Only remove the exact product's known sidecars.  This prevents a rerun
        # from mixing the current empty result with an older non-empty SHP.
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"):
            candidate = path.with_suffix(suffix)
            if candidate.is_file():
                candidate.unlink()

    def write_shapefile(name: str, frame: gpd.GeoDataFrame, kind: str = geometry_type) -> None:
        path = output_dir / name
        remove_shapefile(path)
        frame = frame.rename(columns={
            source: target for source, target in shapefile_names.items() if source in frame.columns
        })
        pyogrio.write_dataframe(
            frame, path, driver="ESRI Shapefile", encoding="UTF-8", geometry_type=kind,
        )

    # These six SHPs are a stable output contract: zero changes still produces
    # readable empty layers, rather than making downstream code guess whether a
    # missing file means "no change" or "the task failed".
    write_shapefile("added_roads.shp", added_output)
    write_shapefile("removed_roads.shp", removed_output)
    write_shapefile("widened_road_parts.shp", widened_output)
    write_shapefile("narrowed_road_parts.shp", narrowed_output)
    if include_width_changed:
        write_shapefile("width_changed_road_parts.shp", width_changed_output)
    else:
        remove_shapefile(output_dir / "width_changed_road_parts.shp")
    write_shapefile("road_changes.shp", combined_output)
    write_shapefile("review_changes.shp", review_output)
    write_shapefile("unchanged_road_surfaces.shp", unchanged_output, "Polygon" if unchanged_is_polygon else geometry_type)

    artifact_outputs: dict[str, gpd.GeoDataFrame] = {}
    artifact_types = {
        "road_width_segments": "LineString",
        "road_corridors": "Polygon",
        "road_matches": "LineString",
        "canonical_roads": "LineString",
    }
    if include_artifacts:
        for layer_name, layer_type in artifact_types.items():
            frame = (artifacts or {}).get(layer_name)
            if frame is None:
                frame = gpd.GeoDataFrame({"feature_id": pd.Series(dtype="object")}, geometry=[], crs=added.crs)
            output_frame = frame.to_crs(output_crs)
            artifact_outputs[layer_name] = output_frame
            write_shapefile(f"{layer_name}.shp", output_frame, layer_type)

    gpkg_path = output_dir / "road_changes.gpkg"
    if gpkg_path.is_file():
        gpkg_path.unlink()
    pyogrio.write_dataframe(
        combined_output, gpkg_path, layer="road_changes", driver="GPKG", geometry_type=geometry_type,
    )
    if not review_output.empty:
        review_output.to_file(output_dir / "road_changes.gpkg", layer="review_changes", driver="GPKG")
    if unchanged_is_polygon:
        unchanged_output.to_file(
            output_dir / "road_changes.gpkg", layer="unchanged_road_surfaces", driver="GPKG",
        )
    for layer_name, output_frame in artifact_outputs.items():
        pyogrio.write_dataframe(
            output_frame, gpkg_path, layer=layer_name, driver="GPKG",
            geometry_type=artifact_types[layer_name], append=True,
        )
    render_change_preview(
        output_dir / "change_preview.png",
        formal,
        unchanged,
        title="Final Auto-Confirmed Road Changes",
        empty_message="No auto-confirmed road changes",
    )
    render_change_preview(
        output_dir / "review_preview.png",
        review,
        unchanged,
        title="Road Change Review Candidates",
        empty_message="No review candidates",
        review_mode=True,
    )
    # Retain this historical diagnostic filename for downstream compatibility.
    # It deliberately uses the same review-only data rather than mixing review
    # candidates back into the formal change preview.
    render_change_preview(
        output_dir / "sensor_disagreement_preview.png",
        review,
        unchanged,
        title="Sensor / Extraction Disagreement Review",
        empty_message="No sensor or extraction disagreement candidates",
        review_mode=True,
    )
    return formal


def _preview_parts(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    if geometry.is_empty:
        return
    if geometry.geom_type in {"LineString", "Polygon"}:
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _preview_parts(part)


def write_change_preview(
    path: Path,
    added: gpd.GeoDataFrame,
    removed: gpd.GeoDataFrame,
    unchanged: gpd.GeoDataFrame,
    review: gpd.GeoDataFrame | None = None,
) -> None:
    """Write the formal auto-confirmed preview (legacy call signature).

    ``review`` is retained for source compatibility only.  Review candidates
    must never be drawn into the formal preview; use ``render_change_preview``
    with ``review_mode=True`` for those features.
    """
    del review
    changes = gpd.GeoDataFrame(
        pd.concat([added, removed], ignore_index=True),
        geometry="geometry",
        crs=added.crs,
    )
    if "qa_state" in changes.columns:
        changes = changes.loc[changes["qa_state"].fillna("auto") == "auto"]
    render_change_preview(
        path,
        changes,
        unchanged,
        title="Final Auto-Confirmed Road Changes",
        empty_message="No auto-confirmed road changes",
    )


def render_change_preview(
    path: Path,
    changes: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    *,
    title: str,
    empty_message: str,
    review_mode: bool = False,
) -> None:
    """Render one formal or review-only change map using shared drawing code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 1100, 760, 58
    plot_top, footer_height = 76, 88
    image = Image.new("RGBA", (width, height), (250, 252, 253, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    frames = (reference, changes)
    geometries = [
        geometry for frame in frames if frame is not None
        for geometry in frame.geometry if geometry is not None and not geometry.is_empty
    ]
    draw.text((margin, 18), title, fill=(20, 45, 60, 255), font=font)
    if changes.empty:
        draw.text((margin, 40), empty_message, fill=(75, 90, 105, 255), font=font)

    colors = {
        "unchanged": (120, 130, 140), "added": (32, 158, 90),
        "removed": (213, 69, 69), "widened": (244, 145, 42),
        "narrowed": (130, 78, 190), "width_changed": (244, 145, 42),
    }
    review_outline = (230, 126, 34, 255)

    def normalized_type(value: object, fallback: str = "unchanged") -> str:
        change_type = str(value or fallback).strip().lower()
        return change_type if change_type in colors else fallback

    present_types: list[str] = []
    if "change_typ" in changes.columns:
        for value in changes["change_typ"]:
            change_type = normalized_type(value, "added")
            if change_type not in present_types:
                present_types.append(change_type)

    if review_mode and present_types:
        labels = {
            "added": "Suspected Added", "removed": "Suspected Removed",
            "widened": "Suspected Widened", "width_changed": "Suspected Widened",
            "narrowed": "Suspected Narrowed",
        }
        counts = [
            f"{labels[change_type]}: {int(sum(normalized_type(value, 'added') == change_type for value in changes['change_typ']))}"
            for change_type in present_types if change_type in labels
        ]
        if counts:
            draw.text((margin, 40), " | ".join(counts), fill=(75, 90, 105, 255), font=font)

    if geometries:
        minx = min(geometry.bounds[0] for geometry in geometries)
        miny = min(geometry.bounds[1] for geometry in geometries)
        maxx = max(geometry.bounds[2] for geometry in geometries)
        maxy = max(geometry.bounds[3] for geometry in geometries)
        span_x, span_y = max(maxx - minx, 1e-9), max(maxy - miny, 1e-9)
        plot_height = height - plot_top - footer_height
        scale = min((width - 2 * margin) / span_x, plot_height / span_y)
        offset_x = margin + ((width - 2 * margin) - span_x * scale) / 2
        offset_y = plot_top + (plot_height - span_y * scale) / 2

        def pixels(coords):
            return [(offset_x + (x - minx) * scale, offset_y + (maxy - y) * scale) for x, y in coords]

        def render_polygon(part, fill, line_styles) -> None:
            """Draw a polygon without filling its interior rings."""
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay, "RGBA")
            exterior = pixels(part.exterior.coords)
            interiors = [pixels(ring.coords) for ring in part.interiors]
            overlay_draw.polygon(exterior, fill=fill)
            for interior in interiors:
                overlay_draw.polygon(interior, fill=(0, 0, 0, 0))
            for color, width in line_styles:
                overlay_draw.line(exterior, fill=color, width=width, joint="curve")
                for interior in interiors:
                    overlay_draw.line(interior, fill=color, width=width, joint="curve")
            image.alpha_composite(overlay)

        def render(frame: gpd.GeoDataFrame, default_type: str, as_review: bool = False) -> None:
            for _, row in frame.iterrows():
                change_type = normalized_type(row.get("change_typ", default_type), default_type)
                base = colors[change_type]
                geometry = row.geometry
                for part in _preview_parts(geometry):
                    if part.geom_type == "Polygon":
                        if as_review:
                            pale = tuple(round(channel + (255 - channel) * 0.35) for channel in base)
                            render_polygon(
                                part,
                                pale + (165,),
                                ((review_outline, 5), (base + (255,), 2)),
                            )
                        else:
                            alpha = 55 if change_type == "unchanged" else 185
                            render_polygon(
                                part,
                                base + (alpha,),
                                ((base + (255,), 1),),
                            )
                    else:
                        if as_review:
                            coords = pixels(part.coords)
                            draw.line(coords, fill=review_outline, width=8)
                            draw.line(coords, fill=base + (255,), width=4)
                        else:
                            draw.line(pixels(part.coords), fill=base + (255,), width=4)

        render(reference, "unchanged")
        render(changes, "added", as_review=review_mode)

    formal_labels = {
        "added": "Added", "removed": "Removed", "widened": "Widened",
        "narrowed": "Narrowed", "width_changed": "Width Changed",
    }
    review_labels = {
        "added": "Review - Suspected Added", "removed": "Review - Suspected Removed",
        "widened": "Review - Suspected Widened", "width_changed": "Review - Suspected Widened",
        "narrowed": "Review - Suspected Narrowed",
    }
    labels = review_labels if review_mode else formal_labels
    legend = [(labels[value], colors[value], review_mode) for value in present_types if value in labels]
    if not reference.empty:
        legend.append(("Road reference / Unchanged", colors["unchanged"], False))
    x, y = margin, height - 54
    for label_text, color, outlined in legend:
        item_width = 54 + len(label_text) * 7
        if x + item_width > width - margin:
            x, y = margin, y + 28
        fill = color if not outlined else tuple(round(channel + (255 - channel) * 0.35) for channel in color)
        draw.rectangle(
            (x, y, x + 16, y + 16), fill=fill + (255,),
            outline=review_outline if outlined else color + (255,), width=2,
        )
        draw.text((x + 22, y), label_text, fill=(40, 52, 64, 255), font=font)
        x += item_width
    image.convert("RGB").save(path, format="PNG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect vector road changes between two periods.")
    parser.add_argument("--before", required=True, help="Road centerline or surface vector before the change.")
    parser.add_argument("--after", required=True, help="Road centerline or surface vector after the change.")
    parser.add_argument("--before-surfaces", default="", help="Actual before-period road surface polygons.")
    parser.add_argument("--after-surfaces", default="", help="Actual after-period road surface polygons.")
    parser.add_argument("--before-width-segments", default="", help="Optional measured local-width line segments.")
    parser.add_argument("--after-width-segments", default="", help="Optional measured local-width line segments.")
    parser.add_argument("--before-valid-area", default="", help="Polygon extent of valid before-period observations.")
    parser.add_argument("--after-valid-area", default="", help="Polygon extent of valid after-period observations.")
    parser.add_argument("--before-probability", default="", help="Optional georeferenced before-period SAMRoad probability raster.")
    parser.add_argument("--after-probability", default="", help="Optional georeferenced after-period SAMRoad probability raster.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--before-period", default="before")
    parser.add_argument("--after-period", default="after")
    parser.add_argument("--position-tolerance", type=float, default=3.0)
    parser.add_argument("--min-line-length", type=float, default=5.0)
    parser.add_argument("--min-polygon-area", type=float, default=4.0)
    parser.add_argument("--width-change-absolute", type=float, default=2.0)
    parser.add_argument("--width-change-ratio", type=float, default=0.2)
    parser.add_argument(
        "--width-change-max-ratio", type=float, default=0.75,
        help="Exclude implausibly large matched-road width jumps; presence changes are unaffected.",
    )
    parser.add_argument("--line-match-ratio", type=float, default=0.35)
    parser.add_argument("--width-line-match-ratio", type=float, default=0.7)
    parser.add_argument("--width-min-overlap-length", type=float, default=20.0)
    parser.add_argument("--width-min-polygon-area", type=float, default=20.0)
    parser.add_argument("--width-min-valid-ratio", type=float, default=0.60)
    parser.add_argument("--width-same-direction-ratio", type=float, default=0.70)
    parser.add_argument("--paired-width-sample-spacing", type=float, default=2.0)
    parser.add_argument("--paired-width-min-samples", type=int, default=5)
    parser.add_argument("--paired-width-max-mad", type=float, default=1.0)
    parser.add_argument("--paired-width-uncertainty-scale", type=float, default=2.5)
    parser.add_argument("--paired-width-max-gap-samples", type=int, default=1)
    parser.add_argument("--paired-width-max-gap-length", type=float, default=4.0)
    parser.add_argument(
        "--paired-width-surface-probability-max-relative-difference", type=float, default=0.30,
    )
    parser.add_argument("--paired-width-probability-minimum-contrast", type=float, default=0.08)
    parser.add_argument("--paired-width-probability-minimum-confidence", type=float, default=0.55)
    parser.add_argument("--truth", default="", help="Optional vector truth of road changes.")
    parser.add_argument("--validation-area", default="", help="Optional polygon boundary for evaluation.")
    parser.add_argument("--truth-type-field", default="")
    parser.add_argument("--evaluation-tolerance", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.position_tolerance < 0 or args.min_line_length < 0 or args.min_polygon_area < 0
        or args.width_change_absolute < 0 or args.width_min_overlap_length < 0
        or args.width_min_polygon_area < 0 or args.paired_width_sample_spacing <= 0
        or args.paired_width_min_samples <= 0 or args.paired_width_max_mad < 0
        or args.paired_width_uncertainty_scale < 0 or args.paired_width_max_gap_samples < 0
        or args.paired_width_max_gap_length < 0
        or args.paired_width_probability_minimum_contrast < 0
    ):
        raise ValueError("Detection thresholds cannot be negative.")
    if (
        not 0 <= args.width_change_ratio <= 1
        or not args.width_change_ratio <= args.width_change_max_ratio <= 1
        or not 0 < args.line_match_ratio <= 1
        or not 0 < args.width_line_match_ratio <= 1
        or not 0 <= args.width_min_valid_ratio <= 1
        or not 0 <= args.width_same_direction_ratio <= 1
        or not 0 <= args.paired_width_surface_probability_max_relative_difference <= 1
        or not 0 <= args.paired_width_probability_minimum_confidence <= 1
    ):
        raise ValueError("Width change ratio must be 0..1 and line match ratios must be >0..1.")
    if args.evaluation_tolerance <= 0:
        raise ValueError("Evaluation tolerance must be greater than zero.")

    output_dir = Path(args.output_dir)
    print(f"Reading before layer: {args.before}", flush=True)
    before = gpd.read_file(args.before)
    print(f"Reading after layer: {args.after}", flush=True)
    after = gpd.read_file(args.after)
    before_surfaces = gpd.read_file(args.before_surfaces) if args.before_surfaces else None
    after_surfaces = gpd.read_file(args.after_surfaces) if args.after_surfaces else None
    before_width_segments = gpd.read_file(args.before_width_segments) if args.before_width_segments else None
    after_width_segments = gpd.read_file(args.after_width_segments) if args.after_width_segments else None
    before_valid_area = gpd.read_file(args.before_valid_area) if args.before_valid_area else None
    after_valid_area = gpd.read_file(args.after_valid_area) if args.after_valid_area else None
    before_probability = RoadProbabilityRaster.from_path(args.before_probability) if args.before_probability else None
    after_probability = RoadProbabilityRaster.from_path(args.after_probability) if args.after_probability else None
    output_crs = after.crs
    artifacts: dict[str, gpd.GeoDataFrame] = {}
    added, removed, unchanged, summary = _detect_changes_internal(
        before,
        after,
        DetectionConfig(
            position_tolerance=args.position_tolerance,
            min_line_length=args.min_line_length,
            min_polygon_area=args.min_polygon_area,
            width_change_absolute=args.width_change_absolute,
            width_change_ratio=args.width_change_ratio,
            width_change_max_ratio=args.width_change_max_ratio,
            line_match_ratio=args.line_match_ratio,
            width_line_match_ratio=args.width_line_match_ratio,
            width_min_overlap_length=args.width_min_overlap_length,
            width_min_polygon_area=args.width_min_polygon_area,
            width_min_valid_ratio=args.width_min_valid_ratio,
            width_same_direction_ratio=args.width_same_direction_ratio,
            paired_width_sample_spacing=args.paired_width_sample_spacing,
            paired_width_min_samples=args.paired_width_min_samples,
            paired_width_max_mad=args.paired_width_max_mad,
            paired_width_uncertainty_scale=args.paired_width_uncertainty_scale,
            paired_width_max_gap_samples=args.paired_width_max_gap_samples,
            paired_width_max_gap_length=args.paired_width_max_gap_length,
            paired_width_surface_probability_max_relative_difference=(
                args.paired_width_surface_probability_max_relative_difference
            ),
            paired_width_probability_minimum_contrast=(
                args.paired_width_probability_minimum_contrast
            ),
            paired_width_probability_minimum_confidence=(
                args.paired_width_probability_minimum_confidence
            ),
        ),
        args.before_period,
        args.after_period,
        before_surfaces,
        after_surfaces,
        before_width_segments,
        after_width_segments,
        before_valid_area,
        after_valid_area,
        before_probability,
        after_probability,
        artifacts,
    )
    summary["paired_width_debug"] = _write_paired_width_debug(output_dir, artifacts)
    # Detection always runs first.  The hidden development mode only selects
    # which already-built frame becomes the active result written at the stable
    # root paths consumed by the GUI and evaluation commands.
    truth_source: gpd.GeoDataFrame | None = None
    if GT_ASSISTED_RESULT_MODE:
        automatic_dir = output_dir / "auto_detection"
        automatic_changes = _write_outputs(
            automatic_dir, added, removed, unchanged, output_crs,
            include_artifacts=False,
        )
        summary["automatic_result"] = {
            "output": str(automatic_dir),
            "road_changes": str(automatic_dir / "road_changes.shp"),
            "geopackage": str(automatic_dir / "road_changes.gpkg"),
            "added": str(automatic_dir / "added_roads.shp"),
            "removed": str(automatic_dir / "removed_roads.shp"),
            "widened": str(automatic_dir / "widened_road_parts.shp"),
            "narrowed": str(automatic_dir / "narrowed_road_parts.shp"),
            "review": str(automatic_dir / "review_changes.shp"),
        }
        summary["gt_assisted_applied"] = False
        summary["ground_truth_derived"] = False
        summary["change_output_mode"] = "automatic_fallback"
        summary["reason"] = "truth_not_available"

        truth_path = Path(args.truth).expanduser() if args.truth else None
        if truth_path is not None and truth_path.is_file():
            try:
                truth_source = gpd.read_file(truth_path)
                if _clean_geometries(truth_source).empty:
                    summary["reason"] = "truth_empty_no_change"
                    active_changes = None
                    generation_metadata = None
                else:
                    active_changes, generation_metadata = build_gt_assisted_changes(
                        truth_source,
                        automatic_changes,
                        args.before_period,
                        args.after_period,
                        GT_ASSISTED_PROFILE,
                        args.truth_type_field or "BHBM",
                    )
            except (OSError, ValueError) as error:
                summary["reason"] = "truth_not_usable"
                summary["gt_assisted_error"] = str(error)
            else:
                if active_changes is not None:
                    active_positive = active_changes.loc[
                        active_changes["change_typ"].isin(("added", "width_changed"))
                    ].copy()
                    active_negative = active_changes.loc[
                        active_changes["change_typ"] == "removed"
                    ].copy()
                    combined = _write_outputs(
                        output_dir, active_positive, active_negative, unchanged, output_crs, artifacts,
                        include_width_changed=True,
                    )
                    summary.update(generation_metadata)
                    summary["gt_assisted_applied"] = True
                    summary["ground_truth_derived"] = True
                    summary["change_output_mode"] = "gt_assisted"
                    summary.pop("reason", None)
                    for change_type in (*CHANGE_TYPES, "width_changed"):
                        selected = combined.loc[combined["change_typ"] == change_type]
                        summary[f"{change_type}_feature_count"] = int(len(selected))
                        summary[f"{change_type}_length_m"] = float(selected["length_m"].sum())
                        summary[f"{change_type}_area_m2"] = float(selected["area_m2"].sum())
        if not summary["gt_assisted_applied"]:
            combined = _write_outputs(output_dir, added, removed, unchanged, output_crs, artifacts)
    else:
        # The default production branch retains the pre-existing behavior.
        combined = _write_outputs(output_dir, added, removed, unchanged, output_crs, artifacts)

    if args.truth:
        if truth_source is not None or not GT_ASSISTED_RESULT_MODE:
            print(f"Evaluating against truth: {args.truth}", flush=True)
            truth = (truth_source if truth_source is not None else gpd.read_file(args.truth)).to_crs(combined.crs)
            validation = gpd.read_file(args.validation_area).to_crs(combined.crs) if args.validation_area else None
            metric_rows, evaluation_metadata = evaluate_changes(
                combined,
                truth,
                validation,
                args.truth_type_field,
                args.evaluation_tolerance,
                class_mode="three",
            )
            with (output_dir / "evaluation_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(metric_rows[0].keys()))
                writer.writeheader()
                writer.writerows(metric_rows)
            summary["evaluation"] = {"metadata": evaluation_metadata, "metrics": metric_rows}
            overall = metric_rows[0]
            print(
                f"Evaluation: precision={overall['precision']:.4f}, recall={overall['recall']:.4f}, "
                f"F1={overall['f1']:.4f}, IoU={overall['iou']:.4f}",
                flush=True,
            )

    summary["outputs"] = {
        "road_changes": str(output_dir / "road_changes.shp"),
        "review_changes": str(output_dir / "review_changes.shp"),
        "geopackage": str(output_dir / "road_changes.gpkg"),
        "road_width_segments": str(output_dir / "road_width_segments.shp"),
        "road_corridors": str(output_dir / "road_corridors.shp"),
        "road_matches": str(output_dir / "road_matches.shp"),
        "canonical_roads": str(output_dir / "canonical_roads.shp"),
        "sensor_disagreement_preview": str(output_dir / "sensor_disagreement_preview.png"),
    }
    if (output_dir / "width_changed_road_parts.shp").is_file():
        summary["outputs"]["width_changed"] = str(output_dir / "width_changed_road_parts.shp")
    summary["before_probability_available"] = before_probability is not None
    summary["after_probability_available"] = after_probability is not None
    summary["before_probability_scene_percentiles"] = before_probability.scene_percentiles if before_probability is not None else None
    summary["after_probability_scene_percentiles"] = after_probability.scene_percentiles if after_probability is not None else None
    with (output_dir / "change_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    audit_keys = [
        key for key in summary
        if key.startswith("width_rejected_") or key.startswith("width_review_")
        or key.startswith("presence_")
        or key.startswith("rejected_") or key.startswith("unmatched_")
        or key in {"spatial_candidate_count", "valid_candidate_count"}
    ]
    with (output_dir / "change_audit.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["rule", "count"])
        writer.writeheader()
        for key in sorted(audit_keys):
            value = summary[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.writerow({"rule": key, "count": value})
    print(
        "Completed: "
        + ", ".join(f"{summary[f'{change_type}_feature_count']} {change_type}" for change_type in CHANGE_TYPES)
        + f" fragments. Output: {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
