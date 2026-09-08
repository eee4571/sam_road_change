from __future__ import annotations

"""Cross-period road-width measurement on a shared canonical axis.

The module measures both periods with the same canonical tangent/normal while
allowing each period to retain its own centreline position. Reliable road
surface sections remain primary; scene-relative probability profiles provide a
conservative fallback and are never allowed to override a conflicting surface.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely import make_valid
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from road_existence_evidence import RoadProbabilityRaster


@dataclass(frozen=True)
class PairedWidthConfig:
    sample_spacing: float = 2.0
    normal_half_length: float = 60.0
    centre_tolerance: float = 0.10
    minimum_width: float = 0.50
    absolute_change: float = 2.0
    relative_change: float = 0.20
    maximum_relative_change: float = 0.75
    minimum_valid_ratio: float = 0.60
    minimum_continuous_length: float = 20.0
    minimum_samples: int = 5
    minimum_direction_ratio: float = 0.70
    maximum_diff_mad: float = 1.0
    uncertainty_scale: float = 2.5
    maximum_gap_samples: int = 1
    maximum_gap_length: float = 4.0
    surface_confidence: float = 0.90
    surface_probability_max_relative_difference: float = 0.30
    probability_minimum_contrast: float = 0.08
    probability_threshold_fraction: float = 0.45
    probability_outer_background_fraction: float = 0.30
    probability_boundary_low_samples: int = 2
    probability_minimum_confidence: float = 0.55


@dataclass(frozen=True)
class ProbabilityWidthMeasurement:
    width: float | None
    left_distance: float | None
    right_distance: float | None
    confidence: float
    contrast: float | None
    reject_reason: str


@dataclass(frozen=True)
class PeriodWidthMeasurement:
    surface_width: float | None
    probability_width: float | None
    final_width: float | None
    width_source: str
    width_confidence: float
    probability_left_distance: float | None
    probability_right_distance: float | None
    probability_contrast: float | None
    probability_reject_reason: str
    reject_reason: str
    probability_confidence: float = 0.0


@dataclass(frozen=True)
class PairedWidthSample:
    canonical_id: str
    sample_index: int
    position_m: float
    t: float
    point: Point
    before_width: float | None
    after_width: float | None
    width_diff: float | None
    valid: bool
    reject_reason: str
    before_surface_width: float | None = None
    before_probability_width: float | None = None
    before_width_source: str = "unresolved"
    before_width_confidence: float = 0.0
    before_probability_left_distance: float | None = None
    before_probability_right_distance: float | None = None
    before_probability_contrast: float | None = None
    before_probability_confidence: float = 0.0
    before_probability_reject_reason: str = ""
    after_surface_width: float | None = None
    after_probability_width: float | None = None
    after_width_source: str = "unresolved"
    after_width_confidence: float = 0.0
    after_probability_left_distance: float | None = None
    after_probability_right_distance: float | None = None
    after_probability_contrast: float | None = None
    after_probability_confidence: float = 0.0
    after_probability_reject_reason: str = ""
    width_confidence: float = 0.0
    surface_probability_disagreement: bool = False


@dataclass(frozen=True)
class PairedWidthProfile:
    canonical_id: str
    canonical_axis: LineString
    samples: tuple[PairedWidthSample, ...]
    valid_ratio: float

    @property
    def valid_samples(self) -> tuple[PairedWidthSample, ...]:
        return tuple(sample for sample in self.samples if sample.valid)


@dataclass(frozen=True)
class PairedWidthRun:
    canonical_id: str
    sign: int
    start_m: float
    end_m: float
    axis: LineString
    samples: tuple[PairedWidthSample, ...]
    before_width: float
    after_width: float
    width_diff: float
    diff_mad: float
    uncertainty: float
    direction_ratio: float
    valid_ratio: float


def _line_parts(geometry: BaseGeometry | None) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _line_parts(part)


def _normal_at(axis: LineString, position: float, spacing: float) -> tuple[float, float] | None:
    length = float(axis.length)
    if length <= 1e-9:
        return None
    delta = min(max(0.25, spacing * 0.25), max(length * 0.25, 0.25))
    start = max(0.0, position - delta)
    end = min(length, position + delta)
    if end - start <= 1e-9:
        return None
    first = axis.interpolate(start)
    second = axis.interpolate(end)
    dx, dy = second.x - first.x, second.y - first.y
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        return None
    return -dy / norm, dx / norm


def _surface_width(
    centre: Point,
    normal: tuple[float, float],
    surface: BaseGeometry | None,
    centre_support: BaseGeometry | None,
    config: PairedWidthConfig,
) -> tuple[float | None, str]:
    if surface is None or surface.is_empty:
        return None, "surface_missing"
    if centre_support is None or not centre_support.covers(centre):
        return None, "centre_outside_surface"
    nx, ny = normal
    half = max(config.normal_half_length, config.minimum_width)
    probe = LineString([
        (centre.x - nx * half, centre.y - ny * half),
        (centre.x + nx * half, centre.y + ny * half),
    ])
    parts = list(_line_parts(surface.intersection(probe)))
    if not parts:
        return None, "surface_cross_section_empty"
    nearby = [part for part in parts if part.distance(centre) <= config.centre_tolerance]
    if not nearby:
        return None, "surface_cross_section_disconnected"
    section = min(nearby, key=lambda part: part.distance(centre))
    width = float(section.length)
    if width < config.minimum_width:
        return None, "surface_cross_section_too_short"
    probe_start, probe_end = Point(probe.coords[0]), Point(probe.coords[-1])
    if section.distance(probe_start) <= 1e-6 or section.distance(probe_end) <= 1e-6:
        return None, "surface_cross_section_truncated"
    return width, ""


def _first_probability_boundary(
    distances: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    required_low_samples: int,
) -> tuple[float | None, str]:
    order = np.argsort(np.abs(distances))
    ordered_distance = np.abs(distances[order])
    ordered_probability = probabilities[order]
    ordered_valid = valid[order]
    if not bool(ordered_valid[0]) or float(ordered_probability[0]) < threshold:
        return None, "probability_center_below_threshold"
    last_high_distance = float(ordered_distance[0])
    last_high_probability = float(ordered_probability[0])
    low_start = -1
    low_count = 0
    for index in range(1, len(order)):
        if not bool(ordered_valid[index]):
            return None, "probability_invalid_before_boundary"
        probability = float(ordered_probability[index])
        if probability >= threshold:
            last_high_distance = float(ordered_distance[index])
            last_high_probability = probability
            low_start = -1
            low_count = 0
            continue
        if low_start < 0:
            low_start = index
        low_count += 1
        if low_count < required_low_samples:
            continue
        low_distance = float(ordered_distance[low_start])
        low_probability = float(ordered_probability[low_start])
        denominator = last_high_probability - low_probability
        fraction = (
            float(np.clip((last_high_probability - threshold) / denominator, 0.0, 1.0))
            if denominator > 1e-9 else 0.5
        )
        boundary = last_high_distance + fraction * (low_distance - last_high_distance)
        return boundary, ""
    return None, "probability_boundary_not_found"


def _probability_width(
    centre: Point,
    normal: tuple[float, float],
    probability: RoadProbabilityRaster | None,
    geometry_crs,
    config: PairedWidthConfig,
) -> ProbabilityWidthMeasurement:
    if probability is None:
        return ProbabilityWidthMeasurement(None, None, None, 0.0, None, "probability_missing")
    try:
        profile = probability.sample_cross_section(
            centre,
            normal,
            geometry_crs,
            search_radius=config.normal_half_length,
        )
    except (TypeError, ValueError):
        return ProbabilityWidthMeasurement(None, None, None, 0.0, None, "probability_profile_invalid")
    distances = np.asarray(profile["distance"], dtype=np.float64)
    values = np.asarray(profile["probability"], dtype=np.float64)
    valid = np.asarray(profile["valid_mask"], dtype=bool) & np.isfinite(values)
    if distances.size < 5 or not bool(valid.any()):
        return ProbabilityWidthMeasurement(None, None, None, 0.0, None, "probability_profile_empty")
    radius = float(config.normal_half_length)
    outer_start = radius * (1.0 - config.probability_outer_background_fraction)
    left_background = values[valid & (distances <= -outer_start)]
    right_background = values[valid & (distances >= outer_start)]
    if left_background.size == 0 or right_background.size == 0:
        return ProbabilityWidthMeasurement(None, None, None, 0.0, None, "probability_background_unavailable")
    sample_step = max(float(profile["sample_step"]), 1e-6)
    center_radius = max(1.5 * sample_step, 0.5 * config.minimum_width)
    center_values = values[valid & (np.abs(distances) <= center_radius)]
    if center_values.size == 0:
        return ProbabilityWidthMeasurement(None, None, None, 0.0, None, "probability_center_unavailable")
    center_probability = float(np.median(center_values))
    left_background_value = float(np.median(left_background))
    right_background_value = float(np.median(right_background))
    background = max(left_background_value, right_background_value)
    contrast = center_probability - background
    if contrast < config.probability_minimum_contrast:
        return ProbabilityWidthMeasurement(None, None, None, 0.0, contrast, "probability_contrast_too_low")
    threshold = background + config.probability_threshold_fraction * contrast
    left_mask = distances <= 0
    right_mask = distances >= 0
    left, left_reason = _first_probability_boundary(
        distances[left_mask], values[left_mask], valid[left_mask], threshold,
        max(1, config.probability_boundary_low_samples),
    )
    right, right_reason = _first_probability_boundary(
        distances[right_mask], values[right_mask], valid[right_mask], threshold,
        max(1, config.probability_boundary_low_samples),
    )
    if left is None or right is None:
        reason = left_reason if left is None else right_reason
        return ProbabilityWidthMeasurement(None, left, right, 0.0, contrast, reason)
    width = float(left + right)
    if width < config.minimum_width:
        return ProbabilityWidthMeasurement(None, left, right, 0.0, contrast, "probability_width_too_short")
    center_rank = probability.percentile_rank(center_probability)
    contrast_scale = max(3.0 * config.probability_minimum_contrast, 0.15)
    contrast_score = float(np.clip(contrast / contrast_scale, 0.0, 1.0))
    rank_score = (
        float(np.clip((center_rank - 0.50) / 0.45, 0.0, 1.0))
        if center_rank is not None else 0.0
    )
    balance_score = float(np.clip(1.0 - abs(left - right) / max(width, 1e-9), 0.0, 1.0))
    confidence = 0.50 * contrast_score + 0.30 * rank_score + 0.20 * balance_score
    if confidence < config.probability_minimum_confidence:
        return ProbabilityWidthMeasurement(
            None, left, right, confidence, contrast, "probability_confidence_too_low",
        )
    return ProbabilityWidthMeasurement(width, left, right, confidence, contrast, "")


def _measure_period_width(
    centre: Point,
    normal: tuple[float, float],
    surface: BaseGeometry | None,
    centre_support: BaseGeometry | None,
    probability: RoadProbabilityRaster | None,
    geometry_crs,
    config: PairedWidthConfig,
) -> PeriodWidthMeasurement:
    surface_width, surface_reason = _surface_width(
        centre, normal, surface, centre_support, config,
    )
    probability_result = _probability_width(
        centre, normal, probability, geometry_crs, config,
    )
    probability_width = probability_result.width
    if surface_width is not None and probability_width is not None:
        relative_difference = abs(surface_width - probability_width) / max(
            surface_width, probability_width, config.minimum_width,
        )
        if relative_difference > config.surface_probability_max_relative_difference:
            return PeriodWidthMeasurement(
                surface_width, probability_width, None, "unresolved", 0.0,
                probability_result.left_distance, probability_result.right_distance,
                probability_result.contrast, probability_result.reject_reason,
                "surface_probability_disagreement",
                probability_confidence=probability_result.confidence,
            )
        surface_weight = config.surface_confidence
        probability_weight = probability_result.confidence
        final_width = (
            surface_weight * surface_width + probability_weight * probability_width
        ) / max(surface_weight + probability_weight, 1e-9)
        confidence = min(1.0, max(surface_weight, probability_weight) + 0.05)
        return PeriodWidthMeasurement(
            surface_width, probability_width, final_width, "fused", confidence,
            probability_result.left_distance, probability_result.right_distance,
            probability_result.contrast, probability_result.reject_reason, "",
            probability_confidence=probability_result.confidence,
        )
    if surface_width is not None:
        return PeriodWidthMeasurement(
            surface_width, probability_width, surface_width, "surface", config.surface_confidence,
            probability_result.left_distance, probability_result.right_distance,
            probability_result.contrast, probability_result.reject_reason, "",
            probability_confidence=probability_result.confidence,
        )
    if probability_width is not None:
        return PeriodWidthMeasurement(
            surface_width, probability_width, probability_width, "probability",
            probability_result.confidence,
            probability_result.left_distance, probability_result.right_distance,
            probability_result.contrast, probability_result.reject_reason, "",
            probability_confidence=probability_result.confidence,
        )
    reasons = [reason for reason in (surface_reason, probability_result.reject_reason) if reason]
    return PeriodWidthMeasurement(
        surface_width, probability_width, None, "unresolved", 0.0,
        probability_result.left_distance, probability_result.right_distance,
        probability_result.contrast, probability_result.reject_reason,
        ";".join(reasons) or "width_unresolved",
        probability_confidence=probability_result.confidence,
    )


def measure_paired_width_profile(
    canonical_id: str,
    canonical_axis: LineString,
    before_part: LineString,
    after_part: LineString,
    before_surface: BaseGeometry | None,
    after_surface: BaseGeometry | None,
    config: PairedWidthConfig = PairedWidthConfig(),
    before_probability: RoadProbabilityRaster | None = None,
    after_probability: RoadProbabilityRaster | None = None,
    geometry_crs=None,
) -> PairedWidthProfile:
    """Measure corresponding before/after widths at approximately 2 m spacing."""
    length = float(canonical_axis.length)
    if length <= 1e-9:
        return PairedWidthProfile(canonical_id, canonical_axis, tuple(), 0.0)
    before_surface = (
        make_valid(before_surface)
        if before_surface is not None and not before_surface.is_empty else before_surface
    )
    after_surface = (
        make_valid(after_surface)
        if after_surface is not None and not after_surface.is_empty else after_surface
    )
    before_support = (
        before_surface.buffer(config.centre_tolerance)
        if before_surface is not None and not before_surface.is_empty else None
    )
    after_support = (
        after_surface.buffer(config.centre_tolerance)
        if after_surface is not None and not after_surface.is_empty else None
    )
    count = max(2, int(np.ceil(length / max(config.sample_spacing, 0.25))) + 1)
    samples: list[PairedWidthSample] = []
    for index in range(count):
        t = index / (count - 1)
        position = length * t
        point = canonical_axis.interpolate(t, normalized=True)
        normal = _normal_at(canonical_axis, position, config.sample_spacing)
        unresolved = PeriodWidthMeasurement(
            None, None, None, "unresolved", 0.0, None, None, None, "",
            "canonical_tangent_invalid",
        )
        before_measurement = after_measurement = unresolved
        if normal is None:
            pass
        else:
            before_centre = before_part.interpolate(t, normalized=True)
            after_centre = after_part.interpolate(t, normalized=True)
            before_measurement = _measure_period_width(
                before_centre, normal, before_surface, before_support,
                before_probability, geometry_crs, config,
            )
            after_measurement = _measure_period_width(
                after_centre, normal, after_surface, after_support,
                after_probability, geometry_crs, config,
            )
        reasons = []
        if before_measurement.reject_reason:
            reasons.append(f"before_{before_measurement.reject_reason}")
        if after_measurement.reject_reason:
            reasons.append(f"after_{after_measurement.reject_reason}")
        before_width = before_measurement.final_width
        after_width = after_measurement.final_width
        valid = before_width is not None and after_width is not None and not reasons
        samples.append(PairedWidthSample(
            canonical_id=canonical_id,
            sample_index=index,
            position_m=position,
            t=t,
            point=point,
            before_width=before_width,
            after_width=after_width,
            width_diff=(after_width - before_width) if valid else None,
            valid=valid,
            reject_reason=";".join(reasons),
            before_surface_width=before_measurement.surface_width,
            before_probability_width=before_measurement.probability_width,
            before_width_source=before_measurement.width_source,
            before_width_confidence=before_measurement.width_confidence,
            before_probability_left_distance=before_measurement.probability_left_distance,
            before_probability_right_distance=before_measurement.probability_right_distance,
            before_probability_contrast=before_measurement.probability_contrast,
            before_probability_confidence=before_measurement.probability_confidence,
            before_probability_reject_reason=before_measurement.probability_reject_reason,
            after_surface_width=after_measurement.surface_width,
            after_probability_width=after_measurement.probability_width,
            after_width_source=after_measurement.width_source,
            after_width_confidence=after_measurement.width_confidence,
            after_probability_left_distance=after_measurement.probability_left_distance,
            after_probability_right_distance=after_measurement.probability_right_distance,
            after_probability_contrast=after_measurement.probability_contrast,
            after_probability_confidence=after_measurement.probability_confidence,
            after_probability_reject_reason=after_measurement.probability_reject_reason,
            width_confidence=(
                min(before_measurement.width_confidence, after_measurement.width_confidence)
                if valid else 0.0
            ),
            surface_probability_disagreement=(
                before_measurement.reject_reason == "surface_probability_disagreement"
                or after_measurement.reject_reason == "surface_probability_disagreement"
            ),
        ))
    valid_count = sum(sample.valid for sample in samples)
    return PairedWidthProfile(
        canonical_id=canonical_id,
        canonical_axis=canonical_axis,
        samples=tuple(samples),
        valid_ratio=valid_count / len(samples) if samples else 0.0,
    )


def robust_change_statistics(samples: Iterable[PairedWidthSample]) -> dict[str, float | int]:
    valid = [sample for sample in samples if sample.valid and sample.width_diff is not None]
    if not valid:
        return {
            "before_width": 0.0, "after_width": 0.0, "width_diff": 0.0,
            "diff_mad": 0.0, "uncertainty": 0.0, "direction_ratio": 0.0,
            "sample_count": 0,
        }
    before = np.asarray([sample.before_width for sample in valid], dtype=np.float64)
    after = np.asarray([sample.after_width for sample in valid], dtype=np.float64)
    differences = np.asarray([sample.width_diff for sample in valid], dtype=np.float64)
    median_diff = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median_diff)))
    sign = 1 if median_diff > 0 else -1 if median_diff < 0 else 0
    direction_ratio = (
        float(np.mean(np.sign(differences) == sign)) if sign else 0.0
    )
    uncertainty = float(1.4826 * mad / np.sqrt(len(valid)))
    return {
        "before_width": float(np.median(before)),
        "after_width": float(np.median(after)),
        "width_diff": median_diff,
        "diff_mad": mad,
        "uncertainty": uncertainty,
        "direction_ratio": direction_ratio,
        "sample_count": len(valid),
    }


def candidate_change_runs(
    profile: PairedWidthProfile,
    config: PairedWidthConfig = PairedWidthConfig(),
) -> list[PairedWidthRun]:
    """Return contiguous sample runs meeting per-sample absolute/relative gates.

    Minimum total length, sample count, valid ratio and robust dispersion are
    evaluated after adjacent canonical pieces have been assembled.
    """
    labels: list[int] = []
    for sample in profile.samples:
        sign = 0
        if sample.valid and sample.width_diff is not None:
            denominator = max(
                float(sample.before_width or 0.0), float(sample.after_width or 0.0), 0.1,
            )
            relative = abs(sample.width_diff) / denominator
            if abs(sample.width_diff) >= config.absolute_change and relative >= config.relative_change:
                sign = 1 if sample.width_diff > 0 else -1
        labels.append(sign)

    bridged = list(labels)
    index = 0
    while index < len(labels):
        if labels[index] != 0:
            index += 1
            continue
        gap_start = index
        while index < len(labels) and labels[index] == 0:
            index += 1
        gap_end = index - 1
        left = gap_start - 1
        right = index
        gap_count = gap_end - gap_start + 1
        if left < 0 or right >= len(labels):
            continue
        same_sign = labels[left] != 0 and labels[left] == labels[right]
        bridge_length = profile.samples[right].position_m - profile.samples[left].position_m
        if (
            same_sign
            and gap_count <= max(0, config.maximum_gap_samples)
            and bridge_length <= config.maximum_gap_length + 1e-6
        ):
            for gap_index in range(gap_start, gap_end + 1):
                bridged[gap_index] = labels[left]

    candidates: list[tuple[int, int, int]] = []
    index = 0
    while index < len(bridged):
        sign = bridged[index]
        if sign == 0:
            index += 1
            continue
        start_index = index
        while index + 1 < len(bridged) and bridged[index + 1] == sign:
            index += 1
        candidates.append((start_index, index, sign))
        index += 1

    result: list[PairedWidthRun] = []
    axis_length = float(profile.canonical_axis.length)
    for first, last, candidate_sign in candidates:
        previous_position = profile.samples[first - 1].position_m if first > 0 else 0.0
        next_position = profile.samples[last + 1].position_m if last + 1 < len(profile.samples) else axis_length
        start = 0.0 if first == 0 else 0.5 * (previous_position + profile.samples[first].position_m)
        end = axis_length if last + 1 == len(profile.samples) else 0.5 * (profile.samples[last].position_m + next_position)
        if end - start <= 1e-6:
            continue
        axis = substring(profile.canonical_axis, start, end)
        if axis.is_empty or axis.geom_type != "LineString":
            continue
        run_samples = tuple(profile.samples[first:last + 1])
        stats = robust_change_statistics(run_samples)
        valid_ratio = sum(sample.valid for sample in run_samples) / len(run_samples)
        result.append(PairedWidthRun(
            canonical_id=profile.canonical_id,
            sign=candidate_sign,
            start_m=start,
            end_m=end,
            axis=axis,
            samples=run_samples,
            before_width=float(stats["before_width"]),
            after_width=float(stats["after_width"]),
            width_diff=float(stats["width_diff"]),
            diff_mad=float(stats["diff_mad"]),
            uncertainty=float(stats["uncertainty"]),
            direction_ratio=float(stats["direction_ratio"]),
            valid_ratio=valid_ratio,
        ))
    return result


def evaluate_change_run(
    samples: Iterable[PairedWidthSample],
    *,
    axis_length: float,
    valid_ratio: float,
    config: PairedWidthConfig = PairedWidthConfig(),
) -> dict[str, float | int | bool | str]:
    stats = robust_change_statistics(samples)
    before_width = float(stats["before_width"])
    after_width = float(stats["after_width"])
    width_diff = float(stats["width_diff"])
    relative = abs(width_diff) / max(before_width, after_width, 0.1)
    reason = "paired_width_change_thresholds_met"
    if valid_ratio < config.minimum_valid_ratio:
        reason = "paired_width_valid_ratio_too_low"
    elif int(stats["sample_count"]) < config.minimum_samples:
        reason = "paired_width_sample_count_too_low"
    elif axis_length < config.minimum_continuous_length:
        reason = "paired_width_continuous_length_too_short"
    elif abs(width_diff) < config.absolute_change or relative < config.relative_change:
        reason = "paired_width_change_threshold_not_met"
    elif relative > config.maximum_relative_change:
        reason = "paired_width_relative_change_excessive"
    elif float(stats["direction_ratio"]) < config.minimum_direction_ratio:
        reason = "paired_width_direction_inconsistent"
    elif float(stats["diff_mad"]) > config.maximum_diff_mad:
        reason = "paired_width_mad_too_large"
    elif abs(width_diff) < config.uncertainty_scale * float(stats["uncertainty"]):
        reason = "paired_width_uncertainty_too_large"
    return {
        **stats,
        "axis_length": float(axis_length),
        "valid_ratio": float(valid_ratio),
        "relative_change": relative,
        "accepted": reason == "paired_width_change_thresholds_met",
        "decision": "change" if reason == "paired_width_change_thresholds_met" else "no_change",
        "reject_reason": "" if reason == "paired_width_change_thresholds_met" else reason,
    }


def profile_debug_rows(
    profile: PairedWidthProfile,
    change_decision: str,
) -> list[dict]:
    stats = robust_change_statistics(profile.valid_samples)
    rows = []
    for sample in profile.samples:
        rows.append({
            "canonical_id": profile.canonical_id,
            "sample_index": sample.sample_index,
            "sample_position_m": sample.position_m,
            "sample_t": sample.t,
            "x": sample.point.x,
            "y": sample.point.y,
            "before_surface_width": sample.before_surface_width,
            "before_probability_width": sample.before_probability_width,
            "before_final_width": sample.before_width,
            "before_width_source": sample.before_width_source,
            "before_width_confidence": sample.before_width_confidence,
            "before_probability_left_distance": sample.before_probability_left_distance,
            "before_probability_right_distance": sample.before_probability_right_distance,
            "before_probability_contrast": sample.before_probability_contrast,
            "before_probability_confidence": sample.before_probability_confidence,
            "before_probability_reject_reason": sample.before_probability_reject_reason,
            "after_surface_width": sample.after_surface_width,
            "after_probability_width": sample.after_probability_width,
            "after_final_width": sample.after_width,
            "after_width_source": sample.after_width_source,
            "after_width_confidence": sample.after_width_confidence,
            "after_probability_left_distance": sample.after_probability_left_distance,
            "after_probability_right_distance": sample.after_probability_right_distance,
            "after_probability_contrast": sample.after_probability_contrast,
            "after_probability_confidence": sample.after_probability_confidence,
            "after_probability_reject_reason": sample.after_probability_reject_reason,
            "before_width": sample.before_width,
            "after_width": sample.after_width,
            "width_diff": sample.width_diff,
            "width_confidence": sample.width_confidence,
            "surface_probability_disagreement": sample.surface_probability_disagreement,
            "valid": sample.valid,
            "reject_reason": sample.reject_reason,
            "mad": stats["diff_mad"],
            "uncertainty": stats["uncertainty"],
            "valid_ratio": profile.valid_ratio,
            "sample_count": stats["sample_count"],
            "change_decision": change_decision,
            "geometry": sample.point,
        })
    return rows
