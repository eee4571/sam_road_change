from __future__ import annotations

"""Cross-period road-width measurement on a shared canonical axis.

The module measures both periods with the same canonical tangent/normal while
allowing each period to retain its own centreline position.  It deliberately
does not infer a width when a road surface does not provide a complete,
contiguous cross-section through the period-specific centre.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely import make_valid
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring


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


def measure_paired_width_profile(
    canonical_id: str,
    canonical_axis: LineString,
    before_part: LineString,
    after_part: LineString,
    before_surface: BaseGeometry | None,
    after_surface: BaseGeometry | None,
    config: PairedWidthConfig = PairedWidthConfig(),
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
        before_width = after_width = None
        reasons: list[str] = []
        if normal is None:
            reasons.append("canonical_tangent_invalid")
        else:
            before_centre = before_part.interpolate(t, normalized=True)
            after_centre = after_part.interpolate(t, normalized=True)
            before_width, before_reason = _surface_width(
                before_centre, normal, before_surface, before_support, config,
            )
            after_width, after_reason = _surface_width(
                after_centre, normal, after_surface, after_support, config,
            )
            if before_reason:
                reasons.append(f"before_{before_reason}")
            if after_reason:
                reasons.append(f"after_{after_reason}")
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
    candidates: list[tuple[int, int]] = []
    current: list[int] = []
    current_sign = 0
    for index, sample in enumerate(profile.samples):
        sign = 0
        if sample.valid and sample.width_diff is not None:
            denominator = max(
                float(sample.before_width or 0.0), float(sample.after_width or 0.0), 0.1,
            )
            relative = abs(sample.width_diff) / denominator
            if abs(sample.width_diff) >= config.absolute_change and relative >= config.relative_change:
                sign = 1 if sample.width_diff > 0 else -1
        if sign and (not current or sign == current_sign):
            current.append(index)
            current_sign = sign
            continue
        if current:
            candidates.append((current[0], current[-1]))
        current = [index] if sign else []
        current_sign = sign
    if current:
        candidates.append((current[0], current[-1]))

    result: list[PairedWidthRun] = []
    axis_length = float(profile.canonical_axis.length)
    for first, last in candidates:
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
        sign = 1 if float(stats["width_diff"]) > 0 else -1
        result.append(PairedWidthRun(
            canonical_id=profile.canonical_id,
            sign=sign,
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
            "before_width": sample.before_width,
            "after_width": sample.after_width,
            "width_diff": sample.width_diff,
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
