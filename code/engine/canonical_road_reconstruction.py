from __future__ import annotations

"""Canonical road-network reconstruction from line and road-surface evidence.

Fast production output uses one medial axis per detected surface ribbon.  The
older line cleanup helpers remain available for focused geometry tests and for
the no-surface fallback.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import time

import cv2
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, polygonize, substring, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree


@dataclass(frozen=True)
class RoadObservation:
    observation_id: int
    samples: np.ndarray
    original_vertex_count: int
    length: float


@dataclass(frozen=True)
class CanonicalRoad:
    road_id: int
    observation_ids: tuple[int, ...]
    points: np.ndarray
    geometry_kind: str
    original_vertex_count: int


@dataclass(frozen=True)
class CanonicalJunction:
    point: np.ndarray
    road_ids: tuple[int, ...]


@dataclass(frozen=True)
class RegionalRoadObservation:
    points: np.ndarray
    width_m: float
    source_id: int
    source_tile: str = ""


@dataclass(frozen=True)
class RegionalCanonicalRoad:
    points: np.ndarray
    width_m: float
    source_ids: tuple[int, ...]
    geometry_kind: str


@dataclass(frozen=True)
class _StraightFragmentGroup:
    fragment_ids: tuple[int, ...]
    center: np.ndarray
    direction: np.ndarray
    minimum: float
    maximum: float
    fragment_coverage: float
    surface_coverage: float
    maximum_unsupported_gap_m: float


@dataclass(frozen=True)
class CanonicalJunctionCandidate:
    road_ids: tuple[int, ...]
    point: np.ndarray
    junction_type: str
    score: float
    branch_endpoints: tuple[tuple[int, bool], ...] = ()
    inference_kind: str = "axis_intersection"


@dataclass(frozen=True)
class _RegionalRoadSeed:
    points: np.ndarray
    width_m: float
    source_ids: tuple[int, ...]
    geometry_kind: str = ""


@dataclass(frozen=True)
class _SameTrackGap:
    first_id: int
    first_at_start: bool
    second_id: int
    second_at_start: bool
    distance: float
    lateral_offset: float
    score: float


def _deduplicate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] <= 1:
        return points
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    return points[keep]


def _polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    points = _deduplicate_points(points)
    if points.shape[0] < 2:
        return points.copy()
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(distances)]
    length = float(cumulative[-1])
    if length <= 0:
        return points[[0]].copy()
    positions = np.arange(0.0, length, max(float(spacing), 1e-6))
    positions = np.r_[positions, length]
    return np.column_stack([
        np.interp(positions, cumulative, points[:, axis]) for axis in range(2)
    ])


def extract_road_observations(
    road_entities: list[np.ndarray],
    unit_size_m: float,
    *,
    sample_spacing_m: float = 2.0,
) -> list[RoadObservation]:
    """Convert traced geometry into uniformly weighted road observations."""
    spacing = sample_spacing_m / max(float(unit_size_m), 1e-9)
    observations: list[RoadObservation] = []
    for observation_id, points in enumerate(road_entities):
        cleaned = _deduplicate_points(points)
        if cleaned.shape[0] < 2:
            continue
        observations.append(RoadObservation(
            observation_id=observation_id,
            samples=_resample_polyline(cleaned, spacing),
            original_vertex_count=int(cleaned.shape[0]),
            length=_polyline_length(cleaned),
        ))
    return observations


def _robust_axis(samples: np.ndarray, lateral_outlier: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray(samples, dtype=np.float64)
    center = np.median(active, axis=0)
    direction = np.asarray([1.0, 0.0])
    for _ in range(3):
        if active.shape[0] < 2:
            break
        covariance = np.cov((active - active.mean(axis=0)).T)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        center = active.mean(axis=0)
        normal = np.asarray([-direction[1], direction[0]])
        residuals = np.abs((samples - center) @ normal)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        limit = max(lateral_outlier, median + 3.5 * max(1.4826 * mad, 1e-6))
        selected = samples[residuals <= limit]
        if selected.shape[0] < max(3, samples.shape[0] // 2) or selected.shape[0] == active.shape[0]:
            break
        active = selected
    if direction[0] < 0 or (abs(direction[0]) < 1e-9 and direction[1] < 0):
        direction = -direction
    return center, direction, active


def estimate_canonical_road_axis(
    observation: RoadObservation,
    unit_size_m: float,
) -> tuple[np.ndarray, str]:
    """Regenerate a straight or curved road axis from sampled observations."""
    samples = np.asarray(observation.samples, dtype=np.float64)
    if samples.shape[0] < 3:
        return samples.astype(np.float32), "straight"
    metre = 1.0 / max(float(unit_size_m), 1e-9)
    center, direction, inliers = _robust_axis(samples, lateral_outlier=5.0 * metre)
    normal = np.asarray([-direction[1], direction[0]])
    longitudinal = (samples - center) @ direction
    residuals = np.abs((samples - center) @ normal)
    span = float(np.ptp(longitudinal))
    inlier_covariance = np.cov((inliers - inliers.mean(axis=0)).T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(inlier_covariance), 0.0)
    linearity = float(eigenvalues[-1] / max(eigenvalues[0], (0.35 * metre) ** 2))
    residual_85 = float(np.percentile(residuals, 85))
    observed_length = max(observation.length, 1e-9)
    is_straight = (
        span >= 12.0 * metre
        and linearity >= 18.0
        and residual_85 <= max(5.0 * metre, 0.055 * span)
        and observed_length <= 1.30 * max(span, 1e-9)
    )
    if is_straight:
        lower, upper = float(np.min(longitudinal)), float(np.max(longitudinal))
        return np.asarray([
            center + lower * direction,
            center + upper * direction,
        ], dtype=np.float32), "straight"

    spacing = max(1.5 * metre, observation.length / 180.0)
    resampled = _resample_polyline(samples, spacing)
    if resampled.shape[0] < 3:
        return resampled.astype(np.float32), "curved"
    radius = max(1, min(8, int(round(5.0 * metre / max(spacing, 1e-9)))))
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
    padded = np.pad(resampled, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.column_stack([
        np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)
    ])
    smoothed[0] = resampled[0]
    smoothed[-1] = resampled[-1]
    epsilon = max(1.5 * metre, min(4.0 * metre, 0.012 * observation.length))
    regenerated = cv2.approxPolyDP(
        smoothed.astype(np.float32).reshape(-1, 1, 2), float(epsilon), closed=False,
    ).reshape(-1, 2)
    if regenerated.shape[0] < 2:
        regenerated = smoothed[[0, -1]]
    return regenerated.astype(np.float32), "curved"


def _node_key(point: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.round(np.asarray(point), 3))


def _nearest_segment(point: np.ndarray, points: np.ndarray) -> tuple[float, int, float, np.ndarray]:
    starts = points[:-1]
    vectors = points[1:] - starts
    squared = np.sum(vectors * vectors, axis=1)
    fractions = np.zeros(vectors.shape[0], dtype=np.float64)
    valid = squared > 0
    fractions[valid] = np.clip(
        np.sum((point - starts[valid]) * vectors[valid], axis=1) / squared[valid], 0.0, 1.0,
    )
    projections = starts + fractions[:, None] * vectors
    distances = np.linalg.norm(projections - point, axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), index, float(fractions[index]), projections[index]


def _junction_intersection(
    point: np.ndarray,
    road_ids: tuple[int, ...],
    geometries: list[np.ndarray],
    max_shift: float,
) -> np.ndarray:
    identity = np.eye(2, dtype=np.float64)
    matrix = np.zeros((2, 2), dtype=np.float64)
    vector = np.zeros(2, dtype=np.float64)
    directions: list[np.ndarray] = []
    for road_id in road_ids:
        geometry = geometries[road_id]
        _distance, segment, _fraction, line_point = _nearest_segment(point, geometry)
        direction = geometry[segment + 1] - geometry[segment]
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            continue
        direction /= norm
        if any(abs(float(np.dot(direction, existing))) > 0.985 for existing in directions):
            continue
        directions.append(direction)
        projection = identity - np.outer(direction, direction)
        matrix += projection
        vector += projection @ line_point
    if len(directions) < 2 or float(np.linalg.det(matrix)) < 1e-8:
        return np.asarray(point, dtype=np.float64)
    solved = np.linalg.solve(matrix, vector)
    if float(np.linalg.norm(solved - point)) > max_shift:
        return np.asarray(point, dtype=np.float64)
    return solved


def _insert_junction(points: np.ndarray, original: np.ndarray, node: np.ndarray, endpoint_snap: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    _distance, segment, fraction, _projection = _nearest_segment(original, points)
    first_distance = float(np.linalg.norm(points[0] - original))
    last_distance = float(np.linalg.norm(points[-1] - original))
    if min(first_distance, last_distance) <= endpoint_snap:
        replaced = points.copy()
        replaced[0 if first_distance <= last_distance else -1] = node
        candidate = _deduplicate_points(replaced)
        return candidate if candidate.shape[0] >= 2 else points
    insert_at = segment + 1
    if fraction <= 1e-4:
        insert_at = segment
    elif fraction >= 1.0 - 1e-4:
        insert_at = segment + 1
    candidate = _deduplicate_points(np.insert(points, insert_at, node, axis=0))
    return candidate if candidate.shape[0] >= 2 else points


def fit_canonical_road_geometry(
    road_entities: list[np.ndarray],
    unit_size_m: float,
) -> tuple[list[CanonicalRoad], list[CanonicalJunction], dict[str, float | int]]:
    """Fit complete road entities first, then solve their shared junction graph."""
    observations = extract_road_observations(road_entities, unit_size_m)
    regenerated: list[np.ndarray] = []
    kinds: list[str] = []
    for observation in observations:
        geometry, kind = estimate_canonical_road_axis(observation, unit_size_m)
        if geometry.shape[0] < 2 or _polyline_length(geometry) <= 1e-8:
            geometry = observation.samples[[0, -1]].astype(np.float32)
        regenerated.append(geometry.astype(np.float64))
        kinds.append(kind)

    shared: dict[tuple[float, float], tuple[np.ndarray, set[int]]] = {}
    for road_id, points in enumerate(road_entities):
        seen: set[tuple[float, float]] = set()
        for point in np.asarray(points):
            key = _node_key(point)
            if key in seen:
                continue
            seen.add(key)
            if key not in shared:
                shared[key] = (np.asarray(point, dtype=np.float64), set())
            shared[key][1].add(road_id)

    junctions: list[CanonicalJunction] = []
    max_shift = 20.0 / max(float(unit_size_m), 1e-9)
    endpoint_snap = 25.0 / max(float(unit_size_m), 1e-9)
    for original, road_set in shared.values():
        if len(road_set) < 2:
            continue
        road_ids = tuple(sorted(road_set))
        node = _junction_intersection(original, road_ids, regenerated, max_shift)
        for road_id in road_ids:
            regenerated[road_id] = _insert_junction(
                regenerated[road_id], original, node, endpoint_snap,
            )
        junctions.append(CanonicalJunction(node.astype(np.float32), road_ids))

    roads = [
        CanonicalRoad(
            road_id=road_id,
            observation_ids=(observation.observation_id,),
            points=regenerated[road_id].astype(np.float32),
            geometry_kind=kinds[road_id],
            original_vertex_count=observation.original_vertex_count,
        )
        for road_id, observation in enumerate(observations)
    ]
    original_vertices = int(sum(observation.original_vertex_count for observation in observations))
    final_vertices = int(sum(road.points.shape[0] for road in roads))
    diagnostics: dict[str, float | int] = {
        "original_feature_count": int(len(observations)),
        "final_feature_count": int(len(roads)),
        "original_vertex_count": original_vertices,
        "final_vertex_count": final_vertices,
        "generated_junction_count": int(len(junctions)),
        "canonical_straight_road_count": int(sum(kind == "straight" for kind in kinds)),
        "canonical_curved_road_count": int(sum(kind == "curved" for kind in kinds)),
        "mean_vertices_per_road_before": float(original_vertices / max(len(observations), 1)),
        "mean_vertices_per_road_after": float(final_vertices / max(len(roads), 1)),
    }
    return roads, junctions, diagnostics


def _endpoint_heading(points: np.ndarray, at_start: bool, lookback: float) -> np.ndarray:
    ordered = points if at_start else points[::-1]
    distances = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    travelled = 0.0
    inward = ordered[-1]
    for index, distance in enumerate(distances):
        if distance <= 0:
            continue
        if travelled + distance >= lookback:
            fraction = (lookback - travelled) / distance
            inward = ordered[index] + fraction * (ordered[index + 1] - ordered[index])
            break
        travelled += float(distance)
    vector = ordered[0] - inward
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _road_axis_summary(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    samples = _resample_polyline(points, max(_polyline_length(points) / 80.0, 1.0))
    center, direction, _inliers = _robust_axis(samples, lateral_outlier=5.0)
    projections = (samples - center) @ direction
    return center, direction, float(np.min(projections)), float(np.max(projections))


def _straight_fragment_summary(
    road: RegionalRoadObservation,
    unit_size_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    points = _deduplicate_points(road.points)
    if points.shape[0] < 2:
        return None
    center, direction, minimum, maximum = _road_axis_summary(points)
    span = maximum - minimum
    if span <= 3.0 / max(float(unit_size_m), 1e-9):
        return None
    normal = np.asarray([-direction[1], direction[0]])
    residual = np.abs((points - center) @ normal)
    straight_limit = max(1.5, min(4.0, 0.06 * span * unit_size_m)) / max(
        float(unit_size_m), 1e-9,
    )
    if float(np.quantile(residual, 0.90)) > straight_limit:
        return None
    return center, direction, minimum, maximum


def _fit_fragment_group_axis(
    fragment_ids: list[int],
    roads: list[RegionalRoadObservation],
    unit_size_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float, list[tuple[float, float]]]:
    spacing = 2.0 / max(float(unit_size_m), 1e-9)
    samples = np.vstack([
        _resample_polyline(roads[fragment_id].points, spacing)
        for fragment_id in fragment_ids
    ])
    center, direction, _inliers = _robust_axis(
        samples, lateral_outlier=3.5 / max(float(unit_size_m), 1e-9),
    )
    intervals: list[tuple[float, float]] = []
    for fragment_id in fragment_ids:
        projections = (_deduplicate_points(roads[fragment_id].points) - center) @ direction
        intervals.append((float(np.min(projections)), float(np.max(projections))))
    minimum = min(start for start, _end in intervals)
    maximum = max(end for _start, end in intervals)
    return center, direction, minimum, maximum, intervals


def _measure_fragment_group_coverage(
    center: np.ndarray,
    direction: np.ndarray,
    minimum: float,
    maximum: float,
    intervals: list[tuple[float, float]],
    surface_geometry,
    unit_size_m: float,
) -> tuple[float, float, float]:
    span = maximum - minimum
    if span <= 1e-9:
        return 0.0, 0.0, 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    fragment_length = sum(end - start for start, end in merged)
    fragment_coverage = float(np.clip(fragment_length / span, 0.0, 1.0))
    axis = LineString([
        center + minimum * direction,
        center + maximum * direction,
    ])
    surface_intervals: list[tuple[float, float]] = []
    if surface_geometry is not None:
        try:
            for part in _surface_line_parts(axis.intersection(surface_geometry)):
                coordinates = np.asarray(part.coords)
                projection = (coordinates - center) @ direction
                surface_intervals.append((float(np.min(projection)), float(np.max(projection))))
        except Exception:
            surface_intervals = []
    surface_length = sum(end - start for start, end in surface_intervals)
    surface_coverage = float(np.clip(surface_length / span, 0.0, 1.0))

    bin_size = 12.0 / max(float(unit_size_m), 1e-9)
    bin_count = max(1, int(np.ceil(span / bin_size)))
    unsupported = 0.0
    maximum_unsupported = 0.0
    for bin_id in range(bin_count):
        start = minimum + span * bin_id / bin_count
        end = minimum + span * (bin_id + 1) / bin_count
        length = end - start
        fragment_overlap = sum(
            max(0.0, min(end, interval_end) - max(start, interval_start))
            for interval_start, interval_end in merged
        )
        surface_overlap = sum(
            max(0.0, min(end, interval_end) - max(start, interval_start))
            for interval_start, interval_end in surface_intervals
        )
        supported = fragment_overlap >= 0.25 * length or surface_overlap >= 0.60 * length
        unsupported = 0.0 if supported else unsupported + length
        maximum_unsupported = max(maximum_unsupported, unsupported)
    return (
        fragment_coverage,
        float(surface_coverage),
        float(maximum_unsupported * unit_size_m),
    )


def _find_straight_fragment_groups(
    roads: list[RegionalRoadObservation],
    unit_size_m: float,
    surface_geometry=None,
) -> tuple[list[_StraightFragmentGroup], set[int], int]:
    """Group collinear observations first, then validate the whole outer-endpoint span."""
    summaries = [_straight_fragment_summary(road, unit_size_m) for road in roads]
    straight_ids = [index for index, summary in enumerate(summaries) if summary is not None]
    if len(straight_ids) < 2:
        return [], set(), 0
    parent = list(range(len(roads)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    metre = 1.0 / max(float(unit_size_m), 1e-9)
    heading_cosine = float(np.cos(np.deg2rad(10.0)))
    candidate_gap = 120.0 * metre
    candidate_pairs: set[tuple[int, int]] = set()
    summary_centers = np.asarray([summaries[index][0] for index in straight_ids])
    summary_directions = np.asarray([summaries[index][1] for index in straight_ids])
    summary_half_spans = np.asarray([
        0.5 * (summaries[index][3] - summaries[index][2]) for index in straight_ids
    ])
    straight_id_array = np.asarray(straight_ids, dtype=int)
    for local_id, first_id in enumerate(straight_ids):
        first_center = summary_centers[local_id]
        first_direction = summary_directions[local_id]
        normal = np.asarray([-first_direction[1], first_direction[0]])
        relative = summary_centers - first_center
        aligned = np.abs(summary_directions @ first_direction) >= heading_cosine
        laterally_near = np.abs(relative @ normal) <= 8.0 * metre
        longitudinally_near = np.abs(relative @ first_direction) <= (
            summary_half_spans[local_id] + summary_half_spans + candidate_gap
        )
        later_ids = straight_id_array > first_id
        candidate_pairs.update(
            (first_id, int(second_id))
            for second_id in straight_id_array[
                aligned & laterally_near & longitudinally_near & later_ids
            ]
        )

    rejected_cross_corridor = 0
    for first_id, second_id in sorted(candidate_pairs):
        first_summary = summaries[first_id]
        assert first_summary is not None
        first_center, first_direction, _first_min, _first_max = first_summary
        second_summary = summaries[second_id]
        assert second_summary is not None
        second_center, second_direction, _second_min, _second_max = second_summary
        alignment = abs(float(np.dot(first_direction, second_direction)))
        if alignment < heading_cosine:
            continue
        if float(np.dot(first_direction, second_direction)) < 0:
            second_direction = -second_direction
        direction = first_direction + second_direction
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        normal = np.asarray([-direction[1], direction[0]])
        lateral = abs(float(np.dot(second_center - first_center, normal)))
        lateral_limit = max(
            3.0,
            0.35 * min(roads[first_id].width_m, roads[second_id].width_m),
        ) * metre
        if lateral > lateral_limit:
            if lateral <= 30.0 * metre:
                rejected_cross_corridor += 1
            continue
        first_projection = (_deduplicate_points(roads[first_id].points) - first_center) @ direction
        second_projection = (_deduplicate_points(roads[second_id].points) - first_center) @ direction
        first_interval = (float(np.min(first_projection)), float(np.max(first_projection)))
        second_interval = (float(np.min(second_projection)), float(np.max(second_projection)))
        longitudinal_gap = max(
            0.0,
            max(first_interval[0], second_interval[0])
            - min(first_interval[1], second_interval[1]),
        )
        if longitudinal_gap > candidate_gap:
            continue
        first_root, second_root = find(first_id), find(second_id)
        if first_root != second_root:
            parent[second_root] = first_root

    candidates: dict[int, list[int]] = {}
    for fragment_id in straight_ids:
        candidates.setdefault(find(fragment_id), []).append(fragment_id)
    groups: list[_StraightFragmentGroup] = []
    absorbed: set[int] = set()
    maximum_gap_limit_m = 36.0
    for fragment_ids in candidates.values():
        if len(fragment_ids) < 2:
            continue
        center, direction, minimum, maximum, intervals = _fit_fragment_group_axis(
            fragment_ids, roads, unit_size_m,
        )
        normal = np.asarray([-direction[1], direction[0]])
        lateral_values = np.concatenate([
            (_deduplicate_points(roads[fragment_id].points) - center) @ normal
            for fragment_id in fragment_ids
        ])
        lateral_limit = max(
            3.5,
            0.40 * float(np.median([roads[index].width_m for index in fragment_ids])),
        ) / max(float(unit_size_m), 1e-9)
        if float(np.quantile(np.abs(lateral_values), 0.90)) > lateral_limit:
            continue
        fragment_coverage, surface_coverage, maximum_gap_m = _measure_fragment_group_coverage(
            center, direction, minimum, maximum, intervals, surface_geometry, unit_size_m,
        )
        enough_evidence = (
            fragment_coverage >= 0.72
            or (fragment_coverage >= 0.30 and surface_coverage >= 0.72)
        )
        if not enough_evidence or maximum_gap_m > maximum_gap_limit_m:
            continue
        group = _StraightFragmentGroup(
            fragment_ids=tuple(sorted(fragment_ids)),
            center=center,
            direction=direction,
            minimum=minimum,
            maximum=maximum,
            fragment_coverage=fragment_coverage,
            surface_coverage=surface_coverage,
            maximum_unsupported_gap_m=maximum_gap_m,
        )
        groups.append(group)
        absorbed.update(fragment_ids)
    return groups, absorbed, rejected_cross_corridor


def _reconstruct_straight_road(
    group: _StraightFragmentGroup,
    roads: list[RegionalRoadObservation],
) -> _RegionalRoadSeed:
    lengths = np.asarray([
        max(_polyline_length(roads[index].points), 1e-9)
        for index in group.fragment_ids
    ])
    return _RegionalRoadSeed(
        points=np.asarray([
            group.center + group.minimum * group.direction,
            group.center + group.maximum * group.direction,
        ], dtype=np.float64),
        width_m=float(np.average(
            [roads[index].width_m for index in group.fragment_ids], weights=lengths,
        )),
        source_ids=tuple(sorted(roads[index].source_id for index in group.fragment_ids)),
        geometry_kind="straight",
    )


def _absorb_reconstructed_fragments(
    roads: list[RegionalRoadObservation],
    groups: list[_StraightFragmentGroup],
    absorbed: set[int],
    unit_size_m: float,
) -> list[_RegionalRoadSeed]:
    seeds = [_reconstruct_straight_road(group, roads) for group in groups]
    unmatched = [
        _RegionalRoadSeed(
            points=_deduplicate_points(road.points),
            width_m=road.width_m,
            source_ids=(road.source_id,),
        )
        for index, road in enumerate(roads) if index not in absorbed
    ]
    return [*seeds, *_fit_regional_seeds(unmatched, unit_size_m)]


def _fit_regional_seeds(
    seeds: list[_RegionalRoadSeed],
    unit_size_m: float,
) -> list[_RegionalRoadSeed]:
    canonical, _junctions, _diagnostics = fit_canonical_road_geometry(
        [seed.points for seed in seeds], unit_size_m,
    )
    fitted: list[_RegionalRoadSeed] = []
    for road in canonical:
        seed = seeds[road.road_id]
        points = np.asarray(road.points, dtype=np.float64)
        if points.shape[0] < 2 or _polyline_length(points) <= 1e-8:
            points = np.asarray(seed.points, dtype=np.float64)
        fitted.append(_RegionalRoadSeed(
            points=points,
            width_m=seed.width_m,
            source_ids=seed.source_ids,
            geometry_kind=road.geometry_kind,
        ))
    return fitted


def _surface_line_support(line: LineString, surface_geometry) -> float:
    if surface_geometry is None or line.length <= 0:
        return 0.5
    try:
        return float(min(1.0, line.intersection(surface_geometry).length / line.length))
    except Exception:
        return 0.5


def _sampled_surface_line_support(line: LineString, prepared_surface) -> float:
    if prepared_surface is None or line.length <= 0:
        return 0.5
    sample_count = max(3, min(9, int(np.ceil(line.length / 3.0)) + 1))
    samples = [line.interpolate(fraction, normalized=True) for fraction in np.linspace(0.0, 1.0, sample_count)]
    try:
        return float(np.mean([prepared_surface.covers(point) for point in samples]))
    except Exception:
        return 0.5


def _surface_line_parts(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        return [part for part in geometry.geoms if part.geom_type == "LineString" and part.length > 0]
    return []


def _center_straight_roads_on_surface(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> tuple[list[_RegionalRoadSeed], int]:
    if surface_geometry is None:
        return roads, 0
    corrected: list[_RegionalRoadSeed] = []
    correction_count = 0
    metre = 1.0 / max(float(unit_size_m), 1e-9)
    for road in roads:
        points = np.asarray(road.points, dtype=np.float64)
        if road.geometry_kind != "straight" or points.shape[0] < 2:
            corrected.append(road)
            continue
        direction = points[-1] - points[0]
        length = float(np.linalg.norm(direction))
        if length <= 0:
            corrected.append(road)
            continue
        direction /= length
        normal = np.asarray([-direction[1], direction[0]])
        search = max(12.0, 1.8 * road.width_m) * metre
        offsets: list[float] = []
        for fraction in (0.20, 0.40, 0.60, 0.80):
            point = points[0] + fraction * (points[-1] - points[0])
            cross = LineString([point - search * normal, point + search * normal])
            parts = _surface_line_parts(cross.intersection(surface_geometry))
            if not parts:
                continue
            nearest = min(parts, key=lambda part: part.distance(Point(point)))
            if nearest.distance(Point(point)) > max(2.0, 0.35 * road.width_m) * metre:
                continue
            if nearest.length > max(8.0, 2.5 * road.width_m) * metre:
                continue
            midpoint = np.asarray(nearest.interpolate(0.5, normalized=True).coords[0])
            offsets.append(float(np.dot(midpoint - point, normal)))
        if len(offsets) < 2:
            corrected.append(road)
            continue
        offset = float(np.median(offsets))
        limit = min(4.0, 0.40 * road.width_m) * metre
        offset = float(np.clip(offset, -limit, limit))
        if abs(offset) < 0.25 * metre:
            corrected.append(road)
            continue
        corrected.append(_RegionalRoadSeed(
            points=(points + offset * normal).astype(np.float32),
            width_m=road.width_m,
            source_ids=road.source_ids,
            geometry_kind=road.geometry_kind,
        ))
        correction_count += 1
    return corrected, correction_count


def _infinite_axis_intersection(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray | None:
    denominator = float(
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(denominator) < 1e-4:
        return None
    delta = second_point - first_point
    numerator = float(
        delta[0] * second_direction[1] - delta[1] * second_direction[0]
    )
    distance = numerator / denominator
    return first_point + distance * first_direction


def _junction_type_from_axes(
    node: np.ndarray,
    road_ids: tuple[int, ...],
    roads: list[_RegionalRoadSeed],
    endpoint_tolerance: float,
) -> str:
    arm_headings: list[np.ndarray] = []
    for road_id in road_ids:
        points = np.asarray(roads[road_id].points, dtype=np.float64)
        _distance, segment, fraction, _projection = _nearest_segment(node, points)
        tangent = points[segment + 1] - points[segment]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        along_start = _polyline_length(points[:segment + 1]) + fraction * float(
            np.linalg.norm(points[segment + 1] - points[segment])
        )
        total = _polyline_length(points)
        if along_start > endpoint_tolerance:
            arm_headings.append(-tangent)
        if total - along_start > endpoint_tolerance:
            arm_headings.append(tangent)
    if len(arm_headings) >= 4:
        return "cross"
    if len(arm_headings) == 3:
        opposite = any(
            float(np.dot(first, second)) < -0.90
            for first, second in combinations(arm_headings, 2)
        )
        return "t" if opposite else "y"
    return "y" if len(road_ids) >= 3 else "junction"


def _infer_endpoint_to_road_attachments(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> list[CanonicalJunctionCandidate]:
    if len(roads) < 2:
        return []
    metre = 1.0 / max(float(unit_size_m), 1e-9)
    maximum = 18.0 * metre
    clearance = 4.0 * metre
    lookback = 30.0 * metre
    lines = [LineString(np.asarray(road.points)) for road in roads]
    tree = STRtree(lines)
    prepared_surface = prep(surface_geometry) if surface_geometry is not None else None
    candidates: list[CanonicalJunctionCandidate] = []
    for branch_id, branch in enumerate(roads):
        points = np.asarray(branch.points, dtype=np.float64)
        for at_start, endpoint in ((True, points[0]), (False, points[-1])):
            heading = _endpoint_heading(points, at_start, lookback)
            for target_value in tree.query(Point(endpoint).buffer(maximum)):
                target_id = int(target_value)
                if target_id == branch_id:
                    continue
                target_points = np.asarray(roads[target_id].points, dtype=np.float64)
                distance, segment, fraction, projection = _nearest_segment(endpoint, target_points)
                segment_vector = target_points[segment + 1] - target_points[segment]
                segment_length = float(np.linalg.norm(segment_vector))
                if segment_length <= 0:
                    continue
                tangent = segment_vector / segment_length
                node = _infinite_axis_intersection(endpoint, heading, projection, tangent)
                if node is None:
                    continue
                forward = float(np.dot(node - endpoint, heading))
                if forward < 0.5 * metre or forward > maximum:
                    continue
                target_distance, target_segment, target_fraction, _ = _nearest_segment(node, target_points)
                if target_distance > 2.5 * metre:
                    continue
                before = _polyline_length(target_points[:target_segment + 1]) + target_fraction * float(
                    np.linalg.norm(target_points[target_segment + 1] - target_points[target_segment])
                )
                target_length = _polyline_length(target_points)
                if before < clearance or target_length - before < clearance:
                    continue
                facing = float(np.dot(heading, (node - endpoint) / max(forward, 1e-9)))
                axis_crossing = 1.0 - abs(float(np.dot(heading, tangent)))
                if facing < 0.70 or axis_crossing < 0.16:
                    continue
                width_ratio = max(branch.width_m, roads[target_id].width_m) / max(
                    min(branch.width_m, roads[target_id].width_m), 1e-6,
                )
                if width_ratio > 2.5:
                    continue
                connector = LineString([endpoint, node])
                support = _sampled_surface_line_support(connector, prepared_surface)
                distance_score = max(0.0, 1.0 - forward / maximum)
                score = (
                    0.30 * facing + 0.22 * min(1.0, axis_crossing / 0.50)
                    + 0.20 * distance_score + 0.18 * support
                    + 0.10 / width_ratio
                )
                if score < 0.56:
                    continue
                candidates.append(CanonicalJunctionCandidate(
                    road_ids=tuple(sorted((branch_id, target_id))),
                    point=np.asarray(node, dtype=np.float32),
                    junction_type="t",
                    score=float(score),
                    branch_endpoints=((branch_id, at_start),),
                    inference_kind="endpoint_to_road",
                ))
    accepted: list[CanonicalJunctionCandidate] = []
    used_endpoints: set[tuple[int, bool]] = set()
    for candidate in sorted(candidates, key=lambda item: -item.score):
        endpoint = candidate.branch_endpoints[0]
        if endpoint in used_endpoints:
            continue
        accepted.append(candidate)
        used_endpoints.add(endpoint)
    return accepted


def _infer_axis_intersections(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> list[CanonicalJunctionCandidate]:
    if len(roads) < 2:
        return []
    metre = 1.0 / max(float(unit_size_m), 1e-9)
    maximum = 18.0 * metre
    lines = [LineString(np.asarray(road.points)) for road in roads]
    tree = STRtree(lines)
    prepared_surface = prep(surface_geometry) if surface_geometry is not None else None
    candidates: list[CanonicalJunctionCandidate] = []
    for first_id, first_line in enumerate(lines):
        for second_value in tree.query(first_line.buffer(maximum)):
            second_id = int(second_value)
            if second_id <= first_id:
                continue
            second_line = lines[second_id]
            first_near, second_near = nearest_points(first_line, second_line)
            first_point = np.asarray(first_near.coords[0])
            second_point = np.asarray(second_near.coords[0])
            _distance, first_segment, _fraction, _projection = _nearest_segment(
                first_point, np.asarray(roads[first_id].points),
            )
            _distance, second_segment, _fraction, _projection = _nearest_segment(
                second_point, np.asarray(roads[second_id].points),
            )
            first_direction = np.diff(np.asarray(roads[first_id].points)[first_segment:first_segment + 2], axis=0)[0]
            second_direction = np.diff(np.asarray(roads[second_id].points)[second_segment:second_segment + 2], axis=0)[0]
            first_direction /= max(float(np.linalg.norm(first_direction)), 1e-9)
            second_direction /= max(float(np.linalg.norm(second_direction)), 1e-9)
            crossing = 1.0 - abs(float(np.dot(first_direction, second_direction)))
            if crossing < 0.16:
                continue
            node = _infinite_axis_intersection(
                first_point, first_direction, second_point, second_direction,
            )
            if node is None:
                continue
            first_gap = float(first_line.distance(Point(node)))
            second_gap = float(second_line.distance(Point(node)))
            if max(first_gap, second_gap) > maximum:
                continue
            connectors = [
                LineString([nearest.coords[0], node])
                for nearest in nearest_points(first_line, Point(node)) + nearest_points(second_line, Point(node))
                if np.linalg.norm(np.asarray(nearest.coords[0]) - node) > 1e-8
            ]
            support = float(np.mean([
                _sampled_surface_line_support(connector, prepared_surface) for connector in connectors
            ])) if connectors else 1.0
            distance_score = max(0.0, 1.0 - (first_gap + second_gap) / (2.0 * maximum))
            score = 0.42 * min(1.0, crossing / 0.50) + 0.35 * distance_score + 0.23 * support
            if score < 0.55:
                continue
            road_ids = (first_id, second_id)
            candidates.append(CanonicalJunctionCandidate(
                road_ids=road_ids,
                point=np.asarray(node, dtype=np.float32),
                junction_type=_junction_type_from_axes(node, road_ids, roads, 3.0 * metre),
                score=float(score),
                inference_kind="axis_intersection",
            ))
    return candidates


def _cluster_junction_candidates(
    candidates: list[CanonicalJunctionCandidate],
    roads: list[_RegionalRoadSeed],
    unit_size_m: float,
) -> list[CanonicalJunctionCandidate]:
    radius = 5.0 / max(float(unit_size_m), 1e-9)
    clusters: list[dict] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    for candidate in sorted(candidates, key=lambda item: -item.score):
        key = tuple(np.floor(candidate.point / max(radius, 1e-9)).astype(int))
        matched = None
        for offset in ((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)):
            for cluster_id in buckets.get((key[0] + offset[0], key[1] + offset[1]), []):
                cluster = clusters[cluster_id]
                if (
                    set(candidate.road_ids) & cluster["road_ids"]
                    and float(np.linalg.norm(candidate.point - cluster["point"])) <= radius
                ):
                    matched = cluster_id
                    break
            if matched is not None:
                break
        if matched is None:
            matched = len(clusters)
            clusters.append({
                "point": np.asarray(candidate.point, dtype=np.float64),
                "weight": candidate.score,
                "road_ids": set(candidate.road_ids),
                "branch_endpoints": set(candidate.branch_endpoints),
                "score": candidate.score,
                "kinds": {candidate.inference_kind},
            })
            buckets.setdefault(key, []).append(matched)
            continue
        cluster = clusters[matched]
        total_weight = cluster["weight"] + candidate.score
        cluster["point"] = (
            cluster["point"] * cluster["weight"] + candidate.point * candidate.score
        ) / max(total_weight, 1e-9)
        cluster["weight"] = total_weight
        cluster["road_ids"].update(candidate.road_ids)
        cluster["branch_endpoints"].update(candidate.branch_endpoints)
        cluster["score"] = max(cluster["score"], candidate.score)
        cluster["kinds"].add(candidate.inference_kind)

    junctions: list[CanonicalJunctionCandidate] = []
    endpoint_tolerance = 3.0 / max(float(unit_size_m), 1e-9)
    for cluster in clusters:
        road_ids = tuple(sorted(cluster["road_ids"]))
        point = np.asarray(cluster["point"], dtype=np.float32)
        junctions.append(CanonicalJunctionCandidate(
            road_ids=road_ids,
            point=point,
            junction_type=_junction_type_from_axes(
                point, road_ids, roads, endpoint_tolerance,
            ),
            score=float(cluster["score"]),
            branch_endpoints=tuple(sorted(cluster["branch_endpoints"])),
            inference_kind="+".join(sorted(cluster["kinds"])),
        ))
    return junctions


def _apply_junction_constraints(
    roads: list[_RegionalRoadSeed],
    junctions: list[CanonicalJunctionCandidate],
    unit_size_m: float,
) -> list[_RegionalRoadSeed]:
    by_road: dict[int, list[tuple[np.ndarray, bool | None]]] = {}
    endpoint_lookup = {
        endpoint: junction.point
        for junction in junctions for endpoint in junction.branch_endpoints
    }
    for junction in junctions:
        for road_id in junction.road_ids:
            flag = None
            if (road_id, True) in endpoint_lookup and np.allclose(endpoint_lookup[(road_id, True)], junction.point):
                flag = True
            elif (road_id, False) in endpoint_lookup and np.allclose(endpoint_lookup[(road_id, False)], junction.point):
                flag = False
            by_road.setdefault(road_id, []).append((np.asarray(junction.point, dtype=np.float64), flag))

    output: list[_RegionalRoadSeed] = []
    for road_id, road in enumerate(roads):
        constraints = by_road.get(road_id, [])
        if not constraints:
            output.append(road)
            continue
        evidence = np.asarray(road.points, dtype=np.float64)
        for node, endpoint_flag in constraints:
            if endpoint_flag is True:
                evidence = np.vstack((node, evidence))
            elif endpoint_flag is False:
                evidence = np.vstack((evidence, node))
            else:
                evidence = _insert_junction(evidence, node, node, endpoint_snap=0.0)
        observation = extract_road_observations([evidence], unit_size_m)[0]
        fitted, geometry_kind = estimate_canonical_road_axis(observation, unit_size_m)
        fitted = np.asarray(fitted, dtype=np.float64)
        if geometry_kind == "straight":
            direction = fitted[-1] - fitted[0]
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            base = constraints[0][0]
            endpoints = []
            for point in (fitted[0], fitted[-1]):
                endpoints.append(base + float(np.dot(point - base, direction)) * direction)
            ordered = [endpoints[0], *(node for node, _flag in constraints), endpoints[1]]
            fitted = np.asarray(sorted(
                ordered, key=lambda point: float(np.dot(point - base, direction)),
            ))
        for node, endpoint_flag in constraints:
            if endpoint_flag is True:
                fitted[0] = node
            elif endpoint_flag is False:
                fitted[-1] = node
            else:
                fitted = _insert_junction(fitted, node, node, endpoint_snap=0.0)
        cleaned = _deduplicate_points(fitted)
        if cleaned.shape[0] < 2 or _polyline_length(cleaned) <= 1e-8:
            cleaned = _deduplicate_points(evidence)
            geometry_kind = road.geometry_kind
        output.append(_RegionalRoadSeed(
            points=cleaned.astype(np.float64),
            width_m=road.width_m,
            source_ids=road.source_ids,
            geometry_kind=geometry_kind,
        ))
    return output


def _network_topology_metrics(roads: list[_RegionalRoadSeed]) -> tuple[int, int, dict[tuple[float, float], list[np.ndarray]]]:
    adjacency: dict[tuple[float, float], set[tuple[float, float]]] = {}
    headings: dict[tuple[float, float], list[np.ndarray]] = {}
    for road in roads:
        points = _deduplicate_points(road.points)
        for first, second in zip(points, points[1:]):
            first_key, second_key = _node_key(first), _node_key(second)
            if first_key == second_key:
                continue
            adjacency.setdefault(first_key, set()).add(second_key)
            adjacency.setdefault(second_key, set()).add(first_key)
            vector = second - first
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            headings.setdefault(first_key, []).append(vector)
            headings.setdefault(second_key, []).append(-vector)
    endpoint_count = int(sum(len(neighbors) == 1 for neighbors in adjacency.values()))
    components = 0
    visited: set[tuple[float, float]] = set()
    for node in adjacency:
        if node in visited:
            continue
        components += 1
        stack = [node]
        visited.add(node)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return endpoint_count, components, headings


def _detect_anchor_centerlines(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> set[int]:
    """Return stable long lines whose existing vertices must remain authoritative."""
    anchors: set[int] = set()
    for road_id, road in enumerate(roads):
        length_m = _polyline_length(road.points) * unit_size_m
        minimum_m = max(45.0, 4.0 * max(float(road.width_m), 1.0))
        support = _surface_line_support(LineString(road.points), surface_geometry)
        if length_m >= minimum_m and (surface_geometry is None or support >= 0.65):
            anchors.add(road_id)
    return anchors


def _seed_axis_interval(
    road: _RegionalRoadSeed,
    center: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    own_center, own_direction, _minimum, _maximum = _road_axis_summary(road.points)
    if center is None or direction is None:
        center, direction = own_center, own_direction
    elif float(np.dot(own_direction, direction)) < 0:
        own_direction = -own_direction
    projection = (np.asarray(road.points) - center) @ direction
    return np.asarray(center), np.asarray(direction), float(np.min(projection)), float(np.max(projection))


def _surface_separates_local_tracks(
    first: LineString,
    second: LineString,
    surface_geometry,
    unit_size_m: float,
) -> bool:
    """Confirm that two persistent lateral bands occupy separate road strips."""
    if surface_geometry is None:
        return True
    sample_line, reference_line = (
        (first, second) if first.length <= second.length else (second, first)
    )
    samples = _resample_polyline(
        np.asarray(sample_line.coords, dtype=np.float64),
        8.0 / max(unit_size_m, 1e-9),
    )
    prepared_surface = prep(surface_geometry)
    separated = []
    for point_array in samples:
        point = Point(point_array)
        projection = reference_line.interpolate(reference_line.project(point))
        distance_m = point.distance(projection) * unit_size_m
        if not 4.0 <= distance_m <= 30.0:
            continue
        midpoint = Point(
            0.5 * (point.x + projection.x),
            0.5 * (point.y + projection.y),
        )
        separated.append(not prepared_surface.covers(midpoint))
    return len(separated) >= 3 and float(np.mean(separated)) >= 0.35


def _parallel_track_pairs(
    roads: list[_RegionalRoadSeed],
    unit_size_m: float,
    surface_geometry=None,
) -> set[tuple[int, int]]:
    """Find two lateral bands that persist across a shared longitudinal range."""
    metre = 1.0 / max(float(unit_size_m), 1e-9)
    pairs: set[tuple[int, int]] = set()
    summaries = [_road_axis_summary(road.points) for road in roads]
    lines = [LineString(road.points) for road in roads]
    tree = STRtree(lines)
    cosine = float(np.cos(np.deg2rad(10.0)))
    candidate_pairs = {
        (first_id, int(second_id))
        for first_id, line in enumerate(lines)
        for second_id in tree.query(line.buffer(30.0 * metre))
        if int(second_id) > first_id
    }
    for first_id, second_id in sorted(candidate_pairs):
        first_center, first_direction, _a, _b = summaries[first_id]
        second_center, second_direction, _c, _d = summaries[second_id]
        if abs(float(np.dot(first_direction, second_direction))) < cosine:
            continue
        normal = np.asarray([-first_direction[1], first_direction[0]])
        signed_separation = float(np.dot(second_center - first_center, normal))
        separation = abs(signed_separation)
        if not 4.0 * metre <= separation <= 30.0 * metre:
            continue
        first_projection = (np.asarray(roads[first_id].points) - first_center) @ first_direction
        second_projection = (np.asarray(roads[second_id].points) - first_center) @ first_direction
        overlap = max(0.0, min(float(first_projection.max()), float(second_projection.max())) - max(
            float(first_projection.min()), float(second_projection.min()),
        ))
        longer = max(1e-9, max(float(np.ptp(first_projection)), float(np.ptp(second_projection))))
        shorter = max(1e-9, min(float(np.ptp(first_projection)), float(np.ptp(second_projection))))
        if (
            shorter * unit_size_m >= 45.0
            and overlap / longer >= 0.45
            and _surface_separates_local_tracks(
                lines[first_id], lines[second_id], surface_geometry, unit_size_m,
            )
        ):
            pairs.add((first_id, second_id))
            continue

        # Either carriageway may still consist of several observations.  Group
        # nearby fragments into two stable lateral bands and compare bin coverage
        # instead of treating the current two features as complete roads.
        band_intervals: tuple[list[tuple[float, float]], list[tuple[float, float]]] = ([], [])
        neighborhood = tree.query(lines[first_id].union(lines[second_id]).envelope.buffer(30.0 * metre))
        band_centres = (0.0, signed_separation)
        for road_id_value in neighborhood:
            road_id = int(road_id_value)
            _center, road_direction, _minimum, _maximum = summaries[road_id]
            if abs(float(np.dot(first_direction, road_direction))) < cosine:
                continue
            road_points = np.asarray(roads[road_id].points)
            lateral = (road_points - first_center) @ normal
            band_id = int(abs(float(np.median(lateral)) - band_centres[1]) < abs(float(np.median(lateral))))
            band_tolerance = min(2.5, max(1.5, 0.20 * separation * unit_size_m)) * metre
            if (
                abs(float(np.median(lateral)) - band_centres[band_id]) > band_tolerance
                or float(np.quantile(np.abs(lateral - np.median(lateral)), 0.90)) > 2.5 * metre
            ):
                continue
            projection = (road_points - first_center) @ first_direction
            band_intervals[band_id].append((float(projection.min()), float(projection.max())))
        if not band_intervals[0] or not band_intervals[1]:
            continue
        overall_min = min(start for intervals in band_intervals for start, _end in intervals)
        overall_max = max(end for intervals in band_intervals for _start, end in intervals)
        bin_size = 15.0 * metre
        if (overall_max - overall_min) * unit_size_m < 60.0:
            continue

        occupied: list[set[int]] = []
        for intervals in band_intervals:
            bins: set[int] = set()
            for start, end in intervals:
                first_bin = int(np.floor((start - overall_min) / bin_size))
                last_bin = int(np.floor((end - overall_min) / bin_size))
                bins.update(range(first_bin, last_bin + 1))
            occupied.append(bins)
        common_bins = occupied[0].intersection(occupied[1])
        if (
            min(len(occupied[0]), len(occupied[1])) >= 3
            and len(common_bins) / max(1, min(len(occupied[0]), len(occupied[1]))) >= 0.45
            and min(len(occupied[0]), len(occupied[1])) / max(len(occupied[0]), len(occupied[1])) >= 0.45
            and _surface_separates_local_tracks(
                lines[first_id], lines[second_id], surface_geometry, unit_size_m,
            )
        ):
            pairs.add((first_id, second_id))
    return pairs


def _same_track_duplicate(
    candidate: _RegionalRoadSeed,
    anchor: _RegionalRoadSeed,
    unit_size_m: float,
) -> bool:
    candidate_length = _polyline_length(candidate.points)
    anchor_length = _polyline_length(anchor.points)
    if candidate_length * unit_size_m > 45.0 or candidate_length > 0.60 * anchor_length:
        return False
    anchor_center, direction, _minimum, _maximum = _road_axis_summary(anchor.points)
    _center, candidate_direction, _a, _b = _road_axis_summary(candidate.points)
    if abs(float(np.dot(direction, candidate_direction))) < float(np.cos(np.deg2rad(12.0))):
        return False
    normal = np.asarray([-direction[1], direction[0]])
    lateral = np.abs((np.asarray(candidate.points) - anchor_center) @ normal)
    same_track_limit = min(2.5, max(1.25, 0.18 * min(candidate.width_m, anchor.width_m))) / max(
        float(unit_size_m), 1e-9,
    )
    if float(np.quantile(lateral, 0.90)) > same_track_limit:
        return False
    anchor_projection = (np.asarray(anchor.points) - anchor_center) @ direction
    candidate_projection = (np.asarray(candidate.points) - anchor_center) @ direction
    overlap = max(0.0, min(float(anchor_projection.max()), float(candidate_projection.max())) - max(
        float(anchor_projection.min()), float(candidate_projection.min()),
    ))
    candidate_span = max(1e-9, float(np.ptp(candidate_projection)))
    if overlap / candidate_span < 0.80:
        return False
    samples = _resample_polyline(candidate.points, 2.0 / max(float(unit_size_m), 1e-9))
    anchor_line = LineString(anchor.points)
    near_ratio = float(np.mean([
        anchor_line.distance(Point(point)) <= same_track_limit for point in samples
    ]))
    return near_ratio >= 0.85


def _remove_same_track_duplicate_fragments(
    roads: list[_RegionalRoadSeed],
    anchor_ids: set[int],
    unit_size_m: float,
) -> tuple[list[_RegionalRoadSeed], int]:
    removed: set[int] = set()
    ordered_anchors = sorted(anchor_ids, key=lambda index: -_polyline_length(roads[index].points))
    anchor_lines = [LineString(roads[index].points) for index in ordered_anchors]
    anchor_tree = STRtree(anchor_lines) if anchor_lines else None
    search_radius = 3.0 / max(float(unit_size_m), 1e-9)
    for candidate_id, candidate in enumerate(roads):
        if candidate_id in anchor_ids:
            continue
        nearby = (
            [ordered_anchors[int(index)] for index in anchor_tree.query(
                LineString(candidate.points).buffer(search_radius),
            )]
            if anchor_tree is not None else []
        )
        for anchor_id in nearby:
            if _same_track_duplicate(candidate, roads[anchor_id], unit_size_m):
                removed.add(candidate_id)
                break
    return [road for road_id, road in enumerate(roads) if road_id not in removed], len(removed)


def _surface_guided_gap_connector(
    start: np.ndarray,
    end: np.ndarray,
    width_m: float,
    surface_geometry,
    unit_size_m: float,
) -> np.ndarray | None:
    """Trace only a short missing interval through the local surface centre."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-8:
        return np.asarray([start, end])
    if surface_geometry is None:
        return np.asarray([start, end])
    direction = delta / distance
    normal = np.asarray([-direction[1], direction[0]])
    cross_radius = max(5.0, min(12.0, 0.85 * max(width_m, 1.0))) / max(float(unit_size_m), 1e-9)
    maximum_track_shift = min(4.0, max(2.0, 0.40 * max(width_m, 1.0))) / max(
        float(unit_size_m), 1e-9,
    )
    sample_count = max(3, min(11, int(np.ceil(distance * unit_size_m / 2.0)) + 1))
    guided = [start]
    for fraction in np.linspace(0.0, 1.0, sample_count)[1:-1]:
        expected = start + float(fraction) * delta
        section = LineString([expected - cross_radius * normal, expected + cross_radius * normal])
        parts = _surface_line_parts(section.intersection(surface_geometry))
        if not parts:
            guided.append(expected)
            continue
        nearest_part = min(parts, key=lambda part: part.distance(Point(expected)))
        if nearest_part.length * unit_size_m > 1.35 * max(width_m, 1.0):
            # A widened intersection can join both carriageways into one broad
            # cross-section.  Its midpoint is not the centre of either track.
            guided.append(expected)
            continue
        centre = nearest_part.interpolate(0.5, normalized=True)
        centre_point = np.asarray(centre.coords[0], dtype=np.float64)
        if float(np.linalg.norm(centre_point - expected)) <= maximum_track_shift:
            guided.append(centre_point)
        else:
            guided.append(expected)
    guided.append(end)
    connector = _deduplicate_points(np.asarray(guided, dtype=np.float64))
    if connector.shape[0] < 2 or _polyline_length(connector) > 1.55 * distance:
        return None
    support = _sampled_surface_line_support(LineString(connector), prep(surface_geometry))
    gap_m = distance * unit_size_m
    required_support = 0.70 + 0.15 * max(0.0, min(1.0, (gap_m - 12.0) / 8.0))
    return connector if support >= required_support else None


def _same_track_gap_candidate(
    roads: list[_RegionalRoadSeed],
    first_id: int,
    first_at_start: bool,
    second_id: int,
    second_at_start: bool,
    unit_size_m: float,
) -> tuple[_SameTrackGap | None, bool]:
    first, second = roads[first_id], roads[second_id]
    first_point = np.asarray(first.points[0 if first_at_start else -1], dtype=np.float64)
    second_point = np.asarray(second.points[0 if second_at_start else -1], dtype=np.float64)
    delta = second_point - first_point
    distance = float(np.linalg.norm(delta))
    maximum_m = 20.0
    if not 0.25 < distance * unit_size_m <= maximum_m:
        return None, False
    lookback_m = min(8.0, max(3.0, 0.25 * distance * unit_size_m))
    first_heading = _endpoint_heading(first.points, first_at_start, lookback_m / max(unit_size_m, 1e-9))
    second_heading = _endpoint_heading(second.points, second_at_start, lookback_m / max(unit_size_m, 1e-9))
    if min(float(np.linalg.norm(first_heading)), float(np.linalg.norm(second_heading))) <= 0:
        return None, False
    unit_delta = delta / distance
    broad_facing = (
        float(np.dot(first_heading, unit_delta)) >= float(np.cos(np.deg2rad(50.0)))
        and float(np.dot(second_heading, -unit_delta)) >= float(np.cos(np.deg2rad(50.0)))
        and float(np.dot(first_heading, second_heading)) <= -float(np.cos(np.deg2rad(45.0)))
    )
    if not broad_facing:
        return None, False
    track_direction = first_heading - second_heading
    track_direction /= max(float(np.linalg.norm(track_direction)), 1e-9)
    normal = np.asarray([-track_direction[1], track_direction[0]])
    lateral = abs(float(np.dot(delta, normal)))
    lateral_limit = min(3.0, max(1.5, 0.20 * min(first.width_m, second.width_m))) / max(
        float(unit_size_m), 1e-9,
    )
    if lateral > lateral_limit:
        return None, lateral * unit_size_m <= 30.0
    strict_facing = (
        float(np.dot(first_heading, unit_delta)) >= float(np.cos(np.deg2rad(38.0)))
        and float(np.dot(second_heading, -unit_delta)) >= float(np.cos(np.deg2rad(38.0)))
    )
    if not strict_facing:
        return None, False
    score = distance + 3.0 * lateral
    return _SameTrackGap(
        first_id, first_at_start, second_id, second_at_start, distance, lateral, score,
    ), False


def _merge_same_track_gap(
    first: _RegionalRoadSeed,
    second: _RegionalRoadSeed,
    gap: _SameTrackGap,
    connector: np.ndarray,
) -> _RegionalRoadSeed:
    first_points = np.asarray(first.points[::-1] if gap.first_at_start else first.points, dtype=np.float64)
    second_points = np.asarray(second.points if gap.second_at_start else second.points[::-1], dtype=np.float64)
    middle = np.asarray(connector, dtype=np.float64)
    if float(np.linalg.norm(middle[0] - first_points[-1])) > float(np.linalg.norm(middle[-1] - first_points[-1])):
        middle = middle[::-1]
    joined = _deduplicate_points(np.vstack((first_points, middle[1:-1], second_points)))
    first_length = max(_polyline_length(first.points), 1e-9)
    second_length = max(_polyline_length(second.points), 1e-9)
    return _RegionalRoadSeed(
        points=joined,
        width_m=float((first.width_m * first_length + second.width_m * second_length) / (first_length + second_length)),
        source_ids=tuple(sorted(set(first.source_ids + second.source_ids))),
        geometry_kind="preserved_with_local_gap",
    )


def _repair_same_track_gaps(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> tuple[list[_RegionalRoadSeed], int, float, int]:
    active = list(roads)
    repaired = 0
    repaired_length_m = 0.0
    rejected_cross_track: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    while True:
        candidates: list[_SameTrackGap] = []
        endpoints = [
            (road_id, at_start, np.asarray(road.points[0 if at_start else -1], dtype=np.float64))
            for road_id, road in enumerate(active) for at_start in (True, False)
        ]
        cell_size = 20.0 / max(float(unit_size_m), 1e-9)
        buckets: dict[tuple[int, int], list[int]] = {}
        for endpoint_id, (_road_id, _at_start, point) in enumerate(endpoints):
            key = tuple(np.floor(point / cell_size).astype(int))
            buckets.setdefault(key, []).append(endpoint_id)
        for first_endpoint_id, (first_id, first_at_start, first_point_array) in enumerate(endpoints):
            key = tuple(np.floor(first_point_array / cell_size).astype(int))
            nearby_endpoint_ids = (
                endpoint_id
                for offset_x in (-1, 0, 1) for offset_y in (-1, 0, 1)
                for endpoint_id in buckets.get((key[0] + offset_x, key[1] + offset_y), [])
                if endpoint_id > first_endpoint_id
            )
            for second_endpoint_id in nearby_endpoint_ids:
                second_id, second_at_start, _second_point_array = endpoints[second_endpoint_id]
                if first_id == second_id:
                    continue
                candidate, rejected = _same_track_gap_candidate(
                    active, first_id, first_at_start, second_id, second_at_start, unit_size_m,
                )
                if rejected:
                    first_point = _node_key(active[first_id].points[0 if first_at_start else -1])
                    second_point = _node_key(active[second_id].points[0 if second_at_start else -1])
                    rejected_cross_track.add(tuple(sorted((first_point, second_point))))
                if candidate is not None:
                    candidates.append(candidate)
        used_roads: set[int] = set()
        merged_roads: list[_RegionalRoadSeed] = []
        for gap in sorted(candidates, key=lambda item: item.score):
            if gap.first_id in used_roads or gap.second_id in used_roads:
                continue
            first, second = active[gap.first_id], active[gap.second_id]
            start = first.points[0 if gap.first_at_start else -1]
            end = second.points[0 if gap.second_at_start else -1]
            connector = _surface_guided_gap_connector(
                start, end, min(first.width_m, second.width_m), surface_geometry, unit_size_m,
            )
            if (
                connector is None
                and gap.distance * unit_size_m <= 16.0
                and gap.lateral_offset * unit_size_m <= 1.5
            ):
                # A small mask hole must not break an otherwise unambiguous
                # collinear track.  The headings and lateral gate above provide
                # the evidence; the missing surface is not treated as geometry.
                connector = np.asarray([start, end], dtype=np.float64)
            if connector is None:
                continue
            merged_roads.append(_merge_same_track_gap(first, second, gap, connector))
            used_roads.update((gap.first_id, gap.second_id))
            repaired += 1
            repaired_length_m += _polyline_length(connector) * unit_size_m
        if not merged_roads:
            break
        active = [
            road for road_id, road in enumerate(active) if road_id not in used_roads
        ] + merged_roads
    return active, repaired, repaired_length_m, len(rejected_cross_track)


def _local_heading(points: np.ndarray, index: int, forward: bool, look_m: float, unit_size_m: float) -> np.ndarray:
    """Estimate a heading beside a candidate window without using its interior."""
    origin = np.asarray(points[index], dtype=np.float64)
    step = 1 if forward else -1
    cursor = index
    travelled = 0.0
    while 0 <= cursor + step < len(points) and travelled * unit_size_m < look_m:
        following = cursor + step
        travelled += float(np.linalg.norm(points[following] - points[cursor]))
        cursor = following
    direction = np.asarray(points[cursor], dtype=np.float64) - origin
    if not forward:
        direction = -direction
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-9 else np.zeros(2, dtype=np.float64)


def _line_surface_clearance(points: np.ndarray, surface_geometry, unit_size_m: float) -> tuple[float, float]:
    """Return mean and lower-quartile distance from a local line to the surface edge."""
    line = LineString(points)
    if line.length <= 1e-9 or surface_geometry is None:
        return 0.0, 0.0
    prepared_surface = prep(surface_geometry)
    samples = [line.interpolate(value) for value in np.linspace(0.0, line.length, 17)]
    if float(np.mean([prepared_surface.covers(point) for point in samples])) < 0.94:
        return 0.0, 0.0
    distances = np.asarray([surface_geometry.boundary.distance(point) * unit_size_m for point in samples])
    return float(np.mean(distances)), float(np.quantile(distances, 0.25))


def _repair_local_offset_jumps(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> tuple[list[_RegionalRoadSeed], int, float]:
    """Replace only short detours rejected by the local road-surface centre."""
    if surface_geometry is None:
        return roads, 0, 0.0
    repaired_roads: list[_RegionalRoadSeed] = []
    repair_count = 0
    repaired_length_m = 0.0
    cosine = float(np.cos(np.deg2rad(18.0)))
    for road in roads:
        points = np.asarray(road.points, dtype=np.float64)
        road_repairs = 0
        while points.shape[0] >= 5 and road_repairs < 3:
            candidates: list[tuple[float, int, int, np.ndarray, float]] = []
            for start in range(1, points.shape[0] - 2):
                maximum_end = min(points.shape[0] - 2, start + 6)
                for end in range(start + 2, maximum_end + 1):
                    original = points[start:end + 1]
                    original_length_m = _polyline_length(original) * unit_size_m
                    chord = points[end] - points[start]
                    chord_length = float(np.linalg.norm(chord))
                    chord_length_m = chord_length * unit_size_m
                    if not 6.0 <= chord_length_m <= 35.0 or original_length_m > 45.0:
                        continue
                    chord_direction = chord / max(chord_length, 1e-9)
                    incoming = _local_heading(points, start, False, 10.0, unit_size_m)
                    outgoing = _local_heading(points, end, True, 10.0, unit_size_m)
                    if (
                        float(np.dot(incoming, outgoing)) < cosine
                        or float(np.dot(incoming, chord_direction)) < cosine
                        or float(np.dot(outgoing, chord_direction)) < cosine
                    ):
                        continue
                    chord_line = LineString([points[start], points[end]])
                    maximum_offset_m = max(
                        chord_line.distance(Point(point)) for point in original[1:-1]
                    ) * unit_size_m
                    minimum_offset_m = max(2.0, min(4.0, 0.18 * max(road.width_m, 1.0)))
                    if maximum_offset_m < minimum_offset_m or original_length_m < 1.035 * chord_length_m:
                        continue
                    connector = _surface_guided_gap_connector(
                        points[start], points[end], road.width_m, surface_geometry, unit_size_m,
                    )
                    if connector is None:
                        continue
                    connector_length_m = _polyline_length(connector) * unit_size_m
                    if connector_length_m > 0.98 * original_length_m:
                        continue
                    original_mean, original_lower = _line_surface_clearance(
                        original, surface_geometry, unit_size_m,
                    )
                    repaired_mean, repaired_lower = _line_surface_clearance(
                        connector, surface_geometry, unit_size_m,
                    )
                    if (
                        repaired_mean < original_mean + 0.75
                        or repaired_lower < original_lower + 0.50
                    ):
                        continue
                    score = (repaired_mean - original_mean) + 0.25 * maximum_offset_m
                    candidates.append((score, start, end, connector, connector_length_m))
            if not candidates:
                break
            _score, start, end, connector, connector_length_m = max(candidates, key=lambda item: item[0])
            points = _deduplicate_points(np.vstack((points[:start], connector, points[end + 1:])))
            road_repairs += 1
            repair_count += 1
            repaired_length_m += connector_length_m
        repaired_roads.append(_RegionalRoadSeed(
            points=points,
            width_m=road.width_m,
            source_ids=road.source_ids,
            geometry_kind=("preserved_with_local_offset_repair" if road_repairs else road.geometry_kind),
        ))
    return repaired_roads, repair_count, repaired_length_m


def _path_turning(points: np.ndarray) -> float:
    segments = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    segments = segments[lengths > 1e-8]
    if segments.shape[0] < 2:
        return 0.0
    directions = segments / np.linalg.norm(segments, axis=1)[:, None]
    return float(np.sum(np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0))))


def _merge_exact_degree_two_chains(
    roads: list[_RegionalRoadSeed],
    unit_size_m: float,
) -> list[_RegionalRoadSeed]:
    """Join fragments only where exactly two collinear chain ends share a node."""
    active = list(roads)
    cosine = float(np.cos(np.deg2rad(35.0)))
    while True:
        endpoints: dict[tuple[float, float], list[tuple[int, bool]]] = {}
        for road_id, road in enumerate(active):
            endpoints.setdefault(_node_key(road.points[0]), []).append((road_id, True))
            endpoints.setdefault(_node_key(road.points[-1]), []).append((road_id, False))
        merge = None
        for entries in endpoints.values():
            if len(entries) != 2 or entries[0][0] == entries[1][0]:
                continue
            first_id, first_at_start = entries[0]
            second_id, second_at_start = entries[1]
            first_heading = _endpoint_heading(
                active[first_id].points, first_at_start, 10.0 / max(unit_size_m, 1e-9),
            )
            second_heading = _endpoint_heading(
                active[second_id].points, second_at_start, 10.0 / max(unit_size_m, 1e-9),
            )
            if float(np.dot(first_heading, second_heading)) > -cosine:
                continue
            merge = (first_id, first_at_start, second_id, second_at_start)
            break
        if merge is None:
            return active
        first_id, first_at_start, second_id, second_at_start = merge
        first, second = active[first_id], active[second_id]
        first_points = first.points[::-1] if first_at_start else first.points
        second_points = second.points if second_at_start else second.points[::-1]
        first_length = max(_polyline_length(first.points), 1e-9)
        second_length = max(_polyline_length(second.points), 1e-9)
        joined = _RegionalRoadSeed(
            _deduplicate_points(np.vstack((first_points, second_points[1:]))),
            float((first.width_m * first_length + second.width_m * second_length) / (first_length + second_length)),
            tuple(sorted(set(first.source_ids + second.source_ids))),
            "preserved_with_local_path_selection",
        )
        active = [
            road for road_id, road in enumerate(active) if road_id not in (first_id, second_id)
        ] + [joined]


def _select_single_path_through_local_corridor_loops(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> tuple[list[_RegionalRoadSeed], int, float]:
    """Resolve a narrow non-junction loop as two competing paths through one track."""
    if len(roads) < 2:
        return roads, 0, 0.0
    lines = [LineString(road.points) for road in roads]
    linework = unary_union(lines)
    prepared_surface = prep(surface_geometry) if surface_geometry is not None else None
    selections: list[tuple[LineString, LineString]] = []
    selected_count = 0
    removed_length_m = 0.0
    for polygon in polygonize(linework):
        ring = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        if ring.shape[0] < 3:
            continue
        perimeter_m = float(polygon.length * unit_size_m)
        if not 10.0 <= perimeter_m <= 220.0:
            continue
        center = np.mean(ring, axis=0)
        covariance = np.cov((ring - center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        normal = np.asarray([-direction[1], direction[0]])
        longitudinal = (ring - center) @ direction
        lateral = (ring - center) @ normal
        span_m = float(np.ptp(longitudinal) * unit_size_m)
        width_m = float(np.ptp(lateral) * unit_size_m)
        if not 8.0 <= span_m <= 120.0 or not 0.5 <= width_m <= 12.0 or span_m < 1.5 * width_m:
            continue

        # Road blocks and true intersections contain substantial transverse
        # geometry.  A same-track diamond is dominated by one local direction.
        segment_vectors = np.diff(np.vstack((ring, ring[0])), axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        valid = segment_lengths > 1e-8
        alignment = np.zeros_like(segment_lengths)
        alignment[valid] = np.abs(segment_vectors[valid] @ direction) / segment_lengths[valid]
        aligned_length = float(np.sum(segment_lengths[alignment >= np.cos(np.deg2rad(35.0))]))
        if aligned_length / max(float(np.sum(segment_lengths)), 1e-9) < 0.62:
            continue

        start = int(np.argmin(longitudinal))
        end = int(np.argmax(longitudinal))
        if start > end:
            start, end = end, start
        first = ring[start:end + 1]
        second = np.vstack((ring[end:], ring[:start + 1]))[::-1]
        if min(first.shape[0], second.shape[0]) < 2:
            continue
        first_line, second_line = LineString(first), LineString(second)
        first_support = (
            _sampled_surface_line_support(first_line, prepared_surface)
            if prepared_surface is not None else 1.0
        )
        second_support = (
            _sampled_surface_line_support(second_line, prepared_surface)
            if prepared_surface is not None else 1.0
        )
        if max(first_support, second_support) < 0.80:
            continue
        first_clearance, _ = _line_surface_clearance(first, surface_geometry, unit_size_m)
        second_clearance, _ = _line_surface_clearance(second, surface_geometry, unit_size_m)
        first_turn = _path_turning(first)
        second_turn = _path_turning(second)
        first_length_m = first_line.length * unit_size_m
        second_length_m = second_line.length * unit_size_m
        first_score = 2.0 * first_support + 0.12 * first_clearance - 0.012 * first_length_m - 0.30 * first_turn
        second_score = 2.0 * second_support + 0.12 * second_clearance - 0.012 * second_length_m - 0.30 * second_turn
        better, worse = (first_line, second_line) if first_score >= second_score else (second_line, first_line)
        if prepared_surface is not None and _sampled_surface_line_support(better, prepared_surface) < 0.80:
            continue
        selections.append((better, worse))
        selected_count += 1
        removed_length_m += worse.length * unit_size_m

    if not selections:
        return roads, 0, 0.0
    selected: list[_RegionalRoadSeed] = []
    for road, line in zip(roads, lines):
        replacements: list[tuple[float, float, LineString, int]] = []
        for selection_id, (better, worse) in enumerate(selections):
            if line.intersection(worse).length * unit_size_m <= 0.20:
                continue
            better_start = Point(better.coords[0])
            better_end = Point(better.coords[-1])
            tolerance = 0.10 / max(unit_size_m, 1e-9)
            if line.distance(better_start) > tolerance or line.distance(better_end) > tolerance:
                continue
            first_distance = line.project(better_start)
            second_distance = line.project(better_end)
            lower, upper = sorted((first_distance, second_distance))
            if (upper - lower) * unit_size_m <= 0.20:
                continue
            replacement = better
            lower_point = line.interpolate(lower)
            if lower_point.distance(better_end) < lower_point.distance(better_start):
                replacement = LineString(list(better.coords)[::-1])
            replacements.append((lower, upper, replacement, selection_id))

        replacements.sort(key=lambda item: (item[0], item[1]))
        accepted: list[tuple[float, float, LineString, int]] = []
        for replacement in replacements:
            if accepted and replacement[0] < accepted[-1][1] - 1e-8:
                continue
            accepted.append(replacement)
        handled = {selection_id for _lower, _upper, _replacement, selection_id in accepted}
        coordinates: list[np.ndarray] = []
        cursor = 0.0
        for lower, upper, replacement, _selection_id in accepted:
            if lower > cursor + 1e-8:
                prefix = substring(line, cursor, lower)
                coordinates.extend(np.asarray(prefix.coords, dtype=np.float64))
            coordinates.extend(np.asarray(replacement.coords, dtype=np.float64))
            cursor = upper
        if cursor < line.length - 1e-8:
            suffix = substring(line, cursor, line.length)
            coordinates.extend(np.asarray(suffix.coords, dtype=np.float64))
        rebuilt = (
            LineString(_deduplicate_points(np.asarray(coordinates, dtype=np.float64)))
            if len(coordinates) >= 2 else line
        )
        unhandled = [
            worse for selection_id, (_better, worse) in enumerate(selections)
            if selection_id not in handled
            and line.intersection(worse).length * unit_size_m > 0.20
        ]
        remainder = rebuilt.difference(unary_union(unhandled)) if unhandled else rebuilt
        parts = _surface_line_parts(remainder)
        for part in parts:
            if part.length * unit_size_m < 0.25:
                continue
            selected.append(_RegionalRoadSeed(
                np.asarray(part.coords, dtype=np.float64), road.width_m, road.source_ids,
                "preserved_with_local_path_selection",
            ))
    return _merge_exact_degree_two_chains(selected, unit_size_m), selected_count, removed_length_m


def _remove_short_self_loops(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> tuple[list[_RegionalRoadSeed], int, float]:
    """Collapse only a short loop that leaves and returns to the same local track."""
    cleaned: list[_RegionalRoadSeed] = []
    removed_count = 0
    removed_length_m = 0.0
    cosine = float(np.cos(np.deg2rad(40.0)))
    for road in roads:
        points = np.asarray(road.points, dtype=np.float64)
        road_removed = 0
        while points.shape[0] >= 5 and road_removed < 3:
            candidates: list[tuple[float, int, int, np.ndarray, float]] = []
            for start in range(1, points.shape[0] - 3):
                for end in range(start + 2, min(points.shape[0] - 1, start + 12) + 1):
                    local = points[start:end + 1]
                    local_length_m = _polyline_length(local) * unit_size_m
                    return_distance_m = float(np.linalg.norm(points[end] - points[start])) * unit_size_m
                    if not 8.0 <= local_length_m <= 70.0 or return_distance_m > 2.5:
                        continue
                    incoming = _local_heading(points, start, False, 8.0, unit_size_m)
                    outgoing = _local_heading(points, end, True, 8.0, unit_size_m)
                    if float(np.dot(incoming, outgoing)) < cosine:
                        continue
                    if return_distance_m <= 0.25:
                        connector = np.asarray([points[start], points[end]], dtype=np.float64)
                        replacement_clearance = float("inf")
                    else:
                        connector = _surface_guided_gap_connector(
                            points[start], points[end], road.width_m, surface_geometry, unit_size_m,
                        )
                        if connector is None:
                            continue
                        _mean, replacement_clearance = _line_surface_clearance(
                            connector, surface_geometry, unit_size_m,
                        )
                    _mean, original_clearance = _line_surface_clearance(
                        local, surface_geometry, unit_size_m,
                    )
                    if replacement_clearance < original_clearance + 0.35:
                        continue
                    candidates.append((local_length_m, start, end, connector, local_length_m))
            if not candidates:
                break
            _score, start, end, connector, removed_m = max(candidates, key=lambda item: item[0])
            points = _deduplicate_points(np.vstack((points[:start], connector, points[end + 1:])))
            road_removed += 1
            removed_count += 1
            removed_length_m += removed_m
        cleaned.append(_RegionalRoadSeed(
            points, road.width_m, road.source_ids,
            "preserved_with_local_loop_cleanup" if road_removed else road.geometry_kind,
        ))
    return cleaned, removed_count, removed_length_m


def _select_same_track_local_paths(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
    parallel_pairs: set[tuple[int, int]],
) -> tuple[list[_RegionalRoadSeed], int, float]:
    """Enforce one canonical path per non-junction lateral track."""
    if len(roads) < 2:
        return roads, 0, 0.0
    metre = 1.0 / max(float(unit_size_m), 1e-9)
    lines = [LineString(road.points) for road in roads]
    tree = STRtree(lines)
    removed: set[int] = set()
    removed_length_m = 0.0
    cosine = float(np.cos(np.deg2rad(25.0)))
    prepared_surface = prep(surface_geometry) if surface_geometry is not None else None
    axes = [_road_axis_summary(road.points) for road in roads]
    quality = []
    for road_id, (road, line) in enumerate(zip(roads, lines)):
        length_m = line.length * unit_size_m
        turning = _path_turning(road.points)
        segment_lengths = np.linalg.norm(np.diff(road.points, axis=0), axis=1)
        reversal = float(np.sum(segment_lengths[1:][
            np.sum(
                np.diff(road.points, axis=0)[:-1] * np.diff(road.points, axis=0)[1:], axis=1,
            ) < 0.0
        ])) if road.points.shape[0] > 2 else 0.0
        density_scale = 50.0 / max(length_m, 20.0)
        quality.append(
            0.020 * min(length_m, 200.0)
            - 1.20 * turning * density_scale
            - 0.04 * reversal * density_scale
            - 1e-8 * road_id
        )

    for candidate_id in sorted(range(len(roads)), key=lambda road_id: quality[road_id]):
        candidate_line = lines[candidate_id]
        candidate_length_m = candidate_line.length * unit_size_m
        if candidate_length_m < 2.0:
            continue
        candidate_points = np.asarray(roads[candidate_id].points)
        _candidate_center, candidate_direction, _minimum, _maximum = axes[candidate_id]
        bundle_radius_m = min(30.0, max(12.0, 1.50 * max(roads[candidate_id].width_m, 1.0)))
        reference_ids = sorted(
            (int(value) for value in tree.query(candidate_line.buffer(bundle_radius_m * metre))),
            key=lambda road_id: quality[road_id], reverse=True,
        )
        for reference_id in reference_ids:
            if (
                reference_id == candidate_id
                or reference_id in removed
                or tuple(sorted((candidate_id, reference_id))) in parallel_pairs
            ):
                continue
            if (
                quality[reference_id] < quality[candidate_id] - 0.05
                or (
                    abs(quality[reference_id] - quality[candidate_id]) <= 0.05
                    and reference_id > candidate_id
                )
            ):
                continue
            reference_line = lines[reference_id]
            reference_length_m = reference_line.length * unit_size_m
            if reference_length_m < 35.0:
                continue
            _reference_center, reference_direction, _a, _b = axes[reference_id]
            if abs(float(np.dot(candidate_direction, reference_direction))) < cosine:
                continue
            start_projection = reference_line.project(Point(candidate_points[0]))
            end_projection = reference_line.project(Point(candidate_points[-1]))
            projected_span_m = abs(end_projection - start_projection) * unit_size_m
            if projected_span_m < max(2.0, 0.30 * candidate_length_m):
                continue
            endpoint_distances_m = (
                reference_line.distance(Point(candidate_points[0])) * unit_size_m,
                reference_line.distance(Point(candidate_points[-1])) * unit_size_m,
            )
            if max(endpoint_distances_m) > bundle_radius_m:
                continue
            reference_part = substring(
                reference_line, min(start_projection, end_projection), max(start_projection, end_projection),
            )
            if reference_part.geom_type != "LineString" or reference_part.length <= 1e-8:
                continue
            reference_points = np.asarray(reference_part.coords)
            reference_direction = reference_points[-1] - reference_points[0]
            reference_direction /= max(float(np.linalg.norm(reference_direction)), 1e-9)
            if abs(float(np.dot(candidate_direction, reference_direction))) < cosine:
                continue
            candidate_samples = _resample_polyline(candidate_points, 2.0 * metre)
            lateral_distances = np.asarray([
                reference_line.distance(Point(point)) * unit_size_m for point in candidate_samples
            ])
            if float(np.quantile(lateral_distances, 0.90)) > bundle_radius_m:
                continue
            # A genuine companion track has already been validated using
            # longitudinal coverage and a surface gap.  Everything else in the
            # same local strip competes for the single permitted path.
            candidate_support = (
                _sampled_surface_line_support(candidate_line, prepared_surface)
                if prepared_surface is not None else 1.0
            )
            reference_support = (
                _sampled_surface_line_support(reference_part, prepared_surface)
                if prepared_surface is not None else 1.0
            )
            if reference_support < 0.65:
                continue
            candidate_turn = _path_turning(candidate_points)
            reference_turn = _path_turning(reference_points)
            length_ratio = candidate_line.length / max(reference_part.length, 1e-9)
            mean_lateral = float(np.mean(lateral_distances))
            candidate_score = (
                2.20 * abs(float(np.dot(candidate_direction, reference_direction)))
                - 0.20 * mean_lateral
                - 0.90 * max(0.0, length_ratio - 1.0)
                - 0.55 * candidate_turn
                + 0.15 * candidate_support
            )
            reference_score = (
                2.20 - 0.55 * reference_turn + 0.15 * reference_support
            )
            if reference_score >= candidate_score + 0.05:
                removed.add(candidate_id)
                removed_length_m += candidate_length_m
                break
    return [road for road_id, road in enumerate(roads) if road_id not in removed], len(removed), removed_length_m


def _conservative_junction_touchup(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> tuple[list[_RegionalRoadSeed], int, float]:
    """Attach only a close, clearly perpendicular branch endpoint to a road interior."""
    if surface_geometry is None:
        return roads, 0, 0.0
    active = list(roads)
    used_endpoints: set[tuple[int, bool]] = set()
    touchups = 0
    added_length_m = 0.0
    candidates: list[tuple[float, int, bool, int, np.ndarray]] = []
    metre = 1.0 / max(float(unit_size_m), 1e-9)
    target_lines = [LineString(road.points) for road in active]
    target_tree = STRtree(target_lines)
    for branch_id in range(len(active)):
        for at_start in (True, False):
            endpoint = np.asarray(active[branch_id].points[0 if at_start else -1], dtype=np.float64)
            target_ids = target_tree.query(Point(endpoint).buffer(8.0 * metre))
            for target_id_value in target_ids:
                target_id = int(target_id_value)
                if target_id == branch_id:
                    continue
                target_line = target_lines[target_id]
                distance, segment, _fraction, projection = _nearest_segment(endpoint, active[target_id].points)
                distance_m = distance * unit_size_m
                if not 0.5 <= distance_m <= min(8.0, max(4.0, 0.75 * active[branch_id].width_m)):
                    continue
                along = target_line.project(Point(projection))
                if min(along, target_line.length - along) < 3.0 * metre:
                    continue
                heading = _endpoint_heading(active[branch_id].points, at_start, 12.0 * metre)
                toward = projection - endpoint
                toward /= max(float(np.linalg.norm(toward)), 1e-9)
                target_direction = active[target_id].points[segment + 1] - active[target_id].points[segment]
                target_direction /= max(float(np.linalg.norm(target_direction)), 1e-9)
                if float(np.dot(heading, toward)) < float(np.cos(np.deg2rad(25.0))):
                    continue
                if abs(float(np.dot(heading, target_direction))) > float(np.cos(np.deg2rad(55.0))):
                    continue
                connector = LineString([endpoint, projection])
                if _sampled_surface_line_support(connector, prep(surface_geometry)) < 0.85:
                    continue
                candidates.append((distance, branch_id, at_start, target_id, projection))
    used_targets: set[tuple[int, tuple[float, float]]] = set()
    for distance, branch_id, at_start, target_id, projection in sorted(candidates):
        if (branch_id, at_start) in used_endpoints:
            continue
        target_key = (target_id, _node_key(projection))
        if target_key in used_targets:
            continue
        branch = active[branch_id]
        target = active[target_id]
        endpoint = branch.points[0 if at_start else -1]
        connector = _surface_guided_gap_connector(
            endpoint, projection, branch.width_m, surface_geometry, unit_size_m,
        )
        if connector is None:
            continue
        if at_start:
            branch_points = _deduplicate_points(np.vstack((connector[::-1], branch.points[1:])))
        else:
            branch_points = _deduplicate_points(np.vstack((branch.points, connector[1:])))
        target_points = _insert_junction(target.points, projection, projection, endpoint_snap=0.0)
        active[branch_id] = _RegionalRoadSeed(
            branch_points, branch.width_m, branch.source_ids, "preserved_with_junction_touchup",
        )
        active[target_id] = _RegionalRoadSeed(
            target_points, target.width_m, target.source_ids, target.geometry_kind or "preserved",
        )
        used_endpoints.add((branch_id, at_start))
        used_targets.add(target_key)
        touchups += 1
        added_length_m += distance * unit_size_m
    return active, touchups, added_length_m


def _anchor_displacement(
    original_roads: list[_RegionalRoadSeed],
    anchor_source_ids: set[int],
    final_roads: list[_RegionalRoadSeed],
    unit_size_m: float,
) -> tuple[float, float]:
    distances: list[float] = []
    for original in original_roads:
        if not anchor_source_ids.intersection(original.source_ids):
            continue
        matches = [road for road in final_roads if set(road.source_ids).intersection(original.source_ids)]
        if not matches:
            continue
        geometry = min((LineString(road.points) for road in matches), key=lambda line: line.distance(Point(original.points[0])))
        distances.extend(geometry.distance(Point(point)) * unit_size_m for point in original.points)
    return (max(distances, default=0.0), float(np.mean(distances)) if distances else 0.0)


def _empty_cleanup_diagnostics() -> dict[str, float | int]:
    return {
        "anchor_line_count": 0,
        "parallel_track_count": 0,
        "same_track_gap_repair_count": 0,
        "same_track_gap_repair_length_m": 0.0,
        "local_offset_jump_repair_count": 0,
        "local_offset_jump_repair_length_m": 0.0,
        "local_loop_removed_count": 0,
        "local_loop_removed_length_m": 0.0,
        "same_track_local_path_removed_count": 0,
        "same_track_local_path_removed_length_m": 0.0,
        "duplicate_fragment_removed_count": 0,
        "cross_track_connection_rejected_count": 0,
        "anchor_max_displacement_m": 0.0,
        "anchor_mean_displacement_m": 0.0,
        "junction_touchup_count": 0,
        "original_feature_count": 0,
        "final_feature_count": 0,
        "original_vertex_count": 0,
        "final_vertex_count": 0,
        "merged_road_entity_count": 0,
        "generated_connection_length_m": 0.0,
        "generated_junction_count": 0,
        "canonical_straight_road_count": 0,
        "canonical_curved_road_count": 0,
        "mean_vertices_per_road_before": 0.0,
        "mean_vertices_per_road_after": 0.0,
        "endpoint_count_before": 0,
        "endpoint_count_after": 0,
        "connected_component_count_before": 0,
        "connected_component_count_after": 0,
        "axis_intersection_count": 0,
        "surface_center_correction_count": 0,
        "regional_regularization_seconds": 0.0,
    }


def _surface_polygon_components(surface_geometry) -> list:
    if surface_geometry is None or surface_geometry.is_empty:
        return []
    if surface_geometry.geom_type == "Polygon":
        return [surface_geometry]
    components = []
    for geometry in getattr(surface_geometry, "geoms", ()):
        if geometry.geom_type == "Polygon" and not geometry.is_empty:
            components.append(geometry)
        elif hasattr(geometry, "geoms"):
            components.extend(_surface_polygon_components(geometry))
    return components


def _surface_skeleton_adjacency(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    points = [tuple(int(value) for value in point) for point in np.argwhere(skeleton)]
    point_set = set(points)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, column in points:
        neighbors = []
        for row_step in (-1, 0, 1):
            for column_step in (-1, 0, 1):
                if not (row_step or column_step):
                    continue
                neighbor = (row + row_step, column + column_step)
                if neighbor not in point_set:
                    continue
                if row_step and column_step and (
                    (row + row_step, column) in point_set
                    or (row, column + column_step) in point_set
                ):
                    continue
                neighbors.append(neighbor)
        adjacency[(row, column)] = sorted(neighbors)
    return adjacency


def _prune_surface_skeleton_spurs(skeleton: np.ndarray, maximum_length_px: float) -> np.ndarray:
    active = (np.asarray(skeleton) > 0).astype(np.uint8)
    for _round in range(8):
        adjacency = _surface_skeleton_adjacency(active)
        endpoints = [point for point, neighbors in adjacency.items() if len(neighbors) <= 1]
        remove: set[tuple[int, int]] = set()
        for endpoint in endpoints:
            path = [endpoint]
            previous = None
            current = endpoint
            length = 0.0
            while True:
                following = [neighbor for neighbor in adjacency[current] if neighbor != previous]
                if len(following) != 1:
                    break
                neighbor = following[0]
                length += float(np.hypot(neighbor[0] - current[0], neighbor[1] - current[1]))
                previous, current = current, neighbor
                path.append(current)
                if len(adjacency[current]) != 2:
                    break
            if len(adjacency.get(current, ())) >= 3 and length <= maximum_length_px:
                remove.update(path[:-1])
        if not remove:
            break
        rows, columns = zip(*remove)
        active[np.asarray(rows), np.asarray(columns)] = 0
    return active


def _trace_surface_skeleton(skeleton: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    adjacency = _surface_skeleton_adjacency(skeleton)
    if not adjacency:
        return []
    junction_mask = np.zeros_like(skeleton, dtype=np.uint8)
    endpoints = []
    for point, neighbors in adjacency.items():
        if len(neighbors) >= 3:
            junction_mask[point] = 1
        elif len(neighbors) <= 1:
            endpoints.append(point)
    junction_count, junction_labels = cv2.connectedComponents(junction_mask, connectivity=8)
    group_pixels: dict[int, list[tuple[int, int]]] = {}
    point_group: dict[tuple[int, int], int] = {}
    for point in adjacency:
        group_id = int(junction_labels[point])
        if group_id > 0:
            group_pixels.setdefault(group_id, []).append(point)
            point_group[point] = group_id
    next_group = junction_count
    for point in endpoints:
        group_pixels[next_group] = [point]
        point_group[point] = next_group
        next_group += 1
    representatives = {
        group_id: np.mean(np.asarray(points, dtype=np.float64), axis=0)
        for group_id, points in group_pixels.items()
    }
    external: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {
        group_id: [] for group_id in group_pixels
    }
    for group_id, points in group_pixels.items():
        for point in points:
            for neighbor in adjacency[point]:
                if point_group.get(neighbor) != group_id:
                    external[group_id].append((point, neighbor))
    degrees = {group_id: len(links) for group_id, links in external.items()}
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def edge(first, second):
        return tuple(sorted((first, second)))

    paths: list[tuple[np.ndarray, int, int]] = []
    for start_group in sorted(external):
        for start_pixel, first in external[start_group]:
            if edge(start_pixel, first) in visited:
                continue
            pixels = [representatives[start_group], np.asarray(start_pixel), np.asarray(first)]
            visited.add(edge(start_pixel, first))
            previous, current = start_pixel, first
            end_group = point_group.get(current)
            while end_group is None:
                following = next((
                    neighbor for neighbor in adjacency[current]
                    if neighbor != previous and edge(current, neighbor) not in visited
                ), None)
                if following is None:
                    break
                visited.add(edge(current, following))
                pixels.append(np.asarray(following))
                previous, current = current, following
                end_group = point_group.get(current)
            if end_group is None:
                continue
            pixels.append(representatives[end_group])
            points = _deduplicate_points(np.asarray(pixels, dtype=np.float64))
            if points.shape[0] >= 2:
                paths.append((points, degrees[start_group], degrees[end_group]))
    for start in sorted(adjacency):
        for first in adjacency[start]:
            if edge(start, first) in visited or start in point_group or first in point_group:
                continue
            pixels = [np.asarray(start), np.asarray(first)]
            visited.add(edge(start, first))
            previous, current = start, first
            while current != start:
                following = next((
                    neighbor for neighbor in adjacency[current]
                    if neighbor != previous and edge(current, neighbor) not in visited
                ), None)
                if following is None:
                    break
                visited.add(edge(current, following))
                pixels.append(np.asarray(following))
                previous, current = current, following
            points = _deduplicate_points(np.asarray(pixels, dtype=np.float64))
            if points.shape[0] >= 2:
                paths.append((points, 2, 2))
    return paths


def _smooth_low_frequency_track(
    road: _RegionalRoadSeed,
    unit_size_m: float,
) -> _RegionalRoadSeed:
    """Remove metre-scale skeleton wobble while fixing both topology endpoints."""
    spacing = 2.5 / max(unit_size_m, 1e-9)
    points = _resample_polyline(np.asarray(road.points, dtype=np.float64), spacing)
    if points.shape[0] < 5:
        return road
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclength = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length_m = float(arclength[-1] * unit_size_m)
    if total_length_m < 12.0:
        return road
    radius = 12.5 / max(unit_size_m, 1e-9)
    endpoint_guard = 15.0 / max(unit_size_m, 1e-9)
    fitted = points.copy()
    for index, cursor in enumerate(arclength):
        selected = np.abs(arclength - cursor) <= radius
        if int(np.count_nonzero(selected)) < 5:
            continue
        local_s = arclength[selected] - cursor
        weights = np.exp(-0.5 * (local_s / max(0.55 * radius, 1e-9)) ** 2)
        degree = min(2, int(np.count_nonzero(selected)) - 1)
        for axis in range(2):
            coefficients = np.polyfit(
                local_s, points[selected, axis], degree, w=weights,
            )
            fitted[index, axis] = float(np.polyval(coefficients, 0.0))
    endpoint_distance = np.minimum(arclength, arclength[-1] - arclength)
    blend = np.clip(endpoint_distance / max(endpoint_guard, 1e-9), 0.0, 1.0)
    smoothed = points + blend[:, None] * (fitted - points)
    displacement = np.linalg.norm(smoothed - points, axis=1)
    maximum_shift = 3.0 / max(unit_size_m, 1e-9)
    excessive = displacement > maximum_shift
    if np.any(excessive):
        smoothed[excessive] = points[excessive] + (
            smoothed[excessive] - points[excessive]
        ) * (maximum_shift / displacement[excessive])[:, None]
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    simplified = LineString(smoothed).simplify(
        0.75 / max(unit_size_m, 1e-9), preserve_topology=False,
    )
    if simplified.geom_type != "LineString" or simplified.length <= 1e-8:
        return road
    return _RegionalRoadSeed(
        points=np.asarray(simplified.coords, dtype=np.float64),
        width_m=road.width_m,
        source_ids=road.source_ids,
        geometry_kind="smoothed_surface_track",
    )


def _broken_corridor_gap_candidate(
    roads: list[_RegionalRoadSeed],
    first_id: int,
    first_at_start: bool,
    second_id: int,
    second_at_start: bool,
    unit_size_m: float,
) -> _SameTrackGap | None:
    """Match a long missing interval only when both stable track ends predict it."""
    first, second = roads[first_id], roads[second_id]
    first_point = np.asarray(first.points[0 if first_at_start else -1], dtype=np.float64)
    second_point = np.asarray(second.points[0 if second_at_start else -1], dtype=np.float64)
    delta = second_point - first_point
    distance = float(np.linalg.norm(delta))
    distance_m = distance * unit_size_m
    if not 18.0 < distance_m <= 70.0:
        return None
    lookback = min(15.0, max(8.0, 0.25 * distance_m)) / max(unit_size_m, 1e-9)
    first_heading = _endpoint_heading(first.points, first_at_start, lookback)
    second_heading = _endpoint_heading(second.points, second_at_start, lookback)
    unit_delta = delta / max(distance, 1e-9)
    facing_cosine = float(np.cos(np.deg2rad(20.0)))
    continuation_cosine = float(np.cos(np.deg2rad(18.0)))
    if (
        float(np.dot(first_heading, unit_delta)) < facing_cosine
        or float(np.dot(second_heading, -unit_delta)) < facing_cosine
        or float(np.dot(first_heading, second_heading)) > -continuation_cosine
    ):
        return None
    track_direction = first_heading - second_heading
    track_direction /= max(float(np.linalg.norm(track_direction)), 1e-9)
    normal = np.asarray([-track_direction[1], track_direction[0]])
    lateral = abs(float(np.dot(delta, normal)))
    lateral_limit = min(3.0, max(1.5, 0.16 * min(first.width_m, second.width_m))) / max(
        unit_size_m, 1e-9,
    )
    if lateral > lateral_limit:
        return None
    observed_length_m = (
        _polyline_length(first.points) + _polyline_length(second.points)
    ) * unit_size_m
    if distance_m > 35.0 and observed_length_m < 70.0 and (
        len(first.source_ids) + len(second.source_ids) < 3
    ):
        return None
    return _SameTrackGap(
        first_id=first_id,
        first_at_start=first_at_start,
        second_id=second_id,
        second_at_start=second_at_start,
        distance=distance,
        lateral_offset=lateral,
        score=distance + 5.0 * lateral,
    )


def _recover_broken_corridors(
    roads: list[_RegionalRoadSeed],
    unit_size_m: float,
) -> tuple[list[_RegionalRoadSeed], int, float]:
    """Recover long fragmented tracks from mutually consistent endpoint bands."""
    active = list(roads)
    recovered = 0
    recovered_length_m = 0.0
    for _round in range(6):
        endpoints = [
            (road_id, at_start, np.asarray(road.points[0 if at_start else -1], dtype=np.float64))
            for road_id, road in enumerate(active) for at_start in (True, False)
        ]
        candidates: list[_SameTrackGap] = []
        cell_size = 70.0 / max(unit_size_m, 1e-9)
        buckets: dict[tuple[int, int], list[int]] = {}
        for endpoint_id, (_road_id, _at_start, point) in enumerate(endpoints):
            key = tuple(np.floor(point / cell_size).astype(int))
            buckets.setdefault(key, []).append(endpoint_id)
        for first_endpoint_id, (first_id, first_at_start, first_point) in enumerate(endpoints):
            key = tuple(np.floor(first_point / cell_size).astype(int))
            nearby = (
                endpoint_id
                for offset_x in (-1, 0, 1) for offset_y in (-1, 0, 1)
                for endpoint_id in buckets.get((key[0] + offset_x, key[1] + offset_y), ())
                if endpoint_id > first_endpoint_id
            )
            for second_endpoint_id in nearby:
                second_id, second_at_start, _second_point = endpoints[second_endpoint_id]
                if first_id == second_id:
                    continue
                candidate = _broken_corridor_gap_candidate(
                    active, first_id, first_at_start, second_id, second_at_start, unit_size_m,
                )
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            break
        best_by_endpoint: dict[tuple[int, bool], _SameTrackGap] = {}
        for candidate in sorted(candidates, key=lambda item: item.score):
            for endpoint in (
                (candidate.first_id, candidate.first_at_start),
                (candidate.second_id, candidate.second_at_start),
            ):
                best_by_endpoint.setdefault(endpoint, candidate)
        accepted = [
            candidate for candidate in candidates
            if best_by_endpoint.get((candidate.first_id, candidate.first_at_start)) is candidate
            and best_by_endpoint.get((candidate.second_id, candidate.second_at_start)) is candidate
        ]
        if not accepted:
            break
        used: set[int] = set()
        merged: list[_RegionalRoadSeed] = []
        for candidate in sorted(accepted, key=lambda item: item.score):
            if candidate.first_id in used or candidate.second_id in used:
                continue
            first, second = active[candidate.first_id], active[candidate.second_id]
            start = first.points[0 if candidate.first_at_start else -1]
            end = second.points[0 if candidate.second_at_start else -1]
            connector = np.asarray([start, end], dtype=np.float64)
            merged.append(_merge_same_track_gap(first, second, candidate, connector))
            used.update((candidate.first_id, candidate.second_id))
            recovered += 1
            recovered_length_m += candidate.distance * unit_size_m
        if not merged:
            break
        active = [road for road_id, road in enumerate(active) if road_id not in used] + merged
    return active, recovered, recovered_length_m


def reconstruct_regional_road_network_from_surface(
    roads: list[RegionalRoadObservation],
    surface_geometry,
    unit_size_m: float = 1.0,
) -> tuple[list[RegionalCanonicalRoad], dict[str, float | int]]:
    """Generate one medial path per road-surface ribbon.

    The extracted lines supply identity, width and coarse directional evidence.
    Their duplicate geometry is discarded; the surface strip determines track
    cardinality and its medial axis supplies the canonical geometry.
    """
    started = time.perf_counter()
    if not roads or surface_geometry is None or surface_geometry.is_empty:
        return regularize_regional_road_network(
            roads, unit_size_m=unit_size_m, surface_geometry=surface_geometry,
        )
    original_lines = [LineString(_deduplicate_points(road.points)) for road in roads]
    original_tree = STRtree(original_lines)
    axis_seeds: list[_RegionalRoadSeed] = []
    rasterized_components = 0
    pruned_spurs = 0
    for component in _surface_polygon_components(surface_geometry):
        if component.area * unit_size_m * unit_size_m < 20.0:
            continue
        minimum_x, minimum_y, maximum_x, maximum_y = component.bounds
        resolution_m = 1.0
        width_px = max(1, int(np.ceil((maximum_x - minimum_x) * unit_size_m / resolution_m)) + 6)
        height_px = max(1, int(np.ceil((maximum_y - minimum_y) * unit_size_m / resolution_m)) + 6)
        pixel_count = width_px * height_px
        if pixel_count > 8_000_000:
            resolution_m *= float(np.sqrt(pixel_count / 8_000_000.0))
            width_px = max(1, int(np.ceil((maximum_x - minimum_x) * unit_size_m / resolution_m)) + 6)
            height_px = max(1, int(np.ceil((maximum_y - minimum_y) * unit_size_m / resolution_m)) + 6)
        resolution = resolution_m / max(unit_size_m, 1e-9)
        origin_x = minimum_x - 3.0 * resolution
        origin_y = maximum_y + 3.0 * resolution
        transform = from_origin(origin_x, origin_y, resolution, resolution)
        mask = rasterize(
            [(component, 1)], out_shape=(height_px, width_px), transform=transform,
            fill=0, dtype=np.uint8,
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        try:
            from skimage.morphology import skeletonize
            skeleton = skeletonize(mask > 0).astype(np.uint8)
        except ImportError:
            skeleton = np.zeros_like(mask)
            working = mask.copy()
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while working.any():
                opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
                skeleton |= working & (1 - opened)
                working = cv2.erode(working, element)
        before_pixels = int(np.count_nonzero(skeleton))
        skeleton = _prune_surface_skeleton_spurs(skeleton, 12.0 / resolution_m)
        pruned_spurs += max(0, before_pixels - int(np.count_nonzero(skeleton)))
        rasterized_components += 1
        for pixels, start_degree, end_degree in _trace_surface_skeleton(skeleton):
            coordinates = np.column_stack((
                origin_x + (pixels[:, 1] + 0.5) * resolution,
                origin_y - (pixels[:, 0] + 0.5) * resolution,
            ))
            line = LineString(coordinates).simplify(1.25 / max(unit_size_m, 1e-9))
            if line.length * unit_size_m < 3.0:
                continue
            nearby = [int(value) for value in original_tree.query(line.buffer(12.0 / max(unit_size_m, 1e-9)))]
            nearby = [road_id for road_id in nearby if original_lines[road_id].distance(line) * unit_size_m <= 12.0]
            if not nearby:
                nearest_id = min(range(len(original_lines)), key=lambda road_id: original_lines[road_id].distance(line))
                nearby = [nearest_id]
            source_ids = tuple(sorted({int(roads[road_id].source_id) for road_id in nearby}))
            width_m = float(np.median([roads[road_id].width_m for road_id in nearby]))
            axis_seeds.append(_RegionalRoadSeed(
                points=np.asarray(line.coords, dtype=np.float64), width_m=width_m,
                source_ids=source_ids, geometry_kind="surface_strip_medial_axis",
            ))
    repaired_seeds, gap_count, gap_length_m, cross_rejected = _repair_same_track_gaps(
        axis_seeds, surface_geometry, unit_size_m,
    )
    repaired_seeds, broken_count, broken_length_m = _recover_broken_corridors(
        repaired_seeds, unit_size_m,
    )
    repaired_seeds = _merge_exact_degree_two_chains(repaired_seeds, unit_size_m)
    repaired_seeds = [
        _smooth_low_frequency_track(seed, unit_size_m) for seed in repaired_seeds
    ]
    outputs = [
        RegionalCanonicalRoad(
            points=seed.points, width_m=seed.width_m, source_ids=seed.source_ids,
            geometry_kind="surface_strip_medial_axis",
        )
        for seed in repaired_seeds
    ]
    raw_seeds = [
        _RegionalRoadSeed(
            points=_deduplicate_points(road.points), width_m=float(road.width_m),
            source_ids=(int(road.source_id),), geometry_kind="extracted",
        )
        for road in roads
    ]
    endpoint_count_before, component_count_before, _ = _network_topology_metrics(raw_seeds)
    endpoint_count_after, component_count_after, headings_after = _network_topology_metrics(repaired_seeds)
    original_vertices = int(sum(seed.points.shape[0] for seed in raw_seeds))
    final_vertices = int(sum(seed.points.shape[0] for seed in repaired_seeds))
    junction_count = int(sum(len(vectors) >= 3 for vectors in headings_after.values()))
    diagnostics = _empty_cleanup_diagnostics()
    diagnostics.update({
        "original_feature_count": int(len(roads)),
        "final_feature_count": int(len(outputs)),
        "original_vertex_count": original_vertices,
        "final_vertex_count": final_vertices,
        "merged_road_entity_count": int(len(roads) - len(outputs)),
        "surface_center_correction_count": int(rasterized_components),
        "same_track_local_path_removed_count": int(max(0, len(roads) - len(outputs))),
        "same_track_gap_repair_count": int(gap_count),
        "same_track_gap_repair_length_m": float(gap_length_m),
        "broken_corridor_recovery_count": int(broken_count),
        "broken_corridor_recovery_length_m": float(broken_length_m),
        "generated_connection_length_m": float(gap_length_m + broken_length_m),
        "cross_track_connection_rejected_count": int(cross_rejected),
        "generated_junction_count": junction_count,
        "mean_vertices_per_road_before": float(original_vertices / max(len(raw_seeds), 1)),
        "mean_vertices_per_road_after": float(final_vertices / max(len(repaired_seeds), 1)),
        "endpoint_count_before": int(endpoint_count_before),
        "endpoint_count_after": int(endpoint_count_after),
        "connected_component_count_before": int(component_count_before),
        "connected_component_count_after": int(component_count_after),
        "surface_axis_pruned_spur_pixel_count": int(pruned_spurs),
        "regional_regularization_seconds": float(time.perf_counter() - started),
    })
    return outputs, diagnostics


def regularize_regional_road_network(
    roads: list[RegionalRoadObservation],
    unit_size_m: float = 1.0,
    *,
    surface_geometry=None,
) -> tuple[list[RegionalCanonicalRoad], dict[str, float | int]]:
    """Preserve reliable geometry and repair only same-track local defects."""
    started = time.perf_counter()
    if not roads:
        return [], _empty_cleanup_diagnostics()
    raw_seeds = [
        _RegionalRoadSeed(
            points=_deduplicate_points(road.points),
            width_m=float(road.width_m),
            source_ids=(int(road.source_id),),
            geometry_kind="preserved",
        )
        for road in roads if _deduplicate_points(road.points).shape[0] >= 2
    ]
    endpoint_count_before, component_count_before, _headings = _network_topology_metrics(raw_seeds)
    anchor_ids = _detect_anchor_centerlines(raw_seeds, surface_geometry, unit_size_m)
    anchor_source_ids = {source_id for road_id in anchor_ids for source_id in raw_seeds[road_id].source_ids}
    without_duplicates, duplicate_count = _remove_same_track_duplicate_fragments(
        raw_seeds, anchor_ids, unit_size_m,
    )
    repaired, gap_count, gap_length_m, cross_rejected = _repair_same_track_gaps(
        without_duplicates, surface_geometry, unit_size_m,
    )
    repaired_anchor_ids = _detect_anchor_centerlines(
        repaired, surface_geometry, unit_size_m,
    )
    cleaned, post_repair_duplicate_count = _remove_same_track_duplicate_fragments(
        repaired, repaired_anchor_ids, unit_size_m,
    )
    duplicate_count += post_repair_duplicate_count
    loop_cleaned, loop_count, loop_length_m = _remove_short_self_loops(
        cleaned, surface_geometry, unit_size_m,
    )
    locally_repaired, offset_count, offset_length_m = _repair_local_offset_jumps(
        loop_cleaned, surface_geometry, unit_size_m,
    )
    corridor_selected, corridor_path_count, corridor_path_length_m = (
        _select_single_path_through_local_corridor_loops(
            locally_repaired, surface_geometry, unit_size_m,
        )
    )
    protected_parallel_pairs = _parallel_track_pairs(
        corridor_selected, unit_size_m, surface_geometry,
    )
    selected, feature_path_count, feature_path_length_m = _select_same_track_local_paths(
        corridor_selected, surface_geometry, unit_size_m, protected_parallel_pairs,
    )
    local_path_count = corridor_path_count + feature_path_count
    local_path_length_m = corridor_path_length_m + feature_path_length_m
    selected_anchor_ids = _detect_anchor_centerlines(selected, surface_geometry, unit_size_m)
    selected, final_duplicate_count = _remove_same_track_duplicate_fragments(
        selected, selected_anchor_ids, unit_size_m,
    )
    selected = _merge_exact_degree_two_chains(selected, unit_size_m)
    duplicate_count += final_duplicate_count
    parallel_pairs = _parallel_track_pairs(selected, unit_size_m, surface_geometry)
    final_seeds, junction_count, junction_length_m = _conservative_junction_touchup(
        selected, surface_geometry, unit_size_m,
    )
    anchor_max, anchor_mean = _anchor_displacement(
        raw_seeds, anchor_source_ids, final_seeds, unit_size_m,
    )
    outputs = [
        RegionalCanonicalRoad(
            points=np.asarray(seed.points, dtype=np.float64),
            width_m=seed.width_m,
            source_ids=seed.source_ids,
            geometry_kind=seed.geometry_kind or "preserved",
        )
        for seed in final_seeds
    ]
    original_vertices = int(sum(seed.points.shape[0] for seed in raw_seeds))
    final_vertices = int(sum(seed.points.shape[0] for seed in final_seeds))
    endpoint_count_after, component_count_after, _headings = _network_topology_metrics(final_seeds)
    diagnostics = _empty_cleanup_diagnostics()
    diagnostics.update({
        "anchor_line_count": int(len(anchor_ids)),
        "parallel_track_count": int(len(parallel_pairs)),
        "same_track_gap_repair_count": int(gap_count),
        "same_track_gap_repair_length_m": float(gap_length_m),
        "local_offset_jump_repair_count": int(offset_count),
        "local_offset_jump_repair_length_m": float(offset_length_m),
        "local_loop_removed_count": int(loop_count),
        "local_loop_removed_length_m": float(loop_length_m),
        "same_track_local_path_removed_count": int(local_path_count),
        "same_track_local_path_removed_length_m": float(local_path_length_m),
        "duplicate_fragment_removed_count": int(duplicate_count),
        "cross_track_connection_rejected_count": int(cross_rejected),
        "anchor_max_displacement_m": float(anchor_max),
        "anchor_mean_displacement_m": float(anchor_mean),
        "junction_touchup_count": int(junction_count),
        "original_feature_count": int(len(raw_seeds)),
        "final_feature_count": int(len(outputs)),
        "original_vertex_count": original_vertices,
        "final_vertex_count": final_vertices,
        "merged_road_entity_count": int(len(raw_seeds) - len(outputs)),
        "generated_connection_length_m": float(gap_length_m + junction_length_m),
        "generated_junction_count": int(junction_count),
        "mean_vertices_per_road_before": float(original_vertices / max(len(raw_seeds), 1)),
        "mean_vertices_per_road_after": float(final_vertices / max(len(outputs), 1)),
        "endpoint_count_before": int(endpoint_count_before),
        "endpoint_count_after": int(endpoint_count_after),
        "connected_component_count_before": int(component_count_before),
        "connected_component_count_after": int(component_count_after),
        "regional_regularization_seconds": float(time.perf_counter() - started),
    })
    return outputs, diagnostics


def write_before_after_visualization(
    before: list[np.ndarray],
    after: list[np.ndarray],
    target: Path,
) -> None:
    """Write an area overview that exposes geometry and vertex-count changes."""
    all_points = [np.asarray(points) for points in [*before, *after] if len(points)]
    if not all_points:
        return
    stacked = np.vstack(all_points)
    minimum = stacked.min(axis=0)
    maximum = stacked.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    panel_width, panel_height, margin = 900, 900, 24
    scale = min((panel_width - 2 * margin) / span[0], (panel_height - 2 * margin) / span[1])

    def panel(lines: list[np.ndarray], title: str, color: tuple[int, int, int]) -> np.ndarray:
        canvas = np.full((panel_height, panel_width, 3), 250, dtype=np.uint8)
        adjacency: dict[tuple[float, float], set[tuple[float, float]]] = {}
        for points in lines:
            normalized = (np.asarray(points) - minimum) * scale + margin
            pixels = np.column_stack((normalized[:, 0], panel_height - normalized[:, 1])).astype(np.int32)
            cv2.polylines(canvas, [pixels], False, color, 2, cv2.LINE_AA)
            for first, second in zip(np.asarray(points), np.asarray(points)[1:]):
                first_key, second_key = _node_key(first), _node_key(second)
                adjacency.setdefault(first_key, set()).add(second_key)
                adjacency.setdefault(second_key, set()).add(first_key)
        for node, neighbors in adjacency.items():
            normalized = (np.asarray(node) - minimum) * scale + margin
            pixel = tuple(np.rint([normalized[0], panel_height - normalized[1]]).astype(int))
            if len(neighbors) == 1:
                cv2.circle(canvas, pixel, 2, (30, 30, 230), -1, cv2.LINE_AA)
            elif len(neighbors) >= 3:
                cv2.circle(canvas, pixel, 3, (35, 160, 35), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (panel_width, 42), (255, 255, 255), -1)
        cv2.putText(canvas, title, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        return canvas

    comparison = np.hstack((
        panel(before, "BEFORE: traced centerline observations", (80, 80, 210)),
        panel(after, "AFTER: local centerline cleanup", (225, 90, 30)),
    ))
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded, buffer = cv2.imencode(".png", comparison)
    if not encoded:
        raise OSError(f"Could not encode canonical road comparison: {target}")
    buffer.tofile(target)
