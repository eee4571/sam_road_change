from __future__ import annotations

"""Geometry regeneration for canonical road entities.

The input polylines are observations.  Their vertices are sampled as evidence and
are never copied directly to the final geometry.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import time

import cv2
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
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
class RoadCorridorCandidate:
    first_road_id: int
    second_road_id: int
    dominant_heading: np.ndarray
    lateral_distance: float
    longitudinal_overlap: float
    surface_support: float
    different_source_tiles: bool
    confidence: float


@dataclass(frozen=True)
class CanonicalJunctionCandidate:
    road_ids: tuple[int, ...]
    point: np.ndarray
    junction_type: str
    score: float
    branch_endpoints: tuple[tuple[int, bool], ...] = ()
    inference_kind: str = "axis_intersection"


@dataclass(frozen=True)
class CanonicalRoadGraph:
    roads: tuple[RegionalCanonicalRoad, ...]
    junctions: tuple[CanonicalJunctionCandidate, ...]


@dataclass(frozen=True)
class _RegionalRoadSeed:
    points: np.ndarray
    width_m: float
    source_ids: tuple[int, ...]
    geometry_kind: str = ""


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


def _find_road_corridor_candidates(
    roads: list[RegionalRoadObservation],
    unit_size_m: float,
    surface_geometry=None,
) -> list[RoadCorridorCandidate]:
    """Find overlapping observations of one corridor without merging carriageways."""
    if not roads:
        return []
    metre = 1.0 / max(float(unit_size_m), 1e-9)
    geometries = [LineString(_deduplicate_points(road.points)) for road in roads]
    tree = STRtree(geometries)
    summaries = [_road_axis_summary(road.points) for road in roads]
    candidates: list[RoadCorridorCandidate] = []
    heading_cosine = float(np.cos(np.deg2rad(12.0)))
    for first_id, first in enumerate(roads):
        first_line = geometries[first_id]
        search_distance = max(3.0, 0.35 * first.width_m) * metre
        for second_value in tree.query(first_line.buffer(search_distance)):
            second_id = int(second_value)
            if second_id <= first_id:
                continue
            second = roads[second_id]
            width_ratio = max(first.width_m, second.width_m) / max(
                min(first.width_m, second.width_m), 1e-6,
            )
            if width_ratio > 1.65:
                continue
            first_center, first_direction, _first_min, _first_max = summaries[first_id]
            second_center, second_direction, _second_min, _second_max = summaries[second_id]
            alignment = abs(float(np.dot(first_direction, second_direction)))
            if alignment < heading_cosine:
                continue
            direction = first_direction
            if float(np.dot(direction, second_direction)) < 0:
                second_direction = -second_direction
            direction = direction + second_direction
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            normal = np.asarray([-direction[1], direction[0]])
            lateral = abs(float(np.dot(second_center - first_center, normal)))
            lateral_limit = max(3.0, 0.35 * min(first.width_m, second.width_m)) * metre
            if lateral > lateral_limit:
                continue
            intervals = []
            for road in (first, second):
                projections = (_deduplicate_points(road.points) - first_center) @ direction
                intervals.append((float(np.min(projections)), float(np.max(projections))))
            overlap = max(0.0, min(intervals[0][1], intervals[1][1]) - max(intervals[0][0], intervals[1][0]))
            shorter_span = max(1e-9, min(
                intervals[0][1] - intervals[0][0], intervals[1][1] - intervals[1][0],
            ))
            overlap_ratio = overlap / shorter_span
            if overlap_ratio < 0.30:
                continue
            first_near, second_near = nearest_points(first_line, geometries[second_id])
            lateral_connector = LineString([first_near.coords[0], second_near.coords[0]])
            surface_support = _surface_line_support(
                lateral_connector, surface_geometry,
            ) if lateral_connector.length > 0 else 1.0
            different_tiles = bool(
                first.source_tile and second.source_tile
                and first.source_tile != second.source_tile
            )
            confidence = (
                0.32 * alignment
                + 0.25 * min(1.0, overlap_ratio)
                + 0.18 * max(0.0, 1.0 - lateral / max(lateral_limit, 1e-9))
                + 0.10 * min(1.0, 1.0 / width_ratio)
                + 0.10 * surface_support
                + 0.05 * (1.0 if different_tiles else 0.5)
            )
            if confidence < 0.70:
                continue
            candidates.append(RoadCorridorCandidate(
                first_road_id=first_id,
                second_road_id=second_id,
                dominant_heading=direction.astype(np.float32),
                lateral_distance=float(lateral),
                longitudinal_overlap=float(overlap),
                surface_support=float(surface_support),
                different_source_tiles=different_tiles,
                confidence=float(confidence),
            ))
    return sorted(candidates, key=lambda candidate: -candidate.confidence)


def _merge_corridor_observations(
    roads: list[RegionalRoadObservation],
    unit_size_m: float,
    surface_geometry=None,
) -> tuple[list[_RegionalRoadSeed], int]:
    candidates = _find_road_corridor_candidates(
        roads, unit_size_m, surface_geometry,
    )
    parent = list(range(len(roads)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for candidate in candidates:
        first = find(candidate.first_road_id)
        second = find(candidate.second_road_id)
        if first != second:
            parent[second] = first
    groups: dict[int, list[int]] = {}
    for road_id in range(len(roads)):
        groups.setdefault(find(road_id), []).append(road_id)

    seeds: list[_RegionalRoadSeed] = []
    spacing = 2.0 / max(float(unit_size_m), 1e-9)
    for road_ids in groups.values():
        if len(road_ids) == 1:
            road = roads[road_ids[0]]
            seeds.append(_RegionalRoadSeed(
                _deduplicate_points(road.points), road.width_m, (road.source_id,),
            ))
            continue
        samples = np.vstack([
            _resample_polyline(roads[road_id].points, spacing) for road_id in road_ids
        ])
        center, direction, _inliers = _robust_axis(
            samples, lateral_outlier=5.0 / max(float(unit_size_m), 1e-9),
        )
        longitudinal = (samples - center) @ direction
        order = np.argsort(longitudinal)
        ordered = samples[order]
        positions = longitudinal[order]
        bin_size = 3.0 / max(float(unit_size_m), 1e-9)
        bins = np.floor((positions - positions[0]) / max(bin_size, 1e-9)).astype(int)
        corridor_points = np.asarray([
            np.median(ordered[bins == bin_id], axis=0)
            for bin_id in np.unique(bins)
        ])
        if corridor_points.shape[0] < 2 or _polyline_length(corridor_points) <= 1e-8:
            longest_id = max(
                road_ids, key=lambda road_id: _polyline_length(roads[road_id].points),
            )
            corridor_points = _deduplicate_points(roads[longest_id].points)
        lengths = np.asarray([
            max(_polyline_length(roads[road_id].points), 1e-9) for road_id in road_ids
        ])
        seeds.append(_RegionalRoadSeed(
            points=_deduplicate_points(corridor_points),
            width_m=float(np.average([roads[road_id].width_m for road_id in road_ids], weights=lengths)),
            source_ids=tuple(sorted(roads[road_id].source_id for road_id in road_ids)),
        ))
    return seeds, int(len(roads) - len(seeds))


def _fit_regional_seeds(
    seeds: list[_RegionalRoadSeed],
    unit_size_m: float,
) -> list[_RegionalRoadSeed]:
    canonical, _junctions, _diagnostics = fit_canonical_road_geometry(
        [seed.points for seed in seeds], unit_size_m,
    )
    return [
        _RegionalRoadSeed(
            points=road.points,
            width_m=seeds[road.road_id].width_m,
            source_ids=seeds[road.road_id].source_ids,
            geometry_kind=road.geometry_kind,
        )
        for road in canonical
    ]


def _regional_endpoint_assignments(
    roads: list[_RegionalRoadSeed],
    unit_size_m: float,
) -> tuple[dict[tuple[int, bool], np.ndarray], float, int]:
    max_gap = 18.0 / max(float(unit_size_m), 1e-9)
    lookback = 25.0 / max(float(unit_size_m), 1e-9)
    endpoints: list[tuple[int, bool, np.ndarray, np.ndarray]] = []
    for road_id, road in enumerate(roads):
        points = _deduplicate_points(road.points)
        endpoints.extend((
            (road_id, True, points[0], _endpoint_heading(points, True, lookback)),
            (road_id, False, points[-1], _endpoint_heading(points, False, lookback)),
        ))
    cell = max(max_gap, 1e-9)
    buckets: dict[tuple[int, int], list[int]] = {}
    for endpoint_id, (_road, _start, point, _heading) in enumerate(endpoints):
        key = tuple(np.floor(point / cell).astype(int))
        buckets.setdefault(key, []).append(endpoint_id)
    direction_cosine = float(np.cos(np.deg2rad(32.0)))
    facing_cosine = float(np.cos(np.deg2rad(38.0)))
    candidates: list[tuple[float, int, int]] = []
    for first_id, (first_road, _first_start, first_point, first_heading) in enumerate(endpoints):
        key = tuple(np.floor(first_point / cell).astype(int))
        for offset in ((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)):
            nearby_key = (key[0] + offset[0], key[1] + offset[1])
            for second_id in buckets.get(nearby_key, []):
                if second_id <= first_id:
                    continue
                second_road, _second_start, second_point, second_heading = endpoints[second_id]
                if first_road == second_road:
                    continue
                distance = float(np.linalg.norm(second_point - first_point))
                if distance > max_gap:
                    continue
                direction_alignment = float(np.dot(first_heading, second_heading))
                if direction_alignment > -direction_cosine:
                    continue
                if distance > 0.75:
                    connector = (second_point - first_point) / distance
                    if (
                        float(np.dot(first_heading, connector)) < facing_cosine
                        or float(np.dot(second_heading, -connector)) < facing_cosine
                    ):
                        continue
                score = distance + max_gap * (1.0 + direction_alignment)
                candidates.append((score, first_id, second_id))
    assignments: dict[tuple[int, bool], np.ndarray] = {}
    used: set[int] = set()
    connection_length = 0.0
    for _score, first_id, second_id in sorted(candidates):
        if first_id in used or second_id in used:
            continue
        first = endpoints[first_id]
        second = endpoints[second_id]
        node = 0.5 * (first[2] + second[2])
        assignments[(first[0], first[1])] = node
        assignments[(second[0], second[1])] = node
        connection_length += float(np.linalg.norm(first[2] - second[2]))
        used.update((first_id, second_id))
    return assignments, connection_length * float(unit_size_m), int(len(used) // 2)


def _regional_entities(
    roads: list[_RegionalRoadSeed],
    assignments: dict[tuple[int, bool], np.ndarray],
    unit_size_m: float,
) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
    points_by_road: list[np.ndarray] = []
    endpoint_keys: list[tuple[tuple[float, float], tuple[float, float]]] = []
    incident: dict[tuple[float, float], list[tuple[int, bool]]] = {}
    for road_id, road in enumerate(roads):
        points = _deduplicate_points(road.points)
        start = assignments.get((road_id, True))
        end = assignments.get((road_id, False))
        if start is not None and np.linalg.norm(start - points[0]) > 1e-8:
            points = np.vstack((start, points))
        if end is not None and np.linalg.norm(end - points[-1]) > 1e-8:
            points = np.vstack((points, end))
        points_by_road.append(points)
        start_key, end_key = _node_key(points[0]), _node_key(points[-1])
        endpoint_keys.append((start_key, end_key))
        incident.setdefault(start_key, []).append((road_id, True))
        incident.setdefault(end_key, []).append((road_id, False))

    continuation: dict[tuple[int, tuple[float, float]], tuple[int, tuple[float, float]]] = {}
    lookback = 25.0 / max(float(unit_size_m), 1e-9)
    cosine = float(np.cos(np.deg2rad(32.0)))
    for node, arms in incident.items():
        if len(arms) == 2:
            first, second = arms
            continuation[(first[0], node)] = (second[0], node)
            continuation[(second[0], node)] = (first[0], node)
            continue
        headings = [
            _endpoint_heading(points_by_road[road_id], at_start, lookback)
            for road_id, at_start in arms
        ]
        candidates = sorted(
            (float(np.dot(headings[a], headings[b])), a, b)
            for a, b in combinations(range(len(arms)), 2)
            if arms[a][0] != arms[b][0]
            and float(np.dot(headings[a], headings[b])) <= -cosine
        )
        used: set[int] = set()
        for _alignment, first_id, second_id in candidates:
            if first_id in used or second_id in used:
                continue
            first, second = arms[first_id], arms[second_id]
            continuation[(first[0], node)] = (second[0], node)
            continuation[(second[0], node)] = (first[0], node)
            used.update((first_id, second_id))

    visited: set[int] = set()
    entities: list[np.ndarray] = []
    source_groups: list[tuple[int, ...]] = []

    def walk(first_road: int, start_node: tuple[float, float]) -> None:
        road_id, node = first_road, start_node
        points: list[np.ndarray] = []
        source_ids: list[int] = []
        while road_id not in visited:
            visited.add(road_id)
            geometry = points_by_road[road_id]
            start, end = endpoint_keys[road_id]
            if start == node:
                oriented, next_node = geometry, end
            else:
                oriented, next_node = geometry[::-1], start
            points.extend(oriented if not points else oriented[1:])
            source_ids.extend(roads[road_id].source_ids)
            paired = continuation.get((road_id, next_node))
            if paired is None or paired[0] in visited:
                break
            road_id, node = paired
        entities.append(np.asarray(points, dtype=np.float64))
        source_groups.append(tuple(sorted(set(source_ids))))

    for road_id, (start, end) in enumerate(endpoint_keys):
        if road_id in visited:
            continue
        if (road_id, start) not in continuation:
            walk(road_id, start)
        elif (road_id, end) not in continuation:
            walk(road_id, end)
    for road_id, (start, _end) in enumerate(endpoint_keys):
        if road_id not in visited:
            walk(road_id, start)
    return entities, source_groups


def _surface_line_support(line: LineString, surface_geometry) -> float:
    if surface_geometry is None or line.length <= 0:
        return 0.5
    try:
        return float(min(1.0, line.intersection(surface_geometry).length / line.length))
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
                support = _surface_line_support(connector, surface_geometry)
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
                _surface_line_support(connector, surface_geometry) for connector in connectors
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
        output.append(_RegionalRoadSeed(
            points=_deduplicate_points(fitted).astype(np.float32),
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


def _count_connectable_dangling_endpoints(
    roads: list[_RegionalRoadSeed],
    surface_geometry,
    unit_size_m: float,
) -> int:
    assignments, _length, _count = _regional_endpoint_assignments(roads, unit_size_m)
    attachments = _infer_endpoint_to_road_attachments(roads, surface_geometry, unit_size_m)
    candidates = set(assignments)
    candidates.update(
        endpoint for attachment in attachments for endpoint in attachment.branch_endpoints
    )
    return int(len(candidates))


def _endpoint_connections_as_junctions(
    roads: list[_RegionalRoadSeed],
    unit_size_m: float,
) -> tuple[list[CanonicalJunctionCandidate], float, int]:
    assignments, length_m, pair_count = _regional_endpoint_assignments(
        roads, unit_size_m,
    )
    grouped: dict[tuple[float, float], list[tuple[int, bool]]] = {}
    points: dict[tuple[float, float], np.ndarray] = {}
    for endpoint, node in assignments.items():
        key = _node_key(node)
        grouped.setdefault(key, []).append(endpoint)
        points[key] = node
    junctions = [
        CanonicalJunctionCandidate(
            road_ids=tuple(sorted({road_id for road_id, _at_start in endpoints})),
            point=np.asarray(points[key], dtype=np.float32),
            junction_type="continuation",
            score=1.0,
            branch_endpoints=tuple(sorted(endpoints)),
            inference_kind="endpoint_to_endpoint",
        )
        for key, endpoints in grouped.items() if len(endpoints) >= 2
    ]
    return junctions, length_m, pair_count


def regularize_regional_road_network(
    roads: list[RegionalRoadObservation],
    unit_size_m: float = 1.0,
    *,
    surface_geometry=None,
) -> tuple[list[RegionalCanonicalRoad], dict[str, float | int]]:
    """Infer area-level corridors, continuations, junctions, and a canonical graph."""
    started = time.perf_counter()
    if not roads:
        return [], {
            "original_feature_count": 0, "final_feature_count": 0,
            "original_vertex_count": 0, "final_vertex_count": 0,
            "merged_road_entity_count": 0, "generated_connection_length_m": 0.0,
            "generated_junction_count": 0,
            "canonical_straight_road_count": 0, "canonical_curved_road_count": 0,
            "mean_vertices_per_road_before": 0.0, "mean_vertices_per_road_after": 0.0,
            "endpoint_count_before": 0, "endpoint_count_after": 0,
            "dangling_endpoint_count_before": 0, "dangling_endpoint_count_after": 0,
            "t_junction_count": 0, "cross_junction_count": 0, "y_junction_count": 0,
            "endpoint_to_endpoint_connection_count": 0,
            "endpoint_to_road_attachment_count": 0, "corridor_merge_count": 0,
            "axis_intersection_count": 0,
            "connected_component_count_before": 0,
            "connected_component_count_after": 0,
            "surface_center_correction_count": 0,
            "regional_regularization_seconds": 0.0,
        }
    raw_seeds = [
        _RegionalRoadSeed(
            points=_deduplicate_points(road.points),
            width_m=road.width_m,
            source_ids=(road.source_id,),
        )
        for road in roads
    ]
    endpoint_count_before, component_count_before, _headings = _network_topology_metrics(raw_seeds)
    dangling_before = _count_connectable_dangling_endpoints(
        raw_seeds, surface_geometry, unit_size_m,
    )

    corridor_seeds, corridor_merge_count = _merge_corridor_observations(
        roads, unit_size_m, surface_geometry,
    )
    preliminary = _fit_regional_seeds(corridor_seeds, unit_size_m)
    assignments, connection_length_m, endpoint_connection_count = (
        _regional_endpoint_assignments(preliminary, unit_size_m)
    )
    entities, source_groups = _regional_entities(
        preliminary, assignments, unit_size_m,
    )
    entity_seeds: list[_RegionalRoadSeed] = []
    for points, source_ids in zip(entities, source_groups):
        source_roads = [roads[source_id] for source_id in source_ids]
        lengths = np.asarray([
            max(_polyline_length(source.points), 1e-9) for source in source_roads
        ])
        entity_seeds.append(_RegionalRoadSeed(
            points=points,
            width_m=float(np.average(
                [source.width_m for source in source_roads], weights=lengths,
            )),
            source_ids=source_ids,
        ))
    fitted_entities = _fit_regional_seeds(entity_seeds, unit_size_m)
    fitted_entities, surface_center_correction_count = _center_straight_roads_on_surface(
        fitted_entities, surface_geometry, unit_size_m,
    )
    secondary_assignments, secondary_length_m, secondary_connection_count = (
        _regional_endpoint_assignments(fitted_entities, unit_size_m)
    )
    if secondary_connection_count:
        secondary_entities, secondary_source_groups = _regional_entities(
            fitted_entities, secondary_assignments, unit_size_m,
        )
        secondary_seeds: list[_RegionalRoadSeed] = []
        for points, source_ids in zip(secondary_entities, secondary_source_groups):
            source_roads = [roads[source_id] for source_id in source_ids]
            lengths = np.asarray([
                max(_polyline_length(source.points), 1e-9) for source in source_roads
            ])
            secondary_seeds.append(_RegionalRoadSeed(
                points=points,
                width_m=float(np.average(
                    [source.width_m for source in source_roads], weights=lengths,
                )),
                source_ids=source_ids,
            ))
        fitted_entities = _fit_regional_seeds(secondary_seeds, unit_size_m)
        fitted_entities, secondary_center_count = _center_straight_roads_on_surface(
            fitted_entities, surface_geometry, unit_size_m,
        )
        surface_center_correction_count += secondary_center_count
        connection_length_m += secondary_length_m
        endpoint_connection_count += secondary_connection_count
    attachment_candidates = _infer_endpoint_to_road_attachments(
        fitted_entities, surface_geometry, unit_size_m,
    )
    axis_candidates = _infer_axis_intersections(
        fitted_entities, surface_geometry, unit_size_m,
    )
    junctions = _cluster_junction_candidates(
        [*attachment_candidates, *axis_candidates], fitted_entities, unit_size_m,
    )
    final_seeds = _apply_junction_constraints(
        fitted_entities, junctions, unit_size_m,
    )
    attachment_count = len(attachment_candidates)
    for _iteration in range(2):
        existing_endpoints = {
            endpoint for junction in junctions for endpoint in junction.branch_endpoints
        }
        endpoint_junctions, _extra_length_m, extra_endpoint_count = (
            _endpoint_connections_as_junctions(final_seeds, unit_size_m)
        )
        endpoint_junctions = [
            candidate for candidate in endpoint_junctions
            if any(endpoint not in existing_endpoints for endpoint in candidate.branch_endpoints)
        ]
        extra_attachments = [
            candidate for candidate in _infer_endpoint_to_road_attachments(
                final_seeds, surface_geometry, unit_size_m,
            )
            if any(endpoint not in existing_endpoints for endpoint in candidate.branch_endpoints)
        ]
        if not endpoint_junctions and not extra_attachments:
            break
        endpoint_connection_count += min(extra_endpoint_count, len(endpoint_junctions))
        attachment_count += len(extra_attachments)
        junctions = _cluster_junction_candidates(
            [*junctions, *endpoint_junctions, *extra_attachments],
            fitted_entities,
            unit_size_m,
        )
        final_seeds = _apply_junction_constraints(
            fitted_entities, junctions, unit_size_m,
        )
    inferred_connection_length_m = float(sum(
        LineString(fitted_entities[road_id].points).distance(Point(junction.point))
        for junction in junctions for road_id in junction.road_ids
    ) * unit_size_m)
    connection_length_m += inferred_connection_length_m
    outputs = [
        RegionalCanonicalRoad(
            points=seed.points,
            width_m=seed.width_m,
            source_ids=seed.source_ids,
            geometry_kind=seed.geometry_kind,
        )
        for seed in final_seeds
    ]
    graph = CanonicalRoadGraph(tuple(outputs), tuple(junctions))
    outputs = list(graph.roads)
    original_vertices = int(sum(np.asarray(road.points).shape[0] for road in roads))
    final_vertices = int(sum(road.points.shape[0] for road in outputs))
    endpoint_count_after, component_count_after, _headings = _network_topology_metrics(final_seeds)
    dangling_after = _count_connectable_dangling_endpoints(
        final_seeds, surface_geometry, unit_size_m,
    )
    t_count = int(sum(junction.junction_type == "t" for junction in junctions))
    cross_count = int(sum(junction.junction_type == "cross" for junction in junctions))
    y_count = int(sum(junction.junction_type == "y" for junction in junctions))
    diagnostics: dict[str, float | int] = {
        "original_feature_count": int(len(roads)),
        "final_feature_count": int(len(outputs)),
        "original_vertex_count": original_vertices,
        "final_vertex_count": final_vertices,
        "generated_junction_count": int(len(junctions)),
        "merged_road_entity_count": int(len(roads) - len(outputs)),
        "canonical_straight_road_count": int(sum(road.geometry_kind == "straight" for road in outputs)),
        "canonical_curved_road_count": int(sum(road.geometry_kind == "curved" for road in outputs)),
        "mean_vertices_per_road_before": float(original_vertices / max(len(roads), 1)),
        "mean_vertices_per_road_after": float(final_vertices / max(len(outputs), 1)),
        "generated_connection_length_m": float(connection_length_m),
        "endpoint_count_before": int(endpoint_count_before),
        "endpoint_count_after": int(endpoint_count_after),
        "dangling_endpoint_count_before": int(dangling_before),
        "dangling_endpoint_count_after": int(dangling_after),
        "t_junction_count": t_count,
        "cross_junction_count": cross_count,
        "y_junction_count": y_count,
        "endpoint_to_endpoint_connection_count": int(endpoint_connection_count),
        "endpoint_to_road_attachment_count": int(attachment_count),
        "corridor_merge_count": int(corridor_merge_count),
        "axis_intersection_count": int(sum(
            "axis_intersection" in junction.inference_kind
            for junction in graph.junctions
        )),
        "connected_component_count_before": int(component_count_before),
        "connected_component_count_after": int(component_count_after),
        "surface_center_correction_count": int(surface_center_correction_count),
        "regional_regularization_seconds": float(time.perf_counter() - started),
    }
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
        panel(after, "AFTER: canonical road axes", (225, 90, 30)),
    ))
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded, buffer = cv2.imencode(".png", comparison)
    if not encoded:
        raise OSError(f"Could not encode canonical road comparison: {target}")
    buffer.tofile(target)
