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


@dataclass(frozen=True)
class RegionalCanonicalRoad:
    points: np.ndarray
    width_m: float
    source_ids: tuple[int, ...]
    geometry_kind: str


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


def _regional_endpoint_assignments(
    roads: list[RegionalRoadObservation],
    unit_size_m: float,
) -> tuple[dict[tuple[int, bool], np.ndarray], float]:
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
    return assignments, connection_length * float(unit_size_m)


def _regional_entities(
    roads: list[RegionalRoadObservation],
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
            source_ids.append(roads[road_id].source_id)
            paired = continuation.get((road_id, next_node))
            if paired is None or paired[0] in visited:
                break
            road_id, node = paired
        entities.append(np.asarray(points, dtype=np.float64))
        source_groups.append(tuple(source_ids))

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


def regularize_regional_road_network(
    roads: list[RegionalRoadObservation],
    unit_size_m: float = 1.0,
) -> tuple[list[RegionalCanonicalRoad], dict[str, float | int]]:
    """Merge tile observations into area-level roads and regenerate their axes."""
    started = time.perf_counter()
    if not roads:
        return [], {
            "original_feature_count": 0, "final_feature_count": 0,
            "original_vertex_count": 0, "final_vertex_count": 0,
            "merged_road_entity_count": 0, "generated_connection_length_m": 0.0,
            "regional_regularization_seconds": 0.0,
        }
    assignments, connection_length_m = _regional_endpoint_assignments(roads, unit_size_m)
    entities, source_groups = _regional_entities(roads, assignments, unit_size_m)
    canonical, _junctions, geometry_diagnostics = fit_canonical_road_geometry(
        entities, unit_size_m,
    )
    outputs: list[RegionalCanonicalRoad] = []
    for road, source_ids in zip(canonical, source_groups):
        source_roads = [roads[source_id] for source_id in source_ids]
        lengths = np.asarray([max(_polyline_length(item.points), 1e-9) for item in source_roads])
        width_m = float(np.average([item.width_m for item in source_roads], weights=lengths))
        outputs.append(RegionalCanonicalRoad(
            points=road.points,
            width_m=width_m,
            source_ids=tuple(sorted(set(source_ids))),
            geometry_kind=road.geometry_kind,
        ))
    original_vertices = int(sum(np.asarray(road.points).shape[0] for road in roads))
    final_vertices = int(sum(road.points.shape[0] for road in outputs))
    diagnostics: dict[str, float | int] = {
        "original_feature_count": int(len(roads)),
        "final_feature_count": int(len(outputs)),
        "original_vertex_count": original_vertices,
        "final_vertex_count": final_vertices,
        "generated_junction_count": int(geometry_diagnostics.get("generated_junction_count", 0)),
        "merged_road_entity_count": int(len(roads) - len(outputs)),
        "canonical_straight_road_count": int(sum(road.geometry_kind == "straight" for road in outputs)),
        "canonical_curved_road_count": int(sum(road.geometry_kind == "curved" for road in outputs)),
        "mean_vertices_per_road_before": float(original_vertices / max(len(roads), 1)),
        "mean_vertices_per_road_after": float(final_vertices / max(len(outputs), 1)),
        "generated_connection_length_m": float(connection_length_m),
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
        for points in lines:
            normalized = (np.asarray(points) - minimum) * scale + margin
            pixels = np.column_stack((normalized[:, 0], panel_height - normalized[:, 1])).astype(np.int32)
            cv2.polylines(canvas, [pixels], False, color, 2, cv2.LINE_AA)
            for point in pixels:
                cv2.circle(canvas, tuple(point), 1, (35, 35, 35), -1, cv2.LINE_AA)
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
