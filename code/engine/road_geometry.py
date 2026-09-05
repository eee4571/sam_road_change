from __future__ import annotations

"""Shared metric track geometry for reconstruction and connection stages."""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString


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


def _node_key(point: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.round(np.asarray(point), 3))


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


def _sampled_surface_line_support(line: LineString, prepared_surface) -> float:
    if prepared_surface is None or line.length <= 0:
        return 0.5
    sample_count = max(3, min(9, int(np.ceil(line.length / 3.0)) + 1))
    samples = [line.interpolate(fraction, normalized=True) for fraction in np.linspace(0.0, 1.0, sample_count)]
    try:
        return float(np.mean([prepared_surface.covers(point) for point in samples]))
    except Exception:
        return 0.5


def _tangent_continuous_connector(
    start: np.ndarray,
    start_heading: np.ndarray,
    end: np.ndarray,
    end_heading: np.ndarray,
    unit_size_m: float,
) -> np.ndarray | None:
    """Generate a monotone cubic Hermite connector from two outward headings."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-8:
        return None
    chord = delta / distance
    first = np.asarray(start_heading, dtype=np.float64)
    second = -np.asarray(end_heading, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1e-9)
    second /= max(float(np.linalg.norm(second)), 1e-9)
    if min(float(np.dot(first, chord)), float(np.dot(second, chord))) <= 0.35:
        return None
    tangent_scale = distance * min(
        1.0,
        0.70 + 0.30 * min(float(np.dot(first, chord)), float(np.dot(second, chord))),
    )
    sample_count = max(9, min(61, int(np.ceil(distance * unit_size_m / 2.0)) + 1))
    parameter = np.linspace(0.0, 1.0, sample_count)
    h00 = 2.0 * parameter ** 3 - 3.0 * parameter ** 2 + 1.0
    h10 = parameter ** 3 - 2.0 * parameter ** 2 + parameter
    h01 = -2.0 * parameter ** 3 + 3.0 * parameter ** 2
    h11 = parameter ** 3 - parameter ** 2
    connector = (
        h00[:, None] * start
        + h10[:, None] * tangent_scale * first
        + h01[:, None] * end
        + h11[:, None] * tangent_scale * second
    )
    segments = np.diff(connector, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-8):
        return None
    if np.any((segments @ chord) < -0.02 * lengths):
        return None
    if float(np.sum(lengths)) > 1.28 * distance:
        return None
    directions = segments / lengths[:, None]
    turns = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0))
    if turns.size and float(np.max(turns)) > np.deg2rad(22.0):
        return None
    return connector


def _network_node_degrees(roads: list[_RegionalRoadSeed]) -> dict[tuple[float, float], int]:
    neighbors: dict[tuple[float, float], set[tuple[float, float]]] = {}
    for road in roads:
        points = _deduplicate_points(road.points)
        for first, second in zip(points, points[1:]):
            first_key, second_key = _node_key(first), _node_key(second)
            if first_key == second_key:
                continue
            neighbors.setdefault(first_key, set()).add(second_key)
            neighbors.setdefault(second_key, set()).add(first_key)
    return {node: len(values) for node, values in neighbors.items()}


def _road_component_labels(roads: list[_RegionalRoadSeed]) -> list[int]:
    """Label roads joined by existing topology so a new link cannot create a loop."""
    parent = list(range(len(roads)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    first_at_node: dict[tuple[float, float], int] = {}
    for road_id, road in enumerate(roads):
        for point in _deduplicate_points(road.points):
            node = _node_key(point)
            previous = first_at_node.setdefault(node, road_id)
            first_root, second_root = find(previous), find(road_id)
            if first_root != second_root:
                parent[second_root] = first_root
    return [find(road_id) for road_id in range(len(roads))]
