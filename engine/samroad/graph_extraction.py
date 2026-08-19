import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import tcod
from collections import Counter
from sklearn.neighbors import KDTree
from skimage.draw import line
from skimage.morphology import skeletonize
import networkx as nx
from graph_utils import nms_points


IMAGE_SIZE = 2048
SAMPLE_MARGIN = 64

def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb



# returns (x, y)
def get_points_and_scores_from_mask(mask, threshold):
    rcs = np.column_stack(np.where(mask > threshold))
    xys = rcs[:, ::-1]
    scores = mask[mask > threshold]
    return xys, scores


def draw_points_on_image(image, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    
    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, (0, 255, 0), -1)

    return image


def draw_points_on_grayscale_image(image, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    
    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, 255, -1)

    return image


# takes xy
def is_connected_bresenham(cost, start, end):
    c0, r0 = start
    c1, r1 = end
    rr, cc = line(r0, c0, r1, c1)
    kp_block_radius = 4
    cv2.circle(cost, start, kp_block_radius, 0, -1)
    cv2.circle(cost, end, kp_block_radius, 0, -1)
    
    # mean_cost = np.mean(cost[rr, cc])
    max_cost = np.max(cost[rr, cc])

    cv2.circle(cost, start, kp_block_radius, 255, -1)
    cv2.circle(cost, end, kp_block_radius, 255, -1)

    return max_cost < 255


def is_connected_astar(pathfinder, cost, start, end, max_path_len):
    # we can still modify the cost matrix after creating the pathfinder with it
    # seems pathfinder uses reference
    c0, r0 = start
    c1, r1 = end
    kp_block_radius = 6
    cv2.circle(cost, start, kp_block_radius, 1, -1)
    cv2.circle(cost, end, kp_block_radius, 1, -1)
    
    path = pathfinder.get_path(r0, c0, r1, c1)
    connected = (len(path) != 0) and (len(path) < max_path_len)

    cv2.circle(cost, start, kp_block_radius, 0, -1)
    cv2.circle(cost, end, kp_block_radius, 0, -1)

    return connected


def create_cost_field(sample_pts, road_mask):
    # road mask shall be uint8 normalized to 0-255
    cost_field = np.zeros(road_mask.shape, dtype=np.uint8)
    kp_block_radius = 4
    for point in sample_pts:
        cv2.circle(cost_field, point, kp_block_radius, 255, -1)
    cost_field = np.maximum(cost_field, 255 - road_mask)
    return cost_field

def _probability01(mask):
    probability = np.asarray(mask, dtype=np.float32)
    if probability.size and float(np.nanmax(probability)) > 1.0:
        probability = probability / 255.0
    return np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _config_value(config, name, default):
    getter = getattr(config, "get", None)
    return getter(name, default) if getter is not None else getattr(config, name, default)


def resolve_road_thresholds(config, profile_name=None):
    """Resolve configurable high/low thresholds, including sensor profiles."""
    profile_name = str(
        profile_name
        if profile_name is not None
        else _config_value(config, "ROAD_THRESHOLD_PROFILE", "default")
    )
    profiles = _config_value(config, "ROAD_THRESHOLD_PROFILES", {}) or {}
    profile = profiles.get(profile_name, {}) if hasattr(profiles, "get") else {}

    def profile_value(name, fallback):
        if hasattr(profile, "get"):
            return profile.get(name, profile.get(name.lower(), fallback))
        return fallback

    legacy_high = float(_config_value(config, "ROAD_THRESHOLD", 0.364))
    high = float(profile_value(
        "ROAD_HIGH_THRESHOLD",
        _config_value(config, "ROAD_HIGH_THRESHOLD", legacy_high),
    ))
    low = float(profile_value(
        "ROAD_LOW_THRESHOLD",
        _config_value(config, "ROAD_LOW_THRESHOLD", min(0.20, high * 0.65)),
    ))
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            f"Road thresholds must satisfy 0 <= low < high <= 1; got low={low}, high={high}"
        )
    return high, low, profile_name


def resolve_scene_diagnostic_thresholds(config):
    """Resolve fixed thresholds used only to diagnose probability calibration."""
    reference_profile = str(
        _config_value(config, "SCENE_DIAGNOSTIC_REFERENCE_PROFILE", "default")
    )
    return resolve_road_thresholds(config, profile_name=reference_profile)


def create_cost_field_astar(
    sample_pts,
    road_mask,
    block_threshold=200,
    *,
    low_threshold=None,
    surface_probability=None,
    surface_threshold=0.60,
):
    """Build a weighted A* field where weak evidence is costly but traversable."""
    probability = _probability01(road_mask)
    low_threshold = (
        max(0.0, min(1.0, (255.0 - float(block_threshold)) / 255.0))
        if low_threshold is None
        else float(low_threshold)
    )
    allowed = probability >= low_threshold
    effective = probability.copy()
    if surface_probability is not None:
        surface = _probability01(surface_probability)
        if surface.shape != probability.shape:
            raise ValueError(
                f"Road/surface probability shape mismatch: {probability.shape} != {surface.shape}"
            )
        surface_allowed = surface >= float(surface_threshold)
        allowed |= surface_allowed
        # Surface evidence permits travel but never makes a weak centerline as
        # cheap as a genuinely high SAMRoad response.
        effective = np.maximum(effective, np.where(surface_allowed, 0.45 * surface, 0.0))
    cost_field = np.clip(np.rint(1.0 + (1.0 - effective) * 199.0), 1, 255).astype(np.uint8)
    cost_field[~allowed] = 0
    kp_block_radius = 6
    for point in sample_pts:
        cv2.circle(cost_field, point, kp_block_radius, 255, -1)
    return cost_field


def _endpoint_vectors(nodes_rc, edges):
    # SAMRoad can emit reciprocal directed edges. Endpoint degree must be based
    # on unique neighboring nodes, otherwise a true degree-one endpoint appears
    # to have degree two and is silently excluded from recovery proposals.
    adjacency = [set() for _ in range(len(nodes_rc))]
    for src_idx, dst_idx in np.asarray(edges, dtype=np.int32).reshape(-1, 2).tolist():
        adjacency[src_idx].add(dst_idx)
        adjacency[dst_idx].add(src_idx)
    vectors = {}
    for node_idx, neighbors in enumerate(adjacency):
        if len(neighbors) != 1:
            continue
        neighbor_idx = next(iter(neighbors))
        vector = nodes_rc[node_idx] - nodes_rc[neighbor_idx]
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            vectors[node_idx] = vector / norm
    return vectors


def _astar_probability_path(
    start_rc,
    end_rc,
    road_probability,
    low_threshold,
    *,
    surface_probability=None,
    surface_threshold=0.60,
    margin=12,
):
    height, width = road_probability.shape
    y0 = max(0, int(math.floor(min(start_rc[0], end_rc[0]) - margin)))
    y1 = min(height, int(math.ceil(max(start_rc[0], end_rc[0]) + margin + 1)))
    x0 = max(0, int(math.floor(min(start_rc[1], end_rc[1]) - margin)))
    x1 = min(width, int(math.ceil(max(start_rc[1], end_rc[1]) + margin + 1)))
    local_surface = None if surface_probability is None else surface_probability[y0:y1, x0:x1]
    cost = create_cost_field_astar(
        [],
        road_probability[y0:y1, x0:x1],
        low_threshold=low_threshold,
        surface_probability=local_surface,
        surface_threshold=surface_threshold,
    )
    start = np.rint(start_rc - np.asarray([y0, x0])).astype(np.int32)
    end = np.rint(end_rc - np.asarray([y0, x0])).astype(np.int32)
    cost[start[0], start[1]] = max(1, int(cost[start[0], start[1]]))
    cost[end[0], end[1]] = max(1, int(cost[end[0], end[1]]))
    path = np.asarray(
        tcod.path.AStar(cost).get_path(int(start[0]), int(start[1]), int(end[0]), int(end[1])),
        dtype=np.int32,
    ).reshape(-1, 2)
    if path.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    path = path + np.asarray([y0, x0], dtype=np.int32)
    if not np.array_equal(path[0], np.rint(start_rc).astype(np.int32)):
        path = np.vstack([np.rint(start_rc).astype(np.int32), path])
    if not np.array_equal(path[-1], np.rint(end_rc).astype(np.int32)):
        path = np.vstack([path, np.rint(end_rc).astype(np.int32)])
    return path


def _path_background_mean(probability, path, offset=3.0):
    if len(path) < 2:
        return 0.0
    samples = []
    for position in range(len(path)):
        before = path[max(0, position - 1)].astype(np.float32)
        after = path[min(len(path) - 1, position + 1)].astype(np.float32)
        tangent = after - before
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-6:
            continue
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32) / norm
        for sign in (-1.0, 1.0):
            point = np.rint(path[position] + sign * offset * normal).astype(np.int32)
            point[0] = np.clip(point[0], 0, probability.shape[0] - 1)
            point[1] = np.clip(point[1], 0, probability.shape[1] - 1)
            samples.append(float(probability[point[0], point[1]]))
    return float(np.mean(samples)) if samples else 0.0


def _recovery_path_evidence(
    path,
    road_probability,
    low_threshold,
    surface_probability,
    parameters,
):
    rows = np.clip(path[:, 0], 0, road_probability.shape[0] - 1)
    cols = np.clip(path[:, 1], 0, road_probability.shape[1] - 1)
    values = road_probability[rows, cols]
    center_mean = float(np.mean(values))
    center_q25 = float(np.quantile(values, 0.25))
    weak_fraction = float(np.mean(values >= low_threshold))
    background_mean = _path_background_mean(
        road_probability, path, parameters["background_offset"]
    )
    contrast = center_mean - background_mean
    surface_mean = surface_fraction = 0.0
    if surface_probability is not None:
        surface_values = surface_probability[rows, cols]
        surface_mean = float(np.mean(surface_values))
        surface_fraction = float(np.mean(surface_values >= parameters["surface_threshold"]))
    road_supported = (
        center_mean >= parameters["min_mean"]
        and center_q25 >= parameters["min_q25"]
        and weak_fraction >= parameters["min_weak_fraction"]
        and contrast >= parameters["min_contrast"]
    )
    surface_supported = (
        surface_probability is not None
        and center_mean >= parameters["surface_min_center"]
        and surface_mean >= parameters["surface_min_mean"]
        and surface_fraction >= parameters["surface_min_fraction"]
    )
    return {
        "center_conf": center_mean,
        "center_q25": center_q25,
        "weak_fraction": weak_fraction,
        "background_conf": background_mean,
        "probability_contrast": contrast,
        "surface_conf": surface_mean,
        "surface_fraction": surface_fraction,
        "road_supported": road_supported,
        "surface_supported": surface_supported,
    }


def _road_evidence_reject_reason(evidence, parameters):
    """Return the first failed road-evidence rule for candidate auditing."""
    if evidence["center_conf"] < parameters["min_mean"]:
        return "mean_probability_low"
    if evidence["center_q25"] < parameters["min_q25"]:
        return "q25_probability_low"
    if evidence["weak_fraction"] < parameters["min_weak_fraction"]:
        return "weak_fraction_low"
    if evidence["probability_contrast"] < parameters["min_contrast"]:
        return "background_contrast_low"
    return "insufficient_independent_support"


def _graph_raster_mask(nodes_rc, edges, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    for src_idx, dst_idx in np.asarray(edges, dtype=np.int32).reshape(-1, 2).tolist():
        src = nodes[int(src_idx)]
        dst = nodes[int(dst_idx)]
        cv2.line(
            mask,
            (int(round(src[1])), int(round(src[0]))),
            (int(round(dst[1])), int(round(dst[0]))),
            1,
            1,
            cv2.LINE_8,
        )
    return mask


def _trace_skeleton_chains(skeleton):
    """Trace an 8-connected skeleton into junction/endpoint-to-junction chains."""
    pixels = {tuple(value) for value in np.column_stack(np.where(skeleton)).tolist()}
    if not pixels:
        return []
    offsets = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )

    def neighbors(point):
        row, col = point
        result = []
        for dr, dc in offsets:
            candidate = (row + dr, col + dc)
            if candidate not in pixels:
                continue
            # Avoid triangular corner shortcuts while retaining true diagonal
            # skeletons that have no orthogonal connection.
            if dr and dc and (
                (row + dr, col) in pixels or (row, col + dc) in pixels
            ):
                continue
            result.append(candidate)
        return result

    adjacency = {point: neighbors(point) for point in pixels}
    anchors = {point for point, items in adjacency.items() if len(items) != 2}
    visited = set()
    chains = []

    def edge_key(first, second):
        return tuple(sorted((first, second)))

    def trace(start, first):
        chain = [start, first]
        visited.add(edge_key(start, first))
        previous, current = start, first
        while current not in anchors:
            following = [item for item in adjacency[current] if item != previous]
            if not following:
                break
            nxt = following[0]
            key = edge_key(current, nxt)
            if key in visited:
                break
            visited.add(key)
            chain.append(nxt)
            previous, current = current, nxt
        return chain

    for anchor in sorted(anchors):
        for neighbor in adjacency[anchor]:
            if edge_key(anchor, neighbor) not in visited:
                chains.append(trace(anchor, neighbor))
    for point in sorted(pixels):
        for neighbor in adjacency[point]:
            if edge_key(point, neighbor) not in visited:
                chains.append(trace(point, neighbor))
    return [np.asarray(chain, dtype=np.int32) for chain in chains if len(chain) >= 2]


def diagnose_scene_confidence(
    road_probability,
    nodes_rc,
    edges,
    config,
    *,
    distance_scale=1.0,
):
    """Describe low-response scenes without changing the selected profile."""
    road = _probability01(road_probability)
    active_high, active_low, active_profile = resolve_road_thresholds(config)
    reference_high, reference_low, reference_profile = resolve_scene_diagnostic_thresholds(config)
    close_size = max(1, int(round(_config_value(config, "WEAK_BOOTSTRAP_CLOSE_KERNEL", 3))))
    low_mask = (road >= reference_low).astype(np.uint8)
    high_mask = (road >= reference_high).astype(np.uint8)
    if close_size > 1:
        kernel = np.ones((close_size, close_size), dtype=np.uint8)
        low_mask = cv2.morphologyEx(low_mask, cv2.MORPH_CLOSE, kernel)
        high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_CLOSE, kernel)
    weak_skeleton = skeletonize(low_mask.astype(bool))
    reference_strong_skeleton = skeletonize(high_mask.astype(bool))
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    graph_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    active_graph_length = float(sum(
        np.linalg.norm(nodes[int(dst_idx)] - nodes[int(src_idx)])
        for src_idx, dst_idx in graph_edges.tolist()
    ))
    weak_length = float(np.count_nonzero(weak_skeleton))
    reference_strong_length = float(np.count_nonzero(reference_strong_skeleton))
    high_ratio = float(np.mean(road >= reference_high)) if road.size else 0.0
    low_ratio = float(np.mean(road >= reference_low)) if road.size else 0.0
    relative_high = high_ratio / max(low_ratio, 1e-9)
    relative_strong = reference_strong_length / max(weak_length, 1e-9)
    minimum_structure = float(
        _config_value(config, "WEAK_BOOTSTRAP_MIN_LENGTH_PX", 48.0)
    ) * max(float(distance_scale), 1e-6)
    has_low_structure = weak_length >= minimum_structure and low_ratio > 0.0
    percentiles = np.quantile(road, [0.50, 0.90, 0.95, 0.99]) if road.size else np.zeros(4)
    if has_low_structure and relative_high < 0.12 and relative_strong < 0.20:
        state = "low_confidence"
    elif has_low_structure and relative_high < 0.30 and relative_strong < 0.35:
        state = "low_confidence"
    elif reference_strong_length <= 0.0 and high_ratio <= 1e-6 and low_ratio <= 1e-6:
        state = "very_low_confidence"
    else:
        state = "normal"
    return {
        "scene_confidence_state": state,
        "recommended_profile": "weak_sensor" if state != "normal" else "default",
        "threshold_profile": active_profile,
        "active_profile": active_profile,
        "diagnostic_reference_profile": reference_profile,
        "active_high_threshold": active_high,
        "active_low_threshold": active_low,
        "reference_high_threshold": reference_high,
        "reference_low_threshold": reference_low,
        "probability_p50": float(percentiles[0]),
        "probability_p90": float(percentiles[1]),
        "probability_p95": float(percentiles[2]),
        "probability_p99": float(percentiles[3]),
        "high_pixel_ratio": high_ratio,
        "low_pixel_ratio": low_ratio,
        "strong_graph_edge_count": int(len(graph_edges)),
        "strong_graph_total_length": active_graph_length,
        "reference_strong_skeleton_total_length": reference_strong_length,
        "weak_skeleton_total_length": weak_length,
    }


def bootstrap_weak_road_network(
    nodes_rc,
    edges,
    road_probability,
    config,
    *,
    surface_probability=None,
    edge_scores=None,
    distance_scale=1.0,
    candidate_audit=None,
):
    """Conservatively add continuous weak chains that do not need strong seeds."""
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    original_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    scores = np.asarray(
        edge_scores if edge_scores is not None else np.full(len(original_edges), np.nan),
        dtype=np.float32,
    )
    if len(scores) != len(original_edges):
        raise ValueError("edge_scores must align with edges")
    road = _probability01(road_probability)
    surface = None if surface_probability is None else _probability01(surface_probability)
    if surface is not None and surface.shape != road.shape:
        raise ValueError(f"Road/surface probability shape mismatch: {road.shape} != {surface.shape}")
    high_threshold, low_threshold, profile_name = resolve_road_thresholds(config)
    scale = max(float(distance_scale), 1e-6)
    parameters = {
        "min_length": float(_config_value(config, "WEAK_BOOTSTRAP_MIN_LENGTH_PX", 48.0)) * scale,
        "min_mean": float(_config_value(config, "WEAK_BOOTSTRAP_MIN_MEAN_PROBABILITY", 0.16)),
        "min_q25": float(_config_value(config, "WEAK_BOOTSTRAP_MIN_Q25_PROBABILITY", 0.12)),
        "min_contrast": float(_config_value(config, "WEAK_BOOTSTRAP_MIN_BACKGROUND_CONTRAST", 0.08)),
        "max_tortuosity": float(_config_value(config, "WEAK_BOOTSTRAP_MAX_TORTUOSITY", 1.35)),
        "min_weak_fraction": float(_config_value(config, "WEAK_BOOTSTRAP_MIN_WEAK_FRACTION", 0.80)),
        "background_offset": float(_config_value(config, "WEAK_RECOVERY_BACKGROUND_OFFSET_PX", 4.0)) * scale,
        "surface_threshold": float(_config_value(config, "WEAK_RECOVERY_SURFACE_THRESHOLD", 0.60)),
        "surface_min_center": float(_config_value(config, "WEAK_RECOVERY_SURFACE_MIN_CENTER_PROBABILITY", 0.10)),
        "surface_min_mean": float(_config_value(config, "WEAK_RECOVERY_SURFACE_MIN_MEAN", 0.70)),
        "surface_min_fraction": float(_config_value(config, "WEAK_RECOVERY_SURFACE_MIN_FRACTION", 0.80)),
        "suppression_radius": max(0, int(round(float(_config_value(config, "WEAK_BOOTSTRAP_STRONG_SUPPRESSION_PX", 3.0)) * scale))),
        "connection_radius": max(1.0, float(_config_value(config, "WEAK_BOOTSTRAP_STRONG_CONNECTION_PX", 10.0)) * scale),
        "sample_step": max(1.0, float(_config_value(config, "WEAK_BOOTSTRAP_SAMPLE_STEP_PX", 12.0)) * scale),
        "auto_score": float(_config_value(config, "WEAK_BOOTSTRAP_AUTO_SCORE", 0.74)),
        "independent_length_factor": float(_config_value(config, "WEAK_BOOTSTRAP_INDEPENDENT_LENGTH_FACTOR", 1.5)),
    }
    metadata = []
    for edge_id, (src_idx, dst_idx) in enumerate(original_edges.tolist()):
        rr, cc = line(
            int(round(nodes[src_idx, 0])), int(round(nodes[src_idx, 1])),
            int(round(nodes[dst_idx, 0])), int(round(nodes[dst_idx, 1])),
        )
        rr = np.clip(rr, 0, road.shape[0] - 1)
        cc = np.clip(cc, 0, road.shape[1] - 1)
        center_conf = float(np.mean(road[rr, cc])) if len(rr) else 0.0
        topology_probability = float(scores[edge_id]) if np.isfinite(scores[edge_id]) else center_conf
        metadata.append({
            "line_source": "samroad", "topology_probability": topology_probability,
            "recovery_score": 0.0, "center_conf": center_conf,
            "background_conf": 0.0, "probability_contrast": center_conf,
            "surface_conf": 0.0, "recovery_reason": "strong_threshold",
            "qa_state": "auto", "recovery_id": "",
        })
    summary = {
        "threshold_profile": profile_name,
        "bootstrap_candidate_count": 0,
        "bootstrap_accepted_candidate_count": 0,
        "bootstrap_recovered_edge_count": 0,
        "bootstrap_auto_count": 0,
        "bootstrap_review_count": 0,
        "bootstrap_rejected_count": 0,
        "bootstrap_reject_reason_counts": {},
    }
    if not bool(_config_value(config, "WEAK_BOOTSTRAP_ENABLED", True)):
        return nodes, original_edges, metadata, summary

    close_size = max(1, int(round(_config_value(config, "WEAK_BOOTSTRAP_CLOSE_KERNEL", 3))))
    low_mask = (road >= low_threshold).astype(np.uint8)
    if close_size > 1:
        kernel = np.ones((close_size, close_size), dtype=np.uint8)
        low_mask = cv2.morphologyEx(low_mask, cv2.MORPH_CLOSE, kernel)
    weak_skeleton = skeletonize(low_mask.astype(bool))
    strong_mask = _graph_raster_mask(nodes, original_edges, road.shape)
    if parameters["suppression_radius"] > 0 and np.any(strong_mask):
        radius = parameters["suppression_radius"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        suppressed = cv2.dilate(strong_mask, kernel) > 0
        weak_skeleton &= ~suppressed
    if np.any(strong_mask):
        strong_distance = cv2.distanceTransform((strong_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        strong_distance = np.full(road.shape, np.inf, dtype=np.float32)
    chains = _trace_skeleton_chains(weak_skeleton)
    combined_nodes = nodes.tolist()
    combined_edges = original_edges.tolist()
    existing_edges = {tuple(sorted(edge)) for edge in combined_edges}
    coordinate_nodes = {
        tuple(np.rint(point).astype(np.int32).tolist()): index
        for index, point in enumerate(nodes)
    }
    node_tree = KDTree(nodes) if len(nodes) else None
    recovery_id = 0
    reject_counts = Counter()

    def reject_candidate(row, reason):
        row["accepted"] = False
        row["qa_state"] = "rejected"
        row["reject_reason"] = reason
        summary["bootstrap_rejected_count"] += 1
        reject_counts[reason] += 1
        if candidate_audit is not None:
            candidate_audit.append(row)

    for path in chains:
        summary["bootstrap_candidate_count"] += 1
        path_float = path.astype(np.float32)
        path_length = float(np.linalg.norm(np.diff(path_float, axis=0), axis=1).sum())
        direct_distance = float(np.linalg.norm(path_float[-1] - path_float[0]))
        tortuosity = path_length / max(direct_distance, 1e-6)
        audit_row = {
            "path_length": path_length,
            "direct_distance": direct_distance,
            "tortuosity": tortuosity,
            "mean_probability": None,
            "q25_probability": None,
            "weak_fraction": None,
            "background_probability": None,
            "background_contrast": None,
            "connection_count": 0,
            "accepted": False,
            "qa_state": "rejected",
            "reject_reason": "",
        }
        if path_length < parameters["min_length"]:
            reject_candidate(audit_row, "too_short")
            continue
        evidence = _recovery_path_evidence(
            path, road, low_threshold, surface, parameters
        )
        endpoint_distances = [
            float(strong_distance[int(point[0]), int(point[1])]) for point in (path[0], path[-1])
        ]
        connection_count = sum(
            distance <= parameters["connection_radius"] for distance in endpoint_distances
        )
        audit_row.update({
            "mean_probability": evidence["center_conf"],
            "q25_probability": evidence["center_q25"],
            "weak_fraction": evidence["weak_fraction"],
            "background_probability": evidence["background_conf"],
            "background_contrast": evidence["probability_contrast"],
            "connection_count": connection_count,
        })
        recovery_gap_limit = float(_config_value(config, "WEAK_RECOVERY_MAX_GAP_PX", 64.0)) * scale
        delegated_gap = connection_count == 2 and path_length <= recovery_gap_limit
        geometry_supported = tortuosity <= parameters["max_tortuosity"]
        independent_supported = (
            connection_count > 0
            or evidence["surface_supported"]
            or (
                path_length >= parameters["min_length"] * parameters["independent_length_factor"]
                and evidence["center_conf"] >= parameters["min_mean"] + 0.01
                and evidence["probability_contrast"] >= parameters["min_contrast"] + 0.02
            )
        )
        evidence_supported = evidence["road_supported"] or evidence["surface_supported"]
        if delegated_gap:
            reject_candidate(audit_row, "delegated_to_weak_recovery")
            continue
        if not geometry_supported:
            reject_candidate(audit_row, "high_tortuosity")
            continue
        if not evidence_supported:
            reject_candidate(
                audit_row,
                _road_evidence_reject_reason(evidence, parameters),
            )
            continue
        if not independent_supported:
            reject_candidate(audit_row, "insufficient_independent_support")
            continue
        directness = min(1.0, 1.0 / max(tortuosity, 1.0))
        proximity = (0.5, 0.8, 1.0)[connection_count]
        recovery_score = (
            0.24 * min(1.0, evidence["center_conf"] / max(high_threshold, 1e-6))
            + 0.18 * min(1.0, evidence["center_q25"] / max(low_threshold, 1e-6))
            + 0.22 * min(1.0, evidence["probability_contrast"] / max(parameters["min_contrast"] * 2.0, 1e-6))
            + 0.14 * min(1.0, path_length / max(parameters["min_length"] * 2.0, 1e-6))
            + 0.10 * directness
            + 0.06 * proximity
            + 0.06 * evidence["surface_conf"]
        )
        qa_state = "auto" if recovery_score >= parameters["auto_score"] else "review"
        distances = np.concatenate([
            np.asarray([0.0], dtype=np.float32),
            np.cumsum(np.linalg.norm(np.diff(path_float, axis=0), axis=1)),
        ])
        sample_positions = np.arange(0.0, distances[-1], parameters["sample_step"])
        sampled = [path_float[min(int(np.searchsorted(distances, value)), len(path_float) - 1)] for value in sample_positions]
        sampled.append(path_float[-1])
        chain = []
        for point_index, point in enumerate(sampled):
            node_idx = None
            if node_tree is not None and point_index in {0, len(sampled) - 1}:
                nearest_distance, nearest_index = node_tree.query(point[np.newaxis, :], k=1)
                if float(nearest_distance[0, 0]) <= parameters["connection_radius"]:
                    node_idx = int(nearest_index[0, 0])
            key = tuple(np.rint(point).astype(np.int32).tolist())
            if node_idx is None:
                node_idx = coordinate_nodes.get(key)
            if node_idx is None:
                combined_nodes.append(point.tolist())
                node_idx = len(combined_nodes) - 1
                coordinate_nodes[key] = node_idx
            if not chain or chain[-1] != node_idx:
                chain.append(node_idx)
        added_count = 0
        for src_idx, dst_idx in zip(chain[:-1], chain[1:]):
            key = tuple(sorted((int(src_idx), int(dst_idx))))
            if src_idx == dst_idx or key in existing_edges:
                continue
            existing_edges.add(key)
            combined_edges.append((int(src_idx), int(dst_idx)))
            metadata.append({
                "line_source": "weak_bootstrap",
                "topology_probability": float(recovery_score),
                "recovery_score": float(recovery_score),
                "center_conf": float(evidence["center_conf"]),
                "background_conf": float(evidence["background_conf"]),
                "probability_contrast": float(evidence["probability_contrast"]),
                "surface_conf": float(evidence["surface_conf"]),
                "recovery_reason": "weak_network_bootstrap",
                "qa_state": qa_state,
                "recovery_id": f"bootstrap:{recovery_id}",
            })
            added_count += 1
        if added_count:
            recovery_id += 1
            summary["bootstrap_accepted_candidate_count"] += 1
            summary["bootstrap_recovered_edge_count"] += added_count
            summary[f"bootstrap_{qa_state}_count"] += 1
            audit_row["accepted"] = True
            audit_row["qa_state"] = qa_state
            audit_row["reject_reason"] = ""
            if candidate_audit is not None:
                candidate_audit.append(audit_row)
        else:
            reject_candidate(audit_row, "duplicate_or_suppressed")
    summary["bootstrap_reject_reason_counts"] = dict(sorted(reject_counts.items()))
    return (
        np.asarray(combined_nodes, dtype=np.float32).reshape(-1, 2),
        np.asarray(combined_edges, dtype=np.int32).reshape(-1, 2),
        metadata,
        summary,
    )


def recover_weak_road_edges(
    nodes_rc,
    edges,
    road_probability,
    config,
    *,
    surface_probability=None,
    edge_scores=None,
    distance_scale=1.0,
    candidate_audit=None,
):
    """Conservatively bridge or extend dangling endpoints through weak evidence."""
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    original_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    road = _probability01(road_probability)
    surface = None if surface_probability is None else _probability01(surface_probability)
    if surface is not None and surface.shape != road.shape:
        raise ValueError(f"Road/surface probability shape mismatch: {road.shape} != {surface.shape}")
    high_threshold, low_threshold, profile_name = resolve_road_thresholds(config)
    enabled = bool(_config_value(config, "WEAK_RECOVERY_ENABLED", True))
    scores = np.asarray(
        edge_scores if edge_scores is not None else np.full(len(original_edges), np.nan),
        dtype=np.float32,
    )
    if len(scores) != len(original_edges):
        raise ValueError("edge_scores must align with edges")
    metadata = []
    for edge_id, (src_idx, dst_idx) in enumerate(original_edges.tolist()):
        rr, cc = line(
            int(round(nodes[src_idx, 0])), int(round(nodes[src_idx, 1])),
            int(round(nodes[dst_idx, 0])), int(round(nodes[dst_idx, 1])),
        )
        rr = np.clip(rr, 0, road.shape[0] - 1)
        cc = np.clip(cc, 0, road.shape[1] - 1)
        center_conf = float(np.mean(road[rr, cc])) if len(rr) else 0.0
        topology_probability = float(scores[edge_id]) if np.isfinite(scores[edge_id]) else center_conf
        metadata.append({
            "line_source": "samroad", "topology_probability": topology_probability,
            "recovery_score": 0.0, "center_conf": center_conf, "surface_conf": 0.0,
            "recovery_reason": "strong_threshold", "qa_state": "auto", "recovery_id": "",
        })
    summary = {
        "threshold_profile": profile_name,
        "road_high_threshold": high_threshold,
        "road_low_threshold": low_threshold,
        "strong_edge_count": int(len(original_edges)),
        "weak_candidate_count": 0,
        "weak_recovered_candidate_count": 0,
        "weak_recovered_edge_count": 0,
        "surface_supported_recovery_count": 0,
        "rejected_weak_candidate_count": 0,
        "weak_recovery_reject_reason_counts": {},
        "recovery_reason_counts": {},
    }
    if not enabled or len(original_edges) == 0 or len(nodes) == 0:
        return nodes, original_edges, metadata, summary

    scale = max(float(distance_scale), 1e-6)
    parameters = {
        "max_gap": float(_config_value(config, "WEAK_RECOVERY_MAX_GAP_PX", 64.0)) * scale,
        "max_extension": float(_config_value(config, "WEAK_RECOVERY_MAX_EXTENSION_PX", 48.0)) * scale,
        "min_extension": float(_config_value(config, "WEAK_RECOVERY_MIN_EXTENSION_PX", 10.0)) * scale,
        "min_alignment": float(_config_value(config, "WEAK_RECOVERY_MIN_DIRECTION_COSINE", 0.65)),
        "max_path_ratio": float(_config_value(config, "WEAK_RECOVERY_MAX_PATH_RATIO", 1.35)),
        "min_mean": float(_config_value(config, "WEAK_RECOVERY_MIN_MEAN_PROBABILITY", max(low_threshold, 0.20))),
        "min_q25": float(_config_value(config, "WEAK_RECOVERY_MIN_Q25_PROBABILITY", max(0.15, low_threshold * 0.85))),
        "min_weak_fraction": float(_config_value(config, "WEAK_RECOVERY_MIN_WEAK_FRACTION", 0.80)),
        "min_contrast": float(_config_value(config, "WEAK_RECOVERY_MIN_BACKGROUND_CONTRAST", 0.08)),
        "background_offset": float(_config_value(config, "WEAK_RECOVERY_BACKGROUND_OFFSET_PX", 4.0)) * scale,
        "surface_threshold": float(_config_value(config, "WEAK_RECOVERY_SURFACE_THRESHOLD", 0.60)),
        "surface_min_center": float(_config_value(config, "WEAK_RECOVERY_SURFACE_MIN_CENTER_PROBABILITY", 0.10)),
        "surface_min_mean": float(_config_value(config, "WEAK_RECOVERY_SURFACE_MIN_MEAN", 0.70)),
        "surface_min_fraction": float(_config_value(config, "WEAK_RECOVERY_SURFACE_MIN_FRACTION", 0.80)),
        "path_margin": max(1.0, float(_config_value(config, "WEAK_RECOVERY_PATH_MARGIN_PX", 16.0)) * scale),
        "sample_step": max(1.0, float(_config_value(config, "WEAK_RECOVERY_SAMPLE_STEP_PX", 12.0)) * scale),
        "auto_score": float(_config_value(config, "WEAK_RECOVERY_AUTO_SCORE", 0.62)),
    }
    endpoint_vectors = _endpoint_vectors(nodes, original_edges)
    endpoint_ids = sorted(endpoint_vectors)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(original_edges.tolist())
    component = {}
    for component_id, members in enumerate(nx.connected_components(graph)):
        for node_idx in members:
            component[node_idx] = component_id

    proposals = []
    reject_counts = Counter()

    def reject_candidate(row, reason):
        row["accepted"] = False
        row["reject_reason"] = reason
        summary["rejected_weak_candidate_count"] += 1
        reject_counts[reason] += 1
        if candidate_audit is not None:
            candidate_audit.append(row)

    def evaluate(start_idx, end_point, kind, end_idx=None):
        start_point = nodes[start_idx]
        delta = np.asarray(end_point, dtype=np.float32) - start_point
        distance = float(np.linalg.norm(delta))
        limit = parameters["max_gap"] if kind == "bridge" else parameters["max_extension"]
        summary["weak_candidate_count"] += 1
        audit_row = {
            "candidate_type": kind,
            "start": start_point.astype(float).tolist(),
            "end": np.asarray(end_point, dtype=np.float32).astype(float).tolist(),
            "distance": distance,
            "direction_cosine": None,
            "path_length": None,
            "path_ratio": None,
            "mean_probability": None,
            "q25_probability": None,
            "weak_fraction": None,
            "background_probability": None,
            "background_contrast": None,
            "surface_probability": None,
            "accepted": False,
            "reject_reason": "",
        }
        if distance <= 1e-6 or distance > limit:
            reject_candidate(audit_row, "distance_too_large")
            return
        direction = delta / distance
        first_alignment = float(np.dot(endpoint_vectors[start_idx], direction))
        audit_row["direction_cosine"] = first_alignment
        if first_alignment < parameters["min_alignment"]:
            reject_candidate(audit_row, "direction_mismatch")
            return
        second_alignment = 1.0
        if kind == "bridge":
            if end_idx is None or component[start_idx] == component[end_idx]:
                reject_candidate(audit_row, "same_component")
                return
            second_alignment = float(np.dot(endpoint_vectors[end_idx], -direction))
            audit_row["direction_cosine"] = min(first_alignment, second_alignment)
            if second_alignment < parameters["min_alignment"]:
                reject_candidate(audit_row, "direction_mismatch")
                return
        elif distance < parameters["min_extension"]:
            reject_candidate(audit_row, "distance_too_large")
            return
        path = _astar_probability_path(
            start_point, end_point, road, low_threshold,
            surface_probability=surface,
            surface_threshold=parameters["surface_threshold"],
            margin=parameters["path_margin"],
        )
        if len(path) < 2:
            reject_candidate(audit_row, "no_astar_path")
            return
        path_length = float(np.linalg.norm(np.diff(path.astype(np.float32), axis=0), axis=1).sum())
        path_ratio = path_length / max(distance, 1e-6)
        audit_row["path_length"] = path_length
        audit_row["path_ratio"] = path_ratio
        evidence = _recovery_path_evidence(path, road, low_threshold, surface, parameters)
        audit_row.update({
            "mean_probability": evidence["center_conf"],
            "q25_probability": evidence["center_q25"],
            "weak_fraction": evidence["weak_fraction"],
            "background_probability": evidence["background_conf"],
            "background_contrast": evidence["probability_contrast"],
            "surface_probability": evidence["surface_conf"],
        })
        if path_ratio > parameters["max_path_ratio"]:
            reject_candidate(audit_row, "path_ratio_too_large")
            return
        if not (evidence["road_supported"] or evidence["surface_supported"]):
            reject_candidate(
                audit_row,
                _road_evidence_reject_reason(evidence, parameters),
            )
            return
        alignment = min(first_alignment, second_alignment)
        directness = min(1.0, 1.0 / max(path_ratio, 1.0))
        recovery_score = (
            0.32 * min(1.0, evidence["center_conf"] / max(high_threshold, 1e-6))
            + 0.18 * min(1.0, evidence["center_q25"] / max(low_threshold, 1e-6))
            + 0.20 * alignment
            + 0.15 * directness
            + 0.15 * evidence["surface_conf"]
        )
        reason = (
            "weak_probability_surface_supported"
            if evidence["surface_supported"]
            else "weak_probability_endpoint_bridge" if kind == "bridge"
            else "weak_probability_endpoint_extension"
        )
        proposals.append({
            "score": recovery_score, "start_idx": start_idx, "end_idx": end_idx,
            "end_point": np.asarray(end_point, dtype=np.float32), "kind": kind,
            "path": path, "path_ratio": path_ratio, "alignment": alignment,
            "reason": reason, "audit_row": audit_row, **evidence,
        })

    if len(endpoint_ids) > 1:
        endpoint_points = nodes[endpoint_ids]
        endpoint_tree = KDTree(endpoint_points)
        for local_idx, start_idx in enumerate(endpoint_ids):
            neighbor_local_ids = endpoint_tree.query_radius(
                endpoint_points[local_idx][np.newaxis, :], r=parameters["max_gap"]
            )[0]
            for neighbor_local_idx in neighbor_local_ids.tolist():
                end_idx = endpoint_ids[neighbor_local_idx]
                if end_idx <= start_idx:
                    continue
                evaluate(start_idx, nodes[end_idx], "bridge", end_idx=end_idx)

    traversable = road >= low_threshold
    if surface is not None:
        traversable |= surface >= parameters["surface_threshold"]
    weak_skeleton = skeletonize(traversable)
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(weak_skeleton.astype(np.uint8), -1, neighbor_kernel) - weak_skeleton
    terminal_rc = np.column_stack(np.where(weak_skeleton & (neighbor_count == 1))).astype(np.float32)
    if len(terminal_rc):
        terminal_tree = KDTree(terminal_rc)
        node_tree = KDTree(nodes)
        for start_idx in endpoint_ids:
            candidates = terminal_tree.query_radius(
                nodes[start_idx][np.newaxis, :], r=parameters["max_extension"]
            )[0]
            for terminal_idx in candidates.tolist():
                target = terminal_rc[terminal_idx]
                nearest_distance, _ = node_tree.query(target[np.newaxis, :], k=1)
                if float(nearest_distance[0, 0]) < max(2.0, parameters["min_extension"] * 0.25):
                    summary["weak_candidate_count"] += 1
                    reject_candidate({
                        "candidate_type": "extension",
                        "start": nodes[start_idx].astype(float).tolist(),
                        "end": target.astype(float).tolist(),
                        "distance": float(np.linalg.norm(target - nodes[start_idx])),
                        "direction_cosine": None,
                        "path_length": None,
                        "path_ratio": None,
                        "mean_probability": None,
                        "q25_probability": None,
                        "weak_fraction": None,
                        "background_probability": None,
                        "background_contrast": None,
                        "surface_probability": None,
                        "accepted": False,
                        "reject_reason": "",
                    }, "duplicate_or_suppressed")
                    continue
                evaluate(start_idx, target, "extension")

    used_endpoints = set()
    combined_nodes = nodes.tolist()
    combined_edges = original_edges.tolist()
    existing_edges = {tuple(sorted(edge)) for edge in combined_edges}
    reason_counts = Counter()
    recovery_id = 0
    for proposal in sorted(proposals, key=lambda row: row["score"], reverse=True):
        endpoint_members = {proposal["start_idx"]}
        if proposal["end_idx"] is not None:
            endpoint_members.add(proposal["end_idx"])
        if endpoint_members & used_endpoints:
            reject_candidate(proposal["audit_row"], "endpoint_already_used")
            continue
        path = proposal["path"]
        path_steps = np.concatenate([
            np.asarray([0.0], dtype=np.float32),
            np.cumsum(np.linalg.norm(np.diff(path.astype(np.float32), axis=0), axis=1)),
        ])
        sample_distances = np.arange(parameters["sample_step"], path_steps[-1], parameters["sample_step"])
        chain = [proposal["start_idx"]]
        for sample_distance in sample_distances.tolist():
            path_idx = min(int(np.searchsorted(path_steps, sample_distance)), len(path) - 1)
            sampled_point = path[path_idx].astype(np.float32)
            if np.array_equal(np.rint(combined_nodes[chain[-1]]), np.rint(sampled_point)):
                continue
            combined_nodes.append(sampled_point.tolist())
            chain.append(len(combined_nodes) - 1)
        if proposal["end_idx"] is None:
            if not np.array_equal(
                np.rint(combined_nodes[chain[-1]]), np.rint(proposal["end_point"])
            ):
                combined_nodes.append(proposal["end_point"].tolist())
                chain.append(len(combined_nodes) - 1)
        else:
            if (
                chain[-1] >= len(nodes)
                and chain[-1] == len(combined_nodes) - 1
                and np.array_equal(
                    np.rint(combined_nodes[chain[-1]]), np.rint(nodes[proposal["end_idx"]])
                )
            ):
                chain.pop()
                combined_nodes.pop()
            chain.append(proposal["end_idx"])
        added_count = 0
        qa_state = "auto" if proposal["score"] >= parameters["auto_score"] else "review"
        for src_idx, dst_idx in zip(chain[:-1], chain[1:]):
            key = tuple(sorted((int(src_idx), int(dst_idx))))
            if src_idx == dst_idx or key in existing_edges:
                continue
            existing_edges.add(key)
            combined_edges.append((int(src_idx), int(dst_idx)))
            metadata.append({
                "line_source": "weak_recovered",
                "topology_probability": float(proposal["score"]),
                "recovery_score": float(proposal["score"]),
                "center_conf": float(proposal["center_conf"]),
                "surface_conf": float(proposal["surface_conf"]),
                "recovery_reason": proposal["reason"],
                "qa_state": qa_state,
                "recovery_id": f"weak:{recovery_id}",
            })
            added_count += 1
        if added_count:
            used_endpoints.update(endpoint_members)
            recovery_id += 1
            summary["weak_recovered_candidate_count"] += 1
            reason_counts[proposal["reason"]] += added_count
            summary["weak_recovered_edge_count"] += added_count
            if proposal["surface_supported"]:
                summary["surface_supported_recovery_count"] += added_count
            proposal["audit_row"]["accepted"] = True
            proposal["audit_row"]["reject_reason"] = ""
            if candidate_audit is not None:
                candidate_audit.append(proposal["audit_row"])
        else:
            reject_candidate(proposal["audit_row"], "duplicate_or_suppressed")
    summary["weak_recovery_reject_reason_counts"] = dict(sorted(reject_counts.items()))
    summary["recovery_reason_counts"] = dict(sorted(reason_counts.items()))
    return (
        np.asarray(combined_nodes, dtype=np.float32).reshape(-1, 2),
        np.asarray(combined_edges, dtype=np.int32).reshape(-1, 2),
        metadata,
        summary,
    )


def postprocess_weak_road_network(
    nodes_rc,
    edges,
    road_probability,
    config,
    *,
    surface_probability=None,
    edge_scores=None,
    distance_scale=1.0,
    weak_candidate_audit=None,
    bootstrap_candidate_audit=None,
):
    """Diagnose, optionally bootstrap, then run the existing endpoint recovery."""
    original_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    diagnosis = diagnose_scene_confidence(
        road_probability,
        nodes_rc,
        original_edges,
        config,
        distance_scale=distance_scale,
    )
    bootstrap_enabled = bool(_config_value(config, "WEAK_BOOTSTRAP_ENABLED", True))
    only_low = bool(_config_value(config, "WEAK_BOOTSTRAP_ONLY_IF_LOW_CONFIDENCE", True))
    should_bootstrap = bootstrap_enabled and (
        not only_low or diagnosis["scene_confidence_state"] in {"low_confidence", "very_low_confidence"}
    )
    if should_bootstrap:
        bootstrap_nodes, bootstrap_edges, bootstrap_metadata, bootstrap_summary = (
            bootstrap_weak_road_network(
                nodes_rc,
                original_edges,
                road_probability,
                config,
                surface_probability=surface_probability,
                edge_scores=edge_scores,
                distance_scale=distance_scale,
                candidate_audit=bootstrap_candidate_audit,
            )
        )
    else:
        bootstrap_nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
        bootstrap_edges = original_edges
        bootstrap_metadata = []
        bootstrap_summary = {
            "bootstrap_candidate_count": 0,
            "bootstrap_accepted_candidate_count": 0,
            "bootstrap_recovered_edge_count": 0,
            "bootstrap_auto_count": 0,
            "bootstrap_review_count": 0,
            "bootstrap_rejected_count": 0,
            "bootstrap_reject_reason_counts": {},
        }
    if bootstrap_metadata:
        combined_scores = np.asarray(
            [row.get("topology_probability", np.nan) for row in bootstrap_metadata],
            dtype=np.float32,
        )
    else:
        combined_scores = edge_scores
    final_nodes, final_edges, final_metadata, recovery_summary = recover_weak_road_edges(
        bootstrap_nodes,
        bootstrap_edges,
        road_probability,
        config,
        surface_probability=surface_probability,
        edge_scores=combined_scores,
        distance_scale=distance_scale,
        candidate_audit=weak_candidate_audit,
    )
    if bootstrap_metadata:
        final_metadata[:len(bootstrap_metadata)] = bootstrap_metadata
    summary = {
        **recovery_summary,
        **diagnosis,
        **bootstrap_summary,
        "strong_edge_count": int(len(original_edges)),
        "bootstrap_ran": bool(should_bootstrap),
    }
    return final_nodes, final_edges, final_metadata, summary


def skeletonize_road_mask(road_mask, threshold, close_kernel_size=3):
    """
    Convert a road probability map into a thin centerline-friendly binary mask.

    A small closing step helps bridge tiny gaps before skeletonization, which is
    especially useful on long straight roads and at patch seams.
    """
    road_bin = (road_mask > threshold).astype(np.uint8)
    if close_kernel_size and close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        road_bin = cv2.morphologyEx(road_bin, cv2.MORPH_CLOSE, kernel)
    road_skel = skeletonize(road_bin.astype(bool))
    return (road_skel.astype(np.uint8)) * 255


def estimate_skeleton_tangents(points, tangent_radius=5.0):
    """Estimate a local unoriented tangent for every skeleton point."""
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    tree = KDTree(points)
    tangents = np.zeros((points.shape[0], 2), dtype=np.float32)
    for point_idx, point in enumerate(points):
        neighbor_indices = tree.query_radius(point[np.newaxis, :], r=tangent_radius)[0]
        neighborhood = points[neighbor_indices].astype(np.float32)
        if neighborhood.shape[0] < 2:
            tangents[point_idx] = (1.0, 0.0)
            continue
        centered = neighborhood - np.mean(neighborhood, axis=0, keepdims=True)
        covariance = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, -1]
        norm = float(np.linalg.norm(tangent))
        tangents[point_idx] = tangent / max(norm, 1e-6)
    return tangents


def branch_aware_nms_points(
    points,
    scores,
    radius,
    min_separation=4.0,
    tangent_radius=5.0,
    parallel_cosine=0.90,
    lateral_cosine=0.55,
):
    """Downsample a skeleton without suppressing a nearby parallel branch.

    Circular NMS collapses divided-road centerlines whenever their separation is
    smaller than the sampling radius. Parallel points are retained when their
    separation is primarily across, rather than along, the local road tangent.
    """
    if points.shape[0] == 0:
        return points
    points = np.asarray(points, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    tangents = estimate_skeleton_tangents(points, tangent_radius=tangent_radius)
    order = np.argsort(-scores, kind="stable")
    active = np.ones(points.shape[0], dtype=bool)
    tree = KDTree(points)
    for point_idx in order.tolist():
        if not active[point_idx]:
            continue
        neighbor_indices = tree.query_radius(points[point_idx][np.newaxis, :], r=radius)[0]
        for neighbor_idx in neighbor_indices.tolist():
            if neighbor_idx == point_idx or not active[neighbor_idx]:
                continue
            delta = points[neighbor_idx] - points[point_idx]
            distance = float(np.linalg.norm(delta))
            if distance <= min_separation:
                active[neighbor_idx] = False
                continue
            direction = delta / max(distance, 1e-6)
            tangent_alignment = abs(float(np.dot(tangents[point_idx], tangents[neighbor_idx])))
            along_alignment = max(
                abs(float(np.dot(direction, tangents[point_idx]))),
                abs(float(np.dot(direction, tangents[neighbor_idx]))),
            )
            is_parallel_branch = tangent_alignment >= parallel_cosine and along_alignment <= lateral_cosine
            if not is_parallel_branch:
                active[neighbor_idx] = False
    return points[active]


def extract_graph_points(keypoint_mask, road_mask, config):
    high_threshold, _low_threshold, _profile_name = resolve_road_thresholds(config)
    kp_candidates, kp_scores = get_points_and_scores_from_mask(keypoint_mask, config.ITSC_THRESHOLD * 255)
    kps_0 = nms_points(kp_candidates, kp_scores, config.ITSC_NMS_RADIUS)
    # The keypoint heatmap contains broad blobs. Keep its historical coarse
    # spacing; divided-road preservation is applied only to the road skeleton.
    if kps_0.shape[0]:
        kps_0 = nms_points(kps_0, np.ones(kps_0.shape[0]), config.ROAD_NMS_RADIUS)
    road_skel_mask = skeletonize_road_mask(road_mask, high_threshold * 255)
    kp_candidates, kp_scores = get_points_and_scores_from_mask(road_skel_mask, 0)
    road_scores = road_mask[kp_candidates[:, 1], kp_candidates[:, 0]] if kp_candidates.shape[0] else kp_scores
    kps_1 = branch_aware_nms_points(
        kp_candidates,
        road_scores,
        config.ROAD_NMS_RADIUS,
        min_separation=float(config.get("ROAD_NMS_MIN_SEPARATION", 4.0)),
        tangent_radius=float(config.get("ROAD_TANGENT_RADIUS", 5.0)),
        parallel_cosine=float(config.get("PARALLEL_BRANCH_COSINE", 0.90)),
        lateral_cosine=float(config.get("PARALLEL_BRANCH_LATERAL_COSINE", 0.55)),
    )
    # Sparse mode keeps the learned intersection core and removes ordinary
    # skeleton samples from its immediate neighborhood.  The legacy 4-pixel
    # merge radius remains selectable for projects that benefit from dense
    # junction sampling.
    if kps_0.shape[0] and kps_1.shape[0]:
        tree = KDTree(kps_0)
        nearest_distance, _ = tree.query(kps_1, k=1)
        junction_mode = str(config.get("JUNCTION_NODE_MODE", "sparse")).strip().casefold()
        if junction_mode not in {"sparse", "dense_legacy"}:
            raise ValueError(f"Unsupported JUNCTION_NODE_MODE: {junction_mode}")
        merge_radius = (
            float(config.get("JUNCTION_SPARSE_RADIUS", 20.0))
            if junction_mode == "sparse"
            else float(config.get("JUNCTION_POINT_MERGE_RADIUS", 4.0))
        )
        kps_1 = kps_1[nearest_distance[:, 0] > merge_radius]
    return np.concatenate([kps_0, kps_1], axis=0).astype(np.float32)


def extract_graph_astar(keypoint_mask, road_mask, config):
    kps = extract_graph_points(keypoint_mask, road_mask, config)

    # cost_field = create_cost_field(kps, road_mask)
    _high_threshold, low_threshold, _profile_name = resolve_road_thresholds(config)
    cost_field = create_cost_field_astar(kps, road_mask, low_threshold=low_threshold)
    viz_cost_field = np.array(cost_field)
    viz_cost_field[viz_cost_field == 0] = 255
    # cv2.imwrite('astar_cost_dbg.png', viz_cost_field)
    pathfinder = tcod.path.AStar(cost_field)

    tree = KDTree(kps)
    graph = nx.Graph()
    checked = set()
    for p in kps:
        # TODO: add radius to config
        neighbor_indices = tree.query_radius(p[np.newaxis, :], r=config.NEIGHBOR_RADIUS)[0]
        for n_idx in neighbor_indices:
            n = kps[n_idx]
            start, end = (int(p[0]), int(p[1])), (int(n[0]), int(n[1]))
            if (start, end) in checked:
                continue
            # if is_connected_bresenham(cost_field, p, n):
            if is_connected_astar(pathfinder, cost_field, p, n, max_path_len=config.NEIGHBOR_RADIUS):
                graph.add_edge(start, end)
            checked.add((start, end))
    return graph

# takes xys    
def visualize_image_and_graph(img, graph):
    # Draw nodes as green squares
    for node in graph.nodes():
        x, y = node
        cv2.rectangle(
            img, (int(x) - 2, int(y) - 2), (int(x) + 2, int(y) + 2), (0, 255, 0), -1
        )
    # Draw edges as white lines
    for start_node, end_node in graph.edges():
        cv2.line(
            img,
            (int(start_node[0]), int(start_node[1])),
            (int(end_node[0]), int(end_node[1])),
            (255, 255, 255),
            1,
        )
    return img
    

if __name__ == '__main__':

    # cost = np.array(
    #     [[1, 0, 1],
    #      [0, 1, 0],
    #      [0, 0, 0]],
    #      dtype=np.int32
    # )
    # pathfinder = tcod.path.AStar(cost)
    # print(pathfinder.get_path(0, 2, 0, 0))
    # cost[1, 1] = 0
    # print(pathfinder.get_path(0, 2, 0, 0))
    # cost[1, 1] = 1
    # print(pathfinder.get_path(0, 2, 0, 0))

    rgb_pattern = './cityscale/20cities/region_{}_sat.png'
    keypoint_mask_pattern = './cityscale/processed/keypoint_mask_{}.png'
    road_mask_pattern = './cityscale/processed/road_mask_{}.png'

    index = 0
    rgb = read_rgb_img(rgb_pattern.format(index))
    road_mask = cv2.imread(road_mask_pattern.format(index), cv2.IMREAD_GRAYSCALE)
    keypoint_mask = cv2.imread(keypoint_mask_pattern.format(index), cv2.IMREAD_GRAYSCALE)

    graph = extract_graph_astar(keypoint_mask, road_mask)
    viz = visualize_image_and_graph(rgb, graph)
    cv2.imwrite('test_graph_astar_blk6_r40_m40_inms.png', viz)
