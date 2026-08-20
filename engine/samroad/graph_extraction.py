import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import tcod
from collections import Counter, defaultdict, deque
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
    if profile_name == "auto":
        raise ValueError(
            "ROAD_THRESHOLD_PROFILE=auto must be resolved from the complete "
            "road-probability image before thresholds are consumed."
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


def _undirected_adjacency(node_count, edges):
    """Build unique undirected neighbors from a possibly reciprocal edge list."""
    adjacency = [set() for _ in range(int(node_count))]
    for src_idx, dst_idx in np.asarray(edges, dtype=np.int32).reshape(-1, 2).tolist():
        if src_idx == dst_idx:
            continue
        adjacency[src_idx].add(dst_idx)
        adjacency[dst_idx].add(src_idx)
    return adjacency


def graph_connectivity_stats(nodes_rc, edges):
    """Return topology metrics using unique undirected neighbors and edges."""
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    unique_edges = {
        tuple(sorted((int(src_idx), int(dst_idx))))
        for src_idx, dst_idx in np.asarray(edges, dtype=np.int32).reshape(-1, 2).tolist()
        if int(src_idx) != int(dst_idx)
    }
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(unique_edges)
    components = list(nx.connected_components(graph))
    largest = max(components, key=len) if components else set()
    largest_edge_count = sum(
        int(src_idx in largest and dst_idx in largest) for src_idx, dst_idx in unique_edges
    )
    node_count = len(nodes)
    return {
        "component_count": int(len(components)),
        "endpoint_count": int(sum(graph.degree(node_idx) == 1 for node_idx in graph.nodes)),
        "largest_component_node_count": int(len(largest)),
        "largest_component_edge_count": int(largest_edge_count),
        "largest_component_fraction": float(len(largest) / node_count) if node_count else 0.0,
    }


def _component_labels(node_count, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(int(node_count)))
    graph.add_edges_from({
        tuple(sorted((int(src_idx), int(dst_idx))))
        for src_idx, dst_idx in np.asarray(edges, dtype=np.int32).reshape(-1, 2).tolist()
        if int(src_idx) != int(dst_idx)
    })
    labels = {}
    for component_id, members in enumerate(nx.connected_components(graph)):
        for node_idx in members:
            labels[int(node_idx)] = int(component_id)
    return labels


def estimate_endpoint_direction(nodes_rc, edges, endpoint_idx, lookback_distance=32.0):
    """Estimate the outward endpoint tangent from a short inward graph trace."""
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    adjacency = _undirected_adjacency(len(nodes), edges)
    endpoint_idx = int(endpoint_idx)
    if endpoint_idx < 0 or endpoint_idx >= len(nodes) or len(adjacency[endpoint_idx]) != 1:
        return None
    endpoint = nodes[endpoint_idx]
    previous = endpoint_idx
    current = next(iter(adjacency[endpoint_idx]))
    travelled = 0.0
    inward_point = nodes[current].copy()
    target_distance = max(float(lookback_distance), 1e-6)
    while True:
        segment_start = nodes[previous]
        segment_end = nodes[current]
        segment_length = float(np.linalg.norm(segment_end - segment_start))
        if travelled + segment_length >= target_distance and segment_length > 1e-6:
            fraction = (target_distance - travelled) / segment_length
            inward_point = segment_start + fraction * (segment_end - segment_start)
            break
        travelled += segment_length
        inward_point = segment_end.copy()
        next_nodes = adjacency[current] - {previous}
        if len(adjacency[current]) != 2 or len(next_nodes) != 1:
            break
        previous, current = current, next(iter(next_nodes))
    vector = endpoint - inward_point
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else None


def _endpoint_vectors(nodes_rc, edges, lookback_distance=32.0):
    # SAMRoad can emit reciprocal directed edges. Endpoint degree must be based
    # on unique neighboring nodes, otherwise a true degree-one endpoint appears
    # to have degree two and is silently excluded from recovery proposals.
    nodes_rc = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    adjacency = [set() for _ in range(len(nodes_rc))]
    for src_idx, dst_idx in np.asarray(edges, dtype=np.int32).reshape(-1, 2).tolist():
        adjacency[src_idx].add(dst_idx)
        adjacency[dst_idx].add(src_idx)
    vectors = {}
    for node_idx, neighbors in enumerate(adjacency):
        if len(neighbors) != 1:
            continue
        vector = estimate_endpoint_direction(
            nodes_rc, edges, node_idx, lookback_distance=lookback_distance
        )
        if vector is not None:
            vectors[node_idx] = vector
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


def _relative_path_evidence(path, relative_context):
    """Sample relative-roadness evidence without imposing raw probability gates."""
    if not relative_context:
        return {
            "relative_score_mean": 0.0,
            "relative_score_q25": 0.0,
            "scene_rank_mean": 0.0,
            "local_background_mean": 0.0,
            "local_contrast_mean": 0.0,
            "normalized_contrast_mean": 0.0,
            "scale_agreement_mean": 0.0,
            "scale_agreement_q25": 0.0,
            "relative_fraction": 0.0,
            "relative_only_fraction": 0.0,
            "relative_supported": False,
        }
    score = np.asarray(relative_context.get("relative_score", []), dtype=np.float32)
    if score.ndim != 2 or score.size == 0:
        return _relative_path_evidence(path, None)
    rows = np.clip(path[:, 0], 0, score.shape[0] - 1)
    cols = np.clip(path[:, 1], 0, score.shape[1] - 1)

    def sampled(name):
        value = relative_context.get(name)
        if value is None:
            return np.zeros(len(rows), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        return array[rows, cols] if array.shape == score.shape else np.zeros(len(rows), dtype=np.float32)

    values = sampled("relative_score")
    scene_rank = sampled("scene_rank")
    local_background = sampled("local_background")
    local_contrast = sampled("local_contrast")
    normalized_contrast = sampled("normalized_contrast")
    scale_agreement = sampled("scale_agreement_fraction")
    relative_structure = sampled("relative_skeleton") > 0
    relative_only = sampled("relative_only_skeleton") > 0
    diagnostics = relative_context.get("diagnostics", {})
    weak_threshold = float(diagnostics.get("relative_weak_threshold", 0.0))
    q25 = float(np.quantile(values, 0.25)) if values.size else 0.0
    fraction = float(np.mean(relative_structure)) if values.size else 0.0
    relative_only_fraction = float(np.mean(relative_only)) if values.size else 0.0
    return {
        "relative_score_mean": float(np.mean(values)) if values.size else 0.0,
        "relative_score_q25": q25,
        "scene_rank_mean": float(np.mean(scene_rank)) if values.size else 0.0,
        "local_background_mean": float(np.mean(local_background)) if values.size else 0.0,
        "local_contrast_mean": float(np.mean(local_contrast)) if values.size else 0.0,
        "normalized_contrast_mean": float(np.mean(normalized_contrast)) if values.size else 0.0,
        "scale_agreement_mean": float(np.mean(scale_agreement)) if values.size else 0.0,
        "scale_agreement_q25": float(np.quantile(scale_agreement, 0.25)) if values.size else 0.0,
        "relative_fraction": fraction,
        "relative_only_fraction": relative_only_fraction,
        "relative_supported": bool(
            fraction >= 0.50
            and q25 >= max(weak_threshold - 1e-6, 0.0)
            and float(np.mean(normalized_contrast)) > 0.0
        ),
    }


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


def _topology_candidate_rasters(nodes_rc, edges, scores, shape, radius=3):
    """Rasterize retained TopoNet candidates as optional Relative evidence."""
    mask = np.zeros(shape, dtype=np.uint8)
    score_map = np.zeros(shape, dtype=np.float32)
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    graph_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(values) != len(graph_edges):
        return mask, score_map
    for (src_idx, dst_idx), score in zip(graph_edges.tolist(), values.tolist()):
        if not (0 <= src_idx < len(nodes) and 0 <= dst_idx < len(nodes)):
            continue
        rr, cc = line(
            int(round(nodes[src_idx, 0])), int(round(nodes[src_idx, 1])),
            int(round(nodes[dst_idx, 0])), int(round(nodes[dst_idx, 1])),
        )
        rr = np.clip(rr, 0, shape[0] - 1)
        cc = np.clip(cc, 0, shape[1] - 1)
        mask[rr, cc] = 1
        score_map[rr, cc] = np.maximum(score_map[rr, cc], float(score))
    if radius > 0 and np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask, kernel)
        score_map = cv2.dilate(score_map, kernel)
    return mask, score_map


def _endpoint_alignment(path, nodes, edges, connection_radius):
    """Measure tangent agreement where Relative endpoints approach an absolute graph."""
    graph_nodes = np.asarray(nodes, dtype=np.float32).reshape(-1, 2)
    graph_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    if len(graph_nodes) == 0 or len(graph_edges) == 0 or len(path) < 2:
        return 0.0
    incident = defaultdict(list)
    for src_idx, dst_idx in graph_edges.tolist():
        vector = graph_nodes[dst_idx] - graph_nodes[src_idx]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            continue
        unit = vector / norm
        incident[int(src_idx)].append(unit)
        incident[int(dst_idx)].append(unit)
    tree = KDTree(graph_nodes)
    alignments = []
    path_float = np.asarray(path, dtype=np.float32)
    for endpoint_index, neighbor_index in ((0, min(8, len(path_float) - 1)), (-1, max(0, len(path_float) - 9))):
        endpoint = path_float[endpoint_index]
        tangent = path_float[neighbor_index] - endpoint
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-6:
            continue
        distance, node_index = tree.query(endpoint[np.newaxis, :], k=1)
        if float(distance[0, 0]) > connection_radius:
            continue
        directions = incident.get(int(node_index[0, 0]), [])
        if directions:
            unit_tangent = tangent / tangent_norm
            alignments.append(max(abs(float(np.dot(unit_tangent, direction))) for direction in directions))
    return float(np.mean(alignments)) if alignments else 0.0


def _skeleton_adjacency(skeleton):
    """Return corner-safe 8-connected adjacency for foreground skeleton pixels."""
    pixels = {tuple(value) for value in np.column_stack(np.where(skeleton)).tolist()}
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

    return {point: neighbors(point) for point in pixels}


def _trace_skeleton_chains(skeleton):
    """Trace an 8-connected skeleton into junction/endpoint-to-junction chains."""
    adjacency = _skeleton_adjacency(skeleton)
    if not adjacency:
        return []

    pixels = set(adjacency)
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


def _shortest_path_in_skeleton(mask, starts, goals):
    """Find an unweighted skeleton path between two pixel sets."""
    adjacency = _skeleton_adjacency(mask)
    start_points = [tuple(map(int, point)) for point in starts if tuple(map(int, point)) in adjacency]
    goal_points = {tuple(map(int, point)) for point in goals if tuple(map(int, point)) in adjacency}
    if not start_points or not goal_points:
        return []
    queue = deque(start_points)
    parents = {point: None for point in start_points}
    target = None
    while queue:
        point = queue.popleft()
        if point in goal_points:
            target = point
            break
        for neighbor in adjacency[point]:
            if neighbor not in parents:
                parents[neighbor] = point
                queue.append(neighbor)
    if target is None:
        return []
    result = []
    while target is not None:
        result.append(target)
        target = parents[target]
    result.reverse()
    return result


def _skeleton_chain_summary(skeleton, min_length):
    chains = _trace_skeleton_chains(skeleton)
    lengths = [float(_relative_chain_geometry(path)["path_length"]) for path in chains]
    return chains, lengths, {
        "chain_count": int(len(chains)),
        "short_chain_count": int(sum(length < min_length for length in lengths)),
        "median_chain_length": float(np.median(lengths)) if lengths else 0.0,
        "max_chain_length": float(max(lengths)) if lengths else 0.0,
    }


def _junction_zone_radius(config, distance_scale):
    """Tie junction clustering to the smallest existing local-background scale."""
    configured = _config_value(config, "RELATIVE_ROADNESS_BACKGROUND_SCALES_PX", [9, 21, 41])
    if not isinstance(configured, (list, tuple)):
        configured = [configured]
    positive = [float(value) for value in configured if float(value) > 0]
    reference = min(positive) if positive else 9.0
    return max(2, int(round(reference * 0.9 * max(float(distance_scale), 1e-6))))


def _junction_cluster_radius(config, distance_scale):
    """Return the grouping reach while keeping the narrower zone influence."""
    configured = _config_value(config, "RELATIVE_ROADNESS_BACKGROUND_SCALES_PX", [9, 21, 41])
    if not isinstance(configured, (list, tuple)):
        configured = [configured]
    positive = [float(value) for value in configured if float(value) > 0]
    reference = min(positive) if positive else 9.0
    return max(2, int(round(reference * 1.25 * max(float(distance_scale), 1e-6))))


def find_skeleton_junction_zones(skeleton, config, *, distance_scale=1.0):
    """Cluster nearby degree>=3 pixels into spatial junction zones."""
    value = np.asarray(skeleton, dtype=bool)
    adjacency = _skeleton_adjacency(value)
    junction_mask = np.zeros(value.shape, dtype=np.uint8)
    for point, neighbors in adjacency.items():
        if len(neighbors) >= 3:
            junction_mask[point] = 1
    radius = _junction_zone_radius(config, distance_scale)
    cluster_radius = _junction_cluster_radius(config, distance_scale)
    cluster_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cluster_radius * 2 + 1, cluster_radius * 2 + 1)
    )
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    expanded = (
        cv2.dilate(junction_mask, cluster_kernel)
        if np.any(junction_mask) else junction_mask.copy()
    )
    zone_count, expanded_labels = cv2.connectedComponents(expanded.astype(np.uint8), 8)
    zone_labels = np.zeros(value.shape, dtype=np.int32)
    zones = []
    next_zone = 1
    for label_id in range(1, zone_count):
        member_mask = (expanded_labels == label_id) & (junction_mask > 0)
        points = np.column_stack(np.where(member_mask))
        if len(points) == 0:
            continue
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        # Cluster with the wider reach, then form the actual zone from a thin
        # convex corridor around its member junctions.  This bridges repeated
        # ladder junctions without giving the zone enough lateral reach to
        # swallow a real branch or an adjacent parallel road.
        row0 = max(0, int(minimum[0]) - radius)
        row1 = min(value.shape[0], int(maximum[0]) + radius + 1)
        col0 = max(0, int(minimum[1]) - radius)
        col1 = min(value.shape[1], int(maximum[1]) + radius + 1)
        member_core = np.zeros((row1 - row0, col1 - col0), dtype=np.uint8)
        hull_points = (
            points[:, ::-1].astype(np.int32)
            - np.asarray([col0, row0], dtype=np.int32)
        )
        if len(hull_points) >= 3:
            cv2.fillConvexPoly(member_core, cv2.convexHull(hull_points), 1)
        elif len(hull_points) == 2:
            cv2.line(member_core, tuple(hull_points[0]), tuple(hull_points[1]), 1, 1)
        else:
            member_core[int(points[0, 0]) - row0, int(points[0, 1]) - col0] = 1
        support = cv2.dilate(member_core, support_kernel) > 0
        zone_crop = zone_labels[row0:row1, col0:col1]
        zone_crop[support] = next_zone
        centered = points.astype(np.float32) - np.mean(points, axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_index = int(np.argmax(eigenvalues))
        major = eigenvectors[:, major_index].astype(np.float32)
        major_value = float(eigenvalues[major_index])
        minor_value = float(eigenvalues[1 - major_index])
        explained = major_value / max(major_value + minor_value, 1e-6)
        zones.append({
            "zone_id": next_zone,
            "pixel_count": int(len(points)),
            "bbox": [int(minimum[0]), int(minimum[1]), int(maximum[0]), int(maximum[1])],
            "centroid": np.mean(points, axis=0).astype(np.float32),
            "major_direction": major,
            "major_explained_fraction": float(explained),
            "cluster_radius": int(cluster_radius),
            "incident_branches": [],
            "branch_tangents": [],
            "branch_lengths": [],
        })
        next_zone += 1
    return junction_mask, zone_labels, zones, radius


def _zone_incident_branches(
    skeleton,
    zone_mask,
    relative_score,
    candidate_mask,
    scale_agreement,
):
    """Summarize outside branches touching one junction zone."""
    value = np.asarray(skeleton, dtype=bool)
    outside = value & ~zone_mask
    component_count, component_labels = cv2.connectedComponents(outside.astype(np.uint8), 8)
    contact = outside & (cv2.dilate(zone_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    contact_count, contact_labels = cv2.connectedComponents(contact.astype(np.uint8), 8)
    inside = value & zone_mask
    inside_points = np.column_stack(np.where(inside))
    records = []
    for contact_id in range(1, contact_count):
        contact_points = np.column_stack(np.where(contact_labels == contact_id))
        if len(contact_points) == 0:
            continue
        component_values = component_labels[contact_points[:, 0], contact_points[:, 1]]
        component_values = component_values[component_values > 0]
        if len(component_values) == 0:
            continue
        component_id = int(np.bincount(component_values).argmax())
        component_points = np.column_stack(np.where(component_labels == component_id))
        entry = np.mean(contact_points, axis=0).astype(np.float32)
        offsets = component_points.astype(np.float32) - entry[None, :]
        distances = np.linalg.norm(offsets, axis=1)
        continuation = float(np.max(distances)) if len(distances) else 0.0
        local = (distances >= min(8.0, max(continuation * 0.4, 1.0))) & (distances <= 30.0)
        if not np.any(local):
            local = distances == np.max(distances)
        local_offsets = offsets[local]
        local_distances = distances[local]
        tangent_offset = local_offsets[int(np.argmax(local_distances))]
        tangent_norm = float(np.linalg.norm(tangent_offset))
        tangent = tangent_offset / max(tangent_norm, 1e-6)
        sample_points = component_points[distances <= min(30.0, max(continuation, 1.0))]
        if len(sample_points) == 0:
            sample_points = contact_points
        rr = sample_points[:, 0]
        cc = sample_points[:, 1]
        score_mean = float(np.mean(relative_score[rr, cc])) if relative_score is not None else 0.0
        candidate_fraction = float(np.mean(candidate_mask[rr, cc] > 0))
        scale_mean = float(np.mean(scale_agreement[rr, cc])) if scale_agreement is not None else 0.0
        if len(inside_points):
            delta = inside_points.astype(np.float32)[:, None, :] - contact_points.astype(np.float32)[None, :, :]
            nearest_flat = int(np.argmin(np.sum(delta * delta, axis=2)))
            inside_index, _contact_index = np.unravel_index(nearest_flat, delta.shape[:2])
            inside_entry = inside_points[inside_index]
        else:
            inside_entry = np.rint(entry).astype(np.int32)
        records.append({
            "contact_points": contact_points,
            "inside_entry": np.asarray(inside_entry, dtype=np.int32),
            "component_id": component_id,
            "component_points": component_points,
            "continuation_length": continuation,
            "outward_tangent": tangent.astype(np.float32),
            "relative_score": score_mean,
            "candidate_coverage": candidate_fraction,
            "scale_agreement": scale_mean,
        })
    return records


def normalize_relative_skeleton(
    raw_skeleton,
    candidate_mask,
    config,
    *,
    relative_score=None,
    scale_agreement=None,
    distance_scale=1.0,
):
    """Collapse elongated dense-junction ladder zones into meaningful corridors."""
    raw = np.asarray(raw_skeleton, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=np.uint8)
    score = None if relative_score is None else np.asarray(relative_score, dtype=np.float32)
    agreement = None if scale_agreement is None else np.asarray(scale_agreement, dtype=np.float32)
    minimum_length = float(_config_value(
        config, "RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX", 48.0
    )) * max(float(distance_scale), 1e-6)
    raw_chains, raw_lengths, raw_summary = _skeleton_chain_summary(raw, minimum_length)
    junction_mask, zone_labels, zones, radius = find_skeleton_junction_zones(
        raw, config, distance_scale=distance_scale
    )
    normalized = raw.copy()
    pruned_spurs = np.zeros(raw.shape, dtype=bool)
    collapsed_zones = np.zeros(raw.shape, dtype=bool)
    collapsed_count = 0
    pruned_count = 0
    zone_records = []

    for zone in zones:
        zone_record = {
            "zone_id": int(zone["zone_id"]),
            "pixel_count": int(zone["pixel_count"]),
            "bbox": zone["bbox"],
            "centroid": np.asarray(zone["centroid"], dtype=float).tolist(),
            "major_direction": np.asarray(zone["major_direction"], dtype=float).tolist(),
            "major_explained_fraction": float(zone["major_explained_fraction"]),
            "incident_branch_count": 0,
            "incident_branches": [],
            "branch_tangents": [],
            "branch_lengths": [],
            "preserved_branch_count": 0,
            "collapsed": False,
            "skip_reason": "",
        }
        # Single real T/X junctions are compact. Ladder artifacts contain many
        # nearby junction pixels distributed along one dominant corridor.
        if zone["pixel_count"] < 4 or zone["major_explained_fraction"] < 0.72:
            zone_record["skip_reason"] = "compact_or_no_dominant_corridor"
            zone_records.append(zone_record)
            continue
        global_zone_mask = zone_labels == zone["zone_id"]
        zone_rows, zone_cols = np.where(global_zone_mask)
        padding = max(32, radius * 4)
        row0 = max(0, int(zone_rows.min()) - padding)
        row1 = min(raw.shape[0], int(zone_rows.max()) + padding + 1)
        col0 = max(0, int(zone_cols.min()) - padding)
        col1 = min(raw.shape[1], int(zone_cols.max()) + padding + 1)
        region = np.s_[row0:row1, col0:col1]
        raw_crop = raw[region]
        zone_mask = global_zone_mask[region]
        inside = raw_crop & zone_mask
        if np.count_nonzero(inside) < 3:
            zone_record["skip_reason"] = "insufficient_inside_skeleton"
            zone_records.append(zone_record)
            continue
        branches = _zone_incident_branches(
            raw_crop,
            zone_mask,
            None if score is None else score[region],
            candidate[region],
            None if agreement is None else agreement[region],
        )
        branch_records = [
            {
                "branch_id": int(branch_index),
                "continuation_length": float(branch["continuation_length"]),
                "outward_tangent": np.asarray(
                    branch["outward_tangent"], dtype=float
                ).tolist(),
                "relative_score": float(branch["relative_score"]),
                "candidate_coverage": float(branch["candidate_coverage"]),
                "scale_agreement": float(branch["scale_agreement"]),
                "role": "incident",
            }
            for branch_index, branch in enumerate(branches)
        ]
        zone_record.update({
            "incident_branch_count": int(len(branches)),
            "incident_branches": branch_records,
            "branch_tangents": [row["outward_tangent"] for row in branch_records],
            "branch_lengths": [row["continuation_length"] for row in branch_records],
        })
        if len(branches) < 2:
            zone_record["skip_reason"] = "fewer_than_two_incident_branches"
            zone_records.append(zone_record)
            continue
        major = zone["major_direction"]
        branch_strengths = []
        for branch in branches:
            projection = float(np.dot(branch["outward_tangent"], major))
            evidence = (
                1.0
                + branch["relative_score"]
                + branch["candidate_coverage"]
                + branch["scale_agreement"]
            )
            strength = abs(projection) * max(branch["continuation_length"], 1.0) * evidence
            branch_strengths.append((projection, strength))
        positive = [index for index, item in enumerate(branch_strengths) if item[0] >= 0]
        negative = [index for index, item in enumerate(branch_strengths) if item[0] < 0]
        if not positive or not negative:
            zone_record["skip_reason"] = "no_opposite_through_pair"
            zone_records.append(zone_record)
            continue
        positive_index = max(positive, key=lambda index: branch_strengths[index][1])
        negative_index = max(negative, key=lambda index: branch_strengths[index][1])
        main_indices = {positive_index, negative_index}
        for branch_index in main_indices:
            branch_records[branch_index]["role"] = "through"
        main_path = _shortest_path_in_skeleton(
            inside,
            [branches[positive_index]["inside_entry"]],
            [branches[negative_index]["inside_entry"]],
        )
        if len(main_path) < 2:
            zone_record["skip_reason"] = "through_pair_not_connected_inside_zone"
            zone_records.append(zone_record)
            continue

        normalized_crop = normalized[region]
        collapsed_crop = collapsed_zones[region]
        pruned_crop = pruned_spurs[region]
        normalized_crop[inside] = False
        main_array = np.asarray(main_path, dtype=np.int32)
        normalized_crop[main_array[:, 0], main_array[:, 1]] = True
        collapsed_crop[inside] = True
        collapsed_count += 1
        preserved_branch_count = 0
        for branch_index, branch in enumerate(branches):
            if branch_index in main_indices:
                continue
            tangent = branch["outward_tangent"]
            main_alignment = abs(float(np.dot(tangent, major)))
            transverse_alignment = math.sqrt(max(0.0, 1.0 - main_alignment * main_alignment))
            extends_outside_zone = branch["continuation_length"] > 1.5 * radius
            evidence_continues = bool(
                branch["candidate_coverage"] >= 0.5
                and (branch["relative_score"] > 0.0 or branch["scale_agreement"] > 0.0)
            )
            # A same-direction rail inside the same dense zone is part of the
            # ladder. A transverse branch that genuinely exits the zone keeps
            # its topology even when it is shorter than 48 px.
            preserve = bool(
                transverse_alignment > main_alignment
                and extends_outside_zone
                and evidence_continues
            )
            if preserve:
                connector = _shortest_path_in_skeleton(
                    inside,
                    [branch["inside_entry"]],
                    main_path,
                )
                if connector:
                    connector_array = np.asarray(connector, dtype=np.int32)
                    normalized_crop[connector_array[:, 0], connector_array[:, 1]] = True
                    preserved_branch_count += 1
                    branch_records[branch_index]["role"] = "preserved_real_branch"
                else:
                    branch_records[branch_index]["role"] = "unconnected_transverse_branch"
            elif branch["continuation_length"] <= 2.0 * radius:
                component_points = branch["component_points"]
                normalized_crop[component_points[:, 0], component_points[:, 1]] = False
                pruned_crop[component_points[:, 0], component_points[:, 1]] = True
                pruned_count += 1
                branch_records[branch_index]["role"] = "pruned_ladder_spur"
            else:
                branch_records[branch_index]["role"] = "suppressed_ladder_rail"
        zone_record.update({
            "preserved_branch_count": int(preserved_branch_count),
            "collapsed": True,
        })
        zone_records.append(zone_record)

    normalized_chains, normalized_lengths, normalized_summary = _skeleton_chain_summary(
        normalized, minimum_length
    )
    raw_chain_labels = np.zeros(raw.shape, dtype=np.int32)
    for chain_id, path in enumerate(raw_chains, start=1):
        raw_chain_labels[path[:, 0], path[:, 1]] = chain_id
    rescued_count = 0
    rescued_length = 0.0
    for path, length in zip(normalized_chains, normalized_lengths):
        if length < minimum_length:
            continue
        if not np.any(collapsed_zones[path[:, 0], path[:, 1]]):
            continue
        raw_ids = set(raw_chain_labels[path[:, 0], path[:, 1]].tolist()) - {0}
        short_ids = [chain_id for chain_id in raw_ids if raw_lengths[chain_id - 1] < minimum_length]
        if len(short_ids) >= 2:
            rescued_count += 1
            rescued_length += length
    normalized_adjacency = _skeleton_adjacency(normalized)
    normalized_junction_count = sum(len(items) >= 3 for items in normalized_adjacency.values())
    diagnostics = {
        "raw_junction_pixel_count": int(np.count_nonzero(junction_mask)),
        "raw_junction_zone_count": int(len(zones)),
        "raw_chain_count": raw_summary["chain_count"],
        "raw_short_chain_count": raw_summary["short_chain_count"],
        "raw_median_chain_length": raw_summary["median_chain_length"],
        "raw_max_chain_length": raw_summary["max_chain_length"],
        "normalized_junction_count": int(normalized_junction_count),
        "normalized_chain_count": normalized_summary["chain_count"],
        "normalized_short_chain_count": normalized_summary["short_chain_count"],
        "normalized_median_chain_length": normalized_summary["median_chain_length"],
        "normalized_max_chain_length": normalized_summary["max_chain_length"],
        "pruned_spur_count": int(pruned_count),
        "collapsed_zone_count": int(collapsed_count),
        "junction_zone_radius_px": int(radius),
        "junction_cluster_radius_px": int(_junction_cluster_radius(config, distance_scale)),
        "structure_rescued_chain_count": int(rescued_count),
        "structure_rescued_length": float(rescued_length),
        "junction_zones": zone_records,
    }
    return {
        "normalized_skeleton": normalized.astype(np.uint8),
        "junction_pixel_mask": junction_mask.astype(np.uint8),
        "junction_zone_mask": (zone_labels > 0).astype(np.uint8),
        "junction_zone_labels": zone_labels,
        "pruned_spur_mask": pruned_spurs.astype(np.uint8),
        "collapsed_zone_mask": collapsed_zones.astype(np.uint8),
        "diagnostics": diagnostics,
    }


def _relative_chain_geometry(path):
    """Describe global directness and local smoothness of one skeleton chain."""
    points = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return {
            "path_length": 0.0,
            "direct_distance": 0.0,
            "tortuosity": float("inf"),
            "mean_turn_degrees": 180.0,
            "sharp_turn_fraction": 1.0,
            "reversal_fraction": 1.0,
            "locally_smooth": False,
        }
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    direct_distance = float(np.linalg.norm(points[-1] - points[0]))
    tortuosity = path_length / max(direct_distance, 1e-6)

    # Pixel skeletons contain harmless one-pixel stair steps.  Measure turns on
    # approximately four-pixel chords so smooth ramps/curves stay smooth while
    # zig-zag texture still produces repeated sharp changes and reversals.
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sample_positions = np.arange(0.0, path_length, 4.0)
    sampled = [points[min(int(np.searchsorted(cumulative, value)), len(points) - 1)] for value in sample_positions]
    sampled.append(points[-1])
    sampled = np.asarray(sampled, dtype=np.float32)
    vectors = np.diff(sampled, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    vectors = vectors[lengths > 1e-6]
    lengths = lengths[lengths > 1e-6]
    if len(vectors) < 2:
        turns = np.zeros((0,), dtype=np.float32)
    else:
        unit = vectors / lengths[:, None]
        cosine = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
        turns = np.degrees(np.arccos(cosine))
    mean_turn = float(np.mean(turns)) if len(turns) else 0.0
    sharp_fraction = float(np.mean(turns > 70.0)) if len(turns) else 0.0
    reversal_fraction = float(np.mean(turns > 125.0)) if len(turns) else 0.0
    locally_smooth = bool(
        mean_turn <= 35.0 and sharp_fraction <= 0.20 and reversal_fraction <= 0.05
    )
    return {
        "path_length": path_length,
        "direct_distance": direct_distance,
        "tortuosity": tortuosity,
        "mean_turn_degrees": mean_turn,
        "sharp_turn_fraction": sharp_fraction,
        "reversal_fraction": reversal_fraction,
        "locally_smooth": locally_smooth,
    }


def _scene_rank_map(probability, bins=4096):
    """Return a calibration-invariant empirical rank for every scene pixel."""
    values = _probability01(probability)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    bin_count = max(32, int(bins))
    indices = np.minimum(
        np.floor(values * (bin_count - 1)).astype(np.int32), bin_count - 1
    )
    histogram = np.bincount(indices.ravel(), minlength=bin_count)
    less_than = np.concatenate(([0], np.cumsum(histogram[:-1], dtype=np.int64)))
    # Mid-ranks keep a constant-valued road above a constant-valued background,
    # without making the result depend on the probability calibration scale.
    ranks = (less_than[indices] + 0.5 * histogram[indices]) / float(values.size)
    return ranks.astype(np.float32)


def _relative_component_elongation(points_rc):
    points = np.asarray(points_rc, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return 1.0
    centered = points - np.mean(points, axis=0, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / max(len(points) - 1, 1))
    return float(math.sqrt(max(float(eigenvalues[-1]), 1e-6) / max(float(eigenvalues[0]), 1e-6)))


def build_relative_candidate_mask(relative_score, config, *, scene_state="normal"):
    """Select relative candidates with scene-adaptive rank thresholds and hysteresis."""
    score = np.asarray(relative_score, dtype=np.float32)
    positive = score[score > 0]
    if positive.size == 0:
        return np.zeros(score.shape, dtype=np.uint8), {
            "relative_weak_percentile": 0.0,
            "relative_strong_percentile": 0.0,
            "relative_weak_threshold": 0.0,
            "relative_strong_threshold": 0.0,
        }
    low_scene = str(scene_state) in {"low_confidence", "very_low_confidence"}
    weak_percentile = float(_config_value(
        config,
        "RELATIVE_ROADNESS_LOW_SCENE_WEAK_PERCENTILE" if low_scene else "RELATIVE_ROADNESS_NORMAL_WEAK_PERCENTILE",
        25.0 if low_scene else 45.0,
    ))
    strong_percentile = float(_config_value(
        config,
        "RELATIVE_ROADNESS_LOW_SCENE_STRONG_PERCENTILE" if low_scene else "RELATIVE_ROADNESS_NORMAL_STRONG_PERCENTILE",
        85.0 if low_scene else 95.0,
    ))
    weak_threshold = float(np.percentile(positive, np.clip(weak_percentile, 0.0, 100.0)))
    strong_threshold = float(np.percentile(positive, np.clip(strong_percentile, 0.0, 100.0)))
    weak = (score >= weak_threshold).astype(np.uint8)
    strong = score >= strong_threshold
    close_size = max(1, int(round(_config_value(config, "RELATIVE_ROADNESS_CLOSE_KERNEL", 3))))
    if close_size > 1:
        kernel = np.ones((close_size, close_size), dtype=np.uint8)
        weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(weak, 8)
    candidate = np.zeros(score.shape, dtype=np.uint8)
    min_pixels = max(2, int(round(_config_value(config, "RELATIVE_ROADNESS_MIN_COMPONENT_PIXELS", 12))))
    # Hysteresis keeps strong-seeded components. A seedless component can only
    # survive when it already has clear line geometry; this is what allows a
    # uniformly weak but continuous road to coexist with a much stronger road.
    min_elongation = float(_config_value(config, "RELATIVE_ROADNESS_MIN_ELONGATION", 3.0))
    for component_id in range(1, component_count):
        component = labels == component_id
        if int(stats[component_id, cv2.CC_STAT_AREA]) < min_pixels:
            continue
        points = np.column_stack(np.where(component))
        elongated = _relative_component_elongation(points) >= min_elongation
        if np.any(strong & component) or elongated:
            candidate[component] = 1
    return candidate, {
        "relative_weak_percentile": weak_percentile,
        "relative_strong_percentile": strong_percentile,
        "relative_weak_threshold": weak_threshold,
        "relative_strong_threshold": strong_threshold,
    }


def extract_relative_skeleton(
    candidate_mask,
    config,
    *,
    distance_scale=1.0,
    relative_score=None,
    scene_rank=None,
    relative_weak_threshold=0.0,
    input_skeleton=None,
):
    """Retain valid chains independently, without rejecting a whole road component."""
    candidate = np.asarray(candidate_mask, dtype=np.uint8) > 0
    component_count, labels = cv2.connectedComponents(candidate.astype(np.uint8), 8)
    retained = np.zeros(candidate.shape, dtype=bool)
    rejected = np.zeros(candidate.shape, dtype=bool)
    scale = max(float(distance_scale), 1e-6)
    min_length = float(_config_value(config, "RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX", 48.0)) * scale
    max_tortuosity = float(_config_value(config, "RELATIVE_ROADNESS_MAX_TORTUOSITY", 1.5))
    retained_components = 0
    rejected_components = 0
    skeleton_before = np.zeros(candidate.shape, dtype=bool)
    chain_count = 0
    geometry_pass_count = 0
    reject_counts = Counter()
    score_map = np.asarray(relative_score, dtype=np.float32) if relative_score is not None else None
    rank_map = np.asarray(scene_rank, dtype=np.float32) if scene_rank is not None else None
    supplied_skeleton = (
        np.asarray(input_skeleton, dtype=bool)
        if input_skeleton is not None and np.asarray(input_skeleton).shape == candidate.shape
        else None
    )
    for component_id in range(1, component_count):
        component = labels == component_id
        skeleton = (
            supplied_skeleton & component
            if supplied_skeleton is not None
            else skeletonize(component)
        )
        skeleton_before |= skeleton
        chains = _trace_skeleton_chains(skeleton)
        component_retained = False
        for path in chains:
            chain_count += 1
            geometry = _relative_chain_geometry(path)
            if geometry["path_length"] < min_length:
                reject_counts["too_short"] += 1
                rejected[path[:, 0], path[:, 1]] = True
                continue
            if score_map is not None and score_map.shape == candidate.shape:
                chain_scores = score_map[path[:, 0], path[:, 1]]
                chain_q25 = float(np.quantile(chain_scores, 0.25))
                chain_ranks = (
                    rank_map[path[:, 0], path[:, 1]]
                    if rank_map is not None and rank_map.shape == candidate.shape
                    else np.ones(len(path), dtype=np.float32)
                )
                # Closing may bridge across weak texture. A valid chain must
                # keep relative evidence along most of its own centerline.
                if (
                    chain_q25 < max(float(relative_weak_threshold) - 1e-6, 0.0)
                    or float(np.quantile(chain_ranks, 0.25)) < 0.50
                ):
                    reject_counts["relative_structure_unsupported"] += 1
                    rejected[path[:, 0], path[:, 1]] = True
                    continue
            # Direct chains retain the historical check. Long chains may be
            # globally curved (ramps, mountain roads, rings) when their local
            # tangent changes remain smooth and never fold back abruptly.
            geometry_ok = bool(
                geometry["tortuosity"] <= max_tortuosity
                or (
                    geometry["path_length"] >= 1.5 * min_length
                    and geometry["locally_smooth"]
                )
            )
            if not geometry_ok:
                reject_counts["tortuosity"] += 1
                rejected[path[:, 0], path[:, 1]] = True
                continue
            retained[path[:, 0], path[:, 1]] = True
            component_retained = True
            geometry_pass_count += 1
        if component_retained:
            retained_components += 1
        else:
            rejected_components += 1
    return retained.astype(np.uint8), rejected.astype(np.uint8), {
        "relative_component_count": max(0, int(component_count - 1)),
        "relative_retained_component_count": int(retained_components),
        "relative_rejected_component_count": int(rejected_components),
        "relative_skeleton_before_structure_filter": int(np.count_nonzero(skeleton_before)),
        "relative_skeleton_after_structure_filter": int(np.count_nonzero(retained)),
        "relative_chain_count": int(chain_count),
        "relative_chain_geometry_pass": int(geometry_pass_count),
        "relative_structure_reject_reason_counts": dict(sorted(reject_counts.items())),
        "relative_skeleton_total_length": int(np.count_nonzero(retained)),
    }


def compute_relative_roadness(
    road_probability,
    config,
    *,
    scene_state="normal",
    distance_scale=1.0,
):
    """Compute full-scene roadness from scene rank and local background contrast.

    All statistics are computed on the supplied (unpadded) image. The returned
    arrays have the same shape and never replace or rewrite road_probability.
    """
    road = _probability01(road_probability)
    enabled = bool(_config_value(config, "RELATIVE_ROADNESS_ENABLED", False))
    empty = np.zeros(road.shape, dtype=np.float32)
    if not enabled or road.size == 0:
        return {
            "relative_score": empty,
            "scene_rank": empty.copy(),
            "local_background": empty.copy(),
            "local_contrast": empty.copy(),
            "normalized_contrast": empty.copy(),
            "scale_support_count": np.zeros(road.shape, dtype=np.uint8),
            "scale_agreement_fraction": empty.copy(),
            "relative_candidate_mask": np.zeros(road.shape, dtype=np.uint8),
            "relative_skeleton_raw": np.zeros(road.shape, dtype=np.uint8),
            "relative_skeleton_normalized": np.zeros(road.shape, dtype=np.uint8),
            "relative_skeleton": np.zeros(road.shape, dtype=np.uint8),
            "relative_rejected_skeleton": np.zeros(road.shape, dtype=np.uint8),
            "junction_zone_mask": np.zeros(road.shape, dtype=np.uint8),
            "pruned_spur_mask": np.zeros(road.shape, dtype=np.uint8),
            "collapsed_zone_mask": np.zeros(road.shape, dtype=np.uint8),
            "absolute_skeleton": np.zeros(road.shape, dtype=np.uint8),
            "relative_only_skeleton": np.zeros(road.shape, dtype=np.uint8),
            "combined_skeleton": np.zeros(road.shape, dtype=np.uint8),
            "diagnostics": {"relative_roadness_enabled": enabled, "relative_skeleton_total_length": 0},
        }
    configured_scales = _config_value(config, "RELATIVE_ROADNESS_BACKGROUND_SCALES_PX", [9, 21, 41])
    if not isinstance(configured_scales, (list, tuple)):
        configured_scales = [configured_scales]
    backgrounds = []
    used_scales = []
    for value in configured_scales:
        size = max(3, int(round(float(value) * max(float(distance_scale), 1e-6))))
        size += 1 - size % 2
        size = min(size, max(3, min(road.shape) // 2 * 2 - 1))
        if size < 3:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        backgrounds.append(cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REFLECT_101))
        used_scales.append(size)
    local_background = np.minimum.reduce(backgrounds) if backgrounds else np.zeros_like(road)
    per_scale_contrast = [np.maximum(road - background, 0.0) for background in backgrounds]
    if per_scale_contrast:
        scale_support_count = np.sum(
            np.stack([value > 1e-8 for value in per_scale_contrast], axis=0), axis=0
        ).astype(np.uint8)
        scale_agreement = scale_support_count.astype(np.float32) / float(len(per_scale_contrast))
    else:
        scale_support_count = np.zeros(road.shape, dtype=np.uint8)
        scale_agreement = np.zeros(road.shape, dtype=np.float32)
    local_contrast = np.maximum(road - local_background, 0.0)
    positive_contrast = local_contrast[local_contrast > 0]
    contrast_scale = float(np.percentile(positive_contrast, 90.0)) if positive_contrast.size else 0.0
    if contrast_scale <= 1e-8:
        normalized_contrast = np.zeros_like(road)
    else:
        normalized_contrast = np.clip(local_contrast / contrast_scale, 0.0, 1.0)
    scene_rank = _scene_rank_map(road)
    relative_score = np.sqrt(np.clip(scene_rank * normalized_contrast, 0.0, 1.0)).astype(np.float32)
    candidate, threshold_summary = build_relative_candidate_mask(
        relative_score, config, scene_state=scene_state
    )
    raw_relative_skeleton = skeletonize(candidate.astype(bool)).astype(np.uint8)
    normalization = normalize_relative_skeleton(
        raw_relative_skeleton,
        candidate,
        config,
        relative_score=relative_score,
        scale_agreement=scale_agreement,
        distance_scale=distance_scale,
    )
    normalized_relative_skeleton = normalization["normalized_skeleton"]
    relative_skeleton, relative_rejected_skeleton, structure_summary = extract_relative_skeleton(
        candidate,
        config,
        distance_scale=distance_scale,
        relative_score=relative_score,
        scene_rank=scene_rank,
        relative_weak_threshold=threshold_summary.get("relative_weak_threshold", 0.0),
        input_skeleton=normalized_relative_skeleton,
    )
    high_threshold, _low_threshold, profile_name = resolve_road_thresholds(config)
    close_size = max(1, int(round(_config_value(config, "WEAK_BOOTSTRAP_CLOSE_KERNEL", 3))))
    absolute_skeleton = skeletonize_road_mask(
        (road * 255.0).astype(np.uint8), high_threshold * 255.0, close_size
    ) > 0
    suppression_radius = max(0, int(round(
        float(_config_value(config, "RELATIVE_ROADNESS_ABSOLUTE_SUPPRESSION_PX", 3.0))
        * max(float(distance_scale), 1e-6)
    )))
    absolute_neighborhood = absolute_skeleton
    if suppression_radius > 0 and np.any(absolute_skeleton):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (suppression_radius * 2 + 1, suppression_radius * 2 + 1),
        )
        absolute_neighborhood = cv2.dilate(absolute_skeleton.astype(np.uint8), kernel) > 0
    relative_only = (relative_skeleton > 0) & ~absolute_neighborhood
    combined = absolute_skeleton | relative_only
    diagnostics = {
        "relative_roadness_enabled": True,
        "relative_scene_state": str(scene_state),
        "relative_threshold_profile": profile_name,
        "relative_background_scales_px": used_scales,
        "relative_contrast_scale": contrast_scale,
        "relative_score_p50": float(np.percentile(relative_score, 50.0)),
        "relative_score_p90": float(np.percentile(relative_score, 90.0)),
        "relative_score_p95": float(np.percentile(relative_score, 95.0)),
        "relative_score_p99": float(np.percentile(relative_score, 99.0)),
        "relative_candidate_pixel_count": int(np.count_nonzero(candidate)),
        "absolute_skeleton_total_length": int(np.count_nonzero(absolute_skeleton)),
        "relative_only_skeleton_total_length": int(np.count_nonzero(relative_only)),
        "combined_skeleton_total_length": int(np.count_nonzero(combined)),
        **threshold_summary,
        **normalization["diagnostics"],
        **structure_summary,
    }
    return {
        "relative_score": relative_score,
        "scene_rank": scene_rank,
        "local_background": local_background.astype(np.float32),
        "local_contrast": local_contrast.astype(np.float32),
        "normalized_contrast": normalized_contrast.astype(np.float32),
        "scale_support_count": scale_support_count,
        "scale_agreement_fraction": scale_agreement.astype(np.float32),
        "relative_candidate_mask": candidate.astype(np.uint8),
        "relative_skeleton_raw": raw_relative_skeleton.astype(np.uint8),
        "relative_skeleton_normalized": normalized_relative_skeleton.astype(np.uint8),
        "relative_skeleton": relative_skeleton.astype(np.uint8),
        "relative_rejected_skeleton": relative_rejected_skeleton.astype(np.uint8),
        "junction_zone_mask": normalization["junction_zone_mask"].astype(np.uint8),
        "pruned_spur_mask": normalization["pruned_spur_mask"].astype(np.uint8),
        "collapsed_zone_mask": normalization["collapsed_zone_mask"].astype(np.uint8),
        "absolute_skeleton": absolute_skeleton.astype(np.uint8),
        "relative_only_skeleton": relative_only.astype(np.uint8),
        "combined_skeleton": combined.astype(np.uint8),
        "diagnostics": diagnostics,
    }


def embed_relative_roadness_context(context, shape):
    """Embed an unpadded relative-roadness context into a padded image shape."""
    height, width = map(int, shape)
    result = {"diagnostics": dict(context.get("diagnostics", {}))}
    for name, value in context.items():
        if name == "diagnostics":
            continue
        array = np.asarray(value)
        canvas = np.zeros((height, width), dtype=array.dtype)
        copy_height = min(height, array.shape[0])
        copy_width = min(width, array.shape[1])
        canvas[:copy_height, :copy_width] = array[:copy_height, :copy_width]
        result[name] = canvas
    result["valid_shape"] = tuple(context.get("relative_score", np.zeros((0, 0))).shape)
    return result


def diagnose_probability_profile(road_probability, config, *, distance_scale=1.0):
    """Diagnose one complete probability image using a fixed reference profile."""
    road = _probability01(road_probability)
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
        # A scene without any usable LOW structure is not recoverable merely by
        # lowering thresholds; keep the standard profile to avoid background
        # hallucination.
        "recommended_profile": "weak_sensor" if state == "low_confidence" else "default",
        "reference_profile": reference_profile,
        "diagnostic_reference_profile": reference_profile,
        "reference_high_threshold": reference_high,
        "reference_low_threshold": reference_low,
        "probability_p50": float(percentiles[0]),
        "probability_p90": float(percentiles[1]),
        "probability_p95": float(percentiles[2]),
        "probability_p99": float(percentiles[3]),
        "high_pixel_ratio": high_ratio,
        "low_pixel_ratio": low_ratio,
        "reference_strong_skeleton_total_length": reference_strong_length,
        "weak_skeleton_total_length": weak_length,
        "relative_high": relative_high,
        "relative_strong": relative_strong,
        "has_low_structure": bool(has_low_structure),
    }


def resolve_effective_road_profile(
    road_probability,
    config,
    *,
    distance_scale=1.0,
    requested_profile=None,
):
    """Resolve ``auto`` to a real threshold profile for one complete image."""
    requested = str(
        requested_profile
        if requested_profile is not None
        else _config_value(config, "ROAD_THRESHOLD_PROFILE", "default")
    )
    if requested not in {"auto", "default", "weak_sensor"}:
        raise ValueError(
            "ROAD_THRESHOLD_PROFILE must be one of auto, default, weak_sensor; "
            f"got {requested!r}."
        )
    diagnosis = diagnose_probability_profile(
        road_probability, config, distance_scale=distance_scale
    )
    effective = diagnosis["recommended_profile"] if requested == "auto" else requested
    high, low, effective = resolve_road_thresholds(config, profile_name=effective)
    return {
        **diagnosis,
        "requested_profile": requested,
        "effective_profile": effective,
        "profile_selection_mode": "automatic" if requested == "auto" else "manual",
        "road_high_threshold": high,
        "road_low_threshold": low,
    }


def summarize_profile_decisions(requested_profile, reference_profile, decisions):
    """Build auditable per-batch profile metadata from per-image decisions."""
    rows = [dict(row) for row in decisions]
    default_count = sum(row.get("effective_profile") == "default" for row in rows)
    weak_count = sum(row.get("effective_profile") == "weak_sensor" for row in rows)
    requested = str(requested_profile)
    return {
        "requested_profile": requested,
        "profile_selection_mode": "automatic" if requested == "auto" else "manual",
        "diagnostic_reference_profile": str(reference_profile),
        "image_count": len(rows),
        "default_image_count": int(default_count),
        "weak_sensor_image_count": int(weak_count),
        "mixed_profile": bool(default_count and weak_count),
        "decisions": rows,
    }


def diagnose_scene_confidence(
    road_probability,
    nodes_rc,
    edges,
    config,
    *,
    distance_scale=1.0,
):
    """Add graph QA fields to the shared probability-only diagnosis."""
    diagnosis = diagnose_probability_profile(
        road_probability, config, distance_scale=distance_scale
    )
    active_high, active_low, active_profile = resolve_road_thresholds(config)
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    graph_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    active_graph_length = float(sum(
        np.linalg.norm(nodes[int(dst_idx)] - nodes[int(src_idx)])
        for src_idx, dst_idx in graph_edges.tolist()
    ))
    return {
        **diagnosis,
        "threshold_profile": active_profile,
        "active_profile": active_profile,
        "active_high_threshold": active_high,
        "active_low_threshold": active_low,
        "strong_graph_edge_count": int(len(graph_edges)),
        "strong_graph_total_length": active_graph_length,
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
    relative_context=None,
    include_absolute_candidates=True,
    topology_candidate_nodes_rc=None,
    topology_candidate_edges=None,
    topology_candidate_scores=None,
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
        relative_evidence = _relative_path_evidence(
            np.column_stack((rr, cc)).astype(np.int32), relative_context
        )
        relative_fraction = relative_evidence["relative_only_fraction"]
        candidate_source = (
            "absolute+relative" if 0.05 < relative_fraction < 0.95
            else "relative" if relative_fraction >= 0.95
            else "absolute"
        )
        topology_probability = float(scores[edge_id]) if np.isfinite(scores[edge_id]) else center_conf
        metadata.append({
            "line_source": "relative_roadness" if candidate_source == "relative" else "samroad",
            "candidate_source": candidate_source,
            "topology_probability": topology_probability,
            "recovery_score": 0.0, "center_conf": center_conf,
            "background_conf": 0.0, "probability_contrast": center_conf,
            "surface_conf": 0.0, "recovery_reason": "strong_threshold",
            "qa_state": "auto", "recovery_id": "",
            **relative_evidence,
        })
    relative_topology_selected = sum(
        row.get("candidate_source") in {"relative", "absolute+relative"}
        for row in metadata
    )
    relative_topology_candidates = 0
    candidate_nodes_array = np.asarray(
        topology_candidate_nodes_rc if topology_candidate_nodes_rc is not None else np.empty((0, 2)),
        dtype=np.float32,
    ).reshape(-1, 2)
    for src_idx, dst_idx in np.asarray(
        topology_candidate_edges if topology_candidate_edges is not None else np.empty((0, 2)),
        dtype=np.int32,
    ).reshape(-1, 2).tolist():
        if not (0 <= src_idx < len(candidate_nodes_array) and 0 <= dst_idx < len(candidate_nodes_array)):
            continue
        rr, cc = line(
            int(round(candidate_nodes_array[src_idx, 0])), int(round(candidate_nodes_array[src_idx, 1])),
            int(round(candidate_nodes_array[dst_idx, 0])), int(round(candidate_nodes_array[dst_idx, 1])),
        )
        rr = np.clip(rr, 0, road.shape[0] - 1)
        cc = np.clip(cc, 0, road.shape[1] - 1)
        if _relative_path_evidence(
            np.column_stack((rr, cc)).astype(np.int32), relative_context
        )["relative_fraction"] >= 0.25:
            relative_topology_candidates += 1
    summary = {
        "threshold_profile": profile_name,
        "bootstrap_candidate_count": 0,
        "bootstrap_accepted_candidate_count": 0,
        "bootstrap_recovered_edge_count": 0,
        "bootstrap_auto_count": 0,
        "bootstrap_review_count": 0,
        "bootstrap_rejected_count": 0,
        "bootstrap_reject_reason_counts": {},
        "relative_candidate_count": 0,
        "relative_accepted_candidate_count": 0,
        "relative_recovered_edge_count": 0,
        "relative_auto_count": 0,
        "relative_review_count": 0,
        "relative_rejected_count": 0,
        "relative_reject_reason_counts": {},
        "relative_topology_candidate_edge_count": int(relative_topology_candidates),
        "relative_topology_selected_edge_count": int(relative_topology_selected),
        "relative_graph_point_count": 0,
    }
    if not bool(_config_value(config, "WEAK_BOOTSTRAP_ENABLED", True)):
        return nodes, original_edges, metadata, summary

    close_size = max(1, int(round(_config_value(config, "WEAK_BOOTSTRAP_CLOSE_KERNEL", 3))))
    absolute_low_mask = (
        (road >= low_threshold).astype(np.uint8)
        if include_absolute_candidates
        else np.zeros(road.shape, dtype=np.uint8)
    )
    relative_skeleton = np.zeros(road.shape, dtype=bool)
    if relative_context is not None:
        # Acceptance uses the calibration-invariant structured skeleton. Actual
        # graph overlap is handled below; suppressing by raw HIGH probability
        # here would make otherwise identical Case A/B/C/D roads diverge.
        skeleton_value = np.asarray(relative_context.get("relative_skeleton", []))
        if skeleton_value.shape == road.shape:
            relative_skeleton = skeleton_value > 0
    if close_size > 1:
        kernel = np.ones((close_size, close_size), dtype=np.uint8)
        absolute_low_mask = cv2.morphologyEx(absolute_low_mask, cv2.MORPH_CLOSE, kernel)
    absolute_weak_skeleton = skeletonize(absolute_low_mask.astype(bool))
    strong_mask = _graph_raster_mask(nodes, original_edges, road.shape)
    if parameters["suppression_radius"] > 0 and np.any(strong_mask):
        radius = parameters["suppression_radius"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        suppressed = cv2.dilate(strong_mask, kernel) > 0
        absolute_weak_skeleton &= ~suppressed
    if np.any(strong_mask):
        strong_distance = cv2.distanceTransform((strong_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        strong_distance = np.full(road.shape, np.inf, dtype=np.float32)
    # Keep the two sources separate. Raster-unioning the broad Relative mask
    # with the LOW mask used to fragment valid roads into thousands of tiny
    # chains at texture intersections before acceptance was even evaluated.
    chains = [
        (path, False) for path in _trace_skeleton_chains(absolute_weak_skeleton)
    ] + [
        (path, True) for path in _trace_skeleton_chains(relative_skeleton)
    ]
    topology_mask, topology_score_map = _topology_candidate_rasters(
        topology_candidate_nodes_rc if topology_candidate_nodes_rc is not None else np.empty((0, 2)),
        topology_candidate_edges if topology_candidate_edges is not None else np.empty((0, 2)),
        topology_candidate_scores if topology_candidate_scores is not None else np.empty((0,)),
        road.shape,
        radius=max(1, int(round(3.0 * scale))),
    )
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
    relative_reject_counts = Counter()

    def reject_candidate(row, reason):
        row["accepted"] = False
        row["decision"] = "rejected"
        row["qa_state"] = "rejected"
        row["reject_reason"] = reason
        row["review_reason"] = ""
        summary["bootstrap_rejected_count"] += 1
        reject_counts[reason] += 1
        if row.get("candidate_source") in {"relative", "absolute+relative"}:
            relative_reject_counts[reason] += 1
        if candidate_audit is not None:
            candidate_audit.append(row)

    for path, direct_relative_chain in chains:
        summary["bootstrap_candidate_count"] += 1
        path_float = path.astype(np.float32)
        geometry = _relative_chain_geometry(path)
        path_length = geometry["path_length"]
        direct_distance = geometry["direct_distance"]
        tortuosity = geometry["tortuosity"]
        if direct_relative_chain:
            summary["relative_graph_point_count"] += max(
                2, int(math.ceil(path_length / parameters["sample_step"])) + 1
            )
        relative_evidence = _relative_path_evidence(path, relative_context)
        relative_fraction = relative_evidence["relative_only_fraction"]
        absolute_fraction = float(np.mean(
            road[
                np.clip(path[:, 0], 0, road.shape[0] - 1),
                np.clip(path[:, 1], 0, road.shape[1] - 1),
            ] >= low_threshold
        ))
        candidate_source = "relative" if direct_relative_chain else (
            "absolute+relative" if relative_fraction >= 0.25 and absolute_fraction >= 0.25
            else "relative" if relative_fraction >= 0.50
            else "absolute"
        )
        relative_branch_candidate = candidate_source != "absolute"
        if relative_branch_candidate:
            summary["relative_candidate_count"] += 1
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
            "endpoint_alignment": 0.0,
            "absolute_proximity_fraction": 0.0,
            "topology_candidate_support_fraction": 0.0,
            "topology_candidate_score_mean": 0.0,
            "mean_turn_degrees": geometry["mean_turn_degrees"],
            "sharp_turn_fraction": geometry["sharp_turn_fraction"],
            "reversal_fraction": geometry["reversal_fraction"],
            "locally_smooth": geometry["locally_smooth"],
            "accepted": False,
            "decision": "rejected",
            "qa_state": "rejected",
            "reject_reason": "",
            "review_reason": "",
            "candidate_source": candidate_source,
            "path": path.astype(int).tolist() if relative_branch_candidate else [],
            **{key: value for key, value in relative_evidence.items() if key != "relative_supported"},
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
        path_rows = np.clip(path[:, 0], 0, road.shape[0] - 1)
        path_cols = np.clip(path[:, 1], 0, road.shape[1] - 1)
        topology_supported_pixels = topology_mask[path_rows, path_cols] > 0
        topology_support_fraction = float(np.mean(topology_supported_pixels))
        topology_score_mean = float(np.mean(topology_score_map[path_rows, path_cols]))
        endpoint_alignment = _endpoint_alignment(
            path, nodes, original_edges, parameters["connection_radius"]
        )
        absolute_proximity_fraction = float(np.mean(
            strong_distance[path_rows, path_cols] <= parameters["connection_radius"]
        ))
        audit_row.update({
            "mean_probability": evidence["center_conf"],
            "q25_probability": evidence["center_q25"],
            "weak_fraction": evidence["weak_fraction"],
            "background_probability": evidence["background_conf"],
            "background_contrast": evidence["probability_contrast"],
            "connection_count": connection_count,
            "endpoint_alignment": endpoint_alignment,
            "absolute_proximity_fraction": absolute_proximity_fraction,
            "topology_candidate_support_fraction": topology_support_fraction,
            "topology_candidate_score_mean": topology_score_mean,
            "candidate_source": candidate_source,
            **{key: value for key, value in relative_evidence.items() if key != "relative_supported"},
        })
        if direct_relative_chain and len(original_edges) and absolute_proximity_fraction >= 0.80:
            reject_candidate(audit_row, "duplicate_or_suppressed")
            continue
        recovery_gap_limit = float(_config_value(config, "WEAK_RECOVERY_MAX_GAP_PX", 64.0)) * scale
        delegated_gap = connection_count == 2 and path_length <= recovery_gap_limit
        geometry_supported = bool(
            tortuosity <= parameters["max_tortuosity"]
            or (relative_branch_candidate and geometry["locally_smooth"])
        )
        relative_supported = bool(relative_evidence["relative_supported"])
        independent_supported = (
            connection_count > 0
            or evidence["surface_supported"]
            or relative_supported
            or (
                path_length >= parameters["min_length"] * parameters["independent_length_factor"]
                and evidence["center_conf"] >= parameters["min_mean"] + 0.01
                and evidence["probability_contrast"] >= parameters["min_contrast"] + 0.02
            )
        )
        evidence_supported = (
            evidence["road_supported"]
            or evidence["surface_supported"]
            or relative_supported
        )
        if delegated_gap:
            reject_candidate(audit_row, "delegated_to_weak_recovery")
            continue
        if not geometry_supported:
            reject_candidate(audit_row, "tortuosity")
            continue
        if not evidence_supported:
            reject_candidate(
                audit_row,
                "relative_structure_unsupported"
                if candidate_source == "relative"
                else _road_evidence_reject_reason(evidence, parameters),
            )
            continue
        if not independent_supported:
            reject_candidate(audit_row, "insufficient_independent_support")
            continue
        directness = min(1.0, 1.0 / max(tortuosity, 1.0))
        proximity = (0.5, 0.8, 1.0)[connection_count]
        if relative_supported:
            recovery_score = (
                0.28 * relative_evidence["relative_score_mean"]
                + 0.18 * relative_evidence["relative_score_q25"]
                + 0.16 * relative_evidence["scene_rank_mean"]
                + 0.12 * relative_evidence["normalized_contrast_mean"]
                + 0.12 * min(1.0, path_length / max(parameters["min_length"] * 2.0, 1e-6))
                + 0.08 * directness
                + 0.06 * proximity
            )
        else:
            recovery_score = (
                0.24 * min(1.0, evidence["center_conf"] / max(high_threshold, 1e-6))
                + 0.18 * min(1.0, evidence["center_q25"] / max(low_threshold, 1e-6))
                + 0.22 * min(1.0, evidence["probability_contrast"] / max(parameters["min_contrast"] * 2.0, 1e-6))
                + 0.14 * min(1.0, path_length / max(parameters["min_length"] * 2.0, 1e-6))
                + 0.10 * directness
                + 0.06 * proximity
                + 0.06 * evidence["surface_conf"]
            )
        if relative_branch_candidate:
            diagnostics = relative_context.get("diagnostics", {}) if relative_context else {}
            relative_weak_threshold = float(diagnostics.get("relative_weak_threshold", 0.0))
            stable_relative = bool(
                relative_supported
                and relative_evidence["relative_fraction"] >= 0.75
                and relative_evidence["relative_score_q25"] >= max(relative_weak_threshold - 1e-6, 0.0)
                and relative_evidence["scene_rank_mean"] >= 0.75
                and relative_evidence["normalized_contrast_mean"] > 0.0
            )
            multiscale_supported = bool(
                relative_evidence["scale_agreement_q25"] >= (2.0 / 3.0)
            )
            topology_supported = bool(topology_support_fraction >= 0.25)
            graph_connected = bool(
                connection_count > 0
                and (endpoint_alignment >= 0.50 or absolute_proximity_fraction >= 0.10)
            )
            long_independent = bool(
                path_length >= parameters["min_length"] * parameters["independent_length_factor"]
                and geometry["locally_smooth"]
                and multiscale_supported
            )
            supporting_evidence = graph_connected or topology_supported or multiscale_supported
            if stable_relative and geometry_supported and supporting_evidence and (
                graph_connected or topology_supported or long_independent
            ):
                qa_state = "auto"
                relative_tier = "A" if (graph_connected or topology_supported) else "B"
            elif relative_supported and geometry_supported:
                qa_state = "review"
                relative_tier = "C"
            else:
                reject_candidate(audit_row, "relative_structure_unsupported")
                continue
            audit_row.update({
                "relative_evidence_tier": relative_tier,
                "multiscale_supported": multiscale_supported,
                "topology_candidate_supported": topology_supported,
                "graph_connection_supported": graph_connected,
            })
            if qa_state == "review":
                summary["bootstrap_review_count"] += 1
                summary["relative_review_count"] += 1
                audit_row["accepted"] = False
                audit_row["decision"] = "review"
                audit_row["qa_state"] = "review"
                audit_row["reject_reason"] = ""
                audit_row["review_reason"] = "relative_evidence_requires_review"
                audit_row["path"] = path.astype(int).tolist()
                if candidate_audit is not None:
                    candidate_audit.append(audit_row)
                continue
        else:
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
                "line_source": "relative_roadness" if relative_branch_candidate else "weak_bootstrap",
                "candidate_source": candidate_source,
                "topology_probability": float(
                    topology_score_mean if relative_branch_candidate else recovery_score
                ),
                "recovery_score": float(recovery_score),
                "center_conf": float(evidence["center_conf"]),
                "background_conf": float(evidence["background_conf"]),
                "probability_contrast": float(evidence["probability_contrast"]),
                "surface_conf": float(evidence["surface_conf"]),
                "recovery_reason": "weak_network_bootstrap",
                "qa_state": qa_state,
                "decision": qa_state,
                "recovery_id": f"bootstrap:{recovery_id}",
                "relative_evidence_tier": audit_row.get("relative_evidence_tier", ""),
                "connection_count": connection_count,
                "endpoint_alignment": endpoint_alignment,
                "absolute_proximity_fraction": absolute_proximity_fraction,
                "topology_candidate_support_fraction": topology_support_fraction,
                "topology_candidate_score_mean": topology_score_mean,
                **relative_evidence,
            })
            added_count += 1
        if added_count:
            recovery_id += 1
            summary["bootstrap_accepted_candidate_count"] += 1
            summary["bootstrap_recovered_edge_count"] += added_count
            summary[f"bootstrap_{qa_state}_count"] += 1
            if relative_branch_candidate:
                summary["relative_accepted_candidate_count"] += 1
                summary["relative_recovered_edge_count"] += added_count
                summary[f"relative_{qa_state}_count"] += 1
            audit_row["accepted"] = True
            audit_row["decision"] = qa_state
            audit_row["qa_state"] = qa_state
            audit_row["reject_reason"] = ""
            audit_row["review_reason"] = ""
            if candidate_audit is not None:
                candidate_audit.append(audit_row)
        else:
            reject_candidate(audit_row, "duplicate_or_suppressed")
    summary["bootstrap_reject_reason_counts"] = dict(sorted(reject_counts.items()))
    summary["relative_reject_reason_counts"] = dict(
        sorted(relative_reject_counts.items())
    )
    summary["relative_rejected_count"] = max(
        0,
        summary["relative_candidate_count"]
        - summary["relative_accepted_candidate_count"]
        - summary["relative_review_count"],
    )
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
        "weak_connectivity_gain_total": 0,
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
        "direction_lookback": max(
            1.0,
            float(_config_value(config, "WEAK_ENDPOINT_DIRECTION_LOOKBACK_PX", 32.0)) * scale,
        ),
    }
    endpoint_vectors = _endpoint_vectors(
        nodes, original_edges, lookback_distance=parameters["direction_lookback"]
    )
    endpoint_ids = sorted(endpoint_vectors)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(original_edges.tolist())
    component = {}
    for component_id, members in enumerate(nx.connected_components(graph)):
        for node_idx in members:
            component[node_idx] = component_id
    initial_component_count = int(nx.number_connected_components(graph))

    proposals = []
    reject_counts = Counter()

    def reject_candidate(row, reason):
        row["accepted"] = False
        row["reject_reason"] = reason
        if row.get("component_count_after") is None:
            row["component_count_after"] = row.get("component_count_before")
        row["connectivity_gain"] = 0
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
            "component_before_start": component.get(int(start_idx)),
            "component_before_target": (
                component.get(int(end_idx)) if end_idx is not None else None
            ),
            "merges_components": bool(
                end_idx is not None and component.get(int(start_idx)) != component.get(int(end_idx))
            ),
            "component_count_before": initial_component_count,
            "component_count_after": initial_component_count,
            "connectivity_gain": 0,
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
                        "component_before_start": component.get(int(start_idx)),
                        "component_before_target": None,
                        "merges_components": False,
                        "component_count_before": initial_component_count,
                        "component_count_after": initial_component_count,
                        "connectivity_gain": 0,
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
        before_stats = graph_connectivity_stats(combined_nodes, combined_edges)
        before_components = _component_labels(len(combined_nodes), combined_edges)
        start_component = before_components.get(int(proposal["start_idx"]))
        target_component = (
            before_components.get(int(proposal["end_idx"]))
            if proposal["end_idx"] is not None else None
        )
        proposal["audit_row"].update({
            "component_before_start": start_component,
            "component_before_target": target_component,
            "merges_components": bool(
                target_component is not None and start_component != target_component
            ),
            "component_count_before": before_stats["component_count"],
        })
        if proposal["end_idx"] is not None and start_component == target_component:
            reject_candidate(proposal["audit_row"], "same_component")
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
        added_metadata_ids = []
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
                "component_before_start": start_component,
                "component_before_target": target_component,
                "merges_components": bool(
                    target_component is not None and start_component != target_component
                ),
                "component_count_before": before_stats["component_count"],
                "component_count_after": None,
                "connectivity_gain": 0,
            })
            added_metadata_ids.append(len(metadata) - 1)
            added_count += 1
        if added_count:
            after_stats = graph_connectivity_stats(combined_nodes, combined_edges)
            connectivity_gain = max(
                0, before_stats["component_count"] - after_stats["component_count"]
            )
            proposal["audit_row"].update({
                "component_count_after": after_stats["component_count"],
                "connectivity_gain": connectivity_gain,
            })
            for metadata_id in added_metadata_ids:
                metadata[metadata_id]["component_count_after"] = after_stats["component_count"]
                metadata[metadata_id]["connectivity_gain"] = connectivity_gain
            used_endpoints.update(endpoint_members)
            recovery_id += 1
            summary["weak_recovered_candidate_count"] += 1
            reason_counts[proposal["reason"]] += added_count
            summary["weak_recovered_edge_count"] += added_count
            summary["weak_connectivity_gain_total"] += connectivity_gain
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


def recover_endpoint_to_segment_connections(
    nodes_rc,
    edges,
    road_probability,
    config,
    *,
    edge_metadata=None,
    surface_probability=None,
    distance_scale=1.0,
    candidate_audit=None,
):
    """Connect dangling endpoints to projected points on other graph components."""
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    edge_array = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    road = _probability01(road_probability)
    surface = None if surface_probability is None else _probability01(surface_probability)
    if surface is not None and surface.shape != road.shape:
        raise ValueError(f"Road/surface probability shape mismatch: {road.shape} != {surface.shape}")
    metadata = [dict(row) for row in (edge_metadata or [])]
    if not metadata:
        metadata = [{
            "line_source": "samroad", "topology_probability": 0.0,
            "recovery_score": 0.0, "center_conf": 0.0, "surface_conf": 0.0,
            "recovery_reason": "strong_threshold", "qa_state": "auto", "recovery_id": "",
        } for _ in range(len(edge_array))]
    if len(metadata) != len(edge_array):
        raise ValueError("edge_metadata must align with edges")

    summary = {
        "endpoint_segment_candidate_count": 0,
        "endpoint_segment_accepted_count": 0,
        "endpoint_segment_rejected_count": 0,
        "endpoint_segment_recovered_edge_count": 0,
        "endpoint_segment_split_edge_delta": 0,
        "endpoint_segment_connectivity_gain": 0,
        "endpoint_segment_reject_reason_counts": {},
    }
    enabled = bool(_config_value(config, "WEAK_SEGMENT_RECOVERY_ENABLED", False))
    if not enabled or len(nodes) == 0 or len(edge_array) == 0:
        return nodes, edge_array, metadata, summary

    _high_threshold, low_threshold, _profile_name = resolve_road_thresholds(config)
    scale = max(float(distance_scale), 1e-6)
    parameters = {
        "max_distance": float(
            _config_value(config, "WEAK_SEGMENT_RECOVERY_MAX_DISTANCE_PX", 64.0)
        ) * scale,
        "min_alignment": float(
            _config_value(config, "WEAK_SEGMENT_RECOVERY_MIN_DIRECTION_COSINE", 0.50)
        ),
        "direction_lookback": max(
            1.0,
            float(_config_value(config, "WEAK_ENDPOINT_DIRECTION_LOOKBACK_PX", 32.0)) * scale,
        ),
        "max_path_ratio": float(_config_value(config, "WEAK_RECOVERY_MAX_PATH_RATIO", 1.35)),
        "min_mean": float(_config_value(config, "WEAK_RECOVERY_MIN_MEAN_PROBABILITY", 0.20)),
        "min_q25": float(_config_value(config, "WEAK_RECOVERY_MIN_Q25_PROBABILITY", 0.17)),
        "min_weak_fraction": float(_config_value(config, "WEAK_RECOVERY_MIN_WEAK_FRACTION", 0.80)),
        "min_contrast": float(_config_value(config, "WEAK_RECOVERY_MIN_BACKGROUND_CONTRAST", 0.08)),
        "background_offset": float(
            _config_value(config, "WEAK_RECOVERY_BACKGROUND_OFFSET_PX", 4.0)
        ) * scale,
        "surface_threshold": float(_config_value(config, "WEAK_RECOVERY_SURFACE_THRESHOLD", 0.60)),
        "surface_min_center": float(
            _config_value(config, "WEAK_RECOVERY_SURFACE_MIN_CENTER_PROBABILITY", 0.10)
        ),
        "surface_min_mean": float(_config_value(config, "WEAK_RECOVERY_SURFACE_MIN_MEAN", 0.70)),
        "surface_min_fraction": float(
            _config_value(config, "WEAK_RECOVERY_SURFACE_MIN_FRACTION", 0.80)
        ),
        "path_margin": max(
            1.0, float(_config_value(config, "WEAK_RECOVERY_PATH_MARGIN_PX", 16.0)) * scale
        ),
        "sample_step": max(
            1.0, float(_config_value(config, "WEAK_RECOVERY_SAMPLE_STEP_PX", 12.0)) * scale
        ),
        "auto_score": float(_config_value(config, "WEAK_RECOVERY_AUTO_SCORE", 0.62)),
    }
    endpoint_vectors = _endpoint_vectors(
        nodes, edge_array, lookback_distance=parameters["direction_lookback"]
    )
    endpoint_ids = sorted(endpoint_vectors)
    components = _component_labels(len(nodes), edge_array)
    initial_stats = graph_connectivity_stats(nodes, edge_array)
    reject_counts = Counter()
    proposals = []
    candidate_sequence = 0

    segment_edge_ids = {}
    for edge_id, (src_idx, dst_idx) in enumerate(edge_array.tolist()):
        if int(src_idx) == int(dst_idx):
            continue
        segment_edge_ids.setdefault(tuple(sorted((int(src_idx), int(dst_idx)))), []).append(edge_id)
    segments = []
    sample_points = []
    sample_segment_ids = []
    spatial_step = max(4.0, min(16.0 * scale, parameters["max_distance"] * 0.5))
    for segment_id, (node_pair, directed_edge_ids) in enumerate(segment_edge_ids.items()):
        src_idx, dst_idx = node_pair
        src, dst = nodes[src_idx], nodes[dst_idx]
        length = float(np.linalg.norm(dst - src))
        if length <= 1e-6:
            continue
        segments.append({
            "segment_id": segment_id,
            "node_pair": node_pair,
            "edge_ids": directed_edge_ids,
            "length": length,
        })
        sample_count = max(1, int(math.ceil(length / spatial_step)))
        for sample_index in range(sample_count + 1):
            fraction = sample_index / sample_count
            sample_points.append(src + fraction * (dst - src))
            sample_segment_ids.append(len(segments) - 1)
    if not sample_points or not endpoint_ids:
        return nodes, edge_array, metadata, summary
    sample_tree = KDTree(np.asarray(sample_points, dtype=np.float32))

    def reject(row, reason):
        row["accepted"] = False
        row["reject_reason"] = reason
        row["connectivity_gain"] = 0
        summary["endpoint_segment_rejected_count"] += 1
        reject_counts[reason] += 1
        if candidate_audit is not None:
            candidate_audit.append(row)

    for endpoint_idx in endpoint_ids:
        nearby_sample_ids = sample_tree.query_radius(
            nodes[endpoint_idx][np.newaxis, :],
            r=parameters["max_distance"] + spatial_step,
        )[0]
        nearby_segment_ids = sorted({sample_segment_ids[index] for index in nearby_sample_ids})
        for segment_index in nearby_segment_ids:
            segment = segments[segment_index]
            src_idx, dst_idx = segment["node_pair"]
            src, dst = nodes[src_idx], nodes[dst_idx]
            tangent = dst - src
            length_squared = float(np.dot(tangent, tangent))
            projection_fraction = float(
                np.clip(np.dot(nodes[endpoint_idx] - src, tangent) / length_squared, 0.0, 1.0)
            )
            projection = src + projection_fraction * tangent
            delta = projection - nodes[endpoint_idx]
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-6 or distance > parameters["max_distance"]:
                continue
            candidate_sequence += 1
            summary["endpoint_segment_candidate_count"] += 1
            start_component = components.get(endpoint_idx)
            target_component = components.get(src_idx)
            row = {
                "candidate_id": f"segment_candidate:{candidate_sequence}",
                "endpoint_node": int(endpoint_idx),
                "target_segment": int(segment["edge_ids"][0]),
                "target_projection": projection.astype(float).tolist(),
                "distance": distance,
                "direction_cosine": None,
                "path_length": None,
                "path_ratio": None,
                "mean_probability": None,
                "q25_probability": None,
                "weak_fraction": None,
                "background_probability": None,
                "background_contrast": None,
                "start_component": start_component,
                "target_component": target_component,
                "connectivity_gain": 0,
                "accepted": False,
                "reject_reason": "",
                "recovery_score": None,
                "component_count_before": initial_stats["component_count"],
                "component_count_after": initial_stats["component_count"],
                "merges_components": bool(start_component != target_component),
                "_path": None,
                "_target_nodes": [src.astype(float).tolist(), dst.astype(float).tolist()],
            }
            if start_component == target_component:
                reject(row, "reject_same_component")
                continue
            endpoint_distance = min(
                float(np.linalg.norm(projection - src)),
                float(np.linalg.norm(projection - dst)),
            )
            if endpoint_distance < max(3.0 * scale, parameters["sample_step"] * 0.25):
                reject(row, "target_near_endpoint")
                continue
            direction = delta / distance
            alignment = float(np.dot(endpoint_vectors[endpoint_idx], direction))
            row["direction_cosine"] = alignment
            if alignment < parameters["min_alignment"]:
                reject(row, "direction_mismatch")
                continue
            target_tangent = tangent / math.sqrt(length_squared)
            parallel_cosine = abs(float(np.dot(direction, target_tangent)))
            if parallel_cosine > 0.92 and endpoint_distance > max(8.0 * scale, 0.2 * segment["length"]):
                reject(row, "target_parallel_mismatch")
                continue
            path = _astar_probability_path(
                nodes[endpoint_idx], projection, road, low_threshold,
                surface_probability=surface,
                surface_threshold=parameters["surface_threshold"],
                margin=parameters["path_margin"],
            )
            if len(path) < 2:
                reject(row, "no_astar_path")
                continue
            path_length = float(
                np.linalg.norm(np.diff(path.astype(np.float32), axis=0), axis=1).sum()
            )
            path_ratio = path_length / max(distance, 1e-6)
            row["path_length"] = path_length
            row["path_ratio"] = path_ratio
            row["_path"] = path.astype(float).tolist()
            if path_ratio > parameters["max_path_ratio"]:
                reject(row, "path_ratio_too_large")
                continue
            evidence = _recovery_path_evidence(path, road, low_threshold, surface, parameters)
            row.update({
                "mean_probability": evidence["center_conf"],
                "q25_probability": evidence["center_q25"],
                "weak_fraction": evidence["weak_fraction"],
                "background_probability": evidence["background_conf"],
                "background_contrast": evidence["probability_contrast"],
            })
            if not (evidence["road_supported"] or evidence["surface_supported"]):
                reject(row, _road_evidence_reject_reason(evidence, parameters))
                continue
            directness = min(1.0, 1.0 / max(path_ratio, 1.0))
            crossing_quality = 1.0 - parallel_cosine
            recovery_score = (
                0.28 * min(1.0, evidence["center_conf"] / max(low_threshold, 1e-6))
                + 0.16 * min(1.0, evidence["center_q25"] / max(low_threshold, 1e-6))
                + 0.22 * alignment
                + 0.14 * directness
                + 0.10 * crossing_quality
                + 0.10 * evidence["surface_conf"]
            )
            row["recovery_score"] = recovery_score
            proposals.append({
                "score": recovery_score,
                "endpoint_idx": endpoint_idx,
                "target_pair": segment["node_pair"],
                "target_segment_id": int(segment["edge_ids"][0]),
                "projection": projection.astype(np.float32),
                "path": path,
                "evidence": evidence,
                "row": row,
            })

    combined_nodes = nodes.tolist()
    combined_edges = edge_array.tolist()
    used_endpoints = set()
    recovery_sequence = 0
    for proposal in sorted(proposals, key=lambda item: item["score"], reverse=True):
        endpoint_idx = int(proposal["endpoint_idx"])
        row = proposal["row"]
        if endpoint_idx in used_endpoints:
            reject(row, "endpoint_already_used")
            continue
        current_adjacency = _undirected_adjacency(len(combined_nodes), combined_edges)
        if endpoint_idx >= len(current_adjacency) or len(current_adjacency[endpoint_idx]) != 1:
            reject(row, "endpoint_already_used")
            continue
        target_pair = tuple(proposal["target_pair"])
        target_edge_ids = [
            edge_id for edge_id, edge in enumerate(combined_edges)
            if tuple(sorted((int(edge[0]), int(edge[1])))) == target_pair
        ]
        if not target_edge_ids:
            reject(row, "target_segment_changed")
            continue
        before_stats = graph_connectivity_stats(combined_nodes, combined_edges)
        current_components = _component_labels(len(combined_nodes), combined_edges)
        start_component = current_components.get(endpoint_idx)
        target_component = current_components.get(int(target_pair[0]))
        row.update({
            "start_component": start_component,
            "target_component": target_component,
            "component_count_before": before_stats["component_count"],
            "merges_components": bool(start_component != target_component),
        })
        if start_component == target_component:
            reject(row, "reject_same_component")
            continue

        candidate_nodes = [list(point) for point in combined_nodes]
        candidate_edges = []
        candidate_metadata = []
        removed = set(target_edge_ids)
        removed_rows = []
        for edge_id, edge in enumerate(combined_edges):
            if edge_id in removed:
                removed_rows.append((edge, metadata[edge_id]))
            else:
                candidate_edges.append(tuple(map(int, edge)))
                candidate_metadata.append(dict(metadata[edge_id]))
        junction_idx = len(candidate_nodes)
        candidate_nodes.append(proposal["projection"].astype(float).tolist())
        split_seen = set()
        for (src_idx, dst_idx), source_metadata in removed_rows:
            for split_edge in ((int(src_idx), junction_idx), (junction_idx, int(dst_idx))):
                if split_edge in split_seen:
                    continue
                split_seen.add(split_edge)
                candidate_edges.append(split_edge)
                split_metadata = dict(source_metadata)
                split_metadata["split_from_segment"] = int(proposal["target_segment_id"])
                candidate_metadata.append(split_metadata)

        path = proposal["path"]
        path_steps = np.concatenate([
            np.asarray([0.0], dtype=np.float32),
            np.cumsum(np.linalg.norm(np.diff(path.astype(np.float32), axis=0), axis=1)),
        ])
        chain = [endpoint_idx]
        for sample_distance in np.arange(
            parameters["sample_step"], path_steps[-1], parameters["sample_step"]
        ).tolist():
            path_idx = min(int(np.searchsorted(path_steps, sample_distance)), len(path) - 1)
            point = path[path_idx].astype(np.float32)
            if np.array_equal(np.rint(candidate_nodes[chain[-1]]), np.rint(point)):
                continue
            candidate_nodes.append(point.astype(float).tolist())
            chain.append(len(candidate_nodes) - 1)
        chain.append(junction_idx)
        qa_state = "auto" if proposal["score"] >= parameters["auto_score"] else "review"
        connector_metadata_ids = []
        connector_edge_count = 0
        existing_undirected = {
            tuple(sorted((int(src_idx), int(dst_idx)))) for src_idx, dst_idx in candidate_edges
        }
        recovery_id = f"segment:{recovery_sequence}"
        for src_idx, dst_idx in zip(chain[:-1], chain[1:]):
            edge_key = tuple(sorted((int(src_idx), int(dst_idx))))
            if src_idx == dst_idx or edge_key in existing_undirected:
                continue
            existing_undirected.add(edge_key)
            candidate_edges.append((int(src_idx), int(dst_idx)))
            evidence = proposal["evidence"]
            candidate_metadata.append({
                "line_source": "weak_segment_connector",
                "topology_probability": float(proposal["score"]),
                "recovery_score": float(proposal["score"]),
                "center_conf": float(evidence["center_conf"]),
                "center_q25": float(evidence["center_q25"]),
                "background_conf": float(evidence["background_conf"]),
                "probability_contrast": float(evidence["probability_contrast"]),
                "surface_conf": float(evidence["surface_conf"]),
                "recovery_reason": "weak_probability_endpoint_to_segment",
                "qa_state": qa_state,
                "recovery_id": recovery_id,
                "target_segment_id": int(proposal["target_segment_id"]),
                "target_projection": proposal["projection"].astype(float).tolist(),
                "component_before_start": start_component,
                "component_before_target": target_component,
                "merges_components": True,
                "component_count_before": before_stats["component_count"],
                "component_count_after": None,
                "connectivity_gain": 0,
            })
            connector_metadata_ids.append(len(candidate_metadata) - 1)
            connector_edge_count += 1
        if connector_edge_count == 0:
            reject(row, "duplicate_or_suppressed")
            continue
        after_stats = graph_connectivity_stats(candidate_nodes, candidate_edges)
        connectivity_gain = int(
            before_stats["component_count"] - after_stats["component_count"]
        )
        row["component_count_after"] = after_stats["component_count"]
        row["connectivity_gain"] = connectivity_gain
        if connectivity_gain < 1:
            reject(row, "no_connectivity_gain")
            continue
        for metadata_id in connector_metadata_ids:
            candidate_metadata[metadata_id]["component_count_after"] = after_stats["component_count"]
            candidate_metadata[metadata_id]["connectivity_gain"] = connectivity_gain
        row["accepted"] = True
        row["reject_reason"] = ""
        row["candidate_id"] = recovery_id
        if candidate_audit is not None:
            candidate_audit.append(row)
        combined_nodes = candidate_nodes
        combined_edges = candidate_edges
        metadata = candidate_metadata
        used_endpoints.add(endpoint_idx)
        recovery_sequence += 1
        summary["endpoint_segment_accepted_count"] += 1
        summary["endpoint_segment_recovered_edge_count"] += connector_edge_count
        summary["endpoint_segment_split_edge_delta"] += len(removed_rows)
        summary["endpoint_segment_connectivity_gain"] += connectivity_gain

    summary["endpoint_segment_reject_reason_counts"] = dict(sorted(reject_counts.items()))
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
    endpoint_segment_candidate_audit=None,
    relative_context=None,
    topology_candidate_nodes_rc=None,
    topology_candidate_edges=None,
    topology_candidate_scores=None,
):
    """Diagnose, bootstrap, recover endpoints, then optionally join endpoints to segments."""
    original_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    connectivity_before = graph_connectivity_stats(nodes_rc, original_edges)
    diagnosis = diagnose_scene_confidence(
        road_probability,
        nodes_rc,
        original_edges,
        config,
        distance_scale=distance_scale,
    )
    if relative_context is None:
        relative_context = compute_relative_roadness(
            road_probability,
            config,
            scene_state=diagnosis["scene_confidence_state"],
            distance_scale=distance_scale,
        )
    relative_enabled = bool(
        relative_context
        and relative_context.get("diagnostics", {}).get("relative_roadness_enabled", False)
    )
    bootstrap_enabled = bool(_config_value(config, "WEAK_BOOTSTRAP_ENABLED", True))
    only_low = bool(_config_value(config, "WEAK_BOOTSTRAP_ONLY_IF_LOW_CONFIDENCE", True))
    absolute_bootstrap = (
        not only_low or diagnosis["scene_confidence_state"] in {"low_confidence", "very_low_confidence"}
    )
    should_bootstrap = bootstrap_enabled and (absolute_bootstrap or relative_enabled)
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
                relative_context=relative_context,
                include_absolute_candidates=absolute_bootstrap,
                topology_candidate_nodes_rc=topology_candidate_nodes_rc,
                topology_candidate_edges=topology_candidate_edges,
                topology_candidate_scores=topology_candidate_scores,
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
            "relative_candidate_count": 0,
            "relative_accepted_candidate_count": 0,
            "relative_recovered_edge_count": 0,
            "relative_auto_count": 0,
            "relative_review_count": 0,
            "relative_rejected_count": 0,
            "relative_reject_reason_counts": {},
            "relative_topology_candidate_edge_count": 0,
            "relative_topology_selected_edge_count": 0,
            "relative_graph_point_count": 0,
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
    final_nodes, final_edges, final_metadata, segment_summary = (
        recover_endpoint_to_segment_connections(
            final_nodes,
            final_edges,
            road_probability,
            config,
            edge_metadata=final_metadata,
            surface_probability=surface_probability,
            distance_scale=distance_scale,
            candidate_audit=endpoint_segment_candidate_audit,
        )
    )
    connectivity_after = graph_connectivity_stats(final_nodes, final_edges)
    relative_topology_edge_count = sum(
        row.get("line_source") == "relative_roadness" for row in final_metadata
    )
    relative_total_edge_count = sum(
        row.get("candidate_source") in {"relative", "absolute+relative"}
        or str(row.get("line_source", "")).startswith("relative")
        for row in final_metadata
    )
    final_nodes_array = np.asarray(final_nodes, dtype=np.float32).reshape(-1, 2)
    relative_final_length = float(sum(
        np.linalg.norm(final_nodes_array[int(dst_idx)] - final_nodes_array[int(src_idx)])
        for (src_idx, dst_idx), row in zip(final_edges.tolist(), final_metadata)
        if row.get("candidate_source") in {"relative", "absolute+relative"}
        or str(row.get("line_source", "")).startswith("relative")
    ))
    final_total_length = float(sum(
        np.linalg.norm(final_nodes_array[int(dst_idx)] - final_nodes_array[int(src_idx)])
        for src_idx, dst_idx in final_edges.tolist()
    ))
    structure_rejects = dict(
        relative_context.get("diagnostics", {}).get("relative_structure_reject_reason_counts", {})
        if relative_context else {}
    )
    acceptance_rejects = dict(bootstrap_summary.get("relative_reject_reason_counts", {}))
    funnel_rejects = Counter(structure_rejects)
    funnel_rejects.update(acceptance_rejects)
    funnel = {
        "relative_candidate_pixels": int(
            relative_context.get("diagnostics", {}).get("relative_candidate_pixel_count", 0)
            if relative_context else 0
        ),
        "relative_candidate_components": int(
            relative_context.get("diagnostics", {}).get("relative_component_count", 0)
            if relative_context else 0
        ),
        "relative_skeleton_before_structure_filter": int(
            relative_context.get("diagnostics", {}).get("relative_skeleton_before_structure_filter", 0)
            if relative_context else 0
        ),
        "relative_skeleton_after_structure_filter": int(
            relative_context.get("diagnostics", {}).get("relative_skeleton_after_structure_filter", 0)
            if relative_context else 0
        ),
        "relative_chain_count": int(
            relative_context.get("diagnostics", {}).get("relative_chain_count", 0)
            if relative_context else 0
        ),
        "raw_chain_count": int(
            relative_context.get("diagnostics", {}).get("raw_chain_count", 0)
            if relative_context else 0
        ),
        "raw_too_short_count": int(
            relative_context.get("diagnostics", {}).get("raw_short_chain_count", 0)
            if relative_context else 0
        ),
        "normalized_chain_count": int(
            relative_context.get("diagnostics", {}).get("normalized_chain_count", 0)
            if relative_context else 0
        ),
        "normalized_too_short_count": int(
            relative_context.get("diagnostics", {}).get("normalized_short_chain_count", 0)
            if relative_context else 0
        ),
        "structure_rescued_chain_count": int(
            relative_context.get("diagnostics", {}).get("structure_rescued_chain_count", 0)
            if relative_context else 0
        ),
        "structure_rescued_length": float(
            relative_context.get("diagnostics", {}).get("structure_rescued_length", 0.0)
            if relative_context else 0.0
        ),
        "relative_chain_geometry_pass": int(
            relative_context.get("diagnostics", {}).get("relative_chain_geometry_pass", 0)
            if relative_context else 0
        ),
        "relative_graph_point_count": int(
            (
                relative_context.get("diagnostics", {}).get("relative_graph_point_count", 0)
                if relative_context else 0
            )
            or bootstrap_summary.get("relative_graph_point_count", 0)
        ),
        "relative_topology_candidate_edge_count": int(
            bootstrap_summary.get("relative_topology_candidate_edge_count", 0)
        ),
        "relative_topology_selected_edge_count": int(
            bootstrap_summary.get("relative_topology_selected_edge_count", 0)
        ),
        "relative_bootstrap_candidate_count": int(bootstrap_summary.get("relative_candidate_count", 0)),
        "relative_auto_count": int(bootstrap_summary.get("relative_auto_count", 0)),
        "relative_review_count": int(bootstrap_summary.get("relative_review_count", 0)),
        "relative_rejected_count": int(bootstrap_summary.get("relative_rejected_count", 0)),
        "relative_final_edge_count": int(relative_total_edge_count),
        "relative_final_length_px": relative_final_length,
        "final_total_centerline_length_px": final_total_length,
        "reject_reason_counts": dict(sorted(funnel_rejects.items())),
    }
    summary = {
        **recovery_summary,
        **diagnosis,
        **bootstrap_summary,
        **segment_summary,
        **(
            {
                key: value
                for key, value in relative_context.get("diagnostics", {}).items()
                if key != "junction_zones"
            }
            if relative_context else {}
        ),
        "strong_edge_count": int(len(original_edges)),
        "final_edge_count": int(len(final_edges)),
        "added_edge_count": int(len(final_edges) - len(original_edges)),
        "bootstrap_ran": bool(should_bootstrap),
        "absolute_bootstrap_ran": bool(should_bootstrap and absolute_bootstrap),
        "relative_bootstrap_ran": bool(should_bootstrap and relative_enabled),
        "relative_topology_edge_count": int(relative_topology_edge_count),
        "relative_total_edge_count": int(relative_total_edge_count),
        "relative_final_length_px": relative_final_length,
        "final_total_centerline_length_px": final_total_length,
        "relative_acceptance_funnel": funnel,
        "relative_chain_candidate_count": int(
            bootstrap_summary.get("relative_candidate_count", 0)
        ),
        "relative_chain_accepted_count": int(
            bootstrap_summary.get("relative_accepted_candidate_count", 0)
        ),
        "relative_chain_auto_count": int(
            bootstrap_summary.get("relative_auto_count", 0)
        ),
        "relative_chain_review_count": int(
            bootstrap_summary.get("relative_review_count", 0)
        ),
        "relative_chain_rejected_count": int(
            bootstrap_summary.get("relative_rejected_count", 0)
        ),
        "component_count_before": connectivity_before["component_count"],
        "component_count_after": connectivity_after["component_count"],
        "endpoint_count_before": connectivity_before["endpoint_count"],
        "endpoint_count_after": connectivity_after["endpoint_count"],
        "largest_component_node_count_before": connectivity_before[
            "largest_component_node_count"
        ],
        "largest_component_node_count_after": connectivity_after[
            "largest_component_node_count"
        ],
        "largest_component_edge_count_before": connectivity_before[
            "largest_component_edge_count"
        ],
        "largest_component_edge_count_after": connectivity_after[
            "largest_component_edge_count"
        ],
        "largest_component_fraction_before": connectivity_before[
            "largest_component_fraction"
        ],
        "largest_component_fraction_after": connectivity_after[
            "largest_component_fraction"
        ],
        "connectivity_gain_total": max(
            0,
            connectivity_before["component_count"] - connectivity_after["component_count"],
        ),
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


def extract_graph_points(keypoint_mask, road_mask, config, *, relative_context=None):
    high_threshold, _low_threshold, _profile_name = resolve_road_thresholds(config)
    kp_candidates, kp_scores = get_points_and_scores_from_mask(keypoint_mask, config.ITSC_THRESHOLD * 255)
    kps_0 = nms_points(kp_candidates, kp_scores, config.ITSC_NMS_RADIUS)
    # The keypoint heatmap contains broad blobs. Keep its historical coarse
    # spacing; divided-road preservation is applied only to the road skeleton.
    if kps_0.shape[0]:
        kps_0 = nms_points(kps_0, np.ones(kps_0.shape[0]), config.ROAD_NMS_RADIUS)
    road_skel_mask = skeletonize_road_mask(road_mask, high_threshold * 255)
    relative_score = None
    if relative_context is not None and bool(
        _config_value(config, "RELATIVE_ROADNESS_ENABLED", False)
    ):
        combined = np.asarray(relative_context.get("combined_skeleton", []))
        if combined.shape == road_skel_mask.shape:
            road_skel_mask = np.maximum(
                road_skel_mask, (combined > 0).astype(np.uint8) * 255
            )
        candidate_score = np.asarray(relative_context.get("relative_score", []), dtype=np.float32)
        if candidate_score.shape == road_skel_mask.shape:
            relative_score = candidate_score
    kp_candidates, kp_scores = get_points_and_scores_from_mask(road_skel_mask, 0)
    road_scores = road_mask[kp_candidates[:, 1], kp_candidates[:, 0]] if kp_candidates.shape[0] else kp_scores
    if relative_score is not None and kp_candidates.shape[0]:
        road_scores = np.maximum(
            road_scores.astype(np.float32) / 255.0,
            relative_score[kp_candidates[:, 1], kp_candidates[:, 0]],
        )
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


def extract_graph_astar(keypoint_mask, road_mask, config, *, relative_context=None):
    kps = extract_graph_points(
        keypoint_mask, road_mask, config, relative_context=relative_context
    )

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
