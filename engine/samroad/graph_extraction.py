import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import time
import tcod
from collections import Counter, defaultdict, deque
from sklearn.neighbors import KDTree
from skimage.draw import line
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, maximum_position
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
            "ribbon_fraction": 0.0,
            "regularized_skeleton_fraction": 0.0,
            "continuous_trace_fraction": 0.0,
            "trace_id": 0,
            "trace_total_length": 0.0,
            "relative_supported": False,
            "backbone_line_source": "",
            "backbone_reason": "",
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
    ribbon_structure = sampled("relative_ribbon_centerline") > 0
    regularized_structure = sampled("relative_regularized_final_skeleton") > 0
    continuous_structure = sampled("relative_continuous_centerline") > 0
    sampled_trace_ids = sampled("relative_trace_id_map").astype(np.int32)
    positive_trace_ids = sampled_trace_ids[sampled_trace_ids > 0]
    dominant_trace_id = (
        int(np.bincount(positive_trace_ids).argmax())
        if len(positive_trace_ids) else 0
    )
    trace_lengths = {
        int(row.get("trace_id", 0)): float(row.get("length", 0.0))
        for row in relative_context.get("diagnostics", {}).get(
            "relative_trace_summaries", []
        )
    }
    source_value = relative_context.get("relative_backbone_source_labels")
    source_array = np.asarray(source_value, dtype=np.uint8) if source_value is not None else None
    source_codes = (
        source_array[rows, cols]
        if source_array is not None and source_array.shape == score.shape
        else np.zeros(len(rows), dtype=np.uint8)
    )
    positive_source_codes = source_codes[source_codes > 0]
    dominant_source = (
        int(np.bincount(positive_source_codes, minlength=5).argmax())
        if len(positive_source_codes) else 0
    )
    backbone_source, backbone_reason = _RELATIVE_BACKBONE_SOURCES.get(
        dominant_source, ("", "")
    )
    ribbon_fraction = float(np.mean(ribbon_structure)) if values.size else 0.0
    continuous_fraction = float(np.mean(continuous_structure)) if values.size else 0.0
    regularized_fraction = float(np.mean(regularized_structure)) if values.size else 0.0
    if regularized_fraction >= 0.50:
        backbone_source = "relative_regularized_skeleton"
        backbone_reason = "regularized_candidate_skeleton"
    elif continuous_fraction >= 0.50:
        backbone_source = "relative_continuous_trace"
        backbone_reason = "continuous_ribbon_trace"
    elif ribbon_fraction >= 0.50:
        backbone_source = "relative_ribbon_centerline"
        backbone_reason = "regularized_ribbon_centerline"
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
        "ribbon_fraction": ribbon_fraction,
        "regularized_skeleton_fraction": regularized_fraction,
        "continuous_trace_fraction": continuous_fraction,
        "trace_id": int(dominant_trace_id),
        "trace_total_length": float(trace_lengths.get(dominant_trace_id, 0.0)),
        "relative_supported": bool(
            fraction >= 0.50
            and q25 >= max(weak_threshold - 1e-6, 0.0)
            and float(np.mean(normalized_contrast)) > 0.0
        ),
        "backbone_line_source": backbone_source,
        "backbone_reason": backbone_reason,
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


def _chain_endpoint_tangent(path, endpoint_index, lookback=8.0):
    """Return the unit vector pointing from one endpoint into its chain."""
    points = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return np.zeros(2, dtype=np.float32)
    ordered = points if endpoint_index == 0 else points[::-1]
    distances = np.concatenate((
        np.asarray([0.0], dtype=np.float32),
        np.cumsum(np.linalg.norm(np.diff(ordered, axis=0), axis=1)),
    ))
    sample_index = min(
        max(int(np.searchsorted(distances, min(float(lookback), float(distances[-1])))), 1),
        len(ordered) - 1,
    )
    tangent = ordered[sample_index] - ordered[0]
    norm = float(np.linalg.norm(tangent))
    return (tangent / max(norm, 1e-6)).astype(np.float32)


def build_relative_chain_corridors(
    skeleton,
    *,
    relative_score=None,
    scene_rank=None,
    scale_agreement=None,
    candidate_mask=None,
    junction_zone_labels=None,
):
    """Group traced micro-chains logically without changing their geometry.

    Chains are joined only at a real shared skeleton endpoint and only when
    they are mutual, unambiguous, near-opposite tangent continuations.  The
    grouping therefore keeps T/X directions separate and cannot bridge a gap
    between nearby parallel roads.
    """
    value = np.asarray(skeleton, dtype=bool)
    chains = _trace_skeleton_chains(value)
    shape = value.shape

    def optional_map(array, dtype=np.float32):
        if array is None:
            return None
        result = np.asarray(array, dtype=dtype)
        return result if result.shape == shape else None

    score_map = optional_map(relative_score)
    rank_map = optional_map(scene_rank)
    agreement_map = optional_map(scale_agreement)
    candidate = optional_map(candidate_mask, np.uint8)
    zone_labels = optional_map(junction_zone_labels, np.int32)

    parent = list(range(len(chains)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    records = []
    endpoint_members = defaultdict(list)
    pixel_adjacency = _skeleton_adjacency(value)
    exact_junction_ids = {}
    next_junction_id = 1
    for chain_index, path in enumerate(chains):
        rows, cols = path[:, 0], path[:, 1]

        def values(array, default):
            return array[rows, cols] if array is not None else np.full(len(path), default)

        scores = values(score_map, 0.0).astype(np.float32)
        ranks = values(rank_map, 1.0).astype(np.float32)
        agreements = values(agreement_map, 1.0).astype(np.float32)
        coverage = values(candidate, 1).astype(np.float32)
        endpoints = [tuple(map(int, path[0])), tuple(map(int, path[-1]))]
        tangents = [
            _chain_endpoint_tangent(path, 0),
            _chain_endpoint_tangent(path, 1),
        ]
        junction_ids = []
        for endpoint_index, point in enumerate(endpoints):
            is_junction = len(pixel_adjacency.get(point, [])) >= 3
            zone_id = int(zone_labels[point]) if zone_labels is not None else 0
            if is_junction:
                if zone_id > 0:
                    junction_id = zone_id
                else:
                    if point not in exact_junction_ids:
                        exact_junction_ids[point] = next_junction_id
                        next_junction_id += 1
                    junction_id = exact_junction_ids[point]
            else:
                junction_id = 0
            junction_ids.append(junction_id)
            endpoint_members[point].append({
                "chain_index": chain_index,
                "endpoint_index": endpoint_index,
                "tangent": tangents[endpoint_index],
            })
        geometry = _relative_chain_geometry(path)
        records.append({
            "chain_id": chain_index + 1,
            "path": path,
            "length": float(geometry["path_length"]),
            "start": list(endpoints[0]),
            "end": list(endpoints[1]),
            "endpoint_tangents": [tangent.astype(float).tolist() for tangent in tangents],
            "relative_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "relative_score_q25": float(np.quantile(scores, 0.25)) if len(scores) else 0.0,
            "scene_rank_mean": float(np.mean(ranks)) if len(ranks) else 0.0,
            "scene_rank_q25": float(np.quantile(ranks, 0.25)) if len(ranks) else 0.0,
            "scale_agreement_mean": float(np.mean(agreements)) if len(agreements) else 0.0,
            "scale_agreement_q25": float(np.quantile(agreements, 0.25)) if len(agreements) else 0.0,
            "candidate_coverage": float(np.mean(coverage > 0)) if len(coverage) else 0.0,
            "neighbor_chain_ids": [],
            "junction_ids": junction_ids,
            "geometry": geometry,
        })

    pairing_count = 0
    ambiguous_junction_count = 0
    for point, members in endpoint_members.items():
        unique_chain_ids = sorted({item["chain_index"] for item in members})
        for chain_index in unique_chain_ids:
            records[chain_index]["neighbor_chain_ids"] = sorted(set(
                records[chain_index]["neighbor_chain_ids"]
                + [other + 1 for other in unique_chain_ids if other != chain_index]
            ))
        if len(members) < 2:
            continue
        pair_scores = {}
        ranked = defaultdict(list)
        for first_index in range(len(members)):
            for second_index in range(first_index + 1, len(members)):
                first, second = members[first_index], members[second_index]
                if first["chain_index"] == second["chain_index"]:
                    continue
                continuation = -float(np.dot(first["tangent"], second["tangent"]))
                if continuation < 0.70:
                    continue
                first_record = records[first["chain_index"]]
                second_record = records[second["chain_index"]]
                if min(first_record["candidate_coverage"], second_record["candidate_coverage"]) < 0.50:
                    continue
                key = (first_index, second_index)
                # Evidence is deliberately a tie-breaker, not a hard gate: a
                # short weak bridge between two strong pieces must stay in the
                # same otherwise coherent corridor.
                evidence_similarity = 1.0 - min(
                    abs(first_record["relative_score_mean"] - second_record["relative_score_mean"]),
                    1.0,
                )
                pair_scores[key] = continuation + 0.05 * evidence_similarity
                ranked[first_index].append((pair_scores[key], second_index))
                ranked[second_index].append((pair_scores[key], first_index))
        best = {}
        ambiguous = set()
        for member_index, candidates in ranked.items():
            candidates.sort(reverse=True)
            best[member_index] = candidates[0][1]
            if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
                ambiguous.add(member_index)
        if ambiguous:
            ambiguous_junction_count += 1
        used = set()
        for first_index, second_index in sorted(best.items()):
            if first_index in used or second_index in used:
                continue
            if first_index in ambiguous or second_index in ambiguous:
                continue
            if best.get(second_index) != first_index:
                continue
            first_chain = members[first_index]["chain_index"]
            second_chain = members[second_index]["chain_index"]
            union(first_chain, second_chain)
            used.update((first_index, second_index))
            pairing_count += 1

    grouped = defaultdict(list)
    for chain_index in range(len(records)):
        grouped[find(chain_index)].append(chain_index)
    ordered_groups = sorted(grouped.values(), key=lambda group: min(group))
    corridors = []
    chain_labels = np.zeros(shape, dtype=np.int32)
    corridor_labels = np.zeros(shape, dtype=np.int32)
    for corridor_id, group in enumerate(ordered_groups, start=1):
        lengths = np.asarray([records[index]["length"] for index in group], dtype=np.float32)
        total_length = float(np.sum(lengths))
        total_weight = max(total_length, 1e-6)

        def weighted(name):
            return float(sum(records[index][name] * records[index]["length"] for index in group) / total_weight)

        corridor = {
            "corridor_id": corridor_id,
            "chain_ids": [index + 1 for index in group],
            "chain_count": int(len(group)),
            "total_length": total_length,
            "relative_score_mean": weighted("relative_score_mean"),
            "relative_score_q25": weighted("relative_score_q25"),
            "scene_rank_mean": weighted("scene_rank_mean"),
            "scene_rank_q25": weighted("scene_rank_q25"),
            "scale_agreement_mean": weighted("scale_agreement_mean"),
            "scale_agreement_q25": weighted("scale_agreement_q25"),
        }
        corridors.append(corridor)
        for chain_index in group:
            record = records[chain_index]
            record["corridor_id"] = corridor_id
            record["corridor_total_length"] = total_length
            path = record["path"]
            chain_labels[path[:, 0], path[:, 1]] = record["chain_id"]
            corridor_labels[path[:, 0], path[:, 1]] = corridor_id
    return {
        "chains": records,
        "corridors": corridors,
        "chain_labels": chain_labels,
        "corridor_labels": corridor_labels,
        "pairing_count": int(pairing_count),
        "junction_count": int(sum(len(items) >= 3 for items in pixel_adjacency.values())),
        "ambiguous_junction_count": int(ambiguous_junction_count),
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
    """Prune only unmistakable tiny spurs; never redraw a junction corridor."""
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
    complex_zone_count = 0
    complex_zone_skipped_count = 0
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
        bbox = zone["bbox"]
        bbox_span = max(int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1]))
        long_branch_count = sum(
            branch["continuation_length"] > 2.0 * radius for branch in branches
        )
        plausible_pairs = 0
        for first_index in range(len(branches)):
            for second_index in range(first_index + 1, len(branches)):
                if -float(np.dot(
                    branches[first_index]["outward_tangent"],
                    branches[second_index]["outward_tangent"],
                )) >= 0.70:
                    plausible_pairs += 1
        complex_zone = bool(
            bbox_span > 6 * radius
            or len(branches) >= 4
            or long_branch_count >= 3
            or plausible_pairs >= 2
        )
        zone_record.update({
            "bbox_span_px": int(bbox_span),
            "long_branch_count": int(long_branch_count),
            "plausible_through_pair_count": int(plausible_pairs),
            "complex_junction_zone": complex_zone,
        })
        if complex_zone:
            # Large/ambiguous zones include T/X crossings, interchanges,
            # parallel rails and chains of nearby junctions.  Keep every raw
            # skeleton pixel and let logical corridor grouping reason about
            # continuation without inventing geometry or topology.
            complex_zone_count += 1
            complex_zone_skipped_count += 1
            zone_record["skip_reason"] = "complex_junction_zone"
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
        pruned_crop = pruned_spurs[region]
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
            # The only allowed physical edit is removal of a component that is
            # wholly local to the zone.  We never clear/retrace the junction
            # interior and never connect an entry to a chosen main path.
            tiny_local_spur = bool(
                branch["continuation_length"] <= radius
                and len(branch["component_points"]) <= max(3, 2 * radius)
                and not extends_outside_zone
            )
            if tiny_local_spur:
                component_points = branch["component_points"]
                normalized_crop[component_points[:, 0], component_points[:, 1]] = False
                pruned_crop[component_points[:, 0], component_points[:, 1]] = True
                pruned_count += 1
                branch_records[branch_index]["role"] = "pruned_ladder_spur"
            else:
                preserved_branch_count += 1
                branch_records[branch_index]["role"] = (
                    "preserved_real_branch" if evidence_continues
                    else "preserved_ambiguous_branch"
                )
        zone_record.update({
            "preserved_branch_count": int(preserved_branch_count),
            "collapsed": False,
            "skip_reason": "tiny_ladder_spur_pruned" if any(
                row["role"] == "pruned_ladder_spur" for row in branch_records
            ) else "no_unmistakable_tiny_spur",
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
        "complex_junction_zone_count": int(complex_zone_count),
        "complex_zone_skipped_collapse_count": int(complex_zone_skipped_count),
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


def extract_relative_ridge_centerline(
    relative_score,
    relative_candidate_mask,
    *,
    min_chain_length=48.0,
):
    """Extract a narrow centerline by NMS across local score-ridge normals.

    A structure tensor estimates the dominant cross-ridge (normal) direction
    from the continuous Relative Roadness field.  Non-maximum suppression is
    then applied only along that normal.  The binary candidate remains a
    spatial support mask; its width and boundary do not define the centerline.
    """
    score = np.asarray(relative_score, dtype=np.float32)
    candidate = np.asarray(relative_candidate_mask, dtype=np.uint8) > 0
    if score.ndim != 2 or candidate.shape != score.shape:
        raise ValueError("relative_score and relative_candidate_mask must be aligned 2D arrays")
    binary_skeleton = skeletonize(candidate).astype(np.uint8)
    if score.size == 0 or not np.any(candidate):
        empty_float = np.zeros(score.shape, dtype=np.float32)
        return {
            "relative_ridge_mask": np.zeros(score.shape, dtype=np.uint8),
            "ridge_orientation": empty_float,
            "ridge_strength": empty_float.copy(),
            "binary_skeleton": binary_skeleton,
            "diagnostics": {
                "candidate_pixel_count": int(np.count_nonzero(candidate)),
                "old_binary_skeleton_length": int(np.count_nonzero(binary_skeleton)),
                "ridge_skeleton_length": 0,
                "old_junction_pixel_count": 0,
                "ridge_junction_pixel_count": 0,
                "old_micro_chain_count": 0,
                "ridge_micro_chain_count": 0,
                "old_too_short_count": 0,
                "ridge_too_short_count": 0,
                "ridge_fallback_used": False,
            },
        }

    smooth = cv2.GaussianBlur(
        score, (0, 0), sigmaX=1.2, sigmaY=1.2,
        borderType=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)
    gradient_col = cv2.Sobel(
        smooth, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101
    )
    gradient_row = cv2.Sobel(
        smooth, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101
    )
    tensor_xy = cv2.GaussianBlur(
        gradient_col * gradient_row, (0, 0), 2.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    tensor_xx = cv2.GaussianBlur(
        gradient_col * gradient_col, (0, 0), 2.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    tensor_yy = cv2.GaussianBlur(
        gradient_row * gradient_row, (0, 0), 2.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    del gradient_col, gradient_row

    # Dominant tensor eigenvector = local ridge normal. Orientation output is
    # the perpendicular road tangent, modulo pi.
    normal_orientation = 0.5 * np.arctan2(
        2.0 * tensor_xy, tensor_xx - tensor_yy
    ).astype(np.float32)
    ridge_orientation = np.mod(
        normal_orientation + np.float32(math.pi / 2.0), np.float32(math.pi)
    ).astype(np.float32)
    tensor_anisotropy = np.sqrt(
        (tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy ** 2
    ).astype(np.float32)
    tensor_coherence = (
        tensor_anisotropy / (tensor_xx + tensor_yy + 1e-12)
    ).astype(np.float32)
    del tensor_xx, tensor_yy, tensor_xy

    # Sample the continuous normal direction with bilinear interpolation.
    # Processing row strips keeps full-scene memory bounded while avoiding the
    # one-pixel discontinuities caused by quantizing orientations into bins.
    ridge_nms = np.zeros(score.shape, dtype=bool)
    ridge_strength = np.zeros(score.shape, dtype=np.float32)
    height, width = score.shape
    base_cols = np.arange(width, dtype=np.float32)[None, :]
    strip_height = 256
    for row0 in range(1, max(height - 1, 1), strip_height):
        row1 = min(row0 + strip_height, height - 1)
        if row1 <= row0:
            continue
        normal = normal_orientation[row0:row1]
        normal_col = np.cos(normal).astype(np.float32)
        normal_row = np.sin(normal).astype(np.float32)
        base_rows = np.arange(row0, row1, dtype=np.float32)[:, None]
        forward = cv2.remap(
            smooth,
            base_cols + normal_col,
            base_rows + normal_row,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        backward = cv2.remap(
            smooth,
            base_cols - normal_col,
            base_rows - normal_row,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        center = smooth[row0:row1]
        cross_section_drop = center - 0.5 * (forward + backward)
        keep = (
            candidate[row0:row1]
            & (center >= forward)
            & (center > backward)
            & (cross_section_drop > np.finfo(np.float32).eps)
            & (tensor_coherence[row0:row1] >= (1.0 / 3.0))
        )
        keep[:, 0] = False
        keep[:, -1] = False
        ridge_nms[row0:row1][keep] = True
        ridge_strength[row0:row1][keep] = (
            cross_section_drop[keep]
            * tensor_coherence[row0:row1][keep]
        )
    del normal_orientation, tensor_anisotropy, tensor_coherence

    narrow_ridge_support = cv2.dilate(
        ridge_nms.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ) > 0
    ridge_skeleton = skeletonize(narrow_ridge_support & candidate).astype(np.uint8)
    fallback_used = not np.any(ridge_skeleton) and np.any(binary_skeleton)
    if fallback_used:
        ridge_skeleton = binary_skeleton.copy()
    ridge_strength *= ridge_skeleton.astype(np.float32)

    old_chains, _old_lengths, old_summary = _skeleton_chain_summary(
        binary_skeleton > 0, float(min_chain_length)
    )
    ridge_chains, _ridge_lengths, ridge_summary = _skeleton_chain_summary(
        ridge_skeleton > 0, float(min_chain_length)
    )
    old_adjacency = _skeleton_adjacency(binary_skeleton > 0)
    ridge_adjacency = _skeleton_adjacency(ridge_skeleton > 0)
    diagnostics = {
        "candidate_pixel_count": int(np.count_nonzero(candidate)),
        "old_binary_skeleton_length": int(np.count_nonzero(binary_skeleton)),
        "ridge_skeleton_length": int(np.count_nonzero(ridge_skeleton)),
        "old_junction_pixel_count": int(sum(
            len(neighbors) >= 3 for neighbors in old_adjacency.values()
        )),
        "ridge_junction_pixel_count": int(sum(
            len(neighbors) >= 3 for neighbors in ridge_adjacency.values()
        )),
        "old_micro_chain_count": int(len(old_chains)),
        "ridge_micro_chain_count": int(len(ridge_chains)),
        "old_too_short_count": int(old_summary["short_chain_count"]),
        "ridge_too_short_count": int(ridge_summary["short_chain_count"]),
        "ridge_fallback_used": bool(fallback_used),
    }
    return {
        "relative_ridge_mask": ridge_skeleton,
        "ridge_orientation": ridge_orientation,
        "ridge_strength": ridge_strength,
        "binary_skeleton": binary_skeleton,
        "diagnostics": diagnostics,
    }


def _bridge_oriented_ribbon_gaps(
    ribbon,
    candidate,
    orientation,
    center_offset,
    *,
    max_gap=8.0,
    max_iterations=2,
):
    """Join only mutually facing, locally oriented breaks inside one ribbon."""
    tracked = np.asarray(ribbon, dtype=np.uint8).copy()
    candidate = np.asarray(candidate, dtype=bool)
    orientation = np.asarray(orientation, dtype=np.float32)
    center_offset = np.asarray(center_offset, dtype=np.float32)
    maximum_gap = max(float(max_gap), math.sqrt(2.0) + 1e-6)
    bridge_count = 0
    bridge_pixel_count = 0

    for _iteration in range(max(1, int(max_iterations))):
        adjacency = _skeleton_adjacency(tracked > 0)
        endpoints = [point for point, neighbors in adjacency.items() if len(neighbors) == 1]
        if len(endpoints) < 2:
            break
        endpoint_array = np.asarray(endpoints, dtype=np.float32)
        _component_count, component_labels = cv2.connectedComponents(tracked, 8)
        endpoint_tree = KDTree(endpoint_array)
        proposals = []
        for first_index, first_point in enumerate(endpoints):
            first_row, first_col = first_point
            first_label = int(component_labels[first_row, first_col])
            first_inside = np.asarray(adjacency[first_point][0], dtype=np.float32)
            first_outward = endpoint_array[first_index] - first_inside
            first_outward /= max(float(np.linalg.norm(first_outward)), 1e-6)
            nearby = endpoint_tree.query_radius(
                endpoint_array[first_index:first_index + 1], r=maximum_gap
            )[0]
            for second_index in nearby:
                second_index = int(second_index)
                if second_index <= first_index:
                    continue
                second_point = endpoints[second_index]
                second_row, second_col = second_point
                if int(component_labels[second_row, second_col]) == first_label:
                    continue
                displacement = endpoint_array[second_index] - endpoint_array[first_index]
                gap_length = float(np.linalg.norm(displacement))
                if gap_length <= math.sqrt(2.0) + 1e-6:
                    continue
                gap_direction = displacement / gap_length
                second_inside = np.asarray(adjacency[second_point][0], dtype=np.float32)
                second_outward = endpoint_array[second_index] - second_inside
                second_outward /= max(float(np.linalg.norm(second_outward)), 1e-6)
                facing = min(
                    float(np.dot(first_outward, gap_direction)),
                    float(np.dot(second_outward, -gap_direction)),
                )
                if facing < 0.50:
                    continue

                first_tangent = np.asarray(
                    [math.sin(float(orientation[first_row, first_col])),
                     math.cos(float(orientation[first_row, first_col]))],
                    dtype=np.float32,
                )
                second_tangent = np.asarray(
                    [math.sin(float(orientation[second_row, second_col])),
                     math.cos(float(orientation[second_row, second_col]))],
                    dtype=np.float32,
                )
                tangent_alignment = min(
                    abs(float(np.dot(first_tangent, gap_direction))),
                    abs(float(np.dot(second_tangent, gap_direction))),
                )
                if tangent_alignment < 0.70:
                    continue

                bridge_rows, bridge_cols = line(
                    first_row, first_col, second_row, second_col
                )
                if not np.all(candidate[bridge_rows, bridge_cols]):
                    continue
                offsets = np.abs(center_offset[bridge_rows, bridge_cols])
                centered_fraction = float(np.mean(np.isfinite(offsets) & (offsets <= 2.5)))
                if centered_fraction < 0.75:
                    continue
                proposals.append((
                    gap_length - 0.5 * tangent_alignment - 0.25 * facing,
                    first_index,
                    second_index,
                    bridge_rows,
                    bridge_cols,
                ))

        if not proposals:
            break
        used_endpoints = set()
        added_this_iteration = 0
        for _cost, first_index, second_index, bridge_rows, bridge_cols in sorted(
            proposals, key=lambda item: item[0]
        ):
            if first_index in used_endpoints or second_index in used_endpoints:
                continue
            new_pixels = int(np.count_nonzero(tracked[bridge_rows, bridge_cols] == 0))
            if not new_pixels:
                continue
            tracked[bridge_rows, bridge_cols] = 1
            used_endpoints.update((first_index, second_index))
            bridge_count += 1
            bridge_pixel_count += new_pixels
            added_this_iteration += new_pixels
        if not added_this_iteration:
            break
        tracked = skeletonize(tracked > 0).astype(np.uint8)

    return tracked, {
        "ribbon_tracking_bridge_count": int(bridge_count),
        "ribbon_tracking_bridge_pixel_count": int(bridge_pixel_count),
    }


def extract_relative_ribbon_centerline(
    relative_score,
    relative_candidate_mask,
    ridge_orientation,
    *,
    scale_agreement=None,
    cross_section_radius=12,
    center_tolerance=0.9,
    orientation_clarity=0.85,
    junction_radius=4,
    tracking_gap=8.0,
):
    """Estimate one stable center per oriented candidate-ribbon cross-section.

    The candidate is treated as a finite-width ribbon rather than geometry.
    Each candidate pixel samples its contiguous cross-section along the local
    normal and moves toward the Relative-score x distance-transform weighted
    centroid.  Fixed points form a thin center support, which is skeletonized
    only after centering and then continued across short, mutually facing gaps
    along the local orientation.  Binary geometry is used solely inside very
    local junction zones so independent T/X branches can meet without
    inventing a shortcut outside candidate support.
    """
    score = np.asarray(relative_score, dtype=np.float32)
    candidate = np.asarray(relative_candidate_mask, dtype=np.uint8) > 0
    orientation = np.asarray(ridge_orientation, dtype=np.float32)
    if score.ndim != 2 or candidate.shape != score.shape or orientation.shape != score.shape:
        raise ValueError(
            "relative_score, relative_candidate_mask, and ridge_orientation "
            "must be aligned 2D arrays"
        )
    agreement = None
    if scale_agreement is not None:
        agreement = np.asarray(scale_agreement, dtype=np.float32)
        if agreement.shape != score.shape:
            raise ValueError("scale_agreement must align with relative_score")

    empty_float = np.zeros(score.shape, dtype=np.float32)
    binary_skeleton = skeletonize(candidate).astype(np.uint8)
    if score.size == 0 or not np.any(candidate):
        return {
            "ribbon_centerline_mask": np.zeros(score.shape, dtype=np.uint8),
            "center_orientation": orientation.copy(),
            "center_confidence": empty_float,
            "orientation_clarity": empty_float.copy(),
            "center_preference": empty_float.copy(),
            "distance_transform": empty_float.copy(),
            "junction_zone_mask": np.zeros(score.shape, dtype=np.uint8),
            "diagnostics": {
                "ribbon_tracking_bridge_count": 0,
                "ribbon_tracking_bridge_pixel_count": 0,
                "ribbon_centerline_length": 0,
                "ribbon_centerline_total_length": 0.0,
                "ribbon_junction_count": 0,
                "ribbon_component_count": 0,
            },
        }

    distance = distance_transform_edt(candidate).astype(np.float32)
    smooth_score = cv2.GaussianBlur(
        score, (0, 0), sigmaX=0.8, sigmaY=0.8,
        borderType=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)
    evidence_weight = smooth_score * distance
    if agreement is not None:
        # Multi-scale agreement is a soft preference, never a hard gate.
        evidence_weight *= 0.5 + 0.5 * np.clip(agreement, 0.0, 1.0)
    doubled_cosine = cv2.GaussianBlur(
        np.cos(2.0 * orientation).astype(np.float32), (0, 0), 2.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    doubled_sine = cv2.GaussianBlur(
        np.sin(2.0 * orientation).astype(np.float32), (0, 0), 2.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    orientation_consistency = np.sqrt(
        doubled_cosine * doubled_cosine + doubled_sine * doubled_sine
    ).astype(np.float32)

    height, width = score.shape
    radius = max(2, int(cross_section_radius))
    tolerance = max(0.5, float(center_tolerance))
    center_offset = np.full(score.shape, np.inf, dtype=np.float32)
    center_confidence = np.zeros(score.shape, dtype=np.float32)
    candidate_u8 = candidate.astype(np.uint8)
    base_cols = np.arange(width, dtype=np.float32)[None, :]
    strip_height = 192
    for row0 in range(0, height, strip_height):
        row1 = min(row0 + strip_height, height)
        tangent = orientation[row0:row1]
        normal_row = np.cos(tangent).astype(np.float32)
        normal_col = -np.sin(tangent).astype(np.float32)
        base_rows = np.arange(row0, row1, dtype=np.float32)[:, None]
        shape = (row1 - row0, width)
        sum_weight = np.zeros(shape, dtype=np.float32)
        sum_offset = np.zeros(shape, dtype=np.float32)

        center_candidate = candidate[row0:row1]
        center_weight = evidence_weight[row0:row1]
        sum_weight[center_candidate] = center_weight[center_candidate]
        valid_positive = center_candidate.copy()
        valid_negative = center_candidate.copy()
        for step in range(1, radius + 1):
            for sign, valid in ((1.0, valid_positive), (-1.0, valid_negative)):
                map_rows = base_rows + sign * float(step) * normal_row
                map_cols = base_cols + sign * float(step) * normal_col
                sampled_candidate = cv2.remap(
                    candidate_u8,
                    map_cols,
                    map_rows,
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ) > 0
                valid &= sampled_candidate
                sampled_weight = cv2.remap(
                    evidence_weight,
                    map_cols,
                    map_rows,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                active_weight = sampled_weight * valid.astype(np.float32)
                sum_weight += active_weight
                sum_offset += sign * float(step) * active_weight

        valid_weight = center_candidate & (sum_weight > np.finfo(np.float32).eps)
        offset_strip = np.full(shape, np.inf, dtype=np.float32)
        offset_strip[valid_weight] = (
            sum_offset[valid_weight] / sum_weight[valid_weight]
        )
        center_offset[row0:row1] = offset_strip
        local_confidence = np.zeros(shape, dtype=np.float32)
        local_confidence[valid_weight] = (
            evidence_weight[row0:row1][valid_weight]
            / np.maximum(sum_weight[valid_weight], np.finfo(np.float32).eps)
        )
        center_confidence[row0:row1] = local_confidence

    centered_support = (
        candidate
        & (np.abs(center_offset) <= tolerance)
        & (orientation_consistency >= float(orientation_clarity))
    )
    ribbon = skeletonize(centered_support).astype(np.uint8)
    ribbon, tracking_diagnostics = _bridge_oriented_ribbon_gaps(
        ribbon,
        candidate,
        orientation,
        center_offset,
        max_gap=tracking_gap,
    )

    binary_adjacency = _skeleton_adjacency(binary_skeleton > 0)
    binary_junction_mask = np.zeros(score.shape, dtype=np.uint8)
    for point, neighbors in binary_adjacency.items():
        if len(neighbors) >= 3:
            binary_junction_mask[point] = 1
    local_radius = max(1, int(junction_radius))
    qualified_junction_mask = np.zeros(score.shape, dtype=np.uint8)
    approach_radius = local_radius * 2
    for row, col in np.column_stack(np.where(binary_junction_mask > 0)):
        row0, row1 = max(0, row - approach_radius), min(height, row + approach_radius + 1)
        col0, col1 = max(0, col - approach_radius), min(width, col + approach_radius + 1)
        local_ribbon = ribbon[row0:row1, col0:col1] > 0
        grid_rows, grid_cols = np.ogrid[row0:row1, col0:col1]
        radius_squared = (grid_rows - row) ** 2 + (grid_cols - col) ** 2
        annulus = (
            (radius_squared >= max(4, local_radius * local_radius))
            & (radius_squared <= approach_radius * approach_radius)
        )
        approach_count = cv2.connectedComponents(
            (local_ribbon & annulus).astype(np.uint8), 8
        )[0] - 1
        if approach_count >= 3:
            qualified_junction_mask[row, col] = 1
    junction_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (local_radius * 2 + 1, local_radius * 2 + 1)
    )
    junction_zone = cv2.dilate(qualified_junction_mask, junction_kernel) > 0
    # Fallback remains strictly local and must touch an already centered track;
    # an isolated noisy Binary junction cannot bootstrap its own centerline.
    ribbon_neighborhood = cv2.dilate(
        ribbon.astype(np.uint8), junction_kernel, iterations=2
    ) > 0
    ribbon |= (binary_skeleton > 0) & junction_zone & ribbon_neighborhood
    ribbon = skeletonize(ribbon > 0).astype(np.uint8)

    ribbon_adjacency = _skeleton_adjacency(ribbon > 0)
    ribbon_chains = _trace_skeleton_chains(ribbon > 0)
    total_length = float(sum(
        _relative_chain_geometry(path)["path_length"] for path in ribbon_chains
    ))
    component_count = (
        cv2.connectedComponents(ribbon, 8)[0] - 1 if np.any(ribbon) else 0
    )
    preference = evidence_weight.copy()
    positive_preference = preference[candidate]
    scale = float(np.percentile(positive_preference, 95.0)) if positive_preference.size else 0.0
    if scale > np.finfo(np.float32).eps:
        preference = np.clip(preference / scale, 0.0, 1.0)
    else:
        preference.fill(0.0)
    preference *= np.exp(-np.minimum(np.abs(center_offset), float(radius)))
    preference *= orientation_consistency
    preference[~candidate] = 0.0
    return {
        "ribbon_centerline_mask": ribbon,
        "center_orientation": orientation.copy(),
        "center_confidence": center_confidence,
        "orientation_clarity": orientation_consistency,
        "center_preference": preference.astype(np.float32),
        "distance_transform": distance,
        "junction_zone_mask": junction_zone.astype(np.uint8),
        "diagnostics": {
            **tracking_diagnostics,
            "ribbon_centerline_length": int(np.count_nonzero(ribbon)),
            "ribbon_centerline_total_length": total_length,
            "ribbon_junction_count": int(sum(
                len(neighbors) >= 3 for neighbors in ribbon_adjacency.values()
            )),
            "ribbon_component_count": int(component_count),
        },
    }


def _bilinear_samples(array, points):
    """Sample one aligned 2D map at subpixel row/column positions."""
    value = np.asarray(array, dtype=np.float32)
    locations = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    rows = np.clip(locations[:, 0], 0.0, float(value.shape[0] - 1))
    cols = np.clip(locations[:, 1], 0.0, float(value.shape[1] - 1))
    row0 = np.floor(rows).astype(np.int32)
    col0 = np.floor(cols).astype(np.int32)
    row1 = np.minimum(row0 + 1, value.shape[0] - 1)
    col1 = np.minimum(col0 + 1, value.shape[1] - 1)
    row_fraction = rows - row0
    col_fraction = cols - col0
    return (
        value[row0, col0] * (1.0 - row_fraction) * (1.0 - col_fraction)
        + value[row1, col0] * row_fraction * (1.0 - col_fraction)
        + value[row0, col1] * (1.0 - row_fraction) * col_fraction
        + value[row1, col1] * row_fraction * col_fraction
    ).astype(np.float32)


def _continuous_cross_section_center(
    predicted,
    tangent,
    candidate,
    center_weight,
    distance_support,
    *,
    radius,
):
    """Recenter one prediction inside its nearest contiguous ribbon interval."""
    point = np.asarray(predicted, dtype=np.float32)
    direction = np.asarray(tangent, dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.asarray([direction[1], -direction[0]], dtype=np.float32)
    offsets = np.arange(-float(radius), float(radius) + 0.25, 0.5, dtype=np.float32)
    samples = point[None, :] + offsets[:, None] * normal[None, :]
    rounded = np.rint(samples).astype(np.int32)
    inside = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < candidate.shape[0])
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < candidate.shape[1])
    )
    supported = np.zeros(len(samples), dtype=bool)
    valid_indices = np.where(inside)[0]
    supported[valid_indices] = candidate[
        rounded[valid_indices, 0], rounded[valid_indices, 1]
    ]
    if not np.any(supported):
        return None

    transitions = np.diff(np.pad(supported.astype(np.int8), (1, 1)))
    starts = np.where(transitions == 1)[0]
    stops = np.where(transitions == -1)[0]
    center_index = int(np.argmin(np.abs(offsets)))
    intervals = list(zip(starts.tolist(), stops.tolist()))
    containing = [item for item in intervals if item[0] <= center_index < item[1]]
    if containing:
        interval = containing[0]
    else:
        interval = min(
            intervals,
            key=lambda item: float(np.min(np.abs(offsets[item[0]:item[1]]))),
        )
        lateral_distance = float(np.min(np.abs(offsets[interval[0]:interval[1]])))
        if lateral_distance > min(6.0, float(radius) * 0.5):
            return None
    selected = slice(interval[0], interval[1])
    selected_points = samples[selected]
    selected_offsets = offsets[selected]
    weights = np.maximum(_bilinear_samples(center_weight, selected_points), 0.0)
    if float(np.sum(weights)) <= 1e-8:
        weights = np.maximum(
            _bilinear_samples(distance_support, selected_points), 0.0
        )
    if float(np.sum(weights)) <= 1e-8:
        weights = np.ones(len(selected_points), dtype=np.float32)
    center_offset = float(np.sum(weights * selected_offsets) / np.sum(weights))
    center = point + center_offset * normal
    return {
        "point": center.astype(np.float32),
        "weight": float(_bilinear_samples(center_weight, center[None, :])[0]),
        "distance": float(_bilinear_samples(distance_support, center[None, :])[0]),
        "interval_width": float(selected_offsets[-1] - selected_offsets[0] + 0.5),
    }


def _continuous_local_tangent(orientation, point, reference):
    """Return the modulo-pi local tangent with sign aligned to the trace."""
    location = np.asarray(point, dtype=np.float32).reshape(1, 2)
    cosine = float(_bilinear_samples(np.cos(2.0 * orientation), location)[0])
    sine = float(_bilinear_samples(np.sin(2.0 * orientation), location)[0])
    angle = 0.5 * math.atan2(sine, cosine)
    tangent = np.asarray([math.sin(angle), math.cos(angle)], dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if float(np.dot(tangent, reference)) < 0.0:
        tangent *= -1.0
    clarity = float(min(1.0, math.hypot(cosine, sine)))
    return tangent, clarity


def _rasterize_polyline_roi(points_rc, shape, *, thickness=1, padding=0):
    """Rasterize a polyline into its local bounding box, never a full-scene mask."""
    points = np.asarray(points_rc, dtype=np.int32).reshape(-1, 2)
    if len(points) == 0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=bool)
    thickness = max(1, int(thickness))
    margin = max(int(padding), (thickness + 1) // 2 + 1)
    row0 = max(0, int(np.min(points[:, 0])) - margin)
    row1 = min(int(shape[0]), int(np.max(points[:, 0])) + margin + 1)
    col0 = max(0, int(np.min(points[:, 1])) - margin)
    col1 = min(int(shape[1]), int(np.max(points[:, 1])) + margin + 1)
    local = np.zeros((row1 - row0, col1 - col0), dtype=np.uint8)
    shifted_xy = points[:, ::-1] - np.asarray([col0, row0], dtype=np.int32)
    cv2.polylines(
        local,
        [shifted_xy.reshape(-1, 1, 2)],
        False,
        1,
        thickness,
        cv2.LINE_8,
    )
    return slice(row0, row1), slice(col0, col1), local > 0


def trace_relative_ribbon_centerlines(
    relative_score,
    relative_candidate_mask,
    ridge_orientation,
    *,
    ridge_mask=None,
    ridge_strength=None,
    scale_agreement=None,
    distance_transform=None,
    center_preference=None,
    step_size=2.0,
    cross_section_radius=16,
    branch_persistence_length=24.0,
    long_trace_length=48.0,
):
    """Trace subpixel ribbon centers forward and backward from sparse seeds.

    Geometry is generated directly from continuous trajectories.  The
    previous trajectory direction predicts the next cross-section, whose
    nearest connected candidate interval is recentered by Relative-score x
    distance support.  No center mask is skeletonized and no global endpoint
    shortcut is drawn.
    """
    score = np.asarray(relative_score, dtype=np.float32)
    candidate = np.asarray(relative_candidate_mask, dtype=np.uint8) > 0
    orientation = np.asarray(ridge_orientation, dtype=np.float32)
    if score.ndim != 2 or candidate.shape != score.shape or orientation.shape != score.shape:
        raise ValueError("Continuous tracing inputs must be aligned 2D arrays")
    agreement = (
        np.ones(score.shape, dtype=np.float32)
        if scale_agreement is None else np.asarray(scale_agreement, dtype=np.float32)
    )
    if agreement.shape != score.shape:
        raise ValueError("scale_agreement must align with relative_score")
    distance = (
        distance_transform_edt(candidate).astype(np.float32)
        if distance_transform is None
        else np.asarray(distance_transform, dtype=np.float32)
    )
    if distance.shape != score.shape:
        raise ValueError("distance_transform must align with relative_score")
    ridge = (
        np.zeros(score.shape, dtype=np.uint8)
        if ridge_mask is None else (np.asarray(ridge_mask, dtype=np.uint8) > 0).astype(np.uint8)
    )
    strength = (
        ridge.astype(np.float32)
        if ridge_strength is None else np.asarray(ridge_strength, dtype=np.float32)
    )
    if ridge.shape != score.shape or strength.shape != score.shape:
        raise ValueError("ridge_mask and ridge_strength must align with relative_score")

    empty_u8 = np.zeros(score.shape, dtype=np.uint8)
    empty_i32 = np.zeros(score.shape, dtype=np.int32)
    if score.size == 0 or not np.any(candidate):
        return {
            "continuous_centerline_mask": empty_u8,
            "trace_id_map": empty_i32,
            "seed_mask": empty_u8.copy(),
            "junction_mask": empty_u8.copy(),
            "confirmed_branch_mask": empty_u8.copy(),
            "rejected_spur_mask": empty_u8.copy(),
            "traces": [],
            "diagnostics": {
                "continuous_trace_count": 0,
                "continuous_centerline_component_count": 0,
                "continuous_centerline_length": 0.0,
                "mean_trace_length": 0.0,
                "median_trace_length": 0.0,
                "long_trace_count": 0,
                "continuous_junction_count": 0,
                "confirmed_branch_count": 0,
                "seed_count": 0,
                "seed_suppressed_existing_trace_count": 0,
                "parallel_duplicate_rejected_count": 0,
                "parallel_duplicate_rejected_length": 0.0,
                "true_branch_count": 0,
                "collision_terminated_count": 0,
                "junction_supported_merge_count": 0,
                "rejected_spur_count": 0,
                "rejected_spur_length": 0.0,
                "continuous_termination_reason_counts": {},
                "relative_trace_summaries": [],
            },
        }

    smooth_score = cv2.GaussianBlur(
        score, (0, 0), 0.8, borderType=cv2.BORDER_REFLECT_101
    ).astype(np.float32)
    center_weight = smooth_score * np.maximum(distance, 0.0)
    center_weight *= 0.5 + 0.5 * np.clip(agreement, 0.0, 1.0)
    positive = center_weight[candidate]
    weight_scale = float(np.percentile(positive, 95.0)) if positive.size else 0.0
    normalized_weight = (
        np.clip(center_weight / weight_scale, 0.0, 1.0).astype(np.float32)
        if weight_scale > 1e-8 else np.zeros(score.shape, dtype=np.float32)
    )
    if center_preference is not None:
        preference = np.asarray(center_preference, dtype=np.float32)
        if preference.shape != score.shape:
            raise ValueError("center_preference must align with relative_score")
        normalized_weight = np.maximum(normalized_weight, 0.5 * np.clip(preference, 0.0, 1.0))
    normalized_weight[~candidate] = 0.0

    seed_rows = []
    ridge_chains = _trace_skeleton_chains(ridge > 0)
    strength_positive = strength[strength > 0]
    strength_scale = float(np.percentile(strength_positive, 95.0)) if strength_positive.size else 1.0
    seed_priority_map = normalized_weight + np.clip(strength / max(strength_scale, 1e-8), 0.0, 1.0)
    for path in ridge_chains:
        values = seed_priority_map[path[:, 0], path[:, 1]]
        path_widths = distance[path[:, 0], path[:, 1]]
        stable_path_width = float(np.median(path_widths)) if len(path_widths) else 0.0
        stable_seed = path_widths <= max(
            stable_path_width + 1.5, 1.35 * stable_path_width
        )
        if len(path) >= 12:
            indices = np.arange(len(path))
            stable_seed &= (
                (indices >= max(3, int(round(0.20 * len(path)))))
                & (indices <= min(len(path) - 4, int(round(0.80 * len(path)))))
            )
        eligible_values = np.where(stable_seed, values, -np.inf)
        if not np.any(np.isfinite(eligible_values)):
            eligible_values = values
        best_value = float(np.max(eligible_values))
        best_indices = np.where(eligible_values >= best_value - 1e-6)[0]
        index = int(best_indices[len(best_indices) // 2])
        point = tuple(map(int, path[index]))
        chain_length = float(_relative_chain_geometry(path)["path_length"])
        tangent_start = path[max(0, index - 3)].astype(np.float32)
        tangent_stop = path[min(len(path) - 1, index + 3)].astype(np.float32)
        chain_tangent = tangent_stop - tangent_start
        chain_tangent /= max(float(np.linalg.norm(chain_tangent)), 1e-6)
        seed_rows.append((
            best_value + 1.0 + min(chain_length / 48.0, 2.0),
            point,
            "ridge_chain",
            chain_length,
            chain_tangent,
        ))

    candidate_count, candidate_labels = cv2.connectedComponents(
        candidate.astype(np.uint8), 8
    )
    if candidate_count > 1:
        positions = maximum_position(
            normalized_weight,
            labels=candidate_labels,
            index=np.arange(1, candidate_count, dtype=np.int32),
        )
        if candidate_count == 2 and isinstance(positions, tuple):
            positions = [positions]
        for point in positions:
            row, col = map(int, point)
            seed_rows.append((
                float(normalized_weight[row, col]),
                (row, col),
                "component_preference",
                0.0,
                None,
            ))
    seed_rows.sort(key=lambda item: (-item[0], item[1][0], item[1][1], item[2]))

    centerline = np.zeros(score.shape, dtype=np.uint8)
    trace_ids = np.zeros(score.shape, dtype=np.int32)
    trace_tangent_angle = np.zeros(score.shape, dtype=np.float32)
    seed_mask = np.zeros(score.shape, dtype=np.uint8)
    confirmed_branch_mask = np.zeros(score.shape, dtype=np.uint8)
    rejected_spur_mask = np.zeros(score.shape, dtype=np.uint8)
    supported_junction_mask = np.zeros(score.shape, dtype=np.uint8)
    traces = []
    rejected_spur_lengths = []
    parallel_duplicate_lengths = []
    seed_suppressed_existing_trace_count = 0
    collision_terminated_count = 0
    junction_supported_merge_count = 0
    step = max(1.0, float(step_size))
    radius = max(3, int(cross_section_radius))
    maximum_steps = max(64, int(2.5 * sum(score.shape) / step))

    def seed_is_claimed_by_parallel_trace(seed_point, seed_tangent):
        row, col = np.rint(seed_point).astype(np.int32)
        local_half_width = max(2.0, float(distance[row, col]))
        search_radius = max(4, min(radius, int(math.ceil(1.25 * local_half_width))))
        row0, row1 = max(0, row - search_radius), min(score.shape[0], row + search_radius + 1)
        col0, col1 = max(0, col - search_radius), min(score.shape[1], col + search_radius + 1)
        local_ids = trace_ids[row0:row1, col0:col1]
        existing = np.column_stack(np.where(local_ids > 0))
        if not len(existing):
            return False
        existing += np.asarray([row0, col0], dtype=np.int32)
        offsets = existing.astype(np.float32) - np.asarray([row, col], dtype=np.float32)
        distances_to_seed = np.linalg.norm(offsets, axis=1)
        order = np.argsort(distances_to_seed)
        seed_component = int(candidate_labels[row, col])
        for index in order.tolist():
            if float(distances_to_seed[index]) > float(search_radius):
                break
            existing_row, existing_col = map(int, existing[index])
            if seed_component <= 0 or int(candidate_labels[existing_row, existing_col]) != seed_component:
                continue
            existing_angle = float(trace_tangent_angle[existing_row, existing_col])
            existing_tangent = np.asarray(
                [math.sin(existing_angle), math.cos(existing_angle)], dtype=np.float32
            )
            if abs(float(np.dot(existing_tangent, seed_tangent))) >= 0.82:
                return True
        return False

    def ray_support_fraction(point, direction, start_distance, stop_distance, mask):
        sample_count = max(3, int(math.ceil(stop_distance - start_distance)) + 1)
        offsets = np.linspace(start_distance, stop_distance, sample_count, dtype=np.float32)
        samples = np.asarray(point, dtype=np.float32)[None, :] + offsets[:, None] * np.asarray(
            direction, dtype=np.float32
        )[None, :]
        rounded = np.rint(samples).astype(np.int32)
        inside = (
            (rounded[:, 0] >= 0) & (rounded[:, 0] < score.shape[0])
            & (rounded[:, 1] >= 0) & (rounded[:, 1] < score.shape[1])
        )
        if np.count_nonzero(inside) < max(2, sample_count // 2):
            return 0.0
        rounded = rounded[inside]
        return float(np.mean(mask[rounded[:, 0], rounded[:, 1]] > 0))

    ridge_neighborhood = cv2.dilate(ridge, np.ones((11, 11), dtype=np.uint8)) > 0

    def junction_is_supported(point, incoming_tangent, existing_tangent, stable_width, expanded):
        """Validate a directional collision from local candidate/ridge geometry."""
        alignment = abs(float(np.dot(incoming_tangent, existing_tangent)))
        if alignment >= 0.72:
            return False
        probe_length = max(7.0, min(18.0, 2.2 * max(float(stable_width), 2.0)))
        probe_start = max(2.0, 0.45 * max(float(stable_width), 2.0))
        candidate_support = (
            ray_support_fraction(point, -incoming_tangent, probe_start, probe_length, candidate),
            ray_support_fraction(point, existing_tangent, probe_start, probe_length, candidate),
            ray_support_fraction(point, -existing_tangent, probe_start, probe_length, candidate),
        )
        if min(candidate_support) < 0.72:
            return False
        forward_support = ray_support_fraction(
            point, incoming_tangent, probe_start, probe_length, candidate
        )
        ridge_support = (
            ray_support_fraction(point, -incoming_tangent, 1.0, probe_length, ridge_neighborhood)
            >= 0.35
            and max(
                ray_support_fraction(point, existing_tangent, 1.0, probe_length, ridge_neighborhood),
                ray_support_fraction(point, -existing_tangent, 1.0, probe_length, ridge_neighborhood),
            ) >= 0.35
        )
        return bool(ridge_support or (expanded and forward_support >= 0.55))

    def existing_trace_relationship(rounded_points, point_tangents, trace_local_mask, row_slice, col_slice):
        """Measure proximity/alignment in a trace-local ROI."""
        rows = rounded_points[:, 0]
        cols = rounded_points[:, 1]
        widths = np.maximum(distance[rows, cols], 1.0)
        margin = max(4, min(radius + 2, int(math.ceil(float(np.percentile(widths, 90.0)) * 1.25))))
        row0 = max(0, int(np.min(rows)) - margin)
        row1 = min(score.shape[0], int(np.max(rows)) + margin + 1)
        col0 = max(0, int(np.min(cols)) - margin)
        col1 = min(score.shape[1], int(np.max(cols)) + margin + 1)
        existing = centerline[row0:row1, col0:col1] > 0
        point_weights = np.ones(len(rounded_points), dtype=np.float32)
        if len(rounded_points) > 1:
            segment_lengths = np.linalg.norm(np.diff(rounded_points.astype(np.float32), axis=0), axis=1)
            point_weights[0] = segment_lengths[0] * 0.5
            point_weights[-1] = segment_lengths[-1] * 0.5
            if len(rounded_points) > 2:
                point_weights[1:-1] = 0.5 * (segment_lengths[:-1] + segment_lengths[1:])
            point_weights = np.maximum(point_weights, 0.25)
        total_weight = max(float(np.sum(point_weights)), 1e-6)
        trace_pixel_count = int(np.count_nonzero(trace_local_mask))
        new_pixel_count = int(np.count_nonzero(
            trace_local_mask & (centerline[row_slice, col_slice] == 0)
        ))
        if not np.any(existing):
            return {
                "near_existing_fraction": 0.0,
                "parallel_fraction": 0.0,
                "independent_fraction": 1.0,
                "independent_length": total_weight,
                "new_pixel_fraction": float(new_pixel_count / max(trace_pixel_count, 1)),
                "new_pixel_count": new_pixel_count,
            }
        nearest_distance, nearest_indices = distance_transform_edt(
            ~existing, return_indices=True
        )
        local_rows = rows - row0
        local_cols = cols - col0
        distances_to_existing = nearest_distance[local_rows, local_cols]
        nearest_rows = nearest_indices[0, local_rows, local_cols] + row0
        nearest_cols = nearest_indices[1, local_rows, local_cols] + col0
        same_component = (
            candidate_labels[rows, cols] > 0
        ) & (
            candidate_labels[rows, cols] == candidate_labels[nearest_rows, nearest_cols]
        )
        near_thresholds = np.maximum(2.0, 0.90 * widths)
        near = same_component & (distances_to_existing <= near_thresholds)
        existing_angles = trace_tangent_angle[nearest_rows, nearest_cols]
        existing_tangents = np.column_stack((np.sin(existing_angles), np.cos(existing_angles)))
        alignment = np.abs(np.sum(existing_tangents * point_tangents, axis=1))
        parallel = near & (alignment >= 0.82)
        near_weight = float(np.sum(point_weights[near]))
        parallel_weight = float(np.sum(point_weights[parallel]))
        independent_weight = float(np.sum(point_weights[~near]))
        return {
            "near_existing_fraction": near_weight / total_weight,
            "parallel_fraction": parallel_weight / max(near_weight, 1e-6),
            "independent_fraction": independent_weight / total_weight,
            "independent_length": independent_weight,
            "new_pixel_fraction": float(new_pixel_count / max(trace_pixel_count, 1)),
            "new_pixel_count": new_pixel_count,
        }

    def robust_existing_tangent(existing_id, point, fallback_angle, width):
        row, col = map(int, point)
        tangent_radius = max(5, min(radius, int(math.ceil(max(float(width), 3.0)))))
        row0, row1 = max(0, row - tangent_radius), min(score.shape[0], row + tangent_radius + 1)
        col0, col1 = max(0, col - tangent_radius), min(score.shape[1], col + tangent_radius + 1)
        coordinates = np.column_stack(np.where(
            trace_ids[row0:row1, col0:col1] == int(existing_id)
        )).astype(np.float32)
        if len(coordinates) >= 3:
            coordinates -= np.mean(coordinates, axis=0, keepdims=True)
            covariance = coordinates.T @ coordinates
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if float(eigenvalues[-1]) > max(1e-6, 1.5 * float(eigenvalues[0])):
                tangent_value = eigenvectors[:, -1].astype(np.float32)
                tangent_value /= max(float(np.linalg.norm(tangent_value)), 1e-6)
                return tangent_value
        return np.asarray(
            [math.sin(fallback_angle), math.cos(fallback_angle)], dtype=np.float32
        )

    def trace_one_direction(seed, initial_tangent, _trace_id, initial_width):
        nonlocal collision_terminated_count
        points = [np.asarray(seed, dtype=np.float32)]
        tangent = np.asarray(initial_tangent, dtype=np.float32)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        local_pixels = {tuple(np.rint(points[0]).astype(np.int32).tolist()): 0}
        contacts = []
        reason = "candidate_end"
        preferences = []
        distances = []
        stable_width = max(2.0, float(initial_width))
        in_width_expansion = False
        for step_index in range(maximum_steps):
            centered = None
            used_lookahead = 0
            selected_prediction = None
            for lookahead in (1, 2, 3):
                predicted = points[-1] + float(lookahead) * step * tangent
                if not (
                    0.0 <= predicted[0] < score.shape[0]
                    and 0.0 <= predicted[1] < score.shape[1]
                ):
                    continue
                proposal = _continuous_cross_section_center(
                    predicted,
                    tangent,
                    candidate,
                    normalized_weight,
                    distance,
                    radius=radius,
                )
                if proposal is None:
                    continue
                first = np.rint(points[-1]).astype(np.int32)
                second = np.rint(proposal["point"]).astype(np.int32)
                rr, cc = line(int(first[0]), int(first[1]), int(second[0]), int(second[1]))
                rr = np.clip(rr, 0, score.shape[0] - 1)
                cc = np.clip(cc, 0, score.shape[1] - 1)
                support_fraction = float(np.mean(candidate[rr, cc])) if len(rr) else 0.0
                if support_fraction < (0.70 if lookahead == 1 else 0.50):
                    continue
                centered = proposal
                used_lookahead = lookahead
                selected_prediction = predicted
                break
            if centered is None:
                reason = "candidate_or_center_support_lost"
                break

            current_width = max(1.0, float(centered["interval_width"]))
            width_expanded = current_width >= 1.55 * stable_width
            if width_expanded:
                in_width_expansion = True
            elif in_width_expansion and current_width <= 1.25 * stable_width:
                in_width_expansion = False
            if not in_width_expansion:
                stable_width = 0.92 * stable_width + 0.08 * current_width

            next_point = centered["point"]
            if in_width_expansion and selected_prediction is not None:
                inertial_point = (
                    0.82 * np.asarray(selected_prediction, dtype=np.float32)
                    + 0.18 * np.asarray(next_point, dtype=np.float32)
                )
                inertial_pixel = np.rint(inertial_point).astype(np.int32)
                if (
                    0 <= inertial_pixel[0] < score.shape[0]
                    and 0 <= inertial_pixel[1] < score.shape[1]
                    and candidate[inertial_pixel[0], inertial_pixel[1]]
                ):
                    next_point = inertial_point
            actual = next_point - points[-1]
            actual_norm = float(np.linalg.norm(actual))
            if actual_norm <= 0.5:
                reason = "no_forward_progress"
                break
            actual /= actual_norm
            history_start = points[max(0, len(points) - 5)]
            history = next_point - history_start
            history /= max(float(np.linalg.norm(history)), 1e-6)
            if float(np.dot(history, tangent)) < 0.0:
                history *= -1.0
            local_tangent, clarity = _continuous_local_tangent(
                orientation, next_point, tangent
            )
            if in_width_expansion:
                updated = (
                    0.72 * tangent + 0.20 * history
                    + 0.06 * actual + 0.02 * clarity * local_tangent
                )
            else:
                updated = (
                    0.55 * tangent + 0.25 * history
                    + 0.15 * actual + 0.05 * clarity * local_tangent
                )
            updated /= max(float(np.linalg.norm(updated)), 1e-6)
            if float(np.dot(updated, tangent)) < 0.60:
                reason = "unreasonable_turn"
                break

            rounded = tuple(np.rint(next_point).astype(np.int32).tolist())
            previous_visit = local_pixels.get(rounded)
            if previous_visit is not None and step_index - previous_visit > 8:
                points.append(next_point)
                reason = "loop_closed"
                break

            row, col = rounded
            row0, row1 = max(0, row - 2), min(score.shape[0], row + 3)
            col0, col1 = max(0, col - 2), min(score.shape[1], col + 3)
            nearby_ids = trace_ids[row0:row1, col0:col1]
            existing_ids = np.unique(nearby_ids[nearby_ids > 0])
            merge_id = 0
            junction_id = 0
            unsupported_collision = False
            for existing_id in existing_ids.tolist():
                existing_points = np.column_stack(np.where(nearby_ids == existing_id))
                offsets = existing_points - np.asarray([row - row0, col - col0])
                existing_local = existing_points[int(np.argmin(np.sum(offsets * offsets, axis=1)))]
                existing_local = existing_local + np.asarray([row0, col0])
                existing_angle = float(trace_tangent_angle[tuple(existing_local)])
                existing_tangent = robust_existing_tangent(
                    existing_id, existing_local, existing_angle, stable_width
                )
                alignment = abs(float(np.dot(existing_tangent, updated)))
                if alignment >= 0.72:
                    merge_id = int(existing_id)
                    next_point = existing_local.astype(np.float32)
                    rounded = tuple(map(int, existing_local))
                    contacts.append({
                        "trace_id": int(existing_id),
                        "alignment": alignment,
                        "kind": "aligned_merge",
                        "point": tuple(map(int, existing_local)),
                    })
                    break
                if junction_is_supported(
                    existing_local,
                    updated,
                    existing_tangent,
                    stable_width,
                    in_width_expansion,
                ):
                    junction_id = int(existing_id)
                    next_point = existing_local.astype(np.float32)
                    rounded = tuple(map(int, existing_local))
                    contacts.append({
                        "trace_id": int(existing_id),
                        "alignment": alignment,
                        "kind": "junction_supported",
                        "point": tuple(map(int, existing_local)),
                    })
                else:
                    unsupported_collision = True
                break

            if unsupported_collision:
                collision_terminated_count += 1
                reason = "collision_terminated"
                break

            points.append(next_point.astype(np.float32))
            preferences.append(float(centered["weight"]))
            distances.append(float(centered["distance"]))
            local_pixels[rounded] = step_index
            tangent = updated
            if merge_id:
                reason = "merged_existing_trace"
                break
            if junction_id:
                reason = "junction_supported_merge"
                break
            if used_lookahead > 1:
                reason = "local_lookahead"
        else:
            reason = "maximum_steps"
        return points, tangent, reason, contacts, preferences, distances

    for _priority, seed_rc, seed_source, seed_chain_length, seed_tangent in seed_rows:
        seed_row, seed_col = seed_rc
        if not candidate[seed_row, seed_col]:
            continue
        initial = (
            np.asarray(seed_tangent, dtype=np.float32)
            if seed_tangent is not None
            else np.asarray(
                [math.sin(float(orientation[seed_row, seed_col])),
                 math.cos(float(orientation[seed_row, seed_col]))],
                dtype=np.float32,
            )
        )
        seed_center = _continuous_cross_section_center(
            np.asarray(seed_rc, dtype=np.float32),
            initial,
            candidate,
            normalized_weight,
            distance,
            radius=radius,
        )
        if seed_center is None:
            continue
        seed = seed_center["point"]
        if seed_is_claimed_by_parallel_trace(seed, initial):
            seed_suppressed_existing_trace_count += 1
            if (
                seed_source == "ridge_chain"
                and 0.0 < float(seed_chain_length) < float(branch_persistence_length)
            ):
                rejected_spur_lengths.append(float(seed_chain_length))
            continue
        next_trace_id = len(traces) + 1
        initial_width = max(2.0, float(seed_center["interval_width"]))
        forward, forward_tangent, forward_reason, forward_contacts, forward_pref, forward_dist = (
            trace_one_direction(seed, initial, next_trace_id, initial_width)
        )
        backward, backward_tangent, backward_reason, backward_contacts, backward_pref, backward_dist = (
            trace_one_direction(seed, -initial, next_trace_id, initial_width)
        )
        points = np.asarray(backward[:0:-1] + forward, dtype=np.float32)
        if len(points) < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        trace_length = float(np.sum(segment_lengths))
        rounded_points = np.rint(points).astype(np.int32)
        rounded_points[:, 0] = np.clip(rounded_points[:, 0], 0, score.shape[0] - 1)
        rounded_points[:, 1] = np.clip(rounded_points[:, 1], 0, score.shape[1] - 1)
        row_slice, col_slice, trace_local_mask = _rasterize_polyline_roi(
            rounded_points, score.shape
        )
        point_tangents = np.gradient(points, axis=0)
        point_tangents /= np.maximum(
            np.linalg.norm(point_tangents, axis=1, keepdims=True), 1e-6
        )
        relationship = existing_trace_relationship(
            rounded_points,
            point_tangents,
            trace_local_mask,
            row_slice,
            col_slice,
        )
        is_persistent = trace_length >= float(branch_persistence_length)
        if not is_persistent:
            if trace_length > 0.0:
                rejected_spur_mask[row_slice, col_slice][trace_local_mask] = 1
                rejected_spur_lengths.append(trace_length)
            continue

        junction_contacts = [
            row for row in (forward_contacts + backward_contacts)
            if row["kind"] == "junction_supported"
        ]
        aligned_contacts = [
            row for row in (forward_contacts + backward_contacts)
            if row["kind"] == "aligned_merge"
        ]
        maximum_contact_alignment = max(
            (float(row["alignment"]) for row in junction_contacts), default=1.0
        )
        is_true_branch = bool(
            junction_contacts
            and maximum_contact_alignment <= 0.72
            and relationship["independent_length"] >= float(branch_persistence_length)
            and relationship["independent_fraction"] >= 0.45
            and relationship["parallel_fraction"] <= 0.45
        )
        is_parallel_duplicate = bool(
            relationship["near_existing_fraction"] >= 0.70
            and relationship["parallel_fraction"] >= 0.75
            and relationship["independent_fraction"] <= 0.30
        )
        if is_parallel_duplicate:
            parallel_duplicate_lengths.append(trace_length)
            continue
        if junction_contacts and not is_true_branch:
            rejected_spur_mask[row_slice, col_slice][trace_local_mask] = 1
            rejected_spur_lengths.append(trace_length)
            continue

        new_pixel_count = int(relationship["new_pixel_count"])
        if new_pixel_count < 2:
            parallel_duplicate_lengths.append(trace_length)
            continue

        trace_id = len(traces) + 1
        seed_pixel = np.rint(seed).astype(np.int32)
        seed_mask[seed_pixel[0], seed_pixel[1]] = 1
        local_trace_ids = trace_ids[row_slice, col_slice]
        local_centerline = centerline[row_slice, col_slice]
        new_pixels = trace_local_mask & (local_trace_ids == 0)
        local_centerline[trace_local_mask] = 1
        local_trace_ids[new_pixels] = trace_id
        angles = np.arctan2(point_tangents[:, 0], point_tangents[:, 1])
        for start, stop, angle in zip(rounded_points[:-1], rounded_points[1:], angles[:-1]):
            rr, cc = line(int(start[0]), int(start[1]), int(stop[0]), int(stop[1]))
            writable = trace_ids[rr, cc] == trace_id
            trace_tangent_angle[rr[writable], cc[writable]] = float(angle)
        branch_parent = int(junction_contacts[0]["trace_id"]) if is_true_branch else 0
        if is_true_branch:
            confirmed_branch_mask[row_slice, col_slice][new_pixels] = 1
            for contact in junction_contacts:
                supported_junction_mask[contact["point"]] = 1
            junction_supported_merge_count += len(junction_contacts)
        all_preferences = forward_pref + backward_pref
        all_distances = forward_dist + backward_dist
        rows = rounded_points[:, 0]
        cols = rounded_points[:, 1]
        traces.append({
            "trace_id": trace_id,
            "length": trace_length,
            "seed_source": seed_source,
            "mean_relative_score": float(np.mean(score[rows, cols])),
            "mean_center_preference": float(np.mean(all_preferences)) if all_preferences else 0.0,
            "mean_distance_support": float(np.mean(all_distances)) if all_distances else 0.0,
            "branch_parent": branch_parent,
            "classification": (
                "true_branch" if is_true_branch
                else "merge" if aligned_contacts else "independent"
            ),
            "near_existing_fraction": float(relationship["near_existing_fraction"]),
            "parallel_fraction": float(relationship["parallel_fraction"]),
            "independent_fraction": float(relationship["independent_fraction"]),
            "new_pixel_fraction": float(relationship["new_pixel_fraction"]),
            "termination_reason": f"backward:{backward_reason};forward:{forward_reason}",
        })

    component_count = (
        int(cv2.connectedComponents(centerline, 8)[0] - 1) if np.any(centerline) else 0
    )
    junction_mask = supported_junction_mask
    junction_count = (
        int(cv2.connectedComponents(junction_mask, 8)[0] - 1)
        if np.any(junction_mask) else 0
    )
    centerline_chains = _trace_skeleton_chains(centerline > 0)
    centerline_length = float(sum(
        _relative_chain_geometry(path)["path_length"] for path in centerline_chains
    ))
    trace_lengths = [float(row["length"]) for row in traces]
    termination_counts = Counter()
    for row in traces:
        for reason in str(row["termination_reason"]).split(";"):
            termination_counts[reason.split(":", 1)[-1]] += 1
    return {
        "continuous_centerline_mask": centerline,
        "trace_id_map": trace_ids,
        "seed_mask": seed_mask,
        "junction_mask": junction_mask,
        "confirmed_branch_mask": confirmed_branch_mask,
        "rejected_spur_mask": rejected_spur_mask,
        "traces": traces,
        "diagnostics": {
            "continuous_trace_count": int(len(traces)),
            "continuous_centerline_component_count": int(component_count),
            "continuous_centerline_length": centerline_length,
            "mean_trace_length": float(np.mean(trace_lengths)) if trace_lengths else 0.0,
            "median_trace_length": float(np.median(trace_lengths)) if trace_lengths else 0.0,
            "long_trace_count": int(sum(
                length >= float(long_trace_length) for length in trace_lengths
            )),
            "continuous_junction_count": int(junction_count),
            "confirmed_branch_count": int(sum(
                int(row["branch_parent"]) > 0 for row in traces
            )),
            "seed_count": int(len(seed_rows)),
            "seed_suppressed_existing_trace_count": int(
                seed_suppressed_existing_trace_count
            ),
            "parallel_duplicate_rejected_count": int(len(parallel_duplicate_lengths)),
            "parallel_duplicate_rejected_length": float(sum(parallel_duplicate_lengths)),
            "true_branch_count": int(sum(
                row.get("classification") == "true_branch" for row in traces
            )),
            "collision_terminated_count": int(collision_terminated_count),
            "junction_supported_merge_count": int(junction_supported_merge_count),
            "rejected_spur_count": int(len(rejected_spur_lengths)),
            "rejected_spur_length": float(sum(rejected_spur_lengths)),
            "continuous_termination_reason_counts": dict(termination_counts),
            "relative_trace_summaries": traces,
        },
    }


def _component_half_widths(labels, stats, distance):
    """Estimate stable ribbon half-widths from local distance maxima."""
    result = {}
    dilated = cv2.dilate(distance.astype(np.float32), np.ones((3, 3), dtype=np.uint8))
    for label_id in range(1, int(stats.shape[0])):
        col, row, width, height, _area = map(int, stats[label_id])
        local_labels = labels[row:row + height, col:col + width]
        local_distance = distance[row:row + height, col:col + width]
        member = local_labels == label_id
        maxima = member & (local_distance >= dilated[row:row + height, col:col + width] - 1e-4)
        values = local_distance[maxima]
        if values.size < 3:
            values = local_distance[member]
        stable = float(np.median(values)) if values.size else 1.0
        result[label_id] = max(1.0, stable)
    return result


def regularize_relative_candidate(
    candidate_mask,
    *,
    sigma_factor=0.20,
    sigma_min=0.8,
    sigma_max=2.5,
    hole_width_ratio=0.45,
    hole_size_ratio=0.75,
):
    """Regularize road ribbons with component-scale signed-distance smoothing."""
    started = time.perf_counter()
    candidate = np.asarray(candidate_mask, dtype=np.uint8) > 0
    before_count = int(np.count_nonzero(candidate))
    performance = {
        "distance_transform_seconds": 0.0,
        "hole_detection_seconds": 0.0,
        "narrow_hole_analysis_seconds": 0.0,
        "hole_fill_seconds": 0.0,
        "smoothing_seconds": 0.0,
    }
    if not np.any(candidate):
        empty = np.zeros(candidate.shape, dtype=np.uint8)
        performance.update({
            "candidate_regularization_seconds": time.perf_counter() - started,
            "candidate_pixel_count_before": before_count,
            "candidate_pixel_count_after": 0,
            "hole_filled_count": 0,
            "hole_count": 0,
            "hole_preserved_count": 0,
            "narrow_hole_count": 0,
            "narrow_hole_filled_count": 0,
            "hole_filled_by_size_count": 0,
            "hole_filled_by_narrow_width_count": 0,
            "noise_component_removed_count": 0,
        })
        return {
            "regularized_candidate": empty,
            "distance_transform": np.zeros(candidate.shape, dtype=np.float32),
            "component_half_widths": {},
            "detected_hole_mask": empty.copy(),
            "filled_hole_mask": empty.copy(),
            "preserved_hole_mask": empty.copy(),
            "narrow_hole_mask": empty.copy(),
            "performance": performance,
        }

    distance_started = time.perf_counter()
    distance = distance_transform_edt(candidate).astype(np.float32)
    performance["distance_transform_seconds"] += time.perf_counter() - distance_started
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    half_widths = _component_half_widths(labels, stats, distance)

    cleaned = candidate.copy()
    removed_noise = 0
    for label_id in range(1, count):
        col, row, width, height, area = map(int, stats[label_id])
        half_width = half_widths.get(label_id, 1.0)
        elongation = float(max(width, height) / max(1, min(width, height)))
        small_area = float(area) < max(4.0, 3.0 * half_width * half_width)
        if small_area and elongation < 2.5:
            local_labels = labels[row:row + height, col:col + width]
            local_cleaned = cleaned[row:row + height, col:col + width]
            local_cleaned[local_labels == label_id] = False
            removed_noise += 1

    hole_started = time.perf_counter()
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8), 8
    )
    distance_started = time.perf_counter()
    distance = distance_transform_edt(cleaned).astype(np.float32)
    performance["distance_transform_seconds"] += time.perf_counter() - distance_started
    half_widths = _component_half_widths(labels, stats, distance)
    hole_detection_started = time.perf_counter()
    background_count, background_labels, background_stats, _ = cv2.connectedComponentsWithStats(
        (~cleaned).astype(np.uint8), 8
    )
    border_labels = set(np.unique(np.concatenate((
        background_labels[0], background_labels[-1],
        background_labels[:, 0], background_labels[:, -1],
    ))).tolist())
    performance["hole_detection_seconds"] = time.perf_counter() - hole_detection_started
    detected_holes = np.zeros(candidate.shape, dtype=np.uint8)
    filled_holes = np.zeros(candidate.shape, dtype=np.uint8)
    preserved_holes = np.zeros(candidate.shape, dtype=np.uint8)
    narrow_holes = np.zeros(candidate.shape, dtype=np.uint8)
    hole_count = 0
    hole_filled_count = 0
    narrow_hole_count = 0
    narrow_hole_filled_count = 0
    filled_by_size_count = 0
    filled_by_narrow_count = 0
    narrow_started = time.perf_counter()
    for hole_id in range(1, background_count):
        if hole_id in border_labels:
            continue
        col, row, width, height, area = map(int, background_stats[hole_id])
        if area <= 0:
            continue
        hole_count += 1
        padding = 1
        row0, row1 = max(0, row - padding), min(candidate.shape[0], row + height + padding)
        col0, col1 = max(0, col - padding), min(candidate.shape[1], col + width + padding)
        local_background_labels = background_labels[row0:row1, col0:col1]
        local_hole = local_background_labels == hole_id
        detected_holes[row0:row1, col0:col1][local_hole] = 1
        local_cleaned = cleaned[row0:row1, col0:col1]
        boundary = (
            cv2.dilate(local_hole.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
        ) & ~local_hole
        local_labels = labels[row0:row1, col0:col1]
        neighbor_ids = local_labels[boundary & local_cleaned]
        neighbor_ids = neighbor_ids[neighbor_ids > 0]
        if not neighbor_ids.size:
            preserved_holes[row0:row1, col0:col1][local_hole] = 1
            continue
        host_id = int(np.bincount(neighbor_ids).argmax())
        host_half_width = half_widths.get(host_id, 1.0)
        scale_padding = max(3, int(math.ceil(2.5 * host_half_width)))
        sample_row0 = max(0, row - scale_padding)
        sample_row1 = min(candidate.shape[0], row + height + scale_padding)
        sample_col0 = max(0, col - scale_padding)
        sample_col1 = min(candidate.shape[1], col + width + scale_padding)
        sample_labels = labels[sample_row0:sample_row1, sample_col0:sample_col1]
        sample_distance = distance[sample_row0:sample_row1, sample_col0:sample_col1]
        local_host_values = sample_distance[sample_labels == host_id]
        local_half_width = (
            float(np.percentile(local_host_values, 90.0))
            if local_host_values.size else float(host_half_width)
        )
        road_width = 2.0 * max(1.0, local_half_width)
        hole_dt = distance_transform_edt(local_hole).astype(np.float32)
        hole_values = hole_dt[local_hole]
        hole_half_width = (
            float(np.percentile(hole_values, 95.0)) if hole_values.size else 0.0
        )
        actual_hole_width = 2.0 * hole_half_width
        width_ratio = actual_hole_width / max(road_width, 1e-6)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        equivalent_diameter = 2.0 * math.sqrt(float(area) / math.pi)
        fill_by_size = equivalent_diameter < float(hole_size_ratio) * road_width
        fill_by_narrow_width = (
            aspect_ratio >= 2.0 and width_ratio < float(hole_width_ratio)
        )
        if fill_by_narrow_width:
            narrow_holes[row0:row1, col0:col1][local_hole] = 1
            narrow_hole_count += 1
        if fill_by_size or fill_by_narrow_width:
            local_cleaned[local_hole] = True
            filled_holes[row0:row1, col0:col1][local_hole] = 1
            hole_filled_count += 1
            if fill_by_size:
                filled_by_size_count += 1
            if fill_by_narrow_width:
                filled_by_narrow_count += 1
                narrow_hole_filled_count += 1
        else:
            preserved_holes[row0:row1, col0:col1][local_hole] = 1
    performance["narrow_hole_analysis_seconds"] = time.perf_counter() - narrow_started
    performance["hole_fill_seconds"] = time.perf_counter() - hole_started

    smoothing_started = time.perf_counter()
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8), 8
    )
    distance_started = time.perf_counter()
    distance = distance_transform_edt(cleaned).astype(np.float32)
    performance["distance_transform_seconds"] += time.perf_counter() - distance_started
    half_widths = _component_half_widths(labels, stats, distance)
    regularized = np.zeros(candidate.shape, dtype=bool)
    for label_id in range(1, count):
        col, row, width, height, _area = map(int, stats[label_id])
        half_width = half_widths.get(label_id, 1.0)
        sigma = float(np.clip(sigma_factor * half_width, sigma_min, sigma_max))
        padding = max(3, int(math.ceil(4.0 * sigma)))
        row0, row1 = max(0, row - padding), min(candidate.shape[0], row + height + padding)
        col0, col1 = max(0, col - padding), min(candidate.shape[1], col + width + padding)
        local_component = labels[row0:row1, col0:col1] == label_id
        inside = distance_transform_edt(local_component).astype(np.float32)
        outside = distance_transform_edt(~local_component).astype(np.float32)
        signed_distance = inside - outside
        smoothed = cv2.GaussianBlur(
            signed_distance,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        regularized[row0:row1, col0:col1] |= smoothed > 0.0

    # A regularizer must never connect distinct input components. If two
    # independently smoothed ribbons touch, restore their original pixels.
    output_count, output_labels, output_stats, _ = cv2.connectedComponentsWithStats(
        regularized.astype(np.uint8), 8
    )
    for output_id in range(1, output_count):
        col, row, width, height, _area = map(int, output_stats[output_id])
        local_output_labels = output_labels[row:row + height, col:col + width]
        output_region = local_output_labels == output_id
        local_source_labels = labels[row:row + height, col:col + width]
        source_ids = np.unique(local_source_labels[output_region & (local_source_labels > 0)])
        if len(source_ids) > 1:
            local_regularized = regularized[row:row + height, col:col + width]
            local_regularized[output_region] = False
            local_regularized[np.isin(local_source_labels, source_ids)] = True
    performance["smoothing_seconds"] = time.perf_counter() - smoothing_started

    distance_started = time.perf_counter()
    final_distance = distance_transform_edt(regularized).astype(np.float32)
    performance["distance_transform_seconds"] += time.perf_counter() - distance_started
    final_count, final_labels, final_stats, _ = cv2.connectedComponentsWithStats(
        regularized.astype(np.uint8), 8
    )
    final_widths = _component_half_widths(final_labels, final_stats, final_distance)
    performance.update({
        "candidate_regularization_seconds": time.perf_counter() - started,
        "candidate_pixel_count_before": before_count,
        "candidate_pixel_count_after": int(np.count_nonzero(regularized)),
        "hole_count": int(hole_count),
        "hole_filled_count": int(hole_filled_count),
        "hole_preserved_count": int(hole_count - hole_filled_count),
        "narrow_hole_count": int(narrow_hole_count),
        "narrow_hole_filled_count": int(narrow_hole_filled_count),
        "hole_filled_by_size_count": int(filled_by_size_count),
        "hole_filled_by_narrow_width_count": int(filled_by_narrow_count),
        "noise_component_removed_count": int(removed_noise),
    })
    return {
        "regularized_candidate": regularized.astype(np.uint8),
        "distance_transform": final_distance,
        "component_half_widths": final_widths,
        "detected_hole_mask": detected_holes,
        "filled_hole_mask": filled_holes,
        "preserved_hole_mask": preserved_holes,
        "narrow_hole_mask": narrow_holes,
        "performance": performance,
    }


def prune_width_aware_spurs(skeleton, distance_transform, *, spur_ratio=2.25, max_iterations=3):
    """Remove only short endpoint-to-junction chains using L / local R."""
    value = np.asarray(skeleton, dtype=np.uint8) > 0
    distance = np.asarray(distance_transform, dtype=np.float32)
    removed = np.zeros(value.shape, dtype=np.uint8)
    removed_lengths = []
    for _iteration in range(max(1, int(max_iterations))):
        adjacency = _skeleton_adjacency(value)
        endpoints = [point for point, neighbors in adjacency.items() if len(neighbors) == 1]
        to_remove = []
        iteration_lengths = []
        visited_edges = set()
        for endpoint in endpoints:
            path = [endpoint]
            previous = None
            current = endpoint
            while True:
                following = [item for item in adjacency.get(current, []) if item != previous]
                if len(following) != 1:
                    break
                nxt = following[0]
                edge = tuple(sorted((current, nxt)))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                path.append(nxt)
                previous, current = current, nxt
                if len(adjacency.get(current, [])) != 2:
                    break
            if len(path) < 2 or len(adjacency.get(path[-1], [])) < 3:
                continue
            points = np.asarray(path, dtype=np.int32)
            length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            junction_row, junction_col = path[-1]
            local_half_width = max(1.0, float(distance[junction_row, junction_col]))
            if length < float(spur_ratio) * local_half_width:
                to_remove.extend(path[:-1])
                iteration_lengths.append(length)
        if not to_remove:
            break
        rows, cols = np.asarray(sorted(set(to_remove)), dtype=np.int32).T
        value[rows, cols] = False
        removed[rows, cols] = 1
        removed_lengths.extend(iteration_lengths)
    return {
        "pruned_skeleton": value.astype(np.uint8),
        "removed_spur_mask": removed,
        "spur_removed_count": int(len(removed_lengths)),
        "spur_removed_length": float(sum(removed_lengths)),
    }


def _chain_level_cycles(skeleton):
    """Return a fundamental cycle basis over topology-bounded skeleton chains."""
    chains = _trace_skeleton_chains(np.asarray(skeleton, dtype=np.uint8) > 0)
    endpoints = [
        (tuple(map(int, path[0])), tuple(map(int, path[-1]))) for path in chains
    ]
    parent = {}
    tree = defaultdict(list)
    cycles = []

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def tree_path(start, goal):
        queue = deque([start])
        previous = {start: (None, None)}
        while queue:
            node = queue.popleft()
            if node == goal:
                break
            for neighbor, edge_id in tree[node]:
                if neighbor not in previous:
                    previous[neighbor] = (node, edge_id)
                    queue.append(neighbor)
        if goal not in previous:
            return []
        result = []
        node = goal
        while previous[node][0] is not None:
            node, edge_id = previous[node]
            result.append(int(edge_id))
        result.reverse()
        return result

    for edge_id, (first, second) in enumerate(endpoints):
        if first == second:
            cycles.append([edge_id])
            continue
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root
            tree[first].append((second, edge_id))
            tree[second].append((first, edge_id))
        else:
            path_edges = tree_path(first, second)
            if path_edges:
                cycles.append(path_edges + [edge_id])
    return chains, endpoints, cycles


def remove_width_scale_small_cycles(
    skeleton,
    distance_transform,
    *,
    relative_score=None,
    max_span_ratio=4.5,
    max_area_ratio=8.0,
    max_perimeter_ratio=13.0,
):
    """Open only road-width-scale cycles by removing their weaker local chain."""
    started = time.perf_counter()
    value = np.asarray(skeleton, dtype=np.uint8) > 0
    distance = np.asarray(distance_transform, dtype=np.float32)
    score = (
        np.asarray(relative_score, dtype=np.float32)
        if relative_score is not None else np.zeros(value.shape, dtype=np.float32)
    )
    detection_started = time.perf_counter()
    chains, endpoints, cycles = _chain_level_cycles(value)
    detection_seconds = time.perf_counter() - detection_started
    detected_mask = np.zeros(value.shape, dtype=np.uint8)
    removed_mask = np.zeros(value.shape, dtype=np.uint8)
    edge_removed = set()
    cycle_records = []
    small_cycle_count = 0
    removed_cycle_count = 0

    edge_graph = defaultdict(list)
    for edge_id, (first, second) in enumerate(endpoints):
        if first != second:
            edge_graph[first].append((second, edge_id))
            edge_graph[second].append((first, edge_id))

    def has_alternate_path(start, goal, excluded_edge):
        queue = deque([start])
        visited = {start}
        while queue:
            node = queue.popleft()
            if node == goal:
                return True
            for neighbor, edge_id in edge_graph[node]:
                if edge_id == excluded_edge or edge_id in edge_removed:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    cleanup_started = time.perf_counter()
    for cycle_edge_ids in cycles:
        points = np.concatenate([chains[index] for index in cycle_edge_ids], axis=0)
        unique_points = np.unique(points, axis=0)
        rows, cols = unique_points[:, 0], unique_points[:, 1]
        local_half_width = max(1.0, float(np.median(distance[rows, cols])))
        road_width = 2.0 * local_half_width
        bbox_width = int(cols.max() - cols.min() + 1)
        bbox_height = int(rows.max() - rows.min() + 1)
        span = float(max(bbox_width, bbox_height))
        hull = cv2.convexHull(unique_points[:, ::-1].astype(np.float32))
        area = float(abs(cv2.contourArea(hull))) if len(hull) >= 3 else 0.0
        perimeter = float(sum(
            _relative_chain_geometry(chains[index])["path_length"]
            for index in cycle_edge_ids
        ))
        span_ratio = span / max(road_width, 1e-6)
        area_ratio = area / max(road_width * road_width, 1e-6)
        perimeter_ratio = perimeter / max(road_width, 1e-6)
        is_small = (
            span_ratio <= float(max_span_ratio)
            and area_ratio <= float(max_area_ratio)
            and perimeter_ratio <= float(max_perimeter_ratio)
        )
        record = {
            "edge_count": int(len(cycle_edge_ids)),
            "perimeter": float(perimeter),
            "bbox_width": int(bbox_width),
            "bbox_height": int(bbox_height),
            "cycle_span": float(span),
            "cycle_area": float(area),
            "local_road_width": float(road_width),
            "cycle_span_ratio": float(span_ratio),
            "cycle_area_ratio": float(area_ratio),
            "cycle_perimeter_ratio": float(perimeter_ratio),
            "small_cycle": bool(is_small),
            "removed": False,
        }
        cycle_records.append(record)
        if not is_small:
            continue
        small_cycle_count += 1
        detected_mask[rows, cols] = 1
        if len(cycle_edge_ids) == 1:
            edge_id = cycle_edge_ids[0]
            path = chains[edge_id]
            first, second = endpoints[edge_id]
            if first == second and len(path) > 5:
                unique_path = path[:-1]
                path_scores = distance[unique_path[:, 0], unique_path[:, 1]]
                path_scores = path_scores + 0.15 * score[
                    unique_path[:, 0], unique_path[:, 1]
                ]
                cut_index = int(np.argmin(path_scores))
                cut_indices = [
                    (cut_index + offset) % len(unique_path) for offset in (-1, 0, 1)
                ]
                cut_points = unique_path[np.asarray(cut_indices, dtype=np.int32)]
                value[cut_points[:, 0], cut_points[:, 1]] = False
                removed_mask[cut_points[:, 0], cut_points[:, 1]] = 1
                edge_removed.add(edge_id)
                removed_cycle_count += 1
                record["removed"] = True
                continue
        removable = []
        for edge_id in cycle_edge_ids:
            path = chains[edge_id]
            first, second = endpoints[edge_id]
            if first == second or len(path) <= 2 or edge_id in edge_removed:
                continue
            path_rows, path_cols = path[:, 0], path[:, 1]
            edge_score = float(np.mean(distance[path_rows, path_cols]))
            edge_score += 0.15 * float(np.mean(score[path_rows, path_cols]))
            removable.append((edge_score, len(path), edge_id))
        removable.sort()
        for _edge_score, _length, edge_id in removable:
            first, second = endpoints[edge_id]
            if not has_alternate_path(first, second, edge_id):
                continue
            path = chains[edge_id]
            interior = path[1:-1]
            value[interior[:, 0], interior[:, 1]] = False
            removed_mask[interior[:, 0], interior[:, 1]] = 1
            edge_removed.add(edge_id)
            removed_cycle_count += 1
            record["removed"] = True
            break
    cleanup_seconds = time.perf_counter() - cleanup_started
    final_detection_started = time.perf_counter()
    _final_chains, _final_endpoints, final_cycles = _chain_level_cycles(value)
    detection_seconds += time.perf_counter() - final_detection_started
    return {
        "cleaned_skeleton": value.astype(np.uint8),
        "detected_cycle_mask": detected_mask,
        "removed_cycle_mask": removed_mask,
        "cycle_detection_seconds": float(detection_seconds),
        "cycle_cleanup_seconds": float(cleanup_seconds),
        "raw_cycle_count": int(len(cycles)),
        "small_cycle_count": int(small_cycle_count),
        "removed_cycle_count": int(removed_cycle_count),
        "preserved_cycle_count": int(len(cycles) - removed_cycle_count),
        "final_cycle_count": int(len(final_cycles)),
        "cycle_records": cycle_records,
        "total_seconds": float(time.perf_counter() - started),
    }


def collapse_width_aware_junction_clusters(skeleton, candidate_mask, distance_transform):
    """Collapse nearby junction pixels while retaining every external port."""
    value = np.asarray(skeleton, dtype=np.uint8) > 0
    candidate = np.asarray(candidate_mask, dtype=np.uint8) > 0
    distance = np.asarray(distance_transform, dtype=np.float32)
    adjacency = _skeleton_adjacency(value)
    junction = np.zeros(value.shape, dtype=np.uint8)
    for point, neighbors in adjacency.items():
        if len(neighbors) >= 3:
            junction[point] = 1
    before_count = int(np.count_nonzero(junction))
    if before_count == 0:
        return {
            "collapsed_skeleton": value.astype(np.uint8),
            "junction_mask": junction,
            "collapsed_zone_mask": np.zeros(value.shape, dtype=np.uint8),
            "junction_pixel_count_before": 0,
            "junction_cluster_count_after": 0,
        }

    expanded = np.zeros(value.shape, dtype=np.uint8)
    junction_rows, junction_cols = np.where(junction > 0)
    radii = np.clip(
        np.rint(0.35 * distance[junction_rows, junction_cols]).astype(np.int32), 1, 5
    )
    for radius in range(1, 6):
        members = np.zeros(value.shape, dtype=np.uint8)
        selected = radii == radius
        members[junction_rows[selected], junction_cols[selected]] = 1
        if np.any(members):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            expanded |= cv2.dilate(members, kernel)
    cluster_count, cluster_labels, cluster_stats, _ = cv2.connectedComponentsWithStats(
        expanded, 8
    )
    collapsed = value.copy()
    collapsed_zones = np.zeros(value.shape, dtype=np.uint8)
    logical_junctions = np.zeros(value.shape, dtype=np.uint8)
    accepted_clusters = 0
    for cluster_id in range(1, cluster_count):
        col, row, width, height, _area = map(int, cluster_stats[cluster_id])
        row0, row1 = max(0, row - 1), min(value.shape[0], row + height + 1)
        col0, col1 = max(0, col - 1), min(value.shape[1], col + width + 1)
        local_cluster_labels = cluster_labels[row0:row1, col0:col1]
        zone = local_cluster_labels == cluster_id
        local_collapsed = collapsed[row0:row1, col0:col1]
        inside = zone & local_collapsed
        if np.count_nonzero(inside) < 2:
            continue
        ring = (cv2.dilate(zone.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0) & ~zone
        ports = ring & local_collapsed
        port_count, port_labels = cv2.connectedComponents(ports.astype(np.uint8), 8)
        if port_count - 1 < 3:
            continue
        inside_points = np.column_stack(np.where(inside))
        centroid = np.mean(inside_points.astype(np.float32), axis=0)
        center_scores = np.linalg.norm(inside_points - centroid[None, :], axis=1)
        global_inside = inside_points + np.asarray([row0, col0], dtype=np.int32)
        center_scores -= 0.20 * distance[global_inside[:, 0], global_inside[:, 1]]
        center_local = inside_points[int(np.argmin(center_scores))]
        center = center_local + np.asarray([row0, col0], dtype=np.int32)
        connections = []
        valid = True
        for port_id in range(1, port_count):
            port_points = np.column_stack(np.where(port_labels == port_id))
            if not len(port_points):
                continue
            port_local = port_points[int(np.argmin(
                np.linalg.norm(port_points - center_local[None, :], axis=1)
            ))]
            port = port_local + np.asarray([row0, col0], dtype=np.int32)
            rr, cc = line(int(center[0]), int(center[1]), int(port[0]), int(port[1]))
            if float(np.mean(candidate[rr, cc])) < 0.90:
                valid = False
                break
            connections.append((rr, cc))
        if not valid or len(connections) < 3:
            continue
        local_collapsed[inside] = False
        collapsed[int(center[0]), int(center[1])] = True
        for rr, cc in connections:
            collapsed[rr, cc] = True
        local_collapsed_zones = collapsed_zones[row0:row1, col0:col1]
        local_collapsed_zones[zone] = 1
        logical_junctions[int(center[0]), int(center[1])] = 1
        accepted_clusters += 1
    return {
        "collapsed_skeleton": collapsed.astype(np.uint8),
        "junction_mask": logical_junctions,
        "collapsed_zone_mask": collapsed_zones,
        "junction_pixel_count_before": before_count,
        "junction_cluster_count_after": int(accepted_clusters),
    }


def simplify_skeleton_polylines(skeleton, *, epsilon=0.75):
    """Simplify each topology-bounded chain without moving its fixed endpoints."""
    simplified = []
    for path in _trace_skeleton_chains(np.asarray(skeleton, dtype=np.uint8) > 0):
        xy = path[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
        approximation = cv2.approxPolyDP(xy, float(epsilon), False).reshape(-1, 2)
        approximation[0] = xy[0, 0]
        approximation[-1] = xy[-1, 0]
        simplified.append(approximation[:, ::-1].astype(np.float32))
    return simplified


def extract_regularized_relative_centerline(
    candidate_mask,
    *,
    relative_score=None,
    junction_collapse=False,
):
    """Run regularization, one skeletonization, local cleanup, and simplification."""
    started = time.perf_counter()
    regularized = regularize_relative_candidate(candidate_mask)
    candidate = regularized["regularized_candidate"]
    skeleton_started = time.perf_counter()
    raw_skeleton = skeletonize(candidate > 0).astype(np.uint8)
    skeletonize_seconds = time.perf_counter() - skeleton_started
    pruning_started = time.perf_counter()
    pruning = prune_width_aware_spurs(
        raw_skeleton, regularized["distance_transform"]
    )
    spur_pruning_seconds = time.perf_counter() - pruning_started
    cycle_result = remove_width_scale_small_cycles(
        pruning["pruned_skeleton"],
        regularized["distance_transform"],
        relative_score=relative_score,
    )
    collapse_started = time.perf_counter()
    if junction_collapse:
        collapsed = collapse_width_aware_junction_clusters(
            cycle_result["cleaned_skeleton"], candidate, regularized["distance_transform"]
        )
    else:
        cycle_skeleton = cycle_result["cleaned_skeleton"].astype(np.uint8)
        adjacency = _skeleton_adjacency(cycle_skeleton > 0)
        junction_mask = np.zeros(cycle_skeleton.shape, dtype=np.uint8)
        for point, neighbors in adjacency.items():
            if len(neighbors) >= 3:
                junction_mask[point] = 1
        collapsed = {
            "collapsed_skeleton": cycle_skeleton,
            "junction_mask": junction_mask,
            "collapsed_zone_mask": np.zeros(cycle_skeleton.shape, dtype=np.uint8),
            "junction_pixel_count_before": int(np.count_nonzero(junction_mask)),
            "junction_cluster_count_after": 0,
        }
    junction_collapse_seconds = time.perf_counter() - collapse_started
    simplify_started = time.perf_counter()
    vector_paths = simplify_skeleton_polylines(collapsed["collapsed_skeleton"])
    vector_simplification_seconds = time.perf_counter() - simplify_started
    cycle_pixels = cycle_result["detected_cycle_mask"] > 0

    def cycle_hole_proximity(mask):
        hole_mask = np.asarray(mask, dtype=np.uint8) > 0
        if not np.any(cycle_pixels) or not np.any(hole_mask):
            return 0.0
        nearest_hole = distance_transform_edt(~hole_mask).astype(np.float32)
        local_scale = regularized["distance_transform"][cycle_pixels]
        return float(np.mean(
            nearest_hole[cycle_pixels] <= (2.0 * np.maximum(local_scale, 1.0) + 2.0)
        ))

    audit = {
        **regularized["performance"],
        "skeletonize_seconds": float(skeletonize_seconds),
        "spur_pruning_seconds": float(spur_pruning_seconds),
        "cycle_detection_seconds": float(cycle_result["cycle_detection_seconds"]),
        "cycle_cleanup_seconds": float(cycle_result["cycle_cleanup_seconds"]),
        "junction_collapse_seconds": float(junction_collapse_seconds),
        "junction_collapse_active": bool(junction_collapse),
        "vector_simplification_seconds": float(vector_simplification_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "skeleton_length_before_pruning": int(np.count_nonzero(raw_skeleton)),
        "skeleton_length_after_pruning": int(np.count_nonzero(pruning["pruned_skeleton"])),
        "final_skeleton_length": int(np.count_nonzero(collapsed["collapsed_skeleton"])),
        "spur_removed_count": int(pruning["spur_removed_count"]),
        "spur_removed_length": float(pruning["spur_removed_length"]),
        "raw_cycle_count": int(cycle_result["raw_cycle_count"]),
        "small_cycle_count": int(cycle_result["small_cycle_count"]),
        "removed_cycle_count": int(cycle_result["removed_cycle_count"]),
        "preserved_cycle_count": int(cycle_result["preserved_cycle_count"]),
        "final_cycle_count": int(cycle_result["final_cycle_count"]),
        "cycle_near_detected_hole_fraction": cycle_hole_proximity(
            regularized["detected_hole_mask"]
        ),
        "cycle_near_filled_hole_fraction": cycle_hole_proximity(
            regularized["filled_hole_mask"]
        ),
        "junction_pixel_count_before": int(collapsed["junction_pixel_count_before"]),
        "junction_cluster_count_after": int(collapsed["junction_cluster_count_after"]),
    }
    return {
        "regularized_candidate": candidate,
        "distance_transform": regularized["distance_transform"],
        "raw_skeleton": raw_skeleton,
        "pruned_skeleton": pruning["pruned_skeleton"],
        "cycle_cleaned_skeleton": cycle_result["cleaned_skeleton"],
        "final_skeleton": collapsed["collapsed_skeleton"],
        "removed_spur_mask": pruning["removed_spur_mask"],
        "detected_hole_mask": regularized["detected_hole_mask"],
        "filled_hole_mask": regularized["filled_hole_mask"],
        "preserved_hole_mask": regularized["preserved_hole_mask"],
        "narrow_hole_mask": regularized["narrow_hole_mask"],
        "detected_cycle_mask": cycle_result["detected_cycle_mask"],
        "removed_cycle_mask": cycle_result["removed_cycle_mask"],
        "cycle_records": cycle_result["cycle_records"],
        "junction_mask": collapsed["junction_mask"],
        "collapsed_zone_mask": collapsed["collapsed_zone_mask"],
        "vector_paths": vector_paths,
        "performance": audit,
    }


_RELATIVE_BACKBONE_SOURCES = {
    1: ("relative_ridge_seed", "ridge_seed"),
    2: ("relative_backbone_bridge", "ridge_to_ridge_bridge"),
    3: ("relative_backbone_extension", "directional_extension"),
    4: ("relative_backbone_branch", "independent_supported_branch"),
}


def build_relative_support_graph(
    binary_skeleton,
    *,
    relative_score=None,
    scene_rank=None,
    ridge_mask=None,
    ridge_strength=None,
    ridge_orientation=None,
    scale_agreement=None,
    candidate_mask=None,
    ridge_projection_radius=2,
):
    """Describe the unmodified Binary Skeleton as traversable micro-chains.

    Ridge pixels are projected only onto nearby, orientation-compatible Binary
    Skeleton pixels.  This lets a ridge seed identify its road direction
    without turning a perpendicular fishbone at the same junction into a seed.
    """
    binary = np.asarray(binary_skeleton, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("binary_skeleton must be a 2D array")

    shape = binary.shape

    def optional_map(value, dtype=np.float32):
        if value is None:
            return None
        array = np.asarray(value, dtype=dtype)
        if array.shape != shape:
            raise ValueError("Relative support maps must align with binary_skeleton")
        return array

    score = optional_map(relative_score)
    rank = optional_map(scene_rank)
    ridge = optional_map(ridge_mask, np.uint8)
    strength = optional_map(ridge_strength)
    orientation = optional_map(ridge_orientation)
    agreement = optional_map(scale_agreement)
    candidate = optional_map(candidate_mask, np.uint8)
    grouping = build_relative_chain_corridors(
        binary,
        relative_score=score,
        scene_rank=rank,
        scale_agreement=agreement,
        candidate_mask=candidate,
    )

    radius = max(0, int(ridge_projection_radius))
    ridge_bool = np.zeros(shape, dtype=bool) if ridge is None else ridge > 0
    if radius > 0 and np.any(ridge_bool):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        ridge_near = cv2.dilate(ridge_bool.astype(np.uint8), kernel) > 0
        nearby_strength = (
            cv2.dilate(np.maximum(strength, 0.0), kernel)
            if strength is not None else ridge_near.astype(np.float32)
        )
    else:
        ridge_near = ridge_bool
        nearby_strength = (
            np.maximum(strength, 0.0)
            if strength is not None else ridge_bool.astype(np.float32)
        )

    projected_seed_mask = np.zeros(shape, dtype=np.uint8)
    for record in grouping["chains"]:
        path = record["path"]
        rows, cols = path[:, 0], path[:, 1]
        points = path.astype(np.float32)
        if len(points) >= 2:
            tangent = np.gradient(points, axis=0)
            tangent_orientation = np.mod(
                np.arctan2(tangent[:, 0], tangent[:, 1]), math.pi
            )
        else:
            tangent_orientation = np.zeros(len(points), dtype=np.float32)
        if orientation is not None:
            orientation_agreement = np.cos(
                tangent_orientation - orientation[rows, cols]
            ) ** 2
        else:
            orientation_agreement = np.ones(len(path), dtype=np.float32)
        seed_flags = ridge_near[rows, cols] & (orientation_agreement >= 0.5)
        # A single contact at a shared junction is not an independent seed.
        if np.count_nonzero(seed_flags) == 1:
            seed_flags[:] = False
        projected_seed_mask[rows[seed_flags], cols[seed_flags]] = 1
        record.update({
            "ridge_overlap": float(np.mean(ridge_bool[rows, cols])) if len(path) else 0.0,
            "ridge_projected_overlap": float(np.mean(seed_flags)) if len(path) else 0.0,
            "ridge_strength": float(np.mean(nearby_strength[rows, cols])) if len(path) else 0.0,
            "local_orientation": float(np.mod(
                np.arctan2(
                    np.mean(np.sin(2.0 * tangent_orientation)),
                    np.mean(np.cos(2.0 * tangent_orientation)),
                ) / 2.0,
                math.pi,
            )) if len(path) else 0.0,
            "orientation_agreement": float(np.mean(orientation_agreement)) if len(path) else 0.0,
            "scale_agreement": float(np.mean(agreement[rows, cols])) if agreement is not None and len(path) else 0.0,
            "ridge_supported": bool(np.any(seed_flags)),
            "_ridge_seed_flags": seed_flags,
        })

    seed_component_count = (
        cv2.connectedComponents(projected_seed_mask, 8)[0] - 1
        if np.any(projected_seed_mask) else 0
    )
    return {
        **grouping,
        "binary_skeleton": binary.astype(np.uint8),
        "projected_ridge_seed_mask": projected_seed_mask,
        "ridge_seed_component_count": int(seed_component_count),
    }


def trace_relative_backbone(
    support_graph,
    *,
    min_chain_length=48.0,
    relative_weak_threshold=0.0,
    independent_length_factor=1.5,
):
    """Select original Binary Skeleton paths using Ridge-seeded corridors.

    A directionally grouped corridor containing Ridge support is retained in
    full: internal Ridge gaps become bridges and the remaining tails become
    endpoint extensions.  A corridor without a Ridge seed is retained only as
    an independently supported long branch.  No path is drawn off the Binary
    Skeleton and no junction geometry is collapsed or reconnected.
    """
    binary = np.asarray(support_graph["binary_skeleton"], dtype=np.uint8) > 0
    backbone = np.zeros(binary.shape, dtype=np.uint8)
    rejected = np.zeros(binary.shape, dtype=np.uint8)
    source_labels = np.zeros(binary.shape, dtype=np.uint8)
    records = support_graph["chains"]
    records_by_id = {int(row["chain_id"]): row for row in records}
    corridors_by_id = {
        int(row["corridor_id"]): row for row in support_graph["corridors"]
    }
    chain_groups = defaultdict(list)
    for record in records:
        chain_groups[int(record["corridor_id"])].append(record)

    weak_gate = max(float(relative_weak_threshold) - 1e-6, 0.0)
    selected_corridor_ids = set()
    source_lengths = Counter()
    source_counts = Counter()
    spur_count = 0
    spur_length = 0.0

    for corridor_id, corridor_records in chain_groups.items():
        corridor = corridors_by_id[corridor_id]
        seed_ids = {
            int(record["chain_id"])
            for record in corridor_records if record.get("ridge_supported", False)
        }
        independent_supported = bool(
            not seed_ids
            and float(corridor["total_length"])
                >= float(min_chain_length) * float(independent_length_factor)
            and float(corridor.get("relative_score_q25", 0.0)) >= weak_gate
            and float(corridor.get("scene_rank_q25", 1.0)) >= 0.50
            and float(corridor.get("scale_agreement_mean", 0.0)) > 0.0
        )
        if not seed_ids and not independent_supported:
            for record in corridor_records:
                path = record["path"]
                rejected[path[:, 0], path[:, 1]] = 1
                record.update({
                    "selected": False,
                    "line_source": "",
                    "backbone_reason": "short_unsupported_spur",
                })
                spur_count += 1
                spur_length += float(record["length"])
            continue

        selected_corridor_ids.add(corridor_id)
        corridor_chain_ids = {int(record["chain_id"]) for record in corridor_records}
        chain_graph = nx.Graph()
        chain_graph.add_nodes_from(corridor_chain_ids)
        for record in corridor_records:
            chain_id = int(record["chain_id"])
            for neighbor_id in record.get("neighbor_chain_ids", []):
                if int(neighbor_id) in corridor_chain_ids:
                    chain_graph.add_edge(chain_id, int(neighbor_id))

        bridge_chain_ids = set()
        if len(seed_ids) >= 2:
            core = chain_graph.copy()
            queue = deque(
                node for node in core.nodes
                if node not in seed_ids and core.degree(node) <= 1
            )
            while queue:
                node = queue.popleft()
                if node not in core or node in seed_ids or core.degree(node) > 1:
                    continue
                neighbors = list(core.neighbors(node))
                core.remove_node(node)
                for neighbor in neighbors:
                    if (
                        neighbor in core and neighbor not in seed_ids
                        and core.degree(neighbor) <= 1
                    ):
                        queue.append(neighbor)
            bridge_chain_ids = set(core.nodes) - seed_ids

        for record in corridor_records:
            path = record["path"]
            chain_id = int(record["chain_id"])
            codes = np.zeros(len(path), dtype=np.uint8)
            if independent_supported:
                codes[:] = 4
            elif record.get("ridge_supported", False):
                seed_flags = np.asarray(record["_ridge_seed_flags"], dtype=bool)
                seed_positions = np.flatnonzero(seed_flags)
                codes[seed_flags] = 1
                if len(seed_positions):
                    first, last = int(seed_positions[0]), int(seed_positions[-1])
                    codes[first:last + 1][codes[first:last + 1] == 0] = 2
                    codes[:first][codes[:first] == 0] = 3
                    codes[last + 1:][codes[last + 1:] == 0] = 3
            elif chain_id in bridge_chain_ids:
                codes[:] = 2
            else:
                codes[:] = 3
            # Degenerate shared endpoints inherit the chain's dominant source.
            if not np.any(codes):
                codes[:] = 3
            dominant_code = int(np.bincount(codes, minlength=5)[1:].argmax() + 1)
            existing = source_labels[path[:, 0], path[:, 1]]
            source_labels[path[:, 0], path[:, 1]] = np.maximum(existing, codes)
            backbone[path[:, 0], path[:, 1]] = 1

            step_lengths = (
                np.linalg.norm(np.diff(path.astype(np.float32), axis=0), axis=1)
                if len(path) >= 2 else np.zeros(0, dtype=np.float32)
            )
            for code in range(1, 5):
                if len(step_lengths):
                    edge_codes = np.maximum(codes[:-1], codes[1:])
                    source_lengths[code] += float(np.sum(step_lengths[edge_codes == code]))
                if np.any(codes == code):
                    source_counts[code] += 1
            line_source, reason = _RELATIVE_BACKBONE_SOURCES[dominant_code]
            record.update({
                "selected": True,
                "line_source": line_source,
                "backbone_reason": reason,
            })

    def length(code):
        return float(source_lengths.get(code, 0.0))

    selected_corridors = [
        row for row in support_graph["corridors"]
        if int(row["corridor_id"]) in selected_corridor_ids
    ]
    diagnostics = {
        "binary_skeleton_length": int(np.count_nonzero(binary)),
        "ridge_seed_length": length(1),
        "ridge_seed_component_count": int(support_graph["ridge_seed_component_count"]),
        "ridge_to_ridge_bridge_count": int(source_counts.get(2, 0)),
        "ridge_to_ridge_bridge_length": length(2),
        "extension_count": int(source_counts.get(3, 0)),
        "extension_length": length(3),
        "branch_count": int(source_counts.get(4, 0)),
        "branch_length": length(4),
        "spur_rejected_count": int(spur_count),
        "spur_rejected_length": float(spur_length),
        "final_backbone_length": float(sum(source_lengths.values())),
    }
    return {
        "relative_backbone_mask": backbone,
        "relative_backbone_source_labels": source_labels,
        "relative_rejected_skeleton": rejected,
        "relative_chain_labels": support_graph["chain_labels"],
        "relative_corridor_labels": support_graph["corridor_labels"],
        "chains": records,
        "corridors": selected_corridors,
        "diagnostics": diagnostics,
    }


def extract_relative_skeleton(
    candidate_mask,
    config,
    *,
    distance_scale=1.0,
    relative_score=None,
    scene_rank=None,
    scale_agreement=None,
    relative_weak_threshold=0.0,
    input_skeleton=None,
    junction_zone_labels=None,
):
    """Filter micro-chains using logical-corridor length and evidence."""
    candidate = np.asarray(candidate_mask, dtype=np.uint8) > 0
    component_count, labels = cv2.connectedComponents(candidate.astype(np.uint8), 8)
    retained = np.zeros(candidate.shape, dtype=bool)
    rejected = np.zeros(candidate.shape, dtype=bool)
    scale = max(float(distance_scale), 1e-6)
    min_length = float(_config_value(config, "RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX", 48.0)) * scale
    max_tortuosity = float(_config_value(config, "RELATIVE_ROADNESS_MAX_TORTUOSITY", 1.5))
    skeleton_before = (
        np.asarray(input_skeleton, dtype=bool)
        if input_skeleton is not None and np.asarray(input_skeleton).shape == candidate.shape
        else skeletonize(candidate)
    )
    geometry_pass_count = 0
    reject_counts = Counter()
    score_map = np.asarray(relative_score, dtype=np.float32) if relative_score is not None else None
    rank_map = np.asarray(scene_rank, dtype=np.float32) if scene_rank is not None else None
    agreement_map = (
        np.asarray(scale_agreement, dtype=np.float32)
        if scale_agreement is not None else None
    )
    corridor_evidence_available = bool(
        score_map is not None and score_map.shape == candidate.shape
    )
    grouping = build_relative_chain_corridors(
        skeleton_before,
        relative_score=score_map,
        scene_rank=rank_map,
        scale_agreement=agreement_map,
        candidate_mask=candidate,
        junction_zone_labels=junction_zone_labels,
    )
    corridors_by_id = {
        row["corridor_id"]: row for row in grouping["corridors"]
    }
    rescued_count = 0
    rescued_length = 0.0
    isolated_short_count = 0
    audited_chains = []
    weak_gate = max(float(relative_weak_threshold) - 1e-6, 0.0)
    for record in grouping["chains"]:
        path = record["path"]
        geometry = record["geometry"]
        path_length = float(record["length"])
        corridor = corridors_by_id[record["corridor_id"]]
        corridor_total = float(corridor["total_length"])
        is_short = path_length < min_length
        rescued = bool(
            is_short
            and corridor_total >= min_length
            and corridor_evidence_available
        )
        classification = "corridor_supported_short" if rescued else "accepted_chain"
        reject_reason = ""
        if is_short and not rescued:
            classification = "isolated_short"
            reject_reason = "isolated_short"
            isolated_short_count += 1
        chain_evidence_ok = bool(
            record["relative_score_q25"] >= weak_gate
            and record["scene_rank_q25"] >= 0.50
        )
        corridor_evidence_ok = bool(
            corridor["relative_score_q25"] >= weak_gate
            and corridor["scene_rank_q25"] >= 0.50
        )
        if not reject_reason and not (
            chain_evidence_ok or (rescued and corridor_evidence_ok)
        ):
            reject_reason = "relative_structure_unsupported"
        geometry_ok = bool(
            geometry["tortuosity"] <= max_tortuosity
            or (
                path_length >= 1.5 * min_length
                and geometry["locally_smooth"]
            )
            or (
                rescued
                and geometry["reversal_fraction"] <= 0.05
            )
        )
        if not reject_reason and not geometry_ok:
            reject_reason = "tortuosity"
        accepted = not reject_reason
        if accepted:
            retained[path[:, 0], path[:, 1]] = True
            geometry_pass_count += 1
            if rescued:
                rescued_count += 1
                rescued_length += path_length
        else:
            rejected[path[:, 0], path[:, 1]] = True
            reject_counts[reject_reason] += 1
        audited_chains.append({
            key: value for key, value in record.items()
            if key not in {"path", "geometry"}
        } | {
            "micro_chain_length": path_length,
            "classification": classification,
            "rescued_by_corridor": bool(accepted and rescued),
            "accepted": bool(accepted),
            "reject_reason": reject_reason,
        })

    retained_component_ids = set(labels[retained].tolist()) - {0}
    rejected_component_ids = set(range(1, component_count)) - retained_component_ids
    audited_corridors = []
    for corridor in grouping["corridors"]:
        audited_corridors.append({
            **corridor,
            "accepted_chain_count": int(sum(
                row["accepted"] for row in audited_chains
                if row["corridor_id"] == corridor["corridor_id"]
            )),
            "rescued_chain_count": int(sum(
                row["rescued_by_corridor"] for row in audited_chains
                if row["corridor_id"] == corridor["corridor_id"]
            )),
        })
    return retained.astype(np.uint8), rejected.astype(np.uint8), {
        "relative_component_count": max(0, int(component_count - 1)),
        "relative_retained_component_count": int(len(retained_component_ids)),
        "relative_rejected_component_count": int(len(rejected_component_ids)),
        "relative_skeleton_before_structure_filter": int(np.count_nonzero(skeleton_before)),
        "relative_skeleton_after_structure_filter": int(np.count_nonzero(retained)),
        "relative_chain_count": int(len(grouping["chains"])),
        "micro_chain_count": int(len(grouping["chains"])),
        "too_short_micro_chain_count": int(sum(
            row["length"] < min_length for row in grouping["chains"]
        )),
        "corridor_count": int(len(grouping["corridors"])),
        "corridor_pairing_count": int(grouping["pairing_count"]),
        "corridor_ambiguous_junction_count": int(grouping["ambiguous_junction_count"]),
        "corridor_rescued_chain_count": int(rescued_count),
        "corridor_rescued_length": float(rescued_length),
        "structure_rescued_chain_count": int(rescued_count),
        "structure_rescued_length": float(rescued_length),
        "isolated_short_rejected_count": int(isolated_short_count),
        "relative_chain_geometry_pass": int(geometry_pass_count),
        "relative_structure_reject_reason_counts": dict(sorted(reject_counts.items())),
        "relative_skeleton_total_length": int(np.count_nonzero(retained)),
        "relative_micro_chains": audited_chains,
        "relative_corridors": audited_corridors,
        "_relative_chain_labels": grouping["chain_labels"],
        "_relative_corridor_labels": grouping["corridor_labels"],
    }


def _regularized_relative_context(
    road,
    relative_score,
    scene_rank,
    local_background,
    local_contrast,
    normalized_contrast,
    scale_support_count,
    scale_agreement,
    candidate,
    threshold_summary,
    config,
    *,
    scene_state,
    used_scales,
    contrast_scale,
    distance_scale,
):
    """Build the Relative context from the regularized-skeleton experiment."""
    junction_collapse_active = bool(
        _config_value(config, "RELATIVE_JUNCTION_COLLAPSE_EXPERIMENTAL", False)
    )
    result = extract_regularized_relative_centerline(
        candidate,
        relative_score=relative_score,
        junction_collapse=junction_collapse_active,
    )
    regularized_candidate = result["regularized_candidate"]
    raw_skeleton = result["raw_skeleton"]
    pruned_skeleton = result["pruned_skeleton"]
    relative_skeleton = result["final_skeleton"]
    grouping = build_relative_chain_corridors(
        relative_skeleton,
        relative_score=relative_score,
        scene_rank=scene_rank,
        scale_agreement=scale_agreement,
        candidate_mask=regularized_candidate,
    )
    corridors_by_id = {
        int(row["corridor_id"]): row for row in grouping["corridors"]
    }
    audited_chains = []
    for record in grouping["chains"]:
        corridor = corridors_by_id[int(record["corridor_id"])]
        audited_chains.append({
            key: value for key, value in record.items()
            if key not in {"path", "geometry"}
        } | {
            "micro_chain_length": float(record["length"]),
            "corridor_total_length": float(corridor["total_length"]),
            "accepted": True,
            "reject_reason": "",
            "line_source": "relative_regularized_skeleton",
            "backbone_reason": "regularized_candidate_skeleton",
        })
    component_count, component_labels = cv2.connectedComponents(
        relative_skeleton.astype(np.uint8), 8
    )
    if np.any(regularized_candidate):
        structure_count, structure_labels = cv2.connectedComponents(
            regularized_candidate.astype(np.uint8), 8
        )
    else:
        structure_count = 1
        structure_labels = np.zeros(road.shape, dtype=np.int32)
    centerline_chains = _trace_skeleton_chains(relative_skeleton > 0)
    centerline_length = float(sum(
        _relative_chain_geometry(path)["path_length"] for path in centerline_chains
    ))
    selected_neighborhood = cv2.dilate(
        relative_skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ) > 0
    rejected_skeleton = ((raw_skeleton > 0) & ~selected_neighborhood).astype(np.uint8)
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
        absolute_neighborhood = cv2.dilate(
            absolute_skeleton.astype(np.uint8), kernel
        ) > 0
    relative_only = (relative_skeleton > 0) & ~absolute_neighborhood
    combined = absolute_skeleton | relative_only
    empty_u8 = np.zeros(road.shape, dtype=np.uint8)
    empty_f32 = np.zeros(road.shape, dtype=np.float32)
    performance = result["performance"]
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
        "relative_component_count": int(structure_count - 1),
        "relative_retained_component_count": int(component_count - 1),
        "relative_rejected_component_count": 0,
        "relative_skeleton_before_structure_filter": int(np.count_nonzero(raw_skeleton)),
        "relative_skeleton_after_structure_filter": int(np.count_nonzero(relative_skeleton)),
        "relative_chain_count": int(len(grouping["chains"])),
        "micro_chain_count": int(len(grouping["chains"])),
        "too_short_micro_chain_count": 0,
        "corridor_count": int(len(grouping["corridors"])),
        "corridor_pairing_count": int(grouping["pairing_count"]),
        "corridor_ambiguous_junction_count": int(grouping["ambiguous_junction_count"]),
        "relative_chain_geometry_pass": int(len(grouping["chains"])),
        "relative_structure_reject_reason_counts": {},
        "relative_skeleton_total_length": int(np.count_nonzero(relative_skeleton)),
        "relative_micro_chains": audited_chains,
        "relative_corridors": grouping["corridors"],
        "ribbon_centerline_total_length": centerline_length,
        "regularized_centerline_length": centerline_length,
        "continuous_tracing_experimental_active": False,
        "regularized_skeleton_experimental_active": True,
        "relative_junction_collapse_experimental_active": junction_collapse_active,
        "continuous_trace_count": 0,
        "continuous_centerline_component_count": 0,
        "continuous_centerline_length": 0.0,
        "continuous_junction_count": 0,
        "confirmed_branch_count": 0,
        "rejected_spur_count": int(performance["spur_removed_count"]),
        "rejected_spur_length": float(performance["spur_removed_length"]),
        "binary_component_count": int(component_count - 1),
        "ridge_component_count": 0,
        "ribbon_component_count": int(component_count - 1),
        "binary_junction_count": int(performance["junction_pixel_count_before"]),
        "pruned_spur_count": int(performance["spur_removed_count"]),
        "collapsed_zone_count": int(performance["junction_cluster_count_after"]),
        "complex_junction_zone_count": int(performance["junction_cluster_count_after"]),
        "complex_zone_skipped_collapse_count": 0,
        "relative_skeleton_performance_audit": performance,
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
        "relative_regularized_candidate": regularized_candidate.astype(np.uint8),
        "relative_regularized_raw_skeleton": raw_skeleton.astype(np.uint8),
        "relative_regularized_pruned_skeleton": pruned_skeleton.astype(np.uint8),
        "relative_regularized_cycle_cleaned_skeleton": result[
            "cycle_cleaned_skeleton"
        ].astype(np.uint8),
        "relative_regularized_final_skeleton": relative_skeleton.astype(np.uint8),
        "relative_detected_hole_mask": result["detected_hole_mask"].astype(np.uint8),
        "relative_filled_hole_mask": result["filled_hole_mask"].astype(np.uint8),
        "relative_preserved_hole_mask": result["preserved_hole_mask"].astype(np.uint8),
        "relative_narrow_hole_mask": result["narrow_hole_mask"].astype(np.uint8),
        "relative_detected_cycle_mask": result["detected_cycle_mask"].astype(np.uint8),
        "relative_removed_cycle_mask": result["removed_cycle_mask"].astype(np.uint8),
        "relative_regularized_vector_paths": result["vector_paths"],
        "relative_skeleton_performance_audit": performance,
        "relative_binary_skeleton": raw_skeleton.astype(np.uint8),
        "relative_ridge_mask": empty_u8.copy(),
        "relative_backbone_mask": empty_u8.copy(),
        "relative_backbone_source_labels": empty_u8.copy(),
        "relative_ribbon_centerline": relative_skeleton.astype(np.uint8),
        "relative_ribbon_component_labels": component_labels.astype(np.int32),
        "relative_ribbon_structure_labels": structure_labels.astype(np.int32),
        "relative_continuous_centerline": empty_u8.copy(),
        "relative_trace_id_map": np.zeros(road.shape, dtype=np.int32),
        "relative_continuous_seed_mask": empty_u8.copy(),
        "relative_continuous_junction_mask": empty_u8.copy(),
        "relative_continuous_branch_mask": empty_u8.copy(),
        "relative_continuous_rejected_spur_mask": empty_u8.copy(),
        "relative_ribbon_center_orientation": empty_f32.copy(),
        "relative_ribbon_center_confidence": empty_f32.copy(),
        "relative_ribbon_orientation_clarity": empty_f32.copy(),
        "relative_ribbon_center_preference": empty_f32.copy(),
        "relative_ribbon_distance_transform": result["distance_transform"].astype(np.float32),
        "ridge_orientation": empty_f32.copy(),
        "ridge_strength": empty_f32.copy(),
        "relative_skeleton_raw": raw_skeleton.astype(np.uint8),
        "relative_skeleton_normalized": relative_skeleton.astype(np.uint8),
        "relative_skeleton": relative_skeleton.astype(np.uint8),
        "relative_rejected_skeleton": rejected_skeleton,
        "relative_chain_labels": grouping["chain_labels"].astype(np.int32),
        "relative_corridor_labels": grouping["corridor_labels"].astype(np.int32),
        "junction_zone_mask": result["junction_mask"].astype(np.uint8),
        "pruned_spur_mask": result["removed_spur_mask"].astype(np.uint8),
        "collapsed_zone_mask": result["collapsed_zone_mask"].astype(np.uint8),
        "absolute_skeleton": absolute_skeleton.astype(np.uint8),
        "relative_only_skeleton": relative_only.astype(np.uint8),
        "combined_skeleton": combined.astype(np.uint8),
        "diagnostics": diagnostics,
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
            "relative_binary_skeleton": np.zeros(road.shape, dtype=np.uint8),
            "relative_ridge_mask": np.zeros(road.shape, dtype=np.uint8),
            "relative_backbone_mask": np.zeros(road.shape, dtype=np.uint8),
            "relative_backbone_source_labels": np.zeros(road.shape, dtype=np.uint8),
            "relative_ribbon_centerline": np.zeros(road.shape, dtype=np.uint8),
            "relative_ribbon_component_labels": np.zeros(road.shape, dtype=np.int32),
            "relative_ribbon_structure_labels": np.zeros(road.shape, dtype=np.int32),
            "relative_continuous_centerline": np.zeros(road.shape, dtype=np.uint8),
            "relative_trace_id_map": np.zeros(road.shape, dtype=np.int32),
            "relative_continuous_seed_mask": np.zeros(road.shape, dtype=np.uint8),
            "relative_continuous_junction_mask": np.zeros(road.shape, dtype=np.uint8),
            "relative_continuous_branch_mask": np.zeros(road.shape, dtype=np.uint8),
            "relative_continuous_rejected_spur_mask": np.zeros(road.shape, dtype=np.uint8),
            "relative_ribbon_center_orientation": empty.copy(),
            "relative_ribbon_center_confidence": empty.copy(),
            "relative_ribbon_orientation_clarity": empty.copy(),
            "relative_ribbon_center_preference": empty.copy(),
            "relative_ribbon_distance_transform": empty.copy(),
            "ridge_orientation": empty.copy(),
            "ridge_strength": empty.copy(),
            "relative_skeleton_raw": np.zeros(road.shape, dtype=np.uint8),
            "relative_skeleton_normalized": np.zeros(road.shape, dtype=np.uint8),
            "relative_skeleton": np.zeros(road.shape, dtype=np.uint8),
            "relative_rejected_skeleton": np.zeros(road.shape, dtype=np.uint8),
            "relative_chain_labels": np.zeros(road.shape, dtype=np.int32),
            "relative_corridor_labels": np.zeros(road.shape, dtype=np.int32),
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
    # Release multi-scale temporaries before allocating full-scene tensor maps.
    del backgrounds, per_scale_contrast
    min_chain_length = float(_config_value(
        config, "RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX", 48.0
    )) * max(float(distance_scale), 1e-6)
    regularized_enabled = bool(_config_value(
        config, "RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL", False
    ))
    if regularized_enabled:
        return _regularized_relative_context(
            road,
            relative_score,
            scene_rank,
            local_background,
            local_contrast,
            normalized_contrast,
            scale_support_count,
            scale_agreement,
            candidate,
            threshold_summary,
            config,
            scene_state=scene_state,
            used_scales=used_scales,
            contrast_scale=contrast_scale,
            distance_scale=distance_scale,
        )
    ridge_result = extract_relative_ridge_centerline(
        relative_score,
        candidate,
        min_chain_length=min_chain_length,
    )
    binary_relative_skeleton = ridge_result["binary_skeleton"]
    raw_relative_skeleton = ridge_result["relative_ridge_mask"]
    support_graph = build_relative_support_graph(
        binary_relative_skeleton,
        relative_score=relative_score,
        scene_rank=scene_rank,
        ridge_mask=raw_relative_skeleton,
        ridge_strength=ridge_result["ridge_strength"],
        ridge_orientation=ridge_result["ridge_orientation"],
        scale_agreement=scale_agreement,
        candidate_mask=candidate,
    )
    backbone_result = trace_relative_backbone(
        support_graph,
        min_chain_length=min_chain_length,
        relative_weak_threshold=threshold_summary.get("relative_weak_threshold", 0.0),
    )
    backbone_skeleton = backbone_result["relative_backbone_mask"]
    source_labels = backbone_result["relative_backbone_source_labels"]
    ribbon_result = extract_relative_ribbon_centerline(
        relative_score,
        candidate,
        ridge_result["ridge_orientation"],
        scale_agreement=scale_agreement,
    )
    ribbon_skeleton = ribbon_result["ribbon_centerline_mask"]
    continuous_enabled = bool(_config_value(
        config, "RELATIVE_CONTINUOUS_TRACING_EXPERIMENTAL", False
    ))
    if continuous_enabled:
        continuous_result = trace_relative_ribbon_centerlines(
            relative_score,
            candidate,
            ridge_result["ridge_orientation"],
            ridge_mask=raw_relative_skeleton,
            ridge_strength=ridge_result["ridge_strength"],
            scale_agreement=scale_agreement,
            distance_transform=ribbon_result["distance_transform"],
            center_preference=ribbon_result["center_preference"],
            long_trace_length=min_chain_length,
        )
    else:
        empty_u8 = np.zeros(road.shape, dtype=np.uint8)
        continuous_result = {
            "continuous_centerline_mask": empty_u8,
            "trace_id_map": np.zeros(road.shape, dtype=np.int32),
            "seed_mask": empty_u8.copy(),
            "junction_mask": empty_u8.copy(),
            "confirmed_branch_mask": empty_u8.copy(),
            "rejected_spur_mask": empty_u8.copy(),
            "traces": [],
            "diagnostics": {
                "continuous_trace_count": 0,
                "continuous_centerline_component_count": 0,
                "continuous_centerline_length": 0.0,
                "mean_trace_length": 0.0,
                "median_trace_length": 0.0,
                "long_trace_count": 0,
                "continuous_junction_count": 0,
                "confirmed_branch_count": 0,
                "seed_count": 0,
                "seed_suppressed_existing_trace_count": 0,
                "parallel_duplicate_rejected_count": 0,
                "parallel_duplicate_rejected_length": 0.0,
                "true_branch_count": 0,
                "collision_terminated_count": 0,
                "junction_supported_merge_count": 0,
                "rejected_spur_count": 0,
                "rejected_spur_length": 0.0,
                "continuous_termination_reason_counts": {},
                "relative_trace_summaries": [],
            },
        }
    # Production keeps the proven Backbone baseline.  The dev recovery tool
    # explicitly enables Continuous Tracing for this isolated experiment.
    relative_skeleton = (
        continuous_result["continuous_centerline_mask"]
        if continuous_enabled else backbone_skeleton
    )
    ribbon_grouping = build_relative_chain_corridors(
        relative_skeleton,
        relative_score=relative_score,
        scene_rank=scene_rank,
        scale_agreement=scale_agreement,
        candidate_mask=candidate,
    )
    relative_chain_labels = ribbon_grouping["chain_labels"]
    relative_corridor_labels = ribbon_grouping["corridor_labels"]
    ribbon_component_count, ribbon_component_labels = cv2.connectedComponents(
        ribbon_skeleton.astype(np.uint8), 8
    )
    ribbon_component_count = max(0, int(ribbon_component_count - 1))
    if np.any(candidate):
        candidate_component_count, ribbon_structure_labels = cv2.connectedComponents(
            candidate.astype(np.uint8), 8
        )
        candidate_components = int(candidate_component_count - 1)
    else:
        candidate_components = 0
        ribbon_structure_labels = np.zeros(candidate.shape, dtype=np.int32)
    binary_component_count = (
        cv2.connectedComponents(binary_relative_skeleton.astype(np.uint8), 8)[0] - 1
        if np.any(binary_relative_skeleton) else 0
    )
    ridge_component_count = (
        cv2.connectedComponents(raw_relative_skeleton.astype(np.uint8), 8)[0] - 1
        if np.any(raw_relative_skeleton) else 0
    )
    selected_neighborhood = cv2.dilate(
        relative_skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ) > 0
    relative_rejected_skeleton = (
        (binary_relative_skeleton > 0) & ~selected_neighborhood
    ).astype(np.uint8)
    corridors_by_id = {
        int(row["corridor_id"]): row for row in ribbon_grouping["corridors"]
    }
    audited_chains = []
    for record in ribbon_grouping["chains"]:
        corridor = corridors_by_id[int(record["corridor_id"])]
        audited_chains.append({
            key: value for key, value in record.items()
            if key not in {"path", "geometry"}
        } | {
            "micro_chain_length": float(record["length"]),
            "corridor_total_length": float(corridor["total_length"]),
            "accepted": True,
            "reject_reason": "",
            "line_source": (
                "relative_continuous_trace" if continuous_enabled
                else "relative_backbone"
            ),
            "backbone_reason": (
                "continuous_ribbon_trace" if continuous_enabled
                else "backbone_baseline"
            ),
        })
    structure_summary = {
        "relative_component_count": int(candidate_components),
        "relative_retained_component_count": int(ribbon_component_count),
        "relative_rejected_component_count": 0,
        "relative_skeleton_before_structure_filter": int(np.count_nonzero(binary_relative_skeleton)),
        "relative_skeleton_after_structure_filter": int(np.count_nonzero(relative_skeleton)),
        "relative_chain_count": int(len(ribbon_grouping["chains"])),
        "micro_chain_count": int(len(ribbon_grouping["chains"])),
        "too_short_micro_chain_count": int(sum(
            float(row["length"]) < min_chain_length for row in ribbon_grouping["chains"]
        )),
        "corridor_count": int(len(ribbon_grouping["corridors"])),
        "corridor_pairing_count": int(ribbon_grouping["pairing_count"]),
        "corridor_ambiguous_junction_count": int(ribbon_grouping["ambiguous_junction_count"]),
        "corridor_rescued_chain_count": 0,
        "corridor_rescued_length": 0.0,
        "structure_rescued_chain_count": 0,
        "structure_rescued_length": 0.0,
        "isolated_short_rejected_count": 0,
        "relative_chain_geometry_pass": int(len(ribbon_grouping["chains"])),
        "relative_structure_reject_reason_counts": {},
        "relative_skeleton_total_length": int(np.count_nonzero(relative_skeleton)),
        "relative_micro_chains": audited_chains,
        "relative_corridors": ribbon_grouping["corridors"],
        **backbone_result["diagnostics"],
        **ribbon_result["diagnostics"],
        **continuous_result["diagnostics"],
        "continuous_tracing_experimental_active": bool(continuous_enabled),
        "binary_component_count": int(binary_component_count),
        "ridge_component_count": int(ridge_component_count),
        "binary_junction_count": int(ridge_result["diagnostics"]["old_junction_pixel_count"]),
        "ribbon_component_count": int(ribbon_component_count),
        # Backbone tracing is selection-only; these compatibility diagnostics
        # explicitly confirm that no junction geometry was rewritten.
        "pruned_spur_count": 0,
        "collapsed_zone_count": 0,
        "complex_junction_zone_count": 0,
        "complex_zone_skipped_collapse_count": 0,
    }
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
        **ridge_result["diagnostics"],
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
        "relative_binary_skeleton": binary_relative_skeleton.astype(np.uint8),
        "relative_ridge_mask": raw_relative_skeleton.astype(np.uint8),
        "relative_backbone_mask": backbone_skeleton.astype(np.uint8),
        "relative_backbone_source_labels": source_labels.astype(np.uint8),
        "relative_ribbon_centerline": ribbon_skeleton.astype(np.uint8),
        "relative_ribbon_component_labels": ribbon_component_labels.astype(np.int32),
        "relative_ribbon_structure_labels": ribbon_structure_labels.astype(np.int32),
        "relative_continuous_centerline": continuous_result["continuous_centerline_mask"].astype(np.uint8),
        "relative_trace_id_map": continuous_result["trace_id_map"].astype(np.int32),
        "relative_continuous_seed_mask": continuous_result["seed_mask"].astype(np.uint8),
        "relative_continuous_junction_mask": continuous_result["junction_mask"].astype(np.uint8),
        "relative_continuous_branch_mask": continuous_result["confirmed_branch_mask"].astype(np.uint8),
        "relative_continuous_rejected_spur_mask": continuous_result["rejected_spur_mask"].astype(np.uint8),
        "relative_ribbon_center_orientation": ribbon_result["center_orientation"].astype(np.float32),
        "relative_ribbon_center_confidence": ribbon_result["center_confidence"].astype(np.float32),
        "relative_ribbon_orientation_clarity": ribbon_result["orientation_clarity"].astype(np.float32),
        "relative_ribbon_center_preference": ribbon_result["center_preference"].astype(np.float32),
        "relative_ribbon_distance_transform": ribbon_result["distance_transform"].astype(np.float32),
        "ridge_orientation": ridge_result["ridge_orientation"].astype(np.float32),
        "ridge_strength": ridge_result["ridge_strength"].astype(np.float32),
        "relative_skeleton_raw": raw_relative_skeleton.astype(np.uint8),
        "relative_skeleton_normalized": relative_skeleton.astype(np.uint8),
        "relative_skeleton": relative_skeleton.astype(np.uint8),
        "relative_rejected_skeleton": relative_rejected_skeleton.astype(np.uint8),
        "relative_chain_labels": relative_chain_labels.astype(np.int32),
        "relative_corridor_labels": relative_corridor_labels.astype(np.int32),
        "junction_zone_mask": (
            continuous_result["junction_mask"]
            if continuous_enabled else ribbon_result["junction_zone_mask"]
        ).astype(np.uint8),
        "pruned_spur_mask": np.zeros(road.shape, dtype=np.uint8),
        "collapsed_zone_mask": np.zeros(road.shape, dtype=np.uint8),
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
        backbone_line_source = relative_evidence.get("backbone_line_source", "")
        metadata.append({
            "line_source": (
                backbone_line_source
                if candidate_source == "relative" and backbone_line_source
                else "relative_roadness" if candidate_source == "relative" else "samroad"
            ),
            "backbone_reason": relative_evidence.get("backbone_reason", ""),
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
    # Selected TopoNet edges are themselves evaluated topology candidates.
    # Some recovery-only callers do not retain the pre-threshold candidate
    # arrays, so keep the audit invariant instead of reporting 0 candidates
    # alongside a positive selected count.
    relative_topology_candidates = max(
        int(relative_topology_candidates), int(relative_topology_selected)
    )
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
        "relative_entered_acceptance_length": 0.0,
        "relative_accepted_auto_length": 0.0,
        "relative_accepted_review_length": 0.0,
        "relative_rejected_length": 0.0,
        "relative_rejected_length_by_reason": {},
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
    ribbon_component_labels = np.zeros(road.shape, dtype=np.int32)
    ribbon_component_lengths = np.zeros(1, dtype=np.float32)
    ribbon_structure_labels = np.zeros(road.shape, dtype=np.int32)
    ribbon_structure_lengths = np.zeros(1, dtype=np.float32)
    continuous_trace_ids = np.zeros(road.shape, dtype=np.int32)
    continuous_trace_lengths = {}
    relative_corridor_labels = np.zeros(road.shape, dtype=np.int32)
    relative_corridors = {}
    relative_endpoint_corridors = defaultdict(set)
    if relative_context is not None:
        # Acceptance uses the calibration-invariant structured skeleton. Actual
        # graph overlap is handled below; suppressing by raw HIGH probability
        # here would make otherwise identical Case A/B/C/D roads diverge.
        skeleton_value = np.asarray(relative_context.get("relative_skeleton", []))
        if skeleton_value.shape == road.shape:
            relative_skeleton = skeleton_value > 0
        ribbon_label_value = np.asarray(
            relative_context.get("relative_ribbon_component_labels", []),
            dtype=np.int32,
        )
        if ribbon_label_value.shape == road.shape:
            ribbon_component_labels = ribbon_label_value
            ribbon_component_lengths = np.bincount(
                ribbon_component_labels.ravel()
            ).astype(np.float32)
        ribbon_structure_value = np.asarray(
            relative_context.get("relative_ribbon_structure_labels", []),
            dtype=np.int32,
        )
        if ribbon_structure_value.shape == road.shape:
            ribbon_structure_labels = ribbon_structure_value
            ribbon_structure_lengths = np.bincount(
                ribbon_structure_labels.ravel(),
                weights=relative_skeleton.astype(np.float32).ravel(),
            ).astype(np.float32)
        continuous_id_value = np.asarray(
            relative_context.get("relative_trace_id_map", []), dtype=np.int32
        )
        if continuous_id_value.shape == road.shape:
            continuous_trace_ids = continuous_id_value
        continuous_trace_lengths = {
            int(row.get("trace_id", 0)): float(row.get("length", 0.0))
            for row in relative_context.get("diagnostics", {}).get(
                "relative_trace_summaries", []
            )
        }
        corridor_value = np.asarray(
            relative_context.get("relative_corridor_labels", []), dtype=np.int32
        )
        if corridor_value.shape == road.shape:
            relative_corridor_labels = corridor_value
        relative_corridors = {
            int(row["corridor_id"]): row
            for row in relative_context.get("diagnostics", {}).get(
                "relative_corridors", []
            )
        }
        for row in relative_context.get("diagnostics", {}).get(
            "relative_micro_chains", []
        ):
            corridor_value = int(row.get("corridor_id", 0))
            if corridor_value <= 0:
                continue
            for endpoint in (row.get("start"), row.get("end")):
                if endpoint is not None and len(endpoint) == 2:
                    relative_endpoint_corridors[tuple(map(int, endpoint))].add(
                        corridor_value
                    )
    ambiguous_relative_endpoints = {
        point for point, corridor_ids in relative_endpoint_corridors.items()
        if len(corridor_ids) > 1
    }
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
    relative_reject_lengths = Counter()

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
            rejected_length = float(row.get("path_length", 0.0))
            summary["relative_rejected_length"] += rejected_length
            relative_reject_lengths[reason] += rejected_length
        if candidate_audit is not None:
            candidate_audit.append(row)

    for path, direct_relative_chain in chains:
        summary["bootstrap_candidate_count"] += 1
        path_float = path.astype(np.float32)
        geometry = _relative_chain_geometry(path)
        path_length = geometry["path_length"]
        direct_distance = geometry["direct_distance"]
        tortuosity = geometry["tortuosity"]
        corridor_id = 0
        corridor = None
        if direct_relative_chain:
            corridor_values = relative_corridor_labels[path[:, 0], path[:, 1]]
            corridor_values = corridor_values[corridor_values > 0]
            if len(corridor_values):
                corridor_id = int(np.bincount(corridor_values).argmax())
                corridor = relative_corridors.get(corridor_id)
        corridor_total_length = float(
            corridor.get("total_length", path_length) if corridor else path_length
        )
        ribbon_values = ribbon_component_labels[path[:, 0], path[:, 1]]
        ribbon_values = ribbon_values[ribbon_values > 0]
        ribbon_component_id = (
            int(np.bincount(ribbon_values).argmax()) if len(ribbon_values) else 0
        )
        ribbon_component_length = float(
            ribbon_component_lengths[ribbon_component_id]
            if 0 < ribbon_component_id < len(ribbon_component_lengths) else 0.0
        )
        ribbon_structure_values = ribbon_structure_labels[path[:, 0], path[:, 1]]
        ribbon_structure_values = ribbon_structure_values[ribbon_structure_values > 0]
        ribbon_structure_id = (
            int(np.bincount(ribbon_structure_values).argmax())
            if len(ribbon_structure_values) else 0
        )
        ribbon_structure_length = float(
            ribbon_structure_lengths[ribbon_structure_id]
            if 0 < ribbon_structure_id < len(ribbon_structure_lengths) else 0.0
        )
        ribbon_structure_supported = bool(
            direct_relative_chain
            and max(ribbon_component_length, ribbon_structure_length)
                >= parameters["min_length"]
        )
        path_trace_values = continuous_trace_ids[path[:, 0], path[:, 1]]
        path_trace_values = path_trace_values[path_trace_values > 0]
        continuous_trace_id = (
            int(np.bincount(path_trace_values).argmax())
            if len(path_trace_values) else 0
        )
        continuous_trace_length = float(
            continuous_trace_lengths.get(continuous_trace_id, 0.0)
        )
        continuous_trace_supported = bool(
            direct_relative_chain
            and continuous_trace_id > 0
            and continuous_trace_length >= parameters["min_length"]
        )
        rescued_by_corridor = bool(
            direct_relative_chain
            and path_length < parameters["min_length"]
            and corridor_total_length >= parameters["min_length"]
        )
        structural_length = max(
            float(path_length),
            float(corridor_total_length) if rescued_by_corridor else 0.0,
            max(float(ribbon_component_length), float(ribbon_structure_length))
                if ribbon_structure_supported else 0.0,
            float(continuous_trace_length) if continuous_trace_supported else 0.0,
        )
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
            summary["relative_entered_acceptance_length"] += float(path_length)
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
            "corridor_id": int(corridor_id),
            "micro_chain_length": float(path_length),
            "corridor_total_length": float(corridor_total_length),
            "rescued_by_corridor": bool(rescued_by_corridor),
            "ribbon_component_id": int(ribbon_component_id),
            "ribbon_component_length": float(ribbon_component_length),
            "ribbon_structure_id": int(ribbon_structure_id),
            "ribbon_structure_length": float(ribbon_structure_length),
            "ribbon_structure_supported": bool(ribbon_structure_supported),
            "trace_id": int(continuous_trace_id),
            "trace_total_length": float(continuous_trace_length),
            "continuous_trace_supported": bool(continuous_trace_supported),
            "short_chain_classification": (
                "continuous_trace_supported_short" if continuous_trace_supported
                else "ribbon_supported_short" if ribbon_structure_supported
                else "corridor_supported_short" if rescued_by_corridor
                else "isolated_short" if path_length < parameters["min_length"]
                else "not_short"
            ),
            "path": path.astype(int).tolist() if relative_branch_candidate else [],
            **{key: value for key, value in relative_evidence.items() if key != "relative_supported"},
        }
        if (
            path_length < parameters["min_length"]
            and not rescued_by_corridor
            and not ribbon_structure_supported
            and not continuous_trace_supported
        ):
            reject_candidate(
                audit_row,
                "isolated_short" if direct_relative_chain else "too_short",
            )
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
        delegated_gap = bool(
            not direct_relative_chain
            and connection_count == 2
            and path_length <= recovery_gap_limit
        )
        geometry_supported = bool(
            tortuosity <= parameters["max_tortuosity"]
            or (relative_branch_candidate and geometry["locally_smooth"])
        )
        diagnostics = relative_context.get("diagnostics", {}) if relative_context else {}
        relative_weak_threshold = float(diagnostics.get("relative_weak_threshold", 0.0))
        corridor_relative_supported = bool(
            rescued_by_corridor
            and corridor is not None
            and float(corridor.get("relative_score_q25", 0.0))
                >= max(relative_weak_threshold - 1e-6, 0.0)
            and float(corridor.get("scene_rank_q25", 0.0)) >= 0.50
        )
        ribbon_relative_supported = bool(
            ribbon_structure_supported
            and relative_evidence.get("ribbon_fraction", 0.0) >= 0.50
            and relative_evidence["relative_score_q25"]
                >= max(relative_weak_threshold - 1e-6, 0.0)
            and relative_evidence["scene_rank_mean"] >= 0.50
        )
        continuous_relative_supported = bool(
            continuous_trace_supported
            and relative_evidence.get("continuous_trace_fraction", 0.0) >= 0.50
            and relative_evidence["relative_score_q25"]
                >= max(relative_weak_threshold - 1e-6, 0.0)
            and relative_evidence["scene_rank_mean"] >= 0.50
        )
        relative_supported = bool(
            relative_evidence["relative_supported"]
            or corridor_relative_supported
            or ribbon_relative_supported
            or continuous_relative_supported
        )
        independent_supported = (
            connection_count > 0
            or evidence["surface_supported"]
            or relative_supported
            or (
                structural_length >= parameters["min_length"] * parameters["independent_length_factor"]
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
                + 0.12 * min(1.0, structural_length / max(parameters["min_length"] * 2.0, 1e-6))
                + 0.08 * directness
                + 0.06 * proximity
            )
        else:
            recovery_score = (
                0.24 * min(1.0, evidence["center_conf"] / max(high_threshold, 1e-6))
                + 0.18 * min(1.0, evidence["center_q25"] / max(low_threshold, 1e-6))
                + 0.22 * min(1.0, evidence["probability_contrast"] / max(parameters["min_contrast"] * 2.0, 1e-6))
                + 0.14 * min(1.0, structural_length / max(parameters["min_length"] * 2.0, 1e-6))
                + 0.10 * directness
                + 0.06 * proximity
                + 0.06 * evidence["surface_conf"]
            )
        if relative_branch_candidate:
            acceptance_score_q25 = float(
                corridor.get("relative_score_q25", 0.0)
                if corridor_relative_supported else relative_evidence["relative_score_q25"]
            )
            acceptance_scene_rank = float(
                corridor.get("scene_rank_mean", 0.0)
                if corridor_relative_supported else relative_evidence["scene_rank_mean"]
            )
            acceptance_scale_q25 = float(
                corridor.get("scale_agreement_q25", 0.0)
                if corridor_relative_supported else relative_evidence["scale_agreement_q25"]
            )
            stable_relative = bool(
                relative_supported
                and (
                    relative_evidence["relative_fraction"] >= 0.75
                    or rescued_by_corridor
                    or ribbon_structure_supported
                    or continuous_trace_supported
                )
                and acceptance_score_q25 >= max(relative_weak_threshold - 1e-6, 0.0)
                and acceptance_scene_rank >= 0.75
                and relative_evidence["normalized_contrast_mean"] > 0.0
            )
            multiscale_supported = bool(
                acceptance_scale_q25 >= (2.0 / 3.0)
            )
            topology_supported = bool(topology_support_fraction >= 0.25)
            graph_connected = bool(
                connection_count > 0
                and (endpoint_alignment >= 0.50 or absolute_proximity_fraction >= 0.10)
            )
            independent_length = max(
                float(path_length),
                float(corridor_total_length) if rescued_by_corridor else 0.0,
                float(ribbon_component_length)
                    if ribbon_component_length >= parameters["min_length"] else 0.0,
                float(continuous_trace_length) if continuous_trace_supported else 0.0,
            )
            long_independent = bool(
                independent_length
                    >= parameters["min_length"] * parameters["independent_length_factor"]
                and geometry["locally_smooth"]
                and multiscale_supported
            )
            supporting_evidence = (
                graph_connected or topology_supported or multiscale_supported
                or rescued_by_corridor or continuous_trace_supported
            )
            if stable_relative and geometry_supported and supporting_evidence and (
                graph_connected or topology_supported or long_independent
                or rescued_by_corridor or continuous_trace_supported
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
                summary["relative_accepted_review_length"] += float(path_length)
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
            coordinate = tuple(np.rint(point).astype(np.int32).tolist())
            split_unconfirmed_crossing = bool(
                direct_relative_chain
                and corridor_id > 0
                and coordinate in ambiguous_relative_endpoints
                and not topology_mask[
                    np.clip(coordinate[0], 0, road.shape[0] - 1),
                    np.clip(coordinate[1], 0, road.shape[1] - 1),
                ]
            )
            key = (
                (coordinate[0], coordinate[1], corridor_id)
                if split_unconfirmed_crossing and node_idx is None
                else coordinate
            )
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
                "line_source": (
                    relative_evidence.get("backbone_line_source", "")
                    if relative_branch_candidate
                    and relative_evidence.get("backbone_line_source", "")
                    else "relative_roadness" if relative_branch_candidate else "weak_bootstrap"
                ),
                "backbone_reason": (
                    relative_evidence.get("backbone_reason", "")
                    if relative_branch_candidate else ""
                ),
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
                "corridor_id": int(corridor_id),
                "micro_chain_length": float(path_length),
                "corridor_total_length": float(corridor_total_length),
                "rescued_by_corridor": bool(rescued_by_corridor),
                "trace_id": int(continuous_trace_id),
                "trace_total_length": float(continuous_trace_length),
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
                summary[
                    "relative_accepted_auto_length"
                    if qa_state == "auto" else "relative_accepted_review_length"
                ] += float(path_length)
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
    summary["relative_rejected_length_by_reason"] = {
        key: float(value) for key, value in sorted(relative_reject_lengths.items())
    }
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
    return_relative_context=False,
):
    """Diagnose, bootstrap, recover endpoints, then optionally join endpoints.

    ``return_relative_context`` exposes the exact computed context to callers
    that need diagnostics without rerunning Relative Roadness or tracing.
    """
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
        str(row.get("line_source", "")).startswith("relative")
        for row in final_metadata
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
    ribbon_final_length = float(sum(
        np.linalg.norm(final_nodes_array[int(dst_idx)] - final_nodes_array[int(src_idx)])
        for (src_idx, dst_idx), row in zip(final_edges.tolist(), final_metadata)
        if row.get("line_source") == "relative_ribbon_centerline"
    ))
    continuous_final_length = float(sum(
        np.linalg.norm(final_nodes_array[int(dst_idx)] - final_nodes_array[int(src_idx)])
        for (src_idx, dst_idx), row in zip(final_edges.tolist(), final_metadata)
        if row.get("line_source") == "relative_continuous_trace"
    ))
    regularized_final_length = float(sum(
        np.linalg.norm(final_nodes_array[int(dst_idx)] - final_nodes_array[int(src_idx)])
        for (src_idx, dst_idx), row in zip(final_edges.tolist(), final_metadata)
        if row.get("line_source") == "relative_regularized_skeleton"
    ))
    final_total_length = float(sum(
        np.linalg.norm(final_nodes_array[int(dst_idx)] - final_nodes_array[int(src_idx)])
        for src_idx, dst_idx in final_edges.tolist()
    ))
    centerline_diagnostics = (
        relative_context.get("diagnostics", {}) if relative_context else {}
    )
    continuous_active = bool(centerline_diagnostics.get(
        "continuous_tracing_experimental_active", False
    ))
    regularized_active = bool(centerline_diagnostics.get(
        "regularized_skeleton_experimental_active", False
    ))
    generated_centerline_length = float(
        centerline_diagnostics.get("regularized_centerline_length", 0.0)
        if regularized_active
        else centerline_diagnostics.get("continuous_centerline_length", 0.0)
        if continuous_active
        else centerline_diagnostics.get("ribbon_centerline_total_length", 0.0)
    )
    selected_final_length = (
        regularized_final_length
        if regularized_active else continuous_final_length
        if continuous_active else ribbon_final_length
    )
    retention_ratio = float(
        selected_final_length / generated_centerline_length
        if generated_centerline_length > 1e-6 else 0.0
    )

    retained_by_existing_graph = 0.0
    if (continuous_active or regularized_active) and relative_context is not None:
        selected_mask = np.asarray(
            relative_context.get(
                "relative_regularized_final_skeleton"
                if regularized_active else "relative_continuous_centerline",
                [],
            ),
            dtype=np.uint8,
        )
        if selected_mask.shape == np.asarray(road_probability).shape:
            original_strong_mask = _graph_raster_mask(
                nodes_rc, original_edges, selected_mask.shape
            )
            overlap_radius = max(0, int(round(
                float(_config_value(
                    config, "RELATIVE_ROADNESS_ABSOLUTE_SUPPRESSION_PX", 3.0
                )) * max(float(distance_scale), 1e-6)
            )))
            if overlap_radius > 0 and np.any(original_strong_mask):
                overlap_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (overlap_radius * 2 + 1, overlap_radius * 2 + 1),
                )
                original_strong_mask = cv2.dilate(
                    original_strong_mask, overlap_kernel
                )
            supported = original_strong_mask > 0
            adjacency = _skeleton_adjacency(selected_mask > 0)
            for point, neighbors in adjacency.items():
                for neighbor in neighbors:
                    if neighbor <= point:
                        continue
                    if supported[point] and supported[neighbor]:
                        retained_by_existing_graph += float(np.linalg.norm(
                            np.asarray(neighbor, dtype=np.float32)
                            - np.asarray(point, dtype=np.float32)
                        ))
    effective_retained_length = min(
        generated_centerline_length,
        selected_final_length + retained_by_existing_graph,
    )
    effective_retention_ratio = float(
        effective_retained_length / generated_centerline_length
        if generated_centerline_length > 1e-6 else 0.0
    )
    required_loss_reasons = (
        "isolated_short", "too_short", "relative_structure_unsupported",
        "insufficient_independent_support", "tortuosity",
        "duplicate_or_suppressed", "background_contrast_low",
        "mean_probability_low", "q25_probability_low",
    )
    raw_rejected_lengths = dict(
        bootstrap_summary.get("relative_rejected_length_by_reason", {})
    )
    known_rejected_length = float(sum(
        float(raw_rejected_lengths.get(reason, 0.0))
        for reason in required_loss_reasons
    ))
    rejected_total = float(bootstrap_summary.get("relative_rejected_length", 0.0))
    rejected_by_reason = {
        reason: float(raw_rejected_lengths.get(reason, 0.0))
        for reason in required_loss_reasons
    }
    rejected_by_reason["other"] = max(0.0, rejected_total - known_rejected_length)
    centerline_loss_audit = {
        "generated_centerline_length": generated_centerline_length,
        "entered_acceptance_length": float(
            bootstrap_summary.get("relative_entered_acceptance_length", 0.0)
        ),
        "accepted_auto_length": float(
            bootstrap_summary.get("relative_accepted_auto_length", 0.0)
        ),
        "accepted_review_length": float(
            bootstrap_summary.get("relative_accepted_review_length", 0.0)
        ),
        "rejected_length": rejected_total,
        "rejected_length_by_reason": rejected_by_reason,
        "ribbon_final_length": ribbon_final_length,
        "continuous_final_length": continuous_final_length,
        "regularized_final_length": regularized_final_length,
        "centerline_to_final_retention_ratio": retention_ratio,
        "raw_centerline_to_final_retention": retention_ratio,
        "effective_centerline_to_final_retention": effective_retention_ratio,
        "retained_by_existing_graph_length": retained_by_existing_graph,
    }
    funnel = {
        **centerline_loss_audit,
        "final_relative_length": relative_final_length,
        "final_total_graph_length": final_total_length,
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
                if key not in {
                    "junction_zones", "relative_micro_chains", "relative_corridors"
                }
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
        "relative_ribbon_final_length_px": ribbon_final_length,
        "relative_continuous_final_length_px": continuous_final_length,
        "relative_regularized_final_length_px": regularized_final_length,
        "final_total_centerline_length_px": final_total_length,
        "centerline_to_final_retention_ratio": retention_ratio,
        "raw_centerline_to_final_retention": retention_ratio,
        "effective_centerline_to_final_retention": effective_retention_ratio,
        "retained_by_existing_graph_length": retained_by_existing_graph,
        "relative_centerline_loss_audit": centerline_loss_audit,
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
    result = (final_nodes, final_edges, final_metadata, summary)
    if return_relative_context:
        return (*result, relative_context)
    return result


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
