import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import tcod
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

def create_cost_field_astar(sample_pts, road_mask, block_threshold=200):
    # road mask shall be uint8 normalized to 0-255
    # for tcod, 0 is blocked
    cost_field = np.zeros(road_mask.shape, dtype=np.uint8)
    kp_block_radius = 6
    for point in sample_pts:
        cv2.circle(cost_field, point, kp_block_radius, 255, -1)
    cost_field = np.maximum(cost_field, 255 - road_mask)
    cost_field[cost_field == 0] = 1
    cost_field[cost_field > block_threshold] = 0

    return cost_field


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
    kp_candidates, kp_scores = get_points_and_scores_from_mask(keypoint_mask, config.ITSC_THRESHOLD * 255)
    kps_0 = nms_points(kp_candidates, kp_scores, config.ITSC_NMS_RADIUS)
    # The keypoint heatmap contains broad blobs. Keep its historical coarse
    # spacing; divided-road preservation is applied only to the road skeleton.
    if kps_0.shape[0]:
        kps_0 = nms_points(kps_0, np.ones(kps_0.shape[0]), config.ROAD_NMS_RADIUS)
    road_skel_mask = skeletonize_road_mask(road_mask, config.ROAD_THRESHOLD * 255)
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
    cost_field = create_cost_field_astar(kps, road_mask)
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
