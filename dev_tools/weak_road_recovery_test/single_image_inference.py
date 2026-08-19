from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rtree
import scipy
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMROAD_ROOT = REPO_ROOT / "engine" / "samroad"
for import_root in (REPO_ROOT, SAMROAD_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import graph_extraction  # noqa: E402
from dataset import read_rgb_img  # noqa: E402
from modelinfer import SAMRoadplus  # noqa: E402
from utils import load_config  # noqa: E402


def resolve_torch_device(device_arg: str):
    device_name = str(device_arg).strip().lower()
    if device_name in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return "cuda", torch.device("cuda")
    if device_name == "auto":
        return (
            ("cuda", torch.device("cuda"))
            if torch.cuda.is_available()
            else ("cpu", torch.device("cpu"))
        )
    if device_name == "cpu":
        return "cpu", torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_arg}")


def load_model(config, checkpoint_path: Path, device: torch.device):
    """Instantiate the formal SAMRoadplus class and load the requested weights."""
    model = SAMRoadplus(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model


def pad_image_to_min_size(image: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    pad_bottom = max(0, patch_size - height)
    pad_right = max(0, patch_size - width)
    if not pad_bottom and not pad_right:
        return image
    return cv2.copyMakeBorder(
        image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def _coverage_positions(extent: int, patch_size: int, overlap: int) -> list[int]:
    if extent <= patch_size:
        return [0]
    stride = max(1, patch_size - 2 * max(0, overlap))
    positions = list(range(0, extent - patch_size + 1, stride))
    final = extent - patch_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def _patches(width: int, height: int, patch_size: int, overlap: int):
    return [
        (x, y, x + patch_size, y + patch_size)
        for x in _coverage_positions(width, patch_size, overlap)
        for y in _coverage_positions(height, patch_size, overlap)
    ]


def filter_graph_to_image_bounds(nodes, edges, height, width, scores):
    nodes = np.asarray(nodes, dtype=np.float32).reshape(-1, 2)
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float32)
    keep = (
        (nodes[:, 0] >= 0) & (nodes[:, 0] < height)
        & (nodes[:, 1] >= 0) & (nodes[:, 1] < width)
    )
    kept = np.flatnonzero(keep)
    if len(kept) == len(nodes):
        return nodes, edges, scores
    remap = {old: new for new, old in enumerate(kept.tolist())}
    filtered_edges, filtered_scores = [], []
    for edge, score in zip(edges.tolist(), scores.tolist()):
        if edge[0] in remap and edge[1] in remap:
            filtered_edges.append((remap[edge[0]], remap[edge[1]]))
            filtered_scores.append(score)
    return (
        nodes[kept],
        np.asarray(filtered_edges, dtype=np.int32).reshape(-1, 2),
        np.asarray(filtered_scores, dtype=np.float32),
    )


def infer_one_image(
    model, image: np.ndarray, config, device: torch.device, timing: dict,
    diagnostic_shape=None,
):
    """Independent single-image orchestration using the formal model and graph code."""
    height, width = image.shape[:2]
    patch_size = int(config.PATCH_SIZE)
    patch_rows = _patches(
        width, height, patch_size, int(config.get("SAMPLE_MARGIN", 0))
    )
    batch_size = int(config.INFER_BATCH_SIZE)
    batch_count = (len(patch_rows) + batch_size - 1) // batch_size

    mask_started = time.perf_counter()
    fused_keypoint = torch.zeros((height, width), dtype=torch.float32, device=device)
    fused_road = torch.zeros((height, width), dtype=torch.float32, device=device)
    pixel_counter = torch.zeros((height, width), dtype=torch.float32, device=device)
    image_features, model_masks, batches = [], [], []
    for batch_index in range(batch_count):
        batch = patch_rows[batch_index * batch_size:(batch_index + 1) * batch_size]
        patches = torch.stack([
            torch.tensor(image[y0:y1, x0:x1, :], dtype=torch.float32)
            for x0, y0, x1, y1 in batch
        ]).contiguous().to(device)
        with torch.inference_mode():
            mask_scores, features = model.infer_masks_and_img_features(patches)
        image_features.append(features)
        model_masks.append(mask_scores.permute(0, 3, 1, 2))
        batches.append(batch)
        for patch_index, (x0, y0, x1, y1) in enumerate(batch):
            keypoint_patch = mask_scores[patch_index, :, :, 0]
            road_patch = mask_scores[patch_index, :, :, 1]
            fused_keypoint[y0:y1, x0:x1] += keypoint_patch
            fused_road[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += 1.0
    valid = pixel_counter > 0
    fused_keypoint = torch.where(
        valid, fused_keypoint / torch.clamp(pixel_counter, min=1.0), torch.zeros_like(fused_keypoint)
    )
    fused_road = torch.where(
        valid, fused_road / torch.clamp(pixel_counter, min=1.0), torch.zeros_like(fused_road)
    )
    keypoint_u8 = (fused_keypoint * 255).to(torch.uint8).cpu().numpy()
    road_u8 = (fused_road * 255).to(torch.uint8).cpu().numpy()
    timing["samroad_mask_seconds"] = time.perf_counter() - mask_started

    topology_started = time.perf_counter()
    diagnostic_probability = road_u8
    if diagnostic_shape is not None:
        diagnostic_probability = road_u8[:diagnostic_shape[0], :diagnostic_shape[1]]
    if str(config.get("ROAD_THRESHOLD_PROFILE", "default")) == "auto":
        decision = graph_extraction.resolve_effective_road_profile(
            diagnostic_probability, config
        )
        config.ROAD_THRESHOLD_PROFILE = decision["effective_profile"]
    graph_points = graph_extraction.extract_graph_points(keypoint_u8, road_u8, config)
    if not len(graph_points):
        timing["topology_seconds"] = time.perf_counter() - topology_started
        empty_edges = np.empty((0, 2), dtype=np.int32)
        empty_scores = np.empty((0,), dtype=np.float32)
        return graph_points, empty_edges, empty_scores, keypoint_u8, road_u8

    spatial_index = rtree.index.Index()
    for point_index, (x, y) in enumerate(graph_points):
        spatial_index.insert(point_index, (x, y, x, y))
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    max_queries = int(config.MAX_NEIGHBOR_QUERIES)
    neighbor_radius = float(config.NEIGHBOR_RADIUS)
    for batch, features, masks in zip(batches, image_features, model_masks):
        point_sets, pair_sets, valid_sets, index_maps = [], [], [], []
        for x0, y0, x1, y1 in batch:
            point_indices = list(spatial_index.intersection((x0, y0, x1, y1)))
            index_maps.append({local: full for local, full in enumerate(point_indices)})
            patch_points = graph_points[point_indices] - np.asarray([[x0, y0]], dtype=np.float32)
            point_count = len(patch_points)
            if point_count:
                tree = scipy.spatial.KDTree(patch_points)
                _distances, neighbor_indices = tree.query(
                    patch_points,
                    k=max_queries + 1,
                    distance_upper_bound=neighbor_radius,
                )
                neighbor_indices = neighbor_indices[:, 1:]
            else:
                neighbor_indices = np.empty((0, max_queries), dtype=np.int64)
            source = np.tile(np.arange(point_count)[:, np.newaxis], (1, max_queries))
            pair_valid = neighbor_indices < point_count
            target = np.where(pair_valid, neighbor_indices, source)
            point_sets.append(patch_points)
            pair_sets.append(np.stack([source, target], axis=-1))
            valid_sets.append(pair_valid)
        padded_count = max((len(points) for points in point_sets), default=0)
        if padded_count == 0:
            continue
        collated_points = np.stack([
            np.pad(points, [(0, padded_count - len(points)), (0, 0)])
            for points in point_sets
        ])
        collated_pairs = np.stack([
            np.pad(pairs, [(0, padded_count - len(pairs)), (0, 0), (0, 0)])
            for pairs in pair_sets
        ])
        collated_valid = np.stack([
            np.pad(values, [(0, padded_count - len(values)), (0, 0)])
            for values in valid_sets
        ])
        with torch.inference_mode():
            topology = model.infer_toponet(
                features,
                torch.tensor(collated_points, device=device),
                torch.tensor(collated_pairs, device=device),
                torch.tensor(collated_valid, device=device),
                masks,
            )
        topology = torch.where(torch.isnan(topology), -100.0, topology).squeeze(-1).cpu().numpy()
        for batch_index in range(topology.shape[0]):
            for source_index in range(topology.shape[1]):
                for pair_index in range(topology.shape[2]):
                    if not collated_valid[batch_index, source_index, pair_index]:
                        continue
                    local_source, local_target = collated_pairs[
                        batch_index, source_index, pair_index
                    ]
                    full_source = index_maps[batch_index][int(local_source)]
                    full_target = index_maps[batch_index][int(local_target)]
                    score = float(topology[batch_index, source_index, pair_index])
                    edge_scores[(full_source, full_target)] += score
                    edge_counts[(full_source, full_target)] += 1.0

    edges, scores = [], []
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge]
        if score > float(config.TOPO_THRESHOLD):
            edges.append(edge)
            scores.append(score)
    timing["topology_seconds"] = time.perf_counter() - topology_started
    return (
        graph_points[:, ::-1].astype(np.float32),
        np.asarray(edges, dtype=np.int32).reshape(-1, 2),
        np.asarray(scores, dtype=np.float32),
        keypoint_u8,
        road_u8,
    )
