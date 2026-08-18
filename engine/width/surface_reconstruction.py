from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SurfaceReconstructionConfig:
    low_probability: float = 0.30
    high_probability: float = 0.55
    min_corridor_px: float = 10.0
    max_corridor_px: float = 60.0
    corridor_width_factor: float = 2.2
    seed_radius_px: int = 3
    close_kernel: int = 5
    open_kernel: int = 3
    min_area_px: int = 80


@dataclass
class SurfaceReconstructionResult:
    surface: np.ndarray
    added: np.ndarray
    removed: np.ndarray
    uncertain: np.ndarray
    centerline: np.ndarray
    metadata: dict


def rasterize_centerline(shape: tuple[int, int], nodes_rc: np.ndarray, edges: np.ndarray) -> np.ndarray:
    line = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    for src_idx, dst_idx in np.asarray(edges, dtype=np.int32).reshape(-1, 2).tolist():
        r0, c0 = np.asarray(nodes_rc[src_idx], dtype=np.float32)
        r1, c1 = np.asarray(nodes_rc[dst_idx], dtype=np.float32)
        p0 = (int(np.clip(round(float(c0)), 0, width - 1)), int(np.clip(round(float(r0)), 0, height - 1)))
        p1 = (int(np.clip(round(float(c1)), 0, width - 1)), int(np.clip(round(float(r1)), 0, height - 1)))
        cv2.line(line, p0, p1, 1, 1, cv2.LINE_8)
    return line


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    result = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            result[labels == label] = 1
    return result


def _estimated_corridor_radius(
    original: np.ndarray,
    centerline: np.ndarray,
    config: SurfaceReconstructionConfig,
) -> tuple[float, int]:
    inside_distance = cv2.distanceTransform(original.astype(np.uint8), cv2.DIST_L2, 5)
    samples = inside_distance[centerline > 0]
    samples = samples[samples > 0]
    if samples.size:
        half_width = float(np.percentile(samples, 90))
        radius = half_width * config.corridor_width_factor
    else:
        half_width = 0.0
        radius = config.min_corridor_px
    radius = float(np.clip(radius, config.min_corridor_px, config.max_corridor_px))
    return radius, int(samples.size)


def reconstruct_surface(
    road_probability: np.ndarray,
    original_surface: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    config: SurfaceReconstructionConfig | None = None,
) -> SurfaceReconstructionResult:
    """Grow a probability-defined road surface from the corrected centerline graph."""
    config = config or SurfaceReconstructionConfig()
    probability = np.asarray(road_probability, dtype=np.float32)
    if probability.max(initial=0.0) > 1.0:
        probability = probability / 255.0
    probability = np.clip(probability, 0.0, 1.0)
    original = (np.asarray(original_surface) > 0).astype(np.uint8)
    if probability.shape != original.shape:
        raise ValueError(f"Probability/surface shape mismatch: {probability.shape} != {original.shape}")

    centerline = rasterize_centerline(original.shape, nodes_rc, edges)
    if not np.any(centerline):
        empty = np.zeros_like(original)
        return SurfaceReconstructionResult(
            surface=empty, added=empty.copy(), removed=original.copy(), uncertain=empty.copy(),
            centerline=centerline,
            metadata={"status": "empty_centerline", "corridor_radius_px": 0.0},
        )

    corridor_radius, width_sample_count = _estimated_corridor_radius(original, centerline, config)
    distance_to_line = cv2.distanceTransform((centerline == 0).astype(np.uint8), cv2.DIST_L2, 5)
    corridor = distance_to_line <= corridor_radius
    low_candidate = (probability >= config.low_probability) & corridor
    high_candidate = probability >= config.high_probability

    seed_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.seed_radius_px * 2 + 1, config.seed_radius_px * 2 + 1),
    )
    near_line = cv2.dilate(centerline, seed_kernel) > 0
    seeds = low_candidate & near_line
    # A high-confidence surface may be a few pixels away from a slightly displaced centerline.
    seeds |= high_candidate & (distance_to_line <= max(config.seed_radius_px * 2, corridor_radius * 0.35))

    component_count, labels = cv2.connectedComponents(low_candidate.astype(np.uint8), connectivity=8)
    selected_labels = np.unique(labels[seeds])
    selected_labels = selected_labels[selected_labels > 0]
    selected = np.isin(labels, selected_labels).astype(np.uint8) if selected_labels.size else np.zeros_like(original)

    if config.close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel))
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
    if config.open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.open_kernel, config.open_kernel))
        selected = cv2.morphologyEx(selected, cv2.MORPH_OPEN, kernel)
    selected &= corridor.astype(np.uint8)
    selected = _remove_small_components(selected, config.min_area_px)

    added = ((selected > 0) & (original == 0)).astype(np.uint8)
    removed = ((original > 0) & (selected == 0)).astype(np.uint8)
    probability_band = (probability >= config.low_probability) & (probability < config.high_probability)
    boundary = cv2.morphologyEx(selected, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    uncovered_line = (centerline > 0) & (selected == 0)
    uncertain = ((probability_band & corridor & (boundary | near_line)) | uncovered_line).astype(np.uint8)

    return SurfaceReconstructionResult(
        surface=selected,
        added=added,
        removed=removed,
        uncertain=uncertain,
        centerline=centerline,
        metadata={
            "status": "ok",
            "low_probability": config.low_probability,
            "high_probability": config.high_probability,
            "corridor_radius_px": corridor_radius,
            "centerline_width_sample_count": width_sample_count,
            "candidate_component_count": max(0, component_count - 1),
            "selected_component_count": int(selected_labels.size),
            "original_surface_px": int(np.count_nonzero(original)),
            "reconstructed_surface_px": int(np.count_nonzero(selected)),
            "added_surface_px": int(np.count_nonzero(added)),
            "removed_surface_px": int(np.count_nonzero(removed)),
            "uncertain_surface_px": int(np.count_nonzero(uncertain)),
            "uncovered_centerline_px": int(np.count_nonzero(uncovered_line)),
        },
    )
