from __future__ import annotations

"""Small, threshold-aligned local-contrast enhancement for Fast inference."""

import cv2
import numpy as np


FAST_RELATIVE_SIGMA = 25.0
FAST_RELATIVE_EPS = 0.003
FAST_RELATIVE_MIN_CONTRAST = 0.004
FAST_RELATIVE_MIN_RATIO = 0.8


def _probability01(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float32)
    if values.size and float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def build_fast_enhanced_road_probability(
    road_probability: np.ndarray,
    *,
    high_threshold: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Grant locally distinct Fast pixels eligibility for native graph extraction."""
    probability = _probability01(road_probability)
    p50, p99 = (
        (float(value) for value in np.percentile(probability, (50, 99)))
        if probability.size else (0.0, 0.0)
    )
    adaptive_floor = float(np.clip(
        p50 + 0.03 * (p99 - p50),
        0.001,
        0.010,
    ))
    graph_candidate_level = float(min(
        1.0,
        max(float(high_threshold) + 0.05, 0.50),
    ))
    local_background = cv2.GaussianBlur(
        probability,
        (0, 0),
        sigmaX=FAST_RELATIVE_SIGMA,
        sigmaY=FAST_RELATIVE_SIGMA,
    )
    contrast = np.maximum(probability - local_background, 0.0)
    relative_ratio = contrast / (local_background + FAST_RELATIVE_EPS)
    relative_candidate = (
        (probability >= adaptive_floor)
        & (contrast >= FAST_RELATIVE_MIN_CONTRAST)
        & (relative_ratio >= FAST_RELATIVE_MIN_RATIO)
    )
    relative_candidate = cv2.morphologyEx(
        relative_candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    ) > 0
    graph_probability = probability.copy()
    graph_probability[relative_candidate] = np.maximum(
        graph_probability[relative_candidate],
        graph_candidate_level,
    )
    graph_probability = graph_probability.clip(0.0, 1.0).astype(np.float32)
    diagnostics: dict[str, float | int] = {
        "relative_candidate_pixel_count": int(np.count_nonzero(relative_candidate)),
    }
    return graph_probability, diagnostics
