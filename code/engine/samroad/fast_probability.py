from __future__ import annotations

"""Small, threshold-aligned local-contrast enhancement for Fast inference."""

import cv2
import numpy as np


FAST_RELATIVE_SIGMA = 15.0
FAST_RELATIVE_EPS = 0.005
FAST_RELATIVE_RATIO_SATURATION = 4.0
FAST_RELATIVE_THRESHOLD_MARGIN = 0.02


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
    """Apply one bounded local-relative boost without making topology decisions."""
    probability = _probability01(road_probability)
    p50, p95, p99 = (
        (float(value) for value in np.percentile(probability, (50, 95, 99)))
        if probability.size else (0.0, 0.0, 0.0)
    )
    adaptive_floor = float(np.clip(
        p50 + 0.05 * (p99 - p50),
        0.003,
        0.015,
    ))
    boost_target = float(min(
        1.0,
        float(high_threshold) + FAST_RELATIVE_THRESHOLD_MARGIN,
    ))
    local_background = cv2.GaussianBlur(
        probability,
        (0, 0),
        sigmaX=FAST_RELATIVE_SIGMA,
        sigmaY=FAST_RELATIVE_SIGMA,
    )
    contrast = np.maximum(probability - local_background, 0.0)
    relative_ratio = contrast / (local_background + FAST_RELATIVE_EPS)
    relative_confidence = np.clip(
        relative_ratio / FAST_RELATIVE_RATIO_SATURATION,
        0.0,
        1.0,
    )
    relative_confidence *= probability >= adaptive_floor
    enhanced = np.maximum(
        probability,
        boost_target * relative_confidence,
    ).clip(0.0, 1.0).astype(np.float32)
    boosted = enhanced > probability + 1e-7
    diagnostics: dict[str, float | int] = {
        "raw_probability_p50": p50,
        "raw_probability_p95": p95,
        "raw_probability_p99": p99,
        "relative_boost_pixel_count": int(np.count_nonzero(boosted)),
        "raw_probability_mean": float(probability.mean()),
        "enhanced_probability_mean": float(enhanced.mean()),
        "enhanced_probability_max": float(enhanced.max()) if enhanced.size else 0.0,
        "fast_graph_high_threshold": float(high_threshold),
        "fast_relative_boost_target": boost_target,
        "fast_relative_adaptive_floor": adaptive_floor,
        "raw_pixels_above_graph_threshold": int(np.count_nonzero(
            probability >= float(high_threshold)
        )),
        "enhanced_pixels_above_graph_threshold": int(np.count_nonzero(
            enhanced >= float(high_threshold)
        )),
    }
    return enhanced, diagnostics
