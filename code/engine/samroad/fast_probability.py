from __future__ import annotations

"""Small, fixed local-contrast enhancement used only by Fast inference."""

import cv2
import numpy as np


FAST_RELATIVE_SIGMA = 15.0
FAST_RELATIVE_EPS = 0.005
FAST_RELATIVE_ABSOLUTE_FLOOR = 0.015
FAST_RELATIVE_RATIO_SATURATION = 4.0
FAST_RELATIVE_MAX_BOOST = 0.25


def _probability01(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float32)
    if values.size and float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def build_fast_enhanced_road_probability(
    road_probability: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply one bounded local-relative boost without making topology decisions."""
    probability = _probability01(road_probability)
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
    relative_confidence *= probability >= FAST_RELATIVE_ABSOLUTE_FLOOR
    enhanced = np.maximum(
        probability,
        FAST_RELATIVE_MAX_BOOST * relative_confidence,
    ).clip(0.0, 1.0).astype(np.float32)
    boosted = enhanced > probability + 1e-7
    diagnostics: dict[str, float | int] = {
        "raw_probability_p50": float(np.percentile(probability, 50)),
        "raw_probability_p95": float(np.percentile(probability, 95)),
        "raw_probability_p99": float(np.percentile(probability, 99)),
        "relative_boost_pixel_count": int(np.count_nonzero(boosted)),
        "raw_probability_mean": float(probability.mean()),
        "enhanced_probability_mean": float(enhanced.mean()),
        "enhanced_probability_max": float(enhanced.max()) if enhanced.size else 0.0,
    }
    return enhanced, diagnostics
