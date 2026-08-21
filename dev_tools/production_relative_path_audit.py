from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMROAD = ROOT / "engine" / "samroad"
if str(SAMROAD) not in sys.path:
    sys.path.insert(0, str(SAMROAD))

import graph_extraction  # noqa: E402
from utils import load_config  # noqa: E402


def _panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.resize(image, (480, 320), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (0, 0), (480, 34), (255, 255, 255), -1)
    cv2.putText(
        panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, (20, 20, 20), 1, cv2.LINE_AA,
    )
    return panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "docs" / "experiment_results" / "production_regularized_skeleton_20260821"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "config" / "samroad_inference.yaml"
    config = load_config(config_path)
    probability = np.full((128, 192), 0.012, dtype=np.float32)
    cv2.line(probability, (12, 65), (180, 65), 0.22, 11)
    cv2.line(probability, (96, 12), (96, 116), 0.19, 9)
    cv2.circle(probability, (56, 65), 3, 0.012, -1)
    cv2.line(probability, (128, 61), (145, 69), 0.012, 2)

    profile = graph_extraction.resolve_effective_road_profile(probability, config)
    config.ROAD_THRESHOLD_PROFILE = profile["effective_profile"]
    context = graph_extraction.compute_relative_roadness(
        probability, config, scene_state=profile["scene_confidence_state"]
    )
    diagnostics = context["diagnostics"]
    performance = diagnostics["relative_skeleton_performance_audit"]

    probability_panel = cv2.applyColorMap(
        np.clip(probability * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )
    candidate_panel = cv2.cvtColor(
        context["relative_candidate_mask"].astype(np.uint8) * 150,
        cv2.COLOR_GRAY2BGR,
    )
    candidate_panel[context["relative_regularized_raw_skeleton"] > 0] = (40, 40, 255)
    regularized_panel = cv2.cvtColor(
        context["relative_regularized_candidate"].astype(np.uint8) * 150,
        cv2.COLOR_GRAY2BGR,
    )
    regularized_panel[context["relative_filled_hole_mask"] > 0] = (0, 165, 255)
    final_panel = probability_panel.copy()
    final_panel[context["relative_regularized_final_skeleton"] > 0] = (0, 255, 0)
    final_panel[context["relative_removed_cycle_mask"] > 0] = (255, 0, 255)
    final_panel[context["pruned_spur_mask"] > 0] = (0, 0, 255)

    comparison = np.concatenate([
        np.concatenate([
            _panel(probability_panel, "1 Production probability"),
            _panel(candidate_panel, "2 Candidate + raw skeleton"),
        ], axis=1),
        np.concatenate([
            _panel(regularized_panel, "3 Regularized + filled holes"),
            _panel(final_panel, "4 Final regularized skeleton"),
        ], axis=1),
    ], axis=0)
    image_path = output_dir / "production_relative_path_result.png"
    cv2.imwrite(str(image_path), comparison)

    audit = {
        "config": str(config_path.resolve()),
        "requested_threshold_profile": "auto",
        "effective_threshold_profile": profile["effective_profile"],
        "relative_roadness_enabled": bool(config.RELATIVE_ROADNESS_ENABLED),
        "relative_centerline_method": "regularized_skeleton",
        "regularized_skeleton_active": bool(
            diagnostics["regularized_skeleton_experimental_active"]
        ),
        "continuous_tracing_active": bool(
            diagnostics["continuous_tracing_experimental_active"]
        ),
        "junction_collapse_active": bool(
            diagnostics["relative_junction_collapse_experimental_active"]
        ),
        "endpoint_segment_recovery_active": bool(config.WEAK_SEGMENT_RECOVERY_ENABLED),
        "hole_crack_fill_active": "hole_fill_seconds" in performance,
        "width_aware_spur_pruning_active": "spur_pruning_seconds" in performance,
        "small_cycle_cleanup_active": "cycle_cleanup_seconds" in performance,
        "candidate_pixel_count_before": int(performance["candidate_pixel_count_before"]),
        "candidate_pixel_count_after": int(performance["candidate_pixel_count_after"]),
        "filled_hole_count": int(performance["hole_filled_count"]),
        "removed_spur_count": int(performance["spur_removed_count"]),
        "removed_cycle_count": int(performance["removed_cycle_count"]),
        "final_skeleton_length": int(performance["final_skeleton_length"]),
        "timing_seconds": {
            key: float(performance[key]) for key in (
                "hole_fill_seconds", "spur_pruning_seconds",
                "cycle_cleanup_seconds", "total_seconds",
            )
        },
        "result_image": str(image_path),
    }
    (output_dir / "production_relative_path_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
