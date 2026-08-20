from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMROAD_ROOT = REPO_ROOT / "engine" / "samroad"
DEV_TOOL_ROOT = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, SAMROAD_ROOT, DEV_TOOL_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import graph_extraction  # noqa: E402
import graph_utils  # noqa: E402
import single_image_inference  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "config" / "samroad_inference.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "samroad" / "samroad.ckpt"
DEFAULT_OUTPUTS = Path(__file__).resolve().parent / "outputs"

WEAK_CANDIDATE_FIELDS = (
    "candidate_type", "start", "end", "distance", "direction_cosine",
    "path_length", "path_ratio", "mean_probability", "q25_probability",
    "weak_fraction", "background_probability", "background_contrast",
    "surface_probability", "accepted", "reject_reason",
    "component_before_start", "component_before_target", "merges_components",
    "component_count_before", "component_count_after", "connectivity_gain",
)
ENDPOINT_SEGMENT_CANDIDATE_FIELDS = (
    "candidate_id", "endpoint_node", "target_segment", "target_projection",
    "distance", "direction_cosine", "path_length", "path_ratio",
    "mean_probability", "q25_probability", "weak_fraction",
    "background_probability", "background_contrast",
    "start_component", "target_component", "connectivity_gain",
    "accepted", "reject_reason", "recovery_score",
)
BOOTSTRAP_CANDIDATE_FIELDS = (
    "path_length", "direct_distance", "tortuosity", "mean_probability",
    "q25_probability", "weak_fraction", "background_probability",
    "background_contrast", "connection_count", "endpoint_alignment",
    "absolute_proximity_fraction", "accepted", "decision", "qa_state",
    "reject_reason", "review_reason", "relative_evidence_tier",
    "candidate_source", "scene_rank_mean", "local_background_mean",
    "local_contrast_mean", "normalized_contrast_mean",
    "relative_score_mean", "relative_score_q25", "relative_fraction",
    "scale_agreement_mean", "scale_agreement_q25",
    "topology_candidate_support_fraction", "topology_candidate_score_mean",
    "corridor_id", "micro_chain_length", "corridor_total_length",
    "rescued_by_corridor", "short_chain_classification",
    "path",
)

OVERRIDES = {
    "road_high_threshold": "ROAD_HIGH_THRESHOLD",
    "road_low_threshold": "ROAD_LOW_THRESHOLD",
    "max_gap": "WEAK_RECOVERY_MAX_GAP_PX",
    "max_extension": "WEAK_RECOVERY_MAX_EXTENSION_PX",
    "min_direction_cosine": "WEAK_RECOVERY_MIN_DIRECTION_COSINE",
    "min_mean_probability": "WEAK_RECOVERY_MIN_MEAN_PROBABILITY",
    "min_q25_probability": "WEAK_RECOVERY_MIN_Q25_PROBABILITY",
    "min_weak_fraction": "WEAK_RECOVERY_MIN_WEAK_FRACTION",
    "min_background_contrast": "WEAK_RECOVERY_MIN_BACKGROUND_CONTRAST",
    "auto_score": "WEAK_RECOVERY_AUTO_SCORE",
    "max_segment_distance": "WEAK_SEGMENT_RECOVERY_MAX_DISTANCE_PX",
    "min_segment_direction_cosine": "WEAK_SEGMENT_RECOVERY_MIN_DIRECTION_COSINE",
    "direction_lookback": "WEAK_ENDPOINT_DIRECTION_LOOKBACK_PX",
    "bootstrap_min_length": "WEAK_BOOTSTRAP_MIN_LENGTH_PX",
    "bootstrap_min_mean_probability": "WEAK_BOOTSTRAP_MIN_MEAN_PROBABILITY",
    "bootstrap_min_q25_probability": "WEAK_BOOTSTRAP_MIN_Q25_PROBABILITY",
    "bootstrap_min_background_contrast": "WEAK_BOOTSTRAP_MIN_BACKGROUND_CONTRAST",
    "bootstrap_max_tortuosity": "WEAK_BOOTSTRAP_MAX_TORTUOSITY",
    "bootstrap_min_weak_fraction": "WEAK_BOOTSTRAP_MIN_WEAK_FRACTION",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Independent SAMRoad weak-road recovery workbench")
    result.add_argument("--image", help="Single image to infer during the first/full run")
    result.add_argument("--run-dir", help="Output/cache directory; required with --recovery-only")
    result.add_argument("--recovery-only", action="store_true", help="Reuse cached probability and original graph")
    result.add_argument("--config", help=f"SAMRoad YAML (default: {DEFAULT_CONFIG})")
    result.add_argument("--checkpoint", help=f"SAMRoad checkpoint (default: {DEFAULT_CHECKPOINT})")
    result.add_argument("--device", default=None, help="cuda, cpu, or auto (default: auto)")
    result.add_argument("--batch-size", type=int, help="Override INFER_BATCH_SIZE for this run")
    result.add_argument("--threshold-profile", help="ROAD_THRESHOLD_PROFILES entry for this test run")
    bootstrap_group = result.add_mutually_exclusive_group()
    bootstrap_group.add_argument("--enable-bootstrap", dest="bootstrap_enabled", action="store_true")
    bootstrap_group.add_argument("--disable-bootstrap", dest="bootstrap_enabled", action="store_false")
    result.set_defaults(bootstrap_enabled=None)
    relative_group = result.add_mutually_exclusive_group()
    relative_group.add_argument(
        "--enable-relative-roadness", dest="relative_roadness_enabled", action="store_true"
    )
    relative_group.add_argument(
        "--disable-relative-roadness", dest="relative_roadness_enabled", action="store_false"
    )
    result.set_defaults(relative_roadness_enabled=None)
    segment_group = result.add_mutually_exclusive_group()
    segment_group.add_argument(
        "--enable-segment-recovery", dest="segment_recovery_enabled", action="store_true"
    )
    segment_group.add_argument(
        "--disable-segment-recovery", dest="segment_recovery_enabled", action="store_false"
    )
    result.set_defaults(segment_recovery_enabled=None)
    result.add_argument("--road-high-threshold", type=float)
    result.add_argument("--road-low-threshold", type=float)
    result.add_argument("--max-gap", type=float)
    result.add_argument("--max-extension", type=float)
    result.add_argument("--min-direction-cosine", type=float)
    result.add_argument("--min-mean-probability", type=float)
    result.add_argument("--min-q25-probability", type=float)
    result.add_argument("--min-weak-fraction", type=float)
    result.add_argument("--min-background-contrast", type=float)
    result.add_argument("--auto-score", type=float)
    result.add_argument("--max-segment-distance", type=float)
    result.add_argument("--min-segment-direction-cosine", type=float)
    result.add_argument("--direction-lookback", type=float)
    result.add_argument("--bootstrap-min-length", type=float)
    result.add_argument("--bootstrap-min-mean-probability", type=float)
    result.add_argument("--bootstrap-min-q25-probability", type=float)
    result.add_argument("--bootstrap-min-background-contrast", type=float)
    result.add_argument("--bootstrap-max-tortuosity", type=float)
    result.add_argument("--bootstrap-min-weak-fraction", type=float)
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_candidate_csv(path: Path, fieldnames, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            for name in ("start", "end", "target_projection", "endpoint_node", "target_segment", "path"):
                if isinstance(payload.get(name), (list, tuple)):
                    payload[name] = json.dumps(payload[name], ensure_ascii=False)
            writer.writerow(payload)


def _progress(message: str) -> None:
    print(message, flush=True)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = image
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), value):
        raise RuntimeError(f"Cannot write image: {path}")


def save_graph(path: Path, nodes_rc: np.ndarray, edges: np.ndarray) -> None:
    graph = graph_utils.convert_to_sat2graph_format(nodes_rc, edges)
    with path.open("wb") as file:
        pickle.dump(graph, file)


def load_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        graph = pickle.load(file)
    nodes, directed_edges = graph_utils.convert_from_sat2graph_format(graph)
    unique_edges = sorted({
        tuple(sorted((int(src_idx), int(dst_idx))))
        for src_idx, dst_idx in directed_edges
        if int(src_idx) != int(dst_idx)
    })
    return (
        np.asarray(nodes, dtype=np.float32).reshape(-1, 2),
        np.asarray(unique_edges, dtype=np.int32).reshape(-1, 2),
    )


def write_original_scores(
    path: Path,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    scores: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "edge_id", "src_row", "src_col", "dst_row", "dst_col", "topology_probability",
        ])
        writer.writeheader()
        for edge_id, ((src_idx, dst_idx), score) in enumerate(zip(edges.tolist(), scores.tolist())):
            src, dst = nodes_rc[int(src_idx)], nodes_rc[int(dst_idx)]
            writer.writerow({
                "edge_id": edge_id,
                "src_row": float(src[0]), "src_col": float(src[1]),
                "dst_row": float(dst[0]), "dst_col": float(dst[1]),
                "topology_probability": float(score),
            })


def read_original_scores(path: Path, nodes_rc: np.ndarray, edges: np.ndarray) -> np.ndarray:
    by_coordinates = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            src = (int(round(float(row["src_row"]))), int(round(float(row["src_col"]))))
            dst = (int(round(float(row["dst_row"]))), int(round(float(row["dst_col"]))))
            by_coordinates[tuple(sorted((src, dst)))] = float(row["topology_probability"])
    scores = []
    for src_idx, dst_idx in edges.tolist():
        src = tuple(int(round(value)) for value in nodes_rc[int(src_idx)])
        dst = tuple(int(round(value)) for value in nodes_rc[int(dst_idx)])
        scores.append(by_coordinates.get(tuple(sorted((src, dst))), float("nan")))
    return np.asarray(scores, dtype=np.float32)


def load_original_scored_graph(
    graph_path: Path,
    score_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reload the exact directed edge list saved by a full test run."""
    nodes, _unique_edges = load_graph(graph_path)
    node_by_coordinate = {
        tuple(int(round(value)) for value in point): node_idx
        for node_idx, point in enumerate(nodes)
    }
    edges = []
    scores = []
    with score_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            src = (int(round(float(row["src_row"]))), int(round(float(row["src_col"]))))
            dst = (int(round(float(row["dst_row"]))), int(round(float(row["dst_col"]))))
            if src not in node_by_coordinate or dst not in node_by_coordinate:
                raise ValueError(
                    f"Cached score edge references a missing graph node: {src} -> {dst}"
                )
            edges.append((node_by_coordinate[src], node_by_coordinate[dst]))
            scores.append(float(row["topology_probability"]))
    return (
        nodes,
        np.asarray(edges, dtype=np.int32).reshape(-1, 2),
        np.asarray(scores, dtype=np.float32),
    )


def _apply_overrides(config, arguments: argparse.Namespace) -> None:
    if arguments.threshold_profile:
        profiles = config.get("ROAD_THRESHOLD_PROFILES", {}) or {}
        if arguments.threshold_profile != "auto" and arguments.threshold_profile not in profiles:
            raise ValueError(
                f"Unknown threshold profile {arguments.threshold_profile!r}; "
                f"available: auto, {', '.join(sorted(profiles))}"
            )
        config.ROAD_THRESHOLD_PROFILE = arguments.threshold_profile
    threshold_overridden = (
        arguments.road_high_threshold is not None or arguments.road_low_threshold is not None
    )
    if threshold_overridden:
        base_profile = str(config.get("ROAD_THRESHOLD_PROFILE", "default"))
        if base_profile == "auto":
            base_profile = str(config.get("SCENE_DIAGNOSTIC_REFERENCE_PROFILE", "default"))
        resolved_high, resolved_low, _profile = graph_extraction.resolve_road_thresholds(
            config, profile_name=base_profile
        )
        high = arguments.road_high_threshold if arguments.road_high_threshold is not None else resolved_high
        low = arguments.road_low_threshold if arguments.road_low_threshold is not None else resolved_low
        if not (np.isclose(high, resolved_high) and np.isclose(low, resolved_low)):
            profiles = dict(config.get("ROAD_THRESHOLD_PROFILES", {}) or {})
            profiles["weak_recovery_test_cli"] = {
                "ROAD_HIGH_THRESHOLD": float(high), "ROAD_LOW_THRESHOLD": float(low),
            }
            config.ROAD_THRESHOLD_PROFILES = profiles
            config.ROAD_THRESHOLD_PROFILE = "weak_recovery_test_cli"
    for argument_name, config_name in OVERRIDES.items():
        value = getattr(arguments, argument_name)
        if value is not None and argument_name not in {"road_high_threshold", "road_low_threshold"}:
            config[config_name] = value
    if arguments.batch_size is not None:
        if arguments.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        config.INFER_BATCH_SIZE = int(arguments.batch_size)
    if arguments.bootstrap_enabled is not None:
        config.WEAK_BOOTSTRAP_ENABLED = bool(arguments.bootstrap_enabled)
    if arguments.relative_roadness_enabled is not None:
        config.RELATIVE_ROADNESS_ENABLED = bool(arguments.relative_roadness_enabled)
    if arguments.segment_recovery_enabled is not None:
        config.WEAK_SEGMENT_RECOVERY_ENABLED = bool(arguments.segment_recovery_enabled)


def _read_probability(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read cached probability: {path}")
    return image.astype(np.float32) / 255.0


def _draw_edges(
    image_rgb: np.ndarray,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    result = image_rgb
    for src_idx, dst_idx in edges.tolist():
        src, dst = nodes_rc[int(src_idx)], nodes_rc[int(dst_idx)]
        cv2.line(
            result,
            (int(round(src[1])), int(round(src[0]))),
            (int(round(dst[1])), int(round(dst[0]))),
            color,
            thickness,
            cv2.LINE_AA,
        )
    return result


def _labeled_preview(image_rgb: np.ndarray, label: str, max_width: int = 1600) -> np.ndarray:
    scale = min(1.0, max_width / max(image_rgb.shape[1], 1))
    if scale < 1.0:
        image_rgb = cv2.resize(
            image_rgb,
            (int(round(image_rgb.shape[1] * scale)), int(round(image_rgb.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    result = image_rgb.copy()
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 520), 54), (0, 0, 0), -1)
    cv2.putText(result, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def write_visualizations(
    run_dir: Path,
    image_rgb: np.ndarray,
    original_nodes: np.ndarray,
    original_edges: np.ndarray,
    recovered_nodes: np.ndarray,
    recovered_edges: np.ndarray,
    metadata: list[dict],
) -> None:
    original_overlay = _draw_edges(
        image_rgb.copy(), original_nodes, original_edges, (255, 220, 0), 3
    )
    recovered_overlay = image_rgb.copy()
    source_styles = {
        "samroad": ((255, 220, 0), 3),
        "relative_roadness": ((0, 255, 0), 4),
        "weak_recovered": ((0, 255, 255), 5),
        "weak_bootstrap": ((255, 0, 255), 5),
        "relative_bootstrap": ((255, 64, 255), 5),
        "weak_segment_connector": ((255, 128, 0), 6),
    }
    for source, (color, thickness) in source_styles.items():
        edge_ids = [
            edge_id for edge_id, row in enumerate(metadata)
            if row.get("line_source") == source
        ]
        if edge_ids:
            recovered_overlay = _draw_edges(
                recovered_overlay,
                recovered_nodes,
                recovered_edges[np.asarray(edge_ids, dtype=np.int32)],
                color,
                thickness,
            )
    _save_png(run_dir / "original_overlay.png", original_overlay)
    _save_png(run_dir / "recovered_overlay.png", recovered_overlay)
    _save_png(run_dir / "bootstrap_overlay.png", recovered_overlay)
    left = _labeled_preview(original_overlay, "Original SAMRoad (yellow)")
    right = _labeled_preview(
        recovered_overlay,
        "Final: strong yellow / relative magenta / weak cyan / segment orange",
    )
    target_height = min(left.shape[0], right.shape[0])
    if left.shape[0] != target_height:
        left = cv2.resize(left, (int(left.shape[1] * target_height / left.shape[0]), target_height))
    if right.shape[0] != target_height:
        right = cv2.resize(right, (int(right.shape[1] * target_height / right.shape[0]), target_height))
    _save_png(run_dir / "recovery_compare.png", np.concatenate([left, right], axis=1))


def _candidate_crop_bounds(points, image_shape, margin=100, minimum_size=256):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    y0 = int(np.floor(points[:, 0].min())) - margin
    y1 = int(np.ceil(points[:, 0].max())) + margin + 1
    x0 = int(np.floor(points[:, 1].min())) - margin
    x1 = int(np.ceil(points[:, 1].max())) + margin + 1
    height, width = image_shape[:2]

    def expand(low, high, limit):
        if high - low < minimum_size:
            extra = minimum_size - (high - low)
            low -= extra // 2
            high += extra - extra // 2
        if low < 0:
            high -= low
            low = 0
        if high > limit:
            low -= high - limit
            high = limit
        return max(0, low), min(limit, high)

    y0, y1 = expand(y0, y1, height)
    x0, x1 = expand(x0, x1, width)
    return x0, y0, x1, y1


def _write_candidate_montage(path: Path, image_paths: list[Path]) -> None:
    if not image_paths:
        montage = np.full((300, 760, 3), 25, dtype=np.uint8)
        cv2.putText(
            montage, "No accepted endpoint-to-segment candidates", (75, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA,
        )
        _save_png(path, montage)
        return
    panels = []
    tile_width, tile_height = 760, 420
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        panel = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        x0 = (tile_width - resized.shape[1]) // 2
        y0 = (tile_height - resized.shape[0]) // 2
        panel[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        panels.append(panel)
    columns = min(3, len(panels))
    while len(panels) % columns:
        panels.append(np.zeros_like(panels[0]))
    montage = np.vstack([
        np.hstack(panels[index:index + columns])
        for index in range(0, len(panels), columns)
    ])
    _save_png(path, montage)


def write_endpoint_segment_candidate_visualizations(
    run_dir: Path,
    image_rgb: np.ndarray,
    recovered_nodes: np.ndarray,
    recovered_edges: np.ndarray,
    metadata: list[dict],
    candidate_audit: list[dict],
) -> None:
    output_dir = (run_dir / "endpoint_segment_candidates").resolve()
    if output_dir.parent != run_dir.resolve():
        raise ValueError(f"Unsafe endpoint segment output directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    accepted = [row for row in candidate_audit if row.get("accepted")]
    image_paths = []
    for row in accepted:
        path_points = np.asarray(row.get("_path") or [], dtype=np.float32).reshape(-1, 2)
        target_nodes = np.asarray(row.get("_target_nodes") or [], dtype=np.float32).reshape(-1, 2)
        points = np.vstack([path_points, target_nodes])
        x0, y0, x1, y1 = _candidate_crop_bounds(points, image_rgb.shape)
        crop = image_rgb[y0:y1, x0:x1].copy()

        def xy(point):
            return int(round(float(point[1]) - x0)), int(round(float(point[0]) - y0))

        source_styles = {
            "samroad": ((255, 220, 0), 2),
            "weak_recovered": ((0, 255, 255), 3),
            "weak_bootstrap": ((255, 0, 255), 3),
        }
        for edge_id, (src_idx, dst_idx) in enumerate(recovered_edges.tolist()):
            source = metadata[edge_id].get("line_source")
            if source not in source_styles:
                continue
            src, dst = recovered_nodes[int(src_idx)], recovered_nodes[int(dst_idx)]
            if (
                max(src[1], dst[1]) < x0 or min(src[1], dst[1]) >= x1
                or max(src[0], dst[0]) < y0 or min(src[0], dst[0]) >= y1
            ):
                continue
            color, thickness = source_styles[source]
            cv2.line(crop, xy(src), xy(dst), color, thickness, cv2.LINE_AA)
        if len(target_nodes) == 2:
            cv2.line(crop, xy(target_nodes[0]), xy(target_nodes[1]), (255, 255, 0), 6, cv2.LINE_AA)
        if len(path_points) >= 2:
            cv2.polylines(
                crop, [np.asarray([xy(point) for point in path_points], dtype=np.int32)],
                False, (255, 128, 0), 5, cv2.LINE_AA,
            )
        endpoint = recovered_nodes[int(row["endpoint_node"])]
        projection = np.asarray(row["target_projection"], dtype=np.float32)
        cv2.circle(crop, xy(endpoint), 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(crop, xy(projection), 8, (255, 0, 0), -1, cv2.LINE_AA)

        canvas_width = max(760, crop.shape[1])
        if crop.shape[1] < canvas_width:
            canvas = np.zeros((crop.shape[0], canvas_width, 3), dtype=np.uint8)
            offset = (canvas_width - crop.shape[1]) // 2
            canvas[:, offset:offset + crop.shape[1]] = crop
            crop = canvas
        header = np.zeros((128, canvas_width, 3), dtype=np.uint8)
        line1 = (
            f"{row['candidate_id']}  dist={float(row['distance']):.1f}  "
            f"dir={float(row['direction_cosine']):.3f}  score={float(row['recovery_score']):.3f}"
        )
        line2 = (
            f"mean={float(row['mean_probability']):.3f}  q25={float(row['q25_probability']):.3f}  "
            f"contrast={float(row['background_contrast']):.3f}  ratio={float(row['path_ratio']):.2f}"
        )
        line3 = (
            f"component {row['start_component']} -> {row['target_component']}  "
            f"connectivity_gain={int(row['connectivity_gain'])}"
        )
        cv2.putText(header, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, line2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, line3, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            header, "yellow=existing  cyan=weak  orange=connector  green=endpoint  red=projection",
            (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 215, 215), 1, cv2.LINE_AA,
        )
        candidate_image = np.vstack([header, crop])
        safe_id = str(row["candidate_id"]).replace(":", "_")
        image_path = output_dir / f"{safe_id}.png"
        _save_png(image_path, candidate_image)
        image_paths.append(image_path)
    _write_candidate_montage(
        output_dir / "endpoint_segment_candidates_montage.png", image_paths
    )


def write_scene_probability_diagnostic(
    run_dir: Path,
    image_rgb: np.ndarray,
    road_probability: np.ndarray,
    config,
) -> None:
    high_threshold, low_threshold, _profile = graph_extraction.resolve_road_thresholds(config)
    probability_u8 = np.clip(np.rint(road_probability * 255.0), 0, 255).astype(np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(probability_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    high_mask = np.repeat(
        ((road_probability >= high_threshold).astype(np.uint8) * 255)[:, :, np.newaxis], 3, axis=2
    )
    low_mask = np.repeat(
        ((road_probability >= low_threshold).astype(np.uint8) * 255)[:, :, np.newaxis], 3, axis=2
    )
    panels = [
        _labeled_preview(image_rgb, "Original image", max_width=800),
        _labeled_preview(heatmap, "Road probability heatmap", max_width=800),
        _labeled_preview(high_mask, f"HIGH mask >= {high_threshold:g}", max_width=800),
        _labeled_preview(low_mask, f"LOW mask >= {low_threshold:g}", max_width=800),
    ]
    target_height = min(panel.shape[0] for panel in panels)
    panels = [
        cv2.resize(panel, (int(round(panel.shape[1] * target_height / panel.shape[0])), target_height))
        if panel.shape[0] != target_height else panel
        for panel in panels
    ]
    target_width = min(panel.shape[1] for panel in panels)
    panels = [
        cv2.resize(panel, (target_width, target_height)) if panel.shape[1] != target_width else panel
        for panel in panels
    ]
    diagnostic = np.concatenate([
        np.concatenate(panels[:2], axis=1),
        np.concatenate(panels[2:], axis=1),
    ], axis=0)
    _save_png(run_dir / "scene_probability_diagnostic.png", diagnostic)


def write_relative_roadness_diagnostic(
    run_dir: Path,
    image_rgb: np.ndarray,
    road_probability: np.ndarray,
    config,
    scene_state: str,
    recovered_nodes: np.ndarray,
    recovered_edges: np.ndarray,
    metadata: list[dict],
    candidate_audit: list[dict],
    recovery_summary: dict,
) -> None:
    context = graph_extraction.compute_relative_roadness(
        road_probability, config, scene_state=scene_state
    )
    probability_u8 = np.clip(np.rint(road_probability * 255.0), 0, 255).astype(np.uint8)
    relative_u8 = np.clip(np.rint(context["relative_score"] * 255.0), 0, 255).astype(np.uint8)
    relative_candidate = context["relative_candidate_mask"].astype(np.uint8) * 255
    high, _low, _profile = graph_extraction.resolve_road_thresholds(config)
    combined = (
        (road_probability >= high) | (context["relative_candidate_mask"] > 0)
    ).astype(np.uint8) * 255
    _save_png(run_dir / "centerline_probability.png", probability_u8)
    _save_png(run_dir / "relative_roadness.png", relative_u8)
    _save_png(run_dir / "relative_candidate.png", relative_candidate)
    _save_png(run_dir / "combined_candidate.png", combined)
    _save_png(
        run_dir / "relative_skeleton_raw.png",
        context["relative_skeleton_raw"].astype(np.uint8) * 255,
    )
    _save_png(
        run_dir / "relative_binary_skeleton.png",
        context["relative_binary_skeleton"].astype(np.uint8) * 255,
    )
    _save_png(
        run_dir / "relative_ridge_centerline.png",
        context["relative_ridge_mask"].astype(np.uint8) * 255,
    )
    _save_png(
        run_dir / "relative_backbone_mask.png",
        context["relative_backbone_mask"].astype(np.uint8) * 255,
    )
    _save_png(
        run_dir / "relative_skeleton_normalized.png",
        context["relative_skeleton_normalized"].astype(np.uint8) * 255,
    )
    _save_png(
        run_dir / "relative_junction_zones.png",
        context["junction_zone_mask"].astype(np.uint8) * 255,
    )

    probability_rgb = cv2.cvtColor(
        cv2.applyColorMap(probability_u8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB
    )
    relative_rgb = cv2.cvtColor(
        cv2.applyColorMap(relative_u8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB
    )
    relative_rgb[relative_candidate > 0] = (255, 165, 0)
    final_overlay = image_rgb.copy()
    for source, color, thickness in (
        ("samroad", (255, 220, 0), 3),
        ("relative_roadness", (0, 255, 0), 4),
        ("relative_ridge_seed", (0, 255, 0), 4),
        ("relative_backbone_bridge", (0, 220, 255), 4),
        ("relative_backbone_extension", (255, 165, 0), 4),
        ("relative_backbone_branch", (255, 64, 255), 4),
        ("weak_recovered", (0, 255, 255), 5),
        ("weak_bootstrap", (255, 0, 255), 5),
        ("relative_bootstrap", (255, 64, 255), 5),
        ("weak_segment_connector", (255, 128, 0), 6),
    ):
        edge_ids = [
            edge_id for edge_id, row in enumerate(metadata)
            if row.get("line_source") == source
        ]
        if edge_ids:
            final_overlay = _draw_edges(
                final_overlay,
                recovered_nodes,
                recovered_edges[np.asarray(edge_ids, dtype=np.int32)],
                color,
                thickness,
            )
    chain_overlay = image_rgb.copy()
    chain_overlay[context["relative_skeleton"] > 0] = (0, 255, 0)
    acceptance_overlay = image_rgb.copy()
    rejected_structure = np.asarray(context.get("relative_rejected_skeleton", []))
    if rejected_structure.shape == image_rgb.shape[:2]:
        acceptance_overlay[rejected_structure > 0] = (255, 0, 0)
    decision_colors = {
        "auto": (0, 255, 0),
        "review": (255, 165, 0),
        "rejected": (255, 0, 0),
    }
    for row in candidate_audit:
        if row.get("candidate_source") not in {"relative", "absolute+relative"}:
            continue
        path = np.asarray(row.get("path", []), dtype=np.int32).reshape(-1, 2)
        if len(path) < 2:
            continue
        cv2.polylines(
            acceptance_overlay,
            [path[:, ::-1].reshape(-1, 1, 2)],
            False,
            decision_colors.get(row.get("decision", row.get("qa_state")), (255, 0, 0)),
            3,
            cv2.LINE_AA,
        )
    panels = [
        _labeled_preview(image_rgb, "Original image", max_width=800),
        _labeled_preview(probability_rgb, "Raw centerline probability", max_width=800),
        _labeled_preview(relative_rgb, "Relative roadness + candidates", max_width=800),
        _labeled_preview(chain_overlay, "Relative chain after structure filter", max_width=800),
        _labeled_preview(acceptance_overlay, "Auto green / Review orange / Rejected red", max_width=800),
        _labeled_preview(final_overlay, "Final vector", max_width=800),
    ]
    target_height = min(panel.shape[0] for panel in panels)
    target_width = min(panel.shape[1] for panel in panels)
    panels = [cv2.resize(panel, (target_width, target_height)) for panel in panels]
    _save_png(
        run_dir / "relative_roadness_compare.png",
        np.concatenate([
            np.concatenate(panels[:3], axis=1),
            np.concatenate(panels[3:], axis=1),
        ], axis=0),
    )
    _save_png(run_dir / "relative_acceptance_overlay.png", acceptance_overlay)
    _write_json(
        run_dir / "relative_roadness_summary.json",
        {
            key: value for key, value in context["diagnostics"].items()
            if key not in {"relative_micro_chains", "relative_corridors"}
        },
    )
    height, width = image_rgb.shape[:2]
    requested_roi = (1513, 3079, 2130, 3557)
    if width >= requested_roi[2] and height >= requested_roi[3]:
        x0, y0, x1, y1 = requested_roi
    else:
        x0, y0, x1, y1 = 0, 0, width, height
    crop = np.s_[y0:y1, x0:x1]

    candidate_panel = image_rgb[crop].copy()
    candidate_panel[context["relative_candidate_mask"][crop] > 0] = (
        0.55 * candidate_panel[context["relative_candidate_mask"][crop] > 0]
        + 0.45 * np.asarray([255, 165, 0])
    ).astype(np.uint8)
    raw_panel = image_rgb[crop].copy()
    raw_panel[context["relative_skeleton_raw"][crop] > 0] = (255, 0, 0)
    binary_panel = candidate_panel.copy()
    binary_panel[context["relative_binary_skeleton"][crop] > 0] = (255, 0, 0)
    ridge_panel = image_rgb[crop].copy()
    ridge_panel[context["relative_ridge_mask"][crop] > 0] = (0, 255, 0)
    zone_panel = image_rgb[crop].copy()
    zone_panel[context["junction_zone_mask"][crop] > 0] = (0, 128, 255)
    zone_panel[context["collapsed_zone_mask"][crop] > 0] = (180, 0, 255)
    zone_panel[context["pruned_spur_mask"][crop] > 0] = (255, 0, 0)
    normalized_panel = image_rgb[crop].copy()
    normalized_panel[context["relative_skeleton_normalized"][crop] > 0] = (0, 255, 0)
    junction_panels = [
        _labeled_preview(image_rgb[crop], "Original", max_width=900),
        _labeled_preview(candidate_panel, "Relative candidate", max_width=900),
        _labeled_preview(raw_panel, "Raw skeleton", max_width=900),
        _labeled_preview(zone_panel, "Junction zones / pruned spurs", max_width=900),
        _labeled_preview(normalized_panel, "Normalized skeleton", max_width=900),
    ]
    panel_height = min(panel.shape[0] for panel in junction_panels)
    junction_panels = [
        cv2.resize(
            panel,
            (max(1, int(round(panel.shape[1] * panel_height / panel.shape[0]))), panel_height),
        )
        if panel.shape[0] != panel_height else panel
        for panel in junction_panels
    ]
    _save_png(run_dir / "relative_junction_debug.png", np.concatenate(junction_panels, axis=1))

    palette = (
        (255, 64, 64), (64, 255, 64), (64, 128, 255),
        (255, 192, 64), (192, 64, 255), (64, 255, 224),
    )

    def chain_panel(name):
        panel = image_rgb[crop].copy()
        chains = graph_extraction._trace_skeleton_chains(context[name][crop] > 0)
        for chain_id, path in enumerate(chains):
            cv2.polylines(
                panel,
                [path[:, ::-1].reshape(-1, 1, 2)],
                False,
                palette[chain_id % len(palette)],
                2,
                cv2.LINE_AA,
            )
        lengths = [
            graph_extraction._relative_chain_geometry(path)["path_length"]
            for path in chains
        ]
        return panel, chains, lengths

    binary_chain_panel, binary_crop_chains, binary_crop_lengths = chain_panel(
        "relative_binary_skeleton"
    )
    raw_chain_panel, raw_crop_chains, raw_crop_lengths = chain_panel("relative_skeleton_raw")
    normalized_chain_panel, normalized_crop_chains, normalized_crop_lengths = chain_panel(
        "relative_skeleton_normalized"
    )
    raw_chain_panel = _labeled_preview(raw_chain_panel, "Raw chains", max_width=1200)
    normalized_chain_panel = _labeled_preview(
        normalized_chain_panel, "Normalized chains", max_width=1200
    )
    target_height = min(raw_chain_panel.shape[0], normalized_chain_panel.shape[0])
    raw_chain_panel = cv2.resize(raw_chain_panel, (raw_chain_panel.shape[1], target_height))
    normalized_chain_panel = cv2.resize(
        normalized_chain_panel, (normalized_chain_panel.shape[1], target_height)
    )
    _save_png(
        run_dir / "relative_chain_debug.png",
        np.concatenate([
            _labeled_preview(binary_chain_panel, "Old binary-skeleton chains", max_width=1200),
            raw_chain_panel,
            normalized_chain_panel,
        ], axis=1),
    )

    ridge_score_panel = cv2.cvtColor(
        cv2.applyColorMap(relative_u8[crop], cv2.COLORMAP_VIRIDIS),
        cv2.COLOR_BGR2RGB,
    )
    ridge_debug_panels = [
        _labeled_preview(image_rgb[crop], "1 Original", max_width=900),
        _labeled_preview(ridge_score_panel, "2 Relative score", max_width=900),
        _labeled_preview(binary_panel, "3 Candidate + old skeleton", max_width=900),
        _labeled_preview(ridge_panel, "4 Ridge centerline", max_width=900),
        _labeled_preview(final_overlay[crop], "5 Final vector", max_width=900),
    ]
    ridge_debug_height = min(panel.shape[0] for panel in ridge_debug_panels)
    ridge_debug_panels = [
        cv2.resize(
            panel,
            (max(1, int(round(panel.shape[1] * ridge_debug_height / panel.shape[0]))), ridge_debug_height),
        )
        if panel.shape[0] != ridge_debug_height else panel
        for panel in ridge_debug_panels
    ]
    _save_png(
        run_dir / "relative_ridge_debug.png",
        np.concatenate(ridge_debug_panels, axis=1),
    )

    backbone_binary_panel = image_rgb.copy()
    backbone_binary_panel[context["relative_binary_skeleton"] > 0] = (255, 64, 64)
    traced_panel = image_rgb.copy()
    source_labels = context["relative_backbone_source_labels"]
    for source_code, color in (
        (1, (0, 255, 0)),       # Ridge-supported Binary pixels.
        (2, (0, 220, 255)),     # Ridge-to-Ridge Binary bridge.
        (3, (255, 165, 0)),     # Directional endpoint extension.
        (4, (255, 64, 255)),    # Independent supported branch.
    ):
        traced_panel[source_labels == source_code] = color
    backbone_panels = [
        _labeled_preview(image_rgb, "1 Original", max_width=800),
        _labeled_preview(
            cv2.cvtColor(
                cv2.applyColorMap(relative_u8, cv2.COLORMAP_VIRIDIS),
                cv2.COLOR_BGR2RGB,
            ),
            "2 Relative score",
            max_width=800,
        ),
        _labeled_preview(backbone_binary_panel, "3 Binary Skeleton", max_width=800),
        _labeled_preview(
            traced_panel,
            "4 Ridge green / bridge cyan / extension orange / branch magenta",
            max_width=800,
        ),
        _labeled_preview(final_overlay, "5 Final vector", max_width=800),
    ]
    backbone_height = min(panel.shape[0] for panel in backbone_panels)
    backbone_panels = [
        cv2.resize(
            panel,
            (max(1, int(round(panel.shape[1] * backbone_height / panel.shape[0]))), backbone_height),
        )
        if panel.shape[0] != backbone_height else panel
        for panel in backbone_panels
    ]
    _save_png(
        run_dir / "relative_backbone_debug.png",
        np.concatenate(backbone_panels, axis=1),
    )

    def labeled_structure_panel(label_name, title):
        panel = image_rgb[crop].copy()
        labels = np.asarray(context.get(label_name, []), dtype=np.int32)
        if labels.shape == image_rgb.shape[:2]:
            labels = labels[crop]
            for label_id in np.unique(labels):
                if label_id <= 0:
                    continue
                hue = int((int(label_id) * 47) % 180)
                color = cv2.cvtColor(
                    np.asarray([[[hue, 220, 255]]], dtype=np.uint8),
                    cv2.COLOR_HSV2RGB,
                )[0, 0]
                mask = labels == label_id
                panel[mask] = (
                    0.30 * panel[mask] + 0.70 * color.astype(np.float32)
                ).astype(np.uint8)
        return _labeled_preview(panel, title, max_width=900)

    corridor_panels = [
        _labeled_preview(image_rgb[crop], "1 Original", max_width=900),
        _labeled_preview(raw_panel, "2 Raw skeleton", max_width=900),
        labeled_structure_panel("relative_chain_labels", "3 Micro chains"),
        labeled_structure_panel("relative_corridor_labels", "4 Logical corridors"),
        _labeled_preview(final_overlay[crop], "5 Final vector", max_width=900),
    ]
    corridor_height = min(panel.shape[0] for panel in corridor_panels)
    corridor_panels = [
        cv2.resize(
            panel,
            (max(1, int(round(panel.shape[1] * corridor_height / panel.shape[0]))), corridor_height),
        )
        if panel.shape[0] != corridor_height else panel
        for panel in corridor_panels
    ]
    _save_png(
        run_dir / "relative_corridor_debug.png",
        np.concatenate(corridor_panels, axis=1),
    )

    minimum_length = float(config.get("RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX", 48.0))

    def crop_chain_stats(chains, lengths):
        return {
            "chain_count": int(len(chains)),
            "too_short_count": int(sum(length < minimum_length for length in lengths)),
            "median_chain_length": float(np.median(lengths)) if lengths else 0.0,
            "max_chain_length": float(max(lengths)) if lengths else 0.0,
        }

    _write_json(
        run_dir / "relative_skeleton_normalization.json",
        {
            **{
                key: value for key, value in context["diagnostics"].items()
                if key not in {"relative_micro_chains", "relative_corridors"}
            },
            "problem_crop_xyxy": [x0, y0, x1, y1],
            "problem_crop_old_binary": crop_chain_stats(
                binary_crop_chains, binary_crop_lengths
            ),
            "problem_crop_raw": crop_chain_stats(raw_crop_chains, raw_crop_lengths),
            "problem_crop_normalized": crop_chain_stats(
                normalized_crop_chains, normalized_crop_lengths
            ),
        },
    )
    _write_json(
        run_dir / "relative_acceptance_funnel.json",
        recovery_summary.get("relative_acceptance_funnel", {}),
    )
    ridge_audit_keys = (
        "candidate_pixel_count", "old_binary_skeleton_length",
        "ridge_skeleton_length", "old_junction_pixel_count",
        "ridge_junction_pixel_count", "old_micro_chain_count",
        "ridge_micro_chain_count", "old_too_short_count",
        "ridge_too_short_count", "corridor_count",
    )
    _write_json(
        run_dir / "relative_ridge_audit.json",
        {
            **{
                key: context["diagnostics"].get(key, 0)
                for key in ridge_audit_keys
            },
            "final_relative_length": float(
                recovery_summary.get("relative_final_length_px", 0.0)
            ),
            "final_total_graph_length": float(
                recovery_summary.get("final_total_centerline_length_px", 0.0)
            ),
        },
    )
    _write_json(
        run_dir / "relative_backbone_audit.json",
        recovery_summary.get("relative_acceptance_funnel", {}),
    )


def _actual_test_config(
    arguments: argparse.Namespace,
    config,
    image_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    device_name: str,
) -> dict:
    high, low, profile = graph_extraction.resolve_road_thresholds(config)
    reference_high, reference_low, reference_profile = (
        graph_extraction.resolve_scene_diagnostic_thresholds(config)
    )
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    weak_parameters = {
        key: value for key, value in config_dict.items()
        if (
            key.startswith("WEAK_RECOVERY_")
            or key.startswith("WEAK_BOOTSTRAP_")
            or key.startswith("WEAK_SEGMENT_RECOVERY_")
            or key.startswith("WEAK_ENDPOINT_DIRECTION_")
            or key.startswith("RELATIVE_ROADNESS_")
        )
    }
    return {
        "input_image": str(image_path.resolve()),
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": device_name,
        "batch_size": int(config.INFER_BATCH_SIZE),
        "ROAD_THRESHOLD_PROFILE": profile,
        "ROAD_HIGH_THRESHOLD": high,
        "ROAD_LOW_THRESHOLD": low,
        "SCENE_DIAGNOSTIC_REFERENCE_PROFILE": reference_profile,
        "SCENE_DIAGNOSTIC_REFERENCE_HIGH_THRESHOLD": reference_high,
        "SCENE_DIAGNOSTIC_REFERENCE_LOW_THRESHOLD": reference_low,
        **weak_parameters,
    }


def _recovery_payload(
    summary: dict,
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    metadata: list[dict],
    original_edge_count: int,
) -> dict:
    recovered_rows = []
    recovered_sources = {
        "weak_recovered", "weak_bootstrap", "relative_bootstrap",
        "relative_roadness", "weak_segment_connector",
    }
    for edge_id, edge_metadata in enumerate(metadata):
        line_source = str(edge_metadata.get("line_source", ""))
        if line_source not in recovered_sources and not line_source.startswith("relative_"):
            continue
        src_idx, dst_idx = edges[edge_id]
        recovered_rows.append({
            "edge_id": edge_id,
            "src_row": float(nodes_rc[src_idx, 0]), "src_col": float(nodes_rc[src_idx, 1]),
            "dst_row": float(nodes_rc[dst_idx, 0]), "dst_col": float(nodes_rc[dst_idx, 1]),
            **edge_metadata,
        })
    return {"summary": summary, "recovered_edges": recovered_rows}


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    total_started = time.perf_counter()
    cached = {}
    if arguments.recovery_only:
        if not arguments.run_dir:
            raise ValueError("--recovery-only requires --run-dir")
        run_dir = Path(arguments.run_dir).expanduser().resolve()
        cached_path = run_dir / "test_config.json"
        if not cached_path.is_file():
            raise FileNotFoundError(f"Missing recovery cache metadata: {cached_path}")
        cached = json.loads(cached_path.read_text(encoding="utf-8"))
        image_path = Path(arguments.image or cached["input_image"]).expanduser().resolve()
    else:
        if not arguments.image:
            raise ValueError("Full mode requires --image")
        image_path = Path(arguments.image).expanduser().resolve()
        run_dir = (
            Path(arguments.run_dir).expanduser().resolve()
            if arguments.run_dir else (DEFAULT_OUTPUTS / image_path.stem).resolve()
        )
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(arguments.config or cached.get("config") or DEFAULT_CONFIG).expanduser().resolve()
    checkpoint_path = Path(
        arguments.checkpoint or cached.get("checkpoint") or DEFAULT_CHECKPOINT
    ).expanduser().resolve()
    config = single_image_inference.load_config(config_path)
    if arguments.recovery_only and arguments.batch_size is None and cached.get("batch_size"):
        config.INFER_BATCH_SIZE = int(cached["batch_size"])
    _apply_overrides(config, arguments)
    requested_device = arguments.device or cached.get("device") or "auto"
    if arguments.recovery_only:
        device_name, device = str(requested_device), None
    else:
        device_name, device = single_image_inference.resolve_torch_device(requested_device)

    timing = {}
    if arguments.recovery_only:
        _progress("Loading cached original graph and probability...")
        load_started = time.perf_counter()
        image_rgb = single_image_inference.read_rgb_img(image_path)
        road_probability = _read_probability(run_dir / "road_probability.png")
        original_nodes, original_edges, original_scores = load_original_scored_graph(
            run_dir / "original_graph.p",
            run_dir / "original_edge_scores.csv",
        )
        if road_probability.shape != image_rgb.shape[:2]:
            raise ValueError(
                f"Cached probability/image shape mismatch: {road_probability.shape} != {image_rgb.shape[:2]}"
            )
        timing["load_cache_seconds"] = time.perf_counter() - load_started
    else:
        _progress("Reading input image...")
        image_started = time.perf_counter()
        image_rgb = single_image_inference.read_rgb_img(image_path)
        timing["image_read_seconds"] = time.perf_counter() - image_started
        _progress(f"Loading model on {device_name}...")
        model_started = time.perf_counter()
        model = single_image_inference.load_model(config, checkpoint_path, device)
        timing["model_load_seconds"] = time.perf_counter() - model_started
        padded = single_image_inference.pad_image_to_min_size(image_rgb, int(config.PATCH_SIZE))
        inference_timing = {}
        _progress(
            f"Processing patches with batch size {int(config.INFER_BATCH_SIZE)}..."
        )
        (
            original_nodes, original_edges, original_scores,
            intersection_probability, road_probability_u8,
        ) = single_image_inference.infer_one_image(
            model, padded, config, device=device, timing=inference_timing,
            diagnostic_shape=image_rgb.shape[:2],
        )
        height, width = image_rgb.shape[:2]
        road_probability_u8 = road_probability_u8[:height, :width]
        intersection_probability = intersection_probability[:height, :width]
        original_nodes, original_edges, original_scores = single_image_inference.filter_graph_to_image_bounds(
            original_nodes, original_edges, height, width, original_scores
        )
        road_probability = road_probability_u8.astype(np.float32) / 255.0
        timing.update(inference_timing)
        _save_png(run_dir / "road_probability.png", road_probability_u8)
        _save_png(run_dir / "intersection_probability.png", intersection_probability)
        save_graph(run_dir / "original_graph.p", original_nodes, original_edges)
        write_original_scores(
            run_dir / "original_edge_scores.csv", original_nodes, original_edges, original_scores
        )

    if str(config.get("ROAD_THRESHOLD_PROFILE", "default")) == "auto":
        decision = graph_extraction.resolve_effective_road_profile(road_probability, config)
        config.ROAD_THRESHOLD_PROFILE = decision["effective_profile"]

    _progress("Scene diagnosis, weak-network bootstrap, and weak recovery...")
    recovery_started = time.perf_counter()
    weak_candidate_audit = []
    bootstrap_candidate_audit = []
    endpoint_segment_candidate_audit = []
    recovered_nodes, recovered_edges, metadata, recovery_summary = (
        graph_extraction.postprocess_weak_road_network(
            original_nodes,
            original_edges,
            road_probability,
            config,
            edge_scores=original_scores,
            weak_candidate_audit=weak_candidate_audit,
            bootstrap_candidate_audit=bootstrap_candidate_audit,
            endpoint_segment_candidate_audit=endpoint_segment_candidate_audit,
        )
    )
    timing["weak_recovery_seconds"] = time.perf_counter() - recovery_started
    save_graph(run_dir / "recovered_graph.p", recovered_nodes, recovered_edges)
    _write_json(
        run_dir / "weak_recovery.json",
        _recovery_payload(
            recovery_summary, recovered_nodes, recovered_edges, metadata, len(original_edges)
        ),
    )
    _write_candidate_csv(
        run_dir / "weak_recovery_candidates.csv",
        WEAK_CANDIDATE_FIELDS,
        weak_candidate_audit,
    )
    _write_candidate_csv(
        run_dir / "bootstrap_candidates.csv",
        BOOTSTRAP_CANDIDATE_FIELDS,
        bootstrap_candidate_audit,
    )
    _write_candidate_csv(
        run_dir / "relative_review_candidates.csv",
        BOOTSTRAP_CANDIDATE_FIELDS,
        [
            row for row in bootstrap_candidate_audit
            if row.get("decision") == "review"
            and row.get("candidate_source") in {"relative", "absolute+relative"}
        ],
    )
    _write_candidate_csv(
        run_dir / "endpoint_segment_candidates.csv",
        ENDPOINT_SEGMENT_CANDIDATE_FIELDS,
        endpoint_segment_candidate_audit,
    )

    _progress("Generating overlays and comparison preview...")
    visualization_started = time.perf_counter()
    write_visualizations(
        run_dir, image_rgb, original_nodes, original_edges, recovered_nodes, recovered_edges,
        metadata,
    )
    write_scene_probability_diagnostic(run_dir, image_rgb, road_probability, config)
    write_relative_roadness_diagnostic(
        run_dir,
        image_rgb,
        road_probability,
        config,
        recovery_summary.get("scene_confidence_state", "normal"),
        recovered_nodes,
        recovered_edges,
        metadata,
        bootstrap_candidate_audit,
        recovery_summary,
    )
    write_endpoint_segment_candidate_visualizations(
        run_dir,
        image_rgb,
        recovered_nodes,
        recovered_edges,
        metadata,
        endpoint_segment_candidate_audit,
    )
    timing["visualization_seconds"] = time.perf_counter() - visualization_started
    timing["total_seconds"] = time.perf_counter() - total_started
    _write_json(run_dir / "timing.json", timing)
    _write_json(
        run_dir / "test_config.json",
        _actual_test_config(
            arguments, config, image_path, config_path, checkpoint_path, device_name
        ),
    )
    _progress(
        "Recovery summary: "
        f"strong_edge_count={recovery_summary.get('strong_edge_count', 0)}, "
        f"weak_candidate_count={recovery_summary.get('weak_candidate_count', 0)}, "
        f"weak_recovered_edge_count={recovery_summary.get('weak_recovered_edge_count', 0)}, "
        f"endpoint_segment_accepted_count={recovery_summary.get('endpoint_segment_accepted_count', 0)}, "
        f"connectivity_gain_total={recovery_summary.get('connectivity_gain_total', 0)}, "
        f"bootstrap_recovered_edge_count={recovery_summary.get('bootstrap_recovered_edge_count', 0)}, "
        f"relative_recovered_edge_count={recovery_summary.get('relative_recovered_edge_count', 0)}, "
        f"scene_confidence_state={recovery_summary.get('scene_confidence_state', 'unknown')}"
    )
    _progress(f"Done. Total seconds: {timing['total_seconds']:.3f}")
    print(json.dumps({"run_dir": str(run_dir), **recovery_summary, "timing": timing}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
