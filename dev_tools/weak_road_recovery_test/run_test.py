from __future__ import annotations

import argparse
import csv
import json
import pickle
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
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _apply_overrides(config, arguments: argparse.Namespace) -> None:
    threshold_overridden = (
        arguments.road_high_threshold is not None or arguments.road_low_threshold is not None
    )
    if threshold_overridden:
        high, low, _profile = graph_extraction.resolve_road_thresholds(config)
        high = arguments.road_high_threshold if arguments.road_high_threshold is not None else high
        low = arguments.road_low_threshold if arguments.road_low_threshold is not None else low
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
) -> None:
    original_overlay = _draw_edges(
        image_rgb.copy(), original_nodes, original_edges, (255, 220, 0), 3
    )
    recovered_overlay = _draw_edges(
        image_rgb.copy(), recovered_nodes, recovered_edges[:len(original_edges)], (255, 220, 0), 3
    )
    recovered_overlay = _draw_edges(
        recovered_overlay, recovered_nodes, recovered_edges[len(original_edges):], (0, 255, 255), 5
    )
    _save_png(run_dir / "original_overlay.png", original_overlay)
    _save_png(run_dir / "recovered_overlay.png", recovered_overlay)
    left = _labeled_preview(original_overlay, "Original SAMRoad (yellow)")
    right = _labeled_preview(recovered_overlay, "Weak recovered additions (cyan)")
    target_height = min(left.shape[0], right.shape[0])
    if left.shape[0] != target_height:
        left = cv2.resize(left, (int(left.shape[1] * target_height / left.shape[0]), target_height))
    if right.shape[0] != target_height:
        right = cv2.resize(right, (int(right.shape[1] * target_height / right.shape[0]), target_height))
    _save_png(run_dir / "recovery_compare.png", np.concatenate([left, right], axis=1))


def _actual_test_config(
    arguments: argparse.Namespace,
    config,
    image_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    device_name: str,
) -> dict:
    high, low, profile = graph_extraction.resolve_road_thresholds(config)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    weak_parameters = {
        key: value for key, value in config_dict.items()
        if key.startswith("WEAK_RECOVERY_")
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
    for edge_id in range(original_edge_count, len(edges)):
        src_idx, dst_idx = edges[edge_id]
        recovered_rows.append({
            "edge_id": edge_id,
            "src_row": float(nodes_rc[src_idx, 0]), "src_col": float(nodes_rc[src_idx, 1]),
            "dst_row": float(nodes_rc[dst_idx, 0]), "dst_col": float(nodes_rc[dst_idx, 1]),
            **metadata[edge_id],
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
        load_started = time.perf_counter()
        image_rgb = single_image_inference.read_rgb_img(image_path)
        road_probability = _read_probability(run_dir / "road_probability.png")
        original_nodes, original_edges = load_graph(run_dir / "original_graph.p")
        original_scores = read_original_scores(
            run_dir / "original_edge_scores.csv", original_nodes, original_edges
        )
        if road_probability.shape != image_rgb.shape[:2]:
            raise ValueError(
                f"Cached probability/image shape mismatch: {road_probability.shape} != {image_rgb.shape[:2]}"
            )
        timing["load_cache_seconds"] = time.perf_counter() - load_started
    else:
        image_started = time.perf_counter()
        image_rgb = single_image_inference.read_rgb_img(image_path)
        timing["image_read_seconds"] = time.perf_counter() - image_started
        model_started = time.perf_counter()
        model = single_image_inference.load_model(config, checkpoint_path, device)
        timing["model_load_seconds"] = time.perf_counter() - model_started
        padded = single_image_inference.pad_image_to_min_size(image_rgb, int(config.PATCH_SIZE))
        inference_timing = {}
        (
            original_nodes, original_edges, original_scores,
            intersection_probability, road_probability_u8,
        ) = single_image_inference.infer_one_image(
            model, padded, config, device=device, timing=inference_timing
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

    recovery_started = time.perf_counter()
    recovered_nodes, recovered_edges, metadata, recovery_summary = (
        graph_extraction.recover_weak_road_edges(
            original_nodes,
            original_edges,
            road_probability,
            config,
            edge_scores=original_scores,
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

    visualization_started = time.perf_counter()
    write_visualizations(
        run_dir, image_rgb, original_nodes, original_edges, recovered_nodes, recovered_edges
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
    print(json.dumps({"run_dir": str(run_dir), **recovery_summary, "timing": timing}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
