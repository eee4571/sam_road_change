import json
from pathlib import Path

import cv2
import numpy as np

from engine.fast_pipeline import (
    _cleanup_fast_final_centerline,
    _enhance_fast_molra_surface,
    _rasterize_fast_topology,
    _unique_topology_edges,
    regularize_fast_road_network,
)


ROOT = Path(
    "project/test_area/_work/tasks/runs/run_20260828_142123/grids/验证区1/"
    "periods/20221020/runs/roads_rerun_1788415454"
).resolve()
OUTPUT = Path(
    "C:/Users/zhoum/.codex/visualizations/2026/09/03/"
    "01a065cb-9117-7ed1-9024-8f7ccdea7396"
)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def render_paths(paths, surface: np.ndarray, title: str) -> np.ndarray:
    canvas = np.full((*surface.shape, 3), 245, dtype=np.uint8)
    canvas[surface > 0] = (220, 220, 220)
    palette = [
        (36, 99, 235), (225, 61, 61), (47, 158, 68), (218, 138, 18),
        (156, 75, 201), (12, 160, 169), (228, 80, 145), (98, 113, 128),
    ]
    for path_id, path in enumerate(paths):
        points = np.rint(path.pixels[:, ::-1]).astype(np.int32)
        cv2.polylines(canvas, [points], False, palette[path_id % len(palette)], 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[0]), 2, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[-1]), 2, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (510, 34), (255, 255, 255), -1)
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


totals = {
    "original": 0, "final": 0, "length_before_m": 0.0, "length_after_m": 0.0,
    "collapsed": 0, "junctions": 0, "attachments": 0, "through_merges": 0,
    "connections": 0,
    "vertices_before": 0, "vertices_after": 0, "straight": 0, "curved": 0,
}
rows = []
for summary_path in sorted((ROOT / "width_review").glob("v*_summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stem = summary["stem"]
    pixel_size = float(summary["pixel_size"])
    surface = (read_gray(ROOT / "surfaces/masks/grid_tiles" / f"{stem}_mask.png") > 0).astype(np.uint8)
    topology_path = ROOT / "inference/road_graphs/grid_tiles/graph" / f"{stem}_fast_topology.npz"
    with np.load(topology_path, allow_pickle=False) as topology:
        nodes = np.asarray(topology["nodes"], dtype=np.float32).reshape(-1, 2)
        edges = _unique_topology_edges(nodes, topology["edges"])
    topology_centerline, _ = _rasterize_fast_topology(surface.shape, nodes, edges)
    probability = np.asarray(np.load(
        ROOT / "width_review/molra_probability_cache" / f"{stem}_probability.npy",
        allow_pickle=False,
    ), dtype=np.float32)
    molra_surface, _ = _enhance_fast_molra_surface(
        probability, topology_centerline, physical_pixel_size_m=pixel_size,
    )
    _cleaned, paths, _ = _cleanup_fast_final_centerline(
        topology_centerline, molra_surface, pixel_size, support_score=probability,
    )
    evidence = surface | molra_surface
    _network, final_paths, diagnostics = regularize_fast_road_network(
        paths, evidence, pixel_size,
    )
    before_length = sum(path.length_px for path in paths) * pixel_size
    after_length = sum(path.length_px for path in final_paths) * pixel_size
    row = {
        "tile": stem,
        "original": len(paths),
        "final": len(final_paths),
        "reduction_pct": round(100.0 * (len(paths) - len(final_paths)) / max(len(paths), 1), 2),
        "length_retained_pct": round(100.0 * after_length / max(before_length, 1e-9), 2),
        "collapsed": diagnostics["regularization_collapsed_junction_edge_count"],
        "junctions": diagnostics["regularization_consolidated_junction_count"],
        "attachments": diagnostics["regularization_path_attachment_count"],
        "through_merges": diagnostics["regularization_intersection_through_merge_count"],
        "connections": diagnostics["regularization_generated_connection_count"],
        "vertices_before": diagnostics["original_vertex_count"],
        "vertices_after": diagnostics["final_vertex_count"],
        "straight": diagnostics["canonical_straight_road_count"],
        "curved": diagnostics["canonical_curved_road_count"],
    }
    rows.append(row)
    for key in (
        "original", "final", "collapsed", "junctions", "attachments",
        "through_merges", "connections", "vertices_before", "vertices_after",
        "straight", "curved",
    ):
        totals[key] += row[key]
    totals["length_before_m"] += before_length
    totals["length_after_m"] += after_length

    if stem in {"v0002", "v0005"}:
        before = render_paths(paths, evidence, f"{stem} before: {len(paths)} features")
        after = render_paths(final_paths, evidence, f"{stem} canonical: {len(final_paths)} features")
        combined = np.hstack((before, after))
        scale = min(1.0, 1800.0 / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUTPUT / f"{stem}_canonical_feature_comparison.png"), combined)

totals["reduction_pct"] = round(
    100.0 * (totals["original"] - totals["final"]) / totals["original"], 2,
)
totals["length_retained_pct"] = round(
    100.0 * totals["length_after_m"] / totals["length_before_m"], 2,
)
print(json.dumps({"tiles": rows, "totals": totals}, ensure_ascii=False, indent=2))
