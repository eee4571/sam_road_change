from __future__ import annotations

import json
import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from molra_centerline_width import maybe_snap_to_mask, scan_one_side


@dataclass
class RoadChain:
    chain_id: int
    node_ids: list[int]
    edge_ids: list[int]


def build_road_chains(nodes: np.ndarray, edges: np.ndarray) -> list[RoadChain]:
    """Merge degree-2 micro-edges into endpoint/junction-to-endpoint/junction chains."""
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(nodes))]
    for edge_id, (src, dst) in enumerate(edges.tolist()):
        adjacency[int(src)].append((int(dst), edge_id))
        adjacency[int(dst)].append((int(src), edge_id))
    degrees = np.asarray([len(items) for items in adjacency], dtype=np.int32)
    visited: set[int] = set()
    chains: list[RoadChain] = []

    def trace(start: int, first_edge: int) -> RoadChain:
        node_ids = [start]
        edge_ids: list[int] = []
        current = start
        edge_id = first_edge
        while edge_id not in visited:
            visited.add(edge_id)
            edge_ids.append(edge_id)
            src, dst = (int(value) for value in edges[edge_id])
            nxt = dst if src == current else src
            node_ids.append(nxt)
            if degrees[nxt] != 2:
                break
            candidates = [(other, eid) for other, eid in adjacency[nxt] if eid not in visited]
            if not candidates:
                break
            current, edge_id = nxt, candidates[0][1]
        return RoadChain(len(chains), node_ids, edge_ids)

    for node_id in np.where(degrees != 2)[0].tolist():
        for _, edge_id in adjacency[node_id]:
            if edge_id not in visited:
                chains.append(trace(node_id, edge_id))
    for edge_id in range(len(edges)):
        if edge_id not in visited:
            chains.append(trace(int(edges[edge_id, 0]), edge_id))
    return chains


def _chain_geometry(chain: RoadChain, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = nodes[np.asarray(chain.node_ids, dtype=np.int32)].astype(np.float32)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return points, np.concatenate(([0.0], np.cumsum(lengths)))


def _point_at(points: np.ndarray, cumulative: np.ndarray, distance: float) -> tuple[np.ndarray, int]:
    edge_offset = min(max(0, int(np.searchsorted(cumulative, distance, side="right") - 1)), len(points) - 2)
    span = max(float(cumulative[edge_offset + 1] - cumulative[edge_offset]), 1e-6)
    ratio = float(np.clip((distance - cumulative[edge_offset]) / span, 0.0, 1.0))
    return points[edge_offset] * (1.0 - ratio) + points[edge_offset + 1] * ratio, edge_offset


def _tangent_at(points: np.ndarray, cumulative: np.ndarray, distance: float, radius: float) -> np.ndarray:
    before, _ = _point_at(points, cumulative, max(0.0, distance - radius))
    after, _ = _point_at(points, cumulative, min(float(cumulative[-1]), distance + radius))
    vector = after - before
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    return vector / norm


def _endpoint_direction(
    chain: RoadChain,
    nodes: np.ndarray,
    node_id: int,
    lookahead: float,
) -> np.ndarray:
    points, cumulative = _chain_geometry(chain, nodes)
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    distance = min(max(float(lookahead), 1.0), total)
    if node_id == chain.node_ids[0]:
        endpoint = points[0]
        interior, _ = _point_at(points, cumulative, distance)
    else:
        endpoint = points[-1]
        interior, _ = _point_at(points, cumulative, total - distance)
    direction = interior - endpoint
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-6 else np.asarray([0.0, 1.0], dtype=np.float32)


def _hampel(values: np.ndarray, valid: np.ndarray, window: int = 3, scale: float = 3.5) -> np.ndarray:
    kept = valid.copy()
    for index in np.where(valid)[0]:
        lo, hi = max(0, index - window), min(len(values), index + window + 1)
        local = values[lo:hi][valid[lo:hi]]
        if local.size < 3:
            continue
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median)))
        threshold = max(1.0, scale * 1.4826 * mad)
        if abs(float(values[index]) - median) > threshold:
            kept[index] = False
    return kept


def _pchip_profile(positions: np.ndarray, values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Shape-preserving interpolation with constant endpoint extrapolation."""
    ids = np.where(anchors)[0]
    if not ids.size:
        return np.zeros(len(values), dtype=np.float32)
    if ids.size == 1:
        return np.full(len(values), float(values[ids[0]]), dtype=np.float32)
    interpolator = PchipInterpolator(positions[ids], values[ids], extrapolate=False)
    result = np.asarray(interpolator(positions), dtype=np.float32)
    result[positions < positions[ids[0]]] = float(values[ids[0]])
    result[positions > positions[ids[-1]]] = float(values[ids[-1]])
    return np.maximum(result, 0.0)


def _fit_side_profile(
    positions: np.ndarray,
    values: np.ndarray,
    grades: np.ndarray,
    mad_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one road side: A is exact, B has 0.35 influence, C is prediction-only."""
    candidates = (grades != "C") & (values > 0)
    kept = _hampel(values, candidates, scale=mad_scale)
    strong = kept & (grades == "A")
    weak = kept & (grades == "B")
    all_fit = _pchip_profile(positions, values, strong | weak)
    if np.any(strong):
        strong_fit = _pchip_profile(positions, values, strong)
        fitted = 0.75 * strong_fit + 0.25 * all_fit
        fitted[weak] = 0.65 * strong_fit[weak] + 0.35 * values[weak]
        fitted[strong] = values[strong]
    else:
        fitted = all_fit
        fitted[weak] = values[weak]
    return np.maximum(fitted, 0.0), kept, candidates & ~kept


def _optimal_segments(values: np.ndarray, min_size: int = 3) -> list[tuple[int, int]]:
    """Exact penalized change-point segmentation (PELT objective, O(n^2) solver)."""
    count = len(values)
    if count == 0:
        return []
    if count < min_size * 2:
        return [(0, count)]
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values * values)))
    scale = max(float(np.median(values)) * 0.15, 1.5)
    penalty = scale * scale * math.log(count + 1.0)

    def cost(start: int, end: int) -> float:
        size = end - start
        total = prefix[end] - prefix[start]
        return float(prefix_sq[end] - prefix_sq[start] - total * total / max(size, 1))

    dp = np.full(count + 1, np.inf, dtype=np.float64)
    previous = np.full(count + 1, -1, dtype=np.int32)
    dp[0] = -penalty
    for end in range(min_size, count + 1):
        starts = [0] + list(range(min_size, end - min_size + 1))
        for start in starts:
            candidate = dp[start] + cost(start, end) + penalty
            if candidate < dp[end]:
                dp[end] = candidate
                previous[end] = start
    if previous[count] < 0:
        return [(0, count)]
    segments: list[tuple[int, int]] = []
    end = count
    while end > 0:
        start = int(previous[end])
        segments.append((start, end))
        end = start
    return list(reversed(segments))


def calculate_chain_hybrid_widths(request):
    from final_width_calculator import FinalWidthResult

    nodes = np.asarray(request.nodes_rc, dtype=np.float32)
    edges = np.asarray(request.edges, dtype=np.int32)
    surface = None if request.road_surface is None else np.asarray(request.road_surface, dtype=np.uint8)
    if surface is None or len(edges) == 0:
        return FinalWidthResult(
            edge_widths=[{"edge_id": i, "width_px": 0.0, "width_units": 0.0, "source": "chain_hybrid_v2", "status": "width_unresolved"} for i in range(len(edges))],
            samples=[], segments=[], algorithm="chain_hybrid_v2",
            metadata={"measured_edge_count": 0, "unresolved_edge_count": len(edges)},
        )

    cfg = request.config
    # Boundary reconstruction requires dense, nearly uniform samples on every
    # junction-to-junction chain. Keep legacy configurations inside the 4-6 px contract.
    sample_step = float(np.clip(float(getattr(cfg, "sample_step_px", 5.0)), 4.0, 6.0))
    tangent_radius = max(sample_step, 4.0)
    agreement_limit = float(getattr(cfg, "hybrid_agreement_ratio", 0.35))
    max_asymmetry = float(getattr(cfg, "max_asymmetry_ratio", 0.65))
    max_search = float(getattr(cfg, "max_search_px", 70.0))
    snap_radius = int(getattr(cfg, "snap_radius_px", 8))
    max_snap = float(getattr(cfg, "max_snap_distance_px", 4.0))
    junction_buffer = float(getattr(cfg, "junction_buffer_px", 20.0))
    pixel_size = float(getattr(cfg, "pixel_size", 1.0))
    distance = cv2.distanceTransform((surface > 0).astype(np.uint8), cv2.DIST_L2, 5)
    degrees = np.bincount(edges.ravel(), minlength=len(nodes))
    chains = build_road_chains(nodes, edges)
    samples: list[dict] = []
    segments: list[dict] = []
    edge_values: dict[int, list[float]] = {edge_id: [] for edge_id in range(len(edges))}
    edge_chain_fallback: dict[int, float] = {}
    edge_chain_id: dict[int, int] = {}
    chain_widths: dict[int, tuple[float, str, str]] = {}
    chain_end_nodes: dict[int, tuple[int, int]] = {}
    chain_geometry: dict[int, tuple[np.ndarray, float]] = {}

    for chain in chains:
        points, cumulative = _chain_geometry(chain, nodes)
        total = float(cumulative[-1])
        if total < 1.0:
            continue
        chain_end_nodes[chain.chain_id] = (chain.node_ids[0], chain.node_ids[-1])
        chain_geometry[chain.chain_id] = (points, total)
        for edge_id in chain.edge_ids:
            edge_chain_id[edge_id] = chain.chain_id
        count = max(1, int(math.ceil(total / sample_step)))
        positions = (np.arange(count, dtype=np.float32) + 0.5) * total / count
        records: list[dict] = []
        left_raw = np.zeros(count, dtype=np.float32)
        right_raw = np.zeros(count, dtype=np.float32)
        sample_grades = np.full(count, "C", dtype="<U1")
        for local_id, position in enumerate(positions.tolist()):
            point, offset = _point_at(points, cumulative, position)
            tangent = _tangent_at(points, cumulative, position, tangent_radius)
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
            row, col, snapped, snap_distance = maybe_snap_to_mask(float(point[0]), float(point[1]), surface, snap_radius)
            rr, cc = int(round(row)), int(round(col))
            inside = bool(0 <= rr < surface.shape[0] and 0 <= cc < surface.shape[1] and surface[rr, cc] > 0)
            left = right = 0.0
            left_stop = right_stop = "outside_mask"
            if inside:
                left, left_stop = scan_one_side(surface, row, col, normal, -1.0, max_search, 1.0)
                right, right_stop = scan_one_side(surface, row, col, normal, 1.0, max_search, 1.0)
            normal_width = left + right + 1.0 if inside else 0.0
            edt_width = 2.0 * float(distance[rr, cc]) if inside else 0.0
            asymmetry = abs(left - right) / max(left + right, 1.0)
            agreement = abs(normal_width - edt_width) / max(float(np.median([normal_width, edt_width])), 1.0)
            local_width = max(normal_width, edt_width, 1.0)
            junction_extent = float(np.clip(
                float(getattr(cfg, "junction_width_factor", 1.75)) * local_width,
                sample_step,
                max(junction_buffer * 2.0, sample_step),
            ))
            near_junction = bool(
                (degrees[chain.node_ids[0]] >= 3 and position < junction_extent)
                or (degrees[chain.node_ids[-1]] >= 3 and total - position < junction_extent)
            )
            boundary_valid = bool(
                inside and not near_junction and snap_distance <= max_snap
                and left_stop == "mask_boundary" and right_stop == "mask_boundary"
            )
            if boundary_valid and agreement_limit >= 0:
                grade = "A" if agreement <= agreement_limit else "B"
            else:
                grade = "C"
            # Split the center pixel evenly so dL + dR remains the measured full width.
            left_distance = left + 0.5 if inside else 0.0
            right_distance = right + 0.5 if inside else 0.0
            left_raw[local_id] = left_distance
            right_raw[local_id] = right_distance
            sample_grades[local_id] = grade
            flags = []
            if not inside: flags.append("outside_surface")
            if near_junction: flags.append("junction_excluded")
            if asymmetry > max_asymmetry: flags.append("asymmetric")
            if agreement > agreement_limit: flags.append("normal_edt_disagreement")
            if snap_distance > max_snap: flags.append("large_snap")
            edge_id = chain.edge_ids[min(offset, len(chain.edge_ids) - 1)]
            records.append({
                "sample_id": len(samples) + local_id, "chain_id": chain.chain_id, "edge_id": edge_id,
                "chain_distance_px": position, "row": float(point[0]), "col": float(point[1]),
                "row_used": float(row), "col_used": float(col), "normal_width_px": normal_width,
                "normal_row": float(normal[0]), "normal_col": float(normal[1]),
                "edt_width_px": edt_width, "hybrid_raw_width_px": normal_width if boundary_valid else 0.0,
                "left_raw_px": left_distance, "right_raw_px": right_distance,
                "asymmetry_ratio": asymmetry, "agreement_ratio": agreement,
                "valid_width": boundary_valid, "quality_grade": grade, "interpolated": False,
                "quality_flags": ";".join(flags), "width_px": 0.0, "width_units": 0.0,
            })

        mad_scale = float(getattr(cfg, "outlier_mad_scale", 3.5))
        left_profile, left_kept, left_outlier = _fit_side_profile(
            positions, left_raw, sample_grades, mad_scale
        )
        right_profile, right_kept, right_outlier = _fit_side_profile(
            positions, right_raw, sample_grades, mad_scale
        )
        kept = left_kept & right_kept
        outlier = left_outlier | right_outlier
        usable = (left_profile > 0) & (right_profile > 0)
        for index, record in enumerate(records):
            if outlier[index]:
                record["quality_flags"] = ";".join(filter(None, [record["quality_flags"], "side_hampel_outlier"]))
                record["quality_grade"] = "C"
            elif sample_grades[index] == "B":
                record["quality_flags"] = ";".join(filter(None, [record["quality_flags"], "weak_boundary_constraint"]))
            record["interpolated"] = bool(not kept[index])
            if record["interpolated"]:
                record["quality_flags"] = ";".join(filter(None, [record["quality_flags"], "continuity_only_interpolated"]))
                record["quality_grade"] = "C"
            record["left_px"] = float(left_profile[index]) if usable[index] else 0.0
            record["right_px"] = float(right_profile[index]) if usable[index] else 0.0
            record["width_px"] = record["left_px"] + record["right_px"]
            record["width_units"] = record["width_px"] * pixel_size
            record["left_boundary_row"] = float(record["row_used"] - record["normal_row"] * record["left_px"])
            record["left_boundary_col"] = float(record["col_used"] - record["normal_col"] * record["left_px"])
            record["right_boundary_row"] = float(record["row_used"] + record["normal_row"] * record["right_px"])
            record["right_boundary_col"] = float(record["col_used"] + record["normal_col"] * record["right_px"])
            if record["width_px"] > 0 and record["quality_grade"] in {"A", "B"}:
                edge_values[record["edge_id"]].append(record["width_px"])
        samples.extend(records)

        usable_ids = np.where(usable & (sample_grades != "C") & ~outlier)[0]
        if usable_ids.size:
            smoothed = left_profile + right_profile
            chain_median = float(np.median(smoothed[usable_ids]))
            chain_widths[chain.chain_id] = (chain_median, "B", "chain_hybrid_v2_chain_fallback")
            for edge_id in chain.edge_ids:
                edge_chain_fallback[edge_id] = chain_median
            profile = smoothed[usable_ids]
            for start, end in _optimal_segments(profile, min_size=max(2, int(getattr(cfg, "width_change_min_samples", 3)))):
                ids = usable_ids[start:end]
                if ids.size == 0:
                    continue
                values = smoothed[ids]
                first, last = records[int(ids[0])], records[int(ids[-1])]
                segments.append({
                    "width_segment_id": len(segments), "chain_id": chain.chain_id,
                    "start_sample_index": int(ids[0]), "end_sample_index": int(ids[-1]),
                    "sample_count": int(ids.size), "start_distance_px": first["chain_distance_px"],
                    "end_distance_px": last["chain_distance_px"], "start_row": first["row_used"],
                    "start_col": first["col_used"], "end_row": last["row_used"], "end_col": last["col_used"],
                    "median_width_px": float(np.median(values)), "mean_width_px": float(np.mean(values)),
                    "median_width_units": float(np.median(values) * pixel_size),
                    "width_cv": float(np.std(values) / max(float(np.mean(values)), 1.0)),
                    "edge_ids_json": json.dumps(chain.edge_ids, separators=(",", ":")),
                    "quality_grade": "B" if any(records[int(item)]["interpolated"] for item in ids) else "A",
                    "width_source": "chain_hybrid_v2",
                })
        else:
            # A correct centerline can still fail the strict normal/EDT agreement tests.
            # Use robust centerline EDT away from junctions before falling back to neighbours.
            edt_candidates = np.asarray([
                float(record["edt_width_px"])
                for record in records
                if float(record["edt_width_px"]) > 0
                and "junction_excluded" not in str(record["quality_flags"])
                and "outside_surface" not in str(record["quality_flags"])
                and "large_snap" not in str(record["quality_flags"])
            ], dtype=np.float32)
            if edt_candidates.size:
                median = float(np.median(edt_candidates))
                mad = float(np.median(np.abs(edt_candidates - median)))
                if mad > 0:
                    edt_candidates = edt_candidates[np.abs(edt_candidates - median) <= 3.5 * 1.4826 * mad]
                width = float(np.median(edt_candidates)) if edt_candidates.size else median
                chain_widths[chain.chain_id] = (width, "C", "chain_hybrid_v2_edt_fallback")

    # Resolve chains with no usable surface crossing from direction-continuous
    # neighbours. At junctions, an unrestricted median can assign a wide main
    # road's width to a narrow branch merely because they share one node.
    node_to_chains: dict[int, set[int]] = {}
    for chain_id, end_nodes in chain_end_nodes.items():
        for node_id in end_nodes:
            node_to_chains.setdefault(node_id, set()).add(chain_id)
    chain_by_id = {chain.chain_id: chain for chain in chains}
    direction_cosine = float(np.clip(getattr(cfg, "neighbor_direction_cosine", 0.80), 0.0, 1.0))
    direction_rejected_count = 0
    direction_inherited_count = 0
    for _ in range(max(1, len(chains))):
        pending: dict[int, float] = {}
        unresolved_chain_ids = [chain.chain_id for chain in chains if chain.chain_id not in chain_widths]
        for chain_id in unresolved_chain_ids:
            candidates: list[tuple[float, float]] = []
            chain = chain_by_id[chain_id]
            for node_id in chain_end_nodes.get(chain_id, ()):
                direction = _endpoint_direction(chain, nodes, node_id, sample_step * 4.0)
                for neighbour_id in node_to_chains.get(node_id, set()):
                    if neighbour_id == chain_id or neighbour_id not in chain_widths:
                        continue
                    neighbour = chain_by_id[neighbour_id]
                    neighbour_direction = _endpoint_direction(
                        neighbour, nodes, node_id, sample_step * 4.0
                    )
                    cosine = abs(float(np.dot(direction, neighbour_direction)))
                    if cosine < direction_cosine:
                        direction_rejected_count += 1
                        continue
                    candidates.append((cosine, float(chain_widths[neighbour_id][0])))
            if candidates:
                best_cosine = max(item[0] for item in candidates)
                aligned_widths = [
                    width for cosine, width in candidates if cosine >= best_cosine - 0.05
                ]
                pending[chain_id] = float(np.median(np.asarray(aligned_widths, dtype=np.float32)))
        if not pending:
            break
        for chain_id, width in pending.items():
            chain_widths[chain_id] = (
                width, "C", "chain_hybrid_v2_directional_neighbor_fallback",
            )
            direction_inherited_count += 1

    known_widths = [value[0] for value in chain_widths.values() if value[0] > 0]
    if known_widths:
        tile_fallback = float(np.median(np.asarray(known_widths, dtype=np.float32)))
    else:
        positive_distance = distance[distance > 0]
        tile_fallback = float(2.0 * np.percentile(positive_distance, 90)) if positive_distance.size else 1.0
    for chain in chains:
        if chain.chain_id not in chain_widths:
            chain_widths[chain.chain_id] = (tile_fallback, "C", "chain_hybrid_v2_tile_fallback")

    segmented_chain_ids = {int(row["chain_id"]) for row in segments}
    for chain_id, (width, grade, source) in chain_widths.items():
        if chain_id in segmented_chain_ids or chain_id not in chain_geometry:
            continue
        points, total = chain_geometry[chain_id]
        start, end = points[0], points[-1]
        edge_ids = next(chain.edge_ids for chain in chains if chain.chain_id == chain_id)
        segments.append({
            "width_segment_id": len(segments), "chain_id": chain_id,
            "start_sample_index": -1, "end_sample_index": -1, "sample_count": 0,
            "start_distance_px": 0.0, "end_distance_px": total,
            "start_row": float(start[0]), "start_col": float(start[1]),
            "end_row": float(end[0]), "end_col": float(end[1]),
            "median_width_px": width, "mean_width_px": width,
            "median_width_units": width * pixel_size, "width_cv": 0.0,
            "edge_ids_json": json.dumps(edge_ids, separators=(",", ":")),
            "quality_grade": grade, "width_source": source,
        })

    edge_widths = []
    measured = 0
    grade_counts = {"A": 0, "B": 0, "C": 0}
    for edge_id in range(len(edges)):
        values = edge_values[edge_id]
        if values:
            width = float(np.median(np.asarray(values, dtype=np.float32)))
            status, source = "measured_chain_hybrid", "chain_hybrid_v2"
            grade = "A"
        elif edge_id in edge_chain_fallback:
            width = edge_chain_fallback[edge_id]
            status, source = "interpolated_on_road_chain", "chain_hybrid_v2_chain_fallback"
        else:
            chain_id = edge_chain_id.get(edge_id)
            width, grade, source = chain_widths.get(
                chain_id, (tile_fallback, "C", "chain_hybrid_v2_tile_fallback")
            )
            status = "auto_estimated_width"
        if values or edge_id in edge_chain_fallback:
            grade = "A" if values else "B"
        measured += int(width > 0)
        grade_counts[grade] += 1
        edge_widths.append({
            "edge_id": edge_id, "width_px": width, "width_units": width * pixel_size,
            "source": source, "status": status, "quality_grade": grade,
        })
    issue_count = sum(bool(row["quality_flags"]) and not row["interpolated"] for row in samples)
    return FinalWidthResult(
        edge_widths=edge_widths, samples=samples, segments=segments, algorithm="chain_hybrid_v2",
        metadata={
            "road_chain_count": len(chains), "measured_edge_count": measured,
            "unresolved_edge_count": len(edges) - measured, "issue_sample_count": issue_count,
            "sample_step_px": sample_step, "hybrid_agreement_ratio": agreement_limit,
            "neighbor_direction_cosine": direction_cosine,
            "direction_inherited_chain_count": direction_inherited_count,
            "direction_rejected_neighbor_count": direction_rejected_count,
            "quality_grade_counts": grade_counts,
        },
    )
