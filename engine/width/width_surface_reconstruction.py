from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class WidthSurfaceConfig:
    min_width_px: float = 1.0
    minimum_render_width_px: float = 3.0
    width_scale: float = 1.0
    close_kernel: int = 3
    min_component_area_px: int = 8
    same_direction_cosine: float = 0.80
    neighbor_width_hops: int = 8
    junction_min_degree: int = 3
    junction_setback_factor: float = 1.75
    junction_max_radius_px: float = 60.0
    parallel_cosine: float = 0.94
    parallel_sample_step_px: float = 4.0
    parallel_min_overlap_px: float = 20.0
    parallel_max_centerline_spacing_px: float = 45.0
    parallel_max_surface_gap_px: float = 16.0
    preserve_reference_surface: bool = False
    # When enabled, each uninterrupted road chain uses one median width;
    # degree-2 bends get a small join only to prevent raster gaps.
    regular_surface: bool = False
    regular_corridor_cosine: float = 0.94
    regular_corridor_width_ratio: float = 0.35
    centerline_support_margin_px: int = 4
    min_unsupported_area_px: int = 12
    chain_width_max_deviation_ratio: float = 0.25
    continuity_close_kernel: int = 7
    continuity_max_gap_px: int = 8
    boundary_snap_min_px: float = 1.0
    boundary_snap_max_px: float = 3.0
    low_boundary_gradient: float = 0.08
    contour_simplify_tolerance_px: float = 1.0
    boundary_smooth_sigma_px: float = 1.5
    missing_centerline_coverage: float = 0.80
    continuity_endpoint_radius_px: int = 3
    max_enclosed_hole_half_width_px: float = 8.0
    max_enclosed_hole_area_px: int = 12000


@dataclass
class WidthSurfaceResult:
    surface: np.ndarray
    added: np.ndarray
    removed: np.ndarray
    metadata: dict


def _positive_float(row: dict, *keys: str) -> float:
    for key in keys:
        try:
            value = float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    return 0.0


def _point(row: dict) -> np.ndarray | None:
    for row_key, col_key in (("row", "col"), ("row_used", "col_used")):
        try:
            point = np.asarray([float(row[row_key]), float(row[col_key])], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(point)):
            return point
    return None


def _draw_variable_edge(
    canvas: np.ndarray,
    start_rc: np.ndarray,
    end_rc: np.ndarray,
    anchors: list[tuple[float, float]],
    width_scale: float,
) -> None:
    height, width = canvas.shape
    vector = end_rc - start_rc

    def point_at(fraction: float) -> tuple[int, int]:
        row, col = start_rc + float(fraction) * vector
        return (
            int(np.clip(round(float(col)), 0, width - 1)),
            int(np.clip(round(float(row)), 0, height - 1)),
        )

    def raster_width(road_width: float) -> int:
        target = max(1, int(round(road_width * width_scale)))
        # OpenCV expands thick lines by roughly one pixel on each side.
        return 1 if target <= 2 else target - 2

    for index, (fraction, road_width) in enumerate(anchors):
        target_width = max(1, int(round(road_width * width_scale)))
        cv2.circle(canvas, point_at(fraction), max(0, target_width // 2), 1, -1, cv2.LINE_8)
        if index == 0:
            continue
        previous_fraction, previous_width = anchors[index - 1]
        segment_width = raster_width(0.5 * (previous_width + road_width))
        cv2.line(
            canvas,
            point_at(previous_fraction),
            point_at(fraction),
            1,
            segment_width,
            cv2.LINE_8,
        )


def _draw_buffer_edge(
    canvas: np.ndarray,
    start_rc: np.ndarray,
    end_rc: np.ndarray,
    width_px: float,
    width_scale: float = 1.0,
) -> None:
    """Rasterize a fixed-width, flat-cap buffer around a straight edge."""
    vector = np.asarray(end_rc, dtype=np.float32) - np.asarray(start_rc, dtype=np.float32)
    length = float(np.linalg.norm(vector))
    if length <= 1e-6 or width_px <= 0:
        return
    tangent = vector / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    half = 0.5 * float(width_px) * float(width_scale)
    polygon_rc = np.asarray(
        [start_rc - normal * half, start_rc + normal * half,
         end_rc + normal * half, end_rc - normal * half],
        dtype=np.float32,
    )
    polygon_xy = np.rint(polygon_rc[:, ::-1]).astype(np.int32)
    cv2.fillPoly(canvas, [polygon_xy], 1, cv2.LINE_8)


def _fill_regular_chain_joins(
    canvas: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    width_scale: float,
) -> int:
    """Close raster gaps at degree-2 bends without making junction disks."""
    adjacency = _edge_adjacency(edges, len(nodes))
    filled = 0
    for node_id, incident in enumerate(adjacency):
        # A degree-2 node is a polyline bend or split point.  Degree >= 3 is a
        # real intersection and must remain the union of branch buffers only.
        if len(incident) != 2:
            continue
        positive = [float(widths[edge_id]) for edge_id in incident if widths[edge_id] > 0]
        if not positive:
            continue
        # The two flat-cap branch buffers already overlap at an ordinary bend.
        # This is strictly a raster-gap stitch, never a width-sized round cap:
        # a full half-width circle turns a dense degree-2 split with an
        # uncertain width into exactly the junction disk we must avoid.
        radius = min(3, max(1, int(np.ceil(0.5 * float(np.median(positive)) * width_scale))))
        row, col = (int(round(float(value))) for value in nodes[node_id])
        cv2.circle(canvas, (col, row), radius, 1, -1, cv2.LINE_8)
        filled += 1
    return filled


def _fill_regular_junction_joins(
    canvas: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    width_scale: float,
    minimum_degree: int = 3,
) -> int:
    """Fill a compact polygonal junction core bounded by branch half-widths."""
    adjacency = _edge_adjacency(edges, len(nodes))
    filled = 0
    height, width = canvas.shape
    for node_id, incident in enumerate(adjacency):
        if len(incident) < minimum_degree:
            continue
        center = nodes[node_id]
        corners: list[np.ndarray] = []
        for edge_id in incident:
            road_width = float(widths[edge_id]) * float(width_scale)
            if road_width <= 0:
                continue
            src_idx, dst_idx = (int(value) for value in edges[edge_id])
            other_id = dst_idx if src_idx == node_id else src_idx
            vector = nodes[other_id] - center
            length = float(np.linalg.norm(vector))
            if length <= 1e-6:
                continue
            tangent = vector / length
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
            half = 0.5 * road_width
            # Extending only one branch half-width from the node closes the
            # intersection wedge without recreating the former oversized disk.
            section = center + tangent * min(half, 0.5 * length)
            corners.extend((section - normal * half, section + normal * half))
        if len(corners) < 3:
            continue
        points_xy = np.asarray(corners, dtype=np.float32)[:, ::-1]
        points_xy[:, 0] = np.clip(points_xy[:, 0], 0, width - 1)
        points_xy[:, 1] = np.clip(points_xy[:, 1], 0, height - 1)
        hull = cv2.convexHull(np.rint(points_xy).astype(np.int32))
        cv2.fillConvexPoly(canvas, hull, 1, cv2.LINE_8)
        filled += 1
    return filled


def _normalized_probability(probability: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if probability is None:
        return None
    result = np.asarray(probability, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"Probability shape mismatch: {result.shape} != {shape}")
    if result.max(initial=0.0) > 1.0:
        result = result / 255.0
    return np.clip(result, 0.0, 1.0)


def _probability_gradient(probability: np.ndarray | None) -> np.ndarray | None:
    if probability is None:
        return None
    dx = cv2.Sobel(probability, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(probability, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)


def _snap_boundary_point(
    point: np.ndarray,
    center: np.ndarray,
    gradient: np.ndarray | None,
    config: WidthSurfaceConfig,
) -> np.ndarray:
    if gradient is None:
        return point
    direction = point - center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return point
    direction /= norm
    offsets = np.linspace(
        -float(config.boundary_snap_max_px),
        float(config.boundary_snap_max_px),
        max(3, int(round(config.boundary_snap_max_px * 4.0)) + 1),
    )
    candidates = point[None, :] + offsets[:, None] * direction[None, :]
    rows = np.clip(np.rint(candidates[:, 0]).astype(np.int32), 0, gradient.shape[0] - 1)
    cols = np.clip(np.rint(candidates[:, 1]).astype(np.int32), 0, gradient.shape[1] - 1)
    scores = gradient[rows, cols]
    best = int(np.argmax(scores))
    if float(scores[best]) <= 0:
        return point
    offset = float(offsets[best])
    if abs(offset) < float(config.boundary_snap_min_px):
        return point
    return candidates[best].astype(np.float32)


def _draw_asymmetric_edge(
    canvas: np.ndarray,
    start_rc: np.ndarray,
    end_rc: np.ndarray,
    rows: list[dict],
    fallback_width: float,
    gradient: np.ndarray | None,
    config: WidthSurfaceConfig,
) -> bool:
    vector = end_rc - start_rc
    length_sq = float(np.dot(vector, vector))
    length = float(np.sqrt(length_sq))
    if length <= 1e-6:
        return False
    tangent = vector / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    profile: list[tuple[float, np.ndarray, float, float, np.ndarray | None, np.ndarray | None]] = []
    for row in rows:
        if str(row.get("quality_grade", "A") or "A").upper() == "C":
            continue
        point = _point(row)
        if point is None:
            continue
        fraction = float(np.clip(np.dot(point - start_rc, vector) / length_sq, 0.0, 1.0))
        left = _positive_float(row, "left_px", "left_distance_px", "dL")
        right = _positive_float(row, "right_px", "right_distance_px", "dR")
        if left <= 0 or right <= 0:
            width = _positive_float(row, "width_px", "median_width_px")
            left = right = 0.5 * width
        if left <= 0 or right <= 0:
            continue
        try:
            left_point = np.asarray(
                [float(row["left_boundary_row"]), float(row["left_boundary_col"])], dtype=np.float32
            )
            right_point = np.asarray(
                [float(row["right_boundary_row"]), float(row["right_boundary_col"])], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError):
            left_point = right_point = None
        profile.append((
            fraction, point, left * config.width_scale, right * config.width_scale,
            left_point, right_point,
        ))
    if not profile:
        if fallback_width < config.min_width_px:
            return False
        profile = [
            (0.0, start_rc, 0.5 * fallback_width * config.width_scale, 0.5 * fallback_width * config.width_scale, None, None),
            (1.0, end_rc, 0.5 * fallback_width * config.width_scale, 0.5 * fallback_width * config.width_scale, None, None),
        ]
    profile.sort(key=lambda item: item[0])
    if profile[0][0] > 1e-4:
        profile.insert(0, (0.0, start_rc, profile[0][2], profile[0][3], None, None))
    if profile[-1][0] < 1.0 - 1e-4:
        profile.append((1.0, end_rc, profile[-1][2], profile[-1][3], None, None))

    previous: tuple[np.ndarray, np.ndarray] | None = None
    for _, center, left_distance, right_distance, measured_left, measured_right in profile:
        raw_left = measured_left if measured_left is not None else center - normal * left_distance
        raw_right = measured_right if measured_right is not None else center + normal * right_distance
        left = _snap_boundary_point(raw_left, center, gradient, config)
        right = _snap_boundary_point(raw_right, center, gradient, config)
        if previous is not None:
            polygon_rc = np.asarray([previous[0], left, right, previous[1]], dtype=np.float32)
            polygon_xy = np.rint(polygon_rc[:, ::-1]).astype(np.int32)
            cv2.fillPoly(canvas, [polygon_xy], 1, cv2.LINE_8)
        previous = (left, right)
    return True


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or not np.any(binary):
        return binary.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    result = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            result[labels == label] = 1
    return result


def _prune_unsupported_reference(
    reference: np.ndarray,
    width_corridors: np.ndarray,
    config: WidthSurfaceConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove connected surface lobes that extend beyond any final centerline corridor."""
    margin = max(0, int(config.centerline_support_margin_px))
    if margin:
        size = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        support = cv2.dilate((width_corridors > 0).astype(np.uint8), kernel) > 0
    else:
        support = width_corridors > 0
    unsupported = ((reference > 0) & ~support).astype(np.uint8)
    removed = np.zeros_like(reference, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(unsupported, connectivity=8)
    removed_component_count = 0
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < max(1, int(config.min_unsupported_area_px)):
            continue
        removed[labels == label] = 1
        removed_component_count += 1
    pruned = (reference > 0).astype(np.uint8)
    pruned[removed > 0] = 0
    return pruned, removed, {
        "centerline_support_margin_px": margin,
        "unsupported_surface_removed_px": int(np.count_nonzero(removed)),
        "unsupported_surface_component_count": removed_component_count,
    }


def _edge_directions(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    directions = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    lengths = np.linalg.norm(directions, axis=1)
    valid = lengths > 1e-6
    directions[valid] /= lengths[valid, None]
    directions[~valid] = np.asarray([0.0, 1.0], dtype=np.float32)
    return directions


def _edge_adjacency(edges: np.ndarray, node_count: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        adjacency[int(src_idx)].append(edge_id)
        adjacency[int(dst_idx)].append(edge_id)
    return adjacency


def _reference_width_for_edge(
    reference_distance: np.ndarray | None,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    if reference_distance is None:
        return 0.0
    length = float(np.linalg.norm(end - start))
    count = max(3, int(np.ceil(length / 8.0)))
    fractions = (np.arange(count, dtype=np.float32) + 0.5) / count
    points = start[None, :] * (1.0 - fractions[:, None]) + end[None, :] * fractions[:, None]
    rows = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, reference_distance.shape[0] - 1)
    cols = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, reference_distance.shape[1] - 1)
    values = 2.0 * reference_distance[rows, cols]
    values = values[values > 0]
    return float(np.median(values)) if values.size else 0.0


def _supplemental_path_groups(edge_metadata: list[dict] | tuple[dict, ...] | None) -> dict[str, list[int]]:
    """Group generated path segments that represent one supplemental road."""
    groups: dict[str, list[int]] = {}
    if not edge_metadata:
        return groups
    supplemental_sources = {"auto_added_gap", "auto_added_surface", "review_added_candidate"}
    for edge_id, row in enumerate(edge_metadata):
        source = str(row.get("source", ""))
        if source not in supplemental_sources:
            continue
        key = str(row.get("line_feature_id", "") or row.get("candidate_id", ""))
        if key:
            groups.setdefault(f"{source}:{key}", []).append(edge_id)
    return groups


def _regularize_supplemental_path_widths(
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    reliable: np.ndarray,
    edge_metadata: list[dict] | tuple[dict, ...] | None,
    config: WidthSurfaceConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Give each generated path a stable width and inherit continuity-gap widths.

    Gap candidates cross places where the source surface is absent, so a normal
    probe on those segments is not width evidence.  Surface-skeleton candidates
    retain their own robust width unless it is an extreme mismatch with an
    aligned road at an attached endpoint.
    """
    result = np.asarray(widths, dtype=np.float32).copy()
    trusted = np.asarray(reliable, dtype=bool).copy()
    groups = _supplemental_path_groups(edge_metadata)
    if not groups:
        return result, trusted, {
            "supplemental_path_width_group_count": 0,
            "supplemental_path_inherited_edge_count": 0,
            "supplemental_path_regularized_edge_count": 0,
        }
    directions = _edge_directions(nodes, edges)
    adjacency = _edge_adjacency(edges, len(nodes))
    inherited_count = 0
    regularized_count = 0
    for edge_ids in groups.values():
        valid_ids = [edge_id for edge_id in edge_ids if 0 <= edge_id < len(edges)]
        if not valid_ids:
            continue
        group_set = set(valid_ids)
        group_node_edges: dict[int, list[int]] = {}
        for edge_id in valid_ids:
            for node_id in edges[edge_id].tolist():
                group_node_edges.setdefault(int(node_id), []).append(edge_id)
        attached_widths: list[float] = []
        for node_id, incident_group in group_node_edges.items():
            if len(incident_group) != 1:
                continue
            group_edge = incident_group[0]
            candidates: list[tuple[float, float]] = []
            for other_id in adjacency[node_id]:
                if other_id in group_set or result[other_id] < config.min_width_px or not trusted[other_id]:
                    continue
                cosine = abs(float(np.dot(directions[group_edge], directions[other_id])))
                if cosine >= config.same_direction_cosine:
                    candidates.append((cosine, float(result[other_id])))
            if candidates:
                best = max(cosine for cosine, _ in candidates)
                attached_widths.extend(width for cosine, width in candidates if cosine >= best - 0.05)

        own_ids = [edge_id for edge_id in valid_ids if trusted[edge_id] and result[edge_id] >= config.min_width_px]
        own_width = float(np.median(result[own_ids])) if own_ids else 0.0
        attached_width = float(np.median(np.asarray(attached_widths, dtype=np.float32))) if attached_widths else 0.0
        source = str(edge_metadata[valid_ids[0]].get("source", "")) if edge_metadata else ""
        target_width = own_width
        inherit = False
        if attached_width >= config.min_width_px:
            if source in {"auto_added_gap", "review_added_candidate"} or own_width < config.min_width_px:
                target_width = attached_width
                inherit = True
            else:
                ratio = own_width / max(attached_width, 1.0)
                if ratio < 0.67 or ratio > 1.50:
                    target_width = attached_width
                    inherit = True
        if target_width < config.min_width_px:
            continue
        for edge_id in valid_ids:
            if abs(float(result[edge_id]) - target_width) > 1e-3:
                regularized_count += 1
            result[edge_id] = target_width
            if inherit or source in {"auto_added_gap", "review_added_candidate"}:
                trusted[edge_id] = True
                inherited_count += 1
    return result, trusted, {
        "supplemental_path_width_group_count": int(len(groups)),
        "supplemental_path_inherited_edge_count": int(inherited_count),
        "supplemental_path_regularized_edge_count": int(regularized_count),
    }


def _resolve_missing_widths(
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    reliable: np.ndarray,
    reference: np.ndarray,
    config: WidthSurfaceConfig,
    edge_metadata: list[dict] | tuple[dict, ...] | None = None,
) -> tuple[np.ndarray, dict]:
    # Grade-C / fallback widths are hypotheses, not width evidence.  In
    # particular, a short edge inside an intersection can report the diameter
    # of the whole junction.  Never let that raw value seed propagation.
    resolved = widths.copy()
    resolved[~reliable] = 0.0
    resolved, reliable, supplemental_metadata = _regularize_supplemental_path_widths(
        nodes, edges, resolved, reliable, edge_metadata, config
    )
    directions = _edge_directions(nodes, edges)
    adjacency = _edge_adjacency(edges, len(nodes))
    inherited = np.zeros(len(edges), dtype=bool)

    for _ in range(max(1, config.neighbor_width_hops)):
        pending: dict[int, float] = {}
        for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
            if reliable[edge_id] or inherited[edge_id]:
                continue
            candidates: list[tuple[float, float]] = []
            for node_id in (int(src_idx), int(dst_idx)):
                for other_id in adjacency[node_id]:
                    if (
                        other_id == edge_id
                        or resolved[other_id] < config.min_width_px
                        or not (reliable[other_id] or inherited[other_id])
                    ):
                        continue
                    cosine = abs(float(np.dot(directions[edge_id], directions[other_id])))
                    if cosine >= config.same_direction_cosine:
                        candidates.append((cosine, float(resolved[other_id])))
            if candidates:
                best = max(value[0] for value in candidates)
                values = [value for cosine, value in candidates if cosine >= best - 0.05]
                pending[edge_id] = float(np.median(np.asarray(values, dtype=np.float32)))
        if not pending:
            break
        for edge_id, value in pending.items():
            resolved[edge_id] = value
            inherited[edge_id] = True

    reference_distance = (
        cv2.distanceTransform((reference > 0).astype(np.uint8), cv2.DIST_L2, 5)
        if np.any(reference) else None
    )
    reference_fallback = np.zeros(len(edges), dtype=bool)
    component_fallback = np.zeros(len(edges), dtype=bool)
    global_fallback = np.zeros(len(edges), dtype=bool)
    degrees = np.asarray([len(items) for items in adjacency], dtype=np.int32)
    junction_reference_rejected = np.zeros(len(edges), dtype=bool)
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        if resolved[edge_id] >= config.min_width_px and (reliable[edge_id] or inherited[edge_id]):
            continue
        # A distance transform measured on a T/X junction is its local lobe
        # radius, not the branch width.  Let component fallback use actual
        # neighbouring evidence for any low-confidence edge incident to an
        # intersection instead of reintroducing a giant junction width.
        if not reliable[edge_id] and (
            degrees[int(src_idx)] >= config.junction_min_degree
            or degrees[int(dst_idx)] >= config.junction_min_degree
        ):
            junction_reference_rejected[edge_id] = True
            continue
        local_width = _reference_width_for_edge(
            reference_distance, nodes[int(src_idx)], nodes[int(dst_idx)]
        )
        if local_width >= config.min_width_px:
            resolved[edge_id] = local_width
            reference_fallback[edge_id] = True

    known_global = resolved[(reliable | inherited | reference_fallback) & (resolved >= config.min_width_px)]
    global_width = float(np.median(known_global)) if known_global.size else config.minimum_render_width_px
    # Remaining edges inherit the median of their connected component before the tile median.
    unseen = set(range(len(edges)))
    while unseen:
        seed = unseen.pop()
        component = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            for node_id in edges[current].tolist():
                for other_id in adjacency[int(node_id)]:
                    if other_id in unseen:
                        unseen.remove(other_id)
                        component.add(other_id)
                        queue.append(other_id)
        component_ids = np.asarray(sorted(component), dtype=np.int32)
        known = resolved[component_ids][
            (reliable | inherited | reference_fallback)[component_ids]
            & (resolved[component_ids] >= config.min_width_px)
        ]
        component_width = float(np.median(known)) if known.size else global_width
        for edge_id in component:
            if resolved[edge_id] >= config.min_width_px and (
                reliable[edge_id] or inherited[edge_id] or reference_fallback[edge_id]
            ):
                continue
            resolved[edge_id] = component_width
            if known.size:
                component_fallback[edge_id] = True
            else:
                global_fallback[edge_id] = True

    resolved = np.maximum(resolved, config.minimum_render_width_px)
    return resolved, {
        **supplemental_metadata,
        "same_direction_inherited_edge_count": int(np.count_nonzero(inherited)),
        "reference_width_fallback_edge_count": int(np.count_nonzero(reference_fallback)),
        "component_width_fallback_edge_count": int(np.count_nonzero(component_fallback)),
        "global_width_fallback_edge_count": int(np.count_nonzero(global_fallback)),
        "junction_reference_width_rejected_edge_count": int(np.count_nonzero(junction_reference_rejected)),
    }


def _fill_junction_envelopes(
    surface: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    reference: np.ndarray,
    probability: np.ndarray | None,
    config: WidthSurfaceConfig,
) -> int:
    adjacency = _edge_adjacency(edges, len(nodes))
    height, width = surface.shape
    filled = 0
    for node_id, incident_edges in enumerate(adjacency):
        if len(incident_edges) < config.junction_min_degree:
            continue
        incident_widths = [float(widths[edge_id]) for edge_id in incident_edges if widths[edge_id] > 0]
        if not incident_widths:
            continue
        setback = min(
            config.junction_max_radius_px,
            max(incident_widths) * config.junction_setback_factor,
        )
        center = nodes[node_id]
        radius = int(np.ceil(min(
            config.junction_max_radius_px,
            max(setback, 1.5 * max(incident_widths) * config.width_scale),
        )))
        center_row, center_col = (int(round(float(value))) for value in center)
        row0, row1 = max(0, center_row - radius), min(height, center_row + radius + 1)
        col0, col1 = max(0, center_col - radius), min(width, center_col + radius + 1)
        local_seed = np.zeros((row1 - row0, col1 - col0), dtype=np.uint8)
        for edge_id in incident_edges:
            src_idx, dst_idx = (int(value) for value in edges[edge_id])
            other_id = dst_idx if src_idx == node_id else src_idx
            direction = nodes[other_id] - center
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                continue
            direction /= norm
            section_center = center + direction * setback
            thickness = max(3, int(round(float(widths[edge_id]) * config.width_scale)))
            cv2.line(
                local_seed,
                (center_col - col0, center_row - row0),
                (int(round(float(section_center[1]))) - col0, int(round(float(section_center[0]))) - row0),
                1, thickness, cv2.LINE_8,
            )
        circle = np.zeros_like(local_seed)
        cv2.circle(circle, (center_col - col0, center_row - row0), radius, 1, -1, cv2.LINE_8)
        local_reference = reference[row0:row1, col0:col1] > 0
        if probability is not None:
            local_probability = probability[row0:row1, col0:col1]
            probability_candidate = local_probability >= 0.30
        else:
            probability_candidate = np.zeros_like(local_reference)
        # The branch ribbons are seeds, while SAM probability and the original
        # pavement supply the actual non-convex intersection envelope.
        candidate = ((local_reference | probability_candidate) & (circle > 0)) | (local_seed > 0)
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        count, labels = cv2.connectedComponents(candidate, connectivity=8)
        selected_labels = np.unique(labels[local_seed > 0])
        selected_labels = selected_labels[selected_labels > 0]
        selected = np.isin(labels, selected_labels).astype(np.uint8) if selected_labels.size else local_seed
        surface[row0:row1, col0:col1] |= selected
        filled += 1
    return filled


def _selective_boundary_update(
    reference: np.ndarray,
    profile: np.ndarray,
    probability: np.ndarray | None,
    gradient: np.ndarray | None,
    config: WidthSurfaceConfig,
    protected_zone: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace only pixel-scale spikes and low-confidence boundary bands."""
    boundary = cv2.morphologyEx(reference, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    neighbor_count = cv2.boxFilter(
        reference.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    isolated_foreground = (reference > 0) & (neighbor_count <= 3.0)
    isolated_hole = (reference == 0) & (neighbor_count >= 6.0)
    high_frequency = boundary & (isolated_foreground | isolated_hole)
    if probability is not None and gradient is not None:
        low_confidence = boundary & (
            (gradient < float(config.low_boundary_gradient))
            | (
                (probability > 0.25)
                & (probability < 0.65)
                & (gradient < 2.0 * float(config.low_boundary_gradient))
            )
        )
    else:
        low_confidence = np.zeros_like(boundary)
    replace_seed = high_frequency | low_confidence
    if protected_zone is not None:
        replace_seed &= protected_zone == 0
    replace_zone = cv2.dilate(
        replace_seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    result = reference.copy()
    result[replace_zone] = profile[replace_zone]
    return result.astype(np.uint8), replace_zone.astype(np.uint8)


def _junction_core_mask(
    shape: tuple[int, int],
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    config: WidthSurfaceConfig,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    adjacency = _edge_adjacency(edges, len(nodes))
    for node_id, incident_edges in enumerate(adjacency):
        if len(incident_edges) < config.junction_min_degree:
            continue
        local_widths = [float(widths[edge_id]) for edge_id in incident_edges if widths[edge_id] > 0]
        if not local_widths:
            continue
        radius = int(np.ceil(min(
            config.junction_max_radius_px,
            max(local_widths) * config.junction_setback_factor * config.width_scale,
        )))
        row, col = nodes[node_id]
        cv2.circle(mask, (int(round(float(col))), int(round(float(row)))), radius, 1, -1, cv2.LINE_8)
    return mask


def _segment_reference_coverage(reference: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    length = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(length)) + 1)
    rows = np.clip(
        np.rint(np.linspace(start[0], end[0], count)).astype(np.int32), 0, reference.shape[0] - 1
    )
    cols = np.clip(
        np.rint(np.linspace(start[1], end[1], count)).astype(np.int32), 0, reference.shape[1] - 1
    )
    return float(np.mean(reference[rows, cols] > 0))


def _edge_centerline_mask(shape: tuple[int, int], start: np.ndarray, end: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(
        mask,
        (int(round(float(start[1]))), int(round(float(start[0])))),
        (int(round(float(end[1]))), int(round(float(end[0])))),
        1, 1, cv2.LINE_8,
    )
    return mask


def _c_continuity_edge_ids(
    nodes: np.ndarray,
    edges: np.ndarray,
    grades: dict[int, str],
    anchored_surface: np.ndarray,
    config: WidthSurfaceConfig,
) -> set[int]:
    """Select C-only components that bridge two already supported road terminals."""
    c_edges = {edge_id for edge_id in range(len(edges)) if grades.get(edge_id) == "C"}
    if not c_edges:
        return set()
    adjacency = _edge_adjacency(edges, len(nodes))
    radius = max(1, int(config.continuity_endpoint_radius_px))
    selected: set[int] = set()
    unseen = set(c_edges)
    while unseen:
        seed = unseen.pop()
        component = {seed}
        queue = [seed]
        component_nodes: set[int] = set()
        while queue:
            edge_id = queue.pop()
            for node_id in edges[edge_id].tolist():
                node_id = int(node_id)
                component_nodes.add(node_id)
                for other_id in adjacency[node_id]:
                    if other_id in unseen and other_id in c_edges:
                        unseen.remove(other_id)
                        component.add(other_id)
                        queue.append(other_id)
        terminals = [
            node_id for node_id in component_nodes
            if sum(edge_id in component for edge_id in adjacency[node_id]) == 1
        ]
        supported_terminals = 0
        for node_id in terminals:
            row, col = nodes[node_id]
            row0, row1 = max(0, int(round(row)) - radius), min(anchored_surface.shape[0], int(round(row)) + radius + 1)
            col0, col1 = max(0, int(round(col)) - radius), min(anchored_surface.shape[1], int(round(col)) + radius + 1)
            supported_terminals += int(np.any(anchored_surface[row0:row1, col0:col1]))
        if supported_terminals >= 2:
            selected.update(component)
    return selected


def _simplify_contours(binary: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance <= 0 or not np.any(binary):
        return binary.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return binary.astype(np.uint8)
    result = np.zeros_like(binary, dtype=np.uint8)
    for index, contour in enumerate(contours):
        simplified = cv2.approxPolyDP(contour, float(tolerance), True)
        color = 1 if int(hierarchy[0, index, 3]) < 0 else 0
        cv2.drawContours(result, [simplified], -1, color, -1, cv2.LINE_8)
    return result


def _smooth_surface_boundary(binary: np.ndarray, sigma: float) -> np.ndarray:
    """Suppress pixel-scale boundary noise without moving straight edges."""
    if sigma <= 0 or not np.any(binary):
        return binary.astype(np.uint8)
    blurred = cv2.GaussianBlur(
        binary.astype(np.float32),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REPLICATE,
    )
    return (blurred >= 0.5).astype(np.uint8)


def _enclosed_holes(binary: np.ndarray) -> np.ndarray:
    """Return background pixels disconnected from the image border."""
    background = (binary == 0).astype(np.uint8)
    padded = cv2.copyMakeBorder(background, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    flood = padded.copy()
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 2)
    exterior = flood[1:-1, 1:-1] == 2
    return ((background > 0) & ~exterior).astype(np.uint8)


def _enclosed_sliver_holes(
    binary: np.ndarray,
    allowed_surface: np.ndarray,
    max_half_width_px: float,
    max_area_px: int,
) -> np.ndarray:
    holes = _enclosed_holes(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    selected = np.zeros_like(holes, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > max(1, int(max_area_px)):
            continue
        component = (labels == label).astype(np.uint8)
        half_width = float(cv2.distanceTransform(component, cv2.DIST_L2, 5).max(initial=0.0))
        if half_width > float(max_half_width_px):
            continue
        selected |= component & allowed_surface.astype(np.uint8)
    return selected


def _edge_chain_ids(edges: np.ndarray, node_count: int) -> np.ndarray:
    adjacency = _edge_adjacency(edges, node_count)
    degrees = np.asarray([len(items) for items in adjacency], dtype=np.int32)
    chain_ids = np.full(len(edges), -1, dtype=np.int32)
    chain_id = 0

    def trace(start_node: int, first_edge: int) -> None:
        nonlocal chain_id
        current_node = start_node
        edge_id = first_edge
        while chain_ids[edge_id] < 0:
            chain_ids[edge_id] = chain_id
            src_idx, dst_idx = (int(value) for value in edges[edge_id])
            next_node = dst_idx if src_idx == current_node else src_idx
            if degrees[next_node] != 2:
                break
            remaining = [item for item in adjacency[next_node] if chain_ids[item] < 0]
            if not remaining:
                break
            current_node, edge_id = next_node, remaining[0]
        chain_id += 1

    for node_id in np.where(degrees != 2)[0].tolist():
        for edge_id in adjacency[node_id]:
            if chain_ids[edge_id] < 0:
                trace(node_id, edge_id)
    for edge_id in range(len(edges)):
        if chain_ids[edge_id] < 0:
            trace(int(edges[edge_id, 0]), edge_id)
    return chain_ids


def _regularize_chain_widths(
    widths: np.ndarray,
    edges: np.ndarray,
    node_count: int,
    max_deviation_ratio: float,
) -> tuple[np.ndarray, dict]:
    """Limit local support-width noise while retaining bounded real width changes."""
    regularized = np.asarray(widths, dtype=np.float32).copy()
    chain_ids = _edge_chain_ids(edges, node_count)
    changed = 0
    ratio = max(0.0, float(max_deviation_ratio))
    for chain_id in np.unique(chain_ids).tolist():
        edge_ids = np.where(chain_ids == chain_id)[0]
        values = regularized[edge_ids]
        positive = values[values > 0]
        if not positive.size:
            continue
        median = float(np.median(positive))
        lower = median * max(0.25, 1.0 - ratio)
        upper = median * (1.0 + ratio)
        clipped = np.clip(values, lower, upper)
        changed += int(np.count_nonzero(np.abs(clipped - values) > 1e-3))
        regularized[edge_ids] = clipped
    return regularized, {
        "road_chain_count": int(len(np.unique(chain_ids))),
        "chain_width_regularized_edge_count": changed,
        "chain_width_max_deviation_ratio": ratio,
    }


def _constant_chain_widths(
    widths: np.ndarray,
    edges: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, int]:
    """Return one robust width for every edge between adjacent intersections."""
    result = np.asarray(widths, dtype=np.float32).copy()
    chain_ids = _edge_chain_ids(edges, node_count)
    changed = 0
    for chain_id in np.unique(chain_ids).tolist():
        edge_ids = np.where(chain_ids == chain_id)[0]
        positive = result[edge_ids][result[edge_ids] > 0]
        if not positive.size:
            continue
        value = float(np.median(positive))
        changed += int(np.count_nonzero(np.abs(result[edge_ids] - value) > 1e-3))
        result[edge_ids] = value
    return result, changed


def _regular_corridor_widths(
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    config: WidthSurfaceConfig,
    reliable: np.ndarray | None = None,
    edge_metadata: list[dict] | tuple[dict, ...] | None = None,
) -> tuple[np.ndarray, int]:
    """Merge connected, near-collinear edges into constant-width corridors.

    Direction and topology define a continuous road.  Width disagreement is
    the value this pass must correct, so it must not prevent two aligned edges
    from joining the same corridor.
    """
    result = np.asarray(widths, dtype=np.float32).copy()
    directions = _edge_directions(nodes, edges)
    adjacency = _edge_adjacency(edges, len(nodes))
    parent = list(range(len(edges)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    # Resampling turns one generated road into many short edges and bends.
    # Its provenance is stronger continuity evidence than any single local
    # tangent, so keep the complete candidate path in one width group.
    for edge_ids in _supplemental_path_groups(edge_metadata).values():
        valid_ids = [edge_id for edge_id in edge_ids if 0 <= edge_id < len(edges)]
        for first, second in zip(valid_ids, valid_ids[1:]):
            union(first, second)

    for node_edges in adjacency:
        for index, first in enumerate(node_edges):
            for second in node_edges[index + 1:]:
                if result[first] <= 0 or result[second] <= 0:
                    continue
                cosine = abs(float(np.dot(directions[first], directions[second])))
                if cosine >= float(config.regular_corridor_cosine):
                    union(first, second)
    changed = 0
    groups: dict[int, list[int]] = {}
    for edge_id in range(len(edges)):
        groups.setdefault(find(edge_id), []).append(edge_id)
    for edge_ids in groups.values():
        selected_ids = edge_ids
        if reliable is not None:
            trusted_ids = [edge_id for edge_id in edge_ids if bool(reliable[edge_id])]
            if trusted_ids:
                selected_ids = trusted_ids
        values = result[selected_ids]
        positive = values[values > 0]
        if not positive.size:
            continue
        lengths = np.linalg.norm(
            nodes[edges[selected_ids, 1]] - nodes[edges[selected_ids, 0]], axis=1
        )
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = np.maximum(lengths[order], 1.0)
        value = float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), 0.5 * float(np.sum(sorted_weights)))])
        changed += int(np.count_nonzero(np.abs(result[edge_ids] - value) > 1e-3))
        result[edge_ids] = value
    return result, changed


def _cap_junction_nearby_width_spikes(
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    config: WidthSurfaceConfig,
) -> tuple[np.ndarray, int]:
    """Clamp only extreme width spikes in the immediate junction neighbourhood.

    Dense degree-2 sampling nodes may sit inside a junction even though they
    are not graph junctions themselves.  A normal probe there measures the
    junction lobe, so compare it with the robust local branch median before
    rasterising.  This deliberately leaves ordinary road-width transitions
    outside the local junction buffer unchanged.
    """
    result = np.asarray(widths, dtype=np.float32).copy()
    adjacency = _edge_adjacency(edges, len(nodes))
    junction_ids = [node_id for node_id, incident in enumerate(adjacency) if len(incident) >= config.junction_min_degree]
    if not junction_ids:
        return result, 0
    radius = max(20.0, min(40.0, float(config.junction_max_radius_px) * 0.5))
    changed = 0
    for junction_id in junction_ids:
        center = nodes[junction_id]
        nearby = []
        for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
            distance = min(
                float(np.linalg.norm(nodes[int(src_idx)] - center)),
                float(np.linalg.norm(nodes[int(dst_idx)] - center)),
            )
            if distance <= radius and result[edge_id] > 0:
                nearby.append(edge_id)
        if len(nearby) < 2:
            continue
        median = float(np.median(result[nearby]))
        if median <= 0:
            continue
        limit = max(median * 1.75, median + 10.0)
        for edge_id in nearby:
            if result[edge_id] > limit:
                result[edge_id] = limit
                changed += 1
    return result, changed


def _fill_parallel_road_gaps(
    surface: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    widths: np.ndarray,
    config: WidthSurfaceConfig,
) -> int:
    if len(edges) < 2:
        return 0
    directions = _edge_directions(nodes, edges)
    chain_ids = _edge_chain_ids(edges, len(nodes))
    adjacency = _edge_adjacency(edges, len(nodes))
    junction_nodes = np.asarray(
        [nodes[node_id] for node_id, incident in enumerate(adjacency) if len(incident) >= config.junction_min_degree],
        dtype=np.float32,
    ).reshape(-1, 2)
    junction_exclusion_radius = max(12.0, min(40.0, 0.5 * float(config.junction_max_radius_px)))
    step = max(2.0, config.parallel_sample_step_px)
    samples: list[tuple[np.ndarray, np.ndarray, int, int, float]] = []
    for edge_id, (src_idx, dst_idx) in enumerate(edges.tolist()):
        start, end = nodes[int(src_idx)], nodes[int(dst_idx)]
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / step)))
        for fraction in (np.arange(count, dtype=np.float32) + 0.5) / count:
            point = start * (1.0 - fraction) + end * fraction
            if junction_nodes.size and float(np.min(np.linalg.norm(junction_nodes - point, axis=1))) <= junction_exclusion_radius:
                continue
            samples.append((point, directions[edge_id], edge_id, int(chain_ids[edge_id]), float(widths[edge_id])))
    if len(samples) < 2:
        return 0

    cell_size = max(step, config.parallel_max_centerline_spacing_px)
    grid: dict[tuple[int, int], list[int]] = {}
    for index, (point, _, _, _, _) in enumerate(samples):
        key = (int(np.floor(point[0] / cell_size)), int(np.floor(point[1] / cell_size)))
        grid.setdefault(key, []).append(index)

    matches: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for index, (point, direction, _, chain_id, road_width) in enumerate(samples):
        row_cell = int(np.floor(point[0] / cell_size))
        col_cell = int(np.floor(point[1] / cell_size))
        best_by_chain: dict[int, tuple[float, np.ndarray]] = {}
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                for other_index in grid.get((row_cell + row_offset, col_cell + col_offset), []):
                    if other_index <= index:
                        continue
                    other_point, other_direction, _, other_chain, other_width = samples[other_index]
                    if other_chain == chain_id:
                        continue
                    cosine = abs(float(np.dot(direction, other_direction)))
                    if cosine < config.parallel_cosine:
                        continue
                    delta = other_point - point
                    distance = float(np.linalg.norm(delta))
                    if distance > config.parallel_max_centerline_spacing_px:
                        continue
                    lateral = abs(float(direction[0] * delta[1] - direction[1] * delta[0]))
                    along = abs(float(np.dot(direction, delta)))
                    half_width_sum = 0.5 * (road_width + other_width) * config.width_scale
                    surface_gap = lateral - half_width_sum
                    width_ratio = min(road_width, other_width) / max(road_width, other_width, 1.0)
                    if (
                        along > step * 1.75
                        or width_ratio < 0.45
                        or surface_gap < -2.0
                        or surface_gap > config.parallel_max_surface_gap_px
                    ):
                        continue
                    previous = best_by_chain.get(other_chain)
                    if previous is None or distance < previous[0]:
                        best_by_chain[other_chain] = (distance, other_point)
        for other_chain, (_, other_point) in best_by_chain.items():
            key = tuple(sorted((chain_id, other_chain)))
            matches.setdefault(key, []).append((point, other_point))

    minimum_matches = max(2, int(np.ceil(config.parallel_min_overlap_px / step)))
    filled_pairs = 0
    connector_width = max(3, int(round(step + 1)))
    for connectors in matches.values():
        if len(connectors) < minimum_matches:
            continue
        midpoints = np.asarray([(first + second) * 0.5 for first, second in connectors], dtype=np.float32)
        extent = float(np.max(np.ptp(midpoints, axis=0))) if len(midpoints) > 1 else 0.0
        if extent < config.parallel_min_overlap_px:
            continue
        for first, second in connectors:
            p0 = (int(round(float(first[1]))), int(round(float(first[0]))))
            p1 = (int(round(float(second[1]))), int(round(float(second[0]))))
            cv2.line(surface, p0, p1, 1, connector_width, cv2.LINE_8)
        filled_pairs += 1
    return filled_pairs


def reconstruct_surface_from_widths(
    shape: tuple[int, int],
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    edge_widths: list[dict],
    width_samples: list[dict],
    reference_surface: np.ndarray | None = None,
    config: WidthSurfaceConfig | None = None,
    road_probability: np.ndarray | None = None,
    edge_metadata: list[dict] | tuple[dict, ...] | None = None,
) -> WidthSurfaceResult:
    """Build the final polygon mask from final routes and their measured width profile."""
    config = config or WidthSurfaceConfig()
    nodes = np.asarray(nodes_rc, dtype=np.float32).reshape(-1, 2)
    graph_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    surface = np.zeros(shape, dtype=np.uint8)
    if reference_surface is None:
        reference = np.zeros(shape, dtype=np.uint8)
    else:
        reference = (np.asarray(reference_surface) > 0).astype(np.uint8)
        if reference.shape != shape:
            raise ValueError(f"Reference surface shape mismatch: {reference.shape} != {shape}")
    probability = _normalized_probability(road_probability, shape)
    probability_gradient = _probability_gradient(probability)
    samples_by_edge: dict[int, list[tuple[float, float]]] = {}
    sample_rows_by_edge: dict[int, list[dict]] = {}

    for sample in width_samples:
        width_px = _positive_float(sample, "width_px", "median_width_px")
        point = _point(sample)
        try:
            edge_id = int(float(sample.get("edge_id", -1)))
        except (TypeError, ValueError):
            continue
        if width_px < config.min_width_px or point is None or not 0 <= edge_id < len(graph_edges):
            continue
        src_idx, dst_idx = graph_edges[edge_id]
        start, end = nodes[int(src_idx)], nodes[int(dst_idx)]
        vector = end - start
        length_sq = float(np.dot(vector, vector))
        if length_sq <= 1e-6:
            continue
        fraction = float(np.clip(np.dot(point - start, vector) / length_sq, 0.0, 1.0))
        samples_by_edge.setdefault(edge_id, []).append((fraction, width_px))
        sample_rows_by_edge.setdefault(edge_id, []).append(sample)

    widths_by_edge: dict[int, float] = {}
    grades_by_edge: dict[int, str] = {}
    reliable_widths = np.zeros(len(graph_edges), dtype=bool)
    for index, row in enumerate(edge_widths):
        try:
            edge_id = int(float(row.get("edge_id", index)))
        except (TypeError, ValueError):
            continue
        if not 0 <= edge_id < len(graph_edges):
            continue
        width_px = _positive_float(row, "width_px", "optimized_width_px")
        widths_by_edge[edge_id] = width_px
        source = str(row.get("source", row.get("optimized_width_source", "")) or "")
        quality = str(row.get("quality_grade", row.get("optimized_quality_grade", "")) or "")
        grades_by_edge[edge_id] = quality.upper()
        reliable_widths[edge_id] = bool(
            width_px >= config.min_width_px
            and "tile_fallback" not in source
            and "global" not in source
            and quality.upper() != "C"
        )
        if edge_metadata and edge_id < len(edge_metadata):
            # A gap was added precisely because no source pavement existed.
            # Any apparent normal measurement there is not independent width
            # evidence; the gap must inherit its attached road's width.
            if str(edge_metadata[edge_id].get("source", "")) in {"auto_added_gap", "review_added_candidate"}:
                reliable_widths[edge_id] = False
    initial_widths = np.asarray(
        [widths_by_edge.get(edge_id, 0.0) for edge_id in range(len(graph_edges))],
        dtype=np.float32,
    )
    for edge_id, rows in samples_by_edge.items():
        reliable_rows = [
            width for (_, width), row in zip(rows, sample_rows_by_edge.get(edge_id, []))
            if str(row.get("quality_grade", "A") or "A").upper() != "C"
        ]
        if reliable_rows:
            initial_widths[edge_id] = float(np.median(reliable_rows))
            reliable_widths[edge_id] = True
    resolved_widths, fallback_metadata = _resolve_missing_widths(
        nodes, graph_edges, initial_widths, reliable_widths, reference, config, edge_metadata
    )
    if config.regular_surface:
        support_widths, regularized_count = _regular_corridor_widths(
            nodes, graph_edges, resolved_widths, config, reliable_widths, edge_metadata
        )
        chain_width_metadata = {
            "road_chain_count": int(len(np.unique(_edge_chain_ids(graph_edges, len(nodes))))),
            "chain_width_regularized_edge_count": int(regularized_count),
            "chain_width_max_deviation_ratio": 0.0,
            "regular_surface_chain_width_count": int(regularized_count),
            "regular_corridor_cosine": float(config.regular_corridor_cosine),
            "regular_corridor_width_ratio": float(config.regular_corridor_width_ratio),
        }
        resolved_widths = support_widths.copy()
    else:
        support_widths, chain_width_metadata = _regularize_chain_widths(
            resolved_widths,
            graph_edges,
            len(nodes),
            config.chain_width_max_deviation_ratio,
        )
    resolved_widths, junction_width_spike_count = _cap_junction_nearby_width_spikes(
        nodes, graph_edges, resolved_widths, config
    )
    if config.regular_surface:
        support_widths = resolved_widths.copy()
    edge_corridors: list[np.ndarray | None] = [None] * len(graph_edges)
    profiled_edge_count = 0
    fallback_edge_count = 0
    for edge_id, (src_idx, dst_idx) in enumerate(graph_edges.tolist()):
        fallback_width = float(resolved_widths[edge_id])
        samples = [] if config.regular_surface else sorted(samples_by_edge.get(edge_id, []))
        if samples:
            profiled_edge_count += 1
            merged: list[tuple[float, float]] = []
            for fraction, road_width in samples:
                if merged and abs(fraction - merged[-1][0]) <= 1e-4:
                    merged[-1] = (fraction, float(np.median([merged[-1][1], road_width])))
                else:
                    merged.append((fraction, road_width))
            anchors = [(0.0, merged[0][1]), *merged, (1.0, merged[-1][1])]
        elif fallback_width >= config.min_width_px:
            fallback_edge_count += 1
            anchors = [(0.0, fallback_width), (1.0, fallback_width)]
        else:
            continue
        boundary_rows = [
            row for row in sample_rows_by_edge.get(edge_id, [])
            if str(row.get("quality_grade", "A") or "A").upper() != "C"
        ]
        if config.preserve_reference_surface and not config.regular_surface and not boundary_rows and grades_by_edge.get(edge_id) == "C":
            continue
        edge_surface = np.zeros(shape, dtype=np.uint8)
        if config.regular_surface:
            _draw_buffer_edge(
                edge_surface,
                nodes[int(src_idx)],
                nodes[int(dst_idx)],
                fallback_width,
                config.width_scale,
            )
        else:
            _draw_asymmetric_edge(
                edge_surface,
                nodes[int(src_idx)],
                nodes[int(dst_idx)],
                boundary_rows,
                fallback_width,
                probability_gradient,
                config,
            )
        surface |= edge_surface
        edge_corridors[edge_id] = edge_surface

    regular_join_count = 0
    regular_junction_join_count = 0
    if config.regular_surface:
        regular_join_count = _fill_regular_chain_joins(
            surface, nodes, graph_edges, resolved_widths, config.width_scale
        )
        regular_junction_join_count = _fill_regular_junction_joins(
            surface, nodes, graph_edges, resolved_widths, config.width_scale, config.junction_min_degree
        )
    width_corridors = surface.copy()
    support_corridors = np.zeros(shape, dtype=np.uint8)
    for edge_id, (src_idx, dst_idx) in enumerate(graph_edges.tolist()):
        support_width = float(support_widths[edge_id])
        if config.regular_surface:
            _draw_buffer_edge(support_corridors, nodes[int(src_idx)], nodes[int(dst_idx)], support_width, config.width_scale)
        else:
            _draw_variable_edge(
                support_corridors,
                nodes[int(src_idx)],
                nodes[int(dst_idx)],
                [(0.0, support_width), (1.0, support_width)],
                config.width_scale,
            )
    if config.regular_surface:
        _fill_regular_chain_joins(
            support_corridors, nodes, graph_edges, support_widths, config.width_scale
        )
        _fill_regular_junction_joins(
            support_corridors, nodes, graph_edges, support_widths, config.width_scale, config.junction_min_degree
        )
    authoritative_corridors = width_corridors.copy()
    for edge_id, (src_idx, dst_idx) in enumerate(graph_edges.tolist()):
        if edge_corridors[edge_id] is not None:
            continue
        if config.regular_surface:
            _draw_buffer_edge(
                authoritative_corridors,
                nodes[int(src_idx)],
                nodes[int(dst_idx)],
                float(resolved_widths[edge_id]),
                config.width_scale,
            )
        else:
            _draw_variable_edge(
                authoritative_corridors,
                nodes[int(src_idx)],
                nodes[int(dst_idx)],
                [
                    (0.0, float(resolved_widths[edge_id])),
                    (1.0, float(resolved_widths[edge_id])),
                ],
                config.width_scale,
            )
    if config.regular_surface:
        _fill_regular_chain_joins(
            authoritative_corridors, nodes, graph_edges, resolved_widths, config.width_scale
        )
        _fill_regular_junction_joins(
            authoritative_corridors, nodes, graph_edges, resolved_widths, config.width_scale, config.junction_min_degree
        )
    junction_repairs = np.zeros(shape, dtype=np.uint8)
    parallel_repairs = np.zeros(shape, dtype=np.uint8)
    local_gap_repairs = np.zeros(shape, dtype=np.uint8)
    regular_hole_repairs = np.zeros(shape, dtype=np.uint8)
    junction_core = np.zeros(shape, dtype=np.uint8) if config.regular_surface else _junction_core_mask(
        shape, nodes, graph_edges, resolved_widths, config
    )
    if config.regular_surface:
        # Regular mode is intentionally width-driven.  Do not reintroduce the
        # segmented SAM surface or circular junction envelopes.
        original_reference = reference.copy()
        unsupported_removed = np.zeros(shape, dtype=np.uint8)
        unsupported_metadata = {
            "centerline_support_margin_px": 0,
            "unsupported_surface_removed_px": 0,
            "unsupported_surface_component_count": 0,
        }
        continuity_repairs = np.zeros(shape, dtype=np.uint8)
        ab_missing_repairs = np.zeros(shape, dtype=np.uint8)
        c_continuity_repairs = np.zeros(shape, dtype=np.uint8)
        ab_missing_edge_count = 0
        c_continuity_ids = set()
        selective_replace_zone = np.zeros(shape, dtype=np.uint8)
        junction_envelope_count = 0
        parallel_pair_count = _fill_parallel_road_gaps(
            parallel_repairs, nodes, graph_edges, resolved_widths, config
        )
        surface |= parallel_repairs
        # In regular Buffer mode a divided-road median is part of the same
        # pavement envelope. Close only narrow internal holes in the union.
        if np.any(surface):
            regular_hole_repairs = _enclosed_sliver_holes(
                surface,
                np.ones_like(surface, dtype=np.uint8),
                max_half_width_px=max(8.0, float(np.median(resolved_widths)) * 1.5),
                max_area_px=max(12000, int(surface.size * 0.05)),
            )
            surface |= regular_hole_repairs
        # The edge ribbons already meet at shared nodes, producing a compact
        # polygonal intersection rather than a large multi-width disk.
    elif config.preserve_reference_surface and np.any(reference):
        # Keep the probability-defined SAM-MLoRA boundary. Width geometry is only
        # allowed to repair topology, not replace the complete road surface.
        original_reference = reference.copy()
        reference, unsupported_removed, unsupported_metadata = _prune_unsupported_reference(
            reference, support_corridors, config
        )
        surface, selective_replace_zone = _selective_boundary_update(
            reference, width_corridors, probability, probability_gradient, config, junction_core
        )
        ab_missing_repairs = np.zeros(shape, dtype=np.uint8)
        ab_missing_edge_count = 0
        coverage_limit = float(np.clip(config.missing_centerline_coverage, 0.0, 1.0))
        for edge_id, (src_idx, dst_idx) in enumerate(graph_edges.tolist()):
            if grades_by_edge.get(edge_id) not in {"A", "B"}:
                continue
            corridor = edge_corridors[edge_id]
            if corridor is None:
                continue
            coverage = _segment_reference_coverage(
                reference, nodes[int(src_idx)], nodes[int(dst_idx)]
            )
            if coverage < coverage_limit:
                ab_missing_repairs |= corridor
                ab_missing_edge_count += 1
        surface |= ab_missing_repairs

        c_continuity_repairs = np.zeros(shape, dtype=np.uint8)
        c_continuity_ids = _c_continuity_edge_ids(
            nodes, graph_edges, grades_by_edge, surface, config
        )
        for edge_id in c_continuity_ids:
            src_idx, dst_idx = graph_edges[edge_id]
            _draw_variable_edge(
                c_continuity_repairs,
                nodes[int(src_idx)],
                nodes[int(dst_idx)],
                [(0.0, float(resolved_widths[edge_id])), (1.0, float(resolved_widths[edge_id]))],
                config.width_scale,
            )
        surface |= c_continuity_repairs
        continuity_repairs = np.zeros(shape, dtype=np.uint8)
        if config.continuity_close_kernel > 1:
            size = max(3, int(config.continuity_close_kernel) | 1)
            continuity_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            closed_reference = cv2.morphologyEx(reference, cv2.MORPH_CLOSE, continuity_kernel)
            continuity_repairs = (
                (closed_reference > 0)
                & (support_corridors > 0)
                & (reference == 0)
            ).astype(np.uint8)
            surface |= continuity_repairs
        junction_envelope_count = _fill_junction_envelopes(
            junction_repairs, nodes, graph_edges, resolved_widths, reference, probability, config
        )
        # Junction geometry may only extend a few pixels beyond the segmented
        # surface. This keeps the real intersection outline instead of replacing
        # it with a width-derived convex disk.
        junction_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        junction_support = cv2.dilate(reference, junction_kernel) > 0
        junction_repairs &= junction_support.astype(np.uint8)
        surface |= junction_repairs

        parallel_pair_count = _fill_parallel_road_gaps(
            parallel_repairs, nodes, graph_edges, resolved_widths, config
        )
        # Median gaps between long, parallel divided roads are an intentional
        # structural merge and may lie beyond the small junction support band.
        surface |= parallel_repairs
        centerline = np.zeros(shape, dtype=np.uint8)
        for src_idx, dst_idx in graph_edges.tolist():
            start = nodes[int(src_idx)]
            end = nodes[int(dst_idx)]
            cv2.line(
                centerline,
                (int(round(float(start[1]))), int(round(float(start[0])))),
                (int(round(float(end[1]))), int(round(float(end[0])))),
                1, 1, cv2.LINE_8,
            )
        uncovered = (centerline > 0) & (reference == 0)
        if np.any(uncovered):
            radius = max(2, int(np.ceil(0.5 * float(np.percentile(support_widths, 75)) * config.width_scale)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            local_gap_zone = cv2.dilate(uncovered.astype(np.uint8), kernel) > 0
            distance_to_reference = cv2.distanceTransform((reference == 0).astype(np.uint8), cv2.DIST_L2, 5)
            local_support = distance_to_reference <= max(1, int(config.continuity_max_gap_px))
            local_gap_repairs = ((support_corridors > 0) & local_gap_zone & local_support).astype(np.uint8)
            surface |= local_gap_repairs
    else:
        original_reference = reference
        unsupported_removed = np.zeros(shape, dtype=np.uint8)
        unsupported_metadata = {
            "centerline_support_margin_px": 0,
            "unsupported_surface_removed_px": 0,
            "unsupported_surface_component_count": 0,
        }
        continuity_repairs = np.zeros(shape, dtype=np.uint8)
        ab_missing_repairs = np.zeros(shape, dtype=np.uint8)
        c_continuity_repairs = np.zeros(shape, dtype=np.uint8)
        ab_missing_edge_count = 0
        c_continuity_ids = set()
        selective_replace_zone = np.zeros(shape, dtype=np.uint8)
        junction_envelope_count = _fill_junction_envelopes(
            surface, nodes, graph_edges, resolved_widths, reference, probability, config
        )
        parallel_pair_count = _fill_parallel_road_gaps(
            surface, nodes, graph_edges, resolved_widths, config
        )

    if config.close_kernel > 1 and np.any(surface):
        size = max(3, int(config.close_kernel) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, kernel)
    surface = _remove_small_components(surface, config.min_component_area_px)
    surface = _simplify_contours(surface, config.contour_simplify_tolerance_px)

    # Enforce a cross-section-complete road envelope. Natural SAM boundaries may
    # deviate slightly from the measured widths, but unsupported lateral branches
    # cannot survive merely because they are connected to a valid road surface.
    margin = max(0, int(config.centerline_support_margin_px))
    if margin:
        natural_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
        )
        allowed_surface = cv2.dilate(authoritative_corridors, natural_kernel) > 0
    else:
        allowed_surface = authoritative_corridors > 0
    allowed_surface |= junction_core > 0
    if np.any(parallel_repairs):
        parallel_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        allowed_surface |= cv2.dilate(parallel_repairs, parallel_kernel) > 0
    if np.any(regular_hole_repairs):
        allowed_surface |= regular_hole_repairs > 0
    unsupported_final = ((surface > 0) & ~allowed_surface).astype(np.uint8)
    surface &= allowed_surface.astype(np.uint8)
    # Fill only genuinely enclosed slivers inside the admissible road envelope.
    # Open background and city blocks remain outside the width envelope.
    corridor_hole_fill = _enclosed_sliver_holes(
        surface,
        allowed_surface,
        config.max_enclosed_hole_half_width_px,
        config.max_enclosed_hole_area_px,
    )
    surface |= corridor_hole_fill
    surface |= junction_repairs
    surface |= parallel_repairs

    # Suppress raster stair steps and short SAM boundary spikes, while keeping
    # the regularized result inside the centerline-supported road envelope.
    pre_smooth_surface = surface.copy()
    surface = _smooth_surface_boundary(surface, config.boundary_smooth_sigma_px)
    surface &= allowed_surface.astype(np.uint8)
    surface |= junction_repairs
    surface |= parallel_repairs
    boundary_smooth_added = ((surface > 0) & (pre_smooth_surface == 0)).astype(np.uint8)
    boundary_smooth_removed = ((pre_smooth_surface > 0) & (surface == 0)).astype(np.uint8)

    # Hard final contract: every retained centerline edge must lie inside a
    # road-width surface, regardless of confidence grade or earlier pruning.
    forced_coverage_repairs = np.zeros(shape, dtype=np.uint8)
    forced_coverage_edge_ids: list[int] = []
    for edge_id, (src_idx, dst_idx) in enumerate(graph_edges.tolist()):
        start = nodes[int(src_idx)]
        end = nodes[int(dst_idx)]
        edge_line = _edge_centerline_mask(shape, start, end)
        if not np.any((edge_line > 0) & (surface == 0)):
            continue
        corridor = edge_corridors[edge_id]
        if corridor is None:
            corridor = np.zeros(shape, dtype=np.uint8)
            width = max(float(resolved_widths[edge_id]), config.minimum_render_width_px)
            _draw_variable_edge(
                corridor, start, end, [(0.0, width), (1.0, width)], config.width_scale
            )
        forced_coverage_repairs |= corridor
        forced_coverage_edge_ids.append(edge_id)
    surface |= forced_coverage_repairs

    final_centerline = np.zeros(shape, dtype=np.uint8)
    for src_idx, dst_idx in graph_edges.tolist():
        final_centerline |= _edge_centerline_mask(
            shape, nodes[int(src_idx)], nodes[int(dst_idx)]
        )
    uncovered_centerline = ((final_centerline > 0) & (surface == 0)).astype(np.uint8)
    if np.any(uncovered_centerline):
        # This should only be reachable for degenerate rasterization cases.
        emergency = cv2.dilate(
            uncovered_centerline,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        forced_coverage_repairs |= emergency
        surface |= emergency
        uncovered_centerline = ((final_centerline > 0) & (surface == 0)).astype(np.uint8)

    added = ((surface > 0) & (original_reference == 0)).astype(np.uint8)
    removed = ((original_reference > 0) & (surface == 0)).astype(np.uint8)
    return WidthSurfaceResult(
        surface=surface,
        added=added,
        removed=removed,
        metadata={
            "status": "ok" if np.any(surface) else "empty_width_surface",
            "profiled_edge_count": profiled_edge_count,
            "fallback_edge_count": fallback_edge_count,
            **fallback_metadata,
            **chain_width_metadata,
            "junction_envelope_count": junction_envelope_count,
            "parallel_road_pair_fill_count": parallel_pair_count,
            "positive_width_sample_count": sum(len(rows) for rows in samples_by_edge.values()),
            "reference_surface_px": int(np.count_nonzero(original_reference)),
            "width_constrained_surface_px": int(np.count_nonzero(surface)),
            "width_adjusted_added_px": int(np.count_nonzero(added)),
            "width_adjusted_removed_px": int(np.count_nonzero(removed)),
            "width_scale": config.width_scale,
            "resolved_widths_px": [float(value) for value in resolved_widths.tolist()],
            "junction_nearby_width_spike_clamped_edge_count": int(junction_width_spike_count),
            "surface_policy": "regular_buffer_surface" if config.regular_surface else ("preserve_sammolra_with_structural_repairs" if config.preserve_reference_surface else "width_reconstruction"),
            "regular_surface": bool(config.regular_surface),
            "regular_chain_join_count": int(regular_join_count),
            "regular_junction_join_count": int(regular_junction_join_count),
            "regular_hole_repair_px": int(np.count_nonzero(regular_hole_repairs)),
            "road_probability_used": probability is not None,
            "ab_missing_surface_edge_count": int(ab_missing_edge_count),
            "ab_missing_surface_added_px": int(np.count_nonzero((ab_missing_repairs > 0) & (original_reference == 0))),
            "c_continuity_edge_count": int(len(c_continuity_ids)),
            "c_continuity_added_px": int(np.count_nonzero((c_continuity_repairs > 0) & (original_reference == 0))),
            "missing_centerline_coverage": float(config.missing_centerline_coverage),
            "final_unsupported_surface_removed_px": int(np.count_nonzero(unsupported_final)),
            "corridor_hole_fill_px": int(np.count_nonzero(corridor_hole_fill)),
            "final_surface_margin_px": margin,
            "max_enclosed_hole_half_width_px": float(config.max_enclosed_hole_half_width_px),
            "max_enclosed_hole_area_px": int(config.max_enclosed_hole_area_px),
            "forced_coverage_edge_count": int(len(forced_coverage_edge_ids)),
            "forced_coverage_added_px": int(np.count_nonzero((forced_coverage_repairs > 0) & (original_reference == 0))),
            "uncovered_centerline_px": int(np.count_nonzero(uncovered_centerline)),
            "selective_boundary_replace_px": int(np.count_nonzero(selective_replace_zone)),
            "boundary_snap_max_px": float(config.boundary_snap_max_px),
            "contour_simplify_tolerance_px": float(config.contour_simplify_tolerance_px),
            "boundary_smooth_sigma_px": float(config.boundary_smooth_sigma_px),
            "boundary_smooth_added_px": int(np.count_nonzero(boundary_smooth_added)),
            "boundary_smooth_removed_px": int(np.count_nonzero(boundary_smooth_removed)),
            "junction_repair_px": int(np.count_nonzero((junction_repairs > 0) & (reference == 0))) if config.preserve_reference_surface else 0,
            "parallel_gap_repair_px": int(np.count_nonzero((parallel_repairs > 0) & (reference == 0))) if config.preserve_reference_surface else 0,
            "local_gap_repair_px": int(np.count_nonzero((local_gap_repairs > 0) & (reference == 0))) if config.preserve_reference_surface else 0,
            "continuity_repair_px": int(np.count_nonzero((continuity_repairs > 0) & (original_reference == 0))) if config.preserve_reference_surface else 0,
            "continuity_close_kernel": int(config.continuity_close_kernel),
            "continuity_max_gap_px": int(config.continuity_max_gap_px),
            **unsupported_metadata,
        },
    )
