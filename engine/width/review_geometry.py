from __future__ import annotations

import json

import cv2
import numpy as np


def accepted_surface_region_polylines(
    stem: str,
    surface_only: np.ndarray | None,
    candidates: list[dict],
    decisions: dict[tuple[str, str, str], str],
) -> dict[str, list[list[tuple[float, float]]]]:
    """Skeletonize accepted surface-only regions that have no prepared candidate."""
    if surface_only is None:
        return {}

    candidate_region_ids = {
        str(row.get("region_id", ""))
        for row in candidates
        if str(row.get("region_id", ""))
    }
    _, labels = cv2.connectedComponents((surface_only > 0).astype(np.uint8), connectivity=8)
    generated: dict[str, list[list[tuple[float, float]]]] = {}
    for value in np.unique(labels):
        if not value:
            continue
        region_id = str(int(value))
        if region_id in candidate_region_ids:
            continue
        if decisions.get((stem, "surface_only_region", region_id), "") != "accept":
            continue

        rows, cols = np.nonzero(labels == int(value))
        if len(rows) < 2:
            continue
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        component = (labels[y0:y1, x0:x1] == int(value)).astype(np.uint8)

        # Import lazily so normal editor/workflow startup does not load the
        # inference stack unless an accepted region actually needs a skeleton.
        from molra_centerline_width import component_candidate_centerlines

        records = component_candidate_centerlines(
            component,
            x0,
            y0,
            sample_step_px=16.0,
            min_branch_length_px=8.0,
            simplify_epsilon_px=1.5,
            min_half_width_px=0.5,
        )
        if not records:
            records = component_candidate_centerlines(
                component,
                x0,
                y0,
                sample_step_px=16.0,
                min_branch_length_px=1.0,
                simplify_epsilon_px=1.5,
                min_half_width_px=0.0,
            )
        polylines = []
        for record in records:
            try:
                points = json.loads(record.get("polyline_points_json", ""))
                points = [(float(row), float(col)) for row, col in points]
            except (json.JSONDecodeError, TypeError, ValueError):
                points = []
            if len(points) >= 2:
                polylines.append(points)
        if polylines:
            generated[region_id] = polylines
    return generated
