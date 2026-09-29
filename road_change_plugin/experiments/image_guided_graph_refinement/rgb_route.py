"""Direction-constrained routes using RGB ribbons and positive cached evidence."""
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

from evidence import stations, longest_hole
from ribbon_diagnostic import raw_ribbon_profile
from graph import build, candidates
from refine import safe_gap


def ribbon_cost_features(gray, fractions):
    """Vectorized common RGB ribbon features for medial fitting and gap search."""
    finite = np.isfinite(gray).all(axis=-1)
    filled = np.nan_to_num(gray)
    inside = filled[..., (fractions >= .2) & (fractions <= .8)]
    median = np.median(inside, axis=-1)
    dl = median-np.median(filled[..., fractions < -.05], axis=-1)
    dr = median-np.median(filled[..., fractions > 1.05], axis=-1)
    spread = np.quantile(inside, .8, axis=-1)-np.quantile(inside, .2, axis=-1)
    contrast = np.minimum(abs(dl), abs(dr))
    score = np.clip(contrast/.4, 0, 1)*np.exp(-spread/.25)*(dl*dr > 0)*finite
    return score, finite, contrast, spread


def ribbon_evidence(line, width, evidence):
    _, xy, _, normals = stations(line, 2.5)
    fractions = np.linspace(-.5, 1.5, 41)
    probe = xy[:, None, :]+normals[:, None, :]*(fractions[None, :, None]-.5)*width
    gray = evidence.image.sample(evidence.image.gray, probe.reshape(-1, 2)).reshape(len(xy), -1)
    valid = np.isfinite(gray).all(axis=1)
    raw = raw_ribbon_profile(np.nan_to_num(gray), fractions)
    supported = np.array(raw['supported_sections']) & valid
    p, m, surface = evidence.values(xy)
    # Absence of SAM/MOLRA does not subtract from RGB support. Derived surfaces
    # are reported only; they never create independent support.
    positive = (p >= .65) | (m >= .65)
    combined = supported | (positive & valid)
    return dict(raw_rgb_support=float(supported.mean()), valid_fraction=float(valid.mean()),
                continuous_support=float(combined.mean()), maximum_unsupported_m=longest_hole(combined, line.length),
                sam_positive_mean=float(np.nanmean(p)) if np.isfinite(p).any() else None,
                molra_positive_mean=float(np.nanmean(m)) if np.isfinite(m).any() else None,
                auxiliary_surface_support=float(surface.mean()),
                flank_contrast=raw['flank_contrast_median'], interior_spread=raw['interior_spread_median'])


def constrained_route(candidate, evidence):
    """Monotonic corridor DP; hard tangent constraints apply inside the search."""
    a, b = np.asarray(candidate.axis.coords)[[0, -1]]
    length = np.linalg.norm(b-a)
    tangent = (b-a)/length
    normal = np.array([-tangent[1], tangent[0]])
    heading_a, heading_b = np.asarray(candidate.geometry['approach_directions'])
    count = max(5, int(np.ceil(length/1.5))+1)
    along = np.linspace(0, length, count)
    step = along[1]-along[0]
    half = min(12., max(3., candidate.width*.5))
    offsets = np.arange(-np.ceil(half/.75), np.ceil(half/.75)+1)*.75
    zero = int(np.argmin(abs(offsets)))
    xy = a+along[:, None, None]*tangent+offsets[None, :, None]*normal
    fractions = np.linspace(-.5, 1.5, 41)
    probe = xy[:, :, None, :]+normal*(fractions[None, None, :, None]-.5)*candidate.width
    gray = evidence.image.sample(evidence.image.gray, probe.reshape(-1, 2)).reshape(count, len(offsets), -1)
    score, finite, _, _ = ribbon_cost_features(gray, fractions)
    p, m, _ = evidence.values(xy.reshape(-1, 2))
    score += .15*np.nan_to_num(p).reshape(score.shape)+.15*np.nan_to_num(m).reshape(score.shape)
    cost = 1.3-score+.04*(offsets/half)**2
    cost[~finite] = np.inf
    dp = np.full(cost.shape, np.inf)
    back = np.full(cost.shape, -1, int)
    dp[0, zero] = cost[0, zero]
    for i in range(1, count):
        for j in range(len(offsets)):
            for previous in range(max(0, j-2), min(len(offsets), j+3)):
                travel = tangent*step+normal*(offsets[j]-offsets[previous])
                travel /= np.linalg.norm(travel)
                if ((along[i] <= min(10., length*.3) and travel@heading_a < .85) or
                    (length-along[i-1] <= min(10., length*.3) and travel@heading_b < .85)):
                    continue
                proposed = dp[i-1, previous]+cost[i, j]+.12*(j-previous)**2
                if proposed < dp[i, j]:
                    dp[i, j], back[i, j] = proposed, previous
    if not np.isfinite(dp[-1, zero]):
        return None
    selected = [zero]
    for i in range(count-1, 0, -1):
        selected.append(int(back[i, selected[-1]]))
    coords = xy[np.arange(count), selected[::-1]]
    coords[0], coords[-1] = a, b
    return LineString(coords).simplify(.1)


def connect_gaps(frame, evidence, roi, config):
    local = frame[frame.intersects(roi.buffer(160))].reset_index(drop=True)
    edges = build(local)
    audit, additions, used = [], [], set()
    for c in candidates(edges, roi, config, ('gap',)):
        route = constrained_route(c, evidence)
        score = ribbon_evidence(route, c.width, evidence) if route is not None else None
        safe = route is not None and safe_gap(route, edges, c)
        decision = bool(safe and score['raw_rgb_support'] >= .65 and score['continuous_support'] >= .8
                        and score['maximum_unsupported_m'] <= min(10., route.length*.25)
                        and not (set(c.ids) & used))
        audit.append(dict(candidate_id=c.key, action='connect' if decision else 'retain_gap',
            candidate_wkt=c.axis.wkt, new_wkt=route.wkt if decision else None, evidence=score,
            direction_constrained_during_search=True, crossing_guard_passed=safe,
            reason='continuous_direction_constrained_rgb_route' if decision else 'route_failed_independent_evidence_or_topology_guard'))
        if decision:
            row = local.iloc[edges[c.ids[0]].sources[0]].to_dict()
            row.update(geometry=route, repair_role='rgb_gap_connection')
            additions.append(row)
            used.update(c.ids)
    if additions:
        import pandas as pd
        frame = gpd.GeoDataFrame(pd.concat([frame, gpd.GeoDataFrame(additions, crs=frame.crs)], ignore_index=True), crs=frame.crs)
    return frame, audit
