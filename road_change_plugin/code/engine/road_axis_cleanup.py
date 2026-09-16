"""Evidence-constrained high-frequency cleanup of metric, junction-split chains."""
import numpy as np
from scipy.signal import savgol_filter
from shapely import STRtree
from shapely.geometry import LineString
from .road_geometry import _RegionalRoadSeed


def remove_exact_duplicates(roads):
    """Remove only identical linework (including reversal), retaining provenance."""
    groups = {}
    for road in roads:
        key = LineString(road.points).normalize().wkb
        groups.setdefault(key, []).append(road)
    result = []
    for group in groups.values():
        first = group[0]
        result.append(first if len(group)==1 else _RegionalRoadSeed(
            first.points, float(np.mean([r.width_m for r in group])),
            tuple(sorted({i for r in group for i in r.source_ids})), first.geometry_kind))
    return result


def _axis_candidate(road):
    """Detect repeated lateral sign changes, not sustained signed curvature.

    The 25 m quadratic window preserves the low-frequency axis. Corrections
    require at least two oscillations with half-period <=12.5 m, and retain
    the existing 2 m maximum displacement. No smoothing of arbitrary curves.
    """
    original = np.asarray(road.points, dtype=float)
    line = LineString(original)
    if line.length < 25 or np.array_equal(original[0], original[-1]):
        return None
    distances = np.arange(0., line.length, 1.)
    distances = np.r_[distances, line.length]
    points = np.asarray([line.interpolate(s).coords[0] for s in distances])
    fitted = savgol_filter(points, 25, 2, axis=0, mode='interp')
    tangent = np.gradient(fitted, axis=0)
    normal = np.column_stack([-tangent[:,1], tangent[:,0]])
    normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
    residual = np.sum((points-fitted)*normal, axis=1)
    significant = np.flatnonzero(np.abs(residual) >= .2)
    if len(significant) < 5:
        return None
    changes = significant[1:][np.sign(residual[significant[1:]]) != np.sign(residual[significant[:-1]])]
    if len(changes) < 4:
        return None
    short = np.diff(distances[changes]) <= 12.5
    mask = np.zeros(len(points), dtype=float)
    run = 0
    for i, keep in enumerate(np.r_[short, False]):
        if keep:
            run += 1
            continue
        if run >= 3:
            lo, hi = changes[i-run], changes[i]
            # Taper local corrections into untouched geometry; true junction
            # endpoints belong to separate chains and remain exactly fixed.
            edge_distance = np.minimum(distances-distances[lo], distances[hi]-distances)
            mask = np.maximum(mask, np.clip(edge_distance/3., 0., 1.))
        run = 0
    if not mask.any():
        return None
    displacement = mask[:,None]*(fitted-points)
    if np.max(np.linalg.norm(displacement,axis=1)) > 2.:
        return None
    corrected = points + displacement
    corrected[[0,-1]] = original[[0,-1]]
    candidate = LineString(corrected)
    if not candidate.is_simple or line.hausdorff_distance(candidate) > 2.:
        return None
    return candidate


def correct_oscillating_chains(roads, evidence):
    """Only supported internal wobble moves; endpoints, junctions, CRS stay fixed."""
    if evidence is None or not roads:
        return roads, 0
    lines = [LineString(r.points) for r in roads]
    tree = STRtree(lines)
    changed = {}
    result = list(roads)
    for i, road in enumerate(roads):
        candidate = _axis_candidate(road)
        if candidate is None:
            continue
        # Never introduce a new crossing or detach an existing interior contact.
        neighbours = set(map(int, tree.query(candidate))) | set(changed)
        safe = True
        for j in neighbours:
            if j == i:
                continue
            other = changed.get(j, lines[j])
            old_contact = lines[i].intersection(other)
            new_contact = candidate.intersection(other)
            if (not new_contact.difference(old_contact.buffer(1e-5)).is_empty
                    or not old_contact.difference(new_contact.buffer(1e-5)).is_empty):
                safe = False
                break
        if not safe:
            continue
        before = evidence.measure(lines[i])
        after = evidence.measure(candidate)
        if (after['joint_support'] < max(.85, before['joint_support'])
                or after['unsupported_run_m'] > before['unsupported_run_m'] + 1e-8):
            continue
        changed[i] = candidate
        result[i] = _RegionalRoadSeed(np.asarray(candidate.coords), road.width_m,
                                      road.source_ids, 'low_frequency_axis')
    return result, len(changed)
