"""Detect missing interior axes between distant boundary-like lines.

This diagnostic deliberately does not mutate the graph: connecting a replacement
axis to the existing branches is a junction-rewire problem outside phase one.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely import points, distance, union_all
from shapely.geometry import LineString
from shapely.strtree import STRtree

from evidence import stations
from graph import Candidate


def raw_ribbon_profile(gray, fractions):
    """A coherent interior plateau with contrasting flanks, independent of SAM.

    This is a structural cue, not a road/non-road semantic model. It is recorded
    as review even when strong, since yards and roofs can produce similar cues.
    """
    gray = np.asarray(gray, float)
    fractions = np.asarray(fractions)
    left = np.nanmedian(gray[:, fractions < -.05], axis=1)
    right = np.nanmedian(gray[:, fractions > 1.05], axis=1)
    inside = gray[:, (fractions >= .2) & (fractions <= .8)]
    middle = np.nanmedian(inside, axis=1)
    spread = np.nanquantile(inside, .8, axis=1)-np.nanquantile(inside, .2, axis=1)
    dl, dr = middle-left, middle-right
    valid = np.isfinite(gray).all(axis=1)
    plateau = valid & (dl*dr > 0) & (np.minimum(abs(dl), abs(dr)) >= .18) & (spread <= .2)
    sign = np.where(middle >= (left+right)/2, 1., -1.)
    weights = np.maximum(sign[:, None]*(gray-(left+right)[:, None]/2)-.1, 0)
    weights[:, (fractions < .05) | (fractions > .95)] = 0
    weights = np.nan_to_num(weights)
    centers = np.divide(weights@fractions, weights.sum(axis=1),
                        out=np.full(len(gray), .5), where=weights.sum(axis=1) > 1e-9)
    return dict(raw_ribbon_support_fraction=float(plateau.mean()),
                raw_valid_fraction=float(valid.mean()),
                flank_contrast_median=float(np.nanmedian(np.minimum(abs(dl), abs(dr)))),
                interior_spread_median=float(np.nanmedian(spread)),
                center_fractions=centers.tolist(), supported_sections=plateau.tolist())


def diagnose(edges, evidence, roi, max_separation=45.):
    """Wider, non-destructive search supplements narrow duplicate candidates."""
    tree = STRtree([e.axis for e in edges])
    network = union_all([e.axis for e in edges])
    reports = []
    for i, edge in enumerate(edges):
        axis = edge.axis
        if axis.length < 12 or not roi.covers(axis):
            continue
        _, xy, tangent, _ = stations(axis)
        for raw in tree.query(axis, predicate="dwithin", distance=max_separation):
            j = int(raw)
            other = edges[j]
            if (j == i or other.level != edge.level or other.axis.length < axis.length or
                    (other.axis.length == axis.length and j < i) or axis.distance(other.axis) < 1.5):
                continue
            target = np.array([other.axis.interpolate(other.axis.project(p)).coords[0] for p in points(xy)])
            delta = target-xy
            separation = np.linalg.norm(delta, axis=1)
            # Reject clamped projections and crossings: paired samples must be
            # approximately perpendicular to the first road and move along both.
            normality = abs(np.sum(delta*tangent, axis=1))/np.maximum(separation, 1e-6)
            motion = np.gradient(target, axis=0)
            cosine = abs(np.sum(motion*tangent, axis=1))/np.maximum(np.linalg.norm(motion, axis=1), 1e-6)
            usable = (separation >= 6) & (separation <= max_separation) & (normality <= .3) & (cosine >= .94)
            if usable.mean() < .65:
                continue
            # Inspect only the contiguous geometrically matched section.
            indices = np.flatnonzero(usable)
            indices = max(np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1), key=len)
            if len(indices) < 5:
                continue
            a, b = xy[indices], target[indices]
            matched = LineString(a)
            if matched.length < 10:
                continue
            fractions = np.linspace(-.5, 1.5, 61)
            probes = a[:, None, :] + (b-a)[:, None, :]*fractions[None, :, None]
            gray = evidence.image.sample(evidence.image.gray, probes.reshape(-1, 2)).reshape(len(a), -1)
            if not np.isfinite(gray).all(axis=1).any():
                continue
            structure = raw_ribbon_profile(gray, fractions)
            if structure["raw_ribbon_support_fraction"] < .5:
                continue
            centers = np.asarray(structure["center_fractions"])
            center_xy = a+(b-a)*centers[:, None]
            center_xy = gaussian_filter1d(center_xy, 1., axis=0)
            proposal = LineString(center_xy)
            if not roi.covers(proposal) or not proposal.is_simple:
                continue
            _, center_points, _, _ = stations(proposal)
            axis_distance = distance(points(center_points), network)
            if float(np.median(axis_distance)) < 4:
                continue  # an interior axis already exists
            p, m, s = evidence.values(center_points)
            key = Candidate("boundary_axis", (i, j), proposal, 0., {}).key
            score = structure["raw_ribbon_support_fraction"]
            reports.append(dict(candidate_id=key, type="boundary_axis", mode="rgb_axis_diagnostic", round=1,
                source_ids=sorted(set(edge.sources+other.sources)),
                geometry_scores=dict(separation_m=float(np.median(separation[indices])),
                    overlap_m=matched.length, minimum_existing_axis_distance_m=float(np.min(axis_distance)),
                    mean_existing_axis_distance_m=float(np.mean(axis_distance))),
                evidence=dict(**structure, probability_mean=float(np.nanmean(p)) if np.isfinite(p).any() else None,
                    probability_valid_fraction=float(np.isfinite(p).mean()),
                    molra_mean=float(np.nanmean(m)) if np.isfinite(m).any() else None,
                    formal_surface_support=float(s.mean()),
                    cross_sections=dict(fractions=fractions.tolist(), gray=gray.tolist(),
                        probability=evidence.values(probes.reshape(-1, 2))[0].reshape(gray.shape).tolist(),
                        stations_xy=a.tolist())),
                evidence_score=score, evidence_decision="single_ribbon_missing_interior_axis_candidate",
                decision="review", reason="rgb_ribbon_suggests_missing_axis_requires_junction_rewire",
                candidate_wkt=matched.wkt, comparison_wkt=other.axis.wkt, result_wkt=proposal.wkt,
                proposal_method="raw_image_interior_contrast_centroid_no_width_measurement",
                auto_applied=False))
    # Suppress overlapping hypotheses for the same missing center section.
    accepted = []
    from shapely import wkt
    for row in sorted(reports, key=lambda r: (-r["evidence_score"], -r["geometry_scores"]["overlap_m"], r["candidate_id"])):
        proposal = wkt.loads(row["result_wkt"])
        if any(proposal.intersection(wkt.loads(r["result_wkt"]).buffer(4)).length > .65*proposal.length for r in accepted):
            continue
        accepted.append(row)
    return accepted
