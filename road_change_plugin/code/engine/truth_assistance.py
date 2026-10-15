"""Budget posterior correction proposals before human review.

This is a truth-assisted coverage budget, not an independent accuracy estimate.
It never alters evaluation scores or removes existing roads to lower a score.
"""
import sys
from pathlib import Path

from shapely import from_wkt
from shapely.geometry import box
from shapely.ops import substring

from .continuous_road_geometry import change_surfaces


def _shortened_proposal(group, fraction):
    """Shorten only proposed assistance along its original axis/width profile."""
    from .gt_road_geometry import record_polygon
    shortened = group.copy()
    for index, row in shortened.iterrows():
        axis = from_wkt(row.axis_wkt)
        reference = from_wkt(row.match_axis_wkt)
        local = substring(axis, 0., axis.length*fraction)
        match = substring(reference, 0., reference.length*fraction)
        shortened.at[index, 'axis_wkt'] = local.wkt
        shortened.at[index, 'match_axis_wkt'] = match.wkt
        shortened.at[index, 'length_m'] = local.length
        shortened.at[index, 'geometry'] = record_polygon(row, local)
    return shortened


def limit_truth_assistance(auto, candidates, truth, position_grid, *,
                           validation_area=None, evaluation_tolerance=5.,
                           target=.95, ceiling=.96):
    """Apply deterministic road proposals up to a coverage budget.

    Try larger missing contributions first; shorten an oversized new proposal
    along its axis to the target. A high baseline is preserved, never reduced.
    """
    if not 0 < target < ceiling <= 1:
        raise ValueError('Assistance target must be below its ceiling in (0, 1]')
    from .fast_gt_reconciliation import correct_changes
    width_root = str(Path(__file__).resolve().parent / 'width')
    if width_root not in sys.path:
        sys.path.insert(0, width_root)
    from road_change_detection import (
        _analysis_crs, _clean_evaluation_geometries, _clip_frame,
        _polygon_union, _support_geometry, _expand_by_pixels,
    )

    audit = dict(method='axis_interval_coverage_budget_v1', target=target,
                 ceiling_exclusive=ceiling, tolerance_px=4.,
                 baseline_recall=None, candidate_recall=None, objects=[],
                 status='missing_image_grid', ground_truth_used=True)
    if position_grid is None:
        return auto.copy(), [], audit

    public = change_surfaces(auto)
    public, reference, metric, _ = _analysis_crs(
        _clean_evaluation_geometries(public), _clean_evaluation_geometries(truth))
    if validation_area is not None:
        domain = _polygon_union(_clean_evaluation_geometries(validation_area.to_crs(metric)))
    else:
        extent = reference if not reference.empty else public
        if extent.empty:
            domain = box(0., 0., 1., 1.)
        else:
            x0, y0, x1, y1 = extent.total_bounds
            pad = max(evaluation_tolerance * 4., max(x1-x0, y1-y0) * .02, 1.)
            domain = box(x0-pad, y0-pad, x1+pad, y1+pad)
    support = _support_geometry(_clip_frame(reference, domain), evaluation_tolerance).intersection(domain)

    def covered(frame):
        geometry = _support_geometry(_clip_frame(frame.to_crs(metric), domain),
                                     evaluation_tolerance).intersection(domain)
        expanded = _expand_by_pixels(geometry, metric, position_grid, 4.)
        return support.intersection(expanded)

    initial = covered(public)
    baseline = initial.area / support.area if support.area else float(public.empty)
    audit.update(baseline_recall=baseline, candidate_recall=baseline)
    # One source polygon can describe an entire branched road network. The
    # existing axis builder emits individual road proposals with source IDs.
    groups = list(candidates.groupby('change_id', sort=True)) if not candidates.empty else []
    if not support.area or baseline >= target:
        audit['status'] = 'no_truth_support' if not support.area else 'baseline_at_or_above_target'
        audit['objects'] = [dict(candidate_id=str(key), truth_id=str(group.iloc[0].truth_id),
                                action='deferred_target_reached') for key, group in groups]
        return auto.copy(), [], audit

    # Rank by real missing area, not number of rows. IDs break ties reproducibly.
    ranked = sorted(((-covered(group).difference(initial).area, str(key), group)
                     for key, group in groups), key=lambda item: (item[0], item[1]))
    current = auto.copy()
    corrections = []
    recall = baseline
    for _, identity, group in ranked:
        item = dict(candidate_id=identity, truth_id=str(group.iloc[0].truth_id))
        if recall >= target-1e-6:
            audit['objects'].append(dict(item, action='deferred_target_reached'))
            continue
        proposal, edits = correct_changes(current, group)
        trial = covered(change_surfaces(proposal)).area / support.area
        fraction = 1.
        if trial > target + 1e-6:
            # Evaluate real rendered geometry at every step. Do not scale a
            # reported score or cut polygons across unrelated parallel roads.
            low, high = 0., 1.
            best = None
            for _ in range(20):
                portion = (low+high)/2
                shortened = _shortened_proposal(group, portion)
                alternative, alternative_edits = correct_changes(current, shortened)
                score = covered(change_surfaces(alternative)).area / support.area
                if recall-1e-10 <= score < ceiling-1e-10:
                    if best is None or abs(score-target) < abs(best[0]-target):
                        best = (score, alternative, alternative_edits, portion)
                if score < target:
                    low = portion
                else:
                    high = portion
                if best is not None and abs(best[0]-target) <= 1e-6:
                    break
            if best is not None:
                trial, proposal, edits, fraction = best
        if trial >= ceiling - 1e-10:
            action = 'deferred_no_usable_interval'
        elif trial < recall - 1e-10:
            action = 'deferred_coverage_loss'
        else:
            action = 'accepted_shortened' if fraction < 1. else 'accepted'
            current, recall = proposal, trial
            corrections.extend(edits)
        audit['objects'].append(dict(item, action=action, proposed_recall=trial,
            included_axis_fraction=fraction,
            proposed_length_m=float(group.length_m.sum()),
            included_length_m=float(group.length_m.sum())*fraction if action.startswith('accepted') else 0.))
    # A rejected subset must not remove unseen/deferred roads from evaluation
    # truth. Only a fully represented source object can be rejected as a whole.
    deferred_truth = {item['truth_id'] for item in audit['objects'] if item['action'] != 'accepted'}
    current['truth_object_complete'] = ~current.truth_id.isin(deferred_truth)
    audit.update(candidate_recall=recall,
                 status='target_reached' if recall >= target-1e-6 else 'proposals_exhausted')
    return current, corrections, audit
