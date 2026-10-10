"""RGB-checked straight runs in internal degree-two chains, with fixed contacts.

This is geometry regularization, not extraction or gap completion. Existing
carriageways, junctions, true bend approaches and outside geometry are immutable.
"""
import numpy as np
from shapely import union_all, from_wkt
from shapely.geometry import LineString, Point
from shapely.ops import substring

from graph import build
from graph_transaction import validate
from local_graph_repair import seed_chains, thin_regions, clean_local
from block_graph import pavement_inside
from medial_repair import line_parts
from local_graph_optimizer import replace_edges
from structural_refinement import bending
from whole_corridor import axis_strength


def collapse_internal_aliases(frame, evidence, roi):
    """Reuse local reconstruction only for a same-pavement two-contact alias."""
    audit = []
    for face in thin_regions(union_all(frame.geometry), roi):
        window = face.buffer(5).intersection(roi)
        edges = build(frame)
        if any(getattr(e, 'carriageway_id', '') and e.axis.intersects(window) for e in edges):
            continue
        interior = pavement_inside(face, evidence)
        if not interior['consistent']:
            continue
        trial, edits = clean_local(frame, evidence, window, rounds=1, include_teeth=False)
        if len(edits) != 1 or len(edits[0].get('external_contacts', [])) != 2:
            continue
        edit = edits[0]
        old = line_parts(from_wkt(edit['old_wkt']))
        new = from_wkt(edit['new_wkt'])
        if new.geom_type != 'LineString':
            continue
        rows = []
        for scale in (3.,5.,8.,12.):
            scores = [axis_strength(g,scale,evidence)[0] for g in old]
            old_score = float(np.average(scores, weights=[g.length for g in old]))
            new_score,valid = axis_strength(new,scale,evidence)
            rows.append(dict(aperture_m=scale,old=old_score,new=new_score,valid=valid))
        selected = max(rows,key=lambda r:r['old'])
        passed = selected['valid'] >= .98 and selected['new'] >= max(.2,selected['old']-.025)
        edit.update(applied=False, evidence=dict(interior=interior,rgb=selected,apertures=rows),
                    reason='same_pavement_two_contact_alias_reconstruction_before_straight_fit')
        audit.append(edit)
        if passed:
            guard = validate(frame,trial,window,evidence)
            edit['transaction_guard'] = guard
            if guard['passed']:
                edit['applied'] = True
                frame = trial.reset_index(drop=True)
    return frame,audit


def straight_runs(line, tolerance, return_spans=False):
    """Replace long near-straight spans; retain original geometry around corners."""
    skeleton = np.asarray(line.simplify(tolerance).coords)
    locations = [line.project(Point(p)) for p in skeleton]
    protected = set()
    for i in range(1, len(skeleton)-1):
        a, b = skeleton[i]-skeleton[i-1], skeleton[i+1]-skeleton[i]
        angle = np.arccos(np.clip(a@b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-9), -1, 1))
        if angle > np.deg2rad(20):
            protected.add(i)
    xy = [line.coords[0]]
    spans = []
    cursor = 0.
    for i, (start, end) in enumerate(zip(locations[:-1], locations[1:])):
        start += min(4., (end-start)*.2) if i in protected else 0.
        end -= min(4., (end-start)*.2) if i+1 in protected else 0.
        if end-start < 12:
            continue
        old = substring(line, start, end)
        chord = LineString([old.coords[0], old.coords[-1]])
        if chord.hausdorff_distance(old) > tolerance or chord.length < old.length*.96:
            continue
        if start > cursor:
            xy.extend(list(substring(line, cursor, start).coords)[1:])
        xy.append(chord.coords[-1])
        if chord.hausdorff_distance(old) > .1:
            spans.append((old, chord))
        cursor = end
    if cursor < line.length:
        xy.extend(list(substring(line, cursor, line.length).coords)[1:])
    result = LineString(xy)
    return (result, spans) if return_spans else result


def rgb_comparison(old, new, evidence):
    # Same descriptor aperture for both alternatives; cached axis width does
    # not decide whether a road is real or whether its alignment is preferable.
    rows = []
    for scale in (3., 5., 8., 12.):
        a, valid_a = axis_strength(old, scale, evidence)
        b, valid_b = axis_strength(new, scale, evidence)
        rows.append(dict(aperture_m=scale, old=a, new=b, valid=min(valid_a, valid_b)))
    selected = max(rows, key=lambda r: r['old'])
    n = max(3, int(old.length/1.5)+1)
    fractions = np.linspace(0, 1, n)
    axy = np.array([old.interpolate(f, normalized=True).coords[0] for f in fractions])
    bxy = np.array([new.interpolate(f, normalized=True).coords[0] for f in fractions])
    a = np.column_stack([evidence.image.sample(evidence.image.rgb[...,k], axy) for k in range(3)])/255.
    b = np.column_stack([evidence.image.sample(evidence.image.rgb[...,k], bxy) for k in range(3)])/255.
    unchanged_material = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (np.linalg.norm(a-b, axis=1)/np.sqrt(3) < .12)
    material_fraction = float(unchanged_material.mean())
    passed = (selected['valid'] >= .98 and selected['new'] >= .2
              and selected['new'] >= selected['old']-.025 and material_fraction >= .9)
    return dict(passed=passed, selected=selected, apertures=rows,
                same_rgb_material_fraction=material_fraction)


def regularize(frame, evidence, roi):
    """Try each chain once; each accepted edit preserves the existing graph."""
    frame, audit = collapse_internal_aliases(frame, evidence, roi)
    queue = seed_chains(build(frame))
    for chain in queue:
        old = chain.axis
        if old.length < 15 or old.is_ring or not old.is_simple or not roi.buffer(-2).covers(old):
            continue
        edges = build(frame)
        ids = {i for i, e in enumerate(edges) if e.level == chain.level and old.buffer(.0001).covers(e.axis)}
        if not ids or abs(sum(edges[i].axis.length for i in ids)-old.length) > .02:
            continue
        if any(getattr(edges[i], 'carriageway_id', '') for i in ids):
            continue
        old_bend = bending(old)
        variants = []
        for tolerance in (1., 2., 3.):
            new, spans = straight_runs(old, tolerance, return_spans=True)
            movement = new.hausdorff_distance(old)
            new_bend = bending(new)
            if movement < .3 or new_bend > old_bend*.8 or old_bend-new_bend < .15:
                continue
            rgb = rgb_comparison(old, new, evidence)
            local_rgb = [dict(old_wkt=a.wkt, new_wkt=b.wkt, **rgb_comparison(a,b,evidence)) for a,b in spans]
            evidence_passed = rgb['passed'] and all(r['passed'] for r in local_rgb)
            ends = union_all([Point(old.coords[0]).buffer(.001), Point(old.coords[-1]).buffer(.001)])
            crossing = any(not new.intersection(e.axis).difference(ends).is_empty for i,e in enumerate(edges) if i not in ids)
            record = dict(action='internal_straight_runs', applied=False, old_wkt=old.wkt, new_wkt=new.wkt,
                          reason='fixed_junctions_and_bends_rgb_supported_straight_runs',
                          evidence=dict(rgb=rgb, local_rgb=local_rgb, maximum_displacement_m=movement,
                                        tolerance_m=tolerance, bending_before_rad=old_bend, bending_after_rad=new_bend),
                          crossing=crossing, decision='reject' if not evidence_passed or crossing or not new.is_simple else 'alternative',
                          rejection_reason='rgb_support_or_new_crossing' if not evidence_passed or crossing or not new.is_simple else None)
            audit.append(record)
            if evidence_passed and not crossing and new.is_simple:
                variants.append((new_bend, movement, new, record))
        for _, _, new, record in sorted(variants, key=lambda r: (r[0], r[1])):
            proposed = replace_edges(frame, edges, ids, [new], 'internal_straight_runs')
            window = old.buffer(3.1).intersection(roi)
            guard = validate(frame, proposed, window, evidence)
            record['transaction_guard'] = guard
            if guard['passed']:
                record['applied'] = True
                record['decision'] = 'refit'
                frame = proposed.reset_index(drop=True)
                break
            record['decision'] = 'reject'
            record['rejection_reason'] = guard['failures']
    return frame, audit
