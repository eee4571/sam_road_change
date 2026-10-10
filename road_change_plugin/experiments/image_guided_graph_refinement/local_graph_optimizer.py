"""Evidence-scored local hypotheses with keep as an explicit competing solution.

Each accepted operation changes a bounded window. There is no ROI-specific
threshold, no control bypass, and no deletion inferred from low SAM probability.
"""
import copy
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import union_all, wkt
from shapely.geometry import LineString, Point
from shapely.ops import substring

from graph import build, candidates, Candidate
from evidence import stations
from medial_repair import propose_axes, replace_corridors, parallel_ribbon_veto, line_parts, pin_axis_contacts
from local_graph_repair import clean_local, thin_regions
from rgb_route import ribbon_evidence, ribbon_cost_features, constrained_route
from refine import safe_gap
from graph_transaction import validate
from color_ribbon import sample_color, descriptor_scales as probe_scales
from engine.canonical_road_surface import _adjacency, _outward


def score_lines(lines, width, evidence, longitudinal=False):
    samples = [(g.length, ribbon_evidence(g, width, evidence)) for g in lines if g.length > 1]
    length = sum(n for n, _ in samples)
    key = 'longitudinal_rgb_support' if longitudinal else 'raw_rgb_support'
    return dict(rgb=sum(n*s[key] for n, s in samples)/max(length, 1),
                raw_rgb=sum(n*s['raw_rgb_support'] for n, s in samples)/max(length, 1),
                evidence_mode='longitudinal_profiles' if longitudinal else 'individual_profiles',
                valid=sum(n*s['valid_fraction'] for n, s in samples)/max(length, 1), length_m=length)


def refit_path(line, width, evidence, spacing=2.5, pin_distance=7.5, corridor=None, continuous_offsets=False):
    """Curvilinear offset search, fixed end nodes, smooth offsets, cached scale only."""
    chainage, xy, _, normals = stations(line, spacing)
    half = min(12., width*.6) if corridor is None else corridor
    offsets = np.arange(-int(half/.75), int(half/.75)+1)*.75
    zero = len(offsets)//2
    if zero == 0:
        return line
    fractions = np.linspace(-.5, 1.5, 41)
    centers = xy[:, None]+normals[:, None]*offsets[None, :, None]
    probe = centers[:, :, None]+normals[:, None, None]*(fractions[None, None, :, None]-.5)*width
    gray = evidence.image.sample(evidence.image.gray, probe.reshape(-1, 2)).reshape(len(xy), len(offsets), -1)
    scores, finite, _, _ = ribbon_cost_features(gray, fractions)
    color = sample_color(evidence, probe, fractions)
    if color is not None:
        scores = np.maximum(scores, color[0])
    cost = 1-scores+.08*(offsets/half)**2
    cost[~finite] = np.inf
    dp = np.full(cost.shape, np.inf)
    back = np.full(cost.shape, -1, int)
    dp[0, zero] = cost[0, zero]
    for i in range(1, len(xy)):
        for j in range(len(offsets)):
            # Pinned approach over the first/last 7.5 m prevents junction kinks.
            if (chainage[i] <= pin_distance or line.length-chainage[i] <= pin_distance) and j != zero:
                continue
            previous = np.arange(max(0, j-1), min(len(offsets), j+2))
            values = dp[i-1, previous]+.2*(previous-j)**2
            k = np.argmin(values)
            dp[i, j], back[i, j] = values[k]+cost[i, j], previous[k]
    if not np.isfinite(dp[-1, zero]):
        return line
    selected = [zero]
    for i in range(len(xy)-1, 0, -1):
        selected.append(back[i, selected[-1]])
    chosen_offsets = offsets[np.asarray(selected[::-1])]
    if continuous_offsets:
        # Subpixel interpolation of the discrete corridor solution, not generic
        # road smoothing. The final path is rescored before any transaction.
        from scipy.ndimage import gaussian_filter1d
        chosen_offsets = gaussian_filter1d(chosen_offsets, 2., mode='nearest')
        taper = np.clip(np.minimum(chainage, line.length-chainage)-pin_distance, 0, 4.5)/4.5
        chosen_offsets *= taper*taper*(3-2*taper)
        chosen_offsets[(chainage <= pin_distance) | (line.length-chainage <= pin_distance)] = 0.
    result = xy+normals*chosen_offsets[:, None]
    result[0], result[-1] = line.coords[0], line.coords[-1]
    return LineString(result).simplify(.1)


def replace_edges(frame, edges, ids, lines, role):
    rows = []
    for i, edge in enumerate(edges):
        if i not in ids:
            row = frame.iloc[edge.sources[0]].to_dict()
            row['geometry'] = edge.axis
            rows.append(row)
    template = frame.iloc[edges[min(ids)].sources[0]].to_dict()
    rows.extend(dict(template, geometry=g, repair_role=role) for g in lines)
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=frame.crs)


def choose(frame, variants, window, evidence, audit, kind, iteration):
    """Evaluate all variants against the same pre-operation graph; commit the best."""
    eligible = []
    transaction_id = hashlib.sha256((window.wkb_hex+kind+str(len(audit))).encode()).hexdigest()[:16]
    for candidate, details in variants:
        details.update(transaction_id=transaction_id, kind=kind, iteration=iteration,
                       window_wkt=window.wkt, alternatives_include_keep=True, applied=False)
        if details.get('evidence_passed', False):
            # Both sides must contain the SAME graph context. Filtering whole
            # features separately is not a spatial crop: an original long feature
            # includes remote contacts that its noded replacement no longer does.
            # That produced false disconnections / outside changes after refits.
            guard = validate(frame.reset_index(drop=True), candidate.reset_index(drop=True), window, evidence)
        else:
            guard = dict(passed=False, failures=['independent_rgb_gain_or_continuity_insufficient'])
        details['transaction_guard'] = guard
        audit.append(details)
        if guard['passed'] and details['objective_gain'] > .02:
            eligible.append((details['objective_gain'], candidate, details))
    if not eligible:
        return frame, False
    _, result, decision = max(eligible, key=lambda v: v[0])
    decision['applied'] = True
    return result.reset_index(drop=True), True


def medial_steps(frame, evidence, roi, audit):
    proposals, seeds = propose_axes(build(frame), evidence, roi)
    viable = []
    for p in proposals:
        veto = parallel_ribbon_veto(p, frame, evidence)
        if veto:
            audit.append(veto)
            continue
        old = [g for geom in frame.geometry.intersection(p.envelope) for g in line_parts(geom) if g.length >= 5]
        a, b = score_lines(old, p.band_scale, evidence, longitudinal=True), score_lines([p.axis], p.band_scale, evidence, longitudinal=True)
        evidence_passed = b['valid'] >= .95 and b['rgb'] >= .60 and b['rgb']-a['rgb'] >= .15
        p.evidence.update(old=a, new=b, evidence_passed=evidence_passed)
        if evidence_passed:
            viable.append(p)
        else:
            audit.append(dict(action='keep_medial', applied=False, candidate_id=p.key,
                candidate_wkt=p.axis.wkt, evidence=p.evidence, reason='keep_competes_no_clear_rgb_improvement'))
    # Intersecting corridors share a transaction, preserving their junction as
    # one graph. Independent corridors are not reverted together.
    groups = []
    for p in viable:
        touching = [g for g in groups if any(p.envelope.intersects(q.envelope) for q in g)]
        combined = [p]+[q for g in touching for q in g]
        groups = [g for g in groups if g not in touching]+[combined]
    for group in groups:
        attempt = list(group)
        for trial in range(3):
            edit_window = union_all([p.envelope for p in attempt]).buffer(max(p.band_scale for p in attempt)*2).intersection(roi)
            fitted = copy.deepcopy(attempt)
            result, edits, contacts = replace_corridors(frame, fitted, roi, evidence)
            old = union_all([wkt.loads(e['old_wkt']) for e in edits if e.get('old_wkt')])
            new = union_all([p.axis for p in fitted]+[wkt.loads(c['geometry_wkt']) for c in contacts])
            actual = [score_lines([p.axis], p.band_scale, evidence, longitudinal=True) for p in fitted]
            gain = float(np.mean([s['rgb']-p.evidence['old']['rgb'] for p, s in zip(fitted, actual)]))-.02
            details = dict(action='medial_graph_rebuild', old_wkt=old.wkt, new_wkt=new.wkt,
                evidence=[p.evidence for p in fitted], final_axis_evidence=actual,
                evidence_passed=all(s['valid'] >= .95 and s['rgb'] >= .60 for s in actual) and gain >= .13,
                objective_gain=gain, hypothesis='joint_group' if trial == 0 else 'anchored_corridor_subset',
                proposals=[dict(id=p.key, axis_wkt=p.axis.wkt, scale=p.band_scale, evidence=p.evidence) for p in fitted],
                branch_contacts=contacts, suboperations=edits,
                reason='rgb_medial_axes_jointly_replace_aliases_and_reattach_external_branches')
            frame, accepted = choose(frame, [(result, details)], edit_window, evidence, audit, 'medial', 0)
            if accepted or len(attempt) == 1:
                break
            # A disconnected neighboring hypothesis must not poison an otherwise
            # anchored whole street. Keep the rejected group in the audit and
            # compare a smaller hypothesis under the SAME evidence/graph guards.
            tips = details['transaction_guard'].get('new_internal_tips', [])
            retained = [p for p in attempt if not any(p.axis.distance(Point(t)) < 2 for t in tips)]
            if not retained or len(retained) == len(attempt):
                retained = [max(attempt, key=lambda p: p.axis.length*p.support)]
            attempt = retained
    return frame


def cycle_steps(frame, evidence, roi, audit, iteration):
    edges = build(frame)
    faces = [face for level in {e.level for e in edges}
             for face in thin_regions(union_all([e.axis for e in edges if e.level == level]), roi)]
    changed = 0
    for face in faces:
        window = face.buffer(3.1).intersection(roi)
        proposed, edits = clean_local(frame, evidence, window, rounds=1, include_teeth=False)
        if not edits:
            continue
        old = union_all([wkt.loads(e['old_wkt']) for e in edits])
        new = union_all([wkt.loads(e['new_wkt']) for e in edits])
        nearby = frame[frame.intersects(old)]
        width = float(nearby.width_m.median())
        a = max((score_lines(line_parts(old), scale, evidence) for scale in probe_scales(width)), key=lambda s: s['rgb'])
        b = max((score_lines(line_parts(new), scale, evidence) for scale in probe_scales(width)), key=lambda s: s['rgb'])
        # A true small ring can have high ribbon support on both arms; keep wins.
        # Subpixel aliases may have equal support, but only if the central path
        # is itself a sustained ribbon. Perimeter never proves that a loop is fake.
        thin_alias = 2*face.area/max(face.length, 1) <= evidence.image.resolution*2
        passed = (b['valid'] >= .95 and b['rgb'] >= .6 and
                  (b['rgb']-a['rgb'] >= .12 or (thin_alias and b['rgb'] >= a['rgb']-.02)))
        gain = b['rgb']-a['rgb']+.08*max(0., 1-new.length/max(old.length, 1))
        details = dict(action='local_cycle_merge_rewire', old_wkt=old.wkt, new_wkt=new.wkt,
            evidence=dict(old=a, new=b, thin_alias=thin_alias, cached_scale_m=width),
            evidence_passed=passed, objective_gain=gain, suboperations=edits,
            reason='central_ribbon_hypothesis_competes_with_closed_ribbon_keep_and_preserves_contacts')
        frame, accepted = choose(frame, [(proposed, details)], window, evidence, audit, 'cycle', iteration)
        changed += accepted
    return frame, changed


def path_steps(frame, evidence, roi, config, audit, iteration):
    changed = 0
    # Gaps are evaluated individually; low SAM is not a veto and formal surface
    # has zero weight in the direction-constrained route search.
    edges = build(frame)
    for c in candidates(edges, roi, config, ('gap',)):
        route = constrained_route(c, evidence)
        score = ribbon_evidence(route, c.width, evidence) if route is not None else None
        safe = route is not None and safe_gap(route, build(frame), c)
        passed = bool(safe and score['raw_rgb_support'] >= .65 and score['continuous_support'] >= .8
                      and score['maximum_unsupported_m'] <= min(10., route.length*.25))
        if route is None:
            audit.append(dict(action='keep_gap', applied=False, candidate_wkt=c.axis.wkt,
                              reason='no_direction_constrained_path'))
            continue
        template = frame.iloc[edges[c.ids[0]].sources[0]].to_dict()
        addition = gpd.GeoDataFrame([dict(template, geometry=route, repair_role='rgb_gap')], crs=frame.crs)
        proposed = gpd.GeoDataFrame(pd.concat([frame, addition], ignore_index=True), crs=frame.crs)
        details = dict(action='connect', new_wkt=route.wkt, candidate_wkt=c.axis.wkt,
            evidence=score, evidence_passed=passed, objective_gain=score['raw_rgb_support']-.5,
            direction_constrained_during_search=True, crossing_guard_passed=safe,
            reason='sustained_rgb_ribbon_and_both_approach_directions_during_search')
        frame, accepted = choose(frame, [(proposed, details)], route.buffer(c.width).intersection(roi),
                                 evidence, audit, 'gap', iteration)
        changed += accepted
    # Local centering hypotheses use pinned true graph nodes. Good roads compete
    # with keep and receive no generic smoothing or length-based spur deletion.
    edges = build(frame)
    queue = [edge.axis.wkb for edge in edges if 25 <= edge.axis.length <= 350
             and not edge.axis.is_ring and roi.buffer(8).covers(edge.axis)]
    for key in queue:
        index = next((i for i, e in enumerate(edges) if e.axis.wkb == key), None)
        if index is None:
            continue
        edge = edges[index]
        if not roi.buffer(-8).covers(edge.axis):
            continue
        width = float(np.median(edge.widths))
        original = max((ribbon_evidence(edge.axis, scale, evidence) for scale in probe_scales(width)),
                       key=lambda s: s['raw_rgb_support'])
        if original['raw_rgb_support'] >= .8 and original['centered_ribbon_strength'] >= .65:
            continue
        variants = []
        for scale in probe_scales(width):
            fitted = refit_path(edge.axis, scale, evidence)
            if fitted.hausdorff_distance(edge.axis) < .5 or not fitted.is_simple:
                continue
            score = ribbon_evidence(fitted, scale, evidence)
            gain = score['raw_rgb_support']-original['raw_rgb_support']
            center_gain = score['centered_ribbon_strength']-original['centered_ribbon_strength']
            passed = (score['valid_fraction'] >= .95 and score['raw_rgb_support'] >= .65 and
                      (gain >= .18 or (center_gain >= .18 and gain >= -.02)))
            # Refit cannot introduce a new road contact anywhere except pinned ends.
            ends = union_all([Point(edge.axis.coords[0]).buffer(.05), Point(edge.axis.coords[-1]).buffer(.05)])
            crossing = any(not fitted.intersection(other.axis).difference(ends).is_empty
                           for j, other in enumerate(edges) if j != index and fitted.intersects(other.axis))
            proposed = replace_edges(frame, edges, {index}, [fitted], 'rgb_refit') if passed and not crossing else frame
            details = dict(action='refit', old_wkt=edge.axis.wkt, new_wkt=fitted.wkt,
                evidence=dict(old=original, new=score, cached_scale_m=width, descriptor_aperture_m=scale),
                evidence_passed=passed and not crossing, objective_gain=max(gain, center_gain)-.04, crossing_guard_passed=not crossing,
                reason='pinned_curvilinear_rgb_centering_improves_independent_support')
            variants.append((proposed, details))
        frame, accepted = choose(frame, variants, edge.axis.buffer(width).intersection(roi),
                                 evidence, audit, 'refit', iteration)
        changed += accepted
        if accepted:
            edges = build(frame)
    return frame, changed


def branch_steps(frame, evidence, roi, config, audit, iteration):
    """Remove a terminal intrusion only with affirmative vegetation evidence.

    The missing ribbon by itself, length by itself, and low SAM are insufficient.
    A sustained parent ribbon and unconnected terminal topology are also required.
    """
    edges = build(frame)
    nodes = _adjacency(edges, set(range(len(edges))))
    changed = 0
    for c in candidates(edges, roi, config, ('spur',)):
        old = c.axis
        terminal = substring(old, 0, old.length*.65)  # candidate traces from the degree-1 tip
        score = max((ribbon_evidence(terminal, scale, evidence) for scale in probe_scales(c.width)), key=lambda s: s['raw_rgb_support'])
        _, xy, _, _ = stations(terminal, 1.5)
        image = evidence.image
        vegetation = 0.
        if hasattr(image, 'rgb'):
            rgb = np.column_stack([image.sample(image.rgb[..., k], xy) for k in range(3)])
            total = np.maximum(rgb.sum(axis=1), 1)
            vegetation = float(np.mean(np.isfinite(rgb).all(axis=1) & ((2*rgb[:, 1]-rgb[:, 0]-rgb[:, 2])/total > .12)))
        junction = next((n for k in c.ids for n in edges[k].nodes if len(nodes[n]) >= 3), None)
        parents = [edges[k] for k, _ in nodes.get(junction, []) if k not in c.ids]
        parent_support = max((ribbon_evidence(e.axis, float(np.median(e.widths)), evidence)['raw_rgb_support'] for e in parents), default=0.)
        positive = max(score['sam_positive_mean'] or 0, score['molra_positive_mean'] or 0)
        passed = (score['valid_fraction'] >= .95 and score['raw_rgb_support'] < .15 and vegetation >= .9
                  and parent_support >= .75 and positive < .65)
        details = dict(action='delete_spur', old_wkt=old.wkt,
            evidence=dict(terminal=score, vegetation_fraction=vegetation, parent_rgb_support=parent_support),
            evidence_passed=passed, objective_gain=vegetation-.75,
            reason='unconnected_terminal_vegetation_intrusion_with_supported_parent_not_length_or_low_sam')
        # IDs are refreshed after a commit; a compound chain cannot partially
        # disappear because some original edge indices were reused.
        current = build(frame)
        ids = {i for i, edge in enumerate(current) if old.buffer(.001).covers(edge.axis)}
        if not ids:
            continue
        proposed = replace_edges(frame, current, ids, [], 'delete_spur')
        frame, accepted = choose(frame, [(proposed, details)], old.buffer(c.width).intersection(roi), evidence, audit, 'spur', iteration)
        changed += accepted
    return frame, changed


def junction_steps(frame, evidence, roi, audit, iteration):
    """Forward-predicted terminal road can reconnect to the interior of a real road.

    Both endpoint-pair search and nearest-point snapping miss this common broken
    access-road case. The target is the first same-level ray intersection, with
    RGB continuity required along the entire route and a fixed target vertex.
    """
    edges = build(frame)
    nodes = _adjacency(edges, set(range(len(edges))))
    tips = [n for n, ends in nodes.items() if len(ends) == 1 and roi.buffer(-8).covers(Point(n[1:]))]
    changed = 0
    for node in tips:
        if node not in nodes or len(nodes[node]) != 1:
            continue
        index, end = nodes[node][0]
        edge = edges[index]
        if edge.axis.length < 12:
            continue
        heading = -_outward(edge, end, 12)
        near_heading = -_outward(edge, end, 6)
        if heading@near_heading < .96:
            continue
        tip = np.array(node[1:])
        ray = LineString([tip, tip+heading*60])
        hits = []
        for j, other in enumerate(edges):
            if j == index:
                continue
            hit = ray.intersection(other.axis)
            if hit.geom_type == 'Point' and 2 <= hit.distance(Point(tip)) <= 60:
                hits.append((hit.distance(Point(tip)), j, hit))
        if not hits:
            continue
        _, target, hit = min(hits, key=lambda h: h[0])
        if edges[target].level != edge.level or not roi.buffer(-2).covers(hit):
            continue
        axis = LineString([tip, hit.coords[0]])
        variants = []
        for scale in probe_scales(float(np.median(edge.widths))):
            c = Candidate('junction_gap', (index, target), axis, scale,
                          dict(approach_directions=[heading.tolist(), heading.tolist()]))
            route = constrained_route(c, evidence)
            if route is None:
                continue
            score = ribbon_evidence(route, scale, evidence)
            parent_score = ribbon_evidence(edge.axis, scale, evidence)['raw_rgb_support']
            target_width = float(np.median(edges[target].widths))
            core_end = max(route.length*.7, route.length-target_width*.5)
            core = substring(route, 0, core_end)
            core_score = ribbon_evidence(core, scale, evidence)
            _, approach_xy, _, _ = stations(substring(route, max(0., core_end-10), core_end), 1.)
            _, junction_xy, _, _ = stations(substring(route, core_end, route.length), 1.)
            approach_gray = evidence.image.sample(evidence.image.gray, approach_xy)
            junction_gray = evidence.image.sample(evidence.image.gray, junction_xy)
            joint_continuity = float(np.mean(np.isfinite(junction_gray) &
                (abs(junction_gray-np.nanmedian(approach_gray)) < .15)))
            target_support = max(ribbon_evidence(edges[target].axis, aperture, evidence)['raw_rgb_support']
                                 for aperture in probe_scales(target_width))
            safe = safe_gap(route, edges, c)
            passed = (safe and parent_score >= .5 and score['raw_rgb_support'] >= .75
                      and core_score['continuous_support'] >= .85 and joint_continuity >= .9
                      and target_support >= .5 and core_score['maximum_unsupported_m'] <= min(7.5, route.length*.2))
            pinned = pin_axis_contacts(edges[target].axis, [hit.coords[0]])
            proposed = frame
            if passed:
                proposed = replace_edges(frame, edges, {target}, [pinned], 'pinned_junction_target')
                template = frame.iloc[edge.sources[0]].to_dict()
                proposed = gpd.GeoDataFrame(pd.concat([proposed, gpd.GeoDataFrame([
                    dict(template, geometry=route, repair_role='rgb_junction_gap')], crs=frame.crs)], ignore_index=True), crs=frame.crs)
            details = dict(action='connect_junction', new_wkt=route.wkt, candidate_wkt=axis.wkt,
                evidence=dict(route=score, core=core_score, junction_pavement_continuity=joint_continuity,
                              target_rgb_support=target_support, parent_rgb_support=parent_score, descriptor_aperture_m=scale),
                evidence_passed=passed, objective_gain=score['raw_rgb_support']-.6,
                crossing_guard_passed=safe, direction_constrained_during_search=True,
                reason='stable_forward_direction_and_continuous_rgb_ribbon_to_first_real_road_contact')
            variants.append((proposed, details))
        frame, accepted = choose(frame, variants, axis.buffer(max(8., float(np.median(edge.widths)))).intersection(roi),
                                 evidence, audit, 'junction_gap', iteration)
        changed += accepted
        if accepted:
            edges = build(frame)
            nodes = _adjacency(edges, set(range(len(edges))))
    return frame, changed


def optimize(frame, evidence, roi, config):
    evidence.multichannel_ribbon = True
    audit, convergence = [], []
    print('  medial graph hypotheses', flush=True)
    result = medial_steps(frame.reset_index(drop=True), evidence, roi, audit)
    for iteration in range(1, 4):
        print(f'  local graph iteration {iteration}', flush=True)
        result, cycles = cycle_steps(result, evidence, roi, audit, iteration)
        result, paths = path_steps(result, evidence, roi, config, audit, iteration)
        result, junctions = junction_steps(result, evidence, roi, audit, iteration)
        result, branches = branch_steps(result, evidence, roi, config, audit, iteration)
        convergence.append(dict(iteration=iteration, committed_cycles=cycles, committed_paths=paths,
                                committed_junctions=junctions, committed_branches=branches))
        if not cycles+paths+junctions+branches:
            break
    return result, audit, convergence
