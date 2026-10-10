"""Skeleton-constrained short-chain, repair-domain and active continuation search.

All alternatives compete against keep. No region-specific tuning, dense extraction,
model execution or width estimation. Cached widths are only descriptor priors.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import union_all, points, distance, set_precision
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, substring

from graph import build, candidates
from evidence import stations
from local_graph_repair import seed_chains
from local_graph_optimizer import refit_path, replace_edges, choose, medial_steps, branch_steps
from block_graph import appearance_scales, consolidate_junctions, complete_streets
from rgb_route import ribbon_evidence, ribbon_cost_features
from color_ribbon import sample_color
from medial_repair import pin_axis_contacts, line_parts
from graph_transaction import validate
from engine.canonical_road_surface import _adjacency


def best_score(line, width, evidence):
    return max((dict(ribbon_evidence(line, s, evidence), aperture_m=s)
                for s in appearance_scales(width)),
               key=lambda r: r['raw_rgb_support']+.25*r['centered_ribbon_strength'])


def bending(line):
    xy = stations(line, 3)[1]
    v = np.diff(xy, axis=0)
    v /= np.maximum(np.linalg.norm(v, axis=1)[:, None], 1e-8)
    return float(np.arccos(np.clip(np.sum(v[1:]*v[:-1], axis=1), -1, 1)).sum())


def parallel_alias_contact(approach, target, contact, width):
    """A dangling side axis beside an existing axis needs reconstruction, not a hook."""
    loc = target.project(Point(contact))
    tangent = np.subtract(target.interpolate(min(target.length, loc+8)).coords[0],
                          target.interpolate(max(0, loc-8)).coords[0])
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    xy = stations(approach, 1.5)[1]
    direction = xy[-1]-xy[0]; direction /= max(np.linalg.norm(direction), 1e-9)
    adjacent = float(np.mean(distance(points(xy), target) <= max(6., min(15., width*.75))))
    return bool(abs(direction@tangent) >= .85 and adjacent >= .6), adjacent


def short_chains(frame, evidence, roi, audit, iteration=0):
    """20..60 m degree-two chains; retain exact real junctions and end approaches."""
    queue = [e for e in seed_chains(build(frame)) if 20 <= e.axis.length <= 60
             and not e.axis.is_ring and roi.buffer(-4).covers(e.axis)]
    changed = 0
    for chain in queue:
        edges = build(frame)
        ids = {i for i, e in enumerate(edges) if e.level == chain.level
               and chain.axis.buffer(.001).covers(e.axis)}
        if not ids or abs(sum(edges[i].axis.length for i in ids)-chain.axis.length) > .05:
            continue
        width = float(np.median(chain.widths))
        old = best_score(chain.axis, width, evidence)
        variants = []
        for aperture in appearance_scales(width):
            fitted = refit_path(chain.axis, aperture, evidence, spacing=1.5,
                                pin_distance=3., corridor=min(4.5, aperture*.55), continuous_offsets=True)
            if fitted.hausdorff_distance(chain.axis) < .4:
                continue
            score = ribbon_evidence(fitted, aperture, evidence)
            gain = score['raw_rgb_support']-old['raw_rgb_support']
            center_gain = score['centered_ribbon_strength']-old['centered_ribbon_strength']
            ends = union_all([Point(chain.axis.coords[0]).buffer(.05), Point(chain.axis.coords[-1]).buffer(.05)])
            crossing = any(not fitted.intersection(e.axis).difference(ends).is_empty
                           for i, e in enumerate(edges) if i not in ids)
            curvature = bending(fitted)-bending(chain.axis)
            passed = (not crossing and fitted.is_simple and score['valid_fraction'] >= .95
                      and score['raw_rgb_support'] >= .65 and curvature <= .6
                      and (gain >= .18 or (center_gain >= .18 and gain >= -.02)))
            proposed = replace_edges(frame, edges, ids, [fitted], 'short_corridor_refit') if passed else frame
            variants.append((proposed, dict(action='short_corridor_refit', old_wkt=chain.axis.wkt,
                new_wkt=fitted.wkt, evidence=dict(old=old, new=score, curvature_change_rad=curvature,
                                                chain_length_m=chain.axis.length),
                evidence_passed=passed, objective_gain=max(gain, center_gain)-.04,
                reason='short_curved_degree2_chain_rgb_centering_with_fixed_contacts_and_bounded_corridor')))
        frame, applied = choose(frame, variants, chain.axis.buffer(5).intersection(roi),
                                evidence, audit, 'short_chain', iteration)
        changed += applied
    return frame, changed


def trace_endpoint(start, heading, width, evidence, network, corridor, max_length=60):
    """Beam search follows ribbon before looking for a target road; 2.5 m steps.

    State includes position, heading and complete path. SAM/MOLRA only reduce
    cost; a path with no RGB ribbon is never accepted from model evidence alone.
    """
    fractions = np.linspace(-.5, 1.5, 25)
    start = np.asarray(start, float)
    normal0 = np.array([-heading[1], heading[0]])
    states = [(0., [start], np.asarray(heading))]
    hits = []
    for step in range(int(max_length/2.5)):
        options = []
        for total, path, direction in states:
            for angle in np.deg2rad([-20, -10, 0, 10, 20]):
                cs, sn = np.cos(angle), np.sin(angle)
                v = np.array([cs*direction[0]-sn*direction[1], sn*direction[0]+cs*direction[1]])
                xy = path[-1]+v*2.5
                if (xy-start)@heading < 1 or abs((xy-start)@normal0) > 18 or not corridor.covers(Point(xy)):
                    continue
                if len(path) > 5 and Point(xy).distance(LineString(path[:-3])) < 2:
                    continue
                options.append((total, path, v, xy, abs(angle)))
        if not options:
            break
        xy = np.array([o[3] for o in options]); normals = np.array([[-o[2][1], o[2][0]] for o in options])
        strength = np.zeros(len(options))
        for aperture in appearance_scales(width):
            probes = xy[:, None]+normals[:, None]*(fractions[None, :, None]-.5)*aperture
            gray = evidence.image.sample(evidence.image.gray, probes.reshape(-1, 2)).reshape(len(xy), -1)
            score, finite, _, _ = ribbon_cost_features(gray, fractions)
            color = sample_color(evidence, probes, fractions)
            if color is not None:
                score = np.maximum(score, color[0])
            strength = np.maximum(strength, np.where(finite, score, 0))
        p, m, _ = evidence.values(xy)
        positive = np.maximum(np.nan_to_num(p), np.nan_to_num(m))
        ranked = []
        for k, (total, path, v, xy, turn) in enumerate(options):
            cost = total+1-strength[k]+.3*turn-.1*np.clip(positive[k], 0, 1)
            path = path+[xy]
            if step >= 1 and Point(xy).distance(network) < 2.:
                target = network.interpolate(network.project(Point(xy))) if network.geom_type == 'LineString' else None
                if target is None:
                    from shapely.ops import nearest_points
                    target = nearest_points(Point(xy), network)[1]
                line = LineString(path[:-1]+[np.array(target.coords[0])]).simplify(.35)
                hits.append((cost/len(path), line))
                continue
            ranked.append((cost, path, v))
        states = []
        bins = set()
        for row in sorted(ranked, key=lambda v: v[0]):
            key = tuple(np.round(row[1][-1]/1.5).astype(int))
            if key not in bins:
                bins.add(key); states.append(row)
            if len(states) == 6:
                break
        if hits or not states:
            break
    return sorted(hits, key=lambda v: v[0])[:6]


def active_continuations(frame, evidence, roi, audit, iteration=0):
    chains = seed_chains(build(frame)); nodes = _adjacency(chains, set(range(len(chains))))
    tips = [n for n, ends in nodes.items() if len(ends) == 1 and roi.buffer(-5).covers(Point(n[1:]))]
    changed = 0
    for node in tips:
        chains = seed_chains(build(frame)); nodes = _adjacency(chains, set(range(len(chains))))
        if len(nodes.get(node, [])) != 1:
            continue
        index, end = nodes[node][0]; chain = chains[index]
        if chain.axis.length < 12:
            continue
        line = chain.axis if end == 1 else LineString(list(chain.axis.coords)[::-1])
        approach = substring(line, max(0, line.length-15), line.length)
        xy = stations(approach, 2)[1]
        direction = xy[-1]-xy[0]; direction /= np.linalg.norm(direction)
        v = xy[-1]-xy[-3]; v /= max(np.linalg.norm(v), 1e-8)
        if v@direction < .9:
            continue
        width = float(np.median(chain.widths))
        source_score = best_score(approach, width, evidence)
        if source_score['raw_rgb_support'] < .65:
            continue
        others = [e for i, e in enumerate(chains) if i != index]
        network = union_all([e.axis for e in others])
        if network.is_empty or Point(node[1:]).distance(network) > 60:
            continue
        start = np.array(node[1:]); normal = np.array([-direction[1], direction[0]])
        from shapely.geometry import Polygon
        corridor = Polygon([start-normal*18-direction, start+normal*18-direction,
                            start+direction*60+normal*18, start+direction*60-normal*18]).intersection(roi.buffer(-1))
        variants = []
        hits = trace_endpoint(start, direction, width, evidence, network, corridor)
        for _, route in hits:
            target = min(others, key=lambda e: e.axis.distance(Point(route.coords[-1])))
            core = substring(route, 0, max(route.length*.65, route.length-4))
            score = best_score(core, width, evidence)
            location = target.axis.project(Point(route.coords[-1]))
            target_local = substring(target.axis, max(0, location-12), min(target.axis.length, location+12))
            target_score = best_score(target_local, float(np.median(target.widths)), evidence)
            alias, adjacent = parallel_alias_contact(approach, target.axis, route.coords[-1], width)
            # The same pavement must persist from the observed approach; a sharp
            # ribbon on a roof/yard is not sufficient evidence for continuation.
            material_fraction = 1.
            if hasattr(evidence.image, 'rgb'):
                def rgb_at(geom):
                    sample_xy = stations(geom, 1.5)[1]
                    return np.column_stack([evidence.image.sample(evidence.image.rgb[..., k], sample_xy) for k in range(3)])/255.
                observed, trial = rgb_at(approach), rgb_at(route)
                low, high = np.nanquantile(observed.mean(axis=1), [.1, .9])
                reference = np.nanmedian(observed/np.maximum(observed.sum(axis=1, keepdims=True), .05), axis=0)
                chroma = trial/np.maximum(trial.sum(axis=1, keepdims=True), .05)
                material_fraction = float(np.mean(np.isfinite(trial).all(axis=1)
                    & (np.linalg.norm(chroma-reference, axis=1) < .09)
                    & (trial.mean(axis=1) >= low-.08) & (trial.mean(axis=1) <= high+.08)))
            # Cross-layer roads stop the trace; they cannot become a target or be crossed.
            end_disks = union_all([Point(route.coords[0]).buffer(.05), Point(route.coords[-1]).buffer(.05)])
            crossing = any(not route.intersection(e.axis).difference(end_disks).is_empty for e in chains)
            passed = (target.level == chain.level and not crossing and not alias and route.is_simple
                      and score['raw_rgb_support'] >= .75 and score['continuous_support'] >= .85
                      and score['maximum_unsupported_m'] <= min(5., route.length*.2)
                      and target_score['raw_rgb_support'] >= .5 and bending(route) <= 1.8
                      and material_fraction >= .85)
            proposed = frame
            if passed:
                # Materialize the exact target contact in the existing line geometry.
                edges = build(frame)
                target_ids = {i for i, e in enumerate(edges) if e.level == target.level
                              and e.axis.distance(Point(route.coords[-1])) < .001}
                pinned = [pin_axis_contacts(edges[i].axis, [route.coords[-1]]) for i in sorted(target_ids)]
                proposed = replace_edges(frame, edges, target_ids, pinned, 'active_target_contact')
                template = frame.iloc[chain.sources[0]].to_dict()
                addition = gpd.GeoDataFrame([dict(template, geometry=route, repair_role='active_continuation')], crs=frame.crs)
                proposed = gpd.GeoDataFrame(pd.concat([proposed, addition], ignore_index=True), crs=frame.crs)
            variants.append((proposed, dict(action='active_continuation', new_wkt=route.wkt,
                evidence=dict(route=score, source=source_score, target=target_score,
                              same_pavement_fraction=material_fraction, target_context_m=24,
                              parallel_alias_contact=alias, approach_adjacent_fraction=adjacent), evidence_passed=passed,
                objective_gain=score['raw_rgb_support']-.55, reason='endpoint_led_ribbon_trace_to_first_existing_road',
                target_preselected=False, direction_constrained_during_search=True)))
            if alias:
                variants[-1][1]['reason'] = 'parallel_side_axis_requires_canonical_reconstruction_not_a_new_hook'
        if not hits:
            audit.append(dict(action='active_continuation', applied=False, candidate_wkt=approach.wkt,
                              reason='bounded_ribbon_trace_did_not_reach_an_existing_road'))
        frame, applied = choose(frame, variants, corridor, evidence, audit, 'active_trace', iteration)
        changed += applied
    return frame, changed


def repair_domains(frame, roi, config):
    edges = build(frame); nodes = _adjacency(edges, set(range(len(edges))))
    events = [(c.axis.centroid, c.kind) for c in candidates(edges, roi, config)]
    events += [(p.centroid, 'loop') for level in {e.level for e in edges}
               for p in polygonize(union_all([set_precision(e.axis, .001) for e in edges if e.level == level]))
               if p.length < 130 and roi.covers(p)]
    events += [(Point(n[1:]), 'junction') for n, ends in nodes.items() if len(ends) >= 4 and roi.covers(Point(n[1:]))]
    ranked = []
    for center, _ in events:
        types = sorted({kind for point, kind in events if point.distance(center) <= 25})
        if len(types) >= 2 and roi.buffer(-26).covers(center):
            ranked.append((len(types), center, types))
    result = []
    for _, center, types in sorted(ranked, key=lambda row: -row[0]):
        if any(center.distance(w.centroid) < 40 for w, _ in result):
            continue
        result.append((center.buffer(25), types))
        if len(result) == 12:
            break
    return result


def domain_objective(frame, window, evidence):
    loss = total = curvature = 0.
    for _, row in frame[frame.intersects(window)].iterrows():
        for line in line_parts(row.geometry.intersection(window)):
            if line.length < 1:
                continue
            score = best_score(line, row.width_m, evidence)
            loss += line.length*(1-score['raw_rgb_support']+.25*(1-score['centered_ribbon_strength']))
            curvature += bending(line); total += line.length
    return dict(loss=loss+.5*curvature, rgb_loss_m=loss, curvature_rad=curvature, length_m=total)


def rebuild_domains(frame, evidence, roi, config, audit, iteration):
    changed = 0
    for window, types in repair_domains(frame, roi, config):
        # Boundary contacts and all outside geometry remain in context. Each branch
        # starts from the SAME input; only the best complete domain is committed.
        mask = frame.intersects(window.buffer(35))
        base = frame[mask].reset_index(drop=True)
        variants = []
        old = domain_objective(base, window, evidence)
        collapsed, ca = consolidate_junctions(base, evidence, window)
        variants.append(('collapse_rewire', collapsed, ca))
        traced_audit = []
        traced, _ = active_continuations(base, evidence, window, traced_audit, iteration)
        variants.append(('connect', traced, traced_audit))
        combined_audit = list(ca)
        combined, _ = short_chains(collapsed, evidence, window, combined_audit, iteration)
        combined, _ = active_continuations(combined, evidence, window, combined_audit, iteration)
        combined, _ = branch_steps(combined, evidence, window, config, combined_audit, iteration)
        variants.append(('collapse_refit_connect_prune', combined, combined_audit))
        choices = []
        for name, proposed, operations in variants:
            applied = [r for r in operations if r.get('applied')]
            if not applied:
                audit.append(dict(action='repair_domain', hypothesis=name, applied=False,
                    alternatives_include_keep=True, event_types=types, iteration=iteration,
                    window_wkt=window.wkt, suboperations=operations,
                    reason='no_feasible_supported_subgraph_for_this_hypothesis'))
                continue
            new = domain_objective(proposed, window, evidence)
            # Reduction in unsupported geometry can favor canonical collapse; a
            # supported new connection has an explicit connectivity benefit.
            connections = sum(r['action'] == 'active_continuation' for r in applied)
            gain = (old['loss']-new['loss']+5*connections)/max(old['length_m'], 20)
            guard = validate(base, proposed, window, evidence)
            record = dict(action='repair_domain', hypothesis=name, applied=False,
                alternatives_include_keep=True, event_types=types, iteration=iteration,
                old_wkt=union_all(base.geometry).intersection(window).wkt,
                new_wkt=union_all(proposed.geometry).intersection(window).wkt, window_wkt=window.wkt,
                objective_gain=gain, evidence=dict(old=old, new=new), transaction_guard=guard,
                suboperations=operations, reason='joint_domain_comparison_with_fixed_external_contacts')
            audit.append(record)
            if guard['passed'] and gain > .02:
                choices.append((gain, proposed, record))
        if choices:
            _, proposed, record = max(choices, key=lambda v: v[0])
            record['applied'] = True; changed += 1
            frame = gpd.GeoDataFrame(pd.concat([frame[~mask], proposed], ignore_index=True), crs=frame.crs)
        else:
            audit.append(dict(action='keep_domain', applied=False, window_wkt=window.wkt,
                event_types=types, reason='keep_wins_no_supported_feasible_joint_improvement'))
    return frame, changed


def optimize(frame, evidence, roi, config):
    evidence.multichannel_ribbon = True; evidence.independent_flanks = True
    audit, convergence = [], []
    print('  canonical same-pavement corridor graph', flush=True)
    frame, corridor_audit = consolidate_junctions(frame.reset_index(drop=True), evidence, roi, corridor_mode=True)
    audit.extend(corridor_audit)
    print('  medial axes with branch reattachment', flush=True)
    result = medial_steps(frame.reset_index(drop=True), evidence, roi, audit)
    for iteration in range(1, 3):
        print(f'  structural domains / short chains / active tracing: round {iteration}', flush=True)
        result, domains = rebuild_domains(result, evidence, roi, config, audit, iteration)
        result, shorts = short_chains(result, evidence, roi, audit, iteration)
        result, traced = active_continuations(result, evidence, roi, audit, iteration)
        # Retain the already verified paired curved route and paved junction
        # hypotheses alongside the new target-free search; no global smoothing.
        result, junction_audit = consolidate_junctions(result, evidence, roi)
        result, pair_audit = complete_streets(result, evidence, roi)
        audit.extend(junction_audit+pair_audit)
        others = sum(r.get('applied', False) for r in junction_audit+pair_audit)
        convergence.append(dict(iteration=iteration, domains=domains, short_chains=shorts,
                                active_connections=traced, retained_operations=others))
        if domains+shorts+traced+others == 0:
            break
    return result, audit, convergence
