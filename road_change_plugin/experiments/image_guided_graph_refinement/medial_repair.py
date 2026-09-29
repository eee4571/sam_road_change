"""Image-guided medial axes and transactional replacement of boundary networks.

All arrays are local descriptors of existing RGB. No model or width estimator is
called. Ribbon envelopes are temporary edit domains, never new width products.
"""
from dataclasses import dataclass
import hashlib

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing
from shapely import union_all, points, distance, wkt
from shapely.geometry import LineString, Point

from evidence import stations
from ribbon_diagnostic import diagnose, raw_ribbon_profile
from rgb_route import ribbon_cost_features


@dataclass
class AxisProposal:
    axis: object
    envelope: object
    band_scale: float
    support: float
    evidence: dict
    seed_ids: list
    level: tuple = ('0', False, False)

    @property
    def key(self):
        if not hasattr(self, '_candidate_key'):
            self._candidate_key = 'medial_' + hashlib.sha256(self.axis.wkb).hexdigest()[:16]
        return self._candidate_key


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return [geometry] if geometry.length > .01 else []
    return [line for part in getattr(geometry, 'geoms', []) for line in line_parts(part)]


def direction(axis):
    xy = np.array(axis.coords)
    v = xy[-1]-xy[0]
    v /= max(np.linalg.norm(v), 1e-9)
    return -v if v[0] < 0 else v


def pin_axis_contacts(axis, contacts):
    """Insert exact shared vertices; interpolation alone can miss GEOS noding."""
    nodes = [(axis.project(Point(coord)), 0, coord) for coord in axis.coords]
    nodes.extend((axis.project(Point(coord)), 1, tuple(coord)) for coord in contacts
                 if Point(coord).distance(axis) < 1e-5)
    selected = []
    for station, priority, coord in sorted(nodes):
        if selected and abs(station-selected[-1][0]) < 1e-6:
            if priority:
                selected[-1] = (station, priority, coord)
        else:
            selected.append((station, priority, coord))
    return LineString([coord for _, _, coord in selected])


def parallel_ribbon_veto(proposal, frame, evidence):
    """A dark median between two independently supported roads is not a road axis."""
    from rgb_route import ribbon_evidence
    from engine.canonical_road_surface import _level
    _, center, tangent, normal = stations(proposal.axis, 5)
    middle_gray = evidence.image.sample(evidence.image.gray, center)
    sides = {-1: [], 1: []}
    search = proposal.axis.buffer(proposal.band_scale, cap_style='flat')
    for _, row in frame[frame.intersects(search)].iterrows():
        if _level(row) != proposal.level:
            continue
        for line in line_parts(row.geometry.intersection(search)):
            if line.length < 20 or abs(direction(line)@direction(proposal.axis)) < .95:
                continue
            support = ribbon_evidence(line, float(row.width_m), evidence)['raw_rgb_support']
            if support < .65:
                continue
            xy = np.array([line.interpolate(line.project(Point(p))).coords[0] for p in center])
            delta = xy-center
            offset = np.sum(delta*normal, axis=1)
            aligned = abs(np.sum(delta*tangent, axis=1)) < 5
            gray = evidence.image.sample(evidence.image.gray, xy)
            for sign in (-1, 1):
                valid = aligned & (offset*sign > max(3., float(row.width_m)*.4)) & np.isfinite(gray)
                if valid.any():
                    sides[sign].append((valid, gray))
    for valid_left, left in sides[-1]:
        for valid_right, right in sides[1]:
            valid = valid_left & valid_right & np.isfinite(middle_gray)
            dl, dr = middle_gray-left, middle_gray-right
            separator = valid & (dl*dr > 0) & (np.minimum(abs(dl), abs(dr)) >= .18)
            if separator.mean() >= .65:
                return dict(action='retain_parallel_roads', applied=False, candidate_id=proposal.key,
                    candidate_wkt=proposal.axis.wkt, evidence=dict(stable_separator_fraction=float(separator.mean()),
                    independent_side_ribbon_minimum_support=.65),
                    reason='two_independent_rgb_road_ribbons_with_stable_separator_do_not_replace_with_middle_axis')
    return None


def seed_groups(edges, evidence, roi):
    reports = diagnose(edges, evidence, roi.buffer(80))
    seeds = [(wkt.loads(r['result_wkt']), r['geometry_scores']['separation_m'], r['candidate_id']) for r in reports]
    # Include a road already carrying one approximate axis. Only sustained RGB
    # ribbons qualify; existing width sets a search scale, not a positive vote.
    from local_graph_repair import seed_chains
    for i, edge in enumerate(seed_chains(edges)):
        if edge.axis.length < 65 or edge.axis.is_ring or not roi.intersects(edge.axis):
            continue
        axis = edge.axis.simplify(3)
        v = direction(axis)
        xy = np.asarray(axis.coords)
        normal = np.array([-v[1], v[0]])
        if np.ptp(xy@normal) > 12:
            continue
        _, centers, _, normals = stations(axis, 5)
        scale = float(np.median(edge.widths))
        if not 5 <= scale <= 40:
            continue
        fractions = np.linspace(-.5, 1.5, 61)
        probes = centers[:, None, :] + normals[:, None, :]*(fractions[None, :, None]-.5)*scale
        gray = evidence.image.sample(evidence.image.gray, probes.reshape(-1, 2)).reshape(len(centers), -1)
        if not np.isfinite(gray).all():
            continue
        raw = raw_ribbon_profile(gray, fractions)
        if raw['raw_ribbon_support_fraction'] >= .65:
            centers += normals*(np.array(raw['center_fractions'])-.5)[:, None]*scale
            seeds.append((LineString(centers), scale, f'existing_axis_{i}'))
    # Pairwise grouping uses direction AND lateral separation. It never joins
    # adjacent parallel roads just because their buffers overlap.
    parent = list(range(len(seeds)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (a, wa, _) in enumerate(seeds):
        va = direction(a)
        normal = np.array([-va[1], va[0]])
        for j in range(i):
            b, wb, _ = seeds[j]
            if abs(va@direction(b)) < .98 or max(wa, wb)/min(wa, wb) > 1.8:
                continue
            delta = np.asarray(a.centroid.coords[0])-np.asarray(b.centroid.coords[0])
            if abs(delta@normal) <= 6 and a.distance(b) <= 160:
                parent[root(i)] = root(j)
    groups = {}
    for i, seed in enumerate(seeds):
        groups.setdefault(root(i), []).append(seed)
    return list(groups.values()), reports


def fit_medial(group, evidence, roi):
    coords = np.concatenate([np.asarray(a.coords) for a, _, _ in group])
    origin = np.median(coords, axis=0)
    _, _, vt = np.linalg.svd(coords-origin, full_matrices=False)
    tangent = vt[0]
    if tangent[0] < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    scale = float(np.median([s for _, s, _ in group]))
    seed_s = (coords-origin)@tangent
    span = np.ptp(seed_s)
    if span < 20:
        return []
    # Fit the full local continuation, not just the few accepted seed stations.
    rectangle = roi.buffer(20)
    chord = LineString([origin-tangent*3000, origin+tangent*3000]).intersection(rectangle)
    if chord.geom_type != 'LineString':
        return []
    ends = (np.asarray(chord.coords)-origin)@tangent
    along = np.arange(ends.min(), ends.max()+3, 3.)
    base = origin+along[:, None]*tangent
    reach = min(40., max(20., scale*1.5))
    offsets = np.arange(-reach, reach+.5, .75)
    fractions = np.linspace(-.5, 1.5, 41)
    lateral = offsets[:, None]+(fractions[None, :]-.5)*scale
    probes = base[:, None, None, :]+normal*lateral[None, :, :, None]
    gray = evidence.image.sample(evidence.image.gray, probes.reshape(-1, 2)).reshape(len(base), len(offsets), -1)
    score, finite, contrast, spread = ribbon_cost_features(gray, fractions)
    score -= .08*(offsets[None, :]/max(scale/2, 1))**2
    # Position-state dynamic programming restricts direction during search.
    jumps = np.arange(-2, 3)
    costs = np.full(score.shape, np.inf)
    back = np.zeros(score.shape, np.int16)
    costs[0] = 1-score[0]
    for i in range(1, len(base)):
        options = np.full((len(jumps), len(offsets)), np.inf)
        for k, jump in enumerate(jumps):
            previous = np.arange(len(offsets))-jump
            valid = (previous >= 0) & (previous < len(offsets))
            options[k, valid] = costs[i-1, previous[valid]] + .12*jump**2
        choice = options.argmin(axis=0)
        costs[i] = options[choice, np.arange(len(offsets))]+1-score[i]
        back[i] = np.arange(len(offsets))-jumps[choice]
    selected = np.zeros(len(base), int)
    selected[-1] = costs[-1].argmin()
    for i in range(len(base)-1, 0, -1):
        selected[i-1] = back[i, selected[i]]
    chosen_gray = gray[np.arange(len(base)), selected]
    safe = np.isfinite(chosen_gray).all(axis=1)
    raw = raw_ribbon_profile(np.where(safe[:, None], chosen_gray, 0), fractions)
    supported = np.array(raw['supported_sections']) & safe
    # At a junction, side contrast disappears while the pavement continues.
    # Fill only short bounded holes; never extrapolate through a long missing road.
    connected = binary_closing(supported, structure=np.ones(15), border_value=0)
    selected_xy = base+normal*gaussian_filter1d(offsets[selected], 2)[:, None]
    indices = np.flatnonzero(connected)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1) if len(indices) else []
    if not hasattr(evidence, 'axis_searches'):
        evidence.axis_searches = []
    evidence.axis_searches.append(dict(scale=scale, tangent=tangent.tolist(), seed_span_m=span,
        seed_ids=[key for _, _, key in group], runs=[dict(length_m=3*len(ids),
        start=selected_xy[ids[0]].tolist(), end=selected_xy[ids[-1]].tolist(),
        support=float(supported[ids].mean())) for ids in runs if len(ids)]))
    result = []
    anchored_runs = [run for run in runs if len(run) >= 20 and
                     along[run].max() >= seed_s.min() and along[run].min() <= seed_s.max()]
    for ids in runs:
        if len(ids) < 20:
            continue
        if supported[ids].mean() < .5:
            continue
        anchored = (along[ids].max() >= seed_s.min()) and (along[ids].min() <= seed_s.max())
        near_prior = distance(points(selected_xy[ids]), evidence.network_prior) <= max(10., scale)
        prior_fraction = float(np.mean(near_prior))
        bounded_continuation = (max(np.mean(near_prior[:10]), np.mean(near_prior[-10:])) >= .8 and
            any(min(abs(along[ids[0]]-along[r[-1]]), abs(along[ids[-1]]-along[r[0]])) <= 6*scale for r in anchored_runs))
        if not anchored and prior_fraction < .8 and not bounded_continuation:
            continue
        axis = LineString(selected_xy[ids]).simplify(.5)
        clipped = axis.intersection(roi)
        for part in line_parts(clipped):
            if part.length < 60 or not part.is_simple:
                continue
            envelope = part.buffer(scale*.58, cap_style='flat', join_style='round').intersection(roi)
            p, m, surface = evidence.values(selected_xy[ids])
            result.append(AxisProposal(part, envelope, scale, float(supported[ids].mean()),
                dict(raw_support_fraction=float(supported[ids].mean()),
                     mean_flank_contrast=float(np.mean(contrast[ids, selected[ids]])),
                     mean_interior_spread=float(np.mean(spread[ids, selected[ids]])),
                     sam_positive_mean=float(np.nanmean(p)) if np.isfinite(p).any() else None,
                     molra_positive_mean=float(np.nanmean(m)) if np.isfinite(m).any() else None,
                     auxiliary_surface_support=float(surface.mean()),
                     scale_source='existing_side_axis_separation_or_cached_width',
                     width_recomputed=False), [key for _, _, key in group]))
    return result


def propose_axes(edges, evidence, roi):
    levels = {e.level for e in edges}
    if len(levels) > 1:
        all_proposals, all_reports = [], []
        for level in sorted(levels):
            proposals, reports = propose_axes([e for e in edges if e.level == level], evidence, roi)
            all_proposals.extend(proposals)
            all_reports.extend(reports)
        return all_proposals, all_reports
    evidence.network_prior = union_all([e.axis for e in edges])
    groups, reports = seed_groups(edges, evidence, roi)
    proposals = [p for group in groups for factor in (1., .75, 1.5)
                 for p in fit_medial([(a, scale*factor, key) for a, scale, key in group], evidence, roi)]
    # Select one proposal for each overlapping continuation using raw support.
    chosen = []
    for p in sorted(proposals, key=lambda p: (-p.axis.length*p.support, p.key)):
        if any(p.axis.intersection(q.axis.buffer(min(p.band_scale, q.band_scale)/3)).length > .7*p.axis.length for q in chosen):
            continue
        chosen.append(p)
    # Complete a supported junction approach to the other medial axis. This is
    # geometric intersection of continued road directions, not endpoint snapping.
    for p in chosen:
        coords = list(p.axis.coords)
        for end in (0, -1):
            tip = np.asarray(coords[end])
            inner = np.asarray(p.axis.interpolate(min(12., p.axis.length/3) if end == 0 else max(0., p.axis.length-12.)).coords[0])
            outward = (tip-inner)/max(np.linalg.norm(tip-inner), 1e-9)
            hits = []
            for q in chosen:
                if q is p or abs(direction(p.axis)@direction(q.axis)) > .8:
                    continue
                # The crossing road's width controls the intersection footprint.
                reach = max(25., p.band_scale+q.band_scale)
                ray = LineString([tip, tip+outward*reach])
                hit = ray.intersection(q.axis)
                if hit.geom_type == 'Point' and roi.covers(hit):
                    probes = np.array([ray.interpolate(s).coords[0] for s in np.linspace(0, Point(tip).distance(hit), 20)])
                    gray = evidence.image.sample(evidence.image.gray, probes)
                    anchor_points = np.array([p.axis.interpolate(s).coords[0] for s in
                        np.linspace(0, min(30., p.axis.length), 15)]) if end == 0 else np.array([
                        p.axis.interpolate(s).coords[0] for s in np.linspace(max(0., p.axis.length-30), p.axis.length, 15)])
                    anchor = np.nanmedian(evidence.image.sample(evidence.image.gray, anchor_points))
                    if np.isfinite(gray).all() and np.mean(abs(gray-anchor) < .3) >= .85:
                        hits.append(hit)
            if hits:
                hit = min(hits, key=lambda point: Point(tip).distance(point))
                coords = [hit.coords[0]]+coords if end == 0 else coords+[hit.coords[0]]
        p.axis = LineString(coords)
        p.envelope = p.axis.buffer(p.band_scale*.58, cap_style='flat').intersection(roi)
    for proposal in chosen:
        proposal.level = next(iter(levels))
    return chosen, reports


def replace_corridors(frame, proposals, roi, evidence=None):
    """Preserve every original piece outside accepted edit corridors exactly."""
    if not proposals:
        return frame.copy(), [], []
    from engine.canonical_road_surface import _level
    import pandas as pd
    levels = frame.apply(_level, axis=1)
    if len(set(levels)) > 1 or any(p.level != levels.iloc[0] for p in proposals):
        parts, all_edits, all_contacts = [], [], []
        for level in sorted(set(levels)):
            part, edits, contacts = replace_corridors(frame[levels == level], [p for p in proposals if p.level == level], roi, evidence)
            parts.append(part)
            all_edits.extend(edits)
            all_contacts.extend(contacts)
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=frame.crs), all_edits, all_contacts
    domain = union_all([p.envelope for p in proposals])
    rows, edits, contacts = [], [], []
    junctions = []
    for i, p in enumerate(proposals):
        for q in proposals[:i]:
            hit = p.axis.intersection(q.axis)
            if hit.geom_type == 'Point' and abs(direction(p.axis)@direction(q.axis)) < .8:
                junctions.append((hit, 1.25*max(p.band_scale, q.band_scale)))
    # A junction is an area, not merely two intersecting strips: turning curbs
    # outside the strips otherwise survive as a second false crossing/short loop.
    junction_domains = [point.buffer(radius).intersection(roi) for point, radius in junctions]
    if junction_domains:
        domain = union_all([domain]+junction_domains)
        edits.append(dict(action='rebuild_junction_domain', domain_wkt=union_all(junction_domains).wkt,
            reason='replace_boundary_turns_at_crossing_rgb_medial_axes_keep_external_arms'))
    if evidence is not None:
        from rgb_route import ribbon_evidence
        ribbons = []
        for proposal in proposals:
            search = proposal.axis.buffer(proposal.band_scale, cap_style='flat').intersection(roi)
            for _, row in frame[frame.intersects(search)].iterrows():
                for part in line_parts(row.geometry.intersection(search)):
                    if part.length < 12 or abs(direction(part)@direction(proposal.axis)) < .95:
                        continue
                    _, xy, _, _ = stations(part, 3)
                    central = np.array([proposal.axis.interpolate(proposal.axis.project(Point(point))).coords[0] for point in xy])
                    separation = np.linalg.norm(central-xy, axis=1)
                    if np.median(separation) < proposal.band_scale*.3:
                        continue
                    fractions = np.linspace(.2, 1., 25)
                    across = xy[:, None, :]*(1-fractions[None, :, None])+central[:, None, :]*fractions[None, :, None]
                    gray = evidence.image.sample(evidence.image.gray, across.reshape(-1, 2)).reshape(len(xy), -1)
                    valid = np.isfinite(gray).all(axis=1)
                    # A sustained separator between two genuine roads breaks this
                    # continuity test. A planted curb at the outer edge does not.
                    continuous = valid & (np.mean(abs(gray-gray[:, -1:]) < .2, axis=1) >= .8)
                    old_score = ribbon_evidence(part, float(row.width_m), evidence)['raw_rgb_support']
                    medial_score = ribbon_evidence(LineString(central), proposal.band_scale, evidence)['raw_rgb_support']
                    if continuous.mean() < .8 or medial_score < .65 or medial_score-old_score < .25:
                        continue
                    ribbons.append(part.buffer(.05))
                    edits.append(dict(action='identify_boundary_alias', old_wkt=part.wkt,
                        evidence=dict(single_ribbon_cross_section_fraction=float(continuous.mean()),
                                      old_raw_rgb_support=old_score, medial_raw_rgb_support=medial_score),
                        reason='parallel_axis_on_edge_of_same_continuous_rgb_ribbon_replace_and_preserve_branch_contacts'))
        if ribbons:
            domain = union_all([domain]+ribbons)
    outside_rows = []
    for source, row in frame.iterrows():
        old = row.geometry
        if not old.intersects(domain):
            rows.append(row.to_dict())
            continue
        inside = old.intersection(domain)
        outside = old.difference(domain)
        edits.append(dict(source_id=int(row.source_id), action='replace', old_wkt=inside.wkt,
                          supporting_candidates=[p.key for p in proposals if old.distance(p.axis) <= p.band_scale*2],
                          reason='replace_road_side_network_with_rgb_medial_axis'))
        for part in line_parts(outside):
            data = row.to_dict()
            data['geometry'] = part
            outside_rows.append(data)
    # Returning fragments just outside an edit envelope used to turn into small
    # rectangular false loops after both ends were reattached. Remove only whole
    # detached sliver components; components reaching genuine internal roads stay.
    graph = nx.Graph()
    nodes = {}
    exact_nodes = {}
    attachments = []
    for i, data in enumerate(outside_rows):
        line = data['geometry']
        ends = [tuple(np.round(line.coords[k], 6)) for k in (0, -1)]
        for node, coord in zip(ends, (line.coords[0], line.coords[-1])):
            exact_nodes[node] = Point(coord)
        graph.add_edge(*ends, item=i)
        for node in ends:
            nodes.setdefault(node, []).append(i)
    slivers = set()
    preserved = union_all([r['geometry'] for r in rows])
    for component in nx.connected_components(graph):
        ids = {i for node in component for i in nodes[node]}
        lines = [outside_rows[i]['geometry'] for i in ids]
        if any(Point(n).distance(preserved) < .02 for n in component):
            continue
        touches = sum(Point(n).distance(domain.boundary) < .02 for n in component)
        if touches < 2 or sum(line.length for line in lines) > 80:
            continue
        sample = np.concatenate([stations(line, 2)[1] for line in lines])
        if distance(points(sample), domain).max() <= 4:
            slivers.update(ids)
            edits.append(dict(action='delete_returning_sliver', old_wkt=union_all(lines).wkt,
                              reason='detached_boundary_return_within_4m_not_an_internal_road'))
    if evidence is not None:
        from rgb_route import ribbon_evidence
        for component in nx.connected_components(graph):
            boundary_nodes = [n for n in component if Point(n).distance(domain.boundary) < .02]
            for a_index, a in enumerate(boundary_nodes):
                for b in boundary_nodes[:a_index]:
                    path = nx.shortest_path(graph, a, b)
                    ids = {graph[u][v]['item'] for u, v in zip(path, path[1:])}
                    if ids & slivers or not ids:
                        continue
                    lines = [outside_rows[i]['geometry'] for i in ids]
                    if sum(line.length for line in lines) > 100:
                        continue
                    samples = np.concatenate([stations(line, 2)[1] for line in lines])
                    margin = float(distance(points(samples), domain).max())
                    if margin > 6*evidence.image.resolution:
                        continue
                    scores = [ribbon_evidence(outside_rows[i]['geometry'], float(outside_rows[i]['width_m']), evidence)['raw_rgb_support'] for i in ids]
                    if max(scores) >= .5:
                        continue
                    slivers.update(ids)
                    for node in path[1:-1]:
                        if any(k not in ids for k in nodes[node]) or Point(node).distance(preserved) < .02:
                            point = exact_nodes[node]
                            closest = min(proposals, key=lambda p: point.distance(p.axis))
                            target = closest.axis.interpolate(closest.axis.project(point))
                            attachments.append((point, target, outside_rows[next(iter(ids))]))
                    edits.append(dict(action='remove_boundary_return_keep_branches', old_wkt=union_all(lines).wkt,
                        evidence=dict(old_raw_rgb_support=scores, distance_to_accepted_domain_m=margin),
                        reason='unsupported_boundary_return_duplicates_medial_path_external_branches_reattached'))
    for i, data in enumerate(outside_rows):
        if i in slivers:
            continue
        rows.append(data)
        part = data['geometry']
        for end in (0, -1):
                point = Point(part.coords[end])
                if point.distance(domain.boundary) < .02:
                    closest = min(proposals, key=lambda p: point.distance(p.axis))
                    target = closest.axis.interpolate(closest.axis.project(point))
                    for center, radius in junctions:
                        if target.distance(center) < radius:
                            target = center
                    if point.distance(target) > .1:
                        attachments.append((point, target, data))
    # Two nearby cut contacts often belong to the same fragmented branch. Join
    # their reattachments at a shared stem instead of creating parallel new axes.
    groups = []
    for item in attachments:
        point, target, data = item
        for group in groups:
            if all(point.distance(p) <= min(4., float(data['width_m'])*.5) and target.distance(t) <= 4.
                   for p, t, _ in group):
                group.append(item)
                break
        else:
            groups.append([item])
    for group in groups:
        source_mean = np.mean([p.coords[0] for p, _, _ in group], axis=0)
        target_mean = Point(np.mean([t.coords[0] for _, t, _ in group], axis=0))
        closest = min(proposals, key=lambda p: target_mean.distance(p.axis))
        target = closest.axis.interpolate(closest.axis.project(target_mean))
        hub = Point(source_mean*.75+np.asarray(target.coords[0])*.25) if len(group) > 1 else target
        for point, _, data in group:
            connector = LineString([point, hub])
            if connector.length > .01:
                rows.append(dict(data, geometry=connector, repair_role='branch_reattachment'))
                contacts.append(dict(source_id=int(data['source_id']), old_contact=list(point.coords[0]),
                    new_contact=list(hub.coords[0]), geometry_wkt=connector.wkt, shared_stem=len(group) > 1))
        if hub.distance(target) > .01:
            connector = LineString([hub, target])
            rows.append(dict(group[0][2], geometry=connector, repair_role='branch_shared_stem'))
            contacts.append(dict(source_id=int(group[0][2]['source_id']), geometry_wkt=connector.wkt,
                                 new_contact=list(target.coords[0]), shared_stem=True))
    for i, proposal in enumerate(proposals):
        proposal.axis = pin_axis_contacts(proposal.axis,
            [c['new_contact'] for c in contacts if 'new_contact' in c]+[p.coords[0] for p, _ in junctions])
        nearest = frame.distance(proposal.axis.centroid).idxmin()
        row = frame.loc[nearest].to_dict()
        row.update(geometry=proposal.axis, source_id=-i-1, repair_role='rgb_medial_axis')
        rows.append(row)
        edits.append(dict(candidate_id=proposal.key, action='add_medial_axis', new_wkt=proposal.axis.wkt,
                          envelope_wkt=proposal.envelope.wkt, evidence=proposal.evidence,
                          seed_ids=proposal.seed_ids, reason='continuous_raw_rgb_ribbon_medial_axis'))
    result = gpd.GeoDataFrame(rows, geometry='geometry', crs=frame.crs)
    return result, edits, contacts
