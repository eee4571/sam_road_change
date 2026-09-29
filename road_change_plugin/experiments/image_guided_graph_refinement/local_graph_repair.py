"""Local alias-loop contraction with all external branch contacts preserved."""
import numpy as np
import geopandas as gpd
from shapely import union_all, points
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, substring, linemerge
from shapely.strtree import STRtree

from graph import build
from evidence import stations
from engine.canonical_road_surface import _adjacency
from medial_repair import line_parts, direction


def thin_regions(network, roi):
    """Adjacent alias faces share one repair domain, so they cannot leave twin hubs."""
    def eligible(face):
        return (face.geom_type == 'Polygon' and roi.buffer(-2).covers(face) and
                face.length <= 65 and 2*face.area/max(face.length, 1e-9) <= 3.5)
    regions = sorted([p for p in polygonize(network) if eligible(p)], key=lambda p: p.area)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(regions):
            for j in range(i):
                b = regions[j]
                if a.boundary.intersection(b.boundary).length <= .001:
                    continue
                combined = a.union(b)
                if eligible(combined):
                    regions[j] = combined
                    regions.pop(i)
                    changed = True
                    break
            if changed:
                break
    return sorted(regions, key=lambda p: p.area)


def clean_local(frame, evidence, roi, rounds=8):
    """Replace thin two-route aliases; never remove a normal road loop by size alone."""
    untouched = frame[~frame.intersects(roi.buffer(2))]
    local = frame[frame.intersects(roi.buffer(2))].reset_index(drop=True)
    audits = []
    for iteration in range(rounds):
        edges = build(local)
        nodes = _adjacency(edges, set(range(len(edges))))
        tree = STRtree([e.axis for e in edges])
        removed, additions = set(), []
        for level in {e.level for e in edges}:
            network = union_all([e.axis for e in edges if e.level == level])
            faces = thin_regions(network, roi)
            for face in faces:
                if not roi.buffer(-2).covers(face) or face.length > 65 or 2*face.area/max(face.length, 1e-9) > 3.5:
                    continue
                ids = {int(k) for k in tree.query(face.buffer(.001), predicate='intersects')
                       if edges[int(k)].level == level and face.buffer(.001).covers(edges[int(k)].axis)}
                if not ids or ids & removed:
                    continue
                candidate_nodes = {n for i in ids for n in edges[i].nodes}
                anchors = [n for n in candidate_nodes if any(i not in ids for i, _ in nodes[n])]
                if len(anchors) < 2:
                    continue  # do not delete an isolated loop/turnaround as a spur
                anchors = sorted(anchors, key=lambda n: (n[1], n[2]))
                if len(anchors) == 2:
                    ring = LineString(face.exterior.coords)
                    loc = sorted(ring.project(Point(n[1:])) for n in anchors)
                    first = substring(ring, loc[0], loc[1])
                    tail = list(substring(ring, loc[1], ring.length).coords)
                    head = list(substring(ring, 0, loc[0]).coords)
                    second = LineString((tail+head[1:])[::-1])
                    fraction = np.linspace(0, 1, max(5, int(max(first.length, second.length)/2)))
                    xy = (np.array([first.interpolate(t, normalized=True).coords[0] for t in fraction])+
                          np.array([second.interpolate(t, normalized=True).coords[0] for t in fraction]))/2
                    ordered = sorted(anchors, key=lambda n: ring.project(Point(n[1:])))
                    xy[0], xy[-1] = ordered[0][1:], ordered[1][1:]
                    replacement = LineString(xy).simplify(.35)
                    new_lines = [replacement]
                else:
                    # Tiny alias cycles at a junction become one hub, retaining
                    # each externally connected arm; no road-class/level crossing.
                    center = np.mean([n[1:] for n in anchors], axis=0)
                    new_lines = [LineString([n[1:], center]) for n in anchors if Point(n[1:]).distance(Point(center)) > .01]
                samples = np.concatenate([stations(line, 1.5)[1] for line in new_lines])
                gray = evidence.image.sample(evidence.image.gray, samples)
                old_samples = np.concatenate([stations(edges[k].axis, 1.5)[1] for k in ids])
                old_gray = evidence.image.sample(evidence.image.gray, old_samples)
                if not np.isfinite(gray).all() or not np.isfinite(old_gray).all():
                    continue
                # Subpixel twin paths need no semantic classification. For larger
                # thin loops, the replacement must remain in the observed local
                # appearance range; SAM absence is not used as negative evidence.
                lo, hi = np.quantile(old_gray, [.05, .95])
                continuity = float(np.mean((gray >= lo-.15) & (gray <= hi+.15)))
                if continuity < .85:
                    continue
                source = edges[min(ids)].sources[0]
                for line in new_lines:
                    row = local.iloc[source].to_dict()
                    row.update(geometry=line, repair_role='thin_loop_center_axis')
                    additions.append(row)
                removed.update(ids)
                audits.append(dict(action='collapse_thin_cycle', iteration=iteration+1,
                    old_wkt=union_all([edges[k].axis for k in ids]).wkt,
                    new_wkt=union_all(new_lines).wkt, external_contacts=[list(n[1:]) for n in anchors],
                    thinness_m=2*face.area/face.length, rgb_continuity=continuity,
                    reason='two_nearby_alias_routes_or_small_junction_cycle_with_preserved_external_contacts'))
        # Remove only sub-resolution terminal teeth, never generic short roads.
        for node, ends in nodes.items():
            if len(ends) != 1:
                continue
            i, end = ends[0]
            if (i in removed or edges[i].axis.length > 2*evidence.image.resolution or
                    len(nodes[edges[i].nodes[1-end]]) < 3 or not roi.buffer(-2).covers(edges[i].axis)):
                continue
            removed.add(i)
            audits.append(dict(action='delete_subresolution_tooth', old_wkt=edges[i].axis.wkt,
                               reason='terminal_tooth_shorter_than_two_analysis_pixels'))
        if not removed:
            break
        rows = []
        for i, edge in enumerate(edges):
            if i not in removed:
                row = local.iloc[edge.sources[0]].to_dict()
                row['geometry'] = edge.axis
                rows.append(row)
        local = gpd.GeoDataFrame(rows+additions, geometry='geometry', crs=frame.crs)
    import pandas as pd
    return gpd.GeoDataFrame(pd.concat([untouched, local], ignore_index=True), geometry='geometry', crs=frame.crs), audits


def seed_chains(edges):
    """Degree-two chains only, without altering the graph used for branch repair."""
    from engine.canonical_road_surface import Edge, _node
    result = []
    for level in {e.level for e in edges}:
        lines = [e.axis for e in edges if e.level == level]
        tree = STRtree(lines)
        refs = [e for e in edges if e.level == level]
        network = union_all(lines)
        merged = linemerge(network) if network.geom_type == 'MultiLineString' else network
        for line in line_parts(merged):
            ids = tree.query(line, predicate='intersects')
            width = float(np.median([np.median(refs[i].widths) for i in ids]))
            result.append(Edge(line, np.array([0., line.length]), np.array([width, width]), np.zeros(2),
                tuple(sorted({s for i in ids for s in refs[i].sources})), level,
                (_node(level, line.coords[0]), _node(level, line.coords[-1]))))
    return result


def smooth_local(frame, evidence, roi):
    """Bounded RGB-consistent smoothing with pinned graph nodes and no new crossings."""
    from engine.road_axis_quality import _pinned_smooth
    untouched = frame[~frame.intersects(roi.buffer(2))]
    local = frame[frame.intersects(roi.buffer(2))].reset_index(drop=True)
    edges = build(local)
    tree = STRtree([e.axis for e in edges])
    rows, audit, accepted_lines = [], [], []
    for index, edge in enumerate(edges):
        row = local.iloc[edge.sources[0]].to_dict()
        row['geometry'] = edge.axis
        if edge.axis.length >= 20 and roi.buffer(-2).covers(edge.axis) and len(edge.axis.coords) >= 4:
            corridor = edge.axis.buffer(float(np.median(edge.widths))*.35, cap_style='flat')
            near_parallel = any(abs(direction(edges[int(k)].axis)@direction(edge.axis)) > .9 and
                edges[int(k)].axis.intersection(corridor).length > min(3., edges[int(k)].axis.length*.5)
                for k in tree.query(corridor, predicate='intersects') if int(k) != index)
            if near_parallel:
                rows.append(row)
                continue  # smoothing must not move one branch towards a nearby parallel axis
            _, xy, _, _ = stations(edge.axis, 2)
            fitted = _pinned_smooth(xy, 1.5)
            displacement = np.linalg.norm(fitted-xy, axis=1)
            proposed = LineString(fitted).simplify(.1)
            old_gray = evidence.image.sample(evidence.image.gray, xy)
            new_gray = evidence.image.sample(evidence.image.gray, fitted)
            rgb_same = float(np.mean(np.isfinite(old_gray) & np.isfinite(new_gray) & (abs(old_gray-new_gray) < .15)))
            endpoints = union_all([Point(xy[0]).buffer(.05), Point(xy[-1]).buffer(.05)])
            crossing = any(not proposed.intersection(edges[int(k)].axis).difference(endpoints).is_empty
                for k in tree.query(proposed, predicate='intersects') if int(k) != index)
            crossing |= any(not proposed.intersection(line).difference(endpoints).is_empty for line in accepted_lines)
            if (proposed.is_simple and not crossing and .25 <= displacement.max() <= min(2., .15*np.median(edge.widths))
                    and rgb_same >= .9 and proposed.length <= edge.axis.length and proposed.length >= .95*edge.axis.length):
                row['geometry'] = proposed
                row['repair_role'] = 'rgb_pinned_smoothing'
                accepted_lines.append(proposed)
                audit.append(dict(action='smooth', old_wkt=edge.axis.wkt, new_wkt=proposed.wkt,
                    evidence=dict(maximum_displacement_m=float(displacement.max()), rgb_appearance_preserved_fraction=rgb_same),
                    reason='bounded_rgb_consistent_smoothing_pinned_nodes_no_new_crossings'))
        rows.append(row)
    if not audit:
        return frame, []
    import pandas as pd
    return gpd.GeoDataFrame(pd.concat([untouched, gpd.GeoDataFrame(rows, crs=frame.crs)], ignore_index=True), crs=frame.crs), audit
