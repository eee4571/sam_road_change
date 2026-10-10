"""Local graph transactions: preserve outside geometry, contacts and supported roads."""
import networkx as nx
import numpy as np
from shapely import union_all, points, distance, set_precision
from shapely.geometry import Point
from shapely.strtree import STRtree

from graph import build, candidates
from color_ribbon import descriptor_scales
from shapely.ops import polygonize
from evidence import stations
from rgb_route import ribbon_evidence
from engine.canonical_road_surface import _level, _adjacency


def components(edges):
    graph = nx.Graph()
    for i, edge in enumerate(edges):
        graph.add_edge(*edge.nodes)
    lookup = {n: i for i, members in enumerate(nx.connected_components(graph)) for n in members}
    return lookup


def validate(before, after, window, evidence, protect_supported=True):
    """No count-only acceptance: hard graph invariants plus independent RGB protection."""
    failures = []
    old_edges, new_edges = build(before), build(after)
    old_nodes = _adjacency(old_edges, set(range(len(old_edges))))
    new_nodes = _adjacency(new_edges, set(range(len(new_edges))))
    old_components, new_components = components(old_edges), components(new_edges)
    old_levels, new_levels = before.apply(_level, axis=1), after.apply(_level, axis=1)
    lost_supported = []
    outside_delta = 0.
    for level in set(old_levels) | set(new_levels):
        old = union_all(before[old_levels == level].geometry)
        new = union_all(after[new_levels == level].geometry)
        a, b = old.difference(window), new.difference(window)
        outside_delta += a.difference(b.buffer(.001)).length+b.difference(a.buffer(.001)).length
        if protect_supported:
            for _, row in before[(old_levels == level) & before.intersects(window)].iterrows():
                line = row.geometry
                if line.length < 8 or line.difference(new.buffer(.1)).length < 3:
                    continue
                support = max(ribbon_evidence(line, scale, evidence)['raw_rgb_support'] for scale in descriptor_scales(row.width_m))
                if support < .65:
                    continue
                _, xy, _, _ = stations(line, 3)
                lost = float(np.mean(distance(points(xy), new) > max(1.5, row.width_m*.35)))
                if lost > .15:
                    lost_supported.append(dict(axis_wkt=line.wkt, rgb_support=support, lost_fraction=lost))
    if outside_delta > .02:
        failures.append('outside_window_changed')
    if lost_supported:
        failures.append('independently_supported_road_lost')
    if any(not g.is_valid or not g.is_simple or g.length < .001 for g in after.geometry):
        failures.append('invalid_or_self_intersecting_geometry')
    # Every graph node outside the editable window is pinned. Preserve all
    # pairwise connectivity equivalence within each old connected component.
    anchors = [n for n in old_nodes if not window.buffer(-.01).covers(Point(n[1:]))]
    missing = [n for n in anchors if n not in new_nodes]
    if missing:
        failures.append('fixed_contact_missing')
    partitions = {}
    for n in anchors:
        partitions.setdefault(old_components[n], set()).add(new_components.get(n, ('missing', n)))
    if any(len(v) > 1 for v in partitions.values()):
        failures.append('previous_external_connections_split')
    if len(set(new_components.values())) > len(set(old_components.values())):
        failures.append('new_disconnected_component')
    # An added crossing must either already be a junction or be an explicitly
    # pinned contact on an old road. New high-degree nodes elsewhere are vetoed.
    old_junctions = [Point(n[1:]) for n, v in old_nodes.items() if len(v) >= 3]
    old_tips = [Point(n[1:]) for n, v in old_nodes.items() if len(v) == 1]
    references = union_all(old_junctions+old_tips)
    unexpected = [list(n[1:]) for n, v in new_nodes.items() if len(v) >= 4 and n not in old_nodes
                  and Point(n[1:]).distance(references) > 25]
    if unexpected:
        failures.append('new_remote_crossing')
    old_tip_geometry = union_all(old_tips)
    new_tips = [list(n[1:]) for n, v in new_nodes.items() if len(v) == 1 and n not in old_nodes
                and window.buffer(-.01).covers(Point(n[1:])) and Point(n[1:]).distance(old_tip_geometry) > .1]
    if new_tips:
        failures.append('new_internal_dangling_endpoint')
    def anomalies(edges):
        loops = sum(1 for level in {e.level for e in edges}
                    for p in polygonize(union_all([set_precision(e.axis, .001) for e in edges if e.level == level]))
                    if p.length <= 60 and window.covers(p))
        duplicates = candidates(edges, window, {}, ('duplicate',))
        return dict(small_loops=loops, duplicate_length_m=sum(c.geometry['overlap_m'] for c in duplicates))
    before_anomalies, after_anomalies = anomalies(old_edges), anomalies(new_edges)
    if after_anomalies['small_loops'] > before_anomalies['small_loops']:
        failures.append('new_small_loop')
    if after_anomalies['duplicate_length_m'] > before_anomalies['duplicate_length_m']+.1:
        failures.append('duplicate_length_increased')
    return dict(passed=not failures, failures=failures, outside_delta_m=outside_delta,
                loop_detection_precision_m=.001,
                fixed_contact_count=len(anchors), missing_contacts=[list(n[1:]) for n in missing],
                supported_roads_lost=lost_supported, unexpected_crossings=unexpected,
                new_internal_tips=new_tips, anomalies_before=before_anomalies, anomalies_after=after_anomalies,
                components_before=len(set(old_components.values())), components_after=len(set(new_components.values())))
