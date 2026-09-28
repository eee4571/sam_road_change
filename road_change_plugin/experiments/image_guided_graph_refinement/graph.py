"""Shared geometry candidates for the B/C ablation; existing engine graph primitives."""
from dataclasses import dataclass
import hashlib

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely import union_all
from shapely.geometry import LineString, Point
from shapely.ops import polygonize
from shapely.strtree import STRtree

import bootstrap
from engine.canonical_road_surface import (
    _build_graph, _adjacency, _outward, _collapse_duplicates,
)


@dataclass
class Candidate:
    kind: str
    ids: tuple
    axis: object
    width: float
    geometry: dict

    @property
    def key(self):
        return self.kind + "_" + hashlib.sha256(self.axis.wkb).hexdigest()[:16]


def build(frame):
    profiles = [(r.geometry, np.array([0., r.geometry.length]), np.array([r.width_m, r.width_m]))
                for _, r in frame.iterrows()]
    metadata = frame.drop(columns="geometry").to_dict("records")
    return _build_graph(profiles, [np.zeros(2) for _ in profiles], metadata)


def as_frame(edges, crs):
    return gpd.GeoDataFrame([dict(edge_id=i, width_m=float(np.median(e.widths)),
                                source_ids=",".join(map(str, e.sources)), layer=e.level[0],
                                bridge=e.level[1], tunnel=e.level[2], geometry=e.axis)
                             for i, e in enumerate(edges)], geometry="geometry", crs=crs)


def candidates(edges, roi, config, kinds=("spur", "gap", "duplicate"), focus=None):
    nodes = _adjacency(edges, set(range(len(edges))))
    result = []
    # Only whole original arms inside the ROI are eligible. Context beyond the ROI
    # remains in the graph, so crop boundaries cannot create artificial tips.
    def eligible(axis):
        return roi.buffer(-.01).covers(axis) and (focus is None or focus.intersects(axis))
    if "spur" in kinds:
        for node, ends in nodes.items():
            if len(ends) != 1:
                continue
            k, end = ends[0]
            ids, coords, seen = [], [], set()
            while k not in seen:
                seen.add(k)
                edge = edges[k]
                ids.append(k)
                xy = list(edge.axis.coords)
                if end:
                    xy.reverse()
                coords.extend(xy if not coords else xy[1:])
                far = edge.nodes[1-end]
                if len(nodes[far]) != 2:
                    break
                k, end = next(pair for pair in nodes[far] if pair[0] != k)
            axis = LineString(coords)
            width = float(np.median(np.concatenate([edges[i].widths for i in ids])))
            if len(nodes[far]) >= 3 and axis.length <= min(30., 3*width) and eligible(axis):
                result.append(Candidate("spur", tuple(ids), axis, width,
                                        dict(length_m=axis.length, threshold_m=min(30., 3*width))))
    if "gap" in kinds:
        tips = [(n, ends[0]) for n, ends in nodes.items() if len(ends) == 1]
        pts = [Point(n[1:]) for n, _ in tips]
        tree = STRtree(pts)
        for a, (node, (i, ie)) in enumerate(tips):
            for b in tree.query(pts[a], predicate="dwithin", distance=config["max_gap_m"]):
                if b <= a:
                    continue
                other, (j, je) = tips[b]
                if i == j or edges[i].level != edges[j].level:
                    continue
                line = LineString([node[1:], other[1:]])
                if line.length < 1 or not eligible(line):
                    continue
                d = np.subtract(other[1:], node[1:])/line.length
                ai, aj = -_outward(edges[i], ie, 8), -_outward(edges[j], je, 8)
                facing = min(float(ai@d), float(aj@-d))
                lateral = line.length*max(abs(ai[0]*d[1]-ai[1]*d[0]), abs(aj[0]*d[1]-aj[1]*d[0]))
                wi, wj = edges[i].widths[0 if ie == 0 else -1], edges[j].widths[0 if je == 0 else -1]
                if facing < config["direction_cosine"] or lateral > config["max_lateral_m"] or max(wi, wj)/min(wi, wj) > config["max_width_ratio"]:
                    continue
                result.append(Candidate("gap", (i, j), line, float((wi+wj)/2),
                    dict(distance_m=line.length, direction_cosine=facing, lateral_m=float(lateral),
                         width_ratio=float(max(wi, wj)/min(wi, wj)), tip_nodes=[node, other],
                         approach_directions=[ai.tolist(), (-aj).tolist()])))
    if "duplicate" in kinds and edges:
        # Dry-run the existing duplicate rules; do not use their returned geometry.
        profiles = [(e.axis, e.stations, e.widths) for e in edges]
        metadata = [dict(layer=e.level[0], bridge=e.level[1], tunnel=e.level[2]) for e in edges]
        _, _, reports = _collapse_duplicates(profiles, [e.quality for e in edges], metadata)
        for report in reports:
            i, j = report["source_feature"], report["canonical_source"]
            if eligible(edges[i].axis) and eligible(edges[j].axis):
                result.append(Candidate("duplicate", (i, j), edges[i].axis,
                                        float(np.median(edges[i].widths)), report))
    return sorted(result, key=lambda c: (c.kind, c.axis.length, c.key))


def stats(edges, roi, config):
    nodes = _adjacency(edges, set(range(len(edges))))
    # Full context graph gives true degrees; report only nodes/edges inside ROI.
    selected = [e for e in edges if e.axis.intersects(roi)]
    local = nx.MultiGraph()
    for edge in selected:
        clipped = edge.axis.intersection(roi)
        parts = [clipped] if clipped.geom_type == "LineString" else list(getattr(clipped, "geoms", []))
        for part in parts:
            if part.geom_type == "LineString" and part.length > 1e-7:
                local.add_edge((edge.level, *np.round(part.coords[0], 6)),
                               (edge.level, *np.round(part.coords[-1], 6)))
    proposals = candidates(edges, roi, config)
    loops = [p for level in {e.level for e in selected}
             for p in polygonize(union_all([e.axis for e in selected if e.level == level]))
             if p.length <= 60 and roi.covers(p)]
    return dict(component_count=nx.number_connected_components(local),
                degree_1_endpoints=sum(len(ends) == 1 and roi.covers(Point(n[1:])) for n, ends in nodes.items()),
                spur_count=sum(c.kind == "spur" for c in proposals),
                spur_length_m=sum(c.axis.length for c in proposals if c.kind == "spur"),
                near_duplicate_count=sum(c.kind == "duplicate" for c in proposals),
                near_duplicate_length_m=sum(c.geometry["overlap_m"] for c in proposals if c.kind == "duplicate"),
                suspicious_gap_pairs=sum(c.kind == "gap" for c in proposals),
                small_loop_count=len(loops),
                complex_junction_count=sum(len(v) >= 4 and roi.covers(Point(n[1:])) for n, v in nodes.items()),
                total_length_m=sum(e.axis.intersection(roi).length for e in selected),
                small_loops=[p.wkt for p in loops],
                complex_junctions=[list(n[1:]) for n, v in nodes.items() if len(v) >= 4 and roi.covers(Point(n[1:]))])
