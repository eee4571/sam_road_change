"""Assemble already detected Auto changes along final road topology.

This module consumes geometry and widths only. It neither tests existence nor
uses probability, imagery, or ground truth to accept/reject a seed change.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import from_wkt, line_merge, make_valid, union_all
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree


@dataclass(frozen=True)
class AssemblyConfig:
    maximum_gap_m: float = 80.0
    endpoint_snap_m: float = 1.0
    axis_projection_m: float = 4.0
    maximum_heading_degrees: float = 35.0
    maximum_path_turn_degrees: float = 50.0
    straight_track_offset_m: float = 3.0


def _parts(geometry, family="LineString"):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == family:
        return [geometry]
    return [p for child in getattr(geometry, "geoms", ()) for p in _parts(child, family)]


def polygonal(geometry):
    return union_all(_parts(make_valid(geometry), "Polygon"))


def _unit(vector):
    return np.asarray(vector)/max(float(np.linalg.norm(vector)), 1e-12)


class _Groups:
    def __init__(self, count):
        self.parent = list(range(count))

    def root(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class FinalNetwork:
    def __init__(self, frame, config):
        self.config = config
        self.edges = _parts(line_merge(union_all(frame.geometry.values)))
        self.tree = STRtree(self.edges)
        coordinates = np.array([edge.coords[end] for edge in self.edges for end in (0, -1)])
        groups = _Groups(len(coordinates))
        if len(coordinates):
            for a, b in sorted(cKDTree(coordinates).query_pairs(config.endpoint_snap_m)):
                groups.join(a, b)
        self.nodes = [(groups.root(2*i), groups.root(2*i+1)) for i in range(len(self.edges))]
        self.adjacency = {}
        for edge_id, (a, b) in enumerate(self.nodes):
            length = self.edges[edge_id].length
            self.adjacency.setdefault(a, []).append((b, edge_id, length))
            self.adjacency.setdefault(b, []).append((a, edge_id, length))

    def seed_axis(self, row):
        value = row.get("axis_wkt")
        if isinstance(value, str) and value:
            return from_wkt(value)
        # Cached pre-assembly runs did not store axes. Recover a longitudinal
        # interval on the final road with maximum seed coverage, not polygon PCA.
        width = max(float(row.get("width_bef", 0)), float(row.get("width_aft", 0)), 2.)
        margin = width/2 + .5 if row.change_typ in {"widened", "narrowed"} else .3
        support = row.geometry.buffer(margin)
        choices = []
        for edge_id in self.tree.query(support, predicate="intersects"):
            edge = self.edges[int(edge_id)]
            overlap = _parts(edge.intersection(support))
            if not overlap:
                continue
            covered = sum(p.length for p in overlap)
            stations = [edge.project(Point(c)) for part in overlap for c in (part.coords[0], part.coords[-1])]
            choices.append((covered, -edge.distance(row.geometry.centroid), int(edge_id), min(stations), max(stations)))
        if not choices:
            return None
        _, _, edge_id, start, end = max(choices)
        return substring(self.edges[edge_id], start, end) if end-start > .01 else None

    def project(self, point, heading):
        candidates = []
        for edge_id in self.tree.query(point, predicate="dwithin", distance=self.config.axis_projection_m):
            edge = self.edges[int(edge_id)]
            station = edge.project(point)
            a = edge.interpolate(max(0, station-5)); b = edge.interpolate(min(edge.length, station+5))
            cosine = abs(float(np.dot(_unit(np.array(b.coords[0])-np.array(a.coords[0])), heading)))
            if cosine >= .8:
                candidates.append((edge.distance(point)+2*(1-cosine), int(edge_id), station))
        if not candidates:
            return None
        _, edge_id, station = min(candidates)
        return edge_id, station

    def route(self, first, second):
        e1, s1 = first; e2, s2 = second
        if e1 == e2:
            part = substring(self.edges[e1], s1, s2)
            return part, []
        queue = []
        for side, node in enumerate(self.nodes[e1]):
            distance = s1 if side == 0 else self.edges[e1].length-s1
            part = substring(self.edges[e1], s1, 0 if side == 0 else self.edges[e1].length)
            heapq.heappush(queue, (distance, node, [list(part.coords)], [node], -1))
        best, solution = {}, None
        while queue:
            distance, node, parts, nodes, last_edge = heapq.heappop(queue)
            if distance > self.config.maximum_gap_m or distance >= best.get(node, float("inf")):
                continue
            best[node] = distance
            if node in self.nodes[e2]:
                side = self.nodes[e2].index(node)
                tail_distance = s2 if side == 0 else self.edges[e2].length-s2
                total = distance+tail_distance
                if total <= self.config.maximum_gap_m and (solution is None or total < solution[0]):
                    tail = substring(self.edges[e2], 0 if side == 0 else self.edges[e2].length, s2)
                    solution = total, parts+[list(tail.coords)], nodes
            for other, edge_id, length in self.adjacency[node]:
                if edge_id in (e1, e2, last_edge):
                    continue
                coords = list(self.edges[edge_id].coords)
                if self.nodes[edge_id][0] != node:
                    coords.reverse()
                heapq.heappush(queue, (distance+length, other, parts+[coords], nodes+[other], edge_id))
        if solution is None:
            return None
        coordinates = []
        for part in solution[1]:
            for point in part:
                if not coordinates or point != coordinates[-1]:
                    coordinates.append(point)
        if len(coordinates) < 2:
            return None
        junctions = [n for n in solution[2] if len(self.adjacency.get(n, ())) >= 3]
        return LineString(coordinates), junctions


def assemble_change_objects(changes, before_roads, after_roads, config=AssemblyConfig()):
    """Return assembled polygons, seed/bridge axes and explicit assembly audit.

    Every input seed belongs to exactly one output object, including unmappable
    seeds. Junction fill is derived only from same-class trajectories on both
    sides. Object grouping never depends on overlapping buffered polygons.
    """
    output_crs = changes.crs
    metric_crs = changes.estimate_utm_crs() if output_crs.is_geographic and len(changes) else output_crs
    if len(changes) and metric_crs.axis_info[0].unit_conversion_factor != 1:
        metric_crs = changes.estimate_utm_crs()
    seeds = changes.to_crs(metric_crs).reset_index(drop=True)
    # WKT axes supplied by the detector are in its metric CRS. Standalone callers
    # likewise provide metric axis WKT, with geometry determining output CRS.
    networks = {"before": FinalNetwork(before_roads.to_crs(metric_crs), config),
                "after": FinalNetwork(after_roads.to_crs(metric_crs), config)}
    groups = _Groups(len(seeds))
    axes, bridges, decisions = {}, [], []
    endpoints = []
    for seed_id, row in seeds.iterrows():
        network = networks["after" if row.change_typ in {"added", "widened"} else "before"]
        axis = network.seed_axis(row)
        if axis is None or axis.geom_type != "LineString" or axis.length < .1:
            continue
        axes[seed_id] = axis
        for side, station in enumerate((0., axis.length)):
            point = axis.interpolate(station)
            inner = axis.interpolate(min(12., axis.length) if side == 0 else max(0., axis.length-12.))
            heading = _unit(np.array(point.coords[0])-np.array(inner.coords[0]))
            projection = network.project(point, heading)
            if projection is not None:
                endpoints.append(dict(seed=seed_id, side=side, point=point, heading=heading,
                                      projection=projection, kind=row.change_typ, network=network))
    endpoint_tree = STRtree([p["point"] for p in endpoints])
    cos_heading = np.cos(np.radians(config.maximum_heading_degrees))
    candidates = []
    for first_id, first in enumerate(endpoints):
        for second_id in endpoint_tree.query(first["point"], predicate="dwithin", distance=config.maximum_gap_m):
            second_id = int(second_id)
            second = endpoints[second_id]
            if second_id <= first_id or first["seed"] == second["seed"] or first["kind"] != second["kind"]:
                continue
            vector = np.array(second["point"].coords[0])-np.array(first["point"].coords[0])
            distance = float(np.linalg.norm(vector))
            facing = -float(np.dot(first["heading"], second["heading"]))
            reason = ""
            if facing < cos_heading or (distance > 1 and min(np.dot(first["heading"], _unit(vector)),
                                                            np.dot(second["heading"], -_unit(vector))) < cos_heading):
                reason = "different_road_direction"
            lateral = abs(float(first["heading"][0]*vector[1]-first["heading"][1]*vector[0]))
            if not reason and facing > .98 and lateral > config.straight_track_offset_m:
                reason = "parallel_track_offset"
            routed = None if reason else first["network"].route(first["projection"], second["projection"])
            if not reason and routed is None:
                reason = "no_short_final_network_path"
            if not reason:
                path, junctions = routed
                if path.geom_type != "LineString" or path.length < .01:
                    path = LineString([first["point"], second["point"]])
                coords = np.array(path.coords)
                vectors = np.diff(coords, axis=0)
                vectors = vectors[np.linalg.norm(vectors, axis=1) > .05]
                directions = np.array([_unit(v) for v in vectors])
                # Every traversed edge must continue forward. A sideways
                # junction connector cannot swap parallel carriageway tracks.
                if len(directions) and np.min(directions @ _unit(vector)) < np.cos(np.radians(config.maximum_path_turn_degrees)):
                    reason = "turn_or_track_switch"
            decision = dict(first_seed=first["seed"], second_seed=second["seed"], change_typ=first["kind"],
                            gap_m=distance, lateral_m=lateral, accepted=False, reason=reason)
            decisions.append(decision)
            if not reason:
                candidates.append((path.length+6*lateral+20*(1-facing), first_id, second_id, path, junctions, len(decisions)-1))
    used = set()
    for _, first_id, second_id, path, junctions, decision_id in sorted(candidates, key=lambda c: (c[0], c[1], c[2])):
        if first_id in used or second_id in used:
            decisions[decision_id]["reason"] = "endpoint_has_better_same_track_continuation"
            continue
        first, second = endpoints[first_id], endpoints[second_id]
        used.update((first_id, second_id))
        groups.join(first["seed"], second["seed"])
        a, b = seeds.iloc[first["seed"]], seeds.iloc[second["seed"]]
        wb = float(np.mean([a.width_bef, b.width_bef])); wa = float(np.mean([a.width_aft, b.width_aft]))
        outer = max(wb, wa, 1.)
        geometry = path.buffer(outer/2, cap_style="square", join_style="round")
        if first["kind"] in {"widened", "narrowed"}:
            # Paired axes can lie between the two final tracks. Meet the saved
            # seed axes smoothly at the ends, following the final network inside
            # the junction. Flat caps keep width changes as two longitudinal
            # ribbons instead of adding spurious transverse rectangular bars.
            stations = np.linspace(0, path.length, max(3, int(np.ceil(path.length/2))+1))
            coordinates = np.array([path.interpolate(s).coords[0] for s in stations])
            first_offset = np.array(first["point"].coords[0])-coordinates[0]
            second_offset = np.array(second["point"].coords[0])-coordinates[-1]
            transition = max(.01, min(15., path.length/2))
            coordinates += np.maximum(0., 1-stations/transition)[:, None]*first_offset
            coordinates += np.maximum(0., 1-(path.length-stations)/transition)[:, None]*second_offset
            path = LineString([coordinates[0]-first["heading"], *coordinates, coordinates[-1]-second["heading"]])
            geometry = path.buffer(outer/2, cap_style="flat", join_style="round").difference(
                path.buffer(min(wb, wa)/2, cap_style="flat", join_style="round"))
        # Do not clip to a junction surface union or recheck probability: the
        # confirmed trajectories themselves supply the missing road corridor.
        bridges.append(dict(first_seed=first["seed"], second_seed=second["seed"], change_typ=first["kind"],
                            junction_count=len(set(junctions)), length_m=path.length, axis=path, geometry=polygonal(geometry)))
        decisions[decision_id].update(accepted=True, reason="same_track_network_continuation", junction_count=len(set(junctions)))
    object_rows, axis_rows, membership = [], [], []
    roots = sorted({groups.root(i) for i in range(len(seeds))})
    for number, root in enumerate(roots, 1):
        ids = [i for i in range(len(seeds)) if groups.root(i) == root]
        local_bridges = [b for b in bridges if groups.root(b["first_seed"]) == root]
        source = seeds.iloc[ids]
        object_id = f"AUTO{number:06d}"
        geometry = polygonal(union_all([*source.geometry, *(b["geometry"] for b in local_bridges)]))
        lengths = source.get("length_m", pd.Series(1., index=source.index)).to_numpy(float)
        wb = float(np.average(source.width_bef, weights=np.maximum(lengths, 1e-6)))
        wa = float(np.average(source.width_aft, weights=np.maximum(lengths, 1e-6)))
        row = source.iloc[0].to_dict()
        row.pop("axis_wkt", None)
        row.update(object_id=object_id, seed_count=len(ids), seed_ids=",".join(map(str, ids)),
                   bridge_count=len(local_bridges), junction_count=sum(b["junction_count"] for b in local_bridges),
                   width_bef=wb, width_aft=wa, width_diff=wa-wb if row["change_typ"] in {"widened", "narrowed"} else row.get("width_diff", 0.),
                   length_m=float(sum(lengths)+sum(b["length_m"] for b in local_bridges)), geometry=geometry)
        object_rows.append(row)
        for seed_id in ids:
            membership.append(dict(seed_id=seed_id, object_id=object_id, change_typ=row["change_typ"]))
            if seed_id in axes:
                axis_rows.append(dict(object_id=object_id, seed_id=seed_id, role="seed", change_typ=row["change_typ"], geometry=axes[seed_id]))
        for bridge in local_bridges:
            axis_rows.append(dict(object_id=object_id, seed_id=-1, role="junction_bridge" if bridge["junction_count"] else "gap_bridge",
                                  change_typ=row["change_typ"], geometry=bridge["axis"]))
    def frame(rows):
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=metric_crs) if rows else gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    objects = (frame(object_rows) if object_rows else seeds.drop(columns=["axis_wkt"], errors="ignore").copy()).to_crs(output_crs)
    if len(objects):
        objects.geometry = objects.geometry.map(polygonal)
    bridge_rows = [{k: v for k, v in b.items() if k != "axis"} for b in bridges]
    summary = {"truth_used": False, "existence_retested": False, "input_seeds": len(seeds), "output_objects": len(objects),
               "unmapped_seeds_retained": len(seeds)-len(axes), "bridge_count": len(bridges),
               "junction_bridge_count": sum(b["junction_count"] > 0 for b in bridges),
               "bridge_length_m": sum(b["length_m"] for b in bridges),
               "by_type": {kind: {"input": int((seeds.change_typ == kind).sum()),
                                    "output": int((objects.change_typ == kind).sum()) if len(objects) else 0}
                           for kind in ("added", "removed", "widened", "narrowed")}}
    return objects, {"change_objects": objects, "object_axes": frame(axis_rows), "assembly_bridges": frame(bridge_rows),
                     "membership": pd.DataFrame(membership), "decisions": pd.DataFrame(decisions), "summary": summary}


def write_assembly_audit(output_dir, seeds, artifacts):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg = output_dir/"network_assembly.gpkg"
    seeds.drop(columns=["axis_wkt"], errors="ignore").to_file(gpkg, layer="local_change_seeds", driver="GPKG")
    for name in ("change_objects", "object_axes", "assembly_bridges"):
        artifacts[name].to_file(gpkg, layer=name, driver="GPKG")
    artifacts["membership"].to_csv(output_dir/"assembly_membership.csv", index=False)
    artifacts["decisions"].to_csv(output_dir/"assembly_decisions.csv", index=False)
    (output_dir/"assembly_summary.json").write_text(json.dumps(artifacts["summary"], indent=2), encoding="utf-8")
