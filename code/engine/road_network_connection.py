from __future__ import annotations

"""Recover a planar network using track continuations and junction attachments.
All decisions use metric geometry, never feature IDs or region-specific rules.
"""
from dataclasses import dataclass
import math
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.prepared import prep
from shapely.strtree import STRtree
from .road_geometry import _RegionalRoadSeed, _deduplicate_points, _endpoint_heading, _polyline_length, _tangent_continuous_connector
from .road_track_corridors import infer_track_corridors, restore_track_corridors, corridor_conflict


def _key(point):
    return tuple(np.round(point, 4))


def _parts(geometry, kind):
    if geometry.is_empty:
        return []
    if geometry.geom_type == kind:
        return [geometry]
    return [part for child in getattr(geometry, 'geoms', ()) for part in _parts(child, kind)]


def _node_network(roads):
    """Insert planar contacts and emit edges between junctions, retaining bodies."""
    lines = [LineString(road.points) for road in roads]
    tree = STRtree(lines)
    cuts = [[] for _ in roads]
    endpoint_contacts = {}
    for i, line in enumerate(lines):
        for raw_j in tree.query(line):
            j = int(raw_j)
            if j <= i:
                continue
            intersection = line.intersection(lines[j])
            points = _parts(intersection, 'Point')
            for overlap in _parts(intersection, 'LineString'):
                points.extend([Point(overlap.coords[0]), Point(overlap.coords[-1])])
            for point in points:
                for index in (i, j):
                    distance = lines[index].project(point)
                    if 1e-6 < distance < lines[index].length-1e-6:
                        cuts[index].append((distance,np.asarray(point.coords[0])))
    # Projected-coordinate roundoff can leave a mathematically touching T-node
    # a few nanometres off the segment. Resolve only numerical contacts here.
    for i,line in enumerate(lines):
        for at_start in (True,False):
            point = Point(line.coords[0 if at_start else -1])
            for raw_j in tree.query(point.buffer(1e-5)):
                j = int(raw_j)
                if j==i or lines[j].distance(point)>1e-5:
                    continue
                distance = lines[j].project(point)
                xy = np.asarray(lines[j].interpolate(distance).coords[0])
                endpoint_contacts[i,at_start] = xy
                if 1e-6<distance<lines[j].length-1e-6:
                    cuts[j].append((distance,xy))
    result = []
    for road_id,(road, line, distances) in enumerate(zip(roads, lines, cuts)):
        positions = [(0.0,endpoint_contacts.get((road_id,True),np.asarray(line.coords[0])))]
        for value,xy in sorted(distances,key=lambda item:item[0]):
            if value-positions[-1][0] > 1e-6:
                positions.append((value,xy))
        positions.append((line.length,endpoint_contacts.get((road_id,False),np.asarray(line.coords[-1]))))
        for (first,first_xy), (second,second_xy) in zip(positions, positions[1:]):
            if second-first > 1e-8:
                points = road.points.copy() if len(positions)==2 else np.asarray(substring(line, first, second).coords)
                points[0],points[-1] = first_xy,second_xy
                result.append(_RegionalRoadSeed(points, road.width_m, road.source_ids, road.geometry_kind))
    return result


def _graph(roads):
    graph = nx.Graph()
    for road in roads:
        for first, second in zip(road.points, road.points[1:]):
            a, b = _key(first), _key(second)
            if a != b:
                weight = float(np.linalg.norm(second-first))
                if not graph.has_edge(a,b) or graph[a][b]['weight'] > weight:
                    graph.add_edge(a,b,weight=weight)
    return graph


def _join_chains(roads):
    ends = {}
    for index, road in enumerate(roads):
        for start in (True, False):
            ends.setdefault(_key(road.points[0 if start else -1]), []).append((index,start))
    links = {}
    for entries in ends.values():
        if len(entries)==2 and entries[0][0] != entries[1][0]:
            links[entries[0]], links[entries[1]] = entries[1], entries[0]
    visited, result = set(), []
    for initial in range(len(roads)):
        if initial in visited:
            continue
        index, start, seen = initial, True, set()
        while (index,start) in links and index not in seen:
            seen.add(index)
            index, attached = links[index,start]
            start = not attached
        chunks, sources, lengths, widths = [], set(), [], []
        while index not in visited:
            road = roads[index]
            visited.add(index)
            points = road.points if start else road.points[::-1]
            chunks.append(points)
            lengths.append(_polyline_length(points))
            widths.append(road.width_m)
            sources.update(road.source_ids)
            if (index,not start) not in links:
                break
            index, start = links[index,not start]
        result.append(_RegionalRoadSeed(_deduplicate_points(np.vstack(chunks)),float(np.average(widths,weights=lengths)),tuple(sorted(sources)),'network_track'))
    return result


@dataclass
class _Port:
    road: int
    start: bool
    point: np.ndarray
    heading: np.ndarray
    anchor: np.ndarray
    tangent: np.ndarray
    trim: float
    length: float
    separation: float = math.inf
    junction: bool = False

    @property
    def key(self):
        return self.road,self.start


@dataclass
class _Candidate:
    first: _Port
    second: _Port | None
    target: int
    points: np.ndarray
    score: float
    row: dict


def _ports(roads,lines,tree,graph):
    ports = []
    outgoing = {}
    for road in roads:
        for start in (True,False):
            point = road.points[0 if start else -1]
            inward = -_endpoint_heading(road.points,start,min(20,_polyline_length(road.points)))
            outgoing.setdefault(_key(point),[]).append(inward)
    for index,road in enumerate(roads):
        length = lines[index].length
        for start in (True,False):
            point = np.asarray(road.points[0 if start else -1])
            degree = graph.degree(_key(point)) if _key(point) in graph else 0
            if degree == 0 or degree == 2:
                continue
            h15 = _endpoint_heading(road.points,start,min(18.0,length*0.8))
            h30 = _endpoint_heading(road.points,start,min(35.0,length))
            heading = 0.7*h15+0.3*h30
            heading /= max(np.linalg.norm(heading),1e-9)
            # A T-node may be missing its through arm. Its unmatched approach
            # remains eligible for a continuation, but never for a new branch.
            if degree>1 and any(heading@direction>np.cos(np.deg2rad(50)) for direction in outgoing[_key(point)]):
                continue
            ordered = road.points if start else road.points[::-1]
            local = ordered[0]-ordered[1]
            local /= max(np.linalg.norm(local),1e-9)
            trim = min(7.0,length*0.20) if degree==1 and local@heading < np.cos(np.deg2rad(15)) and length>10 else 0.0
            anchor = np.asarray(lines[index].interpolate(trim if start else length-trim).coords[0]) if trim else point
            tangent = local
            if trim:
                inward = np.asarray(lines[index].interpolate(trim+0.01 if start else length-trim-0.01).coords[0])
                tangent = anchor-inward
                tangent /= max(np.linalg.norm(tangent),1e-9)
            port = _Port(index,start,point,heading,anchor,tangent,trim,length,junction=degree>1)
            # Nearby parallel tracks are measured behind the approach, avoiding
            # mistaking the candidate continuation itself for another lane.
            normal = np.asarray([-heading[1],heading[0]])
            for behind in np.unique(np.minimum([12,30,60],length*np.asarray([.3,.5,.8]))):
                probe = np.asarray(lines[index].interpolate(behind if start else length-behind).coords[0])
                cross_section = LineString([probe-65*normal,probe+65*normal])
                for raw_j in tree.query(cross_section):
                    j = int(raw_j)
                    if j==index:
                        continue
                    for contact in _parts(cross_section.intersection(lines[j]),'Point'):
                        d = lines[j].project(contact)
                        delta = np.asarray(lines[j].interpolate(min(lines[j].length,d+12)).coords[0])-np.asarray(lines[j].interpolate(max(0,d-12)).coords[0])
                        distance = contact.distance(Point(probe))
                        if 3<=distance<=65 and abs(delta@heading)/max(np.linalg.norm(delta),1e-9)>0.9:
                            port.separation = min(port.separation,distance)
            ports.append(port)
    return ports


def _support(points,surface):
    if surface is None:
        return 0.0
    line = LineString(points)
    return float(np.mean([surface.covers(line.interpolate(t,normalized=True)) for t in np.linspace(0,1,max(5,min(101,int(line.length/3)+1)))]))


def _row(a,b,roads,distance,lateral,support,score,kind):
    return dict(first_sources=list(roads[a.road].source_ids),second_sources=list(roads[b].source_ids),distance_m=float(distance),lateral_offset_m=float(lateral),surface_support=float(support),score=float(score),kind=kind,status='candidate',needs_review=bool(distance>150 or (distance>80 and support<0.25)),first_trim_wkt=_trim_wkt(a,roads),second_trim_wkt='')


def _trim_wkt(port,roads):
    if not port.trim:
        return ''
    line = LineString(roads[port.road].points)
    return substring(line,0,port.trim).wkt if port.start else substring(line,line.length-port.trim,line.length).wkt


def _continuations(ports,roads,maximum,surface):
    points = [Point(port.point) for port in ports]
    index = STRtree(points)
    result = []
    for i,a in enumerate(ports):
        for raw_j in index.query(points[i].buffer(maximum)):
            j = int(raw_j)
            b = ports[j]
            if j<=i:
                continue
            delta = b.point-a.point
            distance = np.linalg.norm(delta)
            if not 1e-5<distance<=maximum:
                continue
            chord = delta/distance
            facing = min(a.heading@chord,b.heading@-chord)
            turn = a.heading@-b.heading
            chord_support = _support([a.point,b.point],surface)
            supported_curve = chord_support>=0.65 and distance<=100 and min(a.separation,b.separation)>30
            facing_degrees = 62 if supported_curve else (55 if distance<25 else 38)
            if facing<np.cos(np.deg2rad(facing_degrees)) or turn<np.cos(np.deg2rad(60)):
                continue
            axis = a.heading-b.heading
            axis /= max(np.linalg.norm(axis),1e-9)
            lateral = abs(delta@np.asarray([-axis[1],axis[0]]))
            limit = min(20.0,max(8.0,distance*0.16),0.45*min(a.separation,b.separation))
            if supported_curve:
                limit = max(limit,min(22.0,distance*0.6))
            if lateral>limit or (distance>40 and min(a.length,b.length)<4):
                continue
            # Averaging two headings can hide a lane swap: opposite lateral
            # errors cancel. Check the corridor of each approach separately.
            if any(np.isfinite(p.separation) and abs(delta@np.asarray([-p.heading[1],p.heading[0]]))>max(3.0,.55*p.separation)
                   for p in (a,b)):
                continue
            if distance>80 and a.length+b.length<distance*0.5:
                continue
            curve = _tangent_continuous_connector(a.anchor,a.tangent,b.anchor,b.tangent,1.0)
            if curve is None:
                continue
            support = _support(curve,surface)
            if distance>160 and support<0.4 and facing<np.cos(np.deg2rad(18)):
                continue
            score = distance+16*lateral+distance*(2*(1-facing)+(1-turn))-8*support
            row = _row(a,b.road,roads,distance,lateral,support,score,'continuation')
            row['second_trim_wkt'] = _trim_wkt(b,roads)
            result.append(_Candidate(a,b,b.road,curve,score,row))
    return result


def _attachments(ports,roads,lines,tree,maximum,surface):
    result = []
    reach = min(150.0,maximum)
    for port in ports:
        if port.junction:
            continue
        ray = LineString([port.point,port.point+reach*port.heading])
        for raw_j in tree.query(Point(port.point).buffer(reach)):
            j = int(raw_j)
            if j==port.road:
                continue
            line = lines[j]
            positions = [line.project(Point(port.point))]
            positions.extend(line.project(p) for p in _parts(ray.intersection(line),'Point'))
            best = None
            for position in positions:
                end = np.asarray(line.interpolate(position).coords[0])
                delta = end-port.point
                distance = np.linalg.norm(delta)
                if not 1e-5<distance<=reach:
                    continue
                chord = delta/distance
                facing = port.heading@chord
                if facing<np.cos(np.deg2rad(70 if distance<8 else 38)):
                    continue
                direction = np.asarray(line.interpolate(min(line.length,position+5)).coords[0])-np.asarray(line.interpolate(max(0,position-5)).coords[0])
                parallel = abs(port.heading@direction)/max(np.linalg.norm(direction),1e-9)
                if parallel>np.cos(np.deg2rad(40)) or distance>max(18,port.length*1.1):
                    continue
                curve = _tangent_continuous_connector(port.anchor,port.tangent,end,-chord,1.0)
                if curve is None:
                    continue
                support = _support(curve,surface)
                score = distance*(1+2*(1-facing))-5*support
                candidate = _Candidate(port,None,j,curve,score,_row(port,j,roads,distance,0,support,score,'junction_attachment'))
                if best is None or score<best.score:
                    best = candidate
            if best is not None:
                result.append(best)
    return result


def _nearby_conflict(candidate,roads,lines,tree):
    line = LineString(candidate.points)
    member_ids = {candidate.first.road,candidate.target}
    chord = candidate.points[-1]-candidate.points[0]
    chord /= max(np.linalg.norm(chord),1e-9)
    spacing = min(candidate.first.separation,candidate.second.separation if candidate.second else math.inf)
    radius = min(5.0,max(1.5,0.5*min(roads[i].width_m for i in member_ids)),0.4*spacing)
    tube = line.buffer(radius,cap_style=2)
    for raw_j in tree.query(tube):
        j = int(raw_j)
        for contact in _parts(line.intersection(lines[j]),'Point'):
            at_first = contact.distance(Point(candidate.points[0]))<1e-4
            at_last = contact.distance(Point(candidate.points[-1]))<1e-4
            if (j==candidate.first.road and at_first) or (j==candidate.target and at_last):
                continue
            if j in member_ids:
                ports = [candidate.first]+([candidate.second] if candidate.second else [])
                in_trim = False
                for port in ports:
                    if port.road!=j or not port.trim:
                        continue
                    position = lines[j].project(contact)
                    if (position if port.start else lines[j].length-position)<=port.trim+1e-4:
                        in_trim = True
                if in_trim:
                    continue
                return 'crosses_member_body'
            position = lines[j].project(contact)
            tangent = np.asarray(lines[j].interpolate(min(lines[j].length,position+4)).coords[0])-np.asarray(lines[j].interpolate(max(0,position-4)).coords[0])
            curve_position = line.project(contact)
            curve_tangent = np.asarray(line.interpolate(min(line.length,curve_position+4)).coords[0])-np.asarray(line.interpolate(max(0,curve_position-4)).coords[0])
            if abs(tangent@curve_tangent)/max(np.linalg.norm(tangent)*np.linalg.norm(curve_tangent),1e-9)>np.cos(np.deg2rad(35)):
                return 'cross_track_connection'
        overlap = lines[j].intersection(tube)
        parallel_length = 0.0
        for part in _parts(overlap,'LineString'):
            for start,end in zip(part.coords,list(part.coords)[1:]):
                direction = np.asarray(end)-start
                length = np.linalg.norm(direction)
                if abs(direction@chord)/max(length,1e-9)>np.cos(np.deg2rad(30)):
                    parallel_length += length
        allowance = candidate.first.trim+(candidate.second.trim if candidate.second else 0) if j in member_ids else 0
        if parallel_length>max(allowance+2,min(10,max(3,line.length*0.2))):
            return 'bypasses_existing_fragment'
        direction = roads[j].points[-1]-roads[j].points[0]
        parallel = abs(direction@chord)/max(np.linalg.norm(direction),1e-9)
        if j not in member_ids and parallel>np.cos(np.deg2rad(30)):
            if line.crosses(lines[j]):
                return 'cross_track_connection'
    return ''


def _short_cycle(candidate,graph,lines):
    # A long existing route around a city block is not a redundant local loop.
    limit = max(25.0,candidate.row['distance_m']*1.7)
    distances = nx.single_source_dijkstra_path_length(graph,_key(candidate.first.point),cutoff=limit,weight='weight')
    if candidate.second is not None:
        if _key(candidate.second.point) in distances:
            return True
        # A long, narrow return path is still a redundant parallel loop, even
        # when its length exceeds the local shortest-path cutoff.
        end = _key(candidate.second.point)
        length_map, paths = nx.single_source_dijkstra(graph,_key(candidate.first.point),cutoff=max(120,candidate.row['distance_m']*3),weight='weight')
        if end in paths:
            from shapely.geometry import Polygon
            perimeter = length_map[end]+LineString(candidate.points).length
            polygon = Polygon(np.vstack((np.asarray(paths[end]),candidate.points[::-1])))
            if polygon.area/max(perimeter,1e-9)<3.0:
                return True
        return False
    line = lines[candidate.target]
    position = line.project(Point(candidate.points[-1]))
    contacts = [(_key(line.coords[0]),0.),(_key(line.coords[-1]),line.length)]
    contacts.extend(graph.graph.get('attachment_contacts',{}).get(candidate.target,()))
    if min(distances.get(node,math.inf)+abs(position-along) for node,along in contacts)<=limit:
        return True
    length_map,paths = nx.single_source_dijkstra(graph,_key(candidate.first.point),cutoff=max(120,candidate.row['distance_m']*3),weight='weight')
    from shapely.geometry import Polygon
    for node,along in contacts:
        if node not in paths:
            continue
        tail = substring(line,along,position)
        tail_points = np.asarray(tail.coords)
        perimeter = length_map[node]+abs(position-along)+LineString(candidate.points).length
        polygon = Polygon(np.vstack((np.asarray(paths[node]),tail_points,candidate.points[::-1])))
        if polygon.area/max(perimeter,1e-9)<5.:
            return True
    return False


def _select(candidates,roads,lines,tree,graph,corridors=()):
    candidates.sort(key=lambda c:(c.score,tuple(c.first.point),tuple(c.points[-1])))
    options = {}
    for c in candidates:
        reason = corridor_conflict(c.points,corridors) or _nearby_conflict(c,roads,lines,tree)
        if not reason and _short_cycle(c,graph,lines):
            reason = 'short_redundant_cycle'
        if reason:
            c.row['status'] = reason
            continue
        options.setdefault(c.first.key,[]).append(c)
        if c.second is not None:
            options.setdefault(c.second.key,[]).append(c)
    used,accepted = set(),[]
    for c in candidates:
        if c.row['status'] != 'candidate':
            continue
        endpoints = [c.first.key]+([c.second.key] if c.second else [])
        reason = ''
        if any(end in used for end in endpoints):
            reason = 'endpoint_already_connected'
        elif any(options[end][0] is not c for end in endpoints):
            reason = 'not_mutual_continuation'
        elif any(len(options[end])>1 and options[end][1].score-c.score<max(1.0,c.row['distance_m']*0.035) for end in endpoints):
            reason = 'ambiguous_continuation'
        else:
            reason = _nearby_conflict(c,roads,lines,tree)
            if not reason and _short_cycle(c,graph,lines):
                reason = 'short_redundant_cycle'
            if not reason:
                for previous in accepted:
                    first_line,second_line = LineString(c.points),LineString(previous.points)
                    for contact in _parts(first_line.intersection(second_line),'Point'):
                        vectors = []
                        for curve in (first_line,second_line):
                            p = curve.project(contact)
                            vectors.append(np.asarray(curve.interpolate(min(curve.length,p+4)).coords[0])-np.asarray(curve.interpolate(max(0,p-4)).coords[0]))
                        if abs(vectors[0]@vectors[1])/max(np.linalg.norm(vectors[0])*np.linalg.norm(vectors[1]),1e-9)>np.cos(np.deg2rad(35)):
                            reason = 'cross_track_connection'
                            break
                    if reason:
                        break
        c.row['status'] = reason or 'accepted'
        if reason:
            continue
        accepted.append(c)
        used.update(endpoints)
        first,second = _key(c.first.point),_key(c.second.point if c.second else c.points[-1])
        graph.add_edge(first,second,weight=c.row['distance_m'])
        if c.second is None:
            target = lines[c.target]
            position = target.project(Point(c.points[-1]))
            graph.add_edge(second,_key(target.coords[0]),weight=position)
            graph.add_edge(second,_key(target.coords[-1]),weight=target.length-position)
            graph.graph.setdefault('attachment_contacts',{}).setdefault(c.target,[]).append((second,position))
    return accepted


def _apply(roads,candidates):
    trims = {}
    for c in candidates:
        for port in [c.first]+([c.second] if c.second else []):
            trims[port.key] = port.trim
    result = []
    for index,road in enumerate(roads):
        start,end = trims.get((index,True),0),trims.get((index,False),0)
        points = np.asarray(substring(LineString(road.points),start,_polyline_length(road.points)-end).coords) if start or end else road.points
        result.append(_RegionalRoadSeed(points,road.width_m,road.source_ids,road.geometry_kind))
    for c in candidates:
        members = [roads[c.first.road],roads[c.target]]
        sources = tuple(sorted(set(members[0].source_ids+members[1].source_ids)))
        result.append(_RegionalRoadSeed(c.points,float(np.mean([r.width_m for r in members])),sources,'connection'))
    return _join_chains(_node_network(result))


def _metrics(roads):
    graph = _graph(roads)
    components = list(nx.connected_components(graph))
    lengths = [sum(data['weight'] for _,_,data in graph.subgraph(nodes).edges(data=True)) for nodes in components]
    return dict(components=len(components),dangling=sum(d==1 for _,d in graph.degree),main_length_ratio=max(lengths,default=0)/max(sum(lengths),1e-9),junctions=sum(d>=3 for _,d in graph.degree))


def retain_main_component(roads):
    """Partition an already noded network by total unique edge length.

    Return the original seed objects, without snapping, splitting or smoothing.
    Equal-length components are ordered by their smallest node coordinate.
    """
    graph = _graph(roads)
    components = list(nx.connected_components(graph))
    if not components:
        return [], []
    main = min(components, key=lambda nodes: (
        -sum(data['weight'] for _, _, data in graph.subgraph(nodes).edges(data=True)),
        min(nodes)))
    kept, removed = [], []
    for road in roads:
        (kept if _key(road.points[0]) in main else removed).append(road)
    return kept, removed


def connect_clean_road_seeds(roads: list[_RegionalRoadSeed],surface_geometry=None,unit_size_m: float=1.0,*,max_gap_m: float=300.0,keep_main_component: bool=False):
    """Recover a noded planar network with paired trajectory constraints.
    Replaced corridor segments are audited; elsewhere only endpoint tails may
    move, within 7 m and 20% of track length. Elevations are not inferred.
    """
    if not np.isfinite(unit_size_m) or unit_size_m<=0 or not np.isfinite(max_gap_m) or max_gap_m<=0:
        raise ValueError('Distances must be finite and positive')
    active = []
    for road in roads:
        points = _deduplicate_points(np.asarray(road.points)*unit_size_m)
        if len(points)<2 or not np.all(np.isfinite(points)):
            raise ValueError('Each road must contain at least two distinct finite XY points')
        active.append(_RegionalRoadSeed(points,road.width_m,road.source_ids,road.geometry_kind))
    if surface_geometry is not None and unit_size_m!=1:
        from shapely.affinity import scale
        surface_geometry = scale(surface_geometry,xfact=unit_size_m,yfact=unit_size_m,origin=(0,0))
    surface = prep(surface_geometry) if surface_geometry is not None and not surface_geometry.is_empty else None
    baseline = _metrics(_node_network(active))
    corridors = infer_track_corridors(active,max_gap_m)
    active,replaced,corridor_bridges,cleanup = restore_track_corridors(active,corridors,max_gap_m) if corridors else (active,LineString(),[],dict(connection_redundant_fragment_count=0,connection_redundant_length_m=0.))
    active = _join_chains(_node_network(active))
    audit = [dict(first_sources=list(sources),second_sources=[],distance_m=line.length,lateral_offset_m=0.,surface_support=_support(line.coords,surface),score=0.,kind='corridor_reconstruction',status='accepted',needs_review=True,first_trim_wkt='',second_trim_wkt='',round=-1,geometry=line) for line,sources in corridor_bridges]
    if not replaced.is_empty:
        audit.append(dict(first_sources=[],second_sources=[],distance_m=replaced.length,lateral_offset_m=0.,surface_support=0.,score=0.,kind='corridor_replacement',status='replaced',needs_review=True,first_trim_wkt='',second_trim_wkt='',round=-1,geometry=replaced))
    additions,rounds = [],0
    # Every operation consumes dangling ends; stop at a fixed point, not after
    # an arbitrary small number of passes. Larger chains inform later gaps.
    while active:
        graph = _graph(active)
        lines = [LineString(r.points) for r in active]
        tree = STRtree(lines)
        ports = _ports(active,lines,tree,graph)
        candidates = _continuations(ports,active,max_gap_m,surface)
        selected = _select(candidates,active,lines,tree,graph,corridors)
        audit.extend({**c.row,'round':rounds,'geometry':LineString(c.points)} for c in candidates)
        if not selected:
            candidates = _attachments(ports,active,lines,tree,min(150,max_gap_m),surface)
            selected = _select(candidates,active,lines,tree,graph,corridors)
            audit.extend({**c.row,'round':rounds,'geometry':LineString(c.points)} for c in candidates)
        if not selected:
            break
        additions.extend(selected)
        active = _apply(active,selected)
        rounds += 1
        if rounds>2*len(roads)+1:
            raise RuntimeError('Connection did not converge')
    components_before_filter = _metrics(active)['components']
    removed = []
    if keep_main_component:
        active, removed = retain_main_component(active)
        audit.extend(dict(first_sources=list(road.source_ids),second_sources=[],
                          distance_m=_polyline_length(road.points),lateral_offset_m=0.,
                          surface_support=0.,score=0.,kind='isolated_component',
                          status='isolated_removed',needs_review=False,
                          first_trim_wkt='',second_trim_wkt='',round=rounds,
                          geometry=LineString(road.points)) for road in removed)
    final = _metrics(active)
    trim_length = sum(c.first.trim+(c.second.trim if c.second else 0) for c in additions)
    lengths = [c.row['distance_m'] for c in additions]
    diagnostics = dict(connection_input_count=len(roads),connection_output_count=len(active),connection_added_count=len(additions),connection_added_length_m=float(sum(lengths)),connection_max_length_m=float(max(lengths,default=0)),connection_round_count=rounds,connection_continuation_count=sum(c.second is not None for c in additions),connection_attachment_count=sum(c.second is None for c in additions),connection_smoothed_tail_length_m=float(trim_length),connection_dangling_before=baseline['dangling'],connection_dangling_after=final['dangling'],connection_components_before=baseline['components'],connection_components_after=final['components'],connection_main_length_ratio_before=baseline['main_length_ratio'],connection_main_length_ratio_after=final['main_length_ratio'],connection_junctions_before=baseline['junctions'],connection_junctions_after=final['junctions'])
    diagnostics.update(connection_corridor_count=len(corridors),connection_corridor_replaced_length_m=replaced.length,connection_corridor_bridge_count=len(corridor_bridges),connection_corridor_bridge_length_m=sum(g.length for g,_ in corridor_bridges))
    diagnostics.update(cleanup)
    diagnostics.update(connection_keep_main_component=keep_main_component,
                       connection_components_before_filter=components_before_filter,
                       connection_isolated_removed_count=len(removed),
                       connection_isolated_removed_length_m=sum(_polyline_length(r.points) for r in removed))
    if unit_size_m!=1:
        from shapely.affinity import scale
        audit = [{**r,'geometry':scale(r['geometry'],xfact=1/unit_size_m,yfact=1/unit_size_m,origin=(0,0))} for r in audit]
        from shapely import wkt
        for row in audit:
            for key in ('first_trim_wkt','second_trim_wkt'):
                if row[key]:
                    row[key] = scale(wkt.loads(row[key]),xfact=1/unit_size_m,yfact=1/unit_size_m,origin=(0,0)).wkt
    output = [_RegionalRoadSeed(r.points/unit_size_m,r.width_m,r.source_ids,r.geometry_kind) for r in active]
    return output,diagnostics,audit
