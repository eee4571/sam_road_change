"""Final road surfaces rebuilt from a noded, attributed road graph.

This module consumes immutable final axes and saved width/image observations.
It does not edit extraction, measurement, change decisions or temporal tracks.
All lengths are metres. Only this renderer owns portals and junction surfaces.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from shapely import union_all, line_interpolate_point, line_locate_point
from shapely.geometry import LineString, Point, Polygon, MultiPoint
from shapely.ops import substring
from shapely.strtree import STRtree
from shapely.prepared import prep

from .auto_change_geometry import corridor
from .auto_change_assembly import polygonal
from .road_surface_quality import stable_width


@dataclass
class Edge:
    axis: object
    stations: np.ndarray
    widths: np.ndarray
    quality: np.ndarray
    sources: tuple
    level: tuple
    nodes: tuple


@dataclass
class Chain:
    axis: object
    stations: np.ndarray
    widths: np.ndarray
    quality: np.ndarray
    sources: tuple
    ends: tuple
    edges: tuple


def _value(row, key):
    value = row.get(key)
    return '' if value is None or str(value).lower() in ('nan', 'none', '') else str(value)


def _level(row):
    layer=_value(row,'layer') or _value(row,'z_level') or '0'
    try:layer=format(float(layer),'g')
    except ValueError:pass
    flag=lambda k: _value(row,k).lower() in ('1','1.0','true','yes')
    return layer,flag('bridge'),flag('tunnel')


def _identity_conflict(a, b):
    # Feature IDs/parent_id/track_id identify processing pieces, not necessarily
    # roads. Only declared road identity/class fields constrain correspondence.
    return any(_value(a, k) and _value(b, k) and _value(a, k) != _value(b, k)
               for k in ('road_ref', 'road_name', 'canonical_road_id', 'road_class'))


def _node(level, point):
    return level, round(float(point[0]), 6), round(float(point[1]), 6)


def _adjacency(edges, active):
    nodes = defaultdict(list)
    for k in sorted(active):
        for end, node in enumerate(edges[k].nodes):
            nodes[node].append((k, end))
    return nodes


def _outward(edge, end, reach=6.):
    s = min(reach, edge.axis.length * .45)
    xy = np.asarray(edge.axis.coords)
    tip = xy[0 if end == 0 else -1]
    q = np.asarray(edge.axis.interpolate(s if end == 0 else edge.axis.length-s).coords[0])
    d = q-tip
    return d/max(np.linalg.norm(d), 1e-12)


def _compatible(first, fe, second, se, metadata):
    if first.level != second.level:
        return 'different_level'
    if any(_identity_conflict(metadata[a], metadata[b]) for a in first.sources for b in second.sources):
        return 'different_road'
    # Both local and approach tangents must agree. Do not flatten a genuine
    # acute corner just because its node has degree two.
    if any(-float(_outward(first, fe, r) @ _outward(second, se, r)) < np.cos(np.pi/4)
           for r in (1., 8.)):
        return 'direction_break'
    a = first.widths[0 if fe == 0 else -1]
    b = second.widths[0 if se == 0 else -1]
    qa = first.quality[0 if fe == 0 else -1]
    qb = second.quality[0 if se == 0 else -1]
    if qa >= .99 and qb >= .99 and max(a,b) > 2*min(a,b):
        return 'road_correspondence_break'
    return ''


def _build_graph(profiles, quality, metadata):
    groups = defaultdict(list)
    for i, (a, _, _) in enumerate(profiles):
        if a.length > 1e-6:
            groups[_level(metadata[i])].append(i)
    edges = []
    for level, ids in groups.items():
        axes = [profiles[i][0] for i in ids]
        tree = STRtree(axes)
        source_cover = {}
        # GEOS nodes true interior crossings and dissolves exact overlaps. Merely
        # nearby parallel lines never enter the same edge or intersection node.
        network = union_all(axes)
        parts = [network] if network.geom_type == 'LineString' else list(network.geoms)
        for axis in parts:
            if axis.is_empty or axis.length < 1e-6:
                continue
            probe = axis.interpolate(.5, normalized=True)
            hits = tree.query(probe, predicate='dwithin', distance=1e-6)
            sources = []
            for raw in hits:
                j=int(raw)
                if j not in source_cover:source_cover[j]=prep(axes[j].buffer(1e-6,cap_style='square'))
                if source_cover[j].covers(axis):sources.append(ids[j])
            sources.sort()
            if not sources:
                raise ValueError('Noded road edge lost its source correspondence')
            s = np.linspace(0., axis.length, max(2, int(np.ceil(axis.length/2.))+1))
            points = line_interpolate_point(axis, s)
            widths, grades = [], []
            for i in sources:
                original, ss, vv = profiles[i]
                locations = line_locate_point(original, points)
                widths.append(np.interp(locations, ss, vv))
                grades.append(np.interp(locations, ss, quality[i]))
            widths, grades = np.asarray(widths), np.asarray(grades)
            # Duplicates are one graph edge, never extra votes for road width.
            best = np.argmax(grades, axis=0)
            columns = np.arange(len(s))
            edges.append(Edge(axis, s, widths[best, columns], grades[best, columns],
                tuple(sorted(sources)), level, (_node(level, axis.coords[0]), _node(level, axis.coords[-1]))))
    return edges


def _join_roundoff_gaps(edges, metadata):
    """Connect only reciprocal, same-road terminal gaps below 25 centimetres.

    Axis coordinates stay intact; an explicit tiny graph edge bridges the gap.
    No endpoint-to-interior snapping or general spatial clustering is allowed.
    """
    nodes = _adjacency(edges, range(len(edges)))
    tips = [(node, ends[0]) for node, ends in nodes.items() if len(ends) == 1]
    points = [Point(node[1:]) for node, _ in tips]
    tree = STRtree(points)
    choices = {}
    for i, (node, (k, end)) in enumerate(tips):
        candidates = []
        for raw in tree.query(points[i], predicate='dwithin', distance=.25):
            j = int(raw)
            if i == j: continue
            other, (l, oe) = tips[j]
            distance = points[i].distance(points[j])
            if k == l or distance < 1e-6 or _compatible(edges[k],end,edges[l],oe,metadata):continue
            direction = (np.asarray(points[j].coords[0])-np.asarray(points[i].coords[0]))/distance
            if -float(_outward(edges[k],end)@direction) < .95 or float(_outward(edges[l],oe)@direction) < .95:continue
            candidates.append(j)
        if len(candidates) == 1:choices[i] = candidates[0]
    count = 0
    for i, j in choices.items():
        if i >= j or choices.get(j) != i:continue
        a, (k,e) = tips[i];b, (l,f) = tips[j]
        axis = LineString([points[i], points[j]])
        w = np.array([edges[k].widths[0 if e==0 else -1], edges[l].widths[0 if f==0 else -1]])
        q = np.array([edges[k].quality[0 if e==0 else -1], edges[l].quality[0 if f==0 else -1]])
        edges.append(Edge(axis,np.array([0.,axis.length]),w,q,
            tuple(sorted(set(edges[k].sources+edges[l].sources))),edges[k].level,(a,b)))
        count += 1
    return count


def _trace(edges, start, pair):
    route, seen = [], set()
    current = start
    while current[0] not in seen:
        k,e = current;seen.add(k);route.append(current)
        current = pair.get((k,1-e))
        if current is None:break
    return route


def _chain(edges, route):
    coords, stations, widths, grades = [], [], [], []
    offset = 0.
    for k,e in route:
        edge = edges[k]
        xy = list(edge.axis.coords)
        s,w,q = edge.stations,edge.widths,edge.quality
        if e:xy=xy[::-1];s=edge.axis.length-s[::-1];w=w[::-1];q=q[::-1]
        if coords and np.linalg.norm(np.asarray(coords[-1])-xy[0]) > 2e-6:
            raise ValueError('Canonical chain has a disconnected graph edge')
        coords.extend(xy[1:] if coords else xy)
        if stations:
            # One sample at the common node, prefer an actual reliable reading.
            if q[0]>grades[-1]:widths[-1]=w[0];grades[-1]=q[0]
            stations.extend(offset+s[1:]);widths.extend(w[1:]);grades.extend(q[1:])
        else:stations.extend(s);widths.extend(w);grades.extend(q)
        offset += edge.axis.length
    axis = LineString(coords)
    s = np.linspace(0.,axis.length,max(2,int(np.ceil(axis.length/2.))+1))
    first,last = route[0],route[-1]
    return Chain(axis,s,np.interp(s,stations,widths),np.interp(s,stations,grades),
        tuple(sorted({src for k,_ in route for src in edges[k].sources})),
        (edges[first[0]].nodes[first[1]],edges[last[0]].nodes[1-last[1]]),tuple(k for k,_ in route))


def _prune_terminals(edges, metadata, evidence, observations):
    active = set(range(len(edges)))
    nodes = _adjacency(edges, active)
    # Trace the whole terminal arm through fragmented degree-2 vertices before
    # measuring length, not each small edge independently. One pass cannot erode
    # a long road by repeatedly shaving small pieces from its end.
    pair = {}
    for ends in nodes.values():
        if len(ends)==2 and ends[0][0]!=ends[1][0]:pair[ends[0]]=ends[1];pair[ends[1]]=ends[0]
    temporal = [STRtree(lines) for lines in observations]
    seen, audit = set(), []
    for node, ends in nodes.items():
        if len(ends)!=1 or ends[0][0] in seen:continue
        route = _trace(edges,ends[0],pair)
        arm = _chain(edges,route);seen.update(arm.edges)
        width = float(np.median(arm.widths))
        if arm.axis.length>min(30.,3*width):continue
        attached = len(nodes[arm.ends[1]])>=3
        report = dict(source_features=list(arm.sources),length_m=float(arm.axis.length),
                      terminal_spur=attached,action='retain',reason='insufficient_evidence')
        if any(_value(metadata[i],'track_id') or metadata[i].get('protect_surface') for i in arm.sources):
            report['reason']='event_constrained'
        elif not np.all(arm.quality==0):
            report['reason']='rgb_support_or_unknown'
        elif temporal:
            # Exclude the junction root: probability on the main road is not
            # evidence for the far end of a tooth-shaped offshoot.
            probe = substring(arm.axis,0.,arm.axis.length*.65) if attached else arm.axis
            points = line_interpolate_point(probe,np.linspace(0.,probe.length,7))
            tolerance = max(1.,min(2.,width/4))
            supported = any(np.mean([len(t.query(p,predicate='dwithin',distance=tolerance))>0 for p in points])>=.5
                            for t in temporal)
            if supported:report['reason']='other_period_support'
            else:
                ev=evidence() if callable(evidence) else evidence
                score=ev.measure(probe) if ev is not None else {}
                report['evidence']=score
                if score.get('probability_valid_fraction',0)>=.9 and score.get('probability_support',1)<.1:
                    active.difference_update(arm.edges)
                    report.update(action='suppress',reason='short_arm_jointly_unsupported')
        audit.append(report)
    return active,audit


def _canonical_chains(edges, active, metadata):
    nodes = _adjacency(edges,active)
    pair, rejects = {}, Counter()
    for ends in nodes.values():
        if len(ends)!=2 or ends[0][0]==ends[1][0]:continue
        (a,e),(b,f) = ends
        reason = _compatible(edges[a],e,edges[b],f,metadata)
        if reason:rejects[reason]+=1;continue
        pair[(a,e)]=(b,f);pair[(b,f)]=(a,e)
    starts = [(k,e) for k in sorted(active) for e in (0,1) if (k,e) not in pair]
    starts += [(k,0) for k in sorted(active)]
    seen,chains = set(),[]
    for start in starts:
        if start[0] in seen:continue
        route=_trace(edges,start,pair);seen.update(k for k,_ in route)
        chains.append(_chain(edges,route))
    return chains,nodes,len(pair)//2,dict(rejects)


def _junction_complexes(chains, nodes):
    """Treat short, connected junction-internal links as one physical junction.

    Nodes are not spatially snapped. Only an existing short road connection may
    join two branch nodes whose road-width footprints overlap. A diameter bound
    prevents transitive clustering along a street. Interior axes remain audited
    and covered by the reconstructed footprint, not deleted as noise.
    """
    members={node:{node} for node,ends in nodes.items() if len(ends)>=3}
    owner={node:node for node in members}
    candidates=sorted(chains,key=lambda c:c.axis.length)
    for c in candidates:
        a,b=c.ends
        if a==b or a not in owner or b not in owner:continue
        ra,rb=owner[a],owner[b]
        if ra==rb:continue
        width=float(np.median(c.widths))
        if c.axis.length>min(6.,width/2):continue
        combined=members[ra]|members[rb]
        xy=np.array([n[1:] for n in combined])
        if np.linalg.norm(np.ptp(xy,axis=0))>min(12.,max(4.,width)):continue
        members[ra]=combined;del members[rb]
        for node in combined:owner[node]=ra
    kept,internal=[],defaultdict(list)
    for c in chains:
        a,b=c.ends
        if a!=b and a in owner and b in owner and owner[a]==owner[b] and c.axis.length<=12.:
            internal[owner[a]].append(c)
        else:kept.append(c)
    external={owner.get(n,n) for c in kept for n in c.ends}
    for group in list(internal):
        if group in external:continue
        # An entire small closed network is not a junction connector. With no
        # external portals, keep its roads instead of swallowing the component.
        kept.extend(internal.pop(group))
        for node in members.pop(group):owner[node]=node;members[node]={node}
    return kept,owner,members,internal


def _portal(chain, end, distance):
    axis=chain.axis
    station=distance if end==0 else axis.length-distance
    # This is exactly corridor()'s cap tangent on the trimmed axis: the next
    # original vertex, not a chord of arbitrary length across a curved road.
    part=substring(axis,station,axis.length) if end==0 else substring(axis,0,station)
    xy=np.asarray(part.coords)
    p=xy[0] if end==0 else xy[-1]
    d=xy[1]-xy[0] if end==0 else xy[-2]-xy[-1]
    d/=max(np.linalg.norm(d),1e-12)
    width=float(np.interp(station,chain.stations,chain.widths))
    n=np.array([-d[1],d[0]])*width/2
    return dict(point=p,direction=d,right=p-n,left=p+n,width=width,station=float(station))


def _junction(portals, center, required_axes):
    anchors=[xy for axis in required_axes for xy in axis.coords]
    anchors.append(center)
    portals=sorted(portals,key=lambda p:np.arctan2(*(p['point']-center)[::-1]))
    boundary=[]
    for i,p in enumerate(portals):
        boundary.extend([p['right'],p['left']])
        q=portals[(i+1)%len(portals)]
        matrix=np.column_stack([p['direction'],-q['direction']])
        if abs(np.linalg.det(matrix))<.05:continue
        t,u=np.linalg.solve(matrix,p['left']-q['right'])
        reach=max(np.linalg.norm(p['point']-center),np.linalg.norm(q['point']-center))
        if 0<=t<=2*reach and 0<=u<=2*reach:
            control=p['left']-t*p['direction']
            f=np.linspace(0.,1.,13)[1:-1,None]
            boundary.extend((1-f)**2*p['left']+2*f*(1-f)*control+f**2*q['right'])
    ring=Polygon(boundary)
    envelope=MultiPoint([v for p in portals for v in (p['right'],p['left'])]+anchors).convex_hull
    # Overlapping portals (very short/acute branches) have no simple cyclic
    # boundary. Their geometric definition is the shared portal envelope, not
    # a union of untrimmed road corridors or an extension of the main road.
    simple=ring.is_valid and not ring.is_empty and ring.buffer(1e-7).covers(union_all(required_axes))
    geometry=ring if simple else envelope
    return geometry,('tangent_portal_ring' if simple else 'overlapping_portal_envelope')


def _generalize(geometry, junctions, chains):
    if geometry.is_empty:return geometry
    candidate=geometry.simplify(.2,preserve_topology=True)
    # Junction footprints and portal seams remain exact. This is a bounded
    # boundary operation only, with no axis movement or morphology on the map.
    if junctions:
        protected=union_all(junctions).buffer(.25)
        candidate=polygonal(union_all([candidate.difference(protected),geometry.intersection(protected)]))
    # A simplifier can replace a portal tangent with a long, slightly oblique
    # edge. Bound the *spatial extent* of edits as well as their displacement:
    # only tiny boundary details may change, never an entire stable road side.
    def local_parts(delta):
        delta=polygonal(delta)
        pieces=[delta] if delta.geom_type=='Polygon' else list(delta.geoms)
        return [p for p in pieces if not p.is_empty and
                np.hypot(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1])<=2.]
    removed=local_parts(geometry.difference(candidate))
    added=local_parts(candidate.difference(geometry))
    candidate=polygonal(union_all([geometry.difference(union_all(removed)),*added]))
    old=[geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)
    new=[candidate] if candidate.geom_type=='Polygon' else list(candidate.geoms)
    if (not candidate.is_valid or len(new)!=len(old) or
        sum(len(p.interiors) for p in new)!=sum(len(p.interiors) for p in old)):
        return geometry
    if geometry.boundary.hausdorff_distance(candidate.boundary)>.25:return geometry
    if not candidate.buffer(1e-6).covers(union_all([c.axis for c in chains])):return geometry
    return candidate


def build_road_surface(profiles, quality, *, metadata=None, evidence=None, observations=()):
    """Return (final polygon, audit); original axes and width arrays are immutable."""
    metadata=list(metadata) if metadata is not None else [{} for _ in profiles]
    if len(profiles)!=len(quality) or len(profiles)!=len(metadata):raise ValueError('Road profile metadata count mismatch')
    profiles=[(a,np.clip(np.asarray(s,float),0.,a.length),np.asarray(w,float)) for a,s,w in profiles]
    edges=_build_graph(profiles,quality,metadata)
    noded_count=len(edges)
    gap_count=_join_roundoff_gaps(edges,metadata)
    active,spur_audit=_prune_terminals(edges,metadata,evidence,observations)
    chains,nodes,merged,rejections=_canonical_chains(edges,active,metadata)
    chains,owner,members,internal=_junction_complexes(chains,nodes)
    chain_audit=[]
    for i,c in enumerate(chains):
        c.widths,report=stable_width(c.stations,c.widths,c.quality>=.99)
        report.update(chain_id=i,source_features=list(c.sources),graph_edges=list(c.edges),
                      length_m=float(c.axis.length),axis_wkt=c.axis.wkt)
        chain_audit.append(report)
    ends=defaultdict(list)
    for i,c in enumerate(chains):
        for e,node in enumerate(c.ends):ends[owner.get(node,node)].append((i,e))
    trim=np.zeros((len(chains),2))
    portals={}
    for node,arms in ends.items():
        if len(arms)<2:continue
        # Closed chains own no junction unless another road meets the ring.
        if len(arms)==2 and arms[0][0]==arms[1][0] and not internal.get(node):continue
        for i,e in arms:
            c=chains[i];width=c.widths[0 if e==0 else -1]
            center=np.mean([n[1:] for n in members.get(node,{node})],axis=0)
            offset=np.linalg.norm(np.asarray(c.axis.coords[0 if e==0 else -1])-center)
            trim[i,e]=min(max(2.,float(width))+offset,c.axis.length*.4)
            portals[(i,e)]=_portal(c,e,trim[i,e])
    bodies=[]
    for i,c in enumerate(chains):
        a,b=trim[i,0],c.axis.length-trim[i,1]
        axis=substring(c.axis,a,b)
        ss=np.unique(np.r_[a,c.stations[(c.stations>a)&(c.stations<b)],b])
        bodies.append(corridor(axis,ss-a,np.interp(ss,c.stations,c.widths)))
        chain_audit[i]['portal_trim_m']=trim[i].tolist()
    junctions=[];junction_audit=[]
    for node,arms in ends.items():
        local=[portals[arm] for arm in arms if arm in portals]
        if not local:continue
        center=np.mean([n[1:] for n in members.get(node,{node})],axis=0)
        required=[c.axis for c in internal.get(node,[])]
        for i,e in arms:
            c=chains[i];station=portals[i,e]['station']
            required.append(substring(c.axis,0,station) if e==0 else substring(c.axis,station,c.axis.length))
        footprint,method=_junction(local,center,required)
        junctions.append(footprint)
        junction_audit.append(dict(node_xy=center.tolist(),degree=len(arms),method=method,
            graph_node_count=len(members.get(node,{node})),
            internal_edges=[e for c in internal.get(node,[]) for e in c.edges],
            portals=[dict(chain_id=i,end=e,station_m=portals[i,e]['station'],
                          width_m=portals[i,e]['width']) for i,e in arms],area_m2=float(footprint.area)))
    surface=polygonal(union_all([*bodies,*junctions])) if bodies or junctions else Polygon()
    surface=_generalize(surface,junctions,chains)
    summary=dict(original_fragment_count=len(profiles),noded_edge_count=noded_count,
        canonical_chain_count=len(chains),merged_pseudo_node_count=merged+sum(len(v)-1 for v in members.values()),
        junction_internal_chain_count=sum(len(v) for v in internal.values()),roundoff_gap_count=gap_count,
        width_stability_interval_count=sum(r['sections'] for r in chain_audit),
        removed_short_spur_count=sum(r['action']=='suppress' and r['terminal_spur'] for r in spur_audit),
        removed_isolated_short_count=sum(r['action']=='suppress' and not r['terminal_spur'] for r in spur_audit),
        reconstructed_junction_count=sum(r['degree']>=3 for r in junction_audit),
        semantic_boundary_count=sum(r['degree']==2 for r in junction_audit),geometry_valid=bool(surface.is_valid),
        axis_coverage_tolerance_m=1e-4,
        active_axis_covered=bool(surface.buffer(1e-4).covers(union_all([edges[k].axis for k in active]))) if active else True)
    return surface,dict(summary=summary,chains=chain_audit,spurs=spur_audit,
                        junctions=junction_audit,merge_rejections=rejections)
