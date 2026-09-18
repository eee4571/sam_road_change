"""Final road surfaces rebuilt from a noded, attributed road graph.

This module consumes immutable final axes and saved width/image observations.
It does not edit extraction, measurement, change decisions or temporal tracks.
All lengths are metres. Only this renderer owns portals and junction surfaces.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from shapely import union_all, line_interpolate_point, line_locate_point, set_precision
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, polygonize
from shapely.strtree import STRtree
from shapely.prepared import prep

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


def _constrained(row):
    return bool(_value(row,'track_id') or _value(row,'lifecycle_state') or
                _value(row,'protect_surface').lower() in ('1','1.0','true','yes'))


def _collapse_duplicates(profiles, quality, metadata):
    """Choose one observed axis, not an averaged axis between separate roads.

    Only the overlapping part is replaced. Exact branch contacts on the old
    axis follow the replacement; mere nearby endpoints are never snapped.
    Source indices and their saved width/quality profiles remain aligned.
    """
    axes=[p[0] for p in profiles]
    tree=STRtree(axes)
    rank=sorted(range(len(axes)),key=lambda i:(-_constrained(metadata[i]),
        -float(np.mean(np.asarray(quality[i])>=.99)),-axes[i].length,i))
    order={i:k for k,i in enumerate(rank)}
    changed={};reports=[]
    for i in rank:
        axis=axes[i]
        if axis.length<2 or axis.is_ring:continue
        width=float(np.median(profiles[i][2]))
        choices=[]
        for raw in tree.query(axis,predicate='dwithin',distance=min(12.,width*.5)):
            j=int(raw);ref=axes[j]
            if order[j]>=order[i] or j in changed or ref.is_ring:continue
            if _level(metadata[i])!=_level(metadata[j]) or _identity_conflict(metadata[i],metadata[j]):continue
            if _constrained(metadata[i]) and _constrained(metadata[j]) and _value(metadata[i],'track_id')!=_value(metadata[j],'track_id'):continue
            # Both axes must lie within the narrower road's own half-width.
            # A fixed 2m limit misses duplicates of a 20-30m wide road.
            tolerance=min(12.,.5*min(width,float(np.median(profiles[j][2]))))
            ss=np.linspace(0,axis.length,max(9,int(axis.length/3)+1))
            pts=line_interpolate_point(axis,ss)
            loc=line_locate_point(ref,pts)
            projected=line_interpolate_point(ref,loc)
            dist=np.array([a.distance(b) for a,b in zip(pts,projected)])
            inside=dist<=tolerance
            full_overlap=inside.mean()>=.85
            # A common road may diverge only near its far end. Collapse the
            # long matched approach, preserving the distinct tail and fork.
            # Merely close parallel roads with unrelated ends do not qualify.
            shared_end=axis.boundary.distance(ref.boundary)<=min(8.,width*.2)
            if not full_overlap and (inside.mean()<.65 or not shared_end):continue
            valid=np.flatnonzero(inside)
            if not full_overlap:
                valid=max(np.split(valid,np.flatnonzero(np.diff(valid)>1)+1),key=len)
            lo,hi=int(valid[0]),int(valid[-1])
            span=abs(loc[hi]-loc[lo]);length=ss[hi]-ss[lo]
            minimum=max(2.,.85*axis.length) if full_overlap else max(24.,2*min(width,float(np.median(profiles[j][2]))))
            if span<minimum or not .9<=span/max(length,1e-9)<=1.1:continue
            # Pointwise tangent agreement rejects crossing, divergent forks and
            # small U shapes whose endpoint chord happens to be parallel.
            v=np.diff(np.array([p.coords[0] for p in pts[lo:hi+1]]),axis=0)
            w=np.diff(np.array([p.coords[0] for p in projected[lo:hi+1]]),axis=0)
            cosine=np.sum(v*w,axis=1)/np.maximum(np.linalg.norm(v,axis=1)*np.linalg.norm(w,axis=1),1e-9)
            if np.quantile(cosine,.1)<.94:
                if full_overlap:continue
                # End the common approach where its direction diverges. Do not
                # project the fork itself just because it is still nearby.
                aligned=np.flatnonzero(cosine>=.94)
                if not len(aligned):continue
                aligned=max(np.split(aligned,np.flatnonzero(np.diff(aligned)>1)+1),key=len)
                lo,hi=lo+int(aligned[0]),lo+int(aligned[-1])+1
                span=abs(loc[hi]-loc[lo]);length=ss[hi]-ss[lo]
                if span<minimum or not .9<=span/max(length,1e-9)<=1.1:continue
            if np.all(dist<1e-7):continue  # exact overlaps are dissolved by noding
            choices.append((float(np.mean(dist)),order[j],j,ss[lo],ss[hi],loc[lo],loc[hi],full_overlap))
        if not choices:continue
        _,_,j,start,stop,a,b,full_overlap=min(choices)
        ref=axes[j]
        middle=substring(ref,min(a,b),max(a,b))
        xy=list(middle.coords)
        if a>b:xy.reverse()
        if start>1e-7:xy=list(substring(axis,0,start).coords)[:-1]+xy
        if stop<axis.length-1e-7:xy+=list(substring(axis,stop,axis.length).coords)[1:]
        replacement=LineString(xy)
        if not replacement.is_simple:continue
        changed[i]=(j,start,stop,replacement)
        reports.append(dict(source_feature=i,canonical_source=j,overlap_m=float(abs(b-a)),
                            maximum_offset_m=float(tolerance),partial_overlap=not full_overlap,
                            method='observed_axis_projection'))
    if not changed:return profiles,quality,reports
    # Insert exact crossings too: not every contact is already a source vertex.
    contacts=defaultdict(dict)
    for i,(j,start,stop,_) in changed.items():
        old=substring(axes[i],start,stop);ref=axes[j]
        for raw in tree.query(old,predicate='intersects'):
            k=int(raw)
            if k in changed or k==j or _level(metadata[k])!=_level(metadata[i]):continue
            hit=old.intersection(axes[k])
            points=[hit] if hit.geom_type=='Point' else list(hit.geoms) if hit.geom_type=='MultiPoint' else []
            for point in points:
                contacts[k][float(axes[k].project(point))]=ref.interpolate(ref.project(point)).coords[0]
    result=[];grades=[]
    for i,(axis,ss,ww) in enumerate(profiles):
        new=changed[i][3] if i in changed else axis
        if i in contacts:
            stations={float(axis.project(Point(p))):p for p in axis.coords}
            stations.update(contacts[i]);new=LineString([p for _,p in sorted(stations.items())])
        if new.equals_exact(axis,1e-10):result.append((axis,ss,ww));grades.append(quality[i]);continue
        s=np.linspace(0,new.length,max(2,int(np.ceil(new.length/2))+1))
        loc=line_locate_point(axis,line_interpolate_point(new,s))
        result.append((new,s,np.interp(loc,ss,ww)))
        grades.append(np.interp(loc,ss,quality[i]))
    return result,grades,reports


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
    if {str(metadata[i].get('track_id') or '') for i in first.sources} != {str(metadata[i].get('track_id') or '') for i in second.sources}:
        return 'lifecycle_boundary'
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
    # Removing a tooth can expose a previously hidden oscillating join. It is
    # a semantic boundary, not a safe degree-2 continuation to concatenate.
    from .road_axis_quality import axis_quality
    def approach(edge,end):
        length=edge.axis.length
        part=substring(edge.axis,0,min(24.,length)) if end==0 else substring(edge.axis,max(0.,length-24.),length)
        return list(part.coords) if end==0 else list(part.coords)[::-1]
    left=approach(first,fe)[::-1];right=approach(second,se)
    if axis_quality(LineString(left+right[1:]),max(a,b))['abnormal']:
        return 'unsafe_axis_join'
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
            hits = tree.query(probe, predicate='dwithin', distance=1e-5)
            sources = []
            for raw in hits:
                j=int(raw)
                if j not in source_cover:source_cover[j]=prep(axes[j].buffer(1e-5,cap_style='square'))
                if source_cover[j].covers(axis):sources.append(ids[j])
            sources.sort()
            if not sources:
                raise ValueError(f'Noded road edge lost its source correspondence: {axis.wkt}; hits={list(hits)}')
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
            priority=grades+np.array([2. if _constrained(metadata[i]) else 0. for i in sources])[:,None]
            best = np.argmax(priority, axis=0)
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
    """Resolve short components, whole terminal arms and tiny loops.

    Missing observations are unknown, not a reason to keep surface noise. Only
    explicit constraints or sustained positive support override the size rule.
    Recheck newly exposed tips after deleting reconnecting teeth. Protected
    roads are assessed once and cannot be stripped by repeated pruning.
    """
    active=set(range(len(edges)));nodes=_adjacency(edges,active)
    pair={}
    for ends in nodes.values():
        if len(ends)==2 and ends[0][0]!=ends[1][0]:
            pair[ends[0]]=ends[1];pair[ends[1]]=ends[0]
    temporal=[STRtree(lines) for lines in observations if len(lines)]
    ev=None;loaded=False;audit=[];handled=set()

    def decide(ids,kind,axis=None):
        nonlocal ev,loaded
        ids=set(ids);sources=sorted({s for k in ids for s in edges[k].sources})
        length=sum(edges[k].axis.length for k in ids)
        width=float(np.median(np.concatenate([edges[k].widths for k in ids])))
        threshold=min(30.,3*width)
        if kind not in ('loop_remnant','u_remnant','cycle_remnant') and length>threshold:return False
        probe=axis if axis is not None else union_all([edges[k].axis for k in ids])
        if kind in ('loop_remnant','u_remnant','cycle_remnant'):
            # A returning tooth has two arms: its perimeter is not comparable
            # to one terminal spur. Also bound the physical footprint so a real
            # road loop cannot be removed merely because its road is wide.
            if np.hypot(probe.bounds[2]-probe.bounds[0],probe.bounds[3]-probe.bounds[1])>min(20.,2*width):return False
            threshold=min(60.,6*width)
        if length>threshold:return False
        parts=[probe] if probe.geom_type=='LineString' else list(probe.geoms)
        # Trace begins at the tip: discard the supported main-road root when
        # assessing a tooth, rather than counting its crossing as road evidence.
        if kind=='terminal_spur':parts=[substring(parts[0],0,parts[0].length*.65)]
        points=[p for part in parts for p in line_interpolate_point(part,np.linspace(0,part.length,9))]
        reason='topology_short_'+kind
        protected=any(_constrained(metadata[i]) for i in sources)
        if protected:reason='event_constrained'
        elif all(float(np.mean(edges[k].quality>=.99))>=.8 for k in ids):
            protected=True;reason='strong_rgb_support'
        elif any(np.mean([len(t.query(p,predicate='dwithin',distance=min(1.5,width/5)))>0 for p in points])>=.85
                 for t in temporal):
            protected=True;reason='strong_other_period_support'
        score=[]
        if not protected:
            if not loaded:ev=evidence() if callable(evidence) else evidence;loaded=True
            if ev is not None:
                score=[ev.measure(part) for part in parts]
                protected=all(r.get('probability_valid_fraction',0)>=.9 and
                              r.get('probability_q25',0)>=.65 and
                              r.get('probability_support',0)>=.9 for r in score)
                if protected:reason='strong_probability_support'
        audit.append(dict(source_features=sources,graph_edges=sorted(ids),length_m=float(length),
            threshold_m=threshold,terminal_spur=kind=='terminal_spur',kind=kind,
            action='retain' if protected else 'suppress',reason=reason,evidence=score))
        handled.update(ids)
        if not protected:active.difference_update(ids)
        return True

    # Connected components include closed/U-shaped remnants without degree-1
    # tips. Measure the whole component before considering its individual arms.
    unseen=set(active)
    while unseen:
        seed=min(unseen);component=set();queue=[seed];unseen.remove(seed)
        while queue:
            k=queue.pop();component.add(k)
            for node in edges[k].nodes:
                for j,_ in nodes[node]:
                    if j in unseen:unseen.remove(j);queue.append(j)
        decide(component,'component')
    for node,ends in nodes.items():
        if len(ends)!=1 or ends[0][0] in handled:continue
        arm=_chain(edges,_trace(edges,ends[0],pair))
        decide(arm.edges,'terminal_spur',arm.axis)
    # Trace branch-to-branch paths as well: a tiny loop returning to its root,
    # or a narrow U detour beside a direct graph connection, is not a long road.
    seen=set(handled)
    for node,ends in nodes.items():
        if len(ends)<3:continue
        for start in ends:
            if start[0] in seen:continue
            arm=_chain(edges,_trace(edges,start,pair));seen.update(arm.edges)
            a,b=arm.ends
            if a==b:
                decide(arm.edges,'loop_remnant',arm.axis)
            elif Point(a[1:]).distance(Point(b[1:]))<=min(4.,float(np.median(arm.widths))/2):
                # Do not delete a short genuine junction connector. Require an
                # alternative direct path between the same graph endpoints.
                direct=any(edges[k].nodes in ((a,b),(b,a)) and k not in arm.edges for k,_ in nodes[a])
                chord=Point(a[1:]).distance(Point(b[1:]))
                if direct and arm.axis.length>2*max(chord,1.):decide(arm.edges,'u_remnant',arm.axis)
    # Teeth may reconnect to the main road and therefore have no degree-1 tip.
    # Resolve small graph faces explicitly. Keep the shortest connecting route
    # between outside contacts, rather than retaining the detour as a junction
    # whose full width would produce a bulb. Real large road loops are excluded.
    levels=defaultdict(list)
    for k in active:levels[edges[k].level].append(k)
    for ids in levels.values():
        lines=[edges[k].axis for k in ids];tree=STRtree(lines)
        for face in polygonize(lines):
            hits=[ids[int(j)] for j in tree.query(face.boundary,predicate='intersects')]
            boundary=prep(face.boundary.buffer(1e-5))
            ring={k for k in hits if boundary.covers(edges[k].axis)}
            if not ring or not ring<=active:continue
            width=float(np.median(np.concatenate([edges[k].widths for k in ring])))
            if (face.length>min(60.,6*width) or
                np.hypot(face.bounds[2]-face.bounds[0],face.bounds[3]-face.bounds[1])>min(20.,2*width)):continue
            ring_nodes=_adjacency(edges,ring)
            if any(len(v)!=2 for v in ring_nodes.values()):continue
            contacts={node for node in ring_nodes if any(k in active-ring for k,_ in nodes[node])}
            if len(contacts)<=1:
                decide(ring,'cycle_remnant');continue
            paths=[];visited=set()
            for node in sorted(contacts):
                for start in ring_nodes[node]:
                    if start[0] in visited:continue
                    route=[];k,e=start
                    while k not in visited:
                        route.append((k,e));visited.add(k);end=edges[k].nodes[1-e]
                        if end in contacts:break
                        k,e=next(item for item in ring_nodes[end] if item[0]!=k)
                    if route:paths.append(_chain(edges,route))
            if paths:
                detour=max(paths,key=lambda c:c.axis.length)
                decide(detour.edges,'cycle_remnant',detour.axis)
    # Removing a reconnecting tooth can expose a new short dead end. Resolve
    # that topology now, before its stale width becomes a junction footprint.
    while True:
        remaining=_adjacency(edges,active);continuation={}
        for ends in remaining.values():
            if len(ends)==2:
                a,b=ends;continuation[a]=b;continuation[b]=a
        removed=0
        for ends in remaining.values():
            if len(ends)!=1 or ends[0][0] not in active:continue
            route=_chain(edges,_trace(edges,ends[0],continuation))
            ids=set(route.edges)
            if ids&handled:continue
            before=len(active);decide(ids,'terminal_spur',route.axis)
            removed+=before-len(active)
        if not removed:break
    return active,audit


def _repair_degree_two(edges, active, metadata, evidence):
    """Classify faults across pseudo nodes, before directional chain splitting.

    Looking at each already-split piece hid stair steps exactly at piece ends.
    Only classified alternating turns are fitted. True branch contacts, road
    identity boundaries and reliable width changes remain graph boundaries.
    """
    from .road_axis_quality import axis_quality,repair_axis,AxisQualityError,_contacts
    nodes=_adjacency(edges,active);pair={}
    for ends in nodes.values():
        if len(ends)!=2 or ends[0][0]==ends[1][0]:continue
        (i,e),(j,f)=ends;a,b=edges[i],edges[j]
        if a.level!=b.level:continue
        if any(_identity_conflict(metadata[x],metadata[y]) or _constrained(metadata[x]) or
               _constrained(metadata[y]) for x in a.sources for y in b.sources):continue
        wa,wb=a.widths[0 if e==0 else -1],b.widths[0 if f==0 else -1]
        if a.quality[0 if e==0 else -1]>=.99 and b.quality[0 if f==0 else -1]>=.99 and max(wa,wb)>2*min(wa,wb):continue
        pair[i,e]=(j,f);pair[j,f]=(i,e)
    starts=[(k,e) for k in sorted(active) for e in (0,1) if (k,e) not in pair]
    starts += [(k,0) for k in sorted(active)]
    tree=STRtree([e.axis for e in edges]);seen=set();audit=[];loaded=False;ev=None
    for start in starts:
        if start[0] in seen:continue
        route=_trace(edges,start,pair);ids={k for k,_ in route};seen.update(ids)
        if len(route)<2:continue
        chain=_chain(edges,route);old=chain.axis;width=float(np.median(chain.widths))
        qa=axis_quality(old,width)
        if not qa.get('oscillation_stations'):continue
        neighbours=[int(j) for j in tree.query(old,predicate='dwithin',distance=max(4.,2*width))
                    if int(j) in active-ids and edges[int(j)].level==edges[start[0]].level]
        contacts={j:_contacts(old,edges[j].axis) for j in neighbours}
        support=prep(old.buffer(max(2.,width/2)))
        def validate(candidate):
            if not support.covers(candidate):return False
            for j,previous in contacts.items():
                now=_contacts(candidate,edges[j].axis)
                if not now.difference(previous.buffer(1e-5)).is_empty or not previous.difference(now.buffer(1e-5)).is_empty:return False
            return True
        if not loaded:ev=evidence() if callable(evidence) else evidence;loaded=True
        try:fixed,report=repair_axis(old,width,ev,validate,force=True)
        except AxisQualityError:continue
        if not report.get('corrected'):continue
        # Interval fits can individually be simple but meet another part of a
        # returning chain. Accept the assembled axis only after full-chain QA.
        if not fixed.is_simple or axis_quality(fixed,float(np.max(chain.widths)))['abnormal'] or not validate(fixed):continue
        s=np.linspace(0,fixed.length,max(2,int(np.ceil(fixed.length/2))+1))
        location=line_locate_point(old,line_interpolate_point(fixed,s))
        keep=min(ids)
        edges[keep]=Edge(fixed,s,np.interp(location,chain.stations,chain.widths),
            np.interp(location,chain.stations,chain.quality),chain.sources,edges[keep].level,chain.ends)
        active.difference_update(ids-{keep})
        audit.append(dict(graph_edges=sorted(ids),source_features=list(chain.sources),
            stage='before_directional_split',axis_before_wkt=old.wkt,axis_after_wkt=fixed.wkt,**report))
    return audit


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
    broken=0
    from .road_axis_quality import axis_quality
    for start in starts:
        if start[0] in seen:continue
        route=_trace(edges,start,pair);seen.update(k for k,_ in route)
        whole=_chain(edges,route)
        if not axis_quality(whole.axis,float(np.max(whole.widths)))['abnormal']:
            chains.append(whole);continue
        # Several individually acceptable joins can collectively form a fold.
        # Keep the graph boundary where the complete continuation becomes unsafe.
        part=[]
        for step in route:
            trial=_chain(edges,part+[step])
            if part and axis_quality(trial.axis,float(np.max(trial.widths)))['abnormal']:
                chains.append(_chain(edges,part));part=[];broken+=1
                rejects['unsafe_chain_continuation']+=1
            part.append(step)
        if part:chains.append(_chain(edges,part))
    return chains,nodes,len(pair)//2-broken,dict(rejects)


def _junction_complexes(chains, nodes):
    """Treat short, connected junction-internal links as one physical junction.

    Nodes are not spatially snapped. Only an existing short road connection may
    join two branch nodes whose road-width footprints overlap. A diameter bound
    prevents transitive clustering along a street. Interior axes remain audited
    and covered by the reconstructed footprint, not deleted as noise.
    """
    members={node:{node} for node,ends in nodes.items() if len(ends)>=3}
    owner={node:node for node in members}
    # A short branch-to-branch connector can contain several degree-2 pieces.
    # Cluster the whole topological connector, not only single source pieces;
    # otherwise a 6m junction loop becomes four independent round junctions.
    adjacency=defaultdict(list)
    for i,c in enumerate(chains):
        for e,node in enumerate(c.ends):adjacency[node].append((i,e))
    candidates=[];visited=set()
    for node in members:
        for i,e in adjacency[node]:
            if i in visited:continue
            route=[];path_nodes={node};cursor=node
            while i not in visited:
                visited.add(i);c=chains[i];route.append(c)
                cursor=c.ends[1-e];path_nodes.add(cursor)
                if cursor in members or len(adjacency[cursor])!=2:break
                i,e=next(arm for arm in adjacency[cursor] if arm[0]!=i)
            if cursor in members and cursor!=node:
                candidates.append((sum(c.axis.length for c in route),node,cursor,path_nodes,route))
    for length,a,b,path_nodes,route in sorted(candidates,key=lambda v:v[0]):
        if a==b or a not in owner or b not in owner:continue
        ra,rb=owner[a],owner[b]
        width=float(np.median(np.concatenate([c.widths for c in route])))
        if length>min(12.,width/2):continue
        combined=members[ra]|members[rb]|path_nodes
        xy=np.array([n[1:] for n in combined])
        if np.linalg.norm(np.ptp(xy,axis=0))>min(12.,max(4.,width)):continue
        members[ra]=combined
        if ra!=rb:del members[rb]
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
    # The road body and junction use this same stable local tangent. A tiny
    # final source segment must not independently rotate one side of the seam.
    part=substring(axis,station,axis.length) if end==0 else substring(axis,0,station)
    xy=np.asarray(part.coords)
    p=xy[0] if end==0 else xy[-1]
    reach=min(max(1.,float(np.interp(station,chain.stations,chain.widths))*.15),part.length*.5)
    near=part.interpolate(reach if end==0 else part.length-reach)
    d=np.asarray(near.coords[0])-p
    d/=max(np.linalg.norm(d),1e-12)
    width=float(np.interp(station,chain.stations,chain.widths))
    n=np.array([-d[1],d[0]])*width/2
    return dict(point=p,direction=d,right=p-n,left=p+n,width=width,station=float(station),
                root_width=float(chain.widths[0 if end==0 else -1]))


def _coordinate_widths(chains, ends, metadata):
    """Resolve low-confidence widths on physical through-routes before portals.

    Reliable plateaux and lifecycle widths remain fixed. Unknown pieces borrow
    a length-weighted route representative, not an average of two endpoints.
    """
    pairs=[];audit=[]
    for node,arms in ends.items():
        choices={}
        for i,e in arms:
            c=chains[i]
            if any(_constrained(metadata[k]) for k in c.sources):continue
            scores=[]
            for j,f in arms:
                if i==j:continue
                d=chains[j]
                if any(_constrained(metadata[k]) for k in d.sources):continue
                if any(_identity_conflict(metadata[x],metadata[y]) for x in c.sources for y in d.sources):continue
                cosine=-float(_outward(c,e,8)@_outward(d,f,8))
                if cosine>=(.71 if len(arms)==2 else .90):scores.append((cosine,j,f))
            if scores:
                scores.sort(reverse=True)
                if len(scores)==1 or scores[0][0]-scores[1][0]>.04:choices[(i,e)]=scores[0][1:]
        for arm,other in choices.items():
            if arm>=other or choices.get(other)!=arm:continue
            pairs.append((node,arm,other))
    neighbours=defaultdict(set)
    for (_, (i,_),(j,_)) in pairs:neighbours[i].add(j);neighbours[j].add(i)
    unseen=set(neighbours)
    while unseen:
        first=min(unseen);unseen.remove(first);group=[];queue=[first]
        while queue:
            i=queue.pop();group.append(i)
            for j in neighbours[i]:
                if j in unseen:unseen.remove(j);queue.append(j)
        reliable=[i for i in group if np.mean(chains[i].quality>=.99)>=.6]
        donors=reliable or group
        samples=sorted((float(np.median(chains[i].widths)),chains[i].axis.length) for i in donors)
        total=sum(length for _,length in samples);acc=0.;representative=samples[-1][0]
        for width,length in samples:
            acc+=length
            if acc>=total/2:representative=width;break
        for i in group:
            c=chains[i]
            if np.mean(c.quality>=.99)>=.6:continue
            before=float(np.median(c.widths))
            if abs(before-representative)<.1:continue
            c.widths=np.full(len(c.widths),representative)
            audit.append(dict(method='reliable_route_width' if reliable else 'unverified_route_representative',
                chains=[i],route_chains=sorted(group),width_before_m=[before],width_m=representative))
    # Existing reliable changes keep their plateaux. Only the shared end region
    # is blended; weak width outliers have already been corrected over the chain.
    targets=defaultdict(dict)
    for node,(i,e),(j,f) in pairs:
        c,d=chains[i],chains[j]
        a=float(c.widths[0 if e==0 else -1]);b=float(d.widths[0 if f==0 else -1])
        if abs(a-b)<.1:continue
        qa=float(np.mean(c.quality>=.99));qb=float(np.mean(d.quality>=.99))
        target=a if qa>=.6 and qb<.6 else b if qb>=.6 and qa<.6 else (a+b)/2
        targets[i][e]=target;targets[j][f]=target
        audit.append(dict(method='reliable_plateau_transition',node_xy=list(node[1:]),chains=[i,j],
            width_before_m=[a,b],width_m=target))
    for i,values in targets.items():
        c=chains[i];original=c.widths.copy()
        for end,target in values.items():
            distance=c.stations if end==0 else c.axis.length-c.stations
            span=min(c.axis.length*.4,max(8.,2*target))
            start=original[0 if end==0 else -1]
            changed=np.flatnonzero(abs(original-start)>max(.5,.1*start))
            if len(changed):span=min(span,float(np.min(distance[changed])))
            t=np.clip(distance/max(span,1e-6),0,1);blend=1-t*t*(3-2*t)
            c.widths+=(target-start)*blend
    return audit


def _junction(portals, center, required_axes):
    """Construct arms with the exact same cross-sections as their road bodies.

    No cyclic portal polygon is clipped against independently buffered stubs.
    That mismatch made valid polygons with rectangular gaps at their seams.
    """
    stubs=required_axes[-len(portals):];minimum=min(p['width'] for p in portals)/2
    envelopes=[];collars=[];roots={}
    for axis,p in zip(stubs,portals):
        xy=list(axis.coords)
        if np.linalg.norm(np.asarray(xy[0])-p['point'])<np.linalg.norm(np.asarray(xy[-1])-p['point']):xy.reverse()
        axis=LineString(xy)
        root_width=p.get('root_width',p['width'])
        envelopes.append(_chain_surface(axis,[0.,axis.length],[root_width,p['width']],
            end_direction=p['direction'],check_axis=False))
        key=tuple(np.round(xy[0],6));roots[key]=max(roots.get(key,0),root_width/2)
        # A shared, narrow collar is part of the portal definition, not a fill
        # over an arbitrary gap. It makes the entire cap common to both sides.
        eps=p['direction']*.025
        collars.append(Polygon([p['right']-eps,p['left']-eps,p['left']+eps,p['right']+eps]))
    envelopes.extend(Point(xy).buffer(radius) for xy,radius in roots.items())
    envelopes.extend(axis.buffer(minimum,cap_style='round') for axis in required_axes[:-len(portals)])
    footprint=polygonal(union_all(envelopes))
    # Local concave corner fillets, independently on each connected footprint.
    # A nearby unrelated arm is not bridged and no portal convex hull is used.
    filleted=[];radius=min(2.,minimum*.35)
    for part in _polygons(footprint):
        rounded=part.buffer(radius).buffer(-radius)
        filleted.append(rounded.intersection(part.buffer(radius*.55)))
    footprint=polygonal(union_all([*filleted,*collars]))
    return footprint,('shared_portal_sweep' if len(_polygons(footprint))==1 else 'local_branch_envelope')


def _repair_canonical_shapes(chains, original_widths, evidence):
    """Reuse classified local axis repair, pinning all actual graph contacts."""
    from .road_axis_quality import axis_quality,repair_axis,AxisQualityError,_contacts
    tree=STRtree([c.axis for c in chains]);audit=[];ev=None;loaded=False
    for i,c in enumerate(chains):
        width=float(np.max(c.widths));qa=axis_quality(c.axis,width)
        if not qa.get('oscillation_stations') and not qa['abnormal']:continue
        old=c.axis
        neighbours=[int(j) for j in tree.query(old,predicate='dwithin',distance=width*2)
                    if int(j)!=i and chains[int(j)].ends[0][0]==c.ends[0][0]]
        contacts={j:_contacts(old,chains[j].axis) for j in neighbours}
        supported=prep(old.buffer(max(2.,width/2)))
        def validate(candidate):
            if not supported.covers(candidate):return False
            for j,previous in contacts.items():
                now=_contacts(candidate,chains[j].axis)
                if not now.difference(previous.buffer(1e-5)).is_empty or not previous.difference(now.buffer(1e-5)).is_empty:return False
            return True
        if not loaded:ev=evidence() if callable(evidence) else evidence;loaded=True
        try:
            fixed,report=repair_axis(old,width,ev,validate,force=True)
            if not fixed.is_simple or axis_quality(fixed,width)['abnormal'] or not validate(fixed):
                raise AxisQualityError('Surface chain fit violates complete-chain topology')
        except AxisQualityError:
            # A width transfer must not introduce an unrepairable axis risk.
            # Restore this chain's own stable width; do not move a real contact.
            if qa['abnormal']:c.widths=original_widths[i].copy()
            audit.append(dict(chain_id=i,corrected=False,quality_state='contact_constrained',
                width_transfer_reverted=qa['abnormal']))
            continue
        if not report.get('corrected'):continue
        stations=np.linspace(0,fixed.length,max(2,int(np.ceil(fixed.length/2))+1))
        old_locations=line_locate_point(old,line_interpolate_point(fixed,stations))
        c.widths=np.interp(old_locations,c.stations,c.widths)
        c.quality=np.interp(old_locations,c.stations,c.quality)
        c.axis=fixed;c.stations=stations
        audit.append(dict(chain_id=i,axis_before_wkt=old.wkt,axis_after_wkt=fixed.wkt,**report))
    return audit


def _polygons(geometry):
    geometry=polygonal(geometry)
    return [geometry] if geometry.geom_type=='Polygon' and not geometry.is_empty else list(geometry.geoms) if not geometry.is_empty else []


def _chain_surface(axis, stations, widths, *, start_direction=None, end_direction=None, check_axis=True):
    """Continuous swept cross-sections of one canonical chain.

    A single offset boundary polygon is unsafe at tight bends: make_valid can
    interpret its winding as holes even where the axis is present. Construct
    the occupied envelope before any overlay instead. These are sampling cells
    of one stable width function, never the original fragment corridors.
    """
    from .road_axis_quality import require_safe_axis
    if check_axis:require_safe_axis(axis,float(np.max(widths)))
    xy=np.asarray(axis.coords)
    vertex=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    ss=np.unique(np.r_[stations,vertex])
    points=np.asarray([p.coords[0] for p in line_interpolate_point(axis,ss)])
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8]
    points=points[keep];radius=np.interp(ss[keep],stations,widths)/2
    d=np.diff(points,axis=0);d/=np.linalg.norm(d,axis=1)[:,None]
    if np.ptp(widths)<1e-8:
        # The constant-width occupied sweep has an exact GEOS construction.
        # Unlike an offset-boundary polygon it cannot discard axis interiors;
        # avoid unioning thousands of redundant sample cells for this case.
        surface=axis.buffer(float(radius[0]),cap_style='round',join_style='round',quad_segs=12)
    else:
        normal=np.c_[-d[:,1],d[:,0]]
        left=normal.copy();right=normal.copy()
        if start_direction is not None:left[0]=[-start_direction[1],start_direction[0]]
        if end_direction is not None:right[-1]=[-end_direction[1],end_direction[0]]
        cells=[polygonal(Polygon([a+n*r,b+m*t,b-m*t,a-n*r]).buffer(0)) for a,b,n,m,r,t in
               zip(points[:-1],points[1:],left,right,radius[:-1],radius[1:])]
        # Round joins only at actual turns; a straight width ramp needs no discs.
        for k in range(1,len(points)-1):
            turn=float(d[k-1,0]*d[k,1]-d[k-1,1]*d[k,0])
            if abs(turn)>1e-8:cells.append(Point(points[k]).buffer(float(radius[k]),quad_segs=12))
        if start_direction is not None:cells.append(Point(points[0]).buffer(float(radius[0]),quad_segs=12))
        if end_direction is not None:cells.append(Point(points[-1]).buffer(float(radius[-1]),quad_segs=12))
        surface=polygonal(union_all(cells))
    # Discs near the ends must not extend beyond their exact flat portal caps.
    # Remove only the local outside half-plane; a hairpin may legitimately
    # return behind a portal far away along the same chain.
    first=d[0] if start_direction is None else np.asarray(start_direction)
    last=d[-1] if end_direction is None else np.asarray(end_direction)
    caps=[]
    for p,t,r,explicit in ((points[0],first,radius[0],start_direction),
                            (points[-1],-last,radius[-1],end_direction)):
        if explicit is None:continue
        n=np.array([-t[1],t[0]])*r;eps=t*.025
        caps.append(Polygon([p-n-eps,p+n-eps,p+n+eps,p-n+eps]))
    if caps:surface=polygonal(union_all([surface,*caps]))
    for p,t,r in ((points[0],first,radius[0]),(points[-1],-last,radius[-1])):
        n=np.array([-t[1],t[0]]);reach=2*r
        outside=Polygon([p+n*reach,p-n*reach,p-n*reach-t*reach,p+n*reach-t*reach])
        # Preserve any remote part of the axis behind the cap.
        if outside.buffer(-1e-5).intersects(axis):continue
        surface=polygonal(surface.difference(outside))
    return surface


def _generalize(geometry, junctions, chains):
    audit=dict(micro_holes_removed=0,slivers_removed=0,narrow_cracks_closed=0,
               generalized_components=0)
    if geometry.is_empty:return geometry,audit
    axes=[c.axis for c in chains];tree=STRtree(axes)
    endpoint_degree=Counter(node for c in chains for node in c.ends)
    junction_tree=STRtree(junctions)
    result=[]
    # Work within each component: never close the gap between parallel roads.
    for part in _polygons(geometry):
        ids=[int(i) for i in tree.query(part,predicate='intersects')]
        if not ids:
            audit['slivers_removed']+=1;continue
        width=float(np.median([np.median(chains[i].widths) for i in ids]))
        radius=min(.8,max(.15,width*.08));tolerance=min(.7,max(.15,width*.07))
        holes=[];protected_holes=[]
        for ring in part.interiors:
            hole=Polygon(ring)
            micro=hole.area<=min(12.,max(1.,width*width*.1))
            slit=(hole.area<=max(20.,width*width*.8) and hole.buffer(-.25).is_empty)
            if micro or slit:audit['micro_holes_removed']+=1
            else:holes.append(ring);protected_holes.append(hole)
        clean=Polygon(part.exterior,holes)
        # Closing removes sub-metre cracks; opening removes small projections.
        # Both are component-local and are later constrained by the axis core,
        # exact junction footprints and the surviving real loop holes.
        closed=clean.buffer(radius,join_style='round').buffer(-radius,join_style='round')
        cracks=_polygons(closed.difference(clean))
        crack_count=sum(p.area>.02 for p in cracks)
        candidate=closed.buffer(-radius*.65).buffer(radius*.65).simplify(tolerance,preserve_topology=True)
        # A tolerance-only simplifier may replace a 100m straight boundary by
        # a slightly oblique chord. Accept edits on the scale of a road width,
        # not long side replacements; the old 2m/.25m limits no longer apply.
        limit=max(12.,3*width)
        local=lambda delta:[p for p in _polygons(set_precision(delta,1e-5)) if
            np.hypot(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1])<=limit]
        candidate=union_all([clean.difference(union_all(local(clean.difference(candidate)))),
                             *local(candidate.difference(clean))])
        local_axes=union_all([axes[i] for i in ids])
        core=clean.intersection(local_axes.buffer(min(1.,width*.15)))
        protected=[junctions[int(i)] for i in junction_tree.query(part,predicate='intersects')]
        # Flat end caps are a graph boundary, not a corner to round away.
        protected.extend(Point(chains[i].axis.coords[0 if e==0 else -1]).buffer(width)
                         for i in ids for e in (0,1) if endpoint_degree[chains[i].ends[e]]==1)
        if protected:
            mask=union_all(protected).buffer(.1)
            candidate=union_all([candidate.difference(mask),clean.intersection(mask)])
        candidate=polygonal(union_all([candidate,core]))
        if protected_holes:candidate=polygonal(candidate.difference(union_all(protected_holes)))
        # Overlay at protected masks can itself introduce micron-sized seams.
        repaired=[];overlay_holes=0;slivers=0
        for piece in _polygons(candidate):
            rings=[]
            for ring in piece.interiors:
                hole=Polygon(ring)
                if hole.area<1e-6 or (hole.area<1. and hole.buffer(-.01).is_empty):
                    overlay_holes+=1
                else:rings.append(ring)
            repaired.append(Polygon(piece.exterior,rings))
        candidate=polygonal(union_all(repaired))
        # Keep all real axes and connectivity; remove only detached numeric
        # crumbs. Boundary displacement is width-relative, no 2m edit-box cap.
        pieces=[]
        for piece in _polygons(candidate):
            if piece.buffer(1e-6).intersects(local_axes):pieces.append(piece)
            else:slivers+=1
        candidate=polygonal(union_all(pieces))
        if (candidate.is_valid and candidate.buffer(1e-4).covers(local_axes) and
            len(_polygons(candidate))==1 and
            candidate.difference(clean.buffer(radius*2+tolerance)).area<1e-7 and
            clean.difference(candidate.buffer(radius*2+tolerance)).area<1e-7):
            result.append(candidate);audit['generalized_components']+=1
            audit['micro_holes_removed']+=overlay_holes
            audit['slivers_removed']+=slivers
            audit['narrow_cracks_closed']+=crack_count
        else:result.append(clean)
    final=[]
    # The final component union can produce additional overlay microholes.
    # Filter only these sub-square-metre residues, preserving real loop holes.
    for part in _polygons(union_all(result)):
        rings=[]
        for ring in part.interiors:
            if Polygon(ring).area<1.:audit['micro_holes_removed']+=1
            else:rings.append(ring)
        final.append(Polygon(part.exterior,rings))
    return polygonal(union_all(final)),audit


def build_road_geometry(profiles, quality, *, metadata=None, evidence=None, observations=()):
    """Return surface, surviving canonical chains and audit from one graph."""
    metadata=list(metadata) if metadata is not None else [{} for _ in profiles]
    if len(profiles)!=len(quality) or len(profiles)!=len(metadata):raise ValueError('Road profile metadata count mismatch')
    profiles=[(a,np.clip(np.asarray(s,float),0.,a.length),np.asarray(w,float)) for a,s,w in profiles]
    profiles,quality,duplicate_audit=_collapse_duplicates(profiles,quality,metadata)
    edges=_build_graph(profiles,quality,metadata)
    noded_count=len(edges)
    gap_count=_join_roundoff_gaps(edges,metadata)
    active,spur_audit=_prune_terminals(edges,metadata,evidence,observations)
    graph_axis_audit=_repair_degree_two(edges,active,metadata,evidence)
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
    original_widths=[c.widths.copy() for c in chains]
    width_audit=_coordinate_widths(chains,ends,metadata)
    axis_audit=graph_axis_audit+_repair_canonical_shapes(chains,original_widths,evidence)
    for c,report in zip(chains,chain_audit):
        report.update(axis_wkt=c.axis.wkt,length_m=c.axis.length,
                      final_width_range_m=[float(c.widths.min()),float(c.widths.max())])
    trim=np.zeros((len(chains),2))
    portals={}
    for node,arms in ends.items():
        if len(arms)<2 and not internal.get(node):continue
        # Closed chains own no junction unless another road meets the ring.
        if len(arms)==2 and arms[0][0]==arms[1][0] and not internal.get(node):continue
        for i,e in arms:
            c=chains[i];width=c.widths[0 if e==0 else -1]
            center=np.mean([n[1:] for n in members.get(node,{node})],axis=0)
            offset=np.linalg.norm(np.asarray(c.axis.coords[0 if e==0 else -1])-center)
            trim[i,e]=min(max(1.,float(width)*.5)+offset,c.axis.length*.35)
            portals[(i,e)]=_portal(c,e,trim[i,e])
    bodies=[]
    for i,c in enumerate(chains):
        a,b=trim[i,0],c.axis.length-trim[i,1]
        axis=substring(c.axis,a,b)
        ss=np.unique(np.r_[a,c.stations[(c.stations>a)&(c.stations<b)],b])
        bodies.append(_chain_surface(axis,ss-a,np.interp(ss,c.stations,c.widths),
            start_direction=portals[i,0]['direction'] if (i,0) in portals else None,
            end_direction=-portals[i,1]['direction'] if (i,1) in portals else None))
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
    surface,cleanup_audit=_generalize(surface,junctions,chains)
    summary=dict(original_fragment_count=len(profiles),noded_edge_count=noded_count,
        canonical_chain_count=len(chains),merged_pseudo_node_count=merged+sum(len(v)-1 for v in members.values())+
            sum(len(r['graph_edges'])-1 for r in graph_axis_audit),
        junction_internal_chain_count=sum(len(v) for v in internal.values()),roundoff_gap_count=gap_count,
        width_stability_interval_count=sum(r['sections'] for r in chain_audit),
        removed_short_spur_count=sum(r['action']=='suppress' and r['terminal_spur'] for r in spur_audit),
        removed_isolated_short_count=sum(r['action']=='suppress' and r['kind']=='component' for r in spur_audit),
        removed_short_component_count=sum(r['action']=='suppress' and r['kind']=='component' for r in spur_audit),
        removed_short_remnant_count=sum(r['action']=='suppress' and r['kind'] in ('u_remnant','loop_remnant','cycle_remnant') for r in spur_audit),
        near_duplicate_merge_count=len(duplicate_audit),width_coordination_count=len(width_audit),
        surface_axis_repair_count=sum(r['corrected'] for r in axis_audit),
        junction_fallback_types=dict(Counter(r['method'] for r in junction_audit)),
        micro_holes_removed=cleanup_audit['micro_holes_removed'],slivers_removed=cleanup_audit['slivers_removed'],
        narrow_cracks_closed=cleanup_audit['narrow_cracks_closed'],
        reconstructed_junction_count=sum(r['degree']>=3 for r in junction_audit),
        semantic_boundary_count=sum(r['degree']==2 for r in junction_audit),geometry_valid=bool(surface.is_valid),
        axis_coverage_tolerance_m=1e-4,
        active_axis_covered=bool(surface.buffer(1e-4).covers(union_all([c.axis for c in chains]+
            [c.axis for group in internal.values() for c in group]))) if active else True)
    canonical = chains + [c for group in internal.values() for c in group]
    # Junction-internal links remain real centerline connections. They are not
    # separate corridor bodies, but their exported profiles still need the same
    # quality-constrained representative-width rule.
    for group in internal.values():
        for c in group:
            c.widths,_=stable_width(c.stations,c.widths,c.quality>=.99)
    summary['formal_centerline_count']=len(canonical)
    return surface,canonical,dict(summary=summary,chains=chain_audit,spurs=spur_audit,
                        junctions=junction_audit,merge_rejections=rejections,
                        near_duplicates=duplicate_audit,width_coordination=width_audit,polygon_cleanup=cleanup_audit,
                        surface_axis_repairs=axis_audit)


def build_road_surface(profiles, quality, *, metadata=None, evidence=None, observations=()):
    """Surface-only API retained for geometry consumers."""
    surface, _, audit = build_road_geometry(profiles, quality, metadata=metadata,
                                          evidence=evidence, observations=observations)
    return surface, audit
