from __future__ import annotations

"""Infer paired road trajectories from extended geometric support in metres.

Individual endpoint tangents are deliberately not the unit of evidence here:
short extraction fragments can turn across a median while the road stays straight.
"""
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree
from shapely.ops import substring


@dataclass
class TrackCorridor:
    origin: np.ndarray
    axis: np.ndarray
    offsets: np.ndarray
    start: float
    end: float
    support: float
    stations: np.ndarray | None = None
    centers: np.ndarray | None = None
    spacings: np.ndarray | None = None

    @property
    def normal(self):
        return np.asarray([-self.axis[1], self.axis[0]])

    @property
    def spacing(self):
        return float(self.offsets[1] - self.offsets[0])

    def coordinates(self, points):
        delta = np.asarray(points) - self.origin
        return delta @ self.axis, delta @ self.normal

    def offsets_at(self, station):
        s = np.asarray(station)
        if self.stations is None:
            return np.broadcast_to(self.offsets,s.shape+(2,))
        center = np.interp(s,self.stations,self.centers)
        spacing = np.interp(s,self.stations,self.spacings)
        return center[...,None] + spacing[...,None]*np.asarray([-.5,.5])

    def point(self, station, lane):
        s = np.asarray(station)
        return self.origin+s[...,None]*self.axis+self.offsets_at(s)[...,lane,None]*self.normal


def _follow_paired_tracks(model, points, tangents):
    """Jointly follow two lateral modes, preserving their order through gaps."""
    station = points @ model.axis
    lateral = points @ model.normal
    parallel = abs(tangents @ model.axis)>np.cos(np.deg2rad(30))
    stations = np.arange(model.start,model.end+40,40.)
    centers = np.mean(model.offsets)+np.arange(-45.,46.,1.5)
    spacings = np.unique(np.r_[model.spacing,np.arange(max(8,model.spacing*.7),min(65,model.spacing*1.8)+1,2.)])
    center_grid, spacing_grid = np.meshgrid(centers,spacings,indexing='ij')
    scores = []
    for s in stations:
        mask = parallel & (abs(station-s)<65) & (abs(lateral-np.mean(model.offsets))<90)
        values = lateral[mask]
        weights = np.maximum(0,1-abs(station[mask]-s)/65)
        lane_scores = []
        for sign in (-.5,.5):
            distance = abs(values[:,None,None]-(center_grid+sign*spacing_grid))
            evidence = np.where(distance<=4.5,np.exp(-.5*(distance/3.5)**2),0.)
            lane_scores.append(np.minimum(12,(evidence*weights[:,None,None]).sum(axis=0)))
        scores.append(2*np.minimum(*lane_scores))
    cost = -.8*np.asarray(scores)
    current = cost[0]+.001*(center_grid-np.mean(model.offsets))**2+.005*(spacing_grid-model.spacing)**2
    backs = []
    count_c,count_w = current.shape
    ids = np.arange(current.size).reshape(current.shape)
    for local in cost[1:]:
        best = np.full_like(current,np.inf)
        back = np.zeros(current.shape,dtype=np.int32)
        for dc in range(-3,4):
            for dw in range(-1,2):
                dest = (slice(max(0,dc),min(count_c,count_c+dc)),slice(max(0,dw),min(count_w,count_w+dw)))
                src = (slice(max(0,-dc),min(count_c,count_c-dc)),slice(max(0,-dw),min(count_w,count_w-dw)))
                proposal = current[src]+.4*dc**2+.4*(spacing_grid[dest]-spacing_grid[src])**2
                improve = proposal<best[dest]
                best[dest] = np.minimum(best[dest],proposal)
                back[dest] = np.where(improve,ids[src],back[dest])
        current = local+best
        backs.append(back.ravel())
    index = int(current.argmin())
    path = [index]
    for back in backs[::-1]:
        index = int(back[index]); path.append(index)
    ci,wi = np.unravel_index(np.asarray(path[::-1]),current.shape)
    model.stations = stations
    model.centers = gaussian_filter1d(centers[ci],1.2,mode='nearest')
    model.spacings = gaussian_filter1d(spacings[wi],1.2,mode='nearest')


def infer_track_corridors(roads, maximum_gap=300.0):
    if not roads:
        return []
    origin = np.mean(np.vstack([r.points[[0, -1]] for r in roads]), axis=0)
    samples, directions, seeds = [], [], []
    for road in roads:
        line = LineString(road.points - origin)
        if line.length < 4:
            continue
        positions = np.linspace(0, line.length, max(2, int(line.length / 4) + 1))
        points = np.asarray([line.interpolate(t).coords[0] for t in positions])
        tangents = np.asarray([np.asarray(line.interpolate(min(line.length, t+12)).coords[0]) -
                               np.asarray(line.interpolate(max(0, t-12)).coords[0]) for t in positions])
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1)[:, None], 1e-9)
        samples.extend(points)
        directions.extend(tangents)
        chord = points[-1] - points[0]
        if line.length >= 60 and np.linalg.norm(chord) / line.length > .92:
            seeds.append((line.length, points.mean(axis=0), chord / np.linalg.norm(chord)))
        for a,b in zip(road.points,road.points[1:]):
            delta = b-a
            length = np.linalg.norm(delta)
            if length>=60:
                seeds.append((length,(a+b)*.5-origin,delta/length))
    if not seeds:
        return []
    points, tangents = np.asarray(samples), np.asarray(directions)
    models = []
    for _, center, axis in sorted(seeds, key=lambda s: -s[0]):
        if axis[0] < -1e-8 or (abs(axis[0])<=1e-8 and axis[1]<0):
            axis = -axis
        mask = np.zeros(len(points), dtype=bool)
        for iteration in range(8):
            normal = np.asarray([-axis[1], axis[0]])
            along = (points-center) @ axis
            mask = (np.abs((points-center) @ normal) < (8 if iteration==0 else 5)) & (np.abs(tangents @ axis) > np.cos(np.deg2rad(25)))
            ids = np.flatnonzero(mask)
            if len(ids) < 20:
                break
            ordered = ids[np.argsort(along[ids])]
            chunks = np.split(ordered, np.flatnonzero(np.diff(along[ordered]) > maximum_gap)+1)
            chunk = min(chunks, key=lambda c: np.min(np.abs(along[c])))
            mask[:] = False
            mask[chunk] = True
            center = np.mean(points[mask], axis=0)
            _, vectors = np.linalg.eigh(np.cov((points[mask]-center).T))
            axis = vectors[:, -1]
            if axis[0] < -1e-8 or (abs(axis[0])<=1e-8 and axis[1]<0):
                axis = -axis
        if mask.sum() < 40:
            continue
        normal = np.asarray([-axis[1], axis[0]])
        s = points[mask] @ axis
        start, end = np.quantile(s, [.01, .99])
        coverage = len(np.unique(np.floor(s/12))) * 12
        if end-start < 300 or coverage < 180:
            continue
        rho = float(center @ normal)
        duplicate = False
        for other in models:
            if abs(axis @ other['axis']) > np.cos(np.deg2rad(3)) and abs((center-other['center']) @ normal) < 7:
                if min(end, other['end']) - max(start, other['start']) > .4 * min(end-start, other['end']-other['start']):
                    duplicate = True
                    break
        if not duplicate:
            models.append(dict(axis=axis, center=center, rho=rho, start=start, end=end, coverage=coverage, mask=mask.copy()))
    pairs = []
    for i, first in enumerate(models):
        for j, second in enumerate(models[i+1:], i+1):
            if first['axis'] @ second['axis'] < np.cos(np.deg2rad(4)):
                continue
            axis = first['axis'] + second['axis']
            axis /= np.linalg.norm(axis)
            normal = np.asarray([-axis[1], axis[0]])
            s1, s2 = points[first['mask']] @ axis, points[second['mask']] @ axis
            overlap = min(s1.max(), s2.max()) - max(s1.min(), s2.min())
            offsets = np.sort([np.median(points[m['mask']] @ normal) for m in (first, second)])
            spacing = offsets[1] - offsets[0]
            if not 8 <= spacing <= 65 or overlap < 220:
                continue
            # A converging fork is not evidence of two stable parallel tracks.
            drift = np.sqrt(max(0, 1-(first['axis'] @ second['axis'])**2)) * overlap
            if drift > spacing*.6:
                continue
            score = min(first['coverage'], second['coverage']) * overlap / max(np.ptp(s1), np.ptp(s2))
            corridor = TrackCorridor(origin, axis, offsets, min(s1.min(), s2.min()), max(s1.max(), s2.max()), score)
            pairs.append((score, i, j, corridor))
    result, used = [], set()
    for _, i, j, corridor in sorted(pairs, key=lambda p: -p[0]*(p[3].end-p[3].start)):
        if i in used or j in used:
            continue
        middle = corridor.origin + .5*(corridor.start+corridor.end)*corridor.axis + np.mean(corridor.offsets)*corridor.normal
        duplicate = False
        for existing in result:
            if abs(existing.axis @ corridor.axis) < np.cos(np.deg2rad(12)):
                continue
            s, t = existing.coordinates(middle)
            if existing.start-100 < s < existing.end+100 and abs(t-np.mean(existing.offsets)) < .6*(corridor.spacing+existing.spacing):
                duplicate = True
                break
        if duplicate:
            continue
        used.update((i, j))
        result.append(corridor)
    for corridor in result:
        s = points @ corridor.axis
        t = points @ corridor.normal
        mask = (abs(tangents @ corridor.axis)>np.cos(np.deg2rad(25))) & (t>corridor.offsets[0]-35) & (t<corridor.offsets[1]+35)
        candidates = np.sort(s[mask])
        # Continue the road corridor beyond the seed fragments; either side
        # can establish its extent when the other side is locally missing.
        for values,at_start in ((candidates[::-1],True),(candidates,False)):
            edge = corridor.start if at_start else corridor.end
            initial = edge
            for value in values:
                gap = edge-value if at_start else value-edge
                if gap<=0:
                    continue
                if gap>maximum_gap or abs(value-initial)>maximum_gap*2:
                    break
                edge = value
            if at_start:
                corridor.start = edge
            else:
                corridor.end = edge
        _follow_paired_tracks(corridor,points,tangents)
    return result


def restore_track_corridors(roads, corridors, maximum_gap=300.0):
    """Retain supported axial pieces and fill gaps in their longitudinal order.

    Unstable pieces between the two trajectories are replaced, including old
    Y-shaped extraction merges. Transverse road bodies stay in the network.
    Every removed piece is returned for explicit geometry-change auditing.
    """
    from shapely.ops import unary_union
    from .road_geometry import _RegionalRoadSeed, _tangent_continuous_connector

    observed = [LineString(r.points) for r in roads]
    observed_index = STRtree(observed)

    def observed_continuation(first,second):
        """A model must never shortcut a continuous observed road body."""
        for raw_i in observed_index.query(Point(first).buffer(.01)):
            line = observed[int(raw_i)]
            if line.distance(Point(first))>.01 or line.distance(Point(second))>.01:
                continue
            a,b = line.project(Point(first)),line.project(Point(second))
            if abs(a-b)>np.linalg.norm(second-first)*1.5+.01:
                continue
            part = substring(line,a,b)
            if part.geom_type=='LineString':
                return np.asarray(part.coords)
        return None

    pieces = {(i, lane): [] for i in range(len(corridors)) for lane in (0, 1)}
    outside, owned = [], []
    for road in roads:
        pending = []
        fixed_lanes = {}
        merged_tips = []
        chord = road.points[-1]-road.points[0]
        total_length = LineString(road.points).length
        if total_length>=300 and np.linalg.norm(chord)>total_length*.9:
            source_line = LineString(road.points)
            for start in (True,False):
                point = road.points[0 if start else -1]
                inward = np.asarray(source_line.interpolate(min(40,total_length) if start else max(0,total_length-40)).coords[0])-point
                inward /= max(np.linalg.norm(inward),1e-9)
                for raw_j in observed_index.query(Point(point).buffer(.01)):
                    other = observed[int(raw_j)]
                    if other.equals(source_line):
                        continue
                    for other_start in (True,False):
                        other_point = np.asarray(other.coords[0 if other_start else -1])
                        if np.linalg.norm(other_point-point)>.01:
                            continue
                        direction = np.asarray(other.interpolate(min(40,other.length) if other_start else max(0,other.length-40)).coords[0])-point
                        if direction @ inward/max(np.linalg.norm(direction),1e-9)>np.cos(np.deg2rad(50)):
                            merged_tips.append(Point(point))
            for i,model in enumerate(corridors):
                if abs(chord @ model.axis)/np.linalg.norm(chord)<np.cos(np.deg2rad(20)):
                    continue
                ss,tt = model.coordinates(road.points)
                inside = (ss>=model.start)&(ss<=model.end)
                if inside.sum()<2 or np.ptp(ss[inside])<200:
                    continue
                offsets = model.offsets_at(ss[inside])
                errors = np.median(abs(tt[inside,None]-offsets),axis=0)
                if min(errors)<=max(15,model.spacing*.35):
                    fixed_lanes[i] = int(np.argmin(errors))
        for a, b in zip(road.points, road.points[1:]):
            vector = b-a
            length = np.linalg.norm(vector)
            if length < 1e-8:
                continue
            choices = []
            for i, model in enumerate(corridors):
                s, t = model.coordinates([a, b])
                if min(s) < model.start or max(s) > model.end:
                    continue
                offsets = model.offsets_at(s)
                margin = max(5,model.spacing*.6) if i in fixed_lanes else 5.0
                if np.any(t < offsets[:,0]-margin) or np.any(t > offsets[:,1]+margin):
                    continue
                axial = abs(vector @ model.axis)/length
                # Short internal crossbars belong to a converging extraction
                # loop only when the entire source is contained in the median.
                source_s, source_t = model.coordinates(road.points)
                source_offsets = model.offsets_at(source_s)
                internal = (LineString(road.points).length < model.spacing*1.5 and axial>np.cos(np.deg2rad(70)) and
                            np.all(source_t > source_offsets[:,0]-5) and np.all(source_t < source_offsets[:,1]+5))
                if axial < np.cos(np.deg2rad(45)) and not internal:
                    continue
                lane = fixed_lanes.get(i,int(np.argmin(np.mean(abs(t[:,None]-offsets),axis=0))))
                error = max(abs(t-offsets[:,lane]))
                choices.append((error/model.spacing, i, lane, s, t, axial))
            if not choices:
                if not pending:
                    pending.append(a)
                pending.append(b)
                continue
            if pending:
                outside.append(_RegionalRoadSeed(np.asarray(pending),road.width_m,road.source_ids,road.geometry_kind))
                pending = []
            _, i, lane, s, t, axial = min(choices,key=lambda c:c[0])
            owned.append(LineString([a,b]))
            model = corridors[i]
            tolerance = min(10, max(3, .22*model.spacing))
            protected = i in fixed_lanes and not any(tip.distance(Point((a+b)*.5))<min(100,total_length*.25) for tip in merged_tips)
            if not protected and (max(abs(t-model.offsets_at(s)[:,lane])) > tolerance or axial < np.cos(np.deg2rad(30))):
                continue
            xy = np.asarray([a,b]) if s[0] < s[1] else np.asarray([b,a])
            lo, hi = sorted(s)
            pieces[i,lane].append((lo,hi,xy,road.width_m,road.source_ids))
        if pending:
            outside.append(_RegionalRoadSeed(np.asarray(pending),road.width_m,road.source_ids,road.geometry_kind))
    # A surviving side also defines the start/end station of its missing mate.
    # Restrict extrapolation to the ordinary search distance and require a
    # substantial observed mate, so isolated short stubs cannot invent roads.
    for model_id,model in enumerate(corridors):
        for lane in (0,1):
            members = pieces[model_id,lane]
            other = pieces[model_id,1-lane]
            if not members or not other:
                continue
            lo,hi = min(p[0] for p in members),max(p[1] for p in members)
            other_lo,other_hi = min(p[0] for p in other),max(p[1] for p in other)
            for start,end in ((other_lo,lo),(hi,other_hi)):
                gap = end-start
                support = sum(max(0,min(p[1],end)-max(p[0],start)) for p in other)
                if not 20<gap<=maximum_gap or support<min(60,gap*.35):
                    continue
                station = start if start==other_lo else end
                epsilon = .01 if start==other_lo else -.01
                xy = model.point(np.asarray([station,station+epsilon]),lane)
                if epsilon<0:
                    xy = xy[::-1]
                sources = tuple(sorted({source for p in members+other for source in p[4]}))
                members.append((min(station,station+epsilon),max(station,station+epsilon),xy,float(np.mean([p[3] for p in members])),sources))
    restored, bridges = [], []
    for (model_id,lane), members in pieces.items():
        model = corridors[model_id]
        members.sort(key=lambda p:(p[0], -p[1]))
        points, sources, widths = [], set(), []
        last_s = None

        def finish():
            if len(points)>1:
                restored.append(_RegionalRoadSeed(np.asarray(points),float(np.mean(widths)),tuple(sorted(sources)),'restored_carriageway'))

        for start,end,xy,width,source_ids in members:
            if last_s is not None and end <= last_s+1e-5:
                continue
            if last_s is not None and start < last_s:
                xy = xy.copy()
                xy[0] += (xy[1]-xy[0]) * (last_s-start)/(end-start)
                start = last_s
            if points and np.linalg.norm(points[-1]-xy[0])>1e-5:
                existing = observed_continuation(points[-1],xy[0])
                other = pieces[model_id,1-lane]
                companion = sum(max(0,min(p[1],start)-max(p[0],last_s)) for p in other)
                permitted = min(900,maximum_gap*3) if companion>min(80,(start-last_s)*.3) else maximum_gap
                if existing is not None:
                    points.extend(existing[1:])
                elif start-last_s > permitted:
                    finish()
                    points, sources, widths = [], set(), []
                else:
                    # Tangents come from retained bodies. If their pixel-scale
                    # jitter prevents a monotone cubic, use the corridor axis.
                    h1 = points[-1]-points[-2] if len(points)>1 else model.axis
                    h2 = xy[0]-xy[1]
                    curve = _tangent_continuous_connector(points[-1],h1,xy[0],h2,1.0)
                    if curve is None:
                        curve = _tangent_continuous_connector(points[-1],model.axis,xy[0],-model.axis,1.0)
                    if curve is None:
                        curve = np.asarray([points[-1],xy[0]])
                    if start-last_s>80:
                        ss = np.linspace(last_s,start,max(9,int((start-last_s)/4)))
                        curve = model.point(ss,lane)
                        blend = np.linspace(0,1,len(ss))
                        curve += (1-blend)[:,None]*(points[-1]-curve[0])+blend[:,None]*(xy[0]-curve[-1])
                    points.extend(curve[1:])
                    bridges.append((LineString(curve),tuple(sorted(sources | set(source_ids)))))
            if not points:
                points.append(xy[0])
            points.append(xy[1])
            sources.update(source_ids)
            widths.append(width)
            last_s = end
        finish()
    restored_geometry = unary_union([LineString(r.points) for r in restored])
    preserved = restored_geometry.buffer(.001)
    # Keep original segment coordinates in the audit. A global union followed
    # by difference can perturb near-coincident branches at large map offsets.
    changed = [g for g in owned if g.difference(preserved).length>1e-5]
    # Existing continuation bodies may also occur in the outside pieces.
    # Remove only exact geometric duplicates here, preserving their source IDs
    # on the restored track and leaving transverse road approaches intact.
    unique_outside = []
    for road in outside:
        difference = LineString(road.points).difference(restored_geometry)
        parts = [difference] if difference.geom_type=='LineString' else list(getattr(difference,'geoms',()))
        for part in parts:
            if part.geom_type=='LineString' and part.length>.001:
                unique_outside.append(_RegionalRoadSeed(np.asarray(part.coords),road.width_m,road.source_ids,road.geometry_kind))
    unique_outside, redundant = _remove_parallel_residuals(unique_outside,restored,corridors)
    changed.extend(redundant)
    replaced = MultiLineString(changed) if changed else LineString()
    cleanup = dict(connection_redundant_fragment_count=len(redundant),connection_redundant_length_m=sum(g.length for g in redundant))
    return unique_outside+restored, replaced, bridges, cleanup


def _remove_parallel_residuals(outside, restored, corridors):
    """Drop short duplicate traces only where both replacement tracks exist.

    A transverse branch, an extended third road, or a fragment reaching outside
    the reconstructed corridor is retained. The removed geometry is audited.
    """
    from shapely.ops import unary_union
    restored_lines = [LineString(r.points) for r in restored]
    restored_index = STRtree(restored_lines)
    outside_lines = [LineString(r.points) for r in outside]
    outside_index = STRtree(outside_lines)
    kept,removed = [],[]
    for road_id,(road,line) in enumerate(zip(outside,outside_lines)):
        redundant = False
        if line.length<=150:
            positions = np.linspace(0,line.length,max(3,int(line.length/4)+1))
            samples = np.asarray([line.interpolate(p).coords[0] for p in positions])
            for model in corridors:
                s,t = model.coordinates(samples)
                if min(s)<model.start or max(s)>model.end or np.ptp(s)<line.length*.85:
                    continue
                offsets = model.offsets_at(s)
                local_spacing = float(np.median(offsets[:,1]-offsets[:,0]))
                radius = min(15,local_spacing*.45)
                if np.max(np.min(abs(t[:,None]-offsets),axis=1))>radius:
                    continue
                ids = [int(i) for i in restored_index.query(line.buffer(local_spacing+radius))]
                nearby = unary_union([restored_lines[i] for i in ids])
                if line.difference(nearby.buffer(radius)).length>line.length*.05:
                    continue
                # Require both completed trajectories at this station range.
                if any(nearby.distance(Point(model.point(float(v),lane)))>max(8,local_spacing*.3)
                       for v in (s.min(),s.max()) for lane in (0,1)):
                    continue
                attached_branch = False
                for point in (samples[0],samples[-1]):
                    for raw_j in outside_index.query(Point(point).buffer(.01)):
                        j = int(raw_j)
                        if j==road_id or outside_lines[j].distance(Point(point))>.01:
                            continue
                        other = outside[j].points
                        ss,tt = model.coordinates(other)
                        oo = model.offsets_at(ss)
                        if np.any(tt<oo[:,0]-radius) or np.any(tt>oo[:,1]+radius):
                            attached_branch = True
                if not attached_branch:
                    redundant = True
                    break
        (removed if redundant else kept).append(line if redundant else road)
    return kept,removed


def corridor_conflict(points, corridors):
    for model in corridors:
        s,t = model.coordinates(points)
        if min(s) < model.start or max(s) > model.end:
            continue
        delta = points[-1]-points[0]
        if abs(delta@model.axis)/max(np.linalg.norm(delta),1e-9) < np.cos(np.deg2rad(45)):
            continue
        offsets = model.offsets_at(s[[0,-1]])
        lanes = np.argmin(abs(t[[0,-1],None]-offsets),axis=1)
        if all(abs(t[[0,-1]]-offsets[np.arange(2),lanes]) < max(12,.4*model.spacing)) and lanes[0] != lanes[1]:
            return 'carriageway_order_conflict'
    return ''
