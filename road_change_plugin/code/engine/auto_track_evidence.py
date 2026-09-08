"""Read-only track correspondence checks used by Auto publication review."""
import numpy as np
from shapely.geometry import Point, LineString


def tangent(line, station):
    a = np.asarray(line.interpolate(max(0., station-4.)).coords[0])
    b = np.asarray(line.interpolate(min(line.length, station+4.)).coords[0])
    delta = b-a
    return delta/max(np.linalg.norm(delta), 1e-9)


def displacement_rescue(axis, source, target, radius=12.):
    """Look beyond the strict match tolerance; never assign or move a track."""
    positions = np.linspace(0., axis.length, max(5, int(np.ceil(axis.length/4.))+1))
    points = [axis.interpolate(s) for s in positions]
    directions = np.array([tangent(axis,s) for s in positions])
    choices = []
    for raw_id in target.tree.query(axis, predicate='dwithin', distance=radius):
        target_id = int(raw_id); line = target.lines[target_id]
        stations = np.array([line.project(p) for p in points])
        matched = [line.interpolate(s) for s in stations]
        delta = np.array([[q.x-p.x,q.y-p.y] for p,q in zip(points,matched)])
        distances = np.linalg.norm(delta,axis=1)
        alignment = np.array([abs(tangent(line,s)@d) for s,d in zip(stations,directions)])
        inside = (distances <= radius)&(alignment >= .95)
        coverage = float(inside.mean())
        if coverage < .8:
            continue
        signed = delta[:,0]*(-directions[:,1])+delta[:,1]*directions[:,0]
        spread = float(np.ptp(np.quantile(signed[inside],[.1,.9])))
        steps = np.diff(stations[inside]); span = float(np.ptp(stations[inside]))
        overlap = span/max(float(np.ptp(positions[inside])),1e-9)
        monotone = max(float(np.mean(steps >= -.1)),float(np.mean(steps <= .1)))
        if spread > 2. or not .8 <= overlap <= 1.2 or monotone < .95:
            continue
        reverse = []
        for p,q,distance,direction in zip(points,matched,distances,directions):
            if distance > radius:
                continue
            alternatives = []
            for i in source.tree.query(q,predicate='dwithin',distance=radius):
                other = source.lines[int(i)]; s = other.project(q)
                if abs(tangent(other,s)@direction) >= .95:
                    alternatives.append(other.distance(q))
            # A distinctly closer source track defeats same-road identity.
            reverse.append(not alternatives or min(alternatives) >= distance-.75)
        topology = float(np.mean(reverse)) if reverse else 0.
        median = float(np.median(distances[inside]))
        choices.append((median+spread,target_id,coverage,median,spread,topology,overlap))
    if not choices:
        return {'displacement_rescue':False,'displacement_track_ambiguous':False}
    choices.sort(); _,target_id,coverage,offset,spread,topology,overlap = choices[0]
    ambiguous = topology < .8 or (len(choices)>1 and choices[1][0]-choices[0][0]<.75)
    return dict(displacement_rescue=not ambiguous,displacement_track_ambiguous=ambiguous,
                displacement_target_axis=target_id,displacement_coverage=coverage,
                displacement_offset_m=offset,displacement_offset_spread_m=spread,
                displacement_reverse_ratio=topology,displacement_longitudinal_overlap=overlap)


def longest_run(mask, weights):
    best = current = 0.
    for valid, weight in zip(mask,weights):
        current = current+float(weight) if valid else 0.
        best = max(best,current)
    return best


def nearby_opposite_support(axis, target, width):
    """Check a parallel corridor even when the other period lost its axis.

    This is a reason to review, never a synthesized match or changed geometry.
    Require longitudinal support so a crossing strip is not mistaken for a road.
    """
    positions=np.linspace(0.,axis.length,max(5,int(np.ceil(axis.length/4.))+1))
    points=np.array([axis.interpolate(s).coords[0] for s in positions])
    direction=np.array([tangent(axis,s) for s in positions])
    normal=np.column_stack((-direction[:,1],direction[:,0]))
    surface=target.surface(axis)
    probability=getattr(target,'probability',None)
    for offset in (-3.,3.,-6.,6.,-9.,9.,-12.,12.):
        shifted=LineString(points+offset*normal)
        coverage=float(np.mean([surface.covers(Point(p)) for p in points+offset*normal]))
        if coverage>=.8:
            return dict(nearby_opposite_reason='nearby_opposite_surface_support',nearby_opposite_offset_m=offset,
                        nearby_opposite_surface_ratio=coverage)
        if probability is not None:
            evidence=probability.sample_axis(shifted,target.crs,road_width=width,position_tolerance=3.)
            rank=evidence['scene_percentile_rank']; background=evidence['background_percentile_rank']
            lower_rank=probability.percentile_rank(evidence['center_probability_q25'])
            if (evidence['probability_valid_ratio']>=.99 and lower_rank is not None and lower_rank>=.85
                    and rank is not None and background is not None and rank-background>=.1
                    and (evidence['local_probability_contrast'] or 0)>0):
                return dict(nearby_opposite_reason='nearby_opposite_probability_support',nearby_opposite_offset_m=offset,
                            nearby_opposite_surface_ratio=coverage,nearby_opposite_probability_q25_rank=lower_rank)
    return dict(nearby_opposite_reason='')
