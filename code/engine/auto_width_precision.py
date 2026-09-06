"""Independent publication review of existing Fast Auto width candidates.

No road, surface, width product, or candidate geometry is modified here.
Measurements reuse the existing paired profile implementation.
"""
from collections import Counter
import numpy as np
import pandas as pd
from shapely import from_wkt
from shapely.geometry import Point
from .auto_track_evidence import tangent, longest_run


def _mad(values):
    return float(np.median(np.abs(values-np.median(values)))) if len(values) else 0.


def review_width_profile(samples, sign, minimum_length=32.):
    """Review a whole interval; do not cherry-pick its favourable samples."""
    data = pd.DataFrame(samples)
    if data.empty:
        return ['insufficient_sustained_width_change'], {}
    valid = data.valid.to_numpy(bool)
    good = data.loc[valid]
    reasons = []
    metrics = dict(width_valid_ratio=float(valid.mean()),width_sample_count=len(data),
                   cross_track_ratio=float(data.cross_track.mean()),
                   junction_ratio=float(data.junction.mean()),end_ratio=float(data.near_end.mean()),
                   offset_bias_ratio=float(data.offset_bias.mean()),surface_disagreement_ratio=float(data.surface_disagreement.mean()))
    if metrics['cross_track_ratio'] > .05:
        reasons.append('cross_track_width_match')
    if metrics['offset_bias_ratio'] > .10:
        reasons.append('centerline_offset_measurement_bias')
    if metrics['surface_disagreement_ratio'] > .10:
        reasons.append('surface_geometry_disagreement')
    if metrics['junction_ratio'] > .10:
        reasons.append('junction_width_instability')
    if metrics['end_ratio'] > .15:
        reasons.append('road_end_width_instability')
    if len(good) < 8 or valid.mean() < .9:
        reasons.append('unstable_width_profile')
    sustained = np.zeros(len(data),bool)
    if len(good):
        b,a = good.before_width.to_numpy(float),good.after_width.to_numpy(float)
        diff = a-b; median = float(np.median(diff)); magnitude = abs(median)
        variation = 1.4826*max(_mad(b),_mad(a),_mad(diff))
        measurement = float(np.median(good.measurement_uncertainty_m))
        pixel = float(good.pixel_uncertainty_m.max())
        required = 2.5*max(pixel,measurement,variation)+.25
        sign_ratio = float(np.mean(sign*diff>0))
        stable_offsets = float(np.ptp(np.quantile(good.signed_offset_m,[.1,.9])))
        metrics.update(width_profile_difference_m=median,width_local_variation_m=variation,
                       width_measurement_uncertainty_m=measurement,width_pixel_uncertainty_m=pixel,
                       width_required_margin_m=required,width_sign_ratio=sign_ratio,
                       width_offset_spread_m=stable_offsets)
        if sign_ratio < .9 or _mad(diff) > max(.75,.2*magnitude):
            reasons.append('unstable_width_profile')
        if stable_offsets > max(1.,2*pixel):
            reasons.append('centerline_offset_measurement_bias')
        if magnitude <= required:
            reasons.append('uncertainty_not_clearly_exceeded')
        sustained[valid] = (sign*diff >= np.maximum(2.,.2*np.maximum(b,a)))
    clean = sustained & ~data.junction.to_numpy(bool) & ~data.near_end.to_numpy(bool)
    run = longest_run(clean,data.spacing_m)
    metrics['width_sustained_length_m'] = run
    if run < minimum_length or sustained.mean() < .85:
        reasons.append('insufficient_sustained_width_change')
    return sorted(set(reasons)),metrics


def _cross_track(scene, axis_id, point, direction, width):
    for i in scene.tree.query(point,predicate='dwithin',distance=max(width/2,2.)):
        if int(i)==axis_id:
            continue
        axis = scene.lines[int(i)]; s=axis.project(point); other=axis.interpolate(s)
        delta=np.array([other.x-point.x,other.y-point.y])
        if abs(tangent(axis,s)@direction) >= .95 and abs(delta@direction)<2. and np.linalg.norm(delta)>1.:
            return True
    return False


def measure_candidate_profile(row, scenes):
    from .fast_auto_change import _normal
    from paired_width_profile import PairedWidthConfig, _measure_period_width, _surface_width
    axis=from_wkt(row.axis_wkt); before,after=scenes['before'],scenes['after']
    config=PairedWidthConfig(sample_spacing=4.)
    count=max(1,int(np.ceil(axis.length/4.))); spacing=axis.length/count
    surfaces=[scene.surface(axis) for scene in (before,after)]
    supports=[surface.buffer(.1) for surface in surfaces]
    junction=before.junction.union(after.junction).buffer(8.)
    pixel=.5*np.hypot(getattr(before.probability,'pixel_size',1.),getattr(after.probability,'pixel_size',1.))
    samples=[]
    for i in range(count):
        position=(i+.5)*spacing; point=axis.interpolate(position)
        sample=dict(candidate_id=int(row.candidate_id),station_m=position,spacing_m=spacing,geometry=point,
                    valid=False,cross_track=False,offset_bias=False,surface_disagreement=False,
                    junction=bool(junction.intersects(point)),near_end=False,
                    before_width=np.nan,after_width=np.nan,signed_offset_m=np.nan,
                    measurement_uncertainty_m=np.nan,pixel_uncertainty_m=pixel,
                    before_axis=-1,after_axis=-1,before_station_m=np.nan,after_station_m=np.nan,
                    sample_reason='no_reliable_same_track')
        match=before.match(axis,position,3.,float(row.width_bef))
        if match is None:
            sample['cross_track']=True; samples.append(sample); continue
        bi=int(match['target']); bs=match['station']; baxis=before.lines[bi]; bp=baxis.interpolate(bs)
        other=after.match(baxis,bs,3.,before.width(bp))
        if other is None:
            sample['cross_track']=True; samples.append(sample); continue
        ai=int(other['target']); ast=other['station']; aaxis=after.lines[ai]; ap=aaxis.interpolate(ast)
        reverse=before.match(aaxis,ast,3.,after.width(ap))
        reliable=(match['reliable'] and other['reliable'] and other['direction']>=.97 and
                  reverse is not None and reverse['reliable'] and reverse['target']==bi)
        sample.update(before_axis=bi,after_axis=ai,before_station_m=bs,after_station_m=ast,
                      near_end=min(bs,baxis.length-bs,ast,aaxis.length-ast)<12.,cross_track=not reliable)
        normal=_normal(baxis,bs); an=_normal(aaxis,ast)
        if normal@an<0: an=-an
        normal=normal+an; normal/=max(np.linalg.norm(normal),1e-9)
        sample['signed_offset_m']=float(np.array([ap.x-bp.x,ap.y-bp.y])@normal)
        measurements=[_measure_period_width(p,normal,surface,support,scene.probability,scene.crs,config)
                      for scene,p,surface,support in zip((before,after),(bp,ap),surfaces,supports)]
        b,a=measurements
        sample.update(before_width=b.final_width,after_width=a.final_width,
                      sample_reason=';'.join(filter(None,(b.reject_reason,a.reject_reason))))
        geometry_disagreement=any('disagreement' in m.reject_reason for m in measurements)
        offset_bias=False
        for scene,own,opposite,surface,support,measurement,axis_id,direction in zip(
                (before,after),(bp,ap),(ap,bp),surfaces,supports,measurements,(bi,ai),(tangent(baxis,bs),tangent(aaxis,ast))):
            width=measurement.surface_width
            if width is not None:
                alternate,_=_surface_width(opposite,normal,surface,support,config)
                if own.distance(opposite)>.5 and (alternate is None or abs(width-alternate)>max(1.,2*pixel)):
                    offset_bias=True
                for angle in (-5.,5.):
                    theta=np.radians(angle)
                    rotated=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])@normal
                    perturbed,_=_surface_width(own,rotated,surface,support,config)
                    if perturbed is None or abs(perturbed-width)>max(1.,.15*width):
                        geometry_disagreement=True
                sample['cross_track'] |= _cross_track(scene,axis_id,own,direction,width)
            for name,value in (('surface_width',width),('probability_width',measurement.probability_width),
                               ('measurement_confidence',measurement.width_confidence)):
                sample[('before_' if scene is before else 'after_')+name]=value
        if all(m.surface_width is not None and m.probability_width is not None for m in measurements):
            surface_diff=a.surface_width-b.surface_width
            probability_diff=a.probability_width-b.probability_width
            if abs(surface_diff-probability_diff)>max(1.5,2*pixel,.4*abs(surface_diff)):
                geometry_disagreement=True
        valid=all(m.final_width is not None and not m.reject_reason for m in measurements)
        if valid:
            radius=max(b.final_width,a.final_width)/2+4.
            valid=before.valid.covers(bp.buffer(radius)) and after.valid.covers(ap.buffer(radius))
            if not valid:
                sample['sample_reason']+=';width_cross_section_boundary'
            sample['measurement_uncertainty_m']=max(pixel,.5*(2-b.width_confidence-a.width_confidence)*max(b.final_width,a.final_width))
        if not reliable:
            sample['sample_reason']+=';cross_track_width_match'
        sample['sample_reason']=sample['sample_reason'].strip(';')
        sample.update(valid=bool(valid and reliable),offset_bias=offset_bias,surface_disagreement=geometry_disagreement)
        samples.append(sample)
    return samples


def qualify_width_candidates(candidate_audit, scenes, minimum_length=24.):
    """Width has its own accepted/review state; existence gates do not apply."""
    result=candidate_audit.copy(); all_samples=[]; profiles={}
    width_ids=result.index[result.change_typ.isin(['widened','narrowed'])]
    for index in width_ids:
        row=result.loc[index]; samples=measure_candidate_profile(row,scenes)
        all_samples.extend(samples); profiles[index]=samples
        reasons,metrics=review_width_profile(samples,1 if row.change_typ=='widened' else -1,max(32.,minimum_length))
        for key,value in metrics.items(): result.loc[index,key]=value
        result.loc[index,'precision_reason']=';'.join(reasons) if reasons else 'stable_paired_width_profile'
        result.loc[index,'publication_state']='review' if reasons else 'accepted'
    # Adjacent opposing signs on the same observed track indicate an unstable
    # profile. Do not compare separate parallel tracks merely because nearby.
    intervals={}
    for index,samples in profiles.items():
        good=[s for s in samples if s['before_axis']>=0]
        if not good: continue
        axis_id=Counter(s['before_axis'] for s in good).most_common(1)[0][0]
        positions=[s['before_station_m'] for s in good if s['before_axis']==axis_id]
        intervals[index]=(axis_id,min(positions),max(positions))
    for i,(axis_i,start_i,end_i) in intervals.items():
        for j,(axis_j,start_j,end_j) in intervals.items():
            if i>=j or axis_i!=axis_j or result.loc[i,'change_typ']==result.loc[j,'change_typ']:
                continue
            gap=max(0.,max(start_i,start_j)-min(end_i,end_j))
            if gap<=24. and min(end_i-start_i,end_j-start_j)<=80.:
                for index in (i,j):
                    result.loc[index,'publication_state']='review'
                    result.loc[index,'precision_reason']+=';alternating_width_change_signs'
    for index in width_ids:
        accepted=result.loc[index,'publication_state']=='accepted'
        result.loc[index,'width_qa_state']='accepted' if accepted else 'review'
        result.loc[index,'qa_state']='confirmed' if accepted else 'uncertain'
        result.loc[index,'confidence']=.85 if accepted else .3
        result.loc[index,'audit_reason']+=';'+result.loc[index,'precision_reason']
    return result,all_samples
