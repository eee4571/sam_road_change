"""Metric axis QA before offsets; repair oscillations, never repair polygons."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from shapely import covers, distance, points, union_all
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import substring
from shapely.strtree import STRtree


from .geometry_recovery import GEOMETRY_ERRORS, current_recovery, attempted_methods


class AxisQualityError(ValueError):
    pass


def _contacts(a,b):
    # GEOS intersections can miss a shared endpoint after CRS/substring rounding.
    # Compare actual contacts, not the length of buffers along shallow angles.
    contacts=[a.intersection(b)]
    for line,other in ((a,b),(b,a)):
        parts=[line] if line.geom_type=='LineString' else list(line.geoms)
        for part in parts:
            for xy in (part.coords[0],part.coords[-1]):
                p=Point(xy)
                if p.distance(other)<=1e-6:contacts.append(p)
    return union_all(contacts)


def axis_quality(axis, width):
    xy=np.asarray(axis.coords)[:,:2]
    keep=np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-7];xy=xy[keep]
    segments=np.diff(xy,axis=0);lengths=np.linalg.norm(segments,axis=1)
    if len(lengths)<2:return dict(abnormal=False,reasons=[],stations=[])
    directions=segments/lengths[:,None]
    turns=np.arctan2(directions[:-1,0]*directions[1:,1]-directions[:-1,1]*directions[1:,0],
                     np.sum(directions[:-1]*directions[1:],axis=1))
    stations=np.cumsum(lengths)[:-1]
    radius=np.minimum(lengths[:-1],lengths[1:])/(2*np.maximum(np.sin(np.abs(turns)/2),1e-8))
    sharp=np.flatnonzero((np.abs(turns)>np.deg2rad(45))&(radius<max(width,2.)))
    flagged=[]
    for i,j in zip(sharp,sharp[1:]):
        if stations[j]-stations[i]<=max(8.,min(2*width,24.)) and turns[i]*turns[j]<0:
            flagged.extend([i,j])
    clusters=[]
    for i in sorted(set(flagged)):
        if clusters and stations[i]-stations[clusters[-1][-1]]<=max(8.,min(2*width,24.)):
            clusters[-1].append(i)
        else:clusters.append([i])
    repeated=[i for cluster in clusters if len(cluster)>=4 for i in cluster]
    # Isolated real bends and sustained hairpins are not oscillations. Repeated
    # alternating turns, or a self-crossing with such turns, are geometry faults.
    crossings=[];crossing_intervals=[]
    if not axis.is_simple:
        edges=[LineString(pair) for pair in zip(xy[:-1],xy[1:])]
        pairs=STRtree(edges).query(edges,predicate='intersects')
        cumulative=np.r_[0.,np.cumsum(lengths)]
        for i,j in pairs.T:
            if j<=i+1 or (i==0 and j==len(edges)-1 and np.array_equal(xy[0],xy[-1])):continue
            hit=edges[i].intersection(edges[j])
            if not hit.is_empty:
                p=hit.representative_point()
                pair=[cumulative[i]+edges[i].project(p),cumulative[j]+edges[j].project(p)]
                crossings.extend(pair);crossing_intervals.append(pair)
    abnormal=bool(repeated) or bool(crossings)
    reasons=[]
    if abnormal:
        reasons=['short_alternating_turns','offset_fold_risk']
        if any(abs(turns[i])>np.deg2rad(120) for i in flagged):reasons.append('local_reversal')
        if any(radius[i]<width/2 for i in flagged):reasons.append('radius_below_half_width')
        if crossings:reasons.append('self_intersection')
        if flagged:
            lo,hi=min(flagged),max(flagged)
            path=stations[hi]-stations[lo]
            chord=np.linalg.norm(xy[hi+1]-xy[lo+1])
            if path>1.15*max(chord,1.):reasons.append('excess_local_path_length')
    fault_stations=sorted(set([float(stations[i]) for i in flagged]+list(map(float,crossings))))
    return dict(abnormal=bool(abnormal),reasons=reasons,
                oscillation_stations=fault_stations,
                crossing_intervals=crossing_intervals,
                stations=sorted(set([float(stations[i]) for i in repeated]+list(map(float,crossings)))) if abnormal else [],
                sharp_turn_count=int(len(sharp)),minimum_turn_radius_m=float(radius.min(initial=np.inf)))


def _pinned_smooth(xy, sigma):
    fitted=gaussian_filter1d(xy,sigma,axis=0,mode='nearest')
    fraction=np.linspace(0.,1.,len(xy))[:,None]
    fitted+=(1-fraction)*(xy[0]-fitted[0])+fraction*(xy[-1]-fitted[-1])
    fitted[[0,-1]]=xy[[0,-1]]
    return fitted


def _evidence_axis(reference, width, evidence):
    """Bounded cross-sections reuse surface boundary/probability, no new inference."""
    xy=np.asarray(reference.coords)
    tangent=np.gradient(xy,axis=0)
    normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
    offsets=np.linspace(-width,width,25)
    probes=xy[:,None,:]+normal[:,None,:]*offsets[None,:,None]
    flat=probes.reshape(-1,2);score=np.zeros(len(flat));supported=np.zeros(len(flat),bool)
    if evidence is not None and evidence.surface is not None:
        surface=evidence.surface.context
        inside=covers(surface,points(flat));supported|=inside
        score+=inside*(1+np.minimum(distance(points(flat),surface.boundary)/max(width/2,1.),1.))
    for raster in (() if evidence is None else (evidence.probability,evidence.molra)):
        if raster is not None:
            values=np.nan_to_num(raster.values(flat),nan=0.)
            supported|=values>=.2;score+=values
    score=score.reshape(len(xy),-1)-.2*(offsets[None,:]/max(width,1.))**2
    score[~supported.reshape(score.shape)]=-np.inf
    valid=np.isfinite(score).any(axis=1)
    fitted=xy.copy();fitted[valid]=probes[np.flatnonzero(valid),np.argmax(score[valid],axis=1)]
    fitted[[0,-1]]=xy[[0,-1]]
    return _pinned_smooth(fitted,max(1.,len(xy)*width/max(reference.length,1.)))


def _evidence_route(axis,width,evidence):
    """Local medial route on already available evidence, bounded to 200k cells."""
    if evidence is None:return None
    from skimage.graph import route_through_array
    bounds=axis.buffer(width).bounds
    x0,y0,x1,y1=bounds
    step=max(1.,width/12,np.sqrt((x1-x0)*(y1-y0)/200000.))
    xx=np.arange(x0,x1+step,step);yy=np.arange(y0,y1+step,step)
    x,y=np.meshgrid(xx,yy);xy=np.c_[x.ravel(),y.ravel()]
    allowed=covers(axis.buffer(width),points(xy));score=np.zeros(len(xy))
    if evidence.surface is not None:score=covers(evidence.surface.context,points(xy)).astype(float)
    for raster in (evidence.probability,evidence.molra):
        if raster is not None:score=np.maximum(score,np.clip(np.nan_to_num(raster.values(xy),nan=0.)/.5,0,1))
    if not np.any(score>.4):return None
    score=score.reshape(x.shape)
    clearance=distance_transform_edt(score>.4)*step
    costs=1+12*(1-score)+3/(1+clearance)
    costs[~allowed.reshape(x.shape)]=np.inf
    ends=np.asarray([axis.coords[0],axis.coords[-1]])
    rc=np.rint((ends-np.array([x0,y0]))/step).astype(int)[:,::-1]
    try:route,_=route_through_array(costs,tuple(rc[0]),tuple(rc[1]),fully_connected=True)
    except ValueError:return None  # no finite local grid path; try conservative fit
    if len(route)<2:return None
    route=np.asarray(route);xy=np.c_[xx[route[:,1]],yy[route[:,0]]]
    xy[[0,-1]]=ends
    line=LineString(xy)
    return np.asarray([line.interpolate(s).coords[0] for s in np.linspace(0,line.length,max(7,int(line.length)+1))])


def _join_tangents(candidate,original,width,pin_start,pin_end):
    """C1 joins to unchanged approach segments, wholly inside the faulty window."""
    length=candidate.length
    if length<1e-6:return candidate
    ss=np.linspace(0,length,max(7,int(length)+1))
    xy=np.asarray([candidate.interpolate(s).coords[0] for s in ss])
    old=np.asarray(original.coords)
    reach=min(width*1.5,length/3)
    for end,pin in ((0,pin_start),(1,pin_end)):
        if not pin:continue
        station=length-reach if end else reach
        join=np.asarray(candidate.interpolate(station).coords[0])
        delta=np.asarray(candidate.interpolate(min(length,station+.5)).coords[0])-np.asarray(candidate.interpolate(max(0,station-.5)).coords[0])
        delta/=max(np.linalg.norm(delta),1e-9)
        tangent=old[-1]-old[-2] if end else old[1]-old[0]
        tangent/=max(np.linalg.norm(tangent),1e-9)
        if end:
            mask=ss>=station
            curve=CubicHermiteSpline([station,length],[join,old[-1]],[delta,tangent])
        else:
            mask=ss<=station
            curve=CubicHermiteSpline([0,station],[old[0],join],[tangent,delta])
        xy[mask]=curve(ss[mask])
    xy[[0,-1]]=old[[0,-1]]
    return LineString(xy)


def _repair_interval(axis,width,evidence,validator,pin_start=False,pin_end=False):
    ss=np.linspace(0.,axis.length,max(7,int(np.ceil(axis.length))+1))
    xy=np.asarray([axis.interpolate(s).coords[0] for s in ss])
    before=evidence.measure(axis) if evidence is not None else dict(joint_support=0.,unsupported_run_m=axis.length)
    attempts=[];fallback=None
    def evaluate(fit,method,conservative=False):
        nonlocal fallback
        candidate=_join_tangents(LineString(fit),axis,width,pin_start,pin_end)
        qa=axis_quality(candidate,width)
        safe=(candidate.length>1e-6 and candidate.is_simple and not qa['abnormal'] and
              axis.hausdorff_distance(candidate)<=max(3.,2*width) and validator(candidate))
        # Radius is a warning, not an automatic rejection of a real tight bend.
        # Verify the fitted offset before accepting it; never fill its defects.
        if safe and qa.get('minimum_turn_radius_m',np.inf)<width/2:
            from .auto_change_geometry import corridor
            from shapely.geometry import Polygon
            for values in ([width*.9,width*1.1],[width*1.1,width*.9]):
                try:probe=corridor(candidate,[0.,candidate.length],values)
                except AxisQualityError:
                    safe=False;break
                polygons=[probe] if probe.geom_type=='Polygon' else probe.geoms
                if any(Polygon(ring).area>=1. for polygon in polygons for ring in polygon.interiors):
                    safe=False;break
        if not safe:
            attempts.append(dict(method=method,rejected='geometry_or_topology',simple=candidate.is_simple,
                abnormal=qa['abnormal'],radius=qa.get('minimum_turn_radius_m'),
                shift=axis.hausdorff_distance(candidate),topology=validator(candidate)));return None
        after=evidence.measure(candidate) if evidence is not None else before
        supported=(after['joint_support']>=max(.85,before['joint_support']-.02) and
                   after['unsupported_run_m']<=max(3.,before['unsupported_run_m']))
        report=dict(method=method,quality_state='evidence_supported' if supported else 'direction_inferred',
                    support_before=before['joint_support'],support_after=after['joint_support'],
                    unsupported_run_m=after['unsupported_run_m'],attempts=list(attempts))
        if supported:return candidate,report
        attempts.append(dict(method=method,rejected='insufficient_evidence'))
        if conservative:fallback=(candidate,report)
        return None
    def accept(fit,method,conservative=False):
        try:return evaluate(fit,method,conservative)
        except GEOMETRY_ERRORS as error:
            attempts.append(dict(method=method,rejected=type(error).__name__,reason=str(error)))
            return None
    step=ss[1]-ss[0]
    reference=None
    for scale in (1.,2.):
        fit=_pinned_smooth(xy,max(1.,width*scale/step))
        found=accept(fit,'local_low_frequency')
        if found:return found
        reference=LineString(fit)
    # Sample at a bounded density before fitting the evidence ridge.
    samples=np.linspace(0.,reference.length,max(7,int(reference.length/2)+1))
    reference=LineString([reference.interpolate(s).coords[0] for s in samples])
    try:
        ridge=_evidence_axis(reference,width,evidence)
        found=accept(ridge,'surface_probability_refit')
        if found:return found
    except GEOMETRY_ERRORS as error:
        attempts.append(dict(method='surface_probability_refit',rejected=type(error).__name__,reason=str(error)))
    try:route=_evidence_route(axis,width,evidence)
    except GEOMETRY_ERRORS as error:
        attempts.append(dict(method='local_medial_refit',rejected=type(error).__name__,reason=str(error)))
        route=None
    if route is not None:
        for scale in (.25,.5,1.):
            found=accept(_pinned_smooth(route,max(1.,scale*width)),'local_medial_refit')
            if found:return found
    # Averaging can cut across a neighbouring branch at a shared junction.
    # Retain the observed approach bend while removing short alternating
    # vertices. These candidates still pass the same contact, offset, evidence
    # and displacement checks; never relax topology to accept a smooth fit.
    for tolerance in (width*.05,width*.1,width*.2,width*.3):
        try:
            fit=np.asarray(axis.simplify(tolerance,preserve_topology=True).coords)
            found=accept(fit,'local_vertex_refit',True)
            if found:return found
        except GEOMETRY_ERRORS as error:
            attempts.append(dict(method='local_vertex_refit',rejected=type(error).__name__,reason=str(error)))
    # Endpoint directions come from stable approach portions, outside the detected
    # reversals. A clamped cubic retains curved approaches; chord is last resort.
    chord=xy[-1]-xy[0];length=np.linalg.norm(chord)
    for method,fit in [('stable_direction_connection',CubicSpline([0.,1.],xy[[0,-1]],
            bc_type=((1,(xy[min(len(xy)-1,max(2,int(width/step)))]-xy[0])/max(width,1.)*length),
                     (1,(xy[-1]-xy[max(0,len(xy)-1-max(2,int(width/step)))])/max(width,1.)*length)))(np.linspace(0,1,max(7,int(length)+1)))),
                       ('conservative_chord',np.linspace(xy[0],xy[-1],max(2,int(length)+1)))]:
        found=accept(fit,method,True)
        if found:return found
    if fallback:
        fallback[1]['attempts']=attempts
        return fallback
    error=AxisQualityError('异常局部经平滑、证据重拟合及保守连接仍不能保持拓扑')
    error.diagnostic=dict(axis_wkt=axis.wkt,width=width,attempts=attempts)
    raise error


def _repair_axis_pass(axis, width, evidence=None, validator=lambda candidate: True, force=False):
    """Constrained low-frequency fit for classified faults; endpoints are fixed.

    Unlike the small-wobble pass, the fault displacement bound is proportional
    to the road width. Every accepted fit must remove the fault, stay simple,
    and retain measured road evidence. Otherwise stop before surface export.
    """
    qa=axis_quality(axis,width)
    if not qa['abnormal']:
        if not force or not qa.get('oscillation_stations'):return axis,dict(qa,corrected=False)
        qa['stations']=qa['oscillation_stations']
    intervals=[]
    # The width must not turn a tiny kink on a wide road into a 100 m window.
    margin=max(3.,width)
    windows=[[s-margin,s+margin] for s in qa['stations']]
    windows += [[a-margin,b+margin] for a,b in qa.get('crossing_intervals',[])]
    for low,high in sorted(windows):
        low=max(0.,low);high=min(axis.length,high)
        if low<width:low=0.
        if axis.length-high<width:high=axis.length
        if intervals and low<=intervals[-1][1]+max(8.,min(width,16.)):intervals[-1][1]=max(intervals[-1][1],high)
        else:intervals.append([low,high])
    parts=[];reports=[];cursor=0.
    for low,high in intervals:
        if low>cursor:parts.extend(list(substring(axis,cursor,low).coords)[int(bool(parts)):])
        local=substring(axis,low,high)
        def local_topology(candidate):
            complete=parts+list(candidate.coords)[int(bool(parts)):]
            if high<axis.length:complete+=list(substring(axis,high,axis.length).coords)[1:]
            return validator(LineString(complete))
        fixed,report=_repair_interval(local,width,evidence,local_topology,low>0,high<axis.length)
        parts.extend(list(fixed.coords)[int(bool(parts)):]);cursor=high
        reports.append(dict(report,start_m=low,end_m=high,length_after_m=fixed.length))
    if cursor<axis.length:parts.extend(list(substring(axis,cursor,axis.length).coords)[1:])
    candidate=LineString(parts)
    return candidate,dict(qa,corrected=True,maximum_shift_m=float(axis.hausdorff_distance(candidate)),
        length_before_m=float(axis.length),length_after_m=float(candidate.length),repairs=reports)


def repair_axis(axis, width, evidence=None, validator=lambda candidate: True, force=False):
    """Repair local faults and recheck the *assembled* axis after each pass.

    Shortening a tooth can expose an alternating turn at the join with the
    unchanged road. A locally safe fit therefore is not a completed axis.
    Compose station maps back to the original axis so widths and feature splits
    keep their correspondence through successive, still-local repairs.
    """
    current=axis;source=np.array([0.,axis.length]);target=source.copy()
    intervals=[];history=[];initial=axis_quality(axis,width)
    for attempt in range(max(2,min(32,len(axis.coords)))):
        candidate,report=_repair_axis_pass(current,width,evidence,validator,force=force if attempt==0 else False)
        if not report.get('corrected'):
            if not history:return candidate,report
            break
        before=[0.];after=[0.];offset=0.;cursor=0.
        for repair in report['repairs']:
            low,high=repair['start_m'],repair['end_m']
            intervals.append(tuple(np.interp([low,high],target,source)))
            offset+=low-cursor;before.append(low);after.append(offset)
            offset+=repair['length_after_m'];before.append(high);after.append(offset);cursor=high
        before.append(current.length);after.append(offset+current.length-cursor)
        # Duplicate endpoint knots are identical, but remove them for a strict
        # monotone interpolation domain (including repairs touching endpoints).
        before,index=np.unique(before,return_index=True);after=np.asarray(after)[index]
        knots=np.unique(np.r_[source,np.interp(before,target,source)])
        target=np.interp(np.interp(knots,source,target),before,after);source=knots
        history.append(report);current=candidate
        if not axis_quality(current,width)['abnormal']:break
    if axis_quality(current,width)['abnormal']:
        error=AxisQualityError('局部轴线拼接后仍有异常折返')
        error.diagnostic=dict(axis_wkt=axis.wkt,corrected_wkt=current.wkt,width=width,passes=history)
        raise error
    if len(history)==1:return current,history[0]
    merged=[]
    for low,high in sorted(intervals):
        if merged and low<=merged[-1][1]+1e-7:merged[-1][1]=max(merged[-1][1],high)
        else:merged.append([low,high])
    methods=sorted({r['method'] for p in history for r in p['repairs']})
    quality=('direction_inferred' if any(r['quality_state']=='direction_inferred'
              for p in history for r in p['repairs']) else 'evidence_supported')
    repairs=[dict(start_m=a,end_m=b,length_after_m=float(np.interp(b,source,target)-np.interp(a,source,target)),
                  method='+'.join(methods),quality_state=quality) for a,b in merged]
    return current,dict(initial,corrected=True,maximum_shift_m=float(axis.hausdorff_distance(current)),
        length_before_m=axis.length,length_after_m=current.length,repairs=repairs,
        repair_passes=history,station_map=dict(before=source.tolist(),after=target.tolist()))


def repair_network_axes(axes, widths, evidence):
    """Join exact degree-2 features for QA; preserve all true junction contacts."""
    if not axes:return [],[]
    evidence_factory=evidence if callable(evidence) else None
    nodes={}
    for i,axis in enumerate(axes):
        for end in (0,1):nodes.setdefault(tuple(np.round(axis.coords[0 if end==0 else -1],6)),[]).append((i,end))
    tree=STRtree(axes);visited=set();result=list(axes);audit=[]
    starts=[(i,e) for ends in nodes.values() if len(ends)!=2 for i,e in ends]
    starts += [(i,0) for i in range(len(axes))]
    for first,start_end in starts:
        if first in visited:continue
        members=[];coords=[];cursor=0.;i,e=first,start_end
        while i not in visited:
            visited.add(i);xy=list(axes[i].coords)
            if e:xy.reverse()
            members.append((i,e,cursor,cursor+axes[i].length));cursor+=axes[i].length
            coords.extend(xy if not coords else xy[1:])
            ends=nodes[tuple(np.round(xy[-1],6))]
            if len(ends)!=2:break
            following=[p for p in ends if p[0]!=i]
            if not following:break
            i,e=following[0]
        checkpoint=len(audit)
        previous={k:result[k] for k,_,_,_ in members}
        try:
            chain=LineString(coords);width=float(np.median([widths[i] for i,_,_,_ in members]))
            qa=axis_quality(chain,width)
            if not qa['abnormal']:continue
            if evidence_factory is not None:
                evidence=evidence_factory();evidence_factory=None
            member_ids={r[0] for r in members};fixed=[0.,chain.length]
            neighbours=set(map(int,tree.query(chain.buffer(max(2.,width*1.5))))) - member_ids
            neighbours={j for j in neighbours if result[j] is not None}
            for j in neighbours:
                contact=chain.intersection(result[j])
                if contact.geom_type=='Point':fixed.append(chain.project(contact))
                elif contact.geom_type=='MultiPoint':fixed.extend(chain.project(p) for p in contact.geoms)
                elif not contact.is_empty:raise AxisQualityError('异常轴线存在重叠道路，无法保持拓扑自动修复')
            fixed=sorted(set(fixed));parts=[];old_stations=[0.];new_stations=[0.];reports=[];changed_ranges=[]
            for a,b in zip(fixed,fixed[1:]):
                original_part=substring(chain,a,b)
                topology_problems=[]
                def topology_safe(candidate):
                    for j in neighbours:
                        old=_contacts(original_part,result[j]);new=_contacts(candidate,result[j])
                        if not new.difference(old.buffer(1e-4)).is_empty or not old.difference(new.buffer(1e-4)).is_empty:
                            topology_problems.append(dict(neighbour=j,old=old.wkt,new=new.wkt,
                                other=result[j].wkt,part=original_part.wkt,candidate=candidate.wkt))
                            return False
                    return True
                try:part,report=repair_axis(original_part,width,evidence,topology_safe,force=True)
                except AxisQualityError as error:
                    error.diagnostic=getattr(error,'diagnostic',{})
                    error.diagnostic.update(topology=topology_problems[-1:] if topology_problems else [])
                    raise
                parts.extend(list(part.coords) if not parts else list(part.coords)[1:])
                cursor=0.;offset=new_stations[-1]
                if report.get('station_map'):
                    mapping=report['station_map']
                    old_stations.extend(a+s for s in mapping['before'][1:])
                    new_stations.extend(offset+s for s in mapping['after'][1:])
                    changed_ranges.extend((a+r['start_m'],a+r['end_m']) for r in report['repairs'])
                    reports.append(report)
                    continue
                for repair in report.get('repairs',[]):
                    low,high=repair['start_m'],repair['end_m']
                    changed_ranges.append((a+low,a+high))
                    offset+=low-cursor
                    old_stations.append(a+low);new_stations.append(offset)
                    offset+=repair['length_after_m']
                    old_stations.append(a+high);new_stations.append(offset);cursor=high
                old_stations.append(b);new_stations.append(offset+b-a-cursor);reports.append(report)
            corrected=LineString(parts)
            if axis_quality(corrected,width)['abnormal']:
                error=AxisQualityError('真实路口约束下仍有异常折返')
                error.diagnostic=dict(axis_wkt=chain.wkt,corrected_wkt=corrected.wkt,width=width,
                                      fixed=fixed,parts=reports,qa=axis_quality(corrected,width),members=list(member_ids))
                raise error
            for j in neighbours:
                old=_contacts(chain,result[j]);new=_contacts(corrected,result[j])
                if not new.difference(old.buffer(1e-5)).is_empty or not old.difference(new.buffer(1e-5)).is_empty:
                    raise AxisQualityError('局部轴线修复会改变真实路口连接')
            for i,e,a,b in members:
                if not any(low<b and high>a for low,high in changed_ranges):continue
                low,high=np.interp([a,b],old_stations,new_stations)
                part=substring(corrected,float(low),float(high))
                xy=list(part.coords)
                original_xy=list(axes[i].coords)[::-1] if e else list(axes[i].coords)
                if not any(lo<a<hi for lo,hi in changed_ranges):xy[0]=original_xy[0]
                if not any(lo<b<hi for lo,hi in changed_ranges):xy[-1]=original_xy[-1]
                part=LineString(xy)
                if e:part=LineString(list(part.coords)[::-1])
                result[i]=part
                old_map=np.unique(np.r_[a,[s for s in old_stations if a<s<b],b])
                new_map=np.interp(old_map,old_stations,new_stations)-low
                old_map=old_map-a
                if e:old_map=b-a-old_map[::-1];new_map=part.length-new_map[::-1]
                new_map=np.clip(new_map,0.,part.length)
                new_map[[0,-1]]=[0.,part.length]
                audit.append(dict(feature=i,corrected=True,reasons=qa['reasons'],
                    maximum_shift_m=float(axes[i].hausdorff_distance(part)),
                    length_before_m=float(axes[i].length),length_after_m=float(part.length),parts=reports,
                    station_map=dict(before=old_map.tolist(),after=new_map.tolist())))
        except GEOMETRY_ERRORS as error:
            del audit[checkpoint:]
            for k,value in previous.items():result[k]=value
            try:
                chain=LineString(coords)
                width=float(np.median([widths[k] for k,_,_,_ in members]))
                member_ids={k for k,_,_,_ in members}
                neighbours=set(map(int,tree.query(chain.buffer(max(2.,width*1.5))))) - member_ids
                pieces,issues=conservative_axis_parts(chain,width,[result[j] for j in neighbours if result[j] is not None],error)
            except GEOMETRY_ERRORS as local:
                pieces=[];issues=[dict(repair_status='skipped',attempted_methods=attempted_methods(error),
                                     exception_type=type(local).__name__,reason=str(local))]
            for k,reverse,a,b in members:
                selected=[];maps=[]
                for low,high,part in pieces:
                    left,right=max(a,low),min(b,high)
                    if right-left<=1e-7:continue
                    begin,end=np.interp([left,right],[low,high],[0.,part.length])
                    section=substring(part,float(begin),float(end))
                    if section.geom_type!='LineString' or section.length<=1e-7:continue
                    if reverse:section=LineString(list(section.coords)[::-1])
                    selected.append(section)
                    old=[left-a,right-a] if not reverse else [b-right,b-left]
                    maps.append(dict(before=old,after=[0.,section.length]))
                if reverse:selected.reverse();maps.reverse()
                result[k]=(selected[0] if len(selected)==1 else MultiLineString(selected)) if selected else None
                status='fallback' if selected else 'skipped'
                details=[dict(r,source_ids=[k]) for r in issues]
                audit.append(dict(feature=k,corrected=bool(selected),axis_quality='low',repair_status=status,
                    piece_maps=maps,parts=[],failures=details))
                recovery=current_recovery()
                if recovery is not None:
                    for r in details:recovery.record('axis_repair',[k],error,r['repair_status'],r['attempted_methods'])
                else:
                    print(f'[WARN] 道路 {k} 轴线修复降级: {status} ({error})',flush=True)
    return result,audit


def conservative_axis_parts(axis,width,neighbours,error):
    """Isolate faulty windows; keep safe approaches and exact topology anchors.

    Returned source station intervals keep measured widths aligned even when
    an irreparable interval is omitted. No rejected source axis is republished.
    """
    methods=attempted_methods(error)
    methods.append('topology_anchored_local_chord')
    try:
        qa=axis_quality(axis,width)
        stations=qa.get('stations') or qa.get('oscillation_stations') or []
        windows=[[max(0.,s-max(3.,width)),min(axis.length,s+max(3.,width))] for s in stations]
        windows += [[max(0.,a-width),min(axis.length,b+width)] for a,b in qa.get('crossing_intervals',[])]
        # A failure with no classifiable interval is scoped to this chain only.
        if not windows:windows=[[0.,axis.length]]
        merged=[]
        for low,high in sorted(windows):
            if merged and low<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],high)
            else:merged.append([low,high])
        pieces=[];issues=[];cursor=0.
        def append_safe(low,high,repair=False):
            if high-low<=1e-7:return
            old=substring(axis,low,high)
            candidate=old
            if repair:
                anchors={0.,old.length}
                for other in neighbours:
                    hit=_contacts(old,other)
                    pts=[hit] if hit.geom_type=='Point' else list(hit.geoms) if hit.geom_type=='MultiPoint' else []
                    if not hit.is_empty and not pts:raise AxisQualityError('Overlapping local topology')
                    anchors.update(old.project(p) for p in pts)
                candidate=LineString([old.interpolate(s).coords[0] for s in sorted(anchors)])
            require_safe_axis(candidate,width)
            if not candidate.is_simple or not candidate.is_valid:raise AxisQualityError('Unsafe conservative axis')
            if repair:
                if old.hausdorff_distance(candidate)>max(3.,2*width):raise AxisQualityError('Conservative displacement exceeded')
                for other in neighbours:
                    before=_contacts(old,other);after=_contacts(candidate,other)
                    if not after.difference(before.buffer(1e-5)).is_empty or not before.difference(after.buffer(1e-5)).is_empty:
                        raise AxisQualityError('Conservative connection changes contacts')
            pieces.append((low,high,candidate))
        for low,high in merged:
            for a,b,repair in ((cursor,low,False),(low,high,True)):
                if b-a<=1e-7:continue
                try:
                    append_safe(a,b,repair)
                    if repair:issues.append(dict(start_m=a,end_m=b,repair_status='fallback',attempted_methods=methods))
                except GEOMETRY_ERRORS as local:
                    issues.append(dict(start_m=a,end_m=b,repair_status='skipped',attempted_methods=methods,
                                       exception_type=type(local).__name__,reason=str(local)))
            cursor=high
        if cursor<axis.length:
            try:append_safe(cursor,axis.length)
            except GEOMETRY_ERRORS as local:
                issues.append(dict(start_m=cursor,end_m=axis.length,repair_status='skipped',attempted_methods=methods,
                                   exception_type=type(local).__name__,reason=str(local)))
        return pieces,issues
    except GEOMETRY_ERRORS as local:
        return [],[dict(repair_status='skipped',attempted_methods=methods,
                        exception_type=type(local).__name__,reason=str(local))]


def require_safe_axis(axis,width):
    if axis_quality(axis,width)['abnormal']:
        raise AxisQualityError('道路轴线仍存在连续折返，必须在 corridor 前修复')
