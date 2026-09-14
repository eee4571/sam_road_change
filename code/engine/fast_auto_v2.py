"""Network/interval Auto, independent of the baseline station analyzer.

Existing profile boundaries define width intervals. Only explicit conflicts use
three cross-section events; neither detection nor qualification creates stations.
Raster evidence never supplies published geometry. Truth is not an input.
"""
from collections import Counter
from dataclasses import dataclass
import time

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely import line_interpolate_point, get_coordinates, line_locate_point
from shapely.ops import substring
from shapely.geometry import Point, LineString

from . import fast_auto_change as legacy
from .auto_presence_candidates import LongitudinalCoverage, line_parts


@dataclass(frozen=True)
class V2Config:
    raster_resolution: float = 1.
    maximum_raster_cells: int = 12_000_000
    stable_position_factor: float = 1.5
    presence_minimum_length: float = 32.
    width_minimum_length: float = 72.
    width_threshold_factor: float = 2.
    source_surface_minimum: float = .55
    opposite_surface_maximum: float = .15
    require_confirmed_absence: bool = True


class ChangeEvidence:
    """Bounded common-grid evidence. Raster boundaries are never published."""
    def __init__(self, before, after, config):
        bounds = [g.bounds for s in (before, after) for g in s.lines]
        self.ready = bool(bounds)
        if not self.ready:
            return
        b = np.asarray(bounds)
        left, bottom = b[:, :2].min(axis=0)-70
        right, top = b[:, 2:].max(axis=0)+70
        self.resolution = max(config.raster_resolution,
                              np.sqrt((right-left)*(top-bottom)/config.maximum_raster_cells))
        self.transform = from_origin(left, top, self.resolution, self.resolution)
        self.height = int(np.ceil((top-bottom)/self.resolution))
        self.width = int(np.ceil((right-left)/self.resolution))
        self.masks = []
        for scene in (before, after):
            shapes = [(g, 1) for g in scene.surfaces.geometry if not g.is_empty]
            self.masks.append(rasterize(shapes, out_shape=(self.height, self.width),
                transform=self.transform, dtype='uint8') if shapes else np.zeros((self.height,self.width),np.uint8))
        self.changed = self.masks[0] != self.masks[1]
        # Distance-to-change supports segment-level localization without GEOS
        # corridor overlays. It never suppresses unmatched presence intervals.
        self.change_distance = (distance_transform_edt(~self.changed).astype('float32')*self.resolution
                                if self.changed.any() else np.full(self.changed.shape,np.inf,np.float32))
        self.road_distance=[distance_transform_edt(mask==0).astype('float32')*self.resolution
                            if mask.any() else np.full(mask.shape,np.inf,np.float32) for mask in self.masks]

    def values(self, array, xy, default=0):
        col = np.floor((xy[...,0]-self.transform.c)/self.resolution).astype(np.int64)
        row = np.floor((self.transform.f-xy[...,1])/self.resolution).astype(np.int64)
        valid = (row>=0)&(row<self.height)&(col>=0)&(col<self.width)
        out = np.full(row.shape, default, dtype=array.dtype)
        out[valid] = array[row[valid],col[valid]]
        return out

    def widths(self, side, axis, positions):
        centers = get_coordinates(line_interpolate_point(axis, positions))
        normals = legacy._normals(axis, positions)
        # Only run-boundary/middle events use this contradiction check. Stored
        # width profiles, not raster widths, define the published corridors.
        step = self.resolution
        distances = np.arange(-np.ceil(65/step), np.ceil(65/step)+1)*step
        hit = self.values(self.masks[side], centers[:,None,:]+normals[:,None,:]*distances[None,:,None]) > 0
        middle = len(distances)//2
        left = np.argmax(~hit[:,:middle+1][:,::-1],axis=1)
        right = np.argmax(~hit[:,middle:],axis=1)
        result = (left+right-1)*step
        result[(~hit[:,middle])|(left==0)|(right==0)] = np.nan
        return result


def _support(evidence,side,axis,tolerance):
    xy=np.asarray(axis.coords)[:,:2]; lengths=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    centers=(xy[:-1]+xy[1:])/2
    points=np.concatenate((centers,xy[:-1],xy[1:]))
    weights=np.concatenate((lengths/2,lengths/4,lengths/4))
    hit=evidence.values(evidence.road_distance[side],points,np.inf)<=tolerance
    return float(np.average(hit,weights=weights)) if weights.sum() else 0.


def _event_coordinates(axis,width,tolerance):
    """Only interval start/middle/end events, independent of its length."""
    positions=np.array([0.,axis.length/2,axis.length])
    points=get_coordinates(line_interpolate_point(axis,positions)); normal=legacy._normals(axis,positions)
    inner=max(width*.70,tolerance+2.)
    centers=(points[:,None,:]+np.array([-1.,0.,1.])[None,:,None]*normal[:,None,:]).reshape(-1,2)
    backgrounds=(points[:,None,:]+np.array([-inner-3,-inner,inner,inner+3])[None,:,None]*normal[:,None,:]).reshape(-1,2)
    return np.concatenate((centers,backgrounds)),len(centers)


def presence_publication(row,config):
    """Explicit absence or strong surface loss; weak ranks alone cannot veto it."""
    source,target=('before','after') if row['change_typ']=='removed' else ('after','before')
    surface_loss=(row.get(f'{source}_valid_ratio',0.) and row.get(f'{target}_valid_ratio',0.)
                  and row.get(f'{source}_surface_ratio',0.)>=config.source_surface_minimum
                  and row.get(f'{target}_surface_ratio',1.)<=config.opposite_surface_maximum
                  and not row.get(f'{target}_probability_ratio',1.))
    if row['length_m']<config.presence_minimum_length:return False,'short_presence_interval'
    if not config.require_confirmed_absence:return True,'interval_evidence'
    if row['qa_state']=='confirmed':return True,'confirmed_absence'
    if surface_loss:return True,'strong_surface_loss_without_opposite_support'
    return False,'opposite_absence_not_confirmed'


def _presence(axis_id,axis,intervals,source,target,side,evidence,tolerance,counts,config):
    parts=[substring(axis,a,b) for a,b in intervals]
    if not parts:return [],[],[]
    widths=[float(np.median(source.widths_at(line_interpolate_point(p,[0.,p.length/2,p.length])))) for p in parts]
    coordinates=[_event_coordinates(p,w,tolerance) for p,w in zip(parts,widths)]
    probabilities=[s.probability.sample_axes(parts,widths,tolerance,coordinates=coordinates) for s in (source,target)]
    counts['v2_probability_event_locations']+=3*len(parts)
    kind='removed' if side==0 else 'added';name='before' if side==0 else 'after'
    records=[];audit=[];detail=[]
    for i,((a,b),part,width) in enumerate(zip(intervals,parts,widths)):
        footprint=part.buffer(max(width/2,tolerance)+1,cap_style='flat')
        ev=[scene.evidence(part,k==0,width,tolerance,probability=probabilities[k][i],
            geometry_evidence=(scene.valid.covers(footprint),_support(evidence,side if k==0 else 1-side,part,tolerance)))
            for k,scene in enumerate((source,target))]
        bef,aft=ev if side==0 else ev[::-1]
        changed=ev[1]['state']!='present'
        qa='confirmed' if ev[0]['state']=='present' and ev[1]['state']=='absent' else 'probable' if ev[0]['state']=='present' else 'uncertain'
        reason=f"{bef['reason']}->{aft['reason']}"
        junction=bool(source.junction.intersects(part) or target.junction.intersects(part))
        audit.append(dict(side=name,axis_id=axis_id,station_m=(a+b)/2,cell_start_m=a,cell_end_m=b,
            candidate_type=kind,matched=False,match_reliable=False,offset_m=None,target_axis=None,junction=junction,
            existence_pass=changed,reason=reason,geometry=part,
            **{f'before_{k}':v for k,v in bef.items()},**{f'after_{k}':v for k,v in aft.items()}))
        detail.append(dict(candidate_type=kind,axis_id=axis_id,start_m=a,end_m=b,length_m=b-a,stage='segment_interval',
                           accepted=changed,qa_state=qa,audit_reason=reason,geometry=part))
        if changed:
            records.append(dict(change_typ=kind,width_bef=width if side==0 else 0.,width_aft=width if side==1 else 0.,
                width_diff=0.,length_m=b-a,source_axis=axis_id,start_m=a,end_m=b,axis_wkt=part.wkt,
                geometry=part.buffer(width/2,cap_style='flat'),confidence={'confirmed':.9,'probable':.6,'uncertain':.3}[qa],
                qa_state=qa,audit_reason=reason,junction=junction,
                **{f'{p}_{field}_ratio':float(v[field]) for p,v in (('before',bef),('after',aft))
                   for field in ('geometry','surface','probability','valid')}))
            records[-1]['v2_publish'],records[-1]['v2_precision_reason']=presence_publication(records[-1],config)
    counts['v2_presence_intervals']+=len(parts)
    return records,audit,detail


def _network_intervals(axis,target,tolerance,buffers,counts):
    """Correspondence at network overlap events; never RoadScene.match."""
    candidates=[]
    for target_id in target.tree.query(axis,predicate='dwithin',distance=tolerance+1e-6):
        target_id=int(target_id);other=target.lines[target_id]
        if target_id not in buffers:buffers[target_id]=other.buffer(tolerance+.1)
        for part in line_parts(axis.intersection(buffers[target_id])):
            if part.length<1.:continue
            xy=np.asarray(part.coords)
            ends=line_locate_point(axis,[Point(xy[0]),Point(xy[-1])]);a,b=float(min(ends)),float(max(ends))
            if b-a<1.:continue
            positions=np.array([a,(a+b)/2,b]);points=line_interpolate_point(axis,positions)
            projected=line_locate_point(other,points);targets=line_interpolate_point(other,projected)
            directions=np.abs(np.sum(legacy._normals(axis,positions)*legacy._normals(other,projected),axis=1))
            if np.median(directions)<.90:continue
            score=float(np.mean([p.distance(q) for p,q in zip(points,targets)])+2*(1-directions.mean()))
            candidates.append((a,b,target_id,score))
    counts['v2_segment_correspondences']+=len(candidates)
    events=sorted({v for a,b,_,_ in candidates for v in (a,b)})
    result=[];previous=None
    for a,b in zip(events,events[1:]):
        active=[r for r in candidates if r[0]<=a+1e-7 and r[1]>=b-1e-7]
        if not active:continue
        chosen=min(active,key=lambda r:(r[3]-(.15 if r[2]==previous else 0),r[2]))
        target_id=chosen[2]
        if result and result[-1][2]==target_id and abs(result[-1][1]-a)<1e-7:
            result[-1]=(result[-1][0],b,target_id)
        else:result.append((a,b,target_id))
        previous=target_id;counts['v2_ambiguous_intervals_resolved']+=int(len(active)>1)
    return result


def _profile_events(scene,part,axis,start,end):
    ids=scene.width_tree.query(part,predicate='dwithin',distance=3.)
    points=[]
    for i in ids:
        xy=get_coordinates(scene.width_geometries[i])
        if len(xy):points.extend((Point(xy[0]),Point(xy[-1])))
    if not points:return []
    positions=line_locate_point(axis,points)
    return positions[(positions>start+1e-6)&(positions<end-1e-6)].tolist()


def _event_validation(axis,other,start,end,before,after,counts):
    """At most three paired cross sections for an explicit profile conflict."""
    positions=np.array([start,(start+end)/2,end]);points=line_interpolate_point(axis,positions)
    projected=line_locate_point(other,points);targets=line_interpolate_point(other,projected)
    normals=legacy._normals(axis,positions);an=legacy._normals(other,projected)
    an[np.sum(normals*an,axis=1)<0]*=-1;normals+=an
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-9)
    part=substring(axis,start,end);surfaces=[s.surface(part) for s in (before,after)]
    supports=[s.buffer(.1) for s in surfaces];config=legacy.PairedWidthConfig();values=[]
    for p,q,n in zip(points,targets,normals):
        measurements=[legacy._measure_period_width(point,n,surface,support,scene.probability,scene.crs,config)
                      for scene,point,surface,support in zip((before,after),(p,q),surfaces,supports)]
        counts['v2_exact_width_event_sections']+=2
        b,a=measurements
        if b.final_width is not None and a.final_width is not None and not b.reject_reason and not a.reject_reason:
            values.append((b.final_width,a.final_width))
    return np.median(values,axis=0) if len(values)>=2 else None


def _width_changes(axis_id,axis,intervals,before,after,evidence,absolute,relative,minimum_length,minimum_area,counts,config):
    records=[];audit=[]
    for start,end,target_id in intervals:
        if end-start<minimum_length:continue
        part=substring(axis,start,end);other=after.lines[target_id]
        projected=line_locate_point(other,line_interpolate_point(axis,[start,end]))
        other_part=substring(other,float(min(projected)),float(max(projected)))
        events=sorted(set([start,end]+_profile_events(before,part,axis,start,end)+_profile_events(after,other_part,axis,start,end)))
        a=np.asarray(events[:-1]);b=np.asarray(events[1:]);valid=b-a>1e-6;a=a[valid];b=b[valid]
        if not len(a):continue
        # Existing profile endpoints define these intervals, not sample spacing.
        points=line_interpolate_point(axis,(a+b)/2);target_positions=line_locate_point(other,points)
        bw=before.widths_at(points);aw=after.widths_at(line_interpolate_point(other,target_positions))
        threshold=np.maximum(absolute,relative*np.maximum(bw,aw));diff=aw-bw
        sign=np.where(np.abs(diff)>=threshold,np.sign(diff),0).astype(int)
        counts['v2_width_profile_intervals']+=len(sign)
        runs=[]
        for i,s in enumerate(sign):
            if s==0:continue
            if runs and runs[-1][2]==s and abs(runs[-1][1]-a[i])<1e-6:
                runs[-1]=(runs[-1][0],float(b[i]),int(s),runs[-1][3]+[i])
            else:runs.append((float(a[i]),float(b[i]),int(s),[i]))
        if not runs:counts['v2_stable_segments']+=1
        for low,high,direction,indexes in runs:
            if high-low<minimum_length:continue
            weights=b[indexes]-a[indexes]
            before_width=float(np.average(bw[indexes],weights=weights));after_width=float(np.average(aw[indexes],weights=weights))
            local=substring(axis,low,high)
            inset=min(evidence.resolution,(high-low)/4)
            positions=np.array([low+inset,(low+high)/2,high-inset]);tpositions=line_locate_point(other,line_interpolate_point(axis,positions))
            rb=evidence.widths(0,axis,positions);ra=evidence.widths(1,other,tpositions)
            conflict=np.isfinite(rb).all() and np.isfinite(ra).all() and direction*float(np.median(ra-rb)) < -max(absolute,2*evidence.resolution)
            if conflict:
                corrected=_event_validation(axis,other,low,high,before,after,counts)
                if corrected is not None:before_width,after_width=map(float,corrected)
            difference=after_width-before_width
            if abs(difference)<max(absolute,relative*max(before_width,after_width)):continue
            # Require sustained margin on every existing profile interval;
            # an average must not hide short spikes or near-threshold sections.
            strong = np.abs(diff[indexes]) >= threshold[indexes]*config.width_threshold_factor
            sustained = max((sum(b[j]-a[j] for j in group) for group in _strong_runs(indexes,strong)),default=0.)
            publish = (sustained >= max(minimum_length,config.width_minimum_length)
                       and abs(difference) >= config.width_threshold_factor*max(absolute,relative*max(before_width,after_width)))
            xy=np.asarray(local.coords)[:,:2]
            partners=get_coordinates(line_interpolate_point(other,line_locate_point(other,[Point(p) for p in xy])))
            canonical=LineString((xy+partners)/2)
            outer,inner=max(before_width,after_width),min(before_width,after_width)
            geometry=canonical.buffer(outer/2,cap_style='flat').difference(canonical.buffer(inner/2,cap_style='flat'))
            if geometry.area<minimum_area:continue
            records.append(dict(change_typ='widened' if difference>0 else 'narrowed',width_bef=before_width,width_aft=after_width,
                width_diff=difference,length_m=local.length,axis_wkt=canonical.wkt,geometry=geometry,source_axis=axis_id,start_m=low,end_m=high,
                qa_state='probable' if conflict else 'confirmed',confidence=.6 if conflict else .85,audit_reason='width_profile_interval',junction=False,
                v2_publish=publish,v2_precision_reason='sustained_width_margin' if publish else 'small_or_short_width_fluctuation'))
            audit.append(dict(axis_id=axis_id,target_axis=target_id,start_m=low,end_m=high,sign=direction,accepted=True,
                              profile_interval_count=len(indexes),event_validation=bool(conflict),geometry=canonical))
    return records,audit


def _strong_runs(indexes, strong):
    run=[]
    for index,keep in zip(indexes,strong):
        if keep:run.append(index)
        elif run:
            yield run
            run=[]
    if run:yield run


def analyze_scenes(before,after,*,tolerance=3.,absolute=2.,relative=.2,minimum_length=24.,minimum_area=4.,presence_audit=None,config=V2Config()):
    started=time.perf_counter();counts=Counter(v2_enabled=1,v2_station_count=0,v2_legacy_analyzer_calls=0,
                                              v2_exact_width_event_sections=0,v2_probability_event_locations=0)
    records=[];audit=[];width_audit=[];evidence=ChangeEvidence(before,after,config)
    counts['timing_v2_change_evidence_seconds']=time.perf_counter()-started
    if not evidence.ready:return records,audit,width_audit,dict(counts)
    counts['v2_evidence_resolution_m']=float(evidence.resolution);buffers={}
    for side,source,target in ((0,before,after),(1,after,before)):
        print(f'[Fast v2] {"before" if side==0 else "after"} network intervals',flush=True)
        coverage=LongitudinalCoverage(target.lines,tolerance*config.stable_position_factor)
        for axis_id,axis in enumerate(source.lines):
            tick=time.perf_counter();intervals=coverage.uncovered(axis)
            local,observations,details=_presence(axis_id,axis,intervals,source,target,side,evidence,tolerance,counts,config)
            records.extend(local);audit.extend(observations)
            if presence_audit is not None:presence_audit.extend(details)
            counts['timing_v2_presence_seconds']+=time.perf_counter()-tick
            if side==0:
                tick=time.perf_counter();matched=_network_intervals(axis,after,tolerance,buffers,counts)
                counts['timing_v2_correspondence_seconds']+=time.perf_counter()-tick
                tick=time.perf_counter();local,details=_width_changes(axis_id,axis,matched,before,after,evidence,absolute,relative,minimum_length,minimum_area,counts,config)
                records.extend(local);width_audit.extend(details)
                counts['timing_v2_width_profile_seconds']+=time.perf_counter()-tick
            counts['v2_network_axes']+=1
    counts['v2_local_sample_count']=counts['v2_probability_event_locations']+counts['v2_exact_width_event_sections']//2
    counts['timing_v2_total_seconds']=time.perf_counter()-started
    print('[Fast v2] '+str(dict(counts)),flush=True)
    return records,audit,width_audit,dict(counts)


def qualify_candidates(candidates,*,minimum_length,minimum_area):
    """Already-classified intervals must not re-enter station-based review."""
    result=candidates.copy();result['candidate_qa_state']=result.qa_state
    accepted=((result.length_m>=minimum_length)&(result.geometry.area>=minimum_area)) if len(result) else np.array([],bool)
    if 'v2_publish' in result:accepted &= result.v2_publish.fillna(False).astype(bool)
    result['publication_state']=np.where(accepted,'accepted','review')
    result['precision_reason']=result['v2_precision_reason'] if 'v2_precision_reason' in result else np.where(accepted,'v2_interval_classification','short_or_small_interval')
    return result,[]
