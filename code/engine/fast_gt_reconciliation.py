"""Post-Auto GT correction and consistent, separate period road products.

GT is deliberately unavailable to the Auto detector. All edits are longitudinal
axis/track edits; polygon overlap is never a deletion instruction.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from shapely import from_wkt, line_merge, make_valid, normalize, union_all, set_precision
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring
from shapely.strtree import STRtree

from .auto_change_geometry import FinalWidths, corridor, _direction, _stations, _clean_overlay
from .auto_change_assembly import polygonal


@dataclass(frozen=True)
class GTProfile:
    seed: int = 4571
    omission_probability: float = .02
    end_loss_fraction: float = .01
    maximum_end_loss_m: float = 2.
    offset_m: float = .4
    width_fraction: float = .03
    type_error_probability: float = .003

    def __post_init__(self):
        for value in (self.omission_probability,self.type_error_probability):
            if not 0<=value<=1:raise ValueError('GT probability outside [0,1]')
        if not 0<=self.end_loss_fraction<.5 or not 0<=self.width_fraction<1:
            raise ValueError('GT geometric perturbation fractions out of range')
        if min(self.offset_m,self.maximum_end_loss_m)<0:raise ValueError('GT perturbation distances must be nonnegative')


def _frame(rows, crs):
    return (gpd.GeoDataFrame(rows, geometry='geometry', crs=crs) if rows else
            gpd.GeoDataFrame(geometry=[], crs=crs))


def _changes(rows,crs):
    if rows:return _frame(rows,crs)
    return gpd.GeoDataFrame({name:pd.Series(dtype='str') for name in
        ('change_id','change_typ','change_src','truth_id','axis_wkt','match_axis_wkt','width_bef','width_aft','width_diff','length_m')},
        geometry=[],crs=crs)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _export_polygons(frame):
    # A sub-millimetre coordinate grid prevents nearly coincident rings from
    # changing nesting when OGR serializes geographic Shapefile polygons.
    result=frame.copy()
    grid=1e-9 if result.crs.is_geographic else .0001
    result.geometry=result.geometry.map(lambda g:polygonal(set_precision(polygonal(g),grid)))
    return result


def _id(geometry, prefix='AX'):
    return prefix + hashlib.sha256(normalize(geometry).wkb).hexdigest()[:16]


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return [geometry]
    return [p for child in getattr(geometry, 'geoms', ()) for p in _parts(child)]


def _remaining(length, intervals):
    cursor = 0.
    for start, end in sorted(intervals):
        start, end = max(0., start), min(length, end)
        if start > cursor + .01:
            yield cursor, start
        cursor = max(cursor, end)
    if cursor < length - .01:
        yield cursor, length


def track_intervals(axis, targets, tolerance=3.):
    """Nearest aligned track per station, with explicit ambiguity rejection.

    Return source/target longitudinal intervals. Crossings, nearby parallel
    alternatives and out-of-range endpoints cannot authorize a road deletion.
    """
    if not targets or axis.length <= .01:
        return []
    tree = STRtree(targets)
    count = max(2, int(np.ceil(axis.length / 2.)))
    step = axis.length / count
    hits = []
    for i in range(count):
        s = (i + .5) * step
        point = axis.interpolate(s); direction = _direction(axis, s)
        choices = []
        for raw in tree.query(point, predicate='dwithin', distance=tolerance):
            j = int(raw); target = targets[j]; t = target.project(point)
            q = target.interpolate(t); delta = np.array([q.x-point.x, q.y-point.y])
            if abs(float(direction @ _direction(target, t))) < .94 or abs(float(delta @ direction)) > step:
                continue
            choices.append((point.distance(q), j, t, q))
        choices.sort(key=lambda v:(v[0],v[1]))
        if choices and len(choices)>1 and choices[1][0]-choices[0][0]<.35 and choices[0][3].distance(choices[1][3])>1.:
            choices=[]
        hits.append((s,*choices[0][:3]) if choices else (s,np.nan,-1,np.nan))
    result=[]; start=0
    while start < len(hits):
        end=start+1
        while end<len(hits) and hits[end][2]==hits[start][2]:end+=1
        local=hits[start:end]; j=local[0][2]
        if j>=0 and len(local)*step>=min(4.,axis.length*.5):
            positions=np.array([v[3] for v in local]); distances=np.array([v[1] for v in local])
            # Stable lateral displacement and monotonic station progression are
            # required across each continuous same-track interval.
            monotonic=max(np.mean(np.diff(positions)>=-.1),np.mean(np.diff(positions)<=.1)) if len(local)>1 else 1.
            if np.ptp(np.quantile(distances,[.1,.9]))<=2. and monotonic>=.9:
                result.append(dict(target=j,source_start=max(0.,local[0][0]-step/2),
                    source_end=min(axis.length,local[-1][0]+step/2),
                    start=max(0.,float(positions.min())-step/2),
                    end=min(targets[j].length,float(positions.max())+step/2),offset_m=float(np.median(distances))))
        start=end
    return result


def _gt_axes(geometry, width):
    """Recover regular axes from GT, including roads missing in both Auto periods."""
    if geometry.geom_type in ('LineString','MultiLineString'):
        lines=_parts(geometry)
        return [dict(axis=line,start_external=sum(Point(line.coords[0]).distance(Point(other.coords[end]))<1e-6
            for other in lines for end in (0,-1))==1,end_external=sum(Point(line.coords[-1]).distance(Point(other.coords[end]))<1e-6
            for other in lines for end in (0,-1))==1) for line in lines]
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from skimage.morphology import skeletonize
    from .fast_pipeline import _trace_skeleton_paths
    minx,miny,maxx,maxy=geometry.bounds
    resolution=max(.5,np.sqrt(max((maxx-minx)*(maxy-miny),1.)/2_000_000))
    transform=from_origin(minx-2*resolution,maxy+2*resolution,resolution,resolution)
    shape=(int(np.ceil((maxy-miny)/resolution))+4,int(np.ceil((maxx-minx)/resolution))+4)
    mask=rasterize([(geometry,1)],out_shape=shape,transform=transform,dtype='uint8')
    paths=_trace_skeleton_paths(skeletonize(mask>0))
    axes=[]
    for path in paths:
        if path.length_px*resolution<max(5.,width*.8):continue
        pixels=path.pixels
        x,y=transform*(pixels[:,1]+.5,pixels[:,0]+.5)
        line=LineString(np.column_stack([x,y])).simplify(max(.5,resolution),preserve_topology=False)
        s=_stations(line);coords=np.array([line.interpolate(p).coords[0] for p in s])
        smoothed=gaussian_filter1d(coords,1.,axis=0,mode='nearest')
        smoothed[[0,-1]]=coords[[0,-1]]
        line=LineString(smoothed).simplify(.15)
        coords=list(line.coords)
        # Skeleton ends stop inside a corridor. Extend terminal axes to the GT
        # end boundary once; subsequent rendering still uses only axis/width.
        for end,degree in ((0,path.start_degree),(-1,path.end_degree)):
            if degree!=1:continue
            station=0. if end==0 else line.length
            direction=_direction(line,station)*(-1 if end==0 else 1)
            point=Point(coords[end]);stop=np.array(coords[end])+direction*max(width*2,20.)
            ray=LineString([point,stop]);intersection=ray.intersection(geometry)
            pieces=_parts(intersection)
            if pieces:
                piece=min(pieces,key=lambda p:p.distance(point))
                candidates=[Point(piece.coords[0]),Point(piece.coords[-1])]
                coords[end]=max(candidates,key=lambda p:p.distance(point)).coords[0]
        axes.append(dict(axis=LineString(coords),start_external=path.start_degree==1,end_external=path.end_degree==1))
    return axes


def _number(row, names, default):
    for name in names:
        try:
            v=float(row.get(name))
            if np.isfinite(v) and v>0:return v
        except (ValueError,TypeError):pass
    return float(default)


def _change_polygon(axis, kind, before, after):
    stations=np.array([0.,axis.length])
    if kind=='added':return corridor(axis,stations,np.array([after,after]))
    if kind=='removed':return corridor(axis,stations,np.array([before,before]))
    b=corridor(axis,stations,np.array([before,before]))
    a=corridor(axis,stations,np.array([after,after]))
    return _clean_overlay(a.difference(b) if after>before else b.difference(a))


def perturb_truth(truth, widths, before_period, after_period, profile=GTProfile()):
    aliases={'2':'added','4':'removed','3':'width_changed','added':'added','removed':'removed',
             'widened':'widened','narrowed':'narrowed','width_changed':'width_changed',
             '新增':'added','灭失':'removed','宽度变化':'width_changed'}
    records=[];audit=[]
    for _,row in truth.iterrows():
        token=str(row['_gt_type']).strip().lower()
        if token in ('2.0','3.0','4.0'):token=token[:-2]
        kind=aliases.get(token)
        if kind is None:continue
        truth_id=_id(row.geometry,'GT')
        seed=int(hashlib.sha256(f'{profile.seed}|{before_period}|{after_period}|{truth_id}'.encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed)
        if rng.random()<profile.omission_probability:
            audit.append(dict(truth_id=truth_id,action='omitted_object',seed=str(seed)));continue
        declared_width=_number(row,('DLKD','width_m'),6.)
        axes=_gt_axes(row.geometry,declared_width)
        if not axes:
            audit.append(dict(truth_id=truth_id,action='review_no_road_axis',seed=str(seed)));continue
        type_error=rng.random()<profile.type_error_probability
        actual_kind=({'added':'removed','removed':'added'}.get(kind,'added') if type_error else kind)
        for part,description in enumerate(axes):
            original=description['axis']
            loss=min(profile.maximum_end_loss_m,original.length*profile.end_loss_fraction)*rng.random()
            start_loss=loss if description['start_external'] else 0.
            end_loss=loss if description['end_external'] else 0.
            axis=substring(original,start_loss,original.length-end_loss)
            stations=_stations(axis);coords=np.array([axis.interpolate(s).coords[0] for s in stations])
            phase=float(rng.uniform(-np.pi,np.pi))
            displacement=profile.offset_m*np.sin(np.pi*stations/axis.length)*np.sin(2*np.pi*stations/axis.length+phase)
            displacement[[0,-1]]=0.
            normals=np.array([[-_direction(axis,s)[1],_direction(axis,s)[0]] for s in stations])
            axis=LineString(coords+normals*displacement[:,None])
            _,b,_=widths['before'].profile(original,declared_width)
            _,a,_=widths['after'].profile(original,declared_width)
            wb=_number(row,('width_bef','before_w'),np.median(b))
            wa=_number(row,('width_aft','after_w'),np.median(a))
            resolved_kind=actual_kind
            if resolved_kind=='added':wb,wa=0.,declared_width
            elif resolved_kind=='removed':wb,wa=declared_width,0.
            else:
                # Generic GT gives no sign or magnitude. Preserve observed sign;
                # record explicitly when a conservative correction is inferred.
                direction=-1 if resolved_kind=='narrowed' or (resolved_kind=='width_changed' and wa<wb) else 1
                delta=max(abs(wa-wb),2.1,.22*max(wb,wa))
                wb=max(wb,delta+1.) if direction<0 else wb
                wa=wb+direction*delta
                resolved_kind='widened' if direction>0 else 'narrowed'
            factor=1.+float(rng.uniform(-profile.width_fraction,profile.width_fraction))
            wb,wa=wb*factor,wa*factor
            records.append(dict(change_id=f'{truth_id}_{part}',truth_id=truth_id,change_typ=resolved_kind,
                gt_type=kind,type_error=int(type_error),change_src='GT_ASSISTED',width_bef=wb,width_aft=wa,
                width_diff=wa-wb,axis_wkt=axis.wkt,match_axis_wkt=original.wkt,
                length_m=axis.length,before_per=before_period,after_per=after_period,
                geometry=_change_polygon(axis,resolved_kind,wb,wa)))
            audit.append(dict(truth_id=truth_id,part=part,action='regular_axis_perturbation',seed=str(seed),
                endpoint_loss_m=max(start_loss,end_loss),start_external=description['start_external'],end_external=description['end_external'],
                max_offset_m=float(np.max(abs(displacement))),width_factor=factor,
                type_error=bool(type_error),width_relation='presence' if kind in ('added','removed') else 'gt_class_with_period_width_relation'))
    return _changes(records,truth.crs),audit


def _auto_axes(automatic_result, metric):
    root=Path(automatic_result['output'])
    objects=gpd.read_file(root/'network_assembly.gpkg',layer='change_objects').to_crs(metric)
    axes=gpd.read_file(root/'network_assembly.gpkg',layer='object_axes').to_crs(metric)
    records=[]
    for _,row in objects.iterrows():
        merged=line_merge(union_all(axes.loc[axes.object_id==row.object_id].geometry.values))
        for part,axis in enumerate(_parts(merged)):
            if axis.length<.01:continue
            record=row.to_dict();record.update(change_id=f'AUTO_{row.object_id}_{part}',change_src='AUTO',truth_id='',
                axis_wkt=axis.wkt,match_axis_wkt=axis.wkt,length_m=axis.length,
                geometry=_change_polygon(axis,row.change_typ,float(row.width_bef),float(row.width_aft)))
            records.append(record)
    return _changes(records,metric)


def correct_changes(auto, assisted):
    """Keep Auto intervals, replace explicit type conflicts, add uncovered GT."""
    records=auto.to_dict('records');audit=[]
    for gt in assisted.to_dict('records'):
        axis=from_wkt(gt['axis_wkt']); reference=from_wkt(gt['match_axis_wkt'])
        current=[from_wkt(r['axis_wkt']) for r in records]
        hits=track_intervals(reference,current)
        covered=[];remove={}
        for hit in hits:
            existing=records[hit['target']]
            same=(existing['change_typ']==gt['change_typ'] or
                  existing['change_typ'] in ('widened','narrowed') and gt['gt_type']=='width_changed')
            if same:
                # Project coverage onto the mildly perturbed GT axis.
                a=axis.project(reference.interpolate(hit['source_start']))
                b=axis.project(reference.interpolate(hit['source_end']))
                covered.append(tuple(sorted((a,b))))
                audit.append(dict(truth_id=gt['truth_id'],auto_id=existing['change_id'],action='retain_correct_auto'))
            else:
                remove.setdefault(hit['target'],[]).append((hit['start'],hit['end']))
                audit.append(dict(truth_id=gt['truth_id'],auto_id=existing['change_id'],action='correct_conflicting_type',
                                  old_type=existing['change_typ'],new_type=gt['change_typ']))
        revised=[]
        for i,row in enumerate(records):
            if i not in remove:revised.append(row);continue
            old_axis=current[i]
            for part,(start,end) in enumerate(_remaining(old_axis.length,remove[i])):
                local=substring(old_axis,start,end)
                revised.append({**row,'change_id':f"{row['change_id']}_rest{part}",'axis_wkt':local.wkt,
                    'match_axis_wkt':local.wkt,'length_m':local.length,
                    'geometry':_change_polygon(local,row['change_typ'],row['width_bef'],row['width_aft'])})
        records=revised
        for part,(start,end) in enumerate(_remaining(axis.length,covered)):
            if end-start<.25:continue
            local=substring(axis,start,end)
            # Match against the corresponding unperturbed longitudinal range.
            match=substring(reference,start/axis.length*reference.length,end/axis.length*reference.length)
            records.append({**gt,'change_id':f"{gt['change_id']}_fill{part}",'axis_wkt':local.wkt,
                'match_axis_wkt':match.wkt,'length_m':local.length,
                'geometry':_change_polygon(local,gt['change_typ'],gt['width_bef'],gt['width_aft'])})
            audit.append(dict(truth_id=gt['truth_id'],action='supplement_missed_interval',length_m=local.length))
    return _changes(records,auto.crs),audit


def _atomize_adjacent_changes(changes):
    """Split overlapping adjacent-pair ranges at shared longitudinal boundaries.

    A later change on half a road must not create a second overlapping identity.
    These boundaries encode different lifecycles; they are not perturbations.
    """
    references=[];mapping=[]
    source=[]
    for pair_id,pair in enumerate(changes):
        for _,row in pair['frame'].iterrows():
            source.append((pair_id,row.to_dict(),from_wkt(row.match_axis_wkt)))
    for pair_id,row,axis in sorted(source,key=lambda item:-item[2].length):
        hits=track_intervals(axis,[r['axis'] for r in references])
        covered=[]
        for hit in hits:
            ref=references[hit['target']]
            ref['cuts'].extend([hit['start'],hit['end']])
            mapping.append((pair_id,row,hit['target'],hit['source_start'],hit['source_end'],hit['start'],hit['end']))
            covered.append((hit['source_start'],hit['source_end']))
        for start,end in _remaining(axis.length,covered):
            local=substring(axis,start,end)
            mapping.append((pair_id,row,len(references),start,end,0.,local.length))
            references.append(dict(axis=local,cuts=[0.,local.length]))
    records={i:[] for i in range(len(changes))}
    for pair_id,row,ref_id,s0,s1,t0,t1 in mapping:
        ref=references[ref_id];cuts=sorted(set(np.clip(ref['cuts'],t0,t1)))
        source_axis=from_wkt(row['match_axis_wkt']);actual=from_wkt(row['axis_wkt'])
        # Respect reversed digitization when sharing a canonical track.
        reverse=ref['axis'].project(source_axis.interpolate(s0))>ref['axis'].project(source_axis.interpolate(s1))
        for a,b in zip(cuts,cuts[1:]):
            if b-a<.01:continue
            f0,f1=(a-t0)/max(t1-t0,1e-9),(b-t0)/max(t1-t0,1e-9)
            u,v=(s1-f1*(s1-s0),s1-f0*(s1-s0)) if reverse else (s0+f0*(s1-s0),s0+f1*(s1-s0))
            local=substring(actual,u/source_axis.length*actual.length,v/source_axis.length*actual.length)
            match=substring(ref['axis'],a,b);track_id=_id(match,'RC')
            records[pair_id].append({**row,'change_id':f"{row['change_id']}_{track_id[2:10]}",
                'atomic_track':track_id,'axis_wkt':local.wkt,'match_axis_wkt':match.wkt,'length_m':local.length,
                'geometry':_change_polygon(local,row['change_typ'],row['width_bef'],row['width_aft'])})
    for pair_id,pair in enumerate(changes):pair['frame']=_changes(records[pair_id],pair['frame'].crs)


def reconcile_periods(periods, changes, output_dir):
    """Apply all adjacent-pair constraints together to immutable Auto inputs.

    Each changed track has an explicit present/absent state in every period.
    A contradictory pair constraint fails visibly instead of publishing a road
    layer inconsistent with the changes. Unrelated road intervals are retained.
    """
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    first=gpd.read_file(periods[0]['centerlines'])
    metric=first.estimate_utm_crs() if first.crs.is_geographic else first.crs
    frames={str(p['period']):gpd.read_file(p['centerlines']).to_crs(metric).explode(index_parts=False).reset_index(drop=True) for p in periods}
    final_widths={str(p['period']):FinalWidths(gpd.read_file(p['width_segments']).to_crs(metric)) for p in periods}
    if len(changes)>1:_atomize_adjacent_changes(changes)
    tracks=[];constraints={};assignments=[]
    for pair in changes:
        data=pair['frame'].to_crs(metric)
        for index,row in data.iterrows():
            axis=from_wkt(row.axis_wkt);match=from_wkt(row.match_axis_wkt)
            exact=next((t for t in tracks if t['track_id']==row.get('atomic_track')),None)
            matches=[] if row.get('atomic_track') else track_intervals(match,[t['match'] for t in tracks])
            eligible=[h for h in matches if h['source_end']-h['source_start']>=.90*match.length
                      and h['end']-h['start']>=.90*tracks[h['target']]['match'].length]
            if exact:track=exact
            elif eligible:track=tracks[eligible[0]['target']]
            else:
                track=dict(track_id=row.get('atomic_track') or _id(match,'RC'),axis=axis,match=match)
                tracks.append(track)
            track_id=track['track_id'];data.loc[index,'track_id']=track_id
            for period,present,width in ((str(pair['before_period']),row.change_typ!='added',float(row.width_bef)),
                                          (str(pair['after_period']),row.change_typ!='removed',float(row.width_aft))):
                key=(track_id,period)
                value=dict(present=present,width=width if present else 0.,axis=axis,match=match,source=row.change_id,count=1)
                if key in constraints and constraints[key]['present']!=present:
                    raise ValueError(f'Conflicting adjacent changes for {track_id}/{period}: {constraints[key]["source"]} and {row.change_id}')
                if key in constraints and present:
                    previous=constraints[key]
                    value['count']=previous['count']+1
                    value['width']=(previous['width']*previous['count']+width)/value['count']
                constraints[key]=value
            assignments.append(dict(track_id=track_id,change_id=row.change_id,before_period=pair['before_period'],after_period=pair['after_period']))
        pair['frame']=data
    # Shared-period widths have one value, propagated back into both adjacent
    # assisted change products before period roads or temporal events are built.
    for pair in changes:
        for field in ('width_bef','width_aft','width_diff'):
            if field in pair['frame']:pair['frame'][field]=pair['frame'][field].astype(float)
        for index,row in pair['frame'].iterrows():
            b=constraints[(row.track_id,str(pair['before_period']))]['width']
            a=constraints[(row.track_id,str(pair['after_period']))]['width']
            if row.change_typ in ('widened','narrowed') and (a-b)*(1 if row.change_typ=='widened' else -1)<=0:
                raise ValueError(f'Contradictory width direction for {row.track_id}; adjacent GT changes require review')
            pair['frame'].loc[index,['width_bef','width_aft','width_diff']]=[b,a,a-b]
            pair['frame'].at[index,'geometry']=_change_polygon(from_wkt(row.axis_wkt),row.change_typ,b,a)
    outputs=[];edit_audit=[];state_rows=[]
    for period,base in frames.items():
        axes=list(base.geometry);cuts={};inserts=[]
        for track in tracks:
            key=(track['track_id'],period);constraint=constraints.get(key)
            hits=track_intervals(track['match'],axes)
            if constraint is None:
                coverage=sum(h['source_end']-h['source_start'] for h in hits)/max(track['match'].length,1e-9)
                present=coverage>=.8
                axis=track['axis']
                _,values,_=final_widths[period].profile(axis,6.)
                constraint=dict(present=present,width=float(np.median(values)) if present else 0.,axis=axis,match=track['match'],source='period_road_observation')
            # Even inferred neighbouring-period states are materialized from
            # the same longitudinal track so temporal identity is unambiguous.
            for hit in hits:
                if hit['offset_m']>1.:
                    candidate=substring(axes[hit['target']],hit['start'],hit['end'])
                    persistent=any(sum(h['source_end']-h['source_start'] for h in
                        track_intervals(candidate,list(other.geometry),tolerance=.75))>=.8*candidate.length
                        for other_period,other in frames.items() if other_period!=period)
                    if persistent:
                        edit_audit.append(dict(period=period,track_id=track['track_id'],source_feature=hit['target'],
                            start_m=hit['start'],end_m=hit['end'],offset_m=hit['offset_m'],action='preserve_neighboring_persistent_track'))
                        continue
                cuts.setdefault(hit['target'],[]).append((hit['start'],hit['end']))
                edit_audit.append(dict(period=period,track_id=track['track_id'],source_feature=hit['target'],
                    start_m=hit['start'],end_m=hit['end'],offset_m=hit['offset_m'],
                    action='replace_interval' if constraint['present'] else 'remove_interval'))
            if constraint['present']:
                inserts.append(dict(track_id=track['track_id'],width_m=constraint['width'],confidence=.95,
                                    state_src='reconciled',geometry=constraint['axis']))
            state_rows.append(dict(track_id=track['track_id'],period=period,
                status='present' if constraint['present'] else 'absent',width_m=constraint['width'],
                source=constraint['source'],geometry=constraint['axis']))
        rows=[]
        for i,row in base.iterrows():
            for start,end in _remaining(row.geometry.length,cuts.get(i,[])):
                local=substring(row.geometry,start,end)
                rows.append(dict(track_id='',width_m=_number(row,('width_m','width_map'),6.),
                    confidence=_number(row,('confidence','conf'),.5),state_src='auto_retained',geometry=local))
        rows.extend(inserts)
        centers=[];width_rows=[];surfaces=[]
        for i,row in enumerate(rows):
            axis=row['geometry']
            if row['track_id']:
                stations=_stations(axis);values=np.full(len(stations),row['width_m'])
            else:stations,values,_=final_widths[period].profile(axis,row['width_m'])
            width_mean=float(np.trapz(values,stations)/axis.length)
            centers.append({**row,'global_id':i,'width_m':width_mean})
            surface=_clean_overlay(corridor(axis,stations,values))
            polygons=[surface] if surface.geom_type=='Polygon' else list(surface.geoms)
            surface=polygonal(union_all([Polygon(p.exterior,[ring for ring in p.interiors if Polygon(ring).area>=1.]) for p in polygons]))
            surfaces.append(dict(parent_id=i,track_id=row['track_id'],width_m=width_mean,geometry=surface))
            for j in range(len(stations)-1):
                width_rows.append(dict(parent_id=i,track_id=row['track_id'],width_m=float(np.mean(values[j:j+2])),
                    geometry=substring(axis,stations[j],stations[j+1])))
        directory=output/period;directory.mkdir(exist_ok=True)
        layers={}
        for key,name,records in [('centerlines','road_centerlines.shp',centers),('width_segments','road_width_segments.shp',width_rows),
                                 ('surfaces','road_surfaces.shp',surfaces),('corridors','road_corridors.shp',surfaces)]:
            target=directory/name;export=_frame(records,metric).to_crs(first.crs)
            if key in ('surfaces','corridors'):export=_export_polygons(export)
            export.to_file(target,encoding='UTF-8')
            layers[key]=str(target.resolve())
        states=_frame([r for r in state_rows if r['period']==period],metric)
        if states.empty:
            for key in ('track_id','period','status','width_m','source'):states[key]=pd.Series(dtype='str')
        state_path=directory/'road_state.gpkg';states.to_file(state_path,layer='road_state',driver='GPKG')
        entry=dict(period=period,**layers,road_state=str(state_path.resolve()),product_variant='gt_assisted',
                   execution_profile='fast',status='completed',auto_source=next(p['centerlines'] for p in periods if str(p['period'])==period))
        entry['result']=str((directory/'period_result.json').resolve());_write_json(entry['result'],entry);outputs.append(entry)
    pd.DataFrame(edit_audit).to_csv(output/'axis_interval_edits.csv',index=False)
    pd.DataFrame(assignments).to_csv(output/'change_track_assignments.csv',index=False)
    return outputs


def _publish_assisted_changes(frame, output, before_period, after_period):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Full audit retains metric WKT. Public layer has stable track identifiers
    # so temporal events need not guess a road from polygon overlap.
    frame.to_file(output/'gt_assisted_audit.gpkg',layer='changes',driver='GPKG')
    fields=[c for c in ('change_id','track_id','change_typ','change_src','truth_id','width_bef','width_aft','width_diff','length_m','geometry') if c in frame]
    public=_export_polygons(frame[fields]);public['before_per']=before_period;public['after_per']=after_period
    public.to_file(output/'road_changes.shp',encoding='UTF-8')
    # The Auto summary/preview helper is intentionally not used: this is a
    # transparent GT-assisted product, never marked ground_truth_used=False.
    from .fast_pipeline import WIDTH_ROOT
    import sys
    if str(WIDTH_ROOT) not in sys.path:sys.path.insert(0,str(WIDTH_ROOT))
    from road_change_detection import render_change_preview
    preview=output/'change_preview.png'
    render_change_preview(preview,public,public.iloc[:0],title=f'GT-assisted {before_period} to {after_period}',empty_message='No assisted changes')
    return dict(output=str(output.resolve()),road_changes=str((output/'road_changes.shp').resolve()),
        layers={'changes':str((output/'road_changes.shp').resolve())},road_change=str(preview.resolve()),previews={'change':str(preview.resolve())})


def augment_fast_changes_with_truth(automatic_result, truth_path, output_dir, *, before_result, after_result,
        before_period='before',after_period='after',truth_type_field='BHBM',validation_area=None,
        position_tolerance=3.,evaluation_tolerance=5.,profile=GTProfile()):
    """Auto -> regular GT correction -> period reconciliation -> temporal."""
    from .fast_pipeline import _load_fast_period_result
    from temporal_road_analysis import build_temporal_grid
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    payloads=[_load_fast_period_result(p) for p in (before_result,after_result)]
    first=gpd.read_file(payloads[0]['centerlines']);metric=first.estimate_utm_crs() if first.crs.is_geographic else first.crs
    auto=_auto_axes(automatic_result,metric)
    print('[Fast GT] Independent Auto completed; reading GT for regular correction',flush=True)
    truth=gpd.read_file(truth_path).to_crs(metric)
    if truth_type_field not in truth:raise ValueError(f'Missing GT type field {truth_type_field}')
    if validation_area:
        area=gpd.read_file(validation_area).to_crs(metric).geometry.union_all()
        truth=truth.loc[truth.intersects(area)].copy()
    truth['_gt_type']=truth[truth_type_field]
    widths={side:FinalWidths(gpd.read_file(p['width_segments']).to_crs(metric)) for side,p in zip(('before','after'),payloads)}
    gt,perturbation=perturb_truth(truth,widths,before_period,after_period,profile)
    corrected,correction=correct_changes(auto,gt)
    gt.to_file(output/'gt_assisted_audit.gpkg',layer='perturbed_gt',driver='GPKG')
    _write_json(output/'perturbation_audit.json',dict(profile=asdict(profile),objects=perturbation))
    _write_json(output/'correction_audit.json',correction)
    periods=[{**p,'period':period} for p,period in zip(payloads,(before_period,after_period))]
    pair=dict(frame=corrected,before_period=before_period,after_period=after_period)
    print(f'[Fast GT] Reconcile {len(corrected)} change intervals into separate period products',flush=True)
    updated=reconcile_periods(periods,[pair],output/'reconciled_periods')
    result=_publish_assisted_changes(pair['frame'],output,before_period,after_period)
    changes=[dict(**result,before_period=before_period,after_period=after_period)]
    print('[Fast GT] Build Auto temporal from original period roads',flush=True)
    auto_temporal=build_temporal_grid('pair',periods,[{**automatic_result,'before_period':before_period,'after_period':after_period}],output/'auto_temporal')
    print('[Fast GT] Build GT-assisted temporal from reconciled period roads',flush=True)
    temporal=build_temporal_grid('pair',updated,changes,output/'gt_assisted_temporal')
    summary=dict(execution_profile='fast',change_source='fast_automatic_gt_augmented',change_output_mode='fast_auto_plus_gt_assisted',
        automatic_result=False,ground_truth_used=True,ground_truth_derived=True,product_variant='gt_assisted',
        before_period=before_period,after_period=after_period,changes_feature_count=len(pair['frame']),
        automatic=automatic_result,gt_assisted_period_results=updated,auto_temporal=auto_temporal,gt_assisted_temporal=temporal,
        automatic_road_changes=automatic_result['road_changes'],detection_source='fast_automatic_change_detection',
        ground_truth_usage='post_auto_regular_correction_and_period_reconciliation',
        truth_path=str(Path(truth_path).resolve()),truth_type_field=truth_type_field,
        pipeline_order=['auto_extraction','auto_change','gt_correction','period_reconciliation','temporal_update'],
        **{f'{k}_feature_count':int(pair['frame'].change_typ.eq(k).sum()) for k in ('added','removed','widened','narrowed')})
    result.update(summary);result['summary']=str(output/'change_summary.json');_write_json(result['summary'],result)
    return result


def build_fast_temporal_outputs(manifest, job_root):
    """Regional downstream rebuild for full runs, reruns and edit refreshes."""
    from temporal_road_analysis import build_from_manifest,build_temporal_grid,clean_name
    from .fast_pipeline import _load_fast_period_result
    root=Path(job_root)
    periods=[]
    for entry in manifest.get('period_results',[]):
        if entry.get('status') in ('failed','stale'):continue
        payload=_load_fast_period_result(Path(entry['result'])) if entry.get('result') and Path(entry['result']).is_file() else {}
        periods.append({**payload,**entry})
    entries=[e for e in manifest.get('change_results',[]) if e.get('status') not in ('failed','stale')]
    auto_changes=[{**e.get('automatic',e),'grid':e.get('grid'),'before_period':e['before_period'],'after_period':e['after_period']} for e in entries]
    auto_manifest={**manifest,'period_results':periods,'change_results':auto_changes}
    auto_temporal=build_from_manifest(auto_manifest,root/'auto_temporal')
    for r in auto_temporal:r['product_variant']='auto'
    manifest['auto_temporal_results']=auto_temporal
    manifest['gt_assisted_period_results']=[];manifest['gt_assisted_change_results']=[]
    assisted_temporal=[]
    for grid in sorted({str(e.get('grid','validation')) for e in entries if e.get('automatic')}):
        local_periods=[p for p in periods if str(p.get('grid','validation'))==grid]
        local_changes=[e for e in entries if str(e.get('grid','validation'))==grid]
        if len(local_periods)<2:continue
        first=gpd.read_file(local_periods[0]['centerlines']);metric=first.estimate_utm_crs() if first.crs.is_geographic else first.crs
        pairs=[]
        for entry in local_changes:
            frame=(gpd.read_file(Path(entry['output'])/'gt_assisted_audit.gpkg',layer='changes').to_crs(metric)
                   if entry.get('automatic') else _auto_axes(entry,metric))
            pairs.append(dict(frame=frame,before_period=entry['before_period'],after_period=entry['after_period'],original=entry))
        target=root/'gt_assisted'/clean_name(grid)
        updated=reconcile_periods(local_periods,pairs,target/'periods')
        for p in updated:p['grid']=grid
        manifest['gt_assisted_period_results'].extend(updated)
        results=[]
        for pair in pairs:
            previous=pair['original'];directory=target/'changes'/f"{pair['before_period']}_to_{pair['after_period']}"
            published=_publish_assisted_changes(pair['frame'],directory,pair['before_period'],pair['after_period'])
            revised={**previous,**published,'grid':grid,'before_period':pair['before_period'],'after_period':pair['after_period'],
                'product_variant':'gt_assisted','ground_truth_used':True,'gt_assisted_period_results':updated,
                'automatic':previous.get('automatic',previous.copy())}
            revised['summary']=str(directory/'change_summary.json');_write_json(revised['summary'],revised)
            results.append(revised)
            if previous.get('automatic'):previous.update(revised)
        manifest['gt_assisted_change_results'].extend(results)
        temporal=build_temporal_grid(grid,updated,results,target/'temporal')
        temporal['product_variant']='gt_assisted';assisted_temporal.append(temporal)
    manifest['gt_assisted_temporal_results']=assisted_temporal
    manifest['temporal_status']='completed'
    assisted_grids={r['grid'] for r in assisted_temporal}
    return assisted_temporal+[r for r in auto_temporal if r['grid'] not in assisted_grids]


def complete_auto_pair_temporal(automatic, before_result, after_result, before_period, after_period):
    """Standalone Fast Auto pair commands also publish temporal products."""
    from .fast_pipeline import _load_fast_period_result
    from temporal_road_analysis import build_temporal_grid
    periods=[{**_load_fast_period_result(p),'period':period}
             for p,period in zip((before_result,after_result),(before_period,after_period))]
    entry={**automatic,'before_period':before_period,'after_period':after_period}
    temporal=build_temporal_grid('pair',periods,[entry],Path(automatic['output'])/'auto_temporal')
    result={**automatic,'auto_temporal':temporal}
    summary=Path(str(automatic.get('summary') or Path(automatic['output'])/'change_summary.json'))
    if summary.is_file():
        payload=json.loads(summary.read_text(encoding='utf-8'));payload['auto_temporal']=temporal;_write_json(summary,payload)
    return result
