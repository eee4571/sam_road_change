"""Build road surfaces around immutable final centerlines and observed widths."""
from collections import defaultdict

import geopandas as gpd
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from shapely import STRtree, make_valid
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


def _polygonal(geometry):
    if geometry.geom_type in ('Polygon','MultiPolygon'):
        return geometry
    return unary_union([_polygonal(g) for g in getattr(geometry,'geoms',[])])


def _smooth_widths(stations, raw, reference):
    valid = np.isfinite(raw) & (raw > 1.) & (raw < 80.)
    values = np.interp(stations,stations[valid],raw[valid]) if valid.any() else np.full(len(stations),reference)
    step = max(float(np.median(np.diff(stations))),.5)
    window = max(3,int(25/step)//2*2+1)
    local = median_filter(values,size=window,mode='nearest')
    deviations = median_filter(np.abs(values-local),size=window,mode='nearest')
    outlier = np.abs(values-local)>np.maximum(2.5,3*1.4826*deviations)
    values[outlier] = local[outlier]
    values = gaussian_filter1d(median_filter(values,size=max(3,window//2*2+1),mode='nearest'),
                               sigma=max(1.,18/step),mode='nearest')
    # Limit full-width change to 8 cm per longitudinal metre.
    for _ in range(2):
        for i in range(1,len(values)):
            limit=.08*(stations[i]-stations[i-1])
            values[i]=np.clip(values[i],values[i-1]-limit,values[i-1]+limit)
        for i in range(len(values)-2,-1,-1):
            limit=.08*(stations[i+1]-stations[i])
            values[i]=np.clip(values[i],values[i+1]-limit,values[i+1]+limit)
    return values,int(outlier.sum())


def _swept_surface(line, stations, widths):
    original=np.asarray(line.coords)
    original_s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(original,axis=0),axis=1))]
    s=np.unique(np.r_[stations,original_s])
    points=np.asarray([line.interpolate(v).coords[0] for v in s])
    radii=np.interp(s,stations,widths)/2
    pieces=[Point(p).buffer(float(r),quad_segs=10) for p,r in zip(points,radii)]
    for p,q,r1,r2 in zip(points[:-1],points[1:],radii[:-1],radii[1:]):
        d=q-p
        if np.linalg.norm(d)<1e-8: continue
        n=np.array([-d[1],d[0]])/np.linalg.norm(d)
        pieces.append(Polygon([p+n*r1,q+n*r2,q-n*r2,p-n*r1]))
    return make_valid(unary_union(pieces))


def build_final_road_surface(centerlines, width_segments, *, step_m=3.):
    """Return per-track polygons, local junction patches, and smoothed samples.

    Centerlines must be projected in metres and already noded. Neither the input
    centerlines nor the segmentation surface is edited or used as output geometry.
    """
    if centerlines.crs is None or centerlines.crs.is_geographic:
        raise ValueError('Use a projected metric CRS')
    lines=list(centerlines.geometry)
    evidence=width_segments.to_crs(centerlines.crs)
    geometries=list(evidence.geometry)
    tree=STRtree(geometries)
    measurements=np.asarray(evidence.width_m,dtype=float)
    profiles=[]; sample_rows=[]; rejected=0
    for road_id,(line,(_,row)) in enumerate(zip(lines,centerlines.iterrows())):
        stations=np.linspace(0,line.length,max(2,int(np.ceil(line.length/step_m))+1))
        raw=np.full(len(stations),np.nan)
        reference=float(row.width_m)
        for i,s in enumerate(stations):
            point=line.interpolate(s)
            tangent=np.asarray(line.interpolate(min(line.length,s+6)).coords[0])-np.asarray(line.interpolate(max(0,s-6)).coords[0])
            tangent/=max(np.linalg.norm(tangent),1e-9)
            candidates=[]
            for idx in tree.query(point.buffer(3.)):
                geometry=geometries[int(idx)]
                if geometry.geom_type!='LineString': continue
                t=geometry.project(point)
                d=np.asarray(geometry.interpolate(min(geometry.length,t+6)).coords[0])-np.asarray(geometry.interpolate(max(0,t-6)).coords[0])
                if abs(d@tangent)/max(np.linalg.norm(d),1e-9)<.9: continue
                distance=geometry.distance(point)
                if distance<=3. and np.isfinite(measurements[idx]) and measurements[idx]>1:
                    candidates.append((distance,measurements[idx]))
            if candidates:
                candidates.sort()
                raw[i]=np.median([w for distance,w in candidates if distance<=candidates[0][0]+.4])
        values,n=_smooth_widths(stations,raw,reference)
        rejected+=n
        profiles.append([stations,raw,values])
    # Match only opposite tangent arms for width continuity across noded junctions.
    endpoints=defaultdict(list)
    for i,line in enumerate(lines):
        for start in (True,False):
            position=0 if start else line.length
            point=np.asarray(line.interpolate(position).coords[0])
            inward=np.asarray(line.interpolate(min(15,line.length) if start else max(0,line.length-15)).coords[0])-point
            inward/=max(np.linalg.norm(inward),1e-9)
            endpoints[tuple(np.round(point,4))].append((i,start,inward))
    targets={}
    for arms in endpoints.values():
        candidates=sorted(((float(a[2]@b[2]),a,b) for k,a in enumerate(arms) for b in arms[k+1:]
                          if a[0]!=b[0] and float(a[2]@b[2])<-.9),key=lambda item:item[0])
        used=set()
        for _,a,b in candidates:
            if a[0] in used or b[0] in used: continue
            av=profiles[a[0]][2][0 if a[1] else -1]; bv=profiles[b[0]][2][0 if b[1] else -1]
            if max(av,bv)>1.7*min(av,bv): continue
            value=(av+bv)/2
            targets[(a[0],a[1])]=value; targets[(b[0],b[1])]=value
            used.update((a[0],b[0]))
    tracks=[]
    for i,(line,profile) in enumerate(zip(lines,profiles)):
        stations,raw,values=profile
        for start in (True,False):
            target=targets.get((i,start))
            if target is None: continue
            distance=stations if start else line.length-stations
            weight=np.maximum(0,1-distance/35.)**2
            values=values+weight*(target-values[0 if start else -1])
        profiles[i][2]=values
        polygon=_swept_surface(line,stations,values)
        tracks.append(dict(road_id=i,kind='road_track',width_min=float(values.min()),
                           width_max=float(values.max()),width_m=float(np.mean(values)),geometry=polygon))
        sample_rows.extend(dict(road_id=i,station_m=float(s),raw_width=float(w),width_m=float(v),
                                geometry=line.interpolate(s)) for s,w,v in zip(stations,raw,values))
    # Local patches only. Parallel carriageways outside these disks stay separate.
    zones=[]
    for point,arms in endpoints.items():
        if len(arms)<3: continue
        widths=[profiles[i][2][0 if start else -1] for i,start,_ in arms]
        radius=min(28.,max(7.,1.15*max(widths)))
        zones.append(Point(point).buffer(radius,quad_segs=16))
    patches=[]
    track_tree=STRtree([row['geometry'] for row in tracks])
    merged_zones=unary_union(zones)
    parts=list(merged_zones.geoms) if hasattr(merged_zones,'geoms') else [merged_zones]
    for zone in parts:
        if zone.is_empty: continue
        nearby=unary_union([tracks[int(i)]['geometry'].intersection(zone) for i in track_tree.query(zone)])
        if nearby.is_empty: continue
        patch=nearby.buffer(2.,quad_segs=10).buffer(-2.,quad_segs=10).union(nearby).intersection(zone)
        patches.append(dict(road_id=-1,kind='junction',width_min=np.nan,width_max=np.nan,width_m=np.nan,geometry=make_valid(patch)))
    for row in tracks:
        row['geometry']=make_valid(row['geometry'].difference(merged_zones)) if zones else row['geometry']
    for row in tracks+patches:
        row['geometry']=_polygonal(row['geometry'])
    surfaces=gpd.GeoDataFrame([r for r in tracks+patches if not r['geometry'].is_empty],crs=centerlines.crs)
    samples=gpd.GeoDataFrame(sample_rows,crs=centerlines.crs)
    stats=dict(centerline_count=len(lines),surface_feature_count=len(surfaces),
               junction_patch_count=len(patches),width_sample_count=len(samples),filtered_sample_count=rejected,
               centerline_modified=False,width_policy='existing_segments_robust_local_filter_low_frequency_smoothing')
    return surfaces,samples,stats
