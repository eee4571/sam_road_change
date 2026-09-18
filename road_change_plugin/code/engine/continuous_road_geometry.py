"""Continuous road boundary construction, shared by period and change exports.

Input axes and longitudinal ranges are authoritative. Flat caps belong only to
open ends of a continuous chain, never to every width station/source feature.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely import union_all
from shapely.geometry import LineString, Point, Polygon, MultiPoint
from shapely.strtree import STRtree


def network_surface(profiles, node_tolerance=.5, _junctions=None, *, quality=None, audit=None, evidence=None,
                    _feature_ids=None):
    """Join compatible incident axes before offsetting, then dissolve boundaries.

    Profiles are (LineString, longitudinal stations, widths). Nearby parallel
    tracks are not snapped: only coincident endpoint nodes participate. A node
    joins degree-2 chains. True junctions retain their branches and share a
    local boundary footprint; a loop is never concatenated onto a trunk axis.
    Final period products supply saved sample quality for chain-scale width
    regularization. Approved change intervals omit it: their widths and extent
    remain authoritative, not reclassified by a cartographic operation.
    """
    from .auto_change_geometry import corridor, _clean_overlay
    from .auto_change_assembly import polygonal
    # Tiny negative roundoff in a remapped station means "from the end" to
    # Shapely.interpolate; clamp before sampling to avoid doubling the axis.
    selected=[i for i,(a,_,_) in enumerate(profiles) if a.length>1e-6]
    _feature_ids=[i if _feature_ids is None else _feature_ids[i] for i in selected]
    if quality is not None:quality=[np.asarray(quality[i],float) for i in selected]
    profiles=[(profiles[i][0],np.clip(np.asarray(profiles[i][1],float),0.,profiles[i][0].length),
               np.asarray(profiles[i][2],float)) for i in selected]
    if not profiles:return Polygon()
    from .road_axis_quality import require_safe_axis, axis_quality
    for axis,_,widths in profiles:require_safe_axis(axis,float(np.median(widths)))
    points=[Point(a.coords[e]) for a,_,_ in profiles for e in (0,-1)]
    tree=STRtree(points);parent=list(range(len(points)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,p in enumerate(points):
        for raw in tree.query(p,predicate='dwithin',distance=node_tolerance):
            j=int(raw)
            if i//2!=j//2:parent[root(j)]=root(i)
    nodes={}
    for i in range(len(points)):nodes.setdefault(root(i),[]).append(i)
    if _junctions is None:
        _junctions=[Point(np.mean([points[i].coords[0] for i in ends],axis=0))
                    for ends in nodes.values() if len(ends)>2]
    junction_tree=STRtree(_junctions)
    # A terminal arm shorter than its own junction footprint cannot define an
    # independent road rectangle. Keep internal connectors and isolated roads;
    # only absorb short leaves attached to an established longer road chain.
    def reaches_long_road(start,excluded):
        pending=[start];seen={excluded};total=0.
        while pending:
            node=pending.pop()
            for end in nodes[node]:
                k=end//2
                if k in seen:continue
                seen.add(k)
                axis,_,width=profiles[k]
                total+=axis.length
                if axis.length>=2.5*float(np.median(width)):return True
                pending.append(root(end^1))
        return total>=3.*float(np.median(profiles[excluded][2]))
    absorbed=set()
    for k,(axis,_,width) in enumerate(profiles):
        if axis.length>1.25*float(np.median(width)):continue
        for end in (2*k,2*k+1):
            if len(nodes[root(end)])>=3 and len(nodes[root(end^1)])==1 and reaches_long_road(root(end),k):
                absorbed.add(k);break
    for ends in nodes.values():
        if ends and all(e//2 in absorbed for e in ends):
            # An entirely short junction still needs a continuous backbone.
            directions={e:np.asarray(points[e^1].coords[0])-np.asarray(points[e].coords[0]) for e in ends}
            choices=[]
            for n,e in enumerate(ends):
                for f in ends[n+1:]:
                    d,t=directions[e],directions[f]
                    choices.append((-float(d@t)/max(np.linalg.norm(d)*np.linalg.norm(t),1e-9),e,f))
            if choices:
                _,e,f=max(choices);absorbed.discard(e//2);absorbed.discard(f//2)
    if absorbed:
        junction_caps=[]
        for ends in nodes.values():
            leaves=[e for e in ends if e//2 in absorbed]
            retained=[e for e in ends if e//2 not in absorbed]
            if not leaves or not retained:continue
            end=max(retained,key=lambda e:profiles[e//2][0].length)
            axis,ss,ww=profiles[end//2]
            xy=np.asarray(axis.coords)
            tip=xy[0] if end%2==0 else xy[-1]
            inside=np.asarray(axis.interpolate(min(6.,axis.length)).coords[0] if end%2==0
                              else axis.interpolate(max(0.,axis.length-6.)).coords[0])
            tangent=tip-inside;tangent/=max(np.linalg.norm(tangent),1e-9)
            reach=[]
            for e in leaves:
                delta=np.asarray(points[e^1].coords[0])-tip
                direction=delta/max(np.linalg.norm(delta),1e-9)
                radius=float(np.median(profiles[e//2][2]))/2
                reach.append(float(delta@tangent)+radius*np.sqrt(max(0.,1.-float(direction@tangent)**2)))
            extension=max([0.]+reach)
            if extension<=0:continue
            # Extend the surface footprint, never move a shared axis endpoint.
            # Moving it breaks degree-2 pairing and can introduce a new fold
            # when a retained axis already curves near the junction.
            endpoint=tip+tangent*extension
            width=float(ww[0 if end%2==0 else -1])
            junction_caps.append(corridor(LineString([tip,endpoint]),[0.,extension],[width,width]))
        surface=network_surface([p for k,p in enumerate(profiles) if k not in absorbed],node_tolerance,_junctions,
            quality=[q for k,q in enumerate(quality) if k not in absorbed] if quality is not None else None,
            audit=audit,evidence=evidence,_feature_ids=[v for k,v in enumerate(_feature_ids) if k not in absorbed])
        return _clean_overlay(polygonal(union_all([surface,*junction_caps])))
    pair={};centers={};junction_polygons=[];footprints={};junction_ends=[]
    for ends in nodes.values():
        center=np.mean([points[i].coords[0] for i in ends],axis=0)
        for i in ends:centers[i]=center
        protected=len(junction_tree.query(Point(center),predicate='dwithin',distance=node_tolerance))>0
        if len(ends)>=2:
            caps=[]
            for i in ends:
                a,_,w=profiles[i//2];e=i%2
                q=np.asarray(a.interpolate(min(2.,a.length) if e==0 else max(0.,a.length-2.)).coords[0])
                d=q-center;d/=max(np.linalg.norm(d),1e-9)
                n=np.array([-d[1],d[0]])*w[0 if e==0 else -1]/2
                caps.extend([center+n,center-n])
            footprint=MultiPoint(caps).convex_hull
            footprints[root(ends[0])]=footprint
            if footprint.area>0 and (len(ends)>2 or protected) and quality is None:junction_polygons.append(footprint)
            if quality is not None and len(ends)>2:junction_ends.append(ends)
        if len(ends)!=2 or protected:continue
        choices=[]
        for n,i in enumerate(ends):
            a,_,w=profiles[i//2];e=i%2
            point=np.asarray(a.interpolate(min(a.length, max(1.,min(6.,a.length/3))) if e==0 else max(0.,a.length-max(1.,min(6.,a.length/3)))).coords[0])
            d=point-center;d/=max(np.linalg.norm(d),1e-9)
            for j in ends[n+1:]:
                if i//2==j//2:continue
                # Proximity is enough to share a surface footprint, but not to
                # move authoritative axis vertices or create a tiny zigzag.
                if points[i].distance(points[j])>1e-6:continue
                b,_,v=profiles[j//2];f=j%2
                q=np.asarray(b.interpolate(min(b.length,max(1.,min(6.,b.length/3))) if f==0 else max(0.,b.length-max(1.,min(6.,b.length/3)))).coords[0])
                t=q-center;t/=max(np.linalg.norm(t),1e-9)
                alignment=-float(d@t)
                threshold=-.8 if len(ends)==2 else .45
                if alignment>=threshold:choices.append((alignment,-abs(w[0 if e==0 else -1]-v[0 if f==0 else -1]),-i,-j))
        for _,__,ni,nj in sorted(choices,reverse=True):
            i,j=-ni,-nj
            if i not in pair and j not in pair:pair[i]=j;pair[j]=i
    visited=set();polygons=list(junction_polygons);endpoint_widths={}
    def emit(coords,values,reliability,ends):
        keep=np.r_[True,np.linalg.norm(np.diff(np.asarray(coords),axis=0),axis=1)>1e-6]
        coords=np.asarray(coords)[keep];values=np.asarray(values)[keep]
        reliability=np.asarray(reliability)[keep]
        if len(coords)<2:return
        axis=LineString(coords)
        ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(coords,axis=0),axis=1))]
        if axis.length<=1e-6:return
        sampling=np.linspace(0,axis.length,max(2,int(np.ceil(axis.length/2))+1))
        ww=np.interp(sampling,ss,values)
        if quality is not None:
            from .road_surface_quality import stable_width
            qq=np.interp(sampling,ss,reliability)>=.99
            ww,report=stable_width(sampling,ww,qq)
            report.update(length_m=float(axis.length),source_features=sorted({_feature_ids[e//2] for e in ends}))
            if audit is not None:audit.append(report)
            endpoint_widths[ends[0]]=float(ww[0]);endpoint_widths[ends[-1]]=float(ww[-1])
            # Remove only sub-pixel collinear chatter; pin the chain endpoints,
            # retain real bends and verify that simplification adds no crossings.
            from .road_surface_quality import surface_axis
            simplified=surface_axis(axis,float(np.median(ww)),evidence)
            if simplified.is_simple and simplified.hausdorff_distance(axis)<=.8:
                from shapely.ops import substring
                interior=substring(simplified,min(.5,simplified.length/3),max(simplified.length-.5,simplified.length*2/3))
                original_interior=substring(axis,min(.5,axis.length/3),max(axis.length-.5,axis.length*2/3))
                nearby=axis_tree.query(interior.union(original_interior))
                own={e//2 for e in ends}
                if all(int(j) in own or not (interior.intersects(profiles[int(j)][0]) or
                    original_interior.intersects(profiles[int(j)][0])) for j in nearby):
                    report['render_axis_shift_m']=float(simplified.hausdorff_distance(axis))
                    sampling=sampling* simplified.length/axis.length
                    axis=simplified
        elif len(ww)>3:ww=gaussian_filter1d(ww,1.,mode='nearest')
        require_safe_axis(axis,float(np.median(ww)))
        polygons.append(corridor(axis,sampling,ww))
    starts=[i for i in range(len(points)) if i not in pair]+list(range(len(points)))
    axis_tree=STRtree([p[0] for p in profiles])
    for start in starts:
        if start//2 in visited:continue
        current=start;coords=[];values=[];reliability=[];chain_ends=[]
        while current//2 not in visited:
            k=current//2;visited.add(k);axis,stations,widths=profiles[k]
            original=np.asarray(axis.coords)
            ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(original,axis=0),axis=1))]
            samples=np.unique(np.r_[stations,ss])
            xy=np.array([axis.interpolate(s).coords[0] for s in samples]);ww=np.interp(samples,stations,widths)
            qq=np.interp(samples,stations,quality[k]) if quality is not None else np.ones(len(samples))
            if current%2:xy=xy[::-1];ww=ww[::-1];qq=qq[::-1]
            if coords:
                joined=LineString(coords+list(xy[1:]))
                if not joined.is_simple or axis_quality(joined,float(np.median(values+list(ww))))['abnormal']:
                    # Two individually sound network paths can cross/overlap.
                    # Keep those paths separate for offsets, then union their
                    # surfaces; never turn their topology into a folded axis.
                    emit(coords,values,reliability,chain_ends);coords=[];values=[];reliability=[];chain_ends=[]
                    footprint=footprints.get(root(current))
                    if footprint is not None and footprint.area>0:polygons.append(footprint)
            if coords:
                values[-1]=(values[-1]+ww[0])/2
                coords.extend(xy[1:]);values.extend(ww[1:])
                reliability[-1]=min(reliability[-1],qq[0]);reliability.extend(qq[1:])
            else:coords.extend(xy);values.extend(ww);reliability.extend(qq)
            chain_ends.extend([current,current^1])
            following=pair.get(current^1)
            if following is None:break
            current=following
        emit(coords,values,reliability,chain_ends)
    if quality is not None:
        from .road_surface_quality import junction_footprint
        for ends in junction_ends:
            arms=[]
            for e in ends:
                a,_,w=profiles[e//2]
                arms.append((LineString(list(a.coords)[::-1]) if e%2 else a,
                             endpoint_widths.get(e,float(w[0 if e%2==0 else -1]))))
            footprint=junction_footprint(arms)
            if not footprint.is_empty:polygons.append(footprint)
            if audit is not None:audit.append(dict(method='junction_portal_footprint' if not footprint.is_empty
                else 'junction_existing_union',degree=len(ends),area_m2=float(footprint.area)))
    result=_clean_overlay(polygonal(union_all(polygons)))
    parts=[result] if result.geom_type=='Polygon' else list(result.geoms)
    axes_tree=STRtree([p[0] for p in profiles])
    cleaned=[]
    for p in parts:
        holes=[]
        for ring in p.interiors:
            hole=Polygon(ring)
            width=float(np.median(profiles[int(axes_tree.nearest(hole.representative_point()))][2]))
            x0,y0,x1,y1=hole.bounds
            if hole.area>=1. and not (hole.area<.5*width**2 and np.hypot(x1-x0,y1-y0)<1.5*width):holes.append(ring)
        cleaned.append(Polygon(p.exterior,holes))
    return polygonal(union_all(cleaned))


def change_surfaces(frame):
    """Render/dissolve each change class; preserve interval metadata in audit."""
    import geopandas as gpd
    import pandas as pd
    from shapely import from_wkt
    rows=[]
    from .gt_road_geometry import road_profile
    work=frame.copy()
    work['_full_road']=work.get('gt_geometry_role',pd.Series(index=work.index,dtype=str)).eq('full_road_change')
    for (kind,full_road),group in work.groupby(['change_typ','_full_road'],sort=False):
        before=[];after=[]
        for row in group.itertuples():
            geometry=from_wkt(row.axis_wkt)
            axes=[geometry] if geometry.geom_type=='LineString' else list(geometry.geoms)
            for axis in axes:
                record=row._asdict()
                if row.width_bef>0:before.append((axis,*road_profile(record,axis,row.width_bef)))
                if row.width_aft>0:after.append((axis,*road_profile(record,axis,row.width_aft)))
        b=network_surface(before);a=network_surface(after)
        geometry=a if kind=='added' or (full_road and kind=='widened') else b if kind=='removed' or (full_road and kind=='narrowed') else a.difference(b) if kind=='widened' else b.difference(a)
        from .auto_change_assembly import polygonal
        geometry=polygonal(geometry)
        parts=[geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)
        for part in parts:
            if part.is_empty:continue
            local=group.loc[group.geometry.intersects(part)]
            if local.empty:local=group
            rows.append(dict(change_id=f'CH{len(rows)+1:06d}',change_typ=kind,
                width_bef=float(local.width_bef.median()),width_aft=float(local.width_aft.median()),
                width_diff=float((local.width_aft-local.width_bef).median()),geometry=part))
    return gpd.GeoDataFrame(rows,geometry='geometry',crs=frame.crs) if rows else frame.iloc[:0].copy()
