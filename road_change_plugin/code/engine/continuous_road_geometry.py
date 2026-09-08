"""Continuous road boundary construction, shared by period and change exports.

Input axes and longitudinal ranges are authoritative. Flat caps belong only to
open ends of a continuous chain, never to every width station/source feature.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely import union_all
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree


def network_surface(profiles, node_tolerance=.5):
    """Join compatible incident axes before offsetting, then dissolve boundaries.

    Profiles are (LineString, longitudinal stations, widths). Nearby parallel
    tracks are not snapped: only coincident endpoint nodes participate. A node
    can pair its most continuous incident directions even at a junction.
    """
    from .auto_change_geometry import corridor, _clean_overlay
    from .auto_change_assembly import polygonal
    profiles=[(a,np.asarray(s,float),np.asarray(w,float)) for a,s,w in profiles if a.length>1e-6]
    if not profiles:return Polygon()
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
        rebuilt=list(profiles)
        for ends in nodes.values():
            leaves=[e for e in ends if e//2 in absorbed]
            retained=[e for e in ends if e//2 not in absorbed]
            if not leaves or not retained:continue
            end=max(retained,key=lambda e:profiles[e//2][0].length)
            axis,ss,ww=rebuilt[end//2]
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
            # Preserve the supported longitudinal reach with one continuation
            # cap; removing transverse patch arms must not shorten a junction.
            endpoint=tip+tangent*extension
            if end%2==0:
                rebuilt[end//2]=(LineString(np.vstack([endpoint,xy])),np.r_[0.,ss+extension],np.r_[ww[0],ww])
            else:
                rebuilt[end//2]=(LineString(np.vstack([xy,endpoint])),np.r_[ss,axis.length+extension],np.r_[ww,ww[-1]])
        return network_surface([p for k,p in enumerate(rebuilt) if k not in absorbed],node_tolerance)
    pair={};centers={}
    for ends in nodes.values():
        center=np.mean([points[i].coords[0] for i in ends],axis=0)
        for i in ends:centers[i]=center
        choices=[]
        for n,i in enumerate(ends):
            a,_,w=profiles[i//2];e=i%2
            point=np.asarray(a.interpolate(min(a.length, max(1.,min(6.,a.length/3))) if e==0 else max(0.,a.length-max(1.,min(6.,a.length/3)))).coords[0])
            d=point-center;d/=max(np.linalg.norm(d),1e-9)
            for j in ends[n+1:]:
                if i//2==j//2:continue
                b,_,v=profiles[j//2];f=j%2
                q=np.asarray(b.interpolate(min(b.length,max(1.,min(6.,b.length/3))) if f==0 else max(0.,b.length-max(1.,min(6.,b.length/3)))).coords[0])
                t=q-center;t/=max(np.linalg.norm(t),1e-9)
                alignment=-float(d@t)
                threshold=-.8 if len(ends)==2 else .45
                if alignment>=threshold:choices.append((alignment,-abs(w[0 if e==0 else -1]-v[0 if f==0 else -1]),-i,-j))
        for _,__,ni,nj in sorted(choices,reverse=True):
            i,j=-ni,-nj
            if i not in pair and j not in pair:pair[i]=j;pair[j]=i
    visited=set();polygons=[]
    starts=[i for i in range(len(points)) if i not in pair]+list(range(len(points)))
    for start in starts:
        if start//2 in visited:continue
        current=start;coords=[];values=[]
        while current//2 not in visited:
            k=current//2;visited.add(k);axis,stations,widths=profiles[k]
            original=np.asarray(axis.coords)
            ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(original,axis=0),axis=1))]
            samples=np.unique(np.r_[stations,ss])
            xy=np.array([axis.interpolate(s).coords[0] for s in samples]);ww=np.interp(samples,stations,widths)
            if current%2:xy=xy[::-1];ww=ww[::-1]
            xy[0]=centers[current];xy[-1]=centers[current^1]
            if coords:
                values[-1]=(values[-1]+ww[0])/2
                coords.extend(xy[1:]);values.extend(ww[1:])
            else:coords.extend(xy);values.extend(ww)
            following=pair.get(current^1)
            if following is None:break
            current=following
        keep=np.r_[True,np.linalg.norm(np.diff(np.asarray(coords),axis=0),axis=1)>1e-6]
        coords=np.asarray(coords)[keep];values=np.asarray(values)[keep]
        if len(coords)<2:continue
        axis=LineString(coords)
        ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(np.array(coords),axis=0),axis=1))]
        if axis.length<=1e-6:continue
        # Smooth across former feature boundaries with a metric-length window.
        sampling=np.linspace(0,axis.length,max(2,int(np.ceil(axis.length/2))+1))
        ww=np.interp(sampling,ss,values)
        if len(ww)>3:ww=gaussian_filter1d(ww,1.,mode='nearest')
        polygons.append(corridor(axis,sampling,ww))
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
    from shapely import from_wkt
    rows=[]
    for kind,group in frame.groupby('change_typ',sort=False):
        before=[];after=[]
        for row in group.itertuples():
            geometry=from_wkt(row.axis_wkt)
            axes=[geometry] if geometry.geom_type=='LineString' else list(geometry.geoms)
            for axis in axes:
                s=[0.,axis.length]
                if row.width_bef>0:before.append((axis,s,[row.width_bef]*2))
                if row.width_aft>0:after.append((axis,s,[row.width_aft]*2))
        b=network_surface(before);a=network_surface(after)
        geometry=a if kind=='added' else b if kind=='removed' else a.difference(b) if kind=='widened' else b.difference(a)
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
