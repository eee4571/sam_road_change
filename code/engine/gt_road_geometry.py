"""Regular GT road fits and profile rendering (posterior use only)."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely import from_wkt
from shapely.geometry import LineString, Polygon
from .auto_change_geometry import corridor, _stations


def compact_road_fit(geometry):
    """Fit transverse section centres; never return the input polygon as output."""
    if geometry.geom_type != 'Polygon' or any(Polygon(r).area > 1 for r in geometry.interiors):
        return None
    rect=np.asarray(geometry.minimum_rotated_rectangle.exterior.coords)[:4]
    edges=np.roll(rect,-1,axis=0)-rect
    lengths=np.linalg.norm(edges,axis=1)
    if lengths.max()/max(lengths.min(),1e-9)>5:
        return None
    direction=edges[int(np.argmax(lengths))]/lengths.max(); normal=np.array([-direction[1],direction[0]])
    origin=np.asarray(geometry.centroid.coords[0]);coords=np.asarray(geometry.exterior.coords)-origin
    projected=coords@direction; lo,hi=projected.min(),projected.max()
    step=min(2.,max(.5,(hi-lo)/30))
    stations=np.linspace(lo+.01,hi-.01,max(3,int(np.ceil((hi-lo)/step))+1))
    centers=[];widths=[]
    radius=lengths.max()*2
    for s in stations:
        c=origin+s*direction
        hit=LineString([c-radius*normal,c+radius*normal]).intersection(geometry)
        pieces=[hit] if hit.geom_type=='LineString' else [g for g in getattr(hit,'geoms',()) if g.geom_type=='LineString']
        if not pieces:return None
        if len(pieces)>1 and sum(p.length for p in pieces)-max(p.length for p in pieces) > .5:return None
        part=max(pieces,key=lambda p:p.length);centers.append(part.interpolate(.5,normalized=True).coords[0]);widths.append(part.length)
    centers=np.asarray(centers);widths=np.asarray(widths)
    # Isolated corner tips have near-zero cross sections: extrapolate the fitted
    # interior to the end planes, preserving square caps rather than spikes.
    good=np.flatnonzero(widths>=.5*np.median(widths))
    if len(good)<2:return None
    for i in range(len(widths)):
        if i<good[0] or i>good[-1]:
            j=good[0] if i<good[0] else good[-1]
            centers[i]=centers[j]+(stations[i]-stations[j])*direction; widths[i]=widths[j]
    centers=gaussian_filter1d(centers,1.,axis=0,mode='nearest')
    centers[0]+=direction*(lo-(centers[0]-origin)@direction)
    centers[-1]+=direction*(hi-(centers[-1]-origin)@direction)
    widths=gaussian_filter1d(widths,1.,mode='nearest')
    axis=LineString(centers)
    return dict(axis=axis,start_external=True,end_external=True,
                fitted_widths=widths.tolist(),fitted_stations=np.linspace(0,1,len(widths)).tolist())


def road_profile(row, axis, width):
    raw=row.get('gt_width_profile')
    if not isinstance(raw,str) or not raw:return np.array([0.,axis.length]),np.array([width,width])
    profile=json.loads(raw);reference=from_wkt(profile['axis'])
    stations=_stations(axis)
    positions=np.array([reference.project(axis.interpolate(s))/max(reference.length,1e-9) for s in stations])
    values=np.interp(positions,profile['stations'],profile['widths'])
    values*=width/max(profile['reference_width'],1e-9)
    return stations,values


def record_polygon(row, axis=None):
    axis=from_wkt(row['axis_wkt']) if axis is None else axis
    b,a=float(row['width_bef']),float(row['width_aft'])
    kind=row['change_typ']
    def surface(w):
        s,v=road_profile(row,axis,w)
        return corridor(axis,s,v)
    if kind=='added':return surface(a)
    if kind=='removed':return surface(b)
    if row.get('gt_geometry_role')=='full_road_change':return surface(max(a,b))
    return surface(a).difference(surface(b)) if a>b else surface(b).difference(surface(a))

def matched_final_roads(predicted, audit_path):
    """Select associated final road intervals before the unchanged metric formula.

    The audit establishes identity; geometry is taken from Final Changes. No GT
    polygon is used for clipping, and missing truth roads remain in the metric's
    truth denominator.
    """
    import geopandas as gpd
    from shapely import make_valid, union_all
    audit=gpd.read_file(audit_path,layer='changes')
    output_crs=predicted.crs
    predicted=predicted.to_crs(audit.crs)
    rows=[]
    if 'truth_id' not in audit:return predicted.iloc[:0].copy()
    selected=audit.loc[audit.truth_id.notna() & audit.truth_id.ne('') & audit.change_typ.isin(['added','removed'])]
    for (truth_id,kind),group in selected.groupby(['truth_id','change_typ']):
        final=predicted.loc[predicted.change_typ.eq(kind)].geometry.union_all()
        # Keep each associated road footprint, excluding attached Auto branches.
        domains=[]
        for record in group.to_dict('records'):
            axis=from_wkt(record['axis_wkt'])
            width=max(float(record['width_bef']),float(record['width_aft']))
            _,values=road_profile(record,axis,width)
            # Cut longitudinal extent only; generous lateral space retains real
            # offset/width errors in Final geometry for the unchanged metric.
            domains.append(axis.buffer(float(np.max(values))/2+8.,cap_style='flat'))
        association=union_all(domains)
        geometry=make_valid(final).intersection(make_valid(association))
        if not geometry.is_empty:rows.append(dict(change_typ=kind,geometry=geometry))
    result=gpd.GeoDataFrame(rows,geometry='geometry',crs=predicted.crs) if rows else predicted.iloc[:0].copy()
    return result.to_crs(output_crs)
