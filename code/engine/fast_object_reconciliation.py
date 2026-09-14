"""Candidate-only, deterministic reconciliation; no imagery or point matching."""
from collections import Counter
import time
import numpy as np
from shapely import from_wkt, get_coordinates, union_all, hausdorff_distance
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree


def _direction(line):
    xy=get_coordinates(line)
    # Endpoint direction is orientation independent after absolute dot products.
    direction=xy[-1]-xy[0]
    norm=np.linalg.norm(direction)
    return direction/norm if norm>1e-8 else np.zeros(2)


def _groups(records,kind):
    side='before' if kind=='removed' else 'after'
    ids=[i for i,r in enumerate(records) if r['change_typ']==kind and
         (r.get('v2_publish',False) or (r.get(f'{side}_valid_ratio',0.) and r.get(f'{side}_surface_ratio',0.)>=.55))]
    lines=[from_wkt(records[i]['axis_wkt']) for i in ids]
    ends=[Point(g.coords[j]) for g in lines for j in (0,-1)]
    tree=STRtree(ends);parent=list(range(len(ids)));choices={}
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i,p in enumerate(ends):
        nearby=[int(j) for j in tree.query(p,predicate='dwithin',distance=1.5)
                if j//2!=i//2 and abs(_direction(lines[i//2])@_direction(lines[j//2]))>=.985]
        if len(nearby)==1:choices[i]=nearby[0]
    for i,j in choices.items():
        if choices.get(j)==i:
            a,b=root(i//2),root(j//2)
            parent[max(a,b)]=min(a,b)
    groups={}
    for i in range(len(ids)):groups.setdefault(root(i),[]).append(i)
    result=[]
    for indexes in groups.values():
        members=[ids[i] for i in indexes];geometry=union_all([lines[i] for i in indexes])
        xy=get_coordinates(geometry);weights=np.array([lines[i].length for i in indexes])
        direction=_direction(lines[max(indexes,key=lambda i:lines[i].length)])
        lengths=xy@direction
        if np.ptp(lengths)<1.:continue
        width=float(np.average([max(records[i]['width_bef'],records[i]['width_aft']) for i in members],weights=weights))
        result.append(dict(members=members,geometry=geometry,xy=xy,direction=direction,width=width,
                           length=geometry.length,junction=any(records[i].get('junction',False) for i in members)))
    return result


def _pair(a,b,radius):
    cosine=abs(a['direction']@b['direction'])
    complex_case=a['junction'] or b['junction']
    if cosine < (.995 if complex_case else .985):return None
    ratio=min(a['length'],b['length'])/max(a['length'],b['length'])
    if ratio < (.9 if complex_case else .8):return None
    direction=a['direction'];pa=a['xy']@direction;pb=b['xy']@direction
    overlap=max(0.,min(pa.max(),pb.max())-max(pa.min(),pb.min()))
    coverage=overlap/max(np.ptp(pa),np.ptp(pb),1e-9)
    if coverage < (.95 if complex_case else .88):return None
    # Compare shape within the common longitudinal span. Endpoint overhang is
    # already bounded by length/coverage gates and is not lateral shape error.
    normal=np.array([-direction[1],direction[0]])
    low,high=max(pa.min(),pb.min()),min(pa.max(),pb.max())
    lateral=np.concatenate((a['xy']@normal,b['xy']@normal));lo,hi=lateral.min()-1,lateral.max()+1
    slab=Polygon([low*direction+lo*normal,high*direction+lo*normal,high*direction+hi*normal,low*direction+hi*normal])
    ga=a['geometry'].intersection(slab);gb=b['geometry'].intersection(slab)
    if ga.is_empty or gb.is_empty:return None
    offset=float((np.asarray(gb.centroid.coords[0])-np.asarray(ga.centroid.coords[0]))@normal)
    limit=min(radius,min(a['width'],b['width'])*(.75 if complex_case else 1.25))
    if abs(offset)>limit:return None
    moved=translate(gb,xoff=-offset*normal[0],yoff=-offset*normal[1])
    error=float(hausdorff_distance(ga,moved))
    if error>(1.5 if complex_case else 2.5):return None
    if min(a['width'],b['width'])/max(a['width'],b['width'])<.7:return None
    return (abs(offset)/max(limit,1.)+error+(1-coverage)*5,abs(offset),error,coverage)


def reconcile_presence(records,*,tolerance=3.,scenes=()):
    """Suppress whole mutually unique groups; retain all original diagnostics."""
    started=time.perf_counter();counts=Counter(v2_reconciled_pairs=0,v2_reconciled_added=0,v2_reconciled_removed=0,
                                             v2_reconciliation_nearby_pairs=0,v2_reconciliation_ambiguous=0,
                                             v2_reconciled_published_added=0,v2_reconciled_published_removed=0)
    removed=_groups(records,'removed');added=_groups(records,'added')
    radius=max(8.,tolerance*4);tree=STRtree([r['geometry'] for r in added]);ranked=[]
    for i,a in enumerate(removed):
        for j in tree.query(a['geometry'],predicate='dwithin',distance=radius):
            j=int(j);counts['v2_reconciliation_nearby_pairs']+=1
            score=_pair(a,added[j],radius)
            if score is not None:ranked.append((score[0],i,j,score))
    by_before={};by_after={}
    for row in sorted(ranked):
        by_before.setdefault(row[1],[]).append(row);by_after.setdefault(row[2],[]).append(row)
    for _,i,j,score in sorted(ranked):
        aa,bb=by_before[i],by_after[j]
        if aa[0][2]!=j or bb[0][1]!=i:continue
        # Competing parallel candidates: do not consume either track.
        if (len(aa)>1 and aa[1][0]-aa[0][0]<.5) or (len(bb)>1 and bb[1][0]-bb[0][0]<.5):
            counts['v2_reconciliation_ambiguous']+=1;continue
        a,b=removed[i],added[j]
        parallel=False
        for group,scene in zip((a,b),scenes):
            for k in scene.tree.query(group['geometry'],predicate='dwithin',distance=radius):
                other=scene.lines[int(k)]
                if other.distance(group['geometry'])<1.5:continue
                if abs(_direction(other)@group['direction'])>.985:
                    parallel=True;break
        if parallel and (score[1]>min(a['width'],b['width'])*.35 or score[2]>.75 or score[3]<.97):
            counts['v2_reconciliation_ambiguous']+=1;continue
        pair_id=f"stable_pair_{counts['v2_reconciled_pairs']}"
        for group,kind in ((a,'removed'),(b,'added')):
            for k in group['members']:
                counts[f'v2_reconciled_published_{kind}']+=int(records[k].get('v2_publish',False))
                records[k].update(v2_publish=False,v2_precision_reason='paired_displacement_stable',
                                  v2_stable_pair=pair_id,v2_pair_offset_m=score[1],v2_pair_shape_error_m=score[2])
                counts[f'v2_reconciled_{kind}']+=1
        counts['v2_reconciled_pairs']+=1
    counts['timing_v2_object_reconciliation_seconds']=time.perf_counter()-started
    return counts
