"""Joint longitudinal RGB axis inference over complete skeleton-seeded corridors.

No semantic extraction, model inference or width-product computation. Apertures
are descriptor scales. Road-level carriageway identity persists through weak
observations; a broad road ribbon is not evidence for collapsing two road axes.
"""
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from shapely import union_all, points, distance, covers
from shapely.geometry import LineString,Point
from shapely.ops import substring

from evidence import stations, longest_hole
from medial_repair import AxisProposal, direction, straight_priors, line_parts, pin_axis_contacts
from color_ribbon import color_features
from rgb_route import ribbon_cost_features, paired_curb_features
from engine.canonical_road_surface import _level


def recurrent_dual_prior(axis, network, aperture):
    """Count repeated paired intersections, rather than one incidental branch."""
    _,xy,_,normals=stations(axis,5.)
    separations=[]
    for point,normal in zip(xy[3:-3],normals[3:-3]):
        cut=LineString([point-normal*aperture*.75,point+normal*aperture*.75])
        hit=network.intersection(cut)
        parts=list(hit.geoms) if hasattr(hit,'geoms') else [hit]
        offsets=sorted((np.array(p.coords[0])-point)@normal for p in parts if p.geom_type=='Point' and not p.is_empty)
        gaps=[b-a for i,a in enumerate(offsets) for b in offsets[i+1:] if 6<=b-a<=aperture*1.3]
        separations.append(max(gaps,default=0.))
    return dict(paired_section_fraction=float(np.mean(np.array(separations)>0)) if separations else 0.,
                paired_sections=int(np.sum(np.array(separations)>0)),sampled_sections=len(separations))


def carriageway_axes(proposal,frame,roi,evidence=None):
    """Retain two road-level tracks where the original road repeatedly has two.

    The caller's confirmed double-road convention includes painted separation,
    not just a vegetated median. This is not lane-by-lane extraction.
    """
    p=proposal
    same_level=frame[frame.apply(_level,axis=1)==p.level]
    layout=recurrent_dual_prior(p.axis,union_all(same_level.geometry),p.band_scale)
    p.evidence.update(layout)
    if p.band_scale<20:
        p.evidence.update(carriageway_count=1,corridor_id=p.key)
        return [p]
    if layout['paired_section_fraction']<.25 or layout['paired_sections']<6:
        p.evidence['layout_reason']='wide_pavement_does_not_prove_single_axis_keep_original'
        return []
    # Refit the road-level direction jointly. Both carriageways inherit the same
    # gentle curvature and retain fixed order; neither can jump to the other.
    _,xy,_,_=stations(p.axis,2.)
    v=direction(p.axis);normal=np.array([-v[1],v[0]]);origin=xy.mean(axis=0)
    s=(xy-origin)@v;offset=(xy-origin)@normal
    coefficients=np.polynomial.polynomial.polyfit(s/max(np.ptp(s),1),offset,2)
    fitted=np.polynomial.polynomial.polyval(s/max(np.ptp(s),1),coefficients)
    # Do not straighten a genuinely curved road across its pavement boundary.
    if np.quantile(abs(fitted-offset),.95)>p.band_scale*.2:return []
    center=origin+s[:,None]*v+fitted[:,None]*normal
    # Offset copies of a clipped axis otherwise end INSIDE the ROI on one side,
    # creating artificial tips. Extend its border ends before offset and clip.
    if Point(p.axis.coords[0]).distance(roi.boundary)<.01:
        tangent=center[1]-center[0];tangent/=np.linalg.norm(tangent)
        center=np.vstack([center[0]-tangent*p.band_scale*2,center])
    if Point(p.axis.coords[-1]).distance(roi.boundary)<.01:
        tangent=center[-1]-center[-2];tangent/=np.linalg.norm(tangent)
        center=np.vstack([center,center[-1]+tangent*p.band_scale*2])
    # Joint image alignment AFTER regularization. Least-squares fitting alone
    # changes road position slightly; optimize one shared offset for both tracks,
    # never independent offsets that let them exchange identity or cross.
    shift=0.
    if evidence is not None:
        trials=[]
        for delta in np.arange(-3.,3.01,.5):
            curve=LineString(center+delta*normal).intersection(roi)
            if curve.geom_type=='LineString':
                strength,valid=axis_strength(curve,p.band_scale,evidence)
                if valid>=.98:trials.append((strength,delta))
        if not trials:return []
        _,shift=max(trials,key=lambda t:(t[0],-abs(t[1])))
        center=center+shift*normal
    answer=[]
    road_envelope=LineString(center).buffer(p.band_scale*.58,cap_style='flat').intersection(roi)
    for track,sign in enumerate((-1,1)):
        track_offset=sign*p.band_scale*.25
        track_strength=None
        if evidence is not None:
            # One constant displacement per carriageway, selected from its own
            # half of the observed pavement. Quarter-width offsets alone can
            # land on parking/curb edges when the broad descriptor includes them.
            alternatives=[]
            for magnitude in np.arange(p.band_scale*.14,p.band_scale*.35+.01,.5):
                curve=LineString(center+sign*magnitude*normal).intersection(roi)
                if curve.geom_type=='LineString':
                    strength,valid=axis_strength(curve,p.band_scale*.4,evidence)
                    if valid>=.98:alternatives.append((strength,sign*magnitude))
            if not alternatives:return []
            track_strength,track_offset=max(alternatives,key=lambda t:t[0])
        axis=LineString(center+track_offset*normal).intersection(roi)
        if axis.geom_type!='LineString' or axis.length<80:return []
        info=dict(p.evidence,carriageway_count=2,carriageway_index=track,corridor_id=p.key,
                  road_aperture_m=p.band_scale,road_axis_wkt=LineString(center).intersection(roi).wkt,
                  shared_rgb_alignment_shift_m=float(shift),
                  carriageway_offset_m=float(track_offset),carriageway_rgb_strength=track_strength,
                  layout_reason='recurrent_double_prior_on_broad_rgb_pavement',
                  axis_wkt=axis.wkt)
        answer.append(AxisProposal(axis,road_envelope,
            p.band_scale*.5,p.support,info,p.seed_ids,p.level))
    spacing=abs(answer[1].evidence['carriageway_offset_m']-answer[0].evidence['carriageway_offset_m'])
    for track in answer:track.evidence['axis_spacing_m']=spacing
    return answer


def pavement_descriptor(rgb,fractions):
    """Bilateral material change relative to within-pavement variability.

    Absolute color contrast alone misses dark asphalt next to dark tree rows.
    A continuous neutral, low-texture interior and two different flanks provide
    independent evidence even when their mean intensities are close.
    """
    interior=rgb[...,(fractions>=.1)&(fractions<=.9),:]
    middle=np.median(interior,axis=-2)
    variability=np.quantile(np.linalg.norm(interior-middle[...,None,:],axis=-1)/np.sqrt(3),.8,axis=-1)
    contrasts=[]
    for mask in (fractions<-.05,fractions>1.05):
        flank=rgb[...,mask,:]
        contrasts.append(np.median(np.linalg.norm(flank-middle[...,None,:],axis=-1)/np.sqrt(3),axis=-1))
    bilateral=np.minimum(*contrasts)
    ratio=bilateral/(variability+.012)
    chroma=np.max(middle,axis=-1)-np.min(middle,axis=-1)
    strength=(1-np.exp(-ratio/2))*np.exp(-variability/.06)*np.exp(-chroma/.12)
    supported=(ratio>=2)&(bilateral>=.025)&(chroma<=.10)
    return strength,supported


def axis_strength(axis,scale,evidence):
    """Evaluate old and new axes with identical independent RGB descriptors."""
    _,xy,_,normal=stations(axis,2.)
    f=np.linspace(-.4,1.4,37)
    probes=xy[:,None]+normal[:,None]*(f[None,:,None]-.5)*scale
    rgb=np.stack([evidence.image.sample(evidence.image.rgb[...,k],probes.reshape(-1,2)) for k in range(3)],axis=-1).reshape(len(xy),len(f),3)/255.
    gray=evidence.image.sample(evidence.image.gray,probes.reshape(-1,2)).reshape(len(xy),len(f))
    pool=median_filter(np.nan_to_num(rgb),size=(21,1,1),mode='nearest')
    pg=median_filter(np.nan_to_num(gray),size=(21,1),mode='nearest')
    scores=[color_features(pool,f,independent_flanks=True)[0],ribbon_cost_features(pg,f)[0],pavement_descriptor(pool,f)[0],
            paired_curb_features(median_filter(np.nan_to_num(gray),size=(7,1),mode='nearest'),f)[0]]
    interior=pool[:,(f>=.15)&(f<=.85)]
    green=np.maximum(2*interior[...,1]-interior[...,0]-interior[...,2],0).mean(axis=-1)
    score=np.max(scores,axis=0)-3*green
    return float(np.mean(score)),float(np.isfinite(rgb).all(axis=(-1,-2)).mean())


def corridor_groups(frame, roi):
    seeds=[]
    for idx,row in frame.iterrows():
        for part in line_parts(row.geometry.intersection(roi.buffer(60))):
            for line in straight_priors(part, maximum_residual=2.):
                if line.length >= 25:
                    seeds.append((line,idx,_level(row)))
    parent=list(range(len(seeds)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,(a,_,level) in enumerate(seeds):
        v=direction(a);n=np.array([-v[1],v[0]])
        for j,(b,_,other) in enumerate(seeds[:i]):
            if level!=other or abs(v@direction(b))<.995 or a.distance(b)>75:
                continue
            delta=np.array(a.centroid.coords[0])-np.array(b.centroid.coords[0])
            if abs(delta@n)>18:
                continue
            parent[root(i)]=root(j)
    groups={}
    for i,seed in enumerate(seeds):groups.setdefault(root(i),[]).append(seed)
    return [g for g in groups.values() if sum(s[0].length for s in g)>=100]


def fit_corridor(group, evidence, roi):
    # Length-weighted direction from individual segments avoids biased PCA when
    # side axes cover different longitudinal intervals.
    v=sum(direction(g)*g.length for g,_,_ in group);v/=np.linalg.norm(v)
    normal=np.array([-v[1],v[0]])
    xy=np.concatenate([stations(g,3)[1] for g,_,_ in group])
    origin=xy.mean(axis=0)
    s=(xy-origin)@v
    full=LineString([origin-v*10000,origin+v*10000]).intersection(roi)
    if full.is_empty or full.geom_type!='LineString':return [],[]
    full_s=(np.array(full.coords)-origin)@v
    # Extrapolate to nearby ROI seams, otherwise only 25m beyond saved priors;
    # internal ends are resolved jointly at receiving street intersections.
    lo=full_s.min()-30 if s.min()<=full_s.min()+45 else s.min()-25
    hi=full_s.max()+30 if s.max()>=full_s.max()-45 else s.max()+25
    chord=LineString([origin+v*lo,origin+v*hi])
    if chord.geom_type!='LineString' or chord.length<100:return [],[]
    ends=(np.array(chord.coords)-origin)@v
    along=np.linspace(ends.min(),ends.max(),int(chord.length/2)+1)
    base=origin+along[:,None]*v
    offsets=np.arange(-24.,24.01,1.)
    fractions=np.linspace(-.4,1.4,37)
    lateral=np.arange(-62.,62.01,.25)
    strip_xy=base[:,None,:]+normal*lateral[None,:,None]
    flat=strip_xy.reshape(-1,2)
    strip_rgb=np.stack([evidence.image.sample(evidence.image.rgb[...,k],flat) for k in range(3)],axis=-1).reshape(len(base),len(lateral),3)/255.
    strip_gray=evidence.image.sample(evidence.image.gray,flat).reshape(len(base),len(lateral))
    pooled_rgb=median_filter(np.nan_to_num(strip_rgb),size=(21,1,1),mode='nearest')
    pooled_gray_strip=median_filter(np.nan_to_num(strip_gray),size=(21,1),mode='nearest')
    curb_strip=median_filter(np.nan_to_num(strip_gray),size=(7,1),mode='nearest')
    choices=[];diagnostics=[]
    for scale in (6.,9.,12.,16.,20.,26.,32.,40.):
        sample_offsets=offsets[:,None]+(fractions[None,:]-.5)*scale
        fractional=(sample_offsets-lateral[0])/.25
        low=np.floor(fractional).astype(int);mix=fractional-low
        def extract(array):
            w=mix[...,None] if array.ndim==3 else mix
            return array[:,low]*(1-w)+array[:,low+1]*w
        rgb=extract(strip_rgb);gray=extract(strip_gray)
        # 42m longitudinal context suppresses vehicles and intermittent marking;
        # raw station evidence remains separately recorded and checked.
        pooled=extract(pooled_rgb);pooled_gray=extract(pooled_gray_strip)
        score,supported,contrast,spread=color_features(pooled,fractions,independent_flanks=True)
        gs,valid,_,_=ribbon_cost_features(pooled_gray,fractions)
        score=np.maximum(score,gs)
        material,material_supported=pavement_descriptor(pooled,fractions)
        score=np.maximum(score,material)
        supported|=material_supported
        curb,curb_supported=paired_curb_features(extract(curb_strip),fractions)
        score=np.maximum(score,curb)
        supported|=curb_supported
        valid=np.isfinite(rgb).all(axis=(-1,-2))
        interior=pooled[:,:,(fractions>=.15)&(fractions<=.85)]
        green=np.maximum(2*interior[...,1]-interior[...,0]-interior[...,2],0).mean(axis=-1)
        score-=3*green
        score[~valid]=-10
        # Shared global offset and scale prevent lane-to-lane switching. Allow
        # slow changes around that identity, not independent station maxima.
        global_scores=np.mean(score,axis=0)
        center=int(np.argmax(global_scores))
        cost=-score+.04*((offsets-offsets[center])/4.)[None,:]**2
        dp=cost[0].copy();back=np.zeros(cost.shape,dtype=int)
        for i in range(1,len(base)):
            values=np.full((3,len(offsets)),np.inf)
            for k,jump in enumerate((-1,0,1)):
                previous=np.arange(len(offsets))+jump
                good=(previous>=0)&(previous<len(offsets))
                values[k,good]=dp[previous[good]]+.25*jump*jump
            best=values.argmin(axis=0)
            back[i]=np.arange(len(offsets))+best-1
            dp=values[best,np.arange(len(offsets))]+cost[i]
        selected=[int(dp.argmin())]
        for i in range(len(base)-1,0,-1):selected.append(back[i,selected[-1]])
        selected=np.array(selected[::-1]);ii=np.arange(len(base))
        support=supported[ii,selected]&valid[ii,selected]
        raw=color_features(rgb[ii,selected],fractions,independent_flanks=True)
        coords=base+normal*gaussian_filter1d(offsets[selected],4)[:,None]
        inside_roi=covers(roi,points(coords))
        if inside_roi.sum()<30:continue
        strength=float(np.mean(score[ii,selected][inside_roi]))
        # Favor the complete road ribbon over an internal lane-marking strip,
        # provided the broad aperture also has bilateral independent evidence.
        objective=strength*(scale/12.)**.35
        axis=LineString(coords).simplify(.25).intersection(roi)
        if axis.geom_type!='LineString' or axis.length<100:
            continue
        near=distance(points(coords),union_all([g for g,_,_ in group]))
        side=(xy-np.array(axis.centroid.coords[0]))@normal
        paired_prior=min(np.mean(side>2),np.mean(side<-2))>=.2
        movement_limit=max(8.,min(20. if paired_prior else 12.,.7*scale))
        info=dict(scale=scale,strength=strength,objective=objective,
            longitudinal_support=float(support[inside_roi].mean()),raw_support=float(raw[1][inside_roi].mean()),
            maximum_weak_m=longest_hole(support[inside_roi],axis.length),valid_fraction=float(valid[ii,selected][inside_roi].mean()),
            center_offset_m=float(np.median(offsets[selected])),
            paired_side_priors=bool(paired_prior),movement_limit_m=movement_limit,
            prior_start_xy=(origin+v*s.min()).tolist(),prior_end_xy=(origin+v*s.max()).tolist(),
            prior_distance_q90_m=float(np.quantile(near[inside_roi],.9)),axis_wkt=axis.wkt)
        diagnostics.append(info)
        if (info['valid_fraction']>=.98 and info['longitudinal_support']>=.55
                and info['maximum_weak_m']<=65 and strength>=.18
                and info['prior_distance_q90_m']<=movement_limit):
            proposal=AxisProposal(axis,axis.buffer(scale*.58,cap_style='flat').intersection(roi),scale,
                info['longitudinal_support'],dict(info,whole_corridor=True),[int(i) for _,i,_ in group],group[0][2])
            choices.append((objective,proposal))
    if not choices:return [],diagnostics
    best=max(x[0] for x in choices)
    # A lane-marking strip and the outer road ribbon are alternative apertures
    # of one road. Prefer the outer supported ribbon when scores are equivalent.
    finalists=[p for score,p in choices if score>=best*.95]
    return [max(finalists,key=lambda p:p.band_scale)],diagnostics


def proposals(frame,evidence,roi):
    candidates=[];audit=[]
    for i,group in enumerate(corridor_groups(frame,roi)):
        found,diagnostics=fit_corridor(group,evidence,roi)
        audit.append(dict(group=i,source_ids=[int(j) for _,j,_ in group],hypotheses=diagnostics))
        candidates.extend(found)
    # Each pavement corridor is committed at most once, including overlapping
    # groups created by short branches and duplicate old axes.
    chosen=[]
    for p in sorted(candidates,key=lambda p:p.axis.length*p.support,reverse=True):
        if any(p.level==q.level and abs(direction(p.axis)@direction(q.axis))>.98
               and p.axis.intersection(q.envelope).length>.6*p.axis.length for q in chosen):continue
        chosen.append(p)
    # Preserve road representation BEFORE junction fitting. Never turn two
    # verified carriageways into one merely because their pavement is continuous.
    represented=[]
    for p in chosen:
        tracks=carriageway_axes(p,frame,roi,evidence)
        audit.append(dict(kind='road_representation',candidate_id=p.key,evidence=p.evidence,
                          proposed_carriageways=len(tracks)))
        represented.extend(tracks)
    chosen=represented
    junction_contacts=[]
    for i,p in enumerate(chosen):
        for q in chosen[:i]:
            if p.level!=q.level:continue
            hit=p.axis.intersection(q.axis)
            if hit.geom_type=='Point':junction_contacts.append((p.level,hit.coords[0]))
    # A T approach must terminate at its receiving road, rather than extending
    # the fitted straight prior through the opposite verge to the ROI border.
    for p in chosen:
        lo,hi=0.,p.axis.length
        starts=[];ends=[]
        for q in chosen:
            if q is p or q.level!=p.level or abs(direction(p.axis)@direction(q.axis))>.8:continue
            hit=p.axis.intersection(q.axis)
            if hit.geom_type!='Point':continue
            position=p.axis.project(hit)
            junction_footprint=max(10.,.75*max(p.evidence.get('road_aperture_m',p.band_scale),q.evidence.get('road_aperture_m',q.band_scale)))
            if position<60 and p.axis.project(Point(p.evidence['prior_start_xy']))>=position-junction_footprint:starts.append(position)
            if p.axis.length-position<60 and p.axis.project(Point(p.evidence['prior_end_xy']))<=position+junction_footprint:ends.append(position)
        # A double-road approach reaches both receiving carriageways, not only
        # the nearest axis. Their intersections remain distinct graph nodes.
        if starts:lo=min(starts)
        if ends:hi=max(ends)
        if hi-lo>50 and (lo>0 or hi<p.axis.length):
            p.evidence['terminal_junction_trim_m']=[lo,p.axis.length-hi]
            p.axis=substring(p.axis,lo,hi)
            if p.evidence.get('carriageway_count')!=2:
                p.envelope=p.axis.buffer(p.band_scale*.58,cap_style='flat').intersection(roi)
    # substring can move an intersection endpoint by a few floating-point ulps.
    # Pin both incident axes to the original shared vertex, not two interpolations.
    for p in chosen:
        p.axis=pin_axis_contacts(p.axis,[xy for level,xy in junction_contacts if level==p.level])
    return chosen,audit
