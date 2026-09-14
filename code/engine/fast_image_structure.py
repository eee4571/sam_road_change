"""Bounded raw-image descriptors. No model evidence or truth participates."""
import cv2
import numpy as np
from scipy.ndimage import uniform_filter, binary_erosion, gaussian_filter1d


def q(values, fraction, default=np.nan):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return float(np.quantile(values,fraction)) if values.size else default


def ncc(a,b,mask):
    if np.count_nonzero(mask)<16:return np.nan
    a=a[mask];b=b[mask]
    if np.std(a)<1e-6 or np.std(b)<1e-6:return np.nan
    return float(np.corrcoef(a,b)[0,1])


def axis_grid(axis,xy):
    """Nearest polyline segment coordinates for raster pixels, including bends."""
    coords=np.asarray(axis.coords)[:,:2];best=np.full(xy.shape[:2],np.inf)
    lateral=np.zeros_like(best);along=np.zeros_like(best);normal=np.zeros_like(xy);length=0.
    for start,end in zip(coords[:-1],coords[1:]):
        vector=end-start;size=float(np.linalg.norm(vector))
        if size<=1e-9:continue
        direction=vector/size;n=np.array([-direction[1],direction[0]])
        delta=xy-start;projection=delta@direction;clamped=np.clip(projection,0,size)
        distance=np.sum((delta-clamped[...,None]*direction)**2,axis=-1);take=distance<best
        best[take]=distance[take];lateral[take]=(delta@n)[take]
        along[take]=(length+clamped)[take];normal[take]=n;length+=size
    return lateral,np.minimum(2,(3*along/max(length,1e-9)).astype(int)),normal


def normalized_gray(rgb):
    gray=np.mean(rgb,axis=0);valid=np.isfinite(gray)
    low,high=q(gray,.1),q(gray,.9)
    if not np.isfinite(high-low) or high-low<=1e-6:return np.zeros(gray.shape,np.float32),valid&False
    return np.nan_to_num(np.clip((gray-low)/(high-low),-1,2),nan=0.).astype(np.float32),valid


def gradients(gray):
    return cv2.Scharr(gray,cv2.CV_32F,1,0)/32, -cv2.Scharr(gray,cv2.CV_32F,0,1)/32


def align_pair(a,b,va,vb,ring,max_shift=3.):
    """Translation only, estimated/validated on background, never road edges.

    Three pixels is a registration safety bound, not a change threshold. An
    uncertain transform is left at identity and reported, not silently trusted.
    """
    valid=binary_erosion(va&vb,iterations=2);mask=valid&ring
    info=dict(dx_px=0.,dy_px=0.,response=0.,accepted=False,gain=1.,offset=0.)
    if mask.sum()<64:return b,vb,info
    ga=np.hypot(*gradients(a));gb=np.hypot(*gradients(b))
    window=cv2.GaussianBlur(mask.astype(np.float32),(5,5),0)
    delta,response=cv2.phaseCorrelate(ga*window,gb*window)
    info['response']=float(response);info['proposed_shift_px']=list(map(float,delta))
    if np.isfinite(delta).all() and np.linalg.norm(delta)<=max_shift and response>0.1:
        matrix=np.float32([[1,0,-delta[0]],[0,1,-delta[1]]])
        shifted=cv2.warpAffine(b,matrix,(b.shape[1],b.shape[0]),flags=cv2.INTER_LINEAR)
        shifted_valid=cv2.warpAffine(vb.astype(np.float32),matrix,(b.shape[1],b.shape[0]),flags=cv2.INTER_LINEAR)>.999
        common=mask&binary_erosion(shifted_valid,iterations=2)
        old=ncc(ga,gb,common);new=ncc(ga,np.hypot(*gradients(shifted)),common)
        if np.isfinite(new) and (not np.isfinite(old) or new>old):
            b=shifted;vb=shifted_valid
            info.update(dx_px=float(delta[0]),dy_px=float(delta[1]),accepted=True)
    mask=ring&va&vb
    # Robust affine illumination correction on background only; no histogram
    # matching of the road, which could erase the very change being tested.
    alo,ahi=q(a[mask],.2),q(a[mask],.8);blo,bhi=q(b[mask],.2),q(b[mask],.8)
    if np.isfinite(ahi-alo) and bhi-blo>1e-6:
        gain=float(np.clip((ahi-alo)/(bhi-blo),2/3,1.5))
        offset=float(np.clip(q(a[mask],.5)-gain*q(b[mask],.5),-.25,.25))
        b=(b*gain+offset).astype(np.float32);info.update(gain=gain,offset=offset)
    return b,vb,info


def describe(gray,valid,core,ring,lateral,bins,normal,width,resolution):
    gx,gy=gradients(gray);magnitude=np.hypot(gx,gy)
    perpendicular=np.abs(gx*normal[...,0]+gy*normal[...,1])
    parallel=np.abs(-gx*normal[...,1]+gy*normal[...,0])
    valid=binary_erosion(valid,iterations=2)
    xx=uniform_filter(gx*gx,5);yy=uniform_filter(gy*gy,5);xy=uniform_filter(gx*gy,5)
    coherence=np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-8)
    background=q(magnitude[ring&valid],.5,0.)
    boundaries=[];scores=[]
    step=max(resolution,.5);edges=np.arange(-width,width+step*1.5,step)
    centers=(edges[1:]+edges[:-1])/2
    for band in range(3):
        use=valid&(bins==band)&(np.abs(lateral)<=width)
        idx=np.clip(np.searchsorted(edges,lateral[use],side='right')-1,0,len(centers)-1)
        counts=np.bincount(idx,minlength=len(centers))
        energy=np.bincount(idx,weights=(perpendicular*coherence)[use],minlength=len(centers))
        profile=gaussian_filter1d(energy/np.maximum(counts,1),.7)
        pair=[];strength=[]
        for sign in (-1,1):
            search=(sign*centers>=max(step*.5,width*.15))&(sign*centers<=width*.95)&(counts>0)
            choices=np.flatnonzero(search)
            if not len(choices):pair.append(np.nan);strength.append(0.);continue
            index=choices[np.argmax(profile[choices])]
            pair.append(float(centers[index]));strength.append(float(profile[index]))
        boundaries.append(pair)
        # Both sides must carry directional structure, unlike a lone field edge.
        scores.append(min(strength)/(background+1e-4))
    angle=np.arctan2(parallel,perpendicular+1e-9)
    hist=np.histogram(angle[core&valid],bins=9,range=(0,np.pi/2),weights=magnitude[core&valid])[0]
    hist=hist/(hist.sum()+1e-9)
    return dict(boundaries=np.asarray(boundaries),road_score=q(scores,.5,0.),boundary_scores=np.asarray(scores),
                hog=hist,coherence=q(coherence[core&valid],.5,0.),normalized=gray,gradient=magnitude,
                directional=perpendicular,image_valid=valid)


def similarity(a,b,mask):
    # Local structural SSIM on normalized luminance, no raw RGB difference.
    x=a['normalized'];y=b['normalized'];mx=uniform_filter(x,7);my=uniform_filter(y,7)
    vx=np.maximum(0,uniform_filter(x*x,7)-mx*mx);vy=np.maximum(0,uniform_filter(y*y,7)-my*my)
    cov=uniform_filter(x*y,7)-mx*my
    ssim=(2*mx*my+.01**2)*(2*cov+.03**2)/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return q(ssim[mask],.5),ncc(a['gradient'],b['gradient'],mask)


def image_features(rgb_a,rgb_b,core,ring,lateral,bins,normal,width,resolution):
    a,va=normalized_gray(rgb_a);b,vb=normalized_gray(rgb_b)
    b,vb,registration=align_pair(a,b,va,vb,ring)
    periods=[describe(g,v,core,ring,lateral,bins,normal,width,resolution) for g,v in ((a,va),(b,vb))]
    a,b=periods;valid=a['image_valid']&b['image_valid']
    cs,cn=similarity(a,b,core&valid);bs,bn=similarity(a,b,ring&valid)
    loss=lambda s,n:(1-np.clip(s,-1,1)+1-np.clip(n,-1,1))/2
    core_loss=loss(cs,cn);ring_loss=loss(bs,bn)
    return dict(periods=periods,registration=registration,ncc=cn,
                core_ncc=ncc(a['normalized'],b['normalized'],core&valid),ssim=cs,ring_ssim=bs,
                anomaly=core_loss-ring_loss,hog_distance=float(np.sum(np.abs(a['hog']-b['hog']))/2),
                score_delta=b['road_score']-a['road_score'],
                left=a['boundaries'][:,0]-b['boundaries'][:,0],right=b['boundaries'][:,1]-a['boundaries'][:,1],
                image_valid=bool(core.any() and np.mean(valid[core])>=.9 and np.count_nonzero(valid&ring)>=32))
