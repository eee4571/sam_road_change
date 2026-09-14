"""Training-free, bounded paired-image verification of Fast2 candidates.

No truth input, learned features, station scan or width-profile modification.
All raster operations are restricted to candidate/control windows.
"""
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import time
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.vrt import WarpedVRT
from scipy.ndimage import sobel,label
from shapely import from_wkt
from shapely.geometry import box
from shapely.strtree import STRtree


class ImageTiles:
    def __init__(self,payload,crs):
        self.stack=ExitStack();self.rows=[];self.crs=crs;self.reads=0
        directory=Path(str(payload.get('width_review') or '__missing__'))
        try:
            for path in sorted(directory.glob('*_summary.json')):
                row=json.loads(path.read_text(encoding='utf-8'))
                if not row.get('image'):continue
                image=Path(row['image'])
                if not image.is_file():continue
                ds=self.stack.enter_context(rasterio.open(image))
                if not ds.crs:continue
                surface=Path(str(row.get('enhanced_molra_surface_mask') or row.get('molra_surface_mask') or '__missing__'))
                mask=self.stack.enter_context(rasterio.open(surface)) if surface.is_file() else None
                if mask is not None and mask.shape!=ds.shape:mask=None
                if mask is not None:
                    mask=self.stack.enter_context(WarpedVRT(mask,src_crs=ds.crs,src_transform=ds.transform,
                        crs=ds.crs,transform=ds.transform,width=ds.width,height=ds.height))
                bounds=transform_bounds(ds.crs,crs,*ds.bounds)
                self.rows.append((ds,mask,box(*bounds)))
            self.tree=STRtree([r[2] for r in self.rows])
        except Exception:
            self.close();raise

    def close(self):self.stack.close()

    def read(self,bounds,shape,transform):
        rgb=np.full((3,*shape),np.nan,np.float32);surface=np.full(shape,np.nan,np.float32)
        for index in sorted(self.tree.query(box(*bounds))):
            ds,mask,_=self.rows[index]
            channels=list(range(1,min(3,ds.count)+1));data=np.full((len(channels),*shape),np.nan,np.float32)
            reproject(rasterio.band(ds,channels),data,src_transform=ds.transform,src_crs=ds.crs,
                      dst_transform=transform,dst_crs=self.crs,dst_nodata=np.nan,
                      resampling=Resampling.bilinear,warp_mem_limit=16)
            self.reads+=1
            if len(channels)<3:data=np.repeat(data[:1],3,axis=0)
            valid=np.isfinite(data).all(axis=0)&~np.isfinite(rgb).all(axis=0)
            rgb[:,valid]=data[:,valid]
            if mask is not None:
                data=np.full(shape,np.nan,np.float32)
                reproject(rasterio.band(mask,1),data,src_transform=ds.transform,src_crs=ds.crs,
                          dst_transform=transform,dst_crs=self.crs,dst_nodata=np.nan,
                          resampling=Resampling.nearest,warp_mem_limit=16)
                surface[valid]=data[valid]>0
                surface[valid&~np.isfinite(data)]=np.nan
                self.reads+=1
        return rgb,surface


def quantile(values,q,default=np.nan):
    v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
    return float(np.quantile(v,q)) if len(v) else default


def road_boundaries(surface,anchor,outer,bins,lateral):
    """Select the component touching this axis, excluding parallel tracks."""
    labels,_=label((surface>0)&outer,structure=np.ones((3,3)))
    ids=labels[anchor];counts=np.bincount(ids)
    if len(counts):counts[0]=0
    component=int(np.argmax(counts)) if len(counts) and counts.max()>0 else 0
    boundaries=[]
    for band in range(3):
        values=lateral[(labels==component)&(bins==band)] if component else []
        boundaries.append([quantile(values,.05),quantile(values,.95)] if len(values)>8 else [np.nan,np.nan])
    return np.asarray(boundaries)


class PatchVerifier:
    def __init__(self,scenes,payloads,maximum_controls=96):
        self.scenes=scenes;self.maximum_controls=maximum_controls;self.counts=Counter()
        self.tiles=[];self.controls=[];self.calibration={};self.audit=[]
        try:
            for p in payloads:self.tiles.append(ImageTiles(p,scenes[0].crs))
        except Exception:
            self.close();raise

    def close(self):
        for tile in self.tiles:tile.close()

    def patch(self,axis,width):
        # One bounded 2-D region, not a line of longitudinal sample positions.
        padding=max(width,12.)
        bounds=axis.buffer(padding,cap_style='flat').bounds
        left,bottom,right,top=bounds
        resolution=max(1.,(right-left)/192,(top-bottom)/192)
        shape=(max(8,int(np.ceil((top-bottom)/resolution))),max(8,int(np.ceil((right-left)/resolution))))
        transform=from_bounds(*bounds,shape[1],shape[0])
        yy,xx=np.indices(shape);xy=np.stack((transform.c+(xx+.5)*transform.a,transform.f+(yy+.5)*transform.e),axis=-1)
        core=rasterize([(axis.buffer(width/2,cap_style='flat'),1)],out_shape=shape,transform=transform).astype(bool)
        outer=rasterize([(axis.buffer(width/2+max(4.,width/3),cap_style='flat'),1)],out_shape=shape,transform=transform).astype(bool)
        flank=outer&~core
        anchor=rasterize([(axis.buffer(min(width/4,2.),cap_style='flat'),1)],out_shape=shape,transform=transform).astype(bool)
        coords=np.asarray(axis.coords)[:,:2];direction=coords[-1]-coords[0]
        chord=float(np.linalg.norm(direction));direction/=max(chord,1e-9);normal=np.array([-direction[1],direction[0]])
        offsets=xy-coords[0];longitudinal=offsets@direction;lateral=offsets@normal
        bins=np.clip((3*longitudinal/max(chord,1e-9)).astype(int),0,2)
        periods=[]
        for scene,tiles in zip(self.scenes,self.tiles):
            rgb,surface=tiles.read(bounds,shape,transform)
            probability=scene.probability._values_at(xy[...,0].ravel(),xy[...,1].ravel()).reshape(shape)
            valid=np.isfinite(probability)
            pcore=quantile(probability[core&valid],.5);pbg=quantile(probability[flank&valid],.5)
            surface_valid=np.isfinite(surface)
            support=float(np.mean(surface[core&surface_valid])) if np.count_nonzero(core&surface_valid)>8 else np.nan
            gray=np.mean(rgb,axis=0);image_valid=np.isfinite(gray)
            low=quantile(gray[image_valid],.1);high=quantile(gray[image_valid],.9)
            informative=np.isfinite(high) and high-low>1e-6
            normalized=np.nan_to_num((gray-low)/max(high-low,1e-6),nan=0.) if informative else np.zeros(shape)
            gx=sobel(normalized,axis=1)/8;gy=-sobel(normalized,axis=0)/8
            edge=np.abs(gx*normal[0]+gy*normal[1]);magnitude=np.hypot(gx,gy)
            strength=(float(np.mean(edge[flank&image_valid])) if np.any(flank&image_valid) and informative else np.nan)
            alignment=(float(np.mean(edge[outer&image_valid]))/(float(np.mean(magnitude[outer&image_valid]))+1e-6)
                       if np.any(outer&image_valid) and informative else np.nan)
            boundaries=road_boundaries(surface,anchor,outer&(longitudinal>=0)&(longitudinal<=chord),bins,lateral)
            periods.append(dict(p=pcore,contrast=pcore-pbg,s= support,edge=strength,alignment=alignment,
                surface_observed=bool(core.any() and np.mean(surface_valid[core])>=.95),
                boundaries=np.asarray(boundaries),rgb=rgb,surface=surface,probability=probability,
                normalized=normalized,gradient=magnitude,image_valid=image_valid))
        a,b=periods;valid=a['image_valid']&b['image_valid']&outer
        ncc=np.nan;core_ncc=np.nan
        if valid.sum()>16 and np.std(a['gradient'][valid])>1e-6 and np.std(b['gradient'][valid])>1e-6:
            # Offset-invariant CV descriptor; no image registration or learning.
            ncc=float(np.corrcoef(a['gradient'][valid],b['gradient'][valid])[0,1])
        interior=valid&core
        if interior.sum()>16 and np.std(a['normalized'][interior])>1e-6 and np.std(b['normalized'][interior])>1e-6:
            core_ncc=float(np.corrcoef(a['normalized'][interior],b['normalized'][interior])[0,1])
        left_delta=a['boundaries'][:,0]-b['boundaries'][:,0]
        right_delta=b['boundaries'][:,1]-a['boundaries'][:,1]
        sufficient=all(scene.valid.covers(axis.buffer(width/2,cap_style='flat')) for scene in self.scenes)
        self.counts['patch_count']+=1;self.counts['patch_pixels']+=core.size
        return dict(periods=periods,ncc=ncc,core_ncc=core_ncc,left=left_delta,right=right_delta,
                    resolution=resolution,straight=chord/max(axis.length,1e-9)>.95,
                    valid=sufficient,bounds=bounds,core=core)

    def calibrate(self,controls):
        started=time.perf_counter()
        # Evenly select in deterministic road order, bounded CPU/I/O work.
        ids=np.linspace(0,len(controls)-1,min(len(controls),self.maximum_controls),dtype=int) if controls else []
        for i in ids:
            row=controls[i];p=self.patch(row['axis'],row['width'])
            if p['valid']:self.controls.append(p)
        c=self.controls
        self.calibration={'count':len(c),'minimum':12,'encoder_features':'unavailable_skipped'}
        for side in range(2):
            for feature in ('p','contrast','s','edge','alignment'):
                values=[p['periods'][side][feature] for p in c]
                if feature=='s':values=[v for v in values if v>0]
                self.calibration[f'{side}_{feature}_low']=quantile(values,.1)
                self.calibration[f'{side}_{feature}_median']=quantile(values,.5)
                self.calibration[f'{side}_{feature}_count']=int(np.isfinite(values).sum())
        for name,values in [('ncc',[p['ncc'] for p in c]),
                             ('core_ncc',[p['core_ncc'] for p in c]),
                             ('left',[quantile(p['left'],.5) for p in c if p['straight']]),
                             ('right',[quantile(p['right'],.5) for p in c if p['straight']])]:
            self.calibration[name+'_low']=quantile(values,.1)
            self.calibration[name+'_median']=quantile(values,.5)
            self.calibration[name+'_high']=quantile(values,.95)
        for feature in ('p','s'):
            diffs=[p['periods'][1][feature]-p['periods'][0][feature] for p in c]
            self.calibration[feature+'_delta_low']=quantile(diffs,.05)
            self.calibration[feature+'_delta_high']=quantile(diffs,.95)
        differences=[quantile(p['left']+p['right'],.5) for p in c if p['straight']]
        self.calibration['boundary_delta_low']=quantile(differences,.05)
        self.calibration['boundary_delta_high']=quantile(differences,.95)
        self.counts['calibration_seconds']=time.perf_counter()-started
        # Free control images; only scalar descriptors remain necessary.
        self.controls=[]

    def reasons(self,p,kind):
        c=self.calibration
        if c.get('count',0)<c.get('minimum',12) or not p['valid']:return [],'insufficient_control_or_valid_area'
        reasons=[];a,b=p['periods']
        if kind in ('added','removed'):
            target=0 if kind=='added' else 1;t=p['periods'][target]
            for feature,name in [('p','opposite_probability_support'),('s','opposite_molra_surface_support')]:
                difference=b[feature]-a[feature]
                normal_loss=(difference<=c[feature+'_delta_high'] if target==0 else difference>=c[feature+'_delta_low'])
                supported=(c.get(f'{target}_{feature}_count',0)>=12 and
                           t[feature]>max(0.,c[f'{target}_{feature}_low']) and normal_loss)
                if feature=='p':supported &= t['contrast']>max(0.,c[f'{target}_contrast_low'])
                if supported:reasons.append(name)
            # A stable surrounding field/building cannot veto a changed road
            # interior: both foreground texture and road-edge structure agree.
            image_same=(np.isfinite(p['ncc']) and p['ncc']>=max(0.,c['ncc_low']) and
                np.isfinite(p['core_ncc']) and p['core_ncc']>=max(0.,c['core_ncc_low']) and
                all(p['periods'][side]['edge']>max(0.,c[f'{side}_edge_low']) and
                    p['periods'][side]['alignment']>=c[f'{side}_alignment_low'] for side in range(2)))
            if image_same:reasons.append('persistent_image_road_structure')
        elif p['straight'] and np.isfinite(p['left']).all() and np.isfinite(p['right']).all():
            if not np.isfinite(c['boundary_delta_high']) or not np.isfinite(c['boundary_delta_low']):
                return [],'surface_calibration_unavailable'
            direction=1 if kind=='widened' else -1
            delta=p['left']+p['right']
            limit=c['boundary_delta_high'] if direction>0 else -c['boundary_delta_low']
            # Also require displacement larger than the local raster uncertainty.
            reliable=direction*delta>max(limit,2*p['resolution'])
            if reliable.sum()<2:reasons.append('surface_expansion_not_sustained')
            l=p['left']-c['left_median'];r=p['right']-c['right_median']
            if np.count_nonzero((l*r<0)&(np.abs(l+r)<=2*p['resolution']))>=2:
                reasons.append('surface_lateral_displacement')
            if np.count_nonzero((delta>=c['boundary_delta_low'])&(delta<=c['boundary_delta_high']))>=2:
                reasons.append('normal_temporal_surface_scale')
        elif p['straight'] and all(q.get('surface_observed',False) for q in p['periods']):
            return ['insufficient_road_boundary_support'],'unconfirmed_width'
        else:return [],'surface_boundaries_unavailable_or_curved'
        return reasons,'extraction_fluctuation' if reasons else 'verified'

    def verify(self,records,controls):
        start=time.perf_counter();self.calibrate(controls)
        for i,row in enumerate(records):
            if not row.get('v2_publish',False):continue
            kind=row['change_typ'];self.counts[f'before_{kind}']+=1
            p=self.patch(from_wkt(row['axis_wkt']),max(row['width_bef'],row['width_aft']))
            reasons,state=self.reasons(p,kind)
            row.update(v2_publish_before_patch=True,v2_patch_state=state,v2_patch_reasons=';'.join(reasons),
                       v2_patch_probability_before=p['periods'][0]['p'],v2_patch_probability_after=p['periods'][1]['p'],
                       v2_patch_surface_before=p['periods'][0]['s'],v2_patch_surface_after=p['periods'][1]['s'],v2_patch_image_ncc=p['ncc'],
                       v2_patch_image_core_ncc=p['core_ncc'])
            if reasons:
                row.update(v2_publish=False,v2_precision_reason=state+':'+reasons[0])
                self.counts['extraction_fluctuation' if state=='extraction_fluctuation' else 'unconfirmed_width']+=1
                self.counts['primary_'+reasons[0]]+=1
                for reason in reasons:self.counts['veto_'+reason]+=1
            else:self.counts[f'after_{kind}']+=1
            if not reasons and state!='verified':self.counts['qa_unknown']+=1
            self.audit.append(dict(candidate=i,kind=kind,state=state,reasons=reasons,
                left=p['left'].tolist(),right=p['right'].tolist(),ncc=p['ncc'],core_ncc=p['core_ncc']))
        self.counts['verification_seconds']=time.perf_counter()-start
        self.counts['raster_window_reads']=sum(t.reads for t in self.tiles)
        return {'timing_patch_verification_seconds':self.counts['verification_seconds'],
                **{'v2_patch_'+k:v for k,v in self.counts.items()}}

    def write_audit(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        (directory/'patch_verification.json').write_text(json.dumps(dict(calibration=self.calibration,
            counts=dict(self.counts),candidates=self.audit),ensure_ascii=False,indent=2),encoding='utf-8')
