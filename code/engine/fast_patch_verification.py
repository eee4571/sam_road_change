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
from scipy.ndimage import label
from .fast_image_structure import axis_grid,image_features
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
        ring=~outer
        lateral,bins,normal=axis_grid(axis,xy)
        periods=[]
        for scene,tiles in zip(self.scenes,self.tiles):
            rgb,surface=tiles.read(bounds,shape,transform)
            probability=scene.probability._values_at(xy[...,0].ravel(),xy[...,1].ravel()).reshape(shape)
            valid=np.isfinite(probability)
            pcore=quantile(probability[core&valid],.5);pbg=quantile(probability[flank&valid],.5)
            surface_valid=np.isfinite(surface)
            support=float(np.mean(surface[core&surface_valid])) if np.count_nonzero(core&surface_valid)>8 else np.nan
            periods.append(dict(p=pcore,contrast=pcore-pbg,s=support,rgb=rgb,surface=surface,probability=probability))
        tick=time.perf_counter()
        features=image_features(periods[0]['rgb'],periods[1]['rgb'],core,ring,lateral,bins,normal,width,resolution)
        for model,raw in zip(periods,features.pop('periods')):model.update(raw)
        self.counts['raw_image_features_seconds']+=time.perf_counter()-tick
        self.counts['registration_accepted']+=int(features['registration']['accepted'])
        sufficient=all(scene.valid.covers(axis.buffer(width/2,cap_style='flat')) for scene in self.scenes)
        self.counts['patch_count']+=1;self.counts['patch_pixels']+=core.size
        return dict(periods=periods,**features,resolution=resolution,
                    valid=sufficient and features['image_valid'],bounds=bounds,core=core,ring=ring)

    def calibrate(self,controls):
        started=time.perf_counter()
        ids=np.linspace(0,len(controls)-1,min(len(controls),self.maximum_controls),dtype=int) if controls else []
        c=[]
        for i in ids:
            row=controls[i];p=self.patch(row['axis'],row['width'])
            if p['valid']:
                # Retain descriptors only, not an image queue for all controls.
                c.append({k:p[k] for k in ('score_delta','anomaly','ncc','ssim','hog_distance','left','right')} |
                         {'scores':[period['road_score'] for period in p['periods']]})
        self.calibration={'count':len(c),'minimum':12,'evidence':'raw_image_primary',
                          'encoder_features':'unavailable_skipped'}
        for name in ('score_delta','anomaly','ncc','ssim','hog_distance','left','right'):
            values=np.asarray([quantile(p[name],.5) if name in ('left','right') else p[name] for p in c])
            median=quantile(values,.5);mad=quantile(np.abs(values-median),.5)
            self.calibration.update({name+'_median':median,name+'_mad':mad,
                name+'_low':quantile(values,.05),name+'_high':quantile(values,.95),
                name+'_count':int(np.isfinite(values).sum())})
        for side in range(2):
            values=[p['scores'][side] for p in c]
            self.calibration[f'{side}_road_low']=quantile(values,.1)
        self.counts['calibration_seconds']=time.perf_counter()-started

    def reasons(self,p,kind):
        c=self.calibration
        if not p['valid']:return ['raw_image_unavailable_or_invalid'],'unconfirmed_image'
        if c.get('count',0)<c.get('minimum',12):
            return ['insufficient_stable_image_controls'],'unconfirmed_image'
        required=('score_delta','anomaly','ncc','ssim','hog_distance')
        if any(c.get(k+'_count',0)<12 or not np.isfinite(p[k]) for k in required):
            return ['uninformative_raw_image'],'unconfirmed_image'
        a,b=p['periods'];reasons=[]
        if kind in ('added','removed'):
            source=1 if kind=='added' else 0;direction=1 if source==1 else -1
            source_present=p['periods'][source]['road_score']>max(0.,c[f'{source}_road_low'])
            target_present=p['periods'][1-source]['road_score']>max(0.,c[f'{1-source}_road_low'])
            same=(p['ncc']>=c['ncc_low'] and p['ssim']>=c['ssim_low'] and
                  p['hog_distance']<=c['hog_distance_high'])
            # Contrast reversal can destroy NCC/SSIM while both road boundaries
            # persist. Compare their positions independently of road brightness.
            stable_edges=np.ones(3,dtype=bool)
            for side in ('left','right'):
                stable_edges &= (np.isfinite(p[side]) &
                    (p[side]>=min(c[side+'_low'],-p['resolution'])) &
                    (p[side]<=max(c[side+'_high'],p['resolution'])))
            if source_present and target_present and np.count_nonzero(stable_edges)>=2:
                reasons.append('persistent_raw_parallel_boundaries')
            if target_present and same:reasons.append('persistent_raw_road_structure')
            delta=direction*(p['score_delta']-c['score_delta_median'])
            bound=(c['score_delta_high']-c['score_delta_median'] if source else
                   c['score_delta_median']-c['score_delta_low'])
            if not source_present:reasons.append('raw_parallel_boundaries_not_supported')
            if delta<=max(bound,1.4826*c['score_delta_mad']):reasons.append('no_directional_structure_appearance')
            if p['anomaly']<=max(c['anomaly_high'],c['anomaly_median']+1.4826*c['anomaly_mad']):
                reasons.append('normal_background_relative_change')
        else:
            if (not np.isfinite(p['left']).all() or not np.isfinite(p['right']).all() or
                any(c.get(k+'_count',0)<12 for k in ('left','right'))):
                return ['raw_boundaries_unavailable'],'unconfirmed_width'
            direction=1 if kind=='widened' else -1
            l=p['left']-c['left_median'];r=p['right']-c['right_median']
            limits=[max(c[k+'_high']-c[k+'_median'] if direction>0 else c[k+'_median']-c[k+'_low'],
                        1.4826*c[k+'_mad'],p['resolution']) for k in ('left','right')]
            supported=all(period['road_score']>max(0.,c[f'{side}_road_low']) for side,period in enumerate((a,b)))
            if not supported:reasons.append('raw_parallel_boundaries_not_supported')
            if np.count_nonzero((l*r<0)&(np.abs(l+r)<=2*p['resolution']))>=2:
                reasons.append('raw_boundary_lateral_displacement')
            if np.count_nonzero((direction*l>limits[0])&(direction*r>limits[1]))<2:
                reasons.append('raw_bilateral_change_not_sustained')
        return reasons,'extraction_fluctuation' if reasons else 'verified'

    def verify(self,records,controls):
        start=time.perf_counter();self.calibrate(controls)
        for i,row in enumerate(records):
            if not row.get('v2_publish',False):continue
            kind=row['change_typ'];self.counts[f'before_{kind}']+=1
            p=self.patch(from_wkt(row['axis_wkt']),max(row['width_bef'],row['width_aft']))
            self.record_decision(row,p,i)
        self.counts['verification_seconds']=time.perf_counter()-start
        self.counts['raster_window_reads']=sum(t.reads for t in self.tiles)
        return {'timing_patch_verification_seconds':self.counts['verification_seconds'],
                **{'v2_patch_'+k:v for k,v in self.counts.items()}}

    def record_decision(self,row,p,i):
        kind=row['change_typ']
        reasons,state=self.reasons(p,kind)
        row.update(v2_publish_before_patch=True,v2_patch_state=state,v2_patch_reasons=';'.join(reasons),
                   v2_patch_probability_before=p['periods'][0]['p'],v2_patch_probability_after=p['periods'][1]['p'],
                   v2_patch_surface_before=p['periods'][0]['s'],v2_patch_surface_after=p['periods'][1]['s'],v2_patch_image_ncc=p['ncc'],
                   v2_patch_image_core_ncc=p['core_ncc'])
        if reasons:
            row.update(v2_publish=False,v2_precision_reason=state+':'+reasons[0])
            self.counts[state]+=1
            self.counts['primary_'+reasons[0]]+=1
            for reason in reasons:self.counts['veto_'+reason]+=1
        else:self.counts[f'after_{kind}']+=1
        if not reasons and state!='verified':self.counts['qa_unknown']+=1
        self.audit.append(dict(candidate=i,kind=kind,state=state,reasons=reasons,
            left=p['left'].tolist(),right=p['right'].tolist(),ncc=p['ncc'],core_ncc=p['core_ncc'],
            ssim=p['ssim'],ring_ssim=p['ring_ssim'],anomaly=p['anomaly'],hog_distance=p['hog_distance'],
            road_scores=[r['road_score'] for r in p['periods']],score_delta=p['score_delta'],
            registration=p['registration'],resolution=p['resolution'],valid=p['valid']))

    def write_audit(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        (directory/'patch_verification.json').write_text(json.dumps(dict(calibration=self.calibration,
            counts=dict(self.counts),candidates=self.audit),ensure_ascii=False,indent=2),encoding='utf-8')
