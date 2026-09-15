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
from .fast2_compensation import Fast2Compensation
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
    def __init__(self,scenes,payloads,maximum_controls=96,*,compensation=None):
        self.compensation=Fast2Compensation(compensation)
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
            probability,surface=self.compensation.surface_probability.apply(probability,surface)
            valid=np.isfinite(probability)
            pcore=quantile(probability[core&valid],.5);pbg=quantile(probability[flank&valid],.5)
            surface_valid=np.isfinite(surface)
            support=float(np.mean(surface[core&surface_valid])) if np.count_nonzero(core&surface_valid)>8 else np.nan
            periods.append(dict(p=pcore,contrast=pcore-pbg,s=support,rgb=rgb,surface=surface,probability=probability))
        tick=time.perf_counter()
        features=image_features(periods[0]['rgb'],periods[1]['rgb'],core,ring,lateral,bins,normal,width,resolution,
                                radiometric=self.compensation.patch)
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
        self.calibration=self.compensation.appearance.fit(c)
        self.counts['calibration_seconds']=time.perf_counter()-started

    def reasons(self,p,kind):
        """Only a conjunction of positive stability evidence can veto.

        Existing descriptors and calibration are unchanged. Failure of a change
        test is an uncertainty reason, never evidence for the stable hypothesis.
        """
        c=self.calibration
        if not p['valid']:return ['raw_image_unavailable_or_invalid'],'uncertain'
        if c.get('count',0)<c.get('minimum',12):
            return ['insufficient_stable_image_controls'],'uncertain'
        required=('score_delta','anomaly','ncc','ssim','hog_distance','left','right')
        if any(c.get(k+'_count',0)<12 or not np.isfinite(p[k]).all() for k in required):
            return ['uninformative_raw_image_or_boundaries'],'uncertain'
        registration=p.get('registration',{})
        proposed=np.asarray(registration.get('proposed_shift_px',[np.nan,np.nan]))
        # align_pair does not mark an identity transform accepted if NCC cannot
        # improve. A subpixel, high-response identity is still a valid alignment.
        reliable=(registration.get('response',0.)>.1 and
            (registration.get('accepted',False) or
             (np.isfinite(proposed).all() and np.linalg.norm(proposed)<=.5)))
        if not reliable:return ['registration_uncertain'],'uncertain'
        supported=all(period['road_score']>max(0.,c[f'{side}_road_low'])
                      for side,period in enumerate(p['periods']))
        high_similarity=(p['ncc']>=max(0.,c['ncc_median']) and
                         p['ssim']>=max(0.,c['ssim_median']) and
                         p['hog_distance']<=c['hog_distance_median'])
        l=p['left']-c['left_median'];r=p['right']-c['right_median']
        stable_limits=[max(p['resolution'],1.4826*c[k+'_mad']) for k in ('left','right')]
        stable_left=np.abs(l)<=stable_limits[0];stable_right=np.abs(r)<=stable_limits[1]
        both_stable=np.count_nonzero(stable_left&stable_right)>=2
        if kind in ('added','removed'):
            if supported and high_similarity and both_stable:
                return ['strong_persistent_raw_road'],'strong_stable'
            source=1 if kind=='added' else 0;direction=1 if source else -1
            source_present=p['periods'][source]['road_score']>max(0.,c[f'{source}_road_low'])
            delta=direction*(p['score_delta']-c['score_delta_median'])
            bound=(c['score_delta_high']-c['score_delta_median'] if source else
                   c['score_delta_median']-c['score_delta_low'])
            directional_change=delta>max(bound,1.4826*c['score_delta_mad'])
            anomaly=p['anomaly']>max(c['anomaly_high'],c['anomaly_median']+1.4826*c['anomaly_mad'])
            if source_present and directional_change and anomaly:
                return ['raw_structure_appearance' if source else 'raw_structure_disappearance'],'strong_change'
            qa=[]
            if not supported:qa.append('raw_parallel_boundaries_not_supported')
            if not high_similarity:qa.append('mixed_raw_similarity')
            if not both_stable:qa.append('incomplete_stable_boundaries')
            if not directional_change:qa.append('no_directional_structure_appearance')
            if not anomaly:qa.append('normal_background_relative_change')
            return qa or ['mixed_raw_evidence'],'uncertain'
        if not p.get('width_geometry_reliable',True):
            return ['width_curvature_or_resolution_uncertain'],'uncertain'
        if not supported:return ['raw_parallel_boundaries_not_supported'],'uncertain'
        if both_stable and high_similarity:
            return ['strong_stable_raw_width_boundaries'],'strong_stable'
        # l/r are outward movements, so opposite signs represent a common
        # signed lateral displacement. Both must exceed measurement uncertainty.
        displaced=(l*r<0)&(np.abs(l)>stable_limits[0])&(np.abs(r)>stable_limits[1])
        displaced &= np.abs(l+r)<=max(p['resolution'],min(stable_limits))
        if np.count_nonzero(displaced)>=2:
            return ['raw_boundary_lateral_displacement'],'strong_stable'
        direction=1 if kind=='widened' else -1
        limits=[max(c[k+'_high']-c[k+'_median'] if direction>0 else c[k+'_median']-c[k+'_low'],
                    1.4826*c[k+'_mad'],p['resolution']) for k in ('left','right')]
        moved_left=direction*l>limits[0];moved_right=direction*r>limits[1]
        sustained=(moved_left&(stable_right|moved_right))|(moved_right&(stable_left|moved_left))
        if np.count_nonzero(sustained)>=2:
            return ['raw_one_or_two_sided_width_change'],'strong_change'
        return ['raw_width_evidence_inconclusive'],'uncertain'

    def verify(self,records,controls,*,profiles=None,width_audit=None,absolute=2.,relative=.2):
        from .fast_candidate_publication import CandidateEvidence
        self.fast2_evidence=CandidateEvidence(self.scenes,profiles,width_audit,absolute,relative)
        self.absolute=absolute;self.relative=relative
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
        # Geometry-only QA uses existing candidate axis/width, no new image
        # descriptor or remeasurement. Unresolved narrow/curved strips stay QA.
        if kind in ('widened','narrowed'):
            axis=from_wkt(row['axis_wkt']);coords=np.asarray(axis.coords)[:,:2]
            positive=[w for w in (row['width_bef'],row['width_aft']) if np.isfinite(w) and w>0]
            p['width_geometry_reliable']=bool(positive and min(positive)>=4*p['resolution'] and
                np.linalg.norm(coords[-1]-coords[0])>=.95*axis.length)
        original_publish=bool(row.get('v2_publish',False))
        original_reason=row.get('v2_precision_reason','')
        reasons,state=self.reasons(p,kind)
        row.update(v2_publish_before_patch=original_publish,v2_precision_reason_before_patch=original_reason,
                   v2_patch_state=state,v2_patch_reasons=';'.join(reasons),
                   v2_patch_probability_before=p['periods'][0]['p'],v2_patch_probability_after=p['periods'][1]['p'],
                   v2_patch_surface_before=p['periods'][0]['s'],v2_patch_surface_after=p['periods'][1]['s'],v2_patch_image_ncc=p['ncc'],
                   v2_patch_image_core_ncc=p['core_ncc'])
        from .fast_candidate_publication import grade_candidate
        facts=self.fast2_evidence.facts(row) if hasattr(self,'fast2_evidence') else {}
        level,conditions,metrics=grade_candidate(row,state,facts,
            absolute=getattr(self,'absolute',2.),relative=getattr(self,'relative',.2))
        row.update(v2_publication_level=level,v2_confidence_conditions=';'.join(conditions))
        self.counts['level_'+level]+=1
        if state=='uncertain':
            for condition in conditions:self.counts['condition_'+condition]+=1
            if conditions:self.counts['primary_condition_'+conditions[0]]+=1
        self.counts[state]+=1
        if state=='strong_stable':
            row.update(v2_publish=False,v2_precision_reason=state+':'+reasons[0])
            self.counts['primary_'+reasons[0]]+=1
            for reason in reasons:self.counts['veto_'+reason]+=1
        else:
            if level=='Candidate':
                row.update(v2_publish=False,v2_precision_reason='candidate_confidence:'+(';'.join(conditions)))
            if row.get('v2_publish',False):self.counts[f'after_{kind}']+=1
            if state=='uncertain':
                self.counts['qa_unknown']+=1
                for reason in reasons:self.counts['qa_'+reason]+=1
        self.audit.append(dict(candidate=i,kind=kind,state=state,reasons=reasons,
            left=p['left'].tolist(),right=p['right'].tolist(),ncc=p['ncc'],core_ncc=p['core_ncc'],
            ssim=p['ssim'],ring_ssim=p['ring_ssim'],anomaly=p['anomaly'],hog_distance=p['hog_distance'],
            road_scores=[r['road_score'] for r in p['periods']],score_delta=p['score_delta'],
            registration=p['registration'],resolution=p['resolution'],valid=p['valid'],
            width_geometry_reliable=p.get('width_geometry_reliable'),
            published=bool(row.get('v2_publish',False)),publication_level=level,
            confidence_conditions=conditions,fast2_evidence=metrics,
            candidate_geometry_wkb=row['geometry'].wkb_hex if level=='Candidate' and 'geometry' in row else None,
            candidate_axis_wkt=row.get('axis_wkt') if level=='Candidate' else None,
            candidate_widths=[row.get('width_bef'),row.get('width_aft')] if level=='Candidate' else None))

    def write_audit(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        (directory/'patch_verification.json').write_text(json.dumps(dict(compensation=self.compensation.metadata(),calibration=self.calibration,
            counts=dict(self.counts),candidates=self.audit),ensure_ascii=False,indent=2),encoding='utf-8')
