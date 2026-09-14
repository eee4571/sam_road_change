"""Post-hoc A/B evaluation. Refuses to open GT until both arms have finished."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import time
import re
from collections import Counter
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from shapely import make_valid, union_all, get_parts, line_interpolate_point, contains_xy, prepare, covers, intersects, clip_by_rect
from shapely.strtree import STRtree
from run_experiment import ROOT, REPO, PERIODS, save, sha, environment

KINDS = ('added','removed','widened','narrowed')
_REGION_CROPS = {}
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return dict(n=len(x), median=float(np.median(x)), mean=float(np.mean(x)),
        p90=float(np.percentile(x,90)), p95=float(np.percentile(x,95))) if len(x) else dict(n=0)

def sample_lines(geometries,step=2.):
    pts=[]
    for geom in geometries:
        for line in get_parts(geom):
            if line.geom_type!='LineString' or line.length<step: continue
            pts.extend(line_interpolate_point(line,np.arange(step/2,line.length,step)))
    return np.asarray(pts,dtype=object)

def nearest(points,frame):
    distance=np.full(len(points),np.nan); index=np.full(len(points),-1,dtype=int)
    if not len(points) or not len(frame): return index,distance
    pairs,d=STRtree(frame.geometry.to_numpy()).query_nearest(points,return_distance=True,all_matches=False)
    index[pairs[0]]=pairs[1];distance[pairs[0]]=d
    return index,distance

def line_comparison(first,second):
    result={}
    for name,a,b in [('before_to_after',first,second),('after_to_before',second,first)]:
        points=sample_lines(a.geometry)
        _,distance=nearest(points,b)
        result[name]=dict(offset_m=stats(distance), **{f'overlap_within_{m}m':float(np.mean(distance<=m)) for m in (1,2,3)})
    result['symmetric_overlap_3m']=float(np.mean([r['overlap_within_3m'] for r in result.values()]))
    return result

def clipped(frame,region):
    tick=time.perf_counter()
    frame=frame.copy()
    frame.geometry=make_valid(frame.geometry.to_numpy())
    # Avoid intersecting every tiny road with the complete, detailed NoData boundary.
    prepare(region)
    touching=intersects(region,frame.geometry.to_numpy())
    frame=frame.loc[touching].copy()
    inside=covers(region,frame.geometry.to_numpy())
    boundary=frame.index[~inside]
    cache=_REGION_CROPS.setdefault(id(region),{})
    def local_intersection(geom):
        left,bottom,right,top=geom.bounds
        key=tuple(int(np.floor(v/256)) for v in (left,bottom,right,top))
        if key not in cache:
            cache[key]=make_valid(clip_by_rect(region,key[0]*256-1e-5,key[1]*256-1e-5,
                (key[2]+1)*256+1e-5,(key[3]+1)*256+1e-5))
        return geom.intersection(cache[key])
    frame.loc[boundary,'geometry']=frame.loc[boundary].geometry.map(local_intersection)
    result=frame.loc[~frame.geometry.is_empty].reset_index(drop=True)
    print('Clipped',len(result),'features;',len(boundary),'boundary;',round(time.perf_counter()-tick,2),'seconds',flush=True)
    return result

def surface_iou(a,b):
    x,y=[union_all(make_valid(f.geometry.to_numpy())) for f in (a,b)]
    intersection=x.intersection(y).area
    return dict(iou=float(intersection/max(x.area+y.area-intersection,1e-9)),
        before_area_m2=float(x.area),after_area_m2=float(y.area),intersection_m2=float(intersection))

def raw_molra_overlap(periods):
    from PIL import Image
    rows=[]
    images=[{Path(r['image']).stem:r for r in read(Path(p['width_review'])/'batch_width_summary.json')['images']} for p in periods]
    sums={k:np.zeros(3,dtype=np.int64) for k in ('raw','enhanced')}
    for key in sorted(set(images[0])&set(images[1])):
        a,b=[v[key] for v in images]
        if not all(r.get('molra_surface_available') and not r.get('molra_surface_error') for r in (a,b)):
            raise RuntimeError(f'MoLRA missing or failed: {key}')
        # Output probability masks share the common production analysis grid.
        with rasterio.open(a['image']) as ds0, rasterio.open(b['image']) as ds1:
            assert ds0.crs==ds1.crs and ds0.transform==ds1.transform and ds0.shape==ds1.shape
            valid=(ds0.dataset_mask()>0)&(ds1.dataset_mask()>0)
        arrays={
            'raw':[np.load(r['molra_probability'],mmap_mode='r')>=.5 for r in (a,b)],
            'enhanced':[np.asarray(Image.open(r['molra_surface_mask']))>0 for r in (a,b)]}
        row=dict(tile=key)
        for kind,(x,y) in arrays.items():
            x=x&valid;y=y&valid
            local=np.array([np.count_nonzero(x&y),np.count_nonzero(x),np.count_nonzero(y)])
            sums[kind]+=local
            row[kind+'_iou']=float(local[0]/max(local[1]+local[2]-local[0],1))
        rows.append(row)
    return {**{k:dict(iou=float(v[0]/max(v[1]+v[2]-v[0],1)),intersection_pixels=int(v[0]),
        before_pixels=int(v[1]),after_pixels=int(v[2])) for k,v in sums.items()},'tiles':rows}

def main():
    started=time.perf_counter()
    os.environ.update(environment())
    cfg=read(ROOT/'config/experiment.json')
    manifests={}
    for arm in ('A','B'):
        assert read(ROOT/f'status_{arm}.json')['state']=='completed'
        m=read(ROOT/arm/'_work/tasks/runs/pair_ab/pipeline_result.json')
        assert not m.get('failures') and len(m['auto_period_results'])==2
        assert not m['input_spec']['truths'] and not m['input_spec']['evaluation_enabled']
        manifests[arm]=m
    result=dict(method=dict(sample_spacing_m=2,matching_tolerance_m=3,width_sign='after minus before',
        evaluation_extent='validation area intersected with valid observations of all four products',
        no_gt_intersection='nonempty prediction within evaluation extent, intersection area with ALL change GT <= 1e-6 m2',
        limitation='No road-extraction GT; retained length and overlap are proxies, not extraction recall.'), arms={})
    payload={arm:[next(p for p in m['auto_period_results'] if p['period']==period) for period in PERIODS] for arm,m in manifests.items()}
    crs=gpd.read_file(payload['A'][0]['centerlines']).estimate_utm_crs()
    region=union_all(gpd.read_file(cfg['validation_area']).to_crs(crs).geometry)
    print('Loading common valid region',flush=True)
    for periods in payload.values():
        for p in periods: region=region.intersection(union_all(gpd.read_file(p['valid_observation']).to_crs(crs).geometry))
    result['evaluation_area_m2']=float(region.area)
    print('Loading and clipping road layers',flush=True)
    layers={}
    for arm,periods in payload.items():
        layers[arm]=[]
        for p in periods:
            layers[arm].append({key:clipped(gpd.read_file(p[key]).to_crs(crs),region) for key in ('centerlines','surfaces','width_segments')})
            print('Layers ready',arm,p['period'],flush=True)
    for arm,periods in payload.items():
        before,after=layers[arm]
        row=dict(centerlines=line_comparison(before['centerlines'],after['centerlines']),
                 surface=surface_iou(before['surfaces'],after['surfaces']))
        row['molra_surface']=raw_molra_overlap(periods)
        row['periods']={}
        for period,p,f in zip(PERIODS,periods,layers[arm]):
            s=read(Path(p['width_review'])/'batch_width_summary.json')
            inference_text=(Path(p['run_root'])/'inference/road_graphs/grid_tiles/inference_time.txt').read_text(encoding='utf-8')
            inference_seconds=float(re.search(r'with ([0-9.]+) recorded seconds',inference_text).group(1))
            row['periods'][period]=dict(centerline_length_m=float(f['centerlines'].length.sum()),
                centerline_count=len(f['centerlines']),width=stats(f['width_segments']['width_m']),
                timings=p['stage_timings'],molra_seconds=sum(r.get('molra_surface_seconds',0) for r in s['images']),
                inference_recorded_seconds=inference_seconds,inference_timing_record=inference_text.strip(),
                molra_tile_count=len(s['images']),molra_failures=[r.get('molra_surface_error') for r in s['images'] if r.get('molra_surface_error')],
                molra_cache_hits=sum(bool(r.get('molra_surface_cache_hit')) for r in s['images']),
                molra_width_ratio=s.get('enhanced_molra_width_ratio'),
                raw_molra_centerline_coverage=s.get('raw_molra_centerline_coverage'))
        auto=read(ROOT/arm/'frozen_fast2/result.json')
        assert not auto['ground_truth_used']
        row['fast2']=dict(counts={k:auto.get(k+'_feature_count',0) for k in KINDS},
                          performance=auto.get('performance',{}),elapsed_seconds=auto.get('auto_change_total_seconds'),
                          path=auto['road_changes'])
        row['wall_seconds']=read(ROOT/f'status_{arm}.json')['elapsed_seconds']
        result['arms'][arm]=row
        save(ROOT/'metrics_progress.json',result)
        print('METRICS',arm,flush=True)
    # Width comparisons use fixed A-reference stations surviving in all four outputs.
    anchors=[];road_ids=[];road_lengths=[]
    for road_id,geom in enumerate(layers['A'][0]['centerlines'].geometry):
        pts=sample_lines([geom])
        anchors.extend(pts);road_ids.extend([road_id]*len(pts));road_lengths.extend([geom.length]*len(pts))
    points=np.asarray(anchors,dtype=object)
    values={};valid=np.ones(len(points),dtype=bool)
    for arm in ('A','B'):
        values[arm]=[]
        for f in layers[arm]:
            frame=f['width_segments']
            idx,d=nearest(points,frame)
            width=frame['width_m'].to_numpy()[np.maximum(idx,0)]
            valid &= (idx>=0)&(d<=3)&np.isfinite(width)&(width>0)
            values[arm].append(width)
    width_table=pd.DataFrame({'x':[p.x for p in points[valid]],'y':[p.y for p in points[valid]],
        'reference_road_id':np.asarray(road_ids)[valid],'reference_road_length_m':np.asarray(road_lengths)[valid]})
    result['reference_repeatability']=dict(width_delta_B_minus_A_m=stats((values['B'][0]-values['A'][0])[valid]),
        absolute_width_delta_m=stats(abs(values['B'][0]-values['A'][0])[valid]),
        note='Same reference image independently inferred in both arms; see reference probability and MoLRA numerical checks.')
    for arm in ('A','B'):
        delta=(values[arm][1]-values[arm][0])[valid]
        width_table[arm+'_delta_m']=delta
        result['arms'][arm]['paired_width']=dict(delta_m=stats(delta),absolute_delta_m=stats(abs(delta)),
            fraction_abs_gt_2m=float(np.mean(abs(delta)>2)),fraction_positive_gt_2m=float(np.mean(delta>2)),
            fraction_negative_lt_minus2m=float(np.mean(delta<-2)))
    (ROOT/'evaluation').mkdir(exist_ok=True)
    width_table.to_csv(ROOT/'evaluation/fixed_station_width_differences.csv',index=False)
    roads=width_table.groupby('reference_road_id').agg(
        sample_count=('A_delta_m','count'),reference_length_m=('reference_road_length_m','first'),
        A_median_delta_m=('A_delta_m','median'),B_median_delta_m=('B_delta_m','median'))
    roads=roads.loc[(roads.reference_length_m>=32)&(roads.sample_count*2>=roads.reference_length_m*.7)]
    roads.to_csv(ROOT/'evaluation/long_road_width_differences.csv')
    for arm in ('A','B'):
        delta=roads[arm+'_median_delta_m'].to_numpy()
        result['arms'][arm]['long_road_width']=dict(eligible_roads=len(roads),median_delta_m=stats(delta),
            median_absolute_delta_m=stats(abs(delta)),whole_road_abs_delta_gt_2m=int(np.sum(abs(delta)>2)),
            definition='A-reference features >=32m, >=70% fixed stations matched within 3m in all four outputs; median per feature')
    result['extraction_retention']={period:dict(
        comparison=line_comparison(layers['A'][i]['centerlines'],layers['B'][i]['centerlines']),
        length_ratio_B_over_A=float(layers['B'][i]['centerlines'].length.sum()/layers['A'][i]['centerlines'].length.sum())) for i,period in enumerate(PERIODS)}
    # Freeze predictions before opening GT, then evaluate with the same GT for both arms.
    save(ROOT/'evaluation/pre_gt_boundary.json',dict(time=time.time(),prediction_sha256={arm:sha(result['arms'][arm]['fast2']['path']) for arm in ('A','B')}))
    original_cfg=read(REPO/'project/test_area/project_config.json')
    truth_path=Path(next(r[3] for r in original_cfg['area_truths'] if r[1:3]==list(PERIODS)))
    truth_dir=ROOT/'evaluation/gt';truth_dir.mkdir(exist_ok=True)
    for file in truth_path.parent.glob(truth_path.stem+'.*'):
        if file.suffix.lower() in {'.shp','.shx','.dbf','.prj','.cpg'}: shutil.copy2(file,truth_dir/('changes'+file.suffix))
    gt=clipped(gpd.read_file(truth_dir/'changes.shp').to_crs(crs),region)
    gt['evaluation_type']=pd.to_numeric(gt['BHBM'],errors='coerce').map({2:'added',3:'width_changed',4:'removed'})
    gt=gt.loc[gt.evaluation_type.notna()].reset_index(drop=True)
    assert len(gt), 'No evaluable BHBM 2/3/4 change truth'
    result['gt_type_counts']=dict(Counter(gt.evaluation_type))
    gt_union=union_all(gt.geometry)
    gt_rows=[]
    for arm in ('A','B'):
        pred=clipped(gpd.read_file(result['arms'][arm]['fast2']['path']).to_crs(crs),region)
        no=Counter();counts=Counter();areas=Counter();no_areas=Counter();zero=[]
        for i,r in pred.iterrows():
            overlap=r.geometry.intersection(gt_union).area
            counts[r.change_typ]+=1
            areas[r.change_typ]+=r.geometry.area
            if overlap<=1e-6:
                no[r.change_typ]+=1;no_areas[r.change_typ]+=r.geometry.area;zero.append(i)
            gt_rows.append(dict(arm=arm,object_id=i,kind=r.change_typ,area_m2=r.geometry.area,
                gt_intersection_m2=overlap,length_m=r.get('length_m',None)))
        merged=union_all(pred.geometry)
        cover=[float(g.intersection(merged).area/max(g.area,1e-9)) for g in gt.geometry]
        result['arms'][arm]['offline_gt']=dict(no_intersection={k:no[k] for k in KINDS},
            no_intersection_area_m2={k:no_areas[k] for k in KINDS},
            counts_in_extent={k:counts[k] for k in KINDS},predicted_area_m2=dict(areas),gt_object_count=len(gt),
            gt_objects_covered_at_least_10pct=sum(v>=.1 for v in cover),gt_object_area_coverage=cover,
            gt_area_coverage=float(gt_union.intersection(merged).area/max(gt_union.area,1e-9)))
        typed={}
        for kind in ('added','removed','width_changed'):
            target=gt.loc[gt.evaluation_type==kind]
            prediction=pred.loc[pred.change_typ.isin(['widened','narrowed'] if kind=='width_changed' else [kind])]
            support=union_all(prediction.geometry)
            coverage=[float(g.intersection(support).area/max(g.area,1e-9)) for g in target.geometry]
            typed[kind]=dict(gt_count=len(target),matched_at_10pct=sum(x>=.1 for x in coverage),area_coverage=coverage)
        result['arms'][arm]['offline_gt']['typed_gt_coverage']=typed
        outside=~contains_xy(gt_union.buffer(3.),width_table.x.to_numpy(),width_table.y.to_numpy())
        delta=width_table.loc[outside,arm+'_delta_m'].to_numpy()
        result['arms'][arm]['offline_gt']['width_outside_gt_buffer_3m']=dict(delta_m=stats(delta),
            absolute_delta_m=stats(abs(delta)),fraction_abs_gt_2m=float(np.mean(abs(delta)>2)))
        if zero: pred.loc[zero].to_file(ROOT/f'evaluation/{arm}_no_gt_intersection.gpkg',driver='GPKG')
    pd.DataFrame(gt_rows).to_csv(ROOT/'evaluation/object_gt_intersections.csv',index=False)
    identities=read(ROOT/'config/code_identity.json')
    result['integrity']=dict(concurrently_changed_production_files=[p for p,h in identities.items() if sha(REPO/p)!=h],
        fast2_comparison_uses_frozen_snapshot=True, snapshot_commit='734fe17cada16effc181336bfe1b18ad5e75daad',
        original_images_unchanged=all(sha(s['original'])==s['sha256'] for s in cfg['sources'].values()),
        reference_copies_identical=sha(ROOT/'inputs/raw/20250118.tif')==sha(ROOT/'inputs/normalized/20250118.tif'),
        model_files_unchanged=all(sha(p)==v['sha256'] for p,v in read(ROOT/'config/models_identity.json').items()))
    result['evaluation_seconds']=time.perf_counter()-started
    save(ROOT/'metrics.json',result)
    print('EVALUATION COMPLETE',flush=True)

if __name__=='__main__': main()
