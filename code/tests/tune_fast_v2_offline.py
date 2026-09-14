"""Offline-only GT analysis. Never imported by the production detector.

Reads immutable Auto caches, not reconciled period roads. Detectability and all
metrics are tuning diagnostics, not replacements for official evaluation.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys
import time

import geopandas as gpd
import numpy as np
from shapely import from_wkt, union_all
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.fast_auto_change import RoadScene, WindowedProbability, finalize_auto_candidates
from engine.fast_auto_v2 import analyze_scenes, V2Config, qualify_candidates, presence_publication
from engine.fast_pipeline import _load_fast_period_result, _read_fast_change_layer
from engine.auto_scene_cache import close_scene, scene_key
from engine.auto_presence_candidates import LongitudinalCoverage
from engine.fast_gt_reconciliation import _gt_axes
from compare_fast_v2_pair import overlay_polygon

KINDS=('added','removed','widened','narrowed')
CONFIGS=dict(previous=V2Config(stable_position_factor=1.,presence_minimum_length=0.,width_minimum_length=0.,width_threshold_factor=1.,require_confirmed_absence=False),
             moderate=V2Config(stable_position_factor=1.5,presence_minimum_length=48.,width_minimum_length=48.,width_threshold_factor=1.5,source_surface_minimum=.8,opposite_surface_maximum=.1),
             conservative=V2Config(stable_position_factor=2.,presence_minimum_length=72.,width_minimum_length=72.,width_threshold_factor=2.,source_surface_minimum=.8,opposite_surface_maximum=.1),
             balanced=V2Config())


def gt_objects(pair,scenes,crs):
    truth=gpd.read_file(pair.get('truth_path') or pair['truth']).to_crs(crs)
    area=union_all(gpd.read_file(pair['validation_area']).to_crs(crs).geometry)
    covers=[LongitudinalCoverage(s.lines,5.) for s in scenes]
    rows=[]
    for index,r in truth.iterrows():
        kind={2:'added',3:'width_changed',4:'removed'}.get(int(r[pair.get('truth_type_field','BHBM')]))
        if not kind:continue
        geometry=overlay_polygon(r.geometry.intersection(area))
        if geometry.is_empty:continue
        axes=[v['axis'] for v in _gt_axes(geometry,8.)]
        length=sum(g.length for g in axes)
        coverage=[1-sum(b-a for g in axes for a,b in c.uncovered(g))/length if length else 0. for c in covers]
        required=[1] if kind=='added' else [0] if kind=='removed' else [0,1]
        minimum=min(coverage[i] for i in required)
        state='upstream_undetectable' if minimum<.1 else 'partial_upstream' if minimum<.5 else 'detectable'
        rows.append(dict(gt_id=str(index),change_typ=kind,before_axis_coverage=coverage[0],after_axis_coverage=coverage[1],
                         upstream_state=state,geometry=geometry))
    return gpd.GeoDataFrame(rows,geometry='geometry',crs=crs),area


def assess(frame,truth,area):
    result={};objects=[];area=overlay_polygon(area)
    # Exclude all predictions touching undetectable/partial GT from tuning FP
    # counts: missing upstream roads must not influence parameter selection.
    excluded=union_all(truth.loc[truth.upstream_state!='detectable'].geometry)
    for kind in KINDS:
        target=truth.loc[(truth.change_typ==('width_changed' if kind in ('widened','narrowed') else kind)) & (truth.upstream_state=='detectable')]
        gt=union_all(target.geometry)
        # Geographic export -> metric reprojection can produce microscopic ring
        # conflicts. Repair only evaluation copies, never published geometry.
        parts=[overlay_polygon(overlay_polygon(g).intersection(area)) for g in frame.loc[frame.change_typ==kind].geometry]
        parts=[p for p in parts if not p.is_empty]
        eligible=[p for p in parts if excluded.is_empty or not p.intersects(excluded)]
        overlaps=[p.intersection(gt).area for p in eligible]
        merged=union_all(parts)
        result[kind]=dict(count=len(parts),tuning_eligible_count=len(eligible),no_gt_overlap=sum(a<=1e-6 for a in overlaps),
            low_gt_overlap=sum(a/max(p.area,1e-9)<.1 for a,p in zip(overlaps,eligible)),
            matched_gt=sum(g.intersection(merged).area/max(g.area,1e-9)>=.1 for g in target.geometry),
            detectable_gt=len(target),gt_area_coverage=gt.intersection(merged).area/max(gt.area,1e-9),
            predicted_area_in_gt=sum(overlaps)/max(sum(p.area for p in eligible),1e-9))
        for i,r in target.iterrows():
            objects.append(dict(gt_id=r.gt_id,predicted_type=kind,area_coverage=r.geometry.intersection(merged).area/max(r.geometry.area,1e-9),
                nearest_prediction_m=min((r.geometry.distance(p) for p in parts),default=None)))
    return dict(types=result,objects=objects)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path)
    parser.add_argument('--finalize',choices=list(CONFIGS))
    args=parser.parse_args();root=args.job/'_profiling'/'fast_v2_tuning';root.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((args.job/'pipeline_result.json').read_text(encoding='utf-8'))
    periods=manifest['auto_period_results'] # Deliberately never fall back to corrected roads.
    report=[]
    for pair in manifest['change_results']:
        key=f"{pair['grid']}_{pair['before_period']}_{pair['after_period']}";output=root/key;output.mkdir(exist_ok=True)
        payloads=[_load_fast_period_result(next(p for p in periods if p['grid']==pair['grid'] and p['period']==pair[k])) for k in ('before_period','after_period')]
        centers=[_read_fast_change_layer(p,'centerlines') for p in payloads]
        crs=centers[0].estimate_utm_crs() if centers[0].crs.is_geographic else centers[0].crs
        fingerprints=[scene_key(p,crs) for p in payloads];scenes=[]
        try:
            for p,center in zip(payloads,centers):
                frames=[_read_fast_change_layer(p,k).to_crs(crs) for k in ('surfaces','width_segments','valid_observation')]
                scenes.append(RoadScene(center.to_crs(crs),*frames,WindowedProbability(p['road_probability'],crs),crs))
            truth_file=output/'gt_upstream.gpkg'
            if truth_file.exists():
                truth=gpd.read_file(truth_file);area=union_all(gpd.read_file(pair['validation_area']).to_crs(crs).geometry)
            else:
                truth,area=gt_objects(pair,scenes,crs);truth.to_file(truth_file,driver='GPKG')
            row=dict(pair=key,upstream=dict(Counter(truth.upstream_state)),variants={})
            for label,config in CONFIGS.items():
                if args.finalize and label not in ('previous',args.finalize):continue
                cache=output/f'{label}.pkl';identity=dict(inputs=fingerprints,config=asdict(config),version=1)
                if cache.exists():
                    saved=pickle.loads(cache.read_bytes())
                    old=saved['identity']
                    assert old['inputs']==identity['inputs'] and all(identity['config'][k]==v for k,v in old['config'].items()),'Tuning cache input/config changed'
                    result,presence,wall=saved['result'],saved['presence'],saved['wall']
                else:
                    print('TUNING',key,label,flush=True);presence=[];started=time.perf_counter()
                    result=analyze_scenes(*scenes,tolerance=float(pair['tolerance']),absolute=float(pair['absolute']),relative=float(pair['ratio']),presence_audit=presence,config=config)
                    wall=time.perf_counter()-started
                    cache.write_bytes(pickle.dumps(dict(identity=identity,result=result,presence=presence,wall=wall),protocol=5))
                # Cached generation evidence can be reused when tuning only the
                # publication gate. Exactly the production decision, with no GT.
                for record in result[0]:
                    if record['change_typ'] in ('added','removed'):
                        record['v2_publish'],record['v2_precision_reason']=presence_publication(record,config)
                frame=gpd.GeoDataFrame(result[0],geometry='geometry',crs=crs)
                qualified,_=qualify_candidates(frame,minimum_length=24.,minimum_area=4.)
                final=qualified.loc[qualified.publication_state=='accepted'].copy()
                metric=assess(final,truth,area);metric.update(wall_seconds=wall,raw_count=len(frame),internal_only=len(frame)-len(final))
                if args.finalize:
                    dest=output/label
                    if not (dest/'road_changes.shp').exists():
                        finalize_auto_candidates(*result,presence_audit=presence,scenes=dict(zip(('before','after'),scenes)),centerlines=centers,
                            output_dir=dest,before_period=pair['before_period'],after_period=pair['after_period'],
                            position_tolerance=float(pair['tolerance']),diagnostics=True,internal_outputs=True)
                    published=gpd.read_file(dest/'road_changes.shp').to_crs(crs)
                    metric['final']=assess(published,truth,area)
                row['variants'][label]=metric
                print('METRICS',key,label,json.dumps(metric['types']),flush=True)
            report.append(row)
            (root/('final_report.json' if args.finalize else 'tuning_report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        finally:
            for scene in scenes:close_scene(scene)


if __name__=='__main__':main()
