"""Cached real-pair experiment. Truth is opened only after Auto has finished."""
import argparse
import json
import pickle
import time
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import geopandas as gpd
from shapely import union_all,make_valid
from tune_fast_v2_offline import (RoadScene,WindowedProbability,_load_fast_period_result,
    _read_fast_change_layer,scene_key,close_scene,assess,finalize_auto_candidates)
from engine import fast_auto_change as legacy
from engine.fast_auto_v2 import analyze_scenes
from engine.fast_multitemporal import neighbor_results,load_contexts
from engine.fast_patch_verification import PatchVerifier


def no_intersection(frame,truth,area):
    gt=union_all(make_valid(truth.geometry.to_numpy()))
    answer=Counter()
    for row in frame.itertuples():
        geom=make_valid(row.geometry).intersection(area)
        if not geom.is_empty and geom.intersection(gt).area<=1e-6:answer[row.change_typ]+=1
    return dict(answer)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);parser.add_argument('--limit',type=int,default=6)
    parser.add_argument('--output-name',default='fast2_patch_verification')
    parser.add_argument('--baseline-name',default='fast2_multitemporal');args=parser.parse_args()
    tuning=args.job/'_profiling'/'fast_v2_tuning'
    previous=args.job/'_profiling'/args.baseline_name
    output=args.job/'_profiling'/args.output_name;output.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((args.job/'pipeline_result.json').read_text(encoding='utf-8'))
    baseline={r['pair']:r for r in json.loads((previous/'comparison.json').read_text(encoding='utf-8'))}
    report=[];batch=time.perf_counter()
    for pair in manifest['change_results'][:args.limit]:
        key=f"{pair['grid']}_{pair['before_period']}_{pair['after_period']}";dest=output/key;dest.mkdir(exist_ok=True)
        started=time.perf_counter();scenes=[];verifier=None
        payloads=[_load_fast_period_result(next(p for p in manifest['auto_period_results'] if p['grid']==pair['grid'] and p['period']==pair[k]))
                  for k in ('before_period','after_period')]
        centers=[_read_fast_change_layer(p,'centerlines') for p in payloads]
        crs=centers[0].estimate_utm_crs() if centers[0].crs.is_geographic else centers[0].crs
        saved=pickle.loads((tuning/key/'balanced.pkl').read_bytes())
        assert saved['identity']['inputs']==[scene_key(p,crs) for p in payloads],'Original Auto inputs changed'
        try:
            for p,c in zip(payloads,centers):
                frames=[_read_fast_change_layer(p,k).to_crs(crs) for k in ('surfaces','width_segments','valid_observation')]
                scenes.append(RoadScene(c.to_crs(crs),*frames,WindowedProbability(p['road_probability'],crs),crs))
            loaded=time.perf_counter();names=sorted(p['period'] for p in manifest['auto_period_results'] if p['grid']==pair['grid'])
            neighbors=neighbor_results(manifest['auto_period_results'],pair['grid'],pair['before_period'],pair['after_period'],names)
            context=load_contexts(neighbors,crs,float(pair['tolerance'])*1.5)
            context_seconds=time.perf_counter()-loaded;presence=[];verifier=PatchVerifier(scenes,payloads);tick=time.perf_counter()
            with patch.object(legacy,'analyze_scenes',side_effect=AssertionError('Fast1 fallback')), \
                 patch.object(RoadScene,'match',side_effect=AssertionError('point match')), \
                 patch.object(legacy,'_measure_period_width',side_effect=AssertionError('remeasurement')):
                result=analyze_scenes(*scenes,tolerance=float(pair['tolerance']),absolute=float(pair['absolute']),
                                     relative=float(pair['ratio']),presence_audit=presence,temporal_context=context,patch_verifier=verifier)
            analysis=time.perf_counter()-tick;tick=time.perf_counter()
            verifier.write_audit(dest)
            (dest/'analysis.pkl').write_bytes(pickle.dumps((result,presence),protocol=5))
            finalize_auto_candidates(*result,presence_audit=presence,scenes=dict(zip(('before','after'),scenes)),centerlines=centers,
                output_dir=dest,before_period=pair['before_period'],after_period=pair['after_period'],
                position_tolerance=float(pair['tolerance']),diagnostics=True,internal_outputs=True)
            export=time.perf_counter()-tick
            processing_total=time.perf_counter()-started
            # Evaluation boundary: no GT is provided to the above analyzer.
            truth=gpd.read_file(tuning/key/'gt_upstream.gpkg')
            area=union_all(gpd.read_file(pair['validation_area']).to_crs(crs).geometry)
            native=gpd.read_file(dest/'road_changes.shp')
            current=native.to_crs(crs)
            old=gpd.read_file(previous/key/'road_changes.shp').to_crs(crs)
            gt=union_all(make_valid(truth.geometry.to_numpy()))
            suppressed=[dict(candidate_index=i,change_typ=r['change_typ'],removed_length_m=r['v2_temporal_removed_length_m'],
                             gt_intersection_m2=make_valid(r['geometry']).intersection(gt).area,geometry=r['geometry'])
                        for i,r in enumerate(result[0]) if r.get('v2_temporal_removed_length_m',0)>0]
            if suppressed:gpd.GeoDataFrame(suppressed,geometry='geometry',crs=crs).to_file(dest/'temporal_suppression_audit.gpkg',driver='GPKG')
            row=dict(pair=key,before=baseline[key]['after'],after=assess(current,truth,area),
                before_no_gt_intersection=no_intersection(old,truth,area),after_no_gt_intersection=no_intersection(current,truth,area),
                baseline_analysis_seconds=baseline[key]['analysis_seconds'],analysis_seconds=analysis,
                scene_load_seconds=loaded-started,context_load_seconds=context_seconds,finalization_seconds=export,
                total_seconds=processing_total,counts=result[3],
                neighbors=list(neighbors),suppressed_candidates_with_gt_overlap=sum(r['gt_intersection_m2']>1e-6 for r in suppressed),
                upstream_gt=dict(Counter(truth.upstream_state)),valid_geometry=bool(native.is_valid.all()),
                invalid_native_geometry_count=int((~native.is_valid).sum()))
            report.append(row);(output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print('PAIR COMPLETE',key,'analysis',analysis,'total',row['total_seconds'],dict(Counter(current.change_typ)),flush=True)
        finally:
            if verifier is not None:verifier.close()
            for scene in scenes:close_scene(scene)
    (output/'batch_time.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-batch)),encoding='utf-8')


if __name__=='__main__':main()
