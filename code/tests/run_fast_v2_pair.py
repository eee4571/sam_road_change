"""Explicit cached-pair v1/v2 experiment; never runs extraction or GT correction."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import shapely
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import fast_auto_change as baseline
from engine.fast_pipeline import _load_fast_period_result, _read_fast_change_layer
from engine.auto_scene_cache import close_scene, scene_key
from profile_process_memory import ProcessMemory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path)
    parser.add_argument('label')
    parser.add_argument('--algorithm',choices=['baseline','v2'],required=True)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    output=args.job/'_profiling'/'fast_v2';output.mkdir(parents=True,exist_ok=True)
    for name in ('fast_auto_change','fast_auto_v2'):
        source=Path(__file__).resolve().parents[1]/'engine'/f'{name}.py'
        (output/f'{args.label}_{name}.py').write_bytes(source.read_bytes())
    manifest=json.loads((args.job/'pipeline_result.json').read_text(encoding='utf-8'))
    pair=manifest['change_results'][0]
    periods=manifest.get('auto_period_results') or manifest['period_results']
    payloads=[_load_fast_period_result(next(p for p in periods if p['grid']==pair['grid'] and p['period']==pair[key]))
              for key in ['before_period','after_period']]
    centers=[_read_fast_change_layer(p,'centerlines') for p in payloads]
    crs=centers[0].crs
    if not crs.is_projected or abs(crs.axis_info[0].unit_conversion_factor-1)>1e-6:
        crs=centers[0].estimate_utm_crs()
    fingerprints=[scene_key(p,crs) for p in payloads]
    scenes=[]
    for p,center in zip(payloads,centers):
        frames=[_read_fast_change_layer(p,k).to_crs(crs) for k in ['surfaces','width_segments','valid_observation']]
        scenes.append(baseline.RoadScene(center.to_crs(crs),*frames,baseline.WindowedProbability(p['road_probability'],crs),crs))
    observations=Counter(); seconds=Counter();restore=[]
    def wrap(owner,name,key,size=False):
        original=getattr(owner,name)
        def measured(*values,**kwargs):
            observations[key+'_calls']+=1
            if size:
                observations[key+'_items']+=np.size(values[1] if name=='_values_at' else values[0])
            started=time.perf_counter()
            try:return original(*values,**kwargs)
            finally:seconds[key]+=time.perf_counter()-started
        setattr(owner,name,measured);restore.append((owner,name,original))
    wrap(baseline.RoadScene,'match','match')
    wrap(baseline.WindowedProbability,'_values_at','probability',True)
    wrap(baseline,'_measure_period_width','exact_width')
    wrap(shapely,'buffer','buffer',True)
    for scene in scenes:wrap(scene,'surface','surface')
    from engine.fast_auto_v2 import analyze_scenes as v2
    run=baseline.analyze_scenes if args.algorithm=='baseline' else v2
    presence=[]
    try:
        with ProcessMemory() as memory:
            started=time.perf_counter()
            result=run(*scenes,tolerance=float(pair['tolerance']),absolute=float(pair['absolute']),relative=float(pair['ratio']),presence_audit=presence)
            wall=time.perf_counter()-started
    finally:
        for owner,name,original in reversed(restore):setattr(owner,name,original)
    summary=dict(algorithm=args.algorithm,analyze_scenes_seconds=wall,work=dict(observations),work_seconds=dict(seconds),
                 memory=memory.result,counts=result[3],types=dict(Counter(r['change_typ'] for r in result[0])),
                 local_sample_count=result[3].get('v2_local_sample_count',len(result[1])),inputs=fingerprints,pair=[pair['before_period'],pair['after_period']])
    with (output/f'{args.label}.pkl').open('wb') as handle:pickle.dump((result,presence),handle,protocol=5)
    (output/f'{args.label}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ANALYSIS',json.dumps(summary,ensure_ascii=False),flush=True)
    if args.finalize:
        started=time.perf_counter()
        final=baseline.finalize_auto_candidates(*result,presence_audit=presence,scenes=dict(zip(['before','after'],scenes)),
            centerlines=centers,output_dir=output/args.label,before_period=pair['before_period'],after_period=pair['after_period'],
            position_tolerance=float(pair['tolerance']),diagnostics=True,internal_outputs=True)
        summary['finalization_seconds']=time.perf_counter()-started
        summary['final_result']=final
        (output/f'{args.label}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    assert fingerprints==[scene_key(p,crs) for p in payloads]
    for scene in scenes:close_scene(scene)


if __name__=='__main__':main()
