"""Six cached-pair comparison; GT is evaluated only after Auto finalization."""
import argparse
import json
from pathlib import Path
import pickle
import time
from unittest.mock import patch
from collections import Counter
import geopandas as gpd
from shapely import union_all
from tune_fast_v2_offline import (RoadScene,WindowedProbability,_load_fast_period_result,
    _read_fast_change_layer,scene_key,close_scene,assess,finalize_auto_candidates)
from engine import fast_auto_change as legacy
from engine.fast_auto_v2 import analyze_scenes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);args=parser.parse_args()
    previous=args.job/'_profiling'/'fast_v2_tuning'
    output=args.job/'_profiling'/'fast2_object_reconciliation';output.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((args.job/'pipeline_result.json').read_text(encoding='utf-8'))
    old_report={r['pair']:r for r in json.loads((previous/'final_report.json').read_text(encoding='utf-8'))}
    report=[]
    for pair in manifest['change_results']:
        key=f"{pair['grid']}_{pair['before_period']}_{pair['after_period']}";dest=output/key;dest.mkdir(exist_ok=True)
        started=time.perf_counter()
        payloads=[_load_fast_period_result(next(p for p in manifest['auto_period_results'] if p['grid']==pair['grid'] and p['period']==pair[k]))
                  for k in ('before_period','after_period')]
        centers=[_read_fast_change_layer(p,'centerlines') for p in payloads]
        crs=centers[0].estimate_utm_crs() if centers[0].crs.is_geographic else centers[0].crs
        fingerprints=[scene_key(p,crs) for p in payloads]
        saved=pickle.loads((previous/key/'balanced.pkl').read_bytes())
        assert saved['identity']['inputs']==fingerprints,'Original Auto inputs changed'
        scenes=[]
        try:
            for p,c in zip(payloads,centers):
                frames=[_read_fast_change_layer(p,k).to_crs(crs) for k in ('surfaces','width_segments','valid_observation')]
                scenes.append(RoadScene(c.to_crs(crs),*frames,WindowedProbability(p['road_probability'],crs),crs))
            load_seconds=time.perf_counter()-started
            presence=[];tick=time.perf_counter()
            with patch.object(legacy,'analyze_scenes',side_effect=AssertionError('Fast1 fallback')),\
                 patch.object(RoadScene,'match',side_effect=AssertionError('point match')),\
                 patch.object(legacy,'_measure_period_width',side_effect=AssertionError('remeasurement')):
                result=analyze_scenes(*scenes,tolerance=float(pair['tolerance']),absolute=float(pair['absolute']),
                                     relative=float(pair['ratio']),presence_audit=presence)
            analysis=time.perf_counter()-tick;tick=time.perf_counter()
            (dest/'analysis.pkl').write_bytes(pickle.dumps((result,presence),protocol=5))
            finalize_auto_candidates(*result,presence_audit=presence,scenes=dict(zip(('before','after'),scenes)),centerlines=centers,
                output_dir=dest,before_period=pair['before_period'],after_period=pair['after_period'],
                position_tolerance=float(pair['tolerance']),diagnostics=True,internal_outputs=True)
            export=time.perf_counter()-tick
            truth=gpd.read_file(previous/key/'gt_upstream.gpkg')
            area=union_all(gpd.read_file(pair['validation_area']).to_crs(crs).geometry)
            current=gpd.read_file(dest/'road_changes.shp')
            valid=bool(current.is_valid.all());metrics=assess(current.to_crs(crs),truth,area)
            # Audit which cancelled candidates touched GT, including all GT
            # categories. This is observational and never feeds reconciliation.
            gt=union_all(truth.geometry)
            suppressed=[dict(candidate_index=i,change_typ=r['change_typ'],pair_id=r['v2_stable_pair'],
                             gt_intersection_m2=r['geometry'].intersection(gt).area,geometry=r['geometry'])
                        for i,r in enumerate(result[0]) if 'v2_stable_pair' in r]
            if suppressed:gpd.GeoDataFrame(suppressed,geometry='geometry',crs=crs).to_file(dest/'paired_stable_audit.gpkg',driver='GPKG')
            row=dict(pair=key,before=old_report[key]['variants']['balanced']['final'],after=metrics,
                baseline_analysis_seconds=old_report[key]['variants']['balanced']['wall_seconds'],
                analysis_seconds=analysis,scene_load_seconds=load_seconds,finalization_seconds=export,
                total_seconds=load_seconds+analysis+export,counts=result[3],valid_geometry=valid,
                paired_candidates_with_gt_overlap=sum(r['gt_intersection_m2']>1e-6 for r in suppressed))
            report.append(row);(output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print('PAIR COMPLETE',key,analysis,export,dict(Counter(current.change_typ)),flush=True)
        finally:
            for scene in scenes:close_scene(scene)


if __name__=='__main__':main()

