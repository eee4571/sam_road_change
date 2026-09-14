"""Cancel experimental conflict restorations using stored vetoes, no image reads."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import time
import geopandas as gpd
from shapely import union_all
from run_fast2_multitemporal import no_intersection
from tune_fast_v2_offline import (RoadScene,WindowedProbability,_read_fast_change_layer,
                                  close_scene,assess,finalize_auto_candidates)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);args=parser.parse_args()
    output=args.job/'_profiling'/'fast2_patch_verification'
    manifest=json.loads((args.job/'pipeline_result.json').read_text(encoding='utf8'))
    report=json.loads((output/'comparison.json').read_text(encoding='utf8'))
    for entry,pair in zip(report,manifest['change_results']):
        started=time.perf_counter();directory=output/entry['pair']
        result,presence=pickle.loads((directory/'analysis.pkl').read_bytes())
        original=pickle.loads((args.job/'_profiling'/'fast2_multitemporal'/entry['pair']/'analysis.pkl').read_bytes())[0][0]
        doc=json.loads((directory/'patch_verification.json').read_text(encoding='utf8'))
        cancelled=0;decision_start=time.perf_counter()
        for audit in doc['candidates']:
            if audit['state']!='image_model_conflict':continue
            i=audit['candidate'];row=result[0][i]
            assert not audit['reasons'] and row['v2_publish']
            reasons=['persistent_image_road_structure'];state='extraction_fluctuation'
            assert original[i]['geometry'].wkb==row['geometry'].wkb and original[i]['axis_wkt']==row['axis_wkt']
            audit.update(reasons=reasons,state=state)
            row.update(v2_patch_reasons=';'.join(reasons),v2_patch_state=state,v2_publish=not reasons,
                       v2_precision_reason=state+':'+reasons[0] if reasons else original[i]['v2_precision_reason'])
            cancelled+=1
        decision_seconds=time.perf_counter()-decision_start
        # Recount decisions while retaining measured raster/calibration timings.
        counts={k:v for k,v in doc['counts'].items() if not k.startswith(('before_','after_','primary_','veto_'))
                and k not in ('qa_unknown','extraction_fluctuation','unconfirmed_width')}
        c=Counter(counts)
        for audit in doc['candidates']:
            kind=audit['kind'];c['before_'+kind]+=1
            if audit['reasons']:
                c['extraction_fluctuation' if audit['state']=='extraction_fluctuation' else 'unconfirmed_width']+=1
                c['primary_'+audit['reasons'][0]]+=1
                for reason in audit['reasons']:c['veto_'+reason]+=1
            else:
                c['after_'+kind]+=1
                if audit['state']!='verified':c['qa_unknown']+=1
        doc.pop('conflict_replay_seconds',None);doc.pop('conflict_restored',None)
        doc['counts']=dict(c);doc['verdict_replay_seconds']=decision_seconds
        doc['restoration_cancelled']=cancelled
        for k in list(result[3]):
            if k.startswith('v2_patch_'):del result[3][k]
        result[3].update({'v2_patch_'+k:v for k,v in c.items()})
        (directory/'analysis.pkl').write_bytes(pickle.dumps((result,presence),protocol=5))
        (directory/'patch_verification.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf8')
        scenes=[]
        payloads=[next(p for p in manifest['auto_period_results'] if p['grid']==pair['grid'] and p['period']==pair[key])
                  for key in ('before_period','after_period')]
        centers=[_read_fast_change_layer(p,'centerlines') for p in payloads];crs=32650
        try:
            for p,center in zip(payloads,centers):
                frames=[_read_fast_change_layer(p,k).to_crs(crs) for k in ('surfaces','width_segments','valid_observation')]
                scenes.append(RoadScene(center.to_crs(crs),*frames,WindowedProbability(p['road_probability'],crs),crs))
            finalize_auto_candidates(*result,presence_audit=presence,scenes=dict(zip(('before','after'),scenes)),centerlines=centers,
                output_dir=directory,before_period=pair['before_period'],after_period=pair['after_period'],
                position_tolerance=float(pair['tolerance']),diagnostics=True,internal_outputs=True)
            # Truth remains exclusively after publication in this offline test.
            truth=gpd.read_file(args.job/'_profiling'/'fast_v2_tuning'/entry['pair']/'gt_upstream.gpkg')
            area=union_all(gpd.read_file(pair['validation_area']).to_crs(crs).geometry)
            frame=gpd.read_file(directory/'road_changes.shp').to_crs(crs)
            for key in ('conflict_replay_seconds','conflict_restored','conflict_republication_seconds'):
                entry.pop(key,None)
            entry.update(after=assess(frame,truth,area),after_no_gt_intersection=no_intersection(frame,truth,area),
                         counts=result[3],verdict_replay_seconds=decision_seconds,restoration_cancelled=cancelled,
                         verdict_republication_seconds=time.perf_counter()-started)
        finally:
            for scene in scenes:close_scene(scene)
        (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print(entry['pair'],'restoration cancelled',cancelled,flush=True)


if __name__=='__main__':main()
