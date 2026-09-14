"""Re-evaluate stored raw descriptors; no image/model/candidate recomputation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import time
import numpy as np
import geopandas as gpd
from shapely import from_wkt,union_all
from engine.fast_patch_verification import PatchVerifier
from run_fast2_patch_verification import no_intersection
from tune_fast_v2_offline import (RoadScene,WindowedProbability,_read_fast_change_layer,
                                  close_scene,assess,finalize_auto_candidates)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);args=parser.parse_args()
    output=args.job/'_profiling'/'fast2_raw_image'
    manifest=json.loads((args.job/'pipeline_result.json').read_text(encoding='utf8'))
    report=json.loads((output/'comparison.json').read_text(encoding='utf8'))
    if not (output/'before_boundary_check.json').exists():
        (output/'before_boundary_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    for entry,pair in zip(report,manifest['change_results']):
        started=time.perf_counter();directory=output/entry['pair']
        result,presence=pickle.loads((directory/'analysis.pkl').read_bytes())
        doc=json.loads((directory/'patch_verification.json').read_text(encoding='utf8'))
        verifier=PatchVerifier.__new__(PatchVerifier);verifier.calibration=doc['calibration'];verifier.audit=[]
        verifier.counts=Counter({k:v for k,v in doc['counts'].items() if not k.startswith(('before_','after_','primary_','veto_'))
            and k not in ('extraction_fluctuation','unconfirmed_image','unconfirmed_width','qa_unknown')})
        changed=0;tick=time.perf_counter()
        for audit in doc['candidates']:
            i=audit['candidate'];row=result[0][i];was=row['v2_publish'];wkb=row['geometry'].wkb
            p={k:audit[k] for k in ('ncc','core_ncc','ssim','ring_ssim','anomaly','hog_distance','score_delta','registration')}
            p.update(left=np.asarray(audit['left']),right=np.asarray(audit['right']))
            axis=from_wkt(row['axis_wkt']);width=max(row['width_bef'],row['width_aft'])
            left,bottom,right,top=axis.buffer(max(width,12.),cap_style='flat').bounds
            p['resolution']=audit.get('resolution',max(1.,(right-left)/192,(top-bottom)/192))
            p['valid']=audit.get('valid','raw_image_unavailable_or_invalid' not in audit['reasons'])
            p['periods']=[dict(road_score=audit['road_scores'][j],p=row[f'v2_patch_probability_{side}'],
                               s=row[f'v2_patch_surface_{side}']) for j,side in enumerate(('before','after'))]
            verifier.counts['before_'+row['change_typ']]+=1
            row['v2_publish']=True;verifier.record_decision(row,p,i)
            assert not row['v2_publish'] or was,'Boundary check must not restore a rejected candidate'
            assert row['geometry'].wkb==wkb
            changed+=int(was and not row['v2_publish'])
        replay=time.perf_counter()-tick
        verifier.write_audit(directory)
        for k in list(result[3]):
            if k.startswith('v2_patch_'):del result[3][k]
        result[3].update({'v2_patch_'+k:v for k,v in verifier.counts.items()})
        (directory/'analysis.pkl').write_bytes(pickle.dumps((result,presence),protocol=5))
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
            truth=gpd.read_file(args.job/'_profiling'/'fast_v2_tuning'/entry['pair']/'gt_upstream.gpkg')
            area=union_all(gpd.read_file(pair['validation_area']).to_crs(crs).geometry)
            frame=gpd.read_file(directory/'road_changes.shp').to_crs(crs)
            entry.update(after=assess(frame,truth,area),after_no_gt_intersection=no_intersection(frame,truth,area),
                         counts=result[3],boundary_check_seconds=replay,boundary_extra_veto=changed,
                         boundary_republication_seconds=time.perf_counter()-started)
        finally:
            for scene in scenes:close_scene(scene)
        (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print(entry['pair'],'extra raw boundary veto',changed,flush=True)


if __name__=='__main__':main()
