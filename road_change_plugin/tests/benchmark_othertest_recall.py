"""Cached real-scene evidence replay. Writes only to the requested diagnostic directory."""
import argparse,json,sys,shutil
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import geopandas as gpd
from shapely import from_wkt,union_all
from engine.fast_auto_change import RoadScene,WindowedProbability,finalize_auto_candidates
from engine.fast_auto_v2 import analyze_scenes,_network_intervals,_paired_profiles
from engine.fast_patch_verification import PatchVerifier
from engine.fast_gt_reconciliation import _auto_axes,_publish_assisted_changes


def run(project,output,local_widths,formal_export=False):
    output.mkdir(parents=True,exist_ok=True)
    payloads=[];scenes=[];centers=[];width_stats={}
    for period in ['20171101','20250209']:
        root=project/f'_work/current/grids/珠海/periods/{period}/radiometric'
        result=next(root.glob('*/latest_result.json'))
        p=json.loads(result.read_text('utf-8'));payloads.append(p)
        if formal_export:
            from engine.fast_pipeline import export_fast_products
            products=result.parent/'runs/roads/products';dest=output/period
            (dest/'raw_width').mkdir(parents=True,exist_ok=True)
            shutil.copy2(products/'regional_products.gpkg',dest/'regional_products.gpkg')
            shutil.copy2(products/'raw_width/width_observations.gpkg',dest/'raw_width/width_observations.gpkg')
            image_dir=Path(p['radiometric_preprocessing'].get('output') or p['radiometric_preprocessing'].get('raw_analysis_source'))
            # Production evidence reads the saved probability and original grid.
            probability_dir=result.parent/'runs/roads/width_review'
            formal=export_fast_products(probability_dir,dest,image_dir=image_dir,width_method='raw_image')
            p.update(formal)
        frames=[gpd.read_file(p[k]) for k in ['centerlines','analysis_surfaces','width_segments','valid_observation']]
        if local_widths and not formal_export:
            frames[2]=gpd.read_file(result.parent/'runs/roads/products/raw_width/width_observations.gpkg',layer='width_observations')
        frames=[f.loc[f.geometry.notna()&~f.geometry.is_empty].copy() for f in frames]
        centers.append(frames[0]);crs=frames[0].crs
        scenes.append(RoadScene(*frames,WindowedProbability(p['road_probability'],crs),crs))
        width_stats[period]=dict(count=len(frames[2]),quality=frames[2].quality_grade.value_counts().to_dict())
    config=json.loads((ROOT/'resources/production_config.json').read_text())['fast2_compensation']
    verifier=PatchVerifier(scenes,payloads,compensation=config)
    try:
        profiles=[];matched=[];c=Counter();buffers={}
        for axis in scenes[0].lines:
            m=_network_intervals(axis,scenes[1],3.,buffers,c);matched.extend(m)
            profiles.extend(_paired_profiles(axis,m,*scenes,32.))
        width_stats['paired']=dict(matched=len(matched),profiles=len(profiles),
            comparable_length_m=sum(float((r[4]-r[3]).sum()) for r in profiles),
            matched_length_m=sum(b-a for a,b,_ in matched))
        presence=[]
        records,audit,width_audit,counts=analyze_scenes(*scenes,presence_audit=presence,patch_verifier=verifier,compensation=config)
        verifier.write_audit(output)
        simple=[{k:v for k,v in r.items() if k!='geometry'} for r in records]
        (output/'records.json').write_text(json.dumps(simple,indent=2,default=str),encoding='utf8')
        result=finalize_auto_candidates(records,audit,width_audit,counts,presence_audit=presence,
            scenes=dict(zip(['before','after'],scenes)),centerlines=centers,output_dir=output,
            before_period='20171101',after_period='20250209',internal_outputs=True)
        frame=_auto_axes(result,crs)
        final=_publish_assisted_changes(frame,output/'review','20171101','20250209',require_review=True)
        original_manifest=json.loads((project/'_work/current/pipeline_result.json').read_text('utf8'))
        review_manifest=dict(project_root=str(output),output_root=str(output/'published'),
            input_spec=original_manifest['input_spec'],
            change_results=[dict(final,grid='珠海',before_period='20171101',after_period='20250209')])
        (output/'review_manifest.json').write_text(json.dumps(review_manifest,ensure_ascii=False,indent=2),encoding='utf8')
        baseline=json.loads((project/'_work/diagnosis_20261013/replayed_candidates.json').read_text('utf8'))
        retained=[r for r in records if r.get('v2_publish')]
        probes={}
        for old_id in [356,544,548,101,133,576]:
            old=baseline[old_id];axis=from_wkt(old['axis_wkt'])
            peers=[from_wkt(r['axis_wkt']) for r in retained if r['change_typ']==old['change_typ']]
            covered=axis.intersection(union_all(peers).buffer(3)).length if peers else 0.
            probes[str(old_id)]=dict(kind=old['change_typ'],length_m=axis.length,retained_fraction=covered/axis.length)
        summary=dict(widths=width_stats,counts=counts,raw_candidates=len(records),
            retained_intervals=len(retained),retained_types=dict(Counter(r['change_typ'] for r in retained)),
            review_features=final['review_candidate_count'],formal_features=final['changes_feature_count'],
            probes=probes,review_candidates=final['review_candidates'])
        (output/'benchmark.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
        print(json.dumps(summary,indent=2,ensure_ascii=False))
    finally:
        verifier.close()
        for scene in scenes:scene.probability.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local-widths',action='store_true');p.add_argument('--formal-export',action='store_true');args=p.parse_args()
    run(args.project.resolve(),args.output.resolve(),args.local_widths,args.formal_export)
