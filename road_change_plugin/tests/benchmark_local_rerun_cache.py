"""Exercise the PySide backend selection on copied caches, never the source project."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import user_pipeline as pipeline
from app.local_rerun import run_selection
from engine.fast_pipeline import export_fast_products
from engine.product_cache import signature


def run(args):
    project=args.project.resolve();output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('验证输出目录必须为空，请指定新的隔离目录')
    output.mkdir(parents=True,exist_ok=True)
    original=pipeline.read_json(project/'_work/current/pipeline_result.json')
    manifest=copy.deepcopy(original)
    manifest.update(project_root=str(output),job_root=str(output),output_root=str(output/'成果输出'),
        run_id='cache_regression',period_results=[],auto_period_results=[],change_results=[],temporal_results=[])
    manifest.pop('final_period_results',None);manifest.pop('published',None)
    manifest['period_orders']={args.grid:{'period_order':[args.before,args.after]}}
    manifest['input_spec']['grids']={args.grid:{p:original['input_spec']['grids'][args.grid][p] for p in [args.before,args.after]}}
    protected=[]
    for period in [args.before,args.after]:
        old=next(e for e in original.get('auto_period_results',original['period_results'])
                 if str(e['grid'])==args.grid and str(e['period'])==period)
        payload=pipeline.read_json(Path(old['result']))
        source=Path(payload['centerlines']).parent
        workspace=output/'grids'/args.grid/'periods'/period
        products=workspace/'runs/roads/products';(products/'raw_width').mkdir(parents=True,exist_ok=True)
        for relative in ['regional_products.gpkg','raw_width/width_observations.gpkg']:
            shutil.copy2(source/relative,products/relative)
            protected.append(source/relative)
        shutil.copy2(Path(payload['workspace'])/'input_manifest.json',workspace/'input_manifest.json')
        protected.extend([Path(old['result']),Path(payload['road_probability']),Path(payload['valid_observation'])])
        payload.update(workspace=str(workspace),run_root=str(products.parent),run_id='roads',
                       centerlines=str(products/'road_centerlines.shp'))
        result=workspace/'latest_result.json';pipeline.write_json(result,payload)
        entry={**payload,'grid':args.grid,'period':period,'result':str(result),'status':'completed'}
        entry.pop('published',None)
        manifest['auto_period_results'].append(entry)
        manifest['period_results'].append(copy.deepcopy(entry))
    before=signature(protected)
    path=output/'pipeline_result.json';pipeline.write_json(path,manifest)
    request=argparse.Namespace(pipeline_manifest=str(path),grid=args.grid,
        selection=json.dumps(dict(selected_periods=[],selected_pairs=[[args.before,args.after]])))
    with patch.object(pipeline,'extract',side_effect=AssertionError('unexpected model extraction')), \
         patch.object(pipeline,'prepare',side_effect=AssertionError('unexpected image preparation')), \
         patch.object(pipeline,'run_command',side_effect=AssertionError('unexpected subprocess stage')), \
         patch('engine.width.raw_image_backend.measure_region',side_effect=AssertionError('unexpected RGB measurement')), \
         patch('engine.fast_pipeline.export_fast_products',wraps=export_fast_products) as exports:
        run_selection(request,pipeline)
        export_count=exports.call_count
    current=pipeline.read_json(path)
    with patch('engine.fast_pipeline.export_fast_products',side_effect=AssertionError('duplicate formal export')), \
         patch('engine.fast_gt_reconciliation._auto_axes',side_effect=AssertionError('completed change geometry rebuilt')):
        pipeline._refresh_manifest_downstream(current)
    assert signature(protected)==before,'source cache modified'
    change=current['change_results'][0]
    summary=dict(formal_exports=export_count,model_extractions=0,rgb_measurements=0,source_cache_unchanged=True,
        review_candidates=change['review_candidate_count'],formal_changes=change['changes_feature_count'],
        status=current['status'],manifest=str(path),repeat_formal_exports=0,repeat_change_geometry_rebuilds=0)
    pipeline.write_json(output/'cache_reuse_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['project','output']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['grid','before','after']:parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
