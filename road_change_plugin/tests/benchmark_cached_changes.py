"""Replay a saved automatic pair and optional adjacent period without inference."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from engine.fast_auto_change import detect_final_road_changes
from engine.fast_gt_reconciliation import _auto_axes,_publish_assisted_changes
import geopandas as gpd


def run(args):
    config=json.loads((ROOT/'resources/production_config.json').read_text())['fast2_compensation']
    contexts={side:str(path) for side,path in [('previous',args.previous_result),('next',args.next_result)] if path}
    result=detect_final_road_changes(args.before_result,args.after_result,args.output/'automatic',
        before_period=args.before,after_period=args.after,position_tolerance=3.,
        width_change_absolute=2.,width_change_ratio=.2,min_change_area=4.,min_change_length=None,
        internal_outputs=True,temporal_results=contexts,compensation=config)
    frame=gpd.read_file(result['road_changes'])
    crs=frame.estimate_utm_crs() if frame.crs.is_geographic else frame.crs
    review=_publish_assisted_changes(_auto_axes(result,crs),args.output/'review',
                                    args.before,args.after,require_review=True)
    summary=dict(automatic_intervals=result['changes_feature_count'],
        review_features=review['review_candidate_count'],formal_features=review['changes_feature_count'],
        performance=result['performance'])
    (args.output/'benchmark.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['before-result','after-result','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ['previous-result','next-result']:
        parser.add_argument('--'+name,type=Path)
    for name in ['before','after']:
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
