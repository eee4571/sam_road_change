"""Re-evaluate cached final products in isolation; never rebuild roads or changes."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tests')]
import user_pipeline as pipeline
from check_cached_temporal import fingerprint


def run(source, output):
    source=source.resolve();output=output.resolve()
    original=pipeline.read_json(source)
    before=fingerprint(original,source)
    output.mkdir(parents=True,exist_ok=False)
    manifest=copy.deepcopy(original)
    manifest.update(job_root=str(output),project_root=str(output),output_root=str(output/'published'))
    records=[]
    with patch.object(pipeline,'extract',side_effect=AssertionError('inference forbidden')), \
         patch.object(pipeline,'_run_fast_change_result',side_effect=AssertionError('detection forbidden')):
        for index,entry in enumerate(manifest['change_results']):
            target=output/str(index);target.mkdir()
            summary=pipeline.read_json(Path(entry['summary']))
            entry['summary']=str(target/'change_summary.json')
            pipeline.write_json(Path(entry['summary']),summary)
            values=entry.get('truth_value_map') or {}
            pipeline._evaluate_existing_changes_impl(argparse.Namespace(
                pipeline_manifest=str(output/'pipeline_result.json'),_manifest=manifest,
                _evaluation_output=str(target),grid=entry['grid'],before_period=entry['before_period'],
                after_period=entry['after_period'],truth=entry.get('truth') or entry['truth_path'],
                truth_type_field=entry.get('truth_type_field') or 'BHBM',
                truth_added_value=values.get('added',''),truth_removed_value=values.get('removed',''),
                truth_width_changed_value=values.get('width_changed',''),
                validation_area=entry.get('validation_area') or manifest.get('validation_area') or '',
                evaluation_tolerance=5.,defer_publish=True,defer_aggregate=True))
            row=pipeline.read_json(Path(entry['summary']))['evaluation']['metrics'][0]
            records.append(dict(grid=entry['grid'],before=entry['before_period'],after=entry['after_period'],
                strict_recall=row['recall'],strict_precision=row['precision'],
                recall=row['change_recall'],precision=row['change_precision'],
                offset_px=row.get('centerline_mean_offset_px'),tolerance_px=row.get('change_match_tolerance_px')))
        pipeline.aggregate_change_evaluations(manifest,output)
    assert fingerprint(original,source)==before,'source cache modified'
    pipeline.write_json(output/'pipeline_result.json',manifest)
    report=dict(source=str(source),unchanged_source_files=len(before),results=records,
                aggregate=manifest['evaluation_summary']['metrics'][0])
    pipeline.write_json(output/'validation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.manifest,args.output)
