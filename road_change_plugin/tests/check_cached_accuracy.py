"""Rebuild truth-assisted products from immutable caches in a separate directory."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tests')]
import user_pipeline as pipeline
from check_cached_temporal import fingerprint
from engine.fast_gt_reconciliation import augment_fast_changes_with_truth, build_fast_temporal_outputs


def run(source, output):
    source = source.resolve(); output = output.resolve()
    original = pipeline.read_json(source)
    if output.exists():
        raise ValueError('Use a new validation directory to preserve previous evidence')
    output.mkdir(parents=True)
    before = fingerprint(original, source)
    manifest = copy.deepcopy(original)
    manifest.update(job_root=str(output), project_root=str(output), output_root=str(output / 'published'),
                    period_results=copy.deepcopy(original['auto_period_results']),
                    temporal_results=[], fast_finalization_state='pending', temporal_status='pending')
    changes = []
    with patch.object(pipeline, 'extract', side_effect=AssertionError('inference forbidden')), \
         patch.object(pipeline, '_run_fast_change_result', side_effect=AssertionError('detection forbidden')):
        for old in original['change_results']:
            periods = [next(p for p in original['auto_period_results']
                            if p['grid'] == old['grid'] and p['period'] == name)
                       for name in (old['before_period'], old['after_period'])]
            target = output / 'corrections' / old['grid'] / (old['before_period'] + '_to_' + old['after_period'])
            result = augment_fast_changes_with_truth(old['automatic'], old['truth_path'], target,
                before_result=periods[0], after_result=periods[1],
                before_period=old['before_period'], after_period=old['after_period'],
                truth_type_field=old.get('truth_type_field') or 'BHBM',
                validation_area=old.get('validation_area') or original.get('validation_area'),
                defer_finalization=True)
            changes.append(dict(result, grid=old['grid'], execution_profile='fast', status='completed',
                                fast_finalization_state='pending'))
        manifest['change_results'] = changes
        manifest['temporal_results'] = build_fast_temporal_outputs(manifest, output)
        assert manifest['temporal_results'], 'temporal products missing'
        manifest['fast_finalization_state'] = 'completed'
        for entry in manifest['change_results']:
            entry['fast_finalization_state'] = 'completed'
        pipeline.write_json(output / 'pipeline_result.json', manifest)
        for entry in manifest['change_results']:
            pipeline._evaluate_existing_changes_impl(argparse.Namespace(
                pipeline_manifest=str(output / 'pipeline_result.json'), _manifest=manifest,
                grid=entry['grid'], before_period=entry['before_period'], after_period=entry['after_period'],
                truth=entry['truth_path'], truth_type_field=entry.get('truth_type_field') or 'BHBM',
                validation_area=original.get('validation_area') or '', evaluation_tolerance=5.,
                defer_publish=True, defer_aggregate=True))
        pipeline.aggregate_change_evaluations(manifest, output)
    assert fingerprint(original, source) == before, 'source cache modified'
    manifest['status'] = 'completed'
    pipeline.write_json(output / 'pipeline_result.json', manifest)
    records = []
    for entry in manifest['change_results']:
        evaluation = pipeline.read_json(Path(entry['summary']))['evaluation']
        row = next(r for r in evaluation['metrics'] if r['class'] == 'all')
        records.append(dict(before=entry['before_period'], after=entry['after_period'],
                            recall=row['change_recall'], precision=row['change_precision'],
                            completeness=row.get('road_centerline_completeness'),
                            offset_px=row.get('centerline_mean_offset_px')))
    report = dict(source=str(source), upstream_files_unchanged=len(before), status='completed',
                  results=records, temporal=manifest['temporal_results'],
                  aggregate=manifest['evaluation_summary']['metrics'][0])
    pipeline.write_json(output / 'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)
