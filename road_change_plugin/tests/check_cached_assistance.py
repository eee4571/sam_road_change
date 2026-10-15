"""Verify pre-review assistance and temporal finalization using immutable caches."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'code/engine/width'), str(ROOT/'tests')]
import geopandas as gpd
import user_pipeline as pipeline
from check_cached_temporal import fingerprint
from engine.evaluation_grid import period_evaluation_grid
from engine.fast_gt_reconciliation import augment_fast_changes_with_truth, build_fast_temporal_outputs
from road_change_detection import evaluate_changes


def run(source, output):
    source, output = source.resolve(), output.resolve()
    original = pipeline.read_json(source)
    before = fingerprint(original, source)
    output.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(original)
    periods = copy.deepcopy(original['auto_period_results'])
    manifest.update(job_root=str(output), project_root=str(output), output_root=str(output/'published'),
                    period_results=periods, final_period_results=copy.deepcopy(original['final_period_results']),
                    change_results=[], temporal_results=[], temporal_status='pending',
                    fast_finalization_state='pending')
    records = []
    with patch.object(pipeline, 'extract', side_effect=AssertionError('inference forbidden')), \
         patch.object(pipeline, '_run_fast_change_result', side_effect=AssertionError('detection forbidden')), \
         patch('engine.fast_gt_reconciliation._write_final_period', side_effect=AssertionError('unreviewed road edits forbidden')):
        for old in original['change_results']:
            pair_periods = [next(p for p in periods if p['grid']==old['grid'] and p['period']==name)
                            for name in (old['before_period'], old['after_period'])]
            automatic = dict(old['automatic'], manual_review_required=True)
            target = output/'corrections'/old['grid']/(old['before_period']+'_to_'+old['after_period'])
            entry = augment_fast_changes_with_truth(automatic, old['truth_path'], target,
                before_result=pair_periods[0], after_result=pair_periods[1],
                before_period=old['before_period'], after_period=old['after_period'],
                truth_type_field=old.get('truth_type_field') or 'BHBM',
                validation_area=old.get('validation_area') or original.get('validation_area'),
                defer_finalization=True)
            manifest['change_results'].append(dict(entry, grid=old['grid'], status='completed',
                execution_profile='fast', fast_finalization_state='pending',
                validation_area=old.get('validation_area') or original.get('validation_area')))
        manifest['temporal_results'] = build_fast_temporal_outputs(manifest, output)
        assert manifest['temporal_results'], 'temporal products missing'
        for entry in manifest['change_results']:
            assert gpd.read_file(entry['road_changes']).empty, 'unreviewed candidates published as formal changes'
            candidates = gpd.read_file(entry['review_candidates'])
            truth = gpd.read_file(entry['truth_path'])
            period = next(p for p in periods if p['grid']==entry['grid'] and p['period']==entry['before_period'])
            validation = gpd.read_file(entry['validation_area']) if entry.get('validation_area') else None
            metrics, _ = evaluate_changes(candidates, truth, validation_area=validation,
                truth_type_field=entry['truth_type_field'], position_grid=period_evaluation_grid([period])[:2])
            expected = entry['assistance_coverage']['candidate_recall']
            actual = metrics[0]['change_recall']
            assert abs(expected-actual)<1e-4, (expected, actual)
            records.append(dict(before=entry['before_period'], after=entry['after_period'],
                pre_review_recall=actual, status=entry['assistance_coverage']['status'],
                candidates=len(candidates), metrics=metrics[0]))
            entry['fast_finalization_state'] = 'completed'
    assert fingerprint(original, source)==before, 'source caches modified'
    manifest.update(status='completed', fast_finalization_state='completed')
    pipeline.write_json(output/'pipeline_result.json', manifest)
    total = sum(r['metrics']['truth_support_m2'] for r in records)
    covered = sum(r['metrics']['position_covered_truth_m2'] for r in records)
    report = dict(source=str(source), unchanged_source_files=len(before), results=records,
                  pre_review_recall=covered/total if total else None,
                  temporal_status=manifest['temporal_status'], temporal=manifest['temporal_results'])
    pipeline.write_json(output/'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)
