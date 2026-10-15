"""Measure extreme review deletions using real cached candidates and evaluators.

Outputs observations, not an assertion that recall must meet a target. Original
products are read only; no extraction, width measurement or detection runs.
"""
import argparse
import copy
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code/engine/width')]
import geopandas as gpd
import user_pipeline as pipeline
from check_cached_temporal import fingerprint
from engine.evaluation_grid import period_evaluation_grid
from engine.reviewed_truth import build_truth_filter, apply_truth_filter
from road_change_detection import evaluate_changes


def run(source, output):
    manifest = pipeline.read_json(source)
    output.mkdir(parents=True, exist_ok=False)
    before = fingerprint(manifest, source)
    inputs, candidates, extra = [], [], set()
    for pair, entry in enumerate(manifest['change_results']):
        public_path = Path(entry['review_candidates'])
        truth_path = Path(entry.get('truth') or entry['truth_path'])
        audit_path = Path(entry['output']) / 'gt_assisted_audit.gpkg'
        public = gpd.read_file(public_path)
        truth = gpd.read_file(truth_path)
        layer = entry.get('review_audit_layer') or 'review_candidates'
        audit = gpd.read_file(audit_path, layer=layer)
        projected = public.to_crs(audit.crs)
        generated = audit.loc[audit.change_src.eq('GT_ASSISTED')]
        for index, row in projected.iterrows():
            overlaps = generated.loc[generated.change_typ.eq(row.change_typ)].geometry.intersection(row.geometry)
            if any(g.area > max(1e-8, row.geometry.area * 1e-10) for g in overlaps):
                candidates.append(dict(pair=pair, index=int(index), area_m2=float(row.geometry.area)))
        period = next(p for p in manifest['auto_period_results']
                      if p['grid'] == entry['grid'] and p['period'] == entry['before_period'])
        grid = period_evaluation_grid([period])[:2]
        area = gpd.read_file(entry['validation_area']) if entry.get('validation_area') else None
        inputs.append((entry, public, truth, audit_path, layer, grid, area))
        for path in (public_path, truth_path, audit_path):
            extra.add(path)
            if path.suffix.lower() == '.shp':
                extra.update(path.parent.glob(path.stem + '.*'))
    def extra_fingerprint():
        return {str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                for p in extra if p.is_file()}
    extra_before = extra_fingerprint()
    cache = {}
    def pair_metric(pair, rejected):
        key = pair, tuple(sorted(rejected))
        if key not in cache:
            entry, public, truth, audit, layer, grid, area = inputs[pair]
            decisions = {str(i): 'reject' if i in rejected else 'keep' for i in range(len(public))}
            review = build_truth_filter(public, decisions, audit, audit_layer=layer, position_grid=grid)
            effective, metadata = apply_truth_filter(truth, review, entry['truth_type_field'])
            rows = evaluate_changes(public.drop(index=list(rejected)), effective,
                validation_area=area, truth_type_field=entry['truth_type_field'],
                position_grid=grid, truth_extent=truth, class_mode='three')[0]
            cache[key] = dict(metric=rows[0], review=metadata)
        return cache[key]
    orders = [('largest_first', sorted(candidates, key=lambda r: -r['area_m2'])),
              ('smallest_first', sorted(candidates, key=lambda r: r['area_m2']))]
    scenarios = [('baseline', [])]
    for label, order in orders:
        scenarios.extend((f'{label}_{count:02}', order[:count]) for count in range(1, len(order) + 1))
    rng = random.Random(20261015)
    for trial in range(20):
        count = round(len(candidates) * (.25, .5, .75, .9)[trial % 4])
        scenarios.append((f'random_{trial:02}', rng.sample(candidates, count)))
    # Also isolate deletion of every assisted candidate in each individual pair.
    for pair in range(len(inputs)):
        scenarios.append((f'all_in_pair_{pair}', [c for c in candidates if c['pair'] == pair]))
    results = []
    for label, selected in scenarios:
        folder = output / label
        folder.mkdir()
        entries, details = [], []
        for pair in range(len(inputs)):
            rejected = {r['index'] for r in selected if r['pair'] == pair}
            value = pair_metric(pair, rejected)
            path = folder / f'pair_{pair}.json'
            pipeline.write_json(path, dict(evaluation=dict(metrics=[value['metric']])))
            entries.append(dict(summary=str(path), evaluation_state='completed'))
            details.append(dict(pair=pair, rejected=sorted(rejected), **value))
        aggregated = pipeline.aggregate_change_evaluations(
            dict(execution_profile='fast', change_results=entries), folder)['metrics'][0]
        assert abs(aggregated['change_recall'] -
                   sum(v['metric']['position_covered_truth_m2'] for v in details) /
                   sum(v['metric']['truth_support_m2'] for v in details)) < 1e-12
        result = dict(name=label, rejected_count=len(selected),
                      rejected_candidate_area_m2=sum(r['area_m2'] for r in selected),
                      project_recall=aggregated['change_recall'],
                      truth_area_m2=aggregated['truth_support_m2'],
                      covered_area_m2=aggregated['position_covered_truth_m2'], pairs=details)
        results.append(result)
        print(label, len(selected), round(result['project_recall'] * 100, 6), flush=True)
        pipeline.write_json(output / 'results.json', dict(candidates=candidates, scenarios=results))
    assert fingerprint(manifest, source) == before, 'Upstream products changed'
    assert extra_fingerprint() == extra_before, 'Truth, audit or candidates changed'
    summary = dict(source=str(source), scenarios=len(results), unique_pair_evaluations=len(cache),
                   assisted_candidates=len(candidates),
                   unchanged_files=len(set(before) | set(extra_before)),
                   below_93=sum(r['project_recall'] < .93 for r in results),
                   minimum=min(results, key=lambda r: r['project_recall'])['name'])
    pipeline.write_json(output / 'summary.json', summary)
    print('DONE', summary, flush=True)


def verify_saved(source, output, names):
    """Verify selected measured scenarios through actual review publication."""
    from app.change_review import ReviewSession
    from app.review_publication import remap
    manifest = pipeline.read_json(source)
    before = fingerprint(manifest, source)
    measured = pipeline.read_json(output / 'results.json')['scenarios']
    baseline = next(row for row in measured if row['name'] == 'baseline')
    saved = []
    for name in names:
        case = next(row for row in measured if row['name'] == name)
        project = output / 'saved_verified' / name
        project.mkdir(parents=True, exist_ok=False)
        local = copy.deepcopy(manifest)
        local.update(project_root=str(project), job_root=str(project), output_root=str(project / 'published'))
        local['change_results'] = []
        for pair, entry in enumerate(manifest['change_results']):
            destination = project / f'input_{pair}'
            shutil.copytree(entry['output'], destination)
            cloned = remap(entry, Path(entry['output']), destination)
            cloned.pop('published', None)
            # Candidate-only caches need an all-keep evaluation for unchanged
            # pairs, otherwise the project aggregate contains only saved pairs.
            summary = pipeline.read_json(Path(cloned['summary']))
            summary['evaluation'] = dict(metrics=[baseline['pairs'][pair]['metric']])
            pipeline.write_json(Path(cloned['summary']), summary)
            cloned['evaluation_state'] = 'completed'
            local['change_results'].append(cloned)
        path = project / 'pipeline_result.json'
        pipeline.write_json(path, local)
        for detail in case['pairs']:
            if not detail['rejected']:
                continue
            session = ReviewSession(path)
            session.open_pair(detail['pair'])
            session.save({str(i): 'reject' if i in detail['rejected'] else 'keep'
                          for i in range(len(session.frame))},
                         progress=lambda message: print(name, message, flush=True))
        actual = pipeline.read_json(path)
        aggregate = pipeline.aggregate_change_evaluations(actual, project / 'verified_aggregate')['metrics'][0]
        assert abs(aggregate['change_recall'] - case['project_recall']) < 1e-8
        assert actual['temporal_status'] == 'completed'
        saved.append(dict(name=name, project_recall=aggregate['change_recall'],
                          temporal_status=actual['temporal_status']))
        pipeline.write_json(output / 'saved_results.json', saved)
        print('SAVED', saved[-1], flush=True)
    assert fingerprint(manifest, source) == before


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--save-cases', nargs='*', default=[])
    args = parser.parse_args()
    run(args.manifest.resolve(), args.output.resolve())
    if args.save_cases:
        verify_saved(args.manifest.resolve(), args.output.resolve(), args.save_cases)
