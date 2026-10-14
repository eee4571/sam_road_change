"""Run complete temporal recovery against real cached project products.

Creates an isolated output project; source roads, states and changes are read
directly from the given manifest. Inference, width and reconciliation are banned.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import geopandas as gpd
import user_pipeline as pipeline
from app.temporal_recovery import run_temporal, validate_cached_inputs
from temporal_road_analysis import build_temporal_grid


def fingerprint(manifest, source):
    files = {source}
    for key in ('period_results', 'auto_period_results', 'final_period_results', 'change_results'):
        for entry in manifest.get(key, []):
            values = [entry.get(k) for k in ('result', 'centerlines', 'surfaces', 'gpkg', 'road_state',
                      'summary', 'correction_audit', 'event_geometry_audit', 'width_review')]
            values += list((entry.get('layers') or {}).values())
            for value in values:
                if not isinstance(value, str):
                    continue
                path = Path(value)
                if path.is_file():
                    files.add(path)
                    if path.suffix == '.shp':
                        files.update(p for p in path.parent.glob(path.stem + '.*') if p.is_file())
    return {str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}


def run(source, output):
    original = pipeline.read_json(source)
    validate_cached_inputs(original)
    before = fingerprint(original, source)
    manifest = copy.deepcopy(original)
    job = output / '_work/current'
    manifest.update(project_root=str(output), output_root=str(output / 'published'), job_root=str(job),
                    status='failed', temporal_status='failed', temporal_results=[],
                    last_error='NotImplementedError: multi-part geometries do not have coordinate sequences')
    job.mkdir(parents=True, exist_ok=True)
    path = job / 'pipeline_result.json'
    pipeline.write_json(path, manifest)
    with patch.object(pipeline, 'extract', side_effect=AssertionError('inference forbidden')), \
         patch.object(pipeline, 'prepare', side_effect=AssertionError('prepare forbidden')), \
         patch.object(pipeline, '_run_fast_change_result', side_effect=AssertionError('change detection forbidden')), \
         patch('engine.fast_gt_reconciliation.reconcile_periods', side_effect=AssertionError('road reconstruction forbidden')):
        result = run_temporal(argparse.Namespace(pipeline_manifest=str(path), grid=None), pipeline)
    assert result['status'] == 'completed'
    records = []
    for item in result['temporal_results']:
        obs = gpd.read_file(item['observations_shp'])
        life = gpd.read_file(item['life_shp'])
        periods = [p for p in (result.get('final_period_results') or result['period_results']) if p['grid'] == item['grid']]
        changes = [c for c in result['change_results'] if c['grid'] == item['grid']]
        for period in periods:
            if not period.get('road_state'):
                continue
            state = gpd.read_file(period['road_state']).to_crs(obs.crs)
            actual = obs.loc[obs.period.astype(str).eq(str(period['period']))].set_index('road_id')
            for _, row in state.iterrows():
                assert row.track_id in actual.index
                assert actual.loc[row.track_id].geometry.equals(row.geometry)
        reviewed = build_temporal_grid(item['grid'], periods, changes, output / 'review' / item['grid'],
            reuse_observations=item)
        assert set(gpd.read_file(reviewed['life_shp']).road_id) == set(life.road_id)
        records.append(dict(grid=item['grid'], roads=len(life), observations=len(obs), events=item['event_count'],
                            multipart_observations=int(obs.geom_type.eq('MultiLineString').sum()), review_rebuild='passed'))
    assert fingerprint(original, source) == before, 'upstream cache modified'
    report = dict(source=str(source), status='completed', upstream_files_unchanged=len(before), results=records)
    pipeline.write_json(output / 'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest.resolve(), args.output.resolve())
