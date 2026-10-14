"""Resume temporal aggregation from saved products, without algorithm upgrades."""
from pathlib import Path


def dataset_ready(value):
    path = Path(str(value or ''))
    required = [path]
    if path.suffix.lower() == '.shp':
        required += [path.with_suffix(s) for s in ('.shx', '.dbf', '.prj')]
    return all(p.is_file() for p in required)


def validate_cached_inputs(manifest, grid=None):
    """Validate coverage and physical files, never compare algorithm revisions."""
    from user_pipeline import _manifest_period_plan
    plan = _manifest_period_plan(manifest)
    if grid is not None:
        if grid not in plan:
            raise ValueError(f'当前任务没有区域：{grid}')
        plan = {grid: plan[grid]}
    if not plan:
        raise ValueError('当前任务没有已处理期次')
    periods = manifest.get('final_period_results') or manifest.get('period_results', [])
    changes = manifest.get('change_results', [])
    if (manifest.get('execution_profile') == 'fast' and not manifest.get('final_period_results')
            and any(c.get('fast_finalization_state') != 'completed' for c in changes)):
        raise ValueError('正式道路与变化成果尚未固化，请继续处理完成这些步骤后再重建长时序')
    for area, names in plan.items():
        if len(names) < 2:
            raise ValueError(f'{area} 至少需要两个期次')
        for period in names:
            rows = [p for p in periods if str(p.get('grid')) == area and str(p.get('period')) == period]
            if len(rows) != 1 or rows[0].get('status') in {'failed', 'stale'} or not dataset_ready(rows[0].get('centerlines')):
                raise ValueError(f'{area}/{period} 单期道路不完整，请先局部重跑该期次')
            if rows[0].get('road_state') and not dataset_ready(rows[0]['road_state']):
                raise ValueError(f'{area}/{period} 道路状态文件缺失')
        for before, after in zip(names, names[1:]):
            rows = [p for p in changes if (str(p.get('grid')), str(p.get('before_period')), str(p.get('after_period'))) == (area, before, after)]
            if len(rows) != 1 or rows[0].get('status') in {'failed', 'stale'}:
                raise ValueError(f'{area}/{before} → {after} 变化成果不完整，请先局部重跑该变化对')
            row = rows[0]
            path = (row.get('layers') or {}).get('changes') or row.get('road_changes') or Path(row.get('output') or '.') / 'road_changes.shp'
            if not dataset_ready(path):
                raise ValueError(f'{area}/{before} → {after} 变化文件不完整')
    return plan


def run_temporal(args, pipeline):
    """Rebuild only temporal products and pending evaluation/publication."""
    import copy
    path = Path(args.pipeline_manifest).resolve()
    manifest = pipeline.read_json(path)
    grid = getattr(args, 'grid', None)
    validate_cached_inputs(manifest, grid)
    scoped = copy.deepcopy(manifest)
    scoped['period_results'] = scoped.get('final_period_results') or scoped['period_results']
    if grid:
        for key in ('period_results', 'change_results'):
            scoped[key] = [r for r in scoped[key] if str(r.get('grid')) == grid]
    root = Path(manifest.get('job_root') or path.parent)
    pipeline.emit('pipeline', stage='长时序道路汇总', status='running', reason='复用已有道路和变化成果')
    try:
        if manifest.get('execution_profile') == 'fast':
            pipeline._finalize_fast_manifest(scoped, root, cached_roads=True)
        else:
            from temporal_road_analysis import build_from_manifest
            scoped['temporal_results'] = build_from_manifest(scoped, root)
            scoped['temporal_status'] = 'completed'
        for key in ('temporal_results', 'change_results'):
            manifest[key] = ([r for r in manifest.get(key, []) if str(r.get('grid')) != grid] if grid else []) + scoped.get(key, [])
        # A selected area cannot mark unfinished areas as complete.
        expected = set(pipeline._manifest_period_plan(manifest))
        ready = {str(r['grid']) for r in manifest['temporal_results'] if pipeline._temporal_result_ready(r)}
        complete = expected <= ready
        manifest.update(temporal_status='completed' if complete else 'pending',
                        status='completed' if complete else 'completed_with_errors',
                        last_error='', downstream_updated_at=pipeline.now_text())
        if complete:
            manifest['fast_finalization_state'] = 'completed'
            manifest['completed_at'] = pipeline.now_text()
        pipeline.aggregate_change_evaluations(manifest, root)
        pipeline._persist_existing_pipeline(manifest, path, temporal_only=True)
    except Exception as exc:
        manifest.update(status='failed', last_error=str(exc), temporal_status='failed')
        # Keep original upstream entries and publications intact on failure.
        pipeline._persist_pipeline(manifest, root, Path(manifest['output_root']))
        raise
    pipeline.emit('complete', stage='rerun-temporal', status=manifest['status'], manifest=str(path),
                  temporal_grid_count=len(scoped['temporal_results']))
    return manifest
