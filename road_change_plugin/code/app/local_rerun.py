"""One resumable selection, using the existing period/change/finalization entries."""
import json
import copy
from pathlib import Path


def scoped_downstream(manifest, grid, pipeline, *, invalidate=False, pairs=None):
    """Use the existing finalization barrier for the affected area only."""
    scoped = copy.deepcopy(manifest)
    keys = ('period_results', 'auto_period_results', 'change_results', 'final_period_results', 'temporal_results')
    for key in keys:
        if key in scoped:scoped[key] = [r for r in scoped[key] if str(r.get('grid')) == grid]
    if 'period_orders' in scoped:
        scoped['period_orders'] = {k:v for k,v in scoped['period_orders'].items() if k == grid}
    if 'grids' in scoped.get('input_spec', {}):
        scoped['input_spec']['grids'] = {k:v for k,v in scoped['input_spec']['grids'].items() if k == grid}
    if invalidate:
        pipeline._invalidate_fast_finalization(scoped,pairs=pairs)
    else:
        pipeline._refresh_manifest_downstream(scoped)
    for key in keys:
        if key in manifest or key in scoped:
            manifest[key] = [r for r in manifest.get(key, []) if str(r.get('grid')) != grid] + scoped.get(key, [])
    for key in ('fast_finalization_state', 'temporal_status', 'downstream_updated_at'):
        if key in scoped:manifest[key] = scoped[key]
    if invalidate:manifest.pop('evaluation_summary', None)
    else:pipeline.aggregate_change_evaluations(manifest, Path(manifest.get('job_root') or '.'))


def selection_plan(manifest, grid, selection, pipeline):
    names = pipeline._manifest_period_plan(manifest).get(grid, [])
    periods = set(selection.get('selected_periods', []))
    pairs = {tuple(p) for p in selection.get('selected_pairs', [])}
    adjacent = list(zip(names, names[1:]))
    if not periods and not pairs:
        raise ValueError('请选择需要重跑的期次或变化对')
    if not periods.issubset(names) or not pairs.issubset(adjacent):
        raise ValueError(f'{grid}：所选期次或变化对不属于当前成果')
    for period in periods:
        pairs.update(pipeline._affected_manifest_pairs(manifest, grid, period))
    return dict(grid=grid, periods=[p for p in names if p in periods],
                pairs=[list(p) for p in adjacent if p in pairs])


def run_selection(args, pipeline):
    """Persist each finished step; retry only the interrupted step on resume."""
    path = Path(args.pipeline_manifest).resolve()
    manifest = pipeline.read_json(path)
    selection = json.loads(args.selection)
    if 'areas' in selection:
        areas = selection['areas']
        if not areas or len({a['grid'] for a in areas}) != len(areas):
            raise ValueError('请选择不重复的区域')
        plan = dict(areas=[selection_plan(manifest, a['grid'], a, pipeline) for a in areas])
    else:
        plan = selection_plan(manifest, str(args.grid), selection, pipeline)
    plans = plan.get('areas', [plan])
    checkpoint = path.parent / 'local_rerun.json'
    saved = pipeline.read_json(checkpoint) if checkpoint.is_file() else {}
    if saved and saved.get('plan') != plan:
        raise ValueError('当前重跑断点与所选范围不一致，请继续原范围或重新开始')
    pipeline._apply_fast2_task_settings(manifest, args)
    steps = []
    for area in plans:
        steps += [(area['grid'], 'period', p) for p in area['periods']]
        steps += [(area['grid'], 'change', pair) for pair in area['pairs']]
        steps.append((area['grid'], 'downstream', None))
    completed = int(saved.get('completed', 0))
    if not 0 <= completed <= len(steps):raise ValueError('局部重跑断点无效')
    if not saved:
        for area in plans:
            scoped_downstream(manifest, area['grid'], pipeline, invalidate=True,pairs=area['pairs'])
        manifest['status'] = 'running'
        pipeline._persist_existing_pipeline(manifest, path)
        pipeline.write_json(checkpoint, dict(plan=plan, completed=0))
    for index in range(completed, len(steps)):
        grid, kind, value = steps[index]
        context = dict(grid=grid)
        if kind == 'period':context['period'] = value
        if kind == 'change':context.update(before_period=value[0], after_period=value[1])
        stage = {'period': '重跑道路期次', 'change': '重跑变化对', 'downstream': '更新长时序与评价'}[kind]
        pipeline.emit('pipeline', stage=stage, status='running', completed=index, total=len(steps), **context)
        try:
            if kind == 'period':
                pipeline._rerun_period_entry(manifest, grid, value)
            elif kind == 'change':
                before, after = value
                pipeline._invalidate_change_evaluation(manifest, grid, before, after)
                pipeline._persist_existing_pipeline(manifest, path)
                pipeline._rerun_change_entry(manifest, grid, before, after)
            else:
                scoped_downstream(manifest, grid, pipeline)
        except Exception as exc:
            if kind == 'change':
                pipeline._mark_change_rerun_failed(manifest, grid, value[0], value[1], exc)
            manifest['status'] = 'failed'
            pipeline._persist_existing_pipeline(manifest, path)
            raise
        manifest['status'] = 'running'
        pipeline._persist_existing_pipeline(manifest, path)
        pipeline.write_json(checkpoint, dict(plan=plan, completed=index + 1))
        pipeline.emit('pipeline', stage=stage, status='complete', completed=index+1, total=len(steps), **context)
    manifest['status'] = 'completed'
    pipeline._persist_existing_pipeline(manifest, path)
    pipeline.emit('complete', stage='rerun-selection', grid=plans[0]['grid'] if len(plans) == 1 else '',
                  grids=[a['grid'] for a in plans],
                  period_count=sum(len(a['periods']) for a in plans),
                  change_count=sum(len(a['pairs']) for a in plans))
    return plan
