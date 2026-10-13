"""Publish reviewed changes and dependent products as one recoverable update.

All computation is staged before any current product is replaced. The journal
allows a failed/interrupted file commit to restore the previous complete set.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from .result_publisher import ResultPublisher, copy_dataset, _dataset_members


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_file(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def same_content(first, second):
    first, second = Path(first), Path(second)
    if not second.is_file() or first.stat().st_size != second.stat().st_size:
        return False
    with first.open('rb') as left, second.open('rb') as right:
        while True:
            block = left.read(1024 * 1024)
            if block != right.read(1024 * 1024):
                return False
            if not block:
                return True


def recover_publication(project_root):
    journal = Path(project_root) / '_work/review_publication.json'
    if not journal.is_file():
        return
    record = read_json(journal)
    if record['state'] != 'committing':
        return
    for item in record['files']:
        target = Path(item['target'])
        if item['backup']:
            # The file that blocked publication may still be open elsewhere;
            # unchanged files need no replacement during rollback.
            if not same_content(item['backup'], target):
                replace_file(item['backup'], target)
        else:
            target.unlink(missing_ok=True)
    record['state'] = 'rolled_back'
    write_json(journal, record)


def commit_publication(project_root, version, files):
    """Files maps final paths to staged paths, or None for an obsolete product."""
    project_root, version = Path(project_root), Path(version)
    recover_publication(project_root)
    backup = version / 'publication_backup'
    backup.mkdir()
    entries = []
    for i, (target, source) in enumerate(files.items()):
        target = Path(target)
        old = backup / str(i)
        if target.is_file():
            shutil.copy2(target, old)
        entries.append(dict(target=str(target), source=str(source) if source else None,
                            backup=str(old) if old.is_file() else None))
    journal = project_root / '_work/review_publication.json'
    record = dict(state='committing', files=entries)
    write_json(journal, record)
    try:
        for item in entries:
            if item['source']:
                replace_file(item['source'], item['target'])
            else:
                Path(item['target']).unlink(missing_ok=True)
        record['state'] = 'completed'
        write_json(journal, record)
    except Exception:
        recover_publication(project_root)
        raise


def remap(value, old, new):
    if isinstance(value, dict):
        return {k: remap(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, old, new) for v in value]
    if isinstance(value, str):
        try:
            return str(new / Path(value).relative_to(old))
        except ValueError:
            pass
    return value


def pair_key(entry):
    return tuple(str(entry.get(k, '')) for k in ('grid', 'before_period', 'after_period'))


def publish_review(session, version, record, progress):
    # Lazy imports keep browsing/preview independent of pipeline initialization.
    import user_pipeline as pipeline
    from temporal_road_analysis import build_temporal_grid

    progress('正在准备正式变化成果…')
    manifest = copy.deepcopy(read_json(session.manifest_path))
    key = pair_key(session.pair)
    matches = [e for e in manifest.get('change_results', []) if pair_key(e) == key]
    if len(matches) != 1:
        raise ValueError('当前变化结果无法唯一定位，请重新打开项目')
    entry = matches[0]
    grid, before, after = key
    periods = [p for p in manifest.get('final_period_results', manifest.get('period_results', []))
               if str(p.get('grid')) == grid and p.get('status') not in {'failed', 'stale'}]
    if len(periods) < 2:
        raise ValueError('缺少本区域的单期道路成果，无法更新长时序；请先完成道路处理')
    project = session.project_root
    output = session.output_root
    job = Path(manifest.get('job_root') or session.manifest_path.parent).resolve()
    reviewed = version / 'road_changes_reviewed.shp'
    original = copy_dataset(session.source, version / 'original_changes.shp')
    entry['layers'] = {'changes': str(reviewed)}
    entry['road_changes'] = str(reviewed)
    # The old geometry audit and PNG describe the unreviewed candidates.
    old_preview = (entry.get('published') or {}).get('road_change')
    for name in ('event_geometry_audit', 'road_change', 'previews', 'gpkg', 'review_change',
                 'evaluation_metrics', 'evaluation_summary', 'evaluation_error'):
        entry.pop(name, None)
    entry['summary'] = str(version / 'change_summary.json')
    retained = session.frame.loc[[record['decisions'][str(i)] == 'keep' for i in range(len(session.frame))]]
    kinds = retained.get('change_typ')
    for kind in ('added', 'removed', 'widened', 'narrowed', 'width_changed'):
        entry[kind + '_feature_count'] = int(kinds.eq(kind).sum()) if kinds is not None else 0
    entry['change_interval_count'] = len(retained)
    entry['changes_feature_count'] = len(retained)
    entry['manual_review'] = dict(original=str(original), source_signature=session.signature,
                                  record=str(version / 'review.json'), counts=record['counts'])
    entry['evaluation_state'] = 'stale'
    entry['evaluation_stale'] = True
    write_json(entry['summary'], entry)
    import sys
    if str(pipeline.WIDTH) not in sys.path:
        sys.path.insert(0, str(pipeline.WIDTH))
    from road_change_detection import render_change_preview
    preview = version / 'change_preview.png'
    render_change_preview(preview, retained, retained.iloc[:0], title=f'{before} → {after}', empty_message='无变化')
    entry['road_change'] = str(preview)
    entry['previews'] = {'change': str(preview)}

    progress('正在更新本区域的长时序变化和统计…')
    temporal_dir = version / 'temporal'
    temporal_dir.mkdir()
    previous = next((t for t in manifest.get('temporal_results', []) if str(t.get('grid')) == grid), {})
    spec = manifest.get('input_spec') or {}
    temporal = build_temporal_grid(grid, periods,
        [e for e in manifest['change_results'] if str(e.get('grid')) == grid], temporal_dir,
        tolerance=float(spec.get('tolerance') or 3), width_absolute=float(spec.get('absolute') or 1),
        width_ratio=float(spec.get('ratio') or .1),
        reuse_observations=previous or None, progress=progress)
    temporal['execution_profile'] = manifest.get('execution_profile', 'fast')
    manifest['temporal_results'] = [t for t in manifest.get('temporal_results', []) if str(t.get('grid')) != grid] + [temporal]

    truth, area, field = pipeline._manifest_change_context(manifest, grid, before, after, entry)
    if truth:
        progress('正在根据真值数据重新计算精度…')
        if record['counts']['reject']:
            audit_path = Path(entry.get('output') or '') / 'gt_assisted_audit.gpkg'
            if audit_path.is_file():
                from engine.reviewed_truth import build_truth_filter
                review_filter = build_truth_filter(session.frame, record['decisions'], audit_path,
                    audit_layer=entry.get('review_audit_layer','changes'))
                entry['manual_review']['truth_filter'] = review_filter
                record['truth_filter'] = review_filter
            elif entry.get('ground_truth_used'):
                raise ValueError('缺少变化来源记录，无法更新复核评价，请检查处理成果')
        values = entry.get('truth_value_map') or manifest.get('truth_value_map') or {}
        pipeline._evaluate_existing_changes_impl(argparse.Namespace(
            pipeline_manifest=str(session.manifest_path), _manifest=manifest, _evaluation_output=str(version),
            defer_publish=True, defer_aggregate=True, grid=grid, before_period=before, after_period=after,
            truth=truth, validation_area=area, truth_type_field=field or 'BHBM',
            truth_added_value=values.get('added', ''), truth_width_changed_value=values.get('width_changed', ''),
            truth_removed_value=values.get('removed', ''),
            evaluation_tolerance=float(spec.get('evaluation_tolerance') or 5)))
    else:
        entry['evaluation_state'] = 'skipped_no_truth'
        entry.pop('evaluation_stale', None)
        write_json(entry['summary'], entry)
    summary = pipeline.aggregate_change_evaluations(manifest, version)
    manifest.update(temporal_status='completed', downstream_updated_at=pipeline.now_text(),
                    manual_review_updated_at=pipeline.now_text(), updated_at=pipeline.now_text())

    progress('正在发布变化成果、评价和统计…')
    stage = version / 'publication'
    publisher = ResultPublisher(stage, project_root=project)
    existing_index = output / publisher.layout.result_index_path.name
    if existing_index.is_file():
        publisher.index = read_json(existing_index)
    entry['published'] = publisher.publish_change(grid, before, after, entry, save=False)
    published_source = Path(entry['published']['changes'])
    temporal['published'] = publisher.publish_temporal(grid, temporal, save=False)
    areas = list(publisher.index.get('areas', {}))
    publisher.publish_evaluation(areas, summary, save=False, within_changes=True)
    # Publish summaries at their established work paths too, so all readers agree.
    files = {}
    for name in ('evaluation_summary.csv', 'evaluation_summary.json'):
        files[job / name] = version / name if (version / name).is_file() else None
    manifest['evaluation_summary'] = remap(summary, version, job)
    manifest = remap(manifest, stage, output)
    entry = next(e for e in manifest['change_results'] if pair_key(e) == key)
    from .change_review import source_signature
    # Use publish_change's actual path (safe_name may normalize a period name).
    entry['manual_review']['published_signature'] = source_signature(published_source, session.inputs)
    record.update(evaluation=entry['evaluation_state'], temporal_updated=True,
                  published_signature=entry['manual_review']['published_signature'])
    write_json(version / 'review.json', record)
    # The final summary contains updated counts, decisions and newly evaluated metrics.
    evaluated = read_json(entry['summary'])
    evaluated.update({k: v for k, v in entry.items() if k != 'evaluation'})
    write_json(entry['summary'], evaluated)
    pipeline._write_task_report(manifest, version)
    publisher.publish_reports(version, save=False)
    publisher.index = remap(publisher.index, stage, output)
    publisher._save()
    # Index JSON must contain final paths, never temporary publication paths.
    manifest.update(result_index=str(existing_index), output_root=str(output), project_root=str(project))
    staged_manifest = version / 'pipeline_result.json'
    write_json(staged_manifest, manifest)
    for path in stage.rglob('*'):
        if path.is_file():
            files[output / path.relative_to(stage)] = path
    # Discard obsolete spatial indexes and preview files for replaced vectors.
    for path in stage.rglob('*.shp'):
        target = output / path.relative_to(stage)
        for old in _dataset_members(target) if target.parent.is_dir() else []:
            if old not in files:
                files[old] = None
    if old_preview:
        preview = Path(old_preview).resolve()
        if preview.is_relative_to(output) and preview not in files:
            files[preview] = None
    # No-truth projects must not keep an old CSV beside the new empty summary.
    if not summary.get('csv'):
        from .result_publisher import safe_name
        for name in areas:
            files[output / safe_name(name) / '02_变化检测/精度评价/evaluation_summary.csv'] = None
    for path in {session.manifest_path, job / 'pipeline_result.json', job / 'job_state.json',
                 project / '_work/current/pipeline_result.json'}:
        files[path] = staged_manifest
    pointer = version / 'current_pointer.json'
    write_json(pointer, {'version': version.name})
    files[session.folder / 'current.json'] = pointer
    for destination in files:
        if not any(Path(destination).resolve().is_relative_to(root) for root in (project, output)):
            raise ValueError(f'更新路径超出当前项目范围：{destination}')
    session.assert_current()
    commit_publication(project, version, files)
    return dict(path=entry['published']['changes'], counts=record['counts'],
                evaluation_state=entry['evaluation_state'], temporal_updated=True,
                original=str(original), published_signature=entry['manual_review']['published_signature'])
