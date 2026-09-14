"""One normalized T2 extraction only. Baselines, Fast2 and Temporal are never run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_experiment import ROOT, CODE, environment, save, sha

OUT = ROOT/'irmad'


def identity():
    base = ROOT/'A/_work/tasks/runs/pair_ab/grids/area/periods'
    paths = list((ROOT/'inputs/raw').glob('*.tif'))
    for period in ('20250118', '20260203'):
        workspace = base/period
        paths.append(workspace/'latest_result.json')
        paths += list((workspace/'runs/roads/products').glob('*'))
        manifest = json.loads((workspace/'input_manifest.json').read_text(encoding='utf-8'))
        paths += list(Path(manifest['images']).glob('*.tif'))
    return {str(p):sha(p) for p in paths if p.is_file()}


def extract():
    from user_pipeline import extract as production_extract
    cfg = json.loads((ROOT/'config/experiment.json').read_text(encoding='utf-8'))
    workspace, images = OUT/'T2', OUT/'normalized_tiles'
    if (workspace/'latest_result.json').exists():
        raise FileExistsError('Normalized T2 already extracted; do not rerun automatically')
    paths = sorted(images.glob('*.tif'))
    if len(paths)!=8: raise ValueError('All eight normalized tiles required')
    txt = workspace/'batches/grid_tiles.txt'
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text('\n'.join(map(str, paths))+'\n', encoding='utf-8-sig')
    save(workspace/'input_manifest.json', dict(workspace=str(workspace), source=str(images), images=str(images),
         tile_count=len(paths), image_txt=str(txt), txt_dir=str(txt.parent), mode='existing_grid'))
    result = production_extract(argparse.Namespace(workspace=str(workspace), source='',
        checkpoint=cfg['checkpoint'], config=cfg['model_config'], device='cuda', pixel_size=0., rescale='off',
        run_id='roads', junction_node_mode='sparse', execution_profile='fast', validation_area=cfg['validation_area'],
        grid='area', period='20260203', resume=True, pipeline_state=''))
    save(OUT/'T2_result.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['normalize', 'extract', '_extract'])
    stage = parser.parse_args().stage
    if stage == '_extract':
        extract()
        return
    OUT.mkdir(exist_ok=True)
    env = environment()
    env.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if stage == 'normalize':
        if (OUT/'identity_before.json').exists(): raise FileExistsError('Existing run identity')
        save(OUT/'identity_before.json', identity())
        script = [str(ROOT/'irmad_rrn.py')]
    else:
        if not (OUT/'normalization.json').exists(): raise FileNotFoundError('Normalize first')
        if not (OUT/'normalize_audit.json').exists(): raise FileNotFoundError('Normalization output audit must pass first')
        script = [str(Path(__file__).resolve()), '_extract']
    start = time.perf_counter()
    with (OUT/f'{stage}.log').open('a', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', *script], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{stage} failed; inspect {OUT/stage}.log')
    before = json.loads((OUT/'identity_before.json').read_text(encoding='utf-8'))
    after = identity()
    assert before == after, 'Raw inputs or reused baseline products changed'
    save(OUT/f'{stage}_audit.json', dict(elapsed_seconds=time.perf_counter()-start, baseline_and_raw_unchanged=True,
        checked_files=len(after), code_snapshot=str(CODE), forbidden_stages_run=[]))


if __name__ == '__main__':
    main()
