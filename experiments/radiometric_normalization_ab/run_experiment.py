"""Isolated, GT-free radiometric A/B using the unmodified production pipeline."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CODE = ROOT/'snapshot/code'
PERIODS = ('20250118', '20260203')

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def environment():
    env = os.environ.copy()
    for key, directory in {'TEMP':'tmp', 'TMP':'tmp', 'TMPDIR':'tmp',
                           'MPLCONFIGDIR':'cache/matplotlib', 'TORCH_HOME':'cache/torch',
                           'HF_HOME':'cache/huggingface', 'XDG_CACHE_HOME':'cache',
                           'CUDA_CACHE_PATH':'cache/cuda', 'NUMBA_CACHE_DIR':'cache/numba'}.items():
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               PYTHONHASHSEED='0', GDAL_PAM_ENABLED='NO')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'bootstrap'), str(CODE), env.get('PYTHONPATH','')])
    env['SAMROAD_MODELS_ROOT'] = str(REPO/'runtime/models')
    return env

def prepare():
    import yaml
    if (ROOT/'config/experiment.json').exists():
        raise FileExistsError('Experiment already prepared; reuse its frozen inputs/configuration.')
    (ROOT/'config').mkdir(parents=True,exist_ok=True)
    archive=ROOT/'config/production_code.zip'
    subprocess.run(['git','archive','HEAD','code','-o',str(archive)],cwd=REPO,check=True)
    shutil.unpack_archive(archive,ROOT/'snapshot')
    cfg = json.loads((REPO/'project/test_area/project_config.json').read_text(encoding='utf-8'))
    area_id, area_path = cfg['validation_areas'][0]
    source_lists = dict(cfg['area_periods'][area_id])
    inputs = ROOT/'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    area = Path(area_path)
    for path in area.parent.glob(area.stem+'.*'):
        if path.suffix.lower() in {'.shp','.shx','.dbf','.prj','.cpg'}:
            shutil.copy2(path, inputs/('validation_area'+path.suffix))
    sources = {}
    for period in PERIODS:
        paths = Path(source_lists[period]).read_text(encoding='utf-8-sig').strip().splitlines()
        if len(paths) != 1:
            raise ValueError('This bounded experiment expects one source TIFF per period')
        src = Path(paths[0].strip().strip('"'))
        dest = inputs/'raw'/f'{period}.tif'
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dest)
        sources[period] = dict(original=str(src), copy=str(dest), sha256=sha(src))
        assert sha(dest) == sources[period]['sha256']
        (inputs/f'A_{period}.txt').write_text(str(dest)+'\n', encoding='utf-8')
    config_path = REPO/'runtime/config/samroad_inference.yaml'
    model_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    model_config['SAM_CKPT_PATH'] = str(REPO/'runtime/models/samroad/sam_vit_b_01ec64.pth')
    config_copy = ROOT/'config/samroad_inference.yaml'
    config_copy.parent.mkdir(exist_ok=True)
    config_copy.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding='utf-8')
    manifest = dict(periods=PERIODS, reference=PERIODS[0], sources=sources,
                    validation_area=str(inputs/'validation_area.shp'), model_config=str(config_copy),
                    checkpoint=str(REPO/'runtime/models/samroad/samroad.ckpt'),
                    normalization=dict(method='median + IQR affine per channel, p1/p99 clipping guard', quantiles=[1,25,50,75,99],
                        sample_stride=8,
                        statistics_region='validation area, valid pixels only; no GT', seed=0),
                    pipeline=dict(execution_profile='fast', absolute=2.0, ratio=0.2, tolerance=3.0,
                        pixel_size=0.0, rescale='off', junction_node_mode='sparse',
                        temporal_context='only the two requested periods in both arms', truth=None))
    save(ROOT/'config/experiment.json', manifest)
    # Record production implementation identity without copying or modifying it.
    code_files = list((REPO/'code').glob('*.py'))
    for directory in ('engine', 'app'):
        code_files += list((REPO/'code'/directory).rglob('*.py'))
    save(ROOT/'config/code_identity.json', {str(p.relative_to(REPO)):sha(p) for p in code_files})
    save(ROOT/'status.json', dict(state='prepared', time=time.time()))
    print('PREPARED', flush=True)

def normalize():
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window
    cfg = json.loads((ROOT/'config/experiment.json').read_text(encoding='utf-8'))
    if (ROOT/'status_B.json').exists():
        raise RuntimeError('B already started: normalized inputs are frozen.')
    tick = time.perf_counter()
    ncfg = cfg['normalization']
    area = gpd.read_file(cfg['validation_area'])
    stats = {}
    # Exact source pixels at a fixed spatial stride, without resampling.
    for period in PERIODS:
        samples = [[] for _ in range(3)]
        with rasterio.open(cfg['sources'][period]['copy']) as ds:
            geometries = list(area.to_crs(ds.crs).geometry)
            for y in range(0, ds.height, 512):
                w = Window(0,y,ds.width,min(512,ds.height-y))
                a = ds.read(window=w)
                valid = (ds.read_masks(window=w)>0).all(axis=0)
                valid &= geometry_mask(geometries, a.shape[1:], ds.window_transform(w), invert=True)
                stride = ncfg['sample_stride']
                valid = valid[::stride,::stride]
                for c in range(3):
                    samples[c].append(a[c,::stride,::stride][valid])
            stats[period] = [dict(n=int(sum(len(s) for s in channel)),
                quantiles=np.percentile(np.concatenate(channel), ncfg['quantiles']).tolist()) for channel in samples]
    gain, offset = [], []
    for ref, target in zip(stats[PERIODS[0]], stats[PERIODS[1]]):
        r, t = np.array(ref['quantiles']), np.array(target['quantiles'])
        if t[3]-t[1] <= 0: raise ValueError('Degenerate channel IQR')
        # Match the median exactly; cap the robust scale so p1..p99 remain
        # in the valid uint8 range. No local contrast or histogram remapping.
        g = float(min((r[3]-r[1])/(t[3]-t[1]),
                      r[2]/max(t[2]-t[0],1e-6),
                      (254-r[2])/max(t[4]-t[2],1e-6)))
        gain.append(g)
        offset.append(float(r[2]-g*t[2]))
    out = ROOT/'inputs/normalized'
    out.mkdir(exist_ok=True)
    shutil.copy2(cfg['sources'][PERIODS[0]]['copy'], out/f'{PERIODS[0]}.tif')
    source = Path(cfg['sources'][PERIODS[1]]['copy'])
    destination = out/f'{PERIODS[1]}.tif'
    clipped = np.zeros(3, dtype=np.int64)
    totals = np.zeros(3, dtype=np.int64)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(source) as src:
        assert src.count == 3 and src.dtypes == ('uint8',)*3 and src.nodata == 255
        profile = dict(src.profile)
        profile.update(compress='deflate', predictor=2)
        with rasterio.open(destination,'w',**profile) as dst:
            dst.colorinterp = src.colorinterp
            dst.update_tags(**{k:v for k,v in src.tags().items() if not k.startswith('STATISTICS_')})
            for c in range(1,4):
                dst.update_tags(c,**{k:v for k,v in src.tags(c).items() if not k.startswith('STATISTICS_')})
            for y in range(0,src.height,256):
                win = Window(0,y,src.width,min(256,src.height-y))
                data = src.read(window=win)
                masks = src.read_masks(window=win)>0
                for c in range(3):
                    valid = masks[c]
                    values = data[c][valid].astype(np.float32)*gain[c]+offset[c]
                    clipped[c] += ((values<0)|(values>254)).sum()
                    totals[c] += valid.sum()
                    data[c][valid] = np.rint(np.clip(values,0,254)).astype(np.uint8)
                dst.write(data,window=win)
                # Preserve per-band NoData masks through unchanged 255 values.
                # A dataset-wide mask would collapse distinct per-band masks.
    checks = {}
    for p in PERIODS:
        (ROOT/f'inputs/B_{p}.txt').write_text(str(out/f'{p}.tif')+'\n',encoding='utf-8')
        with rasterio.open(cfg['sources'][p]['copy']) as a, rasterio.open(out/f'{p}.tif') as b:
            assert a.crs == b.crs and a.transform == b.transform and a.shape == b.shape
            assert a.dtypes == b.dtypes and a.count == b.count and a.nodata == b.nodata
            for y in range(0,a.height,512):
                w = Window(0,y,a.width,min(512,a.height-y))
                assert np.array_equal(a.dataset_mask(window=w),b.dataset_mask(window=w))
                assert np.array_equal(a.read_masks(window=w),b.read_masks(window=w))
            checks[p] = dict(grid_crs_shape_dtype_nodata_masks_preserved=True, sha256=sha(out/f'{p}.tif'))
    save(ROOT/'normalization.json', dict(statistics=stats, gain=gain, offset=offset,
        clipped_fraction=(clipped/totals).tolist(), checks=checks, elapsed_seconds=time.perf_counter()-tick))
    print('NORMALIZED', gain, offset, flush=True)

def run_arm(arm):
    cfg = json.loads((ROOT/'config/experiment.json').read_text(encoding='utf-8'))
    cmd = [sys.executable,'-B',str(CODE/'user_pipeline.py'),'all','--mode','validation',
           '--execution-profile','fast','--validation-area','area',cfg['validation_area'],
           '--output-root',str(ROOT/arm/'results'),'--checkpoint',cfg['checkpoint'],
           '--config',cfg['model_config'],'--device','cuda','--pixel-size','0.0','--rescale','off',
           '--junction-node-mode','sparse','--absolute','2.0','--ratio','0.2','--tolerance','3.0',
           '--run-id','pair_ab','--no-evaluation']
    for p in PERIODS: cmd += ['--period','area',p,str(ROOT/f'inputs/{arm}_{p}.txt')]
    job = ROOT/arm/'_work/tasks/runs/pair_ab'
    if (job/'job_state.json').exists(): cmd += ['--resume']
    save(ROOT/f'config/command_{arm}.json',cmd)
    logs = ROOT/'logs'
    logs.mkdir(exist_ok=True)
    start = time.time()
    save(ROOT/f'status_{arm}.json',dict(state='running',started=start))
    with (logs/f'{arm}_pipeline.log').open('a',encoding='utf-8') as stream:
        result = subprocess.run(cmd,cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT)
    save(ROOT/f'status_{arm}.json',dict(state='completed' if result.returncode==0 else 'failed',
        returncode=result.returncode,started=start,elapsed_seconds=time.time()-start))
    if result.returncode: raise RuntimeError(f'{arm} failed: see logs/{arm}_pipeline.log')
    print('ARM COMPLETE',arm,flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','normalize','A','B'])
    args=parser.parse_args()
    os.environ.update(environment())
    if args.action == 'prepare': prepare()
    elif args.action == 'normalize': normalize()
    else: run_arm(args.action)
