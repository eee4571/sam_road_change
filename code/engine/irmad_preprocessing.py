"""Independent target-to-fixed-reference IR-MAD preprocessing, before inference.

No road/GT/model inputs; no reprojection, no chain and no silent Raw fallback.
Each target period fits one model over its own paired tiles, as in the experiment.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import shutil
import time
import uuid

import numpy as np
import rasterio
from threadpoolctl import threadpool_limits
from . import irmad_core as core

REFERENCE = '20250118'
VERSION = 'fast_irmad_pif_tls_v2'
PARAMETERS = dict(max_iterations=30, tolerance=.01, ncp_threshold=.95, chunk=262144,
                  fit_sample_limit=4_000_000,
                  sampling='equal-bin midpoints of all common-valid pixels in original tile/row order',
                  minimum_common_pixels=1000, regression='orthogonal_TLS', threads=1,
                  output='uint8_rint_clip_valid_range_deflate', grid='strict_identity_no_warp')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()


def configuration(enabled=False, reference_period=REFERENCE):
    if isinstance(reference_period, dict):
        return dict(enabled=bool(enabled),reference_period=reference_period.copy(),version=VERSION,parameters=PARAMETERS.copy())
    reference_period = str(reference_period).strip()
    if not reference_period: raise ValueError("IR-MAD 参考期不能为空")
    return dict(enabled=bool(enabled),reference_period=reference_period,version=VERSION,parameters=PARAMETERS.copy())


def reference_for(value, area):
    """Resolve a saved per-validation-area reference, or a CLI scalar."""
    if isinstance(value,str) and value.lstrip().startswith('{'):
        value=json.loads(value)
    if isinstance(value,dict):
        if str(area) not in value or not value[str(area)]:
            raise ValueError(f'验证区 {area} 尚未选择 IR-MAD 参考期')
        return str(value[str(area)])
    return str(value)


def fingerprint(path):
    path=Path(path).resolve();st=path.stat()
    return dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns)


def request_identity(enabled,period,source,reference=None,grid=None,*,reference_period=REFERENCE):
    """Input fingerprints + analysis-grid request identify extraction dependencies."""
    if not enabled:return 'raw'
    return digest(dict(config=configuration(True,reference_period),period=str(period),source=source,reference=reference,grid=grid))


def extraction_matches(entry,identity):
    return entry.get('radiometric_identity','raw')==identity


def workspace_for(base,identity):
    # A switched input never reuses old model intermediates, even after interruption.
    base=Path(base)
    if identity=='raw':return base
    workspace=base/'radiometric'/identity[:16]
    marker=workspace/'workspace_identity.json'
    if marker.is_file() and json.loads(marker.read_text(encoding='utf8')).get('radiometric_identity')!=identity:
        raise ValueError(f'归一化工作目录身份冲突，禁止复用：{workspace}')
    return workspace


def reserve_workspace_identity(workspace,identity):
    if identity=='raw':return
    workspace=Path(workspace)
    workspace.mkdir(parents=True,exist_ok=True)
    # The short directory is only a locator, never the cache identity. Reserve
    # it with the entire digest, including before input preparation completes.
    marker=workspace/'workspace_identity.json'
    try:
        with marker.open('x',encoding='utf8') as handle:
            json.dump({'radiometric_identity':identity},handle)
    except FileExistsError:
        recorded=json.loads(marker.read_text(encoding='utf8'))
        if recorded.get('radiometric_identity')!=identity:
            raise ValueError(f'归一化工作目录身份冲突，禁止复用：{workspace}')


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def pair_tiles(reference,target):
    refs=sorted(Path(reference).glob('*.tif'));targets=sorted(Path(target).glob('*.tif'))
    if not refs or [p.name for p in refs]!=[p.name for p in targets]:
        raise ValueError(f'IR-MAD 缺少严格配对瓦片：{reference} / {target}')
    pairs=list(zip(refs,targets))
    for ref,src in pairs:
        with rasterio.open(ref) as a,rasterio.open(src) as b:
            if (a.crs!=b.crs or a.transform!=b.transform or a.shape!=b.shape or a.count!=b.count):
                raise ValueError(f'IR-MAD 瓦片像素网格不一致，禁止额外重采样：{ref} / {src}')
            if a.count!=3 or a.dtypes!=('uint8',)*3 or b.dtypes!=('uint8',)*3:
                raise ValueError('IR-MAD 已验证实现要求三波段 uint8 RGB，不进行隐式转换')
            if b.tags().get('RRN_METHOD') or a.tags().get('RRN_METHOD'):
                raise ValueError('IR-MAD 输入已归一化，禁止链式或重复归一化')
    return pairs


def normalize_pair(pairs,out,*,reference_period=REFERENCE):
    """Fast iteration sample; final NCP/PIF/TLS still use the entire population."""
    config=configuration(True,reference_period)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter();path=out/'paired_valid_rgb.bin';counts=[]
    with path.open('xb') as handle:
        for ref,target in pairs:
            count=0
            for _,_,z in core.paired_blocks(ref,target):handle.write(z.tobytes());count+=len(z)
            counts.append(dict(tile=target.name,valid_pairs=count,reference=str(ref),target=str(target)))
    n=sum(c['valid_pairs'] for c in counts)
    if n<PARAMETERS['minimum_common_pixels']:raise ValueError('IR-MAD 共同有效像元不足 1000')
    save(out/'paired_tiles.json',counts)
    z=np.memmap(path,mode='r',dtype='uint8',shape=(n,6))
    try:
        with threadpool_limits(limits=1):
            sampling_start=time.perf_counter()
            sample=core.uniform_iteration_sample(z,PARAMETERS['fit_sample_limit'])
            iteration_pixels=len(sample)
            sampling_seconds=time.perf_counter()-sampling_start
            print(f'[Fast IR-MAD] iteration_pixels={iteration_pixels} / valid_pixels={n}; final PIF/TLS=all',flush=True)
            fit_start=time.perf_counter()
            model,trace,selected,converged=core.fit_irmad(sample,max_iterations=30,tolerance=.01)
            iterations_seconds=time.perf_counter()-fit_start
            del sample
            pif_start=time.perf_counter()
            moments=core.Moments(6)
            for i in range(0,len(z),core.CHUNK):
                block=np.asarray(z[i:i+core.CHUNK],dtype='float64')
                moments.add(block[core.ncp(block,model)>.95])
            gain,offset,correlations=core.tls(*moments.result())
            pif_tls_seconds=time.perf_counter()-pif_start
    finally:
        z._mmap.close()
    result=dict(config=config,selected_iteration=selected,converged=converged,
                iteration_pixels=iteration_pixels,fit_sample_limit=PARAMETERS['fit_sample_limit'],
                sampling=PARAMETERS['sampling'],
                timings=dict(sampling_seconds=sampling_seconds,iterations_seconds=iterations_seconds,
                             full_population_pif_tls_seconds=pif_tls_seconds),
                pixels=n,pif_pixels=int(moments.n),pif_fraction=moments.n/n,
                gain=gain.tolist(),offset=offset.tolist(),pif_correlation=correlations,
                trace=trace,model={k:v.tolist() for k,v in model.items()},pif_by_tile=[])
    with threadpool_limits(limits=1):
        for ref,target in pairs:
            pif=out/'pif'/target.name;pif.parent.mkdir(exist_ok=True)
            with rasterio.open(target) as ds:
                profile=ds.profile.copy();profile.update(count=1,dtype='float32',nodata=-1,compress='deflate')
                with rasterio.open(pif,'w',**profile) as dst:
                    count=valid_count=0
                    for w,valid,block in core.paired_blocks(ref,target):
                        probabilities=core.ncp(block.astype('float64'),model)
                        output=np.full(valid.shape,-1,dtype='float32');output[valid]=probabilities
                        dst.write(output,1,window=w)
                        count+=int((probabilities>.95).sum());valid_count+=len(block)
            result['pif_by_tile'].append(dict(tile=target.name,valid=valid_count,pif=count))
    result['outputs']=[core.write_normalized(t,out/'normalized_tiles'/t.name,gain,offset) for _,t in pairs]
    for row in result['outputs']:
        with rasterio.open(row['output'], 'r+') as dst:
            dst.update_tags(RRN_REFERENCE=reference_period)
    result['seconds']=time.perf_counter()-tick
    save(out/'normalization.json',result)
    return result


@dataclass(frozen=True)
class PreparedInput:
    source: Path
    metadata: dict


def prepare_period(period,sources,cache_root,*,enabled=False,origin=None,reference_period=REFERENCE):
    """Raw analysis directories -> one period's cached images and provenance.

    Source directories must be pre-radiometric analysis tiles. The reference
    entry is never replaced in this mapping; all target fits are independent.
    """
    source=Path(sources[period]).resolve()
    base=dict(config=configuration(enabled,reference_period),period=str(period),raw_analysis_source=str(source))
    if not enabled:return PreparedInput(source,dict(base,status='disabled'))
    if reference_period not in sources:raise ValueError(f'开启 IR-MAD 必须提供参考期 {reference_period}')
    reference=Path(sources[reference_period]).resolve()
    if period==reference_period:
        print(f'[IR-MAD] {reference_period}: 原始参考期，不执行归一化',flush=True)
        return PreparedInput(source,dict(base,status='reference_raw',reference=str(reference)))
    pairs=pair_tiles(reference,source)
    identity=dict(config=configuration(True,reference_period),period=str(period),origin=origin,
                  pairs=[dict(reference=fingerprint(r),target=fingerprint(t)) for r,t in pairs])
    key=digest(identity);root=Path(cache_root)/key;marker=root/'complete.json'
    if marker.is_file():
        stored=json.loads(marker.read_text(encoding='utf8'))
        if stored.get('identity')==identity and stored.get('files') and all(
                Path(row['path']).is_file() and fingerprint(row['path'])==row for row in stored.get('files',[])):
            print(f'[IR-MAD] {period} → {reference_period}: cache hit {key}',flush=True)
            return PreparedInput(root/'normalized_tiles',dict(base,status='cache_hit',cache_identity=key,audit=str(root/'normalization.json')))
        raise RuntimeError(f'IR-MAD 缓存损坏：{root}；请移走该缓存后重试，禁止回退原图')
    # Independent attempt directories prevent partial writes being mistaken for a hit.
    attempt=Path(cache_root)/f'{key}.pending-{uuid.uuid4().hex}'
    try:
        print(f'[IR-MAD] {period} → {reference_period}: 独立 PIF / TLS 拟合',flush=True)
        normalize_pair(pairs,attempt,reference_period=reference_period)
        tiles=attempt/'normalized_tiles'
        for p in source.glob('valid_observation.*'):shutil.copy2(p,tiles/p.name)
        save(tiles/'normalized_cache.json',dict(irmad_identity=key,source=str(source)))
        if root.exists():raise FileExistsError(f'IR-MAD incomplete or concurrent cache: {root}')
        attempt.rename(root)
        audit=json.loads((root/'normalization.json').read_text(encoding='utf8'))
        for row in audit['outputs']:row['output']=str(root/'normalized_tiles'/Path(row['output']).name)
        save(root/'normalization.json',audit)
        files=[fingerprint(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='paired_valid_rgb.bin']
        save(marker,dict(identity=identity,files=files))
    except Exception as exc:
        if attempt.is_dir():save(attempt/'failure.json',dict(error=str(exc),identity=identity))
        raise RuntimeError(f'IR-MAD {period} → {reference_period} 失败：{exc}') from exc
    return PreparedInput(root/'normalized_tiles',dict(base,status='normalized',cache_identity=key,audit=str(root/'normalization.json')))
