"""Independent IR-MAD/PIF relative radiometric normalization (no road/GT inputs).

Mathematics follows the IR-MAD and TLS workflow documented by SMByC/ArrNorm.
This implementation uses SVD CCA and full-population streaming sufficient statistics.
It does not vendor ArrNorm source. All arrays and outputs stay in the experiment.
"""
from pathlib import Path
import json
import time

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.special import gammaincc
from threadpoolctl import threadpool_limits

from run_experiment import ROOT, save

OUT = ROOT / 'irmad'
BASE = ROOT / 'A/_work/tasks/runs/pair_ab/grids/area/periods'
CHUNK = 262144


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def paired_tiles():
    dirs = [Path(read(BASE / p / 'input_manifest.json')['images']) for p in ('20250118', '20260203')]
    files = [sorted(p.glob('*.tif')) for p in dirs]
    if not files[0] or [p.name for p in files[0]] != [p.name for p in files[1]]:
        raise ValueError('Missing or unpaired baseline analysis tiles')
    for ref, target in zip(*files):
        with rasterio.open(ref) as a, rasterio.open(target) as b:
            if (a.crs != b.crs or a.transform != b.transform or a.shape != b.shape or a.count != b.count):
                raise ValueError(f'Not strictly co-gridded: {ref}, {target}; no automatic warp')
            if a.count != 3 or a.dtypes != ('uint8',)*3 or b.dtypes != ('uint8',)*3:
                raise ValueError('This experiment expects three uint8 RGB bands')
        yield ref, target


def windows(ds):
    for row in range(0, ds.height, 128):
        yield Window(0, row, ds.width, min(128, ds.height-row))


def paired_blocks(ref, target):
    with rasterio.open(ref) as a, rasterio.open(target) as b:
        for w in windows(a):
            valid = (a.read_masks(window=w)>0).all(0) & (b.read_masks(window=w)>0).all(0)
            x, y = a.read(window=w), b.read(window=w)
            yield w, valid, np.concatenate([x[:, valid].T, y[:, valid].T], axis=1)


def prepare_pixels(pairs, output_dir=OUT):
    """Cache every mutually valid RGB pair, without spatial or semantic sampling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'paired_valid_rgb.bin'
    if path.exists():
        raise FileExistsError(path)
    counts = []
    with path.open('wb') as f:
        for ref, target in pairs:
            count = 0
            for _, _, z in paired_blocks(ref, target):
                f.write(z.tobytes())
                count += len(z)
            counts.append(dict(tile=target.name, valid_pairs=count, reference=str(ref), target=str(target)))
            print('Cached', target.name, count, flush=True)
    n = sum(c['valid_pairs'] for c in counts)
    if n < 1000:
        raise ValueError('Insufficient common valid pixels')
    save(output_dir/'paired_tiles.json', counts)
    return np.memmap(path, mode='r', dtype='uint8', shape=(n, 6))


class Moments:
    def __init__(self, dimensions):
        self.n = 0.
        self.total = np.zeros(dimensions)
        self.cross = np.zeros((dimensions, dimensions))

    def add(self, z, weights=None):
        # DN values are small (0..255); float64 raw moments are well conditioned.
        z = np.asarray(z, dtype=np.float64)
        wz = z if weights is None else z*weights[:, None]
        self.n += len(z) if weights is None else float(weights.sum())
        self.total += wz.sum(0)
        self.cross += z.T @ wz

    def result(self):
        if self.n <= 10:
            raise ValueError('Degenerate effective population')
        mean = self.total/self.n
        cov = (self.cross - self.n*np.outer(mean, mean))/(self.n-1)
        return mean, (cov+cov.T)*.5


def cca(mean, cov):
    k = len(mean)//2
    def whiten(s):
        vals, vecs = np.linalg.eigh(s)
        if vals[0] <= vals[-1]*1e-10:
            raise ValueError('Rank deficient RGB covariance; do not silently regularize')
        return (vecs/np.sqrt(vals)) @ vecs.T
    wx, wy = whiten(cov[:k, :k]), whiten(cov[k:, k:])
    u, rho, vt = np.linalg.svd(wx @ cov[:k, k:] @ wy)
    return dict(mean=mean, A=wx@u, B=wy@vt.T, rho=np.clip(rho, 0, 1),
                variance=np.maximum(2*(1-rho), 1e-10))


def ncp(z, model):
    k = z.shape[1]//2
    mad = (z[:, :k]-model['mean'][:k])@model['A'] - (z[:, k:]-model['mean'][k:])@model['B']
    chi = (mad*mad/model['variance']).sum(1)
    return gammaincc(k/2, chi/2)


def fit_irmad(z, max_iterations=30, tolerance=.01):
    model, old, trace, candidates = None, np.zeros(z.shape[1]//2), [], []
    for iteration in range(1, max_iterations+1):
        tick = time.perf_counter()
        moments = Moments(z.shape[1])
        for start in range(0, len(z), CHUNK):
            block = np.asarray(z[start:start+CHUNK], dtype=np.float64)
            moments.add(block, None if model is None else ncp(block, model))
        model = cca(*moments.result())
        delta = float(np.max(np.abs(model['rho']-old)))
        trace.append(dict(iteration=iteration, delta=delta, rho=model['rho'].tolist(),
                          weight_sum=moments.n, seconds=time.perf_counter()-tick))
        candidates.append(model)
        print('IR-MAD', trace[-1], flush=True)
        if iteration > 1 and delta < tolerance:
            return model, trace, iteration, True
        old = model['rho']
    best = min(range(1, len(trace)), key=lambda i: trace[i]['delta']) if len(trace)>1 else 0
    return candidates[best], trace, best+1, False


def tls(mean, covariance):
    """Principal-axis orthogonal regression: target (second half) -> reference."""
    k = len(mean)//2
    slopes, intercepts, correlations = [], [], []
    for j in range(k):
        ids = [k+j, j]
        c = covariance[np.ix_(ids, ids)]
        _, vectors = np.linalg.eigh(c)
        v = vectors[:, -1]
        if abs(v[0]) < 1e-10 or c[0, 1] <= 0:
            raise ValueError('Invalid PIF orthogonal regression')
        slope = v[1]/v[0]
        slopes.append(slope)
        intercepts.append(mean[j]-slope*mean[k+j])
        correlations.append(c[0, 1]/np.sqrt(c[0, 0]*c[1, 1]))
    return np.array(slopes), np.array(intercepts), correlations


def write_normalized(source, dest, gain, offset):
    if dest.exists() or source.resolve() == dest.resolve():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clips = np.zeros((3, 2), dtype=np.int64)
    valid_count = 0
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        # Lossless compression avoids introducing another JPEG quantization step.
        profile.update(compress='deflate', photometric='RGB')
        lo, hi = 0, 255
        if src.nodata == 255: hi = 254
        if src.nodata == 0: lo = 1
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(dest, 'w', **profile) as dst:
            for w in windows(src):
                a = src.read(window=w)
                masks = src.read_masks(window=w)>0
                valid_count += int(masks.all(0).sum())
                for b in range(3):
                    values = gain[b]*a[b].astype('float64')+offset[b]
                    clips[b] += [np.count_nonzero((values<lo)&masks[b]), np.count_nonzero((values>hi)&masks[b])]
                    a[b][masks[b]] = np.rint(np.clip(values[masks[b]], lo, hi)).astype('uint8')
                dst.write(a, window=w)
                # Nodata-derived masks can differ by band. An explicit dataset mask
                # would override them, so copy only an actual shared dataset mask.
                if rasterio.enums.MaskFlags.per_dataset in src.mask_flag_enums[0]:
                    dst.write_mask(src.dataset_mask(window=w), window=w)
            dst.colorinterp, dst.descriptions = src.colorinterp, src.descriptions
            dst.scales, dst.offsets, dst.units = src.scales, src.offsets, src.units
            # Structural namespaces are regenerated by GDAL; stale DN statistics excluded.
            for b in range(0, src.count+1):
                for namespace in [None]+src.tag_namespaces(b):
                    if namespace in ('IMAGE_STRUCTURE', 'DERIVED_SUBDATASETS'): continue
                    tags = {k:v for k,v in src.tags(b, ns=namespace).items() if not k.startswith('STATISTICS_')}
                    if tags: dst.update_tags(b, ns=namespace, **tags)
            dst.update_tags(RRN_METHOD='IR-MAD PIF orthogonal TLS', RRN_REFERENCE='20250118')
    with rasterio.open(source) as a, rasterio.open(dest) as b:
        assert (a.crs, a.transform, a.shape, a.nodata, a.dtypes, a.colorinterp) == (b.crs, b.transform, b.shape, b.nodata, b.dtypes, b.colorinterp)
        for w in windows(a):
            assert np.array_equal(a.read_masks(window=w), b.read_masks(window=w))
            invalid = a.read_masks(window=w)==0
            assert np.array_equal(a.read(window=w)[invalid], b.read(window=w)[invalid])
    return dict(source=str(source), output=str(dest), valid_pixels=valid_count, clipped_low_high_per_band=clips.tolist(), grid_mask_nodata_verified=True)


def uniform_iteration_sample(z, limit):
    """One deterministic midpoint per equal-sized bin in the valid-pixel stream."""
    if limit is None or len(z)<=limit:
        return z
    if limit<1:raise ValueError('Sample limit must be positive')
    indices=((2*np.arange(limit,dtype=np.int64)+1)*len(z))//(2*limit)
    return z[indices]


def run(output_dir=OUT, fit_sample_limit=None):
    start = time.perf_counter()
    pairs = list(paired_tiles())
    z = prepare_pixels(pairs, output_dir)
    cache_seconds = time.perf_counter()-start
    tick = time.perf_counter()
    with threadpool_limits(limits=1):
        sample = uniform_iteration_sample(z, fit_sample_limit)
        sampling_seconds = time.perf_counter()-tick
        fit_start = time.perf_counter()
        model, trace, selected, converged = fit_irmad(sample)
        fit_seconds = time.perf_counter()-tick
        iterations_seconds = time.perf_counter()-fit_start
        pif_start = time.perf_counter()
        moments = Moments(6)
        for i in range(0, len(z), CHUNK):
            block = np.asarray(z[i:i+CHUNK], dtype='float64')
            moments.add(block[ncp(block, model)>.95])
        gain, offset, correlation = tls(*moments.result())
        pif_tls_seconds = time.perf_counter()-pif_start
    config = dict(method='IR-MAD CCA + full-population NCP PIF + orthogonal TLS',
                  reference='20250118', target='20260203', ncp_threshold=.95, max_iterations=30,
                  convergence_delta=.01, selected_iteration=selected, converged=converged,
                  pixels=len(z), pif_pixels=int(moments.n), pif_fraction=moments.n/len(z),
                  iteration_pixels=len(sample), fit_sample_limit=fit_sample_limit,
                  sampling='equal-bin midpoints of all common-valid pixels in original tile/row order',
                  gain=gain.tolist(), offset=offset.tolist(), pif_correlation=correlation,
                  trace=trace, model={k:v.tolist() for k,v in model.items()},
                  grid_strategy='Reuse strictly aligned RAW baseline analysis tiles; no new warp',
                  source='https://github.com/SMByC/ArrNorm', gt_or_road_inputs=False,
                  timings=dict(cache_seconds=cache_seconds, irmad_fit_seconds=fit_seconds,
                               sampling_seconds=sampling_seconds, iterations_seconds=iterations_seconds,
                               full_population_pif_tls_seconds=pif_tls_seconds))
    save(output_dir/'normalization.json', config)
    print('TLS', config['gain'], config['offset'], 'PIF', moments.n, flush=True)
    pif_counts = []
    tick = time.perf_counter()
    with threadpool_limits(limits=1):
        for ref, target in pairs:
            path = output_dir/'pif'/target.name
            path.parent.mkdir(exist_ok=True)
            with rasterio.open(target) as ds:
                profile = ds.profile.copy()
                profile.update(count=1, dtype='float32', nodata=-1, compress='deflate')
                with rasterio.open(path, 'w', **profile) as dst:
                    n, valid_n = 0, 0
                    for w, valid, block in paired_blocks(ref, target):
                        probabilities = ncp(block.astype('float64'), model)
                        output = np.full(valid.shape, -1, dtype='float32')
                        output[valid] = probabilities
                        dst.write(output, 1, window=w)
                        n += int((probabilities>.95).sum())
                        valid_n += len(block)
            pif_counts.append(dict(tile=target.name, valid=valid_n, pif=n))
            print('PIF map', target.name, n, flush=True)
    config['pif_by_tile'] = pif_counts
    config['timings']['pif_map_seconds'] = time.perf_counter()-tick
    tick = time.perf_counter()
    config['outputs'] = [write_normalized(t, output_dir/'normalized_tiles'/t.name, gain, offset) for _, t in pairs]
    save(output_dir/'normalization.json', config)
    # Full original T2 output for GIS use; inference uses the unchanged baseline grid above.
    config['outputs'].append(write_normalized(ROOT/'inputs/raw/20260203.tif', output_dir/'normalized_native/20260203.tif', gain, offset))
    config['timings']['write_verify_seconds'] = time.perf_counter()-tick
    config['timings']['normalization_total_seconds'] = time.perf_counter()-start
    save(output_dir/'normalization.json', config)
    print('NORMALIZATION COMPLETE', config['timings'], flush=True)


if __name__ == '__main__':
    run()
