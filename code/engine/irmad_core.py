"""Validated experimental IR-MAD/PIF/TLS kernel, promoted without numeric changes.

Source: experiments/radiometric_normalization_ab/irmad_rrn.py.
Only orchestration and cache paths live outside this module.
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.special import gammaincc
import time

CHUNK = 262144

def uniform_iteration_sample(z, limit):
    """One deterministic midpoint per equal-sized bin in the valid-pixel stream."""
    if limit is None or len(z)<=limit:
        return z
    if limit<1:raise ValueError('Sample limit must be positive')
    indices=((2*np.arange(limit,dtype=np.int64)+1)*len(z))//(2*limit)
    return z[indices]

def windows(ds):
    for row in range(0, ds.height, 128):
        yield Window(0, row, ds.width, min(128, ds.height-row))


def paired_blocks(ref, target):
    with rasterio.open(ref) as a, rasterio.open(target) as b:
        for w in windows(a):
            valid = (a.read_masks(window=w)>0).all(0) & (b.read_masks(window=w)>0).all(0)
            x, y = a.read(window=w), b.read(window=w)
            yield w, valid, np.concatenate([x[:, valid].T, y[:, valid].T], axis=1)


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
