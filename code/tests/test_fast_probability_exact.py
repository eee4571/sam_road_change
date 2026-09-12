"""Byte-exact probability regressions on tiny rasters; no project or model runs."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import Affine, from_origin

from engine.fast_auto_change import WindowedProbability
from probability_probe import install


def _original_values_at(self, x, y):
    """Frozen pre-P2 implementation: never adapt this oracle to an optimization."""
    x, y = self.to_raster.transform(np.asarray(x), np.asarray(y))
    cols, rows = self.inverse * (np.asarray(x), np.asarray(y))
    cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
    result = np.full(cols.shape, np.nan, dtype=float)
    inside = (cols >= 0) & (rows >= 0) & (cols < self.dataset.width) & (rows < self.dataset.height)
    if not inside.any():
        return result
    rr, cc = rows[inside], cols[inside]
    if self._ram is not None:
        picked = self._ram[rr, cc].astype(float).filled(np.nan)
    else:
        height, width = self._block_shape
        block_rows, block_cols = rr//height, cc//width
        if (block_rows == block_rows[0]).all() and (block_cols == block_cols[0]).all():
            br, bc = int(block_rows[0]), int(block_cols[0])
            values = self._cached_block(br, bc)
            picked = values[rr-br*height, cc-bc*width].astype(float).filled(np.nan)
        else:
            unique, groups = np.unique(np.column_stack((block_rows, block_cols)), axis=0, return_inverse=True)
            picked = np.full(rr.shape, np.nan)
            for group, (br, bc) in enumerate(unique):
                values = self._cached_block(int(br), int(bc))
                take = groups == group
                picked[take] = values[rr[take]-br*height, cc[take]-bc*width].astype(float).filled(np.nan)
    result[inside] = picked / self.divisor
    return result


def _original_cached_block(self, br, bc):
    """Frozen pre-P2 cache: visit ordering and byte budget are part of the oracle."""
    key = (br, bc)
    values = self._blocks.pop(key, None)
    if values is None:
        height, width = self._block_shape
        values = self.dataset.read(1, window=rasterio.windows.Window(
            bc*width, br*height, min(width, self.dataset.width-bc*width),
            min(height, self.dataset.height-br*height)), masked=True)
        self._cache_bytes += values.data.nbytes + np.ma.getmaskarray(values).nbytes
    self._blocks[key] = values
    while self._cache_bytes > self._cache_limit and self._blocks:
        _, old = self._blocks.popitem(last=False)
        self._cache_bytes -= old.data.nbytes + np.ma.getmaskarray(old).nbytes
    return values


class _OriginalProbability(WindowedProbability):
    _values_at = _original_values_at
    _cached_block = _original_cached_block


class _RecordingDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.reads = []

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def read(self, *args, **kwargs):
        window = kwargs.get("window")
        self.reads.append((args, None if window is None else tuple(window.flatten()),
                           kwargs.get("masked"), kwargs.get("out_shape")))
        return self.dataset.read(*args, **kwargs)


@contextmanager
def _probe_on_reference(*, detailed, base=_OriginalProbability):
    """Probe monkeypatches must not leak to the production class or other tests."""
    class Reference(base):
        pass

    module = SimpleNamespace(WindowedProbability=Reference, np=np, rasterio=rasterio)
    originals = {name: getattr(Reference, name)
                 for name in ("__init__", "_values_at", "_cached_block")}
    try:
        install(module, detailed=detailed)
        yield Reference
    finally:
        for name, function in originals.items():
            setattr(Reference, name, function)


class ProbabilityExactTests(unittest.TestCase):
    def _raster(self, root, *, dtype="uint8", nodata=None, explicit_mask=False,
                transform=None, crs=32650, strip=False):
        path = Path(root)/"probability.tif"
        data = (np.arange(35*39).reshape(35, 39) % 251).astype(dtype)
        if np.issubdtype(data.dtype, np.floating):
            data /= np.array(257, dtype=dtype)
            data[2, 3], data[4, 5], data[5, 6] = np.nan, np.inf, -np.inf
        if nodata is not None:
            data[3:7, 4:10] = nodata
        if transform is None:
            transform = from_origin(500000, 3500000, 1, 1)
        blocks = dict(tiled=False, blockysize=1) if strip else dict(tiled=True, blockxsize=16, blockysize=16)
        with rasterio.open(path, "w", driver="GTiff", width=39, height=35,
                           count=1, dtype=dtype, nodata=nodata, crs=crs,
                           transform=transform, **blocks) as dataset:
            dataset.write(data, 1)
            if explicit_mask:
                mask = np.full(data.shape, 255, dtype=np.uint8)
                mask[1:9, 2:8] = 0
                mask[-1, -1] = 0
                dataset.write_mask(mask)
        return path

    def _coordinates(self, sampler, cols, rows):
        x, y = sampler.dataset.transform * (np.asarray(cols, dtype=float), np.asarray(rows, dtype=float))
        transform = Transformer.from_crs(sampler.crs, sampler.metric_crs, always_xy=True)
        return transform.transform(x, y)

    def _assert_bytes(self, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype.str, expected.dtype.str)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def _assert_cache(self, actual, expected):
        self.assertEqual(list(actual._blocks), list(expected._blocks))
        self.assertEqual(actual._cache_bytes, expected._cache_bytes)
        self.assertLessEqual(actual._cache_bytes, actual._cache_limit)
        self.assertEqual(actual.dataset.reads, expected.dataset.reads)

    def _exercise(self, path, *, cache_limit, ram_limit=0, metric_crs=32650,
                  implementation=WindowedProbability):
        actual = implementation(path, metric_crs, ram_limit_bytes=ram_limit,
                                cache_limit_bytes=cache_limit)
        expected = _OriginalProbability(path, metric_crs, ram_limit_bytes=ram_limit,
                                        cache_limit_bytes=cache_limit)
        for sampler in (actual, expected):
            sampler.dataset = _RecordingDataset(sampler.dataset)
        # Unsorted, repeated blocks; nodata; partial bottom/right tiles; misses/hits.
        batches = [
            ([38.25, 2.25, 17.25, 2.25, 34.25, 5.25, 18.25, 5.25],
             [34.5, 2.5, 17.5, 2.5, 1.5, 4.5, 1.5, 4.5]),
            ([1.25, 2.25, 3.25, 5.25], [1.5, 2.5, 2.5, 4.5]),
            ([38.25, 38.25, 32.25], [34.5, 34.5, 32.5]),
            ([38.25, 32.25], [34.5, 32.5]),
            ([-1.0, 40.0, -2.0], [3.5, 36.5, -1.0]),
            ([], []),
            ([0.0, 16.0, 32.0, 39.0, -0.0001], [0.0, 16.0, 32.0, 35.0, 1.0]),
        ]
        try:
            self.assertEqual(actual.divisor, expected.divisor)
            for _ in range(2):
                for cols, rows in batches:
                    x, y = self._coordinates(expected, cols, rows)
                    self._assert_bytes(actual._values_at(x, y), expected._values_at(x, y))
                    self._assert_cache(actual, expected)
            # Same coordinates, center/background split, percentile/rank and order.
            coordinates = []
            for cols, rows in batches[:3]:
                x, y = self._coordinates(expected, cols, rows)
                coordinates.append((np.column_stack((x, y)), min(3, len(x))))
            a = actual.sample_axes([], [], 3., coordinates=coordinates)
            b = expected.sample_axes([], [], 3., coordinates=coordinates)
            self.assertEqual(a, b)
            self._assert_cache(actual, expected)
            if hasattr(actual, "_probability_probe"):
                return actual._probability_probe.report()
        finally:
            actual.close()
            expected.close()

    def test_uint8_block_order_eviction_and_partial_edges(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._raster(root)
            for limit in (0, 1, 511, 512, 1024, 10000):
                with self.subTest(cache_limit=limit):
                    self._exercise(path, cache_limit=limit)

    def test_float_nodata_explicit_masks_and_ram(self):
        for dtype, nodata, explicit in (("uint8", 255, False),
                                         ("uint8", None, True),
                                         ("float32", -9999., False),
                                         ("float64", None, True)):
            with self.subTest(dtype=dtype, nodata=nodata, explicit=explicit):
                with tempfile.TemporaryDirectory() as root:
                    path = self._raster(root, dtype=dtype, nodata=nodata, explicit_mask=explicit)
                    for ram in (0, 1000000):
                        self._exercise(path, cache_limit=2048, ram_limit=ram)

    def test_rotated_affine_and_different_crs(self):
        with tempfile.TemporaryDirectory() as root:
            transform = Affine(0.00002, 0.000003, 116.9, 0.000002, -0.00002, 31.7)
            path = self._raster(root, transform=transform, crs=4326, explicit_mask=True)
            self._exercise(path, cache_limit=1024, metric_crs=32650)

    def test_random_multiblock_repeats_and_outside_are_byte_exact(self):
        for strip in (False, True):
            with self.subTest(strip=strip), tempfile.TemporaryDirectory() as root:
                path = self._raster(root, nodata=255, explicit_mask=True, strip=strip)
                for cache_limit in (0, 512, 4096):
                    samplers = [cls(path, 32650, ram_limit_bytes=0, cache_limit_bytes=cache_limit)
                                for cls in (WindowedProbability, _OriginalProbability)]
                    for sampler in samplers:
                        sampler.dataset = _RecordingDataset(sampler.dataset)
                    try:
                        rng = np.random.default_rng(193306)
                        for _ in range(3):
                            columns = rng.integers(-3, 43, 1024).astype(float)+0.25
                            rows = rng.integers(-3, 39, 1024).astype(float)+0.5
                            # Deliberate duplicates stay in their exact original positions,
                            # intermixed with all blocks and outside pixels.
                            indices = rng.permutation(np.tile(np.arange(1024), 4))
                            x, y = self._coordinates(samplers[1], columns[indices], rows[indices])
                            x, y = np.asarray(x).reshape(64, 64), np.asarray(y).reshape(64, 64)
                            self._assert_bytes(samplers[0]._values_at(x, y), samplers[1]._values_at(x, y))
                            self._assert_cache(samplers[0], samplers[1])
                    finally:
                        for sampler in samplers:
                            sampler.close()

    def test_scalar_and_array_masks_are_identical(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._raster(root, dtype="float32")
            samplers = [cls(path, 32650) for cls in (WindowedProbability, _OriginalProbability)]
            try:
                original = samplers[1]._ram.data.copy()
                partial = np.zeros(original.shape, dtype=bool)
                partial[2:6, 3:7] = True
                for mask in (np.ma.nomask, False, True, np.zeros(original.shape, dtype=bool),
                             np.ones(original.shape, dtype=bool), partial):
                    for sampler in samplers:
                        sampler._ram = np.ma.array(original.copy(), mask=mask)
                    x, y = self._coordinates(samplers[1], [3.25, 5.25, 6.25, 38.25], [2.5, 4.5, 5.5, 34.5])
                    self._assert_bytes(samplers[0]._values_at(x, y), samplers[1]._values_at(x, y))
            finally:
                for sampler in samplers:
                    sampler.close()

    def test_detailed_and_trace_probes_preserve_original_values_and_cache(self):
        original_methods = tuple(getattr(WindowedProbability, name)
                                 for name in ("__init__", "_values_at", "_cached_block"))
        with tempfile.TemporaryDirectory() as root:
            path = self._raster(root, nodata=255, explicit_mask=True)
            reports = []
            for base in (_OriginalProbability, WindowedProbability):
                for detailed in (False, True):
                    with self.subTest(base=base.__name__, detailed=detailed):
                        with _probe_on_reference(detailed=detailed, base=base) as implementation:
                            reports.append(self._exercise(path, cache_limit=512, implementation=implementation))
            for report in reports[1:]:
                for key in ("counts", "blocks_per_call", "unique_blocks", "block_accesses",
                            "repeated_accesses", "sample_sha256", "cache_access_sha256"):
                    self.assertEqual(reports[0][key], report[key], key)
            self.assertGreater(reports[0]["counts"]["cache_hit"], 0)
            self.assertGreater(reports[0]["counts"]["cache_miss"], 0)
            self.assertGreater(reports[0]["counts"]["cache_eviction"], 0)
        self.assertEqual(original_methods, tuple(getattr(WindowedProbability, name)
                         for name in ("__init__", "_values_at", "_cached_block")))


if __name__ == "__main__":
    unittest.main()
