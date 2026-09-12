"""Byte-exact probability regressions on tiny rasters; no project or model runs."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import gc
import tempfile
import unittest
import weakref

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
    def __init__(self, dataset, *, retain_arrays=False):
        self.dataset = dataset
        self.reads = []
        self.arrays = []
        self.array_refs = []
        self.retain_arrays = retain_arrays

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def read(self, *args, **kwargs):
        window = kwargs.get("window")
        self.reads.append((args, None if window is None else tuple(window.flatten()),
                           kwargs.get("masked"), kwargs.get("out_shape")))
        result = self.dataset.read(*args, **kwargs)
        self.array_refs.append(weakref.ref(result))
        if self.retain_arrays:
            self.arrays.append(result)
        return result


def _record_logical_cache(sampler):
    original = sampler._cached_block
    sampler.cache_events = []

    def cached(br, bc, **kwargs):
        key = (br, bc)
        miss = key not in sampler._blocks
        old_size = len(sampler._blocks)
        result = original(br, bc, **kwargs)
        evictions = max(0, old_size+int(miss)-len(sampler._blocks))
        sampler.cache_events.append((key, miss, evictions, tuple(sampler._blocks), sampler._cache_bytes))
        return result
    sampler._cached_block = cached


@contextmanager
def _probe_on_reference(*, detailed, base=_OriginalProbability, raster_io=False):
    """Probe monkeypatches must not leak to the production class or other tests."""
    class Reference(base):
        pass

    module = SimpleNamespace(WindowedProbability=Reference, np=np, rasterio=rasterio)
    originals = {name: getattr(Reference, name)
                 for name in ("__init__", "_values_at", "_cached_block")}
    try:
        install(module, detailed=detailed, raster_io=raster_io)
        yield Reference
    finally:
        for name, function in originals.items():
            setattr(Reference, name, function)


class ProbabilityExactTests(unittest.TestCase):
    def _raster(self, root, *, dtype="uint8", nodata=None, explicit_mask=False,
                transform=None, crs=32650, strip=False, strip_height=1):
        path = Path(root)/"probability.tif"
        data = (np.arange(35*39).reshape(35, 39) % 251).astype(dtype)
        if np.issubdtype(data.dtype, np.floating):
            data /= np.array(257, dtype=dtype)
            data[2, 3], data[4, 5], data[5, 6] = np.nan, np.inf, -np.inf
        if nodata is not None:
            data[3:7, 4:10] = nodata
        if transform is None:
            transform = from_origin(500000, 3500000, 1, 1)
        blocks = dict(tiled=False, blockysize=strip_height) if strip else dict(tiled=True, blockxsize=16, blockysize=16)
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
        if hasattr(actual, 'cache_events'):
            self.assertEqual(actual.cache_events, expected.cache_events)
        for key in actual._blocks:
            self._assert_bytes(actual._blocks[key].data, expected._blocks[key].data)
            self._assert_bytes(np.ma.getmaskarray(actual._blocks[key]), np.ma.getmaskarray(expected._blocks[key]))
            self._assert_bytes(actual._blocks[key].fill_value, expected._blocks[key].fill_value)
        if actual._block_shape[1] != actual.dataset.width or actual._STRIP_READ_BATCH == 1:
            self.assertEqual(actual.dataset.reads, expected.dataset.reads)
        else:
            # Physical reads deliberately differ. Expanding each merged strip
            # window must recover every original read in exactly the same order.
            expanded = []
            block_height = actual._block_shape[0]
            for args, window, masked, out_shape in actual.dataset.reads:
                self.assertIsNotNone(window)
                col, row, width, height = map(int, window)
                self.assertEqual(col, 0)
                self.assertEqual(row % block_height, 0)
                self.assertEqual(width, actual.dataset.width)
                self.assertEqual(masked, True)
                self.assertIsNone(out_shape)
                for offset in range(0, height, block_height):
                    expanded.append((args, (col, row+offset, width, min(block_height, height-offset)), masked, out_shape))
            self.assertEqual(expanded, expected.dataset.reads)

    def _exercise(self, path, *, cache_limit, ram_limit=0, metric_crs=32650,
                  implementation=WindowedProbability):
        actual = implementation(path, metric_crs, ram_limit_bytes=ram_limit,
                                cache_limit_bytes=cache_limit)
        expected = _OriginalProbability(path, metric_crs, ram_limit_bytes=ram_limit,
                                        cache_limit_bytes=cache_limit)
        for sampler in (actual, expected):
            sampler.dataset = _RecordingDataset(sampler.dataset)
            _record_logical_cache(sampler)
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
                        _record_logical_cache(sampler)
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

    def _strip_pair(self, path, *, batch, cache_limit, retain_arrays=False, temporary_limit=None):
        actual = WindowedProbability(path, 32650, ram_limit_bytes=0, cache_limit_bytes=cache_limit)
        expected = _OriginalProbability(path, 32650, ram_limit_bytes=0, cache_limit_bytes=cache_limit)
        actual._STRIP_READ_BATCH = batch
        if temporary_limit is not None:
            actual._STRIP_READ_MAX_BYTES = temporary_limit
        for sampler in (actual, expected):
            sampler.dataset = _RecordingDataset(sampler.dataset, retain_arrays=retain_arrays)
            _record_logical_cache(sampler)
        self.addCleanup(actual.close)
        self.addCleanup(expected.close)
        return actual, expected

    def _sample_rows(self, actual, expected, rows):
        # All columns remain within the same full-width strip; duplicates and
        # original coordinate order must survive grouped gathers unchanged.
        rows = np.asarray(rows, dtype=float)
        columns = np.resize(np.asarray([3.25, 5.25, 38.25, 17.25]), rows.shape)
        x, y = self._coordinates(expected, columns, rows+.5)
        self._assert_bytes(actual._values_at(x, y), expected._values_at(x, y))
        self._assert_cache(actual, expected)

    def test_strip_batch_sizes_masks_partial_blocks_and_logical_lru(self):
        configurations = [("uint8", None, False), ("uint8", 255, False),
                          ("uint8", None, True), ("float32", -9999., True),
                          ("float64", None, False)]
        for dtype, nodata, explicit in configurations:
            for block_height in (1, 4):
                with self.subTest(dtype=dtype, nodata=nodata, mask=explicit, block_height=block_height), tempfile.TemporaryDirectory() as root:
                    path = self._raster(root, dtype=dtype, nodata=nodata, explicit_mask=explicit,
                                        strip=True, strip_height=block_height)
                    block_bytes = 39*block_height*(np.dtype(dtype).itemsize+1)
                    for batch in (1, 8, 16, 32):
                        for cache_limit in (0, block_bytes-1, 2*block_bytes, 100000):
                            with self.subTest(batch=batch, cache_limit=cache_limit):
                                actual, expected = self._strip_pair(path, batch=batch, cache_limit=cache_limit)
                                try:
                                    self._sample_rows(actual, expected, list(reversed(range(35)))+[3, 3, 5, 34])
                                    self._sample_rows(actual, expected, [0, 1, 2, 5, 8, 9, 32, 34])
                                    self._sample_rows(actual, expected, list(range(35)))
                                    if batch > 1:
                                        self.assertLess(len(actual.dataset.reads), len(expected.dataset.reads))
                                finally:
                                    actual.close()
                                    expected.close()

    def test_strip_future_hit_evicted_gaps_and_call_boundaries(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._raster(root, strip=True, explicit_mask=True)
            for batch in (8, 16, 32):
                actual, expected = self._strip_pair(path, batch=batch, cache_limit=39*2*2)
                try:
                    # Block 2 is initially a hit, but inserting 0 and 1 evicts
                    # it. Safe lookahead stops before 2 and re-evaluates it later.
                    self._sample_rows(actual, expected, [2])
                    start = len(actual.dataset.reads)
                    self._sample_rows(actual, expected, [0, 1, 2, 3, 4])
                    self.assertEqual([entry[1] for entry in actual.dataset.reads[start:]],
                                     [(0, 0, 39, 2), (0, 2, 39, 3)])
                    self.assertEqual([event[1] for event in actual.cache_events[-5:]], [True]*5)
                    # Current hits, requested row gaps and call boundaries each
                    # force a physical read boundary even with a larger batch.
                    self._sample_rows(actual, expected, [10])
                    start = len(actual.dataset.reads)
                    self._sample_rows(actual, expected, [9, 10, 11, 14, 15])
                    self.assertEqual([entry[1] for entry in actual.dataset.reads[start:]],
                                     [(0, 9, 39, 1), (0, 11, 39, 1), (0, 14, 39, 2)])
                    start = len(actual.dataset.reads)
                    self._sample_rows(actual, expected, [20, 21])
                    self._sample_rows(actual, expected, [22, 23])
                    self.assertEqual([entry[1] for entry in actual.dataset.reads[start:]],
                                     [(0, 20, 39, 2), (0, 22, 39, 2)])
                finally:
                    actual.close()
                    expected.close()

    def test_strip_blocks_own_data_and_mask_and_temporary_byte_limit(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._raster(root, strip=True, strip_height=4, dtype="float32", explicit_mask=True)
            block_bytes = 4*39*5
            for temporary_limit in (block_bytes-1, block_bytes, 3*block_bytes, 1024*1024):
                actual, expected = self._strip_pair(path, batch=32, cache_limit=100000,
                                                  retain_arrays=True, temporary_limit=temporary_limit)
                try:
                    self._sample_rows(actual, expected, range(35))
                    for read, values in zip(actual.dataset.reads, actual.dataset.arrays):
                        size = values.data.nbytes+np.ma.getmaskarray(values).nbytes
                        self.assertLessEqual(size, max(temporary_limit, block_bytes))
                        if read[1][3] > actual._block_shape[0]:
                            for block in actual._blocks.values():
                                self.assertFalse(np.shares_memory(block.data, values.data))
                                self.assertFalse(np.shares_memory(np.ma.getmaskarray(block), np.ma.getmaskarray(values)))
                    for key, block in actual._blocks.items():
                        for other_key, other in actual._blocks.items():
                            if key != other_key:
                                self.assertFalse(np.shares_memory(block.data, other.data))
                                self.assertFalse(np.shares_memory(np.ma.getmaskarray(block), np.ma.getmaskarray(other)))
                finally:
                    actual.close()
                    expected.close()

    def test_strip_gather_or_read_failure_closes_iterator_and_releases_batch(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._raster(root, strip=True, explicit_mask=True)
            for failure in ('gather', 'read'):
                actual, expected = self._strip_pair(path, batch=8, cache_limit=39*2)
                iterators = []
                original_blocks = actual._cached_blocks

                def tracked_blocks(unique):
                    iterator = original_blocks(unique)
                    iterators.append(iterator)
                    return iterator

                actual._cached_blocks = tracked_blocks
                original_read = actual.dataset.read
                original_gather = actual._gather_pixels

                def fail(*args, **kwargs):
                    raise RuntimeError('synthetic '+failure+' failure')

                if failure == 'read':
                    actual.dataset.read = fail
                else:
                    actual._gather_pixels = fail
                x, y = self._coordinates(expected, [3.25]*8, np.arange(8)+.5)
                with self.assertRaisesRegex(RuntimeError, 'synthetic '+failure+' failure'):
                    actual._values_at(x, y)
                self.assertEqual(len(iterators), 1)
                self.assertIsNone(iterators[0].gi_frame)
                gc.collect()
                self.assertTrue(all(reference() is None for reference in actual.dataset.array_refs))
                actual.dataset.read = original_read
                actual._gather_pixels = original_gather
                # The sampler remains usable after a caught error. Its cache
                # remains bounded, and close releases the copied final block.
                self.assertLessEqual(actual._cache_bytes, actual._cache_limit)
                actual._values_at(x, y)
                actual.close()
                self.assertEqual(actual._cache_bytes, 0)
                self.assertFalse(actual._blocks)
                expected.close()

    def test_strip_detailed_and_trace_probe_preserve_logical_events_and_record_physical_io(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._raster(root, strip=True, nodata=255, explicit_mask=True)
            reports = []
            for base in (_OriginalProbability, WindowedProbability):
                for detailed in (False, True):
                    with self.subTest(base=base.__name__, detailed=detailed):
                        with _probe_on_reference(detailed=detailed, base=base, raster_io=True) as implementation:
                            reports.append(self._exercise(path, cache_limit=156, implementation=implementation))
            for report in reports[1:]:
                for key in ('counts', 'blocks_per_call', 'unique_blocks', 'block_accesses',
                            'repeated_accesses', 'sample_sha256', 'cache_access_sha256'):
                    self.assertEqual(reports[0][key], report[key], key)
                self.assertEqual(reports[0]['raster_io']['miss_trace_sha256'], report['raster_io']['miss_trace_sha256'])
            for report in reports[:2]:
                self.assertEqual(report['raster_io']['read_count'], report['counts']['cache_miss'])
            for report in reports[2:]:
                self.assertLess(report['raster_io']['read_count'], report['counts']['cache_miss'])
                self.assertGreater(report['raster_io']['read_seconds'], 0.)

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
