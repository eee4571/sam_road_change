"""Experiment-only probability timing and exact ordered sample/LRU fingerprints."""
import ast
from collections import Counter
import hashlib
import inspect
import struct
import textwrap
import time

import numpy as np


class ProbabilityProbe:
    def __init__(self, *, raster_io=False):
        self.seconds = Counter()
        self.counts = Counter()
        self.blocks_per_call = Counter()
        self.block_visits = Counter()
        self.samples = hashlib.sha256()
        self.accesses = hashlib.sha256()
        self.miss_events = bytearray() if raster_io else None
        self.io_counts = Counter()
        self.io_seconds = 0.
        self.io_windows = hashlib.sha256()

    def blocks(self, br, bc):
        for a, b in zip(br, bc):
            self.block_visits[(int(a), int(b))] += 1

    def record(self, x, y, result):
        self.counts['values_at_calls'] += 1
        self.counts['sample_count'] += result.size
        for values in (x, y, result):
            array = np.ascontiguousarray(values)
            self.samples.update(str((array.shape, array.dtype.str)).encode())
            self.samples.update(array.tobytes())

    def report(self):
        result = dict(seconds=dict(self.seconds), counts=dict(self.counts),
                    blocks_per_call={str(k):v for k,v in sorted(self.blocks_per_call.items())},
                    unique_blocks=len(self.block_visits),
                    block_accesses={f'{r},{c}':n for (r,c),n in sorted(self.block_visits.items())},
                    repeated_accesses=sum(max(0,n-1) for n in self.block_visits.values()),
                    sample_sha256=self.samples.hexdigest(), cache_access_sha256=self.accesses.hexdigest())
        if self.miss_events is not None:
            result['raster_io'] = dict(self.io_counts, read_seconds=self.io_seconds,
                                      window_sha256=self.io_windows.hexdigest(),
                                      miss_trace_sha256=hashlib.sha256(self.miss_events).hexdigest())
        return result


class _RasterIOProbe:
    """Count actual sampling I/O separately from logical cache misses."""
    def __init__(self, dataset, probe):
        self.dataset, self.probe = dataset, probe

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def read(self, *args, **kwargs):
        started = time.perf_counter()
        values = self.dataset.read(*args, **kwargs)
        self.probe.io_seconds += time.perf_counter()-started
        self.probe.io_counts['read_count'] += 1
        self.probe.io_counts['max_read_bytes'] = max(self.probe.io_counts['max_read_bytes'],
                                                    values.data.nbytes + np.ma.getmaskarray(values).nbytes)
        self.probe.io_windows.update(repr(kwargs.get('window')).encode())
        return values


def _instrument(function, replacements, globals_):
    source = textwrap.dedent(inspect.getsource(function))
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    tree = ast.parse(source)
    namespace = dict(globals_, time=time)
    exec(compile(tree, '<probability-stage-probe>', 'exec'), namespace)
    return namespace[function.__name__]


def install(module, *, detailed=False, raster_io=False):
    cls = module.WindowedProbability
    original_init, original_values = cls.__init__, cls._values_at
    original_cached = cls._cached_block

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._probability_probe = ProbabilityProbe(raster_io=raster_io)
        if raster_io:
            self.dataset = _RasterIOProbe(self.dataset, self._probability_probe)
    cls.__init__ = init

    if detailed:
        def stage(name, indentation='    '):
            return f"{indentation}probe.seconds['{name}'] += time.perf_counter()-tick\n{indentation}tick = time.perf_counter()\n"
        values_source = textwrap.dedent(inspect.getsource(original_values))

        def gather_statement(*choices):
            # Preserve whichever exact production/oracle statement is present;
            # instrumentation must not replace pixel-gather semantics itself.
            present = [choice for choice in choices if values_source.count(choice) == 1]
            assert len(present) == 1, (choices, present)
            return present[0]

        ram_gather = gather_statement(
            '        picked = self._ram[rr, cc].astype(float).filled(np.nan)',
            '        picked = self._gather_pixels(self._ram, rr, cc)')
        single_gather = gather_statement(
            '            picked = values[rr-br*height, cc-bc*width].astype(float).filled(np.nan)',
            '            picked = self._gather_pixels(values, rr-br*height, cc-bc*width)')
        group_gather = next(line for line in values_source.splitlines()
                            if line.lstrip().startswith('picked[take] = '))
        group_indent = group_gather[:len(group_gather)-len(group_gather.lstrip())]
        grouping_start = gather_statement(
            '            unique, groups = np.unique',
            '            column_count = (self.dataset.width + width - 1)//width')
        replacements = [
            ('    x, y = self.to_raster.transform', '    probe = self._probability_probe\n    tick = time.perf_counter()\n    x, y = self.to_raster.transform'),
            ('    cols, rows = self.inverse', stage('crs_transform')+'    cols, rows = self.inverse'),
            ('    inside = ', stage('affine_row_col')+'    inside = '),
            ('    if not inside.any():\n        return result', "    if not inside.any():\n        probe.seconds['inside_mask'] += time.perf_counter()-tick\n        return result"),
            ('    rr, cc = ', stage('inside_mask')+'    rr, cc = '),
            ('    if self._ram is not None:', stage('inside_indices')+'    if self._ram is not None:'),
            ('        if (block_rows ==', stage('block_id', '        ')+'        if (block_rows =='),
            ('            br, bc = int(block_rows[0]), int(block_cols[0])', "            probe.blocks(block_rows[:1], block_cols[:1])\n            br, bc = int(block_rows[0]), int(block_cols[0])"),
            ('            values = self._cached_block(br, bc)', "            tick = time.perf_counter()\n            values = self._cached_block(br, bc)\n"+stage('cache_dispatch', '            ').rstrip()),
            (grouping_start, "            tick = time.perf_counter()\n"+grouping_start),
            ('            picked = np.full(rr.shape, np.nan)', stage('block_group_unique', '            ')+"            probe.blocks(unique[:,0], unique[:,1])\n            picked = np.full(rr.shape, np.nan)"),
            (group_gather, group_gather+f"\n{group_indent}probe.seconds['gather_groups'] += time.perf_counter()-tick\n{group_indent}tick = time.perf_counter()"),
            ('    result[inside] = picked / self.divisor', "    tick = time.perf_counter()\n    result[inside] = picked / self.divisor\n    probe.seconds['refill_divisor'] += time.perf_counter()-tick"),
            (ram_gather, ram_gather+"\n        probe.seconds['gather_ram'] += time.perf_counter()-tick"),
            (single_gather, single_gather+"\n            probe.seconds['gather_single'] += time.perf_counter()-tick"),
        ]
        if 'blocks = self._cached_blocks(unique)' in values_source:
            # Iterator advancement includes the batched read/copy and unchanged
            # logical cache operations. Do not mislabel that aggregate as I/O.
            loop = next(line for line in values_source.splitlines()
                        if ' in zip(unique, starts, stops, blocks):' in line)
            loop_indent = loop[:len(loop)-len(loop.lstrip())]
            replacements.append((loop, loop_indent+'tick = time.perf_counter()\n'+loop+'\n'+stage('cache_dispatch', group_indent).rstrip()))
        else:
            replacements.append(('                values = self._cached_block(int(br), int(bc))', "                tick = time.perf_counter()\n                values = self._cached_block(int(br), int(bc))\n"+stage('cache_dispatch', '                ').rstrip()))
        sampled = _instrument(original_values, replacements, module.__dict__)
        cached_replacements = [
            ('    key = (br, bc)', '    probe = self._probability_probe\n    tick = time.perf_counter()\n    key = (br, bc)'),
            ('    if values is None:', stage('cache_lookup')+"    if values is None:\n        probe.counts['cache_miss'] += 1"),
            ('        self._cache_bytes +=', "        probe.seconds['cache_miss_data'] += time.perf_counter()-tick\n        tick = time.perf_counter()\n        self._cache_bytes +="),
            ('    self._blocks[key] = values', "    else:\n        probe.counts['cache_hit'] += 1\n        probe.seconds['cache_hit'] += time.perf_counter()-tick\n        tick = time.perf_counter()\n    self._blocks[key] = values"),
            ('    while self._cache_bytes >', stage('cache_account_insert')+'    while self._cache_bytes >'),
            ('        _, old = self._blocks.popitem(last=False)', "        probe.counts['cache_eviction'] += 1\n        _, old = self._blocks.popitem(last=False)"),
            ('    return values', "    probe.seconds['cache_eviction'] += time.perf_counter()-tick\n    return values"),
        ]
        cached = _instrument(original_cached, cached_replacements, module.__dict__)
    else:
        sampled, cached = original_values, original_cached

    def values_at(self, x, y):
        if raster_io:
            # A run cannot cross a sampling-call boundary or read future axes.
            self._probability_probe.miss_events.extend(struct.pack('<ii?', -1, -1, False))
        count_before = self._probability_probe.counts['cached_block_calls']
        started = time.perf_counter()
        result = sampled(self, x, y)
        self._probability_probe.seconds['values_at_total'] += time.perf_counter()-started
        self._probability_probe.blocks_per_call[self._probability_probe.counts['cached_block_calls']-count_before] += 1
        self._probability_probe.record(x, y, result)
        return result

    def cached_block(self, br, bc, **kwargs):
        probe = self._probability_probe
        miss = (br,bc) not in self._blocks
        if raster_io:
            probe.miss_events.extend(struct.pack('<ii?', int(br), int(bc), miss))
        started = time.perf_counter()
        if not detailed:
            probe.counts['cache_miss' if miss else 'cache_hit'] += 1
            probe.block_visits[(br,bc)] += 1
            size = len(self._blocks)
        values = cached(self, br, bc, **kwargs)
        if not detailed:
            probe.counts['cache_eviction'] += max(0,size+int(miss)-len(self._blocks))
        elapsed = time.perf_counter()-started
        probe.seconds['cached_block_total'] += elapsed
        probe.seconds['cache_miss_path' if miss else 'cache_hit_path'] += elapsed
        probe.counts['cached_block_calls'] += 1
        probe.accesses.update(struct.pack('<qq', int(br), int(bc)))
        return values
    cls._values_at, cls._cached_block = values_at, cached_block


def reports(scenes):
    result = []
    for scene in scenes:
        sampler = scene.probability
        report = sampler._probability_probe.report()
        report['grid'] = dict(crs=sampler.crs.to_wkt(), transform=tuple(sampler.dataset.transform),
                              width=sampler.dataset.width,height=sampler.dataset.height,
                              block_shape=sampler._block_shape, ram=sampler._ram is not None,
                              cache_limit=sampler._cache_limit)
        report['final_lru_sha256'] = hashlib.sha256(repr(list(sampler._blocks.keys())).encode()).hexdigest()
        report['final_cache_bytes'] = sampler._cache_bytes
        result.append(report)
    return result
