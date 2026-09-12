"""Replay recorded probability block accesses; no raster or model processing.

The theoretical run count uses observed logical misses.  The conservative count
replays the original LRU and only merges consecutive, already-requested strip
blocks that are all absent at the moment the first block is requested.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np


EVENT_DTYPE = np.dtype([("row", "<i4"), ("col", "<i4"), ("miss", "?")])


class _Cache:
    def __init__(self, grid, itemsize):
        self.width = int(grid["width"])
        self.height = int(grid["height"])
        self.block_height, self.block_width = map(int, grid["block_shape"])
        self.limit = int(grid["cache_limit"])
        self.itemsize = itemsize
        self.blocks = OrderedDict()
        self.bytes = 0
        self.counts = Counter()

    def block_bytes(self, row, col):
        assert col == 0 and 0 <= row*self.block_height < self.height
        return self.width * min(self.block_height, self.height-row*self.block_height) * (self.itemsize+1)

    def touch(self, row, col, recorded_miss):
        key = (row, col)
        size = self.blocks.pop(key, None)
        miss = size is None
        assert miss == bool(recorded_miss), ("Logical cache mismatch", key, miss, recorded_miss)
        self.counts["cache_miss" if miss else "cache_hit"] += 1
        self.counts["cached_block_calls"] += 1
        if miss:
            size = self.block_bytes(row, col)
            self.bytes += size
        self.blocks[key] = size
        while self.bytes > self.limit and self.blocks:
            _, old = self.blocks.popitem(last=False)
            self.bytes -= old
            self.counts["cache_eviction"] += 1
        return miss

    def verify_final(self, report):
        for name, count in self.counts.items():
            assert count == report["counts"][name], (name, count, report["counts"][name])
        assert self.bytes == report["final_cache_bytes"]
        digest = hashlib.sha256(repr(list(self.blocks)).encode()).hexdigest()
        assert digest == report["final_lru_sha256"]


def _calls(events):
    start = None
    for index in np.flatnonzero(events["row"] == -1):
        assert int(events[index]["col"]) == -1 and not events[index]["miss"]
        if start is not None:
            yield events[start:index]
        else:
            assert index == 0, "Trace must begin at a sampling call boundary"
        start = int(index)+1
    if start is not None:
        yield events[start:]
    else:
        assert not len(events), "Trace has no sampling call boundary"


def _distribution(histogram):
    count = sum(histogram.values())
    total = sum(length*number for length, number in histogram.items())
    ordered = sorted(histogram.items())
    quantiles = {}
    for name, fraction in (("p50", .50), ("p90", .90), ("p95", .95), ("p99", .99)):
        target = max(1, int(np.ceil(count*fraction)))
        cumulative = 0
        quantiles[name] = 0
        for length, number in ordered:
            cumulative += number
            if cumulative >= target:
                quantiles[name] = length
                break
    return dict(histogram={str(k): v for k, v in ordered}, run_count=count,
                block_count=total, mean=total/count if count else 0.,
                max=max(histogram, default=0), **quantiles)


def _observed_runs(events):
    histogram = Counter()
    call_count = 0
    for call in _calls(events):
        call_count += 1
        run, previous = 0, None
        for event in call:
            row, col, miss = int(event["row"]), int(event["col"]), bool(event["miss"])
            assert row >= 0 and col == 0
            adjacent = previous is not None and row == previous[0]+1 and col == previous[1]
            if not miss or not adjacent:
                if run:
                    histogram[run] += 1
                run = 0
            if miss:
                run += 1
            previous = (row, col) if miss else None
        if run:
            histogram[run] += 1
    return histogram, call_count


def _replay(events, grid, itemsize, batch, report):
    cache = _Cache(grid, itemsize)
    histogram = Counter()
    max_read_bytes = 0
    for call in _calls(events):
        index = 0
        while index < len(call):
            event = call[index]
            row, col = int(event["row"]), int(event["col"])
            key = (row, col)
            if key in cache.blocks:
                cache.touch(row, col, event["miss"])
                index += 1
                continue
            stop = index+1
            while stop < min(len(call), index+batch):
                following = call[stop]
                next_row, next_col = int(following["row"]), int(following["col"])
                if next_col != col or next_row != row+(stop-index) or (next_row, next_col) in cache.blocks:
                    break
                stop += 1
            histogram[stop-index] += 1
            read_bytes = 0
            for offset in range(index, stop):
                picked = call[offset]
                block_row, block_col = int(picked["row"]), int(picked["col"])
                read_bytes += cache.block_bytes(block_row, block_col)
                assert cache.touch(block_row, block_col, picked["miss"])
            max_read_bytes = max(max_read_bytes, read_bytes)
            index = stop
    cache.verify_final(report)
    return dict(read_count=sum(histogram.values()), max_read_bytes=max_read_bytes,
                reads_by_block_count={str(k): v for k, v in sorted(histogram.items())},
                logical_cache_counts=dict(cache.counts), final_cache_bytes=cache.bytes,
                exact_trace_and_final_lru=True)


def analyze_scene(summary_path, index, report, dtype, batches):
    grid = report["grid"]
    assert not grid["ram"], "RAM-backed sampling does not have a strip miss sequence"
    assert int(grid["block_shape"][1]) == int(grid["width"]), "Only full-width strip rasters are supported"
    trace_path = summary_path.with_name(f"{summary_path.stem}_scene{index}_miss_trace.bin")
    raw = trace_path.read_bytes()
    assert len(raw) % EVENT_DTYPE.itemsize == 0, "Truncated miss trace"
    assert hashlib.sha256(raw).hexdigest() == report["raster_io"]["miss_trace_sha256"]
    events = np.frombuffer(raw, dtype=EVENT_DTYPE)
    runs, call_count = _observed_runs(events)
    baseline = _replay(events, grid, dtype.itemsize, 1, report)
    assert baseline["read_count"] == report["raster_io"]["read_count"], "Input must use unmerged baseline reads"
    assert baseline["max_read_bytes"] == report["raster_io"]["max_read_bytes"], "dtype/mask byte budget does not match actual reads"
    assert call_count == report["counts"]["values_at_calls"]
    misses = baseline["read_count"]
    assert sum(k*v for k, v in runs.items()) == misses
    experiments = {}
    for batch in batches:
        assert batch >= 1
        theoretical = sum(((length+batch-1)//batch)*count for length, count in runs.items())
        conservative = _replay(events, grid, dtype.itemsize, batch, report)
        assert conservative["read_count"] >= theoretical
        experiments[str(batch)] = dict(
            theoretical_read_count=theoretical,
            theoretical_reduction_percent=100*(misses-theoretical)/misses if misses else 0.,
            conservative_reduction_percent=100*(misses-conservative["read_count"])/misses if misses else 0.,
            **conservative)
    return dict(scene=index, trace=str(trace_path), dtype=dtype.name, dtype_itemsize=dtype.itemsize,
                mask_bytes_per_pixel=1, grid=grid, baseline_read_count=misses,
                baseline_read_seconds=report["raster_io"]["read_seconds"],
                baseline_max_read_bytes=baseline["max_read_bytes"],
                values_at_calls=call_count, observed_miss_runs=_distribution(runs), batches=experiments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--dtype", default="uint8", help="Verified native raster dtype; never inferred from probabilities")
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    dtype = np.dtype(args.dtype)
    scenes = [analyze_scene(args.summary, index, report, dtype, args.batches)
              for index, report in enumerate(summary["probability"])]
    baseline_reads = sum(scene["baseline_read_count"] for scene in scenes)
    totals = {}
    for batch in args.batches:
        chosen = [scene["batches"][str(batch)] for scene in scenes]
        reads = sum(item["read_count"] for item in chosen)
        totals[str(batch)] = dict(theoretical_read_count=sum(item["theoretical_read_count"] for item in chosen),
                                 conservative_read_count=reads,
                                 conservative_reduction_percent=100*(baseline_reads-reads)/baseline_reads if baseline_reads else 0.,
                                 max_read_bytes=max((item["max_read_bytes"] for item in chosen), default=0))
    result = dict(summary=str(args.summary), dtype_argument=args.dtype, baseline_read_count=baseline_reads,
                  baseline_read_seconds=sum(scene["baseline_read_seconds"] for scene in scenes),
                  batches=totals, scenes=scenes)
    output = args.output or args.summary.with_name(f"{args.summary.stem}_miss_analysis.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "scenes"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
