"""Comparable surface timelines for explicit concurrency experiments only."""
import threading
import time


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap(a, b):
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0., min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def install(module):
    original_prepare = module._prepare_axis_surface
    original_pipeline = module._axis_surface_pipeline
    workers, processing = [], []

    def prepare(*args, **kwargs):
        cpu = time.thread_time()
        result = original_prepare(*args, **kwargs)
        workers.append((result[4], result[5], time.thread_time()-cpu, threading.get_ident()))
        return result

    def pipeline(*args, **kwargs):
        stream = original_pipeline(*args, **kwargs)
        try:
            for item in stream:
                started = time.perf_counter()
                try:
                    yield item
                finally:
                    processing.append((started, time.perf_counter()))
        finally:
            stream.close()

    module._prepare_axis_surface = prepare
    module._axis_surface_pipeline = pipeline

    def report():
        active = merged([(a, b) for a, b, _, _ in workers])
        handled = merged(processing)
        return dict(worker_calls=len(workers), worker_threads=len({t for _, _, _, t in workers}),
                    worker_cpu_seconds=sum(c for _, _, c, _ in workers),
                    worker_busy_seconds=sum(b-a for a, b, _, _ in workers),
                    active_wall_seconds=sum(b-a for a, b in active),
                    processing_wall_seconds=sum(b-a for a, b in handled),
                    overlap_seconds=overlap(active, handled))
    return report
