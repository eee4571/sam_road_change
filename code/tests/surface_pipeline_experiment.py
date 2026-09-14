"""Bounded surface-depth experiments; no alternate geometry implementation."""
from collections import deque
import time


def install(module, workers, lookahead):
    """Replace only scheduling for an explicit cached-input benchmark.

    A depth of one reproduces the production planning/submission order.  The
    existing planner still skips empty selected arrays and owns all prefilters.
    The caller installs surface_timing_probe afterwards for global interval
    unions: the legacy overlap counter below only describes the preceding axis.
    """
    if workers not in (2, 3, 4) or lookahead not in (1, 2):
        raise ValueError('Expected 2/3/4 workers and 1/2-axis lookahead')
    active_streams = set()

    def union_seconds(intervals):
        total, end = 0., float('-inf')
        for a, b in sorted(intervals):
            total += max(0., b-max(a, end))
            end = max(end, b)
        return total

    def pipeline(plans, source, target, tolerance, pool, timing):
        def iterate():
            began = time.perf_counter()
            iterator = iter(plans)
            active_jobs = set()
            pending = deque()

            def submit(plan):
                jobs = []
                for scene in (source, target):
                    job = pool.submit(module._prepare_axis_surface, scene, plan[1], tolerance)
                    active_jobs.add(job)
                    jobs.append(job)
                return tuple(jobs)

            try:
                current = next(iterator, None)
                if current is None:
                    return
                jobs = submit(current)
                previous_processing = None
                exhausted = False
                while current is not None:
                    following = []
                    # Planning can overlap the current worker jobs.  Surface
                    # tasks are submitted only after the current result is ready.
                    while not exhausted and len(pending)+len(following) < lookahead:
                        plan = next(iterator, None)
                        if plan is None:
                            exhausted = True
                        else:
                            following.append(plan)
                    wait_started = time.perf_counter()
                    results = tuple(job.result() for job in jobs)
                    timing['axis_surface_wait'] += time.perf_counter()-wait_started
                    active_jobs.difference_update(jobs)
                    busy = [(r[4], r[5]) for r in results]
                    timing['surface_pipeline_worker_busy'] += sum(b-a for a, b in busy)
                    timing['surface_pipeline_active_wall'] += union_seconds(busy)
                    if previous_processing is not None:
                        start, end = previous_processing
                        overlap = [(max(a, start), min(b, end)) for a, b in busy
                                   if min(b, end) > max(a, start)]
                        timing['surface_pipeline_overlap'] += union_seconds(overlap)
                    timing['axis_surface_clip_worker_sum'] += sum(r[2] for r in results)
                    timing['axis_surface_buffer_worker_sum'] += sum(r[3] for r in results)
                    for plan in following:
                        pending.append((plan, submit(plan)))
                    following.clear()
                    processing_started = time.perf_counter()
                    yield current, results
                    previous_processing = (processing_started, time.perf_counter())
                    # Drop consumed geometry before waiting for the next axis.
                    results = None
                    if pending:
                        current, jobs = pending.popleft()
                    else:
                        current, jobs = None, ()
            finally:
                for job in active_jobs:
                    job.cancel()
                pending.clear()
                active_jobs.clear()
                close = getattr(iterator, 'close', None)
                if close is not None:
                    close()
                timing['surface_pipeline_wall'] += time.perf_counter()-began
                active_streams.discard(stream)

        stream = iterate()
        active_streams.add(stream)
        return stream

    def analyze_scenes(before, after, *, tolerance=3., absolute=2., relative=.2,
                       minimum_length=24., minimum_area=4., presence_audit=None,
                       candidate_driven=True):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='auto-axis-surface') as pool:
            try:
                return module._analyze_scenes(
                    before, after, tolerance=tolerance, absolute=absolute,
                    relative=relative, minimum_length=minimum_length,
                    minimum_area=minimum_area, presence_audit=presence_audit,
                    candidate_driven=candidate_driven, surface_pool=pool)
            finally:
                # _analyze_scenes can fail inside its for-loop.  Explicitly
                # close suspended generators before the pool joins its workers.
                for stream in tuple(active_streams):
                    stream.close()
                active_streams.clear()

    module.analyze_scenes = analyze_scenes
    module._axis_surface_pipeline = pipeline
