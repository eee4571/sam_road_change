"""Wall-clock timings at GIS stage boundaries (no per-station logging)."""
from functools import wraps
from time import perf_counter


def timed_stage(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                seconds = perf_counter()-started
                from .batch_runtime import record_timing
                record_timing(name,seconds)
                print(f'[Fast timing] {name}={seconds:.6f}s', flush=True)
        return wrapped
    return decorate
