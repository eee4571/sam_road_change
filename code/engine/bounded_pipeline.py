"""One producer, one pending image: overlap inference with CPU postprocessing."""
from concurrent.futures import ThreadPoolExecutor
import time


class Prefetch:
    def __init__(self, items, loader, enabled=True):
        self.items, self.loader = iter(items), loader
        self.executor = ThreadPoolExecutor(max_workers=1) if enabled else None
        self.wait_seconds = 0.

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        print(f'[Fast batch timing] cpu_pipeline_wait={self.wait_seconds:.6f}s queue_capacity=1', flush=True)

    def __iter__(self):
        if self.executor is None:
            for item in self.items:
                yield self.loader(item)
            return
        item = next(self.items, None)
        if item is None:
            return
        future = self.executor.submit(self.loader,item)
        while future is not None:
            started = time.perf_counter()
            result = future.result()
            self.wait_seconds += time.perf_counter()-started
            item = next(self.items, None)
            future = self.executor.submit(self.loader,item) if item is not None else None
            yield result
