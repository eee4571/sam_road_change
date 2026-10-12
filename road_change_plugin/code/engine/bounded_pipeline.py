"""One producer, one pending image: overlap inference with CPU postprocessing."""
from concurrent.futures import ThreadPoolExecutor
import time


class Prefetch:
    def __init__(self, items, loader, enabled=True):
        self.items, self.loader = iter(items), loader
        self.executor = ThreadPoolExecutor(max_workers=1) if enabled else None
        self.wait_seconds = 0.

    def __enter__(self):
        self.leases=[]
        return self

    def __exit__(self, *exc):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        for lease in self.leases:lease.close()
        print(f'[Fast batch timing] cpu_pipeline_wait={self.wait_seconds:.6f}s queue_capacity=1', flush=True)

    def __iter__(self):
        from .resource_policy import try_speculation
        if self.executor is None:
            for item in self.items:
                yield self.loader(item)
            return
        sentinel=object();item=next(self.items,sentinel);pending=None
        while item is not sentinel:
            if pending is None:
                result=self.loader(item)
            else:
                future,lease=pending
                started=time.perf_counter()
                try:result=future.result()
                finally:
                    self.wait_seconds+=time.perf_counter()-started
                    lease.close();self.leases.remove(lease)
            following=next(self.items,sentinel);pending=None
            lease=try_speculation() if following is not sentinel else None
            if lease is not None:
                self.leases.append(lease)
                # Reservation lasts until consumption, including queued arrays.
                pending=(self.executor.submit(self.loader,following),lease)
            yield result
            del result
            item=following
