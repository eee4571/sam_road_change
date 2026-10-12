"""Process-wide admission control for RGB width work, before array allocation.

The 512 MiB algorithm budget comprises 128 MiB of resident feature cache,
32 MiB of GDAL block cache and 352 MiB of concurrent workspaces. These cover arrays pinned by a
worker after cache eviction. This is not a cap on Python/GDAL/process RSS.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, RLock, local
from weakref import WeakSet
import math

MIB = 1024 * 1024
CACHE_BYTES = 128 * MIB
GDAL_CACHE_BYTES = 32 * MIB


class RoadMemoryLimit(ValueError):
    """Pre-allocation safety refusal; a road can use the existing C fallback."""


@dataclass(frozen=True)
class MemoryPolicy:
    max_window_pixels: int = 512 * 1024
    max_window_side: int = 4096
    max_sections: int = 20000
    max_dense_points: int = 100000

    def check_window(self, width, height):
        # Python integers: never overflow a NumPy integer area multiplication.
        w, h = int(width), int(height)
        if min(w, h) <= 0 or max(w, h) > self.max_window_side or w*h > self.max_window_pixels:
            raise RoadMemoryLimit(f'RGB window exceeds safety limit: {w}x{h} pixels')

    def road_bytes(self, length, config):
        if not math.isfinite(length) or length <= 0:
            raise ValueError('Road length must be finite and positive')
        if not all(math.isfinite(v) and v > 0 for v in
                   (config.spacing, config.cross_step, config.search_radius)):
            from .width_reliability import InputContractError
            raise InputContractError('Invalid RGB sampling configuration')
        n = math.ceil(length/config.spacing) + 1
        dense = max(3, math.ceil(length) + 1)
        offsets = math.ceil(2*(config.search_radius+2)/config.cross_step) + 2
        if n > self.max_sections or dense > self.max_dense_points or offsets > 4096:
            raise RoadMemoryLimit(f'road_profile_sample_budget_exceeded: sections={n}, dense={dense}, cross={offsets}')
        # Includes full profile outputs + batched temporaries, DP candidate/back
        # arrays, Python records, solver copies and dense axis interpolation.
        return (self.window_bytes(self.max_window_pixels) + dense*256 +
                n*(offsets*256 + config.candidates**2*128 + 16384))

    @staticmethod
    def window_bytes(pixels):
        # Native filters, read/mask buffers, cached/pinned arrays and one 512²
        # pointwise block. Conservative allowance, verified by RSS stress tests.
        return 32*MIB + int(pixels)*192


class WorkspaceBudget:
    def __init__(self, capacity=352*MIB):
        self.capacity = capacity
        self.condition = Condition()
        self.thread = local()
        self.used = self.peak = self.active = self.peak_active = self.waits = 0

    @contextmanager
    def reserve(self, amount):
        amount = int(amount)
        held = getattr(self.thread, 'held', 0)
        if held:
            if amount > held:
                raise RuntimeError('RGB nested workspace exceeds its admitted reservation')
            yield
            return
        if amount > self.capacity:
            raise RoadMemoryLimit(f'road_workspace_budget_exceeded: requested={amount}, limit={self.capacity}')
        with self.condition:
            while self.used + amount > self.capacity:
                self.waits += 1
                self.condition.wait()
            self.used += amount
            self.active += 1
            self.peak = max(self.peak, self.used)
            self.peak_active = max(self.peak_active, self.active)
        self.thread.held = amount
        try:
            yield
        finally:
            self.thread.held = 0
            with self.condition:
                self.used -= amount
                self.active -= 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(workspace_limit_bytes=self.capacity, peak_workspace_bytes=self.peak,
                        active_workspace_bytes=self.used, peak_active_workers=self.peak_active,
                        admission_waits=self.waits, cache_limit_bytes=CACHE_BYTES,
                        gdal_cache_limit_bytes=GDAL_CACHE_BYTES,
                        algorithm_limit_bytes=self.capacity+CACHE_BYTES+GDAL_CACHE_BYTES)


WORKSPACES = WorkspaceBudget()
# All readers in this process share the resident limit. Evicted entries still
# pinned by an active worker are covered by that worker's workspace reservation.
CACHE_LOCK = RLock()
CACHES = WeakSet()
_gdal_users = 0
_gdal_previous = None


@contextmanager
def bounded_gdal_cache():
    """GDAL's default is a percentage of host RAM, outside NumPy accounting.

    Its block cache is process-global. Apply one shared cap for all RGB readers
    and restore the previous setting only when the last reader closes.
    """
    from rasterio.env import get_gdal_config, set_gdal_config
    global _gdal_users, _gdal_previous
    with CACHE_LOCK:
        if _gdal_users == 0:
            _gdal_previous = get_gdal_config('GDAL_CACHEMAX')
            set_gdal_config('GDAL_CACHEMAX', min(_gdal_previous, GDAL_CACHE_BYTES))
        _gdal_users += 1
    try:
        yield
    finally:
        with CACHE_LOCK:
            _gdal_users -= 1
            if _gdal_users == 0:
                set_gdal_config('GDAL_CACHEMAX', _gdal_previous)


def admit_cache(cache, size):
    """Called with CACHE_LOCK held; preserve each cache's own LRU contract."""
    while cache.items and cache.bytes + size > cache.max_bytes:
        cache.evict()
    while sum(c.bytes for c in CACHES) + size > CACHE_BYTES:
        victim = next((c for c in CACHES if c.items), None)
        if victim is None:
            return False
        victim.evict()
    return size <= min(cache.max_bytes, CACHE_BYTES)
