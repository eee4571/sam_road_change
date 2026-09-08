from __future__ import annotations

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Iterable
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


def resolve_worker_count(
    requested_workers: int,
    job_count: int,
    *,
    automatic_limit: int = 2,
) -> int:
    """Resolve conservative batch parallelism; zero means automatic."""
    requested = int(requested_workers)
    jobs = max(0, int(job_count))
    if requested < 0:
        raise ValueError("workers must be zero (automatic) or a positive integer")
    if jobs <= 1:
        return jobs
    if requested == 0:
        available = max(1, int(os.cpu_count() or 1))
        return min(jobs, max(1, int(automatic_limit)), available)
    return min(jobs, requested)


def spawn_map(function: Callable[[T], R], payloads: Iterable[T], workers: int) -> list[R]:
    """Map top-level workers through the Windows-compatible spawn context."""
    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        return list(executor.map(function, payloads))
