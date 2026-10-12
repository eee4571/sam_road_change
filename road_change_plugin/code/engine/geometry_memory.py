"""Pre-allocation contracts for optional dense geometric reconstruction."""
import math
import numpy as np

MAX_STATIONS = 100_000


class GeometrySamplingLimit(ValueError):
    """Optional dense reconstruction is too large; retain the original axis."""


def stations(length, count):
    if not math.isfinite(length) or length < 0:
        raise ValueError('Non-finite or negative road length')
    return bounded_linspace(0.,length,count)


def bounded_linspace(start,stop,num,**kwargs):
    if num>MAX_STATIONS:
        raise GeometrySamplingLimit(f'geometry_station_budget_exceeded: {num} > {MAX_STATIONS}')
    return np.linspace(start,stop,num,**kwargs)


def within_budget(length, spacing=1.):
    return math.isfinite(length) and math.ceil(length/spacing)+1 <= MAX_STATIONS


def json_geometry_digest(geometries):
    """Identical to sha256(json.dumps(frame.geometry.to_wkb(hex=True).tolist()))."""
    import hashlib,json
    digest=hashlib.sha256();digest.update(b'[');first=True
    for geometry in geometries:
        if not first:digest.update(b', ')
        digest.update(json.dumps(None if geometry is None else geometry.wkb_hex).encode())
        first=False
    digest.update(b']')
    return digest.hexdigest()
