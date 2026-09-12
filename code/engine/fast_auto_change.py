from __future__ import annotations

"""Fast no-truth change detection on final road products.

The GT-assisted baseline does not import this module. Raster reads are windowed;
matching and run assembly use regional final axes so tile/feature seams do not
become change boundaries. No ground-truth input is accepted here.
"""

import json
from .fast_timing import timed_stage
import sys
import time
from collections import Counter, OrderedDict
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from shapely import intersection, line_merge, make_valid, prepare, union_all, line_interpolate_point, get_coordinates
from shapely.geometry import LineString, Point, box
from shapely.ops import substring
from shapely.strtree import STRtree

WIDTH_ROOT = Path(__file__).resolve().parent / "width"
if str(WIDTH_ROOT) not in sys.path:
    sys.path.insert(0, str(WIDTH_ROOT))
from road_existence_evidence import RoadProbabilityRaster
from paired_width_profile import (
    PairedWidthConfig, PairedWidthProfile, PairedWidthSample,
    _measure_period_width, candidate_change_runs, evaluate_change_run,
)


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    return [part for child in getattr(geometry, "geoms", ()) for part in _parts(child)]


class WindowedProbability(RoadProbabilityRaster):
    """Metric sampling of a georeferenced raster, including geographic rasters."""

    _STRIP_READ_BATCH = 32
    _STRIP_READ_MAX_BYTES = 1024*1024

    def __init__(self, path, metric_crs, *, ram_limit_bytes=128*1024*1024, cache_limit_bytes=64*1024*1024):
        self.dataset = rasterio.open(path)
        self.crs = CRS.from_user_input(self.dataset.crs)
        self.metric_crs = CRS.from_user_input(metric_crs)
        self.to_raster = Transformer.from_crs(metric_crs, self.crs, always_xy=True)
        self.inverse = ~self.dataset.transform
        to_metric = Transformer.from_crs(self.crs, metric_crs, always_xy=True)
        x, y = self.dataset.xy(self.dataset.height//2, self.dataset.width//2)
        x1, y1 = self.dataset.xy(self.dataset.height//2, self.dataset.width//2+1)
        a, b = to_metric.transform(x, y), to_metric.transform(x1, y1)
        self.pixel_size = float(np.hypot(b[0]-a[0], b[1]-a[1]))
        scale = max(1, int(np.ceil(max(self.dataset.shape) / 1000)))
        scene = self.dataset.read(1, out_shape=(max(1, self.dataset.height // scale),
                                               max(1, self.dataset.width // scale)), masked=True)
        self.divisor = 255.0 if float(scene.max()) > 1.0 else 1.0
        values = np.asarray(scene.compressed(), dtype=np.float32) / self.divisor
        values = values[np.isfinite(values)]
        # Float64 avoids a million-element dtype conversion on each scalar search.
        self.scene_values = np.sort(values.astype(np.float64))
        self.scene_percentiles = {f"p{q}": float(np.percentile(values, q)) if len(values) else None
                                  for q in (50, 90, 95, 99)}
        self._ram = None
        self._blocks = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = cache_limit_bytes
        self._block_shape = self.dataset.block_shapes[0]
        required = self.dataset.width*self.dataset.height*(np.dtype(self.dataset.dtypes[0]).itemsize+1)
        if required <= ram_limit_bytes:
            self._ram = self.dataset.read(1, masked=True)

    def close(self):
        self.dataset.close()
        self._ram = None
        self._blocks.clear()
        self._cache_bytes = 0

    def _values_at(self, x, y):
        x, y = self.to_raster.transform(np.asarray(x), np.asarray(y))
        cols, rows = self.inverse * (np.asarray(x), np.asarray(y))
        cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
        result = np.full(cols.shape, np.nan, dtype=float)
        inside = (cols >= 0) & (rows >= 0) & (cols < self.dataset.width) & (rows < self.dataset.height)
        if not inside.any():
            return result
        rr, cc = rows[inside], cols[inside]
        if self._ram is not None:
            picked = self._gather_pixels(self._ram, rr, cc)
        else:
            height, width = self._block_shape
            block_rows, block_cols = rr//height, cc//width
            if (block_rows == block_rows[0]).all() and (block_cols == block_cols[0]).all():
                br, bc = int(block_rows[0]), int(block_cols[0])
                values = self._cached_block(br, bc)
                picked = self._gather_pixels(values, rr-br*height, cc-bc*width)
            else:
                # Integer IDs have the same lexicographic (row, col) order as
                # unique(axis=0). Stable sorting also preserves the input pixel
                # order within each block, without scanning all pixels per block.
                column_count = (self.dataset.width + width - 1)//width
                block_ids = block_rows.astype(np.int64)*column_count + block_cols
                order = np.argsort(block_ids, kind="stable")
                ordered_ids = block_ids[order]
                starts = np.r_[0, np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1])+1]
                stops = np.r_[starts[1:], len(order)]
                unique_ids = ordered_ids[starts]
                unique = np.column_stack((unique_ids//column_count, unique_ids % column_count))
                picked = np.full(rr.shape, np.nan)
                blocks = self._cached_blocks(unique)
                try:
                    for (br, bc), start, stop, values in zip(unique, starts, stops, blocks):
                        take = order[start:stop]
                        picked[take] = self._gather_pixels(values, rr[take]-br*height, cc[take]-bc*width)
                finally:
                    blocks.close()
        result[inside] = picked / self.divisor
        return result

    @staticmethod
    def _gather_pixels(values, rows, cols):
        # Gather the unchanged native pixels before converting to float, just as
        # MaskedArray indexing/astype/filled does, without creating masked-array
        # views and copying their metadata for every block. Keep cached blocks
        # in their original dtype and preserve their exact mask and byte budget.
        picked = values.data[rows, cols].astype(float)
        mask = values.mask
        if np.ndim(mask):
            picked[mask[rows, cols]] = np.nan
        elif mask:
            picked[...] = np.nan
        return picked

    def _cached_blocks(self, unique):
        """Read only consecutive requested misses; consume the original LRU order."""
        height, width = self._block_shape
        block_bytes = height*width*(np.dtype(self.dataset.dtypes[0]).itemsize+1)
        batch_limit = min(self._STRIP_READ_BATCH, max(1, self._STRIP_READ_MAX_BYTES//block_bytes))
        index, count = 0, len(unique)
        batch = None
        try:
            while index < count:
                br, bc = map(int, unique[index])
                stop = index+1
                if width == self.dataset.width and bc == 0 and batch_limit > 1 and (br, bc) not in self._blocks:
                    # Stop at a currently cached block, even if preceding inserts
                    # might later evict it. Never speculate across a hit or gap.
                    while stop < min(count, index+batch_limit):
                        nr, nc = map(int, unique[stop])
                        if nr != br+stop-index or nc != bc or (nr, nc) in self._blocks:
                            break
                        stop += 1
                if stop > index+1:
                    batch = self.dataset.read(1, window=rasterio.windows.Window(
                        0, br*height, width, min((stop-index)*height, self.dataset.height-br*height)), masked=True)
                    for offset in range(stop-index):
                        # Own each block's data/mask: an evicted slice must not
                        # keep the whole read buffer alive in another cache entry.
                        block = batch[offset*height:(offset+1)*height].copy()
                        if index+offset+1 == stop:
                            batch = None
                        yield self._cached_block(br+offset, bc, loaded=block)
                        block = None
                else:
                    yield self._cached_block(br, bc)
                index = stop
        finally:
            batch = None

    def _cached_block(self, br, bc, *, loaded=None):
        key = (br, bc)
        values = self._blocks.pop(key, None)
        if values is None:
            if loaded is None:
                height, width = self._block_shape
                values = self.dataset.read(1, window=rasterio.windows.Window(
                    bc*width, br*height, min(width, self.dataset.width-bc*width),
                    min(height, self.dataset.height-br*height)), masked=True)
            else:
                values = loaded
            self._cache_bytes += values.data.nbytes + np.ma.getmaskarray(values).nbytes
        self._blocks[key] = values
        while self._cache_bytes > self._cache_limit and self._blocks:
            _, old = self._blocks.popitem(last=False)
            self._cache_bytes -= old.data.nbytes + np.ma.getmaskarray(old).nbytes
        return values

    def sample_cross_section(self, center, normal, geometry_crs, *, search_radius):
        distances = np.arange(-search_radius, search_radius + 0.25, 0.5)
        values = self._values_at(center.x + normal[0]*distances, center.y + normal[1]*distances)
        return {"distance": distances, "probability": values,
                "valid_mask": np.isfinite(values), "sample_step": 0.5}

    def _axis_coordinates(self, axis, *, road_width, position_tolerance):
        positions = np.linspace(0, axis.length, max(3, int(np.ceil(axis.length/2))+1))
        points = get_coordinates(line_interpolate_point(axis, positions))
        normals = _normals(axis, positions)
        inner = max(road_width * 0.70, position_tolerance + 2.0)
        centers = (points[:, None, :] + np.asarray([-1., 0., 1.])[None, :, None]*normals[:, None, :]).reshape(-1, 2)
        backgrounds = (points[:, None, :] + np.asarray([-inner-3, -inner, inner, inner+3])[None, :, None]*normals[:, None, :]).reshape(-1, 2)
        return np.concatenate((centers, backgrounds)), len(centers)

    @staticmethod
    def _axis_summary(values, center_count, sampler):
        center, background = values[:center_count], values[center_count:]
        center, background = center[np.isfinite(center)], background[np.isfinite(background)]
        mean = float(np.mean(center)) if len(center) else None
        bg = float(np.median(background)) if len(background) else None
        return {"center_probability_mean": mean,
                "center_probability_q25": float(np.quantile(center, .25)) if len(center) else None,
                "local_background_probability": bg,
                "local_probability_contrast": mean-bg if mean is not None and bg is not None else None,
                "scene_percentile_rank": sampler.percentile_rank(mean),
                "background_percentile_rank": sampler.percentile_rank(bg),
                "probability_valid_ratio": float(np.isfinite(values).mean())}

    def sample_axes(self, axes, widths, tolerance, *, coordinates=None):
        if coordinates is None:
            coordinates = [self._axis_coordinates(axis, road_width=width, position_tolerance=tolerance)
                           for axis, width in zip(axes, widths)]
        if not coordinates:
            return []
        result = []
        for start in range(0, len(coordinates), 256):
            block = coordinates[start:start+256]
            coords = np.concatenate([item[0] for item in block])
            values = self._values_at(coords[:, 0], coords[:, 1])
            offset = 0
            for points, count in block:
                result.append(self._axis_summary(values[offset:offset+len(points)], count, self))
                offset += len(points)
        return result

    def sample_axis(self, axis, axis_crs, *, road_width, position_tolerance):
        coords, count = self._axis_coordinates(axis, road_width=road_width, position_tolerance=position_tolerance)
        return self._axis_summary(self._values_at(coords[:, 0], coords[:, 1]), count, self)



def _normal(line, station):
    a, b = line.interpolate(max(0, station-3)), line.interpolate(min(line.length, station+3))
    direction = np.array([b.x-a.x, b.y-a.y])
    norm = np.linalg.norm(direction)
    return np.array([-direction[1], direction[0]]) / max(norm, 1e-9)


def _normals(line, stations):
    a = get_coordinates(line_interpolate_point(line, np.maximum(0, stations-3)))
    b = get_coordinates(line_interpolate_point(line, np.minimum(line.length, stations+3)))
    d = b-a
    return np.column_stack((-d[:, 1], d[:, 0])) / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-9)


class RoadScene:
    def __init__(self, centerlines, surfaces, widths, valid, probability, crs):
        self.lines = _parts(line_merge(union_all(centerlines.geometry.values)))
        self.tree = STRtree(self.lines)
        self.surfaces = surfaces
        self.surface_tree = STRtree(surfaces.geometry.values)
        self.widths = widths
        self.has_width_values = 'width_m' in widths
        self.width_geometries = widths.geometry.values
        self.width_values = widths["width_m"].to_numpy() if "width_m" in widths else np.full(len(widths), 6.)
        self.width_tree = STRtree(self.width_geometries)
        self.width = lru_cache(maxsize=4096)(self.width)
        self.surface = lru_cache(maxsize=32)(self.surface)
        self.valid = make_valid(union_all(valid.geometry.values))
        prepare(self.valid)
        self.probability = probability
        self.crs = crs
        endpoints = [Point(line.coords[end]) for line in self.lines for end in (0, -1)]
        endpoint_tree = STRtree(endpoints)
        self.junction = union_all([point.buffer(12) for point in endpoints
                                   if len(endpoint_tree.query(point, predicate="dwithin", distance=1.5)) >= 3])
        prepare(self.junction)

    def surface(self, axis, radius=65):
        ids = self.surface_tree.query(axis, predicate="dwithin", distance=radius)
        left, bottom, right, top = axis.bounds
        window = box(left-radius, bottom-radius, right+radius, top+radius)
        # A single connected surface can span the entire scene. Clip before
        # union/buffering so local sections never process that entire polygon.
        return union_all(intersection(self.surfaces.geometry.values[ids], window))

    def width(self, point):
        ids = self.width_tree.query(point, predicate="dwithin", distance=3)
        if not len(ids):
            return 6.0
        index = min(ids, key=lambda i: self.width_geometries[i].distance(point))
        value = self.width_values[index]
        return float(value) if pd.notna(value) and value > 0 else 6.0

    def widths_at(self, points):
        from shapely import distance
        result = np.full(len(points), 6.)
        for start in range(0, len(points), 512):
            block = points[start:start+512]
            pairs = self.width_tree.query(block, predicate='dwithin', distance=3)
            if not pairs.size:
                continue
            distances = distance(block[pairs[0]], self.width_geometries[pairs[1]])
            # Stable ordering retains the scalar STRtree tie choice.
            order = np.argsort(distances, kind='stable')
            _, first = np.unique(pairs[0, order], return_index=True)
            chosen = order[first]
            values = np.asarray(self.width_values[pairs[1, chosen]], dtype=float)
            result[start+pairs[0, chosen]] = np.where(np.isfinite(values) & (values > 0), values, 6.)
        return result

    def match(self, axis, station, tolerance, source_width, *, point=None, normal=None):
        point = axis.interpolate(station) if point is None else point
        local = substring(axis, max(0, station-6), min(axis.length, station+6))
        normal = _normal(axis, station) if normal is None else normal
        ranked = []
        for target_id in self.tree.query(point, predicate="dwithin", distance=tolerance+1e-6):
            target = self.lines[int(target_id)]
            target_station = target.project(point)
            target_point = target.interpolate(target_station)
            distance = point.distance(target_point)
            cosine = abs(float(np.dot(normal, _normal(target, target_station))))
            target_local = substring(target, max(0, target_station-6), min(target.length, target_station+6))
            overlap = local.intersection(target_local.buffer(tolerance+0.1)).length / max(local.length, 1e-9)
            if cosine < .90 or overlap < .50:
                continue
            target_width = self.width(target_point)
            corridor_overlap = local.buffer(source_width/2).intersection(target_local.buffer(target_width/2)).area
            corridor_overlap /= max(min(local.length*source_width, target_local.length*target_width), 1e-9)
            compatibility = min(source_width, target_width) / max(source_width, target_width)
            # Distance dominates: width is supporting evidence, never a veto of real widening.
            score = distance + 2*(1-cosine) + (1-overlap) + .25*(1-corridor_overlap) + .15*(1-compatibility)
            ranked.append((score, int(target_id), target_station, distance, cosine, overlap, corridor_overlap))
        ranked.sort()
        if not ranked:
            return None
        best = ranked[0]
        ambiguous = False
        if len(ranked) > 1 and ranked[1][0]-best[0] < .5:
            other = self.lines[ranked[1][1]].interpolate(ranked[1][2])
            # Two features meeting at one node are segmentation, distinct nearby axes are ambiguous tracks.
            ambiguous = other.distance(self.lines[best[1]].interpolate(best[2])) > 1.0
        return {"target": best[1], "station": best[2], "distance": best[3],
                "direction": best[4], "coverage": best[5], "corridor": best[6],
                "reliable": not ambiguous}

    def evidence(self, axis, geometry_present, width, tolerance, surface_support=None, probability=None, *, geometry_evidence=None):
        if geometry_evidence is None:
            footprint = axis.buffer(max(width/2, tolerance)+1, cap_style="flat")
            valid = self.valid.covers(footprint)
            if surface_support is None:
                surface_support = self.surface(axis, max(width, tolerance)+2).buffer(tolerance)
            coverage = axis.intersection(surface_support).length / max(axis.length, 1e-9)
        else:
            valid, coverage = geometry_evidence
        if probability is None:
            probability = self.probability.sample_axis(axis, self.crs, road_width=width, position_tolerance=tolerance)
        rank = probability["scene_percentile_rank"]
        bg_rank = probability["background_percentile_rank"]
        contrast = probability["local_probability_contrast"]
        supported = rank is not None and bg_rank is not None and rank >= .85 and rank-bg_rank >= .1 and (contrast or 0) > 0
        negative = rank is not None and rank <= .70 and (bg_rank is None or rank-bg_rank <= .10)
        valid = valid and probability.get("probability_valid_ratio", 1.) >= .99
        if not valid:
            state, reason = "uncertain", "invalid_or_boundary"
        elif geometry_present:
            state, reason = "present", "geometry"
        elif coverage >= .55:
            state, reason = "present", "surface_without_centerline"
        elif supported:
            state, reason = "present", "probability_without_centerline"
        elif coverage <= .1 and negative:
            state, reason = "absent", "all_negative"
        else:
            state, reason = "uncertain", "conflicting_or_weak_evidence"
        return {"state": state, "reason": reason, "geometry": bool(geometry_present),
                "surface": float(coverage), "probability": bool(supported), "valid": bool(valid),
                **probability}


def _axis_surface_coverage(axis, support, starts, ends, cells, cell_lengths):
    """Query axis/support intervals; preserve GEOS rounding at cut boundaries.

    Interior cells need no overlay. Boundary cells use the original overlay:
    subtracting projected chainages is mathematically equivalent, but need not
    be bit-identical to GEOS length (especially in large projected coordinates).
    Non-simple/closed axes cannot safely use unique endpoint chainages.
    """
    from shapely import intersection, length, points, line_locate_point

    def exact(indices):
        return length(intersection(cells[indices], support))/np.maximum(cell_lengths[indices], 1e-9)

    if not axis.is_simple or axis.is_closed or axis.length <= 0:
        return exact(slice(None))
    clipped = axis.intersection(support)
    parts, pending = [], [clipped]
    while pending:
        part = pending.pop()
        if part.is_empty:
            continue
        if part.geom_type == 'LineString':
            parts.append(part)
        elif hasattr(part, 'geoms'):
            pending.extend(part.geoms)
    if not parts:
        return np.zeros(len(cells))
    endpoints = np.asarray([p.coords[i][:2] for p in parts for i in (0, -1)])
    intervals = np.sort(line_locate_point(axis, points(endpoints)).reshape(-1, 2), axis=1)
    intervals = intervals[np.argsort(intervals[:, 0], kind='stable')]
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    intervals = np.asarray(merged)
    # Search the last interval starting before each cell; no station x interval
    # matrix, and no per-station Shapely objects or overlays for interior cells.
    indices = np.searchsorted(intervals[:, 0], starts, side='right')-1
    safe = np.maximum(indices, 0)
    full = (indices >= 0) & (intervals[safe, 1] >= ends)
    result = np.where(full, cell_lengths/np.maximum(cell_lengths, 1e-9), 0.)
    # Only cells touching a coverage boundary can have fractional coverage.
    # Include a roundoff guard based on the input coordinate magnitude; this
    # affects execution only, never a coverage/detection threshold.
    scale = max(1., axis.length, float(np.max(np.abs(axis.bounds))))
    guard = 64*np.finfo(float).eps*scale
    boundaries = np.sort(intervals.ravel())
    left = np.searchsorted(boundaries, starts-guard, side='left')
    right = np.searchsorted(boundaries, ends+guard, side='right')
    boundary_cells = np.flatnonzero(right > left)
    result[boundary_cells] = exact(boundary_cells)
    return result


def analyze_scenes(before, after, *, tolerance=3., absolute=2., relative=.2, minimum_length=24., minimum_area=4.,
                   presence_audit=None, candidate_driven=True):
    """Analyze in original order, with bounded parallel axis-surface preparation."""
    from concurrent.futures import ThreadPoolExecutor
    # GEOS overlay/buffer releases the GIL. These two jobs read independent
    # scenes, never touch raster contexts, and finish before station decisions.
    # Scope the pool to this call so failures also join all work and release it.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='auto-axis-surface') as surface_pool:
        return _analyze_scenes(before, after, tolerance=tolerance, absolute=absolute, relative=relative,
                               minimum_length=minimum_length, minimum_area=minimum_area,
                               presence_audit=presence_audit, candidate_driven=candidate_driven,
                               surface_pool=surface_pool)


def _prepare_axis_surface(scene, axis, tolerance):
    started = time.perf_counter()
    began = started
    surface = scene.surface(axis)
    clipped_seconds = time.perf_counter()-started
    started = time.perf_counter()
    support = surface.buffer(tolerance)
    finished = time.perf_counter()
    return surface, support, clipped_seconds, finished-started, began, finished


def _planned_axes(side, source, target, coverage, tolerance, width_config, candidate_driven, counts, timing):
    """Original axis prefilters, evaluated in order with one valid-axis lookahead."""
    from .auto_station_plan import presence_mask, width_mask
    change_type = 'removed' if side == 'before' else 'added'
    for line_id, axis in enumerate(source.lines):
        count = max(1, int(np.ceil(axis.length/4)))
        spacing = axis.length/count
        started = time.perf_counter()
        intervals = coverage.uncovered(axis)
        presence = presence_mask(count, spacing, intervals)
        timing['presence_prefilter'] += time.perf_counter()-started
        stations = (np.arange(count)+.5)*spacing
        points = line_interpolate_point(axis, stations)
        widths = source.widths_at(points)
        started = time.perf_counter()
        width_needed = (width_mask(axis, source, target, stations, widths, tolerance, width_config)
                        if side == 'before' and candidate_driven else np.full(count, side == 'before'))
        timing['width_prefilter'] += time.perf_counter()-started
        selected = np.flatnonzero(presence | width_needed) if candidate_driven else np.arange(count)
        counts['total_station_count'] += count
        counts['candidate_station_count'] += len(selected)
        counts['width_prefilter_skipped_station_count'] += int((~width_needed).sum()) if side == 'before' else 0
        if not len(selected):
            counts.update({f'{change_type}_source_axes': 1, f'{change_type}_uncovered_intervals': 0})
            continue
        yield line_id, axis, count, spacing, intervals, stations, points, widths, width_needed, selected


def _axis_surface_pipeline(plans, source, target, tolerance, pool, timing):
    """At most one next valid axis in flight; consume results in axis order."""
    def submit(plan):
        return tuple(pool.submit(_prepare_axis_surface, scene, plan[1], tolerance)
                     for scene in (source, target))

    def union_seconds(intervals):
        total, end = 0., float('-inf')
        for a, b in sorted(intervals):
            total += max(0., b-max(a, end))
            end = max(end, b)
        return total

    began = time.perf_counter()
    plans = iter(plans)
    current = next(plans, None)
    if current is None:
        return
    jobs = submit(current)
    previous_processing = None
    try:
        while current is not None:
            # Plan only the next effective axis while current surface work runs.
            following = next(plans, None)
            wait_started = time.perf_counter()
            results = tuple(job.result() for job in jobs)
            timing['axis_surface_wait'] += time.perf_counter()-wait_started
            busy = [(r[4], r[5]) for r in results]
            timing['surface_pipeline_worker_busy'] += sum(b-a for a, b in busy)
            timing['surface_pipeline_active_wall'] += union_seconds(busy)
            if previous_processing is not None:
                start, end = previous_processing
                overlap = [(max(a, start), min(b, end)) for a, b in busy if min(b, end)>max(a, start)]
                timing['surface_pipeline_overlap'] += union_seconds(overlap)
            timing['axis_surface_clip_worker_sum'] += sum(r[2] for r in results)
            timing['axis_surface_buffer_worker_sum'] += sum(r[3] for r in results)
            # Submit before handing the current axis back to its station loop.
            jobs = submit(following) if following is not None else ()
            processing_started = time.perf_counter()
            yield current, results
            previous_processing = (processing_started, time.perf_counter())
            current = following
    finally:
        for job in jobs:
            job.cancel()
        timing['surface_pipeline_wall'] += time.perf_counter()-began


def _analyze_scenes(before, after, *, tolerance, absolute, relative, minimum_length, minimum_area,
                    presence_audit, candidate_driven, surface_pool):
    """Plan candidate cells before sampling; retain exact paired-width decisions.

    candidate_driven=False is a full-station verification path for regression
    tests, not a separate detection algorithm or a public pipeline option.
    """
    from .auto_presence_candidates import LongitudinalCoverage, presence_seeds
    from .auto_station_plan import BatchSections
    from shapely import buffer, covers, intersects, length
    records, audit, width_audit = [], [], []
    counts = Counter()
    sampling_seconds = matching_seconds = 0.
    timing = Counter()
    width_config = PairedWidthConfig(sample_spacing=4, absolute_change=absolute,
                                    relative_change=relative, minimum_continuous_length=minimum_length,
                                    maximum_gap_samples=1, maximum_gap_length=8.)
    for side, source, target, change_type in (("before", before, after, "removed"),
                                               ("after", after, before, "added")):
        coverage = LongitudinalCoverage(target.lines, tolerance)
        plans = _planned_axes(side, source, target, coverage, tolerance, width_config, candidate_driven, counts, timing)
        pipeline = _axis_surface_pipeline(plans, source, target, tolerance, surface_pool, timing)
        for plan, surface_results in pipeline:
            line_id, axis, count, spacing, intervals, stations, points, widths, width_needed, selected = plan
            samples, station_rows = [], []
            sampling_started = time.perf_counter()
            source_surface, source_support = surface_results[0][:2]
            target_surface, target_support = surface_results[1][:2]
            width_surfaces = None
            cells_started = time.perf_counter()
            normals = _normals(axis, stations)
            cells = {index: substring(axis, index*spacing, (index+1)*spacing) for index in selected}
            timing['axis_cell_construction'] += time.perf_counter()-cells_started
            geometry_started = time.perf_counter()
            cell_array = np.asarray(list(cells.values()),dtype=object)
            cell_lengths = length(cell_array)
            cell_starts = selected*spacing
            cell_ends = np.minimum((selected+1)*spacing, axis.length)
            footprints = buffer(cell_array,np.maximum(widths[selected]/2,tolerance)+1,
                                cap_style="flat",quad_segs=16)
            geometry_evidence = []
            for scene,support in ((source,source_support),(target,target_support)):
                valid_cells = covers(scene.valid,footprints)
                coverage_started = time.perf_counter()
                coverage_cells = _axis_surface_coverage(axis, support, cell_starts, cell_ends, cell_array, cell_lengths)
                timing['station_surface_coverage'] += time.perf_counter()-coverage_started
                geometry_evidence.append(dict(zip(selected,zip(valid_cells,coverage_cells))))
            junction_cells = dict(zip(selected,intersects(source.junction,cell_array)|intersects(target.junction,cell_array)))
            timing['station_geometry'] += time.perf_counter()-geometry_started
            probability_started = time.perf_counter()
            coordinates = [source.probability._axis_coordinates(cell, road_width=width, position_tolerance=tolerance)
                           for cell, width in zip(cells.values(), widths[selected])]
            source_probabilities = dict(zip(selected, source.probability.sample_axes(list(cells.values()), widths[selected], tolerance, coordinates=coordinates)))
            target_probabilities = dict(zip(selected, target.probability.sample_axes(list(cells.values()), widths[selected], tolerance, coordinates=coordinates)))
            timing['station_probability'] += time.perf_counter()-probability_started
            sampling_seconds += time.perf_counter()-sampling_started
            match_started = time.perf_counter()
            matches = {i: target.match(axis, float(stations[i]), tolerance, widths[i], point=points[i], normal=normals[i]) for i in selected}
            matching_seconds += time.perf_counter()-match_started
            section_started = time.perf_counter()
            centers, target_centers, paired_normals = [], [], {}
            paired_target_points = {}
            if side == 'before':
                for i, match in matches.items():
                    if match is None or not match['reliable'] or not width_needed[i]:
                        continue
                    normal = normals[i].copy()
                    target_normal = _normal(target.lines[match['target']], match['station'])
                    if np.dot(normal, target_normal) < 0:
                        target_normal = -target_normal
                    normal += target_normal
                    normal /= max(np.linalg.norm(normal), 1e-9)
                    paired_normals[i] = normal
                    centers.append(points[i])
                    paired_target_points[i] = target.lines[match['target']].interpolate(match['station'])
                    target_centers.append(paired_target_points[i])
                samplers = [BatchSections(scene.probability, locations, list(paired_normals.values()), width_config.normal_half_length)
                            for scene, locations in ((source, centers), (target, target_centers))]
            timing['exact_width_measurement'] += time.perf_counter()-section_started
            for index in selected:
                sample_started = time.perf_counter()
                width_seconds_before = timing['exact_width_measurement']
                match_seconds = 0.
                station, point, cell, width = float(stations[index]), points[index], cells[index], widths[index]
                match_started = time.perf_counter()
                match = matches[index]
                match_seconds += time.perf_counter()-match_started
                source_evidence = source.evidence(cell, True, width, tolerance, source_support, source_probabilities[index], geometry_evidence=geometry_evidence[0][index])
                target_evidence = target.evidence(cell, match is not None, width, tolerance, target_support, target_probabilities[index], geometry_evidence=geometry_evidence[1][index])
                bef, aft = (source_evidence, target_evidence) if side == "before" else (target_evidence, source_evidence)
                junction = bool(junction_cells[index])
                accepted = bef["valid"] and aft["valid"] and not junction
                row = {"side": side, "axis_id": line_id, "station_m": station,
                       "candidate_type": change_type, "matched": match is not None,
                       "match_reliable": bool(match and match["reliable"]),
                       "offset_m": match["distance"] if match else None,
                       "target_axis": match["target"] if match else None,
                       "junction": junction, "existence_pass": False,
                       "reason": f'{bef["reason"]}->{aft["reason"]}',
                       "geometry": cell,
                       **{f"before_{k}": v for k, v in bef.items()},
                       **{f"after_{k}": v for k, v in aft.items()}}
                station_rows.append(row)
                counts[f"{side}_matched_cells" if match else f"unmatched_{side}_cells"] += 1
                if match is None:
                    counts[f"{change_type}_candidate_cells"] += 1
                if side != "before":
                    matching_seconds += match_seconds
                    sampling_seconds += time.perf_counter()-sample_started-match_seconds
                    continue
                target_point = paired_target_points.get(index)
                if target_point is None:
                    target_point = target.lines[match["target"]].interpolate(match["station"]) if match else point
                common = Point((point.x+target_point.x)/2, (point.y+target_point.y)/2)
                valid_width = width_needed[index] and accepted and match is not None and match["reliable"] and bef["state"] == aft["state"] == "present"
                if valid_width:
                    match_started = time.perf_counter()
                    reverse = source.match(target.lines[match["target"]], match["station"], tolerance, target.width(target_point))
                    match_seconds += time.perf_counter()-match_started
                    valid_width = reverse is not None and reverse["target"] == line_id and reverse["reliable"]
                before_width = after_width = None
                reason = "unreliable_match_or_existence_or_junction"
                if valid_width:
                    exact_started = time.perf_counter()
                    normal = paired_normals[index]
                    if width_surfaces is None:
                        width_surfaces = ((source_surface, source_surface.buffer(.1)),
                                          (target_surface, target_surface.buffer(.1)))
                    measurements = []
                    for (scene, centre), (surface, support), sampler in zip(((source, point), (target, target_point)), width_surfaces, samplers):
                        measurements.append(_measure_period_width(centre, normal, surface, support,
                                                                  sampler, scene.crs, width_config))
                    timing['exact_width_measurement'] += time.perf_counter()-exact_started
                    b, a = measurements
                    before_width, after_width = b.final_width, a.final_width
                    reason = ";".join(filter(None, (b.reject_reason, a.reject_reason)))
                    valid_width = before_width is not None and after_width is not None and not reason
                    if valid_width:
                        # Full measured cross sections must lie in both valid areas.
                        radius = max(before_width, after_width)/2+tolerance+1
                        valid_width = source.valid.covers(point.buffer(radius)) and target.valid.covers(target_point.buffer(radius))
                        if not valid_width:
                            reason = "width_cross_section_boundary"
                samples.append(PairedWidthSample(str(line_id), index, station, station/max(axis.length, 1e-9), common,
                                                before_width, after_width,
                                                after_width-before_width if valid_width else None,
                                                bool(valid_width), reason))
                matching_seconds += match_seconds
                sampling_seconds += (time.perf_counter()-sample_started-match_seconds
                                     -(timing['exact_width_measurement']-width_seconds_before))
            local, intervals, presence_counts = presence_seeds(
                axis, station_rows, source, coverage, change_type, minimum_length, minimum_area, intervals=intervals)
            records.extend(local)
            counts.update(presence_counts)
            if presence_audit is not None:
                presence_audit.extend(intervals)
            audit.extend(station_rows)
            if side == "before" and len(samples) >= 2:
                by_index = {s.sample_index: s for s in samples}
                samples = [by_index.get(i) or PairedWidthSample(str(line_id), i, float(stations[i]),
                           float(stations[i]/axis.length), points[i], None, None, None, False,
                           'certified_stable_width') for i in range(count)]
                # Stations follow the before road; paired points define a shared local axis.
                # Keep source stationing for run lengths, paired normals for each measurement.
                profile = PairedWidthProfile(str(line_id), axis, tuple(samples), sum(s.valid for s in samples)/len(samples))
                counts["width_matched_candidate_axes"] += int(any(row["matched"] for row in station_rows))
                counts["width_valid_profile_axes"] += int(len(profile.valid_samples) >= width_config.minimum_samples)
                runs = candidate_change_runs(profile, width_config)
                counts["width_threshold_runs"] += len(runs)
                for run in runs:
                    decision = evaluate_change_run(run.samples, axis_length=run.axis.length,
                                                   valid_ratio=run.valid_ratio, config=width_config)
                    pixel_uncertainty = .5*np.hypot(getattr(before.probability, "pixel_size", 1.),
                                                   getattr(after.probability, "pixel_size", 1.))
                    uncertainty_pass = abs(run.width_diff) >= width_config.uncertainty_scale*max(pixel_uncertainty, run.uncertainty)
                    counts["width_uncertainty_passed_runs"] += int(uncertainty_pass)
                    counts["width_continuity_passed_runs"] += int(uncertainty_pass and run.axis.length >= minimum_length)
                    decision["accepted"] = bool(decision["accepted"] and uncertainty_pass and all(s.valid for s in run.samples))
                    decision["paired_pixel_uncertainty_m"] = pixel_uncertainty
                    if not all(s.valid for s in run.samples):
                        decision["reject_reason"] = "invalid_sample_gap"
                    if not uncertainty_pass:
                        decision["reject_reason"] = "paired_uncertainty_or_pixel_floor"
                    width_audit.append({"axis_id": line_id, "sign": run.sign, **decision, "geometry": run.axis})
                    if decision["accepted"]:
                        points = [s.point for s in run.samples]
                        canonical = LineString(points)
                        outer, inner = max(run.before_width, run.after_width), min(run.before_width, run.after_width)
                        geometry = canonical.buffer(outer/2, cap_style="flat").difference(canonical.buffer(inner/2, cap_style="flat"))
                        if geometry.area >= minimum_area:
                            records.append({"change_typ": "widened" if run.sign > 0 else "narrowed",
                                            "width_bef": run.before_width, "width_aft": run.after_width,
                                            "width_diff": run.width_diff, "length_m": run.axis.length,
                                            "axis_wkt": canonical.wkt, "geometry": geometry})
                # Include profiles with no threshold run so rejection is visible.
                if not runs:
                    width_audit.append({"axis_id": line_id, "accepted": False,
                                        "sample_count": len(profile.valid_samples), "valid_ratio": profile.valid_ratio,
                                        "reject_reason": "no_sustained_threshold_samples" if profile.valid_samples else "no_valid_paired_samples",
                                        "geometry": axis})
            if line_id % 25 == 0:
                print(f"[Fast Auto] {side} axes {line_id+1}/{len(source.lines)}", flush=True)
    # Keep historical sampling inclusive of the unhidden surface wait; worker
    # time overlapping probability/matching/stations must not be double-counted.
    sampling_seconds += timing['axis_surface_wait']
    print(f"[Fast timing] auto_station_sampling={sampling_seconds:.6f}s auto_matching={matching_seconds:.6f}s", flush=True)
    timing['exact_station_sampling'] = sampling_seconds
    counts['stable_skipped_station_count'] = counts['total_station_count']-counts['candidate_station_count']
    counts['candidate_ratio'] = counts['candidate_station_count']/max(1, counts['total_station_count'])
    counts['width_prefilter_seconds'] = timing['width_prefilter']
    counts['exact_width_measurement_seconds'] = timing['exact_width_measurement']
    counts.update({f'timing_{key}_seconds': value for key, value in timing.items()})
    print('[Fast timing] ' + ' '.join(f'{key}={timing[key]:.6f}s' for key in
          ('presence_prefilter', 'width_prefilter', 'axis_cell_construction', 'axis_surface_wait',
           'surface_pipeline_worker_busy', 'surface_pipeline_active_wall', 'surface_pipeline_overlap',
           'surface_pipeline_wall', 'axis_surface_clip_worker_sum', 'axis_surface_buffer_worker_sum',
           'station_geometry', 'station_surface_coverage', 'station_probability', 'exact_station_sampling', 'exact_width_measurement')) +
          f" candidate_station_count={counts['candidate_station_count']} total_station_count={counts['total_station_count']}" +
          f" stable_skipped_station_count={counts['stable_skipped_station_count']} candidate_ratio={counts['candidate_ratio']:.6f}" +
          f" width_prefilter_seconds={timing['width_prefilter']:.6f} exact_width_measurement_seconds={timing['exact_width_measurement']:.6f}", flush=True)
    return records, audit, width_audit, dict(counts)


@timed_stage("auto_total")
def detect_final_road_changes(before_result, after_result, output_dir, *, before_period, after_period,
                              position_tolerance, width_change_absolute, width_change_ratio,
                              min_change_area, min_change_length, internal_outputs):
    from .fast_pipeline import _load_fast_period_result, _read_fast_change_layer
    started = time.perf_counter()
    payloads = [_load_fast_period_result(value) for value in (before_result, after_result)]
    centerlines = [_read_fast_change_layer(p, "centerlines") for p in payloads]
    output_crs = centerlines[0].crs
    metric_crs = centerlines[0].estimate_utm_crs() if output_crs.is_geographic else output_crs
    if not metric_crs.is_projected or abs(metric_crs.axis_info[0].unit_conversion_factor-1) > 1e-6:
        metric_crs = centerlines[0].estimate_utm_crs()
    from .batch_runtime import active_scene_cache
    from .auto_scene_cache import close_scene
    scene_cache = active_scene_cache()
    scenes = []
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for p, lines in zip(payloads, centerlines):
            def load_scene():
                surfaces, widths, valid = [_read_fast_change_layer(p, key).to_crs(metric_crs)
                                           for key in ("surfaces", "width_segments", "valid_observation")]
                probability = WindowedProbability(p["road_probability"], metric_crs)
                try:
                    return RoadScene(lines.to_crs(metric_crs), surfaces, widths, valid, probability, metric_crs)
                except Exception:
                    probability.close()
                    raise
            scenes.append(scene_cache.get(p,metric_crs,load_scene) if scene_cache is not None else load_scene())
        presence_audit = []
        records, audit, width_audit, counts = analyze_scenes(
            *scenes, tolerance=float(position_tolerance), absolute=float(width_change_absolute),
            relative=float(width_change_ratio), minimum_length=24. if min_change_length is None else float(min_change_length),
            minimum_area=float(min_change_area), presence_audit=presence_audit)
        return finalize_auto_candidates(records, audit, width_audit, counts, presence_audit=presence_audit,
                                        scenes=dict(zip(("before", "after"), scenes)), centerlines=centerlines,
                                        output_dir=output_dir, before_period=before_period, after_period=after_period,
                                        position_tolerance=position_tolerance, min_change_area=min_change_area,
                                        min_change_length=min_change_length, elapsed_seconds=time.perf_counter()-started,
                                        internal_outputs=internal_outputs)
    finally:
        if scene_cache is None:
            for scene in scenes:close_scene(scene)


def finalize_auto_candidates(records, audit, width_audit, counts, *, presence_audit, scenes, centerlines,
                             output_dir, before_period, after_period, position_tolerance=3.,
                             min_change_area=4., min_change_length=None, elapsed_seconds=0., diagnostics=False,
                             internal_outputs=False):
    """Qualify, assemble and publish; also reusable with saved observation evidence."""
    from .fast_pipeline import _fast_polygon_parts, _write_fast_public_changes
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_crs, output_crs = scenes["before"].crs, centerlines[0].crs
    counts = dict(counts)
    def frame(rows):
        return (gpd.GeoDataFrame(rows, geometry="geometry", crs=metric_crs) if rows else
                gpd.GeoDataFrame({"change_typ": pd.Series(dtype=str)}, geometry=[], crs=metric_crs))
    from .auto_change_assembly import assemble_change_objects, write_assembly_audit
    from .auto_presence_candidates import annotate_objects, qualify_presence_candidates
    raw_candidates = frame(records)
    for key, default in (("qa_state", "confirmed"), ("confidence", .9), ("audit_reason", "paired_width_accepted")):
        raw_candidates[key] = raw_candidates[key].fillna(default) if key in raw_candidates else default
    raw_candidates["candidate_id"] = np.arange(len(raw_candidates))
    observation = frame(audit)
    seeds, candidate_audit = qualify_presence_candidates(raw_candidates, scenes, observation,
                                minimum_length=24. if min_change_length is None else float(min_change_length),
                                minimum_area=float(min_change_area))
    from .auto_width_precision import qualify_width_candidates
    candidate_audit, width_samples = qualify_width_candidates(candidate_audit, scenes,
                                minimum_length=24. if min_change_length is None else float(min_change_length))
    seeds = candidate_audit.loc[candidate_audit.publication_state == 'accepted'].copy().reset_index(drop=True)
    assembly_started = time.perf_counter()
    changes, assembly = assemble_change_objects(seeds, centerlines[0], centerlines[1])
    changes = annotate_objects(changes, seeds, assembly["membership"])
    from .auto_change_geometry import render_change_geometry
    changes, geometry_audit = render_change_geometry(
        changes, seeds, assembly, {side: scene.widths for side, scene in scenes.items()}, width_samples)
    assembly["change_objects"] = changes
    print(f"[Fast timing] auto_assembly={time.perf_counter()-assembly_started:.6f}s", flush=True)
    io_started = time.perf_counter()
    if diagnostics:
        write_assembly_audit(output_dir, seeds, assembly)
    elif internal_outputs:
        # Posterior correction/finalization consumes these layers independently
        # of optional candidate diagnostics. Replace stale diagnostic layers too.
        network_path = output_dir / 'network_assembly.gpkg'
        network_path.unlink(missing_ok=True)
        for name in ('change_objects', 'object_axes'):
            assembly[name].to_file(network_path, layer=name, driver='GPKG')
    changes = changes.to_crs(output_crs)
    # Reprojection of touching width ribbons can create sub-pixel ring
    # intersections. Keep only valid polygon components for GIS publication.
    changes.geometry = changes.geometry.map(lambda g: union_all(_fast_polygon_parts(g, min_area=0.)))
    changes["before_per"], changes["after_per"] = before_period, after_period
    _, public_path = _write_fast_public_changes(changes, output_dir)
    if not diagnostics:
        if not internal_outputs:
            (output_dir/'network_assembly.gpkg').unlink(missing_ok=True)
        for name in ('auto_diagnostics.gpkg',
                     'existence_candidates.csv', 'width_candidates.csv',
                     'assembly_membership.csv', 'assembly_decisions.csv',
                     'assembly_summary.json', 'candidate_funnel.json'):
            (output_dir/name).unlink(missing_ok=True)
    if diagnostics:
        gpkg = output_dir / "auto_diagnostics.gpkg"
        changes.to_file(gpkg, layer="changes", driver="GPKG")
        for name, geometry_frame in geometry_audit.items():
            geometry_frame.to_file(gpkg, layer=name, driver="GPKG")
        observation.to_file(gpkg, layer="existence_candidates", driver="GPKG")
        raw_candidates.to_file(gpkg, layer="input_candidates", driver="GPKG")
        candidate_audit.to_file(gpkg, layer="candidate_audit", driver="GPKG")
        candidate_audit.loc[candidate_audit.publication_state == "review"].to_file(gpkg, layer="review_candidates", driver="GPKG")
        frame(presence_audit).to_file(gpkg, layer="presence_intervals", driver="GPKG")
        seeds.to_file(gpkg, layer="local_seeds", driver="GPKG")
        frame(width_audit).to_file(gpkg, layer="width_candidates", driver="GPKG")
        frame(width_samples).to_file(gpkg, layer="width_precision_samples", driver="GPKG")
        candidate_audit.loc[candidate_audit.change_typ.isin(['widened','narrowed'])].to_file(
            gpkg, layer="width_precision_candidates", driver="GPKG")
    layers = {"changes": str(public_path)}
    names = {"added": "added_roads.shp", "removed": "removed_roads.shp",
             "widened": "widened_road_parts.shp", "narrowed": "narrowed_road_parts.shp"}
    for kind, name in names.items():
        selected = changes.loc[changes.change_typ == kind]
        if diagnostics:
            selected.to_file(output_dir/name, encoding="UTF-8")
            layers[kind] = str(output_dir/name)
        counts[f"final_{kind}"] = len(selected)
    if diagnostics:
        evidence = pd.DataFrame([{k: v for k, v in row.items() if k != "geometry"} for row in audit])
        evidence.to_csv(output_dir/"existence_candidates.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{k: v for k, v in row.items() if k != "geometry"} for row in width_audit]).to_csv(
            output_dir/"width_candidates.csv", index=False, encoding="utf-8-sig")
        funnel = {"count_units": "evidence: 4 m station cells; presence candidates: longitudinal intervals; final: network objects",
                  "road_matching": {k: v for k, v in counts.items() if "matched" in k and not k.startswith("width")},
                  "width": {k: v for k, v in counts.items() if k.startswith("width")},
                  "final": {k: v for k, v in counts.items() if k.startswith("final")}}
        for kind, side in (("added", "after"), ("removed", "before")):
            candidates = evidence.loc[(evidence.side == side) & ~evidence.matched] if len(evidence) else evidence
            funnel[kind] = {"candidate_cells": len(candidates)}
            if len(candidates):
                for period in ("before", "after"):
                    funnel[kind][period] = {
                        "geometry_support": int(candidates[f"{period}_geometry"].sum()),
                        "surface_support": int((candidates[f"{period}_surface"] >= .55).sum()),
                        "probability_support": int(candidates[f"{period}_probability"].sum()),
                        "valid_area_pass": int(candidates[f"{period}_valid"].sum()),
                        **candidates[f"{period}_state"].value_counts().to_dict()}
                funnel[kind]["existence_pass"] = int(candidates.existence_pass.sum())
                funnel[kind]["continuity_pass"] = int(candidates.get("continuity_pass", pd.Series(dtype=bool)).eq(True).sum())
            funnel[kind]["final_auto_count"] = counts[f"final_{kind}"]
            funnel[kind]["longitudinal"] = {k.removeprefix(kind+"_"): v for k, v in counts.items() if k.startswith(kind+"_")}
            local = seeds.loc[seeds.change_typ == kind]
            final = changes.loc[changes.change_typ == kind]
            funnel[kind]["local_seed_count"] = len(local)
            funnel[kind]["local_seed_length_m"] = float(local.length_m.sum()) if len(local) else 0.
            funnel[kind]["seed_qa_counts"] = {state: int((local.qa_state == state).sum()) for state in ("confirmed", "probable", "uncertain")}
            funnel[kind]["object_qa_counts"] = {state: int((final.qa_state == state).sum()) for state in ("confirmed", "probable", "uncertain")}
            qa = candidate_audit.loc[candidate_audit.change_typ == kind]
            funnel[kind]["recall_candidate_count"] = len(qa)
            funnel[kind]["review_candidate_count"] = int((qa.publication_state == "review").sum())
            funnel[kind]["precision_reason_counts"] = dict(Counter(reason for reasons in qa.precision_reason for reason in reasons.split(";")))
        funnel["network_assembly"] = assembly["summary"]
        funnel['publication_review'] = {}
        for kind in ('added','removed','widened','narrowed'):
            rows = candidate_audit.loc[candidate_audit.change_typ == kind]
            review = rows.loc[rows.publication_state == 'review']
            funnel['publication_review'][kind] = dict(raw_candidates=len(rows),
                accepted_seeds=int(rows.publication_state.eq('accepted').sum()),review_candidates=len(review),
                review_reason_counts=dict(Counter(reason for reasons in review.precision_reason for reason in reasons.split(';')
                                                  if reason not in ('stable_paired_width_profile','corroborated_source_and_sustained_absence'))))
            if kind in ('widened','narrowed'):
                for reason in ('cross_track_width_match','unstable_width_profile','insufficient_sustained_width_change',
                               'junction_width_instability','road_end_width_instability','centerline_offset_measurement_bias',
                               'surface_geometry_disagreement','uncertainty_not_clearly_exceeded','alternating_width_change_signs'):
                    funnel['publication_review'][kind]['review_reason_counts'].setdefault(reason,0)
        funnel["assembly_rejection_counts"] = (assembly["decisions"].loc[~assembly["decisions"].accepted, "reason"].value_counts().to_dict()
                                                 if len(assembly["decisions"]) else {})
        funnel["count_units"] += "; assembled final: network objects"
        (output_dir/"candidate_funnel.json").write_text(json.dumps(funnel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Fast timing] diagnostics_export_io={time.perf_counter()-io_started:.6f}s", flush=True)
    return complete_written_auto_result(output_dir, before_period=before_period, after_period=after_period,
                                        min_change_length=min_change_length, elapsed_seconds=elapsed_seconds+time.perf_counter()-started,
                                        changes=changes, diagnostics=diagnostics,
                                        performance={key: value for key, value in counts.items()
                                                     if key.startswith('timing_') or key.endswith('station_count')
                                                     or key in ('candidate_ratio', 'width_prefilter_seconds', 'exact_width_measurement_seconds')})


@timed_stage("auto_preview_export_io")
def complete_written_auto_result(output_dir, *, before_period, after_period, min_change_length=None,
                                 elapsed_seconds=None, changes=None, diagnostics=False, performance=None):
    """Publish previews/summary from written results; also resume a failed preview."""
    from road_change_detection import render_change_preview
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    public_path = output_dir/"road_changes.shp"
    gpkg = output_dir/"auto_diagnostics.gpkg"
    if changes is None:
        changes = gpd.read_file(public_path)
    names = {"added": "added_roads.shp", "removed": "removed_roads.shp",
             "widened": "widened_road_parts.shp", "narrowed": "narrowed_road_parts.shp"}
    layers = {"changes": str(public_path)}
    if diagnostics:
        layers.update({kind: str(output_dir/name) for kind, name in names.items()})
    preview = output_dir/"change_preview.png"
    render_change_preview(preview, changes, changes.iloc[:0], title=f"Auto {before_period} to {after_period}",
                          empty_message="No Auto change candidates")
    summary = {"execution_profile": "fast", "change_source": "fast_automatic", "change_output_mode": "fast_automatic",
               "automatic_result": True, "ground_truth_derived": False, "ground_truth_used": False,
               "before_period": before_period, "after_period": after_period,
               "presence_change_source": "final_axis_symmetric_qualified_candidates", "width_change_source": "paired_local_profile",
               "min_change_length_m": 24. if min_change_length is None else float(min_change_length),
               "changes_feature_count": len(changes), **{f"{k}_feature_count": int((changes.change_typ == k).sum()) for k in names},
               "candidate_funnel": str(output_dir/"candidate_funnel.json") if diagnostics else "", "diagnostics": str(gpkg) if diagnostics else "",
               "auto_change_total_seconds": elapsed_seconds+time.perf_counter()-started if elapsed_seconds is not None else None}
    if performance is not None:
        summary['performance'] = performance
    if (output_dir/"assembly_summary.json").is_file():
        summary["network_assembly"] = str(output_dir/"assembly_summary.json")
    summary_path = output_dir/"change_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output": str(output_dir), "summary": str(summary_path), "road_changes": str(public_path),
            "layers": layers, "gpkg": str(gpkg) if diagnostics else "", "previews": {"change": str(preview)}, "road_change": str(preview), **summary}
