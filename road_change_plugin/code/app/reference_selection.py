"""Bounded native-resolution model trials for reference recommendations.

No whole-scene reads or radiometric-medoid ranking. Extracted centerline length
is the objective; no ground-truth accuracy is claimed.
"""
from contextlib import ExitStack, redirect_stdout
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.enums import Resampling
from rasterio.warp import transform, transform_geom, reproject
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union, transform as transform_geometry

from input_catalog import read_path_list
from engine.samroad.reference_probe import RoadModelProbe

VERSION = 'road_length_probe_v5'
PIXEL_SIZE = .5
MAX_SOURCE_WINDOW = 2048
WINDOW_SIZE = 512
GRID_SIZE = 4
PREVIEW_SIZE = 64
INWARD_STEPS = (0., .15, .30, .45, .60, .75)


@lru_cache(maxsize=128)
def _transformer(source_crs, target_crs):
    """Reuse coordinate-operation discovery across dates at the same location."""
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _warp_operation(projection, source_crs):
    operation = projection.get_last_used_operation().definition
    # GDAL's explicit operation uses authority axis order, whereas shapely and
    # raster coordinates above use x/y. Our local target always uses east/north.
    # https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-ct
    axes = CRS(source_crs).axis_info
    if axes[0].direction in ('north', 'south') and axes[1].direction in ('east', 'west'):
        steps = operation[len('proj=pipeline '):] if operation.startswith('proj=pipeline ') else 'step ' + operation
        operation = 'proj=pipeline step proj=axisswap order=2,1 ' + steps
    return operation


def _stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def _write_cache(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def sampling_plan(geometry, supplemental=False):
    """Split a spatially balanced 4x4 plan into disjoint passes of at most 8.

    Serpentine traversal and alternating cells spread the first pass across
    the AOI; the optional second pass fills the remaining cells, without
    revisiting windows that have already been inferred.
    """
    left, bottom, right, top = geometry.bounds
    points, weights = [], []
    for row in range(GRID_SIZE):
        columns = range(GRID_SIZE) if row % 2 == 0 else reversed(range(GRID_SIZE))
        for col in columns:
            cell = box(left + (right-left)*col/GRID_SIZE, bottom + (top-bottom)*row/GRID_SIZE,
                       left + (right-left)*(col+1)/GRID_SIZE, bottom + (top-bottom)*(row+1)/GRID_SIZE)
            part = geometry.intersection(cell)
            if not part.is_empty and part.area > 0:
                point = part.representative_point()
                points.append((point.x, point.y))
                weights.append(float(part.area / geometry.area))
    start = 1 if supplemental else 0
    return points[start::2], weights[start::2]


def _valid_pixels(pixels):
    # Exact fill values, not a brightness threshold: bright textured roofs
    # and pale surfaces remain usable when nodata metadata is absent.
    return (~np.any(np.ma.getmaskarray(pixels),axis=0)
            & ~np.all(pixels.data==0,axis=0) & ~np.all(pixels.data==255,axis=0))


def _preview(dataset, bounds, area_geometry):
    """Check a bounded source rectangle; no per-location CRS or RGB warp."""
    corners = [dataset.index(x,y) for x in (bounds[0],bounds[2]) for y in (bounds[1],bounds[3])]
    top, left = np.min(corners, axis=0)
    bottom, right = np.max(corners, axis=0)+1
    if max(right-left, bottom-top) > MAX_SOURCE_WINDOW:
        raise ValueError('原始影像分辨率需要过大的抽样窗口，请手动选择参考期')
    window = Window(int(left), int(top), int(right-left), int(bottom-top))
    pixels = dataset.read([1,2,3], window=window, masked=True, boundless=True, fill_value=0,
                          out_shape=(3,PREVIEW_SIZE,PREVIEW_SIZE), resampling=Resampling.nearest)
    affine = window_transform(window, dataset.transform)*rasterio.Affine.scale(window.width/PREVIEW_SIZE, window.height/PREVIEW_SIZE)
    inside = geometry_mask([mapping(area_geometry)], (PREVIEW_SIZE,PREVIEW_SIZE), affine, invert=True)
    return float((_valid_pixels(pixels)&inside).mean())


def _patch(dataset, point, area_geometry):
    # Same 256 m square and 0.5 m model sampling grid at every date, even when
    # native resolution/CRS differ. Read only its bounded source window before
    # reprojecting this small array, never pass the complete raster to a warp.
    source_crs = dataset.crs.to_wkt()
    lon, lat = _transformer(source_crs, 'EPSG:4326').transform(*point)
    local_crs = f'+proj=aeqd +lat_0={lat:.12f} +lon_0={lon:.12f} +datum=WGS84 +units=m'
    half = WINDOW_SIZE*PIXEL_SIZE/2
    bounds = _transformer(local_crs, source_crs).transform_bounds(-half, -half, half, half, densify_pts=21)
    corners = [dataset.index(x,y) for x in (bounds[0],bounds[2]) for y in (bounds[1],bounds[3])]
    top,left = np.min(corners,axis=0)-2
    bottom,right = np.max(corners,axis=0)+3
    if max(right-left,bottom-top)>MAX_SOURCE_WINDOW:
        raise ValueError('原始影像分辨率需要过大的抽样窗口，请手动选择参考期')
    window = Window(int(left),int(top),int(right-left),int(bottom-top))
    pixels = dataset.read([1, 2, 3], window=window, masked=True, boundless=True, fill_value=0)
    source_valid = _valid_pixels(pixels)
    source_rgb = pixels.filled(0)
    source_rgb[:, ~source_valid] = 0
    target = rasterio.transform.from_origin(-half, half, PIXEL_SIZE, PIXEL_SIZE)
    source_transform = window_transform(window,dataset.transform)
    source_transform *= rasterio.Affine.scale(window.width/pixels.shape[2], window.height/pixels.shape[1])
    warp = dict(src_transform=source_transform,src_crs=dataset.crs,
                dst_transform=target,dst_crs=local_crs)
    projection = _transformer(source_crs, local_crs)
    local_geometry = transform_geometry(projection.transform, area_geometry)
    # Reuse the resolved PROJ operation instead of asking GDAL to rediscover
    # it for the RGB and mask warps of every date. The array resampling stays
    # in GDAL with the existing bilinear/nearest rules.
    warp['COORDINATE_OPERATION'] = _warp_operation(projection, source_crs)
    values = np.zeros((3,WINDOW_SIZE,WINDOW_SIZE),dtype='uint8')
    reproject(source_rgb,values,resampling=Resampling.bilinear,**warp)
    mask = np.zeros((WINDOW_SIZE,WINDOW_SIZE),dtype='uint8')
    reproject(source_valid.astype('uint8'),mask,resampling=Resampling.nearest,**warp)
    inside = geometry_mask([mapping(local_geometry)], mask.shape, target, invert=True)
    valid = inside & (mask>0)
    coverage = float(valid.sum()/max(1,inside.sum()))
    rgb = np.moveaxis(values,0,-1).copy()
    rgb[mask==0] = 0
    return dict(coverage=coverage,image_coverage=float(mask.mean()),rgb=rgb,valid=valid)


def _open_images(stack, source):
    source = Path(source)
    paths = [entry.path for entry in read_path_list(source).entries] if source.suffix.lower() == '.txt' else [source]
    if not paths:
        raise ValueError('影像清单为空')
    datasets = [stack.enter_context(rasterio.open(path)) for path in paths]
    for ds in datasets:
        if ds.crs is None:
            raise ValueError('影像缺少坐标系，无法进行空间抽样')
        if ds.count < 3 or ds.dtypes[0] != 'uint8' or len(set(ds.dtypes[:3])) != 1:
            raise ValueError('模型试跑推荐需要前三波段为 uint8 RGB 影像，请手动指定参考期')
    files = set(paths + [source])
    for ds in datasets:
        files.update(Path(path) for path in ds.files)
    for path in list(files):
        files.update(sidecar for suffix in ('.ovr', '.msk', '.aux.xml') if (sidecar := Path(str(path)+suffix)).is_file())
    return datasets, [_stamp(path) for path in sorted(files)]


def content_sampling_plan(sources, geometry, crs, cache_dir=None, *, supplemental=False, occupied=()):
    """Relocate blank strata inward with bounded 64px local previews.

    A strict majority of dates must contain imagery; isolated missing dates
    still face the coverage gate instead of dictating everyone else's sites.
    Location selection never sees road predictions. All dates share its result.
    """
    points, weights = sampling_plan(geometry, supplemental)
    center = geometry.representative_point()
    lon, lat = _transformer(crs, 'EPSG:4326').transform(center.x, center.y)
    metric_crs = f'+proj=aeqd +lat_0={lat:.12f} +lon_0={lon:.12f} +datum=WGS84 +units=m'
    metric = _transformer(crs, metric_crs)
    with ExitStack() as stack:
        images, stamps = [], []
        for period, source in sorted(sources):
            datasets, files = _open_images(stack, source)
            images.append([(ds, _transformer(metric_crs, ds.crs.to_wkt()),
                            shape(transform_geom(crs, ds.crs, mapping(geometry)))) for ds in datasets])
            stamps.append((period, files))
        identity = dict(version=VERSION, files=stamps, geometry=geometry.wkb_hex, crs=str(crs),
                        points=points, occupied=occupied, preview=PREVIEW_SIZE, steps=INWARD_STEPS)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf8')).hexdigest()
        can_cache = all(ds.driver != 'VRT' for rows in images for ds, _, _ in rows)
        cache_path = Path(cache_dir)/(key+'.plan.json') if cache_dir and can_cache else None
        if cache_path and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding='utf8'))
                if cached['key'] == key:
                    return cached['points'], weights, cached['details']
            except (OSError, ValueError, KeyError):
                pass
        used = [metric.transform(*p) for p in occupied]
        chosen, details = [], []
        for initial in points:
            best, best_quality, attempts = initial, -1., 0
            for fraction in INWARD_STEPS:
                point = (initial[0]+fraction*(center.x-initial[0]), initial[1]+fraction*(center.y-initial[1]))
                x, y = metric.transform(*point)
                # Keep centers at least one scored-core width apart.
                if any(np.hypot(x-u, y-v) < (WINDOW_SIZE-128)*PIXEL_SIZE for u, v in used):
                    continue
                qualities = []
                for rows in images:
                    quality = 0.
                    for ds, projection, local_geometry in rows:
                        location = projection.transform(x,y)
                        row, col = ds.index(*location)
                        if 0 <= row < ds.height and 0 <= col < ds.width:
                            half = WINDOW_SIZE*PIXEL_SIZE/2
                            bounds = projection.transform_bounds(x-half, y-half, x+half, y+half, densify_pts=3)
                            quality = max(quality, _preview(ds, bounds, local_geometry))
                            if quality >= .98:
                                break
                    qualities.append(quality)
                attempts += 1
                # Lower median requires > half of the dates, including even N.
                quality = sorted(qualities)[(len(qualities)-1)//2] if qualities else 0.
                if quality > best_quality:
                    best, best_quality = point, quality
                if quality >= .98:
                    break
            chosen.append(best)
            used.append(metric.transform(*best))
            details.append(dict(original=list(initial), center=list(best), valid_fraction=max(0., best_quality), attempts=attempts))
        if cache_path:
            try:
                _write_cache(cache_path, dict(key=key, points=chosen, details=details))
            except OSError:
                pass
        return chosen, weights, details


def sample_period(source, geometry, crs, points, weights, cache_dir=None, probe=None):
    probe = probe or RoadModelProbe()
    with ExitStack() as stack:
        datasets, stamps = _open_images(stack, source)
        identity = dict(version=VERSION, geometry=geometry.wkb_hex, crs=str(crs),
                        files=stamps, window=WINDOW_SIZE, pixel_size=PIXEL_SIZE, model=probe.identity(),
                        points=points, weights=weights)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf8')).hexdigest()
        cache_path = Path(cache_dir) / (key + '.json') if cache_dir and all(ds.driver != 'VRT' for ds in datasets) else None
        if cache_path and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding='utf8'))
                if cached.get('key') == key:
                    return {**cached['sample'], 'cached': True}
            except (OSError, ValueError, KeyError):
                pass
        footprints, spatial = [], []
        for ds in datasets:
            bounds = shape(transform_geom(ds.crs, crs, mapping(box(*ds.bounds))))
            footprints.append(bounds)
            xs, ys = transform(crs, ds.crs, [p[0] for p in points], [p[1] for p in points])
            local_geometry = shape(transform_geom(crs, ds.crs, mapping(geometry)))
            spatial.append((ds, list(zip(xs, ys)), local_geometry))
        footprint = float(unary_union(footprints).intersection(geometry).area / geometry.area)
        patches, reads = [], 0
        for index in range(len(points)):
            best = None
            for ds, locations, local_geometry in spatial:
                row, col = ds.index(*locations[index])
                if not (0 <= row < ds.height and 0 <= col < ds.width):
                    continue
                sample = _patch(ds, locations[index], local_geometry)
                reads += 1
                quality = lambda item: (min(item['coverage'], item['image_coverage']), item['coverage'])
                if best is None or quality(sample) > quality(best):
                    best = sample
                if min(best['coverage'], best['image_coverage']) >= .99:
                    break
            if best is None:
                patches.append(dict(coverage=0., model=None))
            else:
                metrics = None
                if best['coverage'] >= .8 and best['image_coverage'] >= .8:
                    metrics = probe.evaluate(best['rgb'], best['valid'])
                patches.append(dict(coverage=best['coverage'], model=metrics))
        coverage = float(sum(p['coverage']*w for p, w in zip(patches, weights))/max(sum(weights),1e-9))
        sample = dict(footprint_coverage=min(1., footprint), sampled_coverage=coverage,
                      patches=patches, windows_read=reads, source_count=len(datasets), cached=False)
        if cache_path:
            try:
                _write_cache(cache_path, dict(key=key, sample=sample))
            except OSError:
                # A read-only project can still be inspected without a cache.
                pass
        return sample


def rank_periods(samples, weights):
    """Rank actual model probes on common locations, never RGB similarity.

    Coverage gates prevent selecting a period merely because its difficult
    locations are missing. Every eligible period is scored on the same windows;
    windows with no roads contribute zero length. Empty locations provide no
    evidence for declaring any candidate better.
    """
    if not samples:
        return None
    available = sum(max(s['patches'][i]['coverage'] for s in samples.values())*w for i, w in enumerate(weights))/max(sum(weights),1e-9)
    eligible = []
    for period, sample in samples.items():
        sample['relative_valid_coverage'] = sample['sampled_coverage'] / max(available, 1e-9)
        sample['available_sample_coverage'] = available
        reasons = []
        if sample['footprint_coverage'] < .9:
            reasons.append('范围覆盖不足')
        if sample['relative_valid_coverage'] < .9:
            reasons.append('抽样有效覆盖不足')
        sample['excluded_reasons'] = reasons
        sample['score'] = None
        if not reasons:
            eligible.append(period)
    if not eligible:
        return None
    common = [i for i in range(len(weights))
              if all(samples[p]['patches'][i]['model'] is not None for p in eligible)]
    evidence = [i for i in common if any(samples[p]['patches'][i]['model']['score'] > 0 for p in eligible)]
    for period in eligible:
        item = samples[period]
        item['common_windows'] = len(common)
        item['road_windows'] = len(evidence)
        if len(common) < 3 or len(evidence) < 2:
            item['excluded_reasons'].append('可比较的道路样本不足')
            continue
        total = sum(weights[i] for i in common)
        item['score'] = float(sum(item['patches'][i]['model']['score']*weights[i] for i in common)/total)
        item['sampled_road_length_m'] = sum(item['patches'][i]['model']['length_m'] for i in common)
        item['sampled_valid_area_m2'] = sum(item['patches'][i]['model']['valid_area_m2'] for i in common)
    ranked = [p for p in eligible if samples[p]['score'] is not None and samples[p]['score'] > 0]
    return min(ranked, key=lambda p: (-samples[p]['score'], p)) if ranked else None


def close_ranking(samples):
    scores = sorted((s['score'] for s in samples.values() if s['score'] is not None), reverse=True)
    return len(scores) > 1 and scores[0]-scores[1] <= .05 * max(scores[0], .01)


def merge_samples(first, second, first_weight, second_weight):
    merged = dict(first)
    merged['sampled_coverage'] = (first['sampled_coverage']*first_weight + second['sampled_coverage']*second_weight)/max(first_weight+second_weight,1e-9)
    merged['patches'] = first['patches'] + second['patches']
    merged['windows_read'] = first['windows_read'] + second['windows_read']
    merged['cached'] = first['cached'] and second['cached']
    return merged


def recommend(data, *, force=False, use_cache=True, probe=None):
    probe = probe or RoadModelProbe()
    refs = dict(data.get('area_irmad_references', {}))
    selections, issues = {}, []
    root = data.get('project_root') or data.get('root')
    cache_dir = Path(root) / '_work/reference_selection' if root and use_cache else None
    for area, boundary in data.get('areas', []):
        if refs.get(area) and not force:
            continue
        try:
            source = gpd.read_file(boundary)
            if not source.crs:
                raise ValueError('验证区缺少坐标系')
            crs = source.crs.to_wkt()
            geometry = unary_union([item for item in source.geometry if item is not None])
            if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
                raise ValueError('验证区范围无效')
            sources = [(period, path) for grid, period, path in data.get('periods', []) if grid == area]
            points, weights, details = content_sampling_plan(sources, geometry, crs, cache_dir)
            samples = {}
            for grid, period, source in data.get('periods', []):
                if grid == area:
                    samples[period] = sample_period(source, geometry, crs, points, weights, cache_dir, probe)
            selected = rank_periods(samples, weights)
            locations = len(points)
            supplemental = selected is None or close_ranking(samples)
            if supplemental:
                more_points, more_weights, more_details = content_sampling_plan(
                    sources, geometry, crs, cache_dir, supplemental=True, occupied=points)
                for grid, period, source in data.get('periods', []):
                    if grid == area:
                        additional = sample_period(source, geometry, crs, more_points, more_weights, cache_dir, probe)
                        samples[period] = merge_samples(samples[period], additional, sum(weights), sum(more_weights))
                selected = rank_periods(samples, weights+more_weights)
                locations += len(more_points)
                details += more_details
            summaries = {period: {key: value for key, value in sample.items() if key != 'patches'} for period, sample in samples.items()}
            selections[area] = dict(period=selected, method=VERSION, sampled_locations=locations, candidates=summaries,
                                   sampling_windows=details, sampling_crs=crs,
                                   supplemental_sampling=supplemental, close_candidates=close_ranking(samples),
                                   basis='相同位置试跑的道路中心线长度，非真值精度')
            if selected is None:
                issues.append(f'{area}：模型试跑未获得足够的有效道路样本，请手动选择参考期')
            else:
                refs[area] = selected
        except (OSError, ValueError, RuntimeError, rasterio.errors.RasterioError) as exc:
            issues.append(f'{area}：自动选择参考期失败：{exc}')
    return dict(issues=issues, area_irmad_references=refs, reference_selections=selections)


def main():
    request = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        report = recommend(request['data'], force=bool(request.get('force')), use_cache=request.get('use_cache', True))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
