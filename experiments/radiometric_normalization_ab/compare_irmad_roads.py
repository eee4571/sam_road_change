"""GT-free road consistency on fixed T1 stations and common valid raster pixels.

Agreement is a consistency proxy, not accuracy or proof of absence of real change.
Avoid expensive intersections with fragmented vector observation boundaries.
"""
from pathlib import Path
import os
import time

from run_experiment import environment
os.environ.update(environment())

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.features import geometry_mask, rasterize
from pyproj import Transformer
import shapely as sh
from shapely import STRtree

from irmad_rrn import ROOT, OUT, BASE, read, paired_tiles
from run_experiment import save

METRIC_CRS = 'EPSG:32650'
SPACING = 2.


def describe(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values): return dict(n=0, mean=None, median=None, p90=None)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)), p90=float(np.percentile(values, 90)))


def stations(frame):
    geoms, distances, weights, ids = [], [], [], []
    for i, line in enumerate(frame.geometry):
        if line.is_empty or line.length<=0: continue
        n = max(1, int(np.ceil(line.length/SPACING)))
        step = line.length/n
        geoms.extend([line]*n)
        distances.extend((np.arange(n)+.5)*step)
        weights.extend([step]*n)
        ids.extend([i]*n)
    geoms, distances = np.array(geoms, dtype=object), np.array(distances)
    points = sh.line_interpolate_point(geoms, distances)
    lo = sh.get_coordinates(sh.line_interpolate_point(geoms, np.maximum(0, distances-2)))
    hi = sh.get_coordinates(sh.line_interpolate_point(geoms, np.minimum(sh.length(geoms), distances+2)))
    directions = hi-lo
    directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-10)
    return dict(points=points, directions=directions, weights=np.array(weights), ids=np.array(ids),
                interior=(distances>2)&(distances<sh.length(geoms)-2))


def match(sample, frame, radius=5.):
    """Nearest direction-compatible polyline, with 30 degree tangent tolerance."""
    lines = frame.geometry.to_numpy()
    n = len(sample['points'])
    result = dict(distance=np.full(n, np.inf), lateral=np.full(n, np.nan), index=np.full(n, -1, dtype=int),
                  interior=np.zeros(n,dtype=bool))
    if not len(lines): return result
    tree = STRtree(lines)
    for start in range(0, n, 20000):
        p, line_id = tree.query(sample['points'][start:start+20000], predicate='dwithin', distance=radius)
        p = p+start
        if not len(p): continue
        local = lines[line_id]
        d = sh.line_locate_point(local, sample['points'][p])
        lo = sh.get_coordinates(sh.line_interpolate_point(local, np.maximum(0, d-2)))
        hi = sh.get_coordinates(sh.line_interpolate_point(local, np.minimum(sh.length(local), d+2)))
        tangent = hi-lo
        norm = np.linalg.norm(tangent, axis=1)
        cosine = np.abs(np.sum(tangent*sample['directions'][p], axis=1))/np.maximum(norm, 1e-10)
        valid = (cosine >= np.cos(np.deg2rad(30))) & (norm>0)
        p, line_id, local, d = p[valid], line_id[valid], local[valid], d[valid]
        if not len(p): continue
        nearest = sh.line_interpolate_point(local, d)
        delta = sh.get_coordinates(nearest)-sh.get_coordinates(sample['points'][p])
        distance = np.linalg.norm(delta, axis=1)
        interior = (d>2)&(d<sh.length(local)-2)
        lateral = np.abs(delta[:, 0]*sample['directions'][p, 1]-delta[:, 1]*sample['directions'][p, 0])
        order = np.lexsort((line_id, distance, p))
        p, line_id, distance, lateral = p[order], line_id[order], distance[order], lateral[order]
        interior=interior[order]
        unique = np.r_[True, p[1:]!=p[:-1]]
        result['distance'][p[unique]] = distance[unique]
        result['lateral'][p[unique]] = lateral[unique]
        result['index'][p[unique]] = line_id[unique]
        result['interior'][p[unique]] = interior[unique]
    return result


class CommonObservation:
    def __init__(self):
        self.tiles = []
        area = gpd.read_file(ROOT/'inputs/validation_area.shp')
        for ref, target in paired_tiles():
            with rasterio.open(ref) as a, rasterio.open(target) as b:
                mask = (a.dataset_mask()>0)&(b.dataset_mask()>0)
                mask &= geometry_mask(area.to_crs(a.crs).geometry, a.shape, a.transform, invert=True)
                self.tiles.append(dict(name=target.stem, transform=a.transform, shape=a.shape,
                                       crs=a.crs, bounds=a.bounds, mask=mask))
        self.transformer = Transformer.from_crs(METRIC_CRS, self.tiles[0]['crs'], always_xy=True)

    def select(self, points):
        coords = sh.get_coordinates(points)
        x, y = self.transformer.transform(coords[:, 0], coords[:, 1])
        valid = np.zeros(len(points), dtype=bool)
        for t in self.tiles:
            inv = ~t['transform']
            col = np.floor(inv.a*x+inv.b*y+inv.c).astype(int)
            row = np.floor(inv.d*x+inv.e*y+inv.f).astype(int)
            inside = (row>=0)&(col>=0)&(row<t['shape'][0])&(col<t['shape'][1])
            valid[inside] |= t['mask'][row[inside], col[inside]]
        return valid


def filter_sample(sample, mask):
    return {k:v[mask] for k,v in sample.items()}


def overlap_counts(a, b, valid):
    aa, bb = a & valid, b & valid
    return np.array([aa.sum(), bb.sum(), (aa&bb).sum(), (aa|bb).sum()], dtype=np.int64)


def overlap_summary(counts):
    a,b,intersection,union = map(int, counts)
    return dict(reference_pixels=a, target_pixels=b, intersection=intersection, union=union,
                iou=intersection/union if union else None, reference_coverage=intersection/a if a else None,
                target_supported=intersection/b if b else None)


def surface_metrics(layers, results, observation):
    summaries = {k:read(Path(r['width_review'])/'batch_width_summary.json') for k,r in results.items()}
    images = {k:{Path(i['image']).stem:i for i in s['images']} for k,s in summaries.items()}
    counts = {kind:{arm:np.zeros(4, dtype='int64') for arm in ('Raw_T2','Normalized_T2')}
              for kind in ('raw_molra_threshold_0_5','enhanced_molra','final_road_surface')}
    surfaces = {k:v['surfaces'].to_crs(observation.tiles[0]['crs']) for k,v in layers.items()}
    tile_metrics = {}
    for tile in observation.tiles:
        name, mask = tile['name'], tile['mask']
        buffers = {}
        for arm in results:
            item = images[arm][name]
            raw = np.load(item['molra_probability'], mmap_mode='r')>=.5
            enhanced = np.asarray(Image.open(item['molra_surface_mask']))>0
            box = sh.box(*tile['bounds'])
            frame = surfaces[arm]
            geoms = frame.geometry.iloc[frame.sindex.query(box)]
            final = (rasterize(((g,1) for g in geoms), out_shape=tile['shape'], transform=tile['transform'], dtype='uint8')>0
                     if len(geoms) else np.zeros(tile['shape'],dtype=bool))
            buffers[arm] = [raw, enhanced, final]
        tile_metrics[name] = {}
        for kind, index in zip(counts, range(3)):
            tile_metrics[name][kind] = {}
            for arm in ('Raw_T2','Normalized_T2'):
                count = overlap_counts(buffers['T1'][index], buffers[arm][index], mask)
                counts[kind][arm] += count
                tile_metrics[name][kind][arm] = overlap_summary(count)
        print('Compared surfaces', name, flush=True)
    return dict(overlap={kind:{arm:overlap_summary(c) for arm,c in arms.items()} for kind,arms in counts.items()},
                by_tile=tile_metrics,
                extraction={arm:{key:s.get(key) for key in ('enhanced_molra_width_ratio', 'raw_molra_mask_pixel_count',
                       'enhanced_molra_surface_pixel_count','enhanced_molra_width_count','fast_mask_fallback_count')}
                            for arm,s in summaries.items()},
                inference_audit={arm:dict(tiles=len(s['images']), errors=[i.get('molra_surface_error') for i in s['images'] if i.get('molra_surface_error')],
                    cache_hits=sum(bool(i.get('molra_surface_cache_hit')) for i in s['images']),
                    seconds=sum(i.get('molra_surface_seconds',0) for i in s['images'])) for arm,s in summaries.items()})


def main():
    start = time.perf_counter()
    out = OUT/'evaluation'
    out.mkdir(exist_ok=True)
    results = {'T1':read(BASE/'20250118/latest_result.json'), 'Raw_T2':read(BASE/'20260203/latest_result.json'),
               'Normalized_T2':read(OUT/'T2/latest_result.json')}
    layers = {arm:{kind:gpd.read_file(r[kind]).to_crs(METRIC_CRS).explode(index_parts=False).reset_index(drop=True)
                   for kind in ('centerlines','surfaces','width_segments')} for arm,r in results.items()}
    observation = CommonObservation()
    summary = dict(method=dict(crs=METRIC_CRS, station_spacing_m=SPACING, matching_tolerance_m=3.,
                  direction_tolerance_degrees=30, scope='common valid pixels inside input validation area; no GT',
                  stable_proxy='T1 road >=50m and >=70% length matched in BOTH raw and normalized T2 at 3m',
                  warning='T1 agreement is not accuracy. Unmatched/novel roads may include true change.'), roads={})
    for arm in layers:
        frame = layers[arm]['centerlines']
        summary['roads'][arm] = dict(objects=len(frame), total_length_m=float(frame.length.sum()),
                                    natural_width_m=describe(layers[arm]['width_segments']['width_m']))
    ref = stations(layers['T1']['centerlines'])
    ref = filter_sample(ref, observation.select(ref['points']))
    matched = {arm:match(ref, layers[arm]['centerlines']) for arm in ('Raw_T2','Normalized_T2')}
    denominator = ref['weights'].sum()
    summary['reference_observed_length_m'] = float(denominator)
    for arm,m in matched.items():
        summary['roads'][arm]['reference_longitudinal_coverage'] = {
            str(radius):float(ref['weights'][m['distance']<=radius].sum()/denominator) for radius in (1,3,5)}
    raw_ok, norm_ok = (matched[arm]['distance']<=3 for arm in ('Raw_T2','Normalized_T2'))
    groups = pd.DataFrame(dict(road=ref['ids'], length=ref['weights'], raw=ref['weights']*raw_ok, normalized=ref['weights']*norm_ok)).groupby('road').sum()
    stable_roads = groups.index[(groups.length>=50)&(groups.raw/groups.length>=.7)&(groups.normalized/groups.length>=.7)].to_numpy()
    stable = np.isin(ref['ids'], stable_roads)&raw_ok&norm_ok
    lateral_stable = stable & ref['interior'] & matched['Raw_T2']['interior'] & matched['Normalized_T2']['interior']
    summary['stable_proxy'] = dict(road_count=len(stable_roads), matched_length_m=float(ref['weights'][stable].sum()))
    for arm,m in matched.items():
        summary['roads'][arm]['stable_lateral_offset_m'] = describe(m['lateral'][lateral_stable])
    summary['reference_recovery'] = dict(recovered_m=float(ref['weights'][~raw_ok&norm_ok].sum()),
        lost_m=float(ref['weights'][raw_ok&~norm_ok].sum()),
        both_unmatched_m=float(ref['weights'][~raw_ok&~norm_ok].sum()))
    table = pd.DataFrame(dict(x=sh.get_x(ref['points']), y=sh.get_y(ref['points']), reference_road=ref['ids'],
         length_weight_m=ref['weights'], raw_distance_m=matched['Raw_T2']['distance'],
         normalized_distance_m=matched['Normalized_T2']['distance'], stable=stable, recovered=~raw_ok&norm_ok, lost=raw_ok&~norm_ok))
    for arm in ('Raw_T2','Normalized_T2'):
        sample = stations(layers[arm]['centerlines'])
        sample = filter_sample(sample, observation.select(sample['points']))
        support = match(sample,layers['T1']['centerlines'])['distance']<=3
        other = 'Raw_T2' if arm=='Normalized_T2' else 'Normalized_T2'
        shared = match(sample,layers[other]['centerlines'])['distance']<=3
        summary['roads'][arm]['target_supported_by_T1_fraction'] = float(sample['weights'][support].sum()/sample['weights'].sum())
        summary['roads'][arm]['unsupported_by_T1_length_m'] = float(sample['weights'][~support].sum())
        summary['roads'][arm]['novel_vs_T1_and_other_T2_m'] = float(sample['weights'][~support&~shared].sum())
        if arm=='Normalized_T2':
            pd.DataFrame(dict(x=sh.get_x(sample['points']), y=sh.get_y(sample['points']),
                length_weight_m=sample['weights'], unsupported_by_T1=~support, novel=~support&~shared)).to_csv(out/'normalized_stations.csv',index=False)
    print('Centerline comparison complete', flush=True)
    width_values, width_valid = {}, stable.copy()
    for arm in layers:
        frame = layers[arm]['width_segments']
        m = match(ref, frame, radius=3.)
        width = frame['width_m'].to_numpy()[np.maximum(m['index'], 0)]
        width_valid &= (m['index']>=0)&np.isfinite(width)&(width>0)
        width_values[arm] = width
    width_roads = pd.DataFrame(dict(road=ref['ids'][width_valid], length=ref['weights'][width_valid]))
    for arm in ('Raw_T2','Normalized_T2'):
        delta = width_values[arm]-width_values['T1']
        table[arm+'_width_difference_m'] = np.where(width_valid,delta,np.nan)
        summary['roads'][arm]['stable_width'] = dict(bias_m=describe(delta[width_valid]), absolute_difference_m=describe(abs(delta[width_valid])),
             fraction_absolute_over_2m=float((abs(delta[width_valid])>2).mean()) if width_valid.any() else None)
        width_roads[arm] = delta[width_valid]
    road_width = width_roads.groupby('road').agg({'length':'sum','Raw_T2':'median','Normalized_T2':'median'})
    road_width = road_width.loc[road_width.length>=30]
    for arm in ('Raw_T2','Normalized_T2'):
        summary['roads'][arm]['stable_whole_road_width'] = dict(roads=len(road_width), median_bias_m=describe(road_width[arm]),
             absolute_median_bias_m=describe(abs(road_width[arm])))
    road_width.to_csv(out/'stable_road_width.csv')
    table.to_csv(out/'reference_stations.csv', index=False)
    summary['surfaces'] = surface_metrics(layers, results, observation)
    summary['timings'] = dict(evaluation_seconds=time.perf_counter()-start,
          normalized_extraction_stages=results['Normalized_T2'].get('stage_timings'),
          baseline_reused=True)
    save(out/'metrics.json', summary)
    print('EVALUATION COMPLETE', time.perf_counter()-start, flush=True)


if __name__ == '__main__':
    main()
