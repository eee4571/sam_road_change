"""Run against existing image/centerlines; all artifacts are written under --output."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from boundary_width import Config, ImageReader, digest, load_roads, measure_road, runs, save_json
from visualize_results import plot_road, contact_sheet, plot_summary

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'radiometric_normalization_ab'


def rows_for_road(road_id, data):
    result = []
    for i, (s, xy, normal) in enumerate(zip(data['s'], data['xy'], data['normal'])):
        row = dict(road_id=road_id, sample_id=i, s_m=s, center_x=xy[0], center_y=xy[1],
                   normal_x=normal[0], normal_y=normal[1], flags=data['flags'][i],
                   accepted=not data['flags'][i])
        for method in ('baseline', 'optimized'):
            left, right = data[method][i]
            lxy, rxy = xy+normal*left, xy-normal*right
            for key, value in dict(left_distance=left, right_distance=right, width=left+right,
                left_x=lxy[0], left_y=lxy[1], right_x=rxy[0], right_y=rxy[1],
                left_confidence=data[method+'_confidence'][i, 0], right_confidence=data[method+'_confidence'][i, 1],
                confidence=data[method+'_confidence'][i].min(),
                left_response=data[method+'_response'][i, 0], right_response=data[method+'_response'][i, 1]).items():
                row[method+'_'+key] = value
        result.append(row)
    return result


def flag_crossings(data):
    # Positive signed distances guarantee section order; guard curve-induced spatial crossings too.
    width = data['optimized']
    left = data['xy']+data['normal']*width[:, :1]
    right = data['xy']-data['normal']*width[:, 1:]
    good = np.array([not f for f in data['flags']])
    for a, b in runs(good):
        if b-a < 2:
            continue
        l, r = LineString(left[a:b]), LineString(right[a:b])
        if l.intersects(r) or not l.is_simple or not r.is_simple:
            for i in range(a, b):
                data['flags'][i] = 'curve_boundary_crossing'
                data['optimized'][i] = data['baseline'][i]
                data['optimized_confidence'][i] = data['baseline_confidence'][i]
                data['optimized_response'][i] = data['baseline_response'][i]


def metrics(frame):
    accepted = frame.accepted.to_numpy(bool)
    adjacency = accepted[1:] & accepted[:-1]
    out = dict(samples=len(frame), accepted=int(accepted.sum()), accepted_fraction=float(accepted.mean()))
    for method in ('baseline', 'optimized'):
        w = frame[method+'_width'].to_numpy()
        jumps = np.abs(np.diff(w))[adjacency]
        out[method+'_mean_abs_width_step_m'] = float(jumps.mean()) if len(jumps) else None
        out[method+'_jumps_gt_3m'] = int((jumps > 3).sum())
        out[method+'_median_width_m'] = float(np.nanmedian(w)) if np.isfinite(w).any() else None
    return out


def export_vectors(frame, crs, output):
    points = gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame.center_x, frame.center_y), crs=crs)
    points.to_file(output/'measurements.gpkg', layer='samples', driver='GPKG')
    sections, boundaries = [], []
    for road_id, group in frame.groupby('road_id', sort=False):
        for method in ('baseline', 'optimized'):
            for row in group.itertuples():
                d = row._asdict()
                if np.isfinite(d[method+'_width']):
                    sections.append(dict(road_id=road_id, sample_id=d['sample_id'], method=method,
                        accepted=d['accepted'], flags=d['flags'], width=d[method+'_width'],
                        geometry=LineString([(d[method+'_left_x'], d[method+'_left_y']),
                                             (d[method+'_right_x'], d[method+'_right_y'])])))
            for a, b in runs(group.accepted.to_numpy(bool)):
                if b-a < 2:
                    continue
                for side in ('left', 'right'):
                    part = group.iloc[a:b]
                    line = LineString(part[[method+'_'+side+'_x', method+'_'+side+'_y']].to_numpy())
                    boundaries.append(dict(road_id=road_id, method=method, side=side,
                        start_sample=int(part.sample_id.iloc[0]), end_sample=int(part.sample_id.iloc[-1]),
                        geometry=line))
    if sections:
        gpd.GeoDataFrame(sections, crs=crs).to_file(output/'measurements.gpkg', layer='cross_sections', driver='GPKG')
    if boundaries:
        gpd.GeoDataFrame(boundaries, crs=crs).to_file(output/'measurements.gpkg', layer='accepted_boundaries', driver='GPKG')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--image', type=Path, default=SOURCE/'inputs/raw/20250118.tif')
    parser.add_argument('--centerlines', type=Path, default=SOURCE/'irmad/raw_toponet/raw_toponet.gpkg')
    parser.add_argument('--layer', default='T1_edges')
    parser.add_argument('--output', type=Path, default=HERE/'results/real_20250118_final')
    parser.add_argument('--max-roads', type=int, default=120, help='Longest N eligible chains; 0 means all')
    parser.add_argument('--min-length', type=float, default=70, help='Minimum chain length in metres; 0 includes short chains')
    parser.add_argument('--skip-road-plots', action='store_true', help='Keep numerical diagnostics but skip thousands of individual PNGs')
    parser.add_argument('--confidence-threshold', type=float, default=.16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Use a new output directory: {args.output}')
    if args.min_length < 0 or args.max_roads < 0:
        parser.error('Road limits must be nonnegative')
    c = Config(max_roads=args.max_roads, min_length=args.min_length, confidence_threshold=args.confidence_threshold)
    args.output.mkdir(parents=True)
    (args.output/'roads').mkdir()
    started = time.time()
    roads, junctions, crs, selection = load_roads(args.centerlines, args.layer, c)
    if not roads:
        raise ValueError('No line meets the minimum length requirement')
    reader = ImageReader(args.image, crs)
    tree = cKDTree(junctions) if len(junctions) else None
    save_json(args.output/'manifest.json', dict(config=asdict(c), image=str(args.image.resolve()),
        centerlines=str(args.centerlines.resolve()), layer=args.layer,
        image_sha256=digest(args.image), centerlines_sha256=digest(args.centerlines),
        metric_crs=crs.to_string(), image_crs=reader.ds.crs.to_string(), image_transform=list(reader.ds.transform),
        source_note='Existing Fast-enhanced SAMRoad/TopoNet saved graph; no surface or width input',
        junctions=len(junctions), selection=selection))
    gpd.GeoDataFrame(dict(road_id=[f'road_{i:03d}' for i in range(len(roads))]),
                     geometry=roads, crs=crs).to_file(args.output/'input_roads.gpkg', driver='GPKG')
    frames, summaries = [], []
    for index, line in enumerate(roads):
        road_id = f'road_{index:03d}'
        data = measure_road(reader, line, tree, c)
        flag_crossings(data)
        frame = pd.DataFrame(rows_for_road(road_id, data))
        frames.append(frame)
        summary = dict(road_id=road_id, length_m=line.length, **metrics(frame))
        summaries.append(summary)
        np.savez_compressed(args.output/'roads'/f'{road_id}_profiles.npz', **{k: v for k, v in data.items() if k != 'flags'})
        frame.to_csv(args.output/'roads'/f'{road_id}.csv', index=False)
        if not args.skip_road_plots:
            plot_road(reader, frame, args.output/'roads'/f'{road_id}.png')
        if index % (100 if args.skip_road_plots else 10) == 0:
            print(f'{index+1}/{len(roads)} {road_id}: accepted={summary["accepted_fraction"]:.1%}', flush=True)
    reader.ds.close()
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.output/'samples.csv', index=False)
    frame[['road_id', 'sample_id', 's_m', 'accepted', 'flags', 'baseline_width', 'optimized_width',
           'optimized_left_distance', 'optimized_right_distance', 'optimized_confidence']].to_csv(
               args.output/'width_profiles.csv', index=False)
    # pandas converts nonfinite missing measurements to strict JSON null.
    (args.output/'samples.json').write_text(frame.to_json(orient='records', indent=2, force_ascii=False), encoding='utf-8')
    export_vectors(frame, crs, args.output)
    pd.DataFrame(summaries).to_csv(args.output/'road_summary.csv', index=False)
    good_pairs = (frame.accepted & frame.accepted.shift(fill_value=False) & frame.road_id.eq(frame.road_id.shift())).to_numpy()
    aggregate = dict(roads=len(roads), samples=len(frame), accepted=int(frame.accepted.sum()),
        accepted_fraction=float(frame.accepted.mean()), flags=frame.loc[~frame.accepted, 'flags'].value_counts().to_dict(),
        elapsed_seconds=time.time()-started, caution='Continuity metrics are not boundary accuracy; no ground truth used')
    for method in ('baseline', 'optimized'):
        jumps = frame[method+'_width'].diff().abs().to_numpy()[good_pairs]
        aggregate[method] = dict(mean_abs_width_step_m=float(jumps.mean()) if len(jumps) else None,
            p95_abs_width_step_m=float(np.percentile(jumps, 95)) if len(jumps) else None,
            jumps_gt_3m=int((jumps > 3).sum()), adjacent_accepted_pairs=len(jumps))
    save_json(args.output/'summary.json', aggregate)
    if not args.skip_road_plots:
        contact_sheet(args.output)
    plot_summary(frame, args.output/'summary.png')
    print(aggregate, flush=True)


if __name__ == '__main__':
    main()
