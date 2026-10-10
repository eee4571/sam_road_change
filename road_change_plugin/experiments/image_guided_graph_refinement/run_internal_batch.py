"""Apply the same internal-road experiment to saved scene windows independently.

Uses a previous complete experimental network, not historic ROI geometries.
Promotion requires explicit visual-review records; it never alters project files.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import geopandas as gpd
from shapely import union_all
from shapely.geometry import box

from bootstrap import ROOT
from data import read_json, discover, fingerprint
from report import write_json
from run_corridor_experiment import output_directory
from run_internal_experiment import main as run_region
from assemble_multi_roi import stitch_regions, graph_summary


def promote(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Promotion must stay inside experiment outputs')
    manifest = read_json(folder/'batch_inputs.json')
    source = Path(manifest['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Source network changed')
    reviews = read_json(folder/'visual_review.json')
    selected = [r for r in read_json(folder/'selection.json')['regions']
                if reviews.get(r['id'], {}).get('promote') is True]
    if not selected:
        raise ValueError('No visually reviewed ROI selected')
    for r in selected:
        if not reviews[r['id']].get('observations'):
            raise ValueError('Promotion requires actual visual observations')
    windows = [box(*r['bbox_m']) for r in selected]
    if any(a.intersection(b).area > .001 for i,a in enumerate(windows) for b in windows[:i]):
        raise ValueError('Overlapping edits require conflict resolution')
    before = gpd.read_file(source, layer='centerlines')
    regions = [gpd.read_file(folder/r['id']/'final_centerlines.gpkg', layer='roi') for r in selected]
    after = stitch_regions(before, regions, windows)
    old,new = graph_summary(before),graph_summary(after)
    domain = union_all(windows)
    a,b = union_all(before.geometry).difference(domain),union_all(after.geometry).difference(domain)
    delta = a.difference(b.buffer(.001)).length+b.difference(a.buffer(.001)).length
    validation = dict(before=old, after=new, outside_changed_m=delta, promoted=[r['id'] for r in selected])
    write_json(folder/'assembly_validation.json', validation)
    if delta > .02 or new['components'] > old['components'] or new['degree_1'] > old['degree_1'] or new['invalid_features'] or new['self_intersecting_features'] > old['self_intersecting_features']:
        raise ValueError('Whole-period topology or boundary check failed')
    after.to_file(folder/'final_centerlines.gpkg', layer='centerlines', driver='GPKG')
    print('PROMOTED',validation,flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--selection',type=Path)
    p.add_argument('--only',nargs='+')
    p.add_argument('--promote',type=Path)
    args = p.parse_args(argv)
    if args.promote:
        return promote(args.promote)
    if not all((args.source,args.inputs,args.selection)):
        p.error('--source, --inputs and --selection are required')
    saved = read_json(args.inputs)
    inputs = discover(saved['project'],saved['period'],saved['area'],record=saved['selected_period_record'])
    signatures = fingerprint(inputs)
    source = args.source.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = gpd.read_file(source,layer='centerlines')
    selection = read_json(args.selection)
    if str(frame.crs) != selection['crs']:
        raise ValueError('Selection CRS mismatch')
    regions = [r for r in selection['regions'] if not args.only or r['id'] in args.only]
    if args.only and len(regions) != len(set(args.only)):
        raise ValueError('Unknown ROI ID')
    folder = output_directory('internal_multi_')
    print('BATCH',folder,flush=True)
    write_json(folder/'batch_inputs.json',dict(source=str(source),source_sha256=digest,inputs=str(args.inputs.resolve()),fingerprints=signatures))
    write_json(folder/'selection.json',dict(regions=regions,crs=str(frame.crs),
        note='Reuse spatial windows only; historical metrics are not current measurements'))
    summaries = []
    for row in regions:
        identifier = row['id']
        if not identifier or not all(c.isalnum() or c=='_' for c in identifier):
            raise ValueError('Invalid ROI ID')
        print('START',identifier,row['category'],flush=True)
        seed = folder/'seeds'/identifier; seed.mkdir(parents=True)
        roi = box(*row['bbox_m'])
        local = frame[frame.intersects(roi.buffer(100))].copy().reset_index(drop=True)
        local.to_file(seed/'final_centerlines.gpkg',layer='context',driver='GPKG')
        write_json(seed/'inputs.json',dict(saved,bbox=row['bbox_m'],fingerprints=signatures))
        child = run_region(['--source-roi',str(seed)])
        shutil.copytree(child,folder/identifier)
        summary = read_json(child/'summary.json')
        summary.update(id=identifier,category=row['category'],reason=row['reason'])
        write_json(folder/identifier/'summary.json',summary)
        summaries.append(summary)
        write_json(folder/'summary.json',summaries)
        print('DONE',identifier,summary['actions'],flush=True)
    unchanged = signatures == fingerprint(inputs) and digest == hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(folder/'RUN_STATUS.json',dict(status='awaiting_visual_review',input_files_unchanged=unchanged,completed=len(summaries)))
    if not unchanged:
        raise ValueError('Inputs changed during batch')
    return folder


if __name__ == '__main__':
    main()
