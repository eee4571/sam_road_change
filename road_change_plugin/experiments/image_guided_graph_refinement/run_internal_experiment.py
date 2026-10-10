"""Regularize internal roads in a previous experimental ROI, preserving its main axes."""
import argparse
import hashlib
import shutil
import geopandas as gpd
from shapely.geometry import box
from shapely import union_all, from_wkt
from bootstrap import ROOT
from data import read_json, discover, fingerprint
from evidence import Evidence
from internal_axis import regularize
from report import write_json
from repair_artifacts import save_roi_result
from run_corridor_experiment import output_directory
from audit_extraction_stages import figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-roi', type=str, required=True)
    parser.add_argument('--full-period-source', type=str, help='Previous experimental full-period GPKG; layer centerlines')
    args = parser.parse_args(argv)
    from pathlib import Path
    source = Path(args.source_roi).resolve()
    manifest = read_json(source/'inputs.json')
    inputs = discover(manifest['project'], manifest['period'], manifest['area'], record=manifest['selected_period_record'])
    signatures = fingerprint(inputs)
    path = source/'final_centerlines.gpkg'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    before = gpd.read_file(path, layer='context')
    roi = box(*manifest['bbox'])
    config = read_json(ROOT/'config.json')
    evidence = Evidence(inputs, before.crs, roi.buffer(100), config)
    evidence.multichannel_ribbon = True
    evidence.independent_flanks = True
    output = output_directory('internal_')
    print(output, flush=True)
    after, audit = regularize(before, evidence, roi)
    summary = save_roi_result(output, before, after, evidence, roi, config,
        dict(id=source.name, category='internal straight runs', bbox_m=list(roi.bounds)), audit,
        dict(reason='two-contact RGB alias reconstruction, then fixed-contact straight runs'))
    locked_before = before[before.carriageway_id.fillna('').astype(str).str.len() > 0] if 'carriageway_id' in before else before.iloc[:0]
    locked_after = after[after.carriageway_id.fillna('').astype(str).str.len() > 0] if 'carriageway_id' in after else after.iloc[:0]
    summary['main_carriageways_unchanged'] = union_all(locked_before.geometry).equals(union_all(locked_after.geometry))
    summary['input_fingerprints_unchanged'] = signatures == fingerprint(inputs)
    summary['source_result_unchanged'] = digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(summary[k] for k in ('main_carriageways_unchanged', 'input_fingerprints_unchanged', 'source_result_unchanged'))
    write_json(output/'summary.json', summary)
    write_json(output/'inputs.json', dict(manifest, source_result=str(path), source_sha256=digest))
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    for p in ROOT.glob('*.py'):
        shutil.copy2(p, snapshot/p.name)
    figure(output/'internal_before_after.png', [before,after], evidence, roi, ['Before internal fit','After internal fit'])
    edits = sorted([r for r in audit if r['applied']], key=lambda r: from_wkt(r['old_wkt']).length, reverse=True)
    windows = []
    for edit in edits:
        extent = from_wkt(edit['old_wkt']).envelope.buffer(15).envelope.intersection(roi)
        if any(extent.intersection(w).area/min(extent.area,w.area) > .6 for w in windows):
            continue
        figure(output/f'detail_{len(windows)+1:02d}.png', [before,after], evidence, extent, ['Before','After'])
        windows.append(extent)
        if len(windows) >= 3:
            break
    if args.full_period_source:
        from assemble_multi_roi import stitch_regions, graph_summary
        full_source = Path(args.full_period_source).resolve()
        source_hash = hashlib.sha256(full_source.read_bytes()).hexdigest()
        full_before = gpd.read_file(full_source, layer='centerlines')
        # Refuse to silently patch a different experiment or period.
        a = union_all(full_before.geometry).intersection(roi)
        b = union_all(before.geometry).intersection(roi)
        if a.difference(b.buffer(.002)).length+b.difference(a.buffer(.002)).length > .02:
            raise ValueError('Full-period input does not match the ROI source')
        clipped = after.copy(); clipped.geometry = clipped.geometry.intersection(roi)
        clipped = clipped.explode(index_parts=False)
        clipped = clipped[(~clipped.is_empty) & (clipped.geom_type=='LineString')]
        full_after = stitch_regions(full_before,[clipped],[roi])
        old_stats,new_stats = graph_summary(full_before),graph_summary(full_after)
        a = union_all(full_before.geometry).difference(roi)
        b = union_all(full_after.geometry).difference(roi)
        delta = a.difference(b.buffer(.001)).length+b.difference(a.buffer(.001)).length
        assert delta < .02 and new_stats['components'] <= old_stats['components']
        assert new_stats['degree_1'] <= old_stats['degree_1'] and new_stats['invalid_features'] == 0
        assert new_stats['self_intersecting_features'] <= old_stats['self_intersecting_features']
        full_output = output/'full_period'; full_output.mkdir()
        full_after.to_file(full_output/'final_centerlines.gpkg',layer='centerlines',driver='GPKG')
        write_json(full_output/'summary.json',dict(source=str(full_source),source_sha256=source_hash,
            before=old_stats,after=new_stats,outside_roi_changed_m=delta,
            coverage='Previous complete period plus this ROI edit only'))
        assert source_hash == hashlib.sha256(full_source.read_bytes()).hexdigest()
    print('ACCEPTED',sum(r['applied'] for r in audit),'OUTPUT',output,flush=True)
    return output


if __name__ == '__main__':
    main()
