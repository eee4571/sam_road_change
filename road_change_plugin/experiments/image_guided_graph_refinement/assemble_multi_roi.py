"""Assemble non-overlapping independently tested ROIs into one full-period product."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely import union_all, set_precision, snap
from shapely.geometry import box, LineString

from bootstrap import ROOT
from data import discover, fingerprint, load_lines, read_json
from run_multi_roi import data_fingerprints
from graph import build
from repair_artifacts import compare, geometry_products
from evidence import ImageGrid
from report import write_json


def graph_summary(frame):
    edges = build(frame)
    graph = nx.MultiGraph()
    for edge in edges:
        graph.add_edge(*edge.nodes)
    return dict(components=nx.number_connected_components(graph),
                degree_1=sum(d == 1 for _, d in graph.degree()),
                junctions=sum(d >= 3 for _, d in graph.degree()),
                total_length_m=float(frame.length.sum()),
                invalid_features=int((~frame.is_valid).sum()),
                self_intersecting_features=int(sum(not g.is_simple for g in frame.geometry)))


def stitch_regions(before, regions, windows):
    """Canonical millimetre interior nodes, then pin exact unmodified seam contacts.

    Graph reconstruction rounds coordinates; clipping that result separately from
    original vectors can leave micron gaps. Only interior vertices are rounded;
    original outside geometry and its boundary contact coordinates are immutable.
    """
    domain = union_all(windows)
    outside = before.copy()
    outside.geometry = outside.geometry.difference(domain)
    outside = outside.explode(index_parts=False)
    outside = outside[(~outside.is_empty) & (outside.geom_type == 'LineString')]
    from engine.canonical_road_surface import _level
    levels = before.apply(_level, axis=1)
    networks = {level: union_all(before[levels == level].geometry) for level in set(levels)}
    stitched = []
    for region, window in zip(regions, windows):
        region = region.copy()
        seams = {level: network.intersection(window.boundary) for level, network in networks.items()}
        for i, row in region.iterrows():
            level = _level(row)
            # Includes shared ROI boundaries; both independent windows use the
            # same original contacts rather than different rounded seam nodes.
            contacts = seams[level]
            rounded = set_precision(row.geometry, .001)
            if rounded.is_empty:
                region.at[i, 'geometry'] = rounded
                continue
            # Drop the GEOS precision model before snapping original contacts;
            # otherwise union may re-round the newly pinned seam coordinates.
            line = LineString(rounded.coords)
            region.at[i, 'geometry'] = snap(line, contacts, .003)
        region = region[~region.is_empty]
        stitched.append(region)
    return gpd.GeoDataFrame(pd.concat([outside]+stitched, ignore_index=True), crs=before.crs)


def assemble(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Assembly may only write inside experimental outputs')
    manifest = read_json(folder/'inputs.json')
    summaries = read_json(folder/'summary.json')
    if len(summaries) != len(read_json(folder/'selection.json')['regions']):
        raise ValueError('Not all selected ROIs have finished')
    inputs = discover(manifest['project'], manifest['period'], manifest['area'], record=manifest['selected_period_record'])
    if data_fingerprints(fingerprint(inputs), inputs['assets']['manifest']) != data_fingerprints(manifest['fingerprints'], inputs['assets']['manifest']):
        raise ValueError('Actual source geometry/raster/cache changed; cannot assemble')
    windows = [box(*s['bbox_m']) for s in summaries]
    if any(a.intersection(b).area > .001 for i, a in enumerate(windows) for b in windows[:i]):
        raise ValueError('Overlapping independent ROI edits require conflict resolution; refusing last-writer overwrite')
    before = load_lines(inputs)
    domain = union_all(windows)
    regions = [gpd.read_file(folder/s['id']/'final_centerlines.gpkg', layer='roi') for s in summaries]
    after = stitch_regions(before, regions, windows)
    # The full graph check uses actual connected geometry, not a sum of cropped
    # component counts, which would double-count cut roads.
    old, new = graph_summary(before), graph_summary(after)
    geometry_before = union_all(before.geometry).difference(domain)
    geometry_after = union_all(after.geometry).difference(domain)
    outside_change = geometry_before.difference(geometry_after.buffer(.001)).length+geometry_after.difference(geometry_before.buffer(.001)).length
    validation = dict(before=old, after=new, outside_roi_changed_m=outside_change,
                      coverage='Full-period export; only the listed ROI interiors were edited',
                      roi_ids=[s['id'] for s in summaries], inputs_unchanged=True,
                      interior_precision_m=.001, seam_snap_tolerance_m=.003)
    write_json(folder/'assembled_summary.json', validation)
    if (outside_change > .02 or new['components'] > old['components'] or new['invalid_features']
            or new['self_intersecting_features'] > old['self_intersecting_features']):
        raise ValueError('Assembly topology/outside validation failed; inspect assembled_summary.json')
    before.to_file(folder/'baseline_centerlines.gpkg', layer='centerlines', driver='GPKG')
    after.to_file(folder/'final_centerlines.gpkg', layer='centerlines', driver='GPKG')
    audit = [row for summary in summaries for row in read_json(folder/summary['id']/'audit.json')]
    validation['geometry_changes'] = geometry_products(folder, before, after, audit)
    write_json(folder/'assembled_summary.json', validation)
    import hashlib
    source = Path(__file__).read_bytes()
    (folder/'source_snapshot'/Path(__file__).name).write_bytes(source)
    write_json(folder/'assembly_provenance.json', dict(source_file=Path(__file__).name, sha256=hashlib.sha256(source).hexdigest()))
    extent = box(*before.total_bounds).buffer(10)
    evidence = SimpleNamespace(image=ImageGrid(inputs['images'], before.crs, extent, 3.))
    compare(folder/'whole_period_before_after.png', before, after, evidence, extent,
            'Full-period export / only the selected ROI interiors are modified')
    print(validation, flush=True)
    return after


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    assemble(parser.parse_args().run_directory)
