from __future__ import annotations

"""Run the connection stage on published centerlines, in a new output directory."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.road_geometry import _RegionalRoadSeed
from engine.road_network_connection import connect_clean_road_seeds


def render_comparison(before, after, audit, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    accepted = [row for row in audit if row['status'] == 'accepted']
    # Historical accepted candidates may belong to subsequently removed islands.
    # Only draw their surviving geometry over the final network.
    from engine.road_network_connection import _parts
    final_area = unary_union(after).buffer(.001)
    accepted = [{**row, 'geometry': part} for row in accepted
                for part in _parts(row['geometry'].intersection(final_area), 'LineString')
                if part.length > .01]
    details = sorted(accepted, key=lambda row: row['distance_m'], reverse=True)[:4]

    def draw(axis, title, connections=False, extent=None):
        axis.add_collection(LineCollection([np.asarray(g.coords) for g in (after if connections else before)], colors='#47617a', linewidths=0.65))
        if connections:
            axis.add_collection(LineCollection([np.asarray(row['geometry'].coords) for row in accepted], colors='#e6550d', linewidths=1.8))
        axis.autoscale()
        if extent is not None:
            x0, y0, x1, y1 = extent
            margin = max(x1-x0, y1-y0, 25) * 0.45
            axis.set_xlim(x0-margin, x1+margin)
            axis.set_ylim(y0-margin, y1+margin)
        axis.set_aspect('equal')
        axis.set_title(title)
        axis.ticklabel_format(useOffset=False, style='plain')
        axis.tick_params(labelsize=7)
    draw(axes.flat[0], f'Clean input: {len(before)} features')
    draw(axes.flat[1], f'Connected: {len(after)} features; {len(accepted)} orange links', True)
    for axis, row in zip(list(axes.flat)[2:], details):
        draw(axis, f"Added gap: {row['distance_m']:.1f} m; surface support {row['surface_support']:.0%}", True, row['geometry'].bounds)
    for axis in list(axes.flat)[2+len(details):]:
        axis.set_visible(False)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def render_network(metric_frame, output):
    """Expose actual connected components, including every remaining fragment."""
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from engine.road_network_connection import _graph, _key

    roads = [_RegionalRoadSeed(np.asarray(row.geometry.coords),row.width_m,(i,))
             for i,row in metric_frame.iterrows()]
    graph = _graph(roads)
    components = sorted(nx.connected_components(graph),key=lambda nodes:sum(
        d['weight'] for _,_,d in graph.subgraph(nodes).edges(data=True)),reverse=True)
    labels = {node:index for index,nodes in enumerate(components) for node in nodes}
    component_ids = [labels.get(_key(road.points[0]),-1) for road in roads]
    metric_frame['component'] = component_ids
    figure,axes = plt.subplots(1,2,figsize=(15,13),constrained_layout=True)
    segments = [road.points for road in roads]
    axes[0].add_collection(LineCollection(segments,colors='#34566d',linewidths=.85))
    axes[1].add_collection(LineCollection(segments,colors=[
        '#34566d' if label==0 else '#d4550a' for label in component_ids],linewidths=.85))
    axes[0].set_title('Recovered road network')
    axes[1].set_title('Blue: main component | Orange: remaining disconnected roads')
    for axis in axes:
        axis.autoscale()
        axis.set_aspect('equal')
        axis.set_axis_off()
    figure.savefig(output/'network_overview.png',dpi=180)
    plt.close(figure)
    return component_ids


def run(centerlines: Path, output: Path, surface: Path | None = None, max_gap_m=300.0):
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    frame = gpd.read_file(centerlines)
    if frame.crs is None or frame.empty:
        raise ValueError('Input must have a CRS and contain centerlines')
    if not frame.geom_type.isin(['LineString', 'MultiLineString']).all():
        raise ValueError('Input must contain only line geometries')
    frame = frame.explode(index_parts=False).reset_index(drop=True)
    definition = CRS.from_user_input(frame.crs)
    metric_crs = frame.estimate_utm_crs() if definition.is_geographic else frame.crs
    if metric_crs is None:
        raise ValueError('Could not determine a metric CRS')
    metric_definition = CRS.from_user_input(metric_crs)
    unit = float(metric_definition.axis_info[0].unit_conversion_factor)
    metric = frame.to_crs(metric_crs)
    print(f'[Connect] loaded {len(metric)} clean lines', flush=True)
    surface_geometry = None
    if surface is not None:
        surface_frame = gpd.read_file(surface)
        if surface_frame.crs is None:
            raise ValueError('Surface input must have a CRS')
        surface_geometry = unary_union(surface_frame.to_crs(metric_crs).geometry)
    widths = np.asarray(metric['width_m'], dtype=float) if 'width_m' in metric else np.full(len(metric), 6.0)
    # Unknown widths affect only the lateral tolerance, never geometry generation.
    widths = np.where(np.isfinite(widths) & (widths > 0), widths, 6.0)
    seeds = [_RegionalRoadSeed(np.asarray(row.geometry.coords), float(widths[index]), (index,), 'clean_input')
             for index, row in metric.iterrows()]
    connected, stats, audit = connect_clean_road_seeds(seeds, surface_geometry, unit, max_gap_m=max_gap_m, keep_main_component=True)
    stats['connection_total_added_count'] = stats['connection_added_count'] + stats.get('connection_corridor_bridge_count',0)
    stats['connection_total_added_length_m'] = sum(row['geometry'].length*unit for row in audit if row['status']=='accepted')
    print(f"[Connect] accepted {stats['connection_total_added_count']} connections; checking preservation", flush=True)
    rows = []
    for index, road in enumerate(connected):
        representative = max(road.source_ids, key=lambda source_id: metric.geometry.iloc[source_id].length)
        row = frame.iloc[representative].drop(labels=['geometry']).to_dict()
        row.update(edge_id=index, src_ids=','.join(map(str, road.source_ids)), width_m=road.width_m,
                   geometry=LineString(road.points))
        rows.append(row)
    result_metric = gpd.GeoDataFrame(rows, crs=metric_crs)
    result = result_metric.to_crs(frame.crs)
    preserved_area = unary_union(result_metric.geometry).buffer(0.001 / unit)
    missing_length = sum(line.difference(preserved_area).length for line in metric.geometry) * unit
    # Geometry changes must be confined to declared endpoint tails and the
    # explicitly audited paired-track reconstruction segments.
    if missing_length > stats['connection_smoothed_tail_length_m'] + stats.get('connection_corridor_replaced_length_m',0) + stats['connection_isolated_removed_length_m'] + 0.01:
        raise RuntimeError(f'Original geometry preservation failed: {missing_length:.9f} m missing')
    from shapely import wkt
    tail_geometries = [wkt.loads(row[key]) for row in audit if row['status']=='accepted'
                       for key in ('first_trim_wkt','second_trim_wkt') if row.get(key)]
    tail_geometries.extend(row['geometry'] for row in audit if row['status'] in ('replaced','isolated_removed'))
    allowed_area = preserved_area.union(unary_union(tail_geometries).buffer(0.001/unit))
    body_missing = sum(line.difference(allowed_area).length for line in metric.geometry)*unit
    if body_missing>0.01:
        missing = [line.difference(allowed_area) for line in metric.geometry]
        detail = sorted((g for g in missing if g.length>0),key=lambda g:-g.length)[:5]
        raise RuntimeError(f'Geometry changed outside declared replacement geometry: {body_missing:.6f} m; '+str([(g.length,g.bounds) for g in detail]))
    output.mkdir(parents=True)
    result['component'] = render_network(result_metric,output)
    gpkg = output / 'connected_network.gpkg'
    result.to_file(gpkg, layer='centerlines', driver='GPKG')
    frame.to_file(gpkg, layer='original_centerlines', driver='GPKG', mode='a')
    shape_result = result.copy()
    shape_result['src_ids'] = shape_result['src_ids'].map(lambda value: value if len(value)<240 else value[:220]+';see_GPKG')
    shape_result.to_file(output / 'road_centerlines.shp', encoding='UTF-8')
    for status, layer in (('accepted', 'added_connections'), ('replaced','replaced_corridor_segments'), ('isolated_removed','removed_isolated_roads'), ('rejected', 'rejected_connections')):
        records = []
        for row in audit:
            category = row['status'] if row['status'] in ('accepted','replaced','isolated_removed') else 'rejected'
            if category != status:
                continue
            records.append({**row, 'first_sources': json.dumps(row['first_sources']),
                            'second_sources': json.dumps(row['second_sources'])})
        if records:
            gpd.GeoDataFrame(records, crs=metric_crs).to_crs(frame.crs).to_file(gpkg, layer=layer, driver='GPKG', mode='a')
    stats.update({
        'input_centerlines': str(centerlines.resolve()),
        'input_surface': str(surface.resolve()) if surface else None,
        'output_gpkg': str(gpkg.resolve()), 'metric_crs': str(metric_crs),
        'maximum_search_gap_m': max_gap_m, 'original_missing_length_m': missing_length,
        'unexpected_body_change_m': body_missing,
        'rejection_reasons': dict(Counter(row['status'] for row in audit if row['status'] not in ('accepted','replaced','isolated_removed'))),
        'attributes': 'src_ids refer to zero-based original_centerlines rows; width_m is observed-length weighted; other attributes use the longest source feature',
        'crossing_policy': 'Planar noded road network; ordinary street blocks may close; bridge elevation is not inferred',
        'candidates': [{key: value for key, value in row.items() if key != 'geometry'} for row in audit],
    })
    (output / 'connection_report.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    render_comparison(list(metric.geometry), list(result_metric.geometry), audit, output / 'connection_comparison.png')
    print(json.dumps({key: value for key, value in stats.items() if key != 'candidates'}, ensure_ascii=False, indent=2))
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--centerlines', type=Path, required=True)
    parser.add_argument('--surface', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-gap-m', type=float, default=300.0)
    args = parser.parse_args()
    run(args.centerlines, args.output, args.surface, args.max_gap_m)
