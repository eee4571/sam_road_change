"""Region-wide network recovery shared by production exporters.

Imports used by readiness checks stay lightweight; geometry work is loaded only
when exporting. Filtering is never performed separately for individual tiles.
"""
from pathlib import Path
import json

NETWORK_CONNECTION_VERSION = 1
NETWORK_REPORT = 'road_network_report.json'


def network_products_current(directory):
    try:
        report = json.loads((Path(directory) / NETWORK_REPORT).read_text(encoding='utf-8'))
        return (report.get('version') == NETWORK_CONNECTION_VERSION
                and report.get('status') == 'completed'
                and (Path(directory) / 'road_network_audit.gpkg').is_file())
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def recover_centerline_frame(frame, surfaces=None, *, authoritative=False):
    """Recover one period/region; keep original CRS and trace every source row."""
    import geopandas as gpd
    import numpy as np
    from pyproj import CRS
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from .road_geometry import _RegionalRoadSeed
    from .road_network_connection import connect_clean_road_seeds

    if frame.crs is None:
        raise ValueError('Network recovery requires a CRS')
    original = frame.explode(index_parts=False).reset_index(drop=True)
    audits = {'connection_input': original.copy()}
    if authoritative:
        return frame.copy(), {'policy': 'manual_authoritative', 'connection_output_count': len(frame)}, audits
    print(f'[Road network] Recovering {len(original)} regional centerlines',flush=True)
    definition = CRS.from_user_input(frame.crs)
    projected = original.estimate_utm_crs() if definition.is_geographic and not original.empty else frame.crs
    if projected is None:
        raise ValueError('Could not determine projected CRS for network recovery')
    metric = original.to_crs(projected)
    unit = float(CRS.from_user_input(projected).axis_info[0].unit_conversion_factor)
    if not metric.empty and not metric.geom_type.eq('LineString').all():
        raise ValueError('Network recovery accepts only line geometries')
    width_field = next((key for key in ('width_m','width_map') if key in metric), None)
    widths = np.asarray(metric[width_field],dtype=float) if width_field else np.full(len(metric),6.)
    widths = np.where(np.isfinite(widths) & (widths>0),widths,6.)
    seeds = [_RegionalRoadSeed(np.asarray(row.geometry.coords),float(widths[i]),(i,))
             for i,row in metric.iterrows()]
    surface = unary_union(surfaces.to_crs(projected).geometry) if surfaces is not None and not surfaces.empty else None
    connected, stats, audit = connect_clean_road_seeds(seeds,surface,unit,keep_main_component=True)
    rows = []
    for index,road in enumerate(connected):
        source = max(road.source_ids,key=lambda i: metric.geometry.iloc[i].length)
        row = original.iloc[source].drop(labels=original.geometry.name).to_dict()
        row.update(edge_id=index,src_ids=','.join(map(str,road.source_ids)),width_m=road.width_m,
                   geometry=LineString(road.points))
        if 'global_id' in row:
            row['global_id'] = index
        if 'width_map' in row:
            row['width_map'] = road.width_m
        rows.append(row)
    result = (gpd.GeoDataFrame(rows,geometry='geometry',crs=projected).to_crs(frame.crs)
              if rows else original.iloc[:0].copy())
    categories = {'accepted':'added_connections','replaced':'replaced_corridor_segments',
                  'isolated_removed':'removed_isolated_roads'}
    for status,layer in categories.items():
        records = [{**row,'first_sources':json.dumps(row['first_sources']),
                    'second_sources':json.dumps(row['second_sources'])}
                   for row in audit if row['status']==status]
        if records:
            audits[layer] = gpd.GeoDataFrame(records,geometry='geometry',crs=projected).to_crs(frame.crs)
    rejected = [{**row,'first_sources':json.dumps(row['first_sources']),
                 'second_sources':json.dumps(row['second_sources'])}
                for row in audit if row['status'] not in categories]
    if rejected:
        audits['rejected_connections'] = gpd.GeoDataFrame(rejected,geometry='geometry',crs=projected).to_crs(frame.crs)
    stats.update(policy='recovered_main_component',metric_crs=str(projected),
                 connection_total_added_count=sum(row['status']=='accepted' for row in audit),
                 width_policy='existing_observation_widths_inherited_on_connections')
    print(f"[Road network] Retained {len(result)} lines; removed "
          f"{stats['connection_isolated_removed_count']} isolated lines",flush=True)
    return result, stats, audits


def write_network_report(directory, stats, audits):
    """Write the completion marker only after product export has succeeded."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    target = directory / 'road_network_audit.gpkg'
    target.unlink(missing_ok=True)
    for index,(layer,frame) in enumerate(audits.items()):
        frame.to_file(target,layer=layer,driver='GPKG',mode='w' if index==0 else 'a')
    report = {**stats,'version':NETWORK_CONNECTION_VERSION,'status':'completed',
              'audit_gpkg':str(target.resolve()),
              'source_ids':'zero-based rows of connection_input in road_network_audit.gpkg'}
    temporary = directory / (NETWORK_REPORT+'.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(directory / NETWORK_REPORT)
    return report


def rebuild_network_width_products(centerlines, measured=None, source_tolerance=2.0,
                                   *, connection_input=None):
    """Use metre coordinates for segmentation and buffering, including lon/lat inputs."""
    from pyproj import CRS
    from shapely.ops import unary_union
    from .width.road_pair_matcher import build_width_segments, build_corridors
    definition = CRS.from_user_input(centerlines.crs)
    projected = centerlines.crs
    if not centerlines.empty and (definition.is_geographic or
                                 abs(definition.axis_info[0].unit_conversion_factor-1)>1e-9):
        projected = centerlines.estimate_utm_crs()
        if projected is None:
            raise ValueError('Could not determine metric CRS for width products')
    metric = centerlines.to_crs(projected)
    observations = measured.to_crs(projected) if measured is not None else None
    segments = build_width_segments(metric,observations,source_tolerance=source_tolerance)
    if connection_input is not None and not segments.empty:
        observed_area = unary_union(connection_input.to_crs(projected).geometry).buffer(.25)
        inferred = segments.geometry.map(lambda line: line.difference(observed_area).length > .05*line.length)
        segments.loc[inferred, 'quality_grade'] = 'C'
        segments.loc[inferred, 'width_quality'] = 'C'
        segments.loc[inferred, 'line_source'] = 'connector'
        segments.loc[inferred, 'valid_ratio'] = 0.
        segments.loc[inferred, 'qa_state'] = 'review'
        segments.loc[inferred, 'qa_reason'] = 'connection_width_inherited'
    corridors = build_corridors(segments)
    return segments.to_crs(centerlines.crs), corridors.to_crs(centerlines.crs)
