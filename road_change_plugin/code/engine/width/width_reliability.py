"""Width failure boundaries and diagnostics. Coordinates are never fitted to an image."""
from collections import Counter
from pathlib import Path
import json
import traceback

import cv2
import numpy as np
from shapely.errors import GEOSException
from shapely.geometry import LineString

VERSION = 1
# I/O, missing dependencies, MemoryError and cancellation deliberately propagate.
ROAD_ERRORS = (ValueError, IndexError, ArithmeticError, GEOSException, cv2.error)


class InputContractError(RuntimeError):
    """The coordinate system or raster contract cannot be interpreted safely."""


def project_default_width():
    import os
    root = os.environ.get('SAMROAD_PROJECT_ROOT')
    config = Path(root)/'project_config.json' if root else None
    value = json.loads(config.read_text(encoding='utf-8-sig')).get('width_default_m', 6.) if config and config.is_file() else 6.
    if not positive(value):
        raise InputContractError('width_default_m must be finite and positive, in metres')
    return float(value)


def implementation_identity():
    import hashlib
    root=Path(__file__).parent
    paths=[root/name for name in ('width_reliability.py','chain_width_calculator.py','molra_centerline_width.py',
        'raw_image_backend.py','raw_boundary_core.py','raw_width_reconstruction.py','raw_width_records.py',
        'raw_feature_cache.py','raw_road_surfaces.py')]
    paths += [root.parent/'fast_pipeline.py']
    digest=hashlib.sha256()
    for path in paths:digest.update(path.read_bytes())
    return [VERSION,digest.hexdigest(),project_default_width()]


def positive(value):
    try:
        return bool(np.isfinite(float(value)) and float(value) > 0)
    except (TypeError, ValueError):
        return False


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def issue(road_id, tile_id, stage, error, *, geometry=None, fallback='pending'):
    # Native numerical libraries sometimes report allocation failure as ValueError/cv2.error.
    if getattr(error,'code',None)==-4 or any(token in str(error).lower() for token in
        ('out of memory','failed to allocate','bad_alloc','insufficient memory')):
        raise error
    bounds = list(geometry.bounds) if geometry is not None and not geometry.is_empty else None
    row = dict(road_id=str(road_id), tile_id=str(tile_id), stage=stage,
               exception_type=type(error).__name__, reason=str(error),
               traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)),
               fallback=fallback, bounds=bounds if bounds is None or np.isfinite(bounds).all() else None)
    print('[Width warning] ' + json.dumps(row, ensure_ascii=False), flush=True)
    return row


def clean_line(line):
    """Remove consecutive duplicates only; preserve bends, crossings and loops."""
    if line is None or line.is_empty or line.geom_type != 'LineString':
        raise ValueError('geometry_not_a_nonempty_linestring')
    xy = np.asarray(line.coords, dtype=float)
    if xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('nonfinite_or_non_2d_coordinates')
    xy = xy[np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)]]
    if len(xy) < 2:
        raise ValueError('zero_length_geometry')
    result = LineString(xy)
    if not positive(result.length):
        raise ValueError('invalid_line_length')
    return result


def clean_frame(frame, diagnostics):
    if frame.crs is None:
        raise InputContractError('Centerlines require an explicit CRS')
    rows = []
    for source_id, row in frame.iterrows():
        geom = row.geometry
        parts = list(geom.geoms) if geom is not None and geom.geom_type == 'MultiLineString' else [geom]
        for part_id, part in enumerate(parts):
            try:
                line = clean_line(part)
            except ROAD_ERRORS as exc:
                diagnostics.append(issue(source_id, row.get('tile', ''), 'geometry', exc,
                                         geometry=part, fallback='invalid_geometry_excluded'))
                continue
            record = row.to_dict()
            record.update(geometry=line, input_id=str(source_id), input_part=part_id)
            rows.append(record)
    import geopandas as gpd
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=frame.crs) if rows else frame.iloc[:0].copy()


def validate_mask(mask, shape=None):
    array = np.asarray(mask)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise InputContractError('Mask must be a nonempty finite 2D array')
    if shape is not None and array.shape != tuple(shape):
        raise InputContractError(f'Mask/image shape mismatch: {array.shape} != {shape}')
    return (array > 0).astype(np.uint8)


def validate_graph(nodes, edges, diagnostics, tile_id=''):
    nodes=np.asarray(nodes,dtype=float)
    edges=np.asarray(edges)
    if nodes.ndim!=2 or nodes.shape[1]!=2 or edges.ndim!=2 or edges.shape[1]!=2:
        raise InputContractError('Road graph must contain N x 2 nodes and E x 2 indices')
    kept=[];ids=[]
    for edge_id,pair in enumerate(edges):
        try:
            if not np.isfinite(pair).all() or not np.equal(pair,np.floor(pair)).all():
                raise ValueError('noninteger_graph_index')
            a,b=map(int,pair)
            if min(a,b)<0 or max(a,b)>=len(nodes):raise IndexError('graph_index_out_of_range')
            if not np.isfinite(nodes[[a,b]]).all():raise ValueError('nonfinite_graph_coordinates')
            if np.array_equal(nodes[a],nodes[b]):raise ValueError('zero_length_graph_edge')
            kept.append((a,b));ids.append(edge_id)
        except ROAD_ERRORS as exc:
            diagnostics.append(issue(edge_id,tile_id,'graph_preflight',exc,fallback='invalid_geometry_excluded'))
    return np.asarray(kept,dtype=np.int64).reshape(-1,2),ids


def surface_width_estimates(lines, surfaces):
    """Metre distance-to-boundary evidence at actual, covered centerline stations."""
    from shapely import STRtree
    polygons=[g for g in surfaces.geometry if g is not None and not g.is_empty and g.geom_type in ('Polygon','MultiPolygon')]
    tree=STRtree(polygons)
    values={}
    for key,line in lines.items():
        estimates=[]
        for station in np.linspace(0,line.length,9)[1:-1]:
            point=line.interpolate(station)
            for index in tree.query(point,predicate='within'):
                width=2*point.distance(polygons[int(index)].boundary)
                if positive(width):estimates.append(width)
        if estimates:values[key]=estimates
    return values


def quality_summary(observations, diagnostics, *, total_input=None):
    records = []
    for road_id, group in observations.groupby('parent_id', sort=False):
        grades = group.quality_grade.fillna('C')
        grade = 'C' if grades.eq('C').any() else 'B' if grades.eq('B').any() else 'A'
        records.append(dict(road_id=str(road_id), tile_id=str(group.iloc[0].get('tile_id', '')),
                            bounds=[float(x) for x in group.total_bounds], quality_grade=grade,
                            length_m=float(group.length_m.sum()),
                            width_sources=sorted(set(group.width_source.astype(str))),
                            fallback_reasons=sorted(set(group.outlier_reason.fillna('').astype(str)) - {''})))
    sources = Counter(source for row in records for source in row['width_sources'])
    fallback_count = sum(any(s not in {'measured', 'smoothed', 'enhanced_molra', 'chain_hybrid_v2'}
                             for s in row['width_sources']) for row in records)
    low = sum(row['quality_grade'] == 'C' for row in records)
    return dict(quality_version=VERSION, status='completed_with_warnings' if diagnostics or low or fallback_count else 'completed',
                total_input=total_input, road_count=len(records),bounds_crs=str(observations.crs),width_unit='m',
                normal_count=len(records)-fallback_count, fallback_count=fallback_count,
                fallback_ratio=fallback_count/max(len(records), 1), source_counts=dict(sources),
                exception_count=len(diagnostics), exception_counts=dict(Counter(d['exception_type'] for d in diagnostics)),
                exceptions=diagnostics, low_confidence_road_count=low,
                affects_width_change_reliability=bool(low),
                quality_counts=dict(Counter(r['quality_grade'] for r in records)),
                quality_length_m={g: sum(r['length_m'] for r in records if r['quality_grade'] == g) for g in ('A','B','C')},
                roads=records)
