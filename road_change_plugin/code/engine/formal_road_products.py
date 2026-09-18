"""One formal road geometry contract, shared by period export and event rebuilds.

No extraction or width measurement happens here. Saved observations supply
profiles; the same surviving graph owns public axes, widths and surfaces.
Module-level imports stay lightweight for pipeline cache readiness checks.
"""
import json
from pathlib import Path

FORMAL_ROAD_REVISION = 1


def implementation_signature():
    from .product_cache import signature
    return signature([Path(__file__).with_name(name) for name in (
        'formal_road_products.py', 'canonical_road_surface.py', 'road_axis_quality.py',
        'road_surface_quality.py', 'auto_change_geometry.py', 'auto_change_assembly.py',
        'gt_road_geometry.py', 'road_connection_evidence.py', 'fast_pipeline.py')]) + [FORMAL_ROAD_REVISION]


def formal_metadata():
    return dict(formal_road_revision=FORMAL_ROAD_REVISION,
                formal_road_implementation=implementation_signature(),
                axis_quality_revision=1, surface_quality_revision=5, regular_surface=True)


def is_formal_result(result):
    return (result.get('formal_road_revision') == FORMAL_ROAD_REVISION and
            result.get('formal_road_implementation') == implementation_signature())


def formal_products_current(directory):
    """Reject legacy/interrupted exports, changed inputs and stale renderer code."""
    from .product_cache import read_completed, signature
    marker = Path(directory) / 'fast_export_cache.json'
    try:
        saved = json.loads(marker.read_text(encoding='utf8'))
        return (is_formal_result(saved['result']) and
                saved['inputs'][0] == signature([Path(directory)/'regional_products.gpkg']) and
                saved['inputs'][3] == signature([v[0] for v in saved['inputs'][3]]) and
                read_completed(marker, saved['inputs']) is not None)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def reconstruct_frames(centers, widths, *, evidence=None, observations=(), directory=None):
    """Rebuild metric frames from existing centerlines and measured profiles.

    Width evidence grades/statistics are copied, never upgraded by geometric
    regularization. Internal measurement observations remain in regional products.
    """
    import hashlib
    import geopandas as gpd
    import numpy as np
    from shapely.ops import substring
    from .auto_change_geometry import FinalWidths, _direction
    from .gt_road_geometry import road_profile
    from .road_axis_quality import repair_network_axes
    from .canonical_road_surface import build_road_geometry
    from .product_cache import read_completed, write_completed, signature

    crs = centers.crs
    if crs is None or crs.is_geographic:
        raise ValueError('Formal roads require a projected metric CRS')
    rows = centers.to_dict('records')
    for row in rows:
        if str(row.get('track_id', '')).lower() in ('none','nan','<na>'):
            row['track_id'] = ''
    axes = list(centers.geometry)
    fallback = []
    for row in rows:
        value = row.get('width_m', row.get('width_map', 6.))
        try: value = float(value)
        except (ValueError, TypeError): value = 6.
        fallback.append(value if np.isfinite(value) and value > 0 else 6.)
    repaired, axis_audit = repair_network_axes(axes, fallback, evidence)
    repairs = {r['feature']: r for r in axis_audit}
    sampler = FinalWidths(widths)
    profile_inputs = signature([Path(__file__).with_name('auto_change_geometry.py'),
                                Path(__file__).with_name('road_surface_quality.py')]) + [str(crs)]
    cache_path = Path(directory)/'width_profile_cache.json' if directory else None
    saved = (read_completed(cache_path, profile_inputs) or {}) if cache_path else {}
    current = {}; profiles = []; quality = []
    for i, (row, old, axis, width) in enumerate(zip(rows, axes, repaired, fallback)):
        if row.get('track_id'):
            ss, vv = road_profile(row, old, width)
            qq = np.ones(len(ss))
        else:
            ids = sorted(sampler.tree.query(old, predicate='dwithin', distance=.75))
            dependencies = [(sampler.frame.geometry.iloc[int(k)].wkb_hex,
                             [str(sampler.frame.iloc[int(k)].get(f)) for f in
                              ('width_m','quality_grade','width_quality','valid_ratio','width_std')]) for k in ids]
            key = hashlib.sha256(json.dumps([old.wkb_hex,width,dependencies]).encode()).hexdigest()
            ss, vv, qq = (map(np.asarray, saved[key]) if key in saved else sampler.profile_quality(old, width))
            current[key] = [ss.tolist(), vv.tolist(), qq.tolist()]
        if i in repairs:
            mapping = repairs[i]['station_map']
            ss = np.interp(ss, mapping['before'], mapping['after'])
            row['axis_fix'] = ','.join(sorted({r['method'] for p in repairs[i]['parts'] for r in p.get('repairs',[])}))
        row['geometry'] = axis
        profiles.append((axis, ss, vv)); quality.append(qq)
    if cache_path:
        write_completed(cache_path, profile_inputs, current, [])
    surface, chains, audit = build_road_geometry(profiles, quality, metadata=rows,
                                                evidence=evidence, observations=observations)
    center_rows = []; width_rows = []
    for chain_id, chain in enumerate(chains):
        sources = sorted(chain.sources)
        owner = next((i for i in sources if rows[i].get('track_id')), sources[0])
        row = {k:v for k,v in rows[owner].items() if k not in ('geometry','_original_row','gt_width_profile')}
        row.update(segment_id=chain_id, width_m=float(np.median(chain.widths)),
                   length_m=chain.axis.length, geometry=chain.axis,
                   source_ids=','.join(map(str,sources)))
        if row.get('track_id'):
            row['gt_width_profile'] = json.dumps(dict(axis=chain.axis.wkt,
                stations=(chain.stations/chain.axis.length).tolist(), widths=chain.widths.tolist(),
                reference_width=row['width_m']))
        center_rows.append(row)
        for a,b,wa,wb in zip(chain.stations[:-1],chain.stations[1:],chain.widths[:-1],chain.widths[1:]):
            if b-a <= 1e-8: continue
            point = chain.axis.interpolate((a+b)/2)
            source = min(sources, key=lambda i: profiles[i][0].distance(point))
            fixed_axis = profiles[source][0]
            station = fixed_axis.project(point)
            if source in repairs:
                mapping = repairs[source]['station_map']
                station = float(np.interp(station, mapping['after'], mapping['before']))
            old = axes[source]; probe = old.interpolate(station)
            candidates = []
            for k in sampler.tree.query(probe, predicate='dwithin', distance=.75):
                observed = sampler.frame.iloc[int(k)]; line = observed.geometry
                if line.geom_type != 'LineString' or not line.length: continue
                cosine = abs(float(_direction(old,station) @ _direction(line,line.project(probe))))
                if cosine >= .95: candidates.append((line.distance(probe),-cosine,int(k)))
            evidence_row = sampler.frame.iloc[min(candidates)[2]].to_dict() if candidates else {}
            local = substring(chain.axis,a,b)
            # A missing observation stays missing. A chain owner's grade must
            # never leak into an unsupported station of the merged chain.
            width_rows.append({**evidence_row,
                'segment_id':chain_id, 'source_ids':row['source_ids'], 'track_id':row.get('track_id',''),
                'state_src':row.get('state_src',''), 'axis_fix':row.get('axis_fix',''),
                'measured_w':evidence_row.get('width_m'), 'width_m':float((wa+wb)/2),
                'geometry':local, 'length_m':local.length})
    def frame(items):
        return gpd.GeoDataFrame(items,geometry='geometry',crs=crs) if items else gpd.GeoDataFrame(geometry=[],crs=crs)
    polygons = [surface] if surface.geom_type == 'Polygon' else list(surface.geoms)
    surfaces = frame([dict(geometry=p) for p in polygons if not p.is_empty])
    frames = dict(centerlines=frame(center_rows), width_segments=frame(width_rows),
                  surfaces=surfaces, corridors=surfaces.copy())
    audit['summary']['formal_width_segment_count']=len(width_rows)
    return frames, axis_audit, audit
