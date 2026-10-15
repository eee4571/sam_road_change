"""Apply human rejection of truth-assisted changes to evaluation truth.

Complete rejected sources are excluded; partial rejections exclude only their
associated spatial support. Original truth files and unrelated roads stay intact.
"""
import json
from pathlib import Path

import geopandas as gpd
from shapely import from_wkt, make_valid, union_all


def build_truth_filter(original, decisions, audit_path, *, audit_layer='changes',
                       position_grid=None, position_tolerance_px=4.):
    audit_path = Path(audit_path)
    audit = gpd.read_file(audit_path, layer=audit_layer)
    if audit.crs is None:
        raise ValueError('变化来源记录缺少坐标系，无法更新复核评价')
    result = {'version': 2, 'audit': str(audit_path), 'crs': audit.crs.to_wkt(), 'objects': [],
              'position_tolerance_px': float(position_tolerance_px) if position_grid else None}
    if audit.empty:
        return result
    required = {'change_src', 'truth_id', 'change_typ'}
    if not required <= set(audit.columns):
        raise ValueError('变化来源记录不完整，无法更新复核评价')
    linked = audit.loc[audit.truth_id.notna() & audit.truth_id.astype(str).str.strip().ne('')]
    generated = linked.loc[linked.change_src.eq('GT_ASSISTED')]
    if generated.empty:
        return result
    if 'gt_type' not in generated:
        raise ValueError('变化来源记录缺少类型，无法更新复核评价')
    public = original.to_crs(audit.crs)
    spatial_index = public.sindex

    def expanded(geometry):
        if position_grid is None:
            return geometry
        from road_change_detection import _expand_by_pixels
        return _expand_by_pixels(geometry, audit.crs, position_grid, position_tolerance_px)

    groups = {key: group for key, group in linked.groupby('truth_id', sort=False)}
    for truth_id, generated_group in generated.groupby('truth_id', sort=False):
        group = groups[truth_id]
        complete = ('truth_object_complete' not in group or
                    group.truth_object_complete.fillna(False).astype(bool).all())
        fully_rejected = bool(complete)
        review_ids, rejected, protected = set(), [], []
        for row in group.itertuples():
            if row.geometry is None or row.geometry.is_empty:
                fully_rejected = False
                continue
            geometry = make_valid(row.geometry)
            hits = []
            for position in spatial_index.query(geometry, predicate='intersects'):
                feature = public.iloc[int(position)]
                if feature.change_typ != row.change_typ:
                    continue
                # New candidates carry explicit provenance. Spatial overlap
                # alone must not transfer an Auto decision to a GT object (or
                # to a different GT source). Old projects lack these fields.
                source = feature.get('change_src')
                if source in ('AUTO', 'GT_ASSISTED') and source != row.change_src:
                    continue
                if source == 'GT_ASSISTED' and feature.get('truth_id') != row.truth_id:
                    continue
                overlap = geometry.intersection(make_valid(feature.geometry))
                if overlap.area <= max(1e-8, geometry.area * 1e-10):
                    continue
                index = str(int(position))
                hits.append(index)
                if decisions[index] != 'reject':
                    protected.append(overlap)
                elif row.change_src == 'GT_ASSISTED':
                    rejected.append(overlap)
                    review_ids.add(index)
            if not hits:
                protected.append(geometry)
            if not hits or any(decisions[index] != 'reject' for index in hits):
                fully_rejected = False
        if not rejected:
            continue
        kinds = {str(value) for value in generated_group.gt_type
                 if str(value) in {'added', 'removed', 'widened', 'narrowed', 'width_changed'}}
        kinds = {'width_changed' if kind in {'widened', 'narrowed'} else kind for kind in kinds}
        if not kinds:
            continue
        item = dict(truth_id=str(truth_id), types=sorted(kinds),
                    review_ids=sorted(review_ids, key=int), scope='whole' if fully_rejected else 'partial')
        if not fully_rejected:
            mask = expanded(union_all(rejected)).difference(expanded(union_all(protected)))
            if mask.is_empty:
                continue
            item['excluded_geometry_wkt'] = make_valid(mask).wkt
        result['objects'].append(item)
    return result


def saved_truth_filter(review, audit_path, *, audit_layer='changes', position_grid=None):
    """Rebuild from saved decisions, including projects saved before partial support."""
    record_path = Path(review.get('record') or '')
    original_path = Path(review.get('original') or '')
    if not record_path.is_file() or not original_path.is_file():
        return review.get('truth_filter')
    record = json.loads(record_path.read_text(encoding='utf-8-sig'))
    decisions = record['decisions']
    if not any(value == 'reject' for value in decisions.values()):
        return None
    if not Path(audit_path).is_file():
        # Older unassisted outputs may have no origin audit at all.
        return review.get('truth_filter')
    return build_truth_filter(gpd.read_file(original_path), decisions, audit_path,
                              audit_layer=audit_layer, position_grid=position_grid)


def apply_truth_filter(truth, review_filter, truth_type_field):
    """Exclude identified reviewed support in memory without changing source files."""
    if not review_filter or not review_filter['objects']:
        return truth, {}
    from .fast_gt_reconciliation import _id
    from road_change_detection import _truth_type_field, _normalized_truth_change_type

    field = _truth_type_field(truth, truth_type_field)
    identities = {(item['truth_id'], kind): item for item in review_filter['objects'] for kind in item['types']}
    projected = truth.to_crs(review_filter['crs'])
    filtered = truth.copy()
    retained, excluded, partial = [], [], []
    for position, (_, row) in enumerate(projected.iterrows()):
        geometry = row.geometry
        kind = _normalized_truth_change_type(row[field], field, 'three') if field else None
        identity = _id(geometry, 'GT') if geometry is not None and not geometry.is_empty else ''
        item = identities.get((identity, kind))
        remaining = geometry
        if item is not None:
            if item.get('scope', 'whole') == 'whole':
                excluded.append(identity)
                continue
            remaining = make_valid(geometry).difference(from_wkt(item['excluded_geometry_wkt']))
            if remaining.is_empty:
                excluded.append(identity)
                continue
            if not remaining.equals(geometry):
                partial.append(identity)
                filtered.iat[position, filtered.columns.get_loc(filtered.geometry.name)] = gpd.GeoSeries(
                    [remaining], crs=projected.crs).to_crs(truth.crs).iloc[0]
        retained.append(position)
    filtered = filtered.iloc[retained].copy()
    metadata = {'original_feature_count': len(truth), 'effective_feature_count': len(filtered),
                'excluded_feature_count': len(excluded), 'excluded_truth_ids': sorted(set(excluded)),
                'partially_excluded_feature_count': len(partial), 'partially_excluded_truth_ids': sorted(set(partial)),
                'truth_filter_version': review_filter.get('version', 1)}
    return filtered, metadata
