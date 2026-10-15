"""Use explicit generation identities to apply human corrections to evaluation truth.

Public polygons dissolve several intervals. A truth object can be excluded only
when every linked interval maps to rejected public polygons. Retained, pending,
unmapped, and Auto-supported intervals protect the original object.
"""
from pathlib import Path

import geopandas as gpd
from shapely import make_valid


def build_truth_filter(original, decisions, audit_path, *, audit_layer='changes'):
    audit_path = Path(audit_path)
    audit = gpd.read_file(audit_path, layer=audit_layer)
    if audit.crs is None:
        raise ValueError('变化来源记录缺少坐标系，无法更新复核评价')
    result = {'audit': str(audit_path), 'crs': audit.crs.to_wkt(), 'objects': []}
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
    groups = {key: group for key, group in linked.groupby('truth_id', sort=False)}
    for truth_id, generated_group in generated.groupby('truth_id', sort=False):
        group = groups[truth_id]
        if 'truth_object_complete' in group and not group.truth_object_complete.fillna(False).astype(bool).all():
            continue
        review_ids = set()
        fully_rejected = True
        for row in group.itertuples():
            if row.geometry is None or row.geometry.is_empty:
                fully_rejected = False
                break
            geometry = make_valid(row.geometry)
            hits = []
            for position in spatial_index.query(geometry, predicate='intersects'):
                feature = public.iloc[int(position)]
                if feature.change_typ != row.change_typ:
                    continue
                # Boundary contact and numerical slivers do not establish origin.
                overlap = geometry.intersection(make_valid(feature.geometry)).area
                if overlap > max(1e-8, geometry.area * 1e-10):
                    hits.append(str(int(position)))
            if not hits or any(decisions[index] != 'reject' for index in hits):
                fully_rejected = False
                break
            review_ids.update(hits)
        if fully_rejected:
            kinds = {str(value) for value in generated_group.gt_type
                     if str(value) in {'added', 'removed', 'widened', 'narrowed', 'width_changed'}}
            kinds = {'width_changed' if kind in {'widened', 'narrowed'} else kind for kind in kinds}
            if kinds:
                result['objects'].append({'truth_id': str(truth_id), 'types': sorted(kinds),
                                          'review_ids': sorted(review_ids, key=int)})
    return result


def apply_truth_filter(truth, review_filter, truth_type_field):
    """Filter in memory; original files and the evaluation formula stay intact."""
    if not review_filter or not review_filter['objects']:
        return truth, {}
    from .fast_gt_reconciliation import _id
    from road_change_detection import _truth_type_field, _normalized_truth_change_type

    field = _truth_type_field(truth, truth_type_field)
    identities = {(item['truth_id'], kind) for item in review_filter['objects'] for kind in item['types']}
    projected = truth.to_crs(review_filter['crs'])
    excluded, applied = [], []
    for _, row in projected.iterrows():
        geometry = row.geometry
        kind = _normalized_truth_change_type(row[field], field, 'three') if field else None
        identity = _id(geometry, 'GT') if geometry is not None and not geometry.is_empty else ''
        remove = (identity, kind) in identities
        excluded.append(remove)
        if remove:
            applied.append(identity)
    # Recomputed from the original truth on every evaluation. A changed source
    # geometry or class cannot silently inherit an old exclusion.
    filtered = truth.iloc[[i for i, remove in enumerate(excluded) if not remove]].copy()
    metadata = {'original_feature_count': len(truth), 'effective_feature_count': len(filtered),
                'excluded_feature_count': len(applied), 'excluded_truth_ids': sorted(set(applied))}
    return filtered, metadata
