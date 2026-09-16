"""Read the formal manifest contract, without loading GIS libraries."""
import json
from pathlib import Path

RESULT_TYPES = ("road_centerline", "road_surface", "road_width", "road_change", "road_temporal", "road_evaluation")


def formal_results(store):
    """Filter old snapshots to the current formal products; keep internal data intact."""
    manifest = store.manifest()
    allowed = {(p['result_type'], p['path']): p for p in read_results(manifest, include_pending=True)} if manifest.is_file() else {}
    result = []
    for item in store.results():
        key = (item['result_type'], item['path'])
        if key in allowed:
            result.append({**item, 'metadata': allowed[key]['metadata']})
    return list({(p['result_type'], p['path']): p for p in result}.values())


def read_results(path, *, include_pending=False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("成果索引必须为 JSON 对象")
    if manifest.get("fast_finalization_state") == "pending" and not include_pending:
        return []
    results, seen = [], set()

    def add(kind, value, name, metadata):
        if not isinstance(value, str) or not value:
            return
        target = Path(value)
        if not target.is_absolute():
            target = path.parent / target
        target = target.resolve()
        key = (kind, str(target))
        if target.is_file() and key not in seen:
            seen.add(key)
            results.append(dict(result_type=kind, name=name, path=str(target), metadata=metadata))

    for entry in manifest.get("final_period_results", manifest.get("period_results", [])):
        products = entry.get("published") or entry
        for key, kind in (("centerlines", "road_centerline"), ("surfaces", "road_surface")):
            add(kind, products.get(key), f"{entry.get('grid', '')} / {entry.get('period', '')} / {kind}", {"grid": entry.get("grid"), "period": entry.get("period")})
    for entry in manifest.get("change_results", []):
        products = entry.get("published") or {**entry, **entry.get("layers", {})}
        meta = {k: entry.get(k) for k in ("grid", "before_period", "after_period")}
        value = products.get('changes')
        if isinstance(value, str) and Path(value).suffix.lower() == '.shp':
            add("road_change", value, f"{meta['grid']} / {meta['before_period']} → {meta['after_period']}",
                {**meta, 'classification_field': 'change_typ'})
    for entry in manifest.get("temporal_results", []):
        products = entry.get("published") or entry
        for key in ("life_shp", "observations_shp", "events_shp", "event_parts_shp", "lineage_shp"):
            add("road_temporal", products.get(key), f"{entry.get('grid', '')} / {key}", {"grid": entry.get("grid")})
    summary = manifest.get("evaluation_summary") or {}
    for key in ("csv", "json"):
        add("road_evaluation", summary.get(key), f"精度评价 / {key}", {})
    return results
