"""Read the formal manifest contract, without loading GIS libraries."""
import json
from pathlib import Path

RESULT_TYPES = ("road_centerline", "road_surface", "road_width", "road_change", "road_temporal", "road_evaluation")


def read_results(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("成果索引必须为 JSON 对象")
    if manifest.get("fast_finalization_state") == "pending":
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
        products = {**entry, **entry.get("published", {})}
        for key, kind in (("centerlines", "road_centerline"), ("surfaces", "road_surface"), ("width_segments", "road_width")):
            add(kind, products.get(key), f"{entry.get('grid', '')} / {entry.get('period', '')} / {kind}", {"grid": entry.get("grid"), "period": entry.get("period")})
    for entry in manifest.get("change_results", []):
        products = {**entry, **entry.get("layers", {}), **entry.get("published", {})}
        meta = {k: entry.get(k) for k in ("grid", "before_period", "after_period")}
        for key in ("changes", "added", "removed", "widened", "narrowed", "gpkg"):
            add("road_change", products.get(key), f"{meta['grid']} / {meta['before_period']} → {meta['after_period']} / {key}", meta)
    for entry in manifest.get("temporal_results", []):
        products = {**entry, **entry.get("published", {})}
        for key in ("life_shp", "observations_shp", "events_shp", "event_parts_shp", "lineage_shp"):
            add("road_temporal", products.get(key), f"{entry.get('grid', '')} / {key}", {"grid": entry.get("grid")})
    summary = manifest.get("evaluation_summary") or {}
    for key in ("csv", "json"):
        add("road_evaluation", summary.get(key), f"精度评价 / {key}", {})
    return results
