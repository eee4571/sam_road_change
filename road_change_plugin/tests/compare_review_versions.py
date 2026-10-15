"""Read-only comparison of one real change pair before/after a desktop rerun.

Run with the backend Python and two pipeline_result.json paths. No inference,
review decisions, evaluation denominator or project files are modified.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import geopandas as gpd
from shapely import normalize
from app.review_history import match_decisions


def comparison(old_manifest, new_manifest, grid, before, after):
    def entry(path):
        manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return next(e for e in manifest['change_results'] if
                    (str(e['grid']), str(e['before_period']), str(e['after_period'])) == (grid, before, after))

    old, new = entry(old_manifest), entry(new_manifest)

    def automatic(e):
        # A review-gated public SHP can be empty even when detection found
        # objects. Compare the pre-truth automatic audit, never that empty export.
        result = e.get('automatic', e)
        audit = Path(result['output']) / 'network_assembly.gpkg'
        return gpd.read_file(audit, layer='change_objects')

    def object_keys(frame):
        return [(normalize(row.geometry).wkb_hex, str(row.change_typ)) for row in frame.itertuples()]

    def compare_frames(a, b):
        matched = sum((Counter(object_keys(a)) & Counter(object_keys(b))).values()) if a.crs == b.crs else 0
        return dict(old_count=len(a), new_count=len(b), exact_geometry_and_type=matched)

    old_auto, new_auto = map(automatic, (old, new))
    review = old['manual_review']
    record = json.loads(Path(review['record']).read_text(encoding='utf-8-sig'))
    original = gpd.read_file(review['original'])
    candidates = gpd.read_file(new['review_candidates'])
    restored = match_decisions(candidates, record, review['original'])
    pairs = []
    old_keys, new_keys = object_keys(original), object_keys(candidates)
    old_counts, new_counts = Counter(old_keys), Counter(new_keys)
    for key in set(old_keys) & set(new_keys):
        if old_counts[key] == new_counts[key] == 1:
            pairs.append([old_keys.index(key), new_keys.index(key)])
    return dict(pair=[grid, before, after], automatic=compare_frames(old_auto, new_auto),
                candidates=compare_frames(original, candidates),
                original_decisions=record['counts'], restored_counts=dict(Counter(restored.values())),
                restored_decisions=restored, unchanged_geometry_pairs=sorted(pairs),
                notes='Exact object matching, including review-relevant widths; no nearest-road or ID-only matching.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_manifest')
    parser.add_argument('new_manifest')
    parser.add_argument('--grid', required=True)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    args = parser.parse_args()
    print(json.dumps(comparison(args.old_manifest, args.new_manifest, args.grid, args.before, args.after),
                     ensure_ascii=False, indent=2))
