"""Compare old/new assistance on one fixed, real automatic detection result.

The summary and its correction audit are read-only. Review decisions here are
synthetic markers, not human judgements or an accuracy evaluation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import geopandas as gpd
from app.review_history import feature_keys, match_decisions
from engine.fast_gt_reconciliation import _auto_axes, correct_changes
from engine.continuous_road_geometry import change_surfaces


def comparison(summary_path):
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8-sig'))
    audit = summary['correction_audit']
    proposals = gpd.read_file(audit, layer='perturbed_gt')
    automatic = _auto_axes(summary['automatic'], proposals.crs)
    # Before commit 79e460e the pipeline applied all these proposals directly.
    old, _ = correct_changes(automatic, proposals)
    new = gpd.read_file(audit, layer='corrected_intervals')
    with tempfile.TemporaryDirectory() as raw:
        original = Path(raw) / 'original.shp'
        candidate = Path(raw) / 'candidate.shp'
        change_surfaces(old).to_file(original, encoding='UTF-8')
        change_surfaces(new).to_file(candidate, encoding='UTF-8')
        before, after = gpd.read_file(original), gpd.read_file(candidate)
        decisions = {str(i): 'keep' if i % 2 else 'reject' for i in range(len(before))}
        restored = match_decisions(after, {'decisions': decisions}, original)
        old_keys, new_keys = feature_keys(before), feature_keys(after)
        old_counts, new_counts = Counter(old_keys.values()), Counter(new_keys.values())
        expected = {key: decisions[i] for i, key in old_keys.items()
                    if key and old_counts[key] == 1}
        for i, key in new_keys.items():
            assert restored[i] == (expected.get(key, 'pending') if new_counts[key] == 1 else 'pending')
        return dict(automatic_input_fixed=True, automatic_intervals=len(automatic),
                    old_assistance_candidates=len(before), new_assistance_candidates=len(after),
                    unchanged_unique_objects=sum(s != 'pending' for s in restored.values()),
                    simulated_restored_counts=dict(Counter(restored.values())),
                    synthetic_review_decisions=True,
                    notes='Same automatic detection and truth proposals; old unlimited correction versus new assistance budget. No project writes, inference or human review claims.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    result = comparison(args.summary)
    Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
