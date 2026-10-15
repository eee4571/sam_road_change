"""Exact review carry-over: reordered, multipart, branched and changed roads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, GeometryCollection, Point
from app.review_history import feature_keys, match_decisions, snapshot_review_source
from app.change_review import source_signature


class HistoryTests(unittest.TestCase):
    def frame(self):
        a = LineString([(0, 0), (10, 0)])
        b = LineString([(1000, 0), (1010, 0)])
        c = LineString([(5, 0), (5, 5)])
        return gpd.GeoDataFrame({'change_typ': ['added'] * 3, 'width_aft': [8.] * 3},
            geometry=[MultiLineString([a, b]), MultiLineString([a, c]),
                      GeometryCollection([a, b, Point(2, 3)])], crs=3857)

    def test_reordered_disconnected_branched_collection_and_ids(self):
        before = self.frame()
        record = {'feature_keys': feature_keys(before), 'decisions': {'0': 'keep', '1': 'reject', '2': 'keep'}}
        after = before.iloc[[2, 0, 1]].reset_index(drop=True)
        after['change_id'] = ['new1', 'new2', 'new3']
        after['confidence'] = [.9, .7, .6]
        after.at[1, 'geometry'] = MultiLineString([list(g.coords)[::-1] for g in before.geometry[0].geoms][::-1])
        self.assertEqual(match_decisions(after, record), {'0': 'keep', '1': 'keep', '2': 'reject'})

    def test_geometry_width_type_changes_and_duplicates_stay_pending(self):
        before = self.frame()
        record = {'feature_keys': feature_keys(before), 'decisions': {str(i): 'reject' for i in range(3)}}
        for field, value in [('change_typ', 'removed'), ('width_aft', 8.01),
                             ('geometry', LineString([(0, 0), (10, 0)]))]:
            after = before.copy(); after.at[0, field] = value
            self.assertEqual(match_decisions(after, record), {'0': 'pending', '1': 'reject', '2': 'reject'})
        duplicate = before.iloc[[0, 0, 1]].reset_index(drop=True)
        self.assertEqual(match_decisions(duplicate, record), {'0': 'pending', '1': 'pending', '2': 'reject'})
        old = {'feature_keys': feature_keys(duplicate), 'decisions': {'0': 'keep', '1': 'reject', '2': 'keep'}}
        self.assertEqual(match_decisions(before, old)['0'], 'pending')

    def test_missing_evidence_never_matches_by_index(self):
        self.assertEqual(set(match_decisions(self.frame(), {'decisions': {'0': 'keep'}}).values()), {'pending'})

    def test_legacy_draft_snapshot_survives_source_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); output = root / 'results'
            source = output / 'a/02_变化检测/b_to_c/road_changes.shp'
            source.parent.mkdir(parents=True)
            frame = self.frame().iloc[:2].copy()
            frame.to_file(source)
            folder = source.parent / '人工复核/road_changes'; folder.mkdir(parents=True)
            inputs = {'b': None, 'c': None}
            record = dict(source_signature=source_signature(source, inputs), decisions={'0': 'keep', '1': 'reject'})
            (folder / 'draft.json').write_text(json.dumps(record), encoding='utf-8')
            manifest = dict(project_root=str(root), output_root=str(output))
            entry = dict(grid='a', before_period='b', after_period='c')
            snapshot_review_source(manifest, entry)
            pointer = json.loads((folder / 'draft_source.json').read_text())
            saved = folder / pointer['version'] / 'original_changes.shp'
            frame.iloc[:0].to_file(source)
            self.assertEqual(match_decisions(frame.iloc[::-1].reset_index(drop=True), record, saved),
                             {'0': 'reject', '1': 'keep'})


if __name__ == '__main__':
    unittest.main()
