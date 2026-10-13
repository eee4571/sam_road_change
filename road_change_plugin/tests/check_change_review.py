"""Synthetic GIS review checks. Run with the plugin runtime; no models required."""
import base64
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import geopandas as gpd
import numpy as np
from PIL import Image
import pyogrio
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString
from app.change_review import ReviewSession, read_json, source_signature
from app.review_publication import write_json


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / '成果输出'
        self.source = self.output / '北区/02_变化检测/before_to_after/road_changes.shp'
        self.source.parent.mkdir(parents=True)
        self.frame = gpd.GeoDataFrame({"change_typ": ["added", "removed", "widened"],
            "road_id": [101, 205, 309], "width_m": [4.25, 8.5, 6.0], "name": ["甲", "乙", "丙"]},
            geometry=[LineString([(500030 + i * 10, 3300030), (500030 + i * 10, 3300060)]) for i in range(3)],
            crs="EPSG:32650")
        pyogrio.write_dataframe(self.frame, self.source, encoding="UTF-8")
        periods = []
        for name, color in (("before", [40, 80, 120]), ("after", [100, 150, 200])):
            image = self.root / (name + ".tif")
            with rasterio.open(image, "w", driver="GTiff", width=100, height=100, count=3,
                               dtype="uint8", crs=self.frame.crs, transform=from_origin(500000, 3300100, 1, 1)) as dataset:
                dataset.write(np.array(color, dtype="uint8")[:, None, None] * np.ones((3, 100, 100), dtype="uint8"))
            (self.root / (name + ".txt")).write_text(image.name + "\n", encoding="utf-8")
            roads = self.root / (name + '_roads.shp')
            pyogrio.write_dataframe(self.frame, roads, encoding='UTF-8')
            periods.append(dict(grid='北区', period=name, centerlines=str(roads)))
        self.manifest = self.root / "pipeline_result.json"
        self.manifest.write_text(json.dumps({'project_root': str(self.root), 'output_root': str(self.output),
            'execution_profile': 'fast', 'status': 'completed', 'period_results': periods,
            "input_spec": {"grids": {"北区": {
            p: {"path": p + ".txt"} for p in ("before", "after")}}}, "change_results": [
            {"grid": "北区", "before_period": "before", "after_period": "after", 'output': str(self.root),
             'layers': {'changes': str(self.source)}, 'execution_profile': 'fast', 'fast_finalization_state': 'completed',
             "published": {"changes": str(self.source)}}]}), encoding="utf-8")
        self.session = ReviewSession(self.manifest)
        self.data = self.session.open_pair(0)

    def test_preview_same_extent_and_color(self):
        preview = self.session.preview("0", size=128)
        self.assertTrue(preview["has_images"])
        self.assertEqual(preview["geometry"]["type"], "LineString")
        first, last = preview["geometry"]["coordinates"]
        self.assertAlmostEqual(first[0], 64)
        self.assertAlmostEqual(last[0], 64)
        images = [np.array(Image.open(io.BytesIO(base64.b64decode(x)))) for x in preview["images"]]
        np.testing.assert_equal(images[0][64, 64], [40, 80, 120, 255])
        np.testing.assert_equal(images[1][64, 64], [100, 150, 200, 255])

    def test_all_candidates_use_existing_tool_but_only_keep_publishes(self):
        candidates=self.root/'review_candidates.shp'
        frame=self.frame.copy();frame['confidence']=[.3,.9,.6]
        pyogrio.write_dataframe(frame,candidates,encoding='UTF-8')
        pyogrio.write_dataframe(frame.iloc[:0],self.source,geometry_type='LineString',encoding='UTF-8')
        manifest=read_json(self.manifest)
        manifest['change_results'][0].update(review_candidates=str(candidates),manual_review_required=True)
        write_json(self.manifest,manifest)
        session=ReviewSession(self.manifest);data=session.open_pair(0)
        self.assertEqual([f['id'] for f in data['features']],['1','2','0'])
        self.assertTrue(all(f['status']=='pending' for f in data['features']))
        before=source_signature(self.source,session.inputs)
        decisions={'0':'keep','1':'pending','2':'reject'}
        session.record(decisions)
        self.assertEqual(source_signature(self.source,session.inputs),before)
        session.save(decisions)
        self.assertEqual(gpd.read_file(self.source).road_id.tolist(),[101])
        reopened=ReviewSession(self.manifest);data=reopened.open_pair(0)
        self.assertEqual(len(data['features']),3)
        self.assertEqual({f['id']:f['status'] for f in data['features']},decisions)
        changed=frame.copy();changed.loc[1,'confidence']=.8
        pyogrio.write_dataframe(changed,candidates,encoding='UTF-8')
        with self.assertRaisesRegex(ValueError,'已改变'):session.save(decisions)
        with self.assertRaisesRegex(ValueError,'已改变'):reopened.save(decisions)
        self.assertTrue(all(f['status']=='pending' for f in ReviewSession(self.manifest).open_pair(0)['features']))

    def test_save_resume_preserves_original_and_attributes(self):
        before = source_signature(self.source, self.session.inputs)
        decisions = {"0": "keep", "1": "reject", "2": "pending"}
        saved = self.session.save(decisions)
        output = gpd.read_file(saved["path"])
        self.assertEqual(list(output.road_id), [101])
        reviewed = gpd.read_file(self.session.source.parent / 'road_changes_reviewed.shp')
        self.assertEqual(list(reviewed["name"]), ["甲"])
        self.assertTrue(output.crs == self.frame.crs)
        self.assertEqual(list(output.width_m), [4.25])
        self.assertTrue(output.geometry.iloc[0].equals_exact(self.frame.geometry.iloc[0], 0))
        rejected = gpd.read_file(self.session.source.parent / "false_changes.shp")
        self.assertEqual(list(rejected.road_id), [205])
        self.assertEqual(before, source_signature(self.session.source, self.session.inputs))
        reopened = ReviewSession(self.manifest).open_pair(0)
        self.assertEqual({f["id"]: f["status"] for f in reopened["features"]}, decisions)
        self.assertEqual(reopened["saved_path"], saved["path"])
        restored = ReviewSession(self.manifest)
        restored.open_pair(0)
        restored.save({'0': 'keep', '1': 'keep', '2': 'pending'})
        self.assertEqual(len(gpd.read_file(self.source)), 2)

    def test_all_rejected_and_empty_input_export_valid_shapefiles(self):
        saved = self.session.save({str(i): "reject" for i in range(3)})
        self.assertEqual(len(gpd.read_file(saved["path"])), 0)
        pyogrio.write_dataframe(self.frame.iloc[:0], self.source, geometry_type="LineString")
        empty = ReviewSession(self.manifest)
        self.assertEqual(empty.open_pair(0)["features"], [])
        saved = empty.save({})
        self.assertEqual(len(gpd.read_file(saved["path"])), 0)

    def test_changed_source_does_not_reuse_review(self):
        self.session.save({str(i): "reject" for i in range(3)})
        altered = self.frame.copy()
        altered.loc[0, "road_id"] = 999
        pyogrio.write_dataframe(altered, self.source, encoding="UTF-8")
        with self.assertRaisesRegex(ValueError, "已改变"):
            self.session.save({str(i): "keep" for i in range(3)})
        loaded = ReviewSession(self.manifest).open_pair(0)
        self.assertTrue(loaded["warning"])
        self.assertTrue(all(f["status"] == "pending" for f in loaded["features"]))

    def test_failed_export_keeps_last_complete_version(self):
        self.session.save({str(i): "keep" for i in range(3)})
        pointer = self.session.folder / "current.json"
        previous = pointer.read_bytes()
        with patch("app.change_review.pyogrio.write_dataframe", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.session.save({str(i): "reject" for i in range(3)})
        self.assertEqual(pointer.read_bytes(), previous)
        with self.assertRaises(ValueError):
            self.session.save({"0": "keep"})

    def test_missing_imagery_and_no_coverage_are_explicit(self):
        (self.root / "after.tif").unlink()
        loaded = self.session.open_pair(0)
        self.assertTrue(loaded["image_warnings"])
        preview = self.session.preview("0", size=64)
        self.assertFalse(preview["has_images"])
        self.assertTrue(preview["warnings"])

    def test_existing_alpha_and_reprojection(self):
        from rasterio.enums import ColorInterp
        from rasterio.warp import transform_bounds
        from rasterio.transform import from_bounds
        bounds = transform_bounds(self.frame.crs, "EPSG:4326", 500000, 3300000, 500100, 3300100)
        with rasterio.open(self.root / "after.tif", "w", driver="GTiff", width=100, height=100, count=4,
                           dtype="uint8", crs="EPSG:4326", transform=from_bounds(*bounds, 100, 100)) as dataset:
            values = np.full((4, 100, 100), 128, dtype="uint8")
            values[3] = 255
            values[3, :, 50:] = 0
            dataset.write(values)
            dataset.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
        self.session.open_pair(0)
        preview = self.session.preview("0", size=128)
        self.assertTrue(preview["has_images"])
        after = np.array(Image.open(io.BytesIO(base64.b64decode(preview["images"][1]))))
        self.assertEqual(after[64, 64, 3], 255)
        self.assertEqual(after[64, 125, 3], 0)

    def polygon_session(self, truth=True):
        frame = self.frame.copy()
        frame['change_typ'] = 'added'
        frame.geometry = frame.geometry.buffer(2)
        pyogrio.write_dataframe(frame, self.source, encoding='UTF-8')
        manifest = read_json(self.manifest)
        if truth:
            reference = frame.iloc[[0]].copy()
            reference['BHBM'] = 2
            path = self.root / 'truth.shp'
            pyogrio.write_dataframe(reference, path, encoding='UTF-8')
            manifest['input_spec']['truths'] = {'北区\0before\0after': {'path': str(path)}}
            manifest['input_spec']['truth_type_field'] = 'BHBM'
        # Another area has its own already evaluated result; it must survive.
        other_source = self.root / 'other.shp'
        pyogrio.write_dataframe(frame.iloc[[0]], other_source, encoding='UTF-8')
        other = copy.deepcopy(manifest['change_results'][0])
        other.update(grid='南区', output=str(self.root / 'other'), summary=str(self.root / 'other_summary.json'),
                     layers={'changes': str(other_source)}, published={'changes': str(other_source)},
                     evaluation_state='completed')
        write_json(other['summary'], {'evaluation': {'metrics': [
            {'class': 'all', 'tp_m2': 120, 'fp_m2': 0, 'fn_m2': 0, 'tn_m2': 0,
             'correctly_classified_m2': 120, 'detected_truth_m2': 120}]}})
        manifest['change_results'].append(other)
        manifest['temporal_results'] = [{'grid': '南区', 'event_count': 7, 'sentinel': 'unchanged'}]
        write_json(self.manifest, manifest)
        session = ReviewSession(self.manifest)
        session.open_pair(0)
        return session, other_source.read_bytes(), manifest['change_results'][1]

    def test_real_evaluation_temporal_and_other_area_are_updated_consistently(self):
        session, other_bytes, other_entry = self.polygon_session()
        first = session.save({str(i): 'keep' for i in range(3)})
        before = read_json(self.manifest)
        precision_before = before['evaluation_summary']['metrics'][0]['change_precision']
        temporal_before = next(t for t in before['temporal_results'] if t['grid'] == '北区')
        self.assertEqual(temporal_before['event_count'], 3)
        second = session.save({'0': 'keep', '1': 'reject', '2': 'reject'})
        self.assertEqual(second['evaluation_state'], 'completed')
        after = read_json(self.manifest)
        self.assertGreater(after['evaluation_summary']['metrics'][0]['change_precision'], precision_before)
        self.assertAlmostEqual(after['evaluation_summary']['metrics'][0]['change_precision'], 1.0)
        self.assertEqual(after['evaluation_summary']['evaluated_task_count'], 2)
        self.assertEqual(after['change_results'][1], other_entry)
        self.assertEqual((self.root / 'other.shp').read_bytes(), other_bytes)
        self.assertEqual(next(t for t in after['temporal_results'] if t['grid'] == '南区')['event_count'], 7)
        temporal_after = next(t for t in after['temporal_results'] if t['grid'] == '北区')
        self.assertEqual(temporal_after['event_count'], 1)
        self.assertEqual(len(gpd.read_file(temporal_after['published']['event_parts_shp'])), 1)
        life = gpd.read_file(temporal_after['published']['life_shp'])
        self.assertEqual(int(life.event_n.sum()), 1)
        for name in ('life_shp', 'events_shp', 'event_parts_shp'):
            self.assertTrue(Path(temporal_after['published'][name]).is_file())
        record = read_json(after['change_results'][0]['manual_review']['record'])
        self.assertEqual(record['evaluation'], 'completed')
        self.assertEqual(after['change_results'][0]['changes_feature_count'], 1)
        summary = read_json(after['change_results'][0]['summary'])
        self.assertEqual(summary['evaluation']['metadata']['evaluation_source'], 'manual_reviewed_changes_vs_ground_truth')
        # Removing all changes must not recreate state-derived events.
        session.save({str(i): 'reject' for i in range(3)})
        final = read_json(self.manifest)
        temporal = next(t for t in final['temporal_results'] if t['grid'] == '北区')
        self.assertEqual(temporal['event_count'], 0)
        self.assertEqual(int(gpd.read_file(temporal['published']['life_shp']).event_n.sum()), 0)

    def test_review_reuses_matching_and_agrees_with_full_temporal_build(self):
        from geopandas.testing import assert_geodataframe_equal
        from temporal_road_analysis import build_temporal_grid
        session, _, _ = self.polygon_session(truth=False)
        session.save({str(i): 'keep' for i in range(3)})
        before = next(t for t in read_json(self.manifest)['temporal_results'] if t['grid'] == '北区')
        observation_bytes = Path(before['observations_shp']).read_bytes()
        with patch('temporal_road_analysis._read_period', side_effect=AssertionError('道路不应重新匹配')):
            session.save({'0': 'keep', '1': 'reject', '2': 'reject'})
        manifest = read_json(self.manifest)
        updated = next(t for t in manifest['temporal_results'] if t['grid'] == '北区')
        self.assertEqual(Path(updated['observations_shp']).read_bytes(), observation_bytes)
        full = build_temporal_grid('北区', manifest['period_results'],
                                  [manifest['change_results'][0]], self.root / 'full_temporal')
        for name in ('life_shp', 'observations_shp', 'events_shp', 'event_parts_shp', 'review_shp'):
            assert_geodataframe_equal(gpd.read_file(updated[name]), gpd.read_file(full[name]))
        # Restoring an earlier rejected change must work with reused observations.
        with patch('temporal_road_analysis._read_period', side_effect=AssertionError('道路不应重新匹配')):
            session.save({str(i): 'keep' for i in range(3)})
        restored = next(t for t in read_json(self.manifest)['temporal_results'] if t['grid'] == '北区')
        self.assertEqual(restored['event_count'], 3)

    def test_no_truth_and_failed_commit_restore_the_complete_previous_result(self):
        from app import review_publication
        self.session.save({str(i): 'keep' for i in range(3)})
        manifest_before = self.manifest.read_bytes()
        current_before = self.source.read_bytes()
        pointer = self.session.folder / 'current.json'
        pointer_before = pointer.read_bytes()
        real_replace = review_publication.replace_file
        calls = []

        def fail_once(source, target):
            calls.append(str(target))
            if len(calls) == 8:
                raise OSError('模拟文件占用')
            return real_replace(source, target)

        with patch.object(review_publication, 'replace_file', side_effect=fail_once):
            with self.assertRaisesRegex(OSError, '模拟文件占用'):
                self.session.save({str(i): 'reject' for i in range(3)})
        self.assertEqual(self.manifest.read_bytes(), manifest_before)
        self.assertEqual(self.source.read_bytes(), current_before)
        self.assertEqual(pointer.read_bytes(), pointer_before)
        result = self.session.save({str(i): 'reject' for i in range(3)})
        self.assertEqual(result['evaluation_state'], 'skipped_no_truth')
        self.assertEqual(read_json(self.manifest)['evaluation_summary']['metrics'], [])

    def test_evaluation_failure_does_not_publish_or_mark_review_saved(self):
        import user_pipeline
        session, _, _ = self.polygon_session()
        original = self.source.read_bytes()
        manifest = self.manifest.read_bytes()
        with patch.object(user_pipeline, '_evaluate_existing_changes_impl', side_effect=ValueError('真值不可读')):
            with self.assertRaisesRegex(ValueError, '真值不可读'):
                session.save({'0': 'keep', '1': 'reject', '2': 'reject'})
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(self.manifest.read_bytes(), manifest)
        self.assertFalse((session.folder / 'current.json').exists())

    def test_interrupted_publication_is_recovered_on_reopen(self):
        from app import review_publication
        self.session.save({str(i): 'keep' for i in range(3)})
        previous = self.source.read_bytes()
        previous_manifest = self.manifest.read_bytes()
        real_replace = review_publication.replace_file
        count = 0

        def interrupt(source, target):
            nonlocal count
            count += 1
            if count == 8:
                raise KeyboardInterrupt('模拟进程中断')
            real_replace(source, target)

        with patch.object(review_publication, 'replace_file', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.session.save({str(i): 'reject' for i in range(3)})
        self.assertEqual(read_json(self.root / '_work/review_publication.json')['state'], 'committing')
        restored = ReviewSession(self.manifest)
        self.assertEqual(self.source.read_bytes(), previous)
        self.assertEqual(self.manifest.read_bytes(), previous_manifest)
        reopened = restored.open_pair(0)
        self.assertTrue(all(f['status'] == 'reject' for f in reopened['features']))
        self.assertTrue(all(s == 'keep' for s in reopened['applied_decisions'].values()))

    def test_record_is_durable_without_publishing_or_running_analysis(self):
        before_manifest = self.manifest.read_bytes()
        before_source = source_signature(self.source, self.session.inputs)
        decisions = {'0': 'keep', '1': 'reject', '2': 'pending'}
        with patch('app.review_publication.publish_review', side_effect=AssertionError('not an apply')):
            self.session.record(decisions)
        self.assertEqual(self.manifest.read_bytes(), before_manifest)
        self.assertEqual(source_signature(self.source, self.session.inputs), before_source)
        self.assertFalse((self.session.folder / 'current.json').exists())
        reopened = ReviewSession(self.manifest)
        self.assertTrue(reopened.review_states()[0]['modified'])
        data = reopened.open_pair(0)
        self.assertEqual({f['id']: f['status'] for f in data['features']}, decisions)
        self.assertTrue(all(s == 'pending' for s in data['applied_decisions'].values()))
        reopened.save(decisions)
        self.assertFalse(ReviewSession(self.manifest).review_states()[0]['modified'])

    def test_failed_record_preserves_previous_durable_judgements(self):
        decisions = {'0': 'keep', '1': 'pending', '2': 'pending'}
        self.session.record(decisions)
        with patch('app.change_review.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.session.record({str(i): 'reject' for i in range(3)})
        data = ReviewSession(self.manifest).open_pair(0)
        self.assertEqual({f['id']: f['status'] for f in data['features']}, decisions)

    def test_changed_detection_does_not_load_autosaved_judgements(self):
        self.session.record({str(i): 'reject' for i in range(3)})
        changed = self.frame.copy()
        changed.loc[0, 'road_id'] = 999
        pyogrio.write_dataframe(changed, self.source, encoding='UTF-8')
        data = ReviewSession(self.manifest).open_pair(0)
        self.assertTrue(data['warning'])
        self.assertTrue(all(f['status'] == 'pending' for f in data['features']))


if __name__ == "__main__":
    unittest.main()
