"""Project lifecycle acceptance using synthetic files; never run models."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from plugin.controller import Controller
from plugin.signals import TaskSignals
from plugin.project_state import ProjectState, read_json, write_json

APP = QApplication.instance() or QApplication([])
spec = importlib.util.spec_from_file_location('plugin_result_publisher_test', ROOT / 'code/app/result_publisher.py')
publisher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publisher
spec.loader.exec_module(publisher)


class SyntheticRunner(TaskSignals):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.commands = []
        self.missing = []

    def missing_runtime(self, profile):
        return self.missing

    def start(self, args, profile):
        self.commands.append(list(args))
        self.running = True
        self.task_started.emit({'task_id': 'internal-test'})

    def cancel(self):
        self.running = False
        self.task_finished.emit({'task_id': 'internal-test', 'status': 'cancelled'})

    def complete(self):
        self.running = False
        self.task_finished.emit({'task_id': 'internal-test', 'status': 'completed'})

    def shutdown(self):
        self.running = False


def file(path, text='original'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return str(path)


class ProjectLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.output = self.root / '成果输出'
        self.store = ProjectState(self.root, self.output)
        self.data = dict(project_root=str(self.root), output=str(self.output), areas=[], periods=[], truths=[],
                         area_irmad_references={}, grid='南区', period='2020', before_period='2020', after_period='2021')
        for area in ('南区', '北区'):
            self.data['areas'].append([area, file(self.root / 'inputs' / area / 'area.shp')])
            self.data['area_irmad_references'][area] = '2020'
            for year in range(2020, 2025):
                image = file(self.root / 'inputs' / area / f'{year}.tif')
                self.data['periods'].append([area, str(year), file(self.root / 'inputs' / area / f'{year}.txt', image)])
        self.cache = Path(file(self.root / '_work/cache/irmad/valid.dat', 'expensive-cache'))
        write_json(self.root / 'project_config.json', {'project': 'unchanged'})
        runner = patch('plugin.controller.Runner', SyntheticRunner)
        runner.start()
        self.addCleanup(runner.stop)
        self.controller = Controller()
        self.addCleanup(self.controller.shutdown)
        self.errors, self.finished = [], []
        self.controller.task_failed.connect(self.errors.append)
        self.controller.task_finished.connect(self.finished.append)

    def checkpoint(self, completed=False):
        state = self.store.state()
        manifest = dict(run_id=state['run_id'], job_root=str(self.store.current), output_root=str(self.output),
                        execution_profile='fast', status='completed' if completed else 'running',
                        input_spec={'width_method': 'raw_image', 'irmad': {'enabled': True}},
                        period_orders={area: {'period_order': [str(y) for y in range(2020, 2025)]} for area in ('南区', '北区')},
                        period_results=[], change_results=[], temporal_results=[])
        index = {'project_root': str(self.root), 'areas': {}}
        if completed:
            for area in ('南区', '北区'):
                node = index['areas'][area] = {'periods': {}, 'changes': {}, 'temporal': {}, 'evaluation': {}}
                for year in range(2020, 2025):
                    year = str(year)
                    p = file(self.output / area / '01_单期道路' / year / 'roads.shp')
                    node['periods'][year] = {'centerlines': p}
                    manifest['period_results'].append({'grid': area, 'period': year, 'published': {'centerlines': p}})
                    file(self.store.current / 'grids' / area / 'periods' / year / 'intermediate.dat')
                for year in range(2020, 2024):
                    before, after = str(year), str(year + 1)
                    key = before + '_to_' + after
                    p = file(self.output / area / '02_变化检测' / key / 'changes.shp')
                    node['changes'][key] = {'changes': p}
                    manifest['change_results'].append({'grid': area, 'before_period': before, 'after_period': after,
                                                       'published': {'changes': p}})
                    file(self.store.current / 'grids' / area / 'changes' / key / 'intermediate.dat')
                temporal = file(self.output / area / '03_长时序/life.shp')
                node['temporal'] = {'life_shp': temporal}
                manifest['temporal_results'].append({'grid': area, 'published': {'life_shp': temporal}})
            write_json(self.output / 'result_index.json', index)
        write_json(self.store.current / 'pipeline_result.json', manifest)
        write_json(self.store.current / 'job_state.json', manifest)
        return manifest

    def existing(self):
        self.controller.run('all', self.data)
        self.assertFalse(self.errors)
        manifest = self.checkpoint(completed=True)
        self.controller.runner.complete()
        self.assertFalse(self.store.resumable())
        return manifest

    def assert_cache(self):
        self.assertEqual(self.cache.read_text(), 'expensive-cache')
        self.assertEqual(read_json(self.root / 'project_config.json'), {'project': 'unchanged'})

    def test_first_run_uses_current_workspace_and_publishes_one_set(self):
        self.controller.run('all', self.data)
        self.assertEqual(self.errors, [])
        self.assertFalse(self.store.current.exists())  # backend exclusively creates it
        layout = publisher.ProjectLayout.from_project(self.root)
        layout.ensure_project_directories()
        self.assertFalse(self.store.current.exists())
        self.assertEqual(layout.full_run_root('anything'), self.store.current)
        self.assertEqual(layout.latest_pipeline_path, self.store.current / 'pipeline_result.json')
        self.checkpoint(True)
        self.controller.runner.complete()
        self.assertEqual(len(self.store.results()), 20)
        self.assertEqual(len(self.finished), 1)
        self.assertFalse((self.root / '_work/tasks/runs').exists())
        self.assert_cache()

    def test_cancel_then_continue_same_checkpoint_and_internal_id(self):
        self.controller.run('all', self.data)
        self.checkpoint()
        marker = Path(file(self.store.current / 'grids/completed.dat'))
        run_id = self.store.state()['run_id']
        self.controller.cancel()
        self.assertTrue(ProjectState(self.root).resumable())
        self.controller.run('all', {**self.data, 'resume': True})
        command = self.controller.runner.commands[-1]
        self.assertIn('--resume', command)
        self.assertEqual(command[command.index('--run-id') + 1], run_id)
        self.assertTrue(marker.exists())
        self.assert_cache()

    def test_cancel_then_full_run_discards_only_bound_workspace(self):
        self.controller.run('all', self.data)
        self.checkpoint()
        run_id = self.store.state()['run_id']
        self.controller.cancel()
        self.controller.run('all', self.data)
        self.assertEqual(self.errors, [])
        self.assertNotEqual(self.store.state()['run_id'], run_id)
        self.assertFalse(self.store.current.exists())
        self.assertNotIn('--resume', self.controller.runner.commands[-1])
        self.assertFalse(self.store.resumable())
        self.assert_cache()

    def test_complete_then_full_run_replaces_existing_formal_results(self):
        self.existing()
        old = [Path(p['path']) for p in self.store.results()]
        self.controller.run('all', self.data)
        self.assertFalse(any(p.exists() for p in old))
        self.assertEqual(self.store.results(), [])
        self.assertFalse((self.output / 'result_index.json').exists())
        self.checkpoint(True)
        self.controller.runner.complete()
        self.assertEqual(len(self.store.results()), 20)
        self.assert_cache()

    def test_period_rerun_cleans_four_period_evidence_dependencies_only(self):
        self.existing()
        original = self.store.results()
        preview = self.store.local_scope('rerun-period', self.data)
        self.controller.run('rerun-period', self.data)
        self.assertEqual(self.errors, [])
        scope = self.store.state()['scope']
        self.assertEqual(scope, preview)
        self.assertEqual(scope['changes'], ['2020_to_2021', '2021_to_2022'])
        for p in original:
            self.assertEqual(Path(p['path']).exists(), not self.store.in_scope(p, scope), p['path'])
        self.assertTrue((self.store.current / 'grids/南区/periods/2021/intermediate.dat').exists())
        self.assertFalse((self.store.current / 'grids/南区/periods/2020').exists())
        self.assertTrue((self.store.current / 'grids/南区/changes/2022_to_2023/intermediate.dat').exists())
        self.assert_cache()

    def test_change_rerun_and_cancel_continue_preserve_roads_and_other_changes(self):
        self.existing()
        original = self.store.results()
        self.controller.run('rerun-change', self.data)
        self.assertEqual(self.errors, [])
        scope = self.store.state()['scope']
        self.assertEqual(scope['periods'], [])
        self.assertEqual(scope['changes'], ['2020_to_2021'])
        preserved = {p['path']: (Path(p['path']).read_bytes(), Path(p['path']).stat().st_mtime_ns)
                     for p in self.store.results()}
        self.controller.cancel()
        self.assertTrue(self.store.resumable())
        self.controller.run('all', {**self.data, 'resume': True, 'grid': '北区', 'after_period': '2024'})
        command = self.controller.runner.commands[-1]
        self.assertEqual(command[0], 'rerun-change')
        self.assertEqual(command[command.index('--grid') + 1], '南区')
        self.assertEqual(command[command.index('--after-period') + 1], '2021')
        self.assertNotIn('--resume', command)
        for p in original:
            if self.store.in_scope(p, scope):
                file(Path(p['path']), 'replacement')
        self.controller.runner.complete()
        self.assertEqual(len(self.store.results()), len(original))
        for path, signature in preserved.items():
            self.assertEqual((Path(path).read_bytes(), Path(path).stat().st_mtime_ns), signature)
        self.assertFalse(self.store.resumable())
        self.assert_cache()

    def test_publisher_does_not_republish_unrelated_finalized_entries(self):
        self.existing()
        self.controller.run('rerun-period', self.data)
        pub = publisher.ResultPublisher(self.output, project_root=self.root)
        with patch.object(pub, '_copy_fields', side_effect=AssertionError('unrelated output was replaced')):
            self.assertTrue(pub.publish_period('南区', '2021', {}))
            self.assertTrue(pub.publish_period('北区', '2020', {}))
            self.assertTrue(pub.publish_change('南区', '2022', '2023', {}))
            self.assertTrue(pub.publish_temporal('北区', {}))
        self.assertIsNone(pub._preserved('南区', 'periods', '2020'))
        self.assertIsNone(pub._preserved('南区', 'changes', '2021_to_2022'))
        self.assertIsNone(pub._preserved('南区', 'temporal'))

    def test_invalid_selection_and_missing_runtime_never_clean_results(self):
        self.existing()
        original = copy.deepcopy(self.store.results())
        self.controller.run('rerun-period', {**self.data, 'period': '2099'})
        self.assertIn('不存在期次', self.errors[-1]['message'])
        self.assertEqual(self.store.results(), original)
        self.controller.runner.missing = ['model']
        self.controller.run('all', self.data)
        self.assertIn('缺少运行资源', self.errors[-1]['message'])
        self.assertEqual(self.store.results(), original)
        self.assert_cache()

    def test_cleanup_rejects_source_files_before_deleting_anything(self):
        self.existing()
        source = Path(file(self.output / '南区/01_单期道路/2020/input.shp'))
        bad = {**self.data, 'areas': [['南区', str(source)]]}
        with self.assertRaisesRegex(ValueError, '输入数据'):
            self.store.fresh(bad)
        self.assertEqual(len(self.store.results()), 20)
        self.assertTrue(source.exists())
        with self.assertRaisesRegex(ValueError, '范围'):
            self.store._delete([self.root.parent / 'never-delete-me'], self.data)
        with self.assertRaisesRegex(ValueError, '通用缓存'):
            self.store._delete([self.cache], self.data)
        self.assert_cache()

    def test_completed_without_manifest_reports_failure_only(self):
        self.controller.run('all', self.data)
        self.controller.runner.complete()
        self.assertFalse(self.finished)
        self.assertEqual(self.store.state()['status'], 'failed')
        self.assertIn('索引不存在', self.errors[-1]['message'])

    def test_second_plugin_cannot_replace_an_active_project(self):
        self.controller.run('all', self.data)
        self.checkpoint()
        before = self.store.state()
        other = Controller()
        self.addCleanup(other.shutdown)
        failures = []
        other.task_failed.connect(failures.append)
        other.run('all', self.data)
        self.assertIn('其他窗口', failures[-1]['message'])
        self.assertFalse(other.runner.commands)
        self.assertEqual(self.store.state(), before)
        self.assertTrue(self.store.manifest().is_file())
        self.controller.cancel()
        other.run('all', {**self.data, 'resume': True})
        self.assertTrue(other.runner.commands)
        self.assertEqual(self.store.state()['run_id'], before['run_id'])

    def test_dependency_contract_matches_backend_without_importing_algorithms(self):
        # Extract the pure dependency function only; importing this module would
        # load GIS dependencies. This guards the project cleanup contract.
        tree = ast.parse((ROOT / 'code/engine/fast_multitemporal.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dependency_periods')
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<dependency contract>', 'exec'), namespace)
        names = [str(y) for y in range(2020, 2025)]
        for period in names:
            expected = [f'{a}_to_{b}' for a, b in zip(names, names[1:]) if period in namespace['dependency_periods'](names, a, b)]
            actual = [f'{a}_to_{b}' for i, (a, b) in enumerate(zip(names, names[1:])) if period in names[max(0, i-1):i+3]]
            self.assertEqual(actual, expected)

    def test_configuration_change_is_not_cleared_by_local_update(self):
        self.existing()
        self.assertEqual(self.store.configuration_status(), 'current')
        write_json(self.root / 'project_config.json', {'area_irmad_references': {'南区': '2024'}})
        self.assertEqual(self.store.configuration_status(), 'changed')
        self.controller.run('rerun-change', self.data)
        self.controller.runner.complete()
        self.assertEqual(self.store.configuration_status(), 'changed')
        self.controller.run('all', self.data)
        self.checkpoint(True)
        self.controller.runner.complete()
        self.assertEqual(self.store.configuration_status(), 'current')

    def test_resume_keeps_original_inputs_after_configuration_change(self):
        self.controller.run('all', self.data)
        self.checkpoint()
        self.controller.cancel()
        self.controller.run('all', {**self.data, 'resume': True, 'areas': [], 'periods': []})
        self.assertFalse(self.errors)
        self.assertIn('--resume', self.controller.runner.commands[-1])
        self.assertIn('--validation-area', self.controller.runner.commands[-1])

    def test_publication_filter_keeps_internal_width_and_unrelated_snapshot(self):
        from plugin.result_parser import formal_results
        manifest = self.existing()
        products = self.store.results()
        width = file(self.output / 'width.shp')
        added = file(self.output / 'added.shp')
        manifest['period_results'][0]['published']['width_segments'] = width
        manifest['change_results'][0]['published']['added'] = added
        write_json(self.store.manifest(), manifest)
        write_json(self.store.results_path, products + [dict(result_type='road_width', path=width, metadata={}),
                                                      dict(result_type='road_change', path=added, metadata={})])
        delivered = formal_results(self.store)
        self.assertNotIn(width, [p['path'] for p in delivered])
        self.assertNotIn(added, [p['path'] for p in delivered])
        self.assertTrue(Path(width).exists())
        self.controller.run('rerun-period', self.data)
        manifest = read_json(self.store.manifest())
        manifest['fast_finalization_state'] = 'pending'
        write_json(self.store.manifest(), manifest)
        self.assertTrue(formal_results(self.store))
        self.assertTrue(all(not self.store.in_scope(p, self.store.state()['scope']) for p in formal_results(self.store)))

    def test_local_update_uses_processed_output_when_new_directory_is_configured(self):
        self.existing()
        self.controller.run('rerun-change', {**self.data, 'output': str(self.root / 'new-output')})
        self.assertFalse(self.errors)
        self.assertEqual(self.store.state()['output'], str(self.output))


if __name__ == '__main__':
    unittest.main()
