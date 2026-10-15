"""Copied legacy plugin projects retain caches and never write to the original."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plugin_project_relocation_test', ROOT/'code/app/project_relocation.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.old, self.new = self.base/'旧项目', self.base/'复制项目'
        self.old.mkdir()
        self.write('project_config.json', dict(project_root=str(self.old), output_root='成果输出'))
        self.write('_work/project_state.json', dict(manifest=str(self.old/'_work/current/pipeline_result.json'),
            output=str(self.old/'成果输出'), parameters={'project_root':str(self.old)}))
        self.write('_work/current/pipeline_result.json', dict(project_root=str(self.old),
            job_root=str(self.old/'_work/current'), run_id='current_original_id',
            result=str(self.old/'_work/current/result.json'), algorithm_revision='old-revision'))
        self.write('_work/current/result.json', dict(road=str(self.old/'_work/current/cache.bin'),
            external=str(self.base/'external.tif'), similar=str(self.base/'旧项目2/data.tif')))
        self.write('_work/results.json', [{'path':str(self.old/'成果输出/roads.shp')}])
        (self.old/'_work/current/cache.bin').write_bytes(b'cached-inference-and-width')
        (self.old/'_work/current/list.txt').write_text(str(self.old/'_work/current/cache.bin')+'\n',encoding='utf-8-sig')
        shutil.copytree(self.old,self.new)
        self.original = {p.relative_to(self.old):p.read_bytes() for p in self.old.rglob('*') if p.is_file()}

    def write(self, relative, data):
        path = self.old/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')

    def test_existing_original_does_not_capture_copied_project(self):
        binary = self.new/'_work/current/cache.bin';before=binary.stat().st_mtime_ns
        result=module.relocate_plugin_project(self.new)
        self.assertGreater(result['modified_files'],0)
        state=json.loads((self.new/'_work/project_state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['manifest'],str(self.new/'_work/current/pipeline_result.json'))
        manifest=json.loads(Path(state['manifest']).read_text(encoding='utf-8'))
        self.assertEqual(manifest['algorithm_revision'],'old-revision')
        self.assertEqual(manifest['run_id'],'current_original_id')
        result=json.loads(Path(manifest['result']).read_text(encoding='utf-8'))
        self.assertEqual(result['road'],str(binary))
        self.assertEqual(result['external'],str(self.base/'external.tif'))
        self.assertEqual(result['similar'],str(self.base/'旧项目2/data.tif'))
        self.assertEqual((self.new/'_work/current/list.txt').read_text(encoding='utf-8-sig').strip(),str(binary))
        self.assertEqual(binary.stat().st_mtime_ns,before)
        self.assertEqual(binary.read_bytes(),b'cached-inference-and-width')
        self.assertEqual({p.relative_to(self.old):p.read_bytes() for p in self.old.rglob('*') if p.is_file()},self.original)
        self.assertEqual(module.relocate_plugin_project(self.new)['modified_files'],0)

    def test_absent_original_uses_copied_metadata(self):
        shutil.rmtree(self.old)
        module.relocate_plugin_project(self.new)
        state=json.loads((self.new/'_work/project_state.json').read_text(encoding='utf-8'))
        self.assertTrue(Path(state['manifest']).is_file())

    def test_corrupt_owned_metadata_does_not_partially_migrate(self):
        bad=self.new/'_work/current/bad.json'
        bad.write_text('{"root":'+json.dumps(str(self.old)),encoding='utf-8')
        before={p:p.read_bytes() for p in self.new.rglob('*.json')}
        with self.assertRaises(ValueError):module.relocate_plugin_project(self.new)
        self.assertEqual({p:p.read_bytes() for p in before},before)

    def test_external_output_requires_explicit_location(self):
        p=self.new/'project_config.json'
        p.write_text(json.dumps(dict(project_root=str(self.old),output_root=str(self.base/'elsewhere'))),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'项目之外'):module.relocate_plugin_project(self.new)

    def test_interrupted_write_can_resume_without_losing_original_backup(self):
        original_write = module._atomic_write_json
        calls = []
        def interrupted(path, value):
            calls.append(path)
            if len(calls) == 2:
                raise OSError('simulated interrupted migration')
            original_write(path, value)
        with patch.object(module, '_atomic_write_json', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                module.relocate_plugin_project(self.new)
        config = json.loads((self.new/'project_config.json').read_text(encoding='utf-8'))
        self.assertEqual(config['project_root'], str(self.old))
        module.relocate_plugin_project(self.new)
        for path in calls:
            backup = path.with_name(path.name + module.BACKUP_SUFFIX)
            self.assertEqual(backup.read_bytes(), self.original[path.relative_to(self.new)])
        state = json.loads((self.new/'_work/project_state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['manifest'],str(self.new/'_work/current/pipeline_result.json'))

    @unittest.skipUnless(sys.platform == 'win32', 'Windows path-length regression')
    def test_backup_and_atomic_write_past_windows_path_limit(self):
        self.assertEqual(self.base, Path(self.temp.name).resolve())
        self.addCleanup(shutil.rmtree, module._io_path(self.base))
        folder = self.new/'_work'
        while len(str(folder)) < 210:
            folder /= 'nested_cache'
        module._io_path(folder).mkdir(parents=True, exist_ok=True)
        path = folder/('road_' + 'x'*25 + '.completed.json')
        original = json.dumps({'path': str(self.old/'_work/current/cache.bin')})
        module._io_path(path).write_text(original, encoding='utf-8')
        self.assertGreater(len(str(path) + module.BACKUP_SUFFIX), 260)
        module.relocate_plugin_project(self.new)
        self.assertEqual(json.loads(module._io_path(path).read_text(encoding='utf-8'))['path'],
                         str(self.new/'_work/current/cache.bin'))
        self.assertEqual(module._io_path(path.with_name(path.name+module.BACKUP_SUFFIX)).read_text(encoding='utf-8'), original)


if __name__=='__main__':unittest.main()
