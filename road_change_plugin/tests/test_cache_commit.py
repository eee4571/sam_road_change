"""Filesystem-only regression tests for Windows cache publication failures."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plugin_cache_commit_test', ROOT / 'code/engine/cache_commit.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def denied(code=5):
    error = PermissionError(f'[WinError {code}] 拒绝访问')
    error.winerror = code
    return error


class CacheCommitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.target = self.base / ('a' * 64)
        self.attempt = self.base / (self.target.name + '.pending-test')
        tiles = self.attempt / 'normalized_tiles'
        tiles.mkdir(parents=True)
        (tiles / 'tile.tif').write_bytes(b'synthetic normalized raster')
        (self.attempt / 'normalization.json').write_text(json.dumps({'outputs': [{'output': str(tiles / 'tile.tif')}]}), encoding='utf8')
        (self.attempt / 'paired_valid_rgb.bin').write_bytes(b'not part of reusable cache')
        self.identity = {'period': '20240106', 'reference': '20250118'}
        cache.seal_prepared(self.attempt, self.target, self.identity)

    def test_transient_windows_errors_retry_then_publish_once(self):
        rename = Path.rename
        errors = [denied(5), denied(32), denied(33)]
        def rename_locked(path, destination):
            if errors:
                raise errors.pop(0)
            return rename(path, destination)
        with patch.object(Path, 'rename', rename_locked), patch.object(cache.time, 'sleep') as sleep:
            cache.commit_prepared(self.attempt, self.target, self.identity)
        self.assertEqual(sleep.call_count, 3)
        self.assertFalse(self.attempt.exists())
        marker = json.loads((self.target / 'complete.json').read_text(encoding='utf8'))
        self.assertEqual(marker['identity'], self.identity)
        self.assertTrue(all(Path(row['path']).is_file() for row in marker['files']))
        self.assertNotIn('paired_valid_rgb.bin', [Path(row['path']).name for row in marker['files']])
        audit = json.loads((self.target / 'normalization.json').read_text(encoding='utf8'))
        self.assertEqual(audit['outputs'][0]['output'], str(self.target / 'normalized_tiles/tile.tif'))

    def test_persistent_lock_keeps_verified_attempt_for_next_submission(self):
        with patch.object(Path, 'rename', side_effect=denied()), patch.object(cache.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'WinError 5'):
                cache.commit_prepared(self.attempt, self.target, self.identity)
        self.assertEqual(sleep.call_count, len(cache.RETRY_DELAYS))
        self.assertFalse((self.target / 'complete.json').exists())
        self.assertEqual(cache.find_prepared(self.target, self.identity), self.attempt)
        cache.commit_prepared(self.attempt, self.target, self.identity)
        self.assertTrue((self.target / 'complete.json').is_file())

    def test_marker_failure_after_rename_is_recoverable(self):
        replace = Path.replace
        def blocked_marker(path, destination):
            if Path(destination).name == 'complete.json':
                raise denied()
            return replace(path, destination)
        with patch.object(Path, 'replace', blocked_marker), patch.object(cache.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                cache.commit_prepared(self.attempt, self.target, self.identity)
        self.assertTrue(self.target.is_dir())
        self.assertFalse((self.target / 'complete.json').exists())
        self.assertEqual(cache.find_prepared(self.target, self.identity), self.target)
        cache.commit_prepared(self.target, self.target, self.identity)
        self.assertTrue((self.target / 'complete.json').is_file())

    def test_changed_or_unfinished_attempt_is_not_reused(self):
        self.assertIsNone(cache.find_prepared(self.target, {'period': 'other'}))
        (self.attempt / 'normalized_tiles/tile.tif').write_bytes(b'corrupt')
        self.assertIsNone(cache.find_prepared(self.target, self.identity))
        with self.assertRaises(ValueError):
            cache.commit_prepared(self.attempt, self.target, self.identity)
        self.assertFalse(self.target.exists())

    def test_existing_destination_is_never_overwritten(self):
        self.target.mkdir()
        marker = self.target / 'owned-by-another-operation.txt'
        marker.write_text('preserve')
        with patch.object(cache.time, 'sleep') as sleep:
            with self.assertRaises(FileExistsError):
                cache.commit_prepared(self.attempt, self.target, self.identity)
        sleep.assert_not_called()
        self.assertEqual(marker.read_text(), 'preserve')

    def test_non_windows_and_unrelated_io_errors_do_not_retry(self):
        for error in (PermissionError('posix permission'), OSError('disk error')):
            with patch.object(Path, 'rename', side_effect=error), patch.object(cache.time, 'sleep') as sleep:
                with self.assertRaises(OSError):
                    cache.commit_prepared(self.attempt, self.target, self.identity)
            sleep.assert_not_called()

    def test_atomic_marker_replace_transient_lock(self):
        replace = Path.replace
        blocked = [True]
        def once(path, destination):
            if Path(destination).name == 'complete.json' and blocked:
                blocked.pop()
                raise denied()
            return replace(path, destination)
        with patch.object(Path, 'replace', once), patch.object(cache.time, 'sleep'):
            cache.commit_prepared(self.attempt, self.target, self.identity)
        self.assertTrue((self.target / 'complete.json').is_file())
        self.assertFalse(list(self.target.glob('.*.tmp')))


if __name__ == '__main__':
    unittest.main()
