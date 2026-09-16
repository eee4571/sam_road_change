"""Explicit backend smoke: tiny generated RGB images, no models or real projects.

Run with the plugin backend Python. Existing numerical regressions are loaded
from the development repository, while every engine import uses the plugin copy.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT.parent / 'code/tests')]
from engine import cache_commit, irmad_preprocessing as rrn
import test_irmad_preprocessing as regressions


class RecoveryTests(unittest.TestCase):
    setUp = regressions.IRMADTests.setUp

    def test_denied_commit_then_resume_does_not_normalize_twice(self):
        error = PermissionError('[WinError 5] 拒绝访问')
        error.winerror = 5
        with patch.object(Path, 'rename', side_effect=error), patch.object(cache_commit, 'RETRY_DELAYS', ()):
            with self.assertRaisesRegex(RuntimeError, 'WinError 5'):
                rrn.prepare_period('20240106', self.sources, self.root / 'cache', enabled=True)
        self.assertEqual(len(list((self.root / 'cache').glob('*.pending-*/prepared.json'))), 1)
        with patch.object(rrn, 'normalize_pair', side_effect=AssertionError('unnecessary recalculation')):
            result = rrn.prepare_period('20240106', self.sources, self.root / 'cache', enabled=True)
            self.assertTrue((result.source / 'v0001.tif').is_file())
            again = rrn.prepare_period('20240106', self.sources, self.root / 'cache', enabled=True)
            self.assertEqual(again.metadata['status'], 'cache_hit')

    def test_diagnostic_write_error_does_not_mask_original_failure(self):
        save = rrn.save
        def fail_diagnostic(path, value):
            if Path(path).name == 'failure.json':
                raise PermissionError('diagnostic denied')
            return save(path, value)
        # Make an attempt directory before failing, as a real normalization would.
        def normalize(pairs, attempt, **kwargs):
            attempt.mkdir(parents=True)
            raise RuntimeError('original failure')
        with patch.object(rrn, 'normalize_pair', normalize), patch.object(rrn, 'save', fail_diagnostic):
            with self.assertRaisesRegex(RuntimeError, 'original failure'):
                rrn.prepare_period('20240106', self.sources, self.root / 'cache', enabled=True)


if __name__ == '__main__':
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(regressions.IRMADTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)])
    assert Path(rrn.__file__).resolve().is_relative_to(ROOT / 'code')
    raise SystemExit(not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful())
