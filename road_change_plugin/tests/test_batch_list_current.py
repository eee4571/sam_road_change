"""Current-project resume path repair, with synthetic files and no GIS imports."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'plugin_batch_list_repair_test', ROOT / 'code/app/project_relocation.py')
relocation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = relocation
spec.loader.exec_module(relocation)


class CurrentBatchListTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.project = self.base / 'project'
        self.job = self.project / '_work/current'

    def listing(self, text, job=None, period='20221020', radiometric=True):
        path = (job or self.job) / 'grids/1/periods' / period
        if radiometric:
            path /= 'radiometric/config-key'
        path /= 'batches/grid_tiles.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8-sig')
        return path

    def image(self, relative, project=None):
        path = (project or self.project) / '_work/cache' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic image')
        return path.resolve()

    def test_existing_current_cache_lists_stay_unchanged(self):
        for cache_path in ('irmad/hash/normalized_tiles/v0001.tif',
                           'normalized/20221020/key/v0001.tif'):
            for radiometric in (True, False):
                with self.subTest(cache=cache_path, radiometric=radiometric):
                    image = self.image(cache_path)
                    listing = self.listing(str(image) + '\n', radiometric=radiometric)
                    original = listing.read_bytes()
                    result = relocation.repair_task_batch_lists(self.job)
                    self.assertEqual(result.modified_paths, 0)
                    self.assertEqual(listing.read_bytes(), original)
                    self.assertFalse(listing.with_name(listing.name + relocation.BACKUP_SUFFIX).exists())

    def test_moved_cache_list_keeps_key_order_blanks_and_bom(self):
        first = self.image('irmad/hash-a/normalized_tiles/v0002.tif')
        second = self.image('irmad/hash-a/normalized_tiles/v0001.tif')
        # Same filenames in another cache entry must not be substituted.
        self.image('irmad/hash-b/normalized_tiles/v0001.tif')
        old = ('Z:\\old-project\\_work\\cache\\irmad\\hash-a\\normalized_tiles\\v0002.tif\n\n'
               'Z:\\old-project\\_work\\cache\\irmad\\hash-a\\normalized_tiles\\v0001.tif\n')
        listing = self.listing(old)
        result = relocation.repair_task_batch_lists(self.job)
        self.assertEqual((result.modified_lists, result.modified_paths), (1, 2))
        self.assertEqual(listing.read_text(encoding='utf-8-sig'), f'{first}\n\n{second}\n')
        self.assertTrue(listing.read_bytes().startswith(b'\xef\xbb\xbf'))
        self.assertEqual(relocation.repair_task_batch_lists(self.job).modified_paths, 0)

    def test_missing_cache_never_uses_external_or_same_name_images(self):
        external = self.image('irmad/hash-a/normalized_tiles/v0001.tif', self.base / 'other')
        self.image('irmad/wrong-key/normalized_tiles/v0001.tif')
        listing = self.listing(str(external) + '\n')
        local_image = listing.parent.parent / 'images/v0001.tif'
        local_image.parent.mkdir()
        local_image.write_bytes(b'wrong image')
        original = listing.read_bytes()
        with self.assertRaises(FileNotFoundError) as raised:
            relocation.repair_task_batch_lists(self.job)
        expected = self.project / '_work/cache/irmad/hash-a/normalized_tiles/v0001.tif'
        self.assertIn(str(expected.resolve()), str(raised.exception))
        self.assertEqual(listing.read_bytes(), original)

    def test_all_lists_are_checked_before_any_rewrite(self):
        image = self.image('normalized/20221020/key/v0001.tif')
        first = self.listing('Z:\\old\\_work\\cache\\normalized\\20221020\\key\\v0001.tif\n')
        second = self.listing('Z:\\old\\_work\\cache\\irmad\\missing\\normalized_tiles\\v0001.tif\n', period='20230222')
        originals = {p: p.read_bytes() for p in (first, second)}
        with self.assertRaises(FileNotFoundError):
            relocation.repair_task_batch_lists(self.job)
        self.assertTrue(image.is_file())
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse(path.with_name(path.name + relocation.BACKUP_SUFFIX).exists())

    def test_cache_path_traversal_is_rejected(self):
        listing = self.listing('Z:\\old\\_work\\cache\\irmad\\..\\..\\..\\outside.tif\n')
        original = listing.read_bytes()
        with self.assertRaisesRegex(ValueError, '逃出当前任务'):
            relocation.repair_task_batch_lists(self.job)
        self.assertEqual(listing.read_bytes(), original)

    def test_period_images_and_legacy_job_layout_still_work(self):
        listing = self.listing('Z:\\old\\images\\v0001.tif\n', radiometric=False)
        local_image = listing.parent.parent / 'images/v0001.tif'
        local_image.parent.mkdir()
        local_image.write_bytes(b'image')
        self.assertEqual(relocation.repair_task_batch_lists(self.job).modified_paths, 1)
        self.assertEqual(listing.read_text(encoding='utf-8-sig').strip(), str(local_image.resolve()))
        legacy = self.project / '_work/tasks/runs/run-old'
        image = self.image('normalized/20221020/key/v0001.tif')
        listing = self.listing(str(image) + '\n', job=legacy)
        self.assertEqual(relocation.repair_task_batch_lists(legacy).modified_paths, 0)
        self.assertEqual(listing.read_text(encoding='utf-8-sig').strip(), str(image))


if __name__ == '__main__':
    unittest.main()
