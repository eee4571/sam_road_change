"""Tiny image roundtrip, without inference or model imports."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np


class LongPathImageTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='win32','Windows image I/O adapter')
    def test_deep_unicode_png_roundtrip(self):
        spec=importlib.util.spec_from_file_location('image_io_adapter',Path(os.environ.get('SAMROAD_TEST_CODE_ROOT',Path(__file__).resolve().parents[1]))/'sitecustomize.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            extended=Path(adapter._filesystem_path(str(Path(folder)/('road_'*20)/('width_'*20)/'中文影像')))
            extended.mkdir(parents=True,exist_ok=True)
            path=extended/'enhanced_probability.png'
            plain=str(path).removeprefix(chr(92)*2+'?'+chr(92))
            self.assertGreater(len(plain),260)
            pixels=np.arange(300,dtype=np.uint8).reshape(10,10,3)
            self.assertTrue(cv2.imwrite(plain,pixels))
            np.testing.assert_array_equal(cv2.imread(plain),pixels)
            path.unlink()
            extended.rmdir()
            extended.parent.rmdir()
            extended.parent.parent.rmdir()


if __name__=='__main__':unittest.main()
