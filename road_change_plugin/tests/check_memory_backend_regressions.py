"""Current plugin contracts on shared numerical fixtures, without model runs.

Older development fixtures predate width quality. Give their explicitly known
synthetic widths grade A; missing/C-grade rejection has separate plugin tests.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT.parent/'code/tests')]
import engine
engine.__path__=[str(ROOT/'code/engine')]
import user_pipeline
from test_fast_auto_change import FastFinalAutoTests
from test_fast_auto_v2 import V2Tests
from test_fast_batch_runtime import BatchRuntimeTests
from test_fast2_compensation import CompensationTests


def main():
    original=FastFinalAutoTests.scene
    def scene(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        result.widths['quality_grade']='A'
        return result
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in (BatchRuntimeTests,V2Tests,CompensationTests)])
    with patch.object(FastFinalAutoTests,'scene',scene):
        result=unittest.TextTestRunner(verbosity=1).run(suite)
    return not result.wasSuccessful()


if __name__=='__main__':raise SystemExit(main())
