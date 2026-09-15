import unittest
import numpy as np
from irmad_rrn import uniform_iteration_sample


class SamplingTests(unittest.TestCase):
    def test_uniform_deterministic_cap(self):
        values=np.arange(101)
        first=uniform_iteration_sample(values,10)
        np.testing.assert_array_equal(first,uniform_iteration_sample(values,10))
        self.assertEqual(len(first),10)
        self.assertEqual(len(np.unique(first)),10)
        self.assertLessEqual(np.diff(first).max()-np.diff(first).min(),1)
        self.assertLess(first[0],11)
        self.assertGreater(first[-1],89)

    def test_small_population_unchanged(self):
        values=np.arange(15)
        self.assertIs(uniform_iteration_sample(values,4_000_000),values)
        self.assertIs(uniform_iteration_sample(values,None),values)


if __name__=='__main__':unittest.main()
