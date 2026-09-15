"""Synthetic ground-truth checks for transient rejection and preservation of change."""
import unittest
import numpy as np
import pandas as pd
from reconstruct_width import ReconstructionConfig, reconstruct_chain


def chain(left, right=None):
    left = np.asarray(left, float)
    right = np.full(len(left), 6.) if right is None else np.asarray(right, float)
    n = len(left)
    return pd.DataFrame(dict(road_id='synthetic', sample_id=np.arange(n), s_m=np.arange(n)*3.,
        center_x=np.arange(n)*3., center_y=0., normal_x=0., normal_y=1., flags='', accepted=True,
        optimized_left_distance=left, optimized_right_distance=right, optimized_width=left+right,
        optimized_left_confidence=.8, optimized_right_confidence=.8, optimized_confidence=.8))


class ReconstructionTests(unittest.TestCase):
    def run_chain(self, frame):
        result = reconstruct_chain(frame, ReconstructionConfig())
        np.testing.assert_allclose(result.final_width,
                                   result.final_left_distance+result.final_right_distance, equal_nan=True)
        self.assertTrue(result.solver_converged.all())
        return result

    def test_unilateral_returning_plateaus_wide_and_narrow(self):
        for value in (1., 12.):
            left = np.full(65, 5.)
            left[25:31] = value
            out = self.run_chain(chain(left))
            self.assertTrue(out.left_outlier_reason.iloc[25:31].ne('').all())
            self.assertTrue(out.right_outlier_reason.eq('').all())
            np.testing.assert_allclose(out.final_left_distance, 5., atol=.05)
            np.testing.assert_allclose(out.final_right_distance, 6., atol=.05)

    def test_ramp_and_sustained_step_survive(self):
        for left in (np.linspace(2, 12, 90), np.r_[np.full(40, 3.), np.full(50, 9.)]):
            out = self.run_chain(chain(left))
            self.assertFalse(out.outlier_reason.ne('').any())
            self.assertLess(np.abs(out.final_left_distance-left).mean(), .15)
            self.assertGreater(out.final_left_distance.iloc[-1]-out.final_left_distance.iloc[0], 5.)

    def test_opposing_side_errors_are_detected_even_when_width_is_constant(self):
        left, right = np.full(65, 5.), np.full(65, 8.)
        left[25:30] += 4
        right[25:30] -= 4
        out = self.run_chain(chain(left, right))
        self.assertTrue(out.left_outlier_reason.iloc[25:30].ne('').all())
        self.assertTrue(out.right_outlier_reason.iloc[25:30].ne('').all())
        np.testing.assert_allclose(out.final_left_distance, 5., atol=.05)

    def test_sustained_bounded_widening_is_not_a_short_excursion(self):
        left = np.r_[np.full(30, 3.), np.full(30, 8.), np.full(30, 3.)]
        out = self.run_chain(chain(left))
        self.assertTrue(out.outlier_reason.eq('').all())
        self.assertGreater(out.final_left_distance.iloc[40:50].mean(), 7.8)

    def test_continuous_polygons_recover_a_rejected_gap_with_asymmetric_sides(self):
        from render_reconstruction import build_continuous_surfaces
        f = chain(np.full(15, 2.))
        f.loc[5:8, 'accepted'] = False
        out = self.run_chain(f)
        surfaces, boundaries, invalid = build_continuous_surfaces(out, 'EPSG:32650')
        self.assertEqual(len(surfaces), 14)
        self.assertEqual(len(boundaries), 2)
        self.assertFalse(invalid)
        np.testing.assert_allclose(surfaces.area, 24.)
        self.assertTrue(surfaces.geometry.is_valid.all())
        self.assertTrue(surfaces.width_source.eq('interpolated').any())

    def test_short_medium_long_gaps_and_end_propagation(self):
        f = chain(np.linspace(3, 7, 120))
        for a, b in [(0, 5), (12, 15), (30, 44), (60, 95)]:
            f.loc[a:b-1, 'accepted'] = False
        out = self.run_chain(f)
        self.assertTrue(out.reconstruction_available.all())
        self.assertTrue(out.width_source.iloc[12:15].eq('interpolated').all())
        self.assertTrue(out.width_source.iloc[30:44].eq('interpolated').all())
        self.assertTrue(out.width_source.iloc[60:95].eq('propagated').all())
        self.assertLess(out.final_confidence.iloc[77], out.final_confidence.iloc[60])
        self.assertLess(np.abs(np.diff(out.final_width)).max(), .6)

    def test_low_confidence_conflict_and_junction_step(self):
        f = chain(np.r_[np.full(40, 3.), np.full(40, 9.)])
        f.loc[35:45, 'flags'] = 'junction'
        f.loc[35:45, 'accepted'] = False
        f.loc[15, 'optimized_left_distance'] = 15.
        f.loc[15, 'optimized_left_confidence'] = .12
        out = self.run_chain(f)
        self.assertNotEqual(out.left_outlier_reason.iloc[15], '')
        self.assertGreater(out.final_left_distance.iloc[41]-out.final_left_distance.iloc[38], 5)
        self.assertLess(out.final_confidence.iloc[39], .8)

    def test_no_anchors_remain_unresolved_and_image_boundary_is_not_evidence(self):
        f = chain(np.full(30, 4.))
        f['accepted'] = False
        f['flags'] = 'image_or_nodata_boundary'
        out = reconstruct_chain(f, ReconstructionConfig())
        self.assertTrue(out.final_width.isna().all())
        self.assertTrue(out.width_source.eq('unresolved').all())
        self.assertTrue(out.final_confidence.eq(0).all())


if __name__ == '__main__':
    unittest.main()
