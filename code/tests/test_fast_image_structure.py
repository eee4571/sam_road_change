import unittest
import cv2
import numpy as np
from shapely.geometry import LineString
from engine.fast_image_structure import axis_grid,align_pair,image_features


class RawImageTests(unittest.TestCase):
    def data(self):
        yy,xx=np.indices((96,128));xy=np.stack((xx,95-yy),axis=-1).astype(float)
        axis=LineString([(4,48),(123,48)])
        lateral,bins,normal=axis_grid(axis,xy)
        rng=np.random.default_rng(432)
        gray=cv2.GaussianBlur(rng.random((96,128)).astype(np.float32),(5,5),0)*.35
        return gray,lateral,bins,normal

    def features(self,a,b,lateral,bins,normal):
        return image_features(np.repeat(a[None],3,0),np.repeat(b[None],3,0),
            np.abs(lateral)<7,np.abs(lateral)>18,lateral,bins,normal,16,1.)

    def test_gain_offset_preserves_structure(self):
        a,lateral,bins,normal=self.data();a[np.abs(lateral)<6]+=.6
        p=self.features(a,a*.8+.1,lateral,bins,normal)
        self.assertGreater(p['ncc'],.99);self.assertGreater(p['ssim'],.99)
        np.testing.assert_array_equal(p['left'],0);np.testing.assert_array_equal(p['right'],0)

    def test_both_edges_move_outward_in_raw_rgb(self):
        a,lateral,bins,normal=self.data();b=a.copy()
        a[np.abs(lateral)<5]+=.6;b[np.abs(lateral)<10]+=.6
        p=self.features(a,b,lateral,bins,normal)
        self.assertTrue(np.all(p['left']>2));self.assertTrue(np.all(p['right']>2))

    def test_road_appearance_strength_increases(self):
        a,lateral,bins,normal=self.data();b=a.copy();b[np.abs(lateral)<6]+=.6
        p=self.features(a,b,lateral,bins,normal)
        self.assertGreater(p['score_delta'],0);self.assertGreater(p['anomaly'],0)

    def test_registration_only_small_translation(self):
        a,lateral,_,_=self.data();valid=np.ones(a.shape,bool);ring=np.abs(lateral)>18
        b=cv2.warpAffine(a,np.float32([[1,0,1.5],[0,1,-1]]),(128,96))
        corrected,_,info=align_pair(a,b,valid,valid,ring)
        self.assertTrue(info['accepted']);self.assertLess(np.mean((a[ring]-corrected[ring])**2),np.mean((a[ring]-b[ring])**2))
        b=cv2.warpAffine(a,np.float32([[1,0,9],[0,1,9]]),(128,96))
        self.assertFalse(align_pair(a,b,valid,valid,ring)[2]['accepted'])

    def test_no_valid_pixels_do_not_become_zero_evidence(self):
        a,lateral,bins,normal=self.data();a[:]=np.nan
        self.assertFalse(self.features(a,a,lateral,bins,normal)['image_valid'])

    def test_curve_coordinates_follow_local_segment(self):
        axis=LineString([(0,0),(10,0),(10,10)])
        xy=np.array([[[5,2],[8,5]]],dtype=float)
        lateral,_,normal=axis_grid(axis,xy)
        np.testing.assert_allclose(lateral,[[2,2]])
        np.testing.assert_allclose(normal,[[[0,1],[-1,0]]])


if __name__=='__main__':unittest.main()
