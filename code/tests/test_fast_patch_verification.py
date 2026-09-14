import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from engine.fast_patch_verification import PatchVerifier,ImageTiles,road_boundaries


class DecisionTests(unittest.TestCase):
    def verifier(self):
        v=PatchVerifier.__new__(PatchVerifier)
        v.calibration=dict(count=24,minimum=12)
        for k,median,low,high,mad in [('score_delta',0,-.2,.2,.08),('anomaly',0,-.1,.1,.03),
                ('ncc',.8,.6,.95,.05),('ssim',.8,.6,.95,.05),('hog_distance',.1,0,.2,.04),
                ('left',0,-1,1,.2),('right',0,-1,1,.2)]:
            v.calibration.update({k+'_median':median,k+'_low':low,k+'_high':high,k+'_mad':mad,k+'_count':24})
        v.calibration.update({'0_road_low':.5,'1_road_low':.5})
        return v

    def patch(self):
        return dict(valid=True,resolution=1.,left=np.full(3,4.),right=np.full(3,4.),ncc=.2,ssim=.2,
                    hog_distance=.5,anomaly=.5,score_delta=2.,
                    periods=[dict(road_score=.1,p=1.,s=1.),dict(road_score=2.1,p=0.,s=0.)])

    def test_raw_appearance_overrides_opposite_model_support(self):
        self.assertEqual(self.verifier().reasons(self.patch(),'added'),([],'verified'))

    def test_removed_is_symmetric(self):
        p=self.patch();p['periods'].reverse();p['score_delta']*=-1
        self.assertEqual(self.verifier().reasons(p,'removed'),([],'verified'))

    def test_model_changes_cannot_affect_verdict(self):
        p=self.patch();v=self.verifier();expected=v.reasons(p,'added')
        for r in p['periods']:r.update(p=np.nan,s=np.nan)
        self.assertEqual(v.reasons(p,'added'),expected)
        p.update(score_delta=0,ncc=.9,ssim=.9,hog_distance=.05,anomaly=0)
        p['periods'][0]['road_score']=2.
        for r in p['periods']:r.update(p=0,s=0)
        self.assertIn('persistent_raw_road_structure',v.reasons(p,'added')[0])

    def test_seasonal_ring_change_cannot_confirm_road_change(self):
        p=self.patch();p['anomaly']=0
        self.assertIn('normal_background_relative_change',self.verifier().reasons(p,'added')[0])

    def test_source_needs_parallel_edges(self):
        p=self.patch();p['periods'][1]['road_score']=.1
        self.assertIn('raw_parallel_boundaries_not_supported',self.verifier().reasons(p,'added')[0])

    def test_contrast_reversal_does_not_create_road_with_stable_edges(self):
        p=self.patch();p['periods'][0]['road_score']=2.
        p.update(ncc=-.3,ssim=.01,left=np.zeros(3),right=np.ones(3))
        self.assertIn('persistent_raw_parallel_boundaries',self.verifier().reasons(p,'added')[0])

    def test_bilateral_width_change(self):
        p=self.patch();p['periods'][0]['road_score']=2.
        self.assertEqual(self.verifier().reasons(p,'widened'),([],'verified'))
        p['left']*=-1;p['right']*=-1
        self.assertEqual(self.verifier().reasons(p,'narrowed'),([],'verified'))

    def test_translation_and_one_sided_width_not_published(self):
        p=self.patch();p['periods'][0]['road_score']=2.;p['left']*=-1
        self.assertIn('raw_boundary_lateral_displacement',self.verifier().reasons(p,'widened')[0])
        p['left'][:]=0
        self.assertIn('raw_bilateral_change_not_sustained',self.verifier().reasons(p,'widened')[0])

    def test_unavailable_image_or_calibration_is_not_model_fallback(self):
        p=self.patch();p['valid']=False
        self.assertEqual(self.verifier().reasons(p,'added')[1],'unconfirmed_image')
        p['valid']=True;v=self.verifier();v.calibration['count']=0
        self.assertEqual(v.reasons(p,'added')[1],'unconfirmed_image')


class RasterTests(unittest.TestCase):
    def test_parallel_surface_is_not_a_width_expansion(self):
        yy,xx=np.indices((32,90));bins=xx//30;lateral=yy-10
        before=(yy>=8)&(yy<=12);after=before|((yy>=20)&(yy<=25))
        anchor=yy==10;outer=np.ones_like(before)
        np.testing.assert_array_equal(road_boundaries(before,anchor,outer,bins,lateral),
                                      road_boundaries(after,anchor,outer,bins,lateral))

    def test_png_uses_image_georeference_and_nodata_is_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);image=root/'image.tif';mask=root/'mask.png'
            transform=from_origin(500000,4000020,1,1)
            with rasterio.open(image,'w',driver='GTiff',height=20,width=20,count=3,dtype='uint8',crs=32650,transform=transform) as ds:
                ds.write(np.full((3,20,20),100,np.uint8))
            array=np.zeros((20,20),np.uint8);array[:,8:12]=255
            with rasterio.open(mask,'w',driver='PNG',height=20,width=20,count=1,dtype='uint8') as ds:ds.write(array,1)
            (root/'tile_summary.json').write_text(json.dumps(dict(image=str(image),molra_surface_mask=str(mask))),encoding='utf8')
            tiles=ImageTiles(dict(width_review=str(root)),32650)
            try:
                rgb,surface=tiles.read((500000,4000000,500020,4000020),(20,20),transform)
                np.testing.assert_array_equal(surface,array>0)
                np.testing.assert_array_equal(rgb,100)
                rgb,surface=tiles.read((600000,4000000,600020,4000020),(20,20),from_origin(600000,4000020,1,1))
                self.assertTrue(np.isnan(rgb).all());self.assertTrue(np.isnan(surface).all())
            finally:tiles.close()


if __name__=='__main__':unittest.main()
