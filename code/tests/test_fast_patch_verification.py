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
        v.calibration=dict(count=20,minimum=12,p_delta_low=-.2,p_delta_high=.2,s_delta_low=-.2,s_delta_high=.2,
            ncc_low=.3,core_ncc_low=.3,left_median=0,right_median=0,boundary_delta_low=-1.,boundary_delta_high=1.)
        for side in range(2):
            for feature,value in [('p',.4),('s',.6),('contrast',.1),('edge',.1),('alignment',.7)]:
                v.calibration[f'{side}_{feature}_low']=value
                v.calibration[f'{side}_{feature}_count']=20
        return v

    def patch(self):
        return dict(valid=True,straight=True,resolution=1.,left=np.full(3,4.),right=np.full(3,4.),ncc=0.,core_ncc=.8,
                    periods=[dict(p=.05,contrast=.02,s=0.,edge=.01,alignment=.4),
                             dict(p=.9,contrast=.5,s=1.,edge=.3,alignment=.9)])

    def test_true_presence_loss_is_not_vetoed(self):
        v=self.verifier();p=self.patch()
        self.assertEqual(v.reasons(p,'added'),([],'verified'))
        p['periods'].reverse()
        self.assertEqual(v.reasons(p,'removed'),([],'verified'))

    def test_probability_and_surface_support_without_axis(self):
        v=self.verifier();p=self.patch()
        p['periods'][0].update(p=.85,contrast=.4,s=.9)
        reasons,state=v.reasons(p,'added')
        self.assertIn('opposite_probability_support',reasons)
        self.assertIn('opposite_molra_surface_support',reasons)
        self.assertEqual(state,'extraction_fluctuation')

    def test_structural_image_support(self):
        v=self.verifier();p=self.patch();p['ncc']=.8
        p['periods'][0].update(p=.5,edge=.3,alignment=.9)
        self.assertIn('persistent_image_road_structure',v.reasons(p,'added')[0])

    def test_model_conflict_does_not_restore_image_veto(self):
        v=self.verifier();p=self.patch();p['ncc']=.8
        p['periods'][0].update(edge=.3,alignment=.9)
        self.assertEqual(v.reasons(p,'added'),(['persistent_image_road_structure'],'extraction_fluctuation'))
        p['periods'].reverse()
        self.assertEqual(v.reasons(p,'removed'),(['persistent_image_road_structure'],'extraction_fluctuation'))

    def test_unchanged_background_cannot_veto_changed_road_interior(self):
        v=self.verifier();p=self.patch();p.update(ncc=.9,core_ncc=.05)
        p['periods'][0].update(edge=.3,alignment=.9)
        self.assertNotIn('persistent_image_road_structure',v.reasons(p,'added')[0])

    def test_boundary_translation_not_widening(self):
        v=self.verifier();p=self.patch();p.update(left=np.full(3,-4.),right=np.full(3,4.))
        self.assertIn('surface_lateral_displacement',v.reasons(p,'widened')[0])

    def test_sustained_one_sided_expansion_is_valid(self):
        v=self.verifier();p=self.patch();p['left']=np.zeros(3)
        self.assertEqual(v.reasons(p,'widened'),([],'verified'))

    def test_temporal_surface_scale_is_calibrated(self):
        v=self.verifier();v.calibration.update(boundary_delta_low=6,boundary_delta_high=10)
        self.assertIn('normal_temporal_surface_scale',v.reasons(self.patch(),'widened')[0])

    def test_missing_calibration_does_not_delete_candidate(self):
        v=self.verifier();v.calibration['count']=0
        self.assertEqual(v.reasons(self.patch(),'added')[0],[])
        v=self.verifier();v.calibration['boundary_delta_high']=np.nan
        self.assertEqual(v.reasons(self.patch(),'widened')[0],[])

    def test_observed_empty_surface_cannot_confirm_width_change(self):
        v=self.verifier();p=self.patch();p['left'][:]=np.nan
        for period in p['periods']:period['surface_observed']=True
        self.assertEqual(v.reasons(p,'widened'),(['insufficient_road_boundary_support'],'unconfirmed_width'))


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
