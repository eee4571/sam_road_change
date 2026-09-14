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
        from collections import Counter
        v=PatchVerifier.__new__(PatchVerifier);v.counts=Counter();v.audit=[]
        v.calibration=dict(count=24,minimum=12)
        for k,median,low,high,mad in [('score_delta',0,-.2,.2,.08),('anomaly',0,-.1,.1,.03),
                ('ncc',.8,.6,.95,.05),('ssim',.8,.6,.95,.05),('hog_distance',.1,0,.2,.04),
                ('left',0,-1,1,.2),('right',0,-1,1,.2)]:
            v.calibration.update({k+'_median':median,k+'_low':low,k+'_high':high,k+'_mad':mad,k+'_count':24})
        v.calibration.update({'0_road_low':.5,'1_road_low':.5})
        return v

    def patch(self):
        return dict(valid=True,resolution=1.,left=np.full(3,4.),right=np.full(3,4.),ncc=.2,ssim=.2,
                    hog_distance=.5,anomaly=.5,score_delta=2.,core_ncc=.2,ring_ssim=.8,
                    registration=dict(accepted=True,response=.9,proposed_shift_px=[1.,0.]),
                    periods=[dict(road_score=.1,p=1.,s=1.),dict(road_score=2.1,p=0.,s=0.)])

    def stable(self):
        p=self.patch();p.update(left=np.zeros(3),right=np.zeros(3),ncc=.9,ssim=.9,hog_distance=.05)
        p['periods'][0]['road_score']=2.
        return p

    def test_raw_change_is_supported_and_symmetric(self):
        p=self.patch();v=self.verifier()
        self.assertEqual(v.reasons(p,'added')[1],'strong_change')
        p['periods'].reverse();p['score_delta']*=-1
        self.assertEqual(v.reasons(p,'removed')[1],'strong_change')

    def test_stability_needs_all_strong_evidence(self):
        v=self.verifier();p=self.stable()
        for kind in ('added','removed','widened','narrowed'):
            self.assertEqual(v.reasons(p,kind)[1],'strong_stable')
        for field,value in [('ncc',.1),('ssim',.1),('hog_distance',.9)]:
            p=self.stable();p.update(score_delta=0,anomaly=0);p[field]=value
            self.assertEqual(v.reasons(p,'added')[1],'uncertain')
        p=self.stable();p['registration']['accepted']=False
        self.assertEqual(v.reasons(p,'added')[1],'uncertain')

    def test_identity_registration_is_not_mistaken_for_failed_registration(self):
        p=self.stable();p['registration'].update(accepted=False,proposed_shift_px=[0.,0.])
        self.assertEqual(self.verifier().reasons(p,'added')[1],'strong_stable')

    def test_model_values_cannot_affect_verdict(self):
        for p in (self.patch(),self.stable()):
            v=self.verifier();expected=v.reasons(p,'added')
            for r in p['periods']:r.update(p=np.nan,s=np.nan)
            self.assertEqual(v.reasons(p,'added'),expected)

    def test_insufficient_change_is_uncertain(self):
        p=self.patch();p.update(anomaly=0,score_delta=0)
        self.assertEqual(self.verifier().reasons(p,'added')[1],'uncertain')
        p['left'][:]=np.nan
        self.assertEqual(self.verifier().reasons(p,'added')[1],'uncertain')

    def test_contrast_reversal_alone_cannot_prove_stability(self):
        p=self.stable();p.update(ncc=-.3,ssim=.01,anomaly=0,score_delta=0)
        self.assertEqual(self.verifier().reasons(p,'added')[1],'uncertain')

    def test_one_or_two_sided_width_change(self):
        p=self.patch();p['periods'][0]['road_score']=2.;v=self.verifier()
        self.assertEqual(v.reasons(p,'widened')[1],'strong_change')
        p['left'][:]=0
        self.assertEqual(v.reasons(p,'widened')[1],'strong_change')
        p['right']*=-1
        self.assertEqual(v.reasons(p,'narrowed')[1],'strong_change')

    def test_translation_requires_resolved_sustained_shift(self):
        p=self.patch();p['periods'][0]['road_score']=2.;p['left']*=-1;v=self.verifier()
        self.assertEqual(v.reasons(p,'widened'),(['raw_boundary_lateral_displacement'],'strong_stable'))
        p['left'][:]=-.2;p['right'][:]=.2;p.update(ncc=.1,ssim=.1)
        self.assertEqual(v.reasons(p,'widened')[1],'uncertain')

    def test_missing_data_or_calibration_is_uncertain(self):
        p=self.patch();p['valid']=False
        self.assertEqual(self.verifier().reasons(p,'added')[1],'uncertain')
        p['valid']=True;v=self.verifier();v.calibration['count']=0
        self.assertEqual(v.reasons(p,'added')[1],'uncertain')
        p['width_geometry_reliable']=False
        self.assertEqual(self.verifier().reasons(p,'widened')[1],'uncertain')

    def test_uncertain_without_strong_fast2_evidence_is_candidate(self):
        p=self.patch();p['valid']=False;v=self.verifier()
        for original in (True,False):
            row=dict(change_typ='added',v2_publish=original,v2_precision_reason='original_fast2')
            v.record_decision(row,p,0)
            self.assertFalse(row['v2_publish'])
            self.assertEqual(row['v2_precision_reason_before_patch'],'original_fast2')
            self.assertEqual(row['v2_publication_level'],'Candidate')
            self.assertEqual(row['v2_patch_state'],'uncertain')
        self.assertFalse(any(k.startswith('veto_') for k in v.counts))

    def test_only_stable_vetoes_and_qa_does_not_count_as_veto(self):
        v=self.verifier();p=self.stable()
        row=dict(change_typ='added',v2_publish=True,v2_precision_reason='original')
        v.record_decision(row,p,0)
        self.assertFalse(row['v2_publish']);self.assertEqual(v.counts['strong_stable'],1)
        self.assertEqual(sum(n for k,n in v.counts.items() if k.startswith('primary_')),1)

    def test_curved_or_unresolved_width_is_not_rejected_as_stable(self):
        p=self.stable();v=self.verifier()
        for axis,width in [('LINESTRING (0 0, 20 0, 20 20)',10),('LINESTRING (0 0, 40 0)',2)]:
            row=dict(change_typ='widened',v2_publish=True,v2_precision_reason='original',
                     width_bef=width,width_aft=width,axis_wkt=axis)
            v.record_decision(row,p,0)
            self.assertFalse(row['v2_publish']);self.assertEqual(row['v2_patch_state'],'uncertain')
            self.assertEqual(row['v2_publication_level'],'Candidate')


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
