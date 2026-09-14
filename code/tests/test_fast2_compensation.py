"""Independent switch contracts and production cache isolation, no model runs."""
import argparse
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

from engine.fast2_compensation import Fast2CompensationConfig as Config, Fast2Compensation, cache_matches
from engine.fast_multitemporal import AUTO_REVISION
from engine.fast_image_structure import align_pair, image_features
import test_fast_image_structure as image_fixture


class CompensationTests(unittest.TestCase):
    def test_presets_validation_and_each_switch_identity(self):
        raw=Config.from_preset('raw_input');normalized=Config.from_preset('normalized_input')
        self.assertTrue(all(raw.to_dict().values()))
        self.assertFalse(any(normalized.to_dict().values()))
        keys={raw.cache_identity,normalized.cache_identity}
        for name in raw.to_dict():
            custom=Config.from_preset('custom',**{name:False})
            self.assertEqual(sum(custom.to_dict().values()),3)
            keys.add(custom.cache_identity)
        self.assertEqual(len(keys),6)
        self.assertEqual(Config.resolve('{"preset":"custom","stable_appearance_calibration":false}'),
                         replace(raw,stable_appearance_calibration=False))
        with self.assertRaises(TypeError):Config(patch_radiometric_normalization='false')
        with self.assertRaises(ValueError):Config.from_preset('raw_input',stable_appearance_calibration=False)

    def test_radiometric_off_preserves_values_and_geometric_registration(self):
        import cv2
        t=image_fixture.RawImageTests();a,lateral,bins,normal=t.data()
        module=Fast2Compensation('normalized_input').patch
        rgb=np.repeat((a*50+17)[None],3,0)
        gray,valid=module.grayscale(rgb)
        np.testing.assert_array_equal(gray,np.mean(rgb,axis=0))
        self.assertTrue(valid.all())
        b=a*.8+.1
        actual,info=module.apply(a,b,np.ones(a.shape,bool))
        self.assertIs(actual,b);self.assertEqual(info,dict(gain=1.,offset=0.))
        shifted=cv2.warpAffine(a,np.float32([[1,0,1.5],[0,1,-1]]),(128,96))
        _,_,registration=align_pair(a,shifted,valid,valid,np.abs(lateral)>18,radiometric=module)
        self.assertTrue(registration['accepted']);self.assertEqual(registration['gain'],1.)
        with patch('engine.fast_image_structure.describe',wraps=__import__(
                'engine.fast_image_structure',fromlist=['describe']).describe) as describe:
            features=image_features(np.repeat(a[None],3,0),np.repeat(shifted[None],3,0),
                np.abs(lateral)<7,np.abs(lateral)>18,lateral,bins,normal,16,1.,radiometric=module)
        self.assertEqual(describe.call_count,2)
        self.assertIn('left',features);self.assertIn('hog_distance',features)

    def test_disabled_width_is_zero_correction_without_profile_mutation(self):
        samples=np.arange(60,dtype=float)%3+4.;saved=samples.tobytes()
        on=Fast2Compensation().width.fit(samples)
        off=Fast2Compensation('normalized_input').width.fit(samples)
        self.assertEqual(on['bias'],5.);self.assertGreater(on['scatter'],0.)
        self.assertEqual(off['bias'],0.);self.assertEqual(off['scatter'],0.)
        self.assertEqual(samples.tobytes(),saved)

    def test_surface_probability_has_no_invented_scale_correction(self):
        p=np.array([0.,.5,np.nan]);s=np.array([1.,0.,np.nan])
        for preset in ('raw_input','normalized_input'):
            result=Fast2Compensation(preset).surface_probability.apply(p,s)
            self.assertIs(result[0],p);self.assertIs(result[1],s)

    def test_disabled_appearance_is_identity_reference_not_learned_distribution(self):
        controls=[dict(score_delta=3.,anomaly=.4,ncc=.7,ssim=.6,hog_distance=.2,
                       left=np.ones(3)*2,right=np.ones(3)*4,scores=[2.,3.]) for _ in range(24)]
        on=Fast2Compensation().appearance.fit(controls)
        off=Fast2Compensation('normalized_input').appearance.fit(controls)
        self.assertEqual(on['score_delta_median'],3.)
        for name in ('left','right','score_delta','anomaly','hog_distance'):
            self.assertEqual(off[name+'_median'],0.)
            self.assertEqual(off[name+'_mad'],0.)
        self.assertEqual(off['ncc_median'],1.)
        self.assertEqual(off['count'],24)
        self.assertEqual(Fast2Compensation('normalized_input').appearance.fit([])['count'],0)

    def test_cache_switches_never_cross_reuse(self):
        raw=Config();normalized=Config.from_preset('normalized_input')
        legacy={'fast_auto_revision':AUTO_REVISION}
        self.assertTrue(cache_matches(legacy,raw));self.assertFalse(cache_matches(legacy,normalized))
        for source in (raw,normalized):
            result=dict(legacy,fast2_compensation_identity=source.cache_identity)
            for target in (raw,normalized):self.assertEqual(cache_matches(result,target),source==target)

    def test_formal_entry_writes_effective_config_and_keeps_raw_candidates(self):
        import json
        import geopandas as gpd
        import test_fast_auto_change as fixtures
        from engine.fast_pipeline import detect_fast_changes
        from engine import fast_auto_change as baseline
        f=fixtures.FastFinalAutoTests();f.setUp();self.addCleanup(f.doCleanups)
        scenes=[f.scene([f.road(70)]),f.scene([f.road(70),f.road(170)])]
        inputs=[]
        for i,scene in enumerate(scenes):
            payload={'road_probability':scene.probability.dataset.name}
            frames=dict(centerlines=scene.widths,surfaces=scene.surfaces,width_segments=scene.widths,
                        valid_observation=gpd.GeoDataFrame(geometry=[scene.valid],crs=f.crs))
            for name,frame in frames.items():
                path=Path(f.tmp.name)/f'{i}_{name}.gpkg';frame.to_file(path);payload[name]=str(path)
            inputs.append(payload)
        with patch.object(baseline,'analyze_scenes',side_effect=AssertionError('Fast1 invoked')), \
             patch.object(baseline.RoadScene,'match',side_effect=AssertionError('station match invoked')):
            for i,config in enumerate((Config(),Config.from_preset('normalized_input'),
                                      Config.from_preset('custom',width_temporal_bias_correction=False))):
                out=Path(f.tmp.name)/f'output{i}'
                result=detect_fast_changes(*inputs,out,compensation=config)
                self.assertEqual(result['fast2_compensation_identity'],config.cache_identity)
                audit=json.loads((out/'patch_verification.json').read_text(encoding='utf8'))
                self.assertEqual(audit['compensation']['config'],config.to_dict())
                self.assertGreater(result['performance']['v2_probability_event_locations'],0)
                self.assertEqual(result['performance']['v2_station_count'],0)
                self.assertTrue(audit['candidates'])
                self.assertTrue(all(c['state']=='uncertain' for c in audit['candidates']))
                self.assertTrue(all(c['publication_level']=='Probable' for c in audit['candidates']))

    def test_cli_presets_do_not_require_new_existing_arguments(self):
        import user_pipeline as p
        args=p.parser().parse_args(['change','--before-result','a','--after-result','b','--output','out',
                                   '--fast2-compensation','normalized_input'])
        self.assertEqual(args.fast2_compensation,'normalized_input')

    def test_batch_resume_config_change_only_recomputes_changes(self):
        import user_pipeline as p
        from app.result_publisher import ResultPublisher
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);source=root/'input';source.mkdir()
            args=argparse.Namespace(source_root=str(source),output_root=str(root/'results'),run_id='run',
                checkpoint='model.ckpt',config='config.yaml',device='cpu',pixel_size='0',rescale='off',
                absolute='2',ratio='.2',tolerance='3',execution_profile='fast')
            def pair(*a,**kw):
                cfg=kw['compensation']
                return dict(fast_auto_revision=AUTO_REVISION,fast2_compensation=cfg.to_dict(),
                            fast2_compensation_identity=cfg.cache_identity)
            def finalize(m,job):
                m['final_period_results']=m['period_results'];m['temporal_results']=[{'grid':'g'}]
                m['fast_finalization_state']='completed'
            with patch.object(p,'discover_grid_periods',return_value={'g':{x:source/f'{x}.txt' for x in ('1','2')}}), \
                 patch.object(p,'prepare') as prepare,patch.object(p,'extract',return_value={}) as extract, \
                 patch.object(p,'_run_fast_change_result',side_effect=pair) as detect, \
                 patch.object(p,'_finalize_fast_manifest',side_effect=finalize), \
                 patch.object(p,'_period_result_ready',return_value=True),patch.object(p,'_change_result_ready',return_value=True), \
                 patch.object(p,'_temporal_result_ready',return_value=True),patch.object(ResultPublisher,'publish_manifest'), \
                 patch.object(ResultPublisher,'publish_reports'),patch.object(p,'aggregate_change_evaluations'):
                p.run_all(args)
                args.resume=True;args.fast2_compensation='normalized_input'
                prepare.reset_mock();extract.reset_mock();detect.reset_mock()
                changed=p.run_all(args)
                self.assertEqual(detect.call_count,1)
                self.assertFalse(any(changed['change_results'][0]['fast2_compensation'].values()))
                prepare.assert_not_called();extract.assert_not_called()
                args.fast2_compensation=None;detect.reset_mock()
                resumed=p.run_all(args)
                detect.assert_not_called()
                self.assertEqual(resumed['input_spec']['fast2_compensation'],Config.from_preset('normalized_input').to_dict())


if __name__=='__main__':unittest.main()
