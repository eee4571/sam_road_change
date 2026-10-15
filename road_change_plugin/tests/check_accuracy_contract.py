"""Known geometric offsets and lossless production correction, without inference."""
import json
import argparse
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code/engine/width')]
import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, box
from shapely.affinity import translate
from engine.fast_gt_reconciliation import GTProfile, perturb_truth, correct_changes, _frame
from engine.auto_change_geometry import FinalWidths
from engine.gt_road_geometry import compact_road_fit, matched_final_roads
from road_change_detection import evaluate_fast_assisted_centerline_metrics
import user_pipeline as pipeline


def frame(rows):
    return _frame(rows, 3857)


class AccuracyTests(unittest.TestCase):
    def evaluate(self, shift=0., empty=False, geographic=False):
        truth = frame([dict(BHBM=2, geometry=box(100, 100, 200, 110))])
        predicted = frame([dict(change_typ='added', geometry=box(100, 100+shift, 200, 110+shift))])
        if geographic:
            truth = truth.to_crs(4326)
        return evaluate_fast_assisted_centerline_metrics(
            predicted.iloc[:0] if empty else predicted, truth, truth_type_field='BHBM',
            image_crs=3857, image_transform=from_origin(0, 500, .5, .5), image_shape=(1000, 1000))

    def test_half_metre_grid_reports_real_pixels_and_four_pixel_tolerance(self):
        for metres, pixels in [(0., 0.), (1.5, 3.), (2., 4.), (24., 48.)]:
            with self.subTest(metres=metres):
                result = self.evaluate(metres)
                self.assertAlmostEqual(result['centerline_mean_offset_px'], pixels, delta=.15)
                self.assertEqual(result['centerline_offset_unit'], 'px')
                self.assertEqual(result['centerline_match_tolerance_px'], 4.)
        self.assertAlmostEqual(self.evaluate(geographic=True)['centerline_mean_offset_px'], 0., delta=.15)
        self.assertEqual(self.evaluate(24.)['road_centerline_completeness'], 0.)

    def test_absent_prediction_is_zero_completeness_but_unavailable_offset(self):
        result = self.evaluate(empty=True)
        self.assertEqual(result['road_centerline_completeness'], 0.)
        self.assertIsNone(result['centerline_mean_offset_px'])
        self.assertEqual(result['centerline_offset_status'], 'unavailable')

    def test_distant_multipart_roads_do_not_allocate_the_empty_region_between_parts(self):
        geometry = MultiPolygon([box(100,100,200,108),box(100100,100100,100200,100108)])
        result = evaluate_fast_assisted_centerline_metrics(
            frame([dict(change_typ='added',geometry=translate(geometry,yoff=1.))]),
            frame([dict(BHBM=2,geometry=geometry)]),truth_type_field='BHBM',image_crs=3857,
            image_transform=from_origin(0,100500,.5,.5),image_shape=(201000,201000))
        self.assertAlmostEqual(result['centerline_mean_offset_px'],2.,delta=.15)
        self.assertGreater(result['road_centerline_completeness'],.98)
        self.assertGreater(result['truth_centerline_length_px'],350)

    def test_partial_correction_identities_cannot_erase_actual_road_coverage(self):
        truth=frame([dict(BHBM=2,geometry=box(100,y,200,y+8)) for y in (100,200)])
        predicted=truth.rename(columns={'BHBM':'change_typ'});predicted['change_typ']='added'
        result=evaluate_fast_assisted_centerline_metrics(predicted,truth,truth_type_field='BHBM',
            image_crs=3857,image_transform=from_origin(0,500,.5,.5),image_shape=(1000,1000),
            offset_predicted=predicted.iloc[:1])
        self.assertEqual(result['road_centerline_completeness'],1.)
        self.assertEqual(result['centerline_mean_offset_px'],0.)

    def test_missing_old_truth_identity_does_not_erase_published_roads(self):
        predicted = frame([dict(change_typ='added', geometry=box(0, 0, 100, 10))])
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / 'audit.gpkg'
            for audit in (predicted, predicted.assign(truth_id='')):
                audit.to_file(path, layer='changes', driver='GPKG')
                actual = matched_final_roads(predicted, path)
                self.assertTrue(actual.geometry.union_all().equals(predicted.geometry.union_all()))

    def test_default_correction_does_not_omit_shift_shorten_or_relabel(self):
        profile = GTProfile()
        self.assertEqual((profile.omission_probability, profile.offset_m, profile.width_fraction,
                          profile.end_loss_fraction, profile.type_error_probability), (0., 0., 0., 0., 0.))
        truth = frame([dict(_gt_type='2', DLKD=8., geometry=box(i*40, 0, i*40+20, 8)) for i in range(60)])
        widths = {s: FinalWidths(frame([dict(width_m=8., geometry=LineString([(0,4),(2400,4)]))]))
                  for s in ('before','after')}
        result, audit = perturb_truth(truth, widths, '1', '2')
        self.assertEqual(result.truth_id.nunique(), len(truth))
        self.assertTrue(result.change_typ.eq('added').all())
        self.assertFalse(any(r['action'] == 'omitted_object' for r in audit))
        self.assertGreater(truth.geometry.union_all().intersection(result.geometry.union_all()).area
                           / truth.geometry.area.sum(), .99)

    def test_missing_identity_for_one_class_does_not_hide_that_class(self):
        predicted=frame([dict(change_typ=kind,geometry=box(0,y,100,y+10))
                         for kind,y in [('added',0),('removed',50)]])
        audit=predicted.copy()
        audit['truth_id']=['known','']
        audit['axis_wkt']=[LineString([(0,5),(100,5)]).wkt, '']
        audit['width_bef']=[0.,10.];audit['width_aft']=[10.,0.]
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)/'audit.gpkg';audit.to_file(path,layer='changes',driver='GPKG')
            actual=matched_final_roads(predicted,path)
            self.assertEqual(set(actual.change_typ),{'added','removed'})
            self.assertTrue(actual.geometry.union_all().equals(predicted.geometry.union_all()))

    def test_ninety_two_percent_auto_surface_is_completed(self):
        axis = LineString([(0,0),(100,0)])
        auto = dict(change_id='auto',change_typ='added',width_bef=0.,width_aft=9.2,
                    axis_wkt=axis.wkt,geometry=axis.buffer(4.6,cap_style='flat'))
        gt = dict(auto,change_id='truth',truth_id='gt',gt_type='added',width_aft=10.,
                  match_axis_wkt=axis.wkt,geometry=axis.buffer(5.,cap_style='flat'))
        result, audit = correct_changes(frame([auto]), frame([gt]))
        self.assertGreater(result.geometry.union_all().intersection(gt['geometry']).area / gt['geometry'].area, .99)
        self.assertTrue(any(r['action']=='complete_same_type_area' for r in audit))

    def test_width_profile_stations_follow_actual_axis_distance(self):
        fit = compact_road_fit(LineString([(0,0),(10,5),(20,0)]).buffer(4.,cap_style='flat'))
        self.assertIsNotNone(fit)
        coords = np.asarray(fit['axis'].coords)
        expected = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(coords,axis=0),axis=1))] / fit['axis'].length
        np.testing.assert_allclose(fit['fitted_stations'], expected)

    def test_aggregation_does_not_treat_uncomputed_pair_as_zero(self):
        computed = dict(self.evaluate(), **{'class':'all','tp_m2':1.,'fp_m2':0.,'fn_m2':0.})
        missing = dict(computed, road_centerline_completeness=None, centerline_mean_offset_px=None,
                       truth_centerline_length_px=None, covered_truth_centerline_length_px=None,
                       predicted_centerline_length_px=None, centerline_offset_integral_px2=None)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); entries=[]
            for i,row in enumerate((computed,missing)):
                path=root/f'{i}.json';path.write_text(json.dumps({'evaluation':{'metrics':[row]}}))
                entries.append(dict(summary=str(path),evaluation_state='completed'))
            result=pipeline.aggregate_change_evaluations(dict(execution_profile='fast',change_results=entries),root)
            row=result['metrics'][0]
            self.assertIsNone(row['road_centerline_completeness'])
            self.assertIsNone(row['centerline_mean_offset_px'])
            self.assertEqual(row['centerline_offset_unit'],'px')

    def test_mixed_old_and_new_completeness_tolerances_are_not_pooled(self):
        current=dict(self.evaluate(), **{'class':'all'})
        older=dict(current);older.pop('centerline_match_tolerance_px')
        current['centerline_match_tolerance_px']=3.
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);entries=[]
            for i,row in enumerate((current,older)):
                path=root/f'{i}.json';path.write_text(json.dumps({'evaluation':{'metrics':[row]}}))
                entries.append(dict(summary=str(path),evaluation_state='completed'))
            row=pipeline.aggregate_change_evaluations(dict(execution_profile='fast',change_results=entries),root)['metrics'][0]
            self.assertIsNone(row['road_centerline_completeness'])
            self.assertEqual(row['centerline_mean_offset_px'],0.)


class RegionalEvaluationTests(unittest.TestCase):
    def test_different_regions_without_gt_flags_use_their_own_pixel_grid(self):
        import rasterio
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); manifest=dict(execution_profile='fast',period_results=[],change_results=[])
            for name,x,y in [('north',500000,3000200),('south',800000,4000200)]:
                (root/name).mkdir()
                image=root/(name+'.tif'); transform=from_origin(x,y,.5,.5)
                with rasterio.open(image,'w',driver='GTiff',width=400,height=400,count=1,
                                   dtype='uint8',crs=3857,transform=transform) as dataset:
                    dataset.write(np.zeros((1,400,400),dtype='uint8'))
                manifest['period_results'].append(dict(grid=name,period='before',road_probability=str(image)))
                truth=frame([dict(BHBM=2,geometry=box(x+50,y-150,x+150,y-142))])
                predicted=frame([dict(change_typ='added',geometry=translate(truth.geometry.iloc[0],yoff=1.))])
                truth_path=root/(name+'_truth.gpkg');changes=root/(name+'_changes.gpkg')
                truth.to_file(truth_path,driver='GPKG');predicted.to_file(changes,driver='GPKG')
                if name=='north':
                    audit=predicted.assign(truth_id='known',width_bef=0.,width_aft=8.,
                        axis_wkt=LineString([(x+50,y-146),(x+150,y-146)]).wkt)
                    audit.to_file(root/name/'gt_assisted_audit.gpkg',layer='changes',driver='GPKG')
                summary=root/(name+'.json')
                # A stale saved grid must not override this region's actual raster.
                summary.write_text(json.dumps(dict(gt_assisted_grid_crs='EPSG:3857',
                    gt_assisted_grid_transform=[1,0,0,0,-1,100],gt_assisted_grid_shape=[100,100])))
                entry=dict(grid=name,before_period='before',after_period='after',summary=str(summary),
                    output=str(root/name),layers=dict(changes=str(changes)),fast_finalization_state='completed')
                manifest['change_results'].append(entry)
                args=argparse.Namespace(pipeline_manifest=str(root/'manifest.json'),_manifest=manifest,
                    grid=name,before_period='before',after_period='after',truth=str(truth_path),
                    truth_type_field='BHBM',validation_area='',evaluation_tolerance=5.,
                    defer_publish=True,defer_aggregate=True)
                pipeline._evaluate_existing_changes_impl(args)
                row=pipeline.read_json(summary)['evaluation']['metrics'][0]
                if name=='north':
                    self.assertEqual(pipeline.read_json(summary)['evaluation']['metadata']['centerline_association'],
                                     'final_road_intervals_from_correction_identity')
                self.assertAlmostEqual(row['recall'],.875,places=5)
                self.assertAlmostEqual(row['change_recall'],1.,places=5)
                self.assertAlmostEqual(row['centerline_mean_offset_px'],2.,delta=.15)
                self.assertGreater(row['road_centerline_completeness'],.98)
                self.assertEqual(row['centerline_match_tolerance_px'],4.)

            # Missing pixel metadata is not a zero or permission to show legacy metres.
            manifest['period_results']=[]
            summary.write_text('{}')
            pipeline._evaluate_existing_changes_impl(args)
            row=pipeline.read_json(summary)['evaluation']['metrics'][0]
            self.assertAlmostEqual(row['recall'],.875,places=5)
            self.assertIsNone(row['change_recall'])
            self.assertIsNone(row['road_centerline_completeness'])
            self.assertIsNone(row['centerline_mean_offset_px'])
            self.assertEqual(row['centerline_offset_status'],'unavailable_image_grid')
            self.assertEqual(row['centerline_offset_unit'],'px')


if __name__ == '__main__':
    unittest.main()
