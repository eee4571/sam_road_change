"""Four source pixels affect matching, never the original area or real offset."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'code/engine/width')]
import geopandas as gpd
from affine import Affine
from shapely.geometry import box
from shapely.affinity import translate, affine_transform
from road_change_detection import evaluate_changes
import user_pipeline as pipeline


def frame(geometries, truth=False):
    field, value = ('BHBM', 2) if truth else ('change_typ', 'added')
    return gpd.GeoDataFrame({field:[value]*len(geometries)}, geometry=geometries, crs=3857)


class PositionToleranceTests(unittest.TestCase):
    def evaluate(self, shift=1., resolution=.5):
        truth=frame([box(100,100,200,108)],True)
        predicted=frame([translate(truth.geometry.iloc[0],yoff=shift)])
        return evaluate_changes(predicted,truth,truth_type_field='BHBM',class_mode='three',
            position_grid=(3857,Affine(resolution,0,0,0,-resolution,500)))[0]

    def test_allowed_translation_matches_original_areas(self):
        for shift in (0.,1.,2.):
            row=self.evaluate(shift)[0]
            self.assertAlmostEqual(row['change_recall'],1.)
            self.assertAlmostEqual(row['change_precision'],1.)
            self.assertEqual(row['truth_support_m2'],800.)
            self.assertEqual(row['predicted_support_m2'],800.)
            self.assertAlmostEqual(row['recall'],(8.-shift)/8.)
            self.assertEqual(row['change_match_tolerance_px'],4.)

    def test_excess_displacement_still_loses_coverage(self):
        row=self.evaluate(3.)[0]
        self.assertAlmostEqual(row['change_recall'],7/8)
        self.assertAlmostEqual(row['change_precision'],7/8)
        row=self.evaluate(24.)[0]
        self.assertEqual(row['change_recall'],0.)
        self.assertEqual(row['change_precision'],0.)

    def test_four_pixels_follow_each_images_resolution(self):
        self.assertAlmostEqual(self.evaluate(3.,1.)[0]['change_recall'],1.)
        self.assertAlmostEqual(self.evaluate(2.,.25)[0]['change_recall'],7/8)

    def test_rotated_anisotropic_grid_and_different_truth_crs(self):
        transform=Affine.translation(500000,3000000)*Affine.rotation(30)*Affine.scale(.5,-1.)
        coefficients=[transform.a,transform.b,transform.d,transform.e,transform.c,transform.f]
        pixel_truth=box(20,20,220,28)
        truth=frame([affine_transform(pixel_truth,coefficients)],True).to_crs(4326)
        predicted=frame([affine_transform(translate(pixel_truth,yoff=3.),coefficients)])
        row=evaluate_changes(predicted,truth,truth_type_field='BHBM',class_mode='three',
            position_grid=(3857,transform))[0][0]
        self.assertAlmostEqual(row['change_recall'],1.,places=6)
        self.assertAlmostEqual(row['change_precision'],1.,places=6)

    def test_real_missing_and_extra_roads_still_count(self):
        truth=frame([box(100,100,200,108),box(100,200,200,208)],True)
        predicted=frame([box(100,101,200,109),box(100,300,200,308)])
        validation=frame([box(0,0,500,500)])
        row=evaluate_changes(predicted,truth,validation,truth_type_field='BHBM',class_mode='three',
            position_grid=(3857,Affine(.5,0,0,0,-.5,500)))[0][0]
        self.assertAlmostEqual(row['change_recall'],.5)
        self.assertAlmostEqual(row['change_precision'],.5)

    def test_type_accuracy_remains_separate_from_detection(self):
        truth=frame([box(100,100,200,108)],True)
        predicted=frame([truth.geometry.iloc[0]]);predicted['change_typ']='removed'
        rows,_=evaluate_changes(predicted,truth,truth_type_field='BHBM',class_mode='three',
            position_grid=(3857,Affine(.5,0,0,0,-.5,500)))
        self.assertEqual(rows[0]['change_recall'],1.)
        self.assertEqual(rows[0]['type_judgment_accuracy'],0.)
        self.assertEqual(next(r for r in rows if r['class']=='added')['change_recall'],0.)

    def test_overwide_prediction_has_its_own_precision_denominator(self):
        truth=frame([box(100,100,200,108)],True)
        predicted=frame([box(100,94,200,114)])
        row=evaluate_changes(predicted,truth,truth_type_field='BHBM',class_mode='three',
            position_grid=(3857,Affine(.5,0,0,0,-.5,500)))[0][0]
        self.assertEqual(row['change_recall'],1.)
        self.assertAlmostEqual(row['change_precision'],.6)
        self.assertEqual(row['truth_support_m2'],800.)
        self.assertEqual(row['predicted_support_m2'],2000.)

    def test_empty_and_duplicate_inputs_do_not_create_coverage(self):
        truth=frame([box(100,100,200,108)],True)
        for predicted,expected in [(frame([]),0.),(frame([truth.geometry.iloc[0]]*2),1.)]:
            row=evaluate_changes(predicted,truth,truth_type_field='BHBM',class_mode='three',
                position_grid=(3857,Affine(.5,0,0,0,-.5,500)))[0][0]
            self.assertEqual(row['change_recall'],expected)
            self.assertEqual(row['change_precision'],expected)
            self.assertEqual(row['truth_support_m2'],800.)
            self.assertEqual(row['predicted_support_m2'],800.*expected)

    def aggregate(self, rows):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);entries=[]
            for index,row in enumerate(rows):
                path=root/f'{index}.json';path.write_text(json.dumps({'evaluation':{'metrics':[row]}}))
                entries.append(dict(summary=str(path),evaluation_state='completed'))
            return pipeline.aggregate_change_evaluations(dict(execution_profile='fast',change_results=entries),root)['metrics'][0]

    def test_aggregate_uses_physical_area_and_separate_match_numerators(self):
        first=dict(self.evaluate()[0])
        second=dict(self.evaluate(3.,.5)[0])
        # A region with four times the road area carries four times the weight.
        for key in ('truth_support_m2','predicted_support_m2','position_covered_truth_m2','position_supported_prediction_m2'):
            second[key]*=4
        result=self.aggregate([first,second])
        self.assertAlmostEqual(result['change_recall'],.9)
        self.assertAlmostEqual(result['change_precision'],.9)

    def test_old_or_uncomputed_or_different_tolerance_is_not_mixed(self):
        new=self.evaluate()[0]
        old={k:v for k,v in new.items() if not k.startswith(('position_','change_'))}
        missing=dict(new,position_covered_truth_m2=None)
        different=dict(new,change_match_tolerance_px=3.)
        for other in (old,missing,different):
            with self.subTest(other=other):
                result=self.aggregate([new,other])
                self.assertIsNone(result['change_recall'])
                self.assertIsNone(result['change_precision'])


if __name__=='__main__':
    unittest.main()
