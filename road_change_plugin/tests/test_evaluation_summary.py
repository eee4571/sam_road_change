"""Display only reported business metrics, with explicit offset units."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plugin.ui.evaluation_summary import metrics_text


class EvaluationSummaryTests(unittest.TestCase):
    def text(self,row):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json'
            path.write_text(json.dumps({'metrics':[{'class':'all',**row}]}),encoding='utf8')
            return metrics_text(path)

    def test_generic_scores_are_not_substituted_for_missing_business_metrics(self):
        text=self.text({'precision':.9,'recall':.8,'f1':.85,'iou':.7})
        self.assertEqual(text.count('—'),5)
        self.assertEqual(len(text.splitlines()),5)
        for word in ('Precision','Recall','F1','IoU'):
            self.assertNotIn(word,text)

    def test_reported_pixel_unit_takes_precedence_over_older_metric(self):
        text=self.text({'centerline_offset_unit':'px','centerline_mean_offset_px':2.48,
                        'centerline_avg_offset_m':3.19,'change_precision':.269,'change_recall':.947})
        self.assertIn('2.48 px',text)
        self.assertNotIn('3.19',text)
        self.assertIn('变化图斑查全率  94.7%',text)

    def test_invalid_values_and_valid_zero(self):
        text=self.text({'change_precision':True,'change_recall':float('nan'),
                        'change_type_accuracy':1.1,'road_centerline_completeness':0.,'centerline_avg_offset_m':0.})
        self.assertEqual(text.count('—'),3)
        self.assertIn('0.0%',text)
        self.assertIn('0.00 m',text)

    def test_unreadable_report_keeps_five_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json'
            self.assertEqual(metrics_text(path).count('—'),5)
            path.write_text('{invalid',encoding='utf8')
            self.assertEqual(metrics_text(path).count('—'),5)
