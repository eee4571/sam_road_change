"""Final profile widths, state tables and temporal checks must agree."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/engine/width'),str(ROOT.parent/'code/tests')]
import engine
engine.__path__=[str(ROOT/'code/engine')]
# Shared fixture imports may prepend the parent application's module paths.
# Resolve the plugin evaluator first so integration tests use one backend.
import road_change_detection
import geopandas as gpd
from shapely.geometry import LineString
from engine.fast_gt_reconciliation import _frame, _final_road_states, reconcile_periods
from temporal_road_analysis import _read_period, _reconciled_tracks
import test_fast_gt_reconciliation as fixtures

CRS='EPSG:32650'


class FinalRoadStateTests(unittest.TestCase):
    def test_split_track_uses_length_weighted_final_width(self):
        axis=LineString([(0,0),(100,0)])
        states=[dict(track_id='R',period='2',status='present',width_m=8.,geometry=axis)]
        roads=_frame([dict(track_id='R',width_m=4.,geometry=LineString([(0,0),(25,0)])),
                      dict(track_id='R',width_m=12.,geometry=LineString([(25,0),(100,0)]))],CRS)
        result=_final_road_states(states,roads)
        self.assertAlmostEqual(result.iloc[0].width_m,10.)
        self.assertTrue(result.iloc[0].geometry.equals(axis))
        self.assertEqual(states[0]['width_m'],8.)

    def test_presence_mismatch_remains_an_error(self):
        axis=LineString([(0,0),(100,0)])
        for status,records in [('present',[]),('absent',[dict(track_id='R',width_m=8.,geometry=axis)])]:
            with self.subTest(status=status),self.assertRaisesRegex(ValueError,'contradicts period centerline'):
                _final_road_states([dict(track_id='R',period='2',status=status,width_m=8.,geometry=axis)],
                                   _frame(records,CRS))

    def test_invalid_final_width_is_not_published(self):
        axis=LineString([(0,0),(100,0)])
        for width in [float('nan'),float('inf'),0.,-1.]:
            with self.subTest(width=width),self.assertRaisesRegex(ValueError,'Invalid final road width'):
                _final_road_states([dict(track_id='R',period='2',status='present',width_m=8.,geometry=axis)],
                    _frame([dict(track_id='R',width_m=width,geometry=axis)],CRS))

    def test_variable_profile_reconciliation_passes_temporal_qa(self):
        fixture=fixtures.FastGTReconciliationTests()
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);axis=LineString([(0,0),(100,0)]);other=LineString([(0,50),(100,50)])
            periods=[fixture.period(root,str(i),[(other,6.)]) for i in (1,2)]
            profile=json.dumps(dict(axis=axis.wkt,stations=[0.,.25,.5,.75,1.],
                                    widths=[4.,4.,4.,8.,8.],reference_width=8.))
            changes=[dict(frame=_frame([dict(change_id='add',change_typ='added',change_src='GT_ASSISTED',
                width_bef=0.,width_aft=8.,axis_wkt=axis.wkt,match_axis_wkt=axis.wkt,
                gt_width_profile=profile,geometry=axis.buffer(4))],CRS),before_period='1',after_period='2')]
            products=reconcile_periods(periods,changes,root/'final')
            frames={p['period']:_read_period(Path(p['centerlines']),CRS) for p in products}
            refs,obs=_reconciled_tracks(products,frames,CRS)
            self.assertEqual(len(refs),1)
            self.assertEqual([o['status'] for o in obs],['absent','present'])
            self.assertGreater(abs(obs[1]['width_m']-8.),.05)
            # A later stale/corrupt state still fails the unchanged consumer check.
            state_path=products[1]['road_state']
            state=gpd.read_file(state_path,layer='road_state');state['width_m']+=1.
            state.to_file(state_path,layer='road_state',driver='GPKG')
            with self.assertRaisesRegex(ValueError,'width contradicts final width'):
                _reconciled_tracks(products,frames,CRS)


if __name__=='__main__':
    unittest.main()
