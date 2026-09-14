import unittest
from collections import Counter
import numpy as np
from shapely.geometry import LineString
from engine.fast_object_reconciliation import reconcile_presence
from engine.fast_auto_v2 import profile_variability


def record(kind,y,start=0,end=100,junction=False):
    line=LineString([(start,y),(end,y)])
    return dict(change_typ=kind,axis_wkt=line.wkt,geometry=line.buffer(5,cap_style='flat'),
                width_bef=10. if kind=='removed' else 0.,width_aft=10. if kind=='added' else 0.,
                v2_publish=True,length_m=line.length,junction=junction)


class ReconciliationTests(unittest.TestCase):
    def test_displaced_pair_retained_as_internal_stable(self):
        rows=[record('removed',0),record('added',6)];wkb=[r['geometry'].wkb for r in rows]
        counts=reconcile_presence(rows)
        self.assertEqual(counts['v2_reconciled_pairs'],1)
        self.assertTrue(all(not r['v2_publish'] for r in rows))
        self.assertEqual(wkb,[r['geometry'].wkb for r in rows])

    def test_split_track_pairs_as_one_object(self):
        rows=[record('removed',0),record('added',6,0,50),record('added',6,50,100)]
        counts=reconcile_presence(rows)
        self.assertEqual(counts['v2_reconciled_added'],2)
        self.assertEqual(counts['v2_reconciled_removed'],1)

    def test_crossing_and_low_longitudinal_overlap_remain(self):
        rows=[record('removed',0),record('added',6,60,160)]
        self.assertEqual(reconcile_presence(rows)['v2_reconciled_pairs'],0)
        rows[1]['axis_wkt']=LineString([(50,-50),(50,50)]).wkt
        self.assertEqual(reconcile_presence(rows)['v2_reconciled_pairs'],0)

    def test_junction_has_stricter_offset_limit(self):
        rows=[record('removed',0,junction=True),record('added',8)]
        self.assertEqual(reconcile_presence(rows)['v2_reconciled_pairs'],0)

    def test_parallel_competitors_are_not_consumed(self):
        rows=[record('removed',0),record('added',5),record('added',7)]
        counts=reconcile_presence(rows)
        self.assertEqual(counts['v2_reconciled_pairs'],0)
        self.assertGreater(counts['v2_reconciliation_ambiguous'],0)

    def test_unpublished_candidates_do_not_cancel_real_change(self):
        rows=[record('removed',0),record('added',6)];rows[0]['v2_publish']=False
        self.assertEqual(reconcile_presence(rows)['v2_reconciled_pairs'],0)
        self.assertTrue(rows[1]['v2_publish'])

    def test_time_reversal_and_record_order(self):
        for kinds in [('added','removed'),('removed','added')]:
            rows=[record(kinds[0],0),record(kinds[1],6)]
            self.assertEqual(reconcile_presence(rows)['v2_reconciled_pairs'],1)
            self.assertEqual([r['change_typ'] for r in rows],list(kinds))

    def test_profile_variability_is_length_weighted(self):
        a=np.array([4.,14.]);b=a+10
        self.assertGreater(2*profile_variability(a,b,np.array([50.,50.])),10)
        split=profile_variability(np.array([4.,4.,14.]),np.array([14.,14.,24.]),np.array([25.,25.,50.]))
        self.assertEqual(split,profile_variability(a,b,np.array([50.,50.])))
        self.assertEqual(profile_variability(np.array([8.,8.]),np.array([14.,14.]),np.array([50.,50.])),0.)


if __name__=='__main__':unittest.main()
