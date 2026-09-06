import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.auto_presence_candidates import LongitudinalCoverage, presence_seeds, annotate_objects


class PresenceRecallTests(unittest.TestCase):
    def setUp(self):
        self.axis = LineString([(0, 0), (100, 0)])
        self.source = SimpleNamespace(width=lambda p: 8.)
        self.empty = LongitudinalCoverage([], 3.)

    def rows(self, other="uncertain", source="present", kind="added", junction=False):
        rows = []
        for i in range(25):
            row = dict(axis_id=0, station_m=4*i+2, junction=junction)
            for period in ("before", "after"):
                own = (period == "after") == (kind == "added")
                row.update({f"{period}_state": source if own else other,
                            f"{period}_geometry": own,
                            f"{period}_surface": 1. if own else .2,
                            f"{period}_probability": own,
                            f"{period}_valid": (source if own else other) != "uncertain",
                            f"{period}_reason": "geometry" if own else "weak_evidence"})
            rows.append(row)
        return rows

    def generate(self, rows, kind="added", coverage=None):
        return presence_seeds(self.axis, rows, self.source, coverage or self.empty, kind, 24., 4.)[0]

    def test_uncertain_opposite_is_one_complete_probable_interval(self):
        seeds = self.generate(self.rows())
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["qa_state"], "probable")
        self.assertAlmostEqual(seeds[0]["length_m"], 100.)
        self.assertAlmostEqual(seeds[0]["geometry"].area, 800.)

    def test_added_removed_strict_time_symmetry(self):
        for state in ("absent", "uncertain", "present"):
            added = self.generate(self.rows(other=state))
            removed = self.generate(self.rows(other=state, kind="removed"), "removed")
            self.assertEqual(len(added), len(removed))
            for a, b in zip(added, removed):
                self.assertTrue(a["geometry"].equals(b["geometry"]))
                self.assertEqual(a["qa_state"], b["qa_state"])
                self.assertEqual(a["confidence"], b["confidence"])

    def test_present_present_is_barrier_even_when_interval_is_unmatched(self):
        rows = self.rows()
        for r in rows[10:15]:
            r["before_state"] = "present"
        seeds = self.generate(rows)
        self.assertEqual([(s["start_m"], s["end_m"]) for s in seeds], [(0., 40.), (60., 100.)])

    def test_confidence_transitions_and_junction_do_not_fragment(self):
        rows = self.rows(other="absent", junction=True)
        for r in rows[8:15]:
            r["before_state"] = "uncertain"
        seeds = self.generate(rows)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["qa_state"], "probable")
        self.assertIn("junction_evidence_discount", seeds[0]["audit_reason"])

    def test_invalid_source_retained_as_uncertain(self):
        seeds = self.generate(self.rows(source="uncertain"))
        self.assertEqual(seeds[0]["qa_state"], "uncertain")

    def test_short_interval_retained_for_assembly(self):
        rows = self.rows(other="present")
        for r in rows[10:12]:
            r["before_state"] = "uncertain"
        seeds = self.generate(rows)
        self.assertEqual(seeds[0]["length_m"], 8.)
        self.assertEqual(seeds[0]["qa_state"], "uncertain")

    def test_analytic_coverage_ignores_crossing_and_respects_feature_seams(self):
        crossing = LongitudinalCoverage([LineString([(50, -20), (50, 20)])], 3.)
        self.assertEqual(crossing.uncovered(self.axis), [(0., 100.)])
        segmented = LongitudinalCoverage([LineString([(0, 2), (40, 2)]), LineString([(40, 2), (100, 2)])], 3.)
        self.assertEqual(segmented.uncovered(self.axis), [])
        extension = LongitudinalCoverage([LineString([(0, 0), (50.5, 0)])], 3.)
        self.assertAlmostEqual(extension.uncovered(self.axis)[0][0], 53.5)

    def test_object_qa_uses_all_members_not_first_seed(self):
        objects = gpd.GeoDataFrame(dict(object_id=["A"], qa_state=["confirmed"], confidence=[.9]), geometry=[self.axis.buffer(4)])
        seeds = pd.DataFrame(dict(qa_state=["confirmed", "probable", "uncertain"], confidence=[.9, .6, .3], audit_reason=["a", "b", "c"]))
        membership = pd.DataFrame(dict(object_id=["A"]*3, seed_id=[0, 1, 2]))
        result = annotate_objects(objects, seeds, membership)
        self.assertTrue(result.geometry.iloc[0].equals(objects.geometry.iloc[0]))
        self.assertEqual(result.qa_state.iloc[0], "uncertain")
        self.assertEqual(result.audit_reason.iloc[0], "a;b;c")


if __name__ == "__main__":
    unittest.main()
