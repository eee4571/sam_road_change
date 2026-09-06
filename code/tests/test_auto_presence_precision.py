import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
from shapely import union_all
from shapely.geometry import LineString, box
from shapely.ops import substring

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.auto_presence_candidates import qualify_presence_candidates, annotate_objects


class PresencePrecisionTests(unittest.TestCase):
    crs = "EPSG:32650"

    def scene(self, lines):
        from shapely.strtree import STRtree
        surface = union_all([line.buffer(4, cap_style="flat") for line in lines])
        return SimpleNamespace(lines=lines,tree=STRtree(lines), valid=box(-200, -200, 200, 200), surface=lambda axis: surface)

    def fixture(self, entries):
        period_lines = {"before": [], "after": []}
        candidates, evidence = [], []
        for kind, axis in entries:
            side, other = ("after", "before") if kind == "added" else ("before", "after")
            axis_id = len(period_lines[side])
            period_lines[side].append(axis)
            candidates.append(dict(change_typ=kind, width_bef=8. if kind == "removed" else 0.,
                                   width_aft=8. if kind == "added" else 0., width_diff=0., axis_wkt=axis.wkt,
                                   source_axis=axis_id, start_m=0., end_m=axis.length, length_m=axis.length,
                                   qa_state="probable", confidence=.6, audit_reason="raw_candidate", junction=False,
                                   geometry=axis.buffer(4, cap_style="flat")))
            for i in range(25):
                row = dict(side=side, axis_id=axis_id, station_m=(i+.5)*axis.length/25, junction=False,
                           geometry=substring(axis, i*axis.length/25, (i+1)*axis.length/25))
                for p in period_lines:
                    own = p == side
                    row.update({f"{p}_state": "present" if own else "absent", f"{p}_valid": True,
                                f"{p}_surface": float(own), f"{p}_probability": own,
                                f"{p}_scene_percentile_rank": .95 if own else .2})
                evidence.append(row)
        return (gpd.GeoDataFrame(candidates, geometry="geometry", crs=self.crs),
                {p: self.scene(lines) for p, lines in period_lines.items()},
                gpd.GeoDataFrame(evidence, geometry="geometry", crs=self.crs))

    def test_supported_isolated_changes_and_time_reversal(self):
        geometry = LineString([(0, 0), (100, 0)])
        for kind in ("added", "removed"):
            inputs = self.fixture([(kind, geometry)])
            accepted, audit = qualify_presence_candidates(*inputs)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted.qa_state.iloc[0], "confirmed")
            self.assertTrue(accepted.geometry.iloc[0].equals(inputs[0].geometry.iloc[0]))

    def test_centerline_without_source_corroboration_is_review_only(self):
        candidates, scenes, evidence = self.fixture([("removed", LineString([(0, 0), (100, 0)]))])
        evidence["before_surface"], evidence["before_probability"], evidence["before_scene_percentile_rank"] = 0., False, .2
        accepted, audit = qualify_presence_candidates(candidates, scenes, evidence)
        self.assertTrue(accepted.empty)
        self.assertIn("source_road_not_corroborated", audit.precision_reason.iloc[0])
        self.assertTrue(audit.geometry.iloc[0].equals(candidates.geometry.iloc[0]))

    def test_weak_opposite_or_invalid_observation_is_not_formal_change(self):
        for key, value in (("before_state", "uncertain"), ("before_valid", False)):
            candidates, scenes, evidence = self.fixture([("added", LineString([(0, 0), (100, 0)]))])
            evidence[key] = value
            accepted, audit = qualify_presence_candidates(candidates, scenes, evidence)
            self.assertTrue(accepted.empty)
            self.assertEqual(audit.publication_state.iloc[0], "review")

    def test_opposing_parallel_tracks_are_both_sent_to_review(self):
        inputs = self.fixture([("added", LineString([(0, 0), (100, 0)])),
                               ("removed", LineString([(0, 12), (100, 12)]))])
        accepted, audit = qualify_presence_candidates(*inputs)
        self.assertTrue(accepted.empty)
        self.assertTrue(audit.precision_reason.str.contains("opposing_parallel_change_tracks").all())

    def test_perpendicular_changes_are_not_parallel_conflicts(self):
        inputs = self.fixture([("added", LineString([(0, 0), (100, 0)])),
                               ("removed", LineString([(50, -50), (50, 50)]))])
        accepted, audit = qualify_presence_candidates(*inputs)
        self.assertEqual(len(accepted), 2)

    def test_weak_opposing_track_still_exposes_cross_period_ambiguity(self):
        candidates, scenes, evidence = self.fixture([("added", LineString([(0, 0), (100, 0)])),
                                                    ("removed", LineString([(0, 25), (100, 25)]))])
        evidence.loc[evidence.side == "after", ["after_surface", "after_scene_percentile_rank"]] = 0.
        evidence.loc[evidence.side == "after", "after_probability"] = False
        accepted, audit = qualify_presence_candidates(candidates, scenes, evidence)
        self.assertTrue(accepted.empty)
        self.assertTrue(audit.precision_reason.str.contains("opposing_parallel_change_tracks").all())

    def test_published_polygon_cannot_extend_beyond_final_source_surface(self):
        candidates, scenes, evidence = self.fixture([("added", LineString([(0, 0), (100, 0)]))])
        surface = box(0, -4, 80, 4)
        scenes["after"].surface = lambda axis: surface
        evidence.loc[evidence.station_m > 80, "after_surface"] = 0.
        accepted, _ = qualify_presence_candidates(candidates, scenes, evidence)
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted.geometry.iloc[0].equals(surface))

    def test_width_geometry_bypasses_presence_qualification(self):
        candidates, scenes, evidence = self.fixture([("added", LineString([(0, 0), (100, 0)]))])
        candidates["change_typ"] = "widened"
        accepted, _ = qualify_presence_candidates(candidates, scenes, evidence.iloc[:0])
        self.assertTrue(accepted.geometry.iloc[0].equals(candidates.geometry.iloc[0]))

    def test_empty_published_objects_keep_auditable_qa_schema(self):
        import pandas as pd
        objects = gpd.GeoDataFrame({'object_id': [], 'change_typ': []}, geometry=[], crs=self.crs)
        result = annotate_objects(objects, objects, pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertTrue({'qa_state', 'confidence', 'audit_reason'}.issubset(result.columns))

    def test_junction_only_and_short_nonjunction_have_distinct_reasons(self):
        for junction in (False,True):
            candidates,scenes,evidence=self.fixture([('added',LineString([(0,0),(12,0)]))])
            evidence['junction']=junction
            _,audit=qualify_presence_candidates(candidates,scenes,evidence)
            self.assertEqual('junction_only_presence_change' in audit.precision_reason.iloc[0],junction)

    def test_absence_needs_a_contiguous_negative_run(self):
        candidates,scenes,evidence=self.fixture([('added',LineString([(0,0),(100,0)]))])
        evidence.loc[evidence.index%5==4,'before_state']='uncertain'
        _,audit=qualify_presence_candidates(candidates,scenes,evidence)
        self.assertEqual(audit.publication_state.iloc[0],'review')
        self.assertIn('opposite_absence_not_sustained',audit.precision_reason.iloc[0])


if __name__ == "__main__":
    unittest.main()
