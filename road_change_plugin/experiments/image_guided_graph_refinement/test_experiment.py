"""Small synthetic regressions; no model, project fixture or inference required."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box
from rasterio.transform import from_origin

from bootstrap import ROOT
from data import discover, read_json
from evidence import Evidence, longest_hole
from graph import build, candidates
from refine import spur_decision, gap_decision, duplicate_decision, safe_duplicate, safe_gap, run
from report import write_json
from ribbon_diagnostic import diagnose, raw_ribbon_profile
from engine.road_connection_evidence import ArrayRoadProbability


CONFIG = read_json(ROOT / "config.json")


def frame(lines, widths=None):
    return gpd.GeoDataFrame(dict(width_m=widths or [10.]*len(lines)), geometry=[LineString(x) for x in lines], crs="EPSG:32650")


def measure(**overrides):
    row = dict(independent_valid_fraction=1., rgb_valid_fraction=1., independent_support_fraction=1.,
               bilateral_edge_score=.8, cached_width_quality=None, probability_mean=.7,
               molra_surface_support=0., continuous_support_fraction=1., maximum_unsupported_m=0.,
               route_direction_cosine=1., route_length_ratio=1., maximum_turn_degrees=0.)
    row.update(overrides)
    return row


class ExperimentTests(unittest.TestCase):
    def test_missing_evidence_never_deletes_or_connects(self):
        unknown = measure(independent_valid_fraction=0.)
        self.assertEqual(spur_decision(unknown)[0], "review")
        self.assertEqual(gap_decision(unknown, CONFIG)[0], "review")

    def test_spur_three_way_decision(self):
        self.assertEqual(spur_decision(measure())[0], "keep")
        self.assertEqual(spur_decision(measure(independent_support_fraction=.05, bilateral_edge_score=.1))[0], "delete")
        self.assertEqual(spur_decision(measure(independent_support_fraction=.4))[0], "review")
        self.assertEqual(spur_decision(measure(independent_support_fraction=0., bilateral_edge_score=0., cached_width_quality=.3))[0], "keep")

    def test_short_gap_has_no_distance_override(self):
        self.assertEqual(gap_decision(measure(probability_mean=0., continuous_support_fraction=0.), CONFIG)[0], "review")
        self.assertEqual(gap_decision(measure(), CONFIG)[0], "connect")
        self.assertEqual(gap_decision(measure(maximum_unsupported_m=20.), CONFIG)[0], "review")
        self.assertEqual(gap_decision(measure(maximum_turn_degrees=90.), CONFIG)[0], "review")

    def test_trace_entire_spur_through_degree_two(self):
        edges = build(frame([[(-30, 0), (30, 0)], [(0, 0), (0, 5)], [(0, 5), (0, 10)]]))
        spurs = candidates(edges, box(-50, -50, 50, 50), CONFIG, ("spur",))
        vertical = [c for c in spurs if abs(c.axis.coords[0][0]) < .01]
        self.assertEqual(len(vertical), 1)
        self.assertAlmostEqual(vertical[0].axis.length, 10.)
        self.assertEqual(len(vertical[0].ids), 2)

    def test_roi_boundary_and_grade_separation(self):
        f = frame([[(-50, 0), (50, 0)], [(0, 0), (0, 15)]])
        edges = build(f)
        self.assertFalse(candidates(edges, box(-10, -10, 10, 10), CONFIG, ("spur",)))
        f["layer"] = [0, 1]
        self.assertEqual(len(build(f)), 2)

    def test_gap_direction_and_crossing_guard(self):
        f = frame([[(-30, 0), (0, 0)], [(15, 0), (45, 0)]])
        edges = build(f)
        gaps = candidates(edges, box(-100, -100, 100, 100), CONFIG, ("gap",))
        self.assertEqual(len(gaps), 1)
        self.assertTrue(safe_gap(gaps[0].axis, edges, gaps[0]))
        crossing = build(frame([[(-30, 0), (0, 0)], [(15, 0), (45, 0)], [(7, -10), (7, 10)]]))
        self.assertFalse(safe_gap(gaps[0].axis, crossing, gaps[0]))
        perpendicular = build(frame([[(-30, 0), (0, 0)], [(15, 0), (15, 30)]]))
        self.assertFalse(candidates(perpendicular, box(-100, -100, 100, 100), CONFIG, ("gap",)))

    def ribbon_evidence(self, separator):
        ev = object.__new__(Evidence)
        ev.config = CONFIG
        transform = from_origin(-10, 10, .25, .25)
        yy = 10-(np.arange(80)+.5)*.25
        mask = np.ones((80, 320))*.8
        if separator:
            mask[(yy > .8) & (yy < 3.2)] = 0.
        ev.probability = ArrayRoadProbability([dict(mask=mask, transform=transform)], "EPSG:32650", "EPSG:32650")
        ev.molra, ev.surface = None, None
        class Image:
            gray = None
            def sample(self, array, xy):
                return np.ones(len(xy))*.5
        ev.image = Image()
        return ev

    def test_single_ribbon_versus_true_parallel_roads(self):
        a, b = LineString([(0, 0), (50, 0)]), LineString([(0, 4), (50, 4)])
        single = self.ribbon_evidence(False).ribbons(a, b)
        two = self.ribbon_evidence(True).ribbons(a, b)
        self.assertEqual(duplicate_decision(single)[0], "merge")
        self.assertEqual(duplicate_decision(two)[0], "keep")
        two["valid_fraction"] = .1
        self.assertEqual(duplicate_decision(two)[0], "review")

    def test_duplicate_ablation_merges_only_single_ribbon(self):
        edges = build(frame([[(0, 0), (50, 0)], [(0, 4), (50, 4)]]))
        roi = box(-10, -10, 60, 15)
        for separator, expected in [(False, "merge"), (True, "keep")]:
            result, audit, changes = run(edges, self.ribbon_evidence(separator), roi, CONFIG,
                                         "image_guided", ("duplicate",))
            self.assertEqual(audit[0]["decision"], expected)
            self.assertEqual(len(result), 2 if separator else 1)
        result, audit, _ = run(edges, self.ribbon_evidence(True), roi, CONFIG, "geometry_only", ("duplicate",))
        self.assertEqual(len(result), 1)
        self.assertEqual(audit[0]["evidence_decision"], "keep")

    def test_rgb_separator_vetoes_blurred_probability(self):
        score = dict(valid_fraction=1., two_ribbon_fraction=0., single_ribbon_fraction=0.,
                     raw_separator_fraction=1., rgb_valid_fraction=1.)
        self.assertEqual(duplicate_decision(score)[0], "keep")

    def test_wide_road_missing_axis_is_found_even_when_sam_misses_it(self):
        class Image:
            gray = None
            def sample(self, array, xy):
                return np.where((xy[:, 1] > 2) & (xy[:, 1] < 28), .9, .2)
        class RawEvidence:
            image = Image()
            def values(self, xy):
                return np.zeros(len(xy)), np.full(len(xy), np.nan), np.zeros(len(xy))
        edges = build(frame([[(0, 0), (60, 0)], [(0, 30), (60, 30)]], [8., 8.]))
        roi = box(-20, -20, 80, 50)
        self.assertFalse(candidates(edges, roi, CONFIG, ("duplicate",)))
        reports = diagnose(edges, RawEvidence(), roi)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["evidence"]["probability_mean"], 0.)
        self.assertFalse(reports[0]["auto_applied"])
        from shapely import wkt
        self.assertAlmostEqual(wkt.loads(reports[0]["result_wkt"]).centroid.y, 15., places=5)

    def test_rgb_ribbon_diagnostic_rejects_two_ribbons_with_dark_separator(self):
        fractions = np.linspace(-.5, 1.5, 61)
        single = np.where((fractions > .05) & (fractions < .95), .9, .2)
        divided = np.where((abs(fractions) < .1) | (abs(fractions-1) < .1), .9, .2)
        self.assertEqual(raw_ribbon_profile(np.tile(single, (10, 1)), fractions)["raw_ribbon_support_fraction"], 1.)
        self.assertEqual(raw_ribbon_profile(np.tile(divided, (10, 1)), fractions)["raw_ribbon_support_fraction"], 0.)

    def test_duplicate_branch_contact_is_protected(self):
        edges = build(frame([[(0, 0), (50, 0)], [(0, 4), (50, 4)], [(25, 4), (25, 30)]]))
        duplicates = candidates(edges, box(-100, -100, 100, 100), CONFIG, ("duplicate",))
        self.assertTrue(duplicates)
        self.assertTrue(all(not safe_duplicate(c, edges) for c in duplicates))

    def test_json_is_strict_and_missing_manifest_fails(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with TemporaryDirectory(dir=ROOT / "outputs") as name:
            path = Path(name) / "audit.json"
            write_json(path, dict(score=np.nan, profile=[np.inf, np.float64(.5)]))
            self.assertEqual(json.loads(path.read_text()), dict(score=None, profile=[None, .5]))
            with self.assertRaises(FileNotFoundError):
                discover(name, "missing")

    def test_no_support_run_is_bounded_by_axis_length(self):
        self.assertEqual(longest_hole(np.zeros(5, bool), 10), 10)

    def test_cached_raster_routing_and_ablation_do_not_write_inputs(self):
        import rasterio
        (ROOT / "outputs").mkdir(exist_ok=True)
        with TemporaryDirectory(dir=ROOT / "outputs") as name:
            folder = Path(name)
            transform = from_origin(-20, 20, .5, .5)
            rgb = np.full((3, 80, 240), 40, np.uint8)
            rgb[:, 32:48] = 180
            probability = np.zeros((80, 240), np.uint8)
            probability[32:48] = 230
            for filename, data in [("rgb.tif", rgb), ("prob.tif", probability[None])]:
                with rasterio.open(folder/filename, "w", driver="GTiff", height=80, width=240,
                                   count=len(data), dtype="uint8", transform=transform, crs="EPSG:32650") as ds:
                    ds.write(data)
            original = {p.name: p.read_bytes() for p in folder.glob("*.tif")}
            inputs = dict(images=[folder/"rgb.tif"], probabilities=[(folder/"prob.tif", None)], molra=[],
                          assets=dict(surfaces=folder/"absent.gpkg", observations=folder/"absent.gpkg"))
            evidence = Evidence(inputs, "EPSG:32650", box(-20, -20, 100, 20), CONFIG)
            edges = build(frame([[(0, 0), (30, 0)], [(55, 0), (85, 0)]], [8., 8.]))
            result, audit, changes = run(edges, evidence, box(-10, -10, 95, 10), CONFIG, "image_guided", ("gap",))
            self.assertEqual(len(changes), 1, audit)
            self.assertEqual(audit[0]["decision"], "connect")
            self.assertEqual(len(result), len(edges)+1)
            self.assertEqual(len(edges), 2)  # input graph is immutable
            spur_edges = build(frame([[(0, 0), (85, 0)], [(15, 0), (15, 15)]], [8., 8.]))
            kept, spur_audit, edits = run(spur_edges, evidence, box(-10, -19, 95, 19), CONFIG,
                                         "image_guided", ("spur",))
            self.assertTrue(any(row["decision"] == "delete" for row in spur_audit), spur_audit)
            self.assertAlmostEqual(sum(edge.axis.length for edge in kept), 85.)
            for p in folder.glob("*.tif"):
                self.assertEqual(p.read_bytes(), original[p.name])


if __name__ == "__main__":
    unittest.main()
