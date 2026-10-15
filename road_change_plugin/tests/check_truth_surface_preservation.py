"""Polygon truth survives assistance, slicing and publication without widening."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'code/engine/width')]
import geopandas as gpd
from shapely import from_wkt, union_all
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box, shape
from engine.auto_change_geometry import FinalWidths
from engine.fast_gt_reconciliation import perturb_truth, correct_changes, _frame, _publish_assisted_changes, _atomize_adjacent_changes
from engine.gt_road_geometry import record_polygon
from engine.continuous_road_geometry import change_surfaces
from engine.truth_assistance import _shortened_proposal, limit_truth_assistance
from engine.truth_surface import partition_surface, surface_interval
from affine import Affine


def generated(geometry, kind='2', width=6.51):
    truth = _frame([dict(_gt_type=kind, BHBM=int(kind), DLKD=width, geometry=geometry)], 3857)
    roads = _frame([dict(width_m=width, geometry=LineString([(-100, 0), (1000, 0)]))], 3857)
    candidates, _ = perturb_truth(truth, {p: FinalWidths(roads) for p in ('before', 'after')}, '1', '2')
    return truth, candidates


class TruthSurfaceTests(unittest.TestCase):
    def assert_same_surface(self, actual, expected):
        self.assertLess(actual.symmetric_difference(expected).area, max(1e-7, expected.area*1e-9))

    def test_real_bent_polygon_does_not_double_its_area(self):
        fixture = json.loads((ROOT/'tests/fixtures/truth_surface_bend.json').read_text())
        geometry = shape(fixture['geometry'])
        truth, candidates = generated(geometry)
        self.assert_same_surface(candidates.geometry.union_all(), geometry)
        self.assert_same_surface(change_surfaces(candidates).geometry.union_all(), geometry)
        self.assertNotIn('gt_width_profile', candidates)
        self.assertTrue(candidates.gt_surface_wkt.notna().all())

    def test_presence_and_width_change_keep_full_truth_polygon(self):
        geometry = Polygon([(0,0), (80,0), (80,7), (30,10), (30,60), (0,60)])
        for kind in ('2', '3', '4'):
            with self.subTest(kind=kind):
                _, candidates = generated(geometry, kind)
                self.assert_same_surface(candidates.geometry.union_all(), geometry)
                self.assert_same_surface(change_surfaces(candidates).geometry.union_all(), geometry)

    def test_hole_and_distant_components_are_preserved(self):
        holed = box(0, 0, 50, 50).difference(box(8, 8, 42, 42))
        geometry = MultiPolygon([holed, box(100000, 100000, 100100, 100008)])
        _, candidates = generated(geometry)
        self.assert_same_surface(candidates.geometry.union_all(), geometry)
        self.assert_same_surface(change_surfaces(candidates).geometry.union_all(), geometry)

    def test_branch_partition_preserves_every_piece_without_overlap(self):
        axes = [LineString([(0,0),(50,0)]), LineString([(50,0),(100,0)]), LineString([(50,0),(50,80)])]
        surface = union_all([axis.buffer(4) for axis in axes])
        parts = partition_surface(surface, axes)
        self.assert_same_surface(union_all(parts), surface)
        self.assertAlmostEqual(sum(part.area for part in parts), surface.area, places=6)
        self.assertTrue(all(part.area > 0 for part in parts))

    def test_partial_interval_and_repeated_shortening_never_restore_full_face(self):
        axis = LineString([(0,0),(100,0)])
        face = Polygon([(0,-2),(100,-6),(100,6),(0,2)])
        row = dict(change_typ='added', change_src='GT_ASSISTED', truth_id='GT', change_id='GT0',
                   axis_wkt=axis.wkt, match_axis_wkt=axis.wkt, length_m=100., width_bef=0., width_aft=8.,
                   gt_surface_wkt=face.wkt, geometry=face)
        first = _shortened_proposal(_frame([row],3857), .5)
        second = _shortened_proposal(first, .5)
        self.assertAlmostEqual(first.geometry.area.sum(), 300., places=6)
        self.assertAlmostEqual(second.geometry.area.sum(), 125., places=6)
        self.assert_same_surface(change_surfaces(second).geometry.union_all(), second.geometry.union_all())
        self.assertLess(second.geometry.union_all().difference(face).area, 1e-8)
        # Later lifecycle width changes must not regenerate this footprint.
        changed = second.iloc[0].to_dict(); changed['width_aft'] = 44.
        self.assert_same_surface(record_polygon(changed), second.geometry.iloc[0])

    def test_u_shaped_interval_does_not_clip_or_connect_other_arm(self):
        axis = LineString([(0,0),(100,0),(100,40),(0,40)])
        face = axis.buffer(4, cap_style='flat')
        local = LineString([(0,0),(60,0)])
        selected = surface_interval(face, axis, local)
        self.assertLess(selected.difference(face).area, 1e-8)
        self.assertAlmostEqual(selected.area, 480., delta=1.)
        self.assertLess(selected.intersection(box(-5,30,105,50)).area, 1e-8)

    def test_line_truth_still_uses_declared_width(self):
        _, candidates = generated(LineString([(0,0),(100,0)]), width=8.)
        self.assertAlmostEqual(candidates.geometry.area.sum(), 800.)
        self.assertAlmostEqual(change_surfaces(candidates).geometry.area.sum(), 800.)

    def test_exact_reversal_uses_endpoint_tangent_without_aborting(self):
        axis = LineString([(0,0),(100,0),(0,0)])
        face = box(0,-4,100,4)
        selected = surface_interval(face, axis, LineString([(0,0),(100,0)]))
        self.assert_same_surface(selected, face)

    def test_mixed_collection_preserves_faces_and_renders_only_lines(self):
        face = box(0, -4, 100, 4)
        line = LineString([(0, 100), (100, 100)])
        mixed = GeometryCollection([face, line, Point(1000, 1000)])
        _, candidates = generated(mixed, width=8.)
        expected = union_all([face, line.buffer(4, cap_style='flat')])
        self.assert_same_surface(candidates.geometry.union_all(), expected)
        self.assert_same_surface(change_surfaces(candidates).geometry.union_all(), expected)

    def test_temporal_split_preserves_footprint_and_shared_track(self):
        axis = LineString([(0,0),(100,0)])
        face = box(0,-4,100,4)
        first = dict(change_id='first', change_typ='added', change_src='GT_ASSISTED',
                     truth_id='GT', axis_wkt=axis.wkt, match_axis_wkt=axis.wkt,
                     width_bef=0., width_aft=8., length_m=100., gt_surface_wkt=face.wkt, geometry=face)
        reverse = LineString([(100,0),(50,0)])
        half = box(50,-4,100,4)
        second = dict(first, change_id='second', change_typ='removed', width_bef=8., width_aft=0.,
                      axis_wkt=reverse.wkt, match_axis_wkt=reverse.wkt, length_m=50.,
                      gt_surface_wkt=half.wkt, geometry=half)
        changes = [dict(frame=_frame([row],3857)) for row in (first, second)]
        _atomize_adjacent_changes(changes)
        for pair, expected in zip(changes, (face, half)):
            self.assert_same_surface(pair['frame'].geometry.union_all(), expected)
            self.assert_same_surface(change_surfaces(pair['frame']).geometry.union_all(), expected)
            self.assertEqual(set(pair['frame'].truth_id), {'GT'})
        self.assertTrue(set(changes[1]['frame'].atomic_track).issubset(changes[0]['frame'].atomic_track))

    def test_budget_and_shp_export_use_same_authoritative_surface(self):
        truth, candidates = generated(box(0,-4,100,4), width=8.)
        grid = (3857, Affine(.5,0,-10,0,-.5,100))
        selected, _, audit = limit_truth_assistance(candidates.iloc[:0], candidates, truth, grid)
        self.assertLess(selected.geometry.union_all().difference(truth.geometry.union_all()).area, 1e-8)
        self.assertAlmostEqual(audit['candidate_recall'], .95, delta=1e-6)
        with tempfile.TemporaryDirectory() as raw:
            exported = _publish_assisted_changes(selected, raw, '1', '2', require_review=True)
            public = gpd.read_file(exported['review_candidates'])
            intervals = gpd.read_file(Path(raw)/'gt_assisted_audit.gpkg', layer='review_candidates')
            self.assert_same_surface(public.geometry.union_all(), selected.geometry.union_all())
            self.assert_same_surface(union_all([from_wkt(wkt) for wkt in intervals.gt_surface_wkt]), selected.geometry.union_all())

    def test_conflict_correction_preserves_polygon(self):
        _, gt = generated(box(0,-4,100,4), width=8.)
        axis = from_wkt(gt.iloc[0].axis_wkt)
        auto = gt.copy(); auto['gt_surface_wkt']=''; auto['change_src']='AUTO';auto['change_typ']='removed'
        auto['width_bef']=8.;auto['width_aft']=0.;auto['change_id']='auto';auto.geometry=[axis.buffer(4)]
        corrected, _ = correct_changes(auto, gt)
        self.assert_same_surface(corrected.geometry.union_all(), gt.geometry.union_all())
        self.assert_same_surface(change_surfaces(corrected).geometry.union_all(), gt.geometry.union_all())


if __name__ == '__main__':
    unittest.main()
