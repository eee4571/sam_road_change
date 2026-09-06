import sys
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.auto_change_assembly import assemble_change_objects


class NetworkAssemblyTests(unittest.TestCase):
    crs = "EPSG:32650"

    def network(self, lines):
        return gpd.GeoDataFrame(geometry=[LineString(c) for c in lines], crs=self.crs)

    def seeds(self, lines, kind="added", before=0., after=8.):
        rows = []
        for coordinates in lines:
            axis = LineString(coordinates)
            geometry = axis.buffer(max(before, after)/2, cap_style="flat")
            if kind in {"widened", "narrowed"}:
                geometry = geometry.difference(axis.buffer(min(before, after)/2, cap_style="flat"))
            rows.append(dict(change_typ=kind, width_bef=before, width_aft=after, width_diff=after-before,
                             length_m=axis.length, axis_wkt=axis.wkt, geometry=geometry))
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=self.crs)

    def test_cross_junction_retains_every_seed_and_fills_road(self):
        roads = self.network([[(-100, 0), (100, 0)], [(0, -100), (0, 100)]])
        seeds = self.seeds([[(-90, 0), (-14, 0)], [(14, 0), (90, 0)]])
        objects, audit = assemble_change_objects(seeds, roads, roads)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects.iloc[0].seed_count, 2)
        self.assertEqual(audit["summary"]["junction_bridge_count"], 1)
        self.assertTrue(objects.geometry.iloc[0].covers(box(-10, -3, 10, 3)))
        self.assertTrue(seeds.geometry.difference(objects.geometry.iloc[0]).area.sum() < 1e-6)

    def test_two_carriageways_cross_junction_without_track_swap(self):
        roads = self.network([[(-100, 0), (100, 0)], [(-100, 10), (100, 10)], [(0, -30), (0, 40)]])
        seeds = self.seeds([[(-90, 0), (-14, 0)], [(14, 0), (90, 0)],
                            [(-90, 10), (-14, 10)], [(14, 10), (90, 10)]])
        objects, audit = assemble_change_objects(seeds, roads, roads)
        self.assertEqual(len(objects), 2)
        self.assertEqual(sorted(objects.seed_ids), ["0,1", "2,3"])
        self.assertEqual(audit["summary"]["junction_bridge_count"], 2)

    def test_width_ribbons_cross_junction(self):
        roads = self.network([[(-100, 0), (100, 0)], [(0, -50), (0, 50)]])
        for kind, wb, wa in (("widened", 8., 14.), ("narrowed", 14., 8.)):
            seeds = self.seeds([[(-90, 0), (-15, 0)], [(15, 0), (90, 0)]], kind, wb, wa)
            objects, audit = assemble_change_objects(seeds, roads, roads)
            self.assertEqual(len(objects), 1)
            self.assertEqual(audit["summary"]["bridge_count"], 1)
            self.assertTrue(objects.geometry.iloc[0].covers(box(-10, 5, 10, 6)))
            self.assertFalse(objects.geometry.iloc[0].covers(box(-10, -2, 10, 2)))
            self.assertEqual(objects.geometry.iloc[0].geom_type, "MultiPolygon")
            self.assertEqual(len(objects.geometry.iloc[0].geoms), 2)

    def test_crossing_directions_and_different_types_stay_separate(self):
        roads = self.network([[(-100, 0), (100, 0)], [(0, -100), (0, 100)]])
        seeds = self.seeds([[(-90, 0), (-14, 0)], [(0, 14), (0, 90)]])
        objects, _ = assemble_change_objects(seeds, roads, roads)
        self.assertEqual(len(objects), 2)
        seeds = self.seeds([[(-90, 0), (-14, 0)], [(14, 0), (90, 0)]])
        seeds.loc[1, "change_typ"] = "removed"
        objects, _ = assemble_change_objects(seeds, roads, roads)
        self.assertEqual(len(objects), 2)

    def test_multiple_junctions_assemble_one_object(self):
        roads = self.network([[(-150, 0), (150, 0)], [(-40, -50), (-40, 50)], [(40, -50), (40, 50)]])
        seeds = self.seeds([[(-140, 0), (-54, 0)], [(-26, 0), (26, 0)], [(54, 0), (140, 0)]])
        objects, audit = assemble_change_objects(seeds, roads, roads)
        self.assertEqual(len(objects), 1)
        self.assertEqual(audit["summary"]["junction_bridge_count"], 2)

    def test_disconnected_and_unmapped_seeds_are_retained(self):
        roads = self.network([[(-100, 0), (-12, 0)], [(12, 0), (100, 0)]])
        seeds = self.seeds([[(-90, 0), (-14, 0)], [(14, 0), (90, 0)], [(0, 100), (50, 100)]])
        objects, audit = assemble_change_objects(seeds, roads, roads)
        self.assertEqual(len(objects), 3)
        self.assertEqual(len(audit["membership"]), 3)

    def test_feature_segmentation_does_not_block_continuation(self):
        roads = self.network([[(-100, 0), (0, 0)], [(0, 0), (100, 0)]])
        seeds = self.seeds([[(-90, 0), (-4, 0)], [(4, 0), (90, 0)]])
        objects, _ = assemble_change_objects(seeds, roads, roads)
        self.assertEqual(len(objects), 1)

    def test_empty_changes_keep_public_schema(self):
        roads = self.network([[(-100, 0), (100, 0)]])
        seeds = self.seeds([[(-90, 0), (-4, 0)]]).iloc[:0]
        objects, audit = assemble_change_objects(seeds, roads, roads)
        self.assertTrue(objects.empty)
        self.assertIn("change_typ", objects)
        self.assertEqual(audit["summary"]["bridge_count"], 0)


if __name__ == "__main__":
    unittest.main()
