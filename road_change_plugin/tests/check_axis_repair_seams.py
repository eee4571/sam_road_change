"""Regression for faults exposed at the joins of a local axis repair."""
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from shapely.geometry import LineString
from shapely.affinity import rotate,translate
from engine.road_axis_quality import axis_quality,repair_axis,repair_network_axes,_repair_axis_pass


def teeth():
    # Translated/rounded local geometry from the failing period, without any
    # project IDs, geographic coordinates, or dataset-specific runtime rules.
    return LineString([(0,0),(24.104,-13.625),(28.774,-15.222),(44.126,-24.086),
        (39.955,-31.454),(40.571,-31.803),(44.796,-24.339),(49.727,-26.204),
        (45.105,-34.368),(45.791,-34.743),(50.408,-26.586),(61.826,-32.988),
        (57.197,-41.165),(57.923,-41.594),(62.636,-33.269),(72.566,-36.711),
        (67.185,-46.215),(67.697,-46.502),(73.066,-37.019),(86.049,-45.037),
        (81.594,-52.906),(82.099,-52.969),(86.437,-45.307),(99.515,-54.377),
        (97.714,-57.557),(98.282,-57.779),(99.214,-58.143),(100.145,-58.507),
        (101.077,-58.871),(101.318,-58.965)])


class SeamRepairTests(unittest.TestCase):
    def test_canonical_repair_does_not_pin_the_erroneous_tooth_envelope(self):
        from engine.canonical_road_surface import Chain,_repair_canonical_shapes
        axis=teeth();width=6.
        chain=Chain(axis,np.array([0.,axis.length]),np.array([width,width]),np.ones(2),
            (0,),((0,*axis.coords[0]),(0,*axis.coords[-1])),(0,))
        audit=_repair_canonical_shapes([chain],[chain.widths.copy()],None)
        self.assertTrue(audit[0]['corrected'])
        self.assertFalse(axis_quality(chain.axis,width)['abnormal'])
        self.assertGreater(chain.axis.hausdorff_distance(axis),width/2)
        self.assertTrue(chain.axis.boundary.equals(axis.boundary))

    def test_single_period_and_finalization_share_polygon_serialization(self):
        import geopandas as gpd
        import tempfile
        from shapely.geometry import Polygon
        from engine.formal_road_products import export_polygons
        from engine.fast_gt_reconciliation import _export_polygons
        self.assertIs(_export_polygons,export_polygons)
        # Nearly coincident rings can cross after CRS/OGR floating arithmetic.
        shape=Polygon([(114,23),(114.001,23),(114.001,23.001),(114,23.001)],
                      [[(114.0004,23-1e-12),(114.0006,23-1e-12),
                        (114.0006,23.0001),(114.0004,23.0001)]])
        frame=gpd.GeoDataFrame(geometry=[shape],crs=4326)
        self.assertFalse(frame.is_valid.all())
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'surface.shp';export_polygons(frame).to_file(target)
            self.assertTrue(gpd.read_file(target).is_valid.all())

    def test_local_pass_is_not_a_completed_network_repair(self):
        axis=teeth()
        first,_=_repair_axis_pass(axis,6.)
        self.assertTrue(axis_quality(first,6.)['abnormal'])
        final,report=repair_axis(axis,6.)
        self.assertFalse(axis_quality(final,6.)['abnormal'])
        self.assertGreater(len(report['repair_passes']),1)
        self.assertEqual(final.coords[0],axis.coords[0])
        self.assertEqual(final.coords[-1],axis.coords[-1])
        self.assertEqual(list(final.coords)[:3],list(axis.coords)[:3])
        self.assertEqual(list(final.coords)[-4:],list(axis.coords)[-4:])

    def test_network_composes_width_station_mapping(self):
        axis=teeth();final,audit=repair_network_axes([axis],[6.],None)
        self.assertFalse(axis_quality(final[0],6.)['abnormal'])
        mapping=audit[0]['station_map']
        self.assertTrue(np.all(np.diff(mapping['before'])>0))
        self.assertTrue(np.all(np.diff(mapping['after'])>=0))
        self.assertAlmostEqual(mapping['before'][-1],axis.length)
        self.assertAlmostEqual(mapping['after'][-1],final[0].length)
        # Stable original vertices beyond the repair retain their physical
        # location when an observed width station is carried to the new axis.
        for xy in list(axis.coords)[:3]+list(axis.coords)[-4:]:
            from shapely.geometry import Point
            old=axis.project(Point(xy));new=np.interp(old,mapping['before'],mapping['after'])
            self.assertLess(final[0].interpolate(new).distance(Point(xy)),1e-6)

    def test_fix_is_translation_rotation_and_digitization_independent(self):
        for angle in (0,47,137):
            axis=translate(rotate(teeth(),angle,origin=(0,0)),400000,3500000)
            for reverse in (False,True):
                local=LineString(list(axis.coords)[::-1]) if reverse else axis
                with self.subTest(angle=angle,reverse=reverse):
                    final,_=repair_axis(local,6.)
                    self.assertFalse(axis_quality(final,6.)['abnormal'])
                    self.assertEqual(final.coords[0],local.coords[0])
                    self.assertEqual(final.coords[-1],local.coords[-1])

    def test_multiple_passes_keep_fragment_boundaries_connected(self):
        from shapely.ops import substring,linemerge
        axis=teeth();stations=[0.,40.,70.,100.,150.,axis.length]
        parts=[substring(axis,a,b) for a,b in zip(stations,stations[1:])]
        parts=[LineString(list(p.coords)[::-1]) if i%2 else p for i,p in enumerate(parts)]
        final,audit=repair_network_axes(parts,[6.]*len(parts),None)
        merged=linemerge(final)
        self.assertEqual(merged.geom_type,'LineString')
        self.assertFalse(axis_quality(merged,6.)['abnormal'])
        self.assertTrue(merged.boundary.equals(axis.boundary))
        self.assertTrue(audit)


if __name__=='__main__':unittest.main()
