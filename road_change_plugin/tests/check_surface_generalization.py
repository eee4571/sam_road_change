"""Bound regional overlays and retain road topology across tile seams."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from shapely import union_all,segmentize,get_num_coordinates
from shapely.geometry import LineString,MultiPolygon,box,Point
from shapely.geometry.base import BaseGeometry
from engine import canonical_road_surface as surface
from engine.geometry_recovery import Recovery,recovery_scope


def chain(axis):
    return surface.Chain(axis,np.array([0.,axis.length]),np.array([6.,6.]),np.ones(2),
                         (0,),((0,*axis.coords[0]),(0,*axis.coords[-1])),(0,))


class GeneralizationTests(unittest.TestCase):
    def test_export_drops_precision_collapsed_slivers(self):
        import geopandas as gpd
        from engine.formal_road_products import export_polygons
        source=gpd.GeoDataFrame({'id':[1,2]},geometry=[box(0,0,1,1),box(2,0,2.000001,1)],crs=3857)
        result=export_polygons(source)
        self.assertEqual(result['id'].tolist(),[1])
        self.assertTrue(result.is_valid.all())
        self.assertFalse(result.is_empty.any())

    def test_enumeration_does_not_redissolve_overlay_slivers(self):
        geometry=MultiPolygon([box(i*2,0,i*2+1,1) for i in range(500)])
        with patch.object(surface,'polygonal',side_effect=AssertionError('unnecessary dissolve')):
            parts=surface._polygons(geometry)
        self.assertEqual(len(parts),500)
        self.assertAlmostEqual(sum(p.area for p in parts),geometry.area)

    def test_coverage_prepares_region_once(self):
        region=box(0,0,1000,1000)
        chains=[SimpleNamespace(axis=LineString([(0,y),(1000,y)])) for y in range(1000)]
        real=BaseGeometry.buffer;calls=[]
        def count(geom,*args,**kwargs):
            if geom is region:calls.append(1)
            return real(geom,*args,**kwargs)
        with patch.object(BaseGeometry,'buffer',count):self.assertTrue(surface._axes_covered(region,chains))
        self.assertEqual(len(calls),1)

    def test_tiled_surface_retains_axes_loops_and_seams(self):
        axes=[LineString([(0,y),(200,y)]) for y in (0,100,200)]
        axes += [LineString([(x,0),(x,200)]) for x in (0,100,200)]
        chains=[chain(a) for a in axes]
        region=segmentize(union_all([a.buffer(3) for a in axes]),.25)
        real=surface._generalize_small;work=[]
        def bounded(geometry,*args):
            work.append(get_num_coordinates(geometry))
            return real(geometry,*args)
        with patch.object(surface,'GENERALIZATION_VERTEX_BUDGET',3000), \
             patch.object(surface,'_generalize_small',side_effect=bounded), recovery_scope(Recovery()):
            result,audit=surface._generalize(region,[],chains)
        self.assertGreater(audit['generalization_tiles'],1)
        self.assertTrue(work)
        self.assertLessEqual(max(work),3000)
        self.assertTrue(result.is_valid)
        self.assertTrue(surface._axes_covered(result,chains))
        self.assertEqual(result.geom_type,'Polygon')
        for x in (50,150):
            for y in (50,150):self.assertFalse(result.covers(Point(x,y)))
        self.assertLess(abs(result.area-region.area)/region.area,.03)

    def test_dense_locality_preserves_valid_surface_with_audit(self):
        region=Point(0,0).buffer(20,quad_segs=100)
        recovery=Recovery()
        with patch.object(surface,'GENERALIZATION_VERTEX_BUDGET',10), \
             patch.object(surface,'_generalize_small',side_effect=AssertionError('unbounded work')), recovery_scope(recovery):
            result,audit=surface._generalize(region,[],[chain(LineString([(-10,0),(10,0)]))])
        self.assertTrue(result.equals(region))
        self.assertEqual(audit['complexity_fallback_count'],1)
        self.assertEqual(recovery.failures[0]['repair_status'],'fallback')

    def test_tiling_does_not_bridge_parallel_components(self):
        axes=[LineString([(0,y),(300,y)]) for y in (0,7)]
        region=segmentize(union_all([a.buffer(3,cap_style='flat') for a in axes]),.25)
        with patch.object(surface,'GENERALIZATION_VERTEX_BUDGET',1500), recovery_scope(Recovery()):
            result,audit=surface._generalize(region,[],[chain(a) for a in axes])
        self.assertGreater(audit['generalization_tiles'],1)
        self.assertEqual(len(surface._polygons(result)),2)
        self.assertFalse(result.covers(Point(150,3.5)))


if __name__=='__main__':unittest.main()
