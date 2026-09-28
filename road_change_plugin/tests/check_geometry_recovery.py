"""Feature failures must not abort formal period export; no model execution."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from check_formal_period_products import FormalPeriodTests,line,gpd
from check_axis_repair_seams import teeth
from shapely.affinity import translate
from shapely.geometry import LineString
from engine import canonical_road_surface as surface
from engine.road_axis_quality import AxisQualityError,axis_quality,conservative_axis_parts
from engine.formal_road_products import reconstruct_frames,formal_products_current


class RecoveryExportTests(unittest.TestCase):
    setUp=FormalPeriodTests.setUp
    regional=FormalPeriodTests.regional
    preview=FormalPeriodTests.preview
    export=FormalPeriodTests.export

    def audits(self,result):
        return [json.loads(Path(result[key]).read_text(encoding='utf8')) for key in
                ('axis_quality_audit','surface_quality_audit')]

    def test_all_axis_repairs_fail_period_still_exports(self):
        healthy=line(0,120,200);bad=teeth()
        self.regional([healthy,bad])
        error=AxisQualityError('forced local topology conflict')
        error.diagnostic={'attempts':[{'method':name} for name in
            ('local_low_frequency','surface_probability_refit','local_medial_refit',
             'local_vertex_refit','stable_direction_connection','conservative_chord')]}
        real=surface._chain_surface
        def safe(axis,*args,**kw):
            self.assertFalse(axis_quality(axis,8.)['abnormal'])
            return real(axis,*args,**kw)
        with patch('engine.road_axis_quality.repair_axis',side_effect=error),patch.object(surface,'_chain_surface',side_effect=safe):
            result=self.export()
        axes=gpd.read_file(result['centerlines'])
        self.assertTrue(any(a.equals(healthy) for a in axes.geometry))
        self.assertTrue(axes.is_valid.all())
        self.assertTrue(gpd.read_file(result['surfaces']).is_valid.all())
        axis,surf=self.audits(result)
        self.assertGreater(axis['summary']['fallback_count']+axis['summary']['skipped_count'],0)
        self.assertTrue(axis['failures'][0]['attempted_methods'])
        self.assertTrue(all(r['source_ids']==[1] for r in axis['failures']))
        self.assertGreater(surf['summary']['warning_count'],0)
        self.assertTrue(formal_products_current(self.output))

    def test_unrecoverable_chain_is_not_sent_to_corridor(self):
        healthy=line(0,120,200);self.regional([healthy,teeth()])
        with patch('engine.road_axis_quality.repair_axis',side_effect=AxisQualityError('failed')), \
             patch('engine.road_axis_quality.conservative_axis_parts',return_value=([],[
                 dict(repair_status='skipped',attempted_methods=['topology_anchored_local_chord'])])):
            result=self.export()
        axes=gpd.read_file(result['centerlines'])
        self.assertEqual(len(axes),1)
        self.assertTrue(axes.geometry.iloc[0].equals(healthy))
        axis,_=self.audits(result)
        self.assertEqual(axis['summary']['skipped_count'],1)

    def test_bad_corridor_skips_only_its_chain(self):
        self.regional([line(0,100),line(0,100,100)])
        real=surface._chain_surface
        def corridor(axis,*args,**kw):
            if axis.centroid.y>50:raise AxisQualityError('forced corridor failure')
            return real(axis,*args,**kw)
        with patch.object(surface,'_chain_surface',side_effect=corridor):result=self.export()
        self.assertEqual(len(gpd.read_file(result['centerlines'])),1)
        _,audit=self.audits(result)
        self.assertEqual(audit['summary']['skipped_count'],1)
        self.assertEqual(audit['failures'][0]['source_ids'],[1])

    def test_bad_junction_keeps_branch_bodies(self):
        self.regional([line(-100,0),line(0,100),LineString([(0,0),(0,100)])])
        with patch.object(surface,'_junction',side_effect=ValueError('forced junction failure')):result=self.export()
        self.assertGreater(len(gpd.read_file(result['centerlines'])),0)
        self.assertTrue(gpd.read_file(result['surfaces']).is_valid.all())
        _,audit=self.audits(result)
        self.assertTrue(any(r['stage']=='junction_footprint' for r in audit['failures']))

    def test_generalization_failure_retains_legal_surface(self):
        self.regional([line(0,100)])
        with patch.object(surface,'_generalize',side_effect=ValueError('forced boundary failure')):result=self.export()
        self.assertFalse(gpd.read_file(result['surfaces']).empty)
        _,audit=self.audits(result)
        self.assertEqual(audit['summary']['fallback_count'],1)

    def test_system_errors_are_not_downgraded(self):
        self.regional([line(0,100)])
        with patch.object(surface,'_chain_surface',side_effect=PermissionError('unwritable')):
            with self.assertRaises(PermissionError):self.export()
        with patch.object(surface,'_chain_surface',side_effect=MemoryError('exhausted')):
            with self.assertRaises(MemoryError):self.export()
        frame=gpd.GeoDataFrame(geometry=[line(0,100)])
        with self.assertRaisesRegex(ValueError,'projected metric CRS'):reconstruct_frames(frame,frame)

    def test_fault_isolated_without_losing_healthy_approaches(self):
        axis=LineString([(0,0),(30,0),(30,5),(35,5),(35,0),(40,0),(40,5),(45,5),(45,0),(80,0)])
        obstacle=LineString([(25,2),(50,2)])
        pieces,issues=conservative_axis_parts(axis,6,[obstacle],AxisQualityError('failed'))
        self.assertTrue(pieces)
        self.assertEqual(pieces[0][2].coords[0],axis.coords[0])
        self.assertEqual(pieces[-1][2].coords[-1],axis.coords[-1])
        self.assertTrue(all(not axis_quality(p,6)['abnormal'] for _,_,p in pieces))
        self.assertTrue(issues)

    def test_bugfix_does_not_restart_completed_period_export(self):
        self.regional([line(0,100)]);first=self.export()
        with patch('engine.formal_road_products.implementation_signature',return_value=['new bugfix']), \
             patch('engine.formal_road_products.reconstruct_frames',side_effect=AssertionError('completed period rebuilt')):
            self.assertTrue(formal_products_current(self.output))
            self.assertEqual(self.export(),first)

    def test_missing_checkpoint_still_retries_failed_export(self):
        self.regional([line(0,100)]);self.export()
        (self.output/'fast_export_cache.json').unlink()
        self.assertFalse(formal_products_current(self.output))
        with patch('engine.formal_road_products.reconstruct_frames',wraps=reconstruct_frames) as rebuild:
            self.export()
            self.assertEqual(rebuild.call_count,1)

    def test_canonical_chain_failure_does_not_drop_other_component(self):
        self.regional([line(0,100),line(0,100,100)])
        real=surface._chain
        def fail_one(edges,route):
            if edges[route[0][0]].axis.centroid.y>50:raise ValueError('disconnected graph edge')
            return real(edges,route)
        with patch.object(surface,'_chain',side_effect=fail_one):result=self.export()
        self.assertEqual(len(gpd.read_file(result['centerlines'])),1)
        _,audit=self.audits(result)
        self.assertTrue(any(r['stage']=='canonical_chain' for r in audit['failures']))

    def test_all_skipped_geometries_export_empty_layers_with_warnings(self):
        self.regional([line(0,100)])
        with patch.object(surface,'_chain_surface',side_effect=AxisQualityError('unrepairable corridor')):result=self.export()
        self.assertTrue(gpd.read_file(result['centerlines']).empty)
        self.assertTrue(gpd.read_file(result['surfaces']).empty)
        _,audit=self.audits(result)
        self.assertGreater(audit['summary']['skipped_count'],0)

    def test_bad_projected_polygon_is_isolated(self):
        from engine.formal_road_products import export_polygons
        from shapely.geometry import box
        from shapely import set_precision as real
        frame=gpd.GeoDataFrame(geometry=[box(0,0,10,10),box(100,100,110,110)],crs=3857)
        audit=dict(summary={})
        def fail_one(geom,*args,**kwargs):
            if geom.centroid.x>50:raise ValueError('projection overlay failure')
            return real(geom,*args,**kwargs)
        with patch('shapely.set_precision',side_effect=fail_one):out=export_polygons(frame,audit)
        self.assertEqual(len(out),1)
        self.assertEqual(audit['summary']['skipped_count'],1)
        self.assertEqual(audit['failures'][0]['source_ids'],[1])


del FormalPeriodTests
if __name__=='__main__':unittest.main()
