"""Final road-surface smoke tests using synthetic saved evidence only."""
import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from engine.road_surface_quality import stable_width
from engine.canonical_road_surface import build_road_surface
from engine.auto_change_geometry import FinalWidths
from engine.continuous_road_geometry import network_surface
from engine.road_connection_evidence import ConnectionEvidence


def line(x0, x1, y=0):
    return LineString([(x0, y), (x1, y)])


class Probability:
    def __init__(self, value):self.value=value
    def values(self, xy):return np.full(len(xy), self.value)


class SurfaceQualityTests(unittest.TestCase):
    def test_canonical_chain_preserves_original_curve(self):
        x=np.linspace(0.,100.,201)
        a=LineString(np.c_[x,10*np.sin(x/40)])
        _,audit=build_road_surface([(a,[0.,a.length],[8.,8.])],[[1,1]])
        from shapely import from_wkt
        self.assertTrue(from_wkt(audit['chains'][0]['axis_wkt']).equals_exact(a,1e-10))

    def test_frequent_width_noise_is_not_a_published_boundary(self):
        s=np.arange(0.,202.,2.);v=8.+2*np.sin(s*.8)
        result,qa=stable_width(s,v,np.ones(len(s)))
        self.assertLess(np.ptp(result),.01)
        self.assertEqual(qa['sections'],1)

    def test_bad_width_hole_inherits_reliable_approaches(self):
        s=np.arange(0.,202.,2.);v=np.full(len(s),8.);q=np.ones(len(s))
        bad=(s>70)&(s<110);v[bad]=.4;q[bad]=0
        result,_=stable_width(s,v,q)
        np.testing.assert_allclose(result,8.)

    def test_sustained_evidenced_width_change_is_retained(self):
        s=np.arange(0.,302.,2.);v=np.where(s<150.,6.,12.)
        result,qa=stable_width(s,v,np.ones(len(s)))
        self.assertAlmostEqual(result[0],6.);self.assertAlmostEqual(result[-1],12.)
        self.assertEqual(qa['sections'],2)
        self.assertLess(np.max(np.abs(np.diff(result))),1.)

    def test_unknown_width_does_not_create_false_confidence(self):
        s=np.arange(0.,102.,2.);v=np.full(len(s),8.);v[25]=.2
        result,qa=stable_width(s,v,np.zeros(len(s)))
        np.testing.assert_allclose(result,8.)
        self.assertEqual(qa['method'],'unverified_representative')

    def test_sustained_middle_widening_is_not_flattened(self):
        s=np.arange(0.,402.,2.);v=np.where((s>=140)&(s<260),12.,8.)
        result,qa=stable_width(s,v,np.ones(len(s)))
        self.assertEqual(qa['sections'],3)
        self.assertAlmostEqual(result[100],12.)
        self.assertAlmostEqual(result[0],8.);self.assertAlmostEqual(result[-1],8.)

    def test_quality_sample_does_not_reuse_parallel_road(self):
        frame=gpd.GeoDataFrame(dict(width_m=[8.,30.],quality_grade=['C','A']),
            geometry=[line(0,100),line(0,100,4)],crs=3857)
        s,v,q=FinalWidths(frame).profile_quality(line(0,100),6.)
        np.testing.assert_allclose(v,8.);np.testing.assert_allclose(q,0.)

    def test_degree_two_splits_do_not_become_width_units(self):
        profiles=[(line(0,40),[0.,40.],[8.,8.]),(line(40,48),[0.,8.],[2.,2.]),
                  (line(48,100),[0.,52.],[8.,8.])]
        surface,qa=build_road_surface(profiles,[[1,1],[0,0],[1,1]])
        self.assertLess(surface.symmetric_difference(box(0,-4,100,4)).area,.01)
        self.assertEqual(len(qa['chains']),1)

    def test_junction_portals_are_tangent_bounded_and_connected(self):
        axes=[line(0,50),line(0,-50),LineString([(0,0),(0,50)])]
        profiles=[(a,[0.,a.length],[8.,8.]) for a in axes]
        raw=network_surface(profiles)
        surface,_=build_road_surface(profiles,[[1.,1.]]*3)
        self.assertTrue(surface.is_valid);self.assertEqual(surface.geom_type,'Polygon')
        self.assertEqual(len(surface.interiors),0)
        self.assertEqual(surface.bounds,raw.bounds)
        delta=surface.difference(raw)
        self.assertGreater(delta.area,0.)
        self.assertLess(delta.area,25.)
        self.assertLess(delta.difference(box(-9,-9,9,9)).area,1e-8)

    def test_nearby_parallel_roads_remain_separate(self):
        profiles=[(line(0,100,y),[0.,100.],[4.,4.]) for y in (0,8)]
        result,_=build_road_surface(profiles,[[1,1],[1,1]])
        self.assertEqual(len(result.geoms),2)

    def test_unsplit_interior_junction_is_not_moved_by_drawing_axis(self):
        a=LineString([(-50,0),(-10,.4),(0,0),(10,-.4),(50,0)])
        b=LineString([(0,-30),(0,30)])
        profiles=[(p,[0.,p.length],[8.,8.]) for p in (a,b)]
        result,qa=build_road_surface(profiles,[[1,1],[1,1]])
        self.assertTrue(result.is_valid)
        self.assertEqual(qa['summary']['canonical_chain_count'],4)
        self.assertEqual(qa['summary']['reconstructed_junction_count'],1)

    def classify(self,axes=None,prob=0.,other=None,grades=None,protected=()):
        axes=axes or [line(0,8)]
        frame=gpd.GeoDataFrame(dict(width_m=[6.]*len(axes),quality_grade=grades or ['C']*len(axes)),geometry=axes,crs=3857)
        profiles=[(a,*FinalWidths(frame).profile_quality(a,6.)[:2]) for a in axes]
        quality=[FinalWidths(frame).profile_quality(a,6.)[2] for a in axes]
        _,qa=build_road_surface(profiles,quality,
            metadata=[{'protect_surface':i in protected} for i in range(len(axes))],
            evidence=ConnectionEvidence(box(200,200,210,210),Probability(prob)),
            observations=[other] if other is not None else ())
        removed={i for r in qa['spurs'] if r['action']=='suppress' for i in r['source_features']}
        return removed,qa['spurs']

    def test_short_noise_needs_all_evidence(self):
        removed,audit=self.classify(other=[line(100,200)])
        self.assertEqual(removed,{0});self.assertEqual(audit[0]['action'],'suppress')

    def test_missing_temporal_or_probability_is_not_negative(self):
        self.assertFalse(self.classify()[0])
        self.assertFalse(self.classify(prob=np.nan,other=[line(100,200)])[0])

    def test_true_isolated_road_with_any_support_is_preserved(self):
        self.assertFalse(self.classify(prob=.8,other=[line(100,200)])[0])
        self.assertFalse(self.classify(other=[line(0,8)])[0])
        self.assertFalse(self.classify(other=[line(100,200)],grades=['A'])[0])

    def test_event_track_and_short_connected_branch_are_preserved(self):
        self.assertFalse(self.classify(other=[line(100,200)],protected=[0])[0])
        self.assertFalse(self.classify(axes=[line(0,8),line(8,100)],other=[line(200,300)])[0])

    def test_fragmented_long_component_is_not_short_noise(self):
        self.assertFalse(self.classify(axes=[line(x,x+8) for x in range(0,80,8)],other=[line(200,300)])[0])

    def test_overlap_and_duplicate_fragments_collapse_to_one_chain(self):
        profiles=[(line(a,b),[0.,b-a],[8.,8.]) for a,b in [(0,60),(40,100),(0,60)]]
        surface,qa=build_road_surface(profiles,[[1,1]]*3)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)
        self.assertEqual(qa['summary']['merged_pseudo_node_count'],2)
        self.assertLess(surface.symmetric_difference(box(0,-4,100,4)).area,1e-8)

    def test_acute_corner_and_different_roads_are_not_merged(self):
        profiles=[(line(0,50),[0.,50.],[8.,8.]),
                  (LineString([(50,0),(50,50)]),[0.,50.],[8.,8.])]
        _,qa=build_road_surface(profiles,[[1,1]]*2)
        self.assertEqual(qa['summary']['canonical_chain_count'],2)
        self.assertIn('direction_break',qa['merge_rejections'])
        profiles[1]=(line(50,100),[0.,50.],[8.,8.])
        _,qa=build_road_surface(profiles,[[1,1]]*2,metadata=[{'road_ref':'A'},{'road_ref':'B'}])
        self.assertEqual(qa['summary']['canonical_chain_count'],2)
        self.assertIn('different_road',qa['merge_rejections'])

    def test_same_level_crossing_is_noded_but_overpass_is_not(self):
        profiles=[(line(-50,50),[0.,100.],[8.,8.]),
                  (LineString([(0,-50),(0,50)]),[0.,100.],[8.,8.])]
        _,qa=build_road_surface(profiles,[[1,1]]*2)
        self.assertEqual(qa['summary']['canonical_chain_count'],4)
        self.assertEqual(qa['summary']['reconstructed_junction_count'],1)
        _,qa=build_road_surface(profiles,[[1,1]]*2,metadata=[{'layer':0},{'layer':1}])
        self.assertEqual(qa['summary']['canonical_chain_count'],2)
        self.assertEqual(qa['summary']['reconstructed_junction_count'],0)

    def test_oblique_interior_crossing_keeps_source_correspondence(self):
        axes=[LineString([(0.13,0.41),(101.37,17.63)]),
              LineString([(32.73,-17.37),(64.17,48.91)])]
        _,qa=build_road_surface([(a,[0.,a.length],[6.,6.]) for a in axes],[[1,1]]*2)
        self.assertEqual(qa['summary']['canonical_chain_count'],4)
        self.assertTrue(qa['summary']['active_axis_covered'])

    def test_reciprocal_collinear_roundoff_gap_but_not_parallel_snap(self):
        profiles=[(line(0,50),[0.,50.],[8.,8.]),(line(50.1,100),[0.,49.9],[8.,8.])]
        _,qa=build_road_surface(profiles,[[1,1]]*2)
        self.assertEqual(qa['summary']['roundoff_gap_count'],1)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)
        profiles[1]=(line(0,50,.1),[0.,50.],[8.,8.])
        _,qa=build_road_surface(profiles,[[1,1]]*2)
        self.assertEqual(qa['summary']['roundoff_gap_count'],0)
        self.assertEqual(qa['summary']['canonical_chain_count'],2)

    def test_negative_terminal_tooth_is_removed_without_touching_main_road(self):
        main=line(-50,50);tooth=LineString([(0,0),(0,8)])
        profiles=[(main,[0.,100.],[8.,8.]),(tooth,[0.,8.],[6.,6.])]
        original=[p[0].wkb for p in profiles]
        surface,qa=build_road_surface(profiles,[[1,1],[0,0]],
            evidence=ConnectionEvidence(None,Probability(0.)),observations=[[main]])
        self.assertEqual(qa['summary']['removed_short_spur_count'],1)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)
        self.assertLess(surface.symmetric_difference(box(-50,-4,50,4)).area,1e-8)
        self.assertEqual(original,[p[0].wkb for p in profiles])
        _,kept=build_road_surface(profiles,[[1,1],[0,0]],
            evidence=ConnectionEvidence(None,Probability(0.)),observations=[[main,tooth]])
        self.assertEqual(kept['summary']['removed_short_spur_count'],0)

    def test_branch_bodies_are_trimmed_before_offsetting(self):
        from unittest.mock import patch
        from engine.auto_change_geometry import corridor
        axes=[line(0,50),line(0,-50),LineString([(0,0),(0,50)])]
        captured=[]
        def record(a,s,w):captured.append(a);return corridor(a,s,w)
        with patch('engine.canonical_road_surface.corridor',side_effect=record):
            _,qa=build_road_surface([(a,[0.,a.length],[8.,8.]) for a in axes],[[1,1]]*3)
        from shapely.geometry import Point
        self.assertTrue(all(a.distance(Point(0,0))>=7.99 for a in captured))
        self.assertEqual(qa['summary']['reconstructed_junction_count'],1)

    def test_junction_internal_micro_link_is_not_a_separate_width_unit(self):
        axes=[line(-50,0),line(0,3),line(3,50),
              LineString([(0,0),(0,40)]),LineString([(3,0),(3,-40)])]
        result,qa=build_road_surface([(a,[0.,a.length],[8.,8.]) for a in axes],[[1,1]]*5)
        self.assertTrue(result.is_valid)
        self.assertEqual(qa['summary']['canonical_chain_count'],4)
        self.assertEqual(qa['summary']['junction_internal_chain_count'],1)
        self.assertEqual(qa['summary']['reconstructed_junction_count'],1)
        self.assertTrue(any(len(j['internal_edges'])==1 for j in qa['junctions']))

    def test_real_ring_remains_a_hole(self):
        ring=LineString([(0,0),(100,0),(100,100),(0,100),(0,0)])
        result,qa=build_road_surface([(ring,[0.,ring.length],[6.,6.])],[[1,1]])
        self.assertEqual(len(result.interiors),1)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)

    def test_collapsed_loop_junction_still_covers_its_internal_connectors(self):
        axes=[line(0,3),LineString([(0,0),(1.5,1),(3,0)]),
              LineString([(0,0),(0,-40),(3,-40),(3,0)])]
        surface,qa=build_road_surface([(a,[0.,a.length],[8.,8.]) for a in axes],[[1,1]]*3)
        self.assertTrue(qa['summary']['active_axis_covered'])
        self.assertTrue(surface.buffer(1e-6).covers(axes[1]))

    def test_final_export_and_cache_use_quality_not_bad_local_width(self):
        from engine.fast_gt_reconciliation import _write_final_period
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);out=root/'out';out.mkdir()
            base=gpd.GeoDataFrame(dict(width_m=[8.]),geometry=[line(0,100)],crs=3857)
            widths=gpd.GeoDataFrame(dict(width_m=[8.,.5,8.],quality_grade=['A','C','A']),
                geometry=[line(0,40),line(40,60),line(60,100)],crs=3857)
            source={}
            for key,frame in [('centerlines',base),('width_segments',widths),
                              ('surfaces',gpd.GeoDataFrame(geometry=[box(0,-4,100,4)],crs=3857))]:
                source[key]=str(root/f'{key}.gpkg');frame.to_file(source[key])
            rows=[dict(base.iloc[0],track_id='',_original_row=0)]
            result=_write_final_period(source,base,{},rows,[],out,3857,3857)
            surfaces=gpd.read_file(result['surfaces'])
            self.assertAlmostEqual(surfaces.area.sum(),800.,delta=.1)
            self.assertTrue(Path(result['surface_quality_audit']).is_file())
            self.assertEqual(result,_write_final_period(source,base,{},rows,[],out,3857,3857))


if __name__=='__main__':unittest.main()
