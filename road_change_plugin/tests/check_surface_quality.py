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
        self.assertGreaterEqual(delta.area,0.)
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

    def test_short_noise_topology_rule(self):
        removed,audit=self.classify(other=[line(100,200)])
        self.assertEqual(removed,{0});self.assertEqual(audit[0]['action'],'suppress')

    def test_missing_evidence_does_not_preserve_short_noise(self):
        self.assertEqual(self.classify()[0],{0})
        self.assertEqual(self.classify(prob=np.nan,other=[line(100,200)])[0],{0})

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
        self.assertEqual(qa['summary']['canonical_chain_count'],1)
        self.assertEqual(qa['summary']['near_duplicate_merge_count'],1)

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
        from engine.canonical_road_surface import _chain_surface
        axes=[line(0,50),line(0,-50),LineString([(0,0),(0,50)])]
        captured=[]
        def record(a,s,w,**kwargs):
            if kwargs.get('check_axis',True):captured.append(a)
            return _chain_surface(a,s,w,**kwargs)
        with patch('engine.canonical_road_surface._chain_surface',side_effect=record):
            _,qa=build_road_surface([(a,[0.,a.length],[8.,8.]) for a in axes],[[1,1]]*3)
        from shapely.geometry import Point
        self.assertEqual(len(captured),3)
        self.assertTrue(all(3.99<=a.distance(Point(0,0))<=4.01 for a in captured))
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

    def test_fragmented_internal_loop_is_one_physical_junction(self):
        axes=[line(-50,0),line(0,6),line(6,50),
              LineString([(0,0),(3,4)]),LineString([(3,4),(6,0)]),
              LineString([(0,0),(0,50)]),LineString([(6,0),(6,-50)])]
        surface,qa=build_road_surface([(a,[0,a.length],[20,20]) for a in axes],[[1,1]]*len(axes))
        self.assertEqual(qa['summary']['reconstructed_junction_count'],1)
        self.assertEqual(qa['summary']['semantic_boundary_count'],0)
        self.assertGreaterEqual(qa['summary']['junction_internal_chain_count'],2)
        self.assertTrue(qa['summary']['active_axis_covered'])
        self.assertTrue(surface.is_valid)

    def test_real_ring_remains_a_hole(self):
        ring=LineString([(0,0),(100,0),(100,100),(0,100),(0,0)])
        result,qa=build_road_surface([(ring,[0.,ring.length],[6.,6.])],[[1,1]])
        self.assertEqual(len(result.interiors),1)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)

    def test_topology_prunes_whole_fragmented_spur_with_unknown_evidence(self):
        axes=[line(-50,50),LineString([(0,0),(0,5)]),LineString([(0,5),(0,12)])]
        _,qa=build_road_surface([(a,[0,a.length],[6,6]) for a in axes],[[-1,-1]]*3)
        self.assertEqual(qa['summary']['removed_short_spur_count'],1)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)
        self.assertTrue(qa['summary']['active_axis_covered'])

    def test_small_isolated_u_and_loop_removed_but_event_component_kept(self):
        axes=[LineString([(0,0),(0,5),(3,5),(3,0)]),
              LineString([(30,0),(34,0),(34,4),(30,4),(30,0)]),line(50,58)]
        _,qa=build_road_surface([(a,[0,a.length],[6,6]) for a in axes],[[-1,-1]]*3,
            metadata=[{},{},{'track_id':'event-road'}])
        self.assertEqual(qa['summary']['removed_short_component_count'],2)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)

    def test_attached_small_loop_is_not_retained_by_missing_evidence(self):
        axes=[line(-50,50),LineString([(0,0),(-2,4),(2,4),(0,0)])]
        _,qa=build_road_surface([(a,[0,a.length],[6,6]) for a in axes],[[-1,-1]]*2)
        self.assertEqual(qa['summary']['removed_short_remnant_count'],1)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)

    def test_reconnecting_triangle_tooth_removed_without_cutting_main_road(self):
        axes=[line(-50,50),LineString([(0,0),(5,4),(10,0)])]
        surface,qa=build_road_surface([(a,[0,a.length],[8,8]) for a in axes],[[-1,-1]]*2)
        self.assertGreater(qa['summary']['removed_short_remnant_count'],0)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)
        self.assertAlmostEqual(surface.symmetric_difference(box(-50,-4,50,4)).area,0.,places=5)
        self.assertTrue(qa['summary']['active_axis_covered'])

    def test_near_duplicate_collapse_reattaches_branch_to_unique_axis(self):
        axes=[line(0,100),line(0,100,1),LineString([(50,1),(50,50)])]
        _,qa=build_road_surface([(a,[0,a.length],[8,8]) for a in axes],[[1,1]]*3)
        self.assertEqual(qa['summary']['near_duplicate_merge_count'],1)
        self.assertEqual(qa['summary']['canonical_chain_count'],3)
        self.assertEqual(qa['summary']['reconstructed_junction_count'],1)
        self.assertTrue(qa['summary']['active_axis_covered'])
        self.assertEqual(axes[2].coords[0],(50.,1.))

    def test_wide_same_track_duplicate_not_limited_to_two_metres(self):
        profiles=[(line(0,100),[0,100],[24,24]),(line(10,90,9),[0,80],[22,22])]
        surface,qa=build_road_surface(profiles,[[-1,-1]]*2)
        self.assertEqual(qa['summary']['near_duplicate_merge_count'],1)
        self.assertEqual(qa['summary']['canonical_chain_count'],1)
        self.assertLess(surface.bounds[3],13.)

    def test_variable_width_canonical_sweep_covers_tight_continuous_bend(self):
        from engine.canonical_road_surface import _chain_surface
        t=np.linspace(0,1.7*np.pi,120)
        axis=LineString(np.c_[10*np.cos(t),10*np.sin(t)])
        s=np.linspace(0,axis.length,100);w=np.linspace(18,26,100)
        surface=_chain_surface(axis,s,w)
        self.assertTrue(surface.is_valid)
        self.assertTrue(surface.buffer(1e-5).covers(axis))

    def test_constant_width_uses_occupied_sweep_at_a_short_returning_tip(self):
        from engine.canonical_road_surface import _chain_surface
        axis=LineString([(0,0),(99,0),(100,.05),(100.15,.06),(100,-.1)])
        surface=_chain_surface(axis,[0,axis.length],[30,30])
        self.assertTrue(surface.is_valid)
        self.assertTrue(surface.buffer(1e-5).covers(axis))
        self.assertFalse(surface.interiors)

    def test_one_portal_junction_complex_keeps_internal_axes(self):
        axes=[line(-50,0),line(0,3),LineString([(0,0),(1.5,1),(3,0)]),
              LineString([(0,0),(1.5,-1),(3,0)])]
        surface,qa=build_road_surface([(a,[0,a.length],[8,8]) for a in axes],[[1,1]]*4)
        self.assertTrue(qa['summary']['active_axis_covered'])
        self.assertTrue(all(surface.buffer(1e-4).covers(a) for a in axes))

    def test_declared_parallel_roads_and_levels_not_collapsed(self):
        profiles=[(line(0,100,y),[0,100],[8,8]) for y in (0,1)]
        for metadata in ([{'road_ref':'A'},{'road_ref':'B'}],[{'layer':0},{'layer':1}],
                         [{'track_id':'event-A'},{'track_id':'event-B'}]):
            _,qa=build_road_surface(profiles,[[1,1]]*2,metadata=metadata)
            self.assertEqual(qa['summary']['near_duplicate_merge_count'],0)

    def test_divergent_fork_is_not_a_same_track_duplicate(self):
        axes=[line(0,100),LineString([(0,0),(25,1),(50,5),(100,20)])]
        _,qa=build_road_surface([(a,[0,a.length],[6,6]) for a in axes],[[1,1]]*2)
        self.assertEqual(qa['summary']['near_duplicate_merge_count'],0)

    def test_long_shared_approach_collapses_but_divergent_tail_is_preserved(self):
        from engine.canonical_road_surface import _collapse_duplicates
        axes=[line(0,150),LineString([(0,0),(80,2),(100,25)])]
        profiles=[(a,[0,a.length],[20,20]) for a in axes]
        result,_,audit=_collapse_duplicates(profiles,[[1,1]]*2,[{},{}])
        self.assertEqual(len(audit),1)
        self.assertTrue(audit[0]['partial_overlap'])
        self.assertGreater(audit[0]['overlap_m'],70.)
        self.assertEqual(result[1][0].coords[-1],(100.,25.))
        self.assertGreater(result[0][0].intersection(result[1][0]).length,70.)

    def test_through_road_endpoint_widths_coordinated_not_cross_road(self):
        axes=[line(-50,0),line(0,50),LineString([(0,0),(0,50)])]
        profiles=[(a,[0,a.length],[w,w]) for a,w in zip(axes,[8,12,4])]
        _,qa=build_road_surface(profiles,[[1,1]]*3)
        self.assertEqual(qa['summary']['width_coordination_count'],1)
        self.assertEqual(qa['width_coordination'][0]['width_m'],10.)
        self.assertEqual(sorted(round(p['width_m'],3) for p in qa['junctions'][0]['portals'])[0],4.)
        _,qa=build_road_surface(profiles,[[1,1]]*3,metadata=[{'track_id':'widened'}, {}, {}])
        self.assertEqual(qa['summary']['width_coordination_count'],0)

    def test_junction_fallback_cannot_fill_empty_branch_envelope(self):
        from engine.canonical_road_surface import _junction
        axes=[LineString([(0,0),(8,0)]),LineString([(0,6),(8,6)])]
        portals=[dict(point=np.array([8.,y]),direction=np.array([1.,0.]),
            right=np.array([8.,y-1]),left=np.array([8.,y+1]),width=2.) for y in (0.,6.)]
        surface,method=_junction(portals,np.array([0.,3.]),axes)
        self.assertEqual(method,'local_branch_envelope')
        self.assertFalse(surface.intersects(box(1,2,7,4)))
        self.assertTrue(surface.buffer(1e-6).covers(axes[0]))

    def test_microholes_and_axis_free_slivers_removed(self):
        from engine.canonical_road_surface import _generalize,Chain
        from shapely import union_all
        axis=line(0,100)
        chain=Chain(axis,np.array([0,100]),np.array([8,8]),np.ones(2),(0,),((0,0,0),(0,100,0)),())
        geometry=union_all([box(0,-4,100,4).difference(box(45,2,45.1,2.1)),box(200,0,200.01,1)])
        result,qa=_generalize(geometry,[],[chain])
        self.assertEqual(result.geom_type,'Polygon')
        self.assertEqual(len(result.interiors),0)
        self.assertEqual(qa['micro_holes_removed'],1)
        self.assertEqual(qa['slivers_removed'],1)
        self.assertAlmostEqual(result.area,800.,places=6)

    def test_small_open_crack_closed_without_breaking_axis(self):
        from engine.canonical_road_surface import _generalize,Chain
        axis=line(0,100)
        chain=Chain(axis,np.array([0,100]),np.array([8,8]),np.ones(2),(0,),((0,0,0),(0,100,0)),())
        original=box(0,-4,100,4).difference(box(45,1,45.15,5))
        result,qa=_generalize(original,[],[chain])
        self.assertLess(800-result.area,.1)
        self.assertTrue(result.covers(axis));self.assertGreater(qa['narrow_cracks_closed'],0)

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

    def test_unreliable_short_neck_borrows_whole_through_road_width(self):
        axes=[line(-100,0),line(0,10),line(10,100),
              LineString([(0,0),(0,50)]),LineString([(10,0),(10,-50)])]
        profiles=[(a,[0,a.length],[w,w]) for a,w in zip(axes,[24,2,24,6,6])]
        surface,qa=build_road_surface(profiles,[[1,1],[-1,-1],[1,1],[1,1],[1,1]])
        narrow=next(r for r in qa['chains'] if 1 in r['source_features'])
        self.assertEqual(narrow['final_width_range_m'],[24.,24.])
        self.assertTrue(surface.covers(box(0,-10,10,10)))
        self.assertTrue(any(r['method']=='reliable_route_width' for r in qa['width_coordination']))
        # A lifecycle width is not an unreliable observation to replace.
        _,protected=build_road_surface(profiles,[[1,1],[-1,-1],[1,1],[1,1],[1,1]],
            metadata=[{}, {'track_id':'width-event'}, {}, {}, {}])
        narrow=next(r for r in protected['chains'] if 1 in r['source_features'])
        self.assertEqual(narrow['final_width_range_m'],[2.,2.])

    def test_fragment_boundaries_cannot_hide_alternating_steps(self):
        from shapely import from_wkt
        from engine.road_axis_quality import axis_quality
        xy=[(-60,0),(-10,0),(-10,5),(0,5),(0,-5),(10,-5),(10,0),(60,0)]
        axes=[LineString([a,b]) for a,b in zip(xy,xy[1:])]
        originals=[a.wkb for a in axes]
        surface,qa=build_road_surface([(a,[0,a.length],[12,12]) for a in axes],[[1,1]]*len(axes))
        self.assertEqual(qa['summary']['canonical_chain_count'],1)
        self.assertTrue(any(r.get('stage')=='before_directional_split' for r in qa['surface_axis_repairs']))
        axis=from_wkt(qa['chains'][0]['axis_wkt'])
        self.assertFalse(axis_quality(axis,12)['abnormal'])
        self.assertEqual({axis.coords[0],axis.coords[-1]}, {xy[0],xy[-1]})
        self.assertEqual(originals,[a.wkb for a in axes])
        self.assertTrue(surface.is_valid)

    def test_portal_body_and_junction_share_full_cap_on_tiny_turn(self):
        from engine.canonical_road_surface import Chain,_portal,_chain_surface,_junction
        from shapely.geometry import Point
        from shapely.ops import substring
        axis=LineString([(0,0),(4,0),(4.01,.01),(50,2)])
        chain=Chain(axis,np.array([0,axis.length]),np.array([8,8]),np.ones(2),(0,),(),())
        p=_portal(chain,0,4.)
        body=substring(axis,4.,axis.length)
        road=_chain_surface(body,[0,body.length],[8,8],start_direction=p['direction'])
        junction,_=_junction([p],np.array([0.,0.]),[substring(axis,0,4.)])
        cap=LineString([p['right'],p['left']])
        self.assertTrue(road.buffer(1e-6).covers(cap))
        self.assertTrue(junction.buffer(1e-6).covers(cap))
        self.assertEqual(road.union(junction).geom_type,'Polygon')
        self.assertTrue(road.union(junction).covers(Point(4,0)))

    def test_local_fits_cannot_introduce_a_global_chain_crossing(self):
        from unittest.mock import patch
        from engine.canonical_road_surface import _build_graph,_repair_degree_two
        xy=[(-60,0),(-10,0),(-10,5),(0,5),(0,-5),(10,-5),(10,0),(60,0)]
        axes=[LineString([a,b]) for a,b in zip(xy,xy[1:])]
        metadata=[{} for _ in axes]
        edges=_build_graph([(a,[0,a.length],[12,12]) for a in axes],[[1,1]]*len(axes),metadata)
        active=set(range(len(edges)));original=[e.axis.wkb for e in edges]
        crossed=LineString([(-60,0),(10,3),(-10,3),(10,-3),(-10,-3),(60,0)])
        self.assertFalse(crossed.is_simple)
        with patch('engine.road_axis_quality.repair_axis',return_value=(crossed,dict(corrected=True))):
            audit=_repair_degree_two(edges,active,metadata,None)
        self.assertEqual(audit,[])
        self.assertEqual(original,[e.axis.wkb for e in edges])
        self.assertEqual(active,set(range(len(edges))))


if __name__=='__main__':unittest.main()
