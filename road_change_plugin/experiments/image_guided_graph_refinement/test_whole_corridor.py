import unittest
import numpy as np
from shapely.geometry import LineString,box,Point
from whole_corridor import pavement_descriptor,fit_corridor,carriageway_axes,recurrent_dual_prior
from medial_repair import AxisProposal,replace_corridors,parallel_ribbon_veto
from shapely import union_all
from test_repair import frame
from test_block_graph import evidence
from run_corridor_experiment import replaced_extent_distance,road_units
from graph import build,candidates


class WholeCorridorTests(unittest.TestCase):
    def test_returning_fragment_outside_roi_is_immutable(self):
        old=frame([[(-90,-2),(90,-2)],[(-90,2),(90,2)],[(-90,-2),(-94,0),(-90,2)]])
        roi=box(-90,-40,90,40);axis=LineString([(-90,0),(90,0)])
        proposal=AxisProposal(axis,axis.buffer(5,cap_style='flat').intersection(roi),8.,1.,dict(whole_corridor=True),[])
        result,_,_=replace_corridors(old,[proposal],roi)
        outside=union_all(old.geometry).difference(roi)
        self.assertLess(outside.difference(union_all(result.geometry).buffer(.0001)).length,.001)

    def test_narrow_road_is_not_doubled(self):
        old=frame([[(-100,-2),(100,-2)],[(-100,2),(100,2)]])
        axis=LineString([(-100,0),(100,0)])
        proposal=AxisProposal(axis,axis.buffer(4),8.,.9,dict(whole_corridor=True),[0,1])
        self.assertEqual(len(carriageway_axes(proposal,old,box(-100,-30,100,30))),1)

    def test_offset_carriageways_reach_the_same_roi_border(self):
        old=frame([[(-110,-17),(110,5)],[(-110,-5),(110,17)]])
        axis=LineString([(-100,-10),(100,10)]);roi=box(-100,-40,100,40)
        proposal=AxisProposal(axis,axis.buffer(13),24.,.9,dict(whole_corridor=True),[0,1])
        pair=carriageway_axes(proposal,old,roi)
        self.assertEqual(len(pair),2)
        for p in pair:
            self.assertLess(Point(p.axis.coords[0]).distance(roi.boundary),1e-6)
            self.assertLess(Point(p.axis.coords[-1]).distance(roi.boundary),1e-6)

    def test_intermittent_double_road_gets_two_ordered_continuous_tracks(self):
        old=frame([[(-100,-6),(100,-6)],[(-100,6),(-30,6)],[(20,6),(100,6)]])
        axis=LineString([(-100,0),(100,0)])
        proposal=AxisProposal(axis,axis.buffer(13),24.,.9,dict(whole_corridor=True),[0,1,2])
        pair=carriageway_axes(proposal,old,box(-100,-30,100,30))
        self.assertEqual(len(pair),2)
        self.assertEqual(len(road_units(pair)),1)
        self.assertTrue(all(p.axis.length>=199 for p in pair))
        self.assertFalse(pair[0].axis.intersects(pair[1].axis))
        self.assertAlmostEqual(pair[0].axis.distance(pair[1].axis),12.)
        self.assertGreater(pair[0].evidence['paired_section_fraction'],.5)

    def test_parallel_bridge_is_not_a_second_carriageway(self):
        old=frame([[(-100,-6),(100,-6)],[(-100,6),(100,6)]],[0,1])
        axis=LineString([(-100,0),(100,0)])
        proposal=AxisProposal(axis,axis.buffer(13),24.,.9,dict(whole_corridor=True),[0])
        self.assertFalse(carriageway_axes(proposal,old,box(-100,-30,100,30)))

    def test_declared_carriageways_are_not_duplicate_axes_but_same_track_is(self):
        f=frame([[(-80,0),(80,0)],[(-80,2),(80,2)]])
        f['carriageway_id']=['road:0','road:1']
        self.assertFalse(candidates(build(f),box(-100,-30,100,30),{},('duplicate',)))
        f['carriageway_id']=['road:0','road:0']
        self.assertTrue(candidates(build(f),box(-100,-30,100,30),{},('duplicate',)))

    def test_connected_graph_can_still_lose_a_long_true_road_arm(self):
        old=frame([[(-90,0),(90,0)],[(-30,0),(-30,25)],[(30,0),(30,25)]])
        truncated=frame([[(-30,0),(30,0)],[(-30,0),(-30,25)],[(30,0),(30,25)]])
        owned=box(-100,-5,100,5)
        self.assertGreater(replaced_extent_distance(old,truncated,owned),55)
        refitted=frame([[(-90,2),(90,2)],[(-30,2),(-30,25)],[(30,2),(30,25)]])
        self.assertLessEqual(replaced_extent_distance(old,refitted,owned),2.01)

    def test_dark_road_relative_material_evidence_and_single_edge_veto(self):
        f=np.linspace(-.4,1.4,37)
        rgb=np.full((5,len(f),3),.30)
        rgb[:,f<0]=[.35,.36,.32];rgb[:,f>1]=[.33,.37,.34]
        self.assertTrue(pavement_descriptor(rgb,f)[1].all())
        rgb[:,f>1]=.30
        self.assertFalse(pavement_descriptor(rgb,f)[1].any())

    def test_complete_corridor_does_not_trim_to_internal_branch_contacts(self):
        old=frame([[(-90,2),(90,2)],[(-30,2),(-30,25)],[(30,2),(30,25)]])
        axis=LineString([(-90,0),(90,0)])
        p=AxisProposal(axis,axis.buffer(5,cap_style='flat'),8.,1.,dict(whole_corridor=True),[])
        result,_,_=replace_corridors(old,[p],box(-90,-40,90,40))
        self.assertAlmostEqual(p.axis.length,180)
        self.assertLess(result.geometry.distance(Point(-30,25)).min(),.001)
        self.assertLess(result.geometry.distance(Point(30,25)).min(),.001)

    def test_fragmented_prior_yields_one_continuous_center_axis(self):
        road=LineString([(-115,0),(115,0)])
        group=[(LineString([(-90,-2),(-15,-2)]),0,('0',False,False)),
               (LineString([(10,2),(90,2)]),1,('0',False,False))]
        found,_=fit_corridor(group,evidence(road),box(-100,-40,100,40))
        self.assertEqual(len(found),1)
        self.assertGreater(found[0].axis.length,195)
        self.assertLess(found[0].axis.centroid.distance(road),1.5)

    def test_remote_roof_like_strip_cannot_replace_the_prior_road(self):
        group=[(LineString([(-100,0),(100,0)]),0,('0',False,False))]
        roof=LineString([(-115,23),(115,23)])
        found,_=fit_corridor(group,evidence(roof),box(-100,-40,100,40))
        self.assertFalse(found)

    def test_real_parallel_roads_are_not_erased_by_a_median_axis(self):
        roads=[LineString([(-115,-12),(115,-12)]),LineString([(-115,12),(115,12)])]
        e=evidence(union_all(roads))
        old=frame([list(r.coords) for r in roads])
        group=[(r,i,('0',False,False)) for i,r in enumerate(roads)]
        found,_=fit_corridor(group,e,box(-100,-45,100,45))
        for proposal in found:
            if abs(proposal.axis.centroid.y)<6:
                self.assertIsNotNone(parallel_ribbon_veto(proposal,old,e))
            else:
                other=roads[0] if proposal.axis.centroid.y>0 else roads[1]
                self.assertGreater(other.difference(proposal.envelope).length,190)


if __name__=='__main__':unittest.main()
