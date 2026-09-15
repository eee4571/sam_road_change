import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString,box
from engine.road_connection_evidence import ConnectionEvidence,ConnectorPropagation
from engine.road_network_products import recover_centerline_frame
from engine.road_track_corridors import restore_track_corridors,infer_track_corridors
from engine.road_geometry import _RegionalRoadSeed


class ConstantProbability:
    def __init__(self,value):self.value=value
    def values(self,xy):return np.full(len(xy),self.value)


class ConnectionEvidenceTests(unittest.TestCase):
    def test_propagation_is_local_not_whole_source_road(self):
        p=ConnectorPropagation();p.add([(0,0),(10,0)],1)
        self.assertEqual(p.depth([(10,0),(20,0)]),2)
        p.add([(10,0),(20,0)],2)
        self.assertEqual(p.depth([(20,0),(30,0)]),3)
        self.assertEqual(p.depth([(100,0),(110,0)]),1)

    def test_local_aligned_partial_support_not_long_gap_relaxation(self):
        e=ConnectionEvidence(box(-1,-2,10,2).union(box(25,-2,40,2)),ConstantProbability(0))
        row={};self.assertTrue(e.evaluate([(0,0),(37,0)],row))
        self.assertEqual(row['decision_reason'],'short_gap_continuity_priority')
        self.assertFalse(e.evaluate([(0,0),(100,0)],{}))

    def test_long_aligned_gap_with_no_image_support_rejected(self):
        row={};evidence=ConnectionEvidence(None,ConstantProbability(.02))
        self.assertFalse(evidence.evaluate([(0,0),(220,0)],row))
        self.assertEqual(row['direction_cosine'],1.)
        self.assertEqual(row['decision_reason'],'excessive_gap_length')

    def test_supported_short_and_long_road_stay_connected(self):
        evidence=ConnectionEvidence(box(-5,-5,300,5),ConstantProbability(.8))
        for length in (8,35,120):
            row={};self.assertTrue(evidence.evaluate([(0,0),(length,0)],row))
            self.assertGreater(row['joint_support'],.95)

    def test_long_blank_middle_cannot_be_hidden_by_mean_support(self):
        surface=box(-5,-5,80,5).union(box(110,-5,220,5))
        row={};self.assertFalse(ConnectionEvidence(surface,ConstantProbability(0.)).evaluate([(0,0),(200,0)],row))
        self.assertGreater(row['unsupported_run_m'],18)

    def test_local_molra_omission_does_not_veto_existing_surface(self):
        row={};e=ConnectionEvidence(box(-5,-5,220,5),ConstantProbability(.01),ConstantProbability(0.))
        self.assertTrue(e.evaluate([(0,0),(120,0)],row))
        self.assertEqual(row['raw_surface_support'],1.)
        self.assertEqual(row['surface_support'],1.)

    def test_short_gap_restored_without_image_and_remote_road_retained(self):
        frame=gpd.GeoDataFrame({'width_m':[6.,6.,6.]},geometry=[LineString([(0,0),(100,0)]),
            LineString([(150,0),(250,0)]),LineString([(1000,0),(1030,0)])],crs=32650)
        result,stats,audit=recover_centerline_frame(frame)
        self.assertAlmostEqual(result.length.sum(),280.)
        self.assertEqual(stats['connection_isolated_removed_count'],0)
        self.assertTrue(audit['added_connections'].decision_reason.eq('short_gap_continuity_priority').all())

    def test_short_gap_image_absence_never_vetoes_and_overlong_always_rejected(self):
        for p in (0.,float('nan')):
            evidence=ConnectionEvidence(None,ConstantProbability(p))
            for length in (5,40,80):
                row={};self.assertTrue(evidence.evaluate([(0,0),(length,0)],row))
        evidence=ConnectionEvidence(box(-1,-3,500,3),ConstantProbability(1.))
        self.assertFalse(evidence.evaluate([(0,0),(151,0)],{}))

    def test_corridor_bridge_callback_cannot_be_bypassed(self):
        roads=[_RegionalRoadSeed(np.array([[a,y],[b,y]],float),6.,(i,))
               for i,(a,b,y) in enumerate([(0,150,0),(200,400,0),(0,150,20),(200,400,20)])]
        models=infer_track_corridors(roads)
        self.assertTrue(models)
        calls=[]
        def reject(curve,sources):calls.append(curve);return False
        result,_,bridges,_=restore_track_corridors(roads,models,bridge_accept=reject)
        self.assertTrue(calls);self.assertFalse(bridges)
        for road in result:
            self.assertLess(LineString(road.points).intersection(box(151,-50,199,50)).length,1e-6)


if __name__=='__main__':unittest.main()
