"""Synthetic axis/state smoke tests; no model inference or real project."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from shapely.geometry import LineString,box
import geopandas as gpd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT.parent/"code/tests")]
import engine
engine.__path__=[str(ROOT/"code/engine")]
import user_pipeline
from engine.road_axis_quality import axis_quality,repair_axis,repair_network_axes,AxisQualityError
from engine.road_connection_evidence import ConnectionEvidence
from engine.auto_change_geometry import corridor
from engine.road_track_lifecycle import resolve_lifecycle
from engine.fast_gt_reconciliation import reconcile_periods,_frame,_write_final_period
from test_fast_gt_reconciliation import FastGTReconciliationTests
from test_continuous_road_geometry import ContinuousRoadGeometryTests
from engine.continuous_road_geometry import network_surface
from temporal_road_analysis import build_temporal_grid


def staircase():
    xy=[(0,0)]
    for x in range(0,100,10):xy.extend([(x+4,0),(x+4,6),(x+8,6),(x+8,0)])
    xy.append((100,0))
    return LineString(xy)


def event(kind,b,a,first,last,source="e"):
    return dict(kind=kind,width_before=b,width_after=a,before=str(first),after=str(last),source=source,origin="AUTO")


class AxisTests(unittest.TestCase):
    def test_fault_over_two_metres_repaired_before_variable_offset(self):
        axis=staircase()
        fixed,qa=repair_axis(axis,10.,ConnectionEvidence(box(-10,-10,110,15)))
        self.assertTrue(qa["corrected"]);self.assertGreater(qa["maximum_shift_m"],2.)
        self.assertEqual(fixed.coords[0],axis.coords[0]);self.assertEqual(fixed.coords[-1],axis.coords[-1])
        surface=corridor(fixed,np.array([0.,fixed.length]),np.array([8.,12.]))
        self.assertTrue(surface.is_valid);self.assertEqual(len(surface.interiors),0)
        with self.assertRaises(AxisQualityError):corridor(axis,[0.,axis.length],[8.,12.])

    def test_degree_two_fragments_repair_as_one_chain(self):
        axis=staircase();xy=list(axis.coords)
        axes=[LineString(xy[i:i+2]) for i in range(len(xy)-1)]
        fixed,qa=repair_network_axes(axes,[10.]*len(axes),ConnectionEvidence(box(-10,-10,110,15)))
        self.assertTrue(qa)
        self.assertEqual(fixed[0].coords[0],xy[0]);self.assertEqual(fixed[-1].coords[-1],xy[-1])
        for a,b in zip(fixed,fixed[1:]):self.assertEqual(a.coords[-1],b.coords[0])
        merged=LineString([p for i,a in enumerate(fixed) for p in list(a.coords)[int(i>0):]])
        self.assertTrue(merged.is_simple)
        self.assertFalse(axis_quality(merged,10.)["abnormal"])

    def test_real_continuous_curves_are_unchanged(self):
        t=np.linspace(0,2.8,100)
        for axis in (LineString(np.c_[40*np.cos(t),40*np.sin(t)]),
                     LineString(np.c_[np.linspace(0,150,100),15*np.sin(np.linspace(0,6,100))])):
            fixed,qa=repair_axis(axis,8.,None)
            self.assertEqual(fixed.wkb,axis.wkb);self.assertFalse(qa["corrected"])

    def test_distant_endpoint_jogs_are_not_one_repeated_fold(self):
        axis=LineString([(0,0),(1,0),(1,.5),(2,.5),(60,.5),(60,0),(61,0),(62,-3)])
        self.assertFalse(axis_quality(axis,22.)['abnormal'])

    def test_local_repair_leaves_distant_normal_vertices_exact(self):
        axis=LineString([(-100,0),(-80,1),(-50,0)]+list(staircase().coords)+[(150,0),(180,1),(200,0)])
        fixed,qa=repair_axis(axis,10.,ConnectionEvidence(axis.buffer(12)))
        self.assertEqual(list(fixed.coords)[:3],list(axis.coords)[:3])
        self.assertEqual(list(fixed.coords)[-3:],list(axis.coords)[-3:])
        self.assertTrue(qa['corrected'])

    def test_self_crossing_axis_rebuilt_before_offset(self):
        axis=LineString([(0,0),(10,10),(0,10),(10,0)])
        self.assertTrue(axis_quality(axis,8.)['abnormal'])
        fixed,qa=repair_axis(axis,8.,ConnectionEvidence(box(-5,-5,15,15)))
        self.assertTrue(fixed.is_simple)
        self.assertEqual(fixed.coords[0],axis.coords[0]);self.assertEqual(fixed.coords[-1],axis.coords[-1])

    def test_unsupported_fault_does_not_enter_surface(self):
        fixed,qa=repair_axis(staircase(),8.,ConnectionEvidence(box(200,200,210,210)))
        self.assertFalse(axis_quality(fixed,8.)['abnormal'])
        self.assertEqual(qa['repairs'][0]['quality_state'],'direction_inferred')

    def test_junction_endpoint_stays_connected(self):
        axis=staircase();branch=LineString([(100,0),(100,-50)])
        third=LineString([(100,0),(150,0)])
        fixed,qa=repair_network_axes([axis,branch,third],[10.,8.,8.],ConnectionEvidence(box(-10,-60,160,15)))
        self.assertEqual(fixed[0].coords[-1],fixed[1].coords[0])
        self.assertEqual(fixed[1].wkb,branch.wkb)
        self.assertEqual(fixed[2].wkb,third.wkb)


class LifecycleTests(unittest.TestCase):
    def statuses(self,events):
        return resolve_lifecycle(["1","2","3","4","5"],events)[0]

    def test_added_inherits_all_later_periods(self):
        states=self.statuses([event("added",0,8,1,2)])
        self.assertEqual([s["present"] for s in states.values()],[False,True,True,True,True])
        self.assertEqual(states["5"]["state_source"],"inherited_event")

    def test_removed_stays_absent_until_added(self):
        states=self.statuses([event("removed",8,0,2,3),event("added",0,10,4,5,"new")])
        self.assertEqual([s["present"] for s in states.values()],[True,True,False,False,True])

    def test_width_only_updates_width(self):
        states=self.statuses([event("widened",6,10,1,2),event("narrowed",10,8,3,4,"n")])
        self.assertTrue(all(s["present"] for s in states.values()))
        self.assertEqual([s["width"] for s in states.values()],[6,10,10,8,8])

    def test_width_does_not_resurrect_removed_road(self):
        states=self.statuses([event("removed",8,0,1,2),event("widened",8,12,3,4,"w")])
        self.assertFalse(states["5"]["present"])
        self.assertEqual(states["4"]["event_conflict"],"width_event_on_absent_track")

    def test_corrected_event_repropagates_entire_future(self):
        before=self.statuses([event("added",0,8,1,2)])
        corrected=dict(event("removed",8,0,1,2),origin="GT_ASSISTED")
        after=self.statuses([corrected])
        self.assertTrue(before["5"]["present"]);self.assertFalse(after["5"]["present"])
        self.assertEqual(after["5"]["event_origin"],"GT_ASSISTED")


class FinalProductTests(unittest.TestCase):
    def test_station_roundoff_does_not_reverse_entire_surface_axis(self):
        axis=LineString([(0,0),(60,0)])
        surface=network_surface([(axis,[-1e-10,60.+1e-10],[6.,10.])])
        self.assertTrue(surface.is_valid)
        self.assertEqual(surface.bounds[0],0.)
        self.assertEqual(surface.bounds[2],60.)
        self.assertEqual(len(surface.interiors),0)
        self.assertAlmostEqual(surface.area,480.,delta=1.)

    def test_junction_loop_is_not_concatenated_into_self_crossing_trunk(self):
        trunk=LineString([(0,0),(50,0)])
        loop=LineString([(50,0),(70,0),(70,20),(50,20),(50,0)])
        profiles=[(a,[0.,a.length],[4.,4.]) for a in (trunk,loop)]
        surface=network_surface(profiles)
        expected=trunk.buffer(2,cap_style='flat').union(loop.buffer(2,cap_style='flat'))
        self.assertTrue(surface.is_valid)
        self.assertEqual(surface.geom_type,'Polygon')
        self.assertEqual(len(surface.interiors),1)
        self.assertLess(surface.symmetric_difference(expected).area,.01)

    def test_missing_extraction_materializes_inherited_track_only(self):
        fixture=FastGTReconciliationTests()
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);main=LineString([(0,0),(100,0)]);other=LineString([(0,40),(100,40)])
            periods=[fixture.period(root,str(i),[(other,6)]+([(main,8)] if i==2 else [])) for i in range(1,6)]
            row=dict(change_id="add",change_typ="added",change_src="AUTO",width_bef=0.,width_aft=8.,
                     axis_wkt=main.wkt,match_axis_wkt=main.wkt,geometry=main.buffer(4))
            pairs=[dict(frame=_frame([row],"EPSG:32650"),before_period="1",after_period="2")]
            outputs=reconcile_periods(periods,pairs,root/"out")
            for i,output in enumerate(outputs,1):
                states=gpd.read_file(output["road_state"])
                self.assertEqual(states.iloc[0].status,"absent" if i==1 else "present")
                if i>=3:self.assertEqual(states.iloc[0].observation_conflict,"extraction_missing")
                roads=gpd.read_file(output["centerlines"])
                self.assertAlmostEqual(roads.loc[roads.track_id.isna()].length.sum(),100.)
                self.assertEqual(len(roads.loc[roads.track_id.notna()]),int(i>=2))
            change_dir=root/"change";change_dir.mkdir()
            pairs[0]["frame"].drop(columns=["axis_wkt","match_axis_wkt"]).to_file(change_dir/"road_changes.shp")
            result=build_temporal_grid("region",outputs,[dict(output=str(change_dir),before_period="1",after_period="2")],root/"temporal")
            events=gpd.read_file(result["events_shp"])
            self.assertEqual(events.loc[events.road_id.str.startswith("RC")].event_typ.tolist(),["added"])

    def test_final_centerline_and_surface_both_use_repaired_axis(self):
        fixture=FastGTReconciliationTests()
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);axis=staircase();original=fixture.period(root,"input",[(axis,10.)])
            base=gpd.read_file(original["centerlines"])
            # Independent observed road ribbon, not a generated self-intersecting polygon.
            gpd.GeoDataFrame(geometry=[box(-2,-4,102,10)],crs=base.crs).to_file(original["surfaces"])
            target=root/"final";target.mkdir()
            rows=[dict(base.iloc[0],track_id="",_original_row=0)]
            output=_write_final_period(original,base,{},rows,[],target,base.crs,base.crs)
            final=gpd.read_file(output["centerlines"]).geometry.iloc[0]
            self.assertGreater(final.hausdorff_distance(axis),2.)
            self.assertFalse(axis_quality(final,10.)["abnormal"])
            surfaces=gpd.read_file(output["surfaces"])
            self.assertTrue(surfaces.is_valid.all())
            self.assertTrue(all(len(g.interiors)==0 for g in surfaces.geometry))
            self.assertTrue(Path(output["axis_quality_audit"]).is_file())


if __name__=="__main__":unittest.main()
