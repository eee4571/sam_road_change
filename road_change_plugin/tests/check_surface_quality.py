"""Final road-surface smoke tests using synthetic saved evidence only."""
import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from engine.road_surface_quality import stable_width, short_noise_indices, junction_footprint, surface_axis
from engine.auto_change_geometry import FinalWidths
from engine.continuous_road_geometry import network_surface
from engine.road_connection_evidence import ConnectionEvidence


def line(x0, x1, y=0):
    return LineString([(x0, y), (x1, y)])


class Probability:
    def __init__(self, value):self.value=value
    def values(self, xy):return np.full(len(xy), self.value)


class SurfaceQualityTests(unittest.TestCase):
    def test_surface_axis_keeps_endpoints_and_low_frequency_curve(self):
        x=np.linspace(0.,100.,201)
        baseline=10*np.sin(x/40)
        a=LineString(np.c_[x,baseline+.4*np.sin(x*2)])
        smooth=surface_axis(a,8.,ConnectionEvidence(a.buffer(4)))
        self.assertEqual(smooth.coords[0],a.coords[0]);self.assertEqual(smooth.coords[-1],a.coords[-1])
        self.assertLess(smooth.length,a.length)
        self.assertLess(smooth.hausdorff_distance(a),.8)
        self.assertGreater(smooth.bounds[3],9.)

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
        qa=[]
        surface=network_surface(profiles,quality=[[1,1],[0,0],[1,1]],audit=qa)
        self.assertLess(surface.symmetric_difference(box(0,-4,100,4)).area,.01)
        self.assertEqual(len(qa),1)

    def test_junction_portals_are_tangent_bounded_and_connected(self):
        axes=[line(0,50),line(0,-50),LineString([(0,0),(0,50)])]
        profiles=[(a,[0.,a.length],[8.,8.]) for a in axes]
        raw=network_surface(profiles)
        surface=network_surface(profiles,quality=[[1.,1.]]*3)
        self.assertTrue(surface.is_valid);self.assertEqual(surface.geom_type,'Polygon')
        self.assertEqual(len(surface.interiors),0)
        self.assertEqual(surface.bounds,raw.bounds)
        delta=surface.difference(raw)
        self.assertGreater(delta.area,0.)
        self.assertLess(delta.area,25.)
        self.assertTrue(box(-9,-9,9,9).covers(delta))

    def test_nearby_parallel_roads_remain_separate(self):
        profiles=[(line(0,100,y),[0.,100.],[4.,4.]) for y in (0,8)]
        result=network_surface(profiles,quality=[[1,1],[1,1]])
        self.assertEqual(len(result.geoms),2)

    def test_unsplit_interior_junction_is_not_moved_by_drawing_axis(self):
        a=LineString([(-50,0),(-10,.4),(0,0),(10,-.4),(50,0)])
        b=LineString([(0,-30),(0,30)])
        profiles=[(p,[0.,p.length],[8.,8.]) for p in (a,b)]
        qa=[]
        result=network_surface(profiles,quality=[[1,1],[1,1]],audit=qa,
            evidence=ConnectionEvidence(a.buffer(4).union(b.buffer(4))))
        self.assertTrue(result.is_valid)
        self.assertEqual(qa[0].get('render_axis_shift_m',0),0)

    def classify(self,axes=None,prob=0.,other=None,grades=None,protected=()):
        axes=axes or [line(0,8)]
        frame=gpd.GeoDataFrame(dict(width_m=[6.]*len(axes),quality_grade=grades or ['C']*len(axes)),geometry=axes,crs=3857)
        return short_noise_indices(axes,[6.]*len(axes),FinalWidths(frame),
            ConnectionEvidence(box(200,200,210,210),Probability(prob)),
            [other] if other is not None else (),protected)

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
