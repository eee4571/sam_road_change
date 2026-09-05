from pathlib import Path
import sys
import unittest

import numpy as np
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.road_geometry import _RegionalRoadSeed
from engine.road_network_connection import connect_clean_road_seeds


class CleanRoadConnectionTests(unittest.TestCase):
    def roads(self, specs, scale=1.0):
        return [_RegionalRoadSeed(np.asarray(p, dtype=float) / scale, 10.0, (i,))
                for i, p in enumerate(specs)]

    def test_main_component_uses_network_length_and_preserves_objects(self):
        from engine.road_network_connection import retain_main_component
        roads = self.roads([[[0,0],[10,0]],[[10,0],[20,0]],[[10,0],[10,10]],
                            [[100,0],[125,0]],[[200,0],[201,0]],
                            [[201,0],[202,0]],[[202,0],[203,0]],[[203,0],[204,0]]])
        kept, removed = retain_main_component(roads)
        self.assertEqual([r.source_ids for r in kept],[(0,),(1,),(2,)])
        self.assertTrue(all(a is b for a,b in zip(kept,roads[:3])))
        self.assertEqual(len(removed),5)
        self.assertEqual(retain_main_component([]),([],[]))

    def test_main_component_equal_lengths_ignore_input_order(self):
        from engine.road_network_connection import retain_main_component
        roads = self.roads([[[0,0],[10,0]],[[100,0],[110,0]]])
        for order in (roads,roads[::-1]):
            kept, _ = retain_main_component(order)
            self.assertEqual(kept[0].source_ids,(0,))

    def test_main_component_filter_audits_removal_in_metric_units(self):
        roads = self.roads([[[0,0],[100,0]],[[1000,0],[1020,0]]],scale=2)
        output, stats, audit = connect_clean_road_seeds(roads,unit_size_m=2,
                                                     max_gap_m=10,keep_main_component=True)
        self.assertEqual(len(output),1)
        self.assertEqual(stats['connection_components_before_filter'],2)
        self.assertEqual(stats['connection_components_after'],1)
        self.assertEqual(stats['connection_isolated_removed_length_m'],20)
        removed = [r for r in audit if r['status']=='isolated_removed']
        self.assertEqual(len(removed),1)
        self.assertEqual(removed[0]['geometry'].length,10)
        self.assertEqual(removed[0]['distance_m'],20)

    def test_complete_curved_roads_keep_lane_identity_when_model_drifts(self):
        from engine.road_track_corridors import TrackCorridor, restore_track_corridors
        from shapely.ops import unary_union
        x = np.linspace(0,1600,321)
        y = 55*np.exp(-((x-800)/220)**2)
        roads = self.roads([np.column_stack([x,y]),np.column_stack([x,y+60])])
        model = TrackCorridor(np.zeros(2),np.asarray([1.,0.]),np.asarray([0.,60.]),0,1600,1000)
        output, _, bridges, _ = restore_track_corridors(roads,[model])
        network = unary_union([LineString(r.points) for r in output])
        for road in roads:
            self.assertLess(LineString(road.points).difference(network.buffer(.001)).length,.01)
        self.assertEqual(len(bridges),0)
        self.assertTrue(all(len(r.source_ids)==1 for r in output))

    def test_redundant_trace_removal_preserves_cross_branch_and_long_third_road(self):
        from engine.road_track_corridors import TrackCorridor, _remove_parallel_residuals
        model = TrackCorridor(np.zeros(2),np.asarray([1.,0.]),np.asarray([0.,40.]),0,1000,1000)
        restored = self.roads([[[0,0],[1000,0]],[[0,40],[1000,40]]])
        outside = self.roads([[[100,5],[150,5]],[[180,40],[180,100]],
                              [[250,5],[500,5]],[[600,5],[650,5]],[[650,5],[650,-80]]])
        kept,removed = _remove_parallel_residuals(outside,restored,[model])
        self.assertEqual(len(removed),1)
        self.assertEqual({r.source_ids for r in kept},{(1,),(2,),(3,),(4,)})
        self.assertAlmostEqual(removed[0].length,50)

    def test_junction_attachment_cannot_close_a_thin_return_loop(self):
        roads = self.roads([[[0,-200],[0,200]],[[0,0],[55,0],[55,6],[15,6]]])
        _, stats, audit = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_added_count'],0)
        self.assertIn('short_redundant_cycle',{r['status'] for r in audit})

    def test_long_chain_has_no_round_limit_and_preserves_all_segments(self):
        roads = self.roads([[[i * 30, 0], [i * 30 + 20, 0]] for i in range(24)])
        output, stats, _ = connect_clean_road_seeds(roads)
        self.assertEqual(len(output), 1)
        self.assertEqual(stats['connection_added_count'], 23)
        line = LineString(output[0].points)
        for road in roads:
            self.assertTrue(line.covers(LineString(road.points)))
        again, second_stats, _ = connect_clean_road_seeds(output)
        np.testing.assert_array_equal(again[0].points, output[0].points)
        self.assertEqual(second_stats['connection_added_count'], 0)

    def test_long_missing_surface_and_metric_scale(self):
        specs = [[[0, 0], [60, 0]], [[160, 0], [230, 0]]]
        for scale in (0.3, 1, 2):
            roads = self.roads(specs, scale)
            output, stats, _ = connect_clean_road_seeds(roads, unit_size_m=scale)
            self.assertEqual(len(output), 1)
            self.assertAlmostEqual(stats['connection_added_length_m'], 100)

    def test_two_broken_carriageways_do_not_swap(self):
        specs = [[[0, 0], [40, 0]], [[100, 0], [150, 0]],
                 [[0, 12], [70, 12]], [[115, 12], [150, 12]]]
        output, stats, _ = connect_clean_road_seeds(self.roads(specs), box(-10, -20, 160, 30))
        self.assertEqual({r.source_ids for r in output}, {(0, 1), (2, 3)})
        self.assertEqual(stats['connection_added_count'], 2)

    def test_one_complete_carriageway_is_unchanged(self):
        specs = [[[0, 0], [200, 0]], [[0, 12], [45, 12]], [[105, 12], [170, 12]]]
        output, _, _ = connect_clean_road_seeds(self.roads(specs))
        complete = next(r for r in output if r.source_ids == (0,))
        np.testing.assert_array_equal(complete.points, specs[0])
        self.assertEqual(len(output), 2)

    def test_crossing_divided_roads_restore_two_through_tracks_after_old_merges(self):
        from shapely.ops import unary_union
        specs = [
            [[-500,-20],[-70,-20]], [[70,-20],[500,-20]],
            [[-500,20],[-90,20]], [[90,20],[500,20]],
            [[-15,-500],[-15,-150],[0,-100]],
            [[15,-500],[15,-150],[0,-100]],
            [[0,-100],[0,100]],
            [[0,100],[-15,150],[-15,500]],
            [[0,100],[15,150],[15,500]],
        ]
        angle = np.deg2rad(27)
        rotation = np.asarray([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        for transform in (np.eye(2),rotation):
            roads = self.roads([np.asarray(p)@transform.T for p in specs])
            output, stats, audit = connect_clean_road_seeds(roads,max_gap_m=350)
            network = unary_union([LineString(r.points) for r in output])
            from shapely.geometry import Point
            for point in [(-15,0),(15,0),(0,-20),(0,20)]:
                self.assertLess(network.distance(Point(np.asarray(point)@transform.T)),.01)
            self.assertGreater(network.distance(Point(np.asarray([0,80])@transform.T)),10)
            self.assertGreaterEqual(stats['connection_corridor_count'],2)
            self.assertGreater(stats['connection_corridor_replaced_length_m'],100)
            self.assertEqual(stats['connection_components_after'],1)

    def test_short_oblique_fragment_cannot_send_a_corridor_to_the_other_lane(self):
        from shapely.ops import unary_union
        from shapely.geometry import Point
        roads = self.roads([
            [[0,0],[400,0]], [[450,0],[1000,0]],
            [[0,40],[390,40]], [[470,40],[1000,40]],
            [[427,42],[434,31]],
        ])
        output, stats, _ = connect_clean_road_seeds(roads)
        network = unary_union([LineString(r.points) for r in output])
        self.assertLess(network.distance(Point(430,0)),.01)
        self.assertLess(network.distance(Point(430,40)),.01)
        self.assertGreater(network.distance(Point(430,20)),15)

    def test_correct_long_parallel_curves_and_cross_street_are_preserved(self):
        from shapely.ops import unary_union
        x = np.linspace(0,1000,201)
        y = 8*np.sin(x/250)
        specs = [np.column_stack([x,y]),np.column_stack([x,y+30]),[[500,-80],[500,100]]]
        roads = self.roads(specs)
        output, stats, _ = connect_clean_road_seeds(roads)
        area = unary_union([LineString(r.points) for r in output]).buffer(.001)
        self.assertLess(sum(LineString(r.points).difference(area).length for r in roads),.01)
        self.assertEqual(stats['connection_corridor_bridge_count'],0)

    def test_paired_track_reconstruction_is_stable_for_large_metric_offsets(self):
        from shapely.ops import unary_union
        specs = [np.asarray(p)+[260000,2613000] for p in [
            [[0,0],[400,0]],[[500,0],[1000,0]],
            [[0,30],[350,30]],[[550,30],[1000,30]]]]
        output, _, _ = connect_clean_road_seeds(self.roads(specs))
        again, stats, _ = connect_clean_road_seeds(output)
        a = unary_union([LineString(r.points) for r in output])
        b = unary_union([LineString(r.points) for r in again])
        self.assertLess(a.hausdorff_distance(b),.001)
        self.assertEqual(stats['connection_added_count'],0)
        self.assertEqual(stats['connection_corridor_bridge_count'],0)

    def test_missing_carriageway_tip_follows_observed_companion_extent(self):
        from shapely.ops import unary_union
        from shapely.geometry import Point
        roads = self.roads([[[0,0],[1000,0]],[[180,30],[1000,30]]])
        output, stats, _ = connect_clean_road_seeds(roads)
        network = unary_union([LineString(r.points) for r in output])
        self.assertLess(network.distance(Point(20,30)),.01)
        self.assertLess(network.distance(Point(0,30)),.01)
        self.assertGreater(stats['connection_corridor_bridge_count'],0)

    def test_companion_extrapolation_respects_maximum_gap(self):
        from shapely.ops import unary_union
        from shapely.geometry import Point
        roads = self.roads([[[0,0],[1500,0]],[[500,30],[1500,30]]])
        output, _, _ = connect_clean_road_seeds(roads,max_gap_m=200)
        network = unary_union([LineString(r.points) for r in output])
        self.assertGreater(network.distance(Point(0,30)),20)

    def test_planar_crossing_creates_a_real_shared_junction(self):
        roads = self.roads([[[0, 0], [40, 0]], [[90, 0], [140, 0]], [[65, -40], [65, 40]]])
        _, stats, audit = connect_clean_road_seeds(roads, box(-10, -50, 150, 50))
        self.assertEqual(stats['connection_added_count'], 1)
        self.assertEqual(stats['connection_components_after'], 1)
        self.assertEqual(stats['connection_junctions_after'], 1)

    def test_gap_cannot_bypass_a_nearby_intervening_fragment(self):
        roads = self.roads([[[0, 0], [40, 0]], [[110, 0], [160, 0]], [[60, 3.5], [90, 3.5]]])
        _, stats, audit = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_added_count'], 2)
        self.assertIn('bypasses_existing_fragment', {row['status'] for row in audit})
        self.assertLess(stats['connection_max_length_m'], 35)

    def test_ambiguous_fork_is_not_chosen_by_input_order(self):
        specs = [[[0, 0], [40, 0]], [[80, -1], [120, -1]], [[80, 1], [120, 1]]]
        for order in (specs, specs[::-1]):
            _, stats, audit = connect_clean_road_seeds(self.roads(order))
            self.assertEqual(stats['connection_added_count'], 0)
            self.assertIn('ambiguous_continuation', {row['status'] for row in audit})

    def test_submetre_gap_is_connected(self):
        _, stats, _ = connect_clean_road_seeds(self.roads([[[0, 0], [20, 0]], [[20.2, 0], [40, 0]]]))
        self.assertEqual(stats['connection_added_count'], 1)

    def test_true_junction_vertex_survives_chain_assembly(self):
        roads = self.roads([[[0, 0], [20, 0]], [[20, 0], [40, 0]], [[20, 0], [20, 30]]])
        output, _, _ = connect_clean_road_seeds(roads)
        self.assertEqual(len(output), 3)

    def test_ordinary_street_block_can_close(self):
        roads = self.roads([[[0, 0], [40, 0]], [[70, 0], [110, 0]],
                            [[0, 0], [0, 60], [110, 60], [110, 0]]])
        _, stats, _ = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_added_count'], 1)
        self.assertEqual(stats['connection_dangling_after'], 0)

    def test_four_block_corners_form_a_closed_network(self):
        roads = self.roads([
            [[0, 20], [0, 0], [20, 0]], [[30, 0], [50, 0], [50, 20]],
            [[50, 30], [50, 50], [30, 50]], [[20, 50], [0, 50], [0, 30]],
        ])
        output, stats, audit = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_added_count'], 4)
        self.assertEqual(stats['connection_dangling_after'], 0)
        self.assertTrue(any(LineString(r.points).is_ring for r in output))

    def test_two_new_straight_crossing_roads_share_one_junction(self):
        roads = self.roads([
            [[-70, 0], [-20, 0]], [[20, 0], [70, 0]],
            [[0, -70], [0, -20]], [[0, 20], [0, 70]],
        ])
        _, stats, audit = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_added_count'], 2)
        self.assertEqual(stats['connection_components_after'], 1)
        self.assertEqual(stats['connection_junctions_after'], 1)

    def test_short_alternate_path_does_not_get_a_redundant_shortcut(self):
        roads = self.roads([[[0,0],[20,0]], [[30,0],[50,0]],
                            [[0,0],[0,1],[50,1],[50,0]]])
        _,stats,_ = connect_clean_road_seeds(roads)
        # The short bypass geometry is preserved; a new parallel copy is rejected.
        self.assertEqual(stats['connection_added_count'],0)

    def test_branch_attaches_to_middle_of_trunk_and_splits_it(self):
        roads = self.roads([[[0,0],[100,0]], [[50,40],[50,12]]])
        output,stats,_ = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_attachment_count'],1)
        self.assertEqual(stats['connection_components_after'],1)
        self.assertEqual(stats['connection_junctions_after'],1)
        self.assertEqual(len(output),3)

    def test_large_projected_coordinates_produce_exact_junction_topology(self):
        roads = self.roads([[[0,0],[100,33]], [[50,60],[50,30]]])
        offset = np.array([259187.123456,2611540.987654])
        roads = [_RegionalRoadSeed(r.points+offset,r.width_m,r.source_ids) for r in roads]
        output,stats,_ = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_components_after'],1)
        _,again,_ = connect_clean_road_seeds(output)
        self.assertEqual(again['connection_added_count'],0)

    def test_short_offset_gap_can_curve_without_sharp_joints(self):
        roads = self.roads([[[0,0],[30,0]],[[36,2],[65,2]]])
        output,stats,_ = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_continuation_count'],1)
        directions = np.diff(output[0].points,axis=0)
        directions /= np.linalg.norm(directions,axis=1)[:,None]
        self.assertGreater(np.min(np.sum(directions[:-1]*directions[1:],axis=1)),np.cos(np.deg2rad(30)))

    def test_parallel_lane_shift_without_correspondence_is_rejected(self):
        roads = self.roads([[[0,0],[100,0]],[[0,12],[40,12]],[[70,0],[140,0]]])
        _,stats,_ = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_added_count'],0)

    def test_gap_must_not_cross_back_over_its_member_body(self):
        roads = self.roads([[[0,0],[40,0]],[[100,0],[140,0],[140,-30],[70,-30],[70,10]]])
        _,_,audit = connect_clean_road_seeds(roads)
        self.assertFalse(any(row['status']=='accepted' and row['kind']=='continuation' for row in audit))

    def test_network_result_is_invariant_to_input_order(self):
        from shapely.ops import unary_union
        specs = [[[0,0],[40,0]],[[70,0],[100,0]],[[30,30],[30,10]],
                 [[0,12],[40,12]],[[70,12],[100,12]]]
        first,_,_ = connect_clean_road_seeds(self.roads(specs))
        second,_,_ = connect_clean_road_seeds(self.roads(specs[::-1]))
        a = unary_union([LineString(r.points) for r in first])
        b = unary_union([LineString(r.points) for r in second])
        self.assertLess(a.hausdorff_distance(b),1e-5)

    def test_curved_connection_preserves_vertices_and_tangents(self):
        roads = self.roads([[[-60, 0], [-30, 0], [0, 0]], [[40, 10], [60, 20], [80, 30]]])
        output, stats, _ = connect_clean_road_seeds(roads)
        self.assertEqual(stats['connection_added_count'], 1)
        for original in roads:
            for point in original.points:
                self.assertTrue(np.any(np.all(output[0].points == point, axis=1)))
        segments = np.diff(output[0].points, axis=0)
        directions = segments / np.linalg.norm(segments, axis=1)[:, None]
        self.assertGreater(np.min(np.sum(directions[:-1] * directions[1:], axis=1)), np.cos(np.deg2rad(30)))

    def test_closed_ring_is_unchanged(self):
        roads = self.roads([[[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]]])
        output, stats, _ = connect_clean_road_seeds(roads)
        np.testing.assert_array_equal(output[0].points, roads[0].points)
        self.assertEqual(stats['connection_added_count'], 0)

    def test_empty_input(self):
        output, stats, audit = connect_clean_road_seeds([])
        self.assertEqual((output, audit), ([], []))
        self.assertEqual(stats['connection_added_count'], 0)

    def test_submillimetre_observation_does_not_fail_topology_lookup(self):
        roads = self.roads([[[0, 0], [0.0001, 0]]])
        output, _, _ = connect_clean_road_seeds(roads)
        np.testing.assert_array_equal(output[0].points, roads[0].points)



if __name__ == '__main__':
    unittest.main()
