"""Lossless multipart temporal products, identity, event and review regressions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, GeometryCollection, Point, shape
import temporal_road_analysis as temporal
from engine.disk_records import record_scope

CRS = 'EPSG:32650'


class MultipartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_geometry(self, geometry, *, nodes=False, crs=CRS):
        linear = temporal._road_geometry(geometry)
        periods = []
        for period, width in [('1', 4.), ('2', 8.)]:
            directory = self.root / period
            directory.mkdir(exist_ok=True)
            center = directory / 'roads.gpkg'
            state = directory / 'state.gpkg'
            gpd.GeoDataFrame([dict(track_id='track-original', width_m=width, geometry=geometry)], crs=crs).to_file(center)
            gpd.GeoDataFrame([dict(track_id='track-original', status='present', width_m=width, geometry=geometry)], crs=crs).to_file(state, layer='road_state')
            periods.append(dict(period=period, centerlines=str(center), road_state=str(state)))
        result = temporal.build_temporal_grid('test', periods, [], self.root / 'temporal')
        life = gpd.read_file(result['life_shp'])
        observations = gpd.read_file(result['observations_shp'])
        events = gpd.read_file(result['events_shp'])
        self.assertEqual(list(life.road_id), ['track-original'])
        self.assertEqual(list(observations.road_id), ['track-original'] * 2)
        self.assertEqual(list(observations.status), ['present', 'present'])
        self.assertEqual(list(events.event_typ), ['widened'])
        for frame in (life, observations, events):
            for geom in frame.geometry:
                self.assertTrue(geom.equals(linear))
                self.assertAlmostEqual(geom.length, linear.length, places=7)
        self.assertEqual(bool(life.from_node.notna().iloc[0]), nodes)
        self.assertEqual(bool(life.to_node.notna().iloc[0]), nodes)
        reviewed = temporal.build_temporal_grid('test', periods, [], self.root / 'review', reuse_observations=result)
        again = gpd.read_file(reviewed['life_shp'])
        self.assertEqual(list(again.road_id), list(life.road_id))
        self.assertTrue(again.geometry.iloc[0].equals(linear))
        self.assertEqual(Path(result['observations_shp']).read_bytes(), Path(reviewed['observations_shp']).read_bytes())

    def test_disconnected_different_lengths(self):
        self.run_geometry(MultiLineString([[(0, 0), (100, 0)], [(200, 0), (220, 0)]]))

    def test_branch(self):
        self.run_geometry(MultiLineString([[(0, 0), (10, 0)], [(10, 0), (20, 10)], [(10, 0), (20, -10)]]))

    def test_connected_parts_have_nodes_without_changing_geometry(self):
        self.run_geometry(MultiLineString([[(0, 0), (10, 0)], [(10, 0), (20, 10)]]), nodes=True)

    def test_nested_collection_keeps_all_lines(self):
        self.run_geometry(GeometryCollection([Point(50, 50), LineString([(0, 0), (100, 0)]),
            GeometryCollection([MultiLineString([[(200, 0), (220, 0)], [(220, 0), (230, 10)]])])]))

    def test_closed_road_has_no_unique_start(self):
        self.run_geometry(LineString([(0, 0), (10, 0), (10, 10), (0, 0)]))

    def test_reviewed_collection_event_evidence_keeps_track_and_all_parts(self):
        geometry = MultiLineString([[(0, 0), (100, 0)], [(200, 0), (230, 0)]])
        self.run_geometry(geometry)
        paths = [dict(period=p, centerlines=str(self.root / p / 'roads.gpkg'),
                      road_state=str(self.root / p / 'state.gpkg')) for p in ('1', '2')]
        evidence = self.root / 'evidence.gpkg'
        gpd.GeoDataFrame([dict(track_id='track-original', change_typ='widened',
            geometry=GeometryCollection([part.buffer(4) for part in geometry.geoms] + [Point(50, 0)]))],
            crs=CRS).to_file(evidence, layer='changes')
        # The formal SHP is the presence contract; the audit retains collection geometry.
        changes = self.root / 'changes.shp'
        gpd.GeoDataFrame([dict(track_id='track-original', change_typ='widened', geometry=geometry.buffer(4))],
                         crs=CRS).to_file(changes)
        pair = dict(before_period='1', after_period='2', layers={'changes': str(changes)},
                    event_geometry_audit=str(evidence), manual_review={'accepted': True})
        result = temporal.build_temporal_grid('test', paths, [pair], self.root / 'accepted')
        fragments = gpd.read_file(result['event_parts_shp'])
        self.assertEqual(len(fragments), 2)
        self.assertEqual(set(fragments.road_id), {'track-original'})
        self.assertEqual(result['event_count'], 1)
        rejected = self.root / 'rejected.shp'
        gpd.GeoDataFrame({'change_typ': []}, geometry=[], crs=CRS).to_file(rejected, geometry_type='Polygon')
        pair.pop('event_geometry_audit')
        pair['layers']['changes'] = str(rejected)
        updated = temporal.build_temporal_grid('test', paths, [pair], self.root / 'rejected', reuse_observations=result)
        self.assertEqual(updated['event_count'], 0)
        life = gpd.read_file(updated['life_shp'])
        self.assertEqual(life.road_id.iloc[0], 'track-original')
        self.assertTrue(life.geometry.iloc[0].equals(geometry))
        self.assertEqual(life.event_n.iloc[0], 0)

    def test_real_reconciled_multiline(self):
        data = json.loads((ROOT / 'tests/fixtures/temporal_real_multiline.json').read_text())
        self.run_geometry(shape(data['geometry']), crs=data['crs'])

    @record_scope
    def test_untracked_multiline_matching_is_finite_and_identity_survives_rebuild(self):
        geom = MultiLineString([[(0, 0), (100, 0)], [(0, 200), (100, 200)]])
        paths = []
        for p in ('1', '2'):
            path = self.root / f'{p}.gpkg'
            gpd.GeoDataFrame([dict(width_m=4., geometry=geom)], crs=CRS).to_file(path)
            paths.append(dict(period=p, centerlines=str(path)))
        result = temporal.build_temporal_grid('test', paths, [], self.root / 'out')
        obs = gpd.read_file(result['observations_shp'])
        self.assertTrue(np.isfinite(obs.match_sc).all())
        self.assertEqual(list(obs.status), ['present', 'present'])
        self.assertEqual(result['road_count'], 1)
        before = gpd.read_file(result['life_shp'])
        temporal.build_temporal_grid('test', paths, [], self.root / 'out')
        after = gpd.read_file(result['life_shp'])
        self.assertEqual(list(before.road_id), list(after.road_id))
        self.assertTrue(after.geometry.iloc[0].equals(geom))


if __name__ == '__main__':
    unittest.main()
