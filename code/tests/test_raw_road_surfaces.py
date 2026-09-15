import unittest
import geopandas as gpd
from shapely.geometry import Polygon, box, LineString, Point
from engine.width.raw_road_surfaces import build_road_surfaces


class RoadSurfaceTests(unittest.TestCase):
    def frame(self, geoms, parents):
        return gpd.GeoDataFrame({'parent_id': parents, 'final_width': range(len(geoms))}, geometry=geoms, crs=32650)

    def profiles(self, axes, left=None, right=None):
        import numpy as np
        from shapely.ops import substring
        rows=[]
        for k,axis in enumerate(axes):
            edges=np.linspace(0,axis.length,max(2,int(axis.length/2)+1))
            for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
                x=(a+b)/2
                rows.append(dict(parent_id=str(k),segment_id=f'{k}_{i:05d}',
                    final_left_distance=left(x) if left else 3.,
                    final_right_distance=right(x) if right else 3.,geometry=substring(axis,a,b)))
        return gpd.GeoDataFrame(rows,geometry='geometry',crs=32650)

    def test_node_to_node_constant_width_and_fine_records_unchanged(self):
        import numpy as np
        from engine.width.raw_road_surfaces import _profile_runs,_display_boundaries
        axis=LineString([(0,0),(200,0)])
        frame=self.profiles([axis],left=lambda x:3 + (2 if 70<x<130 else 0) + .5*np.sin(x*2))
        before=frame.copy(deep=True)
        result=build_road_surfaces(frame,pixel_size=.5)
        self.assertEqual(len(result),1)
        polygon=result.geometry.iloc[0]
        # A node-to-node chain uses its representative display width.
        for x,expected in [(30,3),(100,3),(170,3)]:
            section=polygon.intersection(LineString([(x,0),(x,10)]))
            self.assertAlmostEqual(section.length,expected,delta=.3)
        self.assertTrue(frame.equals(before))
        self.assertEqual(list(frame.geometry.to_wkb()),list(before.geometry.to_wkb()))
        self.assertAlmostEqual(polygon.bounds[0],0.)
        self.assertAlmostEqual(polygon.bounds[2],200.)

    def test_junction_union_and_separated_roads(self):
        frame=self.profiles([LineString([(-30,0),(0,0)]),LineString([(0,0),(30,0)]),
                             LineString([(0,0),(0,30)]),LineString([(-30,-7),(30,-7)])])
        result=build_road_surfaces(frame,pixel_size=.5)
        self.assertEqual(len(result),2)
        self.assertTrue(result.is_valid.all())
        self.assertTrue(result.union_all().covers(Point(0,0)))
        self.assertFalse(result.union_all().covers(Point(20,-3.5)))

    def test_ring_preserves_large_hole(self):
        import numpy as np
        t=np.linspace(0,2*np.pi,180)
        axis=LineString(np.column_stack([50*np.cos(t),50*np.sin(t)]))
        coords=list(axis.coords);coords[-1]=coords[0];axis=LineString(coords)
        result=build_road_surfaces(self.profiles([axis]),pixel_size=.5)
        self.assertEqual(len(result),1)
        self.assertEqual(len(result.iloc[0].geometry.interiors),1)
        self.assertGreater(Polygon(result.iloc[0].geometry.interiors[0]).area,6500)

    def test_missing_width_does_not_bridge_gap(self):
        import numpy as np
        frame=self.profiles([LineString([(0,0),(100,0)])],left=lambda x:np.nan if 40<x<60 else 3)
        result=build_road_surfaces(frame,pixel_size=.5)
        self.assertEqual(len(result),2)
        self.assertFalse(result.union_all().covers(Point(50,0)))

    def test_raw_publication_excludes_internal_width_facets(self):
        import tempfile
        from pathlib import Path
        from app.result_publisher import ResultPublisher
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'input.shp'
            self.frame([box(0,0,10,3)],['r']).to_file(path)
            publisher=ResultPublisher(root/'published')
            old=publisher.publish_period('area','p',dict(surfaces=str(path),corridors=str(path),width_segments=str(path)),save=False)
            result=publisher.publish_period('area','p',dict(width_method='raw_image',surfaces=str(path),corridors=str(path),width_segments=str(path)),save=False)
            self.assertEqual(set(result), {'surfaces'})
            self.assertFalse(Path(old['corridors']).exists())
            self.assertFalse(Path(old['width_segments']).exists())
            self.assertTrue(path.exists())

    def test_presentation_surface_is_not_analysis_input_or_cache_identity(self):
        import tempfile
        from pathlib import Path
        from engine.fast_pipeline import _read_fast_change_layer
        from engine.auto_scene_cache import scene_key
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            evidence=root/'evidence.gpkg';display=root/'display.gpkg'
            self.frame([box(0,0,10,3)],['r']).to_file(evidence)
            self.frame([box(0,0,20,4)],['r']).to_file(display)
            payload={key:str(evidence) for key in ('centerlines','width_segments','valid_observation','road_probability')}
            payload.update(surfaces=str(display),analysis_surfaces=str(evidence))
            key=scene_key(payload,32650)
            self.assertEqual(_read_fast_change_layer(payload,'surfaces').geometry.iloc[0].area,30.)
            display.touch()
            self.assertEqual(key,scene_key(payload,32650))
            evidence.touch()
            self.assertNotEqual(key,scene_key(payload,32650))


if __name__ == '__main__':
    unittest.main()
