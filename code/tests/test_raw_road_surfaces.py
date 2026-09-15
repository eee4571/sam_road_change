import unittest
import geopandas as gpd
from shapely.geometry import Polygon, box
from engine.width.raw_road_surfaces import build_road_surfaces


class RoadSurfaceTests(unittest.TestCase):
    def frame(self, geoms, parents):
        return gpd.GeoDataFrame({'parent_id': parents, 'final_width': range(len(geoms))}, geometry=geoms, crs=32650)

    def test_continuous_boundaries_and_fine_records_unchanged(self):
        frame = self.frame([Polygon([(0,2),(10,3),(10,-4),(0,-2)]),
                            Polygon([(10,3),(20,4),(20,-5),(10,-4)])], ['r','r'])
        before = frame.copy(deep=True)
        result = build_road_surfaces(frame, pixel_size=1)
        self.assertEqual(len(result), 1)
        self.assertLess(result.geometry.iloc[0].symmetric_difference(frame.union_all()).area, 1e-10)
        self.assertTrue(frame.equals(before))
        self.assertEqual(list(frame.geometry.to_wkb()), list(before.geometry.to_wkb()))

    def test_junction_union_and_separated_roads(self):
        frame = self.frame([box(0,-2,20,2),box(8,0,12,12),box(0,13,20,17)], ['a','b','c'])
        result = build_road_surfaces(frame, pixel_size=1)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.is_valid.all())
        self.assertAlmostEqual(result.area.sum(),frame.union_all().area)

    def test_large_holes_preserved_and_tiny_hole_removed(self):
        polygon = Polygon(box(0,0,30,30).exterior, [box(5,5,20,20).exterior, box(22,22,22.1,22.1).exterior])
        result = build_road_surfaces(self.frame([polygon], ['r']), pixel_size=1)
        self.assertEqual(len(result.iloc[0].geometry.interiors), 1)
        self.assertAlmostEqual(result.area.sum(),675)

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


if __name__ == '__main__':
    unittest.main()
