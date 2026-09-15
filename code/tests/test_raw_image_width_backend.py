import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import STRtree
from shapely.geometry import LineString,Point
from engine.width.raw_image_backend import quality,observation_segments
from engine.road_network_products import rebuild_network_width_products
from engine.fast_auto_v2 import _reliable_width_at
from app.fast_settings import width_arguments

class RawWidthBackendTests(unittest.TestCase):
    def test_publication_keeps_required_width_fields_without_internal_audit(self):
        import tempfile
        from pathlib import Path
        from app.result_publisher import ResultPublisher
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'width.shp'
            frame=gpd.GeoDataFrame(dict(final_left=[3.],final_righ=[5.],final_widt=[8.],
                final_conf=[.5],width_sour=['propagated'],outlier_re=[''],quality_gr=['C'],
                width_m=[8.],qa_state=['private']),geometry=[LineString([(0,0),(10,0)])],crs=32650)
            frame.to_file(path)
            published=ResultPublisher(root/'published').publish_period('area','p',
                dict(width_segments=str(path),execution_profile='fast'),save=False)
            public=gpd.read_file(published['width_segments'])
            self.assertNotIn('qa_state',public)
            self.assertIn('width_sour',public)
            full=gpd.read_file(published['width_profiles'])
            self.assertIn('final_left_distance',full)
            self.assertEqual(full.width_source.iloc[0],'propagated')

    def test_quality_and_gui_values(self):
        for source,expected in [('measured','A'),('smoothed','A'),('interpolated','B'),('propagated','C'),('unresolved','C')]:
            self.assertEqual(quality(dict(width_source=source,solver_converged=True,final_confidence=.5)),expected)
        self.assertEqual(quality(dict(width_source='smoothed',solver_converged=True,final_confidence=.1)),'C')
        self.assertEqual(width_arguments('原始影像边界测宽'),['--width-method','raw_image'])
        self.assertEqual(width_arguments('SAM-MoLRA'),['--width-method','sam_molra'])

    def test_propagated_width_is_displayed_but_not_change_evidence(self):
        line=LineString([(0,0),(60,0)])
        rows=[]
        for i,(source,grade) in enumerate([('measured','A'),('interpolated','B'),('propagated','C')]):
            rows.append(dict(road_id='r',s_m=i*25.,center_x=i*25.,center_y=0.,
                final_left_distance=3.,final_right_distance=5.,final_width=8.,final_confidence=.5,
                width_source=source,outlier_reason='',solver_converged=True))
        measured=observation_segments(pd.DataFrame(rows),{'r':line},32650)
        roads=gpd.GeoDataFrame(geometry=[line],crs=32650)
        segments,corridors=rebuild_network_width_products(roads,measured)
        self.assertEqual(measured.quality_grade.tolist(),['A','B','C'])
        self.assertEqual(corridors.width_source.tolist(),['representative'])
        self.assertEqual(segments.quality_grade.tolist(),['B'])
        self.assertAlmostEqual(corridors.area.sum(),480.)
        scene=SimpleNamespace(widths=segments,width_geometries=segments.geometry.values,width_tree=STRtree(segments.geometry.values))
        self.assertEqual(_reliable_width_at(scene,np.array([Point(2,0),Point(25,0),Point(55,0)],object)).tolist(),[True,True,True])
        self.assertEqual(corridors.total_bounds.tolist(),[0.,-5.,60.,3.])

if __name__=='__main__':unittest.main()
