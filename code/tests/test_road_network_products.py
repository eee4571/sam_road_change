from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.road_network_products import (
    NETWORK_REPORT, network_products_current, recover_centerline_frame,
    rebuild_network_width_products, write_network_report,
)


class NetworkProductTests(unittest.TestCase):
    def frame(self):
        return gpd.GeoDataFrame({'width_m':[10.,10.,6.], 'tile':['a','b','c']},
                                geometry=[LineString([(0,0),(100,0)]),
                                          LineString([(130,0),(230,0)]),
                                          LineString([(1000,1000),(1020,1000)])],crs=32650)

    def test_region_connects_across_tiles_and_audits_island(self):
        result,stats,audits = recover_centerline_frame(self.frame())
        self.assertEqual(stats['connection_components_after'],1)
        self.assertEqual(stats['connection_isolated_removed_count'],1)
        self.assertAlmostEqual(result.length.sum(),230)
        self.assertEqual(len(audits['connection_input']),3)
        self.assertEqual(len(audits['removed_isolated_roads']),1)
        self.assertEqual(result.iloc[0].src_ids,'0,1')

    def test_manual_authoritative_and_empty_inputs_preserved(self):
        source = self.frame()
        result,stats,_ = recover_centerline_frame(source,authoritative=True)
        self.assertEqual(list(result.geometry.to_wkb()),list(source.geometry.to_wkb()))
        self.assertEqual(stats['policy'],'manual_authoritative')
        result,stats,_ = recover_centerline_frame(source.iloc[:0])
        self.assertTrue(result.empty)
        self.assertEqual(stats['connection_components_after'],0)

    def test_geographic_width_products_use_metres(self):
        source=self.frame().iloc[:1].copy()
        source.geometry=[LineString([(260000,2610000),(260100,2610000)])]
        geographic=source.to_crs(4490)
        result,_,_=recover_centerline_frame(geographic)
        segments,corridors=rebuild_network_width_products(result)
        self.assertEqual(segments.crs,geographic.crs)
        self.assertGreater(len(segments),5)
        self.assertAlmostEqual(segments.length_m.sum(),100,places=3)
        self.assertAlmostEqual(corridors.area_m2.sum(),1000,places=2)

    def test_new_gap_width_is_not_marked_as_a_direct_measurement(self):
        source = self.frame()
        source['quality_grade'] = 'A'
        final,_,audits = recover_centerline_frame(source)
        segments,_ = rebuild_network_width_products(final,source,
                          connection_input=audits['connection_input'])
        gap = segments[segments.geometry.map(lambda g: 103<g.centroid.x<127)]
        self.assertFalse(gap.empty)
        self.assertTrue(gap.quality_grade.eq('C').all())
        self.assertTrue(gap.qa_reason.eq('connection_width_inherited').all())
        self.assertTrue(segments[segments.geometry.map(lambda g:g.centroid.x<90)].quality_grade.eq('A').all())

    def test_fast_export_rebuilds_width_layers_and_failed_retry_has_no_marker(self):
        from engine.fast_pipeline import export_fast_products
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); width=root/'width'; width.mkdir(); output=root/'products'
            source=self.frame()
            surfaces=gpd.GeoDataFrame(geometry=[box(-10,-10,240,10)],crs=source.crs)
            for i,(name,frame) in enumerate({'centerlines':source,'surfaces':surfaces,
                                            'width_segments':source,'corridors':surfaces}.items()):
                frame.to_file(width/'fast_products.gpkg',layer=name,driver='GPKG',mode='w' if i==0 else 'a')
            with patch('engine.fast_pipeline._write_fast_period_previews',return_value={'fusion':'','width':''}):
                export_fast_products(width,output)
                self.assertTrue(network_products_current(output))
                final=gpd.read_file(output/'roads.gpkg',layer='centerlines')
                measured=gpd.read_file(output/'roads.gpkg',layer='width_segments')
                self.assertLess(unary_union(final.geometry).symmetric_difference(unary_union(measured.geometry)).length,.001)
                self.assertAlmostEqual(final.length.sum(),230)
                first=list(final.geometry.to_wkb())
                export_fast_products(width,output)
                self.assertEqual(first,list(gpd.read_file(output/'roads.gpkg',layer='centerlines').geometry.to_wkb()))
            with patch('engine.fast_pipeline._write_fast_period_previews',side_effect=RuntimeError('preview failure')):
                with self.assertRaises(RuntimeError):
                    export_fast_products(width,output)
            self.assertFalse(network_products_current(output))

    def test_readiness_rejects_old_export_without_blocking_historical_change_input(self):
        import user_pipeline
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            paths={key:root/name for key,name in [('centerlines','road_centerlines.shp'),
                                                  ('surfaces','road_surfaces.shp'),('gpkg','roads.gpkg')]}
            for path in paths.values(): path.touch()
            result=root/'result.json'; result.write_text(json.dumps({k:str(v) for k,v in paths.items()}))
            entry={'result':str(result)}
            self.assertTrue(user_pipeline._period_result_ready(entry))
            self.assertFalse(user_pipeline._period_result_ready(entry,require_current_network=True))
            frame,stats,audits=recover_centerline_frame(self.frame())
            write_network_report(root,stats,audits)
            self.assertTrue(user_pipeline._period_result_ready(entry,require_current_network=True))

    def test_full_export_calls_recovery_before_width_generation(self):
        import rasterio
        import cv2
        from rasterio.transform import from_origin
        from engine.width import production_workflow as production
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); final=root/'final'; final.mkdir(); images=root/'images'; images.mkdir()
            image=images/'tile.tif'
            with rasterio.open(image,'w',driver='GTiff',width=80,height=80,count=3,
                               dtype='uint8',crs=32650,transform=from_origin(260000,2610080,1,1)) as dataset:
                dataset.write(np.zeros((3,80,80),dtype='uint8'))
            production.save_graph(final/'graph.p',[(40,5),(40,70)],[(0,1)])
            (final/'edges.csv').write_text('final_edge_id,optimized_width_px,optimized_quality_grade\n0,10,A\n')
            cv2.imwrite(str(final/'surface.png'),np.full((80,80),255,dtype='uint8'))
            (final/'tile_optimized_summary.json').write_text(json.dumps({'image':str(image),'pixel_size':1,
                'outputs':{'optimized_graph':'graph.p','optimized_edges':'edges.csv','optimized_road_surface':'surface.png'}}))
            output=root/'products'/'roads.gpkg'
            with patch.object(production,'recover_centerline_frame',wraps=recover_centerline_frame) as recover, \
                 patch.object(production,'_write_final_visualization',return_value={}), \
                 patch.object(production,'_write_final_width_visualization',return_value={}):
                production.export_final_products(final,images,output=output)
            self.assertEqual(recover.call_count,1)
            self.assertFalse(recover.call_args.kwargs['authoritative'])
            self.assertTrue(network_products_current(output.parent))
            lines=gpd.read_file(output,layer='final_centerlines')
            segments=gpd.read_file(output,layer='final_width_segments')
            self.assertLess(unary_union(lines.geometry).symmetric_difference(unary_union(segments.geometry)).length,.001)


if __name__=='__main__':
    unittest.main()
