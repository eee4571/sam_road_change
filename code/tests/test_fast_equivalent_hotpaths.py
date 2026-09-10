"""Scalar/optimized equivalence and bounded-resource regressions; no inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, box
from shapely.prepared import prep
from engine.auto_scene_cache import SceneCache
import engine.road_network_connection as network
from engine.road_geometry import _RegionalRoadSeed
from engine.width.road_pair_matcher import build_width_segments, _direction, _quality, _number, WIDTH_FIELDS, _robust
from engine.fast_auto_change import RoadScene, analyze_scenes
import test_fast_auto_change as fixtures


class EquivalentHotpathTests(unittest.TestCase):
    setUp=fixtures.FastFinalAutoTests.setUp
    scene=fixtures.FastFinalAutoTests.scene
    road=staticmethod(fixtures.FastFinalAutoTests.road)

    def test_scalar_geometry_evidence_matches_batch_candidates_and_order(self):
        before=self.scene([self.road(70,8),self.road(160)])
        after=self.scene([self.road(70,14),self.road(205)],valid=box(20,10,250,240))
        original=RoadScene.evidence
        def scalar(scene,*args,**kw):
            kw.pop('geometry_evidence',None)
            return original(scene,*args,**kw)
        with patch.object(RoadScene,'evidence',scalar):
            expected=analyze_scenes(before,after,candidate_driven=False)
        actual=analyze_scenes(before,after,candidate_driven=False)
        for a,b in zip(expected[:3],actual[:3]):
            self.assertEqual(len(a),len(b))
            for old,new in zip(a,b):
                self.assertEqual(old,new)

    def test_network_scalar_support_and_fixed_conflict_cache(self):
        specs=[[(0,0),(45,0)],[(65,0),(120,0)],[(140,0),(180,0)],[(80,35),(80,60)]]
        roads=[_RegionalRoadSeed(np.asarray(x,dtype=float),8.,(i,)) for i,x in enumerate(specs)]
        surface=box(-1,-5,190,5)
        def scalar(points,support):
            if support is None:return 0.
            line=LineString(points)
            return float(np.mean([support.covers(line.interpolate(t,normalized=True)) for t in np.linspace(0,1,max(5,min(101,int(line.length/3)+1)))]))
        for points in specs:
            self.assertEqual(network._support(points,prep(surface)),scalar(points,prep(surface)))
        with patch.object(network,'_support',scalar):
            old=network.connect_clean_road_seeds(roads,surface)
        select=network._select;conflict=network._nearby_conflict
        def counted(*args,**kwargs):
            with patch.object(network,'_nearby_conflict',wraps=conflict) as calls:
                result=select(*args,**kwargs)
                ids=[id(call.args[0]) for call in calls.call_args_list]
                self.assertEqual(len(ids),len(set(ids)))
                return result
        with patch.object(network,'_select',counted):
            new=network.connect_clean_road_seeds(roads,surface)
        self.assertEqual(old[1:],new[1:])
        for a,b in zip(old[0],new[0]):
            np.testing.assert_array_equal(a.points,b.points)
            self.assertEqual((a.width_m,a.source_ids),(b.width_m,b.source_ids))

    def test_width_statistics_match_scalar_order_with_rejected_observations(self):
        axis=LineString([(0,0),(25,2),(60,0)])
        center=gpd.GeoDataFrame({'width_m':[7.], 'width_quality':['A']},geometry=[axis],crs=3857)
        observations=gpd.GeoDataFrame({'width_m':[8.,400.,10.,99.,6.],
            'width_quality':['A','C','B','A','A'],'quality_flags':['','','','boundary','']},
            geometry=[axis,axis,axis,axis,LineString([(30,-20),(30,20)])],crs=3857)
        actual=build_width_segments(center,observations)
        for r in actual.itertuples():
            values=[];overlaps=0.
            for _,candidate in observations.iterrows():
                line=candidate.geometry
                if abs(float(np.dot(_direction(r.geometry),_direction(line))))<.8:continue
                overlap=r.geometry.intersection(line.buffer(2.)).length
                if overlap<=0:continue
                grade=_quality(candidate);value=_number(candidate,WIDTH_FIELDS)
                if value>0 and grade!='C' and not any(t in str(candidate.quality_flags) for t in ('junction','border','boundary','asym','outlier','outside')):
                    values.extend([value]*max(1,int(round(overlap))))
                    overlaps+=min(overlap,r.geometry.length)
            median,std,count=_robust(values)
            self.assertEqual(r.width_m,median);self.assertEqual(r.width_std,std)
            self.assertEqual(r.valid_ratio,min(1.,overlaps/r.geometry.length))
        self.assertEqual(actual.segment_id.tolist(),[f'S{i+1:08d}' for i in range(len(actual))])

    def test_scene_cache_invalidates_sidecars_and_crs_and_closes(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);cache=SceneCache();created=[]
            def factory():
                scene=Mock();created.append(scene);return scene
            def payload(n):
                result={}
                for key in ('centerlines','surfaces','width_segments','valid_observation','road_probability'):
                    path=root/f'{n}_{key}{".tif" if key=="road_probability" else ".shp"}'
                    path.write_bytes(b'original');result[key]=str(path)
                return result
            a,b,c=[payload(n) for n in range(3)]
            first=cache.get(a,'3857',factory)
            self.assertIs(cache.get(a,'3857',factory),first)
            cache.get(b,'3857',factory)
            Path(a['centerlines']).with_suffix('.dbf').write_bytes(b'edited attributes')
            second=cache.get(a,'3857',factory)
            self.assertIsNot(first,second);first.probability.close.assert_called_once()
            third=cache.get(a,'32650',factory)
            second.probability.close.assert_called_once()
            cache.get(c,'32650',factory)
            self.assertLessEqual(len(cache.items),2)
            cache.close()
            for item in created:item.probability.close.assert_called_once()

if __name__=='__main__':unittest.main()
