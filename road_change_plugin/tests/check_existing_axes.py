"""Read-only existing-product QA; writes comparisons only to an explicit directory."""
import argparse
import json
from pathlib import Path
import sys
import time
from collections import Counter
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import geopandas as gpd
from shapely import union_all
from engine.road_axis_quality import axis_quality,repair_network_axes
from engine.road_connection_evidence import ConnectionEvidence,RoadProbability
from engine.auto_change_geometry import FinalWidths,corridor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('products',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--continuous',action='store_true',help='Build the formal whole-network surface using saved widths')
    args=parser.parse_args();root=args.products;out=args.output
    if out.resolve()==root.resolve() or root.resolve() in out.resolve().parents:
        raise ValueError('Comparison output must be separate from original products')
    roads=gpd.read_file(root/'road_centerlines.shp')
    roads=roads.to_crs(roads.estimate_utm_crs() if roads.crs.is_geographic else roads.crs)
    faults=[i for i,r in roads.iterrows() if axis_quality(r.geometry,r.width_m)['abnormal']]
    print('roads',len(roads),'individual_faults',len(faults),flush=True)
    surfaces=gpd.read_file(root/'road_surfaces.shp').to_crs(roads.crs)
    evidence=ConnectionEvidence(union_all(surfaces.geometry),RoadProbability([(root/'road_probability.tif',None)],roads.crs))
    started=time.perf_counter()
    out.mkdir(parents=True,exist_ok=True)
    try:fixed,audit=repair_network_axes(list(roads.geometry),list(roads.width_m),evidence)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(getattr(exc,'diagnostic',{}),indent=2),encoding='utf8')
        raise
    roads.to_file(out/'before.gpkg')
    roads.geometry=fixed;roads.to_file(out/'after.gpkg')
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print('corrected_features',len(audit),'seconds',round(time.perf_counter()-started,2),flush=True)
    before=gpd.read_file(out/'before.gpkg')
    widths=FinalWidths(gpd.read_file(root/'road_width_segments.gpkg').to_crs(roads.crs))
    summaries=[];before_surfaces=[];after_surfaces=[]
    for row in audit:
        i=row['feature'];old=before.geometry.iloc[i];new=roads.geometry.iloc[i]
        ss,ww,_=widths.profile(old,float(roads.width_m.iloc[i]))
        # Only this comparison bypasses the new precondition for old geometry.
        with patch('engine.road_axis_quality.require_safe_axis'):
            original=corridor(old,ss,ww)
        mapping=row['station_map']
        mapped=np.interp(ss,mapping['before'],mapping['after'])
        fixed_surface=corridor(new,mapped,ww)
        def holes(g):
            return sum(len(p.interiors) for p in ([g] if g.geom_type=='Polygon' else g.geoms))
        before_surfaces.append(dict(feature=i,geometry=original));after_surfaces.append(dict(feature=i,geometry=fixed_surface))
        def large_holes(g):
            from shapely.geometry import Polygon
            return sum(Polygon(h).area>=1. for p in ([g] if g.geom_type=='Polygon' else g.geoms) for h in p.interiors)
        summaries.append(dict(feature=i,holes_before=holes(original),holes_after=holes(fixed_surface),
            large_holes_before=large_holes(original),large_holes_after=large_holes(fixed_surface),valid=fixed_surface.is_valid))
    gpd.GeoDataFrame(before_surfaces,crs=roads.crs).to_file(out/'surface_before.gpkg')
    gpd.GeoDataFrame(after_surfaces,crs=roads.crs).to_file(out/'surface_after.gpkg')
    summary=dict(roads=len(roads),corrected=len(audit),
        methods=dict(Counter(r['method'] for a in audit for p in a['parts'] for r in p.get('repairs',[]))),
        features=summaries)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('holes',sum(r['holes_before'] for r in summaries),'->',sum(r['holes_after'] for r in summaries),
          'invalid',sum(not r['valid'] for r in summaries),flush=True)
    if args.continuous:
        from engine.continuous_road_geometry import network_surface
        by_feature={r['feature']:r for r in audit};profiles=[]
        for i,row in roads.iterrows():
            ss,ww,_=widths.profile(before.geometry.iloc[i],float(row.width_m))
            if i in by_feature:
                mapping=by_feature[i]['station_map']
                ss=np.interp(ss,mapping['before'],mapping['after'])
            profiles.append((row.geometry,ss,ww))
        joined=network_surface(profiles)
        pieces=[joined] if joined.geom_type=='Polygon' else list(joined.geoms)
        gpd.GeoDataFrame(geometry=pieces,crs=roads.crs).to_file(out/'full_continuous_surface.gpkg',layer='surfaces')
        print('continuous_parts',len(pieces),'valid',joined.is_valid,flush=True)


if __name__=='__main__':main()
