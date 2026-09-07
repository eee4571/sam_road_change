"""Rerender cached final geometry without repeating classification or inference."""
import json
import sys
from pathlib import Path
import geopandas as gpd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.auto_change_geometry import FinalWidths
from engine.continuous_road_geometry import network_surface
from engine.fast_gt_reconciliation import _publish_assisted_changes
from app.result_publisher import ResultPublisher
from run_fast_final_chain import summarize


def main():
    root=Path(sys.argv[1]);path=root/'pipeline_result.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for period in manifest['final_period_results']:
        axes=gpd.read_file(period['centerlines']);crs=axes.crs
        metric=axes.estimate_utm_crs() if crs.is_geographic else crs
        axes=axes.to_crs(metric);widths=FinalWidths(gpd.read_file(period['width_segments']).to_crs(metric))
        profiles=[]
        for row in axes.itertuples():
            ss,ww,_=widths.profile(row.geometry,float(getattr(row,'width_m',6.)))
            profiles.append((row.geometry,ss,ww))
        surface=network_surface(profiles)
        parts=[surface] if surface.geom_type=='Polygon' else list(surface.geoms)
        frame=gpd.GeoDataFrame(geometry=parts,crs=metric).to_crs(crs)
        for key in ('surfaces','corridors'):frame.to_file(period[key],encoding='UTF-8')
    for result in manifest['change_results']:
        frame=gpd.read_file(result['event_geometry_audit'],layer='changes')
        result.update(_publish_assisted_changes(frame,Path(result['road_changes']).parent,result['before_period'],result['after_period']))
    ResultPublisher(root/'成果输出').publish_manifest(manifest)
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    provenance=json.loads(Path('project/test_area/auto_geometry_20260906_release/input_provenance.json').read_text(encoding='utf-8'))
    summarize(root,manifest,provenance)


if __name__=='__main__':main()
