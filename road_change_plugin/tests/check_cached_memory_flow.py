"""Two real cached periods: network -> RGB -> formal -> Auto -> temporal.

No extraction model or existing product is written. Two fresh scratch runs
compare serial and automatic scheduling on the same 200 m neighbourhood.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('project',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box
    from engine.resource_policy import ResourceSession
    from engine.road_network_products import recover_centerline_frame,rebuild_network_width_products
    from engine.road_connection_evidence import probability_sources,molra_sources
    from engine.width.raw_image_backend import measure_region
    from engine.width.raw_road_surfaces import build_road_surfaces
    from engine.fast_pipeline import export_fast_products,detect_fast_changes
    from engine.fast_gt_reconciliation import build_fast_temporal_outputs
    from check_memory_peak import peak_rss
    project=args.project.resolve();sources=[]
    for period in ('20221020','20230222'):
        choices=list((project/'_work/current/grids/1/periods'/period).glob('radiometric/*/latest_result.json'))
        if len(choices)!=1:raise ValueError(f'Expected one completed cache for {period}')
        path=choices[0];payload=json.loads(path.read_text(encoding='utf8'))
        image_dir=Path(json.loads((path.parent/'input_manifest.json').read_text(encoding='utf8'))['images'])
        working=Path(payload['run_root'])/'width_review/fast_products.gpkg'
        center=gpd.read_file(working,layer='centerlines')
        if not sources:
            metric=center.estimate_utm_crs() if center.crs.is_geographic else center.crs
            frame=center.to_crs(metric);p=frame.geometry.iloc[int(np.argmax(frame.length))].interpolate(.5,normalized=True)
            area=gpd.GeoDataFrame(geometry=[box(p.x-100,p.y-100,p.x+100,p.y+100)],crs=metric)
        local=gpd.clip(center.to_crs(metric),area).explode(index_parts=False).reset_index(drop=True)
        local=local.loc[local.geom_type.eq('LineString') & (local.length>1e-6)].copy()
        surface=gpd.clip(gpd.read_file(working,layer='surfaces').to_crs(metric),area)
        sources.append((period,payload,image_dir,local,surface))
    scratch=project/'_work/_processing_check'/('memory-flow-'+uuid.uuid4().hex[:8]);scratch.mkdir()
    results={};timing={};prior=os.environ.get('ROAD_PIPELINE_PARALLEL')
    try:
        for mode in ('off','auto'):
            os.environ['ROAD_PIPELINE_PARALLEL']=mode
            started=time.perf_counter();target=scratch/mode;target.mkdir();entries=[]
            with ResourceSession():
                for period,original,images,centers,surfaces in sources:
                    out=target/period;out.mkdir();width_dir=Path(original['width_review'])
                    connected,_,_=recover_centerline_frame(centers,surfaces,
                        probability_sources=probability_sources(images,width_dir),molra_sources=molra_sources(width_dir))
                    measured=measure_region(connected,images,out/'raw_width',surface_evidence=surfaces)
                    segments,corridors=rebuild_network_width_products(connected,measured)
                    surface=build_road_surfaces(measured,image_path=next(images.glob('*.tif')))
                    for key,frame in dict(centerlines=connected,width_segments=segments,corridors=corridors,surfaces=surface).items():
                        frame.to_file(out/'regional_products.gpkg',layer=key,driver='GPKG')
                    product=export_fast_products(width_dir,out,image_dir=images,width_method='raw_image')
                    valid=gpd.clip(gpd.read_file(original['valid_observation']).to_crs(metric),area)
                    valid.to_file(out/'valid.gpkg',driver='GPKG')
                    entries.append({**original,**product,'period':period,'grid':'1','valid_observation':str(out/'valid.gpkg')})
                change=detect_fast_changes(*entries,target/'auto',before_period=entries[0]['period'],
                    after_period=entries[1]['period'],internal_outputs=True)
                manifest=dict(period_results=entries,change_results=[dict(change,grid='1',before_period=entries[0]['period'],
                    after_period=entries[1]['period'])],execution_profile='fast')
                build_fast_temporal_outputs(manifest,target/'final')
                results[mode]=manifest;timing[mode]=time.perf_counter()-started
    finally:
        if prior is None:os.environ.pop('ROAD_PIPELINE_PARALLEL',None)
        else:os.environ['ROAD_PIPELINE_PARALLEL']=prior
    checked=[]
    for left,right in zip(results['off']['period_results'],results['auto']['period_results']):
        for key in ('centerlines','width_segments','surfaces','corridors'):
            a,b=gpd.read_file(left[key]),gpd.read_file(right[key])
            assert a.geometry.to_wkb().tolist()==b.geometry.to_wkb().tolist(),key
            pd.testing.assert_frame_equal(pd.DataFrame(a.drop(columns='geometry')),pd.DataFrame(b.drop(columns='geometry')))
            checked.append(dict(period=left['period'],layer=key,features=len(a),exact=True))
    a,b=[gpd.read_file(results[mode]['change_results'][0]['road_changes']) for mode in ('off','auto')]
    assert a.geometry.to_wkb().tolist()==b.geometry.to_wkb().tolist()
    pd.testing.assert_frame_equal(pd.DataFrame(a.drop(columns='geometry')),pd.DataFrame(b.drop(columns='geometry')))
    report=dict(extent_m=200,periods=[s[0] for s in sources],models_executed=False,seconds=timing,
        peak_process_rss=peak_rss(),layers=checked,changes=len(a),changes_exact=True,scratch=str(scratch))
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
