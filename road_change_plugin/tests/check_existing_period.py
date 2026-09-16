"""Explicit bounded existing-data check; never dispatches user_pipeline/models.

Reads one completed period, recovers a 600m neighbourhood, measures at most four
existing chains, and writes only into a unique _work/_processing_check folder.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('project',type=Path)
    parser.add_argument('--period',default='20221020')
    parser.add_argument('--area',default='1')
    args=parser.parse_args()
    project=args.project.resolve()
    period=project/'_work/current/grids'/args.area/'periods'/args.period
    candidates=sorted(period.glob('radiometric/*/latest_result.json'))
    if len(candidates)!=1:raise ValueError('Expected one completed radiometric workspace')
    workspace=candidates[0].parent
    manifest=json.loads(candidates[0].read_text(encoding='utf8'))
    inputs=json.loads((workspace/'input_manifest.json').read_text(encoding='utf8'))
    run=Path(manifest['run_root'])
    working=run/'width_review/fast_products.gpkg'
    if not working.is_file():raise FileNotFoundError(working)
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box,LineString
    from engine.road_network_products import recover_centerline_frame,rebuild_network_width_products
    from engine.road_connection_evidence import probability_sources,molra_sources
    from engine.width.raw_image_backend import measure_region
    center=gpd.read_file(working,layer='centerlines')
    surface=gpd.read_file(working,layer='surfaces')
    projected=center.estimate_utm_crs() if center.crs.is_geographic else center.crs
    metric=center.to_crs(projected)
    longest=metric.geometry.iloc[int(np.argmax(metric.length))]
    middle=longest.interpolate(.5,normalized=True)
    boundary=box(middle.x-300,middle.y-300,middle.x+300,middle.y+300)
    local=gpd.clip(metric,boundary).explode(index_parts=False).reset_index(drop=True)
    local=local[local.geometry.geom_type.eq('LineString') & (local.length>1e-6)].copy()
    local=local.to_crs(center.crs)
    local_surface=gpd.clip(surface.to_crs(projected),boundary).to_crs(center.crs)
    output=project/'_work/_processing_check'/('period-'+args.period+'-'+uuid.uuid4().hex[:8])
    if not output.resolve().is_relative_to(project/'_work'):raise ValueError('Output escaped project')
    output.mkdir(parents=True)
    print(f'LOCAL_CHECK output={output} input_lines={len(local)} extent=600m',flush=True)
    started=time.perf_counter()
    connected,stats,audits=recover_centerline_frame(local,local_surface,
        probability_sources=probability_sources(Path(inputs['images']),run/'width_review'),
        molra_sources=molra_sources(run/'width_review'))
    network_seconds=time.perf_counter()-started
    comparison=output/'axis_comparison.gpkg'
    local.to_file(comparison,layer='before',driver='GPKG')
    connected.to_file(comparison,layer='after',driver='GPKG')
    local_surface.to_file(comparison,layer='surface',driver='GPKG')
    metric_out=connected.to_crs(projected)
    eligible=metric_out[(metric_out.length>=30)&(metric_out.length<=300)]
    sample=eligible.head(4).to_crs(center.crs)
    if sample.empty:raise ValueError('No bounded chains for local width smoke')
    measured=measure_region(sample,Path(inputs['images']),output/'raw_width')
    segments,corridors=rebuild_network_width_products(sample,measured)
    segments.to_file(output/'width_check.gpkg',layer='width_segments',driver='GPKG')
    corridors.to_file(output/'width_check.gpkg',layer='corridors',driver='GPKG')
    width=json.loads((output/'raw_width/summary.json').read_text(encoding='utf8'))
    assert {'width_m','final_width','width_source'}.issubset(measured.columns)
    assert connected.crs==center.crs and measured.crs==center.crs
    # Overlay a corrected chain if present, otherwise show the local network.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    limits=boundary.bounds
    corrected=stats.get('connection_axis_corrected_count',0)
    for axis,title,frame in zip(axes,('Before','After'),(local,connected)):
        local_surface.to_crs(projected).plot(ax=axis,color='#dddddd',edgecolor='none')
        frame.to_crs(projected).plot(ax=axis,color='#205990',linewidth=.8)
        axis.set_xlim(limits[0],limits[2]);axis.set_ylim(limits[1],limits[3])
        axis.set_title(title);axis.set_aspect('equal');axis.ticklabel_format(useOffset=False,style='plain')
        axis.tick_params(labelsize=7)
    fig.suptitle(f'Existing period {args.period}: 600m local check; corrected chains={corrected}')
    fig.tight_layout();fig.savefig(output/'axis_comparison.png',dpi=160);plt.close(fig)
    report=dict(period=args.period,area=args.area,extent_m=600,network_seconds=network_seconds,
        input_lines=len(local),output_lines=len(connected),network=stats,width_chains=len(sample),
        width_seconds=width['seconds'],feature_cache=width['feature_cache'],output=str(output),
        note='One spatial subset only; no model inference or formal result replacement.')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
