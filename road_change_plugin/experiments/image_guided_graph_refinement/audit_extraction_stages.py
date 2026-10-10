"""Read-only stage provenance and RGB overlays; never executes extraction/repair."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib
import uuid

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely import union_all
from shapely.geometry import box, mapping

from bootstrap import ROOT
from data import discover, read_json, read_roi
from evidence import ImageGrid
from report import background, draw, write_json
from rgb_reextract import raw_graph

REGIONS = [
    dict(id='industrial', bbox_m=[260080,2614200,260480,2614540]),
    dict(id='main_axis', bbox_m=[261590,2614140,261735,2614460]),
    dict(id='junction', bbox_m=[260000,2613440,260210,2613650]),
    dict(id='canal', bbox_m=[261740,2614140,261930,2614260]),
]


def snapshot(path):
    path=Path(path)
    return dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def figure(path, frames, evidence, roi, titles, colors=None):
    columns=2 if len(frames)==4 else len(frames)
    rows=int(np.ceil(len(frames)/columns))
    width,height=roi.bounds[2]-roi.bounds[0],roi.bounds[3]-roi.bounds[1]
    fig,axes=plt.subplots(rows,columns,figsize=(7*columns,7*rows*height/width),squeeze=False)
    for ax,frame,title in zip(axes.ravel(),frames,titles):
        background(ax,evidence)
        if frame is not None:
            for g in frame.geometry: draw(ax,g,color='#ffff20',linewidth=1.)
        ax.set(xlim=(roi.bounds[0],roi.bounds[2]),ylim=(roi.bounds[1],roi.bounds[3]),aspect='equal',title=title)
        ax.set_axis_off()
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)


def industrial_details(folder, inputs, crs):
    """Same cached geometries at useful visual-review scales; no drawn target axes."""
    names=['raw_toponet','tile_processed','regional_connected','formal']
    titles=['A Original TopoNet cache','B Tile cleanup / vectorization',
            'C Regional connection recovery','D Formal canonical graph']
    for name,bounds in [
        ('main_road', [260130,2614360,260480,2614540]),
        ('entrance', [260160,2614350,260380,2614480]),
    ]:
        roi=box(*bounds)
        frames=[read_roi(folder/'stages.gpkg',crs,roi,layer) for layer in names]
        evidence=SimpleNamespace(image=ImageGrid(inputs['images'],crs,roi,.4))
        figure(folder/(name+'_stages.png'),frames,evidence,roi,titles)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    args=parser.parse_args()
    saved=read_json(args.inputs)
    inputs=discover(saved['project'],saved['period'],saved['area'],record=saved['selected_period_record'])
    run=Path(inputs['record']['run_root']);products=run/'products'
    stage_files=[run/'width_review/fast_products.gpkg',products/'regional_products.gpkg',products/'roads.gpkg']
    audit_path=products/'road_network_audit.gpkg'
    output=ROOT/'outputs'/('stage_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True);print(output,flush=True)
    used=set(stage_files+[audit_path]+inputs['images'])
    used.update(products/n for n in ['regional_cache.json','fast_export_cache.json','axis_quality_audit.json','surface_quality_audit.json'])
    signatures={str(p):snapshot(p) for p in used}
    rows=[]
    for row in REGIONS:
        roi=box(*row['bbox_m']);folder=output/row['id'];folder.mkdir()
        raw,files=raw_graph(inputs,saved['crs'],roi)
        for p in files:
            if str(p) not in signatures: signatures[str(p)]=snapshot(p)
        frames=[raw]+[read_roi(p,saved['crs'],roi,'centerlines') for p in stage_files]
        names=['raw_toponet','tile_processed','regional_connected','formal']
        stats=[]
        previous=None
        for name,frame in zip(names,frames):
            clipped=frame.copy();clipped.geometry=clipped.geometry.intersection(roi)
            clipped=clipped[~clipped.is_empty].explode(index_parts=False)
            clipped=clipped[clipped.geom_type=='LineString']
            clipped.to_file(folder/'stages.gpkg',layer=name,driver='GPKG')
            geometry=union_all(clipped.geometry)
            stat=dict(stage=name,feature_count=len(clipped),union_length_m=geometry.length)
            if previous is not None:
                stat.update(added_beyond_2m_m=geometry.difference(previous.buffer(2)).length,
                            absent_beyond_2m_m=previous.difference(geometry.buffer(2)).length)
            stats.append(stat);previous=geometry
        connections=gpd.read_file(audit_path,layer='added_connections',engine='pyogrio',fid_as_index=True).to_crs(saved['crs'])
        connections=connections[connections.intersects(roi)].copy()
        connections['audit_fid']=connections.index
        connections.to_file(folder/'stages.gpkg',layer='added_connections',driver='GPKG')
        evidence=SimpleNamespace(image=ImageGrid(inputs['images'],saved['crs'],roi,.5))
        figure(folder/'four_stages.png',frames,evidence,roi,
               ['A Original TopoNet cache','B Tile cleanup / vectorization','C Regional connection recovery','D Formal canonical graph'])
        figure(folder/'rgb_raw_formal.png',[None,raw,frames[-1]],evidence,roi,
               ['Raw RGB','Original TopoNet cache','Formal centerlines'])
        fig,ax=plt.subplots(figsize=(10,10*(roi.bounds[3]-roi.bounds[1])/(roi.bounds[2]-roi.bounds[0])))
        background(ax,evidence)
        for g in frames[1].geometry:draw(ax,g,color='#ffff20',linewidth=.8)
        for fid,r in connections.iterrows():
            draw(ax,r.geometry,color='#00ffff',linewidth=1.4)
            c=r.geometry.centroid
            ax.text(c.x,c.y,str(fid),fontsize=5,color='black',bbox=dict(facecolor='white',alpha=.65,pad=.2,edgecolor='none'))
        ax.set(xlim=(roi.bounds[0],roi.bounds[2]),ylim=(roi.bounds[1],roi.bounds[3]),aspect='equal',title='Yellow: tile input; cyan: added connections (audit FID)');ax.set_axis_off()
        fig.tight_layout();fig.savefig(folder/'added_connections.png',dpi=200,bbox_inches='tight');plt.close(fig)
        records=[]
        for _,r in connections.iterrows():
            item=r.drop('geometry').to_dict();item['geometry_wkt']=r.geometry.wkt;records.append(item)
        write_json(folder/'connections.json',records)
        write_json(folder/'stage_deltas.json',stats)
        if row['id']=='industrial':
            industrial_details(folder,inputs,saved['crs'])
        rows.append(dict(row,stages=stats,connection_count=len(connections)))
        print(row['id'],len(connections),flush=True)
    verified=all(snapshot(p)==s for p,s in signatures.items())
    write_json(output/'provenance.json',dict(run_root=str(run),period=inputs['period'],crs=saved['crs'],
        input_file_signatures=signatures,unchanged=verified,
        interpretation='Current cached stages, not a replay of historical source code. Changes in counts are not accuracy.',
        regions=rows))
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if not verified:raise RuntimeError('A read-only audit input changed concurrently')


if __name__=='__main__':main()
