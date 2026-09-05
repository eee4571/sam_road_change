"""Standalone real-region surface preview; final centerlines are read only."""
from pathlib import Path
import argparse
import json
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from shapely.geometry import box, Point
from shapely.ops import unary_union
from shapely.prepared import prep

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.final_road_surface import build_final_road_surface


def imagery(paths,bounds,size):
    width,height=size
    dst=np.zeros((3,height,width),dtype=np.float32)
    transform=from_bounds(*bounds,width,height)
    for path in paths:
        with rasterio.open(path) as ds:
            footprint=transform_bounds(ds.crs,32650,*ds.bounds)
            if not box(*bounds).intersects(box(*footprint)): continue
            tile=np.zeros_like(dst)
            for b in range(3):
                reproject(rasterio.band(ds,b+1),tile[b],src_transform=ds.transform,src_crs=ds.crs,
                          dst_transform=transform,dst_crs=32650,resampling=Resampling.bilinear,
                          dst_nodata=0,num_threads=2)
            valid=tile.max(axis=0)>0
            dst[:,valid]=tile[:,valid]
    valid=dst.max(axis=0)>0
    result=np.ones_like(dst)
    for b in range(3):
        low,high=np.percentile(dst[b][valid],[1,99]) if valid.any() else (0,255)
        result[b]=np.clip((dst[b]-low)/max(high-low,1),0,1)
        result[b][~valid]=1
    return np.moveaxis(result,0,-1)


def preview(lines,surfaces,old,samples,paths,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Microsoft YaHei'
    bounds=lines.total_bounds
    ratios=lines.length/lines.geometry.map(lambda g:Point(g.coords[0]).distance(Point(g.coords[-1]))).clip(lower=1)
    long=lines[lines.length>150]
    straight=lines[(ratios<1.01)&(lines.length>250)].length.idxmax()
    variability=samples.groupby('road_id').width_m.agg(lambda s:s.max()-s.min())
    variable=variability.loc[long.index].idxmax()
    curved=ratios.loc[long.index].idxmax()
    old_union=prep(unary_union(old.geometry))
    fragmented=max(long.index,key=lambda i:sum(not old_union.covers(lines.geometry.iloc[i].interpolate(t,normalized=True))
                                              for t in np.linspace(.05,.95,21)))
    junctions=surfaces[surfaces.kind=='junction']
    junction=junctions.loc[junctions.area.idxmax()].geometry.centroid
    # Locate a genuine parallel pair of long observed centerline tracks.
    pair=None
    for i in long.length.sort_values(ascending=False).index:
        a=lines.geometry.iloc[i]
        for j in long.index:
            if i==j: continue
            b=lines.geometry.iloc[j]
            distance=a.distance(b)
            if not 8<distance<55: continue
            overlap=a.intersection(b.buffer(60)).length
            if overlap>180:
                p=a.intersection(b.buffer(60)).centroid
                pair=p;break
        if pair is not None:break
    regions=[('普通直路',lines.geometry.iloc[straight].interpolate(.5,normalized=True),160),
             ('宽度变化',lines.geometry.iloc[variable].interpolate(.5,normalized=True),180),
             ('弯路',lines.geometry.iloc[curved].interpolate(.5,normalized=True),180),
             ('双车道',pair or junction,200),('大型路口',junction,190),
             ('原道路面破碎',lines.geometry.iloc[fragmented].interpolate(.5,normalized=True),170)]
    def draw(ax,rgb,extent,kind):
        ax.imshow(rgb,extent=(extent[0],extent[2],extent[1],extent[3]))
        region=box(*extent)
        if kind=='old':
            layer=old[old.intersects(region)]
            if not layer.empty:layer.plot(ax=ax,facecolor='#d756e8',edgecolor='#e878f0',alpha=.35,linewidth=.5)
        if kind=='final':
            layer=surfaces[surfaces.intersects(region)]
            if not layer.empty:layer.plot(ax=ax,facecolor='#25bbd5',edgecolor='#00eaff',alpha=.45,linewidth=.7)
        if kind!='image':
            layer=lines[lines.intersects(region)]
            if not layer.empty:layer.plot(ax=ax,color='#ffcf3c',linewidth=.6)
        ax.set_xlim(extent[0],extent[2]);ax.set_ylim(extent[1],extent[3]);ax.set_aspect('equal');ax.set_axis_off()
    rgb=imagery(paths,bounds,(1300,1660))
    fig,ax=plt.subplots(figsize=(11,14),constrained_layout=True)
    draw(ax,rgb,bounds,'final');ax.set_title('验证区1 · 20221020｜原始影像 + Final Centerline + 重建道路面')
    fig.savefig(output/'overview.png',dpi=160);plt.close(fig)
    sheet,axes=plt.subplots(2,3,figsize=(16,11),constrained_layout=True)
    region_rows=[]
    for index,(name,p,radius) in enumerate(regions):
        extent=(p.x-radius,p.y-radius,p.x+radius,p.y+radius)
        rgb=imagery(paths,extent,(1000,1000))
        fig,axs=plt.subplots(1,3,figsize=(16,5.6),constrained_layout=True)
        for ax,kind,title in zip(axs,['image','old','final'],['原始影像','原分割道路面 + 最终中心线','重建道路面 + 最终中心线']):
            draw(ax,rgb,extent,kind);ax.set_title(title)
        fig.suptitle(name+'｜青色：重建道路面；黄色：未改动的中心线',fontsize=15)
        filename=f'detail_{index+1}.png';fig.savefig(output/filename,dpi=160);plt.close(fig)
        draw(axes.flat[index],rgb,extent,'final');axes.flat[index].set_title(name)
        region_rows.append(dict(name=name,bounds=extent,image=filename))
    sheet.suptitle('真实影像局部效果｜青色：重建道路面；黄色：Final Centerline',fontsize=16)
    sheet.savefig(output/'six_details.png',dpi=170);plt.close(sheet)
    return region_rows


def run(source,images,output):
    output.mkdir(parents=True,exist_ok=False)
    original=gpd.read_file(source/'road_centerlines.shp')
    original_wkb=list(original.geometry.to_wkb())
    lines=original.to_crs(32650).reset_index(drop=True)
    widths=gpd.read_file(source/'road_width_segments.shp').to_crs(32650)
    old=gpd.read_file(source/'road_surfaces.shp').to_crs(32650)
    surfaces,samples,stats=build_final_road_surface(lines,widths)
    print(json.dumps(stats),flush=True)
    gpkg=output/'final_road_surface.gpkg'
    original.to_file(gpkg,layer='final_centerlines',driver='GPKG')
    surfaces.to_crs(original.crs).to_file(gpkg,layer='final_road_surfaces',driver='GPKG',mode='a')
    samples.to_crs(original.crs).to_file(gpkg,layer='smoothed_width_samples',driver='GPKG',mode='a')
    surfaces.to_crs(original.crs).to_file(output/'road_surfaces.shp',encoding='UTF-8')
    assert list(gpd.read_file(gpkg,layer='final_centerlines').geometry.to_wkb())==original_wkb
    assert list(gpd.read_file(source/'road_centerlines.shp').geometry.to_wkb())==original_wkb
    union=unary_union(surfaces.geometry).buffer(.0001)
    prepared=prep(union)
    missing=sum(g.difference(union).length for g in lines.geometry if not prepared.covers(g))
    stats.update(centerline_outside_surface_m=missing,invalid_surface_count=int((~surfaces.is_valid).sum()),
                 source_centerlines=str((source/'road_centerlines.shp').resolve()),
                 source_widths=str((source/'road_width_segments.shp').resolve()),
                 centerline_coordinates_exactly_preserved=True)
    stats['details']=preview(lines,surfaces,old,samples,sorted(images.glob('*.tif')),output)
    (output/'surface_report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.images,args.output)
