"""Cache-only road surface comparison; no extraction or width measurement."""
import argparse
import json
import time
from pathlib import Path
import geopandas as gpd
from engine.width.raw_road_surfaces import build_road_surfaces


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('products',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    before=gpd.read_file(args.products/'roads.gpkg',layer='corridors')
    fingerprint=list(before.geometry.to_wkb());attributes=before.drop(columns='geometry').copy(deep=True)
    image=args.products/'raw_width'/'regional_rgb.tif'
    started=time.perf_counter()
    after=build_road_surfaces(before,image_path=image)
    elapsed=time.perf_counter()-started
    assert fingerprint==list(before.geometry.to_wkb())
    assert attributes.equals(before.drop(columns='geometry'))
    assert after.is_valid.all()
    after.to_file(args.output/'road_surfaces.gpkg',layer='surfaces',driver='GPKG')
    after.to_file(args.output/'road_surfaces.shp',encoding='UTF-8')
    metric=before.estimate_utm_crs() if before.crs.is_geographic else before.crs
    before=before.to_crs(metric);after=after.to_crs(metric)
    baseline=before.union_all();final=after.union_all()
    report=dict(before_polygons=len(before),after_polygons=len(after),seconds=elapsed,
                before_union_area=baseline.area,after_area=final.area,
                added_area=final.difference(baseline).area,lost_area=baseline.difference(final).area,
                before_boundary_length=before.length.sum(),after_boundary_length=after.length.sum(),
                internal_records_unchanged=True,all_valid=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(20,14))
    for ax,frame,title in zip(axes,[before,after],['Before: sampling facets','After: connected road surfaces']):
        frame.plot(ax=ax,color='#e4b82c',edgecolor='#3c3419',linewidth=.25)
        ax.set_title(title);ax.set_aspect('equal');ax.axis('off')
    fig.tight_layout();fig.savefig(args.output/'full_comparison.png',dpi=180);plt.close(fig)
    # Densest local corridor area for a readable close-up.
    center=before.iloc[len(before)//2].geometry.centroid
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    for ax,frame,title in zip(axes,[before,after],['Before','After']):
        frame.cx[center.x-150:center.x+150,center.y-150:center.y+150].plot(ax=ax,color='#e4b82c',edgecolor='#3c3419',linewidth=.7)
        ax.set_xlim(center.x-150,center.x+150);ax.set_ylim(center.y-150,center.y+150)
        ax.set_title(title);ax.set_aspect('equal');ax.axis('off')
    fig.tight_layout();fig.savefig(args.output/'local_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
