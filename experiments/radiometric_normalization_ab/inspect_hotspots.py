"""Post-evaluation visual audit of the largest A removed objects without GT overlap."""
import os
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_bounds
from run_experiment import ROOT, PERIODS, environment, save
os.environ.update(environment())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
metrics=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'))
frame=gpd.read_file(ROOT/'evaluation/A_no_gt_intersection.gpkg').to_crs(32650)
removed=frame.loc[frame.change_typ=='removed'].copy()
removed['audit_area']=removed.area
selected=[]
for _,row in removed.sort_values('audit_area',ascending=False).iterrows():
    if all(row.geometry.distance(other.geometry)>100 for other in selected):selected.append(row)
    if len(selected)==2:break
sources=[ROOT/'inputs/raw/20250118.tif',ROOT/'inputs/raw/20260203.tif',ROOT/'inputs/normalized/20260203.tif']
lines=[]
for arm,period in [('A',PERIODS[0]),('A',PERIODS[1]),('B',PERIODS[1])]:
    p=ROOT/arm/f'_work/tasks/runs/pair_ab/grids/area/periods/{period}/latest_result.json'
    result=json.loads(p.read_text(encoding='utf-8'))
    lines.append(gpd.read_file(result['centerlines']).to_crs(32650))
fig,axes=plt.subplots(len(selected),3,figsize=(12,4*len(selected)),squeeze=False,layout='constrained')
records=[]
for i,row in enumerate(selected):
    bounds=row.geometry.buffer(30).bounds
    left,bottom,right,top=bounds
    width=600;height=max(100,round(600*(top-bottom)/(right-left)))
    transform=from_bounds(*bounds,width,height)
    records.append(dict(rank=i+1,area_m2=row.audit_area,bounds_utm50n=list(bounds),object_id=str(row.name)))
    for j,(path,layer) in enumerate(zip(sources,lines)):
        with rasterio.open(path) as src, WarpedVRT(src,crs=32650,transform=transform,width=width,height=height) as ds:
            rgb=ds.read([1,2,3])
        ax=axes[i,j];ax.imshow(np.moveaxis(rgb,0,-1),extent=(left,right,bottom,top))
        local=layer.cx[left:right,bottom:top]
        if len(local):local.plot(ax=ax,color=['#3ea2ff','#f6c045','#00e5cd'][j],linewidth=.75)
        gpd.GeoSeries([row.geometry],crs=32650).boundary.plot(ax=ax,color='#ff4747',linewidth=1)
        ax.set_xlim(left,right);ax.set_ylim(bottom,top);ax.set_axis_off()
        ax.set_title(['Reference + extracted axes','Raw after + A axes','Normalized after + B axes'][j])
fig.suptitle('Largest A removed objects without GT overlap | red: A removed boundary',fontsize=12)
fig.savefig(ROOT/'diagnostics/removed_hotspots.png',dpi=160)
save(ROOT/'diagnostics/removed_hotspots.json',records)
print('hotspot audit rendered')
