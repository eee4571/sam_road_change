"""Input-only verification and fixed-display previews; never opens GT."""
import json
import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.windows import Window
from run_experiment import ROOT, environment, save
os.environ.update(environment())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cfg=json.loads((ROOT/'config/experiment.json').read_text(encoding='utf-8-sig'))
area=gpd.read_file(cfg['validation_area'])
sources=[ROOT/'inputs/raw/20250118.tif', ROOT/'inputs/raw/20260203.tif',ROOT/'inputs/normalized/20260203.tif']
titles=['20250118 reference (raw)','20260203 raw (A)','20260203 normalized (B)']
fig,axes=plt.subplots(1,3,figsize=(12,5),layout='constrained')
result={}
for path,title,axis in zip(sources,titles,axes):
    with rasterio.open(path) as src:
        geom=list(area.to_crs(src.crs).geometry)
        collected=[]
        for y in range(0,src.height,512):
            w=Window(0,y,src.width,min(512,src.height-y))
            data=src.read(window=w)
            valid=(src.read_masks(window=w)>0).all(axis=0)
            valid &= geometry_mask(geom,data.shape[1:],src.window_transform(w),invert=True)
            collected.append(data[:,::8,::8][:,valid[::8,::8]])
        sample=np.concatenate(collected,axis=1)
        result[title]=dict(quantiles=[1,25,50,75,99],values=np.percentile(sample,[1,25,50,75,99],axis=1).T.tolist(),
            crs=str(src.crs),transform=list(src.transform),shape=list(src.shape),nodata=src.nodata)
        rgb=src.read(out_shape=(3,800,max(1,round(800*src.width/src.height))))
        axis.imshow(np.moveaxis(rgb,0,-1),vmin=0,vmax=255)
        axis.set_title(title);axis.axis('off')
fig.savefig(ROOT/'diagnostics/radiometric_inputs.png',dpi=140)
save(ROOT/'diagnostics/radiometric_qc.json',result)
print('radiometric QC complete')
