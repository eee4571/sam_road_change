"""GIS-ready TIFF copies; strip stale source statistics, preserve run inputs."""
import os
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from run_experiment import ROOT, PERIODS, environment, save, sha
os.environ.update(environment())
output=ROOT/'normalized_tiffs'
output.mkdir(exist_ok=True)
shutil.copy2(ROOT/f'inputs/normalized/{PERIODS[0]}.tif',output/f'{PERIODS[0]}.tif')
source=ROOT/f'inputs/normalized/{PERIODS[1]}.tif'
target=output/f'{PERIODS[1]}.tif'
with rasterio.open(source) as src, rasterio.open(target,'w',**src.profile) as dst:
    dst.colorinterp=src.colorinterp
    dst.update_tags(**{k:v for k,v in src.tags().items() if not k.startswith('STATISTICS_')})
    for band in src.indexes:
        dst.update_tags(band,**{k:v for k,v in src.tags(band).items() if not k.startswith('STATISTICS_')})
    for y in range(0,src.height,256):
        window=Window(0,y,src.width,min(256,src.height-y))
        dst.write(src.read(window=window),window=window)
with rasterio.open(source) as src,rasterio.open(target) as dst:
    assert src.profile==dst.profile
    for y in range(0,src.height,256):
        window=Window(0,y,src.width,min(256,src.height-y))
        assert np.array_equal(src.read(window=window),dst.read(window=window))
        assert np.array_equal(src.read_masks(window=window),dst.read_masks(window=window))
save(ROOT/'diagnostics/published_tiff_identity.json',dict(source=str(source),published=str(target),
    source_sha256=sha(source),published_sha256=sha(target),pixels_masks_and_grid_identical=True,
    difference='Only stale STATISTICS_* metadata removed. Actual B inference input retained unchanged.'))
print('GIS-ready normalized TIFFs verified')
