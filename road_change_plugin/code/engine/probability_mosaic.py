"""Stream existing probability tiles to a sparse, georeferenced disk product."""
from pathlib import Path
import math
import tempfile
import warnings
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from .raster_memory import check_native_blocks

BLOCK = 512
MAX_OUTPUT_BLOCKS = 1_000_000  # Also bound GDAL's TIFF directory, not just pixels.


def windows(width,height,x=0,y=0):
    for row in range(y,y+height,BLOCK):
        for col in range(x,x+width,BLOCK):
            yield Window(col,row,min(BLOCK,x+width-col),min(BLOCK,y+height-row))


def write_probability_mosaic(images,probability_dir,output,suffix,placement_function):
    sources=[]
    for image in images:
        path=Path(probability_dir)/(Path(image).stem+suffix)
        if not path.is_file():continue
        with rasterio.open(image) as ds, warnings.catch_warnings():
            check_native_blocks(ds)
            warnings.simplefilter('ignore',rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(path) as probability:
                check_native_blocks(probability)
                if probability.shape!=ds.shape or ds.crs is None:continue
                if probability.count!=1 or probability.dtypes[0]!='uint8':
                    raise ValueError(f'Expected a single uint8 probability band: {path}')
            sources.append(dict(image=image,path=path,shape=ds.shape,transform=ds.transform,
                                bounds=ds.bounds,crs=ds.crs))
    if not sources:return None
    crs=sources[0]['crs'];sources=[s for s in sources if s['crs']==crs]
    xres=float(np.median([abs(s['transform'].a) for s in sources]))
    yres=float(np.median([abs(s['transform'].e) for s in sources]))
    left=min(s['bounds'].left for s in sources);right=max(s['bounds'].right for s in sources)
    bottom=min(s['bounds'].bottom for s in sources);top=max(s['bounds'].top for s in sources)
    if not all(math.isfinite(v) and v>0 for v in (xres,yres,right-left,top-bottom)):
        raise ValueError('Invalid probability mosaic grid')
    width=max(1,math.ceil((right-left)/xres));height=max(1,math.ceil((top-bottom)/yres))
    if math.ceil(width/BLOCK)*math.ceil(height/BLOCK)>MAX_OUTPUT_BLOCKS:
        raise ValueError('Probability mosaic TIFF directory exceeds safe block count; check grid/region extent')
    transform=from_origin(left,top,xres,yres)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    profile=dict(driver='GTiff',width=width,height=height,count=1,dtype='uint8',
                 crs=crs,transform=transform,tiled=True,blockxsize=BLOCK,blockysize=BLOCK,
                 compress='deflate',SPARSE_OK=True,BIGTIFF='IF_SAFER')
    from .width.raw_memory import bounded_gdal_cache
    with tempfile.TemporaryDirectory(prefix='probability-',dir=output.parent) as work, bounded_gdal_cache(), rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        temporary=Path(work)/'mosaic.tif'
        with rasterio.open(temporary,'w+',**profile) as dst:
            # Creating an internal mask leaves untouched sparse blocks invalid.
            dst.write_mask(np.zeros((1,1),np.uint8),window=Window(0,0,1,1))
            for source in sources:
                placement=placement_function(source['transform'],source['shape'],transform,(height,width)) if placement_function else None
                with rasterio.open(source['image']) as image, warnings.catch_warnings():
                    warnings.simplefilter('ignore',rasterio.errors.NotGeoreferencedWarning)
                    with rasterio.open(source['path']) as probability:
                        if placement is not None:
                            rows,cols=placement
                            for win in windows(probability.width,probability.height):
                                target=Window(win.col_off+cols.start,win.row_off+rows.start,win.width,win.height)
                                dst.write(probability.read(1,window=win),1,window=target)
                                dst.write_mask(image.dataset_mask(window=win),window=target)
                        else:
                            _warp_tile(source,probability,image,dst,Path(work)/'source.tif')
            dst.update_tags(evidence_type='SAMRoad scene-relative road probability',
                            source='existing per-tile probability products; no inference rerun')
        temporary.replace(output)
        Path(str(output)+'.msk').unlink(missing_ok=True)
    return str(output.resolve())


def _warp_tile(source,probability,image,dst,temporary):
    # A small disk-backed native-grid source lets GDAL warp bands and the exact
    # original dataset mask independently, without full-image Python arrays.
    profile=dict(driver='GTiff',width=image.width,height=image.height,count=2,dtype='uint8',
                 crs=source['crs'],transform=source['transform'],tiled=True,
                 blockxsize=BLOCK,blockysize=BLOCK,compress='deflate',BIGTIFF='IF_SAFER')
    with rasterio.open(temporary,'w',**profile) as out:
        for win in windows(image.width,image.height):
            out.write(probability.read(1,window=win),1,window=win)
            out.write(image.dataset_mask(window=win),2,window=win)
    bounds=from_bounds(*source['bounds'],transform=dst.transform)
    x=max(0,math.floor(bounds.col_off));y=max(0,math.floor(bounds.row_off))
    right=min(dst.width,math.ceil(bounds.col_off+bounds.width))
    bottom=min(dst.height,math.ceil(bounds.row_off+bounds.height))
    options=dict(crs=dst.crs,transform=dst.transform,width=dst.width,height=dst.height,
                 warp_mem_limit=32,src_nodata=None,add_alpha=True)
    with rasterio.open(temporary) as native, WarpedVRT(native,resampling=Resampling.bilinear,**options) as rgb, WarpedVRT(native,resampling=Resampling.nearest,**options) as mask:
        for win in windows(right-x,bottom-y,x,y):
            covered=rgb.read(3,window=win)>0
            data=dst.read(1,window=win);valid=dst.dataset_mask(window=win)
            warped=rgb.read(1,window=win);warped_valid=mask.read(2,window=win)
            data[covered]=warped[covered];valid[covered]=warped_valid[covered]
            dst.write(data,1,window=win);dst.write_mask(valid,window=win)
