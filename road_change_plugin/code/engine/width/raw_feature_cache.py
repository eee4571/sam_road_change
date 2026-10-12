"""Bounded, thread-safe RGB tile reader. No mosaics are written or resampled."""
from collections import OrderedDict
from concurrent.futures import Future
from threading import RLock
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import map_coordinates
from .raw_memory import MemoryPolicy, WORKSPACES, CACHE_LOCK, CACHES, admit_cache, bounded_gdal_cache


class TileMosaic:
    def __init__(self, paths):
        self.sources=[]
        self._cache_context=bounded_gdal_cache()
        self._cache_context.__enter__()
        try:
            for path in paths:self.sources.append(rasterio.open(path))
            self._initialize_grid()
        except BaseException:
            self.close()
            raise

    def _initialize_grid(self):
        if not self.sources:raise ValueError('RGB width requires at least one image tile')
        first=self.sources[0]
        self.crs=first.crs;self.count=first.count;self.dtypes=first.dtypes
        offsets=[]
        for ds in self.sources:
            if ds.crs is None:raise ValueError(f'RGB tile lacks CRS: {ds.name}')
            if ds.count<3 or ds.dtypes[:3]!=('uint8',)*3:
                raise ValueError(f'RGB width requires three uint8 bands: {ds.name}')
            if ds.transform.b or ds.transform.d or ds.transform.a<=0 or ds.transform.e>=0:
                raise ValueError(f'RGB tiles require an aligned north-up grid: {ds.name}')
            if ds.crs!=first.crs or ds.res!=first.res or ds.dtypes!=first.dtypes or ds.count!=first.count:
                raise ValueError('RGB tiles must share CRS, resolution and dtype')
            x,y=(~first.transform)*(ds.transform.c,ds.transform.f)
            if abs(x-round(x))+abs(y-round(y))>1e-5:raise ValueError('RGB tiles are not on one pixel grid')
            offsets.append((int(round(x)),int(round(y))))
        minx=min(x for x,y in offsets);miny=min(y for x,y in offsets)
        from affine import Affine
        self.transform=first.transform*Affine.translation(minx,miny)
        self.offsets=[(x-minx,y-miny) for x,y in offsets]
        self.width=max(x+d.width for (x,y),d in zip(self.offsets,self.sources))
        self.height=max(y+d.height for (x,y),d in zip(self.offsets,self.sources))

    def read_window(self,window):
        x,y,w,h=map(int,(window.col_off,window.row_off,window.width,window.height))
        MemoryPolicy().check_window(w,h)
        rgb=np.zeros((h,w,3),np.uint8);valid=np.zeros((h,w),bool)
        for ds,(dx,dy) in zip(self.sources,self.offsets):
            x0,y0=max(x,dx),max(y,dy);x1,y1=min(x+w,dx+ds.width),min(y+h,dy+ds.height)
            if x1<=x0 or y1<=y0:continue
            win=Window(x0-dx,y0-dy,x1-x0,y1-y0)
            data=ds.read([1,2,3],window=win).transpose(1,2,0)
            mask=(ds.read_masks([1,2,3],window=win)>0).all(0)
            sl=np.s_[y0-y:y1-y,x0-x:x1-x]
            use=mask & ~valid[sl];rgb[sl][use]=data[use];valid[sl]|=mask
        return rgb,valid

    def close(self):
        try:
            for ds in self.sources:ds.close()
        finally:
            if self._cache_context is not None:
                self._cache_context.__exit__(None,None,None)
                self._cache_context=None


class FeatureCache:
    """Cache pointwise features in fixed blocks; nonlocal filters keep crop semantics.

    Canny hysteresis can propagate arbitrarily far, so caching independently
    filtered blocks would change the validated algorithm at block boundaries.
    Exact crop-dependent filters are cached by their complete window instead.
    Both caches share one byte budget and LRU, including all derived arrays.
    """
    def __init__(self,ds,max_bytes=128*1024*1024,block_size=512,policy=None):
        self.ds=ds;self.max_bytes=max_bytes;self.block_size=block_size
        if not 1<=block_size<=512:raise ValueError('RGB feature blocks must be at most 512 pixels')
        self.policy=policy or MemoryPolicy()
        self.items=OrderedDict();self.bytes=0;self.lock=CACHE_LOCK
        self.read_lock=RLock();self.pending={}
        self.hits=0;self.misses=0;self.evictions=0
        self.max_window_pixels=0;self.window_builds=0
        with self.lock:CACHES.add(self)

    def evict(self):
        _,old=self.items.popitem(last=False)
        self.bytes-=sum(a.nbytes for a in old);self.evictions+=1

    def clear(self):
        with self.lock:
            self.items.clear();self.bytes=0

    def _get(self,key,build):
        with self.lock:
            if key in self.items:
                self.hits+=1;self.items.move_to_end(key);return self.items[key]
            future=self.pending.get(key)
            owner=future is None
            if owner:
                self.misses+=1;future=Future();self.pending[key]=future
            else:self.hits+=1
        if not owner:return future.result()
        try:
            value=build();size=sum(a.nbytes for a in value)
            with self.lock:
                if admit_cache(self,size):self.items[key]=value;self.bytes+=size
                # Wake same-key readers only after publication; no computations
                # or dataset I/O run while holding the cache-state lock.
                future.set_result(value);self.pending.pop(key)
            return value
        except BaseException as exc:
            with self.lock:
                future.set_exception(exc);self.pending.pop(key,None)
            raise

    def features(self,lo,hi):
        x,y=map(int,lo);w,h=map(int,hi-lo)
        self.policy.check_window(w,h)
        if x<0 or y<0 or x+w>self.ds.width or y+h>self.ds.height:
            raise ValueError('RGB feature window lies outside the image grid')
        with WORKSPACES.reserve(self.policy.window_bytes(w*h)):
            return self._features(x,y,w,h)

    def _features(self,x,y,w,h):
        def build():
            with self.lock:
                self.max_window_pixels=max(self.max_window_pixels,w*h);self.window_builds+=1
            rgbf=np.empty((h,w,3),np.float32);lab=np.empty_like(rgbf)
            gray=np.empty((h,w),np.float32);mask=np.empty((h,w),np.uint8)
            b=self.block_size
            for by in range(y//b,(y+h-1)//b+1):
                for bx in range(x//b,(x+w-1)//b+1):
                    def block():
                        with self.read_lock:
                            rgb,valid=self.ds.read_window(Window(bx*b,by*b,min(b,self.ds.width-bx*b),min(b,self.ds.height-by*b)))
                        f=rgb.astype(np.float32)/255
                        return f,cv2.cvtColor(f,cv2.COLOR_RGB2LAB),cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),valid.astype(np.uint8)
                    parts=self._get(('block',bx,by),block)
                    x0,y0=max(x,bx*b),max(y,by*b);x1,y1=min(x+w,(bx+1)*b),min(y+h,(by+1)*b)
                    src=np.s_[y0-by*b:y1-by*b,x0-bx*b:x1-bx*b];dst=np.s_[y0-y:y1-y,x0-x:x1-x]
                    for out,part in zip((rgbf,lab,gray,mask),parts):out[dst]=part[src]
            blurred=cv2.GaussianBlur(gray,(0,0),.8)
            gx=cv2.Sobel(blurred,cv2.CV_32F,1,0,ksize=3)/8
            gy=cv2.Sobel(blurred,cv2.CV_32F,0,1,ksize=3)/8
            edge=np.hypot(gx,gy)
            canny=cv2.dilate(cv2.Canny((blurred*255).astype('uint8'),40,100),np.ones((3,3),'uint8'))/255
            variance=cv2.boxFilter(gray*gray,-1,(7,7))-cv2.boxFilter(gray,-1,(7,7))**2
            texture=np.sqrt(np.maximum(variance,0))
            valid=cv2.erode(mask,np.ones((3,3),'uint8')).astype(float)
            return rgbf,lab,edge,canny,texture,valid
        return self._get(('window',x,y,w,h),build)
