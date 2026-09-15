"""Bounded, thread-safe RGB tile reader. No mosaics are written or resampled."""
from collections import OrderedDict
from threading import RLock
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import map_coordinates


class TileMosaic:
    def __init__(self, paths):
        self.sources=[rasterio.open(p) for p in paths]
        first=self.sources[0]
        self.crs=first.crs;self.count=first.count;self.dtypes=first.dtypes
        if first.transform.b or first.transform.d:raise ValueError('RGB tiles require an aligned north-up grid')
        offsets=[]
        for ds in self.sources:
            if ds.crs!=first.crs or ds.res!=first.res or ds.dtypes!=first.dtypes:
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
        for ds in self.sources:ds.close()


class FeatureCache:
    """Cache pointwise features in fixed blocks; nonlocal filters keep crop semantics.

    Canny hysteresis can propagate arbitrarily far, so caching independently
    filtered blocks would change the validated algorithm at block boundaries.
    Exact crop-dependent filters are cached by their complete window instead.
    Both caches share one byte budget and LRU, including all derived arrays.
    """
    def __init__(self,ds,max_bytes=128*1024*1024,block_size=512):
        self.ds=ds;self.max_bytes=max_bytes;self.block_size=block_size
        self.items=OrderedDict();self.bytes=0;self.lock=RLock()
        self.hits=0;self.misses=0;self.evictions=0

    def _get(self,key,build):
        if key in self.items:
            self.hits+=1;self.items.move_to_end(key);return self.items[key]
        self.misses+=1;value=build();size=sum(a.nbytes for a in value)
        while self.items and self.bytes+size>self.max_bytes:
            _,old=self.items.popitem(last=False);self.bytes-=sum(a.nbytes for a in old);self.evictions+=1
        if size<=self.max_bytes:self.items[key]=value;self.bytes+=size
        return value

    def features(self,lo,hi):
        x,y=map(int,lo);w,h=map(int,hi-lo)
        with self.lock:
            def build():
                rgbf=np.empty((h,w,3),np.float32);lab=np.empty_like(rgbf)
                gray=np.empty((h,w),np.float32);mask=np.empty((h,w),np.uint8)
                b=self.block_size
                for by in range(y//b,(y+h-1)//b+1):
                    for bx in range(x//b,(x+w-1)//b+1):
                        def block():
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
