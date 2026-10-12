"""Reject native decoder blocks that cannot be made safe by windowed reads."""
import numpy as np
from collections import OrderedDict
from contextlib import ExitStack


def check_native_blocks(dataset,limit=64*1024*1024):
    # PNG scanlines and pixel-interleaved TIFF strips may be decoded in full
    # even when the requested NumPy result is just a small window.
    sizes=[int(h)*int(w)*np.dtype(dtype).itemsize
           for (h,w),dtype in zip(dataset.block_shapes,dataset.dtypes)]
    if sizes and max(sizes)*dataset.count>limit:
        from .width.width_reliability import InputContractError
        raise InputContractError(f'Native raster block exceeds {limit} bytes; convert to tiled TIFF: {dataset.name}')


class RasterPool:
    """Bounded open datasets/VRTs. Returned handles remain valid until next get."""
    def __init__(self, factory, capacity=8):
        self.factory=factory;self.capacity=capacity;self.items=OrderedDict()
    def get(self,key):
        if key not in self.items:
            while len(self.items)>=self.capacity:
                _,(stack,_) = self.items.popitem(last=False);stack.close()
            stack=ExitStack()
            try:
                # DatasetReader otherwise owns an implicit thread-local Env.
                # LRU eviction/closing may happen in a different worker thread;
                # explicit scope keeps that Env out of the dataset ExitStack.
                import rasterio
                with rasterio.Env():value=self.factory(key,stack)
            except BaseException:
                stack.close();raise
            self.items[key]=(stack,value)
        self.items.move_to_end(key)
        return self.items[key][1]
    def close(self):
        for stack,_ in self.items.values():stack.close()
        self.items.clear()
    def __enter__(self):return self
    def __exit__(self,*error):self.close()
