"""Reject native decoder blocks that cannot be made safe by windowed reads."""
import numpy as np


def check_native_blocks(dataset,limit=64*1024*1024):
    # PNG scanlines and pixel-interleaved TIFF strips may be decoded in full
    # even when the requested NumPy result is just a small window.
    sizes=[int(h)*int(w)*np.dtype(dtype).itemsize
           for (h,w),dtype in zip(dataset.block_shapes,dataset.dtypes)]
    if sizes and max(sizes)*dataset.count>limit:
        from .width.width_reliability import InputContractError
        raise InputContractError(f'Native raster block exceeds {limit} bytes; convert to tiled TIFF: {dataset.name}')
