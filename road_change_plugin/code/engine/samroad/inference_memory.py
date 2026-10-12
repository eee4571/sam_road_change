"""Exact tensor spill and pre-decode size contracts. No model imports."""
from pathlib import Path
import tempfile
import numpy as np


def check_image_shape(width, height, maximum_megapixels=25., *, stage='input'):
    width, height = int(width), int(height)
    # A disabled CLI limit must not permit a single unbounded allocation.
    limit = min(float(maximum_megapixels), 25.) if maximum_megapixels > 0 else 25.
    if min(width, height) <= 0 or width*height > limit*1_000_000:
        raise ValueError(f'{stage} grid {width}x{height} exceeds the inference memory contract '
                         f'({limit:g} MP); split into georeferenced tiles before inference')


def preflight_image(path, maximum_megapixels):
    import rasterio
    from engine.raster_memory import check_native_blocks
    with rasterio.open(path) as ds:
        check_image_shape(ds.width, ds.height, maximum_megapixels)
        check_native_blocks(ds)


class TensorSpool:
    """At most one batch is restored; completed batches occupy disk, not VRAM.

    NPY is lossless and never converts dtype. Scratch cleanup occurs on every
    exit, including failure. Disk errors propagate to normal checkpoint recovery.
    """
    def __init__(self, ram_limit=None):
        if ram_limit is None:
            from engine.resource_policy import resources,configured_bytes,MIB
            available=resources().available_ram
            ram_limit=min(1024*MIB,available//8,
                configured_bytes('ROAD_PIPELINE_RAM_MB',available)//4)
        self.temp = tempfile.TemporaryDirectory(prefix='road-features-')
        self.count = 0
        self.ram_limit=ram_limit;self.used=0;self.memory={}

    def append(self, *tensors):
        for column, tensor in enumerate(tensors):
            array=tensor.detach().cpu().numpy()
            if self.used+array.nbytes <= self.ram_limit:
                # Copy CPU views so a small tensor cannot retain a large backing
                # allocation. CUDA .cpu() has already detached device storage.
                self.memory[self.count,column]=array.copy();self.used+=array.nbytes
            else:
                np.save(Path(self.temp.name)/f'{self.count}-{column}.npy',array,allow_pickle=False)
        self.count += 1

    def read(self, index, column, device):
        import torch
        array = self.memory.get((index,column))
        if array is None:array=np.load(Path(self.temp.name)/f'{index}-{column}.npy',allow_pickle=False)
        return torch.from_numpy(array).to(device)

    def close(self):
        self.memory.clear();self.used=0
        self.temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()


def topology_scores(net, features, points, pairs, valid, masks, query_budget=4096):
    """Keep attention's complete neighbour group; split independent queries.

    Small batches retain the original call. Large batches never place all
    pair activations on the GPU at once. Only output probabilities live on CPU.
    """
    import torch
    batch,count=pairs.shape[:2]
    device=features.device
    with torch.no_grad():
        if batch*count<=query_budget:
            result=net.infer_toponet(features,torch.as_tensor(points,device=device),
                torch.as_tensor(pairs,device=device),torch.as_tensor(valid,device=device),masks)
            yield slice(None),slice(None),torch.nan_to_num(result,nan=-100.).squeeze(-1).cpu().numpy()
            return
        for b in range(batch):
            local_points=torch.as_tensor(points[b:b+1],device=device)
            for start in range(0,count,query_budget):
                portion=slice(start,min(count,start+query_budget))
                result=net.infer_toponet(features[b:b+1],local_points,
                    torch.as_tensor(pairs[b:b+1,portion],device=device),
                    torch.as_tensor(valid[b:b+1,portion],device=device),masks[b:b+1])
                yield slice(b,b+1),portion,torch.nan_to_num(result,nan=-100.).squeeze(-1).cpu().numpy()


def topology_batches(patches,batch_size,index,padded_point_budget=8192):
    """Plan before kNN/collation: dense patches cannot pad all 16 neighbours.

    One patch retains all its points. Its 512x512 source bounds the worst
    case, while GPU attention is independently split by topology_scores.
    """
    for batch,start in enumerate(range(0,len(patches),batch_size)):
        local=patches[start:start+batch_size];first=0;largest=0
        for i,(_,low,high) in enumerate(local):
            count=index.count((*low,*high))
            if i>first and max(largest,count)*(i-first+1)>padded_point_budget:
                yield batch,slice(first,i)
                first=i;largest=0
            largest=max(largest,count)
        yield batch,slice(first,len(local))
