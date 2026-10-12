"""Opt-in real GPU comparison on one cached 1024-pixel crop. No cache writes."""
import argparse
import ast
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/engine/samroad')]
os.environ['SAMROAD_MODELS_ROOT']=str(ROOT/'runtime/model')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=1024,choices=[512,1024,2048,4096])
    args=parser.parse_args()
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import torch
    import inferencer as module
    from utils import load_config
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this explicit GPU validation')
    with rasterio.open(args.image) as ds:
        w,h=min(args.size,ds.width),min(args.size,ds.height)
        image=ds.read([1,2,3],window=Window((ds.width-w)//2,(ds.height-h)//2,w,h)).transpose(1,2,0)
    config=load_config(str(ROOT/'runtime/config/samroad_inference.yaml'))
    config.RELATIVE_ROADNESS_ENABLED=False;config.RELATIVE_INJECT_INTO_TOPONET=False
    net=module.SAMRoadplus(config)
    state=torch.load(ROOT/'runtime/model/samroad/samroad.ckpt',map_location='cpu',weights_only=False)
    net.load_state_dict(state['state_dict'],strict=True);del state
    net.eval().cuda()
    module.args.device='cuda';module.args.execution_profile='fast'
    original=subprocess.check_output(['git','show','9b6dc28:road_change_plugin/code/engine/samroad/inferencer.py'],cwd=ROOT,text=True,encoding='utf8')
    node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='infer_one_img')
    node.name='baseline_infer_one_img'
    namespace=dict(module.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'baseline_inferencer','exec'),namespace)
    def run(function):
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();result=function(net,image,copy.deepcopy(config))
        torch.cuda.synchronize()
        return result,dict(seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),
                           peak_reserved=torch.cuda.max_memory_reserved())
    # Warm up real kernels without touching any existing inference outputs.
    warm,_=run(module.infer_one_img);del warm
    baseline,before=run(namespace['baseline_infer_one_img'])
    current,after=run(module.infer_one_img)
    comparison=[]
    for index in range(7):
        a,b=baseline[index],current[index]
        exact=np.array_equal(a,b)
        error=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size and a.shape==b.shape else 0.
        comparison.append(dict(output=index,shape=list(a.shape),exact=exact,maximum_absolute_error=error))
        if not exact:raise AssertionError(f'Inference output {index} changed: {comparison[-1]}')
    report=dict(device=torch.cuda.get_device_name(),shape=list(image.shape),baseline=before,current=after,comparison=comparison)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
