"""Snapshot the unmodified implementation, then compare compensation raw_input.

Synthetic raster/road fixtures only; no model or real project is run.
Usage: python code/tests/run_fast2_compensation_ab.py capture|compare
"""
import argparse
from collections import Counter
from pathlib import Path
import pickle
import sys
import tempfile
import json
import numpy as np
import rasterio

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_fast_auto_change import FastFinalAutoTests
from test_fast_image_structure import RawImageTests
from engine.fast_auto_v2 import analyze_scenes
from engine.fast_patch_verification import PatchVerifier
from engine.fast_multitemporal import estimate_width_bias


def exact(value):
    if hasattr(value, 'wkb'): return ('wkb', value.wkb)
    if isinstance(value, np.ndarray): return ('array', value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (float, np.floating)): return ('float', np.float64(value).tobytes())
    if isinstance(value, dict):
        return {k: exact(v) for k, v in value.items()
                if not k.startswith('timing_') and not k.endswith('_seconds')
                and k not in ('compensation', 'compensation_identity')}
    if isinstance(value, (list, tuple)): return [exact(v) for v in value]
    return value


def snapshot():
    f=FastFinalAutoTests();f.setUp()
    try:
        b=f.scene([f.road(30+6*i,3) for i in range(32)]+[f.road(235,8)])
        a=f.scene([f.road(30+6*i,4) for i in range(32)]+[f.road(15,8)])
        presence=[];network=analyze_scenes(b,a,presence_audit=presence)
        # Real paired patch reads, stable controls and publication, without inference.
        b=f.scene([f.road(20+14*i,6) for i in range(14)]+[f.road(230,8)])
        a=f.scene([f.road(20+14*i,6) for i in range(14)]+[f.road(245,8)])
        payloads=[]
        for i,s in enumerate((b,a)):
            root=Path(f.tmp.name)/f'images{i}';root.mkdir()
            rng=np.random.default_rng(159)
            rgb=rng.random((3,500,600),dtype=np.float32)*.2
            rgb+=s.probability.dataset.read(1)[None]*.65
            if i:rgb=rgb*.8+.1
            image=root/'rgb.tif'
            with rasterio.open(image,'w',driver='GTiff',width=600,height=500,count=3,
                               dtype='float32',crs=f.crs,transform=f.transform) as dst: dst.write(rgb)
            (root/'tile_summary.json').write_text(json.dumps({'image':str(image)}))
            payloads.append({'width_review':str(root)})
        verifier=PatchVerifier((b,a),payloads)
        try:
            presence2=[];verified=analyze_scenes(b,a,presence_audit=presence2,patch_verifier=verifier)
            image_audit=(verifier.calibration,verifier.audit,dict(verifier.counts))
        finally:verifier.close()
        t=RawImageTests();gray,lateral,bins,normal=t.data();other=gray.copy()
        gray[np.abs(lateral)<5]+=.6;other[np.abs(lateral)<10]+=.6
        features=t.features(gray,other*.8+.1,lateral,bins,normal)
        return exact(dict(network=network,presence=presence,verified=verified,presence2=presence2,
                          image_audit=image_audit,features=features,
                          width=estimate_width_bias(np.arange(40)%3+2.)))
    finally:f.doCleanups()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','compare'])
    args=parser.parse_args();path=Path('_profiling/fast2_compensation_ab/baseline.pkl')
    value=snapshot()
    if args.mode=='capture':
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(pickle.dumps(value))
        print('Captured pre-refactor baseline:',path)
    else:
        expected=pickle.loads(path.read_bytes())
        for key in expected: assert expected[key]==value[key], f'Exact mismatch: {key}'
        print('EXACT PASS: all candidates, audits, non-timing statistics, floats and geometry WKB')
