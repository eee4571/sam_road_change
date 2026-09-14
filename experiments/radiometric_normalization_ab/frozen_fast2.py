"""Run identical frozen Fast2 Auto on both fresh, pre-finalization arm products."""
import os
import sys
import time
from run_experiment import ROOT, CODE, PERIODS, environment, save
os.environ.update(environment())
sys.path.insert(0,str(CODE))
from engine.fast_pipeline import detect_fast_changes
from engine.fast_multitemporal import AUTO_REVISION
import json
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--arms',nargs='+',choices=['A','B'],default=['A','B'])
args=parser.parse_args()
for arm in args.arms:
    while True:
        path=ROOT/f'status_{arm}.json'
        state=json.loads(path.read_text(encoding='utf-8'))['state'] if path.exists() else 'pending'
        if state=='completed': break
        if state=='failed': raise RuntimeError(f'{arm} failed')
        time.sleep(5)
    manifest=json.loads((ROOT/arm/'_work/tasks/runs/pair_ab/pipeline_result.json').read_text(encoding='utf-8'))
    assert not manifest.get('failures') and not manifest['input_spec']['truths']
    periods=[next(p for p in manifest['auto_period_results'] if p['period']==period) for period in PERIODS]
    started=time.perf_counter()
    result=detect_fast_changes(*periods,ROOT/arm/'frozen_fast2',before_period=PERIODS[0],after_period=PERIODS[1],
        position_tolerance=3.,width_change_absolute=2.,width_change_ratio=.2,internal_outputs=True,temporal_results=None)
    result.update(frozen_revision=AUTO_REVISION,measured_wall_seconds=time.perf_counter()-started)
    save(ROOT/arm/'frozen_fast2/result.json',result)
    print('FROZEN FAST2 COMPLETE',arm,flush=True)
