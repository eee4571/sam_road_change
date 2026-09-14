"""Finish the frozen detector replay and offline evaluation after B completes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from run_experiment import ROOT, environment, save
os.environ.update(environment())
while True:
    p=ROOT/'status_B.json'
    state=json.loads(p.read_text(encoding='utf-8'))['state'] if p.exists() else 'pending'
    if state=='completed':break
    if state=='failed':raise RuntimeError('B failed; offline evaluation not started')
    time.sleep(5)
scripts=[] if (ROOT/'B/frozen_fast2/result.json').exists() else [('frozen_fast2.py',['--arms','B'])]
scripts += [('evaluate.py',[]),('plot_results.py',[]),('inspect_hotspots.py',[])]
for script,args in scripts:
    with (ROOT/'logs'/f'{Path(script).stem}.log').open('w',encoding='utf-8') as log:
        p=subprocess.run([sys.executable,'-B',str(ROOT/script),*args],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{script} failed; see its log')
save(ROOT/'status.json',dict(state='evaluated',time=time.time()))
print('A/B EVALUATED',flush=True)
