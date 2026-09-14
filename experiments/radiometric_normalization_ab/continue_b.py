"""Run B as soon as A releases the GPU, retaining independent arm caches."""
import time
from run_experiment import ROOT, run_arm
import json
while True:
    state=json.loads((ROOT/'status_A.json').read_text(encoding='utf-8'))
    if state['state']=='failed': raise RuntimeError('A failed; B not launched')
    if state['state']=='completed': break
    time.sleep(5)
run_arm('B')
