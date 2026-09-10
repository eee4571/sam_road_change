"""Batch-owned model worker. No models are imported into the GUI/backend host."""
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PREFIX = '__FAST_WORKER_DONE__'
_active = None


class ModelPool:
    def __init__(self):
        self.items = {}

    def acquire(self, family, key, factory, device):
        started = time.perf_counter()
        identity = (family, key)
        for saved,value in self.items.items():
            if saved != identity:
                getattr(value, 'model', value).to('cpu')
        # Retain at most one configuration per family, on CPU between stages.
        reused = identity in self.items
        if not reused:
            self.items = {k:v for k,v in self.items.items() if k[0] != family}
            self.items[identity] = factory()
        value = self.items[identity]
        getattr(value, 'model', value).to(getattr(value, 'device', device) if str(device) == 'auto' else device)
        print(f'[Fast batch timing] model_load_{family}={time.perf_counter()-started:.6f}s '
              f'{family}_model_reuse={int(reused)} {family}_model_load_count={int(not reused)}', flush=True)
        return value

    def park(self):
        for value in self.items.values():
            getattr(value, 'model', value).to('cpu')


def resource_key(*paths):
    return tuple((str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths)


class Worker:
    def __init__(self):
        self.process = None
        self.metrics = Counter()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.planner = None

    def run(self, command, cwd, env):
        key = command_key(command)
        if self.pending:
            pending_key, future = self.pending
            started = time.perf_counter()
            try:
                code = future.result()
            except Exception:
                if pending_key == key:
                    self.pending = None
                    raise
            print(f'[Fast batch timing] gpu_pipeline_wait={time.perf_counter()-started:.6f}s', flush=True)
            self.metrics['gpu_pipeline_wait'] += time.perf_counter()-started
            if pending_key == key:
                self.pending = None
                self._prefetch(command,cwd,env)
                return code
            pending_root = dict(pending_key[1]).get('--output_root')
            if pending_root and command[1:4] == ['-m','engine.fast_pipeline','width'] and '--output-dir' in command:
                if Path(command[command.index('--output-dir')+1]).parent == Path(pending_root).parent:
                    # Resume may validate/skip the centreline stage entirely.
                    self.pending = None
        code = self._execute(command,cwd,env)
        self._prefetch(command,cwd,env)
        return code

    def _prefetch(self, command, cwd, env):
        if self.planner and Path(command[1]).name == 'inferencer.py' and self.pending is None:
            planner, self.planner = self.planner, None
            try:
                following = planner(command)
            except Exception as exc:
                # Preparation failure belongs to the following period's normal
                # execution/retry, never to the already completed current stage.
                print(f'[Fast batch] prefetch preparation deferred: {exc}', flush=True)
                return
            if following:
                self.pending = (command_key(following), self.executor.submit(self._execute, following, cwd, env))

    def _execute(self, command, cwd, env):
        if self.process is None:
            root = Path(__file__).resolve().parents[1]
            self.process = subprocess.Popen([command[0], '-m', 'engine.batch_runtime'],
                cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        p = self.process
        p.stdin.write(json.dumps({'command':command, 'cwd':str(cwd)})+'\n'); p.stdin.flush()
        for line in p.stdout:
            if line.startswith(PREFIX):
                result = json.loads(line[len(PREFIX):])
                if result.get('error'):
                    raise RuntimeError(result['error'])
                return 0
            print(line.rstrip(), flush=True)
            self.record_line(line)
        raise RuntimeError(f'Fast model worker exited unexpectedly: {p.poll()}')

    def record_line(self, line):
        if '[Fast timing]' in line or '[Fast batch timing]' in line:
            for key,value in re.findall(r'(\w+)=([0-9.]+)',line):
                if key == 'queue_capacity':
                    self.metrics[key] = max(self.metrics[key],float(value))
                else:
                    self.metrics[key] += float(value)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.process is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()


def run_resident(command, cwd, env):
    if _active is None:
        return None
    samroad = len(command)>1 and Path(command[1]).name == 'inferencer.py' and '--execution-profile' in command
    width = command[1:4] == ['-m','engine.fast_pipeline','width']
    if not (samroad or width):
        return None
    return _active.run(command, cwd, env)


def plan_next_centerline(planner):
    if _active is not None:
        _active.planner = planner


def record_timing(name, value):
    if _active is not None:
        _active.metrics[name] += value


def record_line(line):
    if _active is not None:
        _active.record_line(line)


def normalization_lock(function):
    @functools.wraps(function)
    def wrapped(periods, validation_area, output_root, *args, **kwargs):
        from filelock import FileLock
        work = next((p for p in Path(output_root).resolve().parents if p.name == '_work'), None)
        root = Path(kwargs.get('cache_root') or (work/'cache'/'normalized' if work else output_root))
        root.mkdir(parents=True,exist_ok=True)
        started = time.perf_counter()
        with FileLock(str(root/'.normalization.lock')):
            record_timing('normalization_cache_wait',time.perf_counter()-started)
            return function(periods,validation_area,output_root,*args,**kwargs)
    return wrapped


def command_key(command):
    positional, options = [], []
    index = 0
    while index < len(command):
        value = command[index]
        if value.startswith('--'):
            argument = command[index+1] if index+1 < len(command) and not command[index+1].startswith('--') else None
            options.append((value, argument))
            index += 2 if argument is not None else 1
        else:
            positional.append(value); index += 1
    return tuple(positional), tuple(sorted(options))


def batch_models(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        global _active
        if _active is not None:
            return function(*args, **kwargs)
        started = time.perf_counter()
        _active = Worker()
        try:
            return function(*args, **kwargs)
        finally:
            worker, _active = _active, None
            worker.close()
            worker.metrics['total'] = time.perf_counter()-started
            print('[Fast batch summary] '+json.dumps(dict(worker.metrics),sort_keys=True), flush=True)
            print(f'[Fast batch timing] total={time.perf_counter()-started:.6f}s', flush=True)
    return wrapped


def main():
    import traceback
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'engine/samroad'))
    sys.path.insert(0,str(root/'engine/width'))
    pool = ModelPool()
    for line in sys.stdin:
        result = {}
        try:
            request = json.loads(line)
            command = request['command']; os.chdir(request['cwd'])
            if Path(command[1]).name == 'inferencer.py':
                import inferencer
                inferencer.main(command[2:], model_pool=pool)
            else:
                from engine.fast_pipeline import main as fast_main
                fast_main(command[3:], model_pool=pool)
        except BaseException as exc:
            traceback.print_exc()
            result['error'] = f'{type(exc).__name__}: {exc}'
            pool.park()
        print(PREFIX+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
