"""Explicit cached-pair experiment: never invokes extraction or publishes products."""
import argparse
import cProfile
import hashlib
import importlib.util
import json
import pickle
import pstats
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine/width'))
from engine.fast_pipeline import _load_fast_period_result, _read_fast_change_layer
from engine.auto_scene_cache import scene_key, close_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('label')
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--probability-probe', choices=('trace','detailed'))
    parser.add_argument('--raster-io-probe', action='store_true')
    parser.add_argument('--monitor-memory', action='store_true')
    parser.add_argument('--strip-read-batch', type=int, choices=(1, 8, 16, 32))
    args = parser.parse_args()
    output = args.job / '_profiling' / 'auto_pair'
    output.mkdir(parents=True, exist_ok=True)
    source = args.source or Path(__file__).resolve().parents[1] / 'engine/fast_auto_change.py'
    text = source.read_text(encoding='utf-8')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / f'{args.label}_source.py').write_text(text, encoding='utf-8')
    # Instrument exclusive blocks in an experiment-only module. Production
    # code, candidate decisions and all output records remain untouched.
    text = text.replace('    records, audit, width_audit = [], [], []',
                        '    records, audit, width_audit = [], [], []\n    global phases\n    phases = Counter()')
    anchors = [
        ('            sampling_started = time.perf_counter()', '            phase_started = time.perf_counter()\n'),
        ('            match_started = time.perf_counter()\n            matches =', "            phases['axis_preparation'] += time.perf_counter()-phase_started\n            phase_started = time.perf_counter()\n"),
        ('            for index in selected:\n', "            phases['matching_and_section_preparation'] += time.perf_counter()-phase_started\n            phase_started = time.perf_counter()\n"),
        ('            local, intervals, presence_counts = presence_seeds(', "            phases['station_loop'] += time.perf_counter()-phase_started\n            phase_started = time.perf_counter()\n"),
        ('            if side == "before" and len(samples) >= 2:', "            phases['presence_seeds'] += time.perf_counter()-phase_started\n            phase_started = time.perf_counter()\n"),
        ('            if line_id % 25 == 0:', "            phases['width_result_construction'] += time.perf_counter()-phase_started\n"),
    ]
    for anchor, prefix in anchors:
        assert text.count(anchor) == 1, anchor
        text = text.replace(anchor, prefix+anchor)
    module_path = output / f'{args.label}_instrumented.py'
    module_path.write_text(text, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('engine.profile_pair_auto', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if args.strip_read_batch is not None:
        module.WindowedProbability._STRIP_READ_BATCH = args.strip_read_batch
    if args.probability_probe:
        import probability_probe
        probability_probe.install(module, detailed=args.probability_probe=='detailed', raster_io=args.raster_io_probe)
    # cProfile only observes its own thread. Capture the two surface workers
    # separately so a threaded hot path cannot disappear from hotspot reports.
    worker_profiles = []
    if not args.no_profile and hasattr(module, '_prepare_axis_surface'):
        prepare_surface = module._prepare_axis_surface
        local = threading.local()
        def profiled_surface(*values, **kwargs):
            if not hasattr(local, 'profiler'):
                local.profiler = cProfile.Profile()
                worker_profiles.append(local.profiler)
            local.profiler.enable()
            try:
                return prepare_surface(*values, **kwargs)
            finally:
                local.profiler.disable()
        module._prepare_axis_surface = profiled_surface
    manifest = json.loads((args.job/'pipeline_result.json').read_text(encoding='utf-8'))
    pair = manifest['change_results'][0]
    # Completed manifests expose reconciled roads as period_results. Auto
    # profiling must keep using the preserved extraction cache in that case.
    period_cache = manifest.get('auto_period_results') or manifest['period_results']
    periods = [next(p for p in period_cache if p['grid']==pair['grid'] and p['period']==pair[k])
               for k in ('before_period', 'after_period')]
    payloads = [_load_fast_period_result(p) for p in periods]
    centers = [_read_fast_change_layer(p, 'centerlines') for p in payloads]
    crs = centers[0].crs
    if not crs.is_projected or abs(crs.axis_info[0].unit_conversion_factor-1)>1e-6:
        crs = centers[0].estimate_utm_crs()
    identities = [scene_key(p, crs) for p in payloads]
    scenes = []
    for payload, center in zip(payloads, centers):
        frames = [_read_fast_change_layer(payload, k).to_crs(crs) for k in ('surfaces','width_segments','valid_observation')]
        probability = module.WindowedProbability(payload['road_probability'], crs)
        scenes.append(module.RoadScene(center.to_crs(crs), *frames, probability, crs))
    print('PAIR', pair['before_period'], pair['after_period'], 'AXES', [len(s.lines) for s in scenes], flush=True)
    memory = None
    if args.monitor_memory:
        from profile_process_memory import ProcessMemory
        memory = ProcessMemory()
        memory.__enter__()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    presence_audit = []
    try:
        if not args.no_profile:
            profiler.enable()
        result = module.analyze_scenes(*scenes, tolerance=float(pair['tolerance']), absolute=float(pair['absolute']),
                                     relative=float(pair['ratio']), presence_audit=presence_audit)
    finally:
        if not args.no_profile:
            profiler.disable()
        elapsed = time.perf_counter()-started
        if memory is not None:
            memory.__exit__(None, None, None)
    if 'timing_surface_pipeline_wall_seconds' in result[3]:
        # Pipeline waiting occurs in the generator, outside the instrumented
        # preparation block; attribute it to the same phase as old versions.
        module.phases['axis_preparation'] += result[3]['timing_axis_surface_wait_seconds']
    assert identities == [scene_key(p, crs) for p in payloads], 'Inputs changed during profiling'
    probability_stats = probability_probe.reports(scenes) if args.probability_probe else None
    if args.raster_io_probe:
        for index, scene in enumerate(scenes):
            (output/f'{args.label}_scene{index}_miss_trace.bin').write_bytes(scene.probability._probability_probe.miss_events)
    for scene in scenes:
        close_scene(scene)
    with (output/f'{args.label}_results.pkl').open('wb') as f:
        pickle.dump((result,presence_audit), f, protocol=5)
    stats = []
    if not args.no_profile:
        combined = pstats.Stats(profiler)
        for worker_profile in worker_profiles:
            combined.add(worker_profile)
        combined.dump_stats(str(output/f'{args.label}.prof'))
        for (file,line,name),(cc,nc,tt,ct,callers) in combined.stats.items():
            stats.append(dict(file=file,line=line,name=name,primitive_calls=cc,calls=nc,self_seconds=tt,cumulative_seconds=ct))
    summary = dict(label=args.label,elapsed_seconds=elapsed,phases=module.phases,
                   source_sha256=source_hash,inputs=identities,
                   pair=[pair['before_period'],pair['after_period']],counts=result[3],
                   functions=sorted(stats,key=lambda row:row['cumulative_seconds'],reverse=True))
    if probability_stats is not None:
        summary['probability'] = probability_stats
    if memory is not None:
        summary['memory'] = memory.result
    summary['strip_read_batch'] = getattr(module.WindowedProbability, '_STRIP_READ_BATCH', 1)
    (output/f'{args.label}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('functions','inputs','probability')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
