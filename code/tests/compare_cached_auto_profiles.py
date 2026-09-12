"""Compare complete ordered cached-pair outputs, including byte-exact WKB."""
import argparse
import hashlib
import json
import pickle
import struct
from pathlib import Path

import numpy as np
from shapely.geometry.base import BaseGeometry


def canonical(value):
    if isinstance(value, BaseGeometry):
        return {'__wkb__': value.wkb_hex}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {'__array__': value.dtype.str, 'shape': value.shape, 'values': canonical(value.tolist())}
        return {'__array__': value.dtype.str, 'shape': value.shape, 'bytes': value.tobytes().hex()}
    if isinstance(value, np.generic):
        return {'__scalar__': value.dtype.str, 'bytes': value.tobytes().hex()}
    if isinstance(value, float):
        return {'__float64__': struct.pack('<d', value).hex()}
    return value


def encoded(value):
    return json.dumps(canonical(value), sort_keys=True, ensure_ascii=False, allow_nan=True,
                      separators=(',', ':')).encode('utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('before')
    parser.add_argument('after')
    args = parser.parse_args()
    summaries = [json.loads((args.directory/f'{label}.json').read_text(encoding='utf-8')) for label in (args.before,args.after)]
    assert summaries[0]['inputs'] == summaries[1]['inputs'], 'Input fingerprints differ'
    data = [pickle.loads((args.directory/f'{label}_results.pkl').read_bytes()) for label in (args.before,args.after)]
    report = {}
    for name, old, new in zip(('candidates','station_audit','width_audit','presence_audit'),
                              (*data[0][0][:3],data[0][1]), (*data[1][0][:3],data[1][1])):
        a,b = encoded(old),encoded(new)
        report[name] = dict(before_count=len(old),after_count=len(new),exact_equal=a==b,
                            before_sha256=hashlib.sha256(a).hexdigest(),after_sha256=hashlib.sha256(b).hexdigest())
        if a!=b:
            report[name]['first_different_index'] = next((i for i,(x,y) in enumerate(zip(old,new)) if encoded(x)!=encoded(y)),None)
    counts = [{k:v for k,v in item[0][3].items() if 'seconds' not in k} for item in data]
    report['counts_equal'] = encoded(counts[0])==encoded(counts[1])
    report['elapsed_seconds'] = [s['elapsed_seconds'] for s in summaries]
    report['phases'] = [s['phases'] for s in summaries]
    report['all_equal'] = report['counts_equal'] and all(report[k]['exact_equal'] for k in ('candidates','station_audit','width_audit','presence_audit'))
    if all('probability' in s for s in summaries):
        assert len(summaries[0]['probability']) == len(summaries[1]['probability']), 'Probability scene counts differ'
        probability = []
        for old, new in zip(summaries[0]['probability'], summaries[1]['probability']):
            checks = {
                'sampling_coordinates_and_values_exact': old['sample_sha256']==new['sample_sha256'],
                'sample_count_equal': old['counts']['sample_count']==new['counts']['sample_count'],
                'values_at_calls_equal': old['counts']['values_at_calls']==new['counts']['values_at_calls'],
                'cache_access_order_exact': old['cache_access_sha256']==new['cache_access_sha256'],
                'final_lru_order_exact': old['final_lru_sha256']==new['final_lru_sha256'],
                'final_cache_bytes_equal': old['final_cache_bytes']==new['final_cache_bytes'],
                'cache_hit_miss_eviction_equal': all(old['counts'].get(k,0)==new['counts'].get(k,0)
                                                     for k in ('cache_hit','cache_miss','cache_eviction')),
                'all_probability_non_timing_stats_equal': all(old[k]==new[k] for k in (
                    'counts', 'blocks_per_call', 'unique_blocks', 'block_accesses',
                    'repeated_accesses', 'grid')),
            }
            probability.append(checks)
        report['probability'] = probability
        report['all_equal'] = report['all_equal'] and all(all(c.values()) for c in probability)
    path=args.directory/f'{args.before}_vs_{args.after}.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['all_equal']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
