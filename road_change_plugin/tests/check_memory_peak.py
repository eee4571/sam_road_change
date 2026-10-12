"""Isolated-process RSS/elapsed measurements for streaming temporal records."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def peak_rss():
    if sys.platform=='win32':
        class Info(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in
                ('peak','working','paged_peak','paged','nonpaged_peak','nonpaged','pagefile','pagefile_peak')]
        data=Info();data.cb=ctypes.sizeof(data)
        handle=ctypes.windll.kernel32.GetCurrentProcess
        handle.restype=ctypes.c_void_p
        read=ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info),ctypes.c_ulong]
        if not read(handle(),ctypes.byref(data),data.cb):raise ctypes.WinError()
        return data.peak
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rows',type=int)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.rows:
        from engine.disk_records import DiskRows
        start=time.perf_counter();before=peak_rss()
        rows=DiskRows(dict(road_id=str(i%1000),period=str(i//1000),width_m=6.,
                           geometry_wkb=b'x'*500) for i in range(args.rows))
        assert sum(r['width_m'] for r in rows)==args.rows*6.
        after=peak_rss();rows.close()
        print(json.dumps(dict(rows=args.rows,peak_rss=after,baseline_rss=before,seconds=time.perf_counter()-start)))
        return
    results=[json.loads(subprocess.check_output([sys.executable,__file__,'--rows',str(n)],text=True)) for n in (10000,200000)]
    # 20x record count must not introduce proportional resident storage.
    assert results[1]['peak_rss']-results[0]['peak_rss']<64*1024**2,results
    text=json.dumps(results,indent=2)
    if args.output:args.output.write_text(text,encoding='utf8')
    print(text)


if __name__=='__main__':main()
