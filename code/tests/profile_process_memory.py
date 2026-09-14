"""Optional Windows process-memory sampling for explicit cached-pair experiments."""
import ctypes
from ctypes import wintypes
import threading
import time


class ProcessMemory:
    def __init__(self):
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
        self.counter = Counters()
        self.counter.cb = ctypes.sizeof(self.counter)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.handle = kernel.GetCurrentProcess()
        self.read = kernel.K32GetProcessMemoryInfo
        self.read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        self.read.restype = wintypes.BOOL
        self.stop = threading.Event()
        self.result = {}

    def sample(self):
        if not self.read(self.handle, ctypes.byref(self.counter), self.counter.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        for name, value in [('rss', self.counter.WorkingSetSize), ('private', self.counter.PrivateUsage)]:
            self.result.setdefault(name+'_start_bytes', value)
            self.result[name+'_peak_bytes'] = max(self.result.get(name+'_peak_bytes', 0), value)
            self.result[name+'_end_bytes'] = value

    def __enter__(self):
        self.sample()
        self.started = time.perf_counter(), time.process_time(), time.thread_time()
        def monitor():
            while not self.stop.wait(.1):
                self.sample()
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        wall = time.perf_counter()-self.started[0]
        cpu = time.process_time()-self.started[1]
        self.result.update(process_cpu_seconds=cpu,
                           main_thread_cpu_seconds=time.thread_time()-self.started[2],
                           average_cpu_cores=cpu/max(wall, 1e-9),
                           monitored_wall_seconds=wall)
