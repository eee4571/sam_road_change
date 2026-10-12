"""Shared host resource policy for foreground work and optional overlap.

Admission is conservative, not a hard OS RSS limit. Unknown capacity disables
speculation; it never disables foreground work or catches an allocation error.
"""
import ctypes
import os
import sys
import sqlite3
import tempfile
import uuid
from pathlib import Path
from dataclasses import dataclass
from contextlib import closing

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class Resources:
    available_ram: int
    total_ram: int
    available_vram: int | None = None


def resources():
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        status = Status(); status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return Resources(0, 0)
        available, total = int(status.available), int(status.total)
    else:
        try:
            with open('/proc/meminfo') as stream:
                fields = {r.split(':')[0]: int(r.split()[1])*1024 for r in stream if ':' in r}
            available, total = fields.get('MemAvailable', 0), fields['MemTotal']
        except (OSError, ValueError, KeyError):
            return Resources(0, 0)
    gpu = None
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        gpu = int(torch.cuda.mem_get_info()[0])
    return Resources(available, total, gpu)


def configured_bytes(name, default):
    value = os.environ.get(name)
    if value is None:
        return int(default)
    result = int(float(value)*MIB)
    if result <= 0:
        raise ValueError(f'{name} must be a positive MiB value')
    return result


def overlap_allowed(ram_bytes=2*GIB, vram_bytes=0, *, snapshot=None):
    """Reserve room for BOTH the consumer and the next producer.

    Re-evaluated before each submission. A configured RAM budget is a ceiling,
    never evidence that those bytes are actually available on the host.
    """
    state = snapshot or resources()
    margin = max(512*MIB, state.total_ram//10)
    capacity = min(max(0, state.available_ram-margin),
                   configured_bytes('ROAD_PIPELINE_RAM_MB', max(0, state.total_ram*6//10)))
    if os.environ.get('ROAD_PIPELINE_PARALLEL', 'auto').lower() == 'off':
        return False
    if capacity < 2*int(ram_bytes):
        return False
    if vram_bytes:
        if state.available_vram is None:
            return False
        limit = configured_bytes('ROAD_PIPELINE_VRAM_MB', state.available_vram)
        if min(limit, state.available_vram) < int(vram_bytes)+256*MIB:
            return False
    return True


def inference_batch_size(requested, patch_size, *, snapshot=None):
    state = snapshot or resources()
    if state.available_vram is None:
        return int(requested)
    capacity=state.available_vram
    if 'ROAD_PIPELINE_VRAM_MB' in os.environ:
        torch=sys.modules.get('torch')
        resident=int(torch.cuda.memory_reserved()) if torch is not None and torch.cuda.is_available() else 0
        capacity=min(capacity,max(0,configured_bytes('ROAD_PIPELINE_VRAM_MB',capacity)-resident))
    # Includes encoder activations and decoder tensors; weights already resident.
    per_patch = 256*MIB*max(1., (patch_size/512.)**2)
    return max(1, min(int(requested), int(max(0, capacity-512*MIB)//per_patch)))


def worker_count(requested,per_worker=128*MIB):
    if os.environ.get('ROAD_PIPELINE_PARALLEL','auto').lower()=='off':return 1
    state=resources()
    capacity=min(max(0,state.available_ram-max(512*MIB,state.total_ram//10)),
                 configured_bytes('ROAD_PIPELINE_RAM_MB',state.total_ram*6//10))
    path=os.environ.get('ROAD_RESOURCE_LEDGER')
    if path:
        with closing(sqlite3.connect(path,timeout=30)) as db:
            used=db.execute('SELECT COALESCE(SUM(ram),0) FROM leases').fetchone()[0]
        capacity=max(0,capacity-used)
    return max(1,min(int(requested),int(capacity//per_worker)))


class ResourceSession:
    """One ledger inherited by all model subprocesses in a pipeline run."""
    def __enter__(self):
        self.previous=os.environ.get('ROAD_RESOURCE_LEDGER')
        self.temp=None
        if self.previous:return self
        self.temp=tempfile.TemporaryDirectory(prefix='road-resources-')
        path=str(Path(self.temp.name)/'budget.sqlite')
        state=resources()
        capacity=min(configured_bytes('ROAD_PIPELINE_RAM_MB',state.total_ram*6//10),
                     max(0,state.available_ram-max(512*MIB,state.total_ram//10)))
        with closing(sqlite3.connect(path)) as db:
            db.execute('CREATE TABLE capacity (ram INTEGER)')
            db.execute('INSERT INTO capacity VALUES (?)',(capacity,))
            db.execute('CREATE TABLE leases (token TEXT PRIMARY KEY, ram INTEGER)')
            db.commit()
        os.environ['ROAD_RESOURCE_LEDGER']=path
        return self

    def __exit__(self,*error):
        if self.temp is not None:
            os.environ.pop('ROAD_RESOURCE_LEDGER',None)
            self.temp.cleanup()


class Lease:
    def __init__(self,path,token):self.path,self.token=path,token
    def close(self):
        if self.path is not None:
            with closing(sqlite3.connect(self.path,timeout=30)) as db:
                db.execute('DELETE FROM leases WHERE token=?',(self.token,))
                db.commit()
            self.path=None


def try_speculation(ram_bytes=2*GIB):
    """Non-blocking: reserve speculative bytes and equal foreground headroom.

    BEGIN IMMEDIATE makes concurrent parent/worker admission atomic. Failure
    to admit means synchronous execution, never loss or rerun of a task.
    """
    if not overlap_allowed(ram_bytes):return None
    path=os.environ.get('ROAD_RESOURCE_LEDGER')
    if not path:return Lease(None,None)
    token=uuid.uuid4().hex
    with closing(sqlite3.connect(path,timeout=30)) as db:
        db.execute('BEGIN IMMEDIATE')
        capacity=db.execute('SELECT ram FROM capacity').fetchone()[0]
        used=db.execute('SELECT COALESCE(SUM(ram),0) FROM leases').fetchone()[0]
        if used+2*ram_bytes>capacity:return None
        db.execute('INSERT INTO leases VALUES(?,?)',(token,int(ram_bytes)))
        db.commit()
    return Lease(path,token)
