"""Scratch records and bounded frame reuse for large period/temporal products.

Pickles are created and consumed only inside a private temporary directory;
never open a user-supplied database. Published GIS files remain the contract.
"""
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import ExitStack,contextmanager
from contextvars import ContextVar
from functools import wraps
from itertools import islice
import pickle
import sqlite3
import tempfile
from pathlib import Path

_scope = ContextVar('record_scratch', default=None)


def record_scope(function):
    @wraps(function)
    def run(*args, **kwargs):
        with ExitStack() as stack:
            token = _scope.set(stack)
            try:
                return function(*args, **kwargs)
            finally:
                _scope.reset(token)
    return run


def batches(rows, size=2048):
    iterator = iter(rows)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


@contextmanager
def scratch_array(shape,dtype,ram_limit=16*1024*1024):
    """Exact dynamic-programming backpointers; large histories stay on disk."""
    import numpy as np
    import math
    if math.prod(shape)*np.dtype(dtype).itemsize<=ram_limit:
        yield np.empty(shape,dtype=dtype)
        return
    with tempfile.TemporaryDirectory(prefix='road-array-') as temp:
        array=np.memmap(Path(temp)/'array.bin',mode='w+',shape=shape,dtype=dtype)
        try:yield array
        finally:array._mmap.close()


class Row(dict):
    def __init__(self,value,store,index):
        super().__init__(value);self.store=store;self.index=index
    def __setitem__(self,key,value):
        super().__setitem__(key,value);self.store[self.index]=self
    def update(self,*args,**kwargs):
        super().update(*args,**kwargs);self.store[self.index]=self


class DiskRows:
    def __init__(self, rows=(), *, mutable=False):
        self.temp = tempfile.TemporaryDirectory(prefix='road-records-')
        self.db = sqlite3.connect(Path(self.temp.name)/'rows.sqlite')
        self.db.execute('PRAGMA cache_size=-4096')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('CREATE TABLE rows (id INTEGER PRIMARY KEY, road TEXT, period TEXT, value BLOB)')
        self.db.execute('CREATE INDEX road_period ON rows(road,period)')
        self.db.execute('CREATE TABLE keys (kind TEXT, key BLOB, rowid INTEGER, PRIMARY KEY(kind,key))')
        self.count = 0
        self.mutable=mutable
        scope = _scope.get()
        if scope is not None:scope.callback(self.close)
        self.extend(rows)

    def append(self, row):
        self.db.execute('INSERT INTO rows VALUES(?,?,?,?)', (self.count, str(row.get('road_id','')),
            str(row.get('period','')), pickle.dumps(dict(row), protocol=5)))
        if row.get('event_id'):
            self.db.execute('INSERT OR REPLACE INTO keys VALUES(?,?,?)',
                ('__event_id__',pickle.dumps(row['event_id']),self.count))
        self.count += 1

    def extend(self, rows):
        for batch in batches(rows):
            for row in batch:self.append(row)
            self.db.commit()

    def __len__(self):return self.count

    def __iter__(self):
        for index,value in self.db.execute('SELECT id,value FROM rows ORDER BY id'):
            value=pickle.loads(value)
            yield Row(value,self,index) if self.mutable else value

    def __getitem__(self, index):
        if isinstance(index, tuple):
            value=self.get(index)
            if value is None:raise KeyError(index)
            return value
        if isinstance(index, slice):
            start, stop, step = index.indices(self.count)
            return [pickle.loads(value) for value, in self.db.execute(
                'SELECT value FROM rows WHERE id>=? AND id<? ORDER BY id', (start, stop))][::step]
        if index < 0:index += self.count
        row = self.db.execute('SELECT value FROM rows WHERE id=?', (index,)).fetchone()
        if row is None:raise IndexError(index)
        value=pickle.loads(row[0])
        return Row(value,self,index) if self.mutable else value

    def __setitem__(self, index, row):
        self.db.execute('UPDATE rows SET road=?,period=?,value=? WHERE id=?',
            (str(row.get('road_id','')),str(row.get('period','')),pickle.dumps(dict(row),protocol=5),index))

    def __delitem__(self, part):
        if not isinstance(part,slice) or part.stop is not None or part.step is not None:
            raise ValueError('Only tail rollback is supported')
        start=part.start or 0
        self.db.execute('DELETE FROM rows WHERE id>=?',(start,));self.count=start

    def road_ids(self):
        return (r[0] for r in self.db.execute('SELECT DISTINCT road FROM rows ORDER BY road'))

    def road_rows(self, road):
        return [(i,pickle.loads(v)) for i,v in self.db.execute('SELECT id,value FROM rows WHERE road=? ORDER BY id',(road,))]

    def validate_observations(self, road_ids, periods):
        if len(self)!=len(road_ids)*len(periods):return False
        if set(self.road_ids())!=set(road_ids):return False
        expected=set(periods)
        for road in self.road_ids():
            actual=[p for p, in self.db.execute('SELECT period FROM rows WHERE road=?',(road,))]
            if len(actual)!=len(expected) or set(actual)!=expected:return False
        return True

    def get(self, key, default=None):
        row=self.db.execute('SELECT value FROM rows WHERE road=? AND period=? ORDER BY id DESC LIMIT 1',tuple(key)).fetchone()
        return default if row is None else pickle.loads(row[0])

    def index(self,fields):
        return RowIndex(self,fields)

    def sorted(self,key,reverse=False):
        self.db.execute('CREATE TEMP TABLE IF NOT EXISTS ordering (id INTEGER,value REAL)')
        self.db.execute('DELETE FROM ordering')
        for index,row in enumerate(self):
            self.db.execute('INSERT INTO ordering VALUES(?,?)',(index,float(key(row))))
        direction='DESC' if reverse else 'ASC'
        for index, in self.db.execute(f'SELECT id FROM ordering ORDER BY value {direction},id'):
            yield self[index]

    def close(self):
        if self.db is not None:self.db.close();self.db=None
        self.temp.cleanup()

    def __del__(self):
        if hasattr(self,'db') and self.db is not None:self.close()


class RowIndex:
    def __init__(self,rows,fields):
        self.rows=rows;self.kind='|'.join(fields)
        rows.db.execute('DELETE FROM keys WHERE kind=?',(self.kind,))
        for index,row in enumerate(rows):
            key=tuple(row.get(f) for f in fields)
            rows.db.execute('INSERT OR REPLACE INTO keys VALUES(?,?,?)',(self.kind,pickle.dumps(key),index))
    def get(self,key,default=None):
        found=self.rows.db.execute('SELECT rowid FROM keys WHERE kind=? AND key=?',(self.kind,pickle.dumps(key))).fetchone()
        return default if found is None else self.rows[found[0]]
    def __contains__(self,key):return self.get(key) is not None
    def __setitem__(self,key,value):
        if isinstance(value,Row) and value.store is self.rows:index=value.index
        else:
            found=self.rows.db.execute('SELECT rowid FROM keys WHERE kind=? AND key=?',('__event_id__',pickle.dumps(value['event_id']))).fetchone()
            if found is None:raise KeyError('Event must be appended before indexing')
            index=found[0];self.rows[index]=value
        self.rows.db.execute('INSERT OR REPLACE INTO keys VALUES(?,?,?)',(self.kind,pickle.dumps(key),index))
    def pop(self,key,default=None):
        value=self.get(key,default)
        self.rows.db.execute('DELETE FROM keys WHERE kind=? AND key=?',(self.kind,pickle.dumps(key)))
        return value


class DiskMap(Mapping):
    def __init__(self):
        self.rows=DiskRows()
        self.rows.db.execute('CREATE TABLE mapping (key BLOB PRIMARY KEY,value BLOB)')
    def __getitem__(self,key):
        found=self.rows.db.execute('SELECT value FROM mapping WHERE key=?',(pickle.dumps(key),)).fetchone()
        if found is None:raise KeyError(key)
        return pickle.loads(found[0])
    def __setitem__(self,key,value):
        self.rows.db.execute('INSERT OR REPLACE INTO mapping VALUES(?,?)',(pickle.dumps(key),pickle.dumps(value)))
    def __iter__(self):return (pickle.loads(k) for k, in self.rows.db.execute('SELECT key FROM mapping'))
    def __len__(self):return self.rows.db.execute('SELECT COUNT(*) FROM mapping').fetchone()[0]


def observation_index(rows):
    return rows if isinstance(rows, DiskRows) else DiskRows(rows)


class LazyFrames(Mapping):
    """Retain at most two period frames, without changing iteration order."""
    def __init__(self, keys, loader, capacity=2):
        self.keys_list=list(keys);self.loader=loader;self.capacity=capacity;self.cache=OrderedDict()
    def __iter__(self):return iter(self.keys_list)
    def __len__(self):return len(self.keys_list)
    def __getitem__(self,key):
        if key not in self.keys_list:raise KeyError(key)
        if key not in self.cache:
            while len(self.cache)>=self.capacity:self.cache.popitem(last=False)
            self.cache[key]=self.loader(key)
        self.cache.move_to_end(key)
        return self.cache[key]


class FrameSpool:
    """Mutable pair frames: retain one frame and spill it before switching.

    Callers finish mutations before moving to another pair. Published files
    are unaffected; all scratch writes and reads preserve pandas dtypes.
    """
    def __init__(self):
        self.temp=tempfile.TemporaryDirectory(prefix='road-pairs-')
        self.key=None;self.frame=None
        scope=_scope.get()
        if scope is not None:scope.callback(self.close)
    def _path(self,key):return Path(self.temp.name)/f'{key}.pkl'
    def _flush(self):
        if self.key is not None:
            with self._path(self.key).open('wb') as stream:pickle.dump(self.frame,stream,protocol=5)
    def get(self,key):
        if key!=self.key:
            self._flush();self.frame=None
            with self._path(key).open('rb') as stream:self.frame=pickle.load(stream)
            self.key=key
        return self.frame
    def set(self,key,frame):
        if key!=self.key:self._flush()
        self.key,self.frame=key,frame
    def close(self):
        self.key=self.frame=None;self.temp.cleanup()


class FramePair(dict):
    def __init__(self,spool,key,frame,**metadata):
        super().__init__(metadata);self.spool=spool;self.key=key
        spool.set(key,frame)
    def __getitem__(self,key):return self.spool.get(self.key) if key=='frame' else super().__getitem__(key)
    def __setitem__(self,key,value):
        if key=='frame':self.spool.set(self.key,value)
        else:super().__setitem__(key,value)


def write_gpkg_rows(path, layer, rows, crs):
    """Two streaming passes: stable scalar schema, then bounded feature batches."""
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    import pyogrio
    kinds={};counts={}
    for row in rows:
        for key,value in row.items():
            if key=='geometry':continue
            counts[key]=counts.get(key,0)+1
            kind='str' if isinstance(value,str) else 'int' if isinstance(value,(int,np.integer,bool)) else 'float'
            prior=kinds.get(key,kind)
            kinds[key]='str' if 'str' in (prior,kind) else 'float' if 'float' in (prior,kind) else 'int'
    for key in kinds:
        if counts[key]<len(rows) and kinds[key]=='int':kinds[key]='float'
    first=True
    for batch in batches(rows):
        frame=gpd.GeoDataFrame(batch,geometry='geometry',crs=crs)
        for key,kind in kinds.items():
            if key not in frame:frame[key]=None if kind=='str' else np.nan
            if kind!='str':frame[key]=pd.to_numeric(frame[key],errors='raise').astype('int64' if kind=='int' else 'float64')
        frame=frame[[*kinds,'geometry']]
        pyogrio.write_dataframe(frame,path,layer=layer,driver='GPKG',append=not first,geometry_type='Unknown')
        first=False
    if first:
        gpd.GeoDataFrame(geometry=[],crs=crs).to_file(path,layer=layer,driver='GPKG')


def write_csv_rows(path,rows,encoding='utf8'):
    import pandas as pd
    fields={}
    for row in rows:fields.update(dict.fromkeys(row))
    with open(path,'w',encoding=encoding,newline='') as stream:
        first=True
        for batch in batches(rows):
            pd.DataFrame(batch).reindex(columns=list(fields)).to_csv(stream,index=False,header=first)
            first=False
        if first:pd.DataFrame().to_csv(stream,index=False)
