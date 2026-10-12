"""Feature-local geometry failures, distinct from I/O and environment failures."""
from contextlib import contextmanager
from contextvars import ContextVar
import traceback
from shapely.errors import GEOSException

# Do not swallow OSError, MemoryError, ImportError or arbitrary RuntimeError.
GEOMETRY_ERRORS = (GEOSException, ValueError, ArithmeticError)
_current = ContextVar('road_geometry_recovery', default=None)


def attempted_methods(error):
    methods=[]
    def visit(value):
        if isinstance(value,dict):
            if value.get('method') and value['method'] not in methods:methods.append(value['method'])
            for key in ('attempts','repairs','passes','parts','repair_passes'):
                visit(value.get(key))
        elif isinstance(value,(list,tuple)):
            for item in value:visit(item)
    visit(getattr(error,'diagnostic',{}))
    return methods


class Recovery:
    def __init__(self):
        self.failures = []

    def record(self, stage, sources, error, status='skipped', methods=(), final_action=None):
        if not methods:
            methods=attempted_methods(error) or [stage]
        row = dict(stage=stage, source_ids=list(sources), exception_type=type(error).__name__,
                   traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)),
                   reason=str(error), attempted_methods=list(methods), repair_status=status,
                   axis_quality='low', final_action=final_action or ('conservative_local_rebuild' if status=='fallback' else 'skip_local_geometry'))
        self.failures.append(row)
        action = '已使用保守重建' if status == 'fallback' else '已跳过该局部'
        if final_action=='retain_valid_geometry':action='已保留该局部有效几何'
        print(f'[WARN] 道路 {",".join(map(str,sources)) or "unknown"} {stage}: {error}，{action}',flush=True)
        return row

    def summary(self, repaired=0):
        return dict(repaired_count=repaired,
                    fallback_count=sum(r['repair_status']=='fallback' for r in self.failures),
                    skipped_count=sum(r['repair_status']=='skipped' for r in self.failures),
                    warning_count=len(self.failures))


@contextmanager
def recovery_scope(recovery):
    token = _current.set(recovery)
    try:yield recovery
    finally:_current.reset(token)


@contextmanager
def geometry_item(stage, sources, on_error=None, status='skipped', error_types=GEOMETRY_ERRORS,
                  road_local=False, tile_id=''):
    """A transaction boundary around a single geometry, never around file I/O."""
    try:yield
    except (Exception if road_local else error_types) as error:
        if road_local:
            from .width.width_reliability import require_road_local
            require_road_local(error)
        if getattr(error,'code',None)==-4 or any(token in str(error).lower() for token in
            ('out of memory','failed to allocate','bad_alloc','insufficient memory')):
            raise
        recovery = _current.get()
        if recovery is None:raise
        if on_error is not None:on_error()
        record=recovery.record(stage,sources,error,status,final_action='retain_valid_geometry' if status=='fallback' else None)
        if road_local:
            record.update(road_id=','.join(map(str,sources)),tile_id=tile_id,fallback_reason=str(error))


def current_recovery():
    return _current.get()


def geometry_union(items, stage='polygon_union', sources=()):
    """Normal bulk union; isolate only failing operands if GEOS rejects it."""
    from shapely import union_all
    from shapely.geometry import GeometryCollection
    from .disk_records import batches
    # Bound the GEOS operand fan-in. The final geometry is still exact and
    # necessarily proportional to output complexity; no vertex simplification.
    items=iter(items);first=list(next(iter(batches(items,257)),[]))
    if len(first)>256:
        from itertools import chain
        groups=[]
        offset=0
        for group in batches(chain(first,items),256):
            ids=sources[offset:offset+len(group)] if sources else [[i] for i in range(offset,offset+len(group))]
            groups.append(geometry_union(group,stage,ids));offset+=len(group)
        while len(groups)>256:
            groups=[geometry_union(group,stage) for group in batches(groups,256)]
        return union_all(groups)
    items=first
    try:return union_all(items)
    except GEOMETRY_ERRORS as error:
        from .width.width_reliability import require_road_local
        require_road_local(error)
        if _current.get() is None:raise
    result=GeometryCollection()
    for i,item in enumerate(items):
        ids=sources[i] if sources else [i]
        with geometry_item(stage,ids):
            candidate=union_all([result,item])
            if not candidate.is_valid:raise ValueError('Union produced invalid geometry')
            result=candidate
    return result
