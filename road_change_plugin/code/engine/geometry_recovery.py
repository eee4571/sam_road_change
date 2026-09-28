"""Feature-local geometry failures, distinct from I/O and environment failures."""
from contextlib import contextmanager
from contextvars import ContextVar
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
def geometry_item(stage, sources, on_error=None, status='skipped'):
    """A transaction boundary around a single geometry, never around file I/O."""
    try:yield
    except GEOMETRY_ERRORS as error:
        recovery = _current.get()
        if recovery is None:raise
        if on_error is not None:on_error()
        recovery.record(stage,sources,error,status,final_action='retain_valid_geometry' if status=='fallback' else None)


def current_recovery():
    return _current.get()


def geometry_union(items, stage='polygon_union', sources=()):
    """Normal bulk union; isolate only failing operands if GEOS rejects it."""
    from shapely import union_all
    from shapely.geometry import GeometryCollection
    items=list(items)
    try:return union_all(items)
    except GEOMETRY_ERRORS:
        if _current.get() is None:raise
    result=GeometryCollection()
    for i,item in enumerate(items):
        ids=sources[i] if sources else [i]
        with geometry_item(stage,ids):
            candidate=union_all([result,item])
            if not candidate.is_valid:raise ValueError('Union produced invalid geometry')
            result=candidate
    return result
