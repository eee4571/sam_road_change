"""Do not substitute coarser or fabricated accuracy metrics for large grids."""
from functools import wraps


def intersection_groups(predicted,truth):
    """Exact bipartite components with O(P+T) memory, not O(P*T) edges."""
    from shapely import STRtree
    offset=len(predicted);parent=list(range(offset+len(truth)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    tree=STRtree(truth)
    touched=set()
    for p,geometry in enumerate(predicted):
        for raw in tree.query(geometry,predicate='intersects'):
            t=offset+int(raw);a,b=root(p),root(t)
            if a!=b:parent[b]=a
            touched.update((p,t))
    groups={}
    for i in sorted(touched):
        group=groups.setdefault(root(i),(set(),set()))
        group[i>=offset].add(i-offset if i>=offset else i)
    return sorted(groups.values(),key=lambda pair:min(pair[1]))


class EvaluationGridLimit(ValueError):
    pass


def check_grid(shape, maximum=8_000_000):
    if int(shape[0])*int(shape[1]) > maximum:
        raise EvaluationGridLimit(f'exact_evaluation_grid_exceeds_budget: {shape[0]}x{shape[1]} > {maximum} cells')


def bounded_metric(function):
    @wraps(function)
    def evaluate(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except EvaluationGridLimit as error:
            print(f'[WARN] 精确骨架评价未计算（内存预算）：{error}', flush=True)
            # No partial sum, fabricated zero, or coarsened-grid score may leak
            # into the final accuracy report. Other area/object metrics survive.
            return dict(centerline_offset_status='unavailable_resource_limit',
                centerline_offset_reason=str(error), centerline_avg_offset_m=None,
                truth_to_pred_avg_m=None, pred_to_truth_avg_m=None,
                truth_axis_length_m=None, predicted_axis_length_m=None,
                truth_distance_integral_m2=None, predicted_distance_integral_m2=None,
                road_centerline_completeness=None, centerline_mean_offset_px=None,
                truth_centerline_length_px=None, predicted_centerline_length_px=None,
                covered_truth_centerline_length_px=None, centerline_offset_integral_px2=None)
    return evaluate


def greedy_pairs(candidates):
    """External sort preserves the original descending IoU/index tie-breaks."""
    import sqlite3
    import tempfile
    from contextlib import closing
    from pathlib import Path
    from .disk_records import batches
    with tempfile.TemporaryDirectory(prefix='road-matching-') as temp:
        with closing(sqlite3.connect(Path(temp)/'pairs.sqlite')) as db:
            db.execute('PRAGMA cache_size=-4096')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE candidates (iou REAL,p INTEGER,t INTEGER)')
            for batch in batches(candidates):
                db.executemany('INSERT INTO candidates VALUES(?,?,?)',batch)
                db.commit()
            predicted=set();truth=set();pairs=[]
            for p,t in db.execute('SELECT p,t FROM candidates ORDER BY iou DESC,p DESC,t DESC'):
                if p in predicted or t in truth:continue
                predicted.add(p);truth.add(t);pairs.append((p,t))
            return pairs
