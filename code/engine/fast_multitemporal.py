"""Road-only temporal evidence and robust width calibration. No truth inputs."""
from collections import Counter
import time
import numpy as np
from shapely import from_wkt, union_all, prepare
from shapely.ops import substring
from .auto_presence_candidates import LongitudinalCoverage

AUTO_REVISION = 'fast2_paired_patch_precision_v1'


def dependency_periods(names,before,after):
    """Immediate planned neighbors only; never jump over a missing period."""
    result={before,after}
    for period,offset in ((before,-1),(after,1)):
        if period in names:
            index=names.index(period)+offset
            if 0<=index<len(names):result.add(names[index])
    return result


def neighbor_results(entries,grid,before,after,planned=None):
    selected={str(r['period']):r for r in entries if str(r.get('grid'))==str(grid)}
    names=list(planned) if planned is not None else sorted(selected)
    result={}
    for side,period,offset in [('previous',before,-1),('next',after,1)]:
        if period not in names:continue
        index=names.index(period)+offset
        if 0<=index<len(names) and names[index] in selected:
            row=selected[names[index]]
            result[side]=row.get('result') or row
    return result


class RoadContext:
    def __init__(self,lines,valid,tolerance):
        self.coverage=LongitudinalCoverage(lines,tolerance)
        self.valid=valid;prepare(self.valid)

    def intervals(self,axis,footprint):
        if not self.valid.covers(footprint):return None
        return self.coverage.uncovered(axis)


def load_contexts(results,crs,tolerance):
    from .fast_pipeline import _load_fast_period_result,_read_fast_change_layer
    from .auto_presence_candidates import line_parts
    contexts={}
    for side,result in (results or {}).items():
        if side not in ('previous','next'):continue
        p=_load_fast_period_result(result)
        centers=_read_fast_change_layer(p,'centerlines').to_crs(crs)
        valid=_read_fast_change_layer(p,'valid_observation').to_crs(crs)
        contexts[side]=RoadContext([line for g in centers.geometry for line in line_parts(g)],union_all(valid.geometry),tolerance)
    return contexts


def reconcile_temporal(records,contexts,minimum_length=32.):
    tick=time.perf_counter();counts=Counter(v2_temporal_added_suppressed=0,v2_temporal_removed_suppressed=0,
        v2_temporal_suppressed_length_m=0.,v2_temporal_confirmed=0,v2_temporal_unknown=0)
    children=[]
    for index,row in enumerate(records):
        kind=row['change_typ']
        if kind not in ('added','removed') or not row.get('v2_publish',False):continue
        axis=from_wkt(row['axis_wkt']);length=axis.length
        spans={side:context.intervals(axis,row['geometry']) for side,context in contexts.items()}
        cuts=sorted({0.,length}|{p for intervals in spans.values() if intervals is not None for interval in intervals for p in interval})
        parts=[]
        for a,b in zip(cuts,cuts[1:]):
            if b-a<1e-6:continue
            states={side:('unknown' if values is None else 'absent' if any(x<=(a+b)/2<=y for x,y in values) else 'present')
                    for side,values in spans.items()}
            previous=states.get('previous','unknown');following=states.get('next','unknown')
            unstable=(previous=='present' or following=='absent') if kind=='added' else (previous=='absent' or following=='present')
            reason=('transient_extraction_or_gap' if unstable else 'persistent_change' if
                    ((following=='present' or previous=='absent') if kind=='added' else (following=='absent' or previous=='present')) else 'temporal_context_unknown')
            if parts and parts[-1][2:]==(unstable,reason):parts[-1]=(parts[-1][0],b,unstable,reason)
            else:parts.append((a,b,unstable,reason))
        removed=sum(b-a for a,b,unstable,_ in parts if unstable)
        if removed<=1e-6:
            reason=parts[0][3] if parts else 'temporal_context_unknown'
            row['v2_temporal_state']=reason
            counts['v2_temporal_confirmed' if reason=='persistent_change' else 'v2_temporal_unknown']+=1
            if reason=='persistent_change':row['confidence']=max(row.get('confidence',0.),.95)
            continue
        counts[f'v2_temporal_{kind}_suppressed']+=1;counts['v2_temporal_suppressed_length_m']+=removed
        row.update(v2_publish=False,v2_precision_reason='temporal_transient',v2_temporal_state='transient_extraction_or_gap',
                   v2_temporal_removed_length_m=removed)
        for a,b,unstable,reason in parts:
            if unstable:continue
            local=substring(axis,a,b);width=max(row['width_bef'],row['width_aft'])
            child=dict(row,axis_wkt=local.wkt,geometry=local.buffer(width/2,cap_style='flat'),length_m=b-a,
                start_m=row['start_m']+a,end_m=row['start_m']+b,v2_publish=True,v2_temporal_parent=index,
                v2_temporal_removed_length_m=0.,v2_temporal_state=reason,v2_precision_reason='temporal_persistent_interval')
            # Preserve the existing existence length requirement after partition.
            if b-a<minimum_length:child.update(v2_publish=False,v2_precision_reason='short_temporal_interval')
            children.append(child)
    records.extend(children)
    counts['timing_v2_multitemporal_seconds']=time.perf_counter()-tick
    return counts


def weighted_median(values,weights):
    order=np.argsort(values,kind='stable');w=weights[order]
    return float(values[order[np.searchsorted(np.cumsum(w),w.sum()/2)]])


def estimate_width_bias(samples,minimum=30):
    """One vote per reliable road; majority-stable assumption is explicit."""
    values=np.asarray(samples,dtype=float);values=values[np.isfinite(values)]
    bias=float(np.median(values)) if len(values) else 0.
    scatter=float(1.4826*np.median(np.abs(values-bias))) if len(values) else 0.
    reliable=len(values)>=minimum
    return dict(bias=bias if reliable else 0.,scatter=scatter if reliable else 0.,count=len(values),
                reliable=reliable,estimated_bias=bias,estimated_scatter=scatter)
