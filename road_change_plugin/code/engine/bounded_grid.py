"""Grid planning that includes rounding and extreme aspect ratios."""
import math


def bounded_grid_shape(span_x, span_y, resolution, max_cells, *, endpoints=False):
    if not all(math.isfinite(v) and v>0 for v in (span_x,span_y,resolution)) or max_cells<4:
        raise ValueError('Grid extent, resolution and cell budget must be finite and positive')
    extra=int(endpoints)
    def shape(step):
        return max(1,math.ceil(span_x/step))+extra,max(1,math.ceil(span_y/step))+extra
    step=max(resolution,math.sqrt(span_x)*math.sqrt(span_y)/math.sqrt(max_cells))
    width,height=shape(step)
    if width*height<=max_cells:return step,width,height
    # Area-only estimates fail for long thin grids and omit ceil/+1 cells.
    low=step;high=step
    while math.prod(shape(high))>max_cells:high*=2
    for _ in range(60):
        mid=(low+high)/2
        if math.prod(shape(mid))>max_cells:low=mid
        else:high=mid
    width,height=shape(high)
    return high,width,height
