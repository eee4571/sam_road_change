import numpy as np
from shapely.geometry import LineString
from .raw_boundary_core import runs

def rows_for_road(road_id, data):
    result = []
    for i, (s, xy, normal) in enumerate(zip(data['s'], data['xy'], data['normal'])):
        row = dict(road_id=road_id, sample_id=i, s_m=s, center_x=xy[0], center_y=xy[1],
                   normal_x=normal[0], normal_y=normal[1], flags=data['flags'][i],
                   accepted=not data['flags'][i])
        for method in ('baseline', 'optimized'):
            left, right = data[method][i]
            lxy, rxy = xy+normal*left, xy-normal*right
            for key, value in dict(left_distance=left, right_distance=right, width=left+right,
                left_x=lxy[0], left_y=lxy[1], right_x=rxy[0], right_y=rxy[1],
                left_confidence=data[method+'_confidence'][i, 0], right_confidence=data[method+'_confidence'][i, 1],
                confidence=data[method+'_confidence'][i].min(),
                left_response=data[method+'_response'][i, 0], right_response=data[method+'_response'][i, 1]).items():
                row[method+'_'+key] = value
        result.append(row)
    return result

def flag_crossings(data):
    # Positive signed distances guarantee section order; guard curve-induced spatial crossings too.
    width = data['optimized']
    left = data['xy']+data['normal']*width[:, :1]
    right = data['xy']-data['normal']*width[:, 1:]
    good = np.array([not f for f in data['flags']])
    for a, b in runs(good):
        if b-a < 2:
            continue
        l, r = LineString(left[a:b]), LineString(right[a:b])
        if l.intersects(r) or not l.is_simple or not r.is_simple:
            for i in range(a, b):
                data['flags'][i] = 'curve_boundary_crossing'
                data['optimized'][i] = data['baseline'][i]
                data['optimized_confidence'][i] = data['baseline_confidence'][i]
                data['optimized_response'][i] = data['baseline_response'][i]
