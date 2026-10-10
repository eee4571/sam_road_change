"""Deterministic, geometry-only scene sampling; no image/model/width computation."""
import numpy as np
from shapely import union_all
from shapely.geometry import Point, box
from shapely.ops import polygonize
from shapely.strtree import STRtree

from graph import build, candidates
from engine.canonical_road_surface import _adjacency
from local_graph_repair import seed_chains
from medial_repair import direction


def select_regions(frame, config, size=600., count=9, coverage_image=None):
    edges = build(frame)
    nodes = _adjacency(edges, set(range(len(edges))))
    bounds = box(*frame.total_bounds).buffer(1)
    proposals = candidates(edges, bounds, config)
    events = {kind: [c.axis.interpolate(.5, normalized=True) for c in proposals if c.kind == kind]
              for kind in ('spur', 'gap', 'duplicate')}
    events['tips'] = [Point(n[1:]) for n, ends in nodes.items() if len(ends) == 1]
    events['junctions'] = [Point(n[1:]) for n, ends in nodes.items() if len(ends) >= 4]
    events['loops'] = [p.centroid for level in {e.level for e in edges}
                       for p in polygonize(union_all([e.axis for e in edges if e.level == level]))
                       if p.length <= 100]
    # Widely separated parallel chains are boundary-axis *candidates*, not labels.
    chains = [e for e in seed_chains(edges) if e.axis.length >= 60 and not e.axis.is_ring]
    tree = STRtree([e.axis for e in chains])
    events['boundary_axes'] = []
    for i, edge in enumerate(chains):
        for j in tree.query(edge.axis, predicate='dwithin', distance=40):
            other = chains[j]
            if j <= i or edge.level != other.level or abs(direction(edge.axis)@direction(other.axis)) < .96:
                continue
            separation = edge.axis.distance(other.axis)
            overlap = edge.axis.intersection(other.axis.buffer(40)).length
            if 6 < separation < 40 and overlap >= 50:
                events['boundary_axes'].append(edge.axis.interpolate(.5, normalized=True))
    trees = {k: STRtree(v) for k, v in events.items()}
    line_tree = STRtree(list(frame.geometry))
    x0, y0, x1, y1 = frame.total_bounds
    rows = []
    for y in np.arange(y0, y1-size*.35, size*.5):
        for x in np.arange(x0, x1-size*.35, size*.5):
            roi = box(x, y, x+size, y+size)
            ids = line_tree.query(roi, predicate='intersects')
            local = frame.iloc[ids]
            length = float(local.geometry.intersection(roi).length.sum())
            if length < 1200 or len(ids) < 12:
                continue
            scene = dict(valid_fraction=1., nonvegetated_fraction=0., texture=0.)
            if coverage_image is not None:
                xx, yy = np.meshgrid(np.linspace(x+5, x+size-5, 30), np.linspace(y+5, y+size-5, 30))
                probe = np.c_[xx.ravel(), yy.ravel()]
                rgb = np.column_stack([coverage_image.sample(coverage_image.rgb[..., k], probe) for k in range(3)])
                valid = np.isfinite(rgb).all(axis=1) & (np.nan_to_num(rgb).sum(axis=1) > 0)
                if valid.mean() < .98:
                    continue
                red, green, blue = rgb[valid].T
                scene = dict(valid_fraction=float(valid.mean()),
                    nonvegetated_fraction=float(np.mean(~((green > red*1.08) & (green > blue*1.08)))),
                    texture=float(np.nanmedian(abs(np.diff(rgb.reshape(30, 30, 3), axis=0)))))
            metrics = {k: int(len(t.query(roi, predicate='intersects'))) for k, t in trees.items()}
            metrics.update(length_m=length, feature_count=len(ids),
                           short_features=int((local.length < 35).sum()),
                           broad_length_m=float(local[local.width_m >= 14].geometry.intersection(roi).length.sum()))
            rows.append(dict(bbox_m=list(roi.bounds), metrics=metrics, scene_proxy=scene))
    if len(rows) < count:
        raise ValueError('Too few populated windows for requested scene sample')
    keys = list(events)+['short_features', 'broad_length_m', 'length_m']
    values = {key: np.array([r['metrics'][key] for r in rows]) for key in keys}
    ranks = {k: np.searchsorted(np.sort(v), v, side='right')/len(v) for k, v in values.items()}
    nonveg = np.array([r['scene_proxy']['nonvegetated_fraction'] for r in rows])
    nonveg_rank = np.searchsorted(np.sort(nonveg), nonveg, side='right')/len(nonveg)
    anomaly = sum(ranks[k] for k in ('spur', 'loops', 'duplicate', 'tips', 'junctions'))/5
    scores = {
        'wide_boundary': ranks['boundary_axes']*.65+ranks['broad_length_m']*.35,
        'internal_dense': ranks['short_features']*.5+ranks['spur']*.3+ranks['length_m']*.2,
        'small_loops': ranks['loops'],
        'parallel': ranks['duplicate']*.7+ranks['boundary_axes']*.3,
        'gaps': ranks['gap']*.6+ranks['tips']*.4,
        'junctions': ranks['junctions'],
        'building_proxy': ranks['short_features']*.25+ranks['length_m']*.25+nonveg_rank*.5,
        'mixed_roads': ranks['boundary_axes']*.5+(1-ranks['length_m'])*.5,
        'control': 1-anomaly,
    }
    descriptions = {
        'wide_boundary': '宽路缓存尺度与相邻平行链密集；检验偏轴/双侧伪轴',
        'internal_dense': '短道路、短支路及网络密度较高；检验内部道路',
        'small_loops': '闭合小面密集；小环仅为候选，不预设为错误',
        'parallel': '几何重复候选及平行链密集；检验真平行保护',
        'gaps': '断点对与真实 degree-1 端点密集',
        'junctions': '多路口及 degree>=4 节点密集',
        'building_proxy': '短线+密集网格作为建筑密集代理；需 RGB 目视核验场景',
        'mixed_roads': '道路密度较低且有宽间距平行候选；补充混合道路形态',
        'control': '有效覆盖>=98%，道路长度>=3500m、非植被代理>=75%，综合异常排名较低的建筑道路场景对照',
    }
    selected = []
    categories = list(scores)[:-1][:count-2]+['control', 'control']
    for category in categories:
        order = np.argsort(-scores[category], kind='stable')
        if category == 'control':
            # Comparable populated built-road scenes, not sparsely mapped fields
            # or watercourses whose low anomaly counts are caused by missing roads.
            order = [i for i in order if rows[i]['metrics']['length_m'] >= 3500
                     and rows[i]['scene_proxy']['nonvegetated_fraction'] >= .75]
        index = next((int(i) for i in order if all(
            box(*rows[i]['bbox_m']).intersection(box(*s['bbox_m'])).area/size**2 <= .15
            for s in selected)), None)
        if index is None:
            raise ValueError('Cannot obtain spatially diverse ROIs; decrease size/count')
        row = dict(rows[index], id=f'roi_{len(selected)+1:02d}', category=category,
                   reason=descriptions[category], selection_score=float(scores[category][index]),
                   anomaly_percentile_mean=float(anomaly[index]), control=category == 'control')
        selected.append(row)
    return selected, dict(window_size_m=size, stride_m=size*.5, scanned_populated_windows=len(rows),
        maximum_pairwise_overlap_fraction=.15, selection_uses_coarse_rgb_coverage=coverage_image is not None,
        building_density_is_geometry_proxy=True, windows=rows)
