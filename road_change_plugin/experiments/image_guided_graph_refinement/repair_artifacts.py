"""Auditable final vectors, topology diagnostics and RGB comparison sheets."""
from collections import Counter
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely import union_all, wkt, points, distance
from shapely.geometry import box

from evidence import ImageGrid, stations
from graph import build, stats
from medial_repair import line_parts
from report import background, draw, write_json


def compare(path, before, after, evidence, roi, title):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, frame, label in zip(axes, (before, after), ('Before', 'After')):
        background(ax, evidence)
        for line in frame[frame.intersects(roi)].geometry:
            draw(ax, line, color='#ffff33', linewidth=1.1)
        x0, y0, x1, y1 = roi.bounds
        ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect='equal', title=label)
        ax.ticklabel_format(style='plain', useOffset=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def geometry_products(output, before, after, audit):
    from engine.canonical_road_surface import _level
    old_levels, new_levels = before.apply(_level, axis=1), after.apply(_level, axis=1)
    layers = dict(added=[], deleted=[])
    distances, added_length, deleted_length = [], 0., 0.
    for level in sorted(set(old_levels) | set(new_levels)):
        old = union_all(before[old_levels == level].geometry)
        new = union_all(after[new_levels == level].geometry)
        deleted = old.difference(new.buffer(.01))
        added = new.difference(old.buffer(.01))
        added_length += added.length
        deleted_length += deleted.length
        for key, geometry in (('added', added), ('deleted', deleted)):
            layers[key].extend(dict(layer=level[0], bridge=level[1], tunnel=level[2], geometry=g) for g in line_parts(geometry))
        probes = [stations(line, 5)[1] for line in line_parts(deleted)]
        if probes and not new.is_empty:
            distances.extend(distance(points(np.concatenate(probes)), new).tolist())
    for name, key in (('replaced_old', 'old_wkt'), ('replaced_new', 'new_wkt')):
        layers[name] = [dict(edit_id=i, action=r['action'], reason=r.get('reason', ''), geometry=g)
                       for i, r in enumerate(audit) if r.get(key) and r.get('applied', True)
                       for g in line_parts(wkt.loads(r[key]))]
    for name, records in layers.items():
        if records:
            gpd.GeoDataFrame(records, geometry='geometry', crs=before.crs).to_file(
                output/'changes.gpkg', layer=name, driver='GPKG')
    displacement = np.array(distances) if distances else np.zeros(1)
    return dict(added_length_m=added_length, deleted_length_m=deleted_length,
                changed_old_to_final_nearest_distance_mean_m=float(displacement.mean()),
                changed_old_to_final_nearest_distance_max_m=float(displacement.max()),
                delta_tolerance_m=.01, displacement_is_nearest_network_distance_not_accuracy=True,
                actions=dict(Counter(r['action'] for r in audit if r.get('applied', True))),
                rejected_or_unmodified_actions=dict(Counter(r['action'] for r in audit if not r.get('applied', True))))


def topology(frame, roi, config):
    edges = build(frame[frame.intersects(roi.buffer(160))])
    result = stats(edges, roi, config)
    result['self_intersecting_features'] = int(sum(not line.is_simple for line in frame[frame.intersects(roi)].geometry))
    result['invalid_features'] = int(sum(not line.is_valid for line in frame[frame.intersects(roi)].geometry))
    result['zero_length_features'] = int(sum(line.length < .001 for line in frame[frame.intersects(roi)].geometry))
    return result


def final_products(output, before, after, inputs, config, audit, tiles):
    after.to_file(output/'final_centerlines.gpkg', layer='centerlines', driver='GPKG')
    before.to_file(output/'baseline_centerlines.gpkg', layer='centerlines', driver='GPKG')
    write_json(output/'audit.json', audit)
    summary = geometry_products(output, before, after, audit)
    extent = box(*before.total_bounds).buffer(1, cap_style='square')
    print('Whole-period topology and overview', flush=True)
    summary['before'] = topology(before, extent, config)
    summary['after'] = topology(after, extent, config)
    summary['tiles'] = tiles
    summary['accuracy_claim'] = 'Unlabelled experiment; topology and ribbon scores are not ground-truth precision/recall.'
    summary['coverage'] = 'All formal features exported; actual modified/scanned areas are listed per tile.'
    overview = SimpleNamespace(image=ImageGrid(inputs['images'], before.crs, extent, max(2., max(extent.bounds[2]-extent.bounds[0], extent.bounds[3]-extent.bounds[1])/2200)))
    compare(output/'whole_period_before_after.png', before, after, overview, extent, 'Full period: cached baseline / experimental result')
    write_json(output/'summary.json', summary)
    return summary


def write_report(output, summary):
    """Human-readable results alongside machine audit; no accuracy overclaim."""
    from data import read_json
    rows = [('component_count', '连通分量'), ('degree_1_endpoints', 'degree-1 端点'),
            ('spur_count', '短支杈候选'), ('spur_length_m', '短支杈候选总长 m'),
            ('near_duplicate_count', '近平行重复候选'), ('near_duplicate_length_m', '重复候选重叠长度 m'),
            ('suspicious_gap_pairs', '可疑断点对'), ('small_loop_count', '小环（周长≤60m）'),
            ('complex_junction_count', 'degree≥4 节点'), ('total_length_m', '道路总长 m'),
            ('self_intersecting_features', '自交要素'), ('invalid_features', '无效几何')]
    def table(before, after):
        text = '| 指标 | 修复前 | 修复后 |\n|---|---:|---:|\n'
        for key, label in rows:
            text += f'| {label} | {before[key]:.2f} | {after[key]:.2f} |\n'
        return text
    tile = summary['tiles'][0]
    applied = [i for i, t in enumerate(summary['tiles']) if t['accepted']]
    reverted = [i for i, t in enumerate(summary['tiles']) if not t['accepted']]
    text = '# 基于影像证据的道路图实际修复实验报告\n\n'
    text += '本次实际生成了完整期次的 `final_centerlines.gpkg`，没有接入正式 pipeline。示意图仅作为主路形态参考，没有将未画出的厂区道路当作负样本。\n\n'
    text += f'扫描 {len(summary["tiles"])} 个 ROI/分块，{len(applied)} 个通过保护检查；撤回的分块编号：{reverted}。撤回候选保留在审计中，`applied=false`，不计入最终修改。\n\n'
    text += '## 重点工业区\n\n'+table(tile['before'], tile['after'])
    text += '\n![重点 ROI 前后](tile_000/before_after.png)\n\n'
    text += '## 整期\n\n'+table(summary['before'], summary['after'])
    text += '\n![整期前后](whole_period_before_after.png)\n\n'
    text += '## RGB 是否提供了几何之外的信息\n\n'
    text += '东北宽路的旧侧轴和新中央轴用同一搜索尺度采样，SAM/MOLRA 只作为正证据，surface 不参与独立道路判断。下列分数是算法自身使用的特征，不是独立真值精度；分数提高不能替代人工标签或留出样本验证。\n\n'
    text += '| 重点 ROI 中央轴 | 旧轴 RGB 支持 | 新轴 RGB 支持 | SAM 正证据均值 |\n|---|---:|---:|---:|\n'
    for p in read_json(output/'tile_000/proposals.json'):
        e = p['evidence']
        sam = e['sam_positive_mean']
        sam_text = '缺失' if sam is None else f'{sam:.4f}'
        text += f'| {p["id"]} | {e["old_axis_raw_rgb_support"]:.3f} | {e["new_axis_raw_rgb_support"]:.3f} | {sam_text} |\n'
    text += '\n本次有效的信息增量主要是：原网络给出大致走向与范围，RGB 的连续内部区域和两侧反差给出中央位置；低 SAM 区域也能重建主轴。拓扑清理和平滑本身主要依赖几何约束，不能把其全部收益归于影像。\n\n'
    text += '## 实际修改与保护\n\n'
    text += f'独立 gap route 实际通过 {summary["actions"].get("connect", 0)} 条；断点指标下降还可能来自中央轴与路口重建，不能全部计为 gap 路由收益。\n\n'
    for name, count in summary['actions'].items():
        text += f'- `{name}`：{count} 条应用审计记录。\n'
    text += f'\n最终净新增线几何 {summary["added_length_m"]:.1f}m，净移除旧线几何 {summary["deleted_length_m"]:.1f}m（包含替换造成的几何差，不等于永久删路）。旧修改位置到最终同层网络的最近距离：均值 {summary["changed_old_to_final_nearest_distance_mean_m"]:.2f}m、最大 {summary["changed_old_to_final_nearest_distance_max_m"]:.2f}m；该距离不是位置精度。\n\n'
    text += '重建道路带与路口后保留外部支路接触点并重新挂接。相邻切口汇入公共连接段，避免重挂产生双轴。层级分别处理；gap 搜索在路径转移中约束两端方向，并检查穿越既有道路。局部清理重复检查邻域至无新修改或达到明确轮数上限。\n\n'
    text += '## 复用与输出\n\n'
    text += f'输入文件指纹运行前后相同：**{summary["input_fingerprints_unchanged"]}**。未重新运行模型、测宽或正式道路面计算；RGB 描述、图编辑与路由是本实验新增计算。\n\n'
    text += '- `final_centerlines.gpkg` / `baseline_centerlines.gpkg`：完整期次对照。\n- `changes.gpkg`：最终 added/deleted 和已应用操作的 replaced_old/replaced_new。\n- `audit.json`：逐操作几何、证据、原因与是否应用。\n- `tile_*/`：局部对比、候选、交叉断面搜索摘要、支路重挂、收敛和撤回信息。\n\n'
    text += '## 仍然存在的限制与迁移建议\n\n'
    text += '仍保留无法由当前证据可靠区分的短支路、真实转弯/停车区小环和近平行候选；不能把这些候选数直接等同于错误数，也不宣称整期已达到制图真值。RGB 无语义模型时，屋顶、硬化院落、阴影和未提供的跨层信息仍可能混淆。本轮目视检查覆盖输出中的典型图，不是逐条核验整期全部道路。\n\n'
    text += '值得继续验证并考虑迁移的是：独立 raw RGB ribbon 指标、方向约束路由、带外部接触点保护的中央轴重建、事务式拓扑回退。路口范围估计、边界伪轴替换和小环收缩仍需更多独立场景与标注验证，本轮不迁入正式算法。\n'
    (output/'REPAIR_REPORT.md').write_text(text, encoding='utf8')
