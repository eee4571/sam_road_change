"""Reproducible review artifacts and candid per-scene experimental reporting."""
from collections import Counter
import csv
from types import SimpleNamespace
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import wkt
from shapely.geometry import box

from data import read_json
from evidence import ImageGrid
from report import background, draw, write_json
from repair_artifacts import compare


def operation_details(folder, before, after, evidence, roi, audit):
    details = []
    accepted = [(i, a) for i, a in enumerate(audit) if a.get('applied') and (a.get('new_wkt') or a.get('old_wkt'))]
    centers = []
    for index, row in accepted:
        geometry = wkt.loads(row.get('new_wkt') or row['old_wkt'])
        center = geometry.centroid
        if any(center.distance(c) < 60 for c in centers):
            continue
        centers.append(center)
        size = min(150., max(60., max(geometry.bounds[2]-geometry.bounds[0], geometry.bounds[3]-geometry.bounds[1])*.65))
        close = box(center.x-size, center.y-size, center.x+size, center.y+size).intersection(roi)
        path = folder/f'operation_{len(details)+1:02d}.png'
        compare(path, before, after, evidence, close, f"{folder.name}: {row['action']} / audit {index}")
        details.append(dict(file=path.name, audit_index=index, bbox_m=list(close.bounds), action=row['action']))
    write_json(folder/'detail_index.json', details)


def selection_map(output, frame, inputs, regions):
    extent = box(*frame.total_bounds).buffer(20)
    evidence = SimpleNamespace(image=ImageGrid(inputs['images'], frame.crs, extent, 4.))
    fig, ax = plt.subplots(figsize=(10, 12))
    background(ax, evidence)
    for row in regions:
        roi = box(*row['bbox_m'])
        color = '#33ffff' if row['control'] else '#ffff33'
        draw(ax, roi.boundary, color=color, linewidth=1.5)
        ax.text(roi.centroid.x, roi.centroid.y, row['id'], color='black', ha='center',
                bbox=dict(facecolor=color, alpha=.85, edgecolor='none'), fontsize=8)
    ax.set_title('Scene-diverse ROIs / cyan: relative controls')
    ax.ticklabel_format(style='plain', useOffset=False)
    fig.tight_layout()
    fig.savefig(output/'roi_overview.png', dpi=160)
    plt.close(fig)


def write_multi_report(output, summaries):
    verification = read_json(output/'verification.json')
    inputs = read_json(output/'inputs.json')
    notes_path = output/'visual_review.json'
    notes = read_json(notes_path) if notes_path.exists() else {}
    columns = ['id', 'category', 'control', 'committed_operations', 'changed_old_fraction',
               'components_before', 'components_after', 'tips_before', 'tips_after',
               'loops_before', 'loops_after', 'duplicates_before', 'duplicates_after', 'added_m', 'deleted_m']
    rows, all_audit = [], []
    for s in summaries:
        audit = read_json(output/s['id']/'audit.json')
        all_audit.extend(dict(r, roi=s['id']) for r in audit)
        rows.append(dict(id=s['id'], category=s['category'], control=s['control'],
            committed_operations=sum(r.get('applied', False) for r in audit),
            changed_old_fraction=s['deleted_length_m']/max(s['before']['total_length_m'], 1),
            components_before=s['before']['component_count'], components_after=s['after']['component_count'],
            tips_before=s['before']['degree_1_endpoints'], tips_after=s['after']['degree_1_endpoints'],
            loops_before=s['before']['small_loop_count'], loops_after=s['after']['small_loop_count'],
            duplicates_before=s['before']['near_duplicate_count'], duplicates_after=s['after']['near_duplicate_count'],
            added_m=s['added_length_m'], deleted_m=s['deleted_length_m']))
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output/'audit.json', all_audit)
    distributions = {}
    for kind in sorted({a['action'] for a in all_audit}):
        group = [a for a in all_audit if a['action'] == kind]
        distributions[kind] = dict(hypotheses=len(group), applied=sum(a.get('applied', False) for a in group),
            rejected_reasons=dict(Counter(reason for a in group for reason in a.get('transaction_guard', {}).get('failures', []))))
        for decision in (True, False):
            pairs = []
            for a in group:
                e = a.get('evidence', {})
                if not isinstance(e, dict) or a.get('applied', False) != decision:
                    continue
                old, new = e.get('old', {}), e.get('new', {})
                x, y = old.get('raw_rgb_support', old.get('rgb')), new.get('raw_rgb_support', new.get('rgb'))
                if x is not None and y is not None:
                    pairs.append([x, y])
            distributions[kind]['accepted_rgb' if decision else 'kept_rgb'] = dict(
                n=len(pairs), median_before_after=np.median(pairs, axis=0).tolist() if pairs else None,
                q10_before_after=np.quantile(pairs, .1, axis=0).tolist() if pairs else None,
                q90_before_after=np.quantile(pairs, .9, axis=0).tolist() if pairs else None)
    write_json(output/'evidence_distributions.json', distributions)
    text = '# 多 ROI 道路图修复实验报告\n\n'
    text += f"输入：{Path(inputs['project']).name} / {inputs['period']} / 区域 {inputs['area']}；米制 CRS：{inputs['crs']}。每区独立读取同一份正式中心线，统一算法与参数，前一区修改不影响后一区。所有成果仅在实验 outputs 中。\n\n"
    text += f"共 {len(rows)} 个 ROI，其中 {sum(r['control'] for r in rows)} 个相对正常对照。对照同样运行全部算法，没有跳过修改；没有道路真值，低异常排名不等于已确认无错误。\n\n"
    text += '## 视觉验收优先\n\n以原始 RGB 上的路面位置、中心性、连续性、单/双轴一致性和路口接续判断效果。拓扑计数仅用于定位和防退化，不能作为成功判据。下面是逐图查看记录，不是人工真值精度。\n\n'
    text += '| ROI | 目视结论 | 主要剩余问题 |\n|---|---|---|\n'
    for r in rows:
        note = notes.get(r['id'], {})
        text += f"| {r['id']} | {note.get('verdict', '尚未目视验收')} | {note.get('failure', '尚未逐图检查')} |\n"
    text += '\n'
    text += '![ROI 分布](roi_overview.png)\n\n'
    text += '## 方法与迭代\n\n'
    if inputs.get('algorithm') == 'structural':
        text += ('先按完整道路段检验长窄重复环内是否为同一片铺装，再用固定道路入口的 canonical axis 取代重复路径；真实分隔带和真实闭环拒绝合并。原始要素先按实际接点构图，避免漏掉中段支路。\n\n'
                 '中央轴候选不再要求偏轴位置预先通过 RGB 判据。沿已有骨架、固定多孔径、有限横向 corridor 搜索；同时使用原始横断面、纵向重复结构和成对路缘，植被正证据排除树带。重叠且属于同一路面的轴线方案合为一条，接回旧路使用渐变接续而非直角台阶。\n\n')
        text += ('沿用原中心线骨架与 RGB 修复框架，本轮不运行街区全局平滑。保留 medial axis、支路重挂、平行道路 veto、方向约束配对路由和事务回退。\n\n'
                 '20～60m degree-2 短弯链在原线附近最多 4.5m corridor 内作多孔径 RGB 位置搜索，固定真实端点及两端 3m 接近段；对离散偏移作有界亚像素插值，再重新评价中心性、连续性、曲率及新增接触。\n\n'
                 '多类异常在直径 50m 内聚集时形成 repair domain，所有窗外接触固定。keep、collapse/rewire、connect、collapse+refit+connect+prune 从同一输入比较，联合目标包括 RGB 不支持长度、中心性、曲率和有证据连接收益。候选可用性和内部拒绝也完整留在审计中；并非所有域都能产生可行重建。\n\n'
                 '主动追踪由可信端点出发，2.5m 步长、方向状态束搜索，不预先指定 gap pair；最多向前 60m、横向 18m，沿 RGB ribbon 探索到第一条已有道路。接入位置附近 24m 范围验证目标道路，需同铺装连续、方向和曲率合理；跨层或穿越既有路则拒绝。SAM/MOLRA 只降低搜索代价，不能替代 RGB 接受判据；没有找到已有道路的孤立延伸不写入成果。\n\n'
                 '每区最多两轮，局部方案提交后重新构图。所有 ROI 包括 controls 使用相同参数。几何/影像缺陷仍可能漏召回，本轮不将程序检查通过等同于制图正确。\n\n')
    elif inputs.get('algorithm') == 'block':
        text += ('街区级实验：对整个 ROI 的道路顶点和共享路口联合求解稀疏最小二乘问题，目标包含 RGB 横向位置、长距离主方向、曲率和平行道路间距。窗口外道路、端点及尚未判清的平行轴受硬约束。不是逐条短线的独立平滑。\n\n'
                 '路口重建先检查闭合区域内部是否与路缘呈同一铺装；有独立围合地物的真实环路保留。长道路延续根据 25～40m 进出段、局部路面色度和两侧反差提出缺失段，并在方向状态网格中搜索可转弯走廊。两侧允许亮建筑与暗树荫等不同反差方向；共同铺装前庭使用连续材料判据，不要求交叉口内仍有两条路缘。RGB 使用固定多尺度描述，不产生测宽结果，SAM/MOLRA 缺失不作否定证据。\n\n'
                 '所有改图仍通过窗外几何、外部连通、已有独立道路、新增交叉/断线/重复线等保护。联合拟合若增加重复线，会固定冲突位置重新求解；不通过检查的整图候选保留在 block_proposal.gpkg，不作为 final 成果。规则外观不等于真实拓扑，仍须看逐区实图记录。\n\n')
    else:
        text += ('先扫描几何事件，再按场景类别排名和最大 15% 窗口重叠约束选择不同区域。粗分辨率 RGB 只用于有效覆盖和场景代理筛选；建筑代理仍经逐图查看。排除了影像边缘/稀疏林地充当正常对照的选择偏差。\n\n'
             '局部优化显式保留 keep 方案，比较中央轴替换及支路重挂、小环/重复路径的联合收缩、不同缓存尺度下的固定接点 refit、端点对连接、向前预测至道路中段的 junction 连接，以及有肯定植被证据的终端伪支路删除。每次提交后重建邻域图继续检查，最多 3 轮。\n\n'
             'RGB 使用内部颜色均匀性、两侧颜色/灰度反差、纵向支持和中心位置。离散断面孔径由缓存宽度派生，不生成新测宽产品；SAM/MOLRA 只加正证据，surface 不参与独立接受判定。路口末段单独判断铺装延续，避免要求交叉口内部仍有两侧路缘。\n\n'
             '每次事务检查窗外几何、固定接点的连通关系、已有独立 RGB 支持道路、新增孤立端点/交叉/小环/重复线及几何有效性。真实双 ribbon 与稳定分隔带禁止中央合并。没有全区任意平滑，也没有按长度直接删 spur 或按周长直接删环。\n\n')
    text += '## 结构诊断附表（不作为效果评分）\n\n| ROI | 场景 | 提交事务 | 修改旧线比例 | 分量 | 端点 | 小环 | 重复候选 |\n|---|---|---:|---:|---|---|---|---|\n'
    for r in rows:
        text += f"| {r['id']} | {r['category']} | {r['committed_operations']} | {r['changed_old_fraction']:.1%} | {r['components_before']}→{r['components_after']} | {r['tips_before']}→{r['tips_after']} | {r['loops_before']}→{r['loops_after']} | {r['duplicates_before']}→{r['duplicates_after']} |\n"
    text += '\n上述是结构诊断，不是错误数量或准确率。added/deleted 是几何差，包括居中替换，不能理解为新建/永久删除道路长度。不同尺度/迭代的候选不是独立样本。\n\n'
    text += '## 逐区检查\n\n'
    for s, r in zip(summaries, rows):
        text += f"### {s['id']} / {s['category']}\n\n选择原因：{s['reason']}。BBox：{[round(v, 2) for v in s['bbox_m']]}。\n\n"
        text += f"实际应用：{s['actions'] or 'keep；没有满足独立证据和保护条件的修改'}。\n\n"
        note = notes.get(s['id'])
        if note:
            for key, label in [('scene', '影像场景与主要问题'), ('improvement', '目视改善'), ('failure', '仍失败/未解决'), ('side_effect', '副作用检查')]:
                text += f"- {label}：{note.get(key, '未填写')}\n"
        else:
            text += '目视结论尚未填写；程序通过不构成目视验收。\n'
        text += f"\n![前后]({s['id']}/before_after.png)\n\n"
        if (output/s['id']/'rgb_previous_current.png').exists():
            text += f"[原始 RGB / 上一版 / 本版，同范围同线宽]({s['id']}/rgb_previous_current.png)\n\n"
        for k, case in enumerate((note or {}).get('cases', [])):
            path = output/s['id']/f'visual_case_{k+1:02d}.png'
            if path.exists():
                text += f"- [视觉案例：{case['title']}]({s['id']}/{path.name})\n"
        details = read_json(output/s['id']/'detail_index.json')
        for d in details:
            text += f"- [局部 {d['action']}，审计 {d['audit_index']}]({s['id']}/{d['file']})\n"
        text += '\n'
    text += '## 证据和可迁移性\n\n'
    text += '候选级 before/after 分数、正证据、拒绝原因及分位数见 `audit.json` 和 `evidence_distributions.json`。没有人工真值标签，本轮不报告 candidate recall/precision，也不把优化目标分数的提高当作独立准确率。\n\n'
    text += '颜色断面补充了等灰度异色道路的证据，但屋顶、硬化院落、河渠、阴影仍可能混淆。对暗色双幅主路、复杂路口错连、建筑边缘伪线及缺失整段道路，本轮只应按各 ROI 目视记录判断收益，不能宣称已经稳定解决。\n\n'
    text += f"输入文件大小/mtime 指纹前后一致：{verification['input_fingerprints_unchanged']}。未运行模型、IR-MAD 或测宽。正式 pipeline 未修改。\n\n"
    if 'input_assets_unchanged' in verification:
        text += f"实际输入影像、矢量和缓存指纹一致：{verification['input_assets_unchanged']}。发现清单 pipeline_result.json 的并发元数据变化单独记录在 verification.json，不混作本期影像或中心线变化。\n\n"
    text += '## 文件\n\n- `selection.json` / `roi_overview.png`：ROI 清单与空间分布。\n- `roi_*/initial.png` / `before_after.png` / `operation_*.png`：原始影像、全 ROI 和局部对比。\n- `roi_*/baseline_centerlines.gpkg` / `final_centerlines.gpkg`：context 与 roi 图层。\n- `roi_*/changes.gpkg`：added、deleted、replaced_old、replaced_new，包括空图层。\n- `roi_*/audit.json` / `summary.json`：逐方案决定、证据、事务检查与拓扑统计。\n- `summary.csv` / `evidence_distributions.json`：汇总表与特征分布。\n'
    notes_file = output/'EXPERIMENT_NOTES.md'
    if notes_file.exists():
        text += '\n'+notes_file.read_text(encoding='utf8')
    (output/'MULTI_ROI_REPORT.md').write_text(text, encoding='utf8')
