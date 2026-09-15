"""Verify real outputs and publish a local experiment report from saved measurements."""
import argparse
import html
import json
from pathlib import Path
import platform

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import scipy
import cv2
import shapely

from boundary_width import digest, runs, save_json

HERE = Path(__file__).resolve().parent
NOTES = {
    '01_uniform': ('普通近等宽道路', 'road_063 的 90–240 m 段道路与周围植被区分较清楚。连续结果大体贴合可见道路走向，明显减少基线跳到远处背景的尖峰。仍需核实测量口径是否包含路肩，不能把曲线约 6 m 当作真值。'),
    '02_width_change': ('明显宽度变化道路', 'road_094 可见交叉口进口车道渐缩和中央分隔带形状变化，连续方法保留了变化趋势，没有强制整段恒宽。优化结果从约 24 m 降至约 13–14 m，但外侧边界部分包含植被/路侧边缘，变化幅度不能视为已验证的真实路宽变化。'),
    '03_offcenter': ('中心线偏心：可处理部分及失败', 'road_063 局部中心线靠近道路一侧。例如 s=36 m 的左右距离为 5.5 m 与 1.0 m，直接求和为 6.5 m；算法没有强制两边相等。但 s=57–75 m 仍沿远处背景形成约 18.5–21.5 m 的错误平台，而且通过了当前置信度筛选。这说明独立左右距离只能处理几何偏心，无法保证选择正确边界。'),
    '04_shadow_buildings': ('阴影/建筑干扰：失败', 'road_064 中连续边界会沿厂房及阴影边缘延续，产生约 14–15 m 与约 4–5 m 的宽度平台。原图不足以支持将这种差异解释为真实道路宽度突变。部分错误平台的影像证据分数仍然较高，说明当前置信度不能可靠识别语义错误。'),
    '05_junction': ('路口附近道路', 'road_094 的 s=0–12 m 标记 junction，未进入正常优化，诊断点不会与正常边界连成线。固定 12 m 缓冲只能处理图上检测到的路口；较大的交叉口铺装范围及渐变区可能超出缓冲。'),
    '06_offcenter_failure': ('中心线偏到路缘/路外：失败', 'road_031 的中心线贴近道路与绿地交界，连续结果可能将一侧边界放到植被里。若真实道路已全部位于中心线同一侧，正的左右距离参数化无法表示正确边界，连续优化不能修复这一输入问题。'),
    '07_image_boundary': ('影像边界：拒绝测量', 'road_014 的法线窗口进入白色无效影像范围，选例中没有正常宽度成果。仅保留局部估计作为诊断，左右边界矢量不跨越该缺口。此策略较保守，不能将无效覆盖误认为道路变窄。'),
}


def verify(folder, frame, manifest):
    checks = {}
    for method in ('baseline', 'optimized'):
        valid = frame[method+'_width'].notna()
        f = frame.loc[valid]
        np.testing.assert_allclose(f[method+'_width'], f[method+'_left_distance']+f[method+'_right_distance'], atol=1e-9)
        for side in ('left', 'right'):
            dx = f[method+'_'+side+'_x']-f.center_x
            dy = f[method+'_'+side+'_y']-f.center_y
            np.testing.assert_allclose(np.hypot(dx, dy), f[method+'_'+side+'_distance'], atol=1e-6)
        checks[method+'_coordinate_and_width_identity'] = True
    rejected = ~frame.accepted
    np.testing.assert_allclose(frame.loc[rejected, 'optimized_width'], frame.loc[rejected, 'baseline_width'], equal_nan=True)
    fragment_lengths = []
    for _, f in frame.groupby('road_id'):
        for a, b in runs(f.accepted.to_numpy(bool)):
            assert b-a >= manifest['config']['min_run_samples']
            fragment_lengths.append((b-a-1)*manifest['config']['spacing'])
    boundaries = gpd.read_file(folder/'measurements.gpkg', layer='accepted_boundaries')
    optimized = boundaries[boundaries.method == 'optimized']
    for (road_id, start, end), pair in optimized.groupby(['road_id', 'start_sample', 'end_sample']):
        assert len(pair) == 2
        lines = list(pair.geometry)
        assert all(l.is_simple for l in lines) and not lines[0].intersects(lines[1])
        f = frame[(frame.road_id == road_id) & frame.sample_id.between(start, end)]
        assert f.accepted.all() and len(f) == end-start+1
        for row in pair.itertuples():
            np.testing.assert_allclose(np.asarray(row.geometry.coords),
                f[['optimized_'+row.side+'_x', 'optimized_'+row.side+'_y']].to_numpy(), atol=1e-7)
    points = gpd.read_file(folder/'measurements.gpkg', layer='samples')
    assert len(points) == len(frame) and points.crs.to_string() == manifest['metric_crs']
    for kind in ('image', 'centerlines'):
        assert digest(manifest[kind]) == manifest[kind+'_sha256'], kind+' changed during experiment'
    for road_id, f in frame.groupby('road_id', sort=False):
        with np.load(folder/'roads'/f'{road_id}_profiles.npz') as z:
            np.testing.assert_allclose(z['optimized'].sum(1), f.optimized_width, equal_nan=True)
    json_rows = json.loads((folder/'samples.json').read_text(encoding='utf-8'))
    assert len(json_rows) == len(frame)
    checks.update(input_hashes_unchanged=True, csv_json_gpkg_npz_agree=True,
        rejected_samples_are_diagnostic_baseline=True, no_vectors_bridge_rejected_gaps=True,
        accepted_optimized_boundaries_non_crossing=True, optimized_boundary_features=len(optimized),
        accepted_fragments=len(fragment_lengths), median_fragment_span_m=float(np.median(fragment_lengths)),
        longest_fragment_span_m=float(np.max(fragment_lengths)),
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      opencv=cv2.__version__, rasterio=rasterio.__version__, geopandas=gpd.__version__, shapely=shapely.__version__),
        source_sha256={p.name:digest(p) for p in sorted(HERE.glob('*.py'))})
    save_json(folder/'verification.json', checks)
    return checks


def main(folder):
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    cases = json.loads((HERE/'cases.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(folder/'samples.csv').fillna({'flags': ''})
    verification = verify(folder, frame, manifest)
    b, o = summary['baseline'], summary['optimized']
    reduction = 100*(1-o['mean_abs_width_step_m']/b['mean_abs_width_step_m'])
    length = pd.read_csv(folder/'road_summary.csv').length_m.sum()/1000
    intro = f'''# 原始影像道路边界测宽：真实数据实验报告

## 结论

**连续优化有明显价值，但当前实现尚不具备直接替代 SAM-MoLRA 测宽的证据与可靠性。**

清晰、窄、边界可见的路段可获得较连贯结果；建筑阴影、路侧树冠、中心线贴路缘等场景会产生连续但错误的边界。当前方法适合作为独立候选测宽方案继续验证，尚不能作为全场景正式测宽来源。

本次只新增 `experiments/raw_image_boundary_width/`。正式测宽、道路面重建、Fast2 和 SAM-MoLRA 未参与本实验的计算或改动。

## 数据与复现范围

- 原始影像：`{manifest['image']}`，2025-01-18，8936×9991，uint8 RGB，原始 EPSG:4490，约 0.5 m 像素。
- 中心线：`{manifest['centerlines']}`，图层 `{manifest['layer']}`。来自已有 Fast 候选点增强设置下的 SAMRoad/TopoNet 保存预测边，不应称为完全未修改的上游 SAMRoad 推理。
- 输入未使用 SAM-MoLRA 道路面、正式宽度、Fast2 后处理结果或任何真值标签；没有重新推理。
- 反向重复边去重、交点节点化、degree=2 链合并后，长度≥70 m 的道路链共 {manifest['selection']['eligible_roads']} 条；确定性选取最长 {summary['roads']} 条，共约 {length:.2f} km，{summary['samples']:,} 个采样点。
- 距离与输出坐标均为 `{manifest['metric_crs']}` 米制；固定采样间隔 3 m，法线 ±20 m，横向步长 0.5 m，每侧最多 7 个候选。
- 只运行了一个区域、一个时相。较长道路有选择偏差；重点图是观察后选例，不能用作无偏成功率估计。

## 实际比较

最终版本要求每个正常连续段至少包含 3 个采样点，并在 ±20 m 候选范围外读取 2 m 证据缓冲，使搜索顶端能够正确计算色差和触发 search_limit。较早先导版本未包含这两项完整处理；本报告全部使用最终版本的统计与图片。重点位置沿用先导检查选例，没有因最终结果变差而替换。

| 指标 | 单横断面基线 | 连续联合优化 |
|---|---:|---:|
| 相邻有效点平均绝对宽度变化 | {b['mean_abs_width_step_m']:.3f} m | {o['mean_abs_width_step_m']:.3f} m |
| 相邻有效点变化 P95 | {b['p95_abs_width_step_m']:.1f} m | {o['p95_abs_width_step_m']:.1f} m |
| 相邻变化 >3 m 次数 | {b['jumps_gt_3m']:,} | {o['jumps_gt_3m']:,} |
| 比较的相邻有效点对数 | {b['adjacent_accepted_pairs']:,} | {o['adjacent_accepted_pairs']:,} |

平均跳变减少 **{reduction:.1f}%**；这是连续性改进，**不是边界精度提升百分比**。

通过筛选的点为 **{summary['accepted']:,}/{summary['samples']:,}（{summary['accepted_fraction']:.1%}）**。其余点保留诊断数据，不参加正常宽度优化。正常片段共 {verification['accepted_fragments']} 段，中位跨度 {verification['median_fragment_span_m']:.0f} m，最长 {verification['longest_fragment_span_m']:.0f} m；说明覆盖仍然碎片化，不能把部分路段平滑解释为已获得全路连续宽度。

比较对两方法使用相同的最终有效点集合。有效性筛选部分依赖优化后的置信度，因此此比较用于描述该方法的保留输出，不是独立测试集上的算法排名。完整 CSV 保留所有点，便于检查筛选偏差。

![全区连续性与覆盖统计](summary.png)

## 重点真实案例

图中：青色为适度平滑后的 SAMRoad 中心线，绿色/橙色为左右边界，黄色为有效横断面，红色及淡点为诊断结果。紫色为基线宽度，蓝色为连续结果。粉色区间被屏蔽；不跨屏蔽区连线。
'''
    sections, case_rows, gallery = [], [], []
    for case in cases:
        name, road = case['name'], case['road_id']
        title, note = NOTES[name]
        f = frame[(frame.road_id == road) & frame.s_m.between(case['start_m'], case['end_m'])]
        keep = f.accepted.to_numpy(bool)
        adjacency = keep[1:] & keep[:-1]
        result = dict(name=name, road_id=road, start_m=case['start_m'], end_m=case['end_m'],
                      samples=len(f), accepted=int(keep.sum()), accepted_fraction=float(keep.mean()))
        for method in ('baseline', 'optimized'):
            w = f[method+'_width'].to_numpy()
            diff = np.abs(np.diff(w))[adjacency]
            result[method+'_width_median_m'] = float(np.nanmedian(w[keep])) if keep.any() else None
            result[method+'_mean_abs_step_m'] = float(diff.mean()) if len(diff) else None
        case_rows.append(result)
        overlay, profile = f'cases/{name}_overlay.png', f'cases/{name}_profile.png'
        sections.append(f'\n### {title}\n\n`{road}`，s={case["start_m"]}–{case["end_m"]} m，有效 {keep.sum()}/{len(f)} 点。\n\n{note}\n\n![{title}：原图及边界]({overlay})\n\n![{title}：横断面展开和宽度]({profile})\n')
        gallery.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(road)} · {case["start_m"]}–{case["end_m"]} m · 有效 {keep.sum()}/{len(f)} 点</p><p>{html.escape(note)}</p><a href="{overlay}"><img loading="lazy" src="{overlay}"></a><a href="{profile}"><img loading="lazy" src="{profile}"></a></section>')
    pd.DataFrame(case_rows).to_csv(folder/'case_summary.csv', index=False)
    save_json(folder/'case_summary.json', case_rows)
    tail = f'''
## 失败原因与替代可行性

1. **缺少道路语义。** 能量只知道某处颜色/纹理变化强且连续，不能判断它是路缘、树冠、建筑、中央分隔带还是阴影。连续优化会把错误边缘稳定下来。
2. **边界定义不一致。** 当前可能混合车行道边缘、路肩、人行道或整个道路走廊。没有明确测宽口径与标注，宽度数值不可直接作为正式产品。
3. **中心线必须仍在道路内部。** 偏心可以用左右不等距离处理，偏到路外则需要先校正中心线或换用可表示两边同侧的参数化。
4. **置信度尚未标定。** 强而连续的建筑/阴影边缘会取得高分；低置信屏蔽无法覆盖所有错误。
5. **路口屏蔽有遗漏。** 依赖输入图节点；固定缓冲不足以刻画大型交叉口和漏检支路。
6. **连续结果不完整。** 当前约一半采样点仅能诊断，且大量有效段较短；不应为了画出完整曲线而跨缺口插值。

后续最有价值的验证是：先固定车行道/路肩边界口径，在不同场景标注独立左右边界，比较边界距离误差、宽度 MAE、覆盖率及拒绝后的风险，再与同范围 SAM-MoLRA 结果对齐评估。该任务没有提供逐点边界真值，本次未制造人工真值，也未声称优于 SAM-MoLRA。

## 验证与文件

- 5 项合成测试通过：联合 DP 与穷举一致、瞬时伪边缘压制、渐变且不对称宽度保留、正负侧候选不交叉、低置信分段、真实栅格读写的已知 10 m 偏心道路/nodata/平坦影像（后一测试包含多种情况）。
- 验证了两方法的 `width=left_distance+right_distance`、边界坐标到中心点的距离、CSV/NPZ 宽度一致、JSON/GPKG 记录数、矢量坐标与 CSV 一致、有效左右轨迹不相交、不跨屏蔽点、输入 SHA-256 未变化。
- 验证结果见 [verification.json](verification.json)，算法参数/输入来源见 [manifest.json](manifest.json)，逐案例数字见 [case_summary.csv](case_summary.csv)。
- [完整采样点 CSV](samples.csv) · [JSON](samples.json) · [width profile](width_profiles.csv) · [矢量 GeoPackage](measurements.gpkg) · [每条道路汇总](road_summary.csv)。
- 本地记录的处理耗时约 {summary['elapsed_seconds']:.1f} 秒（包含逐路制图和主要导出，未包含后续报告、输入复核和部分汇总制图）；这不是与模型方法的公平速度基准。
- 真实宽度精度、跨传感器泛化、完整道路覆盖以及相对 SAM-MoLRA 的精度仍未验证。
'''
    (folder/'REPORT.md').write_text(intro+''.join(sections)+tail, encoding='utf-8')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>原始影像道路边界测宽实验</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#172b39;background:#f0f4f6;max-width:1280px;margin:auto;padding:28px}}section,header{{background:white;border-radius:12px;padding:24px;margin:24px 0}}h1,h2{{line-height:1.35}}img{{width:100%;height:auto}}a{{color:#006b9b}}.metrics{{font-size:20px;background:#e7f3f9;padding:18px;border-radius:8px}}</style>
<header><h1>原始影像道路边界测宽实验</h1><p><strong>结论：连续优化有效降低跳变，但当前证据不足以替代 SAM-MoLRA 测宽。</strong></p><p>2025-01-18 原始 RGB + 已有 SAMRoad/TopoNet 中心线。120 条道路链、{summary['samples']:,} 个采样点。未改动正式流程。</p><p class="metrics">正常覆盖 {summary['accepted_fraction']:.1%} · 相邻宽度跳变 {b['mean_abs_width_step_m']:.2f} → {o['mean_abs_width_step_m']:.2f} m</p><p>跳变减少不等于精度提升。阴影和建筑场景可出现平滑但错误的边界。图片可点击放大。</p><p><a href="REPORT.md">完整中文报告</a> · <a href="samples.csv">CSV</a> · <a href="samples.json">JSON</a> · <a href="width_profiles.csv">Width profiles</a> · <a href="measurements.gpkg">GeoPackage</a> · <a href="verification.json">验证记录</a></p><img src="summary.png"></header>{''.join(gallery)}<section><h2>解释边界</h2><p>没有逐点边界真值，置信度是未标定证据分数。只有通过筛选的连续段参与优化；其余点是诊断，未插值补全。全部 120 条道路图位于 roads 目录，未隐藏失败案例。</p></section></html>'''
    (folder/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(verification=verification, case_summary=case_rows), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    main(parser.parse_args().folder)
