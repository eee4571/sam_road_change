"""Export reconstructed swath vectors and real-raster PNG comparisons (no web UI)."""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import shapely
from shapely.geometry import LineString, Polygon

from boundary_width import ImageReader, save_json
from reconstruct_width import spans


SOURCE_ORDER = {'measured': 0, 'smoothed': 1, 'interpolated': 2, 'propagated': 3}
SOURCE_COLORS = {'measured': '#1b9e77', 'smoothed': '#2878b5',
                 'interpolated': '#ee9b22', 'propagated': '#9458ba'}


def build_continuous_surfaces(frame, crs):
    records, lines, invalid = [], [], []
    for road_id, group in frame.groupby('road_id', sort=False):
        group = group.sort_values('sample_id')
        rows = list(group.itertuples())
        for a, b in zip(rows[:-1], rows[1:]):
            if not (a.reconstruction_available and b.reconstruction_available and b.sample_id == a.sample_id+1):
                continue
            polygon = Polygon([(a.final_left_x, a.final_left_y), (b.final_left_x, b.final_left_y),
                               (b.final_right_x, b.final_right_y), (a.final_right_x, a.final_right_y)])
            repaired = not polygon.is_valid
            if repaired:
                polygon = shapely.make_valid(polygon)
                invalid.append(dict(road_id=road_id, start_sample=int(a.sample_id)))
            for part in shapely.get_parts(polygon):
                if part.geom_type != 'Polygon' or part.area <= 0:
                    continue
                records.append(dict(road_id=road_id, start_sample=int(a.sample_id), end_sample=int(b.sample_id),
                    width=(a.final_width+b.final_width)/2,
                    left_distance=(a.final_left_distance+b.final_left_distance)/2,
                    right_distance=(a.final_right_distance+b.final_right_distance)/2,
                    confidence=min(a.final_confidence, b.final_confidence)*(.5 if repaired else 1),
                    start_source=a.width_source, end_source=b.width_source, geometry_repaired=repaired,
                    width_source=max((a.width_source, b.width_source), key=SOURCE_ORDER.get),
                    geometry=part))
        for begin, end in spans(group.reconstruction_available.to_numpy(bool)):
            if end-begin < 2:
                continue
            part = group.iloc[begin:end]
            for side in ('left', 'right'):
                geom = LineString(part[[f'final_{side}_x', f'final_{side}_y']].to_numpy())
                lines.append(dict(road_id=road_id, side=side, self_intersection=not geom.is_simple, geometry=geom))
    if not records:
        raise ValueError('No supported consecutive points for surfaces')
    return gpd.GeoDataFrame(records, crs=crs), gpd.GeoDataFrame(lines, crs=crs), invalid


def plot_profile(ax, frame, sides=False):
    if sides:
        for side, color in [('left', '#2466b2'), ('right', '#c13b28')]:
            ax.plot(frame.s_m, frame[f'optimized_{side}_distance'], color=color, alpha=.25, lw=.8)
            ax.plot(frame.s_m, frame[f'final_{side}_distance'], color=color, lw=1.7, label=f'最终{side}')
        ax.set_ylabel('单侧距离 (m)')
    else:
        ax.plot(frame.s_m, frame.optimized_width, color='.65', lw=.8, alpha=.7, label='原始观测（含拒绝值）')
        accepted = frame[frame.accepted]
        ax.scatter(accepted.s_m, accepted.optimized_width, c='black', s=10, label='原始有效观测', zorder=3)
        ax.plot(frame.s_m, frame.final_width, color='#007f78', lw=2, label='连续重建')
        bad = frame[frame.outlier_reason.fillna('').ne('')]
        ax.scatter(bad.s_m, bad.optimized_width, c='#dd3445', marker='x', s=30, label='判定短时异常', zorder=4)
        for source, color in [('interpolated', '#feb24c'), ('propagated', '#ad8bd3')]:
            for begin, end in spans(frame.width_source.eq(source).to_numpy()):
                part = frame.iloc[begin:end]
                ax.axvspan(part.s_m.iloc[0]-1.5, part.s_m.iloc[-1]+1.5, color=color, alpha=.18)
        ax.set_ylabel('宽度 (m)')
    for begin, end in spans(frame['flags'].fillna('').str.contains('junction').to_numpy()):
        p = frame.iloc[begin:end]
        ax.axvspan(p.s_m.iloc[0]-1.5, p.s_m.iloc[-1]+1.5, color='#aaaaaa', alpha=.12)
    ax.set_xlabel('沿道路里程 (m)')
    ax.grid(alpha=.18)
    ax.legend(loc='upper right', fontsize=8, ncol=2)


def select_cases(frame):
    cases = [dict(name='shadow', road_id='road_064', start_m=0, end_m=135,
                  label='阴影 / 建筑邻接道路：检查短时误边界'),
             dict(name='building', road_id='road_064', start_m=135, end_m=270,
                  label='建筑边缘邻接道路：持续误吸附仍可能保留'),
             dict(name='canopy', road_id='road_031', start_m=0, end_m=120,
                  label='树冠 / 植被邻接道路：中心线贴边风险'),
             dict(name='taper', road_id='road_094', start_m=15, end_m=180,
                  label='可见宽度变化与路口：检查是否保留渐变')]
    candidates = []
    for road_id, group in frame.groupby('road_id', sort=False):
        for begin, end in spans(group.width_source.eq('propagated').to_numpy()):
            if begin > 0 and end < len(group) and end-begin >= 25:
                length = group.s_m.iloc[end]-group.s_m.iloc[begin-1]
                if length <= 350:
                    candidates.append((length, road_id, group.s_m.iloc[begin], group.s_m.iloc[end-1]))
    if candidates:
        length, road_id, start, end = max(candidates)
        cases.append(dict(name='long_gap', road_id=road_id, start_m=max(0, start-45), end_m=end+45,
                          label=f'长缺口恢复：约 {length:.0f} m，紫色区域为传播推断'))
    return cases


def main(folder):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(folder/'samples.csv')
    destination = folder/'continuous_surfaces.gpkg'
    if destination.exists():
        raise FileExistsError(destination)
    polygons, boundaries, repairs = build_continuous_surfaces(frame, manifest['metric_crs'])
    polygons.to_file(destination, layer='width_surfaces', driver='GPKG')
    boundaries.to_file(destination, layer='boundaries', driver='GPKG')
    polygons = gpd.read_file(destination, layer='width_surfaces')
    assert polygons.geometry.is_valid.all()
    np.testing.assert_allclose(polygons.width, polygons.left_distance+polygons.right_distance)
    reader = ImageReader(manifest['image'], manifest['metric_crs'])
    background = Image.fromarray(reader.ds.read([1, 2, 3]).transpose(1, 2, 0))
    overlay = background.copy()
    draw = ImageDraw.Draw(overlay, 'RGBA')
    cmap = plt.get_cmap('turbo')
    norm = matplotlib.colors.Normalize(2, 36, clip=True)
    for row in polygons.itertuples():
        xy = reader.pixels(np.asarray(row.geometry.exterior.coords))
        color = tuple(round(v*255) for v in cmap(norm(row.width))[:3])
        draw.polygon([tuple(p) for p in xy], fill=color+(135,))
    overlay.save(folder/'continuous_overlay_native.png', compress_level=3)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    old_path = Path(manifest['observations']).parent/'width_surface_overlay_native.png'
    before = Image.open(old_path) if old_path.exists() else background.copy()
    fig, axes = plt.subplots(1, 2, figsize=(19, 11), constrained_layout=True)
    for ax, im, title in zip(axes, (before, overlay), ('重建前：原始有效面片', '重建后：连续宽度面（包含插值和传播）')):
        thumb = im.copy()
        thumb.thumbnail((2600, 2600))
        ax.imshow(thumb)
        ax.axis('off')
        ax.set_title(title)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=.018, pad=.01, label='宽度 (m)')
    fig.suptitle('原始影像道路测宽｜面矢量直接叠加 PNG；补齐区域为推断，颜色不表示置信度')
    fig.savefig(folder/'continuous_overview_comparison.png', dpi=180)
    fig.savefig(folder/'continuous_preview.jpg', dpi=90)
    plt.close(fig)
    source_view = background.copy()
    source_view.thumbnail((2400, 2400))
    source_draw = ImageDraw.Draw(source_view, 'RGBA')
    factors = np.array(source_view.size)/np.array(background.size)
    for row in polygons.itertuples():
        xy = reader.pixels(np.asarray(row.geometry.exterior.coords))*factors
        color = tuple(round(v*255) for v in matplotlib.colors.to_rgb(SOURCE_COLORS[row.width_source]))
        source_draw.polygon([tuple(p) for p in xy], fill=color+(190,))
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(12, 13), constrained_layout=True)
    ax.imshow(source_view)
    ax.axis('off')
    ax.legend(handles=[Patch(color=c, label=s) for s, c in SOURCE_COLORS.items()], loc='lower right')
    ax.set_title('连续宽度来源｜绿：保留观测；蓝：平滑；橙：插值；紫：低置信传播')
    fig.savefig(folder/'width_source_map.png', dpi=170)
    plt.close(fig)
    cases = select_cases(frame)
    case_dir = folder/'cases'
    case_dir.mkdir(exist_ok=True)
    fig_profiles, profile_axes = plt.subplots(len(cases), 1, figsize=(15, len(cases)*3.2), constrained_layout=True)
    for case, profile_ax in zip(cases, np.atleast_1d(profile_axes)):
        g = frame[(frame.road_id == case['road_id']) & frame.s_m.between(case['start_m'], case['end_m'])]
        if g.empty:
            continue
        fig = plt.figure(figsize=(15, 11), constrained_layout=True)
        grid = fig.add_gridspec(3, 3, height_ratios=[2.1, 1.15, 1.])
        coords = reader.pixels(g[['center_x', 'center_y']].to_numpy())
        low = np.maximum(np.floor(coords.min(0)-65), 0).astype(int)
        high = np.minimum(np.ceil(coords.max(0)+65), background.size).astype(int)
        box = (*low, *high)
        for j, (im, title) in enumerate(zip((background, before, overlay), ('原图 + 中心线', '重建前', '连续重建后'))):
            ax = fig.add_subplot(grid[0, j])
            ax.imshow(im.crop(box))
            if j == 0:
                ax.plot(coords[:, 0]-low[0], coords[:, 1]-low[1], '--', color='#ffff40', lw=1)
            ax.axis('off')
            ax.set_title(title)
        plot_profile(fig.add_subplot(grid[1, :]), g)
        plot_profile(fig.add_subplot(grid[2, :]), g, sides=True)
        fig.suptitle(f"{case['road_id']}｜{case['label']}\n橙底=插值；紫底=传播；灰底=路口。影像场景标签不是误差真值。")
        fig.savefig(case_dir/f"{case['name']}.png", dpi=150)
        fig.savefig(case_dir/f"{case['name']}_preview.jpg", dpi=80)
        plt.close(fig)
        plot_profile(profile_ax, g)
        profile_ax.set_title(f"{case['road_id']}｜{case['label']}")
        g.to_csv(case_dir/f"{case['name']}.csv", index=False)
    fig_profiles.savefig(folder/'profile_comparison.png', dpi=170)
    plt.close(fig_profiles)
    save_json(folder/'selected_cases.json', cases)
    save_json(folder/'geometry_verification.json', dict(polygons=len(polygons), repaired_pairs=repairs,
        boundary_self_intersections=int(boundaries.self_intersection.sum()),
        geometries_valid=True, width_sum_verified=True, native_png_dimensions=list(overlay.size),
        note='Bow-tie quadrilaterals are split into valid polygon parts; this does not correct inaccurate centerlines.'))
    write_report(folder, frame, polygons, boundaries, repairs)
    before.close()
    reader.ds.close()
    print(f'Exported {len(polygons)} polygons; repaired {len(repairs)} crossing faces; PNGs ready.', flush=True)


def write_report(folder, frame, polygons, boundaries, repairs):
    stats = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    sources = '\n'.join(f"| {k} | {v:,} | {stats['source_fractions'][k]:.2%} |"
                        for k, v in stats['source_counts'].items())
    outliers = frame.outlier_reason.fillna('').ne('')
    supported_roads = frame.loc[frame.reconstruction_available, 'road_id'].nunique()
    text = f'''# 全图连续宽度重建实测报告

## 结果文件

- [全图前后面矢量叠加 PNG](continuous_overview_comparison.png)
- [原始分辨率连续面叠加 PNG](continuous_overlay_native.png)
- [宽度来源图 PNG](width_source_map.png)
- [五组 width profile 对比](profile_comparison.png)
- [连续宽度面与左右边界 GeoPackage](continuous_surfaces.gpkg)
- [逐点 CSV](samples.csv)、[JSON](samples.json)、[异常路段 CSV](outlier_segments.csv)

保留原始候选生成和 Viterbi，仅以保存的观测进行第二层重建。全部 {stats['roads']:,} 条链、{stats['samples']:,} 点已处理；{supported_roads:,} 条链有左右重建支撑。原始输入哈希及参数见 manifest.json。

## 覆盖率与连续性

| 指标 | 重建前 | 重建后 |
|---|---:|---:|
| 采样点覆盖率 | {stats['original_valid_sample_fraction']:.2%} | {stats['continuous_sample_fraction']:.2%} |
| 相邻采样点覆盖的长度比例 | {stats['original_valid_length_fraction']:.2%} | {stats['continuous_length_fraction']:.2%} |
| 原始有效相邻点对：平均跳变 | {stats['original_adjacent_jumps']['mean_m']:.3f} m | {stats['final_same_pairs_jumps']['mean_m']:.3f} m |
| 原始有效相邻点对：P95 跳变 | {stats['original_adjacent_jumps']['p95_m']:.3f} m | {stats['final_same_pairs_jumps']['p95_m']:.3f} m |
| 全部有限观测的相同点对：平均跳变 | {stats['raw_diagnostic_same_pairs_jumps']['mean_m']:.3f} m | {stats['final_diagnostic_same_pairs_jumps']['mean_m']:.3f} m |
| 全部有限观测的相同点对：P95 跳变 | {stats['raw_diagnostic_same_pairs_jumps']['p95_m']:.3f} m | {stats['final_diagnostic_same_pairs_jumps']['p95_m']:.3f} m |

原始有效比较使用同一组 {stats['original_adjacent_jumps']['pairs']:,} 点对。重建后所有 {stats['final_all_adjacent_jumps']['pairs']:,} 对的平均 / P95 跳变为 {stats['final_all_adjacent_jumps']['mean_m']:.3f} / {stats['final_all_adjacent_jumps']['p95_m']:.3f} m，该集合包含新增路口和补齐区域，不能与原始有效子集直接当作等价精度比较。长度分母是采样点之间的 {stats['sampled_chain_length_m']/1000:.3f} km，不含各链末端不足采样步长的尾段。

异常判定 {stats['outlier_points']:,} 点、{stats['outlier_segments']:,} 个连续异常段，涉及 {frame.loc[outliers, 'road_id'].nunique():,} 条链；其中 {(outliers & frame.accepted).sum():,} 点原先通过筛选。异常段按同链相邻点计数，左右侧异常合并，不代表真值确认的错误。最终置信度 ≥0.16 的点仅占全体 {stats['final_confidence_ge_016_fraction']:.2%}，覆盖率不能当作准确率。

| 来源 | 点数 | 占全部采样点 |
|---|---:|---:|
{sources}

`measured`：两侧都来自原始有效锚点且各自改动 ≤0.25 m；`smoothed`：稳健拟合的观测（含重新纳入的路口观测）；`interpolated`：两端约束的 ≤75 m 缺口；`propagated`：长缺口或链端延续；`unresolved`：该链至少一侧完全无可靠支撑。最终来源取左右两侧较弱的一类，另保留 left_source/right_source。

## 典型影像案例与失败

- [阴影邻接道路](cases/shadow.png)：road_064 的短时远侧边缘响应得到抑制，补齐原先断开的区间；持续阴影边界仍可能被保留。
- [建筑边缘邻接道路](cases/building.png)：road_064 的约 185–195 m 异常平台被压制；后段持续增宽仍保留。没有边界真值，不能把后段宽度升高全部解释为真实拓宽。
- [树冠 / 植被邻接道路](cases/canopy.png)：road_031 短时尖峰降低，但图中中心线贴近树带，连续结果仍可能描绘错误目标，是明确的适用性失败。
- [渐缩与路口](cases/taper.png)：road_094 约 24 m 到 13 m 的持续下降趋势保留，没有压成定宽。影像可见道路收窄，但幅度仍混有路侧干扰，不能认定为真值宽度变化。
- [长缺口恢复](cases/long_gap.png)：road_008 约 228 m 缺口使用两端稳定值及过渡重建，连续但依赖外推，置信度随距锚点距离下降；不能作为新增实测证据。

案例是对实际原图的场景核查，不通过宽度数值自动判定“阴影/树冠/建筑”成因。所有案例均包含原图、前后面叠加、宽度及左右距离 profile；橙色底为插值，紫色底为传播，灰色底为路口。

## 几何与验证

导出 {len(polygons):,} 个有效 Polygon 面和 {len(boundaries):,} 条边界线；{len(repairs):,} 个弯曲横断面组成的交叉四边形通过 make_valid 拆成合法面片，并在面属性 geometry_repaired 标注、置信度减半。{int(boundaries.self_intersection.sum()):,} 条边界线存在自交标记；几何修复不代表道路形状正确。宽度恒等式及面合法性已检查。图像直接读取导出的矢量叠加原始 RGB，原生尺寸为 8936×9991。

## 判断

第二层重建适合做连续 profile 与候选道路面的后处理，能抑制短时异常并保留持续变化。它不提供新的道路语义信息：持续错误边界、偏出道路的中心线、长距离无观测区域仍然可能得到平滑但错误的结果。短小真实停车湾或局部拓宽也可能满足“返回原宽度”的异常模式。目前没有人工宽度真值，不能证明精度提高或替代 SAM-MoLRA；本轮证明的是连续性改善和缺失恢复能力。
'''
    (folder/'REPORT.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    main(parser.parse_args().folder)
