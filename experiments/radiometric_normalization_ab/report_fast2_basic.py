"""Figures and report for the no-suppression Fast2 normalization comparison."""
import os
from run_experiment import ROOT,environment,save
os.environ.update(environment())
from irmad_rrn import OUT,BASE,read,paired_tiles
from show_raw_toponet import background
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pyproj import Transformer
import shapely as sh

DEST=OUT/'fast2_basic'
EVAL=DEST/'evaluation'
KINDS=('added','removed','widened','narrowed')
COLORS=dict(added='#009b63',removed='#d93838',widened='#1875e3',narrowed='#ac36c4')
NAMES=dict(added='added 新增',removed='removed 灭失',widened='widened 变宽',narrowed='narrowed 变窄')


def figures():
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    frames={arm:gpd.read_file(DEST/arm/'changes.gpkg',layer='changes') for arm in ('raw','normalized')}
    t1=gpd.read_file(read(BASE/'20250118/latest_result.json')['centerlines']).to_crs('EPSG:32650')
    area=gpd.read_file(ROOT/'inputs/validation_area.shp').to_crs(t1.crs)
    bounds=area.total_bounds
    handles=[Patch(facecolor=COLORS[k],label=NAMES[k]) for k in KINDS]
    fig,axes=plt.subplots(1,2,figsize=(13,8),constrained_layout=True)
    for ax,(arm,frame) in zip(axes,frames.items()):
        t1.plot(ax=ax,color='#b9b9b9',linewidth=.25)
        for kind in KINDS:
            data=frame.loc[frame.change_typ==kind]
            if len(data):data.plot(ax=ax,color=COLORS[kind],alpha=.85,linewidth=.1)
        ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(('T1 → Raw T2' if arm=='raw' else 'T1 → IR-MAD T2')+f'：{len(frame)} 个基础区间')
    fig.legend(handles=handles,loc='lower center',ncol=4)
    fig.suptitle('Fast2 基础 segment/interval 变化：无额外假变化过滤\n灰色仅为 T1 道路位置参考；两列使用相同范围与尺度')
    fig.savefig(EVAL/'changes_overview.png',dpi=150);plt.close(fig)

    truth=gpd.read_file(EVAL/'gt/changes.shp').to_crs(t1.crs)
    # Explicitly post-evaluation visualization: GT picks display locations only.
    west=truth.loc[truth.BHBM.astype(str)=='2'].geometry.union_all().centroid
    east=truth.loc[truth.BHBM.astype(str).isin(['3','4'])].geometry.union_all().centroid
    locations=[(west.x,west.y,'GT 新增所在区'),(east.x,east.y,'GT 灭失 / 宽变所在区')]
    save(EVAL/'display_locations.json',dict(selected_after_prediction_freeze=True,locations=locations))
    pairs=list(paired_tiles())
    with __import__('rasterio').open(pairs[0][0]) as ds:crs=ds.crs
    transform=Transformer.from_crs(t1.crs,crs,always_xy=True)
    native={k:v.to_crs(crs) for k,v in frames.items()}
    gt=truth.to_crs(crs)
    fig,axes=plt.subplots(2,2,figsize=(12,11),constrained_layout=True)
    for row,(x,y,label) in enumerate(locations):
        left,bottom=transform.transform(x-230,y-230);right,top=transform.transform(x+230,y+230)
        bound=(min(left,right),min(bottom,top),max(left,right),max(bottom,top))
        for col,arm in enumerate(('raw','normalized')):
            ax=axes[row,col];background(ax,bound,col+1,pairs)
            f=native[arm];ids=f.sindex.query(sh.box(*bound));f=f.iloc[ids]
            for kind in KINDS:
                picked=f.loc[f.change_typ==kind]
                if len(picked):picked.plot(ax=ax,facecolor=COLORS[kind],edgecolor=COLORS[kind],alpha=.6,linewidth=.6)
            indices=gt.sindex.query(sh.box(*bound))
            if len(indices):
                gt.iloc[indices].boundary.plot(ax=ax,color='white',linewidth=2.5)
                gt.iloc[indices].boundary.plot(ax=ax,color='black',linewidth=1.,linestyle='--')
            ax.set_xlim(bound[0],bound[2]);ax.set_ylim(bound[1],bound[3]);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(label+' | '+('Raw T2' if arm=='raw' else 'IR-MAD T2'))
    fig.suptitle('离线评价局部图：黑白虚线为 GT，仅在预测冻结后叠加')
    fig.legend(handles=handles,loc='lower center',ncol=4)
    fig.savefig(EVAL/'changes_GT_regions.png',dpi=145);plt.close(fig)


def report():
    results={arm:read(DEST/arm/'result.json') for arm in ('raw','normalized')}
    metrics=read(EVAL/'metrics.json');consistency=read(OUT/'evaluation/metrics.json')
    rows=[]
    for kind in KINDS:
        a,b=(results[arm]['counts'][kind] for arm in results)
        ga,gb=(metrics['arms'][arm]['counts'][kind]['no_GT_intersection_count'] for arm in results)
        rows.append(f"| {kind} | {a['count']} | {b['count']} | {(b['count']/a['count']-1)*100:+.1f}% | {ga} → {gb} | {a['length_m']/1000:.2f} → {b['length_m']/1000:.2f} |")
    total_a,total_b=(results[arm]['total_count'] for arm in results)
    unmatched_a,unmatched_b=(metrics['arms'][arm]['no_GT_intersection_count'] for arm in results)
    shape_rows=[]
    for kind in KINDS:
        a,b=(metrics['arms'][arm]['counts'][kind] for arm in results)
        shape_rows.append(f"| {kind} | {a['median_interval_length_m']:.1f} → {b['median_interval_length_m']:.1f} | {a['p90_interval_length_m']:.1f} → {b['p90_interval_length_m']:.1f} |")
    area_a=sum(c['area_m2'] for c in results['raw']['counts'].values())
    area_b=sum(c['area_m2'] for c in results['normalized']['counts'].values())
    timing_rows=[]
    for key,label in [('profile_correspondence_seconds','网络匹配与已有宽度区间准备'),('removed_interval_seconds','removed 未覆盖区间'),('added_interval_seconds','added 未覆盖区间'),('width_interval_seconds','基础宽变区间'),('core_total_seconds','基础检测核心合计')]:
        timing_rows.append(f"| {label} | {results['raw']['timings'][key]:.2f} | {results['normalized']['timings'][key]:.2f} |")
    conclusion='仅归一化已有明显的局部收益，但不足以让最终基础 Fast2 变化同时更少、更稳定、更接近 GT。removed 大幅减少；added 和宽变区间反而增加。整体噪声仍占绝大多数，GT 命中没有改善，不能正式接入并宣称已解决假变化。'
    text=f'''# IR-MAD → Fast2 基础变化实验（20250118 → 20260203）

**{conclusion}**

## 变化数量

| 类型 | Raw T2 vs T1 | IR-MAD T2 vs T1 | 数量变化 | 无 GT 交集数量 | 区间长度 km |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}
| 合计 | {total_a} | {total_b} | {(total_b/total_a-1)*100:+.1f}% | {unmatched_a} → {unmatched_b} | — |

这里一个“对象”就是一个基础 segment/interval 记录，不进行对象合并。宽变的双侧带可为一个 MultiPolygon，仍计一条记录，不能与此前经过过滤和对象组装的 Fast2 数量直接比较。

- removed 数量下降 **42.4%**，长度从 **139.35km 降到 66.82km（−52.0%）**，是本轮最明确的正面结果；两组 removed 全部没有正面积 GT 交集。
- added 增加 **24.4%**；widened+narrowed 从 **477 增至 619（+29.8%）**。因此没有证据支持假新增和假宽变数量同步减少。
- 总区间数下降 **11.8%**，无 GT 交集区间从 **2580 降到 2277（−11.7%）**，但无交集比例仍为 **99.92% → 99.96%**。各区间面积相加从 **{area_a/1e6:.3f}km² 降到 {area_b/1e6:.3f}km²**；它是区间面积之和，可能含重叠，不能当作真实变化土地面积。

## GT 仅作最终离线评价

配置中的本对 GT 共 **4 个对象：新增 2、灭失 1、宽变 1**，均位于实验验证范围内。两组都只命中其中 **1 个新增 GT**；灭失和宽变 GT 均未命中。按同类型预测覆盖 GT 的面积比例为 **23.22% → 2.78%**，没有改善，反而下降。

交集定义为米制坐标下多边形交集面积 > 1e−6m²，不给 GT 加缓冲。GT 的宽变类别不区分 widened/narrowed，因此不能评价宽变方向正确性。GT 很少，可能不完备；“无交集”是离线错误代理，不代表每个区间都确定是假变化。但在相同 GT 上，本轮不能声称更接近真实变化。

## 几何分布与形状

新增/灭失保持原始未匹配轴线的宽度走廊；宽变保持配对轴线周围的差宽带，没有 reconciliation、对象组装或形状修正。图中 removed 的覆盖范围明显收缩，同时 normalized 组仍有分布广泛的 added 和宽变条带。

| 类型 | 区间长度中位数 m（Raw → IR-MAD） | P90 长度 m |
|---|---:|---:|
{chr(10).join(shape_rows)}

![同尺度全区基础变化](changes_overview.png)

![GT 所在区离线叠加](changes_GT_regions.png)

## 与道路一致性结果的关系

直接复用上一轮指标，没有重算 A：中心线 3m 匹配率 **29.17% → 60.81%**；稳定道路横向偏移中位数 **0.983 → 0.906m**；原始 SAM-MoLRA 面 IoU **0 → 24.83%**；最终道路面 IoU **32.82% → 55.94%**。

同一批稳定道路采样点的绝对宽差中位数 **3.01 → 1.73m**，整段绝对中位宽差 **2.71 → 1.52m**，确有改善。但 T2 恢复更多道路后，Fast2 几何匹配候选从 **1909 增至 3726**，进入比较的网络发生变化；全体宽变区间数量增加不等价于同一批稳定道路的宽度退化。现有宽度仍大量依赖 fallback，归一化没有消除全部边界、位置和拓扑差异。

## 严格控制实验范围

- T1、Raw T2、IR-MAD T2 的中心线、道路面、自然宽度全部复用，未重新道路提取、SAM-MoLRA 或测宽。正式生产代码和成果不写入。
- 旧 A 的 Fast2 结果已经经过影像验证和 reconciliation，且未保存完整过滤前候选。用户确认后，本次仅补算 A 尚不存在的基础变化结果，没有重复旧 A 流水线。
- 两组复用同一冻结 Fast2 的 `LongitudinalCoverage`、网络匹配和配对宽度区间函数；基础区间发布逻辑在实验 `fast2_basic.py` 中明确实现。它是去掉额外门控的实验消融版本，不是当前生产 Fast2 完整默认流程。
- 两组参数一致：位置容差 3m；原基础 presence factor=1.5（未覆盖判定 4.5m）；宽差≥max(2m, 20%×两期较大宽度)；最小区间长 24m、最小面积 4m²。这些是基础检测尺度，没有新增过滤规则。
- 没有多时相/Temporal、原始影像 patch 验证、segment/object reconciliation、GT 辅助、宽度背景偏差校正、宽度波动门控、额外道路面/概率否决或对象合并。输入道路的已有后处理保留，因为三组道路成果按要求直接复用。
- 两组 GPKG 在第一次读取 GT 前已写入哈希冻结记录；评价后再次核对哈希未变。执行时的实验源文件已在 `source/fast2_basic_executed.py` 归档并验证哈希。源码后续增加 I/O 计时和保持续跑冻结记录的保护，不改变检测计算，未重新运行检测。

## 本轮耗时

| 阶段 | Raw 组 s | IR-MAD 组 s |
|---|---:|---:|
| 道路提取 / 道路面 / 测宽 | 0（复用） | 0（复用） |
{chr(10).join(timing_rows)}

T1 场景加载 2.96s，Raw T2 加载 2.22s，IR-MAD T2 加载 2.83s；T1 场景只加载一次。含各自新增场景加载的检测时间为 16.78s / 23.61s，不含 GPKG/图表写出。首次完成版本未单独记录 I/O 时间，不重复计算以补计时。GT 离线评价 {metrics['evaluation_seconds']:.2f}s。上轮归一化和 IR-MAD T2 提取时间详见上一轮报告，本轮没有重复。

**建议：保留 IR-MAD 作为输入准备方向，但当前不正式接入，也不扩展全时期。它已减少一大类漏提造成的 removed；对最终假新增和假宽变，还不能仅靠颜色域归一化解决。**

成果：`../raw/changes.gpkg`、`../normalized/changes.gpkg`（changes 与 interval_axes 图层）；同目录还有各自 `road_changes.shp`。详细 GT 交集表为本目录两份 `*_GT_overlap.csv`。全部实验生成物均由独立目录的 `.gitignore` 排除。
'''
    (EVAL/'REPORT.md').write_text(text,encoding='utf-8')
    save(EVAL/'conclusion.json',dict(conclusion=conclusion,total_count_change_percent=(total_b/total_a-1)*100,
                                   no_GT_count_change_percent=(unmatched_b/unmatched_a-1)*100))


if __name__=='__main__':
    figures();report()
