"""Build local comparison figures and a concise report from frozen extraction outputs."""
from pathlib import Path
import os
from run_experiment import environment
os.environ.update(environment())
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import shapely as sh

from irmad_rrn import ROOT, OUT, BASE, read, paired_tiles, ncp
from run_experiment import save

EVAL = OUT/'evaluation'


def radiometric_diagnostics():
    """Post-fit DN audit only; this sample never feeds parameter selection."""
    config=read(OUT/'normalization.json')
    z=np.memmap(OUT/'paired_valid_rgb.bin',mode='r',dtype='uint8',shape=(config['pixels'],6))[::64].astype('float64')
    model={key:np.array(value) for key,value in config['model'].items()}
    selected=ncp(z,model)>.95
    audit=dict(sample_stride=64,purpose='post-fit diagnostic only',samples=len(z),pifs=int(selected.sum()),channels=[])
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for b,ax in enumerate(axes):
        x,y=z[:,3+b],z[:,b]
        audit['channels'].append(dict(band=b+1,reference_p1_p50_p99=np.percentile(y,[1,50,99]).tolist(),
            target_p1_p50_p99=np.percentile(x,[1,50,99]).tolist(),pif_target_p1_p50_p99=np.percentile(x[selected],[1,50,99]).tolist(),
            pif_reference_p1_p50_p99=np.percentile(y[selected],[1,50,99]).tolist()))
        ax.scatter(x[::32],y[::32],s=.2,c='#b4bbc4',alpha=.25,rasterized=True)
        ax.scatter(x[selected],y[selected],s=.5,c='#009c79',alpha=.4,rasterized=True)
        xx=np.array([0,255])
        ax.plot(xx,xx*config['gain'][b]+config['offset'][b],color='#cf4d28',lw=1.4)
        ax.set(xlim=(0,255),ylim=(0,255),xlabel='Raw T2 DN',ylabel='T1 DN',title=['Red','Green','Blue'][b])
    fig.suptitle('Gray: all valid-pixel sample; green: NCP > 0.95; red: fitted TLS (no road / GT selection)')
    fig.savefig(EVAL/'pif_regression.png',dpi=160)
    plt.close(fig)
    save(EVAL/'radiometric_qc.json',audit)


def plots():
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False, 'font.size':10})
    ref = pd.read_csv(EVAL/'reference_stations.csv')
    target = pd.read_csv(EVAL/'normalized_stations.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    for ax in axes:
        ax.scatter(ref.x,ref.y,s=.1,c='#b4bbc4',rasterized=True)
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False,style='plain')
        ax.set_xlabel('UTM Easting (m)')
    for flag,color,label in [('recovered','#03965a','Raw 未匹配 → 归一化匹配'),('lost','#df4933','Raw 匹配 → 归一化未匹配')]:
        rows=ref.loc[ref[flag]]
        axes[0].scatter(rows.x,rows.y,s=1.2,c=color,label=label,rasterized=True)
    novelty=target.loc[target.novel]
    axes[1].scatter(novelty.x,novelty.y,s=1,c='#a035ba',label='归一化 T2 未匹配 T1 和 Raw T2',rasterized=True)
    axes[0].set_title('固定 T1 中心线覆盖的恢复 / 丢失')
    axes[1].set_title('新生片段候选（不等于错误）')
    for ax in axes: ax.legend(loc='upper right',markerscale=4)
    fig.savefig(EVAL/'coverage_changes.png',dpi=170)
    plt.close(fig)

    results=[read(BASE/'20250118/latest_result.json'),read(BASE/'20260203/latest_result.json'),read(OUT/'T2/latest_result.json')]
    crs=next(iter(paired_tiles()))[0]
    with rasterio.open(crs) as ds: native=ds.crs
    roads=[gpd.read_file(r['centerlines']).to_crs(native) for r in results]
    transform=Transformer.from_crs('EPSG:32650',native,always_xy=True)
    chosen=[]
    for data, flag, title in [(ref,'recovered','恢复较多区'),(ref,'lost','丢失较多区'),(target,'novel','新生片段较多区')]:
        rows=data.loc[data[flag]].copy()
        if rows.empty: continue
        rows['cell_x']=np.floor(rows.x/250).astype(int)
        rows['cell_y']=np.floor(rows.y/250).astype(int)
        ranked=rows.groupby(['cell_x','cell_y']).length_weight_m.sum().sort_values(ascending=False)
        for cell, length in ranked.items():
            cx,cy=(cell[0]+.5)*250,(cell[1]+.5)*250
            if all(np.hypot(cx-x,cy-y)>250 for x,y,_,_ in chosen):
                chosen.append((cx,cy,title,float(length)))
                break
    fig,axes=plt.subplots(len(chosen),3,figsize=(13,4.2*len(chosen)),squeeze=False,constrained_layout=True)
    pairs=list(paired_tiles())
    for row,(x,y,title,length) in enumerate(chosen):
        lon,lat=transform.transform(x,y)
        left,bottom=transform.transform(x-230,y-230)
        right,top=transform.transform(x+230,y+230)
        bounds=(min(left,right),min(bottom,top),max(left,right),max(bottom,top))
        for col in range(3):
            ax=axes[row,col]
            for a,b in pairs:
                path=[a,b,OUT/'normalized_tiles'/b.name][col]
                with rasterio.open(path) as ds:
                    overlap=(max(ds.bounds.left,bounds[0]),max(ds.bounds.bottom,bounds[1]),min(ds.bounds.right,bounds[2]),min(ds.bounds.top,bounds[3]))
                    if overlap[0]>=overlap[2] or overlap[1]>=overlap[3]: continue
                    w=from_bounds(*overlap,transform=ds.transform).round_offsets().round_lengths()
                    img=ds.read(window=w)
                    wb=rasterio.windows.bounds(w,ds.transform)
                    ax.imshow(img.transpose(1,2,0),extent=(wb[0],wb[2],wb[1],wb[3]))
            for j,color,lw in [(0,'#00ffff',1),(col,'#ff672b',.8)]:
                if col==0 and j==0 and color=='#ff672b': continue
                f=roads[j]
                indices=f.sindex.query(sh.box(*bounds))
                if len(indices): f.iloc[indices].plot(ax=ax,color=color,linewidth=lw)
            ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[1],bounds[3]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{title} | '+['T1 已有','Raw T2 已有','IR-MAD T2 新提取'][col])
    fig.suptitle('青色：T1 中心线；橙色：对应 T2 中心线。范围由一致性指标自动选取，不代表全部错误。')
    fig.savefig(EVAL/'road_hotspots.png',dpi=170)
    plt.close(fig)
    pd.DataFrame(chosen,columns=['utm_x','utm_y','selection','selected_cell_length_m']).to_csv(EVAL/'hotspots.csv',index=False)


def build_report():
    m,c=read(EVAL/'metrics.json'),read(OUT/'normalization.json')
    raw,norm=m['roads']['Raw_T2'],m['roads']['Normalized_T2']
    def pair(path,scale=1,decimals=2):
        def get(d):
            for key in path: d=d[key]
            return 'N/A' if d is None else f'{d*scale:.{decimals}f}'
        return get(raw),get(norm)
    rows=[]
    for label,path,scale,d in [
        ('T1 中心线覆盖 / 匹配率（3m, %）',['reference_longitudinal_coverage','3'],100,2),
        ('T1 中心线覆盖（1m / 严格, %）',['reference_longitudinal_coverage','1'],100,2),
        ('T1 中心线覆盖（5m / 宽松, %）',['reference_longitudinal_coverage','5'],100,2),
        ('稳定代理道路横向偏移中位数（m）',['stable_lateral_offset_m','median'],1,3),
        ('稳定代理道路横向偏移 P90（m）',['stable_lateral_offset_m','p90'],1,3),
        ('道路总长度（km）',['total_length_m'],.001,2),
        ('道路对象数量',['objects'],1,0),
        ('T2 中心线有 T1 匹配支持（%）',['target_supported_by_T1_fraction'],100,2),
        ('T2 未匹配 T1 长度（km）',['unsupported_by_T1_length_m'],.001,2),
        ('同时未匹配 T1 / 另一 T2 的片段（km）',['novel_vs_T1_and_other_T2_m'],.001,2),
        ('稳定点宽度偏差中位数，T2−T1（m）',['stable_width','bias_m','median'],1,2),
        ('稳定点绝对宽差中位数（m）',['stable_width','absolute_difference_m','median'],1,2),
        ('稳定点绝对宽差 >2m（%）',['stable_width','fraction_absolute_over_2m'],100,2),
        ('稳定整条道路绝对中位宽差（m）',['stable_whole_road_width','absolute_median_bias_m','median'],1,2)]:
        a,b=pair(path,scale,d); rows.append(f'| {label} | {a} | {b} |')
    for kind,label in [('raw_molra_threshold_0_5','原始 SAM-MoLRA 面 IoU'),('enhanced_molra','增强 SAM-MoLRA 面 IoU'),('final_road_surface','最终道路面 IoU')]:
        data=m['surfaces']['overlap'][kind]
        a,b=(data[k]['iou'] for k in ('Raw_T2','Normalized_T2'))
        rows.append(f'| {label}（%） | {100*a:.3f} | {100*b:.3f} |')
    assessment=read(EVAL/'assessment.json') if (EVAL/'assessment.json').exists() else dict(conclusion='指标已生成，结论待结合局部图复核。',notes=[])
    recover=m['reference_recovery']
    timing=read(OUT/'extract_audit.json')
    stage_table='\n'.join(f"| {stage['stage']} | {stage['elapsed_seconds']:.1f} |" for stage in m['timings']['normalized_extraction_stages'])
    outputs=c['outputs']
    total=sum(o['valid_pixels'] for o in outputs[:-1])
    clips=np.array([o['clipped_low_high_per_band'] for o in outputs[:-1]]).sum(0)/total*100
    text=f'''# IR-MAD/PIF + TLS：20250118 → 20260203

{assessment['conclusion']}

本轮只重新提取 Normalized T2。T1、Raw T2 已有成果直接复用；未运行 Fast2、变化检测、GT 校正或 Temporal。

## 结果

| 指标（对同一 T1） | Raw T2 已有 | IR-MAD T2 新结果 |
|---|---:|---:|
{chr(10).join(rows)}

T1 道路总长 {m['roads']['T1']['total_length_m']/1000:.2f} km，{m['roads']['T1']['objects']} 个对象；共同可观测采样道路长度 {m['reference_observed_length_m']/1000:.2f} km。
归一化恢复原 Raw 未匹配的 T1 道路约 {recover['recovered_m']/1000:.2f} km，同时丢失原已匹配道路 {recover['lost_m']/1000:.2f} km，净恢复 {(recover['recovered_m']-recover['lost_m'])/1000:.2f} km。
稳定代理集合包含 {m['stable_proxy']['road_count']} 条道路、{m['stable_proxy']['matched_length_m']/1000:.2f} km 匹配采样长度。

{chr(10).join('- '+note for note in assessment['notes'])}

## 实验与评价边界

- 原始影像 CRS/分辨率不一致；复用 baseline 已有 8 对严格共网格、未作辐射归一化的分析瓦片。无新增 warp，所有瓦片共用 T1 回归参数。另存原生 T2 网格 TIFF。
- 全部 {c['pixels']:,} 个共同有效像素参与 IR-MAD，{c['selected_iteration']} 次收敛，最大 30 次，相关系数 delta 阈值 0.01。PIF NCP >0.95，{c['pif_pixels']:,} 个（{c['pif_fraction']*100:.3f}%）；没有尝试 0.99 或基于道路结果选参数。
- RGB TLS gain={np.round(c['gain'],5).tolist()}，offset={np.round(c['offset'],5).tolist()}。PIF 相关系数约 {np.round(c['pif_correlation'],4).tolist()}。分析瓦片有效像素上限裁剪比例 RGB={np.round(clips[:,1],2).tolist()}%，下限={np.round(clips[:,0],3).tolist()}%。高 PIF 相关不能保证所有地物符合单一线性响应。
- 使用原 baseline 冻结生产代码、同一模型和推理配置；道路 Auto profile 自然响应影像。原始副本、baseline 分析输入与成果哈希未变；normalized 网格、nodata、逐波段掩膜及无效像元检查通过。导出时修复逐波段 nodata 掩膜写法，保留失败副本，未重复 IR-MAD 或 baseline。
- 评价仅使用输入验证范围与共同有效像元；固定 T1 每约 2m 采样，距离≤3m、局部方向差≤30°。稳定代理要求 T1 路段≥50m、两种 T2 都覆盖≥70%，用于同一位置偏移和宽度比较；该集合偏向共同提取成功道路，覆盖变化另行全量统计。
- SAM-MoLRA 原始概率阈值≥0.5；增强面和最终导出面分别评价，防止 fallback 面掩盖模型失败。宽度来自提取流程自然输出，不启动变化检测或宽度背景校正。
- T1 不是 GT，少量真实变化、未恢复的漏检和地物变化均可能影响一致性。“未匹配”“新生候选”不能直接称为错误或假变化；本轮不提供 GT 精度或假变化对象数。未扩展到其他时期，未接入生产。

## 耗时

- 全像素配对缓存：{c['timings']['cache_seconds']:.1f}s；IR-MAD 迭代：{c['timings']['irmad_fit_seconds']:.1f}s。
- 本次归一化含写出、验证和导出修复墙钟：{c['timings'].get('normalization_elapsed_including_export_fix_seconds',c['timings'].get('normalization_total_seconds',0)):.1f}s。首次导出错误前 PIF/TLS/写出分项未完整保存，不将差额解释为纯算法时间。
- 新 T2 提取及哈希复核：{timing['elapsed_seconds']:.1f}s；SAM-MoLRA 各瓦片记录合计 {m['surfaces']['inference_audit']['Normalized_T2']['seconds']:.1f}s（包含在提取阶段内，不额外相加）。
- 对比评价：{m['timings']['evaluation_seconds']:.1f}s；baseline 无新增推理耗时。

| 新 T2 提取阶段 | 秒 |
|---|---:|
{stage_table}

## 图与成果

![覆盖恢复、丢失及新生候选](coverage_changes.png)

![自动选取的恢复、丢失和新生候选局部](road_hotspots.png)

数值详见 `metrics.json`；固定站点见 `reference_stations.csv`，整段宽差见 `stable_road_width.csv`。新成果为 `../T2/runs/roads/products/roads.gpkg`；推理影像为 `../normalized_tiles/`，原生网格影像为 `../normalized_native/20260203.tif`。

方法参考：[ArrNorm](https://github.com/SMByC/ArrNorm)、[IR-MAD 实现](https://github.com/SMByC/ArrNorm/blob/master/core/iMad.py)、[TLS 辅助函数](https://github.com/SMByC/ArrNorm/blob/master/core/auxil/auxil.py)。本实验独立实现数值流程，未调用生成式方法、直方图匹配或颜色风格迁移。
'''
    (EVAL/'REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    radiometric_diagnostics()
    plots()
    build_report()
