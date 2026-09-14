"""Offline evaluation report and evidence figures; never imported by Auto."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely import make_valid,union_all,from_wkt
from engine.fast_auto_change import WindowedProbability
from engine.fast_patch_verification import PatchVerifier
from engine.fast_pipeline import _read_fast_change_layer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);args=parser.parse_args()
    output=args.job/'_profiling'/'fast2_patch_verification'
    data=json.loads((output/'comparison.json').read_text(encoding='utf8'))
    manifest=json.loads((args.job/'pipeline_result.json').read_text(encoding='utf8'))
    candidates=[];primary=Counter();all_reasons=Counter();states=Counter();losses=[];patch_time=0.;control_counts=[]
    types=['added','removed','widened','narrowed']
    for pair in data:
        directory=output/pair['pair'];audit=json.loads((directory/'patch_verification.json').read_text(encoding='utf8'))
        states.update(r['state'] for r in audit['candidates'])
        for row in audit['candidates']:
            all_reasons.update(row['reasons'])
            if row['reasons']:primary[row['reasons'][0]]+=1
        patch_time+=audit['counts']['verification_seconds'];control_counts.append(audit['calibration']['count'])
        before={(r['gt_id'],r['predicted_type']):r for r in pair['before']['objects']}
        for row in pair['after']['objects']:
            old=before.get((row['gt_id'],row['predicted_type']))
            if old and row['area_coverage']<old['area_coverage']-1e-6:
                losses.append(dict(pair=pair['pair'],gt_id=row['gt_id'],kind=row['predicted_type'],
                                   before=old['area_coverage'],after=row['area_coverage']))
        truth=gpd.read_file(args.job/'_profiling'/'fast_v2_tuning'/pair['pair']/'gt_upstream.gpkg')
        gt=union_all(make_valid(truth.geometry.to_numpy()))
        detectable=union_all(make_valid(truth.loc[truth.upstream_state=='detectable'].geometry.to_numpy()))
        result,_=pickle.loads((directory/'analysis.pkl').read_bytes())
        for i,row in enumerate(result[0]):
            if not row.get('v2_publish_before_patch'):continue
            geom=make_valid(row['geometry']);hit=geom.intersection(gt).area;detected=geom.intersection(detectable).area
            candidates.append(dict(pair=pair['pair'],index=i,row=row,hit=hit,detectable_hit=detected))
    selected=[]
    groups=[('gt_supported_retained',lambda x:x['detectable_hit']>0 and x['row']['v2_publish'],lambda x:x['detectable_hit']),
        ('added_fluctuation',lambda x:x['hit']==0 and x['row']['change_typ']=='added' and x['row']['v2_patch_state']=='extraction_fluctuation',lambda x:x['row'].get('v2_patch_image_ncc',0)),
        ('removed_fluctuation',lambda x:x['hit']==0 and x['row']['change_typ']=='removed' and x['row']['v2_patch_state']=='extraction_fluctuation',lambda x:x['row'].get('v2_patch_image_ncc',0)),
        ('width_displacement',lambda x:'surface_lateral_displacement' in x['row']['v2_patch_reasons'],lambda x:x['row']['length_m']),
        ('width_unconfirmed',lambda x:x['row']['v2_patch_state']=='unconfirmed_width',lambda x:x['row']['length_m']),
        ('gt_supported_rejected',lambda x:x['detectable_hit']>0 and not x['row']['v2_publish'],lambda x:x['detectable_hit'])]
    for name,keep,score in groups:
        items=[c for c in candidates if keep(c)]
        if items:selected.append((name,max(items,key=score)))
    examples=[]
    for name,item in selected:
        pair=next(p for p in manifest['change_results'] if f"{p['grid']}_{p['before_period']}_{p['after_period']}"==item['pair'])
        payloads=[next(p for p in manifest['auto_period_results'] if p['grid']==pair['grid'] and p['period']==pair[key])
                  for key in ('before_period','after_period')]
        crs=32650;scenes=[];verifier=None
        try:
            for p in payloads:
                valid=union_all(_read_fast_change_layer(p,'valid_observation').to_crs(crs).geometry)
                scenes.append(SimpleNamespace(crs=crs,valid=valid,probability=WindowedProbability(p['road_probability'],crs)))
            verifier=PatchVerifier(scenes,payloads);row=item['row']
            patch=verifier.patch(from_wkt(row['axis_wkt']),max(row['width_bef'],row['width_aft']))
            truth=gpd.read_file(args.job/'_profiling'/'fast_v2_tuning'/item['pair']/'gt_upstream.gpkg').to_crs(crs)
            left,bottom,right,top=patch['bounds'];extent=(left,right,bottom,top)
            fig,axs=plt.subplots(1,4,figsize=(15,4),layout='constrained')
            for side in range(2):
                image=np.moveaxis(patch['periods'][side]['rgb'],0,-1)
                lo=np.nanpercentile(image,2);hi=np.nanpercentile(image,98)
                axs[side].imshow(np.nan_to_num(np.clip((image-lo)/max(hi-lo,1),0,1)),extent=extent)
                axs[side].set_title(f"T{side+1}: {pair['before_period' if side==0 else 'after_period']}")
            sa,sb=[p['surface'] for p in patch['periods']]
            overlay=np.full((*sa.shape,3),.92);overlay[(sa>0)&(sb>0)]=[.3,.3,.3]
            overlay[(sa>0)&~(sb>0)]=[.1,.4,1];overlay[(sb>0)&~(sa>0)]=[1,.25,.15]
            axs[2].imshow(overlay,extent=extent);axs[2].set_title('MoLRA surface: T1 blue / T2 red')
            delta=patch['periods'][1]['probability']-patch['periods'][0]['probability']
            axs[3].imshow(delta,extent=extent,cmap='coolwarm',vmin=-.5,vmax=.5);axs[3].set_title('Road probability: T2 - T1')
            for ax in axs:
                gpd.GeoSeries([row['geometry']],crs=crs).boundary.plot(ax=ax,color='#ffd900',linewidth=1.2)
                nearby=truth.cx[left:right,bottom:top]
                if len(nearby):nearby.boundary.plot(ax=ax,color='#ef00ff',linewidth=1.1)
                ax.set_xlim(left,right);ax.set_ylim(bottom,top);ax.set_aspect('equal');ax.set_axis_off()
            fig.suptitle(f"{name} | {row['change_typ']} | {row['v2_patch_state']}\nYellow=candidate; magenta=GT (offline only)",fontsize=11)
            path=output/(name+'.png');fig.savefig(path,dpi=160);plt.close(fig)
            examples.append(dict(name=name,pair=item['pair'],candidate=item['index'],image=path.name,
                                 reasons=row['v2_patch_reasons'],gt_overlap_m2=item['hit']))
        finally:
            if verifier:verifier.close()
            for scene in scenes:scene.probability.close()
    totals={side:{k:sum(p[side]['types'][k]['count'] for p in data) for k in types} for side in ('before','after')}
    no_gt={side:sum(sum(p[f'{side}_no_gt_intersection'].values()) for p in data) for side in ('before','after')}
    summary=dict(totals=totals,no_gt_intersection=no_gt,states=dict(states),primary_veto=dict(primary),
        all_veto=dict(all_reasons),patch_seconds=patch_time,control_counts=control_counts,
        analysis_seconds=sum(p['analysis_seconds'] for p in data),pipeline_seconds=sum(p['total_seconds'] for p in data),
        restoration_cancelled=sum(p.get('restoration_cancelled',0) for p in data),
        verdict_replay_seconds=sum(p.get('verdict_replay_seconds',0) for p in data),
        verdict_republication_seconds=sum(p.get('verdict_republication_seconds',0) for p in data),
        gt_losses=losses,examples=examples)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# Fast2 候选级双时相影像验证','',
        '使用 7 期原始 Auto 缓存、6 个变化对。Fast2 先生成候选，再用稳定匹配道路的经验分位数校准局部证据。未训练模型、未重新推理，GT 仅在成果写出后用于评价。',
        '每个候选/控制道路读取不超过约 192×192 的局部影像网格。复用 SAMRoad probability 和已缓存 SAM‑MoLRA enhanced mask；RGB 计算归一化梯度、方向边缘及相关性，否决必须同时满足路体内部纹理一致，不能只依赖稳定背景。encoder feature 未导出，跳过。',
        '宽变比较三个纵向区域内全部掩膜像素的左右分位数边界，不重新沿线测宽。宽度 profile、候选几何均不改写。原候选及否决原因留在内部审计。','',
        '本轮优先抑制假变化，取消影像与模型证据冲突时恢复候选的保护逻辑。土路/施工带变成成型道路等模糊对象允许被原影像验证规则否决；不为保留个别对象恢复大量候选。保留路体内部纹理和同轨边界检查。','',
        '|类别|验证前正式对象|验证后正式对象|','|---|---:|---:|']
    lines += [f"|{k}|{totals['before'][k]}|{totals['after'][k]}|" for k in types]
    lines += ['',f"验证区内与任意 GT 无面积交集对象：{no_gt['before']} → {no_gt['after']}。",f'局部候选审核状态：{dict(states)}。正式对象经过网络组装，数量与局部候选不同。','',
              '|证据原因|首要否决（互斥）|全部命中（可重叠）|','|---|---:|---:|']
    lines += [f'|{key}|{primary[key]}|{all_reasons[key]}|' for key in all_reasons]
    lines += ['', '|变化对|四类：前→后（A/R/W/N）|无 GT 交集：前→后|局部验证秒|分析秒|', '|---|---|---:|---:|---:|']
    for p in data:
        before='/'.join(str(p['before']['types'][k]['count']) for k in types);after='/'.join(str(p['after']['types'][k]['count']) for k in types)
        lines.append(f"|{p['pair']}|{before} → {after}|{sum(p['before_no_gt_intersection'].values())} → {sum(p['after_no_gt_intersection'].values())}|{p['counts']['v2_patch_verification_seconds']:.2f}|{p['analysis_seconds']:.2f}|")
    lines += ['',f"新增验证累计 {patch_time:.2f}s（包括控制道路校准）；分析累计 {summary['analysis_seconds']:.2f}s；缓存读取、分析和成果导出累计 {summary['pipeline_seconds']:.2f}s。示例制图耗时单列，不计入验证。",'',
        f"上述为完整影像验证实测。随后使用缓存撤销 {summary['restoration_cancelled']} 个局部候选的冲突恢复，裁决更新耗时 {summary['verdict_replay_seconds']:.4f}s；重新协调/导出/评价额外耗时 {summary['verdict_republication_seconds']:.2f}s，未重复候选生成和影像读取。",'',
        '## 可检测 GT 保留情况','',
        'GT 可检测性沿用原始 Auto 输入的离线分类，不以 GT 调整推理结果。以下为相较本轮前基线新增的覆盖损失；前轮已经漏掉的对象不会由筛选器恢复。','',
        '|变化对|GT 对象|类型|覆盖前→后|','|---|---|---|---:|']
    lines += [f"|{r['pair']}|{r['gt_id']}|{r['kind']}|{r['before']:.1%} → {r['after']:.1%}|" for r in losses]
    if not losses:lines.append('|全部变化对|—|—|未发现新增覆盖损失|')
    lines += ['', '当前无交集数是空间代理指标，不能将每个被否决对象都认定为真实误检。RGB 传统特征会受季节、阴影和相似地物影响。缺失缓存/曲线边界无法可靠比较时记录内部 QA；有效掩膜中没有可比较道路边界的宽变不予确认。',
              '## Patch 示例','', '选择规则：离线 GT 相交且保留的候选、无 GT 相交且被否决的候选、宽变位移/未确认案例，以及有 GT 支持但被否决的冲突案例。不把“无 GT 相交”直接等同于人工确认的假变化。']
    for e in examples:lines += ['',f"### {e['name']} — {e['pair']} / candidate {e['candidate']}",e['reasons'] or '保留',f"![{e['name']}]({e['image']})"]
    lines += ['', '实现文件：engine/fast_patch_verification.py；接入 fast_auto_v2.py 与 fast_auto_change.py；Fast2 内部版本标记更新。CLI、GT 后验、Temporal 和模型链未改。',
        '直接相关及 Fast pipeline 回归测试通过；实际六对运行强制禁止 Fast1、RoadScene.match 和精确重新测宽。结果位于本实验目录，不覆盖正式项目成果。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
