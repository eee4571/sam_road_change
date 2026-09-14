"""Summarize an explicitly requested offline Fast2 tuning experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import geopandas as gpd
import pandas as pd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();root=args.directory
    rows=json.loads((root/'final_report.json').read_text(encoding='utf-8'))
    assert len(rows)==6,'Report requires every requested pair'
    kinds=['added','removed','widened','narrowed'];old=Counter();new=Counter();fpold=fpnew=0;objects=[]
    text=['# Fast2 多期离线参数调整','',
        '生产 Auto 不读取 GT。本次只读取 run_20260910_193306 的 auto_period_results、原始概率缓存和现有多期 GT；没有重跑模型、GT 校正或 Temporal，也没有覆盖现有正式任务成果。',
        '', '## 当前参数', '',
        '- 稳定道路的位置容忍度为输入 tolerance 的 1.5 倍（当前 3 → 4.5 m）；只放宽存在性覆盖，宽变对应关系不扩大到相邻车道。',
        '- 新增/灭失至少 32 m，并要求确认缺失，或本期 surface 覆盖 ≥55%、对期 ≤15%、两期有效且对期没有概率支持。宽松匹配不等于宽松发布。',
        '- 宽变至少有连续 72 m 的明显宽差，同时超过原宽差门槛的 2 倍。当前配置相当于同时达到 4 m 与较大宽度的 40%。',
        '- 未通过发布门槛的局部候选保留在内部 diagnostics。规则道路走廊、网络组装、模型、GT 后验和 Temporal 未修改。',
        '- 仍是 segment/interval 运算，0 个 station 扫描、0 个旧 analyzer fallback。',
        '', '## 六对最终变化数量（调整前 → 调整后）','',
        '| 变化对 | added | removed | widened | narrowed |',
        '|---|---:|---:|---:|---:|']
    false_rows=[];upstream=Counter()
    for row in rows:
        a=row['variants']['previous']['final']['types'];b=row['variants']['balanced']['final']['types']
        key=row['pair'];label=key.split('_',1)[1]
        text.append('|'+label+'|'+'|'.join(f"{a[k]['count']} → {b[k]['count']}" for k in kinds)+'|')
        for k in kinds:old[k]+=a[k]['count'];new[k]+=b[k]['count']
        aa=sum(a[k]['no_gt_overlap'] for k in kinds);bb=sum(b[k]['no_gt_overlap'] for k in kinds)
        fpold+=aa;fpnew+=bb;upstream.update(row['upstream'])
        false_rows.append((label,aa,bb,row['upstream'],row['variants']['balanced']['wall_seconds']))
        gt=gpd.read_file(root/key/'gt_upstream.gpkg').drop(columns='geometry')
        for variant in ('previous','balanced'):
            detail=pd.DataFrame(row['variants'][variant]['final']['objects'])
            if not detail.empty:
                joined=detail.merge(gt,on='gt_id',how='left');joined['pair']=key;joined['variant']=variant
                objects.append(joined)
    text.append('|合计|'+'|'.join(f'{old[k]} → {new[k]}' for k in kinds)+'|')
    text+=['',f'正式对象总数：{sum(old.values())} → {sum(new.values())}，减少 {100*(1-sum(new.values())/sum(old.values())):.1f}%。',
        '', '## 假变化代理统计与上游可检测性','',
        '| 变化对 | 不与对应 GT 相交：前 → 后 | 上游不可检测 | 部分提取 | 可检测 | 新版分析秒数 |',
        '|---|---:|---:|---:|---:|---:|']
    for label,a,b,u,wall in false_rows:
        text.append(f"|{label}|{a} → {b}|{u.get('upstream_undetectable',0)}|{u.get('partial_upstream',0)}|{u.get('detectable',0)}|{wall:.2f}|")
    text+=['',f'无对应 GT 空间交集的对象合计：{fpold} → {fpnew}，减少 {100*(1-fpnew/fpold):.1f}%。仍有明显剩余假变化，不能把减少数量解释为已经达到高精度。',
        '', '统计口径：',
        '- 先裁到任务 validation area，并使用对应变化类型 GT；BHBM=3 不区分拓宽/变窄。凡触及上游不可检测/部分提取 GT 的预测，排除出用于调参的假变化统计，避免用上游漏提惩罚检测。',
        '- 上游检测率使用 GT 道路轴与原始自动中心线的同向纵向覆盖（5 m 容忍）。Added 检查后期、Removed 检查前期、Width Changed 检查两期。必要期覆盖 <10% 为不可检测、10%–50% 为部分提取、≥50% 为可检测。它是离线诊断分类，不是正式指标的新定义。',
        f"- 共 {sum(upstream.values())} 个 GT：不可检测 {upstream['upstream_undetectable']}、部分提取 {upstream['partial_upstream']}、可检测 {upstream['detectable']}。逐对象几何和两期覆盖见每对 gt_upstream.gpkg。",
        '- 可检测不等于能确认变化：无效观测边界、两期已存在道路/路面支持的对象仍不能靠 GT 制造变化。详见 gt_object_comparison.csv 的逐对象覆盖及最近距离。',
        '- BHBM=3 是完整变化道路面，正式宽变输出差值带；二者面积覆盖仅用于描述空间关系，不据此强行扩大变化带。',
        '- 评价副本在米制重投影后做 make_valid 和 1 mm 拓扑网格处理，避免微小环自交；不修改成果几何，不修改现有正式评价公式。',
        '', '## 参数迭代与限制', '',
        '比较了原参数、48 m/1.5 倍宽差、72 m/2 倍宽差以及最终分类型组合。单纯要求概率缺失、或把存在性最短段提高到 48–72 m，会损失真实分段道路，因此最终保留 32 m 存在性门槛和独立的 surface 缺失证据。宽变继续使用更保守的 72 m/2 倍门槛。所有变化对使用同一组全局参数，没有按 GT 对象或区域写特殊规则。',
        '', '本次是同一真实项目六对的离线调参结果，没有独立留出项目，不代表跨区域泛化精度。绝大部分宽变 GT 缺少必要期的原始自动道路；不能把这部分当作已验证的宽变检出能力。',
        '', '## 文件与验证', '',
        '- 生产修改：code/engine/fast_auto_v2.py。外部 CLI/GUI 接口不变。',
        '- 离线脚本：code/tests/tune_fast_v2_offline.py、report_fast_v2_tuning.py。正式推理不导入它们。',
        '- 15 项 Fast2 测试、39 项原 Fast/概率/并发/规则几何相关测试通过。测试包括弱宽差不发布、短段不发布、时间反转对称、surface 明确缺失、禁止读取 GT 和禁止旧 station fallback。',
        '- 每对 previous/road_changes.shp 是旧 Fast2；balanced/road_changes.shp 是本轮结果，可直接加载比较。正式任务已有结果未覆盖；后续通过原 GUI/CLI 重跑会使用新参数。',
        '- final_report.json 含各类数量、低重叠/零重叠对象数、面积覆盖及最近位置距离。tuning_report.json 保留首轮三组参数结果。', '']
    (root/'FAST2_TUNING_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    if objects:pd.concat(objects,ignore_index=True).to_csv(root/'gt_object_comparison.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(11,15))
    for index,(pair_index,gt_id,kind) in enumerate([(1,'3','removed'),(2,'2','added'),(2,'3','added'),(5,'1','added')]):
        folder=root/rows[pair_index]['pair'];truth=gpd.read_file(folder/'gt_upstream.gpkg')
        target=truth.loc[truth.gt_id==gt_id];left,bottom,right,top=target.total_bounds
        margin=max(15.,.1*max(right-left,top-bottom))
        for column,variant in enumerate(('previous','balanced')):
            ax=axes[index,column];frame=gpd.read_file(folder/variant/'road_changes.shp').to_crs(truth.crs)
            local=frame.cx[left-margin:right+margin,bottom-margin:top+margin]
            local=local.loc[local.change_typ==kind]
            if len(local):local.plot(ax=ax,color='#df718a',edgecolor='#99344d',linewidth=.6)
            target.boundary.plot(ax=ax,color='#226bc0',linewidth=1.5)
            ax.set(xlim=(left-margin,right+margin),ylim=(bottom-margin,top+margin))
            ax.set_title(f"{rows[pair_index]['pair'].split('_',1)[1]} / {kind} / GT {gt_id}\n{variant}; blue=GT, pink=Auto")
            ax.set_aspect('equal');ax.set_axis_off()
    fig.tight_layout();fig.savefig(root/'GT_LOCATION_COMPARISON.png',dpi=150);plt.close(fig)
    print(dict(before=dict(old),after=dict(new),unmatched_before=fpold,unmatched_after=fpnew,upstream=dict(upstream)))


if __name__=='__main__':main()
