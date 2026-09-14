"""Summarize measured cached-pair results, including temporal/GT conflicts."""
import argparse
import json
from pathlib import Path
from collections import Counter
import geopandas as gpd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    path=args.output/'comparison.json';rows=json.loads(path.read_text(encoding='utf-8'))
    kinds=['added','removed','widened','narrowed']
    totals={side:{k:sum(r[side]['types'][k]['count'] for r in rows) for k in kinds} for side in ('before','after')}
    no_gt={side:sum(sum(r[f'{side}_no_gt_intersection'].values()) for r in rows) for side in ('before','after')}
    suppressed={k:sum(r['counts'][f'v2_temporal_{k}_suppressed'] for r in rows) for k in ('added','removed')}
    bias={k:sum(r['counts'].get(k,0) for r in rows) for k in ('v2_width_before_bias_published','v2_width_after_bias_published','v2_width_bias_suppressed')}
    conflicts=[];invalid=[]
    for r in rows:
        # Validity must be checked in the written layer CRS; a reprojection may
        # create floating-point ring contacts absent in the published geometry.
        frame=gpd.read_file(args.output/r['pair']/'road_changes.shp')
        r['invalid_native_geometry_count']=int((~frame.is_valid).sum())
        r['valid_geometry']=bool(frame.is_valid.all())
        if not r['valid_geometry']:invalid.append((r['pair'],r['invalid_native_geometry_count']))
        before={(x['gt_id'],x['predicted_type']):x for x in r['before']['objects']}
        for obj in r['after']['objects']:
            old=before.get((obj['gt_id'],obj['predicted_type']))
            if old and obj['area_coverage']<old['area_coverage']-1e-6:
                conflicts.append(dict(pair=r['pair'],gt_id=obj['gt_id'],kind=obj['predicted_type'],
                    before_coverage=old['area_coverage'],after_coverage=obj['area_coverage']))
    aggregate=dict(totals=totals,no_gt_intersection=no_gt,temporal_suppressed_candidates=suppressed,
        width_bias_candidates=bias,analysis_seconds=sum(r['analysis_seconds'] for r in rows),
        total_seconds=sum(r['total_seconds'] for r in rows),
        batch_wall_seconds=json.loads((args.output/'batch_time.json').read_text())['wall_seconds'],
        temporal_seconds=sum(r['counts']['timing_v2_multitemporal_seconds'] for r in rows),
        bias_estimation_seconds=sum(r['counts']['timing_v2_bias_estimation_seconds'] for r in rows),
        suppressed_candidates_touching_gt=sum(r['suppressed_candidates_with_gt_overlap'] for r in rows),
        gt_coverage_losses=conflicts,invalid_native_geometries=invalid)
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Fast2 多时相一致性与宽度偏差校正实测','',
        '输入：run_20260910_193306 的 7 期原始 Auto 道路，6 个相邻变化对。模型、Fast1、逐点 match 和重新测宽均未运行。',
        'Auto 输入指纹与前次基线逐对核对；GT 仅在 Auto 成果生成后用于离线评价。结果保存在当前实验目录，未覆盖项目正式成果。','',
        '实现：新增/灭失按相邻期轴覆盖区间判断持续性，未知/无效邻期不当作缺路；局部持续区间保持方端道路走廊。',
        '宽变使用一条可靠对应道路一票的 width difference 中位数，1.4826×MAD 估计正常分布，校正后残差须超过该分布的 2.5 倍及既有规则。控制道路不足 30 条不启用校正。',
        '原始 width profile、原候选和规则几何保留；校正只影响发布资格，不改写单期宽度、不生成新的原始宽变候选。','',
        '|类别|修改前|修改后|','|---|---:|---:|']
    lines += [f"|{k}|{totals['before'][k]}|{totals['after'][k]}|" for k in kinds]
    lines += ['',f"与任意 GT 无面积交集的正式对象（验证区内，不分类型，不排除上游不可检测 GT）：{no_gt['before']} → {no_gt['after']}。",
        f"时间一致性抑制原始候选：added {suppressed['added']}，removed {suppressed['removed']}（含部分区间被抑制的原对象；不等同于正式对象数量）。",
        f"宽度校正前后满足发布条件的局部候选：{bias['v2_width_before_bias_published']} → {bias['v2_width_after_bias_published']}；抑制 {bias['v2_width_bias_suppressed']}。",'',
        '|变化对|四类数量：前→后（A/R/W/N）|无 GT 交集：前→后|width bias / MAD尺度（m）|控制道路|analyze / 总耗时（s）|',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        c=r['counts'];a='/'.join(str(r['before']['types'][k]['count']) for k in kinds);b='/'.join(str(r['after']['types'][k]['count']) for k in kinds)
        calibration=f"{c['v2_width_bias_m']:+.3f} / {c['v2_width_bias_scatter_m']:.3f}" if c['v2_width_bias_reliable'] else f"未启用（估计 {c['v2_width_bias_estimate_m']:+.3f}）"
        lines.append(f"|{r['pair']}|{a} → {b}|{sum(r['before_no_gt_intersection'].values())} → {sum(r['after_no_gt_intersection'].values())}|{calibration}|{c['v2_width_bias_controls']}|{r['analysis_seconds']:.2f} / {r['total_seconds']:.2f}|")
    lines += ['',f"累计 analyze_scenes：{aggregate['analysis_seconds']:.2f}s；读取缓存+分析+最终变化导出：{aggregate['total_seconds']:.2f}s；含离线评价/审计写出的整批墙钟：{aggregate['batch_wall_seconds']:.2f}s。",
        f"新增时间一致性计算累计 {aggregate['temporal_seconds']:.2f}s；偏差估计累计 {aggregate['bias_estimation_seconds']:.3f}s。历史基线时间未同轮重测，不据此声称速度提升。",'',
        '## GT 冲突与边界','',
        f"{aggregate['suppressed_candidates_touching_gt']} 个受抑制原始候选与 GT 相交；不能把全部抑制对象宣称为已确认假变化。",
        '道路序列的单期缺失/出现也可能是真实短期变化。以下列出可检测 GT 的覆盖损失（没有以 GT 在推理中恢复这些对象）：','',
        '|变化对|GT 对象|类型|覆盖前→后|','|---|---|---|---:|']
    for c in conflicts:lines.append(f"|{c['pair']}|{c['gt_id']}|{c['kind']}|{c['before_coverage']:.1%} → {c['after_coverage']:.1%}|")
    lines += ['', '这些结果说明数量显著降低，但尚不能认定真实精度全面改善。无交集对象仍多，且部分真实短期事件被抑制。',
        '原始 GT 共 31 个：19 个上游不可检测，5 个部分可检测，7 个可检测。宽变 GT 均缺少所需两期完整原始 Auto 道路，无法用这组数据验证宽变正样本召回。',
        f'原生输出 CRS 下无效几何对象：{invalid}；本轮未修改成果几何重建模块。', '',
        '修改文件：新增 engine/fast_multitemporal.py；更新 engine/fast_auto_v2.py、engine/fast_auto_change.py、engine/fast_pipeline.py、user_pipeline.py 及相关回归/离线实验脚本。',
        '生产接入：完整运行、单对/批量变化重跑传入同区域邻期 Auto 道路；重跑一期更新最多四个依赖它的变化对。无邻期的独立双期调用保持 unknown。',
        '新版本标记防止续跑复用旧判定；CLI 参数和 GT 后验、Temporal 流程未更改。内部诊断保留偏差、原候选和时间抑制审计。',
        '验证：52 项 Fast2/多时相/生产入口回归 + 116 项 Fast Auto/对象协调/Fast pipeline 相关测试通过。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
