"""Offline six-pair audit of three-state verdicts; never used by inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path)
    parser.add_argument('--tiers',action='store_true');args=parser.parse_args()
    profiling=args.job/'_profiling';output=profiling/('fast2_publication_tiers' if args.tiers else 'fast2_raw_tristate')
    levels=Counter();conditions=Counter();primary_conditions=Counter();grade_by_kind=Counter()
    data=json.loads((output/'comparison.json').read_text(encoding='utf8'))
    states=Counter();veto=Counter();qa=Counter();transitions=Counter();old_rejected=Counter()
    recovered_uncertain=0;recovered_change=0;previous_kept_vetoed=0;geometry=[];features=[]
    kinds=('added','removed','widened','narrowed')
    # NaNs are missing observations, not unequal re-extracted features.
    def equal(a,b):
        if isinstance(a,float) and isinstance(b,float) and a!=a and b!=b:return True
        if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        return a==b
    for pair in data:
        key=pair['pair'];before=profiling/('fast2_raw_tristate' if args.tiers else 'fast2_raw_image')/key;after=output/key
        old=json.loads((before/'patch_verification.json').read_text(encoding='utf8'))
        new=json.loads((after/'patch_verification.json').read_text(encoding='utf8'))
        previous={r['candidate']:r for r in old['candidates']}
        old_rows=pickle.loads((before/'analysis.pkl').read_bytes())[0][0]
        rows=pickle.loads((after/'analysis.pkl').read_bytes())[0][0]
        assert len(old_rows)==len(rows)
        for a,b in zip(old_rows,rows):
            assert a['geometry'].wkb==b['geometry'].wkb
            for k in ('axis_wkt','change_typ','width_bef','width_aft'):assert a[k]==b[k]
        geometry.append(dict(pair=key,candidates=len(rows),unchanged=True))
        assert len(previous)==len(new['candidates'])
        assert equal(old['calibration'],new['calibration']),'Calibration must remain unchanged'
        for r in new['candidates']:
            prior=previous[r['candidate']];state=r['state'];states[state]+=1
            for k in ('left','right','ncc','core_ncc','ssim','ring_ssim','anomaly','hog_distance',
                      'road_scores','score_delta','registration','resolution','valid'):
                assert equal(prior[k],r[k]),(key,r['candidate'],k)
            transitions[prior['state']+' -> '+state]+=1
            if prior['state']=='extraction_fluctuation':old_rejected[state]+=1
            was=old_rows[r['candidate']]['v2_publish'];now=rows[r['candidate']]['v2_publish']
            recovered_uncertain+=int(not was and now and state=='uncertain')
            recovered_change+=int(not was and now and state=='strong_change')
            previous_kept_vetoed+=int(was and not now)
            if args.tiers:
                level=r['publication_level'];levels[level]+=1;grade_by_kind[(r['kind'],level)]+=1
                assert now==(level in ('Confirmed','Probable'))
                assert prior['state']==state,'Image verdicts must not change'
                if state=='uncertain':
                    conditions.update(r['confidence_conditions'])
                    if r['confidence_conditions']:primary_conditions[r['confidence_conditions'][0]]+=1
                if level=='Candidate':assert r['candidate_geometry_wkb']==rows[r['candidate']]['geometry'].wkb_hex
            else:assert now==(state!='strong_stable')
            if state=='strong_stable':veto.update(r['reasons'])
            if state=='uncertain':qa.update(r['reasons'])
        features.append(dict(pair=key,descriptors=len(new['candidates']),unchanged=True))
    totals={s:{k:sum(p[s]['types'][k]['count'] for p in data) for k in kinds} for s in ('before','after')}
    no_gt={s:sum(sum(p[s+'_no_gt_intersection'].values()) for p in data) for s in ('before','after')}
    matched={s:sum(p[s]['types'][k]['matched_gt'] for p in data for k in kinds) for s in ('before','after')}
    summary=dict(totals=totals,no_gt_intersection=no_gt,states=dict(states),veto_reasons=dict(veto),qa_reasons=dict(qa),
        old_919_reclassification=dict(old_rejected),transitions=dict(transitions),
        uncertain_recovered=recovered_uncertain,strong_change_recovered=recovered_change,
        previously_kept_vetoed=previous_kept_vetoed,detectable_gt_matched=matched,
        verification_seconds=sum(p['counts']['v2_patch_verification_seconds'] for p in data),
        analysis_seconds=sum(p['analysis_seconds'] for p in data),
        total_seconds=sum(p['total_seconds'] for p in data),geometry_checks=geometry,feature_checks=features)
    if args.tiers:
        summary.update(publication_levels=dict(levels),uncertain_conditions=dict(conditions),
            primary_uncertain_conditions=dict(primary_conditions),
            by_kind={k:{level:grade_by_kind[(k,level)] for level in ('Confirmed','Probable','Candidate','Rejected')} for k in kinds})
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    if args.tiers:
        lines=['# Fast2 候选分级与融合发布','',
          '现有 7 期原始 Auto 缓存、6 个变化对。GT 只用于导出后的离线评价；未进入分级或校准。',
          'Confirmed=影像 strong_change；Probable=影像 uncertain 且已有 Fast2 证据全部达到强证据要求；Candidate=影像 uncertain 且 Fast2 证据一般；Rejected=影像 strong_stable。前两级正式发布，Candidate 几何与原因保留于内部 patch_verification.json 和本实验 diagnostics。',
          '配准不可靠/缺图/不可用边界仍是影像 uncertain，不会据此单独判 Rejected。Fast2 原先已否决的候选不被本层恢复。','',
          '|类型|三态全保留版|融合发布版|','|---|---:|---:|']
        lines += [f"|{k}|{totals['before'][k]}|{totals['after'][k]}|" for k in kinds]
        lines += ['',f"无 GT 交集对象：{no_gt['before']} → {no_gt['after']}。这是空间代理指标，不等同于人工逐个确认的假变化。",'',
          '|级别（局部候选）|数量|','|---|---:|']
        lines += [f'|{k}|{levels[k]}|' for k in ('Confirmed','Probable','Candidate','Rejected')]
        lines += ['',f"影像三态仍为 {dict(states)}；本轮没有改变任何影像三态裁决。",'',
          '## 固定的发布证据要求','',
          '新增/灭失对称：连续长度≥max(64m,8倍路宽)，两期有效观测，本期 geometry 与 surface 支持≥0.75，对期 surface≤0.05且无 probability 支持、已确认缺失，未覆盖区间占所属轴≥50%。经过 junction 时还需长度≥max(96m,12倍路宽)且已有多时相持续证据。',
          '宽变：长度≥96m，校正宽差≥2.5倍原名义阈值，超过3倍单期宽度波动和时相散布；已有对应为一对一、匹配覆盖≥75%，且强宽差在已有 profile 上持续覆盖≥90%。不重新测宽或匹配道路。',
          '这些要求只控制 uncertain 是否晋级 Probable，不改变候选生成阈值/几何。没有用固定的 confidence=0.95 或 qa_state=confirmed 单独决定发布。',
          '当前存量资料不保留每个 presence 区间的多重匹配排名，因此分级没有假装拥有该信息；使用已生成未覆盖区间、轴覆盖比例与 junction/时序上下文。宽变使用完整已有 profile 对应的正反一对一关系。','',
          '|uncertain 未达到的条件|首个未满足（互斥）|所有未满足（可重叠）|','|---|---:|---:|']
        lines += [f'|{k}|{primary_conditions[k]}|{conditions[k]}|' for k in conditions]
        lines += ['', '|变化对|A/R/W/N 前→后|无 GT 交集前→后|影像+分级秒|分析秒|总秒|', '|---|---|---|---:|---:|---:|']
        for p in data:
            before='/'.join(str(p['before']['types'][k]['count']) for k in kinds)
            after='/'.join(str(p['after']['types'][k]['count']) for k in kinds)
            lines.append(f"|{p['pair']}|{before} → {after}|{sum(p['before_no_gt_intersection'].values())} → {sum(p['after_no_gt_intersection'].values())}|{p['counts']['v2_patch_verification_seconds']:.2f}|{p['analysis_seconds']:.2f}|{p['total_seconds']:.2f}|")
        lines += ['',f"六对读取/分析/导出总计 {summary['total_seconds']:.2f}s；分析累计 {summary['analysis_seconds']:.2f}s，影像验证与分级 {summary['verification_seconds']:.2f}s。",'',
          f"可检测 GT 匹配：{matched['before']}/7 → {matched['after']}/7，沿用覆盖≥10%的原评价定义；19 个上游不可检测和5个部分检测对象不进入该项统计。",'',
          '|变化对|GT ID|类型|覆盖前→后|','|---|---|---|---:|']
        for p in data:
            old={(r['gt_id'],r['predicted_type']):r for r in p['before']['objects']}
            for r in p['after']['objects']:
                value=old[(r['gt_id'],r['predicted_type'])]['area_coverage']
                lines.append(f"|{p['pair']}|{r['gt_id']}|{r['predicted_type']}|{value:.2%} → {r['area_coverage']:.2%}|")
        lines += ['', '已逐对断言：全部局部候选数量/顺序、geometry WKB、轴线、宽度、变化类型、影像特征、校准值和影像三态完全不变。生产入口直接调用分级模块；实验未覆盖正式项目成果，未运行提取模型或 Fast1。',
          '修改：fast_candidate_publication.py（已有证据分级）；fast_patch_verification.py（分级发布、内部几何审计）；fast_auto_v2.py（仅传递已算出的 profiles / width audit）；fast_multitemporal.py（更新缓存版本）。CLI、归一化实验、GT 和 Temporal 算法未改。']
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
        print(json.dumps(summary,ensure_ascii=False,indent=2));return
    lines=['# Fast2 原始影像三态裁决实测','',
        '六个变化对使用原始 Auto 道路缓存完整运行候选分析、原始影像验证和网络组装。未训练/运行提取模型；GT 只在 Auto 成果导出后评价。',
        '特征、局部配准、辐射归一化、候选生成保持不变；逐对核对全部缓存描述符与校准值一致。变化只在判定及发布规则：strong_stable 否决；strong_change 保留；uncertain 保留传入的 Fast2 判断和 precision reason，并写内部 QA。',
        '新增/灭失的强稳定需同时满足可靠配准、两期道路结构支持、稳定双侧边界及高结构相似度。单项证据不显著不能否决。宽变支持单侧移动+另一侧稳定；整体平移只有双方位移均超过误差且持续一致才否决。曲率/分辨率不足仅记录 uncertain。','',
        '|类型|上一版过严影像验证|本版三态|','|---|---:|---:|']
    lines += [f"|{k}|{totals['before'][k]}|{totals['after'][k]}|" for k in kinds]
    lines += ['',f"无 GT 面积交集对象：{no_gt['before']} → {no_gt['after']}。该数为空间代理指标，不等于已人工确认的假变化。",'',
        '|局部候选三态|数量|','|---|---:|']
    lines += [f'|{k}|{states[k]}|' for k in ('strong_stable','strong_change','uncertain')]
    lines += ['',f"原来的 919 个 extraction_fluctuation：{dict(old_rejected)}。",
        f"此前被否决但本次因 uncertain 恢复 {recovered_uncertain} 个，因 strong_change 保留 {recovered_change} 个；此前已保留但本次判强稳定 {previous_kept_vetoed} 个。局部候选与组装后正式对象数量不同。",
        '恢复依据是进入影像验证前 Fast2 的判断，不复活上游已经否决的候选，也不恢复 GT 指定对象。','',
        '|真正否决原因|数量|','|---|---:|']
    lines += [f'|{k}|{v}|' for k,v in veto.items()]
    lines += ['', '|uncertain QA 原因（允许重叠）|数量|','|---|---:|']
    lines += [f'|{k}|{v}|' for k,v in qa.items()]
    lines += ['', '|变化对|A/R/W/N 前→后|无 GT 交集前→后|影像验证秒|分析秒|', '|---|---|---|---:|---:|']
    for p in data:
        before='/'.join(str(p['before']['types'][k]['count']) for k in kinds)
        after='/'.join(str(p['after']['types'][k]['count']) for k in kinds)
        lines.append(f"|{p['pair']}|{before} → {after}|{sum(p['before_no_gt_intersection'].values())} → {sum(p['after_no_gt_intersection'].values())}|{p['counts']['v2_patch_verification_seconds']:.2f}|{p['analysis_seconds']:.2f}|")
    lines += ['',f"影像验证累计 {summary['verification_seconds']:.2f}s，分析累计 {summary['analysis_seconds']:.2f}s，读取/分析/导出累计 {summary['total_seconds']:.2f}s。",'',
        '## GT 保留（沿用现有评价定义）','',
        f"7 个可检测 GT 中匹配对象 {matched['before']} → {matched['after']}（沿用面积覆盖≥10%的现有定义）。19 个上游不可检测和5个部分可检测对象不参与该统计。",'',
        '|变化对|GT ID|类型|覆盖前→后|','|---|---|---|---:|']
    for p in data:
        old={(r['gt_id'],r['predicted_type']):r for r in p['before']['objects']}
        for r in p['after']['objects']:
            value=old[(r['gt_id'],r['predicted_type'])]['area_coverage']
            lines.append(f"|{p['pair']}|{r['gt_id']}|{r['predicted_type']}|{value:.2%} → {r['area_coverage']:.2%}|")
    lines += ['', '实现：fast_patch_verification.py 修改 reasons / record_decision，fast_multitemporal.py 更新缓存版本；没有改 fast_image_structure.py 或 Fast2 主候选算法。',
        '实验输出仅在本目录，未覆盖真实项目正式成果。所有原候选几何 WKB、轴线、宽度、变化类型及顺序保持一致。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
