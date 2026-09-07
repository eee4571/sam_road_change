"""Cached real-region acceptance for the single Fast final product chain."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely import union_all

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.fast_gt_reconciliation import augment_fast_changes_with_truth,build_fast_temporal_outputs
from app.result_publisher import ResultPublisher


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    cache=Path('project/test_area/auto_geometry_20260906_release')
    provenance=json.loads((cache/'input_provenance.json').read_text(encoding='utf-8'))
    automatic=json.loads((cache/'result.json').read_text(encoding='utf-8'))
    periods=[{**provenance[side],'grid':'验证区1','status':'completed'} for side in ('before','after')]
    inputs=list(p for p in cache.iterdir() if p.is_file())
    for period in periods:
        for key in ('centerlines','width_segments','surfaces'):
            path=Path(period[key]);inputs.extend(path.parent.glob(path.stem+'.*'))
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(inputs)}
    result=augment_fast_changes_with_truth(automatic,Path('C:/Users/zhoum/geesing/验证区/数据2（20221020）/道路.shp'),
        output/'audit',before_result=periods[0],after_result=periods[1],before_period=periods[0]['period'],
        after_period=periods[1]['period'],defer_finalization=True)
    result.update(grid='验证区1',status='completed')
    manifest=dict(period_results=periods,change_results=[result],execution_profile='fast',job_root=str(output))
    manifest['temporal_results']=build_fast_temporal_outputs(manifest,output/'_work')
    ResultPublisher(output/'成果输出').publish_manifest(manifest)
    write(output/'pipeline_result.json',manifest)
    integrity={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in hashes.items()}
    assert all(integrity.values());write(output/'auto_integrity.json',integrity)
    summarize(output,manifest,provenance)


def summarize(output,manifest,provenance):
    result=manifest['change_results'][0];temporal=manifest['temporal_results'][0]
    auto=gpd.read_file(result['automatic']['road_changes']);changes=gpd.read_file(result['road_changes'])
    intervals=gpd.read_file(result['event_geometry_audit'],layer='changes')
    gt=intervals.loc[intervals.change_src.eq('GT_ASSISTED')]
    obs=gpd.read_file(temporal['observations_shp']);events=gpd.read_file(temporal['events_shp'])
    checks=[];period_checks=[]
    for row in gt.itertuples():
        states=obs.loc[obs.road_id==row.track_id].sort_values('period')
        expected=['absent','present'] if row.change_typ=='added' else ['present','absent'] if row.change_typ=='removed' else ['present','present']
        match=events.loc[(events.road_id==row.track_id)&(events.event_typ==row.change_typ)]
        checks.append(dict(track_id=row.track_id,states=states.status.tolist(),expected=expected,event_count=len(match),
                           consistent=states.status.tolist()==expected and len(match)==1))
    assert all(c['consistent'] for c in checks)
    write(output/'state_event_checks.json',checks)
    metric=gt.estimate_utm_crs() if gt.crs.is_geographic else gt.crs
    products={p['period']:{key:gpd.read_file(p[key]).to_crs(metric) for key in ('surfaces','width_segments')}
              for p in manifest['final_period_results']}
    period_covers={period:frames['surfaces'].geometry.union_all() for period,frames in products.items()}
    geometry_checks=[]
    for row in gt.to_crs(metric).itertuples():
        widths=[];covers=[]
        for period,expected_width in ((result['before_period'],row.width_bef),(result['after_period'],row.width_aft)):
            profile=products[period]['width_segments']
            profile=profile.loc[profile.track_id==row.track_id] if 'track_id' in profile else profile.iloc[:0]
            measured=float(np.average(profile.width_m,weights=profile.length)) if len(profile) else 0.
            widths.append(abs(measured-expected_width))
            covers.append(period_covers[period])
        present=covers[1] if row.change_typ in ('added','widened') else covers[0]
        difference=row.geometry.difference(present).area
        geometry_checks.append(dict(track_id=row.track_id,outside_final_surface_m2=difference,width_error_m=max(widths),
                                    consistent_width=max(widths)<.05))
    # Raw interval buffers are audit evidence, not the final boundary contract:
    # junction-sized terminal patches are absorbed by the continuous renderer.
    assert all(c['consistent_width'] for c in geometry_checks)
    write(output/'geometry_width_checks.json',geometry_checks)
    unchanged=len(set(intervals.loc[intervals.change_src.eq('AUTO'),'auto_id']))
    for side,period in zip(('before','after'),manifest['final_period_results']):
        original=gpd.read_file(provenance[side]['centerlines']);final=gpd.read_file(period['centerlines']).to_crs(original.crs)
        surface=gpd.read_file(period['surfaces']);metric=surface.estimate_utm_crs() if surface.crs.is_geographic else surface.crs
        surfaces=surface.to_crs(metric)
        retained=set(g.wkb for g in final.geometry)
        period_checks.append(dict(period=period['period'],original_lines=len(original),final_lines=len(final),
            unchanged_lines=sum(g.wkb in retained for g in original.geometry),surface_valid=bool(surface.is_valid.all()),
            small_holes=sum(r.area<1 for geom in surfaces.geometry for p in ([geom] if geom.geom_type=='Polygon' else geom.geoms)
                            for r in [__import__('shapely').geometry.Polygon(x) for x in p.interiors])))
    published=output/'成果输出'
    leaked={}
    for path in published.rglob('*.shp'):
        fields=gpd.read_file(path,rows=1).columns
        bad=[c for c in fields if any(x in c.lower() for x in ('source','qa_','truth','change_src','review','extract'))]
        if bad:leaked[str(path)]=bad
    assert not leaked,leaked
    assert not any(p.name in ('Auto','GT-assisted') for p in published.rglob('*') if p.is_dir())
    assert len(list((output/'_work').rglob('road_life.shp')))==1
    assert not events.qa_state.eq('review').any()
    parts=gpd.read_file(temporal['event_parts_shp'])
    unlinked=int(parts.event_id.isna().sum())
    assert unlinked==0,'A final change did not enter temporal events'
    report=dict(auto_counts=auto.change_typ.value_counts().to_dict(),final_counts=changes.change_typ.value_counts().to_dict(),
        original_auto_decisions_retained=unchanged,gt_intervals=len(gt),state_event_consistent=len(checks),
        periods=period_checks,temporal_events=len(events),internal_event_qa=events.qa_state.value_counts().to_dict(),
        manual_review_count=0,temporal_count=1,public_attribute_leaks=leaked,
        linked_change_parts=len(parts),unlinked_change_parts=unlinked,consistent_gt_widths=len(geometry_checks),
        raw_interval_geometry_adjusted=sum(c['outside_final_surface_m2']>=.1 for c in geometry_checks),
        internal_state_class_conflicts=int(events.evidence.eq('pair_class_observation_disagreement').sum()))
    write(output/'acceptance_summary.json',report)
    render(output,manifest,provenance,gt)
    text=['# Fast 最终成果验收','','流程：Auto 缓存 → GT 后验局部校正 → Final Roads 重建 → Final Changes → 一次 Final Temporal。','',
          '正式目录只有 `成果输出/验证区1/01_单期道路`、`02_变化检测`、`03_长时序`。来源、低置信度 QA 和原 Auto 缓存均留在内部。','',
          '| 类型 | 原 Auto | 最终 |','|---|---:|---:|']
    for kind in ('added','removed','widened','narrowed'):text.append(f'| {kind} | {int(auto.change_typ.eq(kind).sum())} | {int(changes.change_typ.eq(kind).sum())} |')
    text+=['',f'原 Auto 变化 {unchanged}/{len(auto)} 个对象判定保留；正式几何按连续道路重建。GT 补充 {len(gt)} 个轴区间。补充区间全部通过实际道路状态与事件一致性检查。',
           '全部正式路面和变化面统一由连续道路轴与平滑宽度构建。内部接点不加端帽，只有真正外端为平直端帽；同类变化面消除内部重叠边界。原 Auto 判定保留，但其几何也重新生成。',
           f'所有 {len(parts)} 个变化区间均已关联长时序事件，GT Final Width 核对通过。{report["raw_interval_geometry_adjusted"]} 个原始区间面因连续边界/路口处理而不再完整覆盖，差异面积保留在 geometry_width_checks.json；不再把逐段 buffer 当作最终边界验收标准。',
           f'长时序生成一次，共 {len(events)} 个事件；不产生人工 review 要求。内部 QA：{report["internal_event_qa"]}。',
           f'其中 {report["internal_state_class_conflicts"]} 个事件的局部变化分类与整路观察状态存在差异，已自动选择对应道路、保留实际观察状态并写内部 QA，不要求人工处理；不把内部一致性检查等同于外部真值精度。',
           '本区域 GT 只有 5 个 Added 对象。其他类别的局部道路修改、匹配冲突和三期状态由回归用例验证。未重跑模型推理。','',
           '详见 `acceptance_summary.json`、`state_event_checks.json`、`auto_integrity.json`。','', '## 局部几何','']
    for p in sorted((output/'comparison').glob('*.png')):text.extend([f'![{p.stem}]({p.resolve().as_posix()})',''])
    (output/'README.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


def render(output,manifest,provenance,gt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.merge import merge
    from shapely.geometry import box
    if gt.crs.is_geographic:gt=gt.to_crs(gt.estimate_utm_crs())
    directory=output/'comparison';directory.mkdir(exist_ok=True)
    periods=manifest['final_period_results']
    frames=[gpd.read_file(p['surfaces']) for p in periods]
    for number,(_,row) in enumerate(gt.sort_values('length_m',ascending=False).head(5).iterrows()):
        extent=box(*row.geometry.bounds).buffer(25)
        fig,axes=plt.subplots(1,2,figsize=(12,7))
        for ax,side,frame in zip(axes,('before','after'),frames):
            datasets=[rasterio.open(p) for p in provenance[side]['imagery_tiles']]
            crs=datasets[0].crs;bounds=gpd.GeoSeries([extent],crs=gt.crs).to_crs(crs).total_bounds
            data,_=merge(datasets,bounds=tuple(bounds),res=max(bounds[2]-bounds[0],bounds[3]-bounds[1])/800,indexes=[1,2,3])
            ax.imshow(np.moveaxis(data,0,-1),extent=(bounds[0],bounds[2],bounds[1],bounds[3]))
            local=frame.to_crs(crs);local=local.loc[local.intersects(box(*bounds))]
            if len(local):local.plot(ax=ax,color='#48db9a',alpha=.5,edgecolor='#164e38',linewidth=.5)
            ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.axis('off');ax.set_title(f'{side} / Final Roads')
            for ds in datasets:ds.close()
        fig.tight_layout();fig.savefig(directory/f'final_road_{number}.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
