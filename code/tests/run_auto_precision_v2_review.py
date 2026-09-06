"""Replay saved high-recall candidates; never read truth or change source products."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely import make_valid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.fast_auto_change import RoadScene, WindowedProbability, finalize_auto_candidates

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('baseline',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    source=json.loads((args.baseline/'input_provenance.json').read_text(encoding='utf-8'))
    gpkg=args.baseline/'auto_diagnostics.gpkg'
    raw=gpd.read_file(gpkg,layer='input_candidates'); metric=raw.crs
    if not args.render_only:
        protected=['engine/auto_change_assembly.py','engine/fast_pipeline.py','engine/road_network_connection.py',
                   'engine/road_track_corridors.py','engine/road_network_products.py','engine/width/paired_width_profile.py',
                   'engine/width/production_workflow.py','user_pipeline.py']
        hashes={p:hashlib.sha256((Path('code')/p).read_bytes()).hexdigest() for p in protected}
        roads=[gpd.read_file(source[p]['centerlines']) for p in ('before','after')]
        scenes={}; started=time.perf_counter()
        try:
            for period,lines in zip(('before','after'),roads):
                surfaces,widths,valid=[gpd.read_file(source[period][k]).to_crs(metric)
                                      for k in ('surfaces','width_segments','valid_observation')]
                probability=WindowedProbability(source[period]['road_probability'],metric)
                scenes[period]=RoadScene(lines.to_crs(metric),surfaces,widths,valid,probability,metric)
            evidence=gpd.read_file(gpkg,layer='existence_candidates')
            widths=gpd.read_file(gpkg,layer='width_candidates')
            intervals=gpd.read_file(gpkg,layer='presence_intervals')
            prior=json.loads((args.baseline/'candidate_funnel.json').read_text(encoding='utf-8'))
            counts={**prior['road_matching'],**prior['width']}
            for kind in ('added','removed'):
                counts.update({f'{kind}_{k}':v for k,v in prior[kind]['longitudinal'].items()})
            result=finalize_auto_candidates(raw.to_dict('records'),evidence.to_dict('records'),widths.to_dict('records'),counts,
                presence_audit=intervals.to_dict('records'),scenes=scenes,centerlines=roads,output_dir=output,
                before_period=source['before']['period'],after_period=source['after']['period'],
                elapsed_seconds=time.perf_counter()-started)
            write(output/'result.json',result)
        finally:
            for scene in scenes.values(): scene.probability.close()
        source.update(ground_truth_used=False,baseline_result=str(args.baseline.resolve()),
                      observation_evidence_reused=str(gpkg.resolve()),width_review_source='existing candidates remeasured from cached final surfaces/probability')
        write(output/'input_provenance.json',source)
        write(output/'protected_modules.json',{p:dict(before=h,after=hashlib.sha256((Path('code')/p).read_bytes()).hexdigest()) for p,h in hashes.items()})
    review(output,args.baseline,source,raw)

def review(output,baseline,source,raw):
    old=gpd.read_file(baseline/'network_assembly.gpkg',layer='change_objects').to_crs(raw.crs)
    new=gpd.read_file(output/'network_assembly.gpkg',layer='change_objects').to_crs(raw.crs)
    seeds=gpd.read_file(output/'auto_diagnostics.gpkg',layer='local_seeds')
    prior=gpd.read_file(baseline/'auto_diagnostics.gpkg',layer='local_seeds')
    audit=gpd.read_file(output/'auto_diagnostics.gpkg',layer='candidate_audit')
    pending=audit.loc[audit.publication_state=='review'].copy()
    pending['previously_formal']=pending.candidate_id.isin(prior.candidate_id)
    samples=gpd.read_file(output/'auto_diagnostics.gpkg',layer='width_precision_samples')
    funnel=json.loads((output/'candidate_funnel.json').read_text(encoding='utf-8'))
    report={'by_type':{},'review_reasons':funnel['publication_review'],
            'raw_candidate_count_unchanged':len(raw)==len(audit),
            'all_candidates_accounted_for':len(seeds)+len(pending)==len(raw),'all_objects_valid':bool(new.is_valid.all()),
            'review_geometry_max_difference_m2':float(max((make_valid(row.geometry).symmetric_difference(make_valid(raw.geometry.iloc[int(row.candidate_id)])).area
                                                          for row in pending.itertuples()),default=0.))}
    for kind in ('added','removed','widened','narrowed'):
        report['by_type'][kind]=dict(previous_objects=int(old.change_typ.eq(kind).sum()),new_objects=int(new.change_typ.eq(kind).sum()),
            raw_candidates=int(audit.change_typ.eq(kind).sum()),accepted_seeds=int(seeds.change_typ.eq(kind).sum()),
            review_candidates=int(pending.change_typ.eq(kind).sum()),
            newly_reviewed=int((pending.change_typ.eq(kind)&pending.previously_formal).sum()))
    for layer,frame in (('previous_formal',old),('revised_formal',new),('review_candidates',pending),('previous_seeds',prior),('revised_seeds',seeds)):
        frame.to_file(output/'precision_comparison.gpkg',layer=layer,driver='GPKG')
    pending.drop(columns='geometry').to_csv(output/'review_candidates.csv',index=False,encoding='utf-8-sig')
    samples.drop(columns='geometry').to_csv(output/'width_precision_samples.csv',index=False,encoding='utf-8-sig')
    write(output/'precision_comparison.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    render(output,source,old,new,pending,samples,report)

def render(output,source,old,new,pending,samples,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.merge import merge
    from shapely.geometry import box
    from PIL import Image
    review=output/'review'; review.mkdir(exist_ok=True)
    datasets={side:[rasterio.open(p) for p in source[side]['imagery_tiles']] for side in ('before','after')}
    crs=datasets['after'][0].crs; colors={'added':'#00ed85','removed':'#ff4368','widened':'#00d7ff','narrowed':'#ffbe25'}
    display={name:frame.to_crs(crs) for name,frame in (('previous',old),('new',new),('review',pending))}
    examples=[]
    def plot(extent,name,record=None):
        extent=gpd.GeoSeries([box(*extent)],crs=old.crs).to_crs(crs).total_bounds
        images={}
        for side in datasets:
            array,_=merge(datasets[side],bounds=tuple(extent),res=max(extent[2]-extent[0],extent[3]-extent[1])/1200,
                          indexes=[1,2,3],resampling=rasterio.enums.Resampling.bilinear)
            rgb=np.moveaxis(array,0,-1)
            if rgb.dtype!=np.uint8:
                lo,hi=np.percentile(rgb[rgb>0],[1,99]); rgb=np.clip((rgb-lo)/max(hi-lo,1)*255,0,255).astype('uint8')
            images[side]=rgb
        fig,axes=plt.subplots(1,4,figsize=(21,7))
        for ax,side,key,title in zip(axes,('before','after','after','after'),(None,'previous','new','review'),
                    ('Before image','Previous formal / After image','Revised formal / After image','Review only / After image')):
            ax.imshow(images[side],extent=(extent[0],extent[2],extent[1],extent[3]))
            if key:
                frame=display[key]; local=frame.iloc[frame.sindex.query(box(*extent),predicate='intersects')]
                for kind,color in colors.items():
                    rows=local.loc[local.change_typ==kind]
                    if len(rows):
                        if key=='review': rows.boundary.plot(ax=ax,color=color,linewidth=.7)
                        else: rows.plot(ax=ax,color=color,alpha=.85)
            ax.set_xlim(extent[0],extent[2]); ax.set_ylim(extent[1],extent[3]); ax.axis('off'); ax.set_title(title)
            ax.set_aspect(1/np.cos(np.radians((extent[1]+extent[3])/2)) if crs.is_geographic else 1)
        if record is not None: fig.suptitle(f"Candidate {int(record.candidate_id)}: {record.change_typ} | moved to review",fontsize=12)
        fig.tight_layout(); fig.savefig(review/f'{name}.png',dpi=130); plt.close(fig)
        im=Image.open(review/f'{name}.png'); im.thumbnail((1800,800)); im.convert('RGB').save(review/f'{name}.jpg',quality=92)
    plot(tuple(old.total_bounds),'overview')
    for kind in colors:
        rows=pending.loc[(pending.change_typ==kind)&pending.previously_formal].sort_values('length_m',ascending=False)
        if rows.empty: rows=pending.loc[pending.change_typ==kind].sort_values('length_m',ascending=False)
        chosen=[]
        for _,row in rows.iterrows():
            reason=row.precision_reason
            if chosen and reason==chosen[0].precision_reason: continue
            chosen.append(row)
            if len(chosen)==2: break
        for index,row in enumerate(chosen):
            name=f'{kind}_{index}'; plot(box(*row.geometry.bounds).buffer(35).bounds,name,row)
            examples.append(dict(file=name,candidate_id=int(row.candidate_id),kind=kind,reason=row.precision_reason,
                                 previously_formal=bool(row.previously_formal)))
            data=samples.loc[samples.candidate_id==row.candidate_id] if len(samples) else samples
            if len(data):
                fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
                for field,color in (('before_width','#4099ff'),('after_width','#f39a39')):
                    axes[0].plot(data.station_m,data[field],'.-',label=field,color=color)
                axes[0].set_ylabel('Width (m)'); axes[0].legend()
                axes[1].plot(data.station_m,data.after_width-data.before_width,'.-',label='After - Before')
                axes[1].axhline(0,color='gray'); axes[1].set_ylabel('Difference (m)'); axes[1].set_xlabel('Candidate station (m)')
                fig.tight_layout(); fig.savefig(review/f'{name}_profile.png',dpi=130); plt.close(fig)
    for ds in datasets.values():
        for d in ds:d.close()
    write(output/'review_examples.json',examples)
    rows=['# Fast 无真值 Auto 全图 Precision 复查','','复用当前两期原 Final 产品和全部高召回候选。仅重跑发布审核与冻结的网络组装；宽度候选使用原测量函数复算局部剖面。未读取参考真值。','',
          '| 类型 | 修改前正式对象 | 修改后正式对象 | 原始候选 | 正式 seed | review | 本轮从正式转 review |','|---|---:|---:|---:|---:|---:|---:|']
    for kind,r in report['by_type'].items():
        rows.append(f"| {kind} | {r['previous_objects']} | {r['new_objects']} | {r['raw_candidates']} | {r['accepted_seeds']} | {r['review_candidates']} | {r['newly_reviewed']} |")
    rows += ['','计数单位：正式结果为组装对象，候选为局部 seed，不能直接相减当作 precision。review 不等于已被人工确认是假变化。',
             '全部原始候选保留；review 几何与原候选一致。道路提取后处理、Final Centerline、Final Surface、Width 主流程、GT-assisted 和 auto_change_assembly 均未修改。','',
             '![全图](overview.png)','','## 各类 review reason（同一候选可有多个原因）','','| 类型 | 原因 | 数量 |','|---|---|---:|']
    for kind,r in report['review_reasons'].items():
        for reason,count in sorted(r['review_reason_counts'].items(),key=lambda x:-x[1]): rows.append(f'| {kind} | {reason} | {count} |')
    rows += ['','## 典型转待审候选','']
    for r in examples:
        rows += [f"### {r['kind']} / candidate {r['candidate_id']}",'',r['reason'],'',f"![对照]({r['file']}.png)",'']
        if (review/f"{r['file']}_profile.png").exists(): rows += [f"![宽度剖面]({r['file']}_profile.png)",'']
    rows += ['## 成果','','- `../road_changes.shp`：修改后四类正式变化。','- `../precision_comparison.gpkg`：修改前后正式变化、前后 seed、review candidates。',
             '- `../auto_diagnostics.gpkg`：原始候选、审核属性、宽度逐站复核、原始存在性及宽度诊断。','- `../review_candidates.csv`、`../width_precision_samples.csv`：可表格检查的原因与剖面。',
             '- `../protected_modules.json`：冻结模块运行前后的 SHA-256。','','本轮到此为止，等待用户评判真实结果，不扩展提取或组装算法。']
    (review/'README.md').write_text('\n'.join(rows),encoding='utf-8')

if __name__=='__main__': main()
