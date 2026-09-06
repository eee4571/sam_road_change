"""Geometry-only replay of cached accepted objects; never opens masks or truth."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import from_wkt
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.auto_change_geometry import render_change_geometry
from engine.auto_change_assembly import polygonal
from engine.fast_auto_change import complete_written_auto_result
from engine.fast_pipeline import _write_fast_public_changes


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def polygon_stats(geometry):
    parts = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
    return dict(parts=len(parts), holes=sum(len(p.interiors) for p in parts),
                vertices=sum(len(p.exterior.coords)+sum(len(r.coords) for r in p.interiors) for p in parts),
                small_parts=sum(p.area < 4. for p in parts), area_m2=geometry.area)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    baseline, output=args.baseline.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Use a new output directory; cached baseline is read-only')
    output.mkdir(parents=True)
    source=json.loads((baseline/'input_provenance.json').read_text(encoding='utf-8'))
    protected=['auto_change_assembly.py','auto_presence_candidates.py','auto_width_precision.py',
               'auto_track_evidence.py','fast_pipeline.py','road_network_connection.py',
               'road_network_products.py','road_track_corridors.py','width/paired_width_profile.py',
               'width/production_workflow.py']
    hashes={p:hashlib.sha256((Path('code/engine')/p).read_bytes()).hexdigest() for p in protected}
    gpkg=baseline/'auto_diagnostics.gpkg'
    seeds=gpd.read_file(gpkg,layer='local_seeds')
    old=gpd.read_file(baseline/'network_assembly.gpkg',layer='change_objects').to_crs(seeds.crs)
    assembly={layer:gpd.read_file(baseline/'network_assembly.gpkg',layer=layer).to_crs(seeds.crs)
              for layer in ('object_axes','assembly_bridges')}
    assembly['membership']=pd.read_csv(baseline/'assembly_membership.csv')
    widths={side:gpd.read_file(source[side]['width_segments']).to_crs(seeds.crs) for side in ('before','after')}
    samples=gpd.read_file(gpkg,layer='width_precision_samples')
    new,geometry_audit=render_change_geometry(old,seeds,assembly,widths,samples)
    # Preserve every raw/review/evidence layer and the exact cached assembly
    # relationships. Only final polygons and additional geometry audits change.
    for name in ('auto_diagnostics.gpkg','network_assembly.gpkg','candidate_funnel.json',
                 'assembly_membership.csv','assembly_decisions.csv','assembly_summary.json'):
        shutil.copy2(baseline/name,output/name)
    for name,data in geometry_audit.items():
        data.to_file(output/'auto_diagnostics.gpkg',layer=name,driver='GPKG')
    new.to_file(output/'network_assembly.gpkg',layer='change_objects',driver='GPKG')
    output_crs=gpd.read_file(baseline/'road_changes.shp',rows=1).crs
    public=new.to_crs(output_crs)
    public.geometry=public.geometry.map(polygonal)
    public['before_per'],public['after_per']=source['before']['period'],source['after']['period']
    _write_fast_public_changes(public,output)
    names={'added':'added_roads.shp','removed':'removed_roads.shp',
           'widened':'widened_road_parts.shp','narrowed':'narrowed_road_parts.shp'}
    for kind,name in names.items():
        public.loc[public.change_typ==kind].to_file(output/name,encoding='UTF-8')
    public.to_file(output/'auto_diagnostics.gpkg',layer='changes',driver='GPKG')
    result=complete_written_auto_result(output,before_period=source['before']['period'],after_period=source['after']['period'])
    write(output/'result.json',result)
    write(output/'input_provenance.json',{**source,'geometry_baseline':str(baseline),
        'geometry_source':'final axes + final widths / saved paired profiles',
        'qualification_replayed':False,'assembly_replayed':False,'mask_read_for_geometry':False})
    for layer,data in [('previous_formal',old),('regularized_formal',new),*geometry_audit.items(),
                       ('unchanged_object_axes',assembly['object_axes'])]:
        data.to_file(output/'geometry_comparison.gpkg',layer=layer,driver='GPKG')
    checks=[]
    for previous,current in zip(old.itertuples(),new.itertuples()):
        before_stats,after_stats=polygon_stats(previous.geometry),polygon_stats(current.geometry)
        axes=assembly['object_axes'].loc[assembly['object_axes'].object_id==current.object_id]
        axis=axes.geometry.union_all()
        coverage=axis.intersection(current.geometry.buffer(1e-6)).length/max(axis.length,1e-9)
        checks.append(dict(object_id=current.object_id,kind=current.change_typ,
            before=before_stats,after=after_stats,valid=bool(current.geometry.is_valid),
            axis_coverage=coverage if current.change_typ in ('added','removed') else None))
    assert public.is_valid.all() and all(r['valid'] for r in checks), 'Invalid published geometry'
    assert all(r['axis_coverage'] is None or r['axis_coverage'] >= .999999 for r in checks), 'Lost approved axis coverage'
    pd.testing.assert_frame_equal(old.drop(columns='geometry'),new.drop(columns='geometry'))
    unchanged={}
    for layer in ('input_candidates','candidate_audit','review_candidates','local_seeds'):
        a=gpd.read_file(gpkg,layer=layer); b=gpd.read_file(output/'auto_diagnostics.gpkg',layer=layer)
        unchanged[layer]=a.equals(b)
        assert unchanged[layer], layer
    for name in ('assembly_membership.csv','assembly_decisions.csv','candidate_funnel.json'):
        unchanged[name]=(baseline/name).read_bytes()==(output/name).read_bytes()
        assert unchanged[name],name
    report=dict(by_type={k:dict(before=int(old.change_typ.eq(k).sum()),after=int(new.change_typ.eq(k).sum())) for k in names},
                unchanged=unchanged,object_attributes_unchanged=True,checks=checks,
                source_surface_or_probability_read=False,
                formal_narrowed_note='0 accepted narrowed objects; narrowed illustration is review-only')
    write(output/'geometry_validation.json',report)
    write(output/'protected_modules.json',{p:dict(before=h,after=hashlib.sha256((Path('code/engine')/p).read_bytes()).hexdigest())
                                           for p,h in hashes.items()})
    render(output,baseline,source,old,new,seeds,assembly,widths,samples,report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2),flush=True)


def render(output,baseline,source,old,new,seeds,assembly,widths,samples,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.merge import merge
    from PIL import Image
    review=output/'review'; review.mkdir()
    lines={side:gpd.read_file(source[side]['centerlines']).to_crs(old.crs) for side in ('before','after')}
    datasets={side:[rasterio.open(p) for p in source[side]['imagery_tiles']] for side in ('before','after')}
    colors={'added':'#21ad79','removed':'#e66589','widened':'#139ce0','narrowed':'#edaa28'}
    examples=[]

    def plot(name,previous,current,axes,title,side):
        bounds=box(*current.geometry.total_bounds).buffer(18).bounds
        image_crs=datasets[side][0].crs
        geographic=gpd.GeoSeries([box(*bounds)],crs=old.crs).to_crs(image_crs).total_bounds
        rgb,_=merge(datasets[side],bounds=tuple(geographic),
                    res=max(geographic[2]-geographic[0],geographic[3]-geographic[1])/900,
                    indexes=[1,2,3],resampling=rasterio.enums.Resampling.bilinear)
        rgb=np.moveaxis(rgb,0,-1)
        if rgb.dtype!=np.uint8:
            lo,hi=np.percentile(rgb[rgb>0],[1,99]); rgb=np.clip((rgb-lo)/max(hi-lo,1)*255,0,255).astype('uint8')
        fig,axs=plt.subplots(1,3,figsize=(15,7))
        for i,(ax,data,label) in enumerate(zip(axs,(previous,current,current),('Previous geometry','Regularized geometry','Regularized / final axis / image'))):
            for kind,color in colors.items():
                local=data.loc[data.change_typ==kind]
                if len(local):
                    local=(local.to_crs(image_crs) if i==2 else local)
                    local.plot(ax=ax,color=color,edgecolor='#333333',linewidth=.55,alpha=.7 if i==2 else 1)
            if i==2:
                ax.imshow(rgb,extent=(geographic[0],geographic[2],geographic[1],geographic[3]),zorder=-1)
                road=lines[side]; road=road.iloc[road.sindex.query(box(*bounds),predicate='intersects')]
                if len(road):road.to_crs(image_crs).plot(ax=ax,color='white',linewidth=.55,linestyle='--')
                axes.to_crs(image_crs).plot(ax=ax,color='#ffe552',linewidth=.8)
                ax.set_xlim(geographic[0],geographic[2]);ax.set_ylim(geographic[1],geographic[3])
                ax.set_aspect(1/np.cos(np.radians((geographic[1]+geographic[3])/2)) if image_crs.is_geographic else 1)
            else:
                axes.plot(ax=ax,color='#444444',linewidth=.45,linestyle='--')
                ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect(1)
            ax.set_title(label);ax.axis('off')
        fig.suptitle(title);fig.tight_layout();fig.savefig(review/f'{name}.png',dpi=150);plt.close(fig)
        image=Image.open(review/f'{name}.png');image.thumbnail((1600,850));image.convert('RGB').save(review/f'{name}.jpg',quality=92)
        examples.append(dict(file=name,title=title))

    for kind in ('added','removed','widened'):
        rows=[c for c in report['checks'] if c['kind']==kind]
        chosen=max(rows,key=lambda c:c['before']['vertices']-c['after']['vertices'])
        oid=chosen['object_id']
        plot(kind,old.loc[old.object_id==oid],new.loc[new.object_id==oid],
             assembly['object_axes'].loc[assembly['object_axes'].object_id==oid],f'{kind} | {oid} | accepted object',
             'before' if kind=='removed' else 'after')
    junctions=assembly['object_axes'].loc[assembly['object_axes'].role=='junction_bridge']
    if len(junctions):
        oid=junctions.iloc[0].object_id
        plot('junction',old.loc[old.object_id==oid],new.loc[new.object_id==oid],
             assembly['object_axes'].loc[assembly['object_axes'].object_id==oid],
             f'Junction | {oid} | same frozen route and membership','after')
    # No narrowed object passed the preceding precision review. Demonstrate the
    # renderer on an existing review candidate, never add it to public changes.
    raw=gpd.read_file(baseline/'auto_diagnostics.gpkg',layer='candidate_audit')
    candidate=raw.loc[raw.change_typ=='narrowed'].sort_values('length_m',ascending=False).iloc[[0]].copy().reset_index(drop=True)
    candidate['object_id']='REVIEW_ONLY'
    axis=gpd.GeoDataFrame([dict(seed_id=0,object_id='REVIEW_ONLY',role='seed',change_typ='narrowed',
                               geometry=from_wkt(candidate.iloc[0].axis_wkt))],crs=old.crs)
    diagnostic_assembly={'object_axes':axis,'membership':pd.DataFrame([dict(seed_id=0,object_id='REVIEW_ONLY')]),
                         'assembly_bridges':assembly['assembly_bridges'].iloc[:0]}
    preview,_=render_change_geometry(candidate,candidate,diagnostic_assembly,widths,samples)
    candidate.to_file(output/'geometry_comparison.gpkg',layer='narrowed_review_previous',driver='GPKG')
    preview.to_file(output/'geometry_comparison.gpkg',layer='narrowed_review_geometry_demo',driver='GPKG')
    plot('narrowed_review_only',candidate,preview,axis,
         f'Narrowed | candidate {int(candidate.iloc[0].candidate_id)} | REVIEW ONLY - NOT PUBLISHED','before')
    for ds in datasets.values():
        for d in ds:d.close()
    total=lambda key,which:sum(c[which][key] for c in report['checks'])
    text=['# Fast Auto 正式变化几何规则化','','仅复跑正式几何生成，复用原审核结果和组装连接。未读取原始 mask、surface、probability 或真值；影像仅用于下列对比图。','',
          '| 类型 | 原正式对象 | 新正式对象 |','|---|---:|---:|']
    for kind,r in report['by_type'].items():text.append(f"| {kind} | {r['before']} | {r['after']} |")
    text+=['','全部候选、review、正式 seed、审核属性、组装成员及连接决定保持不变。新增/灭失按对应期次最终轴线和同轨宽度生成；宽度变化按既有 canonical axis 与保存的 paired width 剖面生成；bridge 沿原路径插值两端宽度。','',
           f"轮廓顶点：{total('vertices','before')} → {total('vertices','after')}；多边形分量：{total('parts','before')} → {total('parts','after')}；孔洞：{total('holes','before')} → {total('holes','after')}；小于 4 平方米分量：{total('small_parts','before')} → {total('small_parts','after')}。",'',
           '拓宽/变窄本应是道路两侧的差带，两个规则分量不属于噪声碎块。当前 narrowed 正式结果为 0，因此其示例仅为已有 review 候选的几何演示，未提升为正式变化。','',
           '图中虚线为最终中心线，黄色线为已批准的 seed / bridge / canonical axis；宽度差带分布在 canonical axis 两侧。','']
    for example in examples:
        text += [f"## {example['title']}",'',f"![对比]({example['file']}.png)",'']
    text += ['## 文件','', '- `../road_changes.shp` 和四类独立 Shapefile：正式结果。',
             '- `../geometry_comparison.gpkg`：前后正式面、规则化组成面、宽度剖面及 unchanged_object_axes；narrowed review 演示单独命名。',
             '- `../auto_diagnostics.gpkg`：原检测证据/候选/审核全保留，新增 published_geometry_parts 和 published_width_profiles。',
             '- `../geometry_validation.json`：逐对象面积、顶点、分量、孔洞、有效性和轴线覆盖。',
             '- `../protected_modules.json`：冻结算法文件前后 SHA-256。']
    (review/'README.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':main()
