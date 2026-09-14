"""Geometric result comparison, not a truth-based accuracy evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import geopandas as gpd
import numpy as np
from shapely import union_all, make_valid, set_precision
from shapely.geometry import box


KINDS=('added','removed','widened','narrowed')


def overlay_polygon(geometry):
    geometry=make_valid(geometry)
    if geometry.geom_type=='Polygon':
        return set_precision(geometry,.001)
    return union_all([overlay_polygon(g) for g in getattr(geometry,'geoms',())
                      if g.geom_type in ('Polygon','MultiPolygon','GeometryCollection')])


def compare(a,b):
    report={}; differences=[]
    for kind in KINDS:
        aa=a.loc[a.change_typ==kind];bb=b.loc[b.change_typ==kind]
        old_parts=[overlay_polygon(g) for g in aa.geometry];new_parts=[overlay_polygon(g) for g in bb.geometry]
        old=union_all(old_parts);new=union_all(new_parts)
        intersect=old.intersection(new).area
        old_coverage=[g.intersection(new).area/max(g.area,1e-9) for g in old_parts]
        new_coverage=[g.intersection(old).area/max(g.area,1e-9) for g in new_parts]
        report[kind]=dict(baseline_count=len(aa),v2_count=len(bb),baseline_area_m2=old.area,v2_area_m2=new.area,
            baseline_area_retained=intersect/old.area if old.area else None,
            overlap_iou=intersect/max(old.union(new).area,1e-9),
            baseline_objects_retained_at_50pct=sum(c>=.5 for c in old_coverage),
            baseline_objects_no_overlap=sum(c==0 for c in old_coverage),
            v2_objects_no_overlap=sum(c==0 for c in new_coverage),
            baseline_length_m=float(aa.length_m.sum()) if 'length_m' in aa else None,
            v2_length_m=float(bb.length_m.sum()) if 'length_m' in bb else None,
            invalid_geometries=int((~bb['_source_valid']).sum()) if '_source_valid' in bb else int((~bb.is_valid).sum()),
            invalid_after_metric_reprojection=int((~bb.is_valid).sum()),overlay_grid_m=.001)
        for tag,geometry in [('baseline_only',old.difference(new)),('v2_only',new.difference(old))]:
            if not geometry.is_empty:differences.append(dict(change_typ=kind,difference=tag,geometry=geometry))
    return report,differences


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('baseline');parser.add_argument('v2')
    args=parser.parse_args();root=args.directory
    summaries=[json.loads((root/f'{label}.json').read_text(encoding='utf-8')) for label in [args.baseline,args.v2]]
    assert summaries[0]['inputs']==summaries[1]['inputs'],'Different source fingerprints'
    results=[pickle.loads((root/f'{label}.pkl').read_bytes()) for label in [args.baseline,args.v2]]
    final=[gpd.read_file(root/label/'road_changes.shp') for label in [args.baseline,args.v2]]
    for frame,label in zip(final,[args.baseline,args.v2]):
        frame['_source_valid']=frame.is_valid
        objects=gpd.read_file(root/label/'network_assembly.gpkg',layer='change_objects')
        assert len(objects)==len(frame) and objects.change_typ.tolist()==frame.change_typ.tolist()
        frame['length_m']=objects.length_m.to_numpy()
    crs=final[0].estimate_utm_crs() if final[0].crs.is_geographic else final[0].crs
    final=[f.to_crs(crs) for f in final]
    raw=[gpd.GeoDataFrame(r[0][0],geometry='geometry',crs=crs) for r in results]
    report={}
    for name,frames in [('local_candidates',raw),('final_changes',final)]:
        report[name],diff=compare(*frames)
        if diff:gpd.GeoDataFrame(diff,geometry='geometry',crs=crs).to_file(root/f'{args.v2}_differences.gpkg',layer=name,driver='GPKG')
    old,new=summaries
    report['performance']=dict(baseline_wall=old['analyze_scenes_seconds'],v2_wall=new['analyze_scenes_seconds'],
        speedup_percent=100*(1-new['analyze_scenes_seconds']/old['analyze_scenes_seconds']),
        baseline_work=old['work'],v2_work=new['work'],baseline_samples=old['local_sample_count'],v2_samples=new['local_sample_count'],
        baseline_finalization=old.get('finalization_seconds'),v2_finalization=new.get('finalization_seconds'))
    (root/f'{args.v2}_comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(10,15))
    colors=dict(added='#27b77b',removed='#e96586',widened='#dfb738',narrowed='#7e7ac7')
    for row,kind in enumerate(KINDS):
        frames=final
        choices=[f.loc[f.change_typ==kind] for f in frames]
        # If formal qualification publishes no feature, show candidate shape,
        # explicitly labeled, rather than fabricate a formal example.
        stage='final'
        if not any(len(f) for f in choices):
            frames=raw;choices=[f.loc[f.change_typ==kind] for f in frames];stage='candidate'
        labels=['Baseline / '+stage,'Fast v2 / '+stage]
        if not len(choices[0]) and len(raw[0].loc[raw[0].change_typ==kind]):
            choices[0]=raw[0].loc[raw[0].change_typ==kind]
            labels[0]='Baseline / candidate (no final)'
        base=choices[0] if len(choices[0]) else choices[1]
        if not len(base):
            for ax in axes[row]:ax.set_title(f'{kind}: no objects');ax.axis('off')
            continue
        selected=base.geometry.area.idxmax()
        if len(choices[0]) and len(choices[1]):
            other=union_all([overlay_polygon(g) for g in choices[1].geometry])
            shared=base.geometry.map(lambda g:overlay_polygon(g).intersection(other).area)
            if shared.max()>0:selected=shared.idxmax()
        geom=base.loc[selected].geometry
        p=geom.representative_point();span=min(450,max(80,np.sqrt(geom.area)*4))
        window=box(p.x-span/2,p.y-span/2,p.x+span/2,p.y+span/2)
        for column,(frame,label) in enumerate(zip(choices,labels)):
            ax=axes[row,column]
            visible=frame.loc[frame.intersects(window)].copy()
            if len(visible):visible.plot(ax=ax,facecolor=colors[kind],edgecolor='#333333',linewidth=.65)
            ax.set_xlim(window.bounds[0],window.bounds[2]);ax.set_ylim(window.bounds[1],window.bounds[3])
            ax.set_aspect('equal');ax.set_title(f'{label} / {kind}');ax.axis('off')
    fig.tight_layout();fig.savefig(root/f'{args.v2}_shape_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
