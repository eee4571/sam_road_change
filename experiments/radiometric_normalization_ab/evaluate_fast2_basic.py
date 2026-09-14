"""Offline GT evaluation, strictly after both base change products are frozen."""
import os
import time
import shutil
from pathlib import Path
from run_experiment import ROOT, REPO, environment, save, sha
os.environ.update(environment())
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely as sh
from shapely import STRtree
from irmad_rrn import OUT, read

DEST=OUT/'fast2_basic'
EVAL=DEST/'evaluation'
KINDS=('added','removed','widened','narrowed')


def gt_class(kind):
    return 'width_changed' if kind in ('widened','narrowed') else kind


def compare(pred,truth):
    tree=STRtree(truth.geometry.to_numpy())
    gt_hits=set();gt_typed_hits=set();records=[]
    for _,row in pred.iterrows():
        ids=tree.query(row.geometry,predicate='intersects')
        positive=[];typed=[];area=0.;typed_area=0.
        for i in ids:
            overlap=row.geometry.intersection(truth.geometry.iloc[i]).area
            if overlap>1e-6:
                positive.append(int(i));area+=overlap
                if truth.evaluation_type.iloc[i]==gt_class(row.change_typ):
                    typed.append(int(i));typed_area+=overlap
        gt_hits.update(positive);gt_typed_hits.update(typed)
        records.append(dict(interval_id=int(row.interval_id),change_typ=row.change_typ,
             length_m=row.length_m,area_m2=row.geometry.area,has_GT_overlap=bool(positive),
             has_same_type_GT_overlap=bool(typed),gt_ids=','.join(map(str,positive)),
             overlap_area_m2=area,same_type_overlap_area_m2=typed_area))
    table=pd.DataFrame(records)
    totals={}
    for kind in KINDS:
        rows=table.loc[table.change_typ==kind]
        no=rows.loc[~rows.has_GT_overlap]
        totals[kind]=dict(count=len(rows),no_GT_intersection_count=len(no),
            no_same_type_GT_count=int((~rows.has_same_type_GT_overlap).sum()),
            no_GT_length_m=float(no.length_m.sum()),no_GT_area_m2=float(no.area_m2.sum()),
            median_interval_length_m=float(rows.length_m.median()),p90_interval_length_m=float(rows.length_m.quantile(.9)),
            median_interval_area_m2=float(rows.area_m2.median()),
            GT_supported_count=int(rows.has_GT_overlap.sum()))
    polygons=[]
    for kind in ('added','removed','width_changed'):
        ids=truth.index[truth.evaluation_type==kind].tolist()
        polygons.append(dict(kind=kind,GT_count=len(ids),hit_any_type=sum(i in gt_hits for i in ids),
                             hit_same_type=sum(i in gt_typed_hits for i in ids)))
    coverage=[]
    for i,row in truth.iterrows():
        eligible=pred.loc[pred.change_typ.map(gt_class)==row.evaluation_type]
        idx=eligible.sindex.query(row.geometry,predicate='intersects')
        shapes=eligible.geometry.iloc[idx]
        intersection=sh.union_all([g.intersection(row.geometry) for g in shapes]).area if len(shapes) else 0.
        coverage.append(dict(gt_id=i,kind=row.evaluation_type,area_m2=row.geometry.area,covered_area_m2=intersection))
    return table,dict(counts=totals,total_count=len(table),no_GT_intersection_count=int((~table.has_GT_overlap).sum()),
           no_GT_intersection_fraction=float((~table.has_GT_overlap).mean()),GT_object_hits=polygons,
           typed_GT_area_coverage=sum(r['covered_area_m2'] for r in coverage)/max(sum(r['area_m2'] for r in coverage),1e-9)),coverage


def main():
    start=time.perf_counter()
    frozen=read(DEST/'predictions_frozen_before_GT.json')
    assert frozen['GT_opened'] is False
    for arm,digest in frozen['predictions'].items():assert sha(DEST/arm/'changes.gpkg')==digest
    EVAL.mkdir(exist_ok=True)
    save(EVAL/'GT_access_gate.json',dict(first_evaluation_started_at=time.time(),predictions_frozen=frozen,
         definition='Positive polygon intersection area > 1e-6 m2. No GT buffer. One object = one base interval row.'))
    # First access to GT configuration/data occurs only after the above gate.
    config=read(REPO/'project/test_area/project_config.json')
    source=Path(next(row[3] for row in config['area_truths'] if row[1:3]==['20250118','20260203']))
    truth_dir=EVAL/'gt';truth_dir.mkdir(exist_ok=True)
    for path in source.parent.glob(source.stem+'.*'):
        if path.suffix.lower() in ('.shp','.shx','.dbf','.prj','.cpg'):
            dest=truth_dir/('changes'+path.suffix)
            if dest.exists():assert sha(dest)==sha(path)
            else:shutil.copy2(path,dest)
    truth=gpd.read_file(truth_dir/'changes.shp').to_crs('EPSG:32650')
    truth['evaluation_type']=pd.to_numeric(truth['BHBM'],errors='coerce').map({2:'added',3:'width_changed',4:'removed'})
    truth=truth.loc[truth.evaluation_type.notna()].reset_index(drop=True)
    assert len(truth)>0
    truth.geometry=sh.make_valid(truth.geometry.to_numpy())
    summary=dict(method='post-hoc positive area intersection; GT never supplied to detector',
                  GT_source=str(source),GT_count=len(truth),GT_type_counts=truth.evaluation_type.value_counts().to_dict(),
                  caveat='No-overlap is an offline error proxy, not proof every interval is false. GT width class does not distinguish widening/narrowing.',arms={})
    for arm in ('raw','normalized'):
        pred=gpd.read_file(DEST/arm/'changes.gpkg',layer='changes')
        assert pred.geometry.is_valid.all()
        table,metrics,coverage=compare(pred,truth)
        table.to_csv(EVAL/f'{arm}_GT_overlap.csv',index=False)
        pd.DataFrame(coverage).to_csv(EVAL/f'{arm}_GT_coverage.csv',index=False)
        summary['arms'][arm]=metrics
        print(arm,metrics,flush=True)
    summary['consistency_reused_from']=str(OUT/'evaluation/metrics.json')
    summary['evaluation_seconds']=time.perf_counter()-start
    for arm,digest in frozen['predictions'].items():assert sha(DEST/arm/'changes.gpkg')==digest
    summary['predictions_unchanged_after_GT']=True
    save(EVAL/'metrics.json',summary)
    print('OFFLINE EVALUATION COMPLETE',flush=True)


if __name__=='__main__':main()
