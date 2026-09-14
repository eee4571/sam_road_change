"""Isolated Fast2 segment/interval ablation; no suppression or object assembly.

Reuse frozen Fast2 network correspondence, longitudinal coverage, stored-width
profile boundaries and corridor geometry. Retain only the base size/width criteria.
Do not call the production detector: it unconditionally attaches extra verifiers.
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

from run_experiment import ROOT, CODE, environment, save, sha
os.environ.update(environment())
sys.path.insert(0,str(CODE))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely as sh
from shapely.geometry import LineString
from shapely.ops import substring
from engine.fast_auto_change import RoadScene
from engine.fast_auto_v2 import _network_intervals, _paired_profiles
from engine.auto_presence_candidates import LongitudinalCoverage
from irmad_rrn import OUT, BASE, read

DEST=OUT/'fast2_basic'
PARAMS=dict(position_tolerance_m=3.,presence_position_factor=1.5,
            width_absolute_m=2.,width_relative=.2,minimum_length_m=24.,minimum_area_m2=4.)
KINDS=('added','removed','widened','narrowed')
DISABLED=['temporal_consistency','image_patch_verification','object_reconciliation',
          'segment_reconciliation','GT','width_background_bias_correction','width_variability_veto',
          'surface_probability_veto','object_assembly','additional_false_change_filters']


def period_paths():
    return dict(T1=BASE/'20250118/latest_result.json',Raw_T2=BASE/'20260203/latest_result.json',
                Normalized_T2=OUT/'T2/latest_result.json')


def input_identity():
    paths=[]
    for path in period_paths().values():
        paths.append(path)
        result=read(path)
        for kind in ('centerlines','surfaces','width_segments','valid_observation'):
            source=Path(result[kind])
            paths.extend(p for p in source.parent.glob(source.stem+'.*') if p.suffix in ('.shp','.shx','.dbf','.prj','.cpg'))
    return {str(p):sha(p) for p in sorted(set(paths))}


def load_scene(result):
    frames={key:gpd.read_file(result[key]).to_crs('EPSG:32650')
            for key in ('centerlines','surfaces','width_segments','valid_observation')}
    return RoadScene(frames['centerlines'],frames['surfaces'],frames['width_segments'],
                     frames['valid_observation'],None,frames['centerlines'].crs)


def width_records(axis_id,axis,profiles,after,params):
    records=[]
    for start,end,target_id,a,b,bw,aw in profiles:
        other=after.lines[target_id]
        delta=aw-bw
        threshold=np.maximum(params['width_absolute_m'],params['width_relative']*np.maximum(bw,aw))
        signs=np.where(abs(delta)>=threshold,np.sign(delta),0).astype(int)
        runs=[]
        for i,sign in enumerate(signs):
            if not sign:continue
            if runs and runs[-1][2]==sign and abs(runs[-1][1]-a[i])<1e-6:
                runs[-1]=(runs[-1][0],float(b[i]),int(sign),runs[-1][3]+[i])
            else:runs.append((float(a[i]),float(b[i]),int(sign),[i]))
        for low,high,sign,ids in runs:
            if high-low<params['minimum_length_m']:continue
            weights=b[ids]-a[ids]
            before_width=float(np.average(bw[ids],weights=weights))
            after_width=float(np.average(aw[ids],weights=weights))
            difference=after_width-before_width
            if abs(difference)<max(params['width_absolute_m'],params['width_relative']*max(before_width,after_width)):continue
            local=substring(axis,low,high)
            xy=np.asarray(local.coords)[:,:2]
            positions=sh.line_locate_point(other,sh.points(xy))
            partners=sh.get_coordinates(sh.line_interpolate_point(other,positions))
            canonical=LineString((xy+partners)/2)
            outer,inner=max(before_width,after_width),min(before_width,after_width)
            geometry=canonical.buffer(outer/2,cap_style='flat').difference(canonical.buffer(inner/2,cap_style='flat'))
            if geometry.area<params['minimum_area_m2']:continue
            records.append(dict(change_typ='widened' if difference>0 else 'narrowed',
                width_bef=before_width,width_aft=after_width,width_diff=difference,
                length_m=local.length,source_axis=axis_id,target_axis=target_id,start_m=low,end_m=high,
                axis_wkt=canonical.wkt,geometry=geometry,source_stage='base_width_profile_interval'))
    return records


def analyze(before,after,params=PARAMS):
    start=time.perf_counter();counts=Counter();records=[];profiles={};buffers={}
    for axis_id,axis in enumerate(before.lines):
        matched=_network_intervals(axis,after,params['position_tolerance_m'],buffers,counts)
        profiles[axis_id]=_paired_profiles(axis,matched,before,after,params['minimum_length_m'])
    timings=dict(profile_correspondence_seconds=time.perf_counter()-start)
    for side,source,target in ((0,before,after),(1,after,before)):
        tick=time.perf_counter()
        coverage=LongitudinalCoverage(target.lines,params['position_tolerance_m']*params['presence_position_factor'])
        for axis_id,axis in enumerate(source.lines):
            intervals=coverage.uncovered(axis)
            counts['uncovered_intervals']+=len(intervals)
            for a,b in intervals:
                if b-a<params['minimum_length_m']:continue
                part=substring(axis,a,b)
                width=float(np.median(source.widths_at(sh.line_interpolate_point(part,[0.,part.length/2,part.length]))))
                geometry=part.buffer(width/2,cap_style='flat')
                if geometry.area<params['minimum_area_m2']:continue
                records.append(dict(change_typ='removed' if side==0 else 'added',
                    width_bef=width if side==0 else 0.,width_aft=width if side==1 else 0.,width_diff=0.,
                    length_m=b-a,source_axis=axis_id,target_axis=-1,start_m=a,end_m=b,
                    axis_wkt=part.wkt,geometry=geometry,source_stage='base_uncovered_interval'))
        timings[('removed' if side==0 else 'added')+'_interval_seconds']=time.perf_counter()-tick
        print('Base presence',side,'complete',flush=True)
    tick=time.perf_counter()
    for axis_id,axis in enumerate(before.lines):
        records.extend(width_records(axis_id,axis,profiles[axis_id],after,params))
    timings['width_interval_seconds']=time.perf_counter()-tick
    timings['core_total_seconds']=time.perf_counter()-start
    return records,dict(counts),timings


def write_arm(name,records,counts,timings,elapsed):
    write_start=time.perf_counter()
    directory=DEST/name
    directory.mkdir(exist_ok=True)
    path=directory/'changes.gpkg'
    if path.exists():raise FileExistsError(path)
    frame=gpd.GeoDataFrame(records,geometry='geometry',crs='EPSG:32650')
    frame.insert(0,'interval_id',np.arange(len(frame)))
    frame['before_per']='20250118';frame['after_per']='20260203'
    frame.to_file(path,layer='changes',driver='GPKG')
    axes=frame.drop(columns='geometry').copy()
    axes=gpd.GeoDataFrame(axes,geometry=sh.from_wkt(frame.axis_wkt.to_numpy()),crs=frame.crs)
    axes.to_file(path,layer='interval_axes',driver='GPKG')
    columns=['interval_id','change_typ','width_bef','width_aft','width_diff','length_m','geometry']
    frame[columns].to_crs('EPSG:4490').to_file(directory/'road_changes.shp')
    timings['write_seconds']=time.perf_counter()-write_start
    per_kind={kind:dict(count=int((frame.change_typ==kind).sum()),
        length_m=float(frame.loc[frame.change_typ==kind,'length_m'].sum()),
        area_m2=float(frame.loc[frame.change_typ==kind].area.sum())) for kind in KINDS}
    summary=dict(arm=name,params=PARAMS,disabled=DISABLED,unit='one basic segment/interval per row; no object merge',
                 counts=per_kind,total_count=len(frame),timings=timings,detector_elapsed_seconds=elapsed,
                 diagnostics=counts,gpkg=str(path),prediction_sha256=sha(path),GT_used=False)
    save(directory/'result.json',summary)
    print(name,per_kind,flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--arms',nargs='+',choices=['raw','normalized'],default=['raw','normalized'])
    args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    initial=input_identity()
    if (DEST/'inputs_before.json').exists():assert read(DEST/'inputs_before.json')==initial
    else:save(DEST/'inputs_before.json',initial)
    config=dict(params=PARAMS,disabled=DISABLED,snapshot=str(CODE),
        source_hashes={str(p):sha(p) for p in [Path(__file__),CODE/'engine/fast_auto_v2.py',CODE/'engine/fast_auto_change.py',CODE/'engine/auto_presence_candidates.py']},
        existing_A_incompatible='Previously filtered with patch verifier and object reconciliation; no complete unfiltered candidates saved',
        A_basic_computation_authorized=True,periods={k:str(p) for k,p in period_paths().items()})
    if (DEST/'config.json').exists():
        previous=read(DEST/'config.json')
        assert previous['params']==PARAMS and previous['disabled']==DISABLED
    else:save(DEST/'config.json',config)
    scenes={};load_times={}
    for arm in args.arms:
        if (DEST/arm/'result.json').exists():
            result=read(DEST/arm/'result.json')
            assert sha(DEST/arm/'changes.gpkg')==result['prediction_sha256']
            print('Reuse completed base result:',arm,flush=True)
            continue
        start=time.perf_counter()
        names=['T1','Raw_T2' if arm=='raw' else 'Normalized_T2']
        new_load={}
        for name in names:
            if name not in scenes:
                tick=time.perf_counter();scenes[name]=load_scene(read(period_paths()[name]))
                load_times[name]=time.perf_counter()-tick
                new_load[name]=load_times[name]
        records,counts,timings=analyze(*(scenes[n] for n in names))
        timings['new_scene_load_seconds']=new_load
        write_arm(arm,records,counts,timings,time.perf_counter()-start)
        assert input_identity()==initial
    complete=all((DEST/arm/'result.json').exists() for arm in ('raw','normalized'))
    if complete:
        predictions={arm:sha(DEST/arm/'changes.gpkg') for arm in ('raw','normalized')}
        frozen_path=DEST/'predictions_frozen_before_GT.json'
        if frozen_path.exists():assert read(frozen_path)['predictions']==predictions
        else:save(frozen_path,dict(time=time.time(),input_files_unchanged=True,
            input_files=len(initial),GT_opened=False,predictions=predictions))
    print('BASE CHANGE DETECTION COMPLETE; GT NOT READ',flush=True)


if __name__=='__main__':main()
