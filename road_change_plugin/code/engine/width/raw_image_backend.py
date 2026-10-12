"""Regional RGB boundary backend, sharing the validated experiment core."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point,Polygon
from shapely import make_valid
from shapely.ops import substring
from .raw_boundary_core import Config, ImageReader, load_roads, measure_road
from .raw_width_reconstruction import ReconstructionConfig, reconstruct_chain
from .raw_width_records import rows_for_road, flag_crossings
from .raw_memory import RoadMemoryLimit, WORKSPACES
from .chain_width_calculator import resolve_chain_fallbacks, robust_width
from .width_reliability import (InputContractError, atomic_json, clean_frame,
                                issue, positive, project_default_width, quality_summary, require_road_local)

VERSION=4
FIELDS=('final_left_distance','final_right_distance','final_width','final_confidence','width_source','outlier_reason')

def original_images(image_dir):
    """Read the exact tiles used by inference, including IR-MAD RGB."""
    files=sorted(Path(image_dir).glob('*.tif'))
    if not files:raise FileNotFoundError(f'RGB width requires inference tiles: {image_dir}')
    return files


def quality(row):
    if row.get('quality_grade') in ('A','B','C'):return row['quality_grade']
    source=row['width_source']
    if source=='measured':return 'A'
    if source=='smoothed' and row['solver_converged'] and row['final_confidence']>=Config().confidence_threshold:return 'A'
    if source=='interpolated':return 'B'
    return 'C'

def _observation_segments(samples,roads,crs,diagnostics=None):
    diagnostics=[] if diagnostics is None else diagnostics
    records=[]
    for road_id,group in samples.groupby('road_id',sort=False):
        line=roads[road_id];group=group.sort_values('s_m')
        positions=np.array([line.project(Point(x,y)) for x,y in zip(group.center_x,group.center_y)])
        positions=np.maximum.accumulate(positions)
        edges=np.r_[0,(positions[1:]+positions[:-1])/2,line.length]
        # Shared endpoint sections eliminate butt-buffer seams. Interpolate
        # reconstructed distances, never measure or smooth the widths again.
        left=np.interp(edges,positions,group.final_left_distance)
        right=np.interp(edges,positions,group.final_right_distance)
        xy=np.array([line.interpolate(s).coords[0] for s in edges])
        tangent=np.array([np.array(line.interpolate(min(line.length,s+5)).coords[0])-
                          np.array(line.interpolate(max(0,s-5)).coords[0]) for s in edges])
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
        normal=np.column_stack([-tangent[:,1],tangent[:,0]])
        lxy=xy+normal*left[:,None];rxy=xy-normal*right[:,None]
        for i,(_,row) in enumerate(group.iterrows()):
            if edges[i+1]-edges[i]<=0:continue
            grade=quality(row);value=float(row.final_width)
            try:
                polygon=(make_valid(Polygon([lxy[i],lxy[i+1],rxy[i+1],rxy[i]]))
                         if np.isfinite([*lxy[i],*lxy[i+1],*rxy[i],*rxy[i+1]]).all() else None)
            except Exception as exc:
                diagnostics.append(issue(road_id,row.get('tile_id',''),'corridor',exc,
                                         geometry=line,fallback='symmetric_buffer'))
                polygon=None
                grade='C'
                row=row.copy()
                row['outlier_reason']=';'.join(filter(None,[str(row.get('outlier_reason','')),str(exc)]))
            records.append(dict(**{field:row[field] for field in FIELDS},
                raw_corridor_wkb=polygon.wkb_hex if polygon is not None else '',
                width_m=value,width_std=0.,width_quality=grade,quality_grade=grade,
                valid_ratio=1. if grade in ('A','B') else 0.,width_backend='raw_image',
                parent_id=road_id,segment_id=f'RGB{len(records):09d}',part_id=0,
                tile_id=str(row.get('tile_id','')), width_unit='m',
                line_source='observed',qa_state='review' if grade=='C' else 'auto',qa_reason=str(row.outlier_reason),
                length_m=edges[i+1]-edges[i],geometry=substring(line,edges[i],edges[i+1])))
    return gpd.GeoDataFrame(records,geometry='geometry',crs=crs)


def observation_segments(samples,roads,crs,diagnostics=None):
    diagnostics=[] if diagnostics is None else diagnostics
    frames=[]
    for key,group in samples.groupby('road_id',sort=False):
        try:
            frame=_observation_segments(group.copy(deep=True),{key:roads[key]},crs,diagnostics)
            if frame.empty:raise ValueError('Empty observation geometry')
        except Exception as exc:
            row=group.iloc[0];line=roads[key]
            diagnostics.append(issue(key,row.get('tile_id',''),'observation_geometry',exc,
                                     geometry=line,fallback='whole_axis_buffer'))
            value=_profile_width(group)
            frame=gpd.GeoDataFrame([dict(final_width=value,width_m=value,
                final_left_distance=value/2,final_right_distance=value/2,final_confidence=0.,
                width_source='same_road_median',outlier_reason=str(exc),raw_corridor_wkb='',
                width_std=0.,width_quality='C',quality_grade='C',valid_ratio=0.,width_backend='raw_image',
                parent_id=key,segment_id=key+'_fallback',part_id=0,tile_id=str(row.get('tile_id','')),
                width_unit='m',line_source='observed',qa_state='review',qa_reason=str(exc),
                length_m=line.length,geometry=line)],geometry='geometry',crs=crs)
        frames.append(frame)
    result=gpd.GeoDataFrame(pd.concat(frames,ignore_index=True),geometry='geometry',crs=crs)
    result['segment_id']=[f'RGB{i:09d}' for i in range(len(result))]
    return result

def _unresolved_profile(road_id,line,reason):
    # A single station represents the whole valid axis; no synthetic geometry.
    xy=line.interpolate(line.length/2).coords[0]
    return pd.DataFrame([dict(road_id=road_id,sample_id=0,s_m=line.length/2,
        center_x=xy[0],center_y=xy[1],normal_x=0.,normal_y=0.,
        final_left_distance=np.nan,final_right_distance=np.nan,final_width=np.nan,
        final_confidence=0.,width_source='unresolved',outlier_reason=reason,
        reconstruction_available=False,solver_converged=False)])


def _profile_width(frame,default_width_m=None):
    values=pd.to_numeric(frame.get('final_width',pd.Series(dtype=float)),errors='coerce')
    return robust_width(values) or (project_default_width() if default_width_m is None else default_width_m)


def _solve_road(road_id,line,reader,tree,config,reconstruction,tile_id):
    # Admission encloses all road-owned arrays, profile conversion and solver
    # copies. A refused reservation has not attempted an allocation.
    try:
        with WORKSPACES.reserve(reader.policy.road_bytes(line.length,config)):
            return _solve_road_owned(road_id,line,reader,tree,config,reconstruction,tile_id)
    except RoadMemoryLimit as exc:
        return (_unresolved_profile(road_id,line,str(exc)).assign(tile_id=tile_id),
                [issue(road_id,tile_id,'memory_preflight',exc,geometry=line)])


def _solve_road_owned(road_id,line,reader,tree,config,reconstruction,tile_id):
    """Single-road transaction. No checkpoint/publication writes inside this boundary."""
    errors=[];stage='sampling'
    try:
        data=measure_road(reader,line,tree,config);flag_crossings(data)
        windowed=data.get('memory_windowed',False)
        stage='profile_conversion'
        observations=pd.DataFrame(rows_for_road(road_id,data))
        del data
        if observations.empty:raise ValueError('No cross sections')
        stage='reconstruction'
        try:
            # The solver only owns this road's frame; preserve the accepted raw
            # observations even if an unexpected solver mutation precedes failure.
            frame=reconstruct_chain(observations.copy(deep=True),reconstruction)
        except Exception as exc:
            errors.append(issue(road_id,tile_id,stage,exc,geometry=line))
            stage='reconstruction_fallback'
            frame=observations.copy()
            good=frame.accepted & np.isfinite(frame.optimized_width) & frame.optimized_width.gt(0)
            for side in ('left','right'):
                frame[f'final_{side}_distance']=frame[f'optimized_{side}_distance'].where(good)
            frame['final_width']=frame.final_left_distance+frame.final_right_distance
            frame['final_confidence']=frame.optimized_confidence.where(good,0.)
            frame['width_source']=np.where(good,'measured','unresolved')
            frame['outlier_reason']=str(exc)
            frame['reconstruction_available']=good
            frame['solver_converged']=False
        stage='profile_validation'
        _validate_profile(frame,road_id)
        if windowed:
            reason='bounded_window_sampling: crop-dependent edge connectivity requires review'
            frame['quality_grade']='C';frame['final_confidence']=0.
            finite=np.isfinite(frame.final_width)&frame.final_width.gt(0)
            frame.loc[finite,'width_source']='bounded_rgb_estimate'
            frame['outlier_reason']=frame.outlier_reason.fillna('').map(lambda value:';'.join(filter(None,[value,reason])))
            errors.append(issue(road_id,tile_id,'bounded_window_sampling',RoadMemoryLimit(reason),geometry=line,
                                fallback='bounded_rgb_estimate'))
    except Exception as exc:
        errors.append(issue(road_id,tile_id,stage,exc,geometry=line))
        frame=_unresolved_profile(road_id,line,str(exc))
    frame['tile_id']=tile_id
    return frame,errors


def _validate_profile(frame,road_id):
    required=(*FIELDS,'road_id','s_m','center_x','center_y','solver_converged','reconstruction_available')
    if frame.empty or not set(required).issubset(frame):raise ValueError('Incomplete road profile')
    if not frame.road_id.eq(road_id).all():raise ValueError('Road profile ownership mismatch')
    if not np.isfinite(frame[['s_m','center_x','center_y']].to_numpy(float)).all():
        raise ValueError('Nonfinite profile coordinates')
    if len(frame)>1 and (np.diff(frame.s_m.to_numpy(float))<=0).any():
        raise ValueError('Profile stations must increase')
    # NaN width is an explicit request for fallback; nonnumeric attributes are a
    # local computation failure, not a reason to invalidate the region at merge.
    np.isfinite(frame[['final_width','final_left_distance','final_right_distance']].to_numpy())


def _fill_profiles(frames,roads,default_width_m,local_estimates=None,diagnostics=None):
    diagnostics=[] if diagnostics is None else diagnostics
    reliable={}
    for key,frame in frames.items():
        try:
            grades=frame.apply(quality,axis=1)
            valid=(np.isfinite(frame.final_width) & frame.final_width.gt(0) & grades.isin(['A','B'])
                   & np.isfinite(frame.final_left_distance) & frame.final_left_distance.gt(0)
                   & np.isfinite(frame.final_right_distance) & frame.final_right_distance.gt(0))
            reliable[key]=frame.loc[valid,'final_width'].to_numpy()
        except Exception as exc:
            tile_id=str(frame.iloc[0].get('tile_id','')) if len(frame) else ''
            diagnostics.append(issue(key,tile_id,'width_summary',exc,geometry=roads[key]))
            frames[key]=_unresolved_profile(key,roads[key],str(exc)).assign(tile_id=tile_id)
            reliable[key]=[]
    estimates=resolve_chain_fallbacks(roads,reliable,local_estimates=local_estimates,default_width_m=default_width_m)
    for key,frame in frames.items():
        try:
            frames[key]=_fill_road_profile(frame.copy(deep=True),estimates[key])
        except Exception as exc:
            tile_id=str(frame.iloc[0].get('tile_id',''))
            diagnostics.append(issue(key,tile_id,'width_interpolation',exc,geometry=roads[key]))
            estimate=estimates[key];value=estimate['width_m']
            frames[key]=_unresolved_profile(key,roads[key],str(exc)).assign(
                tile_id=tile_id,final_width=value,final_left_distance=value/2,final_right_distance=value/2,
                width_source=estimate['width_source'],quality_grade='C',reconstruction_available=True)
    for key in {error['road_id'] for error in diagnostics} & frames.keys():
        frames[key]['quality_grade']='C'
        frames[key]['final_confidence']=0.
    return frames


def _fill_road_profile(frame,estimate):
    # Caller gives this operation an owned frame; failed mutations are discarded.
    frame['quality_grade']=frame.apply(quality,axis=1)
    invalid=~(np.isfinite(frame.final_width) & frame.final_width.gt(0)
              & np.isfinite(frame.final_left_distance) & frame.final_left_distance.gt(0)
              & np.isfinite(frame.final_right_distance) & frame.final_right_distance.gt(0))
    if invalid.any():
        good=~invalid & frame.quality_grade.isin(['A','B'])
        if good.any():
            # Preserve asymmetric evidence when interpolating this road's own stations.
            for side in ('left','right'):
                frame.loc[invalid,f'final_{side}_distance']=np.interp(
                    frame.loc[invalid,'s_m'],frame.loc[good,'s_m'],frame.loc[good,f'final_{side}_distance'])
        else:
            for side in ('left','right'):frame.loc[invalid,f'final_{side}_distance']=estimate['width_m']/2
        frame.loc[invalid,'final_width']=frame.loc[invalid,'final_left_distance']+frame.loc[invalid,'final_right_distance']
        frame.loc[invalid,'width_source']=estimate['width_source']
        frame.loc[invalid,'quality_grade']=estimate['quality_grade']
        frame.loc[invalid,'final_confidence']=0.
        frame.loc[invalid,'reconstruction_available']=True
        frame.loc[invalid,'outlier_reason']=frame.loc[invalid,'outlier_reason'].fillna('').astype(str).map(
            lambda reason:';'.join(filter(None,[reason,estimate['fallback_reason']])))
    return frame


def measure_region(centerlines,image_dir,output_dir,*,default_width_m=None,surface_evidence=None):
    """One regional chain solve, including chains crossing source image tiles."""
    from ..product_cache import signature,read_completed,write_completed
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    default_width_m=project_default_width() if default_width_m is None else default_width_m
    if not positive(default_width_m):raise InputContractError('Project default width must be finite and positive (metres)')
    files=original_images(image_dir)
    c=Config(max_roads=0,min_length=0);rc=ReconstructionConfig()
    identities=signature([*files,__file__,Path(__file__).with_name('raw_boundary_core.py'),
        Path(__file__).with_name('raw_width_reconstruction.py'),Path(__file__).with_name('raw_width_records.py'),Path(__file__).with_name('raw_feature_cache.py'),
        Path(__file__).with_name('chain_width_calculator.py'),Path(__file__).with_name('width_reliability.py'),
        Path(__file__).with_name('raw_memory.py')])
    import hashlib
    from ..geometry_memory import json_geometry_digest
    identities += [str(VERSION),str(centerlines.crs),json_geometry_digest(centerlines.geometry),
                   json.dumps(asdict(c),sort_keys=True),json.dumps(asdict(rc),sort_keys=True),float(default_width_m)]
    if surface_evidence is not None:
        if surface_evidence.crs is None:raise InputContractError('Surface evidence requires CRS')
        identities += [str(surface_evidence.crs),json_geometry_digest(surface_evidence.geometry)]
    marker=output/'completed.json';cached=read_completed(marker,identities)
    if cached:return gpd.read_file(cached['observations'],layer='width_observations')
    marker.unlink(missing_ok=True)
    diagnostics=[]
    original_count=len(centerlines)
    centerlines=clean_frame(centerlines,diagnostics)
    atomic_json(output/'preflight.json',dict(input_roads=original_count,valid_roads=len(centerlines),exceptions=diagnostics))
    if centerlines.empty:
        raise InputContractError('No valid centerlines available for RGB width')
    started=time.perf_counter();source=output/'final_centerline_input.gpkg'
    centerlines.to_file(source,layer='centerlines',driver='GPKG')
    roads,junctions,crs,_=load_roads(source,'centerlines',c)
    if not roads:raise ValueError('No nonzero road chains available for RGB width; check extraction and validation area')
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    import os
    tree=cKDTree(junctions) if len(junctions) else None
    road_map={f'road_{i:05d}':line for i,line in enumerate(roads)}
    checkpoint_id=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()
    checkpoints=output/'chains'/checkpoint_id
    checkpoints.mkdir(parents=True,exist_ok=True)
    # Record exact contributing source tiles, including chains crossing tile seams.
    import rasterio
    from shapely.geometry import box
    tile_boxes=[]
    for file in files:
        with rasterio.open(file) as ds:
            bounds=gpd.GeoSeries([box(*ds.bounds)],crs=ds.crs).to_crs(crs).iloc[0]
            tile_boxes.append((file.stem,bounds))
    from shapely import STRtree
    tile_tree=STRtree([item[1] for item in tile_boxes])
    tile_ids={key:','.join(tile_boxes[int(i)][0] for i in tile_tree.query(line,predicate='intersects'))
              for key,line in road_map.items()}
    def solve(item):
        road_id,line=item
        checkpoint=checkpoints/(road_id+'.json')
        completed=checkpoint.with_suffix('.completed.json')
        cached=read_completed(completed,[road_id,line.wkb_hex])
        if cached is not None:
            try:
                frame=pd.read_json(checkpoint,orient='table')
                _validate_profile(frame,road_id)
                return frame,True,cached['exceptions']
            except Exception as exc:
                require_road_local(exc)
                print(f'[RGB width] Recomputing damaged checkpoint {road_id}: {exc}',flush=True)
        frame,errors=_solve_road(road_id,line,reader,tree,c,rc,tile_ids[road_id])
        # Serialization and checkpoint writes are outside road recovery. A failed
        # publication must never produce a completed marker or successful future.
        temporary=checkpoint.with_suffix('.tmp')
        frame.to_json(temporary,orient='table',double_precision=15,index=False)
        temporary.replace(checkpoint)
        write_completed(completed,[road_id,line.wkb_hex],dict(exceptions=errors),[checkpoint])
        return frame,False,errors
    print(f'[RGB width] start chains={len(roads)}; checkpoints={checkpoints}',flush=True)
    frames_by_id={};reused=0;last_report=time.perf_counter()
    reader=ImageReader(files,crs)
    try:
        from ..resource_policy import worker_count
        workers=worker_count(min(4,os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            items=iter(road_map.items())
            futures={}
            def submit_next():
                item=next(items,None)
                if item is not None:futures[pool.submit(solve,item)]=item[0]
            for _ in range(workers*2):submit_next()
            while futures:
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                future=next(iter(ready));road_id=futures.pop(future)
                try:
                    frame,was_reused,errors=future.result()
                except BaseException as error:
                    # Never recover at future.result(): escaped errors may come
                    # from shared resources, publication or cancellation.
                    import traceback
                    for pending in futures:pending.cancel()
                    failure=dict(road_id=road_id,axis_wkt=road_map[road_id].wkt,
                        exception_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
                    try:
                        (output/'failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2),encoding='utf8')
                    except OSError as report_error:
                        print(f'[RGB width] Could not save failure report: {report_error}',flush=True)
                    print(f'[RGB width] failed chain={road_id}: {error}; details={output / "failure.json"}',flush=True)
                    raise
                frames_by_id[road_id]=frame;reused+=int(was_reused)
                diagnostics.extend(errors)
                submit_next()
                if time.perf_counter()-last_report>=5 or len(frames_by_id)==len(roads):
                    print(f'[RGB width] {len(frames_by_id)}/{len(roads)} chains; reused={reused}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
                    last_report=time.perf_counter()
    finally:reader.close()
    local_estimates={}
    if surface_evidence is not None and not surface_evidence.empty:
        from .width_reliability import surface_width_estimates
        unresolved={key:road_map[key] for key,frame in frames_by_id.items()
                    if not (np.isfinite(frame.final_width)&frame.final_width.gt(0)).all()}
        local_estimates=surface_width_estimates(unresolved,surface_evidence.to_crs(crs),diagnostics,tile_ids)
    frames_by_id=_fill_profiles(frames_by_id,road_map,float(default_width_m),local_estimates,diagnostics)
    frames=[frames_by_id[key] for key in road_map]
    samples=pd.concat(frames,ignore_index=True)
    observations=observation_segments(samples,road_map,crs,diagnostics)
    for error in diagnostics:
        key=error['road_id']
        if key in frames_by_id:error['fallback']=','.join(sorted(set(frames_by_id[key].width_source)))
    summary=quality_summary(observations,diagnostics,total_input=original_count)
    observations=observations.to_crs(centerlines.crs)
    path=output/'width_observations.gpkg';temporary=output/'width_observations.partial.gpkg'
    temporary.unlink(missing_ok=True)
    observations.to_file(temporary,layer='width_observations',driver='GPKG')
    profile=gpd.GeoDataFrame(samples,geometry=gpd.points_from_xy(samples.center_x,samples.center_y),crs=crs)
    profile.to_file(temporary,layer='point_profiles',driver='GPKG')
    temporary.replace(path)
    (output/'samples.csv').unlink(missing_ok=True)
    manifest=dict(observations=str(path.resolve()),backend='raw_image',version=VERSION,
        config=asdict(c),reconstruction=asdict(rc),images=[str(p) for p in files],
        memory=dict(**WORKSPACES.snapshot(),policy=asdict(reader.policy),
                    max_observed_window_pixels=reader.feature_cache.max_window_pixels,
                    window_builds=reader.feature_cache.window_builds),
        feature_cache=dict(hits=reader.feature_cache.hits,misses=reader.feature_cache.misses,evictions=reader.feature_cache.evictions,bytes=reader.feature_cache.bytes,max_bytes=reader.feature_cache.max_bytes),
        chains=len(roads),reused_chains=reused,samples=len(samples),
        unresolved_chains=sum(not f.reconstruction_available.any() for f in frames),
        quality=observations.quality_grade.value_counts().to_dict(),
        seconds=time.perf_counter()-started,**summary)
    atomic_json(output/'summary.json',manifest)
    write_completed(marker,identities,manifest,[path,output/'summary.json'])
    (output/'failure.json').unlink(missing_ok=True)
    print(f'[RGB width] chains={len(roads)} samples={len(samples)} seconds={manifest["seconds"]:.3f}',flush=True)
    cache=manifest['feature_cache']
    print(f'[RGB feature cache] hit={cache["hits"]} miss={cache["misses"]} eviction={cache["evictions"]} bytes={cache["bytes"]}/{cache["max_bytes"]}',flush=True)
    return observations

def rebuild_observations(measured):
    """One robust A/B-supported representative width per junction-to-junction chain."""
    records=[];diagnostics=[]
    for parent,group in measured.groupby('parent_id',sort=False):
        try:
            records.append(_representative_observation(group.copy(deep=True)))
        except Exception as exc:
            first=group.iloc[0]
            diagnostics.append(issue(parent,first.get('tile_id',''),'representative_width',exc,
                                     geometry=first.geometry,fallback='retained_observation_segments'))
            # Never invent a connection across malformed profiles: keep each
            # original segment and its identifier instead of concatenating axes.
            value=_profile_width(group)
            for record in group.to_dict('records'):
                record.pop('raw_corridor_wkb',None)
                record.update(final_width=value,width_m=value,final_left_distance=value/2,
                    final_right_distance=value/2,final_confidence=0.,quality_grade='C',width_quality='C',
                    valid_ratio=0.,width_source='same_road_median',outlier_reason=str(exc),
                    qa_state='review',qa_reason=str(exc),width_std=0.)
                records.append(record)
    result=(gpd.GeoDataFrame(records,geometry='geometry',crs=measured.crs) if records else measured.iloc[:0].copy())
    result.attrs['width_diagnostics']=diagnostics
    return result


def _representative_observation(group):
    from shapely.geometry import LineString
    group=group.sort_values('segment_id',kind='stable')
    usable=group[group.quality_grade.isin(['A','B']) & np.isfinite(group.final_width)]
    evidence=usable if len(usable) else group[np.isfinite(group.final_width)]
    if evidence.empty:raise ValueError('No representative width evidence')
    weights=evidence.geometry.length.to_numpy()
    def median(field):
        values=evidence[field].to_numpy();order=np.argsort(values,kind='stable')
        return float(values[order[min(len(order)-1,np.searchsorted(np.cumsum(weights[order]),weights.sum()/2))]])
    left,right=median('final_left_distance'),median('final_right_distance')
    coords=[]
    for line in group.geometry:coords.extend(list(line.coords)[1:] if coords else list(line.coords))
    row=group.iloc[0].to_dict();row.pop('raw_corridor_wkb',None)
    grade=('A' if group.quality_grade.eq('A').all() else 'B') if group.quality_grade.isin(['A','B']).all() else 'C'
    row.update(geometry=LineString(coords),final_left_distance=left,final_right_distance=right,
        final_width=left+right,width_m=left+right,final_confidence=median('final_confidence'),quality_grade=grade,width_quality=grade,
        valid_ratio=float(usable.geometry.length.sum()/max(group.geometry.length.sum(),1e-9)),
        width_source=';'.join(sorted(set(group.width_source.astype(str)))),
        qa_state='review' if grade=='C' else 'auto',
        width_std=float(usable.final_width.std(ddof=0)) if len(usable) else 0.,
        length_m=float(group.geometry.length.sum()))
    return row
