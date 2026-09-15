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

VERSION=2
FIELDS=('final_left_distance','final_right_distance','final_width','final_confidence','width_source','outlier_reason')

def original_images(image_dir):
    """Read the exact tiles used by inference, including IR-MAD RGB."""
    files=sorted(Path(image_dir).glob('*.tif'))
    if not files:raise FileNotFoundError(f'RGB width requires inference tiles: {image_dir}')
    return files


def quality(row):
    source=row['width_source']
    if source=='measured':return 'A'
    if source=='smoothed' and row['solver_converged'] and row['final_confidence']>=Config().confidence_threshold:return 'A'
    if source=='interpolated':return 'B'
    return 'C'

def observation_segments(samples,roads,crs):
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
            if edges[i+1]-edges[i]<=1e-7:continue
            grade=quality(row);value=float(row.final_width)
            polygon=(make_valid(Polygon([lxy[i],lxy[i+1],rxy[i+1],rxy[i]]))
                     if np.isfinite([*lxy[i],*lxy[i+1],*rxy[i],*rxy[i+1]]).all() else None)
            records.append(dict(**{field:row[field] for field in FIELDS},
                raw_corridor_wkb=polygon.wkb_hex if polygon is not None else '',
                width_m=value,width_std=0.,width_quality=grade,quality_grade=grade,
                valid_ratio=1. if grade in ('A','B') else 0.,width_backend='raw_image',
                parent_id=road_id,segment_id=f'RGB{len(records):09d}',part_id=0,
                line_source='observed',qa_state='auto',qa_reason=str(row.outlier_reason),
                length_m=edges[i+1]-edges[i],geometry=substring(line,edges[i],edges[i+1])))
    return gpd.GeoDataFrame(records,geometry='geometry',crs=crs)

def measure_region(centerlines,image_dir,output_dir):
    """One regional chain solve, including chains crossing source image tiles."""
    from ..product_cache import signature,read_completed,write_completed
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    files=original_images(image_dir)
    c=Config(max_roads=0,min_length=0);rc=ReconstructionConfig()
    identities=signature([*files,__file__,Path(__file__).with_name('raw_boundary_core.py'),
        Path(__file__).with_name('raw_width_reconstruction.py'),Path(__file__).with_name('raw_width_records.py'),Path(__file__).with_name('raw_feature_cache.py')])
    import hashlib
    identities += [str(VERSION),str(centerlines.crs),hashlib.sha256(b''.join(centerlines.geometry.to_wkb())).hexdigest(),
                   json.dumps(asdict(c),sort_keys=True),json.dumps(asdict(rc),sort_keys=True)]
    marker=output/'completed.json';cached=read_completed(marker,identities)
    if cached:return gpd.read_file(cached['observations'],layer='width_observations')
    started=time.perf_counter();source=output/'final_centerline_input.gpkg'
    centerlines.to_file(source,layer='centerlines',driver='GPKG')
    roads,junctions,crs,_=load_roads(source,'centerlines',c)
    from concurrent.futures import ThreadPoolExecutor
    import os
    reader=ImageReader(files,crs);tree=cKDTree(junctions) if len(junctions) else None
    road_map={f'road_{i:05d}':line for i,line in enumerate(roads)}
    def solve(item):
        road_id,line=item
        data=measure_road(reader,line,tree,c);flag_crossings(data)
        return reconstruct_chain(pd.DataFrame(rows_for_road(road_id,data)),rc)
    try:
        with ThreadPoolExecutor(max_workers=min(4,os.cpu_count() or 1)) as pool:
            frames=list(pool.map(solve,road_map.items()))
    finally:reader.close()
    if not frames:raise ValueError('No regional roads available for raw boundary width')
    samples=pd.concat(frames,ignore_index=True)
    observations=observation_segments(samples,road_map,crs).to_crs(centerlines.crs)
    path=output/'width_observations.gpkg';observations.to_file(path,layer='width_observations',driver='GPKG')
    profile=gpd.GeoDataFrame(samples,geometry=gpd.points_from_xy(samples.center_x,samples.center_y),crs=crs)
    profile.to_file(path,layer='point_profiles',driver='GPKG')
    (output/'samples.csv').unlink(missing_ok=True)
    manifest=dict(observations=str(path.resolve()),backend='raw_image',version=VERSION,
        config=asdict(c),reconstruction=asdict(rc),images=[str(p) for p in files],
        feature_cache=dict(hits=reader.feature_cache.hits,misses=reader.feature_cache.misses,evictions=reader.feature_cache.evictions,bytes=reader.feature_cache.bytes,max_bytes=reader.feature_cache.max_bytes),
        chains=len(roads),samples=len(samples),quality=observations.quality_grade.value_counts().to_dict(),
        seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    write_completed(marker,identities,manifest,[path,output/'summary.json'])
    print(f'[RGB width] chains={len(roads)} samples={len(samples)} seconds={manifest["seconds"]:.3f}',flush=True)
    return observations

def rebuild_observations(measured):
    """One robust A/B-supported representative width per junction-to-junction chain."""
    from shapely.geometry import LineString
    records=[]
    for parent,group in measured.groupby('parent_id',sort=False):
        group=group.sort_values('segment_id',kind='stable')
        usable=group[group.quality_grade.isin(['A','B']) & np.isfinite(group.final_width)]
        evidence=usable if len(usable) else group[np.isfinite(group.final_width)]
        if evidence.empty:continue
        weights=evidence.geometry.length.to_numpy()
        def median(field):
            values=evidence[field].to_numpy();order=np.argsort(values,kind='stable')
            return float(values[order[min(len(order)-1,np.searchsorted(np.cumsum(weights[order]),weights.sum()/2))]])
        left,right=median('final_left_distance'),median('final_right_distance')
        coords=[]
        for line in group.geometry:coords.extend(list(line.coords)[1:] if coords else list(line.coords))
        row=group.iloc[0].to_dict();row.pop('raw_corridor_wkb',None)
        grade=('A' if (usable.quality_grade=='A').all() else 'B') if len(usable) else 'C'
        row.update(geometry=LineString(coords),final_left_distance=left,final_right_distance=right,
            final_width=left+right,width_m=left+right,final_confidence=median('final_confidence'),quality_grade=grade,width_quality=grade,
            valid_ratio=float(usable.geometry.length.sum()/max(group.geometry.length.sum(),1e-9)),
            width_source='representative',width_std=float(usable.final_width.std(ddof=0)) if len(usable) else 0.,
            length_m=float(group.geometry.length.sum()))
        records.append(row)
    return gpd.GeoDataFrame(records,geometry='geometry',crs=measured.crs) if records else measured.iloc[:0].copy()
