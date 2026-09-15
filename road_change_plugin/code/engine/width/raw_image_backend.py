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

VERSION=1
FIELDS=('final_left_distance','final_right_distance','final_width','final_confidence','width_source','outlier_reason')

def original_images(image_dir):
    """Follow the existing IR-MAD provenance, never measure its normalized RGB."""
    directory=Path(image_dir).resolve()
    workspace=directory.parent/'input_manifest.json'
    if workspace.is_file():
        source=Path(json.loads(workspace.read_text(encoding='utf8')).get('source',''))
        if source.is_dir() and source.resolve()!=directory and (source/'normalized_cache.json').is_file():
            return original_images(source)
    marker=directory/'normalized_cache.json'
    if marker.is_file():
        data=json.loads(marker.read_text(encoding='utf8'))
        if data.get('irmad_identity'):
            directory=Path(data['source']).resolve()
    files=sorted(directory.glob('*.tif'))
    if not files:raise FileNotFoundError(f'Raw boundary width requires original images: {directory}')
    # Workspaces may use links to the normalized cache. Check resolved parent.
    parents={p.resolve().parent for p in files}
    if len(parents)==1 and next(iter(parents))!=directory:
        parent=next(iter(parents))
        if (parent/'normalized_cache.json').is_file():return original_images(parent)
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
    from contextlib import ExitStack
    import rasterio
    from rasterio.merge import merge
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    files=original_images(image_dir)
    c=Config(max_roads=0,min_length=0);rc=ReconstructionConfig()
    identities=signature([*files,__file__,Path(__file__).with_name('raw_boundary_core.py'),
        Path(__file__).with_name('raw_width_reconstruction.py'),Path(__file__).with_name('raw_width_records.py')])
    import hashlib
    identities += [str(VERSION),str(centerlines.crs),hashlib.sha256(b''.join(centerlines.geometry.to_wkb())).hexdigest(),
                   json.dumps(asdict(c),sort_keys=True),json.dumps(asdict(rc),sort_keys=True)]
    marker=output/'completed.json';cached=read_completed(marker,identities)
    if cached:return gpd.read_file(cached['observations'],layer='width_observations')
    started=time.perf_counter();source=output/'final_centerline_input.gpkg'
    centerlines.to_file(source,layer='centerlines',driver='GPKG')
    roads,junctions,crs,_=load_roads(source,'centerlines',c)
    image=files[0]
    if len(files)>1:
        image=output/'regional_rgb.tif'
        with ExitStack() as stack:
            datasets=[stack.enter_context(rasterio.open(p)) for p in files]
            first=datasets[0]
            for ds in datasets:
                if ds.crs!=first.crs or ds.res!=first.res:raise ValueError('Raw boundary images must share CRS and pixel resolution')
            merge(datasets,dst_path=image,mem_limit=64,dst_kwds={'compress':'LZW','tiled':True,'BIGTIFF':'IF_SAFER'})
    reader=ImageReader(image,crs);tree=cKDTree(junctions) if len(junctions) else None
    frames=[];road_map={}
    try:
        for i,line in enumerate(roads):
            road_id=f'road_{i:05d}';road_map[road_id]=line
            data=measure_road(reader,line,tree,c);flag_crossings(data)
            frame=pd.DataFrame(rows_for_road(road_id,data))
            frames.append(reconstruct_chain(frame,rc))
            if i%100==0:print(f'[Raw width] regional chains {i+1}/{len(roads)}',flush=True)
    finally:reader.ds.close()
    if not frames:raise ValueError('No regional roads available for raw boundary width')
    samples=pd.concat(frames,ignore_index=True);samples.to_csv(output/'samples.csv',index=False)
    observations=observation_segments(samples,road_map,crs).to_crs(centerlines.crs)
    path=output/'width_observations.gpkg';observations.to_file(path,layer='width_observations',driver='GPKG')
    manifest=dict(observations=str(path.resolve()),backend='raw_image',version=VERSION,
        config=asdict(c),reconstruction=asdict(rc),images=[str(p) for p in files],
        chains=len(roads),samples=len(samples),quality=observations.quality_grade.value_counts().to_dict(),
        seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    write_completed(marker,identities,manifest,[path,output/'samples.csv',output/'summary.json'])
    print(f'[Raw width] completed: {manifest}',flush=True)
    return observations

def rebuild_observations(measured):
    # Preserve continuous C-grade display profiles instead of reclassifying
    # them through the legacy robust A/B-only estimator.
    return measured.copy()
