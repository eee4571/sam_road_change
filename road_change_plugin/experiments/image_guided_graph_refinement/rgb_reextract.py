"""Local RGB road-area reconstruction with cached extraction as a soft search prior.

No model inference, training, width measurement or formal product writes.
The local output is a new graph, not a splice into the production network.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib
import uuid

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize, shapes
from scipy.ndimage import distance_transform_edt, gaussian_filter, convolve, label
from shapely import union_all
from shapely.geometry import LineString, box, shape
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects, disk
import matplotlib.pyplot as plt

from bootstrap import ROOT
from data import discover, read_json, fingerprint, load_lines
from evidence import ImageGrid
from report import background, draw, write_json
from medial_repair import line_parts
from repair_artifacts import geometry_products


DEFAULT_REGIONS = [
    dict(id='industrial', bbox_m=[260080, 2614200, 260480, 2614540]),
    dict(id='main_road', bbox_m=[261580, 2614800, 261860, 2615100]),
    dict(id='internal', bbox_m=[260900, 2615280, 261200, 2615580]),
    dict(id='garden_control', bbox_m=[259260, 2613230, 259530, 2613500]),
]


def raw_graph(inputs, crs, roi):
    """Read original cached TopoNet nodes (row, column) with tile georeferencing."""
    run = Path(inputs['record']['run_root'])
    records, files = [], []
    for summary in sorted((run/'width_review').glob('v*_summary.json')):
        row = read_json(summary)
        tile = Path(row['image'])
        graph = run/'inference/road_graphs/grid_tiles/graph'/f"{row['stem']}_fast_topology.npz"
        if not graph.exists() or not tile.exists():
            continue
        with rasterio.open(tile) as ds:
            footprint = gpd.GeoSeries([box(*ds.bounds)], crs=ds.crs).to_crs(crs).iloc[0]
            if not footprint.intersects(roi):
                continue
            z = np.load(graph, allow_pickle=False)
            nodes, edges = z['nodes'], z['edges']
            x, y = ds.transform*(nodes[:, 1]+.5, nodes[:, 0]+.5)
            x, y = Transformer.from_crs(ds.crs, crs, always_xy=True).transform(x, y)
            xy = np.column_stack([x, y])
            for a, b in edges:
                g = LineString(xy[[a, b]])
                if g.intersects(roi):
                    records.append(dict(tile=row['stem'], geometry=g))
        files.extend([graph, tile, summary])
    if not records:
        raise ValueError('No georeferenced original graph cache for this ROI')
    return gpd.GeoDataFrame(records, geometry='geometry', crs=crs), files


def odd(m, resolution):
    return max(3, int(round(m/resolution))//2*2+1)


from rgb_material import reconstruct_area


def trace_skeleton(mask, transform, resolution):
    """Vectorize an 8-neighbor image skeleton, collapsing adjacent junction pixels."""
    skeleton = skeletonize(mask)
    pixels = set(map(tuple, np.argwhere(skeleton)))
    def neighbors(p):
        r, c = p
        result = []
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            q = (r+dr, c+dc)
            # No redundant diagonal triangle around a cardinal pixel corner.
            if q in pixels and not (dr and dc and ((r+dr,c) in pixels or (r,c+dc) in pixels)):
                result.append(q)
        return result
    adjacency = {p:neighbors(p) for p in pixels}
    nodes = {p for p, nb in adjacency.items() if len(nb) != 2}
    visited, paths = set(), []
    for start in sorted(nodes)+sorted(pixels-nodes):
        for following in adjacency[start]:
            key = tuple(sorted((start, following)))
            if key in visited:
                continue
            chain = [start, following]; visited.add(key)
            while chain[-1] not in nodes and chain[-1] != start:
                choices = [q for q in adjacency[chain[-1]] if q != chain[-2]]
                if not choices: break
                q = choices[0]; key = tuple(sorted((chain[-1], q)))
                if key in visited: break
                visited.add(key); chain.append(q)
            if len(chain) >= 2:
                rc = np.asarray(chain)
                x,y = transform*(rc[:,1]+.5,rc[:,0]+.5)
                paths.append(LineString(np.c_[x,y]))
    return paths


def simplify_graph(paths, image, fields):
    """Remove only raster-edge teeth; do not delete all short or isolated roads."""
    from collections import Counter
    audit=[]
    for iteration in range(6):
        counts = Counter(p for line in paths for p in (line.coords[0], line.coords[-1]))
        kept=[]
        for line in paths:
            end = counts[line.coords[0]] == 1 or counts[line.coords[-1]] == 1
            if end and line.length < 5.:
                audit.append(dict(action='raster_boundary_tooth', old_wkt=line.wkt, applied=True,
                                  reason='New area skeleton terminal under 5m; not an old-road length rule'))
            else: kept.append(line)
        if len(kept)==len(paths): break
        paths=kept
    # RDP bounded to <= 0.75 m, shared endpoints stay exactly fixed.
    return [p.simplify(.75) for p in paths], audit


def visual(folder, image, original, final, roi, area):
    evidence=SimpleNamespace(image=image)
    fig, axes=plt.subplots(1,3,figsize=(18,6))
    for ax, frame, title in zip(axes,[None,original,final],['Raw RGB','Original extraction cache','RGB area -> new graph']):
        background(ax,evidence)
        if frame is not None:
            for geometry in frame.geometry: draw(ax,geometry,color='#ffff20',linewidth=.9)
        ax.set(xlim=(roi.bounds[0],roi.bounds[2]),ylim=(roi.bounds[1],roi.bounds[3]),title=title,aspect='equal')
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(folder/'rgb_original_reextracted.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8)); background(ax,evidence)
    overlay=np.zeros((*area.shape,4)); overlay[area]=[0,1,1,.28]
    h,w=area.shape; x0,y1=image.transform*(0,0); x1,y0=image.transform*(w,h)
    ax.imshow(overlay,extent=(x0,x1,y0,y1),origin='upper')
    for geometry in final.geometry: draw(ax,geometry,color='#ffff20',linewidth=1)
    ax.set(xlim=(roi.bounds[0],roi.bounds[2]),ylim=(roi.bounds[1],roi.bounds[3]),title='RGB road area and reconstructed axes',aspect='equal'); ax.set_axis_off()
    fig.tight_layout();fig.savefig(folder/'road_area.png',dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--bbox',nargs=4,type=float)
    parser.add_argument('--resolution',type=float,default=.75)
    args=parser.parse_args()
    if not .4 <= args.resolution <= 2: parser.error('resolution must be 0.4..2m')
    saved=read_json(args.inputs)
    inputs=discover(saved['project'],saved['period'],saved['area'],record=saved['selected_period_record'])
    before_fingerprint=fingerprint(inputs)
    regions=[dict(id='custom',bbox_m=args.bbox)] if args.bbox else DEFAULT_REGIONS
    if args.only: regions=[r for r in regions if r['id'] in args.only]
    if not regions: parser.error('No selected region')
    output=ROOT/'outputs'/('rgb_reextract_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(output,flush=True)
    write_json(output/'inputs.json',dict(saved_input_manifest=str(args.inputs.resolve()),project=str(inputs['project']),
        period=inputs['period'],regions=regions,crs=saved['crs'],resolution_m=args.resolution,
        prior_role='original cached TopoNet, soft search extent only',fingerprints=before_fingerprint))
    (output/'rgb_reextract.py').write_bytes(Path(__file__).read_bytes())
    extra_fingerprints={}
    for row in regions:
        print(row['id'],flush=True)
        roi=box(*row['bbox_m']); context=roi.buffer(35)
        folder=output/row['id'];folder.mkdir()
        original,files=raw_graph(inputs,saved['crs'],context)
        for p in files: extra_fingerprints[str(p)]=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        image=ImageGrid(inputs['images'],saved['crs'],context,args.resolution)
        area,fields=reconstruct_area(image,original)
        paths=trace_skeleton(area,image.transform,args.resolution)
        paths,audit=simplify_graph(paths,image,fields)
        paths=[part for p in paths for part in line_parts(p.intersection(roi)) if part.length>.2]
        final=gpd.GeoDataFrame(dict(geometry=paths),geometry='geometry',crs=saved['crs'])
        original=original.copy();original.geometry=original.geometry.intersection(roi)
        original=original[~original.is_empty].explode(index_parts=False)
        original=original[original.geom_type=='LineString']
        original.to_file(folder/'original_cached_centerlines.gpkg',layer='centerlines',driver='GPKG')
        final.to_file(folder/'final_centerlines.gpkg',layer='centerlines',driver='GPKG')
        polygons=[shape(g).intersection(roi) for g,value in shapes(area.astype('uint8'),mask=area,transform=image.transform) if value]
        polygons=[g for g in polygons if not g.is_empty]
        gpd.GeoDataFrame(dict(geometry=polygons),crs=saved['crs']).to_file(folder/'rgb_road_area.gpkg',layer='rgb_area',driver='GPKG')
        audit.append(dict(action='rgb_area_reconstruction',applied=True,reason='Image-derived road area replaces local original graph; no old junctions forced',
            old_wkt=union_all(original.geometry).wkt,new_wkt=union_all(final.geometry).wkt))
        geometry_products(folder,original,final,audit)
        write_json(folder/'audit.json',audit)
        np.savez_compressed(folder/'rgb_evidence.npz',area=area,**fields)
        visual(folder,image,original,final,roi,area)
        write_json(folder/'summary.json',dict(row,original_segments=len(original),new_segments=len(final),
            interpretation='Unvalidated local RGB re-extraction; counts are not accuracy',
            invalid=int((~final.is_valid).sum()),self_intersections=int((~final.is_simple).sum())))
        print('saved',folder,flush=True)
    unchanged=before_fingerprint==fingerprint(inputs)
    cache_unchanged=all(Path(p).stat().st_size==s['bytes'] and Path(p).stat().st_mtime_ns==s['mtime_ns'] for p,s in extra_fingerprints.items())
    write_json(output/'verification.json',dict(original_inputs_unchanged=unchanged,raw_graph_cache_unchanged=cache_unchanged,
        raw_cache_fingerprints=extra_fingerprints,model_inference=False,width_recomputation=False,formal_product_writes=False))
    if not unchanged or not cache_unchanged: raise RuntimeError('Input changed during experiment')


if __name__=='__main__': main()
