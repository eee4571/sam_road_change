"""Read-only image evidence for proposed regional connectors (no inference)."""
from pathlib import Path
import numpy as np
from .geometry_memory import bounded_linspace
from shapely import covers, points, line_interpolate_point
from shapely.geometry import LineString
from shapely.prepared import prep


class RoadProbability:
    def __init__(self, sources, crs, unit=1.):
        # Sources are (probability raster, georeferenced image or None).
        import rasterio
        from pyproj import Transformer
        from collections import OrderedDict
        self.tiles=[];self.unit=unit;self.png_cache=OrderedDict();self.cache_bytes=0
        for path,image in sources:
            path=Path(path)
            if not path.is_file(): continue
            with rasterio.open(image or path) as ds:
                transform,shape,local=ds.transform,ds.shape,ds.crs
            if local is None: raise ValueError(f'Probability image lacks CRS: {image or path}')
            self.tiles.append((path,transform,shape,Transformer.from_crs(crs,local,always_xy=True)))

    def values(self, xy):
        import rasterio
        from rasterio.windows import Window
        from .width.raw_memory import bounded_gdal_cache
        xy=xy/self.unit
        values=np.full(len(xy),np.nan)
        for path,transform,shape,project in self.tiles:
            x,y=project.transform(xy[:,0],xy[:,1]);col,row=(~transform)*(np.asarray(x),np.asarray(y))
            row=np.floor(row).astype(int);col=np.floor(col).astype(int)
            selected=np.flatnonzero((row>=0)&(row<shape[0])&(col>=0)&(col<shape[1]))
            if not len(selected):continue
            r,c=row[selected],col[selected]
            # Group requested pixels into bounded native-coordinate windows;
            # never read the bounding rectangle between distant probes.
            ids=(r//512)*((shape[1]+511)//512)+c//512
            order=np.argsort(ids,kind='stable')
            starts=np.r_[0,np.flatnonzero(np.diff(ids[order]))+1,len(order)]
            with bounded_gdal_cache(),rasterio.open(path) as ds:
                from .raster_memory import check_native_blocks
                check_native_blocks(ds)
                if ds.shape!=shape or ds.count!=1:raise ValueError(f'Probability/image grid mismatch: {path}')
                for start,stop in zip(starts[:-1],starts[1:]):
                    take=order[start:stop];rr,cc=r[take],c[take]
                    r0,c0=int(rr.min()),int(cc.min())
                    h,w=int(rr.max())-r0+1,int(cc.max())-c0+1
                    key=(path,r0,c0,h,w)
                    block=self.png_cache.pop(key,None)
                    if block is None:
                        block=ds.read(1,window=Window(c0,r0,w,h),masked=True)
                        size=block.data.nbytes+np.ma.getmaskarray(block).nbytes
                        while self.png_cache and self.cache_bytes+size>128*1024*1024:
                            _,old=self.png_cache.popitem(last=False)
                            self.cache_bytes-=old.data.nbytes+np.ma.getmaskarray(old).nbytes
                        self.cache_bytes+=size
                    self.png_cache[key]=block
                    sampled=block[rr-r0,cc-c0].astype(float).filled(np.nan)
                    if block.dtype==np.uint8:sampled/=255.
                    values[selected[take]]=np.fmax(values[selected[take]],sampled)
        return values


class ArrayRoadProbability:
    def __init__(self,sources,source_crs,metric_crs,unit=1.):
        from pyproj import Transformer
        self.sources=sources;self.transform=Transformer.from_crs(metric_crs,source_crs,always_xy=True);self.unit=unit

    def values(self,xy):
        x,y=self.transform.transform(xy[:,0]/self.unit,xy[:,1]/self.unit)
        values=np.full(len(xy),np.nan)
        for source in self.sources:
            col,row=(~source['transform'])*(np.asarray(x),np.asarray(y))
            row=np.floor(row).astype(int);col=np.floor(col).astype(int);h,w=source['mask'].shape
            selected=np.flatnonzero((row>=0)&(row<h)&(col>=0)&(col<w))
            values[selected]=np.fmax(values[selected],source['mask'][row[selected],col[selected]])
        return values


class ConnectionEvidence:
    short_gap_m=80.
    maximum_gap_m=150.
    def __init__(self,surface=None,probability=None,molra=None):
        self.surface=prep(surface) if surface is not None and not surface.is_empty else None
        self.probability=probability
        self.molra=molra

    def evaluate(self,curve,row):
        measured=self.measure(curve)
        # Preserve actual approach-facing evidence when supplied by endpoints.
        direction=min(measured['direction_cosine'],row.get('direction_cosine',1.))
        row.update(measured);row['direction_cosine']=direction
        d=measured['gap_length_m'];s=measured['surface_support'];p=measured['probability_support']
        joint=measured['joint_support'];hole=measured['unsupported_run_m']
        if d<=self.short_gap_m:
            # Continuity-first: upstream candidate generation and topology
            # selection supply geometric constraints. Image absence is audit
            # information, never a veto for a local connector.
            accepted=True;reason='short_gap_continuity_priority'
        elif d>self.maximum_gap_m:
            accepted=False;reason='excessive_gap_length'
        else:
            # Intermediate gaps still require sustained image support.
            accepted=(joint>=.85 and (s>=.85 or (p>=.8 and s>=.4)) and hole<=18 and direction>=.94)
            reason='long_gap_strong_image_support' if accepted else 'long_gap_unsupported_or_discontinuous'
        row['decision_reason']=reason
        row['needs_review']=False
        return bool(accepted)

    def measure(self,curve):
        line=curve if isinstance(curve,LineString) else LineString(curve)
        length=line.length
        distances=bounded_linspace(0,length,max(5,int(length/3)+1))
        xy=np.asarray([p.coords[0] for p in line_interpolate_point(line,distances)])
        inside=np.zeros(len(xy),bool) if self.surface is None else covers(self.surface.context,points(xy))
        raw_surface_support=float(inside.mean())
        molra_values=np.full(len(xy),np.nan) if self.molra is None else self.molra.values(xy)
        molra_valid=np.isfinite(molra_values)
        # Local prompted masks can omit a real road. Positive support combines
        # with the existing surface; a zero in one source is not a veto.
        inside=inside | (molra_valid & (molra_values>=.5))
        # Original probability ridge may differ slightly from the final axis.
        tangent=np.gradient(xy,axis=0);norm=np.maximum(np.linalg.norm(tangent,axis=1),1e-9)
        normal=np.column_stack([-tangent[:,1],tangent[:,0]])/norm[:,None]
        probes=np.concatenate([xy,xy+normal*1.5,xy-normal*1.5])
        p=np.full(len(probes),np.nan) if self.probability is None else self.probability.values(probes)
        p=p.reshape(3,-1);valid=np.isfinite(p).any(0)
        p=np.max(np.where(np.isfinite(p),p,-1),axis=0)
        supported=inside | (p>=.2)
        runs=np.diff(np.r_[0,np.flatnonzero(supported)+1,len(supported)+1])-1
        chord=xy[-1]-xy[0];chord/=max(np.linalg.norm(chord),1e-9)
        facing=min(float(chord@tangent[i]/norm[i]) for i in (0,-1))
        return dict(gap_length_m=float(length),surface_support=float(inside.mean()),
                    raw_surface_support=raw_surface_support,molra_valid_fraction=float(molra_valid.mean()),
                    molra_surface_support=float((molra_values[molra_valid]>=.5).mean()) if molra_valid.any() else 0.,
                    direction_cosine=facing,
                    probability_valid_fraction=float(valid.mean()),probability_mean=float(p[valid].mean()) if valid.any() else 0.,
                    probability_q25=float(np.quantile(p[valid],.25)) if valid.any() else 0.,
                    probability_support=float(((p>=.2)&valid).mean()),
                    joint_support=float(supported.mean()),unsupported_run_m=float(runs.max(initial=0)*length/max(1,len(xy)-1)))


class ConnectorPropagation:
    """Track synthetic geometry locally, not entire original road identities."""
    def __init__(self):
        self.lines=[];self.depths=[];self.tree=None;self.dirty=False

    def add(self,curve,depth):
        self.lines.append(LineString(curve));self.depths.append(depth);self.dirty=True

    def depth(self,curve):
        from shapely import STRtree
        if not self.lines:return 1
        if self.dirty:
            self.tree=STRtree(self.lines);self.dirty=False
        xy=np.asarray(curve)
        # Existing endpoint correction trims at most 7m. Only a connection
        # touching that neighbourhood inherits synthetic propagation depth.
        contacts=self.tree.query(points(xy[[0,-1]]),predicate='dwithin',distance=7.1)
        return 1+max((self.depths[int(i)] for i in contacts[1]),default=0)


def probability_sources(image_dir, probability_dir):
    if image_dir is None:return []
    return [(Path(probability_dir)/f'{p.stem}_centerline_probability.png',p)
            for p in sorted(Path(image_dir).glob('*.tif'))]


def molra_sources(width_dir):
    import json
    sources=[]
    for path in sorted(Path(width_dir).glob('*_summary.json')):
        record=json.loads(path.read_text(encoding='utf8'))
        mask=record.get('molra_surface_mask');image=record.get('image')
        if mask and image and Path(mask).is_file() and Path(image).is_file():sources.append((Path(mask),Path(image)))
    return sources
