"""Load multi-period evidence locally instead of retaining all period geometry."""
import numpy as np
from shapely import STRtree, union_all
from shapely.geometry import box, LineString
from .disk_records import LazyFrames


class PeriodEvidence:
    def __init__(self,keys,loader):
        self.keys=list(keys);self.loader=loader
        self.trees=LazyFrames(self.keys,lambda key:STRtree(loader(key)),capacity=1)
    def __iter__(self):
        for key in self.keys:yield self.loader(key)
    def supported(self,points,distance):
        return any(np.mean([len(tree.query(p,predicate='dwithin',distance=distance))>0 for p in points])>=.85
                   for tree in self.trees.values())


class WindowedConnectionEvidence:
    def __init__(self,paths,crs):
        self.paths=paths;self.crs=crs
        self.surface=self.probability=self.molra=None
    def local(self,axis,width):
        import geopandas as gpd
        import pyogrio
        from pyproj import Transformer
        from shapely.ops import transform
        from .road_connection_evidence import ConnectionEvidence
        region=box(*axis.buffer(max(2*width,4.)).bounds)
        result=None
        for path in self.paths:
            source_crs=pyogrio.read_info(path)['crs']
            if not source_crs:raise ValueError(f'Road evidence lacks CRS: {path}')
            project=Transformer.from_crs(self.crs,source_crs,always_xy=True)
            frame=gpd.read_file(path,bbox=transform(project.transform,region).bounds).to_crs(self.crs)
            if frame.empty:continue
            current=union_all(frame.geometry.intersection(region).values)
            result=current if result is None else result.union(current)
        return ConnectionEvidence(result)
    def measure(self,curve):
        axis=curve if isinstance(curve,LineString) else LineString(curve)
        return self.local(axis,4.).measure(axis)
