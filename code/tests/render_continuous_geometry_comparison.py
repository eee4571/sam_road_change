"""Compare cached release and regenerated final boundaries at GT object extents."""
import json
import sys
from pathlib import Path
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely import union_all


def main():
    old,new=map(Path,sys.argv[1:3])
    manifests=[json.loads((p/'pipeline_result.json').read_text(encoding='utf-8')) for p in (old,new)]
    current=manifests[1]['change_results'][0]
    audit=gpd.read_file(current['event_geometry_audit'],layer='changes')
    crs=audit.crs
    frames=[]
    for m in manifests:
        frames.extend([gpd.read_file(m['final_period_results'][1]['surfaces']).to_crs(crs),
                       gpd.read_file(m['change_results'][0]['road_changes']).to_crs(crs)])
    directory=new/'comparison'
    for i,(_,group) in enumerate(audit[audit.change_src.eq('GT_ASSISTED')].groupby('truth_id')):
        extent=union_all(group.geometry).envelope.buffer(15)
        fig,axes=plt.subplots(2,2,figsize=(12,12))
        for ax,frame,title,color in zip(axes.flat,[frames[0],frames[2],frames[1],frames[3]],
                ['Previous period roads','Continuous period roads','Previous changes','Continuous changes'],
                ['#d7b839','#d7b839','#ed91a7','#ed91a7']):
            local=frame[frame.intersects(extent)]
            if len(local):local.plot(ax=ax,color=color,edgecolor='#222222',linewidth=.65)
            x0,y0,x1,y1=extent.bounds
            ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect('equal');ax.axis('off');ax.set_title(title)
        fig.tight_layout();fig.savefig(directory/f'boundary_comparison_{i}.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
