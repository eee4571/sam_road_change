"""Show each image's own final, postprocessed centerlines in aligned panels."""
from show_raw_toponet import background, draw_edges, first_row_bbox
from irmad_rrn import OUT, BASE, read, paired_tiles
from run_experiment import save, sha
from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer


def main():
    output=OUT/'postprocessed_comparison'
    output.mkdir(exist_ok=True)
    results=[read(BASE/'20250118/latest_result.json'),read(BASE/'20260203/latest_result.json'),
             read(OUT/'T2/latest_result.json')]
    sources=[Path(r['centerlines']) for r in results]
    identities={str(p):sha(p) for p in sources}
    pairs=list(paired_tiles())
    roads=[gpd.read_file(p) for p in sources]
    crs=roads[0].crs
    roads=[r.to_crs(crs) for r in roads]
    positions=pd.read_csv(OUT/'evaluation/hotspots.csv')
    transformer=Transformer.from_crs('EPSG:32650',crs,always_xy=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.unicode_minus':False})
    labels=['T1（后处理后）','T2 原始影像（后处理后）','T2 IR-MAD 归一化（后处理后）']
    colors=['#00dcff','#ff542e','#ff542e']
    fig,axes=plt.subplots(len(positions),3,figsize=(15,4.8*len(positions)),squeeze=False,constrained_layout=True)
    for row,position in positions.iterrows():
        x,y=position.utm_x,position.utm_y
        left,bottom=transformer.transform(x-230,y-230)
        right,top=transformer.transform(x+230,y+230)
        bounds=(min(left,right),min(bottom,top),max(left,right),max(bottom,top))
        for col,frame in enumerate(roads):
            ax=axes[row,col]
            background(ax,bounds,col,pairs)
            # Exactly one road layer per panel; no T1 overlay on either T2 panel.
            draw_edges(ax,frame,bounds,colors[col])
            ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[1],bounds[3])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'位置 {row+1} | {labels[col]}')
    fig.suptitle('每张图只显示自身的最终道路中心线；青蓝色：T1，橙红色：对应 T2',fontsize=13)
    fig.savefig(output/'postprocessed_roads_all_regions.png',dpi=135)
    fig.canvas.draw()
    for row in range(len(positions)):
        fig.savefig(output/f'postprocessed_roads_region_{row+1}.png',dpi=135,
                    bbox_inches=first_row_bbox(fig,axes[row]))
    plt.close(fig)
    assert identities=={str(p):sha(p) for p in sources}
    save(output/'manifest.json',dict(stage='final postprocessed centerlines',
         panels=[dict(label=label,source=str(source)) for label,source in zip(labels,sources)],
         reference_overlay=False,source_hashes=identities,sources_unchanged=True,inference_rerun=False))
    print(output/'postprocessed_roads_region_1.png')


if __name__=='__main__': main()
