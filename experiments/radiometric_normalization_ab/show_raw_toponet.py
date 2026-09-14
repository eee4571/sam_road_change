"""Display cached TopoNet predictions directly, bypassing all downstream road stages.

No inference, graph repair, simplification, masking, deduplication or threshold tuning.
The stored graph already uses the original inference candidate generation, patch vote
aggregation, score > 0.5 decision and image-boundary filtering. Fast candidate generation
uses enhanced road probability; these are not outputs of an unmodified upstream SAMRoad.
"""
from pathlib import Path
import os
from run_experiment import ROOT, environment, save, sha
os.environ.update(environment())

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
import shapely as sh

from irmad_rrn import OUT, BASE, read, paired_tiles

DEST = OUT/'raw_toponet'
ARMS = ('T1', 'Raw_T2', 'Normalized_T2')
LABELS = ('T1 · 原始 TopoNet', 'Raw T2 · 原始 TopoNet', 'IR-MAD T2 · 原始 TopoNet')
COLORS = ('#00dcff', '#ff542e', '#ff542e')


def export_graphs():
    pairs=list(paired_tiles())
    sources={
        'T1':BASE/'20250118/runs/roads/inference/road_graphs/grid_tiles/graph',
        'Raw_T2':BASE/'20260203/runs/roads/inference/road_graphs/grid_tiles/graph',
        'Normalized_T2':OUT/'T2/runs/roads/inference/road_graphs/grid_tiles/graph',
    }
    layers, audit = {}, dict(stage='saved TopoNet output before downstream road postprocessing',
        connection_threshold='original score > 0.5; no additional filtering',
        upstream='Existing Fast enhanced-probability candidate points and model inference unchanged',
        downstream_postprocessing=False, inference_rerun=False, geometry_changes='pixel-center coordinates to native CRS only', arms={})
    gpkg=DEST/'raw_toponet.gpkg'
    if gpkg.exists(): raise FileExistsError(gpkg)
    for arm in ARMS:
        edge_frames,node_frames,records=[],[],[]
        for ref,target in pairs:
            image={'T1':ref,'Raw_T2':target,'Normalized_T2':OUT/'normalized_tiles'/target.name}[arm]
            source=sources[arm]/f'{target.stem}_fast_topology.npz'
            digest=sha(source)
            with np.load(source,allow_pickle=False) as z:
                nodes,edges,scores=z['nodes'].copy(),z['edges'].copy(),z['scores'].copy()
            assert nodes.ndim==2 and nodes.shape[1]==2
            assert edges.shape==(len(scores),2)
            assert np.isfinite(nodes).all() and np.isfinite(scores).all()
            assert not len(edges) or (edges.min()>=0 and edges.max()<len(nodes))
            with rasterio.open(image) as ds:
                x,y=rasterio.transform.xy(ds.transform,nodes[:,0],nodes[:,1],offset='center')
                coords=np.column_stack([x,y])
                crs=ds.crs
            frame=gpd.GeoDataFrame(dict(tile=[target.stem]*len(edges),edge_id=np.arange(len(edges)),
                source_node=edges[:,0],target_node=edges[:,1],score=scores),geometry=sh.linestrings(coords[edges]),crs=crs)
            node_frame=gpd.GeoDataFrame(dict(tile=[target.stem]*len(nodes),node_id=np.arange(len(nodes))),
                                        geometry=sh.points(coords),crs=crs)
            edge_frames.append(frame); node_frames.append(node_frame)
            records.append(dict(tile=target.stem,source=str(source),sha256=digest,image=str(image),
                nodes=len(nodes),edges=len(edges),score_min=float(scores.min()) if len(scores) else None))
            assert sha(source)==digest
        frame=gpd.GeoDataFrame(pd.concat(edge_frames,ignore_index=True),crs=crs)
        nodes=gpd.GeoDataFrame(pd.concat(node_frames,ignore_index=True),crs=crs)
        frame.to_file(gpkg,layer=arm+'_edges',driver='GPKG')
        nodes.to_file(gpkg,layer=arm+'_nodes',driver='GPKG')
        # The export must retain every stored edge and its score, including duplicates.
        check=gpd.read_file(gpkg,layer=arm+'_edges')
        assert len(check)==len(frame)
        np.testing.assert_array_equal(check.score.to_numpy(),frame.score.to_numpy())
        assert sh.equals_exact(check.geometry.to_numpy(),frame.geometry.to_numpy(),0).all()
        layers[arm]=frame
        audit['arms'][arm]=dict(nodes=len(nodes),edges=len(frame),tiles=records,
            exported_edges_verified=True,total_edge_length_m=float(frame.to_crs('EPSG:32650').length.sum()))
        print(arm,len(nodes),'nodes',len(frame),'edges; export verified',flush=True)
    save(DEST/'manifest.json',audit)
    return layers,pairs


def background(ax,bounds,col,pairs):
    for ref,target in pairs:
        image=(ref,target,OUT/'normalized_tiles'/target.name)[col]
        with rasterio.open(image) as ds:
            overlap=(max(ds.bounds.left,bounds[0]),max(ds.bounds.bottom,bounds[1]),
                     min(ds.bounds.right,bounds[2]),min(ds.bounds.top,bounds[3]))
            if overlap[0]>=overlap[2] or overlap[1]>=overlap[3]: continue
            w=from_bounds(*overlap,transform=ds.transform).round_offsets().round_lengths()
            rgb=ds.read(window=w)
            b=rasterio.windows.bounds(w,ds.transform)
            ax.imshow(rgb.transpose(1,2,0),extent=(b[0],b[2],b[1],b[3]))


def draw_edges(ax,frame,bounds,color):
    ids=frame.sindex.query(sh.box(*bounds))
    if len(ids): frame.iloc[ids].plot(ax=ax,color=color,linewidth=.7)


def make_figures(layers,pairs):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    locations=pd.read_csv(OUT/'evaluation/hotspots.csv')
    transformer=Transformer.from_crs('EPSG:32650',layers['T1'].crs,always_xy=True)
    for overlay in (False,True):
        fig,axes=plt.subplots(len(locations),3,figsize=(15,4.8*len(locations)),squeeze=False,constrained_layout=True)
        for row,hotspot in locations.iterrows():
            x,y=hotspot.utm_x,hotspot.utm_y
            left,bottom=transformer.transform(x-230,y-230)
            right,top=transformer.transform(x+230,y+230)
            bounds=(min(left,right),min(bottom,top),max(left,right),max(bottom,top))
            for col,arm in enumerate(ARMS):
                ax=axes[row,col]
                background(ax,bounds,col,pairs)
                if overlay and col:
                    draw_edges(ax,layers['T1'],bounds,COLORS[0])
                draw_edges(ax,layers[arm],bounds,COLORS[col])
                ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[1],bounds[3])
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'位置 {row+1} | {LABELS[col]}')
            axes[row,0].set_ylabel('与此前对比图相同位置',fontsize=9)
        title=('青蓝：T1；橙红：T2；右两列叠加 T1 原始 TopoNet' if overlay
               else '每列仅显示该影像自身的 TopoNet 预测边：T1 青蓝，T2 橙红')
        fig.suptitle(title+'\n未执行道路补线、删支、平滑、道路面约束或网络重建',fontsize=13)
        name='toponet_overlay_hotspots' if overlay else 'toponet_hotspots'
        fig.savefig(DEST/f'{name}.png',dpi=135)
        # A compact first-row view matches the user's screenshot exactly.
        for ax in axes[0]:
            ax.set_ylabel('')
        fig.canvas.draw()
        bounds=first_row_bbox(fig,axes[0])
        fig.savefig(DEST/f'{name}_first_region.png',dpi=135,bbox_inches=bounds)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,7),constrained_layout=True)
    bounds=layers['T1'].total_bounds
    for col,arm in enumerate(ARMS):
        layers[arm].plot(ax=axes[col],color=('#007dba' if col==0 else '#d84c2d'),linewidth=.22)
        axes[col].set_xlim(bounds[0],bounds[2]); axes[col].set_ylim(bounds[1],bounds[3])
        axes[col].set_title(LABELS[col]); axes[col].set_xticks([]); axes[col].set_yticks([])
    fig.suptitle('全区原始 TopoNet 预测边（未做道路后处理）')
    fig.savefig(DEST/'toponet_overview.png',dpi=140)
    plt.close(fig)


def first_row_bbox(fig,axes):
    from matplotlib.transforms import Bbox
    renderer=fig.canvas.get_renderer()
    return Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(fig.dpi_scale_trans.inverted()).expanded(1.01,1.04)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    layers,pairs=export_graphs()
    make_figures(layers,pairs)
    manifest=read(DEST/'manifest.json')
    lines=['# 原始 TopoNet 直接展示','',
        '本次仅导出并绘制已保存的模型图，关闭本次展示中的全部下游道路后处理，不重新推理、不修改生产开关。',
        '原始图仍保留原推理候选点生成（Fast 概率增强）、模型滑窗分数汇总、连接置信度 >0.5 和图像边界过滤；不是未阈值化的全部候选边。',
        '不执行道路面约束、补线、删支、骨架替代、平滑、网络重建、测宽、Fast2 或 Temporal。',
        '源 NPZ 哈希和导出边数、坐标、置信度均检查通过。每条边保持原样，未合并或去重；边数不是道路对象数。','',
        '| 输入 | 原始节点数 | 原始预测边数 |','|---|---:|---:|']
    for arm in ARMS:
        d=manifest['arms'][arm]; lines.append(f"| {arm} | {d['nodes']} | {d['edges']} |")
    lines += ['', '独立显示各自预测：','![独立显示](toponet_hotspots.png)','',
              '叠加 T1 便于位置对照：','![叠加对照](toponet_overlay_hotspots.png)','',
              '全区：','![全区](toponet_overview.png)','',
              'GIS 文件：`raw_toponet.gpkg`，含 T1、Raw_T2、Normalized_T2 各自的 edges 和 nodes 图层。',
              '三个局部范围沿用上轮截图的位置，不根据此次 TopoNet 结果重新选择。所有数据、图片和报告继续被实验 .gitignore 排除。']
    (DEST/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('RAW TOPONET DISPLAY COMPLETE',flush=True)


if __name__=='__main__': main()
