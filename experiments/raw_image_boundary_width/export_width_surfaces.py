"""Export measured swath polygons and raster PNG overlays; no web output or surface model."""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from boundary_width import ImageReader, save_json


def build_surfaces(frame, crs):
    """Each face joins two consecutive accepted cross-sections on the same road.

    No centerline buffer, gap filling or extrapolation. Separate faces retain continuous
    along-road width variation; shared cross-sections join the faces exactly.
    """
    records, invalid = [], []
    for road_id, group in frame.groupby('road_id', sort=False):
        group = group.sort_values('sample_id')
        rows = list(group.itertuples())
        for a, b in zip(rows[:-1], rows[1:]):
            if not (a.accepted and b.accepted and b.sample_id == a.sample_id+1):
                continue
            polygon = Polygon([(a.optimized_left_x, a.optimized_left_y),
                               (b.optimized_left_x, b.optimized_left_y),
                               (b.optimized_right_x, b.optimized_right_y),
                               (a.optimized_right_x, a.optimized_right_y)])
            if not polygon.is_valid or polygon.area <= 0:
                invalid.append(dict(road_id=road_id, start_sample=int(a.sample_id), end_sample=int(b.sample_id)))
                continue
            records.append(dict(road_id=road_id, start_sample=int(a.sample_id), end_sample=int(b.sample_id),
                s_start_m=a.s_m, s_end_m=b.s_m, left_distance=(a.optimized_left_distance+b.optimized_left_distance)/2,
                right_distance=(a.optimized_right_distance+b.optimized_right_distance)/2,
                width=(a.optimized_width+b.optimized_width)/2,
                width_start=a.optimized_width, width_end=b.optimized_width,
                confidence=min(a.optimized_confidence,b.optimized_confidence), area_m2=polygon.area,
                geometry=polygon))
    if not records:
        raise ValueError('No adjacent accepted cross-sections can form a polygon')
    return gpd.GeoDataFrame(records,crs=crs),invalid


def main(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(folder/'samples.csv')
    polygons,invalid=build_surfaces(frame,manifest['metric_crs'])
    destination=folder/'width_surface_polygons.gpkg'
    if destination.exists():
        raise FileExistsError(destination)
    polygons.to_file(destination,layer='width_surfaces',driver='GPKG')
    # Render the actual exported vector dataset, rather than regenerating a separate mask.
    polygons=gpd.read_file(destination,layer='width_surfaces')
    assert polygons.geometry.is_valid.all() and (polygons.area > 0).all()
    np.testing.assert_allclose(polygons.width,polygons.left_distance+polygons.right_distance)
    reader=ImageReader(manifest['image'],manifest['metric_crs'])
    raw=reader.ds.read([1,2,3]).transpose(1,2,0)
    background=Image.fromarray(raw)
    del raw
    overlay=background.copy()
    draw=ImageDraw.Draw(overlay,'RGBA')
    cmap=plt.get_cmap('turbo')
    norm=matplotlib.colors.Normalize(2,36,clip=True)
    for row in polygons.itertuples():
        xy=reader.pixels(np.asarray(row.geometry.exterior.coords))
        rgb=tuple(round(v*255) for v in cmap(norm(row.width))[:3])
        draw.polygon([tuple(p) for p in xy],fill=rgb+(115,),outline=rgb+(210,),width=1)
    overlay.save(folder/'width_surface_overlay_native.png',compress_level=3)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})
    small=overlay.copy();small.thumbnail((3000,3000))
    fig,ax=plt.subplots(figsize=(12,13.8),constrained_layout=True)
    ax.imshow(small);ax.axis('off')
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.03,pad=.01,label='相邻横断面平均宽度（m）')
    fig.suptitle(f'全图测宽面矢量叠加原始影像\n{frame.road_id.nunique():,} 条道路链 · {len(polygons):,} 个有效面片 · 低置信缺口不填充')
    fig.savefig(folder/'width_surface_overview.png',dpi=210)
    fig.savefig(folder/'width_surface_preview.jpg',dpi=100)
    plt.close(fig)
    # Four local views are crops of exactly the same native PNG, with matching raw context.
    selected=[('road_063','窄路 / 偏心'),('road_094','渐缩 / 路口'),('road_064','阴影 / 建筑干扰'),('road_031','中心线贴近路缘')]
    fig,axes=plt.subplots(4,2,figsize=(14,19),constrained_layout=True)
    for pair,(road_id,label) in zip(axes,selected):
        f=frame[frame.road_id==road_id]
        if f.empty:
            for ax in pair: ax.axis('off')
            continue
        xy=reader.pixels(f[['center_x','center_y']].to_numpy())
        low=np.maximum(np.floor(xy.min(0)-65),0).astype(int)
        high=np.minimum(np.ceil(xy.max(0)+65),background.size).astype(int)
        box=(*low,*high)
        for ax,image,title in zip(pair,(background,overlay),('原始影像','测宽面矢量叠加')):
            ax.imshow(image.crop(box));ax.axis('off');ax.set_title(f'{road_id} {label}｜{title}')
    fig.suptitle('全图成果局部放大｜彩色区域为面矢量；颜色仅代表宽度，不能证明边界正确')
    fig.savefig(folder/'width_surface_details.png',dpi=180)
    plt.close(fig)
    reader.ds.close()
    stats=dict(roads=int(frame.road_id.nunique()),samples=len(frame),accepted=int(frame.accepted.sum()),
        accepted_fraction=float(frame.accepted.mean()),polygons=len(polygons),invalid_pairs_excluded=invalid,
        all_exported_geometries_valid=True,width_sum_verified=True,
        native_png_dimensions=list(overlay.size),metric_crs=manifest['metric_crs'],
        construction='quadrilateral between adjacent accepted left/right boundary samples; no gap filling',
        rendering='read back exported GeoPackage polygons and alpha-fill on original RGB pixels')
    save_json(folder/'width_surface_summary.json',stats)
    (folder/'WIDTH_SURFACES.md').write_text(f'''# 全图道路测宽面矢量叠加

- [全图概览 PNG](width_surface_overview.png)
- [原始像素尺寸叠加 PNG](width_surface_overlay_native.png)：{overlay.width}×{overlay.height}
- [局部原图与面矢量对比 PNG](width_surface_details.png)
- [面矢量 GeoPackage](width_surface_polygons.gpkg)：`width_surfaces` 图层，{manifest['metric_crs']}

全部 {stats['roads']:,} 条中心线链、{len(frame):,} 个采样点均已尝试测宽，其中 {stats['accepted_fraction']:.1%} 通过现有筛选。由连续有效横断面的左边界→下一左边界→下一右边界→右边界构成 {len(polygons):,} 个面片，保留实际左右不等距离，不使用中心线固定缓冲。

面属性包括左右距离、平均宽度、起止宽度、置信度、面积、道路编号和起止采样点。相邻面片共享横断面边界，低置信缺口不连接、不填充。剔除无效/零面积面片 {len(invalid)} 个。PNG 直接读取导出的面矢量渲染，透明度约45%，叠加原始 RGB 像素。

这些面是左右边界测宽结果的几何连接，不是 SAM-MoLRA 语义道路面。阴影、建筑或错误中心线仍可导致错误面片；颜色仅表示宽度，未进行真值精度验证。
''',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
