"""Rectified original RGB strips and curated case panels; does not alter measurements."""
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boundary_width import ImageReader
from visualize_results import plot_road


def strip(ax, folder, road_id, overlay=True, start=None, end=None):
    frame = pd.read_csv(folder/'roads'/f'{road_id}.csv').fillna({'flags': ''})
    with np.load(folder/'roads'/f'{road_id}_profiles.npz') as z:
        rgb = z['rgb'].copy()
        offsets = z['offsets'].copy()
    mask = np.ones(len(frame), bool)
    if start is not None:
        mask &= frame.s_m >= start
    if end is not None:
        mask &= frame.s_m <= end
    rgb = np.clip(np.nan_to_num(rgb[mask]), 0, 1)
    f = frame.loc[mask]
    half_step = (offsets[1]-offsets[0])/2
    ax.imshow(rgb.transpose(1, 0, 2), extent=(f.s_m.iloc[0]-1.5, f.s_m.iloc[-1]+1.5,
              offsets[-1]+half_step, offsets[0]-half_step), aspect='auto')
    ax.axhline(0, color='cyan', lw=.8)
    if overlay:
        accepted = f.accepted.to_numpy(bool)
        for column, sign, color in [('left_distance', 1, '#50ff72'), ('right_distance', -1, '#ff9b32')]:
            y = f['optimized_'+column].to_numpy()*sign
            ax.scatter(f.s_m[~accepted], y[~accepted], color=color, s=3, alpha=.6)
            y[~accepted] = np.nan
            ax.plot(f.s_m, y, color=color, lw=1)
    ax.set_ylim(-20, 20)
    ax.set_ylabel('Offset (m)')
    ax.set_title(road_id+f' | accepted {f.accepted.mean():.0%}', fontsize=9)
    return f


def atlas(folder):
    files = sorted((folder/'roads').glob('road_*.csv'))
    for page in range((len(files)+11)//12):
        fig, axes = plt.subplots(6, 2, figsize=(18, 13), constrained_layout=True)
        for ax, path in zip(axes.flat, files[page*12:(page+1)*12]):
            strip(ax, folder, path.stem)
        fig.suptitle('Original RGB sampled along normals | cyan=centerline; green/orange=optimized sides')
        fig.savefig(folder/f'strip_atlas_{page:02d}.jpg', dpi=120)
        plt.close(fig)


def cases(folder, config):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    reader = ImageReader(manifest['image'], manifest['metric_crs'])
    destination = folder/'cases'
    destination.mkdir(exist_ok=True)
    for case in config:
        frame = pd.read_csv(folder/'roads'/f'{case["road_id"]}.csv').fillna({'flags': ''})
        frame = frame[(frame.s_m >= case['start_m']) & (frame.s_m <= case['end_m'])]
        plot_road(reader, frame, destination/f'{case["name"]}_overlay.png')
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
        strip(axes[0], folder, case['road_id'], False, case['start_m'], case['end_m'])
        strip(axes[1], folder, case['road_id'], True, case['start_m'], case['end_m'])
        axes[0].set_title('Original RGB cross-sections | cyan: supplied centerline')
        axes[1].set_title('Optimized boundaries | solid: accepted; dots: low confidence')
        for method, color in [('baseline', '#8d5ab5'), ('optimized', '#0079b8')]:
            width = frame[method+'_width'].to_numpy()
            axes[2].plot(frame.s_m, width, color=color, alpha=.25, lw=.8)
            width[~frame.accepted.to_numpy(bool)] = np.nan
            axes[2].plot(frame.s_m, width, color=color, label=method)
        axes[2].set(xlabel='Distance along centerline (m)', ylabel='Width (m)', ylim=(0, 40))
        axes[2].legend(); axes[2].grid(alpha=.2)
        fig.suptitle(case['name']+' | '+case['road_id']+' | '+case['label'])
        fig.savefig(destination/f'{case["name"]}_profile.png', dpi=160)
        plt.close(fig)
    reader.ds.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    p.add_argument('--cases', type=Path)
    args = p.parse_args()
    if args.cases:
        cases(args.folder, json.loads(args.cases.read_text(encoding='utf-8')))
    else:
        atlas(args.folder)
