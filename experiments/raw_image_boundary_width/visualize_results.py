from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})


def plot_road(reader, frame, path):
    center = frame[['center_x', 'center_y']].to_numpy()
    normal = frame[['normal_x', 'normal_y']].to_numpy()
    extent_points = np.concatenate([center+normal*23, center-normal*23])
    rgb, valid, _, origin = reader.crop(extent_points, pad=10)
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[2.1, 1])
    axes = [fig.add_subplot(grid[0, k]) for k in range(3)]
    center_pixel = reader.pixels(center)-origin
    for ax in axes:
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title('Original RGB | cyan: smoothed SAMRoad line')
    axes[0].plot(*center_pixel.T, color='cyan', lw=.9)
    for ax, method in zip(axes[1:], ('baseline', 'optimized')):
        left = reader.pixels(frame[[method+'_left_x', method+'_left_y']].to_numpy())-origin
        right = reader.pixels(frame[[method+'_right_x', method+'_right_y']].to_numpy())-origin
        accepted = frame.accepted.to_numpy(bool)
        segments = np.stack([left, right], axis=1)
        ids = np.arange(len(frame)) % 3 == 0
        ax.add_collection(LineCollection(segments[ids], colors=np.where(accepted[ids], '#ffff6699', '#ff3f5f99'), lw=.7))
        ax.plot(*center_pixel.T, color='cyan', lw=.7)
        for points, color in [(left, '#50ff72'), (right, '#ff9b32')]:
            # Display low-confidence estimates as small dots; solid boundaries only on accepted runs.
            ax.scatter(*points[~accepted].T, s=2, color=color, alpha=.5)
            good_points = points.copy(); good_points[~accepted] = np.nan
            ax.plot(*good_points.T, color=color, lw=1.1)
        ax.set_title(method.title()+' | green: left, orange: right')
    ax = fig.add_subplot(grid[1, :2])
    accepted = frame.accepted.to_numpy(bool)
    s = frame.s_m.to_numpy()
    for method, color in [('baseline', '#8d5ab5'), ('optimized', '#0079b8')]:
        w = frame[method+'_width'].to_numpy()
        ax.plot(s, w, color=color, alpha=.2, lw=.7)
        ax.scatter(s[~accepted], w[~accepted], s=7, color=color, alpha=.4)
        good_width = w.copy(); good_width[~accepted] = np.nan
        ax.plot(s, good_width, color=color, lw=1.4, label=method)
        ax.scatter(s[accepted], w[accepted], s=3, color=color)
    for i in np.flatnonzero(~accepted):
        ax.axvspan(s[i]-1.5, s[i]+1.5, color='#ef7788', alpha=.09, lw=0)
    ax.set(xlabel='Distance along smoothed centerline (m)', ylabel='Width (m)', ylim=(0, 40))
    ax.grid(alpha=.2); ax.legend(loc='upper right')
    cx = fig.add_subplot(grid[1, 2])
    cx.plot(s, frame.optimized_left_confidence, color='#249a48', label='left')
    cx.plot(s, frame.optimized_right_confidence, color='#de8320', label='right')
    cx.axhline(.16, color='#888888', ls='--', lw=.8)
    cx.set(xlabel='Distance (m)', ylabel='Evidence confidence (uncalibrated)', ylim=(0, 1))
    cx.legend(); cx.grid(alpha=.2)
    title = f'{frame.road_id.iloc[0]} | s={s[0]:.0f}-{s[-1]:.0f} m | accepted {accepted.mean():.1%}'
    fig.suptitle(title+'\nRed sections / pink bands: excluded from continuous optimization; faint width = diagnostic only')
    fig.savefig(path, dpi=130)
    plt.close(fig)


def contact_sheet(output):
    files = sorted((output/'roads').glob('road_*.png'))
    for page in range((len(files)+19)//20):
        selected = files[page*20:(page+1)*20]
        sheet = Image.new('RGB', (1600, 5*270), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(selected):
            with Image.open(path) as im:
                # Original and optimized image panels, plus profiles remain in full road image.
                thumb = ImageOps.contain(im, (400, 250))
                x, y = (i%4)*400, (i//4)*270
                sheet.paste(thumb, (x, y+18))
                draw.text((x+8, y+2), path.stem, fill='black')
        sheet.save(output/f'contact_{page:02d}.jpg', quality=90)


def plot_summary(frame, path):
    pairs = frame.accepted & frame.accepted.shift(fill_value=False) & frame.road_id.eq(frame.road_id.shift())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for method, color in [('baseline', '#8d5ab5'), ('optimized', '#0079b8')]:
        steps = frame[method+'_width'].diff().abs()[pairs].dropna().sort_values().to_numpy()
        axes[0].plot(steps, np.arange(1, len(steps)+1)/max(1, len(steps)), label=method, color=color)
        axes[1].hist(frame.loc[frame.accepted, method+'_width'], bins=np.arange(0, 38, 1),
                     histtype='step', label=method, color=color)
    axes[0].set(xlabel='Adjacent width change (m)', ylabel='CDF', xlim=(0, 15)); axes[0].legend()
    axes[1].set(xlabel='Accepted width (m)', ylabel='Samples'); axes[1].legend()
    reasons = frame['flags'].replace('', 'accepted').value_counts().head(7)
    axes[2].barh(np.arange(len(reasons)), reasons.to_numpy(), color='#537e93')
    axes[2].set_yticks(np.arange(len(reasons)), reasons.index, fontsize=7)
    axes[2].set_xlabel('Samples'); axes[2].invert_yaxis()
    fig.suptitle('Continuity comparison on identical accepted samples (not an accuracy evaluation)')
    fig.savefig(path, dpi=160); plt.close(fig)
