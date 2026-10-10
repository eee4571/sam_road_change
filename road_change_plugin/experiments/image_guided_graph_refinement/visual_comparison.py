"""RGB / previous result / current result at identical extent and line weight."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from bootstrap import ROOT
from data import discover, read_json, fingerprint
from evidence import ImageGrid
from report import background, draw, write_json
from run_multi_roi import data_fingerprints
from repair_artifacts import compare


def make_visual_comparison(previous, current):
    previous, current = Path(previous).resolve(), Path(current).resolve()
    if not current.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Visual products must stay in experimental outputs')
    old, new = read_json(previous/'inputs.json'), read_json(current/'inputs.json')
    inputs = discover(new['project'], new['period'], new['area'], record=new['selected_period_record'])
    manifest_path = inputs['assets']['manifest']
    if (old['crs'] != new['crs'] or data_fingerprints(old['fingerprints'], manifest_path) !=
            data_fingerprints(new['fingerprints'], manifest_path) or
            data_fingerprints(fingerprint(inputs), manifest_path) != data_fingerprints(new['fingerprints'], manifest_path)):
        raise ValueError('Different actual inputs; refusing misleading visual comparison')
    original_regions = {r['id']: r for r in read_json(previous/'selection.json')['regions']}
    notes = read_json(current/'visual_review.json') if (current/'visual_review.json').exists() else {}
    index = []
    for row in read_json(current/'selection.json')['regions']:
        key = row['id']
        if key not in original_regions or row['bbox_m'] != original_regions[key]['bbox_m']:
            raise ValueError('ROI extent changed between rounds')
        folder = current/key
        if not (folder/'final_centerlines.gpkg').exists():
            continue
        roi = box(*row['bbox_m'])
        before = gpd.read_file(previous/key/'final_centerlines.gpkg', layer='context')
        after = gpd.read_file(folder/'final_centerlines.gpkg', layer='context')
        evidence = SimpleNamespace(image=ImageGrid(inputs['images'], new['crs'], roi, .5))
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, frame, title in zip(axes, (None, before, after), ('Raw RGB', 'Previous experiment', 'Current experiment')):
            background(ax, evidence)
            if frame is not None:
                for g in frame[frame.intersects(roi)].geometry:
                    draw(ax, g, color='#ffff33', linewidth=.85)
            ax.set(xlim=(roi.bounds[0], roi.bounds[2]), ylim=(roi.bounds[1], roi.bounds[3]), aspect='equal', title=title)
            ax.ticklabel_format(style='plain', useOffset=False)
        fig.suptitle(key+' / Compare actual pavement and connections, not operation counts')
        fig.tight_layout(); fig.savefig(folder/'rgb_previous_current.png', dpi=180); plt.close(fig)
        cases = []
        for k, case in enumerate(notes.get(key, {}).get('cases', [])):
            bbox = box(*case['bbox_m']).intersection(roi)
            path = folder/f'visual_case_{k+1:02d}.png'
            compare(path, before, after, evidence, bbox, case['title']+' / previous -> current')
            cases.append(dict(case, file=str(path.relative_to(current))))
        index.append(dict(roi_id=key, file=f'{key}/rgb_previous_current.png', cases=cases))
    write_json(current/'visual_comparison.json', dict(previous=str(previous), current=str(current),
        inputs_identical=True, line_weight_identical=True, rgb_resolution_m=.5,
        interpretation='Assistant visual inspection; not human-labelled ground truth or accuracy', regions=index,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('current', type=Path)
    args = parser.parse_args()
    make_visual_comparison(args.previous, args.current)
