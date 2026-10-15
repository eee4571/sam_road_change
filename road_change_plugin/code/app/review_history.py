"""Preserve review evidence and match unchanged objects across detection runs."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import uuid


def source_signature(path, inputs):
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode('utf-8'))
    for suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        part = Path(path).with_suffix(suffix)
        digest.update(suffix.encode('ascii'))
        if not part.is_file():
            if suffix in {'.shp', '.shx', '.dbf', '.prj'}:
                raise ValueError(f'变化数据缺少文件：{part}')
            continue
        with part.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def feature_keys(frame):
    """Exact normalized geometry + review-relevant properties; never row IDs.

    No tolerance, clipping, longest-part selection or nearby-feature matching.
    A split, merge, width/type edit or duplicate therefore requires new review.
    """
    import pandas as pd
    from shapely import normalize
    crs = str(frame.crs.to_epsg() or frame.crs.to_wkt()) if frame.crs else ''
    fields = ('change_typ', 'width_bef', 'width_aft', 'width_diff', 'width_m')
    keys = {}
    for index, row in frame.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            keys[str(index)] = None
            continue
        values = [None if pd.isna(row.get(key)) else
                  (str(row[key]) if key == 'change_typ' else float(row[key])) for key in fields]
        payload = json.dumps([crs, values], ensure_ascii=False).encode('utf-8')
        keys[str(index)] = hashlib.sha256(payload + normalize(geometry).wkb).hexdigest()
    return keys


def match_decisions(frame, record, original=None):
    import geopandas as gpd
    old = record.get('feature_keys')
    if old is None and original and Path(original).is_file():
        old = feature_keys(gpd.read_file(original, engine='pyogrio'))
    old = old or {}
    new = feature_keys(frame)
    old_counts, new_counts = Counter(old.values()), Counter(new.values())
    decisions = record.get('decisions', {})
    unique = {key: decisions.get(i, 'pending') for i, key in old.items()
              if key and old_counts[key] == 1}
    return {i: unique.get(key, 'pending') if key and new_counts[key] == 1 else 'pending'
            for i, key in new.items()}


def snapshot_review_source(manifest, entry):
    """Snapshot legacy drafts before overwriting candidates (no GIS imports)."""
    from .result_publisher import safe_name
    from .project_relocation import _io_path
    output = manifest.get('output_root')
    if not output:
        if not manifest.get('project_root'):
            return
        output = Path(manifest['project_root']) / '成果输出'
    output = Path(output)
    pair = f"{safe_name(entry['before_period'])}_to_{safe_name(entry['after_period'])}"
    folder = output / safe_name(entry['grid']) / '02_变化检测' / pair / '人工复核' / 'road_changes'
    draft_path = folder / 'draft.json'
    if not draft_path.is_file():
        return
    draft = json.loads(draft_path.read_text(encoding='utf-8-sig'))
    if draft.get('feature_keys') is not None:
        return
    source = (entry.get('review_candidates') or (entry.get('published') or {}).get('changes')
              or folder.parent.parent / 'road_changes.shp')
    if not source or not Path(source).is_file():
        return
    inputs = manifest.get('input_spec', {}).get('grids', {}).get(entry['grid'], {})
    inputs = {p: inputs.get(p) for p in (entry['before_period'], entry['after_period'])}
    if source_signature(Path(source), inputs) != draft.get('source_signature'):
        return  # Never attach changed candidates to an older set of row numbers.
    target = folder / ('source_' + uuid.uuid4().hex[:12])
    target.mkdir()
    for suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        part = Path(source).with_suffix(suffix)
        if part.is_file():
            shutil.copy2(_io_path(part), _io_path(target / ('original_changes' + suffix)))
    from .project_relocation import _atomic_write_json
    _atomic_write_json(target / 'review.json', draft)
    _atomic_write_json(folder / 'draft_source.json', {'version': target.name})
