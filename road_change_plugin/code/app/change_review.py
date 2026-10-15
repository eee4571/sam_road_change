"""Manual change review: GIS I/O in the plugin runtime, never in the Qt host.

Run as ``python -m app.change_review``. Requests/replies are one JSON object per
line. Publication uses immutable versions and an atomic current.json pointer.
"""
from __future__ import annotations

import base64
from collections import Counter
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
from PIL import Image
import pyogrio
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping
from shapely.ops import transform

from input_catalog import read_path_list
from .review_history import source_signature, feature_keys, match_decisions

PREFIX = "__ROAD_REVIEW__"
STATES = {"pending", "keep", "reject"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ReviewSession:
    def __init__(self, manifest, project_root=None, output_root=None):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = read_json(self.manifest_path)
        self.project_root = Path(project_root or self.manifest.get('project_root') or self.manifest_path.parent).resolve()
        self.output_root = Path(self.manifest.get('output_root') or output_root or self.project_root / '成果输出').resolve()
        from .review_publication import recover_publication
        recover_publication(self.project_root)
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("fast_finalization_state") == "pending":
            raise ValueError("变化成果尚未生成完毕，请等待处理完成")
        self.pairs = []
        self.frame = None
        for entry in self.manifest.get("change_results", []):
            products = entry.get("published") or {**entry, **entry.get("layers", {})}
            value = products.get("changes")
            if not value or Path(value).suffix.lower() != ".shp":
                continue
            path = self.resolve(value)
            if path.is_file():
                self.pairs.append({**{k: entry.get(k, "") for k in ("grid", "before_period", "after_period")},
                                   "path": str(path)})

    def resolve(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.manifest_path.parent / path).resolve()

    def review_states(self):
        """Lightweight catalogue; source validation remains mandatory on open/apply."""
        states = []
        for pair in self.pairs:
            source = Path(pair['path'])
            folder = source.parent / '人工复核' / source.stem
            applied = {}
            if (folder / 'current.json').is_file():
                version = str(read_json(folder / 'current.json')['version'])
                if Path(version).name != version or version in {'.', '..'}:
                    raise ValueError('复核记录版本路径无效')
                applied = read_json(folder / version / 'review.json')
            draft = read_json(folder / 'draft.json') if (folder / 'draft.json').is_file() else applied
            decisions = draft.get('decisions', {})
            entry = next((e for e in self.manifest.get('change_results', [])
                          if all(e.get(k) == pair.get(k) for k in ('grid', 'before_period', 'after_period'))), {})
            baseline = (applied.get('decisions', {}) if entry.get('manual_review') and
                        draft.get('source_signature') == applied.get('source_signature') else {})
            baseline = {key: baseline.get(key, 'pending') for key in decisions}
            states.append({'known': bool(draft), 'modified': decisions != baseline,
                           'counts': {s: sum(v == s for v in decisions.values()) for s in STATES}})
        return states

    def open_pair(self, index):
        self.frame = None
        pair = self.pairs[index]
        self.manifest = read_json(self.manifest_path)
        from .review_publication import pair_key
        entry = next(e for e in self.manifest['change_results'] if pair_key(e) == pair_key(pair))
        source = self.resolve((entry.get('published') or {}).get('changes') or pair['path'])
        pair['path'] = str(source)
        inputs = self.manifest.get("input_spec", {}).get("grids", {}).get(pair["grid"], {})
        inputs = {p: inputs.get(p) for p in (pair["before_period"], pair["after_period"])}
        signature = source_signature(source, inputs)
        self.published_source, self.published_signature = source, signature
        review = entry.get('manual_review') or {}
        if entry.get('review_candidates'):
            source=self.resolve(entry['review_candidates'])
            signature=source_signature(source,inputs)
        elif review.get('original'):
            original = Path(review['original'])
            approved = original.with_name('road_changes_reviewed.shp')
            same_context = review.get('published_signature') == signature
            # Copying a project changes image-list paths in the signature. The
            # immutable published snapshot proves this is still the same result;
            # open the full original, including rejected roads, not just the kept subset.
            same_data = False
            if not same_context and approved.is_file():
                # Publishing drops non-product columns; compare the semantic
                # contents, not DBF bytes or a regenerated feature numbering.
                same_data = (Counter(feature_keys(gpd.read_file(source, engine='pyogrio')).values()) ==
                             Counter(feature_keys(gpd.read_file(approved, engine='pyogrio')).values()))
            if same_context or same_data:
                source = original
                signature = source_signature(source, inputs)
                if same_context and signature != review['source_signature']:
                    raise ValueError('原始变化备份已改变，无法继续复核')
        frame = gpd.read_file(source, engine="pyogrio")
        if frame.crs is None:
            raise ValueError("变化成果缺少坐标系，无法与影像叠加")
        if source_signature(source, inputs) != signature:
            raise ValueError("变化成果正在被修改，请稍后重新打开")
        self.pair, self.source, self.inputs, self.signature = pair, source, inputs, signature
        self.folder = self.published_source.parent / "人工复核" / self.published_source.stem
        statuses, warning, saved_path = {}, "", ""
        record, original, reconciled = {}, None, False
        pointer = self.folder / "current.json"
        if pointer.is_file():
            current = read_json(pointer)
            version = str(current["version"])
            if Path(version).name != version or version in {".", ".."}:
                raise ValueError("复核记录版本路径无效")
            record = read_json(self.folder / version / "review.json")
            original = self.folder / version / "original_changes.shp"
            if record["source_signature"] == signature:
                statuses = record["decisions"]
                saved_path = str(self.folder / version / "road_changes_reviewed.shp")
                if review.get('published_signature') == self.published_signature:
                    saved_path = str(self.published_source)
            else:
                statuses = match_decisions(frame, record, original)
                reconciled = True
        valid_ids = {str(i) for i in range(len(frame))}
        if not isinstance(statuses, dict) or not set(statuses) <= valid_ids or any(s not in STATES for s in statuses.values()):
            raise ValueError("复核记录中的条目或状态无效")
        is_applied = not reconciled and review.get('published_signature') == self.published_signature
        applied_decisions = {key: statuses.get(key, 'pending') if is_applied else 'pending' for key in valid_ids}
        draft_path = self.folder / 'draft.json'
        if draft_path.is_file():
            draft = read_json(draft_path)
            if draft.get('source_signature') == signature:
                statuses = draft['decisions']
                if set(statuses) != valid_ids or any(s not in STATES for s in statuses.values()):
                    raise ValueError('自动保存的复核记录无效')
                # This draft already belongs to the current candidates. Reopening
                # it must not repeat migration or block the batch apply operation.
                reconciled = False
            else:
                draft_original = original if draft.get('source_signature') == record.get('source_signature') else None
                snapshot = self.folder / 'draft_source.json'
                if snapshot.is_file():
                    name = str(read_json(snapshot)['version'])
                    if Path(name).name != name or name in {'.', '..'}:
                        raise ValueError('复核记录版本路径无效')
                    evidence = self.folder / name
                    if read_json(evidence / 'review.json').get('source_signature') == draft.get('source_signature'):
                        draft_original = evidence / 'original_changes.shp'
                statuses = match_decisions(frame, draft, draft_original)
                reconciled = True
        if reconciled:
            count = sum(value in {'keep', 'reject'} for value in statuses.values())
            warning = f'已恢复 {count} 条未变化对象的复核判断，另有 {len(frame)-count} 条待复核；旧版本仍保留。请完成复核并更新成果。'

        # The view uses metres; stored/exported geometry keeps the original CRS.
        valid = frame.geometry.notna() & ~frame.geometry.is_empty
        view_crs = frame.loc[valid].estimate_utm_crs() if valid.any() else frame.crs
        self.view = frame.to_crs(view_crs)
        self.geometry_type = pyogrio.read_info(source)["geometry_type"]
        self.rasters = []
        self.image_warnings = []
        for period in (pair["before_period"], pair["after_period"]):
            tiles = []
            try:
                fingerprint = inputs.get(period)
                value = fingerprint.get("path") if isinstance(fingerprint, dict) else fingerprint
                if not value:
                    raise ValueError("处理记录中没有该期影像路径")
                path = self.resolve(value)
                paths = ([e.path for e in read_path_list(path).entries] if path.suffix.lower() == ".txt"
                         else sorted(p for p in path.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".img", ".jp2", ".vrt"})
                         if path.is_dir() else [path])
                if not paths:
                    raise ValueError("没有可用影像")
                for path in paths:
                    with rasterio.open(path) as dataset:
                        if dataset.crs is None:
                            raise ValueError(f"影像没有坐标系：{path}")
                        bounds = transform_bounds(dataset.crs, view_crs, *dataset.bounds)
                        tiles.append((str(path), bounds))
            except Exception as exc:
                self.image_warnings.append(f"{period}：{exc}")
            self.rasters.append(tiles)
        self.frame = frame
        self.feature_keys = feature_keys(frame)
        if reconciled:
            # Persist the new ID mapping immediately, but do not publish drafts.
            if draft_path.is_file():
                atomic_json(self.folder / ('draft_' + uuid.uuid4().hex[:12] + '.json'), read_json(draft_path))
            self.record({str(i): statuses.get(str(i), 'pending') for i in range(len(frame))})
        ordered=frame.sort_values('confidence',ascending=False,kind='stable') if 'confidence' in frame else frame
        return {"pair": pair, "features": [
            {"id": str(i), "change_type": str(row.get("change_typ", "未分类")),
             "status": statuses.get(str(i), "pending"),
             "attributes": {str(k): str(v) for k, v in row.items() if k != frame.geometry.name}}
            for i, row in ordered.iterrows()], "warning": warning,
            "image_warnings": self.image_warnings, "saved_path": saved_path,
            "source_signature": signature, "published_signature": self.published_signature,
            "applied_decisions": applied_decisions}

    def record(self, decisions):
        """Persist judgements only. Never publish products or run evaluation here."""
        if self.frame is None:
            raise ValueError('请先选择变化对')
        ids = {str(i) for i in range(len(self.frame))}
        if not isinstance(decisions, dict) or set(decisions) != ids or any(s not in STATES for s in decisions.values()):
            raise ValueError('复核条目或状态无效，未保存')
        self.folder.mkdir(parents=True, exist_ok=True)
        atomic_json(self.folder / 'draft.json', {
            'source_signature': self.signature, 'decisions': decisions,
            'feature_keys': self.feature_keys,
            'recorded_at': datetime.now(timezone.utc).isoformat()})
        return {'recorded': True}

    def preview(self, feature_id, context=1.5, size=1024):
        if self.frame is None:
            raise ValueError("请先选择变化对")
        index = int(feature_id)
        if not 0 <= index < len(self.frame):
            raise ValueError("变化条目不存在")
        geom = self.view.geometry.iloc[index]
        if geom is None or geom.is_empty:
            raise ValueError("该条目没有可显示的几何")
        left, bottom, right, top = geom.bounds
        span = max(right - left, top - bottom, 40.0) * max(1.2, min(float(context), 12.0))
        cx, cy = (left + right) / 2, (bottom + top) / 2
        bounds = (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2)
        affine = from_bounds(*bounds, size, size)
        arrays, masks, byte_images = [], [], []
        warnings = list(self.image_warnings)
        for period, tiles in zip((self.pair["before_period"], self.pair["after_period"]), self.rasters):
            canvas = np.zeros((3, size, size), dtype="float32")
            occupied = np.zeros((size, size), dtype=bool)
            is_byte = True
            for path, tile_bounds in tiles:
                if not box(*bounds).intersects(box(*tile_bounds)):
                    continue
                with rasterio.open(path) as dataset:
                    rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                    indexes = ([dataset.colorinterp.index(c) + 1 for c in rgb] if all(c in dataset.colorinterp for c in rgb)
                               else [1, 2, 3] if dataset.count >= 3 else [1, 1, 1])
                    is_byte = is_byte and all(dataset.dtypes[i - 1] == "uint8" for i in indexes)
                    with WarpedVRT(dataset, crs=self.view.crs, transform=affine, width=size, height=size,
                                   resampling=Resampling.bilinear,
                                   add_alpha=ColorInterp.alpha not in dataset.colorinterp) as vrt:
                        values = vrt.read(indexes, out_dtype="float32")
                        coverage = vrt.dataset_mask() > 0
                        coverage &= np.all(np.isfinite(values), axis=0)
                        take = coverage & ~occupied
                        canvas[:, take] = values[:, take]
                        occupied |= coverage
            if not occupied.any():
                warnings.append(f"{period}：当前范围没有有效影像覆盖")
            arrays.append(canvas)
            masks.append(occupied)
            byte_images.append(is_byte)
        # Use one stretch for both dates so independent contrast does not invent change.
        limits = []
        for band in range(3):
            samples = [a[band, m][::8] for a, m in zip(arrays, masks) if m.any()]
            limits.append((0, 255) if all(byte_images) or not samples else np.percentile(np.concatenate(samples), [2, 98]))
        images = []
        for values, mask in zip(arrays, masks):
            rgb = np.empty_like(values, dtype="uint8")
            for band, (low, high) in enumerate(limits):
                rgb[band] = np.clip((values[band] - low) * 255 / max(float(high - low), 1), 0, 255).astype("uint8")
            rgba = np.dstack((rgb.transpose(1, 2, 0), mask.astype("uint8") * 255))
            stream = io.BytesIO()
            Image.fromarray(rgba).save(stream, format="PNG")
            images.append(base64.b64encode(stream.getvalue()).decode("ascii"))
        inverse = ~affine
        pixel_geom = transform(lambda x, y, z=None: inverse * (x, y), geom)
        return {"id": str(index), "images": images, "geometry": mapping(pixel_geom),
                "span_m": span, "size": size, "warnings": warnings,
                "has_images": all(mask.any() for mask in masks)}

    def assert_current(self):
        if (source_signature(self.source, self.inputs) != self.signature or
                source_signature(self.published_source, self.inputs) != self.published_signature):
            raise ValueError('变化数据已改变，请关闭并重新打开复核；本次未覆盖任何成果')

    def save(self, decisions, progress=None):
        if self.frame is None:
            raise ValueError("请先选择变化对")
        ids = {str(i) for i in range(len(self.frame))}
        if not isinstance(decisions, dict) or set(decisions) != ids or any(s not in STATES for s in decisions.values()):
            raise ValueError("复核条目或状态无效，未保存")
        self.assert_current()
        self.record(decisions)
        self.folder.mkdir(parents=True, exist_ok=True)
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        target = self.folder / version
        target.mkdir()
        rejected = np.array([decisions[str(i)] == "reject" for i in range(len(self.frame))], dtype=bool)
        kept=np.array([decisions[str(i)] == 'keep' for i in range(len(self.frame))],dtype=bool)
        for filename, selected in (("road_changes_reviewed.shp", kept), ("false_changes.shp", rejected)):
            pyogrio.write_dataframe(self.frame.loc[selected], target / filename, driver="ESRI Shapefile",
                                    encoding="UTF-8", geometry_type=self.geometry_type)
        counts = {state: sum(s == state for s in decisions.values()) for state in sorted(STATES)}
        record = {"version": 1, "source": str(self.source), "source_signature": self.signature,
                  "pair": self.pair, "decisions": decisions, "counts": counts,
                  "feature_keys": self.feature_keys,
                  "saved_at": datetime.now(timezone.utc).isoformat(),
                  "retained_pending": False, "evaluation": "not_recomputed"}
        atomic_json(target / "review.json", record)
        from .review_publication import publish_review
        result = publish_review(self, target, record, progress or (lambda message: None))
        # Only the published source was overwritten by Apply. A separate
        # candidate source must remain live for stale-data checks on next Apply.
        if self.source == self.published_source:
            self.source = Path(result['original'])
        self.published_source = Path(result['path'])
        self.published_signature = result['published_signature']
        self.pair['path'] = result['path']
        return result


def main():
    session = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "init":
                session = ReviewSession(request["manifest"], request.get('project_root'), request.get('output_root'))
                result = {"pairs": session.pairs, "states": session.review_states()}
            elif session is None:
                raise ValueError("复核会话未初始化")
            elif op == "open":
                result = session.open_pair(int(request["index"]))
            elif op == "preview":
                result = session.preview(request["id"], request.get("context", 1.5))
            elif op == "record":
                result = session.record(request['decisions'])
            elif op == "save":
                result = session.save(request["decisions"], lambda message:
                    print(PREFIX + json.dumps({'op': 'save', 'progress': message}, ensure_ascii=False), flush=True))
            else:
                raise ValueError(f"未知复核操作：{op}")
            reply = {"op": op, "ok": True, "data": result}
        except Exception as exc:
            reply = {"op": request.get("op", ""), "ok": False, "error": str(exc)}
        print(PREFIX + json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
