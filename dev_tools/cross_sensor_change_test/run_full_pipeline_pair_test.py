from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize


ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "user_pipeline.py"
DEFAULT_OUTPUT = TOOL_ROOT / "outputs" / "full_pipeline_pair_test"
DEFAULT_CHECKPOINT = ROOT / "models" / "samroad" / "samroad.ckpt"
DEFAULT_CONFIG = ROOT / "config" / "samroad_inference.yaml"
SEARCH_ROOTS = (TOOL_ROOT, ROOT / "docs" / "experiment_results")
CHANGE_TYPES = ("added", "removed", "widened", "narrowed")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def discover_pair(before_image: str = "", after_image: str = "") -> tuple[Path, Path, list[str]]:
    searched = [str(path.resolve()) for path in SEARCH_ROOTS]
    if before_image or after_image:
        if not before_image or not after_image:
            raise ValueError("--before-image and --after-image must be supplied together")
        return Path(before_image).expanduser().resolve(), Path(after_image).expanduser().resolve(), searched
    candidates = []
    for search_root in SEARCH_ROOTS:
        if not search_root.is_dir():
            continue
        for before in search_root.rglob("A_original.tif"):
            after = before.with_name("B_degraded.tif")
            if not after.is_file():
                continue
            siblings = {path.name.casefold() for path in before.parent.iterdir() if path.is_file()}
            has_change_truth = any(name.startswith("truth_") for name in siblings)
            path_text = str(before.parent).casefold()
            no_change_hint = "no_change" in path_text or "no-change" in path_text
            if has_change_truth:
                continue
            candidates.append((int(no_change_hint), max(before.stat().st_mtime, after.stat().st_mtime), before, after))
    if not candidates:
        raise FileNotFoundError(
            "No existing A_original.tif/B_degraded.tif no-change pair was found. "
            f"Searched: {', '.join(searched)}"
        )
    _hint, _mtime, before, after = max(candidates, key=lambda item: (item[0], item[1]))
    return before.resolve(), after.resolve(), searched


def validate_pair(before: Path, after: Path) -> dict:
    for path in (before, after):
        if not path.is_file():
            raise FileNotFoundError(path)
    with rasterio.open(before) as left, rasterio.open(after) as right:
        checks = {
            "shape_equal": (left.height, left.width, left.count) == (right.height, right.width, right.count),
            "crs_equal": left.crs == right.crs,
            "transform_equal": left.transform == right.transform,
            "bounds_equal": left.bounds == right.bounds,
        }
        metadata = {
            "shape": [left.height, left.width, left.count],
            "crs": str(left.crs),
            "transform": list(left.transform)[:6],
            "bounds": list(left.bounds),
            **checks,
        }
    if not all(checks.values()):
        raise ValueError(f"Selected degraded pair is not spatially aligned: {checks}")
    return metadata


def input_difference(before: Path, after: Path) -> dict:
    absolute_sum = squared_sum = brightness_a = brightness_b = 0.0
    value_count = pixel_count = 0
    with rasterio.open(before) as left, rasterio.open(after) as right:
        band_count = min(3, left.count, right.count)
        for _index, window in left.block_windows(1):
            a = left.read(range(1, band_count + 1), window=window).astype(np.float32)
            b = right.read(range(1, band_count + 1), window=window).astype(np.float32)
            delta = a - b
            absolute_sum += float(np.abs(delta).sum())
            squared_sum += float(np.square(delta).sum())
            brightness_a += float(a.mean(axis=0).sum())
            brightness_b += float(b.mean(axis=0).sum())
            value_count += int(delta.size)
            pixel_count += int(delta.shape[1] * delta.shape[2])
    return {
        "mean_absolute_pixel_difference": absolute_sum / max(value_count, 1),
        "rmse": math.sqrt(squared_sum / max(value_count, 1)),
        "brightness_mean_A": brightness_a / max(pixel_count, 1),
        "brightness_mean_B": brightness_b / max(pixel_count, 1),
    }


def run_command(command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return time.perf_counter() - started


def result_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, TypeError):
        return False
    required = ("centerlines", "surfaces", "width_segments", "gpkg", "run_root")
    return all(payload.get(key) and Path(payload[key]).is_file() for key in required[:-1]) and Path(payload["run_root"]).is_dir()


def extract_image(image: Path, workspace: Path, args, log_path: Path) -> tuple[Path, float]:
    workspace.mkdir(parents=True, exist_ok=True)
    elapsed = run_command([
        sys.executable, str(PIPELINE), "prepare", "--source", str(image),
        "--workspace", str(workspace),
    ], log_path)
    elapsed += run_command([
        sys.executable, str(PIPELINE), "extract", "--workspace", str(workspace),
        "--checkpoint", str(args.checkpoint), "--config", str(args.config),
        "--device", args.device, "--run-id", "full_pipeline_pair",
    ], log_path)
    result = workspace / "latest_result.json"
    if not result_complete(result):
        raise RuntimeError(f"Production extraction did not create a complete result: {result}")
    return result, elapsed


def resolve_cached_results(output: Path, args) -> tuple[Path | None, Path | None]:
    direct_before = Path(args.before_result).expanduser().resolve() if args.before_result else None
    direct_after = Path(args.after_result).expanduser().resolve() if args.after_result else None
    if direct_before or direct_after:
        if not direct_before or not direct_after:
            raise ValueError("--before-result and --after-result must be supplied together")
        return direct_before, direct_after
    local_before = output / "before_workspace" / "latest_result.json"
    local_after = output / "after_workspace" / "latest_result.json"
    if result_complete(local_before) and result_complete(local_after):
        return local_before, local_after
    cache_index = output / "inputs" / "cached_results.json"
    if cache_index.is_file():
        payload = read_json(cache_index)
        before = Path(payload.get("before_result", ""))
        after = Path(payload.get("after_result", ""))
        if result_complete(before) and result_complete(after):
            return before.resolve(), after.resolve()
    return None, None


def _column(frame: gpd.GeoDataFrame, *names: str) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    return None


def _value(row, frame: gpd.GeoDataFrame, *names: str, default=None):
    column = _column(frame, *names)
    if column is None:
        return default
    value = row.get(column, default)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return default
    return value


def _read_vector(path: str | Path, target_crs=None) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if target_crs is not None and frame.crs is not None and frame.crs != target_crs:
        frame = frame.to_crs(target_crs)
    return frame


def _rasterize(frame: gpd.GeoDataFrame, shape, transform, value=1, all_touched=True) -> np.ndarray:
    shapes = [(geometry, value) for geometry in frame.geometry if geometry is not None and not geometry.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=all_touched, dtype=np.uint8)


def observed_surface_source(result: dict, image_stem: str) -> tuple[Path | None, str | None]:
    run_root = Path(result["run_root"])
    preferred = run_root / "width_review" / f"{image_stem}_molra_clean_mask.png"
    if preferred.is_file():
        return preferred, "sam_molra_clean_mask"
    candidates = [
        run_root / "surfaces" / "masks" / "grid_tiles" / f"{image_stem}_mask.tif",
        run_root / "surfaces" / "masks" / "grid_tiles" / f"{image_stem}_mask.png",
    ]
    for path in candidates:
        if path.is_file():
            return path, "sam_molra_source_mask"
    return None, None


def read_observed_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def line_support_rows(
    centerlines: gpd.GeoDataFrame,
    observed: np.ndarray | None,
    transform,
    pixel_size: float,
    period: str,
    support_radius_px: int,
) -> list[dict]:
    if observed is None:
        support = None
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_radius_px * 2 + 1,) * 2)
        support = cv2.dilate(observed.astype(np.uint8), kernel) > 0
    rows = []
    for index, feature in centerlines.iterrows():
        geometry = feature.geometry
        length = float(geometry.length) if geometry is not None and not geometry.is_empty else 0.0
        fraction = None
        if support is not None and length > 0:
            count = max(2, int(math.ceil(length / max(pixel_size, 1e-6))) + 1)
            points = [geometry.interpolate(distance) for distance in np.linspace(0.0, length, count)]
            coords = [(point.x, point.y) for point in points]
            rc = [rasterio.transform.rowcol(transform, x, y) for x, y in coords]
            values = [support[row, col] for row, col in rc if 0 <= row < support.shape[0] and 0 <= col < support.shape[1]]
            fraction = float(np.mean(values)) if values else 0.0
        classification = (
            "unavailable" if fraction is None else
            "with_observed_surface" if fraction >= 0.95 else
            "without_observed_surface" if fraction <= 0.05 else
            "partial_surface"
        )
        rows.append({
            "period": period,
            "feature_id": str(_value(feature, centerlines, "global_id", "id", default=index)),
            "line_source": str(_value(feature, centerlines, "line_sourc", "line_source", default="unknown")),
            "qa_state": _value(feature, centerlines, "qa_state"),
            "length": length,
            "observed_surface_support_fraction": fraction,
            "classification": classification,
        })
    return rows


def frame_audit(result_path: Path, image: Path, period: str, support_radius_px: int) -> tuple[dict, dict]:
    result = read_json(result_path)
    with rasterio.open(image) as dataset:
        shape = (dataset.height, dataset.width)
        transform = dataset.transform
        crs = dataset.crs
        pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
        pixel_size = math.sqrt(max(pixel_area, 1e-12))
    centerlines = _read_vector(result["centerlines"], crs)
    product_surfaces = _read_vector(result["surfaces"], crs)
    product_mask = _rasterize(product_surfaces, shape, transform) > 0
    observed_path, observed_kind = observed_surface_source(result, image.stem)
    observed = read_observed_mask(observed_path, shape)
    line_mask = _rasterize(centerlines, shape, transform) > 0
    line_kernel_radius = max(1, support_radius_px)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_kernel_radius * 2 + 1,) * 2)
    line_neighborhood = cv2.dilate(line_mask.astype(np.uint8), line_kernel) > 0
    support_rows = line_support_rows(centerlines, observed, transform, pixel_size, period, support_radius_px)
    class_lengths = Counter()
    source_lengths = Counter()
    for row in support_rows:
        class_lengths[row["classification"]] += row["length"]
        source_lengths[row["line_source"]] += row["length"]
    total_length = float(centerlines.geometry.length.sum())
    observed_area = float(np.count_nonzero(observed) * pixel_area) if observed is not None else None
    observed_supported = float(np.count_nonzero(observed & line_neighborhood) * pixel_area) if observed is not None else None
    observed_without = float(np.count_nonzero(observed & ~line_neighborhood) * pixel_area) if observed is not None else None
    forced_product = float(np.count_nonzero(product_mask & ~(observed if observed is not None else product_mask)) * pixel_area) if observed is not None else None
    surface_skeleton_length = sum(length for source, length in source_lengths.items() if "surface" in source.casefold())
    weak_length = sum(length for source, length in source_lengths.items() if "weak" in source.casefold() or "relative" in source.casefold())
    connector_length = sum(length for source, length in source_lengths.items() if "connector" in source.casefold() or "gap" in source.casefold())
    width_segments = _read_vector(result["width_segments"], crs)
    width_column = _column(width_segments, "width_m", "width_map")
    widths = np.asarray(width_segments[width_column], dtype=float) if width_column else np.asarray([], dtype=float)
    widths = widths[np.isfinite(widths)]
    audit = {
        "final_centerline_length": total_length,
        "observed_surface_available": observed is not None,
        "observed_surface_source": str(observed_path.resolve()) if observed_path else None,
        "observed_surface_source_type": observed_kind,
        "observed_surface_area": observed_area,
        "product_surface_source": str(Path(result["surfaces"]).resolve()),
        "product_surface_area": float(product_surfaces.geometry.area.sum()),
        "line_with_observed_surface_length": float(class_lengths["with_observed_surface"]),
        "line_without_observed_surface_length": float(class_lengths["without_observed_surface"]),
        "line_partial_surface_length": float(class_lengths["partial_surface"]),
        "line_without_observed_surface_ratio": float(class_lengths["without_observed_surface"] / max(total_length, 1e-9)),
        "observed_surface_supported_by_line_area": observed_supported,
        "observed_surface_without_line_area": observed_without,
        "observed_surface_without_line_ratio": float(observed_without / max(observed_area, 1e-9)) if observed_without is not None else None,
        "surface_skeleton_line_length": float(surface_skeleton_length),
        "weak_recovered_line_length": float(weak_length),
        "connector_line_length": float(connector_length),
        "forced_product_surface_area": forced_product,
        "median_width": float(np.median(widths)) if widths.size else None,
        "width_segment_count": int(widths.size),
        "line_source_length": dict(sorted(source_lengths.items())),
        "timing": {
            str(row.get("stage", "unknown")): float(row.get("elapsed_seconds", 0.0) or 0.0)
            for row in result.get("stage_timings", [])
        },
        "total_seconds": float(result.get("elapsed_seconds", 0.0) or 0.0),
    }
    context = {
        "result": result, "centerlines": centerlines, "product_surfaces": product_surfaces,
        "observed": observed, "product_mask": product_mask, "line_mask": line_mask,
        "line_neighborhood": line_neighborhood, "shape": shape, "transform": transform,
        "crs": crs, "support_rows": support_rows, "width_segments": width_segments,
    }
    return audit, context


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(_jsonable(rows))


def formal_matched_width_rows(matches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Exclude formal unmatched placeholders from matched-width statistics."""
    relation = _column(matches, "relation")
    before_id = _column(matches, "before_seg", "before_segment")
    after_id = _column(matches, "after_seg", "after_segment")
    before_width = _column(matches, "before_w", "before_width")
    after_width = _column(matches, "after_w", "after_width")
    valid = np.ones(len(matches), dtype=bool)
    if relation:
        valid &= matches[relation].fillna("").astype(str).str.casefold().ne("unmatched").to_numpy()
    if before_id:
        valid &= matches[before_id].notna().to_numpy()
    if after_id:
        valid &= matches[after_id].notna().to_numpy()
    if before_width:
        valid &= (matches[before_width].fillna(0).astype(float) > 0).to_numpy()
    if after_width:
        valid &= (matches[after_width].fillna(0).astype(float) > 0).to_numpy()
    return matches.loc[valid].copy()


def width_audit(change_dir: Path, before_result: dict, after_result: dict) -> tuple[list[dict], list[dict], dict, gpd.GeoDataFrame]:
    matches = formal_matched_width_rows(_read_vector(change_dir / "road_matches.shp"))
    before = _read_vector(before_result["width_segments"], matches.crs)
    after = _read_vector(after_result["width_segments"], matches.crs)
    before_id = _column(before, "segment_id")
    after_id = _column(after, "segment_id")
    before_lookup = {str(row[before_id]): row for _, row in before.iterrows()} if before_id else {}
    after_lookup = {str(row[after_id]): row for _, row in after.iterrows()} if after_id else {}
    rows = []
    suspicious = []
    for _, match in matches.iterrows():
        before_segment = str(_value(match, matches, "before_seg", "before_segment", default=""))
        after_segment = str(_value(match, matches, "after_seg", "after_segment", default=""))
        before_row = before_lookup.get(before_segment)
        after_row = after_lookup.get(after_segment)
        before_width = float(_value(match, matches, "before_w", "before_width", default=0.0) or 0.0)
        after_width = float(_value(match, matches, "after_w", "after_width", default=0.0) or 0.0)
        absolute = abs(after_width - before_width)
        relative = absolute / max(abs(before_width), 1e-9)
        def side(row, frame, *names):
            return _value(row, frame, *names) if row is not None else None
        record = {
            "match_id": _value(match, matches, "match_id"),
            "canonical_road_id": _value(match, matches, "canonical_", "canonical_road_id"),
            "before_segment_id": before_segment or None,
            "after_segment_id": after_segment or None,
            "before_width": before_width,
            "after_width": after_width,
            "absolute_width_diff": absolute,
            "relative_width_diff": relative,
            "before_width_std": side(before_row, before, "width_std"),
            "after_width_std": side(after_row, after, "width_std"),
            "before_valid_ratio": side(before_row, before, "valid_rati", "valid_ratio"),
            "after_valid_ratio": side(after_row, after, "valid_rati", "valid_ratio"),
            "before_width_quality": side(before_row, before, "width_qual", "width_quality", "quality_gr"),
            "after_width_quality": side(after_row, after, "width_qual", "width_quality", "quality_gr"),
            "before_line_source": side(before_row, before, "line_sourc", "line_source"),
            "after_line_source": side(after_row, after, "line_sourc", "line_source"),
            "before_surface_conf": side(before_row, before, "surface_co", "surface_conf"),
            "after_surface_conf": side(after_row, after, "surface_co", "surface_conf"),
            "before_qa_state": side(before_row, before, "qa_state"),
            "after_qa_state": side(after_row, after, "qa_state"),
            "match_score": _value(match, matches, "match_scor", "match_score"),
            "before_overlap": _value(match, matches, "before_cov", "before_coverage"),
            "after_overlap": _value(match, matches, "after_cov", "after_coverage"),
            "alignment": _value(match, matches, "alignment"),
            "before_conflict": side(before_row, before, "conflict"),
            "after_conflict": side(after_row, after, "conflict"),
            "junction_related": _value(match, matches, "junction_related"),
            "change_candidate": bool(absolute >= 2.0 and relative >= 0.20),
            "change_result": None,
            "change_qa_state": _value(match, matches, "qa_state"),
        }
        rows.append(record)
        if record["change_candidate"]:
            reasons = ["large_width_difference"]
            std_values = [value for value in (record["before_width_std"], record["after_width_std"]) if isinstance(value, (int, float))]
            if any(float(value) >= 2.0 for value in std_values):
                reasons.append("high_width_std")
            valid_values = [value for value in (record["before_valid_ratio"], record["after_valid_ratio"]) if isinstance(value, (int, float))]
            if any(float(value) < 0.6 for value in valid_values):
                reasons.append("low_valid_ratio")
            if "C" in {str(record["before_width_quality"]), str(record["after_width_quality"])}:
                reasons.append("width_quality_C")
            if record["before_line_source"] and record["after_line_source"] and record["before_line_source"] != record["after_line_source"]:
                reasons.append("different_line_source")
            if record["before_surface_conf"] is not None and record["after_surface_conf"] is not None and abs(float(record["before_surface_conf"]) - float(record["after_surface_conf"])) >= 0.25:
                reasons.append("surface_support_disagreement")
            suspicious.append({**record, "suspicion_reason": ";".join(reasons)})
    absolute_values = np.asarray([row["absolute_width_diff"] for row in rows], dtype=float)
    relative_values = np.asarray([row["relative_width_diff"] for row in rows], dtype=float)
    metrics = {
        "matched_width_segment_count": len(rows),
        "stable_width_segment_count": len(rows) - len(suspicious),
        "large_width_disagreement_count": len(suspicious),
        "large_disagreement_but_not_change_count": len(suspicious),
        "median_abs_width_diff": float(np.median(absolute_values)) if absolute_values.size else None,
        "p90_abs_width_diff": float(np.percentile(absolute_values, 90)) if absolute_values.size else None,
        "median_relative_width_diff": float(np.median(relative_values)) if relative_values.size else None,
        "p90_relative_width_diff": float(np.percentile(relative_values, 90)) if relative_values.size else None,
        "suspicion_reason_counts": dict(sorted(Counter(
            reason
            for row in suspicious
            for reason in str(row["suspicion_reason"]).split(";")
            if reason
        ).items())),
    }
    return rows, suspicious, metrics, matches


def false_change_audit(change_dir: Path, summary: dict) -> tuple[dict, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    auto = _read_vector(change_dir / "road_changes.shp")
    review = _read_vector(change_dir / "review_changes.shp")
    auto_type = _column(auto, "change_typ", "change_type")
    review_type = _column(review, "change_typ", "change_type")
    payload = {}
    total_area = 0.0
    for change_type in CHANGE_TYPES:
        auto_rows = auto[auto[auto_type].astype(str).str.casefold() == change_type] if auto_type else auto.iloc[0:0]
        review_rows = review[review[review_type].astype(str).str.casefold() == change_type] if review_type else review.iloc[0:0]
        area_column = _column(auto_rows, "area_m2")
        area = float(auto_rows[area_column].fillna(0).sum()) if area_column else float(auto_rows.geometry.area.sum())
        payload[f"auto_{change_type}_count"] = int(len(auto_rows))
        payload[f"review_{change_type}_count"] = int(len(review_rows))
        payload[f"{change_type}_area"] = area
        total_area += area
    payload["suppressed_extraction_disagreement_count"] = int(summary.get("suppressed_extraction_disagreement_count", 0) or 0)
    payload["false_change_total_area"] = total_area
    payload["official_width_change_count"] = payload["auto_widened_count"] + payload["auto_narrowed_count"]
    return payload, auto, review


def _read_rgb(path: Path, max_size: int | None = None) -> np.ndarray:
    with rasterio.open(path) as dataset:
        scale = 1.0 if not max_size else min(1.0, max_size / max(dataset.width, dataset.height))
        height, width = max(1, int(dataset.height * scale)), max(1, int(dataset.width * scale))
        data = dataset.read(
            range(1, min(3, dataset.count) + 1), out_shape=(min(3, dataset.count), height, width),
            resampling=rasterio.enums.Resampling.bilinear,
        )
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    return np.moveaxis(data[:3], 0, 2)[:, :, ::-1].astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (255, 255, 255), -1)
    cv2.putText(result, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    return result


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def extraction_visual(image_path: Path, context: dict, label: str) -> np.ndarray:
    image = _read_rgb(image_path)
    observed = context["observed"]
    if observed is not None:
        image[observed] = (0.55 * image[observed] + 0.45 * np.asarray([255, 180, 0])).astype(np.uint8)
    product = context["product_mask"].astype(np.uint8)
    boundary = cv2.morphologyEx(product, cv2.MORPH_GRADIENT, np.ones((5, 5), dtype=np.uint8)) > 0
    image[boundary] = (0, 165, 255)
    frame = context["centerlines"]
    source_col = _column(frame, "line_sourc", "line_source")
    for index, source in enumerate(frame[source_col].fillna("unknown").astype(str).unique() if source_col else ["unknown"]):
        subset = frame[frame[source_col].astype(str) == source] if source_col else frame
        mask = _rasterize(subset, context["shape"], context["transform"]) > 0
        folded = source.casefold()
        color = (255, 0, 255) if "surface" in folded else (0, 255, 255) if "weak" in folded or "relative" in folded else (255, 255, 0) if "connector" in folded or "gap" in folded else (255, 255, 255) if "manual" in folded else (0, 255, 0)
        image[mask] = color
    return _label(_resize(image, 1000, 1000), label + " | cyan=observed, orange=product boundary, green=SAMRoad, purple=surface")


def consistency_visual(image_path: Path, context: dict, label: str) -> np.ndarray:
    image = (_read_rgb(image_path).astype(np.float32) * 0.35).astype(np.uint8)
    observed = context["observed"]
    if observed is None:
        return _label(_resize(image, 1000, 1000), label + " | observed surface unavailable")
    line = context["line_mask"]
    line_support = cv2.dilate(line.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)) > 0
    observed_support = cv2.dilate(observed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    product_forced = context["product_mask"] & ~observed
    image[product_forced] = (0, 165, 255)
    image[observed & ~line_support] = (255, 0, 0)
    image[line & observed_support] = (0, 255, 0)
    image[line & ~observed_support] = (0, 0, 255)
    source_col = _column(context["centerlines"], "line_sourc", "line_source")
    if source_col:
        surface_lines = context["centerlines"][context["centerlines"][source_col].fillna("").astype(str).str.contains("surface", case=False)]
        image[_rasterize(surface_lines, context["shape"], context["transform"]) > 0] = (255, 0, 255)
    return _label(_resize(image, 1000, 1000), label + " | green=agree red=line-only blue=surface-only purple=surface line/orange forced product")


def input_pair_visual(before: Path, after: Path, differences: dict) -> np.ndarray:
    left = _label(_resize(_read_rgb(before, 1000), 1000, 1000), "A ORIGINAL")
    right = _label(_resize(_read_rgb(after, 1000), 1000, 1000), "B DEGRADED")
    result = np.concatenate([left, right], axis=1)
    text = f"MAE={differences['mean_absolute_pixel_difference']:.2f}  RMSE={differences['rmse']:.2f}  brightness={differences['brightness_mean_A']:.1f}/{differences['brightness_mean_B']:.1f}"
    cv2.rectangle(result, (0, result.shape[0] - 45), (result.shape[1], result.shape[0]), (0, 0, 0), -1)
    cv2.putText(result, text, (15, result.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def width_visual(image_path: Path, matches: gpd.GeoDataFrame, transform, shape, metrics: dict, label: str) -> np.ndarray:
    image = (_read_rgb(image_path).astype(np.float32) * 0.55).astype(np.uint8)
    diff_col = _column(matches, "width_diff")
    if diff_col:
        absolute = matches[diff_col].fillna(0).abs()
        for subset, color in ((matches[absolute < 1.0], (0, 255, 0)), (matches[(absolute >= 1.0) & (absolute < 2.0)], (0, 255, 255)), (matches[absolute >= 2.0], (0, 0, 255))):
            image[_rasterize(subset, shape, transform) > 0] = color
    text = f"matched={metrics['matched_width_segment_count']}  large={metrics['large_width_disagreement_count']}  abs P50/P90={metrics['median_abs_width_diff']:.2f}/{metrics['p90_abs_width_diff']:.2f}"
    return _label(_resize(image, 1000, 1000), label + " WIDTH | green<1m yellow<2m red>=2m | " + text)


def change_visual(change_dir: Path) -> np.ndarray:
    panels = []
    for name, label in (("change_preview.png", "OFFICIAL CHANGE"), ("sensor_disagreement_preview.png", "REVIEW / SUPPRESSED")):
        path = change_dir / name
        image = cv2.imread(str(path))
        if image is not None:
            panels.append(_label(_resize(image, 1000, 1000), label))
    if not panels:
        return np.zeros((1000, 2000, 3), dtype=np.uint8)
    if len(panels) == 1:
        panels.append(np.zeros_like(panels[0]))
    return np.concatenate(panels[:2], axis=1)


def overview_visual(artifacts: dict, report: dict) -> np.ndarray:
    def load(name):
        image = cv2.imread(str(artifacts[name]))
        return _resize(image, 1600, 800)
    rows = [load("input_pair"), np.concatenate([_resize(cv2.imread(str(artifacts["before_extraction"])), 800, 800), _resize(cv2.imread(str(artifacts["after_extraction"])), 800, 800)], axis=1), load("line_surface_consistency"), load("width_comparison"), load("change_result")]
    metrics = np.zeros((360, 1600, 3), dtype=np.uint8)
    before, after = report["before"], report["after"]
    width = report["cross_period"]["width_difference"]
    false = report["change_detection"]
    lines = [
        f"STATUS {report['status']} | earliest instability: {report['diagnosis']['earliest_instability_stage']}",
        f"Centerline length A/B: {before['centerline']['length']:.1f} / {after['centerline']['length']:.1f}",
        f"Observed surface area A/B: {before['observed_surface']['area']} / {after['observed_surface']['area']}",
        f"Product surface area A/B: {before['product_surface']['area']:.1f} / {after['product_surface']['area']:.1f}",
        f"Line without observed surface A/B: {before['line_surface_consistency']['line_without_observed_surface_length']:.1f} / {after['line_surface_consistency']['line_without_observed_surface_length']:.1f}",
        f"Median width A/B: {before['width']['median']} / {after['width']['median']} | matched width abs P90: {width['p90_abs_width_diff']}",
        f"False auto Added/Removed/Widened/Narrowed: {false['added']['auto_count']} / {false['removed']['auto_count']} / {false['widened']['auto_count']} / {false['narrowed']['auto_count']}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(metrics, line, (30, 45 + index * 43), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate(rows + [metrics], axis=0)


def pct_diff(before, after):
    return None if before in (None, 0) or after is None else float((after - before) / abs(before))


def build_report(
    before_image: Path, after_image: Path, pair_metadata: dict, differences: dict,
    before_audit: dict, after_audit: dict, width_metrics: dict, false: dict,
    change_summary: dict, artifacts: dict, change_seconds: float,
) -> dict:
    auto_total = sum(false[f"auto_{kind}_count"] for kind in CHANGE_TYPES)
    review_total = sum(false[f"review_{kind}_count"] for kind in CHANGE_TYPES)
    status = "FAIL" if auto_total else "WARN" if review_total or width_metrics["large_width_disagreement_count"] else "PASS"
    centerline_diff = pct_diff(before_audit["final_centerline_length"], after_audit["final_centerline_length"])
    observed_diff = pct_diff(before_audit["observed_surface_area"], after_audit["observed_surface_area"])
    product_diff = pct_diff(before_audit["product_surface_area"], after_audit["product_surface_area"])
    source_names = sorted(set(before_audit["line_source_length"]) | set(after_audit["line_source_length"]))
    source_differences = {
        source: pct_diff(
            before_audit["line_source_length"].get(source, 0.0),
            after_audit["line_source_length"].get(source, 0.0),
        )
        for source in source_names
    }
    direct_centerline_changes = {
        source: difference
        for source, difference in source_differences.items()
        if difference is not None
        and "surface" not in source.casefold()
        and "connector" not in source.casefold()
    }
    largest_direct_source = max(
        direct_centerline_changes,
        key=lambda source: abs(direct_centerline_changes[source]),
        default=None,
    )
    if largest_direct_source is not None and abs(direct_centerline_changes[largest_direct_source]) >= 0.05:
        earliest = "centerline_extraction"
        evidence = (
            f"{largest_direct_source} line-source length changed by "
            f"{direct_centerline_changes[largest_direct_source]:.1%}"
        )
    elif centerline_diff is not None and abs(centerline_diff) >= 0.05:
        earliest = "centerline_extraction"
        evidence = f"centerline length changed by {centerline_diff:.1%}"
    elif observed_diff is not None and abs(observed_diff) >= 0.05:
        earliest = "observed_surface_extraction"
        evidence = f"observed surface area changed by {observed_diff:.1%}"
    elif width_metrics["large_width_disagreement_count"]:
        earliest = "width_estimation"
        evidence = f"{width_metrics['large_width_disagreement_count']} matched segments exceed the audit threshold"
    elif auto_total:
        earliest = "change_detection"
        evidence = f"{auto_total} official auto false changes"
    elif review_total:
        earliest = "cross_period_matching_or_extraction_disagreement"
        evidence = f"{review_total} review changes; evidence is insufficient to assign an earlier stage"
    else:
        earliest = "none_detected"
        evidence = "no automatic or diagnostic instability threshold was exceeded"
    def period(audit):
        return {
            "centerline": {"length": audit["final_centerline_length"], "line_source_length": audit["line_source_length"]},
            "observed_surface": {"available": audit["observed_surface_available"], "source": audit["observed_surface_source"], "source_type": audit["observed_surface_source_type"], "area": audit["observed_surface_area"]},
            "product_surface": {"source": audit["product_surface_source"], "area": audit["product_surface_area"], "forced_area": audit["forced_product_surface_area"]},
            "line_surface_consistency": {key: audit[key] for key in (
                "line_with_observed_surface_length", "line_without_observed_surface_length", "line_partial_surface_length",
                "line_without_observed_surface_ratio", "observed_surface_supported_by_line_area", "observed_surface_without_line_area",
                "observed_surface_without_line_ratio", "surface_skeleton_line_length", "weak_recovered_line_length", "connector_line_length",
            )},
            "width": {"median": audit["median_width"], "segment_count": audit["width_segment_count"]},
            "timing": {"total_seconds": audit["total_seconds"], "stages": audit["timing"]},
        }
    config_data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    production_config = {
        "path": str(DEFAULT_CONFIG.resolve()), "pipeline": str(PIPELINE.resolve()),
        "parameters_overridden": False,
        **{
            key: config_data.get(key)
            for key in (
                "RELATIVE_ROADNESS_ENABLED",
                "RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL",
                "RELATIVE_CONTINUOUS_TRACING_EXPERIMENTAL",
                "RELATIVE_JUNCTION_COLLAPSE_EXPERIMENTAL",
                "WEAK_SEGMENT_RECOVERY_ENABLED",
            )
        },
    }
    return {
        "status": status,
        "inputs": {"before": str(before_image), "after": str(after_image), "spatial_metadata": pair_metadata, "degradation_difference": differences, "ground_truth": {kind: 0 for kind in CHANGE_TYPES}},
        "production_config": production_config,
        "before": period(before_audit),
        "after": period(after_audit),
        "cross_period": {
            "centerline_difference": {
                "relative_length_diff": centerline_diff,
                "line_source_relative_length_diff": source_differences,
            },
            "surface_difference": {"observed_relative_area_diff": observed_diff, "product_relative_area_diff": product_diff},
            "width_difference": width_metrics,
        },
        "change_detection": {
            kind: {"auto_count": false[f"auto_{kind}_count"], "review_count": false[f"review_{kind}_count"], "area": false[f"{kind}_area"]}
            for kind in CHANGE_TYPES
        } | {
            "review": {"total_count": review_total},
            "suppressed": {"count": false["suppressed_extraction_disagreement_count"]},
            "false_change_total_area": false["false_change_total_area"],
            "classification_method": change_summary.get("classification_method"),
        },
        "diagnosis": {"earliest_instability_stage": earliest, "evidence": evidence, "false_width_change_reason_counts": {"unknown": false["official_width_change_count"]}},
        "timing": {"before_total_seconds": before_audit["total_seconds"], "after_total_seconds": after_audit["total_seconds"], "change_seconds": change_seconds, "full_pipeline_seconds": before_audit["total_seconds"] + after_audit["total_seconds"] + change_seconds},
        "artifacts": {key: str(Path(value).resolve()) for key, value in artifacts.items()},
    }


def write_markdown(path: Path, report: dict) -> None:
    before, after = report["before"], report["after"]
    width = report["cross_period"]["width_difference"]
    change = report["change_detection"]
    rows = [
        "# Full pipeline no-change degraded-pair report", "",
        f"## {report['status']}", "",
        f"Earliest observed instability: `{report['diagnosis']['earliest_instability_stage']}` — {report['diagnosis']['evidence']}", "",
        "| Metric | Original | Degraded |", "|---|---:|---:|",
        f"| Centerline length | {before['centerline']['length']:.3f} | {after['centerline']['length']:.3f} |",
        f"| Observed surface area | {before['observed_surface']['area']} | {after['observed_surface']['area']} |",
        f"| Product surface area | {before['product_surface']['area']:.3f} | {after['product_surface']['area']:.3f} |",
        f"| Line without observed surface | {before['line_surface_consistency']['line_without_observed_surface_length']:.3f} | {after['line_surface_consistency']['line_without_observed_surface_length']:.3f} |",
        f"| Observed surface without line | {before['line_surface_consistency']['observed_surface_without_line_area']} | {after['line_surface_consistency']['observed_surface_without_line_area']} |",
        f"| Median width | {before['width']['median']} | {after['width']['median']} |", "",
        "| Width metric | Value |", "|---|---:|",
        *[f"| {key} | {value} |" for key, value in width.items()], "",
        "| False change | Auto | Review | Area |", "|---|---:|---:|---:|",
        *[f"| {kind.title()} | {change[kind]['auto_count']} | {change[kind]['review_count']} | {change[kind]['area']} |" for kind in CHANGE_TYPES], "",
        f"Observed surface source A: `{before['observed_surface']['source']}`", "",
        f"Observed surface source B: `{after['observed_surface']['source']}`", "",
        f"Product surface source A: `{before['product_surface']['source']}`", "",
        f"Product surface source B: `{after['product_surface']['source']}`", "",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the production single-tile no-change degraded-pair regression bench.")
    parser.add_argument("--before-image", default="")
    parser.add_argument("--after-image", default="")
    parser.add_argument("--before-result", default="")
    parser.add_argument("--after-result", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--change-only", action="store_true")
    parser.add_argument("--audit-support-radius-px", type=int, default=1)
    args = parser.parse_args(argv)
    if args.fresh and args.change_only:
        parser.error("--fresh and --change-only are mutually exclusive")
    if Path(args.config).expanduser().resolve() != DEFAULT_CONFIG.resolve():
        parser.error(f"This regression bench must use the production config: {DEFAULT_CONFIG}")
    output = Path(args.output_dir).expanduser().resolve()
    before_image, after_image, searched = discover_pair(args.before_image, args.after_image)
    pair_metadata = validate_pair(before_image, after_image)
    print(f"Selected A: {before_image}")
    print(f"Selected B: {after_image}")
    if args.fresh:
        for name in ("before_workspace", "after_workspace", "change", "audit", "visualization"):
            target = output / name
            if target.exists():
                shutil.rmtree(target)
    for name in ("inputs", "audit", "visualization"):
        (output / name).mkdir(parents=True, exist_ok=True)
    write_json(output / "inputs" / "selected_pair.json", {
        "before": str(before_image), "after": str(after_image), "searched_roots": searched,
        "validation": pair_metadata, "semantic": "same roads; B is sensor degradation only",
    })
    differences = input_difference(before_image, after_image)
    before_result, after_result = resolve_cached_results(output, args)
    log_path = output / "full_pipeline_run.log"
    if args.change_only and (not before_result or not after_result):
        parser.error("--change-only requires complete cached before/after latest_result.json files")
    if before_result is None or not result_complete(before_result):
        if args.change_only:
            parser.error("before extraction cache is incomplete")
        before_result, _elapsed = extract_image(before_image, output / "before_workspace", args, log_path)
    else:
        print(f"Reusing before extraction: {before_result}")
    if after_result is None or not result_complete(after_result):
        if args.change_only:
            parser.error("after extraction cache is incomplete")
        after_result, _elapsed = extract_image(after_image, output / "after_workspace", args, log_path)
    else:
        print(f"Reusing after extraction: {after_result}")
    write_json(output / "inputs" / "cached_results.json", {
        "before_result": str(before_result.resolve()), "after_result": str(after_result.resolve()),
    })
    change_dir = output / "change"
    if change_dir.exists():
        shutil.rmtree(change_dir)
    change_seconds = run_command([
        sys.executable, str(PIPELINE), "change", "--before-result", str(before_result),
        "--after-result", str(after_result), "--output", str(change_dir),
        "--before-period", "A", "--after-period", "B",
    ], log_path)
    summary = read_json(change_dir / "change_summary.json")
    before_audit, before_context = frame_audit(before_result, before_image, "A", args.audit_support_radius_px)
    after_audit, after_context = frame_audit(after_result, after_image, "B", args.audit_support_radius_px)
    write_json(output / "audit" / "extraction_comparison.json", {"before": before_audit, "after": after_audit})
    line_rows = before_context["support_rows"] + after_context["support_rows"]
    write_csv(output / "audit" / "line_surface_audit.csv", line_rows)
    before_payload, after_payload = read_json(before_result), read_json(after_result)
    width_rows, suspicious_rows, width_metrics, matches = width_audit(change_dir, before_payload, after_payload)
    write_csv(output / "audit" / "width_comparison.csv", width_rows)
    write_csv(output / "audit" / "suspicious_width_changes.csv", suspicious_rows, list(width_rows[0]) + ["suspicion_reason"] if width_rows else ["suspicion_reason"])
    false, _auto_changes, _review_changes = false_change_audit(change_dir, summary)
    false["official_widened_count"] = false["auto_widened_count"]
    false["official_narrowed_count"] = false["auto_narrowed_count"]
    width_metrics["official_widened_count"] = false["auto_widened_count"]
    width_metrics["official_narrowed_count"] = false["auto_narrowed_count"]
    width_metrics["official_width_change_count"] = false["official_width_change_count"]
    width_metrics["large_disagreement_but_not_change_count"] = max(0, width_metrics["large_width_disagreement_count"] - false["official_width_change_count"])
    write_json(output / "audit" / "false_change_summary.json", false)
    artifacts = {
        "selected_pair": output / "inputs" / "selected_pair.json",
        "extraction_comparison": output / "audit" / "extraction_comparison.json",
        "line_surface_audit": output / "audit" / "line_surface_audit.csv",
        "width_comparison_csv": output / "audit" / "width_comparison.csv",
        "suspicious_width_changes_csv": output / "audit" / "suspicious_width_changes.csv",
        "false_change_summary": output / "audit" / "false_change_summary.json",
    }
    visuals = output / "visualization"
    images = {
        "input_pair": input_pair_visual(before_image, after_image, differences),
        "before_extraction": extraction_visual(before_image, before_context, "A ORIGINAL EXTRACTION"),
        "after_extraction": extraction_visual(after_image, after_context, "B DEGRADED EXTRACTION"),
        "line_surface_consistency": np.concatenate([consistency_visual(before_image, before_context, "A ORIGINAL"), consistency_visual(after_image, after_context, "B DEGRADED")], axis=1),
        "width_comparison": np.concatenate([
            width_visual(before_image, matches, before_context["transform"], before_context["shape"], width_metrics, "A ORIGINAL"),
            width_visual(after_image, matches, after_context["transform"], after_context["shape"], width_metrics, "B DEGRADED"),
        ], axis=1),
        "change_result": change_visual(change_dir),
    }
    file_names = {
        "input_pair": "01_input_pair.png", "before_extraction": "02_before_extraction.png",
        "after_extraction": "03_after_extraction.png", "line_surface_consistency": "04_line_surface_consistency.png",
        "width_comparison": "05_width_comparison.png", "change_result": "06_change_result.png",
    }
    for key, image in images.items():
        path = visuals / file_names[key]
        cv2.imwrite(str(path), image)
        artifacts[key] = path
    report = build_report(before_image, after_image, pair_metadata, differences, before_audit, after_audit, width_metrics, false, summary, artifacts, change_seconds)
    overview_path = visuals / "07_full_pipeline_overview.png"
    artifacts["full_pipeline_overview"] = overview_path
    report["artifacts"]["full_pipeline_overview"] = str(overview_path.resolve())
    cv2.imwrite(str(overview_path), overview_visual(artifacts, report))
    report_path = output / "full_pipeline_report.json"
    markdown_path = output / "full_pipeline_report.md"
    report["artifacts"]["full_pipeline_report_json"] = str(report_path.resolve())
    report["artifacts"]["full_pipeline_report_markdown"] = str(markdown_path.resolve())
    write_json(report_path, _jsonable(report))
    write_markdown(markdown_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "overview": str(overview_path)}, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
