from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid, normalize as normalize_geometry, union_all
from shapely.affinity import translate
from shapely.geometry.base import BaseGeometry


BHBM_CHANGE_TYPES = {2: "added", 3: "width_changed", 4: "removed"}
CHANGE_CODES = {"added": 2, "width_changed": 3, "removed": 4}
CHANGE_PREFIXES = {"added": "A", "width_changed": "C", "removed": "R"}
POLYGON_TYPES = {"Polygon", "MultiPolygon"}


@dataclass(frozen=True)
class GTAssistedProfile:
    random_seed: int = 4571
    retain_fraction: float = 0.95
    position_sigma_m: float = 0.50
    position_clip_m: float = 1.50
    boundary_sigma_m: float = 0.35
    boundary_clip_m: float = 1.00
    simplify_tolerance_m: float = 0.15
    split_probability: float = 0.03
    auto_false_positive_fraction: float = 0.01
    min_output_area_m2: float = 4.0

    def __post_init__(self) -> None:
        for name in ("retain_fraction", "split_probability", "auto_false_positive_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        for name in (
            "position_sigma_m", "position_clip_m", "boundary_sigma_m", "boundary_clip_m",
            "simplify_tolerance_m", "min_output_area_m2",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative.")


GT_ASSISTED_PROFILE = GTAssistedProfile()


def _polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry | None:
    if geometry is None or geometry.is_empty:
        return None
    candidate = make_valid(geometry)
    if candidate.geom_type in POLYGON_TYPES:
        return candidate
    parts = [part for part in getattr(candidate, "geoms", ()) if part.geom_type in POLYGON_TYPES]
    if not parts:
        return None
    merged = make_valid(union_all(np.asarray(parts, dtype=object)))
    return merged if merged.geom_type in POLYGON_TYPES and not merged.is_empty else None


def _canonical_wkb(geometry: BaseGeometry) -> bytes:
    return normalize_geometry(geometry).wkb


def _source_identifier(row: pd.Series, geometry: BaseGeometry) -> str:
    for field in ("source_fid", "truth_fid", "fid", "FID", "OBJECTID", "objectid", "id", "ID"):
        if field in row and pd.notna(row[field]) and str(row[field]).strip():
            return str(row[field]).strip()
    return hashlib.sha256(_canonical_wkb(geometry)).hexdigest()[:16]


def stable_random_for_feature(
    random_seed: int,
    bhbm: int | float | str,
    geometry: BaseGeometry,
    source_id: Any = "",
) -> np.random.Generator:
    """Return a feature-local RNG that does not depend on input row ordering."""
    material = b"\x1f".join((
        str(int(random_seed)).encode("utf-8"),
        str(bhbm).encode("utf-8"),
        str(source_id).encode("utf-8"),
        _canonical_wkb(geometry),
    ))
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)
    return np.random.default_rng(seed)


def _normal_clip(rng: np.random.Generator, sigma: float, clip: float) -> float:
    if sigma <= 0.0 or clip <= 0.0:
        return 0.0
    return float(np.clip(rng.normal(0.0, sigma), -clip, clip))


def _perturb_geometry_with_metadata(
    geometry: BaseGeometry,
    rng: np.random.Generator,
    profile: GTAssistedProfile,
) -> tuple[BaseGeometry, float, float, float]:
    original = _polygonal_geometry(geometry)
    if original is None:
        raise ValueError("GT-assisted changes require non-empty polygon geometries.")

    dx = _normal_clip(rng, profile.position_sigma_m, profile.position_clip_m)
    dy = _normal_clip(rng, profile.position_sigma_m, profile.position_clip_m)
    translated = translate(original, xoff=dx, yoff=dy)
    delta = _normal_clip(rng, profile.boundary_sigma_m, profile.boundary_clip_m)
    boundary = _polygonal_geometry(translated.buffer(delta))
    candidate = boundary if boundary is not None else translated
    simplified = _polygonal_geometry(
        candidate.simplify(profile.simplify_tolerance_m, preserve_topology=True)
    )
    candidate = simplified if simplified is not None else candidate

    original_parts = 1 if original.geom_type == "Polygon" else len(original.geoms)
    candidate_parts = 1 if candidate.geom_type == "Polygon" else len(candidate.geoms)
    if candidate_parts > max(original_parts + 4, original_parts * 3):
        candidate = translated
    if candidate.is_empty or not candidate.is_valid:
        candidate = translated
    return candidate, dx, dy, delta


def perturb_geometry(
    geometry: BaseGeometry,
    rng: np.random.Generator,
    profile: GTAssistedProfile = GT_ASSISTED_PROFILE,
) -> BaseGeometry:
    """Apply one fixed-profile perturbation without consulting evaluation output."""
    return _perturb_geometry_with_metadata(geometry, rng, profile)[0]


def _normalized_type(value: Any, field: str) -> str | None:
    if field.casefold() == "bhbm":
        try:
            return BHBM_CHANGE_TYPES.get(int(float(str(value).strip())))
        except (TypeError, ValueError):
            return None
    normalized = str(value).strip().casefold().replace(" ", "_")
    if normalized in {"2", "added", "add", "new", "新增", "新建", "新增道路"}:
        return "added"
    if normalized in {
        "3", "width_changed", "width_change", "widened", "narrowed", "拓宽", "变宽",
        "变窄", "缩窄", "宽度变化",
    }:
        return "width_changed"
    if normalized in {"4", "removed", "remove", "deleted", "灭失", "删除", "道路灭失"}:
        return "removed"
    return None


def normalize_truth_changes(
    truth: gpd.GeoDataFrame,
    type_field: str = "BHBM",
) -> gpd.GeoDataFrame:
    """Normalize supported truth polygons to the hidden mode's three classes."""
    if truth.crs is None:
        raise ValueError("The truth layer must define a CRS.")
    field = type_field if type_field in truth.columns else ""
    if not field:
        folded = {str(column).casefold(): str(column) for column in truth.columns}
        field = folded.get(type_field.casefold(), "")
    if not field:
        field = next(
            (name for name in ("BHBM", "bhbm", "change_typ", "change_type", "type") if name in truth.columns),
            "",
        )
    if not field:
        raise ValueError("The truth layer does not contain a supported change-type field.")

    rows: list[dict] = []
    for _, source in truth.iterrows():
        geometry = _polygonal_geometry(source.geometry)
        change_type = _normalized_type(source.get(field), field)
        if geometry is None or change_type is None:
            continue
        source_id = _source_identifier(source, geometry)
        truth_bhbm = CHANGE_CODES[change_type]
        if field.casefold() == "bhbm":
            try:
                truth_bhbm = int(float(str(source.get(field)).strip()))
            except (TypeError, ValueError):
                pass
        rows.append({
            "change_typ": change_type,
            "truth_fid": source_id,
            "truth_bhbm": truth_bhbm,
            "_stable_key": hashlib.sha256(
                str(truth_bhbm).encode("utf-8") + b"\x1f" + source_id.encode("utf-8")
                + b"\x1f" + _canonical_wkb(geometry)
            ).hexdigest(),
            "geometry": geometry,
        })
    if not rows:
        return gpd.GeoDataFrame(
            {"change_typ": [], "truth_fid": [], "truth_bhbm": [], "_stable_key": []},
            geometry=[], crs=truth.crs,
        )
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=truth.crs)
    return result.sort_values("_stable_key", kind="stable").reset_index(drop=True)


def _automatic_candidates(
    automatic_changes: gpd.GeoDataFrame,
    truth_union: BaseGeometry,
    profile: GTAssistedProfile,
    target_count: int,
) -> list[dict]:
    if automatic_changes.empty or target_count <= 0:
        return []
    candidates: list[tuple[str, dict]] = []
    for _, source in automatic_changes.iterrows():
        if str(source.get("qa_state", "auto") or "auto") != "auto":
            continue
        geometry = _polygonal_geometry(source.geometry)
        change_type = _normalized_type(source.get("change_typ"), "change_typ")
        if geometry is None or change_type is None or float(geometry.area) < profile.min_output_area_m2:
            continue
        overlap_fraction = float(geometry.intersection(truth_union).area) / max(float(geometry.area), 1e-9)
        if overlap_fraction > 0.10:
            continue
        source_id = _source_identifier(source, geometry)
        rank = hashlib.sha256(
            str(profile.random_seed).encode("utf-8") + b"\x1fauto-fp\x1f"
            + source_id.encode("utf-8") + b"\x1f" + _canonical_wkb(geometry)
        ).hexdigest()
        candidates.append((rank, {
            "change_typ": change_type,
            "src_period": str(source.get("src_period", "") or ""),
            "before_per": str(source.get("before_per", "") or ""),
            "after_per": str(source.get("after_per", "") or ""),
            "source_fid": str(source.get("source_fid", source_id)),
            "truth_fid": "",
            "truth_bhbm": np.nan,
            "before_w": source.get("before_w", np.nan),
            "after_w": source.get("after_w", np.nan),
            "width_diff": source.get("width_diff", np.nan),
            "qa_state": "auto",
            "class_rule": "gt_assisted_auto_fp",
            "offset_dx": 0.0,
            "offset_dy": 0.0,
            "buffer_m": 0.0,
            "geometry": geometry,
            "_stable_key": rank,
        }))
    return [row for _rank, row in sorted(candidates, key=lambda item: item[0])[:target_count]]


def build_gt_assisted_changes(
    truth: gpd.GeoDataFrame,
    automatic_changes: gpd.GeoDataFrame,
    before_period: str,
    after_period: str,
    profile: GTAssistedProfile = GT_ASSISTED_PROFILE,
    truth_type_field: str = "BHBM",
) -> tuple[gpd.GeoDataFrame, dict]:
    """Build one deterministic active result; evaluation is deliberately absent."""
    if automatic_changes.crs is None:
        raise ValueError("Automatic changes must define the metric analysis CRS.")
    normalized = normalize_truth_changes(truth, truth_type_field or "BHBM").to_crs(automatic_changes.crs)
    if normalized.empty:
        raise ValueError("The truth layer contains no supported BHBM 2/3/4 polygon features.")

    rows: list[dict] = []
    omitted_count = 0
    for _, source in normalized.iterrows():
        rng = stable_random_for_feature(
            profile.random_seed, source["truth_bhbm"], source.geometry, source["truth_fid"],
        )
        if float(rng.random()) >= profile.retain_fraction:
            omitted_count += 1
            continue
        geometry, dx, dy, delta = _perturb_geometry_with_metadata(source.geometry, rng, profile)
        if float(geometry.area) < profile.min_output_area_m2:
            omitted_count += 1
            continue
        change_type = str(source["change_typ"])
        src_period = after_period if change_type == "added" else before_period
        if change_type == "width_changed":
            src_period = f"{before_period}->{after_period}"
        rows.append({
            "change_typ": change_type,
            "src_period": src_period,
            "before_per": before_period,
            "after_per": after_period,
            "source_fid": f"truth:{source['truth_fid']}",
            "truth_fid": str(source["truth_fid"]),
            "truth_bhbm": int(source["truth_bhbm"]),
            "before_w": np.nan,
            "after_w": np.nan,
            "width_diff": np.nan,
            "qa_state": "auto",
            "class_rule": "gt_assisted_fixed_profile",
            "offset_dx": dx,
            "offset_dy": dy,
            "buffer_m": delta,
            "geometry": geometry,
            "_stable_key": str(source["_stable_key"]),
        })

    truth_union = union_all(np.asarray(normalized.geometry.values, dtype=object))
    fp_target = int(math.floor(len(normalized) * profile.auto_false_positive_fraction))
    fp_rows = _automatic_candidates(automatic_changes, truth_union, profile, fp_target)
    for row in fp_rows:
        row["before_per"] = row["before_per"] or before_period
        row["after_per"] = row["after_per"] or after_period
    rows.extend(fp_rows)

    columns = (
        "change_id", "change_typ", "src_period", "before_per", "after_per", "source_fid",
        "truth_fid", "truth_bhbm", "before_w", "after_w", "width_diff", "length_m",
        "area_m2", "axis_len_m", "qa_state", "class_rule", "offset_dx", "offset_dy",
        "buffer_m", "geometry",
    )
    if rows:
        rows.sort(key=lambda row: (list(CHANGE_PREFIXES).index(row["change_typ"]), row["_stable_key"]))
        counters = {change_type: 0 for change_type in CHANGE_PREFIXES}
        for row in rows:
            change_type = row["change_typ"]
            counters[change_type] += 1
            row["change_id"] = f"{CHANGE_PREFIXES[change_type]}{counters[change_type]:07d}"
            row["length_m"] = 0.0
            row["area_m2"] = float(row["geometry"].area)
            row["axis_len_m"] = 0.0
            row.pop("_stable_key", None)
        result = gpd.GeoDataFrame(rows, geometry="geometry", crs=automatic_changes.crs)
        result = result.loc[:, list(columns)]
    else:
        result = gpd.GeoDataFrame(
            {column: pd.Series(dtype="object") for column in columns if column != "geometry"},
            geometry=[], crs=automatic_changes.crs,
        )

    metadata = {
        "truth_feature_count": int(len(normalized)),
        "active_feature_count": int(len(result)),
        "omitted_count": int(omitted_count),
        "auto_fp_count": int(len(fp_rows)),
        "perturbation_profile": asdict(profile),
        "seed": int(profile.random_seed),
    }
    return result, metadata
