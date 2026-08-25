from __future__ import annotations

"""Symmetric, scene-relative evidence for cross-period road existence."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from shapely import make_valid
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry


@dataclass(frozen=True)
class RoadExistenceEvidence:
    geometry_coverage: float
    center_probability_mean: float | None
    center_probability_q25: float | None
    local_background_probability: float | None
    local_probability_contrast: float | None
    scene_percentile_rank: float | None
    background_percentile_rank: float | None
    surface_coverage: float | None
    valid_coverage: float | None
    validity_known: bool
    existence_state: str
    existence_reason: str
    evidence_rule: str

    def to_dict(self) -> dict:
        return asdict(self)


class RoadProbabilityRaster:
    """One georeferenced probability raster with per-scene relative statistics."""

    def __init__(self, values: np.ndarray, transform, crs, valid_mask: np.ndarray | None = None):
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Road probability must be a two-dimensional raster.")
        if float(np.nanmax(array, initial=0.0)) > 1.0:
            array = array / 255.0
        self.values = np.clip(array, 0.0, 1.0)
        self.transform = transform
        self.crs = CRS.from_user_input(crs)
        self.valid_mask = (
            np.asarray(valid_mask, dtype=bool)
            if valid_mask is not None else np.isfinite(self.values)
        )
        if self.valid_mask.shape != self.values.shape:
            raise ValueError("Probability valid mask shape does not match the raster.")
        scene = self.values[self.valid_mask & np.isfinite(self.values)]
        if scene.size > 1_000_000:
            stride = int(np.ceil(scene.size / 1_000_000))
            scene = scene[::stride]
        self.scene_values = np.sort(scene.astype(np.float32, copy=False))
        self.scene_percentiles = {
            f"p{percentile}": float(np.percentile(scene, percentile)) if scene.size else None
            for percentile in (50, 90, 95, 99)
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "RoadProbabilityRaster":
        with rasterio.open(path) as dataset:
            masked = dataset.read(1, masked=True)
            values = np.asarray(masked.data, dtype=np.float32)
            valid = ~np.ma.getmaskarray(masked) & np.isfinite(values)
            return cls(values, dataset.transform, dataset.crs, valid)

    def percentile_rank(self, value: float | None) -> float | None:
        if value is None or not np.isfinite(value) or self.scene_values.size == 0:
            return None
        # Strict rank avoids treating a large tied background plateau (often 0)
        # as high roadness merely because most scene pixels share that value.
        return float(np.searchsorted(self.scene_values, value, side="left") / self.scene_values.size)

    def _in_raster_crs(self, geometry: BaseGeometry, geometry_crs) -> BaseGeometry:
        source = CRS.from_user_input(geometry_crs)
        if source == self.crs:
            return geometry
        transformer = Transformer.from_crs(source, self.crs, always_xy=True)
        return transform_geometry(transformer.transform, geometry)

    def sample_cross_section(
        self,
        center: Point,
        normal: tuple[float, float],
        geometry_crs,
        *,
        search_radius: float,
    ) -> dict[str, np.ndarray | float]:
        """Sample one world-coordinate normal without interpreting road width.

        Distances are expressed in the input geometry CRS (the metric analysis
        CRS used by change detection). Samples outside the raster or masked by
        NoData are returned with ``valid_mask=False`` and ``probability=NaN``.
        """
        radius = float(search_radius)
        nx, ny = float(normal[0]), float(normal[1])
        norm = float(np.hypot(nx, ny))
        if radius <= 0 or norm <= 1e-12:
            raise ValueError("Cross-section search radius and normal must be positive.")
        nx, ny = nx / norm, ny / norm
        source_crs = CRS.from_user_input(geometry_crs or self.crs)
        transformer = (
            None if source_crs == self.crs
            else Transformer.from_crs(source_crs, self.crs, always_xy=True)
        )

        def raster_xy(distance: float) -> tuple[float, float]:
            x = center.x + nx * distance
            y = center.y + ny * distance
            return transformer.transform(x, y) if transformer is not None else (x, y)

        inverse = ~self.transform
        center_xy = raster_xy(0.0)
        left_xy = raster_xy(-radius)
        right_xy = raster_xy(radius)
        center_col, center_row = inverse * center_xy
        left_col, left_row = inverse * left_xy
        right_col, right_row = inverse * right_xy
        pixel_span = float(
            np.hypot(left_col - center_col, left_row - center_row)
            + np.hypot(right_col - center_col, right_row - center_row)
        )
        interval_count = max(2, min(4096, int(np.ceil(pixel_span * 2.0))))
        distances = np.linspace(-radius, radius, interval_count + 1, dtype=np.float64)
        source_x = center.x + nx * distances
        source_y = center.y + ny * distances
        if transformer is not None:
            raster_x, raster_y = transformer.transform(source_x, source_y)
        else:
            raster_x, raster_y = source_x, source_y
        columns_float, rows_float = inverse * (raster_x, raster_y)
        columns = np.floor(np.asarray(columns_float)).astype(np.int64)
        rows = np.floor(np.asarray(rows_float)).astype(np.int64)
        inside = (
            (rows >= 0) & (rows < self.values.shape[0])
            & (columns >= 0) & (columns < self.values.shape[1])
        )
        valid = np.zeros(distances.shape, dtype=bool)
        probability = np.full(distances.shape, np.nan, dtype=np.float32)
        if bool(inside.any()):
            inside_indices = np.flatnonzero(inside)
            inside_rows = rows[inside_indices]
            inside_columns = columns[inside_indices]
            usable = (
                self.valid_mask[inside_rows, inside_columns]
                & np.isfinite(self.values[inside_rows, inside_columns])
            )
            usable_indices = inside_indices[usable]
            valid[usable_indices] = True
            probability[usable_indices] = self.values[
                rows[usable_indices], columns[usable_indices]
            ]
        return {
            "distance": distances,
            "probability": probability,
            "valid_mask": valid,
            "sample_step": float(distances[1] - distances[0]),
        }

    def sample_axis(
        self,
        axis: BaseGeometry,
        axis_crs,
        *,
        road_width: float,
        position_tolerance: float,
    ) -> dict[str, float | None]:
        axis = self._in_raster_crs(axis, axis_crs)
        pixel_x = abs(float(self.transform.a))
        pixel_y = abs(float(self.transform.e))
        pixel = max(min(pixel_x, pixel_y), 1e-6)
        center_radius = max(pixel * 1.25, position_tolerance * 0.20, road_width * 0.12)
        inner_radius = max(center_radius * 2.0, road_width * 0.60, pixel * 2.5)
        outer_radius = max(inner_radius + pixel * 3.0, road_width * 1.35, position_tolerance * 1.5)
        center_geometry = make_valid(axis.buffer(center_radius, cap_style="flat"))
        background_geometry = make_valid(
            axis.buffer(outer_radius, cap_style="flat").difference(
                axis.buffer(inner_radius, cap_style="flat")
            )
        )
        bounds = background_geometry.bounds
        raw_window = from_bounds(*bounds, transform=self.transform)
        col0 = max(0, int(np.floor(raw_window.col_off)))
        row0 = max(0, int(np.floor(raw_window.row_off)))
        col1 = min(self.values.shape[1], int(np.ceil(raw_window.col_off + raw_window.width)))
        row1 = min(self.values.shape[0], int(np.ceil(raw_window.row_off + raw_window.height)))
        if col1 <= col0 or row1 <= row0:
            return {
                "center_probability_mean": None, "center_probability_q25": None,
                "local_background_probability": None, "local_probability_contrast": None,
                "scene_percentile_rank": None, "background_percentile_rank": None,
            }
        window = Window(col0, row0, col1 - col0, row1 - row0)
        window_transform = rasterio.windows.transform(window, self.transform)
        shape = (row1 - row0, col1 - col0)
        center_mask = rasterize(
            [(center_geometry, 1)], out_shape=shape, transform=window_transform,
            fill=0, all_touched=True, dtype="uint8",
        ).astype(bool)
        background_mask = rasterize(
            [(background_geometry, 1)], out_shape=shape, transform=window_transform,
            fill=0, all_touched=True, dtype="uint8",
        ).astype(bool)
        values = self.values[row0:row1, col0:col1]
        valid = self.valid_mask[row0:row1, col0:col1]
        center = values[center_mask & valid & np.isfinite(values)]
        background = values[background_mask & valid & np.isfinite(values)]
        center_mean = float(np.mean(center)) if center.size else None
        center_q25 = float(np.quantile(center, 0.25)) if center.size else None
        background_mean = float(np.mean(background)) if background.size else None
        contrast = (
            center_mean - background_mean
            if center_mean is not None and background_mean is not None else None
        )
        return {
            "center_probability_mean": center_mean,
            "center_probability_q25": center_q25,
            "local_background_probability": background_mean,
            "local_probability_contrast": contrast,
            "scene_percentile_rank": self.percentile_rank(center_mean),
            "background_percentile_rank": self.percentile_rank(background_mean),
        }


def _line_coverage(axis: BaseGeometry, support: BaseGeometry | None) -> float | None:
    if support is None:
        return None
    if axis.is_empty or float(axis.length) <= 0 or support.is_empty:
        return 0.0
    return float(np.clip(axis.intersection(support).length / max(float(axis.length), 1e-9), 0.0, 1.0))


def evaluate_road_existence_evidence(
    candidate_axis: BaseGeometry,
    *,
    centerline_cover: BaseGeometry | None,
    road_surface: BaseGeometry | None,
    valid_area: BaseGeometry | None,
    probability: RoadProbabilityRaster | None = None,
    crs=None,
    road_width: float = 0.0,
    position_tolerance: float = 3.0,
    allow_legacy_absence_without_valid_mask: bool = False,
) -> RoadExistenceEvidence:
    """Classify one period as present, absent or uncertain on a known road axis.

    Positive evidence may come from geometry, scene-relative probability or road
    surface.  Absence requires valid observation plus multiple negative sources.
    """
    geometry_coverage = float(_line_coverage(candidate_axis, centerline_cover) or 0.0)
    surface_coverage = _line_coverage(candidate_axis, road_surface)
    validity_known = valid_area is not None
    valid_coverage = (
        float(_line_coverage(candidate_axis, valid_area))
        if validity_known else None
    )
    probability_values = {
        "center_probability_mean": None, "center_probability_q25": None,
        "local_background_probability": None, "local_probability_contrast": None,
        "scene_percentile_rank": None, "background_percentile_rank": None,
    }
    if probability is not None:
        probability_values = probability.sample_axis(
            candidate_axis, crs, road_width=max(road_width, 2.0 * position_tolerance),
            position_tolerance=position_tolerance,
        )
    center_pct = probability_values["scene_percentile_rank"]
    background_pct = probability_values["background_percentile_rank"]
    scene_p50 = probability.scene_percentiles.get("p50") if probability is not None else None
    scene_p99 = probability.scene_percentiles.get("p99") if probability is not None else None
    scene_dynamic = (
        float(scene_p99 - scene_p50)
        if scene_p50 is not None and scene_p99 is not None else None
    )
    percentile_separation = (
        center_pct - background_pct
        if center_pct is not None and background_pct is not None else None
    )
    local_contrast = probability_values["local_probability_contrast"]
    probability_present = bool(
        center_pct is not None
        and scene_dynamic is not None and scene_dynamic >= 0.01
        and center_pct >= 0.85
        and percentile_separation is not None and percentile_separation >= 0.10
        and local_contrast is not None and local_contrast > 0.0
    )
    surface_present = surface_coverage is not None and surface_coverage >= 0.55

    if geometry_coverage >= 0.50:
        state, reason = "present", "geometry_match_strong"
    elif probability_present:
        state, reason = "present", "reference_centerline_missing_but_probability_present"
    elif surface_present:
        state, reason = "present", "reference_centerline_missing_but_surface_present"
    elif not validity_known and not allow_legacy_absence_without_valid_mask:
        state, reason = "uncertain", "valid_observation_unknown"
    elif valid_coverage is not None and valid_coverage < 0.80:
        state, reason = "uncertain", "invalid_or_nodata_reference"
    else:
        probability_negative = bool(
            center_pct is not None and (
                (
                    scene_dynamic is not None and scene_dynamic < 0.01
                    and (probability_values["center_probability_mean"] or 0.0) <= 0.10
                )
                or (
                    center_pct <= 0.70
                    and (percentile_separation is None or percentile_separation <= 0.10)
                )
            )
        )
        surface_negative = surface_coverage is not None and surface_coverage <= 0.10
        geometry_negative = geometry_coverage <= 0.10
        if (
            geometry_negative and surface_negative
            and (
                (valid_coverage is not None and valid_coverage >= 0.90)
                or (not validity_known and allow_legacy_absence_without_valid_mask)
            )
            and (probability_negative or probability is None)
        ):
            state, reason = "absent", "true_absence_confirmed"
        else:
            state, reason = "uncertain", "insufficient_negative_evidence"
    rule = "geometry_or_relative_probability_or_surface_present; multi_negative_valid_observation_absent"
    return RoadExistenceEvidence(
        geometry_coverage=geometry_coverage,
        surface_coverage=surface_coverage,
        valid_coverage=valid_coverage,
        validity_known=validity_known,
        existence_state=state,
        existence_reason=reason,
        evidence_rule=rule,
        **probability_values,
    )


def cross_period_change_decision(
    before: RoadExistenceEvidence,
    after: RoadExistenceEvidence,
    requested_change_type: str,
) -> tuple[str, str]:
    if requested_change_type == "added" and before.existence_state == "absent" and after.existence_state == "present":
        return "auto", "dual_period_confirmed_change"
    if requested_change_type == "removed" and before.existence_state == "present" and after.existence_state == "absent":
        return "auto", "dual_period_confirmed_change"
    reasons = {before.existence_reason, after.existence_reason}
    if "invalid_or_nodata_reference" in reasons:
        return "review", "invalid_or_nodata_reference"
    if "valid_observation_unknown" in reasons:
        return "review", "valid_observation_unknown"
    if "reference_centerline_missing_but_probability_present" in reasons:
        return "review", "reference_centerline_missing_but_probability_present"
    if "reference_centerline_missing_but_surface_present" in reasons:
        return "review", "reference_centerline_missing_but_surface_present"
    if "insufficient_negative_evidence" in reasons:
        return "review", "insufficient_negative_evidence"
    if before.existence_state == after.existence_state == "present":
        return "review", "cross_sensor_extraction_disagreement"
    return "review", "geometry_mismatch_only"


def evidence_audit_fields(before: RoadExistenceEvidence, after: RoadExistenceEvidence) -> dict:
    return {
        "before_state": before.existence_state,
        "after_state": after.existence_state,
        "before_geom_cov": before.geometry_coverage,
        "after_geom_cov": after.geometry_coverage,
        "before_center_evidence": before.center_probability_mean,
        "after_center_evidence": after.center_probability_mean,
        "before_center_pct": before.scene_percentile_rank,
        "after_center_pct": after.scene_percentile_rank,
        "before_contrast": before.local_probability_contrast,
        "after_contrast": after.local_probability_contrast,
        "before_surface_cov": before.surface_coverage,
        "after_surface_cov": after.surface_coverage,
        "before_valid_cov": before.valid_coverage,
        "after_valid_cov": after.valid_coverage,
        "before_valid_known": before.validity_known,
        "after_valid_known": after.validity_known,
        "evidence_rule": before.evidence_rule,
    }
