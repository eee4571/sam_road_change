from __future__ import annotations

"""Fast no-truth change detection on final road products.

The GT-assisted baseline does not import this module. Raster reads are windowed;
matching and run assembly use regional final axes so tile/feature seams do not
become change boundaries. No ground-truth input is accepted here.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from shapely import intersection, line_merge, make_valid, prepare, union_all
from shapely.geometry import LineString, Point, box
from shapely.ops import substring
from shapely.strtree import STRtree

WIDTH_ROOT = Path(__file__).resolve().parent / "width"
if str(WIDTH_ROOT) not in sys.path:
    sys.path.insert(0, str(WIDTH_ROOT))
from road_existence_evidence import RoadProbabilityRaster
from paired_width_profile import (
    PairedWidthConfig, PairedWidthProfile, PairedWidthSample,
    _measure_period_width, candidate_change_runs, evaluate_change_run,
)


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    return [part for child in getattr(geometry, "geoms", ()) for part in _parts(child)]


class WindowedProbability(RoadProbabilityRaster):
    """Metric sampling of a georeferenced raster, including geographic rasters."""

    def __init__(self, path, metric_crs):
        self.dataset = rasterio.open(path)
        self.crs = CRS.from_user_input(self.dataset.crs)
        self.metric_crs = CRS.from_user_input(metric_crs)
        self.to_raster = Transformer.from_crs(metric_crs, self.crs, always_xy=True)
        self.inverse = ~self.dataset.transform
        to_metric = Transformer.from_crs(self.crs, metric_crs, always_xy=True)
        x, y = self.dataset.xy(self.dataset.height//2, self.dataset.width//2)
        x1, y1 = self.dataset.xy(self.dataset.height//2, self.dataset.width//2+1)
        a, b = to_metric.transform(x, y), to_metric.transform(x1, y1)
        self.pixel_size = float(np.hypot(b[0]-a[0], b[1]-a[1]))
        scale = max(1, int(np.ceil(max(self.dataset.shape) / 1000)))
        scene = self.dataset.read(1, out_shape=(max(1, self.dataset.height // scale),
                                               max(1, self.dataset.width // scale)), masked=True)
        self.divisor = 255.0 if float(scene.max()) > 1.0 else 1.0
        values = np.asarray(scene.compressed(), dtype=np.float32) / self.divisor
        values = values[np.isfinite(values)]
        # Float64 avoids a million-element dtype conversion on each scalar search.
        self.scene_values = np.sort(values.astype(np.float64))
        self.scene_percentiles = {f"p{q}": float(np.percentile(values, q)) if len(values) else None
                                  for q in (50, 90, 95, 99)}

    def close(self):
        self.dataset.close()

    def _values_at(self, x, y):
        x, y = self.to_raster.transform(np.asarray(x), np.asarray(y))
        cols, rows = self.inverse * (np.asarray(x), np.asarray(y))
        cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
        result = np.full(cols.shape, np.nan, dtype=float)
        inside = (cols >= 0) & (rows >= 0) & (cols < self.dataset.width) & (rows < self.dataset.height)
        if not inside.any():
            return result
        c0, c1 = int(cols[inside].min()), int(cols[inside].max()) + 1
        r0, r1 = int(rows[inside].min()), int(rows[inside].max()) + 1
        values = self.dataset.read(1, window=rasterio.windows.Window(c0, r0, c1-c0, r1-r0), masked=True)
        picked = values[rows[inside]-r0, cols[inside]-c0].astype(float).filled(np.nan)
        result[inside] = picked / self.divisor
        return result

    def sample_cross_section(self, center, normal, geometry_crs, *, search_radius):
        distances = np.arange(-search_radius, search_radius + 0.25, 0.5)
        values = self._values_at(center.x + normal[0]*distances, center.y + normal[1]*distances)
        return {"distance": distances, "probability": values,
                "valid_mask": np.isfinite(values), "sample_step": 0.5}

    def sample_axis(self, axis, axis_crs, *, road_width, position_tolerance):
        positions = np.linspace(0, axis.length, max(3, int(np.ceil(axis.length/2))+1))
        centers, backgrounds = [], []
        for station in positions:
            point = axis.interpolate(float(station))
            normal = _normal(axis, float(station))
            inner = max(road_width * 0.70, position_tolerance + 2.0)
            for offset in (-1.0, 0.0, 1.0):
                centers.append((point.x + offset*normal[0], point.y + offset*normal[1]))
            for offset in (-inner-3, -inner, inner, inner+3):
                backgrounds.append((point.x + offset*normal[0], point.y + offset*normal[1]))
        coords = np.asarray(centers + backgrounds)
        values = self._values_at(coords[:, 0], coords[:, 1])
        center, background = values[:len(centers)], values[len(centers):]
        center, background = center[np.isfinite(center)], background[np.isfinite(background)]
        mean = float(np.mean(center)) if len(center) else None
        bg = float(np.median(background)) if len(background) else None
        return {"center_probability_mean": mean,
                "center_probability_q25": float(np.quantile(center, .25)) if len(center) else None,
                "local_background_probability": bg,
                "local_probability_contrast": mean-bg if mean is not None and bg is not None else None,
                "scene_percentile_rank": self.percentile_rank(mean),
                "background_percentile_rank": self.percentile_rank(bg),
                "probability_valid_ratio": float(np.isfinite(values).mean())}


def _normal(line, station):
    a, b = line.interpolate(max(0, station-3)), line.interpolate(min(line.length, station+3))
    direction = np.array([b.x-a.x, b.y-a.y])
    norm = np.linalg.norm(direction)
    return np.array([-direction[1], direction[0]]) / max(norm, 1e-9)


class RoadScene:
    def __init__(self, centerlines, surfaces, widths, valid, probability, crs):
        self.lines = _parts(line_merge(union_all(centerlines.geometry.values)))
        self.tree = STRtree(self.lines)
        self.surfaces = surfaces
        self.surface_tree = STRtree(surfaces.geometry.values)
        self.widths = widths
        self.width_tree = STRtree(widths.geometry.values)
        self.valid = make_valid(union_all(valid.geometry.values))
        prepare(self.valid)
        self.probability = probability
        self.crs = crs
        endpoints = [Point(line.coords[end]) for line in self.lines for end in (0, -1)]
        endpoint_tree = STRtree(endpoints)
        self.junction = union_all([point.buffer(12) for point in endpoints
                                   if len(endpoint_tree.query(point, predicate="dwithin", distance=1.5)) >= 3])
        prepare(self.junction)

    def surface(self, axis, radius=65):
        ids = self.surface_tree.query(axis, predicate="dwithin", distance=radius)
        left, bottom, right, top = axis.bounds
        window = box(left-radius, bottom-radius, right+radius, top+radius)
        # A single connected surface can span the entire scene. Clip before
        # union/buffering so local sections never process that entire polygon.
        return union_all(intersection(self.surfaces.geometry.values[ids], window))

    def width(self, point):
        ids = self.width_tree.query(point, predicate="dwithin", distance=3)
        if not len(ids):
            return 6.0
        index = min(ids, key=lambda i: self.widths.geometry.iloc[i].distance(point))
        value = self.widths.iloc[index].get("width_m", 6.0)
        return float(value) if pd.notna(value) and value > 0 else 6.0

    def match(self, axis, station, tolerance, source_width):
        point = axis.interpolate(station)
        local = substring(axis, max(0, station-6), min(axis.length, station+6))
        normal = _normal(axis, station)
        ranked = []
        for target_id in self.tree.query(point, predicate="dwithin", distance=tolerance+1e-6):
            target = self.lines[int(target_id)]
            target_station = target.project(point)
            target_point = target.interpolate(target_station)
            distance = point.distance(target_point)
            cosine = abs(float(np.dot(normal, _normal(target, target_station))))
            target_local = substring(target, max(0, target_station-6), min(target.length, target_station+6))
            overlap = local.intersection(target_local.buffer(tolerance+0.1)).length / max(local.length, 1e-9)
            if cosine < .90 or overlap < .50:
                continue
            target_width = self.width(target_point)
            corridor_overlap = local.buffer(source_width/2).intersection(target_local.buffer(target_width/2)).area
            corridor_overlap /= max(min(local.length*source_width, target_local.length*target_width), 1e-9)
            compatibility = min(source_width, target_width) / max(source_width, target_width)
            # Distance dominates: width is supporting evidence, never a veto of real widening.
            score = distance + 2*(1-cosine) + (1-overlap) + .25*(1-corridor_overlap) + .15*(1-compatibility)
            ranked.append((score, int(target_id), target_station, distance, cosine, overlap, corridor_overlap))
        ranked.sort()
        if not ranked:
            return None
        best = ranked[0]
        ambiguous = False
        if len(ranked) > 1 and ranked[1][0]-best[0] < .5:
            other = self.lines[ranked[1][1]].interpolate(ranked[1][2])
            # Two features meeting at one node are segmentation, distinct nearby axes are ambiguous tracks.
            ambiguous = other.distance(self.lines[best[1]].interpolate(best[2])) > 1.0
        return {"target": best[1], "station": best[2], "distance": best[3],
                "direction": best[4], "coverage": best[5], "corridor": best[6],
                "reliable": not ambiguous}

    def evidence(self, axis, geometry_present, width, tolerance, surface_support=None):
        footprint = axis.buffer(max(width/2, tolerance)+1, cap_style="flat")
        valid = self.valid.covers(footprint)
        if surface_support is None:
            surface_support = self.surface(axis, max(width, tolerance)+2).buffer(tolerance)
        coverage = axis.intersection(surface_support).length / max(axis.length, 1e-9)
        probability = self.probability.sample_axis(axis, self.crs, road_width=width, position_tolerance=tolerance)
        rank = probability["scene_percentile_rank"]
        bg_rank = probability["background_percentile_rank"]
        contrast = probability["local_probability_contrast"]
        supported = rank is not None and bg_rank is not None and rank >= .85 and rank-bg_rank >= .1 and (contrast or 0) > 0
        negative = rank is not None and rank <= .70 and (bg_rank is None or rank-bg_rank <= .10)
        valid = valid and probability.get("probability_valid_ratio", 1.) >= .99
        if not valid:
            state, reason = "uncertain", "invalid_or_boundary"
        elif geometry_present:
            state, reason = "present", "geometry"
        elif coverage >= .55:
            state, reason = "present", "surface_without_centerline"
        elif supported:
            state, reason = "present", "probability_without_centerline"
        elif coverage <= .1 and negative:
            state, reason = "absent", "all_negative"
        else:
            state, reason = "uncertain", "conflicting_or_weak_evidence"
        return {"state": state, "reason": reason, "geometry": bool(geometry_present),
                "surface": float(coverage), "probability": bool(supported), "valid": bool(valid),
                **probability}


def analyze_scenes(before, after, *, tolerance=3., absolute=2., relative=.2, minimum_length=24., minimum_area=4.,
                   presence_audit=None):
    """Presence uses longitudinal coverage; width retains its paired station path."""
    from .auto_presence_candidates import LongitudinalCoverage, presence_seeds
    records, audit, width_audit = [], [], []
    counts = Counter()
    width_config = PairedWidthConfig(sample_spacing=4, absolute_change=absolute,
                                    relative_change=relative, minimum_continuous_length=minimum_length,
                                    maximum_gap_samples=1, maximum_gap_length=8.)
    for side, source, target, change_type in (("before", before, after, "removed"),
                                               ("after", after, before, "added")):
        coverage = LongitudinalCoverage(target.lines, tolerance)
        for line_id, axis in enumerate(source.lines):
            count = max(1, int(np.ceil(axis.length/4)))
            spacing = axis.length/count
            samples, station_rows = [], []
            source_surface, target_surface = source.surface(axis), target.surface(axis)
            source_support, target_support = source_surface.buffer(tolerance), target_surface.buffer(tolerance)
            width_surfaces = ((source_surface, source_surface.buffer(.1)),
                              (target_surface, target_surface.buffer(.1)))
            for index in range(count):
                station = (index+.5)*spacing
                point = axis.interpolate(station)
                cell = substring(axis, index*spacing, (index+1)*spacing)
                width = source.width(point)
                match = target.match(axis, station, tolerance, width)
                source_evidence = source.evidence(cell, True, width, tolerance, source_support)
                target_evidence = target.evidence(cell, match is not None, width, tolerance, target_support)
                bef, aft = (source_evidence, target_evidence) if side == "before" else (target_evidence, source_evidence)
                junction = source.junction.intersects(cell) or target.junction.intersects(cell)
                accepted = bef["valid"] and aft["valid"] and not junction
                row = {"side": side, "axis_id": line_id, "station_m": station,
                       "candidate_type": change_type, "matched": match is not None,
                       "match_reliable": bool(match and match["reliable"]),
                       "offset_m": match["distance"] if match else None,
                       "target_axis": match["target"] if match else None,
                       "junction": junction, "existence_pass": False,
                       "reason": f'{bef["reason"]}->{aft["reason"]}',
                       "geometry": cell,
                       **{f"before_{k}": v for k, v in bef.items()},
                       **{f"after_{k}": v for k, v in aft.items()}}
                station_rows.append(row)
                counts[f"{side}_matched_cells" if match else f"unmatched_{side}_cells"] += 1
                if match is None:
                    counts[f"{change_type}_candidate_cells"] += 1
                if side != "before":
                    continue
                target_point = target.lines[match["target"]].interpolate(match["station"]) if match else point
                common = Point((point.x+target_point.x)/2, (point.y+target_point.y)/2)
                valid_width = accepted and match is not None and match["reliable"] and bef["state"] == aft["state"] == "present"
                if valid_width:
                    reverse = source.match(target.lines[match["target"]], match["station"], tolerance, target.width(target_point))
                    valid_width = reverse is not None and reverse["target"] == line_id and reverse["reliable"]
                before_width = after_width = None
                reason = "unreliable_match_or_existence_or_junction"
                if valid_width:
                    normal = _normal(axis, station)
                    target_normal = _normal(target.lines[match["target"]], match["station"])
                    if np.dot(normal, target_normal) < 0:
                        target_normal = -target_normal
                    normal = normal + target_normal
                    normal /= max(np.linalg.norm(normal), 1e-9)
                    measurements = []
                    for (scene, centre), (surface, support) in zip(((source, point), (target, target_point)), width_surfaces):
                        measurements.append(_measure_period_width(centre, normal, surface, support,
                                                                  scene.probability, scene.crs, width_config))
                    b, a = measurements
                    before_width, after_width = b.final_width, a.final_width
                    reason = ";".join(filter(None, (b.reject_reason, a.reject_reason)))
                    valid_width = before_width is not None and after_width is not None and not reason
                    if valid_width:
                        # Full measured cross sections must lie in both valid areas.
                        radius = max(before_width, after_width)/2+tolerance+1
                        valid_width = source.valid.covers(point.buffer(radius)) and target.valid.covers(target_point.buffer(radius))
                        if not valid_width:
                            reason = "width_cross_section_boundary"
                samples.append(PairedWidthSample(str(line_id), index, station, station/max(axis.length, 1e-9), common,
                                                before_width, after_width,
                                                after_width-before_width if valid_width else None,
                                                bool(valid_width), reason))
            local, intervals, presence_counts = presence_seeds(
                axis, station_rows, source, coverage, change_type, minimum_length, minimum_area)
            records.extend(local)
            counts.update(presence_counts)
            if presence_audit is not None:
                presence_audit.extend(intervals)
            audit.extend(station_rows)
            if side == "before" and len(samples) >= 2:
                # Stations follow the before road; paired points define a shared local axis.
                # Keep source stationing for run lengths, paired normals for each measurement.
                profile = PairedWidthProfile(str(line_id), axis, tuple(samples), sum(s.valid for s in samples)/len(samples))
                counts["width_matched_candidate_axes"] += int(any(row["matched"] for row in station_rows))
                counts["width_valid_profile_axes"] += int(len(profile.valid_samples) >= width_config.minimum_samples)
                runs = candidate_change_runs(profile, width_config)
                counts["width_threshold_runs"] += len(runs)
                for run in runs:
                    decision = evaluate_change_run(run.samples, axis_length=run.axis.length,
                                                   valid_ratio=run.valid_ratio, config=width_config)
                    pixel_uncertainty = .5*np.hypot(getattr(before.probability, "pixel_size", 1.),
                                                   getattr(after.probability, "pixel_size", 1.))
                    uncertainty_pass = abs(run.width_diff) >= width_config.uncertainty_scale*max(pixel_uncertainty, run.uncertainty)
                    counts["width_uncertainty_passed_runs"] += int(uncertainty_pass)
                    counts["width_continuity_passed_runs"] += int(uncertainty_pass and run.axis.length >= minimum_length)
                    decision["accepted"] = bool(decision["accepted"] and uncertainty_pass and all(s.valid for s in run.samples))
                    decision["paired_pixel_uncertainty_m"] = pixel_uncertainty
                    if not all(s.valid for s in run.samples):
                        decision["reject_reason"] = "invalid_sample_gap"
                    if not uncertainty_pass:
                        decision["reject_reason"] = "paired_uncertainty_or_pixel_floor"
                    width_audit.append({"axis_id": line_id, "sign": run.sign, **decision, "geometry": run.axis})
                    if decision["accepted"]:
                        points = [s.point for s in run.samples]
                        canonical = LineString(points)
                        outer, inner = max(run.before_width, run.after_width), min(run.before_width, run.after_width)
                        geometry = canonical.buffer(outer/2, cap_style="flat").difference(canonical.buffer(inner/2, cap_style="flat"))
                        if geometry.area >= minimum_area:
                            records.append({"change_typ": "widened" if run.sign > 0 else "narrowed",
                                            "width_bef": run.before_width, "width_aft": run.after_width,
                                            "width_diff": run.width_diff, "length_m": run.axis.length,
                                            "axis_wkt": canonical.wkt, "geometry": geometry})
                # Include profiles with no threshold run so rejection is visible.
                if not runs:
                    width_audit.append({"axis_id": line_id, "accepted": False,
                                        "sample_count": len(profile.valid_samples), "valid_ratio": profile.valid_ratio,
                                        "reject_reason": "no_sustained_threshold_samples" if profile.valid_samples else "no_valid_paired_samples",
                                        "geometry": axis})
            if line_id % 25 == 0:
                print(f"[Fast Auto] {side} axes {line_id+1}/{len(source.lines)}", flush=True)
    return records, audit, width_audit, dict(counts)


def detect_final_road_changes(before_result, after_result, output_dir, *, before_period, after_period,
                              position_tolerance, width_change_absolute, width_change_ratio,
                              min_change_area, min_change_length, internal_outputs):
    from .fast_pipeline import _load_fast_period_result, _read_fast_change_layer
    started = time.perf_counter()
    payloads = [_load_fast_period_result(value) for value in (before_result, after_result)]
    centerlines = [_read_fast_change_layer(p, "centerlines") for p in payloads]
    output_crs = centerlines[0].crs
    metric_crs = centerlines[0].estimate_utm_crs() if output_crs.is_geographic else output_crs
    if not metric_crs.is_projected or abs(metric_crs.axis_info[0].unit_conversion_factor-1) > 1e-6:
        metric_crs = centerlines[0].estimate_utm_crs()
    scenes = []
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for p, lines in zip(payloads, centerlines):
            surfaces, widths, valid = [_read_fast_change_layer(p, key).to_crs(metric_crs)
                                       for key in ("surfaces", "width_segments", "valid_observation")]
            probability = WindowedProbability(p["road_probability"], metric_crs)
            scenes.append(RoadScene(lines.to_crs(metric_crs), surfaces, widths, valid, probability, metric_crs))
        presence_audit = []
        records, audit, width_audit, counts = analyze_scenes(
            *scenes, tolerance=float(position_tolerance), absolute=float(width_change_absolute),
            relative=float(width_change_ratio), minimum_length=24. if min_change_length is None else float(min_change_length),
            minimum_area=float(min_change_area), presence_audit=presence_audit)
    finally:
        for scene in scenes:
            scene.probability.close()
    return finalize_auto_candidates(records, audit, width_audit, counts, presence_audit=presence_audit,
                                    scenes=dict(zip(("before", "after"), scenes)), centerlines=centerlines,
                                    output_dir=output_dir, before_period=before_period, after_period=after_period,
                                    position_tolerance=position_tolerance, min_change_area=min_change_area,
                                    min_change_length=min_change_length, elapsed_seconds=time.perf_counter()-started)


def finalize_auto_candidates(records, audit, width_audit, counts, *, presence_audit, scenes, centerlines,
                             output_dir, before_period, after_period, position_tolerance=3.,
                             min_change_area=4., min_change_length=None, elapsed_seconds=0.):
    """Qualify, assemble and publish; also reusable with saved observation evidence."""
    from .fast_pipeline import _fast_polygon_parts, _write_fast_public_changes
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_crs, output_crs = scenes["before"].crs, centerlines[0].crs
    counts = dict(counts)
    def frame(rows):
        return (gpd.GeoDataFrame(rows, geometry="geometry", crs=metric_crs) if rows else
                gpd.GeoDataFrame({"change_typ": pd.Series(dtype=str)}, geometry=[], crs=metric_crs))
    from .auto_change_assembly import assemble_change_objects, write_assembly_audit
    from .auto_presence_candidates import annotate_objects, qualify_presence_candidates
    raw_candidates = frame(records)
    for key, default in (("qa_state", "confirmed"), ("confidence", .9), ("audit_reason", "paired_width_accepted")):
        raw_candidates[key] = raw_candidates[key].fillna(default) if key in raw_candidates else default
    raw_candidates["candidate_id"] = np.arange(len(raw_candidates))
    observation = frame(audit)
    seeds, candidate_audit = qualify_presence_candidates(raw_candidates, scenes, observation,
                                minimum_length=24. if min_change_length is None else float(min_change_length),
                                minimum_area=float(min_change_area))
    changes, assembly = assemble_change_objects(seeds, centerlines[0], centerlines[1])
    changes = annotate_objects(changes, seeds, assembly["membership"])
    assembly["change_objects"] = changes
    write_assembly_audit(output_dir, seeds, assembly)
    changes = changes.to_crs(output_crs)
    # Reprojection of touching width ribbons can create sub-pixel ring
    # intersections. Keep only valid polygon components for GIS publication.
    changes.geometry = changes.geometry.map(lambda g: union_all(_fast_polygon_parts(g, min_area=0.)))
    changes["before_per"], changes["after_per"] = before_period, after_period
    _, public_path = _write_fast_public_changes(changes, output_dir)
    gpkg = output_dir / "auto_diagnostics.gpkg"
    changes.to_file(gpkg, layer="changes", driver="GPKG")
    observation.to_file(gpkg, layer="existence_candidates", driver="GPKG")
    raw_candidates.to_file(gpkg, layer="input_candidates", driver="GPKG")
    candidate_audit.to_file(gpkg, layer="candidate_audit", driver="GPKG")
    candidate_audit.loc[candidate_audit.publication_state == "review"].to_file(gpkg, layer="review_candidates", driver="GPKG")
    frame(presence_audit).to_file(gpkg, layer="presence_intervals", driver="GPKG")
    seeds.to_file(gpkg, layer="local_seeds", driver="GPKG")
    frame(width_audit).to_file(gpkg, layer="width_candidates", driver="GPKG")
    layers = {"changes": str(public_path)}
    names = {"added": "added_roads.shp", "removed": "removed_roads.shp",
             "widened": "widened_road_parts.shp", "narrowed": "narrowed_road_parts.shp"}
    for kind, name in names.items():
        selected = changes.loc[changes.change_typ == kind]
        selected.to_file(output_dir/name, encoding="UTF-8")
        layers[kind] = str(output_dir/name)
        counts[f"final_{kind}"] = len(selected)
    evidence = pd.DataFrame([{k: v for k, v in row.items() if k != "geometry"} for row in audit])
    evidence.to_csv(output_dir/"existence_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in row.items() if k != "geometry"} for row in width_audit]).to_csv(
        output_dir/"width_candidates.csv", index=False, encoding="utf-8-sig")
    funnel = {"count_units": "evidence: 4 m station cells; presence candidates: longitudinal intervals; final: network objects",
              "road_matching": {k: v for k, v in counts.items() if "matched" in k and not k.startswith("width")},
              "width": {k: v for k, v in counts.items() if k.startswith("width")},
              "final": {k: v for k, v in counts.items() if k.startswith("final")}}
    for kind, side in (("added", "after"), ("removed", "before")):
        candidates = evidence.loc[(evidence.side == side) & ~evidence.matched] if len(evidence) else evidence
        funnel[kind] = {"candidate_cells": len(candidates)}
        if len(candidates):
            for period in ("before", "after"):
                funnel[kind][period] = {
                    "geometry_support": int(candidates[f"{period}_geometry"].sum()),
                    "surface_support": int((candidates[f"{period}_surface"] >= .55).sum()),
                    "probability_support": int(candidates[f"{period}_probability"].sum()),
                    "valid_area_pass": int(candidates[f"{period}_valid"].sum()),
                    **candidates[f"{period}_state"].value_counts().to_dict()}
            funnel[kind]["existence_pass"] = int(candidates.existence_pass.sum())
            funnel[kind]["continuity_pass"] = int(candidates.get("continuity_pass", pd.Series(dtype=bool)).eq(True).sum())
        funnel[kind]["final_auto_count"] = counts[f"final_{kind}"]
        funnel[kind]["longitudinal"] = {k.removeprefix(kind+"_"): v for k, v in counts.items() if k.startswith(kind+"_")}
        local = seeds.loc[seeds.change_typ == kind]
        final = changes.loc[changes.change_typ == kind]
        funnel[kind]["local_seed_count"] = len(local)
        funnel[kind]["local_seed_length_m"] = float(local.length_m.sum()) if len(local) else 0.
        funnel[kind]["seed_qa_counts"] = {state: int((local.qa_state == state).sum()) for state in ("confirmed", "probable", "uncertain")}
        funnel[kind]["object_qa_counts"] = {state: int((final.qa_state == state).sum()) for state in ("confirmed", "probable", "uncertain")}
        qa = candidate_audit.loc[candidate_audit.change_typ == kind]
        funnel[kind]["recall_candidate_count"] = len(qa)
        funnel[kind]["review_candidate_count"] = int((qa.publication_state == "review").sum())
        funnel[kind]["precision_reason_counts"] = dict(Counter(reason for reasons in qa.precision_reason for reason in reasons.split(";")))
    funnel["network_assembly"] = assembly["summary"]
    funnel["assembly_rejection_counts"] = (assembly["decisions"].loc[~assembly["decisions"].accepted, "reason"].value_counts().to_dict()
                                             if len(assembly["decisions"]) else {})
    funnel["count_units"] += "; assembled final: network objects"
    (output_dir/"candidate_funnel.json").write_text(json.dumps(funnel, indent=2, ensure_ascii=False), encoding="utf-8")
    return complete_written_auto_result(output_dir, before_period=before_period, after_period=after_period,
                                        min_change_length=min_change_length, elapsed_seconds=elapsed_seconds+time.perf_counter()-started)


def complete_written_auto_result(output_dir, *, before_period, after_period, min_change_length=None,
                                 elapsed_seconds=None):
    """Publish previews/summary from written results; also resume a failed preview."""
    from road_change_detection import render_change_preview
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    public_path = output_dir/"road_changes.shp"
    gpkg = output_dir/"auto_diagnostics.gpkg"
    changes = gpd.read_file(public_path)
    names = {"added": "added_roads.shp", "removed": "removed_roads.shp",
             "widened": "widened_road_parts.shp", "narrowed": "narrowed_road_parts.shp"}
    layers = {"changes": str(public_path), **{kind: str(output_dir/name) for kind, name in names.items()}}
    preview = output_dir/"change_preview.png"
    render_change_preview(preview, changes, changes.iloc[:0], title=f"Auto {before_period} to {after_period}",
                          empty_message="No Auto change candidates")
    summary = {"execution_profile": "fast", "change_source": "fast_automatic", "change_output_mode": "fast_automatic",
               "automatic_result": True, "ground_truth_derived": False, "ground_truth_used": False,
               "before_period": before_period, "after_period": after_period,
               "presence_change_source": "final_axis_symmetric_qualified_candidates", "width_change_source": "paired_local_profile",
               "min_change_length_m": 24. if min_change_length is None else float(min_change_length),
               "changes_feature_count": len(changes), **{f"{k}_feature_count": int((changes.change_typ == k).sum()) for k in names},
               "candidate_funnel": str(output_dir/"candidate_funnel.json"), "diagnostics": str(gpkg),
               "auto_change_total_seconds": elapsed_seconds+time.perf_counter()-started if elapsed_seconds is not None else None}
    if (output_dir/"assembly_summary.json").is_file():
        summary["network_assembly"] = str(output_dir/"assembly_summary.json")
    summary_path = output_dir/"change_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output": str(output_dir), "summary": str(summary_path), "road_changes": str(public_path),
            "layers": layers, "gpkg": str(gpkg), "previews": {"change": str(preview)}, "road_change": str(preview), **summary}
