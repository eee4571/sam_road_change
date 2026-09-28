"""Experiment-only artifacts and human review plots."""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely import union_all
from shapely import wkt
from shapely.geometry import mapping

from evidence import stations
from graph import as_frame, stats


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def save_lines(path, frame, roi):
    frame = frame.copy()
    frame.geometry = frame.geometry.intersection(roi)
    frame = frame[~frame.geometry.is_empty].explode(index_parts=False)
    frame = frame[frame.geom_type == "LineString"]
    frame.to_file(path, layer="centerlines", driver="GPKG")


def draw(ax, geometry, **style):
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        xy = np.array(geometry.coords)
        ax.plot(xy[:, 0], xy[:, 1], **style)
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            draw(ax, part, **style)


def background(ax, evidence):
    grid = evidence.image
    if not hasattr(grid, "display_rgb"):
        rgb = grid.rgb.copy()
        for c in range(3):
            valid = rgb[..., c][grid.valid]
            lo, hi = np.quantile(valid, [.02, .98]) if len(valid) else (0, 1)
            rgb[..., c] = np.clip((rgb[..., c]-lo)/max(hi-lo, 1e-6), 0, 1)
        grid.display_rgb = np.nan_to_num(rgb)
    rgb = grid.display_rgb
    h, w = grid.valid.shape
    left, top = grid.transform*(0, 0)
    right, bottom = grid.transform*(w, h)
    ax.imshow(rgb, extent=(left, right, bottom, top), origin="upper")


def review_plot(path, baseline, after, changes, audit, evidence, roi, title):
    fig, ax = plt.subplots(figsize=(11, 11))
    background(ax, evidence)
    for axis in baseline.geometry:
        draw(ax, axis, color="#ffc83d", linewidth=1.1, alpha=.8)
    for e in after:
        draw(ax, e.axis, color="#38c9ff", linewidth=.8, alpha=.8)
    for row in audit:
        if row["decision"] == "review":
            draw(ax, wkt.loads(row["candidate_wkt"]), color="#e35eff", linewidth=2, alpha=.85)
    for change in changes:
        draw(ax, change["geometry"], color="#47ff60" if change["decision"] == "connect" else "#ff4141", linewidth=2.5)
    x0, y0, x1, y1 = roi.bounds
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal", title=title,
           xlabel="Metric easting (m)", ylabel="Metric northing (m)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=c, label=s) for c, s in
                       [("#ffc83d", "baseline"), ("#38c9ff", "result"), ("#ff4141", "deleted"),
                        ("#47ff60", "added"), ("#e35eff", "review")]], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def candidate_sheets(directory, audit, baseline, evidence):
    """Every candidate gets a local before/decision panel and an evidence profile."""
    if not audit:
        return
    directory.mkdir(exist_ok=True)
    for start in range(0, len(audit), 6):
        batch = audit[start:start+6]
        fig, axes = plt.subplots(len(batch), 2, figsize=(12, 4*len(batch)), squeeze=False)
        for row, (ax, profile_ax) in zip(batch, axes):
            axis = wkt.loads(row["candidate_wkt"])
            bounds = axis.buffer(max(12., min(axis.length*.2, 40.))).bounds
            background(ax, evidence)
            for line in baseline.geometry:
                if line.intersects(axis.envelope.buffer(40)):
                    draw(ax, line, color="#ffc83d", linewidth=1.)
            draw(ax, axis, color="#ff4141", linewidth=2.)
            draw(ax, wkt.loads(row["result_wkt"]), color="#38c9ff", linewidth=1.2)
            if row.get("comparison_wkt"):
                draw(ax, wkt.loads(row["comparison_wkt"]), color="#47ff60", linewidth=2.)
            ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal",
                   title=f"{row['candidate_id']}\n{row['decision']}: {row['reason']}")
            ax.ticklabel_format(useOffset=False, style="plain")
            score = row["evidence"]
            if row["type"] in ("duplicate", "boundary_axis"):
                data = score["cross_sections"]
                for name, color in [("probability", "blue"), ("gray", "black")]:
                    values = np.asarray(data[name], float)
                    for trace in values:
                        profile_ax.plot(data["fractions"], trace, color=color, alpha=.1)
                    finite = np.isfinite(values).sum(axis=0)
                    mean = np.nansum(values, axis=0)/np.maximum(finite, 1)
                    profile_ax.plot(data["fractions"], np.where(finite, mean, np.nan), color=color, label=name)
                profile_ax.axvline(0, color="red", linestyle=":")
                profile_ax.axvline(1, color="green", linestyle=":")
                profile_ax.set_xlabel("Cross-section fraction (axes at 0 and 1)")
            else:
                data = score["profile"]
                for name in ("probability", "molra", "left_edge", "right_edge"):
                    profile_ax.plot(data["stations_m"], data[name], label=name)
                profile_ax.set_xlabel("Distance along candidate (m); spur uses terminal 65%")
            profile_ax.set_ylabel("Image evidence (gray is normalized intensity)")
            profile_ax.legend()
            profile_ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(directory / f"candidates_{start//6+1:03d}.png", dpi=110)
        plt.close(fig)


def ribbon_overview(path, audit, baseline, evidence, roi):
    fig, ax = plt.subplots(figsize=(12, 10))
    background(ax, evidence)
    for axis in baseline.geometry:
        draw(ax, axis, color="#20ddff", linewidth=.9)
    for row in audit:
        draw(ax, wkt.loads(row["result_wkt"]), color="#ff38c7", linewidth=3)
    if audit:
        bounds = union_all([wkt.loads(r["result_wkt"]) for r in audit]).buffer(60).intersection(roi).bounds
    else:
        bounds = roi.bounds
    ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal",
           title="Cyan: existing axes | Magenta: RGB interior-axis hypotheses (REVIEW ONLY)",
           xlabel="Easting (m)", ylabel="Northing (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metrics(edges, changes, baseline, evidence, roi, config):
    result = stats(edges, roi, config)
    result["added_connections"] = sum(c["decision"] == "connect" for c in changes)
    result["deleted_length_m"] = sum(c["geometry"].length for c in changes if c["decision"] != "connect")
    distances = []
    reference = union_all(baseline.geometry)
    for c in changes:
        # On the changed geometries only. Additions measure distance to A;
        # duplicate removal measures displacement onto the retained axis.
        if c["decision"] in ("merge", "connect"):
            _, xy, _, _ = stations(c["geometry"])
            from shapely import points, distance
            distances.extend(distance(points(xy), c["reference"] if c["decision"] == "merge" else reference).tolist())
    result["modification_mean_displacement_m"] = float(np.mean(distances)) if distances else None
    result["modification_max_displacement_m"] = max(distances) if distances else None
    result["displacement_definition"] = "samples on added/merged axes to baseline/retained axis; deletions have no matched displacement"
    values, weights = [], []
    for e in edges:
        clipped = e.axis.intersection(roi)
        parts = [clipped] if clipped.geom_type == "LineString" else list(getattr(clipped, "geoms", []))
        for line in parts:
            if line.geom_type != "LineString" or line.length < .1:
                continue
            # Same fixed images and masks for all arms of the ablation.
            score = evidence.measure(line, float(np.median(e.widths)))["evidence_score"]
            if score is not None:
                values.append(score)
                weights.append(line.length)
    result["mean_image_evidence_score"] = float(np.average(values, weights=weights)) if weights else None
    result["evidence_scored_length_m"] = sum(weights)
    return result


def save_result(directory, edges, audit, changes, baseline, evidence, roi, crs, config, title):
    directory.mkdir(parents=True, exist_ok=False)
    save_lines(directory / "centerlines.gpkg", as_frame(edges, crs), roi)
    write_json(directory / "audit.json", audit)
    # JSON also carries metric CRS explicitly (do not mislabel projected data as RFC7946 GeoJSON).
    write_json(directory / "edits.json", dict(crs=str(crs), features=[
        dict(candidate_id=c["candidate_id"], decision=c["decision"], geometry=mapping(c["geometry"])) for c in changes]))
    result = metrics(edges, changes, baseline, evidence, roi, config)
    write_json(directory / "metrics.json", result)
    review_plot(directory / "review.png", baseline, edges, changes, audit, evidence, roi, title)
    return result
