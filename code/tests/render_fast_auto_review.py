"""Render an existing no-truth Auto run; selection uses diagnostics, never GT.

Usage: python code/tests/render_fast_auto_review.py OUTPUT_DIR LATEST_PIPELINE_JSON
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge
from shapely.geometry import box

COLORS = {"added": "#00ed85", "removed": "#ff4368", "widened": "#00d7ff", "narrowed": "#ffbe25"}


def main():
    output = Path(sys.argv[1]).resolve()
    manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    result = json.loads((output/"result.json").read_text(encoding="utf-8"))
    periods = [result["before_period"], result["after_period"]]
    sources = {r["period"]: r for r in manifest["period_results"]}
    payloads = [sources[p] for p in periods]
    provenance = json.loads((output/"input_provenance.json").read_text(encoding="utf-8"))
    crs = gpd.read_file(payloads[0]["centerlines"]).crs
    roads = [gpd.read_file(p["centerlines"]).to_crs(crs) for p in payloads]
    surfaces = [gpd.read_file(p["surfaces"]).to_crs(crs) for p in payloads]
    changes = gpd.read_file(result["road_changes"]).to_crs(crs)
    audit_metric = gpd.read_file(result["diagnostics"], layer="existence_candidates")
    audit = audit_metric.to_crs(crs)
    widths_metric = gpd.read_file(result["diagnostics"], layer="width_candidates")
    datasets = []
    for side, p in zip(("before", "after"), payloads):
        summary = json.loads((Path(p["width_review"])/"batch_width_summary.json").read_text(encoding="utf-8"))
        paths = [r["image"] for r in summary["images"]]
        provenance[side]["imagery_tiles"] = paths
        datasets.append([rasterio.open(path) for path in paths])
    (output/"input_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    bounds = np.array([r.total_bounds for r in roads])
    full = (float(bounds[:, 0].min()), float(bounds[:, 1].min()),
            float(bounds[:, 2].max()), float(bounds[:, 3].max()))
    # Separate source layers make the review package usable directly in QGIS.
    for side, p in zip(("before", "after"), payloads):
        for key in ("centerlines", "surfaces", "width_segments", "valid_observation"):
            gpd.read_file(p[key]).to_file(output/"input_layers.gpkg", layer=f"{side}_{key}", driver="GPKG")

    def imagery(side, extent, pixels):
        resolution = max((extent[2]-extent[0])/pixels, (extent[3]-extent[1])/pixels)
        indexes = list(range(1, min(3, datasets[side][0].count)+1))
        array, transform = merge(datasets[side], bounds=extent, res=resolution, indexes=indexes,
                                 resampling=rasterio.enums.Resampling.bilinear)
        rgb = np.moveaxis(array, 0, -1)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        if rgb.dtype != np.uint8:
            positive = rgb[np.any(rgb != 0, axis=2)]
            lo, hi = np.percentile(positive, [1, 99]) if positive.size else (0, 1)
            rgb = np.clip((rgb-lo)/max(hi-lo, 1)*255, 0, 255).astype("uint8")
        return rgb

    def subset(frame, extent):
        return frame.iloc[frame.sindex.query(box(*extent), predicate="intersects")]

    def draw(ax, background, extent, *, side=None, change=False, title=""):
        ax.imshow(background, extent=[extent[0], extent[2], extent[1], extent[3]])
        if side is not None:
            selected = subset(surfaces[side], extent)
            if len(selected):
                selected.plot(ax=ax, color="#4fbdff", alpha=.23, linewidth=0)
            selected = subset(roads[side], extent)
            if len(selected):
                selected.plot(ax=ax, color="#faf4bf", linewidth=.8)
        if change:
            for kind, color in COLORS.items():
                selected = subset(changes.loc[changes.change_typ == kind], extent)
                if len(selected):
                    selected.plot(ax=ax, color=color, edgecolor=color, linewidth=.5, alpha=.78)
        ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
        ax.set_aspect(1/np.cos(np.radians((extent[1]+extent[3])/2)) if crs.is_geographic else 1)
        ax.set_title(title, fontsize=11); ax.set_axis_off()

    preview_dir = output/"review"
    preview_dir.mkdir(exist_ok=True)
    full_images = [imagery(i, full, 2400) for i in (0, 1)]
    for side in (0, 1):
        label = "before" if side == 0 else "after"
        Image.fromarray(full_images[side]).save(preview_dir/f"{label}_imagery.png")
        fig, ax = plt.subplots(figsize=(12, 12))
        draw(ax, full_images[side], full, side=side, title=f"{periods[side]} Final Centerline + Surface")
        fig.savefig(preview_dir/f"{label}_final.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    for kind in ("all", *COLORS):
        fig, ax = plt.subplots(figsize=(12, 12))
        draw(ax, full_images[1], full, title=f"Auto {periods[0]} to {periods[1]} | {kind}")
        selected = changes if kind == "all" else changes.loc[changes.change_typ == kind]
        for name, color in COLORS.items():
            part = selected.loc[selected.change_typ == name]
            if len(part):
                part.plot(ax=ax, color=color, edgecolor=color, linewidth=.5, alpha=.85)
        ax.set_xlim(full[0], full[2]); ax.set_ylim(full[1], full[3])
        if selected.empty:
            ax.text(.5, .03, "No accepted changes", color="white", ha="center", transform=ax.transAxes,
                    bbox={"facecolor": "black", "alpha": .6})
        fig.savefig(preview_dir/f"change_{kind}.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    draw(axes[0], full_images[0], full, side=0, title=f"Before {periods[0]} | Final roads")
    draw(axes[1], full_images[1], full, side=1, title=f"After {periods[1]} | Final roads")
    draw(axes[2], full_images[1], full, change=True, title="Auto | green added / red removed / cyan widened / yellow narrowed")
    fig.tight_layout(); fig.savefig(preview_dir/"full_comparison.png", dpi=180); plt.close(fig)

    chosen_centers = []
    region_center = np.mean(audit_metric.total_bounds.reshape(2, 2), axis=0)

    def choose(rows, sort=None):
        if rows.empty:
            return None
        if sort and sort in rows:
            ordered = rows.dropna(subset=[sort]).sort_values(sort, ascending=False)
            if len(ordered):
                best = ordered.iloc[0][sort]
                rows = rows.loc[rows[sort] == best]
        centers = np.array([(g.centroid.x, g.centroid.y) for g in rows.geometry])
        distance_to_center = np.linalg.norm(centers-region_center, axis=1)
        if chosen_centers:
            separation = np.min(np.linalg.norm(centers[:, None, :]-np.array(chosen_centers)[None, :, :], axis=2), axis=1)
            score = np.minimum(separation, 900)-.1*distance_to_center
        else:
            score = -distance_to_center
        index = int(np.argmax(score))
        chosen_centers.append(centers[index])
        return rows.iloc[index].geometry

    stable = audit_metric.loc[audit_metric.matched & ~audit_metric.junction & audit_metric.before_valid & audit_metric.after_valid]
    candidates = []
    candidates.append(("stable_road", "Matched road; visual stability requires review", choose(stable)))
    shifted = stable.loc[(stable.offset_m >= 1) & (stable.offset_m <= 3)]
    candidates.append(("spatial_offset", "Matched 1-3 m axis offset", choose(shifted, "offset_m")))
    # A source axis mapping to multiple target axes is a segmentation/topology candidate.
    segmentation = audit_metric.loc[audit_metric.side == "before"].groupby("axis_id").target_axis.nunique()
    candidates.append(("segmentation", "Multiple target axes on one source chain", choose(stable.loc[(stable.side == "before") & stable.axis_id.isin(segmentation[segmentation > 1].index)])))
    metric_roads = roads[0].to_crs(audit_metric.crs)
    parallel_candidates = []
    for _, row in stable.iloc[::max(1, len(stable)//800)].iterrows():
        nearby = metric_roads.iloc[metric_roads.sindex.query(row.geometry.centroid, predicate="dwithin", distance=18)]
        directions = []
        for line in nearby.geometry:
            if line.geom_type == "LineString" and line.length > 30:
                d = np.array(line.coords[-1])-np.array(line.coords[0]); directions.append(d/max(np.linalg.norm(d), 1e-9))
        if len(directions) > 1 and any(abs(np.dot(directions[0], d)) > .95 for d in directions[1:]):
            parallel_candidates.append({"geometry": row.geometry})
    dual = choose(gpd.GeoDataFrame(parallel_candidates, geometry="geometry", crs=audit_metric.crs)) if parallel_candidates else None
    candidates.append(("dual_carriageway", "Nearby parallel axes candidate", dual))
    for kind, side in (("added", "after"), ("removed", "before")):
        rows = audit_metric.loc[(audit_metric.side == side) & ~audit_metric.matched]
        candidates.append((f"{kind}_candidate", "Existence candidate; see diagnostic decision",
                           choose(rows, "continuity_pass" if "continuity_pass" in rows else "existence_pass")))
    for kind, sign in (("widened", 1), ("narrowed", -1)):
        rows = widths_metric.loc[widths_metric.get("sign", 0) == sign] if "sign" in widths_metric else widths_metric.iloc[:0]
        candidates.append((f"{kind}_candidate", "Paired width candidate; accepted/rejected recorded", choose(rows, "accepted")))
    junctions = audit_metric.loc[audit_metric.junction]
    dense_junction = None
    if len(junctions):
        representatives = junctions.iloc[::max(1, len(junctions)//300)]
        dense_junction = max(representatives.geometry, key=lambda g: len(
            junctions.sindex.query(g.centroid, predicate="dwithin", distance=45)))
    candidates.append(("large_junction", "Dense junction guard area", dense_junction))
    candidates.append(("nodata_boundary", "Invalid observation or image boundary", choose(audit_metric.loc[~audit_metric.before_valid | ~audit_metric.after_valid])))
    conflict = audit_metric.loc[((~audit_metric.before_geometry) & ((audit_metric.before_surface >= .55) | audit_metric.before_probability)) |
                               ((~audit_metric.after_geometry) & ((audit_metric.after_surface >= .55) | audit_metric.after_probability))]
    candidates.append(("geometry_probability_conflict", "Missing axis with positive surface/probability", choose(conflict)))
    crop_index = []
    for name, note, geometry in candidates:
        if geometry is None:
            crop_index.append({"category": name, "status": "No candidate found; no fabricated example"})
            continue
        center = geometry.centroid
        extent = tuple(gpd.GeoSeries([box(center.x-120, center.y-120, center.x+120, center.y+120)], crs=audit_metric.crs).to_crs(crs).total_bounds)
        images = [imagery(i, extent, 700) for i in (0, 1)]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
        draw(axes[0], images[0], extent, title=f"Before {periods[0]}")
        draw(axes[1], images[1], extent, title=f"After {periods[1]}")
        draw(axes[2], images[0], extent, side=0, title="Before final roads (pale yellow / blue)")
        draw(axes[3], images[1], extent, side=1, change=True, title="After final roads + Auto changes")
        candidate = gpd.GeoSeries([geometry], crs=audit_metric.crs).to_crs(crs)
        for ax in axes:
            candidate.plot(ax=ax, color="#ff61eb", linewidth=2)
            ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
        fig.suptitle(f"{name} | {note} | magenta: inspected candidate", fontsize=12)
        fig.tight_layout(); fig.savefig(preview_dir/f"crop_{name}.png", dpi=150); plt.close(fig)
        crop_index.append({"category": name, "status": "diagnostic candidate, not ground truth", "note": note,
                           "file": f"crop_{name}.png", "bounds": extent})
        print(f"[Review crop] {name}", flush=True)
    (preview_dir/"crop_index.json").write_text(json.dumps(crop_index, indent=2), encoding="utf-8")
    funnel = json.loads((output/"candidate_funnel.json").read_text(encoding="utf-8"))
    report = [f"# Fast 无真值 Auto：{periods[0]} → {periods[1]}",
              "仅使用两期 Final Centerline、Final Surface、Road Width、Probability、Valid Area。未提供变化真值。",
              "这是一轮候选结果供人工判断方向，未做真值精度评价。裁图类别由几何和证据自动选出，不代表已确认实际变化或实际稳定。",
              "初步目视抽查：仍有疑似假新增和假变宽，包括两期影像形态接近但自动报变化的位置。本轮保留检测判定，不根据裁图调整阈值。候选接受状态优先、空间分散选图；未根据是否看起来正确筛选。",
              "颜色：新增绿、消失红、变宽青、变窄黄；局部图粉色为检查候选。原影像 PNG 为缩览，原始地理影像路径见 input_provenance.json。",
              "![整图](full_comparison.png)", "![完整 Change Overlay](change_all.png)",
              "## Candidate funnel", "```json", json.dumps(funnel, ensure_ascii=False, indent=2), "```",
              "## 局部检查"]
    for item in crop_index:
        report += [f"### {item['category']}", item["status"]]
        if "file" in item:
            report += [f"![{item['category']}]({item['file']})"]
    report += ["## 可复查文件", "- ../auto_diagnostics.gpkg：候选轴、存在性证据、宽度判定及变化面",
               "- ../input_layers.gpkg：两期中心线、道路面、测宽、有效区",
               "- ../existence_candidates.csv：每个约 4 m 候选单元的各项证据",
               "- ../width_candidates.csv：局部宽度候选的接受/拒绝原因",
               "- ../input_provenance.json：全部输入路径与无真值声明"]
    (preview_dir/"README.md").write_text("\n\n".join(report), encoding="utf-8")
    for group in datasets:
        for dataset in group:
            dataset.close()
    print(str(preview_dir/"README.md"))


if __name__ == "__main__":
    main()
