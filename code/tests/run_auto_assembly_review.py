"""Assemble saved Auto seeds, then render network continuity and reference comparison."""
import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import intersection, make_valid, union_all
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.auto_change_assembly import assemble_change_objects, polygonal, write_assembly_audit
from engine.fast_auto_change import complete_written_auto_result
from engine.fast_pipeline import _write_fast_public_changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.seed_dir/"input_provenance.json").read_text(encoding="utf-8"))
    seeds = gpd.read_file(args.seed_dir/"auto_diagnostics.gpkg", layer="changes")
    metric = seeds.estimate_utm_crs() if seeds.crs.is_geographic else seeds.crs
    seeds = seeds.to_crs(metric)
    seeds.geometry = seeds.geometry.map(polygonal)
    # Use saved accepted width axes to locate cached paired ribbons on the road.
    widths = gpd.read_file(args.seed_dir/"auto_diagnostics.gpkg", layer="width_candidates").to_crs(metric)
    accepted = widths.accepted.astype(str).str.lower().isin(("true", "1"))
    widths = widths.loc[accepted]
    seeds["axis_wkt"] = ""
    for kind, sign in (("widened", 1), ("narrowed", -1)):
        remaining = widths.loc[widths.sign == sign].copy()
        for index, row in seeds.loc[seeds.change_typ == kind].iterrows():
            if remaining.empty:
                break
            scores = remaining.geometry.centroid.distance(row.geometry.centroid)
            best = scores.idxmin()
            if scores.loc[best] < 10:
                seeds.at[index, "axis_wkt"] = remaining.loc[best].geometry.wkt
                remaining = remaining.drop(best)
    before = gpd.read_file(source["before"]["centerlines"])
    after = gpd.read_file(source["after"]["centerlines"])
    objects, artifacts = assemble_change_objects(seeds, before, after)
    write_assembly_audit(output, seeds, artifacts)
    funnel = json.loads((args.seed_dir/"candidate_funnel.json").read_text(encoding="utf-8"))
    funnel["local_detection_final"] = funnel["final"]
    funnel["network_assembly"] = artifacts["summary"]
    funnel["final"] = {f"final_{kind}": row["output"] for kind, row in artifacts["summary"]["by_type"].items()}
    funnel["count_units"] += "; assembled final: network objects (all local seeds retained)"
    for kind in ("added", "removed"):
        funnel[kind]["local_detected_runs"] = funnel[kind]["final_auto_count"]
        funnel[kind]["final_auto_count"] = artifacts["summary"]["by_type"][kind]["output"]
    (output/"candidate_funnel.json").write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    objects.to_file(output/"auto_diagnostics.gpkg", layer="changes", driver="GPKG")
    _write_fast_public_changes(objects, output)
    names = {"added": "added_roads.shp", "removed": "removed_roads.shp", "widened": "widened_road_parts.shp", "narrowed": "narrowed_road_parts.shp"}
    for kind, name in names.items():
        objects.loc[objects.change_typ == kind].to_file(output/name, encoding="UTF-8")
    source["assembly_inputs"] = str((args.seed_dir/"auto_diagnostics.gpkg").resolve())
    source["ground_truth_used_for_assembly"] = False
    (output/"input_provenance.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    result = complete_written_auto_result(output, before_period=source["before"]["period"],
                                          after_period=source["after"]["period"], elapsed_seconds=time.perf_counter()-started)
    (output/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    membership = artifacts["membership"]
    lookup = objects.set_index("object_id")
    lost_area = sum(seeds.geometry.iloc[r.seed_id].difference(lookup.loc[r.object_id].geometry).area for r in membership.itertuples())
    verification = {"seeds": len(seeds), "membership_rows": len(membership),
                    "each_seed_once": len(membership)==membership.seed_id.nunique()==len(seeds),
                    "lost_seed_area_m2": lost_area, "all_objects_valid": bool(objects.is_valid.all())}
    (output/"assembly_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(artifacts["summary"], indent=2), flush=True)
    print(json.dumps(verification, indent=2), flush=True)

    # The algorithm and object files are complete before this optional read.
    truth = None
    evaluation = {"usage": "post_assembly_reference_only", "reference": str(args.reference) if args.reference else None}
    if args.reference:
        truth = gpd.read_file(args.reference).to_crs(metric)
        rows = []
        for i, row in truth.iterrows():
            kind = {"2": "added", "3": "width_changed", "4": "removed"}.get(str(row.get("BHBM", "")))
            kinds = ["widened", "narrowed"] if kind == "width_changed" else [kind]
            target = polygonal(row.geometry)
            local = union_all(seeds.loc[seeds.change_typ.isin(kinds)].geometry.values)
            assembled = union_all(objects.loc[objects.change_typ.isin(kinds)].geometry.values)
            rows.append(dict(reference_id=int(i), change_type=kind,
                             seed_reference_area_coverage=target.intersection(local).area/max(target.area, 1e-9),
                             assembled_reference_area_coverage=target.intersection(assembled).area/max(target.area, 1e-9)))
        evaluation["objects"] = rows
        evaluation["note"] = "Reference polygon area coverage; not classification accuracy or recall of all roads."
    (output/"reference_comparison.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    render(output, args.seed_dir, source, seeds, objects, artifacts, before.to_crs(metric), after.to_crs(metric), truth, evaluation)


def render(output, seed_dir, source, seeds, objects, artifacts, before, after, truth, evaluation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.merge import merge
    from PIL import Image
    colors = {"added": "#00ed85", "removed": "#ff4368", "widened": "#00d7ff", "narrowed": "#ffbe25"}
    datasets = [rasterio.open(p) for p in source["after"]["imagery_tiles"]]
    image_crs = datasets[0].crs
    frames = {"seeds": seeds.to_crs(image_crs), "objects": objects.to_crs(image_crs),
              "bridges": artifacts["assembly_bridges"].to_crs(image_crs), "roads": after.to_crs(image_crs)}
    reference = truth.to_crs(image_crs) if truth is not None else None
    review = output/"review"; review.mkdir(exist_ok=True)
    def subset(frame, extent):
        return frame.iloc[frame.sindex.query(box(*extent), predicate="intersects")]
    def background(extent, pixels=1800):
        res = max((extent[2]-extent[0])/pixels, (extent[3]-extent[1])/pixels)
        array, _ = merge(datasets, bounds=extent, res=res, indexes=[1, 2, 3], resampling=rasterio.enums.Resampling.bilinear)
        rgb = np.moveaxis(array, 0, -1)
        if rgb.dtype != np.uint8:
            lo, hi = np.percentile(rgb[rgb > 0], [1, 99]); rgb = np.clip((rgb-lo)/max(hi-lo, 1)*255, 0, 255).astype("uint8")
        return rgb
    def draw(ax, extent, image, mode, title):
        ax.imshow(image, extent=(extent[0], extent[2], extent[1], extent[3]))
        selected = subset(frames[mode], extent)
        for kind, color in colors.items():
            rows = selected.loc[selected.change_typ == kind]
            if len(rows):
                rows.plot(ax=ax, color=color, edgecolor=color, linewidth=.4, alpha=.8)
        if mode == "objects":
            bridges = subset(frames["bridges"], extent)
            if len(bridges):
                bridges.boundary.plot(ax=ax, color="#f968ff", linewidth=1.4)
        if mode == "bridges" and reference is not None:
            rows = subset(reference, extent)
            if len(rows):
                rows.boundary.plot(ax=ax, color="white", linewidth=1.5, linestyle="--")
        ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3]); ax.set_axis_off()
        ax.set_aspect(1/np.cos(np.radians((extent[1]+extent[3])/2)) if image_crs.is_geographic else 1)
        ax.set_title(title, fontsize=11)
    def panels(extent, name, title, full=False):
        image = background(extent, 2200 if full else 1100)
        fig, axes = plt.subplots(1, 3, figsize=(20, 9 if full else 6))
        draw(axes[0], extent, image, "seeds", "Local detected seeds")
        draw(axes[1], extent, image, "objects", "Network objects | magenta: filled connections")
        draw(axes[2], extent, image, "bridges", "New connections | white dashed: reference only")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(review/f"{name}.png", dpi=160); plt.close(fig)
        thumb = Image.open(review/f"{name}.png"); thumb.thumbnail((1700, 850)); thumb.convert("RGB").save(review/f"{name}.jpg", quality=90)
    extent = tuple(frames["roads"].total_bounds)
    panels(extent, "full_network_assembly", "Auto network assembly: green added / red removed / cyan widened / yellow narrowed", True)
    crops = []
    for kind in colors:
        rows = objects.loc[(objects.change_typ == kind) & (objects.seed_count > 1)].sort_values(["junction_count", "seed_count", "length_m"], ascending=False)
        if rows.empty:
            crops.append(dict(kind=kind, status="No multi-seed object assembled")); continue
        for rank, (_, row) in enumerate(rows.head(2).iterrows(), 1):
            bounds = box(*row.geometry.bounds).buffer(40).bounds
            extent = tuple(gpd.GeoSeries([box(*bounds)], crs=objects.crs).to_crs(image_crs).total_bounds)
            name = f"{kind}_object_{rank}"
            panels(extent, name, f"{row.object_id}: {kind}, {row.seed_count} seeds, {row.bridge_count} links, {row.junction_count} junction traversals")
            crops.append(dict(kind=kind, object_id=row.object_id, seeds=int(row.seed_count), file=f"{name}.png"))
    (review/"crop_index.json").write_text(json.dumps(crops, indent=2), encoding="utf-8")
    reference_crops = []
    if truth is not None:
        for index, row in truth.iterrows():
            bounds = box(*row.geometry.bounds).buffer(50).bounds
            extent = tuple(gpd.GeoSeries([box(*bounds)], crs=truth.crs).to_crs(image_crs).total_bounds)
            name = f"reference_{index}"
            panels(extent, name, f"Reference {index} | shown only after network assembly")
            reference_crops.append(name)
    report = ["# Auto 网络级变化对象组装", "仅组装既有局部结果；未重判存在性，未使用真值生成连接。所有原始变化段保留。",
              "绿色新增、红色消失、青色变宽、黄色变窄。粉色为补入连接边界，白色虚线为事后参考真值。",
              "![整图](full_network_assembly.png)", "## 组装统计", "```json", json.dumps(artifacts["summary"], ensure_ascii=False, indent=2), "```"]
    for crop in crops:
        report += [f"## {crop['kind']} — {crop.get('object_id', crop.get('status'))}"]
        if "file" in crop:
            report.append(f"![完整变化对象]({crop['file']})")
    report += ["## 参考真值对照", "先完成组装并写出结果，再读取参考真值。面积覆盖率仅用于对照；本轮不依据其调参。",
               "```json", json.dumps(evaluation, ensure_ascii=False, indent=2), "```",
               "## 文件", "- ../network_assembly.gpkg：原始段、组装对象、网络轴和补入面；对象带 object_id / seed_ids。",
               "- ../assembly_membership.csv：每个原始段到对象的唯一归属。",
               "- ../assembly_decisions.csv：候选连接的接受和拒绝原因。",
               "- ../assembly_verification.json：原始段保留面积与几何检查。",
               "本轮允许已有伪变化保留。未宣称新增检测召回率提高；本次增加的是既有变化轨迹之间的网络连接。"]
    for name in reference_crops:
        report += [f"### {name}", f"![真值事后对照]({name}.png)"]
    (review/"README.md").write_text("\n\n".join(report), encoding="utf-8")
    for dataset in datasets: dataset.close()
    print(str(review/"README.md"), flush=True)


if __name__ == "__main__":
    main()
