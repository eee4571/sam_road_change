"""Run truth-free Auto first; only then compare saved seeds and reference polygons."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import make_valid, union_all
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.fast_pipeline import detect_fast_changes


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.baseline/"input_provenance.json").read_text(encoding="utf-8"))
    protected = ["engine/auto_change_assembly.py", "engine/fast_pipeline.py", "user_pipeline.py"]
    hashes = {p: hashlib.sha256((Path(__file__).resolve().parents[1]/p).read_bytes()).hexdigest() for p in protected}
    if not args.review_only:
        result = detect_fast_changes(source["before"], source["after"], output,
                                     before_period=source["before"]["period"], after_period=source["after"]["period"])
        write_json(output/"result.json", result)
        source["ground_truth_used"] = False
        source["baseline_result"] = str(args.baseline.resolve())
        write_json(output/"input_provenance.json", source)
        write_json(output/"frozen_module_hashes.json", hashes)
    review(output, args.baseline, source, args.reference)


def review(output, baseline, source, reference_path):
    (output/"candidate_funnel_before.json").write_bytes((baseline/"candidate_funnel.json").read_bytes())
    old = gpd.read_file(baseline/"network_assembly.gpkg", layer="local_change_seeds")
    metric = old.estimate_utm_crs() if old.crs.is_geographic else old.crs
    frames = {"local_seeds_before": old.to_crs(metric),
              "local_seeds_after": gpd.read_file(output/"auto_diagnostics.gpkg", layer="local_seeds").to_crs(metric),
              "network_changes": gpd.read_file(output/"network_assembly.gpkg", layer="change_objects").to_crs(metric)}
    for name, frame in frames.items():
        frame.to_file(output/"seed_recall_comparison.gpkg", layer=name, driver="GPKG")
    new, objects = frames["local_seeds_after"], frames["network_changes"]
    report = {}
    for kind in ("added", "removed", "widened", "narrowed"):
        report[kind] = {name: {"count": int((frame.change_typ == kind).sum()),
                              "length_m": float(frame.loc[frame.change_typ == kind].length_m.sum())} for name, frame in frames.items()}
        for name, frame in (("seed_qa", new), ("object_qa", objects)):
            selected = frame.loc[frame.change_typ == kind]
            report[kind][name] = {state: int((selected.qa_state == state).sum()) for state in ("confirmed", "probable", "uncertain")}
    membership = __import__("pandas").read_csv(output/"assembly_membership.csv")
    report["all_new_seeds_entered_assembly"] = len(membership) == membership.seed_id.nunique() == len(new)
    lookup = objects.set_index("object_id")
    report["lost_seed_area_m2"] = float(sum(make_valid(new.geometry.iloc[r.seed_id]).difference(lookup.loc[r.object_id].geometry).area
                                           for r in membership.itertuples()))
    report["all_objects_valid"] = bool(objects.is_valid.all())
    # No truth file is opened until the core Auto and all vector outputs finish.
    truth = gpd.read_file(reference_path).to_crs(metric) if reference_path else None
    evaluation = {"usage": "post_auto_only", "objects": []}
    if truth is not None:
        roads = {period: gpd.read_file(source[period]["centerlines"]).to_crs(metric) for period in ("before", "after")}
        for index, row in truth.iterrows():
            kind = {"2": "added", "4": "removed", "3": "width_changed"}.get(str(row.get("BHBM", "")))
            kinds = ["widened", "narrowed"] if kind == "width_changed" else [kind]
            polygon = make_valid(row.geometry)
            values = {}
            for name, frame in frames.items():
                selected = frame.loc[frame.change_typ.isin(kinds)]
                covered = make_valid(union_all(selected.geometry.values)).intersection(polygon).area
                values[name+"_area_coverage"] = float(covered/max(polygon.area, 1e-9))
                values[name+"_intersecting_count"] = int((selected.geometry.intersection(polygon).area > .01).sum())
            values["final_axis_length_inside_reference_m"] = {period: float(frame.geometry.intersection(polygon).length.sum()) for period, frame in roads.items()}
            evaluation["objects"].append(dict(reference_id=int(index), change_type=kind, **values))
    evaluation["note"] = "Area coverage of supplied reference polygons, not global detection recall or precision."
    write_json(output/"seed_recall_summary.json", report)
    write_json(output/"reference_coverage_comparison.json", evaluation)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2), flush=True)
    render(output, source, frames, truth, report, evaluation)


def render(output, source, frames, truth, report, evaluation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.merge import merge
    from PIL import Image
    review_dir = output/"review"
    review_dir.mkdir(exist_ok=True)
    datasets = [rasterio.open(p) for p in source["after"]["imagery_tiles"]]
    image_crs = datasets[0].crs
    projected = {name: frame.to_crs(image_crs) for name, frame in frames.items()}
    projected["final_roads"] = gpd.read_file(source["after"]["centerlines"]).to_crs(image_crs)
    reference = truth.to_crs(image_crs) if truth is not None else None
    colors = {"added": "#00ed85", "removed": "#ff4368", "widened": "#00d7ff", "narrowed": "#ffbe25"}
    def panels(extent, name, title, reference_id=None):
        res = max(extent[2]-extent[0], extent[3]-extent[1])/1600
        array, _ = merge(datasets, bounds=extent, res=res, indexes=[1, 2, 3], resampling=rasterio.enums.Resampling.bilinear)
        rgb = np.moveaxis(array, 0, -1)
        if rgb.dtype != np.uint8:
            lo, hi = np.percentile(rgb[rgb > 0], [1, 99])
            rgb = np.clip((rgb-lo)/max(hi-lo, 1)*255, 0, 255).astype("uint8")
        fig, axes = plt.subplots(1, 4, figsize=(24, 7))
        for ax, (key, frame), label in zip(axes, projected.items(), ("Previous local seeds", "Recall-first local seeds", "Frozen network assembly", "After Final Centerline")):
            ax.imshow(rgb, extent=(extent[0], extent[2], extent[1], extent[3]))
            selected = frame.iloc[frame.sindex.query(box(*extent), predicate="intersects")]
            if key == "final_roads":
                if len(selected):
                    selected.plot(ax=ax, color="#00d7ff", linewidth=1.2)
            else:
                for kind, color in colors.items():
                    rows = selected.loc[selected.change_typ == kind]
                    if len(rows):
                        rows.plot(ax=ax, color=color, edgecolor=color, linewidth=.4, alpha=.8)
            if reference is not None:
                selected_truth = reference.iloc[reference.sindex.query(box(*extent), predicate="intersects")]
                if reference_id is not None:
                    selected_truth = reference.loc[[reference_id]]
                if len(selected_truth):
                    selected_truth.boundary.plot(ax=ax, color="white", linewidth=1.5, linestyle="--")
            ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3]); ax.set_axis_off()
            ax.set_aspect(1/np.cos(np.radians((extent[1]+extent[3])/2)) if image_crs.is_geographic else 1)
            ax.set_title(label)
        fig.suptitle(title); fig.tight_layout(); fig.savefig(review_dir/f"{name}.png", dpi=140); plt.close(fig)
        thumb = Image.open(review_dir/f"{name}.png")
        thumb.thumbnail((1700, 850)); thumb.convert("RGB").save(review_dir/f"{name}.jpg", quality=90)
    panels(tuple(projected["network_changes"].total_bounds), "overview", "Green added / red removed / cyan widened / yellow narrowed | white dashed: post-run reference")
    names = []
    if truth is not None:
        for index, row in truth.iterrows():
            extent = gpd.GeoSeries([box(*row.geometry.bounds).buffer(45)], crs=truth.crs).to_crs(image_crs).total_bounds
            name = f"reference_{index}"
            panels(tuple(extent), name, f"Reference {index}: previous seeds / new seeds / assembled / available final roads", reference_id=index)
            names.append(name)
    removed = frames["local_seeds_after"].loc[lambda f: f.change_typ == "removed"].sort_values("length_m", ascending=False)
    for number, (_, row) in enumerate(removed.head(2).iterrows()):
        extent = gpd.GeoSeries([box(*row.geometry.bounds).buffer(45)], crs=removed.crs).to_crs(image_crs).total_bounds
        name = f"removed_candidate_{number}"
        panels(tuple(extent), name, f"Removed candidate: {row.qa_state}, {row.length_m:.0f} m (not a truth label)")
        names.append(name)
    lines = ["# Fast Auto 新增／灭失 seed 召回对照", "网络组装模块冻结。先生成所有 Auto 成果，再读取参考真值。",
             "绿色新增、红色灭失、青色拓宽、黄色变窄；白色虚线为事后参考真值。",
             "![整图](overview.png)", "## 数量", "| 类型 | 原 seed | 新 seed | 组装对象 | confirmed / probable / uncertain seeds |",
             "|---|---:|---:|---:|---|"]
    for kind in colors:
        r = report[kind]
        lines.append(f"| {kind} | {r['local_seeds_before']['count']} | {r['local_seeds_after']['count']} | {r['network_changes']['count']} | "
                     + " / ".join(str(r['seed_qa'][s]) for s in ("confirmed", "probable", "uncertain"))+" |")
    lines += ["", "## 参考对象面积覆盖", "| 对象 | 原 seed | 新 seed | 组装后 |", "|---|---:|---:|---:|"]
    for r in evaluation["objects"]:
        lines.append(f"| {r['reference_id']} | {r['local_seeds_before_area_coverage']:.1%} | {r['local_seeds_after_area_coverage']:.1%} | {r['network_changes_area_coverage']:.1%} |")
    lines += ["", "这些比例是提供的参考多边形面积覆盖，不代表全区召回率或精度；灭失样例是候选，未取得灭失真值验证。",
              "最后一列为 After Final Centerline。参考面内没有最终道路轴时，本轮沿 Final Road 生成候选的算法无法恢复道路主体。",
              "confidence 是排序用规则分数，不是校准概率。uncertain 和 probable 都已进入网络组装。",
              "公共 SHP 字段不变；完整 QA 字段见 GeoPackage。", "", "## 文件",
              "- ../seed_recall_comparison.gpkg：local_seeds_before / local_seeds_after / network_changes。",
              "- ../candidate_funnel.json / candidate_funnel_before.json：新旧新增／灭失候选漏斗和各 QA 状态数量。",
              "- ../auto_diagnostics.gpkg：4 m 证据、连续候选区间、原始 seeds 及最终对象。",
              "- ../reference_coverage_comparison.json：参考对象 coverage 和各期 Final Axis 覆盖。"]
    for name in names:
        lines += ["", f"## {name}", f"![对照]({name}.png)"]
    text = re.sub(r"(?m)^(#{1,6} .+)$", r"\n\1\n", "\n".join(lines)).strip()+"\n"
    (review_dir/"README.md").write_text(text, encoding="utf-8")
    for dataset in datasets:
        dataset.close()


if __name__ == "__main__":
    main()
