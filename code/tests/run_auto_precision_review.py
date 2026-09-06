"""Replay Auto qualification/assembly from unchanged cached observation evidence.

No truth file is read. The two fixed crop extents locate the screenshots supplied
by the user; they are used only for reporting after the vector results are written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import make_valid, union_all
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.fast_auto_change import RoadScene, finalize_auto_candidates


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.baseline/"input_provenance.json").read_text(encoding="utf-8"))
    gpkg = args.baseline/"auto_diagnostics.gpkg"
    raw = gpd.read_file(gpkg, layer="local_seeds")
    metric = raw.crs
    if not args.render_only:
        evidence = gpd.read_file(gpkg, layer="existence_candidates").to_crs(metric)
        widths = gpd.read_file(gpkg, layer="width_candidates").to_crs(metric)
        intervals = gpd.read_file(gpkg, layer="presence_intervals").to_crs(metric)
        roads = [gpd.read_file(source[p]["centerlines"]) for p in ("before", "after")]
        scenes = {}
        for p, lines in zip(("before", "after"), roads):
            surface, width, valid = [gpd.read_file(source[p][key]).to_crs(metric)
                                     for key in ("surfaces", "width_segments", "valid_observation")]
            scenes[p] = RoadScene(lines.to_crs(metric), surface, width, valid, None, metric)
        previous = json.loads((args.baseline/"candidate_funnel.json").read_text(encoding="utf-8"))
        counts = {**previous["road_matching"], **previous["width"]}
        for kind in ("added", "removed"):
            counts.update({f"{kind}_{k}": v for k, v in previous[kind]["longitudinal"].items()})
        result = finalize_auto_candidates(raw.to_dict("records"), evidence.to_dict("records"), widths.to_dict("records"), counts,
                    presence_audit=intervals.to_dict("records"), scenes=scenes, centerlines=roads, output_dir=output,
                    before_period=source["before"]["period"], after_period=source["after"]["period"],
                    elapsed_seconds=time.perf_counter()-started)
        write_json(output/"result.json", result)
        source.update(ground_truth_used=False, observation_evidence_reused=str(gpkg.resolve()),
                      baseline_result=str(args.baseline.resolve()))
        write_json(output/"input_provenance.json", source)
    old_objects = gpd.read_file(args.baseline/"network_assembly.gpkg", layer="change_objects").to_crs(metric)
    objects = gpd.read_file(output/"network_assembly.gpkg", layer="change_objects").to_crs(metric)
    seeds = gpd.read_file(output/"auto_diagnostics.gpkg", layer="local_seeds")
    audit = gpd.read_file(output/"auto_diagnostics.gpkg", layer="candidate_audit")
    pending = audit.loc[audit.publication_state == "review"]
    old_bridges = gpd.read_file(args.baseline/"network_assembly.gpkg", layer="assembly_bridges").to_crs(metric)
    bridges = gpd.read_file(output/"network_assembly.gpkg", layer="assembly_bridges").to_crs(metric)
    report = {"by_type": {kind: {"previous_objects": int((old_objects.change_typ == kind).sum()),
                               "raw_candidates": int((raw.change_typ == kind).sum()),
                               "accepted_seeds": int((seeds.change_typ == kind).sum()),
                               "review_candidates": int((pending.change_typ == kind).sum()),
                               "new_objects": int((objects.change_typ == kind).sum())}
                          for kind in ("added", "removed", "widened", "narrowed")},
              "bridges_before": len(old_bridges), "bridges_after": len(bridges),
              "bridge_length_before_m": float(old_bridges.length_m.sum()),
              "bridge_length_after_m": float(bridges.length_m.sum()) if len(bridges) else 0.,
              "all_objects_valid": bool(objects.is_valid.all()),
              "all_candidates_accounted_for": len(seeds)+len(pending) == len(raw),
              "width_seed_difference_m2": {kind: make_valid(union_all(raw.loc[raw.change_typ == kind].geometry)).symmetric_difference(
                  make_valid(union_all(seeds.loc[seeds.change_typ == kind].geometry))).area for kind in ("widened", "narrowed")},
              "screenshot_candidate_audit": audit.loc[audit.candidate_id.isin([315, 322, 162]),
                       ["candidate_id", "change_typ", "length_m", "publication_state", "precision_reason",
                        "source_support_ratio", "opposite_absent_ratio", "opposing_corridor_ratio"]].to_dict("records")}
    write_json(output/"precision_comparison.json", report)
    for name, frame in (("previous_objects", old_objects), ("accepted_seeds", seeds), ("changes", objects), ("review_candidates", pending),
                        ("previous_bridges", old_bridges), ("assembly_bridges", bridges)):
        frame.to_file(output/"precision_comparison.gpkg", layer=name, driver="GPKG")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    render(output, source, old_objects, objects, pending, bridges, raw, report)


def render(output, source, old, new, pending, bridges, raw, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.merge import merge
    from PIL import Image
    review = output/"review"
    review.mkdir(exist_ok=True)
    datasets = {p: [rasterio.open(path) for path in source[p]["imagery_tiles"]] for p in ("before", "after")}
    crs = datasets["after"][0].crs
    before_roads = gpd.read_file(source["before"]["centerlines"]).to_crs(crs)
    frames = {"before": before_roads, "previous": old.to_crs(crs), "new": new.to_crs(crs), "review": pending.to_crs(crs)}
    links = bridges.to_crs(crs)
    colors = {"added": "#00ed85", "removed": "#ff4368", "widened": "#00d7ff", "narrowed": "#ffbe25"}
    def panels(metric_extent, name, title):
        extent = gpd.GeoSeries([box(*metric_extent)], crs=raw.crs).to_crs(crs).total_bounds
        backgrounds = {}
        for period in datasets:
            res = max(extent[2]-extent[0], extent[3]-extent[1])/1600
            array, _ = merge(datasets[period], bounds=tuple(extent), res=res, indexes=[1, 2, 3], resampling=rasterio.enums.Resampling.bilinear)
            rgb = np.moveaxis(array, 0, -1)
            if rgb.dtype != np.uint8:
                lo, hi = np.percentile(rgb[rgb > 0], [1, 99]); rgb = np.clip((rgb-lo)/max(hi-lo, 1)*255, 0, 255).astype("uint8")
            backgrounds[period] = rgb
        fig, axes = plt.subplots(1, 4, figsize=(24, 7))
        titles = ("Before image + Before final axes", "After image + Previous Auto", "After image + Revised Auto", "Review only (not formal changes)")
        for ax, (key, frame), label in zip(axes, frames.items(), titles):
            ax.imshow(backgrounds["before" if key == "before" else "after"], extent=(extent[0], extent[2], extent[1], extent[3]))
            selected = frame.iloc[frame.sindex.query(box(*extent), predicate="intersects")]
            if key == "before":
                if len(selected): selected.plot(ax=ax, color="#00d7ff", linewidth=.8)
            elif key == "review":
                if len(selected): selected.boundary.plot(ax=ax, color="#f5d6a0", linewidth=.8, linestyle="--")
            else:
                for kind, color in colors.items():
                    rows = selected.loc[selected.change_typ == kind]
                    if len(rows): rows.plot(ax=ax, color=color, edgecolor=color, linewidth=.35, alpha=.85)
                if key == "new" and len(links):
                    local = links.iloc[links.sindex.query(box(*extent), predicate="intersects")]
                    if len(local): local.boundary.plot(ax=ax, color="#f968ff", linewidth=1.)
            ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3]); ax.set_axis_off()
            ax.set_aspect(1/np.cos(np.radians((extent[1]+extent[3])/2)) if crs.is_geographic else 1)
            ax.set_title(label, fontsize=11)
        fig.suptitle(title); fig.tight_layout(); fig.savefig(review/f"{name}.png", dpi=140); plt.close(fig)
        thumb = Image.open(review/f"{name}.png"); thumb.thumbnail((1900, 850)); thumb.convert("RGB").save(review/f"{name}.jpg", quality=92)
    panels((260921.4456, 2614719.7292, 261087.6117, 2614940.7751), "screenshot_1", "Screenshot 1: Auto candidate evidence review (extraction is checked separately)")
    panels(box(*raw.geometry.iloc[162].bounds).buffer(45).bounds, "screenshot_2", "Screenshot 2: conflicting parallel added / removed candidates")
    panels(tuple(old.total_bounds), "overview", "Green added / red removed / cyan widened / yellow narrowed | magenta: new bridges")
    rows = ["# 全图 Auto 候选审核对比（固定提取输入）", "",
            "复用上一轮相同的局部候选、4 m 观测证据和最终道路产品，重跑正式候选审核与网络组装。未读取参考真值。", "",
            "正式结果要求源期路面与概率共同支持、两期观测有效及对期持续负证据。", "",
            "近邻并行 added/removed 轨迹、证据不足和短小候选进入待审层；这不等于认定它们全部是假变化。", "",
            "| 类型 | 旧正式对象 | 原始候选 | 新正式 seed | 待审候选 | 新正式对象 |",
            "|---|---:|---:|---:|---:|---:|"]
    for kind, r in report["by_type"].items():
        rows.append(f"| {kind} | {r['previous_objects']} | {r['raw_candidates']} | {r['accepted_seeds']} | {r['review_candidates']} | {r['new_objects']} |")
    rows += ["", f"补入连接：{report['bridges_before']} → {report['bridges_after']}；所有原始候选都保留在正式或待审图层中。", "",
             "auto_change_assembly 保持冻结；这里只审核进入组装的局部候选。道路提取后处理保持原样，本轮不处理提取乱连。", "",
             "## 截图问题定位", "", "```json", json.dumps(report["screenshot_candidate_audit"], ensure_ascii=False, indent=2), "```", "",
             "## 图件", "", "![截图一](screenshot_1.png)", "", "![截图二](screenshot_2.png)", "", "![整图](overview.png)", "",
             "## 文件", "", "- ../road_changes.shp：修正后的正式 Auto 变化。",
             "- ../precision_comparison.gpkg：旧结果、新正式 seed／对象、待审候选、旧／新连接。",
             "- ../auto_diagnostics.gpkg：原始候选、逐候选审核原因、观测证据和宽度候选。",
             "- ../candidate_funnel.json：正式／待审分流统计及组装拒绝原因。", "",
             "本组固定输入对照复用两期原最终道路，宽度局部 seed 与组装规则均保持一致。", "",
             "数量下降不等于定量 precision 提高；未进行全区人工逐条判定。"]
    (review/"README.md").write_text("\n".join(rows), encoding="utf-8")
    for files in datasets.values():
        for dataset in files: dataset.close()


if __name__ == "__main__":
    main()
