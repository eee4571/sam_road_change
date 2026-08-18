from __future__ import annotations

"""Create deterministic synthetic change truth for exercising the result-stage evaluator."""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import affinity


BHBM_BY_CHANGE_TYPE = {
    "added": 2,
    "widened": 3,
    "narrowed": 3,
    "width_changed": 3,
    "removed": 4,
}


def build_mock_truth(source: Path, output: Path, *, perturb: bool = False) -> dict:
    changes = gpd.read_file(source, layer="road_changes")
    if changes.empty:
        raise ValueError(f"变化结果为空：{source}")
    if "change_typ" not in changes.columns:
        raise ValueError("变化结果缺少 change_typ 字段。")

    metric = changes.to_crs(changes.estimate_utm_crs()) if changes.crs and changes.crs.is_geographic else changes.copy()
    records = []
    omitted = 0
    shifted = 0
    for index, row in metric.reset_index(drop=True).iterrows():
        # Exact copy is the default. Optional perturbation exists only for evaluator tests.
        if perturb and index % 17 == 0:
            omitted += 1
            continue
        geometry = row.geometry
        # Slightly change selected boundaries to exercise area-overlap metrics.
        distance = (0.35, -0.25, 0.15)[index % 3] if perturb else 0.0
        adjusted = geometry.buffer(distance) if distance else geometry
        if adjusted.is_empty:
            adjusted = geometry
        item = {key: row[key] for key in metric.columns if key != "geometry"}
        change_type = str(row["change_typ"])
        item.pop("change_typ", None)
        item["BHBM"] = BHBM_BY_CHANGE_TYPE[change_type]
        item["mock_src"] = "prediction_adjusted"
        item["geometry"] = adjusted
        records.append(item)

        # Add occasional displaced copies, acting as truth changes missed by detection.
        if perturb and index % 31 == 7:
            extra = dict(item)
            extra["mock_src"] = "synthetic_missed"
            extra["geometry"] = affinity.translate(adjusted, xoff=18.0, yoff=-14.0)
            records.append(extra)
            shifted += 1

    truth = gpd.GeoDataFrame(records, geometry="geometry", crs=metric.crs)
    if changes.crs is not None and truth.crs != changes.crs:
        truth = truth.to_crs(changes.crs)
    output.parent.mkdir(parents=True, exist_ok=True)
    truth.to_file(output, driver="ESRI Shapefile", encoding="UTF-8")
    metadata = {
        "synthetic": True,
        "warning": "仅用于测试精度评价流程，不可作为真实精度结论或人工标注。",
        "generation_mode": "perturbed_evaluator_test" if perturb else "exact_copy_of_detection",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_feature_count": int(len(changes)),
        "truth_feature_count": int(len(truth)),
        "omitted_prediction_count": omitted,
        "synthetic_missed_count": shifted,
        "change_type_field": "BHBM",
        "change_type_codes": {"2": "新增", "3": "变化（拓宽和变窄合并）", "4": "灭失"},
        "classes": sorted(int(value) for value in truth["BHBM"].dropna().unique()),
    }
    output.with_suffix(".mock_truth.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="根据变化检测成果生成可复现的模拟真值 SHP")
    parser.add_argument("--source", required=True, help="road_changes.gpkg")
    parser.add_argument("--output", required=True, help="模拟真值 .shp")
    parser.add_argument(
        "--perturb-for-testing", action="store_true",
        help="显式制造删减、边界扰动和平移漏检；默认严格复制变化结果",
    )
    args = parser.parse_args()
    metadata = build_mock_truth(
        Path(args.source), Path(args.output), perturb=bool(args.perturb_for_testing),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
