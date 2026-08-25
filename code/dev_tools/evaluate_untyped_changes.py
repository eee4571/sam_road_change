from __future__ import annotations

"""Evaluate change recall and matched centerline offset without truth classes."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import geopandas as gpd


CODE_ROOT = Path(__file__).resolve().parents[1]
WIDTH_ROOT = CODE_ROOT / "engine" / "width"
for import_root in (CODE_ROOT, WIDTH_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.width import road_change_detection as change_evaluation  # noqa: E402


FAST_TRUTH_TYPE_FIELD = "BHBM"
UNTYPED_CHANGE_CODE = 3


def _load_layer(path: Path, label: str) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise ValueError(f"{label}不存在：{path}")
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"{label}没有 CRS：{path}")
    return frame


def prepare_fast_truth_root(truth_root: Path, output_dir: Path) -> dict:
    """Copy untyped change truth to a Fast-compatible truth directory.

    Untyped change features can only be represented honestly as the generic
    ``BHBM=3`` (change) class.  The source shapefile set is never overwritten.
    """
    truth_root = truth_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not truth_root.is_dir():
        raise ValueError(f"真值目录不存在：{truth_root}")
    if output_dir == truth_root:
        raise ValueError("新真值目录不能与原始真值目录相同。")

    truth_paths = sorted(truth_root.glob("*.shp"))
    if not truth_paths:
        raise ValueError(f"真值目录中没有 SHP：{truth_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for truth_path in truth_paths:
        output_path = output_dir / truth_path.name
        if output_path.exists():
            raise ValueError(f"新真值已存在，不会覆盖：{output_path}")

        truth = _load_layer(truth_path, f"{truth_path.stem} 变化真值").copy()
        existing_field = next(
            (column for column in truth.columns if column.casefold() == FAST_TRUTH_TYPE_FIELD.casefold()),
            None,
        )
        if existing_field and existing_field != FAST_TRUTH_TYPE_FIELD:
            truth = truth.rename(columns={existing_field: FAST_TRUTH_TYPE_FIELD})
        truth[FAST_TRUTH_TYPE_FIELD] = UNTYPED_CHANGE_CODE
        truth.to_file(output_path, driver="ESRI Shapefile", encoding="UTF-8")

        written = _load_layer(output_path, f"{truth_path.stem} 新真值")
        values = sorted(int(value) for value in written[FAST_TRUTH_TYPE_FIELD].dropna().unique())
        if len(written) != len(truth) or values != [UNTYPED_CHANGE_CODE]:
            raise ValueError(f"新真值写出校验失败：{output_path}")
        results.append({
            "source": str(truth_path),
            "output": str(output_path),
            "feature_count": int(len(written)),
            "crs": str(written.crs),
            "type_field": FAST_TRUTH_TYPE_FIELD,
            "type_values": values,
        })

    return {
        "mode": "prepare_fast_truth",
        "source_root": str(truth_root),
        "output_root": str(output_dir),
        "classification": "无类型变化统一标记为 BHBM=3（变化）",
        "files": results,
    }


def _metric_frames(
    predicted: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    validation_area: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, str]:
    predicted = change_evaluation._clean_geometries(predicted)
    truth = change_evaluation._clean_geometries(truth)
    predicted, truth, metric_crs, _output_crs = change_evaluation._analysis_crs(predicted, truth)
    validation = change_evaluation._polygon_union(
        change_evaluation._clean_geometries(validation_area).to_crs(predicted.crs)
    )
    return (
        change_evaluation._clip_frame(predicted, validation),
        change_evaluation._clip_frame(truth, validation),
        str(metric_crs),
    )


def evaluate_pair(
    pair: str,
    predicted_path: Path,
    truth_path: Path,
    validation_area: gpd.GeoDataFrame,
    tolerance: float,
) -> dict:
    predicted = _load_layer(predicted_path, f"{pair} 预测变化")
    truth = _load_layer(truth_path, f"{pair} 变化真值")

    rows, metadata = change_evaluation.evaluate_changes(
        predicted,
        truth,
        validation_area=validation_area,
        truth_type_field="",
        evaluation_tolerance=tolerance,
    )
    overall = rows[0]
    predicted_metric, truth_metric, metric_crs = _metric_frames(predicted, truth, validation_area)
    offset = change_evaluation._centerline_offset_metrics(
        predicted_metric,
        truth_metric,
        tolerance,
    )
    return {
        "pair": pair,
        "predicted": str(predicted_path.resolve()),
        "truth": str(truth_path.resolve()),
        "metric_crs": metric_crs,
        "evaluation_tolerance_m": tolerance,
        "predicted_feature_count": int(len(predicted_metric)),
        "truth_feature_count": int(len(truth_metric)),
        "change_area_recall": float(overall["recall"]),
        "covered_truth_m2": float(overall["tp_m2"]),
        "truth_support_m2": float(overall["truth_support_m2"]),
        "predicted_support_m2": float(overall["predicted_support_m2"]),
        "centerline_offset_status": offset["centerline_offset_status"],
        "centerline_offset_reason": offset["centerline_offset_reason"],
        "centerline_avg_offset_m": offset["centerline_avg_offset_m"],
        "truth_to_pred_avg_m": offset["truth_to_pred_avg_m"],
        "pred_to_truth_avg_m": offset["pred_to_truth_avg_m"],
        "truth_axis_length_m": float(offset["truth_axis_length_m"]),
        "predicted_axis_length_m": float(offset["predicted_axis_length_m"]),
        "truth_distance_integral_m2": float(offset["truth_distance_integral_m2"]),
        "predicted_distance_integral_m2": float(offset["predicted_distance_integral_m2"]),
        "matched_truth_feature_count": int(offset["included_truth_feature_count"]),
        "unmatched_truth_feature_count": int(offset["excluded_truth_feature_count"]),
        "area_metric_definition": metadata["headline_metric_definition"]["change_area_recall"],
    }


def _aggregate(results: list[dict]) -> dict:
    covered_truth = sum(item["covered_truth_m2"] for item in results)
    truth_support = sum(item["truth_support_m2"] for item in results)
    truth_length = sum(item["truth_axis_length_m"] for item in results)
    predicted_length = sum(item["predicted_axis_length_m"] for item in results)
    truth_integral = sum(item["truth_distance_integral_m2"] for item in results)
    predicted_integral = sum(item["predicted_distance_integral_m2"] for item in results)
    offset_available = truth_length > 0 and predicted_length > 0
    return {
        "pair": "all",
        "change_area_recall": covered_truth / truth_support if truth_support else None,
        "covered_truth_m2": covered_truth,
        "truth_support_m2": truth_support,
        "predicted_support_m2": sum(item["predicted_support_m2"] for item in results),
        "centerline_offset_status": "computed" if offset_available else "unavailable",
        "centerline_avg_offset_m": (
            (truth_integral + predicted_integral) / (truth_length + predicted_length)
            if offset_available else None
        ),
        "truth_to_pred_avg_m": truth_integral / truth_length if offset_available else None,
        "pred_to_truth_avg_m": predicted_integral / predicted_length if offset_available else None,
        "truth_axis_length_m": truth_length,
        "predicted_axis_length_m": predicted_length,
        "truth_distance_integral_m2": truth_integral,
        "predicted_distance_integral_m2": predicted_integral,
        "matched_truth_feature_count": sum(item["matched_truth_feature_count"] for item in results),
        "unmatched_truth_feature_count": sum(item["unmatched_truth_feature_count"] for item in results),
    }


def _csv_row(item: dict) -> dict:
    recall = item["change_area_recall"]
    return {
        "pair": item["pair"],
        "change_area_recall": recall,
        "change_area_recall_percent": recall * 100 if recall is not None else None,
        "covered_truth_m2": item["covered_truth_m2"],
        "truth_support_m2": item["truth_support_m2"],
        "predicted_support_m2": item["predicted_support_m2"],
        "centerline_offset_status": item["centerline_offset_status"],
        "centerline_avg_offset_m": item["centerline_avg_offset_m"],
        "truth_to_pred_avg_m": item["truth_to_pred_avg_m"],
        "pred_to_truth_avg_m": item["pred_to_truth_avg_m"],
        "matched_truth_feature_count": item["matched_truth_feature_count"],
        "unmatched_truth_feature_count": item["unmatched_truth_feature_count"],
    }


def evaluate_changes_root(
    changes_root: Path,
    truth_root: Path,
    validation_path: Path,
    output_dir: Path,
    *,
    tolerance: float = 3.0,
) -> dict:
    changes_root = changes_root.expanduser().resolve()
    truth_root = truth_root.expanduser().resolve()
    validation_path = validation_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if tolerance <= 0:
        raise ValueError("评价容差必须大于 0。")
    if not changes_root.is_dir():
        raise ValueError(f"变化结果目录不存在：{changes_root}")
    if not truth_root.is_dir():
        raise ValueError(f"真值目录不存在：{truth_root}")
    validation_area = _load_layer(validation_path, "验证区")

    pairs = sorted(
        child.name
        for child in changes_root.iterdir()
        if child.is_dir()
        and (child / "road_changes.shp").is_file()
        and (truth_root / f"{child.name}.shp").is_file()
    )
    if not pairs:
        raise ValueError("没有找到同时具备 road_changes.shp 和同名真值 SHP 的变化对。")

    results = [
        evaluate_pair(
            pair,
            changes_root / pair / "road_changes.shp",
            truth_root / f"{pair}.shp",
            validation_area,
            tolerance,
        )
        for pair in pairs
    ]
    aggregate = _aggregate(results)
    report = {
        "mode": "untyped_change_evaluation",
        "changes_root": str(changes_root),
        "truth_root": str(truth_root),
        "validation_area": str(validation_path),
        "evaluation_tolerance_m": tolerance,
        "definitions": {
            "change_area_recall": "预测变化与真值变化相交面积 / 真值变化总面积。",
            "centerline_avg_offset_m": (
                "仅对容差范围内空间匹配的无类型变化面计算；变化面先形成道路走廊，"
                "提取骨架后计算真值到预测、预测到真值的双向轴长加权平均距离。"
            ),
            "scope_warning": "真值无变化类型，因此该中心线偏移包含所有匹配变化面，不等同于仅新增/灭失口径。",
        },
        "pairs": results,
        "aggregate": aggregate,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "untyped_evaluation.json"
    csv_path = output_dir / "untyped_evaluation.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = [_csv_row(item) for item in results] + [_csv_row(aggregate)]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report["json"] = str(json_path)
    report["csv"] = str(csv_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评价无变化类型真值的查全率和中心线偏移。")
    parser.add_argument("--changes-root", help="包含多个 <前期>_to_<后期> 子目录的 changes 目录")
    parser.add_argument("--truth-root", required=True, help="包含同名变化真值 SHP 的目录")
    parser.add_argument("--validation-area", help="验证区边界 SHP")
    parser.add_argument("--output-dir", help="结果目录；默认 changes/_untyped_evaluation")
    parser.add_argument(
        "--prepare-fast-truth-output",
        help="将无类型真值复制到新目录，并添加 BHBM=3；该模式不执行评价",
    )
    parser.add_argument("--tolerance", type=float, default=3.0, help="空间匹配容差（米，默认 3.0）")
    args = parser.parse_args(argv)

    if args.prepare_fast_truth_output:
        try:
            report = prepare_fast_truth_root(
                Path(args.truth_root),
                Path(args.prepare_fast_truth_output),
            )
        except (OSError, ValueError) as exc:
            parser.exit(1, f"新真值生成失败：{exc}\n")
        for item in report["files"]:
            print(
                f"{Path(item['output']).name}: {item['feature_count']} 个要素, "
                f"{item['type_field']}={item['type_values'][0]}"
            )
        print(f"新真值目录：{report['output_root']}")
        return 0

    if not args.changes_root or not args.validation_area:
        parser.error("评价模式必须提供 --changes-root 和 --validation-area。")
    changes_root = Path(args.changes_root)
    output_dir = Path(args.output_dir) if args.output_dir else changes_root / "_untyped_evaluation"
    try:
        report = evaluate_changes_root(
            changes_root,
            Path(args.truth_root),
            Path(args.validation_area),
            output_dir,
            tolerance=args.tolerance,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"评价失败：{exc}\n")

    for item in report["pairs"] + [report["aggregate"]]:
        recall = item["change_area_recall"]
        recall_text = f"{recall:.2%}" if recall is not None else "--"
        offset = item["centerline_avg_offset_m"]
        offset_text = f"{offset:.3f} m" if offset is not None else "--"
        print(f"{item['pair']}: 查全率={recall_text}, 中心线双向平均偏移={offset_text}")
    print(f"JSON：{report['json']}")
    print(f"CSV：{report['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
