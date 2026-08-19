from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "user_pipeline.py"
DEFAULT_CHECKPOINT = ROOT / "models" / "samroad" / "samroad.ckpt"
DEFAULT_CONFIG = ROOT / "config" / "samroad_inference.yaml"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def extract_image(image: Path, workspace: Path, args) -> Path:
    run([sys.executable, str(PIPELINE), "prepare", "--source", str(image), "--workspace", str(workspace)])
    run([
        sys.executable, str(PIPELINE), "extract", "--workspace", str(workspace),
        "--checkpoint", str(args.checkpoint), "--config", str(args.config),
        "--device", args.device, "--run-id", "cross_sensor",
        "--resume",
    ])
    return workspace / "latest_result.json"


def frame_stats(result: dict) -> tuple[float, float]:
    centerlines = gpd.read_file(result["centerlines"])
    surfaces = gpd.read_file(result["surfaces"])
    return float(centerlines.geometry.length.sum()), float(surfaces.geometry.area.sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a real extraction-to-change cross-sensor pair test.")
    parser.add_argument("--before-image", default="")
    parser.add_argument("--after-image", default="")
    parser.add_argument("--before-result", default="")
    parser.add_argument("--after-result", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--truth", default="")
    parser.add_argument("--validation-area", default="")
    parser.add_argument("--label", default="cross_sensor_pair")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    before_result = Path(args.before_result).expanduser().resolve() if args.before_result else None
    after_result = Path(args.after_result).expanduser().resolve() if args.after_result else None
    if before_result is None:
        if not args.before_image:
            parser.error("Provide --before-result or --before-image.")
        before_result = extract_image(Path(args.before_image).resolve(), output / "before_workspace", args)
    if after_result is None:
        if not args.after_image:
            parser.error("Provide --after-result or --after-image.")
        after_result = extract_image(Path(args.after_image).resolve(), output / "after_workspace", args)
    change_output = output / "change"
    command = [
        sys.executable, str(PIPELINE), "change", "--before-result", str(before_result),
        "--after-result", str(after_result), "--output", str(change_output),
        "--before-period", "A", "--after-period", "B",
    ]
    if args.truth:
        command.extend(["--truth", str(Path(args.truth).resolve())])
    if args.validation_area:
        command.extend(["--validation-area", str(Path(args.validation_area).resolve())])
    run(command)
    before_payload = json.loads(before_result.read_text(encoding="utf-8"))
    after_payload = json.loads(after_result.read_text(encoding="utf-8"))
    summary = json.loads((change_output / "change_summary.json").read_text(encoding="utf-8"))
    before_length, before_area = frame_stats(before_payload)
    after_length, after_area = frame_stats(after_payload)
    report = {
        "label": args.label,
        "before_centerline_length": before_length,
        "after_centerline_length": after_length,
        "before_surface_area": before_area,
        "after_surface_area": after_area,
        "raw_unmatched_length": summary.get("raw_unmatched_length", 0),
        "geometry_only_added_candidates": summary.get("geometry_only_added_candidates", 0),
        "geometry_only_removed_candidates": summary.get("geometry_only_removed_candidates", 0),
        "auto_added_count": summary.get("added_feature_count", 0),
        "auto_removed_count": summary.get("removed_feature_count", 0),
        "review_added_count": summary.get("review_added_feature_count", 0),
        "review_removed_count": summary.get("review_removed_feature_count", 0),
        "suppressed_extraction_disagreement_count": summary.get("suppressed_extraction_disagreement_count", 0),
        "present_by_probability_count": summary.get("present_by_probability_count", 0),
        "present_by_surface_count": summary.get("present_by_surface_count", 0),
        "uncertain_count": summary.get("uncertain_count", 0),
        "confirmed_absent_count": summary.get("confirmed_absent_count", 0),
        "before_probability_available": summary.get("before_probability_available", False),
        "after_probability_available": summary.get("after_probability_available", False),
    }
    (output / "cross_sensor_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = ["# Cross-sensor Audit Report", "", f"Test: `{args.label}`", "", "| Metric | Value |", "|---|---:|"]
    rows.extend(f"| {key} | {value} |" for key, value in report.items() if key != "label")
    rows.extend([
        "", "Interpretation: review/suppressed candidates are extraction disagreements, not official changes.",
        "Only evidence-confirmed auto Added/Removed enter the formal products.",
    ])
    (output / "cross_sensor_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    shutil.copy2(change_output / "change_preview.png", output / "cross_sensor_change_preview.png")
    shutil.copy2(change_output / "sensor_disagreement_preview.png", output / "sensor_disagreement_preview.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
