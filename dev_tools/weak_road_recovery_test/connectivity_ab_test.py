from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = TOOL_ROOT / "outputs" / "2"
DEFAULT_OUTPUT_DIR = TOOL_ROOT / "outputs" / "2_connectivity_test"
RUN_TEST = TOOL_ROOT / "run_test.py"
CACHE_FILES = (
    "test_config.json",
    "road_probability.png",
    "original_graph.p",
    "original_edge_scores.csv",
)
CASES = (
    {"case": "A", "directory": "A_existing", "segment_enabled": False},
    {"case": "B", "directory": "B_endpoint_segment", "segment_enabled": True},
)
RESULT_FIELDS = (
    "case", "status", "error", "strong_edge_count", "final_edge_count",
    "added_edge_count", "component_count_before", "component_count_after",
    "endpoint_count_before", "endpoint_count_after", "connectivity_gain_total",
    "weak_recovered_candidate_count", "weak_recovered_edge_count",
    "weak_connectivity_gain_total", "endpoint_segment_candidate_count",
    "endpoint_segment_accepted_count", "endpoint_segment_rejected_count",
    "endpoint_segment_recovered_edge_count", "endpoint_segment_connectivity_gain",
    "endpoint_segment_reject_reason_counts", "weak_recovery_seconds", "total_seconds",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run cached endpoint-to-segment connectivity A/B")
    result.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    result.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_case(source_dir: Path, output_dir: Path, case: dict) -> Path:
    case_dir = (output_dir / case["directory"]).resolve()
    if case_dir.parent != output_dir:
        raise ValueError(f"Unsafe case directory: {case_dir}")
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    for name in CACHE_FILES:
        shutil.copy2(source_dir / name, case_dir / name)
    return case_dir


def _command(case_dir: Path, segment_enabled: bool) -> list[str]:
    return [
        sys.executable,
        str(RUN_TEST),
        "--recovery-only",
        "--run-dir", str(case_dir),
        "--threshold-profile", "weak_sensor",
        "--disable-bootstrap",
        "--min-direction-cosine", "0.50",
        "--min-mean-probability", "0.16",
        "--min-q25-probability", "0.12",
        "--min-background-contrast", "0.05",
        "--max-gap", "64",
        "--max-segment-distance", "64",
        "--min-segment-direction-cosine", "0.50",
        "--direction-lookback", "32",
        "--enable-segment-recovery" if segment_enabled else "--disable-segment-recovery",
    ]


def _run_case(source_dir: Path, output_dir: Path, case: dict) -> tuple[dict, Path]:
    row = {name: "" for name in RESULT_FIELDS}
    row.update({"case": case["case"], "status": "failed", "error": ""})
    case_dir = _prepare_case(source_dir, output_dir, case)
    command = _command(case_dir, case["segment_enabled"])
    print(f"[{case['case']}] running {case['directory']}...", flush=True)
    completed = subprocess.run(
        command,
        cwd=str(TOOL_ROOT.parent.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    (case_dir / "connectivity_run.log").write_text(
        "COMMAND\n" + subprocess.list2cmdline(command)
        + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        row["error"] = (completed.stderr or completed.stdout)[-3000:]
        print(f"[{case['case']}] failed", flush=True)
        return row, case_dir
    recovery = json.loads((case_dir / "weak_recovery.json").read_text(encoding="utf-8"))
    summary = recovery.get("summary", recovery)
    timing = json.loads((case_dir / "timing.json").read_text(encoding="utf-8"))
    for name in RESULT_FIELDS:
        if name in {"case", "status", "error", "weak_recovery_seconds", "total_seconds"}:
            continue
        if name == "endpoint_segment_reject_reason_counts":
            row[name] = json.dumps(summary.get(name, {}), ensure_ascii=False, sort_keys=True)
        else:
            row[name] = int(summary.get(name, 0))
    row["weak_recovery_seconds"] = float(timing.get("weak_recovery_seconds", 0.0))
    row["total_seconds"] = float(timing.get("total_seconds", 0.0))
    row["status"] = "ok"
    print(
        f"[{case['case']}] components={row['component_count_before']}→{row['component_count_after']} "
        f"endpoints={row['endpoint_count_before']}→{row['endpoint_count_after']} "
        f"segment_accepted={row['endpoint_segment_accepted_count']}",
        flush=True,
    )
    return row, case_dir


def _write_results(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict]) -> None:
    by_case = {row["case"]: row for row in rows}
    lines = [
        "# Endpoint-to-Segment Connectivity A/B Test",
        "",
        "两组均复用 `outputs/2` 缓存，使用 weak_sensor（HIGH=.24、LOW=.10），Bootstrap 关闭。",
        "固定 endpoint recovery：Direction=.50、Mean=.16、Q25=.12、Contrast=.05、Max Gap=64。",
        "B 仅额外开启 endpoint→segment（Max Distance=64、Direction=.50、Lookback=32）。",
        "",
        "| Case | Strong edges | Components before | Components after | Endpoints before | Endpoints after | Weak endpoint accepted | Segment accepted | Connectivity gain | Added edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['case']} | failed | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {row['case']} | {row['strong_edge_count']} | {row['component_count_before']} | "
            f"{row['component_count_after']} | {row['endpoint_count_before']} | "
            f"{row['endpoint_count_after']} | {row['weak_recovered_candidate_count']} | "
            f"{row['endpoint_segment_accepted_count']} | {row['connectivity_gain_total']} | "
            f"{row['added_edge_count']} |"
        )
    if all(by_case.get(case, {}).get("status") == "ok" for case in ("A", "B")):
        a, b = by_case["A"], by_case["B"]
        component_delta = int(a["component_count_after"]) - int(b["component_count_after"])
        endpoint_delta = int(a["endpoint_count_after"]) - int(b["endpoint_count_after"])
        lines.extend([
            "",
            "## Connectivity effect",
            "",
            f"- 相对 A，B 的最终 connected components 额外减少 **{component_delta}**。",
            f"- 相对 A，B 的 dangling endpoints 额外减少 **{endpoint_delta}**。",
            f"- Endpoint→segment 接受 **{b['endpoint_segment_accepted_count']}** 个，专项 connectivity gain="
            f"**{b['endpoint_segment_connectivity_gain']}**。",
            f"- 候选 {b['endpoint_segment_candidate_count']} 个，拒绝 "
            f"{b['endpoint_segment_rejected_count']} 个；拒绝原因："
            f"`{b['endpoint_segment_reject_reason_counts']}`。",
            f"- 现有 endpoint recovery 接受 {a['weak_recovered_candidate_count']} 个并新增 "
            f"{a['weak_recovered_edge_count']} 条恢复边，但 components/endpoints 均未下降，"
            f"其 connectivity gain={a['weak_connectivity_gain_total']}；这说明 Added edges 不能代替连通性评价。",
        ])
        if int(b["endpoint_segment_accepted_count"]) == 0:
            lines.extend([
                "- 本轮没有 accepted endpoint→segment connector，因此局部 montage 是明确的空结果占位图，"
                "不是可视化生成失败。",
                "- 在不降低 Direction/Mean/Q25/Contrast 的约束下，不应为了获得非零结果强行加入连接。",
            ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "不以连接数量越多越好，也不自动宣称 B 更优。请重点查看 "
        "`endpoint_segment_candidates_montage.png`，核查 T junction 是否真实、是否跨越建筑或非道路区域、"
        "是否误连平行道路。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    source_dir = Path(arguments.source_dir).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("Output directory must differ from source cache")
    missing = [name for name in CACHE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Source cache is missing: {', '.join(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: _sha256(source_dir / name) for name in CACHE_FILES}
    rows = []
    case_dirs = {}
    for case in CASES:
        row, case_dir = _run_case(source_dir, output_dir, case)
        rows.append(row)
        case_dirs[case["case"]] = case_dir
    _write_results(output_dir / "connectivity_results.csv", rows)
    _write_report(output_dir / "connectivity_report.md", rows)
    b_montage = (
        case_dirs["B"] / "endpoint_segment_candidates"
        / "endpoint_segment_candidates_montage.png"
    )
    if b_montage.is_file():
        shutil.copy2(b_montage, output_dir / "endpoint_segment_candidates_montage.png")
    if {name: _sha256(source_dir / name) for name in CACHE_FILES} != source_hashes:
        raise RuntimeError("Source cache changed during connectivity A/B test")
    succeeded = sum(row["status"] == "ok" for row in rows)
    print(f"Connectivity A/B complete: {succeeded}/{len(rows)} cases succeeded", flush=True)
    print(f"Results: {output_dir}", flush=True)
    return 0 if succeeded == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
