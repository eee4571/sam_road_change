from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = TOOL_ROOT / "outputs" / "2"
DEFAULT_ENDPOINT_OUTPUT_DIR = TOOL_ROOT / "outputs" / "2_sweep"
DEFAULT_PROBABILITY_OUTPUT_DIR = TOOL_ROOT / "outputs" / "2_sweep_probability"
RUN_TEST = TOOL_ROOT / "run_test.py"
CACHE_FILES = (
    "test_config.json",
    "road_probability.png",
    "original_graph.p",
    "original_edge_scores.csv",
)
KNOWN_REJECT_REASONS = (
    "direction_mismatch",
    "mean_probability_low",
    "q25_probability_low",
    "background_contrast_low",
    "no_astar_path",
    "path_ratio_too_large",
    "endpoint_already_used",
)
ENDPOINT_CASES = (
    {
        "case": "A",
        "directory": "A_baseline",
        "label": "A baseline",
        "direction_cosine": 0.65,
        "min_mean_probability": 0.20,
        "min_q25_probability": 0.17,
        "min_background_contrast": 0.08,
        "max_gap": 64.0,
    },
    {
        "case": "B",
        "directory": "B_direction_050",
        "label": "B dir=0.50",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.20,
        "min_q25_probability": 0.17,
        "min_background_contrast": 0.08,
        "max_gap": 64.0,
    },
    {
        "case": "C",
        "directory": "C_direction_040",
        "label": "C dir=0.40",
        "direction_cosine": 0.40,
        "min_mean_probability": 0.20,
        "min_q25_probability": 0.17,
        "min_background_contrast": 0.08,
        "max_gap": 64.0,
    },
    {
        "case": "D",
        "directory": "D_probability_relaxed",
        "label": "D dir=0.50 mean=0.16 q25=0.12",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.16,
        "min_q25_probability": 0.12,
        "min_background_contrast": 0.08,
        "max_gap": 64.0,
    },
    {
        "case": "E",
        "directory": "E_contrast_relaxed",
        "label": "E + contrast=0.06",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.16,
        "min_q25_probability": 0.12,
        "min_background_contrast": 0.06,
        "max_gap": 64.0,
    },
)
PROBABILITY_CASES = (
    {
        "case": "E",
        "directory": "E_reference",
        "label": "E reference: mean=.16 q25=.12 contrast=.06",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.16,
        "min_q25_probability": 0.12,
        "min_background_contrast": 0.06,
        "max_gap": 64.0,
    },
    {
        "case": "F",
        "directory": "F_mean014",
        "label": "F mean=.14",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.14,
        "min_q25_probability": 0.12,
        "min_background_contrast": 0.06,
        "max_gap": 64.0,
    },
    {
        "case": "G",
        "directory": "G_q25010",
        "label": "G q25=.10",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.16,
        "min_q25_probability": 0.10,
        "min_background_contrast": 0.06,
        "max_gap": 64.0,
    },
    {
        "case": "H",
        "directory": "H_mean014_q25010",
        "label": "H mean=.14 q25=.10",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.14,
        "min_q25_probability": 0.10,
        "min_background_contrast": 0.06,
        "max_gap": 64.0,
    },
    {
        "case": "I",
        "directory": "I_contrast005",
        "label": "I contrast=.05",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.16,
        "min_q25_probability": 0.12,
        "min_background_contrast": 0.05,
        "max_gap": 64.0,
    },
    {
        "case": "J",
        "directory": "J_contrast004",
        "label": "J contrast=.04",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.16,
        "min_q25_probability": 0.12,
        "min_background_contrast": 0.04,
        "max_gap": 64.0,
    },
    {
        "case": "K",
        "directory": "K_combined",
        "label": "K mean=.14 q25=.10 contrast=.05",
        "direction_cosine": 0.50,
        "min_mean_probability": 0.14,
        "min_q25_probability": 0.10,
        "min_background_contrast": 0.05,
        "max_gap": 64.0,
    },
)
RESULT_FIELDS = (
    "case",
    "status",
    "error",
    "direction_cosine",
    "min_mean_probability",
    "min_q25_probability",
    "min_background_contrast",
    "max_gap",
    "strong_edge_count",
    "weak_candidate_count",
    "weak_recovered_candidate_count",
    "weak_recovered_edge_count",
    "rejected_weak_candidate_count",
    *KNOWN_REJECT_REASONS,
    "other_reject_count",
    "other_reject_reasons",
    "accept_rate",
    "direction_reject_rate",
    "probability_reject_rate",
    "contrast_reject_rate",
    "astar_reject_rate",
    "main_reject_reason",
    "weak_recovery_seconds",
    "total_seconds",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run a controlled weak endpoint recovery parameter sweep from a cached run"
    )
    result.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    result.add_argument(
        "--preset", choices=("endpoint", "probability"), default="endpoint",
        help="endpoint reproduces A-E; probability runs the focused E-K sweep",
    )
    result.add_argument(
        "--output-dir", default=None,
        help="Defaults to outputs/2_sweep or outputs/2_sweep_probability by preset",
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_directories(source_dir: Path, output_dir: Path) -> None:
    if source_dir == output_dir:
        raise ValueError("Sweep output directory must differ from the baseline cache directory")
    missing = [name for name in CACHE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Baseline cache is missing: {', '.join(missing)}")
    if not RUN_TEST.is_file():
        raise FileNotFoundError(f"Missing recovery runner: {RUN_TEST}")


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


def _command(case_dir: Path, case: dict) -> list[str]:
    return [
        sys.executable,
        str(RUN_TEST),
        "--recovery-only",
        "--run-dir",
        str(case_dir),
        "--threshold-profile",
        "weak_sensor",
        "--disable-bootstrap",
        "--min-direction-cosine",
        str(case["direction_cosine"]),
        "--min-mean-probability",
        str(case["min_mean_probability"]),
        "--min-q25-probability",
        str(case["min_q25_probability"]),
        "--min-background-contrast",
        str(case["min_background_contrast"]),
        "--max-gap",
        str(case["max_gap"]),
    ]


def _base_row(case: dict) -> dict:
    return {
        "case": case["case"],
        "status": "failed",
        "error": "",
        "direction_cosine": case["direction_cosine"],
        "min_mean_probability": case["min_mean_probability"],
        "min_q25_probability": case["min_q25_probability"],
        "min_background_contrast": case["min_background_contrast"],
        "max_gap": case["max_gap"],
        **{name: "" for name in RESULT_FIELDS if name not in {
            "case", "status", "error", "direction_cosine", "min_mean_probability",
            "min_q25_probability", "min_background_contrast", "max_gap",
        }},
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _collect_case(case_dir: Path, case: dict) -> dict:
    row = _base_row(case)
    recovery_path = case_dir / "weak_recovery.json"
    timing_path = case_dir / "timing.json"
    candidates_path = case_dir / "weak_recovery_candidates.csv"
    for path in (recovery_path, timing_path, candidates_path):
        if not path.is_file():
            raise FileNotFoundError(f"Expected result is missing: {path.name}")
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    summary = recovery.get("summary", recovery)
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    with candidates_path.open("r", encoding="utf-8-sig", newline="") as file:
        candidate_rows = list(csv.DictReader(file))

    candidates = int(summary.get("weak_candidate_count", 0))
    accepted = int(summary.get("weak_recovered_candidate_count", 0))
    rejects = {
        str(reason): int(count)
        for reason, count in summary.get("weak_recovery_reject_reason_counts", {}).items()
    }
    csv_accepted = sum(str(item.get("accepted", "")).casefold() == "true" for item in candidate_rows)
    if len(candidate_rows) != candidates:
        raise ValueError(
            f"Candidate CSV/summary mismatch: {len(candidate_rows)} rows != {candidates} candidates"
        )
    if csv_accepted != accepted:
        raise ValueError(
            f"Candidate CSV/summary mismatch: {csv_accepted} accepted rows != {accepted} accepted"
        )
    if sum(rejects.values()) != int(summary.get("rejected_weak_candidate_count", 0)):
        raise ValueError("Reject reason counts do not match rejected candidate count")

    other_rejects = {
        reason: count for reason, count in rejects.items() if reason not in KNOWN_REJECT_REASONS
    }
    main_reject = max(rejects.items(), key=lambda item: (item[1], item[0]))[0] if rejects else "none"
    row.update({
        "status": "ok",
        "strong_edge_count": int(summary.get("strong_edge_count", 0)),
        "weak_candidate_count": candidates,
        "weak_recovered_candidate_count": accepted,
        "weak_recovered_edge_count": int(summary.get("weak_recovered_edge_count", 0)),
        "rejected_weak_candidate_count": int(summary.get("rejected_weak_candidate_count", 0)),
        **{reason: rejects.get(reason, 0) for reason in KNOWN_REJECT_REASONS},
        "other_reject_count": sum(other_rejects.values()),
        "other_reject_reasons": "; ".join(
            f"{reason}={count}" for reason, count in sorted(other_rejects.items())
        ),
        "accept_rate": _safe_rate(accepted, candidates),
        "direction_reject_rate": _safe_rate(rejects.get("direction_mismatch", 0), candidates),
        "probability_reject_rate": _safe_rate(
            rejects.get("mean_probability_low", 0) + rejects.get("q25_probability_low", 0),
            candidates,
        ),
        "contrast_reject_rate": _safe_rate(
            rejects.get("background_contrast_low", 0), candidates
        ),
        "astar_reject_rate": _safe_rate(rejects.get("no_astar_path", 0), candidates),
        "main_reject_reason": main_reject,
        "weak_recovery_seconds": float(timing.get("weak_recovery_seconds", 0.0)),
        "total_seconds": float(timing.get("total_seconds", 0.0)),
    })
    return row


def _run_case(source_dir: Path, output_dir: Path, case: dict) -> tuple[dict, Path]:
    row = _base_row(case)
    case_dir = _prepare_case(source_dir, output_dir, case)
    command = _command(case_dir, case)
    print(f"[{case['case']}] running {case['directory']}...", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=str(TOOL_ROOT.parent.parent),
            text=True,
            capture_output=True,
            check=False,
        )
        log = (
            "COMMAND\n"
            + subprocess.list2cmdline(command)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\nSTDERR\n"
            + completed.stderr
        )
        (case_dir / "sweep_run.log").write_text(log, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"run_test.py exited with {completed.returncode}: "
                f"{(completed.stderr or completed.stdout)[-2000:]}"
            )
        row = _collect_case(case_dir, case)
        print(
            f"[{case['case']}] candidates={row['weak_candidate_count']} "
            f"accepted={row['weak_recovered_candidate_count']} "
            f"added_edges={row['weak_recovered_edge_count']}",
            flush=True,
        )
    except Exception as exc:
        row["error"] = str(exc)
        print(f"[{case['case']}] failed: {exc}", flush=True)
    return row, case_dir


def _write_results(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_rate(value) -> str:
    return f"{float(value) * 100:.1f}%" if value != "" else "—"


def _transition(rows_by_case: dict[str, dict], first: str, second: str, label: str) -> str:
    a = rows_by_case[first]
    b = rows_by_case[second]
    if a["status"] != "ok" or b["status"] != "ok":
        return f"- **{first} → {second}（{label}）**：至少一组失败，无法比较。"
    direction_delta = int(a["direction_mismatch"]) - int(b["direction_mismatch"])
    accepted_delta = int(b["weak_recovered_candidate_count"]) - int(
        a["weak_recovered_candidate_count"]
    )
    edge_delta = int(b["weak_recovered_edge_count"]) - int(a["weak_recovered_edge_count"])
    text = (
        f"- **{first} → {second}（{label}）**：direction_mismatch 减少 "
        f"{direction_delta}；Accepted 变化 {accepted_delta:+d}；Added edges 变化 {edge_delta:+d}。"
    )
    if "direction" in label.casefold() and direction_delta > 0 and int(
        b["weak_recovered_candidate_count"]
    ) == 0:
        text += "方向门槛释放了候选，但后续 probability/contrast 等规则仍是瓶颈。"
    return text


def _manual_review_cases(rows_by_case: dict[str, dict]) -> list[str]:
    successful = {key: row for key, row in rows_by_case.items() if row["status"] == "ok"}
    selected = ["A"] if "A" in successful else []
    # Visual review is useful first for cases that actually changed the graph.
    # Preserve experiment order so progressive relaxations remain easy to compare.
    for case in ("B", "C", "D", "E"):
        if case in successful and int(successful[case]["weak_recovered_candidate_count"]) > 0:
            selected.append(case)
        if len(selected) == 3:
            return selected
    # If fewer than three cases changed the graph, fill with successful controls.
    for case in ("B", "C", "D", "E"):
        if case in successful and case not in selected:
            selected.append(case)
        if len(selected) == 3:
            break
    return selected


def _write_endpoint_report(path: Path, rows: list[dict]) -> None:
    rows_by_case = {row["case"]: row for row in rows}
    lines = [
        "# Weak Endpoint Recovery Parameter Sweep",
        "",
        "All cases reuse the same `weak_sensor` SAMRoad cache with Bootstrap disabled.",
        "No road HIGH/LOW, gap, extension, path-ratio, auto-score, or Bootstrap parameter was changed.",
        "",
        "| Case | Direction | Mean | Q25 | Contrast | Candidates | Accepted | Added edges | Accept rate | Main reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "ok":
            lines.append(
                f"| {row['case']} | {float(row['direction_cosine']):.2f} | "
                f"{float(row['min_mean_probability']):.2f} | "
                f"{float(row['min_q25_probability']):.2f} | "
                f"{float(row['min_background_contrast']):.2f} | "
                f"{row['weak_candidate_count']} | {row['weak_recovered_candidate_count']} | "
                f"{row['weak_recovered_edge_count']} | {_fmt_rate(row['accept_rate'])} | "
                f"{row['main_reject_reason']} |"
            )
        else:
            lines.append(
                f"| {row['case']} | {float(row['direction_cosine']):.2f} | "
                f"{float(row['min_mean_probability']):.2f} | "
                f"{float(row['min_q25_probability']):.2f} | "
                f"{float(row['min_background_contrast']):.2f} | — | — | — | — | failed |"
            )
    lines.extend([
        "",
        "## Controlled comparisons",
        "",
        _transition(rows_by_case, "A", "B", "Direction 0.65 → 0.50"),
        _transition(rows_by_case, "B", "C", "Direction 0.50 → 0.40"),
        _transition(rows_by_case, "B", "D", "Mean/Q25 0.20/0.17 → 0.16/0.12"),
        _transition(rows_by_case, "D", "E", "Contrast 0.08 → 0.06"),
        "",
        "## Main reject reason by case",
        "",
    ])
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"- **{row['case']}**: failed — {row['error']}")
            continue
        reject_counts = {
            reason: int(row[reason]) for reason in KNOWN_REJECT_REASONS if int(row[reason])
        }
        if row["other_reject_reasons"]:
            for item in str(row["other_reject_reasons"]).split("; "):
                reason, count = item.split("=", 1)
                reject_counts[reason] = int(count)
        top = sorted(reject_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        lines.append(
            f"- **{row['case']}**: "
            + (", ".join(f"{reason}={count}" for reason, count in top) or "none")
        )
    review_cases = _manual_review_cases(rows_by_case)
    lines.extend([
        "",
        "## Manual review focus",
        "",
        "There is no road ground truth, so this sweep does not select final parameters.",
        "Use A as the fixed visual reference and prioritize manual comparison of: "
        + (", ".join(review_cases) if review_cases else "no successful cases")
        + ".",
        "Inspect whether newly accepted links cross unrelated roads, join parallel roads, or follow genuine weak road evidence.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _panel(image_path: Path, label: str, width: int = 900, image_height: int = 900) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.full((image_height, width, 3), 35, dtype=np.uint8)
        cv2.putText(
            image, "result unavailable", (40, image_height // 2), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (220, 220, 220), 2, cv2.LINE_AA,
        )
    else:
        scale = min(width / image.shape[1], image_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros((image_height, width, 3), dtype=np.uint8)
        y0 = (image_height - resized.shape[0]) // 2
        x0 = (width - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        image = canvas
    header = np.zeros((62, width, 3), dtype=np.uint8)
    cv2.putText(header, label, (18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header, image])


def _write_montage(path: Path, case_dirs: dict[str, Path], cases: tuple[dict, ...]) -> None:
    panels = [
        _panel(case_dirs[case["case"]] / "recovery_compare.png", case["label"])
        for case in cases
    ]
    column_count = 3
    while len(panels) % column_count:
        panels.append(np.zeros_like(panels[0]))
    montage = np.vstack([
        np.hstack(panels[index:index + column_count])
        for index in range(0, len(panels), column_count)
    ])
    if not cv2.imwrite(str(path), montage):
        raise RuntimeError(f"Cannot write montage: {path}")


ACCEPTED_FIELDS = (
    "case",
    "candidate_id",
    "start",
    "end",
    "distance",
    "path_length",
    "path_ratio",
    "direction_cosine",
    "mean_probability",
    "q25_probability",
    "background_contrast",
    "recovery_score",
)


def _point(value: str) -> tuple[float, float]:
    row, col = json.loads(value)
    return float(row), float(col)


def _point_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _ordered_edge_path(edges: list[dict]) -> list[tuple[float, float]]:
    adjacency: dict[tuple[float, float], list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    for index, edge in enumerate(edges):
        src = (float(edge["src_row"]), float(edge["src_col"]))
        dst = (float(edge["dst_row"]), float(edge["dst_col"]))
        adjacency[src].append((index, dst))
        adjacency[dst].append((index, src))
    endpoints = [point for point, links in adjacency.items() if len(links) == 1]
    current = endpoints[0] if endpoints else next(iter(adjacency))
    path = [current]
    used: set[int] = set()
    while len(used) < len(edges):
        available = [(index, other) for index, other in adjacency[current] if index not in used]
        if not available:
            break
        edge_index, other = available[0]
        used.add(edge_index)
        path.append(other)
        current = other
    if len(used) != len(edges):
        raise ValueError("Recovered edge group is not a single connected path")
    return path


def _recovered_path_groups(case_dir: Path) -> list[dict]:
    recovery = json.loads((case_dir / "weak_recovery.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for edge in recovery.get("recovered_edges", []):
        grouped[str(edge.get("recovery_id", ""))].append(edge)
    result = []
    for recovery_id, edges in sorted(grouped.items()):
        if not recovery_id:
            raise ValueError("Recovered edge is missing recovery_id")
        result.append({
            "recovery_id": recovery_id,
            "path": _ordered_edge_path(edges),
            "recovery_score": float(edges[0].get("recovery_score", 0.0)),
        })
    return result


def _accepted_case_records(case_dir: Path, case: str) -> list[dict]:
    with (case_dir / "weak_recovery_candidates.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        accepted = [
            row for row in csv.DictReader(file)
            if str(row.get("accepted", "")).casefold() == "true"
        ]
    groups = _recovered_path_groups(case_dir)
    if len(accepted) != len(groups):
        raise ValueError(
            f"Accepted/path group mismatch for {case}: {len(accepted)} != {len(groups)}"
        )
    unused = set(range(len(groups)))
    records = []
    for sequence, row in enumerate(accepted, start=1):
        start = _point(row["start"])
        end = _point(row["end"])
        matches = []
        for index in unused:
            path = groups[index]["path"]
            forward = _point_distance(start, path[0]) + _point_distance(end, path[-1])
            reverse = _point_distance(start, path[-1]) + _point_distance(end, path[0])
            matches.append((min(forward, reverse), reverse < forward, index))
        match_distance, reverse, group_index = min(matches)
        if match_distance > 4.0:
            raise ValueError(
                f"Cannot match accepted candidate {case}_{sequence:03d} to recovered path "
                f"(endpoint error {match_distance:.2f}px)"
            )
        unused.remove(group_index)
        path = list(groups[group_index]["path"])
        if reverse:
            path.reverse()
        records.append({
            "case": case,
            "candidate_id": f"{case}_{sequence:03d}",
            "start": row["start"],
            "end": row["end"],
            "distance": row["distance"],
            "path_length": row["path_length"],
            "path_ratio": row["path_ratio"],
            "direction_cosine": row["direction_cosine"],
            "mean_probability": row["mean_probability"],
            "q25_probability": row["q25_probability"],
            "background_contrast": row["background_contrast"],
            "recovery_score": groups[group_index]["recovery_score"],
            "_start": start,
            "_end": end,
            "_path": path,
        })
    return records


def _read_original_image(source_dir: Path) -> np.ndarray:
    config = json.loads((source_dir / "test_config.json").read_text(encoding="utf-8"))
    image_path = Path(config["input_image"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read original image: {image_path}")
    return image


def _read_strong_edges(source_dir: Path) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    with (source_dir / "original_edge_scores.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return [
            (
                (float(row["src_row"]), float(row["src_col"])),
                (float(row["dst_row"]), float(row["dst_col"])),
            )
            for row in csv.DictReader(file)
        ]


def _crop_bounds(
    points: list[tuple[float, float]], image_shape: tuple[int, ...], margin: int = 100,
    minimum_size: int = 256,
) -> tuple[int, int, int, int]:
    rows = [point[0] for point in points]
    cols = [point[1] for point in points]
    y0 = int(math.floor(min(rows))) - margin
    y1 = int(math.ceil(max(rows))) + margin + 1
    x0 = int(math.floor(min(cols))) - margin
    x1 = int(math.ceil(max(cols))) + margin + 1
    height, width = image_shape[:2]

    def expand(low: int, high: int, limit: int) -> tuple[int, int]:
        span = high - low
        if span < minimum_size:
            extra = minimum_size - span
            low -= extra // 2
            high += extra - extra // 2
        if low < 0:
            high -= low
            low = 0
        if high > limit:
            low -= high - limit
            high = limit
        return max(0, low), min(limit, high)

    y0, y1 = expand(y0, y1, height)
    x0, x1 = expand(x0, x1, width)
    return x0, y0, x1, y1


def _draw_candidate_image(
    image: np.ndarray,
    strong_edges: list[tuple[tuple[float, float], tuple[float, float]]],
    record: dict,
) -> np.ndarray:
    x0, y0, x1, y1 = _crop_bounds(record["_path"], image.shape)
    crop = image[y0:y1, x0:x1].copy()

    def xy(point: tuple[float, float]) -> tuple[int, int]:
        return int(round(point[1] - x0)), int(round(point[0] - y0))

    for src, dst in strong_edges:
        if (
            max(src[1], dst[1]) < x0 or min(src[1], dst[1]) >= x1
            or max(src[0], dst[0]) < y0 or min(src[0], dst[0]) >= y1
        ):
            continue
        cv2.line(crop, xy(src), xy(dst), (0, 255, 255), 2, cv2.LINE_AA)
    path_points = np.asarray([xy(point) for point in record["_path"]], dtype=np.int32)
    if len(path_points) > 1:
        cv2.polylines(crop, [path_points], False, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(crop, xy(record["_start"]), 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(crop, xy(record["_end"]), 7, (0, 0, 255), -1, cv2.LINE_AA)

    canvas_width = max(640, crop.shape[1])
    if crop.shape[1] < canvas_width:
        padded_crop = np.zeros((crop.shape[0], canvas_width, 3), dtype=np.uint8)
        crop_x = (canvas_width - crop.shape[1]) // 2
        padded_crop[:, crop_x:crop_x + crop.shape[1]] = crop
        crop = padded_crop
    header = np.zeros((104, canvas_width, 3), dtype=np.uint8)
    line1 = (
        f"case={record['case']}  id={record['candidate_id']}  dist={float(record['distance']):.1f}  "
        f"dir={float(record['direction_cosine']):.3f}  score={float(record['recovery_score']):.3f}"
    )
    line2 = (
        f"mean={float(record['mean_probability']):.3f}  "
        f"q25={float(record['q25_probability']):.3f}  "
        f"contrast={float(record['background_contrast']):.3f}  "
        f"path={float(record['path_length']):.1f}"
    )
    cv2.putText(header, line1, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, line2, (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        header, "yellow=strong  cyan=candidate  green=start  red=end", (8, 84),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA,
    )
    return np.vstack([header, crop])


def _accepted_montage(path: Path, image_paths: list[Path]) -> None:
    if not image_paths:
        montage = np.full((300, 700, 3), 28, dtype=np.uint8)
        cv2.putText(
            montage, "No accepted candidates", (145, 160), cv2.FONT_HERSHEY_SIMPLEX,
            1.1, (230, 230, 230), 2, cv2.LINE_AA,
        )
    else:
        tile_width, tile_height = 640, 360
        panels = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
            resized = cv2.resize(
                image,
                (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            panel = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            px = (tile_width - resized.shape[1]) // 2
            py = (tile_height - resized.shape[0]) // 2
            panel[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
            panels.append(panel)
        column_count = min(3, len(panels))
        while len(panels) % column_count:
            panels.append(np.zeros_like(panels[0]))
        montage = np.vstack([
            np.hstack(panels[index:index + column_count])
            for index in range(0, len(panels), column_count)
        ])
    if not cv2.imwrite(str(path), montage):
        raise RuntimeError(f"Cannot write accepted candidate montage: {path}")


def _duplicate_groups(records: list[dict], tolerance: float = 12.0) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for record in records:
        matched = None
        for group in groups:
            reference = group[0]
            direct = max(
                _point_distance(record["_start"], reference["_start"]),
                _point_distance(record["_end"], reference["_end"]),
            )
            reverse = max(
                _point_distance(record["_start"], reference["_end"]),
                _point_distance(record["_end"], reference["_start"]),
            )
            if min(direct, reverse) <= tolerance:
                matched = group
                break
        if matched is None:
            groups.append([record])
        else:
            matched.append(record)
    return groups


def _write_accepted_outputs(
    output_dir: Path, source_dir: Path, case_dirs: dict[str, Path], cases: tuple[dict, ...],
) -> tuple[list[dict], list[list[dict]]]:
    accepted_dir = (output_dir / "accepted_candidates").resolve()
    if accepted_dir.parent != output_dir:
        raise ValueError(f"Unsafe accepted candidate directory: {accepted_dir}")
    if accepted_dir.exists():
        shutil.rmtree(accepted_dir)
    accepted_dir.mkdir(parents=True)
    records = []
    for case in cases:
        records.extend(_accepted_case_records(case_dirs[case["case"]], case["case"]))

    with (output_dir / "accepted_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=ACCEPTED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    original = _read_original_image(source_dir)
    strong_edges = _read_strong_edges(source_dir)
    image_paths = []
    for record in records:
        candidate_image = _draw_candidate_image(original, strong_edges, record)
        image_path = accepted_dir / f"{record['candidate_id']}.png"
        if not cv2.imwrite(str(image_path), candidate_image):
            raise RuntimeError(f"Cannot write accepted candidate image: {image_path}")
        image_paths.append(image_path)
    _accepted_montage(accepted_dir / "accepted_candidates_montage.png", image_paths)
    return records, _duplicate_groups(records)


def _effect(rows_by_case: dict[str, dict], first: str, second: str, label: str) -> str:
    before = rows_by_case[first]
    after = rows_by_case[second]
    if before["status"] != "ok" or after["status"] != "ok":
        return f"- **{label}（{first} → {second}）**：至少一组失败，无法比较。"
    accepted_delta = int(after["weak_recovered_candidate_count"]) - int(
        before["weak_recovered_candidate_count"]
    )
    edge_delta = int(after["weak_recovered_edge_count"]) - int(before["weak_recovered_edge_count"])
    return (
        f"- **{label}（{first} → {second}）**：Accepted {accepted_delta:+d}，"
        f"Added edges {edge_delta:+d}；{second} 共接受 "
        f"{after['weak_recovered_candidate_count']} 个候选。"
    )


def _write_probability_report(
    path: Path, rows: list[dict], duplicate_groups: list[list[dict]],
) -> None:
    rows_by_case = {row["case"]: row for row in rows}
    lines = [
        "# 弱道路恢复 Probability Sweep（E–K）",
        "",
        "本轮全部复用 `outputs/2` 的 `weak_sensor` 缓存，仅执行 recovery-only；Bootstrap 关闭。",
        "固定 Direction=0.50、Max gap=64、ROAD_HIGH=0.24、ROAD_LOW=0.10，未修改道路算法或默认配置。",
        "",
        "| Case | Mean | Q25 | Contrast | Candidates | Accepted | Added edges | Accept rate | Main reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "ok":
            lines.append(
                f"| {row['case']} | {float(row['min_mean_probability']):.2f} | "
                f"{float(row['min_q25_probability']):.2f} | "
                f"{float(row['min_background_contrast']):.2f} | "
                f"{row['weak_candidate_count']} | {row['weak_recovered_candidate_count']} | "
                f"{row['weak_recovered_edge_count']} | {_fmt_rate(row['accept_rate'])} | "
                f"{row['main_reject_reason']} |"
            )
        else:
            lines.append(
                f"| {row['case']} | {row['min_mean_probability']} | {row['min_q25_probability']} | "
                f"{row['min_background_contrast']} | — | — | — | — | failed |"
            )
    lines.extend([
        "",
        "## 单因素因果对比",
        "",
        _effect(rows_by_case, "E", "F", "Mean 0.16 → 0.14"),
        _effect(rows_by_case, "E", "G", "Q25 0.12 → 0.10"),
        _effect(rows_by_case, "E", "I", "Contrast 0.06 → 0.05"),
        _effect(rows_by_case, "I", "J", "Contrast 0.05 → 0.04"),
        "",
        "## 组合效应与非线性",
        "",
        _effect(rows_by_case, "E", "H", "Mean+Q25 联合放宽"),
        _effect(rows_by_case, "H", "K", "联合条件下再将 Contrast 0.06 → 0.05"),
    ])
    if all(rows_by_case[case]["status"] == "ok" for case in ("E", "F", "G", "I", "K")):
        count = lambda case: int(rows_by_case[case]["weak_recovered_candidate_count"])
        additive = (count("F") - count("E")) + (count("G") - count("E")) + (count("I") - count("E"))
        observed = count("K") - count("E")
        lines.append(
            f"- 以 E 为基准，三个单因素增量的简单加和为 {additive:+d}，K 的实际增量为 "
            f"{observed:+d}，非线性交互量为 {observed - additive:+d}。"
        )

    repeated = [group for group in duplicate_groups if len({item["case"] for item in group}) > 1]
    singletons = sum(len(group) == 1 for group in duplicate_groups)
    lines.extend([
        "",
        "## 跨 Case 重复接受候选（端点 12 px 近邻匹配）",
        "",
        f"共得到 {len(duplicate_groups)} 个空间候选组，其中 {len(repeated)} 组在多个 Case 被接受，"
        f"{singletons} 组仅在一个 Case 被接受。",
        "",
    ])
    for index, group in enumerate(repeated, start=1):
        cases = ", ".join(item["case"] for item in group)
        ids = ", ".join(item["candidate_id"] for item in group)
        lines.append(f"- **G{index:03d}**：{cases}（{ids}）")
    if not repeated:
        lines.append("- 未发现跨 Case 重复接受候选。")
    unique_by_case: dict[str, list[str]] = defaultdict(list)
    for group in duplicate_groups:
        if len(group) == 1:
            unique_by_case[group[0]["case"]].append(group[0]["candidate_id"])
    for case, candidate_ids in sorted(unique_by_case.items()):
        lines.append(f"- **仅 {case} 接受**：{', '.join(candidate_ids)}")

    warnings = []
    for row in rows:
        if row["status"] == "ok" and int(row["weak_recovered_candidate_count"]) >= 20:
            warnings.append(
                f"{row['case']} 接受 {row['weak_recovered_candidate_count']} 个候选，已达到数十量级"
            )
    lines.extend(["", "## 风险与人工检查重点", ""])
    if warnings:
        lines.append("- **数量突增警告**：" + "；".join(warnings) + "。需优先排查误连。")
    else:
        lines.append("- 本轮没有 Case 跳升到数十或数百个接受候选。")
    lines.extend([
        "- 不自动选择最佳参数；没有道路真值时，接受数增加不等于质量提高。",
        "- 优先查看 E（参考）、H（Mean+Q25 交互）、J（Contrast 下限）和 K（组合条件）的局部候选；"
        "再用 F/G/I 分辨单因素来源。",
        "- 重点检查是否跨接无关道路、误连平行道路，或仅沿真实弱道路证据延伸。",
        "- 全部局部图见 `accepted_candidates/accepted_candidates_montage.png`；逐候选指标见 "
        "`accepted_candidates.csv`。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    source_dir = Path(arguments.source_dir).expanduser().resolve()
    cases = PROBABILITY_CASES if arguments.preset == "probability" else ENDPOINT_CASES
    default_output = (
        DEFAULT_PROBABILITY_OUTPUT_DIR
        if arguments.preset == "probability"
        else DEFAULT_ENDPOINT_OUTPUT_DIR
    )
    output_dir = Path(arguments.output_dir or default_output).expanduser().resolve()
    _validate_directories(source_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: _sha256(source_dir / name) for name in CACHE_FILES}

    rows = []
    case_dirs = {}
    for case in cases:
        row, case_dir = _run_case(source_dir, output_dir, case)
        rows.append(row)
        case_dirs[case["case"]] = case_dir

    _write_results(output_dir / "sweep_results.csv", rows)
    _write_montage(output_dir / "comparison_montage.png", case_dirs, cases)
    if arguments.preset == "probability" and all(row["status"] == "ok" for row in rows):
        _, duplicate_groups = _write_accepted_outputs(
            output_dir, source_dir, case_dirs, cases
        )
        _write_probability_report(output_dir / "sweep_report.md", rows, duplicate_groups)
    elif arguments.preset == "endpoint":
        _write_endpoint_report(output_dir / "sweep_report.md", rows)
    else:
        _write_probability_report(output_dir / "sweep_report.md", rows, [])

    final_hashes = {name: _sha256(source_dir / name) for name in CACHE_FILES}
    if final_hashes != source_hashes:
        raise RuntimeError("Baseline cache changed during the sweep")
    succeeded = sum(row["status"] == "ok" for row in rows)
    print(f"Sweep complete: {succeeded}/{len(rows)} cases succeeded", flush=True)
    print(f"Results: {output_dir}", flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
