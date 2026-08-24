from __future__ import annotations

"""Per-input-image resume markers for SAMRoad inference.

This module intentionally uses only the Python standard library so marker
validation can be unit tested without importing the model runtime.
"""

import csv
import hashlib
import json
import os
import pickle
import struct
import time
import zlib
from pathlib import Path


MARKER_SCHEMA_VERSION = 1
SUPPORTED_MARKER_SCHEMAS = {MARKER_SCHEMA_VERSION}


def _resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _path_text(path: Path | str) -> str:
    return os.path.normcase(str(_resolved(path)))


def file_identity(path: Path | str, *, sha256: bool = False) -> dict:
    source = _resolved(path)
    stat = source.stat()
    identity = {
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if sha256:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def build_batch_identity(
    checkpoint: Path | str, config: Path | str, parameters: dict,
) -> dict:
    """Fingerprint model/config once per process, never once per image."""
    return {
        "checkpoint": file_identity(checkpoint, sha256=False),
        "config": file_identity(config, sha256=True),
        "parameters": json.loads(json.dumps(parameters, sort_keys=True, default=str)),
    }


def stable_image_key(image_path: Path | str) -> str:
    source = _resolved(image_path)
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in source.stem
    ).strip("._-") or "image"
    digest = hashlib.sha256(_path_text(source).encode("utf-8")).hexdigest()[:16]
    return f"{safe_stem}-{digest}"


def ensure_unique_output_stems(image_paths) -> None:
    grouped: dict[str, list[str]] = {}
    for value in image_paths:
        source = _resolved(value)
        grouped.setdefault(source.stem.casefold(), []).append(str(source))
    conflicts = [paths for paths in grouped.values() if len(paths) > 1]
    if conflicts:
        details = "\n".join("- " + " | ".join(paths) for paths in conflicts)
        raise ValueError(
            "Input images contain duplicate stems and would overwrite existing SAMRoad outputs:\n"
            + details
        )


def required_image_outputs(output_dir: Path | str, stem: str) -> list[dict]:
    root = _resolved(output_dir)
    return [
        {"role": "graph", "path": root / "graph" / f"{stem}.p", "kind": "pickle_graph"},
        {
            "role": "edge_scores", "path": root / "graph" / f"{stem}_edge_scores.csv",
            "kind": "csv", "headers": ("edge_id", "src_row", "src_col", "dst_row", "dst_col"),
        },
        {
            "role": "recovery_summary", "path": root / "graph" / f"{stem}_weak_recovery.json",
            "kind": "json",
        },
        {
            "role": "edge_candidates", "path": root / "graph" / f"{stem}_edge_candidates.csv",
            "kind": "csv", "headers": ("candidate_id", "src_row", "src_col", "dst_row", "dst_col"),
        },
        {"role": "road_mask", "path": root / "mask" / f"{stem}_road.png", "kind": "png"},
        {
            "role": "intersection_mask", "path": root / "mask" / f"{stem}_itsc.png",
            "kind": "png",
        },
        {
            "role": "centerline_probability",
            "path": root / "mask" / f"{stem}_centerline_probability.png", "kind": "png",
        },
        {"role": "graph_visualization", "path": root / "viz" / f"{stem}.png", "kind": "png"},
    ]


def all_image_outputs(output_dir: Path | str, stem: str) -> list[Path]:
    root = _resolved(output_dir)
    paths = [Path(item["path"]) for item in required_image_outputs(root, stem)]
    paths.extend(
        root / directory / f"{stem}{suffix}"
        for directory, suffix in (
            ("mask", "_relative_roadness.png"),
            ("mask", "_relative_candidate.png"),
            ("mask", "_relative_skeleton_raw.png"),
            ("mask", "_relative_skeleton_normalized.png"),
            ("mask", "_junction_zone_mask.png"),
            ("mask", "_pruned_spur_mask.png"),
            ("mask", "_collapsed_zone_mask.png"),
            ("mask", "_combined_candidate.png"),
            ("viz", "_relative_compare.png"),
            ("viz", "_relative_acceptance_overlay.png"),
            ("graph", "_relative_acceptance_funnel.json"),
            ("graph", "_relative_skeleton_normalization.json"),
            ("graph", "_relative_review_candidates.csv"),
        )
    )
    return paths


def _validate_png(path: Path) -> tuple[bool, str]:
    try:
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                return False, "invalid PNG signature"
            length_bytes, chunk_type = stream.read(4), stream.read(4)
            if len(length_bytes) != 4 or struct.unpack(">I", length_bytes)[0] != 13 or chunk_type != b"IHDR":
                return False, "invalid PNG IHDR"
            header = stream.read(13)
            crc_bytes = stream.read(4)
            if len(header) != 13 or len(crc_bytes) != 4:
                return False, "truncated PNG IHDR"
            if struct.unpack(">I", crc_bytes)[0] != (zlib.crc32(chunk_type + header) & 0xFFFFFFFF):
                return False, "invalid PNG IHDR CRC"
            width, height = struct.unpack(">II", header[:8])
            if width <= 0 or height <= 0:
                return False, "invalid PNG dimensions"
            stream.seek(-12, os.SEEK_END)
            ending = stream.read(12)
            if ending != b"\x00\x00\x00\x00IEND\xaeB`\x82":
                return False, "PNG is missing a complete IEND"
        return True, ""
    except (OSError, struct.error) as exc:
        return False, str(exc)


def _valid_coordinate(value) -> bool:
    return (
        isinstance(value, (tuple, list)) and len(value) == 2
        and all(isinstance(number, (int, float)) for number in value)
    )


def _validate_pickle_graph(path: Path) -> tuple[bool, str]:
    try:
        with path.open("rb") as stream:
            graph = pickle.load(stream)
    except Exception as exc:
        return False, f"cannot deserialize graph: {exc}"
    if not isinstance(graph, dict):
        return False, "graph payload is not a dictionary"
    for node, neighbors in graph.items():
        if not _valid_coordinate(node) or not isinstance(neighbors, (list, tuple)):
            return False, "graph adjacency structure is invalid"
        if any(not _valid_coordinate(neighbor) for neighbor in neighbors):
            return False, "graph neighbor coordinate is invalid"
    return True, ""


def _validate_json(path: Path) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"cannot parse JSON: {exc}"
    return (True, "") if isinstance(payload, (dict, list)) else (False, "JSON root is invalid")


def _validate_csv(path: Path, headers) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), None)
    except (OSError, UnicodeError, csv.Error) as exc:
        return False, f"cannot read CSV header: {exc}"
    if not header:
        return False, "CSV header is missing"
    missing = [name for name in headers or () if name not in header]
    return (False, "CSV header is missing: " + ", ".join(missing)) if missing else (True, "")


def validate_output(path: Path | str, kind: str, *, headers=()) -> tuple[bool, str]:
    source = _resolved(path)
    try:
        if not source.is_file() or source.stat().st_size <= 0:
            return False, f"missing or empty output: {source}"
    except OSError as exc:
        return False, str(exc)
    if kind == "png":
        return _validate_png(source)
    if kind == "json":
        return _validate_json(source)
    if kind == "csv":
        return _validate_csv(source, headers)
    if kind == "pickle_graph":
        return _validate_pickle_graph(source)
    return False, f"unsupported output kind: {kind}"


def _same_identity(stored: object, current: dict) -> bool:
    return isinstance(stored, dict) and stored == current


def _read_json_object(path: Path) -> tuple[dict | None, str]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    return (value, "") if isinstance(value, dict) else (None, "JSON root is not an object")


def _state_file_identity(state: dict, key: str) -> dict | None:
    spec = state.get("input_spec") if isinstance(state.get("input_spec"), dict) else {}
    value = spec.get(key) if isinstance(spec.get(key), dict) else None
    if not value:
        return None
    try:
        return {
            "path": str(_resolved(value["path"])),
            "size": int(value["size"]),
            "mtime_ns": int(value["mtime_ns"]),
        }
    except (KeyError, OSError, TypeError, ValueError):
        return None


def load_pipeline_identities(path: Path | str | None) -> dict:
    if not path:
        return {}
    state, _reason = _read_json_object(_resolved(path))
    if state is None:
        return {}
    return {
        key: identity for key in ("checkpoint", "config")
        if (identity := _state_file_identity(state, key)) is not None
    }


def _identity_without_hash(identity: dict) -> dict:
    return {key: identity.get(key) for key in ("path", "size", "mtime_ns")}


def _legacy_batch_compatible(
    legacy_metadata: dict | None, batch_identity: dict, pipeline_identities: dict,
) -> tuple[bool, str]:
    if not isinstance(legacy_metadata, dict):
        return False, "legacy inference_metadata.json is missing or invalid"
    recorded_identity = legacy_metadata.get("resume_identity")
    if isinstance(recorded_identity, dict):
        return (
            (True, "") if recorded_identity == batch_identity
            else (False, "legacy inference identity changed")
        )
    for key in ("checkpoint", "config"):
        metadata_path = legacy_metadata.get(key)
        if not metadata_path or _path_text(metadata_path) != _path_text(batch_identity[key]["path"]):
            return False, f"legacy {key} path changed"
        state_identity = pipeline_identities.get(key)
        if state_identity is None:
            return False, f"legacy {key} fingerprint is unavailable"
        if _identity_without_hash(batch_identity[key]) != state_identity:
            return False, f"legacy {key} fingerprint changed"
    parameter_checks = {
        "device": "device",
        "topology_threshold": "topology_threshold",
        "topology_candidate_threshold": "topology_candidate_threshold",
        "requested_road_threshold_profile": "requested_road_threshold_profile",
        "weak_recovery_enabled": "weak_recovery_enabled",
        "weak_bootstrap_enabled": "weak_bootstrap_enabled",
        "relative_roadness_enabled": "relative_roadness_enabled",
    }
    parameters = batch_identity.get("parameters", {})
    for metadata_key, parameter_key in parameter_checks.items():
        if metadata_key in legacy_metadata and legacy_metadata[metadata_key] != parameters.get(parameter_key):
            return False, f"legacy inference parameter changed: {metadata_key}"
    return True, ""


class ImageResumeManager:
    def __init__(
        self, output_dir: Path | str, batch_identity: dict, *, enabled: bool,
        legacy_metadata: dict | None = None, pipeline_state: Path | str | None = None,
    ) -> None:
        self.output_dir = _resolved(output_dir)
        self.batch_identity = batch_identity
        self.enabled = bool(enabled)
        self.legacy_metadata = legacy_metadata
        self.pipeline_identities = load_pipeline_identities(pipeline_state)
        self.resume_dir = self.output_dir / ".resume"

    def marker_path(self, image_path: Path | str) -> Path:
        return self.resume_dir / f"{stable_image_key(image_path)}.completed.json"

    def _output_records(self, stem: str) -> tuple[list[dict] | None, str]:
        records = []
        for spec in required_image_outputs(self.output_dir, stem):
            valid, reason = validate_output(spec["path"], spec["kind"], headers=spec.get("headers", ()))
            if not valid:
                return None, f"{spec['role']}: {reason}"
            path = Path(spec["path"])
            records.append({
                "role": spec["role"],
                "path": path.relative_to(self.output_dir).as_posix(),
                "kind": spec["kind"],
                "size": int(path.stat().st_size),
            })
        return records, ""

    def _validate_marker(self, marker: dict, image_path: Path) -> tuple[bool, str]:
        if marker.get("schema_version") not in SUPPORTED_MARKER_SCHEMAS:
            return False, "unsupported marker schema"
        if not _same_identity(marker.get("input"), file_identity(image_path)):
            return False, "input image identity changed"
        if not _same_identity(marker.get("checkpoint"), self.batch_identity["checkpoint"]):
            return False, "checkpoint identity changed"
        if not _same_identity(marker.get("config"), self.batch_identity["config"]):
            return False, "config identity changed"
        if marker.get("parameters") != self.batch_identity.get("parameters"):
            return False, "inference parameters changed"
        expected = {item["role"]: item for item in required_image_outputs(self.output_dir, image_path.stem)}
        recorded = {
            str(item.get("role")): item
            for item in marker.get("outputs", []) if isinstance(item, dict)
        }
        if set(recorded) != set(expected):
            return False, "marker output list is incomplete"
        for role, spec in expected.items():
            row = recorded[role]
            path = self.output_dir / str(row.get("path") or "")
            if path.resolve() != Path(spec["path"]).resolve():
                return False, f"output path changed: {role}"
            try:
                if path.stat().st_size != int(row.get("size", -1)):
                    return False, f"output size changed: {role}"
            except (OSError, TypeError, ValueError):
                return False, f"output is missing: {role}"
            valid, reason = validate_output(path, spec["kind"], headers=spec.get("headers", ()))
            if not valid:
                return False, f"{role}: {reason}"
        summary = marker.get("summary")
        if not isinstance(summary, dict) or not isinstance(summary.get("recovery_summary"), dict):
            return False, "marker summary is incomplete"
        if not isinstance(summary.get("profile_decision"), dict):
            return False, "marker profile decision is incomplete"
        return True, ""

    def inspect(self, image_path: Path | str) -> dict:
        image = _resolved(image_path)
        if not self.enabled:
            return {"action": "process", "reason": "image resume is disabled"}
        marker_path = self.marker_path(image)
        if marker_path.is_file():
            marker, reason = _read_json_object(marker_path)
            if marker is None:
                return {"action": "process", "reason": f"invalid marker JSON: {reason}"}
            valid, reason = self._validate_marker(marker, image)
            if valid:
                return {"action": "skip", "origin": "marker", "marker": marker}
            return {"action": "process", "reason": reason}
        return self._adopt_legacy(image)

    def _adopt_legacy(self, image: Path) -> dict:
        compatible, reason = _legacy_batch_compatible(
            self.legacy_metadata, self.batch_identity, self.pipeline_identities,
        )
        if not compatible:
            return {"action": "process", "reason": reason}
        records, reason = self._output_records(image.stem)
        if records is None:
            return {"action": "process", "reason": f"legacy outputs incomplete: {reason}"}
        recovery_path = self.output_dir / "graph" / f"{image.stem}_weak_recovery.json"
        recovery, reason = _read_json_object(recovery_path)
        if recovery is None:
            return {"action": "process", "reason": f"legacy recovery summary invalid: {reason}"}
        recorded_image = str(recovery.get("image") or "").strip()
        if not recorded_image or _path_text(recorded_image) != _path_text(image):
            return {"action": "process", "reason": "legacy recovery summary input path changed"}
        if str(recovery.get("tile") or "") != image.stem:
            return {"action": "process", "reason": "legacy recovery summary stem changed"}
        profile_fields = (
            "image", "tile", "requested_profile", "effective_profile",
            "scene_confidence_state", "diagnostic_reference_profile",
        )
        profile = {key: recovery[key] for key in profile_fields if key in recovery}
        recovery = dict(recovery)
        recovery["resume_origin"] = "legacy_adopted"
        expected_summary_fields = (
            "requested_profile", "effective_profile", "total_image_seconds",
            "strong_edge_count", "weak_candidate_count", "weak_recovered_edge_count",
        )
        recovery["legacy_unknown_fields"] = [
            key for key in expected_summary_fields if key not in recovery
        ]
        profile["resume_origin"] = "legacy_adopted"
        marker = self._write_marker(
            image, records, recovery, profile, origin="legacy_adopted",
        )
        return {"action": "skip", "origin": "legacy_adopted", "marker": marker}

    def prepare_for_processing(self, image_path: Path | str) -> None:
        image = _resolved(image_path)
        self.marker_path(image).unlink(missing_ok=True)
        for path in all_image_outputs(self.output_dir, image.stem):
            path.unlink(missing_ok=True)

    def _write_marker(
        self, image: Path, outputs: list[dict], recovery_summary: dict,
        profile_decision: dict, *, origin: str,
    ) -> dict:
        performance = {
            key: value for key, value in recovery_summary.items()
            if key.endswith("_seconds") or key.endswith("_count")
        }
        marker = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "image_key": stable_image_key(image),
            "stem": image.stem,
            "input": file_identity(image),
            "checkpoint": self.batch_identity["checkpoint"],
            "config": self.batch_identity["config"],
            "parameters": self.batch_identity.get("parameters", {}),
            "outputs": outputs,
            "summary": {
                "recovery_summary": recovery_summary,
                "profile_decision": profile_decision,
                "performance_statistics": performance,
            },
            "origin": origin,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        target = self.marker_path(image)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return marker

    def complete(
        self, image_path: Path | str, recovery_summary: dict, profile_decision: dict,
    ) -> dict:
        image = _resolved(image_path)
        outputs, reason = self._output_records(image.stem)
        if outputs is None:
            self.marker_path(image).unlink(missing_ok=True)
            raise RuntimeError(f"Cannot mark image complete; required output validation failed: {reason}")
        return self._write_marker(
            image, outputs, dict(recovery_summary), dict(profile_decision), origin="inferred",
        )


def marker_summaries(marker: dict) -> tuple[dict, dict]:
    summary = marker.get("summary") if isinstance(marker.get("summary"), dict) else {}
    recovery = dict(summary.get("recovery_summary") or {})
    profile = dict(summary.get("profile_decision") or {})
    return recovery, profile
