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
import zipfile
from dataclasses import dataclass
from pathlib import Path

import sys


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from dependency_identity import (  # noqa: E402
    dependency_identity_equal,
    effective_config_identity,
    ordinary_file_identity_equal,
    stable_file_identity,
)


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
        "checkpoint": stable_file_identity(checkpoint, content_hash=False),
        "config": effective_config_identity(config),
        "parameters": json.loads(json.dumps(parameters, sort_keys=True, default=str)),
    }


def _task_image_identity(image_path: Path | str) -> dict | None:
    source = _resolved(image_path)
    images_root = source.parent
    period_root = images_root.parent
    if (
        images_root.name.casefold() != "images"
        or period_root.parent.name.casefold() != "periods"
    ):
        return None
    return {
        "grid": period_root.parent.parent.name,
        "period": period_root.name,
        "relative_path": source.relative_to(period_root).as_posix(),
        "filename": source.name,
        "size": int(source.stat().st_size),
        "mtime_ns": int(source.stat().st_mtime_ns),
    }


def _safe_stem(image_path: Path | str) -> str:
    source = _resolved(image_path)
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in source.stem
    ).strip("._-") or "image"


def stable_image_key(image_path: Path | str) -> str:
    source = _resolved(image_path)
    task_identity = _task_image_identity(source)
    identity_text = (
        "\0".join(str(task_identity[key]).casefold() for key in ("grid", "period", "relative_path"))
        if task_identity is not None else _path_text(source)
    )
    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:16]
    return f"{_safe_stem(source)}-{digest}"


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


def required_image_outputs(
    output_dir: Path | str, stem: str, execution_profile: str = "full",
) -> list[dict]:
    root = _resolved(output_dir)
    if str(execution_profile).casefold() == "fast":
        return [
            {"role": "road_probability", "path": root / "mask" / f"{stem}_road.png", "kind": "png"},
            {
                "role": "fast_topology",
                "path": root / "graph" / f"{stem}_fast_topology.npz",
                "kind": "npz",
            },
        ]
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


def all_image_outputs(
    output_dir: Path | str, stem: str, execution_profile: str = "full",
) -> list[Path]:
    root = _resolved(output_dir)
    paths = [
        Path(item["path"])
        for item in required_image_outputs(root, stem, execution_profile)
    ]
    if str(execution_profile).casefold() == "fast":
        return paths
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


def _validate_npz(path: Path) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            missing = {"nodes.npy", "edges.npy", "scores.npy"} - names
            if missing:
                return False, "NPZ arrays missing: " + ", ".join(sorted(missing))
            if archive.testzip() is not None:
                return False, "NPZ archive CRC check failed"
    except (OSError, zipfile.BadZipFile) as exc:
        return False, f"cannot read NPZ: {exc}"
    return True, ""


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
    if kind == "npz":
        return _validate_npz(source)
    return False, f"unsupported output kind: {kind}"


def _same_identity(stored: object, current: dict) -> bool:
    return ordinary_file_identity_equal(stored, current)


def _read_json_object(path: Path) -> tuple[dict | None, str]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    return (value, "") if isinstance(value, dict) else (None, "JSON root is not an object")


def _state_file_identity(state: dict, key: str, *, historical: bool = False) -> dict | None:
    provenance = state.get("provenance") if isinstance(state.get("provenance"), dict) else {}
    historical_spec = (
        provenance.get("historical_input_spec")
        if isinstance(provenance.get("historical_input_spec"), dict) else {}
    )
    current = state.get("input_spec") if isinstance(state.get("input_spec"), dict) else {}
    spec = historical_spec if historical and historical_spec else current
    value = spec.get(key) if isinstance(spec.get(key), dict) else None
    if not value:
        return None
    return dict(value)


def load_pipeline_identities(path: Path | str | None) -> dict:
    if not path:
        return {}
    state, _reason = _read_json_object(_resolved(path))
    if state is None:
        return {}
    identities = {
        key: identity for key in ("checkpoint", "config")
        if (identity := _state_file_identity(state, key)) is not None
    }
    identities.update({
        f"historical_{key}": identity for key in ("checkpoint", "config")
        if (identity := _state_file_identity(state, key, historical=True)) is not None
    })
    return identities


def _legacy_batch_compatible(
    legacy_metadata: dict | None, batch_identity: dict, pipeline_identities: dict,
) -> tuple[bool, str]:
    if not isinstance(legacy_metadata, dict):
        return False, "legacy inference_metadata.json is missing or invalid"
    recorded_identity = legacy_metadata.get("resume_identity")
    if isinstance(recorded_identity, dict):
        if not dependency_identity_equal(
            recorded_identity.get("checkpoint"), batch_identity.get("checkpoint"),
            kind="checkpoint",
        ):
            return False, "legacy checkpoint identity changed"
        if not dependency_identity_equal(
            recorded_identity.get("config"), batch_identity.get("config"), kind="config",
        ):
            return False, "legacy config identity changed"
        if recorded_identity.get("parameters") != batch_identity.get("parameters"):
            return False, "legacy inference parameters changed"
        return True, ""
    for key in ("checkpoint", "config"):
        metadata_path = legacy_metadata.get(key)
        if not metadata_path:
            return False, f"legacy {key} path is unavailable"
        state_identity = pipeline_identities.get(f"historical_{key}") or pipeline_identities.get(key)
        if key == "config" and isinstance(legacy_metadata.get("_saved_config_identity"), dict):
            state_identity = legacy_metadata["_saved_config_identity"]
        if state_identity is None:
            return False, f"legacy {key} fingerprint is unavailable"
        if not dependency_identity_equal(state_identity, batch_identity[key], kind=key):
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
        self.execution_profile = str(
            batch_identity.get("parameters", {}).get("execution_profile", "full")
        ).casefold()
        self.enabled = bool(enabled)
        self.legacy_metadata = legacy_metadata
        self.pipeline_identities = load_pipeline_identities(pipeline_state)
        for key in ("checkpoint", "config"):
            pipeline_identity = self.pipeline_identities.get(key)
            if pipeline_identity and dependency_identity_equal(
                pipeline_identity, self.batch_identity.get(key), kind=key,
            ):
                self.batch_identity[key] = {
                    **self.batch_identity[key],
                    **{
                        name: value for name, value in pipeline_identity.items()
                        if name in {
                            "sha256", "effective_sha256", "effective_key_count",
                            "resource_identities",
                        }
                    },
                }
        self.resume_dir = self.output_dir / ".resume"
        self.backup_root = self.resume_dir / "backups"
        self._restore_pending_backups()

    def marker_path(self, image_path: Path | str) -> Path:
        return self.resume_dir / f"{stable_image_key(image_path)}.completed.json"

    def _marker_candidates(self, image_path: Path | str) -> list[Path]:
        image = _resolved(image_path)
        expected = self.marker_path(image)
        candidates = [expected]
        if self.resume_dir.is_dir():
            candidates.extend(
                path for path in sorted(
                    self.resume_dir.glob(f"{_safe_stem(image)}-*.completed.json"), key=str,
                )
                if path != expected
            )
        return candidates

    def _output_records(self, stem: str) -> tuple[list[dict] | None, str]:
        records = []
        for spec in required_image_outputs(self.output_dir, stem, self.execution_profile):
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
        stored_task = marker.get("task_image")
        current_task = _task_image_identity(image_path)
        if isinstance(stored_task, dict):
            if current_task is None or any(
                str(stored_task.get(key) or "").casefold()
                != str(current_task.get(key) or "").casefold()
                for key in ("grid", "period", "relative_path", "filename")
            ):
                return False, "task-relative image identity changed"
        if not _same_identity(marker.get("input"), file_identity(image_path)):
            return False, "input image identity changed"
        if not dependency_identity_equal(
            marker.get("checkpoint"), self.batch_identity["checkpoint"], kind="checkpoint",
        ):
            return False, "checkpoint identity changed"
        if not dependency_identity_equal(
            marker.get("config"), self.batch_identity["config"], kind="config",
        ):
            return False, "config identity changed"
        if marker.get("parameters") != self.batch_identity.get("parameters"):
            return False, "inference parameters changed"
        expected = {
            item["role"]: item
            for item in required_image_outputs(
                self.output_dir, image_path.stem, self.execution_profile,
            )
        }
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
        for marker_path in self._marker_candidates(image):
            if not marker_path.is_file():
                continue
            marker, reason = _read_json_object(marker_path)
            if marker is None:
                return {"action": "process", "reason": f"invalid marker JSON: {reason}"}
            valid, reason = self._validate_marker(marker, image)
            if valid:
                marker = self._relocate_marker(marker_path, marker, image)
                return {"action": "skip", "origin": "marker", "marker": marker}
            return {"action": "process", "reason": reason}
        return self._adopt_legacy(image)

    def _relocate_marker(self, source: Path, marker: dict, image: Path) -> dict:
        target = self.marker_path(image)
        task_identity = _task_image_identity(image)
        updated = dict(marker)
        updated["image_key"] = stable_image_key(image)
        updated["input"] = {**dict(updated.get("input") or {}), "path": str(image)}
        if task_identity is not None:
            updated["task_image"] = task_identity
        summary = dict(updated.get("summary") or {})
        for name in ("recovery_summary", "profile_decision"):
            row = summary.get(name)
            if isinstance(row, dict):
                row = dict(row)
                if "image" in row:
                    row["image"] = str(image)
                summary[name] = row
        updated["summary"] = summary
        if source != target or updated != marker:
            self._atomic_json(target, updated)
            if source != target:
                source.unlink(missing_ok=True)
        return updated

    def _adopt_legacy(self, image: Path) -> dict:
        if self.execution_profile == "fast":
            return {"action": "process", "reason": "legacy Fast outputs predate probability-only markers"}
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
        if not recorded_image or Path(recorded_image).name.casefold() != image.name.casefold():
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
        backup_dir = self.backup_root / stable_image_key(image)
        if backup_dir.is_dir():
            self._restore_backup(backup_dir)
        manifest_path = backup_dir / "backup_manifest.json"
        candidates = [
            *self._marker_candidates(image),
            *all_image_outputs(self.output_dir, image.stem, self.execution_profile),
        ]
        existing = [path for path in candidates if path.is_file()]
        backup_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "image": str(image),
            "files": [path.relative_to(self.output_dir).as_posix() for path in existing],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._atomic_json(manifest_path, manifest)
        try:
            for path in existing:
                relative = path.relative_to(self.output_dir)
                target = backup_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
        except Exception:
            self._restore_backup(backup_dir)
            raise

    @staticmethod
    def _atomic_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_backup(self, backup_dir: Path) -> None:
        manifest, _reason = _read_json_object(backup_dir / "backup_manifest.json")
        if manifest is None:
            return
        for relative_text in manifest.get("files", []):
            relative = Path(str(relative_text))
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                continue
            source = (backup_dir / relative).resolve()
            target = (self.output_dir / relative).resolve()
            try:
                source.relative_to(backup_dir.resolve())
                target.relative_to(self.output_dir)
            except ValueError:
                continue
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
        self._remove_empty_backup(backup_dir)

    def _restore_pending_backups(self) -> None:
        if not self.backup_root.is_dir():
            return
        for backup_dir in sorted(self.backup_root.iterdir(), key=lambda path: path.name):
            if backup_dir.is_dir():
                self._restore_backup(backup_dir)

    def _remove_empty_backup(self, backup_dir: Path) -> None:
        manifest = backup_dir / "backup_manifest.json"
        manifest.unlink(missing_ok=True)
        for directory in sorted(
            (path for path in backup_dir.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts), reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            backup_dir.rmdir()
        except OSError:
            pass

    def _discard_backup(self, image: Path) -> None:
        backup_dir = self.backup_root / stable_image_key(image)
        if not backup_dir.is_dir():
            return
        for path in sorted(
            (path for path in backup_dir.rglob("*") if path.is_file()),
            key=lambda path: len(path.parts), reverse=True,
        ):
            path.unlink(missing_ok=True)
        self._remove_empty_backup(backup_dir)

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
        task_identity = _task_image_identity(image)
        if task_identity is not None:
            marker["task_image"] = task_identity
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
        marker = self._write_marker(
            image, outputs, dict(recovery_summary), dict(profile_decision), origin="inferred",
        )
        self._discard_backup(image)
        return marker


@dataclass(frozen=True)
class TaskMarkerRelocationResult:
    checked_markers: int
    updated_markers: int
    invalid_markers: int


def _marker_period_root(marker_path: Path) -> Path | None:
    for parent in marker_path.parents:
        if parent.parent.name.casefold() == "periods":
            return parent
    return None


def _recorded_outputs_valid(marker: dict, output_dir: Path) -> bool:
    outputs = marker.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False
    for row in outputs:
        if not isinstance(row, dict):
            return False
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            return False
        path = (output_dir / relative).resolve()
        try:
            path.relative_to(output_dir.resolve())
            if path.stat().st_size != int(row.get("size", -1)):
                return False
        except (OSError, TypeError, ValueError):
            return False
        valid, _reason = validate_output(path, str(row.get("kind") or ""))
        if not valid:
            return False
    return True


def relocate_task_image_markers(job_root: Path | str) -> TaskMarkerRelocationResult:
    """Rebind valid copied markers to current task-relative image identities."""
    root = _resolved(job_root)
    checked = updated_count = invalid = 0
    for marker_path in sorted(root.rglob("*.completed.json"), key=str):
        if marker_path.parent.name != ".resume":
            continue
        checked += 1
        marker, _reason = _read_json_object(marker_path)
        period_root = _marker_period_root(marker_path)
        if marker is None or period_root is None:
            invalid += 1
            continue
        input_row = marker.get("input") if isinstance(marker.get("input"), dict) else {}
        filename = Path(str(input_row.get("path") or "")).name
        if not filename:
            task_row = marker.get("task_image") if isinstance(marker.get("task_image"), dict) else {}
            filename = str(task_row.get("filename") or "")
        image = (period_root / "images" / filename).resolve()
        try:
            image.relative_to(root)
            current_identity = file_identity(image, sha256=bool(input_row.get("sha256")))
        except (OSError, ValueError):
            invalid += 1
            continue
        if not _same_identity(input_row, current_identity):
            invalid += 1
            continue
        current_task = _task_image_identity(image)
        stored_task = marker.get("task_image")
        if isinstance(stored_task, dict) and (
            current_task is None or any(
                str(stored_task.get(key) or "").casefold()
                != str(current_task.get(key) or "").casefold()
                for key in ("grid", "period", "relative_path", "filename")
            )
        ):
            invalid += 1
            continue
        output_dir = marker_path.parent.parent.resolve()
        if not _recorded_outputs_valid(marker, output_dir):
            invalid += 1
            continue
        target = marker_path.parent / f"{stable_image_key(image)}.completed.json"
        if target != marker_path and target.is_file():
            invalid += 1
            continue
        migrated = dict(marker)
        migrated["image_key"] = stable_image_key(image)
        migrated["input"] = {**input_row, "path": str(image)}
        if current_task is not None:
            migrated["task_image"] = current_task
        summary = dict(migrated.get("summary") or {})
        for name in ("recovery_summary", "profile_decision"):
            row = summary.get(name)
            if isinstance(row, dict) and "image" in row:
                summary[name] = {**row, "image": str(image)}
        migrated["summary"] = summary
        if target != marker_path or migrated != marker:
            ImageResumeManager._atomic_json(target, migrated)
            if target != marker_path:
                marker_path.unlink(missing_ok=True)
            updated_count += 1
    return TaskMarkerRelocationResult(checked, updated_count, invalid)


def marker_summaries(marker: dict) -> tuple[dict, dict]:
    summary = marker.get("summary") if isinstance(marker.get("summary"), dict) else {}
    recovery = dict(summary.get("recovery_summary") or {})
    profile = dict(summary.get("profile_decision") or {})
    return recovery, profile
