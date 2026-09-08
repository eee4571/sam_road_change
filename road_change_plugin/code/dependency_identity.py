from __future__ import annotations

"""Stable, path-independent identities for pipeline dependencies."""

import ast
import hashlib
import json
import os
from pathlib import Path


IDENTITY_SCHEMA_VERSION = 1
_RESOURCE_KEY_SUFFIXES = ("_PATH", "_ROOT", "_DIR")
_MODEL_RESOURCE_SUFFIXES = {".ckpt", ".pth", ".pt", ".onnx", ".safetensors"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cache(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_write_cache(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_file_identity(
    value: Path | str, *, content_hash: bool = False,
    cache_path: Path | str | None = None,
) -> dict:
    path = Path(value).expanduser().resolve()
    try:
        stat = path.stat()
    except OSError:
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "path": str(path), "name": path.name.casefold(),
            "size": None, "mtime_ns": None,
        }
    identity = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "path": str(path), "name": path.name.casefold(),
        "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
    }
    if not content_hash:
        return identity
    cache_file = Path(cache_path).expanduser().resolve() if cache_path else None
    cache = _read_cache(cache_file)
    key = os.path.normcase(str(path))
    cached = cache.get(key) if isinstance(cache.get(key), dict) else {}
    if (
        cached.get("size") == identity["size"]
        and cached.get("mtime_ns") == identity["mtime_ns"]
        and isinstance(cached.get("sha256"), str)
    ):
        identity["sha256"] = cached["sha256"]
        identity["sha256_cached"] = True
        return identity
    identity["sha256"] = _sha256(path)
    identity["sha256_cached"] = False
    if cache_file is not None:
        cache[key] = {
            "size": identity["size"], "mtime_ns": identity["mtime_ns"],
            "sha256": identity["sha256"],
        }
        _atomic_write_cache(cache_file, cache)
    return identity


def _scalar(text: str):
    value = text.strip()
    lowered = value.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [_scalar(item) for item in body.split(",")]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value.strip("'\"")


def _parsed_config_values(value: Path | str) -> dict:
    """Parse the repository's mapping-only YAML into stable dotted leaves.

    The production config contains scalar mappings and lists; this deliberately
    small parser keeps identity checks model-free and does not pretend to be a
    general YAML implementation.
    """
    path = Path(value).expanduser().resolve()
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    stack: list[tuple[int, str]] = []
    result: dict[str, object] = {}
    pending_list_key: str | None = None
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()
        if text.startswith("-") and pending_list_key:
            result.setdefault(pending_list_key, [])
            if isinstance(result[pending_list_key], list):
                result[pending_list_key].append(_scalar(text[1:].strip()))
            continue
        pending_list_key = None
        if ":" not in text:
            continue
        key, raw_value = text.split(":", 1)
        key = key.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([item[1] for item in stack] + [key])
        if not raw_value.strip():
            stack.append((indent, key))
            pending_list_key = dotted
            continue
        result[dotted] = _scalar(raw_value.strip())
    return result


def effective_config_values(value: Path | str) -> dict:
    return {
        key: item for key, item in _parsed_config_values(value).items()
        if not key.rsplit(".", 1)[-1].upper().endswith(_RESOURCE_KEY_SUFFIXES)
    }


def effective_config_identity(
    value: Path | str, *, hash_resources: bool = False,
    cache_path: Path | str | None = None, inspect_resources: bool = True,
) -> dict:
    path = Path(value).expanduser().resolve()
    values = effective_config_values(path)
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    base = stable_file_identity(path)
    base["effective_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    base["effective_key_count"] = len(values)
    resources = []
    for key, raw_value in _parsed_config_values(path).items():
        leaf = key.rsplit(".", 1)[-1].upper()
        if not leaf.endswith("_PATH") or not isinstance(raw_value, str):
            continue
        resource = Path(raw_value).expanduser()
        if resource.suffix.casefold() not in _MODEL_RESOURCE_SUFFIXES:
            continue
        if not resource.is_absolute():
            resource = path.parent / resource
        identity = (
            stable_file_identity(
                resource, content_hash=hash_resources, cache_path=cache_path,
            )
            if inspect_resources else {
                "schema_version": IDENTITY_SCHEMA_VERSION,
                "path": os.path.abspath(str(resource)), "name": resource.name.casefold(),
                "size": None, "mtime_ns": None,
            }
        )
        identity["config_key"] = key
        resources.append(identity)
    base["resource_identities"] = sorted(resources, key=lambda item: item["config_key"])
    return base


def dependency_identity_equal(previous: object, current: object, *, kind: str) -> bool:
    if previous == current:
        return True
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return previous == current
    if kind == "config":
        first, second = previous.get("effective_sha256"), current.get("effective_sha256")
        if first and second:
            if first != second:
                return False
            previous_resources = {
                row.get("config_key"): row for row in previous.get("resource_identities", [])
                if isinstance(row, dict) and row.get("config_key")
            }
            current_resources = {
                row.get("config_key"): row for row in current.get("resource_identities", [])
                if isinstance(row, dict) and row.get("config_key")
            }
            if previous_resources and current_resources:
                if set(previous_resources) != set(current_resources):
                    return False
                for key in previous_resources:
                    old, new = previous_resources[key], current_resources[key]
                    if old.get("sha256") and new.get("sha256"):
                        if old["sha256"] != new["sha256"]:
                            return False
                    else:
                        old_name = old.get("name") or Path(str(old.get("path") or "")).name.casefold()
                        new_name = new.get("name") or Path(str(new.get("path") or "")).name.casefold()
                        if old_name != new_name:
                            return False
            return True
    first_hash, second_hash = previous.get("sha256"), current.get("sha256")
    if first_hash and second_hash:
        return first_hash == second_hash
    stable_keys = ("name", "size", "mtime_ns")
    previous_name = previous.get("name") or Path(str(previous.get("path") or "")).name.casefold()
    current_name = current.get("name") or Path(str(current.get("path") or "")).name.casefold()
    previous_stable = (previous_name, previous.get("size"), previous.get("mtime_ns"))
    current_stable = (current_name, current.get("size"), current.get("mtime_ns"))
    return previous_stable == current_stable and all(
        value is not None for value in previous_stable[1:] + current_stable[1:]
    )


def ordinary_file_identity_equal(previous: object, current: object) -> bool:
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return previous == current
    if "name" in previous or "name" in current:
        previous_name = previous.get("name") or Path(str(previous.get("path") or "")).name.casefold()
        current_name = current.get("name") or Path(str(current.get("path") or "")).name.casefold()
        if previous_name != current_name:
            return False
    keys = sorted((set(previous) | set(current)) - {"path", "schema_version", "name"})
    return all(previous.get(key) == current.get(key) for key in keys)
