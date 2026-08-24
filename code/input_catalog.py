from __future__ import annotations

"""Shared offline input-list decoding and period ordering utilities."""

import json
import locale
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ListedPath:
    line_number: int
    raw_value: str
    path: Path


@dataclass(frozen=True)
class PathList:
    source: Path
    encoding: str
    entries: tuple[ListedPath, ...]
    attempted_encodings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PeriodInfo:
    value: str
    normalized: str
    kind: str
    sort_key: tuple


def _candidate_encodings(data: bytes) -> list[str]:
    encodings: list[str] = []
    if data.startswith(b"\xef\xbb\xbf"):
        encodings.append("utf-8-sig")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.append("utf-16")
    encodings.append("utf-8-sig")
    if b"\x00" in data[:256]:
        encodings.extend(("utf-16", "utf-16-le", "utf-16-be"))
    encodings.extend(("gb18030", "gbk"))
    preferred = locale.getpreferredencoding(False)
    if preferred:
        encodings.append(preferred)
    encodings.extend(("mbcs", "cp932", "cp950", "big5", "cp1252"))
    unique: list[str] = []
    for encoding in encodings:
        if encoding and encoding.casefold() not in {item.casefold() for item in unique}:
            unique.append(encoding)
    return unique


def _candidate_path_lines(text: str, limit: int = 50) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        value = _clean_list_line(raw_line)
        if value:
            rows.append((line_number, value))
            if len(rows) >= limit:
                break
    return rows

def _configured_path_relocations(
    explicit: Mapping[str, str] | None = None,
) -> list[tuple[Path, Path]]:
    if explicit is None:
        try:
            raw = json.loads(os.environ.get("SAMROAD_PATH_RELOCATIONS", "{}"))
        except json.JSONDecodeError:
            return []
    else:
        raw = explicit
    if not isinstance(raw, dict):
        return []
    mappings = []
    for old, new in raw.items():
        if not str(old).strip() or not str(new).strip():
            continue
        mappings.append((Path(str(old)).expanduser().resolve(), Path(str(new)).expanduser().resolve()))
    return sorted(mappings, key=lambda item: len(item[0].parts), reverse=True)

def _relocated_path(
    path: Path, relocations: Mapping[str, str] | None = None,
) -> Path | None:
    if not path.is_absolute():
        return None
    for old_root, new_root in _configured_path_relocations(relocations):
        try:
            relative = path.resolve().relative_to(old_root)
        except ValueError:
            continue
        return (new_root / relative).resolve()
    return None

def _path_choices(
    candidate: Path, roots: list[Path], relocations: Mapping[str, str] | None = None,
) -> list[Path]:
    if not candidate.is_absolute():
        return [root / candidate for root in roots]
    relocated = _relocated_path(candidate, relocations)
    return [candidate, relocated] if relocated is not None else [candidate]


def _path_hit_count(
    lines: list[tuple[int, str]], roots: list[Path],
    relocations: Mapping[str, str] | None = None,
) -> int:
    hits = 0
    for _line_number, value in lines:
        candidate = Path(value).expanduser()
        choices = _path_choices(candidate, roots, relocations)
        if any(choice.is_file() for choice in choices):
            hits += 1
    return hits


def decode_text_auto(
    path: Path | str, *, search_roots: Iterable[Path | str] = (),
    encoding_override: str | None = None,
    path_relocations: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Score every strict decode using real path hits and text quality."""
    source = Path(path).expanduser().resolve()
    data = source.read_bytes()
    roots = [source.parent, *(Path(root).expanduser().resolve() for root in search_roots)]
    if encoding_override:
        try:
            text = data.decode(encoding_override, errors="strict")
        except (LookupError, UnicodeError) as exc:
            raise UnicodeError(
                f"指定的 TXT 编码无法读取文件：{source}；编码：{encoding_override}；错误：{exc}"
            ) from exc
        if "\x00" in text:
            raise UnicodeError(f"指定编码解码后包含 NUL：{source}；编码：{encoding_override}")
        return text, encoding_override
    failures: list[str] = []
    successes: list[tuple[tuple[int, int, int, int, int], str, str]] = []
    encodings = _candidate_encodings(data)
    for priority, encoding in enumerate(encodings):
        try:
            text = data.decode(encoding, errors="strict")
        except (LookupError, UnicodeError) as exc:
            failures.append(f"{encoding}: {exc}")
            continue
        if "\x00" in text:
            failures.append(f"{encoding}: decoded text contains NUL characters")
            continue
        lines = _candidate_path_lines(text)
        hits = _path_hit_count(lines, roots, path_relocations)
        bad_controls = sum(
            1 for character in text
            if unicodedata.category(character).startswith("C") and character not in "\r\n\t"
        )
        western_fallback_penalty = 1 if encoding.casefold() in {"cp1252", "latin-1"} else 0
        score = (hits, int(bool(lines)), len(lines), -bad_controls - western_fallback_penalty, -priority)
        successes.append((score, text, encoding))
    if successes:
        _score, text, encoding = max(successes, key=lambda item: item[0])
        return text, encoding
    detail = "; ".join(failures[:8])
    attempted = "、".join(encodings)
    raise UnicodeError(
        f"无法可靠识别影像路径 TXT 的编码。文件：{source}。"
        f"已尝试：{attempted}。解码结果：{detail}"
    )


def _clean_list_line(value: str) -> str:
    value = value.strip().strip("\ufeff")
    if not value or value.startswith("#"):
        return ""
    if value[:1] in {'"', "'"} and value[-1:] == value[:1]:
        value = value[1:-1].strip()
    return value


def read_path_list(
    path: Path | str,
    *,
    search_roots: Iterable[Path | str] = (),
    require_files: bool = True,
    encoding_override: str | None = None,
    path_relocations: Mapping[str, str] | None = None,
) -> PathList:
    """Read a TXT path list and retain actionable line-level diagnostics."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"影像路径 TXT 不存在：{source}")
    roots = [source.parent]
    roots.extend(Path(root).expanduser().resolve() for root in search_roots)
    if encoding_override is None:
        try:
            overrides = json.loads(os.environ.get("SAMROAD_TXT_ENCODINGS", "{}"))
        except json.JSONDecodeError:
            overrides = {}
        if isinstance(overrides, dict):
            encoding_override = overrides.get(str(source)) or overrides.get(str(path))
    attempted_encodings = tuple(_candidate_encodings(source.read_bytes()))
    text, encoding = decode_text_auto(
        source, search_roots=roots[1:], encoding_override=encoding_override,
        path_relocations=path_relocations,
    )
    entries: list[ListedPath] = []
    missing: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        value = _clean_list_line(raw_line)
        if not value:
            continue
        candidate = Path(value).expanduser()
        candidates = _path_choices(candidate, roots, path_relocations)
        resolved = next((item.resolve() for item in candidates if item.is_file()), candidates[0].resolve())
        if require_files and not resolved.is_file():
            missing.append(f"第 {line_number} 行：{value} → {resolved}")
            continue
        entries.append(ListedPath(line_number=line_number, raw_value=value, path=resolved))
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n……另有 {len(missing) - 20} 条"
        raise FileNotFoundError(
            f"影像路径 TXT 中有 {len(missing)} 条路径不存在（检测编码：{encoding}）：\n"
            f"TXT：{source}\n已尝试编码：{'、'.join(attempted_encodings)}\n{preview}{suffix}"
        )
    if not entries:
        raise ValueError(f"影像路径 TXT 没有有效路径（检测编码：{encoding}）：{source}")
    return PathList(
        source=source, encoding=encoding, entries=tuple(entries),
        attempted_encodings=attempted_encodings,
    )


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def parse_period(value: str) -> PeriodInfo:
    """Return a validated calendar key for YYYY/YYYYMM/YYYYMMDD when possible."""
    raw = str(value).strip()
    match = re.fullmatch(r"(\d{4})", raw)
    if match:
        year = int(match.group(1))
        parsed = date(year, 1, 1)
        return PeriodInfo(raw, f"{year:04d}", "year", (0, parsed.toordinal(), 0, raw))
    match = re.fullmatch(r"(\d{4})(\d{2})", raw)
    if match:
        year, month = (int(item) for item in match.groups())
        parsed = date(year, month, 1)
        return PeriodInfo(raw, f"{year:04d}{month:02d}", "month", (0, parsed.toordinal(), 1, raw))
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", raw)
    if match:
        year, month, day = (int(item) for item in match.groups())
        parsed = date(year, month, day)
        return PeriodInfo(raw, parsed.strftime("%Y%m%d"), "date", (0, parsed.toordinal(), 2, raw))
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if match:
        year, month, day = (int(item) for item in match.groups())
        parsed = date(year, month, day)
        return PeriodInfo(raw, parsed.strftime("%Y%m%d"), "date", (0, parsed.toordinal(), 2, raw))
    if not raw:
        raise ValueError("期次名称不能为空")
    return PeriodInfo(raw, raw, "custom", (1, natural_key(raw)))


def period_sort_key(value: str) -> tuple:
    return parse_period(value).sort_key


def ordered_periods(values: Iterable[str]) -> list[str]:
    return sorted((str(value).strip() for value in values), key=period_sort_key)


def period_order_manifest(values: Iterable[str]) -> dict:
    ordered = ordered_periods(values)
    info = [parse_period(value) for value in ordered]
    return {
        "period_order": ordered,
        "change_pairs": [[before, after] for before, after in zip(ordered, ordered[1:])],
        "periods": [
            {"value": item.value, "normalized": item.normalized, "kind": item.kind}
            for item in info
        ],
        "custom_order_warning": any(item.kind == "custom" for item in info),
    }
