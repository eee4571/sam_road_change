from __future__ import annotations

"""Shared offline input-list decoding and period ordering utilities."""

import locale
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


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
    encodings.extend(("mbcs", "cp1252"))
    unique: list[str] = []
    for encoding in encodings:
        if encoding and encoding.casefold() not in {item.casefold() for item in unique}:
            unique.append(encoding)
    return unique


def decode_text_auto(path: Path | str) -> tuple[str, str]:
    """Decode common GIS path-list encodings without replacement characters."""
    source = Path(path).expanduser().resolve()
    data = source.read_bytes()
    failures: list[str] = []
    for encoding in _candidate_encodings(data):
        try:
            text = data.decode(encoding, errors="strict")
        except (LookupError, UnicodeError) as exc:
            failures.append(f"{encoding}: {exc}")
            continue
        if "\x00" in text:
            failures.append(f"{encoding}: decoded text contains NUL characters")
            continue
        return text, encoding
    detail = "; ".join(failures[:5])
    raise UnicodeError(f"无法识别 TXT 编码：{source}。尝试结果：{detail}")


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
) -> PathList:
    """Read a TXT path list and retain actionable line-level diagnostics."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"影像路径 TXT 不存在：{source}")
    text, encoding = decode_text_auto(source)
    roots = [source.parent]
    roots.extend(Path(root).expanduser().resolve() for root in search_roots)
    entries: list[ListedPath] = []
    missing: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        value = _clean_list_line(raw_line)
        if not value:
            continue
        candidate = Path(value).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
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
            f"TXT：{source}\n{preview}{suffix}"
        )
    if not entries:
        raise ValueError(f"影像路径 TXT 没有有效路径（检测编码：{encoding}）：{source}")
    return PathList(source=source, encoding=encoding, entries=tuple(entries))


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
