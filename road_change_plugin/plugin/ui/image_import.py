"""Small, read-only image import helpers. No raster/GIS dependencies."""
from datetime import date
from pathlib import Path
import re
from .project_browser import RASTER_SUFFIXES, natural

FILE_FILTER = '影像与清单 (*.txt *.tif *.tiff *.img *.jp2 *.vrt)'
SUPPORTED = RASTER_SUFFIXES | {'.txt'}
DATE = re.compile(r'(?<!\d)(\d{4})(?P<sep>[-_]?)(\d{2})(?P=sep)(\d{2})(?!\d)')


def dates(text):
    found = set()
    for match in DATE.finditer(text):
        try:
            found.add(date(int(match[1]), int(match[3]), int(match[4])).strftime('%Y%m%d'))
        except ValueError:
            pass
    return found


def identify(*levels):
    """First nonempty evidence wins; ambiguous evidence requires user correction."""
    for values in levels:
        found = set().union(*(dates(value) for value in values))
        if found:
            return (next(iter(found)), '') if len(found) == 1 else ('', '检测到多个日期，请填写该期名称')
    return '', '未识别到日期，请填写期次名称'


def read_list(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('影像清单超过 8 MB，请拆分后导入')
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('gb18030')
    entries = []
    for line in text.splitlines():
        value = line.strip().strip('"')
        if value and not value.startswith('#'):
            entries.append((path.parent / value).resolve())
    if not entries:
        raise ValueError('影像清单为空')
    return entries


def folder_files(directory):
    """Only the selected directory and its immediate child directories."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError('影像目录不存在')
    children = list(root.iterdir())
    files = [p for p in children if p.is_file() and p.suffix.lower() in SUPPORTED]
    for child in sorted(children, key=lambda p: natural(p.name)):
        if child.is_dir() and not child.is_symlink():
            files.extend(p for p in child.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    return sorted(files, key=lambda p: natural(str(p)))


def import_periods(paths):
    """Return editable drafts. TXT lists stay intact; direct rasters share dates."""
    paths = list(dict.fromkeys(Path(p).resolve() for p in paths))
    rows, covered, groups, errors = [], set(), {}, []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            errors.append(f'{path.name}：文件不存在或不是支持的影像/清单格式')
    for path in paths:
        if path.suffix.lower() != '.txt' or not path.is_file():
            continue
        try:
            images = read_list(path)
            name, reason = identify([p.stem for p in images], [path.stem],
                                    [p.parent.name for p in images], [path.parent.name])
            covered.update(images)
            rows.append(dict(name=name, source=str(path), count=len(images),
                             origin=path.name, reason=reason, error=''))
        except (OSError, ValueError, UnicodeError) as exc:
            rows.append(dict(name='', source=str(path), count=0, origin=path.name,
                             reason='', error=f'{path.name}：{exc}'))
    for path in paths:
        if path.suffix.lower() not in RASTER_SUFFIXES or not path.is_file() or path in covered:
            continue
        name, reason = identify([path.stem], [path.parent.name])
        key = ('date', name) if name else ('unknown', str(path.parent), reason)
        row = groups.setdefault(key, dict(name=name, paths=[], reason=reason))
        row['paths'].append(path)
    for row in groups.values():
        files = row['paths']
        rows.append(dict(name=row['name'], source='\n'.join(map(str, files)), count=len(files),
                         origin='、'.join(p.name for p in files), reason=row['reason'], error=''))
    rows.sort(key=lambda row: (not bool(row['name']), natural(row['name']), natural(row['origin'])))
    skipped = sum(p in covered for p in paths if p.suffix.lower() in RASTER_SUFFIXES)
    if skipped:
        errors.append(f'{skipped} 幅影像已包含在所选 TXT 中，未重复添加')
    return rows, errors
