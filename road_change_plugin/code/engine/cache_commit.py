"""Recoverable IR-MAD cache publication; standard-library file I/O only."""
import gc
import json
from pathlib import Path
import time
import uuid

RETRY_DELAYS = (.1, .2, .4, .8, 1.6, 2.0)
_INTERNAL = {'prepared.json', 'complete.json', 'failure.json', 'paired_valid_rgb.bin'}


def retry_locked(operation, target):
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return operation()
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in {5, 32, 33}:
                raise
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(f'缓存提交被 Windows 拒绝，重试后仍失败：{target}；'
                                   f'请检查文件占用或目录权限后重试；缓存不会因本次错误被删除。原始错误：{exc}') from exc
            if attempt == 0:
                gc.collect()
            time.sleep(RETRY_DELAYS[attempt])


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        retry_locked(lambda: temporary.replace(path), path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # never replace the original publication error with cleanup noise


def prepared_record(folder, identity):
    """Return a verified ready record, or None for incomplete/changed attempts."""
    folder = Path(folder)
    try:
        if folder.is_symlink():
            return None
        record = json.loads((folder / 'prepared.json').read_text(encoding='utf8'))
        if record.get('identity') != identity or not record.get('files'):
            return None
        for item in record['files']:
            relative = Path(item['relative'])
            path = folder / relative
            if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(folder.resolve()):
                return None
            info = path.stat()
            if not path.is_file() or (info.st_size, info.st_mtime_ns) != (item['size'], item['mtime_ns']):
                return None
        return record
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def find_prepared(root, identity):
    root = Path(root)
    if root.exists():
        if prepared_record(root, identity):
            return root  # rename succeeded, but writing complete.json was interrupted
        raise FileExistsError(f'IR-MAD incomplete or concurrent cache: {root}')
    for candidate in sorted(root.parent.glob(root.name + '.pending-*')):
        if prepared_record(candidate, identity):
            return candidate
    return None


def seal_prepared(attempt, root, identity):
    """Called only after every normalized output and audit has been closed."""
    attempt, root = Path(attempt), Path(root)
    audit_path = attempt / 'normalization.json'
    audit = json.loads(audit_path.read_text(encoding='utf8'))
    for row in audit['outputs']:
        row['output'] = str(root / 'normalized_tiles' / Path(row['output']).name)
    atomic_json(audit_path, audit)
    files = []
    for path in sorted(attempt.rglob('*')):
        if path.is_file() and path.name not in _INTERNAL and not (path.name.startswith('.') and path.suffix == '.tmp'):
            info = path.stat()
            files.append(dict(relative=path.relative_to(attempt).as_posix(), size=info.st_size, mtime_ns=info.st_mtime_ns))
    atomic_json(attempt / 'prepared.json', dict(identity=identity, files=files))


def commit_prepared(attempt, root, identity):
    attempt, root = Path(attempt), Path(root)
    record = prepared_record(attempt, identity)
    if record is None:
        raise ValueError(f'IR-MAD 待提交缓存不完整或文件已改变：{attempt}')
    if attempt != root:
        def rename():
            if root.exists():
                raise FileExistsError(f'IR-MAD incomplete or concurrent cache: {root}')
            attempt.rename(root)
        retry_locked(rename, f'{attempt} → {root}')
    if prepared_record(root, identity) is None:
        raise ValueError(f'IR-MAD 提交后缓存校验失败：{root}')
    files = [dict(path=str((root / row['relative']).resolve()), size=row['size'], mtime_ns=row['mtime_ns'])
             for row in record['files']]
    atomic_json(root / 'complete.json', dict(identity=identity, files=files))
