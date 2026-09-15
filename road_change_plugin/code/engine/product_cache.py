"""Completion caches for deterministic GIS exports, invalidated by file changes."""
import json
from pathlib import Path


def signature(paths):
    result = []
    for value in paths:
        if not value:
            continue
        path = Path(value).resolve()
        members = sorted(path.parent.glob(path.stem+'.*')) if path.suffix.lower() == '.shp' else [path]
        for member in members:
            if member.is_file():
                stat = member.stat()
                result.append([str(member), stat.st_size, stat.st_mtime_ns])
        if not path.is_file():
            result.append([str(path), None, None])
    return result


def read_completed(path, inputs):
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not {'inputs', 'outputs', 'paths', 'result'} <= data.keys():
        return None
    if data['inputs'] == inputs and data['outputs'] == signature(data['paths']):
        return data['result']
    return None


def write_completed(path, inputs, result, paths):
    paths = [str(p) for p in paths]
    data = dict(inputs=inputs, result=result, paths=paths, outputs=signature(paths))
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
