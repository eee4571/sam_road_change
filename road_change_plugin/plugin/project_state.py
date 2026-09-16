"""One current project operation and publication snapshot; no algorithm imports."""
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import stat
import tempfile
import uuid
from .result_parser import read_results


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        temp = Path(stream.name)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def safe_name(value):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(value).strip()).strip('._-') or '未命名'


def is_link(path):
    if path.is_symlink():
        return True
    # Path.is_junction is only available on newer host Python versions.
    try:
        return bool(getattr(path.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))
    except FileNotFoundError:
        return False


def legacy_current(root, output):
    """Import just the former current index, never enumerate historical runs."""
    for path in (root / '_work/tasks/latest_pipeline.json', output / 'latest_pipeline.json'):
        if path.is_file():
            data = read_json(path)
            return path, data
    return None, {}


class ProjectState:
    def __init__(self, root, output=None):
        self.root = Path(root).resolve()
        self.output = Path(output or self.root / '成果输出').resolve()
        self.work = self.root / '_work'
        self.current = self.work / 'current'
        self.path = self.work / 'project_state.json'
        self.results_path = self.work / 'current_results.json'

    def configuration_signature(self):
        path = self.root / 'project_config.json'
        config = read_json(path) if path.is_file() else {}
        keys = ('validation_areas', 'area_periods', 'area_truths',
                'area_irmad_references', 'output_root', 'plugin_processing_parameters')
        content = {key: config.get(key) for key in keys}
        return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()

    def configuration_status(self):
        signature = self.state().get('configuration_signature')
        if not signature:
            return 'unknown'
        return 'current' if signature == self.configuration_signature() else 'changed'

    def remember_configuration(self):
        """Before the first edit of a legacy project, retain its old configuration."""
        state = self.state()
        if state and not state.get('configuration_signature'):
            state['configuration_signature'] = self.configuration_signature()
            write_json(self.path, state)

    def state(self):
        if self.path.is_file():
            return read_json(self.path)
        path = self.current / 'pipeline_result.json'
        if not path.is_file() and (self.current / 'job_state.json').is_file():
            path = self.current / 'job_state.json'
        if not path.is_file():
            path, data = legacy_current(self.root, self.output)
        else:
            data = read_json(path)
        if not path:
            return {}
        return dict(status=data.get('status', 'unknown'), manifest=str(path), run_id=data.get('run_id', ''),
                    action='all', legacy=not path.is_relative_to(self.current))

    def manifest(self):
        state = self.state()
        value = state.get('manifest')
        path = Path(value) if value else self.current / 'pipeline_result.json'
        fallback = path.parent / 'job_state.json'
        return fallback if not path.is_file() and fallback.is_file() else path

    def descriptor(self):
        path = self.manifest()
        if not path.is_file():
            return None
        data = read_json(path)
        return dict(path=str(path), id=data.get('run_id', ''), status=self.state().get('status', data.get('status', 'unknown')), data=data)

    def resumable(self):
        state = self.state()
        if state.get('status') not in {'running', 'queued', 'failed', 'cancelled', 'completed_with_errors'}:
            return False
        manifest = self.manifest()
        if state.get('action', 'all') != 'all':
            return manifest.is_file()
        data = read_json(manifest) if manifest.is_file() else {}
        job = Path(data.get('job_root') or manifest.parent)
        return (job / 'job_state.json').is_file()

    def results(self):
        if self.results_path.is_file():
            return [p for p in read_json(self.results_path) if Path(p['path']).is_file()]
        manifest = self.manifest()
        return read_results(manifest) if manifest.is_file() else []

    def _protected(self, data):
        paths = []
        for key in ('areas', 'periods', 'truths'):
            for row in data.get(key, []):
                if not row[-1]:
                    continue
                source = Path(row[-1]).resolve()
                paths.append(source)
                if source.suffix.lower() == '.txt' and source.is_file():
                    raw = source.read_bytes()
                    try:
                        content = raw.decode('utf-8-sig')
                    except UnicodeDecodeError:
                        content = raw.decode('gb18030')
                    for line in content.splitlines():
                        line = line.strip().strip('"')
                        if line and not line.startswith('#'):
                            paths.append((source.parent / line).resolve())
        return paths + [self.root / 'project_config.json', self.work / 'cache']

    def _validate_targets(self, targets, inputs):
        """Validate the complete deletion plan before touching any file."""
        if self.output == self.root or self.root.is_relative_to(self.output) or self.output.is_relative_to(self.work):
            raise ValueError('成果目录与项目或工作目录重叠，无法安全重跑')
        protected = self._protected(inputs) + self._protected(self.state().get('parameters', {}))
        checked = []
        for target in targets:
            target = Path(target).absolute()
            resolved = target.resolve()
            if not (resolved.is_relative_to(self.work) or resolved.is_relative_to(self.output)):
                raise ValueError(f'运行文件超出项目管理范围：{target}')
            if resolved in {self.work, self.root, self.output}:
                raise ValueError(f'不能清理整个目录：{target}')
            if any(p == resolved or p.is_relative_to(resolved) or resolved.is_relative_to(p) for p in protected):
                raise ValueError(f'运行目录包含输入数据或通用缓存：{target}')
            # Reject links/junctions, including ancestors and nested contents.
            for node in (target, *target.parents):
                if is_link(node):
                    raise ValueError(f'运行目录包含链接，不能清理：{node}')
                if node == self.root or node == self.output:
                    break
            if target.is_dir():
                for folder, dirs, files in os.walk(target, followlinks=False):
                    for name in dirs + files:
                        node = Path(folder) / name
                        if is_link(node) or not node.resolve().is_relative_to(resolved):
                            raise ValueError(f'运行目录包含越界链接：{node}')
            checked.append(resolved)
        return sorted(set(checked), key=lambda p: len(p.parts), reverse=True)

    def _delete(self, targets, inputs):
        for target in self._validate_targets(targets, inputs):
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

    @staticmethod
    def dataset_files(path):
        path = Path(path)
        if path.suffix.lower() == '.shp':
            return [path.with_suffix(suffix) for suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.sbn', '.sbx', '.shp.xml')]
        return [path]

    def fresh(self, data):
        old = self.descriptor()
        products = self.results()
        targets = [self.current, self.work / 'tasks', self.work / 'publication_history']
        targets += [p for product in products for p in self.dataset_files(product['path'])]
        index_path = self.output / 'result_index.json'
        areas = set()
        if old:
            areas.update(e.get('grid') for e in old['data'].get('period_results', []) if e.get('grid'))
        if index_path.is_file():
            index = read_json(index_path)
            if Path(index.get('project_root', str(self.root))).resolve() != self.root:
                raise ValueError('成果目录属于其他项目，不能覆盖')
            areas.update(index.get('areas', {}))
        for area in areas:
            for category in ('01_单期道路', '02_变化检测', '03_长时序', '04_精度评价'):
                targets.append(self.output / safe_name(area) / category)
        if old and old['data'].get('job_root'):
            targets.append(Path(old['data']['job_root']))
        targets += [self.output / name for name in ('result_index.json', 'task_report.csv', 'task_report.json', 'latest_pipeline.json')]
        self._delete(targets, data)
        state = dict(status='running', action='all', manifest=str(self.current / 'pipeline_result.json'),
                     run_id='current_' + uuid.uuid4().hex[:12], output=str(self.output), scope=None,
                     last_error='', parameters=copy.deepcopy(data))
        state['configuration_signature'] = self.configuration_signature()
        write_json(self.results_path, [])
        write_json(self.path, state)
        return state

    @staticmethod
    def in_scope(product, scope):
        meta = product.get('metadata', {})
        kind = product['result_type']
        if kind == 'road_evaluation':
            return True  # aggregate metrics depend on the changed inputs
        if str(meta.get('grid')) != scope['grid']:
            return False
        if kind in {'road_centerline', 'road_surface', 'road_width'}:
            return str(meta.get('period')) in scope['periods']
        if kind == 'road_change':
            return f"{meta.get('before_period')}_to_{meta.get('after_period')}" in scope['changes']
        return kind == 'road_temporal'

    @staticmethod
    def period_order(manifest, grid):
        """Frozen processing order, including failed/missing periods."""
        names = manifest.get('period_orders', {}).get(grid, {}).get('period_order', [])
        if not names:
            names = list(manifest.get('input_spec', {}).get('grids', {}).get(grid, {}))
            if not names:
                names = [str(e['period']) for e in manifest.get('period_results', []) if str(e.get('grid')) == grid]
            names = sorted(set(names), key=lambda v: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', v)])
        names = list(map(str, names))
        return names

    def local_scope(self, action, data):
        """Read-only impact preview, shared by UI and destructive preparation."""
        descriptor = self.descriptor()
        if not descriptor:
            raise ValueError('当前项目没有可更新的成果')
        if action not in {'rerun-period', 'rerun-change'}:
            raise ValueError('请选择更新期次或变化对')
        grid, period = data['grid'], data.get('period', '')
        names = self.period_order(descriptor['data'], grid)
        adjacent = list(zip(names, names[1:]))
        if action == 'rerun-period':
            if period not in names:
                raise ValueError(f'{grid}：当前成果中不存在期次 {period}')
            changes = [f'{before}_to_{after}' for i, (before, after) in enumerate(adjacent)
                       if period in names[max(0, i - 1):i + 3]]
        else:
            if (data['before_period'], data['after_period']) not in adjacent:
                raise ValueError(f'{grid}：当前成果中不存在所选变化对')
            changes = [f"{data['before_period']}_to_{data['after_period']}"]
        return dict(grid=grid, periods=[period] if action == 'rerun-period' else [], changes=changes)

    def local(self, action, data):
        scope = self.local_scope(action, data)
        descriptor = self.descriptor()
        manifest = descriptor['data']
        grid, period, changes = scope['grid'], data.get('period', ''), scope['changes']
        products = self.results()
        removed = [p for p in products if self.in_scope(p, scope)]
        targets = [path for p in removed for path in self.dataset_files(p['path'])]
        job = Path(manifest.get('job_root') or Path(descriptor['path']).parent).resolve()
        if action == 'rerun-period':
            targets.append(job / 'grids' / safe_name(grid) / 'periods' / safe_name(period))
        targets += [job / 'grids' / safe_name(grid) / 'changes' / safe_name(pair) for pair in changes]
        # Exact published category directories include auxiliary files and previews.
        targets += [self.output / safe_name(grid) / '01_单期道路' / safe_name(p) for p in scope['periods']]
        targets += [self.output / safe_name(grid) / '02_变化检测' / safe_name(pair) for pair in changes]
        targets.append(self.output / safe_name(grid) / '03_长时序')
        index_path = self.output / 'result_index.json'
        index = read_json(index_path) if index_path.is_file() else None
        if index:
            if Path(index.get('project_root', str(self.root))).resolve() != self.root:
                raise ValueError('成果目录属于其他项目，不能覆盖')
            for area in index.get('areas', {}):
                targets += [self.output / safe_name(area) / '04_精度评价',
                            self.output / safe_name(area) / '02_变化检测' / '精度评价']
        self._delete(targets, data)
        if index:
            node = index.get('areas', {}).get(grid, {})
            for p in scope['periods']:
                node.get('periods', {}).pop(p, None)
            for pair in scope['changes']:
                node.get('changes', {}).pop(pair, None)
            node['temporal'] = {}
            for area in index.get('areas', {}).values():
                area['evaluation'] = {}
            write_json(index_path, index)
        state = dict(status='running', action=action, manifest=descriptor['path'], run_id=descriptor['id'],
                     scope=scope, parameters=copy.deepcopy(data), output=str(self.output), last_error='')
        state['configuration_signature'] = self.state().get('configuration_signature')
        write_json(self.results_path, [p for p in products if not self.in_scope(p, scope)])
        write_json(self.path, state)
        return state

    def resume(self):
        if not self.resumable():
            raise ValueError('当前项目没有可继续的处理断点')
        state = self.state()
        state.update(status='running', last_error='')
        write_json(self.path, state)
        return state

    def note_error(self, error):
        state = self.state()
        state['last_error'] = error
        write_json(self.path, state)

    def finish(self, status, error=''):
        state = self.state()
        state.update(status=status, last_error=error)
        if status == 'completed':
            if not self.manifest().is_file():
                raise ValueError('处理结束但当前成果索引不存在')
            results = read_results(self.manifest())
            scope = state.get('scope')
            if scope:
                results = self.results() + [p for p in results if self.in_scope(p, scope)]
            unique = {(p['result_type'], p['path']): p for p in results}
            write_json(self.results_path, list(unique.values()))
        write_json(self.path, state)
