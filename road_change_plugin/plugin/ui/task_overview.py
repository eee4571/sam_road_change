"""Read-only stage overview from published products and task events."""
from pathlib import Path
import json

from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel


STAGES = ('道路提取', '测宽与道路面', '变化检测')


def recorded_stage(payloads):
    """Restore only explicit backend stage events, never guess from log text."""
    for payload in reversed(payloads):
        try:
            event = json.loads(payload.get('message', ''))
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get('kind') in {'pipeline', 'stage', 'prepare'} and event.get('stage'):
            return str(event['stage'])
    return '—'


def stage_index(stage):
    if any(word in stage for word in ('变化', '长时序', '评价', '汇总')):
        return 2
    if any(word in stage for word in ('测宽', '道路面', '宽度')):
        return 1
    if any(word in stage for word in ('道路提取', '中心线')):
        return 0
    return None


def product_inventory(descriptor):
    manifest = (descriptor or {}).get('data', {})
    base = Path((descriptor or {}).get('path', '.')).parent
    def exists(value):
        if not isinstance(value, str) or not value:
            return False
        path = Path(value)
        return (path if path.is_absolute() else base / path).is_file()
    periods = manifest.get('final_period_results', manifest.get('period_results', []))
    changes = manifest.get('change_results', [])
    def product(entry, key):
        published = entry.get('published') or entry.get('layers') or {}
        return published.get(key, entry.get(key))
    groups = [[[product(e, key) for key in keys] for e in periods]
              for keys in (('centerlines',), ('surfaces', 'width_segments'))]
    groups.append([[product(e, 'changes')] for e in changes])
    counts = [sum(all(exists(path) for path in paths) for paths in group) for group in groups]
    missing = [any(path and not exists(path) for paths in group for path in paths) for group in groups]
    orders = manifest.get('period_orders')
    if orders:
        plans = [(value.get('period_order', []), value.get('change_pairs'))
                 if isinstance(value, dict) else (value, None) for value in orders.values()]
    else:
        plans = [(list(value), None) for value in manifest.get('input_spec', {}).get('grids', {}).values()]
    expected_periods = sum(len(order) for order, _ in plans)
    expected_changes = sum(len(pairs) if pairs is not None else max(0, len(order)-1)
                           for order, pairs in plans)
    totals = [max(len(periods), expected_periods)] * 2 + [max(len(changes), expected_changes)]
    return counts, totals, missing


def stage_states(descriptor, operation, *, busy=False, active_stage='', failed=False, outdated=False, inventory=None):
    manifest = (descriptor or {}).get('data', {})
    counts, totals, missing = product_inventory(descriptor) if inventory is None else inventory
    complete = not failed and operation.get('status') == 'completed' and manifest.get('fast_finalization_state') != 'pending'
    active = stage_index(active_stage)
    result = []
    for i, (done, total) in enumerate(zip(counts, totals)):
        if busy:
            # A completed period does not imply every period finished this stage.
            text = '进行中' if active == i else '等待 / 已有成果' if done else '等待处理'
        elif failed and active == i:
            text = '失败'
        elif missing[i]:
            text = '成果失效'
        elif outdated and done:
            text = '成果待更新'
        elif complete and total and done == total:
            text = '已完成'
        elif done:
            text = f'已有 {done}/{total}'
        elif complete and i == 2 and not total:
            text = '无变化对'
        else:
            text = '未运行'
        result.append(text)
    return result


class StageOverview(QWidget):
    def __init__(self):
        super().__init__()
        self._descriptor = None
        self._inventory = product_inventory(None)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setSpacing(6)
        self.values = []
        for i, title in enumerate(STAGES):
            box = QWidget()
            layout = QVBoxLayout(box)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            caption, state = QLabel(title), QLabel('未运行')
            for label in (caption, state):
                label.setWordWrap(True)
                label.setMinimumWidth(0)
                layout.addWidget(label)
            state.setProperty('role', 'secondary')
            self.values.append(state)
            grid.addWidget(box, 0, i)
            grid.setColumnStretch(i, 1)

    def refresh(self, descriptor, operation, **kwargs):
        # Refresh file evidence only when the manifest is reloaded, not every timer tick.
        if descriptor is not self._descriptor:
            self._descriptor = descriptor
            self._inventory = product_inventory(descriptor)
        for label, text in zip(self.values, stage_states(descriptor, operation, inventory=self._inventory, **kwargs)):
            label.setText(text)
            status = ('success' if text == '已完成' else 'running' if text == '进行中'
                      else 'error' if text == '失败' else 'warning' if text in {'成果失效', '成果待更新'} else 'neutral')
            if label.property('status') != status:
                label.setProperty('status', status)
                label.style().unpolish(label)
                label.style().polish(label)
