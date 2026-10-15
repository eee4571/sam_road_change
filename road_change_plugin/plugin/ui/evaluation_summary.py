"""Presentation of the formal aggregate report; never computes new metrics."""
import json
import math
from datetime import datetime
from pathlib import Path


def report_metrics(path):
    overall = {}
    report = {}
    try:
        if path.stat().st_size <= 2 * 1024 * 1024:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = report.get("metrics", [])
            overall = next((row for row in rows if isinstance(row, dict) and row.get("class") == "all"), {})
    except (OSError, ValueError, TypeError, AttributeError):
        pass

    def number(key):
        value = overall.get(key)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None

    def percent(key):
        value = number(key)
        return f'{value:.1%}' if value is not None and value <= 1 else '—'

    fields = [('centerline_avg_offset_m', 'm'), ('centerline_mean_offset_px', 'px')]
    if overall.get('centerline_offset_unit') == 'px' or 'centerline_mean_offset_px' in overall:
        # Different estimators are not interchangeable when the requested one failed.
        fields = [('centerline_mean_offset_px', 'px')]
    offset, unit = next(((number(key), unit) for key, unit in fields if number(key) is not None), (None, ''))
    distance = f'{offset:.2f} {unit}' if offset is not None else '—'
    note = str(overall.get('centerline_offset_reason') or '')
    tolerance = number('centerline_match_tolerance_px')
    if tolerance is not None:
        note = f'中心线匹配容差：{tolerance:g} px。' + note
    return dict(recall=percent('change_recall'), precision=percent('change_precision'),
                completeness=percent('road_centerline_completeness'), offset=distance,
                type_accuracy=percent('change_type_accuracy'), centerline_note=note, report=report)


def metrics_text(path):
    values = report_metrics(path)
    return '\n'.join(f'{title}  {values[key]}' for title, key in (
        ('变化图斑查全率', 'recall'), ('变化图斑查准率', 'precision'),
        ('变化道路提取完整度', 'completeness'), ('中心线平均偏移距离', 'offset'), ('变化类型正确率', 'type_accuracy')))


def evaluation_context(descriptor, path):
    """Report only recorded provenance; file time is explicitly labelled as such."""
    manifest = (descriptor or {}).get('data', {})
    changes = manifest.get('change_results', [])
    evaluated = [e for e in changes if e.get('evaluation_state') == 'completed']
    areas = sorted({str(e.get('grid')) for e in evaluated if e.get('grid')})
    scopes = '\n'.join(f"{e.get('grid', '')} / {e.get('before_period', '')} → {e.get('after_period', '')}"
                       for e in evaluated)
    truths = manifest.get('input_spec', {}).get('truths', {})
    truth_lines = []
    for key, value in truths.items():
        value = value if isinstance(value, dict) else {'path': value}
        version = value.get('sha256') or value.get('hash') or value.get('version')
        truth_lines.append(str(key).replace('\0', ' / ') + '：' + str(value.get('path', '未记录路径'))
                           + ('（版本 ' + str(version) + '）' if version else '（版本未记录）'))
    reviewed = sum(bool(e.get('manual_review')) for e in evaluated)
    source = (f'{len(evaluated)} 个已评价变化对，其中 {reviewed} 个已应用人工复核'
              if evaluated else '报告未记录逐变化对评价状态')
    try:
        updated = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    except OSError:
        updated = '未知'
    return '\n'.join(('评价区域：' + ('、'.join(areas) or '未记录'), source,
        '评价范围：\n' + (scopes or '未记录'),
        '评价数据：\n' + ('\n'.join(truth_lines) or '未记录'),
        '报告文件更新时间：' + updated,
        '成果版本：' + str(manifest.get('manual_review_updated_at') or manifest.get('updated_at') or manifest.get('run_id') or '未记录'),
        '不同成果版本的评价不能直接视为同一结果。', '报告路径：' + str(path)))


def show_evaluation(parent, descriptor, path):
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QPlainTextEdit, QPushButton
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtCore import QUrl
    dialog = QDialog(parent)
    dialog.setWindowTitle('评价报告')
    dialog.resize(680, 530)
    layout = QVBoxLayout(dialog)
    details = QPlainTextEdit()
    details.setReadOnly(True)
    values = report_metrics(path) if path.suffix.lower() == '.json' else {}
    sections = ('变化检测\n' + '\n'.join(f'{name}：{values.get(key, "未评价")}' for name, key in
                (('查全率', 'recall'), ('查准率', 'precision'), ('类型正确率', 'type_accuracy')))
                + '\n\n道路提取\n' + '\n'.join(f'{name}：{values.get(key, "未评价")}' for name, key in
                (('道路完整度', 'completeness'), ('中心线平均偏移', 'offset'))))
    details.setPlainText(sections + '\n' + values.get('centerline_note', '')
                         + '\n\n评价依据\n' + evaluation_context(descriptor, path))
    layout.addWidget(details, 1)
    open_file = QPushButton('打开原始评价报告')
    open_file.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))
    layout.addWidget(open_file)
    dialog.exec()
