"""Presentation of the formal aggregate report; never computes new metrics."""
import json
import math


def metrics_text(path):
    overall = {}
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
    if overall.get('centerline_offset_unit') == 'px':
        fields.reverse()
    offset, unit = next(((number(key), unit) for key, unit in fields if number(key) is not None), (None, ''))
    distance = f'{offset:.2f} {unit}' if offset is not None else '—'
    return '\n'.join((
        f"变化图斑查全率  {percent('change_recall')}",
        f"变化图斑准确率  {percent('change_precision')}",
        f"变化道路提取完整度  {percent('road_centerline_completeness')}",
        f"中心线平均偏移距离  {distance}",
        f"变化类型正确率  {percent('change_type_accuracy')}",
    ))
