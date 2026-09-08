"""Presentation of the formal aggregate report; never computes new metrics."""
import json
import math


def metrics_text(path):
    empty = "P — · R — · F1 —"
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return empty
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = report.get("metrics", [])
        overall = next((row for row in rows if isinstance(row, dict) and row.get("class") == "all"), {})

        def value(key):
            metric = overall.get(key)
            if isinstance(metric, (int, float)) and not isinstance(metric, bool) and math.isfinite(metric) and 0 <= metric <= 1:
                return f"{metric:.0%}"
            return "—"

        return f"P {value('precision')} · R {value('recall')} · F1 {value('f1')}"
    except (OSError, ValueError, TypeError, AttributeError):
        return empty
