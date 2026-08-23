from __future__ import annotations

"""Shared visual constants for the Tkinter presentation layer."""

from tkinter import Tk

from app.project_manager import PREVIEW_LABELS

WORKFLOW_STEPS = (
    "数据准备",
    "自动处理",
    "人工编辑（可选）",
    "成果与评价",
)

UI = {
    # Warm ivory + forest green, matching the restrained GIS/industrial
    # application language used by the reference interface.
    "ink": "#18332A",
    "muted": "#68746E",
    "subtle": "#929A94",
    "page": "#F3F0E8",
    "card": "#FCFBF7",
    "line": "#D5D0C5",
    "line_strong": "#AAA79F",
    "blue": "#0B755C",
    "blue_hover": "#075C48",
    "blue_soft": "#E7F1EC",
    "green": "#0B755C",
    "green_soft": "#E7F1EC",
    "amber": "#A66A1F",
    "amber_soft": "#F5EBDD",
    "slate_soft": "#F5F2EA",
    "header": "#1D4034",
    "header_deep": "#17362C",
    "header_text": "#F8F5EC",
    "header_muted": "#AFC3B9",
    "viewer": "#162A23",
}

CONTROL_METRICS = {
    "primary": {"font": ("Microsoft YaHei UI", 10, "bold"), "padding": (16, 9)},
    "regular": {"font": ("Microsoft YaHei UI", 10), "padding": (12, 7)},
    "compact": {"font": ("Microsoft YaHei UI", 9), "padding": (9, 6)},
}

LAYOUT_METRICS = {
    "page_padding": (20, 16, 20, 18),
    "card_padding": (18, 15),
    "section_gap": 14,
    "module_gap": 12,
    "form_gap": 5,
    "form_label_width": 16,
    "content_wrap": 1040,
}


def configure_window_geometry(
    root: Tk, *, base_width: int, base_height: int,
    min_width: int, min_height: int,
) -> float:
    """Size and center a window while preserving usable logical proportions."""
    scale = max(1.0, min(float(root.winfo_fpixels("1i")) / 96.0, 2.5))
    screen_width = max(1, root.winfo_screenwidth())
    screen_height = max(1, root.winfo_screenheight())
    width = min(round(screen_width * 0.94), round(base_width * scale))
    height = min(round(screen_height * 0.90), round(base_height * scale))
    minimum_width = min(round(screen_width * 0.88), round(min_width * scale))
    minimum_height = min(round(screen_height * 0.82), round(min_height * scale))
    root.minsize(max(860, minimum_width), max(580, minimum_height))
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    return scale


def format_percentage(value: object, digits: int = 1) -> str:
    """Format a stored 0-1 metric for display without changing its value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number * 100:.{digits}f}%"
