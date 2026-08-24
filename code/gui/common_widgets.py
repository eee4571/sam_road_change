from __future__ import annotations

"""Shared visual constants and DPI-aware Tk helpers."""

from tkinter import TclError, Tk, Toplevel, font as tkfont
from tkinter import ttk

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
    "primary": {"font": ("Microsoft YaHei UI", 9, "bold"), "padding": (11, 6)},
    "regular": {"font": ("Microsoft YaHei UI", 9), "padding": (10, 6)},
    "compact": {"font": ("Microsoft YaHei UI", 9), "padding": (8, 4)},
}

LAYOUT_METRICS = {
    "page_padding": (12, 10, 12, 10),
    "card_padding": (10, 8),
    "section_gap": 9,
    "module_gap": 7,
    "form_gap": 5,
    "form_label_width": 13,
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


def dpi_scale(root: Tk) -> float:
    """Return monitor pixel scale without assuming that Tk scaling is default."""
    try:
        return max(1.0, min(float(root.winfo_fpixels("1i")) / 96.0, 2.5))
    except (TclError, TypeError, ValueError):
        return 1.0


def font_linespace(root: Tk, font_spec=None) -> int:
    """Measure the actual font selected by Tk instead of guessing its height."""
    try:
        font = tkfont.Font(root=root, font=font_spec or "TkDefaultFont")
        return max(1, int(font.metrics("linespace")))
    except (TclError, TypeError, ValueError):
        return max(16, round(16 * dpi_scale(root)))


def treeview_metrics(root: Tk, font_spec=("Microsoft YaHei UI", 9)) -> dict[str, object]:
    """Compute row/heading metrics that keep CJK glyphs clear at high DPI."""
    scale = dpi_scale(root)
    line = font_linespace(root, font_spec)
    vertical_padding = max(6, round(6 * scale))
    return {
        "font": font_spec,
        "rowheight": line + vertical_padding,
        "heading_padding": (max(6, round(6 * scale)), max(4, round(4 * scale))),
    }


def bind_dynamic_wrap(
    label: ttk.Label, container=None, *, minimum: int = 160, padding: int = 0,
) -> ttk.Label:
    """Keep a label's wraplength synchronized with its containing widget."""
    owner = container or label.master

    def resize(event=None) -> None:
        width = int(getattr(event, "width", 0) or owner.winfo_width()) - int(padding)
        if width > 1:
            label.configure(wraplength=max(int(minimum), width))

    owner.bind("<Configure>", resize, add="+")
    label.after_idle(resize)
    return label


class Tooltip:
    """Small delayed tooltip used for full paths and clipped tree cells."""

    def __init__(self, widget, text=None, *, delay: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _value(self, event=None) -> str:
        value = self.text(event) if callable(self.text) else self.text
        if value is None and hasattr(self.widget, "get"):
            value = self.widget.get()
        return str(value or "").strip()

    def _schedule(self, event=None) -> None:
        self._cancel()
        self._after = self.widget.after(self.delay, lambda: self._show(event))

    def _cancel(self) -> None:
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except TclError:
                pass
            self._after = None

    def _show(self, event=None) -> None:
        value = self._value(event)
        if not value or value.startswith("尚未"):
            return
        self._hide()
        window = Toplevel(self.widget)
        window.wm_overrideredirect(True)
        x = self.widget.winfo_pointerx() + 14
        y = self.widget.winfo_pointery() + 18
        window.wm_geometry(f"+{x}+{y}")
        ttk.Label(
            window, text=value, justify="left", padding=(7, 4),
            relief="solid", borderwidth=1, wraplength=900,
        ).pack()
        self._window = window

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._window is not None:
            try:
                self._window.destroy()
            except TclError:
                pass
            self._window = None


class PathDisplay(ttk.Entry):
    """Single-line, copyable path field with keyboard scrolling and tooltip."""

    def __init__(self, master, *, textvariable=None, **kwargs) -> None:
        kwargs.setdefault("state", "readonly")
        kwargs.setdefault("cursor", "xterm")
        super().__init__(master, textvariable=textvariable, **kwargs)
        self.tooltip = Tooltip(
            self,
            lambda _event=None: textvariable.get() if textvariable is not None else self.get(),
        )
        self.bind("<Control-a>", self._select_all, add="+")
        self.bind("<Control-A>", self._select_all, add="+")

    def _select_all(self, _event=None):
        self.selection_range(0, "end")
        self.icursor("end")
        return "break"


def attach_treeview_tooltip(tree: ttk.Treeview, path_columns: tuple[str, ...] = ("path",)) -> Tooltip:
    """Show the complete path for the tree cell currently under the pointer."""
    def cell_text(_event=None) -> str:
        x, y = tree.winfo_pointerx() - tree.winfo_rootx(), tree.winfo_pointery() - tree.winfo_rooty()
        row, column = tree.identify_row(y), tree.identify_column(x)
        if not row or not column:
            return ""
        if column == "#0":
            return str(tree.item(row, "text") or "")
        try:
            index = int(column[1:]) - 1
            column_name = str(tree["columns"][index])
            if path_columns and column_name not in path_columns:
                return ""
            values = tree.item(row, "values")
            return str(values[index]) if index < len(values) else ""
        except (IndexError, TypeError, ValueError, TclError):
            return ""

    return Tooltip(tree, cell_text)
