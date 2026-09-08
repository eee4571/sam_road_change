"""Optional dock-local appearance, derived from the current host palette."""
from PySide6.QtGui import QColor, QPalette
from pathlib import Path


def dock_style(palette, font):
    base = palette.color(QPalette.ColorRole.Base)
    text = palette.color(QPalette.ColorRole.Text)

    def mix(amount):
        return QColor.fromRgbF(*(base.getRgbF()[i] * (1 - amount) + text.getRgbF()[i] * amount for i in range(3))).name()

    secondary, border, hover = mix(.55), mix(.12), mix(.035)
    accent = palette.color(QPalette.ColorRole.Highlight).name()
    on_accent = palette.color(QPalette.ColorRole.HighlightedText).name()
    title_size = max(10, font.pointSizeF() if font.pointSizeF() > 0 else 10) + 3
    arrow = (Path(__file__).resolve().parents[2] / "resources/chevron_down.svg").as_posix()
    return f"""
        QWidget#roadChangeDock, QWidget#roadChangeBody {{ background: {base.name()}; color: {text.name()}; }}
        #roadChangeDock QLabel {{ background: transparent; color: {text.name()}; }}
        #roadChangeDock QLabel[role="title"] {{ font-size: {title_size}pt; font-weight: 600; }}
        #roadChangeDock QLabel[role="sectionHeading"] {{ font-weight: 600; }}
        #roadChangeDock QLabel[role="secondary"] {{ color: {secondary}; }}
        #roadChangeDock QLineEdit, #roadChangeDock QComboBox {{
            background: {base.name()}; color: {text.name()}; border: 1px solid {border};
            border-radius: 4px; min-height: 30px; padding: 0 8px;
        }}
        #roadChangeDock QLineEdit:focus, #roadChangeDock QComboBox:focus {{ border-color: {accent}; }}
        #roadChangeDock QComboBox::drop-down {{ border: none; width: 24px; }}
        #roadChangeDock QComboBox::down-arrow {{ image: url("{arrow}"); width: 12px; height: 8px; }}
        #roadChangeDock QPushButton, #roadChangeDock QToolButton {{
            background: transparent; color: {text.name()}; border: 1px solid {border};
            border-radius: 4px; min-height: 30px; padding: 0 10px;
        }}
        #roadChangeDock QPushButton:hover, #roadChangeDock QToolButton:hover {{ background: {hover}; }}
        #roadChangeDock QPushButton[role="textAction"] {{ border: none; color: {accent}; padding: 0 4px; }}
        #roadChangeDock QPushButton[role="primary"] {{
            background: {accent}; color: {on_accent}; border-color: {accent};
            font-weight: 600; min-height: 32px; padding: 0 18px;
        }}
        #roadChangeDock QPushButton:disabled, #roadChangeDock QToolButton:disabled {{ color: {secondary}; }}
        #roadChangeDock QPushButton[role="primary"]:disabled {{ background: {hover}; border-color: {border}; color: {secondary}; }}
        #roadChangeDock QWidget[role="resultRow"] {{ border-bottom: 1px solid {border}; background: transparent; }}
        #roadChangeDock QWidget[role="resultRow"]:hover {{ background: {hover}; }}
        #roadChangeDock QProgressBar {{ border: none; border-radius: 3px; background: {hover}; min-height: 6px; max-height: 6px; }}
        #roadChangeDock QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}
        #roadChangeDock QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
        #roadChangeDock QScrollBar::handle:vertical {{ background: {border}; min-height: 24px; border-radius: 4px; }}
        #roadChangeDock QScrollBar::add-line:vertical, #roadChangeDock QScrollBar::sub-line:vertical {{ height: 0; }}
        #roadChangeDock QScrollBar::add-page:vertical, #roadChangeDock QScrollBar::sub-page:vertical {{ background: transparent; }}
    """
