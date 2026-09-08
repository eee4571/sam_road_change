"""Optional dock-local appearance, derived from the current host palette."""
from PySide6.QtGui import QColor, QPalette
from pathlib import Path


def dock_style(palette, font):
    base = palette.color(QPalette.ColorRole.Base)
    text = palette.color(QPalette.ColorRole.Text)

    def mix(amount):
        return QColor.fromRgbF(*(base.getRgbF()[i] * (1 - amount) + text.getRgbF()[i] * amount for i in range(3))).name()

    secondary, border, hover = mix(.50), mix(.18), mix(.07)
    accent = palette.color(QPalette.ColorRole.Highlight).name()
    on_accent = palette.color(QPalette.ColorRole.HighlightedText).name()
    title_size = max(10, font.pointSizeF() if font.pointSizeF() > 0 else 10) + 3
    arrow = (Path(__file__).resolve().parents[2] / "resources/chevron_down.svg").as_posix()
    return f"""
        QWidget#roadChangeDock, QWidget#roadChangeBody {{ background: {mix(.035)}; color: {text.name()}; }}
        #roadChangeDock QGroupBox {{ border: 1px solid {mix(.14)}; border-radius: 0; margin-top: 8px; font-weight: 600; }}
        #roadChangeDock QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 4px; }}
        #roadChangeDock QLabel {{ background: transparent; color: {text.name()}; }}
        #roadChangeDock QLabel[role="title"] {{ font-size: {title_size}pt; font-weight: 600; }}
        #roadChangeDock QLabel[role="sectionHeading"] {{ font-weight: 600; }}
        #roadChangeDock QLabel[role="secondary"] {{ color: {secondary}; }}
        #roadChangeDock QLineEdit, #roadChangeDock QComboBox {{
            background: {base.name()}; color: {text.name()}; border: 1px solid {border};
            border-radius: 1px; min-height: 28px; padding: 0 8px; placeholder-text-color: {mix(.38)};
        }}
        #roadChangeDock QLineEdit:hover, #roadChangeDock QComboBox:hover {{ border-color: {mix(.30)}; }}
        #roadChangeDock QLineEdit:disabled, #roadChangeDock QComboBox:disabled {{ background: {mix(.035)}; color: {mix(.36)}; border-color: {mix(.12)}; }}
        #roadChangeDock QLineEdit:focus, #roadChangeDock QComboBox:focus {{ border-color: {accent}; }}
        #roadChangeDock QComboBox::drop-down {{ border: none; width: 24px; }}
        #roadChangeDock QComboBox::down-arrow {{ image: url("{arrow}"); width: 12px; height: 8px; }}
        #roadChangeDock QPushButton, #roadChangeDock QToolButton {{
            background: {mix(.055)}; color: {text.name()}; border: 1px solid {border};
            border-radius: 1px; min-height: 28px; padding: 0 12px;
        }}
        #roadChangeDock QPushButton:hover, #roadChangeDock QToolButton:hover {{ background: {hover}; }}
        #roadChangeDock QPushButton:pressed, #roadChangeDock QToolButton:pressed {{ background: {mix(.13)}; border-color: {mix(.30)}; }}
        #roadChangeDock QPushButton:focus, #roadChangeDock QToolButton:focus {{ border-color: {accent}; }}
        #roadChangeDock QToolButton {{ background: transparent; padding: 0 6px; border-color: {mix(.12)}; }}
        #roadChangeDock QToolButton::menu-indicator {{ width: 8px; height: 5px; subcontrol-position: right center; }}
        #roadChangeDock QPushButton[role="resultAction"] {{ min-height: 26px; padding: 0 8px; }}
        #roadChangeDock QPushButton[role="primary"] {{
            background: {accent}; color: {on_accent}; border-color: {accent};
            font-weight: 600; min-height: 28px; padding: 0 14px;
        }}
        #roadChangeDock QPushButton[role="primary"]:hover {{ background: {palette.color(QPalette.ColorRole.Highlight).lighter(108).name()}; }}
        #roadChangeDock QPushButton[role="primary"]:pressed {{ background: {palette.color(QPalette.ColorRole.Highlight).darker(112).name()}; }}
        #roadChangeDock QPushButton:disabled, #roadChangeDock QToolButton:disabled {{ background: {mix(.035)}; border-color: {mix(.12)}; color: {mix(.36)}; }}
        #roadChangeDock QPushButton[role="primary"]:disabled {{ background: {hover}; border-color: {border}; color: {secondary}; }}
        #roadChangeDock QWidget[role="resultRow"] {{ border-bottom: 1px solid {mix(.10)}; background: transparent; }}
        #roadChangeDock QWidget[role="resultRow"]:hover {{ background: {hover}; }}
        #roadChangeDock QProgressBar {{ border: 1px solid {border}; border-radius: 0; background: {base.name()}; min-height: 8px; max-height: 8px; }}
        #roadChangeDock QProgressBar::chunk {{ background: {accent}; border-radius: 0; }}
        #roadChangeDock QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
        #roadChangeDock QScrollBar::handle:vertical {{ background: {border}; min-height: 24px; border-radius: 4px; }}
        #roadChangeDock QScrollBar::add-line:vertical, #roadChangeDock QScrollBar::sub-line:vertical {{ height: 0; }}
        #roadChangeDock QScrollBar::add-page:vertical, #roadChangeDock QScrollBar::sub-page:vertical {{ background: transparent; }}
    """
