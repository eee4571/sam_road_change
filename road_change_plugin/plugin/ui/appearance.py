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
    body_size = max(9, font.pointSizeF() if font.pointSizeF() > 0 else 10)
    title_size = body_size + 3
    caption_size = max(8, body_size - 1)
    arrow = (Path(__file__).resolve().parents[2] / "resources/chevron_down.svg").as_posix()
    return f"""
        QWidget#roadChangeDock, QWidget#roadChangeBody {{ background: {mix(.035)}; color: {text.name()}; }}
        #roadChangeDock QGroupBox {{ border: none; border-top: 1px solid {mix(.14)}; border-radius: 0; margin-top: 10px; font-size: {body_size + 1}pt; font-weight: 500; }}
        #roadChangeDock QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 0; padding: 0 8px 0 0; background: {mix(.035)}; }}
        #roadChangeDock QGroupBox[role="processingGroup"] {{ background: {base.name()}; border: 1px solid {border}; border-radius: 8px; margin-top: 10px; }}
        #roadChangeDock QGroupBox[role="processingGroup"]::title {{ left: 12px; padding: 0 5px; background: {base.name()}; }}
        #roadChangeDock QLabel {{ background: transparent; color: {text.name()}; font-size: {body_size}pt; font-weight: 400; }}
        #roadChangeDock QLabel[role="title"] {{ font-size: {title_size}pt; font-weight: 600; }}
        #roadChangeDock QLabel[role="sectionHeading"] {{ font-size: {body_size + 1}pt; font-weight: 500; }}
        #roadChangeDock QLabel[role="secondary"], #roadChangeDock QLabel[role="metricUnit"], #roadChangeDock QLabel[role="logText"] {{ color: {secondary}; font-size: {caption_size}pt; }}
        #roadChangeDock QLabel[role="metricValue"] {{ font-size: {body_size + 1}pt; font-weight: 500; }}
        #roadChangeDock QLabel[role="badge"] {{ border-radius: 6px; background: {hover}; padding: 2px 6px; }}
        #roadChangeDock QLabel[status="success"] {{ color: {'#75d4a3' if base.lightness() < 128 else '#19764d'}; }}
        #roadChangeDock QLabel[status="running"] {{ color: {accent}; }}
        #roadChangeDock QLabel[status="error"] {{ color: {'#ff9090' if base.lightness() < 128 else '#b32626'}; }}
        #roadChangeDock QLabel[status="warning"] {{ color: {'#f0c36e' if base.lightness() < 128 else '#976000'}; }}
        #roadChangeDock QSplitter::handle:vertical {{ background: {border}; height: 6px; }}
        #roadChangeDock QLineEdit, #roadChangeDock QComboBox {{
            background: {base.name()}; color: {text.name()}; border: 1px solid {border};
            border-radius: 2px; min-height: 28px; padding: 0 8px; placeholder-text-color: {mix(.38)};
        }}
        #roadChangeDock QLineEdit:hover, #roadChangeDock QComboBox:hover {{ border-color: {mix(.30)}; }}
        #roadChangeDock QLineEdit:disabled, #roadChangeDock QComboBox:disabled {{ background: {mix(.035)}; color: {mix(.44)}; border-color: {mix(.12)}; }}
        #roadChangeDock QLineEdit:focus, #roadChangeDock QComboBox:focus {{ border-color: {accent}; }}
        #roadChangeDock QComboBox[inputProblem="true"] {{ border-color: {mix(.55)}; }}
        #roadChangeDock QComboBox::drop-down {{ border: none; width: 24px; }}
        #roadChangeDock QComboBox::down-arrow {{ image: url("{arrow}"); width: 12px; height: 8px; }}
        #roadChangeDock QPushButton, #roadChangeDock QToolButton {{
            background: {mix(.055)}; color: {text.name()}; border: 1px solid {border};
            border-radius: 2px; min-height: 28px; padding: 0 12px;
        }}
        #roadChangeDock QPushButton:hover, #roadChangeDock QToolButton:hover {{ background: {hover}; }}
        #roadChangeDock QPushButton:pressed, #roadChangeDock QToolButton:pressed {{ background: {mix(.13)}; border-color: {mix(.30)}; }}
        #roadChangeDock QPushButton:focus, #roadChangeDock QToolButton:focus {{ border-color: {accent}; }}
        #roadChangeDock QToolButton {{ background: transparent; padding: 0 6px; border-color: {mix(.12)}; }}
        #roadChangeDock QToolButton[role="toolbarAction"] {{ border-color: transparent; background: transparent; padding: 0 6px; }}
        #roadChangeDock QToolButton[role="toolbarAction"]:hover {{ background: {hover}; border-color: {mix(.12)}; }}
        #roadChangeDock QToolButton[role="toolbarAction"]:pressed {{ background: {mix(.13)}; }}
        #roadChangeDock QToolButton[role="toolbarAction"]:focus {{ border-color: transparent; }}
        #roadChangeDock QToolButton[role="toolbarAction"]::menu-indicator {{ image: none; width: 0; }}
        #roadChangeDock QToolButton::menu-indicator {{ width: 8px; height: 5px; subcontrol-position: right center; }}
        #roadChangeDock QPushButton[role="resultAction"] {{ min-height: 26px; padding: 0 8px; }}
        #roadChangeDock QPushButton[role="logAction"] {{ min-height: 22px; font-size: {caption_size}pt; border-color: transparent; background: transparent; padding: 0 6px; }}
        #roadChangeDock QPushButton[role="logAction"]:hover {{ background: {hover}; border-color: {border}; }}
        #roadChangeDock QPushButton[role="primary"] {{
            background: {accent}; color: {on_accent}; border-color: {accent};
            font-weight: 600; min-height: 28px; padding: 0 14px;
        }}
        #roadChangeDock QPushButton[role="primary"]:hover {{ background: {palette.color(QPalette.ColorRole.Highlight).lighter(108).name()}; }}
        #roadChangeDock QPushButton[role="primary"]:pressed {{ background: {palette.color(QPalette.ColorRole.Highlight).darker(112).name()}; }}
        #roadChangeDock QPushButton:disabled, #roadChangeDock QToolButton:disabled {{ background: {mix(.035)}; border-color: {mix(.12)}; color: {mix(.44)}; }}
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
