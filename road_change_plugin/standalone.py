"""Launch with gui_env (or a developer host); inference uses runtime/env."""
import sys
from PySide6.QtCore import QRect, QSize
from PySide6.QtWidgets import QApplication
from plugin import create_plugin


def initial_window_geometry(available):
    """Logical pixels: leave room for the native frame and desktop taskbar."""
    size = QSize(min(820, max(1, available.width() - 32)),
                 min(600, max(1, available.height() - 64)))
    rect = QRect(available.topLeft(), size)
    rect.moveCenter(available.center())
    return rect


def main():
    app = QApplication(sys.argv)
    plugin = create_plugin()
    widget = plugin.create_widget()
    widget.setWindowTitle(plugin.name)
    widget.setGeometry(initial_window_geometry(widget.screen().availableGeometry()))
    app.aboutToQuit.connect(plugin.shutdown)
    if "--check" in sys.argv:
        app.processEvents()
        plugin.shutdown()
        widget.close()
        print("UI ready: " + sys.executable)
        return 0
    widget.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
