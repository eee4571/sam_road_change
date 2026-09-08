"""Launch with a host Python containing PySide6; inference uses runtime/env."""
import sys
from PySide6.QtWidgets import QApplication
from plugin import create_plugin


def main():
    app = QApplication(sys.argv)
    plugin = create_plugin()
    widget = plugin.create_widget()
    widget.setWindowTitle(plugin.name)
    widget.resize(460, 760)
    app.aboutToQuit.connect(plugin.shutdown)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
