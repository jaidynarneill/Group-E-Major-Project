import sys
import ctypes

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def set_dark_palette(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1e1e1e"))
    palette.setColor(QPalette.WindowText, QColor("#e6e6e6"))
    palette.setColor(QPalette.Base, QColor("#252526"))
    palette.setColor(QPalette.AlternateBase, QColor("#2d2d2d"))
    palette.setColor(QPalette.ToolTipBase, QColor("#2d2d2d"))
    palette.setColor(QPalette.ToolTipText, QColor("#e6e6e6"))
    palette.setColor(QPalette.Text, QColor("#e6e6e6"))
    palette.setColor(QPalette.Button, QColor("#2d2d2d"))
    palette.setColor(QPalette.ButtonText, QColor("#e6e6e6"))
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor("#094771"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QMainWindow, QDialog, QWidget {
            background-color: #1e1e1e;
            color: #e6e6e6;
        }
        QMenu, QComboBox QAbstractItemView {
            background-color: #252526;
            color: #e6e6e6;
            selection-background-color: #094771;
        }
        QToolTip {
            background-color: #2d2d2d;
            color: #e6e6e6;
            border: 1px solid #555555;
        }
    """)


def set_windows_dark_titlebar(window):
    if sys.platform != "win32":
        return

    try:
        dark_mode = ctypes.c_int(1)
        hwnd = int(window.winId())
        dwmapi = ctypes.WinDLL("dwmapi")
        for attribute in (20, 19):
            result = dwmapi.DwmSetWindowAttribute(
                hwnd,
                attribute,
                ctypes.byref(dark_mode),
                ctypes.sizeof(dark_mode),
            )
            if result == 0:
                break
    except (AttributeError, OSError):
        pass


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    set_dark_palette(app)
    window = MainWindow()
    window.show()
    set_windows_dark_titlebar(window)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()