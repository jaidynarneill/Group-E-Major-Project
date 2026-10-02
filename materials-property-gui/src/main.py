import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


DARK_THEME = """
QWidget {
    background-color: #1e1e1e;
    color: #e6e6e6;
    font-size: 14px;
}
QLabel#title {
    font-size: 23px;
    font-weight: bold;
    padding: 6px 0;
}
QLabel#columnHeader {
    color: #9cdcfe;
    font-weight: bold;
}
QGroupBox {
    border: 1px solid #454545;
    border-radius: 6px;
    margin-top: 12px;
    padding: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QComboBox, QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 7px;
}
QComboBox:hover, QPushButton:hover {
    border-color: #007acc;
}
QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    selection-background-color: #094771;
}
QPushButton {
    background-color: #0e639c;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1177bb;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_THEME)

    window = MainWindow()
    window.showFullScreen()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()