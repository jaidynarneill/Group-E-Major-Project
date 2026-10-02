import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()