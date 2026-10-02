from PyQt5.QtWidgets import QMainWindow

try:
    from ui.plot_panel import PlotPanel
except ModuleNotFoundError as error:
    if error.name != "ui":
        raise
    from src.ui.plot_panel import PlotPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Materials Property GUI")
        self.setGeometry(100, 100, 1200, 850)
        self.plot_panel = PlotPanel()
        self.setCentralWidget(self.plot_panel)