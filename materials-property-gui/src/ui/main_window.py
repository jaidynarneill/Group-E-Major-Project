from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout, QGridLayout
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Materials Property GUI")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.selection_panel = self.create_selection_panel()
        self.plot_panel = self.create_plot_panel()

        self.layout.addWidget(self.selection_panel)
        self.layout.addWidget(self.plot_panel)

    def create_selection_panel(self):
        panel = QWidget()
        layout = QGridLayout()

        # Material selection
        layout.addWidget(QLabel("Select Material:"), 0, 0)
        self.material_dropdown = QComboBox()
        self.material_dropdown.addItems(["Silicon", "Copper", "Aluminum"])
        layout.addWidget(self.material_dropdown, 0, 1)

        # Lattice structure selection
        layout.addWidget(QLabel("Select Lattice Structure:"), 1, 0)
        self.lattice_dropdown = QComboBox()
        self.lattice_dropdown.addItems(["FCC", "BCC", "HCP", "Other"])
        layout.addWidget(self.lattice_dropdown, 1, 1)

        # Method selection
        layout.addWidget(QLabel("Select Method:"), 2, 0)
        self.method_dropdown = QComboBox()
        self.method_dropdown.addItems(["MEAM", "MACE-MP", "DFT"])
        layout.addWidget(self.method_dropdown, 2, 1)

        # Compare button
        self.compare_button = QPushButton("Compare")
        layout.addWidget(self.compare_button, 3, 0, 1, 2, alignment=Qt.AlignCenter)

        panel.setLayout(layout)
        return panel

    def create_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        self.output_label = QLabel("Output Plots will be displayed here.")
        layout.addWidget(self.output_label)

        panel.setLayout(layout)
        return panel