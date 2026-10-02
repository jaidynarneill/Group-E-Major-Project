from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QGridLayout,
    QGroupBox,
)

class SelectionPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Material selection
        material_group = QGroupBox("Select Material")
        self.material_combo = QComboBox()
        self.material_combo.addItems(["Silicon", "Copper", "Aluminum"])
        material_layout = QHBoxLayout()
        material_layout.addWidget(QLabel("Material:"))
        material_layout.addWidget(self.material_combo)
        material_group.setLayout(material_layout)

        # Lattice structure selection
        lattice_group = QGroupBox("Select Lattice Structure")
        self.lattice_combo = QComboBox()
        self.lattice_combo.addItems(["FCC", "BCC", "HCP", "Diamond"])
        lattice_layout = QHBoxLayout()
        lattice_layout.addWidget(QLabel("Lattice Structure:"))
        lattice_layout.addWidget(self.lattice_combo)
        lattice_group.setLayout(lattice_layout)

        # Method selection
        method_group = QGroupBox("Select Method")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["MEAM", "MACE-MP", "DFT"])
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        method_layout.addWidget(self.method_combo)
        method_group.setLayout(method_layout)

        # Comparison rows
        self.compare_button = QPushButton("Compare")
        self.compare_button.clicked.connect(self.compare_results)

        # Layout arrangement
        layout.addWidget(material_group)
        layout.addWidget(lattice_group)
        layout.addWidget(method_group)
        layout.addWidget(self.compare_button)

        self.setLayout(layout)

    def compare_results(self):
        # Logic to compare results based on selected parameters
        pass